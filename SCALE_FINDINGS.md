# E8M0 Scale Numerical Discrepancy — Investigation Findings

## Executive Summary

The E8M0 scale **byte interpretation is correct**: `scale = 2^(byte - 127)` per the OCP MX specification.
This is confirmed by multiple independent code paths in the gfx950-tutorial Triton fork.

The ~101 max error observed between the fused grouped GEMM kernel and the dequantized BF16 reference
is **NOT caused by a different E8M0 interpretation**. The root cause lies elsewhere — most likely in
the data distribution/packing of scale values across MFMA threads.

---

## 1. E8M0 Interpretation: CONFIRMED OCP Spec

### Evidence A: `AccelerateAMDMatmul.cpp` — "0x7F is 1.0 in E8M0"

File: `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`

```cpp
// Lines 1209, 1355
// 0x7F is 1.0 in E8M0
return rewriter.create<arith::ConstantOp>(
    dotOp->getLoc(), newScaleType,
    DenseElementsAttr::get(newScaleType, llvm::APInt(8, 0x7F)));
```

0x7F = 127. `2^(127 - 127) = 2^0 = 1.0`. This directly confirms the OCP interpretation.

### Evidence B: `mxfpScaleFp16` scale conversion

File: `third_party/amd/lib/TritonAMDGPUToLLVM/UpcastMXFPToLLVM.cpp`

```cpp
Value scaleF32 =
    b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)), f32_ty);
```

This places the E8M0 byte into bits [30:23] of a float32 (the exponent field with mantissa=0),
which represents exactly `2^(byte - 127)` as a normalized float32 value.

### Evidence C: `deduceScaleFactor` for block size

File: `lib/Dialect/Triton/IR/Ops.cpp`

```cpp
int32_t scaleFactor = kdim / (*scaleShape)[scaleShape->size() - 1];
if (scaleFactor != 16 && scaleFactor != 32) {
    ...
    return 0;
}
```

The scale factor (block size per scale) is always 32 for MXFP8/E4M3 — matching the standard.


## 2. GFX950 MFMA Lowering Path

### How `tl.dot_scaled(a8, a_sc, "e4m3", b8, ws, "e4m3")` lowers:

1. **Python frontend** (`semantic.py`):
   - `dot_scaled()` → `builder.create_dot_scaled(...)` 
   - `deduce_scale_factor()` → computes scale factor (32 for E4M3)

2. **IR** (`TritonOps.td`):
   - `TT_DotScaledOp` with `a`, `a_scale`, `b`, `b_scale`, element types
   - Scale operands are `Optional<RankedTensorOf<[TT_Float, I8]>>`
   - For E8M0, scales come in as uint8 tensors of shape [M, K//32] and [N, K//32]

3. **AMD transform** (`AccelerateAMDMatmul.cpp`):
   - Converts layouts for scales (LinearLayout encoding)
   - Creates defaults: `0x7F = 1.0 in E8M0` for missing scales

4. **MFMA intrinsic selection** (`MfmaGroup.cpp`):
   - `withScale=true` → maps E4M3 type to **FP4 (E2M1)** for lookup:
   ```cpp
   if (withScale) {
       assert(version == 4 && isF8F6F4(aET) && isF8F6F4(bET));
       // For MXFP types, we have the same intrinsic, which uses FP4 as the key
       aET = bET = b.getType<Float4E2M1FNType>();
   }
   ```
   - Selected intrinsics (gfx950 CDNA4 only):
     - `mfma_scale_f32_16x16x128_f8f6f4` (kBase=32, kDim=128)
     - `mfma_scale_f32_32x32x64_f8f6f4` (kBase=32, kDim=64)

5. **LLVM lowering** (`MFMA.cpp:convertScaledDot`):
   - kBase = 32 (from intrinsic)
   - scaleKWidth = 1 (each thread holds 1 scale per K position)
   - Scales packed 4-per-32-bit register, selected by `opSel`
   - Generates `V_MFMA_SCALE_*_F8F6F4` assembly with packed scale operands

### Scale Packing Logic:

```cpp
// MFMA.cpp line ~674-676
const int scaleAKBase = isAScaleConstant ? 1 : std::min(4, (int)(numRepK * numRepM));
const int scaleBKBase = isAScaleConstant ? 1 : std::min(4, (int)(numRepK * numRepN));

int akPackedVals = isAScaleConstant ? 1 : std::min(4, (int)numRepK);
int bkPackedVals = isAScaleConstant ? 1 : std::min(4, (int)numRepK);

// Line ~781
int mScale = m / aNonKPackedVals;
int nScale = n / bNonKPackedVals;
opSelA = (m * numRepK + k) % (aNonKPackedVals * akPackedVals);
opSelB = (n * numRepK + k) % (bNonKPackedVals * bkPackedVals);
```


## 3. Root Cause Analysis: Where the ~101 Error Comes From

### Hypothesis 0: Scale K-Width Repetition Bug (STRONGEST CANDIDATE)

**This is a potential bug in the Triton compiler's MFMA lowering for scaled dot products.**

When `numRepK = 1` (e.g., BLOCK_K=128 with `mfma_scale_f32_16x16x128_f8f6f4` which has kDim=128),
the `getValuesFromDotOperandLayoutStruct` function hits its special case for `numVecInKBase == 0`:

```cpp
// MFMA.cpp:462-467
if (numVecInKBase == 0) {
    numVecInKBase = 1;
    nonKRep /= kBase / (kRepInKWidth * kWidth);
    assert(nonKRep > 0 && "nonKrep too small");
}
```

For scales: kRepInKWidth = numRepK = 1, kWidth = scaleKWidth = 1, kBase = scaleAKBase = 4.
- `numVecInKBase = 1 * 1 / 4 = 0`
- `nonKRep = numRepM / (4 / 1) = numRepM / 4`

If numRepM=4 (BLOCK_M=128 with 32x? mfma): nonKRep becomes 1.
If numRepM=2 (BLOCK_M=128 with 64x? mfma): nonKRep would be 2/4 = 0 (assertion failure!).

**Consequence**: With the `nonKRep` reduction, the ValueTable entry `{b, mScale, 0}` maps
muiltiple M positions to the SAME packed register. The opSel then selects the correct
byte for each m. This works ONLY IF the 4 scale values in elems[0..3] are for the 4
M positions, each with the SAME kBlock=0 scale.

**But the scale layout has shape [BLOCK_M, BLOCK_K//32]. Each M row has BLOCK_K//32**
scale entries across K. When numRepK=1, there's only 1 K-block, so each M row has 1
scale. The 4 M positions' scales get packed into one register. This is correct if the
layout puts all 4 M positions' scales into elems[0..3] in order.

**However**, if the layout distributes scales across threads differently (some M
positions handled by DIFFERENT threads), this packing would be wrong. The layout
system might assign different M-rows to different threads, meaning a single thread
might not have all 4 scales to pack.

### Hypothesis 1: Scale Layout Mismatch (LIKELY)

The kernel passes scales in **row-major layout** with shapes:
- `a_scales`: [M, K//32] — uint8, row-major
- `w_scales`: [E, N, K//32] — uint8, row-major

But `tl.dot_scaled` expects scales with shape `[..., N, K//scale_factor]` and `[..., M, K//scale_factor]`
according to the assertion in `semantic.py`:

```python
assert lhs_scale_shape[-2:] == [M, K // scale_factor]
assert rhs_scale_shape[-2:] == [N, K // scale_factor]
```

**Wait**: The assertion in `verify_scaled_shape` checks:
- lhs_scale: shape `[..., M, K // scale_factor]` (lhs is A, so M rows, K//32 cols)
- rhs_scale: shape `[..., N, K // scale_factor]` (rhs is B, so N rows, K//32 cols)

In our kernel:
- A has shape [BLOCK_M, BLOCK_K] → lhs_scale should be [BLOCK_M, BLOCK_K//32] ✓
- W has shape [BLOCK_K, HALF_N] → rhs_scale should be `[HALF_N, BLOCK_K//32]`

But wait — in the kernel code, the W scale is loaded as:
```python
ws_base = w_scale_ptr + gid * stride_wse
ws_ptrs_l = ws_base + offs_nl[:, None] * stride_wsn + offs_ks[None, :] * 1
```

Here, `offs_nl` has HALF_N elements, and `offs_ks` has KS = BLOCK_K//32 elements.
So ws_ptrs has shape [HALF_N, KS] in memory — matching the expected `[N_cols, K//32]` layout.

This looks correct.

### Hypothesis 2: Data Distribution to MFMA Threads

The `getValuesFromDotOperandLayoutStruct` function extracts values from the dot operand layout
(which may be a distributed layout across threads). For scales, it uses:
- `scaleKWidth = 1` (one scale element per thread per K-iteration)
- `scaleAKBase` (how many scale values per thread)

The values are indexed as `operandAScale[{b, mScale, bkScale}]` where:
- `bkScale = k / akPackedVals` (which K-block within packed register)
- The actual byte is selected by `opSel`

**Potential issue**: If the scale layout passes `scaleKWidth=1` but the data distribution
in the layout actually gives value-per-32-K-elements (matching the [K//32] shape), the
thread partitioning could be off.

### Hypothesis 3: Constant vs Non-Constant Scale Paths

When scales are ALL the same value (e.g., all `0x7F`), the code takes the constant path:
```cpp
const int scaleAKBase = isAScaleConstant ? 1 : std::min(4, ...);
```

This changes the packing entirely (single value broadcast vs 4-packed registers).
If the non-constant path has bugs in the packing logic, varying scales would show errors
while uniform scales would be correct.

### Hypothesis 4: Precision Difference Between MFMA and Reference Path

The reference dequantization does:
1. `fp8 → float32 × scale → bfloat16` (loses precision at bf16 conversion)
2. `bf16 @ bf16` (matmul in bf16, further precision loss)

The MFMA path does:
1. `fp8 × scale → fp32` (direct, no intermediate bf16)
2. Accumulates in fp32
3. Converts final result to bf16

**Verdict**: While this would cause SOME error, the magnitude (~101) is far too
large for precision differences alone. Expected precision error for a 256×256 matmul
is typically < 1e-3 relative error, not 10-100%.

### Hypothesis 5: `scale_factor` Derived from Scale Shape

In the kernel, a_scales has shape [M, K//32]. The scale_factor is computed from shape:
```
scale_factor = K / (K//32) = 32
```

This is correct. But the Triton backend might derive a different scale_factor if it
doesn't read the shape correctly from the Python-level tensor.

---

## 4. Recommended Next Steps

### A. Run the Minimal Correctness Test (CRITICAL — FIRST THING TO DO)
The test script `scale_correctness.py` tests:
1. Known values with known scales (isolate basic scale arithmetic)
2. E8M0 byte=128 interpretation (is 128 → scale 2.0?)
3. Scale application order (before dot vs after dot)
4. Scale layout verification (per-row varying scales)

Run: `python /workspace/shekhar/grouped-gemms/scale_correctness.py`

### B. Enable Triton/ROCm Debug Logging
```python
import os
os.environ["TRITON_INTERPRET"] = "1"  # Use interpreter to validate lowering
```

### C. Check the Generated Assembly
```python
# After kernel launch, check the compiled kernel's assembly
kernel = fused_grouped_fwd_kernel[...]
asm = kernel.asm["amdgcn"]
# Look for V_MFMA_SCALE instructions and check op_sel operands
```

### D. Compare with `scaled_upcast_fp8` Path
The `upcast_mxfp` op explicitly handles E8M0→BF16 conversion correctly.
Comparing the MFMA path against explicit upcast+regular mfma could isolate
whether the issue is in the scale lowering or the MFMA instruction itself.

### E. Inspect the Dot Operand Layout for Scales
The scale tensor undergoes layout conversion from row-major to the dot operand
encoding. The `getValuesFromDotOperandLayoutStruct` extraction must match the
layout that was assigned. A layout mismatch here could silently produce wrong values.

---

## 5. Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| `dot_scaled` Python entry | `python/triton/language/semantic.py` | L1576-1631 |
| `deduce_scale_factor` Python | `python/triton/language/semantic.py` | L1553-1560 |
| `deduceScaleFactor` C++ | `lib/Dialect/Triton/IR/Ops.cpp` | L368-454 |
| Scale factor = 32 | `lib/Dialect/Triton/IR/Ops.cpp` | L388-409 |
| `0x7F = 1.0 in E8M0` | `third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp` | L1209 |
| FP4-key for scaled intrinsic | `third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp` | L33-36 |
| MFMA scaled lowering | `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp` | L575-830 |
| Scale packing/opSel | `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp` | L651-686 |
| E8M0→FP32 conversion | `third_party/amd/lib/TritonAMDGPUToLLVM/UpcastMXFPToLLVM.cpp` | L42-50 |
| WMMA E8M0 scale format | `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/WMMA.cpp` | L186-193 |

---

## 6. Files Created

- `scale_correctness.py` — Minimal correctness test (4 test scenarios)
- `SCALE_FINDINGS.md` — This document

---

*Investigation completed 2026-06-10. Test script needs GPU execution.*
