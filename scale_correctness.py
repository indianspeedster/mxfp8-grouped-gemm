#!/usr/bin/env python3
"""Minimal correctness test for E8M0 scale interpretation in fused grouped GEMM.

   Tests with KNOWN, SIMPLE scale values to isolate the E8M0 byte interpretation.

   Hypothesis: The gfx950 V_MFMA_SCALE hardware interprets E8M0 as 2^(byte - 127)
   per the OCP spec (confirmed by "0x7F is 1.0 in E8M0" in AccelerateAMDMatmul.cpp).

   But the ~101 max error suggests potential issues with:
   1. Scale packing order (opSel selects wrong byte from packed 32-bit value)
   2. Scale layout distribution across threads
   3. How scale is applied within the MFMA dot product accumulation
"""
import torch
import triton
import triton.language as tl
import os
import sys

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fused_fwd import fused_grouped_fwd


# =============================================================================
# E8M0 Helper Functions
# =============================================================================

def e8m0_byte_to_float(byte_value):
    """OCP E8M0: scale = 2^(byte_value - 127).
       byte_value=0: reserved (NaN/Inf), but treated as 2^-127 here.
    """
    return 2.0 ** (float(byte_value) - 127.0)


def float_to_e8m0_byte(scale_value):
    """Convert a float scale to the nearest E8M0 byte.
       E8M0 has only powers of 2, so we round to nearest power-of-2.
       Returns uint8 byte value.
    """
    import math
    exp = round(math.log2(scale_value)) + 127
    exp = max(0, min(255, exp))
    return exp


# =============================================================================
# Test: Known values with known scales
# =============================================================================

def test_known_scales():
    """Test with small matrices and known, simple scale values.

    Creates 2x2 data with known values, applies known E8M0 scales,
    and compares kernel output vs exact BF16 reference.
    """
    print("=" * 70)
    print("Test 1: Small known-value test with E8M0 scales")
    print("=" * 70)

    # Use multiples of 32 for block scaling
    M, N, K, E = 64, 96, 128, 2  # E=2 groups, group_M=32 each

    # Create known values for A [M, K]
    a_vals = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    a_bf16 = a_vals.to(torch.bfloat16).cuda()

    # Create known values for W [E, N, K]
    w_vals = torch.arange(E * N * K, dtype=torch.float32).reshape(E, N, K) * 0.1
    w_bf16 = w_vals.to(torch.bfloat16).cuda()

    # Quantize to E4M3 with KNOWN E8M0 scales
    # scale=1.0  -> e8m0 byte = 127
    # scale=2.0  -> e8m0 byte = 128
    # scale=0.5  -> e8m0 byte = 126
    # scale=4.0  -> e8m0 byte = 129

    K_blocks = K // 32

    # A scales: alternating 1.0 and 2.0 across blocks
    a_scale_bytes = torch.full((M, K_blocks), 127, dtype=torch.uint8)  # all 1.0
    a_scale_bytes[:, ::2] = 128  # even blocks: scale=2.0
    a_scales = a_scale_bytes.clone().cuda()

    # W scales: all 1.0
    w_scale_bytes = torch.full((E, N, K_blocks), 127, dtype=torch.uint8)
    w_scales = w_scale_bytes.clone().cuda()

    # Quantize A and W using these scales
    a_fp8 = torch.empty(M, K, dtype=torch.float8_e4m3fn).cuda()
    for i in range(M):
        for b in range(K_blocks):
            k_start, k_end = b * 32, (b + 1) * 32
            scale = e8m0_byte_to_float(int(a_scales[i, b].item()))
            block_vals = a_bf16[i, k_start:k_end].float() / scale
            a_fp8[i, k_start:k_end] = block_vals.to(torch.float8_e4m3fn)

    w_fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn).cuda()
    for e in range(E):
        for n in range(N):
            for b in range(K_blocks):
                k_start, k_end = b * 32, (b + 1) * 32
                scale = e8m0_byte_to_float(int(w_scales[e, n, b].item()))
                block_vals = w_bf16[e, n, k_start:k_end].float() / scale
                w_fp8[e, n, k_start:k_end] = block_vals.to(torch.float8_e4m3fn)

    offs = torch.tensor([0, M // 2, M], dtype=torch.int32).cuda()

    # Compute reference: dequantize, then do BF16 matmul
    # Dequantize A (apply E8M0 scales)
    a_deq_bf16 = torch.zeros(M, K, dtype=torch.bfloat16).cuda()
    for i in range(M):
        for b in range(K_blocks):
            k_start, k_end = b * 32, (b + 1) * 32
            scale = e8m0_byte_to_float(int(a_scales[i, b].item()))
            a_deq_bf16[i, k_start:k_end] = (
                a_fp8[i, k_start:k_end].float() * scale
            ).to(torch.bfloat16)

    # Dequantize W
    w_deq_bf16 = torch.zeros(E, N, K, dtype=torch.bfloat16).cuda()
    for e in range(E):
        for n in range(N):
            for b in range(K_blocks):
                k_start, k_end = b * 32, (b + 1) * 32
                scale = e8m0_byte_to_float(int(w_scales[e, n, b].item()))
                w_deq_bf16[e, n, k_start:k_end] = (
                    w_fp8[e, n, k_start:k_end].float() * scale
                ).to(torch.bfloat16)

    # Reference: group-wise BF16 matmul
    ref_deq = torch.zeros(M, N, dtype=torch.bfloat16).cuda()
    for g in range(E):
        gs = g * (M // E)
        ge = (g + 1) * (M // E)
        ref_deq[gs:ge] = a_deq_bf16[gs:ge] @ w_deq_bf16[g].T

    # Run kernel with scales
    print("  Running fused grouped GEMM with scales...")
    c = fused_grouped_fwd(
        a_fp8, w_fp8, offs,
        a_scales=a_scales, w_scales=w_scales,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        GROUP_SIZE_M=1
    )
    torch.cuda.synchronize()

    max_err = (c.float() - ref_deq.float()).abs().max().item()
    mean_err = (c.float() - ref_deq.float()).abs().mean().item()

    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print()

    # Print element-by-element comparison for first few elements
    print("  Element-by-element comparison (first 4 rows x 4 cols):")
    print(f"  {'Row':>4s} {'Col':>4s} {'Kernel':>14s} {'Ref(Deq)':>14s} {'Diff':>12s} {'Elem':>14s}")
    for r in range(min(4, M)):
        for c in range(min(4, N)):
            kern_val = float(c[r, c])
            ref_val = float(ref_deq[r, c])
            diff = kern_val - ref_val
            # Element-wise reference: sum over K of deq values
            elem_sum = sum(
                float(a_deq_bf16[r, k]) * float(w_deq_bf16[r // (M // E), c, k])
                for k in range(K)
            )
            print(f"  {r:4d} {c:4d} {kern_val:14.4f} {ref_val:14.4f} {diff:12.4f} {elem_sum:14.4f}")

    print()
    print(f"  {'PASS' if max_err < 0.1 else '*** FAIL (large error!) ***'}")

    return max_err, mean_err


# =============================================================================
# Test 2: Identity/Powers-of-2 scales to verify byte interpretation
# =============================================================================

def test_scale_bypass():
    """Test that passes different E8M0 byte values through the kernel and
    checks whether output matches 2^(byte-127) interpretation.

    Strategy: Use a single row of A with all-ones, W with all-ones, varying scales.
    Then out[i,j] = sum_k(A[i,k] * W[j,k] * scale_A[i,k//32] * scale_W[j,k//32])
    """
    print()
    print("=" * 70)
    print("Test 2: E8M0 byte interpretation verification")
    print("=" * 70)

    M, N, K, E = 64, 32, 64, 2  # group_M = 32 each, 2 K-blocks per group

    # A: all ones in fp8
    a_bf16 = torch.ones(M, K, dtype=torch.bfloat16).cuda()
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)

    # W: all ones in fp8
    w_bf16 = torch.ones(E, N, K, dtype=torch.bfloat16).cuda()
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)

    K_blocks = K // 32

    # A scales: byte=128 (= scale 2.0) for first K-block, byte=127 (= scale 1.0) for second
    a_scales = torch.full((M, K_blocks), 127, dtype=torch.uint8).cuda()
    a_scales[:, 0] = 128  # scale 2.0
    a_scales[:, 1] = 127  # scale 1.0

    # W scales: all byte=127 (= scale 1.0)
    w_scales = torch.full((E, N, K_blocks), 127, dtype=torch.uint8).cuda()

    offs = torch.tensor([0, M // 2, M], dtype=torch.int32).cuda()

    print("  A scale bytes: block 0 = 128 (scale 2.0), block 1 = 127 (scale 1.0)")
    print("  W scale bytes: all 127 (scale 1.0)")
    print("  Data: all ones (fp8)")
    print()

    # Expected reference (manual):
    # For each output element [m, n]:
    #   sum over k: 1.0 * 1.0 * scale_A[m, k//32] * scale_W[g(m), n, k//32]
    # Group 0 (m in [0,31]): scale_W[0,n,k//32] = 1.0 always
    #   = sum over k: scale_A[m, k//32] * 1.0
    #   = 32 * 2.0 + 32 * 1.0 = 64 + 32 = 96.0
    # Group 1 (m in [32,63]): scale_W[1,n,k//32] = 1.0 always
    #   Same = 96.0
    expected_value = 96.0

    c = fused_grouped_fwd(
        a_fp8, w_fp8, offs,
        a_scales=a_scales, w_scales=w_scales,
        BLOCK_M=64, BLOCK_N=32, BLOCK_K=64,
        GROUP_SIZE_M=1
    )
    torch.cuda.synchronize()

    # Check first few elements
    actual_val = float(c[0, 0])
    print(f"  Expected[0,0]: {expected_value:.2f}")
    print(f"  Actual[0,0]:   {actual_val:.2f}")
    print(f"  Diff:          {actual_val - expected_value:.2f}")
    print()

    # If byte=128 means scale=2.0: expected = 96.0
    # If byte=128 means scale=something else: different value
    max_err = abs(actual_val - expected_value)
    print(f"  {'PASS - 2^(byte-127) interpretation correct' if max_err < 1.0 else '*** Different interpretation detected! ***'}")
    print(f"  Max error across all elements: {(c.float() - expected_value).abs().max().item():.4f}")

    return max_err


# =============================================================================
# Test 3: Check if scale is applied before dot product or to whole dot product
# =============================================================================

def test_scale_application_order():
    """Test whether scales apply per-element (before dot) or per-dot-result.

    If scale applies per-element: C[m,n] = sum(A[m,k] * B[k,n] * sA[m,k//32] * sB[k//32,n])
    If scale applies after dot:   C[m,n] = scale * sum(A[m,k] * B[k,n])

    Use varying scales along K to distinguish these cases.
    """
    print()
    print("=" * 70)
    print("Test 3: Scale application order (before dot vs after dot)")
    print("=" * 70)

    M, N, K = 64, 32, 64
    E = 1  # Single group for simplicity
    K_blocks = K // 32

    a_bf16 = torch.ones(M, K, dtype=torch.bfloat16).cuda()
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)

    w_bf16 = torch.ones(E, N, K, dtype=torch.bfloat16).cuda()
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)

    # A scales: block 0=2.0 (byte 128), block 1=0.5 (byte 126)
    a_scales = torch.full((M, K_blocks), 127, dtype=torch.uint8).cuda()
    a_scales[:, 0] = 128  # 2.0
    a_scales[:, 1] = 126  # 0.5

    # W scales: block 0=4.0 (byte 129), block 1=1.0 (byte 127)
    w_scales = torch.full((E, N, K_blocks), 127, dtype=torch.uint8).cuda()
    w_scales[:, :, 0] = 129  # 4.0
    w_scales[:, :, 1] = 127  # 1.0

    offs = torch.tensor([0, M], dtype=torch.int32).cuda()

    c = fused_grouped_fwd(
        a_fp8, w_fp8, offs,
        a_scales=a_scales, w_scales=w_scales,
        BLOCK_M=64, BLOCK_N=32, BLOCK_K=64,
        GROUP_SIZE_M=1
    )
    torch.cuda.synchronize()

    # Per-element scaling: each k contributes A[m,k]*B[n,k]*sA[m,k//32]*sB[n,k//32]
    # Block 0 (k=0..31): scale = 2.0 * 4.0 = 8.0, contributes 32
    # Block 1 (k=32..63): scale = 0.5 * 1.0 = 0.5, contributes 32
    #
    # Actually: sum over k: 1.0 * 1.0 * scale_A[block] * scale_W[block]
    # Block 0 value: 32 * 1.0 * 1.0 * 2.0 * 4.0 = 32 * 8.0 = 256.0
    # Block 1 value: 32 * 1.0 * 1.0 * 0.5 * 1.0 = 32 * 0.5 = 16.0
    # Total per-element = 256.0 + 16.0 = 272.0

    per_elem_expect = 272.0

    # Post-dot scaling: all scales multiplied together first, then applied
    # avg scale A = (2.0 + 0.5)/2 = 1.25
    # avg scale W = (4.0 + 1.0)/2 = 2.5
    # result = 64 * 1.25 * 2.5 = 200.0
    post_dot_expect = 200.0

    # Shared scale: if single scale applied
    # With E8M0 scale=1.0 (byte=127): result = 64 * 1.0 * 1.0 = 64.0
    single_scale_expect = 64.0

    actual = float(c[0, 0])
    print(f"  Expected if per-element (before dot):  {per_elem_expect:.1f}")
    print(f"  Expected if post-dot (avg scales):     {post_dot_expect:.1f}")
    print(f"  Expected if single scale=1.0:          {single_scale_expect:.1f}")
    print(f"  Actual value:                           {actual:.2f}")
    print()

    diffs = {
        'per-element': abs(actual - per_elem_expect),
        'post-dot': abs(actual - post_dot_expect),
        'single': abs(actual - single_scale_expect),
    }
    best = min(diffs, key=diffs.get)
    print(f"  Closest match: {best}")

    return actual


# =============================================================================
# Test 4: Scale layout verification - check if scale packing is correct
# =============================================================================

def test_scale_layout():
    """Test with carefully chosen scales to verify scale layout/packing.

    Use different scale values for each M-row (in A) and each N-col (in W)
    to detect any layout transposition or offset errors.
    """
    print()
    print("=" * 70)
    print("Test 4: Scale layout/packing verification")
    print("=" * 70)

    M, N, K = 64, 32, 64
    E = 1
    K_blocks = K // 32

    # A scales: each row gets a different scale
    # Row i: scale = 2^(i) → byte = 127 + i for block 0, 127 for block 1
    a_scales = torch.full((M, K_blocks), 127, dtype=torch.uint8).cuda()
    a_scales[:, 1] = 127  # block 1: scale=1.0
    # block 0: row 0=2^0=1.0, row 1=2^1=2.0, row 2=2^2=4.0, ...
    for i in range(M):
        a_scales[i, 0] = min(127 + i, 254)

    # W scales: all 1.0
    w_scales = torch.full((E, N, K_blocks), 127, dtype=torch.uint8).cuda()

    a_bf16 = torch.ones(M, K, dtype=torch.bfloat16).cuda()
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    w_bf16 = torch.ones(E, N, K, dtype=torch.bfloat16).cuda()
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)

    offs = torch.tensor([0, M], dtype=torch.int32).cuda()

    c = fused_grouped_fwd(
        a_fp8, w_fp8, offs,
        a_scales=a_scales, w_scales=w_scales,
        BLOCK_M=64, BLOCK_N=32, BLOCK_K=64,
        GROUP_SIZE_M=1
    )
    torch.cuda.synchronize()

    # Expected: C[m, n] = sum_k 1*1 * sA[m, k//32] * 1.0
    # = 32 * 2^m + 32 * 1.0 = 32 * (2^m) + 32
    print("  Checking per-row scale application:")
    n_errors = 0
    for m in range(min(8, M)):
        expected_scale_a = e8m0_byte_to_float(int(a_scales[m, 0].item()))
        expected = 32.0 * expected_scale_a + 32.0 * 1.0  # 32 elems * scale + 32 elems * 1.0
        actual = float(c[m, 0])
        diff = abs(actual - expected)
        if diff > 1.0:
            n_errors += 1
        print(f"    Row {m}: scale_A={expected_scale_a:.1f}, "
              f"expected={expected:.1f}, actual={actual:.2f}, "
              f"diff={diff:.2f} {'*** MISMATCH' if diff > 1.0 else ''}")

    print(f"  Row mismatches: {n_errors}/{min(8, M)}")
    return n_errors


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Triton: {triton.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
    print("GPU Architecture:", torch.cuda.get_device_capability())
    print()

    test_known_scales()
    test_scale_application_order()
    test_scale_layout()
    test_scale_bypass()
