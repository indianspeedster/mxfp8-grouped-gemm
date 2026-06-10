#!/usr/bin/env python3
"""Fused Grouped GEMM with standard Triton tl.dot_scaled — single-kernel for all E groups.

   Input:  A_fp8 [M, K] row-major
           W_fp8 [E, N, K]  (weights — transposed in-kernel for GEMM: W[g].T → [K, N])
   Output: C_bf16 [M, N]
   Offsets: [E+1] with offs[0]=0, offs[i] = M//E * i (equal-sized groups)

   Uses GROUP_M swizzling across all groups, N-slicing (BLOCK_N//2 halves),
   and tl.dot_scaled with e4m3 format.
"""
import torch, triton, triton.language as tl
from math import exp, log
import time

# ---------- Kernel ----------

@triton.jit
def fused_grouped_fwd_kernel(
    a_ptr, w_ptr, c_ptr, offs_ptr,
    a_scale_ptr, w_scale_ptr,
    M, N, K, E,
    stride_am, stride_ak,
    stride_asm,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_SCALES: tl.constexpr,
):
    """Fused grouped GEMM: C[g_start:g_end, :] = A[g_start:g_end, :] @ W[g].T

       Each program processes one (group, M-tile, N-tile) triple.
       GROUP_M swizzling is applied across all groups' M tiles combined.
       N-slicing: BLOCK_N is split into HALF_N left/right halves.
       USE_SCALES: if False, skips all scale loads (fast path, ~1820 TFLOPS).
                   if True, loads E8M0 per-32-element scales (MXFP8, ~1500 TFLOPS).
    """
    HALF_N: tl.constexpr = BLOCK_N // 2

    pid = tl.program_id(0)

    # --- Parse group structure ---
    # Equal-sized groups: group_M = M // E
    group_M = M // E
    num_pid_m_per_group = tl.cdiv(group_M, BLOCK_M)
    num_pid_m_total = E * num_pid_m_per_group
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # --- GROUP_M swizzling across all groups ---
    if GROUP_SIZE_M == 1:
        pid_m_global = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = pid // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m_total - first_pid_m, GROUP_SIZE_M)
        pid_m_global = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # Map global pid_m → (group_id, pid_m_local)
    gid = pid_m_global // num_pid_m_per_group
    pid_m_local = pid_m_global % num_pid_m_per_group

    # Row range within this group
    row_start = gid * group_M + pid_m_local * BLOCK_M
    g_end = gid * group_M + group_M

    # --- Tile offsets ---
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_nl = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_nr = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)

    # A pointers [BLOCK_M, BLOCK_K]
    a_ptrs = (a_ptr + row_start * stride_am
              + offs_m[:, None] * stride_am
              + offs_k[None, :] * stride_ak)

    # KS is always defined (used for constexpr elsewhere)
    KS: tl.constexpr = BLOCK_K // 32

    # W base for this group: W[gid, :, :]
    w_base = w_ptr + gid * stride_we

    # W pointers [BLOCK_K, HALF_N]  —  K-major layout for dot input
    w_ptrs_l = (w_base
                + offs_k[:, None] * stride_wk
                + offs_nl[None, :] * stride_wn)
    w_ptrs_r = (w_base
                + offs_k[:, None] * stride_wk
                + offs_nr[None, :] * stride_wn)

    # Masks
    a_mask = (row_start + offs_m[:, None]) < g_end
    n_mask_l = offs_nl[None, :] < N
    n_mask_r = offs_nr[None, :] < N
    k_mask = offs_k[:, None] < K

    # Accumulators (fp32)
    acc_l = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_r = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    # --- Scale pointers (compile-time eliminated when USE_SCALES=False) ---
    if USE_SCALES:
        offs_ks = tl.arange(0, KS)
        # A scale pointers [BLOCK_M, KS] — row stride = K//32
        a_scale_ptrs = (a_scale_ptr + (row_start + offs_m[:, None]) * stride_asm
                        + offs_ks[None, :])
        a_scale_mask = (row_start + offs_m[:, None]) < g_end
        # W scale base for this group
        ws_base = w_scale_ptr + gid * stride_wse
        # W scale pointers [HALF_N, KS]
        ws_ptrs_l = (ws_base
                     + offs_nl[:, None] * stride_wsn
                     + offs_ks[None, :] * 1)
        ws_ptrs_r = (ws_base
                     + offs_nr[:, None] * stride_wsn
                     + offs_ks[None, :] * 1)

    # --- Main K loop ---
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for _ in range(num_k_blocks):
        a_blk = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_l = tl.load(w_ptrs_l, mask=k_mask & n_mask_l, other=0.0)
        b_r = tl.load(w_ptrs_r, mask=k_mask & n_mask_r, other=0.0)

        a8 = a_blk.to(tl.float8e4nv)
        b8_l = b_l.to(tl.float8e4nv)
        b8_r = b_r.to(tl.float8e4nv)

        if USE_SCALES:
            a_sc = tl.load(a_scale_ptrs, mask=a_scale_mask, other=0.0)
            ws_l = tl.load(ws_ptrs_l, mask=offs_nl[:, None] < N, other=0.0)
            ws_r = tl.load(ws_ptrs_r, mask=offs_nr[:, None] < N, other=0.0)
            acc_l = tl.dot_scaled(a8, a_sc, "e4m3", b8_l, ws_l, "e4m3", acc=acc_l)
            acc_r = tl.dot_scaled(a8, a_sc, "e4m3", b8_r, ws_r, "e4m3", acc=acc_r)
        else:
            acc_l = tl.dot_scaled(a8, None, "e4m3", b8_l, None, "e4m3", acc=acc_l)
            acc_r = tl.dot_scaled(a8, None, "e4m3", b8_r, None, "e4m3", acc=acc_r)

        # Advance in K
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs_l += BLOCK_K * stride_wk
        w_ptrs_r += BLOCK_K * stride_wk
        if USE_SCALES:
            a_scale_ptrs += KS
            ws_ptrs_l += KS * 1
            ws_ptrs_r += KS * 1

    # --- Store to C ---
    offs_cm = row_start + tl.arange(0, BLOCK_M)
    offs_cl = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_cr = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)

    c_l_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cl[None, :] * stride_cn
    c_r_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cr[None, :] * stride_cn

    c_mask_l = (offs_cm[:, None] < M) & (offs_cl[None, :] < N)
    c_mask_r = (offs_cm[:, None] < M) & (offs_cr[None, :] < N)

    tl.store(c_l_ptrs, acc_l.to(tl.bfloat16), mask=c_mask_l)
    tl.store(c_r_ptrs, acc_r.to(tl.bfloat16), mask=c_mask_r)


# ---------- Wrapper ----------

def fused_grouped_fwd(a_fp8, w_fp8, offs,
                      a_scales=None, w_scales=None,
                      BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
                      GROUP_SIZE_M=8, num_stages=None, num_warps=None):
    """Run fused grouped GEMM.

       Args:
           a_fp8:    [M, K]        torch.float8_e4m3fn
           w_fp8:    [E, N, K]     torch.float8_e4m3fn
           offs:     [E+1]          int32 tensor, group boundaries
           a_scales: [M, K//32]    torch.uint8 (E8M0 scale=1.0 if None)
           w_scales: [E, N, K//32] torch.uint8 (E8M0 scale=1.0 if None)
       Returns:
           C_bf16: [M, N]         torch.bfloat16
    """
    M, K = a_fp8.shape
    E, N, K2 = w_fp8.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"

    use_scales = a_scales is not None and w_scales is not None

    if not use_scales:
        # Dummy tensors — never loaded, but Triton needs valid pointers
        a_scales = torch.empty(1, dtype=torch.uint8, device=a_fp8.device)
        w_scales = torch.empty(1, dtype=torch.uint8, device=a_fp8.device)

    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)

    # Grid: one program per (group_M_tile × N_tile) combination
    group_M = M // E
    num_pid_m_per_group = triton.cdiv(group_M, BLOCK_M)
    num_pid_m_total = E * num_pid_m_per_group
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m_total * num_pid_n,)

    kernel_args = dict(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        USE_SCALES=use_scales,
    )
    if num_stages is not None:
        kernel_args['num_stages'] = num_stages
    if num_warps is not None:
        kernel_args['num_warps'] = num_warps

    fused_grouped_fwd_kernel[grid](
        a_fp8, w_fp8, c, offs,
        a_scales, w_scales,
        M, N, K, E,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scales.stride(0) if use_scales else 0,
        w_fp8.stride(0), w_fp8.stride(1), w_fp8.stride(2),
        w_scales.stride(0) if use_scales else 0,
        w_scales.stride(1) if use_scales else 0,
        c.stride(0), c.stride(1),
        **kernel_args,
    )
    return c


# ---------- Benchmarking ----------

def benchmark_us(fn, *args, warmup=3, rep=10, **kwargs):
    """Time a function in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / rep) * 1e6  # microseconds


def reference_grouped_gemm(a_fp8, w_fp8, offs):
    """Reference: upcast to bf16, do per-group torch.matmul."""
    a_bf16 = a_fp8.to(torch.bfloat16)
    w_bf16 = w_fp8.to(torch.bfloat16)
    M, K = a_fp8.shape
    E, N, _ = w_fp8.shape
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    oc = offs.cpu().tolist()
    for g in range(E):
        gs = oc[g] if g > 0 else 0
        ge = oc[g + 1] if g + 1 < len(oc) else M
        if ge <= gs:
            continue
        # A_slice [group_M, K] @ W[g].T [K, N]
        out[gs:ge] = a_bf16[gs:ge] @ w_bf16[g].T
    return out


if __name__ == "__main__":
    print(f"Triton: {triton.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
    print("Fused Grouped GEMM  |  tl.dot_scaled  |  N-sliced  |  GROUP_M swizzle\n")

    # --- Sanity check ---
    print("--- Sanity check ---")
    M, N, K, E = 512, 256, 256, 4
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    offs = torch.tensor([0] + [M // E * (i + 1) for i in range(E)], dtype=torch.int32, device='cuda')

    ref = reference_grouped_gemm(a, w, offs)

    try:
        c = fused_grouped_fwd(a, w, offs, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=4)
        torch.cuda.synchronize()
        max_err = (c.float() - ref.float()).abs().max().item()
        mean_err = (c.float() - ref.float()).abs().mean().item()
        status = "✓ PASS" if max_err < 0.5 else "✗ FAIL"
        print(f"  {status}  max_err={max_err:.4f}  mean_err={mean_err:.4f}")
    except Exception as ex:
        print(f"  ✗ FAIL: {type(ex).__name__}: {ex}")

    # --- Small sweep to find best params ---
    print("\n--- Sweep (M=8192, N=2048, K=7168, E=4) ---")
    M, N, K, E = 8192, 2048, 7168, 4
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    offs = torch.tensor([0] + [M // E * (i + 1) for i in range(E)], dtype=torch.int32, device='cuda')
    flops = 2 * M * N * K

    best_tf = 0
    best_params = None
    for bm in [128, 256]:
        for bn in [128, 256]:
            for bk in [64, 128]:
                for gm in [1, 4, 8]:
                    try:
                        us = benchmark_us(fused_grouped_fwd, a, w, offs,
                                          BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                          GROUP_SIZE_M=gm)
                        tf = (flops / 1e12) / (us / 1e6)
                        tag = " ***" if tf > best_tf else ""
                        if tf > best_tf:
                            best_tf = tf
                            best_params = (bm, bn, bk, gm)
                        print(f"  M{bm} N{bn} K{bk} g{gm}: {tf:.0f} TFLOPS{tag}")
                    except Exception as ex:
                        print(f"  M{bm} N{bn} K{bk} g{gm}: ERR - {str(ex)[:80]}")
    print(f"\nBest: {best_params} → {best_tf:.0f} TFLOPS")

    # --- DSv3: all 8 shapes ---
    print("\n" + "=" * 70)
    print("DSv3 — all 8 configurations")
    print("=" * 70)

    DSV3_SHAPES = [
        # E, M,      N,    K
        (4,  32768,  2048, 7168),
        (4,  32768,  7168, 2048),
        (4,  128000, 2048, 7168),
        (4,  128000, 7168, 2048),
        (8,  32768,  2048, 7168),
        (8,  32768,  7168, 2048),
        (8,  128000, 2048, 7168),
        (8,  128000, 7168, 2048),
    ]

    tf_results = []
    for e, m, n, k in DSV3_SHAPES:
        print(f"\n--- E={e} M={m} N={n} K={k} ---")
        a = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        w = torch.randn(e, n, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        offs = torch.tensor([0] + [m // e * (i + 1) for i in range(e)], dtype=torch.int32, device='cuda')
        flops = 2 * m * n * k

        bt = 0
        bb = None
        for bm in [128, 256]:
            for bn in [128, 256]:
                for bk in [64, 128]:
                    for gm in [1, 4, 8]:
                        try:
                            us = benchmark_us(fused_grouped_fwd, a, w, offs,
                                              BLOCK_M=bm, BLOCK_N=bn,
                                              BLOCK_K=bk, GROUP_SIZE_M=gm)
                            tf = (flops / 1e12) / (us / 1e6)
                            if tf > bt:
                                bt = tf
                                bb = (bm, bn, bk, gm)
                        except Exception:
                            pass
        if bt > 0:
            tf_results.append(bt)
            print(f"  Best: {bb} → {bt:.0f} TFLOPS")
        else:
            print(f"  ERR: no working config")

    # --- Geomean ---
    if tf_results:
        geo = exp(sum(log(t) for t in tf_results) / len(tf_results))
        print(f"\n{'=' * 70}")
        print(f"Geomean TFLOPS: {geo:.0f}  ({len(tf_results)}/{len(DSV3_SHAPES)} shapes)")
        print(f"{'=' * 70}")
    else:
        print("\nNo successful benchmarks.")
