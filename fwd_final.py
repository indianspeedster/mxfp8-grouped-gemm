#!/usr/bin/env python3
"""Final N-sliced, double-buffered Gluon MXFP8 forward GEMM"""
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def fwd_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr, NUM_WARPS: gl.constexpr, GROUP_M: gl.constexpr):
    
    HALF_N: gl.constexpr = BLOCK_N // 2
    
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, BLOCK_M); num_pid_n = gl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m = pid // num_pid_n; pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = num_pid_m - first_pid_m
        group_size_m = gl.minimum(group_size_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    tpm: gl.constexpr = triton.cdiv(BLOCK_M * BLOCK_K // (NUM_WARPS * 64), 16)
    tpn: gl.constexpr = triton.cdiv(BLOCK_K * HALF_N // (NUM_WARPS * 64), 16)

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[tpm, 16], threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, tpn], threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1])

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64],
        transposed=True, warps_per_cta=[NUM_WARPS // 2, 2])

    sa: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8, order=[1, 0])
    sb: gl.constexpr = gl.SwizzledSharedLayout(vec=16, per_phase=2, max_phase=8, order=[0, 1])
    da: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    db: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)

    NBUF: gl.constexpr = 2
    smem_a = gl.allocate_shared_memory(a_ptr.type.element_ty, [NBUF, BLOCK_M, BLOCK_K], layout=sa)
    smem_bl = gl.allocate_shared_memory(b_ptr.type.element_ty, [NBUF, HALF_N, BLOCK_K], layout=sb)
    smem_br = gl.allocate_shared_memory(b_ptr.type.element_ty, [NBUF, HALF_N, BLOCK_K], layout=sb)

    offs_am = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_a))
    offs_an = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    abs_m = pid_m * BLOCK_M + offs_am
    mask_a = abs_m[:, None] < M

    offs_bnl = gl.arange(0, HALF_N, layout=gl.SliceLayout(1, blocked_b))
    offs_bnr = offs_bnl + HALF_N
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_b))
    b_left_offsets = offs_bnl[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_right_offsets = offs_bnr[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    abs_n_l = pid_n * BLOCK_N + offs_bnl
    abs_n_r = pid_n * BLOCK_N + offs_bnr
    mask_bl = abs_n_l[:, None] < N
    mask_br = abs_n_r[:, None] < N

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Prologue: both buffers
    for buf in range(NBUF):
        k_off = buf * BLOCK_K
        a = gl.amd.cdna4.buffer_load(ptr=a_base + k_off * stride_ak, offsets=offs_an, mask=mask_a)
        bl = gl.amd.cdna4.buffer_load(ptr=b_base + k_off * stride_bk, offsets=b_left_offsets, mask=mask_bl)
        br = gl.amd.cdna4.buffer_load(ptr=b_base + k_off * stride_bk, offsets=b_right_offsets, mask=mask_br)
        smem_a.index(buf).store(a)
        smem_bl.index(buf).store(bl)
        smem_br.index(buf).store(br)

    acc_left = gl.zeros((BLOCK_M, HALF_N), gl.float32, mfma)
    acc_right = gl.zeros((BLOCK_M, HALF_N), gl.float32, mfma)
    iterMax = gl.cdiv(K, BLOCK_K)

    for k in range(0, iterMax):
        buf_prefetch = (k + 1) % NBUF
        k_future = (k + NBUF) * BLOCK_K
        if k_future < K:
            a = gl.amd.cdna4.buffer_load(ptr=a_base + k_future * stride_ak, offsets=offs_an, mask=mask_a)
            bl = gl.amd.cdna4.buffer_load(ptr=b_base + k_future * stride_bk, offsets=b_left_offsets, mask=mask_bl)
            br = gl.amd.cdna4.buffer_load(ptr=b_base + k_future * stride_bk, offsets=b_right_offsets, mask=mask_br)
            smem_a.index(buf_prefetch).store(a)
            smem_bl.index(buf_prefetch).store(bl)
            smem_br.index(buf_prefetch).store(br)
        
        buf_cur = k % NBUF
        cur_a = smem_a.index(buf_cur).load(da)
        cur_bl = smem_bl.index(buf_cur).load(db)
        cur_br = smem_br.index(buf_cur).load(db)
        
        acc_left = gl.amd.cdna4.mfma_scaled(
            a=cur_a, a_scale=None, a_format="e4m3",
            b=cur_bl, b_scale=None, b_format="e4m3", acc=acc_left)
        acc_right = gl.amd.cdna4.mfma_scaled(
            a=cur_a, a_scale=None, a_format="e4m3",
            b=cur_br, b_scale=None, b_format="e4m3", acc=acc_right)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma))
    offs_cl = gl.arange(0, HALF_N, layout=gl.SliceLayout(0, mfma))
    offs_cr = offs_cl + HALF_N
    c_l_offsets = offs_cm[:, None] * stride_cm + offs_cl[None, :] * stride_cn
    c_r_offsets = offs_cm[:, None] * stride_cm + offs_cr[None, :] * stride_cn
    abs_cl_n = pid_n * BLOCK_N + offs_cl
    abs_cr_n = pid_n * BLOCK_N + offs_cr
    mask_cl = (offs_cm[:, None] < M) & (abs_cl_n[None, :] < N)
    mask_cr = (offs_cm[:, None] < M) & (abs_cr_n[None, :] < N)
    gl.amd.cdna4.buffer_store(acc_left.to(gl.bfloat16), c_ptr, c_l_offsets, mask_cl)
    gl.amd.cdna4.buffer_store(acc_right.to(gl.bfloat16), c_ptr, c_r_offsets, mask_cr)


def fwd_ge(x_fp8, w_fp8, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, NUM_WARPS=4, GROUP_M=4):
    M, K = x_fp8.shape; K2, N = w_fp8.shape; assert K == K2
    c = torch.empty(M, N, dtype=torch.bfloat16, device=x_fp8.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    fwd_kernel[grid](
        x_fp8, w_fp8, c, M, N, K,
        x_fp8.stride(0), x_fp8.stride(1), w_fp8.stride(0), w_fp8.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_WARPS=NUM_WARPS, GROUP_M=GROUP_M)
    return c


if __name__ == "__main__":
    from utils import benchmark_cuda_function_in_microseconds
    from math import exp, log
    print(f"Triton: {triton.__version__} | GPU: {torch.cuda.get_device_name(0)}")

    # Sweep
    M, N, K = 8192, 2048, 7168
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(K, N, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    flops = 2 * M * N * K
    print(f"\nSweep M{M} N{N} K{K} (N-sliced, HN==BK only):")
    best = 0
    for bm in [128, 192, 256]:
        for bn2 in [64, 128]:  # HALF_N = bn2, BN = 2*bn2
            bn = bn2 * 2
            for bk in [bn2]:  # BK == HALF_N
                for gm in [1, 4]:
                    try:
                        us = benchmark_cuda_function_in_microseconds(
                            fwd_ge, a, w, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm)
                        tf = (flops / 1e12) / (us / 1e6)
                        tag = " ***" if tf > best else ""
                        if tf > best: best = tf
                        print(f"  M{bm} N{bn} K{bk} g{gm}: {tf:.0f} TFLOPS{tag}")
                    except Exception as ex:
                        print(f"  M{bm} N{bn} K{bk} g{gm}: ERR - {str(ex)[:80]}")
    print(f"\nBest: {best:.0f} TFLOPS")

    # DSv3
    print("\n--- DSv3 shapes (N-sliced) ---")
    DSV3 = [(4, 32768, 2048, 7168), (8, 32768, 2048, 7168),
            (4, 128000, 2048, 7168), (8, 128000, 2048, 7168),
            (4, 32768, 7168, 2048), (8, 32768, 7168, 2048),
            (4, 128000, 7168, 2048), (8, 128000, 7168, 2048)]
    tf_all = []
    for e, m, n, k in DSV3:
        offs = torch.tensor([m//e*(i+1) for i in range(e)], dtype=torch.int32, device='cuda')
        offs[-1] = m
        x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        w3 = torch.randn(e, n, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        flops = 2 * m * n * k
        bt = 0
        for bm in [128, 192, 256]:
            for bk in [64, 128]:  # BK determines BN = 2*BK, constraint HN==BK
                bn = bk * 2
                for gm in [1, 4, 8]:
                    try:
                        def run():
                            oc = offs.cpu().tolist()
                            out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
                            for g in range(e):
                                gs = oc[g-1] if g > 0 else 0; ge = oc[g]
                                Mg = ge - gs
                                if Mg == 0: continue
                                Ag = torch.narrow(x, 0, gs, Mg)
                                BgT = w3[g].permute(1, 0).contiguous()
                                cg = fwd_ge(Ag, BgT, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm)
                                out[gs:ge] = cg
                            return out
                        us = benchmark_cuda_function_in_microseconds(run)
                        tf = (flops / 1e12) / (us / 1e6)
                        if tf > bt: bt = tf
                    except Exception:
                        pass
        if bt > 0:
            tf_all.append(bt)
            print(f"  E{e} M{m} N{n} K{k}: {bt:.0f} TFLOPS")
    if tf_all:
        geo = exp(sum(log(t) for t in tf_all) / len(tf_all))
        print(f"\nGeomean: {geo:.0f} TFLOPS ({len(tf_all)}/{len(DSV3)} shapes)")
