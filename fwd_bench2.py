#!/usr/bin/env python3
"""Double-buffered Gluon forward GEMM benchmark"""
import torch, triton, math
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def fwd_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr, NUM_WARPS: gl.constexpr,
    GROUP_M: gl.constexpr):
    
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
    tpn: gl.constexpr = triton.cdiv(BLOCK_K * BLOCK_N // (NUM_WARPS * 64), 16)

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
    smem_b = gl.allocate_shared_memory(b_ptr.type.element_ty, [NBUF, BLOCK_K, BLOCK_N], layout=sb)

    offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_a))
    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_a))
    offs_an = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_b))
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_b))
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Prologue: load both buffers
    for buf in range(NBUF):
        k_off = buf * BLOCK_K
        if k_off < K:
            a = gl.amd.cdna4.buffer_load(ptr=a_ptr + k_off * stride_ak, offsets=offs_an,
                                           mask=offs_am[:, None] < M)
            b = gl.amd.cdna4.buffer_load(ptr=b_ptr + k_off * stride_bk, offsets=offs_b,
                                           mask=offs_bn[None, :] < N)
            smem_a.index(buf).store(a); smem_b.index(buf).store(b)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma)
    iterMax = gl.cdiv(K, BLOCK_K)

    for k in range(0, iterMax):
        buf_prefetch = (k + 1) % NBUF
        
        k_future = (k + NBUF) * BLOCK_K
        if k_future < K:
            a = gl.amd.cdna4.buffer_load(ptr=a_ptr + k_future * stride_ak, offsets=offs_an,
                                           mask=offs_am[:, None] < M)
            b = gl.amd.cdna4.buffer_load(ptr=b_ptr + k_future * stride_bk, offsets=offs_b,
                                           mask=offs_bn[None, :] < N)
            smem_a.index(buf_prefetch).store(a); smem_b.index(buf_prefetch).store(b)
        
        buf_cur = k % NBUF
        cur_a = smem_a.index(buf_cur).load(da)
        cur_b = smem_b.index(buf_cur).load(db)
        acc = gl.amd.cdna4.mfma_scaled(
            a=cur_a, a_scale=None, a_format="e4m3",
            b=cur_b, b_scale=None, b_format="e4m3", acc=acc)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma))
    offs_c = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna4.buffer_store(acc.to(gl.bfloat16), c_ptr, offs_c, mask_c)


def fwd_ge(x_fp8, w_fp8, BLOCK_M=256, BLOCK_N=128, BLOCK_K=64, NUM_WARPS=4, GROUP_M=4):
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
    print(f"Triton: {triton.__version__} | GPU: {torch.cuda.get_device_name(0)}")

    # Big GEMM sweep
    M, N, K = 8192, 2048, 7168
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(K, N, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    flops = 2 * M * N * K
    
    print(f"\nSweep M{M} N{N} K{K}:")
    best = 0
    for bm in [128, 256]:
        for bn in [128, 256]:
            for bk in [64, 128]:
                for gm in [1, 4]:
                    try:
                        us = benchmark_cuda_function_in_microseconds(
                            fwd_ge, a, w,
                            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, NUM_WARPS=4, GROUP_M=gm)
                        tf = (flops / 1e12) / (us / 1e6)
                        tag = " ***" if tf > best else ""
                        if tf > best: best = tf
                        print(f"  M{bm} N{bn} K{bk} g{gm}: {tf:.0f} TFLOPS{tag}")
                    except Exception as ex:
                        print(f"  M{bm} N{bn} K{bk} g{gm}: ERR - {str(ex)[:80]}")

    print(f"\nBest: {best:.0f} TFLOPS")
    
    # DSv3 shapes
    print("\n--- DSv3 shapes ---")
    DSV3 = [(4, 32768, 2048, 7168), (8, 32768, 2048, 7168),
            (4, 128000, 2048, 7168), (8, 128000, 2048, 7168),
            (4, 32768, 7168, 2048), (8, 32768, 7168, 2048),
            (4, 128000, 7168, 2048), (8, 128000, 7168, 2048)]
    
    tf_list = []
    for e, m, n, k in DSV3:
        offs = torch.tensor([m//e*(i+1) for i in range(e)], dtype=torch.int32, device='cuda')
        offs[-1] = m
        x = torch.randn(m, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        w_3d = torch.randn(e, n, k, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        sx = torch.ones(m, k // 32, dtype=torch.uint8, device='cuda')
        sw = torch.ones(e, n, k // 32, dtype=torch.uint8, device='cuda')
        flops = 2 * m * n * k
        
        best_tf = 0
        for bm in [128, 256]:
            for bn in [64, 128]:
                for bk in [64, 128]:
                    for gm in [1, 4]:
                        try:
                            def run():
                                out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
                                offs_cpu = offs.cpu().tolist()
                                for g in range(e):
                                    gs = offs_cpu[g-1] if g > 0 else 0; ge = offs_cpu[g]
                                    Mg = ge - gs
                                    if Mg == 0: continue
                                    A_g = torch.narrow(x, 0, gs, Mg)
                                    B_g_T = w_3d[g].permute(1, 0).contiguous()
                                    grid = (triton.cdiv(Mg, bm) * triton.cdiv(n, bn),)
                                    c_g = torch.empty(Mg, n, dtype=torch.bfloat16, device='cuda')
                                    fwd_kernel[grid](
                                        A_g, B_g_T, c_g, Mg, n, k,
                                        A_g.stride(0), A_g.stride(1), B_g_T.stride(0), B_g_T.stride(1),
                                        c_g.stride(0), c_g.stride(1),
                                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                        NUM_WARPS=4, GROUP_M=gm)
                                    out[gs:ge] = c_g
                                return out
                            us = benchmark_cuda_function_in_microseconds(run)
                            tf = (flops / 1e12) / (us / 1e6)
                            if tf > best_tf: best_tf = tf
                        except Exception:
                            pass
        if best_tf > 0:
            tf_list.append(best_tf)
            print(f"  E{e} M{m} N{n} K{k}: {best_tf:.0f} TFLOPS")
    
    if tf_list:
        from math import exp, log
        geomean = exp(sum(log(t) for t in tf_list) / len(tf_list))
        print(f"\nGeomean: {geomean:.0f} TFLOPS ({len(tf_list)}/{len(DSV3)} shapes)")
