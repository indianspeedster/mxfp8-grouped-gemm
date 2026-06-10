#!/usr/bin/env python3
"""Standard Triton dot_scaled N-sliced double-buffered forward GEMM"""
import torch, triton, triton.language as tl

@triton.jit
def fwd_kernel_std(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr):
    
    HALF_N: tl.constexpr = BLOCK_N // 2
    
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    
    # A tile
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ak = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_mask = offs_am[:, None] < M
    
    # B tiles (left and right halves)
    offs_bk = tl.arange(0, BLOCK_K)
    offs_bnl = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_bnr = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)
    b_l_ptrs = b_ptr + offs_bnl[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_r_ptrs = b_ptr + offs_bnr[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_l_mask = offs_bnl[:, None] < N
    b_r_mask = offs_bnr[:, None] < N
    
    # Accumulators
    acc_l = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_r = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    
    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_l = tl.load(b_l_ptrs, mask=b_l_mask, other=0.0)
        b_r = tl.load(b_r_ptrs, mask=b_r_mask, other=0.0)
        
        a8 = a.to(tl.float8e4nv)
        b8_l = b_l.to(tl.float8e4nv)
        b8_r = b_r.to(tl.float8e4nv)
        
        acc_l = tl.dot_scaled(a8, None, "e4m3", b8_l, None, "e4m3", acc=acc_l)
        acc_r = tl.dot_scaled(a8, None, "e4m3", b8_r, None, "e4m3", acc=acc_r)
        
        a_ptrs += BLOCK_K * stride_ak
        b_l_ptrs += BLOCK_K * stride_bk
        b_r_ptrs += BLOCK_K * stride_bk
    
    # Store
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cl = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_cr = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)
    c_l_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cl[None, :] * stride_cn
    c_r_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cr[None, :] * stride_cn
    c_l_mask = (offs_cm[:, None] < M) & (offs_cl[None, :] < N)
    c_r_mask = (offs_cm[:, None] < M) & (offs_cr[None, :] < N)
    tl.store(c_l_ptrs, acc_l.to(tl.bfloat16), mask=c_l_mask)
    tl.store(c_r_ptrs, acc_r.to(tl.bfloat16), mask=c_r_mask)


def fwd_std(x_fp8, w_fp8, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4):
    M, K = x_fp8.shape; K2, N = w_fp8.shape; assert K == K2
    c = torch.empty(M, N, dtype=torch.bfloat16, device=x_fp8.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    fwd_kernel_std[grid](
        x_fp8, w_fp8, c, M, N, K,
        x_fp8.stride(0), x_fp8.stride(1), w_fp8.stride(0), w_fp8.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    return c


if __name__ == "__main__":
    from utils import benchmark_cuda_function_in_microseconds
    from math import exp, log
    print(f"Triton: {triton.__version__} | GPU: {torch.cuda.get_device_name(0)}")
    print("Standard Triton dot_scaled | N-sliced forward\n")

    M, N, K = 8192, 2048, 7168
    a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(K, N, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
    flops = 2 * M * N * K
    print(f"Sweep M{M} N{N} K{K}:")
    best = 0
    for bm in [128, 256]:
        for bn in [128, 256]:
            for bk in [128]:
                for gm in [1, 4]:
                    try:
                        us = benchmark_cuda_function_in_microseconds(
                            fwd_std, a, w, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm)
                        tf = (flops / 1e12) / (us / 1e6)
                        tag = " ***" if tf > best else ""
                        if tf > best: best = tf
                        print(f"  M{bm} N{bn} K{bk} g{gm}: {tf:.0f} TFLOPS{tag}")
                    except Exception as ex:
                        print(f"  M{bm} N{bn} K{bk} g{gm}: ERR - {str(ex)[:100]}")
    print(f"\nBest: {best:.0f} TFLOPS")

    # DSv3
    print("\n--- DSv3 (Standard Triton) ---")
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
        bt = 0
        for bm in [128, 256]:
            for bn in [128, 256]:
                for gm in [1, 4]:
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
                                cg = fwd_std(Ag, BgT, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=gm)
                                out[gs:ge] = cg
                            return out
                        us = benchmark_cuda_function_in_microseconds(run)
                        tf = 2 * m * n * k / 1e12 / (us / 1e6)
                        if tf > bt: bt = tf
                    except Exception:
                        pass
        if bt > 0:
            tf_all.append(bt)
            print(f"  E{e} M{m} N{n} K{k}: {bt:.0f} TFLOPS")
    if tf_all:
        geo = exp(sum(log(t) for t in tf_all) / len(tf_all))
        print(f"\nGeomean: {geo:.0f} TFLOPS ({len(tf_all)}/{len(DSV3)} shapes)")
