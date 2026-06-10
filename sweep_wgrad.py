#!/usr/bin/env python3
"""wgrad: comprehensive vs BF16"""
import torch, sys
import os
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache_wgrad_f'
from fused_fwd import fused_grouped_fwd, benchmark_us
from math import log, exp
import warnings
warnings.filterwarnings('ignore')

DSV3 = [
    (4,  2048, 7168, 32768),
    (4,  7168, 2048, 32768),
    (4,  2048, 7168, 128000),
    (4,  7168, 2048, 128000),
    (8,  2048, 7168, 32768),
    (8,  7168, 2048, 32768),
    (8,  2048, 7168, 128000),
    (8,  7168, 2048, 128000),
]

print(f"wgrad MXFP8 vs BF16  ({torch.cuda.get_device_name(0)})\n")

tf_fp8 = []; tf_bf16 = []
for ei, (e,n,k,m) in enumerate(DSV3):
    mg = m // e
    go = torch.randn(n, m, dtype=torch.bfloat16, device='cuda')
    ia = torch.randn(k, m, dtype=torch.bfloat16, device='cuda')
    
    # MXFP8
    A = torch.empty(e * n, mg, dtype=torch.float8_e4m3fn, device='cuda')
    B = torch.empty(e, k, mg, dtype=torch.float8_e4m3fn, device='cuda')
    go_fp8 = go.to(torch.float8_e4m3fn)
    ia_fp8 = ia.to(torch.float8_e4m3fn)
    for g in range(e):
        gs, ge = g * mg, (g + 1) * mg
        A[g * n : (g + 1) * n] = go_fp8[:, gs:ge]
        B[g] = ia_fp8[:, gs:ge]
    offs = torch.tensor([n * (i + 1) for i in range(e)], dtype=torch.int32, device='cuda')
    flops = 2 * m * n * k
    
    bt = 0; bb = ''
    for bm in [128, 256]:
        for bn in [128, 256]:
            for bk in [64, 128]:
                if bk == 64 and bn == 256: continue
                for gm in [1, 4, 8]:
                    for ns in [None, 2, 3]:
                        try:
                            us = benchmark_us(fused_grouped_fwd, A, B, offs,
                                              BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                              GROUP_SIZE_M=gm,
                                              num_stages=ns, num_warps=4)
                            tf = (flops / 1e12) / (us / 1e6)
                            if tf > bt: bt = tf; bb = f"bm={bm} bn={bn} bk={bk} gm={gm} ns={ns}"
                        except: pass
    
    # BF16
    go_bf16 = go.to(torch.bfloat16)
    ia_bf16 = ia.to(torch.bfloat16)
    out_bf16 = torch.empty(e, n, k, dtype=torch.bfloat16, device='cuda')
    def bf16_kernel():
        for g in range(e):
            gs, ge = g * mg, (g + 1) * mg
            out_bf16[g] = go_bf16[:, gs:ge] @ ia_bf16[:, gs:ge].T
    us_bf16 = benchmark_us(bf16_kernel)
    tf_b = (flops / 1e12) / (us_bf16 / 1e6)
    
    tf_fp8.append(bt); tf_bf16.append(tf_b)
    x = bt / tf_b
    print(f"  E={e} M={m:>6} N={n} K={k}: MXFP8={bt:.0f}  BF16={tf_b:.0f}  ({x:.2f}x)  [{bb}]")

geo_fp8 = exp(sum(log(t) for t in tf_fp8) / len(tf_fp8))
geo_bf16 = exp(sum(log(t) for t in tf_bf16) / len(tf_bf16))
print(f"\nGeomean: MXFP8={geo_fp8:.0f} TFLOPS  |  BF16={geo_bf16:.0f} TFLOPS  |  {geo_fp8/geo_bf16:.2f}x")
