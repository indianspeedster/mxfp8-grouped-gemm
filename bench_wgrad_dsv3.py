#!/usr/bin/env python3
"""Wgrad DSv3 benchmark: BF16 torch.matmul vs MXFP8 fused forward"""
import torch, time, sys
import os
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_wgrad_bench'
from fused_fwd import fused_grouped_fwd
from wgrad_fused import wgrad_prepare, wgrad_kernel
from math import log, exp

DSV3 = [
    ("gate_up",   4,  2048, 7168, 32768),
    ("down",      4,  7168, 2048, 32768),
    ("gate_up",   4,  2048, 7168, 128000),
    ("down",      4,  7168, 2048, 128000),
    ("gate_up",   8,  2048, 7168, 32768),
    ("down",      8,  7168, 2048, 32768),
    ("gate_up",   8,  2048, 7168, 128000),
    ("down",      8,  7168, 2048, 128000),
]

print(f"DSv3 Wgrad: BF16 vs MXFP8  ({torch.cuda.get_device_name(0)})\n")
print(f"{'Shape':>8} {'E':>2} {'M':>7} {'N':>5} {'K':>5} {'BF16 TFLOPS':>12} {'MXFP8 TFLOPS':>13} {'Speedup':>8}")

def bench(fn, warmup=5, reps=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

tf_bf16 = []; tf_fp8 = []

for name, e, n, k, m in DSV3:
    mg = m // e
    go = torch.randn(n, m, dtype=torch.bfloat16, device='cuda')
    ia = torch.randn(k, m, dtype=torch.bfloat16, device='cuda')
    flops = 2 * m * n * k

    # --- BF16 baseline ---
    out = torch.empty(e, n, k, dtype=torch.bfloat16, device='cuda')
    def bf16_run():
        for g in range(e):
            gs, ge = g * mg, (g + 1) * mg
            out[g] = go[:, gs:ge] @ ia[:, gs:ge].T
    t = bench(bf16_run)
    t_bf16 = (flops / 1e12) / t

    # --- MXFP8 fused ---
    go8 = go.to(torch.float8_e4m3fn)
    ia8 = ia.to(torch.float8_e4m3fn)
    data = wgrad_prepare(go8, ia8, torch.tensor([mg] * e, dtype=torch.int32, device='cuda'))
    def fp8_run():
        return wgrad_kernel(data)
    t = bench(fp8_run)
    t_fp8 = (flops / 1e12) / t

    tf_bf16.append(t_bf16); tf_fp8.append(t_fp8)
    print(f"  {name:>8} {e:>2} {m:>7} {n:>5} {k:>5} {t_bf16:>12.0f} {t_fp8:>13.0f} {t_fp8/t_bf16:>7.2f}x")

geo_bf16 = exp(sum(log(t) for t in tf_bf16) / len(tf_bf16))
geo_fp8  = exp(sum(log(t) for t in tf_fp8) / len(tf_fp8))
print(f"\n{'Geomean':>24} {geo_bf16:>12.0f} {geo_fp8:>13.0f} {geo_fp8/geo_bf16:>7.2f}x")
