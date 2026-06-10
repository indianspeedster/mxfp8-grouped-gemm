#!/usr/bin/env python3
"""Minimal repro: scale factor=2 bug"""
import torch, triton, triton.language as tl

BK = 128; KS = BK // 32  # 4

@triton.jit
def repro(a_ptr, b_ptr, a_s_ptr, b_s_ptr, c_ptr, M, N, K, stride_asm,
          BM: tl.constexpr, BN: tl.constexpr, BK_: tl.constexpr):
    KS_: tl.constexpr = BK_ // 32
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rk = tl.arange(0, BK_)
    rn = tl.arange(0, BN)
    rks = tl.arange(0, KS_)
    
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=rk[:, None] < K, other=0.0)
    a8 = a.to(tl.float8e4nv)
    b8 = b.to(tl.float8e4nv)
    
    # Correct: use stride_asm (K//32), NOT KS (BK//32)
    as1 = tl.load(a_s_ptr + rm[:, None] * stride_asm + rks[None, :], mask=rm[:, None] < M, other=0.0)
    d1 = tl.dot_scaled(a8, as1, "e4m3", b8, None, "e4m3")
    
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], d1.to(tl.bfloat16), mask=rm[:, None] < M)

M, N, K = 128, 128, 128
a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
a_s = torch.zeros(M, K//32, dtype=torch.uint8, device='cuda').fill_(0x7F)
b_s = torch.zeros(N, K//32, dtype=torch.uint8, device='cuda').fill_(0x7F)
c = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

for bm, bn, bk in [(64, 64, 128), (128, 128, 128), (128, 64, 128), (256, 128, 128)]:
    try:
        c.zero_()
        repro[(1,)](a, b, a_s, b_s, c, M, N, K, a_s.stride(0), BM=bm, BN=bn, BK_=bk)
        torch.cuda.synchronize()
        print(f"  M{bm} N{bn} K{bk}: OK sum={c.float().sum().item():.1f}")
    except Exception as e:
        err = str(e).split('\n')[-2] if '\n' in str(e) else str(e)[:200]
        print(f"  M{bm} N{bn} K{bk}: FAIL - {err[:150]}")

# Now test: what if K (full problem) != BK ?
print("\n--- Test with K > BK ---")
M2, N2, K2 = 512, 256, 256  # K2=256, BK=128, K2//32=8, KS=4
a2 = torch.randn(M2, K2, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
b2 = torch.randn(K2, N2, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
a_s2 = torch.zeros(M2, K2//32, dtype=torch.uint8, device='cuda').fill_(0x7F)
b_s2 = torch.zeros(N2, K2//32, dtype=torch.uint8, device='cuda').fill_(0x7F)
c2 = torch.zeros(M2, N2, dtype=torch.bfloat16, device='cuda')

for bm, bn, bk in [(128, 128, 128), (128, 64, 128)]:
    try:
        c2.zero_()
        repro[(1,)](a2, b2, a_s2, b_s2, c2, M2, N2, K2, a_s2.stride(0), BM=bm, BN=bn, BK_=bk)
        torch.cuda.synchronize()
        print(f"  M{bm} N{bn} K{bk}: OK sum={c2.float().sum().item():.1f}")
    except Exception as e:
        err = str(e).split('\n')[-2] if '\n' in str(e) else str(e)[:200]
        print(f"  M{bm} N{bn} K{bk}: FAIL - {err[:150]}")
