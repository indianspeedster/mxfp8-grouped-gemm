#!/usr/bin/env python3
"""DSv3-16B MoE TPS: benchmark forward + backward independently.

   DSv3-16B EP=8: 8 local experts, ~12K tokens/GPU, dim=2048, hidden=1408.

   Forward:  3× fused_grouped_fwd (gate, up, down) + SiLU
   Backward: 3× dgrad (dC @ W via per-expert matmul) + 3× wgrad (via wgrad_fused)
"""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_fwd import fused_grouped_fwd
from wgrad_fused import wgrad_prepare, wgrad_kernel

torch.backends.cuda.matmul.allow_tf32 = False
device = 'cuda'

# DSv3-16B, EP=8
BS, SEQ = 4, 4096
DIM, HIDDEN = 2048, 1408
E = 8  # local experts
M = BS * SEQ * 6 // 64 * E  # ~12288
MG = M // E  # ~1536

print(f"DSv3-16B EP=8: E={E} M={M} dim={DIM} hidden={HIDDEN}")
print(f"  tokens/expert: {MG}")

# Weights — as GroupedExperts stores them: w[E, hidden, dim] for gate/up, w[E, dim, hidden] for down
w1 = torch.randn(E, HIDDEN, DIM, dtype=torch.bfloat16, device=device)
w2 = torch.randn(E, DIM, HIDDEN, dtype=torch.bfloat16, device=device)
w3 = torch.randn(E, HIDDEN, DIM, dtype=torch.bfloat16, device=device)

# For fused kernel: expects [E, N, K] where N=output, K=reduction
# Gate/up: [E, HIDDEN, DIM] → N=HIDDEN, K=DIM ✓
# Down:    [E, DIM, HIDDEN] → N=DIM, K=HIDDEN ✓
w1_fp8 = w1.to(torch.float8_e4m3fn)
w2_fp8 = w2.to(torch.float8_e4m3fn)
w3_fp8 = w3.to(torch.float8_e4m3fn)

x = torch.randn(M, DIM, dtype=torch.bfloat16, device=device)
x_fp8 = x.to(torch.float8_e4m3fn)

offs = torch.tensor([MG * (i + 1) for i in range(E)], dtype=torch.int32, device=device)

FLOP_FWD = 3 * (2 * M * DIM * HIDDEN)
FLOP_BWD = 2 * FLOP_FWD  # dgrad + wgrad

def time_fn(fn, warmup=10, reps=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

# ─── Forward ───
def fwd():
    hg = fused_grouped_fwd(x_fp8, w1_fp8, offs)
    hu = fused_grouped_fwd(x_fp8, w3_fp8, offs)
    h = torch.nn.functional.silu(hg.to(torch.bfloat16)) * hu.to(torch.bfloat16)
    h_fp8 = h.to(torch.float8_e4m3fn)
    return fused_grouped_fwd(h_fp8, w2_fp8, offs)

out = fwd(); torch.cuda.synchronize()
fwd_s = time_fn(fwd)
fwd_tf = FLOP_FWD / 1e12 / fwd_s
print(f"\nForward:  {fwd_s*1000:.3f} ms  ({fwd_tf:.0f} TFLOPS)")

# ─── Backward (dgrad + wgrad) ───
# Prepare static data
dC = torch.randn(M, DIM, dtype=torch.bfloat16, device=device)

# dgrad: d_gate = dC @ w2  → per-expert matmul (small, this is fine)
# Actually, we can batch: d_h[g] = dC_g @ w2_g
def dgrad():
    dh = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=device)
    for g in range(E):
        gs, ge = g * MG, (g + 1) * MG
        dh[gs:ge] = dC[gs:ge] @ w2[g]
    return dh

# For gate/up dgrad, we need SiLU backward first
# Let's compute all dgrad in one function
h_gate = fused_grouped_fwd(x_fp8, w1_fp8, offs).to(torch.bfloat16)
h_up   = fused_grouped_fwd(x_fp8, w3_fp8, offs).to(torch.bfloat16)
gate_silu = torch.nn.functional.silu(h_gate)

# Pre-compute SiLU backward intermediates
sigmoid_gate = torch.sigmoid(h_gate)
silu_grad_mul = sigmoid_gate * (1 + gate_silu * (1 - sigmoid_gate))

def dgrad_full():
    # Down: dh = dC @ w2 per expert
    dh = torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=device)
    for g in range(E):
        gs, ge = g * MG, (g + 1) * MG
        dh[gs:ge] = dC[gs:ge] @ w2[g]
    
    # SiLU backward
    d_gate = dh * h_up.to(torch.bfloat16) * silu_grad_mul
    d_up = dh * gate_silu
    
    # Gate dgrad: d_x1 = d_gate @ w1 per expert
    # Up dgrad:   d_x2 = d_up @ w3 per expert
    dx = torch.zeros(M, DIM, dtype=torch.bfloat16, device=device)
    for g in range(E):
        gs, ge = g * MG, (g + 1) * MG
        dx[gs:ge] = d_gate[gs:ge] @ w1[g] + d_up[gs:ge] @ w3[g]
    return dx

dgrad_full(); torch.cuda.synchronize()
dg_s = time_fn(dgrad_full, warmup=5, reps=50)

# wgrad via fused kernel
go_down = dC.T.contiguous().to(torch.float8_e4m3fn)
h_bf16 = gate_silu * h_up.to(torch.bfloat16)
ia_down = h_bf16.to(torch.float8_e4m3fn).T.contiguous()
data_down = wgrad_prepare(go_down, ia_down, offs)

d_gate_bf16 = h_up.to(torch.bfloat16) * silu_grad_mul * torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)
go_gate = d_gate_bf16.T.contiguous().to(torch.float8_e4m3fn)
ia_gate = x.to(torch.float8_e4m3fn).T.contiguous()
data_gate = wgrad_prepare(go_gate, ia_gate, offs)

d_up_bf16 = gate_silu * torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)
go_up = d_up_bf16.T.contiguous().to(torch.float8_e4m3fn)
data_up = wgrad_prepare(go_up, ia_gate, offs)

def wgrad_all():
    _dw2 = wgrad_kernel(data_down)
    _dw1 = wgrad_kernel(data_gate)
    _dw3 = wgrad_kernel(data_up)

wgrad_all(); torch.cuda.synchronize()
wg_s = time_fn(wgrad_all, warmup=5, reps=50)

bwd_s = dg_s + wg_s
bwd_tf = FLOP_BWD / 1e12 / bwd_s
step_s = fwd_s + bwd_s
step_tf = (FLOP_FWD + FLOP_BWD) / 1e12 / step_s

print(f"  Dgrad:   {dg_s*1000:.3f} ms")
print(f"  Wgrad:   {wg_s*1000:.3f} ms")
print(f"  Backward:{bwd_s*1000:.3f} ms  ({bwd_tf:.0f} TFLOPS)")
print(f"  Step:    {step_s*1000:.3f} ms  ({step_tf:.0f} TFLOPS)")

# ─── TPS ───
N_MOE = 26
moe_step = N_MOE * step_s

# Dense FFN: dim=2048, hidden=10944, M=12288 (same total tokens, no routing)
FLOP_DENSE_FWD = 2 * M * DIM * 10944
# ~BF16 TFLOPS for dense: ~1100 TFLOPS (torch.matmul)
dense_fwd_s = FLOP_DENSE_FWD / 1100e12
dense_bwd_s = dense_fwd_s * 2
dense_step_s = (dense_fwd_s + dense_bwd_s) * 1.1  # SiLU overhead

# Attention: MLA with 4 large matmuls, ~2*M*dim*dim each direction
FLOP_ATTN = 2 * M * DIM * DIM * 3  # rough
attn_fwd_s = FLOP_ATTN / 800e12  # less efficient than MoE
attn_bwd_s = attn_fwd_s * 2  
attn_step_s = (attn_fwd_s + attn_bwd_s) * 27

# Embedding, LM head, norms: small
other_s = 0.005  # 5ms

total_s = moe_step + dense_step_s + attn_step_s + other_s
tps = BS * SEQ / total_s

print(f"\n{'='*55}")
print(f"DSv3-16B (EP=8) TPS Estimate")
print(f"  MoE ({N_MOE} layers):    {moe_step*1000:.0f} ms")
print(f"  Dense (1 layer):        {dense_step_s*1000:.0f} ms")
print(f"  Attention (27 layers):  {attn_step_s*1000:.0f} ms")
print(f"  Other:                  ~{other_s*1000:.0f} ms")
print(f"  Total step:             ~{total_s*1000:.0f} ms")
print(f"  TPS: {tps:,.0f}")
