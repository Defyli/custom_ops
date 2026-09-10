"""ncu probe: S=16384 steady state for stall attribution."""
import os, sys
sys.path.insert(0, "/home/lifengyi06")
import torch
from custom_ops import ops
torch.manual_seed(0)
S,H,I,E,K = 16384, 2048, 1024, 8, 2
x = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(E, 2*I, H, device="cuda", dtype=torch.bfloat16) * 0.05
w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.05
ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32)
wt = torch.rand(S, K, device="cuda", dtype=torch.float32)
for _ in range(5):
    ops.fuse_moe(x, w1, w2, ids, wt)
torch.cuda.synchronize()
print("[probe] done")
