"""Stage timing baseline: FUSE_MOE_TIME=1 for key shapes, 3 timed runs each."""
import os, sys
sys.path.insert(0, "/home/lifengyi06")
import time
import torch
from custom_ops import ops

torch.manual_seed(0)

SHAPES = [
    (512, 2048, 1024, 8, 2),
    (1024, 2048, 1024, 8, 2),
    (4096, 2048, 1024, 8, 2),
    (16384, 2048, 1024, 8, 2),
    (4096, 4096, 1408, 8, 2),
    (4096, 2048, 1024, 64, 8),
]

for S, H, I, E, K in SHAPES:
    x = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.05
    ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32)
    wt = torch.rand(S, K, device="cuda", dtype=torch.float32)
    for _ in range(10):
        ops.fuse_moe(x, w1, w2, ids, wt)
    torch.cuda.synchronize()
    for r in range(3):
        ops.fuse_moe(x, w1, w2, ids, wt)
    torch.cuda.synchronize()
    del x, w1, w2, ids, wt
    torch.cuda.empty_cache()
    time.sleep(3)

print("[stage-time] done")
