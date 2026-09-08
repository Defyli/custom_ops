import sys
sys.path.insert(0, "/home/lifengyi06")
import torch
from custom_ops import ops

torch.manual_seed(0)
dev = "cuda"
S, H, I, E, K = 128, 128, 64, 8, 2
x = torch.randn(S, H, device=dev, dtype=torch.bfloat16)
w1 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.05
w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.05
ids = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
sc = torch.rand(S, K, device=dev, dtype=torch.float32)
out = ops.fuse_moe(x, w1, w2, ids, sc)
torch.cuda.synchronize()
print("DONE", out.shape, out.float().abs().mean().item())
