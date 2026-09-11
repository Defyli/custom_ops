import sys

import torch

sys.path.insert(0, "/home/lifengyi06")
from custom_ops import ops


def bench(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def eager(x, w):
    n = w.shape[0] // 2
    gu = x @ w.t()
    return torch.nn.functional.silu(gu[:, :n]) * gu[:, n:]


print("warming up clocks...")
dummy = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(200):
    dummy = dummy @ dummy
torch.cuda.synchronize()
del dummy
torch.cuda.empty_cache()
print("warm done")

torch.manual_seed(0)
shapes = [
    (1, 4096, 11008),
    (64, 4096, 11008),
    (96, 4096, 11008),
    (128, 4096, 11008),
    (512, 4096, 11008),
    (2048, 4096, 11008),
    (4096, 4096, 11008),
    (8192, 4096, 11008),
    (2048, 2048, 1024),
    (2048, 4096, 1024),
    (2048, 4096, 2048),
    (2048, 4096, 2816),
    (2048, 4096, 4096),
]
print("%6s %6s %6s | %9s | %13s | %s" % ("M", "K", "N", "eager", "auto-policy", "vs eager"))
for (M, K, N) in shapes:
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(2 * N, K, device="cuda") * 0.02).to(torch.bfloat16)
    t_e = bench(lambda: eager(x, w))
    t = bench(lambda: ops.swiglu(x, w))
    tf = 4 * M * N * K / t / 1e9
    print("%6d %6d %6d | %7.3fms | %7.3fms %4.0fTF | %5.2fx" % (M, K, N, t_e, t, tf, t_e / t))
    del x, w
    torch.cuda.empty_cache()
print("done")
