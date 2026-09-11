"""_swiglu_sweep2.py - dense policy sweep (sched x tile x kstage)"""
import os
import sys

import torch

sys.path.insert(0, "/home/lifengyi06")
from custom_ops import ops


def bench(fn, iters=30, warmup=10):
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


def run(M, K, N, tiles, sched=None, kstage=None, tag=""):
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(2 * N, K, device="cuda") * 0.02).to(torch.bfloat16)
    t_e = bench(lambda: eager(x, w))
    if sched is None:
        os.environ.pop("SWIGLU_SCHED", None)
    else:
        os.environ["SWIGLU_SCHED"] = sched
    if kstage is None:
        os.environ.pop("SWIGLU_KSTAGE", None)
    else:
        os.environ["SWIGLU_KSTAGE"] = str(kstage)
    cells = []
    for tm in tiles:
        os.environ["SWIGLU_TILE_M"] = str(tm)
        t = bench(lambda: ops.swiglu(x, w))
        tf = 4 * M * N * K / t / 1e9
        cells.append("t%d %7.3fms %4.0fTF" % (tm, t, tf))
    os.environ.pop("SWIGLU_TILE_M", None)
    print("%5d %5d %5d %-14s | eager %7.3fms | %s" % (M, K, N, tag, t_e, " | ".join(cells)))
    del x, w
    torch.cuda.empty_cache()


torch.manual_seed(0)
K = 4096
print("== part1: big-W M scan (vert forced) ==")
for M in (1, 64, 96, 128, 512, 2048):
    run(M, K, 11008, (32, 64, 128), sched="vert", tag="vert")
print("== part2: sched crossover (M=2048, W scan) ==")
for N in (1024, 2048, 2816, 4096):
    run(2048, K, N, (64, 128), sched="horizon", tag="horizon")
    run(2048, K, N, (64, 128), sched="vert", tag="vert")
print("== part3: M=1 kstage ==")
run(1, K, 11008, (32, 64), sched="vert", kstage=0, tag="vert ks=deep")
run(1, K, 11008, (32, 64), sched="vert", kstage=2, tag="vert ks=2")
print("done")
