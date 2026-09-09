"""fuse_moe (ours, CUDA C++) vs sglang fused_moe (Triton) 性能对比。

sglang 侧为 `benchmark/sglang_triton_moe/` 自包含移植（kernel 逐字提取自
sglang 源码，无需安装 sglang/sgl-kernel，无需 3rd/sglang checkout），
两种 config 形态：
  - default：查表 miss → get_default_config 启发式
  - tuned  ：5090D tuned JSON（benchmark/tune_sglang_moe.py 生成，查表命中）

数值校验：三方对 fp32 参考（torch 循环）误差均在 bf16 噪声内。

用法：
    python benchmark/benchmark_fuse_moe_vs_sglang.py
    python benchmark/benchmark_fuse_moe_vs_sglang.py --S 512 4096 16384 --iters 50
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))  # custom_ops 父目录（import custom_ops）

import torch  # noqa: E402

from custom_ops import ops  # noqa: E402
from sglang_triton_moe import fused_experts  # noqa: E402


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # µs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, nargs="+", default=[512, 1024, 4096, 8192, 16384])
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--I", type=int, default=1024)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    E_, H_, I_ = args.E, args.H, args.I

    w1 = torch.randn(E_, 2 * I_, H_, device=dev, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E_, H_, I_, device=dev, dtype=torch.bfloat16) * 0.05

    print(f"{'S':>6} | {'ours µs':>9} {'ours TF':>8} | "
          f"{'sgl-def µs':>10} {'def TF':>7} | {'sgl-tun µs':>10} {'tun TF':>7} | "
          f"{'vs def':>6} {'vs tun':>6}")
    print("-" * 88)

    for S in args.S:
        x = torch.randn(S, H_, device=dev, dtype=torch.bfloat16)
        ids = torch.randint(0, E_, (S, args.K), device=dev, dtype=torch.int32)
        wt = torch.rand(S, args.K, device=dev, dtype=torch.float32)

        t_ours = bench(lambda: ops.fuse_moe(x, w1, w2, ids, wt), args.iters, args.warmup)
        # default 启发式（查表 miss；首次调用含 Triton JIT，不计时）
        fused_experts(x, w1, w2, wt, ids, use_tuned_config=False)
        t_def = bench(lambda: fused_experts(x, w1, w2, wt, ids,
                                            use_tuned_config=False),
                      args.iters, args.warmup)
        t_tun = bench(lambda: fused_experts(x, w1, w2, wt, ids),
                      args.iters, args.warmup)

        flops = 2 * S * args.K * (2 * I_ * H_ + I_ * H_)  # up(N=2I) + down(N=H)
        def tf(us):
            return flops / (us * 1e-6) / 1e12
        print(f"{S:>6} | {t_ours:9.1f} {tf(t_ours):8.1f} | "
              f"{t_def:10.1f} {tf(t_def):7.1f} | {t_tun:10.1f} {tf(t_tun):7.1f} | "
              f"{t_def / t_ours:5.2f}x {t_tun / t_ours:5.2f}x")

    # 数值抽检（首个形状，三方 vs fp32 参考）
    S = args.S[0]
    x = torch.randn(S, H_, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E_, (S, args.K), device=dev, dtype=torch.int32)
    wt = torch.rand(S, args.K, device=dev, dtype=torch.float32)
    ref = torch.zeros(S, H_, device=dev, dtype=torch.float32)
    for k in range(args.K):
        for e in range(E_):
            sel = (ids[:, k] == e)
            if sel.sum() == 0:
                continue
            xe = x[sel].float()
            g = xe @ w1[e, :I_].float().t()
            u = xe @ w1[e, I_:].float().t()
            ref[sel] += (torch.nn.functional.silu(g) * u) @ w2[e].float().t() * wt[sel, k:k+1]
    y_ours = ops.fuse_moe(x, w1, w2, ids, wt).float()
    y_sgl = fused_experts(x, w1, w2, wt, ids).float()
    print(f"\n[check] S={S}: ours|max={((y_ours - ref).abs().max()).item():.4f} "
          f"sgl|max={((y_sgl - ref).abs().max()).item():.4f} (bf16 noise expected)")


if __name__ == "__main__":
    main()
