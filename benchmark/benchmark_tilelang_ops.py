"""
benchmark/benchmark_tilelang_ops.py — Tile-lang 后端 vs CUDA 后端性能对比

覆盖 mixed_gemm / fuse_moe / swiglu（FA 的对比见 benchmark/fa_tilelang.py）。
每个 shape 输出：
  - CUDA 后端（ops.<op>(..., backend="cuda")）耗时
  - Tile-lang 后端多 config sweep 择优耗时
  - 加速比（CUDA / Tile-lang）与交叉正确性校验

suite：
  regular — 常规对齐 shape（默认）
  ragged  — ragged 场景：mixed_gemm 非 64/128 对齐 M/N/K（tile 边界 ±1）、
            fuse_moe 倾斜/zipf 路由（token 不均匀分布）、swiglu ragged M
  all     — 两者全跑

运行：
    python benchmark/benchmark_tilelang_ops.py                        # regular
    python benchmark/benchmark_tilelang_ops.py --suite ragged
    python benchmark/benchmark_tilelang_ops.py --ops mixed_gemm fuse_moe
    python benchmark/benchmark_tilelang_ops.py --warmup 20 --iters 100
"""

import argparse
import os
import sys

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops, split_mixed_precision_weight  # noqa: E402
from custom_ops import tilelang_ops  # noqa: E402
from custom_ops.tilelang_ops import _fuse_moe, _mixed_gemm, _swiglu  # noqa: E402


def bench_us(fn, warmup, iters):
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2] * 1e3


# ── mixed_gemm ───────────────────────────────────────────────────────────────

MIXED_GEMM_SHAPES = [
    (16, 4096, 4096),
    (64, 4096, 4096),
    (256, 4096, 4096),
    (1024, 4096, 4096),
    (4096, 4096, 4096),
    (512, 4096, 256),
    (128, 1000, 2048),
]

# ragged：围绕 tile 边界 64/128 的 ±1 位置（同 tests/test_mixed_gemm_ragged
# 的语义）；K 约束 %16==0（CUDA 版 TMA 路径契约）
MIXED_GEMM_RAGGED_SHAPES = [
    (127,  999, 2048),   # M/N 双 ragged（128/64 边界 -1）
    (513, 4096,  256),   # M ragged（512+1）
    (128, 4097, 2048),   # N ragged（4096+1）
    (256, 2048, 2000),   # K ragged（%16==0 但非 64 倍数）
    (129, 1000, 2064),   # M/N/K 全 ragged
    (1024, 1000, 192),   # 深 skinny + ragged N
]


def bench_mixed_gemm(args, shapes=None):
    shapes = shapes if shapes is not None else MIXED_GEMM_SHAPES
    print("=" * 96)
    print("mixed_gemm  (fp8 residual, silu) — CUDA vs Tile-lang")
    header = f"{'shape (M,N,K)':<22}{'CUDA µs':>10}{'TileLang µs':>12}{'CUDA/TL':>9}{'TL cfg':>22}{'check':>8}"
    print(header)
    print("-" * len(header))
    for (M, N, K) in shapes:
        x = torch.randn(M, K, device="cuda") * 0.5
        w = torch.randn(N, K, device="cuda") * 0.05
        w_high, w_low, w_scale = split_mixed_precision_weight(w, backend="fp8")
        try:
            t_cuda = bench_us(
                lambda: ops.mixed_gemm(x, w_high, w_low, None,
                                       activation="silu", backend="cuda"),
                args.warmup, args.iters)
        except Exception as e:
            print(f"{str((M, N, K)):<22} CUDA 不可用: {type(e).__name__}")
            continue

        best = (None, float("inf"))
        for cfg in _mixed_gemm.candidate_configs(M, N, K):
            try:
                t = bench_us(
                    lambda: tilelang_ops.mixed_gemm(
                        x, w_high, w_low, None, activation="silu", config=cfg),
                    args.warmup, args.iters)
            except Exception:
                continue
            if args.verbose:
                print(f"    {cfg} -> {t:.1f}µs")
            if t < best[1]:
                best = (cfg, t)
        cfg, t_tl = best

        y1 = ops.mixed_gemm(x, w_high, w_low, None, activation="silu",
                            backend="cuda")
        y2 = tilelang_ops.mixed_gemm(x, w_high, w_low, None,
                                     activation="silu", config=cfg)
        err = (y1.float() - y2.float()).abs().max().item()
        status = "OK" if err < 0.05 else f"DIFF={err:.3f}"
        cfg_str = (f"{cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']}"
                   f"/s{cfg['num_stages']}") if cfg else "-"
        print(f"{str((M, N, K)):<22}{t_cuda:>9.1f} {t_tl:>10.1f} "
              f"{t_cuda / t_tl:>8.2f}x {cfg_str:>22}{status:>8}")


# ── fuse_moe ─────────────────────────────────────────────────────────────────

MOE_SHAPES = [
    # (S, H, I, E, K)
    (128, 2048, 1024, 8, 2),
    (1024, 2048, 1024, 8, 2),
    (4096, 2048, 1024, 8, 2),
    (16384, 2048, 1024, 8, 2),
    (128, 2048, 1024, 64, 2),
    (4096, 2048, 1024, 64, 8),
    (4096, 4096, 1408, 8, 2),
]

# ragged：token 不均匀分布到 expert——skewed（90% 集中到 expert0）与
# zipf（p_e ∝ 1/(e+1)，长尾倾斜）；ragged S（非 2 的幂）
# (S, H, I, E, K, routing)
MOE_RAGGED_CASES = [
    (4096,  2048, 1024,  8, 2, "skewed"),
    (4096,  2048, 1024, 64, 2, "skewed"),
    (4096,  2048, 1024,  8, 2, "zipf"),
    (4096,  2048, 1024, 64, 2, "zipf"),
    (16384, 2048, 1024,  8, 2, "zipf"),
    (4097,  2048, 1024,  8, 2, "zipf"),   # ragged S
]


def make_moe_inputs(S, H, I, E, K, dtype=torch.bfloat16, seed=42,
                    routing="uniform"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, H, device="cuda", dtype=dtype, generator=g) * 0.3
    w1 = (torch.randn(E, 2 * I, H, device="cuda", dtype=dtype, generator=g) * 0.05)
    w2 = (torch.randn(E, H, I, device="cuda", dtype=dtype, generator=g) * 0.05)
    if routing == "uniform":
        ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32,
                            generator=g)
    elif routing == "skewed":       # 90% token 落到 expert 0
        u = torch.rand(S, K, device="cuda", generator=g)
        hot = torch.zeros(S, K, device="cuda", dtype=torch.int32)
        rest = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32,
                             generator=g)
        ids = torch.where(u < 0.9, hot, rest)
    else:                            # zipf：p_e ∝ 1/(e+1)，长尾倾斜
        w = 1.0 / torch.arange(1, E + 1, device="cuda", dtype=torch.float64)
        cdf = (w / w.sum()).cumsum(0)
        u = torch.rand(S, K, device="cuda", generator=g).double()
        ids = torch.searchsorted(cdf, u).clamp(max=E - 1).to(torch.int32)
    scale = torch.rand(S, K, device="cuda", generator=g)
    return x, w1, w2, ids, scale


def bench_fuse_moe(args, cases=None):
    cases = cases if cases is not None else MOE_SHAPES
    print("=" * 96)
    print("fuse_moe  (bf16) — CUDA vs Tile-lang")
    header = f"{'shape (S,H,I,E,K)':<24}{'CUDA µs':>10}{'TileLang µs':>12}{'CUDA/TL':>9}{'TL cfg':>24}{'check':>8}"
    print(header)
    print("-" * len(header))
    for case in cases:
        if len(case) == 6:
            S, H, I, E, K, routing = case
        else:
            (S, H, I, E, K), routing = case, "uniform"
        x, w1, w2, ids, scale = make_moe_inputs(S, H, I, E, K, routing=routing)
        try:
            t_cuda = bench_us(
                lambda: ops.fuse_moe(x, w1, w2, ids, scale, backend="cuda"),
                args.warmup, args.iters)
        except Exception as e:
            print(f"{str((S, H, I, E, K)):<24} CUDA 不可用: {type(e).__name__}")
            continue

        best = (None, float("inf"))
        for cfg in _fuse_moe.candidate_configs(S, K, E):
            try:
                t = bench_us(
                    lambda: tilelang_ops.fuse_moe(x, w1, w2, ids, scale,
                                                  config=cfg),
                    args.warmup, args.iters)
            except Exception:
                continue
            if args.verbose:
                print(f"    {cfg} -> {t:.1f}µs")
            if t < best[1]:
                best = (cfg, t)
        cfg, t_tl = best

        y1 = ops.fuse_moe(x, w1, w2, ids, scale, backend="cuda")
        y2 = tilelang_ops.fuse_moe(x, w1, w2, ids, scale, config=cfg)
        err = (y1.float() - y2.float()).abs().max().item()
        status = "OK" if err < 0.05 else f"DIFF={err:.3f}"
        cfg_str = (f"BM{cfg['block_M']}/N1-{cfg['block_N1']}/N2-{cfg['block_N2']}"
                   f"/s{cfg['num_stages1']}{cfg['num_stages2']}") if cfg else "-"
        print(f"{str((S, H, I, E, K)) + '/' + routing:<24}{t_cuda:>9.1f} {t_tl:>10.1f} "
              f"{t_cuda / t_tl:>8.2f}x {cfg_str:>24}{status:>8}")


# ── swiglu ───────────────────────────────────────────────────────────────────

SWIGLU_SHAPES = [
    (128, 4096, 4096),
    (1024, 2048, 4096),
    (4096, 4096, 4096),
    (16384, 2048, 4096),
]

# ragged：任意 M（N%64==0、K%64==0 是算子契约，不可 ragged）
SWIGLU_RAGGED_SHAPES = [
    (1023, 2048, 4096),   # 1024-1
    (4097, 2048, 4096),   # 4096+1
    (1000, 4096, 4096),   # 非对齐基数
]


def bench_swiglu(args, shapes=None):
    shapes = shapes if shapes is not None else SWIGLU_SHAPES
    print("=" * 96)
    print("swiglu  (bf16) — CUDA vs Tile-lang")
    header = f"{'shape (M,N,K)':<22}{'CUDA µs':>10}{'TileLang µs':>12}{'CUDA/TL':>9}{'TL cfg':>22}{'check':>8}"
    print(header)
    print("-" * len(header))
    for (M, N, K) in shapes:
        x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5)
        w = (torch.randn(2 * N, K, device="cuda", dtype=torch.bfloat16) * 0.05)
        try:
            t_cuda = bench_us(lambda: ops.swiglu(x, w, backend="cuda"),
                              args.warmup, args.iters)
        except Exception as e:
            print(f"{str((M, N, K)):<22} CUDA 不可用: {type(e).__name__}")
            continue

        best = (None, float("inf"))
        for cfg in _swiglu.candidate_configs(M, N, K):
            try:
                t = bench_us(lambda: tilelang_ops.swiglu(x, w, config=cfg),
                             args.warmup, args.iters)
            except Exception:
                continue
            if args.verbose:
                print(f"    {cfg} -> {t:.1f}µs")
            if t < best[1]:
                best = (cfg, t)
        cfg, t_tl = best

        y1 = ops.swiglu(x, w, backend="cuda")
        y2 = tilelang_ops.swiglu(x, w, config=cfg)
        err = (y1.float() - y2.float()).abs().max().item()
        status = "OK" if err < 0.05 else f"DIFF={err:.3f}"
        cfg_str = (f"{cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']}"
                   f"/s{cfg['num_stages']}"
                   + ("/FR" if cfg.get("fullrow") else "")) if cfg else "-"
        print(f"{str((M, N, K)):<22}{t_cuda:>9.1f} {t_tl:>10.1f} "
              f"{t_cuda / t_tl:>8.2f}x {cfg_str:>22}{status:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+",
                        choices=["mixed_gemm", "fuse_moe", "swiglu"],
                        default=["mixed_gemm", "fuse_moe", "swiglu"])
    parser.add_argument("--suite", choices=["regular", "ragged", "all"],
                        default="regular")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false",
                        default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用。")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"cap: {torch.cuda.get_device_capability(0)}  "
          f"PyTorch: {torch.__version__}  suite: {args.suite}")

    run = args.suite in ("regular", "all")
    ragged = args.suite in ("ragged", "all")
    if "mixed_gemm" in args.ops:
        if run:
            bench_mixed_gemm(args)
        if ragged:
            bench_mixed_gemm(args, MIXED_GEMM_RAGGED_SHAPES)
    if "fuse_moe" in args.ops:
        if run:
            bench_fuse_moe(args)
        if ragged:
            bench_fuse_moe(args, MOE_RAGGED_CASES)
    if "swiglu" in args.ops:
        if run:
            bench_swiglu(args)
        if ragged:
            bench_swiglu(args, SWIGLU_RAGGED_SHAPES)


if __name__ == "__main__":
    main()
