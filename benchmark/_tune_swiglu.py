"""
benchmark/_tune_swiglu.py — swiglu Tile-lang 底层 API 调优实验

重点变体：block_N 扩大 / gemm warp policy / BM64 / stages / threads

运行：
    python benchmark/_tune_swiglu.py
"""

import argparse
import os
import sys

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

_POL = {"square": T.GemmWarpPolicy.Square,
        "fullrow": T.GemmWarpPolicy.FullRow,
        "fullcol": T.GemmWarpPolicy.FullCol}


def build_kernel(M, N, K, dtype="bfloat16",
                 block_M=128, block_N=64, block_K=64,
                 num_stages=2, threads=256, policy="square"):
    @tilelang.jit(
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def factory(M, N, K, dtype, block_M, block_N, block_K, num_stages, threads):
        @T.prim_func
        def main(
            X: T.Tensor((M, K), dtype),
            W: T.Tensor((2 * N, K), dtype),
            Y: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N),
                          threads=threads) as (bx, by):
                X_shared = T.alloc_shared((block_M, block_K), dtype)
                Wg_shared = T.alloc_shared((block_N, block_K), dtype)
                Wu_shared = T.alloc_shared((block_N, block_K), dtype)
                acc_g = T.alloc_fragment((block_M, block_N), "float32")
                acc_u = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(acc_g)
                T.clear(acc_u)
                for k in T.Pipelined(T.ceildiv(K, block_K),
                                     num_stages=num_stages):
                    T.copy(X[bx * block_M:(bx + 1) * block_M,
                             k * block_K:(k + 1) * block_K], X_shared)
                    T.copy(W[by * block_N:(by + 1) * block_N,
                             k * block_K:(k + 1) * block_K], Wg_shared)
                    T.copy(W[N + by * block_N:N + (by + 1) * block_N,
                             k * block_K:(k + 1) * block_K], Wu_shared)
                    T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True,
                           policy=_POL[policy])
                    T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True,
                           policy=_POL[policy])
                for i, j in T.Parallel(block_M, block_N):
                    g = acc_g[i, j]
                    acc_g[i, j] = g / (1.0 + T.exp(-g)) * acc_u[i, j]
                T.copy(acc_g, Y[bx * block_M:(bx + 1) * block_M,
                                by * block_N:(by + 1) * block_N])
        return main

    return factory(M, N, K, dtype, block_M, block_N, block_K,
                   num_stages, threads)


def bench_us(fn, warmup=10, iters=50):
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


SHAPES = {
    # (M, N, K) : CUDA 参考 µs
    "a": ((1024, 2048, 4096), 300.8),
    "b": ((16384, 2048, 4096), 3754.0),
    "c": ((128, 4096, 4096), 122.9),
}


def variants():
    return [
        ("base(BM128/BN64/s2)", {}),
        ("BN128", dict(block_N=128)),
        ("BN128/s3", dict(block_N=128, num_stages=3)),
        ("BN256/BK32", dict(block_N=256, block_K=32)),
        ("policy=FullRow", dict(policy="fullrow")),
        ("policy=FullCol", dict(policy="fullcol")),
        ("th128", dict(threads=128)),
        ("BM64/BN128", dict(block_M=64, block_N=128)),
        ("BM64/BN64/s3", dict(block_M=64, block_N=64, num_stages=3)),
        ("BN128+FullRow", dict(block_N=128, policy="fullrow")),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"tilelang: {tilelang.__version__}")
    for key in args.shapes:
        (M, N, K), t_cuda = SHAPES[key]
        print("=" * 78)
        print(f"shape {M}x{N}x{K}   [CUDA 参考 {t_cuda}µs]")
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
        w = torch.randn(2 * N, K, device="cuda",
                        dtype=torch.bfloat16) * 0.05
        g = x @ w[:N].t()
        u = x @ w[N:].t()
        ref = (g / (1 + torch.exp(-g)) * u).float()

        print(f"{'variant':<24}{'µs':>10}{'vs base':>10}{'CUDA/TL':>10}  check")
        print("-" * 78)
        t_base = None
        for name, kw in variants():
            bm = kw.get("block_M", 128)
            if bm > M:
                continue
            try:
                ker = build_kernel(M, N, K, **kw)
                y = ker(x, w)
                err = (y.float() - ref).abs().max().item()
                t = bench_us(lambda: ker(x, w), iters=args.iters)
            except Exception as e:
                print(f"{name:<24}{'ERR':>10}  {type(e).__name__}: "
                      f"{str(e)[:60]}")
                continue
            if t_base is None:
                t_base = t
            ok = "OK" if err < 0.05 else f"BAD({err:.3f})"
            print(f"{name:<24}{t:>9.1f} {t_base / t:>9.2f}x "
                  f"{t_cuda / t:>9.2f}x  {ok}")


if __name__ == "__main__":
    main()
