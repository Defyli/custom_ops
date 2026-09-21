"""
benchmark/_tune_mixed_gemm.py — mixed_gemm Tile-lang 底层 API 调优实验

逐项验证 tile-lang 底层 API 对 mixed_gemm 的性能影响：
  - T.gemm(policy=Square/FullRow/FullCol)     warp 调度策略
  - T.copy(eviction_policy=evict_first/last)  L2 驱逐提示（X first / W last）
  - T.use_swizzle(panel_size)                 grid 块调度 swizzle（L2 复用）
  - threads / tile 形状 / num_stages
  - PassConfigKey.TL_ENABLE_VECTORIZE_256     256-bit epilogue store

运行：
    python benchmark/_tune_mixed_gemm.py            # 全部 shape
    python benchmark/_tune_mixed_gemm.py --shapes a b
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


def build_kernel(M, N, K, x_dtype="float32", wlow_dtype="float8_e4m3",
                 out_dtype="float32", act_id=1,
                 block_M=128, block_N=128, block_K=64,
                 num_stages=2, threads=256,
                 policy="square", evict=False, swizzle=0, v256=False):
    """带底层 API knobs 的 mixed_gemm kernel（与包内基线同数值语义）。"""
    pc = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    if v256:
        pc[tilelang.PassConfigKey.TL_ENABLE_VECTORIZE_256] = True

    @tilelang.jit(out_idx=[6], pass_configs=pc)
    def factory(M, N, K, x_dtype, wlow_dtype, out_dtype, act_id,
                block_M, block_N, block_K, num_stages, threads):
        @T.prim_func
        def main(
            X:      T.Tensor((M, K), x_dtype),
            W_high: T.Tensor((N, K), "bfloat16"),
            W_low:  T.Tensor((N, K), wlow_dtype),
            Wscale: T.Tensor((N,), "float32"),
            Bias:   T.Tensor((N,), "float32"),
            scale:  T.float32,
            Y:      T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N),
                          threads=threads) as (bx, by):
                if swizzle:
                    T.use_swizzle(panel_size=swizzle)
                X_shared = T.alloc_shared((block_M, block_K), "bfloat16")
                W_shared = T.alloc_shared((block_N, block_K), "bfloat16")
                Wl_shared = T.alloc_shared((block_N, block_K), "bfloat16")
                acc = T.alloc_fragment((block_M, block_N), "float32")
                acc_res = T.alloc_fragment((block_M, block_N), "float32")
                wsc = T.alloc_fragment((block_N,), "float32")
                bias = T.alloc_fragment((block_N,), "float32")
                for j in T.Parallel(block_N):
                    jj = T.min(by * block_N + j, N - 1)
                    wsc[j] = Wscale[jj]
                    bias[j] = Bias[jj]
                T.clear(acc)
                T.clear(acc_res)
                for k in T.Pipelined(T.ceildiv(K, block_K),
                                     num_stages=num_stages):
                    if evict:
                        T.copy(X[bx * block_M:(bx + 1) * block_M,
                                 k * block_K:(k + 1) * block_K], X_shared,
                               eviction_policy="evict_first")
                        T.copy(W_high[by * block_N:(by + 1) * block_N,
                                      k * block_K:(k + 1) * block_K], W_shared,
                               eviction_policy="evict_last")
                        T.copy(W_low[by * block_N:(by + 1) * block_N,
                                     k * block_K:(k + 1) * block_K], Wl_shared,
                               eviction_policy="evict_last")
                    else:
                        T.copy(X[bx * block_M:(bx + 1) * block_M,
                                 k * block_K:(k + 1) * block_K], X_shared)
                        T.copy(W_high[by * block_N:(by + 1) * block_N,
                                      k * block_K:(k + 1) * block_K], W_shared)
                        T.copy(W_low[by * block_N:(by + 1) * block_N,
                                     k * block_K:(k + 1) * block_K], Wl_shared)
                    T.gemm(X_shared, W_shared, acc, transpose_B=True,
                           policy=_POL[policy])
                    T.gemm(X_shared, Wl_shared, acc_res, transpose_B=True,
                           policy=_POL[policy])
                for i, j in T.Parallel(block_M, block_N):
                    v = acc[i, j] + acc_res[i, j] * (wsc[j] * scale) + bias[j]
                    if act_id == 1:
                        acc[i, j] = v / (1.0 + T.exp(-v))
                    elif act_id == 2:
                        t = T.tanh(0.7978845608028654 *
                                   (v + 0.044715 * v * v * v))
                        acc[i, j] = 0.5 * v * (1.0 + t)
                    else:
                        acc[i, j] = v
                T.copy(acc, Y[bx * block_M:(bx + 1) * block_M,
                              by * block_N:(by + 1) * block_N])
        return main

    return factory(M, N, K, x_dtype, wlow_dtype, out_dtype, act_id,
                   block_M, block_N, block_K, num_stages, threads)


# ── 基准工具 ─────────────────────────────────────────────────────────────────

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


def make_inputs(M, N, K):
    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda") * 0.5
    w = torch.randn(N, K, device="cuda") * 0.05
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) * 256.0).clamp(-448, 448)
    w_low = w_low.to(torch.float8_e4m3fn)
    ones = torch.ones(N, device="cuda")
    zeros = torch.zeros(N, device="cuda")
    w_rec = w_high.float() + w_low.float() / 256.0
    ref = x @ w_rec.t()
    ref = ref / (1.0 + torch.exp(-ref))
    return x, w_high, w_low, ones, zeros, ref


# ── 变体定义 ─────────────────────────────────────────────────────────────────
# 每个 shape：(M, N, K), CUDA 参考耗时(µs), 基线 block_M, 额外 tile 变体

SHAPES = {
    "a": ((128, 1000, 2048), 30.9, 128),
    "b": ((512, 4096, 256), 32.8, 128),
    "c": ((1024, 4096, 4096), 557.1, 128),
    "d": ((4096, 4096, 4096), 1879.4, 128),
}


def variants(bm):
    """(名称, kwargs) 列表：kwargs 覆盖 build_kernel 的 knobs。"""
    return [
        ("base", {}),
        ("evict(X1st/Wlast)", dict(evict=True)),
        ("policy=FullRow", dict(policy="fullrow")),
        ("policy=FullCol", dict(policy="fullcol")),
        ("swizzle=64", dict(swizzle=64)),
        ("v256", dict(v256=True)),
        ("th128/s3", dict(threads=128, num_stages=3)),
        ("combo", dict(evict=True, policy="fullrow", swizzle=64, v256=True)),
    ]


EXTRA_TILES = {
    "a": [("BM64/BN64", dict(block_M=64, block_N=64, num_stages=3)),
          ("BM64/BN128", dict(block_M=64, block_N=128)),
          ("BM32/BN64", dict(block_M=32, block_N=64, num_stages=3)),
          ("BM64/BN128+evict", dict(block_M=64, block_N=128, evict=True))],
    "b": [("BM64/BN64", dict(block_M=64, block_N=64, num_stages=3)),
          ("BM64/BN128", dict(block_M=64, block_N=128)),
          ("BM64/BN64+swz", dict(block_M=64, block_N=64, num_stages=3,
                                swizzle=64)),
          ("BN256/BK32", dict(block_N=256, block_K=32, num_stages=2)),
          ("BN64/s4", dict(block_N=64, num_stages=4)),
          ("BK128/th128", dict(block_K=128, threads=128, num_stages=2))],
    "c": [("BM64/BN128", dict(block_M=64, block_N=128)),
          ("BK32/s4", dict(block_K=32, num_stages=4))],
    "d": [("BK32/s4", dict(block_K=32, num_stages=4)),
          ("BN256/BK32", dict(block_N=256, block_K=32))],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"tilelang: {tilelang.__version__}")
    for key in args.shapes:
        (M, N, K), t_cuda, bm0 = SHAPES[key]
        print("=" * 78)
        print(f"shape {M}x{N}x{K}   [CUDA 参考 {t_cuda}µs]")
        x, w_high, w_low, ones, zeros, ref = make_inputs(M, N, K)

        rows = variants(bm0) + EXTRA_TILES.get(key, [])
        # 对 tile 变体也叠加 evict 的组合
        if key in EXTRA_TILES:
            for name, kw in EXTRA_TILES[key]:
                rows.append((name + "+evict", dict(kw, evict=True)))

        print(f"{'variant':<28}{'µs':>10}{'vs base':>10}{'CUDA/TL':>10}  check")
        print("-" * 78)
        t_base = None
        for name, kw in rows:
            try:
                ker = build_kernel(M, N, K, **kw)
                y = ker(x, w_high, w_low, ones, zeros, 1.0 / 256.0)
                err = (y.float() - ref).abs().max().item()
                t = bench_us(lambda: ker(x, w_high, w_low, ones, zeros,
                                         1.0 / 256.0), iters=args.iters)
            except Exception as e:
                print(f"{name:<28}{'ERR':>10}  {type(e).__name__}: "
                      f"{str(e)[:60]}")
                continue
            if t_base is None:
                t_base = t
            ok = "OK" if err < 0.05 else f"BAD({err:.3f})"
            print(f"{name:<28}{t:>9.1f} {t_base / t:>9.2f}x "
                  f"{t_cuda / t:>9.2f}x  {ok}")


if __name__ == "__main__":
    main()
