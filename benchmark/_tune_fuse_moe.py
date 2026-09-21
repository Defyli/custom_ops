"""
benchmark/_tune_fuse_moe.py — fuse_moe Tile-lang 结构调优实验

对照两个 gemm2 归约策略：
  - atomic：当前包内实现——gemm2 epilogue 逐元素 fp32 原子加到 Yacc，
    额外成本 = S×H zeros + S×K×H 次原子 + fp32→bf16 cast
  - mat（物化，与 CUDA 版同构）：gemm2 直写 Down (rows, H) bf16 +
    route 记录 PosMap (S,K) + 独立 reduce kernel 加权求和

附加底层 API knobs：
  - T.copy(eviction_policy)：W1/W2 evict_last（权重复用）/ A evict_first
  - T.gemm(policy=FullRow/Square)
  - atomic_addx4：gemm2 epilogue 4 列向量化原子加（低改动对照）

运行：
    python benchmark/_tune_fuse_moe.py
"""

import argparse
import os
import sys

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

from custom_ops.tilelang_ops._fuse_moe import (  # noqa: E402
    _default_config, _moe_count_factory, _moe_gather_factory,
    _moe_gemm1_factory, _moe_route_factory,
)

_CACHE = {}


def _make(key, factory, *args, **kwargs):
    if key in _CACHE:
        return _CACHE[key]
    k = factory(*args, **kwargs)
    _CACHE[key] = k
    return k


# ── route（+PosMap 记录每路由项的 compact 行号）─────────────────────────────

@tilelang.jit()
def _route_rowk_factory(S, num_topk, E, rows):
    @T.prim_func
    def main(
        Ids:          T.Tensor((S, num_topk), "int32"),
        Scale:        T.Tensor((S, num_topk), "float32"),
        Offsets:      T.Tensor((E + 1,), "int32"),
        Cursor:       T.Tensor((E,), "int32"),
        RowK:         T.Tensor((rows,), "int32"),
        SortedToken:  T.Tensor((rows,), "int32"),
        SortedWeight: T.Tensor((rows,), "float32"),
    ):
        with T.Kernel(T.ceildiv(S * num_topk, 128), threads=128) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < S * num_topk:
                    s = idx // num_topk
                    j = idx % num_topk
                    e = Ids[s, j]
                    pos = Offsets[e] + T.atomic_add(Cursor[e], 1,
                                                    return_prev=True)
                    RowK[pos] = j
                    SortedToken[pos] = s
                    SortedWeight[pos] = Scale[s, j]
    return main


# ── gemm2 物化版：直写 Down2 (S*K, H)，无原子、无 epilogue 乘法 ────────────
# 行号 = tok*K + rowk（每个 (s,k) 路由项恰好写一行）→ reduce 端可
# 纯仿射顺序读，避免随机行 gather 的 uncoalesced 访问

@tilelang.jit()
def _gemm2_mat_factory(E, I, H, S, num_topk, rows, grid_m, dtype,
                       block_M=128, block_N=128, block_K=64,
                       num_stages=2, threads=256,
                       w_evict=False, a_evict=False, fullrow=False):

    @T.prim_func
    def main(
        Offsets:     T.Tensor((E + 1,), "int32"),
        SortedToken: T.Tensor((rows,), "int32"),
        RowK:        T.Tensor((rows,), "int32"),
        Act:         T.Tensor((rows, I), dtype),
        W2:          T.Tensor((E, H, I), dtype),
        Down2:       T.Tensor((S * num_topk, H), dtype),
    ):
        with T.Kernel(grid_m, T.ceildiv(H, block_N),
                      threads=threads) as (bx, by):
            # 越界块的 gemm 用 expert0 权重算垃圾（行被谓词丢弃，不写出）
            esh = T.alloc_shared((1,), "int32")
            esh[0] = -1
            for e in T.serial(E):
                if Offsets[e] <= bx * block_M:
                    if bx * block_M < Offsets[e + 1]:
                        esh[0] = e
            e = esh[0]
            ee = T.if_then_else(e >= 0, e, 0)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            acc = T.alloc_fragment((block_M, block_N), "float32")
            tok = T.alloc_fragment((block_M,), "int32")
            rowk = T.alloc_fragment((block_M,), "int32")
            T.clear(acc)
            for i in T.Parallel(block_M):
                tok[i] = SortedToken[bx * block_M + i]
                rowk[i] = RowK[bx * block_M + i]
            for k in T.Pipelined(T.ceildiv(I, block_K),
                                 num_stages=num_stages):
                if a_evict:
                    T.copy(Act[bx * block_M:(bx + 1) * block_M,
                               k * block_K:(k + 1) * block_K], A_shared,
                           eviction_policy="evict_first")
                else:
                    T.copy(Act[bx * block_M:(bx + 1) * block_M,
                               k * block_K:(k + 1) * block_K], A_shared)
                if w_evict:
                    T.copy(W2[ee, by * block_N:(by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], W_shared,
                           eviction_policy="evict_last")
                else:
                    T.copy(W2[ee, by * block_N:(by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], W_shared)
                if fullrow:
                    T.gemm(A_shared, W_shared, acc, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(A_shared, W_shared, acc, transpose_B=True)
            # epilogue：写 Down2[tok*K+rowk, c]——行内 c 连续，coalesced
            for i, j in T.Parallel(block_M, block_N):
                if tok[i] >= 0:
                    c = by * block_N + j
                    if c < H:
                        Down2[tok[i] * num_topk + rowk[i], c] = acc[i, j]
    return main


# ── reduce kernel：y[s] = Σ_k Scale[s,k]·Down[PosMap[s,k]] ──────────────────

@tilelang.jit()
def _reduce_factory(S, num_topk, H, dtype, block_S=4):
    @T.prim_func
    def main(
        Scale: T.Tensor((S, num_topk), "float32"),
        Down2: T.Tensor((S * num_topk, H), dtype),
        Y:     T.Tensor((S, H), dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), T.ceildiv(H, 256),
                      threads=256) as (bx, by):
            # Down2[s*K+k, c] 为仿射访问（顺序读可向量化）；
            # 行 serial、列 parallel 保证 warp 内 c 连续 coalesced
            acc = T.alloc_fragment((1,), "float32")
            for i in T.serial(block_S):
                for j in T.Parallel(256):
                    s = bx * block_S + i
                    c = by * 256 + j
                    if s < S and c < H:
                        acc[0] = 0.0
                        for k in T.serial(num_topk):
                            acc[0] = acc[0] + \
                                T.cast(Down2[s * num_topk + k, c],
                                       "float32") * Scale[s, k]
                        Y[s, c] = acc[0]
    return main


# ── 端到端实现 ───────────────────────────────────────────────────────────────

def moe_mat(x, w1, w2, topk_ids, topk_scale, cfg=None,
            w_evict=False, a_evict=False, fullrow=False):
    """物化 + reduce 版 fuse_moe（6 kernel）。"""
    S, H = x.shape
    E = w1.shape[0]
    I = w1.shape[1] // 2
    num_topk = topk_ids.shape[1]
    dtype = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[x.dtype]
    cfg = cfg or _default_config(S, num_topk, E)
    BM = cfg["block_M"]
    em_max = S * num_topk + E * (BM - 1)
    grid_m = (em_max + BM - 1) // BM
    rows = grid_m * BM

    dev = x.device
    offsets = torch.empty(E + 1, dtype=torch.int32, device=dev)
    cursor = torch.zeros(E, dtype=torch.int32, device=dev)
    rowk = torch.zeros(rows, dtype=torch.int32, device=dev)
    sorted_token = torch.full((rows,), -1, dtype=torch.int32, device=dev)
    sorted_weight = torch.zeros(rows, dtype=torch.float32, device=dev)
    xc = torch.empty((rows, H), dtype=x.dtype, device=dev)
    act = torch.empty((rows, I), dtype=x.dtype, device=dev)
    down2 = torch.empty((S * num_topk, H), dtype=x.dtype, device=dev)
    y = torch.empty((S, H), dtype=x.dtype, device=dev)

    k1 = _make(("cnt", S, num_topk, E, BM), _moe_count_factory,
               S, num_topk, E, BM)
    k2 = _make(("route_rowk", S, num_topk, E, rows), _route_rowk_factory,
               S, num_topk, E, rows)
    k3 = _make(("gather", S, H, rows, BM, dtype), _moe_gather_factory,
               S, H, rows, BM, dtype)
    k4 = _make(("gemm1", E, I, H, rows, grid_m, dtype,
                cfg["block_M"], cfg["block_N1"], cfg["block_K1"],
                cfg["num_stages1"], cfg["threads"]),
               _moe_gemm1_factory, E, I, H, rows, grid_m, dtype,
               block_M=cfg["block_M"], block_N=cfg["block_N1"],
               block_K=cfg["block_K1"], num_stages=cfg["num_stages1"],
               threads=cfg["threads"])
    k5 = _make(("gemm2mat", E, I, H, S, num_topk, rows, grid_m, dtype,
                cfg["block_M"], cfg["block_N2"], cfg["block_K2"],
                cfg["num_stages2"], cfg["threads"],
                w_evict, a_evict, fullrow),
               _gemm2_mat_factory, E, I, H, S, num_topk, rows, grid_m, dtype,
               block_M=cfg["block_M"], block_N=cfg["block_N2"],
               block_K=cfg["block_K2"], num_stages=cfg["num_stages2"],
               threads=cfg["threads"],
               w_evict=w_evict, a_evict=a_evict, fullrow=fullrow)
    k6 = _make(("reduce", S, num_topk, H, dtype), _reduce_factory,
               S, num_topk, H, dtype)

    k1(topk_ids, offsets)
    k2(topk_ids, topk_scale, offsets, cursor, rowk,
       sorted_token, sorted_weight)
    k3(sorted_token, x, xc)
    k4(offsets, xc, w1, act)
    k5(offsets, sorted_token, rowk, act, w2, down2)
    k6(topk_scale, down2, y)
    return y


# ── 基准 ─────────────────────────────────────────────────────────────────────

def bench_us(fn, warmup=10, iters=30):
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


def make_inputs(S, H, I, E, K, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, H, device="cuda", dtype=torch.bfloat16,
                    generator=g) * 0.3
    w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16,
                     generator=g) * 0.05
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16,
                     generator=g) * 0.05
    ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32,
                        generator=g)
    scale = torch.rand(S, K, device="cuda", generator=g)
    return x, w1, w2, ids, scale


def moe_ref(x, w1, w2, ids, scale):
    S, K = ids.shape
    I = w1.shape[1] // 2
    y = torch.zeros(x.shape[0], x.shape[1], device=x.device,
                    dtype=torch.float32)
    for s in range(S):
        for j in range(K):
            e = int(ids[s, j])
            gu = x[s].float() @ w1[e].float().t()
            act = gu[:I] / (1 + torch.exp(-gu[:I])) * gu[I:]
            y[s] += (act @ w2[e].float().t()) * scale[s, j]
    return y


SHAPES = {
    "s": (128, 2048, 1024, 8, 2),
    "m": (4096, 2048, 1024, 8, 2),
    "l": (16384, 2048, 1024, 8, 2),
}
CUDA_REF = {"s": 181.2, "m": 935.9, "l": 3225.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    from custom_ops.tilelang_ops._fuse_moe import fuse_moe_tilelang

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"tilelang: {tilelang.__version__}")
    for key in args.shapes:
        S, H, I, E, K = SHAPES[key]
        x, w1, w2, ids, scale = make_inputs(S, H, I, E, K)
        ref = None if S > 2048 else moe_ref(x, w1, w2, ids, scale)
        print("=" * 78)
        print(f"shape (S,H,I,E,K)={SHAPES[key]}   "
              f"[CUDA 参考 {CUDA_REF[key]}µs]")

        variants = [
            ("atomic(包内基线)",
             lambda: fuse_moe_tilelang(x, w1, w2, ids, scale)),
            ("mat(物化+reduce)",
             lambda: moe_mat(x, w1, w2, ids, scale)),
            ("mat+evict",
             lambda: moe_mat(x, w1, w2, ids, scale,
                             w_evict=True, a_evict=True)),
            ("mat+evict+FullRow",
             lambda: moe_mat(x, w1, w2, ids, scale,
                             w_evict=True, a_evict=True, fullrow=True)),
            ("mat+FullRow",
             lambda: moe_mat(x, w1, w2, ids, scale, fullrow=True)),
        ]
        # BM=64 tile 变体（S 大时）
        if S >= 4096:
            cfg64 = dict(_default_config(S, K, E), block_M=64)
            variants.append(("mat+FullRow/BM64",
                             lambda: moe_mat(x, w1, w2, ids, scale, cfg=cfg64,
                                             fullrow=True)))
            cfg128n = dict(_default_config(S, K, E), block_N2=64)
            variants.append(("mat+FullRow/BN2-64",
                             lambda: moe_mat(x, w1, w2, ids, scale,
                                             cfg=cfg128n, fullrow=True)))

        print(f"{'variant':<24}{'µs':>10}{'vs base':>10}{'CUDA/TL':>10}  check")
        print("-" * 78)
        t_base = None
        for name, fn in variants:
            try:
                y = fn()
                if ref is not None:
                    err = (y.float() - ref).abs().max().item()
                    ok = "OK" if err < 0.05 else f"BAD({err:.3f})"
                else:
                    # 大 shape 与基线互拍
                    y0 = fuse_moe_tilelang(x, w1, w2, ids, scale)
                    err = (y.float() - y0.float()).abs().max().item()
                    ok = "OK" if err < 0.05 else f"BAD({err:.3f})"
                t = bench_us(fn, iters=args.iters)
            except Exception as e:
                print(f"{name:<24}{'ERR':>10}  {type(e).__name__}: "
                      f"{str(e)[:70]}")
                continue
            if t_base is None:
                t_base = t
            print(f"{name:<24}{t:>9.1f} {t_base / t:>9.2f}x "
                  f"{CUDA_REF[key] / t:>9.2f}x  {ok}")


if __name__ == "__main__":
    main()
