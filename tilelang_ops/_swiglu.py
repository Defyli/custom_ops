"""
custom_ops/tilelang_ops/_swiglu.py — Tile-lang 版 swiglu

与 CUDA 后端（csrc/swiglu/）同语义：

    y = silu(x @ Wg^T) * (x @ Wu^T)

  - x: (M, K) bf16/fp16；weight: (2N, K) 同 dtype（gate 行在前 up 在后）
  - 限制与 CUDA 版一致：K % 64 == 0、N % 64 == 0；M 任意；fp32 累加

单 kernel：同一 X tile 同时算 gate / up 两个 N 面板（X 装载量减半），
epilogue silu(gate)·up 直接写出，无 (M, 2N) 中间量物化。
"""

import torch

import tilelang
import tilelang.language as T

_KERNEL_CACHE = {}


@tilelang.jit(
    out_idx=[2],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _swiglu_factory(M, N, K, dtype,
                    block_M=128, block_N=64, block_K=64,
                    num_stages=2, threads=256, fullrow=False):
    """fullrow=True：T.gemm 用 GemmWarpPolicy.FullRow——每 warp 负责完整
    行带，实测在 BM128/BN128 大 tile 上比默认 Square 快 ~9%
    （(16384,2048,4096)：0.74x → 1.01x vs CUDA）。"""

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
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[bx * block_M:(bx + 1) * block_M,
                         k * block_K:(k + 1) * block_K], X_shared)
                T.copy(W[by * block_N:(by + 1) * block_N,
                         k * block_K:(k + 1) * block_K], Wg_shared)
                T.copy(W[N + by * block_N:N + (by + 1) * block_N,
                         k * block_K:(k + 1) * block_K], Wu_shared)
                if fullrow:
                    T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True)
                    T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True)
            for i, j in T.Parallel(block_M, block_N):
                g = acc_g[i, j]
                acc_g[i, j] = g / (1.0 + T.exp(-g)) * acc_u[i, j]
            T.copy(acc_g, Y[bx * block_M:(bx + 1) * block_M,
                            by * block_N:(by + 1) * block_N])
    return main


def _default_config(M, N, K):
    """实测分档（4090D）：
    - M ≤ 128：BM128/BN64/s2（小 M 下 BN128 面板反而负优化）
    - grid(BM128,BN64) < 1024：BM64/BN64/s3——(1024,2048,4096) 0.86→0.98x
    - 大 M：BM128/BN128/FullRow——(16384,2048,4096) 0.74→1.01x
    """
    if M <= 128:
        # BM 取 2 的幂档位（T.gemm 的 mma 要求 block_M % 16 == 0）
        bm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
        return dict(block_M=bm, block_N=64, block_K=64, num_stages=2,
                    threads=256, fullrow=False)
    grid = ((M + 127) // 128) * ((N + 63) // 64)
    if grid < 1024:
        return dict(block_M=64, block_N=64, block_K=64, num_stages=3,
                    threads=256, fullrow=False)
    return dict(block_M=128, block_N=128, block_K=64, num_stages=2,
                threads=256, fullrow=True)


def candidate_configs(M, N=2048, K=4096):
    base = _default_config(M, N, K)
    cfgs = [base,
            dict(base, num_stages=3),
            dict(base, block_M=max(16, base["block_M"] // 2)),
            dict(base, block_N=64 if base["block_N"] >= 128 else 128,
                 fullrow=False),
            dict(base, fullrow=not base.get("fullrow", False)),
            dict(base, block_N=64, num_stages=3, fullrow=False)]
    # 去重（保持顺序）
    seen, out = set(), []
    for c in cfgs:
        key = tuple(sorted((k, v) for k, v in c.items()
                           if k != "fullrow")) + (c.get("fullrow", False),)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def swiglu_tilelang(x, weight, config=None):
    """Tile-lang 版 swiglu 前端（签名与 ops.swiglu 对齐）。"""
    if x.dim() < 1:
        raise ValueError("x must have at least 1 dim")
    if weight.dim() != 2:
        raise ValueError(f"weight must be (2N, K) 2D, got {tuple(weight.shape)}")
    M = x.numel() // x.shape[-1]
    K = x.shape[-1]
    N2, Kw = weight.shape
    if N2 % 2 != 0 or Kw != K:
        raise ValueError(f"shape mismatch: x(*,{K}) vs weight{tuple(weight.shape)}")
    N = N2 // 2
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x 必须是 bf16 / fp16")
    if weight.dtype != x.dtype:
        raise ValueError("weight 必须与 x 同 dtype")
    if K % 64 != 0 or N % 64 != 0:
        raise ValueError("K % 64 == 0 且 N % 64 == 0（对齐契约）")

    dtype = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[x.dtype]
    cfg = config or _default_config(M, N, K)
    key = ("swiglu", M, N, K, dtype, cfg["block_M"], cfg["block_N"],
           cfg["block_K"], cfg["num_stages"], cfg["threads"],
           cfg.get("fullrow", False))
    if key in _KERNEL_CACHE:
        kernel = _KERNEL_CACHE[key]
    else:
        kernel = _swiglu_factory(
            M, N, K, dtype,
            block_M=cfg["block_M"], block_N=cfg["block_N"],
            block_K=cfg["block_K"], num_stages=cfg["num_stages"],
            threads=cfg["threads"], fullrow=cfg.get("fullrow", False))
        _KERNEL_CACHE[key] = kernel
    x2d = x.reshape(M, K)
    return kernel(x2d, weight)
