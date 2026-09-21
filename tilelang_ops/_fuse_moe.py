"""
custom_ops/tilelang_ops/_fuse_moe.py — Tile-lang 版 fuse_moe

与 CUDA 后端（csrc/fuse_moe/）同语义：

    y[s] = Σ_j topk_scale[s,j] · (Down_{e_j} @ silu(GateUp_{e_j} @ x[s]))

  - x: (S, H) bf16/fp16；gate_up_weight: (E, 2I, H)（gate 在前 up 在后）；
    down_weight: (E, H, I)；topk_ids: (S, K) int32；topk_scale: (S, K) fp32
  - 限制与 CUDA 版一致：H % 64 == 0、I % 64 == 0、K ≤ 128、E ≤ 512

实现为 5 个 tile-lang kernel + 1 个 torch cast：

  1. count/prefix（单 CTA）：expert 直方图（shared 原子）+ 按 block_M padded
     的 exclusive 前缀和 → Offsets (E+1,)
  2. route：每个 (s, j) 原子领取其 expert padded 段内槽位 →
     SortedToken / SortedWeight（compact、expert 有序）
  3. gather：按 SortedToken 把 x 行收集成 padded 布局 Xc (rows, H)
     （之后 gemm 可用仿射 T.copy 进异步流水；-1 哨兵行填 X[0]，其结果
      被 gemm2 的行守卫丢弃）
  4. gemm1（配对融合）：每 CTA 线性扫描 Offsets 定位 expert，同一 X tile
     同时算 gate / up 两个 N 面板，epilogue silu(gate)·up 直写 act_out
  5. gemm2：act @ Down[e]，epilogue 乘 topk 权重后原子加到 fp32 的
     Yacc (S, H)，最后 torch cast 回输入 dtype

与 CUDA 版的差异（性能定位「可用」而非极致）：无跨 tile 连续流水 / PDL /
scatter 预取，gemm2 归约用 fp32 原子加（CUDA 为 expert 有序确定性归约，
本实现数值在容差内一致但浮点求和顺序不确定）；多一次 Xc 物化搬运。
"""

import torch

import tilelang
import tilelang.language as T

_KERNEL_CACHE = {}


# ── kernel 1: expert 直方图 + padded 前缀和（单 CTA）────────────────────────

@tilelang.jit()
def _moe_count_factory(S, num_topk, E, block_M):
    @T.prim_func
    def main(
        Ids:     T.Tensor((S, num_topk), "int32"),
        Offsets: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(1, threads=256) as bx:
            cnt = T.alloc_shared((E,), "int32")
            off = T.alloc_shared((E + 1,), "int32")
            for i in T.Parallel(E):
                cnt[i] = 0
            # 直方图：S*K 个路由项按线程均分，shared 原子累加
            for i in T.Parallel(S * num_topk):
                T.atomic_add(cnt[Ids[i // num_topk, i % num_topk]], 1)
            # padded exclusive 前缀和：每个 expert 的段长向上取整到 block_M
            off[0] = 0
            for e in T.serial(E):
                nb = (cnt[e] + block_M - 1) // block_M
                off[e + 1] = off[e] + nb * block_M
            for e in T.Parallel(E + 1):
                Offsets[e] = off[e]
    return main


# ── kernel 2: 路由（原子领取 expert 段内槽位）───────────────────────────────

@tilelang.jit()
def _moe_route_factory(S, num_topk, E, rows):
    @T.prim_func
    def main(
        Ids:          T.Tensor((S, num_topk), "int32"),
        Scale:        T.Tensor((S, num_topk), "float32"),
        Offsets:      T.Tensor((E + 1,), "int32"),
        Cursor:       T.Tensor((E,), "int32"),
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
                    SortedToken[pos] = s
                    SortedWeight[pos] = Scale[s, j]
    return main


# ── kernel 3: x 行收集为 padded 布局 ────────────────────────────────────────

@tilelang.jit()
def _moe_gather_factory(S, H, rows, block_M, dtype):
    @T.prim_func
    def main(
        SortedToken: T.Tensor((rows,), "int32"),
        X:           T.Tensor((S, H), dtype),
        Xc:          T.Tensor((rows, H), dtype),
    ):
        with T.Kernel(T.ceildiv(rows, block_M), T.ceildiv(H, 256),
                      threads=256) as (bx, by):
            tok = T.alloc_fragment((block_M,), "int32")
            for i in T.Parallel(block_M):
                tok[i] = SortedToken[bx * block_M + i]
            # 行 serial、列 parallel：warp 内 c 连续（coalesced），且规避
            # sm120 上「fp16 + 2D Parallel + 动态行索引」的代码生成 bug
            #（cutlass::half_t → half 转换失败；bf16/仿射索引不受影响）
            for i in T.serial(block_M):
                s = T.if_then_else(tok[i] >= 0, tok[i], 0)
                for j in T.Parallel(256):
                    c = by * 256 + j
                    if c < H:
                        # 哨兵行（-1）取 X[0] 占位；其下游结果被行守卫丢弃
                        Xc[bx * block_M + i, c] = X[s, c]
    return main


# ── kernel 4: gemm1（gate/up 配对融合 + silu·mul epilogue）─────────────────

@tilelang.jit()
def _moe_gemm1_factory(E, I, H, rows, grid_m, dtype,
                       block_M=128, block_N=128, block_K=64,
                       num_stages=2, threads=256, fullrow=False):
    """fullrow=True：GemmWarpPolicy.FullRow。实测 BN1=128 面板下比默认
    Square 快 ~4%，与 BN64→128 叠计 gemm1 提速 1.19x（fuse_moe 主瓶颈）。"""
    @T.prim_func
    def main(
        Offsets: T.Tensor((E + 1,), "int32"),
        Xc:      T.Tensor((rows, H), dtype),
        W1:      T.Tensor((E, 2 * I, H), dtype),
        Act:     T.Tensor((rows, I), dtype),
    ):
        with T.Kernel(grid_m, T.ceildiv(I, block_N),
                      threads=threads) as (bx, by):
            # 线性扫描定位本 m-block 的 expert（越界块 → -1，整块跳过）。
            # 注意：sm120 的 eager builder 不允许 shared 标量读入 Python
            # 变量后再用于 if 条件（"used before definition"），故内联 esh[0]
            esh = T.alloc_shared((1,), "int32")
            esh[0] = -1
            for ei in T.serial(E):
                if Offsets[ei] <= bx * block_M:
                    if bx * block_M < Offsets[ei + 1]:
                        esh[0] = ei
            ee = T.if_then_else(esh[0] >= 0, esh[0], 0)
            if esh[0] >= 0:
                X_shared = T.alloc_shared((block_M, block_K), dtype)
                Wg_shared = T.alloc_shared((block_N, block_K), dtype)
                Wu_shared = T.alloc_shared((block_N, block_K), dtype)
                acc_g = T.alloc_fragment((block_M, block_N), "float32")
                acc_u = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(acc_g)
                T.clear(acc_u)
                for k in T.Pipelined(T.ceildiv(H, block_K),
                                     num_stages=num_stages):
                    T.copy(Xc[bx * block_M:(bx + 1) * block_M,
                              k * block_K:(k + 1) * block_K], X_shared)
                    T.copy(W1[ee, by * block_N:(by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], Wg_shared)
                    T.copy(W1[ee, I + by * block_N:I + (by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], Wu_shared)
                    if fullrow:
                        T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True)
                        T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True)
                # epilogue：silu(gate)·up，原地覆盖 acc_g 后一次写出
                for i, j in T.Parallel(block_M, block_N):
                    g = acc_g[i, j]
                    acc_g[i, j] = g / (1.0 + T.exp(-g)) * acc_u[i, j]
                T.copy(acc_g, Act[bx * block_M:(bx + 1) * block_M,
                                  by * block_N:(by + 1) * block_N])
    return main


# ── kernel 5: gemm2（topk 加权 + fp32 原子归约到 Yacc）─────────────────────

@tilelang.jit()
def _moe_gemm2_factory(E, I, H, S, rows, grid_m, dtype,
                       block_M=128, block_N=128, block_K=64,
                       num_stages=2, threads=256, fullrow=False):
    @T.prim_func
    def main(
        Offsets:      T.Tensor((E + 1,), "int32"),
        SortedToken:  T.Tensor((rows,), "int32"),
        SortedWeight: T.Tensor((rows,), "float32"),
        Act:          T.Tensor((rows, I), dtype),
        W2:           T.Tensor((E, H, I), dtype),
        Yacc:         T.Tensor((S, H), "float32"),
    ):
        with T.Kernel(grid_m, T.ceildiv(H, block_N),
                      threads=threads) as (bx, by):
            esh = T.alloc_shared((1,), "int32")
            esh[0] = -1
            for ei in T.serial(E):
                if Offsets[ei] <= bx * block_M:
                    if bx * block_M < Offsets[ei + 1]:
                        esh[0] = ei
            ee = T.if_then_else(esh[0] >= 0, esh[0], 0)
            if esh[0] >= 0:
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                W_shared = T.alloc_shared((block_N, block_K), dtype)
                acc = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(I, block_K),
                                     num_stages=num_stages):
                    T.copy(Act[bx * block_M:(bx + 1) * block_M,
                               k * block_K:(k + 1) * block_K], A_shared)
                    T.copy(W2[ee, by * block_N:(by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], W_shared)
                    if fullrow:
                        T.gemm(A_shared, W_shared, acc, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(A_shared, W_shared, acc, transpose_B=True)
                # epilogue：乘 topk 权重，原子加到对应 token 行
                tok = T.alloc_fragment((block_M,), "int32")
                wt = T.alloc_fragment((block_M,), "float32")
                for i in T.Parallel(block_M):
                    tok[i] = SortedToken[bx * block_M + i]
                    wt[i] = SortedWeight[bx * block_M + i]
                for i, j in T.Parallel(block_M, block_N):
                    if tok[i] >= 0:
                        c = by * block_N + j
                        if c < H:
                            T.atomic_add(Yacc[tok[i], c], acc[i, j] * wt[i])
    return main


# ─────────────────────────────────────────────────────────────────────────────
# config / 前端
# ─────────────────────────────────────────────────────────────────────────────

def _default_config(S, num_topk, E):
    """block_M 按平均 tokens/expert 自适应（同 CUDA 版 32/64/128 策略）；
    block_N1=128+FullRow：gemm1 大面板 + warp 行带（实测 gemm1 1.19x）。"""
    avg = S * num_topk / max(E, 1)
    if avg >= 256:
        bm = 128
    elif avg >= 64:
        bm = 64
    else:
        bm = 32
    return dict(block_M=bm,
                block_N1=128, block_K1=64, num_stages1=2,
                block_N2=128, block_K2=64, num_stages2=2,
                threads=256, fullrow1=True, fullrow2=False)


def candidate_configs(S, num_topk, E):
    base = _default_config(S, num_topk, E)
    return [
        base,
        dict(base, num_stages1=3, num_stages2=3),
        dict(base, block_N1=64, fullrow1=False),
        dict(base, block_N2=64),
        dict(base, fullrow2=True),
    ]


def _make(key, factory, *args, **kwargs):
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    kernel = factory(*args, **kwargs)
    _KERNEL_CACHE[key] = kernel
    return kernel


def fuse_moe_tilelang(x, gate_up_weight, down_weight, topk_ids, topk_scale,
                      config=None):
    """Tile-lang 版 fuse_moe 前端（签名与 ops.fuse_moe 对齐）。"""
    if x.dim() != 2:
        raise ValueError(f"x must be (S, H) 2D, got {tuple(x.shape)}")
    S, H = x.shape
    if gate_up_weight.dim() != 3 or down_weight.dim() != 3:
        raise ValueError("gate_up_weight (E, 2I, H) / down_weight (E, H, I) 必须是 3D")
    E, N2, Hw = gate_up_weight.shape
    if N2 % 2 != 0:
        raise ValueError("gate_up_weight.shape[1] 必须是 2I（gate 在前 up 在后）")
    I = N2 // 2
    if Hw != H or down_weight.shape != (E, H, I):
        raise ValueError(
            f"shape mismatch: x({S},{H}) w1{tuple(gate_up_weight.shape)} "
            f"w2{tuple(down_weight.shape)}")
    if topk_ids.shape != (S, topk_ids.shape[1]) or topk_ids.dim() != 2:
        raise ValueError("topk_ids must be (S, K) int32")
    num_topk = topk_ids.shape[1]
    if topk_scale.shape != (S, num_topk) or topk_scale.dtype != torch.float32:
        raise ValueError("topk_scale must be (S, K) fp32")
    if topk_ids.dtype != torch.int32:
        raise ValueError("topk_ids must be int32")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x 必须是 bf16 / fp16")
    if gate_up_weight.dtype != x.dtype or down_weight.dtype != x.dtype:
        raise ValueError("权重必须与 x 同 dtype")
    if H % 64 != 0 or I % 64 != 0:
        raise ValueError("H % 64 == 0 且 I % 64 == 0（对齐契约）")
    if num_topk > 128 or E > 512:
        raise ValueError("K ≤ 128 且 E ≤ 512")

    dtype = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[x.dtype]
    cfg = config or _default_config(S, num_topk, E)
    BM = cfg["block_M"]

    # padded 布局上界：Σ ceil(cnt_e/BM)·BM ≤ S·K + E·(BM-1)
    em_max = S * num_topk + E * (BM - 1)
    grid_m = (em_max + BM - 1) // BM
    rows = grid_m * BM

    dev = x.device
    offsets = torch.empty(E + 1, dtype=torch.int32, device=dev)
    cursor = torch.zeros(E, dtype=torch.int32, device=dev)
    sorted_token = torch.full((rows,), -1, dtype=torch.int32, device=dev)
    sorted_weight = torch.zeros(rows, dtype=torch.float32, device=dev)
    xc = torch.empty((rows, H), dtype=x.dtype, device=dev)
    act = torch.empty((rows, I), dtype=x.dtype, device=dev)
    yacc = torch.zeros((S, H), dtype=torch.float32, device=dev)

    k1 = _make(("cnt", S, num_topk, E, BM),
               _moe_count_factory, S, num_topk, E, BM)
    k2 = _make(("route", S, num_topk, E, rows),
               _moe_route_factory, S, num_topk, E, rows)
    k3 = _make(("gather", S, H, rows, BM, dtype),
               _moe_gather_factory, S, H, rows, BM, dtype)
    k4 = _make(("gemm1", E, I, H, rows, grid_m, dtype,
                cfg["block_M"], cfg["block_N1"], cfg["block_K1"],
                cfg["num_stages1"], cfg["threads"],
                cfg.get("fullrow1", False)),
               _moe_gemm1_factory, E, I, H, rows, grid_m, dtype,
               block_M=cfg["block_M"], block_N=cfg["block_N1"],
               block_K=cfg["block_K1"], num_stages=cfg["num_stages1"],
               threads=cfg["threads"], fullrow=cfg.get("fullrow1", False))
    k5 = _make(("gemm2", E, I, H, S, rows, grid_m, dtype,
                cfg["block_M"], cfg["block_N2"], cfg["block_K2"],
                cfg["num_stages2"], cfg["threads"],
                cfg.get("fullrow2", False)),
               _moe_gemm2_factory, E, I, H, S, rows, grid_m, dtype,
               block_M=cfg["block_M"], block_N=cfg["block_N2"],
               block_K=cfg["block_K2"], num_stages=cfg["num_stages2"],
               threads=cfg["threads"], fullrow=cfg.get("fullrow2", False))

    k1(topk_ids, offsets)
    k2(topk_ids, topk_scale, offsets, cursor, sorted_token, sorted_weight)
    k3(sorted_token, x, xc)
    k4(offsets, xc, gate_up_weight, act)
    k5(offsets, sorted_token, sorted_weight, act, down_weight, yacc)
    return yacc.to(x.dtype)
