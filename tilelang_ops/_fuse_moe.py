"""
custom_ops/tilelang_ops/_fuse_moe.py — Tile-lang 版 fuse_moe

与 CUDA 后端（csrc/fuse_moe/）同语义：

    y[s] = Σ_j topk_scale[s,j] · (Down_{e_j} @ (act(Gate_{e_j} @ x[s]) · (Up_{e_j} @ x[s])))
    GLU 使用 act(gate) * up；MLP 使用 act(projection)，只有一个 GEMM 面板。
    act = SiLU（默认）或 GELU exact erf；-1 masked 路由整条忽略。

  - x: (S, H) bf16/fp16；w1: GLU (E, 2I, H)（gate 在前 up 在后）或 MLP (E, I, H)；
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
  4. gemm1：每 CTA 线性扫描 Offsets 定位 expert；GLU 配对 gate / up
     面板写 act(gate)·up，MLP 单投影面板写 act(proj)，均直写 act_out
  5. gemm2：act @ Down[e]，epilogue 乘 topk 权重后原子加到 fp32 的
     Yacc (S, H)，最后 torch cast 回输入 dtype

与 CUDA 版的差异（性能定位「可用」而非极致）：无跨 tile 连续流水 / PDL /
scatter 预取，gemm2 归约用 fp32 原子加（CUDA 为 expert 有序确定性归约，
本实现数值在容差内一致但浮点求和顺序不确定）；多一次 Xc 物化搬运。
legacy fused 的 down/加权保持 FP32，CUDA fused 则先把 down 物化为 Tout；
staged 在两个后端均先 round down/weighted product 到 Tout，再 FP32 累加。
"""

import torch

from custom_ops.recsys import _resolve_fuse_moe_expert_type

import tilelang
import tilelang.language as T

_KERNEL_CACHE = {}


def _exact_gelu_uses_erf(expert_activation, language=None):
    """Host-only capability check; no approximate activation fallback or JIT."""
    language = T if language is None else language
    has_erf = callable(getattr(language, "erf", None))
    if expert_activation == "gelu" and not (
            has_erf or callable(getattr(language, "call_extern", None))):
        raise NotImplementedError(
            "fuse_moe expert_activation='gelu' requires an erf intrinsic "
            "(T.erf) or T.call_extern(..., 'erff', ...); this TileLang "
            "build supports neither, so exact GELU cannot be compiled")
    return has_erf


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
                e = Ids[i // num_topk, i % num_topk]
                if e >= 0:
                    if e < E:
                        T.atomic_add(cnt[e], 1)
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
                    # Guard all indices and the scale load: masked NaN scales
                    # must not reach the weighted accumulation.
                    if e >= 0:
                        if e < E:
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


# ── kernel 4: gemm1（GLU 配对 / MLP 单投影 + activation epilogue）──────────

@tilelang.jit()
def _moe_gemm1_factory(E, I, H, rows, grid_m, dtype,
                       block_M=128, block_N=128, block_K=64,
                       num_stages=2, threads=256, fullrow=False,
                       expert_activation="silu", numerical_mode="fused", *,
                       expert_type="glu"):
    """fullrow=True：GemmWarpPolicy.FullRow。实测 BN1=128 面板下比默认
    Square 快 ~4%，与 BN64→128 叠计 gemm1 提速 1.19x（fuse_moe 主瓶颈）。

    expert_activation / numerical_mode 是编译期特化：gelu 使用 exact erf
    （对齐 F.gelu(approximate="none")，不是 tanh 近似）；staged 在
    projection、激活结果处逐级 round 到输入低精度再回 fp32；
    只有 GLU 分配 up 面板、执行第二次 GEMM 和激活乘法。"""
    if (not isinstance(expert_activation, str)
            or expert_activation not in ("silu", "gelu")):
        raise ValueError(
            f"expert_activation must be 'silu' or 'gelu', "
            f"got {expert_activation!r}")
    if (not isinstance(numerical_mode, str)
            or numerical_mode not in ("fused", "staged")):
        raise ValueError(
            f"numerical_mode must be 'fused' or 'staged', "
            f"got {numerical_mode!r}")
    # 能力探测：优先标准 erf intrinsic，其次 CUDA/HIP 的标准 erff extern。
    # 两者都不可用时 gelu 必须显式失败，绝不回退到 tanh 或其它近似。
    expert_type_id = _resolve_fuse_moe_expert_type(expert_type)
    is_glu = expert_type_id == 0
    factor = 2 if is_glu else 1
    has_erf = _exact_gelu_uses_erf(expert_activation)
    # 预折叠为 Python bool，prim_func 内只做编译期分支（与 fullrow 同机制）
    is_silu = expert_activation == "silu"
    is_staged = numerical_mode == "staged"

    @T.prim_func
    def main(
        Offsets: T.Tensor((E + 1,), "int32"),
        Xc:      T.Tensor((rows, H), dtype),
        W1:      T.Tensor((E, factor * I, H), dtype),
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
                acc_g = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(acc_g)
                if is_glu:
                    Wu_shared = T.alloc_shared((block_N, block_K), dtype)
                    acc_u = T.alloc_fragment((block_M, block_N), "float32")
                    T.clear(acc_u)
                for k in T.Pipelined(T.ceildiv(H, block_K),
                                     num_stages=num_stages):
                    T.copy(Xc[bx * block_M:(bx + 1) * block_M,
                              k * block_K:(k + 1) * block_K], X_shared)
                    T.copy(W1[ee, by * block_N:(by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], Wg_shared)
                    if is_glu:
                        T.copy(W1[ee, I + by * block_N:I + (by + 1) * block_N,
                                  k * block_K:(k + 1) * block_K], Wu_shared)
                    if fullrow:
                        T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        if is_glu:
                            T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True,
                                   policy=T.GemmWarpPolicy.FullRow)
                    else:
                        T.gemm(X_shared, Wg_shared, acc_g, transpose_B=True)
                        if is_glu:
                            T.gemm(X_shared, Wu_shared, acc_u, transpose_B=True)
                # epilogue：GLU act(gate)·up / MLP act(proj)，一次写出。
                # fused 保持 fp32 中间量；staged 在每级算子边界 round
                # 到 Tout（dtype 与输入相同），再转回 FP32 计算下一阶段。
                for i, j in T.Parallel(block_M, block_N):
                    g = (T.cast(T.cast(acc_g[i, j], dtype), "float32")
                         if is_staged else acc_g[i, j])
                    if is_glu:
                        u = (T.cast(T.cast(acc_u[i, j], dtype), "float32")
                             if is_staged else acc_u[i, j])
                    if is_silu:
                        a = g / (1.0 + T.exp(-g))
                    elif has_erf:
                        a = 0.5 * g * (
                            1.0 + T.erf(g * 0.7071067811865476))
                    else:
                        a = 0.5 * g * (
                            1.0 + T.call_extern(
                                "float32", "erff",
                                g * 0.7071067811865476))
                    if is_glu:
                        if is_staged:
                            act_rounded = T.cast(T.cast(a, dtype), "float32")
                            acc_g[i, j] = T.cast(
                                T.cast(act_rounded * u, dtype), "float32")
                        else:
                            acc_g[i, j] = a * u
                    elif is_staged:
                        acc_g[i, j] = T.cast(T.cast(a, dtype), "float32")
                    else:
                        acc_g[i, j] = a
                T.copy(acc_g, Act[bx * block_M:(bx + 1) * block_M,
                                  by * block_N:(by + 1) * block_N])
    return main


# ── kernel 5: gemm2（topk 加权 + fp32 原子归约到 Yacc）─────────────────────

@tilelang.jit()
def _moe_gemm2_factory(E, I, H, S, rows, grid_m, dtype,
                       block_M=128, block_N=128, block_K=64,
                       num_stages=2, threads=256, fullrow=False,
                       numerical_mode="fused", expert_type="glu"):
    if expert_type not in ("glu", "mlp"):
        raise ValueError("expert_type must be 'glu' or 'mlp'")
    if numerical_mode not in ("fused", "staged"):
        raise ValueError(
            f"numerical_mode must be 'fused' or 'staged', "
            f"got {numerical_mode!r}")
    is_staged = numerical_mode == "staged"
    round_down = is_staged or expert_type == "mlp"

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
                if round_down:
                    # New MLP modes match the native down storage boundary;
                    # preserve the legacy GLU/fused FP32 down behavior.
                    # Keep rounding outside the vectorized atomic destination.
                    for i, j in T.Parallel(block_M, block_N):
                        down_rounded = T.cast(T.cast(acc[i, j], dtype), "float32")
                        weighted = down_rounded * wt[i]
                        acc[i, j] = (T.cast(T.cast(weighted, dtype), "float32")
                                     if is_staged else weighted)
                for i, j in T.Parallel(block_M, block_N):
                    if tok[i] >= 0:
                        c = by * block_N + j
                        if c < H:
                            if round_down:
                                T.atomic_add(Yacc[tok[i], c], acc[i, j])
                            else:
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


def _normalize_moe_config(config, intermediate, hidden):
    """Keep each GEMM N panel within one expert's logical projection.

    All dimensions follow the 64-column contract. Partial/oversized panels
    select its base width, avoiding cross-projection loads and predicated
    vectorized atomic destinations. Never mutate caller-owned configuration.
    """
    result = dict(config)
    for key, dimension in (("block_N1", intermediate), ("block_N2", hidden)):
        width = result[key]
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or width % 64:
            raise ValueError(f"{key} must be a positive multiple of 64")
        if width > dimension or dimension % width:
            result[key] = 64
    return result


def _make(key, factory, *args, **kwargs):
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    kernel = factory(*args, **kwargs)
    _KERNEL_CACHE[key] = kernel
    return kernel


def fuse_moe_tilelang(x, gate_up_weight, down_weight, topk_ids, topk_scale,
                      config=None, expert_activation="silu",
                      numerical_mode="fused", *, expert_type="glu"):
    """Tile-lang 版 fuse_moe 前端（签名与 ops.fuse_moe 对齐）。

    expert_type: "glu"（默认）| "mlp"，与激活/数值模式独立。
    MLP 不分配或加载 up 面板，不做 gate/up 乘法；staged 将 projection
    和激活分别 round 到输入 dtype，fused 在激活存储时 round。
    expert_activation: "silu" | "gelu"。gelu 为 exact erf，对齐
        F.gelu(approximate="none")，不使用 tanh 近似。
    numerical_mode: "fused"（默认，act·up 写出时 round，down/加权保持
        FP32 的 TileLang 旧语义）| "staged"（gate/up、激活、乘法、
        down 与 weighted product 逐级 round 到输入低精度再 FP32 累加）。
    -1 路由整条忽略，包括对应的 NaN scale；输入必须为连续 CUDA Tensor。
    """
    if (not isinstance(expert_activation, str)
            or expert_activation not in ("silu", "gelu")):
        raise ValueError(
            f"expert_activation must be 'silu' or 'gelu', "
            f"got {expert_activation!r}")
    if (not isinstance(numerical_mode, str)
            or numerical_mode not in ("fused", "staged")):
        raise ValueError(
            f"numerical_mode must be 'fused' or 'staged', "
            f"got {numerical_mode!r}")
    expert_type_id = _resolve_fuse_moe_expert_type(expert_type)
    factor = 2 if expert_type_id == 0 else 1
    for name, tensor in (("x", x), ("gate_up_weight", gate_up_weight),
                         ("down_weight", down_weight),
                         ("topk_ids", topk_ids),
                         ("topk_scale", topk_scale)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if x.dim() != 2:
        raise ValueError(f"x must be (S, H) 2D, got {tuple(x.shape)}")
    S, H = x.shape
    if S <= 0 or H <= 0:
        raise ValueError("x must have positive S and H")
    if gate_up_weight.dim() != 3:
        raise ValueError(
            "gate_up_weight must be (E, factor*I, H), got "
            f"{tuple(gate_up_weight.shape)}")
    if down_weight.dim() != 3:
        raise ValueError(
            "down_weight must be (E, H, I), got "
            f"{tuple(down_weight.shape)}")
    E, N2, Hw = gate_up_weight.shape
    if E <= 0 or E > 512:
        raise ValueError(f"num_expert E must be in [1, 512], got {E}")
    I = down_weight.shape[2]
    if I <= 0 or I > ((1 << 31) - 1) // 2 or H > (1 << 31) - 1:
        raise ValueError("H/I must be positive and within native int32 bounds")
    if N2 != factor * I or Hw != H or tuple(down_weight.shape) != (E, H, I):
        raise ValueError(
            f"shape mismatch: x({S},{H}) w1{tuple(gate_up_weight.shape)} "
            f"w2{tuple(down_weight.shape)}")
    if topk_ids.dim() != 2:
        raise ValueError(
            f"topk_ids must be (S, K) 2D int32, got {tuple(topk_ids.shape)}")
    if topk_ids.shape[0] != S:
        raise ValueError(
            f"topk_ids.shape[0] ({topk_ids.shape[0]}) must equal S ({S})")
    if topk_ids.dtype != torch.int32:
        raise ValueError(f"topk_ids must be int32, got {topk_ids.dtype}")
    num_topk = topk_ids.shape[1]
    if num_topk <= 0 or num_topk > 128:
        raise ValueError(f"num_topk K must be in [1, 128], got {num_topk}")
    if topk_scale.dim() != 2 or tuple(topk_scale.shape) != (S, num_topk):
        raise ValueError(
            f"topk_scale must be (S, K)=({S},{num_topk}), got "
            f"{tuple(topk_scale.shape)}")
    if topk_scale.dtype != torch.float32:
        raise ValueError(
            f"topk_scale must be float32, got {topk_scale.dtype}")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"x must be bfloat16 or float16, got {x.dtype}")
    if gate_up_weight.dtype != x.dtype or down_weight.dtype != x.dtype:
        raise ValueError(
            "gate_up_weight / down_weight must have the same dtype as x "
            f"({x.dtype}), got {gate_up_weight.dtype} / {down_weight.dtype}")
    if not (gate_up_weight.device == x.device and
            down_weight.device == x.device and
            topk_ids.device == x.device and
            topk_scale.device == x.device):
        raise ValueError(
            "x / gate_up_weight / down_weight / topk_ids / topk_scale "
            "must be on the same device")
    if H % 64 != 0 or I % 64 != 0:
        raise ValueError("H % 64 == 0 且 I % 64 == 0（对齐契约）")
    if not x.is_cuda:
        raise ValueError("fuse_moe requires CUDA tensors")
    # Fail before any allocation or JIT, not after compiling count/route kernels.
    _exact_gelu_uses_erf(expert_activation)

    dtype = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[x.dtype]
    cfg = _normalize_moe_config(config or _default_config(S, num_topk, E), I, H)
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
                expert_activation, numerical_mode, expert_type, factor,
                cfg["block_M"], cfg["block_N1"], cfg["block_K1"],
                cfg["num_stages1"], cfg["threads"],
                cfg.get("fullrow1", False)),
               _moe_gemm1_factory, E, I, H, rows, grid_m, dtype,
               expert_activation=expert_activation,
               numerical_mode=numerical_mode, expert_type=expert_type,
               block_M=cfg["block_M"], block_N=cfg["block_N1"],
               block_K=cfg["block_K1"], num_stages=cfg["num_stages1"],
               threads=cfg["threads"], fullrow=cfg.get("fullrow1", False))
    k5 = _make(("gemm2", E, I, H, S, rows, grid_m, dtype,
                numerical_mode, expert_type,
                cfg["block_M"], cfg["block_N2"], cfg["block_K2"],
                cfg["num_stages2"], cfg["threads"],
                cfg.get("fullrow2", False)),
               _moe_gemm2_factory, E, I, H, S, rows, grid_m, dtype,
               numerical_mode=numerical_mode, expert_type=expert_type,
               block_M=cfg["block_M"], block_N=cfg["block_N2"],
               block_K=cfg["block_K2"], num_stages=cfg["num_stages2"],
               threads=cfg["threads"], fullrow=cfg.get("fullrow2", False))

    k1(topk_ids, offsets)
    k2(topk_ids, topk_scale, offsets, cursor, sorted_token, sorted_weight)
    k3(sorted_token, x, xc)
    k4(offsets, xc, gate_up_weight, act)
    k5(offsets, sorted_token, sorted_weight, act, down_weight, yacc)
    return yacc.to(x.dtype)
