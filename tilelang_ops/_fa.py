"""
custom_ops/tilelang_ops/_fa.py — Tile-lang 版 mha_fwd_with_mask

与 custom_ops.ops.mha_fwd_with_mask（CUDA/CuTe 后端）完全同语义：

  - q: (B, H, Sq, d)  k/v: (B, Hk, Sk, d)  mask: (B, 1, Sq_m, Sk_m) 加性 mask
  - softmax(Q·K^T * scale + mask) · V，全屏蔽行输出 0
  - Sk % 8 == 0，Sq 任意；mask 可比 (Sq, Sk) 大，越界列内容被忽略（恒屏蔽）
  - GQA：H // Hk 组共享 KV 头
  - 自适应 Split-KV：小 grid 长序列时两阶段（split + LSE 归并）

语义对齐细节（与 CuTe 版一致）：
  - softmax(QK^T·scale + mask)：mask 直接加在缩放后的分数上
    （log2 域即 qk*scale_log2 + m*log2e，与 CuTe 版“mask 乘 1/scale 预补偿”等价）
  - 全屏蔽行（row_max == -inf）行最大值 clamp 到 0（FA2 Check_inf 语义），
    归一化分母 sum==0 时取 1 → 输出 0
  - P 矩阵量化回输入 dtype 再做 PV（与 CuTe 的 P-quantization 一致）

实现源自 benchmark/fa_tilelang.py（性能对拍用例见该文件）。
"""

import torch

import tilelang
import tilelang.language as T


# ─────────────────────────────────────────────────────────────────────────────
# Tile-lang kernel
# ─────────────────────────────────────────────────────────────────────────────

_KERNEL_CACHE = {}


@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _fa_mask_factory(batch, heads, heads_kv, seq_q, seq_kv, dim,
                     mask_seq_q, mask_seq_kv, dtype,
                     block_M=64, block_N=64, num_stages=2, threads=128,
                     mask_smem=False):
    """shape 特化的 FA-with-mask kernel 工厂（tilelang.jit 惯用法）。

    mask_smem=True 时 mask tile 经共享内存加载（进入 T.Pipelined 异步流水，
    等价 CuTe 的 cp.async mask 路径），代价是额外 stages×block_M×block_N×2
    字节 smem；False 时用带钳位索引的直接全局读（零 smem，但无法预取）。
    """
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)：log2 域缩放
    group_ratio = heads // heads_kv
    assert heads % heads_kv == 0, "H 必须是 Hk 的整数倍（GQA）"

    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads_kv, seq_kv, dim]
    mask_shape = [batch, 1, mask_seq_q, mask_seq_kv]
    accum_dtype = "float32"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Mask: T.Tensor(mask_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch,
                      threads=threads) as (bx, by, bz):
            # smem 预算（sm89 ≈ 99KB）：Q (block_M×d) + stages×(K+V) (block_N×d)
            #   d128 M128/N64/s2: 32 + 2×32 = 96KB   d64 M128/N128/s2: 16 + 2×32 = 80KB
            # mask 不占 smem（元素级带钳位直接读），O 直接 fragment→global 写出
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            if mask_smem:
                Mask_shared = T.alloc_shared([block_M, block_N], dtype)

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seq_kv, block_N)

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # K/V tile（GQA：组内共享 KV 头）；T.copy 自带 OOB 谓词（零填充）
                T.copy(K[bz, by // group_ratio,
                         k * block_N:(k + 1) * block_N, :], K_shared)
                if mask_smem:
                    # mask tile 进异步流水（OOB 行/列零填充：行越界仅影响
                    # 被丢弃的输出行，列越界被下方 col-guard 屏蔽）
                    T.copy(Mask[bz, 0, bx * block_M:(bx + 1) * block_M,
                                k * block_N:(k + 1) * block_N], Mask_shared)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                # 语义：softmax(QK^T * scale + mask)（log2 域：qk*scale_log2 + m*log2e）
                # col >= Sk 恒屏蔽（忽略预 pad mask 的越界列内容）。
                # mask=-inf 时 finite + (-inf) = -inf，无 NaN。
                # mask 读取用钳位索引保证越界安全（读到的值被外层 if 丢弃）。
                for i, j in T.Parallel(block_M, block_N):
                    c = k * block_N + j
                    if mask_smem:
                        m = T.cast(Mask_shared[i, j], accum_dtype)
                    else:
                        r = bx * block_M + i
                        m = T.cast(
                            Mask[bz, 0, T.min(r, mask_seq_q - 1),
                                 T.min(c, mask_seq_kv - 1)], accum_dtype)
                    acc_s[i, j] = T.if_then_else(
                        c < seq_kv,
                        acc_s[i, j] * scale + m * 1.44269504,
                        -T.infinity(accum_dtype))

                # online softmax（log2 域）
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    # 全屏蔽行守卫：-inf clamp 到 0（此后 p=exp2(-inf-0)=0，
                    # acc_o/logsum 同比例缩放，结果仍正确且无 NaN）
                    scores_max[i] = T.if_then_else(
                        scores_max[i] == -T.infinity(accum_dtype),
                        0.0, scores_max[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by // group_ratio,
                         k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o,
                       policy=T.GemmWarpPolicy.FullRow)

            # 归一化；全屏蔽行 logsum==0 → 除以 1 → 输出 0
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(logsum[i] == 0.0, 1.0, logsum[i])
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main


def make_fa_mask(batch, heads, heads_kv, seq_q, seq_kv, dim,
                 mask_seq_q=None, mask_seq_kv=None, dtype="float16",
                 block_M=64, block_N=64, num_stages=2, threads=128,
                 mask_smem=False):
    """构建（并缓存）一个 shape 特化的 FA-with-mask JITKernel。"""
    mask_seq_q = mask_seq_q or seq_q
    mask_seq_kv = mask_seq_kv or seq_kv
    key = (batch, heads, heads_kv, seq_q, seq_kv, dim,
           mask_seq_q, mask_seq_kv, dtype, block_M, block_N, num_stages,
           threads, mask_smem)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    kernel = _fa_mask_factory(batch, heads, heads_kv, seq_q, seq_kv, dim,
                              mask_seq_q, mask_seq_kv, dtype,
                              block_M=block_M, block_N=block_N,
                              num_stages=num_stages, threads=threads,
                              mask_smem=mask_smem)
    _KERNEL_CACHE[key] = kernel
    return kernel


# ─────────────────────────────────────────────────────────────────────────────
# Split-KV 变体（小 grid + 长 Sk 场景，对应 CuTe 版的 Split-KV 路径）
#   phase 1: grid (m_blocks*B, H, num_split) 每个 CTA 只扫一段 K，输出
#            归一化 partial O（fp32）+ log2 域 LSE；
#   phase 2: grid (m_blocks*B, H) 按 LSE 加权归并各 split。
# ─────────────────────────────────────────────────────────────────────────────

@tilelang.jit(
    out_idx=[4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _fa_split_factory(batch, heads, heads_kv, seq_q, seq_kv, dim,
                      mask_seq_q, mask_seq_kv, dtype,
                      block_M=64, block_N=64, num_stages=2, threads=128,
                      num_split=2):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    group_ratio = heads // heads_kv
    tiles = (seq_kv + block_N - 1) // block_N            # 编译期常量
    tiles_per_split = (tiles + num_split - 1) // num_split
    m_blocks = (seq_q + block_M - 1) // block_M
    accum_dtype = "float32"

    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads_kv, seq_kv, dim]
    mask_shape = [batch, 1, mask_seq_q, mask_seq_kv]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Mask: T.Tensor(mask_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        # 中间全局缓冲：fp32 保精度（与 CuTe splitkv 的 fp32 partial 一致）
        O_partial = T.alloc_global(
            [batch, heads, num_split, seq_q, dim], accum_dtype)
        LSE = T.alloc_global(
            [batch, heads, num_split, seq_q], accum_dtype)

        # ── phase 1: split ──  bx = b*m_blocks + m_block
        with T.Kernel(m_blocks * batch, heads, num_split,
                      threads=threads) as (bx, by, bz):
            b_idx = bx // m_blocks
            mx = bx % m_blocks
            sid = bz

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[b_idx, by, mx * block_M:(mx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # 静态循环长度（T.Pipelined 不接受动态 extent）：每个 split 固定
            # 迈 tiles_per_split 步；超出 tiles 的幻影 tile 由 col-guard
            # （c >= seq_kv → -inf）+ T.copy 零填充天然无害（p=0，sum+=0）。
            # num_split > tiles 时多余 split 全为幻影，同样安全。
            start = sid * tiles_per_split

            for k in T.Pipelined(tiles_per_split, num_stages=num_stages):
                kk = start + k
                T.copy(K[b_idx, by // group_ratio,
                         kk * block_N:(kk + 1) * block_N, :], K_shared)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, block_N):
                    r = mx * block_M + i
                    c = kk * block_N + j
                    m = T.cast(
                        Mask[b_idx, 0, T.min(r, mask_seq_q - 1),
                             T.min(c, mask_seq_kv - 1)], accum_dtype)
                    acc_s[i, j] = T.if_then_else(
                        c < seq_kv,
                        acc_s[i, j] * scale + m * 1.44269504,
                        -T.infinity(accum_dtype))

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_max[i] = T.if_then_else(
                        scores_max[i] == -T.infinity(accum_dtype),
                        0.0, scores_max[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[b_idx, by // group_ratio,
                         kk * block_N:(kk + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o,
                       policy=T.GemmWarpPolicy.FullRow)

            # 每 split 归一化 + LSE（log2 域）。注意顺序：先用真实 logsum 算
            # LSE（log2(0)=-inf → 空 split/全屏蔽行 LSE=-inf，combine 权重为 0），
            # 再把 logsum 钳位到 1 做除法守卫（partial 输出 0）。
            for i in T.Parallel(block_M):
                LSE[b_idx, by, sid, T.min(mx * block_M + i, seq_q - 1)] = (
                    T.log2(logsum[i]) + scores_max[i])
                logsum[i] = T.if_then_else(logsum[i] == 0.0, 1.0, logsum[i])
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_partial[b_idx, by, sid,
                                    mx * block_M:(mx + 1) * block_M, :])

        # ── phase 2: combine ──  bx = (b*m_blocks + m_block)*num_chunks + chunk
        # 按 block_M/num_chunks 行分块，避免 [block_M, dim] 级 fragment（寄存器溢出）
        # 且大幅提高 combine 的 CTA 并行度。
        rows_per_chunk = 32
        num_chunks = block_M // rows_per_chunk
        with T.Kernel(m_blocks * batch * num_chunks, heads,
                      threads=threads) as (bx, by):
            b_idx = (bx // num_chunks) // m_blocks
            mx = (bx // num_chunks) % m_blocks
            sub = bx % num_chunks
            row0 = mx * block_M + sub * rows_per_chunk
            lse = T.alloc_fragment([num_split, rows_per_chunk], accum_dtype)
            w = T.alloc_fragment([num_split, rows_per_chunk], accum_dtype)
            m_global = T.alloc_fragment([rows_per_chunk], accum_dtype)
            wsum = T.alloc_fragment([rows_per_chunk], accum_dtype)
            o_part = T.alloc_fragment([rows_per_chunk, dim], accum_dtype)
            o_acc = T.alloc_fragment([rows_per_chunk, dim], accum_dtype)

            for s, i in T.Parallel(num_split, rows_per_chunk):
                lse[s, i] = LSE[b_idx, by, s,
                                T.min(row0 + i, seq_q - 1)]
            T.reduce_max(lse, m_global, dim=0, clear=True)
            # 全屏蔽行（所有 split LSE=-inf）：clamp 到 0，权重 exp2(-inf)=0
            for i in T.Parallel(rows_per_chunk):
                m_global[i] = T.if_then_else(
                    m_global[i] == -T.infinity(accum_dtype), 0.0, m_global[i])
            for s, i in T.Parallel(num_split, rows_per_chunk):
                w[s, i] = T.exp2(lse[s, i] - m_global[i])
            T.reduce_sum(w, wsum, dim=0)
            for i in T.Parallel(rows_per_chunk):
                wsum[i] = T.if_then_else(wsum[i] == 0.0, 1.0, wsum[i])

            T.clear(o_acc)
            for s in T.serial(num_split):
                for i, j in T.Parallel(rows_per_chunk, dim):
                    o_part[i, j] = O_partial[b_idx, by, s,
                                             T.min(row0 + i, seq_q - 1), j]
                for i, j in T.Parallel(rows_per_chunk, dim):
                    o_acc[i, j] += o_part[i, j] * w[s, i]
            for i, j in T.Parallel(rows_per_chunk, dim):
                o_acc[i, j] /= wsum[i]
            T.copy(o_acc, Output[b_idx, by,
                                 row0:row0 + rows_per_chunk, :])

    return main


def make_fa_split(batch, heads, heads_kv, seq_q, seq_kv, dim,
                  mask_seq_q=None, mask_seq_kv=None, dtype="float16",
                  block_M=64, block_N=64, num_stages=2, threads=128,
                  num_split=2):
    """构建（并缓存）Split-KV FA-with-mask JITKernel。"""
    mask_seq_q = mask_seq_q or seq_q
    mask_seq_kv = mask_seq_kv or seq_kv
    key = ("split", batch, heads, heads_kv, seq_q, seq_kv, dim,
           mask_seq_q, mask_seq_kv, dtype, block_M, block_N, num_stages,
           threads, num_split)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    kernel = _fa_split_factory(batch, heads, heads_kv, seq_q, seq_kv, dim,
                               mask_seq_q, mask_seq_kv, dtype,
                               block_M=block_M, block_N=block_N,
                               num_stages=num_stages, threads=threads,
                               num_split=num_split)
    _KERNEL_CACHE[key] = kernel
    return kernel


# ─────────────────────────────────────────────────────────────────────────────
# Config 选择
# ─────────────────────────────────────────────────────────────────────────────

# sm89 (4090) smem ≈ 99KB：smem = block_M×d×2 + stages × 2×block_N×d×2（字节）
# d128: 128x64/s2=96KB ✓  64x64/s2=80KB ✓  128x64/s3=128KB ✗（sweep 自动跳过）
CONFIGS_D128 = [
    dict(block_M=128, block_N=64, num_stages=2, threads=256),
    dict(block_M=128, block_N=64, num_stages=2, threads=256, mask_smem=True),
    dict(block_M=64, block_N=64, num_stages=2, threads=128, mask_smem=True),
    dict(block_M=128, block_N=64, num_stages=3, threads=256),
    dict(block_M=64, block_N=64, num_stages=2, threads=128),
    dict(block_M=128, block_N=128, num_stages=2, threads=256),
    dict(block_M=64, block_N=64, num_stages=3, threads=128),
]
# d64: 128x128/s2=80KB ✓  128x64/s3=64KB ✓
#   +mask_smem: 128x64/s2 = 16+2×16+2×16 = 80KB ✓  64x64/s3 = 8+48+24 = 80KB ✓
CONFIGS_D64 = [
    dict(block_M=128, block_N=128, num_stages=2, threads=256),
    dict(block_M=128, block_N=64, num_stages=3, threads=256),
    dict(block_M=128, block_N=64, num_stages=2, threads=256, mask_smem=True),
    dict(block_M=64, block_N=64, num_stages=3, threads=128, mask_smem=True),
    dict(block_M=64, block_N=128, num_stages=2, threads=128),
    dict(block_M=128, block_N=128, num_stages=3, threads=256),
]


def candidate_configs(d):
    return CONFIGS_D128 if d >= 128 else CONFIGS_D64


def split_candidate_configs(B, H, Hk, Sq, Sk, d):
    """Split-KV 候选：小 grid（CTA 数不足）时把 K 切到多 CTA 并行。

    仅当 base grid < 256 CTA 时提供；num_split 取 2 的幂、不超过 K tile 数、
    目标总 CTA 数 ≤ 512（≈ 4×SM），最多 4 个候选。"""
    m_blocks = (Sq + 127) // 128
    total_ctas = B * H * m_blocks
    if total_ctas >= 256:
        return []
    base = (dict(block_M=128, block_N=64, num_stages=2, threads=256) if d >= 128
            else dict(block_M=128, block_N=64, num_stages=3, threads=256))
    tiles = (Sk + base["block_N"] - 1) // base["block_N"]
    max_split = min(tiles, 128, 512 // max(total_ctas, 1))
    cand = [s for s in (2, 4, 8, 16, 32, 64, 128) if s <= max_split]
    # log 间隔最多取 4 个：首、尾 + 中间两个
    if len(cand) > 4:
        cand = [cand[0], cand[len(cand) // 3], cand[2 * len(cand) // 3], cand[-1]]
    return [dict(base, num_split=s) for s in cand]


def _best_config(B, H, Hk, Sq, Sk, d, dtype):
    """轻量启发式（bench 路径会做完整 sweep，这里只给正确性路径一个可用值）。"""
    return candidate_configs(d)[0]


def fa_tilelang(q, k, v, mask, config=None):
    """Python 前端：与 ops.mha_fwd_with_mask 同签名（含 GQA）。

    Sk % 8 != 0 时自动 pad 到 8 倍数（与 CUDA 版同语义）：K/V 列尾补零、
    mask 列尾补 -inf（pad 列对 softmax 零贡献），pad 开销计入调用。
    """
    B, H, Sq, d = q.shape
    Hk, Sk = k.shape[1], k.shape[2]
    _, _, Sq_m, Sk_m = mask.shape
    assert q.dtype == k.dtype == v.dtype == mask.dtype
    if Sk % 8 != 0:
        pad = (Sk + 7) // 8 * 8 - Sk
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad),
                                       value=float("-inf"))
        Sk = Sk + pad
        Sk_m = max(Sk_m, Sk)
    dtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[q.dtype]
    cfg = config or _best_config(B, H, Hk, Sq, Sk, d, dtype)
    if cfg.get("num_split", 1) > 1:
        kernel = make_fa_split(B, H, Hk, Sq, Sk, d, Sq_m, Sk_m, dtype, **cfg)
    else:
        kernel = make_fa_mask(B, H, Hk, Sq, Sk, d, Sq_m, Sk_m, dtype, **cfg)
    return kernel(q, k, v, mask)
