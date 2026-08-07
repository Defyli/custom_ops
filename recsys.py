"""
custom_ops/recsys.py — RecsysOps：推荐系统三大核心 CUDA 算子库

算子列表
--------
- mha_fwd_with_mask      Flash Attention 2 前向，支持任意 bf16 加法 mask
- pack_and_prepare_b1    融合 pack tokens + RoPE gather + attn_mask 构建
- jagged_pool_and_collect 融合 jagged pooling（v3：零 H→D copy）

快速使用
--------
    from custom_ops import ops                  # 推荐：直接使用包级单例
    from custom_ops import RecsysOps, ops       # 也可导入类名

    if ops.is_available():
        out  = ops.mha_fwd_with_mask(q, k, v, mask)
        outs = ops.pack_and_prepare_b1(h_s, h_c, h_i, s_len, c_len, i_len,
                                       static_cos, static_sin, S_max, S_mask)
        pool = ops.jagged_pool_and_collect(pool_values, pool_lengths,
                                           val_splits, len_splits,
                                           n_pooling, reduce_mode, ones_cache)
    else:
        # CUDA 不可用或编译失败时做 Python 回退
        ...

命名空间
--------
    torch.ops.recsys_ops.<name>(...)   — C++ 侧调用接口
    ops.<name>(...)                    — Python 封装，带类型标注
"""

from __future__ import annotations

import os
import torch

from custom_ops import CustomOps


# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # custom_ops/ 根目录


# ─────────────────────────────────────────────────────────────────────────────
# RecsysOps — 推荐系统算子库
# ─────────────────────────────────────────────────────────────────────────────

class RecsysOps(CustomOps):
    """
    推荐系统核心 CUDA 算子库。

    命名空间  : recsys_ops  → torch.ops.recsys_ops.<name>(...)
    编译产物  : recsys_ops_kernel.so
    注册算子  :
      - mha_fwd_with_mask       （必需）Flash Attention 2 前向
      - pack_and_prepare_b1     （必需）融合 pack/RoPE/attn_mask 算子
      - jagged_pool_and_collect （必需）融合 jagged pooling 算子
    """

    # ── 配置 ─────────────────────────────────────────────────────────────────
    namespace    = "recsys_ops"
    so_name      = "recsys_ops_kernel"
    required_ops = [
        "mha_fwd_with_mask",
        "pack_and_prepare_b1",
        "jagged_pool_and_collect",
    ]
    optional_ops = []

    # 编译缓存回退目录（优先使用 TORCH_EXTENSIONS_DIR 环境变量）
    build_dir_fallback = "/workdir/tetuan_cache/torch_extensions"

    def get_sources(self) -> list:
        """源文件列表：recsys_bindings.cpp + FA/pack/jagged CUDA kernel。"""
        csrc   = os.path.join(_THIS_DIR, "csrc")
        return [
            os.path.join(csrc, "recsys_bindings.cpp"),
            os.path.join(csrc, "fa",     "fa_fwd_op.cu"),
            os.path.join(csrc, "pack",   "pack_and_prepare.cu"),
            os.path.join(csrc, "jagged", "jagged_forward.cu"),
        ]

    def get_include_dirs(self) -> list:
        """头文件目录：thirdparty（cutlass/cute）+ csrc 各子目录 + 框架宏目录。"""
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,                          # cutlass/cute 头文件
            csrc,                                # custom_ops_macros.h
            os.path.join(csrc, "fa"),            # fa_fwd_op.h 等
            os.path.join(csrc, "pack"),          # pack_and_prepare.h
            os.path.join(csrc, "jagged"),        # jagged_forward.h
        ]

    # ── 显式方法：带类型标注，供 IDE 补全 ────────────────────────────────────

    def mha_fwd_with_mask(
        self,
        q:    torch.Tensor,   # (B, H,  Sq, d)  bfloat16 CUDA
        k:    torch.Tensor,   # (B, Hk, Sk, d)  bfloat16 CUDA
        v:    torch.Tensor,   # (B, Hk, Sk, d)  bfloat16 CUDA
        mask: torch.Tensor,   # (B, 1,  Sq, Sk) bfloat16 CUDA，0=可见 / -inf=屏蔽
    ) -> torch.Tensor:
        """
        Flash Attention 2 前向，支持任意 bf16 加法 mask。

        参数
        ----
        q, k, v : (B, H, S, d) bfloat16 CUDA，连续
        mask    : (B, 1, Sq, Sk) bfloat16 CUDA，加法 mask（0=可见，-inf=屏蔽）

        限制
        ----
        - d ∈ {64, 128}
        - H % Hk == 0（支持 GQA）
        - 不支持 dropout / causal / alibi / RoPE / KV-cache
        """
        return torch.ops.recsys_ops.mha_fwd_with_mask(q, k, v, mask)

    def pack_and_prepare_b1(
        self,
        h_s:        torch.Tensor,   # (s_len, D)        bf16 CUDA
        h_c:        torch.Tensor,   # (c_len, D)        bf16 CUDA
        h_i:        torch.Tensor,   # (i_len, D)        bf16 CUDA
        s_len:      int,
        c_len:      int,
        i_len:      int,
        static_cos: torch.Tensor,   # (1, S_max, 1, hd) bf16 CUDA
        static_sin: torch.Tensor,   # (1, S_max, 1, hd) bf16 CUDA
        S_max:      int,
        S_mask:     int,
    ) -> tuple:
        """
        融合算子：pack tokens + gather RoPE cos/sin + build attn_mask。

        取代原始 Python 实现中的三个独立步骤：
          1. pack_jagged_tokens(mode=4) → h_dense, item_mask, offsets
          2. RoPE gather               → cos, sin
          3. _build_request_attn_mask  → attn_mask

        参数
        ----
        h_s, h_c, h_i : (n, D) bf16 CUDA，ubc/ctx/item token 特征
        s_len, c_len, i_len : int，各段有效长度（Python 侧先 .item() 提取，避免 D2H 同步）
        static_cos, static_sin : (1, S_max, 1, head_dim) bf16 CUDA，预计算 RoPE 表
        S_max  : int，固定序列长度（ubc_max + ctx_max + item_num + padding）
        S_mask : int，attn_mask 输出边长（>= S_max）

        返回
        ----
        h_dense   : (1, S_max, D)             bf16  — pad 后的 dense token 序列
        cos       : (1, S_max, 1, head_dim)   bf16  — RoPE cos
        sin       : (1, S_max, 1, head_dim)   bf16  — RoPE sin
        attn_mask : (1, 1, S_mask, S_mask)    bf16  — causal + item 间屏蔽 mask
        item_mask : (valid_tokens,)           bool  — jagged 空间 item 标记
        offsets   : (2,)                      int64 — [0, valid_tokens]
        """
        outs = torch.ops.recsys_ops.pack_and_prepare_b1(
            h_s, h_c, h_i,
            int(s_len), int(c_len), int(i_len),
            static_cos, static_sin, int(S_max), int(S_mask),
        )
        return tuple(outs)

    def jagged_pool_and_collect(
        self,
        pool_values:     torch.Tensor,   # (total_pool_val_rows, D) bf16/fp32 CUDA
        pool_lengths:    torch.Tensor,   # (total_pool_len_rows,) int32 CUDA
        pool_val_splits: list,           # (n_pooling+1,) Python int list
        pool_len_splits: list,           # (n_pooling+1,) Python int list
        n_pooling:       int,
        reduce_mode:     int,            # 0=mean, 1=sum
        ones_cache:      torch.Tensor,   # (max_n,) int32 CUDA，预分配全 1 缓存
    ) -> tuple:
        """
        融合 pooling 算子：只处理 pooling 特征的批量 segment reduce（v3：零 H→D copy）。

        旧版需要 Python 先 cat 全量特征，新版只需 cat pooling 特征，
        消除了对透传特征的不必要 CUDA 内存拷贝。

        参数
        ----
        pool_values     : (total_pool_val_rows, D) — 所有 pooling 特征 values 拼接（bf16/fp32）
        pool_lengths    : (total_pool_len_rows,) int32 — lengths 拼接
        pool_val_splits : list[int]，len=n_pooling+1，values 行偏移
        pool_len_splits : list[int]，len=n_pooling+1，lengths 行偏移
        n_pooling       : int，pooling 特征数量
        reduce_mode     : int，0=mean / 1=sum
        ones_cache      : (max_n,) int32 CUDA，预分配全 1 缓存，避免每次 alloc

        返回
        ----
        pool_values_out  : (total_pool_out_rows, D) — pooling 结果
        pool_lengths_out : (total_pool_out_rows,) int32 — 全为 1
        out_val_splits   : (n_pooling+1,) int64 CPU — 切分 pool_values_out 的 offsets
        out_len_splits   : (n_pooling+1,) int64 CPU — 切分 pool_lengths_out 的 offsets
        """
        outs = torch.ops.recsys_ops.jagged_pool_and_collect(
            pool_values, pool_lengths,
            pool_val_splits, pool_len_splits,
            n_pooling, reduce_mode, ones_cache,
        )
        return tuple(outs)

    def register_fake_impls(self) -> None:
        """
        为所有算子注册 fake（meta）实现，供 torch.compile / AOTI 做 shape 推断。
        load() 成功后由基类自动调用，无需手动触发。
        """
        # mha_fwd_with_mask: 输出 shape 与 q 相同 (B, H, Sq, d)
        self.register_fake_for(
            "mha_fwd_with_mask",
            lambda q, k, v, mask: torch.empty_like(q),
        )

        # pack_and_prepare_b1: 返回 Tensor[]，各输出 shape 依赖输入
        def _pack_fake(h_s, h_c, h_i, s_len, c_len, i_len,
                       static_cos, static_sin, S_max, S_mask):
            D  = h_s.shape[-1]
            hd = static_cos.shape[-1]
            return [
                torch.empty(1, S_max,  D,  dtype=h_s.dtype,         device=h_s.device),  # h_dense
                torch.empty(1, S_max,  1, hd, dtype=static_cos.dtype, device=h_s.device),  # cos
                torch.empty(1, S_max,  1, hd, dtype=static_sin.dtype, device=h_s.device),  # sin
                torch.empty(1, 1, S_mask, S_mask, dtype=h_s.dtype,    device=h_s.device),  # attn_mask
                torch.empty(s_len + c_len + i_len, dtype=torch.bool,   device=h_s.device),  # item_mask
                torch.empty(2, dtype=torch.int64,                       device=h_s.device),  # offsets
            ]
        self.register_fake_for("pack_and_prepare_b1", _pack_fake)

        # jagged_pool_and_collect: 返回 Tensor[]
        def _jagged_fake(pool_values, pool_lengths, pool_val_splits, pool_len_splits,
                         n_pooling, reduce_mode, ones_cache):
            D        = pool_values.shape[-1]
            out_rows = pool_lengths.shape[0]  # 保守近似（实际 ≤ 这个值）
            return [
                torch.empty(out_rows, D,   dtype=pool_values.dtype, device=pool_values.device),
                torch.empty(out_rows,      dtype=torch.int32,       device=pool_values.device),
                torch.empty(n_pooling + 1, dtype=torch.int64,       device=pool_values.device),
                torch.empty(n_pooling + 1, dtype=torch.int64,       device=pool_values.device),
            ]
        self.register_fake_for("jagged_pool_and_collect", _jagged_fake)
