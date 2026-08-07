"""
custom_ops/example/__init__.py — 示例：使用 CustomOps 框架注册 FA2 算子

展示如何只用几十行 Python 代码，基于 CustomOps 基类构建一个完整的算子插件库。

用法
----
    from custom_ops.example import ops

    if ops.is_available():
        out = ops.mha_fwd_with_mask(q, k, v, mask)
    else:
        # 退化到 SDPA
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
"""

from __future__ import annotations

import os
import torch

from custom_ops import CustomOps


# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_OPS_DIR = os.path.dirname(_THIS_DIR)   # custom_ops/ 根目录


# ─────────────────────────────────────────────────────────────────────────────
# FaOps — 继承 CustomOps，只需覆盖配置属性和两个方法
# ─────────────────────────────────────────────────────────────────────────────

class FaOps(CustomOps):
    """
    Tetuan 全量算子插件库（示例）。

    命名空间  : fa_ops
    编译产物  : fa_ops_kernel.so
    注册算子  :
      - mha_fwd_with_mask      （必需）Flash Attention 2 前向
      - pack_and_prepare_b1    （必需）融合 pack/RoPE/attn_mask 算子
      - jagged_pool_and_collect（必需）融合 jagged pooling 算子
    """

    # ── 配置 ─────────────────────────────────────────────────────────────────
    namespace   = "fa_ops"           # torch.ops 命名空间，对应 bindings.cpp 中的宏定义
    so_name     = "fa_ops_kernel"    # 编译产物名（不含 .so）
    required_ops = [
        "mha_fwd_with_mask",
        "pack_and_prepare_b1",
        "jagged_pool_and_collect",
    ]
    optional_ops = []

    # 编译缓存回退目录（优先使用 TORCH_EXTENSIONS_DIR 环境变量）
    build_dir_fallback = "/workdir/tetuan_cache/torch_extensions"

    def get_sources(self) -> list:
        """源文件列表：bindings.cpp + FA kernel + pack kernel + jagged kernel。"""
        csrc   = os.path.join(_THIS_DIR, "csrc")
        fa     = os.path.join(_CUSTOM_OPS_DIR, "csrc", "fa")
        pack   = os.path.join(_CUSTOM_OPS_DIR, "csrc", "pack")
        jagged = os.path.join(_CUSTOM_OPS_DIR, "csrc", "jagged")
        return [
            os.path.join(csrc,   "bindings.cpp"),
            os.path.join(fa,     "fa_fwd_op.cu"),
            os.path.join(pack,   "pack_and_prepare.cu"),
            os.path.join(jagged, "jagged_forward.cu"),
        ]

    def get_include_dirs(self) -> list:
        """头文件目录：thirdparty（cutlass/cute） + csrc 各子目录 + 框架宏目录。"""
        csrc       = os.path.join(_CUSTOM_OPS_DIR, "csrc")
        thirdparty = os.path.join(_CUSTOM_OPS_DIR, "thirdparty")
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
        mask    : (B, 1, Sq, Sk) bfloat16 CUDA，加法 mask

        限制
        ----
        - d ∈ {64, 128}
        - H % Hk == 0（支持 GQA）
        """
        return torch.ops.fa_ops.mha_fwd_with_mask(q, k, v, mask)

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

        参数
        ----
        h_s, h_c, h_i : (n, D) bf16 CUDA，ubc/ctx/item token 特征
        s_len, c_len, i_len : int，各段有效长度（Python 侧先 .item() 提取）
        static_cos, static_sin : (1, S_max, 1, head_dim) bf16 CUDA，预计算 RoPE 表
        S_max : int，固定序列长度
        S_mask : int，attn_mask 输出边长（>= S_max）

        返回
        ----
        h_dense, cos, sin, attn_mask, item_mask, offsets
        """
        outs = torch.ops.fa_ops.pack_and_prepare_b1(
            h_s, h_c, h_i,
            int(s_len), int(c_len), int(i_len),
            static_cos, static_sin, S_max, S_mask,
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

        参数
        ----
        pool_values : (total_pool_val_rows, D) — 所有 pooling 特征 values 拼接
        pool_lengths : (total_pool_len_rows,) int32 — lengths 拼接
        pool_val_splits : list[int]，len=n_pooling+1，values 行偏移
        pool_len_splits : list[int]，len=n_pooling+1，lengths 行偏移
        n_pooling : int，pooling 特征数量
        reduce_mode : int，0=mean / 1=sum
        ones_cache : (max_n,) int32 CUDA，预分配全 1 缓存，避免 alloc

        返回
        ----
        pool_values_out, pool_lengths_out, out_val_splits, out_len_splits
        """
        outs = torch.ops.fa_ops.jagged_pool_and_collect(
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
        # h_dense (1,S_max,D), cos/sin (1,S_max,1,hd), attn_mask (1,1,S_mask,S_mask),
        # item_mask (s_len+c_len+i_len,), offsets (2,)
        def _pack_fake(h_s, h_c, h_i, s_len, c_len, i_len,
                       static_cos, static_sin, S_max, S_mask):
            D  = h_s.shape[-1]
            hd = static_cos.shape[-1]
            return [
                torch.empty(1, S_max,  D,  dtype=h_s.dtype,        device=h_s.device),  # h_dense
                torch.empty(1, S_max,  1, hd, dtype=static_cos.dtype, device=h_s.device),  # cos
                torch.empty(1, S_max,  1, hd, dtype=static_sin.dtype, device=h_s.device),  # sin
                torch.empty(1, 1, S_mask, S_mask, dtype=h_s.dtype,   device=h_s.device),  # attn_mask
                torch.empty(s_len + c_len + i_len, dtype=torch.bool,  device=h_s.device),  # item_mask
                torch.empty(2, dtype=torch.int64, device=h_s.device),                      # offsets
            ]
        self.register_fake_for("pack_and_prepare_b1", _pack_fake)
        # jagged_pool_and_collect: 返回 Tensor[]
        # 输出 rows 是所有 pooling 特征 sample 总数，此处用 pool_lengths 行数近似
        def _jagged_fake(pool_values, pool_lengths, pool_val_splits, pool_len_splits,
                         n_pooling, reduce_mode, ones_cache):
            D        = pool_values.shape[-1]
            out_rows = pool_lengths.shape[0]  # 保守近似（实际 ≤ 这个值）
            return [
                torch.empty(out_rows, D,  dtype=pool_values.dtype,  device=pool_values.device),
                torch.empty(out_rows,     dtype=torch.int32,        device=pool_values.device),
                torch.empty(n_pooling + 1, dtype=torch.int64,       device=pool_values.device),
                torch.empty(n_pooling + 1, dtype=torch.int64,       device=pool_values.device),
            ]
        self.register_fake_for("jagged_pool_and_collect", _jagged_fake)


# ─────────────────────────────────────────────────────────────────────────────
# 模块级单例：import 时自动触发加载（含 fake 注册）
# ─────────────────────────────────────────────────────────────────────────────

#: 示例算子库的全局实例，import 时自动完成编译加载及 fake 注册。
ops: FaOps = FaOps().load()
