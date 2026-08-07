"""
custom_ops/recsys.py — RecsysOps：推荐系统三大核心 CUDA 算子库

算子列表
--------
- mha_fwd_with_mask      Flash Attention 2 前向，支持任意 bf16 加法 mask

快速使用
--------
    from custom_ops import ops                  # 推荐：直接使用包级单例
    from custom_ops import RecsysOps, ops       # 也可导入类名

    if ops.is_available():
        out  = ops.mha_fwd_with_mask(q, k, v, mask)
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
        "mha_fwd_with_mask"
    ]
    optional_ops = []

    # 编译缓存回退目录（优先使用 TORCH_EXTENSIONS_DIR 环境变量）
    build_dir_fallback = "./torch_extensions"

    def get_sources(self) -> list:
        """源文件列表：recsys_bindings.cpp + FA/pack/jagged CUDA kernel。"""
        csrc   = os.path.join(_THIS_DIR, "csrc")
        return [
            os.path.join(csrc, "recsys_bindings.cpp"),
            os.path.join(csrc, "fa",     "fa_fwd_op.cu"),
        ]

    def get_include_dirs(self) -> list:
        """头文件目录：thirdparty（cutlass/cute）+ csrc 各子目录 + 框架宏目录。"""
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,                          # cutlass/cute 头文件
            csrc,                                # custom_ops_macros.h
            os.path.join(csrc, "fa"),            # fa_fwd_op.h 等
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

