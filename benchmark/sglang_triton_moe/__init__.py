# SPDX-License-Identifier: Apache-2.0
"""sglang Triton fused MoE（bf16/fp16）自包含移植——性能对比基准。

不依赖 sglang/sgl-kernel 安装，也不依赖 3rd/sglang checkout：
kernel / 编排 / config 查找 / tuned 参数全部内置于本包，
移植出处与裁剪范围见各模块头注释。

用法：
    from sglang_triton_moe import fused_experts
    y = fused_experts(x, w1, w2, topk_weights, topk_ids)          # tuned 查表
    y = fused_experts(x, w1, w2, wt, ids, use_tuned_config=False)  # default 启发式
    y = fused_experts(x, w1, w2, wt, ids, up_config=cfg)           # 显式注入（tuning）
"""
from .config import get_default_config, get_moe_configs, try_get_optimal_moe_config
from .runner import fused_experts, moe_align_block_size

__all__ = [
    "fused_experts",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
    "get_moe_configs",
    "get_default_config",
]
