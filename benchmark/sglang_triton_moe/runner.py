# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sglang fused_experts (Triton MoE, bf16) 的自包含移植——编排层。

从 sgl-project/sglang `python/sglang/srt/layers/moe/moe_runner/triton_utils/
fused_moe.py` 的 `_fused_moe_kernel_sequence` 提取非量化路径的五段式：

    moe_align → gemm1(up+gate) → silu_and_mul → gemm2(down) → moe_sum_reduce

与上游的差异：
  - moe_align_block_size：上游在 CUDA 上调用 sgl_kernel 的 AOT 单 kernel；
    此处为语义对齐的 torch 参考实现（多 kernel launch，~10µs 量级，
    E+1 桶空间，"+1 offset" 约定与 AOT 版 buffer contract 一致）。
  - silu_and_mul：上游为 JIT CUDA kernel，此处为等价 Triton 实现
    （见 kernels.py 注释）。
  - config 注入：新增 up_config/down_config 显式参数与 use_tuned_config
    开关，取代上游的 override_config 上下文（tuning 驱动据此扫参，
    无需 monkey-patch）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from . import config as _config
from .kernels import (
    invoke_fused_moe_kernel,
    moe_sum_reduce_triton,
    silu_and_mul,
)


def _moe_align_torch(topk_ids: torch.Tensor, num_experts: int, block_size: int,
                     sorted_token_ids: torch.Tensor,
                     expert_ids: torch.Tensor,
                     num_tokens_post_pad: torch.Tensor) -> None:
    """moe_align_block_size 的 torch 参考实现（sgl_kernel AOT 版的替身）。

    约定与 AOT 版一致：bucket = eid + 1（bucket 0 为 filtered 占位，
    bucket ∈ [1, E]，桶表尺寸 E+1）；sorted_token_ids 的 pad 值 = numel
    （≥ num_valid_tokens，kernel 侧被 token_mask 屏蔽）；expert_ids 每
    block 一个（int32）。
    """
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    bucket = flat + 1
    order = torch.argsort(bucket, stable=True).to(torch.int64)
    cnt = torch.bincount(bucket, minlength=num_experts + 1)
    padded = (cnt + block_size - 1) // block_size * block_size
    excl = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())
    sorted_token_ids[:total] = numel
    # stable 排序后每个 token 的 bucket 内 rank
    sorted_bucket = bucket[order]
    ar = torch.arange(numel, device=flat.device)
    first = torch.full((num_experts + 1,), numel,
                       device=flat.device, dtype=torch.int64)
    first.scatter_reduce_(0, sorted_bucket, ar, reduce="amin", include_self=True)
    rank = ar - first[sorted_bucket]
    pos = excl[sorted_bucket] + rank
    sorted_token_ids[pos] = order.to(sorted_token_ids.dtype)
    nb = (total + block_size - 1) // block_size
    bidx = torch.arange(nb, device=flat.device, dtype=torch.int64)
    bkt = torch.searchsorted(torch.cumsum(padded, 0), bidx * block_size, right=True)
    expert_ids[:nb] = (bkt - 1).to(torch.int32)
    num_tokens_post_pad[0] = total


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """token 按专家对齐到 block_size 的倍数（buffer 尺寸按上游约定）。"""
    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    _moe_align_torch(topk_ids, num_experts, block_size,
                     sorted_ids, expert_ids, num_tokens_post_pad)
    return sorted_ids, expert_ids, num_tokens_post_pad


def _resolve_configs(
    w1: torch.Tensor,
    w2: torch.Tensor,
    M: int,
    topk: int,
    up_config: Optional[Dict[str, Any]],
    down_config: Optional[Dict[str, Any]],
    use_tuned_config: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """显式注入 > tuned 查表 > default 启发式；down 的 BLOCK_SIZE_M
    必须与 up 一致（两段 GEMM 共享一次 align sort，上游同款约束）。"""
    if up_config is not None:
        cfg = dict(up_config)
        dcfg = dict(down_config) if down_config is not None else None
    elif use_tuned_config:
        dtype = None  # bf16/fp16 非量化：文件名无 dtype 后缀
        cfg, (dcfg, _) = _config.try_get_optimal_moe_config(
            w1.shape, w2.shape, topk, dtype, M, return_down_config=True)
    else:
        dtype = None
        E, _, N = w2.shape
        cfg = _config.get_default_config(M, E, N, w1.shape[2], topk, dtype)
        dcfg = None
    if dcfg is not None and dcfg["BLOCK_SIZE_M"] != cfg["BLOCK_SIZE_M"]:
        dcfg["BLOCK_SIZE_M"] = cfg["BLOCK_SIZE_M"]
    return cfg, dcfg


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    up_config: Optional[Dict[str, Any]] = None,
    down_config: Optional[Dict[str, Any]] = None,
    use_tuned_config: bool = True,
) -> torch.Tensor:
    """bf16/fp16 路径的 fused_experts（上游 fused_experts_impl 的非量子集）。

    Args:
        hidden_states: (S, H)
        w1: (E, 2I, H)  gate/up 融合权重
        w2: (E, H, I)   down 投影权重
        topk_weights: (S, topk) float32 路由权重
        topk_ids: (S, topk) int32 专家 id
        up_config / down_config: 显式 tile 配置（tuning 扫参用；均为 None
            且 use_tuned_config=False 时走 default 启发式）
    Returns:
        (S, H)
    """
    assert activation == "silu", "port 仅验证 silu（基准路径）"
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape  # N = 2I
    topk = topk_ids.shape[1]
    compute_type = (tl.bfloat16 if hidden_states.dtype == torch.bfloat16
                    else tl.float16)

    config, down_config = _resolve_configs(
        w1, w2, num_tokens, topk, up_config, down_config, use_tuned_config)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    total_tokens = num_tokens * topk

    # ── gemm1：up + gate ───────────────────────────────────────────
    intermediate_cache1 = torch.empty(
        (total_tokens, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        None,  # gemm1 不乘路由权重（upstream: apply_router_weight_on_input=False）
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=topk,
        config=config,
        compute_type=compute_type,
        filter_expert=True,
    )

    # ── silu(gate) * up（sorted 序保持）────────────────────────────
    intermediate_cache2 = torch.empty(
        (total_tokens, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    silu_and_mul(
        intermediate_cache1.view(-1, N),
        intermediate_cache2,
        expert_ids=topk_ids.view(-1),
        expert_step=1,
    )
    del intermediate_cache1

    # ── gemm2：down，乘路由权重，写 (S, topk, H) ────────────────────
    intermediate_cache3 = torch.empty(
        (num_tokens, topk, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=down_config or config,
        compute_type=compute_type,
        filter_expert=True,
    )
    del intermediate_cache2

    # ── topk 维求和 ────────────────────────────────────────────────
    out_hidden_states = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(intermediate_cache3, out_hidden_states, 1.0)
    return out_hidden_states
