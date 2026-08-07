"""
examples/custom_mask_demo.py — 自定义 mask 用法示例

mha_fwd_with_mask 的核心能力是支持**任意 bf16 加法 mask**（0=可见 / -inf=屏蔽），
这是 PyTorch SDPA FlashAttention 后端做不到的（它只支持 causal）。

本示例构造 4 种典型 mask 并逐一与 SDPA 校验：
  1. causal mask       因果（下三角可见）
  2. sliding window    滑动窗口（只看最近 w 个 token）
  3. padding mask      变长序列 padding 屏蔽
  4. random sparse     随机稀疏屏蔽（如推荐场景的 item 级屏蔽）

运行：
    python examples/custom_mask_demo.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ops import ops  # noqa: E402

NEG_INF = float("-inf")


def make_causal_mask(B, Sq, Sk):
    """下三角可见（Sq <= Sk 时对齐右下角，常用于 decode）。"""
    i = torch.arange(Sq, device="cuda").view(Sq, 1)
    j = torch.arange(Sk, device="cuda").view(1, Sk)
    visible = j <= i + (Sk - Sq)
    return torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16).masked_fill(
        ~visible, NEG_INF)


def make_sliding_window_mask(B, Sq, Sk, window):
    """每个 query 只看最近 window 个 key（含自身）。"""
    i = torch.arange(Sq, device="cuda").view(Sq, 1)
    j = torch.arange(Sk, device="cuda").view(1, Sk)
    diag_offset = Sk - Sq
    visible = (j <= i + diag_offset) & (j > i + diag_offset - window)
    return torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16).masked_fill(
        ~visible, NEG_INF)


def make_padding_mask(B, Sq, Sk, valid_lens):
    """变长序列：每个 batch 只有前 valid_lens[b] 个 key 可见。"""
    j = torch.arange(Sk, device="cuda").view(1, Sk)
    visible = j < valid_lens.view(B, 1)                    # (B, Sk)
    return torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16).masked_fill(
        ~visible.view(B, 1, 1, Sk), NEG_INF)


def make_random_mask(B, Sq, Sk, ratio=0.3):
    """随机屏蔽 ratio 比例的位置（保证第 0 列可见，避免整行屏蔽）。"""
    drop = torch.rand(B, 1, Sq, Sk, device="cuda") < ratio
    drop[..., 0] = False
    return torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16).masked_fill(
        drop, NEG_INF)


def main():
    if not torch.cuda.is_available() or not ops.is_available():
        print(f"CUDA 或算子库不可用: {ops.load_error()}")
        return

    torch.manual_seed(0)
    B, H, Sq, Sk, d = 2, 8, 512, 512, 128
    q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)

    cases = {
        "causal":               make_causal_mask(B, Sq, Sk),
        "sliding_window(w=64)": make_sliding_window_mask(B, Sq, Sk, window=64),
        "padding":              make_padding_mask(
                                    B, Sq, Sk,
                                    valid_lens=torch.tensor([Sk, Sk // 2], device="cuda")),
        "random_sparse(30%)":   make_random_mask(B, Sq, Sk, ratio=0.3),
    }

    for name, mask in cases.items():
        out = ops.mha_fwd_with_mask(q, k, v, mask)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        diff = (out.float() - ref.float()).abs().max().item()
        tag = "PASS ✓" if diff < 0.02 else "FAIL ✗"
        print(f"  {name:<24} max|diff| = {diff:.6f}   [{tag}]")
        assert diff < 0.02, f"{name} 与 SDPA 偏差过大"


if __name__ == "__main__":
    main()
