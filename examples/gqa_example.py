"""
examples/gqa_example.py — GQA（Grouped Query Attention）用法示例

mha_fwd_with_mask 原生支持 GQA：K/V 的头数 Hk 只需整除 Q 的头数 H，
无需像 SDPA 那样手动 repeat_interleave 扩展 K/V（省显存、省一次拷贝）。

运行：
    python examples/gqa_example.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ops import ops  # noqa: E402


def main():
    if not torch.cuda.is_available() or not ops.is_available():
        print(f"CUDA 或算子库不可用: {ops.load_error()}")
        return

    torch.manual_seed(0)

    # GQA：16 个 query head 共享 4 组 K/V head
    B, H, Hk, Sq, Sk, d = 2, 16, 4, 1024, 1024, 128
    q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16)

    # 自定义算子：直接传入未扩展的 K/V
    out = ops.mha_fwd_with_mask(q, k, v, mask)

    # SDPA 参考：需先把 K/V 扩展到 H 个头
    ref = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(H // Hk, dim=1),
        v.repeat_interleave(H // Hk, dim=1),
        attn_mask=mask,
    )

    diff = (out.float() - ref.float()).abs().max().item()
    print(f"GQA H={H} Hk={Hk}:  out {tuple(out.shape)}, max|diff| = {diff:.6f}")
    assert diff < 0.02
    print("PASS ✓")


if __name__ == "__main__":
    main()
