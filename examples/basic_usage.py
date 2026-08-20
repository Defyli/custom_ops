"""
examples/basic_usage.py — mha_fwd_with_mask 最小使用示例

演示：
  1. 导入并加载算子（首次 import 自动 JIT 编译，约 30~60s；之后秒级加载）
  2. 构造 q/k/v/mask 调用算子
  3. 与 PyTorch SDPA 做数值校验

运行：
    python examples/basic_usage.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# 本仓库根目录 custom_ops/ 自身就是 Python 包（根下有 __init__.py），
# 需把它的上一级目录加入 sys.path，`import custom_ops` 才能命中包本体。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../custom_ops（仓库根 = 包根）
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops  # noqa: E402


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用，示例无法运行。")
        return
    if not ops.is_available():
        print(f"算子库加载失败: {ops.load_error()}")
        return

    torch.manual_seed(0)

    B, H, Sq, Sk, d = 2, 8, 512, 512, 64
    q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)

    # 加法 mask：(B, 1, Sq, Sk)，0 = 可见，-inf = 屏蔽
    mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16)

    # 调用自定义算子
    out = ops.mha_fwd_with_mask(q, k, v, mask)

    # 与 SDPA 校验
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    max_diff = (out.float() - ref.float()).abs().max().item()

    print(f"out shape : {tuple(out.shape)}  dtype: {out.dtype}")
    print(f"max |diff| vs SDPA = {max_diff:.6f}")
    assert max_diff < 0.02, "与 SDPA 偏差过大！"
    print("PASS ✓")


if __name__ == "__main__":
    main()
