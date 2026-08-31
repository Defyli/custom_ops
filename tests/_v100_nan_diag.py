"""V100 NaN 归属诊断：kernel 输出 vs SDPA 参考的空 softmax 行语义。"""
import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch
from custom_ops import ops

torch.manual_seed(0)
d = torch.float16

# 场景 1：整行 -inf
q = torch.randn(1, 1, 8, 64, device="cuda", dtype=d)
k = torch.randn(1, 1, 64, 64, device="cuda", dtype=d)
v = torch.randn(1, 1, 64, 64, device="cuda", dtype=d)
m = torch.zeros(1, 1, 8, 64, device="cuda", dtype=d)
m[:, :, 3, :] = float("-inf")  # 第 3 行全屏蔽

out = ops.mha_fwd_with_mask(q, k, v, m)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m)
print("full_row:  out[3,:4] =", out[0, 0, 3, :4].float().tolist())
print("full_row:  ref[3,:4] =", ref[0, 0, 3, :4].float().tolist())
print("full_row:  out has nan:", torch.isnan(out).any().item(),
      "| ref has nan:", torch.isnan(ref).any().item())

# 场景 2：Sk=1（autopad 到 8），某行唯一列被屏蔽
q2 = torch.randn(1, 1, 8, 64, device="cuda", dtype=d)
k2 = torch.randn(1, 1, 1, 64, device="cuda", dtype=d)
v2 = torch.randn(1, 1, 1, 64, device="cuda", dtype=d)
m2 = torch.zeros(1, 1, 8, 1, device="cuda", dtype=d)
m2[:, :, 5, 0] = float("-inf")  # 第 5 行唯一列屏蔽 → 空行

out2 = ops.mha_fwd_with_mask(q2, k2, v2, m2)
ref2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, attn_mask=m2)
print("sk1:       out2[5,:4]=", out2[0, 0, 5, :4].float().tolist())
print("sk1:       ref2[5,:4]=", ref2[0, 0, 5, :4].float().tolist())
print("sk1:       out2 nan:", torch.isnan(out2).any().item(),
      "| ref2 nan:", torch.isnan(ref2).any().item())

# 场景 3：手动参考（空行 → softmax 空集 → 显式置 0，FA 官方语义）
def manual_ref(q, k, v, m):
    s = (q.float() @ k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5) + m.float()
    mx = s.max(-1, keepdim=True)[0]
    mx = torch.where(torch.isinf(mx) & (mx < 0), torch.zeros_like(mx), mx)  # 空行 lse_max 保护
    p = torch.exp(s - mx)
    p = torch.where(torch.isinf(s), torch.zeros_like(p), p)  # -inf 位置权重 0
    denom = p.sum(-1, keepdim=True)
    o = (p @ v.float()) / torch.where(denom == 0, torch.ones_like(denom), denom)
    return o.to(q.dtype)

mr = manual_ref(q, k, v, m)
print("full_row:  manual out==manual ref:",
      torch.allclose(out.float(), mr.float(), atol=1e-2))
mr2 = manual_ref(q2, k2, v2, m2)
print("sk1:       out2==manual ref2:",
      torch.allclose(out2.float(), mr2.float(), atol=1e-2))
print("DIAG DONE")
