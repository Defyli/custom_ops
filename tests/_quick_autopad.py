"""快速验证：自动 pad（Sk%8!=0）+ 基本功能（本地冒烟，4090D/sm89）。"""
import time
import torch

t0 = time.time()
from custom_ops import ops  # noqa: E402
print(f"import+load: {time.time()-t0:.1f}s, available={ops.is_available()}")

torch.manual_seed(0)

# ── 1. 自动 pad 正确性：Sk%8!=0 的 ragged Sk 全扫 ──
for Sk in [7, 9, 55, 100, 101, 777, 1032 - 1, 2053, 4099]:
    for d in (64, 128):
        q = torch.randn(1, 4, 128, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 4, Sk, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 4, Sk, d, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(1, 1, 128, Sk, device="cuda", dtype=torch.bfloat16)
        mask[..., torch.arange(Sk) % 7 == 0] = float("-inf")  # 随机屏蔽
        out = ops.mha_fwd_with_mask(q, k, v, mask)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        diff = (out.float() - ref.float()).abs().max().item()
        status = "PASS" if diff < 0.02 else "FAIL"
        print(f"[{status}] autopad Sk={Sk:5d} d={d}: max_diff={diff:.2e}")

# ── 2. mask 列数未对齐但 Sk 对齐（Sk_mask%8!=0 路径）──
Sk = 128
for extra in (1, 3, 5):
    q = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, Sk, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 4, Sk, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 64, Sk + extra, device="cuda", dtype=torch.bfloat16)
    mask[..., 100:] = float("-inf")
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    ref_mask = mask[..., :Sk]
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=ref_mask)
    diff = (out.float() - ref.float()).abs().max().item()
    status = "PASS" if diff < 0.02 else "FAIL"
    print(f"[{status}] mask-unaligned Sk_mask={Sk+extra}: max_diff={diff:.2e}")

# ── 3. 对齐主路径不受影响（零拷贝）──
q = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
mask = torch.zeros(2, 1, 512, 1024, device="cuda", dtype=torch.bfloat16)
mask[..., ::13] = float("-inf")
out = ops.mha_fwd_with_mask(q, k, v, mask)
ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
diff = (out.float() - ref.float()).abs().max().item()
print(f"[{'PASS' if diff < 0.02 else 'FAIL'}] aligned main path: max_diff={diff:.2e}")

print("ALL DONE")
