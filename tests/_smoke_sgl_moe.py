"""sglang fused_moe (Triton) 移植版冒烟 + 数值对齐 + 快速性能对比。

被测对象为 benchmark/sglang_triton_moe 自包含移植包（无 sglang 安装、
无 3rd/sglang 依赖）；tuning 见 benchmark/tune_sglang_moe.py，
正式对比见 benchmark/benchmark_fuse_moe_vs_sglang.py。
"""
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_CO = os.path.dirname(_THIS)
sys.path.insert(0, os.path.join(_CO, "benchmark"))
sys.path.insert(0, os.path.dirname(_CO))  # custom_ops 包父目录

import torch  # noqa: E402

from custom_ops import ops  # noqa: E402
from sglang_triton_moe import fused_experts  # noqa: E402

print("[smoke] import OK")

torch.manual_seed(0)
dev = "cuda"
S, H_, I, E, K = 512, 2048, 1024, 8, 2

x = torch.randn(S, H_, device=dev, dtype=torch.bfloat16)
w1 = torch.randn(E, 2 * I, H_, device=dev, dtype=torch.bfloat16) * 0.05
w2 = torch.randn(E, H_, I, device=dev, dtype=torch.bfloat16) * 0.05
ids = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
wt = torch.rand(S, K, device=dev, dtype=torch.float32)

y_ours = ops.fuse_moe(x, w1, w2, ids, wt).float()
print("[smoke] ours done")

print("[smoke] calling sglang port (first call incl. Triton JIT)...")
import time as _t
_t0 = _t.time()
y_sgl = fused_experts(x, w1, w2, wt, ids).float()
print(f"[smoke] sglang done in {_t.time() - _t0:.1f}s")

d = (y_ours - y_sgl).abs()
print(f"[smoke] S={S} H={H_} I={I} E={E} K={K}: "
      f"max_abs={d.max().item():.4f} mean_abs={d.mean().item():.6f}")

# 三方对比：torch 参考判断是精度差异还是语义错位
ref = torch.zeros(S, H_, device=dev, dtype=torch.float32)
for k in range(K):
    for e in range(E):
        sel = (ids[:, k] == e)
        if sel.sum() == 0:
            continue
        xe = x[sel].float()
        g = xe @ w1[e, :I].float().t()
        u = xe @ w1[e, I:].float().t()
        ref[sel] += (torch.nn.functional.silu(g) * u) @ w2[e].float().t() * wt[sel, k:k+1]
d_o = (y_ours - ref).abs()
d_s = (y_sgl - ref).abs()
print(f"[smoke] vs ref: ours max={d_o.max().item():.4f} mean={d_o.mean().item():.6f} | "
      f"sgl max={d_s.max().item():.4f} mean={d_s.mean().item():.6f}")

assert d_o.max().item() < 0.3 and d_s.max().item() < 0.3, "MISMATCH vs ref"
print("[smoke] NUMERIC OK (both match fp32 ref within bf16 noise)")


# ── 性能对比（稳态，CUDA event）────────────────────────────────────
def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


for (S2, H2, I2, E2, K2) in [(512, 2048, 1024, 8, 2), (4096, 2048, 1024, 8, 2),
                             (16384, 2048, 1024, 8, 2)]:
    x2 = torch.randn(S2, H2, device=dev, dtype=torch.bfloat16)
    w12 = torch.randn(E2, 2 * I2, H2, device=dev, dtype=torch.bfloat16) * 0.05
    w22 = torch.randn(E2, H2, I2, device=dev, dtype=torch.bfloat16) * 0.05
    ids2 = torch.randint(0, E2, (S2, K2), device=dev, dtype=torch.int32)
    wt2 = torch.rand(S2, K2, device=dev, dtype=torch.float32)
    t_ours = bench(lambda: ops.fuse_moe(x2, w12, w22, ids2, wt2))
    t_sgl = bench(lambda: fused_experts(x2, w12, w22, wt2, ids2))
    flops = 2 * S2 * K2 * I2 * H2 * 3  # 2 GEMM × 2 (mul-add) × 3 (silu·mul 后的 down)
    tf_ours = flops / (t_ours * 1e-6) / 1e12
    print(f"[bench] S={S2:5d}: ours={t_ours:7.1f}us ({tf_ours:5.1f} TF) "
          f"sglang={t_sgl:7.1f}us | speedup={t_sgl / t_ours:.2f}x")
print("[smoke] ALL OK")
