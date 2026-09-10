"""
examples/fuse_moe_demo.py — fused MoE 前向算子使用示例

演示 Mixtral/DeepSeek 风格 MoE FFN 层的融合前向：
    y[s] = Σ_j topk_scale[s,j] * (Down_ej @ silu(GateUp_ej @ x[s]))

问题背景：MoE 前向 = gather → 逐 expert 小 GEMM → 激活 → down GEMM →
scatter 加权归约，PyTorch eager 需 5+ 个 kernel 与多份中间显存；
torch.compile 无法穿透数据依赖的 expert 分发。fused_moe 用 4 个 kernel 的
流水线（count → gate/up 配对 GEMM+激活 → down GEMM → reduce）在 expert 有序
compact 布局上一次完成，权重与激活直入 bf16 tensor core（fp32 累加）。

演示：
  1. Mixtral 风格 MoE 输入构造（E=8 experts, top-k=2 路由）
  2. 算子调用（ops.fuse_moe）
  3. 数值校验：vs 向量化 PyTorch 参考（fp32 金标准）
  4. 性能对比：fused vs PyTorch eager（CUDA event 计时）

运行：
    python examples/fuse_moe_demo.py

约束（op 层校验，违反会报错）：hidden/intermediate 需为 64 的倍数，
topk_ids ∈ [0, E) int32，topk_scale 为 fp32。
"""

import os
import sys

import torch
import torch.nn.functional as F

# 本仓库根目录 custom_ops/ 自身就是 Python 包（根下有 __init__.py），
# 需把它的上一级目录加入 sys.path，`import custom_ops` 才能命中包本体。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops  # noqa: E402


def moe_eager_ref(x, w1, w2, topk_ids, topk_scale):
    # PyTorch eager 参考：per-expert 分组 GEMM + top-k 加权归约（与真实
    # MoE 模型写法同构；dtype 跟随输入——fp32 调用即金标准，bf16 即真实路径）
    S, H = x.shape
    E = w1.shape[0]
    I = w2.shape[2]
    K = topk_ids.shape[1]
    T = S * K
    e_flat = topk_ids.reshape(-1)                      # (T,) 每行的目标 expert
    s_flat = topk_scale.reshape(-1).to(x.dtype)       # (T,) top-k 权重
    xg = x[:, None, :].expand(S, K, H).reshape(T, H)  # (T, H) gather 激活
    out = torch.zeros(S, H, device=x.device, dtype=x.dtype)
    for e in range(E):
        idx = (e_flat == e).nonzero().squeeze(1)       # 该 expert 的 compact 行
        if idx.numel() == 0:
            continue
        gu = xg[idx] @ w1[e].t()                       # (n_e, 2I) gate_up GEMM
        g, u = gu[:, :I], gu[:, I:]
        h = F.silu(g.float()).to(x.dtype) * u          # 激活（fp32 计算后回落）
        y = h @ w2[e].t()                              # (n_e, H) down GEMM
        out.index_add_(0, idx // K, y * s_flat[idx].unsqueeze(1))  # 加权归约
    return out



def bench(fn, iters=20, warmup=5):
    """CUDA event 计时，返回中位耗时（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用，示例无法运行。")
        return
    if not ops.is_available():
        print(f"算子库加载失败: {ops.load_error()}")
        return

    torch.manual_seed(0)

    # ── Mixtral 风格 MoE FFN（缩小版）：S token, H=2048, I=1024, 8 专家 top-2 ──
    S, H, I, E, K = 1024, 2048, 1024, 8, 2
    x = (torch.randn(S, H, device="cuda") * 0.5).to(torch.bfloat16)          # (S, H) 激活
    w1 = (torch.randn(E, 2 * I, H, device="cuda") * 0.05).to(torch.bfloat16)  # (E, 2I, H) gate_up
    w2 = (torch.randn(E, H, I, device="cuda") * 0.05).to(torch.bfloat16)      # (E, H, I) down

    # 路由：topk_ids (S,K) int32 ∈ [0,E)；topk_scale (S,K) fp32（softmax 归一权重）
    logits = torch.randn(S, E, device="cuda")
    topk_scale, topk_idx = torch.topk(logits, K, dim=-1)
    topk_scale = torch.softmax(topk_scale, dim=-1)
    topk_ids = topk_idx.to(torch.int32)

    # ── 1. 调用融合算子 ─────────────────────────────────────────────────────
    out = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)

    # ── 2. 数值校验（fp32 金标准） ──────────────────────────────────────────
    out_ref = moe_eager_ref(x.float(), w1.float(), w2.float(), topk_ids, topk_scale)
    out_eager = moe_eager_ref(x, w1, w2, topk_ids, topk_scale)        # bf16 eager

    def rel(a, ref):
        return ((a.float() - ref).abs() / ref.abs().clamp_min(1e-2)).mean().item()

    hdr = "%16s%12s%14s" % ("", "fused_moe", "eager(bf16)")
    print(f"输出: {tuple(out.shape)} {out.dtype}   （{S} tokens × {E} experts × top-{K}）")
    print(hdr)
    print("%16s%12.2e%14.2e" % ("mean rel-err", rel(out, out_ref), rel(out_eager, out_ref)))
    assert rel(out, out_ref) < 0.05, "fused_moe 精度异常！"

    # ── 3. 性能对比 ────────────────────────────────────────────────────────
    t_fused = bench(lambda: ops.fuse_moe(x, w1, w2, topk_ids, topk_scale))
    t_eager = bench(lambda: moe_eager_ref(x, w1, w2, topk_ids, topk_scale))
    flops = S * K * 6 * H * I
    print("%16s%12.3f%14.3f   加速 %.1fx" % ("耗时(ms)", t_fused, t_eager, t_eager / t_fused))
    print("%16s%10.0f TFLOPS" % ("有效算力", flops / t_fused / 1e9))
    print("PASS ✓")


if __name__ == "__main__":
    main()
