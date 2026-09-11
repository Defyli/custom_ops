"""
examples/swiglu_demo.py — SwiGLU 融合算子使用示例

演示 Llama 风格 FFN 的 SwiGLU 激活层：
    y = silu(x @ Wg^T) * (x @ Wu^T)

问题背景：eager 实现需要 GEMM 物化 (M, 2N) 的 gate_up 中间量，再由独立
kernel 完成 silu*mul——多 2×M×N×2B 的 DRAM 往返。swiglu 算子采用配对
N-tile 机制（完全独立实现，不依赖 fuse_moe）：同一 CTA 用双累加器同时
算 gate/up 两个面板（共享同一 X tile），激活直接在 fp32 累加器上完成后
落盘，单 kernel 零中间量。

演示：
  1. Llama-7B FFN 形状输入构造（M=2048, K=4096, N=11008）
  2. 算子调用（ops.swiglu）
  3. 数值校验：vs fp32 参考
  4. 性能对比：swiglu vs PyTorch eager（GEMM + silu*mul 两段式）

运行：
    python examples/swiglu_demo.py

约束（op 层校验）：K % 64 == 0，N % 64 == 0（weight 行数 = 2N），
M 任意正整数。
"""

import os
import sys

import torch
import torch.nn.functional as F

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops  # noqa: E402


def swiglu_eager(x, w):
    """PyTorch eager 参考：GEMM 物化 gate_up (M, 2N) → 独立激活。"""
    n2 = w.shape[0]
    n = n2 // 2
    gu = x @ w.t()
    return F.silu(gu[:, :n]) * gu[:, n:]


def bench(fn, iters=30, warmup=10):
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

    # ── Llama-7B FFN 形状：2048 token, hidden 4096, intermediate 11008 ──────
    M, K, N = 2048, 4096, 11008
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)   # (M, K) 激活
    w = (torch.randn(2 * N, K, device="cuda") * 0.02).to(torch.bfloat16)  # (2N, K) gate 在前 up 在后

    # ── 1. 调用融合算子 ─────────────────────────────────────────────────────
    y = ops.swiglu(x, w)

    # ── 2. 数值校验（fp32 金标准） ──────────────────────────────────────────
    y_ref = swiglu_eager(x.float(), w.float())
    y_eager = swiglu_eager(x, w)

    def rel(a, ref):
        return ((a.float() - ref).abs() / ref.abs().clamp_min(1e-2)).mean().item()

    print(f"输出: {tuple(y.shape)} {y.dtype}   （M={M}, K={K}, N={N}，Llama-7B FFN）")
    hdr = "%16s%12s%14s" % ("", "swiglu", "eager(bf16)")
    print(hdr)
    print("%16s%12.2e%14.2e" % ("mean rel-err", rel(y, y_ref), rel(y_eager, y_ref)))
    assert rel(y, y_ref) < 0.05, "swiglu 精度异常！"

    # ── 3. 性能对比 ────────────────────────────────────────────────────────
    # 全局时钟热身：DVFS 从 idle（~210MHz）爬到 boost（~3GHz）需要数十 ms
    # 的持续负载，缺失时首个基准会慢 ~8%（4090D 实测）。
    warm = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(300):
        warm = warm @ warm
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()

    t_fused = bench(lambda: ops.swiglu(x, w))
    t_eager = bench(lambda: swiglu_eager(x, w))
    flops = 4 * M * N * K  # gate + up 两个 GEMM：2 × (2·M·N·K)
    print("%16s%12.3f%14.3f   加速 %.1fx" % ("耗时(ms)", t_fused, t_eager, t_eager / t_fused))
    print("%16s%10.0f TFLOPS" % ("有效算力", flops / t_fused / 1e9))
    print("PASS ✓")


if __name__ == "__main__":
    main()
