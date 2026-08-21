"""
examples/mixed_gemm_demo.py — mixed_gemm 使用示例

演示生成式推荐 MLP 层的混合精度 GEMM：
    y = silu(x @ W^T + b)

问题背景：bf16 权重前向精度损失大（推荐模型对精度敏感），fp32/tf32 又太慢。
mixed_gemm 将 fp32 权重离线拆成 bf16 主项 + 低精度 residual 补偿项，
在线用 bf16 tensor core + 低精度 tensor core 两次 GEMM 恢复接近 fp32 的精度，
bias 与激活函数在 epilogue 融合执行（无额外 kernel、无中间显存）。

演示：
  1. 离线权重拆分（split_mixed_precision_weight，一次性）
  2. GEMM 风格调用（含 bias + silu epilogue 融合）
  3. 精度对比：mixed_gemm vs bf16 matmul（相对 fp32 金标准的平均相对误差）

运行：
    python examples/mixed_gemm_demo.py
"""

import os
import sys

import torch

# 本仓库根目录 custom_ops/ 自身就是 Python 包（根下有 __init__.py），
# 需把它的上一级目录加入 sys.path，`import custom_ops` 才能命中包本体。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../custom_ops（仓库根 = 包根）
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops, split_mixed_precision_weight  # noqa: E402


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用，示例无法运行。")
        return
    if not ops.is_available():
        print(f"算子库加载失败: {ops.load_error()}")
        return

    torch.manual_seed(0)

    # ── 典型推荐 MLP 形状：(batch, hidden=4096, 4096) ────────────────────────
    M, N, K = 4096, 4096, 4096
    x = torch.randn(M, K, device="cuda") * 0.5      # fp32 激活
    w = torch.randn(N, K, device="cuda") * 0.05     # fp32 权重
    b = torch.randn(N, device="cuda") * 0.1         # fp32 bias

    # ── 1. 离线权重拆分（模型加载时一次性） ──────────────────────────────────
    # backend="auto"：SM89+ 且 CUDA>=12.4 时用 FP8，否则降级 INT8（SM80+）
    w_high, w_low, w_scale = split_mixed_precision_weight(w)
    backend = "FP8" if w_low.dtype == torch.float8_e4m3fn else "INT8"
    print(f"权重拆分（backend={backend}）:")
    print(f"  w_high : {tuple(w_high.shape)} {w_high.dtype}   主项（bf16 tensor core）")
    print(f"  w_low  : {tuple(w_low.shape)}  {w_low.dtype}   residual 补偿项")
    if w_scale is not None:
        print(f"  w_scale: {tuple(w_scale.shape)} {w_scale.dtype}   per-channel 量化 scale")

    # ── 2. GEMM 风格调用：bias + silu 在 epilogue 融合 ────────────────────────
    y = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=b, activation="silu")

    # ── 3. 精度对比 ──────────────────────────────────────────────────────────
    # 三个指标（相对 fp32 金标准）：
    #   mean rel：平均相对误差（分母下限 1e-3，避免除零）
    #   RMS     ：相对 Frobenius 误差 ‖y-y_ref‖/‖y_ref‖（整体噪声水平）
    #   max abs ：最大绝对误差（尾部风险）
    # mixed_gemm 消除了权重的系统性舍入偏差（剩余误差仅为激活的无偏舍入噪声），
    # bf16 则权重/激活两种舍入误差叠加，且权重部分是系统性的（每步同向）。
    y_ref = torch.nn.functional.silu(x @ w.t() + b)                 # fp32 金标准
    y_bf16 = torch.nn.functional.silu(
        x.to(torch.bfloat16) @ w.to(torch.bfloat16).t() + b.to(torch.bfloat16))

    def mean_rel(a, ref):
        return ((a.float() - ref).abs() / ref.abs().clamp_min(1e-3)).mean().item()

    def rms_rel(a, ref):
        return ((a.float() - ref).norm() / ref.norm()).item()

    def max_abs(a, ref):
        return (a.float() - ref).abs().max().item()

    print(f"\n输出: {tuple(y.shape)} {y.dtype}")
    print(f"{'':18s}{'mixed_gemm':>12s}{'bf16 matmul':>14s}{'mixed 优势':>12s}")
    print(f"{'mean rel-err':18s}{mean_rel(y, y_ref):>12.2e}{mean_rel(y_bf16, y_ref):>14.2e}")
    print(f"{'RMS rel-err':18s}{rms_rel(y, y_ref):>12.2e}{rms_rel(y_bf16, y_ref):>14.2e}"
          f"{rms_rel(y_bf16, y_ref) / rms_rel(y, y_ref):>11.1f}x")
    print(f"{'max abs-err':18s}{max_abs(y, y_ref):>12.2e}{max_abs(y_bf16, y_ref):>14.2e}"
          f"{max_abs(y_bf16, y_ref) / max_abs(y, y_ref):>11.1f}x")

    assert mean_rel(y, y_ref) < 0.02, "mixed_gemm 精度异常！"
    assert rms_rel(y, y_ref) < 0.5 * rms_rel(y_bf16, y_ref), "RMS 优势不足！"
    print("PASS ✓")


if __name__ == "__main__":
    main()
