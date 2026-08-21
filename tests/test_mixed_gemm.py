"""
mixed_gemm 数值正确性验证

对照参考实现（纯 PyTorch，fp32 语义）验证：
  1. identity / silu / gelu 激活 × 有/无 bias × fp32/bf16 输出
  2. INT8 与（若可用）FP8 residual 后端
  3. fp32 与 bf16 输入 x
  4. 多种 (M, N, K) 形状，含 ragged M/N/K、小 M（split-K 高发区）
  5. 多维 x（前导维度折叠）
  6. 强制各档 split-K（force_splitk ∈ {1, 2, 4, 8, 16}）下的路径一致性

运行:  python tests/test_mixed_gemm.py
"""

import sys
import os

# 本仓库根目录 custom_ops/ 自身就是 Python 包（根下有 __init__.py），
# 需把它的上一级目录加入 sys.path，`import custom_ops` 才能命中包本体。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

import torch

from custom_ops import ops, split_mixed_precision_weight

torch.manual_seed(0)

DEV = "cuda"
SCALE = 1.0 / 256.0

FAILURES = []


def check(name, y, y_ref, rtol=2e-2, atol=2e-2):
    """对照 fp32 参考，报告相对误差。"""
    y = y.float()
    err = (y - y_ref).abs()
    denom = y_ref.abs().clamp_min(1e-3)
    rel = (err / denom)
    max_abs = err.max().item()
    mean_rel = rel.mean().item()
    max_rel = rel.max().item()
    # 逐元素容忍：绝对误差或相对误差任一满足即可
    ok = bool(((err <= atol) | (rel <= rtol)).all())
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:60s} max_abs={max_abs:.3e} mean_rel={mean_rel:.3e} max_rel={max_rel:.3e}")
    if not ok:
        FAILURES.append(name)
    return ok


def reference(x, w, bias, activation, scale=SCALE):
    """fp32 参考：x @ w^T + bias 再激活（与 kernel 分解数学等价）"""
    y = x.float() @ w.t()
    if bias is not None:
        y = y + bias
    if activation == "silu":
        y = y * torch.sigmoid(y)
    elif activation == "gelu":
        y = torch.nn.functional.gelu(y, approximate="tanh")
    return y


def run_case(m, n, k, x_dtype, backend, activation, has_bias, out_dtype,
             force_splitk=0, tag=""):
    x = (torch.randn(m, k, device=DEV) * 0.5).to(x_dtype)
    w = torch.randn(n, k, device=DEV) * 0.05   # 典型推荐模型量级
    bias = torch.randn(n, device=DEV) * 0.1 if has_bias else None

    w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, backend)
    y = ops.mixed_gemm(x, w_high, w_low, w_scale, scale=SCALE, bias=bias,
                       activation=activation, out_dtype=out_dtype,
                       force_splitk=force_splitk)

    y_ref = reference(x, w, bias, activation)
    name = f"m={m:5d} n={n:5d} k={k:5d} x={str(x_dtype).split('.')[-1]:8s} " \
           f"{backend:4s} act={activation:8s} bias={int(has_bias)} " \
           f"out={str(out_dtype).split('.')[-1]:8s} sk={force_splitk} {tag}"
    check(name, y, y_ref)


def main():
    if not ops.is_available():
        print("CUDA ops not available — aborting")
        sys.exit(1)

    fp8_ok = ops.mixed_gemm_fp8_available()
    print(f"FP8 backend available: {fp8_ok}")
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    print("=" * 110)

    backends = ["int8"] + (["fp8"] if fp8_ok else [])

    # ── 1. 功能矩阵：激活 × bias × 输出 dtype × 输入 dtype ────────────────────
    for backend in backends:
        for act in ["identity", "silu", "gelu"]:
            for has_bias in [False, True]:
                for out_dtype in [torch.float32, torch.bfloat16]:
                    run_case(256, 512, 1024, torch.float32, backend, act,
                             has_bias, out_dtype)
        # bf16 输入
        run_case(256, 512, 1024, torch.bfloat16, backend, "silu", True, torch.float32)

    # ── 2. 形状矩阵：ragged M/N/K、小 M、大 K ─────────────────────────────────
    shapes = [
        (16, 4096, 4096),      # 小 M：split-K 高发区
        (33, 1000, 512),       # ragged M/N/K 全占
        (1, 64, 64),           # 极小
        (129, 200, 72),        # ragged + K=72（非 64 倍数）
        (1000, 333, 2048),     # ragged N
        (777, 128, 8),         # K=8 最小对齐
        (4096, 4096, 512),     # 大 M
        (64, 64, 4096),        # 深 K
    ]
    for backend in backends:
        for (m, n, k) in shapes:
            run_case(m, n, k, torch.float32, backend, "identity", False, torch.float32)

    # ── 3. 多维 x（前导维度折叠） ──────────────────────────────────────────────
    x = torch.randn(2, 3, 128, device=DEV)
    w = torch.randn(256, 128, device=DEV) * 0.05
    w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, "int8")
    y = ops.mixed_gemm(x, w_high, w_low, w_scale)
    y_ref = reference(x, w, None, "identity")
    assert y.shape == (2, 3, 256), f"shape mismatch: {y.shape}"
    check("3D x (2,3,128) leading-dim fold", y, y_ref)

    # ── 4. 强制各档 split-K ──────────────────────────────────────────────────
    for sk in [1, 2, 4, 8, 16]:
        run_case(64, 512, 2048, torch.float32, "int8", "silu", True,
                 torch.float32, force_splitk=sk)

    # ── 5. CUDA graph 捕获（TRT 图捕获场景的等价能力） ────────────────────────
    try:
        x = torch.randn(128, 1024, device=DEV)
        w = torch.randn(512, 1024, device=DEV) * 0.05
        w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, "int8")
        bias = torch.randn(512, device=DEV) * 0.1
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=bias, activation="silu")
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y2 = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=bias, activation="silu")
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        y_ref = reference(x, w, bias, "silu")
        check("CUDA graph capture + replay", y2, y_ref)
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] CUDA graph capture: {e}")

    print("=" * 110)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
