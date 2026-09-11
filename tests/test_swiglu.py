"""
test_swiglu.py — swiglu 算子正确性测试（vs fp32 参考）

用法（自举 sys.path，可在任意 cwd 运行）：
    python tests/test_swiglu.py           # 全量
    python tests/test_swiglu.py --quick   # 快速冒烟

覆盖：
  - dtype：bf16 / fp16
  - M：1 / 7（极小）、128/129（tile_m 32↔64 边界）、1024/4096（tile 128）、
    1000（M 尾部谓词：1000 = 7×128+104）
  - K/N：64 最小、192/576（K 或 N 非 128 倍数走 kTileK=64 / 配对边界）、
    2048/1024 与 4096/11008（Llama 风格真实形状）
  - 输入校验：非 64 倍数 K/N、dtype 不匹配、CPU 输入拒绝
退出码：0 全部通过；1 存在 FAIL。
"""
import argparse
import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402

from custom_ops import ops  # noqa: E402  (import 时 JIT 编译)

torch.manual_seed(0)
DEV = "cuda"

RESULTS = {"PASS": 0, "FAIL": 0}
FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    RESULTS[tag] += 1
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"[{tag}] {name} {detail}")


def ref_swiglu(x, weight):
    """fp32 参考：y = silu(x @ Wg^T) * (x @ Wu^T)。"""
    N2 = weight.shape[0]
    N = N2 // 2
    xf, wf = x.float(), weight.float()
    gu = xf @ wf.t()
    g, u = gu[:, :N], gu[:, N:]
    return torch.nn.functional.silu(g) * u


def run_case(M, K, N, dtype):
    x = (torch.randn(M, K, device=DEV, dtype=torch.float32) * 0.5).to(dtype)
    w = (torch.randn(2 * N, K, device=DEV, dtype=torch.float32) * 0.05).to(dtype)
    out = ops.swiglu(x, w)
    ref = ref_swiglu(x, w)
    ok_shape = tuple(out.shape) == (M, N) and out.dtype == dtype
    diff = (out.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    rel = (diff / denom).max().item()
    # bf16/fp16 输出量化 + bf16 输入舍入：宽松阈值（同 test_fuse_moe 口径）
    tol = 0.03 if dtype == torch.bfloat16 else 0.02
    ok = ok_shape and torch.allclose(out.float(), ref, atol=0.05, rtol=tol)
    check(f"swiglu [{dtype} M={M} K={K} N={N}]",
          ok, f"shape_ok={ok_shape} max_abs={diff.max().item():.4g} max_rel={rel:.4g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    assert ops.is_available(), "extension load failed"

    if torch.cuda.get_device_capability(0)[0] < 8:
        print("SM80+ required for swiglu; skipping")
        sys.exit(0)

    cases = [
        # (M, K, N)
        (1,    64,  64),     # 最小 shape
        (7,    64,  64),     # 极小批量
        (128,  128, 128),    # tile_m 32↔64 边界
        (129,  128, 128),    # 边界 +1
        (1000, 2048, 1024),  # M 尾部谓词（1000 = 7×128 + 104）
        (333,  192,  64),    # K 非 128 倍数（kTileK=64 路径）
        (256,  128, 192),    # N=192（n=384，n%128==0 的非平凡配对数）
        (1024, 2048, 1024),  # 典型推理 batch
        (4096, 4096, 11008), # Llama-7B FFN 形状（11008 = 172×64）
    ]
    if args.quick:
        cases = cases[:4]

    for dtype in [torch.bfloat16, torch.float16]:
        for (M, K, N) in cases:
            run_case(M, K, N, dtype)

    # ── 输入校验 ──────────────────────────────────────────────────────────────
    x = torch.randn(8, 128, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(256, 128, device=DEV, dtype=torch.bfloat16)
    for name, fn in [
        ("odd_weight_rows_rejected", lambda: ops.swiglu(x, w[:255])),
        ("k_not_mult64_rejected", lambda: ops.swiglu(x[:, :100], w[:, :100])),
        ("n_not_mult64_rejected", lambda: ops.swiglu(x, w[:192])),
        ("dtype_mismatch_rejected", lambda: ops.swiglu(
            x, w.to(torch.float16))),
        ("cpu_input_rejected", lambda: ops.swiglu(x.cpu(), w.cpu())),
    ]:
        try:
            fn()
            check(name, False, "no error raised")
        except Exception:
            check(name, True)

    print()
    print("=" * 60)
    print("TOTAL: %d PASS, %d FAIL" % (RESULTS["PASS"], RESULTS["FAIL"]))
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
