"""
tests/test_mixed_gemm_ragged.py — mixed_gemm ragged shape 全面正确性验证

针对 sm120a TMA 路径新增的边界语义做系统性扫描（sm80 回退路径同样被覆盖）：

  1. ragged M/N 扫描：围绕 kTileM/kTileN ∈ {64, 128} 边界的 ±1 位置
     （TMA load 的 OOB 行/列自动补零 + TMA store 的 OOB 自动裁剪）
  2. ragged K 扫描（k % 16 == 0）：TMA 路径，K 维不足一个 kTileK=64 slab
     时的补零（如 k=80 → 第二个 slab 仅有 16 个有效列）
  3. k % 16 != 0：必须整体回退 sm80 路径（TMA 全局 stride 约束）
  4. M/N/K 同时 ragged 的组合
  5. split-K（2/4/8/16）× ragged：chunk 内 K 长度不均（base/rem 分配）、
     甚至 ntile_k < kSplitK（部分 chunk 空）的退化情形
  6. TMA store 对齐边界：out=bfloat16 时 n%8 决定 bulk store 与 elementwise
     回退（n*2 字节须 16B 对齐）
  7. 激活 × bias × 输出/输入 dtype 在 ragged shape 上的矩阵抽样

运行方式
--------
    python tests/test_mixed_gemm_ragged.py

配合 compute-sanitizer（先设 GEMM_MIXED_DEBUG_GRID=1 可同时统计路由分布）：
    compute-sanitizer --tool memcheck --error-exitcode 1 \
        python tests/test_mixed_gemm_ragged.py
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

import torch

from custom_ops import ops, split_mixed_precision_weight  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
SCALE = 1.0 / 256.0

FAILURES = []
STATS = {"cases": 0}


def check(name, y, y_ref, rtol=2e-2, atol=2e-2):
    y = y.float()
    err = (y - y_ref).abs()
    denom = y_ref.abs().clamp_min(1e-3)
    rel = err / denom
    max_abs = err.max().item()
    mean_rel = rel.mean().item()
    ok = bool(((err <= atol) | (rel <= rtol)).all())
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:64s} max_abs={max_abs:.3e} mean_rel={mean_rel:.3e}")
    STATS["cases"] += 1
    if not ok:
        FAILURES.append(name)
    return ok


def reference(x, w, bias, activation, scale=SCALE):
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
    w = torch.randn(n, k, device=DEV) * 0.05
    bias = torch.randn(n, device=DEV) * 0.1 if has_bias else None

    w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, backend)
    y = ops.mixed_gemm(x, w_high, w_low, w_scale, scale=SCALE, bias=bias,
                       activation=activation, out_dtype=out_dtype,
                       force_splitk=force_splitk)

    y_ref = reference(x, w, bias, activation)
    name = (f"m={m:5d} n={n:5d} k={k:5d} {backend:4s} act={activation:8s} "
            f"bias={int(has_bias)} out={str(out_dtype).split('.')[-1]:8s} "
            f"sk={force_splitk:<2d} {tag}")
    check(name, y, y_ref)


def main():
    if not ops.is_available():
        print("CUDA ops not available — aborting")
        sys.exit(1)

    fp8_ok = ops.mixed_gemm_fp8_available()
    backends = ["int8"] + (["fp8"] if fp8_ok else [])
    print(f"FP8 backend available: {fp8_ok}")
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    print("=" * 118)

    # ── 1. ragged M 扫描（k 对齐 → sm120 TMA 路径；M 维 OOB 补零/裁剪）──────
    m_sweep = [1, 2, 15, 16, 17, 31, 33, 63, 64, 65, 95, 96, 97,
               127, 128, 129, 191, 255, 256, 257, 383, 511]
    for backend in backends:
        for m in m_sweep:
            run_case(m, 512, 1024, torch.float32, backend, "identity",
                     False, torch.float32)

    # ── 2. ragged N 扫描（覆盖 kTileN=64/128 边界与 TMA store 对齐边界）─────
    #    n=1000：fp32 时 stride 4000B（16B 对齐，TMA store）；bf16 时 2000B
    #    （同样对齐）。n=1001/1009：stride 不对齐 → elementwise 回退。
    n_sweep = [1, 2, 15, 17, 31, 63, 64, 65, 95, 96, 127, 128, 129,
               192, 200, 255, 256, 257, 333, 511, 1000, 1001, 1008, 1016]
    for backend in backends:
        for n in n_sweep:
            run_case(256, n, 1024, torch.float32, backend, "identity",
                     False, torch.float32)
        # TMA store 对齐边界（bf16 输出：n%8 决定 bulk/elementwise）
        for n in [56, 1000, 1001, 1007, 1008]:
            run_case(96, n, 512, torch.float32, backend, "silu", True,
                     torch.bfloat16)

    # ── 3. ragged K 扫描，k%16==0（sm120 TMA 路径；K 维 OOB 补零）───────────
    #    k<64（单 slab 不足）、k=80/112（第二 slab 部分）、k=2032（slab 数非
    #    2 的幂）、k=4112（>4096 跨页）等。
    k_sweep = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192,
               320, 576, 1008, 1040, 2032, 2064, 4112, 5008]
    for backend in backends:
        for k in k_sweep:
            run_case(96, 200, k, torch.float32, backend, "identity",
                     False, torch.float32)

    # ── 4. k%16!=0：必须回退 sm80 路径（数值仍须正确）───────────────────────
    #    注：wrapper 要求 k%8==0（sm80 路径 cp.async 128-bit 对齐），本组取
    #    8 的倍数但非 16 的倍数的 k。
    for backend in backends:
        for k in [8, 24, 40, 56, 72, 88, 104, 136, 1000, 2040, 4088]:
            run_case(96, 200, k, torch.float32, backend, "identity",
                     False, torch.float32, tag="sm80-fallback")

    # ── 5. M/N/K 同时 ragged 的组合 ─────────────────────────────────────────
    combos = [
        (33, 1001, 528), (77, 333, 80), (129, 200, 144), (255, 511, 1040),
        (97, 65, 48), (1, 1, 16), (1, 4096, 80), (4095, 33, 16),
        (65, 127, 2064), (63, 63, 63 + 1),  # 63+1=64：恰好一个 slab
    ]
    for backend in backends:
        for (m, n, k) in combos:
            run_case(m, n, k, torch.float32, backend, "identity",
                     False, torch.float32)

    # ── 6. split-K × ragged（chunk K 长度不均 / 空 chunk）───────────────────
    #    k=80, sk=16：ntile_k=2 < 16 → 14 个空 chunk（仅写零 partial）。
    #    k=1008, sk=16：base/rem 混合（63/64 slab 交替）。
    sk_cases = [
        (33, 1000, 1040, 2), (33, 1000, 1040, 8),
        (65, 127, 80, 4), (65, 127, 80, 16),
        (77, 333, 528, 2), (77, 333, 528, 16),
        (16, 4096, 1008, 4), (16, 4096, 1008, 16),
        (96, 1001, 2032, 8), (255, 257, 320, 2),
        (1, 1001, 80, 16), (63, 63, 16, 8),
    ]
    for backend in backends:
        for (m, n, k, sk) in sk_cases:
            run_case(m, n, k, torch.float32, backend, "silu", True,
                     torch.float32, force_splitk=sk)

    # ── 7. 激活 × bias × 输出/输入 dtype 矩阵抽样（ragged shape 上）─────────
    for backend in backends:
        for act in ["identity", "silu", "gelu"]:
            for has_bias in [False, True]:
                for out_dtype in [torch.float32, torch.bfloat16]:
                    run_case(129, 200, 528, torch.float32, backend, act,
                             has_bias, out_dtype)
        run_case(129, 200, 528, torch.bfloat16, backend, "gelu", True,
                 torch.bfloat16)

    # ── 8. 多维 x（前导维度折叠）× ragged ───────────────────────────────────
    for backend in backends:
        # k=72：非 16 倍数（sm80 回退）；折叠后 m=6
        x = torch.randn(2, 3, 72, device=DEV)
        w = torch.randn(127, 72, device=DEV) * 0.05
        w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, backend)
        y = ops.mixed_gemm(x, w_high, w_low, w_scale)
        y_ref = reference(x, w, None, "identity")
        assert y.shape == (2, 3, 127), f"shape mismatch: {y.shape}"
        check(f"3D x (2,3,72) ragged fold [{backend}]", y, y_ref)

        # k=80：16 倍数（sm120 路径，K 维 OOB 补零）；折叠后 m=6
        x = torch.randn(2, 3, 80, device=DEV)
        w = torch.randn(127, 80, device=DEV) * 0.05
        w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, backend)
        y = ops.mixed_gemm(x, w_high, w_low, w_scale)
        y_ref = reference(x, w, None, "identity")
        assert y.shape == (2, 3, 127), f"shape mismatch: {y.shape}"
        check(f"3D x (2,3,80) ragged fold [{backend}]", y, y_ref)

    print("=" * 118)
    if FAILURES:
        print(f"{len(FAILURES)}/{STATS['cases']} CASES FAILED:")
        for name in FAILURES:
            print(f"  FAIL: {name}")
        sys.exit(1)
    print(f"ALL {STATS['cases']} RAGGED TESTS PASSED")


if __name__ == "__main__":
    main()
