"""
remote_verify_suite.py — FA mask 算子全面验证套件（正确性 + ragged + 性能抽样）

在三台架构机器上通用：
  sm89 (4090D)  : bf16 + fp16，base/splitkv/splitkv_m64 路径
  sm120 (5090D) : bf16 + fp16，base/splitkv/splitkv_m64 路径（TMA）
  sm70 (V100)   : 仅 fp16（bf16 应显式报错），base 路径

用法：
  python tests/remote_verify_suite.py            # 全量
  python tests/remote_verify_suite.py --quick    # 快速冒烟
  python tests/remote_verify_suite.py --perf-only
  python tests/remote_verify_suite.py --corr-only

退出码：0 全部通过；1 存在 FAIL。
"""
import argparse
import os
import sys
import time

# 自举：把包父目录（本文件上两级的父目录，即 custom_ops 的上层）加入 sys.path，
# 使脚本可在任意 cwd / 任意 PYTHONPATH 下运行（远程验证免环境配置）
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402

t0 = time.time()
from custom_ops import ops  # noqa: E402
LOAD_TIME = time.time() - t0

torch.manual_seed(0)

DEV = "cuda"
CC = torch.cuda.get_device_capability(0)
ARCH = CC[0] * 10 + CC[1]
IS_SM70 = ARCH == 70

RESULTS = {"PASS": 0, "FAIL": 0}
FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    RESULTS[tag] += 1
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"[{tag}] {name} {detail}")


def sdpa_ref(q, k, v, mask):
    # torch 2.x SDPA 不自动支持 GQA（H != Hk 需 enable_gqa，旧版本无此参数）：
    # 手动展开 kv 到 q 的 head 数（FA 标准语义：q head i 对应 kv head i // ratio，
    # 与 kernel 的 h_h_k_ratio 分组一致）
    if k.size(1) != q.size(1):
        r = q.size(1) // k.size(1)
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def run_case(name, B, H, Hk, Sq, Sk, d, dtype, mask_kind="random",
             tol=0.02, q_rows_extra=0, k_cols_extra=0):
    """单例正确性验证。

    mask_kind:
      random   : 随机 -inf 屏蔽（~1/7 概率）
      all_zero : 全可见
      full_row : 整行 -inf（softmax 空集 → 输出应为 0）
      bias     : 有限值偏置（ALiBi 风格）
    q_rows_extra/k_cols_extra: mask 行/列数超出 Sq/Sk 的量（预 pad 契约）
    """
    q = torch.randn(B, H, Sq, d, device=DEV, dtype=dtype)
    k = torch.randn(B, Hk, Sk, d, device=DEV, dtype=dtype)
    v = torch.randn(B, Hk, Sk, d, device=DEV, dtype=dtype)
    Sq_m, Sk_m = Sq + q_rows_extra, Sk + k_cols_extra
    mask = torch.zeros(B, 1, Sq_m, Sk_m, device=DEV, dtype=dtype)
    if mask_kind == "random":
        mk = torch.rand(B, 1, Sq_m, Sk_m, device=DEV) < (1.0 / 7.0)
        mask.masked_fill_(mk, float("-inf"))
    elif mask_kind == "full_row":
        mask[:, :, Sq_m // 3, :] = float("-inf")
    elif mask_kind == "bias":
        pos_k = torch.arange(Sk_m, device=DEV, dtype=torch.float32)
        mask += ((pos_k % 16) - 8).to(dtype).view(1, 1, 1, -1)
    # all_zero: 保持全 0

    out = ops.mha_fwd_with_mask(q, k, v, mask)
    # mask 预 pad 时（行/列超出 Sq/Sk），kernel 语义 = 只看 [0:Sq, 0:Sk] 区域
    ref_mask = mask if (Sq_m == Sq and Sk_m == Sk) else mask[..., :Sq, :Sk]
    ref = sdpa_ref(q, k, v, ref_mask)
    # 空语义差异兼容：旧版 SDPA（如 torch 2.0）对全 -inf 行输出 NaN（数学严格
    # 语义）；kernel 遵循 FA 官方语义输出 0（且不得产出 NaN）→ ref 的 NaN 位
    # 按 0 对比，并断言 out 自身无 NaN
    ref_f = ref.float()
    nan_rows = torch.isnan(ref_f)
    if nan_rows.any():
        ref_f = torch.where(nan_rows, torch.zeros_like(ref_f), ref_f)
    diff = (out.float() - ref_f).abs().max().item()
    has_nan = torch.isnan(out.float()).any().item()
    if has_nan:
        diff = float("nan")
    check(f"{name} [{mask_kind}]", diff < tol, f"max_diff={diff:.2e}")


# ── 1. 正确性：标准 + ragged ─────────────────────────────────────────────────
def suite_correctness(quick=False):
    print(f"\n=== correctness (arch={ARCH}, sm70={IS_SM70}) ===")
    dtypes = [torch.float16] + ([] if IS_SM70 else [torch.bfloat16])

    for dtype in dtypes:
        tn = "fp16" if dtype == torch.float16 else "bf16"

        # bf16 在 V100 上应显式报错（而非静默错误结果）
        if IS_SM70 and dtype == torch.bfloat16:
            try:
                q = torch.randn(1, 2, 64, 64, device=DEV, dtype=dtype)
                ops.mha_fwd_with_mask(q, q, q, torch.zeros(1, 1, 64, 64, device=DEV, dtype=dtype))
                check(f"{tn} on V100 rejected", False, "no error raised")
            except RuntimeError:
                check(f"{tn} on V100 rejected", True)
            continue

        # 标准 shape（对齐，零拷贝主路径）
        for (B, H, Hk, Sq, Sk, d) in [
            (1, 4, 4, 128, 1024, 64),
            (2, 8, 8, 512, 1024, 128),
            (1, 16, 16, 1024, 4096, 128),
        ][: (1 if quick else 3)]:
            for mk in (["random"] if quick else ["random", "all_zero", "full_row", "bias"]):
                run_case(f"std B{B}H{H}q{Sq}k{Sk}d{d}{tn}", B, H, Hk, Sq, Sk, d, dtype, mk)

        # ragged Sk（%8!=0 → 自动 pad）
        for Sk in ([9, 101] if quick else [1, 7, 9, 55, 63, 100, 101, 777, 1031, 2053, 4099]):
            run_case(f"raggedK Sk={Sk} {tn}", 1, 4, 4, 128, Sk, 64, dtype, "random")
        if not quick:
            for Sk in [101, 4099]:
                run_case(f"raggedK d128 Sk={Sk} {tn}", 1, 4, 4, 128, Sk, 128, dtype, "random")

        # ragged Sq（行谓词，任意 Sq）
        for Sq in ([65, 129] if quick else [1, 3, 63, 65, 127, 129, 333, 1000]):
            run_case(f"raggedQ Sq={Sq} {tn}", 1, 4, 4, Sq, 1024, 64, dtype, "random")

        # GQA
        for (H, Hk) in ([((8, 2))] if quick else [(2, 1), (8, 2), (12, 3)]):
            run_case(f"GQA H{H}/Hk{Hk} {tn}", 2, H, Hk, 256, 512, 128, dtype, "random")

        # mask 预 pad 契约（行/列超出 Sq/Sk）
        if not quick:
            run_case(f"mask prepad rows+13 cols+5 {tn}", 1, 4, 4, 128, 512, 64,
                     dtype, "random", q_rows_extra=13, k_cols_extra=5)
            run_case(f"mask prepad unaligned cols+3 {tn}", 1, 4, 4, 128, 512, 64,
                     dtype, "random", k_cols_extra=3)

        # split-KV 触发（小 grid + 长序列）+ ragged Sk 组合
        if not quick:
            for Sk in [8192 + 3, 16384, 32768 + 5]:
                run_case(f"splitkv Sk={Sk} {tn}", 1, 2, 2, 64, Sk, 128, dtype, "random")
            run_case(f"splitkv d64 Sk=16390 {tn}", 1, 2, 2, 128, 16390, 64, dtype, "random")


# ── 2. 性能抽样（CUDA events，交错 SDPA 基线）─────────────────────────────────
def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def suite_perf(quick=False):
    print(f"\n=== perf sampling (arch={ARCH}) ===")
    dtype = torch.float16 if IS_SM70 else torch.bfloat16
    shapes = [
        (1, 4, 4, 128, 1024, 64),
        (2, 8, 8, 512, 1024, 128),
        (1, 16, 16, 1024, 4096, 128),
        (1, 2, 2, 64, 32768, 128),   # splitkv
        (1, 4, 4, 128, 4099, 64),    # ragged（自动 pad）
    ]
    print(f"{'shape':<34} {'ours(us)':>9} {'sdpa(us)':>9} {'speedup':>8}")
    for (B, H, Hk, Sq, Sk, d) in shapes:
        q = torch.randn(B, H, Sq, d, device=DEV, dtype=dtype)
        k = torch.randn(B, Hk, Sk, d, device=DEV, dtype=dtype)
        v = torch.randn(B, Hk, Sk, d, device=DEV, dtype=dtype)
        mask = torch.zeros(B, 1, Sq, Sk, device=DEV, dtype=dtype)
        t_ours = bench(lambda: ops.mha_fwd_with_mask(q, k, v, mask))
        t_sdpa = bench(lambda: sdpa_ref(q, k, v, mask))
        print(f"B{B} H{H} q{Sq} k{Sk} d{d:<3}{'':<8}{t_ours:>9.1f} {t_sdpa:>9.1f} "
              f"{t_sdpa / t_ours:>7.2f}x")


def suite_mixed_gemm_smoke():
    """mixed_gemm 冒烟：本次重构把 sm120 路由宏从 __CUDA_ARCH_LIST__ 切到
    FA_HAS_SM120（csrc/arch_targets.h），验证数值与路由回退不受影响。
    V100 跳过（mixed_gemm 需 SM80+）。"""
    if IS_SM70:
        print("\n=== mixed_gemm smoke: skipped (requires SM80+) ===")
        return
    print(f"\n=== mixed_gemm smoke (arch={ARCH}) ===")
    from custom_ops import split_mixed_precision_weight
    torch.manual_seed(0)
    # 对齐 benchmark/benchmark_mixed_gemm.py 的正确性约定：fp32 金标准 +
    # mean_rel_err（int8 后端含 kernel 内部 x 动态量化，逐点重建无法建模）
    N, K, M = 512, 1024, 256
    x = torch.randn(M, K, device=DEV) * 0.5
    w = (torch.randn(N, K, device=DEV) * 0.05).float()
    bias = torch.randn(N, device=DEV) * 0.1
    # activation epilogue（gelu 为 tanh 近似，与 kernel 语义一致）× bias 融合
    acts = {
        "identity": lambda t: t,
        "silu": torch.nn.functional.silu,
        "gelu": lambda t: torch.nn.functional.gelu(t, approximate="tanh"),
    }
    for backend in ("fp8", "int8"):
        if backend == "fp8" and not ops.mixed_gemm_fp8_available():
            print("[skip] fp8 backend unavailable on this build")
            continue
        w_high, w_low, w_scale = split_mixed_precision_weight(w, backend=backend)
        for act_name, act_fn in acts.items():
            for use_bias in (False, True):
                b = bias if use_bias else None
                y_ref = act_fn(x @ w.t() + (b if use_bias else 0.0))
                y = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=b,
                                   activation=act_name)
                rel = ((y.float() - y_ref).abs()
                       / y_ref.abs().clamp_min(1e-3)).mean().item()
                check(f"mixed_gemm {backend} {act_name} bias={use_bias}",
                      rel < 0.02, f"mean_rel_err={rel:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--perf-only", action="store_true")
    ap.add_argument("--corr-only", action="store_true")
    args = ap.parse_args()

    print(f"load={LOAD_TIME:.1f}s available={ops.is_available()} "
          f"arch={ARCH} torch={torch.__version__}")
    if not ops.is_available():
        print("FATAL: ops not available:", ops.load_error())
        sys.exit(1)

    if not args.perf_only:
        suite_correctness(args.quick)
        suite_mixed_gemm_smoke()
    if not args.corr_only:
        suite_perf(args.quick)

    print(f"\n=== summary: PASS={RESULTS['PASS']} FAIL={RESULTS['FAIL']} ===")
    for f in FAILURES:
        print("  FAIL:", f)
    sys.exit(1 if RESULTS["FAIL"] else 0)


if __name__ == "__main__":
    main()
