"""
benchmark/benchmark_mixed_gemm.py — mixed_gemm 性能基准测试

mixed_gemm 解决的问题：生成式推荐模型中 bf16 权重精度损失大、tf32/fp32
性能不足。本脚本对比四组实现（同一 fp32 权重、同一输入）：

  1. mixed_gemm : 本仓库算子（bf16 主项 + fp8/int8 residual 补偿，
                  epilogue 融合 bias+silu），精度 ≈ fp32
  2. fp32 matmul: 精度金标准（速度下限）
  3. tf32 matmul: fp32 的快速模式（精度介于 bf16 与 fp32 之间）
  4. bf16 matmul: 速度上限（精度最差）

运行方式
--------
    python benchmark/benchmark_mixed_gemm.py

    # 指定 residual 后端（默认 auto：FP8 可用则 FP8，否则 INT8）
    python benchmark/benchmark_mixed_gemm.py --backend int8
    python benchmark/benchmark_mixed_gemm.py --backend fp8

    # 指定单个 shape、调整迭代次数
    python benchmark/benchmark_mixed_gemm.py --shape 4096 4096 4096
    python benchmark/benchmark_mixed_gemm.py --warmup 20 --iters 100

    # 输出 CSV
    python benchmark/benchmark_mixed_gemm.py --csv result.csv
"""

import argparse
import os
import sys

import torch

# 本仓库根目录 custom_ops/ 自身就是 Python 包（根下有 __init__.py），
# 需把它的上一级目录加入 sys.path，`import custom_ops` 才能命中包本体。
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../custom_ops（仓库根 = 包根）
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops, split_mixed_precision_weight  # noqa: E402  （import 时自动 JIT 编译/加载）

SCALE = 1.0 / 256.0


# ─────────────────────────────────────────────────────────────────────────────
# 测试 shape 集：(M, N, K) — 覆盖生成式推荐 MLP 的典型档位
#   - 小 M（推理 batch）：split-K 高发区
#   - 中大 M（训练 / 大 batch）
#   - ragged N/K
# ─────────────────────────────────────────────────────────────────────────────

SHAPES = [
    (16,    4096,  4096),   # 推理小 batch，split-K 生效
    (64,    4096,  4096),
    (256,   4096,  4096),
    (1024,  4096,  4096),   # 训练档
    (4096,  4096,  4096),   # 大 batch
    (16,   16384,  1024),   # 宽 N
    (128,   1000,  2048),   # ragged N
    (512,   4096,   256),   # 短 K
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def bench_us(fn, warmup, iters):
    """CUDA Event 计时，迭代间用 256MB buffer 清零刷新 L2，返回中位数（µs）。"""
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache.zero_()                 # flush L2
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2] * 1e3   # ms -> µs


def mean_rel_err(y, y_ref):
    """相对 fp32 金标准的平均相对误差（逐元素，分母下限 1e-3）。"""
    y = y.float()
    err = (y - y_ref).abs()
    denom = y_ref.abs().clamp_min(1e-3)
    return (err / denom).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(shapes, args, backend):
    header = (
        f"{'shape (M,N,K)':<22}"
        f"{'mixed':>10}{'TF':>8}"
        f"{'fp32':>10}{'vs fp32':>9}"
        f"{'tf32':>10}{'vs tf32':>9}"
        f"{'bf16':>10}"
        f"{'err mixed':>11}{'err bf16':>10}"
    )
    print(f"\nresidual backend: {backend}"
          + ("（CUDA 编译期决定；int8 为 SM80+ 兼容档）" if backend == "int8" else ""))
    print(header)
    print("-" * len(header))

    rows = []
    for (m, n, k) in shapes:
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        x = torch.randn(m, k, device="cuda", generator=g) * 0.5
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).float()
        bias = torch.randn(n, device="cuda", generator=g) * 0.1

        w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, backend)
        w_bf16 = w.to(torch.bfloat16)   # bf16 基线的离线权重准备（对等 split）

        y_ref = torch.nn.functional.silu(x @ w.t() + bias)   # fp32 金标准
        err_mixed = mean_rel_err(
            ops.mixed_gemm(x, w_high, w_low, w_scale, scale=SCALE,
                           bias=bias, activation="silu"), y_ref)
        err_bf16 = mean_rel_err(
            torch.nn.functional.silu(
                (x.to(torch.bfloat16) @ w_bf16.t()).float() + bias), y_ref)

        fn_mixed = lambda: ops.mixed_gemm(  # noqa: E731
            x, w_high, w_low, w_scale, scale=SCALE, bias=bias, activation="silu")
        fn_fp32 = lambda: torch.nn.functional.silu(x @ w.t() + bias)  # noqa: E731
        fn_bf16 = lambda: x.to(torch.bfloat16) @ w_bf16.t()  # noqa: E731

        tf32_on = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        t_tf32 = bench_us(lambda: torch.nn.functional.silu(x @ w.t() + bias),  # noqa: E731
                           args.warmup, args.iters)
        torch.backends.cuda.matmul.allow_tf32 = tf32_on

        t_mixed = bench_us(fn_mixed, args.warmup, args.iters)
        t_fp32 = bench_us(fn_fp32, args.warmup, args.iters)
        t_bf16 = bench_us(fn_bf16, args.warmup, args.iters)

        # 主 GEMM + epilogue 的有效 FLOPS（2*M*N*K）；mixed_gemm 内部实际执行
        # bf16 主项 + 低精度补偿两次 GEMM，这里按数学有效量计。
        tf = 2.0 * m * n * k / (t_mixed * 1e-6) / 1e12

        shape_str = f"{m},{n},{k}"
        print(f"{shape_str:<22}"
              f"{t_mixed:>8.1f}µs{tf:>7.1f}"
              f"{t_fp32:>8.1f}µs{t_fp32 / t_mixed:>8.2f}x"
              f"{t_tf32:>8.1f}µs{t_tf32 / t_mixed:>8.2f}x"
              f"{t_bf16:>8.1f}µs"
              f"{err_mixed:>10.2e}{err_bf16:>9.2e}")
        rows.append((shape_str, t_mixed, tf, t_fp32, t_tf32, t_bf16,
                     err_mixed, err_bf16))

        del x, w, bias, w_high, w_low, w_scale, y_ref
        torch.cuda.empty_cache()

    return rows


def main():
    parser = argparse.ArgumentParser(description="mixed_gemm benchmark")
    parser.add_argument("--backend", choices=["auto", "fp8", "int8"], default="auto",
                        help="residual 后端（默认 auto：FP8 可用则 FP8，否则 INT8）")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"),
                        help="指定单个 shape，覆盖默认 shape 集")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="", help="结果写入 CSV 文件路径")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用，无法运行 benchmark。")
        sys.exit(1)
    if not ops.is_available():
        print(f"custom_ops 加载失败: {ops.load_error()}")
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {dev}  (sm_{cap[0]}{cap[1]})   PyTorch: {torch.__version__}")
    print("err 列为相对 fp32 金标准的平均相对误差（RMS 误差约再低一个量级，"
          "mixed 比 bf16 低 2 倍+）")

    shapes = [tuple(args.shape)] if args.shape else SHAPES
    if args.backend == "auto":
        backend = "fp8" if ops.mixed_gemm_fp8_available() else "int8"
    else:
        if args.backend == "fp8":
            assert ops.mixed_gemm_fp8_available(), (
                "FP8 后端不可用（需编译期 CUDA >= 12.4 且 GPU 为 SM89+）")
        backend = args.backend
    rows = run_suite(shapes, args, backend)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("shape,mixed_us,mixed_tflops,fp32_us,tf32_us,bf16_us,"
                    "err_mixed,err_bf16\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"\n结果已写入 {args.csv}")


if __name__ == "__main__":
    main()
