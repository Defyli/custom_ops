"""
benchmark/benchmark_fa.py — mha_fwd_with_mask 性能基准测试

对比三组实现：
  1. custom_ops : 本仓库的 mha_fwd_with_mask（FA2 + 任意加法 mask，sm120 优化）
  2. SDPA+mask  : torch.nn.functional.scaled_dot_product_attention(attn_mask=...)
                  （带任意 mask 时只能走 MemEfficient 后端，是公平对照组）
  3. SDPA flash : SDPA 不带 mask（FlashAttention 后端，不支持 mask，仅作性能参照）

运行方式
--------
    # 全量（标准场景 + 小 grid 长序列 splitkv 场景）
    python benchmark/benchmark_fa.py

    # 只跑标准场景 / 只跑 splitkv 场景
    python benchmark/benchmark_fa.py --suite standard
    python benchmark/benchmark_fa.py --suite splitkv

    # 指定单个 shape
    python benchmark/benchmark_fa.py --shape 4 16 16 2048 2048 128

    # 调整迭代次数、输出 CSV
    python benchmark/benchmark_fa.py --warmup 20 --iters 100 --csv result.csv
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

# 仓库根目录（custom_ops/ 的上一级）加入 sys.path，使 `import custom_ops` 可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_ops import ops  # noqa: E402  （import 时自动 JIT 编译/加载）


# ─────────────────────────────────────────────────────────────────────────────
# 测试 shape 集（与 docs/fa_sm120_porting_and_optimization.md 第 4 节一致）
#   (B, H, Hk, Sq, Sk, d)
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_SHAPES = [
    # 标准场景：grid 足够大，splitkv 自动退化为单 kernel
    (1,  16, 16, 1024, 1024, 64),
    (4,  16, 16, 2048, 2048, 64),
    (1,  8,  8,  8192, 8192, 64),
    (32, 16, 16, 1024, 1024, 64),
    (1,  16, 16, 1024, 1024, 128),
    (4,  16, 16, 2048, 2048, 128),
    (1,  8,  8,  8192, 8192, 128),
    (4,  16, 4,  2048, 2048, 128),   # GQA
    (32, 16, 16, 1024, 1024, 128),
]

SPLITKV_SHAPES = [
    # 小 grid + 长序列：B*H*num_m_blocks << SM 数，splitkv 自动生效
    (1, 1, 1, 128,  8192,  128),
    (1, 1, 1, 128,  32768, 128),
    (1, 1, 1, 512,  8192,  128),
    (1, 1, 1, 1024, 8192,  128),
    (1, 2, 1, 512,  16384, 128),    # GQA
    (1, 1, 1, 128,  8192,  64),
    (1, 1, 1, 1024, 8192,  64),
    (1, 2, 1, 512,  16384, 64),     # GQA
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs(B, H, Hk, Sq, Sk, d, mask_ratio=0.1, seed=42):
    """构造 q/k/v 与随机加法 mask（保证每行至少一个可见位置，避免全屏蔽行）。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.bfloat16, generator=g)

    mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16)
    if mask_ratio > 0:
        drop = torch.rand(B, 1, Sq, Sk, device="cuda", generator=g) < mask_ratio
        drop[..., 0] = False          # 保证第 0 列始终可见，避免整行 -inf
        mask = mask.masked_fill(drop, float("-inf"))
    return q, k, v, mask


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


def tflops(B, H, Sq, Sk, d, time_us):
    """Attention 前向 FLOPS = 2 GEMM × 2*B*H*Sq*Sk*d。"""
    return 4.0 * B * H * Sq * Sk * d / (time_us * 1e-6) / 1e12


def sdpa_with_mask(q, k, v, mask):
    """SDPA 对照：GQA 时先扩展 K/V 头数。"""
    H, Hk = q.size(1), k.size(1)
    if H != Hk:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def sdpa_flash_no_mask(q, k, v):
    """SDPA Flash 后端参照（不支持 mask，仅作性能上限参照）。"""
    H, Hk = q.size(1), k.size(1)
    if H != Hk:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)
    return F.scaled_dot_product_attention(q, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(name, shapes, args):
    print(f"\n{'=' * 100}")
    print(f"  {name}   (mask_ratio={args.mask_ratio}, warmup={args.warmup}, iters={args.iters})")
    print(f"{'=' * 100}")
    header = (
        f"{'shape (B,H,Hk,Sq,Sk,d)':<34}"
        f"{'custom':>12}{'custom TF':>11}"
        f"{'SDPA+mask':>12}{'speedup':>9}"
        f"{'SDPA flash*':>13}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for (B, H, Hk, Sq, Sk, d) in shapes:
        q, k, v, mask = make_inputs(B, H, Hk, Sq, Sk, d, args.mask_ratio, args.seed)

        # 正确性快速校验（仅首个 shape 输出，其余静默断言）
        out_custom = ops.mha_fwd_with_mask(q, k, v, mask)
        out_ref = sdpa_with_mask(q, k, v, mask)
        max_diff = (out_custom.float() - out_ref.float()).abs().max().item()
        status = "OK" if max_diff < 0.02 else f"WARN diff={max_diff:.4f}"

        t_custom = bench_us(lambda: ops.mha_fwd_with_mask(q, k, v, mask),
                            args.warmup, args.iters)
        t_sdpa = bench_us(lambda: sdpa_with_mask(q, k, v, mask),
                          args.warmup, args.iters)
        t_flash = bench_us(lambda: sdpa_flash_no_mask(q, k, v),
                           args.warmup, args.iters)

        tf = tflops(B, H, Sq, Sk, d, t_custom)
        speedup = t_sdpa / t_custom

        shape_str = f"{B},{H},{Hk},{Sq},{Sk},{d}"
        print(f"{shape_str:<34}"
              f"{t_custom:>9.1f}µs{tf:>9.1f}"
              f"{t_sdpa:>9.1f}µs{speedup:>8.2f}x"
              f"{t_flash:>10.1f}µs   [{status}]")
        rows.append((shape_str, t_custom, tf, t_sdpa, speedup, t_flash, max_diff))

        del q, k, v, mask, out_custom, out_ref
        torch.cuda.empty_cache()

    return rows


def main():
    parser = argparse.ArgumentParser(description="mha_fwd_with_mask benchmark")
    parser.add_argument("--suite", choices=["all", "standard", "splitkv"],
                        default="all", help="跑哪组 shape（默认 all）")
    parser.add_argument("--shape", nargs=6, type=int, metavar=("B", "H", "Hk", "Sq", "Sk", "d"),
                        help="指定单个 shape，覆盖 --suite")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.1,
                        help="随机 mask 中被屏蔽（-inf）位置的比例，默认 0.1")
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
    print("* SDPA flash 列不支持 mask，仅作性能参照；公平对照是 SDPA+mask 列。")

    all_rows = []
    if args.shape:
        all_rows += run_suite("custom shape", [tuple(args.shape)], args)
    else:
        if args.suite in ("all", "standard"):
            all_rows += run_suite("标准场景（大 grid，splitkv 自动退化单 kernel）",
                                  STANDARD_SHAPES, args)
        if args.suite in ("all", "splitkv"):
            all_rows += run_suite("小 grid + 长序列场景（splitkv 自动生效）",
                                  SPLITKV_SHAPES, args)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("shape,custom_us,custom_tflops,sdpa_mask_us,speedup,sdpa_flash_us,max_diff\n")
            for r in all_rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"\n结果已写入 {args.csv}")


if __name__ == "__main__":
    main()
