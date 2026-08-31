"""
benchmark/benchmark_sm70.py — sm70 (V100) mha_fwd_with_mask 性能基准测试

对比四组实现：
  1. custom_ops : 本仓库的 sm70 FA forward（fp16, WMMA m16n16k16, 64×64 tile）
  2. SDPA+mask  : torch.nn.functional.scaled_dot_product_attention(attn_mask=...)
  3. FlexAttention : torch 官方自定义 mask 算子（torch>=2.5；V100 常见 torch<2.5，
                  不可用时自动输出 N/A）
  4. SDPA 参照  : SDPA 不带 mask（不支持任意 mask，仅作性能参照）

V100 环境特点：
  - 80 SMs, sm_70, fp16 Tensor Core (WMMA m16n16k16)
  - HBM2, ~900 GB/s 带宽
  - 无 bf16 ALU / 无 ldmatrix / 无 cp.async
  - tile 固定 (64,64), 512 threads/CTA

运行方式
--------
    python benchmark/benchmark_sm70.py
    python benchmark/benchmark_sm70.py --suite standard
    python benchmark/benchmark_sm70.py --suite ragged
    python benchmark/benchmark_sm70.py --shape 4 16 16 2048 2048 128
    python benchmark/benchmark_sm70.py --csv result_sm70.csv
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

# 本仓库根目录 custom_ops/ 自身就是 Python 包
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# V100 测试 shape 集 (B, H, Hk, Sq, Sk, d)
#   V100 80 SM, tile (64,64), num_m_blocks = ceil(Sq/64)
#   大 grid: B*H*num_m_blocks >> 80 → 充分利用 SM
#   小 grid: B*H*num_m_blocks <= 80 → SM 未填满
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_SHAPES = [
    # 大 grid 标准场景
    (1,  8, 8,  512,  512,  64),
    (1,  16, 16, 512,  512,  64),
    (2,  16, 16, 512,  512,  64),
    (4,  16, 16, 1024, 1024, 64),
    (1,  8, 8,  2048, 2048, 64),
    (1,  16, 16, 1024, 1024, 64),
    (2,  8, 8,  2048, 2048, 64),
    (4,  16, 16, 2048, 2048, 64),
    # d=128
    (1,  8, 8,  512,  512,  128),
    (2,  16, 16, 512,  512,  128),
    (4,  16, 16, 1024, 1024, 128),
    (1,  8, 8,  2048, 2048, 128),
    (4,  16, 16, 2048, 2048, 128),
    # GQA
    (4,  16, 4,  2048, 2048, 128),
    (4,  16, 4,  2048, 2048, 64),
]

SMALL_GRID_SHAPES = [
    # 小 grid：B*H*num_m_blocks < 80（SM 未填满）
    (1, 1, 1, 64,   512,  64),
    (1, 1, 1, 64,   1024, 64),
    (1, 1, 1, 64,   2048, 64),
    (1, 1, 1, 128,  512,  128),
    (1, 1, 1, 128,  1024, 128),
    (1, 1, 1, 128,  2048, 128),
    (1, 1, 1, 256,  1024, 128),
    (1, 2, 1, 128,  1024, 128),    # GQA
]

RAGGED_SHAPES = [
    # ragged Sk（Sk % 8 != 0 → 算子内部自动 pad 到 8 倍数，一次拷贝）。
    # 对齐参照行与 ragged 行相邻排列，直观评估自动 pad 的开销
    (1,  8,  8, 2048, 2048, 64),    # 对齐参照
    (1,  8,  8, 2048, 2053, 64),    # Sk%8=5 → pad 到 2056
    (1, 16, 16, 1024, 1024, 128),   # 对齐参照
    (1, 16, 16, 1024, 1031, 128),   # Sk%8=7 → pad 到 1032
    (2, 16, 16,  512, 4096, 64),    # 对齐参照
    (2, 16, 16,  512, 4099, 64),    # Sk%8=3 → pad 到 4104
    # ragged Sq（任意 Sq，行谓词裁剪）
    (1, 16, 16, 1000, 1024, 64),
    (1,  8,  8,  127, 1024, 128),
    # Sq + Sk 双 ragged、GQA + ragged
    (2, 16,  4,  500, 2053, 128),
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs(B, H, Hk, Sq, Sk, d, mask_ratio=0.1, seed=42):
    """构造 fp16 q/k/v 与随机加法 mask。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.float16, generator=g)

    mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.float16)
    if mask_ratio > 0:
        drop = torch.rand(B, 1, Sq, Sk, device="cuda", generator=g) < mask_ratio
        drop[..., 0] = False  # 保证第 0 列可见
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
        cache.zero_()  # flush L2
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2] * 1e3  # ms -> µs


def tflops(B, H, Sq, Sk, d, time_us):
    """Attention 前向 FLOPS = 4 * B * H * Sq * Sk * d。"""
    return 4.0 * B * H * Sq * Sk * d / (time_us * 1e-6) / 1e12


def gbps(B, H, Hk, Sq, Sk, d, time_us):
    """估算 HBM 带宽（Q+K+V+O 读写量，bytes）。"""
    # fp16: 2 bytes per element
    bytes_q = B * H * Sq * d * 2
    bytes_k = B * Hk * Sk * d * 2
    bytes_v = B * Hk * Sk * d * 2
    bytes_o = B * H * Sq * d * 2
    total_bytes = bytes_q + bytes_k + bytes_v + bytes_o
    return total_bytes / (time_us * 1e-6) / 1e9


def sdpa_with_mask(q, k, v, mask):
    """SDPA 对照：GQA 时先扩展 K/V 头数。"""
    H, Hk = q.size(1), k.size(1)
    if H != Hk:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def sdpa_flash_no_mask(q, k, v):
    """SDPA 无 mask 参照（不支持任意 mask；V100/torch 2.0 上为 mem-efficient 后端）。"""
    H, Hk = q.size(1), k.size(1)
    if H != Hk:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)
    return F.scaled_dot_product_attention(q, k, v)


# FlexAttention 对照（torch>=2.5；V100 环境常见 torch<2.5，自动降级 N/A）
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention_fn,
        create_block_mask as _create_block_mask_fn,
    )
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = 64
    if hasattr(_dynamo.config, "accumulated_recompile_limit"):
        _dynamo.config.accumulated_recompile_limit = 4096
    _flex_compiled = torch.compile(_flex_attention_fn)
    FLEX_AVAILABLE = True
except Exception:
    FLEX_AVAILABLE = False


def flex_make_runner(q, k, v, mask):
    """预构建 block_mask，返回可计时闭包；不可用或失败时返回 None。"""
    if not FLEX_AVAILABLE:
        return None
    try:
        B, H, Sq, _ = q.shape
        Hk, Sk = k.size(1), k.size(2)
        keep = (mask == 0).squeeze(1).contiguous()   # (B, Sq, Sk) bool，True=可见

        def mask_mod(b, h, q_idx, kv_idx):          # 4 参数为官方要求签名（h 不参与）
            return keep[b, q_idx, kv_idx]

        block_mask = _create_block_mask_fn(
            mask_mod, B=B, H=H, Q_LEN=Sq, KV_LEN=Sk, device="cuda")

        def make(kopts):
            kwargs = {"kernel_options": kopts} if kopts else {}

            def run():
                return _flex_compiled(q, k, v, block_mask=block_mask,
                                      enable_gqa=(H != Hk), **kwargs)
            return run

        # 先试默认配置；smem 超限的架构/shape 逐级降级 BLOCK_M
        for kopts in (None, {"BLOCK_M": 64}, {"BLOCK_M": 32}):
            try:
                run = make(kopts)
                run()
                return run
            except Exception:
                continue
        return None
    except Exception:
        return None


def ref_attn(q, k, v, mask, scale):
    """Manual softmax attention reference (fp32), supports GQA."""
    H, Hk = q.size(1), k.size(1)
    if H != Hk:
        k = k.repeat_interleave(H // Hk, dim=1)
        v = v.repeat_interleave(H // Hk, dim=1)
    qt = q * scale
    scores = torch.matmul(qt, k.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask
    scores = scores.float()
    scores = scores - scores.amax(dim=-1, keepdim=True)
    scores = torch.exp(scores)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    out = torch.matmul(scores.to(q.dtype), v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(name, shapes, args):
    print(f"\n{'=' * 130}")
    print(f"  {name}   (mask_ratio={args.mask_ratio}, warmup={args.warmup}, iters={args.iters})")
    print(f"{'=' * 130}")
    header = (
        f"{'shape (B,H,Hk,Sq,Sk,d)':<34}"
        f"{'custom':>12}{'custom TF':>11}{'GB/s':>8}"
        f"{'SDPA+mask':>12}{'speedup':>9}"
        f"{'FlexAtt':>12}{'vs Flex':>9}"
        f"{'SDPA flash':>12}"
        f"{'grid':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for (B, H, Hk, Sq, Sk, d) in shapes:
        shape_str = f"{B},{H},{Hk},{Sq},{Sk},{d}"
        q, k, v, mask = make_inputs(B, H, Hk, Sq, Sk, d, args.mask_ratio, args.seed)

        # 正确性快速校验
        scale = 1.0 / (d ** 0.5)
        out_custom = ops.mha_fwd_with_mask(q, k, v, mask)
        with torch.no_grad():
            out_ref = ref_attn(q.float(), k.float(), v.float(), mask.float(), scale).half()
        max_diff = (out_custom.float() - out_ref.float()).abs().max().item()
        status = "OK" if max_diff < 0.02 else f"WARN diff={max_diff:.4f}"

        t_custom = bench_us(lambda: ops.mha_fwd_with_mask(q, k, v, mask),
                            args.warmup, args.iters)
        t_sdpa = bench_us(lambda: sdpa_with_mask(q, k, v, mask),
                          args.warmup, args.iters)

        flex_runner = None
        if args.use_flex:
            flex_runner = flex_make_runner(q, k, v, mask)
        t_flex = bench_us(flex_runner, args.warmup, args.iters) \
            if flex_runner is not None else float("nan")
        flex_speedup = t_flex / t_custom \
            if flex_runner is not None else float("nan")

        t_flash = bench_us(lambda: sdpa_flash_no_mask(q, k, v),
                           args.warmup, args.iters)

        tf = tflops(B, H, Sq, Sk, d, t_custom)
        bw = gbps(B, H, Hk, Sq, Sk, d, t_custom)
        speedup = t_sdpa / t_custom
        num_m_blocks = (Sq + 63) // 64
        grid = B * H * num_m_blocks

        flex_str = f"{t_flex:>9.1f}µs{flex_speedup:>8.2f}x" \
            if flex_runner is not None else f"{'N/A':>12}{'':>9}"
        print(f"{shape_str:<34}"
              f"{t_custom:>9.1f}µs{tf:>9.1f}{bw:>7.0f}"
              f"{t_sdpa:>9.1f}µs{speedup:>8.2f}x"
              f"{flex_str}"
              f"{t_flash:>10.1f}µs"
              f"{grid:>8}  [{status}]")
        rows.append((shape_str, t_custom, tf, bw, t_sdpa, speedup,
                     t_flash, t_flex, flex_speedup, grid, max_diff))

        del q, k, v, mask, out_custom, out_ref
        torch.cuda.empty_cache()

    return rows


def main():
    parser = argparse.ArgumentParser(description="sm70 (V100) mha_fwd_with_mask benchmark")
    parser.add_argument("--suite", choices=["all", "standard", "small_grid", "ragged"],
                        default="all", help="跑哪组 shape（默认 all）")
    parser.add_argument("--shape", nargs=6, type=int, metavar=("B", "H", "Hk", "Sq", "Sk", "d"),
                        help="指定单个 shape，覆盖 --suite")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.1,
                        help="随机 mask 中被屏蔽位置的比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="", help="结果写入 CSV 文件路径")
    parser.add_argument("--no-flex", dest="use_flex", action="store_false", default=True,
                        help="跳过 FlexAttention 对照（默认尝试，不可用自动 N/A）")
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
    print(f"dtype: float16 (fp16)   MMA: WMMA m16n16k16   tile: (64,64)   threads: 512/CTA")
    print("* SDPA flash 列为不带 mask 的参照（不支持任意 mask）；公平对照是 SDPA+mask 列。")
    if FLEX_AVAILABLE:
        print("* FlexAtt = torch.nn.attention.flex_attention（官方自定义 mask 算子，"
              "block_mask 预构建、编译开销不计入计时）。")
    else:
        print("* FlexAttention 不可用（需要 torch>=2.5），对应列输出 N/A。")

    all_rows = []
    if args.shape:
        all_rows += run_suite("custom shape", [tuple(args.shape)], args)
    else:
        if args.suite in ("all", "standard"):
            all_rows += run_suite("标准场景（大 grid，SM 充分利用）",
                                  STANDARD_SHAPES, args)
        if args.suite in ("all", "small_grid"):
            all_rows += run_suite("小 grid 场景（SM 未填满）",
                                  SMALL_GRID_SHAPES, args)
        if args.suite in ("all", "ragged"):
            all_rows += run_suite("ragged 场景（Sk%8!=0 自动 pad / 任意 Sq）",
                                  RAGGED_SHAPES, args)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("shape,custom_us,custom_tflops,gbps,sdpa_mask_us,speedup,"
                    "sdpa_flash_us,flex_us,flex_speedup,grid,max_diff\n")
            for r in all_rows:
                f.write(",".join(str(x) for x in r) + "\n")
        print(f"\n结果已写入 {args.csv}")


if __name__ == "__main__":
    main()
