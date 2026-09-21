"""
benchmark/fa_tilelang.py — Tile-lang 版 mha_fwd_with_mask 与 CuTe 版性能对比

Tile-lang（https://github.com/tile-ai/tile-lang）是一个 tile-level GPU DSL，
基于 TVM 技术栈，宣传可达到手写 kernel 的性能。kernel 实现已沉淀为
`custom_ops.tilelang_ops`（算子库的 Tile-lang 后端，`backend="tilelang"`
即可选用），与 custom_ops.ops.mha_fwd_with_mask 完全同语义：

  - q: (B, H, Sq, d)  k/v: (B, Hk, Sk, d)  mask: (B, 1, Sq_m, Sk_m) 加性 mask
  - softmax(Q·K^T * scale + mask) · V，全屏蔽行输出 0
  - Sk % 8 == 0，Sq 任意；mask 可比 (Sq, Sk) 大，越界列内容被忽略（恒屏蔽）
  - GQA：H // Hk 组共享 KV 头
  - 自适应 Split-KV（小 grid 长序列）

运行方式
--------
    # 正确性（与 fp32 参考对拍，覆盖全屏蔽行/预 pad mask/GQA/非对齐形状）
    python benchmark/fa_tilelang.py --test

    # 性能对比（Tile-lang 多 config 自动择优 vs CuTe 版）
    python benchmark/fa_tilelang.py --bench
    python benchmark/fa_tilelang.py --bench --suite splitkv
    python benchmark/fa_tilelang.py --bench --shape 4 16 16 2048 2048 128
"""

import argparse
import os
import sys

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

import tilelang  # noqa: E402

# kernel 实现沉淀在包内 Tile-lang 后端；本文件只保留正确性对拍与调优逻辑
from custom_ops.tilelang_ops._fa import (  # noqa: E402
    candidate_configs,
    fa_tilelang,
    make_fa_mask,
    make_fa_split,
    split_candidate_configs,
)


# ─────────────────────────────────────────────────────────────────────────────
# 参考实现（与 tests/test_mha_fwd_with_mask.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

def ref_attn(q, k, v, mask=None):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
    if mask is not None:
        scores = scores + mask.float()
    scores = scores - scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    scores = torch.exp(scores)
    denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-38)
    return torch.matmul((scores / denom).to(v.dtype), v)


# ─────────────────────────────────────────────────────────────────────────────
# 正确性测试
# ─────────────────────────────────────────────────────────────────────────────

ATOL = 1e-2
RTOL = 1e-2


def _run_case(name, q, k, v, mask, config=None):
    out = fa_tilelang(q, k, v, mask, config)
    torch.cuda.synchronize()
    # GQA：参考实现需要扩展 KV 头
    H, Hk = q.shape[1], k.shape[1]
    if H != Hk:
        rep = H // Hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    ref = ref_attn(q, k, v, mask).to(q.dtype)
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    ref_abs = ref.float().abs().max().item() + 1e-8
    passed = max_err < ATOL or (max_err / ref_abs) < RTOL
    print(f"[{'PASS' if passed else 'FAIL'}] {name:48s} max_err={max_err:.4e}")
    return passed


def test_case(b, h, hk, sq, sk, d, mask_kind, dtype, name, config=None):
    q = (torch.randn(b, h, sq, d, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(b, hk, sk, d, device="cuda") * 0.5).to(dtype)
    v = (torch.randn(b, hk, sk, d, device="cuda") * 0.5).to(dtype)
    if mask_kind == "zero":
        mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
    elif mask_kind == "causal":
        tri = torch.triu(torch.full((sq, sk), float("-inf"), device="cuda"), diagonal=1)
        mask = tri.unsqueeze(0).unsqueeze(0).expand(b, 1, sq, sk).contiguous().to(dtype)
    elif mask_kind == "random":
        mask = (torch.randn(b, 1, sq, sk, device="cuda") * 0.3).to(dtype)
    elif mask_kind == "block":
        mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
        mask[..., sk // 2:] = float("-inf")
    else:  # all_masked_row
        mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
        mask[:, :, 0, :] = float("-inf")
    return _run_case(name, q, k, v, mask, config)


def test_mask_padded(b, h, hk, sq, sk, d, dtype, pad_rows=0):
    q = (torch.randn(b, h, sq, d, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(b, hk, sk, d, device="cuda") * 0.5).to(dtype)
    v = (torch.randn(b, hk, sk, d, device="cuda") * 0.5).to(dtype)
    mask = (torch.randn(b, 1, sq + pad_rows, sk + 8, device="cuda") * 0.3).to(dtype)
    out = fa_tilelang(q, k, v, mask)
    torch.cuda.synchronize()
    ref = ref_attn(q, k, v, mask[:, :, :sq, :sk].contiguous()).to(dtype)
    max_err = (out.float() - ref.float()).abs().max().item()
    passed = max_err < ATOL
    print(f"[{'PASS' if passed else 'FAIL'}] "
          f"mask_padded(rows=+{pad_rows}, cols=+8)[{str(dtype).split('.')[-1]:8s}]"
          f"{'':8s} max_err={max_err:.4e}")
    return passed


def run_tests():
    print("=" * 80)
    print("Tile-lang FA-with-mask correctness test")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"cap: {torch.cuda.get_device_capability(0)}  tilelang: {tilelang.__version__}")
    print("=" * 80)
    all_pass = True
    # 每种 dtype 用第一个碰到的 config；非对齐形状再单独验证
    cfg0 = dict(block_M=64, block_N=64, num_stages=2, threads=128)
    cfg1 = dict(block_M=128, block_N=64, num_stages=2, threads=256)
    for dtype in (torch.float16, torch.bfloat16):
        tag = str(dtype).split(".")[-1]
        for cfg, cfgname in ((cfg0, "c64x64"), (cfg1, "c128x64")):
            all_pass &= test_case(2, 16, 16, 256, 256, 64, "zero", dtype,
                                  f"base_{tag}_d64_{cfgname}", cfg)
            all_pass &= test_case(2, 16, 16, 256, 256, 128, "zero", dtype,
                                  f"base_{tag}_d128_{cfgname}", cfg)
        all_pass &= test_case(2, 16, 16, 256, 256, 64, "causal", dtype, f"causal_{tag}_d64")
        all_pass &= test_case(2, 16, 16, 256, 256, 128, "random", dtype, f"random_{tag}_d128")
        all_pass &= test_case(2, 16, 16, 256, 256, 64, "block", dtype, f"block_{tag}_d64")
        all_pass &= test_case(2, 16, 16, 256, 256, 64, "all_masked_row", dtype, f"allmask_{tag}_d64")
        # 非块对齐：Sq 任意（100/333/1），Sk 为 8 倍数但非块倍数
        all_pass &= test_case(1, 8, 8, 100, 104, 64, "zero", dtype, f"unaligned_{tag}_d64")
        all_pass &= test_case(1, 8, 8, 100, 200, 128, "random", dtype, f"unaligned_{tag}_d128")
        all_pass &= test_case(1, 8, 8, 64, 56, 64, "random", dtype, f"tail_single_{tag}_d64")
        all_pass &= test_case(1, 4, 4, 1, 104, 64, "random", dtype, f"sq1_{tag}_d64")
        all_pass &= test_case(1, 8, 8, 256, 1032, 128, "random", dtype, f"tail8_{tag}_d128")
        # 预 pad mask：越界列填随机陷阱值，必须被忽略
        all_pass &= test_mask_padded(1, 8, 8, 100, 104, 64, dtype)
        all_pass &= test_mask_padded(1, 8, 8, 100, 104, 64, dtype, pad_rows=8)
        # GQA
        all_pass &= test_case(2, 16, 4, 256, 256, 64, "zero", dtype, f"gqa(16,4)_{tag}_d64")
        all_pass &= test_case(1, 32, 8, 512, 384, 128, "causal", dtype, f"gqa(32,8)_{tag}_d128")
        # ragged Sk（Sk%8!=0 → 前端自动 pad，与 CUDA 版同语义）
        all_pass &= test_case(1, 8, 8, 256, 2053, 64, "random", dtype,
                              f"raggedsk_{tag}_d64")
        all_pass &= test_case(1, 4, 4, 128, 1031, 128, "zero", dtype,
                              f"raggedsk_{tag}_d128")
        # 长序列 splitkv 形状（单 kernel 路径）
        all_pass &= test_case(1, 2, 2, 128, 8192, 128, "zero", dtype, f"longkv_{tag}_d128")
        # Split-KV 路径：非 2 幂 split（不均匀 tile 划分）+ GQA + 全屏蔽行
        cfg_sp = dict(block_M=128, block_N=64, num_stages=2, threads=256, num_split=3)
        all_pass &= test_case(1, 2, 2, 128, 8192, 128, "zero", dtype,
                              f"split3_{tag}_d128", cfg_sp)
        all_pass &= test_case(1, 1, 1, 128, 8192, 128, "random", dtype,
                              f"split3r_{tag}_d128", cfg_sp)
        all_pass &= test_case(1, 1, 1, 128, 8192, 128, "all_masked_row", dtype,
                              f"split3all_{tag}_d128", cfg_sp)
        all_pass &= test_case(1, 2, 1, 512, 16384, 128, "causal", dtype,
                              f"split3gqa_{tag}_d128", cfg_sp)
        # num_split > K tile 数（大量空 split）
        all_pass &= test_case(1, 1, 1, 128, 8192, 128, "zero", dtype,
                              f"split_many_{tag}_d128", dict(cfg_sp, num_split=200))
    print("=" * 80)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 80)
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 性能对比
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_SHAPES = [
    (1, 16, 16, 1024, 1024, 64),
    (4, 16, 16, 2048, 2048, 64),
    (1, 8, 8, 8192, 8192, 64),
    (32, 16, 16, 1024, 1024, 64),
    (1, 16, 16, 1024, 1024, 128),
    (4, 16, 16, 2048, 2048, 128),
    (1, 8, 8, 8192, 8192, 128),
    (4, 16, 4, 2048, 2048, 128),   # GQA
    (32, 16, 16, 1024, 1024, 128),
]

SPLITKV_SHAPES = [
    (1, 1, 1, 128, 8192, 128),
    (1, 1, 1, 128, 32768, 128),
    (1, 1, 1, 512, 8192, 128),
    (1, 1, 1, 1024, 8192, 128),
    (1, 2, 1, 512, 16384, 128),    # GQA
    (1, 1, 1, 128, 8192, 64),
    (1, 1, 1, 1024, 8192, 64),
    (1, 2, 1, 512, 16384, 64),     # GQA
]

# 与 benchmark_fa.py 的 RAGGED_SHAPES 同定义：Sk%8!=0（内部自动 pad）、
# 任意 Sq（行谓词裁剪）、Sq+Sk 双 ragged、GQA+ragged。
# 对齐参照行与 ragged 行相邻排列，直观评估非对齐开销。
RAGGED_SHAPES = [
    (1,  8,  8, 2048, 2048, 64),    # 对齐参照
    (1,  8,  8, 2048, 2053, 64),    # Sk%8=5 → pad 到 2056
    (1, 16, 16, 1024, 1024, 128),   # 对齐参照
    (1, 16, 16, 1024, 1031, 128),   # Sk%8=7 → pad 到 1032
    (4, 16, 16, 2048, 4096, 128),   # 对齐参照
    (4, 16, 16, 2048, 4099, 128),   # Sk%8=3 → pad 到 4104
    (1,  8,  8, 8192, 8195, 64),    # 长 Sk ragged
    # ragged Sq（任意 Sq，行谓词裁剪）
    (1, 16, 16, 1000, 1024, 64),
    (2, 16, 16,  333, 1024, 64),
    (1,  8,  8,  127, 2048, 128),
    (1,  8,  8,  129, 2048, 128),
    # Sq + Sk 双 ragged、GQA + ragged
    (1, 16, 16, 1000, 4099, 128),
    (4, 16,  4, 2048, 2053, 128),
]


def make_inputs(B, H, Hk, Sq, Sk, d, mask_ratio=0.1, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, Hk, Sk, d, device="cuda", dtype=torch.bfloat16, generator=g)
    mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16)
    if mask_ratio > 0:
        drop = torch.rand(B, 1, Sq, Sk, device="cuda", generator=g) < mask_ratio
        drop[..., 0] = False
        mask = mask.masked_fill(drop, float("-inf"))
    return q, k, v, mask


def bench_us(fn, warmup, iters):
    """与 benchmark_fa.py 相同：CUDA Event + L2 flush，取中位数。"""
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        cache.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2] * 1e3


def tflops(B, H, Sq, Sk, d, time_us):
    return 4.0 * B * H * Sq * Sk * d / (time_us * 1e-6) / 1e12


def bench_tilelang(q, k, v, mask, warmup, iters, verbose=True):
    """对候选 config（单 kernel + Split-KV）做完整 sweep，返回最优 (config, µs)。

    Sk % 8 != 0（ragged）：sweep 用 pad 后输入（kernel 契约），最终耗时
    走 fa_tilelang 全路径重计（含自动 pad 拷贝），与 CUDA 版端到端对齐。
    """
    B, H, Sq, d = q.shape
    Hk, Sk = k.shape[1], k.shape[2]
    dtype = {torch.float16: "float16", torch.bfloat16: "bfloat16"}[q.dtype]
    ragged = Sk % 8 != 0
    if ragged:
        pad = (Sk + 7) // 8 * 8 - Sk
        k_t = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v_t = torch.nn.functional.pad(v, (0, 0, 0, pad))
        mask_t = torch.nn.functional.pad(mask, (0, pad), value=float("-inf"))
        Sk_t = Sk + pad
    else:
        k_t, v_t, mask_t, Sk_t = k, v, mask, Sk
    best = (None, float("inf"))
    all_cfgs = [(c, False) for c in candidate_configs(d)] + \
               [(c, True) for c in split_candidate_configs(B, H, Hk, Sq, Sk_t, d)]
    for cfg, is_split in all_cfgs:
        try:
            if is_split:
                kernel = make_fa_split(B, H, Hk, Sq, Sk_t, d, Sq, Sk_t, dtype, **cfg)
            else:
                kernel = make_fa_mask(B, H, Hk, Sq, Sk_t, d, Sq, Sk_t, dtype, **cfg)
            t = bench_us(lambda: kernel(q, k_t, v_t, mask_t), warmup, iters)
        except Exception as e:  # smem 超限等
            if verbose:
                print(f"    config {cfg} failed: {type(e).__name__}: {e}")
            continue
        if verbose:
            print(f"    {cfg} -> {t:.1f}µs")
        if t < best[1]:
            best = (cfg, t)
    if ragged and best[0] is not None:
        # ragged：全路径重计（pad 拷贝计入计时）
        cfg = best[0]
        t = bench_us(lambda: fa_tilelang(q, k, v, mask, cfg), warmup, iters)
        best = (cfg, t)
    return best


def run_bench(args):
    from custom_ops import ops

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})  "
          f"PyTorch: {torch.__version__}  tilelang: {tilelang.__version__}")

    shapes = {"standard": STANDARD_SHAPES,
              "splitkv": SPLITKV_SHAPES,
              "ragged": RAGGED_SHAPES}[args.suite]
    if args.shape:
        shapes = [tuple(args.shape)]

    header = (f"{'shape (B,H,Hk,Sq,Sk,d)':<32}"
              f"{'Tile-lang':>10}{'TL TF':>8}{'TL cfg':>16}"
              f"{'CuTe':>10}{'CuTe TF':>9}{'TL/CuTe':>9}")
    print(header)
    print("-" * len(header))

    for (B, H, Hk, Sq, Sk, d) in shapes:
        shape_str = f"{B},{H},{Hk},{Sq},{Sk},{d}"
        q, k, v, mask = make_inputs(B, H, Hk, Sq, Sk, d, args.mask_ratio, args.seed)

        # Tile-lang：sweep config 择优（编译/试跑均在计时外）
        cfg, t_tl = bench_tilelang(q, k, v, mask, args.warmup, args.iters, args.verbose)
        # CuTe 版
        t_cute = bench_us(lambda: ops.mha_fwd_with_mask(q, k, v, mask),
                          args.warmup, args.iters)

        # 正确性交叉校验
        out_tl = fa_tilelang(q, k, v, mask, cfg)
        out_cute = ops.mha_fwd_with_mask(q, k, v, mask)
        max_diff = (out_tl.float() - out_cute.float()).abs().max().item()
        status = "OK" if max_diff < 0.02 else f"DIFF={max_diff:.3f}"

        cfg_str = (f"{cfg['block_M']}x{cfg['block_N']}s{cfg['num_stages']}"
                   + ("m" if cfg.get("mask_smem") else "")
                   + (f"/sp{cfg['num_split']}" if cfg.get("num_split", 1) > 1 else "")
                   ) if cfg else "-"
        print(f"{shape_str:<32}"
              f"{t_tl:>8.1f}µs{tflops(B, H, Sq, Sk, d, t_tl):>7.1f}{cfg_str:>16}"
              f"{t_cute:>8.1f}µs{tflops(B, H, Sq, Sk, d, t_cute):>8.1f}"
              f"{t_cute / t_tl:>8.2f}x   [{status}]")

        del q, k, v, mask, out_tl, out_cute
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Tile-lang FA with mask: test & benchmark")
    parser.add_argument("--test", action="store_true", help="运行正确性测试")
    parser.add_argument("--bench", action="store_true", help="运行性能对比")
    parser.add_argument("--suite", choices=["standard", "splitkv", "ragged"],
                        default="standard")
    parser.add_argument("--shape", nargs=6, type=int,
                        metavar=("B", "H", "Hk", "Sq", "Sk", "d"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false", default=True,
                        help="不打印每个 config 的 sweep 明细")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用。")
        sys.exit(1)

    if args.test:
        ok = run_tests()
        if not (args.bench and ok):
            sys.exit(0 if ok else 1)
    if args.bench:
        run_bench(args)


if __name__ == "__main__":
    main()
