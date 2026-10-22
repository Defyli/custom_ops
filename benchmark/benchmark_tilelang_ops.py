"""
benchmark/benchmark_tilelang_ops.py — Tile-lang 后端 vs CUDA 后端性能对比

覆盖 mixed_gemm / fuse_moe / swiglu（FA 的对比见 benchmark/fa_tilelang.py）。
每个 shape 输出：
  - CUDA 后端（ops.<op>(..., backend="cuda")）耗时
  - Tile-lang 后端多 config sweep 择优耗时
  - 加速比（CUDA / Tile-lang）与交叉正确性校验

suite：
  regular — 常规对齐 shape（默认）
  ragged  — ragged 场景：mixed_gemm 非 64/128 对齐 M/N/K（tile 边界 ±1）、
            fuse_moe 倾斜/zipf 路由（token 不均匀分布）、swiglu ragged M
  all     — 两者全跑

运行：
    python benchmark/benchmark_tilelang_ops.py                        # regular
    python benchmark/benchmark_tilelang_ops.py --suite ragged
    python benchmark/benchmark_tilelang_ops.py --ops mixed_gemm fuse_moe
    python benchmark/benchmark_tilelang_ops.py --warmup 20 --iters 100

standalone CUDA：--repeats（默认 5）仅作用于 --backend cuda；compare 忽略它。
每个 shape 共享输入/路由，各 repeat 确定性轮转模式顺序，每组每轮先 warmup，
分别测 GPU event 和 host wall 批次，旧耗时字段表示 median of batch means。
FUSE_MOE_TIME 只要存在即属于 diagnostic（原生按存在性开启），不可用作正常性能。
JSON 使用排他创建，拒绝覆盖已有证据；GPU clock/power 由调用方外部记录。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops, split_mixed_precision_weight  # noqa: E402

def bench_us(fn, warmup, iters):
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


# ── mixed_gemm ───────────────────────────────────────────────────────────────

MIXED_GEMM_SHAPES = [
    (16, 4096, 4096),
    (64, 4096, 4096),
    (256, 4096, 4096),
    (1024, 4096, 4096),
    (4096, 4096, 4096),
    (512, 4096, 256),
    (128, 1000, 2048),
]

# ragged：围绕 tile 边界 64/128 的 ±1 位置（同 tests/test_mixed_gemm_ragged
# 的语义）；K 约束 %16==0（CUDA 版 TMA 路径契约）
MIXED_GEMM_RAGGED_SHAPES = [
    (127,  999, 2048),   # M/N 双 ragged（128/64 边界 -1）
    (513, 4096,  256),   # M ragged（512+1）
    (128, 4097, 2048),   # N ragged（4096+1）
    (256, 2048, 2000),   # K ragged（%16==0 但非 64 倍数）
    (129, 1000, 2064),   # M/N/K 全 ragged
    (1024, 1000, 192),   # 深 skinny + ragged N
]


def bench_mixed_gemm(args, shapes=None):
    from custom_ops import tilelang_ops
    from custom_ops.tilelang_ops import _mixed_gemm

    shapes = shapes if shapes is not None else MIXED_GEMM_SHAPES
    print("=" * 96)
    print("mixed_gemm  (fp8 residual, silu) — CUDA vs Tile-lang")
    header = f"{'shape (M,N,K)':<22}{'CUDA µs':>10}{'TileLang µs':>12}{'CUDA/TL':>9}{'TL cfg':>22}{'check':>8}"
    print(header)
    print("-" * len(header))
    for (M, N, K) in shapes:
        x = torch.randn(M, K, device="cuda") * 0.5
        w = torch.randn(N, K, device="cuda") * 0.05
        w_high, w_low, w_scale = split_mixed_precision_weight(w, backend="fp8")
        try:
            t_cuda = bench_us(
                lambda: ops.mixed_gemm(x, w_high, w_low, None,
                                       activation="silu", backend="cuda"),
                args.warmup, args.iters)
        except Exception as e:
            print(f"{str((M, N, K)):<22} CUDA 不可用: {type(e).__name__}")
            continue

        best = (None, float("inf"))
        for cfg in _mixed_gemm.candidate_configs(M, N, K):
            try:
                t = bench_us(
                    lambda: tilelang_ops.mixed_gemm(
                        x, w_high, w_low, None, activation="silu", config=cfg),
                    args.warmup, args.iters)
            except Exception:
                continue
            if args.verbose:
                print(f"    {cfg} -> {t:.1f}µs")
            if t < best[1]:
                best = (cfg, t)
        cfg, t_tl = best

        y1 = ops.mixed_gemm(x, w_high, w_low, None, activation="silu",
                            backend="cuda")
        y2 = tilelang_ops.mixed_gemm(x, w_high, w_low, None,
                                     activation="silu", config=cfg)
        err = (y1.float() - y2.float()).abs().max().item()
        status = "OK" if err < 0.05 else f"DIFF={err:.3f}"
        cfg_str = (f"{cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']}"
                   f"/s{cfg['num_stages']}") if cfg else "-"
        print(f"{str((M, N, K)):<22}{t_cuda:>9.1f} {t_tl:>10.1f} "
              f"{t_cuda / t_tl:>8.2f}x {cfg_str:>22}{status:>8}")


# ── fuse_moe ─────────────────────────────────────────────────────────────────

MOE_SHAPES = [
    # (S, H, I, E, K)
    (128, 2048, 1024, 8, 2),
    (1024, 2048, 1024, 8, 2),
    (4096, 2048, 1024, 8, 2),
    (16384, 2048, 1024, 8, 2),
    (128, 2048, 1024, 64, 2),
    (4096, 2048, 1024, 64, 8),
    (4096, 4096, 1408, 8, 2),
]

# ragged：token 不均匀分布到 expert——skewed（90% 集中到 expert0）与
# zipf（p_e ∝ 1/(e+1)，长尾倾斜）；ragged S（非 2 的幂）
# (S, H, I, E, K, routing)
MOE_RAGGED_CASES = [
    (4096,  2048, 1024,  8, 2, "skewed"),
    (4096,  2048, 1024, 64, 2, "skewed"),
    (4096,  2048, 1024,  8, 2, "zipf"),
    (4096,  2048, 1024, 64, 2, "zipf"),
    (16384, 2048, 1024,  8, 2, "zipf"),
    (4097,  2048, 1024,  8, 2, "zipf"),   # ragged S
]


def make_moe_inputs(S, H, I, E, K, dtype=torch.bfloat16, seed=42,
                    routing="uniform", expert_type="glu"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(S, H, device="cuda", dtype=dtype, generator=g) * 0.3
    w1 = (torch.randn(E, (2 if expert_type == "glu" else 1) * I, H,
                      device="cuda", dtype=dtype, generator=g) * 0.05)
    w2 = (torch.randn(E, H, I, device="cuda", dtype=dtype, generator=g) * 0.05)
    if routing == "uniform":
        ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32,
                            generator=g)
    elif routing == "skewed":       # 90% token 落到 expert 0
        u = torch.rand(S, K, device="cuda", generator=g)
        hot = torch.zeros(S, K, device="cuda", dtype=torch.int32)
        rest = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32,
                             generator=g)
        ids = torch.where(u < 0.9, hot, rest)
    else:                            # zipf：p_e ∝ 1/(e+1)，长尾倾斜
        w = 1.0 / torch.arange(1, E + 1, device="cuda", dtype=torch.float64)
        cdf = (w / w.sum()).cumsum(0)
        u = torch.rand(S, K, device="cuda", generator=g).double()
        ids = torch.searchsorted(cdf, u).clamp(max=E - 1).to(torch.int32)
    scale = torch.rand(S, K, device="cuda", generator=g)
    return x, w1, w2, ids, scale


def bench_fuse_moe(args, cases=None):
    from custom_ops import tilelang_ops
    from custom_ops.tilelang_ops import _fuse_moe

    cases = cases if cases is not None else MOE_SHAPES
    print("=" * 96)
    print("fuse_moe  (bf16) — CUDA vs Tile-lang")
    header = f"{'shape (S,H,I,E,K)':<24}{'CUDA µs':>10}{'TileLang µs':>12}{'CUDA/TL':>9}{'TL cfg':>24}{'check':>8}"
    print(header)
    print("-" * len(header))
    for case in cases:
        if len(case) == 6:
            S, H, I, E, K, routing = case
        else:
            (S, H, I, E, K), routing = case, "uniform"
        x, w1, w2, ids, scale = make_moe_inputs(S, H, I, E, K, routing=routing)
        try:
            t_cuda = bench_us(
                lambda: ops.fuse_moe(x, w1, w2, ids, scale, backend="cuda"),
                args.warmup, args.iters)
        except Exception as e:
            print(f"{str((S, H, I, E, K)):<24} CUDA 不可用: {type(e).__name__}")
            continue

        best = (None, float("inf"))
        for cfg in _fuse_moe.candidate_configs(S, K, E):
            try:
                t = bench_us(
                    lambda: tilelang_ops.fuse_moe(x, w1, w2, ids, scale,
                                                  config=cfg),
                    args.warmup, args.iters)
            except Exception:
                continue
            if args.verbose:
                print(f"    {cfg} -> {t:.1f}µs")
            if t < best[1]:
                best = (cfg, t)
        cfg, t_tl = best

        y1 = ops.fuse_moe(x, w1, w2, ids, scale, backend="cuda")
        y2 = tilelang_ops.fuse_moe(x, w1, w2, ids, scale, config=cfg)
        err = (y1.float() - y2.float()).abs().max().item()
        status = "OK" if err < 0.05 else f"DIFF={err:.3f}"
        cfg_str = (f"BM{cfg['block_M']}/N1-{cfg['block_N1']}/N2-{cfg['block_N2']}"
                   f"/s{cfg['num_stages1']}{cfg['num_stages2']}") if cfg else "-"
        print(f"{str((S, H, I, E, K)) + '/' + routing:<24}{t_cuda:>9.1f} {t_tl:>10.1f} "
              f"{t_cuda / t_tl:>8.2f}x {cfg_str:>24}{status:>8}")


# ── swiglu ───────────────────────────────────────────────────────────────────

SWIGLU_SHAPES = [
    (128, 4096, 4096),
    (1024, 2048, 4096),
    (4096, 4096, 4096),
    (16384, 2048, 4096),
]

# ragged：任意 M（N%64==0、K%64==0 是算子契约，不可 ragged）
SWIGLU_RAGGED_SHAPES = [
    (1023, 2048, 4096),   # 1024-1
    (4097, 2048, 4096),   # 4096+1
    (1000, 4096, 4096),   # 非对齐基数
]


def bench_swiglu(args, shapes=None):
    from custom_ops import tilelang_ops
    from custom_ops.tilelang_ops import _swiglu

    shapes = shapes if shapes is not None else SWIGLU_SHAPES
    print("=" * 96)
    print("swiglu  (bf16) — CUDA vs Tile-lang")
    header = f"{'shape (M,N,K)':<22}{'CUDA µs':>10}{'TileLang µs':>12}{'CUDA/TL':>9}{'TL cfg':>22}{'check':>8}"
    print(header)
    print("-" * len(header))
    for (M, N, K) in shapes:
        x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5)
        w = (torch.randn(2 * N, K, device="cuda", dtype=torch.bfloat16) * 0.05)
        try:
            t_cuda = bench_us(lambda: ops.swiglu(x, w, backend="cuda"),
                              args.warmup, args.iters)
        except Exception as e:
            print(f"{str((M, N, K)):<22} CUDA 不可用: {type(e).__name__}")
            continue

        best = (None, float("inf"))
        for cfg in _swiglu.candidate_configs(M, N, K):
            try:
                t = bench_us(lambda: tilelang_ops.swiglu(x, w, config=cfg),
                             args.warmup, args.iters)
            except Exception:
                continue
            if args.verbose:
                print(f"    {cfg} -> {t:.1f}µs")
            if t < best[1]:
                best = (cfg, t)
        cfg, t_tl = best

        y1 = ops.swiglu(x, w, backend="cuda")
        y2 = tilelang_ops.swiglu(x, w, config=cfg)
        err = (y1.float() - y2.float()).abs().max().item()
        status = "OK" if err < 0.05 else f"DIFF={err:.3f}"
        cfg_str = (f"{cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']}"
                   f"/s{cfg['num_stages']}"
                   + ("/FR" if cfg.get("fullrow") else "")) if cfg else "-"
        print(f"{str((M, N, K)):<22}{t_cuda:>9.1f} {t_tl:>10.1f} "
              f"{t_cuda / t_tl:>8.2f}x {cfg_str:>22}{status:>8}")


def parse_moe_shape(value):
    """CLI shape order: S,H,I,E,K; enforce the standalone CUDA contract."""
    try:
        shape = tuple(int(dim) for dim in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be S,H,I,E,K integers") from error
    if len(shape) != 5 or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape must contain five positive integers")
    _, H, I, E, K = shape
    if H % 64 or I % 64 or E > 512 or K > 128:
        raise argparse.ArgumentTypeError("require H/I % 64 == 0, E <= 512, K <= 128")
    return shape


@torch.no_grad()
def bench_cuda_timings(fn, warmup, iters):
    """Separate steady-state batches; host wall includes final GPU completion.

    No cache flush, CUDA graphs, compilation, input creation or library loading
    is included. Both values are amortized microseconds per forward invocation.
    numerical_mode='staged' specifies intermediate dtype rounding for the forward call.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize timing events outside the measured region, including warmup=0.
    start.record()
    end.record()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    gpu_event_batch_us = start.elapsed_time(end) * 1e3
    gpu_event_us = gpu_event_batch_us / iters

    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    host_wall_batch_us = (time.perf_counter() - wall_start) * 1e6
    host_wall_us = host_wall_batch_us / iters
    return {"gpu_event_us": gpu_event_us, "host_wall_us": host_wall_us,
            "gpu_event_batch_us": gpu_event_batch_us,
            "host_wall_batch_us": host_wall_batch_us}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_existing_json(value):
    # lexists also rejects dangling symlinks, which are not safe output targets.
    if value and os.path.lexists(Path(value).expanduser()):
        raise FileExistsError(f"Refusing to overwrite benchmark evidence: {value}")


@contextmanager
def standalone_math():
    matmul = torch.backends.cuda.matmul
    fields = ("allow_tf32", "allow_bf16_reduced_precision_reduction",
              "allow_fp16_reduced_precision_reduction")
    previous = {key: getattr(matmul, key) for key in fields if hasattr(matmul, key)}
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        for key in previous:
            setattr(matmul, key, False)
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        for key, value in previous.items():
            setattr(matmul, key, value)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@torch.no_grad()
@standalone_math()
def bench_fuse_moe_cuda(args):
    """Precompiled-only standalone forward benchmark; never imports TileLang."""
    reject_existing_json(args.json)
    if args.repeats <= 0:
        raise ValueError("require --repeats > 0")
    started_at = datetime.now(timezone.utc).isoformat()
    fuse_moe_env = {key: value for key, value in sorted(os.environ.items())
                    if key.startswith("FUSE_MOE")}
    diagnostic = ("FUSE_MOE_TIME" in fuse_moe_env or
                  "CUDA_LAUNCH_BLOCKING" in os.environ or
                  "TORCH_USE_CUDA_DSA" in os.environ)
    timing_kind = ("diagnostic_serialized_launch" if diagnostic else
                   "steady_state_forward_separate_batches")
    if diagnostic:
        print("DIAGNOSTIC ONLY: timing/synchronization diagnostic environment is set; native stage stdout is not "
              "captured, effective TFLOPS is disabled; do not use as performance baseline.")
    ops.disable_jit()
    library = Path(args.ops_library).expanduser().resolve(strict=True)
    metadata = (Path(args.ops_metadata).expanduser().resolve(strict=True)
                if args.ops_metadata else Path(str(library) + ".json"))
    ops.load_precompiled("fuse_moe", library, metadata)
    with metadata.open(encoding="utf-8") as handle:
        sidecar = json.load(handle)
    artifacts = {
        "library": {"path": str(library), "sha256": file_sha256(library)},
        "sidecar": {"path": str(metadata), "sha256": file_sha256(metadata),
                    "source_fingerprint": sidecar.get("source_fingerprint"),
                    "build_fingerprint": sidecar.get("build_fingerprint"),
                    "build": sidecar.get("build"),
                    "cuda_arch_list": sidecar.get("cuda_arch_list")},
    }
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    modes = ([("silu", "fused"), ("silu", "staged"),
              ("gelu", "fused"), ("gelu", "staged")]
             if args.moe_mode_matrix else
             [(args.expert_activation, args.numerical_mode)])
    if args.moe_shapes:
        cases = [(*shape, args.routing) for shape in args.moe_shapes]
    else:
        cases = []
        if args.suite in ("regular", "all"):
            cases.extend((*shape, args.routing) for shape in MOE_SHAPES)
        if args.suite in ("ragged", "all"):
            cases.extend(MOE_RAGGED_CASES)
    results = []
    for S, H, I, E, K, routing in cases:
        # Reuse exactly the same seeded tensors for all four mode combinations.
        inputs = make_moe_inputs(S, H, I, E, K, dtype=dtype,
                                 seed=args.seed, routing=routing,
                                 expert_type=args.expert_type)
        samples = {mode: [] for mode in modes}
        repeat_orders = []
        for repeat in range(args.repeats):
            offset = repeat % len(modes)
            order = modes[offset:] + modes[:offset]
            repeat_orders.append([f"{act}+{num}" for act, num in order])
            for position, (activation, numerical) in enumerate(order):
                def invoke():
                    return ops.fuse_moe(*inputs, backend="cuda",
                                        expert_type=args.expert_type,
                                        expert_activation=activation,
                                        numerical_mode=numerical)

                # Immediate invocation keeps closure values fixed for this batch.
                timing = bench_cuda_timings(invoke, args.warmup, args.iters)
                samples[(activation, numerical)].append({
                    "repeat": repeat + 1, "order_index": position,
                    "timing_kind": timing_kind, **timing,
                })
        time_keys = ("gpu_event_us", "host_wall_us")
        aggregates = {
            mode: {key: median(sample[key] for sample in raw) for key in time_keys}
            for mode, raw in samples.items()
        }
        baseline = aggregates.get(("silu", "fused"))
        effective_flops = (6 if args.expert_type == "glu" else 4) * S * K * H * I
        for activation, numerical in modes:
            timing = aggregates[(activation, numerical)]
            row = {
                "op": "fuse_moe", "backend": "cuda",
                "shape": dict(zip(("S", "H", "I", "E", "K"), (S, H, I, E, K))),
                "dtype": args.dtype, "routing": routing, "seed": args.seed,
                "expert_type": args.expert_type,
                "mode_key": f"{args.expert_type}/{activation}/{numerical}",
                "expert_activation": activation, "numerical_mode": numerical,
                **timing,
                "timing_kind": timing_kind, "performance_eligible": not diagnostic,
                "raw_samples": samples[(activation, numerical)],
                "repeat_execution_orders": repeat_orders,
                "effective_flops": effective_flops,
                "effective_tflops": (
                    {key: effective_flops / (timing[key] * 1e6) for key in time_keys}
                    if not diagnostic else None),
                "relative_to_silu_fused": (
                    {key: timing[key] / baseline[key] for key in time_keys}
                    if baseline is not None and not diagnostic else None),
            }
            results.append(row)
            label = "DIAGNOSTIC" if diagnostic else "PERFORMANCE"
            print(f"[{label}] {(S, H, I, E, K)} / {routing} / {args.expert_type}/{activation}+{numerical}: "
                  f"GPU median {timing['gpu_event_us']:.2f} us, "
                  f"host wall median {timing['host_wall_us']:.2f} us "
                  f"({args.repeats} batch means)")
            if not diagnostic:
                print(f"  effective TFLOPS: GPU {row['effective_tflops']['gpu_event_us']:.3f}, "
                      f"host wall {row['effective_tflops']['host_wall_us']:.3f}; "
                      f"relative_to_silu_fused: {row['relative_to_silu_fused']}")
        del inputs
    report = {
        "schema_version": 3, "benchmark": "fuse_moe_standalone_cuda",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv), "python_executable": sys.executable,
        "cwd": os.getcwd(), "arguments": vars(args).copy(),
        "artifacts": artifacts, "fuse_moe_environment": fuse_moe_env,
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "TORCH_CUDA_ARCH_LIST",
            "CUDA_LAUNCH_BLOCKING")},
        "timing_kind": timing_kind, "performance_eligible": not diagnostic,
        "dtype_contracts": {"w1": "GLU(E,2I,H); MLP(E,I,H)", "w2": "(E,H,I)",
                            "expert_type_native_codes": {"glu": 0, "mlp": 1},
                            "tf32": False, "reduced_precision_reduction": False},
        "gpu_clock_power": {"collected": False, "source": "external recording required"},
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "ops_library": str(library), "ops_metadata": str(metadata), "allow_jit": False,
        "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats,
        "mode_matrix": args.moe_mode_matrix,
        "measurement": {
            "unit": "us/forward", "aggregation": "median_of_batch_means",
            "timing_kind": timing_kind,
            "legacy_fields": {key: "median_of_repeat_batch_means_us_per_forward"
                              for key in ("gpu_event_us", "host_wall_us")},
            "raw_sample_aggregation": "batch_mean",
            "raw_batch_unit": "us/batch",
            "gpu_event": "one event pair around iters forward calls; elapsed/iters",
            "host_wall": "separate iters batch including final CUDA synchronize; wall/iters",
            "warmup_policy": "each mode in each repeat, before both timed batches",
            "repeat_order": "deterministic left rotation by repeat index modulo mode count",
            "relative_to_silu_fused": "ratio of mode median to baseline median, separately per clock",
            "effective_flops_formula": "GLU:6*S*K*H*I; MLP:4*S*K*H*I; excludes epilogue and scales",
            "relative_baseline_partition": "same expert_type/dtype/shape/routing only; no GLU/MLP mixing",
            "effective_tflops": "effective_flops / (median_us * 1e6); null for diagnostic",
            "native_stage_timing": "not captured; run diagnostic separately and collect stdout externally",
            "cache_flush": False, "cuda_graphs": False, "backward": False,
            "correctness_check": False,
        },
        "results": results,
    }
    if args.json:
        output = Path(args.json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also closes the race after the early existence check.
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"JSON: {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+",
                        choices=["mixed_gemm", "fuse_moe", "swiglu"],
                        default=["mixed_gemm", "fuse_moe", "swiglu"])
    parser.add_argument("--suite", choices=["regular", "ragged", "all"],
                        default="regular")
    parser.add_argument("--backend", choices=["compare", "cuda"], default="compare",
                        help="compare preserves CUDA vs TileLang; cuda is standalone fuse_moe")
    parser.add_argument("--ops-library", help="precompiled fuse_moe .so; required for cuda; no JIT")
    parser.add_argument("--ops-metadata", help="library metadata; default: <ops-library>.json")
    parser.add_argument("--expert-type", choices=["glu", "mlp"], default="glu",
                        help="standalone CUDA only; explicit GLU paired or MLP single projection")
    parser.add_argument("--expert-activation", choices=["silu", "gelu"], default="silu")
    parser.add_argument("--numerical-mode", choices=["fused", "staged"], default="fused")
    parser.add_argument("--moe-mode-matrix", action="store_true",
                        help="explicit silu/fused, silu/staged, gelu/fused, gelu/staged")
    parser.add_argument("--moe-shapes", nargs="+", type=parse_moe_shape, metavar="S,H,I,E,K",
                        help="standalone CUDA shapes; overrides suite")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--routing", choices=["uniform", "skewed", "zipf"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", help="standalone CUDA JSON output path; must not already exist")
    parser.add_argument("--repeats", type=int, default=5,
                        help="positive CUDA-only repeat count (default: 5); compare ignores it")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false",
                        default=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("require --warmup >= 0 and --iters > 0")
    if args.repeats <= 0:
        parser.error("require --repeats > 0")
    if args.backend == "cuda":
        if args.ops != ["fuse_moe"]:
            parser.error("standalone CUDA requires --ops fuse_moe")
        if not args.ops_library:
            parser.error("standalone CUDA requires --ops-library (JIT is forbidden)")
        try:
            reject_existing_json(args.json)
        except FileExistsError as error:
            parser.error(str(error))
    elif (args.ops_library or args.ops_metadata or args.moe_mode_matrix
          or args.moe_shapes or args.json or args.expert_activation != "silu"
          or args.expert_type != "glu"
          or args.numerical_mode != "fused" or args.dtype != "bf16"
          or args.routing != "uniform" or args.seed != 42):
        parser.error("standalone MoE options require --backend cuda --ops fuse_moe")

    if not torch.cuda.is_available():
        print("CUDA 不可用。")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"cap: {torch.cuda.get_device_capability(0)}  "
          f"PyTorch: {torch.__version__}  suite: {args.suite}")

    if args.backend == "cuda":
        bench_fuse_moe_cuda(args)
        return

    run = args.suite in ("regular", "all")
    ragged = args.suite in ("ragged", "all")
    if "mixed_gemm" in args.ops:
        if run:
            bench_mixed_gemm(args)
        if ragged:
            bench_mixed_gemm(args, MIXED_GEMM_RAGGED_SHAPES)
    if "fuse_moe" in args.ops:
        if run:
            bench_fuse_moe(args)
        if ragged:
            bench_fuse_moe(args, MOE_RAGGED_CASES)
    if "swiglu" in args.ops:
        if run:
            bench_swiglu(args)
        if ragged:
            bench_swiglu(args, SWIGLU_RAGGED_SHAPES)


if __name__ == "__main__":
    main()
