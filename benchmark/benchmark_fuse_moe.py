"""
benchmark_fuse_moe.py — fuse_moe 算子性能基准

基线为贴近真实模型的 nn.Module 实现（Mixtral/HF 风格：per-expert nn.Linear
gate_up_proj/down_proj + silu*mul + index_add 路由归约），另测 torch.compile
优化后的性能，与自研 kernel 三方对比。

用法：
    python benchmark/benchmark_fuse_moe.py
    python benchmark/benchmark_fuse_moe.py --iters 50 --warmup 10
    python benchmark/benchmark_fuse_moe.py --no-compile      # 跳过 compile 对照
    python benchmark/benchmark_fuse_moe.py --csv result.csv

输出：每组 (S, H, I, E, K) 的 custom / eager / compiled 耗时、TFLOPS 与加速比。
FLOPS 口径：S*K*6*H*I（gate_up 2·H·2I + down 2·I·H，乘加各计一次）。

注：kernel 中间结果以 bf16 落盘（与真实 bf16 模型一致的数值路径），
与 fp32 参考存在 bf16 精度级差异，校验容差已覆盖。
"""
import argparse
import gc
import hashlib
import json
import math
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import os
import sys
import time

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from custom_ops import ops  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 基线模型（Mixtral/HF 风格 nn.Module）
# ══════════════════════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    """Standard eager FFN: paired GLU or single-projection ordinary MLP.

    One Linear produces 2I (gate/up) for GLU or I for MLP. Low-precision
    Linear and pointwise ops round at their own output boundaries; the
    activation uses PyTorch FP32 opmath without extra conversion kernels.
    """

    def __init__(self, hidden: int, intermediate: int, expert_type="glu",
                 expert_activation="silu", staged=False, device=None, dtype=None):
        super().__init__()
        self.expert_type = expert_type
        self.expert_activation = expert_activation
        self.gate_up_proj = nn.Linear(
            hidden, (2 if expert_type == "glu" else 1) * intermediate,
            bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False,
                                   device=device, dtype=dtype)

    def forward(self, x):
        projection = self.gate_up_proj(x)
        if self.expert_type == "glu":
            gate, up = projection.chunk(2, dim=-1)
        else:
            gate = projection
        # Native PyTorch half/bf16 pointwise ops use FP32 opmath, then round
        # their result to the tensor dtype. No redundant float()/to() kernels:
        # this is the ordinary eager FFN, not an artificially slowed baseline.
        activated = (F.silu(gate) if self.expert_activation == "silu"
                     else F.gelu(gate, approximate="none"))
        return self.down_proj(activated * up if self.expert_type == "glu" else activated)


class MoEFFN(nn.Module):
    """Mixtral/HF 风格 MoE FFN 前向（真实模型写法）。

    路由语义与自研 kernel 一致：topk 展平后位置 i 对应 token i//K；遍历
    expert，nonzero 选出路由到该 expert 的 token，计算后按路由权重
    index_add 加权归约（fp32 归约、bf16 输出）。"""

    def __init__(self, hidden: int, intermediate: int, num_experts: int,
                 expert_type="glu", expert_activation="silu", staged=False,
                 device=None, dtype=None):
        super().__init__()
        self.staged = staged
        self.experts = nn.ModuleList(
            ExpertFFN(hidden, intermediate, expert_type, expert_activation,
                      staged, device, dtype) for _ in range(num_experts))

    def forward(self, x, topk_ids, topk_scale):
        num_seq, num_topk = topk_ids.shape
        out = torch.zeros(x.shape[0], x.shape[1],
                          device=x.device, dtype=torch.float32)
        flat_ids = topk_ids.reshape(-1)
        flat_scale = topk_scale.reshape(-1)
        for e, expert in enumerate(self.experts):
            idx = (flat_ids == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            rows = torch.div(idx, num_topk, rounding_mode="floor")
            y = expert(x[rows])                                  # (n, H) bf16
            weighted = y.float() * flat_scale[idx][:, None]
            if self.staged:
                weighted = weighted.to(x.dtype).float()
            out.index_add_(0, rows, weighted)
        return out.to(x.dtype)


class MoEFFNDense(nn.Module):
    """Graph-friendly MoE 前向：chunked gather + bmm（无数据依赖控制流）。

    为 fullgraph=True + reduce-overhead（CUDA graph）专门编写的等价形式：
    按固定 chunk 展开循环（静态形状可完整捕获），每 chunk gather 权重做
    bmm 后 index_add 归约。FLOPS 与真实 MoE 相同；代价是权重按 token
    复制（内存放大），chunk 大小按权重 tile 字节数自适应（目标 ~2GB 峰值）。"""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, max_chunks: int = 96):
        super().__init__()
        E, N2, H = w1.shape
        I = w2.shape[2]
        self.w1 = nn.Parameter(w1.clone())                      # (E, 2I, H)
        self.w2 = nn.Parameter(w2.clone())                      # (E, H, I)
        # chunk 自适应：单 chunk 权重 gather ≤ ~2GB（对 H=4096 等大权重自动缩小）
        mem_c = max(32, int(2e9 / (N2 * H * w1.element_size())))
        self.chunk = max(32, min(mem_c, 8192))
        self._num_experts = E
        self._intermediate = I

    def forward(self, x, topk_ids, topk_scale):
        S, H = x.shape
        K = topk_ids.shape[1]
        T = S * K
        C = self.chunk
        e = topk_ids.reshape(-1)
        sc = topk_scale.reshape(-1)
        xg = x[:, None, :].expand(S, K, H).reshape(T, H)
        rows_all = torch.div(torch.arange(T, device=x.device), K,
                             rounding_mode="floor")
        out = torch.zeros(S, H, device=x.device, dtype=torch.float32)
        for i in range(0, T, C):  # 静态展开（T/C ≤ ~96 个 chunk）
            j = min(i + C, T)
            ec = e[i:j]
            w1c = self.w1[ec]                                   # (c, 2I, H) gather
            gu = torch.bmm(xg[i:j].unsqueeze(1), w1c.transpose(1, 2)).squeeze(1)
            g, u = gu.chunk(2, dim=-1)
            h = F.silu(g) * u
            w2c = self.w2[ec]                                   # (c, H, I) gather
            y = torch.bmm(h.unsqueeze(1), w2c.transpose(1, 2)).squeeze(1)
            out.index_add_(0, rows_all[i:j], y.float() * sc[i:j][:, None])
        return out.to(x.dtype)


def make_module(w1, w2):
    """按 benchmark 权重张量构建 MoEFFN（拷入权重保证数值一致）。"""
    E, N2, H = w1.shape
    I = w2.shape[2]
    m = MoEFFN(H, I, E).to(device=w1.device, dtype=w1.dtype)
    with torch.no_grad():
        for e, ex in enumerate(m.experts):
            ex.gate_up_proj.weight.copy_(w1[e])
            ex.down_proj.weight.copy_(w2[e])
    return m


def make_dense_module(w1, w2):
    """Graph-friendly 变体（构造时拷入权重）。"""
    return MoEFFNDense(w1, w2).to(device=w1.device, dtype=w1.dtype)


def fp32_ref(x, w1, w2, topk_ids, topk_scale):
    """per-expert 分组 fp32 参考（校验金标准）"""
    S, H = x.shape
    E, _, _ = w1.shape
    I = w2.shape[2]
    K = topk_ids.shape[1]
    out = torch.zeros(S, H, device=x.device, dtype=torch.float32)
    e = topk_ids.reshape(-1)
    xg = x[:, None, :].expand(S, K, H).reshape(S * K, H)
    for ei in range(E):
        idx = (e == ei).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xe = xg[idx].float()
        gu = xe @ w1[ei].float().T
        g, u = gu[:, :I], gu[:, I:]
        h = F.silu(g) * u
        y = h @ w2[ei].float().T
        rows = torch.div(idx, K, rounding_mode="floor")
        out.index_add_(0, rows, y * topk_scale.reshape(-1)[idx][:, None])
    return out


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # µs


def compile_available() -> bool:
    """进程级 torch.compile 可用性探测（仅探测一次；失败永久记忆）。

    环境兼容性：旧 Python（<3.10）+ 新 torch 的组合下 inductor 会因
    `dataclass(slots=True)`（3.10+ 语法）报 TypeError；旧 torch alpha 版亦有
    类似不兼容。probe 必须用真实计算图（matmul + 非逐元素归约）才能触发
    inductor 完整代码生成路径——简单 lambda 可能走 eager fallback 探不出来。
    任何一次真实 compile 失败也会置 False（失败记忆，后续 shape 静默跳过
    而不是逐个报错刷屏）。"""
    global _COMPILE_AVAILABLE
    if _COMPILE_AVAILABLE is not None:
        return _COMPILE_AVAILABLE
    _COMPILE_AVAILABLE = False
    try:
        t = torch.randn(16, 16, device="cuda")
        probe = torch.compile(lambda a: (a @ a.T).relu().sum())
        _ = probe(t)
        torch.cuda.synchronize()
        _COMPILE_AVAILABLE = True
    except Exception as ex:
        print(f"  (torch.compile unavailable on this env: {type(ex).__name__}; "
              f"compile/RO/fullgraph 对照列将显示 N/A)")
    return _COMPILE_AVAILABLE


def _compile_failed_once(ex: Exception) -> None:
    """真实 compile 失败后永久禁用（环境性失败，重试只会重复报错）。"""
    global _COMPILE_AVAILABLE
    if _COMPILE_AVAILABLE:
        _COMPILE_AVAILABLE = False
        print(f"  (torch.compile disabled after failure: {type(ex).__name__}; "
              f"remaining shapes will skip compile columns)")


_COMPILE_AVAILABLE = None


@contextmanager
def strict_math():
    """Restore process settings even when correctness or budget validation fails."""
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    names = ("allow_tf32", "allow_bf16_reduced_precision_reduction",
             "allow_fp16_reduced_precision_reduction")
    previous = {name: getattr(matmul, name) for name in names if hasattr(matmul, name)}
    old_cudnn = cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        for name in previous:
            setattr(matmul, name, False)
        cudnn.allow_tf32 = False
        with torch.no_grad():
            yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        for name, value in previous.items():
            setattr(matmul, name, value)
        cudnn.allow_tf32 = old_cudnn


def storage_shape(value):
    try:
        shape = tuple(int(v) for v in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected S,H,I,E,K integers") from error
    if len(shape) != 5 or any(v <= 0 for v in shape):
        raise argparse.ArgumentTypeError("expected five positive dimensions")
    _, h, i, e, k = shape
    if h % 64 or i % 64 or e > 512 or k > 128 or k > e:
        raise argparse.ArgumentTypeError("require H/I %64=0, E<=512, K<=128 and K<=E")
    return shape


def storage_inputs(shape, dtype, seed, routing, expert_type):
    """Canonical pageable CPU quantized data, generated once, outside timing."""
    s, h, i, e, k = shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(s, h, generator=generator).to(dtype)
    w1 = (torch.randn(e, (2 if expert_type == "glu" else 1) * i, h,
                      generator=generator) / math.sqrt(h)).to(dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    w2 = (torch.randn(e, h, i, generator=generator) / math.sqrt(i)).to(dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    eligible = 2 if routing == "hotset" else e
    # Seeded sampling without replacement; every token has K distinct experts.
    ids = torch.rand(s, eligible, generator=generator).argsort(dim=1)[:, :k].int().contiguous()
    scale = torch.rand(s, k, generator=generator) + 0.125
    if k > 1:
        scale /= scale.sum(dim=1, keepdim=True)
    return x, w1, w2, ids, scale


@torch.no_grad()
def stagewise_cpu_ref(x, w1, w2, ids, scale, expert_type="glu",
                      expert_activation="silu", numerical_mode="staged", sample_tokens=16):
    """Independent CPU FP32 matmuls with explicit storage dtype rounding.

    Only first sample_tokens tokens are computed; weights and inputs were already
    quantized. No nn.Module, CUDA, native op, or eager baseline is used here.
    Fused uses FP32 projection/activation product then dtype intermediate/down;
    staged additionally rounds projection, activation and weighted product.
    """
    if any(t.device.type != "cpu" for t in (x, w1, w2, ids, scale)):
        raise ValueError("CPU reference requires canonical CPU tensors")
    n = min(sample_tokens, x.shape[0])
    dtype = x.dtype
    out = torch.zeros(n, x.shape[1], dtype=torch.float32)
    flat_ids, flat_scale = ids[:n].reshape(-1), scale[:n].reshape(-1)
    k, intermediate = ids.shape[1], w2.shape[2]
    for expert in flat_ids.unique().tolist():
        positions = (flat_ids == expert).nonzero(as_tuple=True)[0]
        rows = positions // k
        projection = x[:n][rows].float() @ w1[expert].float().T
        if numerical_mode == "staged":
            projection = projection.to(dtype).float()
        gate = projection[:, :intermediate]
        # Explicit exact-erf GELU, independent of eager's F.gelu implementation.
        activated = (gate * torch.sigmoid(gate) if expert_activation == "silu"
                     else 0.5 * gate * (1 + torch.erf(gate / math.sqrt(2))))
        if numerical_mode == "staged":
            activated = activated.to(dtype).float()
        if expert_type == "glu":
            activated = activated * projection[:, intermediate:]
        intermediate_value = activated.to(dtype).float()
        down = (intermediate_value @ w2[expert].float().T).to(dtype).float()
        weighted = down * flat_scale[positions, None]
        if numerical_mode == "staged":
            weighted = weighted.to(dtype).float()
        out.index_add_(0, rows, weighted)
    return out.to(dtype)


def storage_module(w1, w2, expert_type, activation):
    e, _, h = w1.shape
    # Construct meta Linear modules: no random allocation/copy of full GPU weights.
    module = MoEFFN(h, w2.shape[2], e, expert_type, activation,
                    staged=True, device="meta", dtype=w1.dtype)
    for expert, ff in enumerate(module.experts):
        ff.gate_up_proj.weight = nn.Parameter(w1[expert], requires_grad=False)
        ff.down_proj.weight = nn.Parameter(w2[expert], requires_grad=False)
    return module.eval()


def check_storage_output(output, shape, dtype, reference, atol, rtol):
    if tuple(output.shape) != tuple(shape) or output.dtype != dtype:
        raise AssertionError("output full shape/dtype contract failed")
    if not torch.isfinite(output).all().item():
        raise AssertionError("output contains nonfinite values")
    actual = output[:reference.shape[0]].detach().cpu().float()
    expected = reference.float()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    difference = (actual - expected).abs()
    return {"passed": True, "full_shape_dtype_finite": True,
            "reference": "independent_cpu_fp32_stagewise",
            "sample_tokens": reference.shape[0], "atol": atol, "rtol": rtol,
            "max_abs": difference.max().item(), "mean_abs": difference.mean().item()}


def summarize_ms(samples):
    ordered = sorted(samples)
    def percentile(q):
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {"samples_ms": samples, "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95), "median_ms": median(samples)}


def resident_batch(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize()  # event initialization excluded
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    gpu_us = start.elapsed_time(end) * 1000 / iters
    host_samples = []
    for _ in range(iters):
        begin = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        host_samples.append((time.perf_counter() - begin) * 1e6)
        del output
    host_us = sum(host_samples) / iters
    return {"gpu_event_us": gpu_us, "host_wall_us": host_us,
            "host_samples_us": host_samples, "gpu_event_batch_us": gpu_us * iters,
            "host_wall_batch_us": sum(host_samples), "iters": iters}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_cpu_weight_bytes(weights):
    storages = {w.untyped_storage().data_ptr(): w.untyped_storage().nbytes() for w in weights}
    return sum(storages.values())


def validate_offload_stats(stats, args, active, expert_bytes, resident_bytes):
    required = ("active_experts", "waves", "weight_h2d_bytes", "gpu_charged_bytes",
                "ram_charged_bytes", "pinned_bytes", "resident_cache_bytes", "num_banks", "wave_capacity")
    if any(key not in stats for key in required):
        raise RuntimeError("offload stats schema lacks required capacity evidence")
    expected = 0 if args.offload_cache == "static_hot" else active * expert_bytes
    if stats["weight_h2d_bytes"] != expected or stats["active_experts"] != active:
        raise AssertionError("offload H2D/active expert contract failed")
    capacity = stats.get("wave_capacity", 0)
    if stats["waves"] <= 0 or (args.offload_cache == "disabled" and
                               (capacity <= 0 or stats["waves"] != math.ceil(active / capacity))):
        raise AssertionError("nonempty routing wave/capacity contract failed")
    if stats.get("dynamic_cache_bytes", 0) or stats.get("dynamic_experts", 0):
        raise AssertionError("dynamic cache must remain disabled")
    if stats["resident_cache_bytes"] != resident_bytes:
        raise AssertionError("static resident allocation differs from explicit cache cap")
    for field, limit in (("gpu_charged_bytes", args.gpu_budget_mib),
                         ("ram_charged_bytes", args.ram_budget_mib),
                         ("pinned_bytes", args.pinned_budget_mib)):
        if stats[field] > limit * 1024**2:
            raise AssertionError(f"{field} exceeded explicit budget")


def measure_offload(cpu, gpu_inputs, reference, args, expert_type):
    """Only public runtime API. Sequential contexts; no CUDA events or TFLOPS."""
    from custom_ops.moe_offload import ExpertStore, PrefillMoERuntime
    x, w1, w2, ids, scale = cpu
    xg, idg, scg = gpu_inputs
    store = ExpertStore({0: (w1, w2)}, ram_budget_bytes=args.ram_budget_mib * 1024**2)
    expert_bytes = (w1[0].numel() + w2[0].numel()) * w1.element_size()
    # Native aligns each resident allocation to 512 bytes.
    resident_bytes = 2 * ((expert_bytes + 511) // 512 * 512) if args.offload_cache == "static_hot" else 0
    options = dict(device=xg.device, gpu_budget_bytes=args.gpu_budget_mib * 1024**2,
                   ram_budget_bytes=args.ram_budget_mib * 1024**2,
                   pinned_budget_bytes=args.pinned_budget_mib * 1024**2,
                   activation_mode="auto", max_route_tile_rows="auto",
                   resident_cache_bytes=resident_bytes,
                   resident_experts={0: [0, 1]} if resident_bytes else {},
                   dynamic_cache_bytes=0, host_profile=False,
                   expert_type=expert_type, expert_activation=args.expert_activation,
                   numerical_mode=args.numerical_mode)
    if resident_bytes >= options["gpu_budget_bytes"]:
        raise ValueError("static_hot leaves no GPU budget for workspace/live output")
    modes = args.offload_staging
    records = {mode: {"raw_repeats": []} for mode in modes}
    orders = []
    active = ids.unique().numel()
    for repeat in range(args.repeats):
        offset = repeat % len(modes)
        order = modes[offset:] + modes[:offset]
        orders.append(order)
        for position, mode in enumerate(order):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            start = time.perf_counter()
            runtime = PrefillMoERuntime(store, async_staging=(mode == "async"), **options)
            torch.cuda.synchronize()
            init_ms = (time.perf_counter() - start) * 1000
            try:
                # Fresh context: first forward, no validation invocation/hidden warmup.
                start = time.perf_counter()
                output = runtime.forward(xg, idg, scg)
                torch.cuda.synchronize()
                cold_ms = (time.perf_counter() - start) * 1000
                cold_check = check_storage_output(output, x.shape, x.dtype, reference,
                                                   args.atol, args.rtol)
                cold_stats = runtime.stats()
                validate_offload_stats(cold_stats, args, active, expert_bytes, resident_bytes)
                del output
                warmup_stats = []
                for _ in range(args.warmup):
                    runtime.forward(xg, idg, scg)
                    stats = runtime.stats()
                    validate_offload_stats(stats, args, active, expert_bytes, resident_bytes)
                    warmup_stats.append(stats)
                torch.cuda.synchronize()
                # Manual per-call stats are outside clocks, retained for every iteration.
                samples, statistics = [], []
                for _ in range(args.iters):
                    start = time.perf_counter()
                    output = runtime.forward(xg, idg, scg)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                    stats = runtime.stats()
                    validate_offload_stats(stats, args, active, expert_bytes, resident_bytes)
                    statistics.append(stats)
                    if len(samples) == args.iters:
                        warm_check = check_storage_output(output, x.shape, x.dtype, reference,
                                                          args.atol, args.rtol)
                    del output
                highwater = {key: max(stat[key] for stat in [cold_stats, *warmup_stats, *statistics])
                             for key in ("gpu_charged_bytes", "ram_charged_bytes", "pinned_bytes")}
                records[mode]["raw_repeats"].append({
                    "repeat": repeat + 1, "order_index": position,
                    "initialization_ms": init_ms, "cold_first_forward_ms": cold_ms,
                    "cold_stats": cold_stats, "correctness": cold_check,
                    "global_gpu_baseline_allocated_bytes": baseline,
                    "warm": {**summarize_ms(samples), "batch_mean_ms": sum(samples) / args.iters},
                    "per_call_stats": statistics, "warmup_stats": warmup_stats,
                    "warm_correctness": warm_check, "context_highwater_bytes": highwater,
                })
            finally:
                runtime.close()
                del runtime
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
    for mode, record in records.items():
        raw = record["raw_repeats"]
        record.update({"storage": "pageable_cpu_weights_bounded_gpu_context",
                       "staging": mode, "clock": "host_wall_forward_plus_synchronize",
                       "warm_median_batch_mean_ms": median(r["warm"]["batch_mean_ms"] for r in raw),
                       "warm_all_calls": summarize_ms([v for r in raw for v in r["warm"]["samples_ms"]]),
                       "warm_total_weight_h2d_bytes": sum(s["weight_h2d_bytes"] for r in raw for s in r["per_call_stats"]),
                       "warm_total_waves": sum(s["waves"] for r in raw for s in r["per_call_stats"]),
                       "actual_num_banks": sorted({s["num_banks"] for r in raw for s in r["per_call_stats"]}),
                       "actual_wave_capacity": sorted({s["wave_capacity"] for r in raw for s in r["per_call_stats"]}),
                       "bank_budget_fallback_observed": any(s["num_banks"] == 1 for r in raw for s in r["per_call_stats"]),
                       "cold": summarize_ms([r["cold_first_forward_ms"] for r in raw]),
                       "initialization": summarize_ms([r["initialization_ms"] for r in raw]),
                       "repeat_orders": orders, "resident_cache_cap_bytes": resident_bytes,
                       "resident_experts": [0, 1] if resident_bytes else [],
                       "dynamic_cache_cap_bytes": 0,
                       "whole_weights_resident": bool(resident_bytes and w1.shape[0] == 2),
                       "weight_bytes_exceed_gpu_budget": store.weight_bytes > options["gpu_budget_bytes"],
                       "cpu_unique_weight_bytes": store.weight_bytes,
                       "gpu_event_us": None, "hardware_tflops": None})
    return records


@torch.no_grad()
def bench_storage(args):
    if args.json and os.path.lexists(Path(args.json).expanduser()):
        raise FileExistsError(f"refusing to overwrite JSON: {args.json}")
    diagnostic = {key: value for key, value in os.environ.items()
                  if key.startswith("FUSE_MOE") or key in ("CUDA_LAUNCH_BLOCKING", "TORCH_USE_CUDA_DSA")}
    if diagnostic:
        raise RuntimeError(f"performance suite rejects diagnostic/tuning environment: {diagnostic}")
    ops.disable_jit()
    library = Path(args.ops_library).expanduser().resolve(strict=True)
    metadata = (Path(args.ops_metadata).expanduser().resolve(strict=True)
                if args.ops_metadata else Path(str(library) + ".json").resolve(strict=True))
    ops.load_precompiled("fuse_moe", library, metadata)
    with metadata.open(encoding="utf-8") as handle:
        sidecar = json.load(handle)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    report = {"schema_version": 1, "benchmark": "fuse_moe_storage_structure",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "argv": sys.argv, "arguments": vars(args), "performance_eligible": True,
              "artifacts": {"library": str(library), "library_sha256": sha256_file(library),
                            "metadata": str(metadata), "metadata_sha256": sha256_file(metadata),
                            "metadata_content": sidecar,
                            "benchmark_source": str(Path(__file__).resolve()),
                            "benchmark_source_sha256": sha256_file(Path(__file__).resolve())},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                              "compute_capability": list(torch.cuda.get_device_capability()),
                              "cpu_threads": torch.get_num_threads(),
                              "variables": {k: os.environ.get(k) for k in
                                            ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                              "clock_power": "not collected; caller must record externally"},
              "contracts": {"w1": "GLU(E,2I,H); MLP(E,I,H)", "w2": "(E,H,I)",
                            "ids": "int32, distinct per token", "scale": "FP32 positive nonconstant; normalized per token for K>1, raw for K=1",
                            "initialization": "x~N(0,1); w1~N(0,H^-1/2); w2~N(0,I^-1/2); quantized before all backends",
                            "activation": "SiLU or exact-erf GELU", "tf32": False,
                            "reduced_precision_reduction": False, "allow_jit": False,
                            "backward": False, "masked_routes": False,
                            "staged": "round projection, act, GLU product, down, weighted product to input dtype; FP32 index_add; dtype output",
                            "eager": "nn.Module nonzero routing + per-expert Linear; native dtype GELU/SiLU/mul with FP32 opmath, no redundant projection/activation casts; weighted product explicitly rounded; always staged; single first projection for MLP"},
              "measurement": {"resident": "GPU event asynchronous batch; separate host-wall per-call forward+synchronize samples; rotated order; median of repeat batch means",
                              "offload": "per-call host wall forward+synchronize; stats outside clock; sequential fresh contexts per repeat; cold no hidden warmup",
                              "initialization": "separate context construction including static cache upload; not cold forward",
                              "context_gpu_peak": "native charge includes workspace/cache/live output; excludes unrelated GPU tensors/allocator reserve/driver",
                              "ram_budget": "canonical unique weights plus native context charge; not process generation/reference peak RSS",
                              "excluded": ["generation", "CPU reference", "library load", "JIT", "compile", "stats", "correctness"],
                              "docnote": "offload and GPU-resident serve different storage constraints; only host-clock slowdown ratio, never offload speedup or hardware TFLOPS"},
              "results": []}
    with strict_math():
        for case_index, shape in enumerate(args.shapes):
            for expert_type in args.expert_types:
                cpu = storage_inputs(shape, dtype, args.seed + case_index, args.routing, expert_type)
                xc, w1c, w2c, idc, scc = cpu
                weight_bytes = unique_cpu_weight_bytes((w1c, w2c))
                if args.offload and weight_bytes > args.ram_budget_mib * 1024**2:
                    raise ValueError("canonical CPU weight bytes exceed RAM budget")
                reference_start = time.perf_counter()
                reference = stagewise_cpu_ref(*cpu, expert_type, args.expert_activation,
                                              args.numerical_mode, args.reference_tokens)
                eager_ref = (reference if args.numerical_mode == "staged" else
                             stagewise_cpu_ref(*cpu, expert_type, args.expert_activation,
                                               "staged", args.reference_tokens))
                reference_wall_ms = (time.perf_counter() - reference_start) * 1000
                x, ids, scale = xc.cuda(), idc.cuda(), scc.cuda()
                w1, w2 = w1c.cuda(), w2c.cuda()
                module = storage_module(w1, w2, expert_type, args.expert_activation)
                def custom():
                    return ops.fuse_moe(x, w1, w2, ids, scale, backend="cuda",
                                        expert_type=expert_type, expert_activation=args.expert_activation,
                                        numerical_mode=args.numerical_mode)
                def eager():
                    return module(x, ids, scale)
                checks = {"custom": check_storage_output(custom(), xc.shape, dtype, reference, args.atol, args.rtol),
                          "eager": check_storage_output(eager(), xc.shape, dtype, eager_ref, args.atol, args.rtol)}
                raw = {"custom": [], "eager": []}
                orders = []
                for repeat in range(args.repeats):
                    order = ["custom", "eager"] if repeat % 2 == 0 else ["eager", "custom"]
                    orders.append(order)
                    for position, name in enumerate(order):
                        timing = resident_batch(custom if name == "custom" else eager, args.warmup, args.iters)
                        raw[name].append({"repeat": repeat + 1, "order_index": position, **timing})
                flops = (6 if expert_type == "glu" else 4) * shape[0] * shape[4] * shape[1] * shape[2]
                resident = {name: {"raw_samples": samples,
                                   "gpu_event_us": median(s["gpu_event_us"] for s in samples),
                                   "host_wall_us": median(s["host_wall_us"] for s in samples),
                                   "correctness": checks[name],
                                   "numerical_mode": args.numerical_mode if name == "custom" else "staged"}
                            for name, samples in raw.items()}
                for values in resident.values():
                    values["effective_gemm_tflops"] = flops / (values["gpu_event_us"] * 1e6)
                row = {"shape": dict(zip(("S", "H", "I", "E", "K"), shape)),
                       "expert_type": expert_type, "expert_type_native_code": 0 if expert_type == "glu" else 1,
                       "mode_key": f"{expert_type}/{args.expert_activation}/{args.numerical_mode}",
                       "expert_activation": args.expert_activation,
                       "numerical_mode": args.numerical_mode, "dtype": args.dtype,
                       "seed": args.seed + case_index, "routing": args.routing,
                       "active_expert_ids": idc.unique().tolist(),
                       "cpu_unique_weight_bytes": weight_bytes,
                       "cpu_reference_metrics": {"wall_ms": reference_wall_ms,
                                                 "sample_tokens": reference.shape[0],
                                                 "includes_eager_staged_reference": args.numerical_mode != "staged",
                                                 "excluded_from_performance": True},
                       "effective_flops": flops,
                       "effective_flops_formula": f"{6 if expert_type == 'glu' else 4}*S*K*H*I",
                       "resident": resident, "resident_repeat_orders": orders,
                       "eager_same_numerical_contract": args.numerical_mode == "staged",
                       "offload": {}}
                # Release ALL baseline GPU weights/modules before claiming capacity.
                del module, w1, w2, custom, eager
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if args.offload:
                    row["offload"] = measure_offload(cpu, (x, ids, scale), reference, args, expert_type)
                    for values in row["offload"].values():
                        values["host_slowdown_vs_resident_custom"] = values["warm_median_batch_mean_ms"] * 1000 / resident["custom"]["host_wall_us"]
                report["results"].append(row)
                print(json.dumps({"shape": shape, "expert_type": expert_type,
                                  "resident": {name: {k: v[k] for k in ("gpu_event_us", "host_wall_us")} for name, v in resident.items()},
                                  "offload_warm_host_ms": {name: v["warm_median_batch_mean_ms"] for name, v in row["offload"].items()}}))
                del cpu, xc, w1c, w2c, idc, scc, x, ids, scale, reference, eager_ref
                gc.collect()
                torch.cuda.empty_cache()
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.json:
        with Path(args.json).expanduser().open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return report


@torch.no_grad()
@strict_math()
def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--suite", choices=("legacy", "storage"), default="legacy")
    parser.add_argument("--ops-library")
    parser.add_argument("--ops-metadata")
    parser.add_argument("--expert-types", nargs="+", choices=("glu", "mlp"), default=["glu"])
    parser.add_argument("--expert-activation", choices=("silu", "gelu"), default="silu")
    parser.add_argument("--numerical-mode", choices=("fused", "staged"), default="staged")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--shapes", nargs="+", type=storage_shape, metavar="S,H,I,E,K")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--gpu-budget-mib", type=int, default=128,
                        help="Bounded offload context budget; actual bank/capacity fallback is recorded")
    parser.add_argument("--ram-budget-mib", type=int, default=2048)
    parser.add_argument("--pinned-budget-mib", type=int, default=64)
    parser.add_argument("--offload-cache", choices=("disabled", "static_hot"), default="disabled")
    parser.add_argument("--offload-staging", nargs="+", choices=("sync", "async"), default=["sync", "async"])
    parser.add_argument("--routing", choices=("uniform", "hotset"), default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-tokens", type=int, default=16)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", help="exclusive creation; parent directory must exist")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--fg-max-tokens", type=int, default=512,
                        help="fullgraph+cudagraph 变体仅在小 T（decode 类）运行："
                             "dense gather 形式的权重流量 O(T·E·H·I)，大 T 无意义且 OOM")
    args = parser.parse_args()
    if args.warmup < 0 or min(args.iters, args.repeats, args.reference_tokens,
                              args.gpu_budget_mib, args.ram_budget_mib,
                              args.pinned_budget_mib) <= 0:
        parser.error("iters/repeats/reference-tokens/budgets must be positive; warmup >= 0")
    if args.seed < 0 or any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("seed >= 0 and finite nonnegative correctness tolerances required")
    if len(set(args.expert_types)) != len(args.expert_types) or len(set(args.offload_staging)) != len(args.offload_staging):
        parser.error("duplicate expert types/staging modes are not allowed")
    if args.suite == "storage":
        if not args.ops_library or args.csv:
            parser.error("storage requires --ops-library; use --json, not --csv")
        if args.no_compile or args.fg_max_tokens != 512:
            parser.error("compile/fullgraph options only apply to legacy")
        if not args.offload and any(arg.split("=", 1)[0] in
                                   ("--offload-cache", "--offload-staging", "--gpu-budget-mib",
                                    "--ram-budget-mib", "--pinned-budget-mib") for arg in sys.argv[1:]):
            parser.error("offload budget/cache/staging options require --offload")
        if args.offload_cache == "static_hot" and args.routing != "hotset":
            parser.error("static_hot requires --routing hotset (only expert IDs 0/1)")
        args.shapes = args.shapes or [(s, 2048, 1024, 64, 2) for s in (128, 1024, 4096)]
        if args.routing == "hotset" and any(e < 2 or k > 2 for _, _, _, e, k in args.shapes):
            parser.error("hotset requires E>=2, K<=2")
        if args.offload_cache == "static_hot" and any(e <= 2 for _, _, _, e, _ in args.shapes):
            parser.error("static_hot capacity evidence requires E>2, not whole weights resident")
        if args.json:
            output = Path(args.json).expanduser()
            if os.path.lexists(output) or not output.parent.is_dir():
                parser.error("JSON must be new and its parent directory must exist")
        if not torch.cuda.is_available():
            parser.error("storage benchmark requires CUDA; CPU reference can be imported separately")
        bench_storage(args)
        return
    storage_flags = ("--ops-library", "--ops-metadata", "--expert-types", "--expert-activation",
                     "--numerical-mode", "--dtype", "--shapes", "--offload", "--gpu-budget-mib",
                     "--ram-budget-mib", "--pinned-budget-mib", "--offload-cache", "--offload-staging",
                     "--routing", "--seed", "--reference-tokens", "--atol", "--rtol", "--repeats", "--json")
    if any(arg.split("=", 1)[0] in storage_flags for arg in sys.argv[1:]):
        parser.error("new structure/storage options require --suite storage")

    assert ops.is_available(), "extension load failed"
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    # (S, H, I, E, K)
    shapes = [
        (128,   2048, 1024,  8, 2),    # 小批量 decode
        (128,   2048, 1024, 64, 2),
        (1024,  2048, 1024,  8, 2),    # prefill
        (1024,  2048, 1024, 64, 2),
        (1024,  4096, 1024,  8, 2),    # 大 H
        (4096,  2048, 1024,  8, 2),    # 大批量
        (4096,  2048, 1024, 64, 8),
        (4096,  4096, 1408,  8, 2),    # 大模型 shape
        (16384, 2048, 1024,  8, 2),    # 超大批量
    ]

    print(f"{'shape (S,H,I,E,K)':<24} {'custom µs':>9} {'TFLOPS':>7} "
          f"{'eager µs':>9} {'vs eager':>8} {'comp µs':>9} {'comp+RO µs':>10} "
          f"{'FG+graph µs':>11} {'vs FG':>8}")
    rows = []
    for (S, H, I, E, K) in shapes:
        x = (torch.randn(S, H, device=dev, dtype=torch.float32) * 0.5).to(dtype)
        w1 = (torch.randn(E, 2 * I, H, device=dev, dtype=torch.float32) * 0.05).to(dtype)
        w2 = (torch.randn(E, H, I, device=dev, dtype=torch.float32) * 0.05).to(dtype)
        topk_ids = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
        topk_scale = torch.rand(S, K, device=dev, dtype=torch.float32)

        module = make_module(w1, w2)
        module.eval()

        # 自研 kernel
        t_custom = bench(lambda: ops.fuse_moe(x, w1, w2, topk_ids, topk_scale),
                         args.warmup, args.iters)

        # eager nn.Module 基线（真实模型写法）
        t_eager = bench(lambda: module(x, topk_ids, topk_scale),
                        args.warmup, args.iters)

        # torch.compile 默认模式（graph break 允许；数据依赖路由下 CUDA graph 自动禁用）
        t_comp = float("nan")
        compiled = None
        if not args.no_compile and compile_available():
            try:
                compiled = torch.compile(module)
                t_comp = bench(lambda: compiled(x, topk_ids, topk_scale),
                               args.warmup, args.iters)
            except Exception as ex:
                _compile_failed_once(ex)

        # torch.compile + mode="reduce-overhead"（请求 CUDA graph）——真实写法：
        # per-expert nonzero 数据依赖路由产生 graph break，torch 会自动禁用
        # cudagraph 捕获并退化为逐 graph 执行，验证「RO 对真实 MoE 无效」。
        t_ro = float("nan")
        if not args.no_compile and compile_available():
            try:
                compiled_ro = torch.compile(module, mode="reduce-overhead")
                t_ro = bench(lambda: compiled_ro(x, topk_ids, topk_scale),
                             args.warmup, args.iters)
                del compiled_ro
            except Exception as ex:
                _compile_failed_once(ex)

        # fullgraph=True + reduce-overhead（CUDA graph）——graph-friendly 写法的上限。
        # 真实写法（nonzero 数据依赖路由）无法 fullgraph，会直接报错，此处
        # 用等价的 chunked dense 形式承接；其权重 gather 流量 O(T·E·H·I)，
        # 仅在小 T（decode 类）下有参考意义，大 T 跳过（避免无意义测试 + OOM）。
        t_fg = float("nan")
        if not args.no_compile and S * K <= args.fg_max_tokens and compile_available():
            try:
                dense = make_dense_module(w1, w2)
                dense.eval()
                compiled_fg = torch.compile(dense, fullgraph=True,
                                             mode="reduce-overhead")
                t_fg = bench(lambda: compiled_fg(x, topk_ids, topk_scale),
                             args.warmup, args.iters)
                # cudagraph 输出会被后续 replay 覆写，正确性校验用 clone 快照
                fg_out = compiled_fg(x, topk_ids, topk_scale).clone()
                ref0 = fp32_ref(x, w1, w2, topk_ids, topk_scale)
                if not torch.allclose(fg_out.float(), ref0, atol=0.1, rtol=0.05):
                    print("  !! fullgraph dense variant mismatch")
                del dense, compiled_fg, fg_out
            except torch.cuda.OutOfMemoryError:
                print("  (fullgraph dense variant: OOM, N/A)")
            except Exception as ex:
                _compile_failed_once(ex)
            finally:
                # cudagraph 私有内存池不随 del 释放，强制回收防后续 shape OOM
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        flops = S * K * 6 * H * I
        tf = flops / (t_custom * 1e-6) / 1e12
        sp_eager = t_eager / t_custom
        sp_fg = t_fg / t_custom if t_fg == t_fg else float("nan")
        fg_str = f"{t_fg:.1f}" if t_fg == t_fg else "—"
        sp_fg_str = f"{sp_fg:.1f}x" if t_fg == t_fg else "—"
        print(f"({S},{H},{I},{E},{K}){'':<6} {t_custom:>9.1f} {tf:>7.1f} "
              f"{t_eager:>9.1f} {sp_eager:>7.2f}x {t_comp:>9.1f} {t_ro:>10.1f} "
              f"{fg_str:>11} {sp_fg_str:>8}")
        rows.append((S, H, I, E, K, t_custom, tf, t_eager, t_comp, t_ro, t_fg))

        # 正确性校验（vs fp32 参考；容差覆盖中间 bf16 落盘：kernel 的
        # gate_up/act/down 中间量以 bf16 写盘，ref 值小的元素相对误差天然偏
        # 大（实测 p99_rel~0.2），故 atol 主导 + rtol 充裕；报错时附误差分位
        # 数详情以便区分精度边界与结构性错误）
        ref = fp32_ref(x, w1, w2, topk_ids, topk_scale)
        out = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)
        diff = (out.float() - ref).abs()
        rel = diff / ref.abs().clamp_min(1e-2)
        ok = torch.allclose(out.float(), ref, atol=0.2, rtol=0.05)
        if not ok:
            p99 = rel.flatten().kthvalue(max(1, int(rel.numel() * 0.99))).values.item()
            print(f"  !! correctness FAILED: max_abs={diff.max().item():.4f} "
                  f"mean={diff.mean().item():.5f} p99_rel={p99:.4f} "
                  f"ref_absmax={ref.abs().max().item():.2f}")
        # 基线自身也抽样校验（防止再次写错基线）
        mout = module(x, topk_ids, topk_scale)
        if not torch.allclose(mout.float(), ref, atol=0.2, rtol=0.05):
            print("  !! baseline module mismatch (baseline bug?)")

        del module
        if compiled is not None:
            del compiled
        gc.collect()
        torch.cuda.empty_cache()

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("S,H,I,E,K,custom_us,custom_tflops,eager_us,compiled_us,"
                    "compiled_ro_us,fullgraph_us\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
        print(f"\ncsv written to {args.csv}")


if __name__ == "__main__":
    main()
