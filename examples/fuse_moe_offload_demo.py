"""CPU expert weights, bounded CUDA execution, cache stats and explicit recovery.

Development (may JIT):
    python examples/fuse_moe_offload_demo.py --jit
Strict deployment (.so and its .so.json must match):
    python examples/fuse_moe_offload_demo.py --ops-library /path/to/native.so
Other examples:
    python examples/fuse_moe_offload_demo.py --jit --layout separate --dtype bf16
    python examples/fuse_moe_offload_demo.py --jit --cache static --show-recovery
    python examples/fuse_moe_offload_demo.py --jit --cache lru --async-staging

Weights are immutable pageable CPU tensors. Never update bound weights in-place:
create a new ExpertStore/runtime instead. Budgets do not cover unrelated tensors,
allocator reserves or checkpoint conversion peak RSS. Async is opt-in, not a
speedup guarantee. This small numerical example is not a performance benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# The repository root is itself the custom_ops package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from custom_ops import ops  # noqa: E402
from custom_ops.moe_offload import (  # noqa: E402
    ExpertStore, LayerKeys, OffloadConfig, PrefillMoERuntime,
)

MIB = 1024**2
EXPERTS, HIDDEN, TOP_K = 8, 128, 2
SEED = 20261016


def load_native(library: Path | None, *, jit: bool = False):
    """Deployment always configures the package singleton, not another facade."""
    if library is not None:
        ops.load_precompiled("fuse_moe", library.resolve())
    elif jit:
        ops.ensure_loaded("fuse_moe")
        if not ops.is_available("fuse_moe"):
            raise RuntimeError(ops.load_error("fuse_moe"))
    else:
        raise ValueError("choose --ops-library or explicitly allow development --jit")


def make_store(*, layers=1, dtype=torch.float16, layout="fused", expert_type="glu"):
    """Explicit checkpoint mapping; adapt expert lists/transposes externally.

    A real checkpoint can use ExpertStore.load(path, layer_keys, dtype=dtype,
    ram_budget_bytes=64*MIB). FP32-to-FP16/BF16 conversion must be explicit.
    Here a synthetic state_dict keeps the example independent of business models.
    """
    if layers not in (1, 2) or layout not in ("fused", "separate"):
        raise ValueError("example supports one/two layers and fused/separate layout")
    if expert_type not in ("glu", "mlp") or (expert_type == "mlp" and layout != "fused"):
        raise ValueError("MLP requires canonical single-projection fused layout")
    generator = torch.Generator().manual_seed(SEED)
    state, keys = {}, {}
    for layer, intermediate in ((0, 128), (5, 192))[:layers]:
        prefix = f"layers.{layer}.experts"
        gate = torch.randn(EXPERTS, intermediate, HIDDEN, generator=generator) * .08
        up = torch.randn(EXPERTS, intermediate, HIDDEN, generator=generator) * .08
        state[f"{prefix}.down"] = torch.randn(
            EXPERTS, HIDDEN, intermediate, generator=generator) * .08
        if layout == "fused":
            state[f"{prefix}.gate_up"] = torch.cat((gate, up), dim=1) if expert_type == "glu" else gate
            keys[layer] = LayerKeys(down=f"{prefix}.down", gate_up=f"{prefix}.gate_up")
        else:
            state[f"{prefix}.gate"], state[f"{prefix}.up"] = gate, up
            keys[layer] = LayerKeys(
                down=f"{prefix}.down", gate=f"{prefix}.gate", up=f"{prefix}.up")
    return ExpertStore.from_state_dict(state, keys, dtype=dtype, ram_budget_bytes=64 * MIB, expert_type=expert_type)


def make_config(store, device, *, cache="reuse", async_staging=False,
                activation_mode="fp32_accumulate", expert_activation="silu",
                numerical_mode="fused", expert_type="glu"):
    # Detect target once during configuration; deployed archives keep this frozen.
    major, minor = torch.cuda.get_device_capability(device)
    expert_bytes = max((w1[0].numel() + w2[0].numel()) * w1.element_size() for w1, w2 in store.layers.values())
    resident_bytes = sum((w1[0].numel() + w2[0].numel()) * w1.element_size() for w1, w2 in store.layers.values())
    return OffloadConfig.from_store(
        store, target_sm=major * 10 + minor,
        gpu_budget_bytes=8 * MIB, ram_budget_bytes=64 * MIB, pinned_budget_bytes=4 * MIB,
        activation_mode=activation_mode, expert_activation=expert_activation,
        numerical_mode=numerical_mode, expert_type=expert_type, max_route_tile_rows="auto", num_banks=2,
        resident_experts={layer: (0,) for layer in store.layers} if cache == "static" else {},
        resident_cache_bytes=resident_bytes if cache == "static" else 0,
        dynamic_cache_bytes=4 * expert_bytes if cache in ("reuse", "lru") else 0,
        cache_policy="lru" if cache == "lru" else "reuse", async_staging=async_staging,
    )


def make_inputs(tokens, device, dtype, *, hotset=False):
    # Construct on CPU: reproducible across CUDA architectures; copy only inputs.
    x = (torch.arange(tokens * HIDDEN, dtype=torch.float32).cos() * .25).reshape(tokens, HIDDEN)
    ids = torch.arange(tokens * TOP_K).reshape(tokens, TOP_K).remainder(2 if hotset else EXPERTS)
    scale = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32)
    return x.to(device=device, dtype=dtype), ids.to(device=device, dtype=torch.int32), scale.to(device)


def rounded_reference(x, w1, w2, ids, scale, *, expert_activation="silu",
                      numerical_mode="fused", expert_type="glu"):
    """CPU FP32 matmuls; exact GELU and explicit native dtype boundaries.

    Fused preserves the historical Act/Down rounding. Staged additionally
    rounds gate/up, the nonlinearity and the scaled product before FP32 summation.
    """
    if expert_activation not in ("silu", "gelu") or numerical_mode not in ("fused", "staged"):
        raise ValueError("invalid expert_activation/numerical_mode")
    if expert_type not in ("glu", "mlp") or w1.shape[1] != (2 if expert_type == "glu" else 1) * w2.shape[2]:
        raise ValueError("expert_type projection shapes")
    value, ids, scale = x.float().cpu(), ids.cpu(), scale.float().cpu()
    result = torch.zeros_like(value)
    staged = numerical_mode == "staged"
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        gate_up = value[rows] @ w1[expert].float().cpu().T
        if staged:
            gate_up = gate_up.to(x.dtype).float()
        gate = gate_up.chunk(2, dim=1)[0] if expert_type == "glu" else gate_up
        act = F.gelu(gate, approximate="none") if expert_activation == "gelu" else F.silu(gate)
        if staged and expert_type == "glu":
            act = act.to(x.dtype).float()
        product = (act * gate_up.chunk(2, dim=1)[1] if expert_type == "glu" else act).to(x.dtype).float()
        down = (product @ w2[expert].float().cpu().T).to(x.dtype).float()
        weighted = down * scale[rows, slots, None]
        if staged:
            weighted = weighted.to(x.dtype).float()
        result.index_add_(0, rows, weighted)
    return result.to(x.dtype)


def check_output(output, expected):
    actual, reference = output.detach().float().cpu(), expected.float().cpu()
    if output.shape != expected.shape or output.dtype != expected.dtype:
        raise AssertionError("output shape/dtype mismatch")
    if not torch.isfinite(actual).all():
        raise AssertionError("non-finite output")
    relative_l2 = ((actual - reference).norm() / reference.norm().clamp_min(1e-12)).item()
    if relative_l2 > .05:
        raise AssertionError(f"numerical mismatch: relative L2={relative_l2:.6g}")
    print(f"output={tuple(output.shape)} {output.dtype}, relative L2={relative_l2:.6g}")


def print_stats(runtime):
    # Stats describe the last SUCCESS, not whether a failed call succeeded.
    stats = runtime.stats()
    names = ("weight_h2d_bytes", "cache_hits", "cache_misses", "cache_admissions",
             "cache_evictions", "waves", "route_tiles", "gpu_charged_bytes", "pinned_bytes")
    print({name: stats[name] for name in names})


def show_recovery(runtime, x, ids, scale):
    """Only this known invalid-route error is acknowledged; never auto-retry."""
    invalid = ids.clone()
    invalid[0, 0] = EXPERTS  # -1 masks a route; E or greater is invalid.
    try:
        runtime.forward(x, invalid, scale, layer_id=0)
    except RuntimeError as error:
        print(f"expected invalid-route failure: {error}")
    else:
        raise AssertionError("invalid route unexpectedly accepted")
    status = runtime.status()
    print("after failure:", status)
    if status["state"] != "recoverable" or not status["pending_error"]:
        raise RuntimeError("unexpected context state: do not retry; close this runtime")
    # check() observes but does not clear; poisoned contexts are close-only.
    try:
        runtime.check()
    except RuntimeError:
        pass
    else:
        raise AssertionError("pending error was not visible")
    runtime.acknowledge()
    runtime.forward(x, ids, scale, layer_id=0)
    runtime.check()
    print("after explicit ACK and valid call:", runtime.status())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    loading = parser.add_mutually_exclusive_group(required=True)
    loading.add_argument("--ops-library", type=Path, help="production .so with matching .so.json; never JIT")
    loading.add_argument("--jit", action="store_true", help="explicit development-only lazy compilation")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--layout", choices=("fused", "separate"), default="fused")
    parser.add_argument("--cache", choices=("none", "static", "reuse", "lru"), default="reuse")
    parser.add_argument("--activation-mode", choices=("route_buffer", "fp32_accumulate", "auto"), default="fp32_accumulate")
    parser.add_argument("--expert-type", choices=("glu", "mlp"), default="glu")
    parser.add_argument("--expert-activation", choices=("silu", "gelu"), default="silu")
    parser.add_argument("--numerical-mode", choices=("fused", "staged"), default="fused")
    parser.add_argument("--async-staging", action="store_true")
    parser.add_argument("--show-recovery", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0 or args.device < 0:
        parser.error("tokens must be positive and device nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("a supported CUDA device is required")
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    load_native(args.ops_library, jit=args.jit)
    store = make_store(dtype=dtype, layout=args.layout, expert_type=args.expert_type)
    config = make_config(store, device, cache=args.cache, async_staging=args.async_staging,
                         activation_mode=args.activation_mode,
                         expert_activation=args.expert_activation, numerical_mode=args.numerical_mode, expert_type=args.expert_type)
    x, ids, scale = make_inputs(args.tokens, device, dtype, hotset=True)
    reference_modes = dict(expert_activation=config.expert_activation,
                           numerical_mode=config.numerical_mode, expert_type=config.expert_type)
    expected = rounded_reference(x, *store.layers[0], ids, scale, **reference_modes)
    print(f"CPU weights={store.weight_bytes} bytes; config={config}")
    # No inference_mode around weight creation: bound CPU weights need versions.
    with torch.no_grad(), PrefillMoERuntime.from_config(store, config, device=device) as runtime:
        for observation in range(3):
            output = runtime.forward(x, ids, scale, layer_id=0)
            runtime.check()
            print(f"hotset observation {observation + 1}")
            check_output(output, expected)
            print_stats(runtime)
            del output
        # Distribute routes over E experts (all are covered when S*K >= E).
        scan_x, scan_ids, scan_scale = make_inputs(args.tokens, device, dtype)
        output = runtime.forward(scan_x, scan_ids, scan_scale, layer_id=0)
        runtime.check()
        print("distributed expert route")
        check_output(output, rounded_reference(
            scan_x, *store.layers[0], scan_ids, scan_scale, **reference_modes))
        print_stats(runtime)
        del output
        if args.show_recovery:
            show_recovery(runtime, x, ids, scale)
    print("OK: context drained and closed")


if __name__ == "__main__":
    main()
