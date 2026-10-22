"""Offline build/export and strict Python deployment of single/two-layer MoE.

Build a production native library offline (requires CUDA/C++ toolchain):
    python examples/fuse_moe_offload_aoti.py build-native --output-dir artifacts/moe_native --arch 8.9
Export a checked PT2 and external CPU weights (never uses native JIT):
    python examples/fuse_moe_offload_aoti.py export --ops-library /path/to/native.so --output-dir artifacts/moe_example --layers 2
Run in a new process, without compiling/loading external CUDA source files:
    python examples/fuse_moe_offload_aoti.py run --ops-library /path/to/native.so --package artifacts/moe_example/model.pt2 --weights artifacts/moe_example/weights.pt

SM120 uses --arch 12.0a with its own matching Torch/CUDA toolchain. Exported S
range is [2,64]; weights and token stay external inputs, config/manifest are graph
constants. Keep the complete official PT2, matching weights archive and native
.so/.so.json together. Only load trusted artifacts; metadata is not a signature.
The C++ companion consumes the SAME PT2/weights files. Use --help for each command.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from custom_ops import setup_compiler  # noqa: E402
from custom_ops.moe_offload import (  # noqa: E402
    ExpertStore, InputBinding, OffloadedMoELayer, PrefillMoERuntime, compile_aoti,
)
from custom_ops.recsys import FuseMoeOps  # noqa: E402
from custom_ops.examples.fuse_moe_offload_demo import (  # noqa: E402
    MIB, check_output, load_native, make_config, make_inputs, make_store,
    print_stats, rounded_reference,
)


class SingleLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = OffloadedMoELayer(0, config=config)

    def forward(self, x, w10, w20, ids, scale, token):
        return (self.layer(x, w10, w20, ids, scale, token),)


class TwoLayers(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first = OffloadedMoELayer(0, config=config)
        self.second = OffloadedMoELayer(5, config=config)

    def forward(self, x, w10, w20, w15, w25, ids, scale, token):
        value = self.first(x, w10, w20, ids, scale, token)
        # Synthetic fixture shares H/E/dtype, so ids/scale can be reused.
        # Real layers usually supply their own router outputs as graph inputs.
        return (self.second(value, w15, w25, ids, scale, token),)


def input_bindings(layer_ids):
    bindings = []
    for index, layer in enumerate(layer_ids):
        bindings.extend((InputBinding(1 + 2 * index, "w1", layer),
                         InputBinding(2 + 2 * index, "w2", layer)))
    bindings.append(InputBinding(3 + 2 * len(layer_ids), "token"))
    return bindings


def graph_inputs(tokens, store, device):
    dtype = store.layers[0][0].dtype
    x, ids, scale = make_inputs(tokens, device, dtype)
    # None is an explicit bound placeholder, not a guessed weight/token slot.
    return (x, *(None for _ in range(2 * len(store.layers))), ids, scale, None)


def load_official_package(path, device):
    """Reviewed official native binding; no CXX version probe or fallback/JIT.

    This is example-level release selection, not a universal business loader.
    No failed constructor is retried with another signature.
    """
    version = str(torch.__version__)
    loader = torch._C._aoti.AOTIModelPackageLoader
    if re.fullmatch(r"2\.6\.0a0(?:\+[^\s]+)?", version):
        return loader(str(path), "model")
    if re.fullmatch(r"2\.11\.0(?:\+[^\s]+)?", version):
        return loader(str(path), "model", False, 1, device.index)
    raise RuntimeError(f"example loader supports reviewed Torch 2.6.0a0/2.11.0, got {version}")


def drain_for_cleanup(device):
    # Cleanup must not mask the original model/native failure.
    original_error = sys.exc_info()[1]
    try:
        torch.cuda.synchronize(device)
    except Exception as cleanup_error:
        if original_error is None:
            raise
        print(f"CUDA cleanup also failed: {cleanup_error}", file=sys.stderr)


def run_calls(runtime, store, artifact_config, loader, device, sizes):
    bindings = input_bindings(tuple(store.layers))
    # Official native loader accepts a flat Tensor list, invoke accepts *args.
    def call_package(*flat):
        return loader.run(list(flat))

    with torch.no_grad():
        for tokens in sizes:
            inputs = graph_inputs(tokens, store, device)
            expected = inputs[0]
            for w1, w2 in store.layers.values():
                expected = rounded_reference(
                    expected, w1, w2, inputs[-3], inputs[-2],
                    expert_activation=artifact_config.expert_activation,
                    numerical_mode=artifact_config.numerical_mode, expert_type=artifact_config.expert_type)
            outputs = runtime.invoke(call_package, inputs,
                                     artifact_config=artifact_config, bindings=bindings)
            if len(outputs) != 1:
                raise AssertionError("expected one output Tensor")
            torch.cuda.synchronize(device)
            if outputs[0].device != device:
                raise AssertionError("output device mismatch")
            print(f"AOTi layers={len(store.layers)}, S={tokens}, "
                  f"expert_type={artifact_config.expert_type}, expert_activation={artifact_config.expert_activation}, "
                  f"numerical_mode={artifact_config.numerical_mode}")
            check_output(outputs[0], expected)
            print_stats(runtime)
            print("status:", runtime.status())
            del outputs


def export_package(args, device):
    setup_compiler()  # Offline only: never probe or configure compilers in run.
    load_native(args.ops_library)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    store = make_store(layers=args.layers, dtype=dtype, layout=args.layout, expert_type=args.expert_type)
    config = make_config(store, device, cache=args.cache, async_staging=args.async_staging,
                         expert_activation=args.expert_activation, numerical_mode=args.numerical_mode, expert_type=args.expert_type)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive_path, package_path = directory / "weights.pt", directory / "model.pt2"
    if archive_path.exists() or package_path.exists():
        raise FileExistsError("output artifacts already exist; choose a new --output-dir")
    # A format2 archive carries the exact frozen contract for Python and C++.
    store.save_cpp(archive_path, weight_version=1, config=config)
    restored = ExpertStore.load_cpp(archive_path, ram_budget_bytes=64 * MIB)
    artifact_config = restored.deployment_config  # From artifact, not runtime self.
    bindings = input_bindings(tuple(restored.layers))
    model = (SingleLayer(artifact_config) if args.layers == 1 else TwoLayers(artifact_config)).eval()
    with PrefillMoERuntime.from_config(restored, artifact_config, device=device) as runtime:
        inputs = runtime.prepare_inputs(graph_inputs(7, restored, device), bindings)
        dim = torch.export.Dim("tokens", min=2, max=64)
        dynamic_shapes = ({0: dim}, *(None for _ in range(2 * args.layers)),
                          {0: dim}, {0: dim}, None)
        exported = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
        if exported.state_dict:
            raise AssertionError("weights/token must remain external, not embedded graph state")
        # Do not replace this with unchecked aoti_compile_and_package: errors
        # must abort graph nodes before downstream sees a failed partial output.
        package = compile_aoti(exported, package_path=str(package_path),
                               inductor_configs={"triton.cudagraphs": False})
        loader = load_official_package(package, device)
        try:
            run_calls(runtime, restored, artifact_config, loader, device, (2, 7, 64))
        finally:
            try:
                drain_for_cleanup(device)
            finally:
                del loader
    print(f"PACKAGE={package_path}\nWEIGHTS={archive_path}")
    print("Move the complete PT2, weights, and matching native .so/.so.json to deployment")


def deploy_package(args, device):
    # No compile/export/lowering registration in this path. The package must
    # already have been compiled using the checked lowering during export.
    load_native(args.ops_library)
    store = ExpertStore.load_cpp(args.weights.resolve(), ram_budget_bytes=64 * MIB)
    artifact_config = getattr(store, "deployment_config", None)
    if artifact_config is None:
        raise ValueError("run requires format2 weights with a frozen configuration")
    if tuple(store.layers) not in ((0,), (0, 5)):
        raise ValueError("this companion example supports layer 0 or layers 0,5")
    with PrefillMoERuntime.from_config(store, artifact_config, device=device) as runtime:
        loader = load_official_package(args.package.resolve(), device)
        try:
            run_calls(runtime, store, artifact_config, loader, device, args.sizes)
        finally:
            try:
                drain_for_cleanup(device)
            finally:
                del loader
    print("OK: strictly precompiled deployment; context closed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-native", help="offline production native build, never a deployment step")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--arch", required=True, help="8.9 or 12.0a; match the target/toolchain")
    export = commands.add_parser("export", help="export checked PT2 plus external CPU weight archive")
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--layers", type=int, choices=(1, 2), default=1)
    export.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    export.add_argument("--layout", choices=("fused", "separate"), default="fused")
    export.add_argument("--cache", choices=("none", "static", "reuse", "lru"), default="reuse")
    export.add_argument("--expert-type", choices=("glu", "mlp"), default="glu")
    export.add_argument("--expert-activation", choices=("silu", "gelu"), default="silu")
    export.add_argument("--numerical-mode", choices=("fused", "staged"), default="fused")
    export.add_argument("--async-staging", action="store_true", help="opt-in frozen config; default sync")
    run = commands.add_parser("run", help="only load and execute; no compiler needed")
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--sizes", type=int, nargs="+", default=[2, 7, 64])
    for command in (export, run):
        command.add_argument("--ops-library", type=Path, required=True, help="production .so and matching .so.json")
        command.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "build-native":
        # Explicit offline build; deployment paths never call this API.
        setup_compiler()
        library = FuseMoeOps(allow_jit=False).build(args.output_dir.resolve(),
                                                   cuda_arch_list=args.arch, verbose=True)
        print(f"NATIVE_LIBRARY={library}\nMETADATA={library}.json")
        return
    if args.device < 0:
        parser.error("device must be nonnegative")
    if args.command == "run" and any(not 2 <= s <= 64 for s in args.sizes):
        parser.error("all sizes must be in the exported range [2,64]")
    if not torch.cuda.is_available():
        raise RuntimeError("a supported CUDA device is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    if args.command == "export":
        export_package(args, device)
    else:
        deploy_package(args, device)


if __name__ == "__main__":
    main()
