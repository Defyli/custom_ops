"""Strict RAM-offload export/AOTI acceptance, including a Python-free executable.

Run: PYTHONPATH=/workdir python tests/test_fuse_moe_offload_aoti.py --ops-library /path/to/recsys_fuse_moe_kernel.so
No compile/load/runtime failure is skipped. --output-dir retains raw acceptance evidence.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass, replace
from functools import partial
import gc
import hashlib
import re
import math
import statistics
import resource
import os
import platform
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading
import tempfile
import time
import zipfile
from unittest import mock

import torch
from torch.utils import _pytree as pytree

from custom_ops import ops, setup_compiler
from custom_ops.moe_offload import ExpertStore, LayerKeys, OffloadConfig, OffloadedMoELayer, PrefillMoERuntime, register_aoti_lowering
from custom_ops.moe_offload import PrefillMoERuntime as _LockedPrefillMoERuntime
from custom_ops.moe_offload_aoti import compile_aoti

E, H, I, K = 4, 64, 64, 2
EXPERT_BYTES = 6 * H * I
GPU_BUDGET = 128 * 1024
# Enough for two one-expert banks, but not two two-expert banks.
PINNED_BUDGET = 4 * (E + 1) + 2 * (EXPERT_BYTES + 24)
# Same total RAM cap in sync/async: CPU weights, pinned/planning buffers,
# and the bounded worker's 1 MiB stack plus fixed bookkeeping allowance.
RAM_BUDGET = 4 * 1024 * 1024
ATOL, RTOL = 2e-4, 2e-2
STATS_FIELDS = (
    "active_experts", "waves", "weight_h2d_bytes", "metadata_h2d_bytes",
    "gpu_charged_bytes", "pinned_bytes", "ram_charged_bytes", "num_banks", "wave_capacity",
    "activation_mode", "route_tiles", "route_tile_rows", "buffer_allocations",
    "buffer_reuses", "gpu_pool_bytes", "pinned_pool_bytes",
    "cache_hits", "cache_misses", "cache_hit_weight_bytes", "resident_cache_bytes",
    "resident_experts", "resident_load_h2d_bytes",
    "dynamic_cache_bytes", "dynamic_experts", "dynamic_cache_hits",
    "cache_admissions", "cache_evictions", "cache_bypasses",
    "cache_promotion_d2d_bytes", "cache_budget_evictions", "cache_metadata_bytes",
)
DYNAMIC_CACHE_CAP = ((EXPERT_BYTES + 511) // 512) * 512
CACHED_GPU_BUDGET = 256 * 1024
CACHE_PRESETS = (("disabled", ()), ("partial", (0, 2)), ("adjacent", (0, 1)),
                 ("all", tuple(range(E))))


@dataclass(frozen=True)
class _CacheInfo:
    name: str
    expert_ids: tuple[int, ...]
    cache_cap: int
    gpu_budget: int
    actual_bytes: int
    load_bytes: int


def _stats(runtime):
    raw = torch.ops.recsys_ops.fuse_moe_offload_stats(runtime.token)
    assert raw.device.type == "cpu" and raw.dtype == torch.int64 and raw.numel() == 31, (
        "Native stats must contain all 31 fields", raw)
    stats = runtime.stats()
    assert set(stats) == set(STATS_FIELDS), "Python runtime stats must match all 31 native fields"
    assert stats == dict(zip(STATS_FIELDS, raw.tolist())), stats
    return stats


def _check_initial_dynamic_stats(stats, *, enabled):
    assert all(stats[field] == 0 for field in STATS_FIELDS[22:30]), stats
    assert (stats["cache_metadata_bytes"] > 0 if enabled
            else stats["cache_metadata_bytes"] >= 0), stats


def _check_dynamic_stats(stats, expected, cache_cap=DYNAMIC_CACHE_CAP):
    active, hits, admissions, evictions, bypasses, cached = expected[:6]
    waves = expected[6] if len(expected) == 7 else active
    misses = active - hits
    assert stats["active_experts"] == active and stats["waves"] == waves, stats
    assert stats["cache_hits"] == stats["dynamic_cache_hits"] == hits, stats
    assert stats["cache_misses"] == misses, stats
    assert stats["cache_hit_weight_bytes"] == hits * EXPERT_BYTES, stats
    # Promotion is D2D after the miss wave: never charge a second weight H2D.
    assert stats["weight_h2d_bytes"] == misses * EXPERT_BYTES, stats
    assert stats["metadata_h2d_bytes"] == misses * 24, stats
    assert stats["cache_admissions"] == admissions and stats["cache_evictions"] == evictions, stats
    assert stats["cache_bypasses"] == bypasses, stats
    assert stats["cache_promotion_d2d_bytes"] == admissions * EXPERT_BYTES, stats
    # cache_update_bytes=0 caps total promotion traffic per call at dynamic capacity.
    assert 0 <= stats["cache_promotion_d2d_bytes"] <= cache_cap, stats
    assert stats["dynamic_experts"] == cached, stats
    assert stats["dynamic_cache_bytes"] == cached * DYNAMIC_CACHE_CAP, stats
    assert stats["cache_budget_evictions"] == 0 and stats["cache_metadata_bytes"] > 0, stats
    assert all(stats[field] == 0 for field in STATS_FIELDS[19:22]), stats
    assert stats["num_banks"] == (2 if misses else 0), stats
    assert stats["wave_capacity"] == (1 if misses else 0), stats
    assert (stats["gpu_pool_bytes"] + stats["dynamic_cache_bytes"] <=
            stats["gpu_charged_bytes"] <= CACHED_GPU_BUDGET), stats


def _check_dynamic_contexts(fn, store, w1, w2, device, label):
    # Fresh tokens make first/second/third-call semantics independent of tracing
    # and previous backends. The callable (especially AOTI) is never recompiled.
    for mode, tile_rows in ((0, 4096), (1, 3), (1, "auto")):
        for policy in ("reuse", "lru"):
            options = dict(device=device, gpu_budget_bytes=CACHED_GPU_BUDGET,
                           ram_budget_bytes=RAM_BUDGET, pinned_budget_bytes=PINNED_BUDGET,
                           num_banks=2, dynamic_cache_bytes=DYNAMIC_CACHE_CAP,
                           cache_decay_interval=32, cache_min_reuse=2, cache_update_bytes=0,
                           activation_mode="route_buffer" if mode == 0 else "fp32_accumulate",
                           max_route_tile_rows=tile_rows)
            if policy == "lru":
                options["cache_policy"] = policy  # reuse deliberately tests the default.
            with PrefillMoERuntime(store, **options) as runtime:
                _check_initial_dynamic_stats(_stats(runtime), enabled=True)
                args = _inputs(7, w1, w2, runtime.token, device)
                hot = (*args[:3], torch.zeros_like(args[3]), *args[4:])

                def check(route, stage, expected):
                    return _check_call(fn, route, runtime,
                                       f"{label} dynamic={policy} mode={mode} tile={tile_rows} {stage}",
                                       mode=mode, dynamic=expected, tile_rows=tile_rows)

                # Fourteen repeated routes still count as one frequency observation.
                check(hot, "first-cold", (1, 0, int(policy == "lru"), 0,
                                          int(policy == "reuse"), int(policy == "lru")))
                check(hot, "second-admission", (1, int(policy == "lru"),
                                                int(policy == "reuse"), 0, 0, 1))
                check(hot, "third-hit-zero-H2D", (1, 1, 0, 0, 0, 1))
                for warm in range(5):
                    check(hot, f"warm-hot-{warm}", (1, 1, 0, 0, 0, 1))
                # Reuse protects the hot key. LRU promotes the first cold scan
                # key, then bypasses the other two after exhausting the update cap.
                check(args, "distinct-scan", (E, 1, 1 if policy == "lru" else 0,
                                              1 if policy == "lru" else 0,
                                              2 if policy == "lru" else 3, 1))
                check(hot, "post-scan-hot", (1, int(policy == "reuse"),
                                             int(policy == "lru"), int(policy == "lru"), 0, 1))
                for s in (2, 64, 7):
                    resized = _inputs(s, w1, w2, runtime.token, device)
                    resized = (*resized[:3], torch.zeros_like(resized[3]), *resized[4:])
                    check(resized, f"dynamic-S={s}-hot-hit", (1, 1, 0, 0, 0, 1))
            # A separate two-entry context verifies grouped dynamic pointers in
            # the same exported/compiled graph, without recompilation.
            options["dynamic_cache_bytes"] = 2 * DYNAMIC_CACHE_CAP
            with PrefillMoERuntime(store, **options) as runtime:
                for stage in range(3):
                    args = _inputs(7, w1, w2, runtime.token, device)
                    hot = (*args[:3], args[3].remainder(2), *args[4:])
                    hits = 2 if (stage >= 1 if policy == "lru" else stage >= 2) else 0
                    admission = 2 if stage == (0 if policy == "lru" else 1) else 0
                    _check_call(fn, hot, runtime,
                                f"{label} grouped-dynamic={policy} mode={mode} stage={stage}",
                                mode=mode, dynamic=(2, hits, admission, 0,
                                                    2 if stage == 0 and policy == "reuse" else 0,
                                                    2 if stage >= (0 if policy == "lru" else 1) else 0,
                                                    1 if hits == 2 else 2),
                                dynamic_cap=2 * DYNAMIC_CACHE_CAP, tile_rows=tile_rows)


def _get_cache_info(runtime, name, expert_ids, cache_cap, gpu_budget, expert_bytes):
    # Snapshot context-lifetime initialization traffic, not per-forward traffic.
    stats = _stats(runtime)
    count = len(expert_ids)
    _check_initial_dynamic_stats(stats, enabled=False)
    actual = stats["resident_cache_bytes"]
    assert stats["resident_experts"] == count, stats
    assert stats["resident_load_h2d_bytes"] == count * expert_bytes, stats
    if count:
        assert count * expert_bytes <= actual <= cache_cap and actual % 512 == 0, stats
    else:
        assert cache_cap == actual == 0, stats
    return _CacheInfo(name, expert_ids, cache_cap, gpu_budget,
                      actual, stats["resident_load_h2d_bytes"])


def _check_cache_stats(stats, cache, *, experts, expert_bytes, miss_capacity):
    assert all(stats[field] == 0 for field in STATS_FIELDS[22:30]), stats
    assert stats["cache_metadata_bytes"] >= 0, stats
    hits = len(cache.expert_ids)  # This fixture activates every expert uniformly.
    misses = experts - hits
    assert stats["cache_hits"] == hits and stats["cache_misses"] == misses, stats
    assert stats["cache_hit_weight_bytes"] == hits * expert_bytes, stats
    assert stats["weight_h2d_bytes"] == misses * expert_bytes, stats
    assert stats["resident_experts"] == hits, stats  # One bound layer in these contexts.
    assert stats["resident_cache_bytes"] == cache.actual_bytes, stats
    assert stats["resident_load_h2d_bytes"] == cache.load_bytes, stats
    # Consecutive hits share a pointer-table wave; misses stop at hit boundaries.
    miss_waves = 0
    run = 0
    for expert in range(experts + 1):
        if expert < experts and expert not in cache.expert_ids:
            run += 1
        elif run > 0:
            assert miss_capacity > 0, "Miss waves require positive capacity"
            miss_waves += (run + miss_capacity - 1) // miss_capacity
            run = 0
    hit_waves = sum(expert in cache.expert_ids and
                    (expert == 0 or expert - 1 not in cache.expert_ids)
                    for expert in range(experts))
    assert stats["waves"] == hit_waves + miss_waves, stats
    assert stats["metadata_h2d_bytes"] == (4 * misses + 2 * miss_waves) * 4, stats
    assert stats["num_banks"] == (2 if misses else 0), stats
    assert stats["wave_capacity"] == (miss_capacity if misses else 0), stats
    assert 0 < stats["gpu_pool_bytes"], stats
    # The static cache is outside the reusable pool but inside the total charge.
    assert stats["gpu_pool_bytes"] + cache.actual_bytes <= stats["gpu_charged_bytes"], stats
    assert 0 < stats["gpu_charged_bytes"] <= cache.gpu_budget, stats


@contextmanager
def _artifact_directory(prefix, output_dir=None):
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
            yield Path(temporary)
    else:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        yield directory


def _environment_signature(library=None):
    signature = dict(python=sys.version, platform=platform.platform(),
                     torch=str(torch.__version__), cuda_build=torch.version.cuda,
                     cxx_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI),
                     ops_library=None if library is None else str(Path(library).resolve()))
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        signature.update(device_index=device, gpu=torch.cuda.get_device_name(device),
                         compute_capability=list(torch.cuda.get_device_capability(device)))
    return signature


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_proxy_proof(source, indices):
    # Match executable calls, not comments or unrelated checked aten statements.
    code = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"',
                  lambda match: " " * len(match[0]),
                  source, flags=re.DOTALL)
    calls = []
    for match in re.finditer(
            r"\baoti_torch_proxy_executor_call_function\s*\(\s*proxy_executor\s*,\s*(\d+)\s*,", code):
        index = int(match[1])
        if index not in indices:
            continue
        checked = re.search(r"\bAOTI_TORCH_ERROR_CODE_CHECK\s*\(\s*$", code[:match.start()]) is not None
        calls.append(dict(extern_index=index, line=source.count("\n", 0, match.start()) + 1,
                          checked=checked))
    return calls


def _verify_checked_package(package_path, *, expected_calls=None):
    package_path = Path(package_path).resolve()
    folder = package_path.parent / (package_path.stem + ".source_proof")
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, package=str(package_path), package_sha256=_sha256(package_path),
                  state="running", sources=[], calls=[])
    try:
        with zipfile.ZipFile(package_path) as archive:
            objects = [name for name in archive.namelist() if name.endswith(".so")]
            assert len(objects) == 1, ("Expected one official model shared object", objects)
            json_name = str(Path(objects[0]).with_suffix(".json"))
            serialized = archive.read(json_name)
            (folder / "extern.json").write_bytes(serialized)
            extern = json.loads(serialized)
            nodes = extern["nodes"]
            # The index in the proxy ABI is the position in official extern JSON,
            # NOT a generated node name (the checked IR subclass changes names).
            indices = {index for index, node in enumerate(nodes)
                       if node["node"]["target"] in (
                            "torch.ops.recsys_ops.fuse_moe_offload.default",
                            "recsys_ops.fuse_moe_offload.default", "recsys_ops::fuse_moe_offload")}
            assert indices, ("No offload targets in official extern JSON", extern)
            if expected_calls is not None:
                assert len(indices) == expected_calls, (indices, expected_calls)
            for ordinal, name in enumerate(archive.namelist()):
                if not name.endswith(".cpp"):
                    continue
                raw = archive.read(name)
                destination = folder / f"{ordinal}_{Path(name).name}"
                destination.write_bytes(raw)  # Original contents; never patch generated code.
                source = raw.decode("utf-8")
                calls = _checked_proxy_proof(source, indices)
                report["sources"].append(dict(member=name, saved=destination.name,
                                              sha256=hashlib.sha256(raw).hexdigest()))
                report["calls"].extend(dict(call, member=name) for call in calls)
            assert report["sources"], "Official package lacks generated .cpp; lead must enable source packaging/rebuild"
            for index in sorted(indices):
                matches = [call for call in report["calls"] if call["extern_index"] == index]
                assert len(matches) == 1 and matches[0]["checked"], (
                    "Every offload proxy must be directly checked exactly once; ask lead to rebuild, never edit generated source",
                    index, matches)
            report.update(state="passed", offload_indices=sorted(indices),
                          extern_sha256=hashlib.sha256(serialized).hexdigest())
    except BaseException as error:
        report.update(state="failed", error=repr(error))
        raise
    finally:
        _write_json(folder / "proof.json", report)
    print(f"PASS CHECKED_SOURCE {package_path.name}: {len(report['calls'])} offload proxies", flush=True)
    return report


def _compile_checked_package(exported, *, package_path, inductor_configs):
    op = torch.ops.recsys_ops.fuse_moe_offload.default
    expected = sum(node.target == op or any(value == op for value in
                   pytree.tree_leaves((node.args, node.kwargs)) if isinstance(value, torch._ops.OpOverload))
                   for node in exported.graph.nodes if node.op == "call_function")
    # The public feature disables pre-lowering caches and audits the official API.
    # Keep independent generated-source proof mandatory for every new package.
    result = compile_aoti(
        exported, package_path=package_path, inductor_configs=inductor_configs)
    _verify_checked_package(result, expected_calls=expected)
    return result


def _run_process(command, directory, label, *, env=None, check=True):
    # A crashing official fallback is a failed acceptance, not a skip or a patched
    # output. The parent retains negative returncode/signal and unmodified stderr.
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    (directory / f"{label}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (directory / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    _write_json(directory / f"{label}.exit.json", dict(
        command=list(map(str, command)), returncode=completed.returncode,
        signal=-completed.returncode if completed.returncode < 0 else None,
        elapsed_ms=(time.perf_counter() - started) * 1000))
    print(completed.stdout, end="", flush=True)
    print(completed.stderr, end="", file=sys.stderr, flush=True)
    if check:
        completed.check_returncode()
    return completed


@contextmanager
def _cli_evidence(output_dir, options):
    if output_dir is None:
        yield
        return
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve original streams while writing complete Python-side reports.
    class Tee:
        def __init__(self, original, archive):
            self.original, self.archive = original, archive
        def write(self, text):
            self.archive.write(text)
            self.archive.flush()
            return self.original.write(text)
        def flush(self):
            self.archive.flush()
            self.original.flush()

    report = dict(schema_version=1, argv=sys.argv, options=vars(options),
                  environment=_environment_signature(options.ops_library), state="running")
    _write_json(directory / "acceptance.exit.json", report)
    with (directory / "acceptance.stdout.txt").open("w", encoding="utf-8") as stdout, \
            (directory / "acceptance.stderr.txt").open("w", encoding="utf-8") as stderr, \
            redirect_stdout(Tee(sys.stdout, stdout)), redirect_stderr(Tee(sys.stderr, stderr)):
        try:
            yield
        except BaseException as error:
            report.update(state="failed", returncode=1, error=repr(error))
            print(f"FAIL acceptance: {error!r}", file=sys.stderr, flush=True)
            raise
        else:
            report.update(state="passed", returncode=0)
        finally:
            _write_json(directory / "acceptance.exit.json", report)
            _write_json(directory / "evidence.hashes.json", {
                str(path.relative_to(directory)): _sha256(path)
                for path in sorted(directory.rglob("*")) if path.is_file() and
                (path.suffix in (".cpp", ".json", ".txt", ".pt2", ".so") or
                 path.name == "fuse_moe_offload_aoti_runner") and
                path.name != "evidence.hashes.json"})


def _load_ops(library=None, wait_seconds=0, *, allow_testing=False, compile_lowering=True):
    """Reuse an explicitly supplied library without entering the shared JIT dir."""
    if library is not None:
        library = Path(library).resolve()
        deadline = time.monotonic() + wait_seconds
        while not library.is_file():
            if time.monotonic() >= deadline:
                raise FileNotFoundError(library)
            time.sleep(1)
        # The public strict loader owns path.json validation, registration and
        # global no-JIT policy. Never bypass metadata or mark private groups loaded.
        ops.load_precompiled("fuse_moe", library, allow_testing=allow_testing)
    ops.ensure_loaded("fuse_moe")
    if not ops.is_available("fuse_moe"):
        raise RuntimeError(ops.load_error("fuse_moe"))
    if library is None:
        candidates = [Path(p).resolve() for p in torch.ops.loaded_libraries
                      if "fuse_moe" in Path(p).name]
        if len(candidates) != 1:
            # cpp_extension imports a pybind module instead of load_library.
            extension = ops._groups["fuse_moe"]._ext
            path = getattr(extension, "__file__", None)
            if path is None:
                raise RuntimeError(f"Cannot identify native offload library: {candidates}")
            library = Path(path).resolve()
        else:
            library = candidates[0]
    # ensure_loaded has registered the existing fake kernel. Registration is
    # idempotent and must precede every export/compile, never repair old artifacts.
    if compile_lowering:
        register_aoti_lowering()
    schema = str(torch.ops.recsys_ops.fuse_moe_offload.default._schema)
    if "Tensor(a!) runtime_token" not in schema:
        raise AssertionError(f"Missing real runtime_token mutation schema: {schema}")
    create_schema = str(torch.ops.recsys_ops.fuse_moe_offload_create.default._schema)
    expected_create = (
        "fuse_moe_offload_create(int device, int gpu_budget_bytes=0, int ram_budget_bytes=0, "
        "int pinned_budget_bytes=0, int num_banks=2, int activation_mode=0, "
        "int max_route_tile_rows=4096, int resident_cache_bytes=0, int dynamic_cache_bytes=0, "
        "int cache_policy=0, int cache_decay_interval=32, int cache_min_reuse=2, "
        "int cache_update_bytes=0, bool async_staging=False, bool host_profile=False, "
        "int tile_m=0, int tile_n=128) -> Tensor")
    assert create_schema.endswith(expected_create), f"Dynamic create schema is not ready: {create_schema}"
    bind_schema = str(torch.ops.recsys_ops.fuse_moe_offload_resident_bind.default._schema)
    assert bind_schema.endswith(
        "fuse_moe_offload_resident_bind(Tensor token, int layer_id, int[] expert_ids) -> ()"
    ), f"P3 resident bind schema is not ready: {bind_schema}"
    print(f"Native library: {library}\nSchema: {schema}\nCreate: {create_schema}\n"
          f"Resident bind: {bind_schema}", flush=True)
    return library


def _weights():
    w1 = (torch.arange(E * 2 * I * H, dtype=torch.float32).sin() * .08)
    w2 = ((torch.arange(E * H * I, dtype=torch.float32) + .3).cos() * .08)
    return w1.reshape(E, 2 * I, H).half(), w2.reshape(E, H, I).half()


def _inputs(s, w1, w2, token, device):
    x = (torch.arange(s * H, dtype=torch.float32).cos() * .25).reshape(s, H)
    ids = (torch.arange(s * K, dtype=torch.int64) % E).reshape(s, K)
    scale = torch.full((s, K), 1.0 / K, dtype=torch.float32)
    return (x.to(device=device, dtype=torch.float16), w1, w2,
            ids.to(device=device, dtype=torch.int32), scale.to(device), token)


def _reference(args):
    x, w1, w2, ids, scale, _ = args
    x, ids, scale = x.float().cpu(), ids.cpu(), scale.cpu()
    out = torch.zeros_like(x)
    for expert in range(E):
        rows, slots = torch.where(ids == expert)
        gate, up = (x[rows] @ w1[expert].float().T).chunk(2, dim=-1)
        values = (torch.nn.functional.silu(gate) * up) @ w2[expert].float().T
        out.index_add_(0, rows, values * scale[rows, slots, None])
    return out


def _single_output(value):
    flat, _ = pytree.tree_flatten(value)
    if len(flat) != 1 or not isinstance(flat[0], torch.Tensor):
        raise AssertionError(f"Expected one tensor output, got {type(value)}")
    return flat[0]


def _check_call(fn, args, runtime, label, *, cache=None, dynamic=None,
                invocations=1, mode=0, check_reuse=False, dynamic_cap=DYNAMIC_CACHE_CAP,
                tile_rows=3):
    token = runtime.token
    before = token.clone()
    ptr = token.data_ptr()
    version = token._version
    result = _single_output(fn(*args))
    torch.cuda.synchronize()
    assert token.data_ptr() == ptr and token[0].item() == before[0].item(), label
    assert token[1].item() == before[1].item() + invocations, f"{label}: lost/doubled token mutation"
    assert token._version == version + invocations, f"{label}: lost/doubled token version bump"
    assert result.device == args[0].device and result.dtype == args[0].dtype
    assert result.shape == args[0].shape
    reference_args = args
    for stage in range(invocations):
        expected = _reference(reference_args)
        if stage + 1 < invocations:
            # Model the actual FP16 boundary between opaque offload calls;
            # never feed unrounded FP32 layer-one output into layer two.
            reference_args = (expected.to(dtype=args[0].dtype), *args[1:])
    # This fixture's two-layer output is ~1e-6; the single-layer 2e-4
    # tolerance would even accept all zeros. Use a subnormal-FP16-aware bound.
    absolute_tolerance = ATOL if invocations == 1 else 1e-7
    torch.testing.assert_close(result.float().cpu(), expected,
                               atol=absolute_tolerance, rtol=RTOL)
    stats = _stats(runtime)
    assert stats["activation_mode"] == mode, stats
    assert stats["route_tiles"] >= 0 and stats["route_tile_rows"] >= 0, stats
    if mode == 1:
        assert stats["route_tiles"] > 0 and stats["route_tile_rows"] > 0, stats
        if tile_rows != "auto":
            assert stats["route_tile_rows"] <= tile_rows, stats
        if tile_rows == 3 and args[0].shape[0] >= 7:
            assert stats["route_tiles"] > 1, stats
    assert stats["buffer_allocations"] >= 0 and stats["buffer_reuses"] >= 0, stats
    if check_reuse:
        assert stats["buffer_allocations"] == 0 and stats["buffer_reuses"] > 0, stats
    assert 0 < stats["gpu_pool_bytes"] <= stats["gpu_charged_bytes"], stats
    assert 0 < stats["pinned_pool_bytes"] <= stats["pinned_bytes"], stats
    if dynamic is not None:
        _check_dynamic_stats(stats, dynamic, dynamic_cap)
    else:
        assert stats["active_experts"] == E, stats
        _check_cache_stats(stats, cache, experts=E, expert_bytes=EXPERT_BYTES, miss_capacity=1)
    if dynamic is None and not cache.expert_ids:
        # Preserve the original disabled-cache multi-wave/budget acceptance.
        assert stats["waves"] > 1 and stats["wave_capacity"] < E, stats
        assert stats["weight_h2d_bytes"] == E * EXPERT_BYTES, stats
        assert stats["num_banks"] == 2 and stats["wave_capacity"] == 1, stats
        assert stats["gpu_charged_bytes"] <= GPU_BUDGET, stats
    assert 0 < stats["pinned_bytes"] <= PINNED_BUDGET, stats
    assert runtime.store.weight_bytes <= stats["ram_charged_bytes"] <= RAM_BUDGET, stats
    assert all(w.device.type == "cpu" and not w.is_pinned() for w in args[1:3])
    print(f"PASS {label}: token={token.tolist()}, version={version}->{token._version}, stats={stats}", flush=True)
    # Never compare pool charges to a single-call theoretical allocation: the
    # persistent high-water pool may be larger, but must remain within budget.
    del result  # Release live output storage before any repeated call.
    return stats


class _TupleLayer(torch.nn.Module):
    """AOTI expects flat tuple output on some 2.6 builds; preserve the actual op."""
    def __init__(self):
        super().__init__()
        self.layer = OffloadedMoELayer(0)

    def forward(self, x, w1, w2, ids, scale, token):
        return (self.layer(x, w1, w2, ids, scale, token),)


class _SharedTokenChain(torch.nn.Module):
    """Two opaque calls sharing one external deployment token and CPU weights."""
    def __init__(self):
        super().__init__()
        self.layer = OffloadedMoELayer(0)

    def forward(self, x, w1, w2, ids, scale, token):
        intermediate = self.layer(x, w1, w2, ids, scale, token)
        return (self.layer(intermediate, w1, w2, ids, scale, token),)


def _check_shared_token_chain(directory, contexts, w1, w2, device, sizes):
    _, runtime, _ = contexts[0]
    chain = _SharedTokenChain().eval()
    for mode, context, cache in contexts:
        for s in sizes:
            _check_call(chain, _inputs(s, w1, w2, context.token, device),
                        context, f"shared-token eager mode={mode} cache={cache.name} S={s}",
                        invocations=2, mode=mode, cache=cache)
    args = _inputs(7, w1, w2, runtime.token, device)
    snapshot = runtime.token.clone()
    version = runtime.token._version
    dim = torch.export.Dim("chain_S", min=2, max=64)
    exported = torch.export.export(
        chain, args, dynamic_shapes=({0: dim}, None, None, {0: dim}, {0: dim}, None))
    torch.testing.assert_close(runtime.token, snapshot, atol=0, rtol=0)
    assert runtime.token._version == version, "Chain export mutated token version"
    assert not exported.state_dict, "Chain weights/token must remain external inputs"
    graph = exported.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    assert len(placeholders) == 6, graph
    _, gate_up, down, ids, scale, token = placeholders
    op = torch.ops.recsys_ops.fuse_moe_offload.default
    calls = [node for node in graph.nodes if node.op == "call_function" and node.target == op]
    assert len(calls) == 2, f"Chain must retain two opaque offload calls: {graph}"
    first, second = calls
    assert first.args[1:6] == (gate_up, down, ids, scale, token), graph
    assert second.args[0] is first, "Chain lost intermediate data dependency"
    assert second.args[1:6] == (gate_up, down, ids, scale, token), graph
    print(f"PASS shared-token export: {graph}", flush=True)
    for mode, context, cache in contexts:
        for s in sizes:
            _check_call(exported.module(), _inputs(s, w1, w2, context.token, device),
                        context, f"shared-token exported mode={mode} cache={cache.name} S={s}",
                        invocations=2, mode=mode, cache=cache)
    snapshot = runtime.token.clone()
    version = runtime.token._version
    package_path = _compile_checked_package(
        exported, package_path=str(directory / "shared_token_chain.pt2"),
        inductor_configs={"triton.cudagraphs": False})
    torch.testing.assert_close(runtime.token, snapshot, atol=0, rtol=0)
    assert runtime.token._version == version, "Chain compile mutated token version"
    assert Path(package_path).is_file() and directory in Path(package_path).resolve().parents
    with zipfile.ZipFile(package_path) as archive:
        so_names = [name for name in archive.namelist() if name.endswith(".so")]
        assert len(so_names) == 1, archive.namelist()
        extern_kernels = json.loads(archive.read(str(Path(so_names[0]).with_suffix(".json"))))
        assert "fuse_moe_offload" in json.dumps(extern_kernels), extern_kernels
    loaded = torch._inductor.aoti_load_package(package_path)
    try:
        for mode, context, cache in contexts:
            for s in (*sizes, sizes[0]):
                _check_call(loaded, _inputs(s, w1, w2, context.token, device),
                            context, f"shared-token Python AOTI mode={mode} cache={cache.name} S={s}",
                            invocations=2, mode=mode, cache=cache)
    finally:
        del loaded
    print("PASS shared-token two-offload PT2 chain", flush=True)


class _TwoLayerChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = OffloadedMoELayer(0)
        self.second = OffloadedMoELayer(5)

    def forward(self, x, w10, w20, w15, w25, ids, scale, token):
        intermediate = self.first(x, w10, w20, ids, scale, token)
        return (self.second(intermediate, w15, w25, ids, scale, token),)


def _checkpoint_store(directory, version):
    generator = torch.Generator().manual_seed(20261012 + version)
    # Random scaled weights keep the real two-layer output well above FP16 subnormals.
    state = {name: torch.randn(shape, generator=generator) * .12
             for name, shape in (("layer0.gate_up", (E, 2 * I, H)),
                                 ("layer0.down", (E, H, I)),
                                 ("layer5.gate", (E, I, H)),
                                 ("layer5.up", (E, I, H)),
                                 ("layer5.down", (E, H, I)))}
    checkpoint = directory / f"state_dict_v{version}.pth"
    torch.save(state, checkpoint)
    keys = {0: LayerKeys(gate_up="layer0.gate_up", down="layer0.down"),
            5: LayerKeys(gate="layer5.gate", up="layer5.up", down="layer5.down")}
    started = time.perf_counter()
    store = ExpertStore.load(checkpoint, keys, dtype=torch.float16, ram_budget_bytes=RAM_BUDGET)
    checkpoint_load_ms = (time.perf_counter() - started) * 1000
    archive = directory / f"cpp_weights_v{version}.pt"
    store.save_cpp(archive, weight_version=version)
    started = time.perf_counter()
    restored = ExpertStore.load_cpp(archive, ram_budget_bytes=RAM_BUDGET)
    archive_load_ms = (time.perf_counter() - started) * 1000
    assert restored.weight_version == version and list(restored.layers) == [0, 5]
    for layer_id in store.layers:
        for original, loaded in zip(store.layers[layer_id], restored.layers[layer_id]):
            torch.testing.assert_close(original, loaded, atol=0, rtol=0)
    print("ACCEPTANCE_LOAD " + json.dumps(dict(
        weight_version=version, checkpoint_load_ms=checkpoint_load_ms,
        archive_load_ms=archive_load_ms,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_note="Linux process lifetime ru_maxrss high-water; not an exact RAM budget")), flush=True)
    return restored, archive


def _chain_inputs(s, store, token, device):
    x, w10, w20, ids, scale, token = _inputs(s, *store.layers[0], token, device)
    return (x, w10, w20, *store.layers[5], ids, scale, token)


def _check_real_chain(fn, args, runtime, label, mode):
    x, w10, w20, w15, w25, ids, scale, token = args
    before, ptr, version = token.clone(), token.data_ptr(), token._version
    expected_first = _reference((x, w10, w20, ids, scale, token))
    expected = _reference((expected_first.half(), w15, w25, ids, scale, token))
    output = _single_output(fn(*args))
    torch.cuda.synchronize()
    assert token.data_ptr() == ptr and token[0].item() == before[0].item(), label
    assert token[1].item() == before[1].item() + 2, label
    assert token._version == version + 2, label
    assert output.device == x.device and output.dtype == x.dtype and output.shape == x.shape
    actual = output.float().cpu()
    # 2e-6 is ~34 FP16 subnormal steps, not the single-layer 2e-4 blanket bound.
    # Combined with 2% pointwise and 1% relative L2, zeros/wrong-layer weights fail.
    assert expected.abs().max().item() > 1e-4, "Real chain fixture underflowed"
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=RTOL)
    relative_l2 = (actual - expected).norm() / expected.norm()
    assert relative_l2.item() < .01, (label, relative_l2.item())
    stats = _stats(runtime)
    cache = _CacheInfo("disabled", (), 0, GPU_BUDGET, 0, 0)
    _check_cache_stats(stats, cache, experts=E, expert_bytes=EXPERT_BYTES, miss_capacity=1)
    assert stats["active_experts"] == E and stats["activation_mode"] == mode, stats
    assert stats["waves"] == E and stats["weight_h2d_bytes"] == E * EXPERT_BYTES, stats
    assert stats["buffer_allocations"] >= 0 and stats["buffer_reuses"] >= 0, stats
    assert 0 < stats["pinned_pool_bytes"] <= stats["pinned_bytes"] <= PINNED_BUDGET, stats
    assert runtime.store.weight_bytes <= stats["ram_charged_bytes"] <= RAM_BUDGET, stats
    if mode == 1:
        assert stats["route_tiles"] > 1 and 0 < stats["route_tile_rows"] <= 3, stats
    assert all(w.device.type == "cpu" and not w.is_pinned() for w in args[1:5])
    print(f"PASS {label}: token={token.tolist()} max_abs={(actual - expected).abs().max().item()} "
          f"relative_l2={relative_l2.item()} stats={stats}", flush=True)
    return actual


def _wide_inputs(s, w1, w2, token, device):
    # Keep this fixture identical to make_wide_inputs() in the standalone runner.
    h = w1.shape[2]
    x = (torch.arange(s * h, dtype=torch.float32).cos() * .25).reshape(s, h)
    ids = torch.tensor([[0, 0], [1, -1], [2, 3], [-1, -1],
                        [3, 1], [2, 2], [0, 3]], dtype=torch.int32)
    scale = torch.tensor([[.75, -.25], [-.5, 1.25], [.125, -.875], [.625, -1.5],
                          [-.375, .5], [1.125, -.625], [-.75, .25]], dtype=torch.float32)
    rows = torch.arange(s).remainder(7)
    return (x.to(device=device, dtype=w1.dtype), w1, w2,
            ids[rows].to(device), scale[rows].to(device), token)


def _check_wide_call(fn, args, runtime, cache, tile_rows, label, *, check_reuse=False):
    x, w1, w2, ids, scale, token = args
    assert x.shape[1] == 128 and w2.shape[2] == 64
    host_ids, host_scale = ids.cpu(), scale.cpu()
    assert (host_ids < 0).any() and (host_ids[:, 0] == host_ids[:, 1]).any()
    assert (host_scale < 0).any() and host_scale.unique().numel() > 2
    counts = [(host_ids == e).sum().item() for e in range(E)]
    assert min(counts) > 0 and len(set(counts)) > 1, counts
    before, ptr, version = token.clone(), token.data_ptr(), token._version
    expected = _reference(args)
    assert expected.abs().max().item() > 1e-3, "Wide fixture must reject a zero output"
    output = _single_output(fn(*args))
    torch.cuda.synchronize(x.device)
    assert token.data_ptr() == ptr and token[0].item() == before[0].item(), label
    assert token[1].item() == before[1].item() + 1 and token._version == version + 1, label
    assert output.device == x.device and output.dtype == x.dtype and output.shape == x.shape
    actual = output.float().cpu()
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    # Check both N tiles independently; a missing second tile must never pass.
    for start in (0, 64):
        ref_tile = expected[:, start:start + 64]
        assert ref_tile.norm().item() > 1e-3, label
        relative_l2 = (actual[:, start:start + 64] - ref_tile).norm() / ref_tile.norm()
        assert relative_l2.item() < .02, (label, start, relative_l2.item())
    masked = (host_ids < 0).all(dim=1)
    assert masked.any() and torch.count_nonzero(actual[masked]).item() == 0, label
    stats = _stats(runtime)
    expert_bytes = 6 * w1.shape[2] * w2.shape[2]
    miss_capacity = min(E - len(cache.expert_ids), 2)
    _check_cache_stats(stats, cache, experts=E, expert_bytes=expert_bytes,
                       miss_capacity=miss_capacity)
    assert stats["active_experts"] == E and stats["activation_mode"] == 1, stats
    assert stats["route_tile_rows"] > 0, stats
    if tile_rows != "auto":
        assert stats["route_tile_rows"] <= tile_rows, stats
    # Two-expert miss banks also force clipping across miss expert intervals.
    wave_rows, group_count, previous_hit = [], 0, None
    for expert, count in enumerate(counts):
        hit = expert in cache.expert_ids
        capacity = E if hit else miss_capacity
        if hit != previous_hit or group_count == capacity:
            wave_rows.append(0)
            group_count = 0
        wave_rows[-1] += count
        group_count += 1
        previous_hit = hit
    if tile_rows in (4096, "auto"):
        assert stats["route_tile_rows"] >= max(wave_rows), stats
        assert stats["route_tiles"] == stats["waves"], stats  # Full-wave fast path.
        if tile_rows == "auto":
            # These small waves fit easily: minimize rows without adding chunks,
            # allowing at most one native tile_m alignment (tile_m <= 128).
            assert stats["route_tile_rows"] < max(wave_rows) + 128, stats
    else:
        assert stats["route_tiles"] == sum(
            math.ceil(rows / stats["route_tile_rows"]) for rows in wave_rows), stats
        if max(wave_rows) > stats["route_tile_rows"]:
            assert stats["route_tiles"] > stats["waves"], stats
        else:
            assert stats["route_tiles"] == stats["waves"], stats
    pinned_cap = 4 * (E + 1) + 2 * (2 * expert_bytes + 40)
    assert 0 < stats["pinned_pool_bytes"] <= stats["pinned_bytes"] <= pinned_cap, stats
    assert runtime.store.weight_bytes <= stats["ram_charged_bytes"] <= RAM_BUDGET, stats
    assert stats["buffer_allocations"] >= 0 and stats["buffer_reuses"] >= 0, stats
    if check_reuse:
        assert stats["buffer_allocations"] == 0 and stats["buffer_reuses"] > 0, stats
    assert all(w.device.type == "cpu" and not w.is_pinned() for w in (w1, w2))
    print(f"PASS {label}: max_abs={(actual - expected).abs().max().item()} stats={stats}", flush=True)


def _check_wide_acceptance(directory, device, *, bf16=True, benchmark_only=False):
    # Static H/dtype requires separate official graphs, never reuse the H64 graph.
    # Each package is compiled once and reused across every tile/cache/runtime.
    half_paths = None
    for dtype in ((torch.float16, torch.bfloat16) if bf16 else (torch.float16,)):
        name = "fp16" if dtype == torch.float16 else "bf16"
        h, intermediate = 128, 64
        generator = torch.Generator().manual_seed(20261012)
        w1 = (torch.randn(E, 2 * intermediate, h, generator=generator) * .08).to(dtype)
        w2 = (torch.randn(E, h, intermediate, generator=generator) * .08).to(dtype)
        expert_bytes = 6 * h * intermediate
        store = ExpertStore({0: (w1, w2)}, ram_budget_bytes=RAM_BUDGET)
        archive_path = directory / f"wide_weights_{name}.pt"
        store.save_cpp(archive_path, weight_version=1)
        gpu_cap = 2 * 1024 * 1024
        pinned_cap = 4 * (E + 1) + 2 * (2 * expert_bytes + 40)
        with ExitStack() as stack:
            contexts = []
            for tile_rows in ((4096,) if benchmark_only else (3, 4096, "auto")):
                for preset, expert_ids in CACHE_PRESETS:
                    cap = E * expert_bytes if expert_ids else 0
                    runtime = stack.enter_context(PrefillMoERuntime(
                        store, device=device, gpu_budget_bytes=gpu_cap,
                        ram_budget_bytes=RAM_BUDGET, pinned_budget_bytes=pinned_cap,
                        num_banks=2, activation_mode="fp32_accumulate",
                        max_route_tile_rows=tile_rows, resident_cache_bytes=cap,
                        resident_experts={0: expert_ids}))
                    cache = _get_cache_info(runtime, preset, expert_ids, cap, gpu_cap, expert_bytes)
                    contexts.append((tile_rows, runtime, cache))
            runtime = contexts[0][1]
            args = _wide_inputs(7, w1, w2, runtime.token, device)
            before, version = runtime.token.clone(), runtime.token._version
            dim = torch.export.Dim(f"wide_{name}_S", min=7, max=16384)
            exported = torch.export.export(_TupleLayer().eval(), args, dynamic_shapes=(
                {0: dim}, None, None, {0: dim}, {0: dim}, None))
            assert not exported.state_dict, "Wide weights/token must remain external"
            placeholders = [n for n in exported.graph.nodes if n.op == "placeholder"]
            calls = [n for n in exported.graph.nodes if n.op == "call_function" and
                     n.target == torch.ops.recsys_ops.fuse_moe_offload.default]
            assert len(placeholders) == 6 and len(calls) == 1, exported.graph
            assert calls[0].args[:6] == tuple(placeholders) and calls[0].args[6] == 0
            torch.testing.assert_close(runtime.token, before, atol=0, rtol=0)
            assert runtime.token._version == version, "Wide export mutated token"
            package_path = _compile_checked_package(
                exported, package_path=str(directory / f"wide_{name}.pt2"),
                inductor_configs={"triton.cudagraphs": False})
            torch.testing.assert_close(runtime.token, before, atol=0, rtol=0)
            assert runtime.token._version == version, "Wide compilation mutated token"
            assert isinstance(package_path, str) and Path(package_path).is_file()
            assert directory in Path(package_path).resolve().parents
            with zipfile.ZipFile(package_path) as package:
                shared_objects = [n for n in package.namelist() if n.endswith(".so")]
                assert len(shared_objects) == 1, package.namelist()
                extern = json.loads(package.read(str(Path(shared_objects[0]).with_suffix(".json"))))
                assert "fuse_moe_offload" in json.dumps(extern), extern
            loaded = torch._inductor.aoti_load_package(package_path)
            try:
                for tile_rows, context, cache in (() if benchmark_only else contexts):
                    for s in (7, 64, 7):
                        for repeat in range(2):
                            _check_wide_call(
                                loaded, _wide_inputs(s, w1, w2, context.token, device), context,
                                cache, tile_rows, f"wide Python AOTI {name} H128/I64 tile={tile_rows} "
                                f"cache={cache.name} S={s} repeat={repeat}", check_reuse=repeat == 1)
            finally:
                del loaded
        if dtype == torch.float16:
            half_paths = (package_path, str(archive_path))
    assert half_paths is not None, "FP16 wide package is mandatory for C++ acceptance"
    return half_paths


def _check_checkpoint_acceptance(directory, executable, package_path, loaded, library, device, sizes,
                                 wide_paths=(), async_staging=False, host_profile=False):
    chain = _TwoLayerChain().eval()
    chain_path = None
    chain_loaded = None
    baseline_outputs = {}
    try:
        for weight_version in (1, 2):
            store, archive = _checkpoint_store(directory, weight_version)
            w1, w2 = store.layers[0]
            with ExitStack() as stack:
                contexts = []
                for mode in (0, 1):
                    for name, expert_ids in CACHE_PRESETS:
                        cap = GPU_BUDGET if expert_ids else 0
                        budget = CACHED_GPU_BUDGET if expert_ids else GPU_BUDGET
                        started = time.perf_counter()
                        runtime = stack.enter_context(PrefillMoERuntime(
                            store, device=device, gpu_budget_bytes=budget,
                            ram_budget_bytes=RAM_BUDGET, pinned_budget_bytes=PINNED_BUDGET,
                            num_banks=2, resident_cache_bytes=cap,
                            resident_experts={0: expert_ids},
                            activation_mode="route_buffer" if mode == 0 else "fp32_accumulate",
                            max_route_tile_rows=3 if mode == 1 else 4096))
                        torch.cuda.synchronize(device)
                        print("ACCEPTANCE_STARTUP " + json.dumps(dict(
                            weight_version=weight_version, mode=mode, cache=name,
                            bind_startup_ms=(time.perf_counter() - started) * 1000,
                            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            rss_note="Linux lifetime high-water; not an exact budget")), flush=True)
                        cache = _get_cache_info(runtime, name, expert_ids, cap, budget, EXPERT_BYTES)
                        contexts.append((mode, runtime, cache))
                        for s in (*sizes, sizes[0]):
                            for repeat in range(2):
                                _check_call(loaded, _inputs(s, w1, w2, runtime.token, device), runtime,
                                            f"checkpoint Python AOTI v={weight_version} mode={mode} "
                                            f"cache={name} S={s} repeat={repeat}",
                                            cache=cache, mode=mode, check_reuse=repeat == 1)
                _check_dynamic_contexts(loaded, store, w1, w2, device,
                                        f"checkpoint Python AOTI v={weight_version}")
                # Use disabled contexts only for the real chain: stats describe layer5.
                chain_contexts = [(mode, runtime) for mode, runtime, cache in contexts if not cache.expert_ids]
                if chain_path is None:
                    runtime = chain_contexts[0][1]
                    args = _chain_inputs(7, store, runtime.token, device)
                    before, version = runtime.token.clone(), runtime.token._version
                    dim = torch.export.Dim("real_chain_S", min=2, max=64)
                    exported = torch.export.export(chain, args, dynamic_shapes=(
                        {0: dim}, None, None, None, None, {0: dim}, {0: dim}, None))
                    assert not exported.state_dict, "Real chain weights/token must be external"
                    placeholders = [n for n in exported.graph.nodes if n.op == "placeholder"]
                    calls = [n for n in exported.graph.nodes if n.op == "call_function" and
                             n.target == torch.ops.recsys_ops.fuse_moe_offload.default]
                    assert len(placeholders) == 8 and len(calls) == 2, exported.graph
                    first, second = calls
                    assert first.args[1:3] == tuple(placeholders[1:3]) and first.args[6] == 0
                    assert second.args[0] is first and second.args[1:3] == tuple(placeholders[3:5])
                    assert second.args[6] == 5 and first.args[3:6] == second.args[3:6]
                    torch.testing.assert_close(runtime.token, before, atol=0, rtol=0)
                    assert runtime.token._version == version
                    for mode, context in chain_contexts:
                        for s in sizes:
                            args = _chain_inputs(s, store, context.token, device)
                            _check_real_chain(chain, args, context, f"real chain eager S={s}", mode)
                            _check_real_chain(exported.module(), args, context, f"real chain exported S={s}", mode)
                    before, version = runtime.token.clone(), runtime.token._version
                    chain_path = _compile_checked_package(
                        exported, package_path=str(directory / "real_two_layer_chain.pt2"),
                        inductor_configs={"triton.cudagraphs": False})
                    torch.testing.assert_close(runtime.token, before, atol=0, rtol=0)
                    assert runtime.token._version == version
                    assert Path(chain_path).is_file() and directory in Path(chain_path).resolve().parents
                    with zipfile.ZipFile(chain_path) as package:
                        shared_objects = [name for name in package.namelist() if name.endswith(".so")]
                        assert len(shared_objects) == 1, package.namelist()
                        extern = json.loads(package.read(str(Path(shared_objects[0]).with_suffix(".json"))))
                        assert "fuse_moe_offload" in json.dumps(extern), extern
                    chain_loaded = torch._inductor.aoti_load_package(chain_path)
                for mode, runtime in chain_contexts:
                    for s in (*sizes, sizes[0]):
                        actual = _check_real_chain(chain_loaded, _chain_inputs(s, store, runtime.token, device),
                                                  runtime, f"real chain AOTI v={weight_version} mode={mode} S={s}", mode)
                        key = mode, s
                        if weight_version == 1:
                            baseline_outputs[key] = actual
                        else:
                            assert not torch.allclose(actual, baseline_outputs[key], atol=2e-6, rtol=RTOL), (
                                "Same PT2 package did not use replacement external weights", key)
                command = [str(executable), package_path, str(library), str(device.index),
                           str(archive), chain_path, *(str(s) for s in sizes)]
                if wide_paths and weight_version == 1:
                    command += ["--wide", *wide_paths]
                command += _runtime_cli_flags(async_staging, host_profile)
                environment = os.environ.copy()
                environment.pop("PYTHONPATH", None)
                environment.pop("PYTHONHOME", None)
                print("Standalone run: " + shlex.join(command), flush=True)
                started = time.perf_counter()
                _run_process(command, directory, f"cpp_checkpoint_v{weight_version}", env=environment)
                print("ACCEPTANCE_PROCESS " + json.dumps(dict(
                    weight_version=weight_version, process_wall_ms=(time.perf_counter() - started) * 1000,
                    children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                    rss_note="Linux child lifetime high-water includes prior children/compilation; not exact budget"
                )), flush=True)
    finally:
        del chain_loaded


def _compile_runner(directory):
    from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths

    setup_compiler()
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit is required for offline C++ runner compilation")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    cuda_libs = library_paths(device_type="cuda")
    source = Path(__file__).with_name("fuse_moe_offload_aoti_runner.cpp")
    executable = directory / "fuse_moe_offload_aoti_runner"
    command = compiler + ["-std=c++17", "-O2", "-pthread",
                          f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                          "-I", str(source.parent.parent / "csrc"),
                          str(source), "-o", str(executable)]
    for path in include_paths(device_type="cuda"):
        command += ["-I", path]
    for path in dict.fromkeys([str(torch_lib)] + cuda_libs):
        command += ["-L", path, f"-Wl,-rpath,{path}"]
    # RunnerCuda and its proxy executor live in libtorch_cuda / libtorch_cpu.
    command += ["-Wl,--no-as-needed", "-ltorch_cuda", "-ltorch_cpu", "-ltorch",
                "-lc10_cuda", "-lc10", "-lcudart", "-ldl"]
    # The custom group has a pybind entry point. Resolve symbols only; never
    # initialize Python or link torch_python into the standalone executable.
    python_libdir = sysconfig.get_config_var("LIBDIR")
    python_library = sysconfig.get_config_var("LDLIBRARY")
    if python_libdir and python_library:
        python_so = Path(python_libdir) / python_library
        if python_so.is_file():
            command += [str(python_so), f"-Wl,-rpath,{python_libdir}"]
    (directory / "fuse_moe_offload_aoti_runner.cpp").write_bytes(source.read_bytes())
    _write_json(directory / "cpp_source.json", dict(
        source=str(source.resolve()), source_sha256=_sha256(source)))
    print("C++ compile: " + shlex.join(command), flush=True)
    _run_process(command, directory, "cpp_compile")
    return executable


def _runtime_cli_flags(async_staging, host_profile):
    return (["--async-staging"] if async_staging else []) + (
        ["--host-profile"] if host_profile else [])


def run_acceptance(*, ops_library=None, sizes=(2, 7, 64), wait_seconds=0, wide_bf16=True,
                   async_staging=False, host_profile=False, allow_testing=False,
                   output_dir=None, public_only=False):
    # All legacy multi-config graphs retain external tokens and weights. Apply
    # explicit create-time flags to every helper's runtime without changing graphs.
    original_runtime = PrefillMoERuntime
    with mock.patch(__name__ + ".PrefillMoERuntime", partial(
            original_runtime, async_staging=async_staging, host_profile=host_profile)):
        return _run_acceptance(
            ops_library=ops_library, sizes=sizes, wait_seconds=wait_seconds,
            wide_bf16=wide_bf16, async_staging=async_staging, host_profile=host_profile,
            allow_testing=allow_testing, output_dir=output_dir, public_only=public_only)


def _run_acceptance(*, ops_library, sizes, wait_seconds, wide_bf16,
                    async_staging, host_profile, allow_testing=False, output_dir=None,
                    public_only=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this acceptance test cannot be skipped")
    if not {2, 7, 64}.issubset(sizes) or any(s < 2 or s > 64 for s in sizes):
        raise ValueError("sizes must include S=2, S=7 and S=64 and stay in [2,64]")
    library = _load_ops(ops_library, wait_seconds, allow_testing=allow_testing)
    w1, w2 = _weights()
    store = ExpertStore({0: (w1, w2)}, ram_budget_bytes=RAM_BUDGET)
    device = torch.device("cuda", torch.cuda.current_device())
    with _artifact_directory("fuse_moe_offload_aoti_", output_dir) as temporary, \
            mock.patch.dict(os.environ, {
                "TORCHINDUCTOR_CACHE_DIR": str(Path(temporary) / "inductor_cache"),
                "TRITON_CACHE_DIR": str(Path(temporary) / "triton_cache"),
            }):
        directory = Path(temporary)
        # Compile the independent executable even if export subsequently fails.
        executable = _compile_runner(directory)
        try:
            _check_public_acceptance(directory, executable, library, device, sizes,
                                     async_staging=async_staging, host_profile=host_profile,
                                     allow_testing=allow_testing)
        finally:
            torch._dynamo.reset()
            gc.collect()
        if public_only:
            return
        with ExitStack() as stack:
            # Release Python compiler/loader cycles before the temporary tree exits.
            stack.callback(gc.collect)
            stack.callback(torch._dynamo.reset)
            budgets = dict(device=device, gpu_budget_bytes=GPU_BUDGET,
                           ram_budget_bytes=RAM_BUDGET, pinned_budget_bytes=PINNED_BUDGET,
                           num_banks=2)
            contexts = []
            for mode in (0, 1):
                for name, expert_ids in CACHE_PRESETS:
                    options = dict(budgets)
                    cache_cap = GPU_BUDGET if expert_ids else 0
                    if expert_ids:
                        # Two live chain outputs plus the resident cache need headroom.
                        options["gpu_budget_bytes"] = CACHED_GPU_BUDGET
                        options.update(resident_cache_bytes=cache_cap,
                                       resident_experts={0: expert_ids})
                    # Disabled configuration deliberately exercises frontend defaults.
                    context = stack.enter_context(PrefillMoERuntime(
                        store, **options,
                        activation_mode="route_buffer" if mode == 0 else "fp32_accumulate",
                        max_route_tile_rows=3 if mode == 1 else 4096))
                    cache = _get_cache_info(context, name, expert_ids, cache_cap,
                                            options["gpu_budget_bytes"], EXPERT_BYTES)
                    contexts.append((mode, context, cache))
            _, runtime, _ = contexts[0]
            layer = _TupleLayer().eval()
            for mode, context, cache in contexts:
                for s in sizes:
                    for repeat in range(2):
                        _check_call(layer, _inputs(s, w1, w2, context.token, device),
                                    context, f"eager mode={mode} cache={cache.name} S={s} repeat={repeat}",
                                    mode=mode, cache=cache, check_reuse=repeat == 1)
            _check_dynamic_contexts(layer, store, w1, w2, device, "eager")
            args = _inputs(7, w1, w2, runtime.token, device)
            snapshot = runtime.token.clone()
            snapshot_version = runtime.token._version
            dim = torch.export.Dim("S", min=2, max=64)
            exported = torch.export.export(
                layer, args, dynamic_shapes=({0: dim}, None, None, {0: dim}, {0: dim}, None))
            torch.testing.assert_close(runtime.token, snapshot, atol=0, rtol=0)
            assert runtime.token._version == snapshot_version, "Export mutated token version"
            assert not exported.state_dict, "Weights/token must remain external graph inputs"
            offload_op = torch.ops.recsys_ops.fuse_moe_offload.default
            # Export may retain the mutable op directly or wrap it in the real
            # auto_functionalized higher-order op. Both must be tested as emitted.
            assert any(node.target == offload_op or
                       any(value == offload_op for value in
                           pytree.tree_leaves((node.args, node.kwargs))
                           if isinstance(value, torch._ops.OpOverload))
                       for node in exported.graph.nodes), exported.graph
            print(f"PASS export: {exported.graph_signature}\n{exported.graph}", flush=True)
            # Mode/tile are deployment-context configuration, never graph inputs.
            for mode, context, cache in contexts:
                for s in sizes:
                    for repeat in range(2):
                        _check_call(exported.module(), _inputs(s, w1, w2, context.token, device),
                                    context, f"exported mode={mode} cache={cache.name} S={s} repeat={repeat}",
                                    mode=mode, cache=cache, check_reuse=repeat == 1)
            _check_dynamic_contexts(exported.module(), store, w1, w2, device, "exported")
            # Functionalization/reinplacing must preserve the deployment token's
            # storage identity. No fake functional kernel or token clone workaround.
            # The official package API calls aot_compile(package=True) and
            # includes the compiler-generated extern-kernel JSON. Do not override
            # output_path: that changes the .so basename but not the JSON basename.
            compile_snapshot = runtime.token.clone()
            compile_version = runtime.token._version
            package_path = _compile_checked_package(
                exported, package_path=str(directory / "offload.pt2"),
                inductor_configs={"triton.cudagraphs": False})
            torch.testing.assert_close(runtime.token, compile_snapshot, atol=0, rtol=0)
            assert runtime.token._version == compile_version, "Compile mutated token version"
            if not isinstance(package_path, str) or not Path(package_path).is_file():
                raise AssertionError(f"Expected compiled PT2 package, got {package_path}")
            if directory not in Path(package_path).resolve().parents:
                raise AssertionError("AOTI artifacts escaped TemporaryDirectory")
            # Inspect, never synthesize/rename, the serialized extern kernel JSON.
            with zipfile.ZipFile(package_path) as archive:
                so_names = [name for name in archive.namelist() if name.endswith(".so")]
                assert len(so_names) == 1, archive.namelist()
                json_name = str(Path(so_names[0]).with_suffix(".json"))
                extern_kernels = json.loads(archive.read(json_name))
                assert "fuse_moe_offload" in json.dumps(extern_kernels), extern_kernels
                print(f"PASS official PT2 package: {so_names[0]}, {json_name}", flush=True)
            loaded = torch._inductor.aoti_load_package(package_path)
            try:
                # Load/compile once; only substitute each context's external token.
                for mode, context, cache in contexts:
                    for s in (*sizes, sizes[0]):
                        for repeat in range(2):
                            _check_call(loaded, _inputs(s, w1, w2, context.token, device),
                                        context, f"Python AOTI mode={mode} cache={cache.name} S={s} repeat={repeat}",
                                        mode=mode, cache=cache, check_reuse=repeat == 1)
                _check_dynamic_contexts(loaded, store, w1, w2, device, "Python AOTI")
                wide_paths = _check_wide_acceptance(directory, device, bf16=wide_bf16)
                _check_checkpoint_acceptance(directory, executable, package_path, loaded,
                                             library, device, sizes, wide_paths=wide_paths,
                                             async_staging=async_staging, host_profile=host_profile)
            finally:
                del loaded
            # Keep all six contexts live for the same two-call graph/package, with
            # no token clone/rebind and exactly two value/version mutations.
            _check_shared_token_chain(directory, contexts, w1, w2, device, sizes)
        # Independent locked artifacts, after every legacy multi-config test.
        _check_locked_acceptance(directory, executable, library, device, sizes,
                                 async_staging=async_staging, host_profile=host_profile)
    print("PASS export + compile + Python AOTI + standalone C++ RAM offload + shared-token chain", flush=True)


def _locked_bindings(chain):
    from custom_ops.moe_offload import InputBinding

    return [InputBinding(7 if chain else 5, "token"),
            InputBinding(1, "w1", 0), InputBinding(2, "w2", 0),
            *([InputBinding(3, "w1", 5), InputBinding(4, "w2", 5)] if chain else [])]


class _LockedSingle(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = OffloadedMoELayer(0, config=config)

    def forward(self, x, w1, w2, ids, scale, token):
        return (self.layer(x, w1, w2, ids, scale, token),)


class _LockedChain(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.first = OffloadedMoELayer(0, config=config)
        self.second = OffloadedMoELayer(5, config=config)

    def forward(self, x, w10, w20, w15, w25, ids, scale, token):
        intermediate = self.first(x, w10, w20, ids, scale, token)
        return (self.second(intermediate, w15, w25, ids, scale, token),)


def _serialized_int_vectors(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "as_ints":
                yield tuple(child)
            else:
                yield from _serialized_int_vectors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _serialized_int_vectors(child)


def _locked_store(checkpoint, chain):
    layers = {layer: tuple(w.clone() for w in checkpoint.layers[layer])
              for layer in ((0, 5) if chain else (0,))}
    if chain:
        # Routes 0/1 alone have smaller outputs than the legacy all-expert routes.
        # Scale only an independent LOCKED clone, before config validation/archive.
        layers[5][1].mul_(4)
    return ExpertStore(layers, ram_budget_bytes=RAM_BUDGET)


def _check_locked_chain_fixture_cpu(store):
    token = torch.zeros(2, dtype=torch.int64)
    maxima = {}
    for s in (2, 7, 64):
        args = _locked_inputs(s, store, token, torch.device("cpu"), True)
        expected = args[0]
        for layer in (0, 5):
            expected = _reference((expected, *store.layers[layer], args[-3], args[-2], token)).half()
        maximum = expected.abs().max().item()
        assert torch.isfinite(expected).all().item() and maximum > 1e-4, (
            f"LOCKED chain CPU fixture S={s}: max abs={maximum:.9g} must exceed 1e-4; "
            "check routes 0/1, seed and locked-only layer5 down-weight scaling")
        maxima[s] = maximum
    return maxima


def _locked_inputs(s, store, token, device, chain):
    args = (_chain_inputs(s, store, token, device) if chain else
            _inputs(s, *store.layers[0], token, device))
    # One static expert and one dynamic expert; repeated routes count once/call.
    ids = torch.tensor([0, 1], dtype=torch.int32, device=device).expand(s, K).contiguous()
    return (*args[:-3], ids, args[-2], token)


def _check_locked_call(loaded, args, runtime, config, chain, label):
    token = runtime.token
    before, version, ptr = token.clone(), token._version, token.data_ptr()
    x, ids, scale = args[0], args[-3], args[-2]
    expected = x
    for layer in ((0, 5) if chain else (0,)):
        expected = _reference((expected, *runtime.store.layers[layer], ids, scale, token)).half()
    # Graph guards are defense-in-depth only; checked lowering is proved at
    # compile time, and artifact preflight remains mandatory on EVERY call.
    output = _single_output(runtime.invoke(
        loaded, args, artifact_config=config, bindings=_locked_bindings(chain)))
    torch.cuda.synchronize(runtime.device)
    invocations = 2 if chain else 1
    assert token.data_ptr() == ptr and token[0].item() == before[0].item(), label
    assert token[1].item() == before[1].item() + invocations, label
    assert token._version == version + invocations, label
    assert output.shape == x.shape and output.dtype == x.dtype and output.device == x.device, label
    actual = output.float().cpu()
    torch.testing.assert_close(actual, expected.float(), atol=2e-6 if chain else ATOL, rtol=RTOL)
    assert expected.abs().max().item() > (1e-4 if chain else 1e-3), label
    assert ((actual - expected.float()).norm() / expected.float().norm()).item() < .02, label
    stats = _stats(runtime)
    assert stats["active_experts"] == 2 and stats["cache_hits"] >= 1, stats
    assert stats["cache_misses"] + stats["cache_hits"] == 2, stats
    assert stats["weight_h2d_bytes"] == stats["cache_misses"] * EXPERT_BYTES, stats
    assert stats["resident_experts"] == len(config.layer_specs), stats
    assert stats["resident_load_h2d_bytes"] == len(config.layer_specs) * EXPERT_BYTES, stats
    assert stats["cache_promotion_d2d_bytes"] == stats["cache_admissions"] * EXPERT_BYTES, stats
    assert stats["gpu_pool_bytes"] + stats["resident_cache_bytes"] + stats["dynamic_cache_bytes"] <= stats["gpu_charged_bytes"] <= config.gpu_budget_bytes, stats
    assert stats["pinned_bytes"] <= config.pinned_budget_bytes, stats
    assert runtime.store.weight_bytes <= stats["ram_charged_bytes"] <= config.ram_budget_bytes, stats
    assert stats["activation_mode"] == 1 and stats["route_tile_rows"] > 0, stats
    print(f"PASS LOCKED {label}: token={token.tolist()} stats={stats}", flush=True)
    return stats


def _check_locked_rejection(loaded, args, runtime, config, label):
    token = runtime.token
    snapshot, version, ptr, stats = token.clone(), token._version, token.data_ptr(), _stats(runtime)
    model_call = mock.Mock(wraps=loaded)
    try:
        # Validate the frozen ARTIFACT config, never runtime.deployment_config.
        runtime.invoke(model_call, args, artifact_config=config,
                       bindings=_locked_bindings(len(config.layer_specs) == 2))
    except (RuntimeError, ValueError) as error:
        assert any(reason in str(error).lower() for reason in
                   ("configuration mismatch", "identity", "version changed", "shape")), (label, error)
    else:
        raise AssertionError(f"LOCKED {label}: incompatible runtime silently accepted")
    model_call.assert_not_called()  # No claim about compiled error propagation.
    assert torch.equal(token, snapshot) and token._version == version and token.data_ptr() == ptr, label
    assert _stats(runtime) == stats, label
    print(f"PASS LOCKED preflight reject-before-mutation {label}", flush=True)


def _check_locked_acceptance(directory, executable, library, device, sizes, *,
                             async_staging, host_profile):
    # Detect the target offline once; all runtime construction uses positive caps.
    major, minor = torch.cuda.get_device_capability(device)
    checkpoint, _ = _checkpoint_store(directory, 17)
    for chain in (False, True):
        kind = "chain" if chain else "single"
        store = _locked_store(checkpoint, chain)
        if chain:
            maxima = _check_locked_chain_fixture_cpu(store)
            print(f"PASS LOCKED chain CPU fixture before config/archive: {maxima}", flush=True)
        config = OffloadConfig.from_store(
            store, gpu_budget_bytes=2 * 1024**2, ram_budget_bytes=RAM_BUDGET,
            pinned_budget_bytes=PINNED_BUDGET, target_sm=10 * major + minor,
            resident_experts={layer: (0,) for layer in store.layers},
            resident_cache_bytes=2 * DYNAMIC_CACHE_CAP, dynamic_cache_bytes=2 * DYNAMIC_CACHE_CAP,
            activation_mode="fp32_accumulate", max_route_tile_rows="auto",
            tile_m=32, tile_n=128 if chain else 64,
            async_staging=async_staging, host_profile=host_profile)
        archive_path = directory / f"locked_{kind}_weights.pt"
        store.save_cpp(archive_path, weight_version=17, config=config)
        restored = ExpertStore.load_cpp(archive_path, ram_budget_bytes=RAM_BUDGET)
        assert restored.deployment_config == config
        store = restored
        with _LockedPrefillMoERuntime.from_config(store, config, device=device) as runtime:
            args = _locked_inputs(7, store, runtime.token, device, chain)
            dim = torch.export.Dim(f"locked_{kind}_S", min=2, max=64)
            shapes = ({0: dim}, *(None for _ in range(4 if chain else 2)), {0: dim}, {0: dim}, None)
            snapshot, version = runtime.token.clone(), runtime.token._version
            model = (_LockedChain(config) if chain else _LockedSingle(config)).eval()
            exported = torch.export.export(model, args, dynamic_shapes=shapes)
            assert not exported.state_dict
            placeholders = [n for n in exported.graph.nodes if n.op == "placeholder"]
            calls = [n for n in exported.graph.nodes if n.op == "call_function" and
                     n.target == torch.ops.recsys_ops.fuse_moe_offload.default]
            assert len(placeholders) == (8 if chain else 6) and len(calls) == (2 if chain else 1), exported.graph
            vectors = [config.layer_config(layer) for layer in store.layers]
            assert [tuple(n.args[7]) for n in calls] == vectors, exported.graph
            manifest = config.manifest()
            assert all(tuple(n.args[8]) == manifest for n in calls), exported.graph
            assert all(all(type(v) is int for v in (*n.args[7], *n.args[8])) for n in calls)
            assert all(n.args[5] is placeholders[-1] for n in calls)
            if chain:
                assert calls[1].args[0] is calls[0]
                assert calls[0].args[1:3] == tuple(placeholders[1:3])
                assert calls[1].args[1:3] == tuple(placeholders[3:5])
            package = _compile_checked_package(
                exported, package_path=str(directory / f"locked_{kind}.pt2"),
                inductor_configs={"triton.cudagraphs": False})
            assert torch.equal(runtime.token, snapshot) and runtime.token._version == version
            with zipfile.ZipFile(package) as artifact:
                objects = [n for n in artifact.namelist() if n.endswith(".so")]
                assert len(objects) == 1
                extern = json.loads(artifact.read(str(Path(objects[0]).with_suffix(".json"))))
                serialized = list(_serialized_int_vectors(extern))
                assert all(serialized.count(vector) == vectors.count(vector) for vector in vectors), extern
                assert serialized.count(manifest) == len(calls), extern
            print(f"PASS LOCKED {kind} baked int[] graph + official extern JSON", flush=True)
            loaded = torch._inductor.aoti_load_package(package)
            try:
                for observation in range(3):
                    stats = _check_locked_call(loaded, args, runtime, config, chain,
                                               f"{kind} observation={observation + 1}")
                    assert stats["dynamic_cache_hits"] == int(observation == 2), stats
                    assert stats["cache_admissions"] == int(observation == 1), stats
                for s in (*sizes, sizes[0]):
                    stats = _check_locked_call(loaded, _locked_inputs(s, store, runtime.token, device, chain),
                                               runtime, config, chain, f"{kind} S={s} warm")
                    assert stats["cache_hits"] == 2 and stats["weight_h2d_bytes"] == 0, stats
                variants = {
                    "tile_m": replace(config, tile_m=64),
                    "tile_n": replace(config, tile_n=64 if chain else 128),
                    "gpu_budget": replace(config, gpu_budget_bytes=config.gpu_budget_bytes + 512),
                    "async": replace(config, async_staging=not config.async_staging),
                    "host_profile": replace(config, host_profile=not config.host_profile),
                    "static_resident": replace(config, layer_specs=tuple(
                        (*spec[:5], (2,)) for spec in config.layer_specs)),
                }
                if chain:
                    variants["layer5_only_static_resident"] = replace(config, layer_specs=tuple(
                        (*spec[:5], (1,)) if spec[0] == 5 else spec for spec in config.layer_specs))
                    assert variants["layer5_only_static_resident"].layer_config(0) == config.layer_config(0)
                for name, wrong in variants.items():
                    with _LockedPrefillMoERuntime.from_config(store, wrong, device=device) as incompatible:
                        _check_locked_rejection(loaded, _locked_inputs(7, store, incompatible.token, device, chain),
                                                incompatible, config, f"{kind}/{name}")
                wrong_store = ExpertStore({layer: (w1[:3].clone(), w2[:3].clone())
                                           for layer, (w1, w2) in store.layers.items()},
                                          ram_budget_bytes=RAM_BUDGET)
                wrong_weights = replace(config, layer_specs=tuple(
                    (spec[0], 3, *spec[2:]) for spec in config.layer_specs))
                with _LockedPrefillMoERuntime.from_config(wrong_store, wrong_weights, device=device) as incompatible:
                    # Keep artifact inputs at compiled shapes; bound runtime weights
                    # differ, so artifact preflight must reject before model execution.
                    _check_locked_rejection(loaded, _locked_inputs(7, store, incompatible.token, device, chain),
                                            incompatible, config, f"{kind}/bound_weight_config")
                if chain:
                    isolated = ExpertStore({layer: tuple(w.clone() for w in pair)
                                            for layer, pair in store.layers.items()},
                                           ram_budget_bytes=RAM_BUDGET)
                    with _LockedPrefillMoERuntime.from_config(isolated, config, device=device) as incompatible:
                        versions = {layer: tuple(w._version for w in pair)
                                    for layer, pair in isolated.layers.items()}
                        # Same bytes/shapes/identity, only layer5's version changes.
                        isolated.layers[5][0].add_(0)
                        assert tuple(w._version for w in isolated.layers[0]) == versions[0]
                        assert isolated.layers[5][0]._version == versions[5][0] + 1
                        _check_locked_rejection(
                            loaded, _locked_inputs(7, isolated, incompatible.token, device, chain),
                            incompatible, config, "chain/layer5_only_weight_version_before_first_layer")
            finally:
                del loaded
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        command = [str(executable), "--locked-chain" if chain else "--locked", str(package),
                   str(library), str(device.index), str(archive_path)]
        print("Standalone LOCKED run: " + shlex.join(command), flush=True)
        _run_process(command, directory, f"cpp_locked_{kind}", env=environment)
        # Corrupt actual format2 metadata, not a replacement loader or fixture schema.
        for name, index, value in (("version", 0, 99), ("target_sm", 22, config.target_sm + 1),
                                   ("weight_shape", 18, 128), ("resident_id", 23, E),
                                   ("count_length", 21, 2), ("zero_budget", 1, 0)):
            archive = torch.jit.load(str(archive_path), map_location="cpu")
            getattr(archive, "offload_config_0")[index] = value
            invalid = directory / f"locked_{kind}_invalid_{name}.pt"
            torch.jit.save(archive, str(invalid))
            rejected = _run_process([*command[:-1], str(invalid)], directory,
                                    f"cpp_locked_{kind}_invalid_{name}", env=environment, check=False)
            assert rejected.returncode != 0 and "LOCKED archive" in rejected.stderr, rejected
            print(f"PASS LOCKED C++ invalid archive {kind}/{name}", flush=True)


# Public-op acceptance only: these are NOT model loaders. Every static H/I/E
# inventory gets its own official graph/package; only S is dynamic within it.
_PUBLIC_SPECS = ((0, 3, 64, 64), (5, 5, 128, 128), (11, 2, 64, 192))
_STATUS_FIELDS = ("version", "state", "managed", "pending_error", "forward_sequence",
                  "success_sequence", "failure_layer", "failure_sequence",
                  "managed_sequence", "success_layer")


class _PublicAdapter(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            OffloadedMoELayer(layer, config=config) for layer, *_ in config.layer_specs)
        self.widths = tuple(spec[2] for spec in config.layer_specs)

    def evaluate(self, business, token, x, flat):
        value = x * .875 + .03125
        for n, layer in enumerate(self.layers):
            width = self.widths[n]
            if value.shape[1] < width:
                value = torch.cat((value, value), dim=1)
            elif value.shape[1] > width:
                value = value[:, :width].contiguous()
            ids, w1, scale, w2 = flat[4 * n:4 * n + 4]
            value = layer(value, w1, w2, ids, scale, token) * .75
        return (value + business * .125,)


class _PublicSingle(_PublicAdapter):
    def forward(self, business, token, x, ids0, w10, scale0, w20):
        return self.evaluate(business, token, x, (ids0, w10, scale0, w20))


class _PublicTwo(_PublicAdapter):
    def forward(self, business, token, x, ids0, w10, scale0, w20, ids5, w15, scale5, w25):
        return self.evaluate(business, token, x, (ids0, w10, scale0, w20, ids5, w15, scale5, w25))


class _PublicThree(_PublicAdapter):
    def forward(self, business, token, x, ids0, w10, scale0, w20, ids5, w15, scale5, w25,
                ids11, w111, scale11, w211):
        return self.evaluate(business, token, x, (ids0, w10, scale0, w20, ids5, w15, scale5, w25,
                                                ids11, w111, scale11, w211))


def _public_bindings(count):
    from custom_ops.moe_offload import InputBinding

    return [InputBinding(1, "token"), *(binding for n, spec in enumerate(_PUBLIC_SPECS[:count])
            for binding in (InputBinding(4 + 4 * n, "w1", spec[0]),
                            InputBinding(6 + 4 * n, "w2", spec[0])))]


def _public_inputs(s, store, device):
    first_h = next(iter(store.layers.values()))[0].shape[2]
    business = (torch.arange(s, dtype=torch.float32).sin() * .1).reshape(s, 1)
    x = (torch.arange(s * first_h, dtype=torch.float32).cos() * .25).reshape(s, first_h)
    # Undefined designated slots deliberately occur in non-tail positions.
    inputs = [business.to(device=device, dtype=torch.float16), None,
              x.to(device=device, dtype=torch.float16)]
    for layer, (w1, _) in store.layers.items():
        ids = (torch.arange(s * K).reshape(s, K) + layer).remainder(w1.shape[0]).int()
        ids[::7, 1] = -1
        scale = torch.tensor([.75, -.25], dtype=torch.float32).expand(s, K).contiguous()
        inputs.extend((ids.to(device), None, scale.to(device), None))
    return inputs


def _public_reference(inputs, store, *, rounded=True):
    # Inductor fuses adjacent scalar arithmetic in FP32 and casts at the
    # custom-op input/output boundaries, not after every eager scalar op.
    value = (inputs[2].cpu().float() * .875 + .03125).half()
    for n, (_, (w1, w2)) in enumerate(store.layers.items()):
        h = w1.shape[2]
        value = (torch.cat((value, value), dim=1) if value.shape[1] < h else
                 value[:, :h].contiguous())
        ids, scale = inputs[3 + 4 * n].cpu(), inputs[5 + 4 * n].cpu()
        expected = _public_layer_reference(value, w1, w2, ids, scale, rounded=rounded)
        # The native output is FP16. Keep its scalar multiply in FP32 until
        # the next custom op, or the final fused business add, consumes it.
        value = expected.half().float() * .75
        if n + 1 < len(store.layers):
            value = value.half()
    return (value + inputs[0].cpu().float() * .125).half()


def _public_layer_reference(value, w1, w2, ids, scale, *, rounded):
    """FP32 math, optionally with the native Act and Down storage boundaries.

    Gate/up accumulators stay FP32 in the fused epilogue. Act is stored in
    the input dtype; Down is also stored in that dtype *before* route scale
    and FP32 reduction, in both route_buffer and fp32_accumulate modes.
    """
    x = value.cpu().float()
    expected = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids.cpu() == expert)
        gate, up = (x[rows] @ w1[expert].float().T).chunk(2, dim=-1)
        act = torch.nn.functional.silu(gate) * up
        if rounded:
            act = act.to(value.dtype).float()
        down = act @ w2[expert].float().T
        if rounded:
            down = down.to(value.dtype).float()
        expected.index_add_(0, rows, down * scale.cpu()[rows, slots, None])
    return expected


def _public_diagnostic_metric(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    error = actual - expected
    norm = expected.norm().item()
    return dict(max_abs_error=error.abs().max().item(), rmse=error.square().mean().sqrt().item(),
                error_norm=error.norm().item(), reference_norm=norm,
                relative_l2=error.norm().item() / norm if norm else None,
                reference_absmax=expected.abs().max().item(),
                exact_equal=torch.equal(actual, expected))


def _run_public_diagnose(options):
    """Read-only, small real GPU replay; no export, compilation or fault hooks."""
    library = _load_ops(options.ops_library, options.wait_seconds,
                        allow_testing=options.allow_testing, compile_lowering=False)
    store = ExpertStore.load_cpp(options.public_archive, ram_budget_bytes=16 * 1024**2)
    config = store.deployment_config
    saved = torch.load(options.public_diagnose, map_location="cpu", weights_only=True)
    bindings = _public_bindings(len(store.layers))
    bound_indices = {b.index for b in bindings}
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = [None if n in bound_indices else value.to(device)
              for n, value in enumerate(saved["inputs"])]
    loader = torch._inductor.aoti_load_package(options.public_package)
    captured, evidence = [], {}

    def native_chain(*bound, eager_scalar=False):
        value = (bound[2] * .875 + .03125 if eager_scalar else
                 (bound[2].float() * .875 + .03125).to(bound[2].dtype))
        for n, (layer, (w1, _)) in enumerate(store.layers.items()):
            h = w1.shape[2]
            value = (torch.cat((value, value), dim=1) if value.shape[1] < h else
                     value[:, :h].contiguous())
            out = torch.ops.recsys_ops.fuse_moe_offload(
                value, bound[4 + 4 * n], bound[6 + 4 * n], bound[3 + 4 * n],
                bound[5 + 4 * n], bound[1], layer, config.layer_config(layer), config.manifest())
            if not eager_scalar:
                captured.append((value.detach().cpu(), out.detach().cpu()))
            value = out * .75 if eager_scalar else out.float() * .75
            if n + 1 < len(store.layers):
                value = value.to(bound[2].dtype)
        return ((value + bound[0] * .125) if eager_scalar else
                (value + bound[0].float() * .125).to(bound[2].dtype),)

    with _LockedPrefillMoERuntime.from_config(store, config, device=device) as runtime:
        aoti = _single_output(runtime.invoke(loader, inputs, artifact_config=config, bindings=bindings)).cpu()
        native = _single_output(runtime.invoke(native_chain, inputs, artifact_config=config, bindings=bindings)).cpu()
        eager = _single_output(runtime.invoke(partial(native_chain, eager_scalar=True), inputs,
                                              artifact_config=config, bindings=bindings)).cpu()
        status = _public_status(runtime)
    math_value = (saved["inputs"][2].float() * .875 + .03125).half()
    typed_value = math_value.clone()
    pure_value = saved["inputs"][2].float() * .875 + .03125
    layers = []
    for n, (layer, (w1, w2)) in enumerate(store.layers.items()):
        h = w1.shape[2]
        def resize(value):
            return torch.cat((value, value), dim=1) if value.shape[1] < h else value[:, :h].contiguous()
        math_value, typed_value, pure_value = map(resize, (math_value, typed_value, pure_value))
        ids, scale = saved["inputs"][3 + 4 * n], saved["inputs"][5 + 4 * n]
        actual_input, actual_output = captured[n]
        local_math = _public_layer_reference(actual_input, w1, w2, ids, scale, rounded=False)
        local_typed = _public_layer_reference(actual_input, w1, w2, ids, scale, rounded=True)
        chain_math = _public_layer_reference(math_value, w1, w2, ids, scale, rounded=False).half()
        chain_typed = _public_layer_reference(typed_value, w1, w2, ids, scale, rounded=True).half()
        pure_value = _public_layer_reference(pure_value, w1, w2, ids, scale, rounded=False) * .75
        legacy = ops.fuse_moe(actual_input.to(device), w1.to(device), w2.to(device),
                              ids.to(device), scale.to(device)).cpu()
        with PrefillMoERuntime(store, device=device, gpu_budget_bytes=config.gpu_budget_bytes,
                              ram_budget_bytes=config.ram_budget_bytes,
                              pinned_budget_bytes=config.pinned_budget_bytes,
                              activation_mode="route_buffer", max_route_tile_rows=config.max_route_tile_rows,
                              tile_m=config.tile_m, tile_n=config.tile_n) as route:
            route_output = route.forward(actual_input.to(device), ids.to(device), scale.to(device),
                                         layer_id=layer).cpu()
        layers.append(dict(layer=layer, input_shape=list(actual_input.shape),
                           native_vs_local_fp32=_public_diagnostic_metric(actual_output, local_math),
                           native_vs_local_rounded=_public_diagnostic_metric(actual_output, local_typed.half()),
                           native_vs_chain_fp32=_public_diagnostic_metric(actual_output, chain_math),
                           native_vs_chain_rounded=_public_diagnostic_metric(actual_output, chain_typed),
                           native_vs_legacy=_public_diagnostic_metric(actual_output, legacy),
                           native_vs_route_buffer=_public_diagnostic_metric(actual_output, route_output)))
        evidence[str(layer)] = dict(input=actual_input, native=actual_output, local_fp32=local_math,
                                    local_rounded=local_typed, chain_fp32=chain_math,
                                    chain_rounded=chain_typed, legacy=legacy, route_buffer=route_output)
        math_value, typed_value = chain_math.float() * .75, chain_typed.float() * .75
        if n + 1 < len(store.layers):
            math_value, typed_value = math_value.half(), typed_value.half()
    business = saved["inputs"][0].float() * .125
    mathematical, rounded = (math_value + business).half(), (typed_value + business).half()
    contribution = mathematical.float() - business.half().float()
    rounded_contribution = rounded.float() - business.half().float()
    report = dict(diagnostic_only=True, library=str(library), library_sha256=_sha256(library),
                  package_sha256=_sha256(options.public_package), archive_sha256=_sha256(options.public_archive),
                  saved_sha256=_sha256(options.public_diagnose), config=asdict(config), layers=layers,
                  status=status, environment_signature=_environment_signature(library),
                  aoti_vs_native=_public_diagnostic_metric(aoti, native),
                  aoti_vs_eager_scalar=_public_diagnostic_metric(aoti, eager),
                  saved_vs_fresh_aoti=_public_diagnostic_metric(saved["output"], aoti),
                  aoti_vs_old_reference=_public_diagnostic_metric(aoti, mathematical),
                  aoti_vs_rounded_reference=_public_diagnostic_metric(aoti, rounded),
                  aoti_vs_pure_fp32=_public_diagnostic_metric(aoti, pure_value + business),
                  old_gate_relative_l2=(aoti.float()-mathematical.float()).norm().item()/contribution.norm().item(),
                  rounded_gate_relative_l2=(aoti.float()-rounded.float()).norm().item()/rounded_contribution.norm().item())
    directory = Path(options.output_dir)
    _write_json(directory / "public_diagnostic.json", report)
    torch.save(dict(layers=evidence, aoti=aoti, native=native, eager_scalar=eager,
                    mathematical=mathematical, rounded=rounded, pure_fp32=pure_value + business),
               directory / "public_diagnostic.tensors.pt")
    print("PUBLIC_DIAGNOSTIC " + json.dumps(report, sort_keys=True), flush=True)
    # No new numeric acceptance policy here: expose every difference for review.
    assert torch.equal(saved["output"], aoti), "saved/fresh AOTI mismatch"
    assert torch.equal(aoti, native), "AOTI differs from native at declared scalar boundaries"
    assert all(layer["native_vs_legacy"]["exact_equal"] and
               layer["native_vs_route_buffer"]["exact_equal"] for layer in layers), (
                   "native differs from legacy/route_buffer", layers)


def _public_status(runtime):
    status = runtime.status()
    assert set(status) == set(_STATUS_FIELDS), status
    assert status["version"] == 1 and status["state"] in ("healthy", "recoverable", "poisoned"), status
    raw = torch.ops.recsys_ops.fuse_moe_offload_status(runtime.token)
    assert raw.device.type == "cpu" and raw.dtype == torch.int64 and raw.shape == (10,), raw
    values = raw.tolist()
    assert values[0] == 1 and status["state"] == ("healthy", "recoverable", "poisoned")[values[1]]
    for n, field in enumerate(_STATUS_FIELDS):
        if field != "state":
            assert status[field] == values[n], (field, status, values)
    return status


def _public_check(loaded, inputs, runtime, config, bindings, directory, label):
    before = _public_status(runtime)
    ptr, version, token = runtime.token.data_ptr(), runtime.token._version, runtime.token.clone()
    expected = _public_reference(inputs, runtime.store)
    mathematical = _public_reference(inputs, runtime.store, rounded=False)
    output = _single_output(runtime.invoke(loaded, inputs, artifact_config=config, bindings=bindings))
    torch.cuda.synchronize(runtime.device)
    count = len(config.layer_specs)
    after = _public_status(runtime)
    torch.save(dict(inputs=[value.cpu() for value in inputs], output=output.cpu(), expected=expected,
                    mathematical=mathematical),
               directory / f"{label}.tensors.pt")
    assert after["state"] == "healthy" and not after["managed"] and not after["pending_error"], after
    assert after["managed_sequence"] == before["managed_sequence"] + 1, after
    assert after["forward_sequence"] == before["forward_sequence"] + count, after
    assert after["success_sequence"] == after["forward_sequence"], after
    assert after["success_layer"] == config.layer_specs[-1][0], after
    assert runtime.token.data_ptr() == ptr and runtime.token[0].item() == token[0].item()
    assert runtime.token[1].item() == token[1].item() + count and runtime.token._version == version + count
    assert output.shape == expected.shape and output.dtype == inputs[2].dtype and output.device == inputs[2].device
    actual = output.cpu()
    # Compare the MoE contribution as well: the extra business output must not
    # make a missing/zero offload result look numerically acceptable.
    business = inputs[0].cpu() * .125
    contribution = expected.float() - business.float()
    error = actual.float() - expected.float()
    contribution_norm = contribution.norm().item()
    relative = error.norm() / contribution.norm()
    report = dict(label=label, status=after, stats=runtime.stats(),
                  relative_l2=relative.item(), contribution_norm=contribution_norm,
                  error_norm=error.norm().item(), max_abs_error=error.abs().max().item(),
                  mathematical_error=_public_diagnostic_metric(actual, mathematical),
                  mathematical_contribution_relative_l2=(actual.float() - mathematical.float()).norm().item() /
                      (mathematical.float() - business.float()).norm().item(),
                  environment_signature=_environment_signature())
    # Retain diagnostics before any numerical assertion: failures are evidence,
    # never a skip or a successful report, and still propagate to the process.
    _write_json(directory / f"{label}.json", report)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-5, rtol=RTOL)
    # Retain the original mathematical FP32 reference's pointwise gate too.
    torch.testing.assert_close(actual.float(), mathematical.float(), atol=2e-5, rtol=RTOL)
    assert contribution_norm > 1e-5, (label, contribution_norm)
    assert relative.item() < .05, (label, relative.item())
    print("PASS PUBLIC " + json.dumps(report, sort_keys=True), flush=True)


def _public_negative(loaded, runtime, config, inputs, bindings, directory, layer_index, *, fault=None):
    before = _public_status(runtime)
    first_layer = config.layer_specs[layer_index][0]
    native_message = None
    if fault is None:
        # Establish the original native diagnostic independently, using the same
        # public begin/callback/check protocol before calling the official AOTI.
        with _LockedPrefillMoERuntime.from_config(runtime.store, config, device=runtime.device) as direct:
            args = direct.prepare_inputs(_public_inputs(7, runtime.store, runtime.device), bindings)
            route = args[3 + 4 * layer_index].clone()
            route[0, 0] = config.layer_specs[layer_index][1]
            x = torch.zeros((7, config.layer_specs[layer_index][2]), device=runtime.device, dtype=torch.float16)
            try:
                direct.invoke(lambda *bound: torch.ops.recsys_ops.fuse_moe_offload(
                    x, bound[4 + 4 * layer_index], bound[6 + 4 * layer_index],
                    route, bound[5 + 4 * layer_index], bound[1], first_layer,
                    config.layer_config(first_layer), config.manifest()),
                    args, artifact_config=config, bindings=bindings)
            except RuntimeError as error:
                native_message = str(error)
                assert "topk_ids must be -1 or in [0,E)" in native_message, error
                assert _public_status(direct)["failure_layer"] == first_layer
                _write_json(directory / "negative.native.json", dict(error=repr(error), status=direct.status()))
            else:
                raise AssertionError("direct checked native route was accepted")
    invalid = list(inputs)
    if fault is None:
        invalid[3 + 4 * layer_index] = invalid[3 + 4 * layer_index].clone()
        invalid[3 + 4 * layer_index][0, 0] = config.layer_specs[layer_index][1]
    else:
        torch.ops.recsys_ops._fuse_moe_offload_test_inject(runtime.token, fault, 1)
    torch.save(dict(inputs=[value.cpu() for value in invalid], status=before),
               directory / "negative.inputs.pt")
    reached = False
    sentinel = object()
    published = sentinel
    def callback(*args):
        nonlocal reached
        reached = True
        return loaded(*args)
    try:
        published = runtime.invoke(callback, invalid, artifact_config=config, bindings=bindings)
    except Exception as error:
        after = _public_status(runtime)
        _write_json(directory / "negative.status.json", dict(
            before=before, after=after, error=repr(error), callback_reached=reached,
            outputs_published=published is not sentinel, fault=fault,
            callback_error=repr(error.__cause__), native_message=native_message))
        assert reached and published is sentinel, "Official loader must be reached; no partial output accepted"
        expected_state = "poisoned" if fault is not None else "recoverable"
        assert after["state"] == expected_state and after["pending_error"] and not after["managed"], after
        assert after["failure_layer"] == first_layer, after
        assert after["forward_sequence"] == after["failure_sequence"], "downstream offload ran after first error"
        if fault is None:
            assert "topk_ids must be -1 or in [0,E)" in str(error), (native_message, error)
            assert error.__cause__ is not None, "native check must supersede the official checked-proxy error"
        assert after["failure_sequence"] == before["forward_sequence"] + layer_index + 1, after
        assert after["success_sequence"] == before["forward_sequence"] + layer_index, after
        # Sticky first failure survives retries; later layer/output cannot mask it.
        blocked = mock.Mock()
        try:
            runtime.invoke(blocked, inputs, artifact_config=config, bindings=bindings)
        except Exception:
            pass
        else:
            raise AssertionError("pending error was silently cleared")
        blocked.assert_not_called()
        sticky = _public_status(runtime)
        assert sticky["failure_layer"] == first_layer and sticky["failure_sequence"] == after["failure_sequence"]
        if expected_state == "poisoned":
            try:
                runtime.acknowledge()
            except RuntimeError as ack_error:
                assert "poison" in str(ack_error).lower(), ack_error
            else:
                raise AssertionError("acknowledge must never clear poisoned state")
            assert _public_status(runtime)["state"] == "poisoned"
            runtime.close()
            return
        runtime.acknowledge()
        restored = _public_status(runtime)
        assert restored["state"] == "healthy" and not restored["pending_error"] and not restored["managed"], restored
        _public_check(loaded, inputs, runtime, config, bindings, directory, "negative_recovered")
    else:
        raise AssertionError("negative route/fault accepted by safe official AOTI wrapper")


def _public_callback_cleanup(loaded, runtime, config, inputs, bindings):
    class CallbackFailure(RuntimeError):
        pass
    original = CallbackFailure("unknown business callback error")
    def fail(*args):
        raise original
    try:
        runtime.invoke(fail, inputs, artifact_config=config, bindings=bindings)
    except CallbackFailure as error:
        assert error is original, "unknown callback exception must survive native check cleanup"
    else:
        raise AssertionError("callback error swallowed")
    assert not _public_status(runtime)["managed"]
    # Successful native check cleanup does not poison the runtime.
    runtime.invoke(loaded, inputs, artifact_config=config, bindings=bindings)
    missing_config = mock.Mock()
    try:
        runtime.invoke(missing_config, inputs, bindings=bindings)
    except TypeError:
        pass
    else:
        raise AssertionError("artifact_config must be mandatory, never inferred from self")
    missing_config.assert_not_called()


def _public_concurrency(package, store, config, device, bindings, directory):
    barrier = threading.Barrier(3, timeout=120)
    def worker(index):
        # Never share an official loader across threads: its thread safety is
        # version-dependent. Immutable pageable CPU weights may be shared.
        torch.cuda.set_device(device)
        loader = torch._inductor.aoti_load_package(package)
        with _LockedPrefillMoERuntime.from_config(store, config, device=device) as runtime:
            inputs = runtime.prepare_inputs(_public_inputs(7, store, device), bindings)
            barrier.wait()
            for repeat in range(2):
                _public_check(loader, inputs, runtime, config, bindings, directory,
                              f"public_thread{index}_repeat{repeat}")
            return runtime.token[0].item()
    with ThreadPoolExecutor(max_workers=3) as pool:
        ids = list(pool.map(worker, range(3)))
    assert len(set(ids)) == 3, "Concurrent contexts must have distinct tokens"
    runtime = _LockedPrefillMoERuntime.from_config(store, config, device=device)
    loader = torch._inductor.aoti_load_package(package)
    inputs = runtime.prepare_inputs(_public_inputs(7, store, device), bindings)
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    def callback(*args):
        entered.set()
        assert release.wait(120), "close/invoke coordination timed out"
        return loader(*args)
    def invoke():
        torch.cuda.set_device(device)
        return runtime.invoke(callback, inputs, artifact_config=config, bindings=bindings)
    def close():
        closing.set()
        runtime.close()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(invoke)
            try:
                assert entered.wait(120)
                stopped = pool.submit(close)
                assert closing.wait(120) and not stopped.done(), "close must serialize behind active invoke"
            finally:
                release.set()  # Release before ThreadPoolExecutor waits on exit.
            assert len(pytree.tree_leaves(running.result(timeout=120))) == 1
            stopped.result(timeout=120)
        callback_after_close = mock.Mock()
        try:
            runtime.invoke(callback_after_close, inputs, artifact_config=config, bindings=bindings)
        except RuntimeError as error:
            assert "closed" in str(error).lower()
        else:
            raise AssertionError("invoke after close accepted")
        callback_after_close.assert_not_called()
    finally:
        release.set()
        runtime.close()


def _deployment_bundle(folder, package, library, archive):
    # T09 requires no external native source tree, not a source-stripped PT2.
    # Preserve the official inventory, names and bytes: generated .cpp, metadata,
    # compile/link flags and extern-kernel JSON are loader-owned package members.
    # Checked export source proof remains separately in adapter.source_proof.
    folder.mkdir(parents=True, exist_ok=True)
    target_package = folder / "adapter.pt2"
    shutil.copy2(package, target_package)
    original_package_sha256 = _sha256(package)
    assert _sha256(target_package) == original_package_sha256, "official PT2 copy changed"
    target_library = folder / Path(library).name
    for source, target in ((Path(library), target_library),
                           (Path(str(library) + ".json"), Path(str(target_library) + ".json")),
                           (Path(archive), folder / "weights.pt")):
        shutil.copy2(source, target)
    _write_json(folder / "bundle.json", dict(
        schema_version=1, original_package_sha256=original_package_sha256,
        official_package_intact=True, external_native_source_tree_required=False,
        embedded_generated_sources_may_be_present=True,
        files={path.name: _sha256(path) for path in sorted(folder.iterdir()) if path.is_file()}))
    return target_package, target_library, folder / "weights.pt"


def _deployment_native_loader(package, *, torch_version=None, loader_type=None):
    # Reviewed release-specific internal binding, not a generic compatibility
    # wrapper. Never retry a failed constructor with a different signature.
    version = str(torch.__version__) if torch_version is None else torch_version
    loader_type = torch._C._aoti.AOTIModelPackageLoader if loader_type is None else loader_type
    if re.fullmatch(r"2\.6\.0a0(?:\+[^\s]+)?", version):
        arguments = (str(package), "model")
        signature = r"str\s*,\s*\w+\s*:\s*str"
    elif re.fullmatch(r"2\.11\.0(?:\+[^\s]+)?", version):
        arguments = (str(package), "model", False, 1, -1)
        signature = (r"str\s*,\s*\w+\s*:\s*str\s*,\s*\w+\s*:\s*bool(?:\s*=\s*False)?"
                     r"\s*,\s*\w+\s*:\s*(?:int|typing\.SupportsInt)(?:\s*=\s*1)?"
                     r"\s*,\s*\w+\s*:\s*(?:int|typing\.SupportsInt)(?:\s*=\s*-1)?")
    else:
        raise RuntimeError(f"deployment native loader: unsupported reviewed torch release {version}")
    constructor_doc = loader_type.__init__.__doc__ or ""
    if re.search(r"__init__\(self(?:\s*:\s*[^,]+)?\s*,\s*\w+\s*:\s*" +
                 signature + r"\s*\)", constructor_doc) is None:
        raise RuntimeError(f"deployment native loader constructor schema mismatch for {version}: {constructor_doc}")
    return loader_type(*arguments)


def _run_compile_native_runner_only(output_dir):
    # Explicit offline compiler entry: no ops registration, lowering, GPU
    # detection/environment signature, package compilation or model execution.
    with _artifact_directory("native_runner_compile_", output_dir) as directory:
        report = dict(schema_version=1, state="running", torch=str(torch.__version__))
        try:
            executable = _compile_runner(directory)
            process = json.loads((directory / "cpp_compile.exit.json").read_text())
            report.update(state="passed", command=process["command"],
                          returncode=process["returncode"], executable=str(executable),
                          executable_sha256=_sha256(executable),
                          source=json.loads((directory / "cpp_source.json").read_text()))
            print(f"COMPILED_NATIVE_RUNNER {executable}", flush=True)
            return executable
        except BaseException as error:
            report.update(state="failed", error=repr(error))
            process_path = directory / "cpp_compile.exit.json"
            if process_path.exists():
                report.update(json.loads(process_path.read_text()))
            raise
        finally:
            _write_json(directory / "compile_report.json", report)


def _run_deployment_child(options):
    # This branch never calls _compile_runner, export, or lowering registration.
    # Import official compiler modules only to install fail-on-use mocks. The
    # immutable PT2 may contain generated sources; no external native source tree
    # or compiler executable may be used during deployment execution.
    import torch.utils.cpp_extension as extension
    import torch._inductor.cpp_builder as builder
    import torch._inductor.package.package as packaging

    attempts = []
    passed = False
    def disallowed(name):
        def fail(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"deployment must not compile/read native sources/spawn tools: {name}")
        return fail
    source_suffixes = {".cpp", ".cc", ".cxx", ".cu", ".cuh", ".h", ".hpp"}
    def audit(event, args):
        if event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
            disallowed(event)()
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(args[0])).suffix.lower() in source_suffixes:
                disallowed("source:" + os.fsdecode(args[0]))()
    directory = Path(options.output_dir).resolve()
    package = Path(options.public_package).resolve()
    assert not any(path.is_file() and path.suffix.lower() in source_suffixes
                   for path in package.parent.rglob("*")), "deployment bundle contains external native sources"
    with zipfile.ZipFile(package) as artifact:
        package_members = artifact.namelist()
        assert any(name.endswith(".so") for name in package_members), "deploy package must already be prelinked"
        embedded_sources = [name for name in package_members
                            if Path(name).suffix.lower() in source_suffixes]
    bundle = json.loads((package.parent / "bundle.json").read_text())
    package_sha256 = _sha256(package)
    assert bundle["official_package_intact"] is True
    assert package_sha256 == bundle["original_package_sha256"], "official PT2 differs from export artifact"
    sys.addaudithook(audit)  # Fresh child only; never pollute the parent process.
    try:
        with ExitStack() as stack:
            for owner, name in (
                    (extension, "load"), (extension, "load_inline"), (extension, "_jit_compile"),
                    (extension, "_run_ninja_build"), (builder.CppBuilder, "build"),
                    (torch, "compile"), (torch.export, "export"),
                    (torch._inductor, "aot_compile"), (torch._inductor, "aoti_compile_and_package"),
                    (packaging, "compile_so")):
                if hasattr(owner, name):
                    stack.enter_context(mock.patch.object(owner, name, side_effect=disallowed(name)))
            stack.enter_context(mock.patch(__name__ + ".setup_compiler", side_effect=disallowed("setup_compiler")))
            stack.enter_context(mock.patch(__name__ + "._compile_runner", side_effect=disallowed("_compile_runner")))
            stack.enter_context(mock.patch(__name__ + ".compile_aoti", side_effect=disallowed("compile_aoti")))
            stack.enter_context(mock.patch(__name__ + ".register_aoti_lowering", side_effect=disallowed("register_aoti_lowering")))
            stack.enter_context(mock.patch.dict(os.environ, {
                "PATH": "", "CXX": "/compiler-unavailable/c++", "CC": "/compiler-unavailable/cc",
                "CUDA_HOME": "/toolkit-unavailable", "CUDACXX": "/toolkit-unavailable/nvcc"}))
            library = _load_ops(options.ops_library, options.wait_seconds,
                                allow_testing=options.allow_testing, compile_lowering=False)
            store = ExpertStore.load_cpp(options.public_archive, ram_budget_bytes=16 * 1024**2)
            config = store.deployment_config
            device = torch.device("cuda", torch.cuda.current_device())
            bindings = _public_bindings(len(store.layers))
            # Use the official C++ package loader through its Python binding.
            # In torch 2.11 the Python package helper probes CPU ISA via
            # `c++ --version`: read-only, but still a compiler dependency.
            # This test has flat Tensor inputs/outputs; no pytree reconstruction
            # or replacement package loading/compilation is needed here.
            loader = _deployment_native_loader(package)
            def invoke_loader(*args):
                return loader.run(list(args))
            with _LockedPrefillMoERuntime.from_config(store, config, device=device) as runtime:
                for size in (2, 7, 64):
                    inputs = runtime.prepare_inputs(_public_inputs(size, store, device), bindings)
                    _public_check(invoke_loader, inputs, runtime, config, bindings, directory, f"deploy_S{size}")
            assert not attempts, attempts
            assert _sha256(package) == package_sha256, "official loader changed deployment PT2"
            passed = True
    finally:
        _write_json(directory / "deployment.json", dict(
            schema_version=1, state="passed" if passed else "failed",
            compiler_attempts=attempts,
            loader_api="torch._C._aoti.AOTIModelPackageLoader",
            callback_api="loader.run(list(flat_tensor_inputs))",
            python_package_helper_used=False, external_source_reads_disallowed=True,
            external_native_source_tree_required=False,
            embedded_source_members=embedded_sources, official_package_intact=True,
            subprocesses_disallowed=True, package_sha256=_sha256(package),
            original_package_sha256=package_sha256,
            package_unchanged_after_load=_sha256(package) == package_sha256,
            library_sha256=_sha256(options.ops_library),
            metadata_sha256=_sha256(str(options.ops_library) + ".json")))
    print(f"PASS DEPLOYMENT_NO_COMPILE library={library}", flush=True)


def _run_public_child(options):
    directory = Path(options.output_dir).resolve()
    library = _load_ops(options.ops_library, options.wait_seconds,
                        allow_testing=options.allow_testing, compile_lowering=False)
    store = ExpertStore.load_cpp(options.public_archive, ram_budget_bytes=16 * 1024**2)
    config = store.deployment_config
    device = torch.device("cuda", torch.cuda.current_device())
    bindings = _public_bindings(len(store.layers))
    loaded = torch._inductor.aoti_load_package(options.public_package)
    with _LockedPrefillMoERuntime.from_config(store, config, device=device) as runtime:
        inputs = runtime.prepare_inputs(_public_inputs(7, store, device), bindings)
        _public_negative(loaded, runtime, config, inputs, bindings, directory,
                         options.public_error_layer, fault=options.public_fault)
    print(f"PASS PUBLIC child official AOTI negative library={library}", flush=True)


def _check_public_acceptance(directory, executable, library, device, sizes, *,
                             async_staging, host_profile, allow_testing):
    major, minor = torch.cuda.get_device_capability(device)
    for count in (1, 2, 3):
        folder = directory / f"public_{count}layer"
        folder.mkdir(parents=True, exist_ok=True)
        # Larger scales protect the three-layer contribution from FP16 underflow.
        generator = torch.Generator().manual_seed(20261014)
        store = ExpertStore({layer: ((torch.randn(e, 2 * i, h, generator=generator) * .18).half(),
                                     (torch.randn(e, h, i, generator=generator) * .18).half())
                             for layer, e, h, i in _PUBLIC_SPECS[:count]},
                            ram_budget_bytes=16 * 1024**2)
        config = OffloadConfig.from_store(
            store, gpu_budget_bytes=8 * 1024**2, ram_budget_bytes=16 * 1024**2,
            pinned_budget_bytes=2 * 1024**2, target_sm=major * 10 + minor,
            activation_mode="fp32_accumulate", max_route_tile_rows=3,
            tile_m=32, tile_n=64, async_staging=async_staging, host_profile=host_profile)
        archive = folder / "weights.pt"
        store.save_cpp(archive, config=config, weight_version=20261014)
        _write_json(folder / "artifact.json", dict(
            schema_version=1, adapter="public_op_arbitrary_scalar_business_no_model",
            manifest=config.manifest(), specs=_PUBLIC_SPECS[:count], dynamic_S=[2, 64],
            input_layout=["business", "token", "x", "per-layer: ids,w1,scale,w2"],
            bindings=[dict(index=b.index, kind=b.kind, layer_id=b.layer_id) for b in _public_bindings(count)],
            environment_signature=_environment_signature(library)))
        with _LockedPrefillMoERuntime.from_config(store, config, device=device) as runtime:
            bindings = _public_bindings(count)
            raw = _public_inputs(7, store, device)
            inputs = runtime.prepare_inputs(raw, bindings)
            assert raw[1] is None and raw[4] is None and raw[6] is None, "prepare must not mutate caller list"
            for index in (0, 2, *(3 + 4 * n for n in range(count)), *(5 + 4 * n for n in range(count))):
                assert inputs[index] is raw[index], "business inputs must remain untouched"
            rejected_callback = mock.Mock()
            before_mapping_rejection = _public_status(runtime)
            for bad in (bindings[:-1], [*bindings, bindings[0]]):
                try:
                    runtime.invoke(rejected_callback, inputs, artifact_config=config, bindings=bad)
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError("incomplete/duplicate public binding accepted")
            rejected_callback.assert_not_called()
            assert _public_status(runtime) == before_mapping_rejection
            replaced_inputs = list(inputs)
            replaced_inputs[4] = inputs[4].clone()
            try:
                runtime.invoke(rejected_callback, replaced_inputs, artifact_config=config, bindings=bindings)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("equal cloned unbound weight identity accepted")
            rejected_callback.assert_not_called()
            assert _public_status(runtime) == before_mapping_rejection
            before_rejection = _public_status(runtime)
            try:
                runtime.invoke(rejected_callback, inputs, artifact_config=replace(config, tile_n=128),
                               bindings=bindings)
            except (ValueError, RuntimeError) as error:
                assert "configuration mismatch" in str(error), error
            else:
                raise AssertionError("explicit wrong artifact config accepted")
            rejected_callback.assert_not_called()
            assert _public_status(runtime) == before_rejection
            dim = torch.export.Dim(f"public_{count}_S", min=2, max=64)
            shapes = ({0: dim}, None, {0: dim}, *(shape for _ in range(count)
                       for shape in ({0: dim}, None, {0: dim}, None)))
            adapter = (_PublicSingle, _PublicTwo, _PublicThree)[count - 1](config).eval()
            snapshot, version = runtime.token.clone(), runtime.token._version
            exported = torch.export.export(adapter, tuple(inputs), dynamic_shapes=shapes)
            assert not exported.state_dict and not list(adapter.parameters()) and not list(adapter.buffers())
            calls = [node for node in exported.graph.nodes if node.op == "call_function" and
                     node.target == torch.ops.recsys_ops.fuse_moe_offload.default]
            assert len(calls) == count, exported.graph
            assert [node.args[6] for node in calls] == [s[0] for s in _PUBLIC_SPECS[:count]]
            assert all(tuple(node.args[8]) == config.manifest() for node in calls), exported.graph
            package = _compile_checked_package(
                exported, package_path=str(folder / "adapter.pt2"),
                inductor_configs={"triton.cudagraphs": False})
            assert torch.equal(snapshot, runtime.token) and runtime.token._version == version
            with zipfile.ZipFile(package) as artifact:
                shared = [name for name in artifact.namelist() if name.endswith(".so")]
                assert len(shared) == 1
                extern = json.loads(artifact.read(str(Path(shared[0]).with_suffix(".json"))))
                vectors = list(_serialized_int_vectors(extern))
                assert vectors.count(config.manifest()) == count, extern
                assert all(config.layer_config(layer) in vectors for layer in store.layers), extern
            loaded = torch._inductor.aoti_load_package(package)
            for s in (*sizes, sizes[0]):
                prepared = runtime.prepare_inputs(_public_inputs(s, store, device), bindings)
                _public_check(loaded, prepared, runtime, config, bindings, folder, f"public_S{s}_seq{runtime.token[1].item()}")
            _public_callback_cleanup(loaded, runtime, config, inputs, bindings)
        deployment_dir = folder / "deployment"
        deploy_package, deploy_library, deploy_archive = _deployment_bundle(
            deployment_dir, package, library, archive)
        deploy_report = deployment_dir / "reports"
        deploy_report.mkdir(exist_ok=True)
        deploy_command = [sys.executable, str(Path(__file__).resolve()), "--deployment-only",
                          "--ops-library", str(deploy_library), "--public-package", str(deploy_package),
                          "--public-archive", str(deploy_archive), "--output-dir", str(deploy_report)]
        if allow_testing:
            deploy_command.append("--allow-testing")
        _run_process(deploy_command, deploy_report, "python_deployment")
        _public_concurrency(package, store, config, device, bindings, folder)
        # The standalone official loader exercises the same intact deployment
        # bundle; it must not depend on the export tree or source-proof directory.
        cpp = [str(executable), "--public", str(deploy_package), str(deploy_library),
               str(device.index), str(deploy_archive)]
        _run_process([*cpp, "--output-dir", str(folder / "cpp_raw")], folder, "cpp_public")
        # Checked proxy must stop before undefined output/downstream work. Each
        # negative is isolated: any official-loader crash is retained as failure.
        for layer_index in range(count):
            negative_dir = folder / f"negative_layer{layer_index}"
            negative_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--ops-library", str(library),
                       "--public-package", str(package), "--public-archive", str(archive),
                       "--public-error-layer", str(layer_index), "--output-dir", str(negative_dir)]
            if allow_testing:
                command.append("--allow-testing")
            _run_process(command, negative_dir, "python_negative")
            _run_process([*cpp, "--negative-layer", str(layer_index),
                          "--output-dir", str(negative_dir / "cpp_raw")], negative_dir, "cpp_negative")
        if allow_testing:
            for fault in ("after_first_h2d", "after_kernel_submit", "thread_create"):
                # Thread creation is lazy and needs async staging. Each child
                # starts a fresh runtime so earlier faults cannot mask this one.
                if fault == "thread_create" and not config.async_staging:
                    continue
                fault_dir = folder / f"fault_{fault}"
                fault_dir.mkdir(exist_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--ops-library", str(library),
                           "--public-package", str(package), "--public-archive", str(archive),
                           "--public-fault", fault, "--allow-testing", "--output-dir", str(fault_dir)]
                _run_process(command, fault_dir, "python_fault")


def _parse_route_tile_rows(value):
    if value == "auto":
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise argparse.ArgumentTypeError("route tile rows must be auto or an integer in [1,INT_MAX]")
    try:
        rows = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("route tile rows must be auto or an integer") from error
    if not 1 <= rows <= 2**31 - 1:
        raise argparse.ArgumentTypeError("route tile rows must be in [1,INT_MAX]; use auto for native 0")
    return rows


@torch.no_grad()
def run_cpp_benchmark(*, ops_library=None, iterations=20, sizes=(64, 8192, 16384), wait_seconds=0,
                      async_staging=False, host_profile=False, allow_testing=False,
                      output_dir=None):
    """Export one external-weight H128/I64 graph, then time Python-free AOTI only."""
    if iterations < 1 or any(s < 7 or s > 16384 for s in sizes) or not sizes:
        raise ValueError("positive iterations and S in [7,16384] are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for standalone AOTI benchmarking")
    library = _load_ops(ops_library, wait_seconds, allow_testing=allow_testing)
    device = torch.device("cuda", torch.cuda.current_device())
    with _artifact_directory("fuse_moe_cpp_benchmark_", output_dir) as temporary, \
            mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": str(Path(temporary) / "inductor_cache"),
                                         "TRITON_CACHE_DIR": str(Path(temporary) / "triton_cache")}), \
            mock.patch(__name__ + ".PrefillMoERuntime", partial(
                PrefillMoERuntime, async_staging=async_staging, host_profile=host_profile)):
        directory = Path(temporary)
        executable = _compile_runner(directory)
        package, archive = _check_wide_acceptance(directory, device, bf16=False, benchmark_only=True)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        command = [str(executable), package, str(library), str(device.index), archive,
                   "--benchmark-wide", str(iterations), *(str(s) for s in sizes)]
        command += _runtime_cli_flags(async_staging, host_profile)
        print("Standalone benchmark: " + shlex.join(command), flush=True)
        try:
            completed = _run_process(command, directory, "cpp_benchmark", env=environment)
            reports = [json.loads(line.removeprefix("BENCHMARK ")) for line in completed.stdout.splitlines()
                       if line.startswith("BENCHMARK ")]
            for report in reports:
                report["environment_signature"].update(_environment_signature(library))
            _write_json(directory / "cpp_benchmark.json", reports)
        finally:
            torch._dynamo.reset()
            gc.collect()


def _benchmark_profiles(selection):
    profiles = {
        "balanced": (1024, 256, 256, 32 * 1024**2),
        "small": (64, 64, 64, 2 * 1024**2),
        "weight_transfer": (128, 512, 512, 48 * 1024**2),
        "activation_near_budget": (8192, 256, 256, 60 * 1024**2),
    }
    if selection in (None, "all"):
        return list(profiles.items())
    if selection not in profiles:
        raise ValueError(f"benchmark_profile must be all or one of {tuple(profiles)}")
    return [(selection, profiles[selection])]


def _benchmark_schedule(scenario, iterations):
    # At least three observations: reuse admits on call two, hits on call three.
    repeats = max(3, iterations)
    if scenario in ("uniform", "skewed_masked"):
        return [(scenario, 0, scenario)] * repeats
    if scenario == "fixed_hot":
        return [("hot", 0, "hot_a")] * repeats
    if scenario == "hot_drift":
        # A has three priming observations; B needs to exceed A + hysteresis.
        return [("drift", 0, "hot_b")] * max(5, iterations)
    if scenario == "scan_return":
        # Each cold expert is seen once in a single invocation, not once per route.
        return [("one_shot_scan", 0, "hot_b")] + [("return_hot", 0, "hot_a")] * repeats
    if scenario == "cross_layer":
        return ([("layer1_compete", 1, "hot_a")] * max(5, iterations) +
                [("layer0_return", 0, "hot_a")] * max(7, iterations))
    raise ValueError(f"Unknown benchmark_scenario: {scenario}")


def _benchmark_latency(values):
    samples = list(values)  # Preserve acquisition order, not just percentiles.
    values = sorted(samples)
    return dict(samples=len(values), samples_ms=samples,
                latency_ms_median=statistics.median(values),
                latency_ms_p95=values[math.ceil(.95 * len(values)) - 1],
                latency_ms_total=sum(values))


def _benchmark_summary(samples):
    # Counters are last-call values; gauges must NOT be summed as traffic.
    counters = ("active_experts", "waves", "weight_h2d_bytes", "metadata_h2d_bytes",
                "cache_hits", "cache_misses", "cache_hit_weight_bytes", "dynamic_cache_hits",
                "cache_admissions", "cache_evictions", "cache_bypasses",
                "cache_promotion_d2d_bytes", "cache_budget_evictions", "route_tiles",
                "buffer_allocations", "buffer_reuses")
    gauges = ("gpu_charged_bytes", "pinned_bytes", "ram_charged_bytes", "gpu_pool_bytes",
              "pinned_pool_bytes", "resident_cache_bytes", "dynamic_cache_bytes",
              "resident_experts", "dynamic_experts", "cache_metadata_bytes")
    totals = {key: sum(sample["stats"][key] for sample in samples) for key in counters}
    stage_summary = {}
    if any("host_stages" in sample for sample in samples) and not all(
            "host_stages" in sample for sample in samples):
        raise ValueError("cannot mix host-profiled and unprofiled samples")
    if all("host_stages" in sample for sample in samples):
        names = sorted({name for sample in samples for name in sample["host_stages"]})
        stage_summary["host_stages"] = {name: dict(
            **_benchmark_latency(sample["host_stages"].get(name, {}).get("cpu_ms", 0) for sample in samples),
            calls_total=sum(sample["host_stages"].get(name, {}).get("calls", 0) for sample in samples))
            for name in names}
    return dict(**stage_summary, **_benchmark_latency(sample["host_ms"] for sample in samples),
                counters_total=totals,
                counters_per_call_mean={key: value / len(samples) for key, value in totals.items()},
                distinct_expert_hit_ratio=totals["cache_hits"] / totals["active_experts"],
                gauges_max={key: max(sample["stats"][key] for sample in samples) for key in gauges},
                gauges_final={key: samples[-1]["stats"][key] for key in gauges},
                actual_banks=sorted({sample["stats"]["num_banks"] for sample in samples}),
                actual_wave_capacities=sorted({sample["stats"]["wave_capacity"] for sample in samples}),
                actual_activation_modes=sorted({sample["stats"]["activation_mode"] for sample in samples}),
                allocator_peak_delta_bytes=max(sample["allocator_peak_delta_bytes"] for sample in samples),
                allocator_allocated_max_bytes=max(sample["allocator_allocated_bytes"] for sample in samples),
                allocator_reserved_max_bytes=max(sample["allocator_reserved_bytes"] for sample in samples))


@torch.no_grad()
def run_benchmark(*, ops_library=None, iterations=20, wait_seconds=0,
                  benchmark_profile="all", benchmark_scenario="all", max_route_tile_rows=512,
                  nvtx=False, host_profile=False, async_staging=False,
                  allow_testing=False, output_dir=None):
    """Synthetic eager-native host timings, including promotion drain, not a speedup claim.

    Profile/scenario filters are Python parameters for caller-owned CLI integration.
    NVTX ranges and host profiling are enabled by explicit parameters or CLI flags.
    Cold is a first forward in a fresh context; warm is a stateful demand sequence,
    not necessarily cache-hit steady state. Initialization and priming are separate.
    """
    profiles = _benchmark_profiles(benchmark_profile)
    scenario_names = ("fixed_hot", "uniform", "hot_drift", "scan_return", "cross_layer",
                      "skewed_masked")
    scenarios = scenario_names if benchmark_scenario in (None, "all") else (benchmark_scenario,)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    max_route_tile_rows = _parse_route_tile_rows(max_route_tile_rows)
    for scenario in scenarios:
        _benchmark_schedule(scenario, iterations)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the offload benchmark")
    library = _load_ops(ops_library, wait_seconds, allow_testing=allow_testing)
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    metadata = dict(environment_signature=_environment_signature(library),
                    torch=str(torch.__version__), cuda_build=torch.version.cuda,
                    gpu=properties.name, device_index=device.index,
                    compute_capability=list(torch.cuda.get_device_capability(device)),
                    gpu_total_memory_bytes=properties.total_memory, ops_library=str(library))
    def region(label):
        stack = ExitStack()
        if nvtx:
            stack.enter_context(torch.cuda.nvtx.range(label))
        return stack

    for profile, (s, h, intermediate, gpu_budget) in profiles:
        experts, k = 16, 8
        expert_bytes = 6 * h * intermediate
        expert_charge = ((expert_bytes + 511) // 512) * 512
        cache_cap = 8 * expert_charge
        ram_budget = 128 * 1024**2
        # Exact native metadata is (4*capacity+2)*4, NOT 24*capacity.
        # Same total pinned cap: bank2 capacity2; bank1 capacity4. No max-wave API.
        pinned_budget = 4 * (experts + 1) + 2 * (2 * expert_bytes + (4 * 2 + 2) * 4)
        generator = torch.Generator().manual_seed(20261012)
        layers = {}
        for layer in range(2 if "cross_layer" in scenarios else 1):
            layers[layer] = (
                (torch.randn(experts, 2 * intermediate, h, generator=generator) * .03).half(),
                (torch.randn(experts, h, intermediate, generator=generator) * .03).half())
        store = ExpertStore(layers, ram_budget_bytes=ram_budget)
        assert all(not w.is_pinned() and w.device.type == "cpu"
                   for pair in layers.values() for w in pair)
        x = (torch.randn(s, h, generator=generator) * .25).to(device=device, dtype=torch.float16)
        scale = torch.full((s, k), 1.0 / k, dtype=torch.float32, device=device)
        slots = torch.arange(k).reshape(1, k)
        rows = torch.arange(s).reshape(s, 1)
        host_routes = {"hot_a": ((slots + rows) % 8).int(),
                       "hot_b": (((slots + rows) % 8) + 8).int(),
                       "uniform": ((slots + rows * k) % experts).int()}
        # Six eligible experts, heavily repeated expert zero, signed scales and
        # fully masked rows. Never let -1 enter reference indexing or demand n.
        skewed = torch.where((rows + slots) % 11 == 0, (rows + slots) % 6,
                             torch.zeros(s, k, dtype=torch.int64))
        skewed[(rows + 2 * slots).remainder(5) == 0] = -1
        skewed[::7] = -1
        host_routes["skewed_masked"] = skewed.int()
        routes = {name: ids.to(device) for name, ids in host_routes.items()}
        active = {name: sorted(ids[ids >= 0].unique().tolist()) for name, ids in host_routes.items()}
        scales = {name: scale for name in routes}
        scales["skewed_masked"] = torch.where(
            ((rows + slots) % 3 == 0).to(device), -scale * .5, scale * 1.25)
        needed = {(0, scenario if scenario in ("uniform", "skewed_masked") else "hot_a")
                  for scenario in scenarios}
        for scenario in scenarios:
            needed.update((layer, route) for _, layer, route in _benchmark_schedule(scenario, iterations))
        references = {}
        # Separate per-expert FP32 matmul reference; never call the benchmark op.
        # Keep only CPU results, not a full GPU-weight replica during measurements.
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            for layer, route in sorted(needed):
                expected = torch.zeros(s, h, dtype=torch.float32, device=device)
                for expert in active[route]:
                    token_rows, token_slots = torch.where(routes[route] == expert)
                    gate, up = (x[token_rows].float() @ layers[layer][0][expert].to(
                        device=device, dtype=torch.float32).T).chunk(2, dim=-1)
                    values = (torch.nn.functional.silu(gate) * up) @ layers[layer][1][expert].to(
                        device=device, dtype=torch.float32).T
                    expected.index_add_(0, token_rows, values * scales[route][token_rows, token_slots, None])
                references[layer, route] = expected.cpu()
                del expected, token_rows, token_slots, gate, up, values
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
        torch.cuda.synchronize(device)

        for scenario in scenarios:
            prime_route = scenario if scenario in ("uniform", "skewed_masked") else "hot_a"
            for mode in ("route_buffer", "fp32_accumulate", "auto"):
                for banks in (1, 2):
                    for policy in ("disabled", "static_hot", "reuse", "lru"):
                        label = f"moe/{profile}/{scenario}/{mode}/bank{banks}/{policy}"
                        options = dict(device=device, gpu_budget_bytes=gpu_budget,
                                       ram_budget_bytes=ram_budget, pinned_budget_bytes=pinned_budget,
                                       num_banks=banks, activation_mode=mode,
                                       max_route_tile_rows=max_route_tile_rows)
                        if policy == "static_hot":
                            options.update(resident_cache_bytes=cache_cap,
                                           resident_experts={0: tuple(range(8))})
                        elif policy in ("reuse", "lru"):
                            options.update(dynamic_cache_bytes=cache_cap, cache_policy=policy,
                                           cache_min_reuse=2, cache_decay_interval=32,
                                           cache_update_bytes=cache_cap)

                        def initialize(stack):
                            torch.cuda.synchronize(device)
                            with region(label + "/initialize"):
                                started = time.perf_counter()
                                runtime = stack.enter_context(PrefillMoERuntime(
                                    store, **options, async_staging=async_staging,
                                    host_profile=host_profile))
                                torch.cuda.synchronize(device)
                                elapsed = (time.perf_counter() - started) * 1000
                            initial = runtime.stats()
                            assert initial["resident_experts"] == (8 if policy == "static_hot" else 0), initial
                            assert initial["resident_load_h2d_bytes"] == (8 * expert_bytes if policy == "static_hot" else 0), initial
                            assert initial["dynamic_experts"] == 0, initial
                            return runtime, elapsed, initial

                        def measure(runtime, phase, layer, route, *, validate=False):
                            before = runtime.stats()
                            torch.cuda.synchronize(device)
                            baseline = torch.cuda.memory_allocated(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            with ExitStack() as observers:
                                if host_profile:
                                    cpu_profile = observers.enter_context(torch.profiler.profile(
                                        activities=[torch.profiler.ProfilerActivity.CPU]))
                                with region(label + f"/{phase}/layer{layer}/{route}"):
                                    started = time.perf_counter()
                                    output = runtime.forward(x, routes[route], scales[route], layer_id=layer)
                                    torch.cuda.synchronize(device)
                                    elapsed = (time.perf_counter() - started) * 1000
                            sample = dict(phase=phase, layer_id=layer, route=route, host_ms=elapsed,
                                          allocator_peak_delta_bytes=torch.cuda.max_memory_allocated(device) - baseline,
                                          allocator_allocated_bytes=torch.cuda.memory_allocated(device),
                                          allocator_reserved_bytes=torch.cuda.memory_reserved(device))
                            if host_profile:
                                sample["host_stages"] = {event.key.removeprefix("fuse_moe_offload::"):
                                    dict(calls=event.count, cpu_ms=event.cpu_time_total / 1000)
                                    for event in cpu_profile.key_averages()
                                    if event.key.startswith("fuse_moe_offload::")}
                            stats = runtime.stats()  # Last-call stats, outside the timer.
                            sample["stats"] = stats
                            assert output.shape == x.shape and output.dtype == x.dtype and output.device == device
                            if validate:
                                actual, expected = output.float().cpu(), references[layer, route]
                                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-2)
                                if route == "skewed_masked":
                                    masked = (host_routes[route] < 0).all(dim=1)
                                    assert masked.any() and torch.count_nonzero(actual[masked]).item() == 0
                                relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                                assert relative_l2.item() < .02, (label, phase, relative_l2.item())
                                sample["reference_relative_l2_error"] = relative_l2.item()
                                del actual, expected
                            del output  # No prior output or GPU reference survives to the next call.
                            n = len(active[route])
                            hits, misses = stats["cache_hits"], stats["cache_misses"]
                            assert stats["active_experts"] == n == hits + misses, stats
                            assert stats["weight_h2d_bytes"] == misses * expert_bytes, stats
                            assert stats["cache_hit_weight_bytes"] == hits * expert_bytes, stats
                            assert stats["cache_promotion_d2d_bytes"] == stats["cache_admissions"] * expert_bytes, stats
                            assert stats["cache_promotion_d2d_bytes"] <= cache_cap, stats
                            assert stats["dynamic_experts"] == (before["dynamic_experts"] +
                                    stats["cache_admissions"] - stats["cache_evictions"]), stats
                            assert stats["dynamic_cache_bytes"] == stats["dynamic_experts"] * expert_charge <= cache_cap, stats
                            assert stats["cache_budget_evictions"] == 0, stats
                            assert stats["gpu_pool_bytes"] + stats["resident_cache_bytes"] + stats["dynamic_cache_bytes"] <= stats["gpu_charged_bytes"] <= gpu_budget, stats
                            assert stats["pinned_pool_bytes"] == stats["pinned_bytes"] <= pinned_budget, stats
                            assert store.weight_bytes <= stats["ram_charged_bytes"] <= ram_budget, stats
                            assert stats["activation_mode"] in (0, 1), stats
                            if mode != "auto":
                                assert stats["activation_mode"] == int(mode == "fp32_accumulate"), stats
                            if stats["activation_mode"] == 1:
                                assert stats["route_tile_rows"] > 0 and stats["route_tiles"] >= stats["waves"], stats
                                if max_route_tile_rows != "auto":
                                    assert stats["route_tile_rows"] <= max_route_tile_rows, stats
                            if misses:
                                actual_banks = stats["num_banks"]
                                assert 1 <= actual_banks <= banks, stats
                                assert 1 <= stats["wave_capacity"] <= min(misses, 4 // actual_banks), stats
                                # Pinned sets an upper bound; GPU workspace/high-water
                                # pools may force a smaller group or single banking.
                                if profile != "activation_near_budget":
                                    assert actual_banks == banks, stats
                                    assert stats["wave_capacity"] == min(misses, 4 // banks), stats
                            else:
                                assert stats["num_banks"] == stats["wave_capacity"] == 0, stats
                            # Resident pointer tables are kernel arguments plus GPU
                            # metadata, never counted as pinned metadata DMA.
                            miss_metadata = stats["metadata_h2d_bytes"] - 16 * misses
                            assert miss_metadata >= 0 and miss_metadata % 8 == 0, stats
                            miss_waves = miss_metadata // 8
                            if host_profile:
                                stages = sample["host_stages"]
                                def calls(name):
                                    return stages.get(name, {}).get("calls", 0)
                                assert calls("drain") == 1, stages
                                assert calls("compute_submit") == stats["route_tiles"], (stages, stats)
                                for name in ("host_weight_copy", "host_metadata", "h2d_submit"):
                                    assert calls(name) == miss_waves, (stages, stats)
                                assert calls("bank_wait") == max(0, miss_waves - stats["num_banks"]), (stages, stats)
                                assert calls("host_ready_wait") == (miss_waves if async_staging else 0), (stages, stats)
                                assert calls("host_bank_wait") == (
                                    max(0, miss_waves - stats["num_banks"]) if async_staging else 0), (stages, stats)
                            assert math.ceil(misses / max(stats["wave_capacity"], 1)) <= miss_waves <= misses, stats
                            hit_waves = stats["waves"] - miss_waves
                            assert math.ceil(hits / 32) <= hit_waves <= hits, stats
                            if hits == n and profile != "activation_near_budget":
                                assert stats["waves"] == 1, stats
                            if policy in ("disabled", "static_hot"):
                                expected_hits = len(set(active[route]) & set(range(8))) if policy == "static_hot" and layer == 0 else 0
                                assert hits == expected_hits and stats["dynamic_experts"] == 0, stats
                                assert stats["cache_admissions"] == stats["cache_evictions"] == stats["cache_bypasses"] == 0, stats
                            else:
                                assert hits == stats["dynamic_cache_hits"], stats
                                assert stats["cache_admissions"] + stats["cache_bypasses"] == misses, stats
                            return sample

                        # Untimed global/kernel warmup in a disposable context; not cold samples.
                        with ExitStack() as stack:
                            runtime, _, _ = initialize(stack)
                            measure(runtime, "kernel_warmup", 0, prime_route)
                        cold, initialization_ms, cold_total_ms = [], [], []
                        for repeat in range(iterations):
                            with ExitStack() as stack:
                                runtime, init_ms, _ = initialize(stack)
                                sample = measure(runtime, "cold", 0, prime_route, validate=repeat == 0)
                                cold.append(sample)
                                initialization_ms.append(init_ms)
                                cold_total_ms.append(init_ms + sample["host_ms"])
                        with ExitStack() as stack:
                            runtime, warm_init_ms, initial = initialize(stack)
                            priming = [measure(runtime, f"observe_{observation + 1}", 0, prime_route,
                                               validate=observation in (0, 2)) for observation in range(3)]
                            n = len(active[prime_route])
                            # Uniform demand exceeds the eight-entry cap; no full
                            # priming/hit promise is valid for that scan workload.
                            if n <= 8 and profile != "activation_near_budget":
                                if policy == "reuse":
                                    assert priming[0]["stats"]["cache_admissions"] == 0, priming
                                    assert priming[1]["stats"]["cache_admissions"] == n, priming
                                    assert priming[2]["stats"]["dynamic_cache_hits"] == n, priming
                                elif policy == "lru":
                                    assert priming[0]["stats"]["cache_admissions"] == n, priming
                                    assert priming[1]["stats"]["dynamic_cache_hits"] == n, priming
                            checked, warm = set(), []
                            for phase, layer, route in _benchmark_schedule(scenario, iterations):
                                key = (phase, layer, route)
                                warm.append(measure(runtime, phase, layer, route, validate=key not in checked))
                                checked.add(key)
                            if policy == "reuse" and scenario == "scan_return":
                                assert warm[0]["stats"]["cache_admissions"] == 0, warm[0]
                                assert warm[1]["stats"]["dynamic_cache_hits"] == 8, warm[1]
                            if policy == "lru" and scenario == "scan_return":
                                assert warm[0]["stats"]["cache_admissions"] == 8 and warm[0]["stats"]["cache_evictions"] == 8, warm[0]
                                assert warm[1]["stats"]["dynamic_cache_hits"] == 0, warm[1]
                            if policy in ("reuse", "lru") and scenario in ("hot_drift", "cross_layer"):
                                assert sum(sample["stats"]["cache_evictions"] for sample in warm) > 0, warm
                                assert warm[-1]["stats"]["dynamic_cache_hits"] == 8, warm[-1]
                            if scenario == "fixed_hot":
                                assert all(sample["stats"]["buffer_allocations"] == 0 and
                                           sample["stats"]["buffer_reuses"] > 0 for sample in warm), warm
                        phase_names = list(dict.fromkeys(sample["phase"] for sample in warm))
                        report = dict(schema_version=2, backend="eager_native", metadata=metadata,
                                      profile=profile, scenario=scenario, mode=mode, cache_policy=policy,
                                      requested_banks=banks, iterations=iterations,
                                      shape=dict(S=s, H=h, I=intermediate, E=experts, K=k),
                                      fixture="seeded_synthetic_fp16_weights_and_inputs_deterministic_nonrandom_routes",
                                      synthetic_weights=True, seed=20261012, layers=len(layers),
                                      gpu_budget_bytes=gpu_budget, ram_budget_bytes=ram_budget,
                                      pinned_budget_bytes=pinned_budget, cache_capacity_experts=8,
                                      resident_cache_cap_bytes=cache_cap if policy == "static_hot" else 0,
                                      dynamic_cache_cap_bytes=cache_cap if policy in ("reuse", "lru") else 0,
                                      static_hot_ids={0: list(range(8))} if policy == "static_hot" else {},
                                      max_route_tile_rows=max_route_tile_rows,
                                      cache_min_reuse=2, cache_decay_interval=32,
                                      cache_update_bytes=cache_cap if policy in ("reuse", "lru") else 0,
                                      fair_budget_note="identical total budgets; pinned upper bounds bank1/cap4 or bank2/cap2; GPU workspace may reduce either; no max-wave API",
                                      workload_note="dominance labels are shape intent, not profiler evidence",
                                      cold_excludes_initialization=True, initialization_excludes_library_loading=True,
                                      timer="host_perf_counter_forward_plus_device_synchronize_includes_promotion_drain",
                                      warm_is_stateful_sequence=True, nvtx_enabled=nvtx,
                                      host_profile_enabled=host_profile,
                                      async_staging_enabled=async_staging,
                                      host_profile_note="CPU observer scopes perturb timing; submit is host API time, not GPU activity",
                                      initialization=_benchmark_latency(initialization_ms),
                                      cold_total_init_plus_forward=_benchmark_latency(cold_total_ms),
                                      warm_context_initialization_ms=warm_init_ms, initialization_stats=initial,
                                      cold=_benchmark_summary(cold), priming=_benchmark_summary(priming),
                                      warm=_benchmark_summary(warm),
                                      warm_by_phase={phase: _benchmark_summary([sample for sample in warm if sample["phase"] == phase])
                                                     for phase in phase_names},
                                      complete_sequence=_benchmark_summary(priming + warm),
                                      trace=dict(cold=cold, priming=priming, warm=warm))
                        report["warm_gpu_budget_peak_fraction"] = report["warm"]["gauges_max"]["gpu_charged_bytes"] / gpu_budget
                        if output_dir is not None:
                            evidence = Path(output_dir).expanduser().resolve()
                            evidence.mkdir(parents=True, exist_ok=True)
                            _write_json(evidence / f"benchmark_{profile}_{scenario}_{mode}_{banks}_{policy}.json", report)
                        print("BENCHMARK " + json.dumps(report, sort_keys=True), flush=True)
        del x, scale, scales, routes, references, store, layers


BENCHMARK_PRESETS = ("disabled", "static_hot", "reuse", "lru")


def _locked_benchmark_schedule(scenario, iterations):
    if scenario == "cross_layer":
        raise ValueError("locked benchmark is one official op/layer; cross_layer is legacy only")
    if scenario == "hotset_transition":
        # Repeated A -> phase change B -> transient scan -> return A. Every
        # measured invocation gets its own route, not a relabelled fixed graph.
        block = ["hot_a"] * 3 + ["hot_b"] * 5 + ["uniform", "hot_a"]
        return [("transition", 0, block[n % len(block)]) for n in range(iterations)]
    schedule = _benchmark_schedule(scenario, iterations)
    return [schedule[n % len(schedule)] for n in range(iterations)]


def _benchmark_resources():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    controls = {}
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat",
                 "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                 "/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
        try:
            controls[path] = Path(path).read_text().strip()
        except OSError:
            controls[path] = None
    return dict(process_cpu_seconds=usage.ru_utime + usage.ru_stime,
                process_peak_rss_bytes=usage.ru_maxrss * 1024,
                affinity=sorted(os.sched_getaffinity(0)), cpu_controls=controls,
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                rss_scope="Linux process lifetime high-water, includes compilation/reference; not isolated case RSS")


def _locked_benchmark_reference(x, pair, ids, scale):
    # Independent real FP32 CPU reference for arbitrary E/H/I/K, no benchmark
    # op, model execution, fake output or GPU weight replica used as a reference.
    x, ids, scale = x.float().cpu(), ids.cpu(), scale.cpu()
    result = torch.zeros_like(x)
    for expert in range(pair[0].shape[0]):
        rows, slots = torch.where(ids == expert)
        if not rows.numel():
            continue
        gate, up = (x[rows] @ pair[0][expert].float().T).chunk(2, -1)
        value = (torch.nn.functional.silu(gate) * up) @ pair[1][expert].float().T
        result.index_add_(0, rows, value * scale[rows, slots, None])
    return result


@torch.no_grad()
def run_locked_benchmark(*, ops_library=None, iterations=40, wait_seconds=0,
                         benchmark_profile="all", benchmark_scenario="all",
                         max_route_tile_rows=512, nvtx=False, host_profile=False,
                         async_staging=False, allow_testing=False, output_dir=None,
                         cpp=False, case_filter=None, tile_m=0, tile_n=128):
    """Final frozen single-op AOTI wall measurements; legacy benchmark stays separate."""
    from custom_ops.moe_offload import InputBinding

    if output_dir is None:
        raise ValueError("--benchmark-locked requires --output-dir for durable raw evidence")
    if ops_library is None:
        raise ValueError("--benchmark-locked requires an explicit strict precompiled --ops-library")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    benchmark_started = time.perf_counter()
    benchmark_resources_before = _benchmark_resources()
    scenarios = ("fixed_hot", "uniform", "skewed_masked", "hotset_transition") if benchmark_scenario == "all" else (benchmark_scenario,)
    schedules = {scenario: _locked_benchmark_schedule(scenario, iterations) for scenario in scenarios}
    max_route_tile_rows = _parse_route_tile_rows(max_route_tile_rows)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen AOTI benchmarking")
    library = _load_ops(ops_library, wait_seconds, allow_testing=allow_testing)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    device = torch.device("cuda", torch.cuda.current_device())
    metadata_path = Path(str(library) + ".json")
    metadata = dict(environment_signature=_environment_signature(library),
                    ops_sha256=_sha256(library), ops_metadata=json.loads(metadata_path.read_text()),
                    ops_metadata_sha256=_sha256(metadata_path), resources=_benchmark_resources(),
                    toolchain=dict(CXX=os.environ.get("CXX", "c++"), CC=os.environ.get("CC", "cc"),
                                   torch_config=torch.__config__.show()),
                    benchmark_source_sha256=_sha256(__file__),
                    runner_source_sha256=_sha256(Path(__file__).with_name("fuse_moe_offload_aoti_runner.cpp")),
                    gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    shutil.copy2(__file__, directory / "benchmark_source.py")
    shutil.copy2(Path(__file__).with_name("fuse_moe_offload_aoti_runner.cpp"), directory / "benchmark_runner_source.cpp")
    executable = _compile_runner(directory) if cpp else None
    bindings = (InputBinding(1, "w1", 0), InputBinding(2, "w2", 0), InputBinding(5, "token"))
    completed_cases = []
    def region(label):
        return torch.cuda.nvtx.range(label) if nvtx else ExitStack()

    for profile, (s, h, intermediate, gpu_budget) in _benchmark_profiles(benchmark_profile):
        experts, k, seed = 16, 8, 20261012
        expert_bytes = 6 * h * intermediate
        cache_cap = 8 * ((expert_bytes + 511) // 512) * 512
        ram_budget = 128 * 1024**2
        pinned_budget = 4 * (experts + 1) + 2 * (2 * expert_bytes + 40)
        generator = torch.Generator().manual_seed(seed)
        store = ExpertStore({0: ((torch.randn(experts, 2 * intermediate, h, generator=generator) * .03).half(),
                                 (torch.randn(experts, h, intermediate, generator=generator) * .03).half())},
                            ram_budget_bytes=ram_budget)
        # Shared exact deterministic fixture also generated directly by C++.
        x = (torch.arange(s * h, dtype=torch.float32).cos() * .25).reshape(s, h).to(device, torch.float16)
        rows, slots = torch.arange(s).reshape(s, 1), torch.arange(k).reshape(1, k)
        host_ids = dict(hot_a=((rows + slots) % 8).int(), hot_b=(((rows + slots) % 8) + 8).int(),
                        uniform=((rows * k + slots) % experts).int())
        skewed = torch.where((rows + slots) % 11 == 0, (rows + slots) % 6, torch.zeros(s, k, dtype=torch.int64))
        skewed[(rows + 2 * slots) % 5 == 0] = -1
        skewed[::7] = -1
        host_ids["skewed_masked"] = skewed.int()
        routes = {name: ids.to(device) for name, ids in host_ids.items()}
        scales = {name: torch.full((s, k), 1 / k, device=device) for name in routes}
        scales["skewed_masked"] = torch.where(((rows + slots) % 3 == 0).to(device),
                                               -scales["skewed_masked"] * .5, scales["skewed_masked"] * 1.25)
        references = {name: _locked_benchmark_reference(x, store.layers[0], routes[name], scales[name]) for name in routes}
        reference_path = directory / f"locked_{profile}_references.pt"
        torch.save(dict(x=x.cpu(), ids=host_ids, scales={name: value.cpu() for name, value in scales.items()},
                        expected=references), reference_path)
        for mode in ("route_buffer", "fp32_accumulate", "auto"):
            for banks in (1, 2):
                for policy in BENCHMARK_PRESETS:
                    case_key = f"{mode}/bank{banks}/{policy}"
                    if case_filter is not None and case_filter != case_key:
                        continue
                    case_dir = directory / f"locked_{profile}_{mode}_{banks}_{policy}"
                    case_dir.mkdir(exist_ok=True)
                    options = dict(gpu_budget_bytes=gpu_budget, ram_budget_bytes=ram_budget,
                                   pinned_budget_bytes=pinned_budget, target_sm=10 * major + minor,
                                   num_banks=banks, activation_mode=mode, max_route_tile_rows=max_route_tile_rows,
                                   async_staging=async_staging, host_profile=host_profile, tile_m=tile_m, tile_n=tile_n)
                    if policy == "static_hot":
                        options.update(resident_cache_bytes=cache_cap, resident_experts={0: tuple(range(8))})
                    elif policy in ("reuse", "lru"):
                        options.update(dynamic_cache_bytes=cache_cap, cache_policy=policy,
                                       cache_min_reuse=2, cache_decay_interval=32, cache_update_bytes=cache_cap)
                    config = OffloadConfig.from_store(store, **options)
                    archive = case_dir / "weights.pt"
                    store.save_cpp(archive, weight_version=seed, config=config)
                    register_aoti_lowering()  # Before export AND compilation of every frozen op.
                    token = torch.zeros(2, dtype=torch.int64)
                    args = (x, *store.layers[0], routes["hot_a"], scales["hot_a"], token)
                    exported = torch.export.export(OffloadedMoELayer(0, config=config).eval(), args)
                    calls = [node for node in exported.graph.nodes if node.op == "call_function" and
                             node.target == torch.ops.recsys_ops.fuse_moe_offload.default]
                    assert len(calls) == 1 and tuple(calls[0].args[7]) == config.layer_config(0)
                    assert tuple(calls[0].args[8]) == config.manifest() and not exported.state_dict
                    compile_start = time.perf_counter()
                    package = _compile_checked_package(exported, package_path=str(case_dir / "model.pt2"),
                                                       inductor_configs={"triton.cudagraphs": False})
                    compilation_ms = (time.perf_counter() - compile_start) * 1000
                    load_start = time.perf_counter()
                    loaded = torch._inductor.aoti_load_package(package)
                    model_load_ms = (time.perf_counter() - load_start) * 1000
                    artifact = dict(package=str(package), package_sha256=_sha256(package), archive=str(archive),
                                    archive_sha256=_sha256(archive), weight_version=seed, archive_format_version=2,
                                    source_proof=str(case_dir / "model.source_proof" / "proof.json"),
                                    reference=str(reference_path), reference_sha256=_sha256(reference_path))
                    for scenario, schedule in schedules.items():
                        label = f"locked/{profile}/{scenario}/{case_key}/host_profile={int(host_profile)}"
                        before_resources = _benchmark_resources()
                        wall_started = time.perf_counter()
                        initialization, cold, warm = [], [], []
                        sample_sequence = 0
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        def initialize():
                            start = time.perf_counter()
                            runtime = _LockedPrefillMoERuntime.from_config(store, config, device=device)
                            torch.cuda.synchronize(device)
                            initialization.append(dict(host_ms=(time.perf_counter() - start) * 1000,
                                                       stats=runtime.stats(), resources=_benchmark_resources()))
                            return runtime
                        def measure(runtime, phase, route):
                            nonlocal sample_sequence
                            sample_sequence += 1
                            torch.cuda.synchronize(device)
                            usage = resource.getrusage(resource.RUSAGE_SELF)
                            cpu_start = usage.ru_utime + usage.ru_stime
                            output = None
                            try:
                                with ExitStack() as observers:
                                    if host_profile:
                                        observer = observers.enter_context(torch.profiler.profile(
                                            activities=[torch.profiler.ProfilerActivity.CPU]))
                                    with region(label + f"/{phase}/{route}"):
                                        start = time.perf_counter()
                                        output = runtime.invoke(loaded, (x, None, None, routes[route], scales[route], None),
                                                                artifact_config=config, bindings=bindings)
                                        torch.cuda.synchronize(device)
                                        elapsed = (time.perf_counter() - start) * 1000
                                usage = resource.getrusage(resource.RUSAGE_SELF)
                                sample = dict(phase=phase, route=route, host_ms=elapsed,
                                              cpu_ms=(usage.ru_utime + usage.ru_stime - cpu_start) * 1000,
                                              stats=runtime.stats(), status=runtime.status(), process_peak_rss_bytes=usage.ru_maxrss * 1024,
                                              gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                              gpu_max_reserved_bytes=torch.cuda.max_memory_reserved(device))
                                if host_profile:
                                    trace_path = case_dir / f"{scenario}_{sample_sequence}_{phase}_{route}.chrome.json"
                                    observer.export_chrome_trace(str(trace_path))
                                    sample["host_trace"] = str(trace_path)
                                    sample["host_stages"] = {event.key: dict(calls=event.count, cpu_ms=event.cpu_time_total / 1000)
                                                             for event in observer.key_averages()
                                                             if event.key.startswith("fuse_moe_offload::")}
                                actual = _single_output(output).float().cpu()
                                torch.testing.assert_close(actual, references[route], atol=2e-5, rtol=2e-2)
                                sample["reference_relative_l2_error"] = ((actual - references[route]).norm() /
                                                                        references[route].norm().clamp_min(1e-12)).item()
                                stats = sample["stats"]
                                assert sample["status"]["state"] == "healthy" and not sample["status"]["managed"]
                                assert stats["weight_h2d_bytes"] == stats["cache_misses"] * expert_bytes
                                assert stats["gpu_charged_bytes"] <= gpu_budget and stats["pinned_bytes"] <= pinned_budget
                                assert stats["ram_charged_bytes"] <= ram_budget
                                return sample
                            finally:
                                # A trace observer may retain this frame even after return/error.
                                # Keep CPU diagnostics, never the invocation's GPU output/history.
                                output = None
                                del output
                        prime = scenario if scenario in ("uniform", "skewed_masked") else "hot_a"
                        # First invocation is cold in every fresh runtime. No hidden kernel warmup.
                        for repeat in range(iterations):
                            with initialize() as runtime:
                                cold.append(measure(runtime, "cold", prime))
                        with initialize() as runtime:
                            priming = [measure(runtime, "priming", prime) for _ in range(3)]
                            for phase, _, route in schedule:
                                warm.append(measure(runtime, phase, route))
                        after_resources = _benchmark_resources()
                        report = dict(schema_version=3, case_version="T04.frozen_config.v1", backend="python_public_aoti",
                                      final_frozen=True, profile=profile, scenario=scenario, case_key=case_key,
                                      initialization_sample_count=len(initialization), cold_sample_count=len(cold),
                                      warm_sample_count=len(warm), gpu_peak_scope="case_since_reset_includes_initialization",
                                      frozen_config=dict(fields=asdict(config), manifest=config.manifest(), layer_config=config.layer_config(0)),
                                      metadata=metadata, artifact=artifact, shape=dict(S=s, H=h, I=intermediate, E=experts, K=k),
                                      seed=seed, route_source="deterministic_arange_hotsets_masked_phase_scan", synthetic_weights=True,
                                      host_profile=host_profile, nvtx=nvtx, timing_kind="diagnostic_trace" if host_profile or nvtx else "normal_wall",
                                      timer="perf_counter_invoke_preflight_begin_loader_check_device_sync", iterations=iterations,
                                      compilation_ms=compilation_ms, model_load_ms=model_load_ms,
                                      case_wall_ms=(time.perf_counter() - wall_started) * 1000,
                                      case_cpu_ms=(after_resources["process_cpu_seconds"] - before_resources["process_cpu_seconds"]) * 1000,
                                      resources_before=before_resources, resources_after=after_resources,
                                      initialization=_benchmark_latency(sample["host_ms"] for sample in initialization),
                                      cold=_benchmark_latency(sample["host_ms"] for sample in cold),
                                      cold_total_init_plus_invoke=_benchmark_latency(
                                          init["host_ms"] + call["host_ms"] for init, call in zip(initialization, cold)),
                                      warm=_benchmark_latency(sample["host_ms"] for sample in warm),
                                      warm_by_route={route: _benchmark_latency(sample["host_ms"] for sample in warm if sample["route"] == route)
                                                     for route in dict.fromkeys(sample["route"] for sample in warm)},
                                      weight_h2d_bytes={phase: dict(total=sum(sample["stats"]["weight_h2d_bytes"] for sample in samples),
                                                                   samples=[sample["stats"]["weight_h2d_bytes"] for sample in samples])
                                                        for phase, samples in (("cold", cold), ("priming", priming), ("warm", warm))},
                                      ram_host_peak_bytes=after_resources["process_peak_rss_bytes"],
                                      gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                      gpu_max_reserved_bytes=torch.cuda.max_memory_reserved(device),
                                      raw=dict(initialization=initialization, cold=cold, priming=priming, warm=warm),
                                      budget_note="same existing hardware budgets; charged stats separate from allocator/RSS high-water",
                                      workload_note="synthetic dominance intent, not a speedup promise; validation outside invocation timer")
                        _write_json(case_dir / f"{scenario}.json", report)
                        print("BENCHMARK " + json.dumps(report, sort_keys=True), flush=True)
                        if executable is not None:
                            command = [str(executable), "--locked-benchmark", str(package), str(library), str(device.index),
                                       str(archive), str(s), str(k), str(iterations), prime,
                                       *(route for _, _, route in schedule), "--output-dir", str(case_dir / f"cpp_{scenario}")]
                            result = _run_process(command, case_dir, f"cpp_{scenario}")
                            cpp_reports = [json.loads(line.removeprefix("BENCHMARK ")) for line in result.stdout.splitlines()
                                           if line.startswith("BENCHMARK ")]
                            assert len(cpp_reports) == 1
                            cpp_report = cpp_reports[0]
                            for phase in ("cold", "priming", "warm"):
                                for sample in cpp_report["raw"][phase]:
                                    sample["stats"] = dict(zip(STATS_FIELDS, sample["stats_values"]))
                            cpp_report["weight_h2d_bytes"] = {
                                phase: dict(total=sum(sample["stats"]["weight_h2d_bytes"] for sample in samples),
                                            samples=[sample["stats"]["weight_h2d_bytes"] for sample in samples])
                                for phase, samples in ((phase, cpp_report["raw"][phase]) for phase in ("cold", "priming", "warm"))}
                            cpp_report.update(artifact=artifact, metadata=metadata, frozen_config=report["frozen_config"],
                                              profile=profile, scenario=scenario, case_key=case_key,
                                              parent_process_wall_ms=json.loads((case_dir / f"cpp_{scenario}.exit.json").read_text())["elapsed_ms"])
                            _write_json(case_dir / f"cpp_{scenario}.json", cpp_report)
                        completed_cases.append(dict(profile=profile, scenario=scenario, case_key=case_key))
                    del loaded, exported
                    torch._dynamo.reset()
                    gc.collect()
        del x, routes, scales, references, store
    if not completed_cases:
        raise ValueError("benchmark case filter matched no cases; use mode/bankN/policy")
    _write_json(directory / "locked_benchmark_index.json", dict(schema_version=1, cases=completed_cases,
                configuration_count_per_profile=24,
                selected_configuration_count_per_profile=1 if case_filter else 24,
                scenario_count=len(scenarios), completed_case_count=len(completed_cases),
                backend_count=2 if cpp else 1, final_frozen=True,
                benchmark_wall_ms=(time.perf_counter() - benchmark_started) * 1000,
                resources_before=benchmark_resources_before, resources_after=_benchmark_resources(),
                wall_scope="entire locked benchmark including strict load/reference/export/compile/Python/C++/evidence"))


def test_locked_chain_fixture_magnitude_cpu_contract():
    # Reproduce v17 checkpoint draws/conversion, without checkpoint/archive loads,
    # native loading, runtime construction or CUDA. Do not change the legacy seed.
    generator = torch.Generator().manual_seed(20261012 + 17)
    w10 = (torch.randn(E, 2 * I, H, generator=generator) * .12).half()
    w20 = (torch.randn(E, H, I, generator=generator) * .12).half()
    gate5 = (torch.randn(E, I, H, generator=generator) * .12).half()
    up5 = (torch.randn(E, I, H, generator=generator) * .12).half()
    w25 = (torch.randn(E, H, I, generator=generator) * .12).half()
    checkpoint = ExpertStore({0: (w10, w20), 5: (torch.cat((gate5, up5), dim=1), w25)},
                             ram_budget_bytes=RAM_BUDGET)
    originals = {layer: tuple(w.clone() for w in pair) for layer, pair in checkpoint.layers.items()}
    versions = {layer: tuple(w._version for w in pair) for layer, pair in checkpoint.layers.items()}
    with mock.patch.object(ops, "ensure_loaded", side_effect=AssertionError("native loading")), \
            mock.patch.object(ExpertStore, "load", side_effect=AssertionError("checkpoint loading")), \
            mock.patch.object(ExpertStore, "load_cpp", side_effect=AssertionError("archive loading")), \
            mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("CUDA detection")):
        single = _locked_store(checkpoint, False)
        chain = _locked_store(checkpoint, True)
        maxima = _check_locked_chain_fixture_cpu(chain)
    assert set(maxima) == {2, 7, 64} and min(maxima.values()) > 1e-4, maxima
    for layer, pair in checkpoint.layers.items():
        assert tuple(w._version for w in pair) == versions[layer]
        for index, weight in enumerate(pair):
            torch.testing.assert_close(weight, originals[layer][index], atol=0, rtol=0)
            assert chain.layers[layer][index].data_ptr() != weight.data_ptr()
            expected = weight * (4 if (layer, index) == (5, 1) else 1)
            torch.testing.assert_close(chain.layers[layer][index], expected, atol=0, rtol=0)
    for index, weight in enumerate(single.layers[0]):
        assert weight.data_ptr() != checkpoint.layers[0][index].data_ptr()
        torch.testing.assert_close(weight, originals[0][index], atol=0, rtol=0)
    print(f"PASS LOCKED chain isolated CPU fixture S=2/7/64 max abs: {maxima}", flush=True)


def test_locked_archive_and_adapter_contract():
    # No native loading, CUDA detection or compilation: validate the real format2
    # archive and graph adapter constants, including the unpatched class alias.
    generator = torch.Generator().manual_seed(20261029)
    store = ExpertStore({layer: (torch.randn(E, 2 * I, H, generator=generator).half(),
                                 torch.randn(E, H, I, generator=generator).half())
                         for layer in (0, 5)}, ram_budget_bytes=RAM_BUDGET)
    with mock.patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError("online SM detection")), \
            mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError("online device detection")):
        config = OffloadConfig.from_store(
            store, gpu_budget_bytes=2 * 1024**2, ram_budget_bytes=RAM_BUDGET,
            pinned_budget_bytes=PINNED_BUDGET, target_sm=89,
            resident_experts={0: (0,), 5: (0,)}, resident_cache_bytes=2 * DYNAMIC_CACHE_CAP,
            dynamic_cache_bytes=2 * DYNAMIC_CACHE_CAP, activation_mode="fp32_accumulate",
            max_route_tile_rows="auto", tile_m=32, tile_n=64)
        with tempfile.TemporaryDirectory(prefix="locked_archive_contract_") as temporary:
            path = Path(temporary) / "weights.pt"
            store.save_cpp(path, config=config, weight_version=17)
            restored = ExpertStore.load_cpp(path)
            assert restored.deployment_config == config and restored.weight_version == 17
            buffers = dict(torch.jit.load(str(path)).named_buffers())
            assert buffers["format_version"].tolist() == [2]
            for layer in (0, 5):
                vector = config.layer_config(layer)
                assert buffers[f"offload_config_{layer}"].tolist() == list(vector)
                assert len(vector) == 23 + vector[21] and vector[15:17] == (32, 64)
            model = _LockedChain(config)
            assert model.first.deployment_config == config.layer_config(0)
            assert model.second.deployment_config == config.layer_config(5)
            manifest = config.manifest()
            expected_manifest = (1, 2, *(value for layer in (0, 5)
                                  for value in (layer, len(config.layer_config(layer)), *config.layer_config(layer))))
            assert manifest == expected_manifest
            assert model.first.deployment_manifest == model.second.deployment_manifest == manifest
            assert not list(model.named_buffers()) and not list(model.named_parameters())
            vectors = [config.layer_config(layer) for layer in (0, 5)]
            serialized = {"nodes": [{"inputs": [{"arg": {"as_ints": list(v)}}]} for v in vectors]}
            assert list(_serialized_int_vectors(json.loads(json.dumps(serialized)))) == vectors
            original_alias = _LockedPrefillMoERuntime
            with mock.patch(__name__ + ".PrefillMoERuntime", partial(original_alias,
                            async_staging=True, host_profile=True)):
                assert _LockedPrefillMoERuntime is original_alias
                assert callable(_LockedPrefillMoERuntime.from_config)
            # Invalid count/length is rejected by the actual Python archive reader.
            archive = torch.jit.load(str(path))
            archive.offload_config_0[21] += 1
            torch.jit.save(archive, str(path))
            try:
                ExpertStore.load_cpp(path)
            except ValueError as error:
                assert "configuration" in str(error)
            else:
                raise AssertionError("invalid locked archive count/length accepted")


def test_public_reference_storage_rounding_cpu_contract():
    # Hand-computed diagonal CPU case distinguishes both storage boundaries.
    # This is not a replacement/rescaling of the GPU acceptance fixture.
    x = torch.tensor([[.2]], dtype=torch.float16)
    w1 = torch.tensor([[[.3], [.7]]], dtype=torch.float16)
    w2 = torch.tensor([[[.9]]], dtype=torch.float16)
    ids = torch.tensor([[0]], dtype=torch.int32)
    scale = torch.tensor([[.75]])
    gate = x.float() * w1[0, 0].float()
    up = x.float() * w1[0, 1].float()
    act = torch.nn.functional.silu(gate) * up
    down = act.half().float() * w2[0, 0].float()
    expected = down.half().float() * scale
    actual = _public_layer_reference(x, w1, w2, ids, scale, rounded=True)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, down * scale), "Down must round before route scale"
    mathematical = _public_layer_reference(x, w1, w2, ids, scale, rounded=False)
    assert torch.equal(mathematical, act * w2[0, 0].float() * scale)
    assert not torch.equal(mathematical, actual), "math and native dtype references must stay distinct"


def test_public_invoke_mandatory_per_call_contract():
    # CPU-only PUBLIC adapter protocol, not evidence about compiled AOTI errors.
    # Native calls are observed; the runtime under test is never replaced.
    from custom_ops.moe_offload import InputBinding

    store = ExpertStore({0: _weights()}, ram_budget_bytes=RAM_BUDGET)
    config = OffloadConfig.from_store(
        store, gpu_budget_bytes=GPU_BUDGET, ram_budget_bytes=RAM_BUDGET,
        pinned_budget_bytes=PINNED_BUDGET, target_sm=89, tile_m=32, tile_n=64)
    token = torch.tensor([123, 0])
    bindings = [InputBinding(1, "token"), InputBinding(3, "w1", 0), InputBinding(5, "w2", 0)]
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)))
        stack.enter_context(mock.patch.object(ops, "ensure_loaded"))
        stack.enter_context(mock.patch.object(ops, "is_available", return_value=True))
        for name in ("bind", "close", "validate", "begin", "check"):
            stack.enter_context(mock.patch.object(torch.ops.recsys_ops, f"fuse_moe_offload_{name}", create=True))
        stack.enter_context(mock.patch.object(torch.ops.recsys_ops, "fuse_moe_offload_create",
                                              create=True, return_value=token))
        with _LockedPrefillMoERuntime.from_config(store, config, device="cuda:0") as runtime:
            business = torch.arange(4)
            raw = [business, None, business, None, business, None]
            prepared = runtime.prepare_inputs(raw, bindings)
            assert raw[1] is None and prepared[0] is business and prepared[2] is business
            assert prepared[1] is token and prepared[3] is store.layers[0][0] and prepared[5] is store.layers[0][1]
            events = []
            native = torch.ops.recsys_ops
            native.fuse_moe_offload_validate.side_effect = lambda *args: events.append("validate")
            native.fuse_moe_offload_begin.side_effect = lambda *args: events.append("begin")
            native.fuse_moe_offload_check.side_effect = lambda *args: events.append("check")
            output = torch.ones(1)
            def callback(*args):
                events.append("callback")
                return (output,)
            for _ in range(2):
                assert runtime.invoke(callback, prepared, artifact_config=config, bindings=bindings) == (output,)
            assert events == ["validate", "begin", "callback", "check"] * 2, events
            missing = mock.Mock()
            try:
                runtime.invoke(missing, prepared, bindings=bindings)
            except TypeError:
                pass
            else:
                raise AssertionError("explicit artifact_config omitted but accepted")
            missing.assert_not_called()
            failure = RuntimeError("unrelated callback failure")
            def fail(*args):
                raise failure
            events.clear()
            try:
                runtime.invoke(fail, prepared, artifact_config=config, bindings=bindings)
            except RuntimeError as error:
                assert error is failure
            else:
                raise AssertionError("callback exception swallowed")
            assert events == ["validate", "begin", "check"], events
            native.fuse_moe_offload_validate.side_effect = RuntimeError("deployment configuration mismatch")
            events.clear()
            try:
                runtime.invoke(missing, prepared, artifact_config=config, bindings=bindings)
            except RuntimeError as error:
                assert "configuration mismatch" in str(error)
            else:
                raise AssertionError("preflight failure accepted")
            missing.assert_not_called()
            assert not events, "begin/check must not run after rejected preflight"


def test_locked_target_sm_precheck_contract():
    store = ExpertStore({0: _weights()}, ram_budget_bytes=RAM_BUDGET)
    config = OffloadConfig.from_store(
        store, gpu_budget_bytes=2 * 1024**2, ram_budget_bytes=RAM_BUDGET,
        pinned_budget_bytes=PINNED_BUDGET, target_sm=89, tile_m=32, tile_n=64)
    with mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)) as capability, \
            mock.patch.object(ops, "ensure_loaded") as load, \
            mock.patch.object(torch.ops.recsys_ops, "fuse_moe_offload_create", create=True) as create:
        try:
            _LockedPrefillMoERuntime.from_config(store, config, device="cuda:0")
        except ValueError as error:
            assert "deployment configuration mismatch: target_sm" in str(error)
        else:
            raise AssertionError("locked target SM mismatch accepted")
        capability.assert_called_once_with(torch.device("cuda:0"))
        load.assert_not_called()
        create.assert_not_called()


def test_acceptance_explicit_runtime_flags_contract():
    # CPU-only: ensure nested helpers receive flags and the original constructor
    # (including its return object) is restored even when acceptance fails.
    for async_staging in (False, True):
        for host_profile in (False, True):
            with mock.patch(__name__ + ".PrefillMoERuntime") as original:
                def acceptance(**kwargs):
                    assert kwargs["async_staging"] is async_staging
                    assert kwargs["host_profile"] is host_profile
                    result = PrefillMoERuntime("store", device="cuda:0")
                    original.assert_called_once_with(
                        "store", device="cuda:0", async_staging=async_staging,
                        host_profile=host_profile)
                    assert result is original.return_value
                    return result

                with mock.patch(__name__ + "._run_acceptance", side_effect=acceptance):
                    assert run_acceptance(async_staging=async_staging,
                                          host_profile=host_profile) is original.return_value
                assert PrefillMoERuntime is original
                with mock.patch(__name__ + "._run_acceptance", side_effect=RuntimeError("failure")):
                    try:
                        run_acceptance(async_staging=async_staging, host_profile=host_profile)
                    except RuntimeError as error:
                        assert str(error) == "failure"
                    else:
                        raise AssertionError("acceptance failures must propagate")
                assert PrefillMoERuntime is original
            flags = _runtime_cli_flags(async_staging, host_profile)
            assert ("--async-staging" in flags) is async_staging
            assert ("--host-profile" in flags) is host_profile
            assert len(flags) == int(async_staging) + int(host_profile)


def test_strict_precompiled_public_loader_contract():
    # Validate loading order/policy without dlopen, native compilation or CUDA.
    class Schema:
        _schema = "recsys_ops::fuse_moe_offload(Tensor(a!) runtime_token) -> Tensor"
    with tempfile.TemporaryDirectory(prefix="strict_loader_contract_") as temporary:
        path = Path(temporary) / "ops.so"
        path.touch()
        events = []
        fake_native = mock.Mock()
        fake_native.fuse_moe_offload.default = Schema()
        fake_native.fuse_moe_offload_create.default._schema = (
            "fuse_moe_offload_create(int device, int gpu_budget_bytes=0, int ram_budget_bytes=0, "
            "int pinned_budget_bytes=0, int num_banks=2, int activation_mode=0, "
            "int max_route_tile_rows=4096, int resident_cache_bytes=0, int dynamic_cache_bytes=0, "
            "int cache_policy=0, int cache_decay_interval=32, int cache_min_reuse=2, "
            "int cache_update_bytes=0, bool async_staging=False, bool host_profile=False, "
            "int tile_m=0, int tile_n=128) -> Tensor")
        fake_native.fuse_moe_offload_resident_bind.default._schema = (
            "fuse_moe_offload_resident_bind(Tensor token, int layer_id, int[] expert_ids) -> ()")
        with mock.patch.object(ops, "load_precompiled", side_effect=lambda *args, **kwargs: events.append("strict")) as strict, \
                mock.patch.object(ops, "ensure_loaded", side_effect=lambda *args: events.append("ensure")), \
                mock.patch.object(ops, "is_available", return_value=True), \
                mock.patch.object(torch.ops, "recsys_ops", fake_native), \
                mock.patch(__name__ + ".register_aoti_lowering",
                           side_effect=lambda: events.append("lowering")) as lowering:
            assert _load_ops(path, allow_testing=True) == path.resolve()
            strict.assert_called_once_with("fuse_moe", path.resolve(), allow_testing=True)
            lowering.assert_called_once_with()
            assert events == ["strict", "ensure", "lowering"], events
        with mock.patch.object(ops, "load_precompiled", side_effect=RuntimeError("missing path.json")), \
                mock.patch.object(ops, "ensure_loaded") as ensure:
            try:
                _load_ops(path)
            except RuntimeError as error:
                assert "path.json" in str(error)
            else:
                raise AssertionError("strict metadata failure swallowed")
            ensure.assert_not_called()


def test_compile_checked_package_public_entry_contract():
    # Mock the public boundary, not the audited official function signature.
    # Cache disabling belongs to compile_aoti; source proof remains mandatory here.
    op = object()
    exported = mock.Mock()
    exported.graph.nodes = [mock.Mock(op="call_function", target=op, args=(), kwargs={})]
    configs = {"triton.cudagraphs": False}
    events = []
    native = mock.Mock()
    native.fuse_moe_offload.default = op
    with mock.patch.object(torch.ops, "recsys_ops", native), \
            mock.patch(__name__ + ".compile_aoti", side_effect=lambda *args, **kwargs:
                       events.append("compile") or "checked.pt2") as compile_package, \
            mock.patch(__name__ + "._verify_checked_package", side_effect=lambda *args, **kwargs:
                       events.append("proof")) as verify:
        assert _compile_checked_package(exported, package_path="checked.pt2",
                                        inductor_configs=configs) == "checked.pt2"
    compile_package.assert_called_once_with(exported, package_path="checked.pt2",
                                            inductor_configs=configs)
    verify.assert_called_once_with("checked.pt2", expected_calls=1)
    assert events == ["compile", "proof"]
    assert configs == {"triton.cudagraphs": False}


def test_checked_proxy_source_contract():
    # Definitions only until lead authorizes tests: no torch/native execution.
    checked = "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_proxy_executor_call_function(proxy_executor, 3, 0, nullptr, 0, nullptr));"
    unchecked = "aoti_torch_proxy_executor_call_function(proxy_executor, 7, 0, nullptr, 0, nullptr);"
    source = ("// " + unchecked + "\n" + checked + "\n" + unchecked +
              "\nAOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_some_aten_call());")
    calls = _checked_proxy_proof(source, {3, 7})
    assert [(call["extern_index"], call["checked"]) for call in calls] == [(3, True), (7, False)]
    assert [call["line"] for call in calls] == [2, 3]
    assert _checked_proxy_proof(source, {42}) == []


def test_locked_benchmark_matrix_and_schedule_cpu_contract():
    # CPU-only definitions; no artifact/native/GPU execution is needed.
    matrix = [(mode, banks, policy) for mode in ("route_buffer", "fp32_accumulate", "auto")
              for banks in (1, 2) for policy in BENCHMARK_PRESETS]
    assert len(matrix) == len(set(matrix)) == 24
    sequence = _locked_benchmark_schedule("hotset_transition", 40)
    assert len(sequence) == 40 and all(layer == 0 for _, layer, _ in sequence)
    assert [route for _, _, route in sequence[:10]] == ["hot_a"] * 3 + ["hot_b"] * 5 + ["uniform", "hot_a"]
    for scenario in ("fixed_hot", "uniform", "skewed_masked", "hot_drift", "scan_return"):
        assert len(_locked_benchmark_schedule(scenario, 40)) == 40
    try:
        _locked_benchmark_schedule("cross_layer", 40)
    except ValueError:
        pass
    else:
        raise AssertionError("single-op locked benchmark cannot use multiple layers")


def test_benchmark_raw_samples_contract():
    values = [3.0, 1.0, 2.0]
    report = _benchmark_latency(iter(values))
    assert report["samples_ms"] == values and report["samples"] == 3
    assert report["latency_ms_median"] == 2.0 and report["latency_ms_p95"] == 3.0


def test_host_stage_summary_contract():
    stats = dict.fromkeys(STATS_FIELDS, 0)
    stats.update(active_experts=1, cache_misses=1, num_banks=1, wave_capacity=1)
    base = dict(stats=stats, host_ms=1.0, allocator_peak_delta_bytes=0,
                allocator_allocated_bytes=0, allocator_reserved_bytes=0)
    samples = [dict(base, host_stages={
        "host_weight_copy": dict(calls=2, cpu_ms=ms),
        "host_ready_wait": dict(calls=2, cpu_ms=ms / 2),
        "host_bank_wait": dict(calls=1, cpu_ms=ms / 4)})
        for ms in (.2, .4)] + [dict(base, host_stages={})]
    summary = _benchmark_summary(samples)
    stage = summary["host_stages"]["host_weight_copy"]
    assert stage["samples"] == 3 and stage["calls_total"] == 4, stage
    assert stage["latency_ms_median"] == .2 and stage["latency_ms_p95"] == .4, stage
    assert math.isclose(stage["latency_ms_total"], .6), stage
    for name, calls, total in (("host_ready_wait", 4, .3), ("host_bank_wait", 2, .15)):
        wait = summary["host_stages"][name]
        assert wait["samples"] == 3 and wait["calls_total"] == calls, wait
        assert math.isclose(wait["latency_ms_total"], total), wait
    assert "host_stages" not in _benchmark_summary([base, base]), summary
    try:
        _benchmark_summary([base, samples[0]])
    except ValueError as error:
        assert "mix" in str(error), error
    else:
        raise AssertionError("mixed profiling samples must fail explicitly")


def test_route_tile_auto_frontend_contract():
    # No CUDA allocation/native build: exercise the real frontend mapping only.
    # Defaults and activation auto semantics must remain independent of tile auto.
    store = ExpertStore({0: _weights()}, ram_budget_bytes=RAM_BUDGET)
    for requested, native_rows in ((4096, 4096), ("auto", 0)):
        with mock.patch.object(torch.ops.recsys_ops, "fuse_moe_offload_create", create=True,
                               return_value=torch.tensor([123, 0])) as create, \
                mock.patch.object(torch.ops.recsys_ops, "fuse_moe_offload_bind", create=True), \
                mock.patch.object(torch.ops.recsys_ops, "fuse_moe_offload_close", create=True), \
                mock.patch.object(ops, "ensure_loaded"), \
                mock.patch.object(ops, "is_available", return_value=True):
            tile_options = {} if requested == 4096 else {"max_route_tile_rows": requested}
            with PrefillMoERuntime(store, device="cuda:0", activation_mode="auto", **tile_options):
                args = create.call_args.args
                assert args[4:7] == (2, 2, native_rows), args
                assert args[7:] == (0, 0, 0, 32, 2, 0, False, False, 0, 128), args
            create.reset_mock()
            try:
                PrefillMoERuntime(store, device="cuda:0", max_route_tile_rows=0)
            except ValueError:
                pass
            else:
                raise AssertionError("Python numeric 0 must be rejected; use auto")
            create.assert_not_called()


def test_deployment_native_loader_release_cpu_contract():
    # Test both reviewed schemas with injected dependencies, never mutate global
    # torch.__version__ or cache a constructor decision across releases.
    schemas = (
        ("2.6.0a0", "__init__(self, arg0: str, arg1: str) -> None", ("bundle.pt2", "model")),
        ("2.11.0+cu130", "__init__(self, path: str, model_name: str, run_single_threaded: bool = False, num_runners: int = 1, device_index: int = -1) -> None",
         ("bundle.pt2", "model", False, 1, -1)),
        # Exact remote 2.11 binding doc; SupportsInt is the reviewed pybind
        # integer spelling, not permission to accept arbitrary annotations.
        ("2.11.0+cu130", "__init__(self: torch._C._aoti.AOTIModelPackageLoader, arg0: str, arg1: str, arg2: bool, arg3: typing.SupportsInt, arg4: typing.SupportsInt) -> None",
         ("bundle.pt2", "model", False, 1, -1)),
    )
    for version, doc, expected in schemas:
        class Loader:
            def __init__(self, *args):
                self.arguments = args
        Loader.__init__.__doc__ = doc
        assert _deployment_native_loader("bundle.pt2", torch_version=version,
                                         loader_type=Loader).arguments == expected
        failure = RuntimeError("original constructor failure")
        with mock.patch.object(Loader, "__init__", side_effect=failure) as constructor:
            constructor.__doc__ = doc
            try:
                _deployment_native_loader("bundle.pt2", torch_version=version, loader_type=Loader)
            except RuntimeError as error:
                assert error is failure
            else:
                raise AssertionError("constructor error swallowed")
            constructor.assert_called_once_with(*expected)
    for version in ("2.10.0", "2.11.1", "2.6.0", "2.11.0rc1"):
        loader = mock.Mock()
        try:
            _deployment_native_loader("bundle.pt2", torch_version=version, loader_type=loader)
        except RuntimeError as error:
            assert "unsupported reviewed torch release" in str(error)
        else:
            raise AssertionError("unsupported release accepted")
        loader.assert_not_called()
    class WrongSchema:
        def __init__(self):
            raise AssertionError("mismatched constructor must not run")
    try:
        _deployment_native_loader("bundle.pt2", torch_version="2.11.0", loader_type=WrongSchema)
    except RuntimeError as error:
        assert "schema mismatch" in str(error)
    else:
        raise AssertionError("mismatched schema accepted")
    for annotation in ("float", "typing.SupportsIndex", "typing.SupportsIntSubclass"):
        for parameter in ("arg3", "arg4"):
            class RejectedAnnotation:
                def __init__(self, *args):
                    raise AssertionError("unreviewed annotation must reject before constructor")
            RejectedAnnotation.__init__.__doc__ = schemas[-1][1].replace(
                f"{parameter}: typing.SupportsInt", f"{parameter}: {annotation}")
            try:
                _deployment_native_loader("bundle.pt2", torch_version="2.11.0+cu130",
                                          loader_type=RejectedAnnotation)
            except RuntimeError as error:
                assert "schema mismatch" in str(error)
            else:
                raise AssertionError("unreviewed integer annotation accepted")
    # Inspect the actually installed official binding without loading a package.
    native = torch._C._aoti.AOTIModelPackageLoader
    class DocumentedLoader:
        def __init__(self, *args):
            self.arguments = args
    DocumentedLoader.__init__.__doc__ = native.__init__.__doc__
    actual = _deployment_native_loader("bundle.pt2", loader_type=DocumentedLoader)
    assert actual.arguments in (("bundle.pt2", "model"), ("bundle.pt2", "model", False, 1, -1))


def test_compile_native_runner_only_cpu_contract():
    with tempfile.TemporaryDirectory(prefix="compile_runner_contract_") as temporary:
        directory = Path(temporary)
        def compile_runner(destination):
            assert destination == directory.resolve()
            executable = destination / "fuse_moe_offload_aoti_runner"
            executable.write_bytes(b"CPU test fixture, not compiled evidence")
            _write_json(destination / "cpp_compile.exit.json", dict(command=["c++", "source.cpp"], returncode=0))
            _write_json(destination / "cpp_source.json", dict(source="source.cpp", source_sha256="fixture"))
            return executable
        with mock.patch(__name__ + "._compile_runner", side_effect=compile_runner) as compile_call, \
                mock.patch(__name__ + "._load_ops", side_effect=AssertionError("ops load forbidden")), \
                mock.patch(__name__ + ".register_aoti_lowering", side_effect=AssertionError("lowering forbidden")), \
                mock.patch(__name__ + "._environment_signature", side_effect=AssertionError("GPU probe forbidden")), \
                mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU probe forbidden")):
            assert _run_compile_native_runner_only(directory) == directory / "fuse_moe_offload_aoti_runner"
        compile_call.assert_called_once_with(directory.resolve())
        report = json.loads((directory / "compile_report.json").read_text())
        assert report["state"] == "passed" and report["returncode"] == 0
        assert report["executable_sha256"] == _sha256(directory / "fuse_moe_offload_aoti_runner")
        assert report["command"] == ["c++", "source.cpp"]
        failure = subprocess.CalledProcessError(17, ["c++", "bad.cpp"])
        def failed_compile(destination):
            _write_json(destination / "cpp_compile.exit.json", dict(command=failure.cmd, returncode=17))
            raise failure
        with mock.patch(__name__ + "._compile_runner", side_effect=failed_compile):
            try:
                _run_compile_native_runner_only(directory)
            except subprocess.CalledProcessError as error:
                assert error is failure
            else:
                raise AssertionError("compiler failure swallowed")
        report = json.loads((directory / "compile_report.json").read_text())
        assert report["state"] == "failed" and report["returncode"] == 17
        assert report["command"] == failure.cmd


def test_fuse_moe_offload_aoti():
    run_acceptance()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-library", help="Strict precompiled ops.so; matching ops.so.json is required")
    parser.add_argument("--allow-testing", action="store_true",
                        help="Explicitly accept testing build metadata and run optional fault hooks")
    parser.add_argument("--output-dir", help="Keep graph packages, executable, raw tensors and JSON/exit evidence")
    parser.add_argument("--compile-native-runner-only", action="store_true",
                        help="Offline compile only the standalone C++ runner; requires output-dir, no GPU execution or ops library")
    parser.add_argument("--public-only", action="store_true", help="Only new public-op T01/T02/T05/T06/T08 acceptance")
    parser.add_argument("--deployment-only", action="store_true",
                        help="Load an intact prelinked official PT2 in a fresh process; forbid external native source reads/compilation/tools")
    parser.add_argument("--public-package", help="Existing public PT2 for deployment-only or isolated negative acceptance")
    parser.add_argument("--public-archive", help="Matching public format2 weights archive")
    parser.add_argument("--public-diagnose", metavar="SAVED_TENSORS",
                        help="Replay saved inputs; report per-layer FP32/rounded/native/legacy differences without compilation")
    parser.add_argument("--public-error-layer", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--public-fault", choices=["after_first_h2d", "after_kernel_submit", "thread_create"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--async-staging", action="store_true",
                        help="Enable explicit asynchronous host staging (default off)")
    parser.add_argument("--host-profile", action="store_true",
                        help="Enable explicit native host profiling (default off)")
    parser.add_argument("--nvtx", action="store_true",
                        help="Enable explicitly labelled diagnostic NVTX ranges; normal benchmark defaults off")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 7, 64])
    parser.add_argument("--wide-bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="Also export/check the separate BF16 H128/I64 package (default on); "
                             "FP16 wide Python/C++ acceptance is always required")
    parser.add_argument("--benchmark-cpp", action="store_true",
                        help="Export H128/I64 once and run standalone C++ AOTI 4096/auto comparison")
    parser.add_argument("--benchmark-cpp-sizes", type=int, nargs="+", default=[64, 8192, 16384])
    parser.add_argument("--benchmark-locked", action="store_true",
                        help="Final frozen single-op AOTI benchmark; requires output-dir; 24 configurations/profile/scenario")
    parser.add_argument("--benchmark-locked-cpp", action="store_true",
                        help="Also measure Python-free public C++ invoke on each identical frozen PT2/archive")
    parser.add_argument("--benchmark-case", help="Independent trace subset: route_buffer|fp32_accumulate|auto/bank1|bank2/disabled|static_hot|reuse|lru")
    parser.add_argument("--benchmark-tile-m", type=int, choices=[0, 32, 64, 128], default=0)
    parser.add_argument("--benchmark-tile-n", type=int, choices=[64, 128], default=128)
    parser.add_argument("--benchmark", action="store_true",
                        help="Historical unfrozen eager-native benchmark; NOT final T04 evidence (use --benchmark-locked)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Cold samples and warm sequence length (locked default 40, legacy 20); priming recorded separately")
    parser.add_argument("--benchmark-profile", default="all",
                        choices=["all", "balanced", "small", "weight_transfer", "activation_near_budget"])
    parser.add_argument("--benchmark-scenario", default="all",
                        choices=["all", "fixed_hot", "uniform", "hot_drift", "scan_return", "cross_layer",
                                 "skewed_masked", "hotset_transition"])
    parser.add_argument("--benchmark-route-tile-rows", type=_parse_route_tile_rows, default=512,
                        help="Route tile row cap or auto (native 0); native planning may shrink it "
                             "to fit unchanged budgets (runtime default is 4096)")
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Wait for an externally built .so to appear; never start a JIT build")
    options = parser.parse_args()
    if options.compile_native_runner_only:
        if not options.output_dir:
            parser.error("--compile-native-runner-only requires --output-dir")
        if any((options.ops_library, options.allow_testing, options.public_only,
                options.deployment_only, options.public_package, options.public_archive,
                options.public_diagnose, options.public_fault, options.benchmark,
                options.benchmark_cpp, options.benchmark_locked, options.benchmark_locked_cpp,
                options.benchmark_case, options.async_staging, options.host_profile, options.nvtx)):
            parser.error("compile-native-runner-only cannot be combined with ops/public/deployment/benchmark execution modes")
        _run_compile_native_runner_only(options.output_dir)
        raise SystemExit(0)  # Deliberately bypass GPU-probing _cli_evidence.
    if sum((options.benchmark_cpp, options.benchmark, options.benchmark_locked)) > 1:
        parser.error("choose one of --benchmark, --benchmark-cpp or --benchmark-locked")
    if options.benchmark_locked_cpp and not options.benchmark_locked:
        parser.error("--benchmark-locked-cpp requires --benchmark-locked")
    if options.benchmark_locked and not (options.output_dir and options.ops_library):
        parser.error("--benchmark-locked requires --output-dir and strict precompiled --ops-library")
    if options.benchmark_case and not options.benchmark_locked:
        parser.error("--benchmark-case is a locked benchmark subset selector")
    if options.benchmark_case and options.benchmark_case not in {
            f"{mode}/bank{banks}/{policy}" for mode in ("route_buffer", "fp32_accumulate", "auto")
            for banks in (1, 2) for policy in BENCHMARK_PRESETS}:
        parser.error("invalid --benchmark-case; expected e.g. fp32_accumulate/bank2/reuse")
    if options.benchmark_locked and (options.public_package or options.public_only or options.public_fault):
        parser.error("locked benchmark cannot be combined with public acceptance/child/fault modes")
    if options.benchmark_locked and options.benchmark_scenario == "cross_layer":
        parser.error("locked benchmark contains exactly one official op; cross_layer is legacy only")
    if options.benchmark and options.benchmark_scenario == "hotset_transition":
        parser.error("hotset_transition requires --benchmark-locked")
    if options.iterations is None:
        options.iterations = 40 if options.benchmark_locked else 20
    if options.public_package and not (options.public_archive and options.output_dir and options.ops_library):
        parser.error("public child requires package/archive/output-dir/ops-library")
    if options.deployment_only and not options.public_package:
        parser.error("--deployment-only requires --public-package/--public-archive/--ops-library/--output-dir")
    if options.deployment_only and (options.public_fault or options.benchmark or options.benchmark_cpp or options.benchmark_locked or options.public_only):
        parser.error("deployment-only cannot export, benchmark or inject faults")
    if options.public_fault and not options.allow_testing:
        parser.error("fault hooks require --allow-testing")
    if options.public_diagnose and (not options.public_package or options.deployment_only or
                                   options.public_fault or options.public_only or options.benchmark or
                                   options.benchmark_cpp or options.benchmark_locked):
        parser.error("--public-diagnose requires an existing public package/archive/library/output-dir and no other execution mode")
    with _cli_evidence(options.output_dir, options):
        if options.public_diagnose:
            _run_public_diagnose(options)
        elif options.deployment_only:
            _run_deployment_child(options)
        elif options.public_package:
            _run_public_child(options)
        elif options.benchmark_locked:
            run_locked_benchmark(ops_library=options.ops_library, iterations=options.iterations,
                                 wait_seconds=options.wait_seconds, benchmark_profile=options.benchmark_profile,
                                 benchmark_scenario=options.benchmark_scenario,
                                 max_route_tile_rows=options.benchmark_route_tile_rows, nvtx=options.nvtx,
                                 host_profile=options.host_profile, async_staging=options.async_staging,
                                 allow_testing=options.allow_testing, output_dir=options.output_dir,
                                 cpp=options.benchmark_locked_cpp, case_filter=options.benchmark_case,
                                 tile_m=options.benchmark_tile_m, tile_n=options.benchmark_tile_n)
        elif options.benchmark_cpp:
            run_cpp_benchmark(ops_library=options.ops_library, iterations=options.iterations,
                              sizes=tuple(options.benchmark_cpp_sizes), wait_seconds=options.wait_seconds,
                              async_staging=options.async_staging, host_profile=options.host_profile,
                              allow_testing=options.allow_testing, output_dir=options.output_dir)
        elif options.benchmark:
            run_benchmark(ops_library=options.ops_library, iterations=options.iterations,
                          wait_seconds=options.wait_seconds, benchmark_profile=options.benchmark_profile,
                          benchmark_scenario=options.benchmark_scenario,
                          max_route_tile_rows=options.benchmark_route_tile_rows,
                          nvtx=options.nvtx, host_profile=options.host_profile,
                          async_staging=options.async_staging, allow_testing=options.allow_testing,
                          output_dir=options.output_dir)
        else:
            run_acceptance(ops_library=options.ops_library, sizes=tuple(options.sizes),
                           wait_seconds=options.wait_seconds, wide_bf16=options.wide_bf16,
                           async_staging=options.async_staging, host_profile=options.host_profile,
                           allow_testing=options.allow_testing, output_dir=options.output_dir,
                           public_only=options.public_only)
