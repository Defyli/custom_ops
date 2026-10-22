"""Local, source-tree delivery acceptance for FuseMoE offloading.

Every selected stage is mandatory. Omitted stages and explicitly omitted wide
BF16 acceptance are recorded, never counted as passes. No test internals are
imported, no loading policy is patched, and no remote service is contacted.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import statistics
import math
import subprocess
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("build", "unit", "aoti", "sanitizer", "performance")
LIBRARY = "recsys_fuse_moe_kernel.so"
OFFLOAD = ROOT / "tests/test_fuse_moe_offload.py"
AOTI = ROOT / "tests/test_fuse_moe_offload_aoti.py"
BASELINE = ROOT / "tests/test_fuse_moe.py"
PERFORMANCE_CASES = tuple("fp32_accumulate/bank2/" + cache for cache in
                          ("disabled", "static_hot", "reuse", "lru")) + ("route_buffer/bank1/disabled",)
ALL_PERFORMANCE_CASES = tuple(f"{mode}/bank{bank}/{cache}"
                              for mode in ("route_buffer", "fp32_accumulate", "auto")
                              for bank in (1, 2) for cache in ("disabled", "static_hot", "reuse", "lru"))
PERFORMANCE_PROFILES = ("weight_transfer", "activation_near_budget")
PERFORMANCE_SCENARIOS = ("skewed_masked", "hotset_transition")
SOURCE_SUFFIXES = {".py", ".cpp", ".cuh", ".cu", ".h", ".hpp"}
EXCLUDED = {"3rd", "thirdparty", ".git", ".venv", "venv", "__pycache__",
            "torch_extensions", "build", "dist", "artifacts", "node_modules"}
MEMCHECK = ("FaultInjectionTests", "ManagedRuntimeTests",
            "OffloadTests.test_inference_mode_and_nondefault_stream",
            "OffloadTests.test_cross_stream_pool_reuse",
            "OffloadTests.test_resident_cross_stream_reuse")
SYNCCHECK = ("OffloadTests.test_activation_modes_tiled_signed_routes",
             "OffloadTests.test_adaptive_signed_masked_dtypes_and_activation_selection",
             "OffloadTests.test_inference_mode_and_nondefault_stream")
GEGLU_SANITIZER_TARGETS = (
    "GEGLUTests.test_full_gpu_four_mode_dtype_matrix",
    "GEGLUTests.test_isolated_scalar_semantics_wide_n_atomic",
    "GEGLUTests.test_offload_route_fp32_cache_and_partial_tiles",
    "GEGLUTests.test_multi_context_modes_do_not_leak",
    "GEGLUTests.test_native_v1_v2_contract_and_failure_before_token_mutation",
    "GEGLUTests.test_masked_topk_and_zero_sequence",
)
MLP_SANITIZER_TARGETS = (
    "MLPTests.test_dual_dtype_four_modes_sync_async_cache_mask_signed_duplicate_edges",
    "MLPTests.test_scalar_activation_and_stage_witnesses",
    "MLPTests.test_v3_context_mismatch_before_token_mutation",
)
AOTI_CPU_CONTRACTS = (
    "test_mlp_v3_reference_archive_create_cpu_contract",
    "test_explicit_expert_modes_v2_config_cpu_contract",
    "test_geglu_staged_reference_cpu_contract",
    "test_prefill_create_explicit_mode_arguments_cpu_contract",
    "test_geglu_metrics_cpu_contract",
)
GEGLU_PRECISIONS = ("fp16", "bf16")
GEGLU_PERFORMANCE_PROFILE = "small"
GEGLU_PERFORMANCE_SCENARIO = "uniform"
GEGLU_PERFORMANCE_CASE = "fp32_accumulate/bank2/disabled"
GEGLU_PERFORMANCE_PRECISION = "bf16"
GEGLU_PERFORMANCE_MODES = (("silu", "fused"), ("gelu", "fused"),
                           ("silu", "staged"), ("gelu", "staged"))
BUILD_CODE = (
    "import sys; from custom_ops.recsys import FuseMoeOps; "
    "print(FuseMoeOps(allow_jit=False).build(sys.argv[1], "
    "cuda_arch_list=sys.argv[2], testing=sys.argv[3]=='testing', verbose=True), flush=True)"
)
DEVICE_CODE = (
    "import json, torch; "
    "d=torch.cuda.current_device(); c=torch.cuda.get_device_capability(d); "
    "print(json.dumps(dict(torch=str(torch.__version__), cuda_build=torch.version.cuda, "
    "cxx_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI), device_index=d, "
    "gpu=torch.cuda.get_device_name(d), compute_capability=list(c), "
    "target_arch='12.0a' if c==(12,0) else '%d.%d'%c)))"
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(output):
    result = {}
    for directory, subdirs, files in os.walk(ROOT):
        parent = Path(directory)
        subdirs[:] = sorted(name for name in subdirs if name not in EXCLUDED
                            and not name.startswith(".")
                            and not (parent / name).is_symlink()
                            and (parent / name).resolve() != output)
        for name in sorted(files):
            path = parent / name
            # Only source code; no dotfiles, environment files, remotes or archives.
            if (not path.is_symlink() and not name.startswith(".")
                    and path.suffix in SOURCE_SUFFIXES
                    and not re.search(r"credential|secret|remote|token", name, re.I)):
                result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return dict(sorted(result.items()))


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection_ns(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def gpu_overlap(database, benchmark):
    """Use CUPTI GPU intervals, not CPU observer durations, for overlap.

    Only exact locked raw-sample phase/route NVTX scopes are eligible. The frozen
    fixture uploads inputs outside those scopes; inside them, expert H2D byte
    multiples distinguish weights from the small expert-count transfer. Runtime
    correlation and process/device/context grouping prevent cross-process joins.
    """
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "CUPTI_ACTIVITY_KIND_MEMCPY", "ENUM_CUDA_MEMCPY_OPER"}
        if not required <= tables:
            raise ValueError("Missing CUPTI/NVTX tables: " + str(sorted(required - tables)))
        kernel_table = next((name for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
                             if name in tables), None)
        if kernel_table is None:
            raise ValueError("CUPTI GPU kernel activity absent")
        strings = {row["id"]: row["value"] for row in connection.execute("SELECT id, value FROM StringIds")}
        # The raw warm list contains actual schedule phase names, e.g.
        # skewed_masked or transition, NOT a literal '/warm/' NVTX segment.
        prefix = (f"locked/{benchmark['profile']}/{benchmark['scenario']}/{benchmark['case_key']}/"
                  f"host_profile={int(benchmark['host_profile'])}/")
        expected_labels = Counter(prefix + sample["phase"] + "/" + sample["route"]
                                  for group in ("cold", "priming", "warm")
                                  for sample in benchmark["raw"][group])
        scopes = []
        observed_labels = Counter()
        for row in connection.execute("SELECT * FROM NVTX_EVENTS WHERE end > start"):
            fields = dict(row)
            label = fields.get("text") or strings.get(fields.get("textId"), "")
            if label in expected_labels:
                scopes.append((fields["start"], fields["end"], fields["globalTid"]))
                observed_labels[label] += 1
        if observed_labels != expected_labels:
            raise ValueError("Locked invocation NVTX coverage mismatch: "
                             + str(dict(observed_labels)) + " expected " + str(dict(expected_labels)))
        runtime = {}
        for row in connection.execute("SELECT start, end, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            if any(row["globalTid"] == tid and start <= row["start"] <= row["end"] <= end
                   for start, end, tid in scopes):
                runtime.setdefault(row["correlationId"], set()).add(row["globalTid"] >> 24)
        kinds = {row["id"] for row in connection.execute("SELECT * FROM ENUM_CUDA_MEMCPY_OPER")
                 if any("htod" in str(value).lower().replace(" ", "")
                        or "host-to-device" in str(value).lower()
                        for value in dict(row).values())}
        if not kinds:
            raise ValueError("Cannot identify the official HtoD enum")
        shape = benchmark["shape"]
        weight_quantum = 2 * shape["H"] * shape["I"]
        count_bytes = 4 * (shape["E"] + 1)
        def eligible(row):
            return (row["globalPid"] >> 24) in runtime.get(row["correlationId"], set())
        copies, kernels = {}, {}
        excluded_copies = 0
        for row in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end > start"):
            if row["copyKind"] not in kinds or not eligible(row):
                continue
            if row["bytes"] <= count_bytes or row["bytes"] % weight_quantum:
                excluded_copies += 1
                continue
            key = (row["globalPid"], row["deviceId"], row["contextId"])
            copies.setdefault(key, []).append((row["start"], row["end"], row["bytes"]))
        # Table names come exclusively from the fixed allowlist above.
        for row in connection.execute("SELECT * FROM " + kernel_table + " WHERE end > start"):
            if eligible(row):
                key = (row["globalPid"], row["deviceId"], row["contextId"])
                kernels.setdefault(key, []).append((row["start"], row["end"]))
        if not copies or not kernels:
            raise ValueError("No scoped weight H2D or GPU kernels; CUPTI collection is incomplete")
        actual_bytes = sum(size for values in copies.values() for _, _, size in values)
        expected_bytes = sum(sample["stats"]["weight_h2d_bytes"]
                             for phase in ("cold", "priming", "warm") for sample in benchmark["raw"][phase])
        if actual_bytes != expected_bytes:
            raise ValueError(f"Scoped CUPTI weight bytes {actual_bytes} != native stats {expected_bytes}; "
                             "cannot claim complete weight-transfer activity")
        groups = []
        for key, intervals in sorted(copies.items()):
            copy_intervals = [(start, end) for start, end, _ in intervals]
            gpu_kernels = kernels.get(key, [])
            union_ns = sum(end - start for start, end in merge_intervals(copy_intervals))
            overlap = intersection_ns(copy_intervals, gpu_kernels)
            groups.append(dict(global_pid=key[0], device_id=key[1], context_id=key[2],
                               weight_h2d_intervals_ns=intervals, kernel_intervals_ns=gpu_kernels,
                               weight_h2d_union_ns=union_ns, kernel_intersection_ns=overlap,
                               overlap_fraction=overlap / union_ns if union_ns else 0))
        return dict(schema_version=1, source="Nsight SQLite CUPTI GPU activities",
                    diagnostic_only=True, sqlite=str(database), sqlite_sha256=sha256(database),
                    nvtx_scopes=len(scopes), excluded_count_or_nonweight_h2d=excluded_copies,
                    weight_h2d_bytes=actual_bytes, expected_native_weight_h2d_bytes=expected_bytes,
                    kernel_intersection_ns=sum(group["kernel_intersection_ns"] for group in groups), groups=groups,
                    note="Weight attribution uses frozen invocation scope and expert byte multiples; "
                         "count H2D and all D2H excluded. Zero measured overlap is valid, "
                         "not a failure or evidence of speedup. Initialization/export activity excluded.")


def parser_for_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Absolute delivery artifact directory (exclusive local lock)")
    parser.add_argument("--target-arch", default=None, choices=("8.9", "12.0a"),
                        help="Build target; default is the active device's CUDA architecture")
    parser.add_argument("--jobs", type=int, default=2, help="MAX_JOBS for native/AOTI builds (1..64)")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                        help="Any order; execution always follows build/unit/aoti/sanitizer/performance")
    parser.add_argument("--skip-performance", action="store_true",
                        help="Explicitly omit performance; record it as skipped, not passed")
    parser.add_argument("--python-full-leak-check", action="store_true",
                        help="Opt-in strict Python-process full-leak diagnostic; failures fail this run, "
                             "separate from mandatory native ownership teardown")
    parser.add_argument("--no-wide-bf16", action="store_true",
                        help="Omit optional wide BF16 AOTI checks; FP16 wide checks remain mandatory")
    parser.add_argument("--performance-cases", nargs="+", choices=ALL_PERFORMANCE_CASES,
                        default=list(PERFORMANCE_CASES),
                        help="Locked subsets (default: four fp32/bank2 caches plus route_buffer/bank1/disabled)")
    parser.add_argument("--geglu-performance-matrix", action=argparse.BooleanOptionalAction, default=True,
                        help="Additional representative BF16 locked four-mode ABBA matrix (default on)")
    parser.add_argument("--performance-cpp", action="store_true",
                        help="Also run locked C++ for one cache case each, sync/async, weight_transfer/hotset_transition")
    parser.add_argument("--performance-trace", action=argparse.BooleanOptionalAction, default=True,
                        help="Independent mandatory Nsight CUDA/NVTX trace (default on); omission is recorded")
    parser.add_argument("--benchmark-activation", choices=("silu", "gelu"), default="silu",
                        help="Explicit expert nonlinearity passed to locked benchmark wrapper (default silu)")
    parser.add_argument("--benchmark-numerical-mode", choices=("fused", "staged"), default="fused",
                        help="Explicit numerical contract passed to locked benchmark wrapper (default fused)")
    parser.add_argument("--benchmark-precision", choices=("fp16", "bf16"), default="fp16",
                        help="Explicit weight/input storage dtype passed to locked benchmark wrapper (default fp16)")
    parser.add_argument("--omp-num-threads", type=int,
                        help="Explicit child OMP_NUM_THREADS; otherwise preserve the caller setting")
    for stage in STAGES:
        parser.add_argument("--" + stage + "-timeout", type=float, default=1200,
                            help="Per-command timeout in seconds for " + stage + " (default 1200)")
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        parser.error("--output-dir must be absolute")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == ROOT or ROOT.is_relative_to(args.output_dir):
        parser.error("--output-dir must not be the project root or its ancestor")
    if not 1 <= args.jobs <= 64:
        parser.error("--jobs must be between 1 and 64")
    if args.omp_num_threads is not None and args.omp_num_threads < 1:
        parser.error("--omp-num-threads must be positive")
    for stage in STAGES:
        value = getattr(args, stage + "_timeout")
        if not 0 < value < float("inf"):
            parser.error("timeouts must be finite and positive")
    if len(set(args.performance_cases)) != len(args.performance_cases):
        parser.error("--performance-cases must not contain duplicates")
    if len(set(args.stages)) != len(args.stages):
        parser.error("--stages must not contain duplicates")
    return args


class Delivery:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir
        self.run_dir = self.output / "runs" / (dt.datetime.now(dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])
        self.run_dir.mkdir(parents=True)
        self.logs = self.run_dir / "raw"
        self.logs.mkdir()
        self.env = os.environ.copy()
        self.env["MAX_JOBS"] = str(args.jobs)
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env["PYTHONPATH"] = str(ROOT.parent) + os.pathsep + self.env.get("PYTHONPATH", "")
        if args.omp_num_threads is not None:
            self.env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
        selected = set(args.stages)
        if args.skip_performance:
            selected.discard("performance")
        self.report = dict(schema_version=1, state="running", started_at=utc_now(),
                           repository=str(ROOT), output_dir=str(self.output),
                           run_dir=str(self.run_dir), argv=sys.argv[1:],
                           settings=dict(target_arch=args.target_arch, jobs=args.jobs,
                                         wide_bf16=not args.no_wide_bf16,
                                         benchmark_activation=getattr(args, "benchmark_activation", "silu"),
                                         benchmark_numerical_mode=getattr(args, "benchmark_numerical_mode", "fused"),
                                         benchmark_precision=getattr(args, "benchmark_precision", "fp16"),
                                         geglu_performance_matrix=getattr(args, "geglu_performance_matrix", True),
                                         timeouts={s: getattr(args, s + "_timeout") for s in STAGES},
                                         environment={k: self.env.get(k) for k in (
                                             "MAX_JOBS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                                             "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")}),
                           stages={s: dict(state="pending" if s in selected else "skipped",
                                           reason=None if s in selected else (
                                               "--skip-performance" if s == "performance" and args.skip_performance
                                               else "not selected"), commands=[], unmet=[]) for s in STAGES},
                           commands=[], telemetry={}, artifacts={}, unmet=[])
        self.save()

    def save(self):
        write_json(self.run_dir / "report.json", self.report)
        write_json(self.output / "report.json", self.report)

    def unmet(self, stage, reason):
        self.report["stages"][stage]["unmet"].append(reason)
        self.save()
        print("UNMET " + stage + ": " + reason, flush=True)

    @staticmethod
    def stop(process):
        # Kill the complete compile/test tree, not just the Python parent.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A compiler/runner descendant may outlive an already-exited parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def command(self, stage, label, argv, *, env=None, timeout=None, unittest=False):
        stdout = self.logs / (label + ".stdout.txt")
        stderr = self.logs / (label + ".stderr.txt")
        entry = dict(stage=stage, label=label, argv=[str(a) for a in argv], cwd=str(ROOT),
                     started_at=utc_now(), state="running", returncode=None,
                     stdout=str(stdout), stderr=str(stderr), timeout_seconds=timeout or (
                         getattr(self.args, stage + "_timeout") if stage in STAGES else 30),
                     environment_overrides=env or {})
        self.report["commands"].append(entry)
        if stage in STAGES:
            self.report["stages"][stage]["commands"].append(label)
        self.save()
        print("RUN " + label, flush=True)
        environment = self.env.copy()
        environment.update(env or {})
        started = time.monotonic()
        process = None
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen(entry["argv"], cwd=ROOT, env=environment,
                                           stdout=out, stderr=err, start_new_session=True)
                try:
                    entry["returncode"] = process.wait(timeout=entry["timeout_seconds"])
                    entry["state"] = "passed" if entry["returncode"] == 0 else "failed"
                except subprocess.TimeoutExpired:
                    self.stop(process)
                    entry.update(state="timeout", returncode=process.returncode)
        except KeyboardInterrupt:
            if process is not None:
                self.stop(process)
                entry["returncode"] = process.returncode
            entry["state"] = "interrupted"
            raise
        except OSError as error:
            entry.update(state="unavailable", error=str(error))
        finally:
            entry.update(ended_at=utc_now(), elapsed_seconds=time.monotonic() - started)
            if entry["returncode"] is not None and entry["returncode"] < 0:
                entry["signal"] = -entry["returncode"]
            if unittest and stderr.exists():
                text = stderr.read_text(encoding="utf-8", errors="replace")
                counts = re.findall(r"Ran (\d+) tests?", text)
                skips = re.findall(r"skipped=(\d+)", text)
                entry["tests_run"] = int(counts[-1]) if counts else None
                entry["tests_skipped"] = int(skips[-1]) if skips else 0
                entry["skip_details"] = [line for line in text.splitlines()
                                         if "skipped " in line or "... skipped" in line]
                if entry["state"] == "passed" and (not entry["tests_run"] or entry["tests_skipped"]):
                    entry["state"] = "unmet"
                    entry["error"] = "No unittest execution evidence or skipped mandatory tests"
            self.save()
        return entry

    def python(self, stage, label, script, options=(), **kwargs):
        return self.command(stage, label, [sys.executable, str(script), *map(str, options)], **kwargs)

    def collect(self):
        sources = source_manifest(self.output)
        self.report["source_sha256"] = sources
        write_json(self.run_dir / "source_sha256.json", sources)
        # Restrict the diff to audited source paths. Do not collect .gitignore,
        # remote.txt, arbitrary untracked content, config files or credentials.
        for label, argv in (
            ("git_head", ["git", "rev-parse", "HEAD"]),
            ("git_diff", ["git", "diff", "--binary", "HEAD", "--", *sources]),
            ("gpu", ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,pstate,"
                     "temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv"]),
            ("cpu", ["lscpu", "-J"]),
            ("nvcc", ["nvcc", "--version"]),
            ("compiler", ["c++", "--version"]),
            ("sanitizer_version", ["compute-sanitizer", "--version"]),
        ):
            entry = self.command("telemetry", label, argv, timeout=30)
            self.report["telemetry"][label] = dict(state=entry["state"], stdout=entry["stdout"],
                                                    stderr=entry["stderr"])
        entry = self.command("telemetry", "device", [sys.executable, "-c", DEVICE_CODE], timeout=30)
        if entry["state"] == "passed":
            try:
                device = json.loads(Path(entry["stdout"]).read_text(encoding="utf-8").splitlines()[-1])
                self.report["telemetry"]["device"] = device
                if self.args.target_arch is None:
                    self.args.target_arch = device["target_arch"]
            except (ValueError, IndexError, KeyError) as error:
                self.report["telemetry"]["device"] = dict(state="invalid", error=str(error))
        else:
            self.report["telemetry"]["device"] = dict(state=entry["state"])
        self.report["settings"]["target_arch"] = self.args.target_arch
        paths = [Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu.stat"),
                 Path("/sys/fs/cgroup/cpuset.cpus.effective"),
                 Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
                 Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")]
        # Include the actual process's nested cgroup, where quota files may live.
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                _, controllers, relative = line.split(":", 2)
                base = Path("/sys/fs/cgroup") / relative.lstrip("/")
                if not controllers:
                    paths.extend(base / name for name in ("cpu.max", "cpu.stat", "cpuset.cpus.effective"))
        except (OSError, ValueError):
            pass
        quota = {}
        for path in paths:
            try:
                quota[str(path)] = path.read_text(encoding="utf-8").strip()
            except OSError as error:
                quota[str(path)] = dict(unavailable=str(error))
        self.report["telemetry"]["cpu_quota"] = quota
        if hasattr(os, "sched_getaffinity"):
            self.report["telemetry"]["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        self.save()

    def library(self, stage, kind):
        build_entries = [entry for entry in self.report["commands"]
                         if entry["label"] == "build_" + kind]
        if build_entries and build_entries[-1]["state"] != "passed":
            self.unmet(stage, "Current build failed; refusing a possibly stale " + kind + " artifact")
            return None
        path = self.output / "native" / kind / LIBRARY
        sidecar = path.with_suffix(".so.json")
        if not path.is_file() or not sidecar.is_file():
            self.unmet(stage, "Required explicit library and sidecar missing: " + str(path))
            return None
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            digest = sha256(path)
            if metadata["library"]["sha256"] != digest:
                raise ValueError("sidecar library checksum mismatch")
            if metadata.get("testing") is not (kind == "testing"):
                raise ValueError("production/testing artifact marker mismatch")
            if (self.args.target_arch is not None
                    and self.args.target_arch not in metadata.get("cuda_arch_list", [])):
                raise ValueError("artifact does not contain the requested target architecture")
            self.report["artifacts"][kind] = dict(library=str(path), sidecar=str(sidecar),
                                                   sha256=digest, sidecar_sha256=sha256(sidecar))
            self.save()
            # Official consumers validate the complete ABI/schema/testing contract.
            return path
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.unmet(stage, "Invalid library artifact: " + str(error))
            return None

    def build(self):
        if self.args.target_arch is None:
            self.unmet("build", "Device architecture unavailable; supply --target-arch 8.9 or 12.0a")
            return
        for kind in ("prod", "testing"):
            directory = self.output / "native" / kind
            entry = self.command("build", "build_" + kind,
                                 [sys.executable, "-c", BUILD_CODE, str(directory), self.args.target_arch, kind])
            if entry["state"] == "passed":
                self.library("build", kind)

    def unit(self):
        self.python("unit", "loading_cpu", ROOT / "tests/test_custom_ops_loading.py", ["-v"], unittest=True)
        self.python("unit", "aoti_lowering_cpu", ROOT / "tests/test_fuse_moe_offload_aoti_lowering.py",
                    ["-v"], unittest=True)
        self.python("unit", "delivery_cpu", ROOT / "tests/test_fuse_moe_offload_delivery.py",
                    ["-v"], unittest=True)
        self.python("unit", "geglu_aoti_contracts_cpu", AOTI,
                    ["--cpu-contracts", *AOTI_CPU_CONTRACTS])
        self.python("unit", "tilelang_host_contracts_cpu", ROOT / "tests/test_tilelang_ops.py",
                    ["--cpu-only"], unittest=True)
        library = self.library("unit", "testing")
        if library:
            for mode in ("sync", "async"):
                options = ["--ops-library", library, "--allow-testing", "-v"]
                if mode == "async":
                    options.append("--async-staging")
                self.python("unit", "native_" + mode, OFFLOAD, options, unittest=True)
        production = self.library("unit", "prod")
        # Inspect the documented argparse entry, not an imported test helper.
        tree = ast.parse(BASELINE.read_text(encoding="utf-8"))
        accepts_library = any(isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                              and any(isinstance(arg, ast.Constant) and arg.value == "--ops-library"
                                      for arg in node.args) for node in ast.walk(tree))
        if not accepts_library:
            self.unmet("unit", "Legacy test_fuse_moe.py needs a public --ops-library CLI; "
                       "baseline not run (no JIT or monkeypatch fallback)")
        elif production:
            baseline_report = self.run_dir / "full_gpu_accuracy.json"
            entry = self.python("unit", "legacy_baseline", BASELINE,
                                ["--ops-library", production, "--accuracy-report", "--json", baseline_report])
            self.baseline_evidence(entry, baseline_report)

    def baseline_evidence(self, entry, report_path):
        if entry["state"] != "passed":
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report["summary"]
            if (report["gpu"]["state"] != "passed" or summary["state"] != "passed"
                    or summary["matrix_cases"] <= 0 or summary["matrix_failed"] != 0
                    or summary["matrix_passed"] != summary["matrix_cases"]
                    or summary["cpu_and_gpu_checks"]["FAIL"] != 0):
                raise ValueError("GPU matrix missing, skipped or failed")
            entry.update(accuracy_report=str(report_path), accuracy_report_sha256=sha256(report_path),
                         matrix_cases=summary["matrix_cases"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry.update(state="unmet", error="Invalid full-GPU baseline evidence: " + str(error))
            self.unmet("unit", entry["error"])
        self.save()

    def geglu_evidence(self, entry, directory, precision, expert_type="glu"):
        suite = "mlp" if expert_type == "mlp" else "geglu"
        if entry["state"] != "passed":
            return None
        try:
            report_path = directory / f"{suite}_report.json"
            exit_path = directory / "acceptance.exit.json"
            package_path = directory / f"{suite}_chain_{precision}.pt2"
            archive_path = directory / f"{suite}_weights_{precision}.pt"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            exit_report = json.loads(exit_path.read_text(encoding="utf-8"))
            if exit_report.get("state") != "passed" or exit_report.get("returncode") != 0:
                raise ValueError("GEGLU acceptance exit report did not pass")
            expected = dict(schema_version=1, state="passed", archive_format_version=2,
                            manifest_framing=1, precision=precision,
                            expert_activation="gelu", numerical_mode="staged",
                            package=str(package_path.resolve()), archive=str(archive_path.resolve()))
            if any(report.get(key) != value for key, value in expected.items()):
                raise ValueError("GEGLU report metadata or mode contract mismatch")
            if not package_path.is_file() or not archive_path.is_file():
                raise ValueError("GEGLU package or archive artifact missing")
            if report.get("expert_type", "glu") != expert_type:
                raise ValueError("Expert type report mismatch")
            config = report["config"]
            if config.get("expert_type", "glu") != expert_type:
                raise ValueError("Frozen expert_type mismatch")
            if (config.get("async_staging") is not True or config.get("host_profile") is not True
                    or config.get("expert_activation") != "gelu" or config.get("numerical_mode") != "staged"):
                raise ValueError("GEGLU frozen config flags/modes mismatch")
            layer_configs = report.get("layer_configs", {})
            if set(layer_configs) != {"0", "5"}:
                raise ValueError("GEGLU layer_configs must contain layers 0 and 5")
            for vector in layer_configs.values():
                fixed = 26 if expert_type == "mlp" else 25
                if (not isinstance(vector, list) or len(vector) < fixed or
                        len(vector) != fixed + vector[21] or vector[0] != (3 if expert_type == "mlp" else 2) or
                        vector[23:25] != [1, 1] or (expert_type == "mlp" and vector[25] != 1)):
                    raise ValueError("GEGLU layer_config v2 header/mode codes mismatch")
            python_cases = report.get("python_cases", [])
            cpp_cases = report.get("cpp_cases", [])
            if len(python_cases) != 10 or len(cpp_cases) != 4:
                raise ValueError("GEGLU Python/C++ sample counts mismatch")
            if [case.get("S") for case in cpp_cases] != [2, 7, 64, 2]:
                raise ValueError("GEGLU C++ dynamic S sequence mismatch")
            for case in cpp_cases:
                if (case.get("expert_type", "glu") != expert_type or case.get("precision") != precision
                        or case.get("expert_activation") != "gelu"
                        or case.get("numerical_mode") != "staged"
                        or case.get("python_uninitialized") is not True
                        or case.get("strict_no_jit") is not True
                        or case.get("token_checked") is not True
                        or case.get("status_checked") is not True
                        or case.get("reference_checked") is not True
                        or case.get("async_staging_enabled") is not True
                        or case.get("host_profile_enabled") is not True):
                    raise ValueError("GEGLU C++ child contract flags mismatch")
            entry.update(geglu_report=str(report_path), geglu_report_sha256=sha256(report_path),
                         package=str(package_path), package_sha256=sha256(package_path),
                         archive=str(archive_path), archive_sha256=sha256(archive_path))
            self.save()
            return report
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry.update(state="unmet", error="Invalid GEGLU AOTI evidence: " + str(error))
            self.unmet("aoti", "Invalid GEGLU AOTI evidence for " + entry["label"] + ": " + str(error))
            return None

    def aoti(self):
        production = self.library("aoti", "prod")
        if production:
            options = ["--ops-library", production, "--output-dir", self.run_dir / "aoti/prod",
                       "--sizes", 2, 7, 64, "--async-staging", "--host-profile"]
            if self.args.no_wide_bf16:
                options.append("--no-wide-bf16")
                self.report["stages"]["aoti"]["wide_bf16"] = dict(state="skipped", reason="--no-wide-bf16")
            else:
                self.report["stages"]["aoti"]["wide_bf16"] = dict(state="required")
            self.python("aoti", "aoti_full_prod", AOTI, options)
            geglu_records = {}
            for precision in GEGLU_PRECISIONS:
                geglu_dir = self.run_dir / f"aoti/geglu_{precision}_prod"
                geglu_options = ["--ops-library", production, "--geglu-only",
                                 "--geglu-precision", precision, "--output-dir", geglu_dir,
                                 "--sizes", 2, 7, 64, "--async-staging", "--host-profile"]
                entry = self.python("aoti", f"aoti_geglu_{precision}_prod", AOTI, geglu_options)
                report = self.geglu_evidence(entry, geglu_dir, precision)
                geglu_records[precision] = dict(
                    state=entry["state"], output_dir=str(geglu_dir),
                    report=entry.get("geglu_report"), package=entry.get("package"),
                    archive=entry.get("archive"), validated=report is not None)
            mlp_records = {}
            for precision in GEGLU_PRECISIONS:
                directory = self.run_dir / f"aoti/mlp_{precision}_prod"
                entry = self.python("aoti", f"aoti_mlp_{precision}_prod", AOTI,
                    ["--ops-library", production, "--mlp-only", "--mlp-precision", precision,
                     "--output-dir", directory, "--sizes", 2, 7, 64, "--async-staging", "--host-profile"])
                report = self.geglu_evidence(entry, directory, precision, expert_type="mlp")
                mlp_records[precision] = dict(state=entry["state"], output_dir=str(directory),
                    package=entry.get("package"), archive=entry.get("archive"), validated=report is not None)
            self.report["stages"]["aoti"]["mlp"] = mlp_records
            self.report["stages"]["aoti"]["geglu"] = geglu_records
            self.save()
        testing = self.library("aoti", "testing")
        if testing:
            self.python("aoti", "aoti_public_testing", AOTI,
                        ["--ops-library", testing, "--allow-testing", "--public-only",
                         "--async-staging", "--output-dir", self.run_dir / "aoti/testing",
                         "--sizes", 2, 7, 64])

    def sanitizer_result(self, entry, *, full_leak=False):
        """Require actual tool evidence as well as the child's exit status."""
        original_state = entry["state"]
        text = "\n".join(Path(entry[key]).read_text(encoding="utf-8", errors="replace")
                         for key in ("stdout", "stderr") if Path(entry[key]).exists())
        errors = re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", text)
        leaks = re.findall(r"LEAK SUMMARY:\s*([\d,]+) bytes(?: leaked)? in ([\d,]+) allocations?", text)
        entry["sanitizer_error_counts"] = [int(value) for value in errors]
        entry["sanitizer_leak_counts"] = [dict(bytes=int(size.replace(",", "")),
                                              allocations=int(count.replace(",", ""))) for size, count in leaks]
        if original_state == "passed" and (not errors or any(int(value) != 0 for value in errors)
                or (full_leak and (not leaks or any(int(size.replace(",", "")) != 0
                                                   or int(count.replace(",", "")) != 0 for size, count in leaks)))):
            entry.update(state="unmet", error="Missing/nonzero sanitizer summary for the requested scope")
        self.save()
        return entry["state"] == "passed"

    def native_fault_leaks(self, tool, library):
        directory = self.run_dir / "native_faults"
        compiled = self.python("sanitizer", "compile_native_fault_runner", AOTI,
                               ["--compile-native-runner-only", "--output-dir", directory])
        if compiled["state"] != "passed":
            return
        executable = directory / "fuse_moe_offload_aoti_runner"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            self.unmet("sanitizer", "Official runner build did not export " + str(executable))
            return
        device = self.report["telemetry"].get("device", {}).get("device_index")
        if type(device) is not int or device < 0:
            self.unmet("sanitizer", "Native fault runner requires a detected CUDA device index")
            return
        entry = self.command("sanitizer", "native_faults_full_leak",
                             [tool, "--tool", "memcheck", "--leak-check", "full", "--error-exitcode", "99",
                              "--target-processes", "all", str(executable), "--native-faults", str(library), str(device)])
        if not self.sanitizer_result(entry, full_leak=True):
            return
        text = Path(entry["stdout"]).read_text(encoding="utf-8", errors="replace")
        try:
            summaries = [json.loads(line.removeprefix("NATIVE_TEARDOWN "))
                         for line in text.splitlines() if line.startswith("NATIVE_TEARDOWN ")]
            if len(summaries) != 1:
                raise ValueError("Expected exactly one NATIVE_TEARDOWN summary")
            summary = summaries[0]
            expected = dict(fault_cases=14, sync_fault_cases=5, async_fault_cases=9, worker_state_cases=7,
                            concurrent_context_cases=14, healthy_checks=16, concurrent_healthy_calls=42,
                            allocation_acknowledgements=4, poison_ack_rejections=10,
                            callback_output_unwind_cases=2, allocated_before_trim=0,
                            allocated_after_trim=0, reserved_after_trim=0,
                            cuda_allocator_empty_cache=True, pinned_allocator_empty_cache=True)
            if any(type(summary.get(key)) is not type(value) or summary[key] != value
                   for key, value in expected.items()):
                raise ValueError("Native fault coverage/teardown contract mismatch")
            if "PASS NATIVE faults full allocator teardown Python_uninitialized" not in text:
                raise ValueError("Native no-interpreter final acceptance marker missing")
            entry.update(native_teardown=summary, executable_sha256=sha256(executable),
                         ownership_scope="standalone native CUDA and pinned allocator teardown; Python uninitialized")
        except (ValueError, KeyError, TypeError) as error:
            entry.update(state="unmet", error=str(error))
        self.save()

    def sanitizer(self):
        record = self.report["stages"]["sanitizer"]
        record["memory_contract"] = dict(
            python_memcheck="Invalid memory access with leak-check no; unchanged fault/runtime/stream tests",
            native_full_leak="Mandatory standalone --native-faults with leak-check full, all fault states, "
                             "lease/drain/teardown assertions and zero sanitizer leak summary",
            python_process_full_leak="Separate opt-in diagnostic, strict failure when selected; "
                                     "no arbitrary Python embedding allocator-zero guarantee",
            evidence_policy="No suppressions, no relabeling previous failures, no allocator/root attribution "
                            "without proof. Prior runs remain immutable.")
        record["python_full_leak"] = dict(state="required" if self.args.python_full_leak_check else "skipped",
                                          reason=None if self.args.python_full_leak_check else "optional diagnostic not selected")
        self.save()
        library = self.library("sanitizer", "testing")
        if not library:
            return
        tool = shutil.which("compute-sanitizer")
        if tool is None:
            self.unmet("sanitizer", "compute-sanitizer unavailable; Python access and native full-leak checks mandatory")
        else:
            for name, targets in (("memcheck", MEMCHECK), ("synccheck", SYNCCHECK)):
                for mode in ("sync", "async"):
                    options = ["--ops-library", str(library), "--allow-testing", "-v", *targets]
                    if mode == "async":
                        options.append("--async-staging")
                    flags = [tool, "--tool", name, "--error-exitcode", "99", "--target-processes", "all"]
                    if name == "memcheck":
                        flags.extend(["--leak-check", "no"])
                    entry = self.command("sanitizer", name + "_" + mode,
                                         [*flags, sys.executable, str(OFFLOAD), *options], unittest=True)
                    self.sanitizer_result(entry)
            self.native_fault_leaks(tool, library)
            production = self.library("sanitizer", "prod")
            if production:
                for mode in ("sync", "async"):
                    options = ["--ops-library", str(production), "-v", *GEGLU_SANITIZER_TARGETS]
                    if mode == "async":
                        options.append("--async-staging")
                    entry = self.command(
                        "sanitizer", "geglu_memcheck_" + mode,
                        [tool, "--tool", "memcheck", "--leak-check", "no",
                         "--error-exitcode", "99", "--target-processes", "all",
                         sys.executable, str(OFFLOAD), *options],
                        unittest=True)
                    self.sanitizer_result(entry)
            if self.args.python_full_leak_check:
                diagnostic_entries = []
                for mode in ("sync", "async"):
                    options = ["--ops-library", str(library), "--allow-testing", "-v", *MEMCHECK]
                    if mode == "async":
                        options.append("--async-staging")
                    entry = self.command("sanitizer", "python_full_leak_" + mode,
                                         [tool, "--tool", "memcheck", "--leak-check", "full", "--error-exitcode", "99",
                                          "--target-processes", "all", sys.executable, str(OFFLOAD), *options], unittest=True)
                    entry["scope"] = "optional strict Python-process diagnostic, not native-only ownership"
                    self.sanitizer_result(entry, full_leak=True)
                    diagnostic_entries.append(entry)
                record["python_full_leak"]["state"] = (
                    "passed" if all(entry["state"] == "passed" for entry in diagnostic_entries) else "failed")
                self.save()
        for mode in ("sync", "async"):
            # Deduplicate focused tests without changing canonical order.
            options = ["--ops-library", library, "--allow-testing", "-v",
                       *dict.fromkeys((*MEMCHECK, *SYNCCHECK))]
            if mode == "async":
                options.append("--async-staging")
            self.python("sanitizer", "malloc_async_" + mode, OFFLOAD, options, unittest=True,
                        env={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync",
                             "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"})

    def locked_options(self, library, directory, profile, scenario, case, mode, *,
                       diagnostic=False, cpp=False, contract=None):
        activation, numerical, precision = contract or (
            getattr(self.args, "benchmark_activation", "silu"),
            getattr(self.args, "benchmark_numerical_mode", "fused"),
            getattr(self.args, "benchmark_precision", "fp16"))
        options = ["--benchmark-locked", "--ops-library", library, "--output-dir", directory,
                   "--benchmark-profile", profile, "--benchmark-scenario", scenario,
                   "--benchmark-case", case, "--benchmark-route-tile-rows", "auto",
                   "--benchmark-tile-m", 0, "--benchmark-tile-n", 128,
                   "--benchmark-activation", activation,
                   "--benchmark-numerical-mode", numerical,
                   "--benchmark-precision", precision,
                   "--iterations", 40]
        if mode == "async":
            options.append("--async-staging")
        if cpp:
            options.append("--benchmark-locked-cpp")
        if diagnostic:
            options.extend(["--host-profile", "--nvtx"])
        return options

    def locked_evidence(self, entry, directory, profile, scenario, case, mode, *,
                        diagnostic=False, cpp=False, contract=None):
        if entry["state"] != "passed":
            return []
        try:
            index_path = directory / "locked_benchmark_index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            expected = dict(profile=profile, scenario=scenario, case_key=case)
            if not index.get("final_frozen") or index["cases"] != [expected]:
                raise ValueError("Locked case index does not match the requested singleton")
            if index["backend_count"] != (2 if cpp else 1):
                raise ValueError("Locked backend count mismatch")
            activation, bank, cache = case.split("/")
            case_dir = directory / f"locked_{profile}_{activation}_{bank[-1]}_{cache}"
            paths = [case_dir / (scenario + ".json")]
            if cpp:
                paths.append(case_dir / ("cpp_" + scenario + ".json"))
            records = []
            for path in paths:
                record = json.loads(path.read_text(encoding="utf-8"))
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError("Locked report case identity mismatch")
                fields = record["frozen_config"]["fields"]
                expert_activation, numerical_mode, precision = contract or (
                    getattr(self.args, "benchmark_activation", "silu"),
                    getattr(self.args, "benchmark_numerical_mode", "fused"),
                    getattr(self.args, "benchmark_precision", "fp16"))
                if (record.get("expert_activation") != expert_activation
                        or record.get("numerical_mode") != numerical_mode
                        or record.get("precision") != precision
                        or fields.get("expert_activation") != expert_activation
                        or fields.get("numerical_mode") != numerical_mode):
                    raise ValueError("Locked expert activation/numerical/dtype contract mismatch")
                vector = record["frozen_config"]["layer_config"]
                fixed = 23 if vector[0] == 1 else 25 if vector[0] == 2 else 0
                if (not fixed or len(vector) != fixed + vector[21]
                        or vector[20] != int(precision == "bf16")):
                    raise ValueError("Locked frozen vector version/dtype mismatch")
                codes = (0, 0) if vector[0] == 1 else tuple(vector[23:25])
                if codes != (int(expert_activation == "gelu"), int(numerical_mode == "staged")):
                    raise ValueError("Locked frozen vector expert mode codes mismatch")
                if fields["async_staging"] is not (mode == "async"):
                    raise ValueError("Frozen async configuration mismatch")
                if not diagnostic and record["timing_kind"] != "normal_wall":
                    raise ValueError("Diagnostic timings cannot enter normal-wall comparison")
                raw = record["raw"]
                initialization = raw.get("initialization", raw.get("initialization_ms", []))
                if len(initialization) != 41:
                    raise ValueError("Expected 40 cold initializations and one warm context")
                for sample in initialization:
                    latency = sample["host_ms"] if isinstance(sample, dict) else sample
                    if not math.isfinite(latency) or latency <= 0:
                        raise ValueError("Invalid initialization latency")
                for phase in ("cold", "priming", "warm"):
                    samples = raw[phase]
                    if len(samples) != (3 if phase == "priming" else 40):
                        raise ValueError("Unexpected locked raw sample count: " + phase)
                    if any(not math.isfinite(sample["host_ms"]) or sample["host_ms"] <= 0 for sample in samples):
                        raise ValueError("Invalid raw latency")
                records.append(dict(path=str(path), sha256=sha256(path), report=record))
            entry.update(case_index=str(index_path), case_reports=[r["path"] for r in records])
            self.save()
            return records
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.unmet("performance", "Invalid locked evidence for " + entry["label"] + ": " + str(error))
            return []

    def performance_summary(self, runs):
        groups = {}
        for run in runs:
            record = run["report"]
            if record.get("timing_kind") != "normal_wall":
                self.unmet("performance", "Diagnostic or unspecified timing cannot enter ABBA summary")
                continue
            key = (record["profile"], record["scenario"], record["case_key"], record["backend"],
                   record["expert_activation"], record["numerical_mode"], record["precision"], record.get("expert_type", "glu"))
            groups.setdefault(key, []).append(run)
        comparisons = []
        for key, records in sorted(groups.items()):
            modes = {mode: [r for r in records if r["mode"] == mode] for mode in ("sync", "async")}
            item = dict(zip(("profile", "scenario", "case", "backend",
                             "expert_activation", "numerical_mode", "precision", "expert_type"), key))
            item.update(state="complete" if all(len(modes[m]) == 2 for m in modes) else "incomplete",
                        runs=[dict(mode=r["mode"], order=r["order"], path=r["path"], sha256=r["sha256"],
                                   raw=r["report"]["raw"], frozen_config=r["report"]["frozen_config"])
                              for r in records], phases={})
            if item["state"] == "complete":
                # Ignore only execution-mode differences; all budgets/tiles/cache
                # fields, deterministic fixtures and routes must remain identical.
                signatures = []
                for run in records:
                    fields = dict(run["report"]["frozen_config"]["fields"])
                    fields.pop("async_staging", None)
                    signatures.append(dict(fields=fields, shape=run["report"]["shape"],
                                           seed=run["report"]["seed"],
                                           routes={phase: [sample["route"] for sample in run["report"]["raw"][phase]]
                                                   for phase in ("cold", "priming", "warm")}))
                if any(signature != signatures[0] for signature in signatures):
                    item["state"] = "incompatible_frozen_config"
                    self.unmet("performance", "ABBA frozen fields differ for " + str(key))
                else:
                    for phase in ("initialization", "cold", "warm"):
                        values = {mode: [sample["host_ms"] if isinstance(sample, dict) else sample
                                         for run in modes[mode] for sample in run["report"]["raw"].get(
                                             phase, run["report"]["raw"].get(phase + "_ms", []))]
                                  for mode in modes}
                        medians = {mode: statistics.median(values[mode]) for mode in modes}
                        item["phases"][phase] = dict(samples_ms=values, pooled_median_ms=medians,
                                                     sync_over_async_median=medians["sync"] / medians["async"])
                    warm = {mode: sum(sample["host_ms"] for run in modes[mode]
                                      for sample in run["report"]["raw"]["warm"]) for mode in modes}
                    item["warm_total_sync_over_async"] = warm["sync"] / warm["async"]
            comparisons.append(item)
        path = self.run_dir / "bench" / "performance_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, dict(schema_version=2, synthetic=True,
                             note="Process-level ABBA; unpaired pooled samples. Ratios are descriptive, "
                                  "not model speedup or confidence claims; no cross-profile aggregation.",
                             comparisons=comparisons))
        self.report["stages"]["performance"]["summary"] = str(path)
        self.save()

    def performance_trace(self, library):
        record = self.report["stages"]["performance"]
        if not self.args.performance_trace:
            record["trace"] = dict(state="skipped", reason="--no-performance-trace")
            self.save()
            return
        tool = shutil.which("nsys")
        if tool is None:
            record["trace"] = dict(state="unmet", reason="nsys unavailable")
            self.unmet("performance", "Mandatory CUDA timeline unavailable: nsys not found")
            return
        directory = self.run_dir / "trace"
        directory.mkdir(parents=True)
        prefix = directory / "locked_async"
        case = "fp32_accumulate/bank2/disabled"
        options = self.locked_options(library, directory / "evidence", "weight_transfer",
                                      "skewed_masked", case, "async", diagnostic=True)
        entry = self.command("performance", "nsys_trace",
                             [tool, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                              "--output", str(prefix), sys.executable, str(AOTI), *map(str, options)],
                             env={"FUSE_MOE_BENCHMARK_NVTX": "1"})
        record["trace"] = dict(state=entry["state"], report=str(prefix.with_suffix(".nsys-rep")),
                               diagnostic_only=True)
        if entry["state"] != "passed":
            self.save()
            return
        evidence = self.locked_evidence(entry, directory / "evidence", "weight_transfer",
                                        "skewed_masked", case, "async", diagnostic=True)
        if not evidence:
            record["trace"]["state"] = "unmet"
            self.save()
            return
        database = directory / "locked_async.sqlite"
        export = self.command("performance", "nsys_export",
                              [tool, "export", "--type=sqlite", "--output", str(database),
                               str(prefix.with_suffix(".nsys-rep"))])
        if export["state"] != "passed":
            record["trace"]["state"] = "failed"
            self.save()
            return
        try:
            timeline = gpu_overlap(database, evidence[0]["report"])
            write_json(directory / "gpu_overlap.json", timeline)
            record["trace"].update(state="passed", sqlite=str(database), overlap=timeline)
        except (OSError, sqlite3.Error, ValueError, KeyError) as error:
            record["trace"].update(state="unmet", error=str(error))
            self.unmet("performance", "CUDA/CUPTI timeline incomplete: " + str(error))
        self.save()

    def geglu_performance(self, library, runs):
        record = self.report["stages"]["performance"]
        if not getattr(self.args, "geglu_performance_matrix", True):
            record["geglu_matrix"] = dict(state="skipped", reason="--no-geglu-performance-matrix")
            self.save()
            return
        profile, scenario, case = (GEGLU_PERFORMANCE_PROFILE,
                                   GEGLU_PERFORMANCE_SCENARIO, GEGLU_PERFORMANCE_CASE)
        record["geglu_matrix"] = dict(state="running", profile=profile, scenario=scenario,
                                      case=case, precision=GEGLU_PERFORMANCE_PRECISION,
                                      modes=GEGLU_PERFORMANCE_MODES,
                                      note="Representative synthetic offload case; not P01-P05 full-GPU performance")
        entries = []
        for activation, numerical in GEGLU_PERFORMANCE_MODES:
            contract = (activation, numerical, GEGLU_PERFORMANCE_PRECISION)
            for index, mode in enumerate(("sync", "async", "async", "sync")):
                label = f"geglu_locked_{activation}_{numerical}_{index}_{mode}"
                directory = self.run_dir / "bench" / label
                options = self.locked_options(library, directory, profile, scenario, case, mode,
                                              contract=contract)
                entry = self.python("performance", label, AOTI, options)
                entries.append(entry)
                evidence = self.locked_evidence(entry, directory, profile, scenario, case, mode,
                                                contract=contract)
                if entry["state"] == "passed" and not evidence:
                    entry.update(state="unmet", error="Missing GEGLU matrix evidence")
                for item in evidence:
                    item.update(mode=mode, order=index)
                    runs.append(item)
        record["geglu_matrix"]["state"] = (
            "passed" if all(entry["state"] == "passed" for entry in entries) else "failed")
        self.save()

    def performance(self):
        library = self.library("performance", "prod")
        if not library:
            return
        cases = self.args.performance_cases
        order = ("sync", "async", "async", "sync")
        record = self.report["stages"]["performance"]
        record["method"] = dict(benchmark="--benchmark-locked", final_frozen=True,
                                profiles=PERFORMANCE_PROFILES, scenarios=PERFORMANCE_SCENARIOS,
                                cases=cases, order=order, iterations=40, route_tile_rows="auto",
                                tile_m=0, tile_n=128, normal_invocations=16 * len(cases),
                                cpp_requested=self.args.performance_cpp,
                                note="Singleton public CLI, no 24-case fallback; Python benchmark does not "
                                     "compile a C++ runner. Trace and C++ samples stay outside ABBA ratios.")
        runs = []
        try:
            for profile in PERFORMANCE_PROFILES:
                for scenario in PERFORMANCE_SCENARIOS:
                    for case in cases:
                        for index, mode in enumerate(order):
                            label = f"locked_{profile}_{scenario}_{case.replace('/', '_')}_{index}_{mode}"
                            directory = self.run_dir / "bench" / label
                            options = self.locked_options(library, directory, profile, scenario, case, mode)
                            entry = self.python("performance", label, AOTI, options)
                            for evidence in self.locked_evidence(entry, directory, profile, scenario, case, mode):
                                evidence.update(mode=mode, order=index)
                                runs.append(evidence)
            self.geglu_performance(library, runs)
            if self.args.performance_cpp:
                # One representative selected configuration per cache, each in
                # both modes; runner builds are opt-in and limited to eight.
                representatives = {}
                for case in cases:
                    representatives.setdefault(case.rsplit("/", 1)[1], case)
                record["cpp"] = dict(state="required", cases=list(representatives.values()))
                for case in representatives.values():
                    for mode in ("sync", "async"):
                        label = "locked_cpp_" + case.replace("/", "_") + "_" + mode
                        directory = self.run_dir / "bench" / label
                        options = self.locked_options(library, directory, "weight_transfer", "hotset_transition",
                                                      case, mode, cpp=True)
                        entry = self.python("performance", label, AOTI, options)
                        self.locked_evidence(entry, directory, "weight_transfer", "hotset_transition", case, mode, cpp=True)
            else:
                record["cpp"] = dict(state="skipped", reason="optional --performance-cpp not selected")
            self.performance_trace(library)
        finally:
            self.performance_summary(runs)

    def run(self):
        interrupted = False
        try:
            self.collect()
            for stage in STAGES:
                record = self.report["stages"][stage]
                if record["state"] == "skipped":
                    continue
                record.update(state="running", started_at=utc_now())
                self.save()
                try:
                    getattr(self, stage)()
                except KeyboardInterrupt:
                    record["state"] = "interrupted"
                    raise
                except Exception:
                    record["error"] = traceback.format_exc()
                    record["unmet"].append("Stage raised an exception; see error")
                entries = [e for e in self.report["commands"] if e["stage"] == stage]
                record.update(state="failed" if record["unmet"] or any(
                    e["state"] != "passed" for e in entries) else "passed", ended_at=utc_now())
                self.save()
        except KeyboardInterrupt:
            interrupted = True
            self.report["unmet"].append("Delivery interrupted; unexecuted stages are not successes")
        except Exception:
            self.report["error"] = traceback.format_exc()
            self.report["unmet"].append("Delivery collection/execution failed")
        finally:
            try:
                after = source_manifest(self.output)
                self.report["source_sha256_after"] = after
                if self.report.get("source_sha256") != after:
                    self.report["unmet"].append("Source tree changed during delivery; evidence is inconsistent")
            except Exception as error:
                self.report["unmet"].append("Cannot verify final source hashes: " + str(error))
            for record in self.report["stages"].values():
                if record["state"] in ("pending", "running"):
                    record.update(state="not_run", reason="delivery terminated before completion")
            selected = [r for r in self.report["stages"].values() if r["state"] != "skipped"]
            failed = bool(self.report["unmet"]) or not selected or any(r["state"] != "passed" for r in selected)
            code = 130 if interrupted else (1 if failed else 0)
            complete = (all(r["state"] == "passed" for r in self.report["stages"].values())
                        and self.report["stages"]["performance"].get("trace", {}).get("state") != "skipped"
                        and self.args.performance_cases == list(PERFORMANCE_CASES)
                        and self.report["stages"]["performance"].get("geglu_matrix", {}).get("state") == "passed")
            bf16 = self.report["stages"]["aoti"].get("wide_bf16", {})
            self.report.update(state="interrupted" if interrupted else ("failed" if failed else (
                "passed" if complete and bf16.get("state") != "skipped" else "partial")),
                               selected_stages_passed=not failed,
                               full_delivery_passed=not failed and complete and bf16.get("state") != "skipped",
                               exit_code=code, ended_at=utc_now())
            self.save()
        print("DELIVERY " + self.report["state"] + ": " + str(self.output / "report.json"), flush=True)
        return code


def main():
    args = parser_for_cli()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Hold throughout all artifact access. Separate run directories avoid archive
    # mixing, while native/{prod,testing} can be reused by explicit stage selection.
    with (args.output_dir / ".delivery.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Delivery directory is already in use; existing evidence left untouched", file=sys.stderr)
            return 1
        previous = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            return Delivery(args).run()
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
