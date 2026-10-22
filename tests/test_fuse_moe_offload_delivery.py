"""CPU-only regression tests for delivery evidence acceptance; no Torch imports."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


_SPEC = importlib.util.spec_from_file_location(
    "offload_delivery_harness", Path(__file__).with_name("run_fuse_moe_offload_delivery.py"))
HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(HARNESS)


class SanitizerEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.delivery = object.__new__(HARNESS.Delivery)
        self.delivery.save = mock.Mock()

    def entry(self, text, state="passed", returncode=0):
        stdout, stderr = self.root / "stdout.txt", self.root / "stderr.txt"
        stdout.write_text(text, encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        return dict(state=state, returncode=returncode, stdout=str(stdout), stderr=str(stderr))

    def test_official_and_legacy_zero_leak_summaries(self):
        for full_leak in (False, True):
            for phrase in ("bytes leaked in", "bytes in"):
                with self.subTest(full_leak=full_leak, phrase=phrase):
                    entry = self.entry("========= LEAK SUMMARY: 0 " + phrase
                                       + " 0 allocations\n========= ERROR SUMMARY: 0 errors\n")
                    self.assertTrue(self.delivery.sanitizer_result(entry, full_leak=full_leak))
                    self.assertEqual(entry["sanitizer_leak_counts"], [dict(bytes=0, allocations=0)])

    def test_access_check_requires_error_summary_but_not_leak_summary(self):
        entry = self.entry("========= ERROR SUMMARY: 0 errors\n")
        self.assertTrue(self.delivery.sanitizer_result(entry, full_leak=False))
        entry = self.entry("========= ERROR SUMMARY: 0 errors\n")
        self.assertFalse(self.delivery.sanitizer_result(entry, full_leak=True))
        self.assertEqual(entry["state"], "unmet")

    def test_missing_error_or_all_summaries_never_pass(self):
        for full_leak in (False, True):
            for text in ("", "========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\n"):
                with self.subTest(full_leak=full_leak, text=text):
                    self.assertFalse(self.delivery.sanitizer_result(self.entry(text), full_leak=full_leak))

    def test_nonzero_errors_fail_both_scopes(self):
        for full_leak in (False, True):
            text = "ERROR SUMMARY: 1 error\nLEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
            self.assertFalse(self.delivery.sanitizer_result(self.entry(text), full_leak=full_leak))

    def test_nonzero_leaks_fail_full_scope_even_when_errors_zero(self):
        for phrase in ("bytes leaked in", "bytes in"):
            for size, count in (("82,000,000", "30"), ("0", "1"), ("1", "0")):
                with self.subTest(phrase=phrase, size=size, count=count):
                    text = f"ERROR SUMMARY: 0 errors\nLEAK SUMMARY: {size} {phrase} {count} allocations\n"
                    entry = self.entry(text)
                    self.assertFalse(self.delivery.sanitizer_result(entry, full_leak=True))
                    self.assertEqual(entry["sanitizer_leak_counts"],
                                     [dict(bytes=int(size.replace(",", "")), allocations=int(count))])

    def test_arbitrary_leak_text_is_not_accepted(self):
        entry = self.entry("ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes maybe leaked in 0 allocations\n")
        self.assertFalse(self.delivery.sanitizer_result(entry, full_leak=True))

    def test_any_nonzero_summary_prevents_full_acceptance(self):
        entry = self.entry("ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
                           "ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 32 bytes leaked in 1 allocation\n")
        self.assertFalse(self.delivery.sanitizer_result(entry, full_leak=True))

    def test_failed_or_incomplete_child_is_never_reclassified(self):
        text = "ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
        for full_leak in (False, True):
            for state, code in (("failed", 99), ("failed", -11), ("timeout", -9),
                                ("interrupted", -15), ("unmet", 0), ("unavailable", None)):
                with self.subTest(full_leak=full_leak, state=state):
                    entry = self.entry(text, state, code)
                    self.assertFalse(self.delivery.sanitizer_result(entry, full_leak=full_leak))
                    self.assertEqual((entry["state"], entry["returncode"]), (state, code))
                    self.assertEqual(entry["sanitizer_error_counts"], [0])


class NativeTeardownTests(unittest.TestCase):
    setUp = SanitizerEvidenceTests.setUp
    entry = SanitizerEvidenceTests.entry

    def native_entry(self, changes=None, marker=True, state="passed", returncode=0):
        summary = dict(fault_cases=14, sync_fault_cases=5, async_fault_cases=9, worker_state_cases=7,
                       concurrent_context_cases=14, healthy_checks=16, concurrent_healthy_calls=42,
                       allocation_acknowledgements=4, poison_ack_rejections=10,
                       callback_output_unwind_cases=2, allocated_before_trim=0, reserved_before_trim=2097152,
                       allocated_after_trim=0, reserved_after_trim=0,
                       cuda_allocator_empty_cache=True, pinned_allocator_empty_cache=True)
        summary.update(changes or {})
        text = "NATIVE_TEARDOWN " + json.dumps(summary) + "\n"
        if marker:
            text += "PASS NATIVE faults full allocator teardown Python_uninitialized\n"
        text += "ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes leaked in 0 allocations\n"
        return self.entry(text, state, returncode)

    def invoke_native(self, entry):
        directory = self.root / "native_faults"
        directory.mkdir(exist_ok=True)
        executable = directory / "fuse_moe_offload_aoti_runner"
        executable.write_text("fixture executable, never executed\n", encoding="utf-8")
        executable.chmod(0o700)
        self.delivery.run_dir = self.root
        self.delivery.report = dict(telemetry=dict(device=dict(device_index=0)))
        self.delivery.python = mock.Mock(return_value=dict(state="passed"))
        self.delivery.command = mock.Mock(return_value=entry)
        self.delivery.unmet = mock.Mock()
        self.delivery.native_fault_leaks("compute-sanitizer", self.root / "testing.so")
        command = self.delivery.command.call_args.args[2]
        self.assertEqual(command[command.index("--leak-check") + 1], "full")
        self.assertIn("--native-faults", command)
        self.assertEqual(command[command.index("--error-exitcode") + 1], "99")
        self.assertIn("--compile-native-runner-only", self.delivery.python.call_args.args[3])
        return entry

    def test_exact_separate_healthy_counts_pass(self):
        entry = self.invoke_native(self.native_entry())
        self.assertEqual(entry["state"], "passed")
        self.assertEqual(entry["native_teardown"]["healthy_checks"], 16)
        self.assertEqual(entry["native_teardown"]["concurrent_healthy_calls"], 42)

    def test_wrong_missing_or_wrong_type_counts_fail(self):
        for changes in (dict(healthy_checks=42), dict(healthy_checks=15), dict(healthy_checks="16"),
                        dict(healthy_checks=None), dict(concurrent_healthy_calls=16),
                        dict(fault_cases=13), dict(worker_state_cases=6), dict(allocated_after_trim=1),
                        dict(reserved_after_trim=1), dict(pinned_allocator_empty_cache=1)):
            with self.subTest(changes=changes):
                self.assertEqual(self.invoke_native(self.native_entry(changes))["state"], "unmet")

    def test_no_interpreter_marker_is_required(self):
        self.assertEqual(self.invoke_native(self.native_entry(marker=False))["state"], "unmet")

    def test_native_nonzero_exit_remains_failed_even_with_zero_summaries(self):
        entry = self.invoke_native(self.native_entry(state="failed", returncode=99))
        self.assertEqual((entry["state"], entry["returncode"]), ("failed", 99))
        self.assertNotIn("native_teardown", entry)


class GegluDeliveryCoverageTests(unittest.TestCase):
    """Ensure delivery evidence includes GEGLU/staged, not only SwiGLU.

    The unit stage executes the whole offload test module on the testing build.
    These CPU-only checks pin that the class is module-level (so
    ``python tests/test_fuse_moe_offload.py GEGLUTests`` selects it) and that
    the harness treats CUDA skips as unmet evidence instead of silent success.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def delivery(self):
        delivery = object.__new__(HARNESS.Delivery)
        delivery.logs = self.root
        delivery.report = {"commands": [],
                           "stages": {"unit": {"commands": [], "unmet": []}}}
        delivery.env = dict(os.environ)
        delivery.args = SimpleNamespace(unit_timeout=30)
        delivery.save = mock.Mock()
        return delivery

    def test_geglu_class_and_single_class_cli_targets(self):
        tree = ast.parse(Path(HARNESS.OFFLOAD).read_text(encoding="utf-8"))
        classes = [node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GEGLUTests"]
        self.assertEqual(len(classes), 1)
        geglu = classes[0]
        methods = {node.name for node in geglu.body
                   if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
        self.assertLessEqual({
            "test_full_gpu_four_mode_dtype_matrix",
            "test_isolated_scalar_semantics_wide_n_atomic",
            "test_offload_route_fp32_cache_and_partial_tiles",
            "test_multi_context_modes_do_not_leak",
            "test_native_v1_v2_contract_and_failure_before_token_mutation",
            "test_masked_topk_and_zero_sequence",
        }, methods)
        self.assertEqual(
            {f"GEGLUTests.{name}" for name in methods},
            set(HARNESS.GEGLU_SANITIZER_TARGETS))
        # GPU coverage must skip (not fail) on CPU-only delivery stages.
        decorators = ast.dump(ast.Module(body=geglu.decorator_list, type_ignores=[]))
        self.assertIn("skipUnless", decorators)
        self.assertIn("is_available", decorators)
        self.assertIn("'cuda'", decorators)
        # The documented main CLI keeps unittest class selection available.
        main_blocks = [node for node in tree.body
                       if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)]
        self.assertEqual(len(main_blocks), 1)
        self.assertTrue(any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "unittest" and node.func.attr == "main"
            for node in ast.walk(main_blocks[0])))

    def test_skipped_geglu_child_is_unmet_not_passed(self):
        skipped = (
            "import unittest\n"
            "class GEGLUTests(unittest.TestCase):\n"
            "    def test_cuda_case(self):\n"
            "        self.skipTest('CUDA required')\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n")
        entry = self.delivery().command(
            "unit", "geglu_skipped", [sys.executable, "-c", skipped], unittest=True)
        self.assertEqual(entry["state"], "unmet")
        self.assertEqual(entry["tests_run"], 1)
        self.assertEqual(entry["tests_skipped"], 1)
        healthy = (
            "import unittest\n"
            "class GEGLUTests(unittest.TestCase):\n"
            "    def test_cuda_case(self):\n"
            "        self.assertTrue(True)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n")
        entry = self.delivery().command(
            "unit", "geglu_healthy", [sys.executable, "-c", healthy], unittest=True)
        self.assertEqual(entry["state"], "passed")
        self.assertEqual((entry["tests_run"], entry["tests_skipped"]), (1, 0))


class GegluPerformanceEvidenceTests(unittest.TestCase):
    """Pin mode/dtype partitioning; synthetic evidence is never GPU acceptance."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.delivery = object.__new__(HARNESS.Delivery)
        self.delivery.run_dir = self.root
        self.delivery.args = SimpleNamespace(benchmark_activation="silu",
                                             benchmark_numerical_mode="fused",
                                             benchmark_precision="fp16",
                                             geglu_performance_matrix=True)
        self.delivery.report = {"stages": {"performance": {}}}
        self.delivery.save = mock.Mock()
        self.delivery.unmet = mock.Mock()

    def test_contract_override_does_not_mutate_default_options(self):
        args = ("ops.so", self.root, "small", "uniform",
                "fp32_accumulate/bank2/disabled", "async")
        override = self.delivery.locked_options(*args, contract=("gelu", "staged", "bf16"))
        for key, value in (("--benchmark-activation", "gelu"),
                           ("--benchmark-numerical-mode", "staged"),
                           ("--benchmark-precision", "bf16")):
            self.assertEqual(override[override.index(key) + 1], value)
        default = self.delivery.locked_options(*args)
        self.assertEqual(default[default.index("--benchmark-activation") + 1], "silu")
        self.assertEqual(default[default.index("--benchmark-numerical-mode") + 1], "fused")
        self.assertEqual(default[default.index("--benchmark-precision") + 1], "fp16")

    def test_geglu_four_modes_each_get_an_independent_abba_run(self):
        self.delivery.python = mock.Mock(return_value={"state": "passed"})
        self.delivery.locked_evidence = mock.Mock(side_effect=lambda *args, **kwargs: [{}])
        runs = []
        self.delivery.geglu_performance("ops.so", runs)
        self.assertEqual(self.delivery.python.call_count, 16)
        self.assertEqual(len(runs), 16)
        self.assertEqual({mode for mode in HARNESS.GEGLU_PERFORMANCE_MODES},
                         {(act, num) for act in ("silu", "gelu") for num in ("fused", "staged")})
        calls = self.delivery.locked_evidence.call_args_list
        for offset, mode in enumerate(HARNESS.GEGLU_PERFORMANCE_MODES):
            block = calls[4 * offset:4 * offset + 4]
            self.assertEqual([call.args[5] for call in block], ["sync", "async", "async", "sync"])
            self.assertTrue(all(call.kwargs["contract"] == (*mode, "bf16") for call in block))
        self.assertEqual(self.delivery.report["stages"]["performance"]["geglu_matrix"]["state"], "passed")

    def test_omission_is_skipped_not_passed(self):
        self.delivery.args.geglu_performance_matrix = False
        self.delivery.python = mock.Mock()
        self.delivery.geglu_performance("ops.so", [])
        self.delivery.python.assert_not_called()
        self.assertEqual(self.delivery.report["stages"]["performance"]["geglu_matrix"]["state"], "skipped")

    def test_summary_partitions_all_activation_numerical_precision_pairs(self):
        runs = []
        for activation, numerical in HARNESS.GEGLU_PERFORMANCE_MODES:
            for precision in ("fp16", "bf16"):
                for index, mode in enumerate(("sync", "async", "async", "sync")):
                    raw = {phase: [{"host_ms": 1. if mode == "sync" else .5, "route": "uniform"}]
                           for phase in ("initialization", "cold", "priming", "warm")}
                    record = dict(profile="small", scenario="uniform",
                                  case_key="fp32_accumulate/bank2/disabled", backend="python_public_aoti",
                                  expert_activation=activation, numerical_mode=numerical, precision=precision,
                                  timing_kind="normal_wall",
                                  frozen_config={"fields": {"async_staging": mode == "async",
                                                           "expert_activation": activation,
                                                           "numerical_mode": numerical}},
                                  shape={"S": 64, "H": 64, "I": 64, "E": 16, "K": 2}, seed=42, raw=raw)
                    runs.append(dict(report=record, mode=mode, order=index, path="synthetic.json", sha256="fixture"))
        self.delivery.performance_summary(runs)
        summary = json.loads((self.root / "bench/performance_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(len(summary["comparisons"]), 8)
        self.assertTrue(all(item["state"] == "complete" for item in summary["comparisons"]))
        self.assertTrue(all(item["phases"]["warm"]["sync_over_async_median"] == 2
                            for item in summary["comparisons"]))
        self.delivery.unmet.assert_not_called()

    def test_diagnostic_timings_cannot_enter_summary(self):
        self.delivery.performance_summary([{"report": {"timing_kind": "host_profile"}}])
        self.delivery.unmet.assert_called_once()
        summary = json.loads((self.root / "bench/performance_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["comparisons"], [])

    def test_baseline_skip_or_missing_json_is_unmet(self):
        path = self.root / "baseline.json"
        entry = dict(state="passed")
        self.delivery.baseline_evidence(entry, path)
        self.assertEqual(entry["state"], "unmet")
        report = dict(gpu={"state": "skipped"}, summary={"state": "gpu_skipped"})
        HARNESS.write_json(path, report)
        entry = dict(state="passed")
        self.delivery.baseline_evidence(entry, path)
        self.assertEqual(entry["state"], "unmet")
        report = dict(gpu={"state": "passed"}, summary=dict(state="passed", matrix_cases=4,
                    matrix_passed=4, matrix_failed=0, cpu_and_gpu_checks={"FAIL": 0, "PASS": 4}))
        HARNESS.write_json(path, report)
        entry = dict(state="passed")
        self.delivery.baseline_evidence(entry, path)
        self.assertEqual(entry["state"], "passed")
        self.assertEqual(entry["matrix_cases"], 4)

    def test_locked_record_modes_and_storage_codes_are_verified(self):
        profile, scenario, case = "small", "uniform", "fp32_accumulate/bank2/disabled"
        directory = self.root / "locked"
        case_dir = directory / "locked_small_fp32_accumulate_2_disabled"
        case_dir.mkdir(parents=True)
        expected = dict(profile=profile, scenario=scenario, case_key=case)
        HARNESS.write_json(directory / "locked_benchmark_index.json",
                           dict(final_frozen=True, backend_count=1, cases=[expected]))
        vector = [0] * 25
        vector[0], vector[20], vector[23], vector[24] = 2, 1, 1, 1
        record = dict(**expected, expert_activation="gelu", numerical_mode="staged", precision="bf16",
                      timing_kind="normal_wall", frozen_config={
                          "fields": {"async_staging": True, "expert_activation": "gelu",
                                     "numerical_mode": "staged"}, "layer_config": vector},
                      raw={phase: [{"host_ms": 1.0}] * count
                           for phase, count in (("initialization", 41), ("cold", 40),
                                                ("priming", 3), ("warm", 40))})
        path = case_dir / "uniform.json"
        HARNESS.write_json(path, record)
        entry = dict(state="passed", label="synthetic_locked")
        self.assertEqual(len(self.delivery.locked_evidence(
            entry, directory, profile, scenario, case, "async", contract=("gelu", "staged", "bf16"))), 1)
        for field, wrong in (("expert_activation", "silu"), ("numerical_mode", "fused"),
                             ("precision", "fp16")):
            with self.subTest(field=field):
                bad = dict(record, **{field: wrong})
                HARNESS.write_json(path, bad)
                self.assertEqual(self.delivery.locked_evidence(
                    entry, directory, profile, scenario, case, "async",
                    contract=("gelu", "staged", "bf16")), [])
        for index, wrong in ((20, 0), (23, 0), (24, 0)):
            with self.subTest(index=index):
                bad_vector = list(vector)
                bad_vector[index] = wrong
                bad = dict(record, frozen_config=dict(record["frozen_config"], layer_config=bad_vector))
                HARNESS.write_json(path, bad)
                self.assertEqual(self.delivery.locked_evidence(
                    entry, directory, profile, scenario, case, "async",
                    contract=("gelu", "staged", "bf16")), [])


class MlpDeliveryEvidenceTests(unittest.TestCase):
    """Host-only checks; synthetic reports are not GPU acceptance evidence."""

    setUp = GegluPerformanceEvidenceTests.setUp

    def test_independent_mlp_class_targets_and_frozen_runner_shapes(self):
        tree = ast.parse(Path(HARNESS.OFFLOAD).read_text(encoding="utf-8"))
        classes = [node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "MLPTests"]
        self.assertEqual(len(classes), 1)
        methods = {node.name for node in classes[0].body
                   if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
        self.assertEqual({f"MLPTests.{name}" for name in methods},
                         set(HARNESS.MLP_SANITIZER_TARGETS))
        runner = Path(HARNESS.OFFLOAD).with_name("fuse_moe_offload_aoti_runner.cpp").read_text()
        self.assertIn("w1.size(1) == projection_factor * w2.size(2)", runner)
        self.assertIn("{E, projection_factor * I, expected_h}", runner)
        self.assertIn('require(v[25] == 1, "LOCKED archive invalid expert_type")', runner)
        self.assertIn('"--mlp"', runner)

    def test_v3_evidence_requires_explicit_mlp_and_type_code(self):
        directory = self.root / "mlp"
        directory.mkdir()
        package = directory / "mlp_chain_bf16.pt2"
        archive = directory / "mlp_weights_bf16.pt"
        package.write_text("synthetic package, never loaded")
        archive.write_text("synthetic archive, never loaded")
        vector = [0] * 26
        vector[0], vector[23], vector[24], vector[25] = 3, 1, 1, 1
        report = dict(schema_version=1, state="passed", archive_format_version=2,
                      manifest_framing=1, precision="bf16", expert_type="mlp",
                      expert_activation="gelu", numerical_mode="staged",
                      package=str(package.resolve()), archive=str(archive.resolve()),
                      config=dict(expert_type="mlp", expert_activation="gelu",
                                  numerical_mode="staged", async_staging=True, host_profile=True),
                      layer_configs={"0": vector, "5": vector}, python_cases=[{}] * 10,
                      cpp_cases=[dict(S=s, precision="bf16", expert_type="mlp",
                                      expert_activation="gelu", numerical_mode="staged",
                                      python_uninitialized=True, strict_no_jit=True,
                                      token_checked=True, status_checked=True,
                                      reference_checked=True, async_staging_enabled=True,
                                      host_profile_enabled=True) for s in (2, 7, 64, 2)])
        HARNESS.write_json(directory / "acceptance.exit.json", dict(state="passed", returncode=0))
        report_path = directory / "mlp_report.json"
        HARNESS.write_json(report_path, report)
        entry = dict(state="passed", label="synthetic_mlp")
        self.assertIsNotNone(self.delivery.geglu_evidence(entry, directory, "bf16", "mlp"))
        for mismatch in ("report", "config", "cpp", "version", "type", "length"):
            with self.subTest(mismatch=mismatch):
                bad = json.loads(json.dumps(report))
                if mismatch == "report":
                    bad.pop("expert_type")
                elif mismatch == "config":
                    bad["config"]["expert_type"] = "glu"
                elif mismatch == "cpp":
                    bad["cpp_cases"][0]["expert_type"] = "glu"
                elif mismatch == "version":
                    bad["layer_configs"]["0"][0] = 2
                elif mismatch == "type":
                    bad["layer_configs"]["0"][25] = 0
                else:
                    bad["layer_configs"]["0"].append(0)
                HARNESS.write_json(report_path, bad)
                entry = dict(state="passed", label="synthetic_mlp")
                self.assertIsNone(self.delivery.geglu_evidence(entry, directory, "bf16", "mlp"))
                self.assertEqual(entry["state"], "unmet")

    def test_performance_summary_never_merges_glu_and_mlp(self):
        runs = []
        for expert_type in ("glu", "mlp"):
            for index, mode in enumerate(("sync", "async", "async", "sync")):
                record = dict(profile="small", scenario="uniform", case_key="same",
                              backend="python_public_aoti", expert_activation="gelu",
                              numerical_mode="staged", precision="bf16", expert_type=expert_type,
                              timing_kind="normal_wall", frozen_config={"fields": {
                                  "expert_type": expert_type, "async_staging": mode == "async"}},
                              shape={"S": 64, "H": 64, "I": 64, "E": 16, "K": 2}, seed=42,
                              raw={phase: [{"host_ms": 1. if mode == "sync" else .5,
                                            "route": "uniform"}]
                                   for phase in ("initialization", "cold", "priming", "warm")})
                runs.append(dict(report=record, mode=mode, order=index,
                                 path="synthetic.json", sha256="fixture"))
        self.delivery.performance_summary(runs)
        summary = json.loads((self.root / "bench/performance_summary.json").read_text())
        self.assertEqual(len(summary["comparisons"]), 2)
        self.assertEqual({item["expert_type"] for item in summary["comparisons"]}, {"glu", "mlp"})
        self.assertTrue(all(item["state"] == "complete" for item in summary["comparisons"]))
        self.delivery.unmet.assert_not_called()


if __name__ == "__main__":
    unittest.main()
