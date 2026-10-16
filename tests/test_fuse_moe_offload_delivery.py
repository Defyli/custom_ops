"""CPU-only regression tests for delivery evidence acceptance; no Torch imports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
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


if __name__ == "__main__":
    unittest.main()
