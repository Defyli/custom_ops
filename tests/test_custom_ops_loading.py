"""CPU-only build/loading contract tests; no native compilation or GPU kernels.

Run: PYTHONPATH=/workdir python tests/test_custom_ops_loading.py
The mocked dlopen registers real CPU dispatcher schemas/kernels, so schema and
fake verification are exercised without depending on nvcc, Ninja or a GPU.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib.abc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

import torch  # noqa: E402

import custom_ops  # noqa: E402
from custom_ops import CustomOps, CustomOpsError  # noqa: E402
from custom_ops.recsys import FAOps, FuseMoeOps, MixedGemmOps, RecsysOps, SwigluOps  # noqa: E402


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="custom-ops-loading-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registrations = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Never disturb other libraries in the test runner process.
        self.stack.enter_context(mock.patch.dict(custom_ops._LIBRARIES, clear=True))
        self.stack.enter_context(mock.patch.dict(custom_ops._OP_OWNERS, clear=True))
        self.stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=False))
        self.stack.enter_context(mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": ""}))
        self.compiler_setup = self.stack.enter_context(mock.patch("custom_ops.setup_compiler"))
        self.dlopen = self.stack.enter_context(mock.patch.object(torch.ops, "load_library"))

    def ops_class(self, *, namespace=None, cuda=False, fake=True):
        namespace = namespace or "custom_loading_" + uuid.uuid4().hex
        source_root = self.root / namespace
        source_root.mkdir()
        source = source_root / ("binding.cu" if cuda else "binding.cpp")
        source.write_text("// test source\n", encoding="utf-8")
        (source_root / "header.h").write_text("// test header\n", encoding="utf-8")

        class TestOps(CustomOps):
            so_name = namespace + "_kernel"
            requires_cuda = cuda
            required_ops = ["identity"]
            op_schemas = {"identity": "identity(Tensor x) -> Tensor"}
            fake_ops = ["identity"] if fake else []

            def get_sources(self):
                return [str(source)]

            def get_include_dirs(self):
                return [str(source_root)]

            def register_fake_impls(self):
                if fake:
                    self.register_fake_for("identity", lambda x: torch.empty_like(x))

        TestOps.namespace = namespace
        return TestOps

    def artifact(self, cls, *, targets="8.9", directory=None):
        instance = cls()
        inputs = instance._build_inputs(targets if instance.requires_cuda else None)
        toolchain = {"cxx": {"command": "c++", "version": "test compiler"}}
        if inputs["with_cuda"]:
            toolchain["cuda_toolkit_version"] = torch.version.cuda
            toolchain["nvcc"] = {"command": "nvcc", "version": "test nvcc"}
        metadata = instance._build_metadata(inputs, instance._source_fingerprint(inputs), toolchain)
        output = directory or self.root / (cls.so_name + "_output")
        output.mkdir(exist_ok=True)
        library = output / (cls.so_name + ".so")
        library.write_bytes(b"mock shared library; never passed to real dlopen")
        metadata["library"] = {
            "filename": library.name, "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "size": library.stat().st_size,
        }
        self.save_metadata(library, metadata)
        return library, metadata

    def save_metadata(self, library, metadata, *, rehash=True, sidecar=None):
        import json
        if rehash:
            metadata["build_fingerprint"] = custom_ops._metadata_fingerprint(metadata)
        sidecar = sidecar or Path(str(library) + ".json")
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        return sidecar

    def install(self, cls, *, schema=None, meta=False, extra_ops=()):
        library = torch.library.Library(cls.namespace, "FRAGMENT")
        library.define(schema or cls.op_schemas["identity"])
        library.impl("identity", lambda x: x.clone(), "CPU")
        if meta:
            library.impl("identity", lambda x: torch.empty_like(x), "Meta")
        for name in extra_ops:
            library.define(f"{name}(Tensor x) -> Tensor")
            library.impl(name, lambda x: x.clone(), "CPU")
        self.registrations.append(library)
        return library

    def assert_pre_dlopen_rejected(self, cls, library, message):
        self.dlopen.reset_mock()
        instance = cls()
        with mock.patch.object(cls, "_compile_library", side_effect=AssertionError("compiled")) as compile_, \
                self.assertRaisesRegex(CustomOpsError, message):
            instance.load_precompiled(library)
        compile_.assert_not_called()
        self.dlopen.assert_not_called()
        self.assertFalse(instance.allow_jit)
        self.assertFalse(instance.is_available())
        self.assertIsNone(instance.library_path)
        self.assertIsNotNone(instance.load_error())
        self.assertFalse(custom_ops._LIBRARIES)

    def test_import_in_fresh_process_is_lazy_without_build_dependencies(self):
        # Block build dependency imports before importing the package. This also
        # proves import itself does not probe GPU capability or call nvcc/Ninja.
        program = """
import importlib.abc
import sys
import torch
from unittest.mock import patch
class BlockBuild(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch.utils.cpp_extension' or fullname.startswith('setuptools'):
            raise AssertionError('build dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockBuild())
with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU probed')):
    from custom_ops import ops
    from custom_ops.recsys import FuseMoeOps
    assert ops.allow_jit is True
    assert set(ops.groups.values()) == {'not_loaded'}
    assert FuseMoeOps().allow_jit is True
"""
        env = dict(os.environ, PYTHONPATH=_PACKAGE_PARENT, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run([sys.executable, "-c", program], env=env, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_precompiled_load_in_fresh_process_without_build_dependencies(self):
        cls = self.ops_class(fake=False)
        library, _ = self.artifact(cls)
        program = """
import importlib.abc
import sys
import torch
from unittest.mock import patch
assert 'torch.utils.cpp_extension' not in sys.modules
class BlockBuild(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch.utils.cpp_extension' or fullname.startswith('setuptools'):
            raise AssertionError('build dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockBuild())
from custom_ops import CustomOps
class DeploymentOps(CustomOps):
    namespace = sys.argv[1]
    so_name = sys.argv[2]
    requires_cuda = False
    required_ops = ['identity']
    op_schemas = {'identity': 'identity(Tensor x) -> Tensor'}
    def get_sources(self):
        raise AssertionError('source access')
    def get_include_dirs(self):
        raise AssertionError('include access')
registrations = []
def dlopen(path):
    lib = torch.library.Library(DeploymentOps.namespace, 'FRAGMENT')
    lib.define('identity(Tensor x) -> Tensor')
    lib.impl('identity', lambda x: x.clone(), 'CPU')
    registrations.append(lib)
with patch.object(torch.ops, 'load_library', side_effect=dlopen) as load, \\
        patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU probed')):
    instance = DeploymentOps().load_precompiled(sys.argv[3])
    assert instance.is_available()
    assert not instance.allow_jit
    load.assert_called_once()
    x = torch.ones(2)
    assert torch.equal(getattr(torch.ops, DeploymentOps.namespace).identity(x), x)
assert 'torch.utils.cpp_extension' not in sys.modules
"""
        env = dict(os.environ, PYTHONPATH=_PACKAGE_PARENT, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(
            [sys.executable, "-c", program, cls.namespace, cls.so_name, str(library)],
            env=env, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_strict_custom_ops_never_checks_cache_or_compiler(self):
        cls = self.ops_class()
        instance = cls(allow_jit=False)
        with mock.patch.object(cls, "_load_extension", side_effect=AssertionError("JIT")) as load_, \
                mock.patch("custom_ops.get_build_dir", side_effect=AssertionError("cache")) as cache:
            with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                instance.load()
        load_.assert_not_called()
        cache.assert_not_called()
        self.compiler_setup.assert_not_called()
        self.assertFalse(instance.is_available())
        self.assertIn("load_precompiled", instance.load_error())

    def test_policy_is_explicit_bool_and_chainable(self):
        cls = self.ops_class()
        for cls_ in (cls, RecsysOps):
            instance = cls_()
            self.assertTrue(instance.allow_jit)
            self.assertIs(instance.disable_jit(), instance)
            self.assertFalse(instance.allow_jit)
            self.assertIs(instance.configure(allow_jit=True), instance)
            instance.allow_jit = False
            self.assertFalse(instance.allow_jit)
            for value in (0, 1, None, "false"):
                with self.subTest(cls=cls_.__name__, value=value):
                    with self.assertRaises(TypeError):
                        cls_(allow_jit=value)

    def test_missing_files_and_metadata_fail_before_dlopen(self):
        cls = self.ops_class()
        self.assert_pre_dlopen_rejected(cls, self.root / "missing.so", "No such file")
        library, _ = self.artifact(cls)
        Path(str(library) + ".json").unlink()
        self.assert_pre_dlopen_rejected(cls, library, "No such file")
        Path(str(library) + ".json").write_text("[]", encoding="utf-8")
        self.assert_pre_dlopen_rejected(cls, library, "JSON object")
        Path(str(library) + ".json").write_text("{", encoding="utf-8")
        self.assert_pre_dlopen_rejected(cls, library, "Precompiled load failed")

    def test_known_compatibility_and_contract_mismatch_reject_before_dlopen(self):
        cls = self.ops_class()
        library, original = self.artifact(cls)
        cases = {
            "format_version": (lambda m: m.update(format_version=999), "format_version"),
            "namespace": (lambda m: m.update(namespace="wrong"), "namespace"),
            "so_name": (lambda m: m.update(so_name="wrong"), "so_name"),
            "build_version": (lambda m: m.update(build_version="999"), "build_version"),
            "torch_version": (lambda m: m["compatibility"].update(torch_version="0.0"), "torch_version"),
            "torch_git_version": (lambda m: m["compatibility"].update(torch_git_version="wrong"), "torch_git_version"),
            "torch_cuda_version": (lambda m: m["compatibility"].update(torch_cuda_version="0.0"), "torch_cuda_version"),
            "cxx11_abi": (lambda m: m["compatibility"].update(cxx11_abi=not m["compatibility"]["cxx11_abi"]), "cxx11_abi"),
            "machine": (lambda m: m["compatibility"].update(machine="other"), "machine"),
            "required_ops": (lambda m: m.update(required_ops=[]), "required_ops"),
            "fake_ops": (lambda m: m.update(fake_ops=[]), "fake_ops"),
            "schemas": (lambda m: m["schemas"].update(identity="identity(Tensor x) -> ()"), "schema contract"),
            "schema_fingerprint": (lambda m: m.update(schema_fingerprint="0" * 64), "schema fingerprint"),
            "source_fingerprint": (lambda m: m.update(source_fingerprint="unknown"), "source_fingerprint"),
            "compiler": (lambda m: m["build"].update(toolchain={}), "compiler information"),
        }
        for label, (mutate, message) in cases.items():
            with self.subTest(label=label):
                metadata = copy.deepcopy(original)
                mutate(metadata)
                self.save_metadata(library, metadata)
                self.assert_pre_dlopen_rejected(cls, library, message)
        metadata = copy.deepcopy(original)
        metadata["build"]["extra_cflags"].append("-DCHANGED=1")
        self.save_metadata(library, metadata, rehash=False)
        self.assert_pre_dlopen_rejected(cls, library, "build fingerprint")

    def test_library_digest_rejects_wrong_or_modified_binary(self):
        cls = self.ops_class()
        library, metadata = self.artifact(cls)
        library.write_bytes(b"another binary")
        self.assert_pre_dlopen_rejected(cls, library, "hash/size mismatch")
        metadata["library"]["size"] = library.stat().st_size
        self.save_metadata(library, metadata)
        self.assert_pre_dlopen_rejected(cls, library, "hash/size mismatch")

    def test_full_arch_isa_and_ptx_targets_are_not_collapsed(self):
        cls = self.ops_class(cuda=True)
        library, metadata = self.artifact(cls, targets="12.0a+PTX")
        for requested in ("12.0", "12.0a", "12.0+PTX", "12.0f+PTX", "8.9"):
            with self.subTest(requested=requested), mock.patch.dict(
                os.environ, {"TORCH_CUDA_ARCH_LIST": requested},
            ):
                self.assert_pre_dlopen_rejected(cls, library, "full CUDA target mismatch")
        self.save_metadata(library, metadata)
        with mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "sm_120a+PTX"}):
            cls()._validate_metadata(library, Path(str(library) + ".json"))
        self.assertEqual(custom_ops._canonical_arch_list("sm_89;compute_120a;8.9"),
                         ["8.9", "12.0a+PTX"])
        self.assertEqual(custom_ops._parse_arch_list("8.9;12.0a+PTX"), [89, 120])
        with self.assertRaisesRegex(CustomOpsError, "Invalid CUDA target"):
            custom_ops._canonical_arch_list("bad_arch")

    def test_visible_sm_mismatch_is_pre_dlopen(self):
        cls = self.ops_class(cuda=True)
        library, _ = self.artifact(cls, targets="8.9;12.0a")
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "device_count", return_value=2), \
                mock.patch.object(torch.cuda, "get_device_capability", side_effect=[(8, 9), (8, 6)]):
            self.assert_pre_dlopen_rejected(cls, library, "SM mismatch")
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "device_count", return_value=2), \
                mock.patch.object(torch.cuda, "get_device_capability", side_effect=[(8, 9), (12, 0)]):
            cls()._validate_metadata(library, Path(str(library) + ".json"))

    def test_cuda_requirement_cannot_be_hidden_in_metadata(self):
        cls = self.ops_class(cuda=True)
        library, metadata = self.artifact(cls)
        metadata.update(with_cuda=False, cuda_arch_list=[])
        metadata["build"].update(with_cuda=False, cuda_arch_list=[])
        self.save_metadata(library, metadata)
        self.assert_pre_dlopen_rejected(cls, library, "CUDA requirement")

    def test_source_less_load_verifies_schema_and_fake_idempotently(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        instance = cls(allow_jit=False)
        with mock.patch.object(cls, "get_sources", side_effect=AssertionError("source read")), \
                mock.patch.object(cls, "get_include_dirs", side_effect=AssertionError("includes read")), \
                mock.patch.object(cls, "_compile_library", side_effect=AssertionError("compiled")), \
                mock.patch.object(cls, "_toolchain", side_effect=AssertionError("compiler queried")):
            self.assertIs(instance.load_precompiled(library), instance)
            self.assertIs(instance.load_precompiled(library), instance)
            self.assertIs(instance.load(), instance)
            other = cls(allow_jit=False).load_precompiled(library)
        self.assertTrue(instance.is_available())
        self.assertTrue(other.is_available())
        self.assertIsNone(instance.load_error())
        self.assertFalse(instance.allow_jit)
        self.assertEqual(instance.library_path, library.resolve())
        self.assertEqual(instance.metadata_path, Path(str(library) + ".json").resolve())
        self.dlopen.assert_called_once_with(str(library.resolve()))
        self.compiler_setup.assert_not_called()
        x = torch.ones(3)
        self.assertTrue(torch.equal(instance.identity(x), x))
        meta = instance.identity(torch.empty(4, device="meta"))
        self.assertEqual(meta.shape, (4,))
        self.assertEqual(meta.device.type, "meta")

    def test_explicit_sidecar_and_symlink_or_hardlink_identity(self):
        cls = self.ops_class()
        library, metadata = self.artifact(cls)
        sidecar = self.root / "deployment.json"
        self.save_metadata(library, metadata, sidecar=sidecar)
        self.dlopen.side_effect = lambda _: self.install(cls)
        cls().load_precompiled(library, sidecar)
        alias = self.root / "alias.so"
        alias.symlink_to(library)
        self.assertTrue(cls().load_precompiled(alias, sidecar).is_available())
        alias.unlink()
        os.link(library, alias)
        self.assertTrue(cls().load_precompiled(alias, sidecar).is_available())
        self.dlopen.assert_called_once()

    def test_copy_of_same_dispatcher_library_is_rejected_without_second_dlopen(self):
        cls = self.ops_class()
        library, metadata = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        cls().load_precompiled(library)
        copy_path = self.root / "copied.so"
        shutil.copyfile(library, copy_path)
        self.save_metadata(copy_path, metadata)
        with self.assertRaisesRegex(CustomOpsError, "Dispatcher library conflict"):
            cls().load_precompiled(copy_path)
        self.dlopen.assert_called_once()

    def test_untracked_existing_schema_is_fail_closed(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.install(cls, meta=True)
        self.assert_pre_dlopen_rejected(cls, library, "untracked library")

    def test_extra_registered_names_are_bookkept(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls, extra_ops=("hidden",))
        cls().load_precompiled(library)
        self.assertIn(f"{cls.namespace}::hidden", custom_ops._OP_OWNERS)
        conflict = self.ops_class()
        conflict.namespace = cls.namespace
        conflict.required_ops = ["hidden"]
        conflict.fake_ops = []
        conflict.op_schemas = {"hidden": "hidden(Tensor x) -> Tensor"}
        other, _ = self.artifact(conflict)
        with self.assertRaisesRegex(CustomOpsError, "Dispatcher library conflict"):
            conflict().load_precompiled(other)
        self.dlopen.assert_called_once()

    def test_missing_or_wrong_native_schema_poisons_loaded_record(self):
        for wrong in (False, True):
            with self.subTest(wrong=wrong):
                cls = self.ops_class()
                library, _ = self.artifact(cls)
                if wrong:
                    self.dlopen.side_effect = lambda _, c=cls: self.install(
                        c, schema="identity(Tensor x, int count=1) -> Tensor")
                else:
                    self.dlopen.side_effect = None
                instance = cls()
                message = "schema mismatch" if wrong else "Required ops not found"
                with self.assertRaisesRegex(CustomOpsError, message):
                    instance.load_precompiled(library)
                self.assertFalse(instance.is_available())
                calls = self.dlopen.call_count
                with self.assertRaisesRegex(CustomOpsError, "previously failed"):
                    cls().load_precompiled(library)
                self.assertEqual(self.dlopen.call_count, calls)

    def test_fake_failure_never_publishes_loaded_or_falls_back(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        instance = cls()
        with mock.patch.object(torch.library, "register_fake", side_effect=RuntimeError("fake exploded")), \
                mock.patch.object(torch.library, "impl_abstract") as fallback:
            with self.assertRaisesRegex(CustomOpsError, "fake exploded"):
                instance.load_precompiled(library)
        fallback.assert_not_called()
        self.assertFalse(instance.is_available())
        self.assertIn("fake exploded", instance.load_error())
        self.assertIsNone(instance.library_path)
        with self.assertRaisesRegex(CustomOpsError, "previously failed"):
            cls().load_precompiled(library)
        self.dlopen.assert_called_once()

    def test_fake_success_is_verified_and_existing_meta_is_legitimate(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls, meta=True)
        with mock.patch.object(torch.library, "register_fake", side_effect=AssertionError("duplicate")):
            self.assertTrue(cls().load_precompiled(library).is_available())
        other_cls = self.ops_class()
        other, _ = self.artifact(other_cls)
        self.dlopen.side_effect = lambda _: self.install(other_cls)
        with mock.patch.object(torch.library, "register_fake", return_value=lambda fn: fn):
            with self.assertRaisesRegex(CustomOpsError, "did not create a Meta kernel"):
                other_cls().load_precompiled(other)

    def test_fake_cannot_be_omitted_by_subclass(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        with mock.patch.object(cls, "register_fake_impls", return_value=None):
            with self.assertRaisesRegex(CustomOpsError, "Fake/Meta implementation missing"):
                cls().load_precompiled(library)

    def test_load_state_is_false_during_fake_registration(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        states = []
        original = cls.register_fake_impls

        def observe(instance):
            states.append(instance.is_available())
            original(instance)

        with mock.patch.object(cls, "register_fake_impls", observe):
            instance = cls().load_precompiled(library)
        self.assertEqual(states, [False])
        self.assertTrue(instance.is_available())

    def test_same_library_with_conflicting_metadata_is_not_dlopen_again(self):
        cls = self.ops_class()
        library, metadata = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        cls().load_precompiled(library)
        metadata["source_fingerprint"] = "f" * 64
        self.save_metadata(library, metadata)
        with self.assertRaisesRegex(CustomOpsError, "conflicting build metadata"):
            cls().load_precompiled(library)
        self.dlopen.assert_called_once()

    def test_dlopen_exception_is_not_retried_or_replaced(self):
        cls = self.ops_class()
        library, metadata = self.artifact(cls)
        self.dlopen.side_effect = OSError("missing libtorch dependency")
        with self.assertRaisesRegex(CustomOpsError, "missing libtorch dependency"):
            cls().load_precompiled(library)
        with self.assertRaisesRegex(CustomOpsError, "previously failed"):
            cls().load_precompiled(library)
        copy_path = self.root / "other.so"
        shutil.copyfile(library, copy_path)
        self.save_metadata(copy_path, metadata)
        with self.assertRaisesRegex(CustomOpsError, "Dispatcher library conflict"):
            cls().load_precompiled(copy_path)
        self.dlopen.assert_called_once()

    def test_two_threads_loading_same_library_only_dlopen_once(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        errors = []
        instances = []
        barrier = threading.Barrier(2)

        def load():
            try:
                barrier.wait(timeout=10)
                instances.append(cls().load_precompiled(library))
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=load) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(instances), 2)
        self.assertTrue(all(instance.is_available() for instance in instances))
        self.dlopen.assert_called_once()

    def test_development_failure_keeps_legacy_warning_but_not_loaded(self):
        cls = self.ops_class()
        instance = cls()
        with mock.patch.object(cls, "_load_extension", side_effect=RuntimeError("compiler missing")), \
                self.assertWarnsRegex(UserWarning, "compiler missing"):
            self.assertIs(instance.load(), instance)
        self.assertFalse(instance.is_available())
        self.assertIn("compiler missing", instance.load_error())

    def test_build_exports_to_user_directory_without_loading(self):
        cls = self.ops_class(cuda=True)
        instance = cls(allow_jit=False)
        output = self.root / "user" / "artifacts"
        tools = {"cxx": {"command": "c++", "version": "test compiler"},
                 "cuda_toolkit_version": torch.version.cuda,
                 "nvcc": {"command": "nvcc", "version": "test nvcc"}}

        def compile_(directory, inputs, *, verbose):
            self.assertEqual(inputs["cuda_arch_list"], ["8.9", "12.0a+PTX"])
            self.assertIn("-DFA_TARGETS=6", inputs["extra_cuda_cflags"])
            self.assertIn("--use_fast_math", inputs["extra_cuda_cflags"])
            self.assertIn("-ffast-math", inputs["extra_cflags"])
            self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "8.9;12.0a+PTX")
            product = directory / (cls.so_name + ".so")
            product.write_bytes(b"offline build output")
            return product

        with mock.patch.object(cls, "_toolchain", return_value=tools), \
                mock.patch.object(cls, "_compile_library", side_effect=compile_) as build_:
            library = instance.build(output, cuda_arch_list="12.0a+PTX;8.9")
        self.assertIsInstance(library, Path)
        self.assertEqual(library, output.resolve() / (cls.so_name + ".so"))
        self.assertTrue(Path(str(library) + ".json").is_file())
        self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "")
        self.assertFalse(instance.is_available())
        self.assertFalse(instance.allow_jit)
        self.assertFalse(custom_ops._LIBRARIES)
        self.dlopen.assert_not_called()
        build_.assert_called_once()
        self.assertFalse(any(path.name.startswith("." + cls.so_name + "-build-")
                             for path in output.iterdir()))
        metadata, _, _ = instance._validate_metadata(library, Path(str(library) + ".json"))
        self.assertEqual(metadata["cuda_arch_list"], ["8.9", "12.0a+PTX"])
        self.assertEqual(metadata["library"]["sha256"], hashlib.sha256(library.read_bytes()).hexdigest())
        self.assertEqual(len(metadata["source_fingerprint"]), 64)
        self.assertEqual(metadata["schemas"], instance._schema_contract())
        self.assertEqual(metadata["build"]["toolchain"], tools)

    def test_explicit_build_flags_are_additive_checksummed_and_do_not_leak(self):
        cls = self.ops_class(cuda=True)
        instance = cls()
        baseline = instance._build_inputs("8.9")
        cflags = ["-DHOST_FEATURE=1"]
        cuda_flags = ["-DDEVICE_FEATURE=1"]
        tools = {"cxx": {"version": "test compiler"},
                 "cuda_toolkit_version": torch.version.cuda}
        captured = []

        def compile_(directory, inputs, *, verbose):
            captured.append(copy.deepcopy(inputs))
            library = directory / (cls.so_name + ".so")
            library.write_bytes(b"mock flagged output")
            return library

        with mock.patch.object(cls, "_toolchain", return_value=tools), \
                mock.patch.object(cls, "_compile_library", side_effect=compile_):
            library = instance.build(self.root / "flags", cuda_arch_list="8.9",
                                     extra_cflags=cflags, extra_cuda_cflags=cuda_flags)
        self.assertEqual(captured[0]["extra_cflags"], baseline["extra_cflags"] + cflags)
        self.assertEqual(captured[0]["extra_cuda_cflags"],
                         baseline["extra_cuda_cflags"] + cuda_flags)
        self.assertEqual(instance._build_inputs("8.9"), baseline)
        self.assertEqual(cflags, ["-DHOST_FEATURE=1"])
        self.assertEqual(cuda_flags, ["-DDEVICE_FEATURE=1"])
        metadata, _, _ = instance._validate_metadata(library, Path(str(library) + ".json"))
        self.assertFalse(metadata["testing"])
        original_hash = metadata["build_fingerprint"]
        metadata["build"]["extra_cuda_cflags"].append("-DOTHER=1")
        self.assertNotEqual(custom_ops._metadata_fingerprint(metadata), original_hash)
        self.save_metadata(library, metadata, rehash=False)
        self.assert_pre_dlopen_rejected(cls, library, "build fingerprint")
        self.dlopen.assert_not_called()

    def test_invalid_extra_flags_fail_before_compilation(self):
        cls = self.ops_class()
        for flags in ("-DFLAG=1", [1], [""], ["   "], ["-DBAD=\0"]):
            for name in ("extra_cflags", "extra_cuda_cflags"):
                with self.subTest(name=name, flags=flags), \
                        mock.patch.object(cls, "_compile_library") as compile_:
                    with self.assertRaisesRegex(TypeError, name):
                        cls().build(self.root / "bad-flags", **{name: flags})
                    compile_.assert_not_called()
        with self.assertRaisesRegex(CustomOpsError, r"C\+\+-only"):
            cls().build(self.root / "no-cuda", extra_cuda_cflags=["-DFLAG=1"])

    def test_fuse_moe_testing_build_enables_cpu_macro_without_mutating_defaults(self):
        with mock.patch.object(CustomOps, "build", return_value=self.root / "artifact.so") as build_:
            instance = FuseMoeOps(allow_jit=False)
            for testing in (False, True):
                flags = ["-DEXTRA=1"]
                result = instance.build(self.root / str(testing), cuda_arch_list="8.9",
                                        testing=testing, extra_cflags=flags,
                                        extra_cuda_cflags=["-DCUDA_EXTRA=1"])
                self.assertEqual(result, self.root / "artifact.so")
                kwargs = build_.call_args.kwargs
                self.assertEqual(kwargs["extra_cflags"], flags + (
                    ["-DCUSTOM_OPS_OFFLOAD_TESTING=1"] if testing else []))
                self.assertEqual(kwargs["extra_cuda_cflags"], ["-DCUDA_EXTRA=1"])
                self.assertEqual(flags, ["-DEXTRA=1"])
                self.assertFalse(any("CUSTOM_OPS_OFFLOAD_TESTING" in flag
                                     for flag in instance.get_extra_cflags()))
            with self.assertRaisesRegex(TypeError, "testing must be a bool"):
                instance.build(self.root / "invalid", testing=1)
        self.dlopen.assert_not_called()

    def test_fuse_moe_production_and_testing_exports_have_independent_metadata(self):
        fixture = self.ops_class(cuda=True)()
        instance = FuseMoeOps(allow_jit=False)
        tools = {"cxx": {"version": "test compiler"},
                 "cuda_toolkit_version": torch.version.cuda}

        def compile_(directory, inputs, *, verbose):
            product = directory / (instance.so_name + ".so")
            product.write_bytes(repr(inputs["extra_cflags"]).encode())
            return product

        with mock.patch.object(FuseMoeOps, "get_sources", side_effect=fixture.get_sources), \
                mock.patch.object(FuseMoeOps, "get_include_dirs", side_effect=fixture.get_include_dirs), \
                mock.patch.object(FuseMoeOps, "_toolchain", return_value=tools), \
                mock.patch.object(FuseMoeOps, "_compile_library", side_effect=compile_):
            production = instance.build(self.root / "production", cuda_arch_list="8.9")
            testing = instance.build(self.root / "testing", cuda_arch_list="8.9", testing=True)
        prod_meta, _, _ = instance._validate_metadata(production, Path(str(production) + ".json"))
        with self.assertRaisesRegex(CustomOpsError, "allow_testing=True"):
            instance._validate_metadata(testing, Path(str(testing) + ".json"))
        test_meta, _, _ = instance._validate_metadata(
            testing, Path(str(testing) + ".json"), allow_testing=True,
        )
        self.assertFalse(prod_meta["testing"])
        self.assertTrue(test_meta["testing"])
        self.assertEqual(prod_meta["format_version"], 2)
        self.assertEqual(prod_meta["source_fingerprint"], test_meta["source_fingerprint"])
        self.assertEqual(prod_meta["build_version"], test_meta["build_version"])
        self.assertNotEqual(prod_meta["build_fingerprint"], test_meta["build_fingerprint"])
        self.assertNotEqual(prod_meta["library"]["sha256"], test_meta["library"]["sha256"])
        self.assertNotIn("_fuse_moe_offload_test_inject", test_meta["required_ops"])
        self.assertFalse(instance.is_available())
        self.dlopen.assert_not_called()

    def test_test_macro_from_subclass_or_extra_flags_is_never_production(self):
        cls = self.ops_class(cuda=True)
        instance = cls()
        tools = {"cxx": {"version": "test"}}
        for kwargs in ({"extra_cflags": ["-DCUSTOM_OPS_OFFLOAD_TESTING=1"]},
                       {"extra_cflags": ["-D", "CUSTOM_OPS_OFFLOAD_TESTING"]},
                       {"extra_cuda_cflags": ["-DCUSTOM_OPS_OFFLOAD_TESTING=0"]}):
            inputs = instance._build_inputs("8.9", **kwargs)
            self.assertTrue(instance._build_metadata(inputs, "0" * 64, tools)["testing"])
        default_flags = cls().get_extra_cflags()
        with mock.patch.object(cls, "get_extra_cflags", return_value=[
                *default_flags, "-DCUSTOM_OPS_OFFLOAD_TESTING=1"]):
            library, metadata = self.artifact(cls)
        self.assertTrue(metadata["testing"])
        self.assertEqual(metadata["build_version"], str(cls.build_version))
        self.assert_pre_dlopen_rejected(cls, library, "allow_testing=True")

    def test_testing_load_requires_per_call_opt_in_even_for_same_library(self):
        cls = self.ops_class()
        default_flags = cls().get_extra_cflags()
        with mock.patch.object(cls, "get_extra_cflags", return_value=[
                *default_flags, "-DCUSTOM_OPS_OFFLOAD_TESTING=1"]):
            library, _ = self.artifact(cls)
        self.assert_pre_dlopen_rejected(cls, library, "allow_testing=True")
        self.dlopen.side_effect = lambda _: self.install(cls)
        instance = cls()
        with mock.patch.object(cls, "get_sources", side_effect=AssertionError("sources")), \
                mock.patch.object(cls, "get_extra_cflags", side_effect=AssertionError("build flags")), \
                mock.patch.object(cls, "_compile_library", side_effect=AssertionError("compiler")):
            self.assertIs(instance.load_precompiled(library, allow_testing=True), instance)
            self.assertTrue(instance.is_available())
            self.assertFalse(instance.allow_jit)
            with self.assertRaisesRegex(CustomOpsError, "allow_testing=True"):
                cls().load_precompiled(library)
            self.assertTrue(cls().load_precompiled(library, allow_testing=True).is_available())
        self.dlopen.assert_called_once()
        with self.assertRaisesRegex(CustomOpsError, "allow_testing must be a bool"):
            cls().load_precompiled(library, allow_testing=1)

    def test_unmarked_native_testing_hooks_fail_closed_even_with_opt_in(self):
        for hook in FuseMoeOps.testing_ops:
            for allow_testing in (False, True):
                with self.subTest(hook=hook, allow_testing=allow_testing):
                    cls = self.ops_class()
                    cls.testing_ops = FuseMoeOps.testing_ops
                    library, metadata = self.artifact(cls)
                    self.assertFalse(metadata["testing"])
                    self.dlopen.reset_mock()
                    self.dlopen.side_effect = lambda _, cls=cls, hook=hook: self.install(
                        cls, extra_ops=(hook,))
                    instance = cls()
                    with mock.patch.object(cls, "register_fake_impls") as fake, \
                            self.assertRaisesRegex(CustomOpsError, "Unmarked production library"):
                        instance.load_precompiled(library, allow_testing=allow_testing)
                    fake.assert_not_called()
                    self.assertFalse(instance.is_available())
                    self.assertFalse(instance.allow_jit)
                    self.assertIsNone(instance.library_path)
                    identity, _ = custom_ops._library_identity(library)
                    record = custom_ops._LIBRARIES[identity]
                    self.assertFalse(record["ready"])
                    self.assertIn("test-hook schemas", record["error"])
                    self.assertEqual(custom_ops._OP_OWNERS[f"{cls.namespace}::{hook}"], identity)
                    with self.assertRaisesRegex(CustomOpsError, "previously failed"):
                        cls().load_precompiled(library, allow_testing=True)
                    self.dlopen.assert_called_once()

    def test_marked_native_testing_hooks_load_with_explicit_opt_in(self):
        cls = self.ops_class()
        cls.testing_ops = FuseMoeOps.testing_ops
        with mock.patch.object(cls, "get_extra_cflags", return_value=[
                "-DCUSTOM_OPS_OFFLOAD_TESTING=1"]):
            library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls, extra_ops=cls.testing_ops)
        self.assertTrue(cls().load_precompiled(library, allow_testing=True).is_available())
        with self.assertRaisesRegex(CustomOpsError, "allow_testing=True"):
            cls().load_precompiled(library)
        self.dlopen.assert_called_once()

    def test_testing_marker_cannot_be_removed_or_disagree_with_flags(self):
        cls = self.ops_class()
        library, original = self.artifact(cls)
        cases = [dict(testing=True), dict(testing=1), dict(testing=None)]
        for updates in cases:
            metadata = copy.deepcopy(original)
            metadata.update(updates)
            self.save_metadata(library, metadata)
            self.assert_pre_dlopen_rejected(cls, library, "testing marker")
        metadata = copy.deepcopy(original)
        del metadata["testing"]
        self.save_metadata(library, metadata)
        self.assert_pre_dlopen_rejected(cls, library, "testing marker")
        metadata = copy.deepcopy(original)
        metadata["build"]["extra_cflags"].append("-DCUSTOM_OPS_OFFLOAD_TESTING=1")
        self.save_metadata(library, metadata)
        self.assert_pre_dlopen_rejected(cls, library, "testing marker")

    def test_facade_testing_opt_in_preserves_strict_policy(self):
        cls = self.ops_class()
        with mock.patch.object(cls, "get_extra_cflags", return_value=[
                "-DCUSTOM_OPS_OFFLOAD_TESTING=1"]):
            library, _ = self.artifact(cls)
        facade = RecsysOps()
        facade._GROUP_CLASSES = dict(facade._GROUP_CLASSES, fuse_moe=cls)
        with self.assertRaisesRegex(CustomOpsError, "allow_testing=True"):
            facade.load_precompiled("fuse_moe", library)
        self.dlopen.assert_not_called()
        self.dlopen.side_effect = lambda _: self.install(cls)
        with mock.patch.object(CustomOps, "load", side_effect=AssertionError("JIT")):
            self.assertIs(facade.load_precompiled("fuse_moe", library, allow_testing=True), facade)
            self.assertIs(facade.ensure_loaded("fuse_moe"), facade)
            self.assertFalse(facade.allow_jit)
            with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                facade.ensure_loaded("fa")

    def test_offload_fake_registration_preserves_shared_python_implementation(self):
        from custom_ops.moe_offload import _fake_offload
        with mock.patch.object(FuseMoeOps, "register_fake_for") as register:
            FuseMoeOps().register_fake_impls()
        self.assertEqual(register.call_args_list[1], mock.call("fuse_moe_offload", _fake_offload))

    def test_build_fails_without_targets_and_does_not_overwrite_existing_artifact(self):
        cls = self.ops_class(cuda=True)
        with mock.patch.object(cls, "_compile_library") as build_:
            with self.assertRaisesRegex(CustomOpsError, "No CUDA build targets"):
                cls().build(self.root / "bad")
        build_.assert_not_called()
        cpp_cls = self.ops_class()
        output = self.root / "existing"
        library, metadata = self.artifact(cpp_cls, directory=output)
        before = library.read_bytes()
        sidecar_before = Path(str(library) + ".json").read_bytes()
        with mock.patch.object(cpp_cls, "_toolchain", return_value={"cxx": {"version": "test"}}), \
                mock.patch.object(cpp_cls, "_compile_library", side_effect=RuntimeError("build failed")):
            with self.assertRaisesRegex(CustomOpsError, "build failed"):
                cpp_cls().build(output)
        self.assertEqual(library.read_bytes(), before)
        self.assertEqual(Path(str(library) + ".json").read_bytes(), sidecar_before)
        self.assertFalse(any(path.name.startswith("." + cpp_cls.so_name + "-build-")
                             for path in output.iterdir()))

    def test_metadata_publication_failure_reports_incomplete_export_and_rejects_load(self):
        cls = self.ops_class()
        output = self.root / "incomplete-export"
        library, _ = self.artifact(cls, directory=output)
        sidecar = Path(str(library) + ".json")
        old_metadata = sidecar.read_bytes()
        replace = os.replace

        def compile_(directory, inputs, *, verbose):
            product = directory / library.name
            product.write_bytes(b"new compiled library")
            return product

        def publish(source, destination):
            if destination == sidecar:
                raise OSError("sidecar publication denied")
            return replace(source, destination)

        with mock.patch.object(cls, "_toolchain", return_value={"cxx": {"version": "test"}}), \
                mock.patch.object(cls, "_compile_library", side_effect=compile_), \
                mock.patch.object(custom_ops.os, "replace", side_effect=publish):
            with self.assertRaisesRegex(CustomOpsError, "export incomplete.*pair is not atomic"):
                cls().build(output)
        self.assertEqual(library.read_bytes(), b"new compiled library")
        self.assertEqual(sidecar.read_bytes(), old_metadata)
        self.assert_pre_dlopen_rejected(cls, library, "hash/size mismatch")
        self.assertFalse(any(path.name.startswith("." + cls.so_name + "-build-")
                             for path in output.iterdir()))

    def test_offline_build_requires_explicit_schema_declarations(self):
        cls = self.ops_class()
        cls.op_schemas = {}
        with mock.patch.object(cls, "_compile_library") as build_:
            with self.assertRaisesRegex(CustomOpsError, "explicit op_schemas"):
                cls().build(self.root / "invalid")
        build_.assert_not_called()
        self.compiler_setup.assert_not_called()

    def test_header_content_fingerprint_detects_same_size_and_mtime_edit(self):
        cls = self.ops_class()
        instance = cls()
        inputs = instance._build_inputs()
        before = instance._source_fingerprint(inputs)
        header = Path(inputs["include_dirs"][0]) / "header.h"
        stat = header.stat()
        data = header.read_bytes()
        header.write_bytes(data.replace(b"test", b"best"))
        os.utime(header, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(header.stat().st_size, stat.st_size)
        self.assertEqual(header.stat().st_mtime_ns, stat.st_mtime_ns)
        self.assertNotEqual(instance._source_fingerprint(inputs), before)

    def test_development_jit_and_cache_reuse_shared_build_inputs(self):
        cls = self.ops_class()
        instance = cls()
        cache = self.root / "development-cache"
        seen = []

        def compile_(directory, inputs, *, verbose):
            seen.append(copy.deepcopy(inputs))
            product = directory / (cls.so_name + ".so")
            product.write_bytes(b"cached output")
            return product

        self.dlopen.side_effect = lambda _: self.install(cls)
        with mock.patch("custom_ops.get_build_dir", return_value=str(cache)), \
                mock.patch.object(cls, "_compile_library", side_effect=compile_) as compile_mock:
            with self.assertWarns(UserWarning):
                instance.load()
            self.assertTrue(instance.is_available())
            self.assertTrue(cls().load().is_available())
            self.assertEqual(compile_mock.call_count, 1)
        self.assertEqual(seen, [instance._build_inputs()])
        self.dlopen.assert_called_once()
        self.assertTrue((cache / cls.so_name / ".build_stamp").is_file())

    def test_standard_build_extension_does_not_mutate_shared_flags(self):
        # Exercise the real public BuildExtension construction/finalization but
        # stub run(), so this remains a unit test with no native compilation.
        from torch.utils.cpp_extension import BuildExtension
        cls = self.ops_class()
        instance = cls()
        inputs = instance._build_inputs()
        before = copy.deepcopy(inputs)
        output = self.root / "standard-build"
        output.mkdir()

        def run(command):
            command.extensions = command.distribution.ext_modules
            command.extensions[0].extra_compile_args["cxx"].append("-DMUTATED=1")
            Path(command.get_ext_fullpath(cls.so_name)).write_bytes(b"standard output")

        with mock.patch.object(BuildExtension, "run", run):
            result = instance._compile_library(output, inputs, verbose=False)
        self.assertEqual(result, output / (cls.so_name + ".so"))
        self.assertEqual(inputs, before)
        self.dlopen.assert_not_called()

    def test_facade_strict_queries_and_ensure_never_compile(self):
        facade = RecsysOps(allow_jit=False)
        with mock.patch.object(CustomOps, "load", side_effect=AssertionError("JIT")) as load_:
            self.assertFalse(facade.is_available("fuse_moe"))
            self.assertIsNone(facade.load_error("fuse_moe"))
            self.assertEqual(set(facade.groups.values()), {"not_loaded"})
            for action in (lambda: facade.ensure_loaded("fuse_moe"),
                           facade.ensure_loaded, facade.load,
                           lambda: facade.fuse_moe(None, None, None, None, None),
                           lambda: facade.fuse_moe_offload_status):
                with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                    action()
        load_.assert_not_called()
        self.assertEqual(facade._groups, {})

    def test_facade_precompiled_success_and_failure_disable_all_groups(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        facade = RecsysOps()
        facade._GROUP_CLASSES = dict(facade._GROUP_CLASSES, fuse_moe=cls)
        existing = FAOps()
        facade._groups["fa"] = existing
        with mock.patch.object(CustomOps, "load", side_effect=AssertionError("JIT")) as load_:
            self.assertIs(facade.load_precompiled("fuse_moe", library), facade)
            self.assertIs(facade.ensure_loaded("fuse_moe"), facade)
            self.assertTrue(facade.is_available("fuse_moe"))
            self.assertFalse(existing.allow_jit)
            self.assertFalse(facade.allow_jit)
            with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                facade.ensure_loaded("mixed_gemm")
            self.assertFalse(facade.is_available("mixed_gemm"))
        load_.assert_not_called()
        failed = RecsysOps()
        with self.assertRaisesRegex(CustomOpsError, "No such file"):
            failed.load_precompiled("fuse_moe", self.root / "missing.so")
        self.assertFalse(failed.allow_jit)
        self.assertFalse(failed.is_available("fuse_moe"))
        self.assertEqual(failed.groups["fuse_moe"], "failed")
        self.assertIn("No such file", failed.load_error("fuse_moe"))

    def test_facade_strict_backend_cannot_switch_or_fallback(self):
        facade = RecsysOps(allow_jit=False)
        instance = FuseMoeOps(allow_jit=False)
        instance._loaded = True
        facade._groups["fuse_moe"] = instance
        with mock.patch.object(facade, "_tilelang", side_effect=AssertionError("Tile-lang")) as tilelang, \
                mock.patch.dict(os.environ, {"CUSTOM_OPS_BACKEND": "tilelang"}):
            self.assertEqual(facade._dispatch(None, "fuse_moe"), (False, None))
            self.assertEqual(facade._dispatch("auto", "fuse_moe"), (False, None))
            self.assertEqual(facade._dispatch("cuda", "fuse_moe"), (False, None))
            with self.assertRaisesRegex(CustomOpsError, "fallback is disabled"):
                facade._dispatch("tilelang", "fuse_moe")
            instance._loaded = False
            with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                facade._dispatch("auto", "fuse_moe")
        tilelang.assert_not_called()

    def test_global_singleton_satisfies_runtime_ensure_without_jit(self):
        cls = self.ops_class()
        library, _ = self.artifact(cls)
        self.dlopen.side_effect = lambda _: self.install(cls)
        facade = custom_ops.ops
        with mock.patch.object(facade, "_groups", {}), \
                mock.patch.object(facade, "_allow_jit", True), \
                mock.patch.object(facade, "_GROUP_CLASSES", dict(facade._GROUP_CLASSES, fuse_moe=cls)), \
                mock.patch.object(CustomOps, "load", side_effect=AssertionError("JIT")) as load_:
            self.assertIs(facade.load_precompiled("fuse_moe", library), facade)
            # These are the exact two global facade calls used by PrefillMoERuntime.
            self.assertIs(custom_ops.ops.ensure_loaded("fuse_moe"), facade)
            self.assertTrue(custom_ops.ops.is_available("fuse_moe"))
            self.assertFalse(custom_ops.ops.allow_jit)
            with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                custom_ops.ops.ensure_loaded("swiglu")
        load_.assert_not_called()

    def test_prefill_runtime_missing_precompiled_group_fails_without_jit(self):
        from custom_ops.moe_offload import ExpertStore, PrefillMoERuntime
        facade = RecsysOps(allow_jit=False)
        with mock.patch.object(custom_ops, "ops", facade), \
                mock.patch.object(CustomOps, "load", side_effect=AssertionError("JIT")) as load_, \
                mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError("GPU init")):
            with self.assertRaisesRegex(CustomOpsError, "JIT is disabled"):
                PrefillMoERuntime(ExpertStore({}), device="cuda:0")
        load_.assert_not_called()
        self.compiler_setup.assert_not_called()

    def test_facade_development_default_remains_group_lazy(self):
        facade = RecsysOps()
        self.assertEqual(facade._groups, {})

        def load_(instance):
            instance._loaded = True
            return instance

        with mock.patch.object(CustomOps, "load", autospec=True, side_effect=load_) as load_mock:
            self.assertIs(facade.ensure_loaded("fuse_moe"), facade)
            self.assertIs(facade.ensure_loaded("fuse_moe"), facade)
            self.assertEqual(set(facade._groups), {"fuse_moe"})
            self.assertTrue(facade._groups["fuse_moe"].allow_jit)
            load_mock.assert_called_once()

    def test_group_names_validate_before_any_load(self):
        facade = RecsysOps()
        with mock.patch.object(CustomOps, "load", side_effect=AssertionError("JIT")) as load_:
            for action in (lambda: facade.ensure_loaded("fuse_moe", "bad"),
                           lambda: facade.load_precompiled("bad", "missing.so"),
                           lambda: facade.is_available("bad"),
                           lambda: facade.load_error("bad")):
                with self.assertRaisesRegex(ValueError, "unknown group"):
                    action()
        load_.assert_not_called()
        self.assertTrue(facade.allow_jit)
        self.assertEqual(facade._groups, {})

    def test_fuse_moe_schema_declares_geglu_trailing_codes(self):
        create_schema = FuseMoeOps.op_schemas["fuse_moe_offload_create"]
        parsed = torch._C.parse_schema("recsys_ops::" + create_schema)
        self.assertEqual(len(parsed.arguments), 20)
        self.assertEqual([argument.name for argument in parsed.arguments[-3:]],
                         ["expert_activation", "numerical_mode", "expert_type"])
        self.assertEqual([argument.default_value for argument in parsed.arguments[-3:]],
                         [0, 0, 0])
        self.assertTrue(all(str(argument.type) == "int"
                            for argument in parsed.arguments[-3:]))
        moe = torch._C.parse_schema("recsys_ops::" + FuseMoeOps.op_schemas["fuse_moe"])
        self.assertEqual(len(moe.arguments), 8)
        self.assertEqual([argument.name for argument in moe.arguments[-3:]],
                         ["expert_activation", "numerical_mode", "expert_type"])
        self.assertEqual([argument.default_value for argument in moe.arguments[-3:]],
                         [0, 0, 0])

    def test_cached_legacy_fuse_moe_schema_is_rejected_before_dlopen(self):
        cls = self.ops_class(fake=False)
        name = "fuse_moe_offload_create"
        cls.required_ops = [name]
        cls.op_schemas = {name: FuseMoeOps.op_schemas[name]}
        library, metadata = self.artifact(cls)
        instance = cls()
        instance._validate_metadata(library, Path(str(library) + ".json"))
        legacy = cls.op_schemas[name].replace(
            ", int expert_activation=0, int numerical_mode=0, int expert_type=0", "")
        self.assertNotEqual(legacy, cls.op_schemas[name])
        self.assertEqual(len(torch._C.parse_schema("recsys_ops::" + legacy).arguments), 17)
        # Simulate a cached artifact built before the two trailing mode args.
        legacy_metadata = copy.deepcopy(metadata)
        legacy_metadata["schemas"][name] = legacy
        self.save_metadata(library, legacy_metadata)
        self.assert_pre_dlopen_rejected(cls, library, "schema contract")
        # The schema fingerprint alone is sufficient to reject a stale cache.
        stale = copy.deepcopy(metadata)
        stale["schema_fingerprint"] = "0" * 64
        self.save_metadata(library, stale)
        self.assert_pre_dlopen_rejected(cls, library, "schema fingerprint")

    def test_required_offload_ops_and_schema_contracts_are_complete(self):
        new_ops = {
            "fuse_moe_offload_resident_bind", "fuse_moe_offload_validate",
            "fuse_moe_offload_begin", "fuse_moe_offload_check", "fuse_moe_offload_status",
            "fuse_moe_offload_acknowledge",
        }
        self.assertTrue(new_ops.issubset(FuseMoeOps.required_ops))
        for name in new_ops:
            self.assertEqual(RecsysOps._OP_TO_GROUP[name], "fuse_moe")
        self.assertNotIn("_fuse_moe_offload_test_inject", FuseMoeOps.required_ops)
        for cls in (FAOps, MixedGemmOps, FuseMoeOps, SwigluOps):
            self.assertEqual(set(cls.required_ops), set(cls()._schema_contract(require_complete=True)))
            self.assertTrue(cls.requires_cuda)
        schemas = FuseMoeOps()._schema_contract()
        self.assertIn("Tensor(a!) runtime_token", schemas["fuse_moe_offload"])
        for suffix, result in (("begin", "()"), ("check", "()"), ("status", "Tensor"),
                               ("acknowledge", "()")):
            name = "fuse_moe_offload_" + suffix
            self.assertEqual(schemas[name], f"recsys_ops::{name}(Tensor runtime_token) -> {result}")


if __name__ == "__main__":
    unittest.main()
