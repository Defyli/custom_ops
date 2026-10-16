"""CPU-only contracts; no native build, numerical stand-ins, or GPU execution.

Run with unittest discovery or pytest. Target-namespace registration checks use
fresh Python processes so they cannot collide with a loaded native extension.
Compiler calls are mocked only to inspect orchestration, not numerical outputs.
"""

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch._inductor import config, ir, lowering
from torch._inductor.codegen import cpp_wrapper_cpu as cpp
from torch._inductor.virtualized import V

# Direct execution and unittest discovery do not necessarily include repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import moe_offload_aoti as feature


SCHEMA = (
    "fuse_moe_offload(Tensor x, Tensor w1, Tensor w2, Tensor ids, Tensor scale, "
    "Tensor(a!) runtime_token, int layer_id, int[]? deployment_config=None, "
    "int[]? deployment_manifest=None) -> Tensor"
)


def meta_only(x, *args, **kwargs):
    # Only output metadata. There is deliberately no CPU or CUDA implementation.
    return torch.empty_like(x)


class LoweringContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        feature._version()  # An unsupported toolchain is a failure, not a skip.
        cls.library = torch.library.Library("offload_aoti_lowering_test", "FRAGMENT")
        cls.library.define(SCHEMA)
        cls.library.impl("fuse_moe_offload", meta_only, "Meta")
        cls.op = torch.ops.offload_aoti_lowering_test.fuse_moe_offload.default

    def child(self, body, *, with_schema=True, with_meta=True):
        setup = ""
        if with_schema:
            setup = "lib = torch.library.Library('recsys_ops', 'FRAGMENT')\n"
            setup += f"lib.define({SCHEMA!r})\n"
            if with_meta:
                setup += "lib.impl('fuse_moe_offload', lambda x, *a: torch.empty_like(x), 'Meta')\n"
            setup += "op = torch.ops.recsys_ops.fuse_moe_offload.default\n"
        code = (
            "import torch\nfrom torch._inductor import config, lowering\n"
            "from unittest import mock\nimport moe_offload_aoti as feature\n"
            + setup + textwrap.dedent(body)
        )
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=ROOT, env=environment,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_reviewed_versions_and_stable_refusal(self):
        for value, expected in (("2.6.0a0", "2.6"), ("2.11.0+cu128", "2.11")):
            with self.subTest(value=value), mock.patch.object(torch, "__version__", value):
                self.assertEqual(feature._version(), expected)
        for value in ("2.6.0", "2.6.1", "2.11.1", "2.11.0a0", "2.12.0"):
            with self.subTest(value=value), mock.patch.object(torch, "__version__", value):
                with self.assertRaisesRegex(RuntimeError, "unsupported PyTorch"):
                    feature._version()

    def test_actual_interfaces_and_inherited_creation(self):
        feature._validate_interfaces(feature._version(), ir, lowering, cpp)
        node = feature._node_class(feature._version(), ir, lowering, cpp, config, V, self.op)
        self.assertIs(node.create.__func__, ir.FallbackKernel.create.__func__)
        self.assertIs(node.process_kernel.__func__, ir.FallbackKernel.process_kernel.__func__)
        self.assertIs(node.__init__, ir.FallbackKernel.__init__)

    def test_foreign_signature_is_rejected(self):
        def different(self, op_overload, raw_args, output_args, raw_outputs, extra):
            raise AssertionError("must not execute")
        foreign = type("ForeignWrapper", (cpp.CppWrapperCpu,), {
            "generate_extern_kernel_args_decl_if_needed": different,
        })
        with self.assertRaisesRegex(RuntimeError, "unreviewed parameters"):
            feature._validate_interfaces(
                feature._version(), ir, lowering, SimpleNamespace(CppWrapperCpu=foreign)
            )

    def test_signature_keyword_only_contract(self):
        def not_keyword_only(c_type, elements, force_mutable=False):
            pass
        with self.assertRaisesRegex(RuntimeError, "keyword-only"):
            feature._signature(
                not_keyword_only, ("c_type", "elements", "force_mutable"),
                keyword_only=("force_mutable",),
            )

    def buffers(self):
        return [
            ir.Buffer(name=f"arg{i}", layout=ir.FixedLayout(
                torch.device("cpu"), torch.int64 if i == 5 else torch.float32,
                [2], [1],
            )) for i in range(6)
        ]

    def test_actual_emitter_packs_handles_and_optional_integer_lists(self):
        feature._validate_schema(self.op)
        emit = feature._emit_checked_26 if feature._version() == "2.6" else feature._emit_checked_211
        for deployment, manifest, packed in (
            (None, None, "3"), ([], [], "3"), ([10, 11], [20, 21], "3, 10, 11, 20, 21"),
        ):
            with self.subTest(deployment=deployment, manifest=manifest):
                wrapper = cpp.CppWrapperCpu.__new__(cpp.CppWrapperCpu)
                wrapper.lines = []
                output = ir.Buffer(name="result", layout=ir.FixedLayout(
                    torch.device("cpu"), torch.float32, [2], [1],
                ))
                emit(wrapper, self.op, self.buffers() + [3, deployment, manifest],
                     ["result"], [output], 4)
                line = wrapper.lines[-1]
                self.assertTrue(line.startswith("AOTI_TORCH_ERROR_CODE_CHECK("))
                self.assertTrue(line.endswith("));"))
                self.assertIn("proxy_executor, 4,", line)
                self.assertIn("{" + packed + "}.data()", line)
                self.assertIn("{arg0, arg1, arg2, arg3, arg4, arg5, result}.data()", line)
                self.assertIn("aoti_torch_new_uninitialized_tensor", "\n".join(wrapper.lines[:-1]))
                self.assertNotIn(".cbegin()", line)

    def test_node_codegen_preserves_extern_json_and_token_mutation(self):
        from torch._inductor.graph import GraphLowering
        from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
        from torch._subclasses.fake_tensor import FakeTensorMode
        graph = torch.fx.Graph()
        graph.output(())
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        fake_mode = FakeTensorMode()
        node_type = feature._node_class(
            feature._version(), ir, lowering, cpp, config, V, self.op
        )
        with fake_mode:
            context = GraphLowering(gm, example_inputs=[], cpp_wrapper=True, aot_mode=True)
        extern_context = V.set_extern_kernel_nodes([]) if feature._version() == "2.11" else nullcontext()
        with V.set_graph_handler(context), extern_context:
            inputs = self.buffers()
            raw_args = inputs + [3, [10, 11], [20, 21]]
            node = node_type(
                ir.MultiOutputLayout(device=torch.device("cpu")), self.op,
                inputs, raw_args[6:], lambda tensors, constants: (
                    tuple(tensors) + tuple(constants), {}
                ),
            )
            output = ir.MultiOutput(
                ir.FixedLayout(torch.device("cpu"), torch.float32, [2], [1]), node, [],
            )
            node.outputs = [output]
            self.assertEqual(len(node.mutation_outputs), 1)
            self.assertEqual(node.mutation_outputs[0].get_mutation_names(), ["arg5"])
            wrapper = CppWrapperGpu.__new__(CppWrapperGpu)
            wrapper.lines = []
            wrapper.comment = "//"
            context.wrapper_code = wrapper
            node.codegen(wrapper)
            nodes = context.extern_kernel_nodes if feature._version() == "2.6" else V.extern_kernel_nodes
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].name, node.get_name())
            self.assertEqual(nodes[0].node.target, self.op.name())
            self.assertEqual(nodes[0].node.outputs[0].as_tensor.name, output.get_name())
            calls = [line for line in wrapper.lines if isinstance(line, str)
                     and "aoti_torch_proxy_executor_call_function(" in line]
            self.assertEqual(len(calls), 1)
            line = calls[0]
            self.assertTrue(line.startswith("AOTI_TORCH_ERROR_CODE_CHECK("))
            self.assertIn("proxy_executor, 0,", line)
            self.assertIn("arg5, " + output.get_name(), line)

            class ForeignWrapper(cpp.CppWrapperCpu):
                pass

            foreign = ForeignWrapper.__new__(ForeignWrapper)
            foreign.lines = []
            with self.assertRaisesRegex(RuntimeError, "unsupported wrapper type"):
                node.codegen(foreign)
            self.assertEqual(foreign.lines, [])
            self.assertEqual(len(nodes), 1)

    def test_missing_operator_and_meta_are_not_silent(self):
        self.child("""
            try:
                feature.register_aoti_lowering()
            except RuntimeError as error:
                assert 'load the native operator' in str(error)
            else:
                raise AssertionError('missing op accepted')
        """, with_schema=False)
        self.child("""
            try:
                feature.register_aoti_lowering()
            except RuntimeError as error:
                assert 'fake/Meta' in str(error)
            else:
                raise AssertionError('missing Meta accepted')
            assert op not in lowering.lowerings
        """, with_meta=False)

    def test_registration_idempotence_and_foreign_ownership(self):
        self.child("""
            def foreign(*args, **kwargs):
                raise AssertionError('must not execute')
            lowering.register_lowering(op, type_promotion_kind=None)(foreign)
            owner = lowering.lowerings[op]
            try:
                feature.register_aoti_lowering()
            except RuntimeError as error:
                assert 'another lowering' in str(error)
            else:
                raise AssertionError('foreign ownership overwritten')
            assert lowering.lowerings[op] is owner
            lowering.lowerings.pop(op)
            lowering.make_fallback(op)
            assert feature._is_official_fallback(lowering.lowerings[op], lowering, op)
            feature.register_aoti_lowering()
            checked = lowering.lowerings[op]
            feature.register_aoti_lowering()
            assert lowering.lowerings[op] is checked
            with config.patch({'triton.cudagraphs': True}):
                try:
                    feature.register_aoti_lowering()
                except RuntimeError as error:
                    assert 'CUDA graph capture' in str(error)
                else:
                    raise AssertionError('unsafe config accepted')
            assert lowering.lowerings[op] is checked
        """)

    def test_compile_scope_disables_caches_and_restores_on_failure(self):
        self.child("""
            from torch._functorch import config as autograd_config
            from torch import _inductor
            ep = torch.export.export(torch.nn.Identity(), (torch.empty(2),))
            keys = ('force_disable_caches', 'fx_graph_cache', 'fx_graph_remote_cache')
            akeys = ('enable_autograd_cache', 'enable_remote_autograd_cache')
            before = {key: getattr(config, key) for key in keys}
            abefore = {key: getattr(autograd_config, key) for key in akeys}
            options = {'triton.cudagraphs': False}
            def observe(*args, **kwargs):
                assert args[0] is ep
                assert kwargs['package_path'] == 'contract.pt2'
                assert config.force_disable_caches is True
                assert config.fx_graph_cache is False
                assert config.fx_graph_remote_cache is False
                assert autograd_config.enable_autograd_cache is False
                assert autograd_config.enable_remote_autograd_cache is False
                assert kwargs['inductor_configs']['force_disable_caches'] is True
                kwargs['inductor_configs']['aot_inductor.package'] = True
                raise RuntimeError('compiler sentinel')
            with mock.patch.object(_inductor, 'aoti_compile_and_package', autospec=True) as compiler:
                compiler.side_effect = observe
                try:
                    feature.compile_aoti(ep, package_path='contract.pt2', inductor_configs=options)
                except RuntimeError as error:
                    assert str(error) == 'compiler sentinel'
                else:
                    raise AssertionError('compiler error swallowed')
                assert compiler.call_count == 1
                for key, value in before.items():
                    assert getattr(config, key) == value
                for key, value in abefore.items():
                    assert getattr(autograd_config, key) == value
                assert options == {'triton.cudagraphs': False}
                try:
                    feature.compile_aoti(ep, inductor_configs={'fx_graph_cache': True})
                except RuntimeError as error:
                    assert 'requires fx_graph_cache=False' in str(error)
                else:
                    raise AssertionError('cache override accepted')
                assert compiler.call_count == 1
                # A return sentinel tests forwarding only, not a fake artifact.
                sentinel = object()
                compiler.side_effect = None
                compiler.return_value = sentinel
                assert feature.compile_aoti(ep, inductor_configs=options) is sentinel
                assert compiler.call_count == 2
                for key, value in before.items():
                    assert getattr(config, key) == value
                for key, value in abefore.items():
                    assert getattr(autograd_config, key) == value
                assert options == {'triton.cudagraphs': False}
        """)

    def test_compile_rejects_non_exported_program(self):
        with self.assertRaisesRegex(TypeError, "ExportedProgram"):
            feature.compile_aoti(object())


if __name__ == "__main__":
    unittest.main()
