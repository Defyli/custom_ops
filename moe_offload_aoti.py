"""Checked AOTInductor proxy lowering for recsys_ops::fuse_moe_offload.

Opt in with register_aoti_lowering() AFTER loading the native operator and its
existing fake implementation, and BEFORE compiling any artifact. Importing this
module neither loads/builds the extension nor changes the lowering registry.
Previously compiled artifacts must be recompiled; registration is compile-time
only, and the generated artifact still uses the official C++ ProxyExecutor.
Use compile_aoti() for a scoped, cache-disabled official compile/package call.
Registration alone does not invalidate existing artifacts or external caches.
Both audited compile_fx versions bypass FX graph caching in AOT mode, but this
is not a general guarantee that a custom lowering is included in cache keys.

Private Inductor integration, deliberately version locked.
Target builds: PyTorch 2.6.0a0 and 2.11.0+cu128 (release 2.11.0). Local build
suffixes are permitted only with the reviewed interfaces; this is not binary
ABI certification of another build. Stable 2.6.0 and other releases are explicitly
unsupported pending a target-matrix review. Signature compatibility is necessary,
not proof of successful binary/GPU execution.
FallbackKernel.create/process_kernel, layouts, aliasing, MutationOutput, output
names, and extern JSON serialization remain entirely upstream-owned. Only this
operator's AOT C++ statement emission is different. No torch class, instance,
installed source, or generated source is patched.

The two isolated emitters mirror the corresponding upstream
CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup_aot methods, delegating
argument packing/output ownership to generate_extern_kernel_args_decl_if_needed.
2.6 uses full-expression std::vector temporaries; 2.11 uses the official mutable
std::array pointer helper. Keeping these temporaries INSIDE the checked call is
important: reinterpret-view RAII handles must survive until the call returns,
and pointers must be mutable even inside a const AOT run lambda.

The official error macro throws on a non-success proxy return before downstream
nodes execute. It does NOT promise preservation of the original native exception
message, rollback, context recovery, or detection of asynchronous CUDA failures
not reported by the operator. Other operators' proxy calls are not changed.
"""

import inspect
import threading

import torch

__all__ = ["register_aoti_lowering", "compile_aoti"]

_LOCK = threading.RLock()
_REGISTERED_OP = None
_REGISTERED_LOWERING = None


def _unsupported(reason):
    return RuntimeError("fuse_moe_offload AOTI lowering: " + reason)


def _version():
    # A local build suffix does not change the reviewed Python interfaces.
    release = str(torch.__version__).split("+", 1)[0]
    if release == "2.6.0a0":
        return "2.6"
    if release == "2.11.0":
        return "2.11"
    raise _unsupported(
        f"unsupported PyTorch {torch.__version__}; reviewed versions are "
        "2.6.0a0 and 2.11.0 (target build +cu128; stable 2.6.0 is unsupported)"
    )


def _signature(fn, names, *, keyword_only=()):
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError) as exc:
        raise _unsupported(f"cannot inspect {fn!r}") from exc
    label = getattr(fn, "__qualname__", type(fn).__name__)
    if tuple(parameters) != tuple(names):
        raise _unsupported(f"unreviewed parameters for {label}: {tuple(parameters)}")
    for name in keyword_only:
        if parameters[name].kind != inspect.Parameter.KEYWORD_ONLY:
            raise _unsupported(f"{label}.{name} must be keyword-only")


def _validate_interfaces(version, ir, lowering, cpp):
    create = inspect.getattr_static(ir.FallbackKernel, "create")
    if not isinstance(create, classmethod):
        raise _unsupported("FallbackKernel.create must be a subclass-aware classmethod")
    _signature(create.__func__, ("cls", "kernel", "args", "kwargs"))
    _signature(ir.FallbackKernel.codegen, ("self", "wrapper"))
    _signature(ir.FallbackKernel.export_extern_kernel_node, ("self",))
    _signature(ir.FallbackKernel.codegen_unbacked_symbol_defs, ("self", "wrapper"))
    registration_names = (
        "aten_fn", "broadcast", "type_promotion_kind", "convert_input_to_bool",
    )
    if version == "2.11":
        registration_names += ("lowering_dict",)
    _signature(lowering.register_lowering, registration_names)
    if version == "2.11" and inspect.signature(
        lowering.register_lowering
    ).parameters["lowering_dict"].default is not lowering.lowerings:
        raise _unsupported("register_lowering must target the official lowering registry")
    for method in (
        "generate_extern_kernel_args_decl_if_needed",
        "generate_fallback_kernel_with_runtime_lookup_aot",
    ):
        _signature(
            getattr(cpp.CppWrapperCpu, method),
            ("self", "op_overload", "raw_args", "output_args", "raw_outputs"),
        )
    _signature(cpp.CppWrapperCpu.writeline, ("self", "line"))
    if version == "2.6":
        names = (
            "self", "buf_name", "python_kernel_name", "cpp_kernel_name",
            "codegen_args", "op_overload", "raw_args", "outputs",
        )
    else:
        names = (
            "self", "buf_name", "python_kernel_name", "get_args",
            "op_overload", "raw_args", "outputs",
        )
        _signature(
            cpp.CppWrapperCpu._generate_temporary_array_pointer,
            ("c_type", "elements", "force_mutable"),
            keyword_only=("force_mutable",),
        )
    _signature(cpp.CppWrapperCpu.generate_fallback_kernel_with_runtime_lookup, names)
    for method in ("process_kernel", "codegen_comment", "tensor_to_layout"):
        if not callable(getattr(ir.FallbackKernel, method, None)):
            raise _unsupported(f"missing FallbackKernel.{method}")
    if not callable(getattr(lowering, "add_needs_realized_inputs", None)):
        raise _unsupported("missing official fallback realization registration")


def _validate_config(config, op):
    if config.disable_cpp_codegen:
        raise _unsupported("disable_cpp_codegen must be False")
    if config.aot_inductor.use_minimal_arrayref_interface:
        raise _unsupported("minimal ArrayRef wrappers are not supported")
    if config.triton.cudagraphs:
        raise _unsupported("CUDA graph capture is not supported by offload")
    shims = getattr(config.aot_inductor, "custom_ops_to_c_shims", {})
    if op in shims or str(op) in shims or op.name() in shims:
        raise _unsupported("a custom C-shim mapping would bypass the reviewed proxy ABI")
    if torch._dynamo.config.suppress_errors:
        raise _unsupported("Dynamo suppress_errors must be False (no silent fallback)")


def _validate_schema(op):
    schema = op._schema
    arguments = schema.arguments
    # Both the original seven-argument ABI and the extended deployment ABI are
    # supported. Never infer a different mutation contract from an op's name.
    if len(arguments) not in (7, 9):
        raise _unsupported(f"unsupported operator schema: {schema}")
    if (
        any(not isinstance(arg.type, torch.TensorType) for arg in arguments[:6])
        or not isinstance(arguments[6].type, torch.IntType)
        or len(schema.returns) != 1
        or not isinstance(schema.returns[0].type, torch.TensorType)
        or schema.returns[0].alias_info is not None
        or any(arg.kwarg_only for arg in arguments)
    ):
        raise _unsupported(f"unsupported operator schema: {schema}")
    for index, arg in enumerate(arguments):
        if index == 5:
            if arg.alias_info is None or not arg.alias_info.is_write:
                raise _unsupported("runtime_token must be an explicitly mutable Tensor")
        elif arg.alias_info is not None:
            raise _unsupported("only runtime_token may have an alias annotation")
    for arg in arguments[7:]:
        type_ = arg.type
        if (
            not isinstance(type_, torch.OptionalType)
            or not isinstance(type_.getElementType(), torch.ListType)
            or not isinstance(type_.getElementType().getElementType(), torch.IntType)
            or not arg.has_default_value()
            or arg.default_value is not None
        ):
            raise _unsupported("deployment arguments must be optional int[] with None defaults")


def _emit_checked_26(wrapper, op, raw_args, output_args, outputs, index):
    tensors, integers = wrapper.generate_extern_kernel_args_decl_if_needed(
        op, raw_args, output_args, outputs
    )
    wrapper.writeline(
        "AOTI_TORCH_ERROR_CODE_CHECK("
        f"aoti_torch_proxy_executor_call_function(proxy_executor, {index}, "
        f"{len(integers)}, std::vector<int64_t>{{{', '.join(integers)}}}.data(), "
        f"{len(tensors)}, std::vector<AtenTensorHandle>{{{', '.join(tensors)}}}.data()));"
    )


def _emit_checked_211(wrapper, op, raw_args, output_args, outputs, index):
    tensors, integers = wrapper.generate_extern_kernel_args_decl_if_needed(
        op, raw_args, output_args, outputs
    )
    int_pointer = wrapper._generate_temporary_array_pointer(
        "int64_t", integers, force_mutable=True
    )
    tensor_pointer = wrapper._generate_temporary_array_pointer(
        "AtenTensorHandle", tensors, force_mutable=True
    )
    wrapper.writeline(
        "AOTI_TORCH_ERROR_CODE_CHECK("
        f"aoti_torch_proxy_executor_call_function(proxy_executor, {index}, "
        f"{len(integers)}, {int_pointer}, {len(tensors)}, {tensor_pointer}));"
    )


def _node_class(version, ir, lowering, cpp, config, V, op):
    emit = _emit_checked_26 if version == "2.6" else _emit_checked_211

    class CheckedFallbackKernel(ir.FallbackKernel):
        # Inherit __init__ and create without reimplementing fake execution,
        # MultiOutput construction, alignment, or mutation dependency tracking.
        def codegen(self, wrapper):
            if _version() != version:
                raise _unsupported("PyTorch version changed after registration")
            _validate_config(config, op)
            if not V.graph.aot_mode:
                # Preserve the existing non-AOT lowering/codegen behavior.
                return super().codegen(wrapper)
            if not V.graph.cpp_wrapper or not isinstance(wrapper, cpp.CppWrapperCpu):
                raise _unsupported("AOT requires the reviewed C++ AtenTensorHandle wrapper")
            # Accept the official CPU base and its official CUDA-derived wrapper,
            # not arbitrary external emitters with lookalike method signatures.
            wrapper_type = type(wrapper)
            if (wrapper_type.__module__, wrapper_type.__name__) not in (
                ("torch._inductor.codegen.cpp_wrapper_cpu", "CppWrapperCpu"),
                ("torch._inductor.codegen.cpp_wrapper_gpu", "CppWrapperGpu"),
            ):
                raise _unsupported(f"unsupported wrapper type: {wrapper_type!r}")
            _validate_interfaces(version, ir, lowering, cpp)
            if self.op_overload is not op:
                raise _unsupported("checked node received a different operator")
            if len(self.outputs) != 1 or not isinstance(self.outputs[0], ir.MultiOutput):
                raise _unsupported("expected the official single Tensor MultiOutput")
            outputs = self.outputs
            output_args = [outputs[0].get_name()]
            nodes = V.graph.extern_kernel_nodes if version == "2.6" else V.extern_kernel_nodes
            index = len(nodes)
            self.use_runtime_dispatch = True
            self.codegen_comment(wrapper)
            raw_args = self.export_extern_kernel_node()
            if len(nodes) != index + 1 or nodes[index].name != self.get_name():
                raise _unsupported("official extern serialization did not append this node")
            emit(wrapper, op, raw_args, output_args, outputs, index)
            self.codegen_unbacked_symbol_defs(wrapper)

    return CheckedFallbackKernel


def _is_official_fallback(handler, lowering, op):
    original = inspect.unwrap(handler)
    exemplar = lowering.fallback_handler(op, add_to_fallback_set=False)
    return (
        getattr(original, "__code__", None) is exemplar.__code__
        and any(cell.cell_contents is op for cell in (original.__closure__ or ()))
    )


def register_aoti_lowering():
    """Register the checked lowering, idempotently, without loading the operator.

    Raises RuntimeError for a missing native schema/fake kernel, an unreviewed
    torch release/interface, incompatible config, or a conflicting lowering.
    Config and wrapper checks run again during codegen, so compile-time config
    patches cannot silently route AOT artifacts through an unchecked emitter.
    Default cpp_wrapper=False outside compilation is allowed: AOT compilation
    selects its C++ wrapper itself. Eager operator behavior is never changed.
    """
    global _REGISTERED_OP, _REGISTERED_LOWERING
    with _LOCK:
        version = _version()
        try:
            op = torch.ops.recsys_ops.fuse_moe_offload.default
        except AttributeError as exc:
            raise _unsupported("load the native operator and register its fake kernel first") from exc
        _validate_schema(op)
        if not torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Meta"):
            raise _unsupported("register the existing fake/Meta kernel before this lowering")

        from torch._inductor import config, ir, lowering
        from torch._inductor.codegen import cpp_wrapper_cpu as cpp
        from torch._inductor.virtualized import V
        from torch.utils import _pytree as pytree

        _validate_interfaces(version, ir, lowering, cpp)
        _validate_config(config, op)
        existing = lowering.lowerings.get(op)
        if op in lowering.decompositions:
            raise _unsupported("an operator decomposition would bypass the checked node")
        if _REGISTERED_OP is op and existing is _REGISTERED_LOWERING:
            return
        if existing is not None and not _is_official_fallback(existing, lowering, op):
            raise _unsupported("another lowering owns this operator; refusing to overwrite it")

        node = _node_class(version, ir, lowering, cpp, config, V, op)

        def lower(*args, **kwargs):
            _validate_config(config, op)
            return pytree.tree_map(
                lambda value: ir.TensorBox.create(value) if isinstance(value, ir.IRNode) else value,
                node.create(op, *args, **kwargs),
            )

        # Match official make_fallback registration (no dtype promotion or
        # broadcasting of mixed CPU weights/token and CUDA activations).
        lower._is_fallback_handler = True
        registered = lowering.register_lowering(op, type_promotion_kind=None)(lower)
        lowering.fallbacks.add(op)
        lowering.add_needs_realized_inputs(op)
        _REGISTERED_OP = op
        _REGISTERED_LOWERING = registered


def compile_aoti(exported_program, *, package_path=None, inductor_configs=None):
    """Compile/package with the official API, without reusing pre-lowering caches.

    Requires the native schema and existing fake implementation to be loaded.
    No model is loaded or executed here. Caller options are copied, and unsafe
    cache overrides are rejected rather than silently honored. Scoped config
    patches are restored on both success and failure; no cache directory is
    deleted. Registration remains process-wide, exactly as register_lowering.

    Config patches are process-global upstream state: callers must not run other
    Inductor compilations concurrently in this process. The feature lock only
    serializes this module's own registration/compilation calls.
    """
    _version()
    if not isinstance(exported_program, torch.export.ExportedProgram):
        raise TypeError("compile_aoti requires a torch.export.ExportedProgram")
    from torch import _inductor
    from torch._functorch import config as autograd_config
    from torch._inductor import config

    _signature(
        _inductor.aoti_compile_and_package,
        ("exported_program", "_deprecated_unused_args", "_deprecated_unused_kwargs",
         "package_path", "inductor_configs"),
        keyword_only=("package_path", "inductor_configs"),
    )
    cache_options = {
        "force_disable_caches": True,
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
    }
    autograd_options = {
        "enable_autograd_cache": False,
        "enable_remote_autograd_cache": False,
    }
    for key in cache_options:
        if not hasattr(config, key):
            raise _unsupported(f"missing reviewed cache control: {key}")
    for key in autograd_options:
        if not hasattr(autograd_config, key):
            raise _unsupported(f"missing reviewed Autograd cache control: {key}")
    if hasattr(autograd_config, "force_autograd_cache"):
        autograd_options["force_autograd_cache"] = False
    options = dict(inductor_configs or {})
    for key, value in cache_options.items():
        if key in options and options[key] is not value:
            raise _unsupported(f"compile_aoti requires {key}={value}")
    options.update(cache_options)
    with _LOCK, config.patch(options), autograd_config.patch(autograd_options):
        register_aoti_lowering()
        return _inductor.aoti_compile_and_package(
            exported_program, package_path=package_path,
            inductor_configs=dict(options),
        )
