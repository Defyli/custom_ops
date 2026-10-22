"""
custom_ops/recsys.py — RecsysOps：推荐系统核心 CUDA 算子库（分组懒加载）

算子分组（每组独立 .so、独立 JIT 编译/缓存，首次访问时才编译）：
- fa         mha_fwd_with_mask      Flash Attention 2 前向，任意 fp16/bf16 加法 mask
- mixed_gemm mixed_gemm (+fp8 查询)  混合精度 GEMM：bf16 主项 + fp8/int8 residual
- fuse_moe   fuse_moe                MoE 前向融合（group GEMM + act*mul + reduce）
- swiglu     swiglu                  SwiGLU 融合（配对 GEMM + silu*mul epilogue）


快速使用
--------
    from custom_ops import ops   # import 零编译

    if ops.is_available():       # 仅检查 CUDA 可用性，不触发编译
        # 首次访问某算子时才编译对应分组（~1-4 分钟，之后秒级缓存加载）
        out = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)
        y   = ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu")

    ops.ensure_loaded()          # 显式全量加载（旧 eager 行为）
    ops.ensure_loaded("fa")      # 显式加载单个分组

命名空间
--------
    torch.ops.recsys_ops.<name>(...)   — C++ 侧调用接口（需先加载对应分组）
    ops.<name>(...)                    — Python 门面，带类型标注与自动分组加载
"""

from __future__ import annotations

import os
import threading
import warnings

import torch

from custom_ops import CustomOps, CustomOpsError


# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # custom_ops/ 根目录

# residual 补偿 scale 的默认值（与原 TRT 插件一致）
DEFAULT_RESIDUAL_SCALE = 1.0 / 256.0

# 激活函数名 → kernel 枚举
_ACTIVATION_TO_ID = {"identity": 0, "silu": 1, "gelu": 2}

# fuse_moe 专家激活 / 数值模式：公开字符串 → native int 线格式编码。
# 两者是独立维度：expert_activation 选择专家非线性，numerical_mode 选择
# 数值契约（fused / staged），与 offload 的 activation_mode（route_buffer
# 等缓冲/累加策略）无关。编码 0 是各自的默认值，因此新增 schema 尾参可安全
# 缺省为 0，保持旧默认语义（silu / fused / glu）。新增参数改变 native
# 签名及 schema；旧二进制必须重编，C++ 默认参数只提供源码调用兼容。
EXPERT_ACTIVATION_TO_ID = {"silu": 0, "gelu": 1}
NUMERICAL_MODE_TO_ID = {"fused": 0, "staged": 1}
EXPERT_TYPE_TO_ID = {"glu": 0, "mlp": 1}


def _resolve_fuse_moe_expert_type(expert_type):
    """Resolve the explicit expert architecture, never infer it from activation."""
    if not isinstance(expert_type, str) or expert_type not in EXPERT_TYPE_TO_ID:
        raise ValueError(
            f"expert_type must be one of {list(EXPERT_TYPE_TO_ID)}, got {expert_type!r}")
    return EXPERT_TYPE_TO_ID[expert_type]


def _resolve_fuse_moe_modes(expert_activation, numerical_mode):
    """严格校验并解析 fuse_moe 的两个公开字符串为 native int 编码。

    只接受精确的字符串枚举（bool/int/None/其它大小写均拒绝），且在触发
    分组加载/编译之前完成校验。
    """
    if (not isinstance(expert_activation, str)
            or expert_activation not in EXPERT_ACTIVATION_TO_ID):
        raise ValueError(
            f"expert_activation must be one of {list(EXPERT_ACTIVATION_TO_ID)}, "
            f"got {expert_activation!r}")
    if (not isinstance(numerical_mode, str)
            or numerical_mode not in NUMERICAL_MODE_TO_ID):
        raise ValueError(
            f"numerical_mode must be one of {list(NUMERICAL_MODE_TO_ID)}, "
            f"got {numerical_mode!r}")
    return (EXPERT_ACTIVATION_TO_ID[expert_activation],
            NUMERICAL_MODE_TO_ID[numerical_mode])


def _check_fuse_moe_shapes(x, w1, w2, ids, scale, factor=None, *, allow_empty_s=False):
    """Symbolic-safe canonical metadata checks; no routing-value inspection."""
    for tensor, rank in ((x, 2), (w1, 3), (w2, 3), (ids, 2), (scale, 2)):
        torch._check(tensor.dim() == rank)
        torch._check(tensor.is_contiguous())
    s, h = x.shape
    e, _, i = w2.shape
    torch._check(s >= 0 if allow_empty_s else s > 0)
    torch._check(h > 0)
    torch._check(h <= (1 << 31) - 1)
    torch._check(h % 64 == 0)
    torch._check(e > 0)
    torch._check(e <= 512)
    torch._check(i > 0)
    torch._check(i <= ((1 << 31) - 1) // 2)
    torch._check(i % 64 == 0)
    torch._check(w1.shape[0] == e)
    torch._check(w1.shape[2] == h)
    torch._check(w2.shape[1] == h)
    if factor is None:
        torch._check((w1.shape[1] == i) | (w1.shape[1] == 2 * i))
    else:
        torch._check(w1.shape[1] == factor * i)
    torch._check(ids.shape[0] == s)
    torch._check(ids.shape[1] > 0)
    torch._check(ids.shape[1] <= 128)
    torch._check(scale.shape[0] == s)
    torch._check(scale.shape[1] == ids.shape[1])
    torch._check(x.dtype in (torch.float16, torch.bfloat16))
    torch._check(w1.dtype == x.dtype)
    torch._check(w2.dtype == x.dtype)
    torch._check(ids.dtype == torch.int32)
    torch._check(scale.dtype == torch.float32)
    torch._check(ids.device == x.device)
    torch._check(scale.device == x.device)


def _fake_fuse_moe(x, gate_up_weight, down_weight, topk_ids, topk_scale,
                   expert_activation=0, numerical_mode=0, expert_type=0):
    """Meta/fake implementation synchronized with native trailing defaults."""
    torch._check(expert_activation in (0, 1))
    torch._check(numerical_mode in (0, 1))
    torch._check(expert_type in (0, 1))
    _check_fuse_moe_shapes(x, gate_up_weight, down_weight, topk_ids, topk_scale,
                          2 if expert_type == 0 else 1)
    torch._check(x.device.type == "cuda")
    torch._check(gate_up_weight.device == x.device)
    torch._check(down_weight.device == x.device)
    return torch.empty_like(x)


def _resolve_backend(backend):
    """解析 backend 参数：显式传参 > 环境变量 CUSTOM_OPS_BACKEND > "auto"。

    - "cuda"：手写 CUDA/CuTe kernel（默认行为，与历史版本一致）
    - "tilelang"：Tile-lang 后端（custom_ops.tilelang_ops，无需 nvcc/
      架构专用路径，Python DSL JIT）
    - "auto"：CUDA 分组加载成功则用 CUDA；失败（无 nvcc / 架构不支持 /
      编译错误）时自动回退 Tile-lang
    """
    if backend is None:
        backend = os.environ.get("CUSTOM_OPS_BACKEND", "auto")
    backend = str(backend).lower()
    if backend not in ("auto", "cuda", "tilelang"):
        raise ValueError(
            f"backend must be 'auto' | 'cuda' | 'tilelang', got {backend!r} "
            f"(or set CUSTOM_OPS_BACKEND env)")
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# 分组 kernel 库（每组一个 CustomOps 子类 → 独立 .so）
# ─────────────────────────────────────────────────────────────────────────────

class FAOps(CustomOps):
    """FA 分组：mha_fwd_with_mask（FlashAttention-2 + 任意加法 mask）。

    sm70 / sm89 / sm120 三路径互不依赖，编译期由 FA_TARGETS 裁剪。
    """

    namespace    = "recsys_ops"
    requires_cuda = True
    so_name      = "recsys_fa_kernel"
    required_ops = ["mha_fwd_with_mask"]
    op_schemas = {
        "mha_fwd_with_mask": "mha_fwd_with_mask(Tensor q, Tensor k, Tensor v, Tensor mask) -> Tensor",
    }
    fake_ops = ["mha_fwd_with_mask"]
    optional_ops = []
    build_dir_fallback = "./torch_extensions"

    def get_sources(self) -> list:
        fa = os.path.join(_THIS_DIR, "csrc", "fa")
        return [
            os.path.join(fa, "fa_bindings.cpp"),
            os.path.join(fa, "fa_fwd_op.cu"),
        ]

    def get_include_dirs(self) -> list:
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,
            csrc,
            os.path.join(csrc, "fa"),
        ]

    def register_fake_impls(self) -> None:
        self.register_fake_for(
            "mha_fwd_with_mask",
            lambda q, k, v, mask: torch.empty_like(q),
        )


class MixedGemmOps(CustomOps):
    """mixed_gemm 分组：混合精度 GEMM（bf16 主项 + fp8/int8 residual 补偿）。"""

    namespace    = "recsys_ops"
    requires_cuda = True
    so_name      = "recsys_mixed_gemm_kernel"
    required_ops = ["mixed_gemm", "mixed_gemm_fp8_available"]
    op_schemas = {
        "mixed_gemm": (
            "mixed_gemm(Tensor x, Tensor w_high, Tensor w_low, Tensor? w_scale=None, "
            "float scale=0.00390625, Tensor? bias=None, int activation=0, "
            "bool fp32_output=True, int force_splitk=0) -> Tensor"
        ),
        "mixed_gemm_fp8_available": "mixed_gemm_fp8_available() -> bool",
    }
    fake_ops = ["mixed_gemm", "mixed_gemm_fp8_available"]
    optional_ops = []
    build_dir_fallback = "./torch_extensions"

    def get_sources(self) -> list:
        mixed = os.path.join(_THIS_DIR, "csrc", "mixed_gemm")
        return [
            os.path.join(mixed, "mixed_gemm_bindings.cpp"),
            os.path.join(mixed, "mixed_gemm_op.cu"),
            os.path.join(mixed, "gemm_bf16xfp32_sm80.cu"),
            os.path.join(mixed, "gemm_bf16xfp32_sm120.cu"),
        ]

    def get_include_dirs(self) -> list:
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,
            csrc,
            os.path.join(csrc, "mixed_gemm"),
        ]

    def register_fake_impls(self) -> None:
        self.register_fake_for(
            "mixed_gemm",
            lambda x, w_high, w_low, w_scale, scale, bias, activation, \
                   fp32_output, force_splitk: torch.empty(
                x.shape[:-1] + (w_high.shape[0],),
                dtype=torch.float32 if fp32_output else torch.bfloat16,
                device=x.device),
        )
        self.register_fake_for(
            "mixed_gemm_fp8_available",
            lambda: True,
        )


class FuseMoeOps(CustomOps):
    """fuse_moe 分组：MoE 前向融合算子（group GEMM + act*mul + reduce）。"""

    namespace    = "recsys_ops"
    requires_cuda = True
    so_name      = "recsys_fuse_moe_kernel"
    required_ops = [
        "fuse_moe", "fuse_moe_offload", "fuse_moe_offload_create",
        "fuse_moe_offload_bind", "fuse_moe_offload_resident_bind",
        "fuse_moe_offload_validate", "fuse_moe_offload_close", "fuse_moe_offload_stats",
        "fuse_moe_offload_begin", "fuse_moe_offload_check", "fuse_moe_offload_status",
        "fuse_moe_offload_acknowledge",
    ]
    op_schemas = {
        "fuse_moe": (
            "fuse_moe(Tensor x, Tensor gate_up_weight, Tensor down_weight, "
            "Tensor topk_ids, Tensor topk_scale, int expert_activation=0, "
            "int numerical_mode=0, int expert_type=0) -> Tensor"
        ),
        "fuse_moe_offload": (
            "fuse_moe_offload(Tensor x, Tensor gate_up_weight_cpu, Tensor down_weight_cpu, "
            "Tensor topk_ids, Tensor topk_scale, Tensor(a!) runtime_token, int layer_id, "
            "int[]? deployment_config=None, int[]? deployment_manifest=None) -> Tensor"
        ),
        "fuse_moe_offload_create": (
            "fuse_moe_offload_create(int device, int gpu_budget_bytes=0, int ram_budget_bytes=0, "
            "int pinned_budget_bytes=0, int num_banks=2, int activation_mode=0, "
            "int max_route_tile_rows=4096, int resident_cache_bytes=0, int dynamic_cache_bytes=0, "
            "int cache_policy=0, int cache_decay_interval=32, int cache_min_reuse=2, "
            "int cache_update_bytes=0, bool async_staging=False, bool host_profile=False, "
            "int tile_m=0, int tile_n=128, int expert_activation=0, "
            "int numerical_mode=0, int expert_type=0) -> Tensor"
        ),
        "fuse_moe_offload_bind": (
            "fuse_moe_offload_bind(Tensor runtime_token, int layer_id, Tensor w1, Tensor w2) -> ()"
        ),
        "fuse_moe_offload_resident_bind": (
            "fuse_moe_offload_resident_bind(Tensor token, int layer_id, int[] expert_ids) -> ()"
        ),
        "fuse_moe_offload_validate": (
            "fuse_moe_offload_validate(Tensor runtime_token, int[] deployment_manifest) -> ()"
        ),
        "fuse_moe_offload_close": "fuse_moe_offload_close(Tensor runtime_token) -> ()",
        "fuse_moe_offload_stats": "fuse_moe_offload_stats(Tensor runtime_token) -> Tensor",
        "fuse_moe_offload_begin": "fuse_moe_offload_begin(Tensor runtime_token) -> ()",
        "fuse_moe_offload_check": "fuse_moe_offload_check(Tensor runtime_token) -> ()",
        "fuse_moe_offload_status": "fuse_moe_offload_status(Tensor runtime_token) -> Tensor",
        "fuse_moe_offload_acknowledge": "fuse_moe_offload_acknowledge(Tensor runtime_token) -> ()",
    }
    # Graph-out lifecycle/status ops do not need fake execution. Test-only hooks
    # are not deployment requirements, but must not appear in production libs.
    testing_ops = ["_fuse_moe_offload_test_inject", "_fuse_moe_offload_test_worker_states"]
    fake_ops = ["fuse_moe", "fuse_moe_offload"]
    optional_ops = []
    build_dir_fallback = "./torch_extensions"

    def build(
        self, output_directory: str | os.PathLike, *,
        cuda_arch_list: str | None = None, verbose: bool = False,
        testing: bool = False, extra_cflags: list[str] | tuple[str, ...] = (),
        extra_cuda_cflags: list[str] | tuple[str, ...] = (),
    ):
        """Export production by default; testing=True enables native fault hooks.

        Test artifacts require load_precompiled(..., allow_testing=True) in a
        separate deployment process. The macro must reach the CPU .cpp compiler,
        not just nvcc. Additional flags augment, never replace, default flags.
        """
        if not isinstance(testing, bool):
            raise TypeError("testing must be a bool")
        if not isinstance(extra_cflags, (list, tuple)):
            raise TypeError("extra_cflags must be a list or tuple of strings")
        cflags = list(extra_cflags)
        if testing:
            cflags.append("-DCUSTOM_OPS_OFFLOAD_TESTING=1")
        return super().build(
            output_directory, cuda_arch_list=cuda_arch_list, verbose=verbose,
            extra_cflags=cflags, extra_cuda_cflags=extra_cuda_cflags,
        )

    def get_sources(self) -> list:
        moe = os.path.join(_THIS_DIR, "csrc", "fuse_moe")
        return [
            os.path.join(moe, "fuse_moe_bindings.cpp"),
            os.path.join(moe, "fuse_moe_op.cu"),
            os.path.join(moe, "fuse_moe_offload.cpp"),
        ]

    def get_include_dirs(self) -> list:
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,
            csrc,
            os.path.join(csrc, "fuse_moe"),
        ]

    def register_fake_impls(self) -> None:
        self.register_fake_for("fuse_moe", _fake_fuse_moe)
        from custom_ops.moe_offload import _fake_offload
        self.register_fake_for("fuse_moe_offload", _fake_offload)


# ── SwiGLU 独立分组（自包含 kernel，与 fuse_moe 完全解耦）─────────────

class SwigluOps(CustomOps):
    """swiglu 分组：SwiGLU 融合算子（配对 GEMM + silu*mul epilogue）。

    设备 kernel 自包含于 csrc/swiglu/（不含任何 fuse_moe 头）——编译
    swiglu 分组完全不会触碰 fuse_moe 的代码，两组互不影响。
    """

    namespace    = "recsys_ops"
    requires_cuda = True
    so_name      = "recsys_swiglu_kernel"
    required_ops = ["swiglu"]
    op_schemas = {"swiglu": "swiglu(Tensor x, Tensor weight) -> Tensor"}
    fake_ops = ["swiglu"]
    optional_ops = []
    build_dir_fallback = "./torch_extensions"

    def get_sources(self) -> list:
        return [
            os.path.join(_THIS_DIR, "csrc", "swiglu", "swiglu_bindings.cpp"),
            os.path.join(_THIS_DIR, "csrc", "swiglu", "swiglu_op.cu"),
        ]

    def get_include_dirs(self) -> list:
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,
            csrc,
            os.path.join(csrc, "swiglu"),
        ]

    def register_fake_impls(self) -> None:
        self.register_fake_for(
            "swiglu",
            lambda x, weight: torch.empty(
                x.shape[:-1] + (weight.shape[0] // 2,),
                dtype=x.dtype, device=x.device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# RecsysOps — 分组懒加载门面
# ─────────────────────────────────────────────────────────────────────────────

class RecsysOps:
    """
    推荐系统算子库门面：按算子分组懒加载。

    - ``ops.<op>(...)``：首次访问时 JIT 编译/加载该算子所属分组（独立 .so），
      之后走缓存（秒级）。分组间互不影响——只测 fuse_moe 不会编译 FA。
    - ``ops.is_available()``：仅检查 CUDA 可用性（不触发编译）；已加载过分组
      时同时反映其编译状态。某分组编译失败时，访问该分组算子会抛出带完整
      错误信息的 RuntimeError（响亮失败，不静默）。
    - ``ops.ensure_loaded([group])``：显式加载（无参 = 全部分组，等价旧
      ``RecsysOps().load()`` 的 eager 行为）。

    分组与算子映射
    --------------
    fa          : mha_fwd_with_mask
    mixed_gemm  : mixed_gemm, mixed_gemm_fp8_available
    fuse_moe    : fuse_moe
    """

    _GROUP_CLASSES = {
        "fa": FAOps,
        "mixed_gemm": MixedGemmOps,
        "fuse_moe": FuseMoeOps,
        "swiglu": SwigluOps,
    }

    _OP_TO_GROUP = {
        "mha_fwd_with_mask": "fa",
        "mixed_gemm": "mixed_gemm",
        "mixed_gemm_fp8_available": "mixed_gemm",
        "fuse_moe": "fuse_moe",
        "fuse_moe_offload": "fuse_moe",
        "fuse_moe_offload_create": "fuse_moe",
        "fuse_moe_offload_bind": "fuse_moe",
        "fuse_moe_offload_resident_bind": "fuse_moe",
        "fuse_moe_offload_validate": "fuse_moe",
        "fuse_moe_offload_close": "fuse_moe",
        "fuse_moe_offload_stats": "fuse_moe",
        "fuse_moe_offload_begin": "fuse_moe",
        "fuse_moe_offload_check": "fuse_moe",
        "fuse_moe_offload_status": "fuse_moe",
        "fuse_moe_offload_acknowledge": "fuse_moe",
        "swiglu": "swiglu",
    }

    def __init__(self, *, allow_jit: bool = True) -> None:
        self._groups: dict = {}   # group name -> CustomOps instance
        self._lock = threading.RLock()
        self._allow_jit = True
        self.configure(allow_jit=allow_jit)

    @property
    def allow_jit(self) -> bool:
        """Facade-wide policy; strict mode also forbids Tile-lang/backend fallback."""
        return self._allow_jit

    @allow_jit.setter
    def allow_jit(self, value: bool) -> None:
        self.configure(allow_jit=value)

    def configure(self, *, allow_jit: bool) -> "RecsysOps":
        """Explicitly configure all current and future groups, without loading any."""
        if not isinstance(allow_jit, bool):
            raise TypeError("allow_jit must be a bool")
        with self._lock:
            self._allow_jit = allow_jit
            for instance in self._groups.values():
                instance.configure(allow_jit=allow_jit)
        return self

    def disable_jit(self) -> "RecsysOps":
        return self.configure(allow_jit=False)

    # ── 分组管理 ──────────────────────────────────────────────────────────────

    def _validate_group(self, name: str) -> None:
        if name not in self._GROUP_CLASSES:
            raise ValueError(
                f"unknown group {name!r}; available: {sorted(self._GROUP_CLASSES)}")

    def _group(self, name: str) -> CustomOps:
        """Lazy development loading; strict mode accepts only loaded groups."""
        with self._lock:
            self._validate_group(name)
            if not self._allow_jit:
                instance = self._groups.get(name)
                if instance is None or not instance.is_available():
                    error = instance.load_error() if instance is not None else None
                    raise CustomOpsError(
                        f"JIT is disabled for group {name!r}; call "
                        f"load_precompiled({name!r}, library_path, metadata_path=None) first."
                        + (f" Previous error: {error}" if error else ""))
                return instance
            if name not in self._groups:
                instance = self._GROUP_CLASSES[name](allow_jit=True)
                self._groups[name] = instance
                instance.load()
            return self._groups[name]

    def load_precompiled(
        self, group: str, library_path: str | os.PathLike,
        metadata_path: str | os.PathLike | None = None, *, allow_testing: bool = False,
    ) -> "RecsysOps":
        """Explicit deployment load; disables JIT/backend fallback for ALL groups.

        Returns self. Default metadata path is '<library_path>.json'. The package
        singleton follows exactly the same policy, so PrefillMoERuntime's later
        global ops.ensure_loaded('fuse_moe') only observes the already loaded group.
        Failure leaves strict policy enabled and the group unavailable; no native
        sources, nvcc or Ninja are needed for loading. Never swaps dispatcher libs.
        Test-hook artifacts are rejected unless allow_testing=True is explicit.
        """
        with self._lock:
            self._validate_group(group)
            self.configure(allow_jit=False)
            if group not in self._groups:
                self._groups[group] = self._GROUP_CLASSES[group](allow_jit=False)
            self._groups[group].load_precompiled(
                library_path, metadata_path, allow_testing=allow_testing,
            )
        return self

    def ensure_loaded(self, *names) -> "RecsysOps":
        """Development: lazy load selected/all groups. Strict: verify, never compile."""
        targets = names if names else tuple(self._GROUP_CLASSES)
        with self._lock:
            for name in targets:
                self._validate_group(name)
            for name in targets:
                self._group(name)
        return self

    # load() 保持旧 API 兼容：RecsysOps().load() == 全量 eager 加载
    def load(self) -> "RecsysOps":
        return self.ensure_loaded()

    @property
    def groups(self) -> dict:
        """{分组名: 状态} 视图（loaded/failed/not_loaded）。"""
        with self._lock:
            return {
                name: ("loaded" if self._groups[name].is_available() else "failed")
                if name in self._groups else "not_loaded"
                for name in self._GROUP_CLASSES
            }

    def is_available(self, group: str | None = None) -> bool:
        """Development: a named group loads lazily. Strict: a read-only query.

        Without a group, CUDA availability and all attempted groups are checked;
        strict mode additionally requires at least one explicitly loaded group.
        """
        with self._lock:
            if group is not None:
                self._validate_group(group)
                if not self._allow_jit:
                    instance = self._groups.get(group)
                    return instance is not None and instance.is_available()
                return self._group(group).is_available()
            if not torch.cuda.is_available():
                return False
            if not self._allow_jit and not self._groups:
                return False
            return all(instance.is_available() for instance in self._groups.values())

    def load_error(self, group: str | None = None):
        """Read-only error query; unlike ensure_loaded, never starts a build."""
        with self._lock:
            if group is not None:
                self._validate_group(group)
                instance = self._groups.get(group)
                return instance.load_error() if instance is not None else None
            for instance in self._groups.values():
                if not instance.is_available():
                    return instance.load_error()
        return None

    # ── 后端选择 ──────────────────────────────────────────────────────────

    def _tilelang(self):
        """返回 Tile-lang 后端门面（懒导入；不可用时抛出 informative 错误）。"""
        try:
            from custom_ops.tilelang_ops import tilelang_ops
        except Exception as e:
            raise RuntimeError(
                "Tile-lang 后端不可用：custom_ops.tilelang_ops 导入失败"
                "（pip install tile-lang 后重试）") from e
        if not tilelang_ops.is_available():
            raise RuntimeError("Tile-lang 后端不可用：需要 tile-lang 与 CUDA")
        return tilelang_ops

    def _dispatch(self, backend, group):
        """返回 (是否走 tile-lang, tile-lang 门面或 None)。

        auto：CUDA 分组加载成功 → CUDA；失败 → Tile-lang（不可用时抛出
        包含 CUDA 编译错误信息的 RuntimeError）。"""
        with self._lock:
            if not self._allow_jit:
                # No environment-based backend switch and no implicit Tile-lang JIT
                # in deployment mode. An explicitly requested non-native backend fails.
                if backend is not None and _resolve_backend(backend) == "tilelang":
                    raise CustomOpsError("Tile-lang/backend fallback is disabled when allow_jit=False")
                self._group(group)
                return False, None
            b = _resolve_backend(backend)
        if b == "tilelang":
            return True, self._tilelang()
        if b == "auto" and not self._group(group).is_available():
            tl = self._tilelang()   # tile-lang 也不可用时抛错
            err = self._group(group).load_error()
            warnings.warn(
                f"CUDA 分组 '{group}' 加载失败，自动回退 Tile-lang 后端"
                + (f"（原因：{err}）" if err else ""))
            return True, tl
        return False, None

    # ── 算子访问 ──────────────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        """ops.<op_name> → 懒加载所属分组 → torch.ops.recsys_ops.<op_name>。"""
        if name.startswith("_"):
            raise AttributeError(name)
        group = self._OP_TO_GROUP.get(name)
        if group is None:
            raise AttributeError(
                f"No op '{name}' in recsys_ops. Available ops: "
                f"{sorted(self._OP_TO_GROUP)} (groups: {sorted(self._GROUP_CLASSES)})")
        return getattr(self._group(group), name)

    def __repr__(self) -> str:
        return (f"RecsysOps(lazy, allow_jit={self.allow_jit}, "
                f"namespace='recsys_ops', groups={self.groups})")

    # ── 显式方法：带类型标注，供 IDE 补全（与门面语义一致，均懒加载）──────

    def mha_fwd_with_mask(
        self,
        q:    torch.Tensor,   # (B, H,  Sq, d)   fp16/bf16 CUDA
        k:    torch.Tensor,   # (B, Hk, Sk, d)   fp16/bf16 CUDA
        v:    torch.Tensor,   # (B, Hk, Sk, d)   fp16/bf16 CUDA
        mask: torch.Tensor,   # (B, 1, ≥Sq, ≥Sk) 与 q 同 dtype，0=可见 / -inf=屏蔽
        backend: str | None = None,   # "auto" | "cuda" | "tilelang"
    ) -> torch.Tensor:
        """Flash Attention 2 前向，支持任意 fp16/bf16 加法 mask（懒加载 FA 分组）。

        backend="tilelang" 时走 Tile-lang 后端；"auto" 时 CUDA 不可用自动回退。
        """
        tl, facade = self._dispatch(backend, "fa")
        if tl:
            return facade.mha_fwd_with_mask(q, k, v, mask)
        return self._group("fa").mha_fwd_with_mask(q, k, v, mask)

    def mixed_gemm(
        self,
        x:       torch.Tensor,                      # (..., K) fp32 或 bf16
        w_high:  torch.Tensor,                      # (N, K) bf16
        w_low:   torch.Tensor,                      # (N, K) fp8_e4m3 或 int8
        w_scale: torch.Tensor | None = None,        # (N,) fp32，仅 INT8 后端
        *,
        scale:        float = DEFAULT_RESIDUAL_SCALE,
        bias:         torch.Tensor | None = None,
        activation:   str | int = "identity",
        out_dtype:    torch.dtype | None = None,
        force_splitk: int = 0,
        backend:      str | None = None,   # "auto" | "cuda" | "tilelang"
    ) -> torch.Tensor:
        """混合精度 GEMM：y = activation(x @ (w_high + w_low*scale)^T + bias)。

        backend="tilelang" 时走 Tile-lang 后端；"auto" 时 CUDA 不可用自动回退。
        """
        tl, facade = self._dispatch(backend, "mixed_gemm")
        if tl:
            if isinstance(activation, int):
                activation = {v: k for k, v in _ACTIVATION_TO_ID}[activation]
            return facade.mixed_gemm(
                x, w_high, w_low, w_scale, scale=scale, bias=bias,
                activation=activation, out_dtype=out_dtype)
        if isinstance(activation, str):
            if activation not in _ACTIVATION_TO_ID:
                raise ValueError(
                    f"activation must be one of {list(_ACTIVATION_TO_ID)}, "
                    f"got {activation!r}")
            activation = _ACTIVATION_TO_ID[activation]
        if out_dtype is None:
            out_dtype = torch.float32
        if out_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("out_dtype must be torch.float32 or torch.bfloat16")
        return self._group("mixed_gemm").mixed_gemm(
            x, w_high, w_low, w_scale, scale, bias,
            activation, out_dtype == torch.float32, force_splitk)

    def mixed_gemm_fp8_available(self) -> bool:
        """FP8 residual 后端可用性（懒加载 mixed_gemm 分组）。"""
        return bool(self._group("mixed_gemm").mixed_gemm_fp8_available())

    def fuse_moe(
        self,
        x:              torch.Tensor,                      # (S, H) bf16/fp16
        gate_up_weight: torch.Tensor,                      # (E, 2I or I, H) 同 dtype
        down_weight:    torch.Tensor,                      # (E, H, I) 同 dtype
        topk_ids:       torch.Tensor,                      # (S, K) int32
        topk_scale:     torch.Tensor,                      # (S, K) fp32
        backend:        str | None = None,   # "auto" | "cuda" | "tilelang"
        *,
        expert_activation: str = "silu",   # "silu" | "gelu"
        numerical_mode:    str = "fused",  # "fused" | "staged"
        expert_type:       str = "glu",    # "glu" | "mlp"
    ) -> torch.Tensor:
        """
        MoE 前向融合算子（单 GPU，bf16/fp16 输入，fp32 累加）：

            y[s] = Σ_j topk_scale[s,j] * (Down_ej @ (act(Gate_ej @ x[s]) * (Up_ej @ x[s])))

        限制：H % 64 == 0 且 I % 64 == 0；num_topk <= 128；num_expert <= 512；
        SM80+（sm89 / sm120）。

        expert_type='glu' 使用 act(gate) * up；'mlp' 只有一个 projection + act，
        w1 分别为 (E, 2I, H) / (E, I, H)，w2 均为 (E, H, I)。
        expert_activation 选择专家非线性（silu / gelu）；numerical_mode 选择
        数值契约（fused / staged）。两者均为关键字参数并严格校验，默认
        保持 silu / fused（native int 编码 0/0）。staged 将 gate/up、激活结果、
        激活乘积、down 结果、加权积依次 round 到 Tout（与 x 同 dtype），
        再以 FP32 归约并 round 到 Tout 输出；不承诺跨后端逐 bit 一致。

        backend="tilelang" 时走 Tile-lang 后端；"auto" 时 CUDA 不可用自动回退。
        """
        expert_activation_id, numerical_mode_id = _resolve_fuse_moe_modes(
            expert_activation, numerical_mode)
        expert_type_id = _resolve_fuse_moe_expert_type(expert_type)
        tl, facade = self._dispatch(backend, "fuse_moe")
        if tl:
            return facade.fuse_moe(
                x, gate_up_weight, down_weight, topk_ids, topk_scale,
                expert_activation=expert_activation,
                numerical_mode=numerical_mode, expert_type=expert_type)
        return self._group("fuse_moe").fuse_moe(
            x, gate_up_weight, down_weight, topk_ids, topk_scale,
            expert_activation_id, numerical_mode_id, expert_type_id)

    def swiglu(
        self,
        x:      torch.Tensor,   # (M, K) bf16/fp16
        weight: torch.Tensor,   # (2N, K) 同 dtype，gate 行在前 up 在后
        backend: str | None = None,   # "auto" | "cuda" | "tilelang"
    ) -> torch.Tensor:
        """
        SwiGLU 融合算子（单 GPU，bf16/fp16 输入，fp32 累加）：

            y = silu(x @ Wg^T) * (x @ Wu^T)

        单 kernel 完成配对 GEMM + 激活（无 (M, 2N) gate_up 中间量物化）。
        限制：K % 64 == 0，N % 64 == 0；M 任意；SM80+（sm89 / sm120）。

        backend="tilelang" 时走 Tile-lang 后端；"auto" 时 CUDA 不可用自动回退。
        """
        tl, facade = self._dispatch(backend, "swiglu")
        if tl:
            return facade.swiglu(x, weight)
        return self._group("swiglu").swiglu(x, weight)


# ─────────────────────────────────────────────────────────────────────────────
# 权重切分（离线，一次性）—— 纯 PyTorch 实现，无需编译算子库
# ─────────────────────────────────────────────────────────────────────────────


def split_mixed_precision_weight(
    w: torch.Tensor,
    scale: float = DEFAULT_RESIDUAL_SCALE,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    将 fp32 权重切分为 mixed_gemm 所需的 (w_high, w_low, w_scale) 三元组。

    分解语义（与 kernel 数值路径一致）：
        w ≈ w_high + w_low * scale
        w_high = bf16(w)                          主项，bf16 tensor core
        w_low  = quant((w - w_high) / scale)      residual 项，低精度 tensor core
    """
    if w.dim() != 2:
        raise ValueError(f"w must be 2D (N, K), got {tuple(w.shape)}")
    if backend not in ("auto", "fp8", "int8"):
        raise ValueError(f"backend must be 'auto' | 'fp8' | 'int8', got {backend!r}")

    w_f32 = w.to(torch.float32)
    w_high = w_f32.to(torch.bfloat16)
    # residual 从 bf16 舍入值再量化，与 kernel 的数值语义一致
    residual_bf16 = ((w_f32 - w_high.to(torch.float32)) / scale).to(torch.bfloat16)

    if backend == "auto":
        backend = "fp8" if _fp8_available_cached() else "int8"

    if backend == "fp8":
        return w_high, residual_bf16.to(torch.float8_e4m3fn), None

    # INT8：per-channel（每个输出通道一行）对称量化
    amax = residual_bf16.abs().amax(dim=1)                     # (N,)
    w_scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    w_low = torch.round(residual_bf16 / w_scale.unsqueeze(1))
    w_low = w_low.clamp(-127, 127).to(torch.int8)
    return w_high, w_low, w_scale.to(torch.float32)


# FP8 可用性进程级缓存（探测需要加载 mixed_gemm 分组，避免反复触发）
_fp8_available_cache: bool | None = None


def _fp8_available_cached() -> bool:
    global _fp8_available_cache
    if _fp8_available_cache is None:
        try:
            # 运行期导入包级单例（此时 custom_ops 已完成初始化，无循环导入）
            from custom_ops import ops
            _fp8_available_cache = ops.is_available("mixed_gemm") and bool(
                torch.ops.recsys_ops.mixed_gemm_fp8_available())
        except Exception:
            _fp8_available_cache = False
    return _fp8_available_cache
