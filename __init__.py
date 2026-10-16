"""
custom_ops/__init__.py — 通用 CUDA 算子插件库框架

设计目标
--------
提供一套可复用的 PyTorch 自定义 CUDA 算子加载框架，业务方只需：
  1. 继承 CustomOps
  2. 覆盖少量配置属性（namespace, so_name, sources, include_dirs）
  3. 在 csrc/bindings.cpp 中用 CUSTOM_OPS_MACROS 注册算子

即可获得：
  - 自动 JIT 编译 / 快速 dlopen 加载
  - 多进程安全（文件锁）
  - GPU 架构自动探测
  - GCC 版本自动配置
  - is_available() / 优雅降级
  - torch.ops.<namespace>.<name>(...) 统一调用接口

快速上手
--------
    from custom_ops import CustomOps

    class MyOps(CustomOps):
        namespace   = "my_ops"          # torch.ops 命名空间
        so_name     = "my_ops_kernel"   # 编译产物 .so 名（不含 .so 后缀）
        required_ops = ["my_op_a"]      # 必需算子，缺失则加载失败
        optional_ops = ["my_op_b"]      # 可选算子，缺失仅警告

        def get_sources(self):
            return ["/path/to/csrc/bindings.cpp",
                    "/path/to/csrc/my_op.cu"]

        def get_include_dirs(self):
            return ["/path/to/thirdparty"]

    ops = MyOps().load()

    if ops.is_available():
        result = ops.my_op_a(tensor)
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import tempfile
import threading
import time
import warnings
from typing import Optional

import torch


# Dispatcher registrations cannot be unloaded. Keep both successful and failed
# attempts: after dlopen starts, retrying a different binary is unsafe.
_LIBRARY_LOCK = threading.RLock()
_LIBRARIES: dict = {}
_OP_OWNERS: dict = {}
_METADATA_VERSION = 2


class CustomOpsError(RuntimeError):
    """An extension could not satisfy its build/loading contract."""


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _metadata_fingerprint(metadata: dict) -> str:
    return _json_hash({key: value for key, value in metadata.items()
                       if key not in ("library", "build_fingerprint")})


@contextmanager
def _arch_environment(targets: list[str]):
    # cpp_extension reads this at build time. Serialize and restore explicit
    # targets so an offline build does not reconfigure later development JIT.
    with _LIBRARY_LOCK:
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        try:
            if targets:
                os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(targets)
            yield
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def _file_stat(path: Path) -> tuple:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _library_identity(path: Path) -> tuple:
    before = _file_stat(path)
    digest = _file_sha256(path)
    if before != _file_stat(path):
        raise CustomOpsError(f"Library changed while being verified: {path}")
    return (before[0], before[1], digest), before


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_compatibility() -> dict:
    # These are known compatibility checks, NOT a general ABI guarantee.
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "pybind11_abi": {
            name: getattr(torch._C, f"_PYBIND11_{name}", None)
            for name in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI")
        },
    }


def _schema(qualified: str) -> str:
    name, _, overload = qualified.partition(".")
    return str(torch._C._dispatch_find_schema_or_throw(name, overload).schema())


def _has_schema(qualified: str) -> bool:
    try:
        _schema(qualified)
        return True
    except RuntimeError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具函数（私有，不对外暴露）
# ─────────────────────────────────────────────────────────────────────────────

def _gcc_major(path: str) -> int:
    """返回给定 gcc 可执行文件的主版本号，失败返回 0。"""
    import subprocess
    try:
        out = subprocess.check_output(
            [path, "-dumpversion"], stderr=subprocess.DEVNULL, text=True
        )
        return int(out.strip().split(".")[0])
    except Exception:
        return 0


def setup_compiler(
    devtoolset_path: str = "/opt/rh/devtoolset-10/root/usr/bin",
) -> None:
    """
    确保编译器满足 PyTorch 2.6+ 要求的 GCC >= 9。

    处理逻辑（三层保护）
    --------------------
    1. 若环境中已有 CC/CXX 且版本 >= 9，直接跳过。
    2. 尝试 devtoolset_path 中的 GCC 10。
    3. 若不满足，打印警告。

    参数
    ----
    devtoolset_path : str
        GCC >= 9 编译器所在目录。
        可通过环境变量 CUSTOM_OPS_GCC_BIN 覆盖，例如：
            export CUSTOM_OPS_GCC_BIN=/usr/local/bin
    """
    # 环境变量优先
    env_gcc_bin = os.environ.get("CUSTOM_OPS_GCC_BIN", "").strip()
    if env_gcc_bin:
        devtoolset_path = env_gcc_bin

    user_cc  = os.environ.get("CC",  "")
    user_cxx = os.environ.get("CXX", "")
    if user_cc and user_cxx and _gcc_major(user_cc) >= 9:
        return

    gcc = os.path.join(devtoolset_path, "gcc")
    gxx = os.path.join(devtoolset_path, "g++")

    if os.path.exists(gcc) and os.path.exists(gxx):
        os.environ["CC"]  = gcc
        os.environ["CXX"] = gxx
        path_dirs = os.environ.get("PATH", "").split(":")
        if devtoolset_path not in path_dirs:
            os.environ["PATH"] = devtoolset_path + ":" + os.environ.get("PATH", "")
    else:
        warnings.warn(
            f"[CustomOps] GCC >= 9 not found at {devtoolset_path}. "
            "PyTorch 2.6+ requires GCC >= 9. Set CUSTOM_OPS_GCC_BIN or CC/CXX env vars."
        )


def _canonical_arch_list(value: str) -> list[str]:
    """Keep the full target, including architecture-specific ISA and PTX."""
    targets = set()
    for token in value.replace(";", " ").split():
        match = re.fullmatch(r"(?:sm_|compute_)(\d+)([af]?)(\+PTX)?", token)
        if match:
            number = int(match[1])
            token = f"{number // 10}.{number % 10}{match[2]}" + (
                "+PTX" if token.startswith("compute_") or match[3] else "")
        match = re.fullmatch(r"(\d+)\.([0-9])([af]?)(\+PTX)?", token)
        if not match:
            raise CustomOpsError(
                f"Invalid CUDA target {token!r}. Use numerical "
                "TORCH_CUDA_ARCH_LIST targets, e.g. '8.9;12.0a', with optional +PTX.")
        targets.add(f"{int(match[1])}.{match[2]}{match[3]}{match[4] or ''}")
    return sorted(targets, key=lambda target: (
        int(target.split(".")[0]), int(target.split(".")[1][0]), target))


def _parse_arch_list(env_val: str) -> list:
    """Return numeric capabilities for FA_TARGETS, not for artifact matching."""
    return sorted({int(target.split(".")[0]) * 10 + int(target.split(".")[1][0])
                   for target in _canonical_arch_list(env_val)})

def _detect_archs() -> list:
    """
    解析/探测本地目标架构，返回两位整数编码列表（如 7.0 → 70、8.9 → 89、12.0 → 120）。

    优先级：
      1. TORCH_CUDA_ARCH_LIST 已设置 → 解析之。
         （显式编译其它架构用，如无 GPU 机器预编译、开发机上 "7.0 8.9"
          同时编入 V100 fp16 路径做旁路测试）
      2. torch.cuda 可用 → 探测所有可见 GPU，并回写 TORCH_CUDA_ARCH_LIST
         （cpp_extension 依赖该环境变量自动生成 gencode，见 get_cuda_arch_flags）。
      3. 无 GPU → 返回空列表。
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        targets = _canonical_arch_list(os.environ["TORCH_CUDA_ARCH_LIST"])
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(targets)
        return _parse_arch_list(";".join(targets))
    if not torch.cuda.is_available():
        return []

    archs: list = []
    caps: set = set()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        if cap not in caps:
            caps.add(cap)
            archs.append(cap[0] * 10 + cap[1])

    # 回写环境变量：arch-specific 变体（sm90a/sm120a 的 wgmma/TMA 等特性仅在
    # 'a' 变体下才会被 cutlass/cute 启用），cpp_extension 由此生成 gencode
    arch_list = [
        f"{a // 10}.{a % 10}" + ("a" if a in (90, 120) else "") for a in archs
    ]
    os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arch_list)
    return archs


def _arch_flags_from(archs: list) -> list:
    """
    生成本次构建的目标架构掩码宏：-DFA_TARGETS=<bits>（唯一的注入宏）。

    位约定（与 csrc/arch_targets.h 标准入口严格一致，修改任一侧须同步）：
      bit0 (0x1) = sm70   （V100，fp16 专用 WMMA 路径）
      bit1 (0x2) = sm8x   （sm_80/86/89/90，cp.async 通用路径）
      bit2 (0x4) = sm120+ （sm_120a，Blackwell consumer TMA 路径）

    host 侧「二进制含哪些 kernel 家族」全部由这一个宏承载（解码见
    arch_targets.h，禁止其它代码自行发明判定宏）；gencode 仍由
    cpp_extension 从 TORCH_CUDA_ARCH_LIST（由 _detect_archs 回写）自动生成，
    不要在此重复传 -gencode（nvcc 会对同一架构编两遍）。
    """
    mask = 0
    for a in set(archs):
        if a == 70:
            mask |= 0x1
        elif 80 <= a < 120:
            mask |= 0x2
        elif a >= 120:
            mask |= 0x4
    return [f"-DFA_TARGETS={mask}"] if mask else []


def get_cuda_arch_flags() -> list:
    """
    自动探测当前可见 GPU 的 compute capability，生成架构裁剪宏。

    只编译本地 GPU 对应架构（gencode 由 cpp_extension 从 TORCH_CUDA_ARCH_LIST
    自动生成），并注入唯一的 FA_TARGETS 位掩码宏供 host 侧做编译期裁剪
    （解码与语义见 csrc/arch_targets.h 标准入口）。无 GPU 时返回空列表。
    """
    return _arch_flags_from(_detect_archs())


def get_build_dir(
    so_name: str,
    env_var: str = "TORCH_EXTENSIONS_DIR",
    fallback_dir: str = "",
) -> str:
    """
    返回编译缓存目录，并确保该目录存在。

    优先级（从高到低）：
      1. env_var 环境变量（默认 TORCH_EXTENSIONS_DIR）
      2. fallback_dir（若非空且可写）
      3. 空串（让 cpp_load 使用 ~/.cache/torch_extensions）

    参数
    ----
    so_name      : str — 仅用于日志，不影响路径
    env_var      : str — 环境变量名
    fallback_dir : str — 回退目录（如 /workdir/xxx_cache/torch_extensions）
    """
    def _try_dir(path: str) -> str:
        try:
            os.makedirs(path, exist_ok=True)
            test_f = os.path.join(path, ".write_test")
            with open(test_f, "w") as _f:
                _f.write("")
            os.remove(test_f)
            return path
        except OSError:
            return ""

    # 1. 环境变量
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        try:
            os.makedirs(explicit, exist_ok=True)
        except OSError:
            pass
        return explicit

    # 2. fallback_dir
    if fallback_dir:
        d = _try_dir(fallback_dir)
        if d:
            os.environ[env_var] = d
            return d

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CustomOps — 通用算子库基类
# ─────────────────────────────────────────────────────────────────────────────

class CustomOps:
    """
    通用 CUDA 算子插件库基类。

    子类只需覆盖类属性和两个方法即可获得完整的加载/分发能力。

    必须覆盖
    --------
    namespace  : str — torch.ops 命名空间（如 "my_ops"）
    so_name    : str — 编译产物名（如 "my_ops_kernel"，不含 .so 后缀）
    get_sources()       → list[str]  — 返回 .cu/.cpp 源文件列表
    get_include_dirs()  → list[str]  — 返回额外头文件目录列表

    可选覆盖
    --------
    required_ops : list[str] — 必需算子名，缺失则 load() 失败
    optional_ops : list[str] — 可选算子名，缺失仅打印警告
    build_dir_fallback : str — 编译缓存回退目录
    devtoolset_path : str    — GCC >= 9 所在目录

    示例
    ----
    见 custom_ops/example/__init__.py
    """

    # ── 子类必须覆盖 ──────────────────────────────────────────────────────────
    namespace  : str = "custom_ops"
    so_name    : str = "custom_ops_kernel"

    # ── 子类可选覆盖 ──────────────────────────────────────────────────────────
    required_ops: list  = []
    optional_ops: list  = []
    build_dir_fallback: str = ""
    devtoolset_path: str = "/opt/rh/devtoolset-10/root/usr/bin"
    # Explicit schema declarations are required for exporting deployment metadata.
    # Keys are local operator names (an overload may use 'name.overload').
    op_schemas: dict = {}
    fake_ops: list = []
    # Native test-only schemas: not requirements, but incompatible with an
    # unmarked production artifact even if allow_testing=True was requested.
    testing_ops: list = []
    build_version: str = "1"
    # Optional source-independent deployment contract (None = infer on build host).
    requires_cuda: Optional[bool] = None

    # ── 实例状态（不覆盖）────────────────────────────────────────────────────
    def __init__(self, *, allow_jit: bool = True) -> None:
        self._loaded: bool = False
        self._ext = None
        self._load_error: Optional[str] = None
        self._library_path: Optional[Path] = None
        self._metadata_path: Optional[Path] = None
        self._fake_ops_registered: set = set()
        self._allow_jit = True
        self.configure(allow_jit=allow_jit)

    # ── 子类必须覆盖 ──────────────────────────────────────────────────────────

    def get_sources(self) -> list:
        """返回需要编译的 .cu/.cpp 源文件路径列表（子类必须覆盖）。"""
        raise NotImplementedError(
            f"{type(self).__name__}.get_sources() must be implemented."
        )

    def get_include_dirs(self) -> list:
        """返回额外的头文件 include 目录列表（子类必须覆盖）。"""
        raise NotImplementedError(
            f"{type(self).__name__}.get_include_dirs() must be implemented."
        )

    @property
    def allow_jit(self) -> bool:
        """Whether load() may consult the development cache or compile."""
        return self._allow_jit

    @allow_jit.setter
    def allow_jit(self, value: bool) -> None:
        self.configure(allow_jit=value)

    def configure(self, *, allow_jit: bool) -> "CustomOps":
        """Explicit loading policy; no environment variable enables JIT."""
        if not isinstance(allow_jit, bool):
            raise TypeError("allow_jit must be a bool")
        with _LIBRARY_LOCK:
            self._allow_jit = allow_jit
        return self

    def disable_jit(self) -> "CustomOps":
        """Require explicit load_precompiled(), even if a JIT cache exists."""
        return self.configure(allow_jit=False)

    @property
    def library_path(self) -> Optional[Path]:
        """The actual loaded library, or None before a successful load."""
        return self._library_path if self._loaded else None

    @property
    def metadata_path(self) -> Optional[Path]:
        """The validated sidecar, or None before a successful load."""
        return self._metadata_path if self._loaded else None

    # The same inputs and flags drive both development and offline builds.
    def get_extra_cflags(self) -> list:
        return ["-O3", "-ffast-math"]

    def get_extra_cuda_cflags(self) -> list:
        return ["-O3", "--use_fast_math", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]

    def get_extra_ldflags(self) -> list:
        return []

    def _schema_contract(self, *, require_complete: bool = False) -> dict:
        schemas = {}
        for name, declaration in self.op_schemas.items():
            qualified = f"{self.namespace}::{name}"
            if not declaration.startswith(f"{self.namespace}::"):
                declaration = f"{self.namespace}::{declaration}"
            parsed = torch._C.parse_schema(declaration)
            actual = parsed.name + (f".{parsed.overload_name}" if parsed.overload_name else "")
            if actual != qualified:
                raise CustomOpsError(f"Schema name mismatch for {qualified}: {actual}")
            schemas[name] = str(parsed)
        if require_complete:
            missing = set(self.required_ops).difference(schemas)
            missing_fake = set(self.fake_ops).difference(schemas)
            if not self.required_ops or missing or missing_fake:
                raise CustomOpsError(
                    "Offline builds require required_ops and explicit op_schemas; "
                    f"missing required schemas: {sorted(missing)}, "
                    f"missing fake schemas: {sorted(missing_fake)}")
        return schemas

    @staticmethod
    def _compile_flags(flags, name: str) -> list:
        if not isinstance(flags, (list, tuple)) or any(
            not isinstance(flag, str) or not flag.strip() or "\0" in flag for flag in flags
        ):
            raise TypeError(f"{name} must be a list or tuple of nonempty strings")
        return list(flags)

    @staticmethod
    def _testing_flags(cflags: list, cuda_cflags: list) -> bool:
        # Conservatively classify any reference to the guarded native macro as
        # testing, including split '-D', 'MACRO=1' flags and subclass overrides.
        return any("CUSTOM_OPS_OFFLOAD_TESTING" in flag
                   for flag in [*cflags, *cuda_cflags])

    def _build_inputs(
        self, cuda_arch_list: Optional[str] = None, *,
        extra_cflags=(), extra_cuda_cflags=(),
    ) -> dict:
        cflags = (self._compile_flags(self.get_extra_cflags(), "get_extra_cflags()")
                  + self._compile_flags(extra_cflags, "extra_cflags"))
        cuda_cflags = (
            self._compile_flags(self.get_extra_cuda_cflags(), "get_extra_cuda_cflags()")
            + self._compile_flags(extra_cuda_cflags, "extra_cuda_cflags")
        )
        sources = [str(Path(source).expanduser().resolve(strict=True))
                   for source in self.get_sources()]
        includes = [str(Path(path).expanduser().resolve(strict=True))
                    for path in self.get_include_dirs()]
        if not sources or any(not Path(path).is_file() for path in sources):
            raise CustomOpsError("get_sources() must return existing source files")
        if any(not Path(path).is_dir() for path in includes):
            raise CustomOpsError("get_include_dirs() must return existing directories")
        detected_cuda = any(Path(source).suffix in (".cu", ".cuh") for source in sources)
        with_cuda = detected_cuda if self.requires_cuda is None else self.requires_cuda
        if detected_cuda and not with_cuda:
            raise CustomOpsError("CUDA source files cannot be built with requires_cuda=False")
        targets = []
        if with_cuda:
            if torch.version.cuda is None or getattr(torch.version, "hip", None):
                raise CustomOpsError("CUDA extensions require an NVIDIA CUDA PyTorch build")
            if cuda_arch_list is not None and not isinstance(cuda_arch_list, str):
                raise TypeError("cuda_arch_list must be a string, e.g. '8.9;12.0a'")
            requested = (cuda_arch_list if cuda_arch_list is not None
                         else os.environ.get("TORCH_CUDA_ARCH_LIST", ""))
            if not requested and torch.cuda.is_available():
                capabilities = {torch.cuda.get_device_capability(device)
                                for device in range(torch.cuda.device_count())}
                requested = ";".join(f"{major}.{minor}" + (
                    "a" if (major, minor) in ((9, 0), (12, 0)) else "")
                    for major, minor in sorted(capabilities))
            targets = _canonical_arch_list(requested)
            if not targets:
                raise CustomOpsError(
                    "No CUDA build targets detected. Pass cuda_arch_list or set "
                    "TORCH_CUDA_ARCH_LIST for an offline build without a GPU.")
        elif cuda_arch_list is not None:
            raise CustomOpsError("cuda_arch_list was specified for a C++-only extension")
        if not with_cuda and extra_cuda_cflags:
            raise CustomOpsError("extra_cuda_cflags was specified for a C++-only extension")
        return {
            "sources": sources,
            "include_dirs": includes,
            "extra_cflags": cflags,
            "extra_cuda_cflags": [*_arch_flags_from(_parse_arch_list(";".join(targets))),
                                  *cuda_cflags] if with_cuda else [],
            "extra_ldflags": list(self.get_extra_ldflags()),
            "with_cuda": with_cuda,
            "cuda_arch_list": targets,
        }

    def _source_fingerprint(self, inputs: dict) -> str:
        # Hash content, not mtimes; include custom headers (including thirdparty)
        # so offline exports and cache invalidation also cover header-only edits.
        paths = {Path(source) for source in inputs["sources"]}
        roots = {Path(source).parent for source in inputs["sources"]}
        roots.update(Path(path) for path in inputs["include_dirs"])
        suffixes = {".h", ".hh", ".hpp", ".cuh", ".inc", ".inl", ".tpp"}
        for root in roots:
            paths.update(path.resolve() for path in root.rglob("*")
                         if path.suffix in suffixes and path.is_file())
        digest = hashlib.sha256()
        for path in sorted(paths):
            digest.update(str(path).encode("utf-8") + b"\0")
            digest.update(_file_sha256(path).encode("ascii") + b"\0")
        return digest.hexdigest()

    def _toolchain(self, with_cuda: bool) -> dict:
        def describe(command: str) -> dict:
            output = subprocess.check_output(
                [*shlex.split(command), "--version"], text=True,
                stderr=subprocess.STDOUT, timeout=15,
            ).strip()
            return {"command": command, "version": output}

        result = {"cxx": describe(os.environ.get("CXX", "c++")),
                  "cc": describe(os.environ["CC"]) if os.environ.get("CC") else None,
                  "python_version": platform.python_version()}
        if with_cuda:
            from torch.utils.cpp_extension import CUDA_HOME
            if not CUDA_HOME:
                raise CustomOpsError("CUDA_HOME / nvcc is required only on the build host")
            result["nvcc"] = describe(shlex.quote(str(Path(CUDA_HOME) / "bin" / "nvcc")))
            match = re.search(r"release\s+([0-9]+\.[0-9]+)", result["nvcc"]["version"])
            if not match:
                raise CustomOpsError("Cannot determine CUDA toolkit version from nvcc")
            result["cuda_toolkit_version"] = match[1]
        return result

    def _compile_library(self, directory: Path, inputs: dict, *, verbose: bool) -> Path:
        # Unlike cpp_extension.load(), BuildExtension never dlopens its output.
        # This permits exporting another target in an already initialized process.
        from setuptools import Distribution
        from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

        extension_factory = CUDAExtension if inputs["with_cuda"] else CppExtension
        extension = extension_factory(
            self.so_name, sources=list(inputs["sources"]),
            include_dirs=list(inputs["include_dirs"]),
            extra_compile_args={"cxx": list(inputs["extra_cflags"]),
                                "nvcc": list(inputs["extra_cuda_cflags"])},
            extra_link_args=list(inputs["extra_ldflags"]),
        )
        distribution = Distribution({"name": self.so_name, "ext_modules": [extension]})
        command = BuildExtension(distribution, no_python_abi_suffix=True)
        command.build_lib = str(directory)
        command.build_temp = str(directory / "objects")
        command.inplace = False
        command.verbose = int(verbose)
        command.ensure_finalized()
        command.run()
        library = Path(command.get_ext_fullpath(self.so_name)).resolve(strict=True)
        if not library.is_file():
            raise CustomOpsError(f"Build did not produce a shared library: {library}")
        return library

    def _build_metadata(self, inputs: dict, source_fingerprint: str, toolchain: dict) -> dict:
        schemas = self._schema_contract(require_complete=True)
        metadata = {
            "format_version": _METADATA_VERSION,
            "producer": "custom_ops",
            "namespace": self.namespace,
            "so_name": self.so_name,
            "build_version": str(self.build_version),
            "testing": self._testing_flags(inputs["extra_cflags"], inputs["extra_cuda_cflags"]),
            "compatibility": _runtime_compatibility(),
            "with_cuda": inputs["with_cuda"],
            "cuda_arch_list": inputs["cuda_arch_list"],
            "required_ops": sorted(self.required_ops),
            "optional_ops": sorted(self.optional_ops),
            "fake_ops": sorted(self.fake_ops),
            "schemas": schemas,
            "schema_fingerprint": _json_hash(schemas),
            "source_fingerprint": source_fingerprint,
            "build": {**inputs, "toolchain": toolchain},
        }
        metadata["build_fingerprint"] = _metadata_fingerprint(metadata)
        return metadata

    def build(
        self, output_directory: str | os.PathLike, *,
        cuda_arch_list: Optional[str] = None, verbose: bool = False,
        extra_cflags: list[str] | tuple[str, ...] = (),
        extra_cuda_cflags: list[str] | tuple[str, ...] = (),
    ) -> Path:
        """Compile/export <so_name>.so and <so_name>.so.json, without loading.

        Uses the same sources/includes/flags as development JIT. Explicit build
        is allowed even on an allow_jit=False instance. For a GPU-less build host,
        pass cuda_arch_list='8.9' (or a full target such as '12.0a+PTX'). Both
        files must be deployed together; the output directory need not be the
        development cache. Metadata records known compatibility, not arbitrary
        libstdc++/driver/LibTorch ABI portability or artifact authenticity.
        Extra flags explicitly augment subclass defaults for this build only and
        are checksummed in metadata. CUSTOM_OPS_OFFLOAD_TESTING flags mark the
        artifact as testing; loading it requires explicit allow_testing=True.
        """
        with _LIBRARY_LOCK:
            try:
                self._schema_contract(require_complete=True)
                setup_compiler(self.devtoolset_path)
                inputs = self._build_inputs(
                    cuda_arch_list, extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                )
                fingerprint = self._source_fingerprint(inputs)
                toolchain = self._toolchain(inputs["with_cuda"])
                metadata = self._build_metadata(inputs, fingerprint, toolchain)
                output = Path(output_directory).expanduser().resolve()
                output.mkdir(parents=True, exist_ok=True)
                library = output / f"{self.so_name}.so"
                sidecar = library.with_suffix(library.suffix + ".json")
                with (output / f"{self.so_name}.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:
                        with tempfile.TemporaryDirectory(
                            prefix=f".{self.so_name}-build-", dir=output,
                        ) as work, _arch_environment(inputs["cuda_arch_list"]):
                            product = self._compile_library(Path(work), inputs, verbose=verbose)
                            if self._source_fingerprint(inputs) != fingerprint:
                                raise CustomOpsError("Sources changed during build; nothing exported")
                            identity, _ = _library_identity(product)
                            metadata["library"] = {
                                "filename": library.name, "sha256": identity[2],
                                "size": product.stat().st_size,
                            }
                            pending = Path(work) / "metadata.json"
                            pending.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                                               encoding="utf-8")
                            # Do not replace a file already mapped into this process.
                            if library.exists() and any(
                                record["identity"][:2] == _file_stat(library)[:2]
                                for record in _LIBRARIES.values()
                            ):
                                raise CustomOpsError(
                                    f"Cannot overwrite an already loaded library: {library}. "
                                    "Build into another directory or use a fresh process.")
                            os.replace(product, library)
                            try:
                                os.replace(pending, sidecar)
                            except OSError as error:
                                raise CustomOpsError(
                                    "Offline build export incomplete: library was replaced but "
                                    "metadata publication failed; the pair is not atomic. "
                                    "Do not deploy these files; rebuild before loading. "
                                    f"{error}") from error
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                return library
            except Exception as error:
                if isinstance(error, (CustomOpsError, TypeError)):
                    raise
                raise CustomOpsError(f"[{type(self).__name__}] Offline build failed: {error}") from error

    # ── 加载管理 ──────────────────────────────────────────────────────────────

    def _validate_metadata(
        self, library: Path, sidecar: Path, *, allow_testing: bool = False,
    ) -> tuple:
        with sidecar.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise CustomOpsError("Precompiled metadata must be a JSON object")
        expected = {
            "format_version": _METADATA_VERSION, "producer": "custom_ops",
            "namespace": self.namespace, "so_name": self.so_name,
            "build_version": str(self.build_version),
        }
        for key, value in expected.items():
            if metadata.get(key) != value or type(metadata.get(key)) is not type(value):
                raise CustomOpsError(
                    f"Precompiled {key} mismatch: built={metadata.get(key)!r}, expected={value!r}")
        compatibility = metadata.get("compatibility")
        if not isinstance(compatibility, dict):
            raise CustomOpsError("Precompiled compatibility information is missing")
        for key, value in _runtime_compatibility().items():
            if key not in compatibility or compatibility[key] != value or (
                type(compatibility[key]) is not type(value)
            ):
                raise CustomOpsError(
                    f"Precompiled compatibility mismatch ({key}): "
                    f"built={compatibility.get(key)!r}, runtime={value!r}")
        for key, ops in (("required_ops", self.required_ops),
                         ("optional_ops", self.optional_ops), ("fake_ops", self.fake_ops)):
            values = metadata.get(key)
            if not isinstance(values, list) or any(not isinstance(op, str) for op in values):
                raise CustomOpsError(f"Precompiled {key} must be a list of operator names")
            if sorted(values) != sorted(ops):
                raise CustomOpsError(
                    f"Precompiled {key} mismatch: built={values}, expected={sorted(ops)}")
        schemas = self._schema_contract(require_complete=True)
        if metadata.get("schemas") != schemas:
            raise CustomOpsError("Precompiled schema contract does not match this Python wrapper")
        if metadata.get("schema_fingerprint") != _json_hash(schemas):
            raise CustomOpsError("Precompiled schema fingerprint is invalid")
        for key in ("source_fingerprint", "build_fingerprint"):
            if not isinstance(metadata.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", metadata[key],
            ):
                raise CustomOpsError(f"Precompiled {key} is missing or invalid")
        if metadata["build_fingerprint"] != _metadata_fingerprint(metadata):
            raise CustomOpsError("Precompiled build fingerprint is invalid")
        with_cuda = metadata.get("with_cuda")
        targets = metadata.get("cuda_arch_list")
        if not isinstance(with_cuda, bool) or not isinstance(targets, list) or any(
            not isinstance(target, str) for target in targets
        ):
            raise CustomOpsError("Precompiled CUDA target information is invalid")
        if self.requires_cuda is not None and with_cuda != self.requires_cuda:
            raise CustomOpsError("Precompiled CUDA requirement does not match this Python wrapper")
        canonical = _canonical_arch_list(";".join(targets))
        if targets != canonical or bool(targets) != with_cuda:
            raise CustomOpsError("Precompiled CUDA targets are incomplete or noncanonical")
        build = metadata.get("build")
        if not isinstance(build, dict) or build.get("with_cuda") is not with_cuda or (
            build.get("cuda_arch_list") != targets
        ):
            raise CustomOpsError("Precompiled build inputs disagree with target information")
        cflags = self._compile_flags(build.get("extra_cflags"), "metadata extra_cflags")
        cuda_cflags = self._compile_flags(
            build.get("extra_cuda_cflags"), "metadata extra_cuda_cflags",
        )
        testing = metadata.get("testing")
        if type(testing) is not bool or testing != self._testing_flags(cflags, cuda_cflags):
            raise CustomOpsError("Precompiled testing marker disagrees with compile flags")
        if testing and not allow_testing:
            raise CustomOpsError("Test-hook library requires explicit allow_testing=True")
        toolchain = build.get("toolchain")
        if not isinstance(toolchain, dict) or not isinstance(toolchain.get("cxx"), dict):
            raise CustomOpsError("Precompiled compiler information is missing")
        if not toolchain["cxx"].get("version"):
            raise CustomOpsError("Precompiled compiler version is missing")
        if with_cuda:
            if torch.version.cuda is None or getattr(torch.version, "hip", None):
                raise CustomOpsError("A CUDA library requires NVIDIA CUDA PyTorch")
            toolkit = toolchain.get("cuda_toolkit_version", "")
            if not isinstance(toolkit, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", toolkit):
                raise CustomOpsError("Precompiled CUDA toolkit version is missing")
            if toolkit.split(".")[0] != torch.version.cuda.split(".")[0]:
                raise CustomOpsError("Precompiled CUDA toolkit major version is incompatible")
            requested = _canonical_arch_list(os.environ.get("TORCH_CUDA_ARCH_LIST", ""))
            missing = set(requested).difference(targets)
            if missing:
                raise CustomOpsError(
                    f"Precompiled full CUDA target mismatch: requested={sorted(missing)}, "
                    f"built={targets} (ISA suffix and +PTX are significant)")
            # Do not assume PTX forward compatibility or same-major SM compatibility.
            # GPU capability cannot report ISA suffixes; explicit target configuration
            # above is checked in full, while visible devices are checked by exact SM.
            if torch.cuda.is_available():
                built_sms = set(_parse_arch_list(";".join(targets)))
                visible = {major * 10 + minor for major, minor in (
                    torch.cuda.get_device_capability(device)
                    for device in range(torch.cuda.device_count()))}
                if not visible.issubset(built_sms):
                    raise CustomOpsError(
                        f"Precompiled SM mismatch: visible={sorted(visible)}, built={sorted(built_sms)}")
        binary = metadata.get("library")
        if not isinstance(binary, dict) or not isinstance(binary.get("filename"), str) or (
            Path(binary["filename"]).name != binary["filename"]
        ):
            raise CustomOpsError("Precompiled library identity information is missing")
        if type(binary.get("size")) is not int or binary["size"] <= 0:
            raise CustomOpsError("Precompiled library size is invalid")
        identity, stat = _library_identity(library)
        if binary.get("sha256") != identity[2] or binary["size"] != stat[2]:
            raise CustomOpsError(f"Precompiled library hash/size mismatch: {library}")
        return metadata, identity, stat

    def _operator_names(self) -> set:
        return {f"{self.namespace}::{name}" for name in (
            set(self.required_ops) | set(self.optional_ops) | set(self.op_schemas))}

    def _check_library_conflicts(self, identity: Optional[tuple], names: set) -> None:
        for record in _LIBRARIES.values():
            if identity is not None and record["identity"][:2] == identity[:2] and (
                record["identity"] != identity
            ):
                raise CustomOpsError("An already loaded library file was modified; use a fresh process")
        if identity in _LIBRARIES and _LIBRARIES[identity]["error"] is not None:
            raise CustomOpsError(
                "This library previously failed after loading started; use a fresh process. "
                + _LIBRARIES[identity]["error"])
        for name in names:
            owner = _OP_OWNERS.get(name)
            if owner is not None and owner != identity:
                record = _LIBRARIES[owner]
                raise CustomOpsError(
                    f"Dispatcher library conflict for {name}: already owned by {record['path']}. "
                    "Loading another copy is unsafe; use a fresh process.")
            if owner is None and _has_schema(name):
                raise CustomOpsError(
                    f"Dispatcher operator {name} is already registered by an untracked library. "
                    "Cannot safely dlopen another copy; use a fresh process and explicit loading.")

    def _dlopen_library(
        self, library: Path, identity: tuple, stat: tuple, metadata: Optional[dict] = None,
    ) -> dict:
        names = self._operator_names()
        self._check_library_conflicts(identity, names)
        record = _LIBRARIES.get(identity)
        if record is not None:
            if record["namespace"] != self.namespace or record["so_name"] != self.so_name:
                raise CustomOpsError("The same library is already associated with another extension")
            fingerprint = metadata["build_fingerprint"] if metadata else None
            if fingerprint and record["metadata_fingerprint"] not in (None, fingerprint):
                raise CustomOpsError("The same library was supplied with conflicting build metadata")
            return record
        if _file_stat(library) != stat:
            raise CustomOpsError(f"Library changed before dlopen: {library}")
        record = {
            "identity": identity, "path": library, "namespace": self.namespace,
            "so_name": self.so_name, "names": set(names), "ready": False, "error": None,
            "metadata_fingerprint": metadata["build_fingerprint"] if metadata else None,
        }
        # Reserve *before* dlopen, including missing required names. Even failed
        # native initialization may leave registrations that must never be replaced.
        _LIBRARIES[identity] = record
        for name in names:
            _OP_OWNERS[name] = identity
        before = set(torch._C._dispatch_get_all_op_names())
        try:
            torch.ops.load_library(str(library))
            if _file_stat(library) != stat:
                raise CustomOpsError(f"Library changed during dlopen: {library}")
        except Exception as error:
            record["error"] = str(error)
            raise
        finally:
            added = set(torch._C._dispatch_get_all_op_names()).difference(before)
            record["names"].update(added)
            for name in added:
                _OP_OWNERS[name] = identity
        return record

    def _finish_loading(self, record: dict, metadata: Optional[dict] = None) -> None:
        self._ext = True
        self._library_path = record["path"]
        if metadata is not None and not metadata["testing"]:
            hooks = [name for name in self.testing_ops
                     if _has_schema(f"{self.namespace}::{name}")]
            if hooks:
                raise CustomOpsError(
                    f"Unmarked production library exposes native test-hook schemas: {hooks}. "
                    "Use a correctly marked testing artifact and explicit allow_testing=True; "
                    "the loaded dispatcher cannot be rolled back, so use a fresh process.")
        self._verify_ops()
        self._fake_ops_registered.clear()
        self.register_fake_impls()
        for name in set(self.fake_ops) | self._fake_ops_registered:
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                f"{self.namespace}::{name}", "Meta",
            ):
                raise CustomOpsError(f"Fake/Meta implementation missing for {self.namespace}::{name}")
        if metadata:
            record["metadata_fingerprint"] = metadata["build_fingerprint"]
        record["ready"] = True
        self._load_error = None
        # Fake registration and verification must succeed *before* publishing availability.
        self._loaded = True

    def load_precompiled(
        self, library_path: str | os.PathLike, metadata_path: Optional[str | os.PathLike] = None,
        *, allow_testing: bool = False,
    ) -> "CustomOps":
        """Validate/load a deployment library; always disables implicit JIT.

        The default sidecar is '<library_path>.json'. Missing/incompatible metadata,
        schema/fake failures and dispatcher identity conflicts raise CustomOpsError;
        none fall back to compilation. Validation does not read C++/CUDA sources,
        import cpp_extension or require a compiler/Ninja/CUDA toolkit. Libraries
        remain registered for the process lifetime. Only the same inode AND bytes
        may be reattached; a copied .so with the same operators is a conflict.
        Test-hook builds are rejected by default, even if already loaded elsewhere;
        allow_testing=True explicitly authorizes them without enabling JIT.
        """
        with _LIBRARY_LOCK:
            self._allow_jit = False
            self._loaded = False
            record = None
            try:
                if not isinstance(allow_testing, bool):
                    raise TypeError("allow_testing must be a bool")
                library = Path(library_path).expanduser().resolve(strict=True)
                if not library.is_file():
                    raise CustomOpsError(f"Precompiled library is not a file: {library}")
                sidecar = (Path(metadata_path).expanduser() if metadata_path is not None
                           else Path(str(Path(library_path).expanduser()) + ".json"))
                sidecar = sidecar.resolve(strict=True)
                metadata, identity, stat = self._validate_metadata(
                    library, sidecar, allow_testing=allow_testing,
                )
                record = self._dlopen_library(library, identity, stat, metadata)
                self._finish_loading(record, metadata)
                self._metadata_path = sidecar
            except Exception as error:
                self._loaded = False
                self._load_error = str(error)
                if record is not None:
                    record["ready"] = False
                    record["error"] = str(error)
                raise CustomOpsError(
                    f"[{type(self).__name__}] Precompiled load failed "
                    f"(JIT disabled): {error}") from error
        return self

    def load(self) -> "CustomOps":
        """Development-only lazy compile/cache load (idempotent).

        Development failures retain the historical warning/is_available=False
        behavior. In strict mode an unloaded instance raises without inspecting
        the cache, sources, compiler or CUDA targets. Use load_precompiled first.
        """
        with _LIBRARY_LOCK:
            if self._loaded:
                return self
            if not self._allow_jit:
                self._load_error = (
                    f"JIT is disabled for {type(self).__name__}; explicitly call "
                    "load_precompiled(library_path, metadata_path=None) before load().")
                raise CustomOpsError(self._load_error)
            record = None
            try:
                record = self._load_extension()
                self._finish_loading(record)
            except Exception as error:
                self._loaded = False
                self._load_error = str(error)
                if record is not None:
                    record["ready"] = False
                    record["error"] = str(error)
                warnings.warn(
                    f"[{type(self).__name__}] Failed to load extension "
                    f"(namespace={self.namespace}), is_available() == False. Error: {error}")
        return self

    def _load_extension(self) -> dict:
        """Development cache only; never entered by strict/precompiled loading."""
        started = time.monotonic()
        base = get_build_dir(so_name=self.so_name, fallback_dir=self.build_dir_fallback)
        if not base:
            from torch.utils.cpp_extension import get_default_build_root
            base = get_default_build_root()
        directory = Path(base).expanduser().resolve() / self.so_name
        directory.mkdir(parents=True, exist_ok=True)
        library = directory / f"{self.so_name}.so"
        stamp = directory / ".build_stamp"
        inputs = self._build_inputs()
        source_fingerprint = self._source_fingerprint(inputs)
        signature = _json_hash({
            **inputs, "source_fingerprint": source_fingerprint,
            "compatibility": _runtime_compatibility(), "schemas": self._schema_contract(),
            "build_version": str(self.build_version),
        })
        with (directory / f"{self.so_name}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                cached = None
                if library.is_file() and stamp.is_file():
                    try:
                        cached = json.loads(stamp.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        cached = None
                if isinstance(cached, dict) and cached.get("fingerprint") == signature:
                    identity, stat = _library_identity(library)
                    if cached.get("library_sha256") == identity[2]:
                        return self._dlopen_library(library, identity, stat)
                # A rebuild cannot replace registrations in the same process.
                self._check_library_conflicts(None, self._operator_names())
                setup_compiler(self.devtoolset_path)
                warnings.warn(
                    f"[{type(self).__name__}] Compiling development extension "
                    f"{self.so_name} (cache={directory}).")
                with tempfile.TemporaryDirectory(
                    prefix=".jit-build-", dir=directory,
                ) as work, _arch_environment(inputs["cuda_arch_list"]):
                    product = self._compile_library(Path(work), inputs, verbose=False)
                    if self._source_fingerprint(inputs) != source_fingerprint:
                        raise CustomOpsError("Sources changed during JIT compilation")
                    pending = Path(work) / "stamp.json"
                    pending.write_text(json.dumps({
                        "fingerprint": signature, "library_sha256": _file_sha256(product),
                    }), encoding="utf-8")
                    os.replace(product, library)
                    os.replace(pending, stamp)
                identity, stat = _library_identity(library)
                record = self._dlopen_library(library, identity, stat)
                warnings.warn(
                    f"[{type(self).__name__}] Compiled and loaded in "
                    f"{time.monotonic() - started:.1f}s.")
                return record
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _verify_ops(self) -> None:
        """Verify actual dispatcher schemas, not Python namespace placeholders."""
        missing = [name for name in self.required_ops
                   if not _has_schema(f"{self.namespace}::{name}")]
        if missing:
            raise CustomOpsError(
                f"Required ops not found in torch.ops.{self.namespace}: {missing}.")
        for name, expected in self._schema_contract().items():
            qualified = f"{self.namespace}::{name}"
            if _has_schema(qualified) and _schema(qualified) != expected:
                raise CustomOpsError(
                    f"Loaded schema mismatch for {qualified}: "
                    f"got={_schema(qualified)!r}, expected={expected!r}")
        for name in self.optional_ops:
            if not _has_schema(f"{self.namespace}::{name}"):
                warnings.warn(
                    f"[{type(self).__name__}] Optional op '{self.namespace}::{name}' not found.")

    # ── AOTI / torch.compile 兼容 ─────────────────────────────────────────────

    def register_fake_impls(self) -> None:
        """
        为已注册的算子注册 fake（meta）实现，供 torch.compile / torch.export / AOTI 做 shape 推断。

        必要性
        --------
        torch.compile / AOTI 在 tracing 阶段使用 FakeTensor（meta 设备）模拟执行：
          - 若有 fake 实现 → tracing 成功，可导出为 AOT Inductor
          - 若没有 fake 实现 → tracing 会尝试运行实际 CUDA 算子，必然失败

        使用方式
        --------
        子类覆盖此方法，在内部调用 ``register_fake_for`` 注册每个算子的 shape 推断函数。
        本方法在 ``load()`` 成功后自动被调用，无需手动触发。

        示例
        ------
            def register_fake_impls(self):
                # out shape 与 q 相同（B, H, Sq, d）
                self.register_fake_for(
                    "my_op",
                    lambda q, k, v, mask: torch.empty_like(q),
                )
        """
        # 默认空实现（子类不覆盖时不影响加载）
        pass

    def register_fake_for(
        self,
        op_name: str,
        fake_fn,
    ) -> None:
        """
        为单个算子注册 fake 实现的辅助方法。

        参数
        ------
        op_name : str
            算子名（不含命名空间前缀），如 "mha_fwd_with_mask"。
            会自动拼接为 "<namespace>::<op_name>" 格式。
        fake_fn : callable
            湿运行级实现，参数与实际算子相同，返回与输出 shape/dtype 匹配的空 tensor。
            示例：``lambda q, k, v, mask: torch.empty_like(q)``

        Only an existing real Meta kernel makes registration idempotent. Missing
        schemas, unsupported APIs and all other registration failures propagate;
        they must not advertise an export-capable loaded extension.
        """
        qualified = f"{self.namespace}::{op_name}"
        with _LIBRARY_LOCK:
            _schema(qualified)  # Missing schemas are errors, never swallowed.
            has_meta = torch._C._dispatch_has_kernel_for_dispatch_key
            if not has_meta(qualified, "Meta"):
                register = getattr(torch.library, "register_fake", None)
                if register is None:
                    register = getattr(torch.library, "impl_abstract", None)
                if register is None:
                    raise CustomOpsError("This PyTorch version cannot register fake implementations")
                register(qualified)(fake_fn)
            if not has_meta(qualified, "Meta"):
                raise CustomOpsError(f"Fake registration did not create a Meta kernel: {qualified}")
            self._fake_ops_registered.add(op_name)

    # ── 公共 API ──────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """返回算子库是否已成功加载。False 时应退化到 Python 回退实现。"""
        return self._loaded

    def load_error(self) -> Optional[str]:
        """返回加载失败的错误信息，成功时返回 None。"""
        return self._load_error

    def __getattr__(self, name: str):
        """
        将 ops.<name>(...) 自动路由到 torch.ops.<namespace>.<name>(...)。

        新增算子后无需修改此类，注册到 torch.ops.<namespace> 即自动可用。
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if not self._loaded:
            raise RuntimeError(
                f"[{type(self).__name__}] ops.{name}() called but extension not loaded. "
                f"Check is_available(). Error: {self._load_error}"
            )
        ns = getattr(torch.ops, self.namespace, None)
        if ns is None:
            raise AttributeError(f"torch.ops.{self.namespace} not found.")
        op = getattr(ns, name, None)
        if op is None:
            raise AttributeError(
                f"No op '{name}' in torch.ops.{self.namespace}."
            )
        return op

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else f"not loaded ({self._load_error})"
        ns = getattr(torch.ops, self.namespace, None)
        ops_list = list(dir(ns)) if (self._loaded and ns) else []
        return (
            f"{type(self).__name__}("
            f"namespace={self.namespace!r}, "
            f"status={status}, "
            f"ops={ops_list})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RecsysOps — 推荐系统核心算子库（分组懒加载门面）
#
# 按算子分组拆分为独立 .so（fa / mixed_gemm / fuse_moe / swiglu），首次访问某算子
# 时才 JIT 编译对应分组——只测/只用一个算子时不再全量编译。
#
# 用法：
#   from custom_ops import ops            # import 零编译
#   out = ops.mha_fwd_with_mask(q,k,v,mask)  # 首次调用才编译 FA 分组
#   y   = ops.fuse_moe(...)                  # 首次调用才编译 MoE 分组
#   act = ops.swiglu(x, w)                  # 首次调用才编译 SwiGLU 分组
#   ops.ensure_loaded()                       # 显式全量加载（旧行为）
# ─────────────────────────────────────────────────────────────────────────────

from custom_ops.recsys import RecsysOps  # noqa: E402  (避免循环导入，置于类定义之后)
from custom_ops.recsys import split_mixed_precision_weight  # noqa: E402

#: 推荐系统算子库的包级全局单例（懒加载：首次访问某算子时才编译对应分组）。
#:
#: 用法::
#:
#:     from custom_ops import ops
#:     out  = ops.mha_fwd_with_mask(q, k, v, mask)
#:     y    = ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu")
#:     z    = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)
ops: RecsysOps = RecsysOps()
