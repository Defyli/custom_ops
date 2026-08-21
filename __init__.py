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

import fcntl
import os
import time
import warnings
from typing import Optional

import torch


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


def get_cuda_arch_flags() -> list:
    """
    自动探测当前可见 GPU 的 compute capability，生成 nvcc -gencode 参数列表。

    优先级：
      1. TORCH_CUDA_ARCH_LIST 已设置 → 返回空列表，让 PyTorch 自行解析。
      2. torch.cuda 可用 → 探测所有可见 GPU。
      3. 无 GPU → 返回空列表。
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    if not torch.cuda.is_available():
        return []

    seen: set = set()
    flags = []
    arch_list = []
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        if cap not in seen:
            seen.add(cap)
            sm = f"{cap[0]}{cap[1]}"
            # arch-specific 变体：sm90a（wgmma）、sm120a（Blackwell consumer 的
            # TMA/STSM 等特性在该变体下才会被 cutlass/cute 启用）
            suffix = "a" if cap in ((9, 0), (12, 0)) else ""
            flags += ["-gencode", f"arch=compute_{sm}{suffix},code=sm_{sm}{suffix}"]
            arch_list.append(f"{cap[0]}.{cap[1]}{suffix}")

    if arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arch_list)

    return flags


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

    # ── 实例状态（不覆盖）────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._loaded: bool = False
        self._ext = None
        self._load_error: Optional[str] = None

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

    # ── 加载管理 ──────────────────────────────────────────────────────────────

    def load(self) -> "CustomOps":
        """
        触发 JIT 编译 / 从缓存加载共享库（幂等，可多次调用）。

        加载策略
        --------
        1. 快速路径：.so 已存在 → 直接 dlopen（< 1s）。
        2. 编译路径：.so 不存在 → JIT 编译（首次约 30~60s）。

        加载失败时记录错误，不抛出异常，is_available() 返回 False。

        Returns
        -------
        self，支持链式调用：ops = MyOps().load()
        """
        if self._loaded:
            return self

        setup_compiler(self.devtoolset_path)

        try:
            self._ext = self._load_extension()
            self._verify_ops()
            self._loaded = True
            # 加载成功后自动注册 fake 实现（AOTI / torch.compile 支持）
            self.register_fake_impls()
        except Exception as e:
            self._load_error = str(e)
            warnings.warn(
                f"[{type(self).__name__}] Failed to load CUDA extension "
                f"(namespace={self.namespace}), is_available() == False. "
                f"Error: {e}"
            )

        return self

    def _load_extension(self):
        """内部：执行快速路径或编译路径加载。"""
        t0 = time.monotonic()

        build_dir = get_build_dir(
            so_name=self.so_name,
            fallback_dir=self.build_dir_fallback,
        )
        so_path = os.path.join(build_dir, f"{self.so_name}.so") if build_dir else ""

        # 快速路径：.so 已存在，直接 dlopen
        if so_path and os.path.isfile(so_path):
            warnings.warn(
                f"[{type(self).__name__}] pid={os.getpid()} "
                f"fast-loading {self.so_name}.so via dlopen: {so_path}"
            )
            torch.ops.load_library(so_path)
            warnings.warn(
                f"[{type(self).__name__}] pid={os.getpid()} "
                f"loaded in {time.monotonic() - t0:.2f}s (fast path)."
            )
            return True

        # 编译路径
        from torch.utils.cpp_extension import load as cpp_load

        arch_flags   = get_cuda_arch_flags()
        inc_dirs     = self.get_include_dirs()
        include_args = [f"-I{d}" for d in inc_dirs]

        warnings.warn(
            f"[{type(self).__name__}] pid={os.getpid()} "
            f"{self.so_name}.so not found, compiling via JIT "
            f"(build_dir={build_dir or '~/.cache/torch_extensions'}). "
            f"This may take ~60s on first build."
        )

        # 多进程安全：文件锁，同目录下只有一个进程编译
        lock_path = os.path.join(build_dir or "/tmp", f"{self.so_name}.lock") if build_dir else None
        lock_fd = None
        if lock_path:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

        try:
            # 再次检查（等待锁期间可能已被其他进程编译完）
            if so_path and os.path.isfile(so_path):
                torch.ops.load_library(so_path)
                return True

            ext = cpp_load(
                name=self.so_name,
                sources=self.get_sources(),
                extra_cuda_cflags=[
                    *arch_flags,
                    *include_args,
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
                extra_cflags=[
                    "-O3",
                    "-ffast-math",
                    *include_args,
                ],
                extra_ldflags=[],
                build_directory=build_dir or None,
                verbose=False,
            )
        finally:
            if lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()

        warnings.warn(
            f"[{type(self).__name__}] pid={os.getpid()} "
            f"compiled and loaded in {time.monotonic() - t0:.1f}s."
        )
        return ext

    def _verify_ops(self) -> None:
        """内部：验证必需算子已注册，可选算子缺失时仅警告。"""
        ns = getattr(torch.ops, self.namespace, None)
        if ns is None:
            raise RuntimeError(
                f"torch.ops.{self.namespace} namespace not found after loading "
                f"{self.so_name}. Check CUSTOM_OPS_NAMESPACE in bindings.cpp."
            )

        missing = []
        for op_name in self.required_ops:
            try:
                getattr(ns, op_name)
            except AttributeError:
                missing.append(op_name)

        if missing:
            raise RuntimeError(
                f"Required ops not found in torch.ops.{self.namespace}: {missing}."
            )

        for op_name in self.optional_ops:
            try:
                getattr(ns, op_name)
            except AttributeError:
                warnings.warn(
                    f"[{type(self).__name__}] Optional op "
                    f"'{self.namespace}::{op_name}' not found."
                )

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

        说明
        ----
        - 内部调用 ``torch.library.register_fake``（PyTorch >= 2.1）或回退到
          ``torch.library.impl_abstract``（PyTorch 2.0）。
        - 已注册的算子重复注册时会静默忽略，不会报错。
        """
        qualified = f"{self.namespace}::{op_name}"
        try:
            # PyTorch >= 2.1 推荐接口
            torch.library.register_fake(qualified)(fake_fn)
        except Exception:
            try:
                # PyTorch 2.0 回退接口
                torch.library.impl_abstract(qualified)(fake_fn)
            except Exception:
                pass  # 版本不支持，静默忽略

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
# RecsysOps — 推荐系统三大核心算子库（包级内置，可直接导入）
#
# 用法：
#   from custom_ops import ops            # 使用预加载单例（推荐）
#   from custom_ops import RecsysOps, ops # 同时导入类与单例
#   from custom_ops import RecsysOps      # 仅导入类，自行实例化
# ─────────────────────────────────────────────────────────────────────────────

from custom_ops.recsys import RecsysOps  # noqa: E402  (避免循环导入，置于类定义之后)
from custom_ops.recsys import split_mixed_precision_weight  # noqa: E402

#: 推荐系统算子库的包级全局单例，import 时自动完成编译加载及 fake 注册。
#:
#: 用法::
#:
#:     from custom_ops import ops
#:     out  = ops.mha_fwd_with_mask(q, k, v, mask)
#:     y    = ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu")
ops: RecsysOps = RecsysOps().load()
