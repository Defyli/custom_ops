"""
custom_ops/recsys.py — RecsysOps：推荐系统核心 CUDA 算子库（分组懒加载）

算子分组（每组独立 .so、独立 JIT 编译/缓存，首次访问时才编译）：
- fa         mha_fwd_with_mask      Flash Attention 2 前向，任意 fp16/bf16 加法 mask
- mixed_gemm mixed_gemm (+fp8 查询)  混合精度 GEMM：bf16 主项 + fp8/int8 residual
- fuse_moe   fuse_moe                MoE 前向融合（group GEMM + silu*mul + reduce）

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
import torch

from custom_ops import CustomOps


# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # custom_ops/ 根目录

# residual 补偿 scale 的默认值（与原 TRT 插件一致）
DEFAULT_RESIDUAL_SCALE = 1.0 / 256.0

# 激活函数名 → kernel 枚举
_ACTIVATION_TO_ID = {"identity": 0, "silu": 1, "gelu": 2}


# ─────────────────────────────────────────────────────────────────────────────
# 分组 kernel 库（每组一个 CustomOps 子类 → 独立 .so）
# ─────────────────────────────────────────────────────────────────────────────

class FAOps(CustomOps):
    """FA 分组：mha_fwd_with_mask（FlashAttention-2 + 任意加法 mask）。

    sm70 / sm89 / sm120 三路径互不依赖，编译期由 FA_TARGETS 裁剪。
    """

    namespace    = "recsys_ops"
    so_name      = "recsys_fa_kernel"
    required_ops = ["mha_fwd_with_mask"]
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
    so_name      = "recsys_mixed_gemm_kernel"
    required_ops = ["mixed_gemm", "mixed_gemm_fp8_available"]
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
    """fuse_moe 分组：MoE 前向融合算子（group GEMM + silu*mul + reduce）。"""

    namespace    = "recsys_ops"
    so_name      = "recsys_fuse_moe_kernel"
    required_ops = ["fuse_moe"]
    optional_ops = []
    build_dir_fallback = "./torch_extensions"

    def get_sources(self) -> list:
        moe = os.path.join(_THIS_DIR, "csrc", "fuse_moe")
        return [
            os.path.join(moe, "fuse_moe_bindings.cpp"),
            os.path.join(moe, "fuse_moe_op.cu"),
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
        self.register_fake_for(
            "fuse_moe",
            lambda x, gate_up_weight, down_weight, topk_ids, topk_scale:
                torch.empty_like(x),
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
    }

    _OP_TO_GROUP = {
        "mha_fwd_with_mask": "fa",
        "mixed_gemm": "mixed_gemm",
        "mixed_gemm_fp8_available": "mixed_gemm",
        "fuse_moe": "fuse_moe",
    }

    def __init__(self) -> None:
        self._groups: dict = {}   # group name -> CustomOps 实例（已尝试加载）

    # ── 分组管理 ──────────────────────────────────────────────────────────────

    def _group(self, name: str) -> CustomOps:
        """加载（或取已加载的）分组实例。编译失败时实例保留（is_available
        为 False、load_error 携带原因），后续访问继续抛出同一信息。"""
        if name not in self._groups:
            self._groups[name] = self._GROUP_CLASSES[name]().load()
        return self._groups[name]

    def ensure_loaded(self, *names) -> "RecsysOps":
        """显式加载分组。无参 = 全部分组（旧 eager 行为）。返回 self。"""
        targets = names if names else tuple(self._GROUP_CLASSES)
        for n in targets:
            if n not in self._GROUP_CLASSES:
                raise ValueError(
                    f"unknown group {n!r}; available: {sorted(self._GROUP_CLASSES)}")
            self._group(n)
        return self

    # load() 保持旧 API 兼容：RecsysOps().load() == 全量 eager 加载
    def load(self) -> "RecsysOps":
        return self.ensure_loaded()

    @property
    def groups(self) -> dict:
        """{分组名: 状态} 视图（loaded/failed/not_loaded）。"""
        status = {}
        for n in self._GROUP_CLASSES:
            if n in self._groups:
                status[n] = "loaded" if self._groups[n].is_available() else "failed"
            else:
                status[n] = "not_loaded"
        return status

    def is_available(self, group: str | None = None) -> bool:
        """
        group 非空：加载该分组并返回其可用性（会触发编译）。
        group 为空：CUDA 可用 且 所有已尝试加载的分组均成功（不触发编译；
                    未加载的分组不影响结果——这正是懒加载语义）。
        """
        if group is not None:
            if group not in self._GROUP_CLASSES:
                raise ValueError(
                    f"unknown group {group!r}; available: {sorted(self._GROUP_CLASSES)}")
            return self._group(group).is_available()
        if not torch.cuda.is_available():
            return False
        return all(g.is_available() for g in self._groups.values())

    def load_error(self, group: str | None = None):
        """返回分组（或首个失败分组）的加载错误信息；无失败返回 None。"""
        if group is not None:
            return self._group(group).load_error()
        for g in self._groups.values():
            if not g.is_available():
                return g.load_error()
        return None

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
        return (f"RecsysOps(lazy, namespace='recsys_ops', groups={self.groups})")

    # ── 显式方法：带类型标注，供 IDE 补全（与门面语义一致，均懒加载）──────

    def mha_fwd_with_mask(
        self,
        q:    torch.Tensor,   # (B, H,  Sq, d)   fp16/bf16 CUDA
        k:    torch.Tensor,   # (B, Hk, Sk, d)   fp16/bf16 CUDA
        v:    torch.Tensor,   # (B, Hk, Sk, d)   fp16/bf16 CUDA
        mask: torch.Tensor,   # (B, 1, ≥Sq, ≥Sk) 与 q 同 dtype，0=可见 / -inf=屏蔽
    ) -> torch.Tensor:
        """Flash Attention 2 前向，支持任意 fp16/bf16 加法 mask（懒加载 FA 分组）。"""
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
    ) -> torch.Tensor:
        """混合精度 GEMM：y = activation(x @ (w_high + w_low*scale)^T + bias)。"""
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
        gate_up_weight: torch.Tensor,                      # (E, 2I, H) 同 dtype
        down_weight:    torch.Tensor,                      # (E, H, I) 同 dtype
        topk_ids:       torch.Tensor,                      # (S, K) int32
        topk_scale:     torch.Tensor,                      # (S, K) fp32
    ) -> torch.Tensor:
        """
        MoE 前向融合算子（单 GPU，bf16/fp16 输入，fp32 累加）：

            y[s] = Σ_j topk_scale[s,j] * (Down_ej @ silu(GateUp_ej @ x[s]))

        限制：H % 64 == 0 且 I % 64 == 0；num_topk <= 128；num_expert <= 512；
        SM80+（sm89 / sm120）。
        """
        return self._group("fuse_moe").fuse_moe(
            x, gate_up_weight, down_weight, topk_ids, topk_scale)


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
