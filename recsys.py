"""
custom_ops/recsys.py — RecsysOps：推荐系统核心 CUDA 算子库

算子列表
--------
- mha_fwd_with_mask      Flash Attention 2 前向，支持任意 bf16 加法 mask
- mixed_gemm             混合精度 GEMM：bf16 主项 + fp8/int8 residual 精度补偿，
                         epilogue 融合 bias 与 silu/gelu 激活

快速使用
--------
    from custom_ops import ops, split_mixed_precision_weight  # 包级单例

    if ops.is_available():
        out  = ops.mha_fwd_with_mask(q, k, v, mask)

        # 混合精度 GEMM：离线切分权重，在线一次调用
        w_high, w_low, w_scale = split_mixed_precision_weight(w_fp32)
        y = ops.mixed_gemm(x, w_high, w_low, w_scale,
                           bias=bias, activation="silu")
    else:
        # CUDA 不可用或编译失败时做 Python 回退
        ...

命名空间
--------
    torch.ops.recsys_ops.<name>(...)   — C++ 侧调用接口
    ops.<name>(...)                    — Python 封装，带类型标注
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
# RecsysOps — 推荐系统算子库
# ─────────────────────────────────────────────────────────────────────────────

class RecsysOps(CustomOps):
    """
    推荐系统核心 CUDA 算子库。

    命名空间  : recsys_ops  → torch.ops.recsys_ops.<name>(...)
    编译产物  : recsys_ops_kernel.so
    注册算子  :
      - mha_fwd_with_mask       （必需）Flash Attention 2 前向
      - mixed_gemm              （必需）混合精度 GEMM（bf16 + fp8/int8 residual）
      - mixed_gemm_fp8_available（必需）FP8 后端可用性查询
    """

    # ── 配置 ─────────────────────────────────────────────────────────────────
    namespace    = "recsys_ops"
    so_name      = "recsys_ops_kernel"
    required_ops = [
        "mha_fwd_with_mask",
        "mixed_gemm",
        "mixed_gemm_fp8_available",
    ]
    optional_ops = []

    # 编译缓存回退目录（优先使用 TORCH_EXTENSIONS_DIR 环境变量）
    build_dir_fallback = "./torch_extensions"

    def get_sources(self) -> list:
        """源文件列表：bindings + FA kernel + 混合精度 GEMM kernel。"""
        csrc = os.path.join(_THIS_DIR, "csrc")
        mixed = os.path.join(csrc, "mixed_gemm")
        return [
            os.path.join(csrc, "recsys_bindings.cpp"),
            os.path.join(csrc, "fa",        "fa_fwd_op.cu"),
            os.path.join(mixed, "mixed_gemm_op.cu"),
            os.path.join(mixed, "gemm_bf16xfp32_sm80.cu"),
            os.path.join(mixed, "gemm_bf16xfp32_sm120.cu"),
        ]

    def get_include_dirs(self) -> list:
        """头文件目录：thirdparty（cutlass/cute）+ csrc 各子目录 + 框架宏目录。"""
        csrc       = os.path.join(_THIS_DIR, "csrc")
        thirdparty = os.path.join(_THIS_DIR, "thirdparty")
        return [
            thirdparty,                          # cutlass/cute 头文件
            csrc,                                # custom_ops_macros.h
            os.path.join(csrc, "fa"),            # fa_fwd_op.h 等
            os.path.join(csrc, "mixed_gemm"),    # mixed_gemm_op.h 等
        ]

    # ── 显式方法：带类型标注，供 IDE 补全 ────────────────────────────────────

    def mha_fwd_with_mask(
        self,
        q:    torch.Tensor,   # (B, H,  Sq, d)  bfloat16 CUDA
        k:    torch.Tensor,   # (B, Hk, Sk, d)  bfloat16 CUDA
        v:    torch.Tensor,   # (B, Hk, Sk, d)  bfloat16 CUDA
        mask: torch.Tensor,   # (B, 1,  Sq, Sk) bfloat16 CUDA，0=可见 / -inf=屏蔽
    ) -> torch.Tensor:
        """
        Flash Attention 2 前向，支持任意 bf16 加法 mask。

        参数
        ----
        q, k, v : (B, H, S, d) bfloat16 CUDA，连续
        mask    : (B, 1, Sq, Sk) bfloat16 CUDA，加法 mask（0=可见，-inf=屏蔽）

                限制
        ----
        - d ∈ {64, 128}
        - H % Hk == 0（支持 GQA）
        - 不支持 dropout / causal / alibi / RoPE / KV-cache
        """
        return torch.ops.recsys_ops.mha_fwd_with_mask(q, k, v, mask)

    # ── 显式方法：混合精度 GEMM ──────────────────────────────────────────────

    def mixed_gemm(
        self,
        x:       torch.Tensor,                      # (..., K) fp32 或 bf16
        w_high:  torch.Tensor,                      # (N, K) bf16
        w_low:   torch.Tensor,                      # (N, K) fp8_e4m3 或 int8
        w_scale: torch.Tensor | None = None,        # (N,) fp32，仅 INT8 后端
        *,
        scale:        float = DEFAULT_RESIDUAL_SCALE,  # residual 补偿 scale
        bias:         torch.Tensor | None = None,   # (N,) fp32，epilogue 融合 bias
        activation:   str | int = "identity",       # "identity"/"silu"/"gelu" 或 0/1/2
        out_dtype:    torch.dtype | None = None,    # None=fp32, 或 torch.bfloat16
        force_splitk: int = 0,                      # 0=自动；>0 强制 split-K（调试）
    ) -> torch.Tensor:
        """
        混合精度 GEMM：y = activation(x @ (w_high + w_low*scale)^T + bias)。

        解决生成式推荐中 bf16 权重精度损失大、tf32 性能不足的问题：
        bf16 tensor core 算主项 + 低精度 tensor core 算 residual 补偿项，
        以接近 bf16 的速度恢复接近 fp32 的精度。bias 与激活在 epilogue
        融合执行，不产生额外 kernel 与中间显存。

        参数
        ----
        x       : (..., K) fp32 或 bf16，连续。前导维度折叠为 M
        w_high  : (N, K) bf16，主项权重（split_mixed_precision_weight 产出）
        w_low   : (N, K) residual 权重，dtype 决定后端：
                  - float8_e4m3fn → FP8 后端（SM89+，需编译期 CUDA >= 12.4）
                  - int8          → INT8 动态量化后端（SM80+，需配 w_scale）
        w_scale : (N,) fp32，INT8 后端的 per-channel 量化 scale
        scale   : residual 补偿 scale，须与权重切分时一致（默认 1/256）
        bias    : (N,) fp32，可选，epilogue 融合相加
        activation : "identity" | "silu" | "gelu"（tanh 近似）或 0/1/2
        out_dtype  : 输出 dtype。None → fp32（推荐，保精度）；
                     torch.bfloat16 → bf16 输出
        force_splitk : 调试用，强制 split-K 值（1/2/4/8/16）

        限制
        ----
        - K % 8 == 0
        - 后端要求：FP8 需 SM89+；INT8 需 SM80+
        """
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
        return torch.ops.recsys_ops.mixed_gemm(
            x, w_high, w_low, w_scale, scale, bias,
            activation, out_dtype == torch.float32, force_splitk)

    def mixed_gemm_fp8_available(self) -> bool:
        """
        FP8 residual 后端在本机是否可用（编译期 CUDA >= 12.4 且 GPU 为 SM89+）。

        False 时 ``split_mixed_precision_weight(backend="auto")`` 会自动
        降级到 INT8 动态量化后端（SM80+ 可用，精度略低于 FP8）。
        """
        return bool(torch.ops.recsys_ops.mixed_gemm_fp8_available())

    def register_fake_impls(self) -> None:
        """
        为所有算子注册 fake（meta）实现，供 torch.compile / AOTI 做 shape 推断。
        load() 成功后由基类自动调用，无需手动触发。
        """
        # mha_fwd_with_mask: 输出 shape 与 q 相同 (B, H, Sq, d)
        self.register_fake_for(
            "mha_fwd_with_mask",
            lambda q, k, v, mask: torch.empty_like(q),
        )
        # mixed_gemm: (..., K) -> (..., N)，dtype 由 fp32_output 决定
        self.register_fake_for(
            "mixed_gemm",
            lambda x, w_high, w_low, w_scale, scale, bias, activation, \
                   fp32_output, force_splitk: torch.empty(
                x.shape[:-1] + (w_high.shape[0],),
                dtype=torch.float32 if fp32_output else torch.bfloat16,
                device=x.device),
        )


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

    参数
    ----
    w       : (N, K) 权重，任意可转 float32 的 dtype，建议 fp32
    scale   : residual 补偿 scale，调用 mixed_gemm 时须传同一值
    backend : "fp8" | "int8" | "auto"
        - "fp8"  ：w_low 产出 float8_e4m3fn（SM89+，需编译期 CUDA >= 12.4）
        - "int8" ：w_low 产出 int8 + per-channel scale（SM80+，兼容性最好）
        - "auto" ：当前机器可用 FP8 则选 FP8，否则降级 INT8

    返回
    ----
    (w_high, w_low, w_scale)：
        w_high : (N, K) bf16
        w_low  : (N, K) float8_e4m3fn（FP8 后端）或 int8（INT8 后端）
        w_scale: (N,) fp32，仅 INT8 后端非 None

    示例
    ----
        w_high, w_low, w_scale = split_mixed_precision_weight(w_fp32)
        y = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=b, activation="silu")
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


# FP8 可用性进程级缓存（探测需要 torch.ops，可能触发 JIT 编译，避免反复触发）
_fp8_available_cache: bool | None = None


def _fp8_available_cached() -> bool:
    global _fp8_available_cache
    if _fp8_available_cache is None:
        try:
            _fp8_available_cache = bool(
                torch.ops.recsys_ops.mixed_gemm_fp8_available())
        except Exception:
            _fp8_available_cache = False
    return _fp8_available_cache

