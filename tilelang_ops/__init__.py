"""
custom_ops/tilelang_ops — 算子库的 Tile-lang 后端（可后端选择）

为 `custom_ops.ops`（CUDA/CuTe 手写 kernel 后端）中的全部算子提供等价的
Tile-lang 实现，供用户按需选择：

- ``ops.mha_fwd_with_mask(q, k, v, mask, backend="tilelang")``
- ``ops.mixed_gemm(..., backend="tilelang")``
- ``ops.fuse_moe(..., backend="tilelang")``
- ``ops.swiglu(..., backend="tilelang")``

也可直接使用门面单例::

    from custom_ops import tilelang_ops
    y = tilelang_ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu")

适用场景
--------
- 本机没有 nvcc / GCC >= 9，或 GPU 架构不在手写路径（sm70/sm89/sm120）内——
  CUDA JIT 分组编译失败时，``backend="auto"`` 自动回退到本后端
- 快速试验 / 教学：Python 层面即可阅读与修改 kernel

数值语义与 CUDA 后端一致（对拍容差内）；性能为「可用且不差」定位，
不追求对手写 kernel 的极致优化（FA 除外——FA 的 Tile-lang 版在标准
Prefill 场景可达 CuTe 版 1.0~1.1x）。
"""

from __future__ import annotations

import torch

from custom_ops.recsys import DEFAULT_RESIDUAL_SCALE


def _tilelang_available() -> bool:
    try:
        import tilelang  # noqa: F401
        return True
    except Exception:
        return False


class TileLangOps:
    """Tile-lang 后端门面：与 ``custom_ops.ops``（RecsysOps）同名的算子方法。

    懒加载：首次调用某算子时才 import 对应子模块（并触发 tile-lang JIT
    编译，首次约数十秒，之后进程内缓存）。
    """

    def is_available(self) -> bool:
        """tile-lang 可安装性 + CUDA 可用性（不触发编译）。"""
        return _tilelang_available() and torch.cuda.is_available()

    # ── fa ────────────────────────────────────────────────────────────────────

    def mha_fwd_with_mask(
        self,
        q:    torch.Tensor,   # (B, H,  Sq, d)   fp16/bf16
        k:    torch.Tensor,   # (B, Hk, Sk, d)   fp16/bf16
        v:    torch.Tensor,   # (B, Hk, Sk, d)   fp16/bf16
        mask:   torch.Tensor,   # (B, 1, ≥Sq, ≥Sk) 与 q 同 dtype
        config: dict | None = None,    # 覆盖默认 tile config（基准测试用）
    ) -> torch.Tensor:
        """Flash Attention 2 前向（任意加法 mask，GQA，自适应 Split-KV）。"""
        from custom_ops.tilelang_ops._fa import fa_tilelang
        return fa_tilelang(q, k, v, mask, config=config)

    # ── mixed_gemm ────────────────────────────────────────────────────────────

    def mixed_gemm(
        self,
        x:       torch.Tensor,                      # (..., K) fp32/bf16/fp16
        w_high:  torch.Tensor,                      # (N, K) bf16
        w_low:   torch.Tensor,                      # (N, K) fp8_e4m3 或 int8
        w_scale: torch.Tensor | None = None,        # (N,) fp32，仅 INT8 residual
        *,
        scale:        float = DEFAULT_RESIDUAL_SCALE,
        bias:         torch.Tensor | None = None,
        activation:   str = "identity",
        out_dtype:    torch.dtype | None = None,
        config:       dict | None = None,    # 覆盖默认 tile config（基准测试用）
    ) -> torch.Tensor:
        """混合精度 GEMM：y = activation(x @ (w_high + w_low*scale)^T + bias)。

        主项与 residual 补偿项均为 bf16 tensor core（w_low 的 int8/fp8 值
        在 bf16 中精确可表），fp32 累加；epilogue 融合 bias 与激活。
        """
        from custom_ops.tilelang_ops._mixed_gemm import mixed_gemm_tilelang
        return mixed_gemm_tilelang(
            x, w_high, w_low, w_scale, scale=scale, bias=bias,
            activation=activation, out_dtype=out_dtype, config=config)

    # ── fuse_moe ──────────────────────────────────────────────────────────────

    def fuse_moe(
        self,
        x:              torch.Tensor,                      # (S, H) bf16/fp16
        gate_up_weight: torch.Tensor,                      # (E, 2I, H) 同 dtype
        down_weight:    torch.Tensor,                      # (E, H, I) 同 dtype
        topk_ids:       torch.Tensor,                      # (S, K) int32
        topk_scale:     torch.Tensor,                      # (S, K) fp32
        config:         dict | None = None,    # 覆盖默认 tile config（基准测试用）
    ) -> torch.Tensor:
        """MoE FFN 前向融合：y[s] = Σ_j topk_scale[s,j]·(Down_ej @ silu(GateUp_ej @ x[s]))。

        count/路由/gather/gemm1(silu·mul)/gemm2(加权原子归约) 共 5 个
        tile-lang kernel；限制与 CUDA 版一致（H%64==0、I%64==0、K≤128、E≤512）。
        """
        from custom_ops.tilelang_ops._fuse_moe import fuse_moe_tilelang
        return fuse_moe_tilelang(x, gate_up_weight, down_weight, topk_ids,
                                 topk_scale, config=config)

    # ── swiglu ────────────────────────────────────────────────────────────────

    def swiglu(
        self,
        x:      torch.Tensor,   # (M, K) bf16/fp16
        weight: torch.Tensor,   # (2N, K) 同 dtype，gate 行在前 up 在后
        config: dict | None = None,    # 覆盖默认 tile config（基准测试用）
    ) -> torch.Tensor:
        """SwiGLU 融合：y = silu(x @ Wg^T) * (x @ Wu^T)（配对 GEMM 单 kernel）。"""
        from custom_ops.tilelang_ops._swiglu import swiglu_tilelang
        return swiglu_tilelang(x, weight, config=config)


tilelang_ops = TileLangOps()


def __getattr__(name):
    """模块级转发：`from custom_ops import tilelang_ops` 拿到的是本模块，
    通过 PEP 562 __getattr__ 把算子方法访问转发到门面单例，
    使模块与单例两种用法等价。"""
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        return getattr(tilelang_ops, name)
    except AttributeError:
        raise AttributeError(
            f"module 'custom_ops.tilelang_ops' has no attribute {name!r}")
