"""
custom_ops/tilelang_ops/_mixed_gemm.py — Tile-lang 版 mixed_gemm

与 CUDA 后端（csrc/mixed_gemm/）同语义：

    y = activation(x @ (w_high + w_low * scale)^T + bias)

  - x: (..., K) fp32 / bf16 / fp16（前导维度折叠为 M，fp32 输入在搬运时
    cast 到 bf16，与 CUDA 版主项路径一致）
  - w_high: (N, K) bf16 主项
  - w_low: (N, K) residual 项，int8（需 w_scale per-channel 反量化）或
    fp8 e4m3（w_scale=None）；两者的值域（|v| ≤ 127 / fp8 尾数 3bit）在
    bf16 中均可精确表示，故 kernel 内 cast 到 bf16 走 tensor core，
    fp32 累加——数值语义等价于 CUDA 版（residual 项不引入额外舍入）
  - epilogue 融合 bias 相加与 silu / gelu(tanh) 激活，out fp32 / bf16

单 kernel 实现：主项与 residual 双 GEMM 累加到两个 fp32 fragment，
epilogue 合并、反量化、加 bias、激活后一次写出。
"""

import torch

import tilelang
import tilelang.language as T

from custom_ops.recsys import DEFAULT_RESIDUAL_SCALE

_KERNEL_CACHE = {}

_ACT_TO_ID = {"identity": 0, "silu": 1, "gelu": 2}


@tilelang.jit(
    out_idx=[6],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _mixed_gemm_factory(M, N, K, x_dtype, wlow_dtype, out_dtype, act_id,
                        block_M=128, block_N=128, block_K=64,
                        num_stages=2, threads=256):
    """shape/语义特化的 mixed_gemm kernel 工厂。"""

    @T.prim_func
    def main(
        X:      T.Tensor((M, K), x_dtype),
        W_high: T.Tensor((N, K), "bfloat16"),
        W_low:  T.Tensor((N, K), wlow_dtype),
        Wscale: T.Tensor((N,), "float32"),   # int8: per-channel 量化 scale；fp8: ones
        Bias:   T.Tensor((N,), "float32"),   # 无 bias 时传 zeros
        scale:  T.float32,                   # residual 全局 scale
        Y:      T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N),
                      threads=threads) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), "bfloat16")
            W_shared = T.alloc_shared((block_N, block_K), "bfloat16")
            Wl_shared = T.alloc_shared((block_N, block_K), "bfloat16")
            acc = T.alloc_fragment((block_M, block_N), "float32")
            acc_res = T.alloc_fragment((block_M, block_N), "float32")
            wsc = T.alloc_fragment((block_N,), "float32")
            bias = T.alloc_fragment((block_N,), "float32")

            # per-channel 反量化因子与 bias（钳位读，越界值被 copy 谓词丢弃）
            for j in T.Parallel(block_N):
                jj = T.min(by * block_N + j, N - 1)
                wsc[j] = Wscale[jj]
                bias[j] = Bias[jj]

            T.clear(acc)
            T.clear(acc_res)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # fp32 输入在搬运时 cast 到 bf16；int8/fp8 residual 同样 cast
                #（值域在 bf16 精确可表，无精度损失）
                T.copy(X[bx * block_M:(bx + 1) * block_M,
                         k * block_K:(k + 1) * block_K], X_shared)
                T.copy(W_high[by * block_N:(by + 1) * block_N,
                              k * block_K:(k + 1) * block_K], W_shared)
                T.copy(W_low[by * block_N:(by + 1) * block_N,
                             k * block_K:(k + 1) * block_K], Wl_shared)
                T.gemm(X_shared, W_shared, acc, transpose_B=True)
                T.gemm(X_shared, Wl_shared, acc_res, transpose_B=True)

            # epilogue：main + residual·(w_scale·scale) + bias → 激活
            for i, j in T.Parallel(block_M, block_N):
                v = acc[i, j] + acc_res[i, j] * (wsc[j] * scale) + bias[j]
                if act_id == 1:      # silu
                    acc[i, j] = v / (1.0 + T.exp(-v))
                elif act_id == 2:    # gelu (tanh 近似)
                    t = T.tanh(0.7978845608028654 *
                               (v + 0.044715 * v * v * v))
                    acc[i, j] = 0.5 * v * (1.0 + t)
                else:
                    acc[i, j] = v
            T.copy(acc, Y[bx * block_M:(bx + 1) * block_M,
                          by * block_N:(by + 1) * block_N])

    return main


def _default_config(M, N, K):
    """tile 按 grid 充满度自适应（实测 4090D/128 SM，目标 ≥2 blocks/SM）。

    grid = ceil(M/BM)·ceil(N/BN)。降档顺序：先降 BM（mma 效率损失小、
    并行度翻倍）再降 BN；实测收益：
      (128,1000,2048) 128x128→32x64   2.71x
      (512,4096,256)  128x128→64x128  1.10x
      (1024,4096,4096) 128x128→64x128 1.05x
    """
    def grid(bm, bn):
        return ((M + bm - 1) // bm) * ((N + bn - 1) // bn)

    if M < 32:
        bm, bn = 16, 128          # 极小 M：BM 压到 16，BN 保大面板
    elif grid(128, 128) >= 512:
        bm, bn = 128, 128         # grid 已 ≥4 waves：大 tile 效率最高
    elif grid(64, 128) >= 256:
        bm, bn = 64, 128
    elif grid(64, 64) >= 256:
        bm, bn = 64, 64
    else:
        bm, bn = 32, 64           # 小 shape：以并行度优先
    return dict(block_M=bm, block_N=bn, block_K=64, num_stages=2, threads=256)


# 候选 config（bench sweep 用）：默认值 + 邻域变体

def candidate_configs(M, N=4096, K=4096):
    base = _default_config(M, N, K)
    cfgs = [base]
    if base["block_M"] >= 64:
        cfgs.append(dict(base, block_M=base["block_M"] // 2))
    if base["block_N"] >= 128:
        cfgs.append(dict(base, block_N=64))
    else:
        cfgs.append(dict(base, block_N=128))
    cfgs.append(dict(base, num_stages=3))
    cfgs.append(dict(base, block_K=32, num_stages=3))
    return cfgs


_DUMMY_CACHE = {}


def _dummy_vec(N, kind):
    """ones / zeros 的 (N,) fp32 缓存（fp8 的 w_scale=None、bias=None 占位）。"""
    key = (N, kind, torch.cuda.current_device())
    if key not in _DUMMY_CACHE:
        _DUMMY_CACHE[key] = (torch.ones if kind == "ones" else torch.zeros)(
            N, dtype=torch.float32, device="cuda")
    return _DUMMY_CACHE[key]


def make_mixed_gemm(M, N, K, x_dtype, wlow_dtype, out_dtype, act_id,
                    block_M, block_N, block_K, num_stages, threads):
    key = (M, N, K, x_dtype, wlow_dtype, out_dtype, act_id,
           block_M, block_N, block_K, num_stages, threads)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    kernel = _mixed_gemm_factory(
        M, N, K, x_dtype, wlow_dtype, out_dtype, act_id,
        block_M=block_M, block_N=block_N, block_K=block_K,
        num_stages=num_stages, threads=threads)
    _KERNEL_CACHE[key] = kernel
    return kernel


def mixed_gemm_tilelang(x, w_high, w_low, w_scale=None, *,
                        scale=DEFAULT_RESIDUAL_SCALE, bias=None,
                        activation="identity", out_dtype=None, config=None):
    """Tile-lang 版 mixed_gemm 前端（签名与 ops.mixed_gemm 对齐）。"""
    if activation not in _ACT_TO_ID:
        raise ValueError(f"activation must be one of {list(_ACT_TO_ID)}, "
                         f"got {activation!r}")
    if x.dim() < 1:
        raise ValueError("x must have at least 1 dim")
    K = x.shape[-1]
    M = x.numel() // K
    if w_high.dim() != 2 or w_low.dim() != 2:
        raise ValueError("w_high / w_low must be 2D (N, K)")
    N, Kw = w_high.shape
    if Kw != K or w_low.shape != (N, K):
        raise ValueError(f"shape mismatch: x(*,{K}) vs w_high{tuple(w_high.shape)} "
                         f"vs w_low{tuple(w_low.shape)}")
    if w_high.dtype != torch.bfloat16:
        raise ValueError("w_high must be bf16 (split_mixed_precision_weight 产出)")
    if w_low.dtype == torch.int8:
        if w_scale is None:
            raise ValueError("INT8 residual 需要 w_scale (N,) fp32")
        if w_scale.shape != (N,) or w_scale.dtype != torch.float32:
            raise ValueError("w_scale must be (N,) fp32")
        wlow_dtype = "int8"
    elif w_low.dtype == torch.float8_e4m3fn:
        if w_scale is not None and w_scale.shape != (N,):
            raise ValueError("FP8 residual 的 w_scale 应为 None")
        wlow_dtype = "float8_e4m3"
        w_scale = _dummy_vec(N, "ones")
    else:
        raise ValueError(f"w_low dtype 必须是 int8 或 float8_e4m3fn，"
                         f"got {w_low.dtype}")
    if bias is None:
        bias = _dummy_vec(N, "zeros")
    elif bias.shape != (N,) or bias.dtype != torch.float32:
        raise ValueError("bias must be (N,) fp32")

    out_dtype = out_dtype or torch.float32
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("out_dtype must be torch.float32 or torch.bfloat16")

    x_dtype = {torch.float32: "float32", torch.bfloat16: "bfloat16",
               torch.float16: "float16"}[x.dtype]
    out_str = {torch.float32: "float32", torch.bfloat16: "bfloat16"}[out_dtype]

    cfg = config or _default_config(M, N, K)
    x2d = x.reshape(M, K)
    kernel = make_mixed_gemm(M, N, K, x_dtype, wlow_dtype, out_str,
                             _ACT_TO_ID[activation], **cfg)
    return kernel(x2d, w_high, w_low, w_scale, bias, scale)
