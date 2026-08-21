/*
 * mixed_gemm_op.h — Mixed-Precision GEMM（bf16 主项 + fp8/int8 residual），
 *                   PyTorch 算子声明
 */

#pragma once

#include <torch/extension.h>

/**
 * mixed_gemm_cuda
 *
 * y = activation(x @ (w_high + w_low * scale)^T + bias)
 *
 * 权重离线预切分（见 recsys.py 的 split_mixed_precision_weight）：
 *   w_high : (N, K) bfloat16        —— 主项（bf16 tensor core）
 *   w_low  : (N, K) residual 项，dtype 决定后端：
 *            - float8_e4m3fn  → FP8 后端（SM89+，需编译期 CUDA >= 12.4）
 *            - int8           → INT8 动态量化后端（SM80+，需配 w_scale）
 *
 * @param x           (..., K) float32 或 bfloat16，CUDA，连续（前导维度折叠为 M）
 * @param w_high      (N, K) bfloat16
 * @param w_low       (N, K) float8_e4m3fn 或 int8
 * @param w_scale     (N,) float32，仅 INT8 后端（w_low 的 per-channel 量化 scale）
 * @param scale       residual 补偿 scale（默认 1/256，与权重切分时一致）
 * @param bias        (N,) float32，可选，epilogue 融合 bias 相加
 * @param activation  0=identity, 1=silu, 2=gelu(tanh)，epilogue 融合激活
 * @param fp32_output true→float32 输出，false→bfloat16 输出
 * @param force_splitk 0=自动选择；>0 强制 split-K（调试用）
 * @return            (..., N) fp32 或 bf16
 *
 * 限制：
 *   - K % 8 == 0（cp.async 128-bit 对齐契约）
 *   - x 前导维度乘积 M 与 N、K 均 <= INT32_MAX
 */
torch::Tensor mixed_gemm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& w_high,
    const torch::Tensor& w_low,
    const c10::optional<torch::Tensor>& w_scale,
    double scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t activation,
    bool fp32_output,
    int64_t force_splitk
);

torch::Tensor mixed_gemm_cpu(
    const torch::Tensor& x,
    const torch::Tensor& w_high,
    const torch::Tensor& w_low,
    const c10::optional<torch::Tensor>& w_scale,
    double scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t activation,
    bool fp32_output,
    int64_t force_splitk
);
