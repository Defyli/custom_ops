/*
 * swiglu_op.h — SwiGLU 融合算子（bf16/fp16，单 GPU），PyTorch 算子声明
 *
 * y = silu(x @ Wg^T) * (x @ Wu^T)
 *   x   : (M, K) 激活
 *   W   : (2N, K) gate/up 融合权重（行主序，gate 在 [0, N)，up 在 [N, 2N)）
 *   y   : (M, N)
 *
 * 独立算子（与 fuse_moe 完全解耦，不含其任何头）：自包含 dense 配对
 * GEMM kernel（swiglu_kernel.cuh）——gate/up 配对 N-tile（双累加器共享
 * X tile 装载）+ silu·mul epilogue（激活在 fp32 累加器上计算，无
 * gate_up 中间量物化）。
 */

#pragma once

#include <torch/extension.h>

/**
 * swiglu_cuda
 *
 * @param x      (M, K)   bfloat16 / float16，CUDA，连续
 * @param weight (2N, K)  与 x 同 dtype（gate 行在前，up 行在后）
 * @return       (M, N)   与 x 同 dtype
 *
 * 限制（tile 约定）：K % 64 == 0，N % 64 == 0（即 weight.size(0) % 128 == 0）
 * M 任意正整数（M 尾部行谓词）；sm80+（sm89 / sm120 主路径）。
 */
torch::Tensor swiglu_cuda(const torch::Tensor& x, const torch::Tensor& weight);
torch::Tensor swiglu_cpu(const torch::Tensor& x, const torch::Tensor& weight);
