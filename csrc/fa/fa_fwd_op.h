/*
 * fa_fwd_op.h — Flash Attention Forward with Additive Mask，函数声明
 */

#pragma once

#include <torch/extension.h>

/**
 * mha_fwd_with_mask_cuda
 *
 * Flash Attention 2 前向，支持任意 bf16 加法 mask。
 *
 * @param q     (B, H,  Sq, d)  bfloat16，CUDA，连续
 * @param k     (B, Hk, Sk, d)  bfloat16，CUDA，连续
 * @param v     (B, Hk, Sk, d)  bfloat16，CUDA，连续
 * @param mask  (B, 1,  Sq, Sk) bfloat16，CUDA，连续；加法 mask：0=可见，-inf=屏蔽
 * @return out  (B, H,  Sq, d)  bfloat16
 *
 * 限制：
 *   - d ∈ {64, 128}
 *   - H % Hk == 0（支持 GQA）
 *   - 不支持 dropout/causal/alibi/RoPE/KV-cache
 */
torch::Tensor mha_fwd_with_mask_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask
);

torch::Tensor mha_fwd_with_mask_cpu(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask
);
