/*
 * fa_fwd_op.h — Flash Attention Forward with Additive Mask，函数声明
 */

#pragma once

#include <torch/extension.h>

/**
 * mha_fwd_with_mask_cuda
 *
 * Flash Attention 2 前向，支持任意 fp16/bf16 加法 mask。
 *
 * @param q     (B, H,  Sq, d)  fp16/bf16，CUDA，连续（Sk 须为 8 的倍数，Sq 任意）
 * @param k     (B, Hk, Sk, d)  与 q 同 dtype，CUDA，连续
 * @param v     (B, Hk, Sk, d)  与 q 同 dtype，CUDA，连续
 * @param mask  (B, 1, Sq_mask, Sk_mask) 与 q 同 dtype，CUDA，连续；加法 mask：0=可见，-inf=屏蔽
 *              q 维：Sq_mask >= Sq 即可，无需 pad（kernel 内谓词/TMA OOB 处理越界行）
 *              k 维：Sk_mask >= Sk 且为 8 的倍数（== Sk 零拷贝；或预 pad——
 *                    越界列内容被 kernel 忽略，col >= Sk 恒为屏蔽）
 * @return out  (B, H,  Sq, d)  与输入同 dtype
 *
 * 限制：
 *   - d ∈ {64, 128}；Sk % 8 == 0（128-bit cp.async / TMA 16B 行对齐）
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
