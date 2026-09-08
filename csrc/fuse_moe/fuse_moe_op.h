/*
 * fuse_moe_op.h — MoE 前向融合算子（bf16/fp16，单 GPU），PyTorch 算子声明
 *
 * 完整 MoE FFN：y[s] = Σ_j topk_scale[s,j] · (Down_ej @ silu(GateUp_ej @ x[s]))
 *   GateUp_ej : (2I, H)  gate/up 融合权重（行主序，gate 在前 up 在后）
 *   Down_ej   : (H, I)
 * kernel 流水（移植自 hpc-ops fuse_moe，sm90 wgmma → sm80 mma.sync + cp.async）：
 *   count/build_indices → gate_up group GEMM（scatter gather）→ silu*mul
 *   → down group GEMM → topk 加权 reduce；sm120 上以 PDL 链接。
 */

#pragma once

#include <torch/extension.h>

/**
 * fuse_moe_cuda
 *
 * @param x              (S, H)   bfloat16 / float16，CUDA，连续
 * @param gate_up_weight (E, 2I, H) 与 x 同 dtype（gate_up_weight.size(1) = 2*I）
 * @param down_weight    (E, H, I) 与 x 同 dtype
 * @param topk_ids       (S, K) int32，取值 ∈ [0, E)
 * @param topk_scale     (S, K) float32
 * @return               (S, H) 与 x 同 dtype
 *
 * 限制（均由 kernel 的 tile 约定决定）：
 *   - H % 64 == 0 且 I % 64 == 0
 *   - num_topk <= 128，num_expert <= 512
 *   - sm80+（sm89 / sm120 主路径）
 */
torch::Tensor fuse_moe_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_scale
);

torch::Tensor fuse_moe_cpu(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_scale
);
