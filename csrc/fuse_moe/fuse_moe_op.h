/*
 * fuse_moe_op.h — MoE 前向融合算子（bf16/fp16，单 GPU），PyTorch 算子声明
 *
 * 完整 MoE FFN：y[s] = Σ_j topk_scale[s,j] · (Down_ej @ (act(Gate_ej @ x[s]) · (Up_ej @ x[s])))
 *   act = SiLU（expert_activation=0，默认）或 GELU-erf（expert_activation=1）
 *   GateUp_ej : (2I, H)  gate/up 融合权重（行主序，gate 在前 up 在后）
 *   Down_ej   : (H, I)
 * kernel 流水（移植自 hpc-ops fuse_moe，sm90 wgmma → sm80 mma.sync + cp.async）：
 *   count/build_indices → gate_up group GEMM（scatter gather）→ act(gate)*up
 *   → down group GEMM → topk 加权 reduce；sm120 上以 PDL 链接。
 */

#pragma once

#include <torch/extension.h>

/**
 * fuse_moe_cuda
 *
 * @param x              (S, H)   bfloat16 / float16，CUDA，连续
 * @param gate_up_weight GLU: (E, 2I, H)；MLP: (E, I, H)，与 x 同 dtype
 * @param down_weight    (E, H, I) 与 x 同 dtype
 * @param topk_ids       (S, K) int32，非负值 ∈ [0, E)，-1 masked 槽位跳过
 * @param topk_scale     (S, K) float32
 * @param expert_activation 0 = SiLU (SwiGLU，默认)，1 = GELU-erf (GEGLU；
 *                       PyTorch gelu exact 式，不用 tanh 近似)
 * @param numerical_mode 0 = fused 旧语义（默认）；1 = staged（gemm1 的
 *                       gate/up、act(gate)、激活乘积、down、加权积依次
 *                       round 到 Tout，再 FP32 reduce，最终 round 到 Tout；
 *                       不承诺跨后端逐 bit 一致）
 * @param expert_type    0 = GLU（默认）；1 = MLP，y = Σ scale·W2 act(W1 x)，
 *                       无 up 分支；staged 在激活前 round 单投影到 Tout
 * @return               (S, H) 与 x 同 dtype
 *
 * 限制（均由 kernel 的 tile 约定决定）：
 *   - H % 64 == 0 且 I % 64 == 0
 *   - 0 < num_topk <= 128，num_expert <= 512
 *   - sm80+（sm89 / sm120 主路径）
 */
torch::Tensor fuse_moe_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_scale,
    int64_t expert_activation = 0,
    int64_t numerical_mode = 0,
    int64_t expert_type = 0
);

torch::Tensor fuse_moe_cpu(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_scale,
    int64_t expert_activation = 0,
    int64_t numerical_mode = 0,
    int64_t expert_type = 0
);
