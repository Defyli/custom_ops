/*
 * fuse_moe_op.cu — MoE 前向融合算子，PyTorch C++ 接口实现
 *
 * 对外暴露：fuse_moe_cuda(x, gate_up_weight, down_weight, topk_ids, topk_scale)
 *
 * 数据流（单 GPU，无 fp8 量化——x/权重 bf16 或 fp16 直入，MMA fp32 累加）：
 *   ① count/build_indices：按 topk_ids 统计每 expert token 数（seqlens/
 *      cu_seqlens）、构建 row_indices（expert 排序后布局的源行号）与
 *      topk_pos（(token, topk) → 排序后行号）、tiles（GEMM tile 数）
 *   ② gemm1 融合（gate/up 配对 N-tile）：act_out[t] = silu(gate)·up，
 *      gate/up = x[row_indices[t]] @ W1[e]^T 的两个 N 半区（共享 X tile）
 *   ③ down group GEMM：down_out[t] = act_out[t] @ W2[e]^T
 *      （W1[e] = gate_up_weight[e]，(2I, H)；W2[e] = down_weight[e]，(H, I)）
 *   ④ reduce：y[s] = Σ_j topk_scale[s,j] * down_out[topk_pos[s,j]]
 *
 * 架构分发：本文件零架构感知——校验 / 架构无关 workspace 分配 / 填
 * FMOE_params 后唯一一次调用 moe_fwd_launch()（见 fuse_moe_launch.h，
 * fuse_moe 的分发标准入口；gemm 引擎选择、tile 策略、PDL 开关全部内封在
 * 各家族 moe_launch_smXX 中，仿 csrc/fa/fa_fwd_launch.h 分层）。
 *
 * 所有中间缓冲经 torch 分配（caching allocator）；sm120 上流水线以 PDL
 * 串联（sm89 自动退化为普通 stream 顺序，见 launch 层判定）。
 */

#include "fuse_moe_op.h"

#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "cutlass/numeric_types.h"

#include "fuse_moe_launch.h"

// ── CUDA 实现 ─────────────────────────────────────────────────────────────────
torch::Tensor fuse_moe_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_weight,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_scale
) {
    // ── 基础校验 ────────────────────────────────────────────────────────────
    TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && down_weight.is_cuda() &&
                    topk_ids.is_cuda() && topk_scale.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && gate_up_weight.is_contiguous() &&
                    down_weight.is_contiguous() && topk_ids.is_contiguous() &&
                    topk_scale.is_contiguous(),
                "all inputs must be contiguous");

    const bool is_bf16 = (x.scalar_type() == torch::kBFloat16);
    const bool is_fp16 = (x.scalar_type() == torch::kHalf);
    TORCH_CHECK(is_bf16 || is_fp16, "x must be bfloat16 or float16");
    const torch::ScalarType elem = x.scalar_type();
    TORCH_CHECK(gate_up_weight.scalar_type() == elem && down_weight.scalar_type() == elem,
                "gate_up_weight/down_weight dtype must match x");
    TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids must be int32");
    TORCH_CHECK(topk_scale.dtype() == torch::kFloat32, "topk_scale must be float32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D (num_seq, hidden)");
    TORCH_CHECK(gate_up_weight.dim() == 3, "gate_up_weight must be 3D (E, 2I, hidden)");
    TORCH_CHECK(down_weight.dim() == 3, "down_weight must be 3D (E, hidden, I)");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D (num_seq, num_topk)");

    const int64_t num_seq = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t num_expert = gate_up_weight.size(0);
    const int64_t intermediate2 = gate_up_weight.size(1);  // 2 * I
    const int64_t num_topk = topk_ids.size(1);

    TORCH_CHECK(num_seq > 0 && hidden > 0 && num_expert > 0, "empty input");
    TORCH_CHECK(intermediate2 % 2 == 0, "gate_up_weight.size(1) must be even (gate+up fused)");
    const int64_t intermediate = intermediate2 / 2;

    TORCH_CHECK(topk_ids.size(0) == num_seq, "topk_ids must share num_seq with x");
    TORCH_CHECK(topk_scale.sizes() == topk_ids.sizes(), "topk_scale shape must match topk_ids");
    TORCH_CHECK(gate_up_weight.size(2) == hidden,
                "gate_up_weight.size(2) must match hidden");
    TORCH_CHECK(down_weight.size(0) == num_expert,
                "down_weight must share num_expert with gate_up_weight");
    TORCH_CHECK(down_weight.size(1) == hidden, "down_weight.size(1) must match hidden");
    TORCH_CHECK(down_weight.size(2) == intermediate,
                "down_weight.size(2) must match intermediate (= gate_up_weight.size(1)/2)");

    // kernel tile 约定：kTileN=64（n=2I 与 n=H 均须整除）、kTileK∈{64,128}
    // （k=H 与 k=I 均须 %64）
    TORCH_CHECK(hidden % 64 == 0, "hidden must be a multiple of 64, got ", hidden);
    TORCH_CHECK(intermediate % 64 == 0,
                "intermediate (= gate_up_weight.size(1)/2) must be a multiple of 64, got ",
                intermediate);
    TORCH_CHECK(num_topk <= 128, "num_topk must be <= 128, got ", num_topk);
    TORCH_CHECK(num_expert <= 512, "num_expert must be <= 512, got ", num_expert);

    // ── 环境与输出 ──────────────────────────────────────────────────────────
    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ── 架构无关 workspace ─────────────────────────────────────────────────
    // 紧凑布局：act_out / down_out 按 compact 行号（expert 有序）。
    // gate_up_out 仅 TMA 引擎使用（其 gemm1 为独立 gate/up 两个 GEMM，无
    // 配对融合变体）；cp.async 引擎的 gemm1 融合 epilogue 直写 act_out，
    // 不分配该缓冲。TMA 引擎的 x_sorted 由 launch 层自行分配。
    torch::Tensor out = torch::empty_like(x);

    auto options_i32 = x.options().dtype(torch::kInt32);
    torch::Tensor act_out =
        torch::empty({(int64_t)num_seq * num_topk, intermediate}, x.options());
    torch::Tensor down_out =
        torch::empty({(int64_t)num_seq * num_topk, hidden}, x.options());
    torch::Tensor topk_pos = torch::empty({num_seq, num_topk}, options_i32);
    torch::Tensor row_indices = torch::empty({(int64_t)num_seq * num_topk}, options_i32);
    torch::Tensor seqlens = torch::zeros({num_expert}, options_i32);
    torch::Tensor cu_seqlens = torch::empty({num_expert + 1}, options_i32);
    torch::Tensor tiles = torch::empty({num_expert}, options_i32);
    torch::Tensor gate_up_out;  // 仅 FUSE_MOE_TMA=1（TMA 引擎）
    if (std::getenv("FUSE_MOE_TMA") != nullptr) {
        gate_up_out =
            torch::empty({(int64_t)num_seq * num_topk, intermediate2}, x.options());
    }

    // ── 填 params 并分发（tile/PDL/引擎策略全部内封在 launch 层）──────────
    fuse_moe::FMOE_params params;
    params.x_ptr = x.data_ptr();
    params.w1_ptr = gate_up_weight.data_ptr();
    params.w2_ptr = down_weight.data_ptr();
    params.topk_ids_ptr = topk_ids.data_ptr<int>();
    params.topk_scale_ptr = topk_scale.data_ptr<float>();

    params.row_indices_ptr = row_indices.data_ptr<int>();
    params.topk_pos_ptr = topk_pos.data_ptr<int>();
    params.seqlens_ptr = seqlens.data_ptr<int>();
    params.cu_seqlens_ptr = cu_seqlens.data_ptr<int>();
    params.tiles_ptr = tiles.data_ptr<int>();
    params.gate_up_out_ptr = gate_up_out.defined() ? gate_up_out.data_ptr() : nullptr;
    params.act_out_ptr = act_out.data_ptr();
    params.down_out_ptr = down_out.data_ptr();
    params.out_ptr = out.data_ptr();

    params.num_seq = static_cast<int>(num_seq);
    params.hidden = static_cast<int>(hidden);
    params.intermediate = static_cast<int>(intermediate);
    params.num_expert = static_cast<int>(num_expert);
    params.num_topk = static_cast<int>(num_topk);
    params.total_num_seq = static_cast<int>(num_seq * num_topk);
    params.avg_tokens_per_expert = static_cast<int>(num_seq * num_topk / num_expert);
    params.is_bf16 = is_bf16;

    fuse_moe::moe_fwd_launch(params, stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ── CPU 桩 ───────────────────────────────────────────────────────────────────
torch::Tensor fuse_moe_cpu(
    const torch::Tensor& /*x*/,
    const torch::Tensor& /*gate_up_weight*/,
    const torch::Tensor& /*down_weight*/,
    const torch::Tensor& /*topk_ids*/,
    const torch::Tensor& /*topk_scale*/
) {
    TORCH_CHECK(false, "fuse_moe is only supported on CUDA");
}
