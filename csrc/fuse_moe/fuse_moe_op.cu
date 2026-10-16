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
#include "fuse_moe_offload.h"

namespace {
__global__ void offload_validate_ids(const int* ids, int64_t n, int experts, int* invalid) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * blockDim.x) {
        if (ids[i] < -1 || ids[i] >= experts) atomicExch(invalid, 1);
    }
}

// Small preflight histogram, independent of O(S*K) route storage. Shared counts
// avoid a global atomic per route on the capacity-pressure slow path.
__global__ void offload_count_ids(const int* ids, int64_t n, int experts,
                                  int first, int bins, int* summary) {
    __shared__ int counts[128];
    if (threadIdx.x <= bins) counts[threadIdx.x] = 0;
    __syncthreads();
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int expert = ids[i];
        if (expert < -1 || expert >= experts) atomicExch(counts + bins, 1);
        else if (expert >= first && expert < first + bins) atomicAdd(counts + expert - first, 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e <= bins; e += blockDim.x) {
        if (e == bins) {
            if (counts[e]) atomicExch(summary + e, 1);
        } else if (counts[e]) atomicAdd(summary + e, counts[e]);
    }
}

// Scatter valid, unique route positions into the global compact scale array.
__global__ void offload_scales_kernel(const int* topk_pos, const float* scale,
                                      float* sorted_scale, int routes) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < routes;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int pos = topk_pos[i];
        if (pos >= 0 && pos < routes) sorted_scale[pos] = scale[i];
    }
}

// Build grouped metadata and weight tables from launch arguments, not pinned DMA.
// A single-expert fallback keeps the original 24-byte metadata and direct address.
__global__ void offload_resident_meta_kernel(int* meta, int groups, MoEResidentWave wave,
                                            int64_t gate_bytes, int tile_m) {
    int* prefix = meta + groups;
    int* tiles = prefix + groups + 1;
    int* cu_tiles = tiles + groups;
    prefix[0] = 0; cu_tiles[0] = 0;
    // (4*groups+2)*sizeof(int) is always 8-byte aligned for device pointers.
    auto** weights = reinterpret_cast<const void**>(meta + 4 * groups + 2);
    for (int g = 0; g < groups; ++g) {
        meta[g] = wave.rows[g];
        prefix[g + 1] = prefix[g] + wave.rows[g];
        tiles[g] = wave.rows[g] / tile_m + (wave.rows[g] % tile_m != 0);
        cu_tiles[g + 1] = cu_tiles[g] + tiles[g];
        if (groups > 1) {
            weights[g] = wave.weights[g];
            weights[groups + g] = static_cast<const uint8_t*>(wave.weights[g]) + gate_bytes;
        }
    }
}

// Clip only the participating expert interval and rebuild tile-local prefixes.
__global__ void offload_tile_meta_kernel(const int* wave_meta, int* tile_meta,
                                         int wave_groups, int first_group, int groups,
                                         int begin, int rows, int tile_m) {
    const int* wave_prefix = wave_meta + wave_groups + first_group;
    int* lens = tile_meta;
    int* prefix = lens + groups;
    int* tiles = prefix + groups + 1;
    int* cu_tiles = tiles + groups;
    const int64_t end = int64_t(begin) + rows;
    prefix[0] = 0;
    cu_tiles[0] = 0;
    for (int g = 0; g < groups; ++g) {
        const int64_t original_start = wave_prefix[g];
        const int64_t original_end = wave_prefix[g + 1];
        const int64_t start = original_start < begin ? begin : original_start;
        const int64_t stop = original_end > end ? end : original_end;
        const int len = stop > start ? static_cast<int>(stop - start) : 0;
        lens[g] = len;
        prefix[g + 1] = prefix[g] + len;
        tiles[g] = len / tile_m + (len % tile_m != 0);
        cu_tiles[g + 1] = cu_tiles[g] + tiles[g];
    }
}

// Convert only after FP32 accumulation, using baseline CUTLASS RN dtype casts.
template <typename T>
__global__ void offload_cast_kernel(const float* accumulator, T* out, int rows, int hidden) {
    const int64_t n = int64_t(rows) * hidden;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * blockDim.x) {
        out[i] = static_cast<T>(accumulator[i]);
    }
}
}

void moe_offload_count(const int* ids, int64_t routes, int experts, int first, int bins,
                       int* summary, cudaStream_t stream) {
    C10_CUDA_CHECK(cudaMemsetAsync(summary, 0, (bins + 1) * sizeof(int), stream));
    if (routes) offload_count_ids<<<128, 256, 0, stream>>>(ids, routes, experts, first, bins, summary);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_route(fuse_moe::FMOE_params& p, int* invalid, cudaStream_t stream) {
    C10_CUDA_CHECK(cudaMemsetAsync(p.seqlens_ptr, 0, p.num_expert * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(invalid, 0, sizeof(int), stream));
    if (p.total_num_seq) {
        offload_validate_ids<<<128, 256, 0, stream>>>(
            p.topk_ids_ptr, p.total_num_seq, p.num_expert, invalid);
        fuse_moe::count_and_build_indices_async(p.topk_ids_ptr, p.row_indices_ptr,
            p.topk_pos_ptr, p.seqlens_ptr, p.cu_seqlens_ptr, p.tiles_ptr, p.cu_tiles_ptr,
            p.num_seq, p.num_topk, p.num_expert, p.tile_m, false, stream,
            /*explicit_config=*/true);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_compute(fuse_moe::FMOE_params& p, cudaStream_t stream) {
    TORCH_CHECK(arch_targets::gpu_major() >= 8, "offload requires SM80+");
    fuse_moe::moe_compute_sm8x(p, stream, /*explicit_config=*/true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_reduce(fuse_moe::FMOE_params& p, cudaStream_t stream) {
    if (p.is_bf16) {
        fuse_moe::reduce_async(static_cast<cutlass::bfloat16_t*>(p.out_ptr),
            static_cast<const cutlass::bfloat16_t*>(p.down_out_ptr), p.topk_pos_ptr,
            p.topk_scale_ptr, p.num_seq, p.hidden, p.num_topk, false, stream,
            /*explicit_config=*/true);
    } else {
        fuse_moe::reduce_async(static_cast<cutlass::half_t*>(p.out_ptr),
            static_cast<const cutlass::half_t*>(p.down_out_ptr), p.topk_pos_ptr,
            p.topk_scale_ptr, p.num_seq, p.hidden, p.num_topk, false, stream,
            /*explicit_config=*/true);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_scales(const int* topk_pos, const float* scale, float* sorted_scale,
                        int routes, cudaStream_t stream) {
    if (routes <= 0) return;
    offload_scales_kernel<<<128, 256, 0, stream>>>(topk_pos, scale, sorted_scale, routes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_resident_meta(int* meta, int groups, const MoEResidentWave& wave,
                               int64_t gate_bytes, int tile_m, cudaStream_t stream) {
    TORCH_CHECK(groups > 0 && groups <= kOffloadResidentGroups,
        "invalid resident group size");
    offload_resident_meta_kernel<<<1, 1, 0, stream>>>(meta, groups, wave, gate_bytes, tile_m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_tile_meta(const int* wave_meta, int* tile_meta, int wave_groups,
                           int first_group, int groups, int begin, int rows,
                           int tile_m, cudaStream_t stream) {
    if (rows <= 0) return;
    TORCH_CHECK(wave_groups > 0 && wave_groups <= 512 && first_group >= 0 &&
                groups > 0 && first_group <= wave_groups - groups,
                "offload tile metadata requires a valid expert interval within 512 groups");
    TORCH_CHECK(tile_m > 0, "offload tile_m must be positive");
    offload_tile_meta_kernel<<<1, 1, 0, stream>>>(wave_meta, tile_meta, wave_groups,
        first_group, groups, begin, rows, tile_m);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_offload_cast(const float* accumulator, void* out, int rows, int hidden,
                      bool bf16, cudaStream_t stream) {
    if (rows <= 0 || hidden <= 0) return;
    if (bf16) {
        offload_cast_kernel<<<128, 256, 0, stream>>>(
            accumulator, static_cast<cutlass::bfloat16_t*>(out), rows, hidden);
    } else {
        offload_cast_kernel<<<128, 256, 0, stream>>>(
            accumulator, static_cast<cutlass::half_t*>(out), rows, hidden);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    torch::Tensor cu_tiles = torch::empty({num_expert + 1}, options_i32);
    torch::Tensor gate_up_out;  // 仅 FUSE_MOE_TMA=1（TMA 引擎）
    if (std::getenv("FUSE_MOE_TMA") != nullptr) {
        gate_up_out =
            torch::empty({(int64_t)num_seq * num_topk, intermediate2}, x.options());
    }

    // ── 填 params 并分发（tile/PDL/引擎策略全部内封在 launch 层）──────────
    fuse_moe::FMOE_params params{};
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
    params.cu_tiles_ptr = cu_tiles.data_ptr<int>();
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
