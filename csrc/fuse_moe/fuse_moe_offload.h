#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <optional>
#include <string>
#include "common/fuse_moe_params.h"

// Graph-external lifecycle operators; the forward schema declares token mutation.
// max_route_tile_rows: positive fixed cap (default 4096); 0 = budget-adaptive
// FP32 tile-count plateau with opportunistic one-expert admission headroom.
at::Tensor fuse_moe_offload_create(int64_t device, int64_t gpu_budget_bytes,
    int64_t ram_budget_bytes, int64_t pinned_budget_bytes, int64_t num_banks,
    int64_t activation_mode, int64_t max_route_tile_rows, int64_t resident_cache_bytes,
    int64_t dynamic_cache_bytes, int64_t cache_policy, int64_t cache_decay_interval,
    int64_t cache_min_reuse, int64_t cache_update_bytes, bool async_staging,
    bool host_profile, int64_t tile_m, int64_t tile_n);
void fuse_moe_offload_bind(const at::Tensor& token, int64_t layer_id,
    const at::Tensor& w1, const at::Tensor& w2);
void fuse_moe_offload_resident_bind(const at::Tensor& token, int64_t layer_id,
    at::IntArrayRef expert_ids);
// Graph-external preflight for AOTI proxies that do not propagate native errors.
void fuse_moe_offload_validate(const at::Tensor& token, at::IntArrayRef deployment_manifest);
// Versioned CPU int64 status. Sequences count native forward attempts, NOT token
// mutations. Stats describe the last successful forward; consult SuccessSequence.
namespace fuse_moe {
enum class OffloadState : int64_t { Healthy = 0, Recoverable = 1, Poisoned = 2 };
enum class OffloadStatusIndex : int64_t {
    Version = 0, State = 1, Managed = 2, PendingError = 3,
    ForwardSequence = 4, SuccessSequence = 5, FailureLayer = 6,
    FailureSequence = 7, ManagedSequence = 8, SuccessLayer = 9, Count = 10
};
constexpr int64_t kOffloadStatusVersion = 1;
}
// Graph-external only. No mutex is leased across calls: callers must serialize
// validate/begin/graph/check/close with their own wrapper lock. Raw interleaving
// is unsupported; a managed owner-thread check is an additional guard only.
void fuse_moe_offload_begin(const at::Tensor& token);
void fuse_moe_offload_check(const at::Tensor& token);
at::Tensor fuse_moe_offload_status(const at::Tensor& token);
void fuse_moe_offload_acknowledge(const at::Tensor& token);
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
void fuse_moe_offload_test_inject(const at::Tensor& token, const std::string& point,
    int64_t countdown);
at::Tensor fuse_moe_offload_test_worker_states();
#endif
void fuse_moe_offload_close(const at::Tensor& token);
at::Tensor fuse_moe_offload_stats(const at::Tensor& token);
at::Tensor fuse_moe_offload(const at::Tensor& x, const at::Tensor& w1,
    const at::Tensor& w2, const at::Tensor& ids, const at::Tensor& scale,
    at::Tensor token, int64_t layer_id,
    std::optional<at::IntArrayRef> config_argument = std::nullopt,
    std::optional<at::IntArrayRef> manifest_argument = std::nullopt);

// All kernel launches live in the existing CUDA translation unit.
void moe_offload_count(const int* ids, int64_t routes, int experts, int first, int bins,
    int* summary, cudaStream_t stream);
void moe_offload_route(fuse_moe::FMOE_params& p, int* invalid, cudaStream_t stream);
void moe_offload_compute(fuse_moe::FMOE_params& p, cudaStream_t stream);
void moe_offload_reduce(fuse_moe::FMOE_params& p, cudaStream_t stream);
void moe_offload_scales(const int* topk_pos, const float* scale, float* sorted_scale,
    int routes, cudaStream_t stream);
// Passed by value as CUDA launch arguments: no host DMA buffer or lifetime lease.
constexpr int kOffloadResidentGroups = 32;
struct MoEResidentWave {
    const void* weights[kOffloadResidentGroups]{};
    int rows[kOffloadResidentGroups]{};
};
void moe_offload_resident_meta(int* meta, int groups, const MoEResidentWave& wave,
    int64_t gate_bytes, int tile_m, cudaStream_t stream);
void moe_offload_tile_meta(const int* wave_meta, int* tile_meta, int wave_groups,
    int first_group, int groups, int begin, int rows, int tile_m, cudaStream_t stream);
void moe_offload_cast(const float* accumulator, void* out, int rows, int hidden,
    bool bf16, cudaStream_t stream);
