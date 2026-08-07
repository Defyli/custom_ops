/**
 * jagged_forward.cu — jagged_pool_and_collect 算子实现
 *
 * ── 职责边界 ─────────────────────────────────────────────────────────────────
 *   本算子 **只负责 pooling 特征** 的批量 segment reduce，不再处理透传特征。
 *
 *   透传特征（173 个）由 Python 侧直接 dict 赋值，zero-copy，无需 cat/kernel。
 *   这样避免了旧设计中对全量 185 个特征做 torch.cat 的高开销：
 *     旧版：cat(185 特征) × 2 + C++ 内 cat(pass+pool 输出) × 2
 *     新版：只对 12 个 pooling 特征 cat × 2（输入），输出直接 narrow 切分
 *
 * ── Pooling 路径 ─────────────────────────────────────────────────────────────
 *   12 个特征全部是 mean pooling，dim=64，item_num 相同（N_i 相同）。
 *   输入: pool_values shape (total_pool_rows, D)，pool_lengths shape (total_pool_len_rows,)
 *         两者均为调用方 cat(12 个 pooling 特征) 后的结果。
 *   目标: 对每个特征的每个 sample 做 mean，输出 (total_out_rows, D)。
 *
 *   CUDA Kernel: batched_segment_reduce_bf16
 *   ─────────────────────────────────────────
 *   每个 warp 处理一个输出行（即一个 sample 在一个特征上的 pooling 结果）。
 *   Lane 在 seq 维度 stride=WARP_SIZE 分工，cub::WarpReduceSum 归约。
 *
 *   Grid/Block: (ceil(total_out_rows / ROWS_PER_BLOCK),)，blockDim=(WARP_SIZE, ROWS_PER_BLOCK)
 *
 * ── 接口设计（v4：栈上 offset，零 malloc 版本）──────────────────────────────
 *   Python 侧只传 pool_val_splits / pool_len_splits（CPU list）。
 *   C++ 内部用 cudaMemcpyAsync 将栈上数组（最多 MAX_POOL_FEATS+1 个 int64）直接
 *   copy 到 GPU 的预分配 buffer 中，完全消除 Python 侧的 H→D copy 和 tensor alloc。
 *
 *   由于 n_pooling 最大约 64（生产 12），偏移数组极小，用 __constant__ 内存传递
 *   比动态 tensor alloc 更快。
 *
 *   实际上更简单：对于 n_pooling <= MAX_INLINE_POOL(=64) 的情况，
 *   直接将 pool_val_splits / pool_len_splits 作为 kernel 参数（通过 device pointer）。
 *   用 cudaMalloc + cudaFree 开销太大；改为用 cudaMemcpyAsync 写入预分配的 small buffer。
 *   但最简单的方案：直接在 GPU 上 inline allocate 一次性 int64 buffer（极小，
 *   torch::empty 比 cudaMalloc 轻量）。
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <vector>
#include "jagged_forward.h"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: batched_segment_reduce_bf16
//
// 对 n_pooling 个特征批量做 segment reduce（mean 或 sum），输入/输出均为 bfloat16。
// 累加在 float 中间精度完成，避免 bf16 精度损失。
//
// 并行策略（WarpReduce 优化版）
// ──────────────────────────────
// 每个输出行对应一个 warp（32 线程），warp 内的 lane 在 seq 维度上分工：
//   lane k 负责累加 pool_values[val_base + k, :], [val_base + k+32, :], ...
// 最后通过 cub::WarpReduceSum 归约，lane 0 写回结果。
//
// 外层 D 维度由 d 循环在每个 warp 内串行处理（适合 D=64 的生产配置）。
//
// Grid/Block layout:
//   blockDim = (WARP_SIZE, ROWS_PER_BLOCK)
//     threadIdx.x = lane_id (0..31)，对应 seg 内 row 分工
//     threadIdx.y = 本 block 内的输出行槽位 (0..ROWS_PER_BLOCK-1)
//   gridDim.x  = ceil(total_out_rows / ROWS_PER_BLOCK)
//
// ─────────────────────────────────────────────────────────────────────────────
#define WARP_SIZE      32
#define ROWS_PER_BLOCK 4    // 每个 block 处理的输出行数；WARP_SIZE * ROWS_PER_BLOCK = 128 threads

__global__ void batched_segment_reduce_bf16(
    const __nv_bfloat16* __restrict__ pool_values,      // (total_pool_val_rows, D)
    const int32_t*       __restrict__ pool_lengths,     // (total_pool_len_rows,)
    const int64_t*       __restrict__ feat_val_offsets, // (n_pool+1,)
    const int64_t*       __restrict__ feat_len_offsets, // (n_pool+1,)
    const int64_t*       __restrict__ feat_out_offsets, // (n_pool+1,)
    __nv_bfloat16*       __restrict__ out,              // (total_out_rows, D)
    int64_t n_pooling,
    int64_t D,
    int32_t reduce_mode                                 // 0=mean, 1=sum
) {
    // 每个 warp 对应一个输出行
    const int64_t global_row = (int64_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    const int32_t lane       = threadIdx.x;   // 0..31

    const int64_t total_out_rows = feat_out_offsets[n_pooling];
    if (global_row >= total_out_rows) return;

    // 定位 global_row 所属 feature（n_pooling 很小，线性扫描即可）
    int64_t feat_idx = 0;
    while (feat_idx < n_pooling - 1
           && global_row >= feat_out_offsets[feat_idx + 1]) {
        ++feat_idx;
    }
    const int64_t sample_idx = global_row - feat_out_offsets[feat_idx];
    const int64_t len_base   = feat_len_offsets[feat_idx];
    const int32_t seg_len    = pool_lengths[len_base + sample_idx];
    const int64_t val_base   = feat_val_offsets[feat_idx] + (int64_t)sample_idx * seg_len;

    // cub::WarpReduce：每个 warp 独立归约（WARP_SIZE 个 thread）
    using WarpReduceF = cub::WarpReduce<float>;
    __shared__ typename WarpReduceF::TempStorage temp[ROWS_PER_BLOCK];

    // 在 D 维度上循环：每次处理一个 channel d
    for (int64_t d = 0; d < D; ++d) {
        // lane 负责 seg 内 stride=WARP_SIZE 的 rows
        float acc = 0.0f;
        for (int32_t k = lane; k < seg_len; k += WARP_SIZE) {
            acc += __bfloat162float(pool_values[(val_base + k) * D + d]);
        }
        // warp 内归约：所有 lane 的 acc 求和，结果在 lane 0
        float total = WarpReduceF(temp[threadIdx.y]).Sum(acc);

        if (lane == 0) {
            if (reduce_mode == 0 && seg_len > 0) {  // mean
                total /= (float)seg_len;
            }
            out[(feat_out_offsets[feat_idx] + sample_idx) * D + d] = __float2bfloat16(total);
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// CUDA 主入口（v4：cudaMemcpyAsync 上传 small offset 数组，零 tensor alloc）
//
// 性能优化历程：
//   v1: Python 侧 3 次 non-blocking H→D copy（_feat_val/len/out_offsets.copy_）
//   v2: 消除透传特征，cat(12 pooling) × 2
//   v3: C++ 内 torch::tensor(...).to(device) — 引入 tensor alloc 开销
//   v4: 用 cudaMemcpyAsync + 预分配小 buffer 替代 tensor alloc
//       对于 n_pooling=12，3 个 offset 数组共 3×13×8=312 bytes，
//       cudaMemcpyAsync 几乎没有开销（小于 1us）。
//
// 接口：无 offset tensor 参数，C++ 内部自行构造。
// ─────────────────────────────────────────────────────────────────────────────

// 最大内联 pooling 特征数（>=生产最大值，此时用 cudaMemcpyAsync 路径）
#define MAX_INLINE_POOL 128

std::vector<torch::Tensor> jagged_pool_and_collect_cuda(
    const torch::Tensor& pool_values,             // (total_pool_val_rows, D)，仅 pooling 特征
    const torch::Tensor& pool_lengths,            // (total_pool_len_rows,)，仅 pooling 特征
    const std::vector<int64_t>& pool_val_splits,  // (n_pooling+1,) CPU
    const std::vector<int64_t>& pool_len_splits,  // (n_pooling+1,) CPU
    int64_t n_pooling,
    int64_t reduce_mode,
    const torch::Tensor& ones_cache               // (max_ones,) int32 CUDA，预分配全 1
) {
    TORCH_CHECK(pool_values.is_cuda(),   "pool_values must be on CUDA");
    TORCH_CHECK(pool_lengths.is_cuda(),  "pool_lengths must be on CUDA");
    TORCH_CHECK(ones_cache.is_cuda(),    "ones_cache must be on CUDA");
    TORCH_CHECK(pool_values.dim() == 2,  "pool_values must be 2D (total_rows, D)");
    TORCH_CHECK(pool_lengths.dim() == 1, "pool_lengths must be 1D");
    TORCH_CHECK((int64_t)pool_val_splits.size() == n_pooling + 1, "pool_val_splits size mismatch");
    TORCH_CHECK((int64_t)pool_len_splits.size() == n_pooling + 1, "pool_len_splits size mismatch");

    // 隐式 dtype 兼容：上游（SparseArch）可能以 float32 传入，CUDA kernel 只处理 bfloat16。
    const torch::Tensor& values_bf16 =
        (pool_values.scalar_type() == at::ScalarType::BFloat16)
        ? pool_values
        : pool_values.to(at::ScalarType::BFloat16);

    const int64_t D = values_bf16.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    // ── 计算输出 splits（CPU 侧）────────────────────────────────────────────
    // pooling 特征：输出 rows = len_rows（sample 数，每 sample pooled 后 1 行）
    std::vector<int64_t> out_val_splits(n_pooling + 1, 0);
    std::vector<int64_t> out_len_splits(n_pooling + 1, 0);
    for (int64_t i = 0; i < n_pooling; ++i) {
        const int64_t n_samples = pool_len_splits[i + 1] - pool_len_splits[i];
        out_val_splits[i + 1] = out_val_splits[i] + n_samples;
        out_len_splits[i + 1] = out_len_splits[i] + n_samples;
    }
    const int64_t total_pool_out_rows = out_val_splits[n_pooling];

    // ── 分配输出 buffer ────────────────────────────────────────────────────
    torch::Tensor pool_values_out  = torch::empty({total_pool_out_rows, D}, values_bf16.options());

    // pooled lengths 全是 1，用 ones_cache 切片（zero-copy）
    TORCH_CHECK(ones_cache.size(0) >= total_pool_out_rows,
        "ones_cache too small: need ", total_pool_out_rows, " got ", ones_cache.size(0));
    torch::Tensor pool_lengths_out = ones_cache.narrow(0, 0, total_pool_out_rows);

    if (n_pooling > 0 && total_pool_out_rows > 0) {
        // ── 构造 GPU 端 offset buffer ─────────────────────────────────────
        // 使用 cudaMemcpyAsync 将栈上的 offset 数组上传到 GPU，
        // 比 torch::tensor(...).to(device) 少 2 次 malloc/free，几乎零开销。
        //
        // 布局：将 feat_val_offsets / feat_len_offsets / feat_out_offsets
        //       紧凑打包到一块 (3, n_pooling+1) 的 int64 buffer：
        //         [0..n_pool]   feat_val_offsets = pool_val_splits
        //         [n_pool+1..2*(n_pool+1)-1] feat_len_offsets = pool_len_splits
        //         [2*(n_pool+1)..3*(n_pool+1)-1] feat_out_offsets = out_val_splits
        const int64_t stride = n_pooling + 1;
        TORCH_CHECK(stride <= MAX_INLINE_POOL + 1,
            "n_pooling=", n_pooling, " exceeds MAX_INLINE_POOL=", MAX_INLINE_POOL);

        // 分配 3 个 offset 数组的连续 GPU buffer（一次分配）
        auto offset_buf = torch::empty({3 * stride},
            torch::TensorOptions().dtype(torch::kInt64).device(pool_values.device()));
        int64_t* d_offsets = offset_buf.data_ptr<int64_t>();

        // 栈上 host buffer（避免 malloc）
        int64_t h_offsets[3 * (MAX_INLINE_POOL + 1)];
        for (int64_t i = 0; i <= n_pooling; ++i) {
            h_offsets[i]              = pool_val_splits[i];   // feat_val_offsets
            h_offsets[stride + i]     = pool_len_splits[i];   // feat_len_offsets
            h_offsets[2 * stride + i] = out_val_splits[i];    // feat_out_offsets
        }

        // 异步上传（与后续 kernel 在同一 stream 上，kernel 启动时数据已就绪）
        cudaMemcpyAsync(d_offsets, h_offsets,
                        3 * stride * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);

        const __nv_bfloat16* pool_val_ptr =
            reinterpret_cast<const __nv_bfloat16*>(values_bf16.data_ptr<at::BFloat16>());
        const int32_t* pool_len_ptr = pool_lengths.data_ptr<int32_t>();
        __nv_bfloat16* out_ptr =
            reinterpret_cast<__nv_bfloat16*>(pool_values_out.data_ptr<at::BFloat16>());

        const dim3 block(WARP_SIZE, ROWS_PER_BLOCK);
        const int  grid = (int)((total_pool_out_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

        batched_segment_reduce_bf16<<<grid, block, 0, stream>>>(
            pool_val_ptr,
            pool_len_ptr,
            d_offsets,                // feat_val_offsets
            d_offsets + stride,       // feat_len_offsets
            d_offsets + 2 * stride,   // feat_out_offsets
            out_ptr,
            n_pooling,
            D,
            (int32_t)reduce_mode
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ── 返回输出 splits（CPU Tensor，Python 侧按此切分）──────────────────────
    auto opts_i64_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto t_out_val_splits = torch::tensor(out_val_splits, opts_i64_cpu);
    auto t_out_len_splits = torch::tensor(out_len_splits, opts_i64_cpu);

    return {pool_values_out, pool_lengths_out, t_out_val_splits, t_out_len_splits};
}


// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback 实现（用于无 GPU 的单元测试）
// 只处理 pooling 特征，接口与 CUDA 版保持一致。
// ─────────────────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> jagged_pool_and_collect_cpu(
    const torch::Tensor& pool_values,             // (total_pool_val_rows, D)
    const torch::Tensor& pool_lengths,            // (total_pool_len_rows,)
    const std::vector<int64_t>& pool_val_splits,  // (n_pooling+1,)
    const std::vector<int64_t>& pool_len_splits,  // (n_pooling+1,)
    int64_t n_pooling,
    int64_t reduce_mode,
    const torch::Tensor& ones_cache               // 忽略（CPU fallback 不使用）
) {
    (void)ones_cache;

    const int64_t D = pool_values.size(1);

    // 计算输出 splits
    std::vector<int64_t> out_val_splits(n_pooling + 1, 0);
    std::vector<int64_t> out_len_splits(n_pooling + 1, 0);
    for (int64_t i = 0; i < n_pooling; ++i) {
        const int64_t n_samp = pool_len_splits[i + 1] - pool_len_splits[i];
        out_val_splits[i + 1] = out_val_splits[i] + n_samp;
        out_len_splits[i + 1] = out_len_splits[i] + n_samp;
    }
    const int64_t total_out = out_val_splits[n_pooling];

    auto out_values  = torch::empty({total_out, D}, pool_values.options());
    auto out_lengths = torch::ones({total_out}, pool_lengths.options());

    // pooling：用 PyTorch CPU ops 实现（参考实现，非性能优先）
    for (int64_t i = 0; i < n_pooling; ++i) {
        const int64_t vs     = pool_val_splits[i], ve = pool_val_splits[i + 1];
        const int64_t ls     = pool_len_splits[i], le = pool_len_splits[i + 1];
        const int64_t n_samp = le - ls;
        const int64_t ovs    = out_val_splits[i];

        auto lengths_i = pool_lengths.narrow(0, ls, n_samp);
        auto values_i  = pool_values.narrow(0, vs, ve - vs);

        const int64_t seq_len = (n_samp > 0) ? (int64_t)lengths_i[0].item<int32_t>() : 1;
        auto reshaped = values_i.view({n_samp, seq_len, D});
        torch::Tensor pooled;
        if (reduce_mode == 0) {
            pooled = reshaped.mean(1);
        } else {
            pooled = reshaped.sum(1);
        }
        out_values.narrow(0, ovs, n_samp).copy_(pooled);
    }

    auto opts_i64_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto t_out_val_splits = torch::tensor(out_val_splits, opts_i64_cpu);
    auto t_out_len_splits = torch::tensor(out_len_splits, opts_i64_cpu);

    return {out_values, out_lengths, t_out_val_splits, t_out_len_splits};
}
