// moe_kernels.cuh — fuse_moe 的非 GEMM kernel 集（架构无关，sm80+ 通用）。
// 架构分层：common/（本文件；GEMM 家族见 sm89/group_gemm_sm89.cuh 与
// sm120/group_gemm_sm120.cuh，架构分发见 fuse_moe_launch.h）。
//   1. count/build_indices：按 topk_ids 统计每 expert token 数、构建
//      row_indices（expert 排序后的源 token 行号）与 topk_pos（(token, topk) →
//      排序后行号），并计算 tiles（每 expert 的 kTileM tile 数）。
//      移植自 hpc-ops src/fuse_moe/sm90/cp_async/count.cu，裁剪 task map
//      与 EP 多卡。
//   2. gather_sorted：x → expert 有序 compact 布局（sm120 TMA gemm1 前置）
//   3. act_mul：silu(gate) * up，gate_up (T, 2I) → down 输入 (T, I)
//   4. reduce：按 topk_scale 加权求和 topk 个 expert 输出 → (S, H)
//
// PDL：所有 kernel 体内统一 pdl_acquire()/pdl_release()（低架构编译为空），
// host 侧由 fuse_moe::pdl_supported() 决定是否携带 PDL launch attribute。

#ifndef FUSE_MOE_SRC_COMMON_MOE_KERNELS_CUH_
#define FUSE_MOE_SRC_COMMON_MOE_KERNELS_CUH_

#include <cuda.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

#include "common/fuse_moe_utils.cuh"

namespace fuse_moe {
namespace kernels {

// ══════════════════════════════════════════════════════════════════════════
// 1. count / build_indices（移植自 hpc-ops cp_async/count.cu）
// ══════════════════════════════════════════════════════════════════════════

// 统计每 expert 的 token 数（block 内 smem 直方图 + 全局 atomicAdd 汇聚），
// 同时把 topk_pos 预填 -1。
__global__ void count_seq_kernel(const int *topk_ids_ptr, int *topk_pos_ptr, int *seqlens_ptr,
                                 int total_num_topk, int num_expert) {
  extern __shared__ int seqlens_shm[];
  for (int i = threadIdx.x; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  pdl_acquire();

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < total_num_topk) {
    int iexpert = topk_ids_ptr[idx];
    // 单 GPU：topk_ids ∈ [0, num_expert) 由 op 层保证；越界值直接跳过计数
    if ((iexpert >= 0) && (iexpert < num_expert)) {
      atomicAdd(&seqlens_shm[iexpert], 1);
    }
    topk_pos_ptr[idx] = -1;
  }
  __syncthreads();

  // 每 block 的直方图汇聚到全局缓冲
  for (int i = threadIdx.x; i < num_expert; i += blockDim.x) {
    int cnt = seqlens_shm[i];
    if (cnt > 0) {
      atomicAdd(&seqlens_ptr[i], cnt);
    }
  }

  pdl_release();
}

// 单 block 计算 cu_seqlens / tiles 前缀（cub BlockScan），同时把 seqlens
// 清零供后续 build_indices 作为全局预留计数器复用。
template <int kThreadPerBlock, int kGroupPerThread, int kTileM>
__global__ void count_cuseq_kernel(int *seqlens_ptr, int *cu_seqlens_ptr, int *tiles_ptr,
                                   int num_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int thread_seqs[kGroupPerThread];
  int thread_tiles[kGroupPerThread];

  pdl_acquire();

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_ptr[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }

  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

  // 清零 seqlens，供 build_indices 作为计数器复用
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = 0;
  }

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
  }

  pdl_release();
}

// 构建 row_indices 与 topk_pos：block 内 smem 计数器分配槽位，每 expert 每
// block 一次全局 atomicAdd 预留（保证 expert 内顺序稳定）。
__global__ void build_indices_kernel(const int *topk_ids_ptr, int *row_indices_ptr,
                                     int *topk_pos_ptr, int *seqlens_ptr,  // 复用为计数器
                                     const int *cu_seqlens_ptr, int total_num_topk, int num_topk,
                                     int num_expert, int num_topk_blocks) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ int counter_shm[];
  for (int i = threadIdx.x; i < num_expert; i += blockDim.x) {
    counter_shm[i] = 0;
  }
  __syncthreads();

  pdl_acquire();

  int my_local_expert = -1;
  int my_local_pos = 0;
  int my_token_idx = 0;
  int my_topk_j = 0;
  if (idx < total_num_topk) {
    int iexpert = topk_ids_ptr[idx];
    if ((iexpert >= 0) && (iexpert < num_expert)) {
      my_local_expert = iexpert;
      my_local_pos = atomicAdd(&counter_shm[my_local_expert], 1);
      my_token_idx = idx / num_topk;
      my_topk_j = idx % num_topk;
    }
  }
  __syncthreads();

  // 每 expert 一次全局预留：base = 本 block 在该 expert 区间内的起始槽位
  for (int e = threadIdx.x; e < num_expert; e += blockDim.x) {
    int cnt = counter_shm[e];
    if (cnt > 0) {
      int base = atomicAdd(&seqlens_ptr[e], cnt);
      counter_shm[e] = base;
    }
  }
  __syncthreads();

  if (my_local_expert >= 0) {
    int slot = counter_shm[my_local_expert] + my_local_pos;
    int pos = cu_seqlens_ptr[my_local_expert] + slot;
    row_indices_ptr[pos] = my_token_idx;
    topk_pos_ptr[my_token_idx * num_topk + my_topk_j] = pos;
  }

  pdl_release();
}

// 小批量融合版（单 block）：count → 前缀和 → build 一次完成。
template <int kThreadPerBlock, int kGroupPerThread, int kTileM>
__global__ void count_and_build_kernel(const int *topk_ids_ptr, int *row_indices_ptr,
                                       int *topk_pos_ptr, int *seqlens_ptr, int *cu_seqlens_ptr,
                                       int *tiles_ptr, int num_seq, int num_topk,
                                       int total_num_topk, int num_expert) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // Phase 1: smem 直方图
  extern __shared__ int seqlens_shm[];
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  pdl_acquire();

  for (int i = idx; i < total_num_topk; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= 0) && (iexpert < num_expert)) {
      atomicAdd(&seqlens_shm[iexpert], 1);
    }
    topk_pos_ptr[i] = -1;
  }
  __syncthreads();

  // Phase 2: 前缀和（单 block cub scan；两组：seqs/tiles）
  int thread_seqs[kGroupPerThread];
  int thread_tiles[kGroupPerThread];
#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      int iseq = seqlens_shm[igroup];
      int itile_num = (iseq + kTileM - 1) / kTileM;
      thread_seqs[i] = iseq;
      thread_tiles[i] = itile_num;
      tiles_ptr[igroup] = itile_num;
    } else {
      thread_seqs[i] = 0;
      thread_tiles[i] = 0;
    }
  }

  using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage1;
  __shared__ typename BlockScan::TempStorage temp_storage2;
  int seqs_aggregate, tiles_aggregate;
  BlockScan(temp_storage1).ExclusiveSum(thread_seqs, thread_seqs, seqs_aggregate);
  BlockScan(temp_storage2).ExclusiveSum(thread_tiles, thread_tiles, tiles_aggregate);

#pragma unroll
  for (int i = 0; i < kGroupPerThread; i++) {
    int igroup = idx * kGroupPerThread + i;
    if (igroup < num_expert) {
      cu_seqlens_ptr[igroup] = thread_seqs[i];
    }
  }
  if (idx == 0) {
    cu_seqlens_ptr[num_expert] = seqs_aggregate;
  }

  // Phase 3: 重置 smem 计数器并分配槽位
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_shm[i] = 0;
  }
  __syncthreads();

  for (int i = idx; i < total_num_topk; i += blockDim.x) {
    int iexpert = topk_ids_ptr[i];
    if ((iexpert >= 0) && (iexpert < num_expert)) {
      int slot = atomicAdd(&seqlens_shm[iexpert], 1);
      int pos = cu_seqlens_ptr[iexpert] + slot;
      int token_idx = i / num_topk;
      int topk_j = i % num_topk;
      row_indices_ptr[pos] = token_idx;
      topk_pos_ptr[token_idx * num_topk + topk_j] = pos;
    }
  }
  __syncthreads();

  // 回写 seqlens（下游 GEMM 读取每 expert token 数）
  for (int i = idx; i < num_expert; i += blockDim.x) {
    seqlens_ptr[i] = seqlens_shm[i];
  }

  pdl_release();
}

// ══════════════════════════════════════════════════════════════════════════
// 2. act_mul：silu(gate) * up（bf16/fp16 in/out，fp32 计算，无量化）
// ══════════════════════════════════════════════════════════════════════════
template <typename T, int kThreadPerBlock>
__global__ void act_mul_kernel(T *__restrict__ out_ptr, const T *__restrict__ gate_up_ptr,
                               int num_row, int num_col, cutlass::FastDivmod block1D22D) {
  int iblockx;
  int iblocky;
  block1D22D(iblocky, iblockx, blockIdx.x);
  uint64_t irow = iblocky;
  int icol = (threadIdx.x + iblockx * kThreadPerBlock) * 8;  // 8 elems = 16B

  pdl_acquire();

  if (icol < num_col) {
    const T *gate_row_ptr = gate_up_ptr + irow * num_col * 2;
    const T *up_row_ptr = gate_row_ptr + num_col;
    T *out_row_ptr = out_ptr + irow * num_col;

    auto gate = to<float>(load<T, 8>(gate_row_ptr + icol));
    auto up = to<float>(load<T, 8>(up_row_ptr + icol));

    vec_t<T, 8> out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out[i] = static_cast<T>(silu(gate[i]) * up[i]);
    }
    store(out_row_ptr + icol, out);
  }

  pdl_release();
}

// ══════════════════════════════════════════════════════════════════════════
// 3. reduce：out[s] = Σ_j topk_scale[s,j] * x[topk_pos[s,j]]（移植自
//    hpc-ops reduce.cu，去掉 shared_output；pos==-1（无效槽位）跳过）
// ══════════════════════════════════════════════════════════════════════════
template <typename T, int kThreadPerBlock, int kNumItemPer16B, int kNumTopkMax>
__global__ void reduce_kernel(T *y_ptr, const T *x_ptr, const int *topk_pos_ptr,
                              const float *topk_scale_ptr, int num_seq, int hidden_size,
                              int num_topk, cutlass::FastDivmod block_divider) {
  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  int iblockx;
  int iblocky;
  block_divider(iblocky, iblockx, iblock);
  uint64_t irow = iblocky;
  int icol = (threadIdx.x + iblockx * kThreadPerBlock) * kNumItemPer16B;

  __shared__ int pos_shm[kNumTopkMax];
  __shared__ float scale_shm[kNumTopkMax];

  pdl_acquire();

#pragma unroll 1
  for (int i = idx; i < num_topk; i += blockDim.x) {
    pos_shm[i] = topk_pos_ptr[irow * num_topk + i];
    scale_shm[i] = topk_scale_ptr[irow * num_topk + i];
  }
  __syncthreads();

  if (icol < hidden_size) {
    vec_t<float, kNumItemPer16B> y_fp32;
#pragma unroll
    for (int i = 0; i < kNumItemPer16B; i++) {
      y_fp32[i] = 0.f;
    }

    auto y_irow_ptr = y_ptr + irow * hidden_size;
    for (int i = 0; i < num_topk; i++) {
      int ipos = pos_shm[i];
      float iscale = scale_shm[i];
      if (ipos >= 0) {
        auto x_irow_ptr = x_ptr + static_cast<uint64_t>(ipos) * hidden_size;
        auto x_fp32 = to<float>(load<T, kNumItemPer16B>(x_irow_ptr + icol));
#pragma unroll
        for (int j = 0; j < kNumItemPer16B; j++) {
          y_fp32[j] += x_fp32[j] * iscale;
        }
      }
    }
    auto y_out = to<T>(y_fp32);
    store(y_irow_ptr + icol, y_out);
  }

  pdl_release();
}

// ══════════════════════════════════════════════════════════════════════════
// 4. gather_sorted：x → compact sorted 布局
// ══════════════════════════════════════════════════════════════════════════
// dst[i] = x[row_indices[i]]：按 expert 排序重排激活，目的地即排序位置 i
//（expert 有序 compact 布局）。供 sm120 gemm1 TMA 坐标基平移直读（见
// sm120/group_gemm_sm120.cuh 头注释）——仅当激活 footprint 可 L2 驻留时
// 划算（大 footprint 下 cp.async gather-on-load 更优，见 launch 层分派）。
// 16B 向量化 grid-stride。
template <typename T, bool kUsePDL>
__global__ void gather_sorted_kernel(T *dst_ptr, const T *x_ptr, const int *row_indices_ptr,
                                    const int *cu_seqlens_ptr, int k, int num_expert) {
  pdl_acquire();

  const int64_t chunks_per_row = k / 8;
  const int64_t total_rows = cu_seqlens_ptr[num_expert];
  const int64_t total_chunks = total_rows * chunks_per_row;
  for (int64_t c = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; c < total_chunks;
       c += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = c / chunks_per_row;
    const int64_t col8 = c - row * chunks_per_row;
    const int64_t src_row = row_indices_ptr[row];
    *reinterpret_cast<uint4 *>(dst_ptr + row * k + col8 * 8) =
        *reinterpret_cast<const uint4 *>(x_ptr + src_row * k + col8 * 8);
  }

  pdl_release();
}

}  // namespace kernels

// ══════════════════════════════════════════════════════════════════════════
// host 侧 launcher
// ══════════════════════════════════════════════════════════════════════════

// tile_m 必须与 GEMM 分发所选 kTileM 一致（tiles[] 按 ceil(len/kTileM) 计算，
// GEMM 按其迭代 itile_m）。输出 cu_seqlens（compact 前缀和）与 tiles。
inline void count_and_build_indices_async(const int *topk_ids_ptr, int *row_indices_ptr,
                                          int *topk_pos_ptr, int *seqlens_ptr,
                                          int *cu_seqlens_ptr, int *tiles_ptr, int num_seq,
                                          int num_topk, int num_expert, int tile_m, bool use_pdl,
                                          cudaStream_t stream) {
  constexpr int kThreadPerBlock = 256;
  constexpr int kGroupPerThread = 2;
  int total_num_topk = num_seq * num_topk;

  if (num_seq <= 128) {
    // 小批量：单 block 融合（topk_ids ≤ 128*128 = 16384 项）
    dim3 grid(1);
    dim3 block(kThreadPerBlock);
    size_t smem = num_expert * sizeof(int);
    if (tile_m <= 32) {
      auto kernel = kernels::count_and_build_kernel<kThreadPerBlock, kGroupPerThread, 32>;
      launch_kernel_pdl(kernel, grid, block, smem, stream, use_pdl, topk_ids_ptr,
                        row_indices_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr,
                        num_seq, num_topk, total_num_topk, num_expert);
    } else if (tile_m >= 128) {
      auto kernel = kernels::count_and_build_kernel<kThreadPerBlock, kGroupPerThread, 128>;
      launch_kernel_pdl(kernel, grid, block, smem, stream, use_pdl, topk_ids_ptr,
                        row_indices_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr,
                        num_seq, num_topk, total_num_topk, num_expert);
    } else {
      auto kernel = kernels::count_and_build_kernel<kThreadPerBlock, kGroupPerThread, 64>;
      launch_kernel_pdl(kernel, grid, block, smem, stream, use_pdl, topk_ids_ptr,
                        row_indices_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr, tiles_ptr,
                        num_seq, num_topk, total_num_topk, num_expert);
    }
    return;
  }

  // Step 1: count
  {
    dim3 grid((total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock);
    dim3 block(kThreadPerBlock);
    size_t smem = num_expert * sizeof(int);
    launch_kernel_pdl(kernels::count_seq_kernel, grid, block, smem, stream, use_pdl, topk_ids_ptr,
                      topk_pos_ptr, seqlens_ptr, total_num_topk, num_expert);
  }

  // Step 2: 前缀和（tile_m 分发）
  {
    dim3 grid(1);
    dim3 block(kThreadPerBlock);
    if (tile_m <= 32) {
      auto kernel = kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, 32>;
      launch_kernel_pdl(kernel, grid, block, 0, stream, use_pdl, seqlens_ptr, cu_seqlens_ptr,
                        tiles_ptr, num_expert);
    } else if (tile_m >= 128) {
      auto kernel = kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, 128>;
      launch_kernel_pdl(kernel, grid, block, 0, stream, use_pdl, seqlens_ptr, cu_seqlens_ptr,
                        tiles_ptr, num_expert);
    } else {
      auto kernel = kernels::count_cuseq_kernel<kThreadPerBlock, kGroupPerThread, 64>;
      launch_kernel_pdl(kernel, grid, block, 0, stream, use_pdl, seqlens_ptr, cu_seqlens_ptr,
                        tiles_ptr, num_expert);
    }
  }

  // Step 3: build indices
  {
    int num_topk_blocks = (total_num_topk + kThreadPerBlock - 1) / kThreadPerBlock;
    dim3 grid(num_topk_blocks);
    dim3 block(kThreadPerBlock);
    size_t smem = num_expert * sizeof(int);
    launch_kernel_pdl(kernels::build_indices_kernel, grid, block, smem, stream, use_pdl,
                      topk_ids_ptr, row_indices_ptr, topk_pos_ptr, seqlens_ptr, cu_seqlens_ptr,
                      total_num_topk, num_topk, num_expert, num_topk_blocks);
  }
}

template <typename T>
inline void act_mul_async(T *out_ptr, const T *gate_up_ptr, int num_row, int num_col,
                          bool use_pdl, cudaStream_t stream) {
  constexpr int kThreadPerBlock = 256;
  int num_block_col = (num_col / 8 + kThreadPerBlock - 1) / kThreadPerBlock;
  int num_block_total = num_row * num_block_col;
  if (num_block_total <= 0) return;
  cutlass::FastDivmod block1D22D(num_block_col);
  dim3 grid(num_block_total);
  dim3 block(kThreadPerBlock);
  launch_kernel_pdl(kernels::act_mul_kernel<T, kThreadPerBlock>, grid, block, 0, stream, use_pdl,
                    out_ptr, gate_up_ptr, num_row, num_col, block1D22D);
}

// ── gather（compact sorted 布局）───────────────────────────────────────────
// grid 压缩到 SM 数（PDL 链上游 grid 过大会被下游提前占位饿死——经典 PDL
// 死锁；循环本身是 grid-stride，任意 grid 均正确）。
template <typename T>
inline void gather_sorted_async(T *dst_ptr, const T *x_ptr, const int *row_indices_ptr,
                                const int *cu_seqlens_ptr, int k, int num_expert, bool use_pdl,
                                cudaStream_t stream) {
  constexpr int kThreadPerBlock = 256;
  dim3 grid(get_sm_count());
  dim3 block(kThreadPerBlock);
  launch_kernel_pdl(kernels::gather_sorted_kernel<T, true>, grid, block, 0, stream, use_pdl,
                    dst_ptr, x_ptr, row_indices_ptr, cu_seqlens_ptr, k, num_expert);
}

template <typename T>
inline void reduce_async(T *y_ptr, const T *x_ptr, const int *topk_pos_ptr,
                         const float *topk_scale_ptr, int num_seq, int hidden_size, int num_topk,
                         bool use_pdl, cudaStream_t stream) {
  constexpr int kThreadPerBlock = 256;
  constexpr int kNumTopkMax = 128;
  constexpr int kNumItemPer16B = 16 / sizeof(T);  // 8
  int num_block_col = (hidden_size / kNumItemPer16B + kThreadPerBlock - 1) / kThreadPerBlock;
  int num_block_total = num_seq * num_block_col;
  if (num_block_total <= 0) return;
  cutlass::FastDivmod block_divider(num_block_col);
  dim3 grid(num_block_total);
  dim3 block(kThreadPerBlock);
  launch_kernel_pdl(kernels::reduce_kernel<T, kThreadPerBlock, kNumItemPer16B, kNumTopkMax>, grid,
                    block, 0, stream, use_pdl, y_ptr, x_ptr, topk_pos_ptr, topk_scale_ptr, num_seq,
                    hidden_size, num_topk, block_divider);
}

}  // namespace fuse_moe

#endif  // FUSE_MOE_SRC_COMMON_MOE_KERNELS_CUH_
