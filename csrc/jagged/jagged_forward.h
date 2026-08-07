#pragma once

#include <torch/extension.h>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
// jagged_pool_and_collect — 只负责 pooling 特征的批量 segment reduce
//
// 职责边界
// ─────────
// 本算子 **只处理 pooling 特征**，不涉及透传特征。
// 透传特征由 Python 侧直接 dict 赋值（zero-copy），无需经过本算子。
//
// 接口设计（v3：零 H→D copy 版本）
// ─────────────────────────────────
// 调用方将 n_pooling 个 pooling 特征的 values/lengths cat 成两个大 Tensor 传入，
// 用 pool_val_splits / pool_len_splits 记录各特征的起始行偏移。
// C++ 内部直接从这两个 CPU list 构造 GPU offset tensor，
// 完全消除 Python 侧的 3 次 H→D copy（_feat_val/len/out_offsets.copy_）。
// 算子输出 pooling 后的 values（每特征每 sample 一行）和全 1 的 lengths。
//
// 参数
// ─────
// pool_values      : (total_pool_val_rows, D) — 所有 pooling 特征 values 拼接（float or bf16）
// pool_lengths     : (total_pool_len_rows,) int32 — 所有 pooling 特征 lengths 拼接
// pool_val_splits  : (n_pooling+1,) int64 — values 的行偏移
// pool_len_splits  : (n_pooling+1,) int64 — lengths 的行偏移
// n_pooling        : int — pooling 特征数量
// reduce_mode      : 0=mean, 1=sum
// ones_cache       : (max_n,) int32 CUDA — 预分配的全 1 Tensor，用于 pooled lengths，避免 alloc
//
// 返回
// ─────
// pool_values_out  : (total_pool_out_rows, D) — pooling 结果
// pool_lengths_out : (total_pool_out_rows,) int32 — 全为 1（来自 ones_cache）
// out_val_splits   : (n_pooling+1,) int64 CPU — 切分 pool_values_out 的 offsets
// out_len_splits   : (n_pooling+1,) int64 CPU — 切分 pool_lengths_out 的 offsets
// ──────────────────────────────────────────────────────────────────────────────

// CUDA 实现（jagged_forward.cu）
std::vector<torch::Tensor> jagged_pool_and_collect_cuda(
    const torch::Tensor& pool_values,             // (total_pool_val_rows, D)
    const torch::Tensor& pool_lengths,            // (total_pool_len_rows,) int32
    const std::vector<int64_t>& pool_val_splits,  // (n_pooling+1,)
    const std::vector<int64_t>& pool_len_splits,  // (n_pooling+1,)
    int64_t n_pooling,
    int64_t reduce_mode,                          // 0=mean, 1=sum
    const torch::Tensor& ones_cache               // (max_n,) int32 CUDA
);

// CPU fallback 实现（用于单测）
std::vector<torch::Tensor> jagged_pool_and_collect_cpu(
    const torch::Tensor& pool_values,
    const torch::Tensor& pool_lengths,
    const std::vector<int64_t>& pool_val_splits,
    const std::vector<int64_t>& pool_len_splits,
    int64_t n_pooling,
    int64_t reduce_mode,
    const torch::Tensor& ones_cache               // CPU fallback 忽略
);
