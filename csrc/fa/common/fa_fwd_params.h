/*
 * Flash Attention Forward with Additive Mask — 共享参数结构体
 *
 * FA_mask_params 是 sm89（cp.async 基线）与 sm120（TMA 流水）两条 kernel 路径
 * 共用的 host→device 参数包，与具体架构无关，故置于 common/。
 * 由 fa_fwd_op.cu 在 host 端填充，各架构 kernel 只读。
 */

#pragma once

#include <cstdint>

#include "namespace_config.h"

namespace FA_MASK_NAMESPACE {

// ── 扩展参数结构体 ─────────────────────────────────────────────────────────────
struct FA_mask_params {
    // QKV
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    int64_t q_batch_stride;
    int64_t k_batch_stride;
    int64_t v_batch_stride;
    int64_t q_row_stride;
    int64_t k_row_stride;
    int64_t v_row_stride;
    int64_t q_head_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;

    // Output
    void *__restrict__ o_ptr;
    int64_t o_batch_stride;
    int64_t o_row_stride;
    int64_t o_head_stride;

    // Additive mask: (B, mask_seqlen_q, mask_seqlen_k), row-major，bf16
    // mask_seqlen_q ∈ [seqlen_q, 任意]：q 维无需 pad——越界行的输出会被 epilogue 丢弃，
    //   sm89 用 cute copy_if 谓词跳过越界行，sm120 由 TMA 原生 OOB zero-fill 处理
    // mask_seqlen_k（= mask_row_stride，连续时即列数）∈ {seqlen_k} ∪ [≥seqlen_k_rounded 且对齐]
    //   k 维越界列必须填 -inf（语义必需：可见的越界列会污染 softmax 分母），
    //   由 host 侧 pad 或调用方预 pad 保证
    void *__restrict__ mask_ptr;
    int64_t mask_batch_stride;    // stride over batch dim
    int64_t mask_row_stride;      // stride over q 维 = mask 实际列数
    int mask_seqlen_q;            // mask 实际行数（≥ seqlen_q，q 维无需对齐）

    // Dims
    int b, h, h_k;
    int h_h_k_ratio;              // h / h_k
    int seqlen_q, seqlen_k;
    int seqlen_k_rounded;         // = ceil(seqlen_k, kBlockN) * kBlockN
    int seqlen_q_rounded;         // = ceil(seqlen_q, kBlockM) * kBlockM
    int d;                        // head dim

    float scale_softmax;          // 1 / sqrt(d)
    float scale_softmax_log2;     // log2(e) * scale_softmax

    bool is_bf16;

    // ── Split-KV（num_splits > 1 时有效，sm120 路径使用）────────────────────
    // O_partial: (num_splits, B, H, seqlen_q_rounded, d) bf16，已按各 split 本地 l 归一化
    // LSE_partial: (num_splits, B, H, seqlen_q_rounded) fp32，lse = m*scale + log(l)
    void *__restrict__ oaccum_ptr = nullptr;
    void *__restrict__ lseaccum_ptr = nullptr;
    int num_splits = 1;
};

}  // namespace FA_MASK_NAMESPACE
