/*
 * Flash Attention Forward with Additive Mask — Core Compute Kernel
 *
 * 从 FA2 的 compute_attn_1rowblock 移植，专为 prefill（非 causal）+ 外部 mask 场景精简：
 *   - 删除 dropout / rotary / KV-cache / alibi / local window / softcap 逻辑
 *   - 删除 split-KV 路径
 *   - 新增：通过 cute GmemTiledCopyMask + smem 缓冲读取 fp16/bf16 additive mask，
 *           在 gemm 后对 acc_s 做 smem→reg copy + 逐元素加法（零分支，无 warp divergence）
 *   - 保留 Is_even_MN / Is_even_K 分支以保证边界正确性
 *
 * Mask 语义（加法 mask，与 SDPA 对齐，即 softmax(S·scale + mask)）：
 *   mask=0    → 可见（score 不变）
 *   mask=-inf → 屏蔽（score → -inf，softmax 后 weight = 0）
 *   有限值    → 任意加法偏置（ALiBi 风格）；实现上 mask 先于 scale 加到 acc_s，
 *               应用点乘 1/scale_softmax 预还原（见 apply_mask_from_smem）
 *
 * Mask tensor 的 global mem 格式：(B, mask_seqlen_q, mask_seqlen_k)，row-major，fp16/bf16（与输入同 dtype）
 *   - q 维：mask_seqlen_q ∈ [seqlen_q, 任意]，无需 pad——copy_g2s_mask 用 cute copy_if
 *     按行谓词跳过越界行（这些行的输出反正被 epilogue 丢弃）
 *   - k 维：mask_seqlen_k ∈ [seqlen_k, 任意] 且 %8==0（128-bit cp.async 行对齐），
 *     无需 pad 到 kBlockN——边界 tile 的越界列由列谓词跳过拷贝、smem 预清 -inf
 *     （语义：col ≥ Sk 恒为屏蔽，mask 越界列内容被忽略）
 *   - 通过 params.mask_ptr / mask_batch_stride / mask_row_stride / mask_seqlen_q / mask_seqlen_k 寻址
 *
 * copy_g2s_mask 行/列联合谓词（Sk%8==0 保证 128-bit 向量不跨 Sk 边界）：
 *   行坐标 >= params.mask_seqlen_q - m_block*kBlockM 的 copy 向量被 copy_if 跳过，
 *   对应 smem 行在 prologue 一次性清 0（防止未初始化 smem 的 NaN 垃圾模式）。
 *   列坐标 >= Sk - n_block*kBlockN 的向量（仅全局边界 tile 非平凡）同样跳过，
 *   对应 smem 列一次性预清 -inf（softmax 语义屏蔽位）。清零与 cp.async 写入位置
 *   不重叠（谓词为假的向量从不被写入），无 race；非边界 tile 的整块拷贝发生在
 *   边界 tile 消费完成之后（迭代间 __syncthreads 分隔），覆写安全。
 *   Is_even_MN（Sk%kBlockN==0 且 Sq%kBlockM==0）时列谓词编译期全部消除。
 *
 * Smem 布局：
 *   [sQ (kBlockM × kHeadDim, swizzled)]
 *   [sK (kBlockN × kHeadDim, swizzled)]
 *   [sV (kBlockN × kHeadDim, swizzled)]
 *   [sMask (kBlockM × kBlockN, swizzled, 1 stage)]
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

#include <c10/cuda/CUDAException.h>   // C10_CUDA_CHECK（launcher 的 cudaFuncSetAttribute 检查）

#include "fa_fwd_mask.h"

// 复用 FA2 的工具函数（softmax、gemm、copy、convert_type 等）
#include "../common/utils.h"
#include "../common/softmax.h"
// 与 sm120 共享的参数包
#include "../common/fa_fwd_params.h"

namespace FA_MASK_NAMESPACE {

using namespace cute;

// FA_mask_params 定义见 common/fa_fwd_params.h（sm89/sm120 共用）

// ── 核心计算函数 ───────────────────────────────────────────────────────────────
// Kernel_traits: FA_mask_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, ...>
// Is_even_MN: seqlen_q/seqlen_k 都是 kBlockM/kBlockN 的倍数
// Is_even_K:  d == kHeadDim
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__forceinline__ __device__ void compute_attn_1rowblock_mask(
    const FA_mask_params &params,
    const int bidb,   // batch index
    const int bidh,   // head index
    const int m_block // Q tile index（行方向）
) {
    using Element      = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t      = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kNWarps   = Kernel_traits::kNWarps;

    // ── 有效序列长度 ──────────────────────────────────────────────────────────
    const int actual_seqlen_q = params.seqlen_q;
    const int actual_seqlen_k = params.seqlen_k;

    if (m_block * kBlockM >= actual_seqlen_q) return;

    const int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);

    // ── Global memory 指针：Q / K / V ────────────────────────────────────────
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + bidb * params.q_batch_stride),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{})
    );
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + bidb * params.k_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{})
    );
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + bidb * params.v_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{})
    );
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    // ── Shared memory 排布 ───────────────────────────────────────────────────
    // [sQ] [sK] [sV] [sMask]
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    // sMask：紧跟 sV 之后，1 stage，(kBlockM, kBlockN)
    Tensor sMask = make_tensor(sV.data() + size(sV),
                               typename Kernel_traits::SmemLayoutMask{});

    // ── Gmem -> Smem copy handles（QKV）──────────────────────────────────────
    typename Kernel_traits::GmemTiledCopyQKV  gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // ── Gmem -> Smem copy handles（Mask）──────────────────────────────────
    // 参照 hstu_mask.h：独立的 GmemTiledCopyMask，128-bit cp.async
    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask = gmem_tiled_copy_Mask.get_thread_slice(tidx);

    // gMask：全局 mask tensor for this batch，用 local_tile 按 n_block 索引
    // 行数 = mask 实际行数（≥ seqlen_q，无需对齐）；越界行的 copy 由行谓词跳过。
    // 列数 = mask 实际列数（≥ seqlen_k 且 %8==0，无需对齐到 kBlockN）；边界 tile 的
    // 越界列向量由列谓词跳过（见 copy_g2s_mask），不发生 gmem 越界读
    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.mask_ptr)
                      + bidb * params.mask_batch_stride),
        make_shape(params.mask_seqlen_q, params.mask_seqlen_k),
        make_stride(params.mask_row_stride, _1{})
    );
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, _));

    Tensor tQgMask = gmem_thr_copy_Mask.partition_S(gMask);
    // sMask 是 3D (kBlockM, kBlockN, kStages=1)，取 stage=0 的 2D slice 再做 partition，
    // 避免 4D partition 时内部嵌套 rank 造成 CopyAtom rank-mismatch 编译错误
    auto sMask_s0 = sMask(_, _, _0{});  // 2D: (kBlockM, kBlockN)
    Tensor tQsMask = gmem_thr_copy_Mask.partition_D(sMask_s0);  // (COPY_V, COPY_M, COPY_N)

    // ── mask 行谓词（q 维不 pad）：行坐标 >= mask 实际剩余行数的 copy 向量被跳过 ──
    // kMaskQFull（host 保证 mask_seqlen_q % kBlockM == 0）时编译期裁掉谓词，纯 copy
    Tensor cMask   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tMcMask = gmem_thr_copy_Mask.partition_S(cMask);      // (CPY_V, CPY_M, CPY_N) 坐标
    Tensor tMpMask = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
    const int mask_rows_left = params.mask_seqlen_q - m_block * kBlockM;
    if constexpr (!Kernel_traits::kMaskQFull) {
        #pragma unroll
        for (int m = 0; m < size<0>(tMpMask); ++m) {
            #pragma unroll
            for (int n = 0; n < size<1>(tMpMask); ++n) {
                tMpMask(m, n) = get<0>(tMcMask(_0{}, m, n)) < mask_rows_left;
            }
        }
        // 边界 CTA：谓词跳过的 smem 行一次性清 0（这些行的输出被 epilogue 丢弃，清 0 仅避免
        // 未初始化 smem 可能出现的 NaN 位模式；与 cp.async 写入位置不重叠，无 race）
        if (mask_rows_left < kBlockM) {
            #pragma unroll
            for (int m = 0; m < size<1>(tQsMask); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tQsMask); ++n) {
                    if (!tMpMask(m, n)) { clear(tQsMask(_, m, n)); }
                }
            }
        }
    }

    // ── mask 列谓词（k 维不 pad）：Sk 非 kBlockN 倍数时，全局边界 tile 的越界列 ───
    // 由 copy 列谓词跳过拷贝（见 copy_g2s_mask），其 smem 一次性预清 -inf——softmax
    // 语义屏蔽位（col ≥ Sk 恒为 -inf）。Sk%8==0 → 128-bit 向量整段落在界内/界外，
    // 向量粒度谓词即可。与 cp.async 写入位置不重叠（谓词为假的向量从不被写入）；
    // 非边界 tile 的整块拷贝在边界 tile 消费之后（迭代间 __syncthreads 分隔）。
    // Is_even_MN（Sk%kBlockN==0）时编译期消除。
    if constexpr (!Is_even_MN) {
        const int mask_k_tail = actual_seqlen_k % kBlockN;
        if (mask_k_tail != 0) {
            const Element mask_neg_inf(static_cast<float>(-INFINITY));
            #pragma unroll
            for (int m = 0; m < size<1>(tQsMask); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tQsMask); ++n) {
                    if (get<1>(tMcMask(_0{}, m, n)) >= mask_k_tail) {
                        #pragma unroll
                        for (int v = 0; v < size<0>(tQsMask); ++v) {
                            tQsMask(v, m, n) = mask_neg_inf;
                        }
                    }
                }
            }
        }
    }

    // ── MMA handles ─────────────────────────────────────────────────────────
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    // ── Smem copy handles（QKV retile）──────────────────────────────────────
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // ── Smem -> Reg copy handle（Mask）───────────────────────────────────────
    // 使用 make_tiled_copy_C（与 acc_s 的 C 分块对齐），参照 hstu_mask.h
    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_mask   = smem_tiled_copy_mask.get_thread_slice(tidx);
    // 同样对 stage=0 的 2D slice 做 partition_S，与 tQsMask 对应
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask_s0);  // (COPY_V, COPY_M, COPY_N)

    // ── Predicate tensors（QKV 边界检查）──────────────────────────────────────
    Tensor cQ   = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV  = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ   = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ   = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ);   ++k) { tQpQ(k)   = get<1>(tQcQ(0, 0, k))   < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

        // ── copy_g2s_mask：将一个 mask tile 从 gmem 搬到 smem（行/列联合谓词）──────
    // 行方向（!kMaskQFull）：越界行的向量被跳过（smem 已预清 0）。
    // 列方向（!Is_even_MN）：仅全局边界 tile（n_block_id == n_block_max-1）且
    // Sk%kBlockN!=0 时存在越界列——联合谓词跳过（smem 已预清 -inf）；其余 tile
    // 整块在界内，走无谓词/行谓词原路径（Is_even_MN 时编译期全部消除列谓词）。
auto copy_g2s_mask = [&](int n_block_id) {
    // tQsMask 是 3D (COPY_V, COPY_M, COPY_N)；tQgMask(_, _, _, n_block_id) rank 匹配
    const bool cols_full = Is_even_MN || (n_block_id < n_block_max - 1) ||
                           (actual_seqlen_k % kBlockN == 0);
    if (cols_full) {
        if constexpr (Kernel_traits::kMaskQFull) {
            cute::copy(gmem_tiled_copy_Mask, tQgMask(_, _, _, n_block_id), tQsMask);
        } else {
            cute::copy_if(gmem_tiled_copy_Mask, tMpMask,
                          tQgMask(_, _, _, n_block_id), tQsMask);
        }
    } else {
        // 行+列联合谓词（分支 CTA-uniform，无 warp divergence；谓词向量的 smem
        // 位置已预清 -inf/0，跳过即保持语义）
        const int cols_left = actual_seqlen_k - n_block_id * kBlockN;  // 8 对齐
        Tensor predK = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
        #pragma unroll
        for (int m = 0; m < size<0>(predK); ++m) {
            #pragma unroll
            for (int n = 0; n < size<1>(predK); ++n) {
                predK(m, n) = (get<1>(tMcMask(_0{}, m, n)) < cols_left)
                           && (Kernel_traits::kMaskQFull ||
                               get<0>(tMcMask(_0{}, m, n)) < mask_rows_left);
            }
        }
        cute::copy_if(gmem_tiled_copy_Mask, predK,
                      tQgMask(_, _, _, n_block_id), tQsMask);
    }
};

    // ── apply_mask_from_smem：smem → register，然后对 acc_s 做加法 ───────────
    //
    // mask 语义（加法 mask，与 SDPA 对齐，即 softmax(S·scale + mask)）：
    //   mask=0    → 可见，score 不变（acc_s += 0）
    //   mask=-inf → 屏蔽，score → -inf（acc_s += -inf）
    //
    // 由于越界列已由 prologue 预清 -inf（列谓词跳过拷贝）、越界行清 0，这里只需纯加法，
    // 无任何分支，SIMD 效率最优，零 warp divergence。
    // 注意：acc_s 此时尚未乘 softmax_scale，mask 需乘 1/scale 预先还原
    // （0/-inf mask 行为不变；有限值 mask 此前被错误地乘了 scale）。
    auto apply_mask_from_smem = [&](auto &acc_s) {
        const float mask_inv_scale = 1.f / params.scale_softmax;
        Tensor rMask = make_tensor<Element>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
        // tSsMask 是 partition_S(sMask) 的结果，rank 与 partition_D 返回的不同（无 stage 维度）
        cute::copy(smem_tiled_copy_mask, tSsMask, tSrMask_view);
        // 纯加法，无分支：可见位加 0，屏蔽位加 -inf，均匀指令流，无 warp divergence
        #pragma unroll
        for (int i = 0; i < size(acc_s); ++i) {
            acc_s(i) += static_cast<float>(rMask(i)) * mask_inv_scale;
        }
    };

    // ── Prologue：异步加载 Q + 第一个 K + 第一个 Mask ────────────────────────
    // Clear_OOB_MN=true: 当 Is_even_MN=false 时，将 OOB 行（>= actual_seqlen_q）在 smem 中清零
    // 防止 smem 中的垃圾 Q 值 × 大 K 值 → +inf，再 + mask(-inf) → NaN
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
        actual_seqlen_q - m_block * kBlockM
    );
    cute::cp_async_fence();

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

    // ── 主循环：从最后一个 n_block 倒序迭代（非 causal，无下三角约束）────────
    int n_block = n_block_max - 1;

    // 加载第一个 K tile（边界 block）
    // Clear_OOB_MN=true: 清零 K 的 OOB 行，防止垃圾 K 值 × Q → +inf + mask(-inf) → NaN
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
        actual_seqlen_k - n_block * kBlockN
    );
    // 异步加载第一个 Mask tile 到 sMask（无条件搬运，越界列 gmem 已填 -inf）
    copy_g2s_mask(n_block);
    cute::cp_async_fence();

    // 等待 Q / K / Mask 加载完成
    FLASH_NAMESPACE::cp_async_wait<0>();
    __syncthreads();

    // ── 处理最后一个（边界）block ──────────────────────────────────────────────
    {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        // 异步加载当前 V tile
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
            actual_seqlen_k - n_block * kBlockN
        );
        cute::cp_async_fence();

        // QK^T gemm（此时 sQ 和 sK 已就绪）
        FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
            smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // ── smem mask → reg，纯加法（越界列已是 -inf）────────────────────────
        apply_mask_from_smem(acc_s);

        // ── Softmax（第一个 block，Is_first=true，Check_inf=true）────────────
        softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
            acc_s, acc_o, params.scale_softmax_log2
        );

        // 等待 V 加载完成
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // 预加载下一个 K + Mask（若存在）
        if (n_block > 0) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV
            );
            copy_g2s_mask(n_block - 1);
            cute::cp_async_fence();
        }

        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
            FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // ── 非边界 block 循环（Is_even_MN=true 路径）────────────────────────────
    for (--n_block; n_block >= 0; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        // 等待上一轮预加载的 K + Mask 完成
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // 预加载下一个 V tile
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV
        );
        cute::cp_async_fence();

        // QK^T（K 已在上方 syncthreads 后就绪，Mask 也在 smem 里）
        FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
            smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // ── smem mask → reg，纯加法 ──────────────────────────────────────────
        apply_mask_from_smem(acc_s);

        // 等待 V 加载完成
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // 预加载下一个 K + Mask（若有下一轮）
        if (n_block > 0) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV
            );
            copy_g2s_mask(n_block - 1);
            cute::cp_async_fence();
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
            acc_s, acc_o, params.scale_softmax_log2
        );

        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
            FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // ── Epilogue：归一化 + 写 O ──────────────────────────────────────────────
    softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
        acc_o, params.scale_softmax, /*rp_dropout=*/1.0f
    );

    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + bidb * params.o_batch_stride),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{})
    );
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor cO   = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        actual_seqlen_q - m_block * kBlockM
    );
}


// ── Split-KV 计算函数（自 sm120 splitkv 移植的 cp.async 版）─────────────────
// 每个 CTA 处理 K 维连续子区间 [n_begin, n_end)，epilogue 不写最终 O，
// 而是写部分结果（落盘布局约定与 sm120 相同，但 combine kernel 为 sm89
// 自有独立副本，见文件末尾，两者可各自演化）：
//   O_partial   : (num_splits, B, H, SqR, d) bf16，已按本地 l 归一化
//   LSE_partial : (num_splits, B, H, SqR) fp32，lse = m*scale + log(l)；
//                 空 split / 全屏蔽行 = -inf（combine 中 scale=0 跳过）
// 主循环结构与 compute_attn_1rowblock_mask 相同（单缓冲 K/V/Mask + 软件流水：
// V 搬运重叠 QK gemm，下一块 K/Mask 搬运重叠 softmax+PV gemm），差异：
//   1. 块范围限定在 [n_begin, n_end)（全局边界块 n_block_max-1 仅落在包含它的 split）
//   2. 首个处理块是 n_end-1（不一定是全局边界块），边界谓词搬运按块运行时判定
//      （条件 CTA 内一致，无 warp divergence；Is_even_MN 时编译期消除）
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mask(
    const FA_mask_params &params,
    const int bidb,     // batch index
    const int bidh,     // head index
    const int m_block,  // Q tile index（行方向）
    const int split_idx // K 维切分序号
) {
    using Element      = typename Kernel_traits::Element;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;

    const int actual_seqlen_q = params.seqlen_q;
    const int actual_seqlen_k = params.seqlen_k;

    if (m_block * kBlockM >= actual_seqlen_q) return;

    const int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);

    // 本 split 的 K 范围（连续区间，L2 友好）
    const int n_per_split = cute::ceil_div(n_block_max, params.num_splits);
    const int n_begin = split_idx * n_per_split;
    const int n_end   = (n_begin + n_per_split < n_block_max) ? (n_begin + n_per_split) : n_block_max;
    const int n_tiles = n_end - n_begin;   // 可能 <= 0（尾部分裂为空）

    // ── Global memory 指针：Q / K / V（与 base kernel 一致）──────────────────
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + bidb * params.q_batch_stride),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{})
    );
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + bidb * params.k_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{})
    );
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + bidb * params.v_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{})
    );
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    // ── Shared memory 排布：[sQ] [sK] [sV] [sMask]（与 base kernel 一致）───────
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sMask = make_tensor(sV.data() + size(sV),
                               typename Kernel_traits::SmemLayoutMask{});

    // ── Gmem -> Smem copy handles（QKV / Mask，与 base kernel 一致）────────────
    typename Kernel_traits::GmemTiledCopyQKV  gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask = gmem_tiled_copy_Mask.get_thread_slice(tidx);

    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.mask_ptr)
                      + bidb * params.mask_batch_stride),
        make_shape(params.mask_seqlen_q, params.mask_seqlen_k),
        make_stride(params.mask_row_stride, _1{})
    );
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, _));

    Tensor tQgMask = gmem_thr_copy_Mask.partition_S(gMask);
    auto sMask_s0 = sMask(_, _, _0{});
    Tensor tQsMask = gmem_thr_copy_Mask.partition_D(sMask_s0);

    // ── mask 行谓词（q 维不 pad，与 base kernel 一致）────────────────────────
    Tensor cMask   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tMcMask = gmem_thr_copy_Mask.partition_S(cMask);
    Tensor tMpMask = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
    const int mask_rows_left = params.mask_seqlen_q - m_block * kBlockM;
    if constexpr (!Kernel_traits::kMaskQFull) {
        #pragma unroll
        for (int m = 0; m < size<0>(tMpMask); ++m) {
            #pragma unroll
            for (int n = 0; n < size<1>(tMpMask); ++n) {
                tMpMask(m, n) = get<0>(tMcMask(_0{}, m, n)) < mask_rows_left;
            }
        }
        // 边界 CTA：谓词跳过的 smem 行一次性清 0（防止未初始化 smem 的 NaN 垃圾模式）
        if (mask_rows_left < kBlockM) {
            #pragma unroll
            for (int m = 0; m < size<1>(tQsMask); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tQsMask); ++n) {
                    if (!tMpMask(m, n)) { clear(tQsMask(_, m, n)); }
                }
            }
        }
    }

    // ── mask 列谓词（k 维不 pad，与 base kernel 同策略；仅含全局边界 tile 的 split）──
    // 守卫 n_end == n_block_max：边界 tile 只属于最后一个 split。其余 split 的所有
    // tile 均整块在界内，首轮即整块拷贝——若也预清 -inf 会与跨线程的清零构成
    // write-after-write race（拷贝在首个 __syncthreads 之前发射），必须跳过。
    // 本 split 首个处理块恰为边界 tile（倒序遍历），列谓词拷贝与预清不重叠，无 race。
    if constexpr (!Is_even_MN) {
        const int mask_k_tail = actual_seqlen_k % kBlockN;
        if (mask_k_tail != 0 && n_end == n_block_max) {
            const Element mask_neg_inf(static_cast<float>(-INFINITY));
            #pragma unroll
            for (int m = 0; m < size<1>(tQsMask); ++m) {
                #pragma unroll
                for (int n = 0; n < size<2>(tQsMask); ++n) {
                    if (get<1>(tMcMask(_0{}, m, n)) >= mask_k_tail) {
                        #pragma unroll
                        for (int v = 0; v < size<0>(tQsMask); ++v) {
                            tQsMask(v, m, n) = mask_neg_inf;
                        }
                    }
                }
            }
        }
    }

    // ── MMA handles（与 base kernel 一致）────────────────────────────────────
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_mask   = smem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask_s0);

    // ── Predicate tensors（QKV 边界检查，与 base kernel 一致）────────────────────
    Tensor cQ   = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV  = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ   = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ   = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ);   ++k) { tQpQ(k)   = get<1>(tQcQ(0, 0, k))   < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // ── copy_g2s_mask（行/列联合谓词，与 base kernel 同策略）──────────────────
    // 仅全局边界 tile（落在包含它的那个 split）且 Sk%kBlockN!=0 时存在列越界
    auto copy_g2s_mask = [&](int n_block_id) {
        const bool cols_full = Is_even_MN || (n_block_id < n_block_max - 1) ||
                               (actual_seqlen_k % kBlockN == 0);
        if (cols_full) {
            if constexpr (Kernel_traits::kMaskQFull) {
                cute::copy(gmem_tiled_copy_Mask, tQgMask(_, _, _, n_block_id), tQsMask);
            } else {
                cute::copy_if(gmem_tiled_copy_Mask, tMpMask,
                              tQgMask(_, _, _, n_block_id), tQsMask);
            }
        } else {
            const int cols_left = actual_seqlen_k - n_block_id * kBlockN;  // 8 对齐
            Tensor predK = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
            #pragma unroll
            for (int m = 0; m < size<0>(predK); ++m) {
                #pragma unroll
                for (int n = 0; n < size<1>(predK); ++n) {
                    predK(m, n) = (get<1>(tMcMask(_0{}, m, n)) < cols_left)
                               && (Kernel_traits::kMaskQFull ||
                                   get<0>(tMcMask(_0{}, m, n)) < mask_rows_left);
                }
            }
            cute::copy_if(gmem_tiled_copy_Mask, predK,
                          tQgMask(_, _, _, n_block_id), tQsMask);
        }
    };

    // ── apply_mask_from_smem（与 base kernel 一致：纯加法，零分支）────────────
    auto apply_mask_from_smem = [&](auto &acc_s) {
        const float mask_inv_scale = 1.f / params.scale_softmax;
        Tensor rMask = make_tensor<Element>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
        cute::copy(smem_tiled_copy_mask, tSsMask, tSrMask_view);
        #pragma unroll
        for (int i = 0; i < size(acc_s); ++i) {
            acc_s(i) += static_cast<float>(rMask(i)) * mask_inv_scale;
        }
    };

    // ── K/V 装载：全局边界块（n_block_max-1）需行谓词搬运，其余块整块在界内 ────
    // 边界块仅落在包含它的那个 split，其它 split 的所有块都走无谓词路径
    auto load_K = [&](int nb) {
        if (Is_even_MN || nb < n_block_max - 1) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, nb), tKsK, tKVcKV, tKVpKV);
        } else {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, nb), tKsK, tKVcKV, tKVpKV,
                actual_seqlen_k - nb * kBlockN);
        }
    };
    auto load_V = [&](int nb) {
        if (Is_even_MN || nb < n_block_max - 1) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, nb), tVsV, tKVcKV, tKVpKV);
        } else {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, nb), tVsV, tKVcKV, tKVpKV,
                actual_seqlen_k - nb * kBlockN);
        }
    };

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    auto lse = make_fragment_like(softmax.row_sum);

    if (n_tiles > 0) {
        const int n_first = n_end - 1;   // 本 split 首个处理块（倒序遍历）

        // ── Prologue：异步加载 Q + 首个 K + 首个 Mask ────────────────────────
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
            actual_seqlen_q - m_block * kBlockM
        );
        cute::cp_async_fence();

        load_K(n_first);
        copy_g2s_mask(n_first);
        cute::cp_async_fence();

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // ── 首个块（与 base kernel 的边界块处理同构：softmax 与 V 搬运重叠）───
        {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            load_V(n_first);
            cute::cp_async_fence();

            FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            apply_mask_from_smem(acc_s);

            softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                acc_s, acc_o, params.scale_softmax_log2
            );

            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            if (n_first > n_begin) {
                load_K(n_first - 1);
                copy_g2s_mask(n_first - 1);
                cute::cp_async_fence();
            }

            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        }

        // ── 其余块（倒序遍历，与 base kernel 主循环同构）─────────────────────
        for (int n_block = n_first - 1; n_block >= n_begin; --n_block) {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            // 等待上一轮预载的 K + Mask
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            load_V(n_block);
            cute::cp_async_fence();

            FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            apply_mask_from_smem(acc_s);

            // 等待 V 就绪
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();

            // 预载下一轮 K + Mask（若有）
            if (n_block > n_begin) {
                load_K(n_block - 1);
                copy_g2s_mask(n_block - 1);
                cute::cp_async_fence();
            }

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                acc_s, acc_o, params.scale_softmax_log2
            );

            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        }

        // Split 约定（与 sm120/FA3 combine 配套）：O_s 按本地 l 归一化；
        // lse_s = m*scale + log(l)；全屏蔽行 sum=0 → lse=-inf
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Split=*/true>(
            acc_o, params.scale_softmax, /*rp_dropout=*/1.0f
        );
    } else {
        // 空 split（尾部分裂）：O_partial 写 0（acc_o 已 clear），LSE 写 -inf
        cute::fill(lse, -INFINITY);
    }

    // ── Epilogue（splitkv）：部分结果落盘 ─────────────────────────────────────
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    // O_partial: (num_splits, B, H, SqR, d) bf16，视作 (SqR, H, B*num_splits)，
    // batch 维索引 = split_idx * B + bidb（与 sm120 splitkv 落盘布局一致）。
    // SqR 已按 kBlockM 对齐 → tile 内所有行都在 padded 缓冲内，行方向无需谓词。
    const int64_t op_slice_elems =
        (int64_t)params.h * params.seqlen_q_rounded * params.d;
    Tensor mOp = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.oaccum_ptr)
                      + ((int64_t)split_idx * params.b + bidb) * op_slice_elems),
        make_shape(params.seqlen_q_rounded, params.h, params.d),
        make_stride((int64_t)params.d, op_slice_elems / params.h, _1{})
    );
    Tensor gOp = local_tile(mOp(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO  = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgOp = gmem_thr_copy_O.partition_D(gOp);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgOp));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor cO   = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOp)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgOp, tOcO, tOpO,
        params.seqlen_q_rounded - m_block * kBlockM
    );

    // ── LSE 落盘：row_sum 已 quad_allreduce，quad 内 4 线程值相同，
    // 仅 col==0 的线程写（FA2/sm120 splitkv 同模式）
    const int64_t tile_lse_offset =
        (((int64_t)split_idx * params.b + bidb) * params.h + bidh) * (int64_t)params.seqlen_q_rounded
        + (int64_t)m_block * kBlockM;
    float* gLSE = reinterpret_cast<float*>(params.lseaccum_ptr) + tile_lse_offset;
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { gLSE[get<0>(taccOcO_row(mi))] = lse(mi); }
    }
}


// ── Split-KV 计算函数（双缓冲版，kStages=2）──────────────────────────────────
// sm120 多级流水（kStages 级 K/V/Mask smem + mbarrier 生产者/消费者）的
// cp.async 对应物：主循环每迭代先把下一 tile 的 K+V+Mask 预发射到另一级
// stage，再计算当前 tile，装载与计算完全重叠。对比单缓冲版的软件流水
// （V 搬运重叠 QK、下一块 K/Mask 搬运重叠 softmax+PV，但每 tile 仍有两次
// 串行的往返等待：等 K → 发 V → 等 V），双缓冲每迭代仅一次 wait+sync。
// 适用配置（smem ≤ sm89 每 CTA 上限 99KB）：
//   d128 (64,64) = 96KB ✓    d64 M64 (64,64) = 56KB ✓
//   d64 M128 (128,128) 需 160KB ✗ → 该配置仍走单缓冲路径。
// 小 grid + 长 Sk 场景每 CTA 仅数个 tile 且 K/V 首次从 DRAM 读入，
// 串行往返延迟占比高，双缓冲收益最大。
// 部分结果的落盘约定与单缓冲版完全一致（combine 为 sm89 自有 kernel，见文件末尾）。
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mask_db(
    const FA_mask_params &params,
    const int bidb,     // batch index
    const int bidh,     // head index
    const int m_block,  // Q tile index（行方向）
    const int split_idx // K 维切分序号
) {
    using Element      = typename Kernel_traits::Element;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kStages   = Kernel_traits::kStages;
    static_assert(kStages == 2, "双缓冲版当前仅实现 kStages=2");

    const int actual_seqlen_q = params.seqlen_q;
    const int actual_seqlen_k = params.seqlen_k;

    if (m_block * kBlockM >= actual_seqlen_q) return;

    const int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);

    // 本 split 的 K 范围（连续区间，L2 友好；与单缓冲版一致）
    const int n_per_split = cute::ceil_div(n_block_max, params.num_splits);
    const int n_begin = split_idx * n_per_split;
    const int n_end   = (n_begin + n_per_split < n_block_max) ? (n_begin + n_per_split) : n_block_max;
    const int n_tiles = n_end - n_begin;   // 可能 <= 0（尾部分裂为空）

    // ── Global memory 指针：Q / K / V（与单缓冲版一致）──────────────────
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + bidb * params.q_batch_stride),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{})
    );
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + bidb * params.k_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{})
    );
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + bidb * params.v_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{})
    );
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    // ── Shared memory 排布（多级缓冲）：[sQ] [sK×S] [sV×S] [sMask×S] ────────
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Kernel_traits::SmemLayoutKVStaged{});       // (N, d, S)
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKVStaged{});       // (N, d, S)
    Tensor sMask = make_tensor(sV.data() + size(sV),
                               typename Kernel_traits::SmemLayoutMaskStaged{});  // (M, N, S)
    Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedStaged{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(),
                                      typename Kernel_traits::SmemLayoutVtransposedStagedNoSwizzle{});

    // ── Gmem -> Smem copy handles（QKV / Mask）──────────────────────────────
    // K/V/Mask 的 destination 直接 partition 3D staged 张量 → 第 4 维为 stage，
    // 装载/消费时按运行期 stage 切片（与 sm120 splitkv 的 tSsK(_, _, _, stage) 同模式；
    // stage 间的 smem 偏移是 swizzle 周期（512 elem）的整数倍，切片后 swizzle 组合不变）
    typename Kernel_traits::GmemTiledCopyQKV  gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);   // (CPY, M, K, stage)
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);   // (CPY, M, K, stage)

    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask = gmem_tiled_copy_Mask.get_thread_slice(tidx);

    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.mask_ptr)
                      + bidb * params.mask_batch_stride),
        make_shape(params.mask_seqlen_q, params.mask_seqlen_k),
        make_stride(params.mask_row_stride, _1{})
    );
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, _));

    Tensor tQgMask = gmem_thr_copy_Mask.partition_S(gMask);
    Tensor tQsMask = gmem_thr_copy_Mask.partition_D(sMask);   // (CPY_V, CPY_M, CPY_N, stage)

    // ── mask 行谓词（q 维不 pad）；边界 CTA 需把所有 stage 的越界行一次性清 0 ──
    Tensor cMask   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tMcMask = gmem_thr_copy_Mask.partition_S(cMask);
    Tensor tMpMask = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
    const int mask_rows_left = params.mask_seqlen_q - m_block * kBlockM;
    if constexpr (!Kernel_traits::kMaskQFull) {
        #pragma unroll
        for (int m = 0; m < size<0>(tMpMask); ++m) {
            #pragma unroll
            for (int n = 0; n < size<1>(tMpMask); ++n) {
                tMpMask(m, n) = get<0>(tMcMask(_0{}, m, n)) < mask_rows_left;
            }
        }
        // 边界 CTA：copy_if 谓词跳过的 smem 行在任何 stage 中都不会被写入
        // （行谓词与 n_block 无关），一次性清 0 防止未初始化 smem 的 NaN 垃圾模式
        if (mask_rows_left < kBlockM) {
            #pragma unroll
            for (int st = 0; st < kStages; ++st) {
                #pragma unroll
                for (int m = 0; m < size<1>(tQsMask); ++m) {
                    #pragma unroll
                    for (int n = 0; n < size<2>(tQsMask); ++n) {
                        if (!tMpMask(m, n)) { clear(tQsMask(_, m, n, st)); }
                    }
                }
            }
        }
    }

    // ── mask 列谓词（k 维不 pad，与单缓冲版同策略；仅含全局边界 tile 的 split）──
    // 边界 tile 是本 split 首个处理块（j=0 → stage 0），所有 stage 一并预清
    //（stage 1 的预清随后即被首个整块拷贝覆写，无害）；stage 0 的 -inf 列在
    // 迭代 j=0 消费完毕（尾部 __syncthreads）之后才会被流水回绕的整块拷贝覆写，
    // 无 race。其余 split 必须跳过（其 tile 全部整块在界内，预清会与首轮整块
    // 拷贝构成跨线程 write-after-write race，见单缓冲版注释）。
    if constexpr (!Is_even_MN) {
        const int mask_k_tail = actual_seqlen_k % kBlockN;
        if (mask_k_tail != 0 && n_end == n_block_max) {
            const Element mask_neg_inf(static_cast<float>(-INFINITY));
            #pragma unroll
            for (int st = 0; st < kStages; ++st) {
                #pragma unroll
                for (int m = 0; m < size<1>(tQsMask); ++m) {
                    #pragma unroll
                    for (int n = 0; n < size<2>(tQsMask); ++n) {
                        if (get<1>(tMcMask(_0{}, m, n)) >= mask_k_tail) {
                            #pragma unroll
                            for (int v = 0; v < size<0>(tQsMask); ++v) {
                                tQsMask(v, m, n, st) = mask_neg_inf;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── MMA handles（fragment 形状与 stage 无关，取 stage=0 的 2D 视图）──────
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK(_, _, _0{}));
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);      // (CPY, M, K, stage)

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);    // (CPY, N, K, stage)

    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_mask   = smem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask);   // (CPY, M, N, stage)

    // ── Predicate tensors（QKV 边界检查，与单缓冲版一致）────────────────────
    Tensor cQ   = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV  = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ   = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ   = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ);   ++k) { tQpQ(k)   = get<1>(tQcQ(0, 0, k))   < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // ── 装载 lambda（带 stage 维；全局边界块 n_block_max-1 需行谓词搬运，
    //    其余块整块在界内——与单缓冲版一致）─────────────────────────────────
    auto load_K = [&](int nb, int stage) {
        // stage 切片先绑定到具名局部变量：FLASH copy 的 dst 形参是非 const
        // 左值引用，不能绑定临时右值
        Tensor tKsK_st = tKsK(_, _, _, stage);
        if (Is_even_MN || nb < n_block_max - 1) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, nb), tKsK_st, tKVcKV, tKVpKV);
        } else {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, nb), tKsK_st, tKVcKV, tKVpKV,
                actual_seqlen_k - nb * kBlockN);
        }
    };
    auto load_V = [&](int nb, int stage) {
        Tensor tVsV_st = tVsV(_, _, _, stage);
        if (Is_even_MN || nb < n_block_max - 1) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, nb), tVsV_st, tKVcKV, tKVpKV);
        } else {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, nb), tVsV_st, tKVcKV, tKVpKV,
                actual_seqlen_k - nb * kBlockN);
        }
    };
    auto copy_g2s_mask = [&](int nb, int stage) {
        Tensor tQsMask_st = tQsMask(_, _, _, stage);
        // 列谓词（k 维不 pad）：仅全局边界 tile 且 Sk%kBlockN!=0 时非平凡
        const bool cols_full = Is_even_MN || (nb < n_block_max - 1) ||
                               (actual_seqlen_k % kBlockN == 0);
        if (cols_full) {
            if constexpr (Kernel_traits::kMaskQFull) {
                cute::copy(gmem_tiled_copy_Mask, tQgMask(_, _, _, nb), tQsMask_st);
            } else {
                cute::copy_if(gmem_tiled_copy_Mask, tMpMask,
                              tQgMask(_, _, _, nb), tQsMask_st);
            }
        } else {
            const int cols_left = actual_seqlen_k - nb * kBlockN;  // 8 对齐
            Tensor predK = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
            #pragma unroll
            for (int m = 0; m < size<0>(predK); ++m) {
                #pragma unroll
                for (int n = 0; n < size<1>(predK); ++n) {
                    predK(m, n) = (get<1>(tMcMask(_0{}, m, n)) < cols_left)
                               && (Kernel_traits::kMaskQFull ||
                                   get<0>(tMcMask(_0{}, m, n)) < mask_rows_left);
                }
            }
            cute::copy_if(gmem_tiled_copy_Mask, predK,
                          tQgMask(_, _, _, nb), tQsMask_st);
        }
    };
    auto apply_mask_from_smem = [&](auto &acc_s, int stage) {
        const float mask_inv_scale = 1.f / params.scale_softmax;
        Tensor rMask = make_tensor<Element>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
        cute::copy(smem_tiled_copy_mask, tSsMask(_, _, _, stage), tSrMask_view);
        #pragma unroll
        for (int i = 0; i < size(acc_s); ++i) {
            acc_s(i) += static_cast<float>(rMask(i)) * mask_inv_scale;
        }
    };

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    auto lse = make_fragment_like(softmax.row_sum);

    if (n_tiles > 0) {
        const int n_first = n_end - 1;   // 本 split 首个处理块（倒序遍历）

        // ── Prologue：Q + 首个 tile 的 K+V+Mask → stage 0 ────────────────
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
            actual_seqlen_q - m_block * kBlockM
        );
        cute::cp_async_fence();

        load_K(n_first, 0);
        load_V(n_first, 0);
        copy_g2s_mask(n_first, 0);
        cute::cp_async_fence();

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // ── 主循环：预发射下一 tile → 计算当前 tile → 等待+同步 ──────────
        // stage 交替：迭代 k（k = n_first - n_block）用 stage k&1；
        // 预发射写入 stage^1（上一轮已 sync 保证其读取全部完成）。
        for (int n_block = n_first; n_block >= n_begin; --n_block) {
            const int stage = (n_first - n_block) & 1;
            const bool has_next = (n_block > n_begin);

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            // 预发射下一 tile 的 K+V+Mask → 另一级 stage（与本次计算完全重叠）
            if (has_next) {
                load_K(n_block - 1, stage ^ 1);
                load_V(n_block - 1, stage ^ 1);
                copy_g2s_mask(n_block - 1, stage ^ 1);
                cute::cp_async_fence();
            }

            // QK^T（K 来自当前 stage）
            FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, stage), tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            apply_mask_from_smem(acc_s, stage);

            if (n_block == n_first) {
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
            } else {
                softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
            }

            // P·V（V 来自当前 stage）
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, stage), tiled_mma,
                                     smem_tiled_copy_V, smem_thr_copy_V);

            // 等待预发射的下一 tile 就绪；sync 同时保证所有线程已读完本 stage
            // 的 K/V/Mask（下一轮将向本 stage 写入）。末轮无条件 sync：保证最后
            // 的 QK gemm 对 sQ 的读取完成后再进入 epilogue 写 sO（sO 复用 sQ 区域）
            if (has_next) {
                FLASH_NAMESPACE::cp_async_wait<0>();
            }
            __syncthreads();
        }

        // Split 约定（与单缓冲版/sm120 combine 配套）：O_s 按本地 l 归一化；
        // lse_s = m*scale + log(l)；全屏蔽行 sum=0 → lse=-inf
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Split=*/true>(
            acc_o, params.scale_softmax, /*rp_dropout=*/1.0f
        );
    } else {
        // 空 split（尾部分裂）：O_partial 写 0（acc_o 已 clear），LSE 写 -inf
        cute::fill(lse, -INFINITY);
    }

    // ── Epilogue（splitkv）：部分结果落盘（与单缓冲版一致）────────────────
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    // O_partial: (num_splits, B, H, SqR, d) bf16，视作 (SqR, H, B*num_splits)，
    // batch 维索引 = split_idx * B + bidb（与 sm120 splitkv 落盘布局一致）。
    const int64_t op_slice_elems =
        (int64_t)params.h * params.seqlen_q_rounded * params.d;
    Tensor mOp = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.oaccum_ptr)
                      + ((int64_t)split_idx * params.b + bidb) * op_slice_elems),
        make_shape(params.seqlen_q_rounded, params.h, params.d),
        make_stride((int64_t)params.d, op_slice_elems / params.h, _1{})
    );
    Tensor gOp = local_tile(mOp(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO  = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgOp = gmem_thr_copy_O.partition_D(gOp);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgOp));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor cO   = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOp)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgOp, tOcO, tOpO,
        params.seqlen_q_rounded - m_block * kBlockM
    );

    // ── LSE 落盘（与单缓冲版一致）：row_sum 已 quad_allreduce，仅 col==0 写 ──
    const int64_t tile_lse_offset =
        (((int64_t)split_idx * params.b + bidb) * params.h + bidh) * (int64_t)params.seqlen_q_rounded
        + (int64_t)m_block * kBlockM;
    float* gLSE = reinterpret_cast<float*>(params.lseaccum_ptr) + tile_lse_offset;
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { gLSE[get<0>(taccOcO_row(mi))] = lse(mi); }
    }
}


template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_mask_kernel(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __grid_constant__
#endif
    const FA_mask_params params
) {
    const int m_block = blockIdx.x;
    const int bidb    = blockIdx.y;
    const int bidh    = blockIdx.z;
    compute_attn_1rowblock_mask<Kernel_traits, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block);
}


// ── split-KV kernel 入口 ───────────────────────────────────────────────────────
// grid.x = num_m_blocks * num_splits：低 m_block 位变化快（同 (b,h,split) 的 CTA
// 共享 K/V，L2 友好）；grid.y/z = B/H（与 sm120 splitkv 相同）
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_mask_kernel_splitkv(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __grid_constant__
#endif
    const FA_mask_params params
) {
    // 先取局部 constexpr 再使用：nvcc 下直接在 device 代码运行时表达式中
    // odr-use Kernel_traits::kBlockM 会报 "undefined in device code"
    constexpr int kBlockM_ = Kernel_traits::kBlockM;
    const int num_m_blocks = cute::ceil_div(params.seqlen_q, kBlockM_);
    const int m_block   = blockIdx.x % num_m_blocks;
    const int split_idx = blockIdx.x / num_m_blocks;
    // kStages=2（smem 预算允许时）：走双缓冲路径；否则单缓冲
    if constexpr (Kernel_traits::kStages == 2) {
        compute_attn_1rowblock_splitkv_mask_db<Kernel_traits, Is_even_MN, Is_even_K>(
            params, blockIdx.y, blockIdx.z, m_block, split_idx);
    } else {
        compute_attn_1rowblock_splitkv_mask<Kernel_traits, Is_even_MN, Is_even_K>(
            params, blockIdx.y, blockIdx.z, m_block, split_idx);
    }
}


template<typename Kernel_traits>
void run_flash_fwd_with_mask(const FA_mask_params &params, cudaStream_t stream) {
    // kSmemSize 已在 FA_mask_kernel_traits 中覆盖基类，包含 mask tile 的额外 smem
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    dim3 grid(num_m_block, params.b, params.h);

    const bool is_even_MN = (params.seqlen_k % Kernel_traits::kBlockN == 0) &&
                             (params.seqlen_q % Kernel_traits::kBlockM == 0);
    const bool is_even_K  = (params.d == Kernel_traits::kHeadDim);

    auto do_launch = [&](bool even_mn, bool even_k) {
        auto launch_kernel = [&](auto IsEvenMN, auto IsEvenK) {
            auto kernel = &flash_fwd_mask_kernel<Kernel_traits, IsEvenMN.value, IsEvenK.value>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
                ));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        };
        if (even_mn && even_k)       { launch_kernel(cute::Int<1>{}, cute::Int<1>{}); }
        else if (even_mn && !even_k) { launch_kernel(cute::Int<1>{}, cute::Int<0>{}); }
        else if (!even_mn && even_k) { launch_kernel(cute::Int<0>{}, cute::Int<1>{}); }
        else                         { launch_kernel(cute::Int<0>{}, cute::Int<0>{}); }
    };
    do_launch(is_even_MN, is_even_K);
}

// ── split-KV host-side dispatch（grid.x = num_m_blocks * num_splits）──────────
template<typename Kernel_traits>
void run_flash_fwd_with_mask_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    // kStages>1（双缓冲路径）：K/V/Mask 各 kStages 级，用 staged smem 尺寸
    constexpr size_t smem_size = (Kernel_traits::kStages > 1)
        ? static_cast<size_t>(Kernel_traits::kSmemSizeStaged)
        : static_cast<size_t>(Kernel_traits::kSmemSize);

    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    dim3 grid(num_m_block * params.num_splits, params.b, params.h);

    const bool is_even_MN = (params.seqlen_k % Kernel_traits::kBlockN == 0) &&
                             (params.seqlen_q % Kernel_traits::kBlockM == 0);
    const bool is_even_K  = (params.d == Kernel_traits::kHeadDim);

    auto do_launch = [&](bool even_mn, bool even_k) {
        auto launch_kernel = [&](auto IsEvenMN, auto IsEvenK) {
            auto kernel = &flash_fwd_mask_kernel_splitkv<Kernel_traits, IsEvenMN.value, IsEvenK.value>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
                ));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        };
        if (even_mn && even_k)       { launch_kernel(cute::Int<1>{}, cute::Int<1>{}); }
        else if (even_mn && !even_k) { launch_kernel(cute::Int<1>{}, cute::Int<0>{}); }
        else if (!even_mn && even_k) { launch_kernel(cute::Int<0>{}, cute::Int<1>{}); }
        else                         { launch_kernel(cute::Int<0>{}, cute::Int<0>{}); }
    };
    do_launch(is_even_MN, is_even_K);
}

// ── Split-KV combine kernel（sm89 独立副本，与 sm120 完全解耦）────────────────
// 本体逻辑与 sm120 的 combine kernel 相同（纯指针加权归约，不依赖 TMA），落盘
// 布局约定一致（见上方 splitkv 计算函数的注释），但代码各归各的：sm89 版无 PDL
// 分支（PDL 为 sm90+ 特性），两条路径可各自独立调优互不影响。
// grid = (SqR/kRows, B*H, kHeadDim/kCols)，block = 128 线程；
// 每个 CTA 处理 kRows 行 × kCols 列的输出子块。tile 尺寸由 host 端自适应选择：
// grid 足够大时用 32×32（摊薄 Phase1 开销）；小 grid 时缩到 32×16 / 16×16，
// 把归约并行度放大 2~4 倍吃满带宽（kCols=16 时每行恰好 1 个 32B sector）。
//   Phase1: 1 个 warp 计算本 CTA kRows 行的 scale_s = exp(lse_s - lse_max) / Σ_s
//   Phase2: 全 CTA uint2(4×bf16) 向量化加权归约，整组跳过非活跃 split
template<int kHeadDim, int kNThreads, int kRows, int kCols, typename Element>
__global__ void __launch_bounds__(kNThreads)
flash_fwd_mask_combine_kernel_sm89(
    const Element* __restrict__ o_partial,  // (S, B, H, SqR, D) bf16
    const float* __restrict__ lse_partial,  // (S, B, H, SqR) fp32
    Element*   __restrict__ o_ptr,          // (B, H, Sq, D)（带 strides）
    const int num_splits, const int seqlen_q, const int seqlen_q_rounded,
    const int num_heads, const int batch,
    const int64_t o_batch_stride, const int64_t o_head_stride, const int64_t o_row_stride) {

    constexpr int kMaxSplits = 64;    // host 侧已 clamp（8KB smem）
    static_assert(kRows <= 32 && kCols >= 16 && kCols % 4 == 0, "tile constraint");
    __shared__ float s_scale[kMaxSplits * kRows];
    __shared__ unsigned long long s_active;   // CTA-uniform：bit s = 该 split 有非零 scale

    const int row0    = blockIdx.x * kRows;             // tile 内全局行起点（含 m_block 偏移）
    const int bh      = blockIdx.y;
    const int bidb    = bh / num_heads;
    const int bidh    = bh % num_heads;
    const int col0    = blockIdx.z * kCols;
    const int tidx    = threadIdx.x;

    const int64_t sqR = seqlen_q_rounded;
    const int64_t split_stride_o   = (int64_t)batch * num_heads * sqR * kHeadDim;
    const int64_t split_stride_lse = (int64_t)batch * num_heads * sqR;
    const int64_t bh_off_o   = ((int64_t)bidb * num_heads + bidh) * sqR * kHeadDim;
    const int64_t bh_off_lse = ((int64_t)bidb * num_heads + bidh) * sqR;

    // Phase 1: per-row scales（warp0，row-per-lane；外层按 split 循环保证合并访存）
    // 同时归约出 CTA-uniform 的活跃 split 位掩码，供 Phase2 整组跳过被 mask 的 split
    if (tidx < 32) {
        unsigned long long bits = 0;
        if (tidx < kRows) {
            const int row = row0 + tidx;
            const float* lse_base = lse_partial + bh_off_lse + row;
            float lse_max = -INFINITY;
            for (int s = 0; s < num_splits; ++s) {
                lse_max = fmaxf(lse_max, lse_base[s * split_stride_lse]);
            }
            // 全 -inf 行（整行被 mask）：lse_max_c=0 → exp(-inf)=0 → scale 全 0 → O=0
            const float lse_max_c = (lse_max == -INFINITY) ? 0.f : lse_max;
            float sum = 0.f;
            for (int s = 0; s < num_splits; ++s) {
                sum += __expf(lse_base[s * split_stride_lse] - lse_max_c);
            }
            const float inv = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            for (int s = 0; s < num_splits; ++s) {
                const float sc = __expf(lse_base[s * split_stride_lse] - lse_max_c) * inv;
                s_scale[s * kRows + tidx] = sc;
                if (sc > 0.f) { bits |= (1ull << s); }
            }
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        // 注意：__reduce_or_sync 仅有 32-bit 重载（and/or/xor 无 64-bit 版），
        // 直接传 64-bit 会被隐式截断 → splits 32~63 活跃位丢失（曾导致结果错误）
        const unsigned bits_lo = __reduce_or_sync(0xffffffffu, static_cast<unsigned>(bits));
        const unsigned bits_hi = __reduce_or_sync(0xffffffffu, static_cast<unsigned>(bits >> 32));
        bits = bits_lo | (static_cast<unsigned long long>(bits_hi) << 32);
#else
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) { bits |= __shfl_down_sync(0xffffffffu, bits, off); }
#endif
        if (tidx == 0) { s_active = bits; }
    }
    __syncthreads();

    // Phase 2: uint2(4×bf16) 向量化加权归约（kCols/4 个 chunk/行，连续线程访问连续 chunk → 合并访存）
    constexpr int kChunksPerRow = kCols / 4;             // 32列→8, 16列→4
    constexpr int kTotalChunks  = kRows * kChunksPerRow;
    const Element* op_base = o_partial + bh_off_o + (int64_t)row0 * kHeadDim + col0;
    Element* o_base = o_ptr + bidb * o_batch_stride + bidh * o_head_stride
                      + (int64_t)row0 * o_row_stride + col0;

    #pragma unroll
    for (int c = tidx; c < kTotalChunks; c += kNThreads) {
        const int r  = c / kChunksPerRow;                // CTA 内行
        const int cc = (c - r * kChunksPerRow) * 4;      // CTA 内列
        const unsigned long long active = s_active;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        const Element* cp_base = op_base + r * kHeadDim + cc;
        // 8 路独立 load+FMA 链展开：暴露 MLP（在途字节 ×8，足以覆盖 DRAM 延迟），
        // 整组跳过非活跃 split
        int s = 0;
        for (; s + 8 <= num_splits; s += 8) {
            if (((active >> s) & 0xffull) == 0ull) { continue; }
            Element pv[8][4];
            float sc[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                *reinterpret_cast<uint2*>(pv[u]) =
                    *reinterpret_cast<const uint2*>(cp_base + (s + u) * split_stride_o);
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) { sc[u] = s_scale[(s + u) * kRows + r]; }
            // 注意：非活跃 split 的 sc=0，但 pv 可能是 masked 区的任意值；0*有限值=0 安全。
            // masked split 的 O_partial 由主 kernel 写成有限值（l=0 时全 0），无 inf/NaN。
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc.x += sc[u] * float(pv[u][0]);
                acc.y += sc[u] * float(pv[u][1]);
                acc.z += sc[u] * float(pv[u][2]);
                acc.w += sc[u] * float(pv[u][3]);
            }
        }
        for (; s + 4 <= num_splits; s += 4) {   // 4 路尾部
            if (((active >> s) & 0xfull) == 0ull) { continue; }
            Element pv[4][4];
            float sc[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                *reinterpret_cast<uint2*>(pv[u]) =
                    *reinterpret_cast<const uint2*>(cp_base + (s + u) * split_stride_o);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) { sc[u] = s_scale[(s + u) * kRows + r]; }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                acc.x += sc[u] * float(pv[u][0]);
                acc.y += sc[u] * float(pv[u][1]);
                acc.z += sc[u] * float(pv[u][2]);
                acc.w += sc[u] * float(pv[u][3]);
            }
        }
        for (; s < num_splits; ++s) {   // 尾部（<4 个）
            const float sc = s_scale[s * kRows + r];
            if (sc > 0.f) {
                Element pv[4];
                *reinterpret_cast<uint2*>(pv) = *reinterpret_cast<const uint2*>(cp_base + s * split_stride_o);
                acc.x += sc * float(pv[0]);  acc.y += sc * float(pv[1]);
                acc.z += sc * float(pv[2]);  acc.w += sc * float(pv[3]);
            }
        }
        if (row0 + r < seqlen_q) {
            Element tmp[4] = {static_cast<Element>(acc.x), static_cast<Element>(acc.y),
                              static_cast<Element>(acc.z), static_cast<Element>(acc.w)};
            *reinterpret_cast<uint2*>(o_base + (int64_t)r * o_row_stride + cc) =
                *reinterpret_cast<const uint2*>(tmp);
        }
    }
}

// combine launcher（sm89）：tile 自适应 + 普通 <<<>>> 启动（无 PDL）
template<int kHeadDim, typename Element>
void run_flash_fwd_mask_combine_sm89(const FA_mask_params &params, cudaStream_t stream) {
    constexpr int kNThreads = 128;
    // tile 自适应：小 grid 时先切列（grid z 翻倍）再切行（grid x 再翻倍），
    // 放大归约并行度（4090D 128 SM，目标 ≥ ~128 CTA）
    const int64_t ctas_32 = (int64_t)(params.seqlen_q_rounded / 32) * params.b * params.h * (kHeadDim / 32);
    int rows_per_cta = 32, cols_per_cta = 32;
    if (ctas_32 < 128) {
        cols_per_cta = 16;
        const int64_t ctas_32x16 = ctas_32 * 2;
        if (ctas_32x16 < 128 && params.seqlen_q_rounded % 16 == 0) {
            rows_per_cta = 16;
        }
    }
    dim3 grid(params.seqlen_q_rounded / rows_per_cta, params.b * params.h,
              kHeadDim / cols_per_cta);
    #define LAUNCH_COMBINE_SM89(R, C) \
        do { \
            auto kernel = &flash_fwd_mask_combine_kernel_sm89<kHeadDim, kNThreads, R, C, Element>; \
            kernel<<<grid, kNThreads, 0, stream>>>( \
                reinterpret_cast<const Element*>(params.oaccum_ptr), \
                reinterpret_cast<const float*>(params.lseaccum_ptr), \
                reinterpret_cast<Element*>(params.o_ptr), \
                params.num_splits, params.seqlen_q, params.seqlen_q_rounded, \
                params.h, params.b, \
                params.o_batch_stride, params.o_head_stride, params.o_row_stride); \
        } while (0)
    if (rows_per_cta == 32 && cols_per_cta == 32)      { LAUNCH_COMBINE_SM89(32, 32); }
    else if (rows_per_cta == 32 && cols_per_cta == 16) { LAUNCH_COMBINE_SM89(32, 16); }
    else                                               { LAUNCH_COMBINE_SM89(16, 16); }
    #undef LAUNCH_COMBINE_SM89
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FA_MASK_NAMESPACE
