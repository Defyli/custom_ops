/*
 * Flash Attention Forward with Additive Mask — Core Compute Kernel
 *
 * 从 FA2 的 compute_attn_1rowblock 移植，专为 prefill（非 causal）+ 外部 mask 场景精简：
 *   - 删除 dropout / rotary / KV-cache / alibi / local window / softcap 逻辑
 *   - 删除 split-KV 路径
 *   - 新增：通过 cute GmemTiledCopyMask + smem 缓冲读取 bf16 additive mask，
 *           在 gemm 后对 acc_s 做 smem→reg copy + 逐元素加法（零分支，无 warp divergence）
 *   - 保留 Is_even_MN / Is_even_K 分支以保证边界正确性
 *
 * Mask 语义（加法 mask，与 SDPA 对齐，即 softmax(S·scale + mask)）：
 *   mask=0    → 可见（score 不变）
 *   mask=-inf → 屏蔽（score → -inf，softmax 后 weight = 0）
 *   有限值    → 任意加法偏置（ALiBi 风格）；实现上 mask 先于 scale 加到 acc_s，
 *               应用点乘 1/scale_softmax 预还原（见 apply_mask_from_smem）
 *
 * Mask tensor 的 global mem 格式：(B, seqlen_q_rounded, seqlen_k_rounded)，row-major，bf16
 *   - seqlen_k_rounded = ceil(seqlen_k, kBlockN) * kBlockN（由调用方 pad，越界填 -inf）
 *   - kBlockN（64 或 128）本身是 8 的倍数，保证 cp.async 128-bit 无越界且整 tile 无越界
 *   - 通过 params.mask_ptr / mask_batch_stride / mask_row_stride 寻址
 *
 * copy_g2s_mask 无需 predicate：
 *   seqlen_k_rounded 是 8 的倍数，gmem 中越界列已填 -inf，
 *   cp.async 128-bit 直接搬整个 tile，越界列的 -inf 自然流入 smem。
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

#include "fa_fwd_mask.h"

// 复用 FA2 的工具函数（softmax、gemm、copy、convert_type 等）
#include "utils.h"
#include "softmax.h"

namespace FA_MASK_NAMESPACE {

using namespace cute;

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

    // Additive mask: (B, seqlen_q_rounded, seqlen_k_rounded), row-major, bf16
    // seqlen_q_rounded = ceil(seqlen_q, kBlockM) * kBlockM（由调用方保证，越界填 -inf）
    // seqlen_k_rounded = ceil(seqlen_k, kBlockN) * kBlockN（由调用方保证，越界填 -inf）
    // 双向对齐保证 copy_g2s_mask 整 tile 搬运无越界；
    // kBlockN 本身是 8 的倍数，满足 cp.async 128-bit 要求
    void *__restrict__ mask_ptr;
    int64_t mask_batch_stride;    // stride over batch dim
    int64_t mask_row_stride;      // stride over seqlen_q_rounded = seqlen_k_rounded

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

    // ── Split-KV（num_splits > 1 时有效）─────────────────────────────────────
    // O_partial: (num_splits, B, H, seqlen_q_rounded, d) bf16，已按各 split 本地 l 归一化
    // LSE_partial: (num_splits, B, H, seqlen_q_rounded) fp32，lse = m*scale + log(l)
    void *__restrict__ oaccum_ptr = nullptr;
    void *__restrict__ lseaccum_ptr = nullptr;
    int num_splits = 1;
};


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
    // mask 已由调用方 pad 到 (seqlen_q_rounded, seqlen_k_rounded)，双向均无越界问题
    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.mask_ptr)
                      + bidb * params.mask_batch_stride),
        make_shape(params.seqlen_q_rounded, params.seqlen_k_rounded),
        make_stride(params.mask_row_stride, _1{})
    );
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, _));

    Tensor tQgMask = gmem_thr_copy_Mask.partition_S(gMask);
    // sMask 是 3D (kBlockM, kBlockN, kStages=1)，取 stage=0 的 2D slice 再做 partition，
    // 避免 4D partition 时内部嵌套 rank 造成 CopyAtom rank-mismatch 编译错误
    auto sMask_s0 = sMask(_, _, _0{});  // 2D: (kBlockM, kBlockN)
    Tensor tQsMask = gmem_thr_copy_Mask.partition_D(sMask_s0);  // (COPY_V, COPY_M, COPY_N)

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

    // ── copy_g2s_mask：将一个 mask tile 从 gmem 无条件搬到 smem ──────────────
    // 关键：seqlen_k_rounded 已由调用方对齐到 8（128bit / sizeof(bf16) = 8），
    // gmem 中越界列已填 -inf，cp.async 128-bit 可以直接搬整个 tile，
    // 越界列的 -inf 自然流入 sMask，无需任何 predicate 判断。
    auto copy_g2s_mask = [&](int n_block_id) {
        // tQsMask 是 3D (COPY_V, COPY_M, COPY_N)，由 partition_D(sMask_s0) 生成
        // tQgMask(_, _, _, n_block_id) 也是 3D，rank 匹配
        cute::copy(gmem_tiled_copy_Mask,
                   tQgMask(_, _, _, n_block_id),
                   tQsMask);
    };

    // ── apply_mask_from_smem：smem → register，然后对 acc_s 做加法 ───────────
    //
    // mask 语义（加法 mask，与 SDPA 对齐，即 softmax(S·scale + mask)）：
    //   mask=0    → 可见，score 不变（acc_s += 0）
    //   mask=-inf → 屏蔽，score → -inf（acc_s += -inf）
    //
    // 由于 copy_g2s_mask 已将越界列的 -inf 搬入 sMask，这里只需纯加法，
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


// ── kernel 入口 ───────────────────────────────────────────────────────────────
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


// ── host-side dispatch ────────────────────────────────────────────────────────
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

} // namespace FA_MASK_NAMESPACE
