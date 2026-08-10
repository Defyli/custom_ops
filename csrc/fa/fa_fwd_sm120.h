/*
 * Flash Attention Forward with Additive Mask — SM120 (Blackwell, RTX 50) Kernel
 *
 * 设计：FA2 计算结构(mma.sync + online softmax) + FA3(Hopper) 式数据搬运。
 *
 * sm120 与 sm90 的关键差异（调研自 thirdparty cutlass/cute）：
 *   - 无 wgmma/GMMA（sm90a 专属），无 tcgen05/TMEM（sm100 专属）
 *     → MMA 只能用 SM80 风格 mma.sync atom（SM80_16x8x16_F32BF16BF16F32_TN）
 *   - TMA 可用，但目标是 shared::cta（不支持 cluster / TMA multicast，
 *     CUTLASS sm120 builder 同样断言 cluster size == 1）
 *   - stmatrix(STSM)/ldmatrix/mbarrier(complete_tx) 均可用
 *   - 每个 CTA 最大动态 smem 仅 101376B（与 Ada 同级，远小于 Hopper 227KB），
 *     因此 tile/stage 配置无法照搬 FA3
 *   - 需以 sm_120a 编译：纯 sm_120 时 cutlass 不定义 CUTE_ARCH_TMA_SM90_ENABLED
 *
 * 相对于 csrc/fa/fa_fwd_kernel.h（cp.async 版）的改动：
 *   1. Q/K/V/Mask 的 gmem→smem 全部由 TMA 完成（单线程发射，mbarrier 完成跟踪），
 *      K/V/Mask 走 kStages 级流水（full/empty mbarrier 对）
 *   2. 边界全部交给 TMA（OOB 读自动补 0；配合 mask 越界列的 -inf，语义与
 *      Clear_OOB_MN=true 的 cp.async 版完全一致），O 写用 TMA store 自动裁剪
 *      → 不再需要 Is_even_MN / Is_even_K 分支
 *   3. Epilogue：acc_o → sO（复用 sQ smem）→ TMA store 直接写 gmem
 *
 * Mask 语义（加法 mask，与 SDPA 对齐）：softmax(S·scale + mask)，
 *   0=可见，-inf=屏蔽，支持任意有限值偏置（ALiBi 风格）。
 *   实现上 mask 加到未缩放的 acc_s 再统一乘 scale，因此应用点需乘 1/scale 预还原
 *   （见各 kernel 内 mask_inv_scale；0/-inf mask 不受此影响）。
 * Mask gmem 布局：(B, seqlen_q_rounded, seqlen_k_rounded)，row-major，bf16，
 *   两个方向均已按 kBlockM/kBlockN pad（越界填 -inf，由 host 侧保证）
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_types.h"

#include "kernel_traits.h"    // Flash_kernel_traits（MMA atom、ldmatrix copy atom）
#include "fa_fwd_kernel.h"    // FA_mask_params
#include "utils.h"
#include "softmax.h"

namespace FA_MASK_NAMESPACE {

using namespace cute;

// ── SM120 kernel traits ──────────────────────────────────────────────────────
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_,
         bool MaskInSmem_ = true, bool QInRegs_ = false,
         typename elem_type = cutlass::bfloat16_t,
         typename Base = Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct FA_mask_kernel_traits_sm120 : public Base {
    using Element        = typename Base::Element;
    using ElementAccum   = typename Base::ElementAccum;
    using index_t        = typename Base::index_t;
    using SmemCopyAtom   = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr int kNWarps  = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kBlockM  = kBlockM_;
    static constexpr int kBlockN  = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kStages  = kStages_;
    // MaskInSmem_=false 时，mask 不经 smem/TMA，由 consumer 线程按 C fragment 布局
    // 直接从 gmem 读入寄存器（mask 已由 host 双向 pad，无需 predicate）。
    // 收益：省下的 smem 可提高 occupancy（d64 → 2 CTA/SM）或加大 kBlockM。
    // mask 在 gmem 中被同 batch 的所有 head 复用，L2 命中率高。
    static constexpr bool kMaskInSmem = MaskInSmem_;
    // QInRegs_=true 时，Q 不经 smem/TMA，prologue 直接从 gmem 预取到 A fragment 寄存器
    // （越界行谓词清零，语义同 TMA 补 0）。省下的 sQ smem 可换取更深的 K/V 流水级数，
    // 是 sm120（无 wgmma smem 操作数）上缓解寄存器/smem 压力的主要手段。
    // 代价：Q fragment 常驻寄存器（d128/8warp 为 32 regs/thread）。
    static constexpr bool kQInRegs = QInRegs_;
    // Epilogue sO 复用 sQ（!kQInRegs）或 sK 区域（kQInRegs），需 kStages*kBlockN >= kBlockM
    static_assert(!kQInRegs || kStages_ * kBlockN_ >= kBlockM_,
                  "QInRegs epilogue reuses sK smem: need kStages*kBlockN >= kBlockM");
    static_assert(kHeadDim % 64 == 0, "sm120 path expects kHeadDim multiple of 64 (SW128 TMA smem atom)");

    static constexpr int kBlockKSmem = 64;

    // MMA：与 FA2 相同的 mma.sync TiledMMA（sm120 无 wgmma/tcgen05）
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<16 * kNWarps>, _16, _16>>;

    // ── Smem layouts（TMA 兼容）─────────────────────────────────────────
    // 必须使用 GMMA 规范 atom（byte 域 Swizzle<3,4,3> + smem_ptr flag）：
    // cute 的 make_tma_copy 只接受这种表达来编程 TMA descriptor 的 SW128 模式。
    // 它与 FA2 的元素域 Swizzle<3,3,3>∘(8,64) 在 16-bit 下数值等价，
    // 因此 ldmatrix（SmemCopyAtom）读取路径与 FA2 完全一致。
    using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>;

    using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));
    using SmemLayoutKVstage = decltype(take<0, 2>(SmemLayoutKV{}));   // (kBlockN, kHeadDim)

    // sVt(d, n, s) = sV(n, d, s)，供 PV gemm 以 LDSM_T 读取
    using SmemLayoutVtransposed = decltype(composition(
        SmemLayoutKV{},
        make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
                    make_stride(Int<kBlockN>{}, _1{}, Int<kBlockN * kHeadDim>{}))));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    // Mask：(kBlockM, kBlockN, kStages)；!kMaskInSmem 时仅保留形状（不被使用），smem 尺寸计 0
    static constexpr int kBlockKSmemMask = kBlockN % 64 == 0 ? 64 : 32;
    using SmemLayoutAtomMask = std::conditional_t<kBlockKSmemMask == 64,
        GMMA::Layout_K_SW128_Atom<Element>, GMMA::Layout_K_SW64_Atom<Element>>;
    using SmemLayoutMask = decltype(tile_to_shape(
        SmemLayoutAtomMask{}, Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>{}));
    using SmemLayoutMaskStage = decltype(take<0, 2>(SmemLayoutMask{}));

    // Epilogue O：复用 sQ smem
    using SmemLayoutO = SmemLayoutQ;
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    // ── Smem 尺寸（含 mbarrier）──────────────────────────────────────────
    static constexpr int kSmemQSize    = kQInRegs ? 0 : size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize    = size(SmemLayoutKV{})   * sizeof(Element);
    static constexpr int kSmemVSize    = kSmemKSize;
    static constexpr int kSmemMaskSize = kMaskInSmem ? size(SmemLayoutMask{}) * sizeof(Element) : 0;
    static constexpr int kSmemBarrierOffset = kSmemQSize + kSmemKSize + kSmemVSize + kSmemMaskSize;
    static constexpr int kSmemBarrierSize   = 3 * kStages * 8;  // full_k(tx) + full_vm(tx) + empty
    static constexpr int kSmemSize = kSmemBarrierOffset + kSmemBarrierSize;

    // ── TMA 类型 ─────────────────────────────────────────────────────────
    // gmem 逻辑视图：(seqlen, d, head, batch)，仅 d 维连续（stride=1）
    using ShapeQKV  = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    // mask：(seqlen_q_rounded, seqlen_k_rounded, batch)
    using ShapeMask  = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideMask = cute::Stride<int64_t, _1, int64_t>;

    using TMA_Q = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutQ{}));
    using TMA_K = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutKVstage{}));
    using TMA_V = TMA_K;
    using TMA_Mask = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeMask{}, StrideMask{}),
        SmemLayoutMaskStage{}));
    using TMA_O = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutO{}));

    static constexpr uint32_t TmaTransactionBytesQ    = static_cast<uint32_t>(size(SmemLayoutQ{}) * sizeof(Element));
    static constexpr uint32_t TmaTransactionBytesK    = static_cast<uint32_t>(size(SmemLayoutKVstage{}) * sizeof(Element));
    static constexpr uint32_t TmaTransactionBytesV    = TmaTransactionBytesK;
    static constexpr uint32_t TmaTransactionBytesMask = static_cast<uint32_t>(size(SmemLayoutMaskStage{}) * sizeof(Element));
    // V（+Mask）同一 barrier（都在 QK gemm 之后才消费）
    static constexpr uint32_t TmaTransactionBytesVMStage =
        TmaTransactionBytesV + (kMaskInSmem ? TmaTransactionBytesMask : 0u);
};

// ── 参数（含 TMA descriptor，需 __grid_constant__ 传参）──────────────────────
template<typename Kernel_traits>
struct FA_mask_params_sm120 {
FA_mask_params base;
typename Kernel_traits::TMA_Q    tma_load_Q;
typename Kernel_traits::TMA_K    tma_load_K;
typename Kernel_traits::TMA_V    tma_load_V;
typename Kernel_traits::TMA_Mask tma_load_Mask;
typename Kernel_traits::TMA_O    tma_store_O;
// splitkv 部分结果 O_partial（bf16, (SqR, d, H, B*S)）的 TMA store descriptor，类型同 TMA_O
typename Kernel_traits::TMA_O    tma_store_Op;
};

// ── 核心 kernel ──────────────────────────────────────────────────────────────
template<typename Kernel_traits>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
flash_fwd_mask_kernel_sm120(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __grid_constant__
#endif
    const FA_mask_params_sm120<Kernel_traits> params) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Element      = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages  = Kernel_traits::kStages;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    const int tidx    = threadIdx.x;
    const int m_block = blockIdx.x;
    const int bidb    = blockIdx.y;
    const int bidh    = blockIdx.z;

    const FA_mask_params &p = params.base;
    const int n_block_max = cute::ceil_div(p.seqlen_k, kBlockN);
    const int bidh_kv = bidh / p.h_h_k_ratio;

    // ── Shared memory 排布：[sQ][sK x kStages][sV x kStages][sMask x kStages][barriers]
    // （kQInRegs 时无 sQ 区域；!kMaskInSmem 时无 sMask 区域）
    extern __shared__ char smem_[];
    constexpr int kQElems    = Kernel_traits::kSmemQSize    / int(sizeof(Element));
    constexpr int kKVElems   = Kernel_traits::kSmemKSize    / int(sizeof(Element));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + kQElems,
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + kKVElems,
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sMask = make_tensor(sV.data() + kKVElems,
                               typename Kernel_traits::SmemLayoutMask{});
    Tensor sVt = make_tensor(sV.data(),
                             typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(),
                                      typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    FullBarrier*  full_k_bar  = reinterpret_cast<FullBarrier*>(smem_ + Kernel_traits::kSmemBarrierOffset);
    FullBarrier*  full_vm_bar = full_k_bar + kStages;
    EmptyBarrier* empty_bar   = reinterpret_cast<EmptyBarrier*>(full_vm_bar + kStages);

    if (tidx < kStages) {
        full_k_bar[tidx].init(1);          // 仅 thread0 arrive_and_expect_tx
        full_vm_bar[tidx].init(1);
        empty_bar[tidx].init(kNThreads);   // 所有 consumer 线程各自 arrive
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    // ── TMA 分区 ─────────────────────────────────────────────────────────
    auto shape_Q  = make_shape(p.seqlen_q, p.d, p.h, p.b);
    auto shape_KV = make_shape(p.seqlen_k, p.d, p.h_k, p.b);
    auto shape_Mask = make_shape(p.seqlen_q_rounded, p.seqlen_k_rounded, p.b);

    Tensor mQ = params.tma_load_Q.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
    Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));  // (bM, bD)
    auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));   // (TMA)
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));   // (TMA)

    Tensor mK = params.tma_load_K.get_tma_tensor(shape_KV)(_, _, bidh_kv, bidb);
    Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));  // (bN, bD, n_block)
    auto block_tma_K = params.tma_load_K.get_slice(_0{});
    Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));   // (TMA, n_block)
    Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));   // (TMA, stage)

    Tensor mV = params.tma_load_V.get_tma_tensor(shape_KV)(_, _, bidh_kv, bidb);
    Tensor gV = local_tile(mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
    auto block_tma_V = params.tma_load_V.get_slice(_0{});
    Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV));
    Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV));

    Tensor mMask = params.tma_load_Mask.get_tma_tensor(shape_Mask)(_, _, bidb);
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));  // (bM, bN, n_block)
    auto block_tma_Mask = params.tma_load_Mask.get_slice(_0{});
    Tensor tMaskgMask = group_modes<0, 3>(block_tma_Mask.partition_S(gMask));  // (TMA, n_block)
    Tensor tMasksMask = group_modes<0, 3>(block_tma_Mask.partition_D(sMask));  // (TMA, stage)

    // mask gmem 直读路径（kMaskInSmem=false）：不经 TMA/smem，直接 ldg 到 C fragment
    Tensor mMaskG = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(p.mask_ptr) + bidb * p.mask_batch_stride),
        make_shape(p.seqlen_q_rounded, p.seqlen_k_rounded),
        make_stride(p.mask_row_stride, _1{}));
    Tensor gMaskG = local_tile(mMaskG, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

    // 发射一个 tile 的 TMA loads（仅 thread0 调用）；with_q 时顺带加载 Q（仅 stage0 第一次）
    // 双 full barrier：K 单独一组（QK gemm 只需等 K），V+Mask 一组（QK 之后才消费），
    // 避免 V/Mask 的传输延迟阻塞 QK 启动。
    // cache hint：K/V/Mask 会被同 batch/head 的其他 CTA 复用 → EVICT_LAST 保留在 L2；
    // Q 每个 CTA 只读一次 → EVICT_FIRST。
    auto issue_tile = [&](int n_block, int stage, bool with_q) {
        uint64_t& bar_k  = reinterpret_cast<uint64_t&>(full_k_bar[stage]);
        uint64_t& bar_vm = reinterpret_cast<uint64_t&>(full_vm_bar[stage]);
        full_k_bar[stage].arrive_and_expect_tx(
            Kernel_traits::TmaTransactionBytesK +
            (with_q ? Kernel_traits::TmaTransactionBytesQ : 0u));
        full_vm_bar[stage].arrive_and_expect_tx(Kernel_traits::TmaTransactionBytesVMStage);
        if (with_q) {
            cute::copy(params.tma_load_Q.with(bar_k, 0, TMA::CacheHintSm90::EVICT_FIRST), tQgQ, tQsQ);
        }
        cute::copy(params.tma_load_K.with(bar_k, 0, TMA::CacheHintSm90::EVICT_LAST), tKgK(_, n_block), tKsK(_, stage));
        cute::copy(params.tma_load_V.with(bar_vm, 0, TMA::CacheHintSm90::EVICT_LAST), tVgV(_, n_block), tVsV(_, stage));
        if constexpr (Kernel_traits::kMaskInSmem) {
            cute::copy(params.tma_load_Mask.with(bar_vm, 0, TMA::CacheHintSm90::EVICT_LAST), tMaskgMask(_, n_block), tMasksMask(_, stage));
        }
    };

    // ── MMA / smem→reg copy handles ──────────────────────────────────────
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
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask);  // (CPY, M, N, stage)

    // mask gmem 直读的 tiled copy（按 C fragment 布局，32-bit 向量化）
    auto gmem_tiled_copy_mask = make_tiled_copy_C(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, Element>{}, tiled_mma);
    auto gmem_thr_copy_mask   = gmem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tMgMask = gmem_thr_copy_mask.partition_S(gMaskG);  // (CPY, M, N, n_block)
    Tensor rMaskG  = make_tensor<Element>(
        partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto tMrMaskG_view = gmem_thr_copy_mask.retile_D(rMaskG);

    // mask → acc_s 纯加法（越界列已由 host pad 成 -inf）
    // smem 版：TMA 已全 tile 搬运；gmem 版：调用前需先 issue 对应 n_block 的 ldg
    // mask 语义对齐 SDPA：softmax(S·scale + mask)。mask 在 scale 之前加到未缩放的 acc_s 上，
    // 因此必须乘 1/scale（-inf × 正数仍为 -inf，0/-inf mask 行为不变；有限值 mask 此前被错误缩放）
    const float mask_inv_scale = 1.f / p.scale_softmax;
    auto apply_mask_from_smem = [&](auto &acc_s, int stage) {
        Tensor rMask = make_tensor<Element>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
    cute::copy(smem_tiled_copy_mask, tSsMask(_, _, _, stage), tSrMask_view);
    #pragma unroll
    for (int i = 0; i < size(acc_s); ++i) {
            acc_s(i) += static_cast<float>(rMask(i)) * mask_inv_scale;
        }
    };
    auto apply_mask_from_gmem = [&](auto &acc_s) {
        #pragma unroll
        for (int i = 0; i < size(acc_s); ++i) {
            acc_s(i) += static_cast<float>(rMaskG(i)) * mask_inv_scale;
        }
    };

    // ── Prologue ─────────────────────────────────────────────────────────
    // kQInRegs：Q 直接 gmem→寄存器（越界行谓词清零，语义同 TMA 补 0；d==kHeadDim 由 host 保证）
    if constexpr (Kernel_traits::kQInRegs) {
        Tensor mQg = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element const*>(p.q_ptr)
                          + bidb * p.q_batch_stride + bidh * p.q_head_stride),
            make_shape(p.seqlen_q, p.d), make_stride(p.q_row_stride, _1{}));
        Tensor gQg = local_tile(mQg, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        Tensor tQrgQ = thr_mma.partition_A(gQg);   // (MMA, MMA_M, MMA_K) gmem view
        Tensor cQ    = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tQcQ  = thr_mma.partition_A(cQ);
        const int q_rows = p.seqlen_q - m_block * kBlockM;
        clear(tSrQ);
        #pragma unroll
        for (int i = 0; i < size(tSrQ); ++i) {
            if (get<0>(tQcQ(i)) < q_rows) { tSrQ(i) = tQrgQ(i); }
        }
    }
    // thread0 预发射前 kStages 个 tile（stage0 顺带加载 Q，仅 !kQInRegs）
    if (tidx == 0) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_Mask.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
        const int n_prologue = n_block_max < kStages ? n_block_max : kStages;
        for (int s = 0; s < n_prologue; ++s) {
            issue_tile(s, s, /*with_q=*/(s == 0 && !Kernel_traits::kQInRegs));
        }
    }

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

    // ── 主循环：升序遍历 n_block，stage = j % kStages ─────────────────────
    for (int j = 0; j < n_block_max; ++j) {
        const int stage = j % kStages;
        const uint32_t phase = (j / kStages) & 1;

        // 等待 K（j=0 且 !kQInRegs 时还包括 Q）就绪即可启动 QK gemm
        full_k_bar[stage].wait(phase);

        // gmem 直读 mask：在 QK gemm 前发射 ldg，用 QK 计算掩盖访存延迟
        if constexpr (!Kernel_traits::kMaskInSmem) {
            cute::copy(gmem_tiled_copy_mask, tMgMask(_, _, _, j), tMrMaskG_view);
        }

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);

        // QK^T（K: smem → ldmatrix → mma.sync；Q 依配置来自 smem 或寄存器）
        if constexpr (Kernel_traits::kQInRegs) {
            Tensor tSrK_view = smem_thr_copy_K.retile_D(tSrK);
            cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}, stage), tSrK_view(_, _, _0{}));
            #pragma unroll
            for (int i = 0; i < size<2>(tSrK); ++i) {
                if (i + 1 < size<2>(tSrK)) {
                    cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1, stage), tSrK_view(_, _, i + 1));
                }
                cute::gemm(tiled_mma, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
            }
        } else {
            FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, stage), tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
        }

        // V（smem 版 Mask）在 QK 之后消费
        full_vm_bar[stage].wait(phase);

        // 加法 mask
        if constexpr (Kernel_traits::kMaskInSmem) {
            apply_mask_from_smem(acc_s, stage);
        } else {
            apply_mask_from_gmem(acc_s);
        }

        // online softmax + rescale acc_o
        if (j == 0) {
            softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                acc_s, acc_o, p.scale_softmax_log2);
        } else {
            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                acc_s, acc_o, p.scale_softmax_log2);
        }

        // P·V
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(),
            FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, stage),
                                 tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // 本 stage 消费完毕
        empty_bar[stage].arrive();

        // thread0 发射 kStages 步之后的 tile（复用当前 stage）
        if (tidx == 0) {
            const int j2 = j + kStages;
            if (j2 < n_block_max) {
                // 等待本 stage 第 (j/kStages) 次 empty 完成（parity 与 full 相同）
                empty_bar[stage].wait(phase);
                issue_tile(j2, stage, /*with_q=*/false);
            }
        }
    }

    // ── Epilogue：归一化 → sO（复用 sQ / sK smem）→ TMA store ────────────
    softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
        acc_o, p.scale_softmax, /*rp_dropout=*/1.0f);

    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    // kQInRegs 时无 sQ 区域，sO 复用 sK（kStages*kBlockN >= kBlockM 由 traits 保证）
    Tensor sO = make_tensor(Kernel_traits::kQInRegs ? sK.data() : sQ.data(),
                            typename Kernel_traits::SmemLayoutO{});
    // sm120a 支持 stmatrix：C 累加器 → swizzled sO 用 STSM（与 FA3 epilogue 相同），
    // 比 AutoVectorizing<128> 逐线程写指令更少、无 bank conflict
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
#else
    using SmemCopyAtomO = typename Kernel_traits::SmemCopyAtomO;
#endif
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    __syncthreads();   // 确保所有线程已读完 sQ/sK（sO 与其别名）
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared();  // 通用代理写对 TMA（async 代理）可见
    __syncthreads();

    if (tidx == 0) {
        Tensor mO = params.tma_store_O.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
        Tensor gO = local_tile(mO, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        auto block_tma_O = params.tma_store_O.get_slice(_0{});
        Tensor tOsO = group_modes<0, 3>(block_tma_O.partition_S(sO));
        Tensor tOgO = group_modes<0, 3>(block_tma_O.partition_D(gO));
        cute::copy(params.tma_store_O, tOsO, tOgO);
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }
#else
    // 非 sm90+/sm120 编译目标：空 kernel（host 侧不会在该架构下分发到这里）
    (void)params;
#endif
}

// ── host-side launcher ───────────────────────────────────────────────────────
template<typename Kernel_traits>
void run_flash_fwd_mask_sm120(const FA_mask_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.q_ptr)),
        make_shape(params.seqlen_q, params.d, params.h, params.b),
        make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride));
    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.k_ptr)),
        make_shape(params.seqlen_k, params.d, params.h_k, params.b),
        make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride));
    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.v_ptr)),
        make_shape(params.seqlen_k, params.d, params.h_k, params.b),
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride));
    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.mask_ptr)),
        make_shape(params.seqlen_q_rounded, params.seqlen_k_rounded, params.b),
        make_stride(params.mask_row_stride, _1{}, params.mask_batch_stride));
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)),
        make_shape(params.seqlen_q, params.d, params.h, params.b),
        make_stride(params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride));

    FA_mask_params_sm120<Kernel_traits> kparams;
    kparams.base = params;
    kparams.tma_load_Q    = make_tma_copy(SM90_TMA_LOAD{},  mQ,    typename Kernel_traits::SmemLayoutQ{});
    kparams.tma_load_K    = make_tma_copy(SM90_TMA_LOAD{},  mK,    typename Kernel_traits::SmemLayoutKVstage{});
    kparams.tma_load_V    = make_tma_copy(SM90_TMA_LOAD{},  mV,    typename Kernel_traits::SmemLayoutKVstage{});
    kparams.tma_load_Mask = make_tma_copy(SM90_TMA_LOAD{},  mMask, typename Kernel_traits::SmemLayoutMaskStage{});
    kparams.tma_store_O   = make_tma_copy(SM90_TMA_STORE{}, mO,    typename Kernel_traits::SmemLayoutO{});

    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    static_assert(smem_size <= 101376, "sm120 per-CTA dynamic smem limit exceeded");

    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    dim3 grid(num_m_block, params.b, params.h);

    auto kernel = &flash_fwd_mask_kernel_sm120<Kernel_traits>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(kparams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ── Persistent kernel ───────────────────────────────────────────────────────
// 1D 步长调度：每个 CTA 从线性工作队列中取任务（m_block, bidh, bidb），
// 处理完一个后 +gridDim.x 取下一个，直到队列耗尽。
// 消除大 grid 下最后几个 CTAs 的 tail 效应，匹配 FA3 StaticPersistentTileScheduler 的思路。
template<typename Kernel_traits>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
flash_fwd_mask_kernel_sm120_persistent(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __grid_constant__
#endif
    const FA_mask_params_sm120<Kernel_traits> params) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Element      = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages  = Kernel_traits::kStages;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    const FA_mask_params &p = params.base;
    const int total_m_blocks = cute::ceil_div(p.seqlen_q, kBlockM);
    const int total_blocks = total_m_blocks * p.h * p.b;

    // 步长式取任务
    int work_idx = blockIdx.x;
    while (work_idx < total_blocks) {
        int t = work_idx;
        int m_block = t % total_m_blocks; t /= total_m_blocks;
        int bidh    = t % p.h; t /= p.h;
        int bidb    = t;

        const int n_block_max = cute::ceil_div(p.seqlen_k, kBlockN);
        const int bidh_kv = bidh / p.h_h_k_ratio;

        // 每次迭代重新初始化 smem barrier（每个 tile 独立，regions 由 m_block 保证不重叠）
        extern __shared__ char smem_[];
        constexpr int kQElems  = Kernel_traits::kSmemQSize / int(sizeof(Element));
        constexpr int kKVElems = Kernel_traits::kSmemKSize / int(sizeof(Element));
        Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                                typename Kernel_traits::SmemLayoutQ{});
        Tensor sK = make_tensor(sQ.data() + kQElems, typename Kernel_traits::SmemLayoutKV{});
        Tensor sV = make_tensor(sK.data() + kKVElems, typename Kernel_traits::SmemLayoutKV{});
        Tensor sMask = make_tensor(sV.data() + kKVElems, typename Kernel_traits::SmemLayoutMask{});
        Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
        Tensor sVtNoSwizzle = make_tensor(sV.data().get(),
                                          typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

        FullBarrier*  full_k_bar  = reinterpret_cast<FullBarrier*>(smem_ + Kernel_traits::kSmemBarrierOffset);
        FullBarrier*  full_vm_bar = full_k_bar + kStages;
        EmptyBarrier* empty_bar   = reinterpret_cast<EmptyBarrier*>(full_vm_bar + kStages);

        if (threadIdx.x < kStages) {
            full_k_bar[threadIdx.x].init(1);
            full_vm_bar[threadIdx.x].init(1);
            empty_bar[threadIdx.x].init(kNThreads);
        }
        cutlass::arch::fence_barrier_init();
        __syncthreads();

        // TMA & MMA handles (与原始 kernel 相同，使用局部的 m_block/bidh/bidb)
        auto shape_Q  = make_shape(p.seqlen_q, p.d, p.h, p.b);
        auto shape_KV = make_shape(p.seqlen_k, p.d, p.h_k, p.b);
        auto shape_Mask = make_shape(p.seqlen_q_rounded, p.seqlen_k_rounded, p.b);

        Tensor mQ = params.tma_load_Q.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
        Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));

        Tensor mK = params.tma_load_K.get_tma_tensor(shape_KV)(_, _, bidh_kv, bidb);
        Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
        auto block_tma_K = params.tma_load_K.get_slice(_0{});
        Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));

        Tensor mV = params.tma_load_V.get_tma_tensor(shape_KV)(_, _, bidh_kv, bidb);
        Tensor gV = local_tile(mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
        auto block_tma_V = params.tma_load_V.get_slice(_0{});
        Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV));
        Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV));

        Tensor mMask = params.tma_load_Mask.get_tma_tensor(shape_Mask)(_, _, bidb);
        Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));
        auto block_tma_Mask = params.tma_load_Mask.get_slice(_0{});
        Tensor tMaskgMask = group_modes<0, 3>(block_tma_Mask.partition_S(gMask));
        Tensor tMasksMask = group_modes<0, 3>(block_tma_Mask.partition_D(sMask));

        Tensor mMaskG = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element const*>(p.mask_ptr) + bidb * p.mask_batch_stride),
            make_shape(p.seqlen_q_rounded, p.seqlen_k_rounded),
            make_stride(p.mask_row_stride, _1{}));
        Tensor gMaskG = local_tile(mMaskG, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

        auto issue_tile = [&](int n_block, int stage, bool with_q) {
            uint64_t& bar_k  = reinterpret_cast<uint64_t&>(full_k_bar[stage]);
            uint64_t& bar_vm = reinterpret_cast<uint64_t&>(full_vm_bar[stage]);
            full_k_bar[stage].arrive_and_expect_tx(
                Kernel_traits::TmaTransactionBytesK +
                (with_q ? Kernel_traits::TmaTransactionBytesQ : 0u));
            full_vm_bar[stage].arrive_and_expect_tx(Kernel_traits::TmaTransactionBytesVMStage);
            if (with_q) {
                cute::copy(params.tma_load_Q.with(bar_k, 0, TMA::CacheHintSm90::EVICT_FIRST), tQgQ, tQsQ);
            }
            cute::copy(params.tma_load_K.with(bar_k, 0, TMA::CacheHintSm90::EVICT_LAST), tKgK(_, n_block), tKsK(_, stage));
            cute::copy(params.tma_load_V.with(bar_vm, 0, TMA::CacheHintSm90::EVICT_LAST), tVgV(_, n_block), tVsV(_, stage));
            if constexpr (Kernel_traits::kMaskInSmem) {
                cute::copy(params.tma_load_Mask.with(bar_vm, 0, TMA::CacheHintSm90::EVICT_LAST), tMaskgMask(_, n_block), tMasksMask(_, stage));
            }
        };

        // MMA / smem handles
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
        Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
        Tensor tSrK  = thr_mma.partition_fragment_B(sK(_, _, _0{}));
        Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
        Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK);
        auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
        Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

        auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_mask   = smem_tiled_copy_mask.get_thread_slice(threadIdx.x);
        Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask);
        auto gmem_tiled_copy_mask = make_tiled_copy_C(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, Element>{}, tiled_mma);
        auto gmem_thr_copy_mask   = gmem_tiled_copy_mask.get_thread_slice(threadIdx.x);
        Tensor tMgMask = gmem_thr_copy_mask.partition_S(gMaskG);
        Tensor rMaskG  = make_tensor<Element>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tMrMaskG_view = gmem_thr_copy_mask.retile_D(rMaskG);

const float mask_inv_scale = 1.f / p.scale_softmax;
auto apply_mask_from_smem = [&](auto &acc, int s) {
Tensor rM = make_tensor<Element>(partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
auto tV = smem_thr_copy_mask.retile_D(rM);
cute::copy(smem_tiled_copy_mask, tSsMask(_, _, _, s), tV);
#pragma unroll
for (int i = 0; i < size(acc); ++i) { acc(i) += static_cast<float>(rM(i)) * mask_inv_scale; }
};
auto apply_mask_from_gmem = [&](auto &acc) {
#pragma unroll
for (int i = 0; i < size(acc); ++i) { acc(i) += static_cast<float>(rMaskG(i)) * mask_inv_scale; }
};

        if constexpr (Kernel_traits::kQInRegs) {
            Tensor mQg = make_tensor(
                make_gmem_ptr(reinterpret_cast<Element const*>(p.q_ptr) + bidb * p.q_batch_stride + bidh * p.q_head_stride),
                make_shape(p.seqlen_q, p.d), make_stride(p.q_row_stride, _1{}));
            Tensor gQg = local_tile(mQg, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
            Tensor tQrgQ = thr_mma.partition_A(gQg);
            Tensor cQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
            Tensor tQcQ = thr_mma.partition_A(cQ);
            const int q_rows = p.seqlen_q - m_block * kBlockM;
            clear(tSrQ);
            #pragma unroll
            for (int i = 0; i < size(tSrQ); ++i) { if (get<0>(tQcQ(i)) < q_rows) tSrQ(i) = tQrgQ(i); }
        }

        // Prologue: issue TMA
        if (threadIdx.x == 0) {
            cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_Mask.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
            const int n_prologue = n_block_max < kStages ? n_block_max : kStages;
            for (int s = 0; s < n_prologue; ++s) {
                issue_tile(s, s, (s == 0 && !Kernel_traits::kQInRegs));
            }
        }

        clear(acc_o);
        FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

        for (int j = 0; j < n_block_max; ++j) {
            const int stage = j % kStages;
            const uint32_t phase = (j / kStages) & 1;
            full_k_bar[stage].wait(phase);

            if constexpr (!Kernel_traits::kMaskInSmem) {
                cute::copy(gmem_tiled_copy_mask, tMgMask(_, _, _, j), tMrMaskG_view);
            }
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            if constexpr (Kernel_traits::kQInRegs) {
                Tensor tSrK_view = smem_thr_copy_K.retile_D(tSrK);
                cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}, stage), tSrK_view(_, _, _0{}));
                #pragma unroll
                for (int i = 0; i < size<2>(tSrK); ++i) {
                    if (i + 1 < size<2>(tSrK)) cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1, stage), tSrK_view(_, _, i + 1));
                    cute::gemm(tiled_mma, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
                }
            } else {
                FLASH_NAMESPACE::gemm<false>(acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, stage), tiled_mma,
                    smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
            }

            full_vm_bar[stage].wait(phase);

            if constexpr (Kernel_traits::kMaskInSmem) apply_mask_from_smem(acc_s, stage);
            else apply_mask_from_gmem(acc_s);

            if (j == 0) softmax.template softmax_rescale_o<true, true>(acc_s, acc_o, p.scale_softmax_log2);
            else softmax.template softmax_rescale_o<false, true>(acc_s, acc_o, p.scale_softmax_log2);

            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, stage),
                                     tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            empty_bar[stage].arrive();

            if (threadIdx.x == 0) {
                const int j2 = j + kStages;
                if (j2 < n_block_max) {
                    empty_bar[stage].wait(phase);
                    issue_tile(j2, stage, false);
                }
            }
        }

        // Epilogue
        softmax.template normalize_softmax_lse<false>(acc_o, p.scale_softmax, 1.0f);
        Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
        Tensor sO = make_tensor(Kernel_traits::kQInRegs ? sK.data() : sQ.data(),
                                typename Kernel_traits::SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

        __syncthreads();
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
        cutlass::arch::fence_view_async_shared();
        __syncthreads();

        if (threadIdx.x == 0) {
            Tensor mO = params.tma_store_O.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
            Tensor gO = local_tile(mO, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
            auto block_tma_O = params.tma_store_O.get_slice(_0{});
            Tensor tOsO = group_modes<0, 3>(block_tma_O.partition_S(sO));
            Tensor tOgO = group_modes<0, 3>(block_tma_O.partition_D(gO));
            cute::copy(params.tma_store_O, tOsO, tOgO);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }

        // 下一个工作项
        work_idx += gridDim.x;
    }
#else
    (void)params;
#endif
}

// ── Persistent kernel launcher ───────────────────────────────────────────────
// 发布 1D grid（2× num_SMs，给调度器留弹性），每个 CTA 从线性工作队列中步长式取任务，
// 消除尾效应，提升小 batch/head 场景的 SM 利用率（FA3 StaticPersistentTileScheduler 的简化版）。
template<typename Kernel_traits>
void run_flash_fwd_mask_sm120_persistent(const FA_mask_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.q_ptr)),
        make_shape(params.seqlen_q, params.d, params.h, params.b),
        make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride));
    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.k_ptr)),
        make_shape(params.seqlen_k, params.d, params.h_k, params.b),
        make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride));
    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.v_ptr)),
        make_shape(params.seqlen_k, params.d, params.h_k, params.b),
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride));
    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.mask_ptr)),
        make_shape(params.seqlen_q_rounded, params.seqlen_k_rounded, params.b),
        make_stride(params.mask_row_stride, _1{}, params.mask_batch_stride));
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)),
        make_shape(params.seqlen_q, params.d, params.h, params.b),
        make_stride(params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride));

    FA_mask_params_sm120<Kernel_traits> kparams;
    kparams.base = params;
    kparams.tma_load_Q    = make_tma_copy(SM90_TMA_LOAD{},  mQ,    typename Kernel_traits::SmemLayoutQ{});
    kparams.tma_load_K    = make_tma_copy(SM90_TMA_LOAD{},  mK,    typename Kernel_traits::SmemLayoutKVstage{});
    kparams.tma_load_V    = make_tma_copy(SM90_TMA_LOAD{},  mV,    typename Kernel_traits::SmemLayoutKVstage{});
    kparams.tma_load_Mask = make_tma_copy(SM90_TMA_LOAD{},  mMask, typename Kernel_traits::SmemLayoutMaskStage{});
    kparams.tma_store_O   = make_tma_copy(SM90_TMA_STORE{}, mO,    typename Kernel_traits::SmemLayoutO{});

    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    static_assert(smem_size <= 101376, "sm120 per-CTA dynamic smem limit exceeded");

    // 获取 SM 数
    int num_sm = 0;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    // 2× SM 数启动，让调度器有弹性
    const int grid_size = num_sm * 2;

    auto kernel = &flash_fwd_mask_kernel_sm120_persistent<Kernel_traits>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid_size, Kernel_traits::kNThreads, smem_size, stream>>>(kparams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ── Split-KV kernel ──────────────────────────────────────────────────────────
// 参考 gemmbf16fp32.cu 的 split-K 思路与 FA2 splitkv：
//   grid = (num_m_blocks * num_splits, B, H)，每个 CTA 处理 K 维的一段连续子范围
//   [split_idx * n_per_split, min(n_block_max, (split_idx+1) * n_per_split))
// Epilogue 不写最终 O，而是写部分结果（FA3 约定：O_s 已按本地 l 归一化，LSE_s = m*scale + log l）：
//   O_partial:   (num_splits, B, H, seqlen_q_rounded, d) bf16 —— STSM→smem→TMA store（流量减半）
//   LSE_partial: (num_splits, B, H, seqlen_q_rounded) fp32 —— quad leader scattered 写
// 之后由 combine kernel 做 scale_s = exp(LSE_s - max) / Σ 加权归约。
template<typename Kernel_traits>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1)
flash_fwd_mask_kernel_sm120_splitkv(
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __grid_constant__
#endif
    const FA_mask_params_sm120<Kernel_traits> params) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Element      = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages  = Kernel_traits::kStages;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;

    const int tidx = threadIdx.x;

    const FA_mask_params &p = params.base;
    const int num_m_blocks = cute::ceil_div(p.seqlen_q, kBlockM);
    // x 维解码：低 m_block 位变化快（同 (b,h,split) 的 CTA 共享 K/V，L2 友好）
    const int m_block   = blockIdx.x % num_m_blocks;
    const int split_idx = blockIdx.x / num_m_blocks;
    const int bidb      = blockIdx.y;
    const int bidh      = blockIdx.z;

    const int n_block_max = cute::ceil_div(p.seqlen_k, kBlockN);
    const int bidh_kv = bidh / p.h_h_k_ratio;

    // 本 split 的 K 范围（连续区间，TMA/L2 友好）
    const int n_per_split = cute::ceil_div(n_block_max, p.num_splits);
    const int n_begin = split_idx * n_per_split;
    const int n_end   = (n_begin + n_per_split < n_block_max) ? (n_begin + n_per_split) : n_block_max;
    const int n_tiles = n_end - n_begin;   // 可能 <= 0（尾部分裂为空）

    extern __shared__ char smem_[];
    constexpr int kQElems  = Kernel_traits::kSmemQSize / int(sizeof(Element));
    constexpr int kKVElems = Kernel_traits::kSmemKSize / int(sizeof(Element));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + kQElems, typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + kKVElems, typename Kernel_traits::SmemLayoutKV{});
    Tensor sMask = make_tensor(sV.data() + kKVElems, typename Kernel_traits::SmemLayoutMask{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(),
                                      typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    FullBarrier*  full_k_bar  = reinterpret_cast<FullBarrier*>(smem_ + Kernel_traits::kSmemBarrierOffset);
    FullBarrier*  full_vm_bar = full_k_bar + kStages;
    EmptyBarrier* empty_bar   = reinterpret_cast<EmptyBarrier*>(full_vm_bar + kStages);

    if (tidx < kStages) {
        full_k_bar[tidx].init(1);
        full_vm_bar[tidx].init(1);
        empty_bar[tidx].init(kNThreads);
    }
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    auto shape_Q  = make_shape(p.seqlen_q, p.d, p.h, p.b);
    auto shape_KV = make_shape(p.seqlen_k, p.d, p.h_k, p.b);
    auto shape_Mask = make_shape(p.seqlen_q_rounded, p.seqlen_k_rounded, p.b);

    Tensor mQ = params.tma_load_Q.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
    Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
    auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));

    Tensor mK = params.tma_load_K.get_tma_tensor(shape_KV)(_, _, bidh_kv, bidb);
    Tensor gK = local_tile(mK, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
    auto block_tma_K = params.tma_load_K.get_slice(_0{});
    Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));
    Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));

    Tensor mV = params.tma_load_V.get_tma_tensor(shape_KV)(_, _, bidh_kv, bidb);
    Tensor gV = local_tile(mV, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, _0{}));
    auto block_tma_V = params.tma_load_V.get_slice(_0{});
    Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV));
    Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV));

    Tensor mMask = params.tma_load_Mask.get_tma_tensor(shape_Mask)(_, _, bidb);
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));
    auto block_tma_Mask = params.tma_load_Mask.get_slice(_0{});
    Tensor tMaskgMask = group_modes<0, 3>(block_tma_Mask.partition_S(gMask));
    Tensor tMasksMask = group_modes<0, 3>(block_tma_Mask.partition_D(sMask));

    Tensor mMaskG = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(p.mask_ptr) + bidb * p.mask_batch_stride),
        make_shape(p.seqlen_q_rounded, p.seqlen_k_rounded),
        make_stride(p.mask_row_stride, _1{}));
    Tensor gMaskG = local_tile(mMaskG, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

    // n_block 为全局 K 块索引；stage 为本 split 内的局部流水级
    auto issue_tile = [&](int n_block, int stage, bool with_q) {
        uint64_t& bar_k  = reinterpret_cast<uint64_t&>(full_k_bar[stage]);
        uint64_t& bar_vm = reinterpret_cast<uint64_t&>(full_vm_bar[stage]);
        full_k_bar[stage].arrive_and_expect_tx(
            Kernel_traits::TmaTransactionBytesK +
            (with_q ? Kernel_traits::TmaTransactionBytesQ : 0u));
        full_vm_bar[stage].arrive_and_expect_tx(Kernel_traits::TmaTransactionBytesVMStage);
        if (with_q) {
            cute::copy(params.tma_load_Q.with(bar_k, 0, TMA::CacheHintSm90::EVICT_FIRST), tQgQ, tQsQ);
        }
        cute::copy(params.tma_load_K.with(bar_k, 0, TMA::CacheHintSm90::EVICT_LAST), tKgK(_, n_block), tKsK(_, stage));
        cute::copy(params.tma_load_V.with(bar_vm, 0, TMA::CacheHintSm90::EVICT_LAST), tVgV(_, n_block), tVsV(_, stage));
        if constexpr (Kernel_traits::kMaskInSmem) {
            cute::copy(params.tma_load_Mask.with(bar_vm, 0, TMA::CacheHintSm90::EVICT_LAST), tMaskgMask(_, n_block), tMasksMask(_, stage));
        }
    };

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
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_mask   = smem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask);
    auto gmem_tiled_copy_mask = make_tiled_copy_C(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, Element>{}, tiled_mma);
    auto gmem_thr_copy_mask   = gmem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tMgMask = gmem_thr_copy_mask.partition_S(gMaskG);
    Tensor rMaskG  = make_tensor<Element>(
        partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto tMrMaskG_view = gmem_thr_copy_mask.retile_D(rMaskG);

    const float mask_inv_scale = 1.f / p.scale_softmax;   // 见主 kernel 同名注释
    auto apply_mask_from_smem = [&](auto &acc_s, int stage) {
        Tensor rMask = make_tensor<Element>(
            partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
cute::copy(smem_tiled_copy_mask, tSsMask(_, _, _, stage), tSrMask_view);
#pragma unroll
for (int i = 0; i < size(acc_s); ++i) { acc_s(i) += static_cast<float>(rMask(i)) * mask_inv_scale; }
};
auto apply_mask_from_gmem = [&](auto &acc_s) {
#pragma unroll
for (int i = 0; i < size(acc_s); ++i) { acc_s(i) += static_cast<float>(rMaskG(i)) * mask_inv_scale; }
};

    if constexpr (Kernel_traits::kQInRegs) {
        Tensor mQg = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element const*>(p.q_ptr)
                          + bidb * p.q_batch_stride + bidh * p.q_head_stride),
            make_shape(p.seqlen_q, p.d), make_stride(p.q_row_stride, _1{}));
        Tensor gQg = local_tile(mQg, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        Tensor tQrgQ = thr_mma.partition_A(gQg);
        Tensor cQ    = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
        Tensor tQcQ  = thr_mma.partition_A(cQ);
        const int q_rows = p.seqlen_q - m_block * kBlockM;
        clear(tSrQ);
        #pragma unroll
        for (int i = 0; i < size(tSrQ); ++i) { if (get<0>(tQcQ(i)) < q_rows) { tSrQ(i) = tQrgQ(i); } }
    }

    clear(acc_o);
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    auto lse = make_fragment_like(softmax.row_sum);

    if (n_tiles > 0) {
        // Prologue: 发射本 split 前 kStages 个 tile
        if (tidx == 0) {
            cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
            cute::prefetch_tma_descriptor(params.tma_load_Mask.get_tma_descriptor());
            const int n_prologue = n_tiles < kStages ? n_tiles : kStages;
            for (int s = 0; s < n_prologue; ++s) {
                issue_tile(n_begin + s, s, /*with_q=*/(s == 0 && !Kernel_traits::kQInRegs));
            }
        }

        for (int j = 0; j < n_tiles; ++j) {
            const int n_block = n_begin + j;
            const int stage = j % kStages;
            const uint32_t phase = (j / kStages) & 1;
            full_k_bar[stage].wait(phase);

            if constexpr (!Kernel_traits::kMaskInSmem) {
                cute::copy(gmem_tiled_copy_mask, tMgMask(_, _, _, n_block), tMrMaskG_view);
            }
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);

            if constexpr (Kernel_traits::kQInRegs) {
                Tensor tSrK_view = smem_thr_copy_K.retile_D(tSrK);
                cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}, stage), tSrK_view(_, _, _0{}));
                #pragma unroll
                for (int i = 0; i < size<2>(tSrK); ++i) {
                    if (i + 1 < size<2>(tSrK)) {
                        cute::copy(smem_tiled_copy_K, tSsK(_, _, i + 1, stage), tSrK_view(_, _, i + 1));
                    }
                    cute::gemm(tiled_mma, tSrQ(_, _, i), tSrK(_, _, i), acc_s);
                }
            } else {
                FLASH_NAMESPACE::gemm</*A_in_regs=*/false>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, stage), tiled_mma,
                    smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K);
            }

            full_vm_bar[stage].wait(phase);

            if constexpr (Kernel_traits::kMaskInSmem) { apply_mask_from_smem(acc_s, stage); }
            else { apply_mask_from_gmem(acc_s); }

            if (j == 0) {
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                    acc_s, acc_o, p.scale_softmax_log2);
            } else {
                softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true>(
                    acc_s, acc_o, p.scale_softmax_log2);
            }

            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            Tensor tOrP = make_tensor(rP.data(),
                FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, stage),
                                     tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            empty_bar[stage].arrive();

            if (tidx == 0) {
                const int j2 = j + kStages;
                if (j2 < n_tiles) {
                    empty_bar[stage].wait(phase);
                    issue_tile(n_begin + j2, stage, /*with_q=*/false);
                }
            }
        }

        // Split 约定（FA3 combine 配套）：O_s 按本地 l 归一化；lse_s = m*scale + log(l)，全屏蔽行 = -inf
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Split=*/true>(
            acc_o, p.scale_softmax, /*rp_dropout=*/1.0f);
    } else {
        // 空 split（尾部分裂）：写 0 / -inf，combine 中 scale=0 跳过
        cute::fill(lse, -INFINITY);
    }

    // ── 部分结果落盘 ─────────────────────────────────────────────────────
    // O_partial 以 bf16 存储（流量减半；相对误差 ~0.4%，低于 bf16 输出量化误差），
    // 复用基线 epilogue 路径：acc_o → bf16 → STSM 写 swizzled sO → TMA store 批量写 gmem。
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    // kQInRegs 时无 sQ 区域，sO 复用 sK；否则复用 sQ（kBlockM*kHeadDim 恰为 sQ 大小）
    Tensor sO = make_tensor(Kernel_traits::kQInRegs ? sK.data() : sQ.data(),
                            typename Kernel_traits::SmemLayoutO{});
#if defined(CUTE_ARCH_STSM_SM90_ENABLED)
    using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
#else
    using SmemCopyAtomO = typename Kernel_traits::SmemCopyAtomO;
#endif
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    __syncthreads();   // 确保所有线程已读完 sQ/sK（sO 与其别名）
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    if (tidx == 0) {
        // O_partial 视作 (SqR, d, H, B*num_splits) 4D 张量，batch 维索引 = split_idx * B + bidb
        Tensor mOp = params.tma_store_Op.get_tma_tensor(
            make_shape(p.seqlen_q_rounded, p.d, p.h, p.b * p.num_splits))(_, _, bidh, split_idx * p.b + bidb);
        Tensor gOp = local_tile(mOp, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, _0{}));
        auto block_tma_Op = params.tma_store_Op.get_slice(_0{});
        Tensor tOsOp = group_modes<0, 3>(block_tma_Op.partition_S(sO));
        Tensor tOgOp = group_modes<0, 3>(block_tma_Op.partition_D(gOp));
        cute::copy(params.tma_store_Op, tOsOp, tOgOp);
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }

    // LSE：row_sum 已 quad_allreduce，quad 内 4 线程值相同，仅 col==0 的线程写（FA2 模式）
    const int64_t tile_lse_offset =
        (((int64_t)split_idx * p.b + bidb) * p.h + bidh) * (int64_t)p.seqlen_q_rounded
        + (int64_t)m_block * kBlockM;
    float* gLSE = reinterpret_cast<float*>(p.lseaccum_ptr) + tile_lse_offset;
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { gLSE[get<0>(taccOcO_row(mi))] = lse(mi); }
    }
#else
    (void)params;
#endif
}

// ── Split-KV combine kernel ──────────────────────────────────────────────────
// grid = (num_m_blocks * (kBlockM/kRows), B*H, kHeadDim/kCols)，block = 128 线程
// 每个 CTA 处理 kRows 行 × kCols 列的输出子块。tile 尺寸由 host 端自适应选择：
// grid 足够大时用 32×32（摊薄 Phase1 开销）；小 grid 时缩到 32×16 / 16×16，
// 把归约并行度放大 2~4 倍吃满带宽（kCols=16 时每行恰好 1 个 32B sector，不浪费带宽）。
//   Phase1: 1 个 warp 计算本 CTA kRows 行的 scale_s = exp(lse_s - lse_max) / Σ_s
//   Phase2: 全 CTA float4 向量化加权归约，转 bf16 写出
template<int kHeadDim, int kNThreads, int kRows, int kCols, typename Element>
__global__ void __launch_bounds__(kNThreads)
flash_fwd_mask_combine_kernel_sm120(
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

    // PDL：与上游 splitkv kernel 重叠启动。等待其全部 CTA 写出 O_partial/LSE 后再读。
    // 未以 PDL 属性启动时此为 no-op，安全。
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif

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
        // 4 路独立 load+FMA 链展开：暴露 MLP（在途字节 ×4），整组跳过非活跃 split
        int s = 0;
        for (; s + 4 <= num_splits; s += 4) {
            if (((active >> s) & 0xfull) == 0ull) { continue; }
            Element pv0[4], pv1[4], pv2[4], pv3[4];
            *reinterpret_cast<uint2*>(pv0) = *reinterpret_cast<const uint2*>(cp_base + (s + 0) * split_stride_o);
            *reinterpret_cast<uint2*>(pv1) = *reinterpret_cast<const uint2*>(cp_base + (s + 1) * split_stride_o);
            *reinterpret_cast<uint2*>(pv2) = *reinterpret_cast<const uint2*>(cp_base + (s + 2) * split_stride_o);
            *reinterpret_cast<uint2*>(pv3) = *reinterpret_cast<const uint2*>(cp_base + (s + 3) * split_stride_o);
            const float sc0 = s_scale[(s + 0) * kRows + r];
            const float sc1 = s_scale[(s + 1) * kRows + r];
            const float sc2 = s_scale[(s + 2) * kRows + r];
            const float sc3 = s_scale[(s + 3) * kRows + r];
            // 注意：非活跃 split 的 sc=0，但 pv 可能是 masked 区的任意值；0*有限值=0 安全。
            // masked split 的 O_partial 由主 kernel 写成有限值（l=0 时全 0），无 inf/NaN。
            acc.x += sc0 * float(pv0[0]) + sc1 * float(pv1[0]) + sc2 * float(pv2[0]) + sc3 * float(pv3[0]);
            acc.y += sc0 * float(pv0[1]) + sc1 * float(pv1[1]) + sc2 * float(pv2[1]) + sc3 * float(pv3[1]);
            acc.z += sc0 * float(pv0[2]) + sc1 * float(pv1[2]) + sc2 * float(pv2[2]) + sc3 * float(pv3[2]);
            acc.w += sc0 * float(pv0[3]) + sc1 * float(pv1[3]) + sc2 * float(pv2[3]) + sc3 * float(pv3[3]);
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

// ── Split-KV launchers ───────────────────────────────────────────────────────
template<typename Kernel_traits>
void run_flash_fwd_mask_sm120_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    using Element = typename Kernel_traits::Element;

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.q_ptr)),
        make_shape(params.seqlen_q, params.d, params.h, params.b),
        make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride));
    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.k_ptr)),
        make_shape(params.seqlen_k, params.d, params.h_k, params.b),
        make_stride(params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride));
    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.v_ptr)),
        make_shape(params.seqlen_k, params.d, params.h_k, params.b),
        make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride));
    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element const*>(params.mask_ptr)),
        make_shape(params.seqlen_q_rounded, params.seqlen_k_rounded, params.b),
        make_stride(params.mask_row_stride, _1{}, params.mask_batch_stride));
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)),
        make_shape(params.seqlen_q, params.d, params.h, params.b),
        make_stride(params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride));
    // O_partial: (num_splits, B, H, SqR, d) bf16 → 视作 (SqR, d, H, B*num_splits) 4D TMA 张量
    // (row, col, h, bs) 偏移 = bs*(H*SqR*d) + h*(SqR*d) + row*d + col
    const int64_t op_row_stride  = params.d;                                               // dim0 (SqR) stride
    const int64_t op_head_stride = static_cast<int64_t>(params.seqlen_q_rounded) * params.d;  // dim2 (H) stride
    const int64_t op_bs_stride   = static_cast<int64_t>(params.h) * op_head_stride;        // dim3 (B*S) stride
    Tensor mOpartial = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.oaccum_ptr)),
        make_shape(params.seqlen_q_rounded, params.d, params.h, params.b * params.num_splits),
        make_stride(op_row_stride, _1{}, op_head_stride, op_bs_stride));

    FA_mask_params_sm120<Kernel_traits> kparams;
    kparams.base = params;
    kparams.tma_load_Q    = make_tma_copy(SM90_TMA_LOAD{},  mQ,    typename Kernel_traits::SmemLayoutQ{});
    kparams.tma_load_K    = make_tma_copy(SM90_TMA_LOAD{},  mK,    typename Kernel_traits::SmemLayoutKVstage{});
    kparams.tma_load_V    = make_tma_copy(SM90_TMA_LOAD{},  mV,    typename Kernel_traits::SmemLayoutKVstage{});
    kparams.tma_load_Mask = make_tma_copy(SM90_TMA_LOAD{},  mMask, typename Kernel_traits::SmemLayoutMaskStage{});
    kparams.tma_store_O   = make_tma_copy(SM90_TMA_STORE{}, mO,    typename Kernel_traits::SmemLayoutO{});
    kparams.tma_store_Op  = make_tma_copy(SM90_TMA_STORE{}, mOpartial, typename Kernel_traits::SmemLayoutO{});

    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    static_assert(smem_size <= 101376, "sm120 per-CTA dynamic smem limit exceeded");

    const int num_m_blocks = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    dim3 grid(num_m_blocks * params.num_splits, params.b, params.h);

    auto kernel = &flash_fwd_mask_kernel_sm120_splitkv<Kernel_traits>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(kparams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// combine tile 自适应：小 grid 时缩小 tile 换并行度（目标 ≥ ~128 CTA 接近吃满 170 SM）
// kCols 最小 16（每行 16×2B=32B 恰好 1 个 sector，再小会浪费 DRAM 带宽）
template<int kHeadDim, int kNThreads, typename Element>
inline void launch_combine_adaptive(
    const FA_mask_params &params, cudaStream_t stream,
    const int rows_per_cta, const int cols_per_cta) {
    dim3 grid(params.seqlen_q_rounded / rows_per_cta, params.b * params.h,
              kHeadDim / cols_per_cta);
    #define LAUNCH_COMBINE(R, C) \
        do { \
            auto kernel = &flash_fwd_mask_combine_kernel_sm120<kHeadDim, kNThreads, R, C, Element>; \
            cudaLaunchConfig_t cfg = {}; \
            cfg.gridDim = grid; \
            cfg.blockDim = dim3(kNThreads, 1, 1); \
            cfg.dynamicSmemBytes = 0; \
            cfg.stream = stream; \
            cudaLaunchAttribute attrs[1]; \
            attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \
            attrs[0].val.programmaticStreamSerializationAllowed = 1; \
            cfg.attrs = attrs; \
            cfg.numAttrs = 1; \
            cudaLaunchKernelEx(&cfg, kernel, \
                reinterpret_cast<const Element*>(params.oaccum_ptr), \
                reinterpret_cast<const float*>(params.lseaccum_ptr), \
                reinterpret_cast<Element*>(params.o_ptr), \
                params.num_splits, params.seqlen_q, params.seqlen_q_rounded, \
                params.h, params.b, \
                params.o_batch_stride, params.o_head_stride, params.o_row_stride); \
        } while (0)
    if (rows_per_cta == 32 && cols_per_cta == 32)      { LAUNCH_COMBINE(32, 32); }
    else if (rows_per_cta == 32 && cols_per_cta == 16) { LAUNCH_COMBINE(32, 16); }
    else                                               { LAUNCH_COMBINE(16, 16); }
    #undef LAUNCH_COMBINE
}

template<int kBlockM, int kHeadDim, typename Element>
void run_flash_fwd_mask_combine_sm120(const FA_mask_params &params, cudaStream_t stream) {
    constexpr int kNThreads = 128;
    const int64_t ctas_32 = (int64_t)(params.seqlen_q_rounded / 32) * params.b * params.h * (kHeadDim / 32);
    int rows_per_cta = 32, cols_per_cta = 32;
    if (ctas_32 < 128) {
        // 先切列（grid z 翻倍），不够再切行（grid x 再翻倍）
        cols_per_cta = 16;
        const int64_t ctas_32x16 = ctas_32 * 2;
        if (ctas_32x16 < 128 && params.seqlen_q_rounded % 16 == 0) {
            rows_per_cta = 16;
        }
    }
    launch_combine_adaptive<kHeadDim, kNThreads, Element>(
        params, stream, rows_per_cta, cols_per_cta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace FA_MASK_NAMESPACE
