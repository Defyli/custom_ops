/*
 * Flash Attention Forward with Additive Mask — Kernel Traits（sm89 基线）
 *
 * 在 FA2 Flash_fwd_kernel_traits 的基础上新增：
 *   - SmemLayoutMask    : (kBlockM, kBlockN, kStages)，swizzled
 *   - GmemTiledCopyMask : cp.async 128-bit
 *   - SmemCopyAtomMask  : smem → register 的 copy atom（与 rab 相同）
 *
 * kSmemSize（覆盖基类）= 基类 kSmemSize + mask tile，供 run_flash_fwd_with_mask 使用。
 *
 * Mask tensor 的 global mem 布局（无 pad 契约，详见 fa_fwd_params.h / fa_fwd_kernel.h）：
 *   (B, mask_seqlen_q, mask_seqlen_k)，row-major，fp16/bf16（与输入同 dtype）
 *   - q 维：mask_seqlen_q ∈ [seqlen_q, 任意]，无需对齐——kernel 行谓词跳过越界行，
 *     这些行的输出被 epilogue 丢弃
 *   - k 维：mask_seqlen_k ∈ [seqlen_k, 任意] 且 %8==0（128-bit cp.async 行对齐），
 *     无需 pad 到 kBlockN——边界 tile 的越界列由列谓词跳过拷贝、smem 预清 -inf
 *     （语义：col ≥ Sk 恒为屏蔽，mask 越界列内容被忽略）
 *
 * Mask 语义：加法 mask（与 PyTorch SDPA 对齐，softmax(S·scale + mask)）
 *   0 → 可见，-inf → 屏蔽，有限值 → 任意偏置（ALiBi 风格）
 *
 * 参数
 * ----
 * kHeadDim_  : attention head dim（64 或 128）
 * kBlockM_   : Q tile 行数
 * kBlockN_   : K/V tile 行数
 * kNWarps_   : warp 数（128×128 tile 需 8，其余 4）
 * MaskQFull_ : host 保证 mask_seqlen_q % kBlockM == 0 时置 true（编译期裁行谓词）
 * elem_type  : fp16 → cutlass::half_t / bf16 → cutlass::bfloat16_t
 * kStages_   : K/V/Mask 多级缓冲级数（splitkv 双缓冲路径用 2，默认 1 = 单缓冲）
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

// 复用 FA2 Flash_fwd_kernel_traits 基类
#include "../common/kernel_traits.h"

using namespace cute;

// ── 带 mask smem 的 fwd kernel traits ───────────────────────────────────────────
// kStages_：K/V/Mask 多级缓冲级数（仅 splitkv 双缓冲路径使用，默认 1 = 单缓冲）。
//   kStages=1 时各布局/尺寸与历史版本完全一致（base/splitkv 单缓冲路径零改动）；
//   kStages>1 时提供 *Staged 布局别名，K/V/Mask 各 kStages 级（sm120 多级流水的
//   cp.async 对应物），smem 预算 = Q + kStages×(K+V+Mask)。
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, bool MaskQFull_=false,
         typename elem_type=cutlass::bfloat16_t, int kStages_=1,
         typename Base=Flash_fwd_kernel_traits<
             kHeadDim_, kBlockM_, kBlockN_, kNWarps_,
             Is_Q_in_regs_, Share_Q_K_smem_, elem_type>>
struct FA_mask_kernel_traits : public Base {

    // ── 继承基类的所有类型别名 ──
    using Element        = typename Base::Element;
    using ElementAccum   = typename Base::ElementAccum;
    using index_t        = typename Base::index_t;
    using SmemCopyAtom   = typename Base::SmemCopyAtom;

    static constexpr int kBlockM    = Base::kBlockM;
    static constexpr int kBlockN    = Base::kBlockN;
    static constexpr int kHeadDim   = Base::kHeadDim;
    static constexpr int kNWarps    = Base::kNWarps;
    static constexpr int kNThreads  = Base::kNThreads;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    // MaskQFull_=true：host 保证 mask_seqlen_q % kBlockM == 0，编译期裁掉 mask 行谓词
    // 路径（copy_if → 无谓词 copy），消除谓词张量的寄存器开销（生产主场景）
    static constexpr bool kMaskQFull = MaskQFull_;
    // K/V/Mask 多级缓冲级数（1 = 单缓冲，2 = splitkv 双缓冲路径）
    static constexpr int kStages = kStages_;

    // ── Mask SmemLayout ────────────────────────────────────────────────────
    // 参照 hstu_mask.h 中 SmemLayoutMask 的做法：
    //   kBlockKSmemMask = kBlockN % 64 == 0 ? 64 : 32
    //   kSwizzleMask    = kBlockKSmemMask == 32 ? 2 : 3
    //   SmemLayoutAtomMask: Swizzle<k,3,3> + Layout<_8, kBlockKSmemMask>
    //   SmemLayoutMask: tile_to_shape → (kBlockM, kBlockN, kStages=1)
    static constexpr int kBlockKSmemMask = kBlockN % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleMask    = kBlockKSmemMask == 32 ? 2 : 3;

    using SmemLayoutAtomMask = decltype(composition(
        Swizzle<kSwizzleMask, 3, 3>{},
        Layout<Shape<_8, Int<kBlockKSmemMask>>,
               Stride<Int<kBlockKSmemMask>, _1>>{}
    ));
    // kStages=1：mask 只需单缓冲，load → compute 串行，无需 double buffering
    using SmemLayoutMask = decltype(tile_to_shape(
        SmemLayoutAtomMask{},
        Shape<Int<kBlockM>, Int<kBlockN>, _1>{}
    ));

    // ── 多级缓冲布局（kStages > 1 时使用）───────────────────────────────
    // K/V: (kBlockN, kHeadDim, kStages)；V 转置视图 sVt(d, n, s) = sV(n, d, s)
    // （布局组合方式与 sm120 的 SmemLayoutVtransposed 一致）
    using SmemLayoutKVStaged = decltype(tile_to_shape(
        typename Base::SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages_>>{}));
    using SmemLayoutVtransposedStaged = decltype(composition(
        SmemLayoutKVStaged{},
        make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages_>>{},
                    make_stride(Int<kBlockN>{}, _1{}, Int<kBlockN * kHeadDim_>{}))));
    using SmemLayoutVtransposedStagedNoSwizzle =
        decltype(get_nonswizzle_portion(SmemLayoutVtransposedStaged{}));
    // Mask: (kBlockM, kBlockN, kStages)
    using SmemLayoutMaskStaged = decltype(tile_to_shape(
        SmemLayoutAtomMask{},
        Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages_>>{}));
    // 双缓冲 smem 总量（Q 单缓冲 + K/V/Mask × kStages）
    static constexpr int kSmemSizeStaged =
        Base::kSmemQSize
        + 2 * kStages_ * kBlockN_ * kHeadDim_ * int(sizeof(Element))
        + kStages_ * kBlockM_ * kBlockN_ * int(sizeof(Element));

    // ── GmemTiledCopyMask：cp.async 128-bit ─────────────────────────────
    // 参照 hstu_mask.h 的 GmemTiledCopyMask（等同于 GmemTiledCopyRab）
    static constexpr int kGmemElemsPerLoad      = sizeof(cute::uint128_t) / sizeof(Element); // 8
    static constexpr int kGmemThreadsPerRowMask = kBlockKSmemMask / kGmemElemsPerLoad;       // 4 or 8

    static_assert(kNThreads % kGmemThreadsPerRowMask == 0,
                  "kNThreads must be divisible by kGmemThreadsPerRowMask");

    using GmemLayoutAtomMask = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRowMask>, Int<kGmemThreadsPerRowMask>>,
        Stride<Int<kGmemThreadsPerRowMask>, _1>
    >;

    // Val layout：每次 128-bit = kGmemElemsPerLoad elem，跨 kBlockKSmemMask 填满 kBlockN
    static constexpr int kMaskRowSize = kBlockN / kBlockKSmemMask;
    using GmemTiledCopyMask = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        GmemLayoutAtomMask{},
        Layout<Shape<Int<kMaskRowSize>, Int<kGmemElemsPerLoad>>,
               Stride<Int<kGmemElemsPerLoad>, _1>>{}
    ));

    // ── Smem total size（覆盖基类 kSmemSize，加上 mask tile）──────────────
    static constexpr int kSmemMaskSize = kBlockM * kBlockN * sizeof(Element);  // 1 stage
    // 覆盖基类的 kSmemSize，使 run_flash_fwd_with_mask 拿到含 mask 的正确大小
    static constexpr int kSmemSize     = Base::kSmemSize + kSmemMaskSize;
};
