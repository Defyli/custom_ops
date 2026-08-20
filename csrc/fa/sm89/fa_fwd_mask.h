/*
 * Flash Attention Forward with Multiplicative Mask — Kernel Traits
 *
 * 在 FA2 Flash_fwd_kernel_traits 的基础上新增：
 *   - SmemLayoutMask    : (kBlockM, kBlockN, kStages=1)，swizzled
 *   - GmemTiledCopyMask : cp.async 128-bit
 *   - SmemCopyAtomMask  : smem → register 的 copy atom（与 rab 相同）
 *
 * kSmemSize（覆盖基类）= 基类 kSmemSize + mask tile，供 run_flash_fwd_with_mask 使用。
 *
 * Mask tensor 的 global mem 布局：(B, seqlen_q, seqlen_k_rounded)，row-major，bf16
 * seqlen_k_rounded = ceil(seqlen_k / kBlockN) * kBlockN（由调用方保证对齐）
 *
 * Mask 语义：乘法 mask
 *   0.0  → 屏蔽（乘 0，softmax 后 weight → 0，等效 -inf）
 *   1.0  → 可见
 *
 * 参数
 * ----
 * kHeadDim_  : attention head dim（64 或 128）
 * kBlockM_   : Q tile 行数
 * kBlockN_   : K/V tile 行数
 * kNWarps_   : warp 数（固定 4）
 * elem_type  : bf16 -> cutlass::bfloat16_t
 */

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

// 复用 FA2 Flash_fwd_kernel_traits 基类
#include "../common/kernel_traits.h"

using namespace cute;

// ── 带 mask smem 的 fwd kernel traits ───────────────────────────────────────
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, bool MaskQFull_=false,
         typename elem_type=cutlass::bfloat16_t,
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
