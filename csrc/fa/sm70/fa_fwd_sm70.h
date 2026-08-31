/*
 * fa_fwd_sm70.h — Flash Attention Forward with Additive Mask, Volta (sm_70) 专用
 *
 * V100 硬件约束：
 *   - Tensor Core: wmma.mma.sync.aligned.m16n16k16.f32.f32 → 16× HMMA.884
 *   - 无 ldmatrix / cp.async / bf16 ALU
 *   - smem 96KB/CTA (>48KB 需 opt-in)
 *
 * ══ v9：WMMA m16n16k16 替换 cute TiledMma (m8n8k4) ═════════════════════════
 * 动机（SASS 分析，v8.1 vs fa-v100）：
 *   - LDS 指令密度 259/iter vs 76.5 等效 —— 3.4× 差距，MMA 路径核心瓶颈
 *   - m8n8k4: 每 K=4 步 load fragment（窄 32/64bit）
 *   - wmma.m16n16k16: 一条指令复用 fragment 16-K，LDS 减 ~3× 且全部向量化
 *     （A row / B col = 2×LDS.128，B row = 4×LDS.64）
 *
 * warp grid（16 warps = 4×4，每 warp 独占 16×16 tile）：
 *   QK^T: warp (m,n) → S-tile rows [16m,16m+16) × cols [16n,16n+16)
 *   PV:   warp (m,n) → O-tile rows [16m,+) × d-cols [16n,+)（d=128 再加 +64 tile）
 *   → 每 S 行只属于一个 warp → softmax 归约 warp-local（2×__shfl_xor）！
 *
 * softmax（v9 结构修正：S 每行横跨 4 个 warp_n tile，需跨 warp 交换）：
 *   - 行 max: 线程内(4列) → 2×__shfl_xor(2,8)（warp 内 4 lanes，同行 c0∈{0,2,8,10}）
 *     → 跨 warp atomicMax(sRowMax)（仅 primary lane: lane&0b1010==0，每 warp 每 16 行
 *     各 lane 独占 {r0, r0+2}）→ sync → 读回 m_global
 *   - 行 l:   partial-l 寄存器化（v8.1，每 warp 累积自己 16 列的贡献，
 *     rescale factor 行全局一致故 partial 可独立累积），epilogue：
 *     2×__shfl_xor → primary atomicAdd(sRowSum) → sync → 读回 l
 *   - vs v8.1：warp 内 shuffle 层数相同；主循环零 atomicAdd（l 在 epilogue）；
 *     acc 布局与 m8n8k4 同构，softmax 逻辑结构沿用
 *
 * v9 swizzle（关键差异）：行 bits {0,1,3} 参与 chunk XOR
 *   旧 Swizzle<3,3,3>（(8,64) atom，行 bits {0,1,2}）：r 与 r+8 同 pattern →
 *     WMMA A/B fragment load 的 lane quarter 内 2-way bank conflict
 *   v9（(16,64) atom，行 bits {0,1,3}，fa-v100 风格）：quarter 内 16 行互异 → 无 conflict
 *   实测（新旧 swizzle 整管线 A/B 对拍）：3215 vs 4170 cycles/iter（-23%）
 *
 * WMMA fragment 布局（identity-probe 实验实测验证）：
 *   A row-major: lane 持 tile 内一整行 16 halves（2×LDS.128）
 *     r_base = (lid&3) + ((lid>>4)&1)*4 + ((lid>>2)&1)*8
 *   B col-major (K^T): lane 持一整列 = sK 一行 16 halves（2×LDS.128）
 *     g=lid>>2; idx=((g>>2)&1)|(g&2); n = tile_n + (lid&3) + (idx<<2)
 *   B row-major (V): lane 持 4 个 4-half chunk（4×LDS.64）
 *     r_base = lid&3; c_base = ((lid>>3)&1)*8 + ((lid>>4)&1)*4
 *   acc: r0 = ((lid>>2)&1)*8 + ((lid>>4)&1)*4 + (lid&1)
 *        c0 = ((lid>>3)&1)*8 + ((lid>>1)&1)*2
 *        x[i] → (r0 + 2*((i>>1)&1), c0 + (i&1) + 4*((i>>2)&1))
 *        → {0,1,4,5}→行 r0，{2,3,6,7}→行 r0+2（与 m8n8k4 acc 同构，
 *           softmax 逻辑结构沿用 v6/v8.1）
 *
 * 沿用的优化：
 *   v6:   P/Mask smem 时间复用；MMA 线程直接管理 softmax
 *   v7:   K smem double buffer（load/compute 重叠）
 *   v8.1: V gmem load 提前到 A 段（MLP）
 *
 * smem 预算（v9，Q 回 smem——WMMA A-fragment 需 lane 持整行；无 sSoftmax）：
 *   d=64:  sQ 8KB + sK×2 16KB + sV 8KB + sP/Mask 8KB = 40KB（< 48KB 默认上限）
 *   d=128: 16 + 32 + 16 + 8 = 72KB（opt-in 96KB）
 *
 * v9 sync 结构（4 次/iter，与 v8.1 持平）：
 *   sync1:  Mask/V/K-prefetch store 完成 → QK^T 可读 sK[n]
 *   syncM:  sRowMax atomicMax 完成 → m_global 可读
 *   sync5:  sP(post-exp) store 完成 → PV 可读（P 行由 4 个 warp_n 分写，跨 warp）
 *   sync6:  PV 读 sP/sV 完成 → 下一轮 A 段可覆写（sMask=sP 复用 + sK buffer 轮换）
 *
 * v9.1: Q A-fragments 常驻寄存器（仅 d=64，环路不变量外提）
 *   - 每 iter 省 8 条 LDS.128；REG 75 → 98（512×98 < 64K，无 spill 无降占用）
 *   - d=128 寄存器预算不足（需 +64 regs）故跳过，Q load 留在环内
 *
 * 实测（V100-PCIE-32GB, torch 2.0.1, fp16, vs fa-v100 无 mask / SDPA+add-mask）：
 *   d=128 峰值 24.2 TFLOPS（v8.1 17.9, +35%；fa-v100 19.5, 1.24x；
 *     32,16,16,1024,1024,128 达 1.31x；长序列 8192² 1.28x）
 *   d=64  峰值 18.0 TFLOPS（v8.1 13.4, +34%；与 fa-v100 持平 0.95~1.07x，
 *     1024² grid 反超；其 BLOCK_N=128 结构在超长序列仍略优，属已知 gap）
 *   vs SDPA+additive-mask（同功能公平对照）：全大 grid shape 1.05~1.77x
 *   每 iter LDS 静态数：d=64 51→43，d=128 86（v8.1 为 259）；零 spill
 *
 * Mask 语义与 sm89/sm120 一致：softmax(S·scale + mask)，越界 -inf。
 * mask 契约（无 pad，同 sm89/sm120）：(B, mask_seqlen_q, mask_seqlen_k)，
 *   Sk%8==0 即可（gmem copy 128-bit 向量对齐）、Sq 任意；语义 col ≥ Sk 恒为屏蔽。
 *   与 sm89 的 prologue 一次性预清不同：sMask 与 sP 时间复用，每轮 Phase C 的
 *   P store 覆写全 tile → 边界 tile 越界列的 -inf 必须逐轮回写（Phase A 内，
 *   受上轮 sync6 保护，无 race）。
 */

#pragma once

#include <cuda_fp16.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

#include "../common/utils.h"
#include "../common/softmax.h"
#include "../common/fa_fwd_params.h"

namespace FA_MASK_NAMESPACE {

using namespace cute;

// ══════════════════════════════════════════════════════════════════════════
// float atomicMax on smem（正确处理负数，v8.1 沿用）
// key 变换: 正 float → bits^0x80000000，负 float → ~bits → unsigned 序与 float 序一致
// ══════════════════════════════════════════════════════════════════════════
__device__ __forceinline__ unsigned int float_to_key(float val) {
    unsigned int val_ui = __float_as_uint(val);
    unsigned int mask = (val_ui & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return val_ui ^ mask;
}
__device__ __forceinline__ float key_to_float(unsigned int key) {
    unsigned int mask = (key & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    return __uint_as_float(key ^ mask);
}

// ══════════════════════════════════════════════════════════════════════════
// WMMA m16n16k16 辅助（自研 PTX 封装，布局经 identity-probe 实验实测验证）
// ══════════════════════════════════════════════════════════════════════════

struct WmmaAFrag { uint32_t x[8]; };   // 16 halves (8×uint32)
struct WmmaBFrag { uint32_t x[8]; };   // 16 halves (8×uint32)
struct WmmaAccFrag { float x[8]; };    // 2 行 × 4 列

// D = A @ B + C（A row-major × B col-major，用于 QK^T）
__device__ __forceinline__ void wmma_mma_row_col(
    WmmaAccFrag& d, const WmmaAFrag& a, const WmmaBFrag& b, const WmmaAccFrag& c) {
    asm volatile(
        "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%24,%25,%26,%27,%28,%29,%30,%31};"
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(a.x[4]), "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]),
          "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
          "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]), "r"(b.x[7]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));
}

// D = A @ B + C（A row-major × B row-major，用于 PV: P@V）
__device__ __forceinline__ void wmma_mma_row_row(
    WmmaAccFrag& d, const WmmaAFrag& a, const WmmaBFrag& b, const WmmaAccFrag& c) {
    asm volatile(
        "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%24,%25,%26,%27,%28,%29,%30,%31};"
        : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3]),
          "=f"(d.x[4]), "=f"(d.x[5]), "=f"(d.x[6]), "=f"(d.x[7])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
          "r"(a.x[4]), "r"(a.x[5]), "r"(a.x[6]), "r"(a.x[7]),
          "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
          "r"(b.x[4]), "r"(b.x[5]), "r"(b.x[6]), "r"(b.x[7]),
          "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),
          "f"(c.x[4]), "f"(c.x[5]), "f"(c.x[6]), "f"(c.x[7]));
}

// ── v9 swizzled smem 地址（返回 half 偏移） ─────────────────────────────────
// 布局：16 行 × 64 列 atom；行内 8-half(16B) chunk，chunk ^= 行 bits {0,1,3}
// atom 内逻辑行 rr 的物理行位置 = bits{2,3} 交换（层级 stride 的自然结果）
// tile grid (rows/16, ldm/64) 列主序；tile 大小 1024 halves
__device__ __forceinline__ int v9_smem_off(int r, int c, int ldm) {
    const int tile = (r >> 4) + ((c >> 6) * (64 >> 4));  // (r/16) + (c/64)*4
    const int rr = r & 15;
    const int cc = c & 63;
    const int phys = (rr & 3) | ((rr & 4) << 1) | ((rr & 8) >> 1);
    const int chunk = (cc >> 3) ^ ((rr & 3) | ((rr >> 3) << 2));
    return tile * 1024 + phys * 64 + chunk * 8 + (cc & 7);
}

// ── fragment load（从 v9 swizzled smem） ────────────────────────────────────

// A row-major：lane 持 tile 内一整行（rows [tile_m, +16)，cols [k0, k0+16)）
// x[i] = halves (r, k0+2i), (r, k0+2i+1)；2×LDS.128
__device__ __forceinline__ WmmaAFrag v9_load_a_row(
    const cutlass::half_t* s, int tile_m, int k0, int ldm) {
    const int lid = threadIdx.x & 31;
    const int r = tile_m + (lid & 3) + ((lid >> 4) & 1) * 4 + ((lid >> 2) & 1) * 8;
    const __half* sp = reinterpret_cast<const __half*>(s);
    WmmaAFrag f;
    const uint4 v0 = *reinterpret_cast<const uint4*>(sp + v9_smem_off(r, k0, ldm));
    const uint4 v1 = *reinterpret_cast<const uint4*>(sp + v9_smem_off(r, k0 + 8, ldm));
    f.x[0] = v0.x; f.x[1] = v0.y; f.x[2] = v0.z; f.x[3] = v0.w;
    f.x[4] = v1.x; f.x[5] = v1.y; f.x[6] = v1.z; f.x[7] = v1.w;
    return f;
}

// B col-major（K^T）：lane 持一整列（物理 = sK 行 n 的 cols [k0, k0+16)）
// B(k, n) = K(n, k)；x[i] = halves (k0+2i, n), (k0+2i+1, n)；2×LDS.128
__device__ __forceinline__ WmmaBFrag v9_load_b_col(
    const cutlass::half_t* s, int tile_n, int k0, int ldm) {
    const int lid = threadIdx.x & 31;
    const int g = lid >> 2;
    const int idx = ((g >> 2) & 1) | (g & 2);
    const int n = tile_n + (lid & 3) + (idx << 2);
    const __half* sp = reinterpret_cast<const __half*>(s);
    WmmaBFrag f;
    const uint4 v0 = *reinterpret_cast<const uint4*>(sp + v9_smem_off(n, k0, ldm));
    const uint4 v1 = *reinterpret_cast<const uint4*>(sp + v9_smem_off(n, k0 + 8, ldm));
    f.x[0] = v0.x; f.x[1] = v0.y; f.x[2] = v0.z; f.x[3] = v0.w;
    f.x[4] = v1.x; f.x[5] = v1.y; f.x[6] = v1.z; f.x[7] = v1.w;
    return f;
}

// B row-major（V）：B(k, j) = V(k, j)（物理 = sV 行 k 列 j）
// lane: r_base = lid&3, c_base = ((lid>>3)&1)*8 + ((lid>>4)&1)*4
// x[2t] = halves (k0+r_base+4t, j0+c_base..+1)；x[2t+1] = (.., +2..+3)；4×LDS.64
__device__ __forceinline__ WmmaBFrag v9_load_b_row(
    const cutlass::half_t* s, int j0, int k0, int ldm) {
    const int lid = threadIdx.x & 31;
    const int r_base = lid & 3;
    const int c_base = ((lid >> 3) & 1) * 8 + ((lid >> 4) & 1) * 4;
    const __half* sp = reinterpret_cast<const __half*>(s);
    WmmaBFrag f;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int k = k0 + r_base + 4 * t;
        const uint2 v = *reinterpret_cast<const uint2*>(
            sp + v9_smem_off(k, j0 + c_base, ldm));
        f.x[2 * t] = v.x;
        f.x[2 * t + 1] = v.y;
    }
    return f;
}

// ── acc fragment lane 基准（tile 内偏移） ───────────────────────────────────
__device__ __forceinline__ void v9_acc_base(int& r0, int& c0) {
    const int lid = threadIdx.x & 31;
    r0 = ((lid >> 2) & 1) * 8 + ((lid >> 4) & 1) * 4 + (lid & 1);
    c0 = ((lid >> 3) & 1) * 8 + ((lid >> 1) & 1) * 2;
}
// x[i] 的 tile 内行列偏移：行 ∈ {0,2}，列 ∈ {0,1,4,5}
__device__ __forceinline__ int v9_acc_row_off(int i) { return 2 * ((i >> 1) & 1); }
__device__ __forceinline__ int v9_acc_col_off(int i) { return (i & 1) + 4 * ((i >> 2) & 1); }

// ══════════════════════════════════════════════════════════════════════════
// sm70 kernel traits
// ══════════════════════════════════════════════════════════════════════════
template<int kHeadDim_, int kBlockM_ = 64, int kBlockN_ = 64>
struct FA_sm70_kernel_traits {
    using Element      = cutlass::half_t;
    using ElementAccum = float;
    using index_t      = int64_t;

    static constexpr int kBlockM   = kBlockM_;
    static constexpr int kBlockN   = kBlockN_;
    static constexpr int kHeadDim  = kHeadDim_;
    static_assert(kHeadDim == 64 || kHeadDim == 128, "sm70 path supports d=64/128");
    static_assert(kBlockM == 64 && kBlockN == 64, "sm70 path tile is fixed 64x64");

    static constexpr int kNWarps   = 16;
    static constexpr int kNThreads = kNWarps * 32;

    // v9 WMMA warp grid: 4×4（每 warp 独占一个 16×16 tile）
    static constexpr int kWarpGridM = kBlockM / 16;  // 4
    static constexpr int kWarpGridN = kBlockN / 16;  // 4
    // PV 的 O-tile 数/每 warp（d=128: warp_n 与 warp_n+4 两个 tile，共享 A fragment）
    static constexpr int kOTilesPerWarp = kHeadDim / kBlockN;  // 1 or 2

    // v9 swizzled smem layouts: (16,64) atom，行 bits {0,1,3} 参与 chunk XOR
    //   （行 bits {2,3} 交换的层级 stride → swizzle 掩码取到行 bits {0,1,3}）
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Shape<_2, _2, _2, _2>, _64>,
               Stride<Stride<_64, _128, _512, _256>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemLayoutMask = SmemLayoutP;

    // gmem copy（沿用 v5+ 机制，layout 无关）
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0);
    static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0);
    using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>,
                                        Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>{}));

    static constexpr int kMaskElemsPerLoad = kGmemElemsPerLoad;
    static constexpr int kMaskThreadsPerRow = kBlockN / kMaskElemsPerLoad;
    static_assert(kNThreads % kMaskThreadsPerRow == 0);
    using GmemLayoutAtomMask = Layout<Shape<Int<kNThreads / kMaskThreadsPerRow>,
                                          Int<kMaskThreadsPerRow>>,
                                    Stride<Int<kMaskThreadsPerRow>, _1>>;
    using GmemTiledCopyMask = decltype(make_tiled_copy(
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
        GmemLayoutAtomMask{},
        Layout<Shape<_1, _8>>{}));

    // smem 尺寸 — v9: Q 回 smem（WMMA A-fragment 按 lane 整行 load），
    // K double buffer，P/Mask 共享，sSoftmax（跨 warp 行 max/l 交换）
    static constexpr int kSmemQSize    = kBlockM * kHeadDim * int(sizeof(Element));
    static constexpr int kSmemKVSize   = kBlockN * kHeadDim * int(sizeof(Element));
    static constexpr int kSmemKDBSize  = kSmemKVSize;  // double buffer for K
    static constexpr int kSmemPMaskSize = kBlockM * kBlockN * int(sizeof(Element));
    static constexpr int kSmemSoftmaxSize = kBlockM * 2 * int(sizeof(float));
    //   d=64:  8 + 2*8 + 8 + 8 + 0.5 = 40.5KB（< 48KB 默认上限）
    //   d=128: 16 + 2*16 + 16 + 8 + 0.5 = 72.5KB（opt-in 96KB）
    static constexpr int kSmemSize =
        kSmemQSize + 2 * kSmemKVSize + kSmemKVSize + kSmemPMaskSize + kSmemSoftmaxSize;

    // WMMA acc 布局常量（每 lane 2 行 × 4 列；与 m8n8k4 acc 同构）
    static constexpr int kAccRowsPerThread = 2;
    static constexpr int kAccColsPerThread = 4;
    // acc_o 每 lane float 数 = 8 * kOTilesPerWarp
};

// ══════════════════════════════════════════════════════════════════════════
// 核心计算（单 m_block，完整遍历 K 维）
// ══════════════════════════════════════════════════════════════════════════
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__forceinline__ __device__ void compute_attn_1rowblock_sm70(
    const FA_mask_params &params,
    const int bidb,
    const int bidh,
    const int m_block
) {
    using Element     = typename Kernel_traits::Element;
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int ldm      = kHeadDim;              // Q/K/V smem 行 stride（halves）
    constexpr int kOTiles  = Kernel_traits::kOTilesPerWarp;
    constexpr int kTilesS  = kHeadDim / 16;         // QK^T K-tiles（4 or 8）
    constexpr int kTilesPV = kBlockN / 16;          // PV K-tiles（4）

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    const int actual_seqlen_q = params.seqlen_q;
    const int actual_seqlen_k = params.seqlen_k;
    if (m_block * kBlockM >= actual_seqlen_q) return;
    const int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);

    // ── smem 排布：v9 [sQ][sK0][sK1][sV][sP=sMask] ────────────────────────
    // Q 回 smem（WMMA A-fragment 按 lane 整行 load，2×LDS.128/16-K）
    // sK double buffer；sMask 与 sP 时间复用；无 sSoftmax
    Element* smem_base = reinterpret_cast<Element*>(smem_);
    Element* sQ_ptr  = smem_base;
    Element* sK0_ptr = sQ_ptr  + kBlockM * kHeadDim;
    Element* sK1_ptr = sK0_ptr + kBlockN * kHeadDim;
    Element* sV_ptr  = sK1_ptr + kBlockN * kHeadDim;
    Element* sP_ptr  = sV_ptr  + kBlockN * kHeadDim;   // = sMask（时间复用）
    // sSoftmax: [sRowMax(kBlockM, key-encoded uint), sRowSum(kBlockM, float)]
    float* sSoftmaxPtr = reinterpret_cast<float*>(
        reinterpret_cast<char*>(sP_ptr + kBlockM * kBlockN));
    float* sRowMax = sSoftmaxPtr;               // 存 key（unsigned）
    float* sRowSum = sSoftmaxPtr + kBlockM;
    unsigned int* sRowMaxKey = reinterpret_cast<unsigned int*>(sRowMax);

    Tensor sQ  = make_tensor(make_smem_ptr(sQ_ptr),  typename Kernel_traits::SmemLayoutQ{});
    Tensor sK0 = make_tensor(make_smem_ptr(sK0_ptr), typename Kernel_traits::SmemLayoutKV{});
    Tensor sK1 = make_tensor(make_smem_ptr(sK1_ptr), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV  = make_tensor(make_smem_ptr(sV_ptr),  typename Kernel_traits::SmemLayoutKV{});
    Tensor sP  = make_tensor(make_smem_ptr(sP_ptr),  typename Kernel_traits::SmemLayoutP{});
    Tensor sMask = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutMask{});

    // ── gmem tensors ──────────────────────────────────────────────────────
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + bidb * params.q_batch_stride),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + bidb * params.k_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + bidb * params.v_batch_stride),
        make_shape(actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    // mask 列数 = mask 实际列数（≥ seqlen_k 且 %8==0，无需 pad 到 kBlockN）；
    // 边界 tile 的越界列由主循环 Phase A 的列谓词跳过拷贝 + -inf 回写
    Tensor mMask = make_tensor(
        make_gmem_ptr(reinterpret_cast<const Element*>(params.mask_ptr)
                      + bidb * params.mask_batch_stride),
        make_shape(params.mask_seqlen_q, params.mask_seqlen_k),
        make_stride(params.mask_row_stride, _1{}));
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, _));

    // ── gmem→smem copy handles（沿用 cute 机制，layout 换 v9） ─────────────
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);      // v9: Q → sQ 专用 smem
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tKsK0 = gmem_thr_copy_QKV.partition_D(sK0);
    Tensor tKsK1 = gmem_thr_copy_QKV.partition_D(sK1);

    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask   = gmem_tiled_copy_Mask.get_thread_slice(tidx);
    Tensor tQgMask = gmem_thr_copy_Mask.partition_S(gMask);
    Tensor tQsMask_g2s = gmem_thr_copy_Mask.partition_D(sMask);

    // ── 边界谓词 ──────────────────────────────────────────────────────────
    Tensor cQ  = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
    Tensor cKV = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
    Tensor tQcQ   = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tQpQ   = make_tensor<bool>(make_shape(size<2>(tQgQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK0)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ);   ++k) { tQpQ(k)   = get<1>(tQcQ(0, 0, k))   < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // ── mask 行谓词（q 维不 pad：行坐标 ≥ mask 剩余行数的 copy 向量被跳过）───
    // OOB 行一次性清 0：仅保护首轮（其后每轮 Phase C 的 P store 覆写全 tile）；
    // 稳态安全靠「OOB 行输出被 epilogue 丢弃 + 残留 P 值 ∈ [0,1] 有限无害」
    Tensor cMask   = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tMcMask = gmem_thr_copy_Mask.partition_S(cMask);
    Tensor tMpMask = make_tensor<bool>(make_shape(size<1>(tMcMask), size<2>(tMcMask)));
    const int mask_rows_left = params.mask_seqlen_q - m_block * kBlockM;
    #pragma unroll
    for (int m = 0; m < size<0>(tMpMask); ++m) {
        #pragma unroll
        for (int n = 0; n < size<1>(tMpMask); ++n) {
            tMpMask(m, n) = get<0>(tMcMask(_0{}, m, n)) < mask_rows_left;
        }
    }
    if (mask_rows_left < kBlockM) {
        #pragma unroll
        for (int m = 0; m < size<1>(tQsMask_g2s); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(tQsMask_g2s); ++n) {
                if (!tMpMask(m, n)) { clear(tQsMask_g2s(_, m, n)); }
            }
        }
    }

    // ── v9 WMMA warp 坐标 + acc fragment 基准 ─────────────────────────────
    const int warp   = tidx >> 5;
    const int warp_m = warp >> 2;   // 0..3
    const int warp_n = warp & 3;    // 0..3
    int r0, c0;
    v9_acc_base(r0, c0);
    // 本线程持有的 2 行（S tile 与 O tile 行一致）
    const int my_row0 = 16 * warp_m + r0;
    const int my_row1 = my_row0 + 2;
    // 本线程的 4 列基准（S tile 内；O tile 内为 tile 各自的 col_base + 同偏移）
    const int s_col_base = 16 * warp_n + c0;

    // ── softmax 状态（row_m/row_l 每 lane partial；跨 warp 交换走 sSoftmax） ──
    float row_m[2] = {-INFINITY, -INFINITY};
    float row_l[2] = {0.f, 0.f};
    // primary lane: 每 warp 每 16 行中独占 {r0, r0+2} 两行的写代表（c0==0）
    const int lane = tidx & 31;
    const bool is_primary = ((lane & 2) == 0) && ((lane & 8) == 0);

    // v9 prologue: sSoftmax 初始化（仅一次，主循环内不再 init）
    //   sRowMax: key(-inf) —— 后续每轮直接 atomicMax 叠加（残留值 = 上轮 m_global）
    //   sRowSum: 0 —— 仅 epilogue 做一次 partial-l 归约用
    {
        const unsigned int neg_inf_key = float_to_key(-INFINITY);
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            sRowMaxKey[my_row0 + 2 * rr] = neg_inf_key;
            sRowSum[my_row0 + 2 * rr] = 0.f;
        }
    }

    // ── acc_o（v9: WMMA accumulator 数组） ────────────────────────────────
    WmmaAccFrag acc_o[kOTiles];
    #pragma unroll
    for (int t = 0; t < kOTiles; ++t) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc_o[t].x[i] = 0.f;
    }

    // ── v9.1: Q A-fragments 常驻寄存器（仅 d=64：4 frags × 8 regs = 32 regs，
    //    REG 75 → ~118 仍充裕；d=128 需 64 regs 会溢出/降 occupancy，跳过）
    //    收益：每 iter 省 kTilesS×2 = 8 条 LDS.128（QK^T 的 A load 环路不变量）
    constexpr int kQFrags = (kHeadDim == 64) ? kTilesS : 0;
    WmmaAFrag q_frag[kQFrags > 0 ? kQFrags : 1];

    // ── Prologue：Q → sQ，prefetch K[0] → sK0 ─────────────────────────────
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
        actual_seqlen_q - m_block * kBlockM);
    {
        const bool is_edge = !Is_even_MN && (0 == n_block_max - 1);
        if (is_edge) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKsK0, tKVcKV, tKVpKV,
                actual_seqlen_k - 0 * kBlockN);
        } else {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKsK0, tKVcKV, tKVpKV);
        }
    }
    __syncthreads();  // prologue sync: Q store + K[0] store done

    // v9.1: Q → 寄存器 fragment（必须在 prologue sync 之后，sQ 已就绪）
    if constexpr (kQFrags > 0) {
        #pragma unroll
        for (int kt = 0; kt < kQFrags; ++kt) {
            q_frag[kt] = v9_load_a_row(sQ_ptr, 16 * warp_m, 16 * kt, ldm);
        }
    }

    // ── 主循环：K double buffer + 3 syncs/iter ─────────────────────────────
    for (int n_block = 0; n_block < n_block_max; ++n_block) {
        const bool use_buf0 = ((n_block & 1) == 0);

        // A. Mask load → smem + V load → smem + prefetch K[n+1] → 另一 buffer
        //    K[n] 已在 prologue 或上一轮 prefetch 中 load 到 sK[n&1]
        //    （沿用 v8.1：三个 gmem 访问并行发射，与 QK^T 计算重叠）
        {
            // mask copy（行/列联合谓词，k 维不 pad 契约：Sk%8==0 即可）
            //   行：越界行跳过（该行输出被 epilogue 丢弃，残留 P 值无害）
            //   列：仅全局边界 tile（!Is_even_MN 且 Sk%kBlockN!=0）存在越界列——
            //     谓词跳过拷贝（防 gmem 越界读，mask_seqlen_k 可非 kBlockN 倍数）
            //     并回写 -inf。sMask 与 sP 时间复用，每轮 P store 覆写全 64×64，
            //     sm89 式 prologue 一次性预清不可用 → -inf 逐轮回写
            //     （Phase A 受上轮 sync6 保护，与 P 读消费不 race）
            const bool mask_cols_full = Is_even_MN || (n_block < n_block_max - 1) ||
                                        (actual_seqlen_k % kBlockN == 0);
            if (mask_cols_full) {
                #pragma unroll
                for (int m = 0; m < size<1>(tQgMask(_, _, _, n_block)); ++m) {
                    #pragma unroll
                    for (int n = 0; n < size<2>(tQgMask(_, _, _, n_block)); ++n) {
                        if (tMpMask(m, n)) {
                            #pragma unroll
                            for (int v = 0; v < size<0>(tQgMask(_, m, n, n_block)); ++v) {
                                tQsMask_g2s(v, m, n) = tQgMask(v, m, n, n_block);
                            }
                        }
                    }
                }
            } else {
                const int mask_cols_left = actual_seqlen_k - n_block * kBlockN;  // 8 对齐
                const Element mask_neg_inf(static_cast<float>(-INFINITY));
                #pragma unroll
                for (int m = 0; m < size<1>(tQgMask(_, _, _, n_block)); ++m) {
                    #pragma unroll
                    for (int n = 0; n < size<2>(tQgMask(_, _, _, n_block)); ++n) {
                        if (get<1>(tMcMask(_0{}, m, n)) < mask_cols_left) {
                            if (tMpMask(m, n)) {
                                #pragma unroll
                                for (int v = 0; v < size<0>(tQgMask(_, m, n, n_block)); ++v) {
                                    tQsMask_g2s(v, m, n) = tQgMask(v, m, n, n_block);
                                }
                            }
                        } else {  // 列越界（≥ Sk）：回写 -inf，softmax 语义屏蔽位
                            #pragma unroll
                            for (int v = 0; v < size<0>(tQsMask_g2s); ++v) {
                                tQsMask_g2s(v, m, n) = mask_neg_inf;
                            }
                        }
                    }
                }
            }
            // V[n] gmem→smem（与 mask copy / K prefetch 并行发射）
            {
                const bool is_edge_v = !Is_even_MN && (n_block == n_block_max - 1);
                if (is_edge_v) {
                    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
                        actual_seqlen_k - n_block * kBlockN);
                } else {
                    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
                }
            }
            // prefetch K[n+1] → 另一 buffer (与 mask copy 并行)
            if (n_block + 1 < n_block_max) {
                const bool is_edge = !Is_even_MN && (n_block + 1 == n_block_max - 1);
                if (use_buf0) {
                    if (is_edge) {
                        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1),
                            tKsK1, tKVcKV, tKVpKV,
                            actual_seqlen_k - (n_block + 1) * kBlockN);
                    } else {
                        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1),
                            tKsK1, tKVcKV, tKVpKV);
                    }
                } else {
                    if (is_edge) {
                        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1),
                            tKsK0, tKVcKV, tKVpKV,
                            actual_seqlen_k - (n_block + 1) * kBlockN);
                    } else {
                        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(
                            gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1),
                            tKsK0, tKVcKV, tKVpKV);
                    }
                }
            }
            __syncthreads();  // sync1: K[n] ready + Mask store + V store + K[n+1] prefetch launched
        }

        // B. QK^T WMMA: warp (m,n) 独占 16×16 S-tile
        //    A = Q rows [16m, +16)（sQ），B = K^T cols [16n, +16)（sK 行 n）
        //    LDS/lane: kTilesS × (2×LDS.128 [A] + 2×LDS.128 [B])
        WmmaAccFrag acc_s;
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc_s.x[i] = 0.f;
        {
            const Element* sK_cur = use_buf0 ? sK0_ptr : sK1_ptr;
            #pragma unroll
            for (int kt = 0; kt < kTilesS; ++kt) {
                const WmmaAFrag a = (kQFrags > 0)
                    ? q_frag[kt]
                    : v9_load_a_row(sQ_ptr, 16 * warp_m, 16 * kt, ldm);
                const WmmaBFrag b = v9_load_b_col(sK_cur, 16 * warp_n, 16 * kt, ldm);
                wmma_mma_row_col(acc_s, a, b, acc_s);
            }
        }

        // B2. mask 加法 + scale → log2 域（读 sMask，4×LDS.32/lane）
        //     关键：避免 inf + (-inf) = nan，mask=-inf 时直接设 s=-inf
        {
            const float mask_inv_scale = 1.f / params.scale_softmax;
            float mask_vals[8];   // [rr][cc_pair] 对应 acc x[i] 的 mask 值
            const __half* smask_sp = reinterpret_cast<const __half*>(sP_ptr);
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int row = my_row0 + 2 * rr;
                const __half2 m01 = *reinterpret_cast<const __half2*>(
                    smask_sp + v9_smem_off(row, s_col_base, kBlockN));
                const __half2 m45 = *reinterpret_cast<const __half2*>(
                    smask_sp + v9_smem_off(row, s_col_base + 4, kBlockN));
                mask_vals[rr * 4 + 0] = __low2float(m01);   // col c0
                mask_vals[rr * 4 + 1] = __high2float(m01);  // col c0+1
                mask_vals[rr * 4 + 2] = __low2float(m45);   // col c0+4
                mask_vals[rr * 4 + 3] = __high2float(m45);  // col c0+5
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = (i >> 1) & 1;                       // 行（0/1）
                const int cc = (i & 1) + 2 * ((i >> 2) & 1);       // 0..3 mask_vals 索引
                const float mask_val = mask_vals[rr * 4 + cc];
                float s;
                if (isinf(mask_val) && mask_val < 0.f) {
                    s = -INFINITY;
                } else {
                    s = acc_s.x[i] + mask_val * mask_inv_scale;
                }
                acc_s.x[i] = s * params.scale_softmax_log2;
            }
        }

        // C. softmax（v9: 全 warp-local——每 S 行只属于一个 warp）
        //    m 归约: 线程内(4列) → 2×__shfl_xor(2,8)（同行 4 lanes）
        //    l 归约: partial-l 寄存器化（v8.1），epilogue 一次性 shuffle 归约
        float scale_o[2];
        float sum_partial[2] = {0.f, 0.f};
        {
            // ── Step 1: 线程内 max（每行 4 列: x[i], i∈{0,1,4,5}→行0） ──
            float m_tile[2];
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                m_tile[rr] = fmaxf(acc_s.x[2 * rr + 0], acc_s.x[2 * rr + 1]);
                m_tile[rr] = fmaxf(m_tile[rr], acc_s.x[2 * rr + 4]);
                m_tile[rr] = fmaxf(m_tile[rr], acc_s.x[2 * rr + 5]);
            }
            // ── Step 2: warp 内归约（同行 4 lanes: lane^2, lane^8） ──
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                m_tile[rr] = fmaxf(m_tile[rr], __shfl_xor_sync(0xffffffffu, m_tile[rr], 2));
                m_tile[rr] = fmaxf(m_tile[rr], __shfl_xor_sync(0xffffffffu, m_tile[rr], 8));
            }
            // ── Step 3: 跨 warp 归约（primary lane atomicMax；S 行横跨 4 个 warp_n） ──
            if (is_primary) {
                #pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    atomicMax(&sRowMaxKey[my_row0 + 2 * rr], float_to_key(m_tile[rr]));
                }
            }
            __syncthreads();  // syncM: atomicMax done
            // ── Step 4: 读回全局 max，计算 rescale factor ──
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const float m_global = key_to_float(sRowMaxKey[my_row0 + 2 * rr]);
                const float m_old = row_m[rr];
                scale_o[rr] = (m_old == -INFINITY) ? 0.f : exp2f(m_old - m_global);
                row_m[rr] = m_global;
            }
            // ── Step 5: rescale acc_o ──
            #pragma unroll
            for (int t = 0; t < kOTiles; ++t) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc_o[t].x[i] *= scale_o[(i >> 1) & 1];
                }
            }
            // ── Step 6: exp + partial sum（寄存器） ──
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const float m_scaled = (row_m[rr] == -INFINITY) ? 0.f : row_m[rr];
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    const int i = 2 * rr + (cc & 1) + ((cc >> 1) & 1) * 4;
                    const float p = exp2f(acc_s.x[i] - m_scaled);
                    acc_s.x[i] = p;
                    sum_partial[rr] += p;
                }
            }
            // ── Step 7: P(post-exp, half) → sP（4×STS.32/lane） ──
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int row = my_row0 + 2 * rr;
                const __half2 p01 = __floats2half2_rn(acc_s.x[2 * rr + 0], acc_s.x[2 * rr + 1]);
                const __half2 p45 = __floats2half2_rn(acc_s.x[2 * rr + 4], acc_s.x[2 * rr + 5]);
                __half* sp = reinterpret_cast<__half*>(sP_ptr);
                *reinterpret_cast<__half2*>(sp + v9_smem_off(row, s_col_base, kBlockN)) = p01;
                *reinterpret_cast<__half2*>(sp + v9_smem_off(row, s_col_base + 4, kBlockN)) = p45;
            }
        }
        __syncthreads();  // sync5: sP store done → PV 可读（P 行由 4 个 warp_n 分写）

        // D. PV WMMA: O += P @ V
        //    A = P rows [16m, +16)（sP，跨 warp 交换），B = V rows [16kt,+16)（sV 行主序）
        //    d=128: 两个 O-tile（warp_n 与 warp_n+4）共享同一 A fragment
        {
            #pragma unroll
            for (int kt = 0; kt < kTilesPV; ++kt) {
                const WmmaAFrag a = v9_load_a_row(sP_ptr, 16 * warp_m, 16 * kt, kBlockN);
                #pragma unroll
                for (int t = 0; t < kOTiles; ++t) {
                    const WmmaBFrag b = v9_load_b_row(
                        sV_ptr, 16 * (warp_n + 4 * t), 16 * kt, ldm);
                    wmma_mma_row_row(acc_o[t], a, b, acc_o[t]);
                }
            }
        }

        // row_l 更新（v8.1: 用本线程 partial sum，零 smem 通信）
        {
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                row_l[rr] = scale_o[rr] * row_l[rr] + sum_partial[rr];
            }
        }

        // sync6: PV mma 读 sP/sV 完成后才能进入下一轮
        // （下一轮 A 的 mask copy 要写 sP 区域（sMask=sP 时间复用），
        //   下一轮 A 的 prefetch 要写 sK[n&1]，均需等本轮读操作全部完成）
        __syncthreads();
    }

    // ── Epilogue：归一化 acc_o / l → fp16 → sO → gmem ─────────────────────
    {
        // v9: partial-l 最终归约：warp 内 2×__shfl_xor → primary atomicAdd sRowSum
        //（行横跨 4 个 warp_n，每 warp 的 partial-l 需跨 warp 求和）
        {
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                float l = row_l[rr];
                l += __shfl_xor_sync(0xffffffffu, l, 2);
                l += __shfl_xor_sync(0xffffffffu, l, 8);
                row_l[rr] = l;
            }
            if (is_primary) {
                #pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    atomicAdd(&sRowSum[my_row0 + 2 * rr], row_l[rr]);
                }
            }
            __syncthreads();
        }

        // 归一化 + half → sO（复用 sK0 空间，K/V 已不再需要）
        // sO [64, d] 用 v9 swizzle；每 lane 4×STS.32 per O-tile
        __half* sO_sp = reinterpret_cast<__half*>(sK0_ptr);
        #pragma unroll
        for (int t = 0; t < kOTiles; ++t) {
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const float l = sRowSum[my_row0 + 2 * rr];
                const float inv_l = (l == 0.f || l != l) ? 1.f : 1.f / l;
                const int row = my_row0 + 2 * rr;
                const int col = 16 * (warp_n + 4 * t) + c0;
                const __half2 o01 = __floats2half2_rn(
                    acc_o[t].x[2 * rr + 0] * inv_l, acc_o[t].x[2 * rr + 1] * inv_l);
                const __half2 o45 = __floats2half2_rn(
                    acc_o[t].x[2 * rr + 4] * inv_l, acc_o[t].x[2 * rr + 5] * inv_l);
                *reinterpret_cast<__half2*>(sO_sp + v9_smem_off(row, col, ldm)) = o01;
                *reinterpret_cast<__half2*>(sO_sp + v9_smem_off(row, col + 4, ldm)) = o45;
            }
        }
        __syncthreads();

        // gmem store: sO → register → gmem（沿用 cute copy，SmemLayoutQ 为 v9 布局）
        Tensor sO = make_tensor(make_smem_ptr(sK0_ptr), typename Kernel_traits::SmemLayoutQ{});
        Tensor mO = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + bidb * params.o_batch_stride),
            make_shape(actual_seqlen_q, params.h, params.d),
            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               make_coord(m_block, 0));
        Tensor tOsO_g = gmem_thr_copy_QKV.partition_S(sO);
        Tensor tOgO   = gmem_thr_copy_QKV.partition_D(gO);

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_QKV, tOsO_g, tOrO);

        Tensor cO_out = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
        Tensor tOcO_out = gmem_thr_copy_QKV.partition_D(cO_out);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO_out(0, 0, k)) < params.d; }
        }
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_QKV, tOrO, tOgO, tOcO_out, tOpO,
            actual_seqlen_q - m_block * kBlockM);
    }
}

// ── kernel 入口 ─────────────────────────────────────────────────────────────
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void __launch_bounds__(Kernel_traits::kNThreads)
flash_attn_with_mask_sm70_kernel(FA_mask_params params) {
    static_assert(Kernel_traits::kNThreads == 512,
                  "launch_bounds must match kNThreads");
    const int bidb = blockIdx.y / params.h;
    const int bidh = blockIdx.y % params.h;
    const int m_block = blockIdx.x;
    compute_attn_1rowblock_sm70<Kernel_traits, Is_even_MN, Is_even_K>(
        params, bidb, bidh, m_block);
}

// ── launcher ────────────────────────────────────────────────────────────────
template<typename Kernel_traits>
void run_flash_fwd_with_mask_sm70(const FA_mask_params &params, cudaStream_t stream) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int smem_size = Kernel_traits::kSmemSize;

    const int num_m_blocks = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid(num_m_blocks, params.b * params.h);

    const bool is_even_mn =
        (params.seqlen_q % kBlockM == 0) && (params.seqlen_k % Kernel_traits::kBlockN == 0);

    static_assert(smem_size <= 96 * 1024, "sm70 smem budget exceeded");

    using KT = Kernel_traits;
    if (is_even_mn) {
        auto kernel = flash_attn_with_mask_sm70_kernel<KT, /*Is_even_MN=*/true, /*Is_even_K=*/true>;
        static bool smem_set = false;
        if (!smem_set) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            smem_set = true;
        }
        kernel<<<grid, KT::kNThreads, smem_size, stream>>>(params);
    } else {
        auto kernel = flash_attn_with_mask_sm70_kernel<KT, /*Is_even_MN=*/false, /*Is_even_K=*/true>;
        static bool smem_set = false;
        if (!smem_set) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            smem_set = true;
        }
        kernel<<<grid, KT::kNThreads, smem_size, stream>>>(params);
    }
}

}  // namespace FA_MASK_NAMESPACE
