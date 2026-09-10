// group_gemm_config.cuh — MoE group GEMM 的跨架构共享 kernel traits。
//
// 架构分层（仿 csrc/fa）：
//   common/                 跨架构共享（本文件：traits + TypeTag；moe_kernels：
//                          count/gather/act/reduce；utils：PDL/向量工具）
//   sm89/group_gemm_sm89.cuh   cp.async 家族（sm80+ 通用基线，含 4090D=sm89）
//   sm120/group_gemm_sm120.cuh TMA + mbarrier 家族（Blackwell consumer，5090）
//   fuse_moe_launch.h       架构分发入口（编译期 FA_HAS_* × 运行期 gpu_major）
//
// config::MoEGemmConfig 被两家族共同引用：MMA atom（sm80 mma.sync 16x8x16，
// sm89/sm120 数值通路一致——sm120 无 wgmma）/ smem 布局（元素域 swizzle，
// ldmatrix 友好）/ cp.async 拷贝类型 / C 别名容量断言。sm120 家族的
// TmaGemmConfig 在其家族文件内继承本 traits 扩展（smem 布局换成 GMMA
// 规范的字节域 swizzle，16-bit 元素下数值等价，见 sm120 文件头注释）。
//
// swapAB 约定（与 hpc/mixed_gemm 一致）：MMA 计算 W @ X^T ——
//   A 操作数 = 权重 tile (kTileN, kTileK)（每组权重 (n, k) 行主序，n 必须是
//   kTileN=64 的倍数），B 操作数 = 激活 tile (kTileM, kTileK)，C = (kTileN,
//   kTileM) 转置视图，输出按 gY(col, row) 写回 (m, n) 行主序 gmem。

#ifndef FUSE_MOE_SRC_COMMON_GROUP_GEMM_CONFIG_CUH_
#define FUSE_MOE_SRC_COMMON_GROUP_GEMM_CONFIG_CUH_

#include <type_traits>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

namespace fuse_moe {
namespace group_gemm {

using namespace cute;  // NOLINT

// ─── Kernel traits ───────────────────────────────────────────────────────────
namespace config {

template <typename T>
struct TypeTag {
  using type = T;
};

template <typename T, int kTileM_, int kTileN_, int kTileK_, int kStage_>
struct MoEGemmConfig {
  using Tin = T;
  using Tout = T;
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;

  // MMA：sm80 mma.sync 16x8x16（sm120 无 wgmma，mma.sync 与 sm89 通用）
  using MMA_ATOM = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
                                      SM80_16x8x16_F32BF16BF16F32_TN,
                                      SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom<MMA_ATOM>{},
      make_layout(make_shape(Int<2>{}, Int<4>{}, Int<1>{})),
      Tile<Int<32>, Int<64>, Int<16>>{}));
  static constexpr int kNThreads = decltype(size(TiledMMA{}))::value;  // 256
  static_assert(kNThreads == 256, "sm89/sm120 MoE gemm assumes 8-warp MMA tiling");
  static_assert(kTileM_ % 32 == 0 && kTileN_ % 64 == 0 && kTileK_ % 64 == 0,
                "tile shape must match MMA tiling (M%32, N%64, K%64)");

  // smem 布局：K-major swizzle atom (8, 64)（16-bit 元素域 Swizzle<3,3,3>，
  // ldmatrix 友好；kTileK=128 由 tile_to_shape 扩展——FA d=128 同款）
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}))));
  using SmemLayoutX = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTileM_>{}, Int<kTileK_>{}, Int<kStage_>{})));
  using SmemLayoutW = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTileN_>{}, Int<kTileK_>{}, Int<kStage_>{})));
  // C：(kTileN, kTileM)，mode0 连续、无 swizzle；M-stride = kTileN+8（=72）
  // 填充消 bank 冲突：r2s 的 warp 内访问模式为 (n, m) = (tid/4,
  // 2*(tid%4)+v0)，地址 n + stride*m。stride=64 时 64*2B=128B ≡ bank 周期，
  // m 维各行全落同一 bank 组 → 8 路冲突（ncu 4090D S=16384 实测
  // 6.37M 过量 wavefront，est. speedup 8%）。stride=72 → 144B 行距（保持
  // 16B 对齐，s2g 的 uint4 读不受影响），bank = (n/2 + 36m) & 31
  // 随 m 展开互异（相邻 n 对共字由硬件写合并）。
  using SmemLayoutC = decltype(make_layout(
      make_shape(Int<kTileN_>{}, Int<kTileM_>{}), make_stride(Int<1>{}, Int<kTileN_ + 8>{})));

  // g2s cp.async：16B = 8 个 16-bit 元素/线程
  static constexpr int kElemsPerAtom = 16 / int(sizeof(T));           // 8
  static constexpr int kThreadsPerRow = kTileK_ / kElemsPerAtom;      // 8 (K=64) / 16 (K=128)
  static constexpr int kRowsPerIter = kNThreads / kThreadsPerRow;     // 32 / 16
  using CpAsyncAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, T>;
  using G2SCopy = decltype(make_tiled_copy(
      CpAsyncAtom{},
      make_layout(make_shape(Int<kRowsPerIter>{}, Int<kThreadsPerRow>{}),
                  make_stride(Int<kThreadsPerRow>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<kElemsPerAtom>{}),
                  make_stride(Int<0>{}, Int<1>{}))));

  static constexpr int shm_xw =
      static_cast<int>(sizeof(T) * (cosize(SmemLayoutX{}) + cosize(SmemLayoutW{})));
  static constexpr int shm_c = static_cast<int>(sizeof(T) * cosize(SmemLayoutC{}));
  // sC 预留区按 (kTileN, max(kTileM, 64)) 分配：TiledMMA 的 C tile 形状
  // 由 Tile<32,64,16> 与 thr (2,4,1) 组合决定（恒为 N=64 × M=64 的覆盖面，
  // 与 kTileM 无关，frag 总量 16×256=4096 elems），kTileM<64 时 r2s 分区
  //（make_tiled_copy_C）会写到 sC 逻辑边界之外（按同 stride 布局计址，最高触及 72*63+64 元素 @M32）。
  // s2g 只读 [0, cosize(SmemLayoutC))，越界写部分属被行谓词丢弃的 M 尾部
  // 垃圾行，语义无关。
  static constexpr int shm_c_pad_elems =
      (kTileN_ + 8) * ((kTileM_ < 64 ? 64 : kTileM_) - 1) + kTileN_;
  static constexpr int shm_c_alloc =
      (shm_c > static_cast<int>(sizeof(T) * shm_c_pad_elems))
          ? shm_c
          : static_cast<int>(sizeof(T) * shm_c_pad_elems);
  // sm89 家族：sC 为独立 smem 区（跨 tile 连续流水需要，见
  // sm89/group_gemm_sm89.cuh）；sm120 家族：别名 operand 基地址。
  // 两家族均满足 shm_c ≤ shm_xw（sm89 预算公式以此为前提）。
};

}  // namespace config

}  // namespace group_gemm
}  // namespace fuse_moe

#endif  // FUSE_MOE_SRC_COMMON_GROUP_GEMM_CONFIG_CUH_
