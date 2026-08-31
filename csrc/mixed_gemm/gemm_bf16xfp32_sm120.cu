// Mixed-precision GEMM (bf16 + fp8/int8 residual) — sm120a (Blackwell consumer,
// RTX 5090/5090D) 专用实现。
//
// 与 gemm_bf16xfp32_sm80.cu 的关系：数值算法完全一致（bf16 主项 mma +
// fp8/int8 residual mma 共享单个 fp32 累加器；split-K 的分块/归约语义不变），
// 仅数据通路不同：
//   sm80  : cp.async（256 线程各自发射 128-bit 拷贝）+ ZFILL 谓词张量
//           + cp_async_wait/__syncthreads 流水 + elementwise 输出
//   sm120 : TMA（thread0 单线程发射 bulk 拷贝）+ mbarrier full/empty 流水
//           + TMA store epilogue
//
// 设计依据（移植自本仓库 FA 的 sm120 移植经验，见
// docs/fa_sm120_porting_and_optimization.md）：
//   - sm120 无 wgmma / tcgen05 → 计算仍用 mma.sync（与 sm80 相同的
//     TiledMMA / TiledMMAFp8 及 fragment 划分，逐位一致）
//   - TMA 仅 shared::cta（无 cluster / multicast）；mbarrier / STSM 可用
//   - warp specialization 在 sm120 上是负收益（FA Phase 4：mma.sync 是同步
//     指令，cooperative 模式已拿到大部分重叠收益）→ 沿用 cooperative 模式：
//     thread0 负责 TMA 发射，其余 255 线程立即进入计算
//   - 每 CTA 动态 smem 上限 101376B（与 Ada 同级）
//
// 相对 sm80 路径的结构性收益：
//   1. gmem→smem 由 TMA 批量搬运：单线程发射（省下 256 线程的 cp.async
//      发射槽）、bulk 传输对 DRAM/L2 更友好、OOB 自动补零
//      （== sm80 的 ZFILL 谓词语义，消灭全部 M/N/K 谓词张量与分支）
//   2. mbarrier full/empty 对替代 cp_async_wait + __syncthreads：warp 间
//      不再在 K-slab 边界强制对齐，producer（thread0）可提前 kStage 级发射
//   3. Epilogue（splitk==1 且对齐满足）：寄存器 →(bias+act+降精度)→
//      swizzled smem → TMA store 批量异步写，自动裁剪越界行列
//   4. L2 cache hint：W_high/W_low（跨 M-tile 复用，kBlockSwizzle=4 让同
//      itile_n 的 4 个 CTA 在时间上相邻）EVICT_LAST 常驻 L2；X（流式，
//      复用距离远）EVICT_FIRST
//
// smem 布局（kTileK=64 固定；每 stage = (kTileM+kTileN)*64*3 字节）：
//   [sX(bf16) | sWH(bf16) | sX8(1B) | sWL8(1B)] × kStage | full/empty barriers
//     (128,128) kStage=2 → 96KB     (64,128) kStage=2 → 72KB
//     (64,64)   kStage=4 → 96KB
//   sY 复用 operand 区域开头（静态断言保证放得下），tile 边界处流水线排空。
//
// persistent tile loop 内的流水线相位记账：producer_slab / consumer_slab 是
// 跨 tile 连续的全局计数（stage = cnt % kStage，phase = (cnt/kStage)&1），
// 因此 tile 边界无需复位任何 mbarrier 状态。
//
// 路由条件（见 mixed_gemm_op.cu）：设备 major==12 且二进制含 sm120a 编译
// 目标（FA_HAS_SM120，见 csrc/arch_targets.h 标准入口；与 FA 的分发同一判定），
// 且 k%16==0（1 字节 residual 张量的 TMA 全局行 stride 需 16B 对齐——
// sm80 路径只需 k%8）+ 各输入指针 16B 对齐；否则整体回退 sm80 路径（数值
// 语义完全一致）。TMA store 额外要求 n*sizeof(TY)%16==0 且 y 指针 16B
// 对齐，不满足时仅 epilogue 降级为 elementwise（其余路径不变）。
//
// 调试环境变量：
//   GEMM_MIXED_FORCE_SM80=1    （mixed_gemm_op.cu）强制回退 sm80 路径，A/B 用
//   GEMM_MIXED_NO_TMA_STORE=1  关闭 TMA store（epilogue 降级 elementwise），A/B 用
//   GEMM_MIXED_DEBUG_GRID=1    打印 kernel 配置 / grid / has_tma_store

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <type_traits>
#include <mutex>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_types.h"
#include "cutlass/fast_math.h"

#include "arch_targets.h"   // FA_HAS_SM120（架构条件编译标准入口）
#include "gemm_bf16xfp32_sm80.h"
#include "gemm_bf16xfp32_sm120.h"
#include "utils.cuh"

namespace mixed_gemm {
namespace kernels_sm120 {

using namespace cute;  // NOLINT

// Residual 后端（与 gemm_bf16xfp32_sm80.cu 的 kernels:: 定义一致；本翻译
// 单元独立持有副本以保持 sm80/sm120 两个文件完全解耦）。
#if MIXED_GEMM_FP8_ENABLED
struct Fp8ResidualBackend {
  using Element = cute::float_e4m3_t;
  using AccElement = float;
  using MmaAtom = SM89_16x8x32_F32E4M3E4M3F32_TN;
};
#endif

struct Int8ResidualBackend {
  using Element = int8_t;
  using AccElement = int32_t;
  using MmaAtom = SM80_16x8x32_S32S8S8S32_TN;
};

// ── persistent tile 调度（与 sm80 路径逐字一致）────────────────────────────
// 线性化 blockIdx.x → (itile_m, itile_n, ichunk)：kBlockSwizzle 沿 M 分组
// 提升 W 的 L2 复用；ichunk 为 split-K 的交错 K-chunk 编号。
template <int kBlockSwizzle, int kSplitK>
__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              cutlass::FastDivmod swizzle_divider,
                                              cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n * kSplitK;
  int total_swizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_swizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);
    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  int ichunk = itile_n % kSplitK;
  itile_n = itile_n / kSplitK;
  return cute::make_tuple(itile_m, itile_n, ichunk);
}

// split-K partial 归约（与 sm80 路径逐字一致，含 float4 对齐约束的全部处理；
// 详见 gemm_bf16xfp32_sm80.cu 中同函数的注释）。
template <typename Tout, typename Activation, bool HasBias, int kTileM, int kTileN,
          int kSplitK, int kNumWarp>
__device__ __forceinline__ void splitk_reduce(Tout *y_ptr, const float *splitk_y_ptr,
                                              const float *bias_ptr, int m, int n,
                                              int itile_m, int itile_n) {
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  if (itile_m * kTileM + iwarp >= m) return;

  int n_tile_start = itile_n * kTileN;
  int n_local = min(kTileN, n - n_tile_start);
  int col0 = ilane * 4;
  if (col0 >= n_local) return;

  auto *y_tile = y_ptr + static_cast<int64_t>(itile_m * kTileM + iwarp) * n + n_tile_start + col0;
  auto *splitk_y_tile =
      splitk_y_ptr + static_cast<int64_t>(itile_m * kTileM + iwarp) * n + n_tile_start + col0;

  int local_m = m - (itile_m * kTileM + iwarp);
  int num_valid_cols = min(4, n_local - col0);
  bool use_vectorized = (num_valid_cols == 4) && (n % 4 == 0);

#pragma unroll
  for (int irow = 0; irow < kTileM; irow += kNumWarp) {
    if (irow >= local_m) return;

    if (use_vectorized) {
      auto y = load<float, 4>(splitk_y_tile + static_cast<int64_t>(irow) * n);
#pragma unroll
      for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
        auto part = load<float, 4>(splitk_y_tile + static_cast<int64_t>(ichunk) * m * n +
                                    static_cast<int64_t>(irow) * n);
#pragma unroll
        for (int i = 0; i < 4; ++i) y[i] += part[i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (HasBias) y[i] += bias_ptr[n_tile_start + col0 + i];
        y[i] = Activation{}(y[i]);
      }
      store(y_tile + static_cast<int64_t>(irow) * n, to<Tout>(y));
    } else {
      const float *base = splitk_y_tile + static_cast<int64_t>(irow) * n;
      for (int i = 0; i < num_valid_cols; ++i) {
        float acc = base[i];
#pragma unroll
        for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
          acc += base[static_cast<int64_t>(ichunk) * m * n + i];
        }
        if constexpr (HasBias) acc += bias_ptr[n_tile_start + col0 + i];
        (y_tile + static_cast<int64_t>(irow) * n)[i] = static_cast<Tout>(Activation{}(acc));
      }
    }
  }
}

// ── Kernel traits ────────────────────────────────────────────────────────────
template <int kTileM_, int kTileN_, int kStage_, int kSplitK_,
          typename ResidualBackend_, typename Tout_, typename Activation_, bool HasBias_>
struct GemmSm120Traits {
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = 64;   // 与 sm80 路径一致（128B bf16 swizzle atom）
  static constexpr int kStage = kStage_;
  static constexpr int kSplitK = kSplitK_;
  static constexpr int kBlockSwizzle = 4;
  static constexpr bool kIsInt8 = std::is_same_v<ResidualBackend_, Int8ResidualBackend>;

  using ResidualBackend = ResidualBackend_;
  using Tin = cute::bfloat16_t;
  using ResElem = typename ResidualBackend_::Element;
  using Tout = Tout_;
  using Activation = Activation_;
  static constexpr bool HasBias = HasBias_;
  // split-K partial 恒为 fp32；非 split-K 输出 dtype = Tout
  using TY = std::conditional_t<(kSplitK > 1), float, Tout>;

  // ── MMA（与 sm80 路径完全一致的 16x8x16 bf16 + 16x8x32 residual atom）──
  using MMA_ATOM = SM80_16x8x16_F32BF16BF16F32_TN;
  using MMAThrLayout = decltype(make_layout(make_shape(Int<2>{}, Int<4>{}, Int<1>{})));
  using MMAPermutation = Tile<Int<32>, Int<64>, Int<16>>;
  using TiledMma = decltype(make_tiled_mma(MMA_Atom<MMA_ATOM>{}, MMAThrLayout{}, MMAPermutation{}));

  using MMA_ATOM_RES = typename ResidualBackend_::MmaAtom;
  using MMAPermutationRes = Tile<Int<32>, Int<64>, Int<32>>;
  using TiledMmaRes = decltype(make_tiled_mma(MMA_Atom<MMA_ATOM_RES>{}, MMAThrLayout{},
                                              MMAPermutationRes{}));

  static constexpr int kNThreads = decltype(size(TiledMma{}))::value;  // 256（8 warps）
  static_assert(kNThreads == 256, "sm120 path assumes the sm80 8-warp MMA tiling");

  // ── smem layouts：必须用 GMMA 规范 atom（byte 域 swizzle + smem_ptr_flag），
  //    make_tma_copy 才能编程 TMA descriptor 的 SWIZZLE_128B/64B 模式。
  //    bf16 K-major 128B：GMMA SW128 atom 与 sm80 的 Swizzle<3,3,3>∘(8,64)
  //    同为「16B 块保持完整」的 ldmatrix 友好模式（FA sm120 已验证）。
  //    1 字节 residual K-major 64B（kTileK=64 元素 = 64B 行宽）。
  using SmemLayoutAtomB16 = GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>;  // (8,64)
  using SmemLayoutAtomRes = GMMA::Layout_K_SW64_Atom<ResElem>;            // (8,64)

  using SmemLayoutX  = decltype(tile_to_shape(SmemLayoutAtomB16{}, Shape<Int<kTileM>, Int<kTileK>, Int<kStage>>{}));
  using SmemLayoutW  = decltype(tile_to_shape(SmemLayoutAtomB16{}, Shape<Int<kTileN>, Int<kTileK>, Int<kStage>>{}));
  using SmemLayoutX8 = decltype(tile_to_shape(SmemLayoutAtomRes{}, Shape<Int<kTileM>, Int<kTileK>, Int<kStage>>{}));
  using SmemLayoutW8 = decltype(tile_to_shape(SmemLayoutAtomRes{}, Shape<Int<kTileN>, Int<kTileK>, Int<kStage>>{}));
  using SmemLayoutXStage  = decltype(take<0, 2>(SmemLayoutX{}));
  using SmemLayoutWStage  = decltype(take<0, 2>(SmemLayoutW{}));
  using SmemLayoutX8Stage = decltype(take<0, 2>(SmemLayoutX8{}));
  using SmemLayoutW8Stage = decltype(take<0, 2>(SmemLayoutW8{}));

  // Epilogue sY：(kTileN, kTileM) 列主（mode0 = MMA_M = GEMM-N = gmem 连续维），
  // 无 swizzle。说明：TMA store 的 smem box 只接受「无 swizzle」或 GMMA byte 域
  // swizzle 模式；GMMA MN-atom 的 mode0 固定为 128B 周期（bf16 512 元素），
  // 无法 tile 到 kTileN∈{64,128}，故此处放弃 swizzle（sm80 路径的手工
  // element 域 swizzle 不被 TMA descriptor 支持）。代价是 r2s copy 存在
  // bank sharing（fp32 标量写、每 4 列共享 bank），但该写仅每 tile 一次、
  // 远小于 mainloop 的 ldmatrix/mma 流量，且换来 TMA store 的 bulk 异步写
  // + OOB 自动裁剪。sm80 的 swizzle 推导见其 slayout_y 注释。
  using SmemLayoutY = decltype(make_layout(Shape<Int<kTileN>, Int<kTileM>>{},
                                           Stride<_1, Int<kTileN>>{}));
  // TMA store 的 smem 永远是 Tout 版本（splitk>1 时不使用 TMA store）
  using SmemLayoutYStore = decltype(make_layout(Shape<Int<kTileN>, Int<kTileM>>{},
                                                 Stride<_1, Int<kTileN>>{}));

  // ── smem 尺寸 ────────────────────────────────────────────────────────
  static constexpr int kSmemXWSize =
      static_cast<int>(sizeof(cute::bfloat16_t) * (cosize(SmemLayoutX{}) + cosize(SmemLayoutW{})) +
                       sizeof(ResElem) * (cosize(SmemLayoutX8{}) + cosize(SmemLayoutW8{})));
  static constexpr int kSmemYSize = static_cast<int>(sizeof(TY) * cosize(SmemLayoutY{}));
  static_assert(kSmemYSize <= kSmemXWSize,
                "epilogue sY must alias the (larger) operand smem region");
  static constexpr int kSmemBarrierOffset = kSmemXWSize;
  static constexpr int kSmemBarrierSize = 2 * kStage * 8;  // full + empty mbarrier
  static constexpr int kSmemSize = kSmemXWSize + kSmemBarrierSize;
  static_assert(kSmemSize <= 101376, "sm120 per-CTA dynamic smem limit exceeded");

  // ── TMA 类型（descriptor 在 host 侧每次调用时构建，经 __grid_constant__
  //    参数传入 kernel；类型仅依赖 smem layout / gmem stride 静态形状）────
  using ShapeXW = Shape<int32_t, int32_t>;
  using StrideXW = Stride<int64_t, _1>;   // (rows, k):(k, 1) row-major
  using StrideYT = Stride<_1, int64_t>;   // (n, m):(1, n)  Y^T 视图
  using TMA_X  = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<cute::bfloat16_t const*>(nullptr)), ShapeXW{}, StrideXW{}),
      SmemLayoutXStage{}));
  using TMA_W  = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<cute::bfloat16_t const*>(nullptr)), ShapeXW{}, StrideXW{}),
      SmemLayoutWStage{}));
  using TMA_X8 = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ResElem const*>(nullptr)), ShapeXW{}, StrideXW{}),
      SmemLayoutX8Stage{}));
  using TMA_W8 = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ResElem const*>(nullptr)), ShapeXW{}, StrideXW{}),
      SmemLayoutW8Stage{}));
  using TMA_Y = decltype(make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Tout*>(nullptr)), ShapeXW{}, StrideYT{}),
      SmemLayoutYStore{}));

  // 一个 stage 的 TMA 事务字节：X/W 的 bf16 + 1 字节 residual 共 4 路
  static constexpr uint32_t TmaTransactionBytes = static_cast<uint32_t>(
      (kTileM + kTileN) * kTileK * (sizeof(cute::bfloat16_t) + sizeof(ResElem)));
};

// ── 参数（含 TMA descriptor，需 __grid_constant__ 传参）──────────────────────
template <typename Traits>
struct GemmSm120Params {
  typename Traits::TMA_X  tma_load_X;
  typename Traits::TMA_W  tma_load_W;
  typename Traits::TMA_X8 tma_load_X8;
  typename Traits::TMA_W8 tma_load_W8;
  typename Traits::TMA_Y  tma_store_Y;   // 仅 has_tma_store 时有效

  const float *bias_ptr;     // 可为 null（!HasBias）
  const float *x_scale_ptr;  // INT8 后端
  const float *w_scale_ptr;  // INT8 后端
  typename Traits::Tout *y_ptr;
  float *splitk_y_ptr;       // splitk>1
  int *split_flag_ptr;       // splitk>1
  int m, n, k;
  float scale;
  bool has_tma_store;
  cutlass::FastDivmod swizzle_divider;
  cutlass::FastDivmod flat_divider;
  cutlass::FastDivmod reduce_flat_divider;
};

// ── 核心 kernel ──────────────────────────────────────────────────────────────
template <typename Traits>
__global__ void __launch_bounds__(Traits::kNThreads, 1)
gemm_bf16xfp32_kernel_sm120(
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __grid_constant__
#endif
    const GemmSm120Params<Traits> params) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  using namespace cute;  // NOLINT
  using bf16 = cute::bfloat16_t;
  using ResElem = typename Traits::ResElem;
  using TY = typename Traits::TY;
  using Activation = typename Traits::Activation;
  constexpr bool kIsInt8 = Traits::kIsInt8;
  constexpr int kTileM = Traits::kTileM;
  constexpr int kTileN = Traits::kTileN;
  constexpr int kTileK = Traits::kTileK;
  constexpr int kStage = Traits::kStage;
  constexpr int kSplitK = Traits::kSplitK;
  constexpr int kBlockSwizzle = Traits::kBlockSwizzle;
  constexpr int kNThreads = Traits::kNThreads;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;

  const int tidx = threadIdx.x;

  // ── smem 排布：[sX | sWH | sX8 | sWL8] × kStage | full/empty barriers ──
  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_x = reinterpret_cast<bf16 *>(shm_data);
  auto *shm_w = shm_x + cosize(typename Traits::SmemLayoutX{});
  auto *shm_x8 = reinterpret_cast<ResElem *>(shm_w + cosize(typename Traits::SmemLayoutW{}));
  auto *shm_w8 = shm_x8 + cosize(typename Traits::SmemLayoutX8{});
  auto *shm_y = reinterpret_cast<TY *>(shm_data);  // 复用 operand 区域（静态断言保证容量）

  auto sX = make_tensor(make_smem_ptr(shm_x), typename Traits::SmemLayoutX{});
  auto sWH = make_tensor(make_smem_ptr(shm_w), typename Traits::SmemLayoutW{});
  auto sX8 = make_tensor(make_smem_ptr(shm_x8), typename Traits::SmemLayoutX8{});
  auto sWL8 = make_tensor(make_smem_ptr(shm_w8), typename Traits::SmemLayoutW8{});
  auto sY = make_tensor(make_smem_ptr(shm_y), typename Traits::SmemLayoutY{});

  FullBarrier *full_bar = reinterpret_cast<FullBarrier *>(shm_data + Traits::kSmemBarrierOffset);
  EmptyBarrier *empty_bar = reinterpret_cast<EmptyBarrier *>(full_bar + kStage);

  if (tidx < kStage) {
    full_bar[tidx].init(1);          // 仅 thread0 arrive_and_expect_tx
    empty_bar[tidx].init(kNThreads); // 所有 consumer 线程各自 arrive
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  // ── tile 计数（kernel 内的 num_tile_n 为未乘 kSplitK 的原始值，与 sm80 一致）──
  int num_tile_m = (params.m + kTileM - 1) / kTileM;
  int num_tile_n = (params.n + kTileN - 1) / kTileN;
  int ntile_k = (params.k + kTileK - 1) / kTileK;
  int ntile_k_base = ntile_k / kSplitK;
  int ntile_k_rem = ntile_k % kSplitK;

  // ── MMA / s2r 句柄（tile 循环外构造一次；划分与 sm80 路径逐位一致）────
  typename Traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);
  typename Traits::TiledMmaRes tiled_mma_res;
  auto thr_mma_res = tiled_mma_res.get_slice(tidx);

  Tensor tWHr = thr_mma.partition_fragment_A(sWH(_, _, _0{}));
  Tensor tXr = thr_mma.partition_fragment_B(sX(_, _, _0{}));
  Tensor tWLr_res = thr_mma_res.partition_fragment_A(sWL8(_, _, _0{}));
  Tensor tXr_res = thr_mma_res.partition_fragment_B(sX8(_, _, _0{}));

  // 单个共享 fp32 累加器（bf16 主项 + residual 折入；INT8 的 int32 residual
  // 累加器贯穿 K-loop，scale 在 tile 末尾统一应用——均与 sm80 一致）
  Tensor tYr = thr_mma.partition_fragment_C(sY);
  Tensor tYr_res_tmp = thr_mma_res.partition_fragment_C(sY);
  Tensor cCoord = thr_mma.partition_C(make_identity_tensor(shape(sY)));

  using LDSM_ATOM = Copy_Atom<SM75_U32x4_LDSM_N, bf16>;
  auto s2r_copy_a = make_tiled_copy_A(LDSM_ATOM{}, tiled_mma);  // A = W
  auto s2r_copy_b = make_tiled_copy_B(LDSM_ATOM{}, tiled_mma);  // B = X
  auto s2r_thr_copy_a = s2r_copy_a.get_slice(tidx);
  auto s2r_thr_copy_b = s2r_copy_b.get_slice(tidx);

  using LDSM_ATOM_RES = Copy_Atom<SM75_U32x4_LDSM_N, ResElem>;
  auto s2r_copy_a_res = make_tiled_copy_A(LDSM_ATOM_RES{}, tiled_mma_res);
  auto s2r_copy_b_res = make_tiled_copy_B(LDSM_ATOM_RES{}, tiled_mma_res);
  auto s2r_thr_copy_a_res = s2r_copy_a_res.get_slice(tidx);
  auto s2r_thr_copy_b_res = s2r_copy_b_res.get_slice(tidx);

  Tensor tXs4r = s2r_thr_copy_b.partition_S(sX);    // (CPY, CPY_N, CPY_K, stage)
  Tensor tWHs4r = s2r_thr_copy_a.partition_S(sWH);  // (CPY, CPY_M, CPY_K, stage)
  Tensor tX8s4r = s2r_thr_copy_b_res.partition_S(sX8);
  Tensor tWL8s4r = s2r_thr_copy_a_res.partition_S(sWL8);

  auto tXr_view = s2r_thr_copy_b.retile_D(tXr);
  auto tWHr_view = s2r_thr_copy_a.retile_D(tWHr);
  auto tXr_res_view = s2r_thr_copy_b_res.retile_D(tXr_res);
  auto tWLr_res_view = s2r_thr_copy_a_res.retile_D(tWLr_res);

  constexpr int kNumSubK = decltype(size<2>(tXr))::value;        // kTileK/16 = 4
  constexpr int kNumSubKRes = decltype(size<2>(tXr_res))::value; // kTileK/32 = 2
  static_assert(kNumSubK == 2 * kNumSubKRes,
               "residual mma (K=32) must exactly halve the bf16 (K=16) sub-k steps");

  auto s2r_copy_ik = [&](int ik, int stage) {
    cute::copy(s2r_copy_b, tXs4r(_, _, ik, stage), tXr_view(_, _, ik));
    cute::copy(s2r_copy_a, tWHs4r(_, _, ik, stage), tWHr_view(_, _, ik));
  };
  auto s2r_copy_ikf = [&](int ikf, int stage) {
    cute::copy(s2r_copy_b_res, tX8s4r(_, _, ikf, stage), tXr_res_view(_, _, ikf));
    cute::copy(s2r_copy_a_res, tWL8s4r(_, _, ikf, stage), tWLr_res_view(_, _, ikf));
  };

  // ── TMA gmem 坐标张量与 smem 侧分区（tile 循环外构造一次）──────────────
  Tensor mX = params.tma_load_X.get_tma_tensor(make_shape(params.m, params.k));
  Tensor mW = params.tma_load_W.get_tma_tensor(make_shape(params.n, params.k));
  Tensor mX8 = params.tma_load_X8.get_tma_tensor(make_shape(params.m, params.k));
  Tensor mW8 = params.tma_load_W8.get_tma_tensor(make_shape(params.n, params.k));
  auto block_tma_X = params.tma_load_X.get_slice(_0{});
  auto block_tma_W = params.tma_load_W.get_slice(_0{});
  auto block_tma_X8 = params.tma_load_X8.get_slice(_0{});
  auto block_tma_W8 = params.tma_load_W8.get_slice(_0{});
  Tensor tXs = group_modes<0, 3>(block_tma_X.partition_D(sX));    // (TMA, stage)
  Tensor tWs = group_modes<0, 3>(block_tma_W.partition_D(sWH));
  Tensor tX8s = group_modes<0, 3>(block_tma_X8.partition_D(sX8));
  Tensor tW8s = group_modes<0, 3>(block_tma_W8.partition_D(sWL8));

  if (tidx == 0) {
    cute::prefetch_tma_descriptor(params.tma_load_X.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_W.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_X8.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_W8.get_tma_descriptor());
    if (params.has_tma_store) {
      cute::prefetch_tma_descriptor(params.tma_store_Y.get_tma_descriptor());
    }
  }

  // ── persistent tile loop ───────────────────────────────────────────────
  int iblock = blockIdx.x;
  int last_tile_m = -1;
  int last_tile_n = -1;
  // 跨 tile 连续的全局 slab 计数：stage = cnt % kStage，phase = (cnt/kStage)&1。
  // tile 边界无需复位 mbarrier（相位按全局使用次数自然延续）。
  int producer_slab = 0;
  int consumer_slab = 0;

  while (true) {
    auto [itile_m, itile_n, ichunk] = get_next_tile<kBlockSwizzle, kSplitK>(
        iblock, num_tile_m, num_tile_n, params.swizzle_divider, params.flat_divider);
    if (itile_m >= num_tile_m) break;
    iblock += gridDim.x;

    int ntile_k_local = ntile_k_base + (ichunk < ntile_k_rem ? 1 : 0);

    // 每 tile 重建 gmem tile 视图与 TMA 源分区（itile_m/itile_n 变化）
    Tensor gX = local_tile(mX, Shape<Int<kTileM>, Int<kTileK>>{}, make_coord(itile_m, _));
    Tensor gW = local_tile(mW, Shape<Int<kTileN>, Int<kTileK>>{}, make_coord(itile_n, _));
    Tensor gX8 = local_tile(mX8, Shape<Int<kTileM>, Int<kTileK>>{}, make_coord(itile_m, _));
    Tensor gW8 = local_tile(mW8, Shape<Int<kTileN>, Int<kTileK>>{}, make_coord(itile_n, _));
    Tensor tXg = group_modes<0, 3>(block_tma_X.partition_S(gX));    // (TMA, ntile_k)
    Tensor tWg = group_modes<0, 3>(block_tma_W.partition_S(gW));
    Tensor tX8g = group_modes<0, 3>(block_tma_X8.partition_S(gX8));
    Tensor tW8g = group_modes<0, 3>(block_tma_W8.partition_S(gW8));

    // producer（仅 thread0）：发射一个 K-slab 的 4 路 TMA 到 stage。
    // W（high+low）跨 M-tile 复用（kBlockSwizzle 分组）→ EVICT_LAST；
    // X 流式 → EVICT_FIRST。
    auto issue_slab = [&](int gk) {
      int stage = producer_slab % kStage;
      uint64_t &bar = reinterpret_cast<uint64_t &>(full_bar[stage]);
      full_bar[stage].arrive_and_expect_tx(Traits::TmaTransactionBytes);
      cute::copy(params.tma_load_W.with(bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                 tWg(_, gk), tWs(_, stage));
      cute::copy(params.tma_load_W8.with(bar, 0, cute::TMA::CacheHintSm90::EVICT_LAST),
                 tW8g(_, gk), tW8s(_, stage));
      cute::copy(params.tma_load_X.with(bar, 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
                 tXg(_, gk), tXs(_, stage));
      cute::copy(params.tma_load_X8.with(bar, 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
                 tX8g(_, gk), tX8s(_, stage));
      ++producer_slab;
    };
    // 复用一个 stage 前先等它的 empty（上一轮消费完成）；首轮使用免等
    auto acquire_and_issue = [&](int gk) {
      if (producer_slab >= kStage) {
        uint32_t ephase = ((producer_slab / kStage) - 1) & 1;
        empty_bar[producer_slab % kStage].wait(ephase);
      }
      issue_slab(gk);
    };

    // prologue：thread0 预发射 min(kStages, ntile_k_local) 个 slab
    if (tidx == 0) {
      int n_prologue = kStage < ntile_k_local ? kStage : ntile_k_local;
      for (int s = 0; s < n_prologue; ++s) {
        acquire_and_issue(ichunk + s * kSplitK);
      }
    }

    clear(tYr);
    if constexpr (kIsInt8) clear(tYr_res_tmp);  // int32 residual 累加器按 tile 清零

    // ── K-loop：寄存器级 s2r/mma 交错与 sm80 完全一致，仅同步原语换成
    //    full/empty mbarrier（无 cp_async_wait / __syncthreads）──────────
    for (int itile = 0; itile < ntile_k_local; ++itile) {
      int stage = consumer_slab % kStage;
      uint32_t phase = (consumer_slab / kStage) & 1;
      full_bar[stage].wait(phase);

      s2r_copy_ik(0, stage);
#pragma unroll
      for (int ik = 0; ik < kNumSubK; ++ik) {
        if (ik + 1 < kNumSubK) {
          s2r_copy_ik(ik + 1, stage);
        }
        if (ik % 2 == 0) {
          int ikf = ik / 2;
          s2r_copy_ikf(ikf, stage);
          if constexpr (!kIsInt8) clear(tYr_res_tmp);
          cute::gemm(tiled_mma_res, tWLr_res(_, _, ikf), tXr_res(_, _, ikf), tYr_res_tmp);
          if constexpr (!kIsInt8) {
            // FP8：逐 mma 应用统一 residualScale（不可预折叠，见 sm80 文件头注释）
#pragma unroll
            for (int i = 0; i < size(tYr_res_tmp); ++i) {
              tYr(i) += tYr_res_tmp(i) * params.scale;
            }
          }
        }
        cute::gemm(tiled_mma, tWHr(_, _, ik), tXr(_, _, ik), tYr);
      }

      empty_bar[stage].arrive();
      ++consumer_slab;

      // producer lookahead：发射 kStage 步之后的 slab（仍在同一 tile 内）
      if (tidx == 0) {
        int itile2 = itile + kStage;
        if (itile2 < ntile_k_local) {
          acquire_and_issue(ichunk + itile2 * kSplitK);
        }
      }
    }

    if constexpr (kIsInt8) {
      // INT8 residual 合并：Y_low = scale * s_x[row] * s_w[col] * acc_int32
      int m0 = itile_m * kTileM;
      int n0 = itile_n * kTileN;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        auto coord = cCoord(i);
        int row = m0 + get<1>(coord);
        int col = n0 + get<0>(coord);
        float sx = (row < params.m) ? params.x_scale_ptr[row] : 0.0f;
        float sw = (col < params.n) ? params.w_scale_ptr[col] : 0.0f;
        tYr(i) += static_cast<float>(tYr_res_tmp(i)) * (params.scale * sx * sw);
      }
    }

    // sY 与 operand smem 别名复用：先确认所有线程读完本 tile 的最后一个 stage
    __syncthreads();

    auto tiled_copy_y = make_tiled_copy_C(Copy_Atom<UniversalCopy<TY>, TY>{}, tiled_mma);
    auto thr_copy_y = tiled_copy_y.get_slice(tidx);

    if constexpr (kSplitK > 1) {
      // partial：fp32 原值（无 bias/act），elementwise 写 splitk_y_ptr
      cute::copy(tiled_copy_y, thr_copy_y.retile_S(tYr), thr_copy_y.partition_D(sY));
      __syncthreads();
      float *y_dst = params.splitk_y_ptr + static_cast<int64_t>(ichunk) * params.m * params.n;
      Tensor gYY = make_tensor(make_gmem_ptr(y_dst), make_shape(params.n, params.m),
                               make_stride(Int<1>{}, params.n));
      Tensor gY = local_tile(gYY, make_tile(Int<kTileN>{}, Int<kTileM>{}),
                             make_coord(itile_n, itile_m));
#pragma unroll 1
      for (int i = tidx; i < kTileM * kTileN; i += kNThreads) {
        int row = i / kTileN;
        int col = i % kTileN;
        if (itile_m * kTileM + row < params.m && itile_n * kTileN + col < params.n) {
          gY(col, row) = sY(col, row);
        }
      }
      __syncthreads();  // partial 写回可见后再标记完成
      if (tidx == 0) {
        if (last_tile_m != -1 && last_tile_n != -1) {
          atomicAdd(params.split_flag_ptr + last_tile_m * num_tile_n + last_tile_n, 1);
        }
        last_tile_m = itile_m;
        last_tile_n = itile_n;
      }
    } else {
      // bias+act+降精度在寄存器内完成（TMA store 与 elementwise 两路共用）。
      // ragged N 时部分 lane 的列坐标越界：bias 读取按 min 钳到合法列
      // （越界 lane 的输出会在 store 侧被裁剪/谓词掉，钳位只是消除 OOB 读）。
      Tensor tYc = make_tensor<TY>(shape(tYr));
      int n_tile_start = itile_n * kTileN;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        float value = tYr(i);
        if constexpr (Traits::HasBias) {
          auto coord = cCoord(i);
          int col = n_tile_start + get<0>(coord);
          value += params.bias_ptr[min(col, params.n - 1)];
        }
        tYc(i) = static_cast<TY>(Activation{}(value));
      }
      cute::copy(tiled_copy_y, thr_copy_y.retile_S(tYc), thr_copy_y.partition_D(sY));

      if (params.has_tma_store) {
        // TMA store：swizzled sY → gmem（自动裁剪越界行列），异步批量写
        cutlass::arch::fence_view_async_shared();
        __syncthreads();
        if (tidx == 0) {
          Tensor mY = params.tma_store_Y.get_tma_tensor(make_shape(params.n, params.m));
          Tensor gY = local_tile(mY, Shape<Int<kTileN>, Int<kTileM>>{},
                                 make_coord(itile_n, itile_m));
          auto block_tma_Y = params.tma_store_Y.get_slice(_0{});
          Tensor tOsY = group_modes<0, 3>(block_tma_Y.partition_S(sY));
          Tensor tOgY = group_modes<0, 3>(block_tma_Y.partition_D(gY));
          cute::copy(params.tma_store_Y, tOsY, tOgY);
          cute::tma_store_arrive();
          cute::tma_store_wait<0>();  // sY 即将被下一 tile 的 TMA load 覆写
        }
      } else {
        // 对齐不满足时的 elementwise 回退（bounds-checked）
        __syncthreads();
        Tensor gYY = make_tensor(make_gmem_ptr(params.y_ptr), make_shape(params.n, params.m),
                                 make_stride(Int<1>{}, params.n));
        Tensor gY = local_tile(gYY, make_tile(Int<kTileN>{}, Int<kTileM>{}),
                               make_coord(itile_n, itile_m));
#pragma unroll 1
        for (int i = tidx; i < kTileM * kTileN; i += kNThreads) {
          int row = i / kTileN;
          int col = i % kTileN;
          if (itile_m * kTileM + row < params.m && itile_n * kTileN + col < params.n) {
            gY(col, row) = sY(col, row);
          }
        }
      }
    }
    // tile 边界：sY 读取 / TMA store 完成、全体线程到齐后，下一 tile 的
    // prologue 才能覆写 operand smem（thread0 在本 sync 之后才发 TMA load）。
    // fence_view_async_shared（fence.proxy.async.shared::cta）：本 tile 对 sY
    // （别名自 operand smem）的 generic 写，与下一 tile TMA load 的 async
    // proxy 写之间按 PTX 内存模型需要代理栅栏（cp.async 无此要求，TMA 有）。
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }

  // ── split-K 终归约（与 sm80 路径逐字一致）───────────────────────────────
  if constexpr (kSplitK > 1) {
    __threadfence();
    __syncthreads();

    if (tidx == 0 && last_tile_m != -1 && last_tile_n != -1) {
      atomicAdd(params.split_flag_ptr + last_tile_m * num_tile_n + last_tile_n, 1);
    }
    __syncthreads();

    iblock = blockIdx.x;
    __threadfence();
    using NVTout = std::conditional_t<std::is_same_v<typename Traits::Tout, float>, float, __nv_bfloat16>;
    while (true) {
      int itile_m, itile_n;
      params.reduce_flat_divider(itile_m, itile_n, iblock);
      if (itile_m >= num_tile_m) break;
      iblock += gridDim.x;

      auto *split_flag = params.split_flag_ptr + itile_m * num_tile_n + itile_n;
      while (load_global_volatile(split_flag) != kSplitK) {
      }
      splitk_reduce<NVTout, Activation, Traits::HasBias, kTileM, kTileN, kSplitK,
                    kNThreads / 32>(reinterpret_cast<NVTout *>(params.y_ptr),
                                    params.splitk_y_ptr, params.bias_ptr, params.m, params.n,
                                    itile_m, itile_n);
      __syncthreads();
      if (tidx == 0) {
        *split_flag = 0;
      }
    }
  }
#else
  // 非 sm120a 编译目标：空 kernel（host 侧 mixed_gemm_sm120_supported() 为
  // false 时不会分发到这里）
  (void)params;
#endif
}

}  // namespace kernels_sm120

// ── host 侧 launcher ─────────────────────────────────────────────────────────
template <typename Traits>
void launch_gemm_bf16xfp32_kernel_sm120(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
                                         const void *x_ptr, const void *w_high_ptr,
                                         const void *w_low_ptr, const float *bias_ptr,
                                         void *x_res_ptr, int m, int n, int k, float scale,
                                         int sm_count, cudaStream_t stream,
                                         const float *w_scale_ptr = nullptr,
                                         float *x_scale_ptr = nullptr) {
  using namespace cute;  // NOLINT
  using bf16 = cute::bfloat16_t;
  using ResElem = typename Traits::ResElem;
  using Tout = typename Traits::Tout;

  // gmem 张量视图（类型须与 traits 中 decltype 完全一致）
  Tensor mX = make_tensor(make_gmem_ptr(reinterpret_cast<bf16 const *>(x_ptr)),
                          make_shape(m, k), make_stride(static_cast<int64_t>(k), _1{}));
  Tensor mW = make_tensor(make_gmem_ptr(reinterpret_cast<bf16 const *>(w_high_ptr)),
                          make_shape(n, k), make_stride(static_cast<int64_t>(k), _1{}));
  Tensor mX8 = make_tensor(make_gmem_ptr(reinterpret_cast<ResElem const *>(x_res_ptr)),
                           make_shape(m, k), make_stride(static_cast<int64_t>(k), _1{}));
  Tensor mW8 = make_tensor(make_gmem_ptr(reinterpret_cast<ResElem const *>(w_low_ptr)),
                           make_shape(n, k), make_stride(static_cast<int64_t>(k), _1{}));

  typename kernels_sm120::GemmSm120Params<Traits> params{};
  params.tma_load_X = make_tma_copy(SM90_TMA_LOAD{}, mX, typename Traits::SmemLayoutXStage{});
  params.tma_load_W = make_tma_copy(SM90_TMA_LOAD{}, mW, typename Traits::SmemLayoutWStage{});
  params.tma_load_X8 = make_tma_copy(SM90_TMA_LOAD{}, mX8, typename Traits::SmemLayoutX8Stage{});
  params.tma_load_W8 = make_tma_copy(SM90_TMA_LOAD{}, mW8, typename Traits::SmemLayoutW8Stage{});

  // TMA store 的额外对齐条件（仅 splitk==1 路径使用；不满足则 epilogue 降级
  // elementwise，其余路径不变）
  params.has_tma_store = false;
  if constexpr (Traits::kSplitK == 1) {
    bool y_ok = (reinterpret_cast<uintptr_t>(y_ptr) % 16 == 0) &&
                (static_cast<int64_t>(n) * static_cast<int64_t>(sizeof(Tout))) % 16 == 0;
    if (std::getenv("GEMM_MIXED_NO_TMA_STORE") != nullptr) y_ok = false;
    if (y_ok) {
      Tensor mY = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                              make_shape(n, m), make_stride(_1{}, static_cast<int64_t>(n)));
      params.tma_store_Y =
          make_tma_copy(SM90_TMA_STORE{}, mY, typename Traits::SmemLayoutYStore{});
      params.has_tma_store = true;
    }
  }

  params.bias_ptr = bias_ptr;
  params.x_scale_ptr = x_scale_ptr;
  params.w_scale_ptr = w_scale_ptr;
  params.y_ptr = reinterpret_cast<Tout *>(y_ptr);
  params.splitk_y_ptr = reinterpret_cast<float *>(splitk_y_ptr);
  params.split_flag_ptr = reinterpret_cast<int *>(split_flag_ptr);
  params.m = m;
  params.n = n;
  params.k = k;
  params.scale = scale;

  // 与 sm80 launcher 相同的 tile/divider 记账（num_tile_n 含 kSplitK 因子）
  int num_tile_m = (m + Traits::kTileM - 1) / Traits::kTileM;
  int num_tile_n = (n + Traits::kTileN - 1) / Traits::kTileN * Traits::kSplitK;
  int num_tile = num_tile_m * num_tile_n;
  params.swizzle_divider = cutlass::FastDivmod(Traits::kBlockSwizzle * num_tile_n);
  params.flat_divider = cutlass::FastDivmod(num_tile_n);
  params.reduce_flat_divider = cutlass::FastDivmod(num_tile_n / Traits::kSplitK);

  auto kernel = &kernels_sm120::gemm_bf16xfp32_kernel_sm120<Traits>;
  // kernel 属性/occupancy 仅依赖编译期配置与目标设备，缓存之（与 sm80 相同）
  static std::once_flag metadata_once;
  static int cached_max_active_blocks_per_sm = 1;
  std::call_once(metadata_once, [&]() {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cached_max_active_blocks_per_sm, kernel,
                                                  Traits::kNThreads, Traits::kSmemSize);
    if (cached_max_active_blocks_per_sm < 1) cached_max_active_blocks_per_sm = 1;
  });

  dim3 block(Traits::kNThreads);
  dim3 grid(std::min(sm_count * cached_max_active_blocks_per_sm, num_tile));
  if (std::getenv("GEMM_MIXED_DEBUG_GRID")) {
    fprintf(stderr,
            "[mixed_gemm sm120 debug] kTileM=%d kTileN=%d kStage=%d kSplitK=%d shm=%d "
            "max_blocks/sm=%d num_tile=%d grid=%d tma_store=%d\n",
            Traits::kTileM, Traits::kTileN, Traits::kStage, Traits::kSplitK, Traits::kSmemSize,
            cached_max_active_blocks_per_sm, num_tile, grid.x,
            static_cast<int>(params.has_tma_store));
  }

  kernel<<<grid, block, Traits::kSmemSize, stream>>>(params);
}

// ── tile/stage 配置选择 + sm120 dispatch ────────────────────────────────────
//
// tile 启发式沿用 sm80 路径的既有结论（select_kTileM / select_kTileN，见
// gemm_bf16xfp32_sm80.h）：m<=64 → 64x128；其余按 wave 利用率在 128x128 与
// 64x64 间二选一。差异仅在 kStage：sm120 与 sm89 一样只有 ~100KB/CTA，但
// TMA 路径没有谓词寄存器开销、且 64x64 的 smem 足够深，因此
//   64x128 → kStage=2（72KB）    128x128 → kStage=2（96KB）
//   64x64  → kStage=4（96KB）
template <int kTileM_, int kTileN_, int kStage_, int kSplitK_>
struct LaunchCfgSm120 {
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kStage = kStage_;
  static constexpr int kSplitK = kSplitK_;
};

template <typename T>
struct OutTypeTag {
  using type = T;
};

template <typename Activation, bool HasBias, typename ResidualBackend>
bool dispatch_gemm_bf16xfp32_sm120(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
                                    const void *x_ptr, const void *w_high_ptr,
                                    const void *w_low_ptr, const float *bias_ptr,
                                    void *x_res_ptr, int m, int n, int k, float scale,
                                    bool use_fp32_output, int splitk, int sm_count,
                                    cudaStream_t stream,
                                    const float *w_scale_ptr = nullptr,
                                    float *x_scale_ptr = nullptr) {
  auto launch = [&](auto cfg_tag, auto out_tag) {
    using Cfg = decltype(cfg_tag);
    using OutT = typename decltype(out_tag)::type;
    using Traits = kernels_sm120::GemmSm120Traits<Cfg::kTileM, Cfg::kTileN, Cfg::kStage,
                                                  Cfg::kSplitK, ResidualBackend, OutT,
                                                  Activation, HasBias>;
    launch_gemm_bf16xfp32_kernel_sm120<Traits>(
        y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_ptr, bias_ptr, x_res_ptr,
        m, n, k, scale, sm_count, stream, w_scale_ptr, x_scale_ptr);
  };

  auto launch_out = [&](auto cfg_tag) {
    if (use_fp32_output) {
      launch(cfg_tag, OutTypeTag<float>());
    } else {
      launch(cfg_tag, OutTypeTag<cute::bfloat16_t>());
    }
  };

  if (m <= 64) {
    switch (splitk) {
      case 16: launch_out(LaunchCfgSm120<64, 128, 2, 16>{}); return true;
      case 8:  launch_out(LaunchCfgSm120<64, 128, 2, 8>{});  return true;
      case 4:  launch_out(LaunchCfgSm120<64, 128, 2, 4>{});  return true;
      case 2:  launch_out(LaunchCfgSm120<64, 128, 2, 2>{});  return true;
      case 1:  launch_out(LaunchCfgSm120<64, 128, 2, 1>{});  return true;
      default: return false;
    }
  }

  int kTileN_sel = select_kTileN(m, n, sm_count);
  if (kTileN_sel == 64) {
    switch (splitk) {
      case 16: launch_out(LaunchCfgSm120<64, 64, 4, 16>{}); return true;
      case 8:  launch_out(LaunchCfgSm120<64, 64, 4, 8>{});  return true;
      case 4:  launch_out(LaunchCfgSm120<64, 64, 4, 4>{});  return true;
      case 2:  launch_out(LaunchCfgSm120<64, 64, 4, 2>{});  return true;
      case 1:  launch_out(LaunchCfgSm120<64, 64, 4, 1>{});  return true;
      default: return false;
    }
  }

  switch (splitk) {
    case 16: launch_out(LaunchCfgSm120<128, 128, 2, 16>{}); return true;
    case 8:  launch_out(LaunchCfgSm120<128, 128, 2, 8>{});  return true;
    case 4:  launch_out(LaunchCfgSm120<128, 128, 2, 4>{});  return true;
    case 2:  launch_out(LaunchCfgSm120<128, 128, 2, 2>{});  return true;
    case 1:  launch_out(LaunchCfgSm120<128, 128, 2, 1>{});  return true;
    default: return false;
  }
}

// ── 固定 epilogue launcher（对应 sm80 的 launch_fixed_epilogue* 家族）───────
#if MIXED_GEMM_FP8_ENABLED
template <typename Activation, bool HasBias>
bool launch_fixed_epilogue_sm120_fp8(void *y_ptr, void *splitk_y_ptr,
                                     void *split_flag_ptr, const void *x_ptr,
                                     const void *w_high_ptr, const void *w_low_fp8_ptr,
                                     const float *bias_ptr, void *x_fp8_ptr,
                                     int m, int n, int k, float scale,
                                     bool use_fp32_output, int splitk, int sm_count,
                                     cudaStream_t stream) {
  return dispatch_gemm_bf16xfp32_sm120<Activation, HasBias, kernels_sm120::Fp8ResidualBackend>(
      y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_fp8_ptr, bias_ptr, x_fp8_ptr,
      m, n, k, scale, use_fp32_output, splitk, sm_count, stream);
}
#endif  // MIXED_GEMM_FP8_ENABLED

template <typename Activation, bool HasBias>
bool launch_fixed_epilogue_sm120_int8(void *y_ptr, void *splitk_y_ptr,
                                      void *split_flag_ptr, const void *x_ptr,
                                      const void *w_high_ptr, const void *w_low_int8_ptr,
                                      const float *w_scale_ptr, const float *bias_ptr,
                                      void *x_int8_ptr, float *x_scale_ptr,
                                      int m, int n, int k, float scale,
                                      bool use_fp32_output, int splitk, int sm_count,
                                      cudaStream_t stream) {
  return dispatch_gemm_bf16xfp32_sm120<Activation, HasBias, kernels_sm120::Int8ResidualBackend>(
      y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_int8_ptr, bias_ptr,
      x_int8_ptr, m, n, k, scale, use_fp32_output, splitk, sm_count, stream, w_scale_ptr,
      x_scale_ptr);
}

}  // namespace mixed_gemm

// ── 对外路由入口（声明见 gemm_bf16xfp32_sm120.h）─────────────────────────────
namespace mixed_gemm {

// sm120 kernel 是否编入本编译单元（编译目标含 sm120a 家族；判定宏来自
// csrc/arch_targets.h 标准入口，与 FA 的分发共用同一套 FA_HAS_* 语义）。
bool mixed_gemm_sm120_compiled() noexcept {
#if FA_HAS_SM120
  return true;
#else
  return false;
#endif
}

// 设备是否可走 sm120 路径（major==12 + kernel 已编入 + env 强制回退开关）。
bool mixed_gemm_sm120_supported() noexcept {
  static const bool supported = []() {
    if (std::getenv("GEMM_MIXED_FORCE_SM80") != nullptr) return false;
#if FA_HAS_SM120
    return arch_targets::gpu_major() == 12;
#else
    return false;
#endif
  }();
  return supported;
}

// sm120 路径的额外数据面约束（详见 gemm_bf16xfp32_sm120.h 的注释）：
// TMA 全局行 stride（k 字节）须 16B 对齐 + 指针 16B 对齐。
bool mixed_gemm_sm120_ptrs_ok(const void *x_ptr, const void *w_high_ptr, const void *w_low_ptr,
                              const void *x_res_ptr, int64_t k) noexcept {
  return (k % 16 == 0) &&
         (reinterpret_cast<uintptr_t>(x_ptr) % 16 == 0) &&
         (reinterpret_cast<uintptr_t>(w_high_ptr) % 16 == 0) &&
         (reinterpret_cast<uintptr_t>(w_low_ptr) % 16 == 0) &&
         (reinterpret_cast<uintptr_t>(x_res_ptr) % 16 == 0);
}

#if MIXED_GEMM_FP8_ENABLED
GemmFixedEpilogueLauncher resolve_gemm_bf16xfp32_epilogue_launcher_sm120(
    int activation_type, bool has_bias) noexcept {
  switch (activation_type) {
    case 0:
      return has_bias ? &launch_fixed_epilogue_sm120_fp8<IdentityActivation, true>
                      : &launch_fixed_epilogue_sm120_fp8<IdentityActivation, false>;
    case 1:
      return has_bias ? &launch_fixed_epilogue_sm120_fp8<SiluActivation, true>
                      : &launch_fixed_epilogue_sm120_fp8<SiluActivation, false>;
    case 2:
      return has_bias ? &launch_fixed_epilogue_sm120_fp8<GeluActivation, true>
                      : &launch_fixed_epilogue_sm120_fp8<GeluActivation, false>;
    default:
      return nullptr;
  }
}
#endif  // MIXED_GEMM_FP8_ENABLED

GemmFixedInt8EpilogueLauncher resolve_gemm_bf16xfp32_int8_launcher_sm120(
    int activation_type, bool has_bias) noexcept {
  switch (activation_type) {
    case 0:
      return has_bias ? &launch_fixed_epilogue_sm120_int8<IdentityActivation, true>
                      : &launch_fixed_epilogue_sm120_int8<IdentityActivation, false>;
    case 1:
      return has_bias ? &launch_fixed_epilogue_sm120_int8<SiluActivation, true>
                      : &launch_fixed_epilogue_sm120_int8<SiluActivation, false>;
    case 2:
      return has_bias ? &launch_fixed_epilogue_sm120_int8<GeluActivation, true>
                      : &launch_fixed_epilogue_sm120_int8<GeluActivation, false>;
    default:
      return nullptr;
  }
}

}  // namespace mixed_gemm