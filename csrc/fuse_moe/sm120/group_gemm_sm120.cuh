// group_gemm_sm120.cuh — MoE group GEMM kernel，sm120a TMA + mbarrier 路径。
//
// 架构分层（仿 csrc/fa）：sm120 家族（Blackwell consumer，RTX 50）。
// cp.async 家族见 sm89/group_gemm_sm89.cuh（sm120 默认引擎）；架构分发见
// fuse_moe_launch.h。traits 共享见 common/group_gemm_config.cuh（本文件
// TmaGemmConfig 继承 MoEGemmConfig 扩展）。
//
// 与 sm89 家族（cp.async，sm80+ 通用）的关系：数值语义/输出布局完全
// 一致（swapAB MMA、紧凑 Y、行谓词 epilogue），仅数据通路不同：
//   cp.async : 256 线程各自发射 16B 拷贝 + ZFILL 谓词 + cp_async_wait/__syncthreads
//   TMA      : thread0 单线程发射 bulk 拷贝 + mbarrier full/empty 流水
//              （mixed_gemm sm120 路径同款 cooperative 模式；sm120 无
//               wgmma，256 线程全部参与计算）
//
// A（激活）走 TMA 的方案（sm_120a 不支持 device 侧 tensormap 修改指令
// fence.proxy.tensormap / tensormap.replace——ptxas 接受语法但硬件
// illegal instruction，hpc sm90 的 per-expert desc 路线不可用；也不需要：
// TMA 坐标本身是元素单位，box 起点可以是任意行）：
//   ⚠️ 前提：A 必须是 expert 有序 compact 布局（每 group 行连续）——
//   本 op 中仅 gemm2 的 A=act_out 满足（流水线自己写出的中间量）或 gemm1
//   经 gather_sorted 预排序后的 x_sorted；x 原始 token 序（group 行按
//   row_indices 散列）无法被 TMA 矩形 box 间接寻址。
//   - A 用一个覆盖 compact 布局 (a_rows, k) 的全局 descriptor（host 构建，
//     按值 __grid_constant__ 传入）
//   - 每 group 把坐标基迭代器平移到 compact 起点 start_token（任意行）：
//     tAg_g = make_tensor(tAg.data() + make_coord(_0{}, start_token),
//     tAg.layout())——机制与 cute::tma_partition 内部 multicast 的
//     domain_offset 同源，坐标 = (itile_k*kTileK, start_token+itile_m*kTileM)
//   - gmem 地址恒 128B 对齐：box 起点行 r0 的地址 = base + r0*k*2B，
//     k%64==0 → 行距 ≥128B；无任何对齐新增约束
//   - 尾 tile 越界语义：组尾读到下一组真实行（in-bounds，mma 垃圾行被
//     epilogue 行谓词丢弃）；末组尾 tile 越过 a_rows → TMA OOB 补零；
//     每行独立计算，NaN 不跨行传播
//   - 输出 Y 保持紧凑布局（epilogue 用 cu_seqlens[igroup] 作紧凑起点），
//     act/reduce 数值路径与 cp.async 版完全一致
//   - B（权重）静态 (E, n, k) 3D，单 descriptor coord mode2=igroup 切片
//
// 任务调度：horizon 扫描（itile_n 内层，移植自 hpc common.cuh）——
// expert 定位线性扫描（跨 task 递增状态，摊销 O(1)），同 (g,m) 的连续
// n-task 共享 A 面板。
//
// smem 排布：[sX][sW]（kStage 级）| [scratch: tiles(E)] |
// [full/empty mbarriers]（8B 对齐）；sC 别名基地址。相位记账跨 tile 连续
//（producer_slab/consumer_slab，mbarrier 状态不复位——mixed_gemm sm120
// 模式）；producer lookahead 仅在 tile 内，边界处流水线自然排空 +
// __syncthreads 保护 sC 别名区域。
//
// TMA 布局约束：SW128 swizzle atom 为 (8, 64)，kTileK=128 时 K 模式被
// tile_to_shape 分解为嵌套 (64, 2)——TMA 坐标同样分解为嵌套子模式，
// 直接 partition raw tensor 时的单模式索引会 rank-mismatch。解法（FA
// sm120 同款模式）：先 local_tile 展开自由 tile 模式（前导模式恰为
// Tiler 形状）再 partition_S/D，最后 group_modes<0,3> 收敛 3 个 TMA 坐标
// 子模式；k%128==0 时选 kTileK=128（每 slab mma 工作量翻倍、barrier/
// 发射开销减半），否则 64。模式顺序约定：local_tile 把未 tile 的中间
// 模式（B 的 E）排到自由 tile 模式之后，故 tBg=(TMA,nn,nk,E)、
// tAg=(TMA,nm,nk)、tAs/tBs=(TMA,stage)。
// kStage 按 96KB smem 预算自适应（occupancy 寄存器限制恒 1 CTA/SM，
// smem 用满零代价）：M64/K64→6、M64/K128→3、M32/K64→6、M32/K128→4。
//
// 前置条件（op 层保证）：n % 64 == 0，k % 64 == 0；A 为 expert 有序
// compact 布局。非 sm90+/sm120a 编译目标下 kernel 体为空
//（CUTE_ARCH_TMA_SM90_ENABLED 未定义），host 侧不分发。

#ifndef FUSE_MOE_SRC_SM120_GROUP_GEMM_SM120_CUH_
#define FUSE_MOE_SRC_SM120_GROUP_GEMM_SM120_CUH_

#include <cuda.h>

#include <cassert>
#include <cstdlib>

#include "cutlass/arch/barrier.h"

#include "common/fuse_moe_utils.cuh"
#include "common/group_gemm_config.cuh"  // MoEGemmConfig（traits 共用）+ config::TypeTag

namespace fuse_moe {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace tma {

// ─── TMA 专用 traits ─────────────────────────────────────────────────────────
// cp.async 路径的 MoEGemmConfig 用元素域 Swizzle<3,3,3>∘(8,64)（ldmatrix
// 友好）；make_tma_copy 只接受 GMMA 规范 atom（字节域 Swizzle<3,4,3>，即
// SW128 模式）——两者在 16-bit 元素下数值等价（FA sm120 路径已验证），
// 故 ldmatrix（s2r）读取路径与 cp.async 版完全一致，仅 smem 布局表达更换。
template <typename BaseConfig, int kStage_>
struct TmaGemmConfig : public BaseConfig {
  using Tin = typename BaseConfig::Tin;
  using Tout = typename BaseConfig::Tout;
  using TiledMMA = typename BaseConfig::TiledMMA;

  static constexpr int kTileM = BaseConfig::kTileM;
  static constexpr int kTileN = BaseConfig::kTileN;
  static constexpr int kTileK = BaseConfig::kTileK;
  static constexpr int kStage = kStage_;
  static constexpr int kNThreads = BaseConfig::kNThreads;

  using SmemLayoutAtomT = GMMA::Layout_K_SW128_Atom<Tin>;  // (8, 64) 字节域 swizzle
  using SmemLayoutX = decltype(tile_to_shape(
      SmemLayoutAtomT{}, Shape<Int<kTileM>, Int<kTileK>, Int<kStage_>>{}));
  using SmemLayoutW = decltype(tile_to_shape(
      SmemLayoutAtomT{}, Shape<Int<kTileN>, Int<kTileK>, Int<kStage_>>{}));
  using SmemLayoutC = typename BaseConfig::SmemLayoutC;  // 无 swizzle，直接沿用

  static constexpr int shm_xw =
      static_cast<int>(sizeof(Tin) * (cosize(SmemLayoutX{}) + cosize(SmemLayoutW{})));
};

// ─── device 调度（TaskLoopPolicy 1/2，移植自 hpc kernels.cuh）────────────────

// horizon：itile_n 内层 + expert 线性扫描（状态递增）。耗尽返回 igroup=-1。
__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_m,
                                                      int &itile_n, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_m, itile_m_total;
  flat_divider(itile_m_total, itile_n, iblock);
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_m_total < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_m_total - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

// ─── 核心 kernel ─────────────────────────────────────────────────────────────
// TmaA/TmaB：launcher 侧 decltype(make_tma_copy(...)) 推得；TmaB 为静态
// (n, k, E)。任务调度恒 horizon（itile_n 内层：同 (g,m) 的连续 n-task
// 共享 A 面板，L2 命中友好）。
template <typename Config, typename TmaA, typename TmaB, bool kUsePDL>
__global__ void __launch_bounds__(Config::kNThreads, 1)
group_gemm_tma_kernel(const __grid_constant__ TmaB tma_b, const __grid_constant__ TmaA tma_a,
                      void *Cptr, const int *seqlens_ptr, const int *cu_seqlens_ptr,
                      const int *tiles_ptr,
                      int a_rows, int n, int k, int num_group, cutlass::FastDivmod flat_divider) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  using namespace cute;  // NOLINT
  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  constexpr int kNThreads = Config::kNThreads;
  constexpr int kNumSubK = kTileK / 16;

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;

  // smem: [sX | sW]（kStage 级 operand）| [scratch: tiles(E)] |
  // [full | empty barriers]（8B 对齐）；sC 别名基地址
  extern __shared__ uint8_t shm_data[] alignas(128);
  Tin *xshm = reinterpret_cast<Tin *>(shm_data);
  Tin *wshm = xshm + cosize(typename Config::SmemLayoutX{});
  int *shm_scratch = reinterpret_cast<int *>(wshm + cosize(typename Config::SmemLayoutW{}));
  // mbarrier 需 8B 对齐：scratch 后 (E+1)*4 偏移可能仅 4 对齐，向上取整
  FullBarrier *full_bar = reinterpret_cast<FullBarrier *>(
      (reinterpret_cast<uintptr_t>(shm_scratch + num_group + 1) + 7) & ~uintptr_t(7));
  EmptyBarrier *empty_bar = reinterpret_cast<EmptyBarrier *>(full_bar + kStage);
  Tout *cshm = reinterpret_cast<Tout *>(shm_data);

  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  // 调试写入统一门控（常量开关，关闭时分支并重排丢弃，零热路径开销）

  // PDL acquire：等上游（count/gather/desc 构建）写完调度数组与 desc
  pdl_acquire();

  if (idx < kStage) {
    full_bar[idx].init(1);           // 仅 thread0 arrive_and_expect_tx
    empty_bar[idx].init(kNThreads);  // 全体计算线程 arrive
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  // 调度 scratch（tiles）载入 smem（task 迭代间复用）
  for (int i = idx; i < num_group; i += kNThreads) {
    shm_scratch[i] = tiles_ptr[i];
  }
  __syncthreads();


  auto sX = make_tensor(make_smem_ptr(xshm), typename Config::SmemLayoutX{});  // (M, K, S)
  auto sW = make_tensor(make_smem_ptr(wshm), typename Config::SmemLayoutW{});  // (N, K, S)
  auto sC = make_tensor(make_smem_ptr(cshm), typename Config::SmemLayoutC{});  // (N, M)

  // ── MMA / s2r 句柄（tile 循环外构造一次；swapAB：A=W, B=X, C=(N, M)）
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tWr = thr_mma.partition_fragment_A(sW(_, _, _0{}));
  auto tXr = thr_mma.partition_fragment_B(sX(_, _, _0{}));
  auto tYr = thr_mma.partition_fragment_C(sC);

  using LDSM_ATOM = Copy_Atom<SM75_U32x4_LDSM_N, Tin>;
  auto s2r_copy_w = make_tiled_copy_A(LDSM_ATOM{}, tiled_mma);
  auto s2r_copy_x = make_tiled_copy_B(LDSM_ATOM{}, tiled_mma);
  auto s2r_thr_copy_w = s2r_copy_w.get_slice(idx);
  auto s2r_thr_copy_x = s2r_copy_x.get_slice(idx);
  auto tWs4r = s2r_thr_copy_w.partition_S(sW);
  auto tXs4r = s2r_thr_copy_x.partition_S(sX);
  auto tXr_view = s2r_thr_copy_x.retile_D(tXr);
  auto tWr_view = s2r_thr_copy_w.retile_D(tWr);

  auto s2r_copy_ik = [&](int ik, int stage) {
    cute::copy(s2r_copy_x, tXs4r(_, _, ik, stage), tXr_view(_, _, ik));
    cute::copy(s2r_copy_w, tWs4r(_, _, ik, stage), tWr_view(_, _, ik));
  };

  // ── TMA 分区（FA sm120 同款模式，K=64/128 双 case 通用）：
  //    1) 先 local_tile 把 tile 模式展开为自由模式（前导模式恰为 Tiler
  //       形状 (M, K)/(N, K)——这是 partition 产出稳定 3 个 TMA 坐标子模式
  //       的前提；直接 partition raw tensor 时 V 模式数随 K 宽度变化，
  //       rank 算术分组不可靠，K=128 下 rank-mismatch）；
  //    2) group_modes<0,3> 把 3 个 TMA 坐标子模式收敛为单模式。
  //    ⚠️ 模式顺序约定：local_tile 把未 tile 的中间模式
  //    （B 的 E）排到自由 tile 模式之后：gB=(N,K,nn,nk,E) → tBg=(TMA,nn,nk,E)；
  //    A 无中间模式：tAg=(TMA,nm,nk)，tAs/tBs=(TMA,stage)。
  //    A 坐标基在 acquire_and_issue 内按 group 平移（compact 任意行起点，
  //    见文件头）；B 坐标 = (itile_n, itile_k, igroup)（3D (n,k,E) desc）。
  Tensor gA = local_tile(tma_a.get_tma_tensor(make_shape(a_rows, k)),
                         Shape<Int<kTileM>, Int<kTileK>>{}, make_coord(_, _));
  Tensor gB = local_tile(tma_b.get_tma_tensor(make_shape(n, k, num_group)),
                         Shape<Int<kTileN>, Int<kTileK>>{}, make_coord(_, _));
  auto btma_a = tma_a.get_slice(_0{});
  auto btma_b = tma_b.get_slice(_0{});
  Tensor tAg = group_modes<0, 3>(btma_a.partition_S(gA));  // (TMA, nm, nk)
  Tensor tAs = group_modes<0, 3>(btma_a.partition_D(sX));  // (TMA, stage)
  Tensor tBg = group_modes<0, 3>(btma_b.partition_S(gB));  // (TMA, nn, nk, E)
  Tensor tBs = group_modes<0, 3>(btma_b.partition_D(sW));  // (TMA, stage)

  const int num_tile_n = size<1>(tBg);  // nn = n / kTileN
  const int ntile_k = size<2>(tAg);    // nk = k / kTileK
  // TMA 事务字节：A+B 完整 box（OOB 补零部分同样计入事务）
  constexpr uint32_t kTransactionBytes =
      static_cast<uint32_t>(sizeof(Tin) * (kTileM + kTileN) * kTileK);

  // ── r2s / s2g 句柄（epilogue，sC 无 swizzle）
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<Tout>, Tout>;
  auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto thr_copy_c = tiled_copy_c.get_slice(idx);
  auto tCs4g = thr_copy_c.partition_D(sC);

  // ── 流水线发射（仅 thread0）：复用 stage 前等 empty；A 为 compact 布局
  //    （每 group 坐标基平移到任意行起点，见文件头）+ B 静态 3D desc
  //    + EVICT_LAST（B 跨 M-tile 复用）。producer_slab/consumer_slab 均
  //    在此声明（跨任务连续，见下方任务循环注释）。
  int producer_slab = 0;
  int consumer_slab = 0;
  auto acquire_and_issue = [&](int itile_m, int itile_n, int itile_k, int a_row_base,
                               int igroup) {
    int stage = producer_slab % kStage;
    if (producer_slab >= kStage) {
      uint32_t ephase = ((producer_slab / kStage) - 1) & 1;
      empty_bar[stage].wait(ephase);
    }
    uint64_t &bar = reinterpret_cast<uint64_t &>(full_bar[stage]);
    full_bar[stage].arrive_and_expect_tx(kTransactionBytes);
    // 坐标基平移：elem0=k 维、elem1=行维，box 起点行 =
    // a_row_base + itile_m*kTileM（任意行，非 tile 对齐）。纯迭代器
    // 算术，thread0 每 slab 一次，开销可忽略。
    Tensor tAg_g = make_tensor(tAg.data() + make_coord(_0{}, a_row_base), tAg.layout());
    cute::copy(tma_a.with(bar), tAg_g(_, itile_m, itile_k), tAs(_, stage));
    cute::copy(tma_b.with(bar, 0, TMA::CacheHintSm90::EVICT_LAST),
               tBg(_, itile_n, itile_k, igroup), tBs(_, stage));
    ++producer_slab;
  };

  // ── 任务循环（horizon 调度）
  // producer_slab/consumer_slab 均为跨任务连续的全局计数（mixed_gemm sm120
  // 同款）：stage = cnt % kStage，phase = (cnt/kStage)&1，两计数器必须同基准
  // ——若 consumer 在任务内重置而 producer 全局连续，相位错位会导致第二个
  // 任务在 full_bar.wait 上等一个已翻转过的 parity（永久死锁）。
  int igroup = 0;
  int itile_m, itile_n;
  int sum_tile_m = 0;
  while (true) {
    get_next_tile_horizon(shm_scratch, iblock, num_group, igroup, itile_m, itile_n,
                          sum_tile_m, flat_divider);
    if (igroup < 0) break;
    const int start_token = cu_seqlens_ptr[igroup];          // 紧凑区间起点（Y 输出）
    const int m = seqlens_ptr[igroup];                       // 本组 token 数
    iblock += gridDim.x;

    // prologue：thread0 预发射 min(kStage, ntile_k) 个 K-slab
    //（A 坐标基 = start_token，compact 任意行）
    if (idx == 0) {
      int n_prologue = kStage < ntile_k ? kStage : ntile_k;
      for (int s = 0; s < n_prologue; ++s) {
        acquire_and_issue(itile_m, itile_n, s, start_token, igroup);
      }
    }

    // K-loop：full/empty mbarrier 流水 + 寄存器级 s2r/mma 交错（计算路径
    // 与 cp.async 版逐位一致——TMA 只改变数据到达机制）。
    // consumer_slab 跨任务连续（与 producer_slab 同基准，见任务循环前注释）。
    clear(tYr);
#pragma unroll 1
    for (int itile = 0; itile < ntile_k; ++itile) {
      int stage = consumer_slab % kStage;
      uint32_t phase = (consumer_slab / kStage) & 1;
      full_bar[stage].wait(phase);

      s2r_copy_ik(0, stage);
#pragma unroll
      for (int ik = 0; ik < kNumSubK; ++ik) {
        if (ik + 1 < kNumSubK) {
          s2r_copy_ik(ik + 1, stage);
        }
        cute::gemm(tiled_mma, tWr(_, _, ik), tXr(_, _, ik), tYr);
      }

      empty_bar[stage].arrive();
      ++consumer_slab;

      // producer lookahead（仅 tile 内，不跨 tile——边界处流水线自然排空）
      if (idx == 0) {
        int itile2 = itile + kStage;
        if (itile2 < ntile_k) {
          acquire_and_issue(itile_m, itile_n, itile2, start_token, igroup);
        }
      }
    }

    // epilogue：fp32 累加器 → Tout → sC（别名 operand 头部）→ 行谓词写
    // gmem（紧凑 Y）。⚠️ 全局坐标 = compact 起点 + tile 偏移：
    //   全局行 = start_token + itile_m*kTileM + row（compact 布局）
    //   全局列 = itile_n*kTileN + col（bugfix：早先版本丢失 tile 偏移，
    //   n-tile≥1 的输出会覆盖 n-tile 0 且高列号区永远未写——表现为后段
    //   列全错）。
    auto tYc = make_tensor<Tout>(shape(tYr));
#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      tYc(i) = static_cast<Tout>(tYr(i));
    }
    __syncthreads();

    cute::copy(tiled_copy_c, thr_copy_c.retile_S(tYc), tCs4g);
    __syncthreads();

    Tout *y_base = reinterpret_cast<Tout *>(Cptr) + uint64_t(start_token) * n;
    const int col_base = itile_n * kTileN;
#pragma unroll 1
    for (int i = idx; i < kTileM * kTileN; i += kNThreads) {
      int row = i / kTileN;  // m 维（tile 内）
      int col = i % kTileN;  // n 维（tile 内）
      if (itile_m * kTileM + row < m) {
        y_base[uint64_t(itile_m * kTileM + row) * n + col_base + col] = sC(col, row);
      }
    }
    __syncthreads();
  }

  pdl_release();
#else
  // 非 sm90+/sm120a 编译目标：空 kernel（host 侧不会在该架构下分发到这里）
  (void)tma_b;
  (void)tma_a;
  (void)Cptr;
  (void)seqlens_ptr;
  (void)cu_seqlens_ptr;
  (void)tiles_ptr;
  (void)a_rows;
  (void)n;
  (void)k;
  (void)num_group;
  (void)flat_divider;
#endif
}

// ─── host 侧入口 ─────────────────────────────────────────────────────────────
// 约束（op 层已校验）：n % 64 == 0，k % 64 == 0；A 为 compact 布局
//（expert 有序紧凑排列，行数 a_rows），坐标基按 group 平移到任意行
//（见文件头）。tile_m 由 launch 层统一决定并显式传入（与 count 阶段
// 一致）；本家族固定 kTileM ≤ 64（M128 仅 cp.async 家族）。
void group_gemm_tma_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                          const void *seqlens_ptr, const void *cu_seqlens_ptr,
                          const void *tiles_ptr, int a_rows, int n,
                          int k, int num_group, int tile_m, bool is_half,
                          bool use_pdl, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  assert(n % 64 == 0 && "group_gemm_tma: n must be a multiple of kTileN=64");
  assert(k % 64 == 0 && "group_gemm_tma: k must be a multiple of 64");

  cutlass::FastDivmod flat_divider(n / 64);  // num_tile_n

  auto launch = [&](auto t_tag, auto m_tag, auto k_tag) {
    using T = typename decltype(t_tag)::type;
    constexpr int kTileM = decltype(m_tag)::value;
    constexpr int kTileK = decltype(k_tag)::value;
    // kStage 自适应：operands 每 stage (kTileM+64)×kTileK×2B；occupancy
    // 寄存器限制下 occupancy 恒 1 CTA/SM，smem 用满 96KB 预算是零代价
    //（sm120 上限 101376B，需留 scratch(E≤512→2KB)+barriers 余量）。
    // cap 6：prologue 深度边际收益递减，ntile_k < kStage 时也用不满。
    // 示例：M64/K64→6（96KB）；M64/K128→3；M128/K64→4；M128/K128→2。
    constexpr int kStage = [] {
      constexpr int bps = (kTileM + 64) * kTileK * 2;
      constexpr int s = (96 * 1024) / bps;
      return (s > 6) ? 6 : ((s < 2) ? 2 : s);
    }();
    using BaseConfig = config::MoEGemmConfig<T, kTileM, 64, kTileK, kStage>;
    using GemmConfig = TmaGemmConfig<BaseConfig, kStage>;
    constexpr int kNThreads = GemmConfig::kNThreads;

    // A desc：compact 布局 (a_rows, k)——host 一次性构建；坐标基在 kernel
    // 内按 group 平移（任意行起点，见文件头）；smem tile 取单 stage 视图
    //（take<0,2>）
    Tensor mA = make_tensor(
        make_gmem_ptr(static_cast<T const *>(x_ptr)),
        make_shape(int32_t(a_rows), int32_t(k)), make_stride(int64_t(k), _1{}));
    // B desc：静态 (n, k, E) 3D
    Tensor mW = make_tensor(
        make_gmem_ptr(static_cast<T const *>(w_ptr)),
        make_shape(int32_t(n), int32_t(k), int32_t(num_group)),
        make_stride(int64_t(k), _1{}, int64_t(n) * k));

    using SmemLayoutXStage = decltype(take<0, 2>(typename GemmConfig::SmemLayoutX{}));
    using SmemLayoutWStage = decltype(take<0, 2>(typename GemmConfig::SmemLayoutW{}));

    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, SmemLayoutXStage{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, mW, SmemLayoutWStage{});
    using TmaA = decltype(tma_a);
    using TmaB = decltype(tma_b);

    // smem：operand（kStage 级）+ scratch（E+1 int，8B 对齐余量）+ 双 barrier
    constexpr int kSmemStatic = GemmConfig::shm_xw + 2 * kStage * static_cast<int>(sizeof(uint64_t));
    static_assert(kSmemStatic <= 101376, "sm120 per-CTA dynamic smem limit exceeded");
    const int shm_size =
        GemmConfig::shm_xw + ((num_group + 1) * static_cast<int>(sizeof(int)) + 7) / 8 * 8 +
        2 * kStage * static_cast<int>(sizeof(uint64_t));

    auto kernel = &group_gemm_tma_kernel<GemmConfig, TmaA, TmaB, true>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int max_blocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, kNThreads, shm_size);
    if (max_blocks < 1) max_blocks = 1;
    dim3 grid(static_cast<unsigned>(get_sm_count() * max_blocks));

    // sm90+ 恒带 PSS 属性（kernel 体恒含 griddepcontrol 指令；无属性时该指令
    // 为 no-op，但 PSS 才能拿到重叠收益）。FUSE_MOE_NO_PDL / FUSE_MOE_TIME
    // 强制普通 launch：串行化下游启动，分段计时才能归因到单个 kernel。
    (void)use_pdl;
    static const bool s_force_plain = std::getenv("FUSE_MOE_NO_PDL") != nullptr ||
                                      std::getenv("FUSE_MOE_TIME") != nullptr;
    if (!s_force_plain) {
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[0].val.programmaticStreamSerializationAllowed = 1;
      cudaLaunchConfig_t cfg{};
      cfg.gridDim = grid;
      cfg.blockDim = dim3(kNThreads);
      cfg.dynamicSmemBytes = shm_size;
      cfg.stream = stream;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      cudaLaunchKernelEx(&cfg, kernel, tma_b, tma_a, y_ptr,
                         reinterpret_cast<const int *>(seqlens_ptr),
                         reinterpret_cast<const int *>(cu_seqlens_ptr),
                         reinterpret_cast<const int *>(tiles_ptr), a_rows, n, k,
                         num_group, flat_divider);
    } else {
      kernel<<<grid, kNThreads, shm_size, stream>>>(
          tma_b, tma_a, y_ptr, reinterpret_cast<const int *>(seqlens_ptr),
          reinterpret_cast<const int *>(cu_seqlens_ptr),
          reinterpret_cast<const int *>(tiles_ptr), a_rows, n, k, num_group,
          flat_divider);
    }

  };

  auto dispatch_k = [&](auto t_tag, auto m_tag) {
    // kTileK 自适应：k%128==0 用 128（group_modes 坐标收敛解锁，见文件头
    // 注释；每 slab mma 工作量翻倍、barrier/发射开销减半），否则 64。
    // 两个 GEMM 的 k 不同（H vs I），各自独立选择。
    if (k % 128 == 0) {
      launch(t_tag, m_tag, Int<128>{});
    } else {
      launch(t_tag, m_tag, Int<64>{});
    }
  };
  auto dispatch_m = [&](auto t_tag) {
    // tile_m 由 launch 层传入（moe_pick_tile_m；128 仅 FUSE_MOE_TILE_M
    // 调试路径，见 fuse_moe_params.h）
    if (tile_m >= 128) {
      dispatch_k(t_tag, Int<128>{});
    } else if (tile_m <= 32) {
      dispatch_k(t_tag, Int<32>{});
    } else {
      dispatch_k(t_tag, Int<64>{});
    }
  };

  if (is_half) {
    dispatch_m(config::TypeTag<cutlass::half_t>{});
  } else {
    dispatch_m(config::TypeTag<cutlass::bfloat16_t>{});
  }
}

}  // namespace tma
}  // namespace group_gemm
}  // namespace fuse_moe

#endif  // FUSE_MOE_SRC_SM120_GROUP_GEMM_SM120_CUH_
