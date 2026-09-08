// group_gemm_sm89.cuh — MoE group GEMM kernel，cp.async 家族（sm80+ 通用）。
//
// 架构分层（仿 csrc/fa）：sm8x 家族（SM80~SM118 cp.async 基线，
// 4090D=sm89）。cp.async 指令在 sm120 硬件同样受支持，sm120 默认引擎
// 亦复用本家族 kernel（见 fuse_moe_launch.h）。TMA 家族见
// sm120/group_gemm_sm120.cuh；架构分发见 fuse_moe_launch.h。
//
// 移植自 hpc-ops src/group_gemm/sm90/cp_async/{group_gemm_fp8_scatter,group_gemm_fp8}.cu，
// 关键改写：
//   1. sm90 wgmma → sm80 mma.sync（SM80_16x8x16_F32BF16BF16F32_TN /
//      F32F16F16F32_TN；sm120 无 wgmma，mma.sync 为 sm89/sm120 通用骨架），
//      与本仓库 mixed_gemm 的 TiledMMA/分块结构同构（MMAThrLayout (2,4,1) ×
//      Tile<32,64,16>，256 线程）。traits 见 common/group_gemm_config.cuh。
//   2. fp8 权重/激活 + per-tensor scale → bf16/fp16 直入直出（无量化），
//      MMA 累加器 fp32。
//   3. 裁剪 EP 多卡与 TMA descriptor，调度用 kernel 内 horizon 扫描
//      （移植自 hpc common.cuh get_next_tile_horizon）。
//
// 两个变体（同一 kernel 模板，kScatterA 区分）：
//   - scatter（gate_up GEMM）：激活行按 row_indices 从原始 x (S, k) 中
//     gather——逐线程手写 cp.async.cg 16B（src-size 语义：越界行零填充），
//     smem 目的地址经 G2SCopy 的 partition_D 计算（swizzle 感知）。
//   - multistage（down GEMM）：激活行连续（expert 排序后布局），走
//     copy_if + ZFILL 谓词拷贝。
// 另有 gemm1 专用变体 group_gemm_gateup_kernel（见下方）：gate/up 配对
// N-tile，把 silu(gate)·up 融进 epilogue，免 gate_up_out 物化。
//
// smem 排布：[sX (kTileM,kTileK,kStage)][sW (kTileN,kTileK,kStage)]
//            [sC 专用区 (kTileN,kTileM)][shm_tiles (num_group)]。
// K 维整除 kTileK（op 层保证 k%64==0，k%128==0 时用 kTileK=128），无 K 谓词。
//
// 流水线（跨 tile 连续）：
//   - slab 发射流全局连续：issued/read 全局计数，stage = cnt % kStage，
//     task 边界不排空——compute 消费当前 task 尾部 slab 时，发射已领先
//     进入下一 task；
//   - scatter 行索引驻留寄存器（cur/next 两组滚动，issue 侧最多领先
//     compute 一个 task）；
//   - 主循环双 __syncthreads/迭代：sync#1（wait 后）数据可见，sync#2
//     （compute 后）覆写保护（发射滞后 read 恰 kStage-1 个迭代，ring
//     写读分离）；
//   - epilogue 16B 向量化写 gmem。
//
// PDL：kernel 体内 pdl_acquire()/pdl_release()（低架构编译为空），host 侧
// 按 fuse_moe::pdl_supported() 决定 launch attribute（sm120 启用）。

#ifndef FUSE_MOE_SRC_SM89_GROUP_GEMM_SM89_CUH_
#define FUSE_MOE_SRC_SM89_GROUP_GEMM_SM89_CUH_

#include <cuda.h>

#include <cassert>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_types.h"

#include "common/fuse_moe_utils.cuh"
#include "common/group_gemm_config.cuh"

namespace fuse_moe {
namespace group_gemm {

using namespace cute;  // NOLINT

namespace kernels {

// horizon 扫描调度（移植自 hpc common.cuh）：iblock → (igroup, itile_m,
// itile_n)。igroup/sum_tile_m 为跨调用递增的调用方状态（CTA 内 iblock 单调
// 递增，扫描从上次位置续起）。耗尽返回 igroup=-1。
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

// scatter 激活装载（移植自 hpc scatter_load_A_tile，16-bit 适配）：按
// 每线程寄存器行索引 gather 激活行，逐线程手写 16B cp.async.cg（src-size 语义：
// 越界行拷 0 字节 = 零填充，语义同 ZFILL）。smem 目的地址取自 G2SCopy 的
// partition_D（swizzle 感知），每个 (row_iter, thread) 恰好对应一个 16B chunk。
// 行索引来源：调用方预取的本线程行数组 my_rows[row_iter]（每个 row_iter 恰
// 一行——行号 = row_iter*kRowsPerIter + threadIdx.x/kThreadsPerRow）。
template <typename Config, bool kFullTile, typename TsXCopy>
__device__ __forceinline__ void scatter_load_x_tile(
    TsXCopy &tsX_copy, const typename Config::Tin *__restrict__ x_pool,
    const int *__restrict__ my_rows, int num_valid, int k_col_base, int k_row_stride,
    int ismem) {
  using Tin = typename Config::Tin;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileK = Config::kTileK;
  constexpr int kElemsPerAtom = 16 / int(sizeof(Tin));               // 8
  constexpr int kThreadsPerRow = kTileK / kElemsPerAtom;             // 8 / 16
  constexpr int kRowsPerIter = Config::kNThreads / kThreadsPerRow;   // 32 / 16
  constexpr int kNumIters = (kTileM + kRowsPerIter - 1) / kRowsPerIter;
  constexpr bool kHasVirtualOOB = (kNumIters * kRowsPerIter > kTileM);

  const int row_in_group = threadIdx.x / kThreadsPerRow;
  const int k_thread = threadIdx.x % kThreadsPerRow;
  const int k_col = k_col_base + k_thread * kElemsPerAtom;

  auto do_one = [&](int row_iter, int local_row) {
    void *smem_ptr = (void *)&tsX_copy(cute::Int<0>{}, row_iter, cute::Int<0>{}, ismem);
    int global_row;
    int src_size;
    if constexpr (kFullTile) {
      global_row = my_rows[row_iter];
      src_size = 16;
    } else {
      const bool valid = local_row < num_valid;
      global_row = valid ? my_rows[row_iter] : 0;
      src_size = valid ? 16 : 0;
    }
    const void *gmem_ptr =
        (const void *)&x_pool[uint64_t(global_row) * k_row_stride + k_col];
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
                 "l"(gmem_ptr), "n"(16), "r"(src_size));
  };

#pragma unroll
  for (int row_iter = 0; row_iter < kNumIters; ++row_iter) {
    const int local_row = row_iter * kRowsPerIter + row_in_group;
    if constexpr (kHasVirtualOOB) {
      if (local_row < kTileM) {
        do_one(row_iter, local_row);
      }
    } else {
      do_one(row_iter, local_row);
    }
  }
}

// ─── 核心 kernel ─────────────────────────────────────────────────────────────
// kScatterA=true：gate_up GEMM（X 按行 gather）；false：down GEMM（X 连续）。
template <typename Config, bool kScatterA>
__global__ void __launch_bounds__(Config::kNThreads, 1)
group_gemm_kernel(void *Cptr, const void *Xptr, const void *Wptr,
                  const int *row_indices_ptr,  // kScatterA 时非空
                  const int *seqlens_ptr, const int *cu_seqlens_ptr, const int *tiles_ptr,
                  int n, int k, int num_group, cutlass::FastDivmod flat_divider) {
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

  // smem: [sX][sW][sC 专用区][shm_tiles]——sC 不再别名 operand 基地址：
  // task 边界的 epilogue 期间流水继续向 operand stage 发射下一 task 的
  // cp.async（跨 tile 连续，见下方任务流注释），别名会与之冲突。
  extern __shared__ uint8_t shm_data[] alignas(128);
  Tin *xshm = reinterpret_cast<Tin *>(shm_data);
  Tin *wshm = xshm + cosize(typename Config::SmemLayoutX{});
  Tout *cshm = reinterpret_cast<Tout *>(wshm + cosize(typename Config::SmemLayoutW{}));
  // shm_tiles 在 sC 预留区（含 r2s 分区越界写部分，见 config 的 shm_c_alloc
  // 注释）之后，不能紧贴 cosize(SmemLayoutC)
  int *shm_tiles =
      reinterpret_cast<int *>(cshm + Config::shm_c_alloc / int(sizeof(Tout)));

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  // PDL acquire：等上游 kernel（count/act）写完 row_indices/tiles/cu_seqlens/x
  pdl_acquire();

  for (int i = idx; i < num_group; i += kNThreads) {
    shm_tiles[i] = tiles_ptr[i];
  }
  __syncthreads();

  auto sX = make_tensor(make_smem_ptr(xshm), typename Config::SmemLayoutX{});  // (M, K, S)
  auto sW = make_tensor(make_smem_ptr(wshm), typename Config::SmemLayoutW{});  // (N, K, S)
  auto sC = make_tensor(make_smem_ptr(cshm), typename Config::SmemLayoutC{});  // (N, M)

  // MMA / s2r 句柄（tile 循环外构造一次；swapAB：A=W, B=X, C=(N, M)）
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tWr = thr_mma.partition_fragment_A(sW(_, _, _0{}));  // (MMA, MMA_N, MMA_K)
  auto tXr = thr_mma.partition_fragment_B(sX(_, _, _0{}));  // (MMA, MMA_M, MMA_K)
  auto tYr = thr_mma.partition_fragment_C(sC);              // (MMA, MMA_N, MMA_M)

  using LDSM_ATOM = Copy_Atom<SM75_U32x4_LDSM_N, Tin>;
  auto s2r_copy_w = make_tiled_copy_A(LDSM_ATOM{}, tiled_mma);
  auto s2r_copy_x = make_tiled_copy_B(LDSM_ATOM{}, tiled_mma);
  auto s2r_thr_copy_w = s2r_copy_w.get_slice(idx);
  auto s2r_thr_copy_x = s2r_copy_x.get_slice(idx);
  auto tWs4r = s2r_thr_copy_w.partition_S(sW);  // (CPY, CPY_N, CPY_K, stage)
  auto tXs4r = s2r_thr_copy_x.partition_S(sX);  // (CPY, CPY_M, CPY_K, stage)

  auto tXr_view = s2r_thr_copy_x.retile_D(tXr);
  auto tWr_view = s2r_thr_copy_w.retile_D(tWr);

  auto s2r_copy_ik = [&](int ik, int stage) {
    cute::copy(s2r_copy_x, tXs4r(_, _, ik, stage), tXr_view(_, _, ik));
    cute::copy(s2r_copy_w, tWs4r(_, _, ik, stage), tWr_view(_, _, ik));
  };

  // g2s 句柄（scatter 路径仅用 partition_D 的 swizzle 地址；W 用 cute copy）
  typename Config::G2SCopy g2s_copy;
  auto g2s_thr_copy = g2s_copy.get_slice(idx);
  auto tsX_copy = g2s_thr_copy.partition_D(sX);  // (CPY, CPY_M, CPY_K, stage)
  auto tsW_copy = g2s_thr_copy.partition_D(sW);

  // r2s / s2g 句柄（sC 无 swizzle，UniversalCopy 标量粒度；每 tile 一次）
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<Tout>, Tout>;
  auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto thr_copy_c = tiled_copy_c.get_slice(idx);
  auto tCs4g = thr_copy_c.partition_D(sC);

  const int ntile = k / kTileK;

  // ── 任务流 + 全局 slab 流水（跨 tile 连续）───────────────────────────
  // issued/read 为全局计数（stage = cnt % kStage），发射流在 task 边界
  // 不排空：compute 消费当前 task 尾部 slab 时，发射已领先 kStage-1
  // 进入下一 task——epilogue 期间下一 task 的 prologue slab 已在途。
  //
  // wait 语义（迭代内顺序：issue → fence → wait；mixed_gemm 同款）：slab r
  // 在迭代 r-kStage+1 发射且被同迭代 fence commit，到其 wait（迭代 r）
  // 之间恰有 kStage-1 个 fence——wait<kStage-1> 保证 r 落地（空 commit
  // group 计入未完成组数）。issue 必须位于 fence 之前：若放到 fence 之后，
  // commit 会落后一个迭代，计数变 kStage-2，wait 不再保证落地。
  //
  // ring 安全（双 sync/迭代）：迭代 i 的发射写 stage (i+kStage-1)%kStage
  // = (i-1)%kStage（上一迭代 ldmatrix 读取的 stage）——sync#1（wait 后）
  // 保证本 slab 全线程可见，sync#2（compute 后）保证读完成后下一迭代
  // 的发射才可覆写。
  struct TaskCtx {
    int valid;
    int igroup, itile_m, itile_n;
    int start_token, m;
  };
  // ⚠️ sched_igroup/sched_sum_tile_m 为跨 pull_task 持久的调度器状态
  //（horizon 线性扫描从上次位置续起，摊销 O(1)）——传临时变量会丢失
  // 续扫位置，任务映射错乱。
  int sched_igroup = 0;
  int sched_sum_tile_m = 0;
  auto pull_task = [&](TaskCtx &t) {
    // 已耗尽：直接返回，不得再扫——耗尽后 sched_igroup=-1，后续调用会
    // 从 i=-1 扫描（tiles_ptr[-1] 越界读 + sum_tile_m 污染累积），当污染
    // 和超过 itile_m_total 时会合成假任务（垃圾坐标 → 散射错误写，实测
    // S=512/H=512/I=192/E=32/K=8 触发 ~5% 元素损坏）。旧 while-break 结构
    // 耗尽即退出；Tn 预取路径必须显式守卫。
    if (sched_igroup < 0) {
      t.valid = false;
      return;
    }
    get_next_tile_horizon(shm_tiles, iblock, num_group, sched_igroup, t.itile_m, t.itile_n,
                          sched_sum_tile_m, flat_divider);
    iblock += gridDim.x;
    t.valid = (sched_igroup >= 0);
    if (t.valid) {
      t.igroup = sched_igroup;
      t.start_token = cu_seqlens_ptr[sched_igroup];
      t.m = seqlens_ptr[sched_igroup];
    }
  };

  // scatter 行索引（寄存器，每线程 kNumRowIters 个；issue 侧最多领先
  // compute 一个 task，故 cur/next 两组足够，task 边界滚动 + 预取）
  constexpr int kNumRowIters =
      (kTileM + Config::kRowsPerIter - 1) / Config::kRowsPerIter;
  [[maybe_unused]] int rows_cur[kNumRowIters];
  [[maybe_unused]] int rows_next[kNumRowIters];
  auto load_my_rows = [&](const TaskCtx &t, int (&rows)[kNumRowIters]) {
    const int row_in_group = idx / Config::kThreadsPerRow;
    const int nvalid_raw = t.m - t.itile_m * kTileM;
    const int nvalid = nvalid_raw < kTileM ? nvalid_raw : kTileM;
    const int base = t.start_token + t.itile_m * kTileM;
#pragma unroll
    for (int i = 0; i < kNumRowIters; ++i) {
      const int lr = i * Config::kRowsPerIter + row_in_group;
      rows[i] = (lr < nvalid) ? row_indices_ptr[base + lr] : 0;
    }
  };

  // K-slab 装载（按发射时刻的 task 上下文；gmem 分区/谓词随 task 变化，
  // 在此重算——layout 代数，代价可忽略）
  auto load_slab = [&](const TaskCtx &t, int itile_k, int istage,
                       [[maybe_unused]] const int *rows) {
    Tensor W = make_tensor(
        make_gmem_ptr((Tin const *)Wptr + uint64_t(t.igroup) * n * k),
        make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor gW =
        local_tile(W, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(t.itile_n, _));
    auto tgW_copy = g2s_thr_copy.partition_S(gW);
    if constexpr (kScatterA) {
      const int num_valid_raw = t.m - t.itile_m * kTileM;
      const int num_valid = num_valid_raw < kTileM ? num_valid_raw : kTileM;
      if (num_valid == kTileM) {
        scatter_load_x_tile<Config, /*kFullTile=*/true>(tsX_copy, (Tin const *)Xptr, rows,
                                                        num_valid, itile_k * kTileK, k, istage);
      } else {
        scatter_load_x_tile<Config, /*kFullTile=*/false>(
            tsX_copy, (Tin const *)Xptr, rows, num_valid, itile_k * kTileK, k, istage);
      }
    } else {
      Tensor A = make_tensor(
          make_gmem_ptr((Tin const *)Xptr + uint64_t(t.start_token) * k),
          make_shape(t.m, k), make_stride(k, Int<1>{}));
      Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                             make_coord(t.itile_m, _));
      auto tgA_copy = g2s_thr_copy.partition_S(gA);
      // M 尾部谓词（K/N 维整除，无谓词）；ZFILL 原子把越界行补零
      auto tIA = g2s_thr_copy.partition_S(
          make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
      auto pred_x = make_tensor<bool>(shape(tIA));
#pragma unroll
      for (int i = 0; i < size(tIA); ++i) {
        pred_x(i) = t.itile_m * kTileM + get<0>(tIA(i)) < t.m;
      }
      cute::copy_if(g2s_copy, pred_x, tgA_copy(_, _, _, itile_k), tsX_copy(_, _, _, istage));
    }
    cute::copy(g2s_copy, tgW_copy(_, _, _, itile_k), tsW_copy(_, _, _, istage));
  };

  TaskCtx Tc{}, Tn{};
  int iss_kk = 0;            // 发射流在当前发射 task 内的 K-slab 位置
  bool iss_is_next = false;  // 发射 task 已推进到 Tn？
  int issued = 0;            // 全局已发射 slab 数（stage = issued % kStage）
  int slab_read = 0;         // 全局已消费 slab 数（stage = slab_read % kStage）

  pull_task(Tc);
  pull_task(Tn);
  if constexpr (kScatterA) {
    load_my_rows(Tc, rows_cur);
    if (Tn.valid) load_my_rows(Tn, rows_next);
  }

  // 发射一步：先发当前 compute task 的剩余 slab，耗尽后切到 Tn（cap：
  // 最多领先 compute 一个 task——行索引寄存器只存 cur/next 两组；
  // 小 ntile 场景流水深度受限，可接受）
  auto issue_step = [&]() {
    if (iss_kk >= ntile && !iss_is_next) {
      iss_is_next = true;
      iss_kk = 0;
    }
    const TaskCtx &t = iss_is_next ? Tn : Tc;
    if (!t.valid || iss_kk >= ntile) return;
    load_slab(t, iss_kk, issued % kStage, iss_is_next ? rows_next : rows_cur);
    ++iss_kk;
    ++issued;
  };

  // prologue：kStage-1 次发射（issue_step 自带流耗尽保护；fence 逐次——
  // commit group 与 slab 一一对应，含空 commit）
#pragma unroll
  for (int i = 0; i < kStage - 1; ++i) {
    issue_step();
    cp_async_fence();
  }

  while (Tc.valid) {
    clear(tYr);
#pragma unroll 1
    for (int kk = 0; kk < ntile; ++kk) {
      // 发射 slab slab_read + kStage - 1（跨 task 连续；写 stage
      // (slab_read-1)%kStage，由上一迭代 sync#2 保护）。⚠️ 必须在 fence
      // 之前（见上方 wait 语义注释）。
      issue_step();

      cp_async_fence();
      cp_async_wait<kStage - 1>();
      __syncthreads();  // sync#1：slab slab_read 就绪，全线程可见

      const int stage = slab_read % kStage;
      // 寄存器级 s2r/mma 交错（下一 sub-k 预取掩盖 ldmatrix 延迟）
      s2r_copy_ik(0, stage);
#pragma unroll
      for (int ik = 0; ik < kNumSubK; ++ik) {
        if (ik + 1 < kNumSubK) {
          s2r_copy_ik(ik + 1, stage);
        }
        cute::gemm(tiled_mma, tWr(_, _, ik), tXr(_, _, ik), tYr);
      }
      ++slab_read;

      __syncthreads();  // sync#2：读取完成，下一迭代的发射才可覆写该 stage
    }

    // epilogue：fp32 累加器 → 转换 Tout → sC（专用区，与在途 cp.async
    // 无冲突；下一 task 的 prologue slab 此刻已在途）→ 行谓词 16B 向量化
    // 写 gmem（n%64==0 列恒在界内，行谓词整行判定）
    auto tYc = make_tensor<Tout>(shape(tYr));
#pragma unroll
    for (int i = 0; i < size(tYr); ++i) {
      tYc(i) = static_cast<Tout>(tYr(i));
    }
    cute::copy(tiled_copy_c, thr_copy_c.retile_S(tYc), tCs4g);
    __syncthreads();

    {
      constexpr int kVecElems = 16 / int(sizeof(Tout));  // 8
      const int nvec = kTileN / kVecElems;
      Tout *y_base = reinterpret_cast<Tout *>(Cptr) + uint64_t(Tc.start_token) * n;
      const int col_base = Tc.itile_n * kTileN;
      const int row_base = Tc.itile_m * kTileM;
#pragma unroll 1
      for (int i = idx; i < kTileM * nvec; i += kNThreads) {
        const int r = i / nvec;
        const int c8 = (i % nvec) * kVecElems;
        if (row_base + r < Tc.m) {
          *reinterpret_cast<uint4 *>(y_base + uint64_t(row_base + r) * n + col_base + c8) =
              *reinterpret_cast<const uint4 *>(&sC(c8, r));
        }
      }
    }
    __syncthreads();  // sC 读取完成（下一 task 的 r2s 覆写前分隔）

    // task 边界：compute task 前移；拉取新 Tn；行索引寄存器滚动。
    // iss_kk 保留（发射流在新 Tc（原 Tn）内的位置）；iss_is_next 复位后
    // 发射继续填补新 Tc 的剩余 slab，耗尽再切新 Tn。
    Tc = Tn;
    pull_task(Tn);
    iss_is_next = false;
    if constexpr (kScatterA) {
#pragma unroll
      for (int i = 0; i < kNumRowIters; ++i) rows_cur[i] = rows_next[i];
      if (Tn.valid) load_my_rows(Tn, rows_next);
    }
  }

  // PDL release：通知下游可提前启动
  pdl_release();
}

// ─── gemm1 专用：gate/up 配对融合 kernel（scatter + silu·mul epilogue）──
// 在 group_gemm_kernel（scatter 变体）的连续流水骨架上把 act_mul 融进
// epilogue：
//   - task 粒度从 N-tile 改为 N-pair：每 task 同时计算 gate 面板（W 行
//     [itile_n·64, +64)）与 up 面板（W 行 [I+itile_n·64, +64)），共享
//     同一 X tile（X 的 smem/L2 装载量减半，task 数减半）；
//   - 双累加器 tYr_g/tYr_u：同一 tiled_mma + 同一 B 分区 ⇒ fragment 索引
//     i 在两次 MMA 中映射到相同 (m_row, n_col)，epilogue 直接
//     act = silu(tYr_g(i)) · tYr_u(i)；
//   - 输出直写 act_out (T, I)：省去 gate_up_out (T, 2I) 物化（2×T×I×2B
//     的 DRAM 写+读）与独立 act_mul kernel。
// smem：[sX (M,K,S)][sWg (64,K,S)][sWu (64,K,S)][sC 预留][shm_tiles]。
// flat_divider = (n/2)/64（配对 tile 数，n = 2I）。
template <typename Config>
__global__ void __launch_bounds__(Config::kNThreads, 1)
group_gemm_gateup_kernel(void *ActPtr, const void *Xptr, const void *Wptr,
                         const int *row_indices_ptr, const int *seqlens_ptr,
                         const int *cu_seqlens_ptr, const int *tiles_ptr, int n, int k,
                         int num_group, cutlass::FastDivmod flat_divider) {
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
  const int act_cols = n / 2;              // I（act_out 行宽）
  const int up_tile_base = act_cols / 64;  // up 面板 tile 索引基（I/64）

  extern __shared__ uint8_t shm_data[] alignas(128);
  Tin *xshm = reinterpret_cast<Tin *>(shm_data);
  Tin *wgshm = xshm + cosize(typename Config::SmemLayoutX{});
  Tin *wushm = wgshm + cosize(typename Config::SmemLayoutW{});
  Tout *cshm = reinterpret_cast<Tout *>(wushm + cosize(typename Config::SmemLayoutW{}));
  int *shm_tiles =
      reinterpret_cast<int *>(cshm + Config::shm_c_alloc / int(sizeof(Tout)));

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  pdl_acquire();

  for (int i = idx; i < num_group; i += kNThreads) {
    shm_tiles[i] = tiles_ptr[i];
  }
  __syncthreads();

  auto sX = make_tensor(make_smem_ptr(xshm), typename Config::SmemLayoutX{});   // (M,K,S)
  auto sWg = make_tensor(make_smem_ptr(wgshm), typename Config::SmemLayoutW{});  // (N,K,S)
  auto sWu = make_tensor(make_smem_ptr(wushm), typename Config::SmemLayoutW{});
  auto sC = make_tensor(make_smem_ptr(cshm), typename Config::SmemLayoutC{});  // (N,M)

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tWr_g = thr_mma.partition_fragment_A(sWg(_, _, _0{}));
  auto tWr_u = thr_mma.partition_fragment_A(sWu(_, _, _0{}));
  auto tXr = thr_mma.partition_fragment_B(sX(_, _, _0{}));
  auto tYr_g = thr_mma.partition_fragment_C(sC);  // gate 半区累加器
  auto tYr_u = thr_mma.partition_fragment_C(sC);  // up 半区累加器

  using LDSM_ATOM = Copy_Atom<SM75_U32x4_LDSM_N, Tin>;
  auto s2r_copy_w = make_tiled_copy_A(LDSM_ATOM{}, tiled_mma);
  auto s2r_copy_x = make_tiled_copy_B(LDSM_ATOM{}, tiled_mma);
  auto s2r_thr_copy_w = s2r_copy_w.get_slice(idx);
  auto s2r_thr_copy_x = s2r_copy_x.get_slice(idx);
  auto tWgs4r = s2r_thr_copy_w.partition_S(sWg);
  auto tWus4r = s2r_thr_copy_w.partition_S(sWu);
  auto tXs4r = s2r_thr_copy_x.partition_S(sX);

  auto tXr_view = s2r_thr_copy_x.retile_D(tXr);
  auto tWgr_view = s2r_thr_copy_w.retile_D(tWr_g);
  auto tWur_view = s2r_thr_copy_w.retile_D(tWr_u);

  auto s2r_copy_ik = [&](int ik, int stage) {
    cute::copy(s2r_copy_x, tXs4r(_, _, ik, stage), tXr_view(_, _, ik));
    cute::copy(s2r_copy_w, tWgs4r(_, _, ik, stage), tWgr_view(_, _, ik));
    cute::copy(s2r_copy_w, tWus4r(_, _, ik, stage), tWur_view(_, _, ik));
  };

  typename Config::G2SCopy g2s_copy;
  auto g2s_thr_copy = g2s_copy.get_slice(idx);
  auto tsX_copy = g2s_thr_copy.partition_D(sX);
  auto tsWg_copy = g2s_thr_copy.partition_D(sWg);
  auto tsWu_copy = g2s_thr_copy.partition_D(sWu);

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<Tout>, Tout>;
  auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto thr_copy_c = tiled_copy_c.get_slice(idx);
  auto tCs4g = thr_copy_c.partition_D(sC);

  const int ntile = k / kTileK;

  // 任务流 + 全局 slab 流水（跨 tile 连续，同 group_gemm_kernel——
  // wait 语义 / ring 安全 / 调度器状态注意事项见该 kernel 的注释块）
  struct TaskCtx {
    int valid;
    int igroup, itile_m, itile_n;
    int start_token, m;
  };
  int sched_igroup = 0;
  int sched_sum_tile_m = 0;
  auto pull_task = [&](TaskCtx &t) {
    // 任务流已耗尽时直接返回：get_next_tile_horizon 耗尽时把 igroup 置 -1，
    // 后续调用会从 i=-1 扫描（tiles_ptr[-1] 越界读 + sum_tile_m 污染累积），
    // 当污染和超过 itile_m_total 时会合成假任务（垃圾坐标 → 散射错误写）。
    // Tn 预取路径在 compute task 结束后仍会调用本函数，必须显式守卫。
    if (sched_igroup < 0) {
      t.valid = false;
      return;
    }
    get_next_tile_horizon(shm_tiles, iblock, num_group, sched_igroup, t.itile_m, t.itile_n,
                          sched_sum_tile_m, flat_divider);
    iblock += gridDim.x;
    t.valid = (sched_igroup >= 0);
    if (t.valid) {
      t.igroup = sched_igroup;
      t.start_token = cu_seqlens_ptr[sched_igroup];
      t.m = seqlens_ptr[sched_igroup];
    }
  };

  constexpr int kNumRowIters =
      (kTileM + Config::kRowsPerIter - 1) / Config::kRowsPerIter;
  int rows_cur[kNumRowIters];
  int rows_next[kNumRowIters];
  auto load_my_rows = [&](const TaskCtx &t, int (&rows)[kNumRowIters]) {
    const int row_in_group = idx / Config::kThreadsPerRow;
    const int nvalid_raw = t.m - t.itile_m * kTileM;
    const int nvalid = nvalid_raw < kTileM ? nvalid_raw : kTileM;
    const int base = t.start_token + t.itile_m * kTileM;
#pragma unroll
    for (int i = 0; i < kNumRowIters; ++i) {
      const int lr = i * Config::kRowsPerIter + row_in_group;
      rows[i] = (lr < nvalid) ? row_indices_ptr[base + lr] : 0;
    }
  };

  auto load_slab = [&](const TaskCtx &t, int itile_k, int istage, const int *rows) {
    Tensor W = make_tensor(
        make_gmem_ptr((Tin const *)Wptr + uint64_t(t.igroup) * n * k),
        make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor gWg = local_tile(W, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(t.itile_n, _));
    Tensor gWu = local_tile(W, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(up_tile_base + t.itile_n, _));
    auto tgWg_copy = g2s_thr_copy.partition_S(gWg);
    auto tgWu_copy = g2s_thr_copy.partition_S(gWu);
    const int num_valid_raw = t.m - t.itile_m * kTileM;
    const int num_valid = num_valid_raw < kTileM ? num_valid_raw : kTileM;
    if (num_valid == kTileM) {
      scatter_load_x_tile<Config, /*kFullTile=*/true>(
          tsX_copy, (Tin const *)Xptr, rows, num_valid, itile_k * kTileK, k, istage);
    } else {
      scatter_load_x_tile<Config, /*kFullTile=*/false>(
          tsX_copy, (Tin const *)Xptr, rows, num_valid, itile_k * kTileK, k, istage);
    }
    cute::copy(g2s_copy, tgWg_copy(_, _, _, itile_k), tsWg_copy(_, _, _, istage));
    cute::copy(g2s_copy, tgWu_copy(_, _, _, itile_k), tsWu_copy(_, _, _, istage));
  };

  TaskCtx Tc{}, Tn{};
  int iss_kk = 0;
  bool iss_is_next = false;
  int issued = 0;
  int slab_read = 0;

  pull_task(Tc);
  pull_task(Tn);
  load_my_rows(Tc, rows_cur);
  if (Tn.valid) load_my_rows(Tn, rows_next);

  auto issue_step = [&]() {
    if (iss_kk >= ntile && !iss_is_next) {
      iss_is_next = true;
      iss_kk = 0;
    }
    const TaskCtx &t = iss_is_next ? Tn : Tc;
    if (!t.valid || iss_kk >= ntile) return;
    load_slab(t, iss_kk, issued % kStage, iss_is_next ? rows_next : rows_cur);
    ++iss_kk;
    ++issued;
  };

#pragma unroll
  for (int i = 0; i < kStage - 1; ++i) {
    issue_step();
    cp_async_fence();
  }

  while (Tc.valid) {
    clear(tYr_g);
    clear(tYr_u);
#pragma unroll 1
    for (int kk = 0; kk < ntile; ++kk) {
      issue_step();

      cp_async_fence();
      cp_async_wait<kStage - 1>();
      __syncthreads();  // sync#1：slab slab_read 就绪，全线程可见

      const int stage = slab_read % kStage;
      s2r_copy_ik(0, stage);
#pragma unroll
      for (int ik = 0; ik < kNumSubK; ++ik) {
        if (ik + 1 < kNumSubK) {
          s2r_copy_ik(ik + 1, stage);
        }
        cute::gemm(tiled_mma, tWr_g(_, _, ik), tXr(_, _, ik), tYr_g);
        cute::gemm(tiled_mma, tWr_u(_, _, ik), tXr(_, _, ik), tYr_u);
      }
      ++slab_read;

      __syncthreads();  // sync#2：读取完成，下一迭代的发射才可覆写该 stage
    }

    // 融合 epilogue：silu(gate)·up（fp32）→ Tout → sC → 行谓词 16B
    // 向量化写 act_out (T, I)。⚠️ fragment 配对正确性依赖：tYr_g/tYr_u
    // 来自同一 tiled_mma + 同一 B 分区的两次 gemm ⇒ 索引 i 映射到相同
    // (m_row, n_col)，仅 W 数据源（gate/up 面板）不同。
    auto tActc = make_tensor<Tout>(shape(tYr_g));
#pragma unroll
    for (int i = 0; i < size(tYr_g); ++i) {
      tActc(i) = static_cast<Tout>(silu(tYr_g(i)) * tYr_u(i));
    }
    cute::copy(tiled_copy_c, thr_copy_c.retile_S(tActc), tCs4g);
    __syncthreads();

    {
      constexpr int kVecElems = 16 / int(sizeof(Tout));  // 8
      const int nvec = kTileN / kVecElems;
      Tout *y_base = reinterpret_cast<Tout *>(ActPtr) + uint64_t(Tc.start_token) * act_cols;
      const int col_base = Tc.itile_n * kTileN;
      const int row_base = Tc.itile_m * kTileM;
#pragma unroll 1
      for (int i = idx; i < kTileM * nvec; i += kNThreads) {
        const int r = i / nvec;
        const int c8 = (i % nvec) * kVecElems;
        if (row_base + r < Tc.m) {
          *reinterpret_cast<uint4 *>(y_base + uint64_t(row_base + r) * act_cols + col_base + c8) =
              *reinterpret_cast<const uint4 *>(&sC(c8, r));
        }
      }
    }
    __syncthreads();  // sC 读取完成（下一 task 的 r2s 覆写前分隔）

    Tc = Tn;
    pull_task(Tn);
    iss_is_next = false;
#pragma unroll
    for (int i = 0; i < kNumRowIters; ++i) rows_cur[i] = rows_next[i];
    if (Tn.valid) load_my_rows(Tn, rows_next);
  }

  pdl_release();
}

}  // namespace kernels

// ─── host 侧入口 ─────────────────────────────────────────────────────────────
// 约束（op 层已校验）：n % 64 == 0，k % 64 == 0；tile_m 由 launch 层统一
// 决定并显式传入（fuse_moe_launch.h 的 moe_pick_tile_m，与 count 阶段
// 一致——tiles[] 按 ceil(len/tile_m) 计算，kernel 按其迭代 itile_m）。本
// 家族无 kTileM=128 实例，tile_m ∈ {32, 64}。
void group_gemm_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                      const void *row_indices_ptr,  // nullptr → 非 scatter（down GEMM）
                      const void *seqlens_ptr, const void *cu_seqlens_ptr,
                      const void *tiles_ptr, int n, int k, int num_group,
                      int tile_m, bool is_half, bool use_pdl,
                      cudaStream_t stream) {
  using namespace cute;  // NOLINT

  assert(n % 64 == 0 && "group_gemm: n must be a multiple of kTileN=64");
  assert(k % 64 == 0 && "group_gemm: k must be a multiple of 64");

  const bool scatter = (row_indices_ptr != nullptr);
  assert(tile_m == 32 || tile_m == 64 || tile_m == 128);

  cutlass::FastDivmod flat_divider(n / 64);  // num_tile_n

  auto launch = [&](auto t_tag, auto m_tag, auto k_tag) {
    using T = typename decltype(t_tag)::type;
    constexpr int kTileM = decltype(m_tag)::value;
    constexpr int kTileK = decltype(k_tag)::value;
    // kStage：按 smem 预算自适应（sC 预留区计入，按 (N=64, max(M,64)) C
    // tile 预留，见 config 的 shm_c_alloc 注释；sm89/sm120 每块动态 smem
    // 上限 99KB/101376B，预算取 96KB 留 scratch 余量）：
    //   M128/K64→3（90KB）、M64/K128→2（80KB）、M64/K64→5（88KB）、
    //   M32/K128→3（80KB）、M32/K64→6（80KB）。
    // M128/K128 需 112KB 超限 → dispatch_k 的 k128_fits 门控降级 K64。
    constexpr int kStage = [] {
      constexpr int bps = (kTileM + 64) * kTileK * 2;             // 每 stage X+W
      constexpr int sc = (kTileM < 64 ? 64 : kTileM) * 64 * 2;   // sC 预留区
      constexpr int s = (96 * 1024 - sc) / bps;
      return (s > 6) ? 6 : ((s < 2) ? 2 : s);
    }();
    using GemmConfig = config::MoEGemmConfig<T, kTileM, 64, kTileK, kStage>;

    constexpr int kNThreads = GemmConfig::kNThreads;
    const int shm_size =
        GemmConfig::shm_xw + GemmConfig::shm_c_alloc +
        num_group * static_cast<int>(sizeof(int));

    auto kernel = scatter ? kernels::group_gemm_kernel<GemmConfig, /*kScatterA=*/true>
                          : kernels::group_gemm_kernel<GemmConfig, /*kScatterA=*/false>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int max_blocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, kNThreads, shm_size);
    if (max_blocks < 1) max_blocks = 1;
    dim3 grid(static_cast<unsigned>(get_sm_count() * max_blocks));

    launch_kernel_pdl(kernel, grid, dim3(kNThreads), shm_size, stream, use_pdl, y_ptr, x_ptr,
                      w_ptr, reinterpret_cast<const int *>(row_indices_ptr),
                      reinterpret_cast<const int *>(seqlens_ptr),
                      reinterpret_cast<const int *>(cu_seqlens_ptr),
                      reinterpret_cast<const int *>(tiles_ptr), n, k, num_group,
                      flat_divider);
  };

  auto dispatch_m = [&](auto t_tag) {
    auto dispatch_k = [&](auto m_tag) {
      // K=128 门控：双 operand 预算（kStage≥2）+ sC 预留须落在 96KB 内。
      // M64/K128→80KB ✓；M128/K128 需 112KB ✗ → 降级 K64（S3，90KB）。
      constexpr int kTileM_v = decltype(m_tag)::value;
      constexpr bool k128_fits =
          ((kTileM_v + 64) * 128 * 2 * 2 +
           (kTileM_v < 64 ? 64 : kTileM_v) * 64 * 2) <= (96 * 1024);
      if (k % 128 == 0 && k128_fits) {
        launch(t_tag, m_tag, Int<128>{});
      } else {
        launch(t_tag, m_tag, Int<64>{});
      }
    };
    if (tile_m >= 128) {
      dispatch_k(Int<128>{});
    } else if (tile_m <= 32) {
      dispatch_k(Int<32>{});
    } else {
      dispatch_k(Int<64>{});
    }
  };

  if (is_half) {
    dispatch_m(config::TypeTag<cutlass::half_t>{});
  } else {
    dispatch_m(config::TypeTag<cutlass::bfloat16_t>{});
  }
}

// ── gemm1 融合入口（gate/up 配对 + silu·mul epilogue）─────────────────
// 输出 act_out (T, n/2)；n = 2I（W 行数，gate 在 [0, I) up 在 [I, 2I)）。
// flat_divider = (n/2)/64（配对 tile 数）。kStage 预算（operands/stage
// = (kTileM + 2·64)·kTileK·2B + sC 预留 8KB）：M64/K64→3、M64/K128→2、
// M32/K64→4、M32/K128→2（双 W 面板挤占 operand 区，深度略低于基线变体，
// 由跨 tile 连续流水补偿）。
void group_gemm_gateup_fused_async(void *act_ptr, const void *x_ptr, const void *w_ptr,
                                   const void *row_indices_ptr, const void *seqlens_ptr,
                                   const void *cu_seqlens_ptr, const void *tiles_ptr, int n,
                                   int k, int num_group, int tile_m, bool is_half, bool use_pdl,
                                   cudaStream_t stream) {
  using namespace cute;  // NOLINT

  assert(n % 128 == 0 && "gateup_fused: n (=2I) must be a multiple of 128");
  assert(k % 64 == 0 && "gateup_fused: k must be a multiple of 64");
  assert(tile_m == 32 || tile_m == 64 || tile_m == 128);

  cutlass::FastDivmod flat_divider((n / 2) / 64);  // 配对 tile 数

  auto launch = [&](auto t_tag, auto m_tag, auto k_tag) {
    using T = typename decltype(t_tag)::type;
    constexpr int kTileM = decltype(m_tag)::value;
    constexpr int kTileK = decltype(k_tag)::value;
    // kStage（sC 预留按 (N=64, max(M,64)) C tile）：M128/K64→2（82KB）、
    // M64/K64→3（82KB）、M32/K64→4（80KB）；K=128 仅 M32（90KB）。
    constexpr int kStage = [] {
      constexpr int bps = (kTileM + 2 * 64) * kTileK * 2;         // X + 双 W 面板
      constexpr int sc = (kTileM < 64 ? 64 : kTileM) * 64 * 2;   // sC 预留区
      constexpr int s = (96 * 1024 - sc) / bps;
      return (s > 6) ? 6 : ((s < 2) ? 2 : s);
    }();
    using GemmConfig = config::MoEGemmConfig<T, kTileM, 64, kTileK, kStage>;

    constexpr int kNThreads = GemmConfig::kNThreads;
    // operands = X + 2×W 面板 = shm_xw + 额外一个 W 面板
    const int shm_size =
        GemmConfig::shm_xw + static_cast<int>(sizeof(T)) * 64 * kTileK * kStage +
        GemmConfig::shm_c_alloc + num_group * static_cast<int>(sizeof(int));
    // 防回归：双面板预算必须落在 sm120 每块动态 smem 硬限内（超限的
    // launch 静默失败 → 输出 NaN，见 dispatch_k 的降级注释）
    assert(shm_size <= 101376);

    auto kernel = kernels::group_gemm_gateup_kernel<GemmConfig>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int max_blocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, kNThreads, shm_size);
    if (max_blocks < 1) max_blocks = 1;
    dim3 grid(static_cast<unsigned>(get_sm_count() * max_blocks));

    launch_kernel_pdl(kernel, grid, dim3(kNThreads), shm_size, stream, use_pdl, act_ptr, x_ptr,
                      w_ptr, reinterpret_cast<const int *>(row_indices_ptr),
                      reinterpret_cast<const int *>(seqlens_ptr),
                      reinterpret_cast<const int *>(cu_seqlens_ptr),
                      reinterpret_cast<const int *>(tiles_ptr), n, k, num_group,
                      flat_divider);
  };

  auto dispatch_m = [&](auto t_tag) {
    auto dispatch_k = [&](auto m_tag) {
      // K=128 门控：双 W 面板 + 2 级流水 + sC 预留须落在 96KB 预算内
      //（仅 M32：90KB ✓）；M64/K128 需 104KB、M128/K128 需 148KB，均超
      // sm120 每块动态 smem 硬限 101376B（launch 失败 → 输出未定义），
      // 降级 K64。
      constexpr int kTileM_v = decltype(m_tag)::value;
      constexpr bool k128_fits =
          ((kTileM_v + 2 * 64) * 128 * 2 * 2 +
           (kTileM_v < 64 ? 64 : kTileM_v) * 64 * 2) <= (96 * 1024);
      if (k % 128 == 0 && k128_fits) {
        launch(t_tag, m_tag, Int<128>{});
      } else {
        launch(t_tag, m_tag, Int<64>{});
      }
    };
    if (tile_m >= 128) {
      dispatch_k(Int<128>{});
    } else if (tile_m <= 32) {
      dispatch_k(Int<32>{});
    } else {
      dispatch_k(Int<64>{});
    }
  };

  if (is_half) {
    dispatch_m(config::TypeTag<cutlass::half_t>{});
  } else {
    dispatch_m(config::TypeTag<cutlass::bfloat16_t>{});
  }
}

}  // namespace group_gemm
}  // namespace fuse_moe

#endif  // FUSE_MOE_SRC_SM89_GROUP_GEMM_SM89_CUH_
