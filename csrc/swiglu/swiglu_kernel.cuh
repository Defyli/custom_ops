/*
 * swiglu_kernel.cuh — SwiGLU 融合 GEMM kernel（完全自包含，无 fuse_moe 依赖）
 *
 * y = silu(x @ Wg^T) * (x @ Wu^T)：单个 kernel 完成 dense 配对 GEMM + 激活：
 *   - gate/up 配对 N-tile：同一 CTA 双累加器（tYr_g/tYr_u）同时算 gate/up
 *     两个 64 宽面板，共享同一 X tile 的 smem 装载（X 装载量减半）；
 *   - 激活在 fp32 累加器上完成后直接落盘，无 (M, 2N) gate_up 中间量物化。
 *
 * 与通用 group GEMM 的差异（dense 单问题特化，kernel 不含 group 概念）：
 *   - X 连续无 gather（无 scatter 路径 / row_indices），M 尾部谓词 + ZFILL；
 *   - 任务映射无需调度表设备缓冲（seqlens/cu_seqlens/tiles/cu_tiles 全部
 *     去掉）：单问题下 (itile_m, itile_n) 可从 kernel 标量参数直接算出——
 *     horizon（M-major）用 FastDivmod，vert（N-major）用取模除法；
 *   - 独立算子无上游 kernel 依赖，不含 PDL 指令，普通 launch。
 *
 * 调度序（use_vert）由 op 层按 W footprint vs L2 容量选择（见 swiglu_op.cu）：
 * W ≤ 0.8×L2 → horizon（M-major，X tile 驻留共享）；W > 0.8×L2 → vert
 *（N-major，W 面板驻留、流式 X）——dense 大 N 场景 horizon 的 W 跨 M-band
 * DRAM 重读会成为主导瓶颈。
 *
 * 流水线（跨 tile 连续）：slab 发射流全局连续（issued/read 全局计数，
 * stage = cnt % kStage，task 边界不排空）；主循环双 __syncthreads/迭代
 *（sync#1 数据可见 / sync#2 覆写保护，ring 写读分离）；epilogue 16B
 * 向量化写 gmem（sC 专用区，与在途 cp.async 无冲突）。
 *
 * swapAB 约定：MMA 计算 W @ X^T —— A = 权重 tile (64, kTileK)（W (n, k)
 * 行主序，gate 行在 [0, n/2)、up 行在 [n/2, n)），B = 激活 tile
 * (kTileM, kTileK)，C = (64, kTileM) 转置视图，输出按 y(col, row) 写回
 * (m, n/2) 行主序 gmem。
 */

#ifndef SWIGLU_SRC_SWIGLU_KERNEL_CUH_
#define SWIGLU_SRC_SWIGLU_KERNEL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_types.h"

namespace swiglu {

using namespace cute;  // NOLINT

// ─── Kernel traits ───────────────────────────────────────────────────────────
namespace config {

template <typename T>
struct TypeTag {
  using type = T;
};

// 自有 traits：MMA（sm80 mma.sync 16x8x16，sm80+ 通用，4090D=sm89 /
// 5090=sm120 数值通路一致）/ smem 布局（元素域 swizzle，ldmatrix 友好）/
// cp.async 拷贝类型 / C 预留容量。
template <typename T, int kTileM_, int kTileN_, int kTileK_, int kStage_>
struct SwigluGemmConfig {
  using Tin = T;
  using Tout = T;
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;

  // MMA：sm80 mma.sync 16x8x16（16-bit 输入，fp32 累加）
  using MMA_ATOM = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
                                      SM80_16x8x16_F32BF16BF16F32_TN,
                                      SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom<MMA_ATOM>{},
      make_layout(make_shape(Int<2>{}, Int<4>{}, Int<1>{})),
      Tile<Int<32>, Int<64>, Int<16>>{}));
  static constexpr int kNThreads = decltype(size(TiledMMA{}))::value;  // 256
  static_assert(kNThreads == 256, "swiglu gemm assumes 8-warp MMA tiling");
  static_assert(kTileM_ % 32 == 0 && kTileN_ % 64 == 0 &&
                    (kTileK_ == 32 || kTileK_ % 64 == 0),
                "tile shape must match MMA tiling (M%32, N%64, K==32|K%64)");

  // smem 布局：K-major swizzle atom (8, 64)（16-bit 元素域 Swizzle<3,3,3>，
  // ldmatrix 友好；kTileK=128 由 tile_to_shape 扩展）。K32 细 slab：32
  // 元素（64B）行不满 128B swizzle atom —— 改用 +8 元素 pad 的 plain
  // 布局（行距 40 元素 = 80B：16B 对齐保持，行间 bank 互异），代价
  // smem +25%/级，换 S3→S6 深流水（DRAM 延迟掩盖）+ mma 依赖链减半。
  // 注：先展开为别名再 conditional_t 选择——decltype 内的 braced-init
  // 直接嵌在模板实参里会被 nvcc 拒绝（type name is not allowed）。
  using SwizzleAtomK64 = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<64>{}), make_stride(Int<64>{}, Int<1>{}))));
  using PadAtomK32 = decltype(make_layout(make_shape(Int<8>{}, Int<32>{}),
                                           make_stride(Int<40>{}, Int<1>{})));
  using SmemLayoutAtom = std::conditional_t<kTileK_ == 32, PadAtomK32, SwizzleAtomK64>;
  using SmemLayoutX = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTileM_>{}, Int<kTileK_>{}, Int<kStage_>{})));
  using SmemLayoutW = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<kTileN_>{}, Int<kTileK_>{}, Int<kStage_>{})));
  // C：(kTileN, kTileM)，mode0 连续、无 swizzle；M-stride = kTileN+8（=72）
  // 填充消 bank 冲突（r2s 的 warp 内访问模式下 stride=64 会 8 路冲突；
  // 72 → 144B 行距保持 16B 对齐，bank 随 m 维展开互异）
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
  // sC 预留区按 (kTileN, max(kTileM, 64)) 分配：TiledMMA 的 C tile 覆盖面
  // 恒为 N=64 × M=64（与 kTileM 无关），kTileM<64 时 r2s 分区会写到 sC
  // 逻辑边界之外（按同 stride 布局计址）。s2g 只读 [0, cosize(SmemLayoutC))，
  // 越界写部分属被行谓词丢弃的 M 尾部垃圾行，语义无关。
  static constexpr int shm_c_pad_elems =
      (kTileN_ + 8) * ((kTileM_ < 64 ? 64 : kTileM_) - 1) + kTileN_;
  static constexpr int shm_c_alloc =
      (shm_c > static_cast<int>(sizeof(T) * shm_c_pad_elems))
          ? shm_c
          : static_cast<int>(sizeof(T) * shm_c_pad_elems);
};

}  // namespace config

// ─── 工具 ────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

inline int get_sm_count() {
  static int num_sm = -1;
  if (num_sm == -1) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    num_sm = prop.multiProcessorCount;
  }
  return num_sm;
}

// ─── 核心 kernel ─────────────────────────────────────────────────────────────
// dense gate/up 配对 + silu·mul epilogue。W (n, k) 行主序：gate 行在
// [0, n/2)、up 行在 [n/2, n)；输出 y (m, n/2)。M 任意正整数（M 尾部谓词），
// n % 128、k % 64（op 层校验）。flat_divider 除数 = (n/2)/64（配对 tile 数）。
namespace kernels {

template <typename Config>
__global__ void __launch_bounds__(Config::kNThreads, 1)
swiglu_gateup_kernel(void *Yptr, const void *Xptr, const void *Wptr, int use_vert,
                     int n, int k, int m, cutlass::FastDivmod flat_divider,
                     int f_full, int s_split, float *ws_ptr, int *cnt_ptr) {
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

  const int act_cols = n / 2;                    // N（输出行宽）
  const int up_tile_base = act_cols / kTileN;    // up 面板 N-tile 基（N/64）

  // smem: [sX (M,K,S)][sWg (64,K,S)][sWu (64,K,S)][sC 预留区]——sC 为独立
  // 专用区（不别名 operand 基地址）：task 边界的 epilogue 期间流水继续向
  // operand stage 发射下一 task 的 cp.async（跨 tile 连续），别名会冲突。
  extern __shared__ uint8_t shm_data[] alignas(128);
  Tin *xshm = reinterpret_cast<Tin *>(shm_data);
  Tin *wgshm = xshm + cosize(typename Config::SmemLayoutX{});
  Tin *wushm = wgshm + cosize(typename Config::SmemLayoutW{});
  Tout *cshm = reinterpret_cast<Tout *>(wushm + cosize(typename Config::SmemLayoutW{}));

  int idx = threadIdx.x;
  int iblock = blockIdx.x;
  const int num_tile_m = (m + kTileM - 1) / kTileM;

  auto sX = make_tensor(make_smem_ptr(xshm), typename Config::SmemLayoutX{});   // (M,K,S)
  auto sWg = make_tensor(make_smem_ptr(wgshm), typename Config::SmemLayoutW{});  // (N,K,S)
  auto sWu = make_tensor(make_smem_ptr(wushm), typename Config::SmemLayoutW{});
  auto sC = make_tensor(make_smem_ptr(cshm), typename Config::SmemLayoutC{});  // (N,M)

  // MMA / s2r 句柄（tile 循环外构造一次；swapAB：A=W, B=X, C=(N, M)）
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  auto tWr_g = thr_mma.partition_fragment_A(sWg(_, _, _0{}));  // (MMA, MMA_N, MMA_K)
  auto tWr_u = thr_mma.partition_fragment_A(sWu(_, _, _0{}));
  auto tXr = thr_mma.partition_fragment_B(sX(_, _, _0{}));  // (MMA, MMA_M, MMA_K)
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

  // g2s 句柄（X 用 copy_if 谓词拷贝，W 用 cute copy）
  typename Config::G2SCopy g2s_copy;
  auto g2s_thr_copy = g2s_copy.get_slice(idx);
  auto tsX_copy = g2s_thr_copy.partition_D(sX);  // (CPY, CPY_M, CPY_K, stage)
  auto tsWg_copy = g2s_thr_copy.partition_D(sWg);
  auto tsWu_copy = g2s_thr_copy.partition_D(sWu);

  // r2s / s2g 句柄（sC 无 swizzle，UniversalCopy 标量粒度；每 tile 一次）
  using R2SCopyAtomC = Copy_Atom<UniversalCopy<Tout>, Tout>;
  auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto thr_copy_c = tiled_copy_c.get_slice(idx);
  auto tCs4g = thr_copy_c.partition_D(sC);
  // split-K partial 的 r2g/g2r 句柄（fp32 直写 workspace，不经 sC——
  // sC 与在途 cp.async 的专用区分工不变）
  auto tiled_copy_cf =
      make_tiled_copy_C(Copy_Atom<UniversalCopy<float>, float>{}, tiled_mma);
  auto thr_copy_cf = tiled_copy_cf.get_slice(idx);
  __shared__ int s_last;  // split 任务本 CTA 是否最后到达（全线程广播）

  const int ntile = k / kTileK;

  // ── 任务流 + 全局 slab 流水（跨 tile 连续）───────────────────────────
  // issued/read 为全局计数（stage = cnt % kStage），发射流在 task 边界
  // 不排空：compute 消费当前 task 尾部 slab 时，发射已领先 kStage-1
  // 进入下一 task——epilogue 期间下一 task 的 prologue slab 已在途。
  //
  // wait 语义（迭代内顺序：issue → fence → wait）：slab r 在迭代
  // r-kStage+1 发射且被同迭代 fence commit，到其 wait（迭代 r）之间恰有
  // kStage-1 个 fence——wait<kStage-1> 保证 r 落地（空 commit group 计入
  // 未完成组数）。issue 必须位于 fence 之前。
  //
  // ring 安全（双 sync/迭代）：迭代 i 的发射写 stage (i+kStage-1)%kStage
  // = (i-1)%kStage（上一迭代 ldmatrix 读取的 stage）——sync#1（wait 后）
  // 保证本 slab 全线程可见，sync#2（compute 后）保证读完成后下一迭代
  // 的发射才可覆写。
  struct TaskCtx {
    int valid;
    int itile_m, itile_n;
    int kb, ke;  // K-slab 范围（split 半区任务为半程）
    int part;    // -1 = 整任务；0/1 = split K 前/后半
    int tidx;    // 任务序号（split counter 索引 = tidx - f_full）
  };
  auto pull_task = [&]() -> TaskCtx {
    TaskCtx t{};
    // slot 枚举：前 f_full 个 slot = 整任务；其后 2*s_split 个 slot =
    // 末尾 s_split 个任务的前/后 K 半区（尾部填充式 split-K，见 op 层）
    const int slot = iblock;
    iblock += gridDim.x;
    int tk;
    if (slot < f_full) {
      tk = slot;
      t.part = -1;
      t.kb = 0;
      t.ke = ntile;
    } else if (slot < f_full + 2 * s_split) {
      const int j = slot - f_full;
      tk = f_full + (j >> 1);
      t.part = j & 1;
      t.kb = (t.part == 0) ? 0 : (ntile >> 1);
      t.ke = (t.part == 0) ? (ntile >> 1) : ntile;
    } else {
      return t;  // 流耗尽（valid = 0）
    }
    t.tidx = tk;
    if (use_vert) {
      // vert（N-major）：同 N-pair 的连续 M-tile 落在相邻 CTA（共享同一
      // W 面板，L2 驻留复用）；耗尽 = itile_n 越过配对 tile 数。
      t.itile_m = tk % num_tile_m;
      t.itile_n = tk / num_tile_m;
      t.valid = (t.itile_n < flat_divider.divisor);
    } else {
      // horizon（M-major）：同 M-tile 的连续 N-pair 落在相邻 CTA（共享
      // 同一 X tile）；耗尽 = itile_m 越过 M tile 数。
      int itile_m_total;
      flat_divider(itile_m_total, t.itile_n, tk);
      t.itile_m = itile_m_total;
      t.valid = (itile_m_total < num_tile_m);
    }
    return t;
  };

  // K-slab 装载（按发射时刻的 task 上下文；gmem 分区/谓词随 task 变化，
  // 在此重算——layout 代数，代价可忽略）
  auto load_slab = [&](TaskCtx t, int itile_k, int istage) {
    Tensor W = make_tensor(make_gmem_ptr((Tin const *)Wptr),
                           make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor gWg = local_tile(W, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(t.itile_n, _));
    Tensor gWu = local_tile(W, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(up_tile_base + t.itile_n, _));
    auto tgWg_copy = g2s_thr_copy.partition_S(gWg);
    auto tgWu_copy = g2s_thr_copy.partition_S(gWu);
    // X：连续 (m, k)，M 尾部谓词 + ZFILL（K 维整除，无谓词）；ZFILL 原子
    // 把越界行补零
    Tensor A = make_tensor(make_gmem_ptr((Tin const *)Xptr),
                           make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                           make_coord(t.itile_m, _));
    auto tgA_copy = g2s_thr_copy.partition_S(gA);
    auto tIA = g2s_thr_copy.partition_S(
        make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto pred_x = make_tensor<bool>(shape(tIA));
#pragma unroll
    for (int i = 0; i < size(tIA); ++i) {
      pred_x(i) = t.itile_m * kTileM + get<0>(tIA(i)) < m;
    }
    cute::copy_if(g2s_copy, pred_x, tgA_copy(_, _, _, itile_k), tsX_copy(_, _, _, istage));
    cute::copy(g2s_copy, tgWg_copy(_, _, _, itile_k), tsWg_copy(_, _, _, istage));
    cute::copy(g2s_copy, tgWu_copy(_, _, _, itile_k), tsWu_copy(_, _, _, istage));
  };

  TaskCtx Tc{}, Tn{};
  int iss_kk = 0;            // 发射流在当前发射 task 内的 K-slab 位置
  bool iss_is_next = false;  // 发射 task 已推进到 Tn？
  int issued = 0;            // 全局已发射 slab 数（stage = issued % kStage）
  int slab_read = 0;         // 全局已消费 slab 数（stage = slab_read % kStage）

  Tc = pull_task();
  Tn = pull_task();
  iss_kk = Tc.valid ? Tc.kb : 0;

  // 发射一步：先发当前 compute task 的剩余 slab，耗尽后切到 Tn（cap：
  // 最多领先 compute 一个 task，小 ntile 场景流水深度受限，可接受）；
  // 切换时位置重置到 Tn 的 kb（split 后半区起点非 0）
  auto issue_step = [&]() {
    if (iss_kk >= Tc.ke && !iss_is_next) {
      iss_is_next = true;
      iss_kk = Tn.kb;
    }
    const TaskCtx t = iss_is_next ? Tn : Tc;
    if (!t.valid || iss_kk >= t.ke) return;
    load_slab(t, iss_kk, issued % kStage);
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
    clear(tYr_g);
    clear(tYr_u);
#pragma unroll 1
    for (int kk = Tc.kb; kk < Tc.ke; ++kk) {
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
        cute::gemm(tiled_mma, tWr_g(_, _, ik), tXr(_, _, ik), tYr_g);
        cute::gemm(tiled_mma, tWr_u(_, _, ik), tXr(_, _, ik), tYr_u);
      }
      ++slab_read;

      __syncthreads();  // sync#2：读取完成，下一迭代的发射才可覆写该 stage
    }

    // 融合 epilogue（整任务直接调用；split 半区由最后到达 CTA 归约后
    // 调用）。⚠️ fragment 配对正确性依赖：tYr_g/tYr_u 来自同一
    // tiled_mma + 同一 B 分区的两次 gemm ⇒ 索引 i 映射到相同 (m_row,
    // n_col)，仅 W 数据源（gate/up 两面板）不同——silu(gate)·up（fp32）
    // → sC（专用区，与在途 cp.async 无冲突；下一 task 的 prologue slab
    // 此刻已在途）→ 行谓词 16B 向量化写 y（N%64==0 列恒在界内）
    auto fused_epilogue = [&]() {
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
      Tout *y_base = reinterpret_cast<Tout *>(Yptr);
      const int col_base = Tc.itile_n * kTileN;
      const int row_base = Tc.itile_m * kTileM;
#pragma unroll 1
      for (int i = idx; i < kTileM * nvec; i += kNThreads) {
        const int r = i / nvec;
        const int c8 = (i % nvec) * kVecElems;
        if (row_base + r < m) {
          *reinterpret_cast<uint4 *>(y_base + uint64_t(row_base + r) * act_cols + col_base + c8) =
              *reinterpret_cast<const uint4 *>(&sC(c8, r));
        }
      }
    }
    __syncthreads();  // sC 读取完成（下一 task 的 r2s 覆写前分隔）
    };

    if (Tc.part < 0) {
      fused_epilogue();
    } else {
      // split 半区：fp32 累加器直写本 slot 的 gate/up 两个 (kTileN,
      // kTileM) panel；atomic counter 到 2 的最后到达 CTA 读回两个半区
      // 求和 → 融合 epilogue。写方 fence+atomic（release）/ 读方 fence
      //（acquire）保证 partial 跨 CTA 可见（threadFenceReduction 模式）。
      // M 尾部垃圾行由 X 的 ZFILL 保证为 0，最终写仍按行谓词。
      const int jslot = 2 * (Tc.tidx - f_full) + Tc.part;
      float *ws = ws_ptr + (size_t)jslot * (2 * kTileN * kTileM);
      Tensor gPg = make_tensor(make_gmem_ptr(ws),
                               make_shape(Int<kTileN>{}, Int<kTileM>{}),
                               make_stride(Int<1>{}, Int<kTileN>{}));
      Tensor gPu = make_tensor(make_gmem_ptr(ws + kTileN * kTileM),
                               make_shape(Int<kTileN>{}, Int<kTileM>{}),
                               make_stride(Int<1>{}, Int<kTileN>{}));
      cute::copy(tiled_copy_cf, thr_copy_cf.retile_S(tYr_g),
                 thr_copy_cf.partition_D(gPg));
      cute::copy(tiled_copy_cf, thr_copy_cf.retile_S(tYr_u),
                 thr_copy_cf.partition_D(gPu));
      __syncthreads();  // 本 CTA partial 写完成（全线程）
      // ⚠️ 全员 fence（release）：partial 由全部线程分散写，仅 thread0
      // fence 不覆盖其他线程的写——最后到达 CTA 会偶发读到 stale
      // partial（对照 CUDA threadFenceReduction 官方模式）。
      __threadfence();
      if (threadIdx.x == 0) {
        s_last = (atomicAdd(cnt_ptr + (Tc.tidx - f_full), 1) == 1);
      }
      __syncthreads();
      if (s_last) {
        auto read_partial = [&](int j, auto &accg, auto &accu) {
          float *pw = ws_ptr + (size_t)j * (2 * kTileN * kTileM);
          Tensor pG = make_tensor(make_gmem_ptr(pw),
                                  make_shape(Int<kTileN>{}, Int<kTileM>{}),
                                  make_stride(Int<1>{}, Int<kTileN>{}));
          Tensor pU = make_tensor(make_gmem_ptr(pw + kTileN * kTileM),
                                  make_shape(Int<kTileN>{}, Int<kTileM>{}),
                                  make_stride(Int<1>{}, Int<kTileN>{}));
          // C-copy 的规范方向是 fragment→memory：fragment 恒为 S 侧
          //（retile_S）、memory 恒为 D 侧（partition_D）——与 A/B copy
          //（s2r：partition_S(smem)→retile_D(reg)）相反。反向使用时
          // src=partition_D(mem)、dst=retile_S(frag)
          cute::copy(tiled_copy_cf, thr_copy_cf.partition_D(pG),
                     thr_copy_cf.retile_S(accg));
          cute::copy(tiled_copy_cf, thr_copy_cf.partition_D(pU),
                     thr_copy_cf.retile_S(accu));
        };
        const int j0 = 2 * (Tc.tidx - f_full);
        auto sum_g = make_tensor<float>(shape(tYr_g));
        auto sum_u = make_tensor<float>(shape(tYr_u));
        read_partial(j0, tYr_g, tYr_u);
        read_partial(j0 + 1, sum_g, sum_u);
#pragma unroll
        for (int i = 0; i < size(tYr_g); ++i) {
          tYr_g(i) += sum_g(i);
          tYr_u(i) += sum_u(i);
        }
        fused_epilogue();
      }
    }

    // task 边界：compute task 前移；拉取新 Tn。iss_kk 保留（发射流在新
    // Tc（原 Tn）内的位置）；iss_is_next 复位后发射继续填补新 Tc 的剩余
    // slab，耗尽再切新 Tn。
    Tc = Tn;
    Tn = pull_task();
    iss_is_next = false;
  }
}

}  // namespace kernels

// ─── host 侧入口 ─────────────────────────────────────────────────────────────
// 约束（op 层已校验）：n % 128 == 0（n = 2N），k % 32 == 0；tile_m ∈
// {32, 64, 128}，tile_k ∈ {32, 64, 128}（K32 = 细 slab 深流水：S3→S6，
// mma 依赖链减半；padded smem atom 见 traits）。独立算子无上游 kernel，
// 普通 launch（无 PDL 属性）。
// 调度序（use_vert）由 op 层按 W footprint vs L2 容量选择；kStage 深度由
// stage_override 控制（0 = smem 预算内最深；2 = 浅流水换 2 CTA/SM 占用）。
inline void swiglu_dense_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                               bool use_vert, int m, int n, int k, int tile_m,
                               int tile_k, bool is_half, int stage_override,
                               float *ws_ptr, int *cnt_ptr, int s_split,
                               cudaStream_t stream) {
  using namespace cute;  // NOLINT

  assert(n % 128 == 0 && "swiglu: n (=2*intermediate) must be a multiple of 128");
  assert(k % 32 == 0 && "swiglu: k must be a multiple of 32");
  assert(m > 0);
  assert(tile_m == 32 || tile_m == 64 || tile_m == 128);
  assert(tile_k == 32 || tile_k == 64 || tile_k == 128);

  cutlass::FastDivmod flat_divider((n / 2) / 64);  // gate/up 配对 tile 数
  // 尾部填充式 split-K：末尾 s_split 个任务各劈 K 前后两半（s_split=0
  // 关闭）；slot = f_full 整任务 + 2*s_split 半区。counter/workspace 由
  // op 层按 wave 尾部 idle 数决定并分配。
  const int f_full = ((m + tile_m - 1) / tile_m) * ((n / 2) / 64) - s_split;
  assert(s_split >= 0 && f_full >= 0);

  auto launch = [&](auto t_tag, auto m_tag, auto k_tag, auto s_tag) {
    using T = typename decltype(t_tag)::type;
    constexpr int kTileM = decltype(m_tag)::value;
    constexpr int kTileK = decltype(k_tag)::value;
    constexpr int kStage = decltype(s_tag)::value;
    using GemmConfig = config::SwigluGemmConfig<T, kTileM, 64, kTileK, kStage>;

    constexpr int kNThreads = GemmConfig::kNThreads;
    // operands = X + 2×W 面板 = shm_xw + 额外一个 W 面板（无调度表）；
    // 第二面板按 cosize(SmemLayoutW) 计（K32 padded 布局行距 40 > kTileK）
    const int shm_size =
        GemmConfig::shm_xw +
        static_cast<int>(sizeof(T)) *
            cosize(typename GemmConfig::SmemLayoutW{}) +
        GemmConfig::shm_c_alloc;
    // 防回归：双面板预算必须落在 sm120 每块动态 smem 硬限内（超限的
    // launch 静默失败 → 输出 NaN）
    assert(shm_size <= 101376);

    auto kernel = kernels::swiglu_gateup_kernel<GemmConfig>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    int max_blocks = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, kNThreads, shm_size);
    if (max_blocks < 1) max_blocks = 1;
    dim3 grid(static_cast<unsigned>(get_sm_count() * max_blocks));

    kernel<<<grid, dim3(kNThreads), shm_size, stream>>>(y_ptr, x_ptr, w_ptr,
                                                         use_vert ? 1 : 0, n, k, m,
                                                         flat_divider, f_full, s_split,
                                                         ws_ptr, cnt_ptr);
  };

  // kStage 深度选择：默认按 smem 预算取最深（X + 双 W 面板 + sC 预留，
  // 96KB 预算：M64/K64→3、M32/K64→4、M128/K64→2、K128 仅 M32→2）；
  // stage_override=2 时降为 2 级浅流水——M32 下 smem 48KB → 2 CTA/SM，
  // 任务数 ≤ 2×SM 时单 wave 完成（tiny-M decode 消除 1.5-wave 尾部，
  // 计算密度低浅流水无损）。
  auto launch_stage = [&](auto t_tag, auto m_tag, auto k_tag) {
    constexpr int kTileM = decltype(m_tag)::value;
    constexpr int kTileK = decltype(k_tag)::value;
    constexpr int kStageDeep = [] {
      constexpr int kCols = (kTileK == 32) ? 40 : kTileK;              // K32 padded 行宽
      constexpr int bps = (kTileM + 2 * 64) * kCols * 2;               // X + 双 W 面板
      constexpr int sc = ((kTileM < 64 ? 64 : kTileM) - 1) * 72 * 2 + 64 * 2;  // sC 预留区
      constexpr int s = (96 * 1024 - sc) / bps;
      return (s > 6) ? 6 : ((s < 2) ? 2 : s);
    }();
    if (stage_override == 2 && kStageDeep > 2) {
      launch(t_tag, m_tag, k_tag, Int<2>{});
    } else {
      launch(t_tag, m_tag, k_tag, Int<kStageDeep>{});
    }
  };

  auto dispatch_m = [&](auto t_tag) {
    auto dispatch_k = [&](auto m_tag) {
      // K=128 门控：双 W 面板 + 2 级流水 + sC 预留须落在 96KB 预算内
      //（仅 M32）；M64/K128 需 104KB、M128/K128 需 148KB，均超 sm120 每
      // 块动态 smem 硬限，降级 K64。
      constexpr int kTileM_v = decltype(m_tag)::value;
      constexpr bool k128_fits =
          ((kTileM_v + 2 * 64) * 128 * 2 * 2 +
           ((kTileM_v < 64 ? 64 : kTileM_v) - 1) * 72 * 2 + 64 * 2) <= (96 * 1024);
      if (k % 128 == 0 && k128_fits && tile_k == 128) {
        launch_stage(t_tag, m_tag, Int<128>{});
      } else if (k % 64 == 0 && tile_k == 64) {
        launch_stage(t_tag, m_tag, Int<64>{});
      } else {
        launch_stage(t_tag, m_tag, Int<32>{});
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

// ─── decode GEMV 路径（M ≤ 8）─────────────────────────────────────────────────
// y = silu(x @ Wg^T) * (x @ Wu^T) 的小 M 特化：纯 W 流式（W 恰好读一遍，
// DRAM 带宽主导）。mma 路径（tile32）在 M=1 实测 826 GB/s（tile 化装载 +
// 流水 barrier 开销），本路径 16B 向量装载 + warp shuffle 归约 + 寄存器内
// gate/up 融合，目标 ~950 GB/s（DRAM 峰值的 ~95%）。
//
// 任务划分：每 block 32 个行对（8 warp × 4 对），gate 行 pair 与 up 行
// N+pair 由同一 warp 持有——dot 完成后在寄存器内直接 silu·mul，无跨 warp
// 通信。x 经 L1 广播复用（M ≤ 8 时 x ≤ 64KB，L1=128KB 驻留）；M 个累加器
// 以 #pragma unroll + 谓词展开（禁止 break——部分展开会把运行时下标泄到
// local memory）。
namespace gemv {

template <typename T>
__global__ void __launch_bounds__(256)
swiglu_gemv_kernel(void *Yptr, const void *Xptr, const void *Wptr,
                   int m, int n, int k, int pairs_per_warp) {
  using Ti = T;
  const int N = n / 2;  // 输出行宽（gate 行 [0, N)，up 行 [N, 2N)）
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int pair0 = blockIdx.x * (8 * pairs_per_warp) + warp * pairs_per_warp;
  const int kvec = k >> 3;  // 每行 16B 向量数（k % 8 == 0 由 k % 32 保证）

  const Ti *X = reinterpret_cast<const Ti *>(Xptr);
  const Ti *W = reinterpret_cast<const Ti *>(Wptr);
  Ti *Y = reinterpret_cast<Ti *>(Yptr);

  for (int pp = 0; pp < pairs_per_warp; ++pp) {
    const int pair = pair0 + pp;
    if (pair >= N) return;
    const Ti *wg = W + static_cast<size_t>(pair) * k;
    const Ti *wu = W + static_cast<size_t>(N + pair) * k;

    float ag[8], au[8];
#pragma unroll
    for (int mm = 0; mm < 8; ++mm) {
      ag[mm] = 0.0f;
      au[mm] = 0.0f;
    }
    // unroll x4：装载级并行——串行 v 循环每 warp 仅 2 个 16B 装载在途，
    // 24 warp/SM 只能到 ~890 GB/s，x4 后可饱和 DRAM
#pragma unroll 4
    for (int v = lane; v < kvec; v += 32) {
      const uint4 wgv = *reinterpret_cast<const uint4 *>(wg + static_cast<size_t>(v) * 8);
      const uint4 wuv = *reinterpret_cast<const uint4 *>(wu + static_cast<size_t>(v) * 8);
      const Ti *wge = reinterpret_cast<const Ti *>(&wgv);
      const Ti *wue = reinterpret_cast<const Ti *>(&wuv);
#pragma unroll
      for (int mm = 0; mm < 8; ++mm) {
        if (mm < m) {
          const uint4 xv =
              *reinterpret_cast<const uint4 *>(X + static_cast<size_t>(mm) * k +
                                               static_cast<size_t>(v) * 8);
          const Ti *xe = reinterpret_cast<const Ti *>(&xv);
          float pg = 0.0f, pu = 0.0f;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            pg += static_cast<float>(wge[e]) * static_cast<float>(xe[e]);
            pu += static_cast<float>(wue[e]) * static_cast<float>(xe[e]);
          }
          ag[mm] += pg;
          au[mm] += pu;
        }
      }
    }
    // warp 归约 + 寄存器内融合 epilogue（lane 0 散落 2B store，量级 M×N 微小）
#pragma unroll
    for (int mm = 0; mm < 8; ++mm) {
      if (mm < m) {
        float g = ag[mm], u = au[mm];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          g += __shfl_xor_sync(0xffffffffu, g, off);
          u += __shfl_xor_sync(0xffffffffu, u, off);
        }
        if (lane == 0) {
          Y[static_cast<size_t>(mm) * N + pair] = static_cast<Ti>(silu(g) * u);
        }
      }
    }
  }
}

// M=1 特化：通用 kernel 的热循环按运行时 m 谓词展开 8 个 mm 分支
//（M=1 时 7/8 空转——ncu 实测 issue 45% / long_scoreboard 69%，空分支
// 占用 issue 槽与寄存器预算 ag[8]+au[8]）。特化后单累加器 + 零谓词，
// 省出的寄存器换 unroll 6 装载深度；__restrict__ 消除别名保守性。
// ncu 基线（(1,4096,11008)，W=180MB 纯 DRAM 流）：881 GB/s vs eager
// cuBLAS 917（DRAM 峰值 1008 的 87% vs 91%）。
template <typename T>
__global__ void __launch_bounds__(256)
swiglu_gemv1_kernel(void *__restrict__ Yptr, const void *__restrict__ Xptr,
                    const void *__restrict__ Wptr, int n, int k,
                    int pairs_per_warp) {
  const int N = n / 2;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int pair0 = blockIdx.x * (8 * pairs_per_warp) + warp * pairs_per_warp;
  const int kvec = k >> 3;

  const T *__restrict__ X = reinterpret_cast<const T *>(Xptr);
  const T *__restrict__ W = reinterpret_cast<const T *>(Wptr);
  T *__restrict__ Y = reinterpret_cast<T *>(Yptr);

  for (int pp = 0; pp < pairs_per_warp; ++pp) {
    const int pair = pair0 + pp;
    if (pair >= N) return;
    const T *__restrict__ wg = W + static_cast<size_t>(pair) * k;
    const T *__restrict__ wu = W + static_cast<size_t>(N + pair) * k;

    float g = 0.0f, u = 0.0f;
#pragma unroll 4
    for (int v = lane; v < kvec; v += 32) {
      const uint4 wgv = *reinterpret_cast<const uint4 *>(wg + static_cast<size_t>(v) * 8);
      const uint4 wuv = *reinterpret_cast<const uint4 *>(wu + static_cast<size_t>(v) * 8);
      const uint4 xv = *reinterpret_cast<const uint4 *>(X + static_cast<size_t>(v) * 8);
      const T *wge = reinterpret_cast<const T *>(&wgv);
      const T *wue = reinterpret_cast<const T *>(&wuv);
      const T *xe = reinterpret_cast<const T *>(&xv);
      float pg = 0.0f, pu = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        pg += static_cast<float>(wge[e]) * static_cast<float>(xe[e]);
        pu += static_cast<float>(wue[e]) * static_cast<float>(xe[e]);
      }
      g += pg;
      u += pu;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      g += __shfl_xor_sync(0xffffffffu, g, off);
      u += __shfl_xor_sync(0xffffffffu, u, off);
    }
    if (lane == 0) {
      Y[pair] = static_cast<T>(silu(g) * u);
    }
  }
}

}  // namespace gemv

// host 入口（op 层在 M ≤ 8 时选择；约束同 dense 路径：n % 128、k % 32）
inline void swiglu_gemv_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                              int m, int n, int k, bool is_half,
                              cudaStream_t stream) {
  assert(m >= 1 && m <= 8);
  const int N = n / 2;
  // 自适应 pairs/warp：目标 block 数 >= ~3x SM（流式 kernel 需足量并发
  // 装载）；大 N 用 4 对/warp（32 对/block），小 N 降到 1 对/warp（8 对
  // /block，block 数 x4）。N 是 64 的倍数保证整除。
  static const int sm_count = get_sm_count();
  int ppw = 1;
  for (int p = 4; p >= 1; p >>= 1) {
    if (N / (8 * p) >= 3 * sm_count) {
      ppw = p;
      break;
    }
  }
  const int blocks = N / (8 * ppw);
  assert(blocks >= 1);
  if (m == 1) {
    if (is_half) {
      gemv::swiglu_gemv1_kernel<cutlass::half_t>
          <<<blocks, 256, 0, stream>>>(y_ptr, x_ptr, w_ptr, n, k, ppw);
    } else {
      gemv::swiglu_gemv1_kernel<cutlass::bfloat16_t>
          <<<blocks, 256, 0, stream>>>(y_ptr, x_ptr, w_ptr, n, k, ppw);
    }
  } else if (is_half) {
    gemv::swiglu_gemv_kernel<cutlass::half_t>
        <<<blocks, 256, 0, stream>>>(y_ptr, x_ptr, w_ptr, m, n, k, ppw);
  } else {
    gemv::swiglu_gemv_kernel<cutlass::bfloat16_t>
        <<<blocks, 256, 0, stream>>>(y_ptr, x_ptr, w_ptr, m, n, k, ppw);
  }
}

}  // namespace swiglu

#endif  // SWIGLU_SRC_SWIGLU_KERNEL_CUH_
