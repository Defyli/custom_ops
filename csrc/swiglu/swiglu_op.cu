/*
 * swiglu_op.cu — SwiGLU 融合算子，PyTorch C++ 接口实现（独立算子，与
 * fuse_moe 完全解耦：不包含任何 fuse_moe 头，kernel 见同目录
 * swiglu_kernel.cuh）
 *
 * y = silu(x @ Wg^T) * (x @ Wu^T)：单个 kernel 完成配对 GEMM + 激活
 *（无 gate_up (M,2N) 中间量物化——省 2×M×N×2B DRAM 往返与独立激活
 * kernel；X tile 装载量减半——gate/up 共享同一 smem 副本）。
 *
 * 调度序选择（W footprint vs L2 容量）在本层完成：W ≤ 0.8×L2 →
 * horizon（M-major：X tile 驻留共享，W 也可 L2 驻留）；W > 0.8×L2 →
 * vert（N-major：W 面板驻留、X 流式但 X 通常远小于 L2）——否则
 * horizon 的 W 跨 M-band DRAM 重读（每 band 一份完整 W）会压过算力
 * 成为主导瓶颈（Llama-7B FFN 实测：W=180MB ≫ 72MB L2，horizon 106 TF
 * vs cuBLAS 137 TF，差距主要来自 W DRAM 流量）。
 *
 * tile/kStage 策略（dense 专属，4090D 实测定标，详见 swiglu_pick_tile_m
 * 与 kernel launch 处注释）：W > 0.4×L2 且 m ≥ 96 → M128（W 的 L2 读流
 * 量 = W × M-band 数，随 tileM 线性下降）；小 W → M64 深流水（S3）；
 * tiny-M → M32 浅流水（S=2）换 2 CTA/SM 占用（消 wave 尾部）。
 * 已知差距（warm 实测）：大 M 1.01-1.07× eager；中段形状原落后 8-18%
 *（wave 量化尾部），现由尾部填充式 split-K（见下方）修复到 0.93-0.99×；
 * M <= 8 decode 走专用 GEMV 路径（见 kernel gemv 命名空间）：4090D 实测
 * 小 N（任务少）快 eager 3 倍+（(1,2048,1024) 18.8us vs 56us）；W ≤ L2
 * 形状 L2 驻留后 2.8 TB/s（(1,4096,4096) 3.17× eager）。M=1 有编译期
 * 特化（swiglu_gemv1_kernel：单累加器 + __restrict__，消除通用路径
 * 8 路 mm 谓词空转——ncu 实测 issue 45%/long_scoreboard 69%）：纯
 * DRAM 流（W=180MB > L2）881→903 GB/s（0.98× eager，DRAM 峰值 90%）；
 * 剩余 ~2% 与 cuBLAS gemv（922）的差距为 DRAM 流效率极限，已证伪
 * unroll 6/8（16 迭代不均 + 寄存器压力，-1.5%）。
 *
 * ncu 实测结论（4090D，sudo ncu 计数器权限）：所有 tile 配置 1 CTA/SM
 *（M128 regs=152 / M64 smem 81KB 双重限制，8 warp = 16.7% 占用），tensor
 * pipe 44% / 内存管道 34-48% 双不饱和——延迟受限但与 cuBLAS（同样 ~8
 * warp/SM）的 49% 相当；剩余差距由 wave 量化尾部精确解释（688 任务 =
 * 6.04 wave → 86% 效率 ↔ 实测 0.89x，512 任务 = 4.49 → 90% ↔ 0.94x）。
 * 尾部填充式 split-K（已实现，4090D 实测全形状 +1.4~+7.1% 无回归）：
 * 仅劈尾部 idle 数个任务（非全局 split），partial fp32 走 L2 驻留
 * workspace（≤15MB），kernel 内 atomic counter 最后到达 CTA 归约。
 * (512,4096,11008) 0.740→0.714ms（0.92→0.95× eager）；(2048,4096,
 * 1024) 0.293→0.274（0.86→0.92×）；(2048,4096,2048) +7.1%。
 * 已证伪/搁置的方向：K32 细 slab 深流水（S3→S5，实测 -3~-14%：padded
 * smem +25% + 双倍 barrier 开销 > 延迟掩盖收益）；朴素全局 split-K
 *（尾部收益 7-10% 被 partial 写读流量吃掉，M512 形状 fp32 workspace
 * 90MB 超 L2——尾部填充式仅劈 idle 数个任务规避此问题）；2 CTA/SM
 *（M64 需 ≤50KB，operands S2=48KB+sC 9.2KB 超限，寄存器直写
 * epilogue 可达但 4B 散落 store 损失 ~2-3%）。
 */

#include "swiglu/swiglu_op.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdlib>
#include <cstring>

#include "cutlass/numeric_types.h"

#include "swiglu/swiglu_kernel.cuh"  // swiglu::swiglu_dense_async

namespace {

// tile 策略（M-tile 高度，W footprint 感知）——dense 专属取舍（与 group
// GEMM 不同：W 的 L2 读流量 = W × M-band 数，与 tileM 强耦合）:
// - W > 0.4×L2 且 m ≥ 96 → 128：W L2 流量随 band 数线性减半（4090D
//   实测 W=33/46/67/180MB 形状 t128 优 0-13%；W > L2 时 vert 调度的
//   面板驻留也依赖大 tile 控制在驻留面板数）；
// - 小 W → 64：W 整体 L2 驻留，流量与 tileM 解耦 → 深流水（S3 vs S2）
//   优先（实测 W≤17MB 形状 t64 恒优 5-8%）；
// - m < 96 → 64（128 tile 行 padding 浪费：m=64 实测 t128 差 24%）、
//   极小 m → 32（padding 最小）。
// SWIGLU_TILE_M=32/64/128 可强制（调优/复现用）。
inline int swiglu_pick_tile_m(int m, bool w_large) {
  int tm;
  if (m <= 32) {
    tm = 32;
  } else if (m < 96) {
    tm = 64;
  } else {
    tm = w_large ? 128 : 64;
  }
  if (const char *env_tm = std::getenv("SWIGLU_TILE_M")) {
    const int v = std::atoi(env_tm);
    if (v == 32 || v == 64 || v == 128) tm = v;
  }
  return tm;
}

}  // namespace

// ── CUDA 实现 ─────────────────────────────────────────────────────────────────
torch::Tensor swiglu_cuda(const torch::Tensor& x, const torch::Tensor& weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x and weight must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "x and weight must be contiguous");
    const bool is_bf16 = (x.scalar_type() == torch::kBFloat16);
    const bool is_fp16 = (x.scalar_type() == torch::kHalf);
    TORCH_CHECK(is_bf16 || is_fp16, "x must be bfloat16 or float16");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (num_tokens, hidden)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (2*intermediate, hidden)");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N2 = weight.size(0);  // 2 * N（gate 在前 up 在后）
    TORCH_CHECK(M > 0 && K > 0 && N2 > 0, "empty input");
    TORCH_CHECK(N2 % 2 == 0, "weight.size(0) must be even (gate+up fused)");
    const int64_t N = N2 / 2;
    TORCH_CHECK(K % 32 == 0, "hidden must be a multiple of 32, got ", K);
    TORCH_CHECK(N % 64 == 0,
                "intermediate (= weight.size(0)/2) must be a multiple of 64, got ", N);

    at::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor out = torch::empty({M, N}, x.options());

    // M ≤ 8 → decode GEMV 路径（纯 W 流式：16B 向量装载 + warp 归约 +
    // 寄存器内 gate/up 融合；mma tile32 路径实测 826 GB/s，GEMV 目标
    // ~950 GB/s）。SWIGLU_GEMV=0 禁用（调优/对比用）；kernel 上限 M=8，
    // 超限形状忽略强制值走 dense 路径。
    bool use_gemv = (M <= 8);
    if (const char *env_gv = std::getenv("SWIGLU_GEMV")) {
        if (std::atoi(env_gv) == 0) use_gemv = false;
    }
    if (use_gemv) {
        swiglu::swiglu_gemv_async(out.data_ptr(), x.data_ptr(), weight.data_ptr(),
                                  /*m=*/static_cast<int>(M), /*n=*/static_cast<int>(N2),
                                  /*k=*/static_cast<int>(K),
                                  /*is_half=*/!is_bf16, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // W footprint vs L2 容量 → 调度序（horizon / vert，见文件头注释）。
    // SWIGLU_SCHED=horizon|vert 强制（调优/复现用）。
    static const int64_t l2_bytes = []() {
        int dev = 0, l2 = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return (int64_t)32 << 20;
        if (cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev) != cudaSuccess) {
            return (int64_t)32 << 20;
        }
        return (int64_t)l2;
    }();
    const int64_t w_bytes = N2 * K * 2;  // bf16/fp16 均 2B
    bool use_vert;
    if (const char *sched_env = std::getenv("SWIGLU_SCHED")) {
        use_vert = (std::strcmp(sched_env, "vert") == 0);
    } else {
        use_vert = w_bytes > (l2_bytes * 4) / 5;  // > 0.8×L2
    }

    // W > 0.4×L2 → 大 tile（见 swiglu_pick_tile_m 注释；阈值来自 4090D
    // W-scan 交叉点：W=17MB 时 t64 优 8%，W=33MB 起持平转 t128 优）
    const int tile_m = swiglu_pick_tile_m(static_cast<int>(M),
                                         w_bytes > (l2_bytes * 2) / 5);

    // kStage 深度：tiny-M（tile32）且任务数超过 SM 数时默认浅流水
    //（S=2）——M32 smem 48KB → 2 CTA/SM，任务 ≤ 2×SM 时单 wave 完成，
    // 消除 1.5-wave 尾部（decode 计算密度低，浅流水无损；任务 ≤ SM 数
    // 时保持深流水，单任务延迟由流水深度主导）。SWIGLU_KSTAGE=0/2 强制。
    static const int sm_count = []() {
        int dev = 0, sms = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 128;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            return 128;
        }
        return sms;
    }();
    const int total_tasks =
        ((static_cast<int>(M) + 31) / 32) * static_cast<int>(N / 64);
    int stage_override = 0;
    if (const char *ks_env = std::getenv("SWIGLU_KSTAGE")) {
        stage_override = std::atoi(ks_env);
    } else if (tile_m == 32 && total_tasks > sm_count) {
        stage_override = 2;
    }

    // K-slab 宽度：默认 64；SWIGLU_TILE_K=32/64/128 强制（调优用）；
    // K 非 64 倍数时强制 32（唯一可行路径）。
    int tile_k = 64;
    if (const char *env_tk = std::getenv("SWIGLU_TILE_K")) {
        const int v = std::atoi(env_tk);
        if (v == 32 || v == 64 || v == 128) tile_k = v;
    }
    if (K % 64 != 0) tile_k = 32;

    // 尾部填充式 split-K（2-way）：wave 量化尾部（idle = waves*grid - T
    // 个 CTA-slot 空转）是中段形状的主要损失（ncu 实测 86-90% 占用效率
    // ↔ 0.85-0.90x eager）。把末尾 s = idle 个任务各劈成 K 前后两半 →
    // slot 数恰为 waves*grid（每 CTA 整数波，尾部归零）；partial（fp32
    // gate+up，每 slot 2*64*tile_m*4B）走 L2 驻留 workspace，kernel 内
    // atomic counter 的最后到达 CTA 归约（无第二 kernel/无等待）。
    // 门控：waves >= 3（2-wave 档 2-way 劈后 ceil 不变，无收益）、
    // idle >= 32（净收益 > 归约流量）、tile_m >= 64（t32 的 r2s 分区会
    // 越界写 partial slot）、tile_k == 64 且 K % 128 == 0（slab 对半；
    // 同时保证 1 CTA/SM 与 grid=sm_count 假设一致）。SWIGLU_SPLITK=0
    // 禁用（调优/对比用）。
    bool use_splitk = true;
    if (const char *env_sk = std::getenv("SWIGLU_SPLITK")) {
        if (std::atoi(env_sk) == 0) use_splitk = false;
    }
    torch::Tensor split_ws, split_cnt;
    int s_split = 0;
    if (use_splitk && tile_k == 64 && tile_m >= 64 && K % 128 == 0) {
        const int sk_ntm = (static_cast<int>(M) + tile_m - 1) / tile_m;
        const int sk_tasks = sk_ntm * static_cast<int>(N / 64);
        const int sk_waves = (sk_tasks + sm_count - 1) / sm_count;
        const int sk_idle = sk_waves * sm_count - sk_tasks;
        if (sk_waves >= 3 && sk_idle >= 32) {
            s_split = sk_idle;
            // 总量 = 2*s_split 个 slot × 每 slot gate+up 两 panel（fp32）
            split_ws = torch::empty({(int64_t)s_split * 4 * 64 * tile_m},
                                    x.options().dtype(torch::kFloat));
            split_cnt = torch::zeros({s_split}, x.options().dtype(torch::kInt));
        }
    }

    swiglu::swiglu_dense_async(out.data_ptr(), x.data_ptr(), weight.data_ptr(),
                               use_vert,
                               /*m=*/static_cast<int>(M), /*n=*/static_cast<int>(N2),
                               /*k=*/static_cast<int>(K), tile_m, tile_k,
                               /*is_half=*/!is_bf16, stage_override,
                               split_ws.defined() ? split_ws.data_ptr<float>() : nullptr,
                               split_cnt.defined() ? split_cnt.data_ptr<int>() : nullptr,
                               s_split, stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ── CPU 桩 ───────────────────────────────────────────────────────────────────
torch::Tensor swiglu_cpu(const torch::Tensor& /*x*/, const torch::Tensor& /*weight*/) {
    TORCH_CHECK(false, "swiglu is only supported on CUDA");
}
