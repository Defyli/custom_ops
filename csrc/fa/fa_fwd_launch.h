/*
 * Flash Attention Forward with Additive Mask — Launch Templates
 *
 * 按架构/场景分发的 host-side 入口：
 *   sm89（RTX 4090）基线：run_mha_fwd_mask_hdim{64,128}
 *     hdim64  : kBlockM=128, kBlockN=128, 4 warps, smem=80KB（动态 smem）
 *     hdim128 : kBlockM=64,  kBlockN=64,  4 warps, smem=56KB
 *   sm120（Blackwell consumer, RTX 50）主路径：run_mha_fwd_mask_hdim{64,128}_sm120
 *     TMA + mbarrier 多级流水（见 fa_fwd_sm120.h 文件头的设计说明）
 *   sm120 persistent：FA_PERSISTENT=1 时启用（大 grid 消 tail 效应，d128 +2~5%）
 *   sm120 split-KV：grid 填不满 SM 时自动启用（cost model 决定 num_splits，
 *     部分结果 bf16 落盘 + combine 归约）；其 Split-M 变体（kBlockM=64）
 *     在极小 grid 时再把 m_block 翻倍
 *
 * 均为 bf16、无 dropout、无 causal（mask 由外部传入，语义同 SDPA：
 * softmax(S·scale + mask)，支持 0/-inf 及任意有限值偏置）。
 */

#pragma once

#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "fa_fwd_kernel.h"
#include "fa_fwd_sm120.h"

namespace FA_MASK_NAMESPACE {

// ── sm120 运行时检测 ─────────────────────────────────────────────────────────
// 编译期：nvcc 预定义宏 __CUDA_ARCH_LIST__（host pass 可见）包含 sm120 目标时，
// 二进制中才存在 sm120 kernel 代码；运行期再确认设备 major==12（Blackwell consumer，
// 如 RTX 5090）。双重检查避免「二进制不含 sm120 代码却分发到空 kernel」的静默错误。
#if defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 1200)
#define FA_MASK_HAS_SM120_KERNEL 1
#endif

inline bool fa_mask_sm120_supported() {
#ifdef FA_MASK_HAS_SM120_KERNEL
    static const bool supported = []() {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
        return major == 12;
    }();
    return supported;
#else
    return false;
#endif
}

// sm120 tile 配置（供 op.cu 计算 mask padding 对齐使用，必须与下方 launcher 一致）
//   hdim=64  : kBlockM=128, kBlockN=64
//   hdim=128 : kBlockM=128, kBlockN=64
// （sm120 每 CTA 动态 smem 上限 101376B）
inline void fa_mask_sm120_block_size(int d, int &kBlockM, int &kBlockN) {
    if (d == 64) { kBlockM = 128; kBlockN = 64; }
    else         { kBlockM = 128; kBlockN = 64; }
}

// ── hdim = 64 ─────────────────────────────────────────────────────────────────
// sm89: non-causal 时 128x128 最优（2 CTAs per SM at 80KB smem on RTX4090）
inline void run_mha_fwd_mask_hdim64(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    // kBlockM=128, kBlockN=128, kNWarps=4
    // smem: Q=16KB + K=16KB + V=16KB + Mask=32KB = 80KB (需要动态 smem)
    run_flash_fwd_with_mask<
        FA_mask_kernel_traits<64, 128, 128, 4, false, false, T>
    >(params, stream);
}

// ── hdim = 128 ────────────────────────────────────────────────────────────────
// sm89: non-causal 时 64x64 最优（可以 2 CTAs per SM）
inline void run_mha_fwd_mask_hdim128(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    // kBlockM=64, kBlockN=64, kNWarps=4
    // smem: Q=16KB + K=16KB + V=16KB + Mask=8KB = 56KB (需要动态 smem)
    run_flash_fwd_with_mask<
        FA_mask_kernel_traits<128, 64, 64, 4, false, false, T>
    >(params, stream);
}

// ── sm120 (Blackwell) 路径：TMA + mbarrier 多级流水 ─────────────────────────
inline void run_mha_fwd_mask_hdim64_sm120(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120<
        FA_mask_kernel_traits_sm120<64, 128, 64, 8, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, T>
    >(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm120(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120<
        FA_mask_kernel_traits_sm120<128, 128, 64, 8, 3, /*MaskInSmem_=*/false, /*QInRegs_=*/true, T>
    >(params, stream);
}

// ── sm120 Persistent kernel 版本 ─────────────────────────────────────────────
// 1D grid（num_SMs × 2），每个 CTA 步长式取任务，消除大 grid 的 tail 效应。
inline void run_mha_fwd_mask_hdim64_sm120_persistent(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120_persistent<
        FA_mask_kernel_traits_sm120<64, 128, 64, 8, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, T>
    >(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm120_persistent(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120_persistent<
        FA_mask_kernel_traits_sm120<128, 128, 64, 8, 3, /*MaskInSmem_=*/false, /*QInRegs_=*/true, T>
    >(params, stream);
}

// ── Split-KV num_splits cost model（5090D 全 shape 实测拟合）──────────────────
// 目标：小 grid（B*H*num_m_blocks ≪ num_SMs）时按 K 维切分提升并行度，
// 同时避免过多 split 带来的读写放大（部分结果 bf16 落盘 + combine 回读）。
// 与 FA2/FA3 的 waves-efficiency 启发式不同，这里是实测拟合的解析 cost model，
// 大 grid 自动返回 1（无 split 开销）。
// 环境变量 FA_NUM_SPLITS > 0 时强制使用指定值（便于按 tile size 微调）。
inline int fa_mask_sm120_num_splits(const FA_mask_params &params, int kBlockM, int kBlockN) {
    static const int num_sms = []() {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    static const int env_splits = []() {
        const char *e = std::getenv("FA_NUM_SPLITS");
        return e ? std::atoi(e) : 0;
    }();

    const int num_m_blocks = (params.seqlen_q + kBlockM - 1) / kBlockM;
    const int num_n_blocks = (params.seqlen_k + kBlockN - 1) / kBlockN;
    const int total_mblocks = params.b * params.h * num_m_blocks;

    auto clamp_splits = [&](int s) {
        // ceil 区间划分下 num_splits > num_n_blocks 会产生空 split，无意义
        return std::max(1, std::min(s, num_n_blocks));
    };
    if (env_splits > 0) { return clamp_splits(env_splits); }

    constexpr int kMaxSplits = 64;   // combine kernel smem 上界（kMaxSplits*32*4B = 8KB）
    // grid 已接近填满 SM → 不 split
    if (total_mblocks >= 0.8f * num_sms) { return 1; }
    // K 块太少 → split 收益不足
    if (num_n_blocks <= 4) { return 1; }

    // Cost model（5090D 全 shape 实测拟合）：T(s) ≈ ceil(nb/s)·K + s·P
    // 关键洞察：combine kernel 是 3D 并行（CTAs = total·(kBlockM/32)·(d/32)），
    // grid 未超过 SM 数时其耗时与 total 无关 → 代价项只挂饱和因子 P，不挂 total！
    // （旧模型 s·total 高估 combine 代价，导致 total≥4 的 shape 最优 s 被系统性压低）
    // K = 每 n_block 主 kernel 延迟 / combine 单 split 代价，分配置拟合：
    //   d128 M128=2  d128 M64=8  d64 M128=1  d64 M64=18
    // 验证：8 个实测 shape 的 s* 预测全部命中实测最优（16/32/16/11/16/48/64/21）
    // 另：连续 s* 常落在 ceil(nb/s) 尾块不平整的悬崖上（实测 zigzag ±15%），
    // 因此在 s* 邻域做离散搜索，主 kernel 项用 ceil(nb/s) 精确刻画量化效应：
    const float K = (params.d == 128)
        ? ((kBlockM == 64) ? 8.0f : 2.0f)
        : ((kBlockM == 64) ? 18.0f : 1.0f);
    const int combine_ctas = total_mblocks * (kBlockM / 32) * (params.d / 32);
    const float P = std::max(1.0f, float(combine_ctas) / float(num_sms));
    const float s_star = std::sqrt(float(num_n_blocks) * K / P);
    const int cap_fill = std::max(1, num_sms / std::max(1, total_mblocks));
    const int hi = std::min({int(s_star) + 8, kMaxSplits, num_n_blocks, cap_fill});
    const int lo = std::min(std::max(1, int(s_star) - 8), hi);  // 防 lo>hi（s* 大而 cap_fill 小时）
    int best = 1;
    float best_cost = 1e30f;
    for (int s = lo; s <= hi; ++s) {
        const float cost = float((num_n_blocks + s - 1) / s) * K + float(s) * P;
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    // M64 护栏：每 split 至少 ~1.3 个 n_block，避免 prologue 占主导的悬崖（实测 s=nb 时劣化 30%+）
    if (kBlockM == 64) { best = std::min(best, std::max(1, int(num_n_blocks * 0.75f))); }
    return clamp_splits(best);
}

// ── sm120 Split-KV 路径 ──────────────────────────────────────────────────────
inline void run_mha_fwd_mask_hdim64_sm120_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120_splitkv<
        FA_mask_kernel_traits_sm120<64, 128, 64, 8, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, T>
    >(params, stream);
    run_flash_fwd_mask_combine_sm120<128, 64, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm120_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    // splitkv 专用配置：每 CTA 仅处理数个 n_block，深流水线收益小；
    // 改用 QInRegs=false（Q 走 TMA 批量加载 + ldmatrix，替代 32KB 标量 gmem 直载）+ kStages=2 腾出 smem
    run_flash_fwd_mask_sm120_splitkv<
        FA_mask_kernel_traits_sm120<128, 128, 64, 8, 2, /*MaskInSmem_=*/false, /*QInRegs_=*/false, T>
    >(params, stream);
    run_flash_fwd_mask_combine_sm120<128, 128, T>(params, stream);
}

// ── sm120 Split-KV + Split-M（kBlockM=64, 4 warps）──────────────────────────
// 小 grid 场景专用：M 维劈半让 m_block 数翻倍，不增加 combine 开销地白捡 2x 并行度；
// Sq<=64 时也避免了 kBlockM=128 半块 padding 的无效计算。代价：K/V 读取总量 ×2
// （带宽受限场景可接受，实测 475/1181 GB/s 有余量）。
inline void run_mha_fwd_mask_hdim64_sm120_splitkv_m64(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120_splitkv<
        FA_mask_kernel_traits_sm120<64, 64, 64, 4, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, T>
    >(params, stream);
    run_flash_fwd_mask_combine_sm120<64, 64, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm120_splitkv_m64(const FA_mask_params &params, cudaStream_t stream) {
    using T = cutlass::bfloat16_t;
    run_flash_fwd_mask_sm120_splitkv<
        FA_mask_kernel_traits_sm120<128, 64, 64, 4, 2, /*MaskInSmem_=*/false, /*QInRegs_=*/false, T>
    >(params, stream);
    run_flash_fwd_mask_combine_sm120<64, 128, T>(params, stream);
}

// SM 数查询（供 op.cu 的 split-M 判定使用）
inline int fa_mask_sm120_num_sms() {
    static const int num_sms = []() {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    return num_sms;
}

} // namespace FA_MASK_NAMESPACE
