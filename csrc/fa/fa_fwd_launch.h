/*
 * Flash Attention Forward with Additive Mask — Launch Templates
 *
 * 提供两个 host-side 入口函数：
 *   run_mha_fwd_mask_hdim64   — head_dim = 64
 *   run_mha_fwd_mask_hdim128  — head_dim = 128
 *
 * 均为 bf16、无 dropout、无 causal（mask 由外部传入）。
 * 仅针对 sm89（RTX4090）优化：
 *   hdim64  : kBlockM=128, kBlockN=128（128*128=16K smem for QKV，+mask=32K，总~64K 在 48K 上限内）
 *             → 实际: kSmemQSize=128*64*2=16KB, kSmemKVSize=128*64*2*2=32KB, +mask=128*128*2=32KB
 *             → 总=80KB 超 48KB，动态 smem 最大 99KB（RTX4090），OK。
 *   hdim128 : kBlockM=64,  kBlockN=64（RTX4090 sm89 最优）
 *             → kSmemQSize=64*128*2=16KB, kSmemKVSize=64*128*2*2=32KB, +mask=64*64*2=8KB
 *             → 总=56KB，需动态 smem
 *
 * 注意：FA2 原始 sm89+hdim128 用 (64, 64, 4) 配置，smem=(64*128 + 2*64*128)*2 = 48KB；
 *       加上 mask tile = 64*64*2=8KB，总 56KB，需设置 MaxDynamicSharedMemorySize。
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

// ── Split-KV num_splits 启发式（FA2/FA3 num_splits_heuristic 的适配版）────────
// 目标：让 total_work = B*H*num_m_blocks*num_splits 恰好填满 SM（~1 wave 且效率高），
// 同时避免过多 split 带来的 HBM 读写放大（部分结果 fp32 落盘 + combine 回读）。
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

    // Cost model（5090D 实测拟合）：T(s) ≈ F + (nb/s)·t_nb·waves + c·s·total
    //   t_nb: 单 CTA 每 n_block 延迟（d128≈3µs, d64≈1.5µs）；c: combine 每 split 每 tile ≈0.15µs
    // waves ≤ 1 时对 s 求导得最优 s* = sqrt(nb·t_nb / (c·total))
    const float t_nb = (params.d == 128) ? 3.0f : 1.5f;
    constexpr float c_combine = 0.15f;
    const float s_star = std::sqrt(float(num_n_blocks) * t_nb / (c_combine * float(total_mblocks)));
    int best = std::max(1, int(s_star + 0.5f));
    best = std::min({best, kMaxSplits, num_n_blocks});
    // 不超过 SM 数太多（>1 wave 时效率模型才适用，此处保守 cap 到刚好填满）
    const int cap_fill = std::max(1, num_sms / std::max(1, total_mblocks));
    best = std::min(best, std::max(cap_fill, 1));
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

} // namespace FA_MASK_NAMESPACE
