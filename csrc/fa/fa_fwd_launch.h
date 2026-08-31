/*
 * fa_fwd_launch.h — Flash Attention Forward with Additive Mask
 *                    架构分发的标准入口（op 层唯一依赖）
 *
 * 分层（自上而下，每层只有一个事实来源）：
 *   csrc/arch_targets.h  仓库级标准入口：FA_TARGETS → FA_HAS_SM70/SM8X/SM120
 *                        （二进制含哪些家族）+ gpu_major()（当前设备是什么）
 *   本文件               FA 分发入口：按「编译期 FA_HAS_* × 运行期 gpu_major()」
 *                        双重校验后分发到各家族；每家族一个策略入口
 *                        fa_launch_smXX（tile 选择、Split-KV cost model、
 *                        Split-M、partial 缓冲分配全部内封）
 *   fa_fwd_op.cu         纯 op 层：输入校验 / 自动 pad / 填 params /
 *                        调 fa_fwd_launch(params, stream)，零架构感知
 *   smXX/fa_fwd_*.h      纯 kernel 实现：device 真身按 __CUDA_ARCH__ 区间
 *                        裁剪（nvcc 对 gencode 逐目标各编译一遍），host
 *                        launcher 无任何分发逻辑
 *
 * 各家族路径概览：
 *   sm8x（SM80~SM118 cp.async 基线，4090D=sm89）
 *     hdim64  : base (128,128,8w) / splitkv / splitkv_m64
 *     hdim128 : base (64,64,4w) / splitkv（按 tiles_per_cta 混合单/双缓冲）
 *   sm120（Blackwell consumer，RTX 50，TMA + mbarrier 多级流水）
 *     hdim64/128 : base (128,64) / splitkv / splitkv_m64
 *   sm70（Volta V100，fp16 专用，WMMA + DefaultCopy + smem softmax）
 *     tile 固定 (64,64)、512 线程，第一版无 split-KV
 *
 * 均支持 fp16/bf16 双 dtype（SM80+ tensor core 原生支持两者；按 params.is_bf16
 * 在 host 侧实例化对应 elem_type 变体）、无 dropout、无 causal（mask 由外部
 * 传入，语义同 SDPA：softmax(S·scale + mask)，支持 0/-inf 及任意有限值偏置）。
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <torch/extension.h>

#include "arch_targets.h"
#include "sm89/fa_fwd_kernel.h"
#include "sm120/fa_fwd_sm120.h"
#include "sm70/fa_fwd_sm70.h"

namespace FA_MASK_NAMESPACE {

// ── 小工具 ───────────────────────────────────────────────────────────────────
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// 按输入 dtype 构造 device tensor options（split-KV partial 缓冲分配用；
// device 取当前设备——op 层调用前已按 q 设 CUDAGuard）
static inline torch::TensorOptions fa_elem_options(bool is_bf16) {
    return torch::TensorOptions()
        .dtype(is_bf16 ? torch::kBFloat16 : torch::kHalf)
        .device(torch::kCUDA);
}

// ════════════════════════════════════════════════════════════════════════════
// sm8x 家族（SM80~SM118，cp.async 路径；4090D = sm89）
// ════════════════════════════════════════════════════════════════════════════
#if FA_HAS_SM8X

// ── hdim = 64 ─────────────────────────────────────────────────────────────────
// sm89: non-causal 时 128x128 最优；kNWarps=8（FA2 同 tile 尺寸的标准配置）。
// 注：kBlockM=128, kBlockN=128 必须配 8 warps。若用 4 warps，每线程仅累加器就需
// acc_s(128)+acc_o(64)=192 个 fp32 寄存器，叠加 MMA/mask/地址等状态后远超 255
// 上限，ptxas 溢出严重（实测 REG:255 + STACK 200B/线程），且主循环内 K/Mask
// 预加载的 64 位 gmem 地址从溢出槽恢复时高位为垃圾值 → illegal memory access
// （sm89 实测；8 warps 把累加器压力减半后实测 REG:~200/无溢出，问题消除）。

template <typename T>
inline void run_mha_fwd_mask_hdim64_impl(const FA_mask_params &params, cudaStream_t stream) {
    // kBlockM=128, kBlockN=128, kNWarps=8（256 threads）
    // smem: Q=16KB + K=16KB + V=16KB + Mask=32KB = 80KB (需要动态 smem)
    // mask q 维对齐 kBlockM 时走编译期无谓词变体（省谓词张量寄存器）
    if (params.mask_seqlen_q % 128 == 0) {
        run_flash_fwd_with_mask<
            FA_mask_kernel_traits<64, 128, 128, 8, false, false, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_with_mask<
            FA_mask_kernel_traits<64, 128, 128, 8, false, false, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
}

inline void run_mha_fwd_mask_hdim64(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim64_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim64_impl<cutlass::half_t>(params, stream);
}

// ── hdim = 128 ────────────────────────────────────────────────────────────────
// sm89: non-causal 时 64x64 最优（可以 2 CTAs per SM）
template <typename T>
inline void run_mha_fwd_mask_hdim128_impl(const FA_mask_params &params, cudaStream_t stream) {
    // kBlockM=64, kBlockN=64, kNWarps=4
    // smem: Q=16KB + K=16KB + V=16KB + Mask=8KB = 56KB (需要动态 smem)
    if (params.mask_seqlen_q % 64 == 0) {
        run_flash_fwd_with_mask<
            FA_mask_kernel_traits<128, 64, 64, 4, false, false, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_with_mask<
            FA_mask_kernel_traits<128, 64, 64, 4, false, false, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
}

inline void run_mha_fwd_mask_hdim128(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim128_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim128_impl<cutlass::half_t>(params, stream);
}

// SM 数查询（供 split-M 判定使用）
inline int fa_mask_sm89_num_sms() {
    static const int num_sms = []() {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    return num_sms;
}

// ── Split-KV num_splits cost model（结构与 sm120 版相同）─────────────────────
//   T(s) ≈ ceil(nb/s)·K + s·P
//   K = 每 n_block 主 kernel 延迟 / combine 单 split 代价
//   P = combine 饱和因子（小 grid 时为纯延迟下限 P0，大 grid 时 ~combine_ctas/SM）
// 参数在 RTX 4090D（128 SM）上用 8 个小-grid shape 的 num_splits 全量
// sweep（s ∈ {2..64}）网格搜索拟合：max regret 13.7%，mean 7.3%。
//   K：d64 (128,128) tile = 3.5，d128 (64,64) tile = 2.25；P0 = 0.35
//   d128 K=2.25 系 DB 混合分派后重拟合（旧 SB-only 标定 K=1.75，混合分派后
//   实测 s* 上移；重拟合后 max regret 11.7%、mean 7.3%，较 K=1.75 改善
//   ~4pp；剩余 regret 主要来自 ceil(nb/s) 量化 zigzag，解析模型难精细刻画）
//   cap_fill = 2×SM/total：允许至多 2 wave（实测 1 wave 精确对齐时尾效应
//   最重（exact-wave 一致性劣化 10~30%），留 2 wave 余量反而更优）
inline int fa_mask_sm89_num_splits(const FA_mask_params &params, int kBlockM, int kBlockN) {
    static const int num_sms = []() {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    const int num_m_blocks = (params.seqlen_q + kBlockM - 1) / kBlockM;
    const int num_n_blocks = (params.seqlen_k + kBlockN - 1) / kBlockN;
    const int total_mblocks = params.b * params.h * num_m_blocks;

    auto clamp_splits = [&](int s) {
        return std::max(1, std::min(s, num_n_blocks));
    };

    constexpr int kMaxSplits = 64;   // combine kernel smem 上界（kMaxSplits*32*4B = 8KB）
    if (total_mblocks >= 0.8f * num_sms) { return 1; }
    if (num_n_blocks <= 4) { return 1; }

    const float K = (kBlockN == 64) ? ((params.d == 128) ? 2.25f : 2.5f) : 3.5f;
    const int combine_ctas = total_mblocks * (kBlockM / 32) * (params.d / 32);
    const float P = std::max(0.35f, float(combine_ctas) / float(num_sms));
    const float s_star = std::sqrt(float(num_n_blocks) * K / P);
    const int cap_fill = std::max(1, (2 * num_sms) / std::max(1, total_mblocks));
    const int hi = std::min({int(s_star) + 8, kMaxSplits, num_n_blocks, cap_fill});
    const int lo = std::min(std::max(1, int(s_star) - 8), hi);
    int best = 1;
    float best_cost = 1e30f;
    for (int s = lo; s <= hi; ++s) {
        const float cost = float((num_n_blocks + s - 1) / s) * K + float(s) * P;
        if (cost < best_cost) { best_cost = cost; best = s; }
    }
    return clamp_splits(best);
}

// ── split-KV 路径（combine kernel 定义在 sm89/fa_fwd_kernel.h）──────────────
template <typename T>
inline void run_mha_fwd_mask_hdim64_splitkv_impl(const FA_mask_params &params, cudaStream_t stream) {
    // 与 base 同配置：(128, 128, 8 warps)，mask q 维对齐时走无谓词变体
    if (params.mask_seqlen_q % 128 == 0) {
        run_flash_fwd_with_mask_splitkv<
            FA_mask_kernel_traits<64, 128, 128, 8, false, false, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_with_mask_splitkv<
            FA_mask_kernel_traits<64, 128, 128, 8, false, false, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
    run_flash_fwd_mask_combine_sm89<64, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim64_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim64_splitkv_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim64_splitkv_impl<cutlass::half_t>(params, stream);
}

template <typename T>
inline void run_mha_fwd_mask_hdim128_splitkv_impl(const FA_mask_params &params, cudaStream_t stream) {
    // (64, 64, 4 warps)，双缓冲（kStages=2）：Q(16KB) + K/V×2(64KB) + Mask×2(16KB) = 96KB ≤ 99KB。
    // 实测（4090D，交错 A/B）：双缓冲消除每 tile 的两次串行往返等待，在每 CTA
    // tile 数较少（≤16）时快 4~10%；但稳态循环吞吐略低（运行期 stage 偏移的
    // 寻址开销 + 96KB smem 限制单 CTA/SM），tile 数多（≥32）的长串行运行慢 2~8%
    // → 按 tiles_per_cta 混合分发。
    // d64-M64 不能套用此结论：其 tile 计算量减半而 smem 32KB→56KB 使 occupancy
    // 从 2~3 CTA/SM 掉到 1，实测 DB 慢 10~40%，保持单缓冲。
    const int nb = (params.seqlen_k + 63) / 64;                 // kBlockN = 64
    const int tiles_per_cta = (nb + params.num_splits - 1) / params.num_splits;
    const bool use_db = (tiles_per_cta <= 16);
    if (use_db) {
        if (params.mask_seqlen_q % 64 == 0) {
            run_flash_fwd_with_mask_splitkv<
                FA_mask_kernel_traits<128, 64, 64, 4, false, false, /*MaskQFull_=*/true, T, /*kStages_=*/2>
            >(params, stream);
        } else {
            run_flash_fwd_with_mask_splitkv<
                FA_mask_kernel_traits<128, 64, 64, 4, false, false, /*MaskQFull_=*/false, T, /*kStages_=*/2>
            >(params, stream);
        }
    } else {
        if (params.mask_seqlen_q % 64 == 0) {
            run_flash_fwd_with_mask_splitkv<
                FA_mask_kernel_traits<128, 64, 64, 4, false, false, /*MaskQFull_=*/true, T>
            >(params, stream);
        } else {
            run_flash_fwd_with_mask_splitkv<
                FA_mask_kernel_traits<128, 64, 64, 4, false, false, /*MaskQFull_=*/false, T>
            >(params, stream);
        }
    }
    run_flash_fwd_mask_combine_sm89<128, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim128_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim128_splitkv_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim128_splitkv_impl<cutlass::half_t>(params, stream);
}

// ── split-KV + Split-M（kBlockM=64，仅 d64）──────────────────────────────────
// 小 grid 场景：M 维劈半让 m_block 数翻倍，不增加 combine 开销地提升并行度；
// Sq<=64 时也避免 kBlockM=128 半块 padding 的无效计算。
// 代价：K/V 读取总量 ×2（小 grid 带宽充裕，L2 可容纳米 swipe）
template <typename T>
inline void run_mha_fwd_mask_hdim64_splitkv_m64_impl(const FA_mask_params &params, cudaStream_t stream) {
    // (64, 64, 4 warps)：不用双缓冲：tile 计算量减半（d64）+ smem 32KB→56KB 使
    // occupancy 从 2~3 CTA/SM 掉到 1 CTA/SM，实测（4090D 交错 A/B）DB 慢
    // 10~40%，单缓冲严格更优
    if (params.mask_seqlen_q % 64 == 0) {
        run_flash_fwd_with_mask_splitkv<
            FA_mask_kernel_traits<64, 64, 64, 4, false, false, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_with_mask_splitkv<
            FA_mask_kernel_traits<64, 64, 64, 4, false, false, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
    run_flash_fwd_mask_combine_sm89<64, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim64_splitkv_m64(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim64_splitkv_m64_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim64_splitkv_m64_impl<cutlass::half_t>(params, stream);
}

// ── sm8x 家族策略入口（自 fa_fwd_op.cu 迁入）─────────────────────────────────
// base / split-KV / split-KV+M64 三选一；partial 缓冲在本层分配。
inline void fa_launch_sm8x(FA_mask_params &params, cudaStream_t stream) {
    // tile 配置（与上方 launcher 一致）：hdim64 → (128,128)，hdim128 → (64,64)
    const int kBlockM = (params.d == 64) ? 128 : 64;
    const int kBlockN = (params.d == 64) ? 128 : 64;
    // partial 缓冲行数按 base kBlockM 取整（M64 变体下略有冗余，combine 按
    // 实际 grid 读写，无害）
    params.seqlen_q_rounded = ceil_div_int(params.seqlen_q, kBlockM) * kBlockM;

    // 小 grid + 长序列时启用 Split-KV（cost model 决定 num_splits）
    int num_splits = fa_mask_sm89_num_splits(params, kBlockM, kBlockN);
    if (num_splits > 1) {
        // Split-M 判定（仅 d64，sm120 同名策略的 sm89 实测修正版）：
        //   kBlockM=64 让 m_block 数翻倍，不增加 combine 开销地提升并行度。
        //   实测（4090D）：Sq>64 且 total<SM/2 时 M64 快 5~21%（细粒度 CTA
        //   负载均衡 + m_block 翻倍）；Sq<=64 时 m_block 不翻倍，小 tile
        //   反而低效（实测慢 ~6%）→ 不采用 sm120 的 Sq<=64 规则
        bool use_m64 = false;
        if (params.d == 64) {
            const int num_sms = fa_mask_sm89_num_sms();
            const int total_mblocks_m128 =
                params.b * params.h * ceil_div_int(params.seqlen_q, 128);
            if ((params.seqlen_q > 64) && (total_mblocks_m128 < num_sms / 2)) {
                use_m64 = true;
                num_splits = fa_mask_sm89_num_splits(params, 64, 64);
            }
        }
        // O_partial 与输入同 dtype（2 字节，partial 流量减半），LSE 保持 fp32
        torch::Tensor oaccum = torch::empty(
            {num_splits, params.b, params.h, params.seqlen_q_rounded, params.d},
            fa_elem_options(params.is_bf16));
        torch::Tensor lseaccum = torch::empty(
            {num_splits, params.b, params.h, params.seqlen_q_rounded},
            fa_elem_options(params.is_bf16).dtype(torch::kFloat32));
        params.oaccum_ptr   = oaccum.data_ptr();
        params.lseaccum_ptr = lseaccum.data_ptr();
        params.num_splits   = num_splits;
        if (use_m64) {
            run_mha_fwd_mask_hdim64_splitkv_m64(params, stream);
        } else if (params.d == 64) {
            run_mha_fwd_mask_hdim64_splitkv(params, stream);
        } else {
            run_mha_fwd_mask_hdim128_splitkv(params, stream);
        }
        return;
    }
    if (params.d == 64) { run_mha_fwd_mask_hdim64(params, stream); }
    else                { run_mha_fwd_mask_hdim128(params, stream); }
}

#endif  // FA_HAS_SM8X

// ════════════════════════════════════════════════════════════════════════════
// sm120 家族（Blackwell consumer，RTX 50，TMA + mbarrier 多级流水）
// ════════════════════════════════════════════════════════════════════════════
#if FA_HAS_SM120

// ── dtype 分发约定 ────────────────────────────────────────────────────────────
// 下列 sm89/sm120 launcher 均为 elem_type 模板（*_impl），同一签名的外层函数按
// params.is_bf16 分发 bf16/fp16 实例——SM80+ tensor core 双 dtype 原生支持，
// kernel 内部 MMA atom / convert_type / TMA / combine 均已按 Element 泛化。

template <typename T>
inline void run_mha_fwd_mask_hdim64_sm120_impl(const FA_mask_params &params, cudaStream_t stream) {
    run_flash_fwd_mask_sm120<
        FA_mask_kernel_traits_sm120<64, 128, 64, 8, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, /*MaskQFull_=*/false, T>
    >(params, stream);
}

inline void run_mha_fwd_mask_hdim64_sm120(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim64_sm120_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim64_sm120_impl<cutlass::half_t>(params, stream);
}

template <typename T>
inline void run_mha_fwd_mask_hdim128_sm120_impl(const FA_mask_params &params, cudaStream_t stream) {
    // mask q 维对齐 kBlockM 时走编译期无边界路径变体（省 ~14 寄存器，splitkv longK 实测 +15%）
    if (params.mask_seqlen_q % 128 == 0) {
        run_flash_fwd_mask_sm120<
            FA_mask_kernel_traits_sm120<128, 128, 64, 8, 3, /*MaskInSmem_=*/false, /*QInRegs_=*/true, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_mask_sm120<
            FA_mask_kernel_traits_sm120<128, 128, 64, 8, 3, /*MaskInSmem_=*/false, /*QInRegs_=*/true, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
}

inline void run_mha_fwd_mask_hdim128_sm120(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim128_sm120_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim128_sm120_impl<cutlass::half_t>(params, stream);
}

// SM 数查询（供 split-M 判定使用）
inline int fa_mask_sm120_num_sms() {
    static const int num_sms = []() {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    return num_sms;
}

// ── Split-KV num_splits cost model（5090D 全 shape 实测拟合）──────────────────
// 目标：小 grid（B*H*num_m_blocks ≪ num_SMs）时按 K 维切分提升并行度，
// 同时避免过多 split 带来的读写放大（部分结果 bf16 落盘 + combine 回读）。
// 与 FA2/FA3 的 waves-efficiency 启发式不同，这里是实测拟合的解析 cost model，
// 大 grid 自动返回 1（无 split 开销）。
inline int fa_mask_sm120_num_splits(const FA_mask_params &params, int kBlockM, int kBlockN) {
    static const int num_sms = []() {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    const int num_m_blocks = (params.seqlen_q + kBlockM - 1) / kBlockM;
    const int num_n_blocks = (params.seqlen_k + kBlockN - 1) / kBlockN;
    const int total_mblocks = params.b * params.h * num_m_blocks;

    auto clamp_splits = [&](int s) {
        // ceil 区间划分下 num_splits > num_n_blocks 会产生空 split，无意义
        return std::max(1, std::min(s, num_n_blocks));
    };

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

// ── split-KV 路径 ──────────────────────────────────────────────────────────────
template <typename T>
inline void run_mha_fwd_mask_hdim64_sm120_splitkv_impl(const FA_mask_params &params, cudaStream_t stream) {
    run_flash_fwd_mask_sm120_splitkv<
        FA_mask_kernel_traits_sm120<64, 128, 64, 8, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, /*MaskQFull_=*/false, T>
    >(params, stream);
    run_flash_fwd_mask_combine_sm120<128, 64, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim64_sm120_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim64_sm120_splitkv_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim64_sm120_splitkv_impl<cutlass::half_t>(params, stream);
}

template <typename T>
inline void run_mha_fwd_mask_hdim128_sm120_splitkv_impl(const FA_mask_params &params, cudaStream_t stream) {
    // splitkv 专用配置：每 CTA 仅处理数个 n_block，深流水线收益小；
    // 改用 QInRegs=false（Q 走 TMA 批量加载 + ldmatrix，替代 32KB 标量 gmem 直载）+ kStages=2 腾出 smem
    if (params.mask_seqlen_q % 128 == 0) {
        run_flash_fwd_mask_sm120_splitkv<
            FA_mask_kernel_traits_sm120<128, 128, 64, 8, 2, /*MaskInSmem_=*/false, /*QInRegs_=*/false, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_mask_sm120_splitkv<
            FA_mask_kernel_traits_sm120<128, 128, 64, 8, 2, /*MaskInSmem_=*/false, /*QInRegs_=*/false, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
    run_flash_fwd_mask_combine_sm120<128, 128, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm120_splitkv(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim128_sm120_splitkv_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim128_sm120_splitkv_impl<cutlass::half_t>(params, stream);
}

// ── split-KV + Split-M（kBlockM=64, 4 warps）──────────────────────────────────
// 小 grid 场景专用：M 维劈半让 m_block 数翻倍，不增加 combine 开销地白捡 2x 并行度；
// Sq<=64 时也避免了 kBlockM=128 半块 padding 的无效计算。代价：K/V 读取总量 ×2
// （带宽受限场景可接受，实测 475/1181 GB/s 有余量）。
template <typename T>
inline void run_mha_fwd_mask_hdim64_sm120_splitkv_m64_impl(const FA_mask_params &params, cudaStream_t stream) {
    // 注：kStages=3 实测负收益（+18%，d64 M64 tile 小，深流水的额外 barrier 开销摊不薄），保持 2
    run_flash_fwd_mask_sm120_splitkv<
        FA_mask_kernel_traits_sm120<64, 64, 64, 4, 2, /*MaskInSmem_=*/true, /*QInRegs_=*/false, /*MaskQFull_=*/false, T>
    >(params, stream);
    run_flash_fwd_mask_combine_sm120<64, 64, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim64_sm120_splitkv_m64(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim64_sm120_splitkv_m64_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim64_sm120_splitkv_m64_impl<cutlass::half_t>(params, stream);
}

template <typename T>
inline void run_mha_fwd_mask_hdim128_sm120_splitkv_m64_impl(const FA_mask_params &params, cudaStream_t stream) {
    // 注：QInRegs+3st 实测全 Sk 负收益（M64 每 CTA 仅 2 个 n_block，第 3 级等不到数据、
    // Q 直载 16KB 标量 ldg 的 prologue 开销反而凸显），保持 Q-TMA + 2 级流水
    if (params.mask_seqlen_q % 64 == 0) {
        run_flash_fwd_mask_sm120_splitkv<
            FA_mask_kernel_traits_sm120<128, 64, 64, 4, 2, /*MaskInSmem_=*/false, /*QInRegs_=*/false, /*MaskQFull_=*/true, T>
        >(params, stream);
    } else {
        run_flash_fwd_mask_sm120_splitkv<
            FA_mask_kernel_traits_sm120<128, 64, 64, 4, 2, /*MaskInSmem_=*/false, /*QInRegs_=*/false, /*MaskQFull_=*/false, T>
        >(params, stream);
    }
    run_flash_fwd_mask_combine_sm120<64, 128, T>(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm120_splitkv_m64(const FA_mask_params &params, cudaStream_t stream) {
    if (params.is_bf16) run_mha_fwd_mask_hdim128_sm120_splitkv_m64_impl<cutlass::bfloat16_t>(params, stream);
    else                run_mha_fwd_mask_hdim128_sm120_splitkv_m64_impl<cutlass::half_t>(params, stream);
}

// ── sm120 家族策略入口（自 fa_fwd_op.cu 迁入）────────────────────────────────
// persistent kernel 已实测全线负收益（+0.2%~+5.1%），不参与分发（kernel 模板
// 保留在 fa_fwd_sm120.h 作历史参考，不被实例化）。
inline void fa_launch_sm120(FA_mask_params &params, cudaStream_t stream) {
    // tile 配置（与上方 launcher 一致）：hdim64/128 均为 (128,64)
    constexpr int kBlockM = 128, kBlockN = 64;
    params.seqlen_q_rounded = ceil_div_int(params.seqlen_q, kBlockM) * kBlockM;

    // Split-KV：cost model 决定 num_splits（大 grid 自动返回 1 → 无开销退化）
    int num_splits = fa_mask_sm120_num_splits(params, kBlockM, kBlockN);
    if (num_splits > 1) {
        // Split-M 判定：kBlockM=64 变体让 m_block 数翻倍（不增加 combine 开销地提升并行度）
        //   ① Sq<=64：kBlockM=128 会浪费半块 padding 计算，M64 严格更优
        //   ② grid 严重填不满（< 0.5 wave）：M64 把并行度翻倍
        bool use_m64 = false;
        {
            const int num_sms = fa_mask_sm120_num_sms();
            const int64_t grid_ctas =
                (int64_t)params.b * params.h * ceil_div_int(params.seqlen_q, 128) * num_splits;
            if (params.seqlen_q <= 64 || grid_ctas < num_sms / 2) {
                use_m64 = true;
                num_splits = fa_mask_sm120_num_splits(params, 64, kBlockN);
            }
        }
        // O_partial 与输入同 dtype（2 字节，partial 流量减半），LSE 保持 fp32
        torch::Tensor oaccum = torch::empty(
            {num_splits, params.b, params.h, params.seqlen_q_rounded, params.d},
            fa_elem_options(params.is_bf16));
        torch::Tensor lseaccum = torch::empty(
            {num_splits, params.b, params.h, params.seqlen_q_rounded},
            fa_elem_options(params.is_bf16).dtype(torch::kFloat32));
        params.oaccum_ptr   = oaccum.data_ptr();
        params.lseaccum_ptr = lseaccum.data_ptr();
        params.num_splits   = num_splits;
        if (use_m64) {
            if (params.d == 64) { run_mha_fwd_mask_hdim64_sm120_splitkv_m64(params, stream); }
            else                { run_mha_fwd_mask_hdim128_sm120_splitkv_m64(params, stream); }
        } else {
            if (params.d == 64) { run_mha_fwd_mask_hdim64_sm120_splitkv(params, stream); }
            else                { run_mha_fwd_mask_hdim128_sm120_splitkv(params, stream); }
        }
        return;
    }
    if (params.d == 64) { run_mha_fwd_mask_hdim64_sm120(params, stream); }
    else                { run_mha_fwd_mask_hdim128_sm120(params, stream); }
}

#endif  // FA_HAS_SM120

// ════════════════════════════════════════════════════════════════════════════
// sm70 家族（Volta V100，fp16 专用：mma.m8n8k4 + DefaultCopy + smem softmax）
// ════════════════════════════════════════════════════════════════════════════
#if FA_HAS_SM70

// 详见 sm70/fa_fwd_sm70.h 文件头。tile 固定 (64,64)、512 线程、fp16。
inline void run_mha_fwd_mask_hdim64_sm70(const FA_mask_params &params, cudaStream_t stream) {
    run_flash_fwd_with_mask_sm70<FA_sm70_kernel_traits<64>>(params, stream);
}

inline void run_mha_fwd_mask_hdim128_sm70(const FA_mask_params &params, cudaStream_t stream) {
    run_flash_fwd_with_mask_sm70<FA_sm70_kernel_traits<128>>(params, stream);
}

// ── sm70 家族策略入口（自 fa_fwd_op.cu 迁入）─────────────────────────────────
// 第一版无 Split-KV（V100 80 SM，B*H*m_blocks 通常足够；benchmark 后再评估）。
inline void fa_launch_sm70(FA_mask_params &params, cudaStream_t stream) {
    params.seqlen_q_rounded = ceil_div_int(params.seqlen_q, 64) * 64;
    if (params.d == 64) { run_mha_fwd_mask_hdim64_sm70(params, stream); }
    else                { run_mha_fwd_mask_hdim128_sm70(params, stream); }
}

#endif  // FA_HAS_SM70

// ════════════════════════════════════════════════════════════════════════════
// 对外唯一分发入口（fa_fwd_op.cu 仅依赖此函数）
// ════════════════════════════════════════════════════════════════════════════
// 「编译期 FA_HAS_*（二进制含哪些家族）× 运行期 gpu_major()（当前设备）」
// 双重校验后分发；均不匹配时显式报错——绝不静默跑空 kernel。
inline void fa_fwd_launch(FA_mask_params &params, cudaStream_t stream) {
    const int major = arch_targets::gpu_major();

#if FA_HAS_SM70
    // Volta 无 bf16 tensor core → bf16 输入落到 SM80+ 家族（无则报错）。
    // FA_FORCE_SM70=1 开发旁路：非 V100 GPU 上强制走 sm70 路径。
    // ⚠️ 仅对 v8.1 及之前（mma.m8n8k4，fragment 布局跨架构一致）有效；v9 起
    // 为手工 WMMA m16n16k16 PTX，fragment lane 映射系 sm70 实测逆向
    // （WMMA 内部布局无跨架构契约——实测 sm89 的 A-fragment 映射与 sm70 不同），
    // 在非 V100 架构上运行会静默产出错误结果。正确性验证必须在真 V100 上做。
    // 注：默认只编译本地 GPU 架构；非 V100 机器上需
    //   TORCH_CUDA_ARCH_LIST="7.0 <本机arch>" 显式编入 sm70 后此旁路才可用。
    if ((major == 7 || std::getenv("FA_FORCE_SM70")) && !params.is_bf16) {
        fa_launch_sm70(params, stream);
        return;
    }
#endif
#if FA_HAS_SM120
    if (major == 12) { fa_launch_sm120(params, stream); return; }
#endif
#if FA_HAS_SM8X
    if (major >= 8) { fa_launch_sm8x(params, stream); return; }
#endif
    TORCH_CHECK(false,
        "no FlashAttention kernel for this GPU (sm_", major,
        "xx, FA_TARGETS=0x", FA_TARGETS, ")",
        params.is_bf16
            ? "; bfloat16 requires SM80+ (Ampere or newer), "
              "V100 only supports the float16 path"
            : "");
}

} // namespace FA_MASK_NAMESPACE
