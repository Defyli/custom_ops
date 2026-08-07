/*
 * fa_fwd_cpu.h — Flash Attention 2 Forward（CPU 实现）接口与 runtime dispatch
 *
 * 提供与 mha_fwd_with_mask_cuda 完全兼容的 CPU 端实现。
 *
 * 架构分层
 * --------
 * mha_fwd_with_mask_cpu()   ← 对外统一接口（fa_fwd_op.cu 调用此函数）
 *   │
 *   ├── AVX2+FMA 路径  → fa_fwd_cpu_kernel_avx2<kBlockM, kBlockN>()
 *   │     · 手写 256-bit SIMD，kBlockM=8（一个 AVX2 向量宽度）
 *   │     · bf16→fp32 转换后在 fp32 下计算，结果转回 bf16
 *   │     · __builtin_prefetch(locality=3) 将 K/V tile 主动拉入 L1
 *   │
 *   └── Scalar 回退路径  → fa_fwd_cpu_kernel_scalar<kBlockM, kBlockN>()
 *         · 纯 C++，OpenMP 并行，编译器自动向量化
 *         · 用于不支持 AVX2 的环境 / CI / 调试
 *
 * Tile 尺寸选取依据
 * -----------------
 * 目标：K_tile + V_tile 驻留在 L1 Cache（≈32KB per core）
 *   kBlockM = 8   (= AVX2 向量宽度，8×fp32 = 256-bit)
 *   kBlockN = 32  (d=128: K_tile=32×128×2=8KB, V_tile=8KB → 总 16KB < 32KB L1)
 *             64  (d=64:  K_tile=64×64×2=8KB,  V_tile=8KB → 总 16KB < 32KB L1)
 * online softmax 的 m_val/l_val 向量 (kBlockM=8 个 fp32) 全程驻留 AVX2 寄存器。
 *
 * mask 语义
 * ---------
 *   mask=0    → 可见（score 不变）
 *   mask=-inf → 屏蔽（score → -inf，softmax 后 weight ≈ 0）
 *
 * 线程数对齐
 * ----------
 * OpenMP 线程数通过 _get_fa_cpu_nthreads() 获取，优先读取
 * OMP_NUM_THREADS（由 Serving 框架设置）以避免核心竞争。
 *
 * 数值精度
 * --------
 * 所有中间计算以 fp32 进行（online softmax 统计量、GEMM 累加器均为 fp32）。
 * 输入 bf16 → 计算 fp32 → 输出 bf16，与 CUDA kernel 保持一致。
 */

#pragma once

#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// 编译时特性检测
// ─────────────────────────────────────────────────────────────────────────────

// GCC / Clang 均支持 __builtin_cpu_supports（链接 -lc 时可用）
// MSVC 不支持，但我们的目标平台是 Linux/GCC
#if defined(__GNUC__) || defined(__clang__)
#  define FA_CPU_HAS_BUILTIN_CPU_SUPPORTS 1
#else
#  define FA_CPU_HAS_BUILTIN_CPU_SUPPORTS 0
#endif

// AVX2+FMA 可用性（编译时和运行时双重检测）
#if defined(__AVX2__) && defined(__FMA__)
#  define FA_CPU_AVX2_COMPILED 1
#  include <immintrin.h>
#else
#  define FA_CPU_AVX2_COMPILED 0
#endif

// AVX-512 可用性（编译时宏）
// 要求 avx512f（基础） + avx512dq（_mm512_extractf32x8_ps 等）
// fa_fwd_cpu_avx512.cpp 通过 #pragma GCC target 激活，此处仅声明前向函数
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#  define FA_CPU_AVX512_COMPILED 1
#else
#  define FA_CPU_AVX512_COMPILED 0
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Tile 参数（编译时常量）
// ─────────────────────────────────────────────────────────────────────────────

// kBlockM：Q 行方向 tile 大小
// AVX2  = 8  个 float（256-bit）
// AVX-512 = 16 个 float（512-bit）
// 对应关系：Q 行 load 进 SIMD 寄存器后，D_UNROLL = d / kVecWidth
static constexpr int kCpuBlockM = 8;   // 行级并行度（OpenMP 外层 head 并行，此值保留兼容）

// kBlockN：K/V 列方向 tile 大小，以 L1 cache 为约束：
//   d=128: kBlockN=32 → K+V = 2×32×128×2 B = 16 KB < 32KB L1
//   d=64:  kBlockN=64 → K+V = 2×64×64×2  B = 16 KB < 32KB L1
// 运行时根据 d 动态选取
static constexpr int kCpuBlockN_d128 = 32;
static constexpr int kCpuBlockN_d64  = 64;

// prefetch 提前量：L1 cache miss 惩罚 ~10 cycles，prefetch 延迟 ~4 cycles
// 提前 2 个 tile 预取（保守策略，避免 cache pollution）
static constexpr int kCpuPrefetchAhead = 2;

// ─────────────────────────────────────────────────────────────────────────────
// 工具：bf16 ↔ fp32 转换（无需 SIMD，逐元素）
// ─────────────────────────────────────────────────────────────────────────────

// bf16 存储为 uint16_t（高 16 位的 fp32）
// 转 fp32：在低 16 位补零即可（IEEE 754 标准）
inline float bf16_to_f32(uint16_t x) {
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// fp32 → bf16：截断低 16 位（Round-to-Nearest-Even 可选，此处用截断）
inline uint16_t f32_to_bf16(float f) {
    // 处理 NaN：保持 NaN 语义
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even（标准做法）
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

// ─────────────────────────────────────────────────────────────────────────────
// 工具：OpenMP 线程数（与 Serving 框架对齐）
// ─────────────────────────────────────────────────────────────────────────────

inline int _get_fa_cpu_nthreads() {
    // 优先读取 OMP_NUM_THREADS（由 Triton Serving 框架统一设置，避免核心竞争）
    const char* env = std::getenv("OMP_NUM_THREADS");
    if (env && env[0] != '\0') {
        int n = std::atoi(env);
        if (n > 0) return n;
    }
    // 次选：PyTorch interop 线程数（已与 serving 对齐）
    int torch_threads = at::get_num_interop_threads();
    if (torch_threads > 0) return torch_threads;
    // 保守回退：1 线程，避免不可控的核心竞争
    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// 运行时 CPU 特性探测
// ─────────────────────────────────────────────────────────────────────────────

inline bool _cpu_has_avx2_fma() {
#if FA_CPU_HAS_BUILTIN_CPU_SUPPORTS
    return static_cast<bool>(__builtin_cpu_supports("avx2"))
        && static_cast<bool>(__builtin_cpu_supports("fma"));
#else
    return false;
#endif
}

// AVX-512 运行时检测：avx512f（基础）+ avx512dq（extractf32x8 等）
inline bool _cpu_has_avx512() {
#if FA_CPU_HAS_BUILTIN_CPU_SUPPORTS
    return static_cast<bool>(__builtin_cpu_supports("avx512f"))
        && static_cast<bool>(__builtin_cpu_supports("avx512dq"));
#else
    return false;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// 前向声明：具体实现在各 .cpp 文件中定义
// ─────────────────────────────────────────────────────────────────────────────

// Scalar / OpenMP 路径（fa_fwd_cpu.cpp）
void fa_fwd_cpu_kernel_scalar(
    const float* __restrict__ q_ptr,     // (H, Sq, d) fp32，连续
    const float* __restrict__ k_ptr,     // (Hk, Sk, d) fp32，连续
    const float* __restrict__ v_ptr,     // (Hk, Sk, d) fp32，连续
    const float* __restrict__ mask_ptr,  // (Sq, Sk) fp32，连续
    float*       __restrict__ out_ptr,   // (H, Sq, d) fp32，输出
    int H, int Hk, int Sq, int Sk, int d,
    float scale, int n_threads
);

// AVX2+FMA 路径（fa_fwd_cpu_avx2.cpp）
// 通过 #pragma GCC target("avx2,fma") 在其翻译单元内激活，此处无条件声明。
// 若当前 CPU 不支持 AVX2，则 dispatch 层不会调用此函数。
void fa_fwd_cpu_kernel_avx2(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    const float* __restrict__ mask_ptr,
    float*       __restrict__ out_ptr,
    int H, int Hk, int Sq, int Sk, int d,
    float scale, int n_threads
);

// AVX-512+FMA 路径（fa_fwd_cpu_avx512.cpp）
// 通过 #pragma GCC target("avx512f,...") 在其翻译单元内激活。
void fa_fwd_cpu_kernel_avx512(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    const float* __restrict__ mask_ptr,
    float*       __restrict__ out_ptr,
    int H, int Hk, int Sq, int Sk, int d,
    float scale, int n_threads
);
