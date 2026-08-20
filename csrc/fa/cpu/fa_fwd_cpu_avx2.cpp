/*
 * fa_fwd_cpu_avx2.cpp — Flash Attention 2 Forward（AVX2+FMA 手写 SIMD）
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 总体策略
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * 问题：CPU FA2 的主要瓶颈是 Q·K^T 内积和 p·V 累加（内存访问密集）。
 * 目标：用 AVX2 + FMA 256-bit 指令一次处理 8 个 float，最大化向量吞吐量。
 *
 * 关键设计
 * --------
 * 1. kBlockM = 8（= AVX2 向量宽度）：
 *    一次处理 8 个 Q 行，online softmax 统计量 m_val/l_val 用 __m256 寄存器持有，
 *    与 CUDA kBlockM 64 在概念上对应（CUDA 是 warps，这里是 SIMD lanes）。
 *    但实际按行展开以充分利用 FMA 硬件，避免转置。
 *
 * 2. d 维的 SIMD 展开：
 *    Q·K^T 内积：对 d 循环，步长 8，每次 _mm256_fmadd_ps（vfmadd231ps）。
 *    p·V 累加：对 d 循环，步长 8，每次 _mm256_fmadd_ps。
 *    当 d=128 时共 16 个 AVX2 步；d=64 时共 8 个 AVX2 步。
 *
 * 3. 寄存器分配（d=128 为例）：
 *    q_regs[16]: __m256[16]，对应一个 Q 行的 128 个 float（寄存器驻留）
 *    acc[16]:    __m256[16]，输出累加器 out_row（p · V 的总和）
 *    以上共 32 个 ymm 寄存器 = AVX2 全部寄存器，完美对应 d=128 场景。
 *    d=64 时只用 16 个，寄存器压力更小。
 *
 * 4. L1 Cache 对齐：
 *    目标：K tile(kBlockN×d) + V tile(kBlockN×d) 共 2×kBlockN×d×4 B ≤ 32KB L1
 *    d=128, kBlockN=32:  2×32×128×4 = 32768 B = 32KB（恰好）
 *    d=64,  kBlockN=64:  2×64×64×4  = 32768 B = 32KB（恰好）
 *    __builtin_prefetch(locality=3) 提前 2 tile 预取，覆盖 L1 miss 惩罚（~10 cycles）。
 *
 * 5. OpenMP 并行粒度：
 *    外层 OpenMP 并行 head 维度，每个线程独占一组 K/V head，cache 不竞争。
 *
 * 6. GQA 支持：
 *    通过 kv_head = bidh / (H / Hk) 映射 Q head 到 KV head，
 *    与 scalar 路径一致。
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 编译选项（由 __init__.py 注入）
 * ─────────────────────────────────────────────────────────────────────────────
 *   -O3 -ffast-math -fopenmp -mavx2 -mfma（通过 #pragma GCC target 激活，
 *   不影响其他翻译单元，无需修改全局编译 flags）
 */

// 仅为本翻译单元激活 AVX2+FMA，不影响其他 .cpp
// online softmax 依赖 -inf 哨兵值的 IEEE 754 语义；
// -ffast-math 的 -ffinite-math-only 会破坏 -inf 上的算术。
#pragma GCC optimize("O3,no-finite-math-only")
#pragma GCC target("avx2,fma")

#include "fa_fwd_cpu.h"

#include <immintrin.h>     // AVX2 intrinsics
#include <omp.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 工具宏
// ─────────────────────────────────────────────────────────────────────────────

// 水平最大值：返回 __m256 中 8 个 float 的最大值（标量）
static inline float _mm256_hmax_ps(__m256 v) {
    // 先折叠高 128 vs 低 128
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    // 再折叠 4→2→1
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ps(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m1);
}

// 水平求和：返回 __m256 中 8 个 float 的和（标量）
static inline float _mm256_hadd_ps_scalar(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ps(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}

// 向量化 expf：对 __m256 逐元素 expf（无 AVX-512 的近似实现）
// 使用多项式逼近（精度 ~2^-23，满足 bf16 需求）
// 基于 Cephes/Intel SVML 的 exp 实现思路
static inline __m256 _mm256_exp_ps_fast(__m256 x) {
    // 限制输入范围，避免溢出
    const __m256 max_x = _mm256_set1_ps(88.3762626647949f);
    const __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    // exp(x) = 2^(x/ln2) = 2^(n + f), 0 <= f < 1
    const __m256 cephes_log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half          = _mm256_set1_ps(0.5f);
    const __m256 one           = _mm256_set1_ps(1.0f);

    // n = round(x / ln2)
    __m256 z = _mm256_fmadd_ps(x, cephes_log2ef, half);
    __m256 n = _mm256_floor_ps(z);

    // f = x - n * ln2（双精度补偿）
    const __m256 cephes_exp_c1 = _mm256_set1_ps(-0.693359375f);
    const __m256 cephes_exp_c2 = _mm256_set1_ps(2.12194440e-4f);
    __m256 f = _mm256_fmadd_ps(n, cephes_exp_c1, x);
    f = _mm256_fmadd_ps(n, cephes_exp_c2, f);

    // Minimax 7 阶多项式逼近 2^f-1 on [0,1)
    const __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);
    __m256 y = p0;
    y = _mm256_fmadd_ps(y, f, p1);
    y = _mm256_fmadd_ps(y, f, p2);
    y = _mm256_fmadd_ps(y, f, p3);
    y = _mm256_fmadd_ps(y, f, p4);
    y = _mm256_fmadd_ps(y, f, p5);
    y = _mm256_fmadd_ps(y, f, one);
    y = _mm256_fmadd_ps(y, f, one);

    // 用 ldexp：乘以 2^n = (int(n)+127) << 23
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(0x7f));
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(ni);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// 模板函数：单头单行的 AVX2 online softmax attention
// D_UNROLL：d / 8，编译时展开（d=128→16, d=64→8）
// ─────────────────────────────────────────────────────────────────────────────

template<int D_UNROLL>
static void compute_one_row_avx2(
    const float* __restrict__ q_row,     // (d,)  fp32，对齐 32B（32B-aligned）
    const float* __restrict__ k_ptr,     // (Sk, d) fp32
    const float* __restrict__ v_ptr,     // (Sk, d) fp32
    const float* __restrict__ mask_row,  // (Sk,)  fp32
    float*       __restrict__ out_row,   // (d,)   fp32，输出
    int Sk, int d, int kBlockN, float scale
) {
    // Q 行加载到 SIMD 寄存器（编译器可能 spill，但 GCC -O3 通常能保持在 ymm）
    __m256 qr[D_UNROLL];
    const float* qp = q_row;
    for (int i = 0; i < D_UNROLL; ++i, qp += 8) {
        qr[i] = _mm256_loadu_ps(qp);
    }

    // 输出累加器（清零）
    __m256 acc[D_UNROLL];
    for (int i = 0; i < D_UNROLL; ++i) {
        acc[i] = _mm256_setzero_ps();
    }

    // Online softmax 统计量
    float m_val = -std::numeric_limits<float>::infinity();
    float l_val = 0.0f;

    // score 缓冲（最大 tile_n=64）
    alignas(32) float s_buf[64];

    for (int n_start = 0; n_start < Sk; n_start += kBlockN) {
        const int n_end  = std::min(n_start + kBlockN, Sk);
        const int tile_n = n_end - n_start;

        // ── 预取下一个 K/V tile 到 L1（locality=3）──────────────────────────
        {
            const int prefetch_n = n_start + kCpuPrefetchAhead * kBlockN;
            if (prefetch_n < Sk) {
                const char* k_next = reinterpret_cast<const char*>(
                    k_ptr + prefetch_n * d);
                const char* v_next = reinterpret_cast<const char*>(
                    v_ptr + prefetch_n * d);
                const int tile_bytes = tile_n * d * sizeof(float);
                for (int off = 0; off < tile_bytes; off += 64) {
                    __builtin_prefetch(k_next + off, 0, 3);
                    __builtin_prefetch(v_next + off, 0, 3);
                }
            }
        }

        // ── 1. 计算 s = Q·K^T × scale + mask（当前 tile）────────────────────
        // 对每个 K[j]，用 D_UNROLL 路 AVX2 FMA 做内积
        for (int j = 0; j < tile_n; ++j) {
            const float* kp = k_ptr + (n_start + j) * d;
            __m256 dot = _mm256_setzero_ps();
            for (int i = 0; i < D_UNROLL; ++i) {
                __m256 kv = _mm256_loadu_ps(kp + i * 8);
                dot = _mm256_fmadd_ps(qr[i], kv, dot);
            }
            // 水平求和 + scale + mask
            s_buf[j] = _mm256_hadd_ps_scalar(dot) * scale + mask_row[n_start + j];
        }

        // ── 2. Online softmax：m_new = max(m_val, max(s_buf)) ────────────────
        // 先单独计算 tile 内的 local max（从 -inf 开始，不从 m_val 开始），
        // 以精确判断全屏蔽：m_tile=-inf 当且仅当全屏蔽。
        float m_tile = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < tile_n; ++j) {
            if (s_buf[j] > m_tile) m_tile = s_buf[j];
        }

        // 全屏蔽：pragma no-finite-math-only 保证 -inf 比较正确。
        if (!(m_tile > -std::numeric_limits<float>::infinity())) {
            continue;
        }

        float m_new = (m_tile > m_val) ? m_tile : m_val;

        // 纠正因子 corr = exp(m_old - m_new)
        // m_val=-inf（第一个有效 tile）：expf(-inf) = 0，acc 清零重起
        float corr = expf(m_val - m_new);

        // rescale 历史累加器 acc *= corr
        {
            __m256 corr_v = _mm256_set1_ps(corr);
            for (int i = 0; i < D_UNROLL; ++i) {
                acc[i] = _mm256_mul_ps(acc[i], corr_v);
            }
        }
        l_val *= corr;

        // ── 3. 累加 p × V（当前 tile）──────────────────────────────────────
        const __m256 m_new_v = _mm256_set1_ps(m_new);
        for (int j = 0; j < tile_n; ++j) {
            float p = expf(s_buf[j] - m_new);
            l_val += p;

            __m256 p_v = _mm256_set1_ps(p);
            const float* vp = v_ptr + (n_start + j) * d;
            for (int i = 0; i < D_UNROLL; ++i) {
                __m256 vv = _mm256_loadu_ps(vp + i * 8);
                acc[i] = _mm256_fmadd_ps(p_v, vv, acc[i]);
            }
        }
        (void)m_new_v;  // 避免 unused-variable 警告

        m_val = m_new;
    }

    // ── 4. 归一化 out /= l ────────────────────────────────────────────────────
    if (l_val > 0.0f) {
        __m256 inv_l = _mm256_set1_ps(1.0f / l_val);
        float* op = out_row;
        for (int i = 0; i < D_UNROLL; ++i, op += 8) {
            _mm256_storeu_ps(op, _mm256_mul_ps(acc[i], inv_l));
        }
    } else {
        // 全屏蔽：输出 0
        std::fill(out_row, out_row + d, 0.0f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// fa_fwd_cpu_kernel_avx2 — 主函数（OpenMP 并行 head，逐行调用 AVX2 内核）
// ─────────────────────────────────────────────────────────────────────────────

void fa_fwd_cpu_kernel_avx2(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    const float* __restrict__ mask_ptr,
    float*       __restrict__ out_ptr,
    int H, int Hk, int Sq, int Sk, int d,
    float scale, int n_threads
) {
    // kBlockN 选取（L1 cache 约束，见头文件注释）
    const int kBlockN = (d >= 128) ? kCpuBlockN_d128 : kCpuBlockN_d64;

    omp_set_num_threads(n_threads);

    // d 分支：编译时确定模板参数，最大化内核中的循环展开
    if (d == 128) {
        // D_UNROLL = 128 / 8 = 16
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int bidh = 0; bidh < H; ++bidh) {
            const int kv_head = bidh / (H / Hk);
            const float* q_head   = q_ptr   + bidh   * Sq * d;
            const float* k_head   = k_ptr   + kv_head * Sk * d;
            const float* v_head   = v_ptr   + kv_head * Sk * d;
            float*       out_head = out_ptr + bidh   * Sq * d;

            for (int m = 0; m < Sq; ++m) {
                compute_one_row_avx2<16>(
                    q_head   + m * d,
                    k_head, v_head,
                    mask_ptr + m * Sk,
                    out_head + m * d,
                    Sk, d, kBlockN, scale
                );
            }
        }
    } else if (d == 64) {
        // D_UNROLL = 64 / 8 = 8
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int bidh = 0; bidh < H; ++bidh) {
            const int kv_head = bidh / (H / Hk);
            const float* q_head   = q_ptr   + bidh   * Sq * d;
            const float* k_head   = k_ptr   + kv_head * Sk * d;
            const float* v_head   = v_ptr   + kv_head * Sk * d;
            float*       out_head = out_ptr + bidh   * Sq * d;

            for (int m = 0; m < Sq; ++m) {
                compute_one_row_avx2<8>(
                    q_head   + m * d,
                    k_head, v_head,
                    mask_ptr + m * Sk,
                    out_head + m * d,
                    Sk, d, kBlockN, scale
                );
            }
        }
    } else {
        // 不支持的 d，回退 scalar（理论上不应到达此处，在接口层已检查）
        fa_fwd_cpu_kernel_scalar(
            q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr,
            H, Hk, Sq, Sk, d, scale, n_threads
        );
    }
}
