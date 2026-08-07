/*
 * fa_fwd_cpu_avx512.cpp — Flash Attention 2 Forward（AVX-512+FMA 手写 SIMD）
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * 为何 AVX-512 > AVX2？
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * 向量宽度对比：
 *   AVX2    : 256-bit = 8  个 float，D_UNROLL = d/8   (d=128→16, d=64→8)
 *   AVX-512 : 512-bit = 16 个 float，D_UNROLL = d/16  (d=128→8,  d=64→4)
 *
 * 主要收益：
 *   1. FMA 单元每拍处理 16 float（vs AVX2 的 8），理论吞吐 2×。
 *   2. Ice Lake (avx512vnni) 没有 AVX-512 频率降档惩罚（Skylake-X 才有），
 *      可全速运行 512-bit 指令。
 *   3. d=128 时 D_UNROLL=8，恰好在 16 个 zmm 寄存器内（Q 行 8 个 zmm，
 *      acc 8 个 zmm），无寄存器 spill，与 CUDA kBlockM 场景类似。
 *
 * L1 Cache Tile 设计（同 AVX2 路径）：
 *   kBlockN_d128 = 32: K_tile + V_tile = 2×32×128×4 = 32KB ≤ L1
 *   kBlockN_d64  = 64: K_tile + V_tile = 2×64×64×4  = 32KB ≤ L1
 *   __builtin_prefetch(locality=3) 提前 2 tile 预取到 L1。
 *
 * AVX-512 exp 实现：
 *   复用 Cephes/Intel 多项式逼近（_mm512_exp_ps_fast），精度 ~2^-23，
 *   满足 bf16 精度需求。
 *   注意：AVX-512 有 _mm512_exp_ps（需要 SVML），但 GCC 标准库不提供，
 *   此处用 Cephes 多项式保证可移植性。
 *
 * 编译指令集（通过 #pragma GCC target 仅激活本 TU）：
 *   avx512f, avx512dq, avx512bw, avx512vl, fma
 */

// online softmax 依赖 -inf 哨兵值的 IEEE 754 语义；
// -ffast-math 的 -ffinite-math-only 会破坏 -inf 上的算术和比较。
#pragma GCC optimize("O3,no-finite-math-only")
#pragma GCC target("avx512f,avx512dq,avx512bw,avx512vl,fma")

#include "fa_fwd_cpu.h"

#include <immintrin.h>   // AVX-512 intrinsics
#include <omp.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 工具函数
// ─────────────────────────────────────────────────────────────────────────────

// 水平最大值：返回 __m512 中 16 个 float 的最大值（标量）
static inline float _mm512_hmax_ps_scalar(__m512 v) {
    // 折叠高 256 vs 低 256
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 m8 = _mm256_max_ps(lo, hi);
    // 折叠 8→4→2→1
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m8),
                            _mm256_extractf128_ps(m8, 1));
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ps(m2, _mm_shuffle_ps(m2, m2, 1));
    return _mm_cvtss_f32(m1);
}

// 水平求和：返回 __m512 中 16 个 float 的和（标量）
static inline float _mm512_hadd_ps_scalar(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8),
                            _mm256_extractf128_ps(s8, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ps(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}

// 向量化 expf（Cephes 多项式，__m512 版本）
// 精度 ~2^-23，满足 bf16 需求（bf16 精度 ~2^-7）
static inline __m512 _mm512_exp_ps_fast(__m512 x) {
    const __m512 max_x = _mm512_set1_ps( 88.3762626647949f);
    const __m512 min_x = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, max_x);
    x = _mm512_max_ps(x, min_x);

    const __m512 log2ef  = _mm512_set1_ps(1.44269504088896341f);
    const __m512 half    = _mm512_set1_ps(0.5f);
    const __m512 one     = _mm512_set1_ps(1.0f);

    // n = round(x / ln2)
    __m512 z = _mm512_fmadd_ps(x, log2ef, half);
    __m512 n = _mm512_floor_ps(z);

    // f = x - n * ln2（双精度补偿）
    const __m512 c1 = _mm512_set1_ps(-0.693359375f);
    const __m512 c2 = _mm512_set1_ps( 2.12194440e-4f);
    __m512 f = _mm512_fmadd_ps(n, c1, x);
    f = _mm512_fmadd_ps(n, c2, f);

    // Minimax 7-阶多项式 on [0,1)
    const __m512 p0 = _mm512_set1_ps(1.9875691500E-4f);
    const __m512 p1 = _mm512_set1_ps(1.3981999507E-3f);
    const __m512 p2 = _mm512_set1_ps(8.3334519073E-3f);
    const __m512 p3 = _mm512_set1_ps(4.1665795894E-2f);
    const __m512 p4 = _mm512_set1_ps(1.6666665459E-1f);
    const __m512 p5 = _mm512_set1_ps(5.0000001201E-1f);
    __m512 y = p0;
    y = _mm512_fmadd_ps(y, f, p1);
    y = _mm512_fmadd_ps(y, f, p2);
    y = _mm512_fmadd_ps(y, f, p3);
    y = _mm512_fmadd_ps(y, f, p4);
    y = _mm512_fmadd_ps(y, f, p5);
    y = _mm512_fmadd_ps(y, f, one);
    y = _mm512_fmadd_ps(y, f, one);

    // pow2n = 2^n via bit-shift
    __m512i ni = _mm512_cvtps_epi32(n);
    ni = _mm512_add_epi32(ni, _mm512_set1_epi32(0x7f));
    ni = _mm512_slli_epi32(ni, 23);
    __m512 pow2n = _mm512_castsi512_ps(ni);
    return _mm512_mul_ps(y, pow2n);
}

// ─────────────────────────────────────────────────────────────────────────────
// 模板函数：单头单行的 AVX-512 online softmax attention
// D_UNROLL：d / 16，编译时展开（d=128→8, d=64→4）
// ─────────────────────────────────────────────────────────────────────────────

template<int D_UNROLL>
static void compute_one_row_avx512(
    const float* __restrict__ q_row,     // (d,)  fp32
    const float* __restrict__ k_ptr,     // (Sk, d) fp32
    const float* __restrict__ v_ptr,     // (Sk, d) fp32
    const float* __restrict__ mask_row,  // (Sk,)  fp32
    float*       __restrict__ out_row,   // (d,)   fp32，输出
    int Sk, int d, int kBlockN, float scale
) {
    // Q 行预加载到 zmm 寄存器（D_UNROLL 个 __m512）
    // d=128: 8 个 zmm；d=64: 4 个 zmm —— 寄存器压力极小
    __m512 qr[D_UNROLL];
    const float* qp = q_row;
    for (int i = 0; i < D_UNROLL; ++i, qp += 16) {
        qr[i] = _mm512_loadu_ps(qp);
    }

    // 输出累加器（D_UNROLL 个 zmm）
    __m512 acc[D_UNROLL];
    for (int i = 0; i < D_UNROLL; ++i) {
        acc[i] = _mm512_setzero_ps();
    }

    // Online softmax 统计量
    float m_val = -std::numeric_limits<float>::infinity();
    float l_val = 0.0f;

    // score 缓冲（最大 tile_n=64，栈上安全）
    alignas(64) float s_buf[64];

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
        // 每个 K[j]：D_UNROLL 路 AVX-512 FMA 内积，水平规约
        for (int j = 0; j < tile_n; ++j) {
            const float* kp = k_ptr + (n_start + j) * d;
            __m512 dot = _mm512_setzero_ps();
            for (int i = 0; i < D_UNROLL; ++i) {
                __m512 kv = _mm512_loadu_ps(kp + i * 16);
                dot = _mm512_fmadd_ps(qr[i], kv, dot);
            }
            // _mm512_reduce_add_ps：AVX-512 内置水平规约（编译器通常优化成
            // vextractf32x8 + vaddps 序列，与手写等价）
            s_buf[j] = _mm512_hadd_ps_scalar(dot) * scale
                       + mask_row[n_start + j];
        }

        // ── 2. Online softmax：m_new = max(m_val, max(s_buf)) ────────────────
        // 先计算 tile 内的 local max，从 -inf 开始（不从 m_val 开始）。
        // 这样可以精确判断 tile 内全屏蔽（关键！）：
        //   如果 tile 内所有 s_buf[j] = -inf → m_tile = -inf → 跳过。
        //   如果 tile 内有有限小于 m_val 的分数 → m_tile = 有限 → 就算 m_tile <= m_val 也不跳过。
        float m_tile = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < tile_n; ++j) {
            if (s_buf[j] > m_tile) m_tile = s_buf[j];
        }

        // 全屏蔽 tile：pragma no-finite-math-only 保证 -inf 比较正确。
        // m_tile == -inf 当且仅当所有 s_buf[j] = -inf。
        if (!(m_tile > -std::numeric_limits<float>::infinity())) {
            continue;
        }

        float m_new = (m_tile > m_val) ? m_tile : m_val;

        // correction factor = exp(m_old - m_new)
        // m_val=-inf（第一个有效 tile）：expf(-inf) = 0，zmm acc 清零重起
        float corr = expf(m_val - m_new);

        // rescale 历史累加器 acc *= corr
        {
            __m512 corr_v = _mm512_set1_ps(corr);
            for (int i = 0; i < D_UNROLL; ++i) {
                acc[i] = _mm512_mul_ps(acc[i], corr_v);
            }
        }
        l_val *= corr;

        // ── 3. 累加 p × V（当前 tile）──────────────────────────────────────
        for (int j = 0; j < tile_n; ++j) {
            float p = expf(s_buf[j] - m_new);
            l_val += p;

            __m512 p_v = _mm512_set1_ps(p);
            const float* vp = v_ptr + (n_start + j) * d;
            for (int i = 0; i < D_UNROLL; ++i) {
                __m512 vv = _mm512_loadu_ps(vp + i * 16);
                acc[i] = _mm512_fmadd_ps(p_v, vv, acc[i]);
            }
        }

        m_val = m_new;
    }

    // ── 4. 归一化 out /= l ────────────────────────────────────────────────────
    if (l_val > 0.0f) {
        __m512 inv_l = _mm512_set1_ps(1.0f / l_val);
        float* op = out_row;
        for (int i = 0; i < D_UNROLL; ++i, op += 16) {
            _mm512_storeu_ps(op, _mm512_mul_ps(acc[i], inv_l));
        }
    } else {
        std::fill(out_row, out_row + d, 0.0f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// fa_fwd_cpu_kernel_avx512 — 主函数
// ─────────────────────────────────────────────────────────────────────────────

void fa_fwd_cpu_kernel_avx512(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    const float* __restrict__ mask_ptr,
    float*       __restrict__ out_ptr,
    int H, int Hk, int Sq, int Sk, int d,
    float scale, int n_threads
) {
    const int kBlockN = (d >= 128) ? kCpuBlockN_d128 : kCpuBlockN_d64;

    omp_set_num_threads(n_threads);

    if (d == 128) {
        // D_UNROLL = 128 / 16 = 8  (8 个 zmm 寄存器持有 Q 行)
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int bidh = 0; bidh < H; ++bidh) {
            const int kv_head = bidh / (H / Hk);
            const float* q_head   = q_ptr   + bidh   * Sq * d;
            const float* k_head   = k_ptr   + kv_head * Sk * d;
            const float* v_head   = v_ptr   + kv_head * Sk * d;
            float*       out_head = out_ptr + bidh   * Sq * d;

            for (int m = 0; m < Sq; ++m) {
                compute_one_row_avx512<8>(
                    q_head   + m * d,
                    k_head, v_head,
                    mask_ptr + m * Sk,
                    out_head + m * d,
                    Sk, d, kBlockN, scale
                );
            }
        }
    } else if (d == 64) {
        // D_UNROLL = 64 / 16 = 4  (4 个 zmm 寄存器持有 Q 行)
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int bidh = 0; bidh < H; ++bidh) {
            const int kv_head = bidh / (H / Hk);
            const float* q_head   = q_ptr   + bidh   * Sq * d;
            const float* k_head   = k_ptr   + kv_head * Sk * d;
            const float* v_head   = v_ptr   + kv_head * Sk * d;
            float*       out_head = out_ptr + bidh   * Sq * d;

            for (int m = 0; m < Sq; ++m) {
                compute_one_row_avx512<4>(
                    q_head   + m * d,
                    k_head, v_head,
                    mask_ptr + m * Sk,
                    out_head + m * d,
                    Sk, d, kBlockN, scale
                );
            }
        }
    } else {
        // 不支持的 d，回退 scalar（理论上不应到达）
        fa_fwd_cpu_kernel_scalar(
            q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr,
            H, Hk, Sq, Sk, d, scale, n_threads
        );
    }
}
