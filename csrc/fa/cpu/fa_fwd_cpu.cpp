/*
 * fa_fwd_cpu.cpp — Flash Attention 2 Forward（CPU Scalar/OpenMP 实现）
 *
 * 本文件实现两件事：
 *   1. fa_fwd_cpu_kernel_scalar()：OpenMP 并行 + 编译器自动向量化的参考实现
 *      - 正确性基准，适合 CI / CPU-only 环境 / AVX2 不可用时的回退路径
 *      - 外层 OpenMP 并行：batch × head 维度（无同步）
 *      - 内层 tiled flash attention online softmax（kBlockM × kBlockN tile）
 *      - __builtin_prefetch(locality=3) 主动将 K/V tile 拉入 L1 cache
 *
 *   2. mha_fwd_with_mask_cpu()：对外接口，负责：
 *      a. 输入验证（shape / dtype / contiguous）
 *      b. bf16 → fp32 转换（CPU 无原生 bf16 SIMD）
 *      c. runtime dispatch：AVX2 可用时调用 fa_fwd_cpu_kernel_avx2，
 *                           否则调用 fa_fwd_cpu_kernel_scalar
 *      d. fp32 → bf16 转换，写入输出 tensor
 *
 * 编译选项
 * --------
 * 本文件以 -O3 -ffast-math -fopenmp 编译，允许浮点结合律以激活更多自动向量化。
 * AVX2 指令集不在本文件启用（由 fa_fwd_cpu_avx2.cpp 通过 #pragma GCC target 单独激活）。
 *
 * Online Softmax 算法（FA2 核心）
 * --------------------------------
 * 对每个 Q 行 i，迭代所有 K/V tile j：
 *   s_ij = Q_i · K_j^T / sqrt(d) + mask_ij        ← kBlockN 个 score
 *   m_new = max(m_i, max(s_ij))
 *   l_i   = l_i * exp(m_i - m_new) + sum(exp(s_ij - m_new))
 *   o_i   = o_i * exp(m_i - m_new) + exp(s_ij - m_new) @ V_j
 *   m_i   = m_new
 * 最终 output_i = o_i / l_i
 * 全程不需要存储 Sq×Sk attention 矩阵，内存开销 O(Sq×d)。
 */

// online softmax 依赖 -inf 哨兵值的比较正确性（初始 m_val = -inf）。
// -ffast-math 包含 -ffinite-math-only，会告诉编译器程序中无 inf/NaN，
// 导致 isinf() 恒返回 false、-inf 上的算术行为未定义。
// 通过 pragma 在本翻译单元内关闭该假设，恢复 IEEE 754 inf 语义。
#pragma GCC optimize("O3,no-finite-math-only")

#include "fa_fwd_cpu.h"
#include "../fa_fwd_op.h"

#include <torch/extension.h>
#include <omp.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// 内部工具
// ─────────────────────────────────────────────────────────────────────────────

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// 安全 exp：避免 -inf 输入导致 NaN（-inf → exp = 0）
static inline float safe_expf(float x) {
    // 对 score=-inf 时，exp(-inf)=0，是正确的行为（屏蔽位权重应为 0）
    // expf 本身对 -inf 返回 0，所以直接调用即可
    return expf(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// 单头单行的 online softmax attention 计算
// （作为内联函数，让编译器有充分的优化空间）
// ─────────────────────────────────────────────────────────────────────────────

// 处理一个 Q 行（row_q）对所有 K/V 序列的 attention
// q_row    : (d,)        fp32，当前 Q 行
// k_ptr    : (Sk, d)     fp32，K 矩阵（当前 head）
// v_ptr    : (Sk, d)     fp32，V 矩阵（当前 head）
// mask_row : (Sk,)       fp32，当前 Q 行对应的 mask 行
// out_row  : (d,)        fp32，输出（已清零）
// kBlockN  : tile 大小（运行时参数，根据 d 选取）
static void compute_one_row(
    const float* __restrict__ q_row,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    const float* __restrict__ mask_row,
    float*       __restrict__ out_row,
    int Sk, int d, int kBlockN, float scale
) {
    // Online softmax 统计量
    // m_val: 历史最大 score；初始为 -inf
    // l_val: 历史 exp 求和
    float m_val = -std::numeric_limits<float>::infinity();
    float l_val = 0.0f;

    // 逐 tile 迭代
    for (int n_start = 0; n_start < Sk; n_start += kBlockN) {
        const int n_end = std::min(n_start + kBlockN, Sk);
        const int tile_n = n_end - n_start;

        // ── 预取下一个 K/V tile 到 L1 cache ──────────────────────────────
        // locality=3: 请求保留在 L1（最高优先级，对应 _MM_HINT_T0）
        // 提前 kCpuPrefetchAhead(=2) 个 tile 预取，覆盖 L1 miss 延迟
        const int prefetch_start = n_start + kCpuPrefetchAhead * kBlockN;
        if (prefetch_start < Sk) {
            const char* k_next = reinterpret_cast<const char*>(
                k_ptr + prefetch_start * d);
            const char* v_next = reinterpret_cast<const char*>(
                v_ptr + prefetch_start * d);
            // 以 cache line（64B）为粒度分散预取
            for (int offset = 0; offset < tile_n * d * sizeof(float); offset += 64) {
                __builtin_prefetch(k_next + offset, 0, 3);
                __builtin_prefetch(v_next + offset, 0, 3);
            }
        }

        // ── 1. 计算 s = Q · K^T × scale + mask（当前 tile）────────────────
        // s_buf: (tile_n,) fp32，局部 score 缓冲
        // tile_n <= kBlockN <= 64，栈上分配安全
        float s_buf[64];  // 最大 tile_n = 64

        for (int j = 0; j < tile_n; ++j) {
            const float* k_row = k_ptr + (n_start + j) * d;
            float dot = 0.0f;
            // 编译器可以将此循环自动向量化（-O3 + -ffast-math）
            for (int k = 0; k < d; ++k) {
                dot += q_row[k] * k_row[k];
            }
            s_buf[j] = dot * scale + mask_row[n_start + j];
        }

        // ── 2. Online softmax：更新 m_new，rescale l 和 out ─────────────────
        // 先单独计算 tile 内的 local max（从 -inf 开始，不从 m_val 开始），
        // 以精确判断全屏蔽：m_tile=-inf 当且仅当 tile 内全部 score 为 -inf。
        float m_tile = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < tile_n; ++j) {
            if (s_buf[j] > m_tile) m_tile = s_buf[j];
        }

        // 全屏蔽 tile：pragma no-finite-math-only 保证 -inf 比较正确。
        if (!(m_tile > -std::numeric_limits<float>::infinity())) {
            continue;
        }

        float m_new = (m_tile > m_val) ? m_tile : m_val;

        // correction factor = exp(m_old - m_new)：rescale 历史统计量。
        // m_val=-inf（第一个有效 tile）：safe_expf(-inf) = 0，acc 清零重起。
        // m_val=有限：safe_expf(有限 - 有限)，正常 rescale。
        float corr = safe_expf(m_val - m_new);

        // rescale 历史累加器 out_row *= corr
        for (int k = 0; k < d; ++k) {
            out_row[k] *= corr;
        }
        l_val *= corr;

        // ── 3. 累加 softmax_prob × V（当前 tile）───────────────────────────
        for (int j = 0; j < tile_n; ++j) {
            float p = safe_expf(s_buf[j] - m_new);
            l_val += p;

            const float* v_row = v_ptr + (n_start + j) * d;
            for (int k = 0; k < d; ++k) {
                out_row[k] += p * v_row[k];
            }
        }

        m_val = m_new;
    }

    // ── 4. 归一化 out /= l ───────────────────────────────────────────────────
    // l_val = 0 当且仅当全 mask=-inf（全屏蔽），此时 out 已为 0，跳过除法
    if (l_val > 0.0f) {
        float inv_l = 1.0f / l_val;
        for (int k = 0; k < d; ++k) {
            out_row[k] *= inv_l;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// fa_fwd_cpu_kernel_scalar — Scalar/OpenMP 实现
// ─────────────────────────────────────────────────────────────────────────────

void fa_fwd_cpu_kernel_scalar(
    const float* __restrict__ q_ptr,     // (H, Sq, d)
    const float* __restrict__ k_ptr,     // (Hk, Sk, d)
    const float* __restrict__ v_ptr,     // (Hk, Sk, d)
    const float* __restrict__ mask_ptr,  // (Sq, Sk)  注：batch 维在外部循环
    float*       __restrict__ out_ptr,   // (H, Sq, d)
    int H, int Hk, int Sq, int Sk, int d,
    float scale, int n_threads
) {
    // kBlockN 根据 d 选取，目标：K_tile + V_tile ≤ L1 cache
    //   d=128: 32×128×4×2 = 32KB = L1（刚好），保守取 32
    //   d=64:  64×64×4×2  = 32KB = L1（刚好），保守取 64
    //   其他 d：以 L1=32KB 为上限推算
    const int kBlockN = (d >= 128) ? kCpuBlockN_d128 : kCpuBlockN_d64;

    omp_set_num_threads(n_threads);

    // 外层 OpenMP 并行：batch×head 已经在 mha_fwd_with_mask_cpu 外循环处理 batch，
    // 此处仅并行 head 维度
    // collapse(1) 即可，head 间完全独立
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int bidh = 0; bidh < H; ++bidh) {
        const int kv_head = bidh / (H / Hk);   // GQA: H/Hk 个 Q head 共享一个 KV head

        const float* q_head   = q_ptr   + bidh   * Sq * d;
        const float* k_head   = k_ptr   + kv_head * Sk * d;
        const float* v_head   = v_ptr   + kv_head * Sk * d;
        float*       out_head = out_ptr + bidh   * Sq * d;

        for (int m = 0; m < Sq; ++m) {
            const float* q_row   = q_head   + m * d;
            const float* mask_row = mask_ptr + m * Sk;
            float*       out_row = out_head  + m * d;

            // 清零输出行（online softmax 从 0 开始累加）
            std::fill(out_row, out_row + d, 0.0f);

            compute_one_row(q_row, k_head, v_head, mask_row, out_row,
                            Sk, d, kBlockN, scale);
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// mha_fwd_with_mask_cpu — 对外接口（dispatch + bf16 转换）
// ─────────────────────────────────────────────────────────────────────────────

torch::Tensor mha_fwd_with_mask_cpu(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask
) {
    // ── 输入验证 ──────────────────────────────────────────────────────────────
    TORCH_CHECK(!q.is_cuda(),    "mha_fwd_with_mask_cpu: q must be a CPU tensor");
    TORCH_CHECK(!k.is_cuda(),    "mha_fwd_with_mask_cpu: k must be a CPU tensor");
    TORCH_CHECK(!v.is_cuda(),    "mha_fwd_with_mask_cpu: v must be a CPU tensor");
    TORCH_CHECK(!mask.is_cuda(), "mha_fwd_with_mask_cpu: mask must be a CPU tensor");

    TORCH_CHECK(q.dtype()    == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype()    == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype()    == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(mask.dtype() == torch::kBFloat16, "mask must be bfloat16");

    TORCH_CHECK(q.is_contiguous(),    "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(),    "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(),    "v must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");

    TORCH_CHECK(q.dim() == 4,    "q must be 4D (B, H, Sq, d)");
    TORCH_CHECK(k.dim() == 4,    "k must be 4D (B, Hk, Sk, d)");
    TORCH_CHECK(v.dim() == 4,    "v must be 4D (B, Hk, Sk, d)");
    TORCH_CHECK(mask.dim() == 4, "mask must be 4D (B, 1, Sq, Sk)");

    const int B   = q.size(0);
    const int H   = q.size(1);
    const int Sq  = q.size(2);
    const int d   = q.size(3);
    const int Hk  = k.size(1);
    const int Sk  = k.size(2);

    TORCH_CHECK(k.size(0) == B && v.size(0) == B && mask.size(0) == B,
                "Batch size mismatch");
    TORCH_CHECK(k.size(3) == d && v.size(3) == d,
                "head_dim mismatch between q/k/v");
    TORCH_CHECK(k.size(2) == Sk && v.size(2) == Sk,
                "seqlen_k mismatch between k/v");
    TORCH_CHECK(H % Hk == 0,
                "n_heads_q must be divisible by n_heads_kv (GQA)");
    TORCH_CHECK(d == 64 || d == 128,
                "CPU FA2: only head_dim=64 or head_dim=128 is supported");
    TORCH_CHECK(mask.size(1) == 1 && mask.size(2) == Sq && mask.size(3) == Sk,
                "mask shape must be (B, 1, Sq, Sk)");

    // ── bf16 → fp32（CPU 无原生 bf16 SIMD，所有计算在 fp32 下进行）──────────
    // 使用 PyTorch 的 .to(float32) 做批量转换（内部会用 SIMD 加速）
    auto q_f    = q.to(torch::kFloat32);       // (B, H, Sq, d) fp32
    auto k_f    = k.to(torch::kFloat32);       // (B, Hk, Sk, d) fp32
    auto v_f    = v.to(torch::kFloat32);       // (B, Hk, Sk, d) fp32
    auto mask_f = mask.to(torch::kFloat32);    // (B, 1, Sq, Sk) fp32

    // ── 分配输出（fp32，稍后转 bf16）─────────────────────────────────────────
    auto out_f = torch::zeros({B, H, Sq, d}, q_f.options());  // (B, H, Sq, d) fp32

    // softmax scale = 1 / sqrt(d)
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    // ── 获取线程数（与 serving 对齐）──────────────────────────────────────────
    static const int n_threads = _get_fa_cpu_nthreads();

    // ── 运行时 dispatch：AVX-512 > AVX2 > Scalar ─────────────────────────────
    // 优先级说明：
    //   1. AVX-512 (512-bit): 每步 16 float，FMA 吞吐是 AVX2 的 2×。
    //      Ice Lake (Xeon 8352Y) 无频率降档，可全速运行。
    //   2. AVX2+FMA (256-bit): 每步 8 float，性能约 AVX-512 的 50%。
    //   3. Scalar+OpenMP: 依赖编译器自动向量化（-O3 -ffast-math），作为兜底。
    //
    // 编译时宏 FA_CPU_AVX512_COMPILED / FA_CPU_AVX2_COMPILED 控制函数是否可链接。
    // 由于 fa_fwd_cpu_avx512.cpp / fa_fwd_cpu_avx2.cpp 均无条件编译（仅靠
    // #pragma GCC target 激活目标指令集），这两个符号始终存在于链接单元中，
    // 故直接用运行时检测函数决定调用路径，无需编译时条件编译。
    static const bool use_avx512 = _cpu_has_avx512();
    static const bool use_avx2   = use_avx512 ? false : _cpu_has_avx2_fma();

    // ── 逐 batch 调用 kernel ──────────────────────────────────────────────────
    for (int bidb = 0; bidb < B; ++bidb) {
        const float* q_b    = q_f.data_ptr<float>()    + bidb * H  * Sq * d;
        const float* k_b    = k_f.data_ptr<float>()    + bidb * Hk * Sk * d;
        const float* v_b    = v_f.data_ptr<float>()    + bidb * Hk * Sk * d;
        // mask: (B, 1, Sq, Sk)，batch stride = Sq * Sk
        const float* mask_b = mask_f.data_ptr<float>() + bidb * Sq * Sk;
        float*       out_b  = out_f.data_ptr<float>()  + bidb * H  * Sq * d;

        if (use_avx512) {
            fa_fwd_cpu_kernel_avx512(
                q_b, k_b, v_b, mask_b, out_b,
                H, Hk, Sq, Sk, d, scale, n_threads
            );
        } else if (use_avx2) {
            fa_fwd_cpu_kernel_avx2(
                q_b, k_b, v_b, mask_b, out_b,
                H, Hk, Sq, Sk, d, scale, n_threads
            );
        } else {
            fa_fwd_cpu_kernel_scalar(
                q_b, k_b, v_b, mask_b, out_b,
                H, Hk, Sq, Sk, d, scale, n_threads
            );
        }
    }

    // ── fp32 → bf16 输出 ─────────────────────────────────────────────────────
    return out_f.to(torch::kBFloat16);
}
