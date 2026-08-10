/*
 * fa_fwd_op.cu — Flash Attention Forward with Additive Mask
 *                PyTorch C++ 接口实现
 *
 * 对外暴露：
 *   mha_fwd_with_mask_cuda(q, k, v, mask) -> Tensor
 *
 * 输入约定：
 *   q, k, v : (B, n_heads, seqlen, head_dim)，bfloat16，CUDA tensor，连续
 *   mask    : (B, 1, seqlen_q, seqlen_k_or_rounded)，bfloat16，CUDA tensor，连续
 *             加法 mask，语义与 PyTorch SDPA 对齐：softmax(S·scale + mask)
 *             —— 0=可见，-inf=屏蔽，也支持任意有限值偏置（ALiBi 风格）
 *             seqlen_k_or_rounded >= seqlen_k，允许调用方预先 pad 到 kBlockN 整数倍（推荐）
 *
 * 输出：
 *   out     : (B, n_heads, seqlen_q, head_dim)，bfloat16
 *
 * 内部处理：
 *   - 若 mask 的 Sk 维度已是 kBlockN 整数倍（pre-padded），则 zero-copy 直接使用
 *   - 否则自动 pad 到 kBlockN 的整数倍（pad 填 -inf）
 *
 * 注意：
 *   - 只支持 head_dim ∈ {64, 128}（bf16），更多 hdim 可按需扩展
 *   - 只支持 GQA（n_heads_q >= n_heads_kv，n_heads_q % n_heads_kv == 0）
 *   - 不支持 dropout、causal mask、RoPE、KV-cache
 *     （alibi 类相对位置偏置可直接通过有限值加法 mask 表达）
 */

#include "fa_fwd_op.h"
#include "fa_fwd_launch.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using namespace FA_MASK_NAMESPACE;

// ── 工具：计算 ceil_div ───────────────────────────────────────────────────────
static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// ── CUDA 实现 ─────────────────────────────────────────────────────────────────
torch::Tensor mha_fwd_with_mask_cuda(
    const torch::Tensor& q,      // (B, H, Sq, d)
    const torch::Tensor& k,      // (B, Hk, Sk, d)
    const torch::Tensor& v,      // (B, Hk, Sk, d)
    const torch::Tensor& mask    // (B, 1, Sq, Sk)  additive mask (0=可见, -inf=屏蔽)，bf16
) {
    // ── 输入验证 ──────────────────────────────────────────────────────────────
    TORCH_CHECK(q.is_cuda(),    "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(),    "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(),    "v must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");

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

    const int B       = q.size(0);
    const int H       = q.size(1);
    const int Sq      = q.size(2);
    const int d       = q.size(3);
    const int Hk      = k.size(1);
    const int Sk      = k.size(2);

    TORCH_CHECK(k.size(0) == B && v.size(0) == B && mask.size(0) == B,
                "Batch size mismatch");
    TORCH_CHECK(k.size(3) == d && v.size(3) == d, "head_dim mismatch between q/k/v");
    TORCH_CHECK(k.size(2) == Sk && v.size(2) == Sk, "seqlen_k mismatch between k/v");
    TORCH_CHECK(H % Hk == 0, "n_heads must be divisible by n_kv_heads (GQA)");
    TORCH_CHECK(d == 64 || d == 128,
                "Only head_dim=64 or head_dim=128 is supported");

    // ── 计算 Sk_rounded / Sq_rounded（各对齐到整数块）─────────────────────────
    // copy_g2s_mask（cp.async 版）与 TMA 版均无条件搬运整个 (kBlockM, kBlockN) tile，
    // 块大小必须与实际运行的 kernel 配置一致：
    //   sm89 路径: hdim=64 → (128,128)，hdim=128 → (64,64)（见 fa_fwd_launch.h）
    //   sm120 路径: hdim=64/128 均为 (128,64)（见 fa_mask_sm120_block_size）
    // 注意 sm120 splitkv 的 Split-M 变体 kernel 内用 kBlockM=64，但 host 侧 padding
    // 仍按基准 (128,64) 对齐——128 的整数倍必然 64 对齐，M64 无需额外处理。
    // 两个方向越界的 pad 值均为 -inf（mask 语义：屏蔽 → softmax 后归零，不影响结果）
    const bool use_sm120 = fa_mask_sm120_supported();
    int kBlockM, kBlockN;
    if (use_sm120) {
        fa_mask_sm120_block_size(d, kBlockM, kBlockN);
    } else {
        kBlockN = (d == 64) ? 128 : 64;
        kBlockM = (d == 64) ? 128 : 64;
    }
    const int Sk_rounded = ceil_div_int(Sk, kBlockN) * kBlockN;
    const int Sq_rounded = ceil_div_int(Sq, kBlockM) * kBlockM;

    // mask 允许以下合法形状（Sq_mask 为 mask 实际行数，Sk_mask 为实际列数）：
    //   (B, 1, Sq,         Sk)          — 未 pre-pad，算子内部双向 pad
    //   (B, 1, Sq,         Sk_rounded)  — 列已 pre-pad
    //   (B, 1, Sq_rounded, Sk_rounded)  — 双向已 pre-pad（zero-copy）
    const int64_t Sq_mask = mask.size(2);
    const int64_t Sk_mask = mask.size(3);
    TORCH_CHECK(
        mask.size(1) == 1
        && (Sq_mask == Sq || Sq_mask == Sq_rounded)
        && (Sk_mask == Sk || Sk_mask == Sk_rounded),
        "mask shape must be (B, 1, Sq_or_Sq_rounded, Sk_or_Sk_rounded)"
    );

    torch::Tensor mask_padded;
    const bool sq_needs_pad = (static_cast<int>(Sq_mask) < Sq_rounded);
    const bool sk_needs_pad = (static_cast<int>(Sk_mask) < Sk_rounded);
    if (!sq_needs_pad && !sk_needs_pad) {
        // 调用方已双向 pre-pad（zero-copy，确保 contiguous）
        mask_padded = mask.is_contiguous() ? mask : mask.contiguous();
    } else {
        // 至少有一个方向需要 pad，填 -inf
        // torch::nn::functional::pad 的 padding 顺序为从最后维度向前：
        //   {left_last, right_last, left_2nd_last, right_2nd_last, ...}
        //   即 {pad_Sk_left, pad_Sk_right, pad_Sq_left, pad_Sq_right}
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const int pad_sk = sk_needs_pad ? (Sk_rounded - static_cast<int>(Sk_mask)) : 0;
        const int pad_sq = sq_needs_pad ? (Sq_rounded - static_cast<int>(Sq_mask)) : 0;
        mask_padded = torch::nn::functional::pad(
            mask,
            torch::nn::functional::PadFuncOptions({0, pad_sk, 0, pad_sq})
            .mode(torch::kConstant).value(neg_inf)
        ).contiguous();
    }

    // ── 构造输出 tensor ───────────────────────────────────────────────────────
    auto out = torch::empty_like(q);  // (B, H, Sq, d), bf16

    // ── 填充 params ──────────────────────────────────────────────────────────
    FA_mask_params params;
    params.q_ptr    = q.data_ptr();
    params.k_ptr    = k.data_ptr();
    params.v_ptr    = v.data_ptr();
    params.o_ptr    = out.data_ptr();
    params.mask_ptr = mask_padded.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);

    params.q_row_stride   = q.stride(2);   // stride over seqlen_q
    params.k_row_stride   = k.stride(2);
    params.v_row_stride   = v.stride(2);
    params.o_row_stride   = out.stride(2);

    params.q_head_stride  = q.stride(1);   // stride over heads
    params.k_head_stride  = k.stride(1);
    params.v_head_stride  = v.stride(1);
    params.o_head_stride  = out.stride(1);

    // mask_padded: (B, 1, Sq_rounded, Sk_rounded) row-major
    //   batch_stride = stride over batch dim
    //   row_stride   = Sk_rounded（Sq_rounded 行方向 stride）
    params.mask_batch_stride = mask_padded.stride(0);   // B 维 stride
    params.mask_row_stride   = mask_padded.stride(2);   // Sq 维 stride（= Sk_rounded）

    params.b              = B;
    params.h              = H;
    params.h_k            = Hk;
    params.h_h_k_ratio    = H / Hk;
    params.seqlen_q         = Sq;
    params.seqlen_k         = Sk;
    params.seqlen_k_rounded = Sk_rounded;
    params.seqlen_q_rounded = Sq_rounded;
    params.d              = d;
    params.is_bf16        = true;

    const float softmax_scale     = 1.0f / sqrtf(static_cast<float>(d));
    params.scale_softmax          = softmax_scale;
    params.scale_softmax_log2     = softmax_scale * static_cast<float>(M_LOG2E);

    // ── 启动 kernel ──────────────────────────────────────────────────────────
    at::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 环境变量 FA_PERSISTENT=1 时使用 persistent kernel（1D grid + 步长调度）
    static const bool use_persistent = []() {
        const char* env = std::getenv("FA_PERSISTENT");
        return env && (std::strcmp(env, "1") == 0);
    }();
    // 环境变量 FA_SPLITKV=0 可禁用 split KV（默认启用，由启发式决定 num_splits）
    static const bool splitkv_enabled = []() {
        const char* env = std::getenv("FA_SPLITKV");
        return !(env && (std::strcmp(env, "0") == 0));
    }();
    // 环境变量 FA_SPLITM=0 可禁用 splitkv 的 kBlockM=64 变体（默认启用）
    static const bool splitm_enabled = []() {
        const char* env = std::getenv("FA_SPLITM");
        return !(env && (std::strcmp(env, "0") == 0));
    }();

    if (use_sm120 && use_persistent) {
        if (d == 64) { run_mha_fwd_mask_hdim64_sm120_persistent(params, stream); }
        else         { run_mha_fwd_mask_hdim128_sm120_persistent(params, stream); }
        return out;
    }

    // Split-KV：grid 填不满 SM 时按 K 维切分，多 CTA 并行后 combine 归约
    if (use_sm120 && splitkv_enabled) {
        int num_splits = fa_mask_sm120_num_splits(params, kBlockM, kBlockN);
        if (num_splits > 1) {
            // Split-M 判定：kBlockM=64 变体让 m_block 数翻倍（不增加 combine 开销地提升并行度）
            //   ① Sq<=64：kBlockM=128 会浪费半块 padding 计算，M64 严格更优
            //   ② grid 严重填不满（< 0.5 wave）：M64 把并行度翻倍
            bool use_m64 = false;
            if (splitm_enabled) {
                const int num_sms = fa_mask_sm120_num_sms();
                const int64_t grid_ctas =
                    (int64_t)B * H * ceil_div_int(Sq, 128) * num_splits;
                if (Sq <= 64 || grid_ctas < num_sms / 2) {
                    use_m64 = true;
                    num_splits = fa_mask_sm120_num_splits(params, 64, kBlockN);
                }
            }
            // O_partial 用 bf16（partial 流量减半；~0.4% 相对误差 < bf16 输出量化误差），LSE 保持 fp32
            torch::Tensor oaccum   = torch::empty({num_splits, B, H, Sq_rounded, d}, q.options());
            torch::Tensor lseaccum = torch::empty({num_splits, B, H, Sq_rounded},
                                                  q.options().dtype(torch::kFloat32));
            params.oaccum_ptr   = oaccum.data_ptr();
            params.lseaccum_ptr = lseaccum.data_ptr();
            params.num_splits   = num_splits;
            if (use_m64) {
                if (d == 64) { run_mha_fwd_mask_hdim64_sm120_splitkv_m64(params, stream); }
                else         { run_mha_fwd_mask_hdim128_sm120_splitkv_m64(params, stream); }
            } else {
                if (d == 64) { run_mha_fwd_mask_hdim64_sm120_splitkv(params, stream); }
                else         { run_mha_fwd_mask_hdim128_sm120_splitkv(params, stream); }
            }
            return out;
        }
    }

    if (use_sm120) {
        if (d == 64) { run_mha_fwd_mask_hdim64_sm120(params, stream); }
        else         { run_mha_fwd_mask_hdim128_sm120(params, stream); }
    } else {
        if (d == 64) { run_mha_fwd_mask_hdim64(params, stream); }
        else         { run_mha_fwd_mask_hdim128(params, stream); }
    }

    return out;
}

// ── CPU 桩（不支持，仅用于编译完整性）──────────────────────────────────────
torch::Tensor mha_fwd_with_mask_cpu(
    const torch::Tensor& /*q*/,
    const torch::Tensor& /*k*/,
    const torch::Tensor& /*v*/,
    const torch::Tensor& /*mask*/
) {
    TORCH_CHECK(false, "mha_fwd_with_mask is only supported on CUDA");
}
