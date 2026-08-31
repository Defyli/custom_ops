/*
 * fa_fwd_op.cu — Flash Attention Forward with Additive Mask
 *                PyTorch C++ 接口实现
 *
 * 对外暴露：
 *   mha_fwd_with_mask_cuda(q, k, v, mask) -> Tensor
 *
 * 输入约定：
 *   q, k, v : (B, n_heads, seqlen, head_dim)，fp16 或 bf16，CUDA tensor，连续
 *   mask    : (B, 1, seqlen_q, seqlen_k_or_padded)，与 q 同 dtype，CUDA tensor，连续
 *             加法 mask，语义与 PyTorch SDPA 对齐：softmax(S·scale + mask)
 *             —— 0=可见，-inf=屏蔽，也支持任意有限值偏置（ALiBi 风格）
 *             k 维 ≥ seqlen_k（== seqlen_k 零拷贝，或调用方预 pad——
 *             col ≥ Sk 恒为屏蔽，越界列内容被 kernel 忽略）
 *
 * 输出：
 *   out     : (B, n_heads, seqlen_q, head_dim)，与输入同 dtype
 *
 * 内部处理：
 *   - Sk % 8 == 0（128-bit 向量对齐）时零拷贝直接使用（主路径）；
 *   - Sk % 8 != 0 时算子自动 pad 到 8 倍数（一次拷贝：K/V 零填充 + mask 截断
 *     后 pad -inf，见下方「自动 pad」注释），对上层透明；
 *   - Sq 任意（行谓词 + epilogue 丢弃越界行）
 *
 * 架构分发：本文件零架构感知——校验/自动 pad/填 params 后唯一一次调用
 * fa_fwd_launch()（见 fa_fwd_launch.h，FA 的分发标准入口；策略与 kernel
 * 选择全部内封在各家族 fa_launch_smXX 中）。
 *
 * 注意：
 *   - 只支持 head_dim ∈ {64, 128}（fp16/bf16），更多 hdim 可按需扩展
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

// ── CUDA 实现 ─────────────────────────────────────────────────────────────────
torch::Tensor mha_fwd_with_mask_cuda(
    const torch::Tensor& q,      // (B, H, Sq, d)
    const torch::Tensor& k,      // (B, Hk, Sk, d)
    const torch::Tensor& v,      // (B, Hk, Sk, d)
    const torch::Tensor& mask    // (B, 1, Sq, Sk)  additive mask (0=可见, -inf=屏蔽)，fp16/bf16
) {
    // ── 输入验证 ──────────────────────────────────────────────────────────────
    TORCH_CHECK(q.is_cuda(),    "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(),    "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(),    "v must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");

    // dtype：fp16 全架构可用；bf16 需 SM80+ tensor core（Volta 不支持，
    // 无可用路径时由 fa_fwd_launch 显式报错，不在此重复架构判定）
    const bool is_fp16 = (q.dtype() == torch::kHalf);
    const bool is_bf16 = (q.dtype() == torch::kBFloat16);
    TORCH_CHECK(is_fp16 || is_bf16, "q must be float16 or bfloat16");
    const torch::ScalarType expect_dtype = is_fp16 ? torch::kHalf : torch::kBFloat16;
    TORCH_CHECK(k.dtype()    == expect_dtype, "k dtype must match q");
    TORCH_CHECK(v.dtype()    == expect_dtype, "v dtype must match q");
    TORCH_CHECK(mask.dtype() == expect_dtype, "mask dtype must match q");

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

    // ── mask 契约（无 pad：对齐要求降至 8）─────────────────────────────────
    // 16-bit × 8 = 128-bit cp.async / TMA 16B 行对齐 → Sk 与 mask 列数只需 8 倍数
    // （同 mixed_gemm 的 K%8 约束）。Sq 任意（Q 行谓词 + epilogue 丢弃）。
    // kernel 语义：col ≥ Sk 恒为屏蔽，mask 越界列内容被忽略（预 pad 的 -inf 列
    // 不再是正确性依赖）。

    // mask 合法形状（Sq_mask/Sk_mask 为 mask 实际行/列数）：
    //   q 维：Sq_mask >= Sq 即可，无需对齐——越界行的输出被 epilogue 丢弃，
    //         sm89/sm70 用谓词跳过、sm120 由 TMA 原生 OOB zero-fill，均不越界读
    //   k 维：Sk_mask >= Sk（== Sk 零拷贝；或调用方预 pad）
    const int64_t Sq_mask = mask.size(2);
    const int64_t Sk_mask = mask.size(3);
    TORCH_CHECK(
        mask.size(1) == 1 && Sq_mask >= Sq && Sk_mask >= Sk,
        "mask shape must be (B, 1, >=Sq, k-dim >= Sk)"
    );

    // ── 自动 pad（Sk % 8 != 0 时算子层兑底，对上层透明）─────────────────────
    // 对齐要求源于 128-bit 向量粒度的边界谓词（cp.async 列向量 / TMA 16B 行对齐）：
    //   Sk8 = ceil(Sk/8)*8，K/V pad 零填充行，mask 截断到 Sk 列后 pad 到 Sk8、
    //   pad 列填 -inf（col ∈ [Sk, Sk8) 语义上属 k 越界 → 屏蔽；K/V 零填充保证
    //   score 有限，无 NaN×-inf 风险）。
    // 仅在未对齐时发生一次拷贝；Sk % 8 == 0 时零拷贝（主路径不受影响）。
    // 注：Sk%8!=0 ⇒ Sk%64!=0 ⇒ pad 至多 <8 列，不引入新 tile，边界 tile 略增。
    torch::Tensor k_pad_storage, v_pad_storage, mask_pad_storage;
    int64_t Sk8 = Sk;                                    // 实际生效的 k 维长度
    if (Sk % 8 != 0) {
        Sk8 = (Sk + 7) / 8 * 8;
        k_pad_storage   = torch::constant_pad_nd(
            k, {0, 0, 0, 0, 0, Sk8 - Sk, 0, 0}, 0.0);
        v_pad_storage   = torch::constant_pad_nd(
            v, {0, 0, 0, 0, 0, Sk8 - Sk, 0, 0}, 0.0);
        const float neg_inf = -std::numeric_limits<float>::infinity();
        mask_pad_storage = torch::constant_pad_nd(
            mask.narrow(3, 0, Sk),                       // 截断到 Sk 列（越界列语义已死）
            {0, Sk8 - Sk, 0, 0, 0, 0, 0, 0}, neg_inf);
    } else if (Sk_mask % 8 != 0) {
        // Sk 已对齐但 mask 列数未对齐：仅 pad mask（pad 列 >= Sk 被 kernel 忽略，0 即可）
        const int64_t Sk_mask8 = (Sk_mask + 7) / 8 * 8;
        mask_pad_storage = torch::constant_pad_nd(
            mask, {0, Sk_mask8 - Sk_mask, 0, 0, 0, 0, 0, 0}, 0.0);
    }
    const torch::Tensor &k_eff   = (Sk % 8 != 0) ? k_pad_storage   : k;
    const torch::Tensor &v_eff   = (Sk % 8 != 0) ? v_pad_storage   : v;
    const torch::Tensor &mask_padded = mask_pad_storage.defined()
        ? mask_pad_storage : mask;

    // ── 构造输出 tensor ───────────────────────────────────────────────────────
    auto out = torch::empty_like(q);  // (B, H, Sq, d)，fp16/bf16

    // ── 填充 params ──────────────────────────────────────────────────────────
    FA_mask_params params;
    params.q_ptr    = q.data_ptr();
    params.k_ptr    = k_eff.data_ptr();
    params.v_ptr    = v_eff.data_ptr();
    params.o_ptr    = out.data_ptr();
    params.mask_ptr = mask_padded.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k_eff.stride(0);
    params.v_batch_stride = v_eff.stride(0);
    params.o_batch_stride = out.stride(0);

    params.q_row_stride   = q.stride(2);   // stride over seqlen_q
    params.k_row_stride   = k_eff.stride(2);
    params.v_row_stride   = v_eff.stride(2);
    params.o_row_stride   = out.stride(2);

    params.q_head_stride  = q.stride(1);   // stride over heads
    params.k_head_stride  = k_eff.stride(1);
    params.v_head_stride  = v_eff.stride(1);
    params.o_head_stride  = out.stride(1);

    // mask: (B, 1, mask_seqlen_q, mask_seqlen_k) row-major
    //   batch_stride = stride over batch dim；row_stride = q 维 stride（= pad 后实际列数）
    //   mask_seqlen_q = 实际行数（≥ Sq，q 维不 pad，kernel 侧谓词/TMA OOB 处理）
    //   mask_seqlen_k = pad 后实际列数（%8==0；kernel 侧列谓词/强制 -inf）
    params.mask_batch_stride = mask_padded.stride(0);   // B 维 stride
    params.mask_row_stride   = mask_padded.stride(2);   // q 维 stride（= mask 实际列数）
    params.mask_seqlen_q     = static_cast<int>(mask_padded.size(2));
    params.mask_seqlen_k     = static_cast<int>(mask_padded.size(3));

    params.b              = B;
    params.h              = H;
    params.h_k            = Hk;
    params.h_h_k_ratio    = H / Hk;
    params.seqlen_q         = Sq;
    params.seqlen_k         = static_cast<int>(Sk8);   // 自动 pad 后的生效长度（%8==0）
    // seqlen_q_rounded / num_splits / oaccum_ptr / lseaccum_ptr 由所选架构路径
    // 的 fa_launch_smXX 按其 tile 配置填写（见 fa_fwd_launch.h）
    params.d              = d;
    params.is_bf16        = is_bf16;

    const float softmax_scale     = 1.0f / sqrtf(static_cast<float>(d));
    params.scale_softmax          = softmax_scale;
    params.scale_softmax_log2     = softmax_scale * static_cast<float>(M_LOG2E);

    // ── 启动 kernel：一次调用，架构/tile/split 策略全自适应 ───────────────────
    at::cuda::CUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fa_fwd_launch(params, stream);

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
