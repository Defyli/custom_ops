/**
 * pack_and_prepare.cu  (vectorized edition)
 *
 * 三个 CUDA Kernel 的实现，融合：
 *   Kernel 1: pack_tokens_kernel  — 替换 pack_jagged_tokens(mode=4)
 *   Kernel 2: gather_rope_kernel  — 替换 cos/sin 索引
 *   Kernel 3: build_attn_mask_kernel — 替换 _build_request_attn_mask
 *
 * ── 向量化优化（Kernel 1 / 2）─────────────────────────────────────────────────
 *
 * bf16 逐元素 memcpy 类 kernel 的瓶颈在于内存带宽。
 * 向量化策略（VEC = 8）：
 *   - 将 float4（128-bit = 8 个 bf16）作为访存单元，一次 LDG.128 / STG.128。
 *   - 每个线程负责输出 tensor 中一行（pos 维度）的 D/8 个 float4，循环展开。
 *   - 要求 D % 8 == 0（线上 D=512 满足），不满足时自动退化为标量路径。
 *
 * Kernel 3（attn_mask）是 control-flow 类，每个 thread 的逻辑不同，暂不向量化。
 *
 * 设计假设：
 *   - B=1（单请求），无 batch 维度处理
 *   - 所有 Tensor 在同一 GPU 上
 *   - 输入 dtype: bf16
 *   - S_max 固定，由 host 侧传入
 *
 * 注意：PyTorch 编译时启用了 -D__CUDA_NO_BFLOAT16_CONVERSIONS__，
 * 不能使用 data_ptr<__nv_bfloat16>()，改用 data_ptr<at::BFloat16>()
 * + 内部 reinterpret_cast。
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <limits>
#include "pack_and_prepare.h"

// ── 指针类型转换宏 ────────────────────────────────────────────────────────────
#define AS_BF16_PTR(p)       (reinterpret_cast<__nv_bfloat16*>(p))
#define AS_CONST_BF16_PTR(p) (reinterpret_cast<const __nv_bfloat16*>(p))

// ── 向量化宽度：一次 float4 = 8 个 bf16 ──────────────────────────────────────
#define VEC 8

// float4 作为 128-bit 访存单元
// 8 个 bf16 = 4 个 uint32 = 1 个 float4，bit pattern 完全等价
using f4 = float4;


// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: pack_tokens_kernel（向量化版）
//
// 每个线程负责输出序列中一个 token（pos），对应一整行 D 个 bf16。
// 若 D % VEC == 0，使用 float4（128-bit）向量化读写：
//   · src 行首地址天然 16-byte 对齐（PyTorch 保证 >= 256-byte 起始对齐）
//   · 每个线程循环 D/VEC 次，每次 LDG.128 + STG.128
// 否则退化到标量逐元素。
//
// 同时在 item_mask 中写 bool（每 token 只写一次，无竞争）。
//
// Grid:  ceil(S_max / PACK_BLOCK_T)      （1D，只覆盖 pos 维度）
// Block: PACK_BLOCK_T
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pack_tokens_kernel(
    const __nv_bfloat16* __restrict__ h_s,
    const __nv_bfloat16* __restrict__ h_c,
    const __nv_bfloat16* __restrict__ h_i,
    __nv_bfloat16*       __restrict__ h_dense,
    bool*                __restrict__ item_mask,
    int64_t s_len,
    int64_t c_len,
    int64_t i_len,
    int64_t S_max,
    int64_t D
) {
    const int64_t pos = (int64_t)blockIdx.x * PACK_BLOCK_T + threadIdx.x;
    if (pos >= S_max) return;

    const int64_t valid_tokens = s_len + c_len + i_len;
    const int64_t item_pos     = s_len + c_len;

    // 确定本 token 的源行
    const __nv_bfloat16* src = nullptr;
    int64_t src_row = 0;
    bool is_padding = false;
    bool is_item    = false;

    if (pos < s_len) {
        src     = h_s;
        src_row = pos;
    } else if (pos < item_pos) {
        src     = h_c;
        src_row = pos - s_len;
    } else if (pos < valid_tokens) {
        src     = h_i;
        src_row = pos - item_pos;
        is_item = true;
    } else {
        is_padding = true;
    }

    __nv_bfloat16* dst = h_dense + pos * D;

    // ── 向量化路径（D % VEC == 0）────────────────────────────────────────────
    if (D % VEC == 0) {
        const int64_t n_vec = D / VEC;
        if (!is_padding) {
            const f4* src_v = reinterpret_cast<const f4*>(src + src_row * D);
            f4*       dst_v = reinterpret_cast<f4*>(dst);
            #pragma unroll 4
            for (int64_t v = 0; v < n_vec; ++v) {
                dst_v[v] = src_v[v];
            }
        } else {
            f4* dst_v = reinterpret_cast<f4*>(dst);
            const f4 zero = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll 4
            for (int64_t v = 0; v < n_vec; ++v) {
                dst_v[v] = zero;
            }
        }
    } else {
        // ── 标量回退路径 ──────────────────────────────────────────────────────
        if (!is_padding) {
            const __nv_bfloat16* src_row_ptr = src + src_row * D;
            for (int64_t d = 0; d < D; ++d) {
                dst[d] = src_row_ptr[d];
            }
        } else {
            for (int64_t d = 0; d < D; ++d) {
                dst[d] = BF16_ZERO;
            }
        }
    }

    // item_mask：仅写有效区（padding 行不写）
    if (pos < valid_tokens) {
        item_mask[pos] = is_item;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: gather_rope_kernel（向量化版）
//
// 对 cos/sin 两路 gather，每个线程负责输出序列中一个 seq_pos，
// 将 static_cos/sin[pid, :] 整行复制到 cos_out/sin_out[seq_pos, :]。
// 使用 float4（128-bit）向量化读写，head_dim % VEC == 0 时启用。
//
// Grid:  ceil(S_max / ROPE_BLOCK_T)      （1D）
// Block: ROPE_BLOCK_T
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gather_rope_kernel(
    const __nv_bfloat16* __restrict__ static_cos,
    const __nv_bfloat16* __restrict__ static_sin,
    __nv_bfloat16*       __restrict__ cos_out,
    __nv_bfloat16*       __restrict__ sin_out,
    int64_t s_len,
    int64_t c_len,
    int64_t S_max,
    int64_t head_dim
) {
    const int64_t seq_pos = (int64_t)blockIdx.x * ROPE_BLOCK_T + threadIdx.x;
    if (seq_pos >= S_max) return;

    const int64_t item_pos = s_len + c_len;
    const int64_t pid      = (seq_pos < item_pos) ? seq_pos : item_pos;

    const __nv_bfloat16* cos_src = static_cos + pid     * head_dim;
    const __nv_bfloat16* sin_src = static_sin + pid     * head_dim;
    __nv_bfloat16*       cos_dst = cos_out    + seq_pos * head_dim;
    __nv_bfloat16*       sin_dst = sin_out    + seq_pos * head_dim;

    // ── 向量化路径（head_dim % VEC == 0）────────────────────────────────────
    if (head_dim % VEC == 0) {
        const int64_t n_vec = head_dim / VEC;
        const f4* cs = reinterpret_cast<const f4*>(cos_src);
        const f4* ss = reinterpret_cast<const f4*>(sin_src);
        f4*       cd = reinterpret_cast<f4*>(cos_dst);
        f4*       sd = reinterpret_cast<f4*>(sin_dst);
        #pragma unroll 4
        for (int64_t v = 0; v < n_vec; ++v) {
            cd[v] = cs[v];
            sd[v] = ss[v];
        }
    } else {
        // ── 标量回退路径 ──────────────────────────────────────────────────────
        for (int64_t d = 0; d < head_dim; ++d) {
            cos_dst[d] = cos_src[d];
            sin_dst[d] = sin_src[d];
        }
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: build_attn_mask_kernel（control-flow 类，不向量化）
//
// 输出 attn_mask 形状: (S_mask, S_mask)，其中 S_mask >= S_max。
//
// 有效区（row < S_max && col < S_max）：按 causal + cross_item + padding 行规则填写。
// padding 区（row >= S_max 或 col >= S_max）：填 -inf，供 FA2 copy_g2s_mask 无越界读取。
// ─────────────────────────────────────────────────────────────────────────────
__global__ void build_attn_mask_kernel(
    __nv_bfloat16* __restrict__ attn_mask,
    int64_t s_len,
    int64_t c_len,
    int64_t i_len,
    int64_t S_max,
    int64_t S_mask   // 输出边长（>= S_max），padding 区填 -inf
) {
    const int64_t col = (int64_t)blockIdx.x * MASK_BLOCK + threadIdx.x;
    const int64_t row = (int64_t)blockIdx.y * MASK_BLOCK + threadIdx.y;

    if (row >= S_mask || col >= S_mask) return;

    // padding 区：row >= S_max 或 col >= S_mask 范围，全部填 -inf
    if (row >= S_max || col >= S_max) {
        attn_mask[row * S_mask + col] = BF16_NEG_INF;
        return;
    }

    const int64_t valid_tokens = s_len + c_len + i_len;
    const int64_t item_pos     = s_len + c_len;

    // row_valid: padding 行（row >= valid_tokens）完全屏蔽。
    // 全零 Q 向量若不屏蔽，softmax(Q*K^T / sqrt(d)) 会产生非零 weight，
    // 累加到 O 后经 LayerNorm / FFN 放大为 NaN。
    const bool row_valid  = (row < valid_tokens);
    const bool causal_ok  = (col <= row);
    const bool is_item_r  = (row >= item_pos) && (row < valid_tokens);
    const bool is_item_c  = (col >= item_pos) && (col < valid_tokens);
    const bool cross_item = is_item_r && is_item_c && (row != col);

    const bool visible = row_valid && causal_ok && !cross_item;
    attn_mask[row * S_mask + col] = visible ? BF16_ZERO : BF16_NEG_INF;
}


// ─────────────────────────────────────────────────────────────────────────────
// 主入口：pack_and_prepare_b1_cuda
// ─────────────────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> pack_and_prepare_b1_cuda(
    const torch::Tensor& h_s,
    const torch::Tensor& h_c,
    const torch::Tensor& h_i,
    int64_t s_len,   // 直接传标量，Python 侧已 .item() 提取，消除 D2H 同步点
    int64_t c_len,
    int64_t i_len,
    const torch::Tensor& static_cos,
    const torch::Tensor& static_sin,
    int64_t S_max,
    int64_t S_mask   // attn_mask 输出边长，>= S_max（FA2: S_max_rounded，SDPA: S_max）
) {
    TORCH_CHECK(h_s.device().is_cuda(),        "h_s must be on CUDA");
    TORCH_CHECK(h_c.device().is_cuda(),        "h_c must be on CUDA");
    TORCH_CHECK(h_i.device().is_cuda(),        "h_i must be on CUDA");
    TORCH_CHECK(static_cos.device().is_cuda(), "static_cos must be on CUDA");
    TORCH_CHECK(static_sin.device().is_cuda(), "static_sin must be on CUDA");

    TORCH_CHECK(h_s.scalar_type() == torch::kBFloat16, "h_s must be bf16");
    TORCH_CHECK(h_c.scalar_type() == torch::kBFloat16, "h_c must be bf16");
    TORCH_CHECK(h_i.scalar_type() == torch::kBFloat16, "h_i must be bf16");
    TORCH_CHECK(static_cos.scalar_type() == torch::kBFloat16, "static_cos must be bf16");
    TORCH_CHECK(static_sin.scalar_type() == torch::kBFloat16, "static_sin must be bf16");

    TORCH_CHECK(h_s.dim() == 2,        "h_s must be 2D");
    TORCH_CHECK(h_c.dim() == 2,        "h_c must be 2D");
    TORCH_CHECK(h_i.dim() == 2,        "h_i must be 2D");
    TORCH_CHECK(static_cos.dim() == 4, "static_cos must be 4D");
    TORCH_CHECK(static_sin.dim() == 4, "static_sin must be 4D");

    TORCH_CHECK(s_len >= 0, "s_len must be non-negative");
    TORCH_CHECK(c_len >= 0, "c_len must be non-negative");
    TORCH_CHECK(i_len >= 0, "i_len must be non-negative");

    const int64_t D        = h_s.size(1);
    const int64_t head_dim = static_cos.size(3);

    TORCH_CHECK(h_c.size(1) == D, "h_c and h_s must have same D");
    TORCH_CHECK(h_i.size(1) == D, "h_i and h_s must have same D");
    TORCH_CHECK(S_max > 0,        "S_max must be positive");
    TORCH_CHECK(static_cos.size(1) == S_max, "static_cos S_max mismatch");
    TORCH_CHECK(static_sin.size(1) == S_max, "static_sin S_max mismatch");
    TORCH_CHECK(S_mask >= S_max, "S_mask must be >= S_max");

    // 连续化（保证行首地址对齐）
    auto h_s_c        = h_s.contiguous();
    auto h_c_c        = h_c.contiguous();
    auto h_i_c        = h_i.contiguous();
    auto static_cos_c = static_cos.contiguous();
    auto static_sin_c = static_sin.contiguous();

    const int64_t valid_tokens = s_len + c_len + i_len;

    TORCH_CHECK(valid_tokens <= S_max,   "valid_tokens > S_max");
    TORCH_CHECK(h_s_c.size(0) == s_len, "h_s rows mismatch s_len");
    TORCH_CHECK(h_c_c.size(0) == c_len, "h_c rows mismatch c_len");
    TORCH_CHECK(h_i_c.size(0) == i_len, "h_i rows mismatch i_len");

    // 分配输出
    auto opts_bf16 = h_s.options();
    auto opts_bool = h_s.options().dtype(torch::kBool);
    auto opts_i64  = h_s.options().dtype(torch::kInt64);

    auto h_dense   = torch::zeros({1, S_max, D}, opts_bf16);
    auto cos_out   = torch::empty({1, S_max, 1, head_dim}, opts_bf16);
    auto sin_out   = torch::empty({1, S_max, 1, head_dim}, opts_bf16);
    auto attn_mask = torch::empty({1, 1, S_mask, S_mask}, opts_bf16);
    auto item_mask = torch::empty({valid_tokens}, opts_bool);
    auto offsets   = torch::tensor({(int64_t)0, valid_tokens}, opts_i64);

    auto stream = at::cuda::getCurrentCUDAStream();

    // ── Kernel 1：向量化 pack tokens ─────────────────────────────────────────
    // 1D grid：每个 thread 负责 S_max 中的一个 pos（一整行 D 个 bf16）
    {
        const int grid1 = (int)((S_max + PACK_BLOCK_T - 1) / PACK_BLOCK_T);
        pack_tokens_kernel<<<grid1, PACK_BLOCK_T, 0, stream>>>(
            AS_CONST_BF16_PTR(h_s_c.data_ptr<at::BFloat16>()),
            AS_CONST_BF16_PTR(h_c_c.data_ptr<at::BFloat16>()),
            AS_CONST_BF16_PTR(h_i_c.data_ptr<at::BFloat16>()),
            AS_BF16_PTR(h_dense.data_ptr<at::BFloat16>()),
            item_mask.data_ptr<bool>(),
            s_len, c_len, i_len, S_max, D
        );
    }

    // ── Kernel 2：向量化 gather RoPE ─────────────────────────────────────────
    // 1D grid：每个 thread 负责 S_max 中的一个 seq_pos（整行 head_dim 个 bf16）
    {
        const int grid2 = (int)((S_max + ROPE_BLOCK_T - 1) / ROPE_BLOCK_T);
        gather_rope_kernel<<<grid2, ROPE_BLOCK_T, 0, stream>>>(
            AS_CONST_BF16_PTR(static_cos_c.data_ptr<at::BFloat16>()),
            AS_CONST_BF16_PTR(static_sin_c.data_ptr<at::BFloat16>()),
            AS_BF16_PTR(cos_out.data_ptr<at::BFloat16>()),
            AS_BF16_PTR(sin_out.data_ptr<at::BFloat16>()),
            s_len, c_len, S_max, head_dim
        );
    }

    // ── Kernel 3：build attn mask（2D，control-flow 类）────────────────────────
    // Grid 覆盖 S_mask × S_mask（包含 padding 区），padding 区填 -inf，
    // 保证 FA2 copy_g2s_mask 整 tile 搬运无越界。
    {
        dim3 block3(MASK_BLOCK, MASK_BLOCK);
        dim3 grid3(
            (int)((S_mask + MASK_BLOCK - 1) / MASK_BLOCK),
            (int)((S_mask + MASK_BLOCK - 1) / MASK_BLOCK)
        );
        build_attn_mask_kernel<<<grid3, block3, 0, stream>>>(
            AS_BF16_PTR(attn_mask.data_ptr<at::BFloat16>()),
            s_len, c_len, i_len, S_max, S_mask
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {h_dense, cos_out, sin_out, attn_mask, item_mask, offsets};
}


// ─────────────────────────────────────────────────────────────────────────────
// CPU fallback 实现
// ─────────────────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> pack_and_prepare_b1_cpu(
    const torch::Tensor& h_s,
    const torch::Tensor& h_c,
    const torch::Tensor& h_i,
    int64_t s_len,
    int64_t c_len,
    int64_t i_len,
    const torch::Tensor& static_cos,
    const torch::Tensor& static_sin,
    int64_t S_max,
    int64_t S_mask   // attn_mask 输出边长，>= S_max
) {
    TORCH_CHECK(S_mask >= S_max, "S_mask must be >= S_max on CPU");

    const int64_t valid_tokens = s_len + c_len + i_len;
    const int64_t item_pos     = s_len + c_len;
    const int64_t D            = h_s.size(1);

    TORCH_CHECK(valid_tokens <= S_max, "valid_tokens > S_max on CPU");

    auto h_dense = torch::zeros({1, S_max, D}, h_s.options());
    if (s_len > 0) h_dense[0].narrow(0, 0,        s_len).copy_(h_s);
    if (c_len > 0) h_dense[0].narrow(0, s_len,    c_len).copy_(h_c);
    if (i_len > 0) h_dense[0].narrow(0, item_pos, i_len).copy_(h_i);

    auto item_mask = torch::zeros({valid_tokens}, torch::kBool);
    if (i_len > 0) item_mask.narrow(0, item_pos, i_len).fill_(true);

    auto offsets = torch::tensor({(int64_t)0, valid_tokens}, torch::kInt64);

    auto pos_ids = torch::arange(S_max, torch::kInt64).clamp_max(item_pos);
    using torch::indexing::Slice;
    auto cos_out = static_cos.index({Slice(), pos_ids, Slice(), Slice()});
    auto sin_out = static_sin.index({Slice(), pos_ids, Slice(), Slice()});

    // 构建 S_mask × S_mask 的 attn_mask
    // 有效区（S_max × S_max）：按 causal + cross_item + padding 行规则填写
    // padding 区（S_max:S_mask）：全部 -inf
    auto row_idx = torch::arange(S_max, torch::kInt64);
    auto col_idx = torch::arange(S_max, torch::kInt64);
    auto row_2d  = row_idx.unsqueeze(1).expand({S_max, S_max});
    auto col_2d  = col_idx.unsqueeze(0).expand({S_max, S_max});

    // row_valid: padding 行完全屏蔽（与 CUDA kernel 保持完全一致）
    auto row_valid  = (row_2d < valid_tokens);
    auto causal_ok  = (col_2d <= row_2d);
    auto is_item_r  = (row_2d >= item_pos) & (row_2d < valid_tokens);
    auto is_item_c  = (col_2d >= item_pos) & (col_2d < valid_tokens);
    auto cross_item = is_item_r & is_item_c & (row_2d != col_2d);
    auto visible    = row_valid & causal_ok & ~cross_item;

    auto valid_mask = torch::where(
        visible,
        torch::zeros({S_max, S_max}, h_s.options()),
        torch::full({S_max, S_max},
                    -std::numeric_limits<float>::infinity(),
                    h_s.options())
    );   // (S_max, S_max)

    // 若 S_mask > S_max，pad 到 (S_mask, S_mask)，padding 区填 -inf
    torch::Tensor attn_mask;
    if (S_mask > S_max) {
        int64_t pad = S_mask - S_max;
        attn_mask = torch::nn::functional::pad(
            valid_mask,
            torch::nn::functional::PadFuncOptions({0, pad, 0, pad})
                .value(-std::numeric_limits<float>::infinity())
        );
    } else {
        attn_mask = valid_mask;
    }

    return {h_dense, cos_out, sin_out, attn_mask.unsqueeze(0).unsqueeze(0), item_mask, offsets};
}
