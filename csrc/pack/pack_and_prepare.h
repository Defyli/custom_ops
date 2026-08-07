#pragma once

#include <torch/extension.h>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
// pack_and_prepare_b1  — 融合算子声明
//
// 将 pack_jagged_tokens(mode=4) + RoPE gather + attn_mask 构建三个步骤
// 合并为单次算子调用，消除 Python 层 CPU-GPU 同步点和碎片化 kernel launch。
// ──────────────────────────────────────────────────────────────────────────────

// ── 公共宏 ──────────────────────────────────────────────────────────────────

// Kernel 1 (pack tokens)：1D block，每线程负责输出 tensor 中一行（pos 维）
// 使用 float4 向量化读写，每次处理 8 个 bf16
#define PACK_BLOCK_T   128

// Kernel 2 (gather RoPE)：1D block，每线程负责一个 seq_pos 的整行 head_dim
#define ROPE_BLOCK_T   128

// Kernel 3 (attn mask)：2D tile，control-flow 类，不向量化
#define MASK_BLOCK     32

// bf16 特殊值
#define BF16_NEG_INF   (__float2bfloat16(-INFINITY))
#define BF16_ZERO      (__float2bfloat16(0.0f))

// ── 函数声明 ─────────────────────────────────────────────────────────────────

// CUDA 实现（pack_and_prepare.cu）
// 注意：s_len / c_len / i_len 改为 int64_t 直接传入（Python 侧先 .item() 提取），
// 彻底消除 Kernel 内部的 D2H 同步点，避免跨 stream 竞争导致 NaN。
std::vector<torch::Tensor> pack_and_prepare_b1_cuda(
    const torch::Tensor& h_s,        // (total_s_tokens, D)
    const torch::Tensor& h_c,        // (total_c_tokens, D)
    const torch::Tensor& h_i,        // (item_num, D)
    int64_t s_len,                   // ubc token 数（标量，Python 侧 .item() 后传入）
    int64_t c_len,                   // ctx token 数
    int64_t i_len,                   // item token 数
    const torch::Tensor& static_cos, // (1, S_max, 1, head_dim)
    const torch::Tensor& static_sin, // (1, S_max, 1, head_dim)
    int64_t S_max,
    int64_t S_mask                   // attn_mask 输出边长，>= S_max（FA2 路径传 S_max_rounded，SDPA 传 S_max）
                                     // padding 区域（S_max:S_mask）填 -inf，保证 copy_g2s_mask 无越界
);

// CPU fallback 实现（用于单测，无 CUDA）
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
    int64_t S_mask                   // 同 CUDA 版本
);
