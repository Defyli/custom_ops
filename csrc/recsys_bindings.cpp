/**
 * csrc/recsys_bindings.cpp — RecsysOps 算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.<name>(...)
 *
 * 注册算子:
 *   1. mha_fwd_with_mask      — Flash Attention 2 前向，支持任意 bf16 加法 mask
 *   2. pack_and_prepare_b1    — 融合 pack tokens + RoPE gather + attn_mask 构建
 *   3. jagged_pool_and_collect — 融合 jagged pooling（v3：零 H→D copy）
 */

// Step 1: 定义命名空间
#define CUSTOM_OPS_NAMESPACE recsys_ops

// Step 2: 包含通用框架宏
#include "custom_ops_macros.h"

// Step 3: 包含算子声明头文件
#include "fa/fa_fwd_op.h"
#include "pack/pack_and_prepare.h"
#include "jagged/jagged_forward.h"


// ─────────────────────────────────────────────────────────────────────────────
// Step 4: 分发函数
// ─────────────────────────────────────────────────────────────────────────────

// ── mha_fwd_with_mask ────────────────────────────────────────────────────────
CUSTOM_OP_DISPATCH_FN(
    mha_fwd_with_mask,
    (const torch::Tensor& q,
     const torch::Tensor& k,
     const torch::Tensor& v,
     const torch::Tensor& mask),
    (q, k, v, mask),
    q
)

// ── pack_and_prepare_b1 ───────────────────────────────────────────────────────
// s_len/c_len/i_len 以 int64_t 直接传入（Python 侧先 .item() 提取，避免 D2H 同步）
CUSTOM_OP_DISPATCH_FN(
    pack_and_prepare_b1,
    (const torch::Tensor& h_s,
     const torch::Tensor& h_c,
     const torch::Tensor& h_i,
     int64_t s_len,
     int64_t c_len,
     int64_t i_len,
     const torch::Tensor& static_cos,
     const torch::Tensor& static_sin,
     int64_t S_max,
     int64_t S_mask),
    (h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_mask),
    h_s
)

// ── jagged_pool_and_collect ───────────────────────────────────────────────────
// pool_val_splits / pool_len_splits 以 int[] 传递（映射到 std::vector<int64_t>）
CUSTOM_OP_DISPATCH_FN(
    jagged_pool_and_collect,
    (const torch::Tensor&        pool_values,
     const torch::Tensor&        pool_lengths,
     const std::vector<int64_t>& pool_val_splits,
     const std::vector<int64_t>& pool_len_splits,
     int64_t                     n_pooling,
     int64_t                     reduce_mode,
     const torch::Tensor&        ones_cache),
    (pool_values, pool_lengths, pool_val_splits, pool_len_splits,
     n_pooling, reduce_mode, ones_cache),
    pool_values
)


// ─────────────────────────────────────────────────────────────────────────────
// Step 5: 注册 schema
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        mha_fwd_with_mask,
        "Tensor q, Tensor k, Tensor v, Tensor mask",
        "-> Tensor"
    )
    CUSTOM_OP_SCHEMA(
        pack_and_prepare_b1,
        "Tensor h_s,"
        "Tensor h_c,"
        "Tensor h_i,"
        "int s_len,"
        "int c_len,"
        "int i_len,"
        "Tensor static_cos,"
        "Tensor static_sin,"
        "int S_max,"
        "int S_mask",
        "-> Tensor[]"
    )
    CUSTOM_OP_SCHEMA(
        jagged_pool_and_collect,
        "Tensor pool_values,"
        "Tensor pool_lengths,"
        "int[] pool_val_splits,"
        "int[] pool_len_splits,"
        "int n_pooling,"
        "int reduce_mode,"
        "Tensor ones_cache",
        "-> Tensor[]"
    )
CUSTOM_OPS_LIBRARY_END


// ─────────────────────────────────────────────────────────────────────────────
// Step 6: 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(mha_fwd_with_mask)
    CUSTOM_OP_BIND(pack_and_prepare_b1)
    CUSTOM_OP_BIND(jagged_pool_and_collect)
CUSTOM_OPS_IMPL_END


// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps — recsys_ops CUDA operator library")
