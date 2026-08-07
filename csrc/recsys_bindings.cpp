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


// ─────────────────────────────────────────────────────────────────────────────
// Step 5: 注册 schema
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        mha_fwd_with_mask,
        "Tensor q, Tensor k, Tensor v, Tensor mask",
        "-> Tensor"
    )
CUSTOM_OPS_LIBRARY_END


// ─────────────────────────────────────────────────────────────────────────────
// Step 6: 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(mha_fwd_with_mask)
CUSTOM_OPS_IMPL_END


// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps — recsys_ops CUDA operator library")
