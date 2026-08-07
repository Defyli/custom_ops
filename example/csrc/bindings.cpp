/**
 * example/csrc/bindings.cpp — 使用 custom_ops 框架注册 mha_fwd_with_mask 的示例
 *
 * 展示如何用 custom_ops_macros.h 中的宏以最少代码注册一个 CUDA 算子。
 *
 * 步骤：
 *   1. 定义命名空间（必须在包含 custom_ops_macros.h 之前）
 *   2. 包含框架宏头文件
 *   3. 包含算子声明头文件
 *   4. 用 CUSTOM_OP_DISPATCH_FN 声明分发函数（一行）
 *   5. 用 CUSTOM_OPS_LIBRARY_BEGIN/END 注册 schema
 *   6. 用 CUSTOM_OPS_IMPL_BEGIN/END 绑定实现
 *   7. 用 CUSTOM_OPS_PYBIND_MODULE 声明 pybind 入口
 */

// Step 1: 定义命名空间（决定 torch.ops.<namespace> 的名称）
#define CUSTOM_OPS_NAMESPACE fa_ops

// Step 2: 包含通用框架宏
#include "custom_ops_macros.h"

// Step 3: 包含算子声明头文件
#include "fa/fa_fwd_op.h"
#include "pack/pack_and_prepare.h"
#include "jagged/jagged_forward.h"


// ─────────────────────────────────────────────────────────────────────────────
// Step 4: 分发函数 — 每个算子一行
// CUSTOM_OP_DISPATCH_FN(name, (params...), (arg_names...), device_tensor)
// ─────────────────────────────────────────────────────────────────────────────

// ── mha_fwd_with_mask ────────────────────────────────────────────────────────
CUSTOM_OP_DISPATCH_FN(
    mha_fwd_with_mask,
    (const torch::Tensor& q,
     const torch::Tensor& k,
     const torch::Tensor& v,
     const torch::Tensor& mask),
    (q, k, v, mask),
    q   // 用 q 判断设备（CUDA / CPU）
)

// ── pack_and_prepare_b1 ───────────────────────────────────────────────────────
// 融合算子：pack tokens + gather RoPE cos/sin + build attn_mask
// s_len/c_len/i_len 已改为 int64_t 直接传入（Python 侧先 .item() 提取，避免 D2H 同步）
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
    h_s   // 用 h_s 判断设备（CUDA / CPU）
)

// ── jagged_pool_and_collect ───────────────────────────────────────────────────
// 融合 pooling 算子：只处理 pooling 特征的批量 segment reduce（v3：零 H→D copy）
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
    pool_values   // 用 pool_values 判断设备（CUDA / CPU）
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
// Step 7: pybind 入口（torch.utils.cpp_extension.load 要求）
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("FA ops — Flash Attention with mask + pack_and_prepare_b1 + jagged_pool_and_collect, via custom_ops framework")
