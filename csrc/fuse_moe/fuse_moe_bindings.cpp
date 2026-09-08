/**
 * fuse_moe_bindings.cpp — fuse_moe 分组算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.fuse_moe(...)
 * 本分组独立编译为 recsys_fuse_moe_kernel.so（懒加载：首次访问该算子时才
 * JIT——只测 fuse_moe 时不再编译 FA/mixed_gemm）。
 */

// Step 1: 定义命名空间
#define CUSTOM_OPS_NAMESPACE recsys_ops

// Step 2: 包含通用框架宏
#include "custom_ops_macros.h"

// Step 3: 包含算子声明头文件
#include "fuse_moe/fuse_moe_op.h"

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: 分发函数
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_DISPATCH_FN(
    fuse_moe,
    (const torch::Tensor& x,
     const torch::Tensor& gate_up_weight,
     const torch::Tensor& down_weight,
     const torch::Tensor& topk_ids,
     const torch::Tensor& topk_scale),
    (x, gate_up_weight, down_weight, topk_ids, topk_scale),
    x
)

// ─────────────────────────────────────────────────────────────────────────────
// Step 5/6: 注册 schema + 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        fuse_moe,
        "Tensor x, Tensor gate_up_weight, Tensor down_weight, "
        "Tensor topk_ids, Tensor topk_scale",
        "-> Tensor"
    )
CUSTOM_OPS_LIBRARY_END

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(fuse_moe)
CUSTOM_OPS_IMPL_END

// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps/fuse_moe — fused MoE FFN (group GEMM + silu*mul + reduce)")
