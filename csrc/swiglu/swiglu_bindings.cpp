/**
 * swiglu_bindings.cpp — swiglu 分组算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.swiglu(...)
 * 本分组独立编译为 recsys_swiglu_kernel.so（懒加载：首次访问该算子时才
 * JIT——只测 swiglu 时不再编译 FA/mixed_gemm/fuse_moe）。
 */

#define CUSTOM_OPS_NAMESPACE recsys_ops

#include "custom_ops_macros.h"

#include "swiglu/swiglu_op.h"

// ─────────────────────────────────────────────────────────────────────────────
// 分发函数
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_DISPATCH_FN(
    swiglu,
    (const torch::Tensor& x, const torch::Tensor& weight),
    (x, weight),
    x
)

// ─────────────────────────────────────────────────────────────────────────────
// 注册 schema + 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        swiglu,
        "Tensor x, Tensor weight",
        "-> Tensor"
    )
CUSTOM_OPS_LIBRARY_END

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(swiglu)
CUSTOM_OPS_IMPL_END

// ─────────────────────────────────────────────────────────────────────────────
// pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps/swiglu — SwiGLU fused GEMM (gate/up pairing + silu*mul epilogue)")
