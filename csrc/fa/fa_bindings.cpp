/**
 * fa_bindings.cpp — FA 分组（mha_fwd_with_mask）算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.mha_fwd_with_mask(...)
 * 本分组独立编译为 recsys_fa_kernel.so（懒加载：首次访问该算子时才 JIT）。
 * 多个分组 .so 共享同一 namespace，各自定义不相交的算子（PyTorch 允许多
 * library 贡献同一 namespace；重复 def 同名算子才会报错）。
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
// Step 5/6: 注册 schema + 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        mha_fwd_with_mask,
        "Tensor q, Tensor k, Tensor v, Tensor mask",
        "-> Tensor"
    )
CUSTOM_OPS_LIBRARY_END

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(mha_fwd_with_mask)
CUSTOM_OPS_IMPL_END

// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps/FA — mha_fwd_with_mask (FlashAttention-2 + additive mask)")
