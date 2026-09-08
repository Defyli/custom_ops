/**
 * custom_ops_macros.h — 通用 CUDA 算子注册框架宏
 *
 * 使用方式
 * --------
 * 在业务 bindings.cpp 中：
 *
 *   // 1. 定义命名空间（必须在包含本文件之前）
 *   #define CUSTOM_OPS_NAMESPACE my_ops
 *
 *   // 2. 包含本文件
 *   #include "custom_ops_macros.h"
 *
 *   // 3. 包含算子头文件
 *   #include "fa/fa_fwd_op.h"
 *
 *   // 4. 声明分发函数
 *   CUSTOM_OP_DISPATCH_FN(mha_fwd_with_mask,
 *       (const torch::Tensor& q, const torch::Tensor& k,
 *        const torch::Tensor& v, const torch::Tensor& mask),
 *       (q, k, v, mask), q)
 *
 *   // 5. 注册 schema 与实现
 *   CUSTOM_OPS_LIBRARY_BEGIN
 *       CUSTOM_OP_SCHEMA(mha_fwd_with_mask,
 *           "Tensor q, Tensor k, Tensor v, Tensor mask", "-> Tensor")
 *   CUSTOM_OPS_LIBRARY_END
 *
 *   CUSTOM_OPS_IMPL_BEGIN
 *       CUSTOM_OP_BIND(mha_fwd_with_mask)
 *   CUSTOM_OPS_IMPL_END
 */

#pragma once

#include <torch/extension.h>


// ─────────────────────────────────────────────────────────────────────────────
// 命名空间配置
// 在包含本文件前 #define CUSTOM_OPS_NAMESPACE <your_namespace>，
// 否则使用默认值 custom_ops。
// ─────────────────────────────────────────────────────────────────────────────

#ifndef CUSTOM_OPS_NAMESPACE
#define CUSTOM_OPS_NAMESPACE custom_ops
#endif

// 将宏值转为字符串字面量（两层宏展开保证正确展开）
#define _CUSTOM_OPS_STR2(x) #x
#define _CUSTOM_OPS_STR(x)  _CUSTOM_OPS_STR2(x)
#define CUSTOM_OPS_NAMESPACE_STR  _CUSTOM_OPS_STR(CUSTOM_OPS_NAMESPACE)


// ─────────────────────────────────────────────────────────────────────────────
// CUSTOM_OP_DEVICE_DISPATCH(name, first_tensor, ...)
//
// 根据 first_tensor 所在设备路由到 name_cuda(...) 或 name_cpu(...)。
// 在分发函数体内使用（生成 return 语句）。
//
// 示例：
//   CUSTOM_OP_DEVICE_DISPATCH(my_op, x, x, n)
//   → return x.device().is_cuda() ? my_op_cuda(x, n) : my_op_cpu(x, n)
// ─────────────────────────────────────────────────────────────────────────────
#define CUSTOM_OP_DEVICE_DISPATCH(name, first_tensor, ...)      \
    return (first_tensor).device().is_cuda()                    \
           ? name##_cuda(__VA_ARGS__)                           \
           : name##_cpu(__VA_ARGS__)


// ─────────────────────────────────────────────────────────────────────────────
// CUSTOM_OP_DISPATCH_FN(name, params, args, device_tensor)
//
// 一行声明完整的分发函数（含函数签名 + 函数体）。
//
// 参数：
//   name          — 算子名（不含 _cuda/_cpu）
//   params        — 带括号的完整参数列表，如 (const torch::Tensor& x, int64_t n)
//   args          — 不带类型的参数名列表（用于转发），如 (x, n)
//   device_tensor — 用于判断设备的 Tensor 变量名，如 x
//
// 示例：
//   CUSTOM_OP_DISPATCH_FN(my_op,
//       (const torch::Tensor& x, int64_t n),
//       (x, n), x)
// ─────────────────────────────────────────────────────────────────────────────
#define CUSTOM_OP_DISPATCH_FN(name, params, args, device_tensor)    \
    static auto _dispatch_##name params {                           \
        CUSTOM_OP_DEVICE_DISPATCH(name, device_tensor,              \
            _CUSTOM_OP_UNPACK args);                                \
    }

// 辅助宏：去掉括号展开参数包
#define _CUSTOM_OP_UNPACK(...) __VA_ARGS__


// ─────────────────────────────────────────────────────────────────────────────
// CUSTOM_OP_SCHEMA(name, args_str, ret_str)
//
// 在 TORCH_LIBRARY 块内声明算子 schema。
// args_str — 参数列表字符串（不含括号），如 "Tensor x, int n"
// ret_str  — 返回值字符串（含 ->），如 "-> Tensor"
// ─────────────────────────────────────────────────────────────────────────────
#define CUSTOM_OP_SCHEMA(name, args_str, ret_str)       \
    m.def(#name "(" args_str ") " ret_str);


// ─────────────────────────────────────────────────────────────────────────────
// CUSTOM_OP_BIND(name)
//
// 在 TORCH_LIBRARY_IMPL 块内绑定 _dispatch_<name> 到已注册算子。
// ─────────────────────────────────────────────────────────────────────────────
#define CUSTOM_OP_BIND(name)    \
    m.impl(#name, &_dispatch_##name);


// ─────────────────────────────────────────────────────────────────────────────
// CUSTOM_OPS_LIBRARY_BEGIN / END
// CUSTOM_OPS_IMPL_BEGIN / END
//
// 展开为 TORCH_LIBRARY / TORCH_LIBRARY_IMPL 块（使用参数化命名空间）。
// 在 BEGIN 和 END 之间填写 CUSTOM_OP_SCHEMA / CUSTOM_OP_BIND 调用。
// ─────────────────────────────────────────────────────────────────────────────
// 两层宏展开技巧：先把 CUSTOM_OPS_NAMESPACE 展开为实际值，
// 再传给 _CUSTOM_OPS_LIBRARY_BEGIN_IMPL，确保 TORCH_LIBRARY 拿到展开后的命名空间名。
//
// TORCH_LIBRARY_FRAGMENT（非 TORCH_LIBRARY）：允许多个 .so（分组懒加载，
// 每组一个独立扩展）向同一 namespace 贡献算子定义——PyTorch 限制单个
// TORCH_LIBRARY 每 namespace 只能注册一次。各分组 def 的算子集合不相交；
// 单 .so 场景（旧用法）下 FRAGMENT 行为与 TORCH_LIBRARY 完全一致。
#define _CUSTOM_OPS_LIBRARY_BEGIN_IMPL(ns)   TORCH_LIBRARY_FRAGMENT(ns, m) {
#define CUSTOM_OPS_LIBRARY_BEGIN             _CUSTOM_OPS_LIBRARY_BEGIN_IMPL(CUSTOM_OPS_NAMESPACE)
#define CUSTOM_OPS_LIBRARY_END               }

#define _CUSTOM_OPS_IMPL_BEGIN_IMPL(ns)      TORCH_LIBRARY_IMPL(ns, CompositeExplicitAutograd, m) {
#define CUSTOM_OPS_IMPL_BEGIN                _CUSTOM_OPS_IMPL_BEGIN_IMPL(CUSTOM_OPS_NAMESPACE)
#define CUSTOM_OPS_IMPL_END                  }


// ─────────────────────────────────────────────────────────────────────────────
// CUSTOM_OPS_PYBIND_MODULE
//
// 展开 PYBIND11_MODULE 空入口（torch.utils.cpp_extension.load 要求有此符号）。
// ─────────────────────────────────────────────────────────────────────────────
#define CUSTOM_OPS_PYBIND_MODULE(doc_str)               \
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {          \
        m.doc() = doc_str;                              \
    }
