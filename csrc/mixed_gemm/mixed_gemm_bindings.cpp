/**
 * mixed_gemm_bindings.cpp — mixed_gemm 分组算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.{mixed_gemm, mixed_gemm_fp8_available}
 * 本分组独立编译为 recsys_mixed_gemm_kernel.so（懒加载）。
 */

// Step 1: 定义命名空间
#define CUSTOM_OPS_NAMESPACE recsys_ops

// Step 2: 包含通用框架宏
#include "custom_ops_macros.h"

// Step 3: 包含算子声明头文件
#include "mixed_gemm/mixed_gemm_op.h"
#include "mixed_gemm/gemm_bf16xfp32_sm80.h"

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: 分发函数
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_DISPATCH_FN(
    mixed_gemm,
    (const torch::Tensor& x,
     const torch::Tensor& w_high,
     const torch::Tensor& w_low,
     const c10::optional<torch::Tensor>& w_scale,
     double scale,
     const c10::optional<torch::Tensor>& bias,
     int64_t activation,
     bool fp32_output,
     int64_t force_splitk),
    (x, w_high, w_low, w_scale, scale, bias, activation, fp32_output, force_splitk),
    x
)

// ── mixed_gemm_fp8_available ─────────────────────────────────────────────────
// 无 tensor 参数的纯查询函数：不适用 CUSTOM_OP_DISPATCH_FN（无设备分派语义），
// 手写 dispatch 体。
static bool _dispatch_mixed_gemm_fp8_available() {
    if (!mixed_gemm::mixed_gemm_fp8_compiled()) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess) {
        return false;
    }
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) {
        return false;
    }
    // FP8 e4m3 mma 需要 SM89+（Ada）；INT8 后端在 SM80+ 均可用。
    return major > 8 || (major == 8 && minor >= 9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5/6: 注册 schema + 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        mixed_gemm,
        "Tensor x, Tensor w_high, Tensor w_low, "
        "Tensor? w_scale=None, float scale=0.00390625, "
        "Tensor? bias=None, int activation=0, "
        "bool fp32_output=True, int force_splitk=0",
        "-> Tensor"
    )
    CUSTOM_OP_SCHEMA(
        mixed_gemm_fp8_available,
        "",
        "-> bool"
    )
CUSTOM_OPS_LIBRARY_END

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(mixed_gemm)
    CUSTOM_OP_BIND(mixed_gemm_fp8_available)
CUSTOM_OPS_IMPL_END

// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps/mixed_gemm — bf16+fp8/int8 residual GEMM with fused epilogue")
