/**
 * csrc/recsys_bindings.cpp — RecsysOps 算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.<name>(...)
 *
 * 注册算子:
 *   1. mha_fwd_with_mask      — Flash Attention 2 前向，支持任意 bf16 加法 mask
 *   2. mixed_gemm             — 混合精度 GEMM：bf16 主项 + fp8/int8 residual 精度补偿，
 *                               epilogue 融合 bias + silu/gelu 激活
 */

// Step 1: 定义命名空间
#define CUSTOM_OPS_NAMESPACE recsys_ops

// Step 2: 包含通用框架宏
#include "custom_ops_macros.h"

// Step 3: 包含算子声明头文件
#include "fa/fa_fwd_op.h"
#include "mixed_gemm/mixed_gemm_op.h"
#include "mixed_gemm/gemm_bf16xfp32_sm80.h"

#include <cuda_runtime.h>

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

// ── mixed_gemm ──────────────────────────────────────────────────────────────
// 注意：c10::optional<Tensor> 对应 schema 中的 "Tensor? ..."；
// dispatch 宏按参数名原样转发。
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
// Step 5: 注册 schema
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        mha_fwd_with_mask,
        "Tensor q, Tensor k, Tensor v, Tensor mask",
        "-> Tensor"
    )
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


// ─────────────────────────────────────────────────────────────────────────────
// Step 6: 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(mha_fwd_with_mask)
    CUSTOM_OP_BIND(mixed_gemm)
    CUSTOM_OP_BIND(mixed_gemm_fp8_available)
CUSTOM_OPS_IMPL_END


// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps — recsys_ops CUDA operator library")
