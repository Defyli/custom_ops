// Mixed-precision GEMM — sm120a (Blackwell consumer, RTX 5090/5090D) TMA 路径
// 的对外接口。实现见 gemm_bf16xfp32_sm120.cu；数值算法与 sm80 路径一致，
// 仅数据通路换成 TMA + mbarrier（见该文件的文件头注释）。

#ifndef MIXED_GEMM_SRC_GEMM_BF16XFP32_SM120_H_
#define MIXED_GEMM_SRC_GEMM_BF16XFP32_SM120_H_

#include "gemm_bf16xfp32_sm80.h"

#include <cuda_runtime_api.h>
#include <cstdint>

namespace mixed_gemm {

// sm120 kernel 是否编入本编译单元（编译目标含 >= sm120a 的 arch；
// 与 FA 的 fa_mask_sm120_supported 同一判定方式，编译期/运行期双重检查）。
bool mixed_gemm_sm120_compiled() noexcept;

// 设备是否可走 sm120 路径：major == 12 且 kernel 已编入。
// GEMM_MIXED_FORCE_SM80=1 时恒为 false（A/B 调试用）。
bool mixed_gemm_sm120_supported() noexcept;

// sm120 路径的额外数据面约束（不满足则调用方回退 sm80 路径）：
//   - k % 16 == 0：TMA 要求所有全局 stride 为 16B 倍数；1 字节 residual
//     张量的行 stride 即 k 字节（sm80 路径只需 k%8==0，此为更强约束）
//   - x / w_high / w_low / x_residual 指针均 16B 对齐（TMA 全局基址要求）
// 输出 y 的对齐不在此检查（不满足时仅 epilogue 降级 elementwise）。
bool mixed_gemm_sm120_ptrs_ok(const void *x_ptr, const void *w_high_ptr,
                              const void *w_low_ptr, const void *x_res_ptr,
                              int64_t k) noexcept;

// 与 resolve_gemm_bf16xfp32_epilogue_launcher / _int8_launcher（sm80）签名
// 完全一致的 sm120 版本，由 mixed_gemm_op.cu 按 supported()+ptrs_ok() 选择。
#if MIXED_GEMM_FP8_ENABLED
GemmFixedEpilogueLauncher resolve_gemm_bf16xfp32_epilogue_launcher_sm120(
    int activation_type, bool has_bias) noexcept;
#endif
GemmFixedInt8EpilogueLauncher resolve_gemm_bf16xfp32_int8_launcher_sm120(
    int activation_type, bool has_bias) noexcept;

}  // namespace mixed_gemm

#endif  // MIXED_GEMM_SRC_GEMM_BF16XFP32_SM120_H_
