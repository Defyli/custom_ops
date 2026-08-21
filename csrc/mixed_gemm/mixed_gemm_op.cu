/*
 * mixed_gemm_op.cu — Mixed-Precision GEMM，PyTorch C++ 接口实现
 *
 * 对外暴露：
 *   mixed_gemm_cuda(x, w_high, w_low, w_scale, scale, bias, activation,
 *                   fp32_output, force_splitk) -> Tensor
 *
 * 计算语义：
 *   y = activation(x @ (w_high + w_low * scale)^T + bias)
 *
 * 其中 (w_high, w_low) 由 fp32 权重离线预切分（split_mixed_precision_weight）：
 *   w_high = bf16(W)                              主项，bf16 tensor core
 *   w_low  = quant((W - fp32(w_high)) / scale)    residual 项，低精度 tensor core
 *
 * residual 后端由 w_low 的 dtype 决定：
 *   - float8_e4m3fn：FP8 后端（SM89+；编译期需 CUDA >= 12.4，运行时查询
 *     mixed_gemm_fp8_compiled()）
 *   - int8 + w_scale：INT8 动态量化后端（SM80+；per-channel weight scale
 *     离线提供，per-row activation scale 运行时生成）
 *
 * 本文件替代原 TRT 插件的 mixed_gemm_runtime.cpp 调度层：tile/split-K
 * 启发式原样移植（P3/P4 wall-clock 模型），workspace 由 torch caching
 * allocator 分配，DispatchPlan 序列化结构对 torch 接口无意义故不再需要。
 */

#include "mixed_gemm_op.h"
#include "gemm_bf16xfp32_sm80.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace mixed_gemm;

// ── split-K 启发式（移植自 TRT runtime，模型与原版逐字一致） ────────────────
//
// P3/P4：对每个候选 split_k 直接建模 persistent kernel 的关键路径墙钟时间
//   relative_time(split_k) = ceil(tiles*split_k / grid) / split_k * (1 + overhead)
// 并要求 >=5% 的预测收益才放弃更小的 split_k（防止在模型误差内抖动）。
// 详见原 3rd/trt_plugin/MixedPrecisionGemm/src/mixed_gemm_runtime.cpp 的
// P3/P4 注释（98 形状 A/B 标定：mean regret 0.7%）。
static inline double splitk_relative_time(long long unsplit_tiles, int split_k, int ntile_k,
                                          int sm_count) {
  long long num_tile = unsplit_tiles * split_k;
  long long grid = std::min<long long>(sm_count, num_tile);
  double ceil_tiles_per_cta = std::ceil(static_cast<double>(num_tile) / static_cast<double>(grid));
  constexpr double kOverheadScale = 1.0;
  double k_per_split = static_cast<double>(ntile_k) / split_k;
  double grid_busyness = std::min(1.0, static_cast<double>(unsplit_tiles) / sm_count);
  double overhead = kOverheadScale / k_per_split * grid_busyness;
  return (ceil_tiles_per_cta / split_k) * (1.0 + overhead * (split_k - 1));
}

static int select_split_k(int m, int n, int k, int kTileM, int kTileN, int sm_count) {
  if (const char *force = std::getenv("GEMM_MIXED_FORCE_SPLITK")) {
    return std::atoi(force);
  }
  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;
  long long unsplit_tiles = static_cast<long long>(num_tile_m) * num_tile_n;

  constexpr int kTileK = 64;
  int ntile_k = (k + kTileK - 1) / kTileK;  // 向上取整：感知 ragged-K

  constexpr double kMinRelativeGain = 0.05;  // 要求 >=5% 预测提升
  int best_k = 1;
  double best_t = splitk_relative_time(unsplit_tiles, 1, ntile_k, sm_count);
  for (int candidate : {2, 4, 8, 16}) {
    if (ntile_k < candidate) continue;
    double t = splitk_relative_time(unsplit_tiles, candidate, ntile_k, sm_count);
    if (t < best_t * (1.0 - kMinRelativeGain)) {
      best_k = candidate;
      best_t = t;
    }
  }
  return best_k;
}

// ── workspace 分段辅助（256 字节对齐） ────────────────────────────────────────
namespace {

constexpr size_t kWorkspaceAlignment = 256;

inline size_t align_up(size_t value) {
  return (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

}  // namespace

// ── CUDA 实现 ─────────────────────────────────────────────────────────────────
torch::Tensor mixed_gemm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& w_high,
    const torch::Tensor& w_low,
    const c10::optional<torch::Tensor>& w_scale,
    double scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t activation,
    bool fp32_output,
    int64_t force_splitk
) {
    // ── 输入验证 ──────────────────────────────────────────────────────────────
    TORCH_CHECK(x.is_cuda(),      "x must be a CUDA tensor");
    TORCH_CHECK(w_high.is_cuda(), "w_high must be a CUDA tensor");
    TORCH_CHECK(w_low.is_cuda(),  "w_low must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kBFloat16,
                "x must be float32 or bfloat16");
    TORCH_CHECK(w_high.dtype() == torch::kBFloat16, "w_high must be bfloat16");
    const bool w_low_is_fp8 = (w_low.dtype() == torch::kFloat8_e4m3fn);
    const bool w_low_is_int8 = (w_low.dtype() == torch::kInt8);
    TORCH_CHECK(w_low_is_fp8 || w_low_is_int8,
                "w_low must be float8_e4m3fn (FP8 backend) or int8 (INT8 backend); "
                "produce it with split_mixed_precision_weight()");

    TORCH_CHECK(x.is_contiguous(),      "x must be contiguous");
    TORCH_CHECK(w_high.is_contiguous(), "w_high must be contiguous");
    TORCH_CHECK(w_low.is_contiguous(),  "w_low must be contiguous");

    TORCH_CHECK(x.dim() >= 1,      "x must have at least 1 dim (..., K)");
    TORCH_CHECK(w_high.dim() == 2, "w_high must be 2D (N, K)");
    TORCH_CHECK(w_low.dim() == 2,  "w_low must be 2D (N, K)");

    const int64_t k = x.size(-1);
    const int64_t n = w_high.size(0);
    const int64_t k_w = w_high.size(1);

    TORCH_CHECK(k > 0 && n > 0, "K and N must be positive");
    TORCH_CHECK(k % 8 == 0, "K must be a multiple of 8 (cp.async 128-bit alignment)");
    TORCH_CHECK(w_low.size(0) == n && w_low.size(1) == k_w,
                "w_low shape must match w_high (N, K)");
    TORCH_CHECK(k_w == k, "K mismatch between x (", k, ") and w_high (", k_w, ")");

    const bool use_int8 = w_low_is_int8;
    if (use_int8) {
        TORCH_CHECK(w_scale.has_value(),
                    "INT8 backend requires w_scale (per-channel quantization scale, "
                    "shape (N,), float32)");
        TORCH_CHECK(w_scale->dtype() == torch::kFloat32, "w_scale must be float32");
        TORCH_CHECK(w_scale->is_contiguous(), "w_scale must be contiguous");
        TORCH_CHECK(w_scale->dim() == 1 && w_scale->size(0) == n,
                    "w_scale must be 1D of length N");
    } else {
        TORCH_CHECK(!w_scale.has_value(),
                    "w_scale is only accepted by the INT8 backend "
                    "(w_low.dtype == torch.int8)");
    }

    const bool has_bias = bias.has_value();
    if (has_bias) {
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n,
                    "bias must be 1D of length N");
    }

    TORCH_CHECK(activation >= 0 && activation <= 2,
                "activation must be 0 (identity), 1 (silu) or 2 (gelu-tanh)");

    const int64_t m = (x.dim() == 1) ? 1 : x.numel() / k;
    TORCH_CHECK(m > 0, "x has zero elements");
    TORCH_CHECK(m <= INT32_MAX && n <= INT32_MAX && k <= INT32_MAX,
                "M/N/K must fit in int32");

    // ── 架构与后端能力检查 ────────────────────────────────────────────────────
    const c10::cuda::OptionalCUDAGuard device_guard(x.device());
    const cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
    const int major = props->major;
    const int minor = props->minor;
    const int sm_count = props->multiProcessorCount;

    TORCH_CHECK(major >= 8,
                "mixed_gemm requires compute capability >= 8.0 (Ampere); got ",
                major, ".", minor);
    if (!use_int8) {
        // FP8 e4m3 mma 需要 SM89+，且需编译期 CUDA >= 12.4
        TORCH_CHECK(mixed_gemm_fp8_compiled(),
                    "FP8 backend (w_low.dtype == float8_e4m3fn) was not compiled into "
                    "this build: the SM89 e4m3 mma.sync requires CUDA >= 12.4 at "
                    "build time. Rebuild with a newer CUDA toolkit, or use the INT8 "
                    "backend (split_mixed_precision_weight(..., backend='int8')).");
        TORCH_CHECK(major > 8 || (major == 8 && minor >= 9),
                    "FP8 backend requires compute capability >= 8.9 (Ada); got ",
                    major, ".", minor, ". Use the INT8 backend instead.");
    }

    // ── tile / split-K 选择 ───────────────────────────────────────────────────
    // 注意：tile 形状同时由 kernel 内部 dispatch 用相同的 (m, n, sm_count) 与
    // 相同的 GEMM_MIXED_FORCE_TILE_* env 重新计算，两侧结果一致，因此这里
    // 计算的 split_flag buffer 形状与 kernel 实际 tile 划分总是吻合的。
    const int tile_m = select_kTileMForN(static_cast<int>(m), static_cast<int>(n), sm_count);
    const int tile_n = select_kTileN(static_cast<int>(m), static_cast<int>(n), sm_count);
    int split_k;
    if (force_splitk > 0) {
        TORCH_CHECK(force_splitk == 1 || force_splitk == 2 || force_splitk == 4 ||
                    force_splitk == 8 || force_splitk == 16,
                    "force_splitk must be one of {1, 2, 4, 8, 16}");
        split_k = static_cast<int>(force_splitk);
    } else {
        split_k = select_split_k(static_cast<int>(m), static_cast<int>(n),
                                 static_cast<int>(k), tile_m, tile_n, sm_count);
    }

    // ── 输出分配 ──────────────────────────────────────────────────────────────
    auto out_sizes = x.sizes().vec();
    out_sizes.back() = n;
    auto out_options = x.device().is_cuda()
        ? torch::TensorOptions().dtype(fp32_output ? torch::kFloat32 : torch::kBFloat16)
                                 .device(x.device())
        : torch::TensorOptions();
    torch::Tensor y = torch::empty(out_sizes, out_options);

    // ── workspace 分配（单块 byte tensor，256 字节对齐分段） ──────────────────
    const bool x_is_fp32 = (x.dtype() == torch::kFloat32);
    const size_t m64 = static_cast<size_t>(m);
    const size_t k64 = static_cast<size_t>(k);
    const size_t n64 = static_cast<size_t>(n);

    size_t cursor = 0;
    size_t x_bf16_off = 0;
    if (x_is_fp32) {
        x_bf16_off = cursor;
        cursor = align_up(cursor + m64 * k64 * 2);
    }
    size_t x_res_off = cursor;  // fp8 / int8 residual 副本，1 字节/元素
    cursor = align_up(cursor + m64 * k64);
    size_t x_scale_off = 0;
    if (use_int8) {
        x_scale_off = cursor;
        cursor = align_up(cursor + m64 * sizeof(float));
    }
    size_t split_y_off = 0, split_flag_off = 0;
    const size_t tiles_m = (m64 + tile_m - 1) / tile_m;
    const size_t tiles_n = (n64 + tile_n - 1) / tile_n;
    if (split_k > 1) {
        split_y_off = cursor;
        cursor = align_up(cursor + static_cast<size_t>(split_k) * m64 * n64 * sizeof(float));
        split_flag_off = cursor;
        cursor = align_up(cursor + tiles_m * tiles_n * sizeof(int));
    }

    auto ws_options = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    torch::Tensor workspace = (cursor > 0) ? torch::empty({(int64_t)cursor}, ws_options)
                                           : torch::empty({0}, ws_options);
    uint8_t* ws_base = workspace.data_ptr<uint8_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    void* x_bf16_ptr = x_is_fp32 ? (ws_base + x_bf16_off) : x.data_ptr();
    void* x_res_ptr = ws_base + x_res_off;
    float* x_scale_ptr = use_int8 ? reinterpret_cast<float*>(ws_base + x_scale_off) : nullptr;
    void* split_y_ptr = (split_k > 1) ? (ws_base + split_y_off) : nullptr;
    void* split_flag_ptr = (split_k > 1) ? (ws_base + split_flag_off) : nullptr;

    // split-K 的 per-tile 完成计数器必须清零（kernel 内 atomicAdd 累加）
    if (split_k > 1) {
        cudaMemsetAsync(split_flag_ptr, 0, tiles_m * tiles_n * sizeof(int), stream);
    }

    // ── activation 量化（x → bf16 主项输入 + 低精度 residual 副本） ──────────
    bool quantized = false;
    if (use_int8) {
        quantized = x_is_fp32
            ? quantize_x_fp32_to_int8(
                  reinterpret_cast<const float*>(x.data_ptr()), x_bf16_ptr, x_res_ptr,
                  x_scale_ptr, static_cast<int>(m), k64, stream)
            : quantize_x_bf16_to_int8(
                  x.data_ptr(), x_res_ptr, x_scale_ptr, static_cast<int>(m), k64, stream);
    } else {
#if MIXED_GEMM_FP8_ENABLED
        quantized = x_is_fp32
            ? convert_x_fp32_to_bf16_fp8(
                  reinterpret_cast<const float*>(x.data_ptr()), x_bf16_ptr, x_res_ptr,
                  static_cast<int64_t>(m) * k, stream)
            : quantize_x_bf16_to_fp8(
                  x.data_ptr(), x_res_ptr, static_cast<int64_t>(m) * k, stream);
#else
        TORCH_CHECK(false, "unreachable: FP8 backend guarded above");
#endif
    }
    TORCH_CHECK(quantized, "activation quantization kernel launch failed");

    // ── 主 GEMM launcher ──────────────────────────────────────────────────────
    const float scale_f = static_cast<float>(scale);
    const float* bias_ptr = has_bias ? bias->data_ptr<float>() : nullptr;
    bool launched = false;
    if (use_int8) {
        auto launcher = resolve_gemm_bf16xfp32_int8_launcher(
            static_cast<int>(activation), has_bias);
        TORCH_CHECK(launcher != nullptr, "INT8 launcher resolution failed");
        launched = launcher(y.data_ptr(), split_y_ptr, split_flag_ptr,
                            x_bf16_ptr, w_high.data_ptr(), w_low.data_ptr(),
                            w_scale->data_ptr<float>(), bias_ptr, x_res_ptr,
                            x_scale_ptr, static_cast<int>(m), static_cast<int>(n),
                            static_cast<int>(k), scale_f, fp32_output, split_k,
                            sm_count, stream);
    } else {
#if MIXED_GEMM_FP8_ENABLED
        auto launcher = resolve_gemm_bf16xfp32_epilogue_launcher(
            static_cast<int>(activation), has_bias);
        TORCH_CHECK(launcher != nullptr, "FP8 launcher resolution failed");
        launched = launcher(y.data_ptr(), split_y_ptr, split_flag_ptr,
                            x_bf16_ptr, w_high.data_ptr(), w_low.data_ptr(),
                            bias_ptr, x_res_ptr, static_cast<int>(m),
                            static_cast<int>(n), static_cast<int>(k), scale_f,
                            fp32_output, split_k, sm_count, stream);
#else
        TORCH_CHECK(false, "unreachable: FP8 backend guarded above");
#endif
    }
    TORCH_CHECK(launched, "mixed_gemm kernel launch failed");
    return y;
}

// ── CPU 回退：不支持，显式报错 ────────────────────────────────────────────────
torch::Tensor mixed_gemm_cpu(
    const torch::Tensor& x,
    const torch::Tensor& w_high,
    const torch::Tensor& w_low,
    const c10::optional<torch::Tensor>& w_scale,
    double scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t activation,
    bool fp32_output,
    int64_t force_splitk
) {
    TORCH_CHECK(false,
                "mixed_gemm is CUDA-only (sm80+ tensor cores); got a CPU tensor");
}
