// Mixed-precision GEMM (bf16 + fp8/int8 residual) — kernel entry points.
//
// Ported from 3rd/trt_plugin/MixedPrecisionGemm (TensorRT plugin) into the
// open-source repo as a torch custom op. See gemm_bf16xfp32_sm80.cu for the
// full algorithm description and mixed_gemm_op.cu for the torch-facing API.

#ifndef MIXED_GEMM_SRC_GEMM_BF16XFP32_SM80_H_
#define MIXED_GEMM_SRC_GEMM_BF16XFP32_SM80_H_

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdint.h>

// FP8 residual 后端的编译期可用性：SM89 e4m3 mma.sync 需要 CUDA >= 12.4
// 才能汇编（与 cute/arch/mma_sm89.hpp 的 CUTE_ARCH_MMA_F32_SM89_SUPPORTED
// 同一判定，这里独立重写以便本头文件不依赖 cute）。编译期不可用时
// （如 CUDA 11.8）仅 INT8 后端可用；运行时由 mixed_gemm_fp8_compiled()
// 查询。
#if !defined(MIXED_GEMM_FP8_ENABLED)
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 12 || \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define MIXED_GEMM_FP8_ENABLED 1
#else
#define MIXED_GEMM_FP8_ENABLED 0
#endif
#endif

namespace mixed_gemm {

// P0 (OPTIMIZATION_PLAN.md 1.1 / P0): must mirror the m<=64 -> kTileM=64
// dispatch in gemm_bf16xfp32_async_sm80 (gemm_bf16xfp32_sm80.cu) exactly,
// since both entry.cc's split_flag scratch buffer shape and its
// select_split_k tile-count math are only correct if they agree with the
// kTileM the kernel actually launches with.
//
// A kTileM=32 extension (matching hpc-ops' sm90 kernel, which goes as
// narrow as kTileM=16) was implemented and A/B measured here, but showed
// zero throughput gain: with m<=32 the CTA-tile's shared-memory traffic is
// dominated by the W_high/W_low loads (kTileN*kTileK*2*sizeof(bf16) =
// 32KB, independent of kTileM) rather than the X load (kTileM*kTileK*
// sizeof(bf16), only 4KB at kTileM=32 vs 8KB at kTileM=64) -- so halving
// kTileM only shrinks the minority operand's traffic/compute, while the
// majority-share weight cp.async time (unaffected by kTileM) remains the
// bottleneck. Direct measurement confirmed m=16 and m=32 take *identical*
// wall-clock time under both kTileM=32 and kTileM=64 across every shape
// tried (weight-bandwidth-bound and launch-overhead-bound alike) -- see
// the gemm_bf16xfp32_sm80.cu P0 comment for the fuller writeup. This is a
// direct consequence of *not* having sm90's TMA + producer/consumer
// warpgroup split (which decouples weight-load latency from MMA issue
// rate); on sm80's single-warpgroup cp.async pipeline, weight load and MMA
// compete for the same critical path regardless of kTileM.
inline int select_kTileM(int m) { return m <= 64 ? 64 : 128; }

// P5/P6 (wave-quantization- and occupancy-aware CTA tile shape for the m>64
// path): pick between a 128x128 and a 64x64 CTA tile based on how well
// `m`x`n` divides into the grid of persistent CTAs, *and* on how many waves
// the 128x128 grid needs -- see the P6 comment in gemm_bf16xfp32_sm80.cu for
// why occupancy (not just wave utilization) now matters here.
//
// Background: this kernel launches a persistent grid of
// min(sm_count * max_active_blocks_per_sm, num_tile) CTAs (see the P6 grid
// sizing in launch_gemm_bf16xfp32_kernel_sm80 and gemm_bf16xfp32_kernel's
// get_next_tile loop). At kTileM=kTileN=128 the CTA's register usage limits
// occupancy to exactly 1 block/SM regardless of P6, so its grid is stuck at
// sm_count; whenever num_tile doesn't divide sm_count evenly, the last
// "wave" is only partially full and every SM still has to wait for it --
// e.g. m=n=1331 at 128x128 is 11*11=121 tiles over 114 SMs: 2 waves, but
// the second wave only has 7 live tiles (utilization 121/(2*114) = 53%).
// At kTileM=kTileN=64, register pressure is low enough for 2 blocks/SM, so
// P6 doubles the grid to 2*sm_count -- this both raises occupancy (better
// latency hiding) *and* usually improves wave utilization, at the cost of
// ~4x the tile count (and therefore ~2x the total X/W operand traffic,
// since each CTA tile reads a full K-depth strip of both operands).
//
// Because 64x64 gets both a utilization *and* an occupancy win, it beats
// 128x128 far more often than pre-P6 measurements suggested: A/B testing
// across ~30 shapes after the P6 grid change showed 128x128 only wins once
// the 128x128 grid is both (a) reasonably well-utilized (util >= ~0.85) and
// (b) large enough in absolute wave count (>= ~3 waves) that the fixed
// per-wave latency-hiding penalty of running at 1 block/SM is amortized
// against enough real work to outweigh 64x64's extra traffic. Below either
// threshold, 64x64 wins (sometimes by >20%). A 128x64 middle tile was also
// measured (kept kTileM=128, so still occupancy-limited to 1 block/SM) --
// it never beat both endpoints in any tested shape (it inherits 128x128's
// occupancy ceiling while still paying 64x64-like extra W traffic), so it
// was removed in favor of this plain binary choice.
//
// Only applies to the m>64 path (select_kTileM already returns 64x128 for
// m<=64, its own separate P0 fix for a different problem -- wasted MMA
// cycles on out-of-range rows, not wave quantization -- and was measured
// to see no further gain from also narrowing kTileN there).
//
// Returns the selected kTileN; select_kTileMForN returns the matching kTileM
// (they always move together: 128x128 or 64x64, never mixed).
inline bool select_use_wide_tile(int m, int n, int sm_count) {
  long long ntm = (m + 127) / 128;
  long long ntn = (n + 127) / 128;
  long long nt = ntm * ntn;
  long long waves = (nt + sm_count - 1) / sm_count;
  double util = static_cast<double>(nt) / (static_cast<double>(waves) * sm_count);

  constexpr double kUtilThreshold = 0.85;
  constexpr long long kMinWaves = 3;
  return util >= kUtilThreshold && waves >= kMinWaves;
}

inline int select_kTileN(int m, int n, int sm_count) {
  if (m <= 64) return 128;  // P0 path already uses the widest safe kTileN

  // Debug/testing override: force a specific kTileN (64 or 128)
  if (const char *force = std::getenv("GEMM_MIXED_FORCE_TILE_N")) {
    return std::atoi(force);
  }

  return select_use_wide_tile(m, n, sm_count) ? 128 : 64;
}

// select_kTileMForN returns the kTileM corresponding to select_kTileN's
// choice: 128 iff select_kTileN returned 128 (128x128), else 64 (64x64).
inline int select_kTileMForN(int m, int n, int sm_count) {
  if (m <= 64) return select_kTileM(m);

  // Debug/testing override: force a specific kTileM (64 or 128)
  if (const char *force = std::getenv("GEMM_MIXED_FORCE_TILE_M")) {
    return std::atoi(force);
  }

  return select_use_wide_tile(m, n, sm_count) ? 128 : 64;
}

// Converts one row-major FP32 activation buffer to the two row-major
// activation buffers consumed by the GEMM. FP8 conversion is deliberately
// performed from the rounded BF16 value to preserve the v1 numerical path.
// All FP8-backend declarations below are only present when the FP8 residual
// backend is compiled in (CUDA >= 12.4); query mixed_gemm_fp8_compiled().
#if MIXED_GEMM_FP8_ENABLED
bool convert_x_fp32_to_bf16_fp8(const float *x_fp32_ptr, void *x_bf16_ptr,
                                void *x_fp8_ptr, int64_t numel,
                                cudaStream_t stream) noexcept;
bool quantize_x_bf16_to_fp8(const void *x_bf16_ptr, void *x_fp8_ptr,
                             int64_t numel, cudaStream_t stream) noexcept;
#endif
bool convert_x_fp32_to_bf16(const float *x_fp32_ptr, void *x_bf16_ptr,
                            int64_t numel, cudaStream_t stream) noexcept;

// INT8 后端（SM80+）：一次读取 FP32 activation，产出主项所需的 BF16、
// residual 所需的 int8（按 row 动态对称量化，量化源为 BF16 舍入值）和
// per-row scale [m]。k 必须满足算子的 K%8==0 契约。bf16 输入版只产出
// int8 + per-row scale（x 本身已是主项输入）。
bool quantize_x_fp32_to_int8(const float *x_fp32_ptr, void *x_bf16_ptr,
                             void *x_int8_ptr, float *x_scale_ptr, int m,
                             int64_t k, cudaStream_t stream) noexcept;
bool quantize_x_bf16_to_int8(const void *x_bf16_ptr, void *x_int8_ptr,
                             float *x_scale_ptr, int m, int64_t k,
                             cudaStream_t stream) noexcept;

// FP8 residual 后端（SM89 e4m3 mma）是否编入了本编译单元。
// 仅取决于编译时的 CUDA 版本（cute 要求 >= 12.4）；GPU 架构支持由调用方
// 在运行时检查（FP8 需 sm89+，INT8 需 sm80+）。
bool mixed_gemm_fp8_compiled() noexcept;

// Computes y = (x @ (w_high + w_low * scale)^T) using a BF16 tensor-core
// GEMM plus an FP8 E4M3 tensor-core correction GEMM. W_low is quantized to
// E4M3 offline; only the dynamic X activation is quantized at runtime.
#if MIXED_GEMM_FP8_ENABLED
//   x_ptr, w_high_ptr : BF16 byte payloads, shapes (m,k), (n,k).
//                       w_high may be transported as INT32 [n,k/2].
//   w_low_fp8_ptr     : offline-quantized E4M3, shape (n,k)
//   x_fp8_ptr         : caller-provided E4M3 workspace, shape (m,k),
//                       overwritten by this call before the GEMM launch
//   y_ptr                        : output, shape (m,n), dtype bf16 or fp32
//   splitk_y_ptr                 : scratch fp32 buffer, shape (splitk,m,n),
//                                  required iff splitk > 1
//   split_flag_ptr               : scratch int32 buffer, shape
//                                  (ceil(m/kTileM), ceil(n/kTileN)), zero
//                                  initialized, required iff splitk > 1.
//                                  kTileM/kTileN are chosen by
//                                  select_kTileMForN/select_kTileN (see
//                                  above) -- callers must use those same
//                                  helpers (with the same sm_count) when
//                                  sizing this buffer, since they must
//                                  agree with whatever tile shape this call
//                                  actually dispatches to.
//   sm_count                     : device's multiProcessorCount, used only
//                                  to pick between the 128x128 and 64x64 CTA
//                                  tile shapes (see select_kTileN) -- does
//                                  not affect numerics.
//
// Returns false if the requested splitk value is not supported.
using GemmFixedEpilogueLauncher = bool (*)(
    void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
    const void *x_ptr, const void *w_high_ptr, const void *w_low_fp8_ptr,
    const float *bias_ptr, void *x_fp8_ptr, int m, int n, int k, float scale,
    bool use_fp32_output, int splitk, int sm_count, cudaStream_t stream);

GemmFixedEpilogueLauncher resolve_gemm_bf16xfp32_epilogue_launcher(
    int activation_type, bool has_bias) noexcept;
#endif  // MIXED_GEMM_FP8_ENABLED

// INT8 residual 后端（SM80+，s8*s8->s32 tensor core）的固定 launcher：
// 额外携带 per-channel weight scale（w_scale_ptr，常量）与 per-row
// activation scale（x_scale_ptr，workspace，由 quantize_x_fp32_to_int8
// 在每次调用时生成）。x_fp8_ptr 槽位承载 int8 activation workspace。
using GemmFixedInt8EpilogueLauncher = bool (*)(
    void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
    const void *x_ptr, const void *w_high_ptr, const void *w_low_int8_ptr,
    const float *w_scale_ptr, const float *bias_ptr,
    void *x_int8_ptr, float *x_scale_ptr, int m, int n, int k, float scale,
    bool use_fp32_output, int splitk, int sm_count, cudaStream_t stream);

GemmFixedInt8EpilogueLauncher resolve_gemm_bf16xfp32_int8_launcher(
    int activation_type, bool has_bias) noexcept;

}  // namespace mixed_gemm

#endif  // MIXED_GEMM_SRC_GEMM_BF16XFP32_SM80_H_
