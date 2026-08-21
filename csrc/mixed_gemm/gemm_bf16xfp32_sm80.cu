// Mixed-precision GEMM: y = x @ (w_high + w_low * scale)^T, optionally
// fused with bias + activation in the epilogue.
//
// Solves the generative-recommender precision/performance dilemma: bf16
// weights lose too much accuracy for the final linear layers, while tf32
// tensor cores are too slow. Instead, one fp32 weight W is pre-split offline
// (see split_mixed_precision_weight in recsys.py) into
//     w_high = bf16(W)                                (main term)
//     w_low  = quant((W - fp32(w_high)) / scale)      (residual term)
// and the GEMM computes the sum of a bf16 tensor-core GEMM (w_high) plus a
// low-precision residual GEMM (w_low), recovering near-fp32 accuracy at
// bf16-like speed.
//
// This file is a port of the 3rd/trt_plugin/MixedPrecisionGemm TensorRT
// plugin kernel (which itself is an sm80/sm89 re-implementation of
// hpc-ops' sm90 operator): the kernel body, tile/split-K heuristics and
// numerics are unchanged; the TRT plugin/runtime layers are replaced by a
// torch custom op (see mixed_gemm_op.cu).
//
// The sm80/sm89 implementation uses only Ampere-era primitives:
//   - cp.async based multi-stage global->shared pipeline (no TMA / mbarrier)
//   - ldmatrix (SM75_U32x4_LDSM_N) based shared->register loads
//   - mma.sync m16n8k16 (SM80_16x8x16_F32BF16BF16F32_TN) bf16 tensor cores
//
// Residual backends (compile-time selectable, see ResidualBackend below):
//   - FP8 (e4m3) m16n8k32 mma (SM89+, requires CUDA >= 12.4 at build time;
//     guarded by MIXED_GEMM_FP8_ENABLED below)
//   - INT8 dynamic per-row quantization m16n8k32 mma (SM80+, any CUDA)
//
// Numerical algorithm (same decomposition as the sm90 kernel; combine
// strategy differs -- see below):
//   Given a "true" fp32 weight W, the caller pre-splits it (see
//   split_mixed_precision_weight（recsys.py）) into:
//     w_high = bf16(W)
//     w_low  = bf16((W - fp32(w_high)) / scale)      # scale = 1/256
//   so that  W ~= w_high + w_low * scale.
//   w_high @ X runs as a bf16 tensor-core mma (m16n8k16), same as before.
//   w_low  @ X runs as an fp8 (e4m3) tensor-core mma (m16n8k32, Ada-only --
//   see the W_low-in-fp8 comment at its use site in gemm_bf16xfp32_kernel):
//   w_low/X are requantized to e4m3 up front, by a separate `quantize_kernel`
//   pass (see the A2/pre-quantization comment below), not in-kernel.
//   Unlike w_high, `scale` can NOT be folded into w_low before this fp8 cast
//   -- w_low's natural range (+-0.0625) sits inside e4m3's normal range, but
//   w_low*scale's range (+-2^-12) underflows e4m3's smallest subnormal,
//   silently zeroing the whole term (verified numerically). So scale is
//   instead folded in *after* each fp8 mma, in fp32, directly into the
//   (single, shared) fp32 accumulator:
//     tYr  = sum_ik w_high[ik] @ X[ik]                  (bf16 mma, direct accumulate)
//     tYr += scale * (w_low[ikf] @ X[ikf])              (fp8 mma -> scratch, scaled, folded in)
//   Both mmas accumulate into the *same* tYr fragment (a single fp32
//   accumulator, not two) because TiledMMA (bf16) and TiledMMAFp8 share
//   the same MMAThrLayout and Permutation_M/N (only Permutation_K differs),
//   so partition_fragment_C/partition_C produce bit-identical per-thread
//   layouts for both -- verified. This halves the accumulator's register
//   footprint vs. a naive two-accumulator (tYr_high/tYr_low) design
//   (measured as ~56% of the kernel's total regs/thread before this merge),
//   the single largest lever on this kernel's occupancy (register-limited
//   to 1 block/SM -- see the launch bounds discussion below).
//   (An A/B experiment additionally tried sharing w_high/w_low's *operand*
//   fragments via a 2-slot ping-pong register array, on top of the tYr
//   merge above. It was measured to *increase* total register usage in
//   most configs -- the extra micro-step indexing/branching needed to
//   ping-pong the slots outweighed the savings, since ptxas already reuses
//   short-lived fragment registers on its own -- so it was reverted; each
//   of w_high/w_low keeps its own independent fragment below.)
//   (A second A/B experiment went further and time-shared W_low/W_high's
//   *smem* buffer itself via a single-slot ping-pong -- issue+drain
//   W_low, ldmatrix it, then issue+drain W_high into the now-free slot,
//   serializing what used to be two independently-prefetched kStage-deep
//   buffers. This did free ~32KB/CTA of smem (98KB -> ~66KB at
//   kTileM=kTileN=128, kStage=2), enough headroom to target 2 blocks/SM
//   on sm89's 100KB budget instead of 1. Measured on the three
//   production shapes plus a broader shape sweep, it was a clear net
//   loss: -15% to -30% throughput. Losing W's cross-K-slab cp.async
//   prefetch (W load and consume become strictly serialized every
//   K-slab: issue -> wait -> sync -> ldmatrix -> sync, twice, before a
//   single mma pair can even start) cost far more on the critical path
//   than the extra occupancy ever recovered. Reverted; W_low/W_high each
//   keep their own independent kStage-deep smem buffer below.)
//
// A2 (pre-quantization via a scratch buffer, instead of in-kernel
// register-level requantization):
//   An earlier revision of this kernel derived W_low's and X's e4m3
//   fragments *in-kernel*, purely in registers, right before each fp8 mma
//   issue (converting straight out of the bf16 smem tiles X/W_high already
//   use -- see git history for that version). That fused design was
//   measured (via cuobjdump SASS inspection + `-Xptxas -v`) to need ~272
//   registers/thread at kTileM=kTileN=128 -- *above* sm89's 255
//   register/thread hardware ceiling -- forcing ptxas to spill ~200+
//   bytes/thread to local memory (itself backed by global memory) spread
//   across both the prologue and the steady-state K-loop, which is far more
//   expensive than the extra fp8 tensor-core throughput the fused fp8 mma
//   was supposed to buy. The root cause: the fp8 path needed its *own*
//   independent set of live registers (tWLr_fp8/tXr_fp8 fragments, the
//   fp8_converter's implicit temporaries, tYr_fp8_tmp) alive
//   *simultaneously* with the bf16 path's tWHr/tXr/tYr and the g2s
//   copy/predicate state -- ptxas had no opportunity to time-share those
//   registers because both paths are interleaved every `ik` sub-step (see
//   the software-pipelining comment below the kernel).
//
//   This revision instead precomputes W_low's and X's e4m3 versions once,
//   up front, via a dedicated `quantize_kernel` pass that writes them to a
//   plain row-major fp8 scratch buffer in global memory (see
//   launch_gemm_bf16xfp32_kernel_sm80: quantize_kernel launches on the same
//   stream immediately before gemm_bf16xfp32_kernel, so no extra
//   synchronization is needed and the two kernels compose safely inside an
//   externally-captured CUDA graph -- e.g. a TensorRT plugin's own graph
//   capture -- since neither one creates or depends on a graph of its own).
//   gemm_bf16xfp32_kernel then treats the fp8 scratch buffers exactly like
//   a third pair of (X, W) operands: cp.async g2s into their own swizzled
//   smem tiles (sX8/sWL8, parallel to sX/sWH), then ldmatrix s2r into
//   registers (tXr_fp8/tWLr_fp8) via the *same* Copy_Atom<SM75_U32x4_LDSM_N>
//   mechanism already used for the bf16 operands -- no per-element
//   coordinate lookup, no in-kernel NumericConverter, no extra live
//   registers beyond what a completely ordinary ldmatrix-fed mma operand
//   needs. This trades one extra (cheap, memory-bound, fully parallel
//   elementwise) kernel launch for eliminating the register pressure that
//   made the fused design spill -- measured to bring the main kernel's
//   register usage back under the 255/thread ceiling with zero spill (see
//   OPTIMIZATION_RESULTS.md).
//
// Split-K: works exactly like the sm90 kernel -- the K dimension is split
// into `splitk` interleaved chunks, each processed by a (possibly)
// different CTA; partial fp32 results are written to a scratch buffer
// shaped (splitk, m, n) and reduced by the same CTAs once all splitk
// partials for a given output tile are available (tracked via an atomic
// counter per output tile).

#include <cuda.h>
#include <stdio.h>

#include <algorithm>
#include <type_traits>
#include <mutex>

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"
#include "gemm_bf16xfp32_sm80.h"
#include "utils.cuh"

// FP8 residual backend availability is a *build-time* property: the SM89
// e4m3 mma.sync instruction only assembles under CUDA >= 12.0 (cute gates
// it behind CUTE_ARCH_MMA_F32_SM89_SUPPORTED, defined by
// cute/arch/mma_sm89.hpp when __CUDACC_VER_MAJOR__ >= 12.4). Under older
// toolchains (e.g. CUDA 11.8) only the INT8 dynamic-quantization backend is
// compiled; mixed_gemm_op.cu queries mixed_gemm_fp8_compiled() to pick the
// default backend at runtime. The macro itself is defined in
// gemm_bf16xfp32_sm80.h (same version gate, rewritten so the header does not
// depend on cute); this cross-check keeps the two definitions honest.
#if defined(CUTE_ARCH_MMA_F32_SM89_SUPPORTED)
static_assert(MIXED_GEMM_FP8_ENABLED, "cute says FP8 mma is supported but MIXED_GEMM_FP8_ENABLED is 0");
#else
static_assert(!MIXED_GEMM_FP8_ENABLED, "MIXED_GEMM_FP8_ENABLED is 1 but cute says FP8 mma is unsupported");
#endif

namespace mixed_gemm {

namespace kernels {

using namespace cute;  // NOLINT

// Residual 后端抽象：主项 BF16 mainloop、split-K、
// workspace、epilogue 全部共享；后端差异仅在 residual 的元素类型、MMA
// atom、累加器类型和 scale 还原方式。
//
// Fp8ResidualBackend（SM89+，需 CUDA >= 12.4 编译）：W_low/X 以 e4m3 表示，
// FP8 MMA（K=32）→ FP32 scratch，逐次 mma 乘统一 residualScale 后并入主
// FP32 累加器。scale 不能预折叠（会低于 e4m3 最小亚正规值），所以必须逐
// mma 应用。
//
// Int8ResidualBackend（SM80+，含无法使用 FP8 MMA 的老驱动 SM89 机器）：
// W_low 按 output-channel 对称量化（s_w[n] = max|R[n,:]|/127，离线），
// X 按 row 动态对称量化（s_x[m] = max|BF16(X)[m,:]|/127，运行时由
// quantize_x_*_to_int8 kernel 生成，量化源与 FP8 路径一致——BF16 舍入
// 后的值）。s8*s8->s32 MMA（K=32，与 FP8 atom 的 A/B/C fragment 布局
// 逐位一致）→ int32 累加器贯穿整个 K-loop，每个输出 tile 结束时一次性
// 按 residualScale * s_x[row] * s_w[col] 逐元素 scale 并入主 FP32 累加器。
// INT8 的 scale 可以安全地推迟到 K-loop 末尾：int32 累加是精确的
// （|x|,|w| <= 127，K 不超过数百万时不会溢出 int32），线性性保证
// sum(acc)*s == sum(acc*s)。
#if MIXED_GEMM_FP8_ENABLED
struct Fp8ResidualBackend {
  using Element = cute::float_e4m3_t;
  using AccElement = float;
  using MmaAtom = SM89_16x8x32_F32E4M3E4M3F32_TN;
};
#endif

struct Int8ResidualBackend {
  using Element = int8_t;
  using AccElement = int32_t;
  using MmaAtom = SM80_16x8x32_S32S8S8S32_TN;
};

// Linearizes blockIdx.x into (itile_m, itile_n, ichunk):
//   - kBlockSwizzle groups CTAs into MxN super-tiles to improve L2 reuse of
//     the weight matrix across the M dimension.
//   - ichunk in [0, kSplitK) selects which interleaved K-chunk this CTA
//     will process for tile (itile_m, itile_n).
template <int kBlockSwizzle, int kSplitK>
__device__ __forceinline__ auto get_next_tile(int iblock, int num_tile_m, int num_tile_n,
                                              cutlass::FastDivmod swizzle_divider,
                                              cutlass::FastDivmod flat_divider) {
  int itile_m, itile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n * kSplitK;
  int total_swizzle_blocks = num_tile_m / kBlockSwizzle * num_tile_bxn;

  if (iblock >= total_swizzle_blocks) {
    flat_divider(itile_m, itile_n, iblock);
  } else {
    int i_bxn, i_bxn_res;
    swizzle_divider(i_bxn, i_bxn_res, iblock);
    itile_m = i_bxn * kBlockSwizzle + i_bxn_res % kBlockSwizzle;
    itile_n = i_bxn_res / kBlockSwizzle;
  }

  int ichunk = itile_n % kSplitK;
  itile_n = itile_n / kSplitK;
  return cute::make_tuple(itile_m, itile_n, ichunk);
}

// Reduces the kSplitK partial fp32 results (scratch buffer shaped
// (kSplitK, m, n) row-major) for one (kTileM, kTileN) output tile and
// writes the final (possibly down-cast to bf16) sum into y_ptr.
//
// N-raggedness (n not a multiple of kTileN) needs its own handling distinct
// from the g2s/epilogue predicate machinery above, for two separate
// reasons:
//   1. This function reduces+stores 4 contiguous N-columns per `ld`/`st`
//      (one float4 each), so a tile whose N-columns run off the end of `n`
//      in the *middle* of a 4-wide group (i.e. n_local % 4 != 0, where
//      n_local is this tile's real column count) can't just skip or keep
//      the whole group -- the in-range columns of that boundary group
//      still need reducing, just not vectorized.
//   2. More subtly, every row of the (kSplitK, m, n) scratch buffer (and
//      of y_ptr) is `n` elements apart. float4 vectorized ld/st requires
//      16-byte (4-float) alignment: a *fixed* column offset
//      (n_tile_start + col0, both always multiples of 4 here) keeps row 0's
//      address aligned, but if n itself isn't a multiple of 4 then
//      successive rows land at *different* mod-4 offsets (row r starts
//      r*n floats into the buffer, and r*n mod 4 varies with r whenever
//      n mod 4 != 0) -- so the float4 fast path is only safe to use at all
//      when n % 4 == 0, regardless of whether this particular tile is the
//      ragged N-tile or not. This is a strictly stronger requirement than
//      #1 and subsumes it: whenever n % 4 != 0 every row of every tile
//      (not just the ragged trailing group of the last N-tile) falls back
//      to the elementwise path.
template <typename Tout, typename Activation, bool HasBias, int kTileM, int kTileN,
          int kSplitK, int kNumWarp>
__device__ __forceinline__ void splitk_reduce(Tout *y_ptr, const float *splitk_y_ptr,
                                              const float *bias_ptr, int m, int n,
                                              int itile_m, int itile_n) {
  int iwarp = threadIdx.x / 32;
  int ilane = threadIdx.x % 32;

  if (itile_m * kTileM + iwarp >= m) return;

  int n_tile_start = itile_n * kTileN;
  int n_local = min(kTileN, n - n_tile_start);  // this tile's real column count (<= kTileN)
  int col0 = ilane * 4;
  if (col0 >= n_local) return;

  auto *y_tile = y_ptr + static_cast<int64_t>(itile_m * kTileM + iwarp) * n + n_tile_start + col0;
  auto *splitk_y_tile =
      splitk_y_ptr + static_cast<int64_t>(itile_m * kTileM + iwarp) * n + n_tile_start + col0;

  int local_m = m - (itile_m * kTileM + iwarp);
  int num_valid_cols = min(4, n_local - col0);  // 4 unless this is the ragged trailing group
  bool use_vectorized = (num_valid_cols == 4) && (n % 4 == 0);  // see reason #2 above

#pragma unroll
  for (int irow = 0; irow < kTileM; irow += kNumWarp) {
    if (irow >= local_m) return;

    if (use_vectorized) {
      auto y = load<float, 4>(splitk_y_tile + static_cast<int64_t>(irow) * n);
#pragma unroll
      for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
        auto part = load<float, 4>(splitk_y_tile + static_cast<int64_t>(ichunk) * m * n +
                                    static_cast<int64_t>(irow) * n);
#pragma unroll
        for (int i = 0; i < 4; ++i) y[i] += part[i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (HasBias) y[i] += bias_ptr[n_tile_start + col0 + i];
        y[i] = Activation{}(y[i]);
      }
      store(y_tile + static_cast<int64_t>(irow) * n, to<Tout>(y));
    } else {
      // Either the ragged trailing group (n_local % 4 != 0, only ever
      // reached by the single warp lane whose col0 <= n_local - 1 <
      // col0 + 4) or n % 4 != 0 (every lane, every tile, see reason #2
      // above): elementwise reduce, no vectorized load/store.
      const float *base = splitk_y_tile + static_cast<int64_t>(irow) * n;
      for (int i = 0; i < num_valid_cols; ++i) {
        float acc = base[i];
#pragma unroll
        for (int ichunk = 1; ichunk < kSplitK; ++ichunk) {
          acc += base[static_cast<int64_t>(ichunk) * m * n + i];
        }
        if constexpr (HasBias) acc += bias_ptr[n_tile_start + col0 + i];
        (y_tile + static_cast<int64_t>(irow) * n)[i] = static_cast<Tout>(Activation{}(acc));
      }
    }
  }
}

// A2 pre-quantization pass (see the file-level comment above): converts a
// row-major bf16 (rows, k) matrix to a row-major e4m3 (rows, k) matrix,
// elementwise, with no scaling (the fp8 mma's result is scaled *after* the
// mma -- see the W_low-in-fp8 discussion above -- so this pass is a pure
// numeric_conversion, not a quant/dequant with a scale factor).
//
// This is intentionally a completely generic, un-tiled elementwise kernel
// (grid-stride loop, 128-bit vectorized bf16 loads where alignment allows):
// it runs once per gemm_bf16xfp32 call on each of X and W_low (see
// launch_gemm_bf16xfp32_kernel_sm80), is purely memory-bound (8x compute
// intensity headroom vs. the GEMM itself), and every output element is
// independent -- there is no reason to tie its tiling to kTileM/kTileN/
// kTileK at all, unlike the main GEMM kernel. Writing straight to a plain
// row-major fp8 buffer (not the GEMM kernel's swizzled smem layout --
// that swizzle only applies to the *shared-memory* tile the main kernel's
// cp.async pipeline stages into, not to this global-memory scratch buffer)
// keeps this kernel maximally simple; gemm_bf16xfp32_kernel's own g2s
// cp.async + smem swizzle machinery (identical in structure to the bf16
// operands' own g2s/swizzle, just narrower dtype) handles the layout
// transform into something ldmatrix can consume.
//
// `numel` need not be a multiple of anything -- the tail (numel % 8 != 0,
// i.e. not a whole vectorized group) is handled elementwise by the last
// few threads that would otherwise read/write out of bounds.
#if MIXED_GEMM_FP8_ENABLED
template <typename Tin>
__device__ __forceinline__ void quantize_to_fp8_range(const Tin *__restrict__ src,
                                                       cute::float_e4m3_t *__restrict__ dst, int64_t numel,
                                                       int64_t tid, int64_t stride) {
  using fp8e4m3 = cute::float_e4m3_t;
  cutlass::NumericConverter<fp8e4m3, Tin> fp8_converter;

  constexpr int kVec = 8;  // 8 x bf16 (16B) in, 8 x fp8 (8B) out per iteration
  int64_t numel_vec = numel / kVec * kVec;  // largest multiple of kVec <= numel

  for (int64_t idx = tid * kVec; idx < numel_vec; idx += stride * kVec) {
    auto v_in = load<Tin, kVec>(src + idx);
    vec_t<fp8e4m3, kVec> v_out;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      v_out[i] = fp8_converter(v_in[i]);
    }
    store(dst + idx, v_out);
  }
  // Tail: leftover elements (numel % kVec != 0), handled scalar by however
  // many threads of the grid's first block(s) are needed -- at most kVec-1
  // elements total, so this is always covered by thread indices
  // [0, kVec) regardless of grid/block dims (blockDim.x is always >= 32).
  if (tid < numel - numel_vec) {
    dst[numel_vec + tid] = fp8_converter(src[numel_vec + tid]);
  }
}

template <typename Tin>
__global__ void quantize_to_fp8_kernel(const Tin *__restrict__ src, cute::float_e4m3_t *__restrict__ dst,
                                       int64_t numel) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  quantize_to_fp8_range<Tin>(src, dst, numel, tid, stride);
}
#endif  // MIXED_GEMM_FP8_ENABLED

// Read FP32 X once and materialize both activation representations required by
// the unchanged main GEMM. FP8 is converted from the rounded BF16 value, not
// directly from FP32, so this is numerically identical to the old TRT
// FP32->BF16 boundary followed by the BF16->FP8 pre-pass.
__global__ void convert_x_fp32_to_bf16_kernel(
    const float *__restrict__ src, cute::bfloat16_t *__restrict__ dst_bf16,
    int64_t numel) {
  cutlass::NumericConverter<cute::bfloat16_t, float> converter;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = tid; index < numel; index += stride) {
    dst_bf16[index] = converter(src[index]);
  }
}

// INT8 后端的 activation 量化（fp32 输入版）：
// 一个 warp 处理一行。第一遍把 FP32 转为 BF16（主项 GEMM 的输入，语义与
// convert_x_fp32_to_bf16 一致）并做行内 amax warp 归约；第二遍按
// s_x = amax/127 对称量化到 int8。量化源是 BF16 舍入后的值——与 FP8 路径
// “FP8 从 BF16 舍入值转换”的数值语义保持一致。amax == 0 的全零行取
// s_x = 1 避免除零（量化结果恒 0，residual 贡献为 0，正确）。
__global__ void quantize_x_fp32_to_int8_kernel(
    const float *__restrict__ src, cute::bfloat16_t *__restrict__ dst_bf16,
    int8_t *__restrict__ dst_int8, float *__restrict__ dst_scale,
    int m, int64_t k) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int64_t row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp_id;
  if (row >= m) return;
  const float *src_row = src + row * k;
  cute::bfloat16_t *bf16_row = dst_bf16 + row * k;
  int8_t *int8_row = dst_int8 + row * k;
  cutlass::NumericConverter<cute::bfloat16_t, float> bf16_converter;

  float amax = 0.0f;
  for (int64_t col = lane; col < k; col += 32) {
    cute::bfloat16_t value = bf16_converter(src_row[col]);
    bf16_row[col] = value;
    amax = fmaxf(amax, fabsf(static_cast<float>(value)));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  float inv_scale = 1.0f / scale;
  if (lane == 0) dst_scale[row] = scale;

  for (int64_t col = lane; col < k; col += 32) {
    float value = static_cast<float>(bf16_row[col]);
    int quantized = __float2int_rn(value * inv_scale);
    quantized = quantized > 127 ? 127 : (quantized < -127 ? -127 : quantized);
    int8_row[col] = static_cast<int8_t>(quantized);
  }
}

// INT8 后端的 activation 量化（bf16 输入版）：x 已经是 BF16，无需
// FP32->BF16 转换（量化源同样为 BF16 舍入值，数值语义与 fp32 输入版
// 完全一致），只做行内 amax 归约 + 对称量化。
__global__ void quantize_x_bf16_to_int8_kernel(
    const cute::bfloat16_t *__restrict__ src, int8_t *__restrict__ dst_int8,
    float *__restrict__ dst_scale, int m, int64_t k) {
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int64_t row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) + warp_id;
  if (row >= m) return;
  const cute::bfloat16_t *src_row = src + row * k;
  int8_t *int8_row = dst_int8 + row * k;

  float amax = 0.0f;
  for (int64_t col = lane; col < k; col += 32) {
    amax = fmaxf(amax, fabsf(static_cast<float>(src_row[col])));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  float inv_scale = 1.0f / scale;
  if (lane == 0) dst_scale[row] = scale;

  for (int64_t col = lane; col < k; col += 32) {
    float value = static_cast<float>(src_row[col]);
    int quantized = __float2int_rn(value * inv_scale);
    quantized = quantized > 127 ? 127 : (quantized < -127 ? -127 : quantized);
    int8_row[col] = static_cast<int8_t>(quantized);
  }
}

#if MIXED_GEMM_FP8_ENABLED
__global__ void convert_x_fp32_to_bf16_fp8_kernel(
    const float *__restrict__ src, cute::bfloat16_t *__restrict__ dst_bf16,
    cute::float_e4m3_t *__restrict__ dst_fp8, int64_t numel) {
  using bf16 = cute::bfloat16_t;
  using fp8e4m3 = cute::float_e4m3_t;
  cutlass::NumericConverter<bf16, float> bf16_converter;
  cutlass::NumericConverter<fp8e4m3, bf16> fp8_converter;
  constexpr int kVec = 4;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t numel_vec = numel / kVec * kVec;
  for (int64_t idx = tid * kVec; idx < numel_vec; idx += stride * kVec) {
    auto v_in = load<float, kVec>(src + idx);
    vec_t<bf16, kVec> v_bf16;
    vec_t<fp8e4m3, kVec> v_fp8;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      v_bf16[i] = bf16_converter(v_in[i]);
      v_fp8[i] = fp8_converter(v_bf16[i]);
    }
    store(dst_bf16 + idx, v_bf16);
    store(dst_fp8 + idx, v_fp8);
  }
  for (int64_t idx = numel_vec + tid; idx < numel; idx += stride) {
    bf16 value = bf16_converter(src[idx]);
    dst_bf16[idx] = value;
    dst_fp8[idx] = fp8_converter(value);
  }
}
#endif  // MIXED_GEMM_FP8_ENABLED

// Main GEMM kernel. A single warp-set both issues cp.async loads and
// performs the MMAs (no producer/consumer split; there is no async TMA
// engine to overlap against on sm80/sm89).
//
//   Tin   : bf16 element type of x / w_high / w_low
//   TY    : epilogue smem element type. fp32 when kSplitK > 1 (partials must
//           stay in fp32 until reduced), otherwise == Tout.
//   Tout  : final output element type (bf16 or fp32)
//   TiledMMA    : built from SM80_16x8x16_F32BF16BF16F32_TN, used for W_high
//                 (and, historically, W_low -- see TiledMMAFp8 below).
//   TiledMMAFp8 : built from SM89_16x8x32_F32E4M3E4M3F32_TN, same 8-warp
//                 ThrLayout as TiledMMA (see launch_gemm_bf16xfp32_kernel_sm80),
//                 used for W_low. See the W_low-in-fp8 comment at its use
//                 site below for the full rationale/derivation.
//   G2SCopy     : cp.async tiled copy for the bf16 operands, shared by X and
//                 W_high (both tiles are (rows, kTileK))
//   G2SCopyFp8  : cp.async tiled copy for the pre-quantized fp8 operands
//                 (X_fp8, W_low_fp8 -- see the A2 file-level comment above),
//                 narrower-dtype analogue of G2SCopy, same (rows, kTileK)
//                 tile shape
//   SLayoutX/W/Y : swizzled, K-major (X, W) / plain (Y) smem layouts, bf16
//   SLayoutXFp8/WFp8 : swizzled, K-major smem layouts for the fp8 operands
//                      (narrower swizzle atom -- see launch_gemm_bf16xfp32_kernel_sm80)
//   kStage       : number of cp.async pipeline stages
// (A W_low/W_high single-slot smem ping-pong was tried here to free smem
// for 2 blocks/SM on sm89 -- see the file-level comment above. It was
// measured to regress throughput by 15-30% and was reverted; W_low and
// W_high each keep their own independent kStage-deep smem buffer.)
//
// W_low-in-fp8 (real-time register-level requantization):
//   W_low and X are stored in smem/loaded via cp.async as bf16, exactly as
//   before -- no change to smem footprint, g2s copy, or ldmatrix for W_high
//   / the "X-for-W_high" consumer. But the W_low x X term is, by
//   construction (see split_fp32_weight in split_mixed_precision_weight（recsys.py）),
//   *only* ever used to correct W_high's bf16 rounding error: W_low's
//   values live in [-0.0625, 0.0625] (bf16(W) - W, scaled by 2^-8 back up
//   into a normal bf16 range) and get multiplied by scale=2^-8 before
//   they're added to the final result. Since Ada (sm89) has a native fp8
//   (e4m3) tensor-core mma.sync at m16n8k32 -- 2x the K-throughput per
//   instruction of the m16n8k16 bf16 mma -- this term can be computed from
//   fp8 (e4m3) versions of W_low and X that are derived, purely in
//   registers, from the exact same bf16 smem tiles X/W_high already use,
//   and folded (post-scale) into the *same* fp32 accumulator tYr already
//   used by W_high's bf16 mma (see the accumulator-merge comment at the
//   top of this file for why one shared tYr, not two, is correct here):
//     tYr += scale * sum_k  e4m3(W_low[n,k]) * e4m3(X[m,k])
//   Three correctness-critical details:
//     1. `scale` must NOT be folded into W_low before the fp8 cast (unlike
//        W_high's bf16 path, which accumulates directly with no scale at
//        all -- scale only ever multiplies W_low's contribution). W_low's
//        natural range (+-0.0625 = 2^-4) sits comfortably inside e4m3's
//        normal range (2^-6 .. 448), but W_low*scale's range (+-2^-12) is
//        *below* e4m3's smallest subnormal (2^-9) -- casting the
//        pre-scaled value to e4m3 underflows every single element to
//        exactly 0 (verified numerically), silently zeroing out this whole
//        term. Scale is instead applied once, in fp32, right after each
//        fp8 mma issue (see tYr_fp8_tmp below).
//     2. The bf16-mma fragment (tWHr/tXr, partitioned via TiledMMA) and the
//        fp8-mma fragment (tWLr_fp8/tXr_fp8, partitioned via TiledMMAFp8)
//        do NOT share a common per-thread register layout -- their
//        MMA_Traits::ALayout/BLayout differ (verified: a naive flat-index
//        copy between the two produces garbage). The robust construction
//        is to never reason about either atom's internal register layout
//        at all: partition an *identity* tensor with TiledMMAFp8 to learn
//        which logical (row, k) coordinate of the bf16 smem tile each fp8
//        fragment slot corresponds to, then read+convert that coordinate
//        directly from smem (see s2r_copy_ik_fp8 below). This is correct
//        by construction regardless of either mma atom's register-layout
//        details.
//     3. Unlike the bf16-mma fragment, the fp8-mma's *C* fragment (i.e. the
//        accumulator) DOES share a common per-thread layout with the
//        bf16-mma's C fragment, because partition_fragment_C/partition_C
//        only depend on MMAThrLayout and Permutation_M/N -- both identical
//        between TiledMMA and TiledMMAFp8 (only Permutation_K differs, and
//        K doesn't factor into the M/N-only output tile) -- verified by
//        partitioning an identity tensor with both and comparing every
//        thread's per-slot (row, col) coordinate assignment. So tYr is a
//        valid destination for both tiled_mma's and tiled_mma_fp8's `C`
//        operand, and the two mmas' contributions can be folded into one
//        shared accumulator instead of needing independent tYr_high/tYr_low
//        accumulators.
template <typename Tin, typename TY, typename Tout, typename TiledMMA, typename TiledMMAFp8,
          typename G2SCopy, typename G2SCopyFp8, typename SLayoutX, typename SLayoutW,
          typename SLayoutXFp8, typename SLayoutWFp8, typename SLayoutY,
          int kTileM, int kTileN, int kTileK, int kStage, int kBlockSwizzle, int kSplitK,
          int kNumThreadsLB, typename Activation, bool HasBias, typename ResidualBackend,
          bool HighOnly = false>
__global__ void __launch_bounds__(kNumThreadsLB, 1)
    gemm_bf16xfp32_kernel(const Tin *__restrict__ x_ptr, const Tin *__restrict__ w_high_ptr,
                          const typename ResidualBackend::Element *__restrict__ x_fp8_ptr,
                          const typename ResidualBackend::Element *__restrict__ w_low_fp8_ptr,
                          const float *__restrict__ x_scale_ptr,
                          const float *__restrict__ w_scale_ptr,
                          const float *__restrict__ bias_ptr, Tout *__restrict__ y_ptr,
                          float *__restrict__ splitk_y_ptr, int *__restrict__ split_flag_ptr,
                          int m, int n, int k, float scale, cutlass::FastDivmod swizzle_divider,
                          cutlass::FastDivmod flat_divider,
                          cutlass::FastDivmod reduce_flat_divider) {
  using namespace cute;  // NOLINT
  // 名字保留 fp8 是历史原因：该别名现在指向 residual 后端的元素类型
  // （FP8 为 e4m3，INT8 为 int8_t，同为 1 字节元素）。
  using fp8e4m3 = typename ResidualBackend::Element;
  using ResAcc = typename ResidualBackend::AccElement;
  constexpr bool kIsInt8 = std::is_same_v<ResidualBackend, Int8ResidualBackend>;
  constexpr int kNumThreads = size(TiledMMA{});
  int idx = threadIdx.x;

  // Layout: [sX (bf16) | sWH (bf16) | sX8 (fp8) | sWL8 (fp8)], all kStage-deep.
  // W_low no longer needs a bf16 smem buffer at all (see the A2 file-level
  // comment above) -- only its pre-quantized fp8 tile (sWL8) is staged
  // through smem now, freeing the smem W_low's bf16 buffer used to occupy.
  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_x = reinterpret_cast<Tin *>(shm_data);
  auto *shm_wh = shm_x + cosize(SLayoutX{});
  auto *shm_x8 = reinterpret_cast<fp8e4m3 *>(shm_wh + cosize(SLayoutW{}));
  auto *shm_wl8 = shm_x8 + cosize(SLayoutXFp8{});
  // CUTLASS epilogues stage accumulators in their native ElementAccumulator
  // type, then convert only while leaving shared memory. Keep sY fp32 for
  // both bf16 and fp32 outputs: besides avoiding an early precision loss,
  // this gives every scalar shared access one full 4-byte bank word (bf16
  // scalar stores intrinsically share a bank between adjacent elements and
  // cannot be made fully conflict-free by an address swizzle alone).
  auto *shm_y = reinterpret_cast<float *>(shm_data);  // reuses the operand scratch region

  auto sX = make_tensor(make_smem_ptr(shm_x), SLayoutX{});    // (kTileM, kTileK, kStage)
  auto sWH = make_tensor(make_smem_ptr(shm_wh), SLayoutW{});  // (kTileN, kTileK, kStage)
  auto sX8 = make_tensor(make_smem_ptr(shm_x8), SLayoutXFp8{});    // (kTileM, kTileK, kStage), fp8
  auto sWL8 = make_tensor(make_smem_ptr(shm_wl8), SLayoutWFp8{});  // (kTileN, kTileK, kStage), fp8
  auto sY = make_tensor(make_smem_ptr(shm_y), SLayoutY{});  // (kTileN, kTileM), fp32

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN;
  // Ragged-K support (OPTIMIZATION_PLAN.md / K-not-multiple-of-64 removal),
  // now combined with split-K: ntile_k is a ceiling division, so k need not
  // be a multiple of kTileK, and it also need not be a multiple of kSplitK.
  // The interleaved assignment (gk = ichunk + itile * kSplitK, see
  // get_next_tile) means chunk `ichunk` owns global K-tile indices
  // {ichunk, ichunk + kSplitK, ichunk + 2*kSplitK, ...}. If ntile_k isn't a
  // multiple of kSplitK, the `rem = ntile_k % kSplitK` leftover tiles are
  // indices [ntile_k - rem, ntile_k), i.e. exactly the chunks with
  // ichunk in [0, rem) get one extra tile -- and since the ragged tile is
  // always global index ntile_k - 1, it's always owned by chunk
  // `rem - 1` (when rem > 0) or by every chunk equally landing on a plain
  // multiple (when rem == 0, i.e. non-ragged K). So each chunk can
  // determine both its own tile count and whether it's the one that must
  // handle the ragged tile purely from (ntile_k, kSplitK, ichunk), with no
  // cross-chunk coordination needed -- ragged-K and split-K compose safely.
  // (ntile_k_local itself is computed per-tile below, once `ichunk` is
  // known from get_next_tile -- different output tiles can land on
  // different ichunk, so it can't be hoisted above the while loop.)
  int ntile_k = (k + kTileK - 1) / kTileK;
  int ntile_k_base = ntile_k / kSplitK;
  int ntile_k_rem = ntile_k % kSplitK;

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(idx);
  TiledMMAFp8 tiled_mma_fp8;
  auto thr_mma_fp8 = tiled_mma_fp8.get_slice(idx);

  // "A" operand = weight (high/low), "B" operand = activation, so the MMA
  // computes W^T * X and the CTA output tile is stored as (kTileN, kTileM),
  // matching the sm90 kernel's (N, M) convention.
  //
  // W_low/X's fp8 fragments (tWLr_fp8/tXr_fp8) are now populated via a real
  // ldmatrix s2r copy (see s2r_copy_a_fp8/s2r_copy_b_fp8 below), exactly
  // parallel to W_high/X's bf16 fragments -- see the A2 file-level comment
  // above for why this replaced the old per-element coordinate-lookup
  // requantization.
  auto tWHr = thr_mma.partition_fragment_A(sWH(_, _, 0));
  auto tXr = thr_mma.partition_fragment_B(sX(_, _, 0));  // (MMA, MMA_M, MMA_K)

  auto tWLr_fp8 = thr_mma_fp8.partition_fragment_A(sWL8(_, _, 0));
  auto tXr_fp8 = thr_mma_fp8.partition_fragment_B(sX8(_, _, 0));


  // Single fp32 accumulator, shared by both the bf16 (W_high) and fp8
  // (W_low) mmas. This works because partition_fragment_C only depends on
  // MMAThrLayout and Permutation_M/N (both identical between TiledMMA and
  // TiledMMAFp8 -- see launch_gemm_bf16xfp32_kernel_sm80), not on
  // Permutation_K -- verified: thr_mma.partition_C(identity) and
  // thr_mma_fp8.partition_C(identity) produce bit-identical per-thread
  // (row, col) coordinate assignments, so tYr below is a valid destination
  // for both tiled_mma's and tiled_mma_fp8's `C` operand.
  //   acc = sum_ik W_high[ik] * X[ik]        (bf16 mma, accumulated directly)
  //   acc += scale * (W_low[ikf] * X[ikf])   (fp8 mma -> small scratch fragment,
  //                                            scaled and folded in right after
  //                                            each fp8 mma issue -- see the
  //                                            K-loop below)
  // `scale` still can't be folded into the fp8 operands pre-mma (underflow --
  // see the W_low-in-fp8 comment above), so each fp8 mma targets its own
  // *short-lived* scratch fragment (tYr_fp8_tmp, cleared+consumed once per
  // fp8 mma issue) instead of accumulating across the whole K-loop like tYr
  // does; the scaled result is folded into tYr immediately after. This keeps
  // only one long-lived fp32 accumulator (tYr) alive across the entire
  // K-loop -- half the register footprint of the old tYr_high/tYr_low pair.
  auto tYr = thr_mma.partition_fragment_C(sY);  // (MMA, MMA_N, MMA_M), fp32
  // residual 累加器：FP8 后端作为逐 mma 清零的 FP32 scratch（scale 必须逐次
  // 应用，见文件头注释）；INT8 后端作为贯穿整个 K-loop 的 int32 持久累加器
  // （per-element scale 推迟到 tile 末尾一次性应用，见 Int8ResidualBackend）。
  // partition_fragment_C 的元素类型取自 MMA traits（fp8->float / int8->int32）。
  auto tYr_fp8_tmp = thr_mma_fp8.partition_fragment_C(sY);
  // INT8 折叠所需的逐 slot (col, row) 坐标：CLayout 与 BF16 MMA 完全一致
  // （SM80_16x8_Row），坐标可直接复用。constexpr 守护，FP8 路径零开销。
  auto cCoord = thr_mma.partition_C(make_identity_tensor(shape(sY)));

  G2SCopy g2s_copy;
  auto g2s_thr_copy = g2s_copy.get_slice(idx);
  G2SCopyFp8 g2s_copy_fp8;
  auto g2s_thr_copy_fp8 = g2s_copy_fp8.get_slice(idx);

  using LDSM_ATOM = Copy_Atom<SM75_U32x4_LDSM_N, Tin>;
  auto s2r_copy_a = make_tiled_copy_A(LDSM_ATOM{}, tiled_mma);
  auto s2r_copy_b = make_tiled_copy_B(LDSM_ATOM{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_copy_a.get_slice(idx);
  auto s2r_thr_copy_b = s2r_copy_b.get_slice(idx);

  // fp8 s2r ldmatrix copy for W_low/X, exactly parallel to the bf16
  // s2r_copy_a/b above -- see the A2 file-level comment for why this
  // (rather than a register-layout-based/coordinate-lookup copy) is now
  // used. LDSM_ATOM_FP8's per-thread value count (128 bits / 8 bits =
  // 16 fp8 values) differs from LDSM_ATOM's (128 bits / 16 bits = 8 bf16
  // values), but make_tiled_copy_A/B derives the correct thread/value
  // tiling from TiledMMAFp8 (whose Permutation_K = 32, twice bf16's 16)
  // automatically -- no manual layout arithmetic needed here.
  using LDSM_ATOM_FP8 = Copy_Atom<SM75_U32x4_LDSM_N, fp8e4m3>;
  auto s2r_copy_a_fp8 = make_tiled_copy_A(LDSM_ATOM_FP8{}, tiled_mma_fp8);
  auto s2r_copy_b_fp8 = make_tiled_copy_B(LDSM_ATOM_FP8{}, tiled_mma_fp8);
  auto s2r_thr_copy_a_fp8 = s2r_copy_a_fp8.get_slice(idx);
  auto s2r_thr_copy_b_fp8 = s2r_copy_b_fp8.get_slice(idx);

  int iblock = blockIdx.x;
  int last_tile_m = -1;
  int last_tile_n = -1;

  while (true) {
    auto [itile_m, itile_n, ichunk] = get_next_tile<kBlockSwizzle, kSplitK>(
        iblock, num_tile_m, num_tile_n, swizzle_divider, flat_divider);
    if (itile_m >= num_tile_m) break;
    iblock += gridDim.x;

    // Number of K-slabs owned by this CTA per output tile: chunks
    // ichunk < ntile_k_rem get the one extra (possibly-ragged) tile, see
    // the ntile_k comment above.
    int ntile_k_local = ntile_k_base + (ichunk < ntile_k_rem ? 1 : 0);

    Tensor X = make_tensor(make_gmem_ptr(x_ptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor WH = make_tensor(make_gmem_ptr(w_high_ptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor X8 = make_tensor(make_gmem_ptr(x_fp8_ptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor WL8 = make_tensor(make_gmem_ptr(w_low_fp8_ptr), make_shape(n, k), make_stride(k, Int<1>{}));

    Tensor gX = local_tile(X, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(itile_m, _));
    Tensor gWH = local_tile(WH, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(itile_n, _));
    Tensor gX8 = local_tile(X8, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(itile_m, _));
    Tensor gWL8 = local_tile(WL8, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(itile_n, _));

    auto tXg = g2s_thr_copy.partition_S(gX);    // (CPY, CPY_M, CPY_K, ntile_k)
    auto tWHg = g2s_thr_copy.partition_S(gWH);  // (CPY, CPY_N, CPY_K, ntile_k)
    auto tX8g = g2s_thr_copy_fp8.partition_S(gX8);    // (CPY8, CPY8_M, CPY8_K, ntile_k)
    auto tWL8g = g2s_thr_copy_fp8.partition_S(gWL8);  // (CPY8, CPY8_N, CPY8_K, ntile_k)

    auto tXs = g2s_thr_copy.partition_D(sX);    // (CPY, CPY_M, CPY_K, kStage)
    auto tWHs = g2s_thr_copy.partition_D(sWH);  // (CPY, CPY_N, CPY_K, kStage)
    auto tX8s = g2s_thr_copy_fp8.partition_D(sX8);    // (CPY8, CPY8_M, CPY8_K, kStage)
    auto tWL8s = g2s_thr_copy_fp8.partition_D(sWL8);  // (CPY8, CPY8_N, CPY8_K, kStage)

    // M needs a predicate (ragged rows, always). K is ragged only in its
    // very last tile (global index gk == ntile_k - 1) when k isn't an exact
    // multiple of kTileK. With split-K enabled, only the one chunk that
    // actually reaches gk == ntile_k - 1 in its interleaved sequence
    // (ichunk == ntile_k_rem - 1, see the ntile_k_local computation above)
    // will ever hit the ragged branch below; every other chunk's gk values
    // are all < ntile_k - 1 and take the plain (pred_x_full) path
    // unconditionally.
    //
    // N is ragged whenever n isn't an exact multiple of kTileN (entry.cc
    // no longer requires any n alignment at all -- see its TORCH_CHECK and
    // comment). Unlike K, N-raggedness doesn't depend on which K-slab is
    // being loaded: a given CTA's itile_n is fixed for its entire tile (the
    // whole K-loop), so the N bound folds into one bool-per-copy-unit
    // predicate computed once, outside the K-loop -- not a per-K-slab check
    // like is_ragged_gk. Only W_high/W_low need this (their gmem tensor is
    // shaped (n, k), so out-of-range rows == out-of-range N); X's gmem
    // tensor is (m, k), entirely unrelated to n.
    //
    // Only two predicate tensors are needed, not one per (K-ragged x
    // N-ragged) combination: `copy_if` with an all-true predicate costs
    // nothing extra over the unpredicated `copy` (the ZFILL copy atom's PTX
    // encodes the predicate as cp.async's %3 = pred ? 16 : 0 operand
    // regardless -- see SM80_CP_ASYNC_CACHEGLOBAL_ZFILL in
    // cute/arch/copy_sm80.hpp -- so there's no separate "predicated" vs.
    // "plain" instruction to choose between), so pred_x_ragged_k and
    // pred_w_ragged fold the M/N bounds in unconditionally and are reused
    // for every K-tile, ragged or not; only the K bound actually needs an
    // is_ragged_gk(gk) branch to pick between "all K-tiles" and "just the
    // last one" versions.
    auto tIX =
        g2s_thr_copy.partition_S(make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto tIW =
        g2s_thr_copy.partition_S(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileK>{})));
    auto pred_x_ragged_k = make_tensor<bool>(shape(tIX));   // M bound, valid for gk == ntile_k - 1
    auto pred_x_full = make_tensor<bool>(shape(tIX));       // M bound, valid for gk != ntile_k - 1
    auto pred_w_ragged_k = make_tensor<bool>(shape(tIW));   // N + K bound, for gk == ntile_k - 1
    auto pred_w = make_tensor<bool>(shape(tIW));            // N bound only, for gk != ntile_k - 1
    int k_last_tile_start = (ntile_k - 1) * kTileK;
    int n_tile_start = itile_n * kTileN;
#pragma unroll
    for (int i = 0; i < size(tIX); ++i) {
      bool in_m = itile_m * kTileM + get<0>(tIX(i)) < m;
      pred_x_full(i) = in_m;
      pred_x_ragged_k(i) = in_m && (k_last_tile_start + get<1>(tIX(i)) < k);
    }
#pragma unroll
    for (int i = 0; i < size(tIW); ++i) {
      bool in_n = n_tile_start + get<0>(tIW(i)) < n;
      pred_w(i) = in_n;
      pred_w_ragged_k(i) = in_n && (k_last_tile_start + get<1>(tIW(i)) < k);
    }

    // Same predicate construction as above, but for the fp8 g2s copy's own
    // (potentially different) thread/value tiling -- G2SCopyFp8's val count
    // per thread differs from G2SCopy's (1-byte fp8 elements vs. 2-byte
    // bf16, see launch_gemm_bf16xfp32_kernel_sm80), so it needs its own
    // identity-tensor partition, independently computed from
    // g2s_thr_copy_fp8 rather than reused from tIX/tIW above.
    auto tIX8 =
        g2s_thr_copy_fp8.partition_S(make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto tIW8 =
        g2s_thr_copy_fp8.partition_S(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileK>{})));
    auto pred_x8_ragged_k = make_tensor<bool>(shape(tIX8));
    auto pred_x8_full = make_tensor<bool>(shape(tIX8));
    auto pred_w8_ragged_k = make_tensor<bool>(shape(tIW8));
    auto pred_w8 = make_tensor<bool>(shape(tIW8));
#pragma unroll
    for (int i = 0; i < size(tIX8); ++i) {
      bool in_m = itile_m * kTileM + get<0>(tIX8(i)) < m;
      pred_x8_full(i) = in_m;
      pred_x8_ragged_k(i) = in_m && (k_last_tile_start + get<1>(tIX8(i)) < k);
    }
#pragma unroll
    for (int i = 0; i < size(tIW8); ++i) {
      bool in_n = n_tile_start + get<0>(tIW8(i)) < n;
      pred_w8(i) = in_n;
      pred_w8_ragged_k(i) = in_n && (k_last_tile_start + get<1>(tIW8(i)) < k);
    }
    // gk (the global K-tile index, interleaved by kSplitK) equals
    // ntile_k - 1 -- i.e. the ragged tile -- only for the one chunk that
    // owns it (ichunk == ntile_k_rem - 1 when ntile_k_rem > 0, see the
    // ntile_k comment above); every other chunk's gk sequence never reaches
    // ntile_k - 1, so this comparison is still correct (and cheap: a single
    // warp-uniform int compare) with split-K enabled.
    auto is_ragged_gk = [&](int gk) { return gk == ntile_k - 1; };

    clear(tYr);
    if constexpr (!HighOnly && kIsInt8) clear(tYr_fp8_tmp);  // int32 acc 按 tile 清零

    // Issues one K-slab's worth of cp.async loads (X, W_high, X_fp8,
    // W_low_fp8) into smem stage `istage`. pred_x_full/pred_w (and their
    // fp8 counterparts) already carry the M/N bounds unconditionally (see
    // the comment above on why copy_if is free when the predicate is
    // all-true), so the only per-K-slab decision left is whether this is
    // the ragged last K-tile, in which case the *_ragged_k variants
    // additionally AND in the K bound. The ZFILL copy atom (G2SCopy/
    // G2SCopyFp8 are built from SM80_CP_ASYNC_CACHEGLOBAL_ZFILL, see
    // launch_gemm_bf16xfp32_kernel_sm80) zero-fills the smem bytes whose
    // predicate is false -- required for correctness here since those
    // out-of-range M/N/K elements don't exist in gmem at all and the
    // ldmatrix/mma below has no other way to know they shouldn't
    // contribute a real value.
    auto load_slab = [&](int gk, int istage) {
      if (is_ragged_gk(gk)) {
        cute::copy_if(g2s_copy, pred_x_ragged_k, tXg(_, _, _, gk), tXs(_, _, _, istage));
        cute::copy_if(g2s_copy, pred_w_ragged_k, tWHg(_, _, _, gk), tWHs(_, _, _, istage));
        if constexpr (!HighOnly) {
          cute::copy_if(g2s_copy_fp8, pred_x8_ragged_k, tX8g(_, _, _, gk), tX8s(_, _, _, istage));
          cute::copy_if(g2s_copy_fp8, pred_w8_ragged_k, tWL8g(_, _, _, gk), tWL8s(_, _, _, istage));
        }
      } else {
        cute::copy_if(g2s_copy, pred_x_full, tXg(_, _, _, gk), tXs(_, _, _, istage));
        cute::copy_if(g2s_copy, pred_w, tWHg(_, _, _, gk), tWHs(_, _, _, istage));
        if constexpr (!HighOnly) {
          cute::copy_if(g2s_copy_fp8, pred_x8_full, tX8g(_, _, _, gk), tX8s(_, _, _, istage));
          cute::copy_if(g2s_copy_fp8, pred_w8, tWL8g(_, _, _, gk), tWL8s(_, _, _, istage));
        }
      }
    };

    // Prologue: prefetch kStage - 1 K-slabs (this CTA's own interleaved
    // slabs: global k-index = ichunk + i * kSplitK).
    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int i = 0; i < kStage - 1; ++i) {
      if (i < ntile_k_local) {
        load_slab(ichunk + i * kSplitK, ismem_write);
      }
      cp_async_fence();
      ++itile_to_read;
      ++ismem_write;
    }

    // P0b (OPTIMIZATION_PLAN.md 1.3b / P0b): software-pipelined K-loop
    // (register-level prefetch/mma interleaving only).
    //
    // The naive version of this loop did, per big K-tile: issue next
    // cp.async -> cp_async_wait -> __syncthreads() -> (s2r-copy all sub-k
    // blocks, then mma all sub-k blocks) -> __syncthreads() (again, purely
    // to protect smem from the *next* iteration's cp.async writes). That
    // put every ldmatrix (s2r copy) on the critical path right next to its
    // mma, with zero overlap between "load next sub-k-block into
    // registers" and "compute on the current one".
    //
    // This version interleaves the register-level prefetch of the next
    // sub-k-block's operands (W_low, W_high, X all loaded per `ik`) with
    // the mma for the current sub-k-block, so the ldmatrix latency for the
    // next `ik` hides behind the mma issued right before it --
    // CUTLASS's sm80_mma_multistage.hpp mainloop does the same thing at
    // the same ik granularity; only the *order* of copy-vs-mma issue
    // changes here.
    //
    // NOTE on the *other* half of the original plan (dropping the second,
    // "redundant-looking" __syncthreads() per big K-tile): this was tried
    // and reverted. cp_async_wait<N>() only tracks *this thread's own*
    // cp.async completion count; it has no idea whether *other* warps have
    // finished reading the smem buffer slot that a new cp.async is about
    // to overwrite. compute-sanitizer --tool racecheck caught real RAW
    // hazards between a warp's cp.async write (new K-tile) and another
    // warp's ldmatrix read (previous K-tile still in flight) once the
    // second __syncthreads() was removed -- see OPTIMIZATION_RESULTS.md
    // M2 for the reproduction. CUTLASS's own multistage mainloop avoids
    // this by advancing its smem_pipe_write/read indices a full iteration
    // early (right when issuing cp.async, not at the loop boundary) *and*
    // waiting on Stages-2 (not Stages-1) outstanding groups, which
    // together are enough stage-separation not to need a second sync --
    // reproducing that exact accounting is a larger, riskier restructuring
    // than the register-prefetch win justifies at kStage=2 (only one
    // prefetch stage of slack to begin with). So both __syncthreads() are
    // kept here; only the ik-level prefetch/mma interleave is applied.
    int ismem_read = 0;
    constexpr int kNumSubK = decltype(size<2>(tXr))::value;          // kTileK/16 (bf16, W_high)
    constexpr int kNumSubKFp8 = decltype(size<2>(tXr_fp8))::value;   // kTileK/32 (fp8, W_low)
    static_assert(kNumSubK == 2 * kNumSubKFp8,
                 "fp8 mma (K=32) must exactly halve the number of bf16 (K=16) sub-k steps");

    // W_high/X's bf16 s2r copy: unchanged from before the A2 change.
    auto s2r_copy_ik = [&](auto smem_view_x, auto smem_view_wh, int ik) {
      auto tXr_view = s2r_thr_copy_b.retile_D(tXr);
      auto tWHr_view = s2r_thr_copy_a.retile_D(tWHr);
      cute::copy(s2r_copy_b, smem_view_x(_, _, ik), tXr_view(_, _, ik));
      cute::copy(s2r_copy_a, smem_view_wh(_, _, ik), tWHr_view(_, _, ik));
    };
    // W_low/X's fp8 s2r copy: a real ldmatrix-based Copy_Atom copy now (see
    // the A2 file-level comment above), exactly parallel in structure to
    // s2r_copy_ik above -- just using the fp8 tiled copies/smem views
    // instead of the bf16 ones. No coordinate lookup, no NumericConverter:
    // ldmatrix reads the pre-quantized e4m3 bytes straight out of smem.
    auto s2r_copy_ik_fp8 = [&](auto smem_view_x8, auto smem_view_wl8, int ikf) {
      auto tXr_fp8_view = s2r_thr_copy_b_fp8.retile_D(tXr_fp8);
      auto tWLr_fp8_view = s2r_thr_copy_a_fp8.retile_D(tWLr_fp8);
      cute::copy(s2r_copy_b_fp8, smem_view_x8(_, _, ikf), tXr_fp8_view(_, _, ikf));
      cute::copy(s2r_copy_a_fp8, smem_view_wl8(_, _, ikf), tWLr_fp8_view(_, _, ikf));
    };

#pragma unroll 1
    for (int itile = 0; itile < ntile_k_local; ++itile) {
      if (itile_to_read < ntile_k_local) {
        int gk = ichunk + itile_to_read * kSplitK;
        load_slab(gk, ismem_write);
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();
      cp_async_wait<kStage - 1>();
      __syncthreads();

      auto tXs4r_cur = s2r_thr_copy_b.partition_S(sX(_, _, ismem_read));
      auto tWHs4r_cur = s2r_thr_copy_a.partition_S(sWH(_, _, ismem_read));
      auto tX8s4r_cur = s2r_thr_copy_b_fp8.partition_S(sX8(_, _, ismem_read));
      auto tWL8s4r_cur = s2r_thr_copy_a_fp8.partition_S(sWL8(_, _, ismem_read));

      s2r_copy_ik(tXs4r_cur, tWHs4r_cur, 0);

#pragma unroll
      for (int ik = 0; ik < kNumSubK; ++ik) {
        if (ik + 1 < kNumSubK) {
          s2r_copy_ik(tXs4r_cur, tWHs4r_cur, ik + 1);
        }
        // Interleave W_low's fp8 ldmatrix at every other bf16 ik step, so
        // its 2 fp8 sub-k steps (kNumSubKFp8 == kNumSubK/2) are spread
        // evenly across the bf16 loop instead of front-loaded -- giving
        // the warp scheduler more, smaller windows to hide the fp8
        // ldmatrix latency behind neighboring mma.sync issue (same
        // rationale as the bf16 prefetch/mma interleave above).
        if constexpr (!HighOnly) if (ik % 2 == 0) {
          int ikf = ik / 2;
          s2r_copy_ik_fp8(tX8s4r_cur, tWL8s4r_cur, ikf);
          if constexpr (!kIsInt8) clear(tYr_fp8_tmp);
          cute::gemm(tiled_mma_fp8, tWLr_fp8(_, _, ikf), tXr_fp8(_, _, ikf), tYr_fp8_tmp);
          if constexpr (!kIsInt8) {
            // FP8：逐 mma 应用统一 residualScale（不可预折叠，见文件头注释）
#pragma unroll
            for (int i = 0; i < size(tYr_fp8_tmp); ++i) {
              tYr(i) += tYr_fp8_tmp(i) * scale;
            }
          }
          // INT8：int32 精确累加贯穿 K-loop，scale 在 tile 末尾统一应用
        }

        cute::gemm(tiled_mma, tWHr(_, _, ik), tXr(_, _, ik), tYr);
      }

      ismem_read = (ismem_read + 1) % kStage;
      __syncthreads();  // smem buffers about to be overwritten by next iter's cp.async
    }

    if constexpr (!HighOnly && kIsInt8) {
      // INT8 后端的 residual 合并（Int8ResidualBackend 注释）：
      //   Y_low[m,n] = residualScale * s_x[m] * s_w[n] * acc_int8[m,n]
      // 逐 slot 从 cCoord 取 (col, row) 坐标；越界行/列 scale 取 0（对应的
      // 输出本来就不会写回，但 gmem 读必须防越界）。
      int m0 = itile_m * kTileM;
      int n0 = itile_n * kTileN;
#pragma unroll
      for (int i = 0; i < size(tYr); ++i) {
        auto coord = cCoord(i);
        int row = m0 + get<1>(coord);
        int col = n0 + get<0>(coord);
        float sx = (row < m) ? x_scale_ptr[row] : 0.0f;
        float sw = (col < n) ? w_scale_ptr[col] : 0.0f;
        tYr(i) += static_cast<float>(tYr_fp8_tmp(i)) * (scale * sx * sw);
      }
    }

    // Epilogue: native fp32 accumulator -> swizzled fp32 smem -> gmem.
    // This follows CUTLASS's epilogue convention of using ElementAccumulator
    // as shared storage. Conversion to TY (bf16 for the ordinary output,
    // float for split-K partials/fp32 output) happens only at the final gmem
    // assignment below.
    using R2SCopyAtomY = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy_y = make_tiled_copy_C(R2SCopyAtomY{}, tiled_mma);
    auto thr_copy_y = tiled_copy_y.get_slice(idx);
    cute::copy(tiled_copy_y, thr_copy_y.retile_S(tYr), thr_copy_y.partition_D(sY));
    __syncthreads();

    TY *y_dst_base = (kSplitK > 1)
                         ? reinterpret_cast<TY *>(splitk_y_ptr) + static_cast<int64_t>(ichunk) * m * n
                         : reinterpret_cast<TY *>(y_ptr);
    Tensor gYY = make_tensor(make_gmem_ptr(y_dst_base), make_shape(n, m), make_stride(Int<1>{}, n));
    Tensor gY =
        local_tile(gYY, make_tile(Int<kTileN>{}, Int<kTileM>{}), make_coord(itile_n, itile_m));

#pragma unroll 1
    for (int i = idx; i < kTileM * kTileN; i += kNumThreads) {
      int row = i / kTileN;
      int col = i % kTileN;
      if (itile_m * kTileM + row < m && itile_n * kTileN + col < n) {
        if constexpr (kSplitK > 1) {
          // Split-K partials must remain linear; activation is applied once,
          // after all FP32 partial sums have been reduced.
          gY(col, row) = static_cast<TY>(sY(col, row));
        } else {
          float value = sY(col, row);
          if constexpr (HasBias) value += bias_ptr[itile_n * kTileN + col];
          gY(col, row) = static_cast<TY>(Activation{}(value));
        }
      }
    }

    if constexpr (kSplitK > 1) {
      __syncthreads();  // ensure this tile's gY writes are visible before flagging done
      if (idx == 0) {
        if (last_tile_m != -1 && last_tile_n != -1) {
          atomicAdd(split_flag_ptr + last_tile_m * num_tile_n + last_tile_n, 1);
        }
        last_tile_m = itile_m;
        last_tile_n = itile_n;
      }
    }
    __syncthreads();
  }

  if constexpr (kSplitK > 1) {
    __threadfence();
    __syncthreads();

    if (idx == 0 && last_tile_m != -1 && last_tile_n != -1) {
      atomicAdd(split_flag_ptr + last_tile_m * num_tile_n + last_tile_n, 1);
    }
    __syncthreads();

    iblock = blockIdx.x;
    __threadfence();
    using NVTout = std::conditional_t<std::is_same_v<Tout, float>, float, __nv_bfloat16>;
    while (true) {
      int itile_m, itile_n;
      reduce_flat_divider(itile_m, itile_n, iblock);
      if (itile_m >= num_tile_m) break;
      iblock += gridDim.x;

      auto *split_flag = split_flag_ptr + itile_m * num_tile_n + itile_n;
      while (load_global_volatile(split_flag) != kSplitK) {
      }
      splitk_reduce<NVTout, Activation, HasBias, kTileM, kTileN, kSplitK,
                    kNumThreads / 32>(reinterpret_cast<NVTout *>(y_ptr),
                                      splitk_y_ptr, bias_ptr, m, n,
                                      itile_m, itile_n);
      __syncthreads();
      if (idx == 0) {
        *split_flag = 0;
      }
    }
  }
}

}  // namespace kernels

bool convert_x_fp32_to_bf16(const float *x_fp32_ptr, void *x_bf16_ptr,
                            int64_t numel, cudaStream_t stream) noexcept {
  if (x_fp32_ptr == nullptr || x_bf16_ptr == nullptr || numel <= 0) return false;
  constexpr int kThreads = 256;
  int grid = static_cast<int>(std::min<int64_t>((numel + kThreads - 1) / kThreads, 65535));
  kernels::convert_x_fp32_to_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      x_fp32_ptr, reinterpret_cast<cute::bfloat16_t *>(x_bf16_ptr), numel);
  return cudaPeekAtLastError() == cudaSuccess;
}

#if MIXED_GEMM_FP8_ENABLED
bool convert_x_fp32_to_bf16_fp8(const float *x_fp32_ptr, void *x_bf16_ptr,
                                void *x_fp8_ptr, int64_t numel,
                                cudaStream_t stream) noexcept {
  if (x_fp32_ptr == nullptr || x_bf16_ptr == nullptr || x_fp8_ptr == nullptr || numel <= 0) {
    return false;
  }
  constexpr int kThreads = 256;
  constexpr int kVec = 4;
  int64_t vector_count = (numel + kVec - 1) / kVec;
  int64_t grid64 = (vector_count + kThreads - 1) / kThreads;
  int grid = static_cast<int>(std::min<int64_t>(grid64, 65535));
  kernels::convert_x_fp32_to_bf16_fp8_kernel<<<grid, kThreads, 0, stream>>>(
      x_fp32_ptr, reinterpret_cast<cute::bfloat16_t *>(x_bf16_ptr),
      reinterpret_cast<cute::float_e4m3_t *>(x_fp8_ptr), numel);
  return cudaPeekAtLastError() == cudaSuccess;
}

// bf16 输入版：x 本身就是主项 GEMM 的输入，只需量化出 FP8 residual 副本。
bool quantize_x_bf16_to_fp8(const void *x_bf16_ptr, void *x_fp8_ptr, int64_t numel,
                            cudaStream_t stream) noexcept {
  if (x_bf16_ptr == nullptr || x_fp8_ptr == nullptr || numel <= 0) return false;
  constexpr int kThreads = 256;
  constexpr int kVec = 8;
  int64_t vector_count = (numel + kVec - 1) / kVec;
  int64_t grid64 = (vector_count + kThreads - 1) / kThreads;
  int grid = static_cast<int>(std::min<int64_t>(grid64, 65535));
  kernels::quantize_to_fp8_kernel<cute::bfloat16_t><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const cute::bfloat16_t *>(x_bf16_ptr),
      reinterpret_cast<cute::float_e4m3_t *>(x_fp8_ptr), numel);
  return cudaPeekAtLastError() == cudaSuccess;
}
#endif  // MIXED_GEMM_FP8_ENABLED

bool quantize_x_fp32_to_int8(const float *x_fp32_ptr, void *x_bf16_ptr,
                             void *x_int8_ptr, float *x_scale_ptr, int m,
                             int64_t k, cudaStream_t stream) noexcept {
  if (x_fp32_ptr == nullptr || x_bf16_ptr == nullptr || x_int8_ptr == nullptr ||
      x_scale_ptr == nullptr || m <= 0 || k <= 0) {
    return false;
  }
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  int grid = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  kernels::quantize_x_fp32_to_int8_kernel<<<grid, kThreads, 0, stream>>>(
      x_fp32_ptr, reinterpret_cast<cute::bfloat16_t *>(x_bf16_ptr),
      reinterpret_cast<int8_t *>(x_int8_ptr), x_scale_ptr, m, k);
  return cudaPeekAtLastError() == cudaSuccess;
}

// bf16 输入版：x 已是 BF16，只量化出 int8 residual 副本 + per-row scale。
bool quantize_x_bf16_to_int8(const void *x_bf16_ptr, void *x_int8_ptr,
                             float *x_scale_ptr, int m, int64_t k,
                             cudaStream_t stream) noexcept {
  if (x_bf16_ptr == nullptr || x_int8_ptr == nullptr || x_scale_ptr == nullptr ||
      m <= 0 || k <= 0) {
    return false;
  }
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  int grid = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
  kernels::quantize_x_bf16_to_int8_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const cute::bfloat16_t *>(x_bf16_ptr),
      reinterpret_cast<int8_t *>(x_int8_ptr), x_scale_ptr, m, k);
  return cudaPeekAtLastError() == cudaSuccess;
}

// FP8 residual 后端是否编入了本 .so（CUDA >= 12.4 编译时为 true）。
// 与 GPU 是否支持 SM89 FP8 MMA 无关——那由调用方在运行时另行检查。
bool mixed_gemm_fp8_compiled() noexcept { return MIXED_GEMM_FP8_ENABLED != 0; }

template <typename Tin, typename Tout, typename Activation, bool HasBias, bool HighOnly,
          int kTileM, int kTileN, int kTileK, int kStage, int kSplitK,
          typename ResidualBackend>
void launch_gemm_bf16xfp32_kernel_sm80(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
                                       const void *x_ptr, const void *w_high_ptr,
                                       const void *w_low_fp8_ptr, const float *bias_ptr,
                                       void *x_fp8_ptr,
                                       int m, int n, int k, float scale, int sm_count,
                                       cudaStream_t stream,
                                       const float *w_scale_ptr = nullptr,
                                       float *x_scale_ptr = nullptr) {
  using namespace cute;  // NOLINT
  using fp8e4m3 = typename ResidualBackend::Element;  // FP8: e4m3 / INT8: int8_t

  constexpr int kBlockSwizzle = 4;
  using TY = std::conditional_t<(kSplitK > 1), float, Tout>;

  // 16x8x16 bf16 mma.sync atom, tiled over 8 warps in a (2, 4, 1) AtomLayoutMNK
  // (2 warps along the weight/N dim, 4 warps along the activation/M dim),
  // with an explicit Permutation tile of (32, 64, 16). Permutation_X is the
  // *total* MNK extent covered by one "thrfrg" unit (thr_layout_X * atom_X
  // when the atom itself isn't repeated further), so here:
  //   M: 2 warps * 16 (atom M) = 32   -> kTileM/32 Rest repeats
  //   N: 4 warps * 16           = 64  -> kTileN/64 Rest repeats (atom N=8,
  //                                      so each warp covers N=16 via 2 atom
  //                                      reps before the 4-warp tiling)
  //   K: 1 * 16 (atom K)        = 16  -> kTileK/16 Rest repeats
  // This exact (thr_layout, permutation) ratio -- Permutation_{M,N} = 16 *
  // thr_layout_{M,N} -- is what keeps make_tiled_copy_A/B's static_assert
  // (TiledCopy val count vs. SM75_U32x4_LDSM_N's per-thread val count) happy;
  // shrinking Permutation_N down to 32 (i.e. 1:1 with thr_layout) breaks the
  // ldmatrix.x4 tiling for the B (activation) operand.
  using MMA_ATOM = SM80_16x8x16_F32BF16BF16F32_TN;
  static_assert(kTileN % 64 == 0 && kTileM % 32 == 0 && kTileK % 16 == 0,
               "kTileN must be a multiple of 64, kTileM of 32, kTileK of 16");
  using MMAThrLayout = decltype(make_layout(make_shape(Int<2>{}, Int<4>{}, Int<1>{})));
  using MMAPermutation = Tile<Int<32>, Int<64>, Int<16>>;
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_ATOM>{}, MMAThrLayout{}, MMAPermutation{});

  // W_low-in-fp8 (see the kernel-level comment in gemm_bf16xfp32_kernel):
  // SM89_16x8x32_F32E4M3E4M3F32_TN is Ada's native fp8 (e4m3) tensor-core
  // mma.sync -- same M=16/N=8 as the bf16 atom above, but K=32 (2x the
  // bf16 atom's K=16). Reusing the *exact same* 8-warp MMAThrLayout keeps
  // both mmas' CTA-tile-to-warp assignment identical (each warp still owns
  // the same (M,N) output sub-tile for both W_high's and W_low's
  // contribution), so only Permutation_K needs to change, to fp8's native
  // K=32 -- Permutation_M/N stay bit-for-bit the same as MMAPermutation
  // above. kTileK == 64 (static_assert above) is exactly 2x fp8's K=32, so
  // one K-slab maps to exactly 2 fp8 mma issues (kNumSubKFp8 == 2), vs. 4
  // bf16 mma issues (kNumSubK == 4) for the same slab -- see kNumSubK/
  // kNumSubKFp8 in gemm_bf16xfp32_kernel.
  // residual MMA atom：FP8 为 SM89_16x8x32_F32E4M3E4M3F32_TN，INT8 为
  // SM80_16x8x32_S32S8S8S32_TN（两者 A/B/C fragment 布局逐位一致）。
  using MMA_ATOM_FP8 = typename ResidualBackend::MmaAtom;
  using MMAPermutationFp8 = Tile<Int<32>, Int<64>, Int<32>>;
  auto tiled_mma_fp8 = make_tiled_mma(MMA_Atom<MMA_ATOM_FP8>{}, MMAThrLayout{}, MMAPermutationFp8{});

  // K-major, 128B-swizzled smem layout (kTileK * sizeof(bf16) == 128B when
  // kTileK == 64); this is the classic Ampere ldmatrix-friendly swizzle atom.
  static_assert(kTileK == 64, "sm80 kernel assumes kTileK == 64 for the 128B swizzle atom");
  using SwizzleAtomK =
      decltype(composition(Swizzle<3, 3, 3>{},
                           make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                       make_stride(Int<kTileK>{}, Int<1>{}))));

  auto slayout_x =
      tile_to_shape(SwizzleAtomK{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_w =
      tile_to_shape(SwizzleAtomK{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));

  // Epilogue accumulator layout. make_tiled_copy_C maps each warp's first
  // scalar store as:
  //   col = lane / 4, row = 2 * (lane % 4)
  // so the plain column-major offset col + row*kTileN maps four lanes per
  // column onto the same bank whenever kTileN*sizeof(float) is a multiple
  // of the 128-byte bank period. In CUTE's Swizzle<B,M,S> element-offset
  // notation, preserve col's three low bits (M=3), and XOR the two row-group
  // bits (B=2) into the next two bank bits. Because row=2*q, those source
  // bits begin at offset bit log2(kTileN)+1, hence S=5 for kTileN=128 and
  // S=4 for kTileN=64. The resulting bank index is col + 8*q: all 32 lanes
  // hit distinct banks. This derivation is specific to the scalar C-fragment
  // store and intentionally differs from the 128-bit X/W ldmatrix swizzles.
  static_assert(kTileN == 64 || kTileN == 128, "epilogue swizzle is derived for 64/128-wide tiles");
  auto slayout_y_base =
      make_layout(make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<1>{}, Int<kTileN>{}));
  auto slayout_y = [&]() {
    if constexpr (kTileN == 128) {
      return composition(Swizzle<2, 3, 5>{}, slayout_y_base);
    } else {
      return composition(Swizzle<2, 3, 4>{}, slayout_y_base);
    }
  }();

  // fp8 (e4m3, 1-byte) analogue of SwizzleAtomK above, for the pre-quantized
  // X_fp8/W_low_fp8 smem tiles (see the A2 file-level comment). Same
  // (8, kTileK) base shape and 128-bit (16-byte) ldmatrix access granule as
  // the bf16 atom, but re-derived for 1-byte elements:
  //   MBase  = log2(16 bytes / sizeof(fp8) = 16)          = 4
  //   SShift = log2(kTileK / (16 bytes / sizeof(fp8)))
  //          = log2(64 / 16) = log2(4)                    = 2
  //   BBits  = min(log2(rows=8)=3, SShift=2)               = 2
  //          (capped at SShift: Swizzle<B,M,S> requires abs(S) >= B, and
  //          fp8's 16-byte chunks only divide each 64-element/64-byte row
  //          into 4 chunks -- vs. bf16's 8 chunks per 128-byte row -- so
  //          there are only 2 bits' worth of distinguishable chunk-columns
  //          to permute among here, not 3).
  // This is exactly cutlass's "B64" (64-byte-period) K-major swizzle atom
  // in element (rather than bit) units -- see e.g.
  // cute::GMMA::Layout_K_SW64_Atom_Bits = Swizzle<2,4,3> on a (8,512)-bit
  // base layout; upcast<8>(Swizzle<2,4,3>) = Swizzle<2, 4-log2(8), 3> =
  // Swizzle<2,1,3> would be the *bit-shift-preserving* GMMA-descriptor
  // convention, but that convention is specific to how GMMA descriptors
  // encode strides and doesn't apply here -- this kernel's swizzle
  // composes directly with an element-indexed Layout (see SwizzleAtomK
  // above, which is similarly *not* derived via upcast from a GMMA atom),
  // so SShift must independently equal log2(elements-per-row /
  // elements-per-chunk) = 2 in *this* (non-GMMA) convention, not 3.
  using SwizzleAtomKFp8 =
      decltype(composition(Swizzle<2, 4, 2>{},
                           make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                       make_stride(Int<kTileK>{}, Int<1>{}))));
  auto slayout_x8 =
      tile_to_shape(SwizzleAtomKFp8{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{}));
  auto slayout_wl8 =
      tile_to_shape(SwizzleAtomKFp8{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}));

  // shm layout: [sX (bf16) | sWH (bf16) | sX8 (fp8) | sWL8 (fp8)] (see the
  // A2 file-level comment / gemm_bf16xfp32_kernel's shm_data layout
  // comment) -- W_low's bf16 smem buffer is gone (only its fp8 tile is
  // staged now), replaced by the (much smaller, 1-byte-element) X_fp8/
  // W_low_fp8 tiles.
  int shm_xw = sizeof(Tin) * (cosize(slayout_x) + cosize(slayout_w));
  if constexpr (!HighOnly) {
    shm_xw += sizeof(fp8e4m3) * (cosize(slayout_x8) + cosize(slayout_wl8));
  }
  int shm_y = sizeof(float) * cosize(slayout_y);
  int shm_size = std::max(shm_xw, shm_y);

  // cp.async g2s copy: 128-bit vectorized; X, W_high tiles have shape
  // (rows, kTileK) so they share one tiled-copy definition.
  constexpr int kNumThreads = decltype(size(tiled_mma))::value;
  constexpr int kElemPerLoad = 16 / sizeof(Tin);         // 8 bf16 elements per 128-bit load
  constexpr int kThreadsPerRow = kTileK / kElemPerLoad;  // threads needed to cover one K-row
  static_assert(kNumThreads % kThreadsPerRow == 0, "thread count must divide evenly");
  constexpr int kRowsPerIter = kNumThreads / kThreadsPerRow;

  // _ZFILL (vs. plain SM80_CP_ASYNC_CACHEGLOBAL) is required for ragged-K
  // support (K need not be a multiple of kTileK, see the K-not-a-multiple-
  // of-64 removal in OPTIMIZATION_PLAN.md / entry.cc): its underlying PTX
  // (cp.async.cg ... %2, %3, with %3 = pred ? 16 : 0) zero-fills the smem
  // destination bytes whenever the predicate is false, instead of just
  // skipping the copy and leaving smem stale -- exactly what's needed for
  // the last, partial K-tile's out-of-range columns (see gk ==
  // ntile_k - 1 handling / load_slab in gemm_bf16xfp32_kernel below). For
  // an all-true predicate (every non-ragged K-tile) this degenerates to
  // the exact same cp.async.cg PTX as the plain atom, so there's no
  // overhead for the common (non-ragged) case.
  using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Tin>;
  using G2SCopy = decltype(make_tiled_copy(
      G2SCopyAtom{},
      make_layout(make_shape(Int<kRowsPerIter>{}, Int<kThreadsPerRow>{}),
                 make_stride(Int<kThreadsPerRow>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<kElemPerLoad>{}))));

  // fp8 analogue of G2SCopy for X_fp8/W_low_fp8. Deliberately 64-bit (8
  // fp8 elements per load), NOT 128-bit (16 elements) like the bf16 copy
  // above, even though the fp8 *smem* destination tile is always
  // 128-bit-alignable (smem tile width == kTileK == 64 fp8 bytes, a
  // multiple of 16): the *source* is gmem x_fp8_ptr/w_low_fp8_ptr, a
  // plain row-major (rows, k) e4m3 buffer (1 byte/elem) written by
  // quantize_to_fp8_kernel, whose row stride is exactly `k` bytes.
  // entry.cc only guarantees k % 8 == 0 (needed for the bf16 operands'
  // 128-bit/16-byte alignment, since bf16 is 2 bytes/elem: k%8==0 =>
  // k*2%16==0) -- for a 1-byte-per-elem fp8 tensor that same k%8==0 only
  // guarantees 8-byte alignment, not 16. A 128-bit fp8 vectorized load
  // would misalign (cudaErrorMisalignedAddress) on every row after the
  // first whenever k is a multiple of 8 but not 16 (e.g. k=1944).
  // Matching bf16's actual alignment guarantee (8 bytes) here, instead of
  // assuming the stronger 16-byte bound, keeps every k accepted by
  // entry.cc's existing check safe for both operand paths without
  // tightening that public contract.
  constexpr int kElemPerLoadFp8 = 8 / sizeof(fp8e4m3);          // 8 fp8 elements per 64-bit load
  constexpr int kThreadsPerRowFp8 = kTileK / kElemPerLoadFp8;   // threads needed to cover one K-row
  static_assert(kNumThreads % kThreadsPerRowFp8 == 0, "thread count must divide evenly");
  constexpr int kRowsPerIterFp8 = kNumThreads / kThreadsPerRowFp8;
  using G2SCopyAtomFp8 = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cute::uint64_t>, fp8e4m3>;
  using G2SCopyFp8 = decltype(make_tiled_copy(
      G2SCopyAtomFp8{},
      make_layout(make_shape(Int<kRowsPerIterFp8>{}, Int<kThreadsPerRowFp8>{}),
                 make_stride(Int<kThreadsPerRowFp8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<kElemPerLoadFp8>{}))));

  auto kernel =
      kernels::gemm_bf16xfp32_kernel<Tin, TY, Tout, decltype(tiled_mma), decltype(tiled_mma_fp8),
                                     G2SCopy, G2SCopyFp8, decltype(slayout_x), decltype(slayout_w),
                                     decltype(slayout_x8), decltype(slayout_wl8),
                                     decltype(slayout_y), kTileM, kTileN, kTileK, kStage,
                                     kBlockSwizzle, kSplitK, kNumThreads, Activation, HasBias,
                                     ResidualBackend, HighOnly>;
  // Kernel attributes and occupancy depend only on this compile-time
  // specialization and target device. Cache them outside the inference hot path.
  static std::once_flag metadata_once;
  static int cached_max_active_blocks_per_sm = 1;
  std::call_once(metadata_once, [&]() {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &cached_max_active_blocks_per_sm, kernel, kNumThreads, shm_size);
    if (cached_max_active_blocks_per_sm < 1) cached_max_active_blocks_per_sm = 1;
  });

  int num_tile_m = (m + kTileM - 1) / kTileM;
  int num_tile_n = (n + kTileN - 1) / kTileN * kSplitK;
  int num_tile = num_tile_m * num_tile_n;
  int num_tile_bxn = kBlockSwizzle * num_tile_n;
  cutlass::FastDivmod swizzle_divider(num_tile_bxn);
  cutlass::FastDivmod flat_divider(num_tile_n);
  cutlass::FastDivmod reduce_flat_divider(num_tile_n / kSplitK);

  // P6 (multi-block-per-SM persistent grid): this is a *persistent* kernel
  // -- each CTA loops internally (get_next_tile + `iblock += gridDim.x`)
  // over as many output tiles as needed, so the grid size need not (and,
  // it turns out, *should* not) be capped at exactly one CTA per SM.
  //
  // Previously grid was hardcoded to min(sm_count, num_tile): exactly one
  // persistent CTA per SM, forever, regardless of how many CTAs the
  // kernel's actual register/smem footprint would allow to co-reside.
  // Measured with ncu on this kernel's heaviest configs (e.g. 128x128 at
  // m=1331,n=2304,k=768): 250 regs/thread * 256 threads/block = 64,000
  // regs, just under sm89's 65,536/SM budget -- Achieved Occupancy 16.7%
  // (8 warps/SM, only 2 active warps/scheduler out of a possible 12) with
  // "Block Limit Registers/Shared Mem" = 1, so only 1 block/SM was even
  // *possible* there. But at 64x64 (e.g. the m=2042,n=2042,k=768 case),
  // the smaller tile's smaller X/W_high/W_low smem+register footprint
  // raises Block Limit Registers/Shared Mem to 2 -- yet Achieved Occupancy
  // was *still* only 16.7% (measured), because the hardcoded grid size
  // meant only 1 CTA was ever launched per SM in the first place: the
  // *scheduler* never got a second CTA to fill that available slot with,
  // even though the hardware had room for one. Every one of this kernel's
  // configs is latency-bound, not occupancy-by-design-bound: ncu's warp
  // state statistics show ~44-46% of each warp's issue-to-issue cycles
  // spent stalled on "execution pipe busy" (the tensor core pipeline
  // oversubscribed by too few concurrently-schedulable warps) -- exactly
  // the kind of stall that a second concurrently-resident CTA's warps
  // would help hide, by giving the SM's warp schedulers alternative
  // eligible warps to issue while the first CTA's warps wait on their own
  // mma.sync latency.
  //
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor asks the driver exactly
  // how many CTAs of *this* kernel (with its actual register count and
  // the dynamic smem size just configured above) can simultaneously
  // reside on one SM -- the same number ncu's "Block Limit Registers"/
  // "Block Limit Shared Mem" report, computed the same way the hardware
  // scheduler itself will apply it at launch time. Multiplying that into
  // the grid size (still capped at num_tile, so small problems don't
  // over-launch empty persistent CTAs) lets the scheduler actually place
  // that many CTAs per SM instead of leaving the extra slots empty.
int max_active_blocks_per_sm = cached_max_active_blocks_per_sm;

dim3 block(kNumThreads);
dim3 grid(std::min(sm_count * max_active_blocks_per_sm, num_tile));
  if (std::getenv("GEMM_MIXED_DEBUG_GRID")) {
    fprintf(stderr,
            "[mixed_gemm debug] kTileM=%d kTileN=%d kSplitK=%d shm_size=%d max_active_blocks_per_sm=%d "
            "num_tile=%d grid=%d\n",
            kTileM, kTileN, kSplitK, static_cast<int>(shm_size), max_active_blocks_per_sm, num_tile,
            grid.x);
  }

  kernel<<<grid, block, shm_size, stream>>>(
      reinterpret_cast<const Tin *>(x_ptr), reinterpret_cast<const Tin *>(w_high_ptr),
      reinterpret_cast<const fp8e4m3 *>(x_fp8_ptr), reinterpret_cast<const fp8e4m3 *>(w_low_fp8_ptr),
      x_scale_ptr, w_scale_ptr,
      bias_ptr, reinterpret_cast<Tout *>(y_ptr), reinterpret_cast<float *>(splitk_y_ptr),
      reinterpret_cast<int *>(split_flag_ptr), m, n, k, scale, swizzle_divider, flat_divider,
      reduce_flat_divider);
}

template <int kTileM_, int kTileN_, int kTileK_, int kStage_, int kSplitK_>
struct LaunchCfgSm80 {
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  static constexpr int kSplitK = kSplitK_;
};

// P0 (OPTIMIZATION_PLAN.md 1.1 / P0): narrow-M CTA tile for small-M
// workloads.
//
// With the fixed kTileM=128 tile, an m<=64 GEMM only ever fills 64 (or
// fewer) of the 128 rows every CTA computes MMAs for -- the `pred_x`
// predicate in gemm_bf16xfp32_kernel only gates the *global-memory load* of
// out-of-range rows, not the `mma.sync` issued for them, so >=50% of every
// Tensor Core cycle is wasted on rows that don't exist. Halving kTileM to
// 64 for these shapes directly halves that waste (down to 0% waste at
// m=64, 50% at m<=32 -- kTileM=32 would help m<=32 further and needs no
// MMAThrLayout/Permutation change (Permutation_M is already 32), but was
// implemented and A/B measured with *zero* throughput gain: the CTA's
// shared-memory traffic is dominated by the W_high/W_low loads
// (kTileN*kTileK*2*sizeof(bf16) = 32KB, independent of kTileM), not the X
// load (kTileM*kTileK*sizeof(bf16), just 4-16KB depending on kTileM) --
// so shrinking kTileM only reduces the minority operand's cost while the
// majority-share weight cp.async time, which sets the critical path,
// stays exactly the same. m=16 and m=32 measured *identical* wall-clock
// time under both kTileM=32 and kTileM=64 for every shape tried. See
// gemm_bf16xfp32_sm80.h's select_kTileM comment for the fuller writeup;
// not implemented here since it would only add kernel-instantiation/binary
// size cost for no measured benefit).
//
// kTileM=64 keeps the exact same MMAThrLayout=(2,4,1)/Permutation=(32,64,16)
// as kTileM=128 (Permutation_M=32 already divides 64 evenly, i.e. just 2
// repeats instead of 4), so no MMA-tiling/TiledCopy static_assert changes
// are needed -- this is the "recommended, minimal-change" option from the
// plan.
//
// (A P2 experiment -- shrinking kTileN to 64 in exchange for a deeper
// kStage=3/4 pipeline on the narrow-M path -- was tried and measured with a
// careful A/B benchmark; see OPTIMIZATION_RESULTS.md's P2 section. It
// showed no real throughput gain once GPU-clock noise was controlled for,
// so it was removed again rather than kept as a dead opt-in path.)
template <typename Activation, bool HasBias, bool HighOnly = false,
          typename ResidualBackend>
bool dispatch_gemm_bf16xfp32_activation(void *y_ptr, void *splitk_y_ptr, void *split_flag_ptr,
                                         const void *x_ptr, const void *w_high_ptr,
                                         const void *w_low_fp8_ptr, const float *bias_ptr,
                                         void *x_fp8_ptr,
                                         int m, int n, int k, float scale, bool use_fp32_output,
                                         int splitk, int sm_count, cudaStream_t stream,
                                         const float *w_scale_ptr = nullptr,
                                         float *x_scale_ptr = nullptr) {
  using bf16 = cute::bfloat16_t;

  auto launch = [&](auto cfg_tag) {
    using Cfg = decltype(cfg_tag);
    if (use_fp32_output) {
      launch_gemm_bf16xfp32_kernel_sm80<bf16, float, Activation, HasBias, HighOnly,
                                        Cfg::kTileM, Cfg::kTileN, Cfg::kTileK,
                                        Cfg::kStage, Cfg::kSplitK, ResidualBackend>(
          y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_fp8_ptr,
          bias_ptr, x_fp8_ptr, m, n, k, scale, sm_count, stream,
          w_scale_ptr, x_scale_ptr);
    } else {
      launch_gemm_bf16xfp32_kernel_sm80<bf16, bf16, Activation, HasBias, HighOnly,
                                        Cfg::kTileM, Cfg::kTileN, Cfg::kTileK,
                                        Cfg::kStage, Cfg::kSplitK, ResidualBackend>(
          y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr, w_low_fp8_ptr,
          bias_ptr, x_fp8_ptr, m, n, k, scale, sm_count, stream,
          w_scale_ptr, x_scale_ptr);
    }
  };

  // Stage count is capped at 2 (rather than sm90's 4+) because Ada (sm89)
  // only has 99KB of opt-in dynamic shared memory per block (vs. 227KB on
  // Hopper): each pipeline stage holds X + W_high + W_low tiles of
  // (kTileM*64 + 2*kTileN*64) * sizeof(bf16), so kStage=2 -> 96KB at
  // kTileM=128,kTileN=128, right at the sm89 limit, while kStage=3 (144KB)
  // would exceed it. kTileM=64 or kTileN=64 halves the corresponding
  // portion, but kStage=2 is kept for all configs for consistency.
  //
  // Dispatch on m: <=64 uses the narrow 64x128x64 tile (see rationale
  // above); everything else picks between 128x128x64 (default) and
  // 64x64x64 (P5/P6, see select_kTileN's comment in gemm_bf16xfp32_sm80.h)
  // by how wave-quantized *and* how occupancy-limited the 128x128 grid
  // would be for this (m, n).
  if (m <= 64) {
    switch (splitk) {
      case 16:
        launch(LaunchCfgSm80<64, 128, 64, 2, 16>{});
        return true;
      case 8:
        launch(LaunchCfgSm80<64, 128, 64, 2, 8>{});
        return true;
      case 4:
        launch(LaunchCfgSm80<64, 128, 64, 2, 4>{});
        return true;
      case 2:
        launch(LaunchCfgSm80<64, 128, 64, 2, 2>{});
        return true;
      case 1:
        launch(LaunchCfgSm80<64, 128, 64, 2, 1>{});
        return true;
      default:
        return false;
    }
  }

  // select_kTileN and select_kTileMForN always move together (128x128 or
  // 64x64, never mixed -- see select_kTileN's comment in
  // gemm_bf16xfp32_sm80.h), so branching on kTileN_sel alone is sufficient.
  int kTileN_sel = select_kTileN(m, n, sm_count);

  if (kTileN_sel == 64) {
    // P5/P6: 64x64 square tile (wave-quantized and/or occupancy-limited at
    // 128x128 -- see select_kTileN's comment in gemm_bf16xfp32_sm80.h)
    switch (splitk) {
      case 16:
        launch(LaunchCfgSm80<64, 64, 64, 2, 16>{});
        return true;
      case 8:
        launch(LaunchCfgSm80<64, 64, 64, 2, 8>{});
        return true;
      case 4:
        launch(LaunchCfgSm80<64, 64, 64, 2, 4>{});
        return true;
      case 2:
        launch(LaunchCfgSm80<64, 64, 64, 2, 2>{});
        return true;
      case 1:
        launch(LaunchCfgSm80<64, 64, 64, 2, 1>{});
        return true;
      default:
        return false;
    }
  }

  switch (splitk) {
    case 16:
      launch(LaunchCfgSm80<128, 128, 64, 2, 16>{});
      return true;
    case 8:
      launch(LaunchCfgSm80<128, 128, 64, 2, 8>{});
      return true;
    case 4:
      launch(LaunchCfgSm80<128, 128, 64, 2, 4>{});
      return true;
    case 2:
      launch(LaunchCfgSm80<128, 128, 64, 2, 2>{});
      return true;
    case 1:
      launch(LaunchCfgSm80<128, 128, 64, 2, 1>{});
      return true;
    default:
      return false;
  }
}

#if MIXED_GEMM_FP8_ENABLED
template <typename Activation, bool HasBias>
bool launch_fixed_epilogue(void *y_ptr, void *splitk_y_ptr,
                           void *split_flag_ptr, const void *x_ptr,
                           const void *w_high_ptr,
                           const void *w_low_fp8_ptr,
                           const float *bias_ptr, void *x_fp8_ptr,
                           int m, int n, int k, float scale,
                           bool use_fp32_output, int splitk, int sm_count,
                           cudaStream_t stream) {
  return dispatch_gemm_bf16xfp32_activation<Activation, HasBias, false,
                                            kernels::Fp8ResidualBackend>(
      y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr,
      w_low_fp8_ptr, bias_ptr, x_fp8_ptr, m, n, k, scale,
      use_fp32_output, splitk, sm_count, stream);
}

GemmFixedEpilogueLauncher resolve_gemm_bf16xfp32_epilogue_launcher(
    int activation_type, bool has_bias) noexcept {
  switch (activation_type) {
    case 0:
      return has_bias ? &launch_fixed_epilogue<IdentityActivation, true>
                      : &launch_fixed_epilogue<IdentityActivation, false>;
    case 1:
      return has_bias ? &launch_fixed_epilogue<SiluActivation, true>
                      : &launch_fixed_epilogue<SiluActivation, false>;
    case 2:
      return has_bias ? &launch_fixed_epilogue<GeluActivation, true>
                      : &launch_fixed_epilogue<GeluActivation, false>;
    default:
      return nullptr;
  }
}
#endif  // MIXED_GEMM_FP8_ENABLED

// INT8 后端的 launcher（与 FP8 的 GemmFixedEpilogueLauncher 平行）：
// 额外携带 per-channel weight scale（w_scale_ptr，插件常量输入）和运行时
// 生成的 per-row activation scale（x_scale_ptr，workspace）。x_fp8_ptr 槽位
// 承载 int8 activation workspace（与 fp8 同为 1 字节元素）。
template <typename Activation, bool HasBias>
bool launch_fixed_epilogue_int8(void *y_ptr, void *splitk_y_ptr,
                                void *split_flag_ptr, const void *x_ptr,
                                const void *w_high_ptr,
                                const void *w_low_int8_ptr,
                                const float *w_scale_ptr,
                                const float *bias_ptr, void *x_int8_ptr,
                                float *x_scale_ptr,
                                int m, int n, int k, float scale,
                                bool use_fp32_output, int splitk, int sm_count,
                                cudaStream_t stream) {
  return dispatch_gemm_bf16xfp32_activation<Activation, HasBias, false,
                                            kernels::Int8ResidualBackend>(
      y_ptr, splitk_y_ptr, split_flag_ptr, x_ptr, w_high_ptr,
      w_low_int8_ptr, bias_ptr, x_int8_ptr, m, n, k, scale,
      use_fp32_output, splitk, sm_count, stream, w_scale_ptr, x_scale_ptr);
}

GemmFixedInt8EpilogueLauncher resolve_gemm_bf16xfp32_int8_launcher(
    int activation_type, bool has_bias) noexcept {
  switch (activation_type) {
    case 0:
      return has_bias ? &launch_fixed_epilogue_int8<IdentityActivation, true>
                      : &launch_fixed_epilogue_int8<IdentityActivation, false>;
    case 1:
      return has_bias ? &launch_fixed_epilogue_int8<SiluActivation, true>
                      : &launch_fixed_epilogue_int8<SiluActivation, false>;
    case 2:
      return has_bias ? &launch_fixed_epilogue_int8<GeluActivation, true>
                      : &launch_fixed_epilogue_int8<GeluActivation, false>;
    default:
      return nullptr;
  }
}

}  // namespace mixed_gemm
