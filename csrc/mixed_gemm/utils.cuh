// Minimal device-side utilities needed by gemm_bf16xfp32_sm80.
//
// This is a trimmed-down, standalone copy of the small subset of
// hpc-ops' src/utils/utils.cuh that gemm_bf16xfp32_sm80.cu actually needs
// (vectorized load/store + fp32<->bf16 conversion helpers), so that this
// repository has no dependency on the hpc-ops tree at all.

#ifndef MIXED_GEMM_SRC_UTILS_CUH_
#define MIXED_GEMM_SRC_UTILS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>

namespace mixed_gemm {

// ============================
//    Load/Store (vectorized)
// ============================
template <typename T, int N>
struct vec_t {
  T data[N];

  using type = T;
  static constexpr int num = N;
  static constexpr int kNum = N;

  __device__ __forceinline__ constexpr T &operator[](int idx) { return data[idx]; }
  __device__ __forceinline__ constexpr const T &operator[](int idx) const { return data[idx]; }
};

template <typename U, typename T, int N>
__device__ __forceinline__ constexpr auto to(const vec_t<T, N> &v) {
  if constexpr (std::is_same_v<T, float> && std::is_same_v<U, __nv_bfloat16>) {
    vec_t<__nv_bfloat16, N> o;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      o[i] = __float2bfloat16(v[i]);
    }
    return o;
  } else if constexpr (std::is_same_v<T, U>) {
    return v;
  }
}

template <typename T, int N>
__device__ __forceinline__ constexpr auto load(const void *ptr) {
  using V = vec_t<T, N>;
  V v;

  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "not support for T x N");

  if constexpr (kBytes == 4) {
    using L = uint32_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 8) {
    using L = uint64_t;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  } else if constexpr (kBytes == 16) {
    using L = uint4;
    *reinterpret_cast<L *>(&v) = *reinterpret_cast<const L *>(ptr);
  }

  return v;
}

template <typename T, int N>
__device__ __forceinline__ constexpr void store(void *ptr, const vec_t<T, N> &v) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "not support for T x N");

  if constexpr (kBytes == 4) {
    using S = uint32_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 8) {
    using S = uint64_t;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  } else if constexpr (kBytes == 16) {
    using S = uint4;
    *reinterpret_cast<S *>(ptr) = *reinterpret_cast<const S *>(&v);
  }
}

// ============================
//    Fused epilogue activations
// ============================
struct IdentityActivation {
  __device__ __forceinline__ float operator()(float x) const { return x; }
};

struct SiluActivation {
  __device__ __forceinline__ float operator()(float x) const {
    return x / (1.0f + __expf(-x));
  }
};

struct GeluActivation {
  __device__ __forceinline__ float operator()(float x) const {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    constexpr float kCubic = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + kCubic * x3)));
  }
};

// ================================
//    Memory-order-aware LD Primitives
// ================================
__device__ __forceinline__ int load_global_volatile(int *ptr) {
  int val;
  asm volatile("ld.volatile.global.s32 {%0}, [%1];\n" : "=r"(val) : "l"(ptr));
  return val;
}

// ============================
//    Device Information
// ============================
inline int get_sm_count() {
  static int num_sm = -1;
  if (num_sm == -1) {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    num_sm = prop.multiProcessorCount;
  }
  return num_sm;
}

}  // namespace mixed_gemm

#endif  // MIXED_GEMM_SRC_UTILS_CUH_
