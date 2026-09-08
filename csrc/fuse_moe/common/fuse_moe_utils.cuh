// fuse_moe_utils.cuh — fuse_moe 算子的通用工具（向量 load/store、类型转换、
// silu、PDL 辅助、SM 数查询）。
//
// 自包含（不依赖 hpc-ops 树）：向量工具沿用 csrc/mixed_gemm/utils.cuh 的形态，
// 扩展了 16-bit (bf16/fp16) ↔ fp32 双向转换；PDL 辅助封装 sm90+ 的
// cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion，
// 非 sm90+ 编译目标编译为空（host 侧按 compute capability 决定是否携带
// PDL launch attribute）。

#ifndef FUSE_MOE_SRC_FUSE_MOE_UTILS_CUH_
#define FUSE_MOE_SRC_FUSE_MOE_UTILS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cutlass/numeric_types.h>

namespace fuse_moe {

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

// 通用向量类型转换：任意元素类型经 float 往返（cutlass::half_t /
// cutlass::bfloat16_t 的 float 构造/转换均为 RN 舍入；float→float 恒等）。
template <typename U, typename T, int N>
__device__ __forceinline__ constexpr vec_t<U, N> to(const vec_t<T, N> &v) {
  vec_t<U, N> o;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    o[i] = static_cast<U>(static_cast<float>(v[i]));
  }
  return o;
}

template <typename T, int N>
__device__ __forceinline__ constexpr vec_t<T, N> load(const void *ptr) {
  vec_t<T, N> v;

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
//    Activation
// ============================
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// ============================
//    PDL 辅助（sm90+；低架构编译为空）
// ============================
// acquire：等待上游 kernel 全部完成（其 gmem 写入对 本 grid 可见）。
// 未以 PDL attribute 启动时为 no-op（安全，FA combine kernel 同款用法）。
__device__ __forceinline__ void pdl_acquire() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
}

// release：通知下游可以提前启动。未以 PDL 启动时为 no-op。
__device__ __forceinline__ void pdl_release() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// host 侧启动。⚠️ 关键约束：kernel 体内的 griddepcontrol 指令（pdl_acquire/
// pdl_release，sm90+ 编译目标恒内联）必须与 launch 的 PSS 属性配对——
// 非 PDL 启动的 kernel 执行 griddepcontrol.wait 是 UB（实测 compute-
// sanitizer synccheck 表现为部分线程挂起 → mbarrier Missing init → 死锁；
// 早先「无属性时为 no-op」的假设是错的）。因此 sm90+ 设备上恒带 PSS 属性：
// 无上游 trigger 时驱动退化为「等上游完成」的正常串行语义，无害；sm89
// 及以下 kernel 体无指令，普通 launch，天然配对。use_pdl 参数保留仅为
// 调用方签名兼容（不再控制属性）。
template <typename KernelT, typename... Args>
static inline void launch_kernel_pdl(KernelT kernel, dim3 grid, dim3 block, size_t smem,
                                      cudaStream_t stream, bool use_pdl, Args... args) {
  (void)use_pdl;
  static const bool s_pdl_hw = []() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
      return false;
    }
    return major >= 9;
  }();
  // FUSE_MOE_NO_PDL / FUSE_MOE_TIME：强制普通 launch（串行化下游启动）。
  // kernel 体的 griddepcontrol 指令在无 PSS 属性时为 no-op（PTX 规定，
  // FA combine kernel 同款用法），安全；串行化是 FUSE_MOE_TIME 分段计时
  // 能归因到单个 kernel 的前提。
  static const bool s_force_plain = std::getenv("FUSE_MOE_NO_PDL") != nullptr ||
                                    std::getenv("FUSE_MOE_TIME") != nullptr;
  if (s_pdl_hw && !s_force_plain) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
  } else {
    kernel<<<grid, block, smem, stream>>>(args...);
  }
}

// ============================
//    Device Information
// ============================
inline int get_sm_count() {
  static int num_sm = -1;
  if (num_sm == -1) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    num_sm = prop.multiProcessorCount;
  }
  return num_sm;
}

// PDL 是否可用于当前设备（compute capability >= 9.0；sm120 为 true，sm89 为 false）
inline bool pdl_supported() {
  static const bool supported = []() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
      return false;
    }
    return major >= 9;
  }();
  return supported;
}

}  // namespace fuse_moe

#endif  // FUSE_MOE_SRC_FUSE_MOE_UTILS_CUH_
