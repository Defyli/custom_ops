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
#include <map>
#include <memory>
#include <mutex>

#include <c10/cuda/CUDAException.h>

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
// sm90+ 的 griddepcontrol 指令须与硬件对应的 PSS launch 属性配对。
__device__ __forceinline__ void pdl_acquire() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
}

// release：通知下游可以提前启动；sm90+ 同样要求匹配 PSS launch 属性。
__device__ __forceinline__ void pdl_release() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// host 侧启动。⚠️ 关键约束：kernel 体内的 griddepcontrol 指令（pdl_acquire/
// pdl_release，sm90+ 编译目标恒内联）必须与 launch 的 PSS 属性配对——
// 非 PDL 启动的 kernel 执行 griddepcontrol.wait 是 UB（实测 compute-
// sanitizer synccheck 表现为部分线程挂起 → mbarrier Missing init → 死锁；
// 早先「无属性时为 no-op」的假设是错的）。offload 每次按当前设备匹配 PSS：
// sm90+ 带属性，无上游 trigger 时驱动退化为「等上游完成」的串行语义；
// sm89 及以下 kernel 体无指令，普通 launch，天然配对。legacy 仍保留
// 首次设备缓存和诊断环境覆盖；这些覆盖不允许用于 offload。use_pdl
// 参数保留仅为调用方签名兼容（不再控制属性）。
template <typename KernelT, typename... Args>
static inline void launch_kernel_pdl_config(KernelT kernel, dim3 grid, dim3 block, size_t smem,
                                            cudaStream_t stream, bool use_pdl,
                                            bool allow_environment, Args... args) {
  (void)use_pdl;
  const auto detect_pdl_hw = []() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
      return false;
    }
    return major >= 9;
  };
  bool pdl_hw;
  if (allow_environment) {
    // Preserve the legacy first-device cache, initialized only by legacy launches.
    static const bool s_pdl_hw = detect_pdl_hw();
    pdl_hw = s_pdl_hw;
  } else {
    // Offload may switch between devices with different SM capabilities.
    pdl_hw = detect_pdl_hw();
  }
  // Legacy diagnostic overrides retain their force-plain behavior for serialized
  // launches/timing; offload must not read them or override hardware-matched PSS.
  bool force_plain = false;
  if (allow_environment) {
    static const bool s_force_plain = std::getenv("FUSE_MOE_NO_PDL") != nullptr ||
                                      std::getenv("FUSE_MOE_TIME") != nullptr;
    force_plain = s_force_plain;
  }
  if (pdl_hw && !force_plain) {
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

// Legacy launches retain environment overrides; explicit offload launches bypass them.
template <typename KernelT, typename... Args>
static inline void launch_kernel_pdl(KernelT kernel, dim3 grid, dim3 block, size_t smem,
                                     cudaStream_t stream, bool use_pdl, Args... args) {
  launch_kernel_pdl_config(kernel, grid, block, smem, stream, use_pdl,
                          /*allow_environment=*/true, args...);
}

// Function attributes belong to a device/kernel, not a context or stream.
// Publish a stable hardware limit once; never shrink it to a wave's group count.
// CUDA device reset/module unload while these caches are live is unsupported.
template <typename KernelT>
inline void configure_dynamic_shared_memory(KernelT kernel, int requested) {
  struct State {
    std::once_flag once;
    int limit = 0;
  };
  using Cache = std::map<int, std::map<KernelT, std::shared_ptr<State>>>;
  static Cache cache;
  static std::mutex cache_mutex;
  // Warm calls avoid the registry mutex; all threads share each state's once flag.
  static thread_local Cache local_cache;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  auto& state = local_cache[device][kernel];
  if (!state) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto& shared = cache[device][kernel];
    if (!shared) shared = std::make_shared<State>();
    state = shared;
  }
  std::call_once(state->once, [&] {
    int optin = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    TORCH_CHECK(optin >= 0 && attributes.sharedSizeBytes <= static_cast<size_t>(optin),
                "kernel static shared memory exceeds device opt-in limit");
    const int limit = optin - static_cast<int>(attributes.sharedSizeBytes);
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit));
    state->limit = limit;
  });
  TORCH_CHECK(requested >= 0 && requested <= state->limit,
              "requested dynamic shared memory (", requested,
              " bytes) exceeds device/kernel limit (", state->limit,
              " bytes) on CUDA device ", device);
}

// ============================
//    Device Information
// ============================
inline int get_sm_count() {
  // Each thread caches checked device queries independently, with no shared writes.
  static thread_local std::map<int, int> counts;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const auto found = counts.find(device);
  if (found != counts.end()) return found->second;
  int count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
  return counts.emplace(device, count).first->second;
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
