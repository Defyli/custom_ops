/*
 * arch_targets.h — 目标架构条件编译的标准入口（全仓库唯一汇聚点）
 *
 * 职责边界
 * --------
 * 构建侧只回答一个问题：「本次编译的 gencode 覆盖了哪些 kernel 家族」，
 * 以一个宏注入：-DFA_TARGETS=<bits>（唯一注入方：custom_ops/__init__.py
 * 的 _arch_flags_from）。本文件回答所有其余问题：
 *   1. 每个家族是否编入本二进制（FA_HAS_SM70 / FA_HAS_SM8X / FA_HAS_SM120）
 *   2. 运行时设备能力（arch_targets::gpu_major()，进程内 static 缓存）
 *
 * 位约定（注入方与本文件必须一致；修改任一侧须同步）：
 *   bit0 (0x1)  FA_TGT_SM70  : sm_70（V100，fp16 专用 WMMA 路径）
 *   bit1 (0x2)  FA_TGT_SM8X  : sm_80/86/89/90（cp.async 通用路径，含 4090）
 *   bit2 (0x4)  FA_TGT_SM120 : sm_120a（Blackwell consumer，TMA 路径，5090）
 *
 * 未注入时缺省全家族（0x7，如脱离本框架直接 nvcc / setup.py 构建）：
 * host 侧照常分发，cubin 缺失的路径在 cudaLaunch 处显式报
 * "no kernel image for device"——响亮失败，绝不静默。
 *
 * 使用约定
 * --------
 * - 所有需要按「二进制含哪些 kernel」做编译期裁剪的 host 代码，只 include
 *   本文件并用 FA_HAS_* 宏，禁止各自发明判定宏；
 * - device 侧 kernel 真身的按目标裁剪仍用 __CUDA_ARCH__ 区间（nvcc 对
 *   gencode 列表逐目标各编译一遍，这是按目标裁剪的唯一可靠手段，见各
 *   smXX kernel 头内的 stub 模式），与 host 宏解耦；
 * - 分发处必须「编译期 FA_HAS_* × 运行期 gpu_major()」双重校验后才调
 *   launcher，二者缺一不可（防「宏说有、cubin 没有」与「cubin 有、设备
 *   不匹配」两类静默错误）。
 */

#pragma once

#include <cuda_runtime.h>

#ifndef FA_TARGETS
#define FA_TARGETS 0x7
#endif

#define FA_TGT_SM70  0x1
#define FA_TGT_SM8X  0x2
#define FA_TGT_SM120 0x4

#if FA_TARGETS & FA_TGT_SM70
#  define FA_HAS_SM70 1
#else
#  define FA_HAS_SM70 0
#endif

#if FA_TARGETS & FA_TGT_SM8X
#  define FA_HAS_SM8X 1
#else
#  define FA_HAS_SM8X 0
#endif

#if FA_TARGETS & FA_TGT_SM120
#  define FA_HAS_SM120 1
#else
#  define FA_HAS_SM120 0
#endif

namespace arch_targets {

// 运行时 compute capability major（首次调用缓存，进程内不再查询）。
// 查询失败返回 -1（分发侧按「无可用路径」处理并显式报错）。
inline int gpu_major() {
    static const int major = []() {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return -1;
        int m = 0;
        if (cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor,
                                   dev) != cudaSuccess) {
            return -1;
        }
        return m;
    }();
    return major;
}

}  // namespace arch_targets
