/*
 * fuse_moe_launch.h — MoE 前向融合算子架构分发的标准入口（op 层唯一依赖）
 *
 * 分层（仿 csrc/fa/fa_fwd_launch.h，自上而下每层只有一个事实来源）：
 *   csrc/arch_targets.h  仓库级标准入口：FA_TARGETS → FA_HAS_SM70/SM8X/SM120
 *                        （二进制含哪些 gencode 家族）+ gpu_major()（当前设备）
 *   本文件               fuse_moe 分发入口：按「编译期 FA_HAS_* × 运行期
 *                        gpu_major()」双重校验后分发到家族策略入口；gemm
 *                        引擎选择 / tile 策略 / PDL 开关全部内封在家族入口
 *   fuse_moe_op.cu       纯 op 层：输入校验 / workspace 分配 / 填 params /
 *                        调 moe_fwd_launch(params, stream)，零架构感知
 *   common/              跨架构共享：moe_kernels（count/gather/act/reduce）、
 *                        group_gemm_config（GEMM traits）、utils（PDL）
 *   sm89/  sm120/        GEMM kernel 家族实现（host launcher 无分发逻辑）
 *
 * 各家族路径概览：
 *   sm8x（SM80~SM118 cp.async 基线，4090D=sm89）
 *     gemm1 = gate/up 配对融合（scatter gather-on-load + silu·mul epilogue）
 *     gemm2 = cp.async multistage（激活连续）
 *     → 跨 tile 连续流水（slab 发射流全局计数 + 寄存器行索引 +
 *        向量化 epilogue，见 sm89/group_gemm_sm89.cuh）
 *   sm120（Blackwell consumer，RTX 50）
 *     默认与 sm8x 共享 cp.async 引擎（各 shape 实测更优）；FUSE_MOE_TMA=1
 *     可选切换 TMA + mbarrier 引擎（gemm1 = gather_sorted + TMA、
 *     gemm2 = TMA 直读 compact act_out，见 sm120/group_gemm_sm120.cuh）
 *
 * 环境开关：
 *   FUSE_MOE_TMA=1    sm120 上启用 TMA 引擎（可选路径）
 *   FUSE_MOE_TILE_M   强制 kTileM：32 / 64 / 128（128 仅 cp.async 引擎）
 *   FUSE_MOE_NO_PDL   强制普通 launch（分段计时归因用）
 *   FUSE_MOE_TIME=1   流水线分段计时（各 kernel wall time）
 */

#pragma once

#include <cstdlib>

#include <torch/extension.h>

#include "arch_targets.h"
#include "common/fuse_moe_params.h"
#include "common/moe_kernels.cuh"
#include "sm89/group_gemm_sm89.cuh"
#if FA_HAS_SM120
#include "sm120/group_gemm_sm120.cuh"
#endif

namespace fuse_moe {

// ════════════════════════════════════════════════════════════════════════════
// sm8x 家族（SM80~SM118，cp.async 路径；4090D = sm89）
// ════════════════════════════════════════════════════════════════════════════
// cp.async kernel 在所有 sm80+ gencode（含 sm_120a）下均有 cubin，故本家族
// 无 FA_HAS 编译期门槛——运行期 major>=8 即可分发；sm120 的默认引擎亦复用
// 本接线（见下方 sm120 家族）。
template <typename T>
inline void moe_launch_sm8x_impl(FMOE_params &params, cudaStream_t stream) {
    const int intermediate2 = params.intermediate * 2;
    const bool is_half = !params.is_bf16;

    // ① count / build indices（row_indices / topk_pos / cu_seqlens / tiles）
    FM_DEBUG_MARK("count_and_build");
    count_and_build_indices_async(params.topk_ids_ptr, params.row_indices_ptr,
                                  params.topk_pos_ptr, params.seqlens_ptr, params.cu_seqlens_ptr,
                                  params.tiles_ptr, params.num_seq, params.num_topk,
                                  params.num_expert, params.tile_m, params.use_pdl, stream);

    // ② gemm1 融合：gate/up 配对 N-tile + silu·mul epilogue，直写
    //    act_out (T, I)——免 gate_up_out (T, 2I) 物化与独立 act_mul kernel
    //    （省 2×T×I×2B DRAM 往返；X tile 装载量减半，见
    //    sm89/group_gemm_sm89.cuh 的 group_gemm_gateup_kernel）
    FM_DEBUG_MARK("gemm1_fused");
    group_gemm::group_gemm_gateup_fused_async(
        params.act_out_ptr, params.x_ptr, params.w1_ptr, params.row_indices_ptr,
        params.seqlens_ptr, params.cu_seqlens_ptr, params.tiles_ptr,
        /*n=*/intermediate2, /*k=*/params.hidden, params.num_expert, params.tile_m, is_half,
        params.use_pdl, stream);

    // ③ down group GEMM（cp.async multistage：激活连续）
    FM_DEBUG_MARK("gemm2_cp");
    group_gemm::group_gemm_async(
        params.down_out_ptr, params.act_out_ptr, params.w2_ptr, /*row_indices=*/nullptr,
        params.seqlens_ptr, params.cu_seqlens_ptr, params.tiles_ptr,
        /*n=*/params.hidden, /*k=*/params.intermediate, params.num_expert, params.tile_m, is_half,
        params.use_pdl, stream);

    // ④ topk 加权 reduce
    FM_DEBUG_MARK("reduce");
    reduce_async(reinterpret_cast<T *>(params.out_ptr),
                 reinterpret_cast<const T *>(params.down_out_ptr), params.topk_pos_ptr,
                 params.topk_scale_ptr, params.num_seq, params.hidden, params.num_topk,
                 params.use_pdl, stream);
}

// sm8x 家族策略入口：tile 选择 + dtype 分发
inline void moe_launch_sm8x(FMOE_params &params, cudaStream_t stream) {
    params.tile_m = moe_pick_tile_m(params.avg_tokens_per_expert, /*allow_128=*/true);
    if (params.is_bf16) {
        moe_launch_sm8x_impl<cutlass::bfloat16_t>(params, stream);
    } else {
        moe_launch_sm8x_impl<cutlass::half_t>(params, stream);
    }
    params.marks.finish();
}

// ════════════════════════════════════════════════════════════════════════════
// sm120 家族（Blackwell consumer，RTX 50）
// ════════════════════════════════════════════════════════════════════════════
// 默认引擎：sm8x 家族的连续流水 cp.async kernel（cp.async 指令在 sm120 硬件
// 受支持，kernel 对全 sm80+ gencode 均有 cubin；各 shape 端到端实测均优于
// 本家族 TMA 引擎 3~15%，5090D / PDL 稳态）。
// FUSE_MOE_TMA=1：TMA + mbarrier 可选引擎（gemm1 = gather_sorted 物化 +
// TMA、gemm2 = TMA 直读 compact act_out，见 sm120/group_gemm_sm120.cuh）。
// TMA 家族固定 kTileM ≤ 64。
#if FA_HAS_SM120

template <typename T>
inline void moe_launch_sm120_impl(FMOE_params &params, cudaStream_t stream) {
    const int intermediate2 = params.intermediate * 2;
    const bool is_half = !params.is_bf16;

    if (std::getenv("FUSE_MOE_TMA") == nullptr) {
        moe_launch_sm8x_impl<T>(params, stream);
        return;
    }

    // ── FUSE_MOE_TMA=1：TMA 引擎 ──────────────────────────────────────
    // TMA 家族固定 kTileM ≤ 64（count 阶段尚未运行，此处改写安全）。
    if (params.tile_m > 64) params.tile_m = 64;
    torch::Tensor x_sorted = torch::empty(
        {(int64_t)params.total_num_seq, (int64_t)params.hidden},
        torch::TensorOptions().dtype(params.is_bf16 ? torch::kBFloat16 : torch::kHalf).device(
            torch::kCUDA));

    // ① count / build indices
    FM_DEBUG_MARK("count_and_build");
    count_and_build_indices_async(params.topk_ids_ptr, params.row_indices_ptr,
                                  params.topk_pos_ptr, params.seqlens_ptr, params.cu_seqlens_ptr,
                                  params.tiles_ptr, params.num_seq, params.num_topk,
                                  params.num_expert, params.tile_m, params.use_pdl, stream);

    // ②a gather：x → x_sorted（expert 有序 compact，TMA gemm1 的 A 前提）
    FM_DEBUG_MARK("gather_sorted");
    gather_sorted_async(reinterpret_cast<T *>(x_sorted.data_ptr()),
                        reinterpret_cast<const T *>(params.x_ptr), params.row_indices_ptr,
                        params.cu_seqlens_ptr, params.hidden, params.num_expert, params.use_pdl,
                        stream);

    // ②b gate_up group GEMM（TMA：A = x_sorted compact，任意行坐标基）
    FM_DEBUG_MARK("gemm1_tma");
    group_gemm::tma::group_gemm_tma_async(
        params.gate_up_out_ptr, x_sorted.data_ptr(), params.w1_ptr, params.seqlens_ptr,
        params.cu_seqlens_ptr, params.tiles_ptr,
        /*a_rows=*/params.total_num_seq,
        /*n=*/intermediate2, /*k=*/params.hidden, params.num_expert, params.tile_m, is_half,
        params.use_pdl, stream);

    // ③ silu(gate) * up（compact）
    FM_DEBUG_MARK("act_mul");
    act_mul_async(reinterpret_cast<T *>(params.act_out_ptr),
                  reinterpret_cast<const T *>(params.gate_up_out_ptr), params.total_num_seq,
                  params.intermediate, params.use_pdl, stream);

    // ④ down group GEMM（TMA：A = act_out compact，任意行坐标基）
    FM_DEBUG_MARK("gemm2_tma");
    group_gemm::tma::group_gemm_tma_async(
        params.down_out_ptr, params.act_out_ptr, params.w2_ptr, params.seqlens_ptr,
        params.cu_seqlens_ptr, params.tiles_ptr,
        /*a_rows=*/params.total_num_seq,
        /*n=*/params.hidden, /*k=*/params.intermediate, params.num_expert, params.tile_m, is_half,
        params.use_pdl, stream);

    // ⑤ topk 加权 reduce
    FM_DEBUG_MARK("reduce");
    reduce_async(reinterpret_cast<T *>(params.out_ptr),
                 reinterpret_cast<const T *>(params.down_out_ptr), params.topk_pos_ptr,
                 params.topk_scale_ptr, params.num_seq, params.hidden, params.num_topk,
                 params.use_pdl, stream);
}

// sm120 家族策略入口：tile 选择 + dtype 分发 + 引擎选择
inline void moe_launch_sm120(FMOE_params &params, cudaStream_t stream) {
    params.tile_m = moe_pick_tile_m(params.avg_tokens_per_expert, /*allow_128=*/true);
    if (params.is_bf16) {
        moe_launch_sm120_impl<cutlass::bfloat16_t>(params, stream);
    } else {
        moe_launch_sm120_impl<cutlass::half_t>(params, stream);
    }
    params.marks.finish();
}

#endif  // FA_HAS_SM120

// ════════════════════════════════════════════════════════════════════════════
// 对外唯一分发入口（fuse_moe_op.cu 仅依赖此函数）
// ════════════════════════════════════════════════════════════════════════════
// 「编译期 FA_HAS_*（二进制含哪些 gencode）× 运行期 gpu_major()（当前设备）」
// 双重校验后分发；均不匹配时显式报错——绝不静默跑空 kernel。
// PDL 开关在此统一判定（sm90+ 硬件；FUSE_MOE_NO_PDL 强制普通 launch，
// 分段计时归因用——launch_kernel_pdl 对 FUSE_MOE_TIME 亦自动串行化）。
inline void moe_fwd_launch(FMOE_params &params, cudaStream_t stream) {
    const int major = arch_targets::gpu_major();

    params.use_pdl = (major >= 9) && (std::getenv("FUSE_MOE_NO_PDL") == nullptr);
    params.marks.bind(stream);

#if FA_HAS_SM120
    // sm120 家族：默认 cp.async 引擎，FUSE_MOE_TMA=1 可选 TMA 引擎（入口
    // 内部判定，TMA 需要 gencode 覆盖 >= 90——FA_HAS_SM120 恰是此语义）。
    if (major == 12) {
        moe_launch_sm120(params, stream);
        return;
    }
#endif
    // cp.async 家族：kernel 在所有 sm80+ gencode 下均有 cubin（含 sm_120a），
    // 无编译期门槛；运行期 major>=8 保证硬件支持 cp.async。
    if (major >= 8) {
        moe_launch_sm8x(params, stream);
        return;
    }
    TORCH_CHECK(false,
                "no fuse_moe kernel for this GPU (sm_", major,
                "xx, FA_TARGETS=0x", FA_TARGETS,
                "); fuse_moe requires compute capability >= 8.0 (cp.async)");
}

}  // namespace fuse_moe
