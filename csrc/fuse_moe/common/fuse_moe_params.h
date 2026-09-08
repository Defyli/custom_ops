/*
 * fuse_moe_params.h — fuse_moe 的跨架构参数结构与流水线计时工具
 *
 * 架构分层（仿 csrc/fa）：
 *   fuse_moe_op.cu        纯 op 层：输入校验 / workspace 分配 / 填 params /
 *                         调 moe_fwd_launch()，零架构感知
 *   fuse_moe_launch.h     分发入口：按「编译期 FA_HAS_* × 运行期 gpu_major()」
 *                         双重校验后分发到各家族策略入口（gemm 引擎、tile
 *                         策略、PDL 开关全部内封在家族入口内）
 *   common/               跨架构共享（本文件 + moe_kernels + utils +
 *                         group_gemm_config traits）
 *   sm89/  sm120/         GEMM kernel 家族实现
 *
 * FMOE_params 沿用 FA_mask_params 的职责约定：op 层填输入指针/形状与
 * 架构无关的 workspace；tile_m / use_pdl 等策略字段由所选家族入口填写
 * （见 fuse_moe_launch.h）。
 */

#ifndef FUSE_MOE_SRC_COMMON_FUSE_MOE_PARAMS_H_
#define FUSE_MOE_SRC_COMMON_FUSE_MOE_PARAMS_H_

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace fuse_moe {

// ── 流水线分段计时（FUSE_MOE_TIME=1）───────────────────────────────────
// 每个 kernel launch 前 CUDA event 打点，finish() 时同步打印各段耗时。
// 计时模式下 launch 侧自动强制普通串行 launch（见 fuse_moe_utils.cuh 的
// launch_kernel_pdl），分段时长即各 kernel 的 wall time（PDL 重叠会让
// 分段无法归因）。事件上限 12：各家族 pipeline 的 kernel 数 ≤ 6。
struct FmMarks {
    cudaStream_t stream = nullptr;
    bool time_on = false;
    cudaEvent_t ev[12] = {};
    const char *name[12] = {};
    int n = 0;

    FmMarks() : time_on(std::getenv("FUSE_MOE_TIME") != nullptr) {}

    void bind(cudaStream_t s) { stream = s; }

    void mark(const char *tag) {
        if (time_on && n < 12) {
            cudaEventCreate(&ev[n]);
            cudaEventRecord(ev[n], stream);
            name[n] = tag;
            ++n;
        }
    }

    // 分段计时结果：段 i = 事件 i → i+1 之间执行的 kernel（事件在各 kernel
    // launch 前记录）。流水线全部 launch 完成后调用。
    void finish() {
        if (!time_on || n == 0) return;
        cudaEvent_t ev_end;
        cudaEventCreate(&ev_end);
        cudaEventRecord(ev_end, stream);
        cudaStreamSynchronize(stream);
        fprintf(stderr, "[fm_time]");
        for (int i = 0; i < n; ++i) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ev[i], (i + 1 < n) ? ev[i + 1] : ev_end);
            fprintf(stderr, " %s=%.0fus", name[i], ms * 1000.f);
        }
        float total_ms = 0.f;
        cudaEventElapsedTime(&total_ms, ev[0], ev_end);
        fprintf(stderr, " | total=%.0fus\n", total_ms * 1000.f);
        std::fflush(stderr);
        for (int i = 0; i < n; ++i) cudaEventDestroy(ev[i]);
        cudaEventDestroy(ev_end);
        n = 0;
    }
};

// ── tile 策略（count/gather/GEMM 三者共用同一 kTileM）───────────────────
// avg<=48 → 32；avg>=256 且允许 → 128；否则 64。
// kTileM=128 将每 M-tile 装载的 W 复用面扩大一倍（W 流量随 task 数减半），
// 代价是每 expert 平均 M/2 行 padding 浪费（大 avg 时占比低）。
// M128 仅 cp.async 家族；TMA 家族（可选路径）固定 64。
// FUSE_MOE_TILE_M=32/64/128 可强制（调优/复现用）。
inline int moe_pick_tile_m(int avg_tokens_per_expert, bool allow_128) {
    int tm;
    if (avg_tokens_per_expert <= 48) {
        tm = 32;
    } else if (avg_tokens_per_expert >= 256 && allow_128) {
        tm = 128;
    } else {
        tm = 64;
    }
    if (const char *env_tm = std::getenv("FUSE_MOE_TILE_M")) {
        const int v = std::atoi(env_tm);
        if (v == 32 || v == 64 || (v == 128 && allow_128)) tm = v;
    }
    return tm;
}

// ── 参数结构（op 层填写；策略字段由家族入口覆盖）────────────────────────
struct FMOE_params {
    // 输入
    const void *x_ptr;        // (S, H)，token 序
    const void *w1_ptr;       // (E, 2I, H) gate_up 权重
    const void *w2_ptr;       // (E, H, I) down 权重
    const int *topk_ids_ptr;  // (S, K) int32
    const float *topk_scale_ptr;  // (S, K) fp32

    // workspace（op 层分配，架构无关）。gate_up_out 仅 TMA 家族使用
    //（其 gemm1 无 gate/up 配对融合变体）；cp.async 家族的 gemm1 融合
    // epilogue 直写 act_out，不消费该缓冲。
    int *row_indices_ptr;  // (T) expert 排序后布局的源行号
    int *topk_pos_ptr;     // (S, K) (token, topk) → 排序后行号（无效槽位 -1）
    int *seqlens_ptr;      // (E)
    int *cu_seqlens_ptr;   // (E+1) compact 前缀和
    int *tiles_ptr;        // (E) 每 expert 的 kTileM tile 数
    void *gate_up_out_ptr;  // (T, 2I)，仅 TMA 家族
    void *act_out_ptr;      // (T, I)
    void *down_out_ptr;     // (T, H)
    void *out_ptr;          // (S, H) 最终输出

    // 形状
    int num_seq;       // S
    int hidden;        // H（%64==0）
    int intermediate;  // I（%64==0）
    int num_expert;    // E（<=512）
    int num_topk;      // K（<=128）
    int total_num_seq; // T = S * K
    int avg_tokens_per_expert;  // T / E
    bool is_bf16;               // false → fp16

    // 策略（fuse_moe_launch 分发时填写）
    int tile_m;    // count/GEMM 共用 kTileM（moe_pick_tile_m）
    bool use_pdl;  // sm90+ 硬件且未设 FUSE_MOE_NO_PDL

    // 分段计时（FUSE_MOE_TIME）
    FmMarks marks;
};

// 家族入口内的 kernel 打点宏（params 为 FMOE_params& 形参名约定）
#define FM_DEBUG_MARK(tag) (params).marks.mark(tag)

}  // namespace fuse_moe

#endif  // FUSE_MOE_SRC_COMMON_FUSE_MOE_PARAMS_H_
