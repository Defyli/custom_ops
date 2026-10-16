/**
 * fuse_moe_bindings.cpp — fuse_moe 分组算子注册入口
 *
 * 命名空间: recsys_ops → torch.ops.recsys_ops.fuse_moe(...)
 * 本分组独立编译为 recsys_fuse_moe_kernel.so（懒加载：首次访问该算子时才
 * JIT——只测 fuse_moe 时不再编译 FA/mixed_gemm）。
 */

// Step 1: 定义命名空间
#define CUSTOM_OPS_NAMESPACE recsys_ops

// Step 2: 包含通用框架宏
#include "custom_ops_macros.h"

// Step 3: 包含算子声明头文件
#include "fuse_moe/fuse_moe_op.h"
#include "fuse_moe/fuse_moe_offload.h"

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: 分发函数
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_DISPATCH_FN(
    fuse_moe,
    (const torch::Tensor& x,
     const torch::Tensor& gate_up_weight,
     const torch::Tensor& down_weight,
     const torch::Tensor& topk_ids,
     const torch::Tensor& topk_scale),
    (x, gate_up_weight, down_weight, topk_ids, topk_scale),
    x
)

// ─────────────────────────────────────────────────────────────────────────────
// Step 5/6: 注册 schema + 绑定实现
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(
        fuse_moe,
        "Tensor x, Tensor gate_up_weight, Tensor down_weight, "
        "Tensor topk_ids, Tensor topk_scale",
        "-> Tensor"
    )
    m.def("fuse_moe_offload(Tensor x, Tensor gate_up_weight_cpu, Tensor down_weight_cpu, "
          "Tensor topk_ids, Tensor topk_scale, Tensor(a!) runtime_token, int layer_id, "
          "int[]? deployment_config=None, int[]? deployment_manifest=None) -> Tensor");
    m.def("fuse_moe_offload_create(int device, int gpu_budget_bytes=0, int ram_budget_bytes=0, "
          "int pinned_budget_bytes=0, int num_banks=2, int activation_mode=0, "
          "int max_route_tile_rows=4096, int resident_cache_bytes=0, int dynamic_cache_bytes=0, "
          "int cache_policy=0, int cache_decay_interval=32, int cache_min_reuse=2, "
          "int cache_update_bytes=0, bool async_staging=False, bool host_profile=False, "
          "int tile_m=0, int tile_n=128) -> Tensor");
    m.def("fuse_moe_offload_bind(Tensor runtime_token, int layer_id, Tensor w1, Tensor w2) -> ()");
    m.def("fuse_moe_offload_resident_bind(Tensor token, int layer_id, int[] expert_ids) -> ()");
    m.def("fuse_moe_offload_validate(Tensor runtime_token, int[] deployment_manifest) -> ()");
    m.def("fuse_moe_offload_begin(Tensor runtime_token) -> ()");
    m.def("fuse_moe_offload_check(Tensor runtime_token) -> ()");
    m.def("fuse_moe_offload_status(Tensor runtime_token) -> Tensor");
    m.def("fuse_moe_offload_acknowledge(Tensor runtime_token) -> ()");
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
    m.def("_fuse_moe_offload_test_inject(Tensor token, str point, int countdown=1) -> ()");
    m.def("_fuse_moe_offload_test_worker_states() -> Tensor");
#endif
    m.def("fuse_moe_offload_close(Tensor runtime_token) -> ()");
    m.def("fuse_moe_offload_stats(Tensor runtime_token) -> Tensor");
CUSTOM_OPS_LIBRARY_END

CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(fuse_moe)
    m.impl("fuse_moe_offload", &fuse_moe_offload);
    m.impl("fuse_moe_offload_create", &fuse_moe_offload_create);
    m.impl("fuse_moe_offload_bind", &fuse_moe_offload_bind);
    m.impl("fuse_moe_offload_resident_bind", &fuse_moe_offload_resident_bind);
    m.impl("fuse_moe_offload_validate", &fuse_moe_offload_validate);
    m.impl("fuse_moe_offload_begin", &fuse_moe_offload_begin);
    m.impl("fuse_moe_offload_check", &fuse_moe_offload_check);
    m.impl("fuse_moe_offload_status", &fuse_moe_offload_status);
    m.impl("fuse_moe_offload_acknowledge", &fuse_moe_offload_acknowledge);
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
    m.impl("_fuse_moe_offload_test_inject", &fuse_moe_offload_test_inject);
    m.impl("_fuse_moe_offload_test_worker_states", &fuse_moe_offload_test_worker_states);
#endif
    m.impl("fuse_moe_offload_close", &fuse_moe_offload_close);
    m.impl("fuse_moe_offload_stats", &fuse_moe_offload_stats);
CUSTOM_OPS_IMPL_END

// ─────────────────────────────────────────────────────────────────────────────
// Step 7: pybind 入口
// ─────────────────────────────────────────────────────────────────────────────

CUSTOM_OPS_PYBIND_MODULE("RecsysOps/fuse_moe — fused MoE FFN (group GEMM + silu*mul + reduce)")
