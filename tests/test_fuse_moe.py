"""
test_fuse_moe.py — fuse_moe 算子正确性测试（vs 向量化 torch 参考）

用法（自举 sys.path，可在任意 cwd 运行）：
    python tests/test_fuse_moe.py           # 全量
    python tests/test_fuse_moe.py --quick   # 快速冒烟

覆盖：
  - dtype：bf16 / fp16（fp16 仅 SM80+）
  - shape：小批量（S<=128 单 block count 路径）/ 大批量（多 block 路径）、
    H/I ∈ {64, 128, 192?（非 128 倍数走 kTileK=64）}
  - 分布：均匀随机 topk、全部 token 打到单一 expert（极端倾斜）、
    部分专家空闲、topk=1、重复 expert（同一 token 的多个 topk 命中同一 expert）、
    topk=-1 masked（无效路由必须整条忽略）
  - 专家激活 × 数值模式四组合：silu/gelu × fused/staged（bf16 + fp16）
  - 枚举校验：非法/混合类型在加载分组前即拒绝

参考实现刻意区分两种数值契约：
  - fused：gate/up 以 FP32 累加后直接算 act·up，乘积只 round 一次到 Tout；
    down GEMM FP32 累加后 round 到 Tout，scale 加权在 FP32 求和。
  - staged：gate/up → act → 乘积 → down → scale-weighted product 逐级
    round 到 Tout，再以 FP32 累加。参考 GEMM 全部在 CPU FP32 上做
    （禁用 TF32，也不使用 tanh 近似 GELU）；不要求与 kernel 逐 bit 一致。
退出码：0 全部通过；1 存在 FAIL。
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest import mock

# 自举：把包父目录加入 sys.path（远程验证免环境配置）
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402

from custom_ops import ops  # noqa: E402  (import 不编译；开发期首次访问懒加载)

torch.manual_seed(0)
DEV = "cuda"

RESULTS = {"PASS": 0, "FAIL": 0}
FAILURES = []
CASE_REPORTS = []
CHECK_REPORTS = []


def error_metrics(actual, expected):
    """Scalar-only metrics; relative error uses abs(reference).clamp_min(1e-3)."""
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    error = actual - expected
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()
                  and torch.isfinite(error).all())
    absolute = error.abs().flatten()
    relative = (error.abs() / expected.abs().clamp_min(1e-3)).flatten()

    def quantiles(values):
        if not finite or not values.numel():
            return dict(p50=None, p99=None, max=None)
        return dict(p50=values.quantile(.5).item(), p99=values.quantile(.99).item(),
                    max=values.max().item())

    norm = expected.norm().item()
    return dict(abs_error=quantiles(absolute), rel_error=quantiles(relative),
                rmse=error.square().mean().sqrt().item() if finite and error.numel() else None,
                relative_l2=error.norm().item() / norm if finite and norm else None,
                finite=finite, numel=error.numel(), relative_denominator_floor=1e-3)


def _metric_text(metric):
    return json.dumps(metric, sort_keys=True, allow_nan=False)


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    RESULTS[tag] += 1
    CHECK_REPORTS.append(dict(name=name, state=tag.lower(), detail=detail))
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"[{tag}] {name} {detail}")


def _round(tensor, dtype):
    """Round to expert storage dtype, preserving the reference accumulator dtype."""
    return tensor.to(dtype).to(tensor.dtype)


def ref_activation(value, expert_activation):
    """Activation reference in input dtype. GELU is exact erf (never tanh-approx)."""
    if expert_activation == "silu":
        return value * torch.sigmoid(value)
    if expert_activation == "gelu":
        return 0.5 * value * (1.0 + torch.erf(value * math.sqrt(0.5)))
    raise ValueError(expert_activation)


def ref_fuse_moe(x, gate_up_weight, down_weight, topk_ids, topk_scale,
                 expert_activation="silu", numerical_mode="fused", *,
                 accumulator_dtype=torch.float32, output_cast=True,
                 intermediate_rounding=True, stages=None, expert_type="glu"):
    """Per-expert CPU reference (defaults preserve the FP32 numerical contracts).

    accumulator_dtype may be FP32 or FP64; output_cast=False returns the sum
    before the final storage cast. intermediate_rounding=False defines the
    unrounded ideal reference. Optional stages is for small CPU samples only;
    these are reference-model values, NOT CUDA internal accumulators.

    CPU float32 matmul keeps the accumulation reference free of TF32/tensor-core
    shortcuts. Intermediate rounding mimics the kernel's cast chain; the test
    compares with tolerances, never bitwise.
    """
    if accumulator_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulator_dtype must be torch.float32 or torch.float64")
    if numerical_mode not in ("fused", "staged"):
        raise ValueError(numerical_mode)
    if expert_type not in ("glu", "mlp"):
        raise ValueError("expert_type")
    S, H = x.shape
    E, N2, _ = gate_up_weight.shape
    I = down_weight.shape[2]
    if gate_up_weight.shape != (E, I * (2 if expert_type == "glu" else 1), H) or down_weight.shape != (E, H, I):
        raise ValueError("expert_type projection shapes")
    K = topk_ids.shape[1]
    xf = x.detach().cpu().to(accumulator_dtype)
    w1 = gate_up_weight.detach().cpu().to(accumulator_dtype)
    w2 = down_weight.detach().cpu().to(accumulator_dtype)
    ids = topk_ids.detach().cpu().reshape(-1)
    scale = topk_scale.detach().cpu().to(accumulator_dtype).reshape(-1)
    out = torch.zeros(S, H, dtype=accumulator_dtype)

    def materialize(value):
        return _round(value, x.dtype) if intermediate_rounding else value

    def record(stage, value):
        if stages is not None:
            stages.setdefault(stage, []).append(value.detach().clone())
    for expert in range(E):
        flat_rows = torch.nonzero(ids == expert, as_tuple=False).flatten()
        if flat_rows.numel() == 0:
            continue
        tokens = torch.div(flat_rows, K, rounding_mode="floor")
        gu = xf[tokens] @ w1[expert].t()
        if numerical_mode == "staged":
            gu = materialize(gu)                            # gate/up materialize
        record("gate_up", gu)
        gate = gu[:, :I] if expert_type == "glu" else gu
        act = ref_activation(gate, expert_activation)
        if numerical_mode == "staged" and expert_type == "glu":
            act = materialize(act)
        hidden = materialize(act * gu[:, I:] if expert_type == "glu" else act)
        record("activation", act if expert_type == "glu" else hidden)
        record("product" if expert_type == "glu" else "hidden", hidden)
        down = materialize(hidden @ w2[expert].t())         # down output, both modes
        record("down", down)
        weighted = down * scale[flat_rows].unsqueeze(1)
        if numerical_mode == "staged":
            weighted = materialize(weighted)                # weighted product
        record("weighted", weighted)
        out.index_add_(0, tokens, weighted)
    record("final_pre_cast", out)
    result = out.to(x.dtype).to(accumulator_dtype) if output_cast else out
    record("final_post_cast", out.to(x.dtype).to(accumulator_dtype))
    return result


def accuracy_report(*, include_gpu=False):
    """Bounded, deterministic CPU stage model versus unrounded FP64, not instrumentation."""
    generator = torch.Generator(device="cpu").manual_seed(20261020)
    S, H, I, E, K = 5, 64, 64, 2, 2
    base_x = torch.randn(S, H, generator=generator) * .5
    base_w1 = torch.randn(E, 2 * I, H, generator=generator) * .05
    base_w2 = torch.randn(E, H, I, generator=generator) * .05
    ids = torch.tensor([[0, 1], [1, 1], [0, 0], [-1, 1], [-1, -1]], dtype=torch.int32)
    scale = torch.rand(S, K, generator=generator).masked_fill(ids < 0, float("nan"))
    cases = []
    for dtype in (torch.bfloat16, torch.float16):
        x, w1, w2 = (value.to(dtype) for value in (base_x, base_w1, base_w2))
        for activation in ("silu", "gelu"):
            ideal_stages = {}
            ideal = ref_fuse_moe(x, w1, w2, ids, scale, activation,
                                 accumulator_dtype=torch.float64, output_cast=False,
                                 intermediate_rounding=False, stages=ideal_stages)
            for mode in ("fused", "staged"):
                models = {}
                for accumulator in (torch.float32, torch.float64):
                    stages = {}
                    ref_fuse_moe(x, w1, w2, ids, scale, activation, mode,
                                 accumulator_dtype=accumulator, stages=stages)
                    models[str(accumulator)] = {
                        stage: error_metrics(torch.cat(values).flatten(),
                                             torch.cat(ideal_stages[
                                                 "final_pre_cast" if stage == "final_post_cast" else stage
                                             ]).flatten())
                        for stage, values in stages.items()
                    }
                case = dict(dtype=str(dtype), expert_activation=activation,
                            numerical_mode=mode, reference_models=models)
                if include_gpu:
                    output = ops.fuse_moe(x.cuda(), w1.cuda(), w2.cuda(), ids.cuda(),
                                          scale.cuda(), expert_activation=activation,
                                          numerical_mode=mode)
                    case["gpu_final_vs_unrounded_fp64"] = error_metrics(output, ideal)
                    case["gpu_final_vs_cast_fp64"] = error_metrics(output, ideal.to(dtype).double())
                else:
                    case["gpu_state"] = "skipped"
                cases.append(case)
    return dict(
        selection="cpu-fp64", seed=20261020,
        sample_shape=dict(S=S, H=H, I=I, E=E, K=K),
        routes="uniform, duplicate and masked(-1) with NaN masked scales",
        reference="CPU FP64 GEMM + exact erf GELU / sigmoid SiLU, quantized inputs, no intermediate/output rounding",
        stage_semantics="CPU reference-model stage outputs after each contract cast; cumulative errors, not CUDA internal accumulators",
        final_semantics="both final_pre_cast and final_post_cast compare to the SAME unrounded FP64 sum",
        model_semantics={"torch.float32": "FP32 GEMM/activation/sum with contract casts",
                         "torch.float64": "FP64 GEMM/activation/sum with contract casts (rounding-only model)"},
        cases=cases)


def numerical_semantics_fixture(dtype, numerical_mode, h=128, i=256):
    """Sparse single-product GEMMs; select witnesses using quantized CPU outputs.

    No reduction ordering can hide a missing cast. The two duplicate routes
    expose weighted-product rounding before FP32 accumulation. Each candidate
    is a representable gate in [-3, 3], not a hand-tuned error threshold.
    """
    gates = torch.linspace(-3, 3, 12289).to(dtype).float().unique()
    ups = torch.tensor([.50390625, .9921875, 1.0078125, 1.2578125,
                        1.9921875, 2.015625]).to(dtype).float()
    g = gates.repeat_interleave(ups.numel())
    u = ups.repeat(gates.numel())
    # Identity gate for erf discrimination; a single nontrivial scalar multiply
    # in staged also makes the gate/up materialization cast observable.
    gate_factor = 1.0 if numerical_mode == "fused" else 1.0078125
    up_factor, down_factor = 1.0078125, 1.0078125
    scales = torch.tensor([.37, .71], dtype=torch.float32)

    def evaluate(mode, omit=None, tanh=False, perturb=0.0):
        gate, up = g * gate_factor, u * up_factor
        if mode == "staged" and omit != "gate_up":
            gate, up = _round(gate, dtype), _round(up, dtype)
        act = (torch.nn.functional.gelu(gate, approximate="tanh") if tanh
               else ref_activation(gate, "gelu"))
        act = act + perturb * (act.abs() + 1)
        if mode == "staged" and omit != "activation":
            act = _round(act, dtype)
        hidden = act * up
        if omit != "product":
            hidden = _round(hidden, dtype)
        down = hidden * down_factor
        if omit != "down":
            down = _round(down, dtype)
        weighted = down[:, None] * scales
        if mode == "staged" and omit != "weighted":
            weighted = _round(weighted, dtype)
        return _round(weighted.sum(1), dtype)

    expected = evaluate(numerical_mode)
    mutants = {"tanh": evaluate(numerical_mode, tanh=True),
               "omit_product": evaluate(numerical_mode, omit="product"),
               "omit_down": evaluate(numerical_mode, omit="down")}
    if numerical_mode == "staged":
        mutants.update({"fused": evaluate("fused"),
                        **{f"omit_{stage}": evaluate("staged", omit=stage)
                           for stage in ("gate_up", "activation", "weighted")}})
    stable = ((expected == evaluate(numerical_mode, perturb=2e-6))
              & (expected == evaluate(numerical_mode, perturb=-2e-6)))
    witnesses = []
    for name, wrong in mutants.items():
        candidates = torch.nonzero(stable & (expected != wrong)).flatten()
        assert candidates.numel(), f"fixture cannot distinguish {dtype} {numerical_mode} {name}"
        # Maximize the *representable* separation, not a fragile fixed epsilon.
        separation = (expected[candidates] - wrong[candidates]).abs()
        witnesses.append(candidates[separation.argmax()].item())
    selected = torch.tensor(sorted(set(witnesses)))
    # Repeat over a partial M tile, with active features in different N tiles.
    selected = selected.repeat((65 + selected.numel() - 1) // selected.numel())[:65]
    x = torch.zeros(65, h, dtype=dtype)
    x[:, 0], x[:, 1] = g[selected].to(dtype), u[selected].to(dtype)
    w1 = torch.zeros(1, 2 * i, h, dtype=dtype)
    w2 = torch.zeros(1, h, i, dtype=dtype)
    for feature, output in ((0, 0), (i - 1, h - 1)):
        w1[0, feature, 0] = gate_factor
        w1[0, i + feature, 1] = up_factor
        w2[0, output, feature] = down_factor
    ids = torch.zeros(65, 2, dtype=torch.int32)
    scale = scales.expand(65, 2).contiguous()
    out = torch.zeros(65, h)
    out[:, 0] = out[:, -1] = expected[selected]
    alternatives = {}
    for name, wrong in mutants.items():
        alternative = torch.zeros_like(out)
        alternative[:, 0] = alternative[:, -1] = wrong[selected]
        assert not torch.equal(out, alternative), f"lost witness for {name}"
        alternatives[name] = alternative
    return x, w1, w2, ids, scale, out, alternatives


def test_isolated_numerical_semantics():
    """Strong scalar witnesses supplement (not replace) random allclose tests."""
    for dtype in (torch.float16, torch.bfloat16):
        for numerical_mode in ("fused", "staged"):
            x, w1, w2, ids, scale, expected, alternatives = numerical_semantics_fixture(
                dtype, numerical_mode)
            ref = ref_fuse_moe(x, w1, w2, ids, scale, "gelu", numerical_mode)
            torch.testing.assert_close(ref, expected, atol=0, rtol=0)
            out = ops.fuse_moe(x.cuda(), w1.cuda(), w2.cuda(), ids.cuda(), scale.cuda(),
                               expert_activation="gelu", numerical_mode=numerical_mode)
            actual = out.float().cpu()
            check(f"isolated {dtype} gelu/{numerical_mode}",
                  torch.equal(actual, expected),
                  f"max_abs={(actual - expected).abs().max().item():.4g}")
            for name, wrong in alternatives.items():
                check(f"discriminate {dtype} {numerical_mode} vs {name}",
                      not torch.equal(actual, wrong))


def run_case(name, S, H, I, E, K, dtype, mode="uniform",
             expert_activation="silu", numerical_mode="fused", *, repeats=1, expert_type="glu"):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    dev = torch.device(DEV)
    x = (torch.randn(S, H, device=dev, dtype=torch.float32) * 0.5).to(dtype)
    w1 = (torch.randn(E, (2 if expert_type == "glu" else 1) * I, H, device=dev, dtype=torch.float32) * 0.05).to(dtype)
    w2 = (torch.randn(E, H, I, device=dev, dtype=torch.float32) * 0.05).to(dtype)

    if mode == "uniform":
        topk_ids = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
    elif mode == "single_expert":
        topk_ids = torch.full((S, K), E - 1, device=dev, dtype=torch.int32)
    elif mode == "sparse_experts":
        # 只用前 2 个 expert（其余空闲）
        topk_ids = torch.randint(0, 2, (S, K), device=dev, dtype=torch.int32)
    elif mode == "duplicate":
        # 同一 token 的全部 topk 命中同一 expert（重复计数）
        base = torch.randint(0, E, (S, 1), device=dev, dtype=torch.int32)
        topk_ids = base.expand(S, K).contiguous()
    elif mode == "masked":
        topk_ids = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
        topk_ids[::3] = -1
        topk_ids[0] = -1
    else:
        raise ValueError(mode)

    topk_scale = torch.rand(S, K, device=dev, dtype=torch.float32)
    if mode == "duplicate":
        topk_scale = torch.full((S, K), 1.0 / K, device=dev)
    if mode == "masked":
        # 无效槽位的 scale 即使为 NaN 也必须被忽略。
        topk_scale = topk_scale.masked_fill(topk_ids < 0, float("nan"))

    out = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale,
                       expert_activation=expert_activation,
                       numerical_mode=numerical_mode, expert_type=expert_type)
    ref = ref_fuse_moe(x, w1, w2, topk_ids, topk_scale,
                       expert_activation=expert_activation,
                       numerical_mode=numerical_mode, expert_type=expert_type)

    ok_shape = out.shape == x.shape and out.dtype == x.dtype
    actual = out.float().cpu()
    metric = error_metrics(actual, ref)
    # bf16/fp16 输出量化 + kernel 内中间精度：宽松阈值，不要求 bitwise。
    tol = 0.03 if dtype == torch.bfloat16 else 0.02
    ok = ok_shape and metric["finite"] and torch.allclose(actual, ref, atol=0.05, rtol=tol)
    label = (f"{name} [{dtype} S={S} H={H} I={I} E={E} K={K} {mode} "
             f"{expert_activation}/{numerical_mode}]")
    check(label, ok, f"shape_ok={ok_shape} metrics={_metric_text(metric)}")
    case = dict(name=name, shape=dict(S=S, H=H, I=I, E=E, K=K), dtype=str(dtype),
                routing=mode, expert_type=expert_type, expert_activation=expert_activation, numerical_mode=numerical_mode,
                state="passed" if ok else "failed", shape_ok=ok_shape,
                tolerance=dict(atol=0.05, rtol=tol), metrics=metric, repeat_runs=[])
    CASE_REPORTS.append(case)
    # Same tensors/routes for every call; allow last-bit reduction differences
    # under the original accuracy tolerance, never relax it or require bitwise.
    for repeat in range(1, repeats):
        repeated = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale,
                                expert_activation=expert_activation,
                                numerical_mode=numerical_mode, expert_type=expert_type)
        repeat_actual = repeated.float().cpu()
        vs_ref = error_metrics(repeat_actual, ref)
        vs_first = error_metrics(repeat_actual, actual)
        repeat_shape = repeated.shape == x.shape and repeated.dtype == x.dtype
        healthy = (repeat_shape and vs_ref["finite"] and vs_first["finite"]
                   and torch.allclose(repeat_actual, ref, atol=0.05, rtol=tol)
                   and torch.allclose(repeat_actual, actual, atol=0.05, rtol=tol))
        case["repeat_runs"].append(dict(iteration=repeat + 1,
                                       state="passed" if healthy else "failed",
                                       shape_ok=repeat_shape, vs_reference=vs_ref, vs_first=vs_first,
                                       exact_equal=torch.equal(repeat_actual, actual)))
        if not healthy:
            case["state"] = "failed"
        check(f"{label} repeat={repeat + 1}", healthy,
              f"vs_reference={_metric_text(vs_ref)} vs_first={_metric_text(vs_first)}")
    return case


def test_mode_resolution_and_validation():
    """CPU-only: strict enum parsing happens before any group load/compile."""
    from custom_ops.recsys import _resolve_fuse_moe_modes

    for activation, activation_id in (("silu", 0), ("gelu", 1)):
        for numerical, numerical_id in (("fused", 0), ("staged", 1)):
            check(f"mode encode {activation}/{numerical}",
                  _resolve_fuse_moe_modes(activation, numerical)
                  == (activation_id, numerical_id))
    invalid = {
        "expert_activation": ("", "GELU", "gelu ", "none", 1, 0, True, None, [], {}),
        "numerical_mode": ("", "Staged", "fused ", "legacy", 1, 0, True, None, [], {}),
    }
    x = torch.zeros(1, 64)
    w1 = torch.zeros(1, 128, 64)
    w2 = torch.zeros(1, 64, 64)
    ids = torch.zeros(1, 1, dtype=torch.int32)
    scale = torch.zeros(1, 1)
    with (mock.patch.object(ops, "_dispatch", side_effect=AssertionError("group dispatch")),
          mock.patch.object(ops, "_group", side_effect=AssertionError("group load"))):
        for field, values in invalid.items():
            for value in values:
                kwargs = {field: value}
                try:
                    ops.fuse_moe(x, w1, w2, ids, scale, **kwargs)
                except ValueError as error:
                    check(f"reject {field}={value!r}", field in str(error))
                except Exception as error:  # noqa: BLE001 - wrong failure class
                    check(f"reject {field}={value!r}", False, type(error).__name__)
                else:
                    check(f"reject {field}={value!r}", False, "no error")


def test_geglu_staged_matrix(quick=False):
    """GEGLU/SwiGLU × fused/staged for both dtypes and route patterns."""
    cases = [
        (129, 128, 128, 8, 4, "uniform"),
        (64, 256, 192, 8, 2, "single_expert"),
        (97, 128, 64, 8, 4, "duplicate"),
        (256, 128, 128, 16, 2, "masked"),
    ]
    if quick:
        cases = cases[:2]
    for dtype in (torch.bfloat16, torch.float16):
        for (S, H, I, E, K, mode) in cases:
            for expert_activation in ("silu", "gelu"):
                for numerical_mode in ("fused", "staged"):
                    run_case("geglu", S, H, I, E, K, dtype, mode,
                             expert_activation=expert_activation,
                             numerical_mode=numerical_mode)


def test_boundary_and_repeat_health():
    """V07/V08: small samples across both dtypes and all four mode pairs."""
    cases = [
        ("V07_K128", 3, 64, 64, 8, 128, "duplicate"),
        ("V07_E512", 3, 64, 64, 512, 2, "uniform"),
        ("V07_E1_min_HI", 1, 64, 64, 1, 1, "uniform"),
        ("V08_repeat_health", 7, 64, 64, 4, 4, "uniform"),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for name, S, H, I, E, K, routing in cases:
            for activation in ("silu", "gelu"):
                for numerical in ("fused", "staged"):
                    run_case(name, S, H, I, E, K, dtype, routing,
                             expert_activation=activation, numerical_mode=numerical,
                             repeats=3 if name == "V08_repeat_health" else 1)


def mlp_numerical_semantics_fixture(dtype, numerical_mode, expert_activation="gelu", h=64, i=64):
    """Single projection scalar witnesses, including partial M/N tiles."""
    values = torch.linspace(-3, 3, 12289).to(dtype).float().unique()
    factor = 1.0078125
    scales = torch.tensor([.37, -.71])

    def evaluate(omit=None, tanh=False, activation=None):
        project = values * factor
        if numerical_mode == "staged" and omit != "project":
            project = _round(project, dtype)
        act = (torch.nn.functional.gelu(project, approximate="tanh") if tanh
               else ref_activation(project, activation or expert_activation))
        hidden = act if omit == "activation" else _round(act, dtype)
        down = hidden * factor
        if omit != "down":
            down = _round(down, dtype)
        weighted = down[:, None] * scales
        if numerical_mode == "staged" and omit != "weighted":
            weighted = _round(weighted, dtype)
        return _round(weighted.sum(1), dtype)

    expected = evaluate()
    mutants = {"omit_activation": evaluate("activation"), "omit_down": evaluate("down"),
               "wrong_activation": evaluate(activation="silu" if expert_activation == "gelu" else "gelu")}
    if expert_activation == "gelu":
        mutants["tanh"] = evaluate(tanh=True)
    if numerical_mode == "staged":
        mutants.update(omit_project=evaluate("project"), omit_weighted=evaluate("weighted"))
    selected = []
    for name, wrong in mutants.items():
        candidates = torch.nonzero(expected != wrong).flatten()
        assert candidates.numel(), f"MLP witness missing: {dtype}/{numerical_mode}/{name}"
        selected.append(candidates[(expected[candidates] - wrong[candidates]).abs().argmax()].item())
    selected = torch.tensor(sorted(set(selected))).repeat(65)[:65]
    x = torch.zeros(65, h, dtype=dtype)
    x[:, 0] = values[selected]
    w1, w2 = torch.zeros(1, i, h, dtype=dtype), torch.zeros(1, h, i, dtype=dtype)
    for feature, output in ((0, 0), (i - 1, h - 1)):
        w1[0, feature, 0] = factor
        w2[0, output, feature] = factor
    ids = torch.zeros(65, 2, dtype=torch.int32)
    scale = scales.expand(65, 2).contiguous()
    out = torch.zeros(65, h)
    out[:, 0] = out[:, -1] = expected[selected]
    alternatives = {}
    for name, wrong in mutants.items():
        alternative = torch.zeros_like(out)
        alternative[:, 0] = alternative[:, -1] = wrong[selected]
        alternatives[name] = alternative
    return x, w1, w2, ids, scale, out, alternatives


def test_mlp_cpu_contracts():
    from custom_ops.recsys import _resolve_fuse_moe_expert_type
    for name, code in (("glu", 0), ("mlp", 1)):
        check(f"expert type {name}", _resolve_fuse_moe_expert_type(name) == code)
    inputs = (torch.zeros(1, 64), torch.zeros(1, 64, 64), torch.zeros(1, 64, 64),
              torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1))
    with mock.patch.object(ops, "_dispatch", side_effect=AssertionError("dispatch")):
        for value in ("", "MLP", "mlp ", "linear", -1, 0, 1, 2, True, None, [], {}):
            try:
                ops.fuse_moe(*inputs, expert_type=value)
            except ValueError as error:
                check(f"invalid expert_type={value!r}", "expert_type" in str(error))
            else:
                check(f"invalid expert_type={value!r}", False)
    for dtype in (torch.float16, torch.bfloat16):
        for activation in ("silu", "gelu"):
            for mode in ("fused", "staged"):
                x, w1, w2, ids, scale, expected, alternatives = mlp_numerical_semantics_fixture(dtype, mode, activation)
                stages = {}
                actual = ref_fuse_moe(x, w1, w2, ids, scale, activation, mode,
                                      expert_type="mlp", stages=stages)
                check(f"MLP scalar {dtype}/{activation}/{mode}", torch.equal(actual, expected))
                check(f"MLP single activation store {dtype}/{activation}/{mode}",
                      "product" not in stages and torch.equal(stages["hidden"][0], stages["activation"][0]))
                for name, wrong in alternatives.items():
                    check(f"MLP witness {dtype}/{activation}/{mode}/{name}", not torch.equal(actual, wrong))
                glu = ref_fuse_moe(x, torch.cat((w1, w1), 1), w2, ids, scale, activation, mode)
                check(f"MLP not fake GLU {dtype}/{activation}/{mode}", not torch.equal(actual, glu))


def test_mlp_matrix(quick=False):
    cases = [(7, 64, 64, 8, 2, "uniform"), (129, 128, 192, 8, 4, "masked"),
             (65, 192, 128, 8, 4, "duplicate"), (3, 64, 64, 8, 128, "duplicate"),
             (3, 64, 64, 512, 2, "uniform"), (1, 64, 64, 1, 1, "uniform")]
    for dtype in (torch.bfloat16, torch.float16):
        for activation in ("silu", "gelu"):
            for numerical in ("fused", "staged"):
                for s, h, i, e, k, routing in (cases[:2] if quick else cases):
                    run_case("mlp", s, h, i, e, k, dtype, routing, activation, numerical,
                             expert_type="mlp", repeats=2)
                x, w1, w2, ids, scale, expected, alternatives = mlp_numerical_semantics_fixture(dtype, numerical, activation)
                actual = ops.fuse_moe(x.cuda(), w1.cuda(), w2.cuda(), ids.cuda(), scale.cuda(),
                                     expert_activation=activation, numerical_mode=numerical, expert_type="mlp").float().cpu()
                check(f"MLP native scalar {dtype}/{activation}/{numerical}", torch.equal(actual, expected))
                for name, wrong in alternatives.items():
                    check(f"MLP native witness {name}", not torch.equal(actual, wrong))
    for code in (-1, 2):
        try:
            torch.ops.recsys_ops.fuse_moe(x.cuda(), w1.cuda(), w2.cuda(), ids.cuda(), scale.cuda(), 0, 0, code)
        except RuntimeError as error:
            check(f"native illegal expert_type={code}", "expert_type" in str(error))
        else:
            check(f"native illegal expert_type={code}", False)
    glu_w1 = torch.cat((w1, w1), 1).cuda()
    default = torch.ops.recsys_ops.fuse_moe(x.cuda(), glu_w1, w2.cuda(), ids.cuda(), scale.cuda())
    explicit = torch.ops.recsys_ops.fuse_moe(x.cuda(), glu_w1, w2.cuda(), ids.cuda(), scale.cuda(), 0, 0, 0)
    check("native public default GLU", torch.equal(default, explicit))


def _load_storage_benchmark():
    """Import the sibling benchmark by path; no repo-root PYTHONPATH required."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "benchmark" / "benchmark_fuse_moe.py"
    spec = importlib.util.spec_from_file_location(
        "custom_ops_storage_benchmark_host_contracts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import storage benchmark from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_storage_benchmark_host_contracts():
    """CPU-only checks for the storage suite: reference math, flops convention,
    output/reference conversion and offload stat validation. Never dispatches a
    native op, initializes CUDA, or reports timings/performance."""
    bench = _load_storage_benchmark()
    # The performance baseline uses ordinary dtype pointwise ops, not extra
    # conversion kernels inserted around activation/multiplication.
    eager_checks = []
    with torch.no_grad():
        for dtype in (torch.float16, torch.bfloat16):
            for architecture in ("glu", "mlp"):
                values = bench.storage_inputs((17, 64, 64, 8, 2), dtype, 42, "uniform", architecture)
                for activation in ("silu", "gelu"):
                    model = bench.storage_module(values[1], values[2], architecture, activation)
                    actual = model(values[0], values[3], values[4])
                    expected = bench.stagewise_cpu_ref(*values, architecture, activation, "staged", 17)
                    eager_checks.append(torch.allclose(actual, expected, atol=.002, rtol=.02))
    check("storage ordinary dtype eager matches staged CPU contract", all(eager_checks))

    import inspect
    flops_source = inspect.getsource(bench.bench_storage)
    check("storage flops convention glu=6/mlp=4 in resident rows",
          '(6 if expert_type == "glu" else 4) * shape[0] * shape[4] * shape[1] * shape[2]'
          in flops_source and "effective_flops_formula" in flops_source)

    summary = bench.summarize_ms([1, 2, 4])
    check("storage summarize_ms raw/p50/p95/median",
          summary["samples_ms"] == [1, 2, 4] and summary["p50_ms"] == 2
          and summary["median_ms"] == 2 and abs(summary["p95_ms"] - 3.8) < 1e-12,
          f"p50={summary['p50_ms']} p95={summary['p95_ms']} median={summary['median_ms']}")

    fixture_shape = (17, 64, 64, 8, 2)
    check("storage_shape accepts positive/layout limits",
          bench.storage_shape("17,64,64,8,2") == fixture_shape)
    invalid_shapes = ("17,64,64,8", "17,64,64,8,2,1", "0,64,64,8,2",
                      "17,63,64,8,2", "17,64,63,8,2", "17,64,64,513,2",
                      "17,64,64,512,129", "17,64,64,8,9", "x,64,64,8,2")
    rejected = []
    for value in invalid_shapes:
        try:
            bench.storage_shape(value)
        except argparse.ArgumentTypeError:
            rejected.append(value)
    check("storage_shape rejects malformed/nonpositive/limit violations",
          len(rejected) == len(invalid_shapes),
          f"accepted={sorted(set(invalid_shapes) - set(rejected))}")

    dtype = torch.bfloat16
    cpu = bench.storage_inputs(fixture_shape, dtype, 20261021, "uniform", "glu")
    x, w1, w2, ids, scale = cpu
    check("storage_inputs canonical shape/dtype/route contract",
          x.shape == (17, 64) and x.dtype == dtype
          and w1.shape == (8, 128, 64) and w2.shape == (8, 64, 64)
          and ids.shape == (17, 2) and ids.dtype == torch.int32
          and scale.shape == (17, 2) and scale.dtype == torch.float32
          and bool((ids >= 0).all()) and bool((ids < 8).all())
          and all(row.unique().numel() == row.numel() for row in ids)
          and bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
          and torch.allclose(scale.sum(1), torch.ones(17), atol=1e-6))
    one = bench.storage_inputs((17, 64, 64, 8, 1), dtype, 20261022, "uniform", "glu")
    check("storage_inputs K=1 keeps raw positive scale",
          one[3].shape == (17, 1) and one[4].shape == (17, 1)
          and bool((one[4] > 0).all()) and not torch.allclose(one[4], one[4][:1]))
    hx, hw1, hw2, hids, hscale = bench.storage_inputs(
        fixture_shape, dtype, 20261023, "hotset", "mlp")
    check("storage_inputs hotset stays in experts 0/1 and K distinct",
          hw1.shape == (8, 64, 64) and hids.shape == (17, 2)
          and set(hids.unique().tolist()) <= {0, 1}
          and all(row.unique().numel() == row.numel() for row in hids)
          and bool((hscale > 0).all()))

    check("storage unique CPU weight bytes dedupes shared storages",
          bench.unique_cpu_weight_bytes([x, x[1:3]]) == x.untyped_storage().nbytes())

    # Reference contract: independent FP32 stagewise matmuls (storage rounding)
    # must agree with the full per-expert reference on the sampled prefix.
    sample_tokens = 9
    max_abs = 0.0
    mismatches = []
    for ref_dtype in (torch.bfloat16, torch.float16):
        for expert_type in ("glu", "mlp"):
            ref_cpu = bench.storage_inputs(fixture_shape, ref_dtype, 20261021,
                                           "uniform", expert_type)
            for activation in ("silu", "gelu"):
                for numerical in ("fused", "staged"):
                    sampled = bench.stagewise_cpu_ref(
                        *ref_cpu, expert_type, activation, numerical, sample_tokens)
                    expected = ref_fuse_moe(
                        *ref_cpu, activation, numerical, expert_type=expert_type
                    )[:sample_tokens].to(ref_dtype)
                    ok = (sampled.shape == expected.shape
                          and sampled.dtype == expected.dtype
                          and torch.allclose(sampled, expected, atol=.002, rtol=.02))
                    if not ok:
                        mismatches.append(f"{ref_dtype}/{expert_type}/{activation}/{numerical}")
                    if sampled.numel():
                        max_abs = max(max_abs, (sampled.float() - expected.float())
                                      .abs().max().item())
    check("storage CPU stage ref matches full ref first n tokens (2 dtype x 2 act x 2 mode x glu/mlp)",
          not mismatches, f"sample={sample_tokens} max_abs={max_abs:.4g} bad={mismatches}")

    # An MLP must not be silently evaluated through a fake GLU projection.
    fake_glu_mismatches = []
    for ref_dtype in (torch.bfloat16, torch.float16):
        ref_cpu = bench.storage_inputs(fixture_shape, ref_dtype, 20261021,
                                       "uniform", "mlp")
        mlp = bench.stagewise_cpu_ref(*ref_cpu, "mlp", "gelu", "staged", sample_tokens)
        fake_glu = bench.stagewise_cpu_ref(
            ref_cpu[0], torch.cat((ref_cpu[1], ref_cpu[1]), dim=1), *ref_cpu[2:],
            "glu", "gelu", "staged", sample_tokens)
        if torch.allclose(fake_glu, mlp, atol=.002, rtol=.02):
            fake_glu_mismatches.append(str(ref_dtype))
    check("storage MLP reference discriminates fake GLU", not fake_glu_mismatches,
          f"indistinguishable={fake_glu_mismatches}")

    # Host output/reference conversion: full shape/dtype/finite, first n sample.
    reference = torch.linspace(-1, 1, sample_tokens * 64).reshape(sample_tokens, 64)
    reference = reference.to(dtype).float()
    output = torch.zeros(17, 64, dtype=dtype)
    output[:sample_tokens] = reference.to(dtype)
    try:
        evidence = bench.check_storage_output(output, (17, 64), dtype, reference, .002, .02)
        check("storage check_storage_output samples first n of full output",
              evidence["passed"] and evidence["sample_tokens"] == sample_tokens
              and evidence["max_abs"] == 0)
    except Exception as error:  # noqa: BLE001 - report the contract failure type
        check("storage check_storage_output samples first n of full output", False,
              f"{type(error).__name__}: {error}")
    bad_outputs = {
        "shape": (torch.zeros(16, 64, dtype=dtype), reference),
        "dtype": (torch.zeros(17, 64, dtype=torch.float32), reference),
        "nonfinite": (output.clone(), reference),
        "value": (output + .25, reference),
    }
    bad_outputs["nonfinite"][0][0, 0] = float("nan")
    rejected = []
    for name, (candidate, expected) in bad_outputs.items():
        try:
            bench.check_storage_output(candidate, (17, 64), dtype, expected, .002, .02)
        except (AssertionError, RuntimeError):
            rejected.append(name)
    check("storage check_storage_output rejects shape/dtype/nonfinite/value errors",
          len(rejected) == len(bad_outputs), f"accepted={sorted(set(bad_outputs) - set(rejected))}")

    # Offload stat validation, exactly the public host-side evidence contract.
    def stats_case(**overrides):
        values = dict(active_experts=8, waves=1, weight_h2d_bytes=8 * 16 * 1024,
                      gpu_charged_bytes=1024, ram_charged_bytes=2048, pinned_bytes=512,
                      resident_cache_bytes=0, num_banks=1, wave_capacity=64)
        values.update(overrides)
        return values

    disabled_args = SimpleNamespace(offload_cache="disabled", gpu_budget_mib=128,
                                    ram_budget_mib=4, pinned_budget_mib=1)

    def validate(label, stats, expect_ok, *, args=disabled_args, active=8,
                 expert_bytes=16 * 1024, resident=0):
        try:
            bench.validate_offload_stats(stats, args, active, expert_bytes, resident)
        except (AssertionError, RuntimeError) as error:
            check(label, not expect_ok, f"{type(error).__name__}: {error}")
        except Exception as error:  # noqa: BLE001 - wrong failure class
            check(label, False, f"unexpected {type(error).__name__}: {error}")
        else:
            check(label, expect_ok)

    validate("storage stats accepts actual bank/cap64", stats_case(), True)
    validate("storage stats accepts wave ceil with cap2",
             stats_case(waves=4, wave_capacity=2), True)
    missing = stats_case()
    missing.pop("num_banks")
    validate("storage stats requires num_banks schema", missing, False)
    validate("storage stats rejects 0 H2D bytes", stats_case(weight_h2d_bytes=0), False)
    validate("storage stats rejects wrong active expert size",
             stats_case(active_experts=7), False)
    validate("storage stats rejects wrong wave ceil", stats_case(waves=2), False)
    validate("storage stats rejects zero wave capacity", stats_case(wave_capacity=0), False)
    validate("storage stats rejects dynamic cache",
             stats_case(dynamic_cache_bytes=1), False)
    validate("storage stats rejects wrong resident cache",
             stats_case(resident_cache_bytes=512), False)
    validate("storage stats rejects GPU budget overrun",
             stats_case(gpu_charged_bytes=129 * 1024**2), False)
    validate("storage stats rejects RAM budget overrun",
             stats_case(ram_charged_bytes=5 * 1024**2), False)
    validate("storage stats rejects pinned budget overrun",
             stats_case(pinned_bytes=2 * 1024**2), False)

    static_args = SimpleNamespace(offload_cache="static_hot", gpu_budget_mib=128,
                                  ram_budget_mib=4, pinned_budget_mib=1)
    expert_bytes = 16 * 1024
    resident = 2 * ((expert_bytes + 511) // 512 * 512)
    static_stats = stats_case(active_experts=2, waves=1, weight_h2d_bytes=0,
                              resident_cache_bytes=resident)
    validate("storage static_hot expects 0 H2D and 2 aligned weights",
             static_stats, True, args=static_args, active=2,
             expert_bytes=expert_bytes, resident=resident)
    validate("storage static_hot rejects redundant H2D",
             {**static_stats, "weight_h2d_bytes": expert_bytes}, False,
             args=static_args, active=2, expert_bytes=expert_bytes, resident=resident)
    validate("storage static_hot rejects wrong resident cap",
             {**static_stats, "resident_cache_bytes": expert_bytes}, False,
             args=static_args, active=2, expert_bytes=expert_bytes, resident=resident)
    validate("storage static_hot rejects dynamic cache",
             {**static_stats, "dynamic_cache_bytes": expert_bytes}, False,
             args=static_args, active=2, expert_bytes=expert_bytes, resident=resident)

    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        with bench.strict_math():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            raise RuntimeError("exercise restore path")
    except RuntimeError:
        pass
    check("storage strict_math restores precision/TF32 after exception",
          torch.get_float32_matmul_precision() == precision
          and torch.backends.cuda.matmul.allow_tf32 == tf32)


def _run(args, report):

    # Enum parsing is CPU-only and must reject before loading/compiling a group.
    test_mode_resolution_and_validation()
    test_mlp_cpu_contracts()
    test_storage_benchmark_host_contracts()

    gpu_available = not args.cpu_only and torch.cuda.is_available()
    gpu_supported = gpu_available and torch.cuda.get_device_capability(0)[0] >= 8
    if args.accuracy_report:
        # CPU evidence is available even when no library/GPU can be loaded.
        report["accuracy_report"] = accuracy_report()
    if not gpu_supported:
        reason = ("--cpu-only selected" if args.cpu_only else
                  "CUDA unavailable" if not gpu_available else "SM80+ required")
        report["gpu"] = dict(state="skipped", reason=reason)
        print(f"{reason}; GPU correctness matrix SKIPPED (not passed)")
        print(f"TOTAL CPU checks: {RESULTS['PASS']} PASS, {RESULTS['FAIL']} FAIL; GPU SKIPPED")
        return

    report["gpu"] = dict(state="running")
    if args.ops_library:
        ops.load_precompiled("fuse_moe", args.ops_library)
    assert ops.is_available(), "extension load failed"

    dtypes = [torch.bfloat16, torch.float16]
    if args.accuracy_report:
        report["accuracy_report"] = accuracy_report(include_gpu=True)
        print("ACCURACY_REPORT " + _metric_text(report["accuracy_report"]))

    if args.mlp_only:
        test_mlp_matrix(args.quick)
        report["gpu"]["state"] = "failed" if RESULTS["FAIL"] else "passed"
        return

    # ── 基础 shape 矩阵 ──────────────────────────────────────────────────────
    cases = [
        # (S, H, I, E, K, mode)
        (1,    64,  64,  8, 1, "uniform"),        # 最小 shape / topk=1
        (7,    64,  64,  8, 2, "uniform"),        # 小批量单 block count 路径
        (128,  128, 128, 16, 4, "uniform"),       # 单 block count 边界
        (129,  128, 128, 16, 4, "uniform"),       # 多 block count 路径边界
        (1024, 128, 128, 16, 4, "uniform"),       # 大批量
        (512,  512, 192, 32, 8, "uniform"),       # 大 H、I 非 128 倍数(kTileK=64)
        (333,  192,  64, 64, 2, "uniform"),       # H 非 128 倍数、E=64
        (64,   256, 256,  4, 8, "single_expert"), # 极端倾斜
        (256,  128, 128, 32, 2, "sparse_experts"),# 多数 expert 空闲
        (97,   128,  64,  8, 4, "duplicate"),     # 重复 expert
    ]
    if args.quick:
        cases = cases[:4]

    for dtype in dtypes:
        for (S, H, I, E, K, mode) in cases:
            run_case("moe", S, H, I, E, K, dtype, mode)

    # ── GEGLU / staged 四组合（独立 shape 子集，避免放大全部用例）─────────
    test_geglu_staged_matrix(args.quick)
    test_boundary_and_repeat_health()
    test_isolated_numerical_semantics()
    test_mlp_matrix(args.quick)

    # ── 边界：直接 fuse_moe 空 token 集合由 op 层拒绝（offload 支持 S=0，见
    #    test_fuse_moe_offload.py::GEGLUTests）───────────────────────────────
    try:
        ops.fuse_moe(torch.empty(0, 64, device=DEV, dtype=torch.bfloat16),
                     torch.randn(8, 128, 64, device=DEV, dtype=torch.bfloat16),
                     torch.randn(8, 64, 64, device=DEV, dtype=torch.bfloat16),
                     torch.empty(0, 2, device=DEV, dtype=torch.int32),
                     torch.empty(0, 2, device=DEV, dtype=torch.float32))
        check("empty_input_rejected", False, "no error raised")
    except Exception:
        check("empty_input_rejected", True)

    print("\n" + "=" * 60)
    print(f"TOTAL: {RESULTS['PASS']} PASS, {RESULTS['FAIL']} FAIL")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print("  -", f)
    report["gpu"]["state"] = "failed" if RESULTS["FAIL"] else "passed"


def _library_info(requested):
    # Observe already loaded groups only: collecting evidence must never JIT.
    group = getattr(ops, "_groups", {}).get("fuse_moe")
    loaded = group.library_path if group is not None else None
    path = Path(loaded or requested).expanduser().resolve() if loaded or requested else None
    info = dict(requested=requested, loaded=str(loaded) if loaded else None,
                path=str(path) if path else None, loaded_successfully=loaded is not None)
    if path is not None and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        info.update(size_bytes=path.stat().st_size, sha256=digest.hexdigest())
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mlp-only", action="store_true", help="Only MLP GPU matrix; CPU contracts still run")
    parser.add_argument("--ops-library", help="Explicit precompiled fuse_moe library; no implicit JIT")
    parser.add_argument("--json", nargs="?", const="-", metavar="PATH",
                        help="Write full structured evidence to PATH, or stdout when omitted / '-'")
    parser.add_argument("--accuracy-report", nargs="?", const="cpu-fp64", choices=("cpu-fp64",),
                        help="Small CPU FP64 stage-reference report (also GPU final output when available)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run CPU validation/reference only; never load/build library or run GPU")
    args = parser.parse_args()
    if args.json and args.json != "-" and os.path.lexists(Path(args.json).expanduser()):
        parser.error("refusing to overwrite existing correctness evidence: " + args.json)
    RESULTS.update(PASS=0, FAIL=0)
    FAILURES.clear()
    CASE_REPORTS.clear()
    CHECK_REPORTS.clear()
    torch.manual_seed(0)
    started = datetime.now(timezone.utc)
    report = dict(schema_version=1, started_at=started.isoformat(),
                  execution_date=started.date().isoformat(), seed=0,
                  options=vars(args), gpu=dict(state="not_run"))
    failed = False
    # --json stdout stays machine-readable; all human progress goes to stderr.
    with redirect_stdout(sys.stderr if args.json == "-" else sys.stdout):
        try:
            _run(args, report)
        except Exception as error:  # Emit incomplete evidence rather than lose all previous cases.
            failed = True
            report["error"] = dict(type=type(error).__name__, message=str(error))
            if report["gpu"]["state"] == "running":
                report["gpu"]["state"] = "failed"
            print(f"ERROR: {type(error).__name__}: {error}")
        if args.accuracy_report and report["gpu"]["state"] == "skipped":
            print("ACCURACY_REPORT " + _metric_text(report["accuracy_report"]))
    report.update(
        finished_at=datetime.now(timezone.utc).isoformat(), cases=CASE_REPORTS,
        checks=CHECK_REPORTS, failures=FAILURES,
        summary=dict(cpu_and_gpu_checks=dict(RESULTS), matrix_cases=len(CASE_REPORTS),
                     matrix_passed=sum(c["state"] == "passed" for c in CASE_REPORTS),
                     matrix_failed=sum(c["state"] == "failed" for c in CASE_REPORTS),
                     gpu_state=report["gpu"]["state"],
                     state="failed" if failed or RESULTS["FAIL"] else
                           "gpu_skipped" if report["gpu"]["state"] == "skipped" else "passed"),
        environment=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                         cuda_version=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         gpu_name=torch.cuda.get_device_name(0) if not args.cpu_only and torch.cuda.is_available() else None,
                         gpu_capability=list(torch.cuda.get_device_capability(0)) if not args.cpu_only and torch.cuda.is_available() else None),
        library=_library_info(args.ops_library))
    if args.json:
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if args.json == "-":
            print(text)
        else:
            output = Path(args.json).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
    return 1 if failed or RESULTS["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
