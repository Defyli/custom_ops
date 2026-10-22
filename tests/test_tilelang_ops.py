"""
tests/test_tilelang_ops.py — Tile-lang 后端（custom_ops.tilelang_ops）正确性测试

覆盖：
- mixed_gemm：fp8 / int8 residual、x fp32/bf16、bias / 激活 / out_dtype 组合、
  非 tile 对齐 M、非 64 倍数 K
- fuse_moe：均匀/倾斜路由、单专家、重复路由、多专家、bf16/fp16、
  与 CUDA 版交叉对拍
- swiglu：配对 GEMM + silu·mul、任意 M
- fa：冒烟（完整对拍见 benchmark/fa_tilelang.py --test）
- backend 选路：ops.<op>(..., backend="tilelang") 与 tilelang_ops 等价

运行：
    python tests/test_tilelang_ops.py
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

from custom_ops import ops, split_mixed_precision_weight  # noqa: E402
from custom_ops import tilelang_ops  # noqa: E402

torch.manual_seed(42)

ATOL = 2e-2
RTOL = 2e-2

_results = []


def check(name, out, ref, atol=ATOL, rtol=RTOL):
    # MoE references intentionally execute on CPU FP32 to avoid TF32.
    # Compare on one device, including CUDA-vs-CUDA cross-backend checks.
    out = out.detach().float().cpu()
    ref = ref.detach().float().cpu()
    err = (out - ref).abs()
    max_err = err.max().item()
    ref_scale = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_scale) < rtol
    print(f"[{'PASS' if passed else 'FAIL'}] {name:56s} max_err={max_err:.4e}")
    _results.append(passed)
    return passed


class FuseMoETileLangContracts(unittest.TestCase):
    """Host contracts with an identity JIT decorator; no TileLang/GPU required.

    This checks failure policy and input validation only, never claims a kernel
    was compiled or emulates a successful GPU execution.
    """

    @classmethod
    def setUpClass(cls):
        language = ModuleType("tilelang.language")
        tilelang = ModuleType("tilelang")
        tilelang.language = language
        tilelang.jit = lambda *args, **kwargs: lambda factory: factory
        spec = importlib.util.spec_from_file_location(
            "_fuse_moe_host_contract", Path(_PKG_ROOT) / "tilelang_ops/_fuse_moe.py")
        cls.module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"tilelang": tilelang,
                                           "tilelang.language": language}):
            spec.loader.exec_module(cls.module)

    def inputs(self):
        return [torch.zeros(2, 64, dtype=torch.bfloat16),
                torch.zeros(2, 128, 64, dtype=torch.bfloat16),
                torch.zeros(2, 64, 64, dtype=torch.bfloat16),
                torch.zeros(2, 2, dtype=torch.int32),
                torch.ones(2, 2, dtype=torch.float32)]

    def test_mlp_invalid_types_and_shapes_before_jit(self):
        with mock.patch.object(self.module, "_make", side_effect=AssertionError("JIT")):
            for value in ("", "MLP", "mlp ", -1, 0, 1, 2, True, None, [], {}):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "expert_type"):
                    self.module.fuse_moe_tilelang(*self.inputs(), expert_type=value)
            for architecture in ("glu", "mlp"):
                values = self.inputs()
                if architecture == "glu":
                    values[1] = values[1][:, :64].contiguous()
                with self.assertRaisesRegex(ValueError, "weight|shape|expert_type"):
                    self.module.fuse_moe_tilelang(*values, expert_type=architecture)

    def test_mlp_reference_single_projection_matches_independent_cpu(self):
        from test_fuse_moe import mlp_numerical_semantics_fixture, ref_fuse_moe
        for dtype in (torch.float16, torch.bfloat16):
            for activation in ("silu", "gelu"):
                for numerical in ("fused", "staged"):
                    x,w1,w2,ids,scale,expected,_ = mlp_numerical_semantics_fixture(dtype,numerical,activation)
                    actual = _moe_ref(x,w1,w2,ids,scale,activation,numerical,"mlp")
                    # MLP has the native Down store in both numerical modes.
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertFalse(torch.equal(actual,ref_fuse_moe(x,torch.cat((w1,w1),1),w2,ids,scale,activation,numerical)))

    def test_no_erf_fails_before_prim_func_or_compilation(self):
        with self.assertRaisesRegex(NotImplementedError, "exact GELU cannot be compiled"):
            self.module._moe_gemm1_factory(2, 64, 64, 64, 2, "bfloat16",
                                           expert_activation="gelu")
        self.assertEqual(self.module._KERNEL_CACHE, {})

    def test_intrinsic_and_extern_are_the_only_exact_options(self):
        check = self.module._exact_gelu_uses_erf
        self.assertTrue(check("gelu", SimpleNamespace(erf=lambda value: value)))
        self.assertFalse(check("gelu", SimpleNamespace(call_extern=lambda *args: None)))
        for language in (SimpleNamespace(), SimpleNamespace(erf=None, call_extern=None),
                         SimpleNamespace(erf=False, call_extern="erff",
                                         tanh=lambda value: value)):
            with self.subTest(language=language):
                with self.assertRaises(NotImplementedError):
                    check("gelu", language)
                self.assertFalse(check("silu", language))

    def test_cpu_and_empty_input_rejected_before_jit(self):
        with mock.patch.object(self.module, "_make", side_effect=AssertionError("JIT")):
            with self.assertRaisesRegex(ValueError, "CUDA tensors"):
                self.module.fuse_moe_tilelang(*self.inputs())
            values = self.inputs()
            values[0] = values[0][:0]
            with self.assertRaisesRegex(ValueError, "positive S and H"):
                self.module.fuse_moe_tilelang(*values)

    def test_n_panels_stay_within_expert_projection(self):
        base = self.module._default_config(7, 2, 2)
        for intermediate, hidden, expected in ((64, 64, (64, 64)),
                                                (128, 128, (128, 128)),
                                                (192, 256, (64, 128)),
                                                (256, 192, (128, 64))):
            with self.subTest(intermediate=intermediate, hidden=hidden):
                resolved = self.module._normalize_moe_config(base, intermediate, hidden)
                self.assertEqual((resolved["block_N1"], resolved["block_N2"]), expected)
                self.assertEqual(base["block_N1"], 128)
                self.assertEqual(base["block_N2"], 128)
        for width in (0, 32, 65, True, "128"):
            with self.subTest(width=width):
                with self.assertRaisesRegex(ValueError, "block_N1"):
                    self.module._normalize_moe_config(dict(base, block_N1=width), 64, 64)

    def test_each_noncontiguous_input_rejected_before_jit(self):
        names = ("x", "gate_up_weight", "down_weight", "topk_ids", "topk_scale")
        with mock.patch.object(self.module, "_make", side_effect=AssertionError("JIT")):
            for index, name in enumerate(names):
                with self.subTest(name=name):
                    values = self.inputs()
                    values[index] = values[index].transpose(-1, -2).contiguous().transpose(-1, -2)
                    self.assertFalse(values[index].is_contiguous())
                    with self.assertRaisesRegex(ValueError, name + " must be contiguous"):
                        self.module.fuse_moe_tilelang(*values)


# ── mixed_gemm ───────────────────────────────────────────────────────────────

def _silu(v):
    return v * torch.sigmoid(v)


def _gelu(v):
    return 0.5 * v * (1.0 + torch.tanh(
        0.7978845608028654 * (v + 0.044715 * v ** 3)))


_ACT_FNS = {"identity": lambda v: v, "silu": _silu, "gelu": _gelu}


def _gelu_erf(v):
    """Exact erf GELU used by fuse_moe expert_activation='gelu'."""
    return 0.5 * v * (1.0 + torch.erf(v * math.sqrt(0.5)))


def _expert_act(v, expert_activation):
    if expert_activation == "gelu":
        return _gelu_erf(v)
    return _silu(v)


def test_mixed_gemm():
    print("─" * 80)
    print("mixed_gemm (Tile-lang)")
    for (M, N, K) in [(16, 128, 64), (128, 256, 128), (513, 256, 192),
                      (1024, 512, 256)]:
        for res_backend in ("fp8", "int8"):
            for x_dtype in (torch.float32, torch.bfloat16):
                for act in ("identity", "silu", "gelu"):
                    x = torch.randn(M, K, device="cuda", dtype=x_dtype) * 0.5
                    w = torch.randn(N, K, device="cuda") * 0.05
                    w_high, w_low, w_scale = split_mixed_precision_weight(
                        w, backend=res_backend)
                    bias = (torch.randn(N, device="cuda") * 0.1
                            if act != "identity" else None)
                    y = tilelang_ops.mixed_gemm(
                        x, w_high, w_low, w_scale, bias=bias, activation=act)
                    # 参考：用拆分后的权重重建（与 kernel 数值语义一致）
                    w_rec = w_high.float()
                    if w_low.dtype == torch.int8:
                        w_rec = w_rec + (w_low.float()
                                         * w_scale.unsqueeze(1)) / 256.0
                    else:
                        w_rec = w_rec + w_low.float() / 256.0
                    ref = x.float() @ w_rec.t()
                    if bias is not None:
                        ref = ref + bias
                    ref = _ACT_FNS[act](ref)
                    tag = (f"mg {M}x{N}x{K} {res_backend} "
                           f"{str(x_dtype).split('.')[-1]} {act}")
                    if not check(tag, y, ref, atol=3e-2, rtol=3e-2):
                        return
                    # out_dtype=bf16
                    y16 = tilelang_ops.mixed_gemm(
                        x, w_high, w_low, w_scale, bias=bias, activation=act,
                        out_dtype=torch.bfloat16)
                    check(tag + " bf16out", y16, ref, atol=3e-2, rtol=3e-2)


# ── fuse_moe ─────────────────────────────────────────────────────────────────

def _moe_ref(x, w1, w2, topk_ids, topk_scale,
             expert_activation="silu", numerical_mode="fused", expert_type="glu"):
    """CPU FP32 per-expert reference; no TF32 matmul, exact erf GELU."""
    S, K = topk_ids.shape
    E = w1.shape[0]
    I = w2.shape[2]
    if expert_type not in ("glu", "mlp") or w1.shape[1] != I * (2 if expert_type == "glu" else 1):
        raise ValueError("expert_type projection shapes")
    dtype = x.dtype
    xf = x.detach().cpu().float()
    w1f = w1.detach().cpu().float()
    w2f = w2.detach().cpu().float()
    ids = topk_ids.detach().cpu()
    scale = topk_scale.detach().cpu().float()
    y = torch.zeros(S, x.shape[1], dtype=torch.float32)

    def rounded(value):
        return value.to(dtype).float()

    for e in range(E):
        rows, slots = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        gu = xf[rows] @ w1f[e].t()
        if numerical_mode == "staged":
            gu = rounded(gu)
        gate = gu.chunk(2, -1)[0] if expert_type == "glu" else gu
        act = _expert_act(gate, expert_activation)
        if numerical_mode == "staged" and expert_type == "glu":
            act = rounded(act)
        hidden = rounded(act * gu.chunk(2, -1)[1] if expert_type == "glu" else act)
        down = hidden @ w2f[e].t()
        if numerical_mode == "staged" or expert_type == "mlp":
            down = rounded(down)
        weighted = down * scale[rows, slots, None]
        if numerical_mode == "staged":
            weighted = rounded(weighted)
        y.index_add_(0, rows, weighted)
    return y.to(dtype).float()


def _run_moe(S, H, I, E, K, dtype, kind, name,
             expert_activation="silu", numerical_mode="fused", expert_type="glu"):
    x = (torch.randn(S, H, device="cuda") * 0.3).to(dtype)
    w1 = (torch.randn(E, (2 if expert_type == "glu" else 1) * I, H, device="cuda") * 0.05).to(dtype)
    w2 = (torch.randn(E, H, I, device="cuda") * 0.05).to(dtype)
    if kind == "uniform":
        topk_ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32)
    elif kind == "skewed":       # 90% 落到 expert 0
        ids = torch.where(torch.rand(S, K, device="cuda") < 0.9,
                          torch.zeros(S, K, device="cuda", dtype=torch.long),
                          torch.randint(0, E, (S, K), device="cuda"))
        topk_ids = ids.to(torch.int32)
    elif kind == "single":       # 全部单专家
        topk_ids = torch.zeros(S, K, device="cuda", dtype=torch.int32)
    elif kind in ("masked", "all_masked"):
        topk_ids = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32)
        topk_ids[::3] = -1
        topk_ids[:, -1] = -1
        if kind == "all_masked":
            topk_ids.fill_(-1)
    elif kind == "duplicate":     # 同一 token 重复路由到同一专家
        e = torch.randint(0, E, (S, 1), device="cuda")
        topk_ids = e.expand(S, K).contiguous().to(torch.int32)
    else:
        raise ValueError(kind)
    topk_scale = torch.rand(S, K, device="cuda")
    if kind in ("masked", "all_masked"):
        topk_scale.masked_fill_(topk_ids < 0, float("nan"))

    y = tilelang_ops.fuse_moe(x, w1, w2, topk_ids, topk_scale,
                              expert_activation=expert_activation,
                              numerical_mode=numerical_mode, expert_type=expert_type)
    ref = _moe_ref(x, w1, w2, topk_ids, topk_scale,
                   expert_activation, numerical_mode, expert_type)
    ok = check(name, y, ref, atol=3e-2, rtol=3e-2)
    if not torch.isfinite(y).all():
        print(f"[FAIL] {name}: nonfinite output")
        _results.append(False)
        ok = False
    if kind == "all_masked":
        zero = bool(torch.count_nonzero(y) == 0)
        _results.append(zero)
        print(f"[{'PASS' if zero else 'FAIL'}] {name}: all masked output is zero")
        ok = ok and zero

    # 与 CUDA 版交叉对拍（可用时）
    try:
        y_cuda = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale, backend="cuda",
                              expert_activation=expert_activation,
                              numerical_mode=numerical_mode, expert_type=expert_type)
        check(name + " vs CUDA", y, y_cuda, atol=3e-2, rtol=3e-2)
    except Exception as e:
        if ops.is_available("fuse_moe"):
            # An explicitly loaded/available backend failing execution is a
            # regression, not missing optional coverage.
            raise
        print(f"       (CUDA 版不可用，跳过交叉对拍: {type(e).__name__})")
    return ok


def test_fuse_moe():
    print("─" * 80)
    print("fuse_moe (Tile-lang)")
    _run_moe(128, 128, 64, 4, 2, torch.bfloat16, "uniform", "moe 128,128,64,4,2 bf16")
    _run_moe(512, 256, 128, 8, 2, torch.bfloat16, "uniform", "moe 512,256,128,8,2 bf16")
    _run_moe(300, 256, 128, 8, 2, torch.float16, "uniform", "moe 300,256,128,8,2 fp16")
    _run_moe(128, 128, 64, 64, 2, torch.bfloat16, "uniform", "moe 128,128,64,64,2 多专家")
    _run_moe(64, 128, 64, 1, 1, torch.bfloat16, "single", "moe 64,128,64,1,1 单专家")
    _run_moe(256, 128, 64, 4, 3, torch.bfloat16, "skewed", "moe 256,128,64,4,3 倾斜路由")
    _run_moe(256, 128, 64, 4, 2, torch.bfloat16, "duplicate", "moe 256,128,64,4,2 重复路由")
    _run_moe(1024, 512, 256, 8, 2, torch.bfloat16, "uniform", "moe 1024,512,256,8,2 bf16")

    # GEGLU / staged 四组合（bf16 + fp16），含 masked NaN scales。
    # masked 路由必须连 scale 一起跳过；全 masked 输入必须输出全零。
    print("─" * 80)
    print("fuse_moe GEGLU / staged (Tile-lang)")
    for dtype in (torch.bfloat16, torch.float16):
        for expert_activation in ("silu", "gelu"):
            for numerical_mode in ("fused", "staged"):
                _run_moe(129, 128, 128, 8, 4, dtype, "uniform",
                         f"moe 129,128,128,8,4 {str(dtype).split('.')[-1]} "
                         f"{expert_activation}/{numerical_mode}",
                         expert_activation, numerical_mode)
                _run_moe(256, 128, 64, 4, 2, dtype, "skewed",
                         f"moe skew 256,128,64,4,2 {str(dtype).split('.')[-1]} "
                         f"{expert_activation}/{numerical_mode}",
                         expert_activation, numerical_mode)
                for kind in ("masked", "all_masked"):
                    _run_moe(7, 64, 64, 2, 2, dtype, kind,
                             f"moe {kind} {dtype} {expert_activation}/{numerical_mode}",
                             expert_activation, numerical_mode)

    # Precision witnesses must detect each staged materialization boundary;
    # random allclose alone can hide omitted casts or atomics to a local buffer.
    from test_fuse_moe import numerical_semantics_fixture
    for dtype in (torch.float16, torch.bfloat16):
        inputs = numerical_semantics_fixture(dtype, "staged")
        x, w1, w2, ids, scale, expected, _ = inputs
        output = tilelang_ops.fuse_moe(x.cuda(), w1.cuda(), w2.cuda(), ids.cuda(), scale.cuda(),
                                      expert_activation="gelu", numerical_mode="staged")
        exact = torch.equal(output.float().cpu(), expected)
        _results.append(exact)
        print(f"[{'PASS' if exact else 'FAIL'}] staged scalar materialization witness {dtype}")


def test_mlp():
    for dtype in (torch.float16, torch.bfloat16):
        for activation in ("silu", "gelu"):
            for numerical in ("fused", "staged"):
                for s, h, i, e, k, routing in ((7,64,64,1,1,"single"), (129,128,192,8,4,"masked"),
                                              (65,192,128,8,4,"duplicate"), (3,64,64,512,128,"uniform"),
                                              (7,64,64,2,2,"all_masked")):
                    _run_moe(s,h,i,e,k,dtype,routing,f"mlp {dtype}/{activation}/{numerical}/{routing}",
                             activation,numerical,"mlp")
                from test_fuse_moe import mlp_numerical_semantics_fixture
                x,w1,w2,ids,scale,expected,alternatives = mlp_numerical_semantics_fixture(dtype,numerical,activation)
                output = tilelang_ops.fuse_moe(x.cuda(),w1.cuda(),w2.cuda(),ids.cuda(),scale.cuda(),
                    expert_activation=activation,numerical_mode=numerical,expert_type="mlp").float().cpu()
                exact = torch.equal(output, expected)
                _results.append(exact)
                print(f"[{'PASS' if exact else 'FAIL'}] MLP scalar {dtype}/{activation}/{numerical}: "
                      f"max_abs={(output - expected).abs().max().item():.6g}")
                for name, wrong in alternatives.items():
                    discriminated = not torch.equal(output, wrong)
                    _results.append(discriminated)
                    print(f"[{'PASS' if discriminated else 'FAIL'}] MLP witness {name} "
                          f"{dtype}/{activation}/{numerical}")


# ── swiglu ───────────────────────────────────────────────────────────────────

def test_swiglu():
    print("─" * 80)
    print("swiglu (Tile-lang)")
    for (M, N, K) in [(128, 128, 256), (37, 64, 128), (1024, 256, 512)]:
        for dtype in (torch.bfloat16, torch.float16):
            x = (torch.randn(M, K, device="cuda") * 0.5).to(dtype)
            w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(dtype)
            y = tilelang_ops.swiglu(x, w)
            g = x.float() @ w[:N].float().t()
            u = x.float() @ w[N:].float().t()
            ref = _silu(g) * u
            check(f"swiglu {M}x{N}x{K} {str(dtype).split('.')[-1]}", y, ref,
                  atol=3e-2, rtol=3e-2)


# ── fa 冒烟 ──────────────────────────────────────────────────────────────────

def test_fa_smoke():
    print("─" * 80)
    print("fa (Tile-lang) 冒烟 — 完整对拍见 benchmark/fa_tilelang.py --test")
    for (B, H, Hk, Sq, Sk, d) in [(2, 16, 16, 256, 256, 128),
                                  (2, 16, 4, 256, 256, 64),
                                  (1, 1, 1, 128, 8192, 128)]:
        for dtype in (torch.float16, torch.bfloat16):
            q = (torch.randn(B, H, Sq, d, device="cuda") * 0.5).to(dtype)
            k = (torch.randn(B, Hk, Sk, d, device="cuda") * 0.5).to(dtype)
            v = (torch.randn(B, Hk, Sk, d, device="cuda") * 0.5).to(dtype)
            mask = (torch.randn(B, 1, Sq, Sk, device="cuda") * 0.3).to(dtype)
            mask[..., :, 0] = 0
            out = tilelang_ops.mha_fwd_with_mask(q, k, v, mask)
            kk = k.repeat_interleave(H // Hk, dim=1)
            vv = v.repeat_interleave(H // Hk, dim=1)
            s = (q.float() / d ** 0.5) @ kk.float().transpose(-2, -1) + mask.float()
            p = torch.softmax(s, dim=-1)
            ref = (p.to(vv.dtype) @ vv).float()
            check(f"fa {B},{H},{Hk},{Sq},{Sk},{d} {str(dtype).split('.')[-1]}",
                  out, ref)


# ── backend 选路 ─────────────────────────────────────────────────────────────

def test_backend_dispatch():
    print("─" * 80)
    print("backend 选路")
    x = torch.randn(64, 128, device="cuda") * 0.5
    w = torch.randn(128, 128, device="cuda") * 0.05
    w_high, w_low, w_scale = split_mixed_precision_weight(w, backend="int8")
    y1 = ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu",
                        backend="tilelang")
    y2 = tilelang_ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu")
    check("ops.mixed_gemm(backend='tilelang') ≡ tilelang_ops", y1, y2)

    try:
        y3 = ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu",
                            backend="cuda")
        check("cuda vs tilelang (mixed_gemm silu)", y3, y1, atol=5e-2, rtol=5e-2)
    except Exception as e:
        print(f"       (CUDA 版不可用，跳过: {type(e).__name__})")

    # CUSTOM_OPS_BACKEND 环境变量
    os.environ["CUSTOM_OPS_BACKEND"] = "tilelang"
    try:
        y4 = ops.mixed_gemm(x, w_high, w_low, w_scale, activation="silu")
        check("env CUSTOM_OPS_BACKEND=tilelang", y4, y2)
    finally:
        del os.environ["CUSTOM_OPS_BACKEND"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-only", action="store_true")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only host contract tests; no GPU or TileLang compilation")
    parser.add_argument("--ops-library", help="Explicit precompiled CUDA fuse_moe library for cross checks")
    parser.add_argument("--ops", nargs="+", choices=("mixed_gemm", "fuse_moe", "swiglu", "fa", "dispatch"),
                        default=("mixed_gemm", "fuse_moe", "swiglu", "fa", "dispatch"))
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FuseMoETileLangContracts)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        sys.exit(1)
    if args.cpu_only:
        print("Host contracts passed; GPU kernels SKIPPED, not validated.")
        sys.exit(0)
    if args.ops_library:
        ops.load_precompiled("fuse_moe", args.ops_library)
    if not torch.cuda.is_available():
        print("[SKIP] CUDA 不可用，Tile-lang kernel 无法运行。")
        sys.exit(0)
    if not tilelang_ops.is_available():
        print("[SKIP] tile-lang 不可用（未安装或其 CUDA 不可用），按现有结构跳过。")
        sys.exit(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"cap: {torch.cuda.get_device_capability(0)}")
    if args.mlp_only:
        test_mlp()
        sys.exit(0 if all(_results) else 1)
    if "mixed_gemm" in args.ops:
        test_mixed_gemm()
    if "fuse_moe" in args.ops:
        test_fuse_moe()
        test_mlp()
    if "swiglu" in args.ops:
        test_swiglu()
    if "fa" in args.ops:
        test_fa_smoke()
    if "dispatch" in args.ops:
        test_backend_dispatch()
    print("=" * 80)
    print(f"{'ALL TESTS PASSED' if all(_results) else 'SOME TESTS FAILED'} "
          f"({sum(_results)}/{len(_results)})")
    print("=" * 80)
    sys.exit(0 if all(_results) else 1)
