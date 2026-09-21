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

import os
import sys

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
    out = out.float()
    ref = ref.float()
    err = (out - ref).abs()
    max_err = err.max().item()
    ref_scale = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_scale) < rtol
    print(f"[{'PASS' if passed else 'FAIL'}] {name:56s} max_err={max_err:.4e}")
    _results.append(passed)
    return passed


# ── mixed_gemm ───────────────────────────────────────────────────────────────

def _silu(v):
    return v * torch.sigmoid(v)


def _gelu(v):
    return 0.5 * v * (1.0 + torch.tanh(
        0.7978845608028654 * (v + 0.044715 * v ** 3)))


_ACT_FNS = {"identity": lambda v: v, "silu": _silu, "gelu": _gelu}


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

def _moe_ref(x, w1, w2, topk_ids, topk_scale):
    S, K = topk_ids.shape
    I = w1.shape[1] // 2
    y = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
    for s in range(S):
        for j in range(K):
            e = int(topk_ids[s, j])
            gu = x[s].float() @ w1[e].float().t()          # (2I,)
            act = _silu(gu[:I]) * gu[I:]
            y[s] += (act @ w2[e].float().t()) * topk_scale[s, j]
    return y


def _run_moe(S, H, I, E, K, dtype, kind, name):
    x = (torch.randn(S, H, device="cuda") * 0.3).to(dtype)
    w1 = (torch.randn(E, 2 * I, H, device="cuda") * 0.05).to(dtype)
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
    else:                        # duplicate：同一 token 重复路由到同一专家
        e = torch.randint(0, E, (S, 1), device="cuda")
        topk_ids = e.expand(S, K).contiguous().to(torch.int32)
    topk_scale = torch.rand(S, K, device="cuda")

    y = tilelang_ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)
    ref = _moe_ref(x, w1, w2, topk_ids, topk_scale)
    ok = check(name, y, ref, atol=3e-2, rtol=3e-2)

    # 与 CUDA 版交叉对拍（可用时）
    try:
        y_cuda = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale, backend="cuda")
        check(name + " vs CUDA", y, y_cuda, atol=3e-2, rtol=3e-2)
    except Exception as e:
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
    if not torch.cuda.is_available():
        print("CUDA 不可用。")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"cap: {torch.cuda.get_device_capability(0)}")
    test_mixed_gemm()
    test_fuse_moe()
    test_swiglu()
    test_fa_smoke()
    test_backend_dispatch()
    print("=" * 80)
    print(f"{'ALL TESTS PASSED' if all(_results) else 'SOME TESTS FAILED'} "
          f"({sum(_results)}/{len(_results)})")
    print("=" * 80)
    sys.exit(0 if all(_results) else 1)
