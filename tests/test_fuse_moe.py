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
    部分专家空闲、topk=1、重复 expert（同一 token 的多个 topk 命中同一 expert）
退出码：0 全部通过；1 存在 FAIL。
"""
import argparse
import os
import sys

# 自举：把包父目录加入 sys.path（远程验证免环境配置）
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402

from custom_ops import ops  # noqa: E402  (import 时 JIT 编译)

torch.manual_seed(0)
DEV = "cuda"

RESULTS = {"PASS": 0, "FAIL": 0}
FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    RESULTS[tag] += 1
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"[{tag}] {name} {detail}")


def ref_fuse_moe(x, gate_up_weight, down_weight, topk_ids, topk_scale):
    """向量化 fp32 参考：y[s] = Σ_j scale[s,j] * (Down_ej @ silu(GateUp_ej @ x[s]))"""
    S, H = x.shape
    E, N2, _ = gate_up_weight.shape
    I = down_weight.shape[2]
    K = topk_ids.shape[1]
    T = S * K

    xf = x.float()
    e = topk_ids.reshape(-1)                                   # (T,)
    xg = xf[:, None, :].expand(S, K, H).reshape(T, H)          # (T, H)
    w1e = gate_up_weight.float()[e]                            # (T, 2I, H)
    gu = torch.bmm(xg.unsqueeze(1), w1e.transpose(1, 2)).squeeze(1)   # (T, 2I)
    g, u = gu[:, :I], gu[:, I:]
    h = torch.nn.functional.silu(g) * u                        # (T, I)
    w2e = down_weight.float()[e]                               # (T, H, I)
    y = torch.bmm(h.unsqueeze(1), w2e.transpose(1, 2)).squeeze(1)     # (T, H)
    y = y.view(S, K, H)
    out = (y * topk_scale.unsqueeze(-1)).sum(dim=1)            # (S, H)
    return out


def run_case(name, S, H, I, E, K, dtype, mode="uniform"):
    dev = torch.device(DEV)
    x = (torch.randn(S, H, device=dev, dtype=torch.float32) * 0.5).to(dtype)
    w1 = (torch.randn(E, 2 * I, H, device=dev, dtype=torch.float32) * 0.05).to(dtype)
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
    else:
        raise ValueError(mode)

    topk_scale = torch.rand(S, K, device=dev, dtype=torch.float32)
    if mode == "duplicate":
        topk_scale = torch.full((S, K), 1.0 / K, device=dev)

    out = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)
    ref = ref_fuse_moe(x, w1, w2, topk_ids, topk_scale)

    ok_shape = out.shape == x.shape and out.dtype == x.dtype
    diff = (out.float() - ref).abs()
    denom = ref.abs().clamp_min(1e-3)
    rel = (diff / denom).max().item()
    # bf16/fp16 输出量化 + kernel 内 bf16 中间精度：宽松阈值
    tol = 0.03 if dtype == torch.bfloat16 else 0.02
    ok = ok_shape and torch.allclose(out.float(), ref, atol=0.05, rtol=tol)
    check(f"{name} [{dtype} S={S} H={H} I={I} E={E} K={K} {mode}]",
          ok, f"shape_ok={ok_shape} max_abs={diff.max().item():.4g} max_rel={rel:.4g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    assert ops.is_available(), "extension load failed"

    dtypes = [torch.bfloat16, torch.float16]
    if torch.cuda.get_device_capability(0)[0] < 8:
        print("SM80+ required for fuse_moe; skipping")
        sys.exit(0)

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

    # ── 边界：全部 expert 空闲不可能（T>0），但空 token 集合由 op 层拒绝 ────
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
        sys.exit(1)


if __name__ == "__main__":
    main()
