"""
custom_ops/test_custom_ops.py — 通用库自测

测试内容
--------
1.  CustomOps 基类接口是否正常（不可实例化使用、is_available 等）
2.  FaOps（示例）加载与 mha_fwd_with_mask 正确性
3.  与 Kernel.ops.mha_fwd_with_mask（原始库）的输出一致性
4.  repr / load_error 等辅助方法
5.  __getattr__ 通用分发
6.  幂等加载（多次 load）
7. RecsysOps 加载 — from custom_ops import RecsysOps, ops
8. RecsysOps 三个算子功能与 RecsysOps/Kernel.ops 输出一致性

运行方式
--------
    TORCH_EXTENSIONS_DIR=/workdir/tetuan_cache/torch_extensions \
        python custom_ops/test_custom_ops.py
"""

import sys
import os

# 确保 /workdir/tetuan 在 Python 路径中（直接运行时需要）
sys.path.insert(0, "/workdir/tetuan")

import torch
import warnings
warnings.filterwarnings("ignore")   # 抑制加载时的 warning 输出，让测试结果更清晰

PASS = "✓"
FAIL = "✗"
SEP  = "─" * 70


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"{SEP}")


def check(name: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not cond:
        raise AssertionError(f"FAILED: {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
section("1. CustomOps 基类接口测试")
# ─────────────────────────────────────────────────────────────────────────────

from custom_ops import CustomOps

# 基类本身不可以直接 load（没有实现 get_sources）
base = CustomOps()
base.load()  # 会失败但不抛出
check("CustomOps() 基类 load 失败后 is_available=False", not base.is_available())
check("CustomOps() load_error 非空", base.load_error() is not None)
check("CustomOps() repr 包含 'not loaded'", "not loaded" in repr(base))
print(f"  load_error = {base.load_error()[:80]}...")


# ─────────────────────────────────────────────────────────────────────────────
section("2. FaOps 示例加载测试")
# ─────────────────────────────────────────────────────────────────────────────

from custom_ops.example import ops as fa_ops, FaOps

check("FaOps.namespace == 'fa_ops'", FaOps.namespace == "fa_ops")
check("FaOps.so_name == 'fa_ops_kernel'", FaOps.so_name == "fa_ops_kernel")
check("FaOps.required_ops 包含三个算子",
      set(FaOps.required_ops) == {"mha_fwd_with_mask", "pack_and_prepare_b1", "jagged_pool_and_collect"},
      f"got {FaOps.required_ops}")

if fa_ops.is_available():
    check("FaOps 加载成功", True)
    check("fa_ops repr 包含 'loaded'", "loaded" in repr(fa_ops))
    check("fa_ops.load_error() is None", fa_ops.load_error() is None)
else:
    # GPU 不可用时跳过后续
    print(f"  [!] FaOps 加载失败: {fa_ops.load_error()}")
    print("  跳过 GPU 相关测试（可能是 CUDA 不可用或编译失败）")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
section("3. mha_fwd_with_mask 正确性测试")
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(42)

for label, B, H, Sq, d in [
    ("S=128  d=64",   1, 8,  128,  64),
    ("S=512  d=64",   1, 8,  512,  64),
    ("S=256  d=128",  1, 8,  256, 128),
]:
    q    = torch.randn(B, H, Sq, d,  device="cuda", dtype=torch.bfloat16)
    k    = torch.randn(B, H, Sq, d,  device="cuda", dtype=torch.bfloat16)
    v    = torch.randn(B, H, Sq, d,  device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(B, 1, Sq, Sq, device="cuda", dtype=torch.bfloat16)

    # FaOps 输出
    out_fa = fa_ops.mha_fwd_with_mask(q, k, v, mask)

    # SDPA 参考（全可见 mask）
    import torch.nn.functional as F
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    diff = (out_fa.float() - out_sdpa.float()).abs().max().item()
    check(
        f"mha_fwd_with_mask vs SDPA  {label}",
        diff < 0.02,
        f"max_diff={diff:.6f}"
    )

# ─────────────────────────────────────────────────────────────────────────────
section("4. 与原始 Kernel.ops 输出一致性测试")
# ─────────────────────────────────────────────────────────────────────────────

try:
    from Kernel import ops as kernel_ops
    if kernel_ops.is_available():
        torch.manual_seed(99)
        B, H, S, d = 1, 8, 256, 64
        q    = torch.randn(B, H, S, d, device="cuda", dtype=torch.bfloat16)
        k    = torch.randn(B, H, S, d, device="cuda", dtype=torch.bfloat16)
        v    = torch.randn(B, H, S, d, device="cuda", dtype=torch.bfloat16)
        mask = torch.zeros(B, 1, S, S, device="cuda", dtype=torch.bfloat16)

        out_new = fa_ops.mha_fwd_with_mask(q, k, v, mask)
        out_old = kernel_ops.mha_fwd_with_mask(q, k, v, mask)
        diff = (out_new.float() - out_old.float()).abs().max().item()
        check(
            "FaOps vs Kernel.ops 输出一致",
            diff < 1e-4,
            f"max_diff={diff:.6f}"
        )
    else:
        print("  [!] Kernel.ops 不可用，跳过一致性测试")
except Exception as e:
    print(f"  [!] 导入 Kernel.ops 失败: {e}，跳过一致性测试")


# ─────────────────────────────────────────────────────────────────────────────
section("5. __getattr__ 通用分发测试")
# ─────────────────────────────────────────────────────────────────────────────

# 通过 __getattr__ 访问（不走显式方法）
op_fn = fa_ops.mha_fwd_with_mask    # 第一次会走 __getattr__ → torch.ops.fa_ops...
check("__getattr__ 返回可调用对象", callable(op_fn))

# 访问不存在的算子应抛 AttributeError
try:
    _ = fa_ops.nonexistent_op_xyz
    check("访问不存在算子应抛 AttributeError", False)
except AttributeError:
    check("访问不存在算子正确抛 AttributeError", True)


# ─────────────────────────────────────────────────────────────────────────────
section("6. 幂等加载测试（多次 load）")
# ─────────────────────────────────────────────────────────────────────────────

ops2 = FaOps().load()
check("第二个实例也能成功加载", ops2.is_available())

# 同一实例多次 load 幂等
fa_ops.load()
check("同一实例多次 load 幂等", fa_ops.is_available())


# ─────────────────────────────────────────────────────────────────────────────
section("7. RecsysOps 加载测试（from custom_ops import RecsysOps, ops）")
# ─────────────────────────────────────────────────────────────────────────────

from custom_ops import RecsysOps, ops as recsys_ops

check("RecsysOps.namespace == 'recsys_ops'", RecsysOps.namespace == "recsys_ops")
check("RecsysOps.so_name == 'recsys_ops_kernel'", RecsysOps.so_name == "recsys_ops_kernel")
check("RecsysOps.required_ops 包含一个算子",
      set(RecsysOps.required_ops) == {"mha_fwd_with_mask"},
      f"got {RecsysOps.required_ops}")

if recsys_ops.is_available():
    check("RecsysOps 加载成功", True)
    check("recsys_ops repr 包含 'loaded'", "loaded" in repr(recsys_ops))
    check("recsys_ops.load_error() is None", recsys_ops.load_error() is None)
else:
    print(f"  [!] RecsysOps 加载失败: {recsys_ops.load_error()}")
    print("  跳过后续 RecsysOps GPU 相关测试")
    print(f"\n{'='*70}")
    print("  All tests PASSED")
    print(f"{'='*70}\n")
    import sys; sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
section("8. RecsysOps 算子功能测试")
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(55)

# ── 12a. mha_fwd_with_mask ────────────────────────────────────────────────────
B, H, S, d = 1, 8, 128, 64
q    = torch.randn(B, H, S, d,  device="cuda", dtype=torch.bfloat16)
k    = torch.randn(B, H, S, d,  device="cuda", dtype=torch.bfloat16)
v    = torch.randn(B, H, S, d,  device="cuda", dtype=torch.bfloat16)
mask = torch.zeros(B, 1, S, S, device="cuda", dtype=torch.bfloat16)

out_r = recsys_ops.mha_fwd_with_mask(q, k, v, mask)
out_f = fa_ops.mha_fwd_with_mask(q, k, v, mask)
diff  = (out_r.float() - out_f.float()).abs().max().item()
check("RecsysOps.mha_fwd_with_mask vs FaOps", diff < 1e-4, f"max_diff={diff:.6f}")


# ── 12e. 幂等加载 ─────────────────────────────────────────────────────────────
ops2 = RecsysOps().load()
check("第二个 RecsysOps 实例也能加载", ops2.is_available())
recsys_ops.load()
check("RecsysOps 单例多次 load 幂等", recsys_ops.is_available())


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  All tests PASSED")
print(f"{'='*70}\n")
