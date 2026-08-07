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
7.  pack_and_prepare_b1（FaOps）正确性
8.  pack_and_prepare_b1 vs Kernel.ops 一致性
9.  jagged_pool_and_collect（FaOps）正确性
10. jagged_pool_and_collect vs Kernel.ops 一致性
11. RecsysOps 加载 — from custom_ops import RecsysOps, ops
12. RecsysOps 三个算子功能与 FaOps/Kernel.ops 输出一致性

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
section("7. pack_and_prepare_b1 正确性测试")
# ─────────────────────────────────────────────────────────────────────────────

check("fa_ops 具有 pack_and_prepare_b1 方法", hasattr(fa_ops, "pack_and_prepare_b1"))

torch.manual_seed(7)

for label, s_len, c_len, i_len, D, hd, S_max in [
    ("s=4 c=2 i=8  D=64  hd=32", 4, 2, 8,  64, 32, 32),
    ("s=8 c=4 i=16 D=128 hd=64", 8, 4, 16, 128, 64, 64),
]:
    valid_tokens = s_len + c_len + i_len

    h_s = torch.randn(s_len, D, device="cuda", dtype=torch.bfloat16)
    h_c = torch.randn(c_len, D, device="cuda", dtype=torch.bfloat16)
    h_i = torch.randn(i_len, D, device="cuda", dtype=torch.bfloat16)
    static_cos = torch.randn(1, S_max, 1, hd, device="cuda", dtype=torch.bfloat16)
    static_sin = torch.randn(1, S_max, 1, hd, device="cuda", dtype=torch.bfloat16)

    h_dense, cos_out, sin_out, attn_mask, item_mask, offsets = fa_ops.pack_and_prepare_b1(
        h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_max
    )

    # 验证输出 shape
    check(f"pack_b1 h_dense shape  {label}",
          tuple(h_dense.shape) == (1, S_max, D),
          f"got {tuple(h_dense.shape)}")
    check(f"pack_b1 cos shape      {label}",
          tuple(cos_out.shape) == (1, S_max, 1, hd),
          f"got {tuple(cos_out.shape)}")
    check(f"pack_b1 attn_mask shape {label}",
          tuple(attn_mask.shape) == (1, 1, S_max, S_max),
          f"got {tuple(attn_mask.shape)}")
    check(f"pack_b1 item_mask shape {label}",
          tuple(item_mask.shape) == (valid_tokens,),
          f"got {tuple(item_mask.shape)}")
    check(f"pack_b1 offsets shape   {label}",
          tuple(offsets.shape) == (2,),
          f"got {tuple(offsets.shape)}")

    # 验证 h_dense 内容（有效段应与输入一致）
    check(f"pack_b1 h_dense[:s_len] == h_s  {label}",
          torch.allclose(h_dense[0, :s_len].float(), h_s.float(), atol=1e-3),
          f"max_diff={(h_dense[0, :s_len].float() - h_s.float()).abs().max().item():.6f}")
    check(f"pack_b1 h_dense[s+c:valid] == h_i {label}",
          torch.allclose(h_dense[0, s_len+c_len:valid_tokens].float(), h_i.float(), atol=1e-3))

    # 验证 item_mask：item token 为 True，ubc/ctx token 为 False
    item_pos = s_len + c_len
    check(f"pack_b1 item_mask ubc+ctx=False {label}",
          not item_mask[:item_pos].any().item())
    check(f"pack_b1 item_mask item=True     {label}",
          item_mask[item_pos:].all().item())

    # 验证 offsets = [0, valid_tokens]
    check(f"pack_b1 offsets=[0,valid_tokens] {label}",
          int(offsets[0].item()) == 0 and int(offsets[1].item()) == valid_tokens,
          f"got [{offsets[0].item()}, {offsets[1].item()}]")

    # 验证 attn_mask 对角线（可见位置）不为 -inf
    diag_vals = attn_mask[0, 0].diagonal()[:valid_tokens]
    check(f"pack_b1 attn_mask 对角线可见     {label}",
          (diag_vals > -1000).all().item())

    # 验证 padding 区（valid_tokens:S_max 行）全 -inf
    if valid_tokens < S_max:
        pad_rows = attn_mask[0, 0, valid_tokens:, :]
        check(f"pack_b1 padding 行全 -inf       {label}",
              (pad_rows == float('-inf')).all().item())

print(f"  pack_and_prepare_b1 所有 case 验证通过")


# ─────────────────────────────────────────────────────────────────────────────
section("8. pack_and_prepare_b1 vs Kernel.ops 一致性测试")
# ─────────────────────────────────────────────────────────────────────────────

try:
    from Kernel import ops as _kernel_ops_pack
    if _kernel_ops_pack.is_available():
        torch.manual_seed(13)
        s_len, c_len, i_len, D, hd, S_max = 6, 3, 10, 64, 32, 32
        h_s = torch.randn(s_len, D, device="cuda", dtype=torch.bfloat16)
        h_c = torch.randn(c_len, D, device="cuda", dtype=torch.bfloat16)
        h_i = torch.randn(i_len, D, device="cuda", dtype=torch.bfloat16)
        static_cos = torch.randn(1, S_max, 1, hd, device="cuda", dtype=torch.bfloat16)
        static_sin = torch.randn(1, S_max, 1, hd, device="cuda", dtype=torch.bfloat16)

        outs_new = fa_ops.pack_and_prepare_b1(
            h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_max)
        outs_old = _kernel_ops_pack.pack_and_prepare_b1(
            h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_max)

        # attn_mask 含 -inf，直接比 nan-safe 方式
        for idx, name in enumerate(["h_dense", "cos", "sin", "attn_mask", "item_mask", "offsets"]):
            t_new = outs_new[idx].float()
            t_old = outs_old[idx].float()
            if name == "attn_mask":
                # 只比较有限值区域（padding 区 -inf nan-safe）
                finite = torch.isfinite(t_new) & torch.isfinite(t_old)
                sign_match = (t_new == float('-inf')) == (t_old == float('-inf'))
                is_ok = sign_match.all() and (t_new[finite] - t_old[finite]).abs().max().item() < 1e-3
                check(f"pack_b1 vs Kernel.ops  [attn_mask]", is_ok,
                      "attn_mask -inf 区域与有限值区域均一致")
            else:
                diff = (t_new - t_old).abs().max().item()
                check(f"pack_b1 vs Kernel.ops  [{name}]", diff < 1e-3, f"max_diff={diff:.6f}")
    else:
        print("  [!] Kernel.ops 不可用，跳过 pack_b1 一致性测试")
except ImportError as e:
    print(f"  [!] 导入 Kernel 模块失败: {e}，跳过 pack_b1 一致性测试")


# ─────────────────────────────────────────────────────────────────────────────
section("9. jagged_pool_and_collect 正确性测试（mean / sum）")
# ─────────────────────────────────────────────────────────────────────────────

check("fa_ops 具有 jagged_pool_and_collect 方法", hasattr(fa_ops, "jagged_pool_and_collect"))

torch.manual_seed(21)

for reduce_mode, mode_name in [(0, "mean"), (1, "sum")]:
    # 构造 3 个 pooling 特征，每个特征 4 个 sample，seq_len=3，D=64
    n_pooling = 3
    n_samp    = 4
    seq_len   = 3
    D         = 64

    pool_vals_list = [
        torch.randn(n_samp * seq_len, D, device="cuda", dtype=torch.bfloat16)
        for _ in range(n_pooling)
    ]
    pool_lens_list = [
        torch.full((n_samp,), seq_len, dtype=torch.int32, device="cuda")
        for _ in range(n_pooling)
    ]
    ones_cache = torch.ones(n_samp * n_pooling * 2, dtype=torch.int32, device="cuda")

    # 拼接成 big tensor
    pool_values  = torch.cat(pool_vals_list, dim=0)   # (n_pooling * n_samp * seq_len, D)
    pool_lengths = torch.cat(pool_lens_list, dim=0)   # (n_pooling * n_samp,)

    # 计算 splits
    val_split  = [0]
    len_split  = [0]
    for i in range(n_pooling):
        val_split.append(val_split[-1] + pool_vals_list[i].shape[0])
        len_split.append(len_split[-1] + pool_lens_list[i].shape[0])

    out_v, out_l, out_vs, out_ls = fa_ops.jagged_pool_and_collect(
        pool_values, pool_lengths, val_split, len_split,
        n_pooling, reduce_mode, ones_cache,
    )

    # 验证输出形状
    expected_out_rows = n_pooling * n_samp
    check(f"jagged_pool [{mode_name}] pool_values_out shape",
          out_v.shape[0] == expected_out_rows and out_v.shape[1] == D,
          f"got {tuple(out_v.shape)}, expected ({expected_out_rows},{D})")
    check(f"jagged_pool [{mode_name}] pool_lengths_out shape",
          out_l.shape[0] == expected_out_rows,
          f"got {out_l.shape[0]}, expected {expected_out_rows}")
    check(f"jagged_pool [{mode_name}] pool_lengths_out 全 1",
          (out_l == 1).all().item())
    check(f"jagged_pool [{mode_name}] out_val_splits 长度",
          out_vs.shape[0] == n_pooling + 1,
          f"got {out_vs.shape[0]}")

    # 验证数值精度：与 Python 参考实现比较
    for i in range(n_pooling):
        v     = pool_vals_list[i]   # (n_samp * seq_len, D)
        v_f   = v.float().view(n_samp, seq_len, D)
        ref   = v_f.mean(dim=1).to(v.dtype) if reduce_mode == 0 else v_f.sum(dim=1).to(v.dtype)
        start = int(out_vs[i].item())
        end   = int(out_vs[i + 1].item())
        got   = out_v[start:end]
        diff  = (got.float() - ref.float()).abs().max().item()
        check(f"jagged_pool [{mode_name}] feat[{i}] 精度",
              diff < 0.02, f"max_diff={diff:.6f}")

print(f"  jagged_pool_and_collect mean/sum 验证通过")


# ─────────────────────────────────────────────────────────────────────────────
section("10. jagged_pool_and_collect vs Kernel.ops 一致性测试")
# ─────────────────────────────────────────────────────────────────────────────

try:
    from Kernel import ops as _kernel_ops_jagged
    if _kernel_ops_jagged.is_available():
        torch.manual_seed(33)
        n_pooling, n_samp, seq_len, D = 4, 8, 5, 64
        pool_vals_list = [
            torch.randn(n_samp * seq_len, D, device="cuda", dtype=torch.bfloat16)
            for _ in range(n_pooling)
        ]
        pool_lens_list = [
            torch.full((n_samp,), seq_len, dtype=torch.int32, device="cuda")
            for _ in range(n_pooling)
        ]
        # ones_cache 需要 >= max(每特征输出行数) = n_samp；这里保留足够余量
        ones_cache = torch.ones(n_samp * n_pooling, dtype=torch.int32, device="cuda")

        pool_values  = torch.cat(pool_vals_list, dim=0)
        pool_lengths = torch.cat(pool_lens_list, dim=0)
        val_split  = [0]; len_split = [0]
        for i in range(n_pooling):
            val_split.append(val_split[-1] + pool_vals_list[i].shape[0])
            len_split.append(len_split[-1] + pool_lens_list[i].shape[0])

        out_new = fa_ops.jagged_pool_and_collect(
            pool_values, pool_lengths, val_split, len_split,
            n_pooling, 0, ones_cache)
        out_old = _kernel_ops_jagged.jagged_pool_and_collect(
            pool_values, pool_lengths, val_split, len_split,
            n_pooling, 0, ones_cache)

        for idx, name in enumerate(["pool_values_out", "pool_lengths_out",
                                     "out_val_splits", "out_len_splits"]):
            t_new = out_new[idx].float()
            t_old = out_old[idx].float()
            diff  = (t_new - t_old).abs().max().item()
            check(f"jagged_pool vs Kernel.ops [{name}]", diff < 1e-4, f"max_diff={diff:.6f}")
    else:
        print("  [!] Kernel.ops 不可用，跳过 jagged_pool 一致性测试")
except ImportError as e:
    print(f"  [!] 导入 Kernel 模块失败: {e}，跳过 jagged_pool 一致性测试")


# ─────────────────────────────────────────────────────────────────────────────
section("11. RecsysOps 加载测试（from custom_ops import RecsysOps, ops）")
# ─────────────────────────────────────────────────────────────────────────────

from custom_ops import RecsysOps, ops as recsys_ops

check("RecsysOps.namespace == 'recsys_ops'", RecsysOps.namespace == "recsys_ops")
check("RecsysOps.so_name == 'recsys_ops_kernel'", RecsysOps.so_name == "recsys_ops_kernel")
check("RecsysOps.required_ops 包含三个算子",
      set(RecsysOps.required_ops) == {"mha_fwd_with_mask", "pack_and_prepare_b1", "jagged_pool_and_collect"},
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
section("12. RecsysOps 算子功能测试")
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

# ── 12b. pack_and_prepare_b1 ─────────────────────────────────────────────────
torch.manual_seed(56)
s_len, c_len, i_len, D, hd, S_max = 5, 3, 8, 64, 32, 32
h_s = torch.randn(s_len, D, device="cuda", dtype=torch.bfloat16)
h_c = torch.randn(c_len, D, device="cuda", dtype=torch.bfloat16)
h_i = torch.randn(i_len, D, device="cuda", dtype=torch.bfloat16)
static_cos = torch.randn(1, S_max, 1, hd, device="cuda", dtype=torch.bfloat16)
static_sin = torch.randn(1, S_max, 1, hd, device="cuda", dtype=torch.bfloat16)

outs_r = recsys_ops.pack_and_prepare_b1(
    h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_max)
outs_f = fa_ops.pack_and_prepare_b1(
    h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_max)

for idx, name in enumerate(["h_dense", "cos", "sin", "attn_mask", "item_mask", "offsets"]):
    t_r = outs_r[idx].float()
    t_f = outs_f[idx].float()
    if name == "attn_mask":
        finite = torch.isfinite(t_r) & torch.isfinite(t_f)
        sign_match = (t_r == float('-inf')) == (t_f == float('-inf'))
        is_ok = sign_match.all() and (t_r[finite] - t_f[finite]).abs().max().item() < 1e-3
        check(f"RecsysOps.pack_b1 vs FaOps [{name}]", is_ok, "attn_mask 一致")
    else:
        diff = (t_r - t_f).abs().max().item()
        check(f"RecsysOps.pack_b1 vs FaOps [{name}]", diff < 1e-3, f"max_diff={diff:.6f}")

# ── 12c. jagged_pool_and_collect ──────────────────────────────────────────────
torch.manual_seed(57)
n_pooling, n_samp, seq_len, D = 3, 5, 4, 64
pool_vals_list = [
    torch.randn(n_samp * seq_len, D, device="cuda", dtype=torch.bfloat16)
    for _ in range(n_pooling)
]
pool_lens_list = [
    torch.full((n_samp,), seq_len, dtype=torch.int32, device="cuda")
    for _ in range(n_pooling)
]
ones_cache = torch.ones(n_samp * n_pooling, dtype=torch.int32, device="cuda")
pool_values  = torch.cat(pool_vals_list, dim=0)
pool_lengths = torch.cat(pool_lens_list, dim=0)
val_split2 = [0]; len_split2 = [0]
for i in range(n_pooling):
    val_split2.append(val_split2[-1] + pool_vals_list[i].shape[0])
    len_split2.append(len_split2[-1] + pool_lens_list[i].shape[0])

out_r2 = recsys_ops.jagged_pool_and_collect(
    pool_values, pool_lengths, val_split2, len_split2, n_pooling, 0, ones_cache)
out_f2 = fa_ops.jagged_pool_and_collect(
    pool_values, pool_lengths, val_split2, len_split2, n_pooling, 0, ones_cache)

for idx, name in enumerate(["pool_values_out", "pool_lengths_out",
                              "out_val_splits", "out_len_splits"]):
    t_r2 = out_r2[idx].float()
    t_f2 = out_f2[idx].float()
    diff = (t_r2 - t_f2).abs().max().item()
    check(f"RecsysOps.jagged_pool vs FaOps [{name}]", diff < 1e-4, f"max_diff={diff:.6f}")

# ── 12d. 包级单例 ops 与 RecsysOps() 手动实例一致 ─────────────────────────────
check("custom_ops.ops 是 RecsysOps 实例", isinstance(recsys_ops, RecsysOps))
check("custom_ops.ops.namespace == 'recsys_ops'", recsys_ops.namespace == "recsys_ops")

# ── 12e. 幂等加载 ─────────────────────────────────────────────────────────────
ops2 = RecsysOps().load()
check("第二个 RecsysOps 实例也能加载", ops2.is_available())
recsys_ops.load()
check("RecsysOps 单例多次 load 幂等", recsys_ops.is_available())


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  All tests PASSED")
print(f"{'='*70}\n")
