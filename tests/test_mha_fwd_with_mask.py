#!/usr/bin/env python3
"""mha_fwd_with_mask correctness test vs manual reference (sm89 / sm120).

Covers both supported dtypes on the SM80+ paths (fp16 and bf16 — SM80+
tensor cores natively support both; the launcher instantiates the matching
elem_type variant). Runs on any CUDA GPU >= sm80; the sm70 (V100) fp16 path
has its own test (test_sm70_correctness.py).

Kernel-path coverage by shape (auto-routed, see fa_fwd_op.cu):
  - base kernel       : large grid (B*H*num_m_blocks >= 0.8 * num_SMs)
  - Split-KV          : small grid + long seqlen_k (cost model picks splits)
  - Split-KV + Split-M: tiny grid, d64 (kBlockM=64 variant)

Mask semantics: additive, SDPA-compatible — 0=visible, -inf=masked, any
finite value is a bias (ALiBi style). Random finite masks are included
deliberately: 0/-inf-only masks hide scale-semantics bugs.

Alignment contract (no internal padding): Sk must be a multiple of 8
(128-bit cp.async / TMA 16B row alignment); Sq is unconstrained (row
predicates + epilogue drops). mask may be larger than (Sq, Sk) — content
beyond seqlen_k is ALWAYS masked regardless of what it holds (padded
columns filled with visible values are a deliberate trap in
test_mask_padded).
"""
import torch
from custom_ops import ops

torch.manual_seed(42)

# fp16 输出量化 ~2^-11、bf16 ~2^-8；与 fp32 参考对齐后 1e-2 双向容差足够捕捉
# 映射/归约类错误（这类错误通常表现为 O(0.1+) 的系统性偏差）
ATOL = 1e-2
RTOL = 1e-2

DTYPES = [torch.float16, torch.bfloat16]


def ref_attn(q, k, v, mask=None):
    """Manual fp32 softmax attention reference. q/k/v: (b, h, s, d).

    全屏蔽行（mask 全 -inf）约定输出 0，与 kernel 一致
    （normalize_softmax_lse 对 sum==0 的 inv_sum=1 守卫）。
    """
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.matmul(q.float() * scale, k.float().transpose(-2, -1))
    if mask is not None:
        scores = scores + mask.float()
    # 全屏蔽行：amax=-inf → NaN；用有限下界替换后 -inf - (-1e30) = -inf → exp=0
    scores = scores - scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    scores = torch.exp(scores)
    denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-38)  # 0/ε = 0
    # P 矩阵量化到输入 dtype 再做 PV（与 kernel 的 P-quantization 语义一致）
    return torch.matmul((scores / denom).to(v.dtype), v)


def _run_case(name, q, k, v, mask, expect_dtype):
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()
    assert out.dtype == expect_dtype, f"{name}: dtype {out.dtype} != {expect_dtype}"
    ref = ref_attn(q, k, v, mask).to(expect_dtype)
    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    ref_abs = ref.float().abs().max().item() + 1e-8
    passed = max_err < ATOL or (max_err / ref_abs) < RTOL
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:44s} max_err={max_err:.4e}")
    return passed


def test_case(b, h, sq, sk, d, mask_kind, dtype, name):
    """One correctness case. mask_kind: zero | causal | random | block | all_masked_row."""
    q = (torch.randn(b, h, sq, d, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(b, h, sk, d, device="cuda") * 0.5).to(dtype)
    v = (torch.randn(b, h, sk, d, device="cuda") * 0.5).to(dtype)
    if mask_kind == "zero":
        mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
    elif mask_kind == "causal":
        tri = torch.triu(torch.full((sq, sk), float("-inf"), device="cuda"), diagonal=1)
        mask = tri.unsqueeze(0).unsqueeze(0).expand(b, 1, sq, sk).contiguous().to(dtype)
    elif mask_kind == "random":
        mask = (torch.randn(b, 1, sq, sk, device="cuda") * 0.3).to(dtype)
    elif mask_kind == "block":
        mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
        mask[..., sk // 2:] = float("-inf")
    else:  # all_masked_row: first query row fully masked -> output must be 0
        mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
        mask[:, :, 0, :] = float("-inf")
    return _run_case(f"{name}[{str(dtype).split('.')[-1]}]", q, k, v, mask, dtype)


def test_mask_padded(b, h, sq, sk, d, dtype, pad_rows=0):
    """Pre-padded mask (Sq_mask > Sq / Sk_mask > Sk): pad columns are filled
    with random values (visible semantics — a trap); the kernel must ignore
    mask content beyond (Sq, Sk): col >= Sk is ALWAYS masked, extra rows are
    dropped by the epilogue. Reference = attention with the mask cropped
    to (Sq, Sk) (cropped columns ≡ masked)."""
    q = (torch.randn(b, h, sq, d, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(b, h, sk, d, device="cuda") * 0.5).to(dtype)
    v = (torch.randn(b, h, sk, d, device="cuda") * 0.5).to(dtype)
    # pad 列/行填非零随机值（若 kernel 误读为可见，输出将系统性偏离）
    mask = (torch.randn(b, 1, sq + pad_rows, sk + 8, device="cuda") * 0.3).to(dtype)
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()
    ref = ref_attn(q, k, v, mask[:, :, :sq, :sk].contiguous()).to(dtype)
    max_err = (out.float() - ref.float()).abs().max().item()
    passed = max_err < ATOL
    print(f"[{'PASS' if passed else 'FAIL'}] "
          f"mask_padded(rows=+{pad_rows}, cols=+8)[{str(dtype).split('.')[-1]:8s}]"
          f"{'':14s} max_err={max_err:.4e}")
    return passed


def test_gqa(b, h, hk, sq, sk, d, dtype):
    q = (torch.randn(b, h, sq, d, device="cuda") * 0.5).to(dtype)
    k = (torch.randn(b, hk, sk, d, device="cuda") * 0.5).to(dtype)
    v = (torch.randn(b, hk, sk, d, device="cuda") * 0.5).to(dtype)
    mask = torch.zeros(b, 1, sq, sk, dtype=dtype, device="cuda")
    rep = h // hk
    k_ref = k.repeat_interleave(rep, dim=1)
    v_ref = v.repeat_interleave(rep, dim=1)
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()
    ref = ref_attn(q, k_ref, v_ref, mask).to(dtype)
    max_err = (out.float() - ref.float()).abs().max().item()
    passed = max_err < ATOL
    print(f"[{'PASS' if passed else 'FAIL'}] gqa(h={h},hk={hk})[{str(dtype).split('.')[-1]:8s}]"
          f"{'':18s} max_err={max_err:.4e}")
    return passed


def test_error_paths(dtype):
    ok = True
    # Sk 非 8 倍数：算子层自动 pad（p2 新契约），不再报错且结果正确
    try:
        q = torch.randn(1, 1, 64, 64, device="cuda").to(dtype)
        k = torch.randn(1, 1, 100, 64, device="cuda").to(dtype)
        v = torch.randn(1, 1, 100, 64, device="cuda").to(dtype)
        m = torch.zeros(1, 1, 64, 100, dtype=dtype, device="cuda")
        m[..., ::7] = float("-inf")
        out = ops.mha_fwd_with_mask(q, k, v, m)
        ref = ref_attn(q, k, v, m)
        max_err = (out.float() - ref.float()).abs().max().item()
        passed = max_err < ATOL
        print(f"[{'PASS' if passed else 'FAIL'}] error_paths: Sk%8!=0 autopad "
              f"(max_err={max_err:.4e})")
        ok = ok and passed
    except RuntimeError as e:
        print(f"[FAIL] error_paths: Sk%8!=0 autopad raised: {e}")
        ok = False
    # fp32 输入应报错
    try:
        q = torch.randn(1, 1, 64, 64, device="cuda")
        m = torch.zeros(1, 1, 64, 64, device="cuda")
        ops.mha_fwd_with_mask(q, q, q, m)
        print("[FAIL] error_paths: fp32 input should raise")
        ok = False
    except RuntimeError:
        print("[PASS] error_paths: fp32 input raises")
    # dtype 不匹配应报错
    try:
        q = torch.randn(1, 1, 64, 64, device="cuda").to(dtype)
        k = torch.randn(1, 1, 64, 64, device="cuda").to(
            torch.bfloat16 if dtype == torch.float16 else torch.float16)
        m = torch.zeros(1, 1, 64, 64, dtype=dtype, device="cuda")
        ops.mha_fwd_with_mask(q, k, q, m)
        print("[FAIL] error_paths: mixed dtype should raise")
        ok = False
    except RuntimeError:
        print("[PASS] error_paths: mixed dtype raises")
    return ok


if __name__ == "__main__":
    print("=" * 80)
    print("mha_fwd_with_mask correctness test (sm89/sm120, fp16 + bf16)")
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 80)

    all_pass = True

    for dtype in DTYPES:
        tag = f"d{str(dtype).split('.')[-1]}"
        # ── base kernel 路径（大 grid，无 split）──
        for b, h, sq, sk, d in [(2, 16, 256, 256, 64), (2, 16, 256, 256, 128),
                                (1, 24, 512, 384, 64), (1, 24, 384, 512, 128)]:
            all_pass &= test_case(b, h, sq, sk, d, "zero", dtype, f"base_{tag}_d{d}")

        # ── mask 语义 ──
        all_pass &= test_case(2, 16, 256, 256, 64, "causal", dtype, f"causal_{tag}_d64")
        all_pass &= test_case(2, 16, 256, 256, 128, "causal", dtype, f"causal_{tag}_d128")
        all_pass &= test_case(2, 16, 256, 256, 64, "random", dtype, f"random_{tag}_d64")
        all_pass &= test_case(2, 16, 256, 256, 128, "random", dtype, f"random_{tag}_d128")
        all_pass &= test_case(2, 16, 256, 256, 64, "block", dtype, f"block_{tag}_d64")
        all_pass &= test_case(2, 16, 256, 256, 128, "block", dtype, f"block_{tag}_d128")
        all_pass &= test_case(2, 16, 256, 256, 64, "all_masked_row", dtype, f"allmask_{tag}_d64")

        # ── 非块对齐边界形状（新契约：Sk%8==0 即可，Sq 任意）──
        all_pass &= test_case(1, 8, 100, 104, 64, "zero", dtype, f"unaligned_{tag}_d64")
        all_pass &= test_case(1, 8, 100, 200, 128, "random", dtype, f"unaligned_{tag}_d128")

        # ── 非对齐专项：tail 覆盖 40/72/8（kBlockN=128/64 边界残块）+ 极端形状 ──
        # sk=1032 = 16*64+8：多块 + 尾块 tail=8（长序列覆盖 splitkv 守卫路径）
        all_pass &= test_case(1, 8, 256, 1032, 128, "random", dtype, f"tail8_{tag}_d128")
        # sk=56 < kBlockN（单块 tail）、sk=8（最小对齐单位）
        all_pass &= test_case(1, 8, 64, 56, 64, "random", dtype, f"tail_single_{tag}_d64")
        all_pass &= test_case(1, 8, 128, 8, 128, "causal", dtype, f"sk_min_{tag}_d128")
        # sq=1：Sq 完全无对齐要求的极端（行谓词 + epilogue 裁剪）
        all_pass &= test_case(1, 4, 1, 104, 64, "random", dtype, f"sq1_{tag}_d64")
        # mask 预 pad：pad 列填非零陷阱值——越界列内容必须被忽略（col ≥ Sk 恒屏蔽）；
        # pad_rows>0 时验证 Sq_mask > Sq 的多余行被丢弃
        all_pass &= test_mask_padded(1, 8, 100, 104, 64, dtype)
        all_pass &= test_mask_padded(1, 8, 100, 104, 64, dtype, pad_rows=8)
        # Split-KV + 非对齐 Sk：全局边界 tile 归属最后一个 split（n_end 守卫）
        all_pass &= test_case(1, 2, 128, 1032, 128, "random", dtype, f"splitkv_unaligned_{tag}")

        # ── Split-KV 路径（小 grid + 长序列；cost model 自动选择 num_splits）──
        # d64: total_mblocks 远小于 SM 数 → Split-KV + Split-M(M64)
        all_pass &= test_case(1, 1, 128, 8192, 64, "zero", dtype, f"splitkv_m64_{tag}_d64")
        all_pass &= test_case(1, 1, 64, 4096, 64, "causal", dtype, f"splitkv_m64_{tag}_d64c")
        # d64: total_mblocks ∈ [SM/2, 0.8*SM) → Split-KV 不带 M64（4090D: 128 SM）
        all_pass &= test_case(1, 40, 256, 4096, 64, "zero", dtype, f"splitkv_{tag}_d64")
        # d128: 小 grid 长 K → Split-KV（DB/SB 变体按 tiles_per_cta 自动分发）
        all_pass &= test_case(1, 2, 128, 8192, 128, "zero", dtype, f"splitkv_{tag}_d128")
        all_pass &= test_case(1, 2, 256, 16384, 128, "random", dtype, f"splitkv_{tag}_d128r")

        # ── GQA ──
        all_pass &= test_gqa(2, 16, 4, 256, 256, 64, dtype)
        all_pass &= test_gqa(1, 32, 8, 512, 384, 128, dtype)

    all_pass &= test_error_paths(torch.float16)

    print("=" * 80)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)
    raise SystemExit(0 if all_pass else 1)
