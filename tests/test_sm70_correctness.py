#!/usr/bin/env python3
"""sm70 FA correctness test vs manual reference (fp16, V100).

Validating the sm70 kernel itself requires a real V100 (sm_70): the kernel
hand-codes the WMMA m16n16k16 fragment lane mapping measured on Volta, which
has no cross-architecture contract (it differs on sm89+). The FA_FORCE_SM70
dev bypass is therefore NOT a valid correctness channel for this kernel.

On non-V100 GPUs (SM80+) this test still runs and passes: fp16 inputs are
routed to the generic SM80+ path (see test_mha_fwd_with_mask.py for its
full coverage) — this exercises the same shapes against that path instead.

Alignment contract (no internal padding): Sk must be a multiple of 8
(128-bit copy alignment); Sq is unconstrained. mask may be larger than
(Sq, Sk) — content beyond seqlen_k is always masked (padded columns are
ignored regardless of what they hold).
"""
import torch
import torch.nn.functional as F
from custom_ops import ops

torch.manual_seed(42)
atol = 5e-3
rtol = 5e-3

def ref_attn(q, k, v, mask=None, scale=None):
    """Manual softmax attention reference."""
    # q/k/v: (b, h, s, d)
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    qt = q * scale  # (b,h,s,d)
    # scores: (b,h,sq,sk)
    scores = torch.matmul(qt, k.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask
    # softmax in fp32
    scores = scores.float()
    scores = scores - scores.amax(dim=-1, keepdim=True)
    scores = torch.exp(scores)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    out = torch.matmul(scores.to(q.dtype), v)
    return out

def test_case(b, h, s_q, s_k, d, mask_val=0.0, name=""):
    """Run one correctness case."""
    q = (torch.randn(b, h, s_q, d, device="cuda") * 0.5).half()
    k = (torch.randn(b, h, s_k, d, device="cuda") * 0.5).half()
    v = (torch.randn(b, h, s_k, d, device="cuda") * 0.5).half()
    mask = torch.full((b, 1, s_q, s_k), mask_val, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        ref = ref_attn(q.float(), k.float(), v.float(),
                       mask.float() if mask is not None else None,
                       scale=1.0/(d**0.5)).half()

    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()

    diff = (out - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ref_abs = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_abs) < rtol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:40s} b={b} h={h} sq={s_q} sk={s_k} d={d} mask={mask_val} "
          f"max_err={max_err:.4e} mean_err={mean_err:.4e}")
    return passed

def test_causal(b, h, s, d):
    """Causal mask test."""
    q = (torch.randn(b, h, s, d, device="cuda") * 0.5).half()
    k = (torch.randn(b, h, s, d, device="cuda") * 0.5).half()
    v = (torch.randn(b, h, s, d, device="cuda") * 0.5).half()
    causal = torch.triu(torch.full((s, s), float('-inf'), dtype=torch.float16, device="cuda"), diagonal=1)
    mask = causal.unsqueeze(0).unsqueeze(0).expand(b, 1, s, s).contiguous()

    with torch.no_grad():
        ref = ref_attn(q.float(), k.float(), v.float(),
                       mask.float(), scale=1.0/(d**0.5)).half()
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()

    diff = (out - ref).abs()
    max_err = diff.max().item()
    ref_abs = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_abs) < rtol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {'causal':40s} b={b} h={h} s={s} d={d} "
          f"max_err={max_err:.4e}")
    return passed

def test_random_mask(b, h, s, d):
    """Random additive mask test."""
    q = (torch.randn(b, h, s, d, device="cuda") * 0.5).half()
    k = (torch.randn(b, h, s, d, device="cuda") * 0.5).half()
    v = (torch.randn(b, h, s, d, device="cuda") * 0.5).half()
    mask = (torch.randn(b, 1, s, s, device="cuda") * 0.3).half()

    with torch.no_grad():
        ref = ref_attn(q.float(), k.float(), v.float(),
                       mask.float(), scale=1.0/(d**0.5)).half()
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()

    diff = (out - ref).abs()
    max_err = diff.max().item()
    ref_abs = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_abs) < rtol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {'random_mask':40s} b={b} h={h} s={s} d={d} "
          f"max_err={max_err:.4e}")
    return passed

def test_mask_padded(b, h, s_q, s_k, d):
    """Pre-padded mask trap: pad columns hold nonzero values (visible
    semantics) — the kernel must ignore mask content beyond Sk
    (col >= Sk is always masked). Reference = mask cropped to (Sq, Sk)."""
    q = (torch.randn(b, h, s_q, d, device="cuda") * 0.5).half()
    k = (torch.randn(b, h, s_k, d, device="cuda") * 0.5).half()
    v = (torch.randn(b, h, s_k, d, device="cuda") * 0.5).half()
    mask = (torch.randn(b, 1, s_q, s_k + 8, device="cuda") * 0.3).half()

    with torch.no_grad():
        ref = ref_attn(q.float(), k.float(), v.float(),
                       mask[:, :, :s_q, :s_k].float().contiguous(),
                       scale=1.0/(d**0.5)).half()
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()

    max_err = (out - ref).abs().max().item()
    ref_abs = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_abs) < rtol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {'mask_padded_trap':40s} b={b} h={h} sq={s_q} sk={s_k} d={d} "
          f"max_err={max_err:.4e}")
    return passed


def test_gqa(b, h, hk, s_q, s_k, d):
    """GQA test (H != Hk): K/V head count divides Q head count."""
    q = (torch.randn(b, h, s_q, d, device="cuda") * 0.5).half()
    k = (torch.randn(b, hk, s_k, d, device="cuda") * 0.5).half()
    v = (torch.randn(b, hk, s_k, d, device="cuda") * 0.5).half()
    mask = torch.zeros(b, 1, s_q, s_k, dtype=torch.float16, device="cuda")

    # Reference: expand K/V heads to match Q
    rep = h // hk
    k_ref = k.repeat_interleave(rep, dim=1)
    v_ref = v.repeat_interleave(rep, dim=1)
    with torch.no_grad():
        ref = ref_attn(q.float(), k_ref.float(), v_ref.float(),
                       mask.float(), scale=1.0/(d**0.5)).half()
    out = ops.mha_fwd_with_mask(q, k, v, mask)
    torch.cuda.synchronize()

    diff = (out - ref).abs()
    max_err = diff.max().item()
    ref_abs = ref.abs().max().item() + 1e-8
    passed = max_err < atol or (max_err / ref_abs) < rtol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {'gqa':40s} b={b} h={h} hk={hk} sq={s_q} sk={s_k} d={d} "
          f"max_err={max_err:.4e}")
    return passed


if __name__ == "__main__":
    print("=" * 80)
    print("sm70 FA correctness test (fp16, V100)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cap: {torch.cuda.get_device_capability(0)}")
    print("=" * 80)

    all_pass = True

    # Basic shapes, d=64
    for b, h, sq, sk in [(1,1,64,64), (1,1,128,128), (1,1,128,256), (1,1,256,128), (2,4,128,128)]:
        all_pass &= test_case(b, h, sq, sk, 64, 0.0, "zero_mask_d64")

    # d=128
    for b, h, sq, sk in [(1,1,64,64), (1,1,128,128), (1,1,128,256), (2,4,128,128)]:
        all_pass &= test_case(b, h, sq, sk, 128, 0.0, "zero_mask_d128")

    # Non-zero mask
    all_pass &= test_case(1, 1, 128, 128, 64, -1.0, "neg_mask_d64")
    all_pass &= test_case(1, 1, 128, 128, 128, -1.0, "neg_mask_d128")

    # Causal
    all_pass &= test_causal(1, 1, 64, 64)
    all_pass &= test_causal(1, 1, 128, 64)
    all_pass &= test_causal(1, 1, 128, 128)

    # Random mask
    all_pass &= test_random_mask(1, 1, 64, 64)
    all_pass &= test_random_mask(1, 1, 128, 128)
    all_pass &= test_random_mask(2, 4, 128, 128)

    # ── 非块对齐形状（新契约：Sk%8==0 即可，Sq 任意）──
    all_pass &= test_case(1, 1, 100, 104, 64, 0.0, "unaligned_d64")
    all_pass &= test_case(1, 1, 100, 200, 64, 0.0, "unaligned2_d64")
    all_pass &= test_case(1, 1, 100, 104, 128, 0.0, "unaligned_d128")
    # sk < kBlockN（单块 tail）/ sk = 8（最小对齐单位）/ sq = 1（行谓词极端）
    all_pass &= test_case(1, 1, 64, 56, 64, 0.0, "tail_single_d64")
    all_pass &= test_case(1, 1, 128, 8, 64, 0.0, "sk_min_d64")
    all_pass &= test_case(1, 1, 1, 104, 64, 0.0, "sq1_d64")
    # mask 预 pad：pad 列填非零陷阱值，越界列内容必须被忽略
    all_pass &= test_mask_padded(1, 1, 100, 104, 64)

    # GQA (H != Hk)
    all_pass &= test_gqa(1, 4, 2, 128, 128, 64)
    all_pass &= test_gqa(1, 4, 1, 128, 256, 128)
    all_pass &= test_gqa(2, 8, 2, 128, 128, 128)

    print("=" * 80)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)
