# custom_ops — High-Performance CUDA Operators for Generative Recommendation

[中文](README.md) | English

A high-performance CUDA operator library for generative recommender systems,
with two core operators:

- **`mha_fwd_with_mask`**: a FlashAttention-2 forward implementation supporting
  **arbitrary additive masks** (0 = visible / -inf = masked), deeply optimized for
  consumer GPUs (sm120 / sm89) and significantly faster than PyTorch SDPA with a mask
- **`mixed_gemm`**: a mixed-precision GEMM that solves the **bf16 weight-precision
  loss vs tf32 speed** dilemma of recommendation models — a bf16 tensor-core main
  term plus a low-precision tensor-core residual correction term restores
  near-fp32 accuracy at near-bf16 cost, with bias addition and silu/gelu
  activation fused into the epilogue

**Highlights**

**Attention operator `mha_fwd_with_mask`**

- **Arbitrary masks go straight into softmax**: causal, sliding-window, padding,
  random-sparse (item-level masking) — any pattern, no kernel changes
- **sm120 (RTX 5090D)**: **160–197 TFLOPS** on standard shapes (70–85% of the
  measured cuBLAS bf16 peak), **2–2.7x** over SDPA+mask; up to **53x** on
  small-grid long-sequence shapes
- **sm89 (RTX 4090D)**: works out of the box — **1.1–1.9x** over SDPA+mask on
  standard shapes, up to **34x** on small-grid long-sequence shapes
- **Native GQA**: K/V head count only needs to divide Q head count,
  no manual expansion required
- **Adaptive Split-KV**: a cost model picks the split count automatically;
  large grids fall back to a single kernel with zero overhead — nothing to tune

**Mixed-precision GEMM `mixed_gemm`**

- **Accuracy ≈ fp32, faster than tf32**: weights are split offline into a bf16
  main term plus an fp8/int8 residual correction term, eliminating the systematic
  weight-rounding bias (**2.3x lower RMS error** and **3.9x lower max error** than
  plain bf16); **1.7–2.6x** over fp32 matmul on RTX 5090D (sm120a TMA path,
  another **1.6–3.0x** over the generic path) and **1.3–2.5x** on RTX 4090D,
  up to **118 TFLOPS** effective on large shapes
- **sm120a TMA path**: enabled automatically on RTX 5090-class GPUs — all four
  operand streams are moved via TMA (automatic OOB zero-fill, no per-thread
  predicates) with an mbarrier double-barrier pipeline and a bulk TMA-store
  epilogue; numerically identical to the generic path, with automatic fallback
  whenever alignment constraints are not met
- **Epilogue fusion**: bias addition and silu/gelu activation are fused into the
  GEMM kernel — no extra kernels, no intermediate buffers
- **Dual residual backends**: FP8 e4m3 (SM89+, requires CUDA >= 12.4 at build
  time, slightly faster on small M, no scale storage) and INT8 dynamic
  quantization (SM80+, works with CUDA 11.8, faster on large M) — identical
  accuracy, selected automatically at compile time
- **Adaptive tile/split-K**: the wall-clock heuristics of the original
  TensorRT plugin are retained — small-M shapes get split-K automatically

**Common infrastructure**

- **JIT build**: the first import compiles automatically (multi-process safe),
  `torch.compile`/AOTI compatible

> Full record of the FA porting and optimization journey:
> [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md) (in Chinese).

## Performance

### RTX 5090D (sm120, bf16)

Standard shapes: **160–197 TFLOPS** (70–85% of the measured cuBLAS bf16 peak),
**2–2.7x** over SDPA+mask:

| Shape | custom | SDPA+mask | Speedup |
|---|---|---|---|
| d64 B=4 H=16 S=2048 | 377.4µs (182.1 TF) | 749.3µs | **1.99x** |
| d64 B=32 H=16 S=1024 | 697.2µs (197.1 TF) | 1388.2µs | **1.99x** |
| d128 B=4 H=16 S=2048 | 776.4µs (177.0 TF) | 1873.7µs | **2.41x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 772.1µs (178.0 TF) | 1877.7µs | **2.43x** |

Small-grid + long-sequence shapes (Split-KV kicks in automatically):
**15–53x** over SDPA+mask:

| Shape | custom | SDPA+mask | Speedup |
|---|---|---|---|
| d128 Sq=128 Sk=8192 | 24.7µs | 553.5µs | **22.4x** |
| d128 Sq=128 Sk=32768 | 41.2µs | 2199.0µs | **53.3x** |
| d128 Sq=1024 Sk=8192 | 36.2µs (118.6 TF) | 555.0µs | **15.3x** |
| d64 Sq=128 Sk=8192 | 20.7µs | 432.6µs | **20.9x** |

> Note: with an arbitrary `attn_mask`, PyTorch SDPA can only use the MemEfficient
> backend (the FlashAttention backend does not support arbitrary masks). On some
> shapes this operator **with a mask is even faster than SDPA's mask-free Flash
> backend**.

### RTX 4090D (sm89, bf16)

The sm89 path is a cp.async implementation with the same adaptive Split-KV
ported from sm120: small-grid long-sequence shapes split along the K dimension
to fill the SMs automatically, while large grids fall back to a single kernel —
zero configuration needed.

Test setup: RTX 4090D / PyTorch 2.6.0 / CUDA 11.8, default benchmark settings
(see the [Benchmark](#benchmark) section).

Standard shapes (large grid):

| Shape | custom | SDPA+mask | Speedup |
|---|---|---|---|
| d64 B=1 H=16 S=1024 | 73.7µs (58.3 TF) | 84.0µs | **1.14x** |
| d64 B=4 H=16 S=2048 | 587.8µs (116.9 TF) | 782.3µs | **1.33x** |
| d64 B=1 H=8 S=8192 | 1166.3µs (117.8 TF) | 1686.5µs | **1.45x** |
| d64 B=32 H=16 S=1024 | 1195.0µs (115.0 TF) | 1508.4µs | **1.26x** |
| d128 B=1 H=16 S=1024 | 103.4µs (83.1 TF) | 170.0µs | **1.64x** |
| d128 B=4 H=16 S=2048 | 1103.8µs (124.5 TF) | 1936.4µs | **1.75x** |
| d128 B=1 H=8 S=8192 | 2172.9µs (126.5 TF) | 3811.3µs | **1.75x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 1099.7µs (125.0 TF) | 2040.8µs | **1.86x** |
| d128 B=32 H=16 S=1024 | 2269.2µs (121.1 TF) | 3866.6µs | **1.70x** |

Small-grid + long-sequence shapes (Split-KV kicks in automatically):

| Shape | custom | SDPA+mask | Speedup | SDPA flash* |
|---|---|---|---|---|
| d128 Sq=128 Sk=8192 | 22.5µs (23.8 TF) | 462.8µs | **20.55x** | 29.7µs |
| d128 Sq=128 Sk=32768 | 53.2µs (40.3 TF) | 1824.8µs | **34.27x** | 52.2µs |
| d128 Sq=512 Sk=8192 | 39.9µs (53.8 TF) | 429.1µs | **10.74x** | 37.9µs |
| d128 Sq=1024 Sk=8192 | 55.3µs (77.7 TF) | 428.3µs | **7.74x** | 52.2µs |
| d128 H=2 Hk=1 Sq=512 Sk=16384 | 96.3µs (89.2 TF) | 840.8µs | **8.73x** | 122.9µs |
| d64 Sq=128 Sk=8192 | 19.5µs (13.8 TF) | 412.6µs | **21.21x** | 16.4µs |
| d64 Sq=1024 Sk=8192 | 41.8µs (51.3 TF) | 415.7µs | **9.94x** | 33.8µs |
| d64 H=2 Hk=1 Sq=512 Sk=16384 | 57.3µs (74.9 TF) | 809.0µs | **14.11x** | 69.6µs |

> \* The SDPA flash column is the mask-free FlashAttention backend, shown for
> reference only (it does not support arbitrary masks); the fair comparison is
> the SDPA+mask column. Thanks to Split-KV, some small-grid shapes now match or
> beat even this reference (e.g. d128 H=2 Hk=1 Sq=512 Sk=16384: 96.3µs vs 122.9µs).

For reference: without Split-KV, small-grid long-sequence shapes are processed
by a single CTA serially walking all KV blocks, taking 234–935µs — with
Split-KV enabled automatically this drops to 19–53µs (**10–18x**).

### mixed_gemm (FP8 / INT8 residual backends)

Recommendation MLP layer `silu(x @ W^T + b)` with fp32 weights/activations.

#### RTX 5090D (sm120a TMA path)

Test setup: RTX 5090D / PyTorch 2.11 / CUDA 12.8, FP8 backend. On sm120a the
TMA + mbarrier data path is enabled automatically (numerically identical to the
generic path); the `generic` column is a same-machine A/B run with the path
forced off (`GEMM_MIXED_FORCE_SM80=1`).

The sm120a path is **1.6–3.0x** faster than the generic path and **1.7–2.6x**
faster than fp32 matmul, reaching **118 TFLOPS** effective on 4096³ (72% of a
bf16 matmul on the same shape).

| Shape (M,N,K) | mixed (sm120a) | generic | speedup | fp32 | vs fp32 | tf32 | bf16* |
|---|---|---|---|---|---|---|---|
| 16, 4096, 4096 | 53.2µs | 91.7µs | 1.72x | 112.6µs | **2.11x** | 71.7µs | 53.2µs |
| 64, 4096, 4096 | 57.3µs | 93.9µs | 1.64x | 98.3µs | **1.71x** | 108.5µs | 55.3µs |
| 256, 4096, 4096 | 104.5µs | 313.3µs | **3.00x** | 184.4µs | **1.76x** | 127.3µs | 69.7µs |
| 1024, 4096, 4096 | 337.9µs | 985.1µs | **2.92x** | 598.0µs | **1.77x** | 405.9µs | 219.1µs |
| 4096, 4096, 4096 | 1163.3µs (118.1 TF) | 2198.5µs | 1.89x | 2521.8µs | **2.17x** | 1453.1µs | 843.7µs |
| 16, 16384, 1024 | 53.2µs | 90.1µs | 1.69x | 137.2µs | **2.58x** | 73.7µs | 41.0µs |
| 128, 1000, 2048 | 18.4µs | 43.0µs | **2.34x** | 30.8µs | **1.67x** | 28.6µs | 26.6µs |
| 512, 4096, 256 | 22.5µs | 65.5µs | **2.91x** | 47.1µs | **2.09x** | 31.2µs | 16.4µs |

> \* The bf16 column uses the same methodology as the RTX 4090D section below:
> `bf16(x) @ bf16(W)^T` (weights pre-cast offline) + fp32 epilogue.

Design highlights of the sm120a path (see the file-header comment in
`csrc/mixed_gemm/gemm_bf16xfp32_sm120.cu`):

- All four operand streams (bf16 main + 1-byte residual, for both X and W) are
  moved via TMA: single-thread bulk issue with automatic OOB zero-fill replaces
  per-thread predication, removing the predicate-register and address
  arithmetic overhead of the generic path (sm120 has no wgmma, so the compute
  skeleton stays on `mma.sync`)
- A `full`/`empty` mbarrier double-barrier pipeline replaces cp.async waits +
  `__syncthreads`: fully asynchronous producer/consumer, with barrier phases
  continuing naturally across tiles
- The epilogue uses a bulk TMA store when Y is aligned (automatic clipping of
  out-of-range rows/columns), falling back to bounds-checked elementwise stores
  otherwise; sY aliases the operand smem (protected by a proxy fence)
- Cache-hint split: W (reused across M-tiles) EVICT_LAST, X (streaming)
  EVICT_FIRST
- Routing constraints: k%16==0 and 16B-aligned pointers (TMA global-stride
  requirement); otherwise the op falls back to the generic path with identical
  numerics

#### RTX 4090D (generic path)

Test setup: RTX 4090D / PyTorch 2.11 / CUDA 12.8 (FP8 and INT8 backends measured
on the same machine); when built with CUDA < 12.4, FP8 falls back to INT8
automatically (identical accuracy).

The best generic-path backend per shape runs at **1.3–2.5x** over fp32 matmul
and **1.2–1.6x** over tf32, up to **100 TFLOPS** effective on large shapes;
accuracy is close to fp32 (the systematic weight-rounding bias is fully
eliminated — the remaining error is just unbiased activation rounding noise).

| Shape (M,N,K) | mixed FP8 | mixed INT8 | fp32 | vs fp32† | tf32 | bf16* |
|---|---|---|---|---|---|---|
| 16, 4096, 4096 | 96.3µs | 103.4µs | 122.9µs | **1.28x** | 117.8µs | 61.4µs |
| 64, 4096, 4096 | 100.4µs | 109.6µs | 130.8µs | **1.30x** | 122.9µs | 65.5µs |
| 256, 4096, 4096 | 151.6µs | 145.4µs | 245.8µs | **1.69x** | 210.8µs | 86.8µs |
| 1024, 4096, 4096 | 511.0µs | 440.4µs | 935.0µs | **2.12x** | 602.1µs | 296.6µs |
| 4096, 4096, 4096 | 1730.6µs (79.4 TF) | 1374.2µs (100.0 TF) | 3449.9µs | **2.51x** | 2188.3µs | 1044.5µs |
| 512, 4096, 256 | 30.8µs | 29.7µs | 56.3µs | **1.90x** | 45.9µs | 19.5µs |

Accuracy (4096³, vs the fp32 golden reference; FP8 and INT8 backends measure
identically): mean rel-err mixed **8.0e-3** vs bf16 1.2e-2; RMS error **1.7e-3**
vs 4.0e-3 (2.3x lower); max absolute error **1.6e-2** vs 6.1e-2 (3.9x lower).

Backend selection notes:

- **FP8 is slightly faster on small M (≤64)** (7–10%, simpler quantization
  kernel); **INT8 is faster on large M** (up to 21%); `vs fp32†` takes the better
  backend per row. `backend="auto"` picks FP8 when available (identical
  accuracy, no per-channel scale storage); for maximum large-M throughput pass
  `backend="int8"` explicitly
- The two residual backends are accuracy-identical: the weight-rounding bias is
  eliminated either way, and the dominant remaining error is the unbiased bf16
  rounding noise of the activations (both quantization grids are fine enough)
- Building with CUDA 11.8 makes the same INT8 kernel ~17% slower on large shapes
  (nvcc codegen differences; 4096³ measures 1658µs) — small shapes are
  unaffected; prefer a recent CUDA toolkit

> \* The bf16 column is `bf16(x) @ bf16(W)^T` (weights pre-cast offline, on par
> with mixed_gemm's offline weight split) + fp32 epilogue: 1.3–1.7x faster (a
> single GEMM vs the main-plus-correction dual GEMM) but with fully uncompensated
> weight-rounding error.

## Requirements

- NVIDIA GPU: sm120 (RTX 5090D, both the FA TMA main path and the mixed_gemm
  sm120a TMA path extensively tested) or sm89
  (RTX 4090D, cp.async path with Split-KV ported from sm120, tested); other sm80+
  architectures should compile but are unverified
- CUDA >= 11.8 (FA sm89 path / mixed_gemm INT8 backend) / >= 12.4 (mixed_gemm FP8
  backend, SM89+) / >= 12.8 (FA / mixed_gemm sm120a paths), GCC >= 9
- PyTorch >= 2.1 (CUDA build), bf16

## Quick Start

The repository root is itself the Python package `custom_ops` — no installation
needed. Just add the **parent directory** of the repo to `PYTHONPATH`:

```bash
git clone <repo_url> custom_ops
export PYTHONPATH=$(dirname $(pwd)/custom_ops):$PYTHONPATH   # or use sys.path.insert in code
```

```python
import torch
from custom_ops import ops   # first import triggers JIT build (~30–60s); cached afterwards

B, H, Sq, Sk, d = 2, 16, 1024, 1024, 128
q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)
mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16)
mask[..., 512:] = float("-inf")   # arbitrary additive mask: 0 = visible, -inf = masked

out = ops.mha_fwd_with_mask(q, k, v, mask)   # (B, H, Sq, d) bf16

# ── Mixed-precision GEMM: y = silu(x @ W^T + b) ───────────────────
from custom_ops import split_mixed_precision_weight

x = torch.randn(4096, 4096, device="cuda")              # fp32 activations
w = torch.randn(4096, 4096, device="cuda") * 0.05      # fp32 weights
b = torch.randn(4096, device="cuda") * 0.1

w_high, w_low, w_scale = split_mixed_precision_weight(w)  # one-time split at model load
y = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=b,
                   activation="silu")                   # fp32 output, accuracy ≈ fp32
```

Build artifacts are cached in `~/.cache/torch_extensions` by default; override with
`TORCH_EXTENSIONS_DIR`. Concurrent first-time builds from multiple processes are
protected by a file lock.

## API

### `ops.mha_fwd_with_mask(q, k, v, mask) -> Tensor`

FlashAttention-2 forward with arbitrary bf16 additive mask.

| Argument | Shape | Description |
|---|---|---|
| `q` | (B, H, Sq, d) | bf16 CUDA contiguous tensor |
| `k`, `v` | (B, Hk, Sk, d) | bf16 CUDA contiguous tensors |
| `mask` | (B, 1, Sq, Sk) | bf16 additive mask: 0 = visible / -inf = masked |
| Returns | (B, H, Sq, d) | bf16 |

Limitations:

- Head dim `d ∈ {64, 128}`
- `H % Hk == 0` (GQA); MHA (Hk = H) is the special case
- Forward only; no dropout / causal flag / alibi / RoPE / KV-cache
  (causal can be expressed via the mask — see examples)

### `ops.mixed_gemm(x, w_high, w_low, w_scale, *, scale, bias, activation, out_dtype, force_splitk) -> Tensor`

Mixed-precision GEMM: `y = activation(x @ (w_high + w_low * scale)^T + bias)`.
A bf16 tensor-core main term plus a low-precision tensor-core residual correction
term restores near-fp32 accuracy at near-bf16 cost; bias and activation are fused
into the epilogue.

| Argument | Shape | Description |
|---|---|---|
| `x` | (..., K) | fp32 / bf16 CUDA contiguous tensor; leading dims fold into M |
| `w_high` | (N, K) | bf16 main-term weight (produced by `split_mixed_precision_weight`) |
| `w_low` | (N, K) | residual weight: `float8_e4m3fn` (FP8 backend) or `int8` (INT8 backend) |
| `w_scale` | (N,) | fp32 per-channel quantization scale for the INT8 backend (`None` for FP8) |
| `scale` | float | residual compensation scale; must match the weight split (default 1/256) |
| `bias` | (N,) | fp32, optional, fused into the epilogue |
| `activation` | str | `"identity"` / `"silu"` / `"gelu"` (tanh approximation) |
| `out_dtype` | dtype | `None` (fp32, recommended) or `torch.bfloat16` |
| Returns | (..., N) | output, dtype per `out_dtype` |

The residual backend is selected by `w_low.dtype`; `mixed_gemm_fp8_available()`
queries FP8 backend availability (CUDA >= 12.4 at build time and an SM89+ GPU).

Limitations:

- `K % 8 == 0`
- FP8 backend requires SM89+; INT8 backend requires SM80+

### `split_mixed_precision_weight(w, scale=1/256, backend="auto") -> (w_high, w_low, w_scale)`

Offline (one-time, at model load) split of fp32 weights into the triple needed
by mixed_gemm: `w ≈ w_high + w_low * scale`. With `backend="auto"` the FP8
backend is used when available on the machine, falling back to INT8 otherwise.
Pure PyTorch — no compiled operator required.

### Tuning environment variables (default `auto` is near-optimal; for tuning/debug only)

| Env var | Effect |
|---|---|
| `FA_NUM_SPLITS=n` | Force the Split-KV split count (0 = auto cost model) |
| `FA_SPLITKV=0` | Disable Split-KV |
| `FA_PERSISTENT=1` | Enable the persistent kernel (+2–5% for some d128 shapes) |
| `GEMM_MIXED_FORCE_SPLITK=n` | Force the mixed_gemm split-K value (1/2/4/8/16, debug only) |

## Examples

```bash
python examples/basic_usage.py        # minimal example: invocation + SDPA verification
python examples/custom_mask_demo.py   # 4 typical masks: causal / sliding-window / padding / random-sparse
python examples/gqa_example.py        # GQA: no K/V head expansion needed
python examples/mixed_gemm_demo.py    # mixed-precision GEMM: weight split + epilogue fusion + accuracy
```

## Benchmark

```bash
# Full run: standard shapes + small-grid long-sequence (Split-KV) shapes
python benchmark/benchmark_fa.py

# A specific suite / a single shape
python benchmark/benchmark_fa.py --suite standard
python benchmark/benchmark_fa.py --suite splitkv
python benchmark/benchmark_fa.py --shape 4 16 16 2048 2048 128

# Tune iterations and output
python benchmark/benchmark_fa.py --warmup 20 --iters 100 --mask-ratio 0.3 --csv result.csv
```

For each shape the script reports latency, TFLOPS and speedup for
custom / SDPA+mask / SDPA flash (reference), and runs a numerical check
against SDPA.

```bash
# mixed_gemm: latency and accuracy vs fp32 / tf32 / bf16 matmul
python benchmark/benchmark_mixed_gemm.py                       # backend auto (FP8 if available)
python benchmark/benchmark_mixed_gemm.py --backend int8        # pick the residual backend
python benchmark/benchmark_mixed_gemm.py --shape 4096 4096 4096 --csv result.csv
```

## Repository Structure

```
custom_ops/
├── __init__.py            # CustomOps: generic JIT operator loading framework (base class)
├── recsys.py              # RecsysOps: mha_fwd_with_mask / mixed_gemm wrappers + weight-split helper
├── csrc/
│   ├── recsys_bindings.cpp        # torch.ops registration entry
│   ├── fa/                        # FA2 + mask kernel (sm89 + sm120 paths, decoupled)
│   │   ├── fa_fwd_op.cu           # operator entry / arch dispatch
│   │   ├── fa_fwd_launch.h        # per-arch launchers + Split-KV cost model
│   │   ├── sm89/fa_fwd_kernel.h   # sm89 cp.async kernel: base / splitkv / combine
│   │   ├── sm120/fa_fwd_sm120.h   # sm120 TMA pipeline kernel / splitkv / persistent / combine
│   │   ├── cpu/                   # CPU reference implementation (not wired into dispatch)
│   │   └── common/                # shared params / softmax / utils
│   └── mixed_gemm/                # mixed-precision GEMM kernel (ported from a TRT plugin)
│       ├── mixed_gemm_op.cu       # torch operator entry: validation / workspace / dispatch
│       ├── gemm_bf16xfp32_sm80.cu # kernel: tile/split-K heuristics + bf16 main + fp8/int8 residual dual GEMM
│       ├── gemm_bf16xfp32_sm80.h  # kernel entry declarations + FP8 compile-time guard
│       ├── gemm_bf16xfp32_sm120.cu # sm120a TMA+mbarrier kernel (auto-routed, decoupled from the sm80 path)
│       └── gemm_bf16xfp32_sm120.h # sm120a path entry declarations (support/alignment checks)
├── thirdparty/            # CUTLASS / CuTe (header-only dependencies)
├── benchmark/             # performance benchmark
├── examples/              # usage examples
├── tests/                 # numerical correctness tests
└── docs/                  # full porting & optimization record (roofline / NCU analysis)
```

## Reusing the CustomOps Framework

The `CustomOps` base class can be reused for any PyTorch CUDA custom operator
library — just subclass it and override a few configuration attributes:

```python
from custom_ops import CustomOps

class MyOps(CustomOps):
    namespace    = "my_ops"           # torch.ops.my_ops.<name>(...)
    so_name      = "my_ops_kernel"
    required_ops = ["my_op"]

    def get_sources(self):
        return ["csrc/bindings.cpp", "csrc/my_op.cu"]

    def get_include_dirs(self):
        return ["thirdparty"]

ops = MyOps().load()
if ops.is_available():
    out = ops.my_op(x)
```

You get out of the box: automatic JIT build / fast dlopen, multi-process file
locking, GPU arch auto-detection, GCC configuration, graceful degradation, and
`torch.compile`/AOTI fake registration.

## Acknowledgments & Citations

The kernel implementation in this repository is derived from the official
FlashAttention source code (the compute skeleton follows FA2, while the data
movement path adopts the FA3/hopper-style TMA design ported to sm120), and it
depends on NVIDIA CUTLASS/CuTe (vendored as headers under `thirdparty/`).
If this project helps you, please also cite the original works:

- FlashAttention official repo: https://github.com/Dao-AILab/flash-attention
- CUTLASS: https://github.com/NVIDIA/cutlass

```bibtex
@inproceedings{dao2022flashattention,
  title     = {FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{dao2023flashattention2,
  title     = {FlashAttention-2: Better Attention with Better Parallelism and Work Partitioning},
  author    = {Dao, Tri},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

## License

This project is licensed under [BSD 3-Clause](LICENSE).

- The code in this repository is derived from
  [FlashAttention](https://github.com/Dao-AILab/flash-attention) (BSD 3-Clause);
  the original copyright notices are retained in the corresponding source files;
- CUTLASS/CuTe under `thirdparty/` are governed by their original BSD 3-Clause
  license — see [thirdparty/LICENSE.cutlass](thirdparty/LICENSE.cutlass).
