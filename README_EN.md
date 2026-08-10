# custom_ops — FlashAttention-2 with Arbitrary Mask (Optimized for sm120/Blackwell)

[中文](README.md) | English

A high-performance CUDA operator library for generative recommender systems. The core
operator `mha_fwd_with_mask` is a FlashAttention-2 forward implementation supporting
**arbitrary additive masks** (0 = visible / -inf = masked), deeply optimized for
consumer Blackwell GPUs (RTX 5090D, sm120) with TMA + mbarrier multi-stage pipelining
and adaptive Split-KV parallelism — significantly outperforming PyTorch SDPA.

> Full record of the porting and optimization journey:
> [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md) (in Chinese).

## Features

- **Arbitrary additive mask**: bf16 mask participates directly in softmax
  (0 = visible / -inf = masked) — supports causal, sliding-window, padding,
  random sparse (item-level masking), or any custom pattern
- **Native GQA support**: K/V head count only needs to divide Q head count,
  no manual expansion required
- **Adaptive Split-KV**: automatically splits along the K dimension for
  small-grid long-sequence shapes to improve SM utilization; a cost model picks
  the near-optimal split count, while large-grid shapes fall back to a single
  kernel with zero overhead
- **JIT compilation framework**: the `CustomOps` base class provides automatic
  build/load, multi-process file locking, GPU arch auto-detection, and
  `torch.compile`/AOTI fake registration — reusable for any custom operator library

## Performance (RTX 5090D, sm120, bf16)

Standard shapes: **160–197 TFLOPS** (70–85% of the measured cuBLAS bf16 peak),
**2–2.7x** over SDPA:

| Shape | custom | SDPA+mask | Speedup |
|---|---|---|---|
| d64 B=4 H=16 S=2048 | 377.4µs (182.1 TF) | 749.3µs | **1.99x** |
| d64 B=32 H=16 S=1024 | 697.2µs (197.1 TF) | 1388.2µs | **1.99x** |
| d128 B=4 H=16 S=2048 | 776.4µs (177.0 TF) | 1873.7µs | **2.41x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 772.1µs (178.0 TF) | 1877.7µs | **2.43x** |

Small-grid + long-sequence shapes (Split-KV kicks in automatically):
**15–53x** over SDPA:

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

## Requirements

- NVIDIA GPU with compute capability >= 9.0 (Hopper / Blackwell; requires TMA.
  Extensively tested on sm120)
- CUDA >= 12.8 (for sm120a support), GCC >= 9
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

### Tuning environment variables (default `auto` is near-optimal; for tuning/debug only)

| Env var | Effect |
|---|---|
| `FA_NUM_SPLITS=n` | Force the Split-KV split count (0 = auto cost model) |
| `FA_SPLITKV=0` | Disable Split-KV |
| `FA_PERSISTENT=1` | Enable the persistent kernel (+2–5% for some d128 shapes) |

## Examples

```bash
python examples/basic_usage.py        # minimal example: invocation + SDPA verification
python examples/custom_mask_demo.py   # 4 typical masks: causal / sliding-window / padding / random-sparse
python examples/gqa_example.py        # GQA: no K/V head expansion needed
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

## Repository Structure

```
custom_ops/
├── __init__.py            # CustomOps: generic JIT operator loading framework (base class)
├── recsys.py              # RecsysOps: wrapper for the mha_fwd_with_mask operator
├── csrc/
│   ├── recsys_bindings.cpp        # torch.ops registration entry
│   └── fa/                        # FA2 + mask CUDA kernel (sm120 optimized)
│       ├── fa_fwd_op.cu           # operator entry / dispatch
│       ├── fa_fwd_sm120.h         # sm120 TMA pipeline kernel / splitkv / persistent
│       ├── fa_fwd_kernel.h        # compute mainloop / softmax / epilogue
│       └── ...
├── thirdparty/            # CUTLASS / CuTe (header-only dependencies)
├── benchmark/             # performance benchmark
├── examples/              # usage examples
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
