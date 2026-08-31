# custom_ops — High-Performance CUDA Operators for Generative Recommendation

[中文](README.md) | English

A high-performance CUDA operator library for generative recommender systems,
with two core operators:

- **`mha_fwd_with_mask`**: a FlashAttention-2 forward implementation supporting
  **arbitrary additive masks** (0 = visible / -inf = masked), deeply optimized for
  consumer GPUs (sm120 / sm89, bf16 / fp16) and Volta data-center GPUs (sm70 /
  V100, fp16), significantly faster than PyTorch SDPA with a mask
- **`mixed_gemm`**: a mixed-precision GEMM that solves the **bf16 weight-precision
  loss vs tf32 speed** dilemma of recommendation models — a bf16 tensor-core main
  term plus a low-precision tensor-core residual correction term restores
  near-fp32 accuracy at near-bf16 cost, with bias addition and silu/gelu
  activation fused into the epilogue

**Highlights**

**Attention operator `mha_fwd_with_mask`**

- **Arbitrary masks go straight into softmax**: causal, sliding-window, padding,
  random-sparse (item-level masking) — any pattern, no kernel changes
- **sm120 (RTX 5090D)**: **160–180 TFLOPS** on standard shapes (68–76% of the
  measured cuBLAS bf16 peak), **1.62–2.48x** over SDPA+mask and **1.5–3.5x**
  over official FlexAttention in the same mask setting; up to **60x** on
  small-grid long-sequence shapes (**104x** vs FlexAttention); ragged shapes
  (Sk%8≠0 auto-pad / arbitrary Sq) at **1.65–7.8x** over SDPA+mask
- **sm89 (RTX 4090D)**: works out of the box — **1.15–1.86x** over SDPA+mask
  and **1.3–4.5x** over FlexAttention on standard shapes, up to **31x** on
  small-grid long-sequence shapes (**51x** vs FlexAttention); most ragged
  shapes keep their speedup
- **sm70 (Tesla V100)**: a dedicated fp16 path (V100 has no bf16 tensor cores)
  with a fully hand-built data path — WMMA m16n16k16 plus a conflict-free
  swizzle (sm70 has no ldmatrix / cp.async / TMA); standard shapes reach
  **16–24 TFLOPS at d=128 / 11–17 TFLOPS at d=64** (13–29% of the measured
  cuBLAS fp16 peak), **1.07–1.82x** over SDPA+mask; on the mask-free
  equivalent it beats the open-source flash-attention-v100 reference by
  **1.23–1.31x** at d=128; no Split-KV yet (small-grid long-sequence shapes
  are a known gap)
- **Native GQA**: K/V head count only needs to divide Q head count,
  no manual expansion required
- **Adaptive Split-KV** (sm89/sm120): a cost model picks the split count
  automatically; large grids fall back to a single kernel with zero overhead —
  nothing to tune

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

> Full records of the FA porting and optimization journey:
> [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md)
> and
> [docs/fa_sm70_porting_and_optimization.md](docs/fa_sm70_porting_and_optimization.md)
> (in Chinese).

## Performance

### RTX 5090D (sm120, bf16)

Test setup: RTX 5090D / PyTorch 2.11 / CUDA 12.8, default benchmark settings
(see the [Benchmark](#benchmark) section), with a 10% random -inf mask.

Standard shapes: **160–180 TFLOPS** (68–76% of the measured cuBLAS bf16 peak),
**1.62–2.48x** over SDPA+mask and **1.5–3.5x** over official FlexAttention
(same mask setting):

| Shape | custom | SDPA+mask | Speedup | FlexAtt | vs Flex |
|---|---|---|---|---|---|
| d64 B=4 H=16 S=2048 | 387.1µs (177.5 TF) | 754.6µs | **1.95x** | 1355.6µs | **3.50x** |
| d64 B=32 H=16 S=1024 | 858.8µs (160.0 TF) | 1394.8µs | **1.62x** | 2339.3µs | **2.72x** |
| d64 B=1 H=8 S=8192 | 853.9µs (161.0 TF) | 1672.2µs | **1.96x** | 2996.2µs | **3.51x** |
| d128 B=4 H=16 S=2048 | 775.9µs (177.1 TF) | 1870.9µs | **2.41x** | 1710.8µs | **2.20x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 775.9µs (177.1 TF) | 1925.9µs | **2.48x** | 1711.1µs | **2.21x** |
| d128 B=32 H=16 S=1024 | 1527.1µs (180.0 TF) | 3572.7µs | **2.34x** | 3101.1µs | **2.03x** |

Small-grid + long-sequence shapes (Split-KV kicks in automatically):
**13–60x** over SDPA+mask and **22–104x** over FlexAttention:

| Shape | custom | SDPA+mask | Speedup | FlexAtt | vs Flex |
|---|---|---|---|---|---|
| d128 Sq=128 Sk=8192 | 22.5µs | 558.1µs | **24.8x** | 962.5µs | **42.7x** |
| d128 Sq=128 Sk=32768 | 36.9µs | 2202.6µs | **59.8x** | 3823.6µs | **103.7x** |
| d128 Sq=1024 Sk=8192 | 43.0µs (99.9 TF) | 558.1µs | **13.0x** | 960.5µs | **22.3x** |
| d64 Sq=128 Sk=8192 | 16.4µs | 436.3µs | **26.6x** | 1053.7µs | **64.3x** |

Ragged shapes (`--suite ragged`, Sk%8!=0 auto-padding / arbitrary Sq; the
"aligned equiv." column quantifies the auto-pad overhead; **1.17–19.7x** vs
FlexAttention):

| Shape (B,H,Hk,Sq,Sk,d) | custom | aligned equiv. (pad overhead) | SDPA+mask | Speedup | vs Flex |
|---|---|---|---|---|---|
| d64 Sq=2048 Sk=2053 | 100.4µs (85.8 TF) | 57.7µs (+74%) | 184.3µs | 1.83x | 2.57x |
| d128 Sq=1024 Sk=1031 | 98.3µs (88.0 TF) | 67.9µs (+45%) | 178.3µs | 1.81x | 1.17x |
| d128 B=4 Sq=2048 Sk=4099 | 2003.7µs (137.3 TF) | 1485.1µs (+35%) | 3983.4µs | 1.99x | 1.30x |
| d64 Sq=8192 Sk=8195 | 1214.5µs (113.2 TF) | 853.9µs (+42%) | 2006.0µs | 1.65x | 2.27x |
| d64 Sq=1000 Sk=1024 (any Sq) | 32.7µs (128.1 TF) | ≈zero | 80.9µs | 2.47x | 4.44x |
| d64 B=2 Sq=333 Sk=1024 | 30.7µs (90.9 TF) | ≈zero | 79.8µs | 2.60x | 4.59x |
| d128 Sq=127 Sk=2048 | 18.4µs | ≈zero | 143.4µs | 7.78x | 19.7x |
| d128 Sq=1000 Sk=4099 (both ragged) | 301.1µs (111.5 TF) | — | 620.4µs | 2.06x | 1.28x |
| d128 B=4 Hk=4 Sq=2048 Sk=2053 (GQA) | 896.7µs (153.6 TF) | 775.9µs (+16%) | 2094.1µs | 2.34x | 1.49x |

> Note: the absolute pad-copy cost for ragged Sk is on the same order as sm89,
but the sm120 kernel is faster, so the relative overhead is higher (+35–72%);
GQA drops to +16% (smaller K/V). For latency-critical paths with a fixed Sk,
pre-align to a multiple of 8 (zero-copy main path).

> Note: with an arbitrary `attn_mask`, PyTorch SDPA can only use the MemEfficient
> backend (the FlashAttention backend does not support arbitrary masks). On some
> shapes this operator **with a mask is even faster than SDPA's mask-free Flash
> backend**.
>
> FlexAtt = `torch.nn.attention.flex_attention` (PyTorch's official operator for
> custom masks, torch≥2.5): `create_block_mask` pre-building + `torch.compile`;
> block-mask construction and Triton compilation happen during warmup and are
> excluded from timing, mirroring this operator's "mask pre-built, straight into
> the kernel". With a 10% random -inf mask almost every block is partial, so
> FlexAttention cannot exploit block sparsity. It has no Split-KV, hence the
> largest gaps on small-grid long-sequence shapes (22–104x).

### RTX 4090D (sm89, bf16)

The sm89 path is a cp.async implementation with the same adaptive Split-KV
ported from sm120: small-grid long-sequence shapes split along the K dimension
to fill the SMs automatically, while large grids fall back to a single kernel —
zero configuration needed.

Test setup: RTX 4090D / PyTorch 2.11 / CUDA 12.8, default benchmark settings
(see the [Benchmark](#benchmark) section).

Standard shapes (large grid): **1.15–1.86x** over SDPA+mask and **1.30–4.50x**
over FlexAttention:

| Shape | custom | SDPA+mask | Speedup | FlexAtt | vs Flex |
|---|---|---|---|---|---|
| d64 B=1 H=16 S=1024 | 72.7µs (59.1 TF) | 84.0µs | **1.15x** | 94.2µs | **1.30x** |
| d64 B=4 H=16 S=2048 | 582.7µs (117.9 TF) | 784.3µs | **1.35x** | 2363.5µs | **4.06x** |
| d64 B=1 H=8 S=8192 | 1121.2µs (122.6 TF) | 1644.5µs | **1.47x** | 5047.3µs | **4.50x** |
| d64 B=32 H=16 S=1024 | 1266.7µs (108.5 TF) | 1469.4µs | **1.16x** | 4751.4µs | **3.75x** |
| d128 B=1 H=16 S=1024 | 100.4µs (85.6 TF) | 164.9µs | **1.64x** | 246.8µs | **2.46x** |
| d128 B=4 H=16 S=2048 | 1175.5µs (116.9 TF) | 1874.9µs | **1.60x** | 2734.1µs | **2.33x** |
| d128 B=1 H=8 S=8192 | 2133.0µs (128.9 TF) | 3701.8µs | **1.74x** | 5029.9µs | **2.36x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 1069.2µs (128.5 TF) | 1983.4µs | **1.86x** | 2446.3µs | **2.29x** |
| d128 B=32 H=16 S=1024 | 2215.9µs (124.0 TF) | 3768.3µs | **1.70x** | 5006.7µs | **2.26x** |

Small-grid + long-sequence shapes (Split-KV kicks in automatically):
**7.1–31.3x** over SDPA+mask and **9.6–50.8x** over FlexAttention:

| Shape | custom | SDPA+mask | Speedup | SDPA flash* | FlexAtt | vs Flex |
|---|---|---|---|---|---|---|
| d128 Sq=128 Sk=8192 | 23.5µs (22.9 TF) | 417.8µs | **17.81x** | 29.7µs | 621.8µs | **26.5x** |
| d128 Sq=128 Sk=32768 | 54.3µs (39.6 TF) | 1696.7µs | **31.26x** | 52.9µs | 2226.2µs | **41.0x** |
| d128 Sq=512 Sk=8192 | 36.9µs (58.3 TF) | 414.7µs | **11.25x** | 39.1µs | 622.7µs | **16.9x** |
| d128 Sq=1024 Sk=8192 | 58.4µs (73.6 TF) | 414.7µs | **7.11x** | 48.5µs | 562.2µs | **9.6x** |
| d128 H=2 Hk=1 Sq=512 Sk=16384 | 94.2µs (91.2 TF) | 818.1µs | **8.68x** | 123.9µs | 1233.9µs | **13.1x** |
| d64 Sq=128 Sk=8192 | 20.5µs (13.1 TF) | 444.4µs | **21.70x** | 17.2µs | 1039.6µs | **50.8x** |
| d64 Sq=1024 Sk=8192 | 41.0µs (52.4 TF) | 403.5µs | **9.85x** | 33.8µs | 1044.5µs | **25.5x** |
| d64 H=2 Hk=1 Sq=512 Sk=16384 | 55.3µs (77.7 TF) | 774.4µs | **14.00x** | 66.4µs | 2012.2µs | **36.4x** |

> \* The SDPA flash column is the mask-free FlashAttention backend, shown for
> reference only (it does not support arbitrary masks); the fair comparison is
> the SDPA+mask column. Thanks to Split-KV, some small-grid shapes now match or
> beat even this reference (e.g. d128 H=2 Hk=1 Sq=512 Sk=16384: 94.2µs vs 123.9µs).

For reference: without Split-KV, small-grid long-sequence shapes are processed
by a single CTA serially walking all KV blocks, taking 234–935µs — with
Split-KV enabled automatically this drops to 19–53µs (**10–18x**).

Ragged shapes (`Sk % 8 != 0` auto-padding / arbitrary `Sq`, `--suite ragged`).
The "aligned equiv." column shows the custom time of the same shape with Sk
rounded down to the multiple of 8, quantifying the auto-pad overhead;
**1.30–12.8x** vs FlexAttention:

| Shape (B,H,Hk,Sq,Sk,d) | custom | aligned equiv. (pad overhead) | SDPA+mask | Speedup | vs Flex |
|---|---|---|---|---|---|
| d64 Sq=2048 Sk=2053 | 179.2µs (48.1 TF) | 120.8µs (+48%) | 178.2µs | 0.99x | 2.31x |
| d128 Sq=1024 Sk=1031 | 138.2µs (62.6 TF) | 100.4µs (+38%) | 187.4µs | 1.36x | 1.39x |
| d128 B=4 Sq=2048 Sk=4099 | 2845.7µs (96.7 TF) | 2107.2µs (+35%) | 4064.3µs | 1.43x | 1.30x |
| d64 Sq=8192 Sk=8195 | 1702.9µs (80.7 TF) | 1121.2µs (+52%) | 2063.4µs | 1.21x | 2.39x |
| d64 Sq=1000 Sk=1024 (any Sq) | 67.6µs (62.1 TF) | ≈zero | 77.8µs | 1.15x | 3.92x |
| d64 B=2 Sq=333 Sk=1024 | 37.9µs (73.7 TF) | ≈zero | 66.6µs | 1.76x | 3.89x |
| d128 Sq=127 Sk=2048 | 31.9µs | ≈zero | 111.6µs | 3.50x | 12.8x |
| d128 Sq=129 Sk=2048 | 32.8µs | ≈zero | 104.4µs | 3.19x | 4.53x |
| d128 Sq=1000 Sk=4099 (both ragged) | 477.2µs (70.4 TF) | — | 655.4µs | 1.37x | 1.39x |
| d128 B=4 Hk=4 Sq=2048 Sk=2053 (GQA) | 1235.0µs (111.6 TF) | 1069.2µs (+16%) | 2154.7µs | 1.74x | 1.52x |

> Note: when `Sk % 8 != 0`, the op internally pads K/V/mask with one copy pass
> (the `Sq×Sk` mask dominates), and the padded `Sk8` may cross one extra KV-tile
> boundary; GQA lowers the overhead (smaller K/V). Arbitrary `Sq` is copy-free
> (row predicates) and scales with the actual Sq. For latency-critical paths with
> a fixed Sk, pre-align to a multiple of 8 (zero-copy main path).
>
> FlexAtt note: FlexAttention's default config (BLOCK_M=128) **fails to compile**
> on sm89 at d=128 (smem requirement ~112KB exceeds the ~99KB Ada hardware
> limit; Inductor reports "No valid triton configs"). The tables above use its
> downgraded `BLOCK_M=64` — the only official config that runs on this
> architecture. sm120 (Blackwell, 228KB smem) has no such issue and uses the
> default config throughout.

### Tesla V100 (sm70, fp16)

The sm70 path is **fp16-only** (V100 has no bf16 tensor cores): passing fp16
tensors to `mha_fwd_with_mask` enables it automatically. The data path is
built entirely by hand — WMMA m16n16k16 plus a conflict-free swizzle (sm70 has
no ldmatrix / cp.async / TMA); d=128 opts into the full 96KB of smem. No
Split-KV yet — small-grid long-sequence shapes are a known gap (porting from
sm120 is planned).

Test setup: V100-PCIE-32GB / PyTorch 2.0.1 / CUDA 11.7, with a 10% random
-inf mask.

Standard shapes (large grid): **1.07–1.82x** over SDPA+mask, peaking at
**24.2 TFLOPS** at d=128 (29% of the measured cuBLAS fp16 peak of 84.2 TF):

| Shape | custom | SDPA+mask | Speedup |
|---|---|---|---|
| d64 B=4 H=16 S=1024 | 1001.8µs (17.1 TF) | 1711.6µs | **1.71x** |
| d64 B=4 H=16 S=2048 | 3982.4µs (17.3 TF) | 6765.7µs | **1.70x** |
| d64 B=4 H=16 Hk=4 S=2048 (GQA) | 3986.7µs (17.2 TF) | 7268.9µs | **1.82x** |
| d128 B=1 H=8 S=512 | 65.2µs (16.5 TF) | 92.4µs | **1.42x** |
| d128 B=4 H=16 S=1024 | 1450.8µs (23.7 TF) | 1864.9µs | **1.29x** |
| d128 B=4 H=16 S=2048 | 5694.7µs (24.1 TF) | 7313.5µs | **1.28x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 5688.6µs (24.2 TF) | 7761.5µs | **1.36x** |
| d128 B=1 H=8 S=2048 | 878.2µs (19.6 TF) | 938.2µs | **1.07x** |

> Note: on V100 + torch 2.0.1, SDPA with an `attn_mask` can only use the math
> backend. On the mask-free equivalent (zero additive mask) this operator beats
> the open-source flash-attention-v100 reference by **1.23–1.31x** across all
> d=128 shapes (19.1–24.2 TF vs 15.5–19.5 TF — and the reference does not
> support arbitrary masks); d=64 is on par (0.95–1.07x).

Ragged shapes (`--suite ragged`, Sk%8!=0 auto-padding / arbitrary Sq):

| Shape (B,H,Hk,Sq,Sk,d) | custom | aligned equiv. (pad overhead) | SDPA+mask | Speedup |
|---|---|---|---|---|
| d64 Sq=2048 Sk=2053 | 735.4µs (11.7 TF) | 614.3µs (+20%) | 1175.6µs | 1.60x |
| d128 Sq=1024 Sk=1031 | 558.6µs (15.5 TF) | 446.4µs (+25%) | 764.1µs | 1.37x |
| d64 B=2 Sq=512 Sk=4099 | 1511.4µs (11.4 TF) | 1211.1µs (+25%) | 2253.6µs | 1.49x |
| d64 Sq=1000 Sk=1024 (any Sq) | 312.6µs (13.4 TF) | ≈zero | 422.5µs | 1.35x |
| d128 B=2 Hk=4 Sq=500 Sk=2053 (GQA, both ragged) | 1026.5µs (16.4 TF) | — | 1747.6µs | 1.70x |

> Note: auto-pad overhead on V100 is +20–25% (the kernel itself is slower, so
> the relative share is lower than on sm89/sm120). Very small-Sq shapes (e.g.
> Sq=127 d128 → only 16 CTAs) lose to SDPA's math backend without Split-KV —
> the known gap above. FlexAttention requires torch≥2.5 and is unavailable on
> this environment (torch 2.0.1, N/A).

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
  sm120a TMA path extensively tested), sm89
  (RTX 4090D, cp.async path with Split-KV ported from sm120, tested), or sm70
  (Tesla V100, FA fp16 WMMA path, tested; mixed_gemm is not supported on sm70);
  other sm80+ architectures should compile but are unverified
- CUDA >= 11.8 (FA sm89 / sm70 paths / mixed_gemm INT8 backend) / >= 12.4
  (mixed_gemm FP8 backend, SM89+) / >= 12.8 (FA / mixed_gemm sm120a paths),
  GCC >= 9
- PyTorch >= 2.1 (CUDA build), fp16 / bf16

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

> Dtype note: fp16 works on all supported architectures (sm70 / sm89 / sm120);
> bf16 requires SM80+ (V100 has no bf16 tensor cores — use fp16 there). On
> sm89/sm120 either dtype is fine; pick per your model's precision strategy.

Build artifacts are cached in `~/.cache/torch_extensions` by default; override with
`TORCH_EXTENSIONS_DIR`. Concurrent first-time builds from multiple processes are
protected by a file lock.

## API

### `ops.mha_fwd_with_mask(q, k, v, mask) -> Tensor`

FlashAttention-2 forward with an arbitrary additive mask. fp16 is supported on
all architectures (sm70 / sm89 / sm120); bf16 requires SM80+. On V100, fp16
inputs are auto-routed to the dedicated sm70 path.

| Argument | Shape | Description |
|---|---|---|
| `q` | (B, H, Sq, d) | fp16 (all archs) or bf16 (SM80+) CUDA contiguous tensor |
| `k`, `v` | (B, Hk, Sk, d) | CUDA contiguous tensors, same dtype as `q` |
| `mask` | (B, 1, Sq, Sk) | additive mask, same dtype as `q`: 0 = visible / -inf = masked |
| Returns | (B, H, Sq, d) | same dtype as the input |

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

# V100 (sm70, fp16): standard + small-grid + ragged shapes (vs SDPA+mask / SDPA reference)
python benchmark/benchmark_sm70.py

# A specific suite / a single shape
python benchmark/benchmark_fa.py --suite standard
python benchmark/benchmark_fa.py --suite splitkv
python benchmark/benchmark_fa.py --suite ragged   # Sk%8!=0 auto-pad / arbitrary Sq
python benchmark/benchmark_fa.py --shape 4 16 16 2048 2048 128

# Tune iterations and output; --no-flex skips the FlexAttention comparison
# (saves per-shape Triton compilation)
python benchmark/benchmark_fa.py --warmup 20 --iters 100 --mask-ratio 0.3 --csv result.csv
python benchmark/benchmark_fa.py --no-flex
```

For each shape the script reports latency, TFLOPS and speedup for
custom / SDPA+mask / FlexAttention / SDPA flash (reference), and runs a
numerical check against SDPA. For the FlexAttention comparison (torch≥2.5),
block-mask construction and Triton compilation happen during warmup and are
excluded from timing; on torch<2.5 or when the default config fails to compile
(sm89 d=128) it auto-degrades (BLOCK_M=64) or reports N/A.

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
│   ├── arch_targets.h             # arch conditional-compilation entry: FA_TARGETS → FA_HAS_SM70/SM8X/SM120 + gpu_major()
│   ├── fa/                        # FA2 + mask kernel (sm70 / sm89 / sm120 paths, decoupled)
│   │   ├── fa_fwd_op.cu           # operator entry: validate / auto-pad / fill params (arch-agnostic)
│   │   ├── fa_fwd_launch.h        # arch dispatch entry: fa_launch_smXX policies + Split-KV cost model
│   │   ├── sm70/fa_fwd_sm70.h     # sm70 WMMA m16n16k16 kernel (fp16, no split-KV)
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
