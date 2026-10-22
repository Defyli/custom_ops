# custom_ops — High-Performance CUDA Operators for Generative Recommendation

[中文](README.md) | English

A high-performance CUDA operator library for generative recommender systems,
with three core operators:

- **`mha_fwd_with_mask`**: a FlashAttention-2 forward implementation supporting
  **arbitrary additive masks** (0 = visible / -inf = masked), deeply optimized for
  consumer GPUs (sm120 / sm89, bf16 / fp16) and Volta data-center GPUs (sm70 /
  V100, fp16), significantly faster than PyTorch SDPA with a mask
- **`mixed_gemm`**: a mixed-precision GEMM that solves the **bf16 weight-precision
  loss vs tf32 speed** dilemma of recommendation models — a bf16 tensor-core main
  term plus a low-precision tensor-core residual correction term restores
  near-fp32 accuracy at near-bf16 cost, with bias addition and silu/gelu
  activation fused into the epilogue
- **`fuse_moe`**: fused MoE FFN forward (single GPU) — the default cp.async path
  combines counting/routing, projection + activation, down group GEMM and topk
  reduction into 4 kernels. Explicitly supports three-matrix **GLU**
  (SwiGLU/GEGLU) and two-matrix **ordinary MLP** (SiLU/GELU), with bf16/fp16
  inputs and fp32 accumulation; offload streams CPU expert weights within a GPU budget

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

**Fused MoE FFN `fuse_moe`**

- **171–190 effective TFLOPS on large shapes** (82–91% of the mma.sync peak):
  **188 TF** at 4096×4096×1408 and 16384×2048×1024; **1.9–14.8x** over PyTorch
  eager, **1.4–2.5x** over `torch.compile`, and up to **98x** over
  compile + CUDA graph (small-batch large-E shapes)
- **Fused gemm1 (gate/up N-tile pairing)**: one CTA computes both the gate and
  up N panels of W1 (sharing the X tile, halving X loads); the epilogue applies
  `silu(gate)·up` and writes act_out directly — no `gate_up_out (T, 2I)`
  materialization and no separate activation kernel
- **Single-projection ordinary MLP**: `expert_type="mlp"` computes `act(W1·x)`
  without an up projection or gating product; shares routing/down/reduction
  with GLU and retains the four-kernel default path
- **Cross-tile continuous pipeline**: the slab issue stream uses global
  counters and never drains at task (tile) boundaries; scatter row indices live
  in registers with rolling prefetch; activations stay in the expert-sorted
  compact layout (no padding holes, no pre-materialization)
- **Architecture-adaptive**: sm89 / sm120 share one cp.async engine (an optional
  TMA + mbarrier engine on sm120 via `FUSE_MOE_TMA=1`); kTileM picked from
  32/64/128 by avg tokens/expert, kStage adapted to the smem budget; the four
  kernels are chained via PDL

**Common infrastructure**

- **JIT build**: the first import compiles automatically (multi-process safe),
  `torch.compile`/AOTI compatible

> Full records of the FA porting and optimization journey:
> [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md)
> and
> [docs/fa_sm70_porting_and_optimization.md](docs/fa_sm70_porting_and_optimization.md);
> fuse_moe in
> [docs/fuse_moe_porting_and_optimization.md](docs/fuse_moe_porting_and_optimization.md)
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
> on sm89 at d=128 (smem requirement ≈112KB exceeds the ≈99KB Ada hardware
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
- Building with CUDA 11.8 makes the same INT8 kernel ≈17% slower on large shapes
  (nvcc codegen differences; 4096³ measures 1658µs) — small shapes are
  unaffected; prefer a recent CUDA toolkit

> \* The bf16 column is `bf16(x) @ bf16(W)^T` (weights pre-cast offline, on par
> with mixed_gemm's offline weight split) + fp32 epilogue: 1.3–1.7x faster (a
> single GEMM vs the main-plus-correction dual GEMM) but with fully uncompensated
> weight-rounding error.

### fuse_moe (fused MoE FFN, single GPU)

The original tables below are historical **GLU / SiLU / fused** measurements.
The new ordinary-MLP and offload tables use a separate environment and numerical
contract; two- and three-matrix GEMM FLOPS are never mixed.

Test setup: RTX 5090D / PyTorch 2.6 / CUDA 12.8, bf16, uniform routing.
The sm120 default engine is the continuous-pipeline cp.async kernel
(outperforms the TMA engine by 3–15% across shapes); `FUSE_MOE_TMA=1`
optionally selects the TMA + mbarrier engine (2807µs vs 2197µs default,
S=16384). sm89 (RTX 4090D) uses the same cp.async path.

Large shapes reach **171–190 effective TFLOPS** (82–91% of the mma.sync
peak): **1.9–14.8x** vs eager, **1.4–2.5x** vs `torch.compile`, and up to
**98x** vs compile + reduce-overhead (CUDA graph):

| Shape (S,H,I,E,K) | custom µs | TFLOPS | eager µs | vs eager | comp µs | comp+RO µs | FG+graph µs | vs FG |
|---|---|---|---|---|---|---|---|---|
| (128,2048,1024,8,2) | 83.6 | 38.5 | 1235 | **14.8x** | 1728 | 1636 | 8198 | **98.0x** |
| (128,2048,1024,64,2) | 634.4 | 5.1 | 8030 | **12.7x** | 12819 | 12280 | 9565 | 15.1x |
| (1024,2048,1024,8,2) | 175.3 | 147.0 | 1125 | 6.4x | 1800 | 1676 | — | — |
| (1024,4096,1024,8,2) | 339.7 | 151.7 | 1183 | 3.5x | 1813 | 1695 | — | — |
| (4096,2048,1024,8,2) | 577.0 | 178.7 | 1415 | 2.5x | 1866 | 1947 | — | — |
| (4096,2048,1024,64,8) | 2413.7 | 170.8 | 9301 | **3.9x** | 14236 | 13726 | — | — |
| (4096,4096,1408,8,2) | 1507.2 | 188.1 | 3191 | 2.1x | 3592 | 3300 | — | — |
| (16384,2048,1024,8,2) | 2196.6 | 187.7 | 4180 | **1.9x** | 4471 | 4638 | — | — |

Design highlights (implementation under `csrc/fuse_moe/`; full optimization
log in [docs/fuse_moe_porting_and_optimization.md](docs/fuse_moe_porting_and_optimization.md)):

- **Cross-tile continuous pipeline**: the slab issue stream uses global
  counters (`stage = cnt % kStage`) and never drains at task (tile)
  boundaries; scatter row indices live in registers with rolling prefetch;
  kStage adapts to the 96KB smem budget
- **Fused gemm1 (gate/up N-tile pairing)**: one CTA computes both the gate
  and up N panels of W1 (sharing the X tile, halving X loads); the epilogue
  applies `silu(gate)·up` and writes act_out directly — no `gate_up_out
  (T, 2I)` materialization and no separate activation kernel
- **Compact reads**: activations stay in the expert-sorted compact layout
  (no padding holes); the cp.async scatter path gathers token-ordered x
  on load
- **Adaptive tiling**: kTileM = 32/64/128 by avg tokens/expert (128 doubles
  the W reuse span, enabled at avg≥256); kTileK = 64/128 by k%128
  (gated by smem)
- **PDL chaining**: count → gemm1 → gemm2 → reduce chained via
  Programmatic Dependent Launch (sm89 falls back to stream order)

### Ordinary MLP and GLU MoE (GPU-resident, new benchmark)

Measured on 2026-10-21: **PyTorch 2.11.0+cu128 / CUDA 12.8**, bf16,
exact-erf GELU, `numerical_mode="staged"`, default cp.async engine.
The 4090D host uses Xeon Platinum 8576C / PCIe 4.0 x16; the 5090D **v2** host
uses EPYC 9K65 / PCIe 5.0 x16. Offload differences across these machines cannot
be attributed to the GPU alone. Fixed `H=2048, I=1024, E=64, K=2`; only S varies.
Seeded uniform routing selects two distinct experts per token (seed=42, +1
for each subsequent shape).

Both topologies use 2 group-GEMM launches, but **GLU gemm1 computes paired
gate/up projections; ordinary MLP computes only one**. Counting multiply and add
separately, GEMM FLOPS are `6*S*K*H*I` for GLU and `4*S*K*H*I` for MLP.
All bf16 expert weights occupy **768 MiB / 512 MiB**, respectively.

| GPU | S | Expert type | custom µs | eager µs | vs eager | custom TFLOPS |
|---|---:|---|---:|---:|---:|---:|
| 4090D | 128 | GLU (GEGLU) | 883.8 | 10719.6 | 12.13x | 3.6 |
| 4090D | 128 | Ordinary MLP | 601.7 | 9512.8 | 15.81x | 3.6 |
| 4090D | 1024 | GLU (GEGLU) | 1008.2 | 11331.6 | 11.24x | 25.6 |
| 4090D | 1024 | Ordinary MLP | 676.5 | 10261.5 | 15.17x | 25.4 |
| 4090D | 4096 | GLU (GEGLU) | 1227.0 | 11558.9 | 9.42x | 84.0 |
| 4090D | 4096 | Ordinary MLP | 855.5 | 10498.0 | 12.27x | 80.3 |
| 5090D v2 | 128 | GLU (GEGLU) | 635.8 | 6934.6 | 10.91x | 5.1 |
| 5090D v2 | 128 | Ordinary MLP | 430.8 | 6833.9 | 15.86x | 5.0 |
| 5090D v2 | 1024 | GLU (GEGLU) | 692.2 | 8215.3 | 11.87x | 37.2 |
| 5090D v2 | 1024 | Ordinary MLP | 470.4 | 7431.2 | 15.80x | 36.5 |
| 5090D v2 | 4096 | GLU (GEGLU) | 768.5 | 8305.4 | 10.81x | 134.1 |
| 5090D v2 | 4096 | Ordinary MLP | 539.1 | 7553.1 | 14.01x | 127.5 |

Latency is the median of 5 **CUDA-event batch means**, with 20 warmup and 50
timed calls per repeat; custom/eager order rotates. Eager is a real `nn.Module`
using per-expert `nonzero` routing, low-precision `Linear`, standard dtype
GELU/multiply and FP32 `index_add`. It includes the full E scan and data-dependent
synchronization, not redundant activation casts, and is not an optimized
grouped-GEMM or compile baseline. TF32, reduced-precision reduction and gradients
are disabled. Generation, CPU reference, loading, compilation and checking are
excluded. Large-shape benchmark correctness checks the first 16 tokens against
an independent CPU FP32 stage reference and checks full-output shape/dtype/finite;
full GPU matrices, exact scalar rounding witnesses and masked/duplicate routes
are verified separately by the test suite.

### MoE offload (CPU weights, explicit GPU budget)

Same inputs and GELU/staged contract, measured through `PrefillMoERuntime.forward`,
not AOTI. MoE context budgets: **GPU 128 MiB / RAM 2048 MiB / pinned 64 MiB**;
`activation_mode="auto"`, automatic route tiles, no dynamic cache.
GPU-resident baseline weights are released before offload measurement. GPU charge
includes context workspace/cache and live output, excluding x, unrelated model
tensors, allocator reserves and the driver. **512/768 MiB of expert weights both
exceed the 128 MiB context budget**: this does not secretly keep all weights on
GPU, nor is it a whole-device OOM test.

All numbers below use **host wall-clock: forward + CUDA synchronize**, including
host packing, routing, H2D, compute and the final drain; stats/checking are outside
the clock. Cold is the median first forward of 5 fresh runtimes; initialization is
recorded separately and excluded from cold. Warm is the median of 5 means, each
over 50 calls after 20 warmups in the same runtime. P95 pools all 250 timed calls.
Sync/async order rotates, without concurrent work on the same GPU. **Do not
compute speedups by mixing GPU-event µs above with host-wall ms below**;
a GPU-resident host-wall reference is included for the same routing scenario.

**No cache / uniform routing** (active expert weights transferred again on every call):

| GPU | S | Type | H2D MiB/call | resident wall ms | cold async ms | warm sync ms | warm async ms | async P95 ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 4090D | 128 | GLU | 744 | 0.919 | 63.824 | 58.434 | 62.031 | 63.175 |
| 4090D | 128 | MLP | 496 | 0.632 | 39.101 | 38.371 | 38.350 | 38.821 |
| 4090D | 1024 | GLU | 768 | 1.038 | 60.901 | 59.105 | 60.834 | 75.004 |
| 4090D | 1024 | MLP | 512 | 0.712 | 40.157 | 40.102 | 39.647 | 41.186 |
| 4090D | 4096 | GLU | 768 | 1.256 | 61.022 | 60.684 | 59.972 | 61.641 |
| 4090D | 4096 | MLP | 512 | 0.890 | 40.687 | 39.871 | 40.298 | 42.512 |
| 5090D v2 | 128 | GLU | 744 | 0.653 | 41.443 | 39.344 | 40.057 | 40.447 |
| 5090D v2 | 128 | MLP | 496 | 0.448 | 27.915 | 26.105 | 25.846 | 26.593 |
| 5090D v2 | 1024 | GLU | 768 | 0.712 | 42.477 | 40.055 | 40.527 | 41.441 |
| 5090D v2 | 1024 | MLP | 512 | 0.491 | 28.774 | 26.996 | 27.036 | 27.948 |
| 5090D v2 | 4096 | GLU | 768 | 0.794 | 42.166 | 40.816 | 40.658 | 41.396 |
| 5090D v2 | 4096 | MLP | 512 | 0.568 | 28.444 | 27.135 | 26.240 | 27.924 |

- S=128 activates 62 experts, so H2D is below all 64 experts; S=1024/4096 activate all 64.
- Actual plans use 2 banks: GLU wave capacity 2, MLP capacity 3; wave counts
  are 31/32/32 and 21/22/22. Context GPU highwater is **49.53–96.93 MiB**;
  initialization is **1.17–1.47 ms** (both sync/async cold/init are retained in JSON).
- Host packing and weight transfer dominate; MLP transfers 2/3 of the GLU bytes.
  **Async staging is not a speedup guarantee**: some shapes are slower or have
  higher P95, and those results are not hidden by selecting the faster mode.

**Static hotset hits**: only experts 0/1 are routed and preloaded; the other 62
experts stay on CPU. Static cache is **24 MiB GLU / 16 MiB MLP**; weight H2D per
forward is **0 B**, actual transfer banks=0 and resident waves=1. This cache-hit
scenario **does not represent uniform routing**.

| GPU | S | Type | resident wall ms | cold async ms | warm sync ms | warm async ms | async P95 ms |
|---|---:|---|---:|---:|---:|---:|---:|
| 4090D | 128 | GLU | 0.134 | 0.424 | 0.153 | 0.154 | 0.164 |
| 4090D | 128 | MLP | 0.106 | 0.423 | 0.126 | 0.126 | 0.139 |
| 4090D | 1024 | GLU | 0.427 | 0.724 | 0.448 | 0.450 | 0.462 |
| 4090D | 1024 | MLP | 0.315 | 0.618 | 0.337 | 0.337 | 0.347 |
| 4090D | 4096 | GLU | 0.884 | 1.297 | 0.907 | 0.907 | 0.914 |
| 4090D | 4096 | MLP | 0.634 | 1.058 | 0.649 | 0.650 | 0.661 |
| 5090D v2 | 128 | GLU | 0.069 | 0.436 | 0.081 | 0.081 | 0.093 |
| 5090D v2 | 128 | MLP | 0.057 | 0.493 | 0.072 | 0.072 | 0.082 |
| 5090D v2 | 1024 | GLU | 0.320 | 0.778 | 0.336 | 0.336 | 0.340 |
| 5090D v2 | 1024 | MLP | 0.234 | 0.646 | 0.250 | 0.251 | 0.256 |
| 5090D v2 | 4096 | GLU | 0.606 | 1.257 | 0.629 | 0.628 | 0.635 |
| 5090D v2 | 4096 | MLP | 0.431 | 1.008 | 0.452 | 0.452 | 0.461 |

Context GPU highwater is **18.01–88.07 MiB**; initialization, including static
weight upload, is **2.31–3.51 ms**, excluded from warm latency.
Local evidence is retained in `artifacts/moe_structures_sm89_20261021/` and
`artifacts/moe_structures_sm120_20261021/`, files
`storage_{uniform,hotset}_bf16_gelu_staged_review1.json`. They include library/script
SHA256, 5-repeat raw timings, cold/init measurements, reference checks and per-call
H2D/budget stats (artifacts are not committed with source). Reproduction commands
are in [Benchmark](#benchmark).

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

# ── Default SwiGLU MoE: down(silu(gate(x)) * up(x)), then topk weighting
S, H, I, E, K = 4096, 2048, 1024, 8, 2
x  = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.05
topk_ids   = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32)
topk_scale = torch.rand(S, K, device="cuda")            # fp32 routing weights

y = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)       # (S, H) bf16

# Ordinary two-matrix GELU MLP MoE: down(gelu(w1(x))), no up projection
w1_mlp = torch.randn(E, I, H, device="cuda", dtype=x.dtype) * 0.05
y_mlp = ops.fuse_moe(x, w1_mlp, w2, topk_ids, topk_scale,
                     expert_type="mlp", expert_activation="gelu",
                     numerical_mode="staged")
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

### `ops.fuse_moe(x, gate_up_weight, down_weight, topk_ids, topk_scale, backend=None, *, expert_activation="silu", numerical_mode="fused", expert_type="glu") -> Tensor`

`expert_type` explicitly selects the topology, never inferred from activation or weight shapes:

- `"glu"` (default): `y[s] = Σ_j scale[s,j] · Down_e(act(Gate_e·x[s]) · (Up_e·x[s]))`,
  a three-matrix gated FFN; gemm1 computes paired gate/up projections.
- `"mlp"`: `y[s] = Σ_j scale[s,j] · Down_e(act(W1_e·x[s]))`,
  an ordinary two-matrix FFN; gemm1 computes one projection, without gating multiplication.

`expert_activation="silu"|"gelu"` applies to either topology: SwiGLU/GEGLU for GLU,
SiLU/GELU MLP otherwise. GELU uses exact erf (`F.gelu(approximate="none")`), not
an implicit tanh approximation. Both topologies use 4 kernels and 2 group-GEMM
launches on the default cp.async path. The optional TMA path includes gathering
and a separate activation, totaling 6 kernels.

| Param | Shape | Notes |
|---|---|---|
| `x` | (S, H) | bf16 / fp16, token order |
| `gate_up_weight` | GLU (E, 2I, H) / MLP (E, I, H) | same dtype as x; GLU gate first/up second; MLP single projection |
| `down_weight` | (E, H, I) | same dtype as x |
| `topk_ids` | (S, K) | contiguous int32, nonnegative values in [0, E); -1 slots are ignored |
| `topk_scale` | (S, K) | contiguous fp32 topk weights; do not downcast before calling |
| Returns | (S, H) | same dtype as x |

Constraints: contiguous CUDA inputs, positive S/H/I, matching x/weight dtype,
`H % 64 == 0`, `I % 64 == 0`, `1 <= K <= 128`, `1 <= E <= 512`, SM80+.

- `numerical_mode="fused"` fuses activation (and the GLU product) after FP32
  projection before rounding. `"staged"` rounds projection, activation, GLU
  product, down and weighted product to the output dtype before FP32 reduction.
  MLP has no up/product stage. This is a **forward rounding contract, not
  autograd or bitwise parity**.
- All three selectors accept exact strings only, not integers, booleans or case variants.
- GLU concatenates gate/up `(E,I,H)` along dim=1, gate first, into `(E,2I,H)`.
  MLP uses one `(E,I,H)` W1 with `expert_type="mlp"`. Both use `(E,H,I)` down;
  callers explicitly adapt flattened/transposed source layouts.
- Offload freezes `expert_type`, `expert_activation` and `numerical_mode` in
  `OffloadConfig`; `activation_mode` only selects buffering/reduction. GLU
  silu/fused emits v1 (23 fixed fields), other GLU pairs emit v2 (25 fixed fields,
  mode codes at 23/24); MLP always emits v3 (26 fixed fields, type code 1 at 25,
  resident IDs from 26). Manifest framing=1 and weight archive format=2 remain
  unchanged. Rebuild `.so/.so.json/.pt2/weights.pt` together after the native
  schema change; do not mix old binaries with new callers.
- TileLang supports GLU/MLP and guards masked routes, failing explicitly when
  exact erf is unavailable. Legacy GLU/fused retains TileLang FP32 down/weighting;
  new MLP modes both preserve the native down rounding boundary. Staged has
  the same stage contract across backends; comparisons still use tolerances.

[GEGLU development and acceptance notes](docs/MOE_GEGLU_EXTENSION.md) preserve
the earlier GLU-extension record. Its historical “ordinary MLP is out of contract”
scope statement is superseded by `expert_type="mlp"` above.

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
| `FUSE_MOE_TMA=1` | Select the TMA + mbarrier engine on sm120 (default cp.async) |
| `FUSE_MOE_TILE_M=32/64/128` | Force the fuse_moe kTileM (default adapts to avg tokens) |
| `FUSE_MOE_TIME=1` | Print per-kernel timings of the fuse_moe pipeline |

## Examples

```bash
python examples/basic_usage.py        # minimal example: invocation + SDPA verification
python examples/custom_mask_demo.py   # 4 typical masks: causal / sliding-window / padding / random-sparse
python examples/gqa_example.py        # GQA: no K/V head expansion needed
python examples/mixed_gemm_demo.py    # mixed-precision GEMM: weight split + epilogue fusion + accuracy
python examples/fuse_moe_demo.py      # GLU MoE: forward + numeric check + vs eager
python examples/fuse_moe_demo.py --expert-type mlp --expert-activation gelu --numerical-mode staged
python examples/fuse_moe_offload_demo.py --jit --expert-type mlp   # development; use --ops-library for deployment
python examples/swiglu_demo.py        # SwiGLU: FFN activation + vs eager
python tests/test_fuse_moe.py         # GLU/MLP × two activations × two rounding modes, masked/duplicate routes
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

# fuse_moe: historical GLU suite, eager / torch.compile / compile+reduce-overhead
python benchmark/benchmark_fuse_moe.py
```

**Unified ordinary-MLP / GLU and offload benchmark** (separate from the legacy suite):
first build the current production schema offline. Use `8.9` for SM89 or `12.0a` for SM120:

```bash
python examples/fuse_moe_offload_aoti.py build-native --arch 8.9 --output-dir artifacts/moe_native
export MOE_SO="$PWD/artifacts/moe_native/recsys_fuse_moe_kernel.so"
mkdir -p artifacts/moe_benchmark
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# GPU-resident + uncached offload: CPU weights, active experts transferred every call
python benchmark/benchmark_fuse_moe.py --suite storage --ops-library "$MOE_SO" \
  --expert-types glu mlp --expert-activation gelu --numerical-mode staged --dtype bf16 \
  --shapes 128,2048,1024,64,2 1024,2048,1024,64,2 4096,2048,1024,64,2 \
  --offload --gpu-budget-mib 128 --ram-budget-mib 2048 --pinned-budget-mib 64 \
  --offload-cache disabled --routing uniform --offload-staging sync async \
  --seed 42 --repeats 5 --warmup 20 --iters 50 --json artifacts/moe_benchmark/uniform.json

# Static hotset: only experts 0/1 resident; the other 62 experts stay on CPU
python benchmark/benchmark_fuse_moe.py --suite storage --ops-library "$MOE_SO" \
  --expert-types glu mlp --expert-activation gelu --numerical-mode staged --dtype bf16 \
  --shapes 128,2048,1024,64,2 1024,2048,1024,64,2 4096,2048,1024,64,2 \
  --offload --gpu-budget-mib 128 --ram-budget-mib 2048 --pinned-budget-mib 64 \
  --offload-cache static_hot --routing hotset --offload-staging sync async \
  --seed 42 --repeats 5 --warmup 20 --iters 50 --json artifacts/moe_benchmark/hotset.json
```

Omit `--offload` and the budget/cache/staging options to measure GPU-resident
only. `--expert-types mlp` measures ordinary MLP alone; choose SiLU or fused
explicitly. The standard eager baseline is always stage-rounded, so the fair
comparison on this page uses staged. JSON output is created exclusively: use a
new path for reruns. It retains library/script hashes, raw timings, reference
checks and per-call offload stats. Clear `FUSE_MOE_*` diagnostic/tuning variables
before measuring; diagnostic runs are not performance evidence.

## Repository Structure

```
custom_ops/
├── __init__.py            # CustomOps: generic JIT operator loading framework (base class)
├── recsys.py              # RecsysOps: lazy group-loading facade (fa/mixed_gemm/fuse_moe) + weight-split helper
├── csrc/
│   ├── custom_ops_macros.h       # shared operator registration macros
│   ├── arch_targets.h             # arch conditional-compilation entry: FA_TARGETS → FA_HAS_SM70/SM8X/SM120 + gpu_major()
│   ├── fa/                        # FA2 + mask kernel (sm70 / sm89 / sm120 paths, decoupled)
│   │   ├── fa_bindings.cpp       # FA group registration entry (own .so, lazy-loaded)
│   │   ├── fa_fwd_op.cu           # operator entry: validate / auto-pad / fill params (arch-agnostic)
│   │   ├── fa_fwd_launch.h        # arch dispatch entry: fa_launch_smXX policies + Split-KV cost model
│   │   ├── sm70/fa_fwd_sm70.h     # sm70 WMMA m16n16k16 kernel (fp16, no split-KV)
│   │   ├── sm89/fa_fwd_kernel.h   # sm89 cp.async kernel: base / splitkv / combine
│   │   ├── sm120/fa_fwd_sm120.h   # sm120 TMA pipeline kernel / splitkv / persistent / combine
│   │   ├── cpu/                   # CPU reference implementation (not wired into dispatch)
│   │   └── common/                # shared params / softmax / utils
│   ├── mixed_gemm/                # mixed-precision GEMM kernel (ported from a TRT plugin)
│   │   ├── mixed_gemm_op.cu       # torch operator entry: validation / workspace / dispatch
│   │   ├── gemm_bf16xfp32_sm80.cu # kernel: tile/split-K heuristics + bf16 main + fp8/int8 residual dual GEMM
│   │   ├── gemm_bf16xfp32_sm80.h  # kernel entry declarations + FP8 compile-time guard
│   │   ├── gemm_bf16xfp32_sm120.cu # sm120a TMA+mbarrier kernel (auto-routed, decoupled from the sm80 path)
│   │   └── gemm_bf16xfp32_sm120.h # sm120a path entry declarations (support/alignment checks)
│   └── fuse_moe/                  # fused MoE FFN kernel (own .so, lazy-loaded)
│       ├── fuse_moe_bindings.cpp  # fuse_moe group registration entry
│       ├── fuse_moe_op.cu/.h      # operator entry: validation / workspace / arch dispatch
│       ├── fuse_moe_launch.h      # arch dispatch entry (gemm engine / tile / PDL policies)
│       ├── common/                # shared: count/act/reduce kernels, GEMM traits, params
│       ├── sm89/                  # cp.async family (sm80+; also the sm120 default engine)
│       └── sm120/                 # TMA + mbarrier family (optional engine via FUSE_MOE_TMA=1)
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
movement path adopts the FA3/hopper-style TMA design ported to sm120);
fuse_moe is ported from the sm90 implementation in
[hpc-ops](https://github.com/meituan-hpc/hpc-ops) (group GEMM scheduling and
the count/build_indices flow); and it depends on NVIDIA CUTLASS/CuTe (vendored
as headers under `thirdparty/`). If this project helps you, please also cite
the original works:

- FlashAttention official repo: https://github.com/Dao-AILab/flash-attention
- hpc-ops: https://github.com/Tencent/hpc-ops
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
