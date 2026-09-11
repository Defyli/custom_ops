# custom_ops — 面向生成式推荐的高性能 CUDA 算子库

中文 | [English](README_EN.md)

面向生成式推荐系统的高性能 CUDA 算子库，包含三个核心算子：

- **`mha_fwd_with_mask`**：支持**任意加法 mask**（0=可见 / -inf=屏蔽）的
  FlashAttention-2 前向，深度适配消费级 GPU（sm120 / sm89，bf16 / fp16）与 Volta
  数据中心 GPU（sm70 / V100，fp16），带 mask 性能显著优于 PyTorch SDPA
- **`mixed_gemm`**：混合精度 GEMM，解决推荐模型 **bf16 权重精度损失大、
  tf32 性能不足**的问题——bf16 tensor core 算主项 + 低精度 tensor core 算
  residual 补偿项，以接近 bf16 的开销恢复接近 fp32 的精度，且在 epilogue
  融合执行 bias 相加与 silu/gelu 激活
- **`fuse_moe`**：MoE FFN 前向融合（单 GPU），计数 / 路由重排 / gate_up
  group GEMM / silu·mul / down group GEMM / topk 加权归约全部融合为 4 个
  kernel，bf16/fp16 输入、fp32 累加

**亮点**

**注意力算子 `mha_fwd_with_mask`**

- **任意 mask 直达 softmax**：因果、滑动窗口、padding、随机稀疏（item 级屏蔽）
  等任意形态，无需改 kernel
- **sm120（RTX 5090D）**：标准场景 **160–180 TFLOPS**（cuBLAS bf16 实测峰值的
  68–76%），SDPA+mask 的 **1.62–2.48x**、官方 FlexAttention 同场景的 **1.5–3.5x**；
  小 grid 长序列场景最高 **60x**（vs FlexAttention 最高 **104x**）；ragged
  （Sk%8≠0 自动 pad / 任意 Sq）场景 SDPA+mask 的 **1.65–7.8x**
- **sm89（RTX 4090D）**：开箱即用，标准场景 SDPA+mask 的 **1.15–1.86x**、
  FlexAttention 的 **1.3–4.5x**；小 grid 长序列场景最高 **31x**（vs FlexAttention
  最高 **51x**）；ragged 场景多数 shape 保持加速
- **sm70（Tesla V100）**：fp16 专用路径（V100 无 bf16 tensor core），数据
  通路全手工——WMMA m16n16k16 + 冲突消除 swizzle（sm70 无 ldmatrix /
  cp.async / TMA）；标准场景 d=128 **16–24 TFLOPS**、d=64 **11–17 TFLOPS**
  （cuBLAS fp16 实测峰值的 13–29%），SDPA+mask 的 **1.07–1.82x**；无 mask
  等价场景 d=128 超开源 flash-attention-v100 参考的 **1.23–1.31x**；
  暂无 Split-KV（小 grid 长序列为已知短板）
- **原生 GQA**：K/V 头数整除 Q 头数即可，无需手动扩展
- **自适应 Split-KV**（sm89/sm120）：cost model 自动选择 split 数，大 grid
  自动退化为单 kernel，零开销、无需调参

**混合精度 GEMM `mixed_gemm`**

- **精度 ≈ fp32，速度 > tf32**：权重离线拆分为 bf16 主项 + fp8/int8
  residual 补偿项，消除权重的系统性舍入偏差（RMS 误差比纯 bf16 低 **2.3 倍**，
  最大误差低 **3.9 倍**），RTX 5090D 上为 fp32 matmul 的 **1.7–2.6x**（sm120a
  TMA 专用路径，较通用路径再快 **1.6–3.0x**），RTX 4090D 上为 **1.3–2.5x**，
  大 shape 有效算力最高 **118 TFLOPS**
- **sm120a TMA 专用路径**：RTX 5090 系自动启用——4 路 operand 全 TMA 搬运
  （OOB 自动补零，无谓词开销）+ mbarrier 双屏障流水线 + bulk TMA store
  epilogue，数值语义与通用路径完全一致，不对齐等不满足约束时自动回退
- **Epilogue 融合**：bias 相加 + silu/gelu 激活融合在 GEMM kernel 内，
  不产生额外 kernel 与中间显存
- **双 residual 后端**：FP8 e4m3（SM89+，需编译期 CUDA >= 12.4，小 M 略快、
  免 scale 存储）与 INT8 动态量化（SM80+，CUDA 11.8 即可，大 M 更快），
  精度一致，编译期自动选择
- **自适应 tile/split-K**：沿用原 TensorRT 插件的 wall-clock 启发式，
  小 M 自动 split-K，无需调参

**MoE 前向融合 `fuse_moe`**

- **大 shape 有效算力 171–190 TFLOPS**（mma.sync 峰值的 82–91%）：
  4096×4096×1408 **188 TF**、16384×2048×1024 **188 TF**；vs PyTorch eager
  **1.9–14.8x**，vs `torch.compile` **1.4–2.5x**，vs compile+CUDA graph
  最高 **98x**（小 batch 大 E 场景）
- **gemm1 gate/up 配对融合**：同一 CTA 同时计算 W1 的 gate 与 up 两个 N
  面板（共享 X tile，装载量减半），epilogue 直接 `silu(gate)·up` 写
  act_out——免 `gate_up_out (T, 2I)` 物化与独立激活 kernel
- **跨 tile 连续流水**：slab 发射流全局计数，task（tile）边界不排空；
  scatter 行索引驻留寄存器滚动预取；激活全程 expert 有序 compact 布局
  （无 padded 空洞与预物化）
- **架构自适应**：sm89 / sm120 统一 cp.async 引擎（sm120 可选 TMA +
  mbarrier 引擎，`FUSE_MOE_TMA=1`）；kTileM 按 avg tokens/expert 选
  32/64/128、kStage 按 smem 预算自适应；PDL 串联四个 kernel

**通用能力**

- **JIT 自动编译**：首次 import 自动构建，多进程安全，支持
  `torch.compile`/AOTI

> FA 算子完整的移植与优化过程记录见
> [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md)
> 与
> [docs/fa_sm70_porting_and_optimization.md](docs/fa_sm70_porting_and_optimization.md)；
> fuse_moe 见
> [docs/fuse_moe_porting_and_optimization.md](docs/fuse_moe_porting_and_optimization.md)。

## 性能

### RTX 5090D (sm120, bf16)

测试环境：RTX 5090D / PyTorch 2.11 / CUDA 12.8，benchmark 默认参数
（见 [Benchmark](#benchmark) 小节），带 10% 随机 -inf mask。

标准场景：**160–180 TFLOPS**（cuBLAS bf16 实测峰值的 68–76%），SDPA+mask 的
**1.62–2.48x**、官方 FlexAttention（同 mask 场景）的 **1.5–3.5x**：

| Shape | custom | SDPA+mask | 加速比 | FlexAtt | vs Flex |
|---|---|---|---|---|---|
| d64 B=4 H=16 S=2048 | 387.1µs (177.5 TF) | 754.6µs | **1.95x** | 1355.6µs | **3.50x** |
| d64 B=32 H=16 S=1024 | 858.8µs (160.0 TF) | 1394.8µs | **1.62x** | 2339.3µs | **2.72x** |
| d64 B=1 H=8 S=8192 | 853.9µs (161.0 TF) | 1672.2µs | **1.96x** | 2996.2µs | **3.51x** |
| d128 B=4 H=16 S=2048 | 775.9µs (177.1 TF) | 1870.9µs | **2.41x** | 1710.8µs | **2.20x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 775.9µs (177.1 TF) | 1925.9µs | **2.48x** | 1711.1µs | **2.21x** |
| d128 B=32 H=16 S=1024 | 1527.1µs (180.0 TF) | 3572.7µs | **2.34x** | 3101.1µs | **2.03x** |

小 grid + 长序列场景（Split-KV 自动生效）：SDPA+mask 的 **13–60x**、
FlexAttention 的 **22–104x**：

| Shape | custom | SDPA+mask | 加速比 | FlexAtt | vs Flex |
|---|---|---|---|---|---|
| d128 Sq=128 Sk=8192 | 22.5µs | 558.1µs | **24.8x** | 962.5µs | **42.7x** |
| d128 Sq=128 Sk=32768 | 36.9µs | 2202.6µs | **59.8x** | 3823.6µs | **103.7x** |
| d128 Sq=1024 Sk=8192 | 43.0µs (99.9 TF) | 558.1µs | **13.0x** | 960.5µs | **22.3x** |
| d64 Sq=128 Sk=8192 | 16.4µs | 436.3µs | **26.6x** | 1053.7µs | **64.3x** |

ragged 场景（`--suite ragged`，Sk%8!=0 自动 pad / 任意 Sq，「对齐等价」列度量
自动 pad 开销；vs FlexAttention **1.17–19.7x**）：

| Shape (B,H,Hk,Sq,Sk,d) | custom | 对齐等价（pad 开销） | SDPA+mask | 加速比 | vs Flex |
|---|---|---|---|---|---|
| d64 Sq=2048 Sk=2053 | 100.4µs (85.8 TF) | 57.7µs（+74%） | 184.3µs | 1.83x | 2.57x |
| d128 Sq=1024 Sk=1031 | 98.3µs (88.0 TF) | 67.9µs（+45%） | 178.3µs | 1.81x | 1.17x |
| d128 B=4 Sq=2048 Sk=4099 | 2003.7µs (137.3 TF) | 1485.1µs（+35%） | 3983.4µs | 1.99x | 1.30x |
| d64 Sq=8192 Sk=8195 | 1214.5µs (113.2 TF) | 853.9µs（+42%） | 2006.0µs | 1.65x | 2.27x |
| d64 Sq=1000 Sk=1024（任意 Sq） | 32.7µs (128.1 TF) | ≈零开销 | 80.9µs | 2.47x | 4.44x |
| d64 B=2 Sq=333 Sk=1024 | 30.7µs (90.9 TF) | ≈零开销 | 79.8µs | 2.60x | 4.59x |
| d128 Sq=127 Sk=2048 | 18.4µs | ≈零开销 | 143.4µs | 7.78x | 19.7x |
| d128 Sq=1000 Sk=4099（双 ragged） | 301.1µs (111.5 TF) | — | 620.4µs | 2.06x | 1.28x |
| d128 B=4 Hk=4 Sq=2048 Sk=2053（GQA） | 896.7µs (153.6 TF) | 775.9µs（+16%） | 2094.1µs | 2.34x | 1.49x |

> 注：Sk ragged 的 pad 拷贝开销绝对量与 sm89 同量级，但 sm120 kernel 本身更快，
> 相对占比更高（+35–72%）；GQA 因 K/V 更小而降至 +16%。性能敏感且 Sk 固定的
> 场景建议数据侧预对齐到 8 倍数（零拷贝主路径）。

> 注：SDPA 带任意 `attn_mask` 时只能走 MemEfficient 后端（FlashAttention 后端不支持
> 任意 mask）。部分 shape 下本算子**带 mask 甚至比 SDPA 不带 mask 的 Flash 后端更快**。
>
> FlexAtt = `torch.nn.attention.flex_attention`（torch 官方为自定义 mask 设计的算子，
> torch≥2.5）：`create_block_mask` 预构建 + `torch.compile`，block_mask 构建与
> Triton 编译开销均不计入计时，与本算子「mask 预构建后直进 kernel」对齐；
> 10% 随机 -inf mask 下几乎全部 block 为 partial，FlexAttention 无法利用块稀疏
> 跳过。FlexAttention 无 Split-KV，小 grid 长序列 shape 差距最大（22–104x）。

### RTX 4090D (sm89, bf16)

sm89 路径为 cp.async 实现，同样具备自适应 Split-KV（自 sm120 移植）：小 grid
长序列自动切分 K 维填满 SM，大 grid 自动退化为单 kernel，无需任何配置。

测试环境：RTX 4090D / PyTorch 2.11 / CUDA 12.8，benchmark 默认参数
（见 [Benchmark](#benchmark) 小节）。

标准场景（大 grid）：SDPA+mask 的 **1.15–1.86x**、FlexAttention 的
**1.30–4.50x**：

| Shape | custom | SDPA+mask | 加速比 | FlexAtt | vs Flex |
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

小 grid + 长序列场景（Split-KV 自动生效）：SDPA+mask 的 **7.1–31.3x**、
FlexAttention 的 **9.6–50.8x**：

| Shape | custom | SDPA+mask | 加速比 | SDPA flash* | FlexAtt | vs Flex |
|---|---|---|---|---|---|---|
| d128 Sq=128 Sk=8192 | 23.5µs (22.9 TF) | 417.8µs | **17.81x** | 29.7µs | 621.8µs | **26.5x** |
| d128 Sq=128 Sk=32768 | 54.3µs (39.6 TF) | 1696.7µs | **31.26x** | 52.9µs | 2226.2µs | **41.0x** |
| d128 Sq=512 Sk=8192 | 36.9µs (58.3 TF) | 414.7µs | **11.25x** | 39.1µs | 622.7µs | **16.9x** |
| d128 Sq=1024 Sk=8192 | 58.4µs (73.6 TF) | 414.7µs | **7.11x** | 48.5µs | 562.2µs | **9.6x** |
| d128 H=2 Hk=1 Sq=512 Sk=16384 | 94.2µs (91.2 TF) | 818.1µs | **8.68x** | 123.9µs | 1233.9µs | **13.1x** |
| d64 Sq=128 Sk=8192 | 20.5µs (13.1 TF) | 444.4µs | **21.70x** | 17.2µs | 1039.6µs | **50.8x** |
| d64 Sq=1024 Sk=8192 | 41.0µs (52.4 TF) | 403.5µs | **9.85x** | 33.8µs | 1044.5µs | **25.5x** |
| d64 H=2 Hk=1 Sq=512 Sk=16384 | 55.3µs (77.7 TF) | 774.4µs | **14.00x** | 66.4µs | 2012.2µs | **36.4x** |

> \* SDPA flash 列为不带 mask 的 FlashAttention 后端参照（不支持任意 mask）；
> 公平对照是 SDPA+mask 列。得益于 Split-KV，部分小 grid shape 已追平甚至超过
> 该参照（如 d128 H=2 Hk=1 Sq=512 Sk=16384: 94.2µs vs 123.9µs）。

作为参照：未启用 Split-KV 时，小 grid 长序列 shape 只能由单个 CTA 串行处理全部
KV 块，耗时 234–935µs；Split-KV 自动生效后降至 19–53µs（**10–18x**）。

ragged 场景（`Sk % 8 != 0` 自动 pad / 任意 `Sq`，`--suite ragged`）。「对齐等价」
列为 Sk 向上取整到 8 倍数前同 shape 的 custom 耗时，用于度量自动 pad 的开销；
vs FlexAttention **1.30–12.8x**：

| Shape (B,H,Hk,Sq,Sk,d) | custom | 对齐等价（pad 开销） | SDPA+mask | 加速比 | vs Flex |
|---|---|---|---|---|---|
| d64 Sq=2048 Sk=2053 | 179.2µs (48.1 TF) | 120.8µs（+48%） | 178.2µs | 0.99x | 2.31x |
| d128 Sq=1024 Sk=1031 | 138.2µs (62.6 TF) | 100.4µs（+38%） | 187.4µs | 1.36x | 1.39x |
| d128 B=4 Sq=2048 Sk=4099 | 2845.7µs (96.7 TF) | 2107.2µs（+35%） | 4064.3µs | 1.43x | 1.30x |
| d64 Sq=8192 Sk=8195 | 1702.9µs (80.7 TF) | 1121.2µs（+52%） | 2063.4µs | 1.21x | 2.39x |
| d64 Sq=1000 Sk=1024（任意 Sq） | 67.6µs (62.1 TF) | ≈零开销 | 77.8µs | 1.15x | 3.92x |
| d64 B=2 Sq=333 Sk=1024 | 37.9µs (73.7 TF) | ≈零开销 | 66.6µs | 1.76x | 3.89x |
| d128 Sq=127 Sk=2048 | 31.9µs | ≈零开销 | 111.6µs | 3.50x | 12.8x |
| d128 Sq=129 Sk=2048 | 32.8µs | ≈零开销 | 104.4µs | 3.19x | 4.53x |
| d128 Sq=1000 Sk=4099（双 ragged） | 477.2µs (70.4 TF) | — | 655.4µs | 1.37x | 1.39x |
| d128 B=4 Hk=4 Sq=2048 Sk=2053（GQA） | 1235.0µs (111.6 TF) | 1069.2µs（+16%） | 2154.7µs | 1.74x | 1.52x |

> 注：`Sk % 8 != 0` 时算子内部对 K/V/mask 做一次 pad 拷贝（mask 为 `Sq×Sk`
> 大张量，主导开销），pad 后 `Sk8` 跨过 KV tile 边界也会略增尾部 tile 计算；
> GQA 场景 K/V 更小，开销也随之下降。`Sq` 任意无拷贝（行谓词），耗时随实际
> Sq 缩放。对性能敏感且 Sk 固定的场景，建议数据侧预对齐到 8 倍数（零拷贝
> 主路径）；极端小 shape（如首行 0.99x 一例）建议直接比对后选用。
>
> FlexAtt 对照说明：FlexAttention 默认配置（BLOCK_M=128）在 sm89 d=128 上
> **无法编译**（smem 需求约 112KB 超出 Ada 架构约 99KB 上限，Inductor 报
> 「No valid triton configs」），上表为其降级 `BLOCK_M=64` 后的结果——
> 这是该架构下能跑起来的唯一官方配置。sm120（Blackwell，228KB smem）
> 无此问题，均用默认配置。

### Tesla V100 (sm70, fp16)

sm70 路径为 **fp16 专用**（V100 无 bf16 tensor core）：`mha_fwd_with_mask`
传入 fp16 张量即自动启用。数据通路全手工搭建——WMMA m16n16k16 + 冲突消除
swizzle（sm70 无 ldmatrix / cp.async / TMA），d=128 需 opt-in 96KB smem。
暂无 Split-KV，小 grid 长序列为已知短板（计划自 sm120 移植）。

测试环境：V100-PCIE-32GB / PyTorch 2.0.1 / CUDA 11.7，带 10% 随机 -inf mask。

标准场景（大 grid）：SDPA+mask 的 **1.07–1.82x**，d=128 峰值 **24.2 TFLOPS**
（本机 cuBLAS fp16 实测峰值 84.2 TF 的 29%）：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d64 B=4 H=16 S=1024 | 1001.8µs (17.1 TF) | 1711.6µs | **1.71x** |
| d64 B=4 H=16 S=2048 | 3982.4µs (17.3 TF) | 6765.7µs | **1.70x** |
| d64 B=4 H=16 Hk=4 S=2048 (GQA) | 3986.7µs (17.2 TF) | 7268.9µs | **1.82x** |
| d128 B=1 H=8 S=512 | 65.2µs (16.5 TF) | 92.4µs | **1.42x** |
| d128 B=4 H=16 S=1024 | 1450.8µs (23.7 TF) | 1864.9µs | **1.29x** |
| d128 B=4 H=16 S=2048 | 5694.7µs (24.1 TF) | 7313.5µs | **1.28x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 5688.6µs (24.2 TF) | 7761.5µs | **1.36x** |
| d128 B=1 H=8 S=2048 | 878.2µs (19.6 TF) | 938.2µs | **1.07x** |

> 注：V100 + torch 2.0.1 上 SDPA 带 `attn_mask` 只能走 math 后端。FlexAttention
> 需要 torch≥2.5，该环境不可用（N/A）。无 mask 等价场景（zero additive mask）
> 下本算子 d=128 全面超过开源 flash-attention-v100 参考实现 **1.23–1.31x**
> （19.1–24.2 TF vs 15.5–19.5 TF，且对方不支持任意 mask）；d=64 持平
> （0.95–1.07x）。

ragged 场景（`--suite ragged`，Sk%8!=0 自动 pad / 任意 Sq）：

| Shape (B,H,Hk,Sq,Sk,d) | custom | 对齐等价（pad 开销） | SDPA+mask | 加速比 |
|---|---|---|---|---|
| d64 Sq=2048 Sk=2053 | 735.4µs (11.7 TF) | 614.3µs（+20%） | 1175.6µs | 1.60x |
| d128 Sq=1024 Sk=1031 | 558.6µs (15.5 TF) | 446.4µs（+25%） | 764.1µs | 1.37x |
| d64 B=2 Sq=512 Sk=4099 | 1511.4µs (11.4 TF) | 1211.1µs（+25%） | 2253.6µs | 1.49x |
| d64 Sq=1000 Sk=1024（任意 Sq） | 312.6µs (13.4 TF) | ≈零开销 | 422.5µs | 1.35x |
| d128 B=2 Hk=4 Sq=500 Sk=2053（GQA 双 ragged） | 1026.5µs (16.4 TF) | — | 1747.6µs | 1.70x |

> 注：V100 上自动 pad 开销 +20–25%（kernel 本身较慢，相对占比低于 sm89/sm120）。
> 极小 Sq 的 shape（如 Sq=127 d128，仅 16 CTA）无 Split-KV 时不如 SDPA math
> 后端，为已知短板（见上）。

### mixed_gemm（FP8 / INT8 residual 双后端）

推荐 MLP 层 `silu(x @ W^T + b)`（fp32 权重/输入）。

#### RTX 5090D（sm120a TMA 专用路径）

测试环境：RTX 5090D / PyTorch 2.11 / CUDA 12.8，FP8 后端。sm120a 上自动
启用 TMA + mbarrier 专用数据通路（数值语义与通用路径完全一致）；`sm80 路径`
列为同机强制回退的 A/B 对照（`GEMM_MIXED_FORCE_SM80=1`）。

sm120a 专用路径较通用路径提升 **1.6–3.0x**，为 fp32 matmul 的 **1.7–2.6x**；
4096³ 大 shape 有效算力 **118 TFLOPS**（同 shape bf16 matmul 的 72%）。

| Shape (M,N,K) | mixed (sm120a) | sm80 路径 | 提升 | fp32 | vs fp32 | tf32 | bf16* |
|---|---|---|---|---|---|---|---|
| 16, 4096, 4096 | 53.2µs | 91.7µs | 1.72x | 112.6µs | **2.11x** | 71.7µs | 53.2µs |
| 64, 4096, 4096 | 57.3µs | 93.9µs | 1.64x | 98.3µs | **1.71x** | 108.5µs | 55.3µs |
| 256, 4096, 4096 | 104.5µs | 313.3µs | **3.00x** | 184.4µs | **1.76x** | 127.3µs | 69.7µs |
| 1024, 4096, 4096 | 337.9µs | 985.1µs | **2.92x** | 598.0µs | **1.77x** | 405.9µs | 219.1µs |
| 4096, 4096, 4096 | 1163.3µs (118.1 TF) | 2198.5µs | 1.89x | 2521.8µs | **2.17x** | 1453.1µs | 843.7µs |
| 16, 16384, 1024 | 53.2µs | 90.1µs | 1.69x | 137.2µs | **2.58x** | 73.7µs | 41.0µs |
| 128, 1000, 2048 | 18.4µs | 43.0µs | **2.34x** | 30.8µs | **1.67x** | 28.6µs | 26.6µs |
| 512, 4096, 256 | 22.5µs | 65.5µs | **2.91x** | 47.1µs | **2.09x** | 31.2µs | 16.4µs |

> \* bf16 列口径与下方 4090D 小节相同：`bf16(x) @ bf16(W)^T`（权重离线预转）
> + fp32 epilogue。

sm120a 专用路径的设计要点（实现见
`csrc/mixed_gemm/gemm_bf16xfp32_sm120.cu` 文件头注释）：

- 4 路 operand（bf16 主项 + 1 字节 residual 的 X/W）全部 TMA 搬运：单线程
  发射 bulk 拷贝，OOB 自动补零取代逐线程谓词，消除 sm80 路径的谓词寄存器
  开销与地址计算开销（sm120 无 wgmma，计算骨架沿用 `mma.sync`）
- `full`/`empty` mbarrier 双屏障流水线取代 `cp.async` wait +
  `__syncthreads`：producer/consumer 全异步，跨 tile 相位自然延续
- Epilogue 在 Y 对齐时用 bulk TMA store（自动裁剪越界行列），不满足时
  降级 bounds-checked elementwise；sY 与 operand 共享 smem（代理栅栏保护）
- Cache hint 分流：W（跨 M-tile 复用）EVICT_LAST，X（流式）EVICT_FIRST
- 路由约束：k%16==0 且各指针 16B 对齐（TMA 全局 stride 要求）；不满足时
  整体回退 sm80 通用路径，数值语义不变

#### RTX 4090D（通用路径）

测试环境：RTX 4090D /
PyTorch 2.11 / CUDA 12.8（FP8 与 INT8 后端同机对照）；编译期 CUDA < 12.4 时
FP8 自动降级 INT8（精度相同）。

通用路径最优后端为 fp32 matmul 的 **1.3–2.5x**、tf32 的 **1.2–1.6x**，
大 shape 有效算力 **100 TFLOPS**；精度接近 fp32（权重舍入误差被完全消除，
误差仅剩激活的无偏舍入噪声）。

| Shape (M,N,K) | mixed FP8 | mixed INT8 | fp32 | vs fp32† | tf32 | bf16* |
|---|---|---|---|---|---|---|
| 16, 4096, 4096 | 96.3µs | 103.4µs | 122.9µs | **1.28x** | 117.8µs | 61.4µs |
| 64, 4096, 4096 | 100.4µs | 109.6µs | 130.8µs | **1.30x** | 122.9µs | 65.5µs |
| 256, 4096, 4096 | 151.6µs | 145.4µs | 245.8µs | **1.69x** | 210.8µs | 86.8µs |
| 1024, 4096, 4096 | 511.0µs | 440.4µs | 935.0µs | **2.12x** | 602.1µs | 296.6µs |
| 4096, 4096, 4096 | 1730.6µs (79.4 TF) | 1374.2µs (100.0 TF) | 3449.9µs | **2.51x** | 2188.3µs | 1044.5µs |
| 512, 4096, 256 | 30.8µs | 29.7µs | 56.3µs | **1.90x** | 45.9µs | 19.5µs |

精度（4096³，相对 fp32 金标准，FP8 与 INT8 后端实测一致）：mean rel-err
mixed **8.0e-3** vs bf16 1.2e-2；RMS 误差 **1.7e-3** vs 4.0e-3（低 2.3x）；
最大绝对误差 **1.6e-2** vs 6.1e-2（低 3.9x）。

后端选择说明：

- **小 M（≤64）FP8 略快**（7–10%，量化 kernel 更简单）；**大 M INT8 更快**
  （最高 21%）；`vs fp32†` 取每行更优后端的倍率。`backend="auto"` 在 FP8
  可用时默认选 FP8（精度相同、免 per-channel scale 存储），大 M 追求极致
  性能可显式 `backend="int8"`
- 两种 residual 后端精度一致：权重舍入偏差均被消除，剩余误差主导项是激活的
  bf16 舍入噪声（两种量化精度都已足够细）
- CUDA 11.8 编译时同一 INT8 kernel 大 shape 慢约 17%（nvcc 代码生成差异，
  4096³ 实测 1658µs），小 shape 不受影响；建议用较新 CUDA 编译

> \* bf16 列为 `bf16(x) @ bf16(W)^T`（权重离线预转，与 mixed_gemm 的离线权重拆分
> 对等）+ fp32 epilogue：速度快 1.3–1.7x（单次 GEMM vs 主项+补偿双 GEMM），但
> 权重舍入误差完全未补偿。

### fuse_moe（MoE FFN 前向融合，单 GPU）

测试环境：RTX 5090D / PyTorch 2.6 / CUDA 12.8，bf16，均匀路由。sm120 上
默认引擎为 cp.async 连续流水 kernel（各 shape 实测均优于 TMA 引擎
3–15%）；`FUSE_MOE_TMA=1` 可选切换 TMA + mbarrier 引擎（16384 case
2807µs vs 默认 2197µs）。sm89（RTX 4090D）走同一 cp.async 路径。

大 shape 有效算力 **171–194 TFLOPS**（mma.sync 峰值的 82–94%；峰值 194.3 TF
@4096×4096×1408，含 gemm2 相邻 N-pair 变体）；vs eager **1.9–14.8x**，vs
`torch.compile` **1.4–2.5x**，vs compile+reduce-overhead（CUDA graph）最高
**98x**；vs sglang 生产 Triton MoE（本机 tuned，
`benchmark/benchmark_fuse_moe_vs_sglang.py` + 自包含移植包
`benchmark/sglang_triton_moe/`，无需安装 sglang）
**5090D 1.3–5.2x / 4090D 1.1–3.9x**（详见 docs 第 5 节）。

5090D（sm120）：

| Shape (S,H,I,E,K) | custom µs | sglang-def µs | sglang-tuned µs | vs tuned |
|---|---|---|---|---|
| (512,2048,1024,8,2) | 99.6 | 513.5 | 513.7 | **5.2x** |
| (1024,2048,1024,8,2) | 171.0 | 534.3 | 533.9 | 3.1x |
| (4096,2048,1024,8,2) | 574.3 | 961.8 | 966.2 | 1.7x |
| (16384,2048,1024,8,2) | 2132.4 | 2790.0 | 2771.8 | 1.3x |

4090D（sm89，同一移植包与 tuning 流程，125.0 TF vs sglang 106.8 TF）：

| Shape (S,H,I,E,K) | custom µs | sglang-def µs | sglang-tuned µs | vs tuned |
|---|---|---|---|---|
| (512,2048,1024,8,2) | 176.4 | 718.0 | 721.5 | **4.1x** |
| (1024,2048,1024,8,2) | 299.0 | 701.0 | 700.8 | 2.3x |
| (4096,2048,1024,8,2) | 914.9 | 1307.5 | 1304.9 | 1.4x |
| (16384,2048,1024,8,2) | 3299.4 | 3860.0 | 3856.7 | 1.2x |

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

设计要点（实现见 `csrc/fuse_moe/`，完整优化历程见
[docs/fuse_moe_porting_and_optimization.md](docs/fuse_moe_porting_and_optimization.md)）：

- **跨 tile 连续流水**：slab 发射流全局计数（`stage = cnt % kStage`），
  task（tile）边界不排空；scatter 行索引驻留寄存器滚动预取；kStage 按
  smem 预算自适应（96KB）
- **gemm1 gate/up 配对融合**：同一 CTA 同时计算 W1 的 gate 与 up 两个
  N 面板（共享 X tile，X 装载量减半），epilogue 直接 `silu(gate)·up`
  直写 act_out——免 `gate_up_out (T, 2I)` 物化与独立激活 kernel
- **compact 直读**：激活按 expert 有序紧凑布局流动，无 padded 空洞；
  cp.async scatter 路径 gather-on-load 直读 token 序 x
- **gemm2 相邻 N-pair**：每 task 覆盖相邻两个 64 宽 N 面板（共享 X tile
  与 smem stage），per-slab MMA 密度 ×2——ncu 归因 gemm2 延迟掩盖不足
  （occupancy 16.7% × 浅流水）后引入，全 shape 谱系 e2e 优 2–3%
- **tile 自适应**：kTileM 按 avg tokens/expert 选 32/64/128（128 将 W
  复用面翻倍，avg≥256 启用）；kTileK 按 k%128 选 64/128（smem 门控）
- **PDL 串联**：count → gemm1 → gemm2 → reduce 四个 kernel 以
  Programmatic Dependent Launch 串联（sm89 自动退化为 stream 顺序）

## 环境要求

- NVIDIA GPU：sm120（RTX 5090D，FA TMA 主路径与 mixed_gemm sm120a TMA 路径
  均充分测试）、sm89（RTX 4090D，cp.async 路径（含自 sm120 移植的
  Split-KV），已测试）或 sm70（Tesla V100，FA fp16 WMMA 路径，已测试；
  mixed_gemm 不支持 sm70）；其余 sm80+ 架构理论上可编译运行，未验证
- CUDA >= 11.8（FA sm89 / sm70 路径 / mixed_gemm INT8 后端）/ >= 12.4
  （mixed_gemm FP8 后端，SM89+）/ >= 12.8（FA / mixed_gemm 的 sm120a 路径），
  GCC >= 9
- PyTorch >= 2.1（CUDA 版本），fp16 / bf16

## 快速开始

本仓库根目录即为 Python 包 `custom_ops`，无需安装，将仓库**上一级目录**加入
`PYTHONPATH` 即可：

```bash
git clone <repo_url> custom_ops
export PYTHONPATH=$(dirname $(pwd)/custom_ops):$PYTHONPATH   # 或在代码中 sys.path.insert
```

```python
import torch
from custom_ops import ops   # 首次 import 自动 JIT 编译（约 30~60s），之后秒级加载

B, H, Sq, Sk, d = 2, 16, 1024, 1024, 128
q = torch.randn(B, H, Sq, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, Sk, d, device="cuda", dtype=torch.bfloat16)
mask = torch.zeros(B, 1, Sq, Sk, device="cuda", dtype=torch.bfloat16)
mask[..., 512:] = float("-inf")   # 任意加法 mask：0=可见，-inf=屏蔽

out = ops.mha_fwd_with_mask(q, k, v, mask)   # (B, H, Sq, d) bf16

# ── 混合精度 GEMM：y = silu(x @ W^T + b) ──────────────────────────
from custom_ops import split_mixed_precision_weight

x = torch.randn(4096, 4096, device="cuda")              # fp32 激活
w = torch.randn(4096, 4096, device="cuda") * 0.05      # fp32 权重
b = torch.randn(4096, device="cuda") * 0.1

w_high, w_low, w_scale = split_mixed_precision_weight(w)  # 模型加载时一次性拆分
y = ops.mixed_gemm(x, w_high, w_low, w_scale, bias=b,
                   activation="silu")                   # fp32 输出，精度 ≈ fp32

# ── MoE FFN：y[s] = Σ_j scale[s,j]·(Down_ej @ silu(GateUp_ej @ x[s])) ─
S, H, I, E, K = 4096, 2048, 1024, 8, 2
x  = torch.randn(S, H, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.05
w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.05
topk_ids   = torch.randint(0, E, (S, K), device="cuda", dtype=torch.int32)
topk_scale = torch.rand(S, K, device="cuda")            # fp32 加权

y = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)       # (S, H) bf16
```

> dtype 约定：fp16 全架构可用（sm70 / sm89 / sm120）；bf16 仅 SM80+（V100 无
> bf16 tensor core，请使用 fp16）。sm89/sm120 上两种 dtype 均可，按上层模型
> 的精度策略自选。

编译缓存默认在 `~/.cache/torch_extensions`，可用 `TORCH_EXTENSIONS_DIR` 指定；
多进程同时首次编译由文件锁保护，是安全的。

## API

### `ops.mha_fwd_with_mask(q, k, v, mask) -> Tensor`

FlashAttention-2 前向，支持任意加法 mask。fp16 全架构可用（sm70 / sm89 /
sm120）；bf16 仅 SM80+。V100 上 fp16 自动路由到 sm70 专用路径。

| 参数 | shape | 说明 |
|---|---|---|
| `q` | (B, H, Sq, d) | fp16（全架构）或 bf16（SM80+）CUDA 连续张量 |
| `k`, `v` | (B, Hk, Sk, d) | 与 `q` 同 dtype 的 CUDA 连续张量 |
| `mask` | (B, 1, Sq, Sk) | 加法 mask，与 `q` 同 dtype，0=可见 / -inf=屏蔽 |
| 返回 | (B, H, Sq, d) | 与输入同 dtype |

限制：

- head dim `d ∈ {64, 128}`
- `H % Hk == 0`（GQA）；MHA（Hk=H）是特例
- 仅前向，不支持 dropout / causal 标志位 / alibi / RoPE / KV-cache
  （causal 可通过 mask 表达，见 examples）

### `ops.mixed_gemm(x, w_high, w_low, w_scale, *, scale, bias, activation, out_dtype, force_splitk) -> Tensor`

混合精度 GEMM：`y = activation(x @ (w_high + w_low * scale)^T + bias)`。
bf16 tensor core 算主项 + 低精度 tensor core 算 residual 补偿项，以接近
bf16 的开销恢复接近 fp32 的精度；bias 与激活在 epilogue 融合执行。

| 参数 | shape | 说明 |
|---|---|---|
| `x` | (..., K) | fp32 / bf16 CUDA 连续张量，前导维度折叠为 M |
| `w_high` | (N, K) | bf16 主项权重（`split_mixed_precision_weight` 产出） |
| `w_low` | (N, K) | residual 权重：`float8_e4m3fn`（FP8 后端）或 `int8`（INT8 后端） |
| `w_scale` | (N,) | fp32，INT8 后端的 per-channel 量化 scale（FP8 后端传 `None`） |
| `scale` | float | residual 补偿 scale，须与权重拆分时一致（默认 1/256） |
| `bias` | (N,) | fp32，可选，epilogue 融合相加 |
| `activation` | str | `"identity"` / `"silu"` / `"gelu"`（tanh 近似） |
| `out_dtype` | dtype | `None`（fp32，推荐）或 `torch.bfloat16` |
| 返回 | (..., N) | 输出，dtype 由 `out_dtype` 决定 |

residual 后端由 `w_low.dtype` 决定，`mixed_gemm_fp8_available()` 可查询
FP8 后端可用性（编译期 CUDA >= 12.4 且 GPU 为 SM89+）。

限制：

- `K % 8 == 0`
- FP8 后端需 SM89+；INT8 后端需 SM80+

### `ops.fuse_moe(x, gate_up_weight, down_weight, topk_ids, topk_scale) -> Tensor`

MoE FFN 前向融合：
`y[s] = Σ_j topk_scale[s,j] · (Down_{e_j} @ silu(GateUp_{e_j} @ x[s]))`。
count / 路由重排 / 两个 group GEMM / 激活 / topk 加权归约全部融合为
4 个 kernel，bf16/fp16 输入、fp32 累加。

| 参数 | shape | 说明 |
|---|---|---|
| `x` | (S, H) | bf16 / fp16，token 序 |
| `gate_up_weight` | (E, 2I, H) | 与 x 同 dtype（gate 在前 up 在后） |
| `down_weight` | (E, H, I) | 与 x 同 dtype |
| `topk_ids` | (S, K) | int32，取值 ∈ [0, E) |
| `topk_scale` | (S, K) | fp32，topk 加权系数 |
| 返回 | (S, H) | 与 x 同 dtype |

限制：`H % 64 == 0`、`I % 64 == 0`、`K <= 128`、`E <= 512`、SM80+。

### `split_mixed_precision_weight(w, scale=1/256, backend="auto") -> (w_high, w_low, w_scale)`

离线（模型加载时一次性）将 fp32 权重拆分为 mixed_gemm 所需的三元组：
`w ≈ w_high + w_low * scale`。`backend="auto"` 时当前机器可用 FP8 则选 FP8，
否则降级 INT8。纯 PyTorch 实现，无需编译算子库。

### 调优环境变量（默认 auto 即接近最优，仅调优/调试用）

| 环境变量 | 作用 |
|---|---|
| `FA_NUM_SPLITS=n` | 强制 split KV 的 split 数（0=auto cost model） |
| `FA_SPLITKV=0` | 禁用 split KV |
| `FA_PERSISTENT=1` | 启用 persistent kernel（d128 部分场景 +2–5%） |
| `GEMM_MIXED_FORCE_SPLITK=n` | 强制 mixed_gemm 的 split-K 值（1/2/4/8/16，调试用） |
| `FUSE_MOE_TMA=1` | sm120 上启用 TMA + mbarrier 引擎（默认 cp.async） |
| `FUSE_MOE_TILE_M=32/64/128` | 强制 fuse_moe 的 kTileM（默认按 avg tokens 自适应） |
| `FUSE_MOE_TILE_N=64` | 强制 gemm2 回单 64 宽 N 基线（默认相邻 N-pair 128 宽，per-slab MMA 密度 ×2，实测全 shape 优 2–3%；hidden%128!=0 自动回退） |
| `FUSE_MOE_SCHED=vert` | cp.async 引擎 task 遍历序改为 N-major/vert（默认 horizon；实测本仓 shape 族均不优，见 docs 第 4 节，W_per_expert > L2 时可选） |
| `FUSE_MOE_TIME=1` | 打印 fuse_moe 各 kernel 分段耗时 |

## Examples

```bash
python examples/basic_usage.py        # 最小示例：调用 + 与 SDPA 校验
python examples/custom_mask_demo.py   # 4 种典型 mask：causal / 滑窗 / padding / 随机稀疏
python examples/gqa_example.py        # GQA：无需扩展 K/V 头
python examples/mixed_gemm_demo.py    # 混合精度 GEMM：权重拆分 + epilogue 融合 + 精度对比
python examples/fuse_moe_demo.py      # 融合 MoE：Mixtral 风格前向 + 数值校验 + vs eager 性能对比
python examples/swiglu_demo.py      # SwiGLU：Llama 风格 FFN 激活 + vs eager 性能对比
python tests/test_fuse_moe.py         # fuse_moe：21 用例正确性（含单专家/稀疏/重复路由）
```

## Benchmark

```bash
# 全量：标准场景 + 小 grid 长序列（splitkv）场景 + ragged 场景
python benchmark/benchmark_fa.py

# V100 (sm70, fp16)：标准场景 + 小 grid 场景 + ragged 场景（vs SDPA+mask / SDPA 参照）
python benchmark/benchmark_sm70.py

# 指定单组 / 单个 shape
python benchmark/benchmark_fa.py --suite standard
python benchmark/benchmark_fa.py --suite splitkv
python benchmark/benchmark_fa.py --suite ragged   # Sk%8!=0 自动 pad / 任意 Sq
python benchmark/benchmark_fa.py --shape 4 16 16 2048 2048 128

# 调整迭代与输出；--no-flex 跳过 FlexAttention 对照（省去逐 shape Triton 编译）
python benchmark/benchmark_fa.py --warmup 20 --iters 100 --mask-ratio 0.3 --csv result.csv
python benchmark/benchmark_fa.py --no-flex
```

输出每组 shape 的 custom / SDPA+mask / FlexAttention / SDPA flash（参照）耗时、
TFLOPS 与加速比，并对每个 shape 做一次 SDPA 数值校验。FlexAttention 对照
（torch≥2.5）block_mask 预构建、Triton 编译开销不计入计时；torch<2.5 或
sm89 d=128 默认配置编译失败时自动降级（BLOCK_M=64）/输出 N/A。

```bash
# mixed_gemm：对比 fp32 / tf32 / bf16 matmul 的耗时与精度
python benchmark/benchmark_mixed_gemm.py                       # 后端 auto（FP8 可用则 FP8）
python benchmark/benchmark_mixed_gemm.py --backend int8        # 指定 residual 后端
python benchmark/benchmark_mixed_gemm.py --shape 4096 4096 4096 --csv result.csv

# fuse_moe：对比 PyTorch eager / torch.compile / compile+reduce-overhead（CUDA graph）
python benchmark/benchmark_fuse_moe.py
```

## 仓库结构

```
custom_ops/
├── __init__.py            # CustomOps 通用 JIT 算子加载框架（基类）
├── recsys.py              # RecsysOps：分组懒加载门面（fa/mixed_gemm/fuse_moe）+ 权重拆分 helper
├── csrc/
│   ├── custom_ops_macros.h       # 通用算子注册框架宏
│   ├── arch_targets.h             # 架构条件编译标准入口：FA_TARGETS → FA_HAS_SM70/SM8X/SM120 + gpu_major()
│   ├── fa/
│   │   ├── fa_bindings.cpp       # FA 分组注册入口（独立 .so，懒加载）
│   │   ├── fa_fwd_op.cu           # 算子入口：校验 / 自动 pad / 填 params（零架构感知）
│   │   ├── fa_fwd_launch.h        # 架构分发入口：fa_launch_smXX 策略 + Split-KV cost model
│   │   ├── sm70/fa_fwd_sm70.h     # sm70 WMMA m16n16k16 kernel（fp16，无 split-KV）
│   │   ├── sm89/fa_fwd_kernel.h   # sm89 cp.async kernel：base / splitkv / combine
│   │   ├── sm120/fa_fwd_sm120.h   # sm120 TMA 流水线 kernel / splitkv / persistent / combine
│   │   ├── cpu/                   # CPU 参考实现（未接入算子分发，仅供参考）
│   │   └── common/                # 两路共用的参数包 / softmax / utils
│   ├── mixed_gemm/                # 混合精度 GEMM kernel（自 TRT 插件移植，独立 .so 懒加载）
│   │   ├── mixed_gemm_bindings.cpp # 分组注册入口（mixed_gemm + fp8 可用性查询）
│   │   ├── mixed_gemm_op.cu       # torch 算子入口：校验 / workspace / 分发
│   │   ├── gemm_bf16xfp32_sm80.cu # kernel：tile/split-K 启发式 + bf16 主项 + fp8/int8 补偿双 GEMM
│   │   ├── gemm_bf16xfp32_sm80.h  # kernel 入口声明 + FP8 编译期守卫
│   │   ├── gemm_bf16xfp32_sm120.cu # sm120a TMA+mbarrier 专用 kernel（自动路由，与 sm80 路径解耦）
│   │   └── gemm_bf16xfp32_sm120.h # sm120a 路径入口声明（supported/对齐检查）
│   └── fuse_moe/                  # MoE 前向融合 kernel（独立 .so 懒加载）
│       ├── fuse_moe_bindings.cpp  # 分组注册入口
│       ├── fuse_moe_op.cu/.h      # torch 算子入口：校验 / workspace / 架构分发
│       ├── fuse_moe_launch.h      # 架构分发标准入口（gemm 引擎/tile/PDL 策略内封）
│       ├── common/                # 跨架构共享：count/act/reduce kernels、GEMM traits、params
│       ├── sm89/                  # cp.async 家族（sm80+ 通用，sm120 默认引擎亦复用）
│       └── sm120/                 # TMA + mbarrier 家族（FUSE_MOE_TMA=1 可选引擎）
├── thirdparty/            # CUTLASS / CuTe（头文件依赖）
├── benchmark/             # 性能基准测试
├── examples/              # 使用示例
├── tests/                 # 数值正确性测试
└── docs/                  # 移植与优化全记录（含 roofline / NCU 分析）
```

## 复用 CustomOps 框架

`CustomOps` 基类可复用于任何 PyTorch CUDA 自定义算子库，只需继承并覆盖少量配置：

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

即可获得：自动 JIT 编译/快速 dlopen、多进程文件锁、GPU 架构自动探测、
GCC 版本配置、优雅降级、`torch.compile`/AOTI fake 注册。

## 致谢与引用

本项目的 kernel 实现基于 FlashAttention 官方源码修改而来（计算骨架沿用 FA2，
数据通路参考 FA3/hopper 的 TMA 写法移植至 sm120）；fuse_moe 移植自
[hpc-ops](https://github.com/meituan-hpc/hpc-ops) 的 sm90 实现（group GEMM
调度与 count/build_indices 流程）；并依赖 NVIDIA CUTLASS/CuTe
（已作为头文件内置于 `thirdparty/`）。如果本项目对您有帮助，请同时引用原项目：

- FlashAttention 官方仓库：https://github.com/Dao-AILab/flash-attention
- hpc-ops：https://github.com/Tencent/hpc-ops
- CUTLASS：https://github.com/NVIDIA/cutlass

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

本项目采用 [BSD 3-Clause](LICENSE) 许可证。

- 本仓库代码基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention)
  （BSD 3-Clause）修改，原版权声明保留在对应源文件头部；
- `thirdparty/` 下的 CUTLASS/CuTe 遵循其原始 BSD 3-Clause 许可证，
  见 [thirdparty/LICENSE.cutlass](thirdparty/LICENSE.cutlass)。
