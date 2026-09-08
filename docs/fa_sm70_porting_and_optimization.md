# FlashAttention-2 在 Volta (V100/sm70) 上的移植与优化全记录

> **硬件**：Tesla V100-PCIE-32GB（sm_70，80 SMs，96KB smem/CTA（默认 48KB），fp16 Tensor Core 理论峰值 ≈125 TFLOPS，cuBLAS 实测 **84.2 TFLOPS**）
> **软件**：CUDA 11.7/11.8 / PyTorch 2.0.1+cu117 / CUTLASS+CuTe（thirdparty）
> **算子**：带任意加法 mask 的 FA2 Forward（fp16，d∈{64,128}，支持 GQA）
> **最终结果**：标准场景 d=128 **16–24 TFLOPS**（cuBLAS 峰值的 19–29%，SDPA+mask 的 **1.05–1.42x**，无 mask 等价场景超开源 fa-v100 参考的 **1.23–1.31x**）；d=64 **10–18 TFLOPS**（SDPA+mask 的 1.4–1.8x，与 fa-v100 持平 0.95–1.07x）；从大序列 2.4 TF 的起点累计 **7–8x**

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [第一步：摸清 sm70 的硬件约束与性能参照系](#2-第一步摸清-sm70-的硬件约束与性能参照系)
3. [优化历程时间线](#3-优化历程时间线)
   - [Phase 1（v3）：Swizzle smem——搬运方向正确，行映射 softmax 全军覆没](#phase-1v3swizzle-smem搬运方向正确行映射-softmax-全军覆没)
   - [Phase 2（v4）：MMA 线程直接管理 softmax——fragment 映射实验先行](#phase-2v4mma-线程直接管理-softmaxfragment-映射实验先行)
   - [Phase 3（v5）：P/Mask smem 时间复用 + warp 数实验（否决 8-warp）](#phase-3v5pmask-smem-时间复用--warp-数实验否决-8-warp)
   - [Phase 4（v6）：同步结构优化——大序列场景的 3.3–3.8x](#phase-4v6同步结构优化大序列场景的-338x)
   - [Phase 5（v7）：Q-in-regs + K double buffer——1.4–1.6x](#phase-5v7q-in-regs--k-double-buffer146x)
   - [Phase 6（v8/v8.1）：SASS 驱动的清理——死代码、spill、softmax 通信](#phase-6v8v81sass-驱动的清理死代码spillsoftmax-通信)
   - [Phase 7（v9）：WMMA m16n16k16——LDS 密度 3.4x 差距的终极解](#phase-7v9wmma-m16n16k16lds-密度-34x-差距的终极解d128-35)
   - [Phase 8（v9.1）：Q A-fragment 常驻寄存器——d=64 反超开源参考](#phase-8v91q-a-fragment-常驻寄存器d64-反超开源参考)
4. [最终性能全景](#4-最终性能全景)
5. [经验总结：可复用的方法论](#5-经验总结可复用的方法论)

---

## 1. 背景与目标

业务需要一个支持**任意加法 mask**（item 级屏蔽，0=可见 / -inf=屏蔽）的 Attention Forward 算子，目标硬件是 Volta（V100, sm70）。这与 sm120 的需求同源，但硬件代际差了整整三代。

V100 上的现实格局：

- **PyTorch SDPA 带 `attn_mask` 只能走 math 后端**（mem_efficient 后端不支持任意 additive mask），还要物化并读取整个 `[B,H,Sq,Sk]` mask 张量——大序列下非常昂贵
- **开源 flash-attention-v100** 性能优秀但不支持任意 additive mask（仅 causal/window/alibi），也无 split-KV
- 本仓库已有的 sm70 路径（cute `SM70_8x8x4` mma atom，64×64 tile）实测只有 **4–8 TFLOPS**——SDPA fp16 的 0.37–0.6x，硬件峰值的 ≈7%

**目标**：显著超过 SDPA+mask（同功能公平对照），追平乃至反超 fa-v100（无 mask 参照），并全程保持任意 mask / GQA / 非对齐 seqlen 的正确性。

---

## 2. 第一步：摸清 sm70 的硬件约束与性能参照系

### 2.1 指令集差异（决定技术路线）

| 特性 | sm80 (A100) | sm70 (V100) | 移植决策 |
|---|---|---|---|
| `ldmatrix` | ✅ | ❌（sm75+ 才有） | fragment 加载手工 `LDS.128` + swizzle |
| `cp.async` | ✅ | ❌（sm80+ 才有） | gmem→smem 走 LDG→STS，靠多路并行发射（MLP）重叠 |
| `wmma.m16n16k16` | ✅ | ✅ | v9 核心武器（PTX 内联，展开为 16×HMMA.884） |
| `mma.m8n8k4` | ✅ | ✅ | v2–v8.1 使用（cute `SM70_8x8x4_F32F16F16F32_TN` atom） |
| bf16 | ✅ | ❌ | fp16 only |
| smem/CTA | 164KB | 96KB（默认 48KB，超限需 opt-in） | d=128 必须 opt-in 96KB |
| fp16 峰值 | 312 TF | ≈125 TF 理论 / **84.2 TF cuBLAS 实测** | 参照系 |

> **核心结论（后被反复验证）**：sm70 没有 Ampere 及以后的一切数据搬运加速器（ldmatrix / cp.async / TMA），**数据通路必须全手工搭**（swizzle + 向量化 LDS + gmem 多路发射）；但 **WMMA m16n16k16 从第一代 Tensor Core 就存在**，计算通路可以直接用最宽的 MMA 指令。"手工通路 + 宽指令"这个组合贯穿了整个优化史——前半程（v3–v8.1）我们在手工通路上精进，后半程（v9）才动计算指令。

### 2.2 性能参照系（全部在本机实测，shape 4,16,16,2048,2048,128）

| 参照 | TFLOPS | 说明 |
|---|---:|---|
| cuBLAS fp16 GEMM（4096³） | **84.2** | 硬件实用峰值 |
| SDPA mem_efficient 后端（无 mask） | 33.5 | torch 2.0.1 在 sm70 上 flash 后端不可用（需 sm80+），无 mask 即走此路径 |
| SDPA math 后端 | 30.7 | 带 additive mask 时的唯一官方 SDPA 路径 |
| fa-v100 开源参考（无 mask） | 19.5 | 无任意 mask 能力，仅作性能参照 |

（注：后文 benchmark 输出中的 "SDPA flash" 列在 V100 上实际是 mem_efficient 后端。）

### 2.3 参考实现（flash-attention-v100）调研结论

**借鉴**：WMMA m16n16k16 的 fragment 复用（一条指令的 A/B fragment 供 16 条 HMMA 用）、统一 XOR swizzle（16B 粒度、行基）、softmax 零 atomic。
**不借鉴**：O 累加器放 smem 做 RMW（压寄存器的代价，我们寄存器 acc_o 更快）、smem 占用大（d=64 72.5KB / d=128 84KB，vs 我们 v7 的 32.5/56.5KB）、不支持任意 additive mask。

---

## 3. 优化历程时间线

### 版本里程碑总览（shape 4,16,16,2048,2048；v6 起为带 mask benchmark）

| 版本 | 核心变化 | d=64 耗时/TF | d=128 耗时/TF | vs 上一版 |
|---|---|---|---|---|
| v2 初始 | m8n8k4 + plain smem + 8–9 sync/iter + 跨 8 warp atomic 归约 | （小 shape 4–8 TF） | （小 shape 4–8 TF） | — |
| v3 | Swizzle<3,3,3> + 行映射 softmax | 行映射方案错误，正确性未过 | | |
| v4 | MMA 线程直接管理 softmax（acc fragment 上做归约） | 正确性全过，性能未单独记录 | | |
| v5 | + P/Mask smem 时间复用 | 28723µs / 2.4 | 47446µs / 2.9 | 基线* |
| v6 | sync 6→4 + V load 重叠 + Q-in-regs | 8634µs / 8.0 | 12622µs / 10.9 | **3.3–3.8x** |
| v7 | + K double buffer + Q 常驻寄存器 | 5512µs / 12.5 | 8679µs / 15.8 | **1.4–1.6x** |
| v8.1 | 死代码删除 + spill 消除 + partial-l 寄存器化 + V load 提前 | 5131µs / 13.9 | 7725µs / 17.9 | 1.07–1.18x |
| v9 | WMMA m16n16k16 + v9 swizzle + 4×4 warp grid | 4029µs / 17.1 | 5695µs / 24.1 | **1.27–1.36x** |
| v9.1 | + Q A-fragment 常驻寄存器（仅 d=64） | 3974µs / 17.3（峰值 18.0） | 5686µs / 24.2 | d=64 +1–6% |

\* v5 是首次完整 V100 benchmark。各版本数字混合了两种口径：benchmark_sm70.py（带 10% 随机 -inf mask）与开发期临时对拍脚本（zero additive mask ≡ 无 mask，用于与 fa-v100 对比）——两种口径实测差异 <1%，三方对比详见表 4.3。

### Phase 1（v3）：Swizzle smem——搬运方向正确，行映射 softmax 全军覆没

**问题**：plain smem 的 bank conflict 灾难——理论分析 A/B fragment LDS 16-way、C-fragment store 64-way（fp16 行宽 128B = bank 全周期，同列访问全撞同一 bank）。

**实验**（host 端 cute layout 分析）：引入 `Swizzle<3,3,3>`（XOR-8，(8,64) atom）后 A/B 降到 4-way、C 降到 16-way，且 16B 向量化保持（swizzle 以 16B 为粒度，LDS.128/STS.128 可用）。选 `Swizzle<3,3,3>` 与 sm80 成熟模式一致，cute 的向量化分析对它最友好。

**失败**：同期的"行映射 softmax"（`tidx/8` 分行 + warp 内 `__shfl_xor(1|2|4)` 归约、零 atomic）**完全错误**——MMA 线程持有的 S 行由 atom 的 CLayout 决定，与 `tidx/8` 的行划分毫无关系。全部测试误差 0.3–2.3。

> 教训：**线程-数据映射不能靠直觉，必须先用实验 dump 出来**。cute atom 的 CLayout 是嵌套 layout，手推极易出错。这次失败直接催生了下一阶段的 fragment 映射实验方法论。

### Phase 2（v4）：MMA 线程直接管理 softmax——fragment 映射实验先行

**实验**：dump `SM70_8x8_32b` 的 CLayout，实测每线程 8 个 float = **2 行 × 4 列**，元素 {0,1,4,5}→row0、{2,3,6,7}→row1。基于这个映射重写 softmax：

- 归约链：线程内 4 列 → warp 内 `__shfl_xor(2)`（同 2 行的 2 线程）→ 跨 warp smem `atomicMax`/`atomicAdd`（每行跨 4 warp）
- m/l 状态直接放 MMA 线程寄存器，acc_s → exp → fp16 → sP 一条龙，不再有独立的 softmax 线程和 sS/sSoftmax smem 中转

**这一阶段修了四个数值 bug**（每一个都是通向正确的必经之路）：

1. **float atomicMax 负数错误**：softmax 分数可以为负，直接用 unsigned atomicMax 会选错最大值。修复：key 变换（正 float → `bits^0x80000000`，负 float → `~bits`），使 unsigned 比较序与 float 序一致
2. **sRowMax 每轮初始化为 -inf 而非 row_m**：导致 m_global 只含当前 block 的 max，丢失 online softmax 的历史 max，rescale factor 全错
3. **QK 溢出 +inf 与 mask -inf 相加产生 NaN**：修复为 mask=-inf 时直接置 s=-inf，跳过加法
4. **mask 的 scale 语义**：正确语义是 `softmax(S·scale + mask)`，kernel 必须把 mask 乘 `1/scale` 预还原后再加到未缩放的 QK^T 上

V100 首跑：18 项正确性测试全过（zero/neg mask、causal、随机 mask、odd shape、GQA）。

### Phase 3（v5）：P/Mask smem 时间复用 + warp 数实验（否决 8-warp）

- **P/Mask 时间复用**：mask 在 QK^T+mask 加法后就死了，sP 覆写同一区域——省 8KB smem，d=64 从 40.5→32.5KB（默认 48KB 限额内可 2 CTA/SM）
- **warp 数实验**：`AtomLayout<4,8,1>`（256 线程，8 warps）变体被实测否决——d=128 的 occupancy 瓶颈是 smem 而非线程数，减 warp 不改善占用，反而每 warp 承担双倍归约工作、跨 warp atomic 竞争加剧。保持 512 线程（16 warps）

**首次完整 V100 benchmark 暴露关键问题**：小序列（512²）尚可（5.8–8.5 TF），**大序列（2048²）灾难性退化到 2.4–2.9 TF**（SDPA+mask 的 0.14–0.23x），GB/s 只有 1–5——不是带宽问题，是每 n_block 迭代的固定开销在长序列上线性放大。

**诊断**：每 iter 6 次 `__syncthreads`（含 softmax 内部的 init/atomic 归约同步）+ 每轮 2 组跨 warp smem atomic + V load 串行在 softmax 之后。

### Phase 4（v6）：同步结构优化——大序列场景的 3.3–3.8x

三板斧（全部围绕"削减每 iter 固定开销"）：

1. **V load 提前**：从 softmax 之后挪到 QK^T 完成后，gmem 延迟与 mask 加法/softmax 计算重叠
2. **sync 6→4**：合并 sRowMax/sRowSum 的初始化同步；合并 atomicMax 完成同步与 atomicAdd 初始化同步
3. **Q 进寄存器**：消除每轮对 sQ 的读依赖

```
大序列 4,16,16,2048,2048:   d64  28723µs →  8634µs (3.3x)   d128  47446µs → 12622µs (3.8x)
```

> 教训：**老硬件上同步成本是第一杀手**。512 线程 CTA 的一次全同步开销不低，长序列（n_block 数百次迭代）把它线性放大成主导项。sync 计数应该和 smem/寄存器一样被当作一等资源记账。

### Phase 5（v7）：Q-in-regs + K double buffer——1.4–1.6x

**诊断**（读 `gemm()` 结构）：主循环没有 pipeline——每 iter 的 smem→register fragment copy 完成后才开始 mma，两者串行；K smem 单 buffer，当前轮计算无法与下一轮 K load 重叠。

**方案**：

- **Q 常驻寄存器**：prologue 把 Q 先落到 sK1（借用 K 的第二 buffer 空间），sync 后拷入寄存器 fragment，主循环零 Q LDS（同时腾出 sQ 空间）
- **K smem double buffer**：sK0/sK1 轮换，当前轮 QK^T 计算与下一轮 K 的 gmem→smem 重叠

**结果**：d=64 12.5 TF / d=128 15.8 TF（1.4–1.6x）。**d=64 首次超过 SDPA+mask（1.25x）**，d=128 达到 SDPA+mask 的 0.85x。

### Phase 6（v8/v8.1）：SASS 驱动的清理——死代码、spill、softmax 通信

诊断工具升级为 **SASS 静态统计 + ptxas -v**（`cuobjdump` dump 后按 opcode 计数），发现了四个问题：

1. **v7 死代码**：pre-exp sP 的 smem 写入无任何读者（softmax 全在寄存器做），白白浪费 8KB smem 写流量 + 1 次 sync
2. **d=128 寄存器溢出**：REG:128 + **STACK:152**（160B spill stores / 196B spill loads）——v7 的 K 双 buffer fragment 常驻加剧了压力
3. **softmax 通信**：每轮 2 组跨 warp atomic（max + add）+ 初始化 sync；fa-v100 为零
4. **LDS 指令密度**：我们 259/iter vs fa-v100 等效 76.5/iter——**3.4 倍差距**（详见 Phase 7）

SASS 指令画像（d=64，归一化到相同 MMA 工作量）：

| 指标 | 我们 v8 (1 iter) | fa-v100 (等效 1 iter) |
|---|---:|---:|
| HMMA（有效） | 128 | 128 |
| **LDS** | **259** | **76.5** |
| STS | 32 | 40 |
| BAR.SYNC | 5 | 5 |
| ATOM | 2 | 0 |

**v8**：删除死代码（省 8KB 写流量 + 1 sync）；K fragment 声明移入 if/else 分支块（生命周期不重叠 → 寄存器复用，STACK 152→112→0）。

**v8.1**：**partial-l 寄存器化**——关键数学：`l_global = Σ_thr partial_l`，而 rescale factor 对整行一致，因此各线程的 partial-l 可以**独立累积**、无需每轮跨线程求和，epilogue 一次 atomicAdd 归约即可。删掉每轮的 shuffle 归约 + atomicAdd + init 写；V load 再提前到 A 段（mask copy / K-prefetch / V-load **三路 gmem 并行发射**，与 QK^T 计算重叠）。

**结果**：d=128 17.9 TF（+11–18%），d=64 13.9 TF（+7–10%）；与 fa-v100 差距缩小到 0.76–0.96x；每轮 sync 4 次、主循环零 atomicAdd。

### Phase 7（v9）：WMMA m16n16k16——LDS 密度 3.4x 差距的终极解（d=128 +35%）

这是全项目收益最大的一轮重构，把 MMA 路径从 cute `m8n8k4` TiledMma 整体切换到手工 PTX 的 `wmma.mma.sync.aligned.m16n16k16.f32.f32`。

#### 7.1 动机与决策依据

`m8n8k4` 每 K=4 步就要 load fragment（窄 32/64bit 访问），259 条 LDS/iter 对 192 条 HMMA/iter——**LDS 成了发射瓶颈**。WMMA 一条指令复用 fragment 整整 16-K，且 A row / B col 加载全部变成 2×LDS.128、B row 变成 4×LDS.64。

#### 7.2 同构性发现（决定可行性的关键）

从 fa-v100 的 PTX 文档级分析 + 我们的实验确认：WMMA acc 布局每线程 **2 行 × 4 列**、元素 {0,1,4,5}→row0——**与 m8n8k4 的 acc 布局完全同构**。这意味着整套 softmax / partial-l / epilogue 逻辑可以**零改动迁移**，只换 MMA 发射和 fragment 加载。

#### 7.3 实验四部曲（V100 实测）

1. **Part A**：v9 swizzle 公式（`v9_smem_off`）与 cute layout 的一致性对拍
2. **Part B**：WMMA fragment 映射实测（identity-B probe 逐 lane dump）——A row-major 每 lane 持一整行 16 halves、B col-major 每 lane 持一整列、acc 2 行×4 列，与理论完全一致
3. **Part C**：完整 64×64 QK^T+PV 管线 vs CPU 参考（d=64/128 全 PASS）
4. **Part D**：新旧 swizzle 整管线 A/B 计时——**3215 vs 4170 cycles/iter（-23%）**，bank conflict 理论被直接证实

> 插曲：Part C 首轮"失败"实为**测试数据的 half 舍入假象**（2049→2048 精度截断被当成了布局错误）。教训：验证布局的测试数据必须选 half 精确可表示的值（如 `r*32+c`），否则会把数值误差误诊为映射 bug。

#### 7.4 v9 swizzle：行 bits {0,1,3} 参与 chunk XOR

旧 `Swizzle<3,3,3>`（(8,64) atom，行 bits {0,1,2} 参与 XOR）在 WMMA 的 lane quarter（16 行）内会让 r 与 r+8 落到同一 pattern → 2-way bank conflict。v9 改用 (16,64) atom、**行 bits {0,1,3}** 参与 chunk XOR，quarter 内 16 行互异 → 零 conflict（即 Part D 的 23% 收益来源）。

#### 7.5 softmax 4x bug（本轮最重要的捕获）

设计时假设"每 S 行只属于一个 warp"（4×4 warp grid、每 warp 独占 16×16 tile），于是把跨 warp 归约全删了，softmax 全 warp-local。**实际结构是：每行的 64 列由 4 个 warp_n 分算**——warp-local 归约只覆盖了 1/4 的列。

诊断手法教科书式干净：受控输入（K=V=identity）下输出**恰好是正确值的 4 倍**、每个 16 列 tile 的行和各 = 1.0——一步锁定"四个 warp 各自独立归一化"。修复：恢复跨 warp 归约（主循环 atomicMax 到 sRowMax + epilogue atomicAdd 到 sRowSum，从 v8.1 移植回来）。

> 教训：**结构性假设（数据归属哪个线程/warp）必须画图 + 受控实验双重验证**，"看起来每行在一个 warp 里"的直觉在 warp grid 是二维的时候会骗人。

#### 7.6 smem 重构

Q 回 smem（WMMA A-fragment 要求 lane 持整行，prologue 一次落地）；sP/sMask 时间复用沿用；softmax 交换区仅剩 sRowMax+sRowSum（0.5KB）：

```
d=64:  sQ 8KB + sK×2 16KB + sV 8KB + sP/Mask 8KB = 40.5KB（< 48KB 默认上限）
d=128: 16 + 32 + 16 + 8 = 72.5KB（opt-in 96KB）
```

#### 7.7 JIT 缓存陷阱（与 sm120 文档 8.4-2 同款）

v9 首次正确性测试全 FAIL，一度以为 kernel 写错——实际是 `custom_ops` 的 fast-load 路径加载了**旧 .so**（不校验源码 hash）。删 `.so` 强制重编后新 kernel 才真正上机。**改 C++ 源码后必须确认重编译真实发生**。

#### 7.8 结果

| shape (无 mask 等价三方对比) | fa-v100 | v9.1 | v9.1/v100 |
|---|---|---|---|
| 4,16,16,2048,2048,128 | 19.5 TF | **24.2 TF** | **1.24x** |
| 1,8,8,8192,8192,128 | 18.8 TF | **24.1 TF** | **1.28x** |
| 32,16,16,1024,1024,128 | 18.5 TF | **24.1 TF** | **1.31x** |
| 4,16,16,2048,2048,64 | 17.5 TF | 17.1 TF | 0.98x |

- d=128 相对 v8.1 **+35%**（17.9→24.2 TF），全面反超开源参考 24–31%
- LDS 静态数：259 → 86（d=128）/ 51（d=64）；**零 spill**
- d=64 持平但未反超——差距不在指令密度（已拉平），在 fa-v100 的 BLOCK_N=128 tile 结构（超长序列仍 0.95x，属已知结构性 gap）

### Phase 8（v9.1）：Q A-fragment 常驻寄存器——d=64 反超开源参考

QK^T 的 A-fragment（Q）是**环路不变量**：d=64 时 4 个 fragment × 8 regs = 32 regs，外提到 prologue 后每 iter 省 8 条 LDS.128。

- **d=64 启用**：REG 75→98（512×98 < 64K，无 spill 无降占用）
- **d=128 跳过**：需 +64 regs 必然 spill/降 occupancy，`if constexpr` 门控留在环内

首版把加载放在 prologue copy **之前**（sQ 未就绪）导致 d=64 全 NaN——**寄存器化外提必须重新审视数据就绪时序**（load 与 sync 的相对位置变了）。修复后：

```
32,16,16,1024,1024,64:  17.0 → 18.0 TF（1.07x，反超 fa-v100）
1,16,16,1024,1024,64:   12.1 → 12.8 TF（1.06x）
```

收益集中在 tile 数少、每 CTA 迭代少的 shape（LDS 占比高）；d=128 路径完全不受影响。

---

## 4. 最终性能全景

### 4.1 指令与资源画像（v8.1 → v9.1）

| 指标 | v8.1 | v9.1 |
|---|---|---|
| LDS 静态数/iter（d=64 / d=128） | 259 / 259 | **43 / 86** |
| REG（d=64 / d=128） | 75 / 128（d=128 残留 112B spill） | 98 / 128（**零 spill**） |
| smem（d=64 / d=128） | 32.5 / 56.5 KB | 40.5 / 72.5 KB |
| sync/iter | 4 | 4（sync1/syncM/sync5/sync6） |
| 主循环 atomicAdd | 0 | 0（atomicMax 2/lane + epilogue 一次 atomicAdd 归约 l） |

（v9 用 +8–16KB smem 换 LDS 密度 -67–83%——这笔账只有对着逐版本资源账本才看得清。）

### 4.2 标准场景（带 10% 随机 -inf mask，vs SDPA+mask 同功能公平对照）

| shape (B,H,Hk,Sq,Sk,d) | custom | custom TF | SDPA+mask | 加速比 |
|---|---:|---:|---:|---:|
| 1,8,8,512,512,64 | 49.7µs | 10.8 | 84.0µs | **1.69x** |
| 4,16,16,1024,1024,64 | 1001.0µs | 17.2 | 1706.8µs | **1.71x** |
| 4,16,16,2048,2048,64 | 3974.1µs | 17.3 | 6900.8µs | **1.74x** |
| 4,16,4,2048,2048,64 (GQA) | 4013.3µs | 17.1 | 7087.9µs | **1.77x** |
| 1,8,8,512,512,128 | 65.5µs | 16.4 | 93.0µs | **1.42x** |
| 4,16,16,1024,1024,128 | 1446.0µs | 23.8 | 1833.5µs | **1.27x** |
| 4,16,16,2048,2048,128 | 5686.4µs | 24.2 | 7315.9µs | **1.29x** |
| 4,16,4,2048,2048,128 (GQA) | 5712.9µs | 24.1 | 7721.1µs | **1.35x** |
| 1,8,8,2048,2048,128 | 878.1µs | 19.6 | 925.8µs | **1.05x** |

全部大 grid shape 1.05–1.77x；d=128 峰值 24.2 TF = cuBLAS 峰值的 29%。

### 4.3 三方对比（无 mask 等价：zero additive mask ≡ 无 mask）

| shape | fa-v100 | **ours v9.1** | v9.1/v100 | SDPA(mem_eff, 无 mask) |
|---|---:|---:|---:|---:|
| 1,16,16,1024,1024,64 | 12.1 TF | 12.8 TF | 1.06x | 17.7 TF |
| 32,16,16,1024,1024,64 | 16.9 TF | **18.0 TF** | **1.07x** | — |
| 1,8,8,8192,8192,64 | 18.0 TF | 17.1 TF | 0.95x | — |
| 1,16,16,1024,1024,128 | 15.5 TF | **19.1 TF** | **1.23x** | 28.1 TF |
| 4,16,16,2048,2048,128 | 19.5 TF | **24.2 TF** | **1.24x** | 35.7 TF |
| 1,8,8,8192,8192,128 | 18.8 TF | **24.1 TF** | **1.28x** | 35.4 TF |
| 32,16,16,1024,1024,128 | 18.5 TF | **24.1 TF** | **1.31x** | — |

**d=128 全 shape 反超开源参考 23–31%**（且我们带任意 mask 能力）；d=64 持平（1024² grid 反超，8192² 超长序列 0.95x——fa-v100 的 BLOCK_N=128 结构优势，属已知 gap）。与 SDPA mem_efficient（无 mask、不支持任意 mask）相比 0.72x，但该后端无法承接业务需求。

### 4.4 小 grid / decode 场景（已知短板）

| shape | custom | SDPA+mask | 加速比 |
|---|---:|---:|---:|
| 1,2,1,128,1024,128 (GQA) | 114.8µs | 348.8µs | **3.04x** |
| 1,1,1,128,2048,128 | 221.5µs | 41.6µs | 0.19x |
| 1,1,1,128,8192,128（vs fa-v100 615µs） | 905.7µs | — | 0.68x |

单 CTA 串行 K 维导致 grid<SM 数时并行度枯竭（与 sm120 Phase 7 之前同款问题）。**split-KV 尚未在 sm70 路径实现**——sm120 文档中的完整方案（cost model + combine 并行化 + Split-M）可直接移植，这是后续最明确的方向。

### 4.5 精度

- 全部正确性测试 PASS：zero/neg additive mask、causal、随机 mask、odd shape（sq=100/sk=200 等）、GQA
- vs fa-v100 输出 diff ≤ 2.44e-04（fp16 正常水平）；vs SDPA allclose 全过

---

## 5. 经验总结：可复用的方法论

1. **老硬件先补指令集课**：sm70 缺 ldmatrix/cp.async 决定了数据通路全手工（swizzle + 向量化 LDS + gmem 多路发射）；但 WMMA 宽指令从 Volta 就有。"手工通路 + 宽指令"的正确组合在动手前就要想清楚。
2. **SASS 静态统计先于 profile**：`cuobjdump` 按 opcode 计数（LDS 259 vs 76.5）在没跑任何 profiler 的情况下就锁定了核心瓶颈——静态指令密度是结构性问题的直接证据，而且零成本。
3. **fragment 映射实验先行**：两次关键迁移（softmax 上 acc fragment、m8n8k4→WMMA）都靠先 dump 映射再写代码；"acc 布局同构"的发现让整条 softmax/partial-l/epilogue 逻辑零改动迁移，v9 的工程风险因此可控。
4. **结构性假设要受控实验验证**："S 行属于一个 warp"的错误假设造成 4x bug；K=V=identity 受控输入 + "输出恰为 4 倍、每 tile 行和=1"一步定位。**数据归属（谁算哪些行列）是 kernel 正确性的第一公理，必须画图验证。**
5. **数学等价变换消除通信**：partial-l 寄存器化的依据是"rescale factor 行全局一致 → partial 可独立累积"——比"减少 atomic 次数"更彻底的优化是**证明这步通信在数学上不必要**。
6. **寄存器预算按 head dim 分别决策**：同一个 Q-寄存器化优化，d=64（+32 regs）有益、d=128（+64 regs）必然 spill——`if constexpr` 分配置，不做一刀切。
7. **版本演进要记资源账本**：每版记 smem / sync / REG+spill / LDS 四件套。v9 用 +8–16KB smem 换 LDS -67–83%、v9.1 用 +23 regs 换 -8 条 LDS.128，这些 tradeoff 只有对着账本才能持续做对。
8. **回归环境要防"假阴性"**：JIT fast-load 不校验源码 hash（新旧 .so 混淆）、half 测试数据的舍入假象、`-inf` mask 对 scale 语义 bug 的完全隐形（0/-inf mask 下不可见，需有限值随机 mask 才能抓到）——三条都在本项目真实踩过。
9. **手工逆向的 fragment 映射是架构绑定的**：v9 手工 WMMA PTX 依赖的 lane 映射系 sm70 实测逆向，而 WMMA 内部布局**无跨架构契约**——实测 sm89 的 A-fragment 映射与 sm70 不同（lane 持交错行列块而非整行）。因此 `FA_FORCE_SM70` 旁路在 v8.1 前（m8n8k4，布局跨架构一致）可做交叉验证，对 v9+ 在非 V100 架构上会**静默产出错误结果**。手工布局 kernel 的正确性验证必须在与目标架构一致的真机上做。

### 关键文件索引

| 文件 | 内容 |
|---|---|
| `csrc/fa/sm70/fa_fwd_sm70.h` | sm70 kernel 全量实现（含逐版本设计注释、fragment 映射与 swizzle 公式） |
| `tests/test_sm70_correctness.py` | 正确性回归（zero/neg/causal/随机 mask、odd shape、GQA） |
| `benchmark/benchmark_sm70.py` | 带 mask 标准/小 grid benchmark（custom vs SDPA+mask vs SDPA 无 mask 参照） |

> 文中引用的其余实验脚本（cute layout 分析、WMMA fragment 映射 dump、swizzle A/B 计时等）
> 为开发期一次性验证工具，未随仓库发布；其结论已固化在 kernel 文件头注释与本文档中。
