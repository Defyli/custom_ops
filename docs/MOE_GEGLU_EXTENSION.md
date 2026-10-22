# fuse_moe GEGLU 扩展开发记录：GELU-erf 与 staged 数值模式

> **状态（2026-10-21）**：通用三矩阵 GLU 前向已支持 SiLU / exact-erf GELU 与 `fused` / `staged` 舍入契约。完成此前本地 SM89 基线后，数值模式和文档已中性化；Python 3.12 / PyTorch 2.11 / CUDA 12.8 的 SM89、SM120 远程环境已探测，新增运行证据单独记录。**历史表不是此次源码重测，也不是全架构 production-ready 声明**。
> **范围**：`custom_ops` 仓库内 `fuse_moe` 全 GPU 路径、`fuse_moe_offload` 路径、TileLang `fuse_moe` 后端、公开 Python 接口与 offload 部署配置/制品版本。
> **不在本文范围**：外部模型、router、权重来源与恢复、serving 系统、反向传播。本文只定义通用张量布局、数学运算、逐阶段舍入及构建/部署契约，不绑定调用方用途、私有类或模型拓扑。
> **证据约定**：`F` = 静态代码事实；`E` = 实际环境探测；`D` = 实施决策；`R` = 有命令、退出状态及原始文件的运行证据；`P` / BLOCKED = 未完成或无环境。本文不把 `F` 或 CPU mock 契约测试当作 GPU 运行结果。

## 1. 目标与边界

### 1.1 要解决的问题

原 `fuse_moe` 的专家非线性固定为 SiLU（SwiGLU）。本扩展在相同三矩阵门控结构上增加 **GELU exact erf（GEGLU）** 与独立的逐阶段舍入选择：

\[
g = W_g x,\qquad u = W_u x,\qquad h = \operatorname{gelu}_{\mathrm{erf}}(g)\odot u,\qquad y = W_d h
\]

本扩展要让同一条融合 MoE 流水线在**不改变路由、权重布局、输出契约**的前提下，选择：

- `expert_activation="silu"`（旧默认，SwiGLU）或 `"gelu"`（GELU-erf，GEGLU）；
- `numerical_mode="fused"`（旧默认，gate/up 激活乘积融合后一次舍入；down 和最终输出仍有各自的 dtype 边界）或 `"staged"`（中间结果逐级 round 到输出 dtype，再以 FP32 累加）。

`activation_mode` 继续只表示 offload 的缓冲/累加策略，**不是**激活函数选择器。

### 1.2 与普通 GELU MLP、旧 SwiGLU 的区别

| 路径 | 数学形式 | 线性矩阵数 | 专家非线性 | 与本仓库的关系 |
|---|---|---:|---|---|
| 普通两矩阵 GELU MLP | \(y = W_2\,\operatorname{gelu}(W_1 x)\) | 2 | GELU | **不是**本扩展目标；没有 gate/up 配对和逐元素乘积 |
| 三矩阵 GEGLU | \(y = W_d(\operatorname{gelu}_{\mathrm{erf}}(W_g x)\odot W_u x)\) | 3 | GELU exact erf | 本扩展承接的语义 |
| 旧 `fuse_moe` 默认 | \(y = W_d(\operatorname{silu}(W_g x)\odot W_u x)\) | 3 | SiLU | `expert_activation="silu"`，默认不变 |

需要区分三个计数：

- 普通 MLP 是**两个线性矩阵**，只有一个上投影，不能靠切换激活函数变成 GEGLU；
- 本仓库 `fuse_moe` 的入参是**两个权重 Tensor** `gate_up_weight (E,2I,H)` + `down_weight (E,H,I)`，但数学上是 **gate、up、down 三个矩阵**，gate/up 被拼在同一个 Tensor 里；
- 默认 cp.async 路径有**两次 GEMM kernel launch**：gemm1 在同一 K 循环中以同一 `tXr` 分别执行 gate/up 的 `cute::gemm`，累加到 `tYr_g/tYr_u`，随后融合计算 `act(gate)·up`；gemm2 为 down。矩阵乘法算量没有省去，只合并 launch、共享 X 加载并避免 gate/up 中间落盘。

无论 `silu`/`gelu`、`fused`/`staged`，上述投影和 GEMM launch 数均相同。`staged` 只在 epilogue/归约改变 cast 边界；TMA 仍是 gate/up 拼接 GEMM + down GEMM，中间多一个独立逐元素 `act_mul`，不是第三次 GEMM。Offload 会按 wave 重复这两类计算，实际 launch 数随 wave 数变化，与数值模式无关。

因此：

- `expert_activation="gelu"` 只改变专家非线性，不会把普通 2 矩阵 MLP 变成 GEGLU；
- 两矩阵 MLP 与三矩阵 GLU 是不同运算，不能以复制权重等方式静默互换；本算子仅接受第 4 节明确的 gate/up/down 布局。

### 1.3 本轮不做

- 不实现外部模型类、router、top-k 选择或权重恢复。
- 不实现反向传播；`numerical_mode="staged"` 只是**前向舍入语义**，不是 autograd。
- 不把 GELU 的 tanh 近似作为 exact erf 的静默回退。
- 不修改独立 `swiglu` 算子（`csrc/swiglu/`、`tilelang_ops/_swiglu.py`）。
- 不凭本地/历史结果外推架构或工具链支持；远程验证按具体 CUDA/PyTorch/SM 独立记录。

## 2. 代码事实与 2026-10-21 环境

### 2.1 原本地基线环境快照（E）

| 项目 | 记录值 | 说明 |
|---|---|---|
| PyTorch | `2.6.0a0` | 2026-10-21 实际探测 |
| Python | `3.9.16` | 同上 |
| PyTorch CUDA / nvcc | `11.8` / `V11.8.89` | nvcc 与驱动版本分开记录 |
| GPU | NVIDIA GeForce RTX 4090 D，24564 MiB | `sm89`；开始时另有 6097 MiB 被既有进程使用 |
| NVIDIA driver | `535.183.01` | 不是 nvcc 版本 |
| 编译器 / ABI | devtoolset-10 GCC/G++ `10.2.1`，`_GLIBCXX_USE_CXX11_ABI=0` | 构建 sidecar 保存实际 flags/toolchain |
| CPU threads / MAX_JOBS | `4` / `2` | 未更改既有服务/GPU 时钟配置 |
| TileLang | 未安装 | 只验证无 erf/输入校验 CPU 契约，不能宣称 GPU kernel 已编译 |
| 本地 SM120 / PyTorch 2.11 | 缺少 | 必须在对应机器独立验收 |
本轮制品和原始证据位于 `artifacts/geglu_sm89_20261021/`。旧 `artifacts/geglu_sm89/` 的历史结果没有覆盖、没有冒充本轮多次重复验收。正常性能前后 GPU 状态记录见 `gpu_before_performance.csv` / `gpu_after_performance.csv`；前后快照不是全过程锁频证据。

### 2.2 工作树改动与职责（F）

2026-10-19 已存在大批未提交扩展改动，本轮在其基础上增量补齐，保留调用方文件、未提交 git。下表为核心路径静态事实；运行状态独立维护在第 5 节：

| 文件 | 硬事实 |
|---|---|
| `csrc/fuse_moe/common/fuse_moe_params.h` | `FMOE_params` 新增 `expert_activation`（0/1）、`numerical_mode`（0/1），由 op/offload 层填写 |
| `csrc/fuse_moe/common/fuse_moe_utils.cuh` | 新增 `gelu_erf(float)`：`0.5*x*(1+erff(x*0.70710678118654752440f))`；`expert_act<kActivation>` 编译期选择 SiLU/GELU |
| `csrc/fuse_moe/common/moe_kernels.cuh` | `act_mul_kernel` / `reduce_kernel` 增加编译期数值模式；host launcher 做 4 路/2 路模板分发 |
| `csrc/fuse_moe/sm89/group_gemm_sm89.cuh` | `group_gemm_gateup_kernel` 增加 `kActivation`、`kRoundIntermediate`；`group_gemm_gateup_fused_async` 做 4 路模板分发；down 的 fused-accum 路径增加 `kRoundProduct` |
| `csrc/fuse_moe/fuse_moe_launch.h` | 校验两个枚举；sm120 默认仍复用 cp.async 家族；`FUSE_MOE_TMA=1` 的 TMA 路径单独调用 `act_mul_async` |
| `csrc/fuse_moe/fuse_moe_op.cu` / `.h` | 两个入口增加尾参；CUDA 入口校验 0/1，CPU 入口仍明确拒绝非 CUDA；workspace 逻辑未因激活改变 |
| `csrc/fuse_moe/fuse_moe_offload.cpp` / `.h` | offload context 固化 `expert_activation/numerical_mode`；create 增加两个尾参；配置校验支持 v1/v2 |
| `csrc/fuse_moe/fuse_moe_bindings.cpp` | 注册 schema 增加两个默认尾参 |
| `recsys.py` | 公开字符串枚举与 int 编码；`RecsysOps.fuse_moe` 增加关键字参数；fake 实现同步 |
| `moe_offload.py` | `OffloadConfig`、`PrefillMoERuntime`、create 调用、v1/v2 配置、format1/2 归档同步 |
| `tilelang_ops/__init__.py` / `_fuse_moe.py` | TileLang 前端与 gemm1/gemm2 编译期特化；erf 能力探测与显式失败 |
| `tests/test_fuse_moe.py` / `test_fuse_moe_offload.py` / `test_fuse_moe_offload_aoti.py` | 四模式/dtype/边界/重复运行、严格标量舍入见证、v1/v2/归档/AOTI；新增 p50/p99/max、FP64 小样本分阶段参考、公开 JSON/CPU 契约入口 |

`csrc/swiglu/` 与 `tilelang_ops/_swiglu.py` **不在本次 diff 中**。

#### 2.2.1 实施顺序（静态梳理）

1. 在 `FMOE_params` 引入 `expert_activation/numerical_mode`，默认 0，先保证旧路径不需要改调用点。
2. 在 CUDA epilogue/act_mul/reduce 增加 exact GELU 与逐级 round，并在 host 侧按 4 种组合分发模板，保留 legacy 特化。
3. 在 offload context、create schema、配置校验中加入两个字段；默认仍输出 v1，非默认输出 v2。
4. 在 Python 公开层加入严格字符串校验、schema、fake，并让 offload 配置/归档/form2 贯通。
5. 在 TileLang 前端加入编译期特化与 erf 能力探测；无 erf 时显式失败。
6. 在测试侧加入四组合、masked 路由、v1/v2 配置与 AOTI GEGLU 用例；2026-10-21 在 SM89 实际执行，结果见第 5 节。
7. 补 TileLang count/route 的非负/上界守卫、连续/CUDA/空输入校验；无 erf 在分配/JIT 前失败。
8. 补验收证据防线：四模式性能分组键含 dtype/激活/舍入；GPU skipped 不得作为 baseline passed；诊断耗时不得进入正常 ABBA 汇总。

### 2.3 公开 API 与线格式（F）

Python 公开接口默认 `expert_activation="silu"`、`numerical_mode="fused"`；两个新参数只能关键字传入。GEGLU/staged 调用示例：

```python
output = ops.fuse_moe(
    x, gate_up_weight, down_weight, topk_ids, topk_scale,
    backend="cuda",
    expert_activation="gelu",
    numerical_mode="staged",
)
```

- 字符串到 native int 的映射：`{"silu": 0, "gelu": 1}`、`{"fused": 0, "staged": 1}`。
- Python 侧只接受精确字符串；`bool`、`int`、`None`、大小写变体均拒绝，且在触发分组加载/编译前完成校验。
- native schema 的尾参默认值为 0，因此 native 层旧的 `fuse_moe(x, w1, w2, ids, scale)` 调用仍表示 `silu/fused`；但新的 Python 封装即使使用默认值也会显式传 0/0，不能配旧 `.so`。
- offload create 的 schema 同样增加 `expert_activation=0, numerical_mode=0` 两个默认尾参。

### 2.4 CUDA 计算路径（F）

- `group_gemm_gateup_fused_async` 在 host 侧按 `expert_activation × numerical_mode` 一次分发到 4 个模板特化，kernel 内没有逐元素动态分支。
- `expert_activation=1` 的 epilogue 使用 `gelu_erf`，即 `erff` exact 式；**没有 tanh 近似分支**。
- `numerical_mode=1` 时，gemm1 融合 epilogue 的舍入顺序是：

```text
gate 累加器 -> round Tout
up   累加器 -> round Tout
act(gate_rounded) -> round Tout
act_rounded * up_rounded（FP32 计算）-> round Tout，写 act_out
```

- `reduce_async` 在 `numerical_mode=1` 时对每个 weighted product 先 round 到 `Tout`，再以 FP32 累加，最后一次 round 写回；`numerical_mode=0` 保持旧的 FP32 加权累加。
- down 阶段的结果以 `Tout` 物化（`sC` 为 `Tout`）。fused 与 staged 的差异不在“down 是否落盘为 dtype”，而在 gate/up、act、乘积和 weighted product 是否逐级 round；offload 的 fused-accum 路径在 staged 下对 `sC * scale` 再 round 后 FP32 atomicAdd。
- sm120 默认引擎仍是 cp.async（`moe_launch_sm120_impl` 在未设置 `FUSE_MOE_TMA` 时直接复用 `moe_launch_sm8x_impl`）。
- 设置 `FUSE_MOE_TMA` 的 TMA 路径仍为两次 GEMM：gemm1 一次计算 `N=2I` 的 gate/up 拼接投影，再用独立 `act_mul_async`。`fused` 的 `gate_up_out` 为 FP32，保留 gate/up 累加值直到激活乘积后才 round；`staged` 为输入 dtype，依次 round gate/up、激活和乘积。额外 FP32 workspace 是为履行同一数值契约，不是额外 GEMM，默认 cp.async 不分配它。
- offload 的 `moe_compute_sm8x` 直接读取 `FMOE_params` 中的两个字段，因此每个 offload wave 继承 create 时固化的激活/数值契约。

### 2.5 Offload 配置与制品（F）

配置向量：

- **v1**：`config[0]=1`，固定 23 个字段，之后是 resident expert IDs；v1 隐式表示 `silu/fused`。
- **v2**：`config[0]=2`，`0:23` 与 v1 相同，`config[23]=expert_activation`，`config[24]=numerical_mode`，resident IDs 从 `25` 开始；长度为 `25 + resident_count`。
- Python `OffloadConfig.layer_config()` 在 `silu/fused` 时输出 v1，任一非默认时输出 v2。
- Python `from_layer_configs()` 要求所有层使用同一版本、同一 `activation/numerical` 组合；v2 且两个 code 均为 0 被视为非 canonical 并拒绝。
- C++ `validate_deployment_config()` 按每条 record 的 `config[0]` 选择 23/25 头；v1 record 在 context 为 GEGLU/staged 时明确拒绝，避免“省略即兼容”的误用。
- Manifest 外层 framing 固定为版本 1：`[1, layer_count, layer_id, config_length, *config, ...]`；每条 record 内部可以是 v1 或 v2。
- CPU 归档 `_ExpertArchive.format_version`：无配置为 `1`，带配置为 `2`；`load_cpp()` 只接受 `[1]`/`[2]`，format 2 要求每个层存在 `offload_config_{layer_id}`。
- `activation_mode` 仍是 `route_buffer / fp32_accumulate / auto`，与 `expert_activation`、`numerical_mode` 完全独立；例如 `activation_mode="fp32_accumulate"` + `expert_activation="gelu"` 是合法组合。

### 2.6 TileLang 路径（F）

- `_moe_gemm1_factory` 把 `expert_activation`、`numerical_mode` 作为编译期特化参数；`_moe_gemm2_factory` 把 `numerical_mode` 作为编译期特化参数。
- GELU 能力探测顺序：
  1. `T.erf` 可用则用 intrinsic；
  2. 否则尝试 `T.call_extern("float32", "erff", ...)`；
  3. 两者都不可用时抛出 `NotImplementedError`，明确说明 exact GELU 无法编译。
- **没有** tanh 近似回退；不会静默改变数学语义。`_exact_gelu_uses_erf()` 在前端分配/JIT 前检查能力；CPU 契约用 identity JIT decorator 验证拒绝策略，不模拟 GPU 成功执行。
- count/route 在访存前均检查 `0 <= e < E`，`-1` 的 NaN scale 整条忽略；连续性、CUDA、正 S/H 在 JIT 前检查。CUDA 12.8 远程测试已执行四模式、双 dtype 和 masked/all_masked，证据见第 10 节。
- 为保持旧默认数值路径，TileLang `fused` gemm2 的 down/加权保持 FP32，而 CUDA `fused` 先将 down 物化为 Tout；这是已知跨后端数值差异，不用改参考/容差来掩盖。
- `staged` 在 gemm1 中逐级 round gate/up、act、乘积；gemm2 的 down 与 weighted product 逐级 round 后先写回 FP32 fragment，再对全局输出做 atomicAdd。保持 cast 与带谓词原子目标分离，避免 SM120/TileLang 0.1.14 将目标局部化。N tile 必须整除对应 I/H；不符合的配置选择 64 列基线，配置副本不修改调用方 dict。
- kernel cache key 已包含 `expert_activation`/`numerical_mode`（gemm1）和 `numerical_mode`（gemm2），避免不同模式复用同一编译产物。
- 独立 `swiglu` 未改动；其 GELU 语义不在本扩展内。

### 2.7 调用方张量契约

本仓库只接收三矩阵 GLU 的 canonical 权重与显式路由，不从类名、模型配置或 checkpoint 键名推断 gate/up 语义。任何展平/转置源布局必须由调用方明确适配，普通两矩阵 MLP 不在本算子契约内。

- 权重 restore 与 router 仍由调用方负责；`fuse_moe` 只消费 canonical 权重、`topk_ids`、`topk_scale`。
- gate/up 各自形状为 `(E, I, H)`，按 **gate 在前、up 在后** 在 `dim=1` 拼接：

```python
gate_up_weight = torch.cat([gate_weight, up_weight], dim=1).contiguous()
```

- down 权重要求 `(E, H, I)`。调用方源权重若存储为 `(E, I, H)`，显式转置，而不是让算子猜测：

```python
down_weight = down_source.reshape(E, I, H).transpose(1, 2).contiguous()
```

- `topk_ids` 必须是 `int32`、连续；非负值在 `[0, E)`。`-1` 作为 masked 槽位由现有 count/reduce 路径跳过（新增测试已覆盖），但调用方不得依赖任意越界值。
- `topk_scale` 必须是 `float32`、连续。所有数值模式均应提供 float32 scale，**不要在调用前降精度**。
- 全 GPU 路径要求上述 Tensor 在 CUDA；offload 路径的专家权重是只读 CPU canonical Tensor，`x/ids/scale` 在 CUDA。
- dtype：`x`、`gate_up_weight`、`down_weight` 必须同为 bf16 或 fp16；输出与 `x` 同 dtype。

## 3. 实施决策

| 决策 | 选择 | 理由 | 状态 |
|---|---|---|---|
| 激活函数维度 | `expert_activation` 独立于 `activation_mode` | 前者是专家非线性，后者是 offload 缓冲/累加策略；混用会导致配置无法表达组合 | D/F |
| 默认值 | `silu` + `fused` | 保持旧调用语义与旧数值路径；新能力显式开启 | D/F |
| GELU 形式 | exact erf | 对齐 `F.gelu(approximate="none")`；tanh 近似不是同一函数 | D/F |
| 无 erf 的 TileLang 构建 | 明确失败 | 静默回退 tanh 会产生不可见的精度/语义变化 | D/F |
| 模板分发位置 | host 侧一次分发 4 个特化 | kernel 内无逐元素分支，避免热路径判断 | D/F |
| staged 的语义 | 运算边界逐阶段 round 到 `Tout` 后 FP32 累加 | 数值规则与用途无关；不承诺不同 GEMM/归约顺序 bitwise 一致 | D/F |
| 与 autograd 的关系 | 仅前向；不实现反向 | 数值模式只定义舍入边界，不增加梯度能力 | D/F |
| 配置版本 | 默认 v1；非默认 v2 | v1 保持 23 字段旧契约；v2 在 23/24 追加两个 code | D/F |
| 制品 | schema 改动后重编 `.so` 与 `.pt2` | 旧 `.so` 不认识新尾参；AOTI 生成代码与 schema 绑定 | D/F |
| TMA 路径 | gemm1 不融合 act，独立 `act_mul` | 复用现有 TMA 结构，避免为 GELU 分叉 GEMM 骨架 | D/F |
| 独立 `swiglu` | 不动 | 语义、布局和调用方都不同；避免把 GEGLU 改动扩散到无关算子 | D/F |
| 首版性能优化 | 不改 GEMM 主骨架，先复用现有 epilogue | 先建立正确性/性能基线，再决定是否为 erff 单独优化 | D |

### 3.1 为什么不做 tanh 近似

exact GELU 与 tanh 近似在尾部和中段都存在差异。若在 `erf` 不可用时静默替换：

- 调用方无法从 API 看出实际执行的是另一种函数；
- 数学参考、精度评估和跨后端对拍都会失去可比性；
- 该差异会被 GEMM 误差掩盖，难以在回归中发现。

因此 TileLang 的决策是：能拿到 `erf`/`erff` 就编译 exact GELU；拿不到就失败，并给出明确错误。

### 3.2 staged 舍入顺序

融合 CUDA 路径（cp.async gemm1）在 `numerical_mode=1` 时：

```text
gate_fp32, up_fp32
  -> gate_r = round_Tout(gate_fp32)
  -> up_r   = round_Tout(up_fp32)
  -> act_r  = round_Tout(gelu_erf(float(gate_r)))
  -> prod_r = round_Tout(float(act_r) * float(up_r))
  -> act_out = prod_r
down gemm
  -> down_r = round_Tout(down_fp32)
  -> weighted_r = round_Tout(float(down_r) * scale_fp32)
  -> FP32 累加（reduce 或 fused atomic）
  -> 最终 round_Tout 写出
```

TMA 在 `staged` 下的 gemm1 先把 gate/up 物化为 `Tout`，再由 `act_mul_async` 做：

```text
gate_r/up_r 已是 Tout
  -> act_r = round_Tout(gelu_erf(float(gate_r)))
  -> prod_r = round_Tout(float(act_r) * float(up_r))
```

TMA 的 `fused` 则将 gate/up 物化为 FP32，激活与乘积保持 FP32，乘积写出时才 round 到 `Tout`。该输出类型由 host 模板分发，不改变 TMA 输入 dtype、Tensor Core MMA 或 GEMM 数量。

TileLang 路径的 gemm1/gemm2 采用同样的逐级 round 规则，但 gemm2 使用 FP32 atomicAdd，累加顺序与 CUDA reduce 不同。

**结论边界**：`staged` 表示上述确定的运算边界逐阶段舍入规则，而不是某种调用方用途。不同 GEMM 累加顺序或 atomic 顺序可产生末位差异，不承诺与 PyTorch、CUDA 或 TileLang 之间 bitwise 一致。

### 3.3 为什么 v1 不能表达 GEGLU

v1 是旧的 23 字段契约，没有激活/数值字段。若允许 v1 + GEGLU context：

- 旧制品可以在不声明的情况下被 GEGLU 复用，配置与计算语义不一致；
- 反向重建配置时无法判断应恢复成 silu 还是 gelu；
- 归档 format 1 也没有 `offload_config_*` 可承载这两个 code。

因此 v1 始终隐式 `silu/fused`；GEGLU 或 staged 必须输出 v2。

### 3.4 模式中性化与 `.so` / `.pt2` 的兼容边界

公开字符串仅为 `fused` / `staged`；没有原实验名称别名。`staged` 的 native int 仍为 1，schema 参数名/顺序、v2 向量 [24] 和 Manifest framing 都未改变：已按整数记录的计算配置无需迁移，改名本身不是新增 ABI。以模式字符串持久化的调用/配置必须改为 `staged`；历史运行 JSON 保留原样且不被重分类为本次运行，下面历史表中的 code=1 以通用规则名标注。

以下兼容约束针对新增尾参之前的 native schema，而不是本次字符串中性化：

新 schema 在 `fuse_moe` 和 `fuse_moe_offload_create` 末尾各增加两个默认 int 参数。其影响是：

- 旧 Python 代码（只传 5 个参数）配旧 `.so` 仍可按 `silu/fused` 工作；新 Python 封装即使使用默认值也会向 native 传 0/0，不能配旧 `.so`；
- 新调用传 `gelu`/`staged` 时，旧 `.so` 的注册 schema 不认识这些尾参，不能依赖它“自动兼容”；
- AOTI 生成的 `.cpp` 会按新 schema 发出 extern 调用，旧 `.so` 或旧 `.pt2` 可能参数数量/签名不匹配；
- 测试代码明确要求：生成源码不可手改，必须让 lead 重新构建。

**操作结论**：GEGLU/staged 上线必须使用同一份源码、同一工具链重新构建 `.so`，并用该 `.so` 重新 export/compile `.pt2`；`.so`、`.so.json`、`.pt2`、weights archive 必须成套部署。

## 4. 权重、路由与调用契约

### 4.1 canonical 形状

| 符号 | 形状 | dtype | 说明 |
|---|---|---|---|
| `x` | `(S, H)` | bf16/fp16 | token 序，连续 |
| `gate_up_weight` | `(E, 2I, H)` | 与 `x` 同 | gate 在 `[0,I)`，up 在 `[I,2I)` |
| `down_weight` | `(E, H, I)` | 与 `x` 同 | down 投影 |
| `topk_ids` | `(S, K)` | int32 | 取值 `[0,E)` |
| `topk_scale` | `(S, K)` | float32 | 调用方权重，不要求归一化 |

限制沿用现有 kernel：`H % 64 == 0`、`I % 64 == 0`、`K <= 128`、`E <= 512`、SM80+。`topk_ids` 的 `-1` masked 槽位见 2.7 节；canonical 非负值仍应在 `[0,E)`。

### 4.2 显式 canonical 权重打包

以下只是调用方接入示例，**不是本仓库已实现代码**：

```python
# gate 在前，up 在后；两套权重各自 reshape 为 (E, I, H)
w1_canonical = torch.cat(
    [gate_source.reshape(E, I, H), up_source.reshape(E, I, H)], dim=1
).contiguous()  # (E, 2I, H)

# 源 down 按 (E, I, H) 存储时，显式转成算子要求的 (E, H, I)
w2_canonical = down_source.reshape(E, I, H).transpose(1, 2).contiguous()
```

调用方还需保证：

- `w1_canonical`/`w2_canonical` 与 `x` 同 dtype；
- 全 GPU 路径在 CUDA；offload 路径权重在 CPU；
- `topk_scale` 保持 float32，不要先 cast 到 bf16/fp16；
- router 的 top-k、归一化、mask、重复路由策略均由调用方决定，算子只按 `topk_ids`/`topk_scale` 计算。

### 4.3 与普通 GELU MLP 的接入差异

普通 GELU MLP 只有一个 `(I,H)` 投影和一个 `(H,I)` down 投影。它接入本算子时会缺少 gate/up 的另一半，不能直接套用：

- 若把普通 MLP 的 `W1` 复制成 gate/up 两半，得到的不是原模型；
- 若只把 `expert_activation` 改成 `gelu` 而仍传普通 MLP 权重，形状/语义都不匹配；
- 本扩展服务于通用三矩阵 GEGLU/SwiGLU，不改变两矩阵 MLP 的拓扑。

## 5. 原本地基线验证矩阵与运行证据（2026-10-21）

> 本节为公开模式中性化之前的本地基线；表内 code=1 使用现行通用名称 `staged` 标注，历史 JSON 和命令没有改写。PASS 仅表示当时明确列出的环境/用例通过，不是中性化源码的重测。CUDA 12.8 远程新增证据见第 10 节；CPU 契约、GPU 数值验证和未执行项分开记录。

### 5.1 正确性

| ID | 场景 | 判据与证据 | 环境 | 状态 |
|---|---|---|---|---|
| V01 | 旧默认 silu/fused | CPU FP32 参考；native 五输入默认与显式 0/0 相同，R02/R03 | SM89 | PASS |
| V02 | gelu/fused | exact-erf 参考；严格标量见证能区分 tanh、漏 product/down cast，R02 | SM89 | PASS |
| V03 | gelu/staged | 分阶段参考；逐级漏 cast 的 mutant 均可区分，R02/R03/R04 | SM89 | PASS |
| V04 | silu/staged | 两种 dtype 的独立舍入模式，R02/R03 | SM89 | PASS |
| V05 | 路由形态 | uniform/single/sparse/duplicate/masked；signed/skewed 见完整 offload；NaN masked scale 忽略，R02/R03 | SM89 | PASS |
| V06 | bf16/fp16 | 84 个矩阵 case，shape/dtype/finite 检查，R02 | SM89 | PASS |
| V07 | K/E/H/I 边界 | K=1/128、E=1/512、H/I=64，四模式 × 两 dtype；R02/R03 | SM89 | PASS |
| V08 | 数值健康/复用 | 同输入重复调用和严格标量 witness；cache/多 context/S=0 恢复，R02/R03 | SM89 | PASS |

### 5.2 API、配置与制品

| ID | 场景 | 判据与证据 | 状态 |
|---|---|---|---|
| C01 | 严格字符串 | int/bool/None/大小写/空格变体在加载前拒绝，R02/R03 | PASS |
| C02 | native integer code | 绕开 Python，fuse_moe/create 的 -1/2 直接调用均明确拒绝，R03 | PASS |
| C03 | v1 默认配置 | 23 字段 + IDs，round-trip；R03 的 StoreTests | PASS |
| C04 | v2 GEGLU | [23]=1/[24]=0、IDs 从 25；R03 | PASS |
| C05 | v2 staged | [23]=0/[24]=1 和组合 1/1；R03/R04 | PASS |
| C06 | v1 + GEGLU | native 前置拒绝且 token 不变；R03 | PASS |
| C07 | v2 0/0 canonicality | Python 拒绝非 canonical；native 明确允许与默认 context 匹配的 v2 0/0；R03 | PASS |
| C08 | Manifest | 外层 framing=1；缺层/乱序/长度/其他层失配前置拒绝；R03/R04 | PASS |
| C09 | 归档 format1/2 | 带/不带配置 round-trip、缺 metadata/多余 buffer 拒绝；R03/R04 | PASS |
| C10 | 旧制品兼容拒绝 | R05 模拟旧 17 参数 create metadata/schema，在 dlopen 前拒绝；没有加载旧二进制实测 | PASS（元数据负向） |
| C11 | AOTI 重新构建 | FP16/BF16 每种 10 Python + 4 无 Python C++ case、S=2/7/64/2；v2 常量和失配提前阻断；R04 | PASS |

### 5.3 后端、安全与架构

| ID | 场景 | 判据与证据 | 状态 |
|---|---|---|---|
| B01 | SM89 cp.async | 两种 dtype 四模式，R01/R02/R03 | PASS |
| B02 | SM120 默认 cp.async | 必须用 SM120 / PyTorch 2.11 工具链重新验收 | BLOCKED：本地无环境 |
| B03 | SM120 TMA | TMA + 独立 act_mul 的静态接线不代替 GPU 实测 | BLOCKED：本地无环境 |
| B04 | route_buffer | 标量 witness/四模式/cache/mask，同步与异步 staging，R03/R06 | PASS |
| B05 | fp32_accumulate | 小 row cap/partial tiles、wide-N atomic、weighted round、context 隔离，R03/R06 | PASS |
| B06 | TileLang GELU GPU | 已定义测试；未安装 TileLang | BLOCKED |
| B07 | TileLang 无 erf | 4 个主机契约：仅 intrinsic/erff 可选，tanh 不替代；非连续/CPU/空输入在 JIT 前拒绝，R05 | PASS（CPU 拒绝策略） |
| B08 | TileLang staged GPU | 逐级 cast/guard 静态存在，尚无 GPU 编译/对拍结果 | BLOCKED |
| B09 | 独立 swiglu | 未修改 kernel；独立 SM89 库，原全量 23 项检查通过，R07 | PASS |
| S01 | GEGLU memcheck/synccheck | sync/async memcheck 各 6 tests、async synccheck 6 tests，均 0 errors，R06 | PASS |
| S02 | 独立 C++ full-leak | 故障注入/排空，14 fault cases，allocated/reserved after trim 为 0，0 bytes leaked，R06 | PASS |

### 5.4 精度指标与本轮判据

本轮**未放宽既有阈值**：全 GPU `atol=.05`，bf16 `rtol=.03` / fp16 `.02`；offload 小尺度合成参考 `atol=.004/rtol=.06`；AOTI chain `atol=2e-6`，同时 relative L2 < .02；严格标量见证依旧 `atol=rtol=0`。大形状/真实 checkpoint 不由小尺度绝对阈值推定通过。

`full_gpu_accuracy.json` 保存每个 case 的绝对/相对 p50、p99、max，RMSE、relative L2、finite、形状、dtype 与重复运行。相对误差分母固定 `abs(reference).clamp_min(1e-3)`，零附近 max-relative 不单独作为不合格结论。以下为 **同模式各 case 指标的上界**，不是将全部元素汇总得到的全局 p99；旧 silu/fused 含更多 legacy shape，不用于跨模式精度排名。

| dtype | 模式 | case 数 | max abs 上界 | case p99-abs 上界 | relative L2 上界 |
|---|---|---:|---:|---:|---:|
| bf16 | silu/fused | 18 | 0.00390625 | 0 | 2.7096e-4 |
| bf16 | gelu/fused | 8 | 0.0009765625 | 0 | 1.5577e-4 |
| bf16 | silu/staged | 8 | 0.00048828125 | 0 | 5.9748e-5 |
| bf16 | gelu/staged | 8 | 0.000244140625 | 0 | 9.1439e-5 |
| fp16 | silu/fused | 18 | 0.00048828125 | 0.0001220703125 | 1.1348e-4 |
| fp16 | gelu/fused | 8 | 0.0001220703125 | 0.0000152587890625 | 1.6163e-4 |
| fp16 | silu/staged | 8 | 0.000244140625 | 0.00000762939453125 | 6.6663e-5 |
| fp16 | gelu/staged | 8 | 0.0001220703125 | 0.0000152587890625 | 6.7264e-5 |

- AOTI GEGLU/staged 两层 fixture：fp16 最大绝对误差 `2.3841858e-7`、relative L2 `2.3152178e-4`；bf16 当前 fixture 为 exact equal。该结果**不推广为所有输入 bitwise 一致**。
- `--accuracy-report` 在固定 `(S,H,I,E,K)=(5,64,64,2,2)` 及同一 seed/quantized input 上，比较未舍入 CPU FP64 理想式与 FP32/FP64 合同模型的 gate_up/activation/product/down/weighted/final_pre_cast/final_post_cast。cast 前后使用**同一个未舍入 FP64 输出**为分母和目标；这是累计参考误差，不是直接读取 CUDA 内部累加器。
- 上述小样本分阶段证据证明报告链可复现，但不能替代调用方实际 shape/checkpoint 的误差分布。没有为测 ULP 新增侵入性 native debug op；FP32 `erff` 内部输出的 max ULP 尚未直接采集。exact-erf 选择由源码与可区分 tanh 的严格标量 witness 共同验证。
- 误差来源保留 MMA/atomic 累加顺序、cast 链、libdevice erf 与 PyTorch 的差异、旧 SiLU `__expf` 近似以及 float32 route-scale 加权。独立 swiglu 未改变。

### 5.5 原始证据索引

以下路径均相对 `artifacts/geglu_sm89_20261021/`；各命令、日期和退出状态在原始记录内，未改写历史失败。

| 证据 | 命令/范围 | 结果 | 文件 |
|---|---|---|---|
| R01 | build-native --arch 8.9 + testing=True | 两库构建 exit=0，gemm1 四特化和 act_mul 四特化均可构建 | `native_build.stdout.txt`；`native/{prod,testing}/recsys_fuse_moe_kernel.so(.json)` |
| R02 | test_fuse_moe.py --ops-library ... --accuracy-report --json ... | 84 matrix case / 149 PASS / 0 FAIL，exit=0 | `full_gpu_accuracy.json`、`full_gpu.stdout.txt` |
| R03 | test_fuse_moe_offload.py --allow-testing [--async-staging]；生产 GEGLUTests | 完整 sync/async 各 136 tests；专项各 6 tests；exit=0 | `offload_full.stdout.txt`；交付 run 的 `raw/native_{sync,async}.stderr.txt` |
| R04 | test_fuse_moe_offload_aoti.py --geglu-only --geglu-precision fp16/bf16 --async-staging --host-profile | 各 10 Python / 4 C++ case，重新 export/compile，exit=0 | `aoti_{fp16,bf16}/{geglu_report.json,acceptance.exit.json,evidence.hashes.json}` |
| R05 | loading / lowering / delivery / TileLang CPU contract / StoreTests | 分别 52、13、21、4、39 项；GPU skipped 不能算 passed | 交付 run 的 `raw/*_cpu.*.txt`、`raw/geglu_aoti_contracts_cpu.*.txt` |
| R06 | compute-sanitizer GEGLUTests + standalone --native-faults | access/sync 0 errors；native full-leak 0 bytes / 0 allocations，exit=0 | `sanitizer.stdout.txt`；交付 run 的 sanitizer raw 文件 |
| R07 | 独立 swiglu 全量、GEGLU demo | 23 PASS / 0 FAIL；demo forward 对拍通过 | `native/swiglu/`、`stage_profile.stdout.txt` |
| R08 | 双 dtype P01–P05 四模式正常测量 | 各 5 repeats × 50 calls，median-of-batch-means，exit=0 | `p01_p05_{bf16,fp16}.json`、`performance.stdout.txt` |
| R09 | 冻结源码的一键选定交付 | 64 条命令、unit/AOTI/sanitizer/performance 全部通过；源码 hash 前后一致；trace + 四模式 ABBA 有效 | `runs/20261021T023615Z-36a603e2a594/report.json`、`source_sha256.json`、`raw/`；`delivery.stdout.txt` |
| R10 | 单独复测 BF16 P01–P05 分段 | 五次轮转、500 stage records，排除 warmup 后每模式 10 样本，exit=0 | `stage_profile_isolated.stdout.txt`、`p01_p05_bf16_diagnostic_isolated.json` |

R09 最终 `state=partial`、`selected_stages_passed=true`、`full_delivery_passed=false`、exit=0：本次在同目录复用了 R01 刚构建的 native 库，未选择 build stage；旧性能 family 仅选择 `fp32_accumulate/bank2/disabled`，不等于默认五个配置的 full delivery。没有把 omitted 配置标为通过。正式全配置验收用第 7.3 节默认命令另选新的制品目录。

R09 完成后仅修正 `tests/test_tilelang_ops.py` 的参考比较 glue（CPU FP32 ref 与 GPU output 比较前移到同设备；已加载 CUDA 后端执行失败不得被 optional skip 掩盖），主机契约重跑通过；未重写 R09 的源码 hash，也未宣称后续修改已执行真实 TileLang kernel。native 计算源码、模式/舍入契约与 R01/R09 相同。

生产库摘要（只用于定位，完整 hash 在 sidecar）：library SHA256 前缀 `6427b68a8dcd`，source fingerprint 前缀 `36a35f6e77d1`；CUDA flags `-DFA_TARGETS=2 -O3 --use_fast_math -U__CUDA_NO_BFLOAT16_CONVERSIONS__`。测试库明确带 `CUSTOM_OPS_OFFLOAD_TESTING=1`，不混为生产制品。

## 6. 原本地 SM89 性能基线与优化决策

### 6.1 对照矩阵

每个 shape 至少跑以下四组，保持同一进程、同一设备、同一输入/路由：

| 组 | `expert_activation` | `numerical_mode` |
|---|---|---|
| A | `silu` | `fused` |
| B | `gelu` | `fused` |
| C | `silu` | `staged` |
| D | `gelu` | `staged` |

本轮正常性能采用以下既定 shape（uniform 路由，同 seed）：

| ID | S | H | I | E | K | 目的 |
|---|---:|---:|---:|---:|---:|---|
| P01 | 128 | 2048 | 1024 | 64 | 2 | 小批量、低 avg tokens/expert |
| P02 | 512 | 2048 | 1024 | 64 | 2 | 中批量 |
| P03 | 4096 | 2048 | 1024 | 64 | 2 | 大 shape 主路径 |
| P04 | 16384 | 2048 | 1024 | 64 | 2 | 吞吐上限 |
| P05 | 4096 | 4096 | 1408 | 128 | 2 | 宽 H/I、较大专家数 |

后端维度：

- CUDA cp.async（SM89 默认；SM120 默认也复用）；
- SM120 `FUSE_MOE_TMA=1`（仅远程 SM120 环境）；
- TileLang（有 erf 的构建）；
- offload `route_buffer` / `fp32_accumulate`（若该 shape 能进预算）。

### 6.2 测量规范

- 先 warmup，至少 5 次重复，报告 median；同 run 内做 A/B，避免跨 run 时钟漂移；
- 记录 `FUSE_MOE_TIME=1` 的 `gemm1_fused`/`gemm2_cp`/`reduce` 分段，或 TMA 的 `gemm1_tma`/`act_mul`/`gemm2_tma`；
- 端到端与分段同时报告；GELU 的 epilogue 成本可能被 GEMM 掩盖，只看 e2e 容易误判；
- 记录 GPU 型号、SM、时钟/功耗状态、PyTorch/CUDA 版本、扩展 SHA、编译 flags；
- 不同后端/不同架构的结果不得混在一张表里比较。

### 6.3 正常性能结果（GPU event µs/forward）

warmup=20、iters=50、repeats=5；每 repeat 左轮转四模式执行顺序，每模式每轮都 warmup。每个样本为 50 次调用的批均值，表内为**五个批均值的中位数**，不是逐次调用分位；host wall 用独立批次并包含最终 synchronize，原始时间和有效 TFLOPS 全部保留在 R08 JSON。未启用 CUDA graph、缓存 flush 或 FUSE_MOE_TIME，不包含 JIT/权重生成/库加载。

| ID | dtype（SM89） | A silu/fused | B gelu/fused | C silu/staged | D gelu/staged | B/A | D/A |
|---|---|---:|---:|---:|---:|---:|---:|
| P01 | bf16 | 871.40 | 872.10 | 872.36 | 872.10 | 1.00080 | 1.00080 |
| P02 | bf16 | 924.43 | 925.59 | 925.53 | 926.42 | 1.00126 | 1.00215 |
| P03 | bf16 | 1227.59 | 1238.32 | 1231.71 | 1253.24 | 1.00875 | 1.02090 |
| P04 | bf16 | 3816.07 | 3836.09 | 3837.79 | 3841.50 | 1.00524 | 1.00666 |
| P05 | bf16 | 6128.93 | 6125.51 | 6127.74 | 6128.60 | 0.99944 | 0.99995 |
| P01 | fp16 | 871.22 | 872.26 | 872.08 | 872.59 | 1.00120 | 1.00158 |
| P02 | fp16 | 924.84 | 925.49 | 925.39 | 926.13 | 1.00071 | 1.00139 |
| P03 | fp16 | 1253.21 | 1255.94 | 1266.95 | 1293.80 | 1.00218 | 1.03239 |
| P04 | fp16 | 3825.01 | 3848.04 | 3835.14 | 3860.82 | 1.00602 | 1.00936 |
| P05 | fp16 | 6128.97 | 6126.00 | 6129.91 | 6128.02 | 0.99952 | 0.99985 |

比值 >1 表示变慢，<1 表示变快；接近 1 的差异不称为稳定收益。当前合成矩阵 B/A 最大约 +0.87%，D/A 最大约 +3.24%（fp16 P03）；没有指定的调用方性能 SLA，因此仅建立基线，不宣称“性能达标”或完整模型 speedup。较大 E 的 P01/P05 不能拿 README 旧 E=8 的 TFLOPS 外推比较。SM120 cp.async/TMA 和 TileLang 性能仍 BLOCKED。

### 6.3.1 分段诊断与边界

`FUSE_MOE_TIME=1` 分别记录 count/gemm1_fused/gemm2_cp/reduce，5 repeats、每组 warmup=3、iters=1；原始 500 条 stage 记录和 diagnostic JSON 见 R10。其他本轮 GPU 验收任务结束后单独采集，按 JSON 的确定轮转顺序映射各 stage，排除每轮 warmup 后每模式保留 **10 个分段样本**。下表均为各分段独立中位数（µs），因此其和不必等于 total 中位数。

| ID | 模式 | count | gemm1_fused | gemm2_cp | reduce | total |
|---|---|---:|---:|---:|---:|---:|
| P01 | silu/fused | 6.0 | 576.0 | 296.5 | 5.0 | 884.0 |
| P01 | gelu/fused | 5.5 | 578.5 | 296.0 | 5.0 | 885.0 |
| P01 | silu/staged | 6.0 | 578.0 | 297.0 | 4.0 | 885.0 |
| P01 | gelu/staged | 5.0 | 577.5 | 296.5 | 4.0 | 884.5 |
| P02 | silu/fused | 10.0 | 604.0 | 315.0 | 6.5 | 937.5 |
| P02 | gelu/fused | 11.0 | 604.5 | 313.0 | 6.5 | 936.5 |
| P02 | silu/staged | 10.0 | 604.5 | 313.0 | 6.0 | 936.0 |
| P02 | gelu/staged | 10.5 | 605.5 | 314.5 | 6.5 | 938.5 |
| P03 | silu/fused | 11.0 | 796.0 | 390.0 | 35.0 | 1232.0 |
| P03 | gelu/fused | 10.0 | 794.0 | 390.5 | 35.0 | 1230.5 |
| P03 | silu/staged | 11.0 | 800.5 | 390.0 | 35.0 | 1236.0 |
| P03 | gelu/staged | 10.5 | 802.5 | 390.0 | 35.0 | 1238.5 |
| P04 | silu/fused | 11.0 | 2384.0 | 1222.0 | 195.0 | 3811.0 |
| P04 | gelu/fused | 11.0 | 2405.0 | 1222.0 | 195.0 | 3832.5 |
| P04 | silu/staged | 11.5 | 2406.0 | 1222.0 | 195.0 | 3835.0 |
| P04 | gelu/staged | 11.5 | 2409.0 | 1222.0 | 195.0 | 3838.0 |
| P05 | silu/fused | 10.0 | 4097.5 | 1930.5 | 85.0 | 6130.0 |
| P05 | gelu/fused | 11.0 | 4095.5 | 1933.0 | 85.0 | 6130.0 |
| P05 | silu/staged | 11.0 | 4110.5 | 1940.0 | 85.0 | 6144.0 |
| P05 | gelu/staged | 10.5 | 4101.5 | 1929.5 | 85.0 | 6130.0 |

诊断标记 `performance_eligible=false`，TFLOPS 和性能 ratio 为空，不进入正常表/ABBA 汇总；串行 launch/同步/打印改变计时路径，不能用它宣称线上吞吐。首次与小型正确性任务有时间交集的诊断仍保存为 `stage_profile.stdout.txt` / `p01_p05_bf16_diagnostic.json`，不据此推断稳定性能。

### 6.3.2 Offload ABBA 与 GPU 时间线

R09 的代表性 `small/uniform/fp32_accumulate/bank2/disabled` BF16 四模式，正常 host wall 的 warm pooled median（ms）：

| 模式 | sync | async |
|---|---:|---:|
| silu/fused | 0.51768 | 0.52585 |
| gelu/fused | 0.50700 | 0.54972 |
| silu/staged | 0.50293 | 0.53444 |
| gelu/staged | 0.51418 | 0.52244 |

各模式两次 sync、两次 async，各次 40 cold/40 warm/3 priming，模型构建不进入 warm 时间；是合成单层描述性统计，跨进程未建立置信区间，不能据 <1ms 的差异对模式或异步策略作稳定收益判断。其汇总 `schema_version=2`，八个 comparison 分组全部 complete（四个旧 family 场景 + 四个新 GEGLU 模式），没有混 dtype/激活/舍入。

独立 Nsight/CUPTI trace 覆盖 83 个匹配 NVTX scope，权重 H2D `783286272` bytes 与 native 流量统计一致，GPU kernel/H2D 交集 **0 ns**。这是有效的“未观察到重叠”证据，不因存在双 stream/async staging 就宣称重叠收益；trace 原始 `.nsys-rep`/SQLite/`gpu_overlap.json` 见 R09 的 `trace/`。

### 6.4 优化决策与未验证假设

**本轮决策：暂不另立 GELU epilogue 优化，不改变默认 GEMM/tile 策略。**正常 P01–P05 结果没有显示大幅回退；最多约 3.24% 的 staged 开销不足以在缺少调用方实际 shape 和远程架构数据时引入新的数值近似/调度分叉。未来先采集目标路由/权重规模，再做独立 A/B 和精度回归。

以下仍是未验证假设，不是性能承诺：

- **H1**：大 avg tokens/expert 时，GELU 的 `erff` epilogue 成本可被 GEMM 主循环部分掩盖；小批量/低占用时相对开销可能更明显。需用分段计时验证。
- **H2**：staged 的逐级 cast 会增加 epilogue 指令数，但可能仍被内存/MMA 掩盖；需比较 A/B/C/D 四组。
- **H3**：TMA 路径多一次 `gate_up_out` 物化和独立 `act_mul`，GELU 的相对成本可能比 cp.async 融合路径更可见；需在 SM120 上单独测量。
- **H4**：TileLang 的 `T.erf` 可能落到 libdevice 调用，性能与 CUDA `erff` 不同；需看生成代码和实测，不能直接外推。
- **H5**：现有 `FUSE_MOE_TILE_M`/`TILE_N`/`SCHED` 的调优点是为 SiLU 路径调的；GELU 可能改变最优点，但首版不自动改策略，先做 A/B 再决定。

### 6.5 可复用的调优开关

| 开关 | 作用 | 备注 |
|---|---|---|
| `FUSE_MOE_TIME=1` | 分段计时，自动串行 launch | 只用于归因，不用于线上 |
| `FUSE_MOE_TILE_M=32/64/128` | 强制 kTileM | 需与 count 阶段 tile 一致 |
| `FUSE_MOE_TILE_N=64` | 关闭 gemm2 宽 N 变体 | 仅对比基线 |
| `FUSE_MOE_SCHED=vert` | cp.async 的 N-major 调度 | W 面板复用场景 |
| `FUSE_MOE_TMA=1` | SM120 切换 TMA 引擎 | 需要 gencode 覆盖 sm90+ |
| `FUSE_MOE_NO_PDL` | 强制普通 launch | 归因/排查用 |

## 7. 部署与复现命令模板

> 下列为可复现调用模板；实际执行记录见第 5.5 节。输出目录必须是新的，以免覆盖已有 `.so/.pt2/JSON` 证据。SM89 和 SM120 必须使用各自工具链/架构制品；SM120 命令在本轮未执行。

### 7.1 构建原生库

```bash
cd /workdir/custom_ops
python examples/fuse_moe_offload_aoti.py build-native \
  --output-dir artifacts/moe_geglu_native \
  --arch 8.9
```

SM120 使用 `--arch 12.0a`，并在其自身的 Python 3.12 / PyTorch 2.11 / CUDA 12.8 环境构建；不得复用 SM89 的 `.so`。

### 7.2 导出与运行 AOTI

```bash
cd /workdir/custom_ops
python examples/fuse_moe_offload_aoti.py export \
  --ops-library <path-to-native.so> \
  --output-dir artifacts/moe_geglu \
  --layers 2 --expert-activation gelu --numerical-mode staged

python examples/fuse_moe_offload_aoti.py run \
  --ops-library <path-to-native.so> \
  --package artifacts/moe_geglu/model.pt2 \
  --weights artifacts/moe_geglu/weights.pt
```

要求：

- 导出前设置 `OffloadConfig(expert_activation="gelu", numerical_mode="fused"|"staged")`，确保 graph 常量是 v2；
- 旧 `.pt2` 不得复用于新 schema；必须用新 `.so` 重新 export/compile；
- 生成的 `.cpp` 不可手改；失败时让 lead 重编。

### 7.3 测试命令

```bash
cd /workdir/custom_ops

# 全 GPU：旧默认 + 两 dtype 四模式 + 边界 + FP64 小样本分阶段报告；新 JSON 路径
PYTHONPATH=/workdir python tests/test_fuse_moe.py \
  --ops-library <path-to-prod.so> --accuracy-report --json <new-accuracy.json>

# CPU 契约（不编译/执行 GPU）
PYTHONPATH=/workdir python tests/test_tilelang_ops.py --cpu-only
PYTHONPATH=/workdir python tests/test_fuse_moe_offload_aoti.py --cpu-contracts \
  test_explicit_expert_modes_v2_config_cpu_contract \
  test_geglu_staged_reference_cpu_contract \
  test_prefill_create_explicit_mode_arguments_cpu_contract \
  test_geglu_metrics_cpu_contract

# offload 前端/原生完整回归（故障注入需 testing 库且必须显式 opt-in）
PYTHONPATH=/workdir python tests/test_fuse_moe_offload.py \
  --ops-library <path-to-testing.so> --allow-testing -v
PYTHONPATH=/workdir python tests/test_fuse_moe_offload.py \
  --ops-library <path-to-testing.so> --allow-testing --async-staging -v

# 专项 GEGLU AOTI / 无 Python C++；fp16/bf16 分别使用新输出目录
PYTHONPATH=/workdir python tests/test_fuse_moe_offload_aoti.py \
  --ops-library <path-to-prod.so> --geglu-only --geglu-precision bf16 \
  --async-staging --host-profile --output-dir <new-geglu-dir> --sizes 2 7 64

# TileLang GPU（需已安装且支持 exact erf；本轮未执行）
PYTHONPATH=/workdir python tests/test_tilelang_ops.py \
  --ops fuse_moe swiglu --ops-library <path-to-prod.so>
```

一键交付模板：

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir python tests/run_fuse_moe_offload_delivery.py \
  --output-dir /workdir/custom_ops/artifacts/geglu_delivery_sm89 \
  --target-arch 8.9 \
  --jobs 2 \
  --build-timeout 1800 \
  --unit-timeout 1200 \
  --aoti-timeout 1800 \
  --sanitizer-timeout 1200 \
  --performance-timeout 1800
```

SM120 在独立环境使用 `--target-arch 12.0a`，输出目录不得覆盖 SM89 证据。

### 7.4 全 GPU 四模式性能复现

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir python benchmark/benchmark_tilelang_ops.py \
  --backend cuda --ops fuse_moe --ops-library <path-to-prod.so> \
  --moe-mode-matrix --dtype bf16 --repeats 5 --warmup 20 --iters 50 \
  --moe-shapes 128,2048,1024,64,2 512,2048,1024,64,2 \
               4096,2048,1024,64,2 16384,2048,1024,64,2 \
               4096,4096,1408,128,2 \
  --json <new-bf16-performance.json>
```

fp16 用相同命令改 `--dtype fp16` 和新 JSON 路径；不要在正常测量时设置 `FUSE_MOE_TIME`。分段归因使用独立进程加 `FUSE_MOE_TIME=1`，`--warmup 3 --iters 1 --repeats 5`，单独保存 stdout 和 diagnostic JSON；不能把诊断 ratio/TFLOPS 回填正常表。

一键交付默认追加代表性 BF16 offload `small/uniform/fp32_accumulate/bank2/disabled` 四模式，各自 sync/async ABBA 共 16 次独立 CLI；不是 P01–P05 常驻 GPU 性能。`--no-geglu-performance-matrix` 会记为 skipped 而非 full delivery passed；汇总按 backend/shape/激活/数值/dtype 分区。CPU-only 或 SM80 以下的 baseline JSON 必须拒绝为通过。

### 7.5 制品一致性检查

- `.so`、`.so.json`、`.pt2`、weights `.pt` 必须来自同一源码/工具链；
- 检查 `OffloadConfig.manifest()` 与归档 `offload_config_*` 的 v1/v2 一致；
- 若只改 Python 配置、不改 `.so`，GEGLU/staged 仍会在旧 schema 上失败；
- 若只改 `.so`、不重编 `.pt2`，AOTI 生成代码的参数列表可能与新 schema 不一致。

## 8. 开发日志与待办

### 8.1 2026-10-19（历史记录）

- 明确通用拓扑：三矩阵 GEGLU，普通两矩阵 MLP 不是本扩展目标。
- 确认两个新维度：`expert_activation`、`numerical_mode`；默认 `silu/fused`；`activation_mode` 继续表示缓冲/累加模式。
- 工作树中已存在 CUDA、offload、Python、TileLang 的相关代码改动；这些改动不是本文撰写产生的，本文只做静态梳理。
- 完成配置 v1/v2、manifest、归档 format 1/2 的代码路径梳理。
- 记录待验证矩阵、性能对照和部署模板。
- 撰写期间 `tests/` 也被并行修改，加入 GEGLU/staged 与配置 v2 用例；本文只记录其静态存在，不宣称运行结果。
- **本文未执行**：GPU kernel、单元测试、AOTI 编译、性能采集。编译/测试由 lead 在另一进程进行，结果表待回填。
- 本次文档任务只新增 `docs/MOE_GEGLU_EXTENSION.md`，未改动其它文件，未做 git commit。

### 8.2 2026-10-21 增量实施

- 明确通用 gate/up/down 的三矩阵布局，不引入私有模型类、checkpoint 键名或拓扑假设。
- 原生 gemm1/wide-N smem 预算守卫改为 release 保留的 `TORCH_CHECK`，同时保留已有按 device/kernel 实际 opt-in 上限校验。
- 修复 TileLang mask OOB 风险：count/route 都在索引前 guard，NaN masked scale 不参与计算；补连续/CUDA/正 S/H 校验和无 erf 的纯主机失败测试。保留 TileLang fused 旧 FP32 down 数值路径，显式记下跨后端差异。
- 正确性新增结构化精度分位、FP64 参考模型、K=128/E=512/E=1 和重复健康；标量精确-erf/逐级 cast 的强 witness 未降低阈值。新 JSON 输出拒绝覆盖旧证据。
- 补 native 非法 -1/2 code 拒绝，修正 `GEGLUTests --async-staging` 确实向 runtime 传递异步策略（原 CLI 只影响旧 OffloadTests）；配置 canonical 向量随异步设置保持一致。
- AOTI 增加公开 `--cpu-contracts`，指标有 finite/分位；重新构建两个 dtype 的双层 `.pt2`，独立 C++ 没有初始化 Python，失配在 downstream 前阻断。
- 性能改为同 Tensor/route、四模式每轮轮转、五次批均值 median，保留事件/host 两类时间、raw、SHA/flags/env；诊断没有性能比值。交付 baseline 校验 JSON 的真实 GPU state，四模式 ABBA 分组不可混 dtype/激活/舍入。
- 更新现有 `README.md` / `README_EN.md` / offload 历史文档 API；全 GPU demo 可选激活、数值模式、dtype、预编译库且保持 float32 router-scale。
- 执行 R01–R10，冻结源码的一键选定 stage 全部通过但明确 partial（未选 build/完整配置性能）；独立 stage 复测、Nsight 0 ns 重叠、主机比较 glue 的事后修正均如实记录，未改写旧 evidence，未做 git commit。

### 8.3 任务收敛与剩余边界

| ID | 事项 | 责任 | 当前状态 |
|---|---|---|---|
| T01 | SM89 两种构建、四种 epilogue 特化 | 本轮 | 完成；R01 |
| T02 | V01–V08 结果与日志 | 本轮 | 完成；R02/R03 |
| T03 | C01–C11 配置/归档/AOTI | 本轮 | 已回填；C10 为 metadata/schema 模拟负向，不冒充旧 binary 实测 |
| T04 | B01–B09 支持矩阵 | 本轮 | 已回填；原本地矩阵保留，第 10 节补充 CUDA 12.8 的 SM120 默认/TMA 与 TileLang 实测 |
| T05 | SM120 + PyTorch 2.11 构建/对拍 | 本轮远程验证 | 完成 CUDA 12.8 实测；两种舍入、两激活、两个 dtype；TMA 数值与 barrier 修复见第 10 节 |
| T06 | 本地精度阈值/分位/FP64 参考 | 本轮 | 完成本地合成基线、既有阈值不放宽；内部 erff FP32 ULP 和调用方权重仍未采集 |
| T07 | P01–P05 正常性能与分段诊断 | 本轮 | 完成；正常双 dtype R08、独立 BF16 分段 R10，offload/trace R09；不混为性能承诺 |
| T08 | 是否立项 GELU epilogue 优化 | 本轮 | 已决策：暂不立项，不改 GEMM 主骨架或默认策略 |
| T09 | 现有 README/旧 offload API | 本轮 | 完成；旧 SM120 结果明确只适用于旧源码 |

CUDA 12.8 / PyTorch 2.11 和 TileLang 0.1.14 的新增远程验收见第 10 节，不覆盖此前本地环境快照。调用方实际权重/路由的端到端质量、内部 FP32 erf ULP、远程性能 SLA 不由本轮合成对拍推定。

### 8.4 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| TileLang 构建无 `erf`/`erff` | GELU 无法编译 | 显式 `NotImplementedError`；换支持 erf 的构建，禁止 tanh 回退 |
| 旧 `.so`/`.pt2` 混用 | 参数数量/签名不匹配或静默走旧语义 | 同一源码重编并成套部署；负向测试 C10/C11 |
| staged 非 bitwise | 与 torch 参考存在末位差异 | 使用分阶段参考和容差；调用方不得要求逐 bit |
| v1 配置被误用于 GEGLU | 语义退化为 silu/fused | Python 输出 v2；C++ 对 v1+GEGLU context 明确拒绝 |
| TMA 路径多一次物化 | GELU 相对开销可能更明显 | SM120 上单独测 B03；不据 cp.async 结果外推 |
| 跨架构/工具链混用 | 本机与远程 CUDA/ABI 不同 | 独立构建与实测；CUDA 12.8/ABI=1 制品不得在本机 CUDA 11.8/ABI=0 直接加载 |
| 性能结论被单次 run 误导 | A/B 差异落在时钟漂移量级 | 同 run 多轮 median；记录环境与 flags |

## 9. 结论边界与通用数值模式

`numerical_mode="staged"` 严格表示中间运算边界逐阶段舍入规则，不表示外部程序处于何种用途或运行阶段，不改变模型拓扑、路由或 GEMM 数量。公开接口和文档不包含调用方私有类名/路径/权重键；历史实验制品保持本地且不纳入开源源码打包。

- 本地 SM89（PyTorch 2.6.0a0/CUDA 11.8）重新构建、84 case 正确性、完整 offload、双 dtype GEGLU AOTI、sanitizer 和 P01–P05 多轮性能均有实际运行证据；这部分不再 pending。
- 通用名称修订后的 CUDA 12.8 / PyTorch 2.11 新制品在 SM89、SM120 的默认四模式、offload、双 dtype AOTI 及 TileLang 均有独立运行记录；TMA 精度修复后已通过四模式 84 case，安全检查结果按第 10 节单独标记，不能由数值通过外推。
- 精度判据限已列出的合成 fixture，未放宽旧阈值；性能是算子前向基线，不是完整模型 QoS/上线承诺。分阶段参考不是 CUDA 内部 instrumentation，内部 FP32 erf ULP 与调用方权重质量仍待补。
- `.so/.so.json/.pt2/weights` 必须同 schema/工具链/配置成套部署，禁止混用旧制品；测试 build 不作为 production .so。
- 外部模型、权重恢复、router、bias/shared expert、反向传播仍归调用方。已完成所列环境的算子验证不意味着任意架构、任意输入 production-ready，未提交 git。

## 10. 通用模式与 CUDA 12.8 远程复验（2026-10-21）

### 10.1 开源边界与兼容性

- 公开接口只接受 `expert_activation="silu"|"gelu"`、`numerical_mode="fused"|"staged"`；删除旧实验名称映射，不保留别名。源码/测试/示例/README 不再包含私有模型类、私有目录或按用途命名的数值模式。
- `fused=0`、`staged=1`；native schema 参数名、int code、v1/v2 固定字段、Manifest framing=1 和归档 format=2 不变。改名本身不新增 ABI，旧字符串调用必须显式改名，整数配置向量无需迁移。
- `.gitignore` 已排除连接配置、`3rd/` 和 `artifacts/`。远程仅复制公开算子源码及 `thirdparty/` 构建依赖到独立目录，不复制凭据、私有模型目录、旧 `.so/.pt2`，不覆盖远程原有工作树。
- 两份远程审计清单 `source_sha256_review3.json` 的 100 个源文件逐项一致，并与本地源码一致；README/本文的事后证据补充不作为 native 计算源码变化。历史失败、初版库和修复后运行分别保存，不改写历史 JSON。

### 10.2 实测环境

| 项目 | SM89 | SM120 |
|---|---|---|
| GPU | RTX 4090 D | RTX 5090 D v2 |
| Python / PyTorch | 3.12.11 / 2.11.0+cu128 | 3.12.11 / 2.11.0+cu128 |
| CUDA build / nvcc | 12.8 / V12.8.61 | 12.8 / V12.8.61 |
| GCC/G++ / C++11 ABI | 10.3.1 / 1 | 10.3.1 / 1 |
| TileLang | 0.1.14，`T.erf` 可用 | 0.1.14，`T.erf` 可用 |
| 官方 compute-sanitizer | 2025.1.0.0 | 2025.1.0.0 |
| native target | 8.9 | 12.0a |

两边都采用 `MAX_JOBS=2`，生产与 `CUSTOM_OPS_OFFLOAD_TESTING=1` 制品分开构建，未更改 GPU 时钟或其他服务。

### 10.3 远程执行发现与实际修复

1. **TileLang 小维度 staged 原子写**：初次 SM120 `(7,64,64,2,2)` masked 用例出现 launch failure；memcheck 报设备 trap，生成 CUDA 中出现 `AtomicAddx2` 到 `Yacc_local_cast` 而不是全局输出。修复为先在 FP32 fragment 内完成 down/weighted 舍入，再以独立语句原子写全局输出；不再把双 cast 嵌入带谓词原子表达式。N tile 只使用整除投影维度的宽度，超宽/partial panel 使用 64 列基线，避免跨 gate/up/expert 区域依赖；配置副本不会修改调用方 dict。四模式双 dtype mask 和严格 scalar witness 重新执行，未放宽容差。
2. **TMA fused 提前舍入**：初次 TMA `--quick` 为 119 PASS / 2 FAIL，BF16 GELU/fused 的严格标量见证失败，不能由随机 allclose 掩盖。原因是 gemm1 固定物化输入 dtype；现 `fused` gemm1 输出 FP32 gate/up，`staged` 仍输出输入 dtype。`act_mul` 为类型参数化输入，FP32 每半区用两次 16B load，输出低精度不变；没有新增 GEMM/激活 kernel，没有切换 GELU 近似。
3. **TMA mbarrier 初始化**：数值/memcheck 已通过但 synccheck 的原分布式 init 报 `Missing init`；旧低精度库和增加跟踪容量同样失败。改为 thread0 逐项初始化 full/empty barrier，保持 arrival count、fence、CTA 同步、流水 phase 与别名区不变。默认参数和 No-PDL 诊断复跑均 0 errors；不把原失败径直判为误报，不修改官方工具或屏蔽检查。

FP32 C 的最大别名覆盖（M32/M64）为 `(72*63+64)*4=18400B`，最小 operand 区为 73728B，增加编译期容量断言；Tensor Core input 始终 BF16/FP16，MMA 数量未改变。

### 10.4 验收范围与原始记录

本地已取回 `artifacts/staged_cuda128_sm89/` 和 `artifacts/staged_cuda128_sm120/`；以下路径相对各架构目录。最终制品使用 `native/review3/{prod,testing}/`，不是初次或 `review2` 库。

| 范围 | SM89 | SM120 | 原始文件 |
|---|---|---|---|
| 生产/测试 native 重新构建 | PASS | PASS | `native_review3.stdout.txt`、`native_testing_review3.stdout.txt`、各 `.so.json` |
| 默认 cp.async 全 GPU 四模式 / 两 dtype | 84 cases / 149 PASS / 0 FAIL | 84 cases / 149 PASS / 0 FAIL | `full_gpu_cp_review3.json` / `.stdout.txt` |
| TMA 全 GPU 四模式 / 两 dtype | 不适用 | 84 cases / 149 PASS / 0 FAIL | `full_gpu_tma_review3.json` / `.stdout.txt` |
| 完整 offload 同步/异步 | 各 136 tests / PASS | 各 136 tests / PASS | `offload_{sync,async}_review3.stdout.txt` |
| GELU/staged AOTI 重编、Python/独立 C++ | 两 dtype 各 10 / 4 cases | 两 dtype 各 10 / 4 cases | `aoti_review3_{bf16,fp16}/geglu_report.json`、stdout、package/weights/native runner |
| TileLang MoE + 独立 swiglu / native 对拍 | 96 / 96 PASS | 96 / 96 PASS | `tilelang_review3.stdout.txt` |
| TMA memcheck / synccheck | 不适用 | 各 121 checks PASS，0 errors | `tma_{memcheck,synccheck}_review3.stdout.txt` |
| TMA No-PDL 诊断 synccheck | 不适用 | 121 checks PASS，0 errors | `tma_no_pdl_synccheck_review3.stdout.txt` |
| offload sanitizer | async GEGLU memcheck 6 tests，0 errors | async GEGLU synccheck 6 tests，0 errors | `offload_{memcheck,synccheck}_review3.stdout.txt` |
| TileLang masked/all_masked sanitizer | 主矩阵已覆盖 | 两 dtype 四模式与 native 对拍，0 errors | `tilelang_all_masked_memcheck_review2.stdout.txt`（TileLang 修复源码未再变化） |
| TileLang 192 对齐维度 / duplicate 路由 | 未追加专项 | 8 mode/dtype cases，16 comparisons PASS | `tilelang_192_review3.stdout.txt` |

SM120 AOTI synthetic fixture 的 BF16 当前 exact equal；FP16 最大绝对误差 `2.3841858e-7`、relative L2 上界约 `2.4304e-4`。这是限定 fixture 结果，不承诺任意输入 bitwise 一致。CPU 加载52项、lowering13项、delivery21项、AOTI四契约、TileLang五契约也通过，纯主机结果未计入真实 GPU 成功项。

本轮没有运行全配置一键 delivery 或重新采集 P01–P05 远程性能；第 6 节仍为此前本地 CUDA 11.8 描述性基线，不能当作修复后 TMA/TileLang 的 CUDA 12.8 性能承诺。
