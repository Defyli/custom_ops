# Prefill-only 流式 MoE：消费级 GPU 的多级存储扩展方案

> **定位与交付状态**：本项目是通用算子仓库，不是模型加载框架或 serving 系统。Prefill MoE 的 RAM 流式执行、缓存、低显存激活、冻结配置、公开 Python/C++ 接入、checked AOTi 编译、失败状态与排空已实现。**T01–T09 在约定范围内全部完成；2026-10-15 冻结源码的 SM89、SM120 全阶段交付均 exit=0，`full_delivery_passed=true`。**当前接口、支持矩阵、复现命令及原始证据以第 15.15 节为准；第 15.13–15.14 节保留前阶段接口/结果，不代表本轮最新状态。全部任务状态及未立项方向仍只维护在[第 16 节](#16-统一待办清单与-review-结论)。没有业务模型不阻塞算子交付，也不宣称业务质量已验证；其他算子组的 AOTi、任意 shape/ABI 或任意 Python 进程零泄漏不由本轮推定。
> **日期**：设计 2026-10-09；初次待办收敛 2026-10-13；仓库定位修订 2026-10-14；最终开发、双架构验收与交付收敛 2026-10-15。
> **场景**：通用 Prefill/批量 MoE 推理前向；生成式推荐是应用背景而非绑定的业务实现，无自回归 Decode。
> **结论**：保留现有全 GPU `fuse_moe` 快路径，新增对外单算子的 `fuse_moe_offload`，由原生 C++ 运行时实现显存有界的专家权重流式执行。**AOTi 是首版强制能力**；参数从输入与 canonical CPU 权重元数据推导，`state_dict.pth` 是可选接入来源，RAM/pinned/GPU 额度可配置，数值验收采用合理容差而非逐位一致。首版由 GPU 完成全部专家计算，CPU 负责存储、调度与搬运；SSD、CPU 专家计算和量化属于未立项的可选方向（见第 16.4 节），不是当前交付承诺。
>
> **2026-10-21 GEGLU API 补充**：本页第 15/16 节的交付结论对应 2026-10-15 的旧 `silu/fused` 源码，**不能推定新增 GEGLU/staged 在 SM120 已通过**。现行全 GPU 接口新增关键字 `expert_activation="silu"|"gelu"`、`numerical_mode="fused"|"staged"`；offload 在 `OffloadConfig` 和 native create 固化同名维度，`activation_mode` 不改变含义。默认配置仍为 v1（23 字段）；非默认为 v2（25 字段，23/24 是激活/舍入 code，resident IDs 从 25 开始），Manifest 外层 framing 仍为 1，format2 CPU 归档携带各层冻结配置。`.so/.so.json/.pt2/weights.pt` 必须按新 schema 重新构建并成套部署；`staged` 仅定义前向舍入、不提供反向。当前验证与复现命令统一见 [GEGLU 扩展记录](MOE_GEGLU_EXTENSION.md)：第 10 节记录通用 `staged` 命名后的 CUDA 12.8 / PyTorch 2.11 在 SM89、SM120 默认/TMA、TileLang、双 dtype AOTI 和安全检查的独立复验；这些新结论不由下文旧交付记录外推。下文历史配置/schema/TileLang 未支持 mask 的描述保留为对应日期的事实。

## 1. 背景与范围

### 1.1 要解决的问题

当前 `fuse_moe` 要求一个 MoE 层的全部专家权重以连续 CUDA Tensor 传入。该设计适合专家已经驻留显存的高吞吐计算，但存在两类容量问题：

1. 单层专家权重就超过显存，现有接口无法直接调用。
2. 单层能够放下，但全模型各层专家的总和放不下，需要跨层复用显存中的权重空间。

此外，Prefill 的 route 数较大时，`act_out`、`down_out` 等中间激活也可能成为显存瓶颈。只将权重放到 RAM，并不能保证任意规模输入都能执行。

本方案的目标是 **Prefill-only out-of-core Grouped MoE**：在保持实际路由及专家计算语义的前提下，通过专家分批加载和 GPU GEMM 执行，使权重显存占用不再随全量专家数线性增长。

### 1.2 首版目标

- 单 NVIDIA GPU，沿用现有 CUDA/CuTe 计算路径；目标为本机及 `remote.txt` 的 4090D/5090D，重点验证 sm89/sm120，**排除 V100/sm70**。
- 推理前向，输入与专家权重为 BF16/FP16，MMA 使用 FP32 累加。
- 权重保存在 RAM，仅被实际路由选中的专家需要进入 GPU。
- 对外一次算子调用，内部一次路由分组、多批专家计算、最终一次输出 cast；分阶段 GPU launcher 是内部实现细节。
- 首版必须通过 `torch.export -> AOTInductor -> 加载执行` 验收，不能以 eager 或 `torch.compile` 可调用代替。
- GPU/RAM/pinned 有独立可配置上限和保留余量；按当前层权重、实际输入与路由规划资源，不固定专家 slot 数。
- 全 GPU 路径的接口、默认策略和性能不因新增运行时而改变。
- 提供可复用的图外生命周期、冻结配置与安全调用接入层，支持 Python 集成和无 Python 的 LibTorch/AOTi 集成；部署端显式加载预编译扩展，不在初始化或请求时隐式补编译。
- 用合法合成形状、路由、权重和负向输入独立验收，不把完整业务模型、发布系统或业务质量指标作为算子交付条件。
- P1 先用完整 per-route Down 建立流式正确性基线；P2 增加 FP32 累加与激活 tile 化以扩展输入容量。两种模式均明确预检容量边界。

### 1.3 首版不包含

- Decode、推测解码、MTP、Decode 专用 GEMV、KV/prefix cache。
- 跨 token 热专家预测、跨层推测路由、图内 CPU/GPU doorbell 协议。
- CPU 专家计算、动态低精度替换、专家跳过或路由重写。
- SSD 后备、GDS、Unified Memory 超额分配、多 GPU 专家并行。
- 整段流式执行的 CUDA Graph capture/replay、反向传播和训练。**AOTi 不在排除项内**，其与 CUDA Graph 是不同能力。

上述能力是首版范围边界，不是分散的开发待办；是否另行立项统一见第 16.4 节。相同结构新权重通过新 store/context 显式重新部署，并不意味着本算子支持训练反向。

### 1.4 Prefill-only 带来的简化

`topk_ids` 在调用当前 MoE 层时已经确定。因此，同一层可以精确知道后续要用哪些专家，按确定顺序加载即可，无需学习或预测下一批专家。

这不意味着下一层路由也已确定：下一层依赖当前层输出。首版不提前猜测下一层专家。

**没有 Decode，不等于每个专家都有大 batch。**短输入、较大的专家数和倾斜路由都会产生低工作量专家；性能分析仍须使用每个专家实际收到的 token 数。

### 1.5 仓库职责与完整能力的定义（2026-10-14 修订）

**“完整”指当前公开支持范围内的计算、资源、生命周期、失败语义和接入闭环完整，不指支持任意 MoE/任意 shape，也不指替业务交付整套部署系统。**缺少业务模型不是仓库开发阻塞；合成验收可以关闭算子任务，但不能推出真实 workload 的缓存收益或业务无损。

| 层次 | 本仓负责 | 外部集成方负责 |
|---|---|---|
| 计算与路由 | 公开数学语义、dtype/shape/layout/SM 支持边界、CPU 权重流式执行、容差正确性、非法输入拒绝；保持旧全 GPU 路径 | 模型结构、router、业务前后处理、是否替换模型中的某一层 |
| 权重与配置 | canonical Tensor 契约、显式 `LayerKeys`/state_dict 适配、CPU 归档、冻结配置、权重身份/版本和预算校验 | 具体 checkpoint 键名与非 canonical 布局转换、权重来源可信性、业务模型/权重发布与热切换编排 |
| 生命周期与接入 | create/bind/validate/execute/stats/close 的可复用封装；明确逐次图外预检、绑定与执行串行边界；无 Python C++ 调用 | 组装完整模型图、其他层权重与输入绑定、选择官方 AOTi loader 并适配业务接口 |
| 构建与兼容 | 明确目标工具链/SM/schema，离线构建及显式预编译加载；部署加载失败可见、禁止隐式 JIT | 镜像/包仓库、签名/发布审批、业务 package 命名、灰度与回滚 |
| 资源与失败 | 单 context 的 RAM/pinned/GPU 预算、线程/stream/缓存隔离、排空释放、运行期失败可观察与 context 可复用边界 | 全进程/多实例总额、服务并发队列、其他层 reserve、NUMA 绑定策略，以及请求重试/进程重启 |
| 验收与观测 | 合成参数化正确性/容量/并发/故障/AOTi 矩阵，算子级正常性能与诊断、可复现命令及证据 | 真实请求 latency/QPS、业务模型误差、Recall/NDCG/排序稳定性及业务上线标准 |

仓库交付不能只把风险写成“调用方必须注意”：与算子正确使用直接相关的生命周期、绑定、逐次预检和失败检查应有易用的参考封装及测试；同时不尝试推断任意业务模型的参数顺序或实施服务恢复策略。对应设计见第 4.6 节，未完成项只在第 16 节登记。

## 2. 现有实现与可复用边界

### 2.1 输入与数学语义

| 符号 | 含义 | 当前布局 |
|---|---|---|
| `S` | 本次 MoE 前向的 token 行数，可由多个样本展平得到 | `x: (S, H)` |
| `H` | hidden size | — |
| `I` | 单专家 intermediate size | — |
| `E` | 当前层全局专家数 | `w1: (E, 2I, H)`；`w2: (E, H, I)` |
| `K` | 每个 token 的路由条目数 | `topk_ids/topk_scale: (S, K)` |
| `T = SK` | route 总数；重复专家条目仍分别计入 | per-route 中间结果共 `T` 行 |
| `m_e` | 专家 `e` 收到的 route 数 | 有效输入下，\(\sum_e m_e=T\) |
| `b` | 输入/权重/中间激活的元素字节数 | 首版 BF16/FP16，`b=2` |

对每个 token `s`，专家输出为：

\[
F_e(x_s)=W_{2,e}\bigl(\operatorname{SiLU}(W_{g,e}x_s)\odot W_{u,e}x_s\bigr)
\]

\[
y_s=\sum_{j=0}^{K-1}a_{s,j}F_{e_{s,j}}(x_s)
\]

`topk_scale` 是调用者给定的权重，不要求其和为 1，也不允许在各个 wave 内重新归一化。同一 token 的多个路由条目允许指向同一专家，不能因为去重搬运权重而漏掉这些加权贡献。

首版沿用 `H % 64 == 0`、`I % 64 == 0`、`E <= 512`、`K <= 128` 的计算约束，并显式检查正维度、路由值域、dtype、连续布局和目标设备一致性。尺寸乘积用有检查的 64 位计算；传入现有 int32 索引/参数前拒绝溢出。这些是当前 backend 的支持边界，**不是要求业务提前给出固定参数**；wave 内 group 数小于 512 也不代表全局 route kernel 已支持任意 `E`。

### 2.1.1 参数发现与模型版本

| 参数/变化 | 发现时机 | 处理方式 |
|---|---|---|
| `S` | 每次调用读取 `x.size(0)` | AOTi 以动态输入维度导出；依据本次大小重新检查 workspace |
| `K` | 每次调用读取 `topk_ids.size(1)` | 算子可运行时读取；整模型若 router 固定 K，则每个部署版本固定，不强求全图符号化 |
| `H/I/E` | 加载/注册各层权重时从 shape 推导，并与 `x` 核对 | 每层独立元数据与专家大小，不假定所有层一致 |
| 活跃专家、`m_e` | 本次真实 `topk_ids` | GPU 分组后在算子内部规划 wave；不要求预先提供路由 trace |
| 相同结构、新权重 | 新 `state_dict.pth` 到达 | 新 generation/权重版本；排空旧请求后重新绑定并清空旧 cache，满足制品契约时可复用 AOTi 程序 |
| 层数、连接、`H/I/E/K` 或 dtype 变化 | 新版本加载与图结构检查 | 重建元数据、重新 export/compile 对应模型；不承诺一个制品覆盖任意训练结构变化 |

动态 S 的导出范围仍有边界：可由服务配置的资源上限或输入分桶确定并随版本更新，不要求现在提供永久最大值；超出制品范围时明确拒绝或选择另一个制品。超预算的上层重新 batching 仅在模型允许按样本切分时使用，不能随意切断存在跨 token 依赖的序列。

开发阶段用多组合法合成形状及 uniform/Zipf/倾斜路由做参数化验证，真实 shape 和路由可在接入后采集用于调优；它们不再是开始实现的阻塞条件。

### 2.2 现有流水线

当前默认 CUDA 路径为：

```text
count/build_indices
    -> Gate/Up Grouped GEMM + SiLU·mul epilogue
    -> Down Grouped GEMM
    -> top-k weighted reduce
```

关键源码：

| 文件/符号 | 现有事实 | 对扩展的意义 |
|---|---|---|
| `csrc/fuse_moe/fuse_moe_op.cu` / `fuse_moe_cuda` | 五个输入均要求 CUDA；权重连续；分配本次 workspace | 不应仅删除设备检查来接收 CPU 权重 |
| `csrc/fuse_moe/fuse_moe_launch.h` / `moe_launch_sm8x_impl` | count、两个 GEMM、reduce 已有独立 launcher | 可新增分阶段入口，保留旧组合入口 |
| `csrc/fuse_moe/sm89/group_gemm_sm89.cuh` | 权重按本地 group ID 加固定偏移寻址 | 连续 GPU slot + 本地 group 映射适合首版 |
| `csrc/fuse_moe/common/moe_kernels.cuh` | expert-major compact 路由与 `topk_pos`；独立 `reduce_async` | 全局路由布局与最终归约可以保留 |
| `csrc/fuse_moe/common/fuse_moe_params.h` | tile、调度与 workspace 参数 | 新路径需要区分全局专家 ID、wave group ID 和权重 slot |
| `recsys.py` / `FuseMoeOps.register_fake_impls` | 现有 fake 输出形状为 `x.shape` | 新运行时不能仅靠同一个 fake 就获得导出支持 |
| `tilelang_ops/_fuse_moe.py` | count/route 对 expert ID 直接索引，未支持负 ID 屏蔽 | 不可直接复用 CUDA 的 `-1` 跳过策略 |

源码行号会随版本漂移，实施时以函数和实际布局为准。

### 2.3 不是零 kernel 改动

现有完整算子不暴露 per-route 输出和路由计划。正式流式路径至少需要：

- 在内部提供 route、compute-wave、finalize/accumulate 三个阶段；对外封装为一个新算子，不要求模型图或调用者自行循环 wave。
- 让 wave 的局部 group 使用已建立的全局 route 区间。
- 指定 wave 输出在全局 `down_out` 中的位置。
- 将 workspace 和 staging 生命周期交给运行时管理。

可复用的是 GEMM 骨架、compact 布局和归约算法，不是把完整 `ops.fuse_moe` 调用循环几遍就完成生产方案。

## 3. 设计决策

| 项目 | 首版决策 | 理由 |
|---|---|---|
| 专家计算位置 | 全部 GPU | 保持已有计算优势，避免同时开发 CPU 后端 |
| RAM 表示 | 与 GPU 相同的 BF16/FP16 权重 | 存储与精度优化分开验证 |
| GPU 权重空间 | 固定容量、连续 slot、两组 bank | 复用现有寻址并实现 copy/compute 重叠 |
| 专家加载顺序 | 基线按全局 ID 顺序分 wave，资源足够后可按实测代价调优 | 无需预测；不以严格确定性限制后续优化 |
| 同一层专家传输 | 每个活跃非驻留专家至多一次 H2D | 防止 token 分块导致重复搬权重 |
| 路由 | 每次调用只分组一次 | 避免各 wave 重做 `S*K` 的 count/build |
| 最终输出 | 基线 `route_buffer` 统一 reduce；低显存 `fp32_accumulate` 分 wave 累加后一次 cast | 容差验收允许改变求和顺序，减少 `SKH` 缓冲 |
| 持久热专家 cache | 首版不必启用 | Prefill 工作集可能很大，先建立流式基线 |
| 跨层内存管理 | 一个设备级池可供各层顺序复用 | 防止每层独立分配 staging 后再次 OOM |
| 后端 | 新路径首版明确使用 CUDA cp.async 家族 | sm89/sm120 已有可复用路径，暂不承诺 TileLang/TMA 扩展 |
| 导出与执行 | 一个 opaque 自定义算子 + 原生 C++ executor，AOTi 从 P0/P1 验收 | host 调度不能依赖运行期 Python，CUDA Graph 单独处理 |
| 权重接入 | `state_dict.pth` CPU 加载/布局适配 -> CPU canonical 权重 -> 显式算子输入 | 不把全量专家隐式编译/搬运为 GPU 常量 |
| 参数与额度 | 模型元数据/输入推导 + 部署配置 | 不固化业务 shape、卡容量或 RAM/pinned 额度 |

两组 bank 并不保证隐藏全部传输。一个 bank 能容纳但两个不能容纳时，允许降为单 bank 串行加载模式并报告实际配置。

## 4. 运行时架构与接口边界

### 4.1 分层

```text
state_dict.pth -> CPU 加载/键名与布局适配 -> CPU ExpertStore（图外初始化）
                                  │ CPU 权重、版本、资源配置
外部模型 / 合成验证 AOTi 图        │
    │ x、topk_ids、topk_scale、CPU w1/w2、runtime_token
    ▼
recsys_ops::fuse_moe_offload        对外一个 opaque 自定义算子
    ▼
PrefillMoERuntime（原生 C++）
    ├── ExpertStore          RAM 权重、身份和布局校验；后续可接 SSD
    ├── ResourcePlanner      GPU/RAM/pinned/workspace 预算与 wave 规划
    ├── RoutePlan            全局分组、route 位置和专家计数
    ├── TransferScheduler    host staging、copy stream、事件和生命周期
    └── GPU Compute Backend  route -> compute waves -> finalize/accumulate
```

`ExpertStore` 的专家身份应至少包含 `(model/weight_version, layer_id, expert_id, representation)`。不同层编号相同的专家不能混用；更新权重后，旧 plan、slot 和缓存必须失效。

CPU 在首版承担调度与存储职责，不执行专家 GEMM。初始化/注册权重应从 CPU 或文件来源进行，不要求先把完整模型加载到 GPU 再复制回来。

### 4.2 旧接口保持不变

`ops.fuse_moe(x, w1_cuda, w2_cuda, topk_ids, topk_scale, backend=...)` 继续代表全 GPU 常驻执行：

- 不隐式启用 SSD、CPU 计算或权重缓存。
- 不改变 `backend="cuda"/"tilelang"` 的含义。
- 不通过捕获 OOM 后临时改变语义来自动降级。
- 不修改当前 fake/schema，只为新增路径注册必要的独立接口。

### 4.3 新接口职责建议

以下名称和字段是提案，不是现有可调用 API：

| 对象/方法 | 职责 |
|---|---|
| `ExpertStore.register_layer(...)` | 接收 CPU 权重，检查 `E/H/I/dtype`，绑定权重版本 |
| `PrefillMoERuntime(...)` | 设置 device、GPU/pinned 预算、bank 数、compute backend |
| `runtime.forward(x, topk_ids, topk_scale, layer_id=...)` | 构建本次计划并执行，返回与 `x` 同 shape/dtype/device 的输出 |
| `runtime.stats()` | 提供本次传输字节、wave、等待、内存峰值等统计 |
| `runtime.close()` | 排空在飞工作，释放 pinned、GPU 和后续 I/O 资源 |

`cuda/tilelang` 是 kernel 后端；`resident/streamed` 是执行和存储策略，两者不能混成一个参数。Python API 是易用封装与导出期接入层，不是 AOTi 部署时必需执行的调度器；核心存储、wave、事件与计算调度必须在 C++ 实现。

当前一个 executor 的原生调用由 context mutex 串行化；各层可顺序共享 staging，返回结果为独立 Tensor。完整“预检→begin→执行→check”的公开接入层及多 context 隔离已完成，证据见第 15.15 节，任务关闭记录见第 16.2 节 T01/T06；不承诺共享 context 并行执行。跨 context 总资源额度与服务排队由外部集成方管理，不是本仓待开发的服务功能。

新增 GPU primitive 的 schema 应准确描述可变输出/workspace，而不是把持久运行时的状态变更伪装成无副作用 Tensor 运算。

### 4.4 对外单算子，不等于单 CUDA kernel

建议新增 `fuse_moe_offload`，不把 CPU 权重混入现有 `fuse_moe` 的语义。其内部可以执行多个 CUDA kernel、H2D、host 等待和 wave 循环；AOTi 图只看到该层一次调用。分阶段 launcher 用于复用与内部调试，不作为业务必需接口。

2026-10-09 的候选 schema（以下基础形式已经实现并通过 token mutation、export/AOTi 和独立 C++ 验收；包含冻结配置/Manifest 的现行扩展 schema 见第 15.13.2 节）：

```text
fuse_moe_offload(Tensor x, Tensor gate_up_weight_cpu,
                Tensor down_weight_cpu, Tensor topk_ids, Tensor topk_scale,
                Tensor(a!) runtime_token, int layer_id) -> Tensor
```

- `x/topk_ids/topk_scale` 是 CUDA Tensor；权重是只读 CPU canonical Tensor，初始化后由 store 持有。
- `runtime_token` 是显式 CPU 状态 Tensor：包含执行 context 的标识/generation，并按真实 schema 更新调用序号以体现有状态依赖；它不是输出或可由编译器忽略的未声明副作用。它作为部署输入绑定，不能将某进程的裸指针或 handle 常量烘焙进制品。
- `layer_id` 是图内稳定的层标识；context 校验设备、层、权重版本、Tensor 身份/版本及配置。更新权重必须走显式重新绑定，禁止在飞请求期间原地修改。
- 权重明确列为图输入，保证模型计算依赖可见；资源与 slot 属于 context。cache key 不能只看数据指针，必须包括权重 generation、层和表示。
- fake 只依据 `x` 推导新输出并校验可见 shape/dtype 约束，不读取 token 数值、权重内容或 `topk_ids` 数据，不进行文件 I/O/运行时查找。
- 真实输出是新分配 Tensor，与权重、token、staging 均不别名；内部 workspace 不作为未声明的图输出。
- 首版 context 单调用；mutating token 并非线程锁。并发靠独立 context/token 或有界串行队列，生命周期与 CUDA 在飞工作一起管理。

显式 CPU 权重输入会增加完整图的输入绑定数量。本仓接入层负责按显式映射绑定 offload 权重/token；完整模型其他输入和 `forward` 由业务组装，不承诺解析任意业务图。导出期需要将 offloaded 参数从模型常量/`nn.Parameter` 捕获路径提取为外部输入，部署期不执行对整个模型的无差别 `.cuda()`。否则即使专家 kernel 会 offload，AOTi 的常量初始化也可能提前耗尽显存。

已验收工具链能够保存 token mutation 和 CPU/CUDA 外部调用，P0 的候选通路风险已闭环（第 15.14 节）。仍保留显式 CPU 权重/token 输入，不采用纯 handle 隐藏权重模式；其他接口方向统一归入第 16.4 节，不在此另列计划。

### 4.5 `state_dict.pth` 权重加载契约

历史应用约定的权重来源是 `state_dict.pth`，即 `torch.save(model.state_dict(), "state_dict.pth")` 的直接保存格式；**这不是通用算子强制要求的文件名或唯一输入来源，也不代表当前已有可实验的业务 checkpoint**。当前 `ExpertStore(...)` 可直接接收 canonical CPU Tensor，`ExpertStore.load()` / `from_state_dict()` 提供可选的图外 CPU 加载与显式 `LayerKeys` 适配，`save_cpp()` / `load_cpp()` 提供 C++ 可加载归档。不要求业务转换为 GGUF，不在 MoE `forward` 内解析文件。完整 Python module/ExportedProgram 与业务 package 的加载由外部完成；本仓接入封装与合成验证分别见第 16.2 节 T01/T05。

- 使用 `torch.load("state_dict.pth", map_location="cpu", weights_only=True)`，校验 state_dict 键、所需 Tensor、shape、dtype 与布局；不自动回退 `weights_only=False`。
- 若实际文件额外套有 checkpoint 容器，应由适配配置显式指定 state_dict 字段，不靠猜测兼容任意容器。
- 键名及层/专家映射由模型适配器提供，不能从 shape 猜 gate/up 语义。兼容的 PyTorch 版本与序列化格式下可选 `mmap=True`，其内存边界见下文。
- `state_dict.pth` 是权重来源，**不是 AOTi 编译制品**；模型定义/forward 与权重分开接入，之后进行 export/AOTInductor 编译。

推荐数据流：

```text
state_dict.pth
    -> state_dict CPU 加载与权重键名/布局适配（图外）
    -> 每层 canonical w1:(E,2I,H)、w2:(E,H,I)
    -> layer metadata + model/weight generation + CPU ExpertStore
    -> AOTi 模型绑定 CPU 权重输入
    -> fuse_moe_offload 按当前路由加载活跃专家
```

加载与转换要求：

- 当前 `LayerKeys` 支持 canonical 3D fused gate_up/down 或分离 gate/up/down；后者按 gate 在前、up 在后沿 dim=1 拼接，dtype 转换须显式选择。expert list、转置布局或其他 gate/up 行序由外部模型适配器先转换，`from_state_dict()` 不猜测这些语义；不能只因矩阵是 `(2I,H)` 就认定其行序正确。已有 canonical/state_dict 适配及加载转换峰值的合成验证归入第 16.2 节 T05；具体模型布局与完整 checkpoint 峰值由集成方验收。
- `E/H/I/dtype` 从各层转换后的 Tensor 推导；键名、层映射、来源/目标 dtype、表示、schema/布局版本与权重 generation 随模型保存为加载元数据。不依赖永久固定的业务参数。
- 不默认允许 FP32/量化权重静默降为 BF16/FP16；由配置显式选择目标表示并进行数值验证。低 bit 来源若无相应解码 backend，应拒绝或在额度内显式解码。
- `map_location="cpu"` 避免 checkpoint 原 CUDA 保存位置触发全量显存分配；`weights_only=True` 不等于任意文件都安全或受资源限制，输入仍需可信来源、大小/shape 检查。
- 加载峰值包含 checkpoint storage、packing/转换输出和 pinned bank；能复用 canonical CPU storage 时尽量零拷贝引用。需变换的大文件分层/分专家转换并及时释放临时副本，不默认同时保留全部原权重和全部转换权重。
- `mmap=True` 只适用于受支持的 torch serialization 格式；它减少立即拷贝但不自动形成有界 RAM cache，不代表锁页或异步 I/O。RAM-only 模式仍按完整权重后备需求预检，不能偷偷依赖 major page fault 将 P4 提前包装为已支持能力。
- 原始 `state_dict.pth` 是以 storage 为单位的序列化容器，不默认是按专家连续可随机读取的 flat blob。SSD 模式必须核查 storage/offset/stride，必要时在图外派生版本化 expert extent 文件及索引；派生文件仅是运行时缓存/制品，上游来源仍是 `state_dict.pth`。P1 不要求生成这种文件。
- **图外 Python 适配不等于推理依赖 Python**：当前 `save_cpp()` / `load_cpp()` 使用标准 TorchScript Tensor buffer 归档，独立 C++ runner 通过 `torch::jit::load` 加载，format1/2 的具体契约已验收（第 15.13.4 节）。不是直接假定 C++ `torch::load` 能读取 Python 保存的 state_dict。
- 相同结构更新权重：排空旧 context/请求，校验结构/表示，绑定新 generation 并失效 plan/slot/cache；本方案先支持显式重新部署，不把训练中的原地更新与推理 cache 同时读写。

`state_dict.pth` 提供参数及持久 buffer，不提供可独立导出的模型架构/forward；业务 AOTi 导出需要业务提供模型定义或已有可转换的导出图。本仓可用合成单层/多层模块独立验证算子导出与接入，不以业务模型到位为前提。

### 4.6 面向外部集成的目标仓库设计

以下职责设计已按第 15.15 节落地；任务完成状态仅见第 16 节。构建/加载仍在通用框架，checked AOTi 版本适配位于 `moe_offload_aoti.py`，C++ 薄接入位于 `csrc/fuse_moe/fuse_moe_offload_runtime.h`，不新增业务 loader 或 serving 框架。

- **保持当前目录与分组结构**：`__init__.py` 的 `CustomOps` 负责通用构建/加载；`recsys.py` 保留分组门面与 fake 注册；`moe_offload.py` 放权重/冻结配置/导出适配及 Python 图外接入；`csrc/fuse_moe/` 放原生运行时与算子注册；`tests/` 放合成矩阵、官方 AOTi loader 驱动和独立 C++ 验证。沿用 `recsys_ops` 命名空间以保持兼容，不因定位调整全仓改名或引入新的 serving/package 框架。
- **三个边界**：图外做权重准备、context 初始化/关闭与失败检查；图内仅保留 `fuse_moe_offload` opaque 单算子及显式 CPU 权重/token 输入；GPU launcher、worker、bank/event/workspace 均为原生私有实现，外部无需自行拼装 wave。
- **图外接入层是算子级适配器，不是业务 loader**：Python 基于现有 `ExpertStore`、`OffloadConfig`、`PrefillMoERuntime`、`OffloadedMoELayer` 收敛入口；C++ 提供薄的 LibTorch/dispatcher 生命周期与作用域清理封装。调用者提供来自待执行制品的冻结契约、显式 layer→权重/token 绑定及调用入口/映射；封装管理同 context 的“预检→调用→失败检查”，不从 context 自己反推制品配置，也不猜任意整模型参数顺序。借助官方 AOTi loader 执行，不复制一套 loader。Python/C++ 具有对等的算子语义，但不承诺稳定 C ABI 或跨任意 LibTorch 版本兼容。
- **开发与部署加载分开**：保留开发期分组懒 JIT/源码缓存；部署期显式加载已构建 `.so`，缺失/不兼容即失败，不能悄悄 JIT 或换 backend。Python 导出端仍需注册 fake；独立 C++ 执行端只需原生注册，不需要 Python fake。旧 `.build_stamp` 是历史源码缓存机制，不是部署凭证；现行构建导出 `.so` 与 format2 `.so.json`，校验内容/构建/schema/目标工具链信息并禁止隐式 JIT，具体见第 15.15.1 节。通用构建/加载能力可复用，但本轮交付验收限定 `fuse_moe`/offload，不能据此宣称其他算子组 AOTi 已验收。
- **不耦合制品职责**：运行配置/全层 Manifest、CPU 权重归档、扩展构建兼容信息分开；AOTi package 的完整图和非 offload 输入属于外部。记录版本/schema/目标 SM 与已验收工具链并在可判断范围内前置拒绝，不以记录字符串保证任意 ABI 兼容。权重内容 hash/签名/发布标签的端到端认证不是算子计算必要条件，可选完整性补强见第 16.3 节 R02；不能在每次调用中扫描全部 CPU 权重进行 hash。
- **可复用而非测试装置外移**：现有 `tests/fuse_moe_offload_aoti_runner.cpp` 的 `LockedRuntime` 是固定合成图验证装置。提炼其生命周期/预检语义，不将固定输入生成、参考计算、模型层数假设复制到公开接口；接入示例只依赖公开能力和官方 loader，不能要求业务导入测试模块。

## 5. 路由计划与专家 wave

### 5.1 RoutePlan：全局 route 不随 wave 改变

使用真实 `topk_ids` 建立：

- `counts[e] = m_e`。
- `prefix[e]`：专家 `e` 在全局 compact route 空间中的起始行。
- `row_indices[q]`：全局 compact 行 `q` 对应的输入 token 行。
- `topk_pos[s,j]`：原始路由条目 `(s,j)` 对应的 compact 行。
- `active_experts`：`m_e > 0` 的全局专家，按 ID 顺序排列。

同一专家的全部 route 连续；没有命中的专家不产生 compute task，也不加载权重。重复 route 在 `row_indices/topk_pos` 中保留各自位置。

GPU 上的 count/build 是一次执行。首版可以将 `E` 个计数异步复制到小型 pinned host buffer，等待一次路由完成事件后由 host 规划 wave。此等待是明确的计划开销，不称为完全无 host 同步。无需把完整 `x` 或每个 wave 的路由表反复 D2H。

### 5.2 WavePlan：本地 group 与全局专家区分

每个 wave 最多装下 `G` 个完整专家：

- `expert_ids_local[g]`：本地 group `g` 对应的全局专家。
- 权重 bank：`w1_stage[g]`、`w2_stage[g]`，形状分别为 `(G,2I,H)`、`(G,H,I)`。
- `counts_local[g]`、`prefix_local[g]`、`tiles_local[g]`。
- `route_begin` 和 `route_count`：该 wave 在全局 compact route 空间中的区间。

按 ID 顺序连续划分活跃专家，使 wave 的 route 区间仍然连续。这样可以：

- Gate/Up 使用 `row_indices + route_begin` 和局部前缀，直接从原始 `x` gather-on-load。
- `act_wave` 使用局部 compact 行，大小按 wave 最大 route 数规划。
- Down 输出写入 `down_out + route_begin * H`。
- 最终 `topk_pos` 不变，直接定位各 wave 已写好的输出。

两个 GEMM 必须使用一致的局部前缀和行基址。将全局 expert ID 直接当 slot ID，或只平移权重指针而忘记 route 基址，都会产生看似正常但错误的结果。

基础流式路径按全局 ID 划分 wave；当前静态/动态 cache 已使用 route 映射支持命中与 miss 混合执行，不能将上述连续区间假设推广到任意调度策略。按 `m_e` 改变 engine/tile 或代价平衡重排属于第 16.4 节的可选方向，本节仅保留映射正确性的设计约束。

### 5.3 完整专家优先，不拆投影的依赖链

一个 slot 先存放专家完整的 Gate/Up 和 Down。该专家在本次调用中的全部 token tile 执行完成后才释放权重。

若一个完整专家加上最低 workspace 都装不下，首版在执行前报出需求和可用预算。分投影、分权重 tile 的流式 GEMM 或 CPU fallback 不在当前实现范围内（第 16.4 节），不能靠普通 wave 调度解决。

## 6. 传输与计算流水线

### 6.1 三段流水

RAM 权重没有全部 pinned 时，执行三段：

1. host worker：pageable RAM -> 有界 pinned bank。
2. copy stream：pinned bank -> GPU 权重 bank。
3. compute stream：等待权重 ready，执行 Gate/Up 与 Down。

若权重本身已经 pinned，可直接 H2D，省掉 host bounce copy；其 pinned 占用也必须纳入预算。首版不默认 pin 完整模型。

CPU 的 memcpy、H2D 的源读取和模型其他 CPU 工作可能竞争 DRAM。单独测出的 PCIe 或内存带宽不是协同负载下的保证。

### 6.2 双 bank 时间线

```text
Host staging: [准备 wave 0] [准备 wave 1] ... [准备 wave 2]
Copy stream:  [H2D wave 0] [H2D wave 1] ... [H2D wave 2]
Compute:          wait 0 -> [GU 0 -> Down 0] -> [GU 1 -> Down 1] -> ... -> reduce
GPU bank A:   load 0 ------ in use 0 ------- reusable ------ load 2
GPU bank B:               load 1 ---------- in use 1 ------- reusable
```

实际发射顺序由事件约束，图示不保证某次加载能完全落在前一 wave 的计算时间内。

### 6.3 事件与复用规则

| 资源 | 可复用条件 |
|---|---|
| pinned host bank | H2D 已完成；不能在 DMA 读取时覆盖 |
| GPU 权重 bank | 本 bank 上一次 wave 的所有权重读取已完成 |
| 局部 counts/prefix/tiles | 消费它们的 kernel 已完成；元数据同样需要 bank 化或有序复用 |
| `act_wave` | 对应 Down 已消费完；单 compute stream 下可顺序复用 |
| 全局 `down_out` | 最终 reduce 完成，且没有外部读者再引用 |

建议维护带 generation 的 slot 状态：`FREE -> LOADING -> READY -> IN_USE -> FREE`。状态描述不能替代 CUDA event；`READY` 必须对应已完成的权重和元数据传输。

首版：

- compute stream 使用调用方当前 CUDA stream，并保持输入生产和输出消费的顺序。
- copy stream 复用 GPU bank 前等待对应 `compute_done`。
- compute stream 使用权重前等待 `load_ready`。
- 异步工作排空前持有源权重、pinned buffer、GPU workspace 的引用；涉及跨 stream 的 Tensor 使用应正确管理 allocator 生命周期。
- 不在每个 wave 调用 `cudaDeviceSynchronize()`。
- PDL 仅在经过验证的 GPU kernel 依赖链内使用，不代替 H2D/event 依赖；首版可以关闭新路径的 PDL 建立基线，旧路径不变。

取消或失败后必须停止新发射，等待或安全回收在飞资源，不向调用方返回部分完成的输出。

## 7. GPU 与 RAM 资源预算

### 7.1 单专家大小

\[
W_e = (2IH+HI)b=3HIb
\]

| `H/I` 与表示 | 单专家权重 |
|---|---:|
| `2048/1024`，BF16/FP16 | 12 MiB |
| `4096/14336`，BF16/FP16 | 336 MiB |
| Strata `2560/640`，canonical Q2_0 | 1,382,400 B，约 1.32 MiB |

前两行是布局推导；最后一行来自本地 Strata 的固定 blob 契约。不能由专家个数相同推断 staging 容量相同，也不能照搬 Strata 的 ring slot 数。

### 7.2 `route_buffer` 模式的 runtime-owned GPU 内存

`gpu_budget_bytes` 限制运行时拥有的 GPU 权重、workspace、路由元数据和输出。**已返回但其 storage 尚未释放的历史输出仍计入本 executor 的额度**，不因当前没有在飞调用就归零；多层 alias/共享 storage 按实际所有权去重计价。外部预先分配的 `x/topk_ids/topk_scale` 不重复计入这一额度，但必须计入进程峰值和可用显存预检。

C++ runtime 需维护返回 output storage 的生命周期/字节账本，在 storage 真正释放后归还额度（跨 stream 在飞引用也必须排空），不能用 Python 引用计数或 `forward` 返回作为释放信号。调用者连续保留结果时，下次调用必须扣除历史输出；不能满足预算则显式拒绝。若实现无法可靠追踪这类 storage，应改为明确的调用方输出池/预算契约并同步修订公式，不能宣称硬额度已闭合。

对当前计划：

\[
M_{\mathrm{owned}} = M_{\mathrm{resident}} + N_b G W_e
+ b R_{\max} I + bTH + M_{\mathrm{route}} + M_{\mathrm{out}}
+ M_{\mathrm{live\ out}} + M_{\mathrm{other}}
\]

其中：

- `N_b` 是 GPU 权重 bank 数，默认尝试 2。
- `R_max` 是本次各 wave 的最大 route 数。
- `b R_max I` 是可复用的 `act_wave`。
- `bTH` 是基线 `route_buffer` 模式的完整 `down_out`；`fp32_accumulate` 使用第 7.5 节单独公式。
- `M_out = bSH` 是本次新输出；`M_live out` 是前序已返回且仍存活的 output storage 总量，两者不重复计价。
- `M_resident` 首版可为 0；后续常驻 cache 与 staging 分别计价。
- `M_other` 包括对齐、allocator 实际开销、事件相关缓冲和引擎额外空间，不只数 Tensor 的理论元素大小。

设备可分配余量还应扣除安全 reserve，并尊重调用方已有的模型权重、Attention/其他算子 workspace 和并发负载。不能把 `cudaMemGetInfo` 的瞬时空闲当作整次推理的保证。

`G` 变大不仅增加权重空间，也可能增加 `R_max`。规划器应对候选 `G/bank` 计算完整预算，而不是仅用“空闲显存除以专家大小”。失败时优先在执行前选择更小 `G` 或单 bank；若仍不满足则明确拒绝。

### 7.3 `route_buffer` 的容量边界

首版全局 `down_out` 仍随 `SKH` 增长。比如 `S=16384,K=8,H=4096,b=2` 时，仅该缓冲就约 1 GiB。

`route_buffer` 模式必须说明容量边界：只有当一组完整专家、必要路由、`act_wave` 和全局 Down 输出能够同时容纳时，才承诺执行。容差精度允许选择第 7.5/9.1 节的 FP32 累加与激活 tile 化，不应隐式使用交换或 GPU 页迁移。

### 7.4 RAM 与 pinned 预算

\[
M_{\mathrm{host}} = M_{\mathrm{weight\ store}} + M_{\mathrm{pinned\ staging}}
+ M_{\mathrm{metadata}} + M_{\mathrm{packing\ copies}}
\]

全 RAM BF16/FP16 后备的权重总量为各层 `E_l * W_e,l` 之和。pinned staging、布局转换副本、调用者尚未释放的原权重也是真实内存成本。

推荐：

- 初始化时完成布局准备，避免每 wave 临时转置/重新 packing。
- pinned 分配失败时，显式报告或选择受支持的降速传输模式，不假称 `non_blocking=True` 已实现 DMA 重叠。
- 多层共享 host staging，避免每层都保留一份大 pinned ring。
- RAM 不足时先明确拒绝；SSD 后备启用前不依赖 OS swap 兜底。

### 7.5 可配置额度与低显存累加模式

本节保留最初的资源配置设计，所有额度使用 bytes，日志以 MiB/GiB 展示。现行 LOCKED 部署将显式额度与执行参数固化到制品，同时在图外按同一配置创建 context；不能将早期“图外可调”的设计理解为可以修改已冻结制品，现行字段与语义见第 15.13.1 节。

| 配置项（提案） | 含义 |
|---|---|
| `gpu_budget_bytes` | 此 executor 的权重 staging、cache、workspace、元数据、本次及历史仍存活输出额度；外部其他组件的分配另由服务级预算协调 |
| `gpu_reserve_bytes` | 不可借给 MoE 的余量，覆盖其他层峰值/allocator/外部负载 |
| `ram_budget_bytes` | store、RAM cache、pinned、metadata、packing 等可计量 host 资源总额度；pinned 是其中一部分，不额外叠加一次 |
| `ram_reserve_bytes` | 服务其他组件、OS 和加载瞬时峰值的保留量 |
| `pinned_budget_bytes` | 总 host 额度中的锁页上限，含 DMA bank、counts 和其他 pinned metadata |
| `resident_cache_bytes` | 可选 GPU 常驻 cache 子额度，包含在 GPU 总额度内，默认可为 0 |
| `bank_count` | `auto/1/2`；auto 在满足完整预算时尝试双 bank |
| `activation_mode` | `route_buffer/fp32_accumulate/auto`；auto 只在已通过数值验收的模式间选择 |
| `max_route_tile_rows` | 激活 tile 行数上限；规划器可进一步缩小，不控制业务 S 的数学语义 |
| `over_budget_policy` | 默认明确报错；只能降 bank/G/cache 或切换已支持模式，不隐式 OOM 重试/交换 |

可提供 `auto` 额度：根据设备可用显存、Linux/cgroup 内存限额、当前 host 使用及 reserve 计算保守初值；一旦 context 创建便记录实际额度，再按每次需求预检。自动探测不能保证未来独占资源，调用方可显式覆盖，并通过服务级资源管理协调其他模型/算子的峰值。

精度不要求逐位一致后，可新增 `fp32_accumulate`：逐专家/route tile 计算 GU/Down，将每条 route 的原始 scale 加权结果 scatter-add 到 `(S,H)` FP32 accumulator，最终一次 cast 到输出 dtype。配合 tile 化后，粗略 GPU 预算为：

\[
M_{\mathrm{acc}} = M_{\mathrm{resident}} + N_bGW_e
+ bR_{\mathrm{tile}}I + bR_{\mathrm{tile}}H + 4SH
+ M_{\mathrm{route}} + bSH + M_{\mathrm{live\ out}} + M_{\mathrm{other}}
\]

其中 `M_live out` 与第 7.2 节相同，计入历史仍存活 output storage。上式是早期保留 dtype Down tile 的保守设计估算；**当前 `fp32_accumulate` 模式的 Down epilogue 已融合 weighted FP32 atomic accumulate，同时保留 FP32→输入 dtype→FP32 的舍入边界；该模式不分配全局 Down 或 Down tile**（`execution_plan()` 在累加模式下令 `sizes[Down]=0`，`group_gemm_sm89.cuh` 的 `kFusedAccum` 分支从 shared dtype 值加权累加）。`route_buffer` 模式仍保留全局 Down。因此不能再将融合 epilogue 列为待办；现行预算按实际 pool/charge 计算。初始化 accumulator 的 zero kernel、scale/route 映射和原子开销仍计入性能，不免费宣称更快。

- 相比完整 `bSKH` Down，accumulator 项是 `4SH`；忽略 tile 成本时 BF16/FP16 仅在 `K > 2` 才天然更小，不能对所有 K 默认选择。
- 单 compute stream 不能消除同一 kernel 中多个 route 写同一 token 的竞争；必须用正确的 FP32 atomic 或无冲突分段归约，不能普通写加。重复专家路由仍逐条计入。
- `M_route` 仍随 `SK/E` 增长；输入、输出与至少一个专家必须放得下。此模式减少激活压力，不等于无限 S 或任意巨大专家可运行。
- 减少 route tile 大小不释放该专家权重，优先在一次加载中处理完其全部 tile；G 的选择同时考虑 GPU 与 pinned bank 预算。
- 不得覆盖仍由 DMA/kernel 使用的 bank。当前没有原地修改 context 额度的管理 API；LOCKED 配置变化须生成配套制品并新建 context，不能沿用早期“更换额度无需重导出”的提案。

## 8. 性能模型与 wave 选择

忽略激活等非 GEMM 开销，一个专家的计算量约为：

\[
F_e \approx 6m_eHI
\]

相对于一次完整权重 H2D：

\[
\frac{F_e}{W_e}=\frac{2m_e}{b}
\]

BF16/FP16 下为 `m_e` FLOP/byte。这里的分母是 PCIe 搬运的权重字节，不是 GPU kernel 的全部 DRAM 流量，不能当作 GPU roofline 算术强度。

粗略估算：

\[
T_{\mathrm{H2D},e}\approx \frac{W_e}{B_{\mathrm{H2D,eff}}},\qquad
T_{\mathrm{compute},e}\approx \frac{6m_eHI}{P_{\mathrm{GPU,eff}}(m_e,H,I)}
\]

假设实测 H2D 为 25 GB/s，336 MiB 专家的纯传输约 14 ms。这是量级估算，不是当前机器测量。实际模型还需加入 host bounce、排队、两矩阵 copy、元数据、尾 tile 和同步开销。

整个层的时间包括：路由/规划、流水线启动、各 wave 的受限服务时间、收尾和最终归约。理想稳态由 host staging、H2D、GPU compute 中较慢的一段限制；实际重叠率必须实测。

基线使用 ID 顺序，资源预算由每次部署配置；波次参数按本次输入/路由动态推导。以下是性能分析原则，不是另一个待办清单；算子级测量见第 16.2 节 T04，输入覆盖见 T05，可选调度策略见第 16.4 节：

- wave 太小会增加 kernel、copy 和 event 固定开销。
- wave 太大会挤占 workspace，减少 pipeline bank 数或输入容量。
- 单个巨大专家可能占满 wave；同一专家的 token tile 应尽量在一次权重驻留中处理完。
- `m_e=0` 不加载；低 `m_e` 不自动交 CPU，先测是否真的划算。
- 以真实路由计算活跃专家平均和分布，不以全局 `SK/E` 代替全部判断。
- 必要时比较按估算计算时间平衡 wave，但重排前先完成非连续输出映射和数值验证。

衡量目标依次是：容量可用、正确、资源有界，再到吞吐和尾延迟；不承诺接近全专家常驻的速度。

## 9. 扩展设计依据：激活、缓存、SSD 与 CPU

本节保留方案取舍，不维护开发计划。FP32 累加/激活 tile 化及静态/动态 cache 已实现；其他方案的状态与是否立项统一见第 16.4 节，不能仅凭本节的实现要求认定它们属于当前待办。

### 9.1 激活分块：避免重复加载权重

当全局 `down_out` 或单 wave `act_wave` 过大时，可选择：

| 方案 | 特点 | 代价/边界 |
|---|---|---|
| 保留全局 Down，按专家 token tile 复用小 `act_wave` | 权重仍只加载一次，最终归约不变 | `TH` 的显存占用仍在 |
| per-route Down 放到 host 后备，按 token 块回 GPU 归约 | 可保留固定 top-k 顺序 | 增加输出 D2H/H2D、pinned 和带宽成本 |
| 各 wave/route tile 将加权贡献加到 `(S,H)` 的 FP32 accumulator | P2 优先实现的低显存模式，不保留全局 `TH` Down | 波次/原子顺序改变归约；容差验收，预算比较包含 tile 与 accumulator |
| token-outer 分块，每块依次扫描所需专家 | 容量实现直观 | 可能反复加载同一专家，H2D 成倍增加 |

不能同时无条件宣称“权重只传一次、任意激活规模可处理、零额外传输、完全保留原始求和顺序”。这些目标之间有真实取舍。

对于可调整的推荐模型 batching，减少 `S` 也是显式的容量策略；须区分算子内部 token 分块与上层重新分批执行整个模型。

### 9.2 跨调用/跨层常驻专家

Prefill 也可以从重复请求中的权重复用获益，但不应默认引入复杂热度预测：

- 先比较无持久 cache、静态放置和简单 LRU/LFU。
- cache key 包含层与权重版本；共享设备级预算。
- 为本次实际 wave 需求和 workspace 留出下限空间。
- 单次 Prefill 扫描大量专家时防止 scan pollution。
- cache 优化只减少加载，不修改专家集合或 scale。

Strata 为 Prefill 借用专家 cache 空间的做法可参考，但本项目没有 Decode 热集需要恢复；若无持久驻留专家，不必建立复杂的借还协议。

### 9.3 SSD 作为容量层

后续默认数据路径：`SSD -> 有界 RAM cache/staging -> pinned staging -> GPU bank -> GEMM`。

同一层活跃专家已经确定，可以提前读取下一 wave 的准确权重，不依赖预测。实现要求：

- 用索引定位专家的完整权重 extent，校验 shape、dtype、大小、版本和文件范围。
- 对确定的读取计划预取、合并相邻 I/O，demand fetch 优先。
- host buffer 在读取和 DMA 完成前均不可复用。
- 控制 RAM cache、I/O 队列和 pinned 峰值，不让文件 page cache 变成未计价内存。
- 明确短读、磁盘不足、文件损坏和取消的失败路径。
- `mmap` 不保证数据驻留 RAM，也不保证异步完成；major page fault 可能阻塞关键线程。
- direct I/O 与 buffered I/O 分开评测，考虑对齐和文件系统支持。
- `cp.async/TMA` 是 GPU 内部搬运，pinned H2D 是 PCIe 搬运；它们均不代表已经支持 SSD 直达 GPU/GDS。

SSD 首先扩展容量。连续流式读取可能有较好吞吐，但短请求和冷态 I/O 尾延迟仍可能很差。

### 9.4 CPU 专家计算仅作为可选优化

当短 Prefill 或大量低 `m_e` 专家导致搬权重不划算时，再比较：

- 权重 H2D + GPU compute。
- 输入激活 D2H + CPU expert compute + 输出 H2D。

必须按协同负载下的关键路径增量选择，而不是只比较孤立 kernel 时间。CPU 仍读取专家权重、竞争 DRAM；AMX/AVX-512 能力不代表所有消费级 CPU 都支持。

可评估本地 SGLang CPU kernels、外部 KT-Kernel 或通用 BLAS/oneDNN 基线，保持可选依赖和 ISA 探测。CPU/GPU 两支共用 per-route 身份及明确的最终归约契约，可统一写 per-route buffer 或按 scale 加入 FP32 accumulator；不简单各自低精度 reduce 后相加并宣称误差无变化。

### 9.5 量化与 offload 正交

首版不改权重精度。INT8/INT4、FP8 或 tile 内解码可降低 RAM/PCIe/VRAM 成本，但需要独立 packing、scale、CPU/GPU 格式和业务质量验证。

不能把已有 `mixed_gemm` 的 BF16 主项加 residual 补偿理解成专家权重压缩；它与低 bit 专家存储不是同一能力。量化还可能需要重复表示和临时反量化空间，预算必须按真实执行路径计价。

## 10. 正确性与兼容性

### 10.1 语义不变量

- 每条有效 `(s,j)` route 恰好执行一次，不漏、不重复、不指向另一专家。
- 权重和实际路由不因预算、缓存命中或传输完成时机而替换。
- 不跳过低 router score 专家，不修改 top-k，不在 wave 内重归一化 scale。
- compute-wave 不读取尚未到达或已经被覆盖的权重/元数据。
- `m_e=0` 不发射计算；空 wave 不进入 GEMM。
- 任何失败都不返回部分结果。

CUDA 现有 count 对负/越界 ID 的跳过只是内部行为，不能当成输入已被校验。首版新路径要求外部实际路由合法，内部 wave 使用独立映射；不依赖重复调用完整算子并将其他专家屏蔽为 `-1`。

### 10.2 数值契约

当前默认 cp.async 路径的关键边界为：

1. Gate/Up MMA 累加 FP32；SiLU·mul 在 epilogue 完成。
2. 激活写入输入 dtype 的 `act_out`。
3. Down MMA 累加 FP32，写入输入 dtype 的 `down_out`。
4. reduce 按原始 `j=0..K-1` 顺序，将 Down 转 FP32、乘原 scale、累加，最终一次 cast。

**算子数值契约为合理精度范围内等价，不要求严格可重复/逐位一致。**`route_buffer` 模式沿用以上 cast 边界用于排错；`fp32_accumulate` 允许专家顺序、tile 和原子累加顺序变化，不以 bitwise equality 作为上线门槛。

- GU/Down MMA 仍使用 FP32 累加，SiLU 与乘法沿用既有策略；scale 保持 FP32，最终输出按输入 dtype cast。
- 每条 compact route 需保存/恢复原始 `(s,j)` 或对应 scale，使不同 wave 都能乘正确 `a[s,j]`。同 token 重复 route 不被合并漏算，不在 wave 内归一化。
- P1/P2 首先保留 Down 写 BF16/FP16 的边界再累加；允许后续使用未舍入 Down FP32 值，但必须单独标识算术配置并验证。
- 正确性分别对拍现有全 GPU 路径与小形状 FP32 数学参考，避免仅比较两个共享 bug 的实现。报告 max/分位 absolute error、使用近零保护的 relative error、归一化 L2/RMSE、NaN/Inf，不能只看一个最大相对误差。
- `atol/rtol` 与汇总阈值按 BF16/FP16、K、输入尺度分别配置；在当前尚无真实分布时先作为开发门槛并记录，不给所有模型许诺统一误差数字。对抵消后的近零输出依靠 absolute error，不能过度放大 relative error。
- 参数化对拍覆盖不同 G/bank/tile、倾斜路由、重复 route、较大 K 与符号/尺度变化。允许数值随机性并不允许 slot race、错权重、漏 route 或失败返回部分结果。

更换 GEMM engine、CPU backend、量化等须由本仓独立验证数学误差、算术边界和性能/资源代价。外部集成方在真实模型上评估适用的 Recall/NDCG/排序稳定性等；这些不是本仓算子交付门槛，也不据算子容差测试承诺业务无损。

“循环调用完整 `fuse_moe` 后加总波次输出”仅适合作为原型：重复 count/reduce、低精度中间求和和全量 workspace 仍可能浪费；不要求 bitwise 并不意味着这是生产实现的最佳边界。

### 10.3 现有后端限制

- CUDA count/build/reduce 具有负 ID 跳过逻辑；TileLang count/route 目前没有同等 guard。
- CUDA `vert` 的任务映射使用总 tile 数作除数；全零 task 必须短路，不能直接进入该调度。
- 新路径首版固定 CUDA cp.async backend，不静默回退到尚未适配的 TileLang。
- TMA 的权重 descriptor、wave group 数和地址也要分别适配，不能只复用旧 descriptor 后修改 host 指针。
- 新路径不修改旧接口，也不把这些边界修复混入无关功能。

### 10.4 AOTi 是首版强制能力，CUDA Graph 独立评估

**AOTi 编译不要求把专家 wave 循环展开到图内，也不等于 CUDA Graph capture。**原生 C++ 自定义算子可以作为 opaque 外部调用存在于编译图中；动态路由、host 计数等待、权重 H2D 与事件调度在真实算子调用时执行。Python 中直接写 `.item()`/wave 循环不可追踪，不代表同样逻辑放进原生算子仍必然 graph break。

2026-10-09 设计时项目只有 schema/fake 和 `torch.compile` 基准；目前 offload 已完成严格 AOTi 导出、制品加载和无 Python C++ 部署验收（第 15.14 节）。fake 能返回 `empty_like(x)` 只是必要条件，不是这项完成结论的依据。

首版实施要求：

1. **预编译扩展**：构建 CUDA/C++ `.so`，独立提供不依赖 Python 初始化的 `TORCH_LIBRARY` 注册与 C++ runtime 管理接口。部署时提前加载，不能依赖首次调用的 Python JIT 编译。
2. **导出期**：加载 schema，显式确认 fake 注册成功；将专家 CPU 权重和运行时状态提取为外部输入，输出 shape 由 `x` 推导。动态 S 与关联 route 维度共享符号约束；K 是否动态由整模型 router 和目标版本能力决定。
3. **AOTInductor 编译期**：保持 offload op 不分解为 Python 规划，把它留作已注册 C++ 外部调用。验证目标版本生成的 custom-op fallback/proxy 路径及所需 metadata 的制品打包；若缺失则实现受支持的 lowering/shim，不以 Python callback 部署作为替代。
4. **部署期**：先加载扩展并创建 C++ context，再加载 AOTi 制品，绑定 CPU 权重/新 runtime token 与设备。运行时单次调用使用 AOTi 调用方的当前 CUDA stream，不捕获/保存编译时 stream 或进程指针。
5. **兼容性与错误**：明确锁定并记录 PyTorch、CUDA、编译器、扩展 ABI、目标 SM 与 schema 版本。读取配置、加载新权重在图外完成；过期 context、错误 generation、shape/预算不符应有明确错误路径。

P0 已通过本机 PyTorch `2.6.0a0` 的 `generate_fallback_kernel_with_runtime_lookup_aot` / `aoti_torch_proxy_executor_call_function` 通路及远程 `2.11.0+cu128` 的完整 C++ 调用闭环。调用链的异常传播边界见第 15.13.3 节，未闭环部分只在第 16.2 节 T02 跟踪。

强制验收：`export -> compile/package -> 新进程加载 -> 多个 S/不同路由/多 wave 执行`；同时在 C++ runner 中不启动 Python 解释器完成相同测试。检查没有全量专家 GPU 常量初始化、没有运行期编译、没有 Python host 调度；可变 token 的 functionalization、跨多层依赖、相同结构更新权重后 cache 失效必须覆盖。

首版流式 op 明确不支持整段 CUDA Graph capture/replay，应检测 capture 并报清晰错误，关闭该段的 capture 配置不应关闭 AOTi 编译。后续可以单独 capture 固定地址/容量的 GPU compute-wave 子图，再设计 host 调度与 replay 协同；Python 规划、SSD I/O 和 CPU future 不会随 replay 自动重新执行，PDL 也不解决这些依赖。

## 11. 实施阶段与验收

以下为原始里程碑及验收定义，不是现行开发排期。P0/P1 的通用算子与 AOTi 闭环、P2 的低显存机制、P3 的缓存机制已实现；P2 的现行版本重叠/性能复测及 P3 的合成热集/扫描行为复测仍跟踪于第 16.2 节 T04；输入/容量覆盖见 T05。P3 的真实业务路由收益由外部集成方评估，不再列为本仓待办。P4/P5 未立项，统一见第 16.4 节。

| 阶段 | 交付内容 | 验收重点 |
|---|---|---|
| P0：接口与 AOTi 闭环 | 最小 C++ offload op/schema/fake，CPU 权重/状态外部绑定，export/compile/load/C++ runner；合成参数成本基线与硬件探测 | 先证实目标 PyTorch/ABI 可无 Python 调用；无需真实模型固定 shape 才启动，记录 GPU/RAM/PCIe 成本 |
| P1：RAM 流式单算子 | C++ RoutePlan、连续 wave bank、单/双 bank、compute-only 与统一 finalize；state_dict.pth CPU 适配契约、参数推导、可配置额度 | 多 wave 的完整算子通过 AOTi/C++ 验收，容量超驻留层可执行，权重不重复传输，旧快路径不回归 |
| P2：流水线与低显存激活 | 持久有界缓冲、wave 调优、route tile 化、FP32 accumulator | 容差正确，无 slot/atomic race，两种激活模式预算完整，重叠有证据；同一 AOTi 单算子边界 |
| P3：可选驻留 cache | 设备级静态/LRU等基线及清晰准入 | 对真实 Prefill 请求降低搬运，不被扫描流量拖垮；权重更新后无 stale cache |
| P4：SSD 后备 | 从 state_dict.pth 适配为可索引 extent、有界 RAM cache、异步读、错误与取消协议 | 真冷/暖态分别测量，无隐式 swap/页缓存容量承诺；AOTi 调用无需 Python I/O |
| P5：按收益选配 | CPU 专家、低 bit、固定 GPU 子图 capture 等 | 每项单独证明正确性、业务质量与资源/性能收益；不是首次补 AOTi 支持 |

P1 不要求实现 P3-P5，但 **AOTi 不能推迟**。P2 若只缩小 `act_wave`，仍有 P1 的全局 Down 容量边界；低显存版本要同时实现 route tile 和加权 FP32 累加，再按完整预算选择模式。

## 12. 评价矩阵与观测指标

### 12.1 输入与路由

- `S/H/I/E/K` 用合成矩阵覆盖已支持的维度/对齐/索引及预算边界，不虚构未提供的“典型生产范围”。
- uniform、Zipf、single-expert、空闲专家、重复 expert、合成请求间热集变化与扫描污染。
- 多层合成图及可复现路由序列；均匀随机不能单独代表缓存效果，合成收益也不能当作真实业务收益。外部可补入实际路由 trace，但不是本仓验收前提。
- 专家容量：全部装得下、恰好一 wave、多 wave、活跃集合超过 staging、单专家装不下。
- workspace 容量：激活小、Down 接近预算、超预算明确拒绝；连续调用保留历史输出、跨层复用/alias 的 storage 不漏计或重复计价。
- BF16/FP16，sm89/sm120，不同 CPU/RAM/PCIe 配置。
- 当前单 context 串行调用；并发隔离验收统一见第 16.2 节 T06。

### 12.2 必须报告

| 维度 | 指标 |
|---|---|
| 算子完整调用 | 包含规划/搬运/计算/完成等待的 wall latency 分位数、calls/s、有效输入 token/s；业务 QPS/整模型 latency 不纳入本仓强制报告 |
| GPU 计算 | route、GU、Down、finalize 时间，wave 数，实际 tile/engine |
| 传输 | 权重 H2D bytes/call、bytes/token、传输次数、有效带宽、host bounce bytes/time |
| 等待 | route 规划等待、load-ready 等待、bank 回收等待、启动/收尾、可重叠比例 |
| GPU 容量 | 输入外部占用、输出、workspace、staging、resident cache、分配/预留峰值 |
| Host 容量 | RAM 权重、pinned staging、packing 副本、后续应用 cache 与 OS page cache |
| 缓存 | 若启用，demand/byte-weighted hit、驱逐、重复加载、scan pollution |
| SSD | 若启用，实际设备读取、队列与尾延迟、major faults、冷/暖状态 |
| 正确性 | route/权重身份、容差误差与 NaN/Inf；确定性仅诊断，不作硬门槛；算术改变时另报算子误差，业务质量由外部评估 |
| AOTi | export/compile/load/C++ runner、多动态 S、多 wave、外部 CPU 权重与状态绑定、无 Python/运行期 JIT、制品与 ABI 版本 |

本场景无 Decode，不使用 TPOT 或生成 token/s 作为主验收指标。token 吞吐须说明计算的是有效输入 token、是否包括规划与搬运；不能把 `S*K` route 数算成处理 token 数。

### 12.3 比较方法

- 同一形状、路由、精度和预算：全 GPU 常驻（能运行时）、串行按需 offload、双 bank 流水三者对比。
- 常驻无法运行的大形状，用显存足够的设备或小形状参考验证正确性；不得宣称不可运行基线的加速比。
- 全 GPU kernel 时间与包括 RAM/SSD/host 的端到端时间分开报告。
- 初始化/权重首次加载、首次调用、稳态调用分开；后续 SSD 对比不能只测 page-cache 热态。
- CUDA Event 用于 GPU 分段，CPU wall time 用于完整调用；计时包含调用前必要依赖和输出完成，不只数异步 enqueue 时间。
- 分段 profiler 可能扰动流水线，正常吞吐和诊断计时分别运行。
- 固定/记录 GPU 时钟与温度、RAM 通道、实际 PCIe 宽度/代际、CPU affinity、SSD 类型。
- 4090D 上限制 staging 为 4/8/12 GiB 可验证容量算法，不等价于真实 12 GB 卡的算力和带宽。

### 12.4 正确性验收用例

验收设计覆盖：最后一个不足 wave、仅一个活跃专家、多个重复 route、`K > E` 的合法重复路由、跨层同 ID 不串权重、权重更新失效、多次调用不同路由、不同 wave/bank 大小结果对拍、零任务短路、越界 ID/溢出输入拒绝、预算不足拒绝，以及资源分配/异步失败和取消后的排空。此处是用例定义，不代表每种失败均已实际触发；最新覆盖见第 15.14 节，尚缺的可控失败专项统一见第 16.2 节 T03。

旧全 GPU 路径在相同构建/环境下独立回归；没有测量证据时不改变其默认策略。

## 13. 调研依据与复用选择

### 13.1 本地源码依据

- [当前算子接口与 workspace](../csrc/fuse_moe/fuse_moe_op.cu)。
- [当前 CUDA 分阶段 launcher](../csrc/fuse_moe/fuse_moe_launch.h)。
- [compact 路由与固定序归约](../csrc/fuse_moe/common/moe_kernels.cuh)。
- [cp.async GEMM 与 group 权重寻址](../csrc/fuse_moe/sm89/group_gemm_sm89.cuh)。
- [TileLang 当前路由实现](../tilelang_ops/_fuse_moe.py)。
- [现有正确性参考](../tests/test_fuse_moe.py)。
- [Strata Prefill](../3rd/Strata/src/prefill/prefill.cpp)：专家 ring、copy/compute stream、事件、scratch 生命周期复用。
- [Strata Prefill 接口](../3rd/Strata/include/strata/prefill/prefill.hpp)：buffer sizing、借用 region 和 relayout。
- [SGLang offloader](../3rd/sglang/python/sglang/srt/utils/offloader.py)：参数存储迁移及跨模块 onload，不等价于 CPU 专家计算。
- [SGLang KT wrapper](../3rd/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py)：后续 CPU/GPU submit-compute-sync 的接口参考；依赖外部 `kt_kernel`。

本方案优先借鉴 Strata 的 **Prefill 存储/传输流水线**，不搬入它的模型几何、量化契约、Decode graph、MTP、QSA 或会话管理。

### 13.2 外部一手来源

| 来源 | 已取得的主要依据 | 对 Prefill-only 的位置 |
|---|---|---|
| [mixtral-offloading](https://github.com/dvmazur/mixtral-offloading) / [技术报告](https://arxiv.org/abs/2312.17238) | 分专家 offload、GPU LRU、HQQ；README 区分尚未提供的 prefetch 功能 | RAM->GPU 基线与缓存参考，不照搬量化和模型依赖 |
| [MoE-Infinity 论文 v3](https://arxiv.org/html/2401.14361) / [项目](https://github.com/EfficientMoE/MoE-Infinity) | 个人机器 batch=1 的 GPU 专家 cache；公开版与论文版不同 | 缓存与预取争用参考；Decode 局部性结论不能直接作为推荐 Prefill 假设 |
| [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html) | RAM/NVMe 权重逐层进入 GPU，预取与大 batch 摊销 | 流式与分层 I/O 原理参考，不采用其 dense 系统吞吐作为 MoE 性能承诺 |
| [FlexGen/FlexLLMGen](https://github.com/FMInference/FlexLLMGen) | 权重/KV/激活分层与 block schedule | 借鉴复用与 I/O，量化和大 batch 的效果单独评估 |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | CPU 专家后端与分阶段执行文档 | 仅后续可选 CPU 路线，不作为首版强制依赖 |
| [Fiddler](https://arxiv.org/abs/2402.07033) / [代码](https://github.com/efeslab/fiddler) | 移动激活与 CPU 专家执行思路；公开原型与论文版本需要区分 | 短 Prefill 的可选成本比较，不作为当前计算后端 |

外部材料在本次前序调研中部分取得，未在本项目复现；HOBBIT、ProMoE 未取得足够一手材料，不作为当前决策依据。本文不引用其他系统的 token/s 来预测本方案吞吐。

本地 Strata 为 MIT、SGLang 为 Apache-2.0。若复用源码，保留对应许可/版权声明并独立核查子组件；引用论文或借鉴思路不等于已许可任意第三方代码。首版不要求引入整个 Strata/SGLang serving runtime。

## 14. 已确认约束、目标硬件与最终建议

### 14.1 已确认，不再列为业务前置问题

1. **形态**：对外一个新自定义算子，内部 C++ runtime 分阶段/分 wave；旧 `fuse_moe` 保留。
2. **参数**：不同业务/模型版本可以不同；输入/权重自动发现并参数化测试，不要求现在提供准确 `S/H/I/E/K`。架构变化时重新导出，非任意制品无限动态。
3. **AOTi**：首版强制支持，P0 先验证外部算子链路，P1 必须端到端通过；不与 CUDA Graph 绑定。
4. **精度**：合理误差容差即可，不要求严格重复或 bitwise；优先为低显存增加 FP32 accumulator。
5. **资源**：GPU/RAM/pinned 上限与 reserve 可配置，auto 为保守初始化选项，按实际请求规划，不硬编码卡容量。
6. **权重**：核心接收 canonical CPU Tensor；可选接入 PyTorch state_dict 与 C++ Tensor 归档，键名/来源布局显式适配，支持显式新 store/context 权重版本切换；当前没有可实验的业务模型。
7. **硬件**：本机及 `remote.txt` 的 4090D/5090D；排除 V100。本文不复制登录凭据。

### 14.2 硬件与工具链依据（2026-10-09）

| 环境 | 2026-10-09 的已知事实 | 当时的方案位置与证据边界 |
|---|---|---|
| 本机 | RTX 4090 D，SM 8.9；`nvidia-smi` 总显存 24564 MiB，驱动 535.183.01；查询到 PCIe Gen4 x16 | sm89 主开发/测试；显存总量不是 runtime 默认预算，有效 H2D 仍需测量 |
| 本机 CPU/内存 | 虚拟化环境报告 Xeon Platinum 8576C、2 NUMA；进程 affinity 为 17 个逻辑 CPU，cgroup CPU quota 为 17 核等效；内存限额 85,794,488,320 B（约 80 GiB） | host copy/worker 按实际 affinity/quota 配置，不按拓扑中 160 个 CPU 创建线程；AVX-512/AMX flags 不等于 CPU 后端已经可用 |
| 本机软件/存储 | PyTorch `2.6.0a0`，torch CUDA/nvcc 均 11.8，`CUDA_HOME=/usr/local/cuda`；AOTi compile/load API 存在；工作区文件系统约 2 TiB | alpha 版本与 custom-op proxy/ABI 必须实测锁定；文件系统容量不代表 NVMe 或有效 SSD 带宽 |
| remote.txt 的 4090D | 文件标注 CUDA 12.8、4090D | 标注不是实时探测；验证驱动、torch、CUDA 编译器、CPU/RAM/cgroup、PCIe 与磁盘，不能直接沿用本机内存配置 |
| remote.txt 的 5090D | 文件标注 5090D | 目标 sm120，核验实际型号/SM/显存、完整工具链与 host 配置；不能从型号推导可用额度或 H2D 性能 |
| remote.txt 的 V100 | 已明确排除 | 不将 SM70、其 FP16 backend 或测试作为扩展验收范围 |

上述为 2026-10-09 的只读探测记录，不代表当前仍未连接远程；后续 SM89/SM120 工具链与性能验收见第 15 节，本机与远程 4090D 不假定是同一环境。本机当时的磁盘设备查询受系统库版本冲突影响，未确认 NVMe/转速；SSD 不在当前范围。目标工具链/加载与测量环境采集的统一状态见第 16.2 节 T09/T04，不在本表另列待核验任务。

本机 CUDA 11.8 可用于 sm89 路径，但不能作为 sm120 编译环境；5090D 需要支持该架构的 CUDA 12.8+ 工具链、匹配驱动/PyTorch/CUTLASS 与架构编译配置，实际兼容性须验证。外部 `.so` 和 AOTi 制品分别构建/验证 sm89/sm120 版本，不能默认一份制品跨架构、跨 CUDA/PyTorch ABI 通用。

### 14.3 外部集成契约与统一入口

仓库与外部业务的边界见第 1.5 节，接入结构见第 4.6 节。当前没有业务模型，所有仓库内待办必须可用合成权重/图/路由独立推进和验收；具体模型定义、checkpoint 布局、整图绑定、业务 reserve 和质量标准属于外部，不以“等待资料”保留为本仓必做任务。已验收工具链只证明所列算子/AOTi 环境可用，不能代替任意业务发布环境的验证。

**最终路线：先 P0 的 AOTi 原生单算子闭环，再 P1 的 CPU 权重按需加载 + GPU Grouped GEMM；P2 利用容差精度补齐 FP32 累加与有界激活。**

CPU-GPU 协同首版指 CPU 的存储、调度、搬运与 GPU 计算协作，不强制 CPU 执行专家 GEMM。只有测得低工作量专家或 RAM 容量确实成为瓶颈后，再分别引入 CPU 计算、SSD 或量化；不为不存在的 Decode 场景构建复杂运行时。

## 15. 开发日志与中断恢复检查点

### 15.1 记录规则与当前入口

本节在每个阶段完成后更新，不只在整项开发结束时记录。恢复工作时先读本节，再核对命令退出状态和现有产物；不要因 IDE 中断重复启动仍在运行的 GPU 测试。

- **已完成**：预算自适应切块、静态/动态缓存、有界 host staging worker、流水线诊断与独立 C++ AOTi 对照；新增不可变 `OffloadConfig`、图内全层 Manifest、format2 权重归档、图外部署预检与封存防降级保护。
- **本阶段进展**：offload 执行参数不再依赖环境变量，预算/Staging/Tile/SM 固化到制品；本机 SM89 和远程 SM120 各完成同步 107、异步 107 项回归、LOCKED 单层/真实双层 AOTi 与无 Python C++ 部署验收。最新结果及范围见第 15.14 节，不能用历史 88 项结果替代当前验收。
- **唯一待办入口**：[第 16 节](#16-统一待办清单与-review-结论)。本节及其他历史检查点不再维护下一阶段任务；现行接口见第 15.13 节、功能结果见第 15.14 节、历史性能见第 15.12 节。
- **完成范围**：上一开发阶段的实现与回归结束，不等于全部算子开发/接入/验收事项关闭；业务部署不是本仓完成门槛。验收缺口与未立项能力在第 16 节分开管理。
- 工作树含用户此前未提交的改动；本阶段不重置、不自动提交，不把这些改动误当成新增改动。
- 文档不保存远程登录凭据。远程连接配置仅沿用 `remote.txt`，测试结果应独立于连接凭据保存。

### 15.2 实现落点与配置契约

| 文件 | 已实现内容/恢复时关注位置 |
|---|---|
| `csrc/fuse_moe/fuse_moe_offload.cpp` | 原生 executor、资源池、动态缓存与 `execution_plan`；全层 `validate_deployment_manifest`、只读图外 `fuse_moe_offload_validate`、封存与 `seal_requires_manifest`；预算包含仍存活历史输出 |
| `csrc/fuse_moe/fuse_moe_offload.h` / `fuse_moe_bindings.cpp` | 原生管理与显式配置接口；前向支持可选 `deployment_config` / `deployment_manifest`；C++ `max_route_tile_rows=0` 表示预算自适应 |
| `moe_offload.py` | `OffloadConfig` 深度冻结、`PrefillMoERuntime.from_config` / `validate_config`、format1/2 归档；Python `max_route_tile_rows='auto'` 映射原生 `0`，数值 `0` 拒绝；`OffloadedMoELayer` 将配置/Manifest 导出为图内 `int[]` 常量 |
| `csrc/fuse_moe/sm89/group_gemm_sm89.cuh` | Down FP32 weighted atomic epilogue；连续列映射改善原子写访存；改动影响共享 kernel，原始全 GPU 路径须独立回归 |
| `tests/test_fuse_moe_offload.py` | 107 项合成权重/前端契约/原生回归；配置负向、全层拒绝前无副作用、只读预检、封存防降级、归档快照预算、缓存及跨 stream；`--async-staging` 选择异步默认 |
| `tests/test_fuse_moe_offload_aoti.py` | 严格 export/compile/load/C++ 与 LOCKED 单层/双层验收；`--async-staging` / `--host-profile` / `--nvtx` 为显式开关；保留 eager 和 `--benchmark-cpp` 对照 |
| `tests/fuse_moe_offload_aoti_runner.cpp` | 不启动 Python 的独立 C++ runner；`LockedRuntime` 校验 format2 后建 context，模型执行前使用冻结制品 Manifest 预检；`--benchmark-wide` 比较 `4096` 与原生 `0` |
| `csrc/fuse_moe/common/fuse_moe_utils.cuh` | offload launch 按当前设备动态匹配 SM90+ PSS，不受 legacy 环境覆盖或首设备缓存影响；旧全 GPU 路径保留原有诊断配置 |

默认/固定 cap 仍可用，本阶段不以合成测试结果强制全局切换为自适应。`auto` 追求在预算内减少 tile 数，不承诺最优延迟，也不等于 activation mode 的 `auto`。全缓存路径实际可以是零 staging bank，即使配置请求双 bank；对照时必须看运行时统计。

### 15.3 2026-10-13 验证检查点

| 验证 | 已确认结果 | 口径/限制 |
|---|---|---|
| 远程 CUDA 12.8 4090D 与 5090D | 两份任务均正常退出，严格 AOTi PASS；每卡 18 条独立 C++ benchmark、192 条 eager benchmark | GPU 型号的“双方验证”不是多 GPU 并行执行；使用各自构建的扩展/制品 |
| 独立 C++ AOTi | 每个配对均 `reference_checked=true`，50 个计时样本 | 合成 `H128/I64/E4/K2`，偏斜/屏蔽 route/有符号 scale，动态 S；不是完整业务模型 |
| eager 对照 | 固定 FP32、请求 banks=2、10 次 warm 调用 | static/reuse 的命中稳态实际 banks=0，disabled 实际 banks=2；冷态/初始化不混入 warm |
| 原生 offload 完整回归 | 本机 `Ran 80 tests` / OK，任务正常退出 | 既有任务 `5390b322-2393-4d8a-964e-b6b557d40dcc`；不是本阶段再次运行 |
| 自适应专项 sanitizer | memcheck 5 tests / OK、synccheck 2 tests / OK；均 `ERROR SUMMARY: 0 errors` | 既有任务 `f65fdefa-b80d-406a-91c0-3d1cef4d85a3`，覆盖长热路由、plateau、历史输出/缓存租约等边界 |
| Python `auto` 前端契约 | 本机标准 Python 直接运行契约函数通过 | 本机没有 pytest；第一次 pytest 命令失败，不能记为 pytest 通过 |
| 原始 `tests/test_fuse_moe.py` | 本机 21 PASS / 0 FAIL，BF16/FP16 均覆盖 | 同一已加载扩展执行；包含重复/稀疏路由及空输入拒绝，不证明所有生产形状 |
| 本阶段代码诊断 | 涉及的 Python/C++ 文件未报告诊断错误 | 诊断不是 CUDA 编译或运行时验收的替代 |

本机无 pytest 时，已成功执行的替代命令如下；仅直接调用这个不依赖 fixture 的契约函数，不用此方法替代完整 pytest suite。

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir python -c 'import runpy; module=runpy.run_path("tests/test_fuse_moe_offload_aoti.py"); module["test_route_tile_auto_frontend_contract"]()'
PYTHONPATH=/workdir python tests/test_fuse_moe.py
```

### 15.4 独立 C++ AOTi：固定 4096 与自适应对照

单位为 ms；`speedup = median(4096) / median(auto)`。这不是两个代码版本的对比。`gpu_charged` 是原生运行时计价峰值，不等于 allocator reserved 或全进程 GPU 峰值。

| GPU | S | cache | median 4096→auto | p95 4096→auto | speedup | tiles 4096→auto | gpu_charged MiB 4096→auto |
|---|---:|---|---|---|---:|---|---|
| 4090D | 8192 | disabled | 0.193214→0.152154 | 0.224399→0.171980 | 1.270× | 4→2 | 6.877→7.234 |
| 4090D | 8192 | static | 0.186588→0.121225 | 0.199383→0.138807 | 1.539× | 4→1 | 6.876→7.948 |
| 4090D | 8192 | reuse | 0.187728→0.141813 | 0.208251→0.159727 | 1.324× | 4→2 | 6.971→7.328 |
| 4090D | 16384 | disabled | 0.265999→0.161684 | 0.279133→0.180068 | 1.645× | 7→2 | 13.064→14.279 |
| 4090D | 16384 | static | 0.243081→0.136949 | 0.269818→0.153539 | 1.775× | 7→1 | 13.064→15.707 |
| 4090D | 16384 | reuse | 0.244785→0.150996 | 0.260881→0.168706 | 1.621× | 7→2 | 13.159→14.373 |
| 5090D | 8192 | disabled | 0.174471→0.137367 | 0.208664→0.161856 | 1.270× | 4→2 | 6.877→7.234 |
| 5090D | 8192 | static | 0.163301→0.113327 | 0.182845→0.152236 | 1.441× | 4→1 | 6.876→7.948 |
| 5090D | 8192 | reuse | 0.168731→0.148131 | 0.193586→0.174146 | 1.139× | 4→2 | 6.971→7.328 |
| 5090D | 16384 | disabled | 0.358507→0.264889 | 0.397461→0.308943 | 1.353× | 7→2 | 13.064→14.279 |
| 5090D | 16384 | static | 0.311222→0.191405 | 0.350402→0.226454 | 1.626× | 7→1 | 13.064→15.707 |
| 5090D | 16384 | reuse | 0.313217→0.219610 | 0.369772→0.286723 | 1.426× | 7→2 | 13.159→14.373 |

所有配对的 weight H2D、metadata H2D、promotion D2D 流量差均为零；50 次 warm 的 disabled weight H2D 合计 9.375 MiB、reuse 4.6875 MiB、static 0。hit/miss 计数也相同。因此大 S 的收益与少切块一致，但不能写成“减少了权重传输”。reuse 的 `fixed_cache_layout=false`，计数相同不代表布局严格相同。

S=64 的实际 tiles、内存、流量均不变；两卡三种缓存的 median speedup 在 0.984–1.017× 之间，最差约慢 1.62%。短输入无一致收益，不能宣称全 shape 零退化。5090D 在 S16384 的绝对延迟也没有稳定优于 4090D；硬件型号不能代替实测归因。

### 15.5 Eager：固定 FP32、请求双 bank 的边界

| profile / cache | 4090D warm median 4096→auto (ms) | speedup | 5090D warm median 4096→auto (ms) | speedup | tiles 4096→auto |
|---|---|---:|---|---:|---|
| balanced / disabled | 0.417773→0.389460 | 1.073× | 0.242404→0.217905 | 1.112× | 4→3 |
| balanced / static | 0.147436→0.112847 | 1.307× | 0.116217→0.096408 | 1.206× | 2→1 |
| balanced / reuse | 0.146167→0.120900 | 1.209× | 0.117687→0.093952 | 1.253× | 2→1 |
| weight_transfer / disabled | 1.037910→1.025151 | 1.012× | 0.423075→0.425965 | 0.993× | 3→3 |
| weight_transfer / static | 0.107662→0.108985 | 0.988× | 0.078404→0.084164 | 0.932× | 1→1 |
| weight_transfer / reuse | 0.104048→0.105728 | 0.984× | 0.078633→0.080763 | 0.974× | 1→1 |
| activation_near_budget / disabled | 0.770413→0.554229 | 1.390× | 0.484544→0.330037 | 1.468× | 13→3 |
| activation_near_budget / static | 0.506357→0.306479 | 1.652× | 0.354256→0.212566 | 1.667× | 11→1 |
| activation_near_budget / reuse | 0.506894→0.305433 | 1.660× | 0.355111→0.215004 | 1.652× | 11→1 |

此表的实际 activation mode 始终为 FP32（1→1），没有混入模式切换的收益。static warm 为 8 resident experts，reuse 为 6 dynamic experts，两者每调用 6 hit / 0 miss、0 weight H2D；disabled 为 0 hit / 6 miss。两种 tile 配置下缓存状态与权重流量一致，warm 无预算驱逐。

`activation_near_budget` 的运行时预算为 60 MiB，自适应的 charged 峰值增加约 18.61–19.94 MiB；最大的 allocator allocated 峰值由 32.127 增至 52.127 MiB。以上为这组单层合成调用的空间/时间权衡，不等于多层并发的进程预算保证。

`weight_transfer` 的 tiles、峰值及 warm cache state 不变，5090D static 实测慢约 7.35%、reuse 慢约 2.71%。仅 10 个 warm 样本，不足以判断显著性，但必须保留实测退化；profile 名称也不是 profiler 已确认的瓶颈。当时提出的时间线归因已在第 15.7–15.12 节开展，现行合成性能复测统一见第 16.2 节 T04。

### 15.6 历史证据索引与计划归档

本轮结果来源：2026-10-13 的两份正常退出任务。4090D 与 5090D 的任务标识分别为 `6ee46042-191e-43d0-a7bc-60fa1137fa10`、`6b5b6977-96ca-429d-9474-78d72c822c56`；两份输出中独立 C++ benchmark 位于 1986–2020 行，eager benchmark 位于 2025–2220 行。ID 和行范围用于恢复时定位原始 JSON，不复制登录信息；长期证据留存统一跟踪于第 16.2 节 T08。

当时的时间线归因、供给路径诊断及按证据选择试验已在第 15.7–15.12 节完成阶段闭环；输出并存与 cache/workspace 预算已有回归。这些历史结果不替代现行版本的合成性能与支持范围验收，仓库内缺口见第 16.2 节 T04/T05；真实业务模型验收属于外部职责，见第 16.4 节。

### 15.7 流水线诊断：采集检查点

本机 Nsight Systems `2022.4.2.1` 可用。2026-10-13 已执行下述短诊断，任务正常退出（约 11 秒），随后导出标准 SQLite 并完成 warm NVTX 范围分析；不将进程初始化或 FP32 数学参考的 kernel 混入算子成本。采集任务标识 `4e197657-0980-4ab8-9d64-193f01bbd41d`，汇总导出任务标识 `de8dab7f-dd74-41ac-b06f-87f1b8e7a898`；分析结论见第 15.8 节。

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --stats=false --output=/tmp/fuse_moe_pipeline_20261013 python tests/test_fuse_moe_offload_aoti.py --ops-library /workdir/custom_ops/torch_extensions/recsys_fuse_moe_kernel/recsys_fuse_moe_kernel.so --nvtx --benchmark --iterations 3 --benchmark-profile weight_transfer --benchmark-scenario skewed_masked --benchmark-route-tile-rows auto
```

注意：输出路径不覆盖已有报告；恢复时先检查现有任务与文件。`/tmp` 的诊断产物不是长期归档，归因结束后清理 `.nsys-rep` 和 `.sqlite`，本文保留结论与复现命令；需要长期保存原始报告时另行指定目录。该诊断跑完整 mode/bank/cache 矩阵，但只用目标 warm 范围计算阶段指标。

源码已确认：miss wave 的 pageable→pinned `std::memcpy` 在调用线程执行；GPU 权重与元数据 H2D 走 copy stream，compute stream 等待 `ready`；bank 再用前 host 等待 `done`（含 promotion 完成）。这允许重叠，但不代表实际重叠充分；未经时间线分析不能直接引入后台线程或放松事件依赖。

### 15.8 流水线诊断：结果与归因边界

**环境与口径**：本机 RTX 4090 D、CUDA build 11.8、FP16、`S128/H512/I512/E16/K8`、`max_route_tile_rows='auto'`。这里只比较固定 `fp32_accumulate`，每组 3 次 warm，选取精确 NVTX 标签 `moe/weight_transfer/skewed_masked/fp32_accumulate/bank{1,2}/{policy}/skewed_masked/layer0/skewed_masked`。不是 CUDA 12.8 远程结果，也不是两卡并行执行；以下诊断延迟不作为正常性能或统计显著性的依据。

| 请求配置 | NVTX median (ms) | kernel 合并活跃 (ms) | weight H2D 活跃 (ms) | kernel ∩ weight H2D (ms) | `cudaEventSynchronize` API 累计 median (ms) |
|---|---:|---:|---:|---:|---:|
| bank1 / disabled | 1.581371 | 0.059744 | 0.379809 | 0 | 0.253219 |
| bank2 / disabled | 1.371919 | 0.083264 | 0.382690 | 0 | 0.003746 |
| bank1 / static_hot | 0.212440 | 0.048832 | 0 | 0 | 0 |
| bank2 / static_hot | 0.209129 | 0.048768 | 0 | 0 | 0 |
| bank1 / reuse | 0.213480 | 0.048864 | 0 | 0 | 0 |
| bank2 / reuse | 0.246182 | 0.048672 | 0 | 0 | 0 |

- disabled 的实际配置为 bank1/cap4/2 waves 与 bank2/cap2/3 waves；因此这不是仅修改 bank 数、保持 group 大小完全相同的对照。每次均搬 9 MiB 权重，另有 112/120 B 元数据；DMA 活动时间对应约 24.8/24.7 GB/s，但这不是包含 host 准备的端到端供给带宽。
- static/reuse 的实际配置均为 0 bank/1 wave，6 hit/0 miss；请求 bank 参数不代表它们实际进行了双缓冲。warm 没有权重 H2D 或 D2D promotion。
- 所有组每调用仍有 68 B D2H（2 次）；“没有权重 H2D”不等于没有路由检查/计数等待。
- disabled 的 `cudaStreamSynchronize` API 累计 median 为 bank1 0.137348 ms、bank2 0.136503 ms；其等待与 GPU 传输/执行并不互斥，不能把 API 耗时再加到 GPU 时间上作为独立阶段总量。

**实际供给空窗**：合并本进程 kernel、memcpy、memset 后，disabled bank1/bank2 的 GPU 活跃 median 为 0.447841/0.475137 ms；NVTX 内没有上述活动的时间为 1.133724/0.896782 ms。此处是被记录进程的区间覆盖率，不是整块 GPU 硬件利用率。

代表性的第二次 warm 中，bank2 第一波 Down 在相对 NVTX 起点 0.623497 ms 完成，下一波权重 H2D 在 0.797865 ms 才开始；第二波 Down 在 0.962058 ms 完成，第三波 H2D 在 1.102730 ms 才开始。两段 GPU 空窗分别为 0.174368/0.140672 ms。各组 3 次 warm 的 kernel–weight H2D 重叠均为 **0 ns**，不能继续仅凭 copy/compute stream 的存在宣称重叠已充分实现。

在每波 weight H2D 提交前，前一 CUDA API 返回至 H2D API 开始的 host 无 API 窗口累计 median 为 bank1 0.882800 ms、bank2 0.898361 ms。源码在该位置准备 pinned bank 和元数据，但 **本次未采 libc memcpy/OS runtime 事件**：这只是 host 准备/提交空窗，不是 `std::memcpy` 的精确耗时，可能包含 CPU 调度及其他未记录工作。

**本阶段判断**：此合成 disabled workload 是权重供给路径（host 准备/提交、H2D、单 bank 复用等待）主导，不是 GEMM 主导；bank2 减少了复用同步等待，却尚未使后续 DMA 提早覆盖上一波计算。该结论仅适用于此 trace，不解释 5090D 的波动，也不能据此保证后台搬运线程必然加速。

**分析方法**：只读标准 SQLite，NVTX 按完整标签筛选；同进程/设备的事件按半开调用区间裁剪，排序合并活跃区间，计算 kernel 与 weight H2D 合并区间的交集。18 个选定范围没有跨边界事件。weight H2D 按本 fixture 的大传输识别，并与原生 9 MiB 权重流量精确核对；GU/Down 用完整 kernel 模板参数区分，不能按共同的 `group_gemm_gateup_kernel` 短名字合并。各项单独取中位数，表格列不能任意相加得到 wall time。

### 15.9 历史开发计划的完成映射

原分散在此处的开发顺序已由后续阶段执行，不再作为当前待办：

| 当时的计划 | 当前核对结论 | 完成依据 / 剩余事项入口 |
|---|---|---|
| 直接观测 host staging | 已实现默认关闭的原生 HostStage 与 benchmark 阶段汇总 | 第 15.10–15.11 节 |
| 依据归因试验供给改进 | 连续 memcpy 试验已撤回；有界 worker 已实现，保留同步默认 | 第 15.11–15.12 节 |
| bank/权重生命周期、预算、异常/close/drain 机制 | 机制与正常关闭/校验失败恢复已实现；不等于任意执行中失败均有证据 | 机制见第 15.12.1 节；可控故障专项仅在第 16.2 节 T03 跟踪 |
| 本机及远程正确性、正常对照和独立 trace | worker 历史版本已有验收；冻结配置版本只有第 15.14 节所列功能验收 | 现行版本合成性能、支持范围及目标矩阵补测见第 16.2 节 T04/T05/T07；真实业务模型验收属于外部职责，见第 16.4 节 |

当前固定 cap、同步默认和缓存默认策略保持不变，不因归档历史计划而增加新优化承诺。

### 15.10 Host 阶段直接诊断：实现与测量

> 本节性能与任务记录属于引入显式配置前的历史版本。文中 `FUSE_MOE_OFFLOAD_PROFILE` 环境开关已移除，当前 API 为 `host_profile=True`，验收/benchmark 使用 `--host-profile`；旧数据不因配置入口变化而视为重新测量。

2026-10-13 接续开发：已加入默认关闭的 `FUSE_MOE_OFFLOAD_PROFILE=1` 原生阶段标记，在 context 创建时采样，forward 内不反复读环境变量。基于 PyTorch C++ `RecordFunction(USER_SCOPE)`，不新增算子 schema、stats 字段、CUDA timing event、GPU/pinned 缓冲或 NVTX 链接依赖；可由 CPU profiler 或 `emit_nvtx` 观察。禁用时不创建 RecordFunction，不调用 observer 回调。

阶段名：`fuse_moe_offload::host_weight_copy`（pageable→pinned）、`host_metadata`、`bank_wait`、`h2d_submit`、`compute_submit`、`drain`。submit 标记表示 host API 时间，不等于 GPU 执行时间；bank_wait/drain 内等待和 GPU 活动可能重叠，不能把所有列相加为总延迟。

新增两项回归覆盖：环境开关精确匹配/创建时锁定、31 项 stats 不变、单/双 bank 和多 tile 事件计数；动态缓存晋升/全命中、空路由、零输入、非法路由恢复、跨 stream。编译任务 `3f257cfa-7718-4484-942e-981b17fe37d0` 正常退出，`Ran 2 tests` / OK，原生增量编译 9.4 秒。最终逐专家拷贝版本任务 `d4544985-e3b6-4437-b0ba-5db19366b6f3` 已完成 `Ran 84 tests` / OK，以及 export/compile/Python AOTi/独立 C++/shared-token chain 验收；环境开关为 1 时的无 profiler C++ 部署正常。该串行任务在 memcheck 启动后以 `137` 退出，无 ERROR SUMMARY，后续 synccheck/原始 MoE/最终诊断未执行，**不记为 sanitizer 通过**；后续独立补跑已通过，详见第 15.11 节。此前试验版本任务 `048cb817-9387-4b81-aa0c-c6d8613af39c` 为 83 tests + AOTi + memcheck 3 tests 全通过，但不能代替最终版本证据。

直接采集任务 `83174202-6b09-4a85-a57e-0d5460ed27c3` 已正常退出：完整 24 组 mode/bank/policy 矩阵，每组 10 warm。固定 FP32、disabled、同第 15.8 节形状/路由/预算，每调用 9 MiB：

| 请求 banks / 实际 cap / waves | host weight copy median / p95 (ms) | host metadata median (ms) | bank wait median (ms) | H2D submit median (ms) | compute submit median (ms) | drain median (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 4 / 2 | 0.844552 / 1.397537 | 0.000567 | 0.253613 | 0.040874 | 0.035031 | 0.125267 |
| 2 / 2 / 3 | 0.857018 / 1.429803 | 0.000526 | 0.004745 | 0.054614 | 0.048580 | 0.132111 |

这是每 forward 内同名范围的累计 CPU wall 时间，不是单次 memcpy 时间，包含范围计时成本和可能的 CPU 调度。权重 copy 只包住 pageable→pinned 循环，不混 metadata/CUDA API；有效供给约 11.2/11.0 GB/s。与此前约 0.9 ms host 空窗相符，但不能跨两轮 profiler 直接相减。static/reuse/lru warm 全命中，只有 compute submit/drain，没有 miss 阶段。诊断 wall median 2.009/1.937 ms 明显受 profiler 扰动，不用于性能收益结论。

**数据驱动的第一项试验及决策**：曾对当前 miss wave 内源 expert ID 严格连续的 run 合并 CPU memcpy，独立 w1/w2 各一次，w2 目的偏移始终为 `capacity*w1_bytes`，不跨空洞、wave、resident 或 bank 租约。正确性与 memcheck 通过，但单轮正常延迟及直接 staging 数据未证明稳定收益；**已撤回该试验，最终代码保留原逐专家 memcpy 和默认关闭诊断**。不减少流量、内存或分配，减少 memcpy 调用数本身并未解决约 0.85 ms staging 成本。无 worker/预取/额外缓冲，没有新的流水线重叠结论。详见第 15.11 节。

当前可复现直接采集入口：运行 `--host-profile --benchmark --benchmark-profile weight_transfer --benchmark-scenario skewed_masked --benchmark-route-tile-rows auto --iterations 10`，显式启用原生标记与 benchmark CPU profiler，不再设置旧 PROFILE 环境变量。每调用独立 CPU profiler，`trace.*[].host_stages` 保留阶段调用次数和累计 CPU ms，warm summary 汇总阶段 median/p95/total 和调用次数。启用 profiler 后的 wall time 明确标为诊断值，不与正常延迟混用；正式 31 项 runtime stats 不变。

当前 `--nvtx` 的直接 NVTX range 可与 CPU profiler 并用；`torch.autograd.profiler.emit_nvtx()` 是独立观察模式，不应与 CPU profiler 嵌套。原生 `host_profile` 在 context 创建时固定；不开 profiler 的 standalone C++ 即使该配置为 true 也不依赖 Python observer。setup 的可传播诊断异常会被忽略；observer 回调遵循 libtorch 的错误处理契约，不承诺隔离进程信号、OOM、任意 observer 崩溃或重入同一 context。

### 15.11 连续拷贝试验、最终代码与验收收尾

**环境与对照**：本机 RTX 4090 D、PyTorch CUDA build 11.8、`S128/H512/I512/E16/K8`、合成偏斜/屏蔽/有符号 route、固定 FP32、tile auto、disabled cache、相同 GPU/RAM/pinned 预算。正常 benchmark 不开 profiler，各组 40 warm，初始化/冷调用/priming 单列。本轮不是远程 CUDA 12.8，也不是两卡并行。

| 配置 | 逐专家 baseline median / p95 (ms) | 连续 run 试验 median / p95 (ms) | 直接 weight staging median 前→后 (ms) |
|---|---:|---:|---:|
| bank1 / cap4 / 2 waves | 1.580220 / 1.883879 | 1.675881 / 1.835570 | 0.844552 → 0.888413 |
| bank2 / cap2 / 3 waves | 1.328792 / 1.717538 | 1.315132 / 1.696877 | 0.857018 → 0.841753 |

baseline 任务 `5026509d-35bc-4ea7-9be1-9f9ceee409cf`；试验正常 benchmark + 10 warm 直接诊断 + 84 tests + 原始 21 PASS 任务 `272eb231-d54d-4d27-a7c8-f8e8c86d3eba`。每个正常矩阵 24 组，输出与数学参考和流量/预算契约检查均通过。只有一次前后运行，未交错复测、未建立置信区间；bank1 median 变慢约 6%，bank2 变化约 1%，staging 同量级，**不足以证明稳定收益，因此撤回连续 run 分支**。保留对应稀疏/连续、部分 wave、resident 边界测试，作为后续供给调整的回归保护。

最终交付范围：默认关闭的原生 `HostStage`、六个 CPU 阶段、benchmark 可选采集与事件数量核对、汇总单位/补零/混合样本拒绝契约，以及四项新增原生回归；不改变正式 stats、预算、cache 策略、bank 事件依赖、逐专家权重复制或增加缓冲。分离 weight copy 与 metadata 循环用于测量，不是异步 worker。

**最终验收检查点（已完成）**：`d4544985-e3b6-4437-b0ba-5db19366b6f3` 的 84 tests 和完整 AOTi 已通过，随后 memcheck 启动阶段被终止（137），缺少工具错误汇总；原因为未知。恢复时资源检查可用内存 74149 MiB（约 72.4 GiB）、GPU 空闲但已有约 6 GiB 服务占用；`dmesg` 无读取权限，不能据此宣称 OOM。独立补跑正常结束，最终代码证据如下：

| 验收 | 最终结果 | 任务 / 边界 |
|---|---|---|
| offload 完整回归 | 84 tests / OK | `d4544985-e3b6-4437-b0ba-5db19366b6f3`，包含 4 项新增原生回归 |
| export + Python AOTi + 独立 C++ + shared-token chain | PASS | 同上，`FUSE_MOE_OFFLOAD_PROFILE=1 FUSE_MOE_TILE_N=64`，验证无 Python observer 的 standalone 路径 |
| memcheck 专项 | 3 tests / OK，0 errors | `f0533087-4f15-4623-a102-0f074baafd10`，连续/稀疏/部分 wave/resident、开关事件计数、缓存/跨 stream/失败恢复；工具与 CPU profiler 共用 CUPTI 时出现 multiple subscribers warning，CUDA profiler 活动缺席不当作 GPU trace |
| synccheck 专项 | 1 test / OK，0 errors | `f3a5f843-c046-4d5d-9e68-b1f6e4f857ca`，两种 dtype/两种模式/单双 bank/缓存交错 |
| 原始全 GPU MoE | 21 PASS / 0 FAIL | 同上，最终库，FP16/BF16 与重复/稀疏路由 |
| 最终阶段诊断 + 直接 NVTX | 24 组完整矩阵，3 warm/组，事件数量/流量/预算/参考全部核对 | 同上，CPU profiler + direct NVTX；不是新的 GPU overlap trace |
| CPU-only 契约 + 格式检查 | PASS | 同上，直接运行 summary 与 auto 前端契约，无 pytest；`git diff --check` 通过 |
| 编辑文件诊断 | 无 linter errors | 最终 `fuse_moe_offload.cpp` 与两份测试文件 |

本阶段任务已全部结束；未提交 git 改动，也未创建需保留的临时 helper/脚本或原始 trace。后续 worker 的 CUDA 12.8 4090D/5090D 验收见第 15.12.3 节，不再把当时的远程验证计划列为未完成任务；本节结果本身仍只适用于本机。

**已落实的 worker 设计约束**：单 context 有界 staging 队列、pinned/GPU 分离释放点、promotion 依赖、正常 close/drain、预算 reservation 与跨 stream/新权重 context 边界见第 15.12 节。当前开发入口统一为第 16 节，不在此重列性能或故障验收计划。

### 15.12 有界 host staging worker：实现、最终验收与保守默认

> 本节 88 项回归、性能和 trace 是 worker 阶段的历史检查点；其环境变量入口与“不改变 schema”描述仅适用于当时版本。现行 `async_staging=True` / `--async-staging` 以及扩展后的 schema、107 项回归以第 15.13–15.14 节为准，性能未在冻结配置版本重测。

2026-10-13 接续第 15.11 节。新增 `FUSE_MOE_OFFLOAD_ASYNC_STAGING=1`，仅精确 `1` 启用，在 context 创建时锁定；默认仍为原同步路径。一个懒启动、持久 pthread producer，仅在首次 miss 启动，空路由/全缓存命中不创建线程。队列为固定两槽，实际提交深度不超过实际 banks，每个 bank 最多一个 queued/running/ready job；只对本次已确定 miss wave 供给，不预测下一层、不执行专家 GEMM。

#### 15.12.1 生命周期与预算

- worker 只读取已绑定不变 CPU 权重，写已有 pinned 权重/metadata bank，CUDA DMA 与计算仍由调用线程提交。逐专家 memcpy 不变，CPU/GPU 权重、路由、scale 与一次 H2D 语义不变。
- pinned bank 在本轮 `ready`（权重及 metadata H2D 完成）后即可重填；GPU bank 独立等待 `done`（全部 route tile 计算及 promotion 完成）后才可覆盖。调用线程消费 host-ready 才提交下一轮 DMA。重复 record 的 `ready` 通过每 bank 单任务握手隔代，不允许下一代 DMA 越过前代 worker wait。
- queued 任务可取消，running 任务必须排空；主调用的 `Flight` 在释放 plan/weights/pool 前先 drain producer，再 drain CUDA streams。worker 执行异常传回调用线程、context 标记 closed；线程创建/提交失败同样关闭 context。非法路由/预算不足发生在提交之前，仍允许原有校验失败恢复。
- `close()` 先 shutdown/join worker 再释放 DMA 资源、pool 与权重。worker 命名为 `moe_staging` 仅用于 OS 诊断，不决定执行行为。正常 forward 仍在返回前排空，不增加跨调用在飞语义。
- GPU/pinned pool、31 项正式 stats 和算子 schema 不变。固定 stack 至少 1 MiB，guard 按系统页大小显式配置，线程/任务控制块有有界 reservation；本机与远程本次实测 RAM charged 增量 **1,056,992 B/context**（约 1.008 MiB），创建时即预留，即使尚未启动。计价不是 pthread/libtorch TLS/allocator 的精确 RSS 保证，服务仍须留 reserve。
- 补齐 pool 首次跨 stream 使用的 allocation→copy 事件依赖；静态 resident binding 与动态 promotion 新分配后也完成 caller-stream allocation 再 copy，覆盖 `cudaMallocAsync` 的 stream-ordered 分配边界。静态绑定及 promotion 本来就是同步慢路径，不放松源 bank 租约。
- 保留既有六个 HostStage；async 新增 `host_ready_wait`（调用线程、每 miss wave）与 `host_bank_wait`（worker、每复用 bank）。诊断启用时逐任务传播标准 PyTorch TLS，正常禁用不复制 TLS。线程范围计时可能重叠，不将阶段累计值相加解释为 latency。
- `lookup` 对当前 forward/worker 观察回调直接重入同 context 明确拒绝，避免直接锁等待环；不承诺任意 observer 的跨线程/跨 context 嵌套、信号、栈溢出或析构异常隔离。诊断 setup 尝试 best effort，但标准 TLS 部分安装/恢复不是本实现提供的强事务。

#### 15.12.2 契约测试与真实失败记录

新增四项原生专项：精确开关/创建时锁定/跨任务 profiler TLS/31 stats 对照，懒启动与 close join，worker RAM 不足拒绝，跨 stream/层/cache 晋升/非法路由恢复/新权重 context。稀疏 expert、resident/miss 交错、末尾不足 wave、两种 dtype/激活模式/单双 bank 沿用前轮参数化保护。

首次远程全 async suite 的紧 RAM partial-reclaim 用例及 AOTi 在 bind 时明确拒绝：原 1 MiB AOTi 额度不足以容纳 CPU 后备及 worker。本轮没有移除线程计价或减少 GPU/pinned 压力：AOTi Python/C++ 统一使用相同 **4 MiB RAM 总预算**；紧 RAM 用例测量 sync/async charged delta，以 `99728 + delta` 配置 async、sync 仍 `99728`，扣除 reservation 后可用空间完全相同，原 partial-reclaim 断言及独立不足拒绝测试保留。

5090D 一次全 suite 的旧懒启动测试用“profiler 前后所有 task ID 集合差”判断 close，报告未观察到 retired task。20 次独立交错复跑及 5 轮无 profiler 原生观测未复现，close 后立即至 1 s 均无 worker 残留；**原失败根因未确定**，不能抹掉或声称确定为 `/proc` 竞态。最终测试以实际 `moe_staging` 身份识别，并有界 1 s 检查 task 移除，超时仍失败。历史日志 `/tmp/moe_5090d_final_tests_20261013_1340/suite_env0.log` 保留。

#### 15.12.3 最终版本验收（均已完成，正常退出）

| 环境 / 验收 | 最终结果 | 证据 / 范围 |
|---|---|---|
| 本机 CUDA 11.8 4090D | sync 88 tests / OK，async 88 tests / OK | 最终任务 `6431b6ae-e62e-44d2-abfc-7b72df508ed8`，增量编译 10.1 s |
| 本机严格 AOTi | export/compile/Python load/独立 C++/shared-token/真实两层链/wide FP16+BF16 PASS | 同上，async=1、profile=1；1812 为 PASS 日志标记数，不等于独立测试数 |
| 本机 sanitizer | memcheck 2 tests / OK、synccheck 1 test / OK，均 0 errors | 同上，稀疏/partial/resident、跨 stream/层/动态 cache 与版本更新；不是全部失败注入 |
| 本机 stream-ordered allocator | `backend:cudaMallocAsync` 两专项 / OK | 同上，最终 cache allocation 安全补充后执行 |
| 远程 CUDA 12.8 4090D | sync 88 / OK，async 88 / OK；完整严格 AOTi；legacy 21 PASS；cudaMallocAsync 两专项 / OK | `/tmp/moe_offload_4090d_allocator_20261013T060545Z/`，所有命令 exit=0 |
| 远程 CUDA 12.8 5090D v2 / SM120 | sync 88 / OK，async 88 / OK；完整严格 AOTi；legacy 21 PASS；cudaMallocAsync 两专项 / OK | `/tmp/moe_5090d_allocator_safety_final_20261013_1400/`，所有命令 exit=0 |
| 本机原全 GPU / CPU 契约 | legacy 21 PASS / 0 FAIL；summary 与 auto 契约通过；编辑文件无诊断错误 | 本机没有 pytest，CPU 函数直接运行；格式检查通过 |

远程使用各自 PyTorch `2.11.0+cu128` / CUDA 12.8 与目标 sm89/sm120 构建，不复用本机 `2.6.0a0` / CUDA 11.8 制品。各目录保留源文件、库及日志校验清单，不保存连接凭据；`/tmp` 不是长期归档保证。

#### 15.12.4 正常性能与默认决策

同预算、固定 FP32、disabled cache、tile auto、合成 `skewed_masked`。`weight_transfer` 为 `S128/H512/I512/E16/K8`，`balanced` 为 `S1024/H256/I256/E16/K8`。下表单位 ms，`median / p95`；不是多卡并行。

| 环境 / profile | 请求 banks | sync warm | async warm |
|---|---:|---:|---:|
| 本机 CUDA 11.8 / weight_transfer | 1 | 1.5021 / 1.7347 | 1.3740 / 1.5936 |
| 本机 CUDA 11.8 / weight_transfer | 2 | 1.2625 / 1.8924 | 1.1593 / 1.2277 |
| 远程 4090D / weight_transfer | 1 | 1.263908 / 1.294168 | 1.329758 / 1.366688 |
| 远程 4090D / weight_transfer | 2 | 1.027012 / 1.063624 | 1.082001 / 1.122899 |
| 远程 4090D / balanced | 1 | 0.460714 / 0.506015 | 0.498859 / 0.540268 |
| 远程 4090D / balanced | 2 | 0.386017 / 0.419799 | 0.433483 / 0.487732 |
| 远程 5090D / weight_transfer | 1 | 0.545237 / 0.597026 | 0.553792 / 0.614626 |
| 远程 5090D / weight_transfer | 2 | 0.424640 / 0.455989 | 0.409021 / 0.469489 |
| 远程 5090D / balanced | 1 | 0.248809 / 0.286453 | 0.247294 / 0.304993 |
| 远程 5090D / balanced | 2 | 0.192201 / 0.225835 | 0.192980 / 0.236704 |

本机仅一次 BA（async→sync），每配置 40 warm，未顺序平衡；远程两轮 ABBA、每位置 30 warm，聚合每配置 120 个原始样本（nearest-rank p95），未建立置信区间。本机 cold median 增约 18–20%；远程两卡 cold median 全部变慢。4090D warm 变慢，5090D bank2 weight_transfer median 改善约 3.68% 但 p95 升高，**无跨环境一致收益，保持同步默认**。

实际 weight_transfer 为 bank1/cap4/2 waves 与 bank2/cap2/3 waves，每调用 9 MiB，GPU charged 7,388,672 / 7,366,656 B，pinned 6,291,596 / 6,291,604 B；balanced 每调用 2.25 MiB，GPU charged 6,029,312 / 5,942,272 B，pinned 1,573,004 / 1,573,012 B。相同 bank 的 sync/async 流量、GPU/pinned、tile/wave 均相同，仅 RAM reservation 增加。全命中组实际 banks=0，不把它们混成流水线收益。

性能证据：本机任务 `fdcdc32a-a257-4db8-863d-7cc32cab4b26`（48 reports）；远程各 384 reports，目录 `/tmp/moe_offload_4090d_20261013T052625Z/` 与 `/tmp/moe_5090d_acceptance_20261013_1328/`。这些性能数据早于最终线程命名和 cache allocation 安全补充，**不是最终版本重测**；所选 disabled warm 不经过改变的 resident/promotion 慢路径，故保留其限定结论，不外推到 cache 冷初始化。

#### 15.12.5 独立诊断与真实重叠

异步 CPU 阶段诊断本机 24 组矩阵、3 warm/组，事件数量、参考、流量与预算全部核对；disabled bank1/2 的 host_weight_copy 累计 median 0.683255 / 0.694808 ms，host_ready_wait 0.719500 / 0.712829 ms。与前轮同步诊断不是同一次采集，不能跨轮相减给出稳定 memcpy 加速。任务 `a684732d-390e-4958-9182-8a8abe3b0a14`。

同任务另行采集 Nsight CUDA/NVTX（async=1、不启 CPU profiler），用第 15.8 节的半开范围/同进程同设备/合并区间方法，精确选取 FP32 disabled 的 3 warm/配置，并按大 H2D 尺寸与每调用 9 MiB 核对：

| 配置 | NVTX median (ms) | kernel 活跃 (ms) | weight H2D 活跃 (ms) | kernel ∩ weight H2D |
|---|---:|---:|---:|---:|
| bank1 / cap4 | 1.472140 | 0.059648 | 0.554467 | **0 ns** |
| bank2 / cap2 | 1.235532 | 0.083648 | 0.564578 | **0 ns** |

六次调用交集均为零。bank2 第二次 warm 的前波 Down→下波 H2D 空窗为 0.123233 / 0.017664 ms；供给提前不等于 DMA 已覆盖计算。此为诊断 trace，不是正常性能，无 libc memcpy/OS runtime 精确归因，也不是整卡利用率。该 trace 早于最终 cache allocation 补充，但 disabled warm 不走该慢路径。分析后清理本轮 `.nsys-rep` / `.sqlite`；结论及任务 ID 保留。

#### 15.12.6 worker 阶段收尾与统一待办引用

worker 阶段实现与本节列明的验收任务已结束，无自动提交、无临时 helper 留存。现行调用方式为 `async_staging=True`，配置/部署契约见第 15.13 节，最新功能结果见第 15.14 节。

原列于此处的事项已集中重定范围：故障注入 → 第 16.2 节 T03；当前版本合成性能/重叠及 CPU/NUMA/PCIe 测量环境记录 → T04；支持范围内较大输入/权重与多层合成验收 → T05；真实业务模型/质量 → 第 16.4 节外部职责，不再列为本仓待办；非承诺能力 → 第 16.4 节技术扩展。此处仅保留历史验收边界，不再维护独立的“下一步”。

### 15.13 冻结配置与部署契约（现行接口）

2026-10-13 在 worker 基础上完成显式配置与部署安全闭环。生产接入采用 LOCKED 模式：**配置在离线构建期冻结，原生 context 在加载期按同一配置创建，制品执行前在图外强制预检，图内 Guard 再核对**。不依赖运行期 Python 调度，也不从环境变量猜测 offload 行为。

#### 15.13.1 显式配置与兼容边界

`OffloadConfig` 是 frozen dataclass，`layer_specs` 被规范化为按 layer ID 排序的深度不可变 tuple，resident IDs 排序去重校验；拒绝 bool 冒充整数、非法类型、int64/int32 越界、重复层和不支持的形状。`from_store()` 与 `validate_store()` 从当前实际 tensors 重新校验形状及 backing storage，而不是信任 `ExpertStore` 早先缓存的计价。

| 配置 | 现行语义 |
|---|---|
| `gpu_budget_bytes` / `ram_budget_bytes` / `pinned_budget_bytes` | LOCKED 三项均为显式正 int64 字节数；不允许 `0` 预算自动探测。兼容的直接 `PrefillMoERuntime(...)` / 原生 create 仍保留 `0` 初始化探测，但不代表冻结部署 |
| `target_sm` | 离线显式填写，例如 `89` / `120`；`from_config()` 在 create/权重绑定前核对所选设备，原生 Manifest 再检查。设备 ordinal 不进入契约，同 SM 副本可使用不同本地 ordinal |
| `num_banks` / `activation_mode` | 请求 1/2 bank；`route_buffer` / `fp32_accumulate` / `auto`。规划后的实际 bank 数仍可能为 0，不将配置请求数当作运行统计 |
| `max_route_tile_rows` | 正整数为固定 cap；Python `'auto'` 序列化为原生 `0`，指预算自适应路由切块，不是预算自动探测 |
| `resident_cache_bytes` / 每层 resident IDs | 静态部署缓存；所有层 resident 的 512 B 对齐 charge 合计必须符合全 context cap 与 GPU 预算 |
| `dynamic_cache_bytes` / `cache_policy` / `cache_decay_interval` / `cache_min_reuse` / `cache_update_bytes` | 动态 cache 的容量与策略进入冻结配置；动态 admission/eviction 和 heat 是执行状态，不改变静态 resident 契约 |
| `async_staging` / `host_profile` | 严格 bool，默认 false，创建 context 时固定；分别控制有界 producer 与原生 HostStage 标记 |
| `tile_m` / `tile_n` | `tile_m` 为 `0/32/64/128`，`0` 固定确定性 shape-based 选择策略，不接受环境覆盖；`tile_n` 为 `64/128`，默认 128 |
| `layer_specs` | `(layer_id, E, H, I, dtype_code, resident_ids)`；dtype code 为 FP16=0、BF16=1，CPU canonical 布局仍为 `w1(E,2I,H)` / `w2(E,H,I)` |

offload 的 `FUSE_MOE_OFFLOAD_ASYNC_STAGING` / `FUSE_MOE_OFFLOAD_PROFILE` 入口已移除；共享 launcher 中的 `FUSE_MOE_TILE_M` / `FUSE_MOE_TILE_N` / `FUSE_MOE_TIME` / `FUSE_MOE_NO_PDL` 不得覆盖 offload 参数。旧全 GPU `fuse_moe` 仍保留其原有环境诊断能力，**不是整个仓库禁止环境变量**。测试/benchmark 改用 `--async-staging` / `--host-profile` / `--nvtx`，构建配置如 `MAX_JOBS`、`TORCH_CUDA_ARCH_LIST` 和 PyTorch allocator 配置不属于 offload 执行契约。

PSS（Programmatic Stream Serialization）由硬件匹配决定，不作为任意部署开关：offload launch 每次查询当前设备，SM90+ 的 `griddepcontrol` 指令必须配对 PSS launch 属性，SM89 使用普通 launch；无上游 trigger 时维持串行依赖语义。不复用 legacy 的首设备静态缓存，也不允许其环境诊断开关使 offload 的指令/属性失配。该修复是架构正确性保护，不是新的 H2D/compute 重叠性能证据。

#### 15.13.2 图内配置、全层 Manifest 与 schema

现行前向 schema：

```text
fuse_moe_offload(
    Tensor x, Tensor gate_up_weight_cpu, Tensor down_weight_cpu,
    Tensor topk_ids, Tensor topk_scale, Tensor(a!) runtime_token, int layer_id,
    int[]? deployment_config=None, int[]? deployment_manifest=None
) -> Tensor
fuse_moe_offload_validate(Tensor runtime_token, int[] deployment_manifest) -> ()
```

保留可选参数兼容旧调用；**新 LOCKED 制品必须提供配置与完整 Manifest**。`OffloadedMoELayer(layer_id, config=config)` 将二者作为 tuple 捕获，export/AOTi 中变为常量 `int[]`，不是额外 Tensor 输入，也不是 `nn.Parameter` / GPU 常量。CPU 权重与可变 token 继续作为图输入；双层共享 token，并将第一层输出实际传给第二层。

版本 1 每层配置有 23 个固定 int64 字段，随后为静态 resident IDs：

```text
[1, gpu, ram, pinned, banks, activation_mode, max_route_tile_rows,
 resident_cache_bytes, dynamic_cache_bytes, cache_policy, cache_decay_interval,
 cache_min_reuse, cache_update_bytes, async_staging, host_profile, tile_m, tile_n,
 E, H, I, dtype_code, resident_count, target_sm, *sorted_static_resident_ids]
```

完整模型 Manifest 格式：

```text
[1, layer_count, layer_id, config_length, *layer_config, ...]
```

Manifest 必须按 layer ID 严格升序、无重复/尾随字段，且覆盖 context 的**全部绑定层**。每一层都核对 context 配置、SM、权重 shape/dtype、静态 resident IDs、绑定时 Tensor version counter；当前层配置还必须与自身 Manifest 片段相同。因此仅第二层静态配置或权重版本错误，也在第一层执行前拒绝。动态缓存项不混入静态 resident IDs，不因晋升改变制品契约。

#### 15.13.3 AOTi 异常传播限制与必需图外预检

2026-10-13 实测上述工具链的**默认 unchecked fallback**会忽略 executor 返回错误码；原生 throw 后输出可能仍是未定义 Tensor，图外检查尚未执行，下游就可能消费失败输出。**这项旧限制已由第 15.15.1 节的 checked lowering + managed begin/check 闭环**：新制品必须用 `compile_aoti()` 重新编译，不能以只做图外预检或新增状态查询来修补旧制品。本小节以下条目记录原预检设计的边界。

- Python：每次执行加载后的 package 之前，调用 `runtime.validate_config(artifact_config)`，参数必须来自**该制品的冻结配置**，不能拿运行时自身配置代替。`from_config()` 创建时已预检，但不能替代逐次预检。
- C++：在 `AOTIModelPackageLoader::run()` 之前，直接通过 dispatcher 调用 `recsys_ops::fuse_moe_offload_validate(token, artifact_manifest)`；不得把这个校验重新放入 AOTi graph/fallback。`LockedRuntime` 创建后的校验和每次 run 前校验均已落地。
- 预检是只读操作：成功和拒绝均不改变 token 内容/version/storage、cache、正式 stats、GPU 分配，也不封存 context。负向验收同时断言模型尚未被调用；不能仅凭最终 error 文本判断拒绝时机。
- 调用方须串行化同一 context 的完整“预检→模型执行”流程，并保证期间权重不变、token/权重输入就是该 context 已绑定对象。预检不会替调用方校验任意 graph 输入或提供跨调用事务。
- 本方案解决**部署配置/绑定权重契约错误**的可见性，没有修复 PyTorch 对所有图内异常的传播。运行期非法 route、真实 allocator OOM、CUDA/worker 失败不由该预检保证可见；T02 已在第 15.15 节关闭，但不能宣称任意第三方算子、旧 unchecked 制品或驱动故障获得全异常透明性。

#### 15.13.4 封存、权重版本与归档

- 带非空配置的 forward 通过前置 Guard 并进入执行路径时设置 `sealed`；携带 Manifest 时同时设置 `seal_requires_manifest`。后续拒绝新 weight/resident bind，也拒绝空配置或丢弃 Manifest 的降级调用。预检不封存；不把“执行已封存”理解为整个 forward 已成功或可回滚，预算/分配失败后不保证自动解封。
- 新训练权重必须使用新的 store/context；旧 context 对权重 storage identity/offset/shape/stride/dtype 与绑定时 version counter 的变化拒绝。全层 Manifest 能在首层之前发现另一层的版本变化，但不能阻止未遵守契约的并发写或通过不递增版本的外部内存写。
- `save_cpp(path, weight_version=..., config=None)` 保留 format1；传入 `config` 产生 format2，除 `format_version` / `weight_version` / `layer_ids` / `w1_{layer}` / `w2_{layer}` 外，逐层新增 CPU int64 `offload_config_{layer}` buffer。
- `ExpertStore.load_cpp()` 校验元数据、精确 buffer inventory、canonical 权重与配置，format2 恢复 `store.deployment_config`；C++ LOCKED runner 要求 format2，在 create 前校验全层配置/SM/预算，再按该配置 create/bind/resident-bind/validate。
- `save_cpp()` 先重新校验当前 store，再 clone 独立快照；共享/重叠 backing storage 在 clone 后可能扩张，必须以**实际快照 retained bytes**再次核对配置 RAM 预算，失败不打开/覆盖目标文件。这不保证导出或 `torch.jit.load` 的瞬时 RSS，服务需预留源权重、快照、转换及加载峰值。
- 归档的 `weight_version` 是调用方给定的非负模型发布标签；原生检查的是绑定时各 Tensor 的 `_version()`。当前 `OffloadConfig` / Manifest **不包含权重内容 hash 或发布标签匹配**，不能证明同形状权重属于期望训练版本。这是当前能力边界，不将业务发布配对当作算子缺失。算子级绑定/预检封装与工具链/扩展加载分别跟踪于第 16.2 节 T01/T09；可选内容完整性见第 16.3 节 R02，期望业务训练版本由外部发布层验证。不提供跨 SM 制品通用保证。

#### 15.13.5 最小 Python 接入示例

以下为单层合成接入，示例预算不是业务推荐值。目标 SM 必须来自离线目标配置；SM120 应使用另行构建的配置/扩展/制品，不修改已加载制品的 `target_sm`。

```python
from pathlib import Path
import torch
from custom_ops import ops
from custom_ops.moe_offload import (
    ExpertStore, OffloadConfig, OffloadedMoELayer, PrefillMoERuntime,
    InputBinding, compile_aoti,
)

# Load an already-built target-specific library and its .so.json. No implicit JIT.
ops.load_precompiled("fuse_moe", "/absolute/path/recsys_fuse_moe_kernel.so")
store = ExpertStore({0: ((torch.randn(2, 128, 64) * 0.02).half(),
                        (torch.randn(2, 64, 64) * 0.02).half())})
artifact_config = OffloadConfig.from_store(
    store, target_sm=89,
    gpu_budget_bytes=2 << 20, ram_budget_bytes=4 << 20, pinned_budget_bytes=1 << 20,
    activation_mode="fp32_accumulate", max_route_tile_rows="auto",
    async_staging=False, host_profile=False, tile_m=32, tile_n=64,
)
store.save_cpp(Path("weights.pt"), weight_version=17, config=artifact_config)
store = ExpertStore.load_cpp("weights.pt")

with PrefillMoERuntime.from_config(store, artifact_config, device="cuda:0") as runtime:
    w1, w2 = store.layers[0]
    x = torch.randn(7, 64, device="cuda:0", dtype=torch.float16)
    ids = torch.tensor([0, 1], device="cuda:0", dtype=torch.int32).expand(7, 2).contiguous()
    scale = torch.full((7, 2), 0.5, device="cuda:0", dtype=torch.float32)
    args = (x, w1, w2, ids, scale, runtime.token)
    dim = torch.export.Dim("S", min=2, max=64)
    exported = torch.export.export(
        OffloadedMoELayer(0, config=artifact_config).eval(), args,
        dynamic_shapes=({0: dim}, None, None, {0: dim}, {0: dim}, None),
    )
    # Offline compilation only; deployment loads the resulting package unchanged.
    package = compile_aoti(
        exported, package_path="model.pt2", inductor_configs={"triton.cudagraphs": False},
    )
    loaded = torch._inductor.aoti_load_package(package)
    bindings = (InputBinding(1, "w1", 0), InputBinding(2, "w2", 0), InputBinding(5, "token"))
    output = runtime.invoke(
        loaded, (x, None, None, ids, scale, None),
        artifact_config=artifact_config, bindings=bindings,
    )
```

这是用于演示完整流程的离线编译示例，部署不重复 export/compile；按第 15.15.2 节加载完整官方 package、配套冻结配置及权重后调用 `invoke`。示例 `S∈[2,64]` 是该图导出边界，不等于原生算子的全局输入上限。Torch 2.11 的 Python package helper 可能探测 C++ 编译器版本；无工具链部署直接采用官方原生 `AOTIModelPackageLoader` 绑定，具体参数及验收见第 15.15.2 节。

### 15.14 冻结配置版本：2026-10-13 历史验收与恢复入口

#### 15.14.1 环境与结果（2026-10-13，已完成）

| 验收 | 本机 RTX 4090 D / SM89 | 远程 RTX 5090 D v2 / SM120 |
|---|---|---|
| 工具链 | PyTorch `2.6.0a0`，CUDA build / nvcc 11.8 | Python 3.12.11，PyTorch `2.11.0+cu128`，CUDA / nvcc 12.8（12.8.61），驱动 580.105.08；`TORCH_CUDA_ARCH_LIST=12.0a` |
| 同步全量 `test_fuse_moe_offload.py` | 107/107，4.758 s | 107/107，无跳过，127.821 s（含全新编译） |
| 异步全量 `--async-staging` | 107/107，4.900 s | 107/107，无跳过，8.184 s |
| export / Python AOTi / 独立无 Python C++ | legacy、shared-token、真实双层链、wide FP16/BF16、LOCKED single/chain 及拒绝验收均通过 | legacy、shared-token、真实双层链、wide FP16、LOCKED single/chain 及拒绝验收均通过；使用 `--no-wide-bf16`，可选 wide BF16 **本轮未运行** |
| memcheck 部署与供给子集 | 6/6，`ERROR SUMMARY: 0 errors`，14.739 s | 6/6，0 errors，4.465 s |
| synccheck 子集 | 2/2，0 errors，10.068 s | 2/2，0 errors，3.247 s |
| `backend:cudaMallocAsync` 部署/跨 stream 子集 | 6/6，0.404 s | 本轮不以普通 allocator 只读检查冒充单独 cudaMallocAsync 复测；既有 allocator 证据见第 15.12.3 节 |
| 原全 GPU MoE | 21 PASS / 0 FAIL，FP16/BF16、重复/稀疏与空输入拒绝 | 本表不把前轮 21 项结果当作冻结配置版本的新复测 |
| Python 语法 / 补丁空白 / 编辑文件诊断 | `py_compile` 与 `git diff --check` 通过，未报告 linter errors | 源码及扩展单独复制/构建，不复用本机二进制 |

AOTi 的大量 `PASS` 为断言/观察日志，远程本轮 1819 条不等于 1819 个独立单测。sanitizer 为指定子集，不代表整套 107 项或任意故障注入均受 sanitizer 检查。远程 CUPTI multiple-subscribers 警告导致部分 CUDA profiler activity 缺席；该事实不影响本轮工具错误汇总，但不能作为完整 GPU trace 使用。SM89/SM120 分别构建并验收，不是两卡并行或一个制品跨架构执行。

LOCKED 验收明确检查：export graph 与官方 extern JSON 存在冻结 `int[]`；单层/双层输出参考；动态 cache 从 miss→admission→hit；配置字段/权重绑定不匹配；仅另一层 resident 或 version 错误也在首次模型执行前拒绝；正确前向后封存不能降级；format2 的版本/SM/shape/resident/count/零预算损坏拒绝。只读校验的 GPU allocation 断言在调用前后同步，并**只包围原生调用**，不把 `torch.testing.assert_close` 自身临时 GPU 分配混入采样。

#### 15.14.2 失败处理与证据口径

- 本机最终串行任务 `1d971a79-37c5-4cff-a9c8-e99c776759ce` 的两轮 107 项及 AOTi 已通过，但 sanitizer 命令引用旧方法名 `test_deployment_config_model_manifest_rejects_other_layer_before_execution`，触发 `AttributeError`，最终 exit=1。其 `0 errors` 不能被记成整个任务成功。
- 正确方法名为 `test_deployment_manifest_checks_other_layer_before_execution_and_seals`。补跑任务 `778d63ee-0a71-4d75-8913-46388f19e94b` 完成 memcheck 6、synccheck 2、cudaMallocAsync 6 项，整体 exit=0，补齐本机验收。原全 GPU 任务 `07caaf7d-f187-463e-a887-5ac9b121ebd7` exit=0，21 PASS。
- 跨架构回归中的采样/fixture 问题没有通过放宽数值或分配断言规避：同步采样隔离测试框架临时分配；LOCKED chain 使用独立克隆的合成权重增强第二层输出，并预先以 CPU 参考检查 `S=2/7/64` 的非退化幅度。legacy/checkpoint 权重不被改写，原点值/归一化误差阈值保留。
- SM120 最终验收使用隔离临时目录；结束后源码副本、构建缓存与临时日志已清理，无遗留测试进程，未覆盖远程原工作树。保留本节结果而非承诺临时原始日志永久存在；长期制品/结果归档的缺口统一见第 16.2 节 T08，不保存凭据。
- 本次只验证功能与部署保护，**没有重测第 15.12 节性能，也未新增实际 DMA/compute overlap 证据**。同步仍默认，不能因 107 项全绿宣称异步更快或业务质量验收完成。

#### 15.14.3 前阶段复现命令（历史记录）

以下是 2026-10-13 使用的命令；现行一键交付入口见第 15.15.2 节，不再以本段历史缓存路径或 unfrozen benchmark 验收新制品。历史本机使用 SM89 工具链，运行前确认 GPU 无其他回归任务、扩展与目标 PyTorch/CUDA 匹配。SM120 使用自身 CUDA 12.8+ 环境单独构建，不能拷贝本机 `.so`。测试主体是 `unittest` 脚本，不依赖本机安装 pytest。

```bash
cd /workdir/custom_ops
MAX_JOBS=2 PYTHONPATH=/workdir python tests/test_fuse_moe_offload.py
MAX_JOBS=2 PYTHONPATH=/workdir python tests/test_fuse_moe_offload.py --async-staging
PYTHONPATH=/workdir python tests/test_fuse_moe_offload_aoti.py --ops-library /workdir/custom_ops/torch_extensions/recsys_fuse_moe_kernel/recsys_fuse_moe_kernel.so --async-staging --host-profile
PYTHONPATH=/workdir python tests/test_fuse_moe.py
```

远程最终 AOTi 在上述参数后增加 `--no-wide-bf16`；该可选矩阵的补测状态统一见第 16.2 节 T07，新增结果不追溯修改本轮范围。

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir compute-sanitizer --tool memcheck --error-exitcode 99 python tests/test_fuse_moe_offload.py --async-staging OffloadTests.test_deployment_native_validate_is_read_only_and_does_not_seal OffloadTests.test_deployment_manifest_checks_other_layer_before_execution_and_seals OffloadTests.test_deployment_manifest_other_weight_version_rejected_before_first_forward OffloadTests.test_deployment_config_native_rejection_is_before_mutation_and_seals_context OffloadTests.test_deployment_config_dynamic_admissions_do_not_change_static_contract OffloadTests.test_host_weight_runs_holes_partial_wave_and_cache_boundaries
PYTHONPATH=/workdir compute-sanitizer --tool synccheck --error-exitcode 99 python tests/test_fuse_moe_offload.py --async-staging OffloadTests.test_deployment_manifest_checks_other_layer_before_execution_and_seals OffloadTests.test_host_weight_runs_holes_partial_wave_and_cache_boundaries
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync PYTHONPATH=/workdir python tests/test_fuse_moe_offload.py --async-staging OffloadTests.test_deployment_native_validate_is_read_only_and_does_not_seal OffloadTests.test_deployment_manifest_checks_other_layer_before_execution_and_seals OffloadTests.test_deployment_manifest_other_weight_version_rejected_before_first_forward OffloadTests.test_deployment_config_native_rejection_is_before_mutation_and_seals_context OffloadTests.test_deployment_config_dynamic_admissions_do_not_change_static_contract OffloadTests.test_async_staging_stream_layer_cache_recovery_and_new_weight_version
```

正常 benchmark 不启 profiler；诊断单独执行。例如当前显式异步/host 诊断入口：

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir python tests/test_fuse_moe_offload_aoti.py --ops-library /workdir/custom_ops/torch_extensions/recsys_fuse_moe_kernel/recsys_fuse_moe_kernel.so --async-staging --host-profile --nvtx --benchmark --benchmark-profile weight_transfer --benchmark-scenario skewed_masked --benchmark-route-tile-rows auto --iterations 10
```

#### 15.14.4 阶段收尾与统一待办引用

冻结配置阶段的实现及第 15.14.1 节明确列出的回归已结束，**不等于全部待办关闭**。原分散事项已迁移至第 16 节，并在 2026-10-14 按通用算子仓库定位重新定界：保留算子接入、错误/排空、合成性能/覆盖等缺口，移出业务模型 loader、质量与服务职责；本节只保留验收结果。

恢复工作先读第 16 节当前任务状态，再按第 15.15 节核对公开入口和最终证据；不重复历史验收、不丢弃用户未提交改动。未自动提交 Git。

### 15.15 最终冻结源码：2026-10-15 双架构全阶段交付

本节是现行接口和最终验收入口。SM89、SM120 的完整 harness 均 **exit=0、`state=passed`、`full_delivery_passed=true`**，build/unit/AOTi/sanitizer/performance 五阶段全部通过。两目标审计的 **97 个源码文件 hash 相同，运行前后相同，并与最终磁盘源码一致**；远程完整源码及第三方依赖清单另含 903 个文件，不把依赖清单数混作源码数。两套库分别构建，不能共享二进制或推定 ABI 相同。本次仅补录已有交付证据，没有重新运行测试。

| 最终身份 | SM89 / 本机 | SM120 / 独立远程环境 |
|---|---|---|
| GPU / 工具链 | RTX 4090 D；Torch `2.6.0a0`，CUDA/nvcc 11.8，GCC 10.2.1，CXX ABI=0，arch `8.9`，`FA_TARGETS=2` | RTX 5090 D v2；Python 3.12.11，Torch `2.11.0+cu128`，CUDA/nvcc 12.8.61，GCC 10.3.1，CXX ABI=1，arch `12.0a`，`FA_TARGETS=4` |
| UTC 起止 / elapsed | `2026-10-15T11:53:28.016337+00:00` → `12:30:53.406985+00:00`；2245.5 s | `2026-10-15T11:59:11.672335+00:00` → `13:10:36.183307+00:00`；4284.5 s |
| 原始 run 目录 | `/workdir/custom_ops/artifacts/offload_delivery_sm89/runs/20261015T115328Z-c6d45e8dfe73` | `/home/lifengyi06/custom_ops_delivery_20261014/evidence_sm120/runs/20261015T115911Z-a5f70078ea63` |
| report | `/workdir/custom_ops/artifacts/offload_delivery_sm89/report.json` | 原始 `<run>/report.json`；忠实拉取副本 `/workdir/custom_ops/artifacts/offload_final_sm120_20261015_last/report.json` |
| report SHA256 | `f58d378c3de66531bf56f01e3a5d551fbf05accc669047784cbcee1a5fed4aa4` | `211d309e826bcf9235d88fd6f4a1bdf5babbc297065afcd33b51ceec80d4d5fb` |

远程独立只读审计索引为 `/workdir/custom_ops/artifacts/offload_final_sm120_20261015_last/final_audit_index.json`；索引不是改写后的原始报告。原始 stdout/stderr、命令、配置、数值与性能样本、trace 和源码身份由 report/run 索引关联。远程复制树不含 Git 元数据时 `git_head`/`git_diff` telemetry 可失败，源码身份由文件摘要证明，不伪称 Git 命令成功。产物位于本机 `artifacts/` 和远程独立持久目录，不在 `/tmp`；大库/trace 不提交 Git，保留至人工清理，不承诺永久留存。远程原工作树未覆盖；用户未提交改动保留，未 commit/push，未新增临时 helper 文件。

#### 15.15.1 已实现的公开接入、构建与失败契约

- **离线构建/严格加载**：`__init__.py` 的 `CustomOps.build(output_dir, cuda_arch_list)` 复用现有分组 sources/flags；`load_precompiled()` 校验 format2 `.so.json` 的原生内容 SHA、required schemas、fake、build flags、Torch/CUDA/CXX ABI 与目标 arch，在可判断范围内于 dlopen 前拒绝不兼容。元数据 build version=1、schema hash/interface version=2，不保证任意 ABI 兼容。`recsys.py` 的 `RecsysOps.load_precompiled()` 令整个门面严格 `allow_jit=False`，无 backend fallback；进程库身份防止同库复制后的重复注册，fake 注册错误 fail closed。开发默认 JIT 与缓存仍保留，旧 `.build_stamp` 只作历史记录。
- **Python 算子级接入**：`moe_offload.py` 的 `PrefillMoERuntime.from_config()`、`prepare_inputs()`、`invoke(callable, args, artifact_config=..., bindings=...)` 管理生命周期与同 context 的预检→managed begin→调用→check，`artifact_config` 必须来自制品，不能省略。`InputBinding(index, kind, layer_id)` 的 kind 为 `token/w1/w2`；`None` 占位按显式绑定补齐，已提供 Tensor 则严格校验，不静默修正错绑。调用者提供官方 loader/callable 与业务参数映射；close 与 invoke 共用线程互斥，不接受混用 raw/managed 或并发改绑定/权重，也不承诺 observer callback 崩溃隔离。原单算子 raw schema 未改变，CPU 权重快照和 Tensor version 校验仍有效。
- **无 Python C++**：`csrc/fuse_moe/fuse_moe_offload_runtime.h` 的 `custom_ops::offload::Runtime` 是 RAII/dispatcher 薄封装，`InputBinding::Role` 为 `Token/GateUp/Down`，携带 index 与可选 layer ID；CPU canonical fused 权重的 GateUp/Down 对应 Python w1/w2。它管理 context 而非任意模型 loader，接受外部官方 loader 的调用入口。独立 runner 不初始化 Python；链接 `libpython` 仅满足符号依赖，不等于运行解释器。
- **checked AOTi**：`moe_offload_aoti.py` 提供 `register_aoti_lowering()`、`compile_aoti()`，并由 `moe_offload.py` 公开导出。私有 Inductor 适配仅审计 Torch `2.6.0a0`、`2.11.0`（含对应 local suffix），未审计 release/config/ABI 拒绝。实现采用官方 IR `FallbackKernel` 子类，create/args/output metadata/mutation 及 extern JSON 沿用上游；仅 checked C++ 调用语句由直接 AST emit 发出，在失败节点返回错误码并**先于下游中止**。没有 monkeypatch 安装的 Torch，也没有替换生成源码字符串。编译 helper 在局部作用域禁用 CUDA Graph/autograd/local/remote caches 并恢复设置；不支持自动 CUDA Graph。旧 unchecked package 必须重建，不能靠新增状态接口修复；其他第三方算子的失败通道仍由其调用者负责。
- **状态与 sticky error**：CPU int64 `[10]` v1 为 `[version,state,managed,pending_error,forward_sequence,success_sequence,failure_layer,failure_sequence,managed_sequence,success_layer]`，state=0 healthy、1 recoverable、2 poisoned。只读 preflight 不改 sequence/token，不 ACK 错误；eager 可恢复失败后立即重试仍保留首个 sticky error，managed begin 必须先 ACK，下游失败不能覆盖首错。预分配/route guard 可恢复错误可 ACK；真实 CUDA/worker/thread 提交错误保守 poison，只允许 close，不接受部分输出。

离线构建 API 示意（目标来自离线配置；构建由分组实例完成，部署由门面严格加载）：

```python
from pathlib import Path
from custom_ops.recsys import FuseMoeOps, RecsysOps
output_dir = Path("/absolute/path/offload_native_sm89")
library = FuseMoeOps().build(output_dir, cuda_arch_list="8.9")
recsys_ops = RecsysOps()
recsys_ops.load_precompiled("fuse_moe", library)
# The whole facade is now strict: missing groups cannot silently JIT.
# Deploy this .so together with its format2 .so.json and matching package.
```

#### 15.15.2 全阶段矩阵、官方部署与复现入口

下表两目标均为本轮完整执行，不借用历史通过；计数是对应脚本/观察范围，不汇总成虚构的独立测试总数。

| 阶段/范围 | SM89 | SM120 |
|---|---|---|
| build | prod/testing 各自通过 | prod/testing 各自通过 |
| CPU 子进程 | loading 50、lowering 10、delivery parser 12 | loading 50、lowering 10、delivery parser 12 |
| offload 全量 | sync 125、async 125，0 skips | sync 125、async 125，0 skips |
| 原全 GPU MoE | 21 PASS / 0 FAIL | 21 PASS / 0 FAIL |
| AOTi | prod FULL、testing PUBLIC 1/2/3 层；动态 S=2/7/64 | 同左 |
| Python wide | FP16/BF16 各 72 次调用观察 | FP16/BF16 各 72 次调用观察 |
| 独立 C++ | full native、公开 1/2/3 层 FP16 输入流与动态 S；wide FP16 明确通过 | 同左；不独立宣称可选 C++ wide BF16 通过 |
| 无工具链部署 | 每个库 3 个部署子进程，官方 loader 成功 | 同左 |
| AOTi 故障 | 3 个图 × after_first_h2d/after_kernel_submit/thread_create，共 9 个子进程 | 同左 |
| 坏归档 | 12 个预期拒绝子进程 exit=1，父任务正确判定通过 | 同左 |
| Python memcheck / synccheck | memcheck sync 13/async 13（`--leak-check no`）；synccheck 各 3，0 errors | 同左 |
| cudaMallocAsync | sync 15/async 15 | 同左 |
| native full memcheck | `--native-faults`，0 bytes / 0 allocations / 0 errors | 同左 |
| performance / nsys | 80/80 正常 benchmark，20 ABBA 组；独立 GPU trace 完整 | 同左 |

AOTi 验收保留官方 extern JSON 与 checked C++ 调用证明，GPU 输入/CPU 权重/token 显式绑定；故障子进程确认 poisoned 错误可见、原始 native error 保留且不接受部分输出。部署以空 PATH、编译工具阻断、无外部 C++/CUDA 源码树验证，保留官方 PT2 package 完整成员（包括 loader 所需 `.cpp/.o/.metadata.json`），**不是宣称包内无嵌入源码**。直接使用官方原生 Python binding：Torch 2.6 构造参数为 `(path, 'model')`，2.11 为 `(path, 'model', False, 1, -1)`；整型位置接受 int/`typing.SupportsInt`，不传占位字符串。2.11 不用会只读探测 CXX 版本的 Python `aoti_load_package` helper；调用者仍可自行选择已适配的官方 API，再交给 `runtime.invoke`，本仓不提供通用模型 loader。

本机完整命令（文档记录，不在本次编辑执行）：

```bash
cd /workdir/custom_ops
PYTHONPATH=/workdir python tests/run_fuse_moe_offload_delivery.py --output-dir /workdir/custom_ops/artifacts/offload_delivery_sm89 --target-arch 8.9 --jobs 2 --build-timeout 1800 --unit-timeout 1200 --aoti-timeout 1800 --sanitizer-timeout 1200 --performance-timeout 1800
```

SM120 在自身 Python 3.12/Torch 2.11/CUDA 12.8 环境的独立源码树运行同一脚本，`PYTHONPATH` 指向该树父目录，`--output-dir /home/lifengyi06/custom_ops_delivery_20261014/evidence_sm120 --target-arch 12.0a`，其余 timeout/jobs 同上；不复用 SM89 库、不覆盖原工作树。默认 full stages 不选可选 C++ performance matrix；原生 C++ 功能已验收，该性能矩阵不是 T 类关闭条件。

#### 15.15.3 正常性能基线与独立 GPU 时间线

每个目标完成 **80 次正常 benchmark 进程调用，即 20 组 ABBA**，不是 80 组 ABBA：2 个 profile × 2 个 route scenario × 5 个配置 × `sync/async/async/sync`。`weight_transfer` 为 S=128、H=I=512、E=16、K=8、GPU budget=48 MiB；`activation_near_budget` 为 S=8192、H=I=256、E=16、K=8、GPU budget=60 MiB。scenario 为 `skewed_masked/hotset_transition`；配置为 `fp32_accumulate/bank2/{disabled,static_hot,reuse,lru}` 和 `route_buffer/bank1/disabled`，固定显式 cap/cache/bank，tile M=0、N=128、route rows=auto。

每进程记录 40 cold、40 warm、3 priming 和 41 initialization 样本；每目标分别为 3200/3200/240/3280 个样本。正常 wall stopwatch 包含公开 `runtime.invoke` 的预检、begin、官方 loader、check 与完成等待，不启 profiler。原始 p50/p95、calls/s、有效 token/s、H2D、charged RAM/pinned/GPU 与进程/allocator 峰值见 `<run>/bench/performance_summary.json` 及其样本索引；进程峰值可能包含编译和参考计算，不等于算子 RAM 预算保证。

| 描述性指标 | SM89 | SM120 |
|---|---|---|
| pooled warm median 的 sync/async 比值范围 | 0.9141764–1.2234931 | 0.8873376–1.1690096 |
| 独立 trace 匹配的权重 H2D 字节 | 783286272 | 783286272 |
| 权重 H2D 时间区间并集 | 45995518 ns | 19766343 ns |
| kernel 与权重 H2D 交集 | 0 ns | 773368 ns |
| 交集 / 权重 H2D 并集 | 0% | 3.91255% |

两端 trace 各匹配 83 个 phase/route scope，权重字节与 native stats 一致；只使用同进程/device/context 的 CUPTI kernel/memcpy 时间区间，排除 count/nonweight H2D 与 D2H，不用 host 发射时间或 warm cache 命中冒充传输重叠。`report.stages.performance.trace` 索引 SQLite、原始 trace 和 `gpu_overlap.json`。trace 是独立诊断，不混入 ABBA 耗时；SM89 的 0 交集是实际观测，不是缺失 kernel 数据。

SM89 affinity 为 17 个逻辑 CPU（85、118–133），quota=1700000/100000；SM120 为 36 个逻辑 CPU（318–353），quota=3600000/100000；两端 OMP/Torch threads=4。硬件/NUMA/PCIe 快照见 report telemetry，不宣称专属 GPU 或相同 CPU 条件。以上是合成样本的描述性结果，不提供统计显著性或业务性能保证；部分配置异步更慢，**保持 `async_staging=False` 默认**。没有把可选 C++ performance matrix 或完整 OS runtime/libc memcpy 归因写成已验收。

#### 15.15.4 制品身份、故障排空与泄漏验收边界

| 库 | SHA256 |
|---|---|
| SM89 prod | `8e4cafac712669d1adfa09cdcf4912336bdeec4993f4a5903f579e2ac55f966e` |
| SM89 testing | `367405ac63628ed040ea8ccb6bf30c2f41570005c576ffca29c3af86f521553b` |
| SM120 prod | `56628f72cd9014abece12d3055df3826639bc33a94fe9e948ba48212a8411a87` |
| SM120 testing | `d22e166573ea9df8b0dbea7d23bd11cc3e6aa7a76c563b3b7c896edbe7a7dd08` |

配套 sidecar 摘要保存在 report 的 `artifacts`；部署必须同时保留匹配的 `.so/.so.json`，testing 库不作为生产库。故障入口仅在 `CUSTOM_OPS_OFFLOAD_TESTING=1` 下注册，测试加载需显式允许 testing。九个注入点覆盖 GPU/pinned 分配、线程创建、running worker、首个 H2D 后、event 后、kernel 后、worker ready 后及 queued drain 前。

两目标的独立 C++ `--native-faults` full memcheck 均覆盖 14 个故障 case、7 个 queued/running/ready 状态 case、14 个并发 context case；`healthy_checks=16`、`concurrent_healthy_calls=42`，另有 4 次可恢复分配失败 ACK、10 次 poisoned ACK 拒绝和 2 次 callback unwind。teardown 的 allocated-before-trim=0、reserved-before-trim=31457280，使用官方 ATen CUDA/pinned allocator 清理缓存后 allocated/reserved 均为 0，memcheck 为 **0 bytes / 0 allocations / 0 errors**；没有使用 `cudaDeviceReset` 或 suppression。

Python 默认 memcheck 的验收范围是非法内存访问（`--leak-check no`），不是整个解释器进程零泄漏证明。SM120 历史 full leak-check 曾在 13 项测试通过后报告 82254353 bytes / 30 allocations、exit=99；原始失败不改写、不忽略。已修复 `invoke` 返回 frame 持有 output 的问题，但剩余观测不能据此全部归因或宣称全部修复。算子完整所有权泄漏验收由上述独立 C++ 完成；可选 `--python-full-leak-check` 仍保留严格失败判定，未解决的解释器/工具诊断集中登记为 R03。

## 16. 统一待办清单与 review 结论

### 16.1 维护规则与本次核对口径

**本节是全文唯一维护未完成事项的地方。**其他章节可以描述设计约束、实现事实、历史结果和当前限制，但只能引用本节任务 ID，不再各自增加“下一步/接续顺序”清单。后续发现的新缺口也在此登记；关闭任务须记录对应代码与验收证据，不能只因当前编码会话结束就写“全部待办关闭”。

2026-10-13 对照 offload 实现/测试/C++ runner 与既有验收收敛待办；2026-10-14 根据用户明确的通用算子仓库定位，再核对 `__init__.py`、`recsys.py`、`moe_offload.py` 及独立 runner，修订目标、职责和完成标准。**2026-10-14 的该次 review 仅修改设计文档，未开发新接口或新跑验收。**随后完成 T01–T09 开发与 2026-10-15 双架构完整验收，最终关闭记录见第 15.15、16.2 节；表中优先级保留立项含义，不是当前未完成计划或第 11 节 P0–P5 编号。

**任务 ID 保持可追踪，但范围已修订**：T01 从业务 loader 收敛为算子接入；T02 移除服务恢复编排；T05 从“等待真实模型”改为独立合成验收；T06 移除服务级总资源调度；T09 从旧 T01 拆出通用预编译构建/加载。不把移出仓库的工作记为“实现完成”，不保留业务模型为仓库阻塞依赖。

| 历史描述 / 容易误判的事项 | review 后的归类 |
|---|---|
| 第 4.4/10.4 节候选 schema、token mutation、AOTi 和 C++ 链路“尚未实现/验证” | 已完成；现行接口见第 15.13 节，功能证据见第 15.14 节，不重新列待办 |
| 第 4.5 节 CPU 权重加载与 C++ 序列化契约 | `ExpertStore.load/from_state_dict`、format1/2 与 `torch::jit::load` 已完成；公开接入封装/合成覆盖见 T01/T05，具体模型 loader/布局适配属于外部 |
| 第 7.5 节融合 Down epilogue 的未来式 | `fp32_accumulate` 已实现融合累加，保留 dtype 舍入边界且不分配 Down buffer；`route_buffer` 仍保留全局 Down，不列融合 epilogue 待办 |
| 第 9 节激活分块、静态/动态 cache | 通用机制已实现；合成性能/热集/扫描复测属 T04，真实业务收益及质量移出仓库待办 |
| 原第 15.6/15.9/15.11 节诊断、连续拷贝试验、有界 worker 与远程功能验证计划 | 诊断/worker 已完成，连续拷贝试验按数据撤回；历史阶段验收已归档，不重复立项 |
| 原第 15.12.6 与第 15.14.4 节未完成事项 | 仓库内部分收敛至 T01–T05/T09，业务模型/质量移出；不把配置预检和功能测试当作性能/全部故障闭环 |
| 第 12.1 节并发隔离、第 15.14 节明确跳过的专项、原始证据长期留存 | 原末尾清单遗漏的既有事项，补入 T06–T08 |

### 16.2 统一清单：T01–T09 全部 CLOSED

截至 2026-10-15，**9 项仓库内必做事项（T01–T09）全部 CLOSED，当前必做未完成数为 0**。关闭依据为第 15.15 节同一冻结源码的双架构完整验收；2026-10-16 收尾复核确认两报告五阶段均 passed、源码仍一致。本次收尾仅补录证据，没有重跑 GPU 测试。下表保留任务范围、立项时缺口与原完成标准以便追踪；“立项时能力”不是当前接口描述，现行实现见下方关闭证据表及第 15.15.1 节。可选 R 类和第 16.4 节外部职责不计入必做 backlog，不因无业务模型阻塞交付。

| ID / 立项优先级 | 事项与当前状态 | 立项时能力 / 历史缺口 | 保留的完成标准与边界 |
|---|---|---|---|
| **T01 / 高** | **可复用的算子生命周期与 AOTi 安全接入层：CLOSED** | `ExpertStore`/format2/冻结配置、原生预检及 Python 示例已存在；create/bind/validate/stats/close 有原生接口。完整的逐次预检和调用绑定仍主要依赖手工约定与固定测试装置，不能要求业务复用测试模块 | 在现有结构中提供 Python 易用封装及薄 C++/LibTorch 生命周期封装：按显式层映射绑定 CPU 权重/token，持有必要生命周期，作用域清理；接受**待执行制品的冻结契约**及官方 loader 调用入口/明确输入映射，串行化同 context 的“逐次预检→执行→失败检查”，close 不能与该区间穿插。拒绝配置/SM/关闭状态/token 与已绑定权重身份/version/输入错绑，验证拒绝发生在执行之前；正常、多层和相同结构新权重以合成图验收。不得从 context 自身代替制品契约；不推断任意业务图，也不认证同形状权重的业务发布身份。可先落地生命周期/预检，再接 T02 的运行期失败检查后闭环；T09 负责不隐式编译的加载。依据：第 4.6、15.13.3–15.13.5 节 |
| **T02 / 高** | **算子/AOTi 运行期失败可见性与 context 状态契约：CLOSED** | 图外预检只解决配置/绑定契约错误；实测 fallback/ProxyExecutor 可能忽略图内原生错误码，非法 route、allocator OOM、CUDA/worker 错误不因此可靠返回给调用方 | 定义执行中失败的公开可观察通路（评估受支持 shim/lowering 或原生 context 失败状态及图外检查，不预设方案已实现），区分最近调用成功/失败、可继续/仅可关闭；不能用上次成功 stats 或陈旧输出表示本次成功。通过 eager、Python AOTi、独立 C++ 合成负向场景证明调用方收到失败且不能接受部分输出，多层某层失败不能被末层成功掩盖，后续调用不得清掉未观察错误。预检失败不误标已执行；说明异步 CUDA 报错的可检测/不可恢复边界。与 T01/T03 联动；请求重试、业务降级、进程重启由外部决策，不承诺驱动故障全覆盖或全图回滚。依据：第 15.13.3 节 |
| **T03 / 高** | **可控故障注入与失败排空专项：CLOSED** | `StagingWorker` 已捕获/重抛 job 异常、处理 pthread 启动失败并提供 cancel/drain/shutdown；`Flight` 与 `close()` 管理排空。现有测试覆盖正常 join、worker 预算不足、非法 route 恢复和 sanitizer，但不等于真实执行中失败注入 | 建立可控、仅供测试且正式执行默认禁用的注入方式，覆盖 running worker 抛异常、pthread 创建失败、资源/pinned 分配失败、部分 DMA/event 已发射后失败，以及 queued/running/ready 状态下的 drain/shutdown。核对停止新发射、先排空后释放、线程退出、无悬空 DMA/资源泄漏、异常可见性及 context 后续处置；同步/异步及必要的跨 stream 场景分别验收。AOTi 错误可见性由 T02 一并验证，预算预拒绝和 memcheck/synccheck 不能代替此专项。running job 只安全排空，不扩大为强制取消、全 forward 回滚或驱动故障超时承诺。依据：第 12.4、15.12.1–15.12.3 节 |
| **T04 / 中；接口稳定后** | **现行版本合成性能、缓存行为与重叠基线：CLOSED** | 正常 sync/async 对照、CPU profiler、NVTX/nsys 与历史 trace 已有；最新 107 项是功能结果，不能作为最终接口/错误检查成本的性能数据。没有业务 workload 不妨碍算子基线 | 固定代码/库/工具链、合成 shape/route 种子、预算/cache，平衡 sync/async 测量顺序；报初始化/冷调用/稳态完整 wall p50/p95、calls/s 与有效 token/s、H2D、RAM/pinned/GPU charged 及 allocator/进程峰值。覆盖 disabled/static/reuse/LRU、重复热集/热集切换/扫描，验证传输统计与实际行为；明确合成收益不代表业务收益。用独立 trace 核对 bank/cap/wave/tile、host 等待与 kernel∩weight-H2D，需要归因才补 OS runtime/libc memcpy。SM89/SM120 分别构建，记录 CPU affinity/quota/NUMA/PCIe，不将 profiler 延迟或 warm 命中当正常耗时/传输重叠；记录 T01/T02 检查开销，保留同步/固定 cap 基线，无稳定收益保持同步默认。依据：第 12、15.12.4–15.12.5、15.14.2 节 |
| **T05 / 中；无需模型资料** | **支持范围内的参数化、较大 Prefill、权重适配与多层组合合成验收：CLOSED** | 已有多种 shape/route、wide dtype 和真实数据依赖的两层合成链，`LayerKeys` 支持 canonical fused/分离 gate/up/down；不是“完全没有参数化测试”，但尚无按公开支持范围汇总的系统化组合/容量矩阵 | 先映射已有用例，补缺不重复已有覆盖；以种子驱动多组 `S/H/I/E/K`、各层不同 E/I/合法 H 衔接与 dtype、动态 S 导出范围、不同层 ID/权重、重复/倾斜 route、单专家/多 wave/激活 tile/预算近界及超限拒绝。小形状对 FP32 数学参考与旧全 GPU 对拍，较大 S/较大权重用资源允许的受控 case 分开验证数值与容量；不能承诺任意尺寸。覆盖合成 state_dict fused/分离映射、显式 dtype 转换、format2/新 store-context 重新绑定，记录 load/转换/归档 retained bytes 与 RSS 峰值但不将后者写成预算保证。eager/Python AOTi/C++ 的矩阵分层运行，关键组合端到端验收；统计覆盖和跳过理由，不要求穷举笛卡尔积。具体业务模型布局/路由/质量移出待办，依赖仅为目标 GPU/工具链/合理测试预算。依据：第 2.1.1、4.5、10.2、12 节 |
| **T06 / 中** | **算子级多 context 并发、串行边界与关闭隔离：CLOSED** | context mutex 与独立 pool 已有；单 context 原生调用串行，预算属于各 executor。现有跨 stream/跨层测试不能代替多 context 并发；没有全进程总预算承诺 | 以多线程/不同调用 stream 并发独立 context/token，覆盖共享只读权重、不同路由/预算/cache、历史输出留存、关闭及一个 context 可控失败不影响健康 context，核对 slot/event/workspace/统计不串用、不死锁和资源归属。同 context 经 T01 接入层保证完整“预检→执行/检查/close”串行，不增加共享 context 并行执行。Python 与独立 C++ 关键用例均验收，约束测试总额度以免制造无关真实 OOM。全进程 RAM/GPU/pinned 协调、服务队列、CPU/NUMA/reserve 策略由外部负责；本仓只说明每 context 额度与不能求和推定的边界。依据：第 4.3、12.1、15.13.3 节 |
| **T07 / 低；补充覆盖** | **冻结配置版本的远程遗漏矩阵：CLOSED** | 最新远程 SM120 AOTi 使用 `--no-wide-bf16`，本机 wide BF16 已通过；最新远程未单独重跑 cudaMallocAsync，历史两卡 allocator 专项已通过。远程 legacy 21 项也未作为本轮新复测登记 | 补测所需远程目标的 wide BF16、`backend:cudaMallocAsync` 部署/跨 stream 子集及需要重新声明的原全 GPU 基线，明确工具链、源码/库版本、退出状态与范围。若可选矩阵不纳入本次目标，只记录跳过及理由，不能将历史通过或普通 allocator 检查改写成最新通过；结果作为新增检查点，不追溯篡改第 15.14.1 节。依据：第 15.14.1–15.14.3 节 |
| **T08 / 低；随新验收建立** | **可复现验收入口与轻量证据留存：CLOSED** | 文档有历史任务 ID/临时路径/命令摘要，部分原始记录已清理；临时日志不构成可迁移长期证据。不是需要建设制品发布平台 | 对新结果保存代码/库 hash、冻结配置、种子/shape/route、硬件/工具链、完整命令/退出状态、原始数值/性能样本与必要 trace 索引；采用现有可持久存储或 CI artifact，不把大二进制/trace 强行提交仓库，也不建设业务制品库。合成 benchmark/测试提供可重跑入口并明确矩阵/跳过理由。已删除历史记录注明不可恢复，不伪造补档，不为此全量重复历史验收。依据：第 15 节 |
| **T09 / 高；从旧 T01 拆出** | **离线构建、显式预编译加载与兼容边界：CLOSED** | `CustomOps.load()` 有源码指纹/arch 的 JIT 缓存及 dlopen 快路径；原生 `.so` 自注册，C++ runner 已显式 dlopen。Python `PrefillMoERuntime` 仍调用分组 `ensure_loaded`，缺缓存时可能 JIT；开发缓存不等于正式部署入口 | 复用现有分组 sources/flags，提供可复现的离线构建与产物定位/兼容信息，以及 Python 显式加载预编译 `.so`/严格禁 JIT 模式；加载后 schema 与 fake 可按需正确、幂等注册，`PrefillMoERuntime` 不绕回懒编译。C++ 复用自注册扩展/官方 loader，生命周期内不卸载注册库。记录目标 SM、PyTorch/LibTorch、CUDA、编译器/C++ ABI、schema/构建版本，前置拒绝可判断的不匹配；不承诺仅靠元数据发现所有 ABI 问题。合成新进程部署在无 C++/CUDA 源码、无 nvcc/Ninja 且禁编译的环境仍可加载执行，缺 `.so`/缺算子/错版本的负向场景清晰失败；不能静默回退 JIT/backend。限定 offload/`fuse_moe` 验收，不强制引入 wheel、稳定 C ABI、跨 SM 单制品或全仓重构。依据：`__init__.py`、`recsys.py`、第 4.6/10.4/15.14 节 |

#### 16.2.1 逐项关闭依据（2026-10-15）

| ID | 具体实现与验收证据 |
|---|---|
| T01 | `moe_offload.py` 的 `InputBinding/PrefillMoERuntime.prepare_inputs/invoke` 与 `csrc/fuse_moe/fuse_moe_offload_runtime.h` 的 `custom_ops::offload::Runtime`；显式制品契约、绑定/version、同 context 调用/close 串行化。`tests/test_fuse_moe_offload.py` 和官方 Python/独立 C++ 1/2/3 层 AOTi 覆盖正常及执行前拒绝，见 §15.15.1–15.15.2。 |
| T02 | `moe_offload_aoti.py` checked lowering 与 `fuse_moe_offload.cpp/fuse_moe_bindings.cpp` 的 begin/check/status/acknowledge；`tests/test_fuse_moe_offload_aoti_lowering.py` 10 项、eager sticky/ACK 和 9 个 AOTi 故障子进程证明失败先于下游中止。旧 unchecked 制品须重建；仅审计指定 Torch 版本，见 §15.15.1–15.15.2。 |
| T03 | 原生 testing 宏及九个故障注入点、worker cancel/drain/shutdown；`tests/fuse_moe_offload_aoti_runner.cpp --native-faults` 双端 full memcheck 的 14 fault、7 worker-state case，0 bytes/allocations/errors，见 §15.15.4。Python whole-process leak-check 不作零泄漏宣称，单列 R03。 |
| T04 | `tests/test_fuse_moe_offload_aoti.py --benchmark-locked` 与 delivery harness 的 ABBA、原始样本、cache/H2D/资源统计和 Nsight SQLite 分析；两端各 80 次调用/20 组完整对照及独立 GPU 时间线。公开调用检查计入 wall；没有稳定全面收益，保持同步默认，见 §15.15.3。 |
| T05 | `tests/test_fuse_moe_offload.py` 的输入映射、归档资源、权重适配、预算/tile/cache/route 覆盖与 `tests/test_fuse_moe_offload_aoti.py` 多层/动态 S/wide 组合；两端 sync/async 各 125 项无 skips，旧全 GPU 21 PASS，Python wide FP16/BF16 各 72 次观察。三层参考补齐 Act/Down dtype 舍入，不放宽阈值；仅声明所测范围，见 §15.15.2。 |
| T06 | Python/C++ runtime 完整临界区、独立 context 资源隔离；`common/fuse_moe_utils.cuh` 的 device+kernel `configure_dynamic_shared_memory` call_once 和 thread_local SM 查询修复并发属性竞态。两端 native 14 个并发 context case/42 healthy calls 与 Python 并发测试通过，见 §15.15.4；不提供全进程总预算。 |
| T07 | 同一冻结源码在 SM120 独立完整构建/部署：Python wide BF16、cudaMallocAsync sync/async 各 15、旧全 GPU 21 PASS 均有本轮证据。独立 C++ wide BF16 及可选 C++ performance matrix 未作为本轮必做，不借用历史结果，见 §15.15.2。 |
| T08 | `tests/run_fuse_moe_offload_delivery.py` 五阶段自动入口、`tests/test_fuse_moe_offload_delivery.py` 12 项证据解析测试、持久 run/report/raw/bench/trace 索引及源码/制品 hash；双端 `full_delivery_passed=true`、unmet 为空，复现命令与留存边界见 §15.15。 |
| T09 | `__init__.py` 的 `CustomOps.build/load_precompiled`、`recsys.py` 门面全局严格禁 JIT 与 format2 sidecar；`tests/test_custom_ops_loading.py` 50 项及每库 3 个无工具链官方部署子进程验证加载、兼容性拒绝和缺失时不 fallback。只在可判断范围前置检查，不保证任意 ABI，见 §15.15.1–15.15.2。 |

必做任务已按仓库能力闭环完成；不再保留历史推进顺序为当前计划。后续新增缺口仍在本节登记，不将业务集成或可选 R 类伪记为已实现。

### 16.3 可选验证与完整性补强

以下不计入 T01–T09 的当前必做范围；集中登记，避免隐含扩大算子仓库职责。将来选择对外正式承诺这些能力时，应转为 T 类任务并补验收。

| ID | 状态与发现 | 适用条件 / 完成标准 |
|---|---|---|
| **R01：兼容路径的预算自动探测专项** | 实现已有，尚无自动探测专项证据。`auto_ram_budget()` 解析物理可用页、cgroup v1/v2 与祖先剩余额度；原生 create 的 `0` 触发初始化探测。LOCKED 明确拒绝 `0`，因此不是冻结部署功能阻塞 | 若继续对外使用直接 create/`PrefillMoERuntime(...)` 的自动预算模式，补 cgroup v1/v2、挂载/命名空间与祖先限额、不可探测/非法数据拒绝等用例，验证保守额度并固定失败口径；不把自动探测当作未来独占保证。只使用 LOCKED 时可保持显式预算，不因此增加必做开发项 |
| **R02：可选制品完整性与期望权重版本校验** | 当前归档校验结构/version/config，不认证内容 hash 或业务发布身份；从旧 T01 必做标准移出，不记为已实现 | 若外部需要，可在离线导出/加载阶段提供可复用的归档内容摘要及调用方传入的期望标签比较；明确摘要对象、format/schema、Python/C++ 一致性、损坏/同形状错权重负向测试及信任边界。不得将 hash 当作签名/来源认证，不逐次扫描全权重，也不由本仓建立模型发布平台。外部已有签名/制品验证时直接复用，不强加第二套协议 |
| **R03：Python 全进程 full leak-check 诊断** | 可选、未解决，不属于本轮算子所有权交付承诺。SM120 历史严格运行报告 82254353 bytes / 30 allocations、exit=99，其中 device 81788928 bytes/12 allocations、pinned 465425 bytes/18 allocations。纯 Torch 最小观察也有 pinned allocator 缓存留存；返回 frame 持有 GPU 输入/权重的观测尚不足以确定工具/解释器根因。已修复 output frame 引用，不宣称所有残余已消失 | 若需要对外承诺整个 Python 进程零泄漏，再隔离框架缓存、解释器 frame 与 instrumentation 的根因并补严格复测。`--python-full-leak-check` 选用时仍严格失败，原始记录保留；当前 Python 非法访问检查与独立 C++ full ownership 0 bytes/allocations/errors 的验收范围见 §15.15.4，不用 suppression/reset 或历史改写将此项伪记完成。 |

第 15.12.2 节记录的 5090D 历史懒启动观测失败根因仍未知；测试已修订且既有复跑未复现，保留原失败记录，不伪称根因已确定，也不在没有复现线索时重复立项。

### 16.4 未立项方向与不扩大承诺的边界

**从仓库待办移出的外部职责（不是“待业务资料后本仓补做”）**：完整业务模型/package 加载与任意 checkpoint 布局自动适配、业务权重发布标签/签名/灰度/回滚、模型 router/前后处理/输入 batching、服务级队列/全进程资源协调/请求重试或进程重建、真实模型 latency/QPS 与 Recall/NDCG/排序质量。外部可基于公开算子和 AOTi 接入层完成这些工作；本仓提供接口约束和合成证据，不接管业务体系。

以下技术扩展**不是当前必做待办，也不是已实现能力**；若明确立项，应转为带完成标准的 T 类任务，再更新实现/验收状态。

- **存储/计算扩展**：SSD 后备、有界 RAM cache 与 expert extent、GDS/Unified Memory；CPU expert GEMM；低 bit/FP8 等权重表示、多 GPU 专家并行、TileLang/TMA offload backend。它们分别需要资源/失败协议、ISA/格式适配及独立质量/性能证据，不能由 RAM-only 运行时推定支持。
- **容量/调度/算术优化**：单专家分投影或权重 tile 流式 GEMM、per-route Down host 后备、token-outer 扫描、按 `m_e` 的 engine/tile 或代价平衡 wave 重排、LFU/预测/额外预取、直接使用未舍入 Down FP32 值、纯 handle 隐藏权重。当前已实现的 fused epilogue 保留 dtype 舍入边界，不能与另一算术配置混淆；这些方向均须由 T04 的性能与 T05 的容量/数值数据证明必要性后再考虑，不要求先有业务模型。
- **图与模型范围**：固定 compute-wave CUDA 子图 capture、全段流式 CUDA Graph capture/replay；Decode、推测解码/MTP、KV/prefix cache、跨层猜测路由、doorbell、反向/训练。AOTi 已是强制已验收能力，不能因 CUDA Graph 未支持而混淆二者。
- **不做无条件保证**：任意 shape/无限 S 或单个超预算专家可执行、单制品跨架构/任意 ABI 通用、所有 observer 崩溃隔离、running job 强制取消、驱动故障超时、全 forward 回滚，以及异步必然更快/业务必然无损。真实业务质量由外部在实际模型/workload 上给结论，不能由 T05 的合成验收推出。

### 16.5 算子仓库交付完成标准

在声明的 dtype/shape/layout/单 GPU SM 与已验证工具链范围内，具备以下证据即可判断算子能力与接入闭环完成；不以接入业务模型或完成 serving 上线为条件。

1. **能力正确且边界可判断**：CPU 权重流式、激活模式、静态/动态 cache 与旧全 GPU 基线的已有实现有支持范围矩阵，数值、容量、权重更新及组合缺口由 T05 补齐；不宣称未实现 backend 或任意尺寸支持。
2. **外部可直接集成**：T09 的预编译加载与 T01 的公开 Python/C++ 接入，不依赖测试内部实现、Python 调度或部署 JIT；官方 AOTi export/package/load/无 Python C++ 的代表性合成图闭环有可运行用例。
3. **失败不伪装成功**：T02/T03 定义并验证配置/输入/执行中失败通路、context 状态及先排空后释放；T06 验证串行边界和独立 context 隔离。无法恢复的驱动故障明确边界，而不是承诺全图事务。
4. **性能与证据可信**：T04 提供现行版本合成正常耗时、传输/预算及独立诊断基线；T07 对目标矩阵标明已测/未测，T08 提供可复现入口和原始结果索引。异步无收益也可完成验收，保持同步默认，不以调优结果虚构业务收益。

上述四项完成标准已在声明的支持范围内由 T01–T09 实现及第 15.15 节双架构验收满足，当前达到**算子仓库可交付状态**，不等于任意业务模型或 serving 上线验收。可选 R01–R03 与未立项方向未承诺完成；保持同步默认及预算/数值边界。2026-10-16 收尾仅更新本设计文档并复核冻结源码/报告，未新增 GPU 验收、未自动 commit/push。后续仅在第 16 节维护任务，第 15 节记录实际开发与验收证据。