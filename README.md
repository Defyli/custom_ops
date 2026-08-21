# custom_ops — 面向生成式推荐的高性能 CUDA 算子库

中文 | [English](README_EN.md)

面向生成式推荐系统的高性能 CUDA 算子库，包含两个核心算子：

- **`mha_fwd_with_mask`**：支持**任意加法 mask**（0=可见 / -inf=屏蔽）的
  FlashAttention-2 前向，深度适配消费级 GPU（sm120 / sm89），带 mask 性能
  显著优于 PyTorch SDPA
- **`mixed_gemm`**：混合精度 GEMM，解决推荐模型 **bf16 权重精度损失大、
  tf32 性能不足**的问题——bf16 tensor core 算主项 + 低精度 tensor core 算
  residual 补偿项，以接近 bf16 的开销恢复接近 fp32 的精度，且在 epilogue
  融合执行 bias 相加与 silu/gelu 激活

**亮点**

**注意力算子 `mha_fwd_with_mask`**

- **任意 mask 直达 softmax**：因果、滑动窗口、padding、随机稀疏（item 级屏蔽）
  等任意形态，无需改 kernel
- **sm120（RTX 5090D）**：标准场景 **160~197 TFLOPS**（cuBLAS bf16 实测峰值的
  70~85%），SDPA+mask 的 **2~2.7x**；小 grid 长序列场景最高 **53x**
- **sm89（RTX 4090D）**：开箱即用，标准场景 SDPA+mask 的 **1.1~1.9x**，
  小 grid 长序列场景最高 **34x**
- **原生 GQA**：K/V 头数整除 Q 头数即可，无需手动扩展
- **自适应 Split-KV**：cost model 自动选择 split 数，大 grid 自动退化为
  单 kernel，零开销、无需调参

**混合精度 GEMM `mixed_gemm`**

- **精度 ≈ fp32，速度 > tf32**：权重离线拆分为 bf16 主项 + fp8/int8
  residual 补偿项，消除权重的系统性舍入偏差（RMS 误差比纯 bf16 低 **2.3 倍**，
  最大误差低 **3.9 倍**），RTX 4090D 上为 fp32 matmul 的 **1.3~2.5x**、
  tf32 的 **1.2~1.6x**，大 shape 有效算力 **100 TFLOPS**
- **Epilogue 融合**：bias 相加 + silu/gelu 激活融合在 GEMM kernel 内，
  不产生额外 kernel 与中间显存
- **双 residual 后端**：FP8 e4m3（SM89+，需编译期 CUDA >= 12.4，小 M 略快、
  免 scale 存储）与 INT8 动态量化（SM80+，CUDA 11.8 即可，大 M 更快），
  精度一致，编译期自动选择
- **自适应 tile/split-K**：沿用原 TensorRT 插件的 wall-clock 启发式，
  小 M 自动 split-K，无需调参

**通用能力**

- **JIT 自动编译**：首次 import 自动构建，多进程安全，支持
  `torch.compile`/AOTI

> FA 算子完整的移植与优化过程记录见
> [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md)。

## 性能

### RTX 5090D (sm120, bf16)

标准场景：**160~197 TFLOPS**（cuBLAS bf16 实测峰值的 70~85%），SDPA+mask 的
**2~2.7x**：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d64 B=4 H=16 S=2048 | 377.4µs (182.1 TF) | 749.3µs | **1.99x** |
| d64 B=32 H=16 S=1024 | 697.2µs (197.1 TF) | 1388.2µs | **1.99x** |
| d128 B=4 H=16 S=2048 | 776.4µs (177.0 TF) | 1873.7µs | **2.41x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 772.1µs (178.0 TF) | 1877.7µs | **2.43x** |

小 grid + 长序列场景（Split-KV 自动生效）：SDPA+mask 的 **15~53x**：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d128 Sq=128 Sk=8192 | 24.7µs | 553.5µs | **22.4x** |
| d128 Sq=128 Sk=32768 | 41.2µs | 2199.0µs | **53.3x** |
| d128 Sq=1024 Sk=8192 | 36.2µs (118.6 TF) | 555.0µs | **15.3x** |
| d64 Sq=128 Sk=8192 | 20.7µs | 432.6µs | **20.9x** |

> 注：SDPA 带任意 `attn_mask` 时只能走 MemEfficient 后端（FlashAttention 后端不支持
> 任意 mask）。部分 shape 下本算子**带 mask 甚至比 SDPA 不带 mask 的 Flash 后端更快**。

### RTX 4090D (sm89, bf16)

sm89 路径为 cp.async 实现，同样具备自适应 Split-KV（自 sm120 移植）：小 grid
长序列自动切分 K 维填满 SM，大 grid 自动退化为单 kernel，无需任何配置。

测试环境：RTX 4090D / PyTorch 2.6.0 / CUDA 11.8，benchmark 默认参数
（见 [Benchmark](#benchmark) 小节）。

标准场景（大 grid）：

| Shape | custom | SDPA+mask | 加速比 |
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

小 grid + 长序列场景（Split-KV 自动生效）：

| Shape | custom | SDPA+mask | 加速比 | SDPA flash* |
|---|---|---|---|---|
| d128 Sq=128 Sk=8192 | 22.5µs (23.8 TF) | 462.8µs | **20.55x** | 29.7µs |
| d128 Sq=128 Sk=32768 | 53.2µs (40.3 TF) | 1824.8µs | **34.27x** | 52.2µs |
| d128 Sq=512 Sk=8192 | 39.9µs (53.8 TF) | 429.1µs | **10.74x** | 37.9µs |
| d128 Sq=1024 Sk=8192 | 55.3µs (77.7 TF) | 428.3µs | **7.74x** | 52.2µs |
| d128 H=2 Hk=1 Sq=512 Sk=16384 | 96.3µs (89.2 TF) | 840.8µs | **8.73x** | 122.9µs |
| d64 Sq=128 Sk=8192 | 19.5µs (13.8 TF) | 412.6µs | **21.21x** | 16.4µs |
| d64 Sq=1024 Sk=8192 | 41.8µs (51.3 TF) | 415.7µs | **9.94x** | 33.8µs |
| d64 H=2 Hk=1 Sq=512 Sk=16384 | 57.3µs (74.9 TF) | 809.0µs | **14.11x** | 69.6µs |

> \* SDPA flash 列为不带 mask 的 FlashAttention 后端参照（不支持任意 mask）；
> 公平对照是 SDPA+mask 列。得益于 Split-KV，部分小 grid shape 已追平甚至超过
> 该参照（如 d128 H=2 Hk=1 Sq=512 Sk=16384: 96.3µs vs 122.9µs）。

作为参照：未启用 Split-KV 时，小 grid 长序列 shape 只能由单个 CTA 串行处理全部
KV 块，耗时 234~935µs；Split-KV 自动生效后降至 19~53µs（**10~18x**）。

### mixed_gemm（RTX 4090D，FP8 / INT8 residual 双后端）

推荐 MLP 层 `silu(x @ W^T + b)`（fp32 权重/输入）。测试环境：RTX 4090D /
PyTorch 2.11 / CUDA 12.8（FP8 与 INT8 后端同机对照）；编译期 CUDA < 12.4 时
FP8 自动降级 INT8（精度相同）。

mixed_gemm 最优后端为 fp32 matmul 的 **1.3~2.5x**、tf32 的 **1.2~1.6x**，
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

- **小 M（≤64）FP8 略快**（7~10%，量化 kernel 更简单）；**大 M INT8 更快**
  （最高 21%）；`vs fp32†` 取每行更优后端的倍率。`backend="auto"` 在 FP8
  可用时默认选 FP8（精度相同、免 per-channel scale 存储），大 M 追求极致
  性能可显式 `backend="int8"`
- 两种 residual 后端精度一致：权重舍入偏差均被消除，剩余误差主导项是激活的
  bf16 舍入噪声（两种量化精度都已足够细）
- CUDA 11.8 编译时同一 INT8 kernel 大 shape 慢 ~17%（nvcc 代码生成差异，
  4096³ 实测 1658µs），小 shape 不受影响；建议用较新 CUDA 编译

> \* bf16 列为 `bf16(x) @ bf16(W)^T`（权重离线预转，与 mixed_gemm 的离线权重拆分
> 对等）+ fp32 epilogue：速度快 1.3~1.7x（单次 GEMM vs 主项+补偿双 GEMM），但
> 权重舍入误差完全未补偿。

## 环境要求

- NVIDIA GPU：sm120（RTX 5090D，TMA 主路径，充分测试）或 sm89（RTX 4090D，
  cp.async 路径（含自 sm120 移植的 Split-KV），已测试）；其余 sm80+ 架构理论上
  可编译运行，未验证
- CUDA >= 11.8（FA sm89 路径 / mixed_gemm INT8 后端）/ >= 12.4（mixed_gemm FP8
  后端，SM89+）/ >= 12.8（FA sm120a），GCC >= 9
- PyTorch >= 2.1（CUDA 版本），bf16

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
```

编译缓存默认在 `~/.cache/torch_extensions`，可用 `TORCH_EXTENSIONS_DIR` 指定；
多进程同时首次编译由文件锁保护，是安全的。

## API

### `ops.mha_fwd_with_mask(q, k, v, mask) -> Tensor`

FlashAttention-2 前向，支持任意 bf16 加法 mask。

| 参数 | shape | 说明 |
|---|---|---|
| `q` | (B, H, Sq, d) | bf16 CUDA 连续张量 |
| `k`, `v` | (B, Hk, Sk, d) | bf16 CUDA 连续张量 |
| `mask` | (B, 1, Sq, Sk) | bf16 加法 mask，0=可见 / -inf=屏蔽 |
| 返回 | (B, H, Sq, d) | bf16 |

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

### `split_mixed_precision_weight(w, scale=1/256, backend="auto") -> (w_high, w_low, w_scale)`

离线（模型加载时一次性）将 fp32 权重拆分为 mixed_gemm 所需的三元组：
`w ≈ w_high + w_low * scale`。`backend="auto"` 时当前机器可用 FP8 则选 FP8，
否则降级 INT8。纯 PyTorch 实现，无需编译算子库。

### 调优环境变量（默认 auto 即接近最优，仅调优/调试用）

| 环境变量 | 作用 |
|---|---|
| `FA_NUM_SPLITS=n` | 强制 split KV 的 split 数（0=auto cost model） |
| `FA_SPLITKV=0` | 禁用 split KV |
| `FA_PERSISTENT=1` | 启用 persistent kernel（d128 部分场景 +2~5%） |
| `GEMM_MIXED_FORCE_SPLITK=n` | 强制 mixed_gemm 的 split-K 值（1/2/4/8/16，调试用） |

## Examples

```bash
python examples/basic_usage.py        # 最小示例：调用 + 与 SDPA 校验
python examples/custom_mask_demo.py   # 4 种典型 mask：causal / 滑窗 / padding / 随机稀疏
python examples/gqa_example.py        # GQA：无需扩展 K/V 头
python examples/mixed_gemm_demo.py    # 混合精度 GEMM：权重拆分 + epilogue 融合 + 精度对比
```

## Benchmark

```bash
# 全量：标准场景 + 小 grid 长序列（splitkv）场景
python benchmark/benchmark_fa.py

# 指定单组 / 单个 shape
python benchmark/benchmark_fa.py --suite standard
python benchmark/benchmark_fa.py --suite splitkv
python benchmark/benchmark_fa.py --shape 4 16 16 2048 2048 128

# 调整迭代与输出
python benchmark/benchmark_fa.py --warmup 20 --iters 100 --mask-ratio 0.3 --csv result.csv
```

输出每组 shape 的 custom / SDPA+mask / SDPA flash（参照）耗时、TFLOPS 与加速比，
并对每个 shape 做一次 SDPA 数值校验。

```bash
# mixed_gemm：对比 fp32 / tf32 / bf16 matmul 的耗时与精度
python benchmark/benchmark_mixed_gemm.py                       # 后端 auto（FP8 可用则 FP8）
python benchmark/benchmark_mixed_gemm.py --backend int8        # 指定 residual 后端
python benchmark/benchmark_mixed_gemm.py --shape 4096 4096 4096 --csv result.csv
```

## 仓库结构

```
custom_ops/
├── __init__.py            # CustomOps 通用 JIT 算子加载框架（基类）
├── recsys.py              # RecsysOps：mha_fwd_with_mask / mixed_gemm 封装 + 权重拆分 helper
├── csrc/
│   ├── recsys_bindings.cpp        # torch.ops 注册入口
│   ├── fa/                        # FA2 + mask kernel（sm89 + sm120 双路径，互不依赖）
│   │   ├── fa_fwd_op.cu           # 算子入口 / 架构分发
│   │   ├── fa_fwd_launch.h        # 各架构 launcher + Split-KV cost model
│   │   ├── sm89/fa_fwd_kernel.h   # sm89 cp.async kernel：base / splitkv / combine
│   │   ├── sm120/fa_fwd_sm120.h   # sm120 TMA 流水线 kernel / splitkv / persistent / combine
│   │   ├── cpu/                   # CPU 参考实现（未接入算子分发，仅供参考）
│   │   └── common/                # 两路共用的参数包 / softmax / utils
│   └── mixed_gemm/                # 混合精度 GEMM kernel（自 TRT 插件移植）
│       ├── mixed_gemm_op.cu       # torch 算子入口：校验 / workspace / 分发
│       ├── gemm_bf16xfp32_sm80.cu # kernel：tile/split-K 启发式 + bf16 主项 + fp8/int8 补偿双 GEMM
│       └── gemm_bf16xfp32_sm80.h  # kernel 入口声明 + FP8 编译期守卫
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
数据通路参考 FA3/hopper 的 TMA 写法移植至 sm120），并依赖 NVIDIA CUTLASS/CuTe
（已作为头文件内置于 `thirdparty/`）。如果本项目对您有帮助，请同时引用原项目：

- FlashAttention 官方仓库：https://github.com/Dao-AILab/flash-attention
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
