# custom_ops — 支持任意 Mask 的 FlashAttention-2 (sm120/Blackwell 优化)

中文 | [English](README_EN.md)

面向生成式推荐系统场景的高性能 CUDA 算子库。核心算子 `mha_fwd_with_mask` 是支持
**任意加法 mask**（0=可见 / -inf=屏蔽）的 FlashAttention-2 前向实现，针对消费级
Blackwell（RTX 5090D, sm120）深度优化：TMA + mbarrier 多级流水线、Split KV 自适应
并行，性能显著超越 PyTorch SDPA；同时内置 sm89（RTX 40）cp.async 基线路径，
非 Blackwell GPU 开箱即用，实测同样稳定超越 SDPA（见下方性能小节）。

> 完整的移植与优化过程记录见 [docs/fa_sm120_porting_and_optimization.md](docs/fa_sm120_porting_and_optimization.md)。

## 特性

- **任意加法 mask**：bf16 mask 直接参与 softmax（0=可见 / -inf=屏蔽），支持因果、
  滑动窗口、padding、随机稀疏（item 级屏蔽）等任意形态
- **GQA 原生支持**：K/V 头数整除 Q 头数即可，无需手动扩展
- **Split KV 自适应**：小 grid 长序列场景自动拆分 K 维提升 SM 利用率，
  cost model 选择最优 split 数，大 grid 自动退化为单 kernel 零开销
- **JIT 编译框架**：`CustomOps` 基类提供自动编译/加载、多进程文件锁、
  GPU 架构自动探测、`torch.compile`/AOTI fake 注册，可复用于其他自定义算子

## 性能

### RTX 5090D (sm120, bf16)

标准场景：**160~197 TFLOPS**（cuBLAS bf16 实测峰值的 70~85%），SDPA 的 **2~2.7x**：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d64 B=4 H=16 S=2048 | 377.4µs (182.1 TF) | 749.3µs | **1.99x** |
| d64 B=32 H=16 S=1024 | 697.2µs (197.1 TF) | 1388.2µs | **1.99x** |
| d128 B=4 H=16 S=2048 | 776.4µs (177.0 TF) | 1873.7µs | **2.41x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 772.1µs (178.0 TF) | 1877.7µs | **2.43x** |

小 grid + 长序列场景（splitkv 自动生效）：SDPA 的 **15~53x**：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d128 Sq=128 Sk=8192 | 24.7µs | 553.5µs | **22.4x** |
| d128 Sq=128 Sk=32768 | 41.2µs | 2199.0µs | **53.3x** |
| d128 Sq=1024 Sk=8192 | 36.2µs (118.6 TF) | 555.0µs | **15.3x** |
| d64 Sq=128 Sk=8192 | 20.7µs | 432.6µs | **20.9x** |

> 注：SDPA 带任意 `attn_mask` 时只能走 MemEfficient 后端（FlashAttention 后端不支持
> 任意 mask）。部分 shape 下本算子**带 mask 甚至比 SDPA 不带 mask 的 Flash 后端更快**。

### RTX 4090D (sm89, bf16)

sm89 走 cp.async 基线 kernel（TMA / 多级流水线 / Split KV 为 sm120 专属优化），
tile 配置：d64 → (M=128, N=128, 8 warps)，d128 → (M=64, N=64, 4 warps)。
对 SDPA+mask（MemEfficient 后端）保持 **1.14~1.86x** 优势，d128 大 shape 达
**120+ TFLOPS**。

测试环境：RTX 4090 D（sm_89）/ PyTorch 2.6.0 / CUDA 11.8；benchmark 默认参数
（mask_ratio=0.1、warmup=10、iters=50，迭代间 256MB L2 刷新，取中位数）。

标准场景（大 grid，全部 9 组）：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d64 B=1 H=16 S=1024 | 73.7µs (58.3 TF) | 84.0µs | **1.14x** |
| d64 B=4 H=16 S=2048 | 587.8µs (116.9 TF) | 784.4µs | **1.33x** |
| d64 B=1 H=8 S=8192 | 1166.7µs (117.8 TF) | 1680.4µs | **1.44x** |
| d64 B=32 H=16 S=1024 | 1196.0µs (114.9 TF) | 1497.1µs | **1.25x** |
| d128 B=1 H=16 S=1024 | 103.4µs (83.1 TF) | 170.0µs | **1.64x** |
| d128 B=4 H=16 S=2048 | 1103.9µs (124.5 TF) | 1941.5µs | **1.76x** |
| d128 B=1 H=8 S=8192 | 2174.0µs (126.4 TF) | 3815.3µs | **1.76x** |
| d128 B=4 H=16 Hk=4 S=2048 (GQA) | 1101.7µs (124.8 TF) | 2044.8µs | **1.86x** |
| d128 B=32 H=16 S=1024 | 2269.2µs (121.1 TF) | 3874.8µs | **1.71x** |

小 grid + 长序列场景（sm89 上 Split KV 不生效，全部 8 组）：

| Shape | custom | SDPA+mask | 加速比 |
|---|---|---|---|
| d128 Sq=128 Sk=8192 | 238.6µs | 431.1µs | **1.81x** |
| d128 Sq=128 Sk=32768 | 934.9µs | 1694.7µs | **1.81x** |
| d128 Sq=512 Sk=8192 | 237.6µs | 430.1µs | **1.81x** |
| d128 Sq=1024 Sk=8192 | 239.5µs (17.9 TF) | 428.0µs | **1.79x** |
| d128 H=2 Hk=1 Sq=512 Sk=16384 | 473.1µs | 841.7µs | **1.78x** |
| d64 Sq=128 Sk=8192 | 233.5µs | 411.6µs | **1.76x** |
| d64 Sq=1024 Sk=8192 | 233.5µs (9.2 TF) | 413.7µs | **1.77x** |
| d64 H=2 Hk=1 Sq=512 Sk=16384 | 461.8µs | 810.0µs | **1.75x** |

sm89 结果说明：

- **加速比低于 sm120（2~2.7x）**：sm89 路径是 FA2 式 cp.async 基线，没有 TMA /
  多级流水线 / persistent 等优化，主要价值是让非 Blackwell GPU 开箱即用；
  sm89 上也无法超越 SDPA 不带 mask 的 Flash 后端（“带 mask 超 Flash”是
  sm120 上的现象）。
- **小 grid 长序列是 sm89 的短板**：Split KV 尚未移植到 sm89，B=1、H=1、Sq=128
  这类 shape 只启动 1 个 CTA 串行处理 64~256 个 KV 块（延迟受限，耗时基本与 Sk
  成正比），虽然仍是 SDPA+mask 的 1.75~1.8x，但远落后于内置 KV 切分的 Flash
  后端（如 d64 Sq=128 Sk=8192 Flash 仅 16µs）——把 splitkv 移植到 sm89 尚有
  可观优化空间。
- d64 加速比（1.14~1.44x）低于 d128（1.64~1.86x）：主因是 SDPA MemEfficient
  在 d64 上本身表现更好（约 88 TF vs d128 的 71 TF），而本算子两种 head dim
  效率接近（115~126 TF）。

## 环境要求

- NVIDIA GPU：sm120（RTX 5090D，TMA 主路径，充分测试）或 sm89（RTX 4090D，
  cp.async 基线路径，已测试）；其余 sm80+ 架构理论上可编译运行，未验证
- CUDA >= 11.8（sm89 路径）/ >= 12.8（sm120a），GCC >= 9
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

### 调优环境变量（默认 auto 即接近最优，仅调优/调试用）

| 环境变量 | 作用 |
|---|---|
| `FA_NUM_SPLITS=n` | 强制 split KV 的 split 数（0=auto cost model） |
| `FA_SPLITKV=0` | 禁用 split KV |
| `FA_PERSISTENT=1` | 启用 persistent kernel（d128 部分场景 +2~5%） |

## Examples

```bash
python examples/basic_usage.py        # 最小示例：调用 + 与 SDPA 校验
python examples/custom_mask_demo.py   # 4 种典型 mask：causal / 滑窗 / padding / 随机稀疏
python examples/gqa_example.py        # GQA：无需扩展 K/V 头
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

## 仓库结构

```
custom_ops/
├── __init__.py            # CustomOps 通用 JIT 算子加载框架（基类）
├── recsys.py              # RecsysOps：mha_fwd_with_mask 算子封装
├── csrc/
│   ├── recsys_bindings.cpp        # torch.ops 注册入口
│   └── fa/                        # FA2 + mask CUDA kernel（sm120 优化）
│       ├── fa_fwd_op.cu           # 算子入口 / 路径分发
│       ├── fa_fwd_sm120.h         # sm120 TMA 流水线 kernel / splitkv / persistent
│       ├── fa_fwd_kernel.h        # 计算 mainloop / softmax / epilogue
│       └── ...
├── thirdparty/            # CUTLASS / CuTe（头文件依赖）
├── benchmark/             # 性能基准测试
├── examples/              # 使用示例
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
