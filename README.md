# custom_ops — 通用 PyTorch CUDA 算子插件库框架

提供一套可复用的 CUDA 算子注册/加载框架，业务方继承 `CustomOps` 基类并实现极少量配置，即可获得完整的 JIT 编译、快速加载、多进程安全、GPU 架构自动探测等能力。

`RecsysOps` 是框架的**内置实现**，将推荐系统三大核心算子直接内置于包中，可开箱即用。

## 目录结构

```
custom_ops/
├── __init__.py                  # CustomOps 通用基类 + RecsysOps 导出 + ops 单例
├── recsys.py                    # RecsysOps 内置算子库（包级固有）
├── csrc/
│   ├── custom_ops_macros.h      # C++ 注册宏（CUSTOM_OP_DISPATCH_FN 等）
│   ├── recsys_bindings.cpp      # RecsysOps 算子注册入口（命名空间 recsys_ops）
│   ├── fa/                      # Flash Attention 2 kernel 源码
│   ├── pack/                    # pack_and_prepare_b1 kernel 源码
│   └── jagged/                  # jagged_pool_and_collect kernel 源码
├── thirdparty/                  # cutlass / cute 头文件（软链）
├── example/                     # 自定义子类示例（FaOps，演示框架用法）
│   ├── __init__.py              # FaOps 子类（继承 CustomOps）
│   └── csrc/bindings.cpp        # 算子注册文件（C++，约 40 行）
└── test_custom_ops.py           # 自测脚本（12 个 section）
```

## 内置算子库：RecsysOps

三大核心算子已作为包级固有功能内置，无需任何额外配置，直接导入即可使用：

```python
from custom_ops import ops          # 推荐：使用包级全局单例（import 时自动加载）
from custom_ops import RecsysOps    # 也可导入类，自行实例化

if ops.is_available():
    # Flash Attention 2 前向（支持任意 bf16 加法 mask）
    out = ops.mha_fwd_with_mask(q, k, v, mask)

    # 融合算子：pack tokens + RoPE gather + attn_mask 构建
    h_dense, cos, sin, attn_mask, item_mask, offsets = ops.pack_and_prepare_b1(
        h_s, h_c, h_i, s_len, c_len, i_len,
        static_cos, static_sin, S_max, S_mask,
    )

    # 融合 jagged pooling（v3：零 H→D copy）
    pool_values_out, pool_lengths_out, out_val_splits, out_len_splits = \
        ops.jagged_pool_and_collect(
            pool_values, pool_lengths,
            pool_val_splits, pool_len_splits,
            n_pooling, reduce_mode, ones_cache,
        )
```

### 三个算子说明

| 算子 | 签名 | 说明 |
|------|------|------|
| `mha_fwd_with_mask` | `(q, k, v, mask) → Tensor` | FA2 前向，`d ∈ {64,128}`，支持 GQA |
| `pack_and_prepare_b1` | `(h_s,h_c,h_i, s/c/i_len, cos,sin, S_max,S_mask) → Tensor[6]` | 融合 pack + RoPE + attn_mask，返回 `h_dense, cos, sin, attn_mask, item_mask, offsets` |
| `jagged_pool_and_collect` | `(pool_values,pool_lengths, val/len_splits, n_pooling, reduce_mode, ones_cache) → Tensor[4]` | 批量 segment reduce，`reduce_mode`：0=mean / 1=sum，返回 `pool_values_out, pool_lengths_out, out_val_splits, out_len_splits` |

> `s_len / c_len / i_len` 需在 Python 侧先 `.item()` 提取为 Python int 后传入，避免 D2H 同步引发 NaN。

### Python 等价实现

以下 Python 代码与对应 CUDA 算子的输出完全等价，可用于理解算子语义或在无 GPU 环境下作为回退实现。

#### `mha_fwd_with_mask`

Flash Attention 2 的 Python 等价实现直接使用 PyTorch 内置 SDPA：

```python
import torch.nn.functional as F

def mha_fwd_with_mask_py(q, k, v, mask):
    """
    Python 等价（精度略低，无 FlashAttention 内存优化）。
    q, k, v : (B, H, S, d) bfloat16
    mask    : (B, 1, Sq, Sk) bfloat16，0=可见 / -inf=屏蔽
    """
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

#### `pack_and_prepare_b1`

融合了三个步骤：pack jagged tokens → gather RoPE cos/sin → build attn_mask。

```python
import torch

def pack_and_prepare_b1_py(h_s, h_c, h_i, s_len, c_len, i_len,
                            static_cos, static_sin, S_max, S_mask):
    """
    h_s, h_c, h_i : (n, D) bfloat16
    s_len, c_len, i_len : Python int（勿传 CUDA Tensor，会 D2H 同步）
    static_cos/sin : (1, S_max, 1, head_dim) bfloat16
    返回：h_dense, cos, sin, attn_mask, item_mask, offsets
    """
    device, dtype = h_s.device, h_s.dtype
    valid_tokens = s_len + c_len + i_len
    item_pos     = s_len + c_len
    D            = h_s.shape[-1]

    # ── Step 1: pack jagged → dense ──────────────────────────────────────────
    h_dense = torch.zeros(1, S_max, D, dtype=dtype, device=device)
    if s_len > 0: h_dense[0, :s_len]                 = h_s
    if c_len > 0: h_dense[0, s_len:item_pos]         = h_c
    if i_len > 0: h_dense[0, item_pos:valid_tokens]  = h_i

    # ── Step 2: item_mask & offsets ──────────────────────────────────────────
    item_mask = torch.zeros(valid_tokens, dtype=torch.bool, device=device)
    if i_len > 0:
        item_mask[item_pos:] = True
    offsets = torch.tensor([0, valid_tokens], dtype=torch.int64, device=device)

    # ── Step 3: gather RoPE cos/sin（pos_id 在 item_pos 处截断）────────────
    # ubc+ctx 各位置独立编码，item 位置共享同一 pos_id = item_pos
    pos_ids = torch.arange(S_max, device=device).clamp_max(item_pos)
    cos_out = static_cos[:, pos_ids, :, :]   # (1, S_max, 1, head_dim)
    sin_out = static_sin[:, pos_ids, :, :]

    # ── Step 4: build attn_mask ──────────────────────────────────────────────
    # 规则：causal + item 间互相屏蔽（item 只看自己）+ padding 行全 -inf
    causal = torch.ones(S_mask, S_mask, dtype=torch.bool, device=device).tril()
    attn_mask = torch.zeros(S_mask, S_mask, dtype=dtype, device=device)
    attn_mask = attn_mask.masked_fill(~causal, float('-inf'))

    # item 间屏蔽：两个不同 item 的 token 互相不可见
    im = torch.zeros(S_mask, dtype=torch.bool, device=device)
    im[:valid_tokens] = torch.cat([
        torch.zeros(item_pos, dtype=torch.bool, device=device),
        torch.ones(i_len,    dtype=torch.bool, device=device),
    ])
    cross = im.unsqueeze(1) & im.unsqueeze(0)               # (S, S)
    cross = cross & ~torch.eye(S_mask, dtype=torch.bool, device=device)  # 自身可见
    attn_mask = attn_mask.masked_fill(cross, float('-inf'))

    # padding 行全 -inf（防止全零 Q 产生均匀 softmax → NaN）
    row_ids = torch.arange(S_mask, device=device)
    pad_rows = (row_ids >= valid_tokens).view(S_mask, 1).expand(S_mask, S_mask)
    attn_mask = attn_mask.masked_fill(pad_rows, float('-inf'))

    return h_dense, cos_out, sin_out, attn_mask.unsqueeze(0).unsqueeze(0), item_mask, offsets
```

#### `jagged_pool_and_collect`

对多个 pooling 特征分别做 segment reduce（mean 或 sum），输出 pooled values 及切分 splits：

```python
import torch

def jagged_pool_and_collect_py(pool_values, pool_lengths,
                                pool_val_splits, pool_len_splits,
                                n_pooling, reduce_mode, ones_cache):
    """
    pool_values     : (total_rows, D)  拼接的所有 pooling 特征 values
    pool_lengths    : (total_len_rows,) int32  拼接的 lengths
    pool_val_splits : list[int]  len=n_pooling+1，values 行偏移
    pool_len_splits : list[int]  len=n_pooling+1，lengths 行偏移
    n_pooling       : int
    reduce_mode     : 0=mean / 1=sum
    ones_cache      : (max_n,) int32 CUDA（输出 lengths 切片自此获取，避免 alloc）
    返回：pool_values_out, pool_lengths_out, out_val_splits, out_len_splits
    """
    out_v_list, out_l_list = [], []

    for i in range(n_pooling):
        # 取出第 i 个特征的 values 与 lengths
        v = pool_values[pool_val_splits[i] : pool_val_splits[i + 1]]   # (n_i * seq_len, D)
        l = pool_lengths[pool_len_splits[i] : pool_len_splits[i + 1]]  # (n_i,)

        n_samp  = l.shape[0]
        seq_len = int(l[0].item())     # 假设每 sample 等长（实际算子支持不等长）
        D       = v.shape[-1]

        # segment reduce（fp32 累加保精度，结果转回原 dtype）
        vf = v.float().view(n_samp, seq_len, D)
        pooled = vf.mean(dim=1) if reduce_mode == 0 else vf.sum(dim=1)
        out_v_list.append(pooled.to(v.dtype))
        out_l_list.append(ones_cache[:n_samp])  # lengths 全为 1

    # 拼接并计算输出 splits
    out_val_splits = [0] * (n_pooling + 1)
    out_len_splits = [0] * (n_pooling + 1)
    for i in range(n_pooling):
        out_val_splits[i + 1] = out_val_splits[i] + out_v_list[i].shape[0]
        out_len_splits[i + 1] = out_len_splits[i] + out_l_list[i].shape[0]

    pool_values_out  = torch.cat(out_v_list,  dim=0)
    pool_lengths_out = torch.cat(out_l_list,  dim=0)
    t_vs = torch.tensor(out_val_splits, dtype=torch.int64)
    t_ls = torch.tensor(out_len_splits, dtype=torch.int64)
    return pool_values_out, pool_lengths_out, t_vs, t_ls
```

> **性能对比**：CUDA 版本在生产规模（185 特征，batch=100，D=64）比上述 Python 实现快约 **10~20×**，且消除了多次 H→D copy 和碎片化 kernel launch。

---

## 快速上手（自定义算子）

如需注册自己的 CUDA 算子，继承 `CustomOps` 基类：

### Step 1：实现算子（C++）

新建 `your_project/csrc/bindings.cpp`：

```cpp
// 1. 定义 torch.ops 命名空间
#define CUSTOM_OPS_NAMESPACE my_ops

// 2. 包含框架宏
#include "custom_ops_macros.h"

// 3. 包含算子声明
#include "my_op.h"

// 4. 一行声明分发函数（自动路由 CUDA / CPU）
CUSTOM_OP_DISPATCH_FN(
    my_op,
    (const torch::Tensor& x, int64_t n),
    (x, n),
    x    // 用于判断设备的 tensor
)

// 5. 注册 schema
CUSTOM_OPS_LIBRARY_BEGIN
    CUSTOM_OP_SCHEMA(my_op, "Tensor x, int n", "-> Tensor")
CUSTOM_OPS_LIBRARY_END

// 6. 绑定实现
CUSTOM_OPS_IMPL_BEGIN
    CUSTOM_OP_BIND(my_op)
CUSTOM_OPS_IMPL_END

// 7. pybind 入口（必须有）
CUSTOM_OPS_PYBIND_MODULE("my ops library")
```

算子实现遵循命名约定：
- GPU 实现：`my_op_cuda(const torch::Tensor& x, int64_t n)`
- CPU 实现：`my_op_cpu(const torch::Tensor& x, int64_t n)`

### Step 2：继承 CustomOps（Python）

```python
import os
from custom_ops import CustomOps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyOps(CustomOps):
    namespace    = "my_ops"         # 对应 bindings.cpp 中的宏定义
    so_name      = "my_ops_kernel"  # 编译产物名（不含 .so）
    required_ops = ["my_op"]        # 必需算子，缺失则加载失败
    optional_ops = []               # 可选算子，缺失仅警告

    def get_sources(self):
        return [
            os.path.join(_THIS_DIR, "csrc", "bindings.cpp"),
            os.path.join(_THIS_DIR, "csrc", "my_op.cu"),
        ]

    def get_include_dirs(self):
        custom_ops_dir = os.path.dirname(os.path.dirname(__file__))
        return [
            os.path.join(custom_ops_dir, "thirdparty"),  # cutlass/cute
            os.path.join(custom_ops_dir, "csrc"),        # custom_ops_macros.h
        ]

ops = MyOps().load()
```

### Step 3：注册 fake 实现（AOTI / torch.compile 支持）

若需要通过 `torch.compile` 或导出为 AOT Inductor（AOTI）模型，必须为每个算子提供 **fake（meta）实现**，告知 tracing 引擎输出的 shape/dtype，无需实际执行 CUDA 代码。

覆盖 `register_fake_impls()` 方法，`load()` 成功后会自动调用，无需手动触发：

```python
class MyOps(CustomOps):
    # ...（同上）

    def register_fake_impls(self):
        # lambda 参数与实际算子相同，返回与输出 shape/dtype 匹配的空 tensor
        self.register_fake_for(
            "my_op",
            lambda x, n: torch.empty_like(x),
        )
        # 多输出算子示例（返回 tuple）
        self.register_fake_for(
            "my_op_two_out",
            lambda x: (torch.empty_like(x), torch.empty(x.shape[0], dtype=torch.int32, device=x.device)),
        )
```

> **不覆盖此方法时不影响普通 eager 模式运行**，仅在使用 `torch.compile` / `torch.export` / AOTI 时才必须实现。

### Step 4：调用

```python
from your_project import ops

if ops.is_available():
    result = ops.my_op(tensor, n)
else:
    result = fallback_implementation(tensor, n)
```

---

## CustomOps 基类接口

| 方法 / 属性 | 说明 |
|------------|------|
| `namespace` | `torch.ops` 命名空间，对应 `CUSTOM_OPS_NAMESPACE` 宏 |
| `so_name` | 编译产物文件名（不含 `.so`） |
| `required_ops` | 必需算子名列表，缺失则 `load()` 抛异常 |
| `optional_ops` | 可选算子名列表，缺失仅打印警告 |
| `build_dir_fallback` | 编译缓存回退目录 |
| `devtoolset_path` | GCC >= 9 所在目录（默认 devtoolset-10） |
| `get_sources()` | **必须覆盖**，返回 `.cu/.cpp` 源文件路径列表 |
| `get_include_dirs()` | **必须覆盖**，返回头文件目录列表 |
| `register_fake_impls()` | 可选覆盖，注册所有算子的 fake 实现（AOTI / `torch.compile` 必需），`load()` 成功后自动调用 |
| `register_fake_for(op_name, fake_fn)` | 辅助方法，在 `register_fake_impls()` 内调用，为单个算子注册 fake 实现 |
| `load()` | 触发编译/加载，幂等，失败不抛出 |
| `is_available()` | 是否加载成功 |
| `load_error()` | 加载失败的错误信息（成功时为 `None`） |
| `ops.<name>(...)` | 通过 `__getattr__` 路由到 `torch.ops.<namespace>.<name>` |

---

## C++ 宏参考

| 宏 | 作用 |
|----|------|
| `CUSTOM_OPS_NAMESPACE` | 定义命名空间（在 `#include "custom_ops_macros.h"` 前设置） |
| `CUSTOM_OP_DEVICE_DISPATCH(name, first_tensor, ...)` | 在函数体内路由 CUDA/CPU |
| `CUSTOM_OP_DISPATCH_FN(name, params, args, device_tensor)` | 一行声明完整分发函数 |
| `CUSTOM_OP_SCHEMA(name, args_str, ret_str)` | 在 `LIBRARY_BEGIN/END` 内声明 schema |
| `CUSTOM_OP_BIND(name)` | 在 `IMPL_BEGIN/END` 内绑定实现 |
| `CUSTOM_OPS_LIBRARY_BEGIN / END` | 展开为 `TORCH_LIBRARY(ns, m) { ... }` |
| `CUSTOM_OPS_IMPL_BEGIN / END` | 展开为 `TORCH_LIBRARY_IMPL(ns, ...) { ... }` |
| `CUSTOM_OPS_PYBIND_MODULE(doc)` | pybind 模块入口（必须有） |

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `TORCH_EXTENSIONS_DIR` | 编译缓存目录（优先级最高） |
| `CUSTOM_OPS_GCC_BIN` | GCC >= 9 所在目录（覆盖 `devtoolset_path`） |
| `CC` / `CXX` | 直接指定编译器（版本 >= 9 时跳过自动配置） |
| `TORCH_CUDA_ARCH_LIST` | 手动指定 GPU 架构（设置后跳过自动探测） |

---

## 加载机制

1. **快速路径**：`build_dir/<so_name>.so` 已存在 → 直接 `dlopen`（< 1s）  
2. **编译路径**：`.so` 不存在 → JIT 编译（首次约 30~60s，带文件锁保证多进程安全）
3. **fake 注册**：加载成功后自动调用 `register_fake_impls()`，完成 AOTI / `torch.compile` 支持

---

## AOTI / torch.compile 兼容

### 工作原理

`torch.compile` 和 `torch.export`（AOTI）在 tracing 阶段使用 **FakeTensor**（meta 设备）模拟算子执行，只关心输出的 shape/dtype，不实际运行 CUDA 代码。自定义算子若没有 fake 实现，tracing 会直接失败。

```
eager 模式        → torch.ops.<ns>.<op>(...)  直接执行 CUDA kernel
torch.compile     → FakeTensor tracing → 需要 fake 实现推断 shape → 生成优化图
torch.export/AOTI → FakeTensor tracing → 需要 fake 实现推断 shape → 导出为静态模型
```

### fake 实现的规则

- 参数签名与真实算子完全一致（包括参数名和顺序）
- 返回值是与输出 shape/dtype **匹配的空 tensor**（用 `torch.empty_like` 或 `torch.empty`）
- 不允许访问 tensor 的实际数值，只能访问 `.shape`、`.dtype`、`.device`

### 示例

```python
class MyOps(CustomOps):
    def register_fake_impls(self):
        # 单输出，shape 与输入相同
        self.register_fake_for("my_op", lambda x, n: torch.empty_like(x))

        # 输出 shape 依赖输入 shape 计算
        self.register_fake_for(
            "my_pool",
            lambda x, kernel: torch.empty(
                x.shape[0], x.shape[1], x.shape[2] // kernel,
                dtype=x.dtype, device=x.device,
            ),
        )

        # 多输出（返回 tuple）
        self.register_fake_for(
            "my_topk",
            lambda x, k: (
                torch.empty(x.shape[0], k, dtype=x.dtype, device=x.device),
                torch.empty(x.shape[0], k, dtype=torch.int64, device=x.device),
            ),
        )
```

### RecsysOps 内置 fake 实现

`RecsysOps` 已在 `register_fake_impls()` 中完整实现三个算子的 fake，开箱即支持 `torch.compile` / AOTI：

```python
# mha_fwd_with_mask: 输出 shape 与 q 相同
self.register_fake_for("mha_fwd_with_mask", lambda q, k, v, mask: torch.empty_like(q))

# pack_and_prepare_b1: 6 个输出，shape 由参数推算
def _pack_fake(h_s, h_c, h_i, s_len, c_len, i_len, static_cos, static_sin, S_max, S_mask):
    D, hd = h_s.shape[-1], static_cos.shape[-1]
    return [
        torch.empty(1, S_max, D, ...),               # h_dense
        torch.empty(1, S_max, 1, hd, ...),           # cos
        torch.empty(1, S_max, 1, hd, ...),           # sin
        torch.empty(1, 1, S_mask, S_mask, ...),      # attn_mask
        torch.empty(s_len + c_len + i_len, ...),     # item_mask (bool)
        torch.empty(2, dtype=torch.int64, ...),      # offsets
    ]

# jagged_pool_and_collect: 4 个输出
def _jagged_fake(pool_values, pool_lengths, ..., n_pooling, ...):
    out_rows = pool_lengths.shape[0]
    return [pool_values_out, pool_lengths_out, out_val_splits, out_len_splits]
```

---

## 运行测试

```bash
TORCH_EXTENSIONS_DIR=/workdir/tetuan_cache/torch_extensions \
    PYTHONPATH=/workdir/tetuan/3rd:$PYTHONPATH \
    python custom_ops/test_custom_ops.py
```

测试脚本共 12 个 section，覆盖基类接口、FaOps 示例、三个算子正确性、与 `Kernel.ops` 一致性、`RecsysOps` 加载和功能验证。
