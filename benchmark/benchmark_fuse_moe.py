"""
benchmark_fuse_moe.py — fuse_moe 算子性能基准

基线为贴近真实模型的 nn.Module 实现（Mixtral/HF 风格：per-expert nn.Linear
gate_up_proj/down_proj + silu*mul + index_add 路由归约），另测 torch.compile
优化后的性能，与自研 kernel 三方对比。

用法：
    python benchmark/benchmark_fuse_moe.py
    python benchmark/benchmark_fuse_moe.py --iters 50 --warmup 10
    python benchmark/benchmark_fuse_moe.py --no-compile      # 跳过 compile 对照
    python benchmark/benchmark_fuse_moe.py --csv result.csv

输出：每组 (S, H, I, E, K) 的 custom / eager / compiled 耗时、TFLOPS 与加速比。
FLOPS 口径：S*K*6*H*I（gate_up 2·H·2I + down 2·I·H，乘加各计一次）。

注：kernel 中间结果以 bf16 落盘（与真实 bf16 模型一致的数值路径），
与 fp32 参考存在 bf16 精度级差异，校验容差已覆盖。
"""
import argparse
import gc
import os
import sys
import time

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from custom_ops import ops  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 基线模型（Mixtral/HF 风格 nn.Module）
# ══════════════════════════════════════════════════════════════════════════

class ExpertFFN(nn.Module):
    """单个 expert 的 FFN：down(silu(gate) * up)。

    gate/up 融合为一个 Linear（输出 2I，gate 在前 up 在后），与自研 kernel 的
    (E, 2I, H) 融合权重布局一致。Linear 内部走 cuBLAS bf16 GEMM（fp32 累加），
    与 kernel 的数值路径对等。"""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        gu = self.gate_up_proj(x)
        g, u = gu.chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class MoEFFN(nn.Module):
    """Mixtral/HF 风格 MoE FFN 前向（真实模型写法）。

    路由语义与自研 kernel 一致：topk 展平后位置 i 对应 token i//K；遍历
    expert，nonzero 选出路由到该 expert 的 token，计算后按路由权重
    index_add 加权归约（fp32 归约、bf16 输出）。"""

    def __init__(self, hidden: int, intermediate: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            ExpertFFN(hidden, intermediate) for _ in range(num_experts))

    def forward(self, x, topk_ids, topk_scale):
        num_seq, num_topk = topk_ids.shape
        out = torch.zeros(x.shape[0], x.shape[1],
                          device=x.device, dtype=torch.float32)
        flat_ids = topk_ids.reshape(-1)
        flat_scale = topk_scale.reshape(-1)
        for e, expert in enumerate(self.experts):
            idx = (flat_ids == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            rows = torch.div(idx, num_topk, rounding_mode="floor")
            y = expert(x[rows])                                  # (n, H) bf16
            out.index_add_(0, rows, y.float() * flat_scale[idx][:, None])
        return out.to(x.dtype)


class MoEFFNDense(nn.Module):
    """Graph-friendly MoE 前向：chunked gather + bmm（无数据依赖控制流）。

    为 fullgraph=True + reduce-overhead（CUDA graph）专门编写的等价形式：
    按固定 chunk 展开循环（静态形状可完整捕获），每 chunk gather 权重做
    bmm 后 index_add 归约。FLOPS 与真实 MoE 相同；代价是权重按 token
    复制（内存放大），chunk 大小按权重 tile 字节数自适应（目标 ~2GB 峰值）。"""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, max_chunks: int = 96):
        super().__init__()
        E, N2, H = w1.shape
        I = w2.shape[2]
        self.w1 = nn.Parameter(w1.clone())                      # (E, 2I, H)
        self.w2 = nn.Parameter(w2.clone())                      # (E, H, I)
        # chunk 自适应：单 chunk 权重 gather ≤ ~2GB（对 H=4096 等大权重自动缩小）
        mem_c = max(32, int(2e9 / (N2 * H * w1.element_size())))
        self.chunk = max(32, min(mem_c, 8192))
        self._num_experts = E
        self._intermediate = I

    def forward(self, x, topk_ids, topk_scale):
        S, H = x.shape
        K = topk_ids.shape[1]
        T = S * K
        C = self.chunk
        e = topk_ids.reshape(-1)
        sc = topk_scale.reshape(-1)
        xg = x[:, None, :].expand(S, K, H).reshape(T, H)
        rows_all = torch.div(torch.arange(T, device=x.device), K,
                             rounding_mode="floor")
        out = torch.zeros(S, H, device=x.device, dtype=torch.float32)
        for i in range(0, T, C):  # 静态展开（T/C ≤ ~96 个 chunk）
            j = min(i + C, T)
            ec = e[i:j]
            w1c = self.w1[ec]                                   # (c, 2I, H) gather
            gu = torch.bmm(xg[i:j].unsqueeze(1), w1c.transpose(1, 2)).squeeze(1)
            g, u = gu.chunk(2, dim=-1)
            h = F.silu(g) * u
            w2c = self.w2[ec]                                   # (c, H, I) gather
            y = torch.bmm(h.unsqueeze(1), w2c.transpose(1, 2)).squeeze(1)
            out.index_add_(0, rows_all[i:j], y.float() * sc[i:j][:, None])
        return out.to(x.dtype)


def make_module(w1, w2):
    """按 benchmark 权重张量构建 MoEFFN（拷入权重保证数值一致）。"""
    E, N2, H = w1.shape
    I = w2.shape[2]
    m = MoEFFN(H, I, E).to(device=w1.device, dtype=w1.dtype)
    with torch.no_grad():
        for e, ex in enumerate(m.experts):
            ex.gate_up_proj.weight.copy_(w1[e])
            ex.down_proj.weight.copy_(w2[e])
    return m


def make_dense_module(w1, w2):
    """Graph-friendly 变体（构造时拷入权重）。"""
    return MoEFFNDense(w1, w2).to(device=w1.device, dtype=w1.dtype)


def fp32_ref(x, w1, w2, topk_ids, topk_scale):
    """per-expert 分组 fp32 参考（校验金标准）"""
    S, H = x.shape
    E, _, _ = w1.shape
    I = w2.shape[2]
    K = topk_ids.shape[1]
    out = torch.zeros(S, H, device=x.device, dtype=torch.float32)
    e = topk_ids.reshape(-1)
    xg = x[:, None, :].expand(S, K, H).reshape(S * K, H)
    for ei in range(E):
        idx = (e == ei).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        xe = xg[idx].float()
        gu = xe @ w1[ei].float().T
        g, u = gu[:, :I], gu[:, I:]
        h = F.silu(g) * u
        y = h @ w2[ei].float().T
        rows = torch.div(idx, K, rounding_mode="floor")
        out.index_add_(0, rows, y * topk_scale.reshape(-1)[idx][:, None])
    return out


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # µs


def compile_available() -> bool:
    """进程级 torch.compile 可用性探测（仅探测一次；失败永久记忆）。

    环境兼容性：旧 Python（<3.10）+ 新 torch 的组合下 inductor 会因
    `dataclass(slots=True)`（3.10+ 语法）报 TypeError；旧 torch alpha 版亦有
    类似不兼容。probe 必须用真实计算图（matmul + 非逐元素归约）才能触发
    inductor 完整代码生成路径——简单 lambda 可能走 eager fallback 探不出来。
    任何一次真实 compile 失败也会置 False（失败记忆，后续 shape 静默跳过
    而不是逐个报错刷屏）。"""
    global _COMPILE_AVAILABLE
    if _COMPILE_AVAILABLE is not None:
        return _COMPILE_AVAILABLE
    _COMPILE_AVAILABLE = False
    try:
        t = torch.randn(16, 16, device="cuda")
        probe = torch.compile(lambda a: (a @ a.T).relu().sum())
        _ = probe(t)
        torch.cuda.synchronize()
        _COMPILE_AVAILABLE = True
    except Exception as ex:
        print(f"  (torch.compile unavailable on this env: {type(ex).__name__}; "
              f"compile/RO/fullgraph 对照列将显示 N/A)")
    return _COMPILE_AVAILABLE


def _compile_failed_once(ex: Exception) -> None:
    """真实 compile 失败后永久禁用（环境性失败，重试只会重复报错）。"""
    global _COMPILE_AVAILABLE
    if _COMPILE_AVAILABLE:
        _COMPILE_AVAILABLE = False
        print(f"  (torch.compile disabled after failure: {type(ex).__name__}; "
              f"remaining shapes will skip compile columns)")


_COMPILE_AVAILABLE = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--fg-max-tokens", type=int, default=512,
                        help="fullgraph+cudagraph 变体仅在小 T（decode 类）运行："
                             "dense gather 形式的权重流量 O(T·E·H·I)，大 T 无意义且 OOM")
    args = parser.parse_args()

    assert ops.is_available(), "extension load failed"
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    # (S, H, I, E, K)
    shapes = [
        (128,   2048, 1024,  8, 2),    # 小批量 decode
        (128,   2048, 1024, 64, 2),
        (1024,  2048, 1024,  8, 2),    # prefill
        (1024,  2048, 1024, 64, 2),
        (1024,  4096, 1024,  8, 2),    # 大 H
        (4096,  2048, 1024,  8, 2),    # 大批量
        (4096,  2048, 1024, 64, 8),
        (4096,  4096, 1408,  8, 2),    # 大模型 shape
        (16384, 2048, 1024,  8, 2),    # 超大批量
    ]

    print(f"{'shape (S,H,I,E,K)':<24} {'custom µs':>9} {'TFLOPS':>7} "
          f"{'eager µs':>9} {'vs eager':>8} {'comp µs':>9} {'comp+RO µs':>10} "
          f"{'FG+graph µs':>11} {'vs FG':>8}")
    rows = []
    for (S, H, I, E, K) in shapes:
        x = (torch.randn(S, H, device=dev, dtype=torch.float32) * 0.5).to(dtype)
        w1 = (torch.randn(E, 2 * I, H, device=dev, dtype=torch.float32) * 0.05).to(dtype)
        w2 = (torch.randn(E, H, I, device=dev, dtype=torch.float32) * 0.05).to(dtype)
        topk_ids = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
        topk_scale = torch.rand(S, K, device=dev, dtype=torch.float32)

        module = make_module(w1, w2)
        module.eval()

        # 自研 kernel
        t_custom = bench(lambda: ops.fuse_moe(x, w1, w2, topk_ids, topk_scale),
                         args.warmup, args.iters)

        # eager nn.Module 基线（真实模型写法）
        t_eager = bench(lambda: module(x, topk_ids, topk_scale),
                        args.warmup, args.iters)

        # torch.compile 默认模式（graph break 允许；数据依赖路由下 CUDA graph 自动禁用）
        t_comp = float("nan")
        compiled = None
        if not args.no_compile and compile_available():
            try:
                compiled = torch.compile(module)
                t_comp = bench(lambda: compiled(x, topk_ids, topk_scale),
                               args.warmup, args.iters)
            except Exception as ex:
                _compile_failed_once(ex)

        # torch.compile + mode="reduce-overhead"（请求 CUDA graph）——真实写法：
        # per-expert nonzero 数据依赖路由产生 graph break，torch 会自动禁用
        # cudagraph 捕获并退化为逐 graph 执行，验证「RO 对真实 MoE 无效」。
        t_ro = float("nan")
        if not args.no_compile and compile_available():
            try:
                compiled_ro = torch.compile(module, mode="reduce-overhead")
                t_ro = bench(lambda: compiled_ro(x, topk_ids, topk_scale),
                             args.warmup, args.iters)
                del compiled_ro
            except Exception as ex:
                _compile_failed_once(ex)

        # fullgraph=True + reduce-overhead（CUDA graph）——graph-friendly 写法的上限。
        # 真实写法（nonzero 数据依赖路由）无法 fullgraph，会直接报错，此处
        # 用等价的 chunked dense 形式承接；其权重 gather 流量 O(T·E·H·I)，
        # 仅在小 T（decode 类）下有参考意义，大 T 跳过（避免无意义测试 + OOM）。
        t_fg = float("nan")
        if not args.no_compile and S * K <= args.fg_max_tokens and compile_available():
            try:
                dense = make_dense_module(w1, w2)
                dense.eval()
                compiled_fg = torch.compile(dense, fullgraph=True,
                                             mode="reduce-overhead")
                t_fg = bench(lambda: compiled_fg(x, topk_ids, topk_scale),
                             args.warmup, args.iters)
                # cudagraph 输出会被后续 replay 覆写，正确性校验用 clone 快照
                fg_out = compiled_fg(x, topk_ids, topk_scale).clone()
                ref0 = fp32_ref(x, w1, w2, topk_ids, topk_scale)
                if not torch.allclose(fg_out.float(), ref0, atol=0.1, rtol=0.05):
                    print("  !! fullgraph dense variant mismatch")
                del dense, compiled_fg, fg_out
            except torch.cuda.OutOfMemoryError:
                print("  (fullgraph dense variant: OOM, N/A)")
            except Exception as ex:
                _compile_failed_once(ex)
            finally:
                # cudagraph 私有内存池不随 del 释放，强制回收防后续 shape OOM
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        flops = S * K * 6 * H * I
        tf = flops / (t_custom * 1e-6) / 1e12
        sp_eager = t_eager / t_custom
        sp_fg = t_fg / t_custom if t_fg == t_fg else float("nan")
        fg_str = f"{t_fg:.1f}" if t_fg == t_fg else "—"
        sp_fg_str = f"{sp_fg:.1f}x" if t_fg == t_fg else "—"
        print(f"({S},{H},{I},{E},{K}){'':<6} {t_custom:>9.1f} {tf:>7.1f} "
              f"{t_eager:>9.1f} {sp_eager:>7.2f}x {t_comp:>9.1f} {t_ro:>10.1f} "
              f"{fg_str:>11} {sp_fg_str:>8}")
        rows.append((S, H, I, E, K, t_custom, tf, t_eager, t_comp, t_ro, t_fg))

        # 正确性校验（vs fp32 参考；容差覆盖中间 bf16 落盘：kernel 的
        # gate_up/act/down 中间量以 bf16 写盘，ref 值小的元素相对误差天然偏
        # 大（实测 p99_rel~0.2），故 atol 主导 + rtol 充裕；报错时附误差分位
        # 数详情以便区分精度边界与结构性错误）
        ref = fp32_ref(x, w1, w2, topk_ids, topk_scale)
        out = ops.fuse_moe(x, w1, w2, topk_ids, topk_scale)
        diff = (out.float() - ref).abs()
        rel = diff / ref.abs().clamp_min(1e-2)
        ok = torch.allclose(out.float(), ref, atol=0.2, rtol=0.05)
        if not ok:
            p99 = rel.flatten().kthvalue(max(1, int(rel.numel() * 0.99))).values.item()
            print(f"  !! correctness FAILED: max_abs={diff.max().item():.4f} "
                  f"mean={diff.mean().item():.5f} p99_rel={p99:.4f} "
                  f"ref_absmax={ref.abs().max().item():.2f}")
        # 基线自身也抽样校验（防止再次写错基线）
        mout = module(x, topk_ids, topk_scale)
        if not torch.allclose(mout.float(), ref, atol=0.2, rtol=0.05):
            print("  !! baseline module mismatch (baseline bug?)")

        del module
        if compiled is not None:
            del compiled
        gc.collect()
        torch.cuda.empty_cache()

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("S,H,I,E,K,custom_us,custom_tflops,eager_us,compiled_us,"
                    "compiled_ro_us,fullgraph_us\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")
        print(f"\ncsv written to {args.csv}")


if __name__ == "__main__":
    main()
