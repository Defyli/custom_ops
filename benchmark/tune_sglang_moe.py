"""为 sglang Triton MoE 移植版生成 tuned config JSON。

移植 sglang 官方 tuning 方案（benchmark/kernels/fused_moe_triton/
tuning_fused_moe_triton*.py）的轻量驱动——不依赖 ray / server_args：

  1. 候选网格按 sm120 剪枝（官方全网格 1920 个；含 smem 上限预剪枝）
  2. 每个 M 网点：粗筛全部候选（少量迭代）→ top-K 精筛（CUDA event）
  3. 双轮：Round-1 扫 up config（down 走查表/default），Round-2 固定最优
     up、扫 down config——通过 fused_experts 的 up_config/down_config
     显式注入参数实现（无需 monkey-patch；down 的 BLOCK_SIZE_M 对齐
     约束由 runner 内部强制）
  4. 逐 M 子进程隔离：特定 (up,down) tile 组合可能触发 kernel 越界
     （sticky CUDA error），子进程粒度隔离使其只影响单个 M 网点
  5. 按 sglang 文件名约定写入本包 configs 目录：
       benchmark/sglang_triton_moe/configs/triton_<ver>/
         E=<E>,N=<I>,device_name=<name>.json        (up)
         E=<E>,N=<I>,device_name=<name>_down.json   (down)
     运行时 fused_experts(use_tuned_config=True) 自动查表（M 最近邻）。

用法（5090D，对齐 benchmark_fuse_moe 的 E=8/H=2048/I=1024/K=2 形状族）：
    python benchmark/tune_sglang_moe.py                       # 默认 M 网点
    python benchmark/tune_sglang_moe.py --ms 1024 8192 32768  # 自选 M
    python benchmark/tune_sglang_moe.py --dry-run             # 只扫不写
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)

SMEM_LIMIT = 101376  # GeForce sm120 每块共享内存上限（字节）

# 本包 configs 目录（写目标）
_CONFIGS_ROOT = os.path.join(_THIS, "sglang_triton_moe", "configs")


def get_grid():
    """候选网格（对齐 sglang 官方 compute-bound 网格的剪枝版）。

    官方全网格 1920 个：block_m×block_n×block_k×group×warps×stages =
    [16,32,64,128,256]×[32,64,128,256]×[64,128,256]×[1,16,32,64]×[4,8]×[2,3,4,5]。
    本版裁剪：block_m 去除 256（sm120/sm89 下鲜有最优）、stages 去除 5
    （99KB smem 下几乎必被剪），补充官方 default 启发式会用到的
    block_k=32 / block_n=32 / group=8——实测 4090D 大 M 时 K=32 的
    default 配置优于任何 K>=64 候选，不能缺席。

    smem 预剪枝：粗估 (BLOCK_M + BLOCK_N) × BLOCK_K × 2B × stages 超限的
    组合必然 launch 时 OutOfResources，无需实编实测。
    """
    configs = []
    for block_m in (16, 32, 64, 128):
        for block_n in (32, 64, 128, 256):
            for block_k in (32, 64, 128, 256):
                for group_m in (1, 8, 16, 32, 64):
                    for warps in (4, 8):
                        for stages in (2, 3, 4):
                            if (block_m + block_n) * block_k * 2 * stages > SMEM_LIMIT:
                                continue
                            configs.append({
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_m,
                                "num_warps": warps,
                                "num_stages": stages,
                            })
    return configs


def bench(fn, iters, warmup=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # µs


def tune_single(M: int, args) -> dict:
    """单个 M 网点的双轮扫描，返回 {"up": cfg, "down": cfg}。"""
    import torch

    from sglang_triton_moe import fused_experts

    grid = get_grid()

    H_, I_, E_ = args.H, args.I, args.E
    w1 = torch.randn(E_, 2 * I_, H_, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E_, H_, I_, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(M, H_, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, E_, (M, args.K), device="cuda", dtype=torch.int32)
    wt = torch.rand(M, args.K, device="cuda", dtype=torch.float32)
    fwd = lambda **kw: fused_experts(  # noqa: E731
        x, w1, w2, wt, ids, up_config=kw.get("up"), down_config=kw.get("down"))

    def coarse(up_cfg, label):
        """粗筛：返回 [(us, cfg)]；sticky error 时快速失败。"""
        results, first_err = [], None
        for cfg in grid:
            try:
                us = bench(lambda: fwd(up=up_cfg, down=cfg), args.coarse_iters)
            except Exception as e:  # noqa: BLE001
                if first_err is None:
                    first_err = repr(e)[:300]
                    # 检测 sticky CUDA error：同步一次确认 context 是否已坏
                    try:
                        torch.cuda.synchronize()
                        torch.zeros(1, device="cuda")
                    except Exception:  # noqa: BLE001
                        raise RuntimeError(
                            f"{label} sticky CUDA error at {cfg}; "
                            f"first: {first_err}") from None
                continue
            results.append((us, cfg))
        if not results:
            raise RuntimeError(f"{label}: all {len(grid)} configs failed; "
                               f"first: {first_err}")
        results.sort(key=lambda t: t[0])
        return results

    def fine(pairs):
        """精筛 (up, down) 对列表，返回 [(us, pair)]——返回完整对，
        由调用侧取本轮扫的维度（Round1 取 pair[0]，Round2 取 pair[1]）。"""
        out = []
        for pair in pairs[:args.topk_candidates]:
            try:
                out.append((bench(lambda: fwd(up=pair[0], down=pair[1]),
                                  args.fine_iters), pair))
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda t: t[0])
        return out

    # ── Round 1：扫 up（down 走查表/default）───────────────────
    t0 = time.time()
    results = coarse(None, f"M={M} up")
    # fine 必须重放 (candidate, None)——up 候选各不同
    fine_r = fine([(cfg, None) for _, cfg in results])
    if not fine_r:
        raise RuntimeError(f"M={M} up fine-scan all failed")
    best_up = fine_r[0][1][0]
    print(f"[tune] M={M:6d} up: {best_up} -> {fine_r[0][0]:.1f}us "
          f"({time.time() - t0:.0f}s, {len(results)}/{len(grid)} valid)",
          flush=True)

    # ── Round 2：固定最优 up，扫 down ──────────────────────────
    t0 = time.time()
    results = coarse(best_up, f"M={M} down")
    fine_r = fine([(best_up, cfg) for _, cfg in results])
    if not fine_r:
        raise RuntimeError(f"M={M} down fine-scan all failed")
    best_down = fine_r[0][1][1]
    print(f"[tune] M={M:6d} down: {best_down} -> {fine_r[0][0]:.1f}us "
          f"({time.time() - t0:.0f}s)", flush=True)

    return {"up": best_up, "down": best_down}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--I", type=int, default=1024)
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--ms", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096, 8192, 16384],
                    help="M（= num_tokens）调优网点；运行时按最近邻查表")
    ap.add_argument("--coarse-iters", type=int, default=8)
    ap.add_argument("--fine-iters", type=int, default=40)
    ap.add_argument("--topk-candidates", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="只扫不写 JSON")
    ap.add_argument("--single", type=int, default=None, help="内部：单 M 子进程模式")
    ap.add_argument("--out", type=str, default=None, help="内部：单 M 结果 JSON 路径")
    args = ap.parse_args()

    # ── 子进程模式：单 M 扫描，结果落盘 ──────────────────────────
    if args.single is not None:
        res = tune_single(args.single, args)
        with open(args.out, "w") as f:
            json.dump({"M": args.single, **res}, f)
        return

    # ── 主进程：逐 M spawn 子进程（sticky CUDA error 隔离在单 M 内）──
    import tempfile

    t_all = time.time()
    best_up, best_down = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for M in args.ms:
            out = os.path.join(td, f"tune_M{M}.json")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--single", str(M), "--out", out,
                   "--H", str(args.H), "--I", str(args.I),
                   "--E", str(args.E), "--K", str(args.K),
                   "--coarse-iters", str(args.coarse_iters),
                   "--fine-iters", str(args.fine_iters),
                   "--topk-candidates", str(args.topk_candidates)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
            tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-2:])
            print(f"[tune] M={M:6d} worker: {tail}", flush=True)
            if r.returncode != 0 or not os.path.exists(out):
                print(f"[tune] M={M:6d} FAILED (sticky error?), "
                      f"skipped（运行时将走 default 启发式）", flush=True)
                continue
            with open(out) as f:
                d = json.load(f)
            best_up[M] = d["up"]
            best_down[M] = d["down"]

    print(f"[tune] total {time.time() - t_all:.0f}s, "
          f"{len(best_up)}/{len(args.ms)} M points tuned")

    if args.dry_run or not best_up:
        print("[tune] dry-run or no results: JSON not written")
        return

    # ── 写 JSON（sglang 文件名约定：N = w2.shape[2] = I；键 = M 字符串）───
    import torch
    import triton

    from sglang_triton_moe import config as _cfg

    dev_name = torch.cuda.get_device_name(0).replace(" ", "_")
    ver = triton.__version__.split(".")
    triton_dir = os.path.join(
        _CONFIGS_ROOT,
        f"triton_{ver[0]}_{ver[1]}_0")  # sglang 目录名约定（triton_3_6_0）
    os.makedirs(triton_dir, exist_ok=True)
    # 失效进程内 lru_cache，保证后续查表读到新 JSON
    _cfg.get_moe_configs.cache_clear()

    def dump(best, suffix):
        path = os.path.join(
            triton_dir, f"E={args.E},N={args.I},device_name={dev_name}{suffix}.json")
        payload = {str(m): c for m, c in sorted(best.items())}
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"[tune] wrote {path} ({len(payload)} M points)")

    dump(best_up, "")
    dump(best_down, "_down")


if __name__ == "__main__":
    main()
