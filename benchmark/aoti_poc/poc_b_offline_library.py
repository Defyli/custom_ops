"""
Path B 验证：tilelang export_library 产物自包含性。

1. 在本进程用 tilelang JIT 编译 swiglu kernel，export_library 导出独立 .so
2. nm 查看 ABI 符号
3. 【关键】子进程不 import tilelang，仅用 tvm_ffi 加载该 .so 并运行 → 验证产物自包含
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, "/home/lifengyi06")

OUT = "/tmp/aoti_poc/tl_swielu_export.so"
os.makedirs("/tmp/aoti_poc", exist_ok=True)

# ── 1. JIT 编译 + 导出 ─────────────────────────────────────────────────────
from custom_ops.tilelang_ops._swiglu import _swiglu_factory, _default_config

M, N, K = 256, 512, 1024
dtype = "bfloat16"
cfg = _default_config(M, N, K)
k = _swiglu_factory(M, N, K, dtype,
                    block_M=cfg["block_M"], block_N=cfg["block_N"],
                    block_K=cfg["block_K"], num_stages=cfg["num_stages"],
                    threads=cfg["threads"],
                    fullrow=cfg.get("fullrow", False))
k.export_library(OUT)
print(f"[1] export_library: {OUT} {os.path.getsize(OUT)} B")

# 同进程先用 JIT 验证参考输出
x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(torch.bfloat16)
y_jit = k(x, w)
g = x.float() @ w[:N].float().t()
u = x.float() @ w[N:].float().t()
ref = (g / (1 + torch.exp(-g)) * u)
err = (y_jit.float() - ref).abs().max().item()
print(f"[2] JIT run ref: max_err={err:.4e}")

# ── 2. ABI 符号 ────────────────────────────────────────────────────────────
syms = subprocess.run(["nm", "-D", OUT], capture_output=True, text=True).stdout
main_syms = [l for l in syms.split("\n") if "__tvm_ffi_main" in l or "TVMFFI" in l]
print("[3] .so ABI symbols:", main_syms[:4])

# ── 3. 子进程：不 import tilelang，仅 tvm_ffi 加载 .so ────────────────────
child = r'''
import ctypes
import os
import sys
import time

import torch

# 确认本进程未加载 tilelang（JIT 编译器）
assert "tilelang" not in sys.modules, "tilelang must NOT be imported here"

import tvm_ffi

# 预加载 tilelang 的纯运行时库（注册 ffi.Module.load_from_bytes.cuda loader）。
# 这是部署产物的一部分：kernel .so + libtvm_runtime.so + tvm_ffi（无编译器）
RT = "/home/lifengyi06/.local/lib/python3.12/site-packages/tilelang/lib/libtvm_runtime.so"
# tilelang 自身用 RTLD_LAZY 加载（延迟符号绑定）；NOW 模式会因宿主
# libstdc++ 缺个别符号而立即失败
cTypes_mode = os.RTLD_LAZY | os.RTLD_GLOBAL
ctypes.CDLL("libstdc++.so.6", mode=cTypes_mode)
lib = ctypes.CDLL(RT, mode=cTypes_mode)
print("libtvm_runtime.so preloaded:", bool(lib))

t0 = time.time()
mod = tvm_ffi.load_module("/tmp/aoti_poc/tl_swielu_export.so")
entry = mod.entry_name
if callable(entry):
    entry = entry()
fn = mod.get_function(entry)
print(f"load: {time.time()-t0:.2f}s entry={entry}")

M, N, K = 256, 512, 1024
torch.manual_seed(42)
x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(torch.bfloat16)

# packed ABI：out_idx=[2] → 第 3 参数是输出 prototype（callee 内部
# EnvTensorAlloc 分配真实输出并作为返回值），调用方取返回值
proto = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
out = fn(x, w, proto)
torch.cuda.synchronize()

g = x.float() @ w[:N].float().t()
u = x.float() @ w[N:].float().t()
ref = g / (1 + torch.exp(-g)) * u
err = (out.float() - ref).abs().max().item()
print(f"no-tilelang process run: max_err={err:.4e}",
      "PASS" if err < 0.05 else "FAIL")
print("mods contains tilelang:", "tilelang" in sys.modules)

def bench(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

t = bench(lambda: fn(x, w, proto))
print(f"perf: {t:.1f}\u00b5s")
'''
r = subprocess.run([sys.executable, "-c", child],
                   capture_output=True, text=True)
print("[4] child (no tilelang import):")
for line in r.stdout.strip().split("\n"):
    print("   ", line)
if r.returncode != 0:
    print("    STDERR:", r.stderr.strip().split("\n")[-4:])
