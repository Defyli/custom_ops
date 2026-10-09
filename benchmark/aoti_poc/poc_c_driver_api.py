"""
PoC C：cubin + 纯 CUDA driver API —— 零 tile-lang / 零 tvm 依赖的部署路径。

背景（见 poc_b/_probe_dissect 结论）：
  tilelang export_library 的 .so 不含设备代码（无 fatbin），设备码以 TVM
  cuda module 形式经 libtvm_runtime 加载/启动（TVMBackendGetFuncFromEnv +
  ffi.Module.load_from_bytes.cuda loader）→ .so 路径确实绑死 libtvm_runtime。

本 PoC 验证替代路径：
  编译期（需 tilelang + CUDA toolkit）：
    k.export_ptx() 导出 PTX → ptxas 离线编为 cubin（注：tilelang 的
    arch-specific PTX 不能被驱动 JIT，直接 cuModuleLoadData 会报
    CUDA_ERROR_INVALID_PTX(222)；离线 ptxas 产出精确 SASS，零 JIT）
    → 从 host source 机械提取 launch config（grid/block/dyn_smem）
  部署期（仅需 cubin + launch.json + libcuda，torch 已加载）：
    ctypes 直接 cuModuleLoadData / cuLaunchKernel

运行：python poc_c_driver_api.py
"""
import json
import os
import re
import subprocess
import sys

import torch

sys.path.insert(0, "/home/lifengyi06")

OUT = "/tmp/aoti_poc"
os.makedirs(OUT, exist_ok=True)
PTX = f"{OUT}/kernel.ptx"
CUBIN = f"{OUT}/kernel.cubin"
CFGJ = f"{OUT}/launch.json"

M, N, K = 256, 512, 1024

# ══════════════════════════════════════════════════════════════════════════
# 阶段 1（编译期，需 tilelang）：导出 PTX + 提取 launch config
# ══════════════════════════════════════════════════════════════════════════
from custom_ops.tilelang_ops._swiglu import _default_config, _swiglu_factory

cfg = _default_config(M, N, K)
k = _swiglu_factory(M, N, K, "bfloat16",
                    block_M=cfg["block_M"], block_N=cfg["block_N"],
                    block_K=cfg["block_K"], num_stages=cfg["num_stages"],
                    threads=cfg["threads"], fullrow=cfg.get("fullrow", False))
k.export_ptx(PTX)

# PTX → cubin（离线 ptxas；驱动 JIT 不接受 tilelang 的 arch-specific PTX）
cap = torch.cuda.get_device_capability(0)
arch = f"sm_{cap[0]}{cap[1]}"
r = subprocess.run(["ptxas", f"-arch={arch}", PTX, "-o", CUBIN],
                   capture_output=True, text=True)
assert r.returncode == 0, f"ptxas failed: {r.stderr[:300]}"

host = k.get_host_source()

# launch config 从 host source 机械提取：main_kernel 的 packed 调用参数为
# (W, X, Y, gridX, gridY, blockX, blockY, blockZ, dyn_smem)，即 stack 槽位
# 3..8 的 6 个 int64 段（槽位 0..2 是指针）。后写覆盖前写 → 取最后一组
idx = host.rfind("main_kernel_packed")
seg = host[max(0, idx - 8000):idx]
vals = {}
for i_s, v_s in re.findall(
        r"stack_ffi_any\)\[(\d+)\]\.v_int64\)\s*=\s*\(\(int64_t\)(\d+)\);",
        seg):
    vals[int(i_s)] = int(v_s)
assert all(i in vals for i in range(3, 9)), f"launch config 提取失败: {vals}"
grid_x, grid_y, block_x, block_y, block_z, smem = (vals[3], vals[4], vals[5],
                                                   vals[6], vals[7], vals[8])
# 自检：与 factory 参数互验（T.Kernel(cdiv(M,BM), cdiv(N,BN)), threads）
assert (grid_x, grid_y) == ((M + cfg["block_M"] - 1) // cfg["block_M"],
                            (N + cfg["block_N"] - 1) // cfg["block_N"])
assert block_x == cfg["threads"] and smem > 0
launch = dict(entry="main_kernel", grid=[grid_x, grid_y, 1],
              block=[block_x, block_y, block_z], dyn_smem=smem)
json.dump(launch, open(CFGJ, "w"), indent=1)
print(f"[1] {arch}: cubin {os.path.getsize(CUBIN)}B + launch.json: {launch}")

# 参考输出（编译进程内算好，供子进程对拍）
torch.manual_seed(42)
x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(torch.bfloat16)
g = x.float() @ w[:N].float().t()
u = x.float() @ w[N:].float().t()
ref = (g / (1 + torch.exp(-g)) * u)
torch.save({"x": x.cpu(), "w": w.cpu(), "ref": ref.cpu()},
           f"{OUT}/inputs.pt")

# ══════════════════════════════════════════════════════════════════════════
# 阶段 2（部署期，零 tile-lang / 零 tvm 依赖）：纯 libcuda 启动
# ══════════════════════════════════════════════════════════════════════════
child = r'''
import ctypes
import json
import os
import sys
import time

import torch

for bad in ("tilelang", "tvm_ffi", "tvm"):
    assert bad not in sys.modules, f"{bad} must NOT be imported"

OUT = "/tmp/aoti_poc"
launch = json.load(open(f"{OUT}/launch.json"))
d = torch.load(f"{OUT}/inputs.pt")
x, w, ref = d["x"].cuda(), d["w"].cuda(), d["ref"].cuda()
M, N = x.shape[0], ref.shape[1]
y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

cuda = ctypes.CDLL("libcuda.so.1")
check = lambda r, ctx: (_ for _ in ()).throw(RuntimeError(
    f"{ctx} -> CUresult {r}")) if r != 0 else None

# cuModuleLoadData：直接加载 cubin（ELF SASS 镜像，无 JIT；torch 已建好上下文）
image = open(f"{OUT}/kernel.cubin", "rb").read()
check(cuda.cuInit(0), "cuInit")
mod = ctypes.c_void_p()
check(cuda.cuModuleLoadDataEx(ctypes.byref(mod), image, 0, None, None),
      "cuModuleLoadDataEx")

func = ctypes.c_void_p()
check(cuda.cuModuleGetFunction(ctypes.byref(func), mod,
                               launch["entry"].encode()), "cuModuleGetFunction")

smem = launch["dyn_smem"]
if smem > 48 * 1024:   # 动态 smem 超 48KB 需 opt-in（属性 8）
    check(cuda.cuFuncSetAttribute(func, 8, ctypes.c_int(smem)),
          "cuFuncSetAttribute(MAX_DYNAMIC_SHARED)")

gx, gy, gz = launch["grid"]
bx, by, bz = launch["block"]
stream = torch.cuda.current_stream().cuda_stream

def run():
    p0 = ctypes.c_uint64(w.data_ptr())
    p1 = ctypes.c_uint64(x.data_ptr())
    p2 = ctypes.c_uint64(y.data_ptr())
    params = (ctypes.c_void_p * 3)(
        ctypes.addressof(p0), ctypes.addressof(p1), ctypes.addressof(p2))
    check(cuda.cuLaunchKernel(
        func, gx, gy, gz, bx, by, bz, ctypes.c_uint(smem),
        ctypes.c_void_p(stream), params, None), "cuLaunchKernel")

t0 = time.time()
run()
torch.cuda.synchronize()
err = (y.float() - ref).abs().max().item()
print(f"load+first-launch: {time.time()-t0:.2f}s")
print(f"pure-driver-API run: max_err={err:.4e}",
      "PASS" if err < 0.05 else "FAIL")

def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

print(f"perf: {bench(run):.1f}µs  (依赖: torch + libcuda, 无 tilelang/tvm)")
'''
r = subprocess.run([sys.executable, "-c", child],
                   capture_output=True, text=True)
print("[2] child (pure driver API):")
for line in r.stdout.strip().splitlines():
    print("   ", line)
if r.returncode != 0:
    print("    STDERR:", r.stderr.strip().splitlines()[-4:])
    sys.exit(1)
print("POC_C_DONE")
