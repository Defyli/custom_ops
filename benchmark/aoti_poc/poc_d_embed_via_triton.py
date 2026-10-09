"""
PoC D：tilelang cubin 借道 user-defined Triton kernel + embed_kernel_binary 嵌入 AOTI .so

思路（回答「能否用 aot_inductor.embed_kernel_binary 嵌入底层代码」）：
  embed_kernel_binary 不是任意 cubin 注入 API——它只决定 inductor 自编译 kernel
  （含 user-defined Triton kernel）的 cubin 以「嵌入 .so」还是「外置文件」打包。
  但 cubin 的产生点是 triton.compile → 若在 AOTI 编译期劫持该调用，把 shim
  kernel 的编译产物替换为 tilelang 的 cubin（保留 metadata.num_warps/shared、
  改 name 为 cubin 入口 main_kernel），后续「写 cubin → cpp codegen → 嵌入」
  全部走官方管线，产物为单一自包含 model.so。

  1. 定义真实 triton shim（签名 (W, X, Y) 3 指针 + constexpr），eager/tracing 语义正确
  2. patch triton.compile：识别 shim → 正常编译取合法 CompiledKernel →
     覆写 asm["cubin"]=tilelang cubin、metadata.shared/num_warps、name="main_kernel"
  3. torch.export 捕获 shim 调用 → aot_compile(embed_kernel_binary=True)
  4. 验证：(a) model.so 内嵌 tilelang cubin 字节；(b) 裸进程（零 op 注册、
     零 Python 依赖面）加载运行 PASS

运行：python poc_d_embed_via_triton.py
"""
import os
import subprocess
import sys

import torch

sys.path.insert(0, "/home/lifengyi06")

OUT = "/tmp/aoti_poc_d"
os.makedirs(OUT, exist_ok=True)

M, N, K = 256, 512, 1024
BM, BN, BK = 64, 64, 64          # grid=(4,8)；tilelang kernel 同款 tile
NUM_WARPS = 8                    # tilelang threads=256

torch.manual_seed(42)
x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(torch.bfloat16)
g = x.float() @ w[:N].float().t()
u = x.float() @ w[N:].float().t()
ref = (g / (1 + torch.exp(-g)) * u)

# ══════════════════════════════════════════════════════════════════════
# 1. tilelang：编 cubin + launch config（复用 PoC C 逻辑）
# ══════════════════════════════════════════════════════════════════════
import re  # noqa: E402

from custom_ops.tilelang_ops._swiglu import _default_config, _swiglu_factory  # noqa: E402

cfg = _default_config(M, N, K)
tl_kernel = _swiglu_factory(
    M, N, K, "bfloat16",
    block_M=cfg["block_M"], block_N=cfg["block_N"], block_K=cfg["block_K"],
    num_stages=cfg["num_stages"], threads=cfg["threads"],
    fullrow=cfg.get("fullrow", False))
PTX = f"{OUT}/kernel.ptx"
CUBIN = f"{OUT}/kernel.cubin"
tl_kernel.export_ptx(PTX)
cap = torch.cuda.get_device_capability(0)
r = subprocess.run(["ptxas", f"-arch=sm_{cap[0]}{cap[1]}", PTX, "-o", CUBIN],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[:300]

host = tl_kernel.get_host_source()
idx = host.rfind("main_kernel_packed")
vals = {int(i): int(v) for i, v in re.findall(
    r"stack_ffi_any\)\[(\d+)\]\.v_int64\)\s*=\s*\(\(int64_t\)(\d+)\);",
    host[max(0, idx - 8000):idx])}
grid_x, grid_y, smem = vals[3], vals[4], vals[8]
assert (grid_x, grid_y) == (4, 8) and smem == 73728, (grid_x, grid_y, smem)
CUBIN_BYTES = open(CUBIN, "rb").read()
print(f"[1] tilelang cubin: {len(CUBIN_BYTES)}B grid=({grid_x},{grid_y}) "
      f"smem={smem} num_warps={NUM_WARPS}")

# ══════════════════════════════════════════════════════════════════════
# 2. 真实 triton shim（eager 正确；签名与 tilelang main_kernel(W,X,Y) 对齐）
# ══════════════════════════════════════════════════════════════════════
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def swiglu_shim(W, X, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    x_ptrs = X + rm[:, None] * K + rk[None, :]
    wg_ptrs = W + rn[:, None] * K + rk[None, :]
    wu_ptrs = W + (N + rn)[:, None] * K + rk[None, :]
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        xt = tl.load(x_ptrs)
        wg = tl.load(wg_ptrs)
        wu = tl.load(wu_ptrs)
        acc_g += tl.dot(xt, tl.trans(wg))
        acc_u += tl.dot(xt, tl.trans(wu))
        x_ptrs += BK
        wg_ptrs += BK
        wu_ptrs += BK
    y = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    y_ptrs = Y + rm[:, None] * N + rn[None, :]
    tl.store(y_ptrs, y.to(Y.dtype.element_ty))


def run_shim(x, w):
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    swiglu_shim[(grid_x, grid_y)](w, x, y, M, N, K, BM, BN, BK,
                                  num_warps=NUM_WARPS)
    return y


try:
    y_shim = run_shim(x, w)
    err = (y_shim.float() - ref).abs().max().item()
    print(f"[2] triton shim eager: max_err={err:.4e}",
          "OK" if err < 0.05 else "FAIL")
except Exception as e:   # eager 校验非致命：dynamo/export 不执行 kernel，
    # 最终权威是步骤[6]裸进程对 tilelang cubin 的运行结果
    print(f"[2] triton shim eager 跳过（{type(e).__name__}: "
          f"{str(e).splitlines()[0][:60]}）")

# ══════════════════════════════════════════════════════════════════════
# 3. 劫持 triton.compile：AOTI 编译期把 shim 的 cubin 换成 tilelang 的
# ══════════════════════════════════════════════════════════════════════
import dataclasses  # noqa: E402

_orig_triton_compile = triton.compile
_hijack_count = {"n": 0}
_renamed_cubin = bytearray()   # 劫持时捕获，供步骤[5]嵌入校验


def _patched_triton_compile(*args, **kwargs):
    kernel = _orig_triton_compile(*args, **kwargs)
    try:
        src = args[0]
        name = getattr(src, "name", None) or getattr(
            getattr(src, "fn", None), "__name__", "")
    except Exception:
        name = ""
    if name == "swiglu_shim":
        # 策略：不改符号名，而是把 tilelang PTX 的入口重命名为 triton 编译出的
        # 真实符号（kernel.metadata.name）——名字与 cubin 天然一致，且不被
        # pickled/缓存回读路径破坏（只换机器码 + shared 大小）
        target_sym = kernel.metadata.name
        ptx_text = open(PTX).read()
        ptx_text2, n = re.subn(r"\.entry\s+main_kernel\(",
                               f".entry {target_sym}(", ptx_text)
        assert n == 1, f"PTX 入口重命名失败: {n} 处"
        r2 = subprocess.run(
            ["ptxas", f"-arch=sm_{cap[0]}{cap[1]}", "-o", CUBIN, "-"],
            input=ptx_text2, capture_output=True, text=True)
        assert r2.returncode == 0, f"ptxas: {r2.stderr[:300]}"
        cubin_renamed = open(CUBIN, "rb").read()
        _renamed_cubin.clear()
        _renamed_cubin.extend(cubin_renamed)
        kernel.asm["cubin"] = cubin_renamed
        m = kernel.metadata
        if hasattr(m, "_replace"):            # NamedTuple：仅改启动资源
            object.__setattr__(kernel, "metadata",
                               m._replace(shared=smem, num_warps=NUM_WARPS))
        _hijack_count["n"] += 1
        print(f"[3] triton.compile hijacked (#{_hijack_count['n']}): "
              f"tilelang cubin {len(cubin_renamed)}B，入口重命名→{target_sym}")
    return kernel


triton.compile = _patched_triton_compile

# ══════════════════════════════════════════════════════════════════════
# 4. export → aot_compile（embed_kernel_binary=True）
# ══════════════════════════════════════════════════════════════════════
from torch._inductor import config as icfg  # noqa: E402

icfg.aot_inductor.embed_kernel_binary = True


class SwigluModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        y = swiglu_embed(w, x)
        return y + 1.0 - 1.0   # 图里保留一点 aten 算子


# torch 2.11 的 torch.export 不直接拦截裸 triton kernel（本环境实测，
# FakeTensor data_ptr 报错）；官方桥接是 @triton_op + wrap_triton：
# export 时是普通 custom op，AOTI 编译时 inductor 通过 triton_ops_to_kernels
# 提取内部 triton kernel 并编译——正是我们劫持 triton.compile 的时机
from torch._library.triton import triton_op, wrap_triton  # noqa: E402


@triton_op("tilelang_aoti::swiglu_embed", mutates_args=())
def swiglu_embed(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    # 官方形态：wrap_triton(kernel)[grid](args) —— 包装 kernel 对象后下标，
    # fake trace 时捕获为 HOP 而不执行（写成 wrap_triton(kernel[grid](args))
    # 会先真执行再包装，FakeTensor 下即 data_ptr 报错）
    wrap_triton(swiglu_shim)[(grid_x, grid_y)](
        w, x, y, M, N, K, BM, BN, BK, num_warps=NUM_WARPS)
    return y
# 注意：不要显式 register_fake —— triton_op 自动把 fn 注册为 fake，
# wrap_triton 在 fake trace 时把 kernel 捕获为 HOP（AOTI 编译提取的正是它）；
# 自定义 fake 会覆盖掉捕获，导致分解失败


model = SwigluModel().eval()
# strict=False：user-defined triton kernel 走 dynamo 全特性路径
# （strict 模式下 _export_for_training 会落入逐行 trace → FakeTensor data_ptr 报错）
exported = torch.export.export(model, (x, w), strict=False)

from torch._inductor import aot_compile  # noqa: E402

with torch.no_grad():
    so_path = aot_compile(exported.module(), (x, w))
triton.compile = _orig_triton_compile          # 编译完恢复
print(f"[4] aot_compile: {so_path}")
print(f"    hijack triggered: {_hijack_count['n']} 次")

so_bytes = open(so_path, "rb").read()
# 用重命名 cubin 的尾部（.text 段与符号名无关）做嵌入校验
embedded = bytes(_renamed_cubin[-4096:]) in so_bytes
print(f"[5] tilelang cubin 嵌入 model.so: {'YES' if embedded else 'NO'} "
      f"(so={len(so_bytes)/1e6:.1f}MB, renamed_cubin={len(_renamed_cubin)}B)")
assert embedded, "cubin 未嵌入！"
assert _hijack_count["n"] >= 1, "triton.compile 劫持未触发"

# ══════════════════════════════════════════════════════════════════════
# 5. 裸进程验证：无 op 注册、无 Python 依赖面，加载 model.so 运行
# ══════════════════════════════════════════════════════════════════════
torch.save({"x": x.cpu(), "w": w.cpu(), "ref": ref.cpu()}, f"{OUT}/inputs.pt")
child = f'''
import time
import torch

t0 = time.time()
runner = torch._C._aoti.AOTIModelContainerRunnerCuda({so_path!r}, 1)
print(f"load: {{time.time()-t0:.2f}}s")

d = torch.load({f"{OUT}/inputs.pt"!r})
x, w, ref = d["x"].cuda(), d["w"].cuda(), d["ref"].cuda()
y = runner.run([x, w])
y = y[0] if isinstance(y, (list, tuple)) else y
torch.cuda.synchronize()
err = (y.float() - ref).abs().max().item()
print(f"bare-process AOTI run: dtype={{y.dtype}} shape={{tuple(y.shape)}} "
      f"max_err={{err:.4e}}", "PASS" if err < 0.05 else "FAIL")

def bench(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1e3

t = bench(lambda: runner.run([x, w]))
print(f"perf: {{t:.1f}}µs")
print("deps: torch only (no tilelang/tvm/triton op registration)")
'''
r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
print("[6] bare process:")
for line in r.stdout.strip().splitlines():
    print("   ", line)
if r.returncode != 0:
    print("    STDERR:", r.stderr.strip().splitlines()[-4:])
    sys.exit(1)
print("POC_D_DONE")
