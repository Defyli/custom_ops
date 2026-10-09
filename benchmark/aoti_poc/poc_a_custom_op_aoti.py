"""
PoC：Tile-lang kernel 接入 torch AOTI（AOTInductor）编译

路径 A：Python custom op 契约
  1. swiglu_tilelang 包装为 torch.library.custom_op（自动 schema + fake meta）
  2. torch.export.export 捕获调用图
  3. torch._inductor.aot_compile 产出 AOTI .so（tilelang op 作为 extern kernel 调用）
  4. torch._C._aoti.AOTIModelContainerRunnerCuda 加载运行，对拍验证

运行：python /tmp/_aoti_poc_a.py
"""
import os
import sys

import torch

sys.path.insert(0, "/home/lifengyi06")

OUT_DIR = "/tmp/aoti_poc"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)

# ── 1. 定义 custom op（tilelang 后端）─────────────────────────────────────

try:
    torch.library.delete_library("tilelang_aoti")
except Exception:
    pass

lib = torch.library.Library("tilelang_aoti", "FRAGMENT")  # noqa: F841


def _swiglu_ref(x, w):
    N = w.shape[0] // 2
    g = x.float() @ w[:N].float().t()
    u = x.float() @ w[N:].float().t()
    return g / (1 + torch.exp(-g)) * u


@torch.library.custom_op("tilelang_aoti::swiglu", mutates_args=())
def swiglu_tl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """SwiGLU：y = silu(x @ Wg^T) * (x @ Wu^T)（Tile-lang 后端）。"""
    from custom_ops.tilelang_ops._swiglu import swiglu_tilelang
    return swiglu_tilelang(x, weight)


@swiglu_tl.register_fake
def _(x, weight):
    return torch.empty(x.shape[:-1] + (weight.shape[0] // 2,),
                       dtype=x.dtype, device=x.device)


# ── 2. eager 正确性 ────────────────────────────────────────────────────────

M, N, K = 256, 512, 1024
x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(torch.bfloat16)

y_eager = swiglu_tl(x, w)
ref = _swiglu_ref(x, w)
err = (y_eager.float() - ref).abs().max().item()
print(f"[1] eager custom op: max_err={err:.4e}", "OK" if err < 0.05 else "FAIL")
assert err < 0.05

# ── 3. torch.export 捕获 ──────────────────────────────────────────────────


class SwigluModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # 图里带一个 PyTorch 算子（mm），验证 tilelang op 与 inductor 生成码共存
        attn_bias = torch.zeros(x.shape[0], w.shape[0] // 2,
                                dtype=x.dtype, device=x.device)
        y = swiglu_tl(x, w)
        return y + attn_bias


model = SwigluModel().eval()
example = (x, w)
with torch.no_grad():
    exported = torch.export.export(model, example)
print("[2] torch.export: OK")
print(exported.graph_module.code.strip().split("\n")[:8]
      if hasattr(exported.graph_module, "code") else "")

# ── 4. AOTI 编译 ───────────────────────────────────────────────────────────
from torch._inductor import aot_compile  # noqa: E402

with torch.no_grad():
    so_path = aot_compile(exported.module(), example)
print(f"[3] aot_compile: {so_path}")
assert isinstance(so_path, str) and so_path.endswith(".so")

# ── 5. 加载运行（同进程：Python op 已注册）────────────────────────────────
runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1)
with torch.no_grad():
    y_aoti = runner.run(list(example))
torch.cuda.synchronize()
y_aoti = y_aoti[0] if isinstance(y_aoti, (list, tuple)) else y_aoti
err = (y_aoti.float() - ref).abs().max().item()
print(f"[4] AOTI run: dtype={y_aoti.dtype} shape={tuple(y_aoti.shape)} "
      f"max_err={err:.4e}", "OK" if err < 0.05 else "FAIL")
assert err < 0.05

# ── 6. 简单性能对照（eager op vs AOTI 端到端）─────────────────────────────


def bench(fn, iters=100):
    for _ in range(10):
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


t_eager = bench(lambda: swiglu_tl(x, w))
t_aoti = bench(lambda: runner.run(list(example)))
print(f"[5] perf: eager-op={t_eager:.1f}µs  AOTI-end-to-end={t_aoti:.1f}µs")

# ── 7. 产物清单 ───────────────────────────────────────────────────────────
for f in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, f)
    print(f"    {f:40s} {os.path.getsize(p):>9d} B")
print("POC_A_DONE")
