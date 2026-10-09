"""
AOTI 跨进程部署验证。

用法：
    python _aoti_deploy_test.py <so_path> bare      # 不 import op 模块（预期失败）
    python _aoti_deploy_test.py <so_path> import    # import op 模块（预期成功+缓存命中）
    python _aoti_deploy_test.py <so_path> cgraph    # import + CUDA Graph 兼容性
"""
import sys
import time

import torch

so_path, mode = sys.argv[1], sys.argv[2]

if mode in ("import", "cgraph"):
    sys.path.insert(0, "/tmp")
    import _tl_aoti_ops  # noqa: F401  (注册 tilelang custom op)

M, N, K = 256, 512, 1024
torch.manual_seed(42)
x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
w = (torch.randn(2 * N, K, device="cuda") * 0.05).to(torch.bfloat16)


def ref(x, w):
    n = w.shape[0] // 2
    g = x.float() @ w[:n].float().t()
    u = x.float() @ w[n:].float().t()
    return (g / (1 + torch.exp(-g)) * u)


r = ref(x, w)

t0 = time.time()
runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1)
print(f"runner construct: {time.time() - t0:.1f}s (含 tilelang JIT 缓存加载)")

y = runner.run([x, w])
y = y[0] if isinstance(y, (list, tuple)) else y
torch.cuda.synchronize()
err = (y.float() - r).abs().max().item()
print(f"[{mode}] run OK: shape={tuple(y.shape)} dtype={y.dtype} "
      f"max_err={err:.4e} {'PASS' if err < 0.05 else 'FAIL'}")

if mode == "cgraph":
    # AOTI runner 的 CUDA Graph 支持（若可用）
    for name in ("run_with_cuda_graph", "cuda_graph_run", "capture"):
        fn = getattr(runner, name, None)
        if fn:
            print("cuda graph method:", name)
            try:
                y2 = fn([x, w])
                y2 = y2[0] if isinstance(y2, (list, tuple)) else y2
                torch.cuda.synchronize()
                err2 = (y2.float() - r).abs().max().item()
                print(f"  cudagraph run: max_err={err2:.4e}",
                      "PASS" if err2 < 0.05 else "FAIL")
            except Exception as e:
                print(f"  cudagraph run ERR: {type(e).__name__}: {str(e)[:80]}")
            break
    else:
        print("runner methods:", [a for a in dir(runner) if not a.startswith("_")])

    # torch CUDA Graph 手动 capture（tilelang adapter 的 stream 语义测试）
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            runner.run([x, w])
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y3 = runner.run([x, w])
        g.replay()
        torch.cuda.synchronize()
        y3 = y3[0] if isinstance(y3, (list, tuple)) else y3
        err3 = (y3.float() - r).abs().max().item()
        print(f"torch.cuda.graph capture+replay: max_err={err3:.4e}",
              "PASS" if err3 < 0.05 else "FAIL")
    except Exception as e:
        print(f"torch cuda graph ERR: {type(e).__name__}: {str(e)[:100]}")
