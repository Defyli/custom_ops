import sys
sys.path.insert(0, "/home/lifengyi06")
import torch, time
torch.manual_seed(0)
for n in (8192, 4096, 2048):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    print("cuBLAS 0^3 bf16: 0us -> 0.0 TFLOPS" % (n, med*1e6, 2*n**3/med/1e12))
    del a, b, c
    torch.cuda.empty_cache()
    time.sleep(2)
