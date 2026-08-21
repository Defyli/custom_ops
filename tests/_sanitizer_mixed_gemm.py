"""临时 sanitizer 冒烟脚本：覆盖 mixed_gemm 主要路径（各 tile 形状 / split-K / dtype）。"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PKG_ROOT))

import torch

from custom_ops import ops, split_mixed_precision_weight

torch.manual_seed(0)
SCALE = 1.0 / 256.0

BACKENDS = ["int8", "fp8"] if ops.mixed_gemm_fp8_available() else ["int8"]


def case(m, n, k, x_dtype, backend, activation="silu", has_bias=True, splitk=0):
    x = (torch.randn(m, k, device="cuda") * 0.5).to(x_dtype)
    w = torch.randn(n, k, device="cuda") * 0.05
    bias = torch.randn(n, device="cuda") * 0.1 if has_bias else None
    w_high, w_low, w_scale = split_mixed_precision_weight(w, SCALE, backend)
    y = ops.mixed_gemm(x, w_high, w_low, w_scale, scale=SCALE, bias=bias,
                       activation=activation, force_splitk=splitk)
    torch.cuda.synchronize()
    del x, w, bias, w_high, w_low, w_scale, y


def main():
    assert ops.is_available()
    for backend in BACKENDS:
        # 小 M（64x128 tile）、中 M（64x64 tile）、大 M（128x128 tile）
        case(16, 512, 1024, torch.float32, backend)               # m<=64 tile path
        case(16, 512, 1024, torch.float32, backend, splitk=4)     # m<=64 + split-K
        case(256, 4096, 2048, torch.float32, backend)             # 64x64 tile path
        case(256, 4096, 2048, torch.float32, backend, splitk=2)
        case(2048, 2048, 1024, torch.float32, backend)            # 128x128 tile path
        case(2048, 2048, 1024, torch.bfloat16, backend, activation="gelu", has_bias=False)
        case(1, 64, 64, torch.float32, backend)                   # 极小
        case(33, 200, 72, torch.float32, backend, splitk=16)      # ragged + 最大 split-K
    print("sanitizer smoke OK (backends: %s)" % ",".join(BACKENDS))


if __name__ == "__main__":
    main()
