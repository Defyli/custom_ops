"""
AOTI 跨进程部署语义：op 注册模块（tilelang kernel 的 custom op 契约）。
运行进程 import 本模块后即可加载 AOTI .so 调用 tilelang kernel。
"""
import sys

import torch

sys.path.insert(0, "/home/lifengyi06")

if "tilelang_aoti" not in str(getattr(torch.ops, "_loaded_ns", [])):
    try:
        torch.library.delete_library("tilelang_aoti")
    except Exception:
        pass

    @torch.library.custom_op("tilelang_aoti::swiglu", mutates_args=())
    def swiglu_tl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """SwiGLU：y = silu(x @ Wg^T) * (x @ Wu^T)（Tile-lang 后端）。"""
        from custom_ops.tilelang_ops._swiglu import swiglu_tilelang
        return swiglu_tilelang(x, weight)

    @swiglu_tl.register_fake
    def _(x, weight):
        return torch.empty(x.shape[:-1] + (weight.shape[0] // 2,),
                           dtype=x.dtype, device=x.device)
