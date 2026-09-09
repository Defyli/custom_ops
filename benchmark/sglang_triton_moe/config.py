# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sglang fused_moe config 查找逻辑的自包含移植（bf16/fp16 路径）。

从 sgl-project/sglang `python/sglang/srt/layers/moe/moe_runner/triton_utils/
fused_moe_triton_config.py` 提取：查表（M 最近邻）、default 启发式、
up/down 两份 config 的 BLOCK_SIZE_M 对齐约束。

与上游的差异：
  - configs 目录固定为本包内 `configs/`（可用 SGLANG_MOE_CONFIG_DIR 覆盖），
    文件名/目录名约定与上游一致（triton_<ver>/E=*,N=*,device_name=*.json，
    down 带 _down 后缀）；
  - 只保留非量化 dtype 分支（bf16/fp16 → 查表无 dtype 后缀）。
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import triton

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


def get_device_name(idx: int = 0) -> str:
    return torch.cuda.get_device_name(idx)


def get_config_file_name(
    E: int,
    N: int,
    dtype: Optional[str],
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    down_moe_selector = "_down" if down_moe else ""
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{down_moe_selector}.json"


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    down_moe: bool = False,
) -> Optional[Dict[int, Any]]:
    """返回 {M: config} 查表（M 最近邻取用）；无 tuned 文件时返回 None。"""
    config_dir = os.environ.get("SGLANG_MOE_CONFIG_DIR", _CONFIG_DIR)
    configs_root = os.path.join(config_dir, "configs")

    json_file_name = get_config_file_name(E, N, dtype, down_moe=down_moe)

    # 上游约定：优先精确匹配当前 Triton 版本目录，其次按版本降序回退
    triton_version = triton.__version__
    if not os.path.isdir(configs_root):
        return None
    available_versions = sorted(
        (
            d.removeprefix("triton_").replace("_", ".")
            for d in os.listdir(configs_root)
            if d.startswith("triton_")
        ),
        key=lambda v: tuple(int(x) for x in v.split(".")),
        reverse=True,
    )
    try_versions = [triton_version] + [v for v in available_versions
                                       if v != triton_version]
    for ver in try_versions:
        config_file_path = os.path.join(
            configs_root, f"triton_{ver.replace('.', '_')}", json_file_name)
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                return {int(key): val for key, val in json.load(f).items()}
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
) -> Dict[str, int]:
    """查表 miss 时的启发式（上游非量化分支逐字移植）。"""
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    return_down_config: bool = False,
):
    """解析 (up, down) config：显式注入优先，其次查表，最后 default。"""
    # w1: (E, 2I, H)，w2: (E, H, I)；config 文件名约定 N = w2.shape[2] = I
    E, _, N = w2_shape

    configs = get_moe_configs(E, N, dtype, down_moe=False)
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = get_default_config(M, E, N, w1_shape[2], top_k, dtype)

    down_config = None
    max_block_m = None
    if return_down_config:
        down_configs = get_moe_configs(E, N, dtype, down_moe=True)
        if down_configs:
            down_config = down_configs[
                min(down_configs.keys(), key=lambda x: abs(x - M))
            ]
            down_config = dict(**down_config)
            max_block_m = max(
                [cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()]
            )
        if (
            down_config is not None
            and config["BLOCK_SIZE_M"] != down_config["BLOCK_SIZE_M"]
        ):
            # Both kernels share one moe_align_block_size sort, so the down
            # config must use the up config's BLOCK_SIZE_M.
            down_config["BLOCK_SIZE_M"] = config["BLOCK_SIZE_M"]
        return config, (down_config, max_block_m)
    return config
