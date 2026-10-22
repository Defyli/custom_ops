"""Parameterized RAM-offload correctness, lifecycle, budget and checkpoint tests."""
from __future__ import annotations

import faulthandler
import functools
import gc
import json
import math
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import tempfile
import time
import unittest
import traceback
import weakref
from unittest import mock
from pathlib import Path
from dataclasses import FrozenInstanceError, replace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from custom_ops import moe_offload, ops
from custom_ops.moe_offload import (
    ExpertStore, LayerKeys, OffloadConfig, OffloadedMoELayer, PrefillMoERuntime,
)


def reference(x, w1, w2, ids, scale, expert_activation="silu", numerical_mode="fused", expert_type="glu"):
    """CPU FP32 per-expert reference for GEGLU/SwiGLU and both numeric modes.

    The CPU matmul is a genuine FP32 accumulation reference (never TF32/tensor
    cores). GELU is the exact erf form; staged simulates the kernel's cast
    chain (gate/up, activation, product, down, weighted product) with dtype
    rounding before the final FP32 reduction.
    """
    if expert_type not in ("glu", "mlp") or w1.shape[1] != (2 if expert_type == "glu" else 1) * w2.shape[2]:
        raise ValueError("expert_type projection shapes")
    xf, ids, scale = x.float().cpu(), ids.cpu(), scale.cpu()
    dtype = x.dtype

    def rounded(value):
        return value.to(dtype).float()

    out = torch.zeros_like(xf)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        gu = xf[rows] @ w1[e].float().T
        if numerical_mode == "staged":
            gu = rounded(gu)
        gate = gu.chunk(2, -1)[0] if expert_type == "glu" else gu
        if expert_activation == "gelu":
            act = 0.5 * gate * (1.0 + torch.erf(gate * math.sqrt(0.5)))
        else:
            act = gate * torch.sigmoid(gate)
        if numerical_mode == "staged" and expert_type == "glu":
            act = rounded(act)
        hidden = rounded(act * gu.chunk(2, -1)[1] if expert_type == "glu" else act)
        values = rounded(hidden @ w2[e].float().T)
        weighted = values * scale[rows, slots, None]
        if numerical_mode == "staged":
            weighted = rounded(weighted)
        out.index_add_(0, rows, weighted)
    return out.to(dtype).float()


class StoreTests(unittest.TestCase):
    def test_deployment_config_frozen_deep_snapshot_and_offline_target(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({5: tuple(w.clone() for w in pair), 0: pair})
        resident = {5: [2, 0], 0: [1]}
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.cuda, "get_device_capability") as capability):
            config = OffloadConfig.from_store(
                store, gpu_budget_bytes=200000, ram_budget_bytes=300000,
                pinned_budget_bytes=50000, target_sm=89,
                resident_cache_bytes=100000, resident_experts=resident)
            resident[5].append(1)
            resident.clear()
            self.assertEqual(config.layer_specs, ((0, 3, 64, 64, 0, (1,)),
                                                  (5, 3, 64, 64, 0, (0, 2))))
            self.assertIs(type(config.layer_specs), tuple)
            self.assertTrue(all(type(spec) is tuple and type(spec[5]) is tuple
                                for spec in config.layer_specs))
            self.assertEqual(config.layer_config(5),
                             (1, 200000, 300000, 50000, 2, 0, 4096, 100000, 0,
                              0, 32, 2, 0, 0, 0, 0, 128, 3, 64, 64, 0, 2, 89, 0, 2))
            with self.assertRaises(FrozenInstanceError):
                config.gpu_budget_bytes = 1
            with self.assertRaises(TypeError):
                config.layer_specs[0][5][0] = 2
            options = config.runtime_options()
            options["resident_experts"][5] = (1,)
            options["gpu_budget_bytes"] = 1
            self.assertEqual(config.layer_config(5)[23:], (0, 2))
            self.assertEqual(config.gpu_budget_bytes, 200000)
            mutable_specs = [[5, 3, 64, 64, 0, [2, 0]], [0, 3, 64, 64, 0, [1]]]
            snapshot = replace(config, layer_specs=mutable_specs)
            mutable_specs[0][5].clear()
            mutable_specs.clear()
            self.assertEqual(snapshot, config)
            self.assertEqual(hash(snapshot), hash(config))
            manifest = (1, 2, 0, len(config.layer_config(0)), *config.layer_config(0),
                        5, len(config.layer_config(5)), *config.layer_config(5))
            self.assertIs(type(config.manifest()), tuple)
            self.assertEqual(config.manifest(), manifest)
            capability.assert_not_called()
            # Hardware discovery is an explicit caller-side offline step, never
            # an implicit dependency of config construction or archive loading.
            for sm, device in (((8, 9), "cuda:3"), ((12, 0), "cuda:7")):
                capability.return_value = sm
                major, minor = torch.cuda.get_device_capability(device)
                captured = OffloadConfig.from_store(
                    store, gpu_budget_bytes=200000, ram_budget_bytes=300000,
                    pinned_budget_bytes=50000, target_sm=major * 10 + minor)
                self.assertEqual(captured.target_sm, major * 10 + minor)
                capability.assert_called_with(device)
            self.assertEqual(capability.call_count, 2)
            load.assert_not_called()
            current_device.assert_not_called()

    def test_deployment_config_validation_before_native_load(self):
        store = ExpertStore({0: (torch.zeros(3, 128, 64, dtype=torch.float16),
                                 torch.zeros(3, 64, 64, dtype=torch.float16))})
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=300000,
                       pinned_budget_bytes=50000, target_sm=89)
        invalid = {
            **{field: (0, -1, 1 << 63, True, False, 1.0, "auto", "1", None, [], {})
               for field in ("gpu_budget_bytes", "ram_budget_bytes", "pinned_budget_bytes")},
            **{field: (-1, 1 << 63, True, False, 1.0, "1", None)
               for field in ("resident_cache_bytes", "dynamic_cache_bytes", "cache_update_bytes")},
            "target_sm": (0, 79, 1 << 31, True, False, 89.0, "89", None),
            "num_banks": (0, 3, True, False, 1.0, "1", None),
            "activation_mode": ("", "route", 0, True, None, [], {}),
            "cache_policy": ("", "LRU", 0, True, None, [], {}),
            "max_route_tile_rows": (0, -1, 1 << 31, True, False, 1.0, "17", None),
            "cache_decay_interval": (0, 1, 1 << 31, True, False, 2.0, "2", None),
            "cache_min_reuse": (0, 1, 1 << 31, True, False, 2.0, "2", None),
            "async_staging": (0, 1, 1.0, "1", None),
            "host_profile": (0, 1, 1.0, "1", None),
            "tile_m": (-1, 16, 129, True, False, 32.0, "32", None),
            "tile_n": (0, 32, 256, True, False, 64.0, "64", None),
        }
        native = mock.Mock()
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.cuda, "get_device_capability") as capability,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                        OffloadConfig.from_store(store, **{**options, field: value})
            for field in options:
                with self.subTest(missing=field), self.assertRaises(TypeError):
                    OffloadConfig.from_store(store, **{k: v for k, v in options.items() if k != field})
            for resident in ({7: [0]}, {True: []}, {0: [True]}, {0: [1.0]},
                             {0: [-1]}, {0: [3]}, {0: [1, 1]}):
                with self.subTest(resident=resident), self.assertRaises(ValueError):
                    OffloadConfig.from_store(store, **options, resident_cache_bytes=100000,
                                             resident_experts=resident)
            with self.assertRaisesRegex(ValueError, "resident"):
                OffloadConfig.from_store(store, **options, resident_experts={0: [0]})
            with self.assertRaisesRegex(ValueError, "resident_cache_bytes"):
                OffloadConfig.from_store(store, **options, resident_cache_bytes=200001)
            with self.assertRaisesRegex(ValueError, "ram_budget_bytes"):
                OffloadConfig.from_store(store, **{**options, "ram_budget_bytes": store.weight_bytes - 1})
            config = OffloadConfig.from_store(store, **options)
            for specs in ((), ((0, 3, 64, 64, 0, ()),) * 2,
                          ((True, 3, 64, 64, 0, ()),), ((0, 0, 64, 64, 0, ()),),
                          ((0, 513, 64, 64, 0, ()),), ((0, 3, 65, 64, 0, ()),),
                          ((0, 3, 64, 65, 0, ()),), ((0, 3, 64, 64, 2, ()),),
                          ((0, 3, 64, 64, False, ()),), ((0, 3, 64, 64, 0),)):
                with self.subTest(specs=specs), self.assertRaises(ValueError):
                    replace(config, layer_specs=specs)
            # Large byte budgets remain int64, while tile/counter fields use INT_MAX.
            large = OffloadConfig.from_store(store, **{**options,
                "gpu_budget_bytes": (1 << 63) - 1, "ram_budget_bytes": 1 << 32,
                "pinned_budget_bytes": 1 << 32}, max_route_tile_rows="auto")
            self.assertEqual(large.layer_config(0)[1:4], ((1 << 63) - 1, 1 << 32, 1 << 32))
            self.assertEqual(large.layer_config(0)[6], 0)
            for bad in (None, {}, (), object()):
                with self.subTest(config=bad), self.assertRaisesRegex(TypeError, "OffloadConfig"):
                    PrefillMoERuntime.from_config(store, bad)
                if bad is not None:
                    with self.assertRaisesRegex(TypeError, "OffloadConfig"):
                        OffloadedMoELayer(config=bad)
            load.assert_not_called()
            current_device.assert_not_called()
            capability.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()

    def test_deployment_config_store_mismatch_before_load(self):
        def store_for(e=3, h=64, i=64, dtype=torch.float16, layer=0):
            return ExpertStore({layer: (torch.zeros(e, 2 * i, h, dtype=dtype),
                                       torch.zeros(e, h, i, dtype=dtype))})
        store = store_for()
        config = OffloadConfig.from_store(store, gpu_budget_bytes=200000,
            ram_budget_bytes=300000, pinned_budget_bytes=50000, target_sm=89)
        mismatches = (store_for(e=2), store_for(h=128), store_for(i=128),
                      store_for(dtype=torch.bfloat16), store_for(layer=5),
                      ExpertStore({}), ExpertStore({**store.layers, **store_for(layer=5).layers}))
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.cuda, "get_device_capability") as capability,
              tempfile.TemporaryDirectory() as temporary):
            for other in mismatches:
                with self.subTest(specs=tuple(other.layers)):
                    with self.assertRaisesRegex(ValueError, "deployment configuration mismatch"):
                        config.validate_store(other)
                    with self.assertRaisesRegex(ValueError, "deployment configuration mismatch"):
                        PrefillMoERuntime.from_config(other, config)
                    if other.layers:
                        with self.assertRaisesRegex(ValueError, "deployment configuration mismatch"):
                            other.save_cpp(Path(temporary) / "invalid.pt", config=config)
            config.validate_store(store_for())  # Contract describes layout, not storage identity.
            with self.assertRaisesRegex(ValueError, "not in the deployment"):
                OffloadedMoELayer(5, config=config)
            load.assert_not_called()
            current_device.assert_not_called()
            capability.assert_not_called()

    def test_deployment_config_wrong_sm_and_total_resident_cap_before_init(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=300000,
                       pinned_budget_bytes=50000, target_sm=89)
        expert_bytes = 6 * 64 * 64
        native = mock.Mock()
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)) as capability,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            # Each layer fits individually, but the model-wide fixed cache does not.
            for cap in (expert_bytes, 2 * expert_bytes - 1):
                with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "resident_cache_bytes"):
                    OffloadConfig.from_store(store, **options, resident_cache_bytes=cap,
                                             resident_experts={0: [0], 5: [1]})
            config = OffloadConfig.from_store(store, **options,
                resident_cache_bytes=2 * expert_bytes, resident_experts={0: [0], 5: [1]})
            with self.assertRaisesRegex(ValueError, "deployment configuration mismatch: target_sm"):
                PrefillMoERuntime.from_config(store, config, device="cuda:3")
            capability.assert_called_once_with(torch.device("cuda:3"))
            current_device.assert_not_called()
            load.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()
            native.fuse_moe_offload_bind.assert_not_called()
            native.fuse_moe_offload_resident_bind.assert_not_called()
            native.fuse_moe_offload_close.assert_not_called()

    def test_deployment_config_canonical_vectors_reject_non_int64_fields(self):
        store = ExpertStore({0: (torch.zeros(3, 128, 64, dtype=torch.float16),
                                 torch.zeros(3, 64, 64, dtype=torch.float16))})
        config = OffloadConfig.from_store(store, gpu_budget_bytes=200000,
            ram_budget_bytes=300000, pinned_budget_bytes=50000, target_sm=89)
        canonical = list(config.layer_config(0))
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "get_device_capability") as capability):
            for index in range(len(canonical)):
                for value in (True, float(canonical[index]), str(canonical[index]),
                              None, 1 << 63, -(1 << 63) - 1):
                    altered = canonical.copy()
                    altered[index] = value
                    with self.subTest(index=index, value=value), self.assertRaisesRegex(
                            ValueError, "deployment configuration format"):
                        OffloadConfig.from_layer_configs({0: altered})
            for values in ({}, [], {True: canonical}, {0: iter(canonical)}, {0: "invalid"}):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    OffloadConfig.from_layer_configs(values)
            self.assertEqual(OffloadConfig.from_layer_configs({0: canonical}), config)
            load.assert_not_called()
            capability.assert_not_called()

    def test_deployment_config_runtime_schema_and_forward_contract(self):
        store = ExpertStore({5: (torch.zeros(3, 128, 64, dtype=torch.bfloat16),
                                 torch.zeros(3, 64, 64, dtype=torch.bfloat16))})
        config = OffloadConfig.from_store(store, gpu_budget_bytes=200000,
            ram_budget_bytes=300000, pinned_budget_bytes=50000, target_sm=89,
            num_banks=1, activation_mode="fp32_accumulate", max_route_tile_rows="auto",
            resident_cache_bytes=50000, resident_experts={5: [2, 0]},
            dynamic_cache_bytes=60000, cache_policy="lru", cache_decay_interval=7,
            cache_min_reuse=3, cache_update_bytes=25000, async_staging=True,
            host_profile=True, tile_m=32, tile_n=64)
        expected = (1, 200000, 300000, 50000, 1, 1, 0, 50000, 60000, 1,
                    7, 3, 25000, 1, 1, 32, 64, 3, 64, 64, 1, 2, 89, 0, 2)
        self.assertEqual(config.layer_config(5), expected)
        manifest = (1, 1, 5, len(expected), *expected)
        self.assertEqual(config.manifest(), manifest)
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        ids = torch.zeros(2, 1, dtype=torch.int32)
        scale = torch.ones(2, 1)
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native),
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)) as capability,
              mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0",
                  "FUSE_MOE_OFFLOAD_PROFILE": "0", "FUSE_MOE_TILE_M": "128",
                  "FUSE_MOE_TILE_N": "128"})):
            with PrefillMoERuntime.from_config(store, config, device="cuda:3") as runtime:
                native.fuse_moe_offload_create.assert_called_once_with(
                    3, 200000, 300000, 50000, 1, 1, 0, 50000, 60000, 1, 7, 3, 25000,
                    True, True, 32, 64, 0, 0, 0)
                self.assertIs(runtime.deployment_config, config)
                self.assertEqual(native.mock_calls[1:3], [
                    mock.call.fuse_moe_offload_bind(token, 5, *store.layers[5]),
                    mock.call.fuse_moe_offload_resident_bind(token, 5, [0, 2])])
                self.assertEqual(native.mock_calls[3],
                                 mock.call.fuse_moe_offload_validate(token, manifest))
                native.fuse_moe_offload_validate.assert_called_once_with(token, manifest)
                output = runtime.forward(x, ids, scale, layer_id=5)
                native.fuse_moe_offload.assert_called_once_with(
                    x, *store.layers[5], ids, scale, token, 5, expected, manifest)
                self.assertIs(output, native.fuse_moe_offload.return_value)
                native.fuse_moe_offload.reset_mock()
                adapter = OffloadedMoELayer(5, config=config)
                self.assertIs(type(adapter.deployment_config), tuple)
                self.assertEqual(adapter.deployment_config, expected)
                self.assertIs(type(adapter.deployment_manifest), tuple)
                self.assertEqual(adapter.deployment_manifest, manifest)
                self.assertEqual(tuple(adapter.named_buffers()), ())
                adapter(x, *store.layers[5], ids, scale, token)
                native.fuse_moe_offload.assert_called_once_with(
                    x, *store.layers[5], ids, scale, token, 5, expected, manifest)
            native.fuse_moe_offload_close.assert_called_once_with(token)
            native.reset_mock()
            with PrefillMoERuntime(store, device="cuda:3") as legacy:
                legacy.forward(x, ids, scale, layer_id=5)
                native.fuse_moe_offload.assert_called_once_with(
                    x, *store.layers[5], ids, scale, token, 5, (), ())
                native.fuse_moe_offload.reset_mock()
                OffloadedMoELayer(5)(x, *store.layers[5], ids, scale, token)
                native.fuse_moe_offload.assert_called_once_with(
                    x, *store.layers[5], ids, scale, token, 5, (), ())
            current_device.assert_not_called()
            capability.assert_called_once_with(torch.device("cuda:3"))

    def test_runtime_validate_config_uses_artifact_and_revalidates_current_store(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: pair})
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=300000,
                       pinned_budget_bytes=50000, target_sm=89)
        config = OffloadConfig.from_store(store, **options)
        artifact = OffloadConfig.from_store(store, **{**options, "num_banks": 1})
        wrong_store = ExpertStore({0: pair})
        wrong_config = OffloadConfig.from_store(wrong_store, **options)
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native),
              mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9))):
            with PrefillMoERuntime.from_config(store, config, device="cuda:3") as runtime:
                native.reset_mock()
                # The package artifact, not runtime.deployment_config, is authoritative.
                self.assertIsNone(runtime.validate_config(artifact))
                self.assertEqual(native.mock_calls, [
                    mock.call.fuse_moe_offload_validate(token, artifact.manifest())])
                self.assertIs(runtime.deployment_config, config)
                self.assertEqual(runtime._deployment_manifest, config.manifest())
                native.reset_mock()
                for bad in (None, {}, (), object()):
                    with self.subTest(config=bad), self.assertRaisesRegex(
                            TypeError, "config must be an OffloadConfig"):
                        runtime.validate_config(bad)
                with self.assertRaisesRegex(ValueError, "store layers/weights"):
                    runtime.validate_config(wrong_config)
                # Valid but different exact H/I/E/dtype must fail before native validation.
                for replacement in (
                        (torch.zeros(3, 128, 128, dtype=torch.float16),
                         torch.zeros(3, 128, 64, dtype=torch.float16)),
                        (torch.zeros(3, 256, 64, dtype=torch.float16),
                         torch.zeros(3, 64, 128, dtype=torch.float16)),
                        (torch.zeros(2, 128, 64, dtype=torch.float16),
                         torch.zeros(2, 64, 64, dtype=torch.float16)),
                        tuple(w.to(torch.bfloat16) for w in pair)):
                    store.layers[5] = replacement
                    with self.subTest(shapes=tuple(w.shape for w in replacement),
                                      dtype=replacement[0].dtype), self.assertRaisesRegex(
                            ValueError, "store layers/weights"):
                        runtime.validate_config(config)
                store.layers[5] = (torch.zeros(3, 126, 64, dtype=torch.float16), pair[1])
                with self.assertRaisesRegex(ValueError, "invalid canonical E/H/I"):
                    runtime.validate_config(config)
                store.layers[5] = pair
                native.fuse_moe_offload_validate.assert_not_called()
                # Recomputed storage bytes must also reject a formerly tied store.
                tight = OffloadConfig.from_store(store, **{**options,
                    "ram_budget_bytes": store.weight_bytes})
                store.layers[5] = tuple(w.clone() for w in pair)
                with self.assertRaisesRegex(ValueError, "CPU backing storages exceed"):
                    runtime.validate_config(tight)
                store.layers[5] = pair
                native.fuse_moe_offload_validate.assert_not_called()
                error = RuntimeError("artifact native mismatch")
                native.fuse_moe_offload_validate.side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    runtime.validate_config(artifact)
                self.assertIs(raised.exception, error)
                native.fuse_moe_offload_validate.assert_called_once_with(token, artifact.manifest())
                self.assertFalse(runtime._closed)
                native.fuse_moe_offload_close.assert_not_called()
            native.fuse_moe_offload_close.assert_called_once_with(token)
            native.reset_mock()
            for bad in (config, artifact, None):
                with self.subTest(closed_config=bad), self.assertRaisesRegex(
                        RuntimeError, "offload runtime is closed"):
                    runtime.validate_config(bad)
            self.assertEqual(native.mock_calls, [])

    def test_deployment_config_from_config_failure_cleanup_baseexception(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        config = OffloadConfig.from_store(store, gpu_budget_bytes=200000,
            ram_budget_bytes=300000, pinned_budget_bytes=50000, target_sm=89,
            resident_cache_bytes=50000, resident_experts={5: [1]})
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native),
              mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9))):
            for operation in ("fuse_moe_offload_create", "fuse_moe_offload_bind",
                              "fuse_moe_offload_resident_bind", "fuse_moe_offload_validate"):
                for error_type in (RuntimeError, KeyboardInterrupt, SystemExit):
                    native.reset_mock(side_effect=True)
                    error = error_type("deployment setup failure")
                    getattr(native, operation).side_effect = (
                        [None, error] if operation == "fuse_moe_offload_bind" else error)
                    with self.subTest(operation=operation, error_type=error_type):
                        with self.assertRaises(error_type) as raised:
                            PrefillMoERuntime.from_config(store, config, device="cuda:3")
                        self.assertIs(raised.exception, error)
                        gc.collect()
                        native.fuse_moe_offload_create.assert_called_once()
                        if operation == "fuse_moe_offload_create":
                            native.fuse_moe_offload_bind.assert_not_called()
                            native.fuse_moe_offload_close.assert_not_called()
                        else:
                            native.fuse_moe_offload_close.assert_called_once_with(token)
                            self.assertEqual(native.mock_calls[-1],
                                             mock.call.fuse_moe_offload_close(token))
                        if operation == "fuse_moe_offload_validate":
                            self.assertEqual(native.mock_calls[1:4], [
                                mock.call.fuse_moe_offload_bind(token, 0, *store.layers[0]),
                                mock.call.fuse_moe_offload_bind(token, 5, *store.layers[5]),
                                mock.call.fuse_moe_offload_resident_bind(token, 5, [1])])
                            native.fuse_moe_offload_validate.assert_called_once_with(
                                token, config.manifest())
                        else:
                            native.fuse_moe_offload_validate.assert_not_called()

    def test_runtime_configuration_validation(self):
        w1 = torch.zeros(2, 128, 64, dtype=torch.float16)
        w2 = torch.zeros(2, 64, 64, dtype=torch.float16)
        store = ExpertStore({0: (w1, w2)})
        # Invalid configuration fails before loading native code or touching CUDA.
        with mock.patch.object(ops, "ensure_loaded") as load:
            for mode in ("route", "accumulate", "", None, 0, [], {}):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "activation_mode"):
                    PrefillMoERuntime(store, activation_mode=mode)
            for rows in (0, -1, 1.5, "17", None, True, False):
                with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "max_route_tile_rows"):
                    PrefillMoERuntime(store, max_route_tile_rows=rows)
            load.assert_not_called()

    def test_explicit_execution_configuration_validation_before_load(self):
        store = ExpertStore({0: (torch.zeros(2, 128, 64, dtype=torch.float16),
                                 torch.zeros(2, 64, 64, dtype=torch.float16))})
        invalid = {
            "async_staging": (0, 1, -1, 0.0, 1.0, "0", "1", "true", None, [], {}),
            "host_profile": (0, 1, -1, 0.0, 1.0, "0", "1", "true", None, [], {}),
            "tile_m": (-1, 1, 16, 31, 33, 65, 129, 1 << 31,
                       True, False, 0.0, 32.0, "32", None, [], {}),
            "tile_n": (-1, 0, 1, 32, 63, 65, 129, 256, 1 << 31,
                       True, False, 64.0, "128", None, [], {}),
        }
        native = mock.Mock()
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                        PrefillMoERuntime(store, **{field: value})
            load.assert_not_called()
            current_device.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()
            native.fuse_moe_offload_close.assert_not_called()

    def test_explicit_execution_configuration_schema_ignores_environment(self):
        store = ExpertStore({0: (torch.zeros(2, 128, 64, dtype=torch.float16),
                                 torch.zeros(2, 64, 64, dtype=torch.float16))})
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.ops, "recsys_ops", native),
              mock.patch.dict(os.environ)):
            for async_staging in (False, True):
                for host_profile in (False, True):
                    for tile_m in (0, 32, 64, 128):
                        for tile_n in (64, 128):
                            native.reset_mock()
                            os.environ.update({
                                "FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0" if async_staging else "1",
                                "FUSE_MOE_OFFLOAD_PROFILE": "0" if host_profile else "1",
                                "FUSE_MOE_TILE_M": "128" if tile_m != 128 else "32",
                                "FUSE_MOE_TILE_N": "128" if tile_n == 64 else "64",
                            })
                            with self.subTest(async_staging=async_staging, host_profile=host_profile,
                                              tile_m=tile_m, tile_n=tile_n), PrefillMoERuntime(
                                    store, device="cuda:3", async_staging=async_staging,
                                    host_profile=host_profile, tile_m=tile_m, tile_n=tile_n):
                                native.fuse_moe_offload_create.assert_called_once_with(
                                    3, 0, 0, 0, 2, 0, 4096, 0, 0, 0, 32, 2, 0,
                                    async_staging, host_profile, tile_m, tile_n, 0, 0, 0)
                                for argument in native.fuse_moe_offload_create.call_args.args[13:15]:
                                    self.assertIs(type(argument), bool)
                                for argument in native.fuse_moe_offload_create.call_args.args[15:20]:
                                    self.assertIs(type(argument), int)
                            native.fuse_moe_offload_close.assert_called_once_with(token)
            native.reset_mock()
            os.environ.update({"FUSE_MOE_OFFLOAD_ASYNC_STAGING": "1",
                               "FUSE_MOE_OFFLOAD_PROFILE": "1",
                               "FUSE_MOE_TILE_M": "32", "FUSE_MOE_TILE_N": "64"})
            with PrefillMoERuntime(store, device="cuda:3"):
                native.fuse_moe_offload_create.assert_called_once_with(
                    3, 0, 0, 0, 2, 0, 4096, 0, 0, 0, 32, 2, 0, False, False, 0, 128, 0, 0, 0)
            current_device.assert_not_called()

    def test_runtime_configuration_schema(self):
        w1 = torch.zeros(2, 128, 64, dtype=torch.float16)
        w2 = torch.zeros(2, 64, 64, dtype=torch.float16)
        store = ExpertStore({0: (w1, w2)}, ram_budget_bytes=100000)
        native = mock.Mock()
        native.fuse_moe_offload_create.return_value = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_stats.return_value = torch.arange(31)
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for mode, value in (("route_buffer", 0), ("fp32_accumulate", 1), ("auto", 2)):
                with PrefillMoERuntime(store, device="cuda:3", gpu_budget_bytes=200000,
                                       pinned_budget_bytes=50000, num_banks=1,
                                       activation_mode=mode, max_route_tile_rows=17) as runtime:
                    native.fuse_moe_offload_create.assert_called_with(
                        3, 200000, 100000, 50000, 1, value, 17, 0, 0, 0, 32, 2, 0,
                        False, False, 0, 128, 0, 0, 0)
                    self.assertEqual(runtime.activation_mode, mode)
                    self.assertEqual(runtime.max_route_tile_rows, 17)
                    stats = runtime.stats()
                    self.assertEqual(tuple(stats), (
                        "active_experts", "waves", "weight_h2d_bytes", "metadata_h2d_bytes",
                        "gpu_charged_bytes", "pinned_bytes", "ram_charged_bytes", "num_banks",
                        "wave_capacity", "activation_mode", "route_tiles", "route_tile_rows",
                        "buffer_allocations", "buffer_reuses", "gpu_pool_bytes", "pinned_pool_bytes",
                        "cache_hits", "cache_misses", "cache_hit_weight_bytes", "resident_cache_bytes",
                        "resident_experts", "resident_load_h2d_bytes",
                        "dynamic_cache_bytes", "dynamic_experts", "dynamic_cache_hits",
                        "cache_admissions", "cache_evictions", "cache_bypasses",
                        "cache_promotion_d2d_bytes", "cache_budget_evictions", "cache_metadata_bytes"))
                    self.assertEqual(list(stats.values()), list(range(31)))
            with PrefillMoERuntime(store, device="cuda:3") as runtime:
                native.fuse_moe_offload_create.assert_called_with(
                    3, 0, 100000, 0, 2, 0, 4096, 0, 0, 0, 32, 2, 0,
                    False, False, 0, 128, 0, 0, 0)
                self.assertEqual(runtime.dynamic_cache_bytes, 0)
                self.assertEqual(runtime.cache_policy, "reuse")
                self.assertEqual(runtime.cache_decay_interval, 32)
                self.assertEqual(runtime.cache_min_reuse, 2)
                self.assertEqual(runtime.cache_update_bytes, 0)

    def test_adaptive_configuration_validation_before_load(self):
        store = ExpertStore({0: (torch.zeros(2, 128, 64, dtype=torch.float16),
                                 torch.zeros(2, 64, 64, dtype=torch.float16))})
        native = mock.Mock()
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.cuda, "current_device", return_value=0) as current_device,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            native.fuse_moe_offload_create.return_value = torch.zeros(2, dtype=torch.int64)
            for rows in (0, -1, -(1 << 31), 1 << 31, 1 << 32, True, False,
                         1.0, 0.0, float("inf"), "0", "17", "AUTO", "", None, [], {}):
                with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "max_route_tile_rows"):
                    runtime = PrefillMoERuntime(store, max_route_tile_rows=rows)
                    runtime.close()
            load.assert_not_called()
            current_device.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()
            native.fuse_moe_offload_close.assert_not_called()

    def test_adaptive_configuration_schema_and_defaults(self):
        store = ExpertStore({0: (torch.zeros(2, 128, 64, dtype=torch.float16),
                                 torch.zeros(2, 64, 64, dtype=torch.float16))},
                            ram_budget_bytes=100000)
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        native.fuse_moe_offload_stats.return_value = torch.arange(31)
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for mode, mode_id in (("route_buffer", 0), ("fp32_accumulate", 1), ("auto", 2)):
                for rows, native_rows in (("auto", 0), (1, 1), (4096, 4096),
                                          ((1 << 31) - 1, (1 << 31) - 1)):
                    native.reset_mock()
                    with self.subTest(mode=mode, rows=rows), PrefillMoERuntime(
                            store, device="cuda:3", gpu_budget_bytes=200000,
                            pinned_budget_bytes=50000, num_banks=1,
                            activation_mode=mode, max_route_tile_rows=rows,
                            resident_cache_bytes=30000, resident_experts={0: [1]},
                            dynamic_cache_bytes=40000, cache_policy="lru",
                            cache_decay_interval=7, cache_min_reuse=3,
                            cache_update_bytes=25000) as runtime:
                        native.fuse_moe_offload_create.assert_called_once_with(
                            3, 200000, 100000, 50000, 1, mode_id, native_rows,
                            30000, 40000, 1, 7, 3, 25000, False, False, 0, 128, 0, 0, 0)
                        self.assertIs(type(native.fuse_moe_offload_create.call_args.args[6]), int)
                        self.assertEqual(runtime.max_route_tile_rows, rows)
                        self.assertEqual(runtime.activation_mode, mode)
                        self.assertEqual(list(runtime.stats().values()), list(range(31)))
                    if native.fuse_moe_offload_create.called:
                        native.fuse_moe_offload_close.assert_called_once_with(token)
            native.reset_mock()
            with PrefillMoERuntime(store, device="cuda:3") as runtime:
                native.fuse_moe_offload_create.assert_called_once_with(
                    3, 0, 100000, 0, 2, 0, 4096, 0, 0, 0, 32, 2, 0,
                    False, False, 0, 128, 0, 0, 0)
                self.assertEqual(runtime.max_route_tile_rows, 4096)
                self.assertEqual(runtime.activation_mode, "route_buffer")
            current_device.assert_not_called()

    def test_resident_configuration_validation_before_load(self):
        store = ExpertStore({0: (torch.zeros(3, 128, 64, dtype=torch.float16),
                                 torch.zeros(3, 64, 64, dtype=torch.float16))})
        invalid = [dict(resident_cache_bytes=value) for value in (-1, True, False, 1.5, "1", None)]
        invalid += [dict(resident_cache_bytes=100000, resident_experts=value) for value in (
            [], [(0, [1])], {7: []}, {True: []}, {"0": []}, {0: None}, {0: "01"},
            {0: {1}}, {0: iter([1])}, {0: [True]}, {0: [False]}, {0: [1.0]},
            {0: ["1"]}, {0: [-1]}, {0: [3]}, {0: [1, 1]})]
        invalid.append(dict(resident_experts={0: [0]}))
        with mock.patch.object(ops, "ensure_loaded") as load:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "resident"):
                    PrefillMoERuntime(store, **kwargs)
            load.assert_not_called()

    def test_resident_configuration_schema_order_and_cleanup(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        native.fuse_moe_offload_stats.return_value = torch.arange(31)
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native)):
            config = {5: [2, 0], 0: []}
            with PrefillMoERuntime(store, device="cuda:3", resident_cache_bytes=100000,
                                   resident_experts=config) as runtime:
                native.fuse_moe_offload_create.assert_called_once_with(
                    3, 0, 0, 0, 2, 0, 4096, 100000, 0, 0, 32, 2, 0,
                    False, False, 0, 128, 0, 0, 0)
                self.assertEqual(native.mock_calls[1:5], [
                    mock.call.fuse_moe_offload_bind(token, 0, *store.layers[0]),
                    mock.call.fuse_moe_offload_bind(token, 5, *store.layers[5]),
                    mock.call.fuse_moe_offload_resident_bind(token, 5, [2, 0]),
                    mock.call.fuse_moe_offload_resident_bind(token, 0, [])])
                config[5].append(1)
                config.clear()
                self.assertEqual(runtime.resident_cache_bytes, 100000)
                self.assertEqual(dict(runtime.resident_experts), {5: (2, 0), 0: ()})
                with self.assertRaises(TypeError):
                    runtime.resident_experts[0] = (1,)
                self.assertEqual(len(runtime.stats()), 31)
                for count in (0, 16, 22, 30, 32):
                    native.fuse_moe_offload_stats.return_value = torch.arange(count)
                    with self.subTest(count=count), self.assertRaisesRegex(
                            RuntimeError, f"schema mismatch: expected 31 values, got {count}"):
                        runtime.stats()
            native.fuse_moe_offload_close.assert_called_once_with(token)
            for config in (None, {}, {0: []}):
                native.reset_mock()
                with PrefillMoERuntime(store, device="cuda:3", resident_experts=config) as runtime:
                    self.assertEqual(runtime.resident_cache_bytes, 0)
                    self.assertEqual(dict(runtime.resident_experts),
                                     {layer: tuple(ids) for layer, ids in (config or {}).items()})
                    self.assertEqual(native.fuse_moe_offload_resident_bind.call_count,
                                     int(config == {0: []}))
            for failing_op in (native.fuse_moe_offload_bind, native.fuse_moe_offload_resident_bind):
                native.reset_mock()
                failing_op.side_effect = RuntimeError("bind failure")
                with self.assertRaisesRegex(RuntimeError, "bind failure"):
                    PrefillMoERuntime(store, device="cuda:3", resident_cache_bytes=100000,
                                      resident_experts={0: [1]})
                native.fuse_moe_offload_close.assert_called_once_with(token)
                failing_op.side_effect = None

    def test_dynamic_configuration_validation_before_load(self):
        store = ExpertStore({0: (torch.zeros(3, 128, 64, dtype=torch.float16),
                                 torch.zeros(3, 64, 64, dtype=torch.float16))})
        invalid = {
            "dynamic_cache_bytes": (-1, True, False, 1.5, "1", None),
            "cache_policy": ("", "LRU", "frequency", 0, 1, True, False, None, [], {}),
            "cache_decay_interval": (0, 1, -1, 1 << 31, True, False, 1.5, "1", None),
            "cache_min_reuse": (0, 1, -1, 1 << 31, True, False, 2.5, "2", None),
            "cache_update_bytes": (-1, True, False, 1.5, "1", None),
        }
        native = mock.Mock()
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                        PrefillMoERuntime(store, **{field: value})
            load.assert_not_called()
            current_device.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()
            native.fuse_moe_offload_close.assert_not_called()

    def test_dynamic_configuration_schema_and_independent_caps(self):
        store = ExpertStore({0: (torch.zeros(3, 128, 64, dtype=torch.float16),
                                 torch.zeros(3, 64, 64, dtype=torch.float16))})
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        native.fuse_moe_offload_stats.return_value = torch.arange(31)
        # Caps are independent and need not sum to less than the total GPU budget.
        # Zero update cap is passed through; bytes are not restricted to INT_MAX.
        configs = ((0, 0, None, 2, 2, 0),
                   (1 << 32, 0, None, (1 << 31) - 1, (1 << 31) - 1, 0),
                   (200000, 150000, {0: [1]}, 7, 3, 1 << 32))
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for policy, value in (("reuse", 0), ("lru", 1)):
                for dynamic, resident, experts, decay, minimum, update in configs:
                    native.reset_mock()
                    with self.subTest(policy=policy, dynamic=dynamic, update=update):
                        with PrefillMoERuntime(
                                store, device="cuda:3", gpu_budget_bytes=100000,
                                resident_cache_bytes=resident, resident_experts=experts,
                                dynamic_cache_bytes=dynamic, cache_policy=policy,
                                cache_decay_interval=decay, cache_min_reuse=minimum,
                                cache_update_bytes=update) as runtime:
                            native.fuse_moe_offload_create.assert_called_once_with(
                                3, 100000, 0, 0, 2, 0, 4096, resident,
                                dynamic, value, decay, minimum, update, False, False, 0, 128, 0, 0, 0)
                            self.assertEqual(runtime.dynamic_cache_bytes, dynamic)
                            self.assertEqual(runtime.cache_policy, policy)
                            self.assertEqual(runtime.cache_decay_interval, decay)
                            self.assertEqual(runtime.cache_min_reuse, minimum)
                            self.assertEqual(runtime.cache_update_bytes, update)
                            self.assertEqual(runtime.resident_cache_bytes, resident)
                            self.assertEqual(dict(runtime.resident_experts),
                                             {layer: tuple(ids) for layer, ids in (experts or {}).items()})
                            native.fuse_moe_offload_bind.assert_called_once_with(token, 0, *store.layers[0])
                            if experts is None:
                                native.fuse_moe_offload_resident_bind.assert_not_called()
                            else:
                                native.fuse_moe_offload_resident_bind.assert_called_once_with(token, 0, [1])
                            stats = runtime.stats()
                            self.assertEqual(len(stats), 31)
                            self.assertEqual(list(stats.values())[22:], list(range(22, 31)))
                        runtime.close()
                        native.fuse_moe_offload_close.assert_called_once_with(token)

    def test_dynamic_configuration_failure_cleanup(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        config = dict(device="cuda:3", resident_cache_bytes=100000,
                      resident_experts={0: [1]}, dynamic_cache_bytes=200000,
                      cache_policy="lru", cache_decay_interval=7, cache_min_reuse=3,
                      cache_update_bytes=50000)
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for operation in ("fuse_moe_offload_create", "fuse_moe_offload_bind",
                              "fuse_moe_offload_resident_bind"):
                for error_type in (RuntimeError, KeyboardInterrupt):
                    native.reset_mock(side_effect=True)
                    failing_op = getattr(native, operation)
                    # Fail after a partial layer binding to check context cleanup.
                    error = error_type("dynamic setup failure")
                    failing_op.side_effect = [None, error] if operation == "fuse_moe_offload_bind" else error
                    with self.subTest(operation=operation, error_type=error_type):
                        with self.assertRaisesRegex(error_type, "dynamic setup failure"):
                            PrefillMoERuntime(store, **config)
                        gc.collect()
                        native.fuse_moe_offload_create.assert_called_once_with(
                            3, 0, 0, 0, 2, 0, 4096, 100000, 200000, 1, 7, 3, 50000,
                            False, False, 0, 128, 0, 0, 0)
                        if operation == "fuse_moe_offload_create":
                            native.fuse_moe_offload_bind.assert_not_called()
                            native.fuse_moe_offload_close.assert_not_called()
                        else:
                            native.fuse_moe_offload_close.assert_called_once_with(token)
                            self.assertEqual(native.mock_calls[-1], mock.call.fuse_moe_offload_close(token))
                            if operation == "fuse_moe_offload_bind":
                                native.fuse_moe_offload_resident_bind.assert_not_called()

    def test_cpp_archive_tied_snapshot_ram_checked_before_destination_write(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        tied = ExpertStore({0: pair, 5: pair})
        self.assertEqual(tied.weight_bytes, 3 * 6 * 64 * 64)
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=tied.weight_bytes,
                       pinned_budget_bytes=50000, target_sm=89)
        config = OffloadConfig.from_store(tied, **options)
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "get_device_capability") as capability):
            path = Path(temporary) / "weights.pt"
            tied.save_cpp(path)  # Preserve an existing valid destination on failure.
            original = path.read_bytes()
            missing = Path(temporary) / "not_created.pt"
            with mock.patch.object(torch.jit, "save") as save:
                for destination in (path, missing):
                    with self.subTest(destination=destination.name), self.assertRaisesRegex(
                            ValueError, "ram_budget_bytes"):
                        tied.save_cpp(destination, config=config)
                save.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(missing.exists())
            self.assertEqual(tuple(ExpertStore.load_cpp(path).layers), (0, 5))
            # The exact independent-clone budget permits the format2 snapshot.
            expanded = replace(config, ram_budget_bytes=2 * tied.weight_bytes)
            tied.save_cpp(path, config=expanded)
            restored = ExpertStore.load_cpp(path)
            self.assertEqual(restored.weight_bytes, 2 * tied.weight_bytes)
            self.assertEqual(restored.deployment_config, expanded)
            load.assert_not_called()
            capability.assert_not_called()

    def test_deployment_config_mutated_store_revalidated_before_load_or_save(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: pair})
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=store.weight_bytes,
                       pinned_budget_bytes=50000, target_sm=89)
        config = OffloadConfig.from_store(store, **options)
        cached_bytes = store.weight_bytes
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "get_device_capability") as capability,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              mock.patch.object(torch.ops, "recsys_ops", mock.Mock()) as native):
            path = Path(temporary) / "weights.pt"
            store.save_cpp(path)
            original = path.read_bytes()
            cases = (("bad_shape", (torch.zeros(3, 126, 64, dtype=torch.float16), pair[1])),
                     ("expanded_storage", tuple(w.clone() for w in pair)))
            for name, changed in cases:
                store.layers[5] = changed
                self.assertEqual(store.weight_bytes, cached_bytes)  # Stale accounting must not be trusted.
                with self.subTest(case=name), mock.patch.object(torch.jit, "save") as save:
                    for operation in (lambda: config.validate_store(store),
                                      lambda: OffloadConfig.from_store(store, **options),
                                      lambda: PrefillMoERuntime.from_config(store, config, device="cuda:3"),
                                      lambda: store.save_cpp(path, config=config)):
                        with self.assertRaises(ValueError):
                            operation()
                    save.assert_not_called()
                    self.assertEqual(path.read_bytes(), original)
            store.layers[5] = pair
            config.validate_store(store)
            load.assert_not_called()
            capability.assert_not_called()
            current_device.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()

    def test_cpp_archive_deployment_format2_roundtrip(self):
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "get_device_capability") as capability,
              mock.patch.object(torch.cuda, "current_device") as current_device,
              tempfile.TemporaryDirectory() as temporary):
            for dtype in (torch.float16, torch.bfloat16):
                for target_sm in (89, 120):
                    with self.subTest(dtype=dtype, target_sm=target_sm):
                        pair = (torch.randn(3, 128, 64).to(dtype),
                                torch.randn(3, 64, 64).to(dtype))
                        store = ExpertStore({5: tuple(w.clone() for w in pair), 0: pair})
                        config = OffloadConfig.from_store(store, gpu_budget_bytes=200000,
                            ram_budget_bytes=300000, pinned_budget_bytes=50000, target_sm=target_sm,
                            resident_cache_bytes=100000, resident_experts={5: [2, 0]},
                            dynamic_cache_bytes=50000, cache_policy="lru", num_banks=1,
                            activation_mode="fp32_accumulate", max_route_tile_rows="auto",
                            cache_decay_interval=7, cache_min_reuse=3, cache_update_bytes=25000,
                            async_staging=True, host_profile=True, tile_m=32, tile_n=64)
                        path = Path(temporary) / "configured.pt"
                        with torch.inference_mode():
                            store.save_cpp(path, weight_version=17, config=config)
                            restored = ExpertStore.load_cpp(path)
                        self.assertEqual(restored.deployment_config, config)
                        self.assertEqual(restored.deployment_config.manifest(), config.manifest())
                        self.assertEqual(restored.weight_version, 17)
                        self.assertEqual(restored.weight_bytes, store.weight_bytes)
                        self.assertEqual(tuple(restored.layers), (0, 5))
                        archive = torch.jit.load(str(path), map_location="cpu")
                        self.assertEqual(archive.format_version.tolist(), [2])
                        self.assertEqual(set(dict(archive.named_buffers())), {
                            "format_version", "weight_version", "layer_ids",
                            "w1_0", "w2_0", "w1_5", "w2_5", "offload_config_0", "offload_config_5"})
                        for layer in (0, 5):
                            metadata = getattr(archive, f"offload_config_{layer}")
                            self.assertEqual(metadata.dtype, torch.int64)
                            self.assertEqual(metadata.device.type, "cpu")
                            self.assertEqual(tuple(metadata.tolist()), config.layer_config(layer))
                            for original, tensor in zip(store.layers[layer], restored.layers[layer]):
                                torch.testing.assert_close(original, tensor, atol=0, rtol=0)
                                self.assertTrue(tensor.is_contiguous())
                                self.assertFalse(tensor.is_inference() or tensor.is_pinned() or tensor.requires_grad)
                                self.assertNotEqual(original.untyped_storage()._cdata,
                                                    tensor.untyped_storage()._cdata)
                        # Returned metadata is deeply frozen and survives another archive roundtrip.
                        with self.assertRaises(FrozenInstanceError):
                            restored.deployment_config.target_sm = 80
                        restored.save_cpp(path, weight_version=18, config=restored.deployment_config)
                        self.assertEqual(ExpertStore.load_cpp(path).deployment_config, config)
                        with self.assertRaisesRegex(ValueError, "ram_budget"):
                            ExpertStore.load_cpp(path, ram_budget_bytes=1)
            load.assert_not_called()
            current_device.assert_not_called()
            capability.assert_not_called()

    def test_cpp_archive_deployment_format2_malformed(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        config = OffloadConfig.from_store(store, gpu_budget_bytes=200000,
            ram_budget_bytes=300000, pinned_budget_bytes=50000, target_sm=89,
            resident_cache_bytes=100000, resident_experts={0: [0, 2], 5: [1]})
        canonical = list(config.layer_config(0))
        cases = []
        for index, value in ((0, 2), (1, 0), (2, -1), (3, 0), (4, 3), (5, 3),
                             (6, -1), (7, -1), (8, -1), (9, 2), (10, 1), (11, 1),
                             (12, -1), (13, 2), (14, -1), (15, 16), (16, 32),
                             (17, 0), (18, 65), (19, 65), (20, 2), (21, -1),
                             (21, 1), (22, 79), (23, -1), (24, 3), (24, 0)):
            values = canonical.copy()
            values[index] = value
            cases.append((f"field_{index}_{value}", "offload_config_0",
                          torch.tensor(values, dtype=torch.int64)))
        cases.extend((
            ("short", "offload_config_0", torch.tensor(canonical[:22], dtype=torch.int64)),
            ("extra", "offload_config_0", torch.tensor(canonical + [1], dtype=torch.int64)),
            ("unsorted", "offload_config_0", torch.tensor(canonical[:23] + [2, 0], dtype=torch.int64)),
            ("float_metadata", "offload_config_0", torch.tensor(canonical, dtype=torch.float32)),
            ("matrix_metadata", "offload_config_0", torch.tensor(canonical, dtype=torch.int64)[None]),
            ("format1_with_config", "format_version", torch.tensor([1], dtype=torch.int64)),
        ))
        # Shape/dtype mismatches and common settings which are valid individually
        # must still be rejected rather than silently rewriting the archive contract.
        for index, value in ((1, 200001), (17, 4), (18, 128), (19, 128), (20, 1), (22, 120)):
            values = canonical.copy()
            values[index] = value
            cases.append((f"mismatch_{index}", "offload_config_0", torch.tensor(values, dtype=torch.int64)))
        with (tempfile.TemporaryDirectory() as temporary,
              mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.cuda, "get_device_capability") as capability):
            path = Path(temporary) / "malformed.pt"
            for name, field, value in cases:
                with self.subTest(case=name):
                    store.save_cpp(path, config=config)
                    archive = torch.jit.load(str(path), map_location="cpu")
                    setattr(archive, field, value)
                    torch.jit.save(archive, str(path))
                    with self.assertRaises(ValueError):
                        ExpertStore.load_cpp(path)
            # A format2 header without the required per-layer buffers is invalid.
            store.save_cpp(path)
            archive = torch.jit.load(str(path), map_location="cpu")
            archive.format_version = torch.tensor([2], dtype=torch.int64)
            torch.jit.save(archive, str(path))
            with self.assertRaisesRegex(ValueError, "buffers"):
                ExpertStore.load_cpp(path)
            for bad in ({}, (), object()):
                with self.subTest(config=bad), self.assertRaisesRegex(TypeError, "OffloadConfig"):
                    store.save_cpp(path, config=bad)
            load.assert_not_called()
            capability.assert_not_called()

    def test_cpp_archive_multilayer_roundtrip(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                generator = torch.Generator().manual_seed(20261012)
                state = {name: torch.randn(shape, generator=generator).transpose(1, 2)
                         for name, shape in (("fused", (3, 64, 128)), ("down0", (3, 64, 64)),
                                             ("gate5", (2, 128, 64)), ("up5", (2, 128, 64)),
                                             ("down5", (2, 64, 128)))}
                keys = {5: LayerKeys(gate="gate5", up="up5", down="down5"),
                        0: LayerKeys(gate_up="fused", down="down0")}
                checkpoint = Path(temporary) / "state_dict.pth"
                torch.save(state, checkpoint)
                store = ExpertStore.load(checkpoint, keys, dtype=dtype)
                again = ExpertStore.load(checkpoint, dict(reversed(list(keys.items()))), dtype=dtype)
                archive_path = Path(temporary) / "weights.pt"
                with torch.inference_mode():
                    store.save_cpp(archive_path, weight_version=17)
                    loaded = ExpertStore.load_cpp(archive_path)
                self.assertEqual(list(loaded.layers), [0, 5])
                self.assertEqual(loaded.weight_version, 17)
                self.assertEqual(loaded.weight_bytes, store.weight_bytes)
                scripted = torch.jit.load(str(archive_path), map_location="cpu")
                self.assertEqual(scripted.layer_ids.tolist(), [0, 5])
                self.assertEqual(scripted.format_version.tolist(), [1])
                for layer_id in (0, 5):
                    for original, repeated, restored in zip(
                            store.layers[layer_id], again.layers[layer_id], loaded.layers[layer_id]):
                        torch.testing.assert_close(original, repeated, atol=0, rtol=0)
                        torch.testing.assert_close(original, restored, atol=0, rtol=0)
                        self.assertEqual(restored.dtype, dtype)
                        self.assertTrue(restored.is_contiguous())
                        self.assertFalse(restored.is_pinned() or restored.requires_grad or restored.is_inference())
                        version = restored._version
                        restored.add_(0)
                        self.assertEqual(restored._version, version + 1)
                torch.testing.assert_close(store.layers[5][0],
                                           torch.cat((state["gate5"].to(dtype), state["up5"].to(dtype)), 1),
                                           atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "ram_budget"):
                    ExpertStore.load_cpp(archive_path, ram_budget_bytes=1)

    def test_cpp_archive_invalid_metadata(self):
        pair = (torch.zeros(2, 128, 64, dtype=torch.float16),
                torch.zeros(2, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.pt"
            for version in (-1, True, 1.5, 1 << 63):
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "weight_version"):
                    store.save_cpp(path, weight_version=version)
            for field, value, message in (
                    ("format_version", torch.tensor([3]), "format_version"),
                    ("format_version", torch.tensor([1.0]), "metadata"),
                    ("weight_version", torch.tensor([-1]), "weight_version"),
                    ("weight_version", torch.tensor([0, 1]), "weight_version"),
                    ("layer_ids", torch.tensor([5, 0]), "layer_ids"),
                    ("layer_ids", torch.tensor([0, 0]), "layer_ids"),
                    ("layer_ids", torch.tensor([-1, 5]), "layer_ids"),
                    ("layer_ids", torch.tensor([0, 7]), "buffers"),
                    ("layer_ids", torch.empty(0, dtype=torch.int64), "layer_ids")):
                store.save_cpp(path)
                archive = torch.jit.load(str(path))
                setattr(archive, field, value)
                torch.jit.save(archive, str(path))
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                    ExpertStore.load_cpp(path)
            with self.assertRaisesRegex(ValueError, "at least one layer"):
                ExpertStore({}).save_cpp(path)

    def test_default_modes_emit_v1_and_nondefault_modes_emit_v2(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair})
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=300000,
                       pinned_budget_bytes=50000, target_sm=89)
        config = OffloadConfig.from_store(store, **options)
        v1 = config.layer_config(0)
        self.assertEqual(v1[0], 1)
        self.assertEqual(len(v1), 23)
        self.assertEqual(v1[17:23], (3, 64, 64, 0, 0, 89))
        self.assertEqual(config.manifest(), (1, 1, 0, len(v1), *v1))
        self.assertEqual((config.expert_activation, config.numerical_mode),
                         ("silu", "fused"))
        # Explicit defaults are byte-identical to omission: v1 stays canonical.
        explicit = OffloadConfig.from_store(
            store, **options, expert_activation="silu", numerical_mode="fused")
        self.assertEqual(explicit, config)
        self.assertEqual(explicit.layer_config(0), v1)

        resident = dict(resident_cache_bytes=100000, resident_experts={0: [2, 0]})
        baseline = OffloadConfig.from_store(store, **options, **resident)
        baseline_v1 = baseline.layer_config(0)
        self.assertEqual(len(baseline_v1), 25)
        self.assertEqual(baseline_v1[21], 2)
        self.assertEqual(baseline_v1[23:], (0, 2))
        for expert_activation, numerical_mode, activation_code, numerical_code in (
                ("silu", "staged", 0, 1),
                ("gelu", "fused", 1, 0),
                ("gelu", "staged", 1, 1)):
            with self.subTest(expert_activation=expert_activation,
                              numerical_mode=numerical_mode):
                selected = OffloadConfig.from_store(
                    store, **options, **resident,
                    expert_activation=expert_activation, numerical_mode=numerical_mode)
                v2 = selected.layer_config(0)
                self.assertEqual(v2[0], 2)
                self.assertEqual(len(v2), 27)
                self.assertEqual(v2[1:23], baseline_v1[1:23])
                self.assertEqual(v2[23:25], (activation_code, numerical_code))
                self.assertEqual(v2[25:], (0, 2))
                self.assertEqual(selected.manifest(), (1, 1, 0, len(v2), *v2))
                # The outer manifest framing is always version 1.
                self.assertEqual(selected.manifest()[0], 1)
                self.assertEqual(selected.expert_activation, expert_activation)
                self.assertEqual(selected.numerical_mode, numerical_mode)
                options_dict = selected.runtime_options()
                self.assertEqual(options_dict["expert_activation"], expert_activation)
                self.assertEqual(options_dict["numerical_mode"], numerical_mode)
                self.assertEqual(OffloadConfig.from_layer_configs({0: v2}), selected)

    def test_v2_archive_roundtrip_and_manifest_framing(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.bfloat16),
                torch.zeros(3, 64, 64, dtype=torch.bfloat16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        config = OffloadConfig.from_store(
            store, gpu_budget_bytes=200000, ram_budget_bytes=300000,
            pinned_budget_bytes=50000, target_sm=89,
            resident_cache_bytes=200000, resident_experts={5: [2, 0]},
            expert_activation="gelu", numerical_mode="staged")
        vector = config.layer_config(5)
        self.assertEqual(vector[0], 2)
        self.assertEqual((len(vector), vector[23:25], vector[25:]), (27, (1, 1), (0, 2)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geglu.pt"
            store.save_cpp(path, weight_version=7, config=config)
            archive = torch.jit.load(str(path), map_location="cpu")
            buffers = dict(archive.named_buffers())
            self.assertEqual(buffers["format_version"].tolist(), [2])
            self.assertEqual(buffers["weight_version"].tolist(), [7])
            self.assertEqual(tuple(buffers["offload_config_5"].tolist()), vector)
            loaded = ExpertStore.load_cpp(path)
            self.assertEqual(loaded.weight_version, 7)
            self.assertIsInstance(loaded.deployment_config, OffloadConfig)
            self.assertEqual(loaded.deployment_config, config)
            self.assertEqual(loaded.deployment_config.layer_config(5), vector)
            self.assertEqual(loaded.deployment_config.manifest()[0], 1)
            self.assertEqual(loaded.deployment_config.manifest(),
                             (1, 2, 0, len(config.layer_config(0)), *config.layer_config(0),
                              5, len(vector), *vector))
            for expert_activation in ("silu", "gelu"):
                for numerical_mode in ("fused", "staged"):
                    other = OffloadConfig.from_store(
                        store, gpu_budget_bytes=200000, ram_budget_bytes=300000,
                        pinned_budget_bytes=50000, target_sm=89,
                        expert_activation=expert_activation,
                        numerical_mode=numerical_mode)
                    store.save_cpp(path, config=other)
                    self.assertEqual(ExpertStore.load_cpp(path).deployment_config, other)

    def test_v2_default_codes_are_noncanonical_in_python(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair})
        config = OffloadConfig.from_store(
            store, gpu_budget_bytes=200000, ram_budget_bytes=300000,
            pinned_budget_bytes=50000, target_sm=89,
            resident_cache_bytes=100000, resident_experts={0: [1]})
        v1 = list(config.layer_config(0))
        v2_defaults = (2, *v1[1:23], 0, 0, *v1[23:])
        self.assertEqual(len(v2_defaults), 26)
        # Python canonical form for defaults is v1, so a v2(0,0) record must be
        # rejected even though native validation accepts it.
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            OffloadConfig.from_layer_configs({0: v2_defaults})
        self.assertEqual(OffloadConfig.from_layer_configs({0: tuple(v1)}), config)

    def test_illegal_and_mixed_mode_schema_rejected_before_load(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        options = dict(gpu_budget_bytes=200000, ram_budget_bytes=300000,
                       pinned_budget_bytes=50000, target_sm=89)
        invalid_modes = {
            "expert_activation": ("", "GELU", "gelu ", "none", "silu/gelu",
                                  0, 1, True, False, None, [], {}),
            "numerical_mode": ("", "Staged", "fused ", "legacy", "staged/fused",
                               0, 1, True, False, None, [], {}),
        }
        native = mock.Mock()
        with (mock.patch.object(ops, "ensure_loaded") as load,
              mock.patch.object(torch.ops, "recsys_ops", native)):
            config = OffloadConfig.from_store(store, **options)
            for field, values in invalid_modes.items():
                for value in values:
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(
                            ValueError, field):
                        OffloadConfig.from_store(store, **{**options, field: value})
                    with self.subTest(replace=field, value=value), self.assertRaisesRegex(
                            ValueError, field):
                        replace(config, **{field: value})
                    with self.subTest(runtime=field, value=value), self.assertRaisesRegex(
                            ValueError, field):
                        PrefillMoERuntime(store, **{field: value})
            load.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()
            native.fuse_moe_offload_validate.assert_not_called()

            base = OffloadConfig.from_store(
                store, **options, resident_cache_bytes=100000,
                resident_experts={0: [0], 5: [1]})
            gelu = OffloadConfig.from_store(
                store, **options, resident_cache_bytes=100000,
                resident_experts={0: [0], 5: [1]},
                expert_activation="gelu", numerical_mode="fused")
            staged = OffloadConfig.from_store(
                store, **options, resident_cache_bytes=100000,
                resident_experts={0: [0], 5: [1]},
                expert_activation="silu", numerical_mode="staged")
            v1, v2_gelu = base.layer_config(0), gelu.layer_config(0)
            v2_staged = staged.layer_config(0)
            invalid = {
                "mixed versions": {0: tuple(v1), 5: tuple(v2_gelu)},
                "mixed activations": {0: tuple(v2_gelu),
                                      5: tuple((2, *v2_gelu[1:23], 0, 0, *v2_gelu[25:]))},
                "mixed numerical": {0: tuple(v2_staged),
                                    5: tuple((2, *v2_staged[1:23], 0, 0, *v2_staged[25:]))},
                "activation code 2": {0: tuple((2, *v2_gelu[1:23], 2, 0, *v2_gelu[25:]))},
                "numerical code 2": {0: tuple((2, *v2_gelu[1:23], 0, 2, *v2_gelu[25:]))},
                "negative code": {0: tuple((2, *v2_gelu[1:23], -1, 0, *v2_gelu[25:]))},
                "bool code": {0: tuple((2, *v2_gelu[1:23], True, 0, *v2_gelu[25:]))},
                "truncated v2": {0: tuple(v2_gelu[:-1])},
                "trailing v2": {0: tuple((*v2_gelu, 0))},
                "unsorted ids": {0: tuple((2, *v2_gelu[1:23], 1, 0, 1, 0))},
            }
            for label, configs in invalid.items():
                with self.subTest(label=label), self.assertRaises(ValueError):
                    OffloadConfig.from_layer_configs(configs)
            load.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()

    def test_mode_schema_before_native_load_and_v2_forward_contract(self):
        store = ExpertStore({0: (torch.zeros(3, 128, 64, dtype=torch.float16),
                                 torch.zeros(3, 64, 64, dtype=torch.float16))})
        config = OffloadConfig.from_store(
            store, gpu_budget_bytes=200000, ram_budget_bytes=300000,
            pinned_budget_bytes=50000, target_sm=89,
            expert_activation="gelu", numerical_mode="staged")
        self.assertEqual(config.layer_config(0)[-2:], (1, 1))
        native = mock.Mock()
        token = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_create.return_value = token
        x = torch.zeros(2, 128, dtype=torch.float16)
        ids = torch.zeros(2, 1, dtype=torch.int32)
        scale = torch.ones(2, 1)
        with (mock.patch.object(ops, "ensure_loaded"),
              mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native),
              mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9))):
            with PrefillMoERuntime.from_config(store, config, device="cuda:3") as runtime:
                native.fuse_moe_offload_create.assert_called_once_with(
                    3, 200000, 300000, 50000, 2, 0, 4096, 0, 0, 0, 32, 2, 0,
                    False, False, 0, 128, 1, 1, 0)
                native.fuse_moe_offload_validate.assert_called_once_with(
                    token, config.manifest())
                self.assertEqual(runtime._deployment_contracts[0],
                                 tuple(config.layer_config(0)))
                out = runtime.forward(x, ids, scale, layer_id=0)
                native.fuse_moe_offload.assert_called_once_with(
                    x, *store.layers[0], ids, scale, token, 0,
                    tuple(config.layer_config(0)), config.manifest())
                self.assertIs(out, native.fuse_moe_offload.return_value)
                adapter = OffloadedMoELayer(0, config=config)
                self.assertEqual(adapter.deployment_config,
                                 tuple(config.layer_config(0)))
                self.assertEqual(adapter.deployment_manifest, config.manifest())
            native.fuse_moe_offload_close.assert_called_once_with(token)

    def test_checkpoint(self):
        gate = torch.randn(4, 64, 64).half()
        up, down = torch.randn_like(gate), torch.randn_like(gate)
        state = {"g": gate, "u": up, "d": down}
        keys = {0: LayerKeys(gate="g", up="u", down="d")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state_dict.pth"
            torch.save(state, path)
            store = ExpertStore.load(path, keys)
            torch.testing.assert_close(store.layers[0][0], torch.cat((gate, up), 1))
            self.assertEqual(store.weight_bytes, 6 * 4 * 64 * 64)
        with self.assertRaises(ValueError):
            ExpertStore.from_state_dict({key: value.float() for key, value in state.items()}, keys)
        converted = ExpertStore.from_state_dict(
            {key: value.float() for key, value in state.items()}, keys, dtype=torch.bfloat16)
        self.assertEqual(converted.layers[0][0].dtype, torch.bfloat16)
        with self.assertRaises(ValueError):
            ExpertStore.from_state_dict(state, keys, ram_budget_bytes=1)
        with self.assertRaises(ValueError):
            ExpertStore.from_state_dict(state, {0: LayerKeys(down="d", gate="g", up="u", gate_up="g")})
        with torch.inference_mode():
            adapted = ExpertStore.from_state_dict(state, keys)
            self.assertFalse(adapted.layers[0][0].is_inference())
            inference_state = {key: value.clone() for key, value in state.items()}
            adapted = ExpertStore.from_state_dict(inference_state, keys)
            self.assertFalse(adapted.layers[0][1].is_inference())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OffloadTests(unittest.TestCase):
    default_async_staging = False

    def run(self, result=None):
        if not self.default_async_staging:
            return super().run(result)
        # Scope suite-wide opt-in to CUDA tests, leaving CPU default-schema
        # assertions and direct native create calls unchanged. Explicit kwargs
        # override partial defaults, including sync probes in RAM delta tests.
        original = PrefillMoERuntime
        with mock.patch(f"{__name__}.PrefillMoERuntime",
                        functools.partial(original, async_staging=True)):
            return super().run(result)

    @classmethod
    def setUpClass(cls):
        ops.ensure_loaded("fuse_moe")
        if not ops.is_available("fuse_moe"):
            raise RuntimeError(ops.load_error("fuse_moe"))
        torch.manual_seed(7)

    def tensors(self, s=129, h=64, i=64, e=8, k=4, dtype=torch.float16):
        x = (torch.randn(s, h, device="cuda") * .3).to(dtype)
        w1 = (torch.randn(e, 2 * i, h) * .05).to(dtype)
        w2 = (torch.randn(e, h, i) * .05).to(dtype)
        ids = torch.randint(e, (s, k), device="cuda", dtype=torch.int32)
        scale = torch.rand(s, k, device="cuda")
        return x, w1, w2, ids, scale

    def runtime(self, w1, w2, banks=2, capacity=1, **kwargs):
        e, _, h = w1.shape
        expert_bytes = 6 * h * w2.shape[2]
        pin = (e + 1) * 4 + banks * (capacity * expert_bytes + (4 * capacity + 2) * 4)
        return PrefillMoERuntime(ExpertStore({0: (w1, w2)}),
            gpu_budget_bytes=kwargs.pop("gpu_budget_bytes", 16 << 20),
            ram_budget_bytes=kwargs.pop("ram_budget_bytes", 64 << 20),
            pinned_budget_bytes=kwargs.pop("pinned_budget_bytes", pin), num_banks=banks, **kwargs)

    def test_deployment_native_validate_is_read_only_and_does_not_seal(self):
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        store = ExpertStore({0: pair, 5: tuple(w.clone() for w in pair)})
        device = torch.device("cuda", torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability(device)
        config = OffloadConfig.from_store(store, gpu_budget_bytes=16 << 20,
            ram_budget_bytes=64 << 20, pinned_budget_bytes=100000,
            target_sm=major * 10 + minor, num_banks=1)
        native = torch.ops.recsys_ops.fuse_moe_offload_validate
        # Ordinary graph-external schema: no token write alias or fake output needed.
        schema = native.default._schema
        self.assertEqual(len(schema.arguments), 2)
        self.assertEqual(str(schema.arguments[1].type), "List[int]")
        self.assertIsNone(schema.arguments[0].alias_info)
        self.assertEqual(len(schema.returns), 0)
        with moe_offload.PrefillMoERuntime.from_config(store, config, device=device) as runtime:
            before, token_value = runtime.stats(), runtime.token.clone()
            token_version = runtime.token._version
            weight_versions = tuple(w._version for pair in store.layers.values() for w in pair)
            bad_tokens = (torch.zeros(2, dtype=torch.int32),
                          torch.zeros(1, dtype=torch.int64),
                          torch.zeros(3, dtype=torch.int64),
                          torch.zeros(1, 2, dtype=torch.int64),
                          torch.zeros(2, 1, dtype=torch.int64),
                          torch.zeros(4, dtype=torch.int64)[::2])
            for bad_token in bad_tokens:
                with self.subTest(token_shape=bad_token.shape, token_stride=bad_token.stride(),
                                  token_dtype=bad_token.dtype):
                    torch.cuda.synchronize(device)
                    allocated = torch.cuda.memory_allocated(device)
                    with self.assertRaisesRegex(
                            RuntimeError, r"runtime_token must be contiguous CPU int64\[2\]"):
                        native(bad_token, config.manifest())
                    torch.cuda.synchronize(device)
                    self.assertEqual(torch.cuda.memory_allocated(device), allocated)
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            with self.assertRaisesRegex(RuntimeError, "runtime_token identity mismatch"):
                native(runtime.token.clone(), config.manifest())
            torch.cuda.synchronize(device)
            self.assertEqual(torch.cuda.memory_allocated(device), allocated)
            for invoke in (lambda: native(runtime.token, config.manifest()),
                           lambda: runtime.validate_config(config)):
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                result = invoke()
                torch.cuda.synchronize(device)
                self.assertEqual(torch.cuda.memory_allocated(device), allocated)
                self.assertIsNone(result)
                self.assertEqual(runtime.stats(), before)
                torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                self.assertEqual(runtime.token._version, token_version)
                self.assertEqual(runtime.token[1].item(), 0)
                self.assertEqual(tuple(w._version for pair in store.layers.values() for w in pair),
                                 weight_versions)
            # Both from_config's implicit preflight and repeated validation leave
            # bindings mutable; only an executing forward may seal the context.
            torch.ops.recsys_ops.fuse_moe_offload_bind(runtime.token, 6, *pair)
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            with self.assertRaisesRegex(RuntimeError, "manifest must include every bound layer"):
                runtime.validate_config(config)
            torch.cuda.synchronize(device)
            self.assertEqual(torch.cuda.memory_allocated(device), allocated)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            native(runtime.token, config.manifest())
        torch.cuda.synchronize(device)
        self.assertEqual(torch.cuda.memory_allocated(device), allocated)

    def test_deployment_manifest_checks_other_layer_before_execution_and_seals(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=3, k=1)
        ids.zero_()
        other = tuple(w.clone() for w in (w1, w2))
        store = ExpertStore({5: other, 0: (w1, w2)})
        major, minor = torch.cuda.get_device_capability(x.device)
        config = OffloadConfig.from_store(store, gpu_budget_bytes=16 << 20,
            ram_budget_bytes=64 << 20, pinned_budget_bytes=100000,
            target_sm=major * 10 + minor, num_banks=1,
            resident_cache_bytes=2 * 6 * 64 * 64, resident_experts={0: [0], 5: [1]})
        cfg0, cfg5 = config.layer_config(0), config.layer_config(5)
        record0, record5 = (0, len(cfg0), *cfg0), (5, len(cfg5), *cfg5)
        manifest = (1, 2, *record0, *record5)
        self.assertEqual(config.manifest(), manifest)
        altered5 = list(cfg5)
        altered5[23] = 2
        wrong_other_ids = (1, 2, *record0, 5, len(altered5), *altered5)
        altered5 = list(cfg5)
        altered5[18] = 128
        wrong_other_shape = (1, 2, *record0, 5, len(altered5), *altered5)
        malformed = (
            ("header", [1]), ("version", (2, *manifest[1:])),
            ("negative_count", (1, -1, *manifest[2:])),
            ("truncated_record", manifest[:-1]), ("trailing", (*manifest, 0)),
            ("record_order", (1, 2, *record5, *record0)),
            ("duplicate_layer", (1, 2, *record0, *record0)),
            ("negative_layer", (1, 2, -1, len(cfg0), *cfg0, *record5)),
            ("short_config", (1, 2, 0, 22, *cfg0[:22], *record5)),
            ("long_config", (1, 2, 0, (1 << 63) - 1, *cfg0, *record5)),
        )
        mismatches = [("other_static_ids", wrong_other_ids),
                      ("other_shape", wrong_other_shape),
                      ("missing_layer", (1, 1, *record0)),
                      ("unbound_layer", (1, 2, *record0, 6, len(cfg5), *cfg5))]
        for index, value in ((1, cfg5[1] + 1), (2, cfg5[2] + 1),
                             (3, cfg5[3] + 1), (4, 2), (17, 4), (19, 128), (20, 1)):
            altered5 = list(cfg5)
            altered5[index] = value
            mismatches.append((f"other_field_{index}",
                               (1, 2, *record0, 5, len(altered5), *altered5)))
        for index, value in ((1, 0), (2, -1), (3, 0), (4, 3), (18, 65), (19, 65)):
            altered5 = list(cfg5)
            altered5[index] = value
            malformed += ((f"other_invalid_field_{index}",
                           (1, 2, *record0, 5, len(altered5), *altered5)),)
        native = torch.ops.recsys_ops.fuse_moe_offload
        with moe_offload.PrefillMoERuntime.from_config(store, config, device=x.device) as runtime:
            adapter0 = OffloadedMoELayer(0, config=config)
            adapter5 = OffloadedMoELayer(5, config=config)
            self.assertEqual(adapter0.deployment_manifest, manifest)
            self.assertEqual(adapter5.deployment_manifest, manifest)
            first = None
            for sealed in (False, True):
                before, token_value = runtime.stats(), runtime.token.clone()
                token_version = runtime.token._version
                versions = tuple(w._version for pair in store.layers.values() for w in pair)
                # Drain assertion temporaries before each baseline; compare live
                # allocations before any tensor assertions can launch GPU work.
                torch.cuda.synchronize(x.device)
                allocated = torch.cuda.memory_allocated(x.device)
                result = runtime.validate_config(config)
                torch.cuda.synchronize(x.device)
                self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
                self.assertIsNone(result)
                self.assertEqual(runtime.stats(), before)
                torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                self.assertEqual(runtime.token._version, token_version)
                validate_malformed = (("empty", ()), ("zero_layers", (1, 0))) + malformed
                for cases, error in ((validate_malformed, "invalid deployment manifest format"),
                                     (mismatches, "deployment configuration mismatch")):
                    for name, bad_manifest in cases:
                        with self.subTest(validate=True, sealed=sealed, case=name):
                            torch.cuda.synchronize(x.device)
                            allocated = torch.cuda.memory_allocated(x.device)
                            with self.assertRaisesRegex(RuntimeError, error):
                                torch.ops.recsys_ops.fuse_moe_offload_validate(runtime.token, bad_manifest)
                            torch.cuda.synchronize(x.device)
                            self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
                            self.assertEqual(runtime.stats(), before)
                            torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                            self.assertEqual(runtime.token._version, token_version)
                            self.assertEqual(tuple(w._version for pair in store.layers.values() for w in pair),
                                             versions)
                            if first is not None:
                                torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
                for cases, error in ((malformed, "invalid deployment manifest format"),
                                     (mismatches, "deployment configuration mismatch")):
                    for name, bad_manifest in cases:
                        with self.subTest(sealed=sealed, case=name):
                            torch.cuda.synchronize(x.device)
                            allocated = torch.cuda.memory_allocated(x.device)
                            with self.assertRaisesRegex(RuntimeError, error):
                                native(x, w1, w2, ids, scale, runtime.token, 0, cfg0, bad_manifest)
                            torch.cuda.synchronize(x.device)
                            self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
                            self.assertEqual(runtime.stats(), before)
                            torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                            self.assertEqual(runtime.token._version, token_version)
                            self.assertEqual(tuple(w._version for pair in store.layers.values() for w in pair),
                                             versions)
                            if first is not None:
                                torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
                if not sealed:
                    # The adapter submits the current config and all-layer manifest
                    # as the two constant tails (nine native positional arguments).
                    first = adapter0(x, w1, w2, ids, scale, runtime.token)
                    snapshot = first.clone()
                    torch.testing.assert_close(first.float().cpu(), reference(x, w1, w2, ids, scale),
                                               atol=.002, rtol=.04)
            before, version, counter = runtime.stats(), runtime.token._version, runtime.token[1].item()
            # A valid manifest must not permit an empty per-layer config after sealing.
            torch.cuda.synchronize(x.device)
            allocated = torch.cuda.memory_allocated(x.device)
            with self.assertRaisesRegex(RuntimeError, "sealed|deployment configuration mismatch"):
                native(x, w1, w2, ids, scale, runtime.token, 0, [], manifest)
            torch.cuda.synchronize(x.device)
            self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
            self.assertEqual(runtime.stats(), before)
            self.assertEqual(runtime.token._version, version)
            self.assertEqual(runtime.token[1].item(), counter)
            # A matching per-layer config cannot downgrade a full-manifest seal
            # to standalone validation. Optional schema tails also accept explicit None.
            for changed_other_weight in (False, True):
                if changed_other_weight:
                    other[0].add_(0)
                before, token_value = runtime.stats(), runtime.token.clone()
                token_version = runtime.token._version
                versions = tuple(w._version for pair in store.layers.values() for w in pair)
                for tail_name, tail in (("omitted", ()), ("none", (None,)),
                                        ("empty_list", ([],)), ("empty_tuple", ((),))):
                    with self.subTest(changed_other_weight=changed_other_weight, manifest=tail_name):
                        torch.cuda.synchronize(x.device)
                        allocated = torch.cuda.memory_allocated(x.device)
                        with self.assertRaisesRegex(
                                RuntimeError, "sealed deployment context requires its compiled manifest"):
                            native(x, w1, w2, ids, scale, runtime.token, 0, cfg0, *tail)
                        torch.cuda.synchronize(x.device)
                        self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
                        self.assertEqual(runtime.stats(), before)
                        torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                        self.assertEqual(runtime.token._version, token_version)
                        self.assertEqual(tuple(w._version for pair in store.layers.values() for w in pair),
                                         versions)
                        torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
                if not changed_other_weight:
                    out = adapter5(x, *other, ids, scale, runtime.token)
                    torch.testing.assert_close(out.float().cpu(), reference(x, *other, ids, scale),
                                               atol=.002, rtol=.04)
                    del out
                else:
                    # Supplying the required manifest exposes the stale other-layer
                    # version, rather than allowing layer0 alone to execute.
                    torch.cuda.synchronize(x.device)
                    allocated = torch.cuda.memory_allocated(x.device)
                    with self.assertRaisesRegex(RuntimeError, "deployment configuration mismatch: weight version"):
                        native(x, w1, w2, ids, scale, runtime.token, 0, cfg0, manifest)
                    torch.cuda.synchronize(x.device)
                    self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
                    self.assertEqual(runtime.stats(), before)
                    torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                    self.assertEqual(runtime.token._version, token_version)
                    torch.testing.assert_close(first, snapshot, atol=0, rtol=0)

    def test_deployment_manifest_other_weight_version_rejected_before_first_forward(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=3, k=1)
        ids.zero_()
        major, minor = torch.cuda.get_device_capability(x.device)
        for changed_weight in (0, 1):
            other = tuple(w.clone() for w in (w1, w2))
            store = ExpertStore({0: (w1, w2), 5: other})
            config = OffloadConfig.from_store(store, gpu_budget_bytes=16 << 20,
                ram_budget_bytes=64 << 20, pinned_budget_bytes=100000,
                target_sm=major * 10 + minor, num_banks=1)
            with self.subTest(changed_weight=changed_weight), moe_offload.PrefillMoERuntime.from_config(
                    store, config, device=x.device) as runtime:
                # Shapes, dtype and storage identity still match; only layer5's
                # canonical version changes, before layer0 has ever executed.
                other[changed_weight].add_(0)
                before, token_value = runtime.stats(), runtime.token.clone()
                version = runtime.token._version
                weight_versions = tuple(w._version for pair in store.layers.values() for w in pair)
                for invoke in (lambda: runtime.validate_config(config),
                               lambda: torch.ops.recsys_ops.fuse_moe_offload_validate(
                                   runtime.token, config.manifest()),
                               lambda: runtime.forward(x, ids, scale),
                               lambda: OffloadedMoELayer(0, config=config)(
                                   x, w1, w2, ids, scale, runtime.token)):
                    torch.cuda.synchronize(x.device)
                    allocated = torch.cuda.memory_allocated(x.device)
                    with self.assertRaisesRegex(RuntimeError, "deployment configuration mismatch: weight version"):
                        invoke()
                    torch.cuda.synchronize(x.device)
                    self.assertEqual(torch.cuda.memory_allocated(x.device), allocated)
                    self.assertEqual(runtime.stats(), before)
                    torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                    self.assertEqual(runtime.token._version, version)
                    self.assertEqual(runtime.token[1].item(), 0)
                    self.assertEqual(tuple(w._version for pair in store.layers.values() for w in pair),
                                     weight_versions)
                # Preflight rejection must not seal or mutate bindings.
                torch.ops.recsys_ops.fuse_moe_offload_bind(runtime.token, 6, w1, w2)

    def test_deployment_config_native_rejection_is_before_mutation_and_seals_context(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=4, k=2)
        ids.copy_(torch.tensor([0, 1], device=x.device, dtype=torch.int32).expand_as(ids))
        store = ExpertStore({0: (w1, w2)})
        major, minor = torch.cuda.get_device_capability(x.device)
        target_sm = major * 10 + minor
        config = OffloadConfig.from_store(store, gpu_budget_bytes=16 << 20,
            ram_budget_bytes=64 << 20, pinned_budget_bytes=100000, target_sm=target_sm,
            num_banks=1, activation_mode="fp32_accumulate", max_route_tile_rows=7,
            resident_cache_bytes=2 * 6 * 64 * 64, resident_experts={0: [2, 0]},
            dynamic_cache_bytes=2 * 6 * 64 * 64, cache_policy="lru",
            cache_decay_interval=7, cache_min_reuse=3, cache_update_bytes=6 * 64 * 64,
            async_staging=False, host_profile=False, tile_m=32, tile_n=128)
        canonical = list(config.layer_config(0))
        mismatches = []
        for index, value in ((1, canonical[1] + 1), (2, canonical[2] + 1),
                             (3, canonical[3] + 1), (4, 2), (5, 0), (6, 8),
                             (7, canonical[7] + 1), (8, canonical[8] + 1), (9, 0),
                             (10, 8), (11, 4), (12, canonical[12] + 1), (13, 1),
                             (14, 1), (15, 64), (16, 64), (17, 5), (18, 128),
                             (19, 128), (20, 1),
                             (22, 120 if target_sm != 120 else 89), (23, 1), (24, 3)):
            altered = canonical.copy()
            altered[index] = value
            mismatches.append((f"field_{index}", altered))
        fewer_residents = canonical[:-1]
        fewer_residents[21] = 1
        mismatches.append(("field_21_resident_count", fewer_residents))
        malformed = [("version_only", [1]), ("short", canonical[:22]),
                     ("missing_ids", canonical[:23]), ("extra", canonical + [3])]
        for index, value in ((0, 0), (0, 2), (1, 0), (2, -1), (3, 0), (4, 3),
                             (5, 3), (6, -1), (6, 1 << 31), (7, -1), (8, -1),
                             (9, 2), (10, 1), (11, 1), (12, -1), (13, 2), (14, -1),
                             (15, 16), (16, 32), (17, 0), (17, 513), (18, 65),
                             (19, 65), (20, 2), (21, -1), (21, 5), (21, 1),
                             (22, 79), (23, -1), (24, 4), (24, 0)):
            altered = canonical.copy()
            altered[index] = value
            malformed.append((f"field_{index}_{value}", altered))
        malformed.append(("unsorted_ids", canonical[:23] + [2, 0]))
        native = torch.ops.recsys_ops.fuse_moe_offload
        # Use the real class: CLI constructor partials must not override a baked config.
        with moe_offload.PrefillMoERuntime.from_config(store, config, device=x.device) as runtime:
            def assert_unchanged(before, token_value, token_version, weight_versions):
                self.assertEqual(runtime.stats(), before)
                torch.testing.assert_close(runtime.token, token_value, atol=0, rtol=0)
                self.assertEqual(runtime.token._version, token_version)
                self.assertEqual((w1._version, w2._version), weight_versions)

            first = None
            for sealed in (False, True):
                before = runtime.stats()
                token_value = runtime.token.clone()
                token_version = runtime.token._version
                weight_versions = (w1._version, w2._version)
                for cases, error in ((mismatches, "deployment configuration mismatch"),
                                     (malformed, "invalid deployment configuration format")):
                    for name, altered in cases:
                        with self.subTest(sealed=sealed, case=name, error=error):
                            with self.assertRaisesRegex(RuntimeError, error):
                                native(x, w1, w2, ids, scale, runtime.token, 0, altered)
                            assert_unchanged(before, token_value, token_version, weight_versions)
                            if first is not None:
                                torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
                if not sealed:
                    # Rejected contracts do not seal an otherwise mutable context.
                    torch.ops.recsys_ops.fuse_moe_offload_bind(runtime.token, 5, w1, w2)
                    # This test intentionally binds an extra layer outside the
                    # config; exercise only the per-layer contract, not the manifest.
                    first = native(x, w1, w2, ids, scale, runtime.token, 0, canonical)
                    snapshot = first.clone()
                    torch.testing.assert_close(first.float().cpu(), reference(x, w1, w2, ids, scale),
                                               atol=.002, rtol=.04)
                    self.assertEqual(runtime.stats()["dynamic_experts"], 1)
            before = runtime.stats()
            token_value = runtime.token.clone()
            token_version = runtime.token._version
            weight_versions = (w1._version, w2._version)
            for tail_name, tail in (("omitted", ()), ("none", (None,)),
                                    ("empty_list", ([],)), ("empty_tuple", ((),))):
                with self.subTest(legacy_tail=tail_name), self.assertRaisesRegex(RuntimeError, "sealed"):
                    native(x, w1, w2, ids, scale, runtime.token, 0, *tail)
                assert_unchanged(before, token_value, token_version, weight_versions)
            for operation, args in (
                    (torch.ops.recsys_ops.fuse_moe_offload_bind, (runtime.token, 6, w1, w2)),
                    (torch.ops.recsys_ops.fuse_moe_offload_resident_bind, (runtime.token, 0, [])),
                    (torch.ops.recsys_ops.fuse_moe_offload_resident_bind, (runtime.token, 0, [3]))):
                with self.assertRaisesRegex(RuntimeError, "sealed"):
                    operation(*args)
                assert_unchanged(before, token_value, token_version, weight_versions)
            out = native(x, w1, w2, ids, scale, runtime.token, 0, canonical)
            torch.testing.assert_close(out, snapshot, atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
            torch.testing.assert_close(first, snapshot, atol=0, rtol=0)

    def test_deployment_config_dynamic_admissions_do_not_change_static_contract(self):
        expert_bytes = 6 * 64 * 64
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=7, e=4, k=2, dtype=dtype)
            store = ExpertStore({0: (w1, w2)})
            major, minor = torch.cuda.get_device_capability(x.device)
            for policy in ("reuse", "lru"):
                with self.subTest(dtype=dtype, policy=policy):
                    config = OffloadConfig.from_store(store, gpu_budget_bytes=16 << 20,
                        ram_budget_bytes=64 << 20, pinned_budget_bytes=100000,
                        target_sm=major * 10 + minor, num_banks=1,
                        resident_cache_bytes=expert_bytes, resident_experts={0: [0]},
                        dynamic_cache_bytes=expert_bytes, cache_policy=policy,
                        cache_decay_interval=1000)
                    contract = list(config.layer_config(0))
                    with moe_offload.PrefillMoERuntime.from_config(store, config, device=x.device) as runtime:
                        for expert in (1, 2):
                            ids.copy_(torch.tensor([0, expert], device=x.device,
                                                  dtype=torch.int32).expand_as(ids))
                            expected = reference(x, w1, w2, ids, scale)
                            # Reuse replacement must beat the previous key's
                            # four observations plus the quarter-expert byte margin.
                            for _ in range(3 if expert == 1 else 7):
                                out = runtime.forward(x, ids, scale)
                                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                                del out
                            stats = runtime.stats()
                            self.assertEqual(stats["resident_experts"], 1)
                            self.assertEqual(stats["dynamic_experts"], 1)
                            self.assertEqual(stats["dynamic_cache_hits"], 1)
                            self.assertEqual(stats["cache_hits"], 2)
                            self.assertEqual(stats["weight_h2d_bytes"], 0)
                            # Repeated validation must exclude dynamic residents,
                            # including after replacement of the previous dynamic key.
                            self.assertEqual(config.layer_config(0), tuple(contract))
                            out = torch.ops.recsys_ops.fuse_moe_offload(
                                x, w1, w2, ids, scale, runtime.token, 0, contract, config.manifest())
                            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                            del out
                        bad = contract + [2]
                        bad[21] = 2
                        before, version = runtime.stats(), runtime.token._version
                        counter = runtime.token[1].item()
                        with self.assertRaisesRegex(RuntimeError, "deployment configuration mismatch"):
                            torch.ops.recsys_ops.fuse_moe_offload(
                                x, w1, w2, ids, scale, runtime.token, 0, bad, config.manifest())
                        self.assertEqual(runtime.stats(), before)
                        self.assertEqual(runtime.token._version, version)
                        self.assertEqual(runtime.token[1].item(), counter)

    def test_async_staging_exact_one_latched_and_stats_parity(self):
        x, w1, w2, ids, scale = self.tensors(s=19, e=4, k=2)
        ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32).reshape_as(ids) % 4)
        expected = reference(x, w1, w2, ids, scale)
        prefix = "fuse_moe_offload::"
        for banks in (1, 2):
            sync_stats = None
            for enabled, setting in ((enabled, setting) for enabled in (False, True)
                                     for setting in (None, "", "0", "true", "01", "1 ", "1")):
                with self.subTest(banks=banks, async_staging=enabled, setting=setting), mock.patch.dict(
                        os.environ, {"FUSE_MOE_OFFLOAD_PROFILE": "0"}):
                    os.environ.pop("FUSE_MOE_OFFLOAD_ASYNC_STAGING", None)
                    if setting is not None:
                        os.environ["FUSE_MOE_OFFLOAD_ASYNC_STAGING"] = setting
                    with self.runtime(w1, w2, banks=banks, activation_mode="fp32_accumulate",
                                      max_route_tile_rows=7, async_staging=enabled,
                                      host_profile=True) as runtime:
                        # Environment changes cannot override explicit context configuration.
                        os.environ["FUSE_MOE_OFFLOAD_ASYNC_STAGING"] = "0" if enabled else "1"
                        os.environ["FUSE_MOE_OFFLOAD_PROFILE"] = "1"
                        for observation in range(2):
                            # Second observation exercises TLS propagation to an already running worker.
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                                out = runtime.forward(x, ids, scale)
                            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                            del out
                            stats = runtime.stats()
                            self.assertEqual(len(stats), 31)
                            stages = [event for event in profile.events() if event.name.startswith(prefix)]
                            counts = {name: sum(event.name == prefix + name for event in stages) for name in (
                                "host_weight_copy", "host_metadata", "bank_wait", "h2d_submit",
                                "compute_submit", "drain", "host_ready_wait", "host_bank_wait")}
                            self.assertEqual(counts, dict(host_weight_copy=4, host_metadata=4,
                                bank_wait=4 - banks, h2d_submit=4, compute_submit=stats["route_tiles"],
                                drain=1, host_ready_wait=4 if enabled else 0,
                                host_bank_wait=4 - banks if enabled else 0))
                            submit_threads = {event.thread for event in stages if event.name == prefix + "h2d_submit"}
                            copy_threads = {event.thread for event in stages if event.name == prefix + "host_weight_copy"}
                            self.assertEqual(len(submit_threads), 1)
                            self.assertEqual(len(copy_threads), 1)
                            if enabled:
                                self.assertTrue(copy_threads.isdisjoint(submit_threads))
                                self.assertEqual({event.thread for event in stages
                                    if event.name == prefix + "host_bank_wait"}, copy_threads)
                            else:
                                self.assertEqual(copy_threads, submit_threads)
                            if observation == 0:
                                if sync_stats is None:
                                    sync_stats = stats
                                for key in stats:
                                    if key not in ("ram_charged_bytes", "buffer_allocations", "buffer_reuses"):
                                        self.assertEqual(stats[key], sync_stats[key], key)
                                if enabled:
                                    self.assertGreater(stats["ram_charged_bytes"] - sync_stats["ram_charged_bytes"], 1 << 20)
                                else:
                                    self.assertEqual(stats["ram_charged_bytes"], sync_stats["ram_charged_bytes"])

    def test_async_staging_lazy_start_and_thread_cleanup(self):
        tasks = Path("/proc/self/task")
        if not tasks.is_dir():
            self.skipTest("Linux task enumeration required")
        def workers():
            result = set()
            for task in tasks.iterdir():
                try:
                    if (task / "comm").read_text().strip() == "moe_staging":
                        result.add(task.name)
                except FileNotFoundError:
                    pass  # Kernel removal can race task enumeration.
            return result

        x, w1, w2, ids, scale = self.tensors(s=19, e=4, k=2)
        ids.zero_()
        masked = torch.full_like(ids, -1)
        # Warm CUDA and profiler support before inspecting native worker lifetime.
        with mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_PROFILE": "0",
                                          "FUSE_MOE_OFFLOAD_ASYNC_STAGING": "1"}):
            with self.runtime(w1, w2, async_staging=False, host_profile=True) as warm:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
                    out = warm.forward(x, ids, scale)
                del out
        with mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_PROFILE": "0",
                                          "FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0"}):
            for _ in range(3):
                before = workers()
                with self.runtime(w1, w2, resident_cache_bytes=6 * 64 * 64,
                                  resident_experts={0: [0]}, async_staging=True,
                                  host_profile=True) as runtime:
                    # Create/bind, empty routes and all-cache-hit calls must not start staging.
                    for route in (masked, ids):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                            out = runtime.forward(x, route, scale)
                        del out
                        names = [event.name for event in profile.events()]
                        self.assertNotIn("fuse_moe_offload::host_ready_wait", names)
                        self.assertNotIn("fuse_moe_offload::host_weight_copy", names)
                    idle_tasks = workers()
                    self.assertEqual(idle_tasks, before)
                    # Identify the actual worker, not incidental profiler threads.
                    missing = torch.full_like(ids, 1)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        out = runtime.forward(x, missing, scale)
                    del out
                    self.assertIn("fuse_moe_offload::host_ready_wait", [event.name for event in profile.events()])
                    started = workers() - idle_tasks
                    self.assertTrue(started, "first miss must lazily create a persistent staging worker")
                    runtime.close()
                    runtime.close()
                # pthread_join guarantees termination, but /proc task removal
                # visibility can trail it. Bound observation; never accept a leak.
                deadline = time.monotonic() + 1.0
                while started & workers() and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertFalse(started & workers(), "close must join the staging worker")
                self.assertEqual(workers(), before)

    def test_async_staging_worker_ram_budget_refusal_and_retry(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=2, k=1)
        ids.zero_()
        weight_bytes = sum(w.numel() * w.element_size() for w in (w1, w2))
        # Fits synchronous backing/workspace, but not the >1 MiB worker reservation.
        budget = weight_bytes + (1 << 20)
        with mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_ASYNC_STAGING": "1"}):
            with self.runtime(w1, w2, ram_budget_bytes=budget, async_staging=False) as runtime:
                out = runtime.forward(x, ids, scale)
                self.assertLessEqual(runtime.stats()["ram_charged_bytes"], budget)
                del out
        with mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0"}):
            with self.assertRaisesRegex(RuntimeError, "ram_budget|RAM|budgets"):
                with self.runtime(w1, w2, ram_budget_bytes=budget, async_staging=True) as runtime:
                    runtime.forward(x, ids, scale)
            with self.runtime(w1, w2, async_staging=True) as runtime:
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale), atol=.002, rtol=.04)

    def test_async_staging_stream_layer_cache_recovery_and_new_weight_version(self):
        original = PrefillMoERuntime
        with (mock.patch(f"{__name__}.PrefillMoERuntime",
                         functools.partial(original, async_staging=True)),
              mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0"})):
            # Direct Python constructors opt in too; direct native create calls
            # retain their synchronous default.
            # Reuse established assertions without weakening their original budgets.
            self.test_dynamic_cross_stream_grouped_wave_promotion_and_live_output()
            self.test_route_modes_and_layer_sharing()
            self.test_validation_and_recovery()
            self.test_dynamic_existing_identity_version_and_route_validation()
            x, w1, w2, ids, scale = self.tensors(s=7, e=2, k=1)
            ids.zero_()
            with self.runtime(w1, w2) as runtime:
                out = runtime.forward(x, ids, scale)
                del out
                w1.add_(.01)
                with self.assertRaisesRegex(RuntimeError, "version"):
                    runtime.forward(x, ids, scale)
            with self.runtime(w1, w2) as replacement:
                out = replacement.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale), atol=.002, rtol=.04)

    def test_host_stage_profile_opt_in_latched_and_stats_unchanged(self):
        x, w1, w2, ids, scale = self.tensors(s=19, e=4, k=2)
        ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32).reshape_as(ids) % 4)
        expected = reference(x, w1, w2, ids, scale)
        prefix = "fuse_moe_offload::"
        for banks in (1, 2):
            for async_staging, host_profile, setting in (
                    (async_staging, host_profile, setting)
                    for async_staging in (False, True) for host_profile in (False, True)
                    for setting in (None, "0", "true", "1")):
                with self.subTest(banks=banks, async_staging=async_staging,
                                  host_profile=host_profile, setting=setting), mock.patch.dict(os.environ):
                    os.environ["FUSE_MOE_OFFLOAD_ASYNC_STAGING"] = "0" if async_staging else "1"
                    os.environ.pop("FUSE_MOE_OFFLOAD_PROFILE", None)
                    if setting is not None:
                        os.environ["FUSE_MOE_OFFLOAD_PROFILE"] = setting
                    with self.runtime(w1, w2, banks=banks, activation_mode="fp32_accumulate",
                                      max_route_tile_rows=7, async_staging=async_staging,
                                      host_profile=host_profile) as runtime:
                        warm = runtime.forward(x, ids, scale)
                        del warm
                        before = runtime.stats()
                        # Neither environment flag can mutate explicit configuration in forward.
                        os.environ["FUSE_MOE_OFFLOAD_PROFILE"] = "0" if host_profile else "1"
                        os.environ["FUSE_MOE_OFFLOAD_ASYNC_STAGING"] = "1" if async_staging else "0"
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                            out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        after = runtime.stats()
                        self.assertEqual(len(after), 31)
                        for key in before:
                            if key in ("buffer_allocations", "buffer_reuses"):
                                continue
                            self.assertEqual(before[key], after[key], key)
                        stages = [event for event in profile.events() if event.name.startswith(prefix)]
                        if not host_profile:
                            self.assertEqual(stages, [])
                            continue
                        counts = {name: sum(event.name == prefix + name for event in stages) for name in (
                            "host_weight_copy", "host_metadata", "bank_wait", "h2d_submit",
                            "compute_submit", "drain", "host_ready_wait", "host_bank_wait")}
                        self.assertEqual(counts, dict(host_weight_copy=4, host_metadata=4,
                            bank_wait=4 - banks, h2d_submit=4,
                            compute_submit=after["route_tiles"], drain=1,
                            host_ready_wait=4 if async_staging else 0,
                            host_bank_wait=4 - banks if async_staging else 0))
                        self.assertTrue(all(event.cpu_time_total >= 0 for event in stages))

    def test_host_stage_profile_cache_empty_invalid_and_cross_stream(self):
        x, w1, w2, ids, scale = self.tensors(s=17, e=4, k=2)
        ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32).reshape_as(ids) % 4)
        expected = reference(x, w1, w2, ids, scale)
        producer = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(producer)
        with mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_PROFILE": "0"}):
            for mode in ("route_buffer", "fp32_accumulate"):
                with self.subTest(mode=mode), self.runtime(w1, w2, activation_mode=mode,
                        dynamic_cache_bytes=4 * 6 * 64 * 64, host_profile=True) as runtime:
                    # Enabled without a profiler must still support miss/promotion.
                    for _ in range(3):
                        with torch.cuda.stream(stream):
                            output = runtime.forward(x, ids, scale)
                        del output
                    self.assertEqual(runtime.stats()["cache_hits"], 4)
                    for kind in ("hits", "masked", "zero", "invalid", "recovery"):
                        route = ids if kind in ("hits", "recovery") else torch.full_like(
                            ids, 4 if kind == "invalid" else -1)
                        rows = 0 if kind == "zero" else x.shape[0]
                        before = runtime.stats()
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                            with torch.cuda.stream(stream):
                                if kind == "invalid":
                                    with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                                        runtime.forward(x[:rows], route[:rows], scale[:rows])
                                else:
                                    output = runtime.forward(x[:rows], route[:rows], scale[:rows])
                        stages = [event.name for event in profile.events()
                                  if event.name.startswith("fuse_moe_offload::")]
                        for name in ("host_weight_copy", "host_metadata", "bank_wait", "h2d_submit",
                                     "host_ready_wait", "host_bank_wait"):
                            self.assertNotIn("fuse_moe_offload::" + name, stages)
                        if kind == "invalid":
                            self.assertEqual(before, runtime.stats())
                            self.assertEqual(stages, [])
                        else:
                            target = expected if kind in ("hits", "recovery") else torch.zeros_like(expected[:rows])
                            torch.testing.assert_close(output.float().cpu(), target, atol=.002, rtol=.04)
                            self.assertEqual(stages.count("fuse_moe_offload::drain"), 1)
                            self.assertEqual(stages.count("fuse_moe_offload::compute_submit"),
                                             runtime.stats()["route_tiles"])
                            del output

    def test_host_stage_profile_nvtx_modes(self):
        x, w1, w2, ids, scale = self.tensors(s=19, e=4, k=2)
        expected = reference(x, w1, w2, ids, scale)
        with mock.patch.dict(os.environ, {"FUSE_MOE_OFFLOAD_PROFILE": "0"}):
            with self.runtime(w1, w2, host_profile=True) as runtime:
                # Direct NVTX is compatible with CPU profiling, but emit_nvtx
                # uses the profiler state and must be a separate observation.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    with torch.cuda.nvtx.range("offload_host_stage_direct_nvtx"):
                        out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                self.assertIn("fuse_moe_offload::host_weight_copy", [event.name for event in profile.events()])
                del out
                with torch.autograd.profiler.emit_nvtx():
                    out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)

    def test_host_weight_runs_holes_partial_wave_and_cache_boundaries(self):
        active = [0, 1, 4, 5, 6, 9, 10, 13, 14, 15]
        expert_bytes = 6 * 64 * 64
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=37, e=16, k=3, dtype=dtype)
            order = torch.tensor(active[::-1], device=x.device, dtype=torch.int32)
            ids.copy_(order[torch.arange(ids.numel(), device=x.device).reshape_as(ids) % len(active)])
            ids[::7, -1] = -1
            scale.copy_(torch.linspace(-1, 1, scale.numel(), device=x.device).reshape_as(scale))
            expected = reference(x, w1, w2, ids, scale)
            for banks in (1, 2):
                for mode in ("route_buffer", "fp32_accumulate"):
                    for cached in ((), (1, 9)):
                        with self.subTest(dtype=dtype, banks=banks, mode=mode, cached=cached):
                            with self.runtime(w1, w2, banks=banks, capacity=4,
                                    activation_mode=mode, max_route_tile_rows=7,
                                    resident_cache_bytes=len(cached) * expert_bytes,
                                    resident_experts={0: list(cached)}) as runtime:
                                for routes in (ids, torch.full_like(ids, -1), ids):
                                    out = runtime.forward(x, routes, scale)
                                    target = expected if routes is ids else torch.zeros_like(expected)
                                    torch.testing.assert_close(out.float().cpu(), target, atol=.002, rtol=.04)
                                    stats = runtime.stats()
                                    if routes is ids:
                                        self.assertEqual(stats["cache_hits"], len(cached))
                                        self.assertEqual(stats["weight_h2d_bytes"], (len(active) - len(cached)) * expert_bytes)
                                        self.assertEqual(stats["wave_capacity"], 4)
                                        self.assertEqual(stats["num_banks"], banks)
                                    else:
                                        self.assertEqual(stats["weight_h2d_bytes"], 0)
                                    del out

    def test_adaptive_native_tile_boundaries(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=2, k=1)
        ids.zero_()
        expected = reference(x, w1, w2, ids, scale)
        create = torch.ops.recsys_ops.fuse_moe_offload_create
        device = torch.cuda.current_device()
        for rows in (-1, -(1 << 31), 1 << 31, 1 << 32):
            with self.subTest(rows=rows), self.assertRaisesRegex(RuntimeError, "max_route_tile_rows"):
                create(device, 16 << 20, 64 << 20, 50000, 1, 1, rows)
        for mode in (-1, 3):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "activation_mode"):
                create(device, 16 << 20, 64 << 20, 50000, 1, mode, 0)
        for mode in (0, 1, 2):
            for rows in (0, 1, (1 << 31) - 1):
                with self.subTest(mode=mode, rows=rows):
                    token = create(device, 16 << 20, 64 << 20, 50000, 1, mode, rows)
                    try:
                        torch.ops.recsys_ops.fuse_moe_offload_bind(token, 0, w1, w2)
                        out = torch.ops.recsys_ops.fuse_moe_offload(x, w1, w2, ids, scale, token, 0)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                    finally:
                        torch.ops.recsys_ops.fuse_moe_offload_close(token)

    def test_adaptive_long_hot_routes_exceed_default_cap(self):
        # Long S has only H=I=64 and one expert; no expensive long CPU oracle.
        x, w1, w2, ids, scale = self.tensors(s=4097, e=2, k=2)
        ids.zero_()
        scale.copy_(torch.linspace(-1, 1, scale.numel(), device=x.device).reshape_as(scale))
        ids[0] = -1
        scale[0] = float("nan")
        total = (x.shape[0] - 1) * 2
        with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                          max_route_tile_rows=4096) as fixed:
            baseline = fixed.forward(x, ids, scale)
            capped = fixed.stats()
        with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                          max_route_tile_rows="auto") as runtime:
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
            self.assertTrue(torch.isfinite(out).all().item())
            torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=0, rtol=0)
            stats = runtime.stats()
            self.assertEqual(stats["route_tile_rows"], total)
            self.assertGreater(stats["route_tile_rows"], 4096)
            self.assertEqual(stats["route_tiles"], 1)
            self.assertEqual(capped["route_tiles"], 2)
            del out
            out = runtime.forward(x[:19], ids[:19], scale[:19])
            torch.testing.assert_close(out.float().cpu(), reference(x[:19], w1, w2, ids[:19], scale[:19]),
                                       atol=.002, rtol=.04)

    def test_adaptive_plateau_minimum_rows_preserves_each_wave_tiles(self):
        x, w1, w2, ids, scale = self.tensors(s=1001, e=2, k=2)
        ids[:, 0] = 0
        ids[:, 1] = 1
        ids[601:, 1] = -1
        # Miss waves have 1001 and 601 rows. At best=800 their tile counts
        # are [2,1], so minimum=601, aligned to 608 for forced tile_m=32.
        with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "128", "FUSE_MOE_TILE_N": "64"}):
            with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                              max_route_tile_rows=800, tile_m=32) as probe:
                baseline = probe.forward(x, ids, scale)
                budget = probe.stats()["gpu_charged_bytes"]
            del probe
            results = {}
            for cap in ((1 << 31) - 1, "auto"):
                with self.subTest(cap=cap), self.runtime(
                        w1, w2, banks=1, gpu_budget_bytes=budget,
                        activation_mode="fp32_accumulate", max_route_tile_rows=cap, tile_m=32) as runtime:
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
                    stats = runtime.stats()
                    results[cap] = stats
                    self.assertEqual(stats["waves"], 2)
                    self.assertEqual(stats["route_tiles"], 3)
                    self.assertEqual(stats["route_tile_rows"], 608 if cap == "auto" else 800)
                    self.assertEqual([(rows + stats["route_tile_rows"] - 1) // stats["route_tile_rows"]
                                      for rows in (1001, 601)], [2, 1])
                    self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                    del out
            self.assertLess(results["auto"]["gpu_pool_bytes"], results[(1 << 31) - 1]["gpu_pool_bytes"])
            self.assertLess(results["auto"]["gpu_charged_bytes"],
                            results[(1 << 31) - 1]["gpu_charged_bytes"])
        # If rounding up crosses best, use the exact minimum, not tile_m.
        with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "32", "FUSE_MOE_TILE_N": "64"}):
            with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                              max_route_tile_rows=604, tile_m=128) as probe:
                out = probe.forward(x, ids, scale)
                budget = probe.stats()["gpu_charged_bytes"]
                del out
            with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                              activation_mode="fp32_accumulate", max_route_tile_rows="auto",
                              tile_m=128) as runtime:
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
                self.assertEqual(runtime.stats()["route_tile_rows"], 601)
                self.assertEqual(runtime.stats()["route_tiles"], 3)

    def test_adaptive_highwater_live_outputs_trim_and_recover(self):
        x, w1, w2, ids, scale = self.tensors(s=257, e=2, k=4)
        ids.zero_()
        with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                          max_route_tile_rows="auto") as probe:
            out = probe.forward(x, ids, scale)
            budget = probe.stats()["gpu_charged_bytes"]
            del out
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                          activation_mode="fp32_accumulate", max_route_tile_rows="auto") as runtime:
            first = runtime.forward(x, ids, scale)
            snapshot = first.clone()
            highwater = runtime.stats()["gpu_pool_bytes"]
            small = runtime.forward(x[:17], ids[:17], scale[:17])
            torch.testing.assert_close(small.float().cpu(), reference(x[:17], w1, w2, ids[:17], scale[:17]),
                                       atol=.002, rtol=.04)
            self.assertLess(runtime.stats()["gpu_pool_bytes"], highwater)
            self.assertLessEqual(runtime.stats()["gpu_charged_bytes"], budget)
            torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
            del first, small
            gc.collect()
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out, snapshot, atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["route_tiles"], 1)
            runtime.close()
            torch.testing.assert_close(out, snapshot, atol=.002, rtol=.04)

    def test_adaptive_dynamic_eligible_reservation_and_highwater_exact_fallback(self):
        x, w1, w2, ids, scale = self.tensors(s=1001, e=2, k=2)
        ids.zero_()
        expert_bytes = 6 * 64 * 64
        with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "128", "FUSE_MOE_TILE_N": "64"}):
            with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                              max_route_tile_rows=800, tile_m=32, dynamic_cache_bytes=expert_bytes,
                              cache_min_reuse=1000) as probe:
                baseline = probe.forward(x, ids, scale)
                budget = probe.stats()["gpu_charged_bytes"]
            # Without reservation best=800 -> three tiles -> minimum=668 ->672.
            # Reserving one expert leaves best=608 ->four tiles ->minimum=501 ->512.
            configs = (("reuse", expert_bytes, 0, 2, True),
                       ("lru", expert_bytes, 0, 2, True),
                       ("reuse", expert_bytes - 1, 0, 2, False),
                       ("reuse", expert_bytes, expert_bytes - 1, 2, False),
                       ("reuse", expert_bytes, 0, 1000, False))
            for policy, cap, update, minimum, eligible in configs:
                with self.subTest(policy=policy, cap=cap, update=update, minimum=minimum), self.runtime(
                        w1, w2, banks=1, gpu_budget_bytes=budget,
                        activation_mode="fp32_accumulate", max_route_tile_rows="auto",
                        dynamic_cache_bytes=cap, cache_policy=policy, tile_m=32,
                        cache_update_bytes=update, cache_min_reuse=minimum) as runtime:
                    if policy == "reuse":
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
                        cold = runtime.stats()
                        self.assertEqual(cold["route_tile_rows"], 672)
                        self.assertEqual(cold["cache_admissions"], 0)
                        del out
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(stats["route_tile_rows"], 512 if eligible else 672)
                    self.assertEqual(stats["route_tiles"], 4 if eligible else 3)
                    self.assertEqual(stats["num_banks"], 1)
                    self.assertEqual(stats["wave_capacity"], 1)
                    self.assertEqual(stats["cache_admissions"], int(eligible))
                    self.assertEqual(stats["cache_bypasses"], int(not eligible))
                    self.assertEqual(stats["dynamic_experts"], int(eligible))
                    self.assertEqual(stats["cache_promotion_d2d_bytes"], int(eligible) * expert_bytes)
                    self.assertEqual(stats["weight_h2d_bytes"], expert_bytes)
                    self.assertEqual(stats["cache_evictions"], 0)
                    self.assertEqual(stats["cache_budget_evictions"], 0)
                    self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                    if policy == "reuse" and eligible:
                        # The old 672-row pool cannot fit alongside the reserved
                        # expert. configure must retry exact-required capacities.
                        self.assertLess(stats["gpu_pool_bytes"], cold["gpu_pool_bytes"])
                    del out
                    if eligible:
                        out = runtime.forward(x[:19], ids[:19], scale[:19])
                        torch.testing.assert_close(out.float().cpu(),
                            reference(x[:19], w1, w2, ids[:19], scale[:19]), atol=.002, rtol=.04)
                        self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
                        self.assertEqual(runtime.stats()["weight_h2d_bytes"], 0)
                        del out

    def test_adaptive_reservation_floor_never_sacrifices_banks_or_capacity(self):
        x, w1, w2, ids, scale = self.tensors(s=37, e=4, k=2)
        ids.copy_(torch.arange(2, device=x.device, dtype=torch.int32).expand_as(ids))
        expected = reference(x, w1, w2, ids, scale)
        expert_bytes = 6 * 64 * 64
        with self.runtime(w1, w2, banks=2, capacity=2, activation_mode="fp32_accumulate",
                          max_route_tile_rows=1, dynamic_cache_bytes=expert_bytes,
                          cache_min_reuse=1000) as probe:
            out = probe.forward(x, ids, scale)
            budget = probe.stats()["gpu_charged_bytes"]
            del out
        for policy in ("reuse", "lru"):
            plans = {}
            for cap in ((1 << 31) - 1, "auto"):
                with self.subTest(policy=policy, cap=cap), self.runtime(
                        w1, w2, banks=2, capacity=2, gpu_budget_bytes=budget,
                        activation_mode="fp32_accumulate", max_route_tile_rows=cap,
                        dynamic_cache_bytes=expert_bytes, cache_policy=policy) as runtime:
                    for _ in range(2):
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        self.assertEqual(stats["num_banks"], 2)
                        self.assertEqual(stats["wave_capacity"], 2)
                        self.assertEqual(stats["dynamic_experts"], 0)
                        self.assertEqual(stats["cache_admissions"], 0)
                        self.assertEqual(stats["cache_bypasses"], 2)
                        self.assertEqual(stats["cache_evictions"], 0)
                        self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                        del out
                    plans[cap] = stats
            for key in ("num_banks", "wave_capacity", "route_tile_rows", "route_tiles", "gpu_pool_bytes"):
                self.assertEqual(plans["auto"][key], plans[(1 << 31) - 1][key], key)

    def test_adaptive_full_cache_leased_hit_not_evicted_for_larger_tile(self):
        x, w1, w2, ids, scale = self.tensors(s=1001, e=2, k=2)
        ids.copy_(torch.arange(2, device=x.device, dtype=torch.int32).expand_as(ids))
        expert_bytes = 6 * 64 * 64
        with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "128", "FUSE_MOE_TILE_N": "64"}):
            with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                              max_route_tile_rows=800, tile_m=32, resident_cache_bytes=expert_bytes,
                              resident_experts={0: [1]}, dynamic_cache_bytes=expert_bytes,
                              cache_min_reuse=1000) as probe:
                baseline = probe.forward(x, ids, scale)
                budget = probe.stats()["gpu_charged_bytes"]
            for policy, cap in ((policy, cap) for policy in ("reuse", "lru")
                                for cap in (expert_bytes, 2 * expert_bytes - 1)):
                with self.subTest(policy=policy, cap=cap), self.runtime(
                        w1, w2, banks=1, gpu_budget_bytes=budget,
                        activation_mode="fp32_accumulate", max_route_tile_rows="auto",
                        dynamic_cache_bytes=cap, cache_policy=policy, tile_m=32) as runtime:
                    # Full cap or a nonfull cap with less than one expert left:
                    # neither permits reservation for the leading miss.
                    hot_ids = torch.ones_like(ids[:7])
                    for _ in range(3):
                        out = runtime.forward(x[:7], hot_ids, scale[:7])
                        del out
                    self.assertEqual(runtime.stats()["dynamic_experts"], 1)
                    for _ in range(2):
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        self.assertEqual(stats["route_tile_rows"], 512)
                        self.assertEqual(stats["route_tiles"], 4)
                        self.assertEqual(stats["dynamic_cache_hits"], 1)
                        self.assertEqual(stats["dynamic_experts"], 1)
                        self.assertEqual(stats["dynamic_cache_bytes"], expert_bytes)
                        self.assertEqual(stats["weight_h2d_bytes"], expert_bytes)
                        self.assertEqual(stats["cache_bypasses"], 1)
                        for key in ("cache_admissions", "cache_evictions", "cache_budget_evictions"):
                            self.assertEqual(stats[key], 0, key)
                        del out

    def test_adaptive_inactive_full_cache_shrinks_tile_before_budget_eviction(self):
        x, w1, w2, ids, scale = self.tensors(s=1001, e=3, k=2)
        ids.zero_()
        expert_bytes = 6 * 64 * 64
        with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "128", "FUSE_MOE_TILE_N": "64"}):
            with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                              max_route_tile_rows=800, tile_m=32, dynamic_cache_bytes=expert_bytes,
                              cache_min_reuse=1000) as probe:
                baseline = probe.forward(x, ids, scale)
                budget = probe.stats()["gpu_charged_bytes"]
            for policy in ("reuse",):
                with self.subTest(policy=policy), self.runtime(
                        w1, w2, banks=1, gpu_budget_bytes=budget,
                        activation_mode="fp32_accumulate", max_route_tile_rows="auto",
                        dynamic_cache_bytes=expert_bytes, cache_policy=policy, tile_m=32) as runtime:
                    hot_ids = torch.full_like(ids[:7], 2)
                    for _ in range(3):
                        out = runtime.forward(x[:7], hot_ids, scale[:7])
                        del out
                    self.assertEqual(runtime.stats()["dynamic_experts"], 1)
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out, baseline, atol=.002, rtol=.04)
                    stats = runtime.stats()
                    # Keeping inactive expert 2 leaves best=608 and minimum=512.
                    # Evicting it merely to recover best=800 is forbidden.
                    # Reuse sees only frequency one for the new candidate, so
                    # admission cannot disguise a planner budget eviction.
                    self.assertEqual(stats["route_tile_rows"], 512)
                    self.assertEqual(stats["route_tiles"], 4)
                    self.assertEqual(stats["dynamic_experts"], 1)
                    self.assertEqual(stats["dynamic_cache_bytes"], expert_bytes)
                    self.assertEqual(stats["cache_bypasses"], 1)
                    for key in ("cache_admissions", "cache_evictions", "cache_budget_evictions"):
                        self.assertEqual(stats[key], 0, key)
                    del out
                    out = runtime.forward(x[:7], hot_ids, scale[:7])
                    torch.testing.assert_close(out.float().cpu(),
                        reference(x[:7], w1, w2, hot_ids, scale[:7]), atol=.002, rtol=.04)
                    self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
                    self.assertEqual(runtime.stats()["weight_h2d_bytes"], 0)

    def test_adaptive_invalid_routes_preserve_heat_clock_and_live_outputs(self):
        x, w1, w2, ids, scale = self.tensors(s=19, e=3, k=2)
        ids.fill_(1)
        expert_bytes = 6 * 64 * 64
        expected = reference(x, w1, w2, ids, scale)
        with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                          max_route_tile_rows="auto", dynamic_cache_bytes=2 * expert_bytes,
                          cache_decay_interval=4) as runtime:
            first = runtime.forward(x, ids, scale)
            snapshot = first.clone()
            before = runtime.stats()
            counter, version = runtime.token[1].item(), runtime.token._version
            for bad in (-2, w1.shape[0], -2):
                invalid = ids.clone()
                invalid[0, 0] = bad
                invalid[1, 0] = 0
                with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                    runtime.forward(x, invalid, scale)
                self.assertEqual(runtime.stats(), before)
                self.assertEqual(runtime.token[1].item(), counter)
                self.assertEqual(runtime.token._version, version)
                torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["cache_admissions"], 1)
            del out
            # Successful masked calls advance the local clock without adding heat.
            for rows in (19, 0):
                out = runtime.forward(x[:rows], torch.full_like(ids[:rows], -1),
                                      torch.full_like(scale[:rows], float("nan")))
                torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
                self.assertEqual(runtime.stats()["dynamic_experts"], 1)
                del out
            fresh_ids = torch.zeros_like(ids)
            for admissions in (0, 1):
                out = runtime.forward(x, fresh_ids, scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, fresh_ids, scale),
                                           atol=.002, rtol=.04)
                self.assertEqual(runtime.stats()["cache_admissions"], admissions)
                del out
            runtime.close()
            torch.testing.assert_close(first, snapshot, atol=0, rtol=0)

    def test_adaptive_signed_masked_dtypes_and_activation_selection(self):
        expert_bytes = 6 * 64 * 64
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=35, e=3, k=4, dtype=dtype)
            ids.copy_(torch.tensor([0, 2, 2, -1], device=x.device, dtype=torch.int32).expand_as(ids))
            ids[0] = -1
            scale.copy_(torch.linspace(-1.125, .875, scale.numel(), device=x.device).reshape_as(scale))
            scale[ids == -1] = float("nan")
            expected = reference(x, w1, w2, ids, scale)
            for policy in ("reuse", "lru"):
                with self.subTest(dtype=dtype, policy=policy), self.runtime(
                        w1, w2, banks=1, activation_mode="fp32_accumulate",
                        max_route_tile_rows="auto", dynamic_cache_bytes=2 * expert_bytes,
                        cache_policy=policy) as runtime:
                    for call in range(3):
                        out = runtime.forward(x, ids, scale)
                        self.assertEqual(out.dtype, dtype)
                        self.assertTrue(torch.isfinite(out).all().item())
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=0, rtol=0)
                        stats = runtime.stats()
                        self.assertEqual(stats["activation_mode"], 1)
                        self.assertEqual(stats["dynamic_cache_hits"],
                                         2 if call >= (2 if policy == "reuse" else 1) else 0)
                        self.assertEqual(stats["cache_budget_evictions"], 0)
                        if call:
                            self.assertGreater(stats["buffer_reuses"], 0)
                        del out
            # Tile 'auto' is independent of activation 'auto': on all-resident
            # K=1/4, activation still compares the two actual feasible charges.
            for k in (1, 4):
                routed = torch.zeros(x.shape[0], k, device=x.device, dtype=torch.int32)
                routed_scale = torch.full_like(routed, .5, dtype=torch.float32)
                peaks = []
                for mode in ("route_buffer", "fp32_accumulate", "auto"):
                    with self.runtime(w1, w2, activation_mode=mode, max_route_tile_rows="auto",
                                      resident_cache_bytes=expert_bytes, resident_experts={0: [0]}) as runtime:
                        out = runtime.forward(x, routed, routed_scale)
                        torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, routed, routed_scale),
                                                   atol=.002, rtol=.04)
                        stats = runtime.stats()
                        if mode == "auto":
                            chosen = 0 if peaks[0] <= peaks[1] else 1
                            self.assertEqual(stats["activation_mode"], chosen)
                            self.assertEqual(stats["gpu_charged_bytes"], min(peaks))
                        else:
                            peaks.append(stats["gpu_charged_bytes"])
                        del out

    def test_matrix(self):
        shapes = [(1, 64, 64, 8, 1), (7, 128, 192, 8, 3),
                  (129, 192, 64, 8, 4), (1024, 128, 128, 16, 4), (0, 64, 64, 8, 2)]
        for dtype in (torch.float16, torch.bfloat16):
            for shape in shapes:
                for banks in (1, 2):
                    with self.subTest(dtype=dtype, shape=shape, banks=banks):
                        x, w1, w2, ids, scale = self.tensors(*shape, dtype=dtype)
                        if ids.numel():
                            ids.view(-1)[0] = -1
                        with self.runtime(w1, w2, banks) as runtime:
                            version = runtime.token._version
                            out = runtime.forward(x, ids, scale)
                            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                                       atol=.002, rtol=.04)
                            self.assertGreater(runtime.token._version, version)
                            stats = runtime.stats()
                            active = len(set(ids.cpu().flatten().tolist()) - {-1})
                            self.assertEqual(stats["active_experts"], active)
                            self.assertEqual(len(stats), 31)
                            self.assertEqual(stats["cache_hits"], 0)
                            self.assertEqual(stats["cache_misses"], active)
                            for key in ("cache_hit_weight_bytes", "resident_cache_bytes",
                                        "resident_experts", "resident_load_h2d_bytes"):
                                self.assertEqual(stats[key], 0)
                            self.assertEqual(stats["weight_h2d_bytes"], active * 6 * shape[1] * shape[2])
                            if active:
                                self.assertEqual(stats["waves"], active)
                                self.assertEqual(stats["wave_capacity"], 1)

    @staticmethod
    def charged_bytes(size):
        return (size + 511) // 512 * 512 if size else 0

    def assert_resident_stats(self, runtime, ids, w1, w2, resident_ids, resident_bytes, load_bytes):
        stats = runtime.stats()
        active = sorted(set(ids.cpu().flatten().tolist()) - {-1})
        hits = len(set(active) & set(resident_ids))
        expert_bytes = 6 * w1.shape[2] * w2.shape[2]
        self.assertEqual(len(stats), 31)
        self.assertEqual(stats["active_experts"], len(active))
        self.assertEqual(stats["cache_hits"], hits)
        self.assertEqual(stats["cache_misses"], len(active) - hits)
        self.assertEqual(stats["cache_hit_weight_bytes"], hits * expert_bytes)
        self.assertEqual(stats["weight_h2d_bytes"], (len(active) - hits) * expert_bytes)
        self.assertEqual(stats["resident_cache_bytes"], resident_bytes)
        self.assertEqual(stats["resident_load_h2d_bytes"], load_bytes)
        if len(active) == hits:
            self.assertEqual(stats["num_banks"], 0)
            self.assertEqual(stats["wave_capacity"], 0)
            self.assertGreaterEqual(stats["waves"], (hits + 31) // 32)
            self.assertLessEqual(stats["waves"], hits)
        else:
            capacity = stats["wave_capacity"]
            self.assertGreater(capacity, 0)
            # Consecutive cached experts may share a pointer-table wave. Tight
            # budgets can shrink these groups down to the original single expert.
            minimum, maximum, run = 0, 0, []
            for expert in active + [None]:
                hit = expert in resident_ids if expert is not None else None
                if run and (hit != run[-1] or expert is None):
                    limit = 32 if run[-1] else capacity
                    minimum += (len(run) + limit - 1) // limit
                    maximum += len(run) if run[-1] else (len(run) + capacity - 1) // capacity
                    run = []
                if expert is not None:
                    run.append(hit)
            self.assertGreaterEqual(stats["waves"], minimum)
            self.assertLessEqual(stats["waves"], maximum)
        return stats

    def test_resident_mixed_all_hit_and_empty_all_modes_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=53, k=5, dtype=dtype)
            ids = torch.arange(53, device="cuda", dtype=torch.int32).remainder(8)[:, None].expand(53, 5).contiguous()
            ids[::7, -1] = -1
            scale = torch.linspace(-1, 1, ids.numel(), device="cuda").reshape_as(scale)
            resident_ids = [2, 5]
            load_bytes = len(resident_ids) * 6 * 64 * 64
            for mode in ("route_buffer", "fp32_accumulate", "auto"):
                for all_hit in (False, True):
                    with self.subTest(dtype=dtype, mode=mode, all_hit=all_hit):
                        routed = ids.clone()
                        if all_hit:
                            routed = torch.where(routed == -1, routed, 2 + 3 * routed.remainder(2))
                        with self.runtime(w1, w2, capacity=3, activation_mode=mode,
                                          max_route_tile_rows=17, resident_cache_bytes=8 * load_bytes,
                                          resident_experts={0: resident_ids}) as runtime:
                            initial = runtime.stats()
                            actual = initial["resident_cache_bytes"]
                            self.assertGreaterEqual(actual, load_bytes)
                            self.assertLess(actual, runtime.resident_cache_bytes)
                            self.assertEqual(initial["resident_experts"], 2)
                            self.assertEqual(initial["resident_load_h2d_bytes"], load_bytes)
                            for call_ids in (routed, routed, torch.full_like(routed, -1), routed[:0], routed):
                                rows = call_ids.shape[0]
                                out = runtime.forward(x[:rows], call_ids, scale[:rows])
                                torch.testing.assert_close(out.float().cpu(),
                                    reference(x[:rows], w1, w2, call_ids, scale[:rows]), atol=.002, rtol=.04)
                                stats = self.assert_resident_stats(
                                    runtime, call_ids, w1, w2, resident_ids, actual, load_bytes)
                                self.assertEqual(stats["resident_experts"], 2)
                                self.assertEqual(stats["gpu_charged_bytes"], stats["gpu_pool_bytes"] + actual
                                                 + self.charged_bytes(out.numel() * out.element_size()))
                                if not all_hit and rows and (call_ids >= 0).any().item() and mode != "auto":
                                    self.assertEqual(stats["wave_capacity"], 3)
                                del out

    def test_resident_layer_identity_shape_and_dtype(self):
        cases = [self.tensors(s=23, h=64, i=128, k=3),
                 self.tensors(s=23, h=64, i=128, k=3),
                 self.tensors(s=19, h=128, i=64, k=3, dtype=torch.bfloat16),
                 self.tensors(s=11, h=192, i=192, k=3)]
        # Layers 0/1 intentionally have identical shapes/dtypes but different weights.
        self.assertFalse(torch.equal(cases[0][1][0], cases[1][1][0]))
        for case in cases:
            case[3].copy_(torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32).expand_as(case[3]))
        store = ExpertStore({layer: (case[1], case[2]) for layer, case in enumerate(cases)})
        load_bytes = sum(6 * case[1].shape[2] * case[2].shape[2] for case in cases)
        for mode in ("route_buffer", "fp32_accumulate", "auto"):
            with self.subTest(mode=mode):
                with PrefillMoERuntime(store, gpu_budget_bytes=4 << 20, pinned_budget_bytes=250000,
                                       activation_mode=mode, max_route_tile_rows=7,
                                       resident_cache_bytes=2 * load_bytes,
                                       resident_experts={layer: [0] for layer in store.layers}) as runtime:
                    actual = runtime.stats()["resident_cache_bytes"]
                    saved = []
                    for layer in (0, 1, 2, 3, 1, 0):
                        x, w1, w2, ids, scale = cases[layer]
                        out = runtime.forward(x, ids, scale, layer_id=layer)
                        torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                                   atol=.002, rtol=.04)
                        stats = self.assert_resident_stats(runtime, ids, w1, w2, [0], actual, load_bytes)
                        self.assertEqual(stats["resident_experts"], len(cases))
                        saved.append((out, out.clone()))
                    runtime.close()
                    for out, snapshot in saved:
                        torch.testing.assert_close(out, snapshot, atol=0, rtol=0)

    def test_resident_native_additive_validation_and_late_bind(self):
        x, w1, w2, ids, scale = self.tensors(s=7, k=1)
        ids.fill_(0)
        expert_bytes = 6 * 64 * 64
        bind = torch.ops.recsys_ops.fuse_moe_offload_resident_bind
        with self.runtime(w1, w2, resident_cache_bytes=3 * expert_bytes) as runtime:
            bind(runtime.token, 0, [])
            self.assertEqual(runtime.stats()["resident_experts"], 0)
            for layer, invalid in ((0, [1, 1]), (0, [-1]), (0, [8]), (99, [0])):
                with self.subTest(layer=layer, invalid=invalid), self.assertRaises(RuntimeError):
                    bind(runtime.token, layer, invalid)
                self.assertEqual(runtime.token[1].item(), 0)
                self.assertEqual(runtime.stats()["resident_experts"], 0)
            # An unsuccessful forward does not freeze deployment binding.
            bad_ids = torch.full_like(ids, 8)
            with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                runtime.forward(x, bad_ids, scale)
            self.assertEqual(runtime.token[1].item(), 0)
            bind(runtime.token, 0, [0])
            first = runtime.stats()
            bind(runtime.token, 0, [])
            self.assertEqual(runtime.stats()["resident_load_h2d_bytes"], first["resident_load_h2d_bytes"])
            for invalid in ([0], [1, 0], [1, 1]):
                with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                    bind(runtime.token, 0, invalid)
                self.assertEqual(runtime.stats()["resident_experts"], 1)
            bind(runtime.token, 0, [2])
            self.assertEqual(runtime.stats()["resident_experts"], 2)
            self.assertEqual(runtime.stats()["resident_load_h2d_bytes"], 2 * expert_bytes)
            self.assertEqual(runtime.token[1].item(), 0)
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale), atol=.002, rtol=.04)
            self.assertEqual(runtime.token[1].item(), 1)
            for late_ids in ([1], []):
                with self.subTest(late_ids=late_ids), self.assertRaises(RuntimeError):
                    bind(runtime.token, 0, late_ids)
            self.assertEqual(runtime.stats()["resident_experts"], 2)

    def test_resident_empty_success_also_freezes_binding(self):
        x, w1, w2, ids, scale = self.tensors(s=7, k=1)
        for zero_rows in (False, True):
            with self.subTest(zero_rows=zero_rows):
                with self.runtime(w1, w2, resident_cache_bytes=2 * 6 * 64 * 64,
                                  resident_experts={0: [0]}) as runtime:
                    empty_ids = torch.full_like(ids, -1)
                    rows = 0 if zero_rows else x.shape[0]
                    out = runtime.forward(x[:rows], empty_ids[:rows], scale[:rows])
                    torch.testing.assert_close(out, torch.zeros_like(x[:rows]), atol=0, rtol=0)
                    self.assertEqual(runtime.token[1].item(), 1)
                    for late_ids in ([], [1]):
                        with self.subTest(late_ids=late_ids), self.assertRaises(RuntimeError):
                            torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, late_ids)

    def test_resident_weight_version_validation(self):
        for weight in (0, 1):
            for after_forward in (False, True):
                with self.subTest(weight=weight, after_forward=after_forward):
                    x, w1, w2, ids, scale = self.tensors(s=7, k=1)
                    ids.fill_(0)
                    with self.runtime(w1, w2, resident_cache_bytes=3 * 6 * 64 * 64,
                                      resident_experts={0: [0]}) as runtime:
                        if after_forward:
                            out = runtime.forward(x, ids, scale)
                            del out
                        (w1, w2)[weight].add_(1)
                        # Cached weights must not silently run a stale snapshot.
                        with self.assertRaisesRegex(RuntimeError, "version"):
                            runtime.forward(x, ids, scale)
                        if not after_forward:
                            with self.assertRaisesRegex(RuntimeError, "version"):
                                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, [1])
                            self.assertEqual(runtime.stats()["resident_experts"], 1)

    def test_resident_bound_version_checked_with_input_alias(self):
        x, w1, w2, ids, scale = self.tensors(s=7, k=1)
        ids.fill_(0)
        for weight in (0, 1):
            with self.subTest(weight=weight):
                with self.runtime(w1, w2, resident_cache_bytes=6 * 64 * 64,
                                  resident_experts={0: [0]}) as runtime:
                    alias1, alias2 = w1.data, w2.data
                    (w1, w2)[weight].add_(1)
                    with self.assertRaisesRegex(RuntimeError, "version"):
                        torch.ops.recsys_ops.fuse_moe_offload(
                            x, alias1, alias2, ids, scale, runtime.token, 0)

    def test_resident_native_budget_refusal_and_pageable_load(self):
        x, w1, w2, ids, scale = self.tensors(s=7, k=1)
        ids.fill_(0)
        expert_bytes = 6 * 64 * 64
        create = torch.ops.recsys_ops.fuse_moe_offload_create
        # The old seven-argument ABI remains usable with the appended default.
        token = create(torch.cuda.current_device(), 16 << 20, 64 << 20, 1, 1, 0, 4096)
        try:
            self.assertEqual(torch.ops.recsys_ops.fuse_moe_offload_stats(token).numel(), 31)
        finally:
            torch.ops.recsys_ops.fuse_moe_offload_close(token)
        for cache_cap, gpu_budget in ((-1, 16 << 20), (expert_bytes, expert_bytes - 1)):
            with self.subTest(cache_cap=cache_cap), self.assertRaises(RuntimeError):
                create(torch.cuda.current_device(), gpu_budget, 64 << 20, 1, 1, 0, 4096, cache_cap)
        with self.runtime(w1, w2, resident_cache_bytes=expert_bytes - 1) as runtime:
            with self.assertRaises(RuntimeError):
                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, [0])
            self.assertEqual(runtime.stats()["resident_experts"], 0)
            self.assertEqual(runtime.stats()["resident_load_h2d_bytes"], 0)
        with self.runtime(w1, w2, resident_cache_bytes=expert_bytes) as runtime:
            torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, [0])
            with self.assertRaises(RuntimeError):
                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, [1])
            self.assertEqual(runtime.stats()["resident_experts"], 1)
            self.assertEqual(runtime.stats()["resident_load_h2d_bytes"], expert_bytes)
        # Binding is synchronous from pageable CPU storage, with no staging pool.
        with self.runtime(w1, w2, pinned_budget_bytes=1, resident_cache_bytes=expert_bytes,
                          resident_experts={0: [0]}) as runtime:
            stats = runtime.stats()
            self.assertEqual(stats["resident_load_h2d_bytes"], expert_bytes)
            self.assertEqual(stats["pinned_pool_bytes"], 0)
            self.assertFalse(w1.is_pinned())
            self.assertFalse(w2.is_pinned())

    def test_resident_cache_cap_is_context_wide_and_failed_create_releases(self):
        _, w1, w2, _, _ = self.tensors(s=7)
        store = ExpertStore({0: (w1, w2), 5: (w1.clone(), w2.clone())})
        expert_bytes = 6 * 64 * 64
        with PrefillMoERuntime(store, gpu_budget_bytes=16 << 20,
                               resident_cache_bytes=expert_bytes, resident_experts={0: [0]}) as runtime:
            with self.assertRaises(RuntimeError):
                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 5, [0])
            self.assertEqual(runtime.stats()["resident_experts"], 1)
            self.assertEqual(runtime.stats()["resident_load_h2d_bytes"], expert_bytes)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        # First layer loads successfully, second exceeds the shared cap. The
        # frontend must close the partially created context and release its cache.
        with self.assertRaises(RuntimeError):
            PrefillMoERuntime(store, gpu_budget_bytes=16 << 20,
                              resident_cache_bytes=expert_bytes, resident_experts={0: [0], 5: [0]})
        gc.collect()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), before)

    def test_resident_actual_charge_squeezes_scratch_budget(self):
        x, w1, w2, ids, scale = self.tensors(s=257, k=4)
        ids.copy_(torch.arange(4, device="cuda", dtype=torch.int32).expand_as(ids))
        expected = reference(x, w1, w2, ids, scale)
        expert_bytes = 6 * 64 * 64
        # Cache an inactive expert so no saved miss bank can mask its GPU cost.
        with self.runtime(w1, w2, banks=1, activation_mode="fp32_accumulate",
                          max_route_tile_rows=8, resident_cache_bytes=expert_bytes,
                          resident_experts={0: [7]}) as probe:
            out = probe.forward(x, ids, scale)
            stats = probe.stats()
            budget = stats["gpu_charged_bytes"]
            actual = stats["resident_cache_bytes"]
            self.assertEqual(budget, stats["gpu_pool_bytes"] + actual
                             + self.charged_bytes(out.numel() * out.element_size()))
            del out
        gc.collect()
        for mode in ("fp32_accumulate", "auto"):
            with self.subTest(mode=mode):
                # Deliberately set cache cap == total GPU budget. It must not be
                # reserved in full or treated as an independent extra GPU budget.
                with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                                  activation_mode=mode, max_route_tile_rows=4096,
                                  resident_cache_bytes=budget, resident_experts={0: [7]}) as runtime:
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                    stats = self.assert_resident_stats(runtime, ids, w1, w2, [7], actual, expert_bytes)
                    self.assertEqual(stats["activation_mode"], 1)
                    self.assertGreater(stats["route_tile_rows"], 0)
                    self.assertLessEqual(stats["route_tile_rows"], 8)
                    self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                    del out
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                          activation_mode="fp32_accumulate", max_route_tile_rows=4096) as runtime:
            out = runtime.forward(x, ids, scale)
            self.assertGreater(runtime.stats()["route_tile_rows"], 8)
            del out
        # Route-buffer scratch cannot shrink: the baseline exact peak fits only
        # without the inactive cached expert. Cache cap is not reusable headroom.
        with self.runtime(w1, w2, banks=1) as probe:
            out = probe.forward(x, ids, scale)
            route_budget = probe.stats()["gpu_charged_bytes"]
            del out
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=route_budget,
                          resident_cache_bytes=route_budget, resident_experts={0: [7]}) as runtime:
            self.assertEqual(runtime.stats()["resident_cache_bytes"], actual)
            with self.assertRaisesRegex(RuntimeError, "gpu_budget|budgets|history"):
                runtime.forward(x, ids, scale)

    def test_resident_pool_close_preserves_live_output(self):
        x, w1, w2, ids, scale = self.tensors(s=129, k=4)
        ids.copy_(torch.arange(4, device="cuda", dtype=torch.int32).expand_as(ids))
        expert_bytes = 6 * 64 * 64
        with self.runtime(w1, w2, banks=1, resident_cache_bytes=expert_bytes,
                          resident_experts={0: [0]}) as probe:
            out = probe.forward(x, ids, scale)
            budget = probe.stats()["gpu_charged_bytes"]
            del out
        gc.collect()
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                          resident_cache_bytes=budget, resident_experts={0: [0]}) as runtime:
            out = runtime.forward(x, ids, scale)
            snapshot = out.clone()
            stats = runtime.stats()
            pool, cache = stats["gpu_pool_bytes"], stats["resident_cache_bytes"]
            self.assertGreater(pool, 0)
            self.assertGreater(cache, 0)
            self.assertEqual(stats["gpu_charged_bytes"], pool + cache
                             + self.charged_bytes(out.numel() * out.element_size()))
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            runtime.close()
            torch.cuda.synchronize()
            self.assertGreaterEqual(before - torch.cuda.memory_allocated(), pool + cache)
            torch.testing.assert_close(out, snapshot, atol=0, rtol=0)
            runtime.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                runtime.stats()
            with self.assertRaises(RuntimeError):
                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, [1])

    def test_grouped_waves_and_kernel_limits(self):
        for shape in ((513, 128, 192, 8, 4), (5, 64, 64, 512, 128), (1537, 64, 64, 8, 4)):
            with self.subTest(shape=shape):
                x, w1, w2, ids, scale = self.tensors(*shape)
                if shape[0] == 1537:
                    ids.remainder_(3)
                with self.runtime(w1, w2, capacity=3) as runtime:
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                               atol=.002, rtol=.04)
                    active = len(set(ids.cpu().flatten().tolist()))
                    self.assertEqual(runtime.stats()["wave_capacity"], min(3, active))
                    self.assertEqual(runtime.stats()["waves"], (active + 2) // 3)

    def test_activation_modes_tiled_signed_routes(self):
        # Six mode/dtype cases avoid multiplying the legacy shape/bank matrix.
        # Eight experts leave a partial capacity-3 wave; duplicates force tiles.
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=53, k=5, dtype=dtype)
            ids = torch.arange(53, device="cuda", dtype=torch.int32).remainder(8)[:, None].expand(53, 5).contiguous()
            ids[::7, -1] = -1
            scale = torch.linspace(-1, 1, ids.numel(), device="cuda").reshape_as(scale)
            expected = reference(x, w1, w2, ids, scale)
            for mode in ("route_buffer", "fp32_accumulate", "auto"):
                with self.subTest(dtype=dtype, mode=mode):
                    with self.runtime(w1, w2, capacity=3, activation_mode=mode,
                                      max_route_tile_rows=17) as runtime:
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        self.assertEqual(len(stats), 31)
                        self.assertEqual(stats["active_experts"], 8)
                        if mode == "auto":
                            # Auto compares GPU charge across feasible candidates,
                            # so it need not retain the largest pinned capacity.
                            self.assertGreaterEqual(stats["wave_capacity"], 1)
                            self.assertLessEqual(stats["wave_capacity"], 3)
                            self.assertEqual(stats["waves"], (8 + stats["wave_capacity"] - 1)
                                             // stats["wave_capacity"])
                        else:
                            self.assertEqual(stats["waves"], 3)
                            self.assertEqual(stats["wave_capacity"], 3)
                        self.assertEqual(stats["weight_h2d_bytes"], 8 * 6 * 64 * 64)
                        self.assertIn(stats["activation_mode"], (0, 1))
                        if mode != "auto":
                            self.assertEqual(stats["activation_mode"], int(mode == "fp32_accumulate"))
                        self.assertGreater(stats["buffer_allocations"], 0)
                        self.assertGreater(stats["gpu_pool_bytes"], 0)
                        self.assertGreater(stats["pinned_pool_bytes"], 0)
                        if stats["activation_mode"] == 1:
                            self.assertGreater(stats["route_tiles"], stats["waves"])
                            self.assertGreater(stats["route_tile_rows"], 0)
                            self.assertLessEqual(stats["route_tile_rows"], 17)
                        snapshot = out.clone()
                        del out
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out, snapshot, atol=.002, rtol=.04)
                        self.assertEqual(runtime.stats()["buffer_allocations"], 0)
                        self.assertGreater(runtime.stats()["buffer_reuses"], 0)

    def test_empty_routes_and_recovery_all_modes(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=19, k=3, dtype=dtype)
            for mode in ("route_buffer", "fp32_accumulate", "auto"):
                with self.subTest(dtype=dtype, mode=mode):
                    with self.runtime(w1, w2, activation_mode=mode, max_route_tile_rows=7) as runtime:
                        valid_ids = ids.clone()
                        expected = reference(x, w1, w2, valid_ids, scale)
                        for bad in (-2, w1.shape[0]):
                            ids.copy_(valid_ids)
                            ids[0, 0] = bad
                            with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                                runtime.forward(x, ids, scale)
                            ids.copy_(valid_ids)
                            out = runtime.forward(x, ids, scale)
                            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                            del out
                        ids.fill_(-1)
                        for empty_x, empty_ids, empty_scale in ((x, ids, scale), (x[:0], ids[:0], scale[:0])):
                            out = runtime.forward(empty_x, empty_ids, empty_scale)
                            torch.testing.assert_close(out, torch.zeros_like(empty_x), atol=0, rtol=0)
                            self.assertEqual(runtime.stats()["active_experts"], 0)
                            self.assertEqual(runtime.stats()["weight_h2d_bytes"], 0)
                            self.assertEqual(runtime.stats()["waves"], 0)
                            self.assertEqual(runtime.stats()["route_tiles"], 0)
                        ids.copy_(valid_ids)
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)

    def test_inference_mode_and_nondefault_stream(self):
        x, w1, w2, ids, scale = self.tensors(s=7)
        with torch.inference_mode():
            runtime = self.runtime(w1, w2)
        try:
            self.assertFalse(runtime.token.is_inference())
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            self.assertEqual(runtime.token[1].item(), 1)
        finally:
            runtime.close()
        with self.assertRaises(ValueError):
            ExpertStore({0: (w1.pin_memory(), w2)})

    def test_cross_stream_pool_reuse(self):
        x, w1, w2, ids, scale = self.tensors(s=41, k=3)
        expected = reference(x, w1, w2, ids, scale)
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        producer = torch.cuda.current_stream()
        for mode in ("route_buffer", "fp32_accumulate", "auto"):
            with self.subTest(mode=mode):
                with self.runtime(w1, w2, activation_mode=mode, max_route_tile_rows=13) as runtime:
                    saved = []
                    for index, stream in enumerate(streams):
                        stream.wait_stream(producer)
                        if index:
                            stream.wait_stream(streams[index - 1])
                        with torch.cuda.stream(stream):
                            out = runtime.forward(x, ids, scale)
                            saved.append((out, out.clone()))
                        producer.wait_stream(stream)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        if index:
                            self.assertEqual(runtime.stats()["buffer_allocations"], 0)
                            self.assertGreater(runtime.stats()["buffer_reuses"], 0)
                    for out, snapshot in saved:
                        torch.testing.assert_close(out, snapshot, atol=0, rtol=0)

    def test_resident_cross_stream_reuse(self):
        producer = torch.cuda.default_stream()
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        with torch.cuda.stream(producer):
            x, w1, w2, ids, scale = self.tensors(s=41, k=3)
            mixed_ids = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32).expand_as(ids).contiguous()
            all_hit_ids = torch.tensor([0, 2, 0], device="cuda", dtype=torch.int32).expand_as(ids).contiguous()
        routes = (mixed_ids, all_hit_ids, mixed_ids)
        expected = [reference(x, w1, w2, routed, scale) for routed in routes]
        expert_bytes = 6 * w1.shape[2] * w2.shape[2]
        for mode in ("route_buffer", "fp32_accumulate"):
            with self.subTest(mode=mode), torch.cuda.stream(producer):
                # Wait only for inputs before resident binding. No post-bind wait
                # masks its synchronous drain when the first caller reads cache.
                streams[0].wait_stream(producer)
                with self.runtime(w1, w2, activation_mode=mode, max_route_tile_rows=13,
                                  gpu_budget_bytes=16 << 20, resident_cache_bytes=2 * expert_bytes,
                                  resident_experts={0: [0, 2]}) as runtime:
                    resident_bytes = runtime.stats()["resident_cache_bytes"]
                    self.assertEqual(runtime.token[1].item(), 0)
                    saved = []
                    previous = None
                    for index, routed in enumerate(routes):
                        stream = streams[index % len(streams)]
                        if previous is not None:
                            stream.wait_stream(producer)
                            stream.wait_stream(previous)
                        version = runtime.token._version
                        with torch.cuda.stream(stream):
                            out = runtime.forward(x, routed, scale)
                            self.assertTrue(stream.query(), "forward must drain the caller stream")
                            self.assertGreater(runtime.token._version, version)
                            self.assertEqual(runtime.token[1].item(), index + 1)
                            saved.append((out, out.clone()))
                        producer.wait_stream(stream)
                        torch.testing.assert_close(out.float().cpu(), expected[index], atol=.002, rtol=.04)
                        stats = self.assert_resident_stats(
                            runtime, routed, w1, w2, [0, 2], resident_bytes, 2 * expert_bytes)
                        self.assertEqual(stats["resident_experts"], 2)
                        self.assertLessEqual(stats["gpu_charged_bytes"], 16 << 20)
                        if index:
                            self.assertGreater(stats["buffer_reuses"], 0)
                        previous = stream
                    runtime.close()
                    for out, snapshot in saved:
                        torch.testing.assert_close(out, snapshot, atol=0, rtol=0)

    def test_shape_layer_dtype_pool_changes(self):
        cases = [self.tensors(s=73, h=64, i=128, k=3),
                 self.tensors(s=29, h=128, i=64, k=5, dtype=torch.bfloat16),
                 self.tensors(s=11, h=192, i=192, k=2)]
        store = ExpertStore({layer: (case[1], case[2]) for layer, case in enumerate(cases)})
        for mode in ("route_buffer", "fp32_accumulate", "auto"):
            with self.subTest(mode=mode):
                with PrefillMoERuntime(store, gpu_budget_bytes=4 << 20,
                                       ram_budget_bytes=4 << 20, pinned_budget_bytes=250000,
                                       num_banks=1, activation_mode=mode, max_route_tile_rows=19) as runtime:
                    saved = []
                    # Revisit both shapes and layers after H/I/dtype changes.
                    for layer, rows in ((0, 73), (1, 29), (2, 11), (0, 17), (1, 7), (0, 73)):
                        x, w1, w2, ids, scale = cases[layer]
                        x, ids, scale = x[:rows], ids[:rows], scale[:rows]
                        out = runtime.forward(x, ids, scale, layer_id=layer)
                        torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                                   atol=.002, rtol=.04)
                        saved.append((out, out.clone()))
                        stats = runtime.stats()
                        self.assertLessEqual(stats["gpu_charged_bytes"], 4 << 20)
                        self.assertLessEqual(stats["pinned_bytes"], 250000)
                        self.assertLessEqual(stats["gpu_pool_bytes"], stats["gpu_charged_bytes"])
                        self.assertLessEqual(stats["pinned_pool_bytes"], stats["pinned_bytes"])
                        active = len(set(ids.cpu().flatten().tolist()) - {-1})
                        self.assertEqual(stats["weight_h2d_bytes"], active * 6 * x.shape[1] * w2.shape[2])
                    for out, snapshot in saved:
                        torch.testing.assert_close(out, snapshot, atol=0, rtol=0)

    def test_pool_highwater_budget_trim_and_close(self):
        x, w1, w2, ids, scale = self.tensors(s=257, k=4)
        with self.runtime(w1, w2, banks=1) as probe:
            out = probe.forward(x, ids, scale)
            budget = probe.stats()["gpu_charged_bytes"]
            del out
        gc.collect()
        runtime = self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget)
        try:
            first = runtime.forward(x, ids, scale)
            snapshot = first.clone()
            highwater = runtime.stats()["gpu_pool_bytes"]
            # Keeping the large output alive leaves less than the old pool plus a
            # new output. A smaller shape can fit only by trimming pool capacity.
            small = runtime.forward(x[:17], ids[:17], scale[:17])
            torch.testing.assert_close(small.float().cpu(), reference(x[:17], w1, w2, ids[:17], scale[:17]),
                                       atol=.002, rtol=.04)
            stats = runtime.stats()
            self.assertLess(stats["gpu_pool_bytes"], highwater)
            self.assertLessEqual(stats["gpu_charged_bytes"], budget)
            self.assertLessEqual(stats["pinned_pool_bytes"], stats["pinned_bytes"])
            torch.testing.assert_close(first, snapshot, atol=0, rtol=0)
            del first, snapshot, small
            gc.collect()
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            pool_bytes = runtime.stats()["gpu_pool_bytes"]
            self.assertGreater(pool_bytes, 0)
            snapshot = out.clone()
            torch.cuda.synchronize()
            before_close = torch.cuda.memory_allocated()
            runtime.close()
            torch.cuda.synchronize()
            # Compare live allocations, not caching-allocator reserve. The output
            # remains alive; only context-owned GPU tensors should disappear.
            self.assertGreaterEqual(before_close - torch.cuda.memory_allocated(), pool_bytes)
            torch.testing.assert_close(out, snapshot, atol=0, rtol=0)
            runtime.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                runtime.stats()
        finally:
            runtime.close()

    def test_auto_budget_selection_and_fp32_tile_shrink(self):
        x, w1, w2, ids, scale = self.tensors(s=37, k=8)
        # Every expert is active and has enough rows for several tiles.
        ids.copy_(torch.arange(8, device="cuda", dtype=torch.int32).expand_as(ids))
        expected = reference(x, w1, w2, ids, scale)
        peaks = {}
        # The 64-column Act tile (FP16/BF16) is charged in 512-byte units;
        # fp32_accumulate no longer has a Down tile. Seven and eight Act rows
        # cost the same; use an alignment boundary so this exact byte budget
        # truly imposes the asserted row bound after replanning.
        probe_rows = 8
        for mode in ("route_buffer", "fp32_accumulate"):
            with self.runtime(w1, w2, banks=1, activation_mode=mode, max_route_tile_rows=probe_rows) as probe:
                out = probe.forward(x, ids, scale)
                peaks[mode] = probe.stats()["gpu_charged_bytes"]
                del out
        self.assertLess(peaks["fp32_accumulate"], peaks["route_buffer"])
        budget = peaks["fp32_accumulate"]
        for mode in ("fp32_accumulate", "auto"):
            with self.subTest(mode=mode):
                with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                                  activation_mode=mode, max_route_tile_rows=4096) as runtime:
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(stats["activation_mode"], 1)
                    self.assertGreater(stats["route_tile_rows"], 0)
                    self.assertLessEqual(stats["route_tile_rows"], probe_rows)
                    self.assertGreater(stats["route_tiles"], stats["waves"])
                    self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                    self.assertEqual(stats["weight_h2d_bytes"], 8 * 6 * 64 * 64)
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=budget,
                          activation_mode="route_buffer", max_route_tile_rows=probe_rows) as runtime:
            with self.assertRaisesRegex(RuntimeError, "gpu_budget|budgets"):
                runtime.forward(x, ids, scale)
        # With all candidates feasible, auto still prefers the smaller GPU charge.
        with self.runtime(w1, w2, banks=1, activation_mode="auto", max_route_tile_rows=probe_rows) as runtime:
            out = runtime.forward(x, ids, scale)
            self.assertEqual(runtime.stats()["activation_mode"], 1)
            self.assertLessEqual(runtime.stats()["gpu_charged_bytes"], min(peaks.values()))
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)

    def test_auto_small_k_prefers_route_buffer(self):
        x, w1, w2, ids, scale = self.tensors(s=7, k=1)
        ids.fill_(0)
        peaks = {}
        for mode in ("route_buffer", "fp32_accumulate", "auto"):
            with self.runtime(w1, w2, banks=1, activation_mode=mode) as runtime:
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                           atol=.002, rtol=.04)
                peaks[mode] = runtime.stats()["gpu_charged_bytes"]
                if mode == "auto":
                    self.assertEqual(runtime.stats()["activation_mode"], 0)
        self.assertLess(peaks["route_buffer"], peaks["fp32_accumulate"])
        self.assertLessEqual(peaks["auto"], min(peaks["route_buffer"], peaks["fp32_accumulate"]))

    def test_route_modes_and_layer_sharing(self):
        x, w1, w2, ids, scale = self.tensors(s=37)
        store = ExpertStore({0: (w1, w2), 5: (w1.clone(), w2.clone())})
        with PrefillMoERuntime(store, gpu_budget_bytes=4 << 20, ram_budget_bytes=4 << 20,
                              pinned_budget_bytes=50000) as runtime:
            saved = []
            for mode in ("duplicate", "single", "invalid", "sparse"):
                if mode == "duplicate":
                    ids = torch.arange(37, device="cuda", dtype=torch.int32).remainder(8)[:, None].expand(37, 4).contiguous()
                elif mode == "single":
                    ids.fill_(7)
                elif mode == "invalid":
                    ids.fill_(-1)
                else:
                    ids = torch.randint(2, (37, 4), device="cuda", dtype=torch.int32)
                out = runtime.forward(x, ids, scale, layer_id=5)
                torch.testing.assert_close(out.float().cpu(), reference(x, *store.layers[5], ids, scale),
                                           atol=.002, rtol=.04)
                saved.append((out, out.clone()))
            for out, snapshot in saved:
                torch.testing.assert_close(out, snapshot, atol=0, rtol=0)

    def test_validation_and_recovery(self):
        x, w1, w2, ids, scale = self.tensors(s=7)
        with self.runtime(w1, w2) as runtime:
            for bad in (-2, 8):
                ids[0, 0] = bad
                with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                    runtime.forward(x, ids, scale)
            ids[0, 0] = 0
            runtime.forward(x, ids, scale)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                torch.ops.recsys_ops.fuse_moe_offload(x, w1, w2, ids, scale, runtime.token.clone(), 0)
            w1.add_(1)
            with self.assertRaisesRegex(RuntimeError, "version"):
                runtime.forward(x, ids, scale)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.forward(x, ids, scale)
        with self.assertRaises(RuntimeError):
            self.runtime(w1, w2, ram_budget_bytes=1)

    def test_budget_and_live_outputs(self):
        x, w1, w2, ids, scale = self.tensors(s=129)
        with self.runtime(w1, w2, banks=1) as baseline:
            out = baseline.forward(x, ids, scale)
            peak = baseline.stats()["gpu_charged_bytes"]
            del out
        gc.collect()
        # Default route_buffer retains full route buffers: unlike fp32 accumulation,
        # it cannot shrink tile rows to fit a second retained output at this budget.
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=peak) as runtime:
            first = runtime.forward(x, ids, scale)
            with self.assertRaisesRegex(RuntimeError, "budgets|history"):
                runtime.forward(x, ids, scale)
            del first
            gc.collect()
            runtime.forward(x, ids, scale)
        with self.runtime(w1, w2, gpu_budget_bytes=1) as runtime:
            with self.assertRaisesRegex(RuntimeError, "gpu_budget"):
                runtime.forward(x, ids, scale)
        # Two requested banks must safely degrade when only one fits pinned.
        pin = (w1.shape[0] + 1) * 4 + 6 * 64 * 64 + 24
        with self.runtime(w1, w2, pinned_budget_bytes=pin) as runtime:
            runtime.forward(x, ids, scale)
            self.assertEqual(runtime.stats()["num_banks"], 1)

    def test_dynamic_disabled_stats_append_only(self):
        x, w1, w2, ids, scale = self.tensors(s=19, k=3)
        ids.copy_(torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32).expand_as(ids))
        appended = (
            "dynamic_cache_bytes", "dynamic_experts", "dynamic_cache_hits",
            "cache_admissions", "cache_evictions", "cache_bypasses",
            "cache_promotion_d2d_bytes", "cache_budget_evictions", "cache_metadata_bytes")
        original = (
            "active_experts", "waves", "weight_h2d_bytes", "metadata_h2d_bytes",
            "gpu_charged_bytes", "pinned_bytes", "ram_charged_bytes", "num_banks",
            "wave_capacity", "activation_mode", "route_tiles", "route_tile_rows",
            "buffer_allocations", "buffer_reuses", "gpu_pool_bytes", "pinned_pool_bytes",
            "cache_hits", "cache_misses", "cache_hit_weight_bytes", "resident_cache_bytes",
            "resident_experts", "resident_load_h2d_bytes")
        for policy in ("reuse", "lru"):
            with self.subTest(policy=policy), self.runtime(
                    w1, w2, capacity=1, dynamic_cache_bytes=0, cache_policy=policy) as runtime:
                for _ in range(3):
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                               atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(tuple(stats), original + appended)
                    self.assertEqual(stats["cache_hits"], 0)
                    self.assertEqual(stats["cache_misses"], 2)
                    self.assertEqual(stats["weight_h2d_bytes"], 2 * 6 * 64 * 64)
                    self.assertEqual(stats["cache_hit_weight_bytes"], 0)
                    for key in appended:
                        self.assertEqual(stats[key], 0, key)
                    del out

    def test_dynamic_cold_second_promotes_third_hits(self):
        expert_bytes = 6 * 64 * 64
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=53, k=5, dtype=dtype)
            ids.fill_(3)
            ids[::7, -1] = -1
            scale.copy_(torch.linspace(-1, 1, scale.numel(), device="cuda").reshape_as(scale))
            expected = reference(x, w1, w2, ids, scale)
            for mode in ("route_buffer", "fp32_accumulate", "auto"):
                with self.subTest(dtype=dtype, mode=mode), self.runtime(
                        w1, w2, capacity=1, activation_mode=mode, max_route_tile_rows=17,
                        dynamic_cache_bytes=expert_bytes, cache_policy="reuse") as runtime:
                    metadata = None
                    total_h2d = 0
                    for call, (hits, admissions, bypasses) in enumerate(
                            ((0, 0, 1), (0, 1, 0), (1, 0, 0)), 1):
                        version = runtime.token._version
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        self.assertEqual(stats["active_experts"], 1)
                        self.assertEqual(stats["cache_hits"], hits)
                        self.assertEqual(stats["dynamic_cache_hits"], hits)
                        self.assertEqual(stats["cache_misses"], 1 - hits)
                        self.assertEqual(stats["cache_hit_weight_bytes"], hits * expert_bytes)
                        self.assertEqual(stats["weight_h2d_bytes"], (1 - hits) * expert_bytes)
                        self.assertEqual(stats["cache_admissions"], admissions)
                        self.assertEqual(stats["cache_bypasses"], bypasses)
                        self.assertEqual(stats["cache_promotion_d2d_bytes"], admissions * expert_bytes)
                        self.assertEqual(stats["cache_evictions"], 0)
                        self.assertEqual(stats["cache_budget_evictions"], 0)
                        self.assertEqual(stats["dynamic_experts"], int(call >= 2))
                        self.assertEqual(stats["dynamic_cache_bytes"], int(call >= 2) * expert_bytes)
                        for key in ("resident_cache_bytes", "resident_experts", "resident_load_h2d_bytes"):
                            self.assertEqual(stats[key], 0)
                        self.assertGreater(stats["cache_metadata_bytes"], 0)
                        if metadata is None:
                            metadata = stats["cache_metadata_bytes"]
                        self.assertEqual(stats["cache_metadata_bytes"], metadata)
                        self.assertEqual(stats["gpu_charged_bytes"], stats["gpu_pool_bytes"]
                                         + stats["dynamic_cache_bytes"]
                                         + self.charged_bytes(out.numel() * out.element_size()))
                        self.assertLessEqual(stats["gpu_charged_bytes"], 16 << 20)
                        self.assertGreater(runtime.token._version, version)
                        self.assertEqual(runtime.token[1].item(), call)
                        total_h2d += stats["weight_h2d_bytes"]
                        del out
                    self.assertEqual(total_h2d, 2 * expert_bytes)

    def test_dynamic_token_and_slot_counts_are_deduplicated(self):
        x, w1, w2, ids, scale = self.tensors(s=1000, k=4)
        ids.fill_(2)
        expert_bytes = 6 * 64 * 64
        expected = reference(x, w1, w2, ids, scale)
        with self.runtime(w1, w2, capacity=1, dynamic_cache_bytes=expert_bytes,
                          cache_policy="reuse", cache_min_reuse=3) as runtime:
            for call in range(1, 5):
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["active_experts"], 1)
                self.assertEqual(stats["cache_hits"], int(call == 4))
                self.assertEqual(stats["dynamic_cache_hits"], int(call == 4))
                self.assertEqual(stats["cache_misses"], int(call != 4))
                self.assertEqual(stats["cache_admissions"], int(call == 3))
                self.assertEqual(stats["cache_bypasses"], int(call < 3))
                self.assertEqual(stats["dynamic_experts"], int(call >= 3))
                self.assertEqual(stats["weight_h2d_bytes"], int(call != 4) * expert_bytes)
                self.assertEqual(stats["cache_promotion_d2d_bytes"], int(call == 3) * expert_bytes)
                del out

    def test_dynamic_reuse_scan_retains_hot_vs_lru_pollution(self):
        x, w1, w2, ids, scale = self.tensors(s=23, k=2)
        expert_bytes = 6 * 64 * 64
        for policy in ("reuse", "lru"):
            with self.subTest(policy=policy), self.runtime(
                    w1, w2, capacity=1, dynamic_cache_bytes=2 * expert_bytes,
                    cache_policy=policy, cache_min_reuse=2) as runtime:
                def call(experts):
                    routed = torch.tensor(experts, device="cuda", dtype=torch.int32).expand_as(ids).contiguous()
                    out = runtime.forward(x, routed, scale)
                    torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, routed, scale),
                                               atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(stats["active_experts"], len(set(experts)))
                    self.assertEqual(stats["cache_hits"], stats["dynamic_cache_hits"])
                    self.assertEqual(stats["cache_misses"] + stats["cache_hits"], len(set(experts)))
                    self.assertEqual(stats["weight_h2d_bytes"], stats["cache_misses"] * expert_bytes)
                    self.assertEqual(stats["cache_hit_weight_bytes"], stats["cache_hits"] * expert_bytes)
                    self.assertEqual(stats["cache_promotion_d2d_bytes"], stats["cache_admissions"] * expert_bytes)
                    self.assertEqual(stats["cache_budget_evictions"], 0)
                    self.assertLessEqual(stats["dynamic_cache_bytes"], 2 * expert_bytes)
                    return stats

                # Six successful layer calls heat each hot expert six times, not 23 times.
                for index in range(6):
                    stats = call([0, 1])
                    cold = index < (2 if policy == "reuse" else 1)
                    self.assertEqual(stats["cache_hits"], 0 if cold else 2)
                    self.assertEqual(stats["cache_admissions"],
                                     2 * int(index == (1 if policy == "reuse" else 0)))
                for expert in range(2, 8):
                    stats = call([expert, expert])
                    self.assertEqual(stats["cache_hits"], 0)
                    self.assertEqual(stats["cache_admissions"], int(policy == "lru"))
                    self.assertEqual(stats["cache_evictions"], int(policy == "lru"))
                    self.assertEqual(stats["cache_bypasses"], int(policy == "reuse"))
                    self.assertEqual(stats["dynamic_experts"], 2)
                    self.assertEqual(stats["dynamic_cache_bytes"], 2 * expert_bytes)
                stats = call([0, 1])
                self.assertEqual(stats["cache_hits"], 2 if policy == "reuse" else 0)
                self.assertEqual(stats["weight_h2d_bytes"], 0 if policy == "reuse" else 2 * expert_bytes)
                self.assertEqual(stats["cache_admissions"], 0 if policy == "reuse" else 2)
                self.assertEqual(stats["cache_evictions"], 0 if policy == "reuse" else 2)
                stats = call([0, 1])
                self.assertEqual(stats["cache_hits"], 2)
                self.assertEqual(stats["cache_admissions"], 0)
                self.assertEqual(stats["cache_evictions"], 0)
                self.assertEqual(stats["cache_bypasses"], 0)

    def test_dynamic_decay_phase_shift_and_layer_local_clock(self):
        x, w1, w2, ids, scale = self.tensors(s=11, k=1)
        store = ExpertStore({0: (w1, w2), 5: (w1.clone(), w2.clone())})
        expert_bytes = 6 * 64 * 64
        pin = (w1.shape[0] + 1) * 4 + expert_bytes + 24
        # Boundary calls below are empty, so promotion does not depend on the
        # order of increment/halving within a successful boundary invocation.
        for interval in (2, 4):
            with self.subTest(interval=interval), PrefillMoERuntime(
                    store, gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20,
                    pinned_budget_bytes=pin, num_banks=1, dynamic_cache_bytes=expert_bytes,
                    cache_decay_interval=interval, cache_min_reuse=2) as runtime:
                def call(expert, layer=0):
                    routed = torch.full_like(ids, expert)
                    pair = store.layers[layer]
                    out = runtime.forward(x, routed, scale, layer_id=layer)
                    torch.testing.assert_close(out.float().cpu(), reference(x, *pair, routed, scale),
                                               atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(stats["cache_hits"], stats["dynamic_cache_hits"])
                    self.assertEqual(stats["weight_h2d_bytes"], stats["cache_misses"] * expert_bytes)
                    self.assertEqual(stats["cache_promotion_d2d_bytes"], stats["cache_admissions"] * expert_bytes)
                    self.assertEqual(stats["cache_budget_evictions"], 0)
                    return stats

                for _ in range(3):
                    stats = call(0)
                self.assertEqual(stats["dynamic_experts"], 1)
                # These 2*interval successful calls must not decay layer 0.
                for _ in range(2 * interval):
                    stats = call(-1, layer=5)
                    self.assertEqual(stats["cache_admissions"], 0)
                    self.assertEqual(stats["cache_evictions"], 0)
                    self.assertEqual(stats["cache_bypasses"], 0)
                    self.assertEqual(stats["dynamic_experts"], 1)
                # Heat the rival without exceeding the frozen hot benefit.
                # Equal-frequency candidates must bypass the hysteresis test.
                for _ in range(3):
                    stats = call(1, layer=5)
                    self.assertEqual(stats["cache_hits"], 0)
                    self.assertEqual(stats["cache_admissions"], 0)
                    self.assertEqual(stats["cache_bypasses"], 1)
                    self.assertEqual(stats["cache_evictions"], 0)
                # Both local clocks now reach an empty boundary: 2 -> 1 or
                # 3 -> 1. No routing frequency is added by either empty call.
                for layer in (5, 0):
                    stats = call(-1, layer=layer)
                    self.assertEqual(stats["cache_admissions"], 0)
                    self.assertEqual(stats["cache_evictions"], 0)
                    self.assertEqual(stats["cache_bypasses"], 0)
                    self.assertEqual(stats["dynamic_experts"], 1)
                # Rival becomes frequency two: 2B > 1B + .25B. Replacement
                # remains a miss this call; only the following call is a hit.
                stats = call(1, layer=5)
                self.assertEqual(stats["cache_hits"], 0)
                self.assertEqual(stats["cache_misses"], 1)
                self.assertEqual(stats["cache_admissions"], 1)
                self.assertEqual(stats["cache_evictions"], 1)
                self.assertEqual(stats["cache_bypasses"], 0)
                self.assertEqual(stats["dynamic_experts"], 1)
                stats = call(1, layer=5)
                self.assertEqual(stats["cache_hits"], 1)
                self.assertEqual(stats["cache_admissions"], 0)
                self.assertEqual(stats["cache_evictions"], 0)

    def test_dynamic_default_decay_after_32_successful_layer_calls(self):
        x, w1, w2, ids, scale = self.tensors(s=7, k=1)
        expert_bytes = 6 * 64 * 64
        with self.runtime(w1, w2, banks=1, capacity=1,
                          dynamic_cache_bytes=expert_bytes) as runtime:
            def call(expert):
                routed = torch.full_like(ids, expert)
                out = runtime.forward(x, routed, scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, routed, scale),
                                           atol=.002, rtol=.04)
                return runtime.stats()

            # No further heat: at success 32, hot frequency 2 halves to 1.
            call(0)
            stats = call(0)
            self.assertEqual(stats["cache_admissions"], 1)
            for success in range(3, 33):
                stats = call(-1)
                self.assertEqual(runtime.token[1].item(), success)
                self.assertEqual(stats["cache_admissions"], 0)
                self.assertEqual(stats["cache_evictions"], 0)
                self.assertEqual(stats["cache_bypasses"], 0)
                self.assertEqual(stats["dynamic_experts"], 1)
            stats = call(1)
            self.assertEqual(stats["cache_admissions"], 0)
            self.assertEqual(stats["cache_bypasses"], 1)
            stats = call(1)
            self.assertEqual(stats["cache_hits"], 0)
            self.assertEqual(stats["cache_admissions"], 1)
            self.assertEqual(stats["cache_evictions"], 1)
            self.assertEqual(stats["cache_promotion_d2d_bytes"], expert_bytes)
            stats = call(1)
            self.assertEqual(stats["dynamic_cache_hits"], 1)
            self.assertEqual(stats["weight_h2d_bytes"], 0)

    def test_dynamic_variable_bytes_multivictim_strict_hysteresis(self):
        small = self.tensors(s=11, h=64, i=64, k=1)
        large = self.tensors(s=13, h=128, i=128, k=1, dtype=torch.bfloat16)
        cases = {0: small, 5: large}
        store = ExpertStore({layer: (case[1], case[2]) for layer, case in cases.items()})
        unit = self.charged_bytes(6 * 64 * 64)
        cap = 4 * unit
        pin = (small[1].shape[0] + 1) * 4 + 4 * unit + 24
        with PrefillMoERuntime(store, gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20,
                               pinned_budget_bytes=pin, num_banks=1,
                               dynamic_cache_bytes=cap, cache_policy="reuse") as runtime:
            def call(layer, expert):
                x, w1, w2, ids, scale = cases[layer]
                routed = torch.full_like(ids, expert)
                out = runtime.forward(x, routed, scale, layer_id=layer)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, routed, scale),
                                           atol=.002, rtol=.04)
                stats = runtime.stats()
                raw = (w1[0].numel() + w2[0].numel()) * w1.element_size()
                self.assertEqual(stats["cache_hits"], stats["dynamic_cache_hits"])
                self.assertEqual(stats["cache_misses"] + stats["cache_hits"], 1)
                self.assertEqual(stats["weight_h2d_bytes"], stats["cache_misses"] * raw)
                self.assertEqual(stats["cache_hit_weight_bytes"], stats["cache_hits"] * raw)
                self.assertEqual(stats["cache_promotion_d2d_bytes"], stats["cache_admissions"] * raw)
                self.assertLessEqual(stats["dynamic_cache_bytes"], cap)
                self.assertEqual(stats["dynamic_cache_bytes"] % 512, 0)
                self.assertEqual(stats["cache_budget_evictions"], 0)
                for key in ("resident_cache_bytes", "resident_experts", "resident_load_h2d_bytes"):
                    self.assertEqual(stats[key], 0)
                return stats

            # Four B-byte victims have frequencies [3,3,3,2], total benefit 11B.
            for expert, calls in ((0, 3), (1, 3), (2, 3), (3, 2)):
                for _ in range(calls):
                    call(0, expert)
            self.assertEqual(runtime.stats()["dynamic_experts"], 4)
            self.assertEqual(runtime.stats()["dynamic_cache_bytes"], cap)
            # Candidate is 4B bytes, so its hysteresis is B. Frequency three
            # gives 12B == 11B+B: strict comparison must NOT evict any victim.
            for frequency in (1, 2, 3, 4):
                stats = call(5, 0)
                self.assertEqual(stats["cache_hits"], 0)
                self.assertEqual(stats["cache_admissions"], int(frequency == 4))
                self.assertEqual(stats["cache_evictions"], 4 * int(frequency == 4))
                self.assertEqual(stats["cache_bypasses"], int(frequency < 4))
                self.assertEqual(stats["dynamic_experts"], 1 if frequency == 4 else 4)
                self.assertEqual(stats["dynamic_cache_bytes"], cap)
            stats = call(5, 0)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["cache_admissions"], 0)
            # Same expert ID in another layer is a miss, not an identity alias.
            stats = call(0, 0)
            self.assertEqual(stats["cache_hits"], 0)
            self.assertEqual(stats["cache_bypasses"], 1)
            self.assertEqual(stats["cache_admissions"], 0)
            self.assertEqual(stats["cache_evictions"], 0)
            self.assertEqual(stats["dynamic_experts"], 1)
            self.assertEqual(call(5, 0)["cache_hits"], 1)

    def test_dynamic_cap_below_one_expert_bypasses_without_partial_charge(self):
        x, w1, w2, ids, scale = self.tensors(s=11, h=64, i=192, k=1)
        ids.fill_(0)
        raw = (w1[0].numel() + w2[0].numel()) * w1.element_size()
        cap = self.charged_bytes(raw) - 1
        for policy in ("reuse", "lru"):
            with self.subTest(policy=policy), self.runtime(
                    w1, w2, capacity=1, dynamic_cache_bytes=cap,
                    cache_policy=policy, cache_min_reuse=2) as runtime:
                for _ in range(3):
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                               atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(stats["dynamic_cache_bytes"], 0)
                    self.assertEqual(stats["dynamic_experts"], 0)
                    self.assertEqual(stats["cache_hits"], 0)
                    self.assertEqual(stats["cache_misses"], 1)
                    self.assertEqual(stats["weight_h2d_bytes"], raw)
                    self.assertEqual(stats["cache_bypasses"], 1)
                    for key in ("cache_admissions", "cache_evictions", "cache_budget_evictions",
                                "cache_promotion_d2d_bytes"):
                        self.assertEqual(stats[key], 0)
                    del out

    def test_dynamic_fixed_resident_protection_and_aggregate_statistics(self):
        x, w1, w2, ids, scale = self.tensors(s=19, k=3)
        expert_bytes = 6 * 64 * 64
        for policy in ("reuse", "lru"):
            with self.subTest(policy=policy), self.runtime(
                    w1, w2, capacity=1, resident_cache_bytes=expert_bytes,
                    resident_experts={0: [0]}, dynamic_cache_bytes=expert_bytes,
                    cache_policy=policy) as runtime:
                def call(expert):
                    routed = torch.tensor([0, expert, expert], device="cuda", dtype=torch.int32)
                    routed = routed.expand_as(ids).contiguous()
                    out = runtime.forward(x, routed, scale)
                    torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, routed, scale),
                                               atol=.002, rtol=.04)
                    stats = runtime.stats()
                    self.assertEqual(stats["active_experts"], 2)
                    self.assertEqual(stats["resident_cache_bytes"], expert_bytes)
                    self.assertEqual(stats["resident_experts"], 1)
                    self.assertEqual(stats["resident_load_h2d_bytes"], expert_bytes)
                    self.assertEqual(stats["cache_hits"], 1 + stats["dynamic_cache_hits"])
                    self.assertEqual(stats["cache_misses"], 1 - stats["dynamic_cache_hits"])
                    self.assertEqual(stats["cache_hit_weight_bytes"], stats["cache_hits"] * expert_bytes)
                    self.assertEqual(stats["weight_h2d_bytes"], stats["cache_misses"] * expert_bytes)
                    self.assertEqual(stats["cache_promotion_d2d_bytes"], stats["cache_admissions"] * expert_bytes)
                    self.assertLessEqual(stats["dynamic_cache_bytes"], expert_bytes)
                    self.assertEqual(stats["cache_budget_evictions"], 0)
                    return stats

                for _ in range(3):
                    stats = call(1)
                self.assertEqual(stats["dynamic_cache_hits"], 1)
                self.assertEqual(stats["dynamic_experts"], 1)
                # Hot dynamic frequency is three; rival needs four, while LRU
                # replaces on its first miss, regardless of historical frequency.
                for frequency in range(1, 5):
                    stats = call(2)
                    if policy == "reuse":
                        self.assertEqual(stats["dynamic_cache_hits"], 0)
                        self.assertEqual(stats["cache_admissions"], int(frequency == 4))
                        self.assertEqual(stats["cache_evictions"], int(frequency == 4))
                        self.assertEqual(stats["cache_bypasses"], int(frequency < 4))
                    else:
                        self.assertEqual(stats["dynamic_cache_hits"], int(frequency > 1))
                        self.assertEqual(stats["cache_admissions"], int(frequency == 1))
                        self.assertEqual(stats["cache_evictions"], int(frequency == 1))
                        self.assertEqual(stats["cache_bypasses"], 0)
                    self.assertEqual(stats["dynamic_experts"], 1)
                stats = call(2)
                self.assertEqual(stats["cache_hits"], 2)
                self.assertEqual(stats["dynamic_cache_hits"], 1)
                self.assertEqual(stats["cache_admissions"], 0)
                self.assertEqual(stats["cache_evictions"], 0)

    def test_dynamic_update_bytes_limit_and_zero_means_dynamic_cap(self):
        x, w1, w2, ids, scale = self.tensors(s=17, k=3)
        ids.copy_(torch.arange(3, device="cuda", dtype=torch.int32).expand_as(ids))
        expected = reference(x, w1, w2, ids, scale)
        expert_bytes = 6 * 64 * 64
        for update_cap in (0, expert_bytes, expert_bytes - 1):
            with self.subTest(update_cap=update_cap), self.runtime(
                    w1, w2, capacity=1, dynamic_cache_bytes=3 * expert_bytes,
                    cache_update_bytes=update_cap) as runtime:
                before_experts = 0
                total_h2d = total_d2d = 0
                for call in range(1, 6):
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                    stats = runtime.stats()
                    admissions = (min(3 - before_experts,
                                      3 if update_cap == 0 else int(update_cap >= expert_bytes))
                                  if call >= 2 else 0)
                    self.assertEqual(stats["cache_hits"], before_experts)
                    self.assertEqual(stats["dynamic_cache_hits"], before_experts)
                    self.assertEqual(stats["cache_misses"], 3 - before_experts)
                    self.assertEqual(stats["weight_h2d_bytes"], (3 - before_experts) * expert_bytes)
                    self.assertEqual(stats["cache_hit_weight_bytes"], before_experts * expert_bytes)
                    self.assertEqual(stats["cache_admissions"], admissions)
                    self.assertEqual(stats["cache_bypasses"], 3 - before_experts - admissions)
                    self.assertEqual(stats["cache_promotion_d2d_bytes"], admissions * expert_bytes)
                    self.assertLessEqual(stats["cache_promotion_d2d_bytes"], update_cap or 3 * expert_bytes)
                    self.assertEqual(stats["cache_evictions"], 0)
                    self.assertEqual(stats["cache_budget_evictions"], 0)
                    before_experts += admissions
                    self.assertEqual(stats["dynamic_experts"], before_experts)
                    self.assertEqual(stats["dynamic_cache_bytes"], before_experts * expert_bytes)
                    total_h2d += stats["weight_h2d_bytes"]
                    total_d2d += stats["cache_promotion_d2d_bytes"]
                    del out
                self.assertEqual(total_h2d, {0: 6, expert_bytes: 9, expert_bytes - 1: 15}[update_cap]
                                 * expert_bytes)
                self.assertEqual(total_d2d, (0 if update_cap == expert_bytes - 1 else 3) * expert_bytes)

    def test_dynamic_promotion_headroom_never_sacrifices_banks_or_tiles(self):
        x, w1, w2, ids, scale = self.tensors(s=37, k=2)
        ids.copy_(torch.arange(2, device="cuda", dtype=torch.int32).expand_as(ids))
        expected = reference(x, w1, w2, ids, scale)
        expert_bytes = 6 * 64 * 64
        for mode in ("route_buffer", "fp32_accumulate"):
            with self.subTest(mode=mode):
                with self.runtime(w1, w2, banks=2, capacity=1, activation_mode=mode,
                                  max_route_tile_rows=8, dynamic_cache_bytes=expert_bytes,
                                  cache_min_reuse=1000) as probe:
                    out = probe.forward(x, ids, scale)
                    baseline = probe.stats()
                    budget = baseline["gpu_charged_bytes"]
                    del out
                self.assertEqual(baseline["num_banks"], 2)
                self.assertEqual(baseline["wave_capacity"], 1)
                # Less than one aligned expert of actual headroom. A planner
                # must not choose one bank/smaller tiles merely to promote.
                with self.runtime(w1, w2, banks=2, capacity=1, gpu_budget_bytes=budget + expert_bytes - 512,
                                  activation_mode=mode, max_route_tile_rows=8,
                                  dynamic_cache_bytes=expert_bytes) as runtime:
                    for _ in range(3):
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        self.assertEqual(stats["num_banks"], baseline["num_banks"])
                        self.assertEqual(stats["wave_capacity"], baseline["wave_capacity"])
                        self.assertEqual(stats["route_tile_rows"], baseline["route_tile_rows"])
                        self.assertEqual(stats["gpu_pool_bytes"], baseline["gpu_pool_bytes"])
                        self.assertEqual(stats["cache_hits"], 0)
                        self.assertEqual(stats["cache_misses"], 2)
                        self.assertEqual(stats["weight_h2d_bytes"], 2 * expert_bytes)
                        self.assertEqual(stats["cache_bypasses"], 2)
                        for key in ("dynamic_cache_bytes", "dynamic_experts", "cache_admissions",
                                    "cache_evictions", "cache_budget_evictions", "cache_promotion_d2d_bytes"):
                            self.assertEqual(stats[key], 0)
                        self.assertLessEqual(stats["gpu_charged_bytes"], budget + expert_bytes - 512)
                        del out

    def test_dynamic_large_s_budget_eviction_tile_fallback_and_oom_floor(self):
        x, w1, w2, ids, scale = self.tensors(s=257, k=4)
        ids.copy_(torch.arange(4, device="cuda", dtype=torch.int32).expand_as(ids))
        warm_ids = torch.full_like(ids[:7], 7)
        expected = reference(x, w1, w2, ids, scale)
        expert_bytes = 6 * 64 * 64
        with self.runtime(w1, w2, banks=1, capacity=1, activation_mode="fp32_accumulate",
                          max_route_tile_rows=8, dynamic_cache_bytes=expert_bytes,
                          cache_min_reuse=1000) as probe:
            out = probe.forward(x, ids, scale)
            baseline = probe.stats()
            budget = baseline["gpu_charged_bytes"]
            del out
        self.assertEqual(baseline["route_tile_rows"], 8)
        for mode in ("fp32_accumulate", "auto", "route_buffer"):
            with self.subTest(mode=mode), self.runtime(
                    w1, w2, banks=1, capacity=1, gpu_budget_bytes=budget,
                    activation_mode=mode, max_route_tile_rows=4096,
                    dynamic_cache_bytes=expert_bytes) as runtime:
                for _ in range(3):
                    out = runtime.forward(x[:7], warm_ids, scale[:7])
                    torch.testing.assert_close(out.float().cpu(),
                        reference(x[:7], w1, w2, warm_ids, scale[:7]), atol=.002, rtol=.04)
                    del out
                self.assertEqual(runtime.stats()["dynamic_experts"], 1)
                self.assertEqual(runtime.stats()["dynamic_cache_bytes"], expert_bytes)
                if mode == "route_buffer":
                    # Even evicting all dynamic storage cannot fit full-route
                    # activation/output scratch at the FP32-tiled byte floor.
                    counter = runtime.token[1].item()
                    with self.assertRaisesRegex(RuntimeError, "gpu_budget|budgets|history"):
                        runtime.forward(x, ids, scale)
                    self.assertEqual(runtime.token[1].item(), counter)
                    self.assertEqual(runtime.stats()["dynamic_experts"], 1)
                    self.assertEqual(runtime.stats()["dynamic_cache_bytes"], expert_bytes)
                    continue
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["activation_mode"], 1)
                self.assertGreater(stats["route_tile_rows"], 0)
                self.assertLessEqual(stats["route_tile_rows"], 8)
                self.assertGreater(stats["route_tiles"], stats["waves"])
                self.assertEqual(stats["cache_budget_evictions"], 1)
                self.assertEqual(stats["cache_evictions"], 1)
                self.assertEqual(stats["dynamic_experts"], 0)
                self.assertEqual(stats["dynamic_cache_bytes"], 0)
                self.assertEqual(stats["cache_hits"], 0)
                self.assertEqual(stats["cache_misses"], 4)
                self.assertEqual(stats["weight_h2d_bytes"], 4 * expert_bytes)
                self.assertEqual(stats["cache_admissions"], 0)
                self.assertEqual(stats["cache_bypasses"], 4)
                self.assertEqual(stats["cache_promotion_d2d_bytes"], 0)
                self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                del out
                # These are last-call eviction counters, not lifetime totals.
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["cache_budget_evictions"], 0)
                self.assertEqual(stats["cache_evictions"], 0)
                self.assertEqual(stats["cache_admissions"], 0)
                self.assertEqual(stats["cache_promotion_d2d_bytes"], 0)
                self.assertEqual(stats["cache_bypasses"], 4)
                self.assertEqual(stats["weight_h2d_bytes"], 4 * expert_bytes)
                del out
        # Static experts cannot be sacrificed to rescue the identical pressure.
        with self.runtime(w1, w2, banks=1, capacity=1, gpu_budget_bytes=budget,
                          activation_mode="fp32_accumulate", max_route_tile_rows=4096,
                          resident_cache_bytes=expert_bytes, resident_experts={0: [7]},
                          dynamic_cache_bytes=expert_bytes) as runtime:
            with self.assertRaisesRegex(RuntimeError, "gpu_budget|budgets|history"):
                runtime.forward(x, ids, scale)
            self.assertEqual(runtime.stats()["resident_experts"], 1)
            self.assertEqual(runtime.stats()["resident_cache_bytes"], expert_bytes)

    def test_dynamic_invalid_ids_do_not_heat_admit_or_advance_decay(self):
        x, w1, w2, ids, scale = self.tensors(s=19, k=2)
        ids.fill_(1)
        expert_bytes = 6 * 64 * 64
        with self.runtime(w1, w2, banks=1, capacity=1, dynamic_cache_bytes=2 * expert_bytes,
                          cache_decay_interval=4) as runtime:
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            del out
            self.assertEqual(runtime.stats()["cache_admissions"], 0)
            self.assertEqual(runtime.token[1].item(), 1)
            for bad in (-2, w1.shape[0], -2):
                invalid = ids.clone()
                invalid[0, 0] = bad
                invalid[1, 0] = 0
                # Valid routes to experts 0 and 1 must not commit any heat.
                # Three failures would also shift a buggy interval-4 clock.
                version = runtime.token._version
                with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                    runtime.forward(x, invalid, scale)
                self.assertEqual(runtime.token[1].item(), 1)
                self.assertEqual(runtime.token._version, version)
                self.assertEqual(runtime.stats()["dynamic_experts"], 0)
                self.assertEqual(runtime.stats()["dynamic_cache_bytes"], 0)
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            del out
            stats = runtime.stats()
            self.assertEqual(runtime.token[1].item(), 2)
            self.assertEqual(stats["cache_hits"], 0)
            self.assertEqual(stats["cache_admissions"], 1)
            self.assertEqual(stats["cache_promotion_d2d_bytes"], expert_bytes)
            self.assertEqual(stats["dynamic_experts"], 1)
            for rows in (19, 0):
                empty_ids = torch.full_like(ids[:rows], -1)
                out = runtime.forward(x[:rows], empty_ids, scale[:rows])
                torch.testing.assert_close(out, torch.zeros_like(x[:rows]), atol=0, rtol=0)
                stats = runtime.stats()
                self.assertEqual(stats["dynamic_experts"], 1)
                for key in ("active_experts", "cache_hits", "cache_misses", "dynamic_cache_hits",
                            "cache_admissions", "cache_evictions", "cache_bypasses",
                            "cache_budget_evictions", "cache_promotion_d2d_bytes", "weight_h2d_bytes"):
                    self.assertEqual(stats[key], 0, key)
                del out
            ids.fill_(0)
            # Expert zero was never heated by -1 padding or failed calls.
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["cache_admissions"], 0)
            self.assertEqual(runtime.stats()["cache_bypasses"], 1)
            del out
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["cache_admissions"], 1)
            self.assertEqual(runtime.stats()["dynamic_experts"], 2)

    def test_dynamic_full_cache_invalid_or_infeasible_requests_preserve_entries(self):
        x, w1, w2, ids, scale = self.tensors(s=300, k=1)
        expert_bytes = 6 * 64 * 64
        with self.runtime(w1, w2, banks=1, gpu_budget_bytes=80000,
                          dynamic_cache_bytes=2 * expert_bytes, cache_policy="lru") as runtime:
            for expert in (0, 1):
                ids.fill_(expert)
                out = runtime.forward(x[:7], ids[:7], scale[:7])
                torch.testing.assert_close(out.float().cpu(),
                    reference(x[:7], w1, w2, ids[:7], scale[:7]), atol=.002, rtol=.04)
                del out
            before = runtime.stats()
            self.assertEqual(before["dynamic_experts"], 2)
            counter, version = runtime.token[1].item(), runtime.token._version
            for expert in (w1.shape[0], -2, 0):
                ids.fill_(expert)
                message = "topk_ids" if expert != 0 else "budgets"
                # Routing/output alone needs reclamation. A valid but impossible
                # full-route workspace must also fail before committing victims.
                with self.assertRaisesRegex(RuntimeError, message):
                    runtime.forward(x, ids, scale)
                self.assertEqual(runtime.stats(), before)
                self.assertEqual(runtime.token[1].item(), counter)
                self.assertEqual(runtime.token._version, version)
            for expert in (0, 1):
                ids.fill_(expert)
                out = runtime.forward(x[:7], ids[:7], scale[:7])
                torch.testing.assert_close(out.float().cpu(),
                    reference(x[:7], w1, w2, ids[:7], scale[:7]), atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["dynamic_cache_hits"], 1)
                self.assertEqual(stats["weight_h2d_bytes"], 0)
                self.assertEqual(stats["dynamic_experts"], 2)
                self.assertEqual(stats["cache_budget_evictions"], 0)
                del out

    def test_dynamic_preflight_max_experts_with_tight_live_output_history(self):
        x, w1, w2, ids, scale = self.tensors(s=1, e=1, k=1)
        other_x, other_w1, other_w2, other_ids, other_scale = self.tensors(s=1, e=512, k=1)
        ids.zero_()
        expert_bytes = 6 * 64 * 64
        store = ExpertStore({0: (w1, w2), 1: (other_w1, other_w2)})
        with PrefillMoERuntime(store, gpu_budget_bytes=80000,
                              ram_budget_bytes=64 << 20, pinned_budget_bytes=50000,
                              num_banks=1, dynamic_cache_bytes=expert_bytes,
                              cache_policy="lru") as runtime:
            out = runtime.forward(x, ids, scale)
            del out
            saved = [runtime.forward(x, ids, scale) for _ in range(104)]
            self.assertGreater(runtime.stats()["gpu_charged_bytes"], 79000)
            before = runtime.stats()
            for invalid in (-2, 512):
                other_ids.fill_(invalid)
                with self.assertRaisesRegex(RuntimeError, "topk_ids"):
                    runtime.forward(other_x, other_ids, other_scale, layer_id=1)
                self.assertEqual(runtime.stats(), before)
            # Full 513-int histogram cannot fit before eviction. A segmented
            # one-quantum preflight can validate all 512 IDs without eviction.
            other_ids.fill_(-1)
            out = runtime.forward(other_x, other_ids, other_scale, layer_id=1)
            torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
            stats = runtime.stats()
            self.assertEqual(stats["active_experts"], 0)
            self.assertEqual(stats["cache_budget_evictions"], 1)
            self.assertEqual(stats["dynamic_experts"], 0)
            self.assertLessEqual(stats["gpu_charged_bytes"], 80000)
            expected = reference(x, w1, w2, ids, scale)
            for snapshot in saved:
                torch.testing.assert_close(snapshot.float().cpu(), expected, atol=.002, rtol=.04)

    def test_dynamic_preflight_histogram_slice_boundaries(self):
        x, w1, w2, ids, scale = self.tensors(s=1, h=128, i=128, e=1, k=1)
        other_x, other_w1, other_w2, other_ids, other_scale = self.tensors(s=7, e=512, k=7)
        ids.zero_()
        boundaries = (126, 127, 253, 254, 507, 508, 511)
        other_ids.copy_(torch.tensor(boundaries, dtype=torch.int32, device="cuda").expand_as(other_ids))
        store = ExpertStore({0: (w1, w2), 1: (other_w1, other_w2)})
        # Keep live outputs plus one larger cached expert. The next full Routes
        # needs reclamation, but smaller misses fit after evicting the large key.
        with PrefillMoERuntime(store, gpu_budget_bytes=220000,
                              ram_budget_bytes=64 << 20, pinned_budget_bytes=100000,
                              num_banks=1, dynamic_cache_bytes=6 * 128 * 128,
                              cache_policy="lru", activation_mode="fp32_accumulate",
                              max_route_tile_rows=1) as runtime:
            out = runtime.forward(x, ids, scale)
            self.assertEqual(runtime.stats()["dynamic_experts"], 1)
            del out
            saved = [runtime.forward(x, ids, scale) for _ in range(220)]
            out = runtime.forward(other_x, other_ids, other_scale, layer_id=1)
            torch.testing.assert_close(out.float().cpu(),
                reference(other_x, other_w1, other_w2, other_ids, other_scale), atol=.002, rtol=.04)
            stats = runtime.stats()
            self.assertEqual(stats["active_experts"], len(boundaries))
            self.assertEqual(stats["cache_misses"], len(boundaries))
            self.assertEqual(stats["cache_budget_evictions"], 1)
            self.assertEqual(stats["weight_h2d_bytes"], len(boundaries) * 6 * 64 * 64)
            self.assertLessEqual(stats["gpu_charged_bytes"], 220000)
            expected = reference(x, w1, w2, ids, scale)
            for snapshot in saved:
                torch.testing.assert_close(snapshot.float().cpu(), expected, atol=.002, rtol=.04)

    def test_dynamic_partial_reclaim_preserves_hit_when_full_reclaim_exceeds_ram(self):
        x, w1, w2, ids, scale = self.tensors(s=180, e=2, k=1)
        extra = (w1[:1].clone(), w2[:1].clone())
        expert_bytes = 6 * 64 * 64
        for policy in ("reuse", "lru"):
            options = dict(banks=1, gpu_budget_bytes=100000, pinned_budget_bytes=50000,
                           dynamic_cache_bytes=2 * expert_bytes, cache_policy=policy)
            measured = {}
            ids.fill_(0)
            # Identical input/layout and ample RAM isolate the bounded worker
            # reservation, rather than guessing a larger tight-test budget.
            for async_staging in (False, True):
                with mock.patch.dict(os.environ, {
                        "FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0" if async_staging else "1"}):
                    with self.runtime(w1, w2, ram_budget_bytes=64 << 20,
                                      async_staging=async_staging, **options) as probe:
                        out = probe.forward(x[:7], ids[:7], scale[:7])
                        torch.testing.assert_close(out.float().cpu(),
                            reference(x[:7], w1, w2, ids[:7], scale[:7]), atol=.002, rtol=.04)
                        measured[async_staging] = probe.stats()
                        del out
            reserved_delta = measured[True]["ram_charged_bytes"] - measured[False]["ram_charged_bytes"]
            self.assertGreater(reserved_delta, 1 << 20)
            for key in measured[False]:
                if key not in ("ram_charged_bytes", "buffer_allocations", "buffer_reuses"):
                    self.assertEqual(measured[False][key], measured[True][key], key)
            for async_staging in (False, True):
                reservation = reserved_delta if async_staging else 0
                budget = 99728 + reservation
                self.assertEqual(budget - reservation, 99728)
                with self.subTest(policy=policy, async_staging=async_staging), mock.patch.dict(
                        os.environ, {"FUSE_MOE_OFFLOAD_ASYNC_STAGING": "0" if async_staging else "1"}):
                    with self.runtime(w1, w2, ram_budget_bytes=budget,
                                      async_staging=async_staging, **options) as runtime:
                        for expert in (0, 1):
                            ids.fill_(expert)
                            for _ in range(2):
                                out = runtime.forward(x[:7], ids[:7], scale[:7])
                                torch.testing.assert_close(out.float().cpu(),
                                    reference(x[:7], w1, w2, ids[:7], scale[:7]), atol=.002, rtol=.04)
                                del out
                        self.assertEqual(runtime.stats()["dynamic_experts"], 2)
                        # A legal additional CPU binding tightens RAM without changing
                        # the current layer's hit. Reloading that hit would need staging
                        # which exceeds RAM; evicting only its inactive peer is feasible.
                        torch.ops.recsys_ops.fuse_moe_offload_bind(runtime.token, 5, *extra)
                        ids.fill_(0)
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                                   atol=.002, rtol=.04)
                        stats = runtime.stats()
                        self.assertEqual(stats["dynamic_cache_hits"], 1)
                        self.assertEqual(stats["weight_h2d_bytes"], 0)
                        self.assertEqual(stats["dynamic_experts"], 1)
                        self.assertEqual(stats["cache_budget_evictions"], 1)
                        self.assertEqual(stats["cache_evictions"], 1)
                        self.assertEqual(stats["num_banks"], 0)
                        self.assertLessEqual(stats["gpu_charged_bytes"], 100000)
                        self.assertLessEqual(stats["ram_charged_bytes"], budget)
                        del out
                        out = runtime.forward(x, ids, scale)
                        self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
                        self.assertEqual(runtime.stats()["cache_budget_evictions"], 0)
                        del out

    def test_dynamic_existing_identity_version_and_route_validation(self):
        expert_bytes = 6 * 64 * 64
        for weight in (0, 1):
            with self.subTest(weight=weight):
                x, w1, w2, ids, scale = self.tensors(s=7, k=2)
                ids.fill_(0)
                with self.runtime(w1, w2, capacity=1, dynamic_cache_bytes=expert_bytes) as runtime:
                    for _ in range(3):
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                                   atol=.002, rtol=.04)
                        del out
                    self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
                    native = torch.ops.recsys_ops.fuse_moe_offload
                    alias1, alias2 = w1.data, w2.data
                    out = native(x, alias1, alias2, ids, scale, runtime.token, 0)
                    torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                               atol=.002, rtol=.04)
                    del out
                    self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
                    counter, version = runtime.token[1].item(), runtime.token._version
                    for bad_token, bad_w1, bad_w2 in (
                            (runtime.token.clone(), w1, w2),
                            (runtime.token, w1.clone(), w2),
                            (runtime.token, w1, w2.clone())):
                        with self.assertRaisesRegex(RuntimeError, "identity|version"):
                            native(x, bad_w1, bad_w2, ids, scale, bad_token, 0)
                    for bad_x, bad_ids, bad_scale in (
                            (x.T.contiguous().T, ids, scale),
                            (x, ids.long(), scale), (x, ids, scale.half()),
                            (x, ids.T.contiguous().T, scale),
                            (x, ids, scale.T.contiguous().T),
                            (x.clone().requires_grad_(), ids, scale),
                            (x, ids, scale.clone().requires_grad_())):
                        with self.assertRaises(RuntimeError):
                            runtime.forward(bad_x, bad_ids, bad_scale)
                    self.assertEqual(runtime.token[1].item(), counter)
                    self.assertEqual(runtime.token._version, version)
                    self.assertEqual(runtime.stats()["dynamic_experts"], 1)
                    (w1, w2)[weight].add_(1)
                    # .data aliases have independent version counters; the
                    # bound canonical version must still reject stale cache.
                    with self.assertRaisesRegex(RuntimeError, "version"):
                        native(x, alias1, alias2, ids, scale, runtime.token, 0)
                    with self.assertRaisesRegex(RuntimeError, "version"):
                        runtime.forward(x, ids, scale)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    runtime.forward(x, ids, scale)

    def test_dynamic_cross_stream_grouped_wave_promotion_and_live_output(self):
        expert_bytes = 6 * 64 * 64
        producer = torch.cuda.current_stream()
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=37, k=4, dtype=dtype)
            ids.copy_(torch.arange(ids.numel(), device="cuda", dtype=torch.int32).reshape_as(ids).remainder(8))
            scale.copy_(torch.linspace(-1, 1, scale.numel(), device="cuda").reshape_as(scale))
            hot_ids = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32).expand_as(ids).contiguous()
            for mode in ("route_buffer", "fp32_accumulate"):
                for banks in (1, 2):
                    with self.subTest(dtype=dtype, mode=mode, banks=banks), self.runtime(
                            w1, w2, banks=banks, capacity=3, activation_mode=mode,
                            max_route_tile_rows=13, dynamic_cache_bytes=2 * expert_bytes,
                            cache_min_reuse=2, cache_update_bytes=0) as runtime:
                        saved = []
                        for index, routed in enumerate((ids, ids, hot_ids)):
                            stream = streams[index % len(streams)]
                            stream.wait_stream(producer)
                            # Input producer ordering is required, but there is
                            # intentionally no wait on the prior runtime stream:
                            # forward's internal drain owns cache/pool safety.
                            with torch.cuda.stream(stream), torch.inference_mode():
                                out = runtime.forward(x, routed, scale)
                                self.assertTrue(stream.query(), "forward must drain promotion D2D too")
                                saved.append((out, out.clone()))
                            producer.wait_stream(stream)
                            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, routed, scale),
                                                       atol=.002, rtol=.04)
                            stats = runtime.stats()
                            self.assertEqual(stats["cache_hits"], 2 if index == 2 else 0)
                            self.assertEqual(stats["dynamic_cache_hits"], 2 if index == 2 else 0)
                            self.assertEqual(stats["cache_misses"], 0 if index == 2 else 8)
                            self.assertEqual(stats["weight_h2d_bytes"], 0 if index == 2 else 8 * expert_bytes)
                            self.assertEqual(stats["cache_admissions"], 2 if index == 1 else 0)
                            self.assertEqual(stats["cache_bypasses"], (8, 6, 0)[index])
                            self.assertEqual(stats["cache_promotion_d2d_bytes"],
                                             2 * expert_bytes if index == 1 else 0)
                            self.assertEqual(stats["cache_evictions"], 0)
                            self.assertEqual(stats["cache_budget_evictions"], 0)
                            self.assertEqual(stats["dynamic_experts"], 2 if index else 0)
                            self.assertEqual(stats["dynamic_cache_bytes"], 2 * expert_bytes if index else 0)
                            self.assertEqual(runtime.token[1].item(), index + 1)
                            if index == 2:
                                self.assertEqual(stats["waves"], 1)
                                self.assertEqual(stats["num_banks"], 0)
                                self.assertEqual(stats["wave_capacity"], 0)
                            else:
                                self.assertEqual(stats["num_banks"], banks)
                                self.assertEqual(stats["wave_capacity"], 3)
                                self.assertEqual(stats["waves"], 3)
                        runtime.close()
                        for out, snapshot in saved:
                            torch.testing.assert_close(out, snapshot, atol=0, rtol=0)
                        runtime.close()
                        with self.assertRaisesRegex(RuntimeError, "closed"):
                            runtime.stats()

    def test_dynamic_irreparable_cross_layer_pinned_pressure_preserves_hot_cache(self):
        x, w1, w2, ids, scale = self.tensors(s=19, k=1)
        large_x, large_w1, large_w2, large_ids, large_scale = self.tensors(
            s=19, h=128, i=128, e=8, k=1)
        ids.fill_(0)
        large_ids.fill_(0)
        expert_bytes = (w1[0].numel() + w2[0].numel()) * w1.element_size()
        large_expert_bytes = (large_w1[0].numel() + large_w2[0].numel()) * large_w1.element_size()
        pin = (w1.shape[0] + 1) * 4 + expert_bytes + 24
        self.assertGreater(large_expert_bytes, pin)
        store = ExpertStore({0: (w1, w2), 1: (large_w1, large_w2)})
        expected = reference(x, w1, w2, ids, scale)
        with PrefillMoERuntime(
                store, gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20,
                pinned_budget_bytes=pin, num_banks=1,
                dynamic_cache_bytes=expert_bytes) as runtime:
            for _ in range(3):
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                del out
            before = runtime.stats()
            self.assertEqual(before["dynamic_cache_hits"], 1)
            self.assertEqual(before["weight_h2d_bytes"], 0)
            self.assertEqual(before["dynamic_cache_bytes"], expert_bytes)
            self.assertEqual(before["dynamic_experts"], 1)
            counter, version = runtime.token[1].item(), runtime.token._version
            with self.assertRaisesRegex(RuntimeError, "pinned|budgets"):
                runtime.forward(large_x, large_ids, large_scale, layer_id=1)
            self.assertEqual(runtime.token[1].item(), counter)
            self.assertEqual(runtime.token._version, version)
            failed = runtime.stats()
            for key in ("dynamic_cache_bytes", "dynamic_experts"):
                self.assertEqual(failed[key], before[key], key)
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
            stats = runtime.stats()
            self.assertEqual(stats["dynamic_cache_hits"], 1)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["weight_h2d_bytes"], 0)
            self.assertEqual(stats["dynamic_cache_bytes"], expert_bytes)
            self.assertEqual(stats["dynamic_experts"], 1)
            for key in ("cache_admissions", "cache_evictions", "cache_budget_evictions"):
                self.assertEqual(stats[key], 0, key)

    def test_dynamic_lru_promotion_preserves_future_high_id_hit_lease(self):
        x, w1, w2, ids, scale = self.tensors(s=37, k=2)
        ids.copy_(torch.tensor([0, 7], device="cuda", dtype=torch.int32).expand_as(ids))
        warm_ids = torch.full_like(ids, 7)
        expert_bytes = (w1[0].numel() + w2[0].numel()) * w1.element_size()
        expected = reference(x, w1, w2, ids, scale)
        warm_expected = reference(x, w1, w2, warm_ids, scale)
        # Verify the budgets really permit grouping two cold experts. A hit/miss
        # boundary, not an undersized staging bank, must split the lease case.
        with self.runtime(w1, w2, banks=1, capacity=2,
                          dynamic_cache_bytes=expert_bytes, cache_min_reuse=1000) as probe:
            out = probe.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
            self.assertGreaterEqual(probe.stats()["wave_capacity"], 2)
            del out
        with self.runtime(w1, w2, banks=1, capacity=2,
                          dynamic_cache_bytes=expert_bytes, cache_policy="lru") as runtime:
            out = runtime.forward(x, warm_ids, scale)
            torch.testing.assert_close(out.float().cpu(), warm_expected, atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["cache_admissions"], 1)
            self.assertEqual(runtime.stats()["dynamic_experts"], 1)
            del out
            # ID 0 executes/promotes before the planned resident wave for ID 7.
            # Its miss must bypass rather than evict that future wave's pointer.
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
            stats = runtime.stats()
            self.assertEqual(stats["active_experts"], 2)
            self.assertEqual(stats["waves"], 2)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["dynamic_cache_hits"], 1)
            self.assertEqual(stats["cache_misses"], 1)
            self.assertEqual(stats["cache_bypasses"], 1)
            self.assertEqual(stats["weight_h2d_bytes"], expert_bytes)
            self.assertEqual(stats["cache_hit_weight_bytes"], expert_bytes)
            self.assertEqual(stats["dynamic_cache_bytes"], expert_bytes)
            self.assertEqual(stats["dynamic_experts"], 1)
            for key in ("cache_admissions", "cache_evictions", "cache_budget_evictions",
                        "cache_promotion_d2d_bytes"):
                self.assertEqual(stats[key], 0, key)
            del out
            out = runtime.forward(x, warm_ids, scale)
            torch.testing.assert_close(out.float().cpu(), warm_expected, atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["dynamic_cache_hits"], 1)
            self.assertEqual(runtime.stats()["weight_h2d_bytes"], 0)
            self.assertEqual(runtime.stats()["dynamic_experts"], 1)

    def test_resident_grouped_pointer_tables_and_tile_boundaries(self):
        # Nonadjacent canonical IDs form one compact wave; each cached weight
        # has its own allocation. Odd group counts also exercise table alignment.
        for dtype in (torch.float16, torch.bfloat16):
            for h, intermediate, s in ((64, 64, 13), (128, 192, 65), (128, 128, 1025)):
                x, w1, w2, ids, scale = self.tensors(s=s, h=h, i=intermediate, e=8, k=5, dtype=dtype)
                ids.copy_(torch.tensor([1, 3, 7, 3, -1], device=x.device, dtype=torch.int32).expand_as(ids))
                scale.copy_(torch.linspace(-1, 1, scale.numel(), device=x.device).reshape_as(scale))
                expected = reference(x, w1, w2, ids, scale)
                cache_bytes = 3 * 6 * h * intermediate
                for mode in ("route_buffer", "fp32_accumulate", "auto"):
                    with self.subTest(dtype=dtype, h=h, intermediate=intermediate, s=s, mode=mode), self.runtime(
                            w1, w2, activation_mode=mode, max_route_tile_rows=17,
                            pinned_budget_bytes=4 * (w1.shape[0] + 1),
                            resident_cache_bytes=cache_bytes, resident_experts={0: [7, 1, 3]}) as runtime:
                        for _ in range(2):
                            out = runtime.forward(x, ids, scale)
                            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                            stats = runtime.stats()
                            self.assertEqual(stats["waves"], 1)
                            self.assertEqual(stats["cache_hits"], 3)
                            self.assertEqual(stats["weight_h2d_bytes"], 0)
                            self.assertEqual(stats["metadata_h2d_bytes"], 0)
                            self.assertEqual(stats["num_banks"], 0)
                            self.assertEqual(stats["pinned_bytes"], 4 * (w1.shape[0] + 1))
                            if stats["activation_mode"] == 1:
                                self.assertEqual(stats["route_tiles"], (s * 4 + 16) // 17)
                            else:
                                self.assertEqual(stats["route_tiles"], 1)
                            del out

    def test_resident_tile_compact_expert_intervals(self):
        # The first tile crosses three experts; later tiles begin at a nonzero
        # pointer-table offset. Exact boundaries and single-expert tails must
        # select their direct weights rather than the wave's first expert.
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=19, h=128, i=192, e=8, k=4, dtype=dtype)
            ids.fill_(-1)
            ids[0, 0] = 1
            ids[:3, 1] = 3
            ids[:, 2] = 5
            ids[:, 3] = 7
            scale.copy_(torch.linspace(-.5, .5, scale.numel(), device=x.device).reshape_as(scale))
            expected = reference(x, w1, w2, ids, scale)
            expert_bytes = 6 * 128 * 192
            for policy in ("static", "reuse", "lru"):
                for tile_rows in (1, 4, 7, 23, 42):
                    options = (dict(resident_cache_bytes=4 * expert_bytes,
                                    resident_experts={0: [7, 1, 5, 3]}) if policy == "static" else
                               dict(dynamic_cache_bytes=4 * expert_bytes, cache_policy=policy))
                    with self.subTest(dtype=dtype, policy=policy, tile_rows=tile_rows), self.runtime(
                            w1, w2, activation_mode="fp32_accumulate",
                            max_route_tile_rows=tile_rows, **options) as runtime:
                        for stage in range(4):
                            out = runtime.forward(x, ids, scale)
                            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                            stats = runtime.stats()
                            if policy == "static" or stage >= (2 if policy == "reuse" else 1):
                                self.assertEqual(stats["waves"], 1)
                                self.assertEqual(stats["cache_hits"], 4)
                                self.assertEqual(stats["route_tiles"], (42 + tile_rows - 1) // tile_rows)
                                self.assertEqual(stats["weight_h2d_bytes"], 0)
                                self.assertEqual(stats["metadata_h2d_bytes"], 0)
                            del out

    def test_resident_tile_intervals_reset_after_miss_and_group_limit(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(s=34, e=40, k=3, dtype=dtype)
            ids.copy_(torch.arange(34, device=x.device, dtype=torch.int32)[:, None].expand_as(ids))
            expected = reference(x, w1, w2, ids, scale)
            for banks in (1, 2):
                with self.subTest(dtype=dtype, banks=banks), self.runtime(
                        w1, w2, banks=banks, activation_mode="fp32_accumulate", max_route_tile_rows=4,
                        resident_cache_bytes=33 * 6 * 64 * 64, resident_experts={0: range(1, 34)}) as runtime:
                    for _ in range(2):
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        # One leading miss, 32 cached experts, then a direct-weight
                        # cached tail. Wave-local prefixes reset at global row 99.
                        self.assertEqual(stats["waves"], 3)
                        self.assertEqual(stats["route_tiles"], 26)
                        self.assertEqual(stats["cache_hits"], 33)
                        self.assertEqual(stats["weight_h2d_bytes"], 6 * 64 * 64)
                        self.assertEqual(stats["metadata_h2d_bytes"], 24)
                        del out

    def test_resident_grouped_budget_fallback_without_eviction(self):
        x, w1, w2, ids, scale = self.tensors(s=65, e=8, k=4)
        ids.copy_(torch.arange(4, device=x.device, dtype=torch.int32).expand_as(ids))
        expected = reference(x, w1, w2, ids, scale)
        cache_bytes = 4 * 6 * 64 * 64
        # Original single-expert floor: compact routing + full down buffer +
        # one expert's act scratch + one 24-byte resident metadata allocation.
        route_bytes = 8 * ids.numel() + (4 * w1.shape[0] + 3) * 4
        output_bytes = self.charged_bytes(x.numel() * x.element_size())
        budget = (cache_bytes + output_bytes + self.charged_bytes(route_bytes)
                  + self.charged_bytes(ids.numel() * 64 * 2)
                  + self.charged_bytes(x.shape[0] * 64 * 2) + 512)
        with self.runtime(w1, w2, gpu_budget_bytes=budget, pinned_budget_bytes=36,
                          resident_cache_bytes=cache_bytes, resident_experts={0: range(4)}) as runtime:
            for _ in range(2):
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["waves"], 4)
                self.assertEqual(stats["cache_hits"], 4)
                self.assertEqual(stats["weight_h2d_bytes"], 0)
                self.assertEqual(stats["cache_evictions"], 0)
                self.assertLessEqual(stats["gpu_charged_bytes"], budget)
                del out

    def test_resident_grouped_max_experts_and_reused_metadata(self):
        x, w1, w2, ids, scale = self.tensors(s=8, e=512, k=128)
        ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32).reshape_as(ids).remainder(512))
        expected = reference(x, w1, w2, ids, scale)
        cache_bytes = 512 * 6 * 64 * 64
        with self.runtime(w1, w2, gpu_budget_bytes=32 << 20, pinned_budget_bytes=4 * 513,
                          resident_cache_bytes=cache_bytes, resident_experts={0: range(512)}) as runtime:
            out = runtime.forward(x, ids, scale)
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
            self.assertEqual(runtime.stats()["waves"], 16)
            self.assertEqual(runtime.stats()["cache_hits"], 512)
            del out
            for count in (3, 1, 31, 32):
                selected = torch.arange(count, device=x.device, dtype=torch.int32)[None, :]
                selected = selected.expand(x.shape[0], count).contiguous()
                selected_scale = torch.full_like(selected, 1.0 / count, dtype=torch.float32)
                out = runtime.forward(x, selected, selected_scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, selected, selected_scale),
                                           atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["waves"], 1)
                self.assertEqual(stats["cache_hits"], count)
                self.assertEqual(stats["metadata_h2d_bytes"], 0)
                del out

    def test_dynamic_grouped_future_hit_leases_during_lru_promotion(self):
        x, w1, w2, ids, scale = self.tensors(s=19, e=8, k=4)
        expert_bytes = 6 * 64 * 64
        hot = torch.tensor([5, 7, 5, 7], device=x.device, dtype=torch.int32).expand_as(ids).contiguous()
        ids.copy_(torch.tensor([0, 5, 7, 0], device=x.device, dtype=torch.int32).expand_as(ids))
        for mode in ("route_buffer", "fp32_accumulate"):
            with self.subTest(mode=mode), self.runtime(w1, w2, capacity=2,
                    activation_mode=mode, max_route_tile_rows=13, dynamic_cache_bytes=2 * expert_bytes,
                    cache_policy="lru") as runtime:
                out = runtime.forward(x, hot, scale)
                self.assertEqual(runtime.stats()["cache_admissions"], 2)
                del out
                # Low-ID miss executes first; both future grouped pointers must
                # stay leased, even though LRU would otherwise replace either.
                out = runtime.forward(x, ids, scale)
                torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                           atol=.002, rtol=.04)
                stats = runtime.stats()
                self.assertEqual(stats["waves"], 2)
                self.assertEqual(stats["dynamic_cache_hits"], 2)
                self.assertEqual(stats["cache_bypasses"], 1)
                self.assertEqual(stats["cache_evictions"], 0)
                self.assertEqual(stats["cache_admissions"], 0)
                del out

    def test_static_bind_current_stream_pending_allocation_reuse_ordering(self):
        sleep = getattr(torch.cuda, "_sleep", None)
        if not callable(sleep):
            self.skipTest("torch.cuda._sleep unavailable")
        x, w1, w2, ids, scale = self.tensors(s=19, k=1)
        ids.fill_(0)
        expert_bytes = (w1[0].numel() + w2[0].numel()) * w1.element_size()
        expected = reference(x, w1, w2, ids, scale)
        producer = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(producer)
        # Bind CPU weights only: resident initialization must occur after junk
        # is freed, with no intervening synchronization of its pending writes.
        with self.runtime(w1, w2, resident_cache_bytes=expert_bytes) as runtime:
            self.assertEqual(runtime.stats()["resident_experts"], 0)
            with torch.cuda.stream(stream):
                junk = torch.empty(expert_bytes, dtype=torch.uint8, device=x.device)
                try:
                    sleep(10000000)
                except (AttributeError, NotImplementedError) as error:
                    self.skipTest(f"CUDA sleep unavailable: {error}")
                except RuntimeError as error:
                    if any(text in str(error).lower() for text in (
                            "not implemented", "not available", "not compiled", "not supported")):
                        self.skipTest(f"CUDA sleep unavailable: {error}")
                    raise
                junk.fill_(0xFF)
                del junk
                # Same-size Byte CUDA allocation encourages reuse, but allocator
                # address equality is deliberately not part of this contract.
                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(runtime.token, 0, [0])
                self.assertTrue(stream.query(), "resident bind must drain pending current-stream writes")
                out = runtime.forward(x, ids, scale)
            producer.wait_stream(stream)
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
            stats = runtime.stats()
            self.assertEqual(stats["resident_experts"], 1)
            self.assertEqual(stats["resident_cache_bytes"], expert_bytes)
            self.assertEqual(stats["resident_load_h2d_bytes"], expert_bytes)
            self.assertEqual(stats["cache_hits"], 1)
            self.assertEqual(stats["cache_misses"], 0)
            self.assertEqual(stats["weight_h2d_bytes"], 0)

    def test_fp32_epilogue_forced_tiles_signed_masked_routes(self):
        # H=128 exercises one N-pair; H=256 exercises multiple N-pairs and
        # forced single-N. H=64/192 must retain the single-N fallback. I=128 also
        # exercises GEMM2's K128 path where the forced M permits it.
        for dtype in (torch.float16, torch.bfloat16):
            for h, intermediate, tile_n in ((64, 64, 64), (128, 128, 128),
                                            (128, 128, 64), (192, 64, 128), (192, 64, 64),
                                            (256, 128, 128), (256, 128, 64)):
                for tile_m in (32, 64, 128):
                    with self.subTest(dtype=dtype, h=h, i=intermediate, m=tile_m, n=tile_n), mock.patch.dict(
                            os.environ, {"FUSE_MOE_TILE_M": str(tile_m), "FUSE_MOE_TILE_N": str(tile_n)}):
                        # Use cp.async in the legacy oracle too, including on SM120.
                        os.environ.pop("FUSE_MOE_TMA", None)
                        x, w1, w2, ids, scale = self.tensors(
                            s=tile_m + 3, h=h, i=intermediate, e=8, k=4, dtype=dtype)
                        ids.copy_(torch.tensor([0, 3, 7, 3], device=x.device,
                                               dtype=torch.int32).expand_as(ids))
                        ids[::5, 2] = -1
                        ids[0] = -1
                        scale.copy_(torch.linspace(-1.25, .75, scale.numel(),
                                                   device=x.device).reshape_as(scale))
                        scale[1::7, 1] = 0
                        scale[ids == -1] = float("nan")
                        legacy = ops.fuse_moe(x, w1.to(x.device), w2.to(x.device), ids, scale,
                                              backend="cuda")
                        self.assertTrue(torch.isfinite(legacy).all().item())
                        torch.testing.assert_close(legacy[0], torch.zeros_like(legacy[0]), atol=0, rtol=0)
                        # Unlike the broad FP32 reference tolerance, bound errors
                        # at the actual output magnitude and low-precision ULP scale.
                        eps = torch.finfo(dtype).eps
                        atol = max(1e-7, legacy.float().abs().max().item() * eps * 2)
                        for resident_ids in ((), (7, 0, 3), (0, 7)):
                            cache_bytes = len(resident_ids) * 6 * h * intermediate
                            with self.subTest(resident=resident_ids), self.runtime(
                                    w1, w2, banks=2, capacity=3, activation_mode="fp32_accumulate",
                                    max_route_tile_rows=tile_m - 1, tile_m=tile_m, tile_n=tile_n,
                                    resident_cache_bytes=cache_bytes,
                                    resident_experts={0: resident_ids}) as runtime:
                                out = runtime.forward(x, ids, scale)
                                self.assertEqual(out.dtype, dtype)
                                self.assertTrue(torch.isfinite(out).all().item())
                                torch.testing.assert_close(out.float(), legacy.float(), atol=atol, rtol=eps * 2)
                                torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=0, rtol=0)
                                stats = runtime.stats()
                                self.assertEqual(stats["activation_mode"], 1)
                                self.assertEqual(stats["cache_hits"], len(resident_ids))
                                self.assertEqual(stats["cache_misses"], 3 - len(resident_ids))
                                self.assertEqual(stats["weight_h2d_bytes"], (3 - len(resident_ids))
                                                 * 6 * h * intermediate)
                                self.assertEqual(stats["waves"], 3 if len(resident_ids) == 2 else 1)
                                self.assertGreater(stats["route_tiles"], stats["waves"])
                                self.assertEqual(stats["route_tile_rows"], tile_m - 1)
                                if len(resident_ids) == 3:
                                    self.assertEqual(stats["num_banks"], 0)
                                    self.assertEqual(stats["metadata_h2d_bytes"], 0)
                                del out

    def test_fp32_epilogue_preserves_down_rn_before_scale(self):
        # One nonzero GEMM2 product and one route per token eliminate both MMA
        # reduction-order noise and atomic-order noise. Binary-exact, non-power-
        # of-two scales expose moving the DownTout RN cast *after* the multiply.
        for dtype in (torch.float16, torch.bfloat16):
            for h in (64, 128, 192):
                for tile_m in (32, 64, 128):
                    with self.subTest(dtype=dtype, h=h, m=tile_m), mock.patch.dict(
                            os.environ, {"FUSE_MOE_TILE_M": str(tile_m), "FUSE_MOE_TILE_N": "128"}):
                        os.environ.pop("FUSE_MOE_TMA", None)
                        x = torch.zeros(7, h, device="cuda", dtype=dtype)
                        x[:, 0] = 1
                        w1 = torch.zeros(2, 128, h, dtype=dtype)
                        w1[1, 0, 0] = 1
                        w1[1, 64, 0] = 1
                        w2 = torch.zeros(2, h, 64, dtype=dtype)
                        w2[1, :, 0] = torch.linspace(.5, 1.5, h).to(dtype)
                        ids = torch.ones(7, 1, device=x.device, dtype=torch.int32)
                        scale = torch.tensor([1.5, -.75, 1.25, -1.5, .75, -1.25, 1.5],
                                             device=x.device, dtype=torch.float32)[:, None]
                        act = torch.nn.functional.silu(torch.tensor(1.0)).to(dtype).float()
                        down_fp32 = act * w2[1, :, 0].float()
                        expected = (down_fp32.to(dtype).float()[None, :] * scale.cpu()).to(dtype)
                        late_round = (down_fp32[None, :] * scale.cpu()).to(dtype)
                        self.assertGreater(torch.count_nonzero(expected != late_round).item(), 0,
                                           "fixture must distinguish Down RN from scale-before-RN")
                        legacy = ops.fuse_moe(x, w1.to(x.device), w2.to(x.device), ids, scale,
                                              backend="cuda")
                        torch.testing.assert_close(legacy.cpu(), expected, atol=0, rtol=0)
                        for resident, tile_rows in ((False, 1), (True, 1), (False, 7), (True, 7)):
                            with self.subTest(resident=resident, rows=tile_rows), self.runtime(
                                    w1, w2, activation_mode="fp32_accumulate", max_route_tile_rows=tile_rows,
                                    tile_m=tile_m, tile_n=128,
                                    resident_cache_bytes=6 * h * 64 if resident else 0,
                                    resident_experts={0: [1]} if resident else None) as runtime:
                                out = runtime.forward(x, ids, scale)
                                torch.testing.assert_close(out, legacy, atol=0, rtol=0)
                                self.assertEqual(runtime.stats()["route_tiles"], 7 if tile_rows == 1 else 1)
                                del out
                                # A reused accumulator must not leak the last full
                                # wave into a masked-only call, even with NaN scales.
                                out = runtime.forward(x, torch.full_like(ids, -1),
                                                      torch.full_like(scale, float("nan")))
                                torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
                                self.assertEqual(runtime.stats()["route_tiles"], 0)
                                del out

    def test_fp32_epilogue_clipped_expert_boundaries_and_full_wave_reuse(self):
        # Compact counts [1, 3, S, 2*S] put short experts, exact boundaries,
        # empty groups before/after a tile, and repeated routes in one wave.
        # Resident and miss waves must use the same clipped expert interval.
        for dtype, h, tile_n in ((dtype, h, tile_n) for dtype in (torch.float16, torch.bfloat16)
                                  for h, tile_n in ((64, 64), (128, 64), (128, 128))):
            x, w1, w2, ids, scale = self.tensors(s=35, h=h, i=64, e=8, k=5, dtype=dtype)
            ids.fill_(-1)
            ids[0, 0] = 0
            ids[:3, 1] = 1
            ids[:, 2] = 3
            ids[:, 3:] = 7
            scale.copy_(torch.linspace(-1.125, .875, scale.numel(), device=x.device).reshape_as(scale))
            scale[ids == -1] = float("nan")
            counts = torch.bincount(ids[ids >= 0].cpu().long(), minlength=8)
            self.assertEqual(counts.tolist(), [1, 3, 0, 35, 0, 0, 0, 70])
            total = int(counts.sum().item())
            with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "64", "FUSE_MOE_TILE_N": str(tile_n)}):
                os.environ.pop("FUSE_MOE_TMA", None)
                legacy = ops.fuse_moe(x, w1.to(x.device), w2.to(x.device), ids, scale, backend="cuda")
                eps = torch.finfo(dtype).eps
                atol = max(1e-7, legacy.float().abs().max().item() * eps * 2)
                for resident in (False, True):
                    for tile_rows in (1, 4, 5, 39, total - 1, total, total + 1):
                        with self.subTest(dtype=dtype, h=h, tile_n=tile_n, resident=resident,
                                          tile_rows=tile_rows), self.runtime(
                                w1, w2, capacity=4, activation_mode="fp32_accumulate",
                                max_route_tile_rows=tile_rows, tile_m=64, tile_n=tile_n,
                                resident_cache_bytes=4 * 6 * h * 64 if resident else 0,
                                resident_experts={0: [7, 1, 0, 3]} if resident else None) as runtime:
                            for call in range(2):
                                out = runtime.forward(x, ids, scale)
                                torch.testing.assert_close(out.float(), legacy.float(), atol=atol, rtol=eps * 2)
                                stats = runtime.stats()
                                self.assertEqual(stats["active_experts"], 4)
                                self.assertEqual(stats["waves"], 1)
                                self.assertEqual(stats["route_tile_rows"], min(tile_rows, total))
                                self.assertEqual(stats["route_tiles"], (total + tile_rows - 1) // tile_rows)
                                self.assertEqual(stats["cache_hits"], 4 if resident else 0)
                                self.assertEqual(stats["weight_h2d_bytes"], 0 if resident else 4 * 6 * h * 64)
                                self.assertEqual(stats["metadata_h2d_bytes"], 0 if resident else (4 * 4 + 2) * 4)
                                if call:
                                    self.assertEqual(stats["buffer_allocations"], 0)
                                    self.assertGreater(stats["buffer_reuses"], 0)
                                del out

    def test_auto_selects_lower_peak_in_both_directions(self):
        # All-resident fixtures remove staging/capacity choices from the peak
        # comparison. K=1 favours route_buffer; K=4 favours tiled FP32 output.
        for dtype in (torch.float16, torch.bfloat16):
            for k, chosen in ((1, 0), (4, 1)):
                x, w1, w2, ids, scale = self.tensors(s=128, h=128, i=64, e=4, k=k, dtype=dtype)
                ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
                          .remainder(4).reshape_as(ids))
                expected = reference(x, w1, w2, ids, scale)
                peaks = []
                for mode in ("route_buffer", "fp32_accumulate", "auto"):
                    with self.subTest(dtype=dtype, k=k, mode=mode), self.runtime(
                            w1, w2, activation_mode=mode, max_route_tile_rows=17,
                            resident_cache_bytes=4 * 6 * 128 * 64,
                            resident_experts={0: [3, 0, 2, 1]}) as runtime:
                        out = runtime.forward(x, ids, scale)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        stats = runtime.stats()
                        peaks.append(stats["gpu_charged_bytes"])
                        self.assertEqual(stats["weight_h2d_bytes"], 0)
                        self.assertEqual(stats["cache_hits"], 4)
                        if mode == "auto":
                            self.assertEqual(stats["activation_mode"], chosen)
                            self.assertEqual(peaks[-1], peaks[chosen])
                            self.assertLess(peaks[chosen], peaks[1 - chosen])
                        del out

    def test_fp32_epilogue_exact_resident_pool_without_down(self):
        # Native layout: Routes has two T-int arrays, four E-sized metadata
        # arrays, two prefix sentinels and one invalid-ID sentinel. Counts is
        # pinned-only. Fresh, all-resident contexts have no bank/high-water or
        # dynamic-cache allocations to hide the removed Down charge.
        for dtype in (torch.float16, torch.bfloat16):
            for groups in (1, 3):
                x, w1, w2, ids, scale = self.tensors(s=9, h=128, i=64, e=8, k=4, dtype=dtype)
                resident_ids = (0,) if groups == 1 else (0, 3, 7)
                ids.fill_(-1)
                ids[:, :groups] = torch.tensor(resident_ids, device=x.device, dtype=torch.int32)
                scale.copy_(torch.linspace(-1, .75, scale.numel(), device=x.device).reshape_as(scale))
                counts = torch.bincount(ids[ids >= 0].cpu().long(), minlength=w1.shape[0])
                self.assertEqual(counts.tolist(), [9 if expert in resident_ids else 0 for expert in range(8)])
                total = int(counts.sum().item())
                s, h, intermediate, e, t = x.shape[0], x.shape[1], w2.shape[2], w1.shape[0], ids.numel()
                counts_bytes = (e + 1) * 4
                cache_bytes = groups * self.charged_bytes(6 * h * intermediate)
                output_bytes = self.charged_bytes(s * h * x.element_size())
                expected = reference(x, w1, w2, ids, scale)
                for requested_rows in (1, 8):
                    sizes = {"Routes": (2 * t + 4 * e + 3) * 4, "Down": 0,
                             "Act": requested_rows * intermediate * 2, "Accumulator": s * h * 4,
                             "Scales": t * 4, "TileMeta": (4 * groups + 2) * 4,
                             "ResidentMeta": (4 * groups + 2) * 4 + (2 * groups * 8 if groups > 1 else 0)}
                    pool_bytes = sum(self.charged_bytes(size) for size in sizes.values())
                    removed_down = self.charged_bytes(requested_rows * h * 2)
                    old_pool_bytes = pool_bytes + removed_down
                    budget = cache_bytes + output_bytes + pool_bytes
                    with self.subTest(dtype=dtype, groups=groups, rows=requested_rows):
                        # First prove the independently computed pool, not merely
                        # a lower budget derived from an old native probe.
                        with self.runtime(w1, w2, activation_mode="fp32_accumulate",
                                          max_route_tile_rows=requested_rows, resident_cache_bytes=cache_bytes,
                                          resident_experts={0: resident_ids}, pinned_budget_bytes=counts_bytes) as probe:
                            out = probe.forward(x, ids, scale)
                            stats = probe.stats()
                            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                            self.assertEqual(stats["gpu_pool_bytes"], pool_bytes, sizes)
                            self.assertEqual(old_pool_bytes - stats["gpu_pool_bytes"], removed_down)
                            self.assertEqual(stats["gpu_charged_bytes"] - output_bytes - cache_bytes, pool_bytes)
                            self.assertEqual(stats["route_tile_rows"], requested_rows)
                            del out
                        for shortfall in (0, 1):
                            # One byte below the one-row floor cannot shrink any
                            # 512-byte charge (even grouping=1 has two meta quanta).
                            # At eight rows, however, shrinking Act to four rows is
                            # legal: do not mislabel an adaptable plan as OOM.
                            max_rows = 1 if requested_rows == 1 else 4096
                            with self.subTest(shortfall=shortfall), self.runtime(
                                    w1, w2, gpu_budget_bytes=budget - shortfall,
                                    activation_mode="fp32_accumulate", max_route_tile_rows=max_rows,
                                    resident_cache_bytes=cache_bytes, resident_experts={0: resident_ids},
                                    pinned_budget_bytes=counts_bytes) as runtime:
                                if shortfall and requested_rows == 1:
                                    with self.assertRaisesRegex(RuntimeError, "gpu_budget|budgets|history"):
                                        runtime.forward(x, ids, scale)
                                    continue
                                out = runtime.forward(x, ids, scale)
                                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                                stats = runtime.stats()
                                rows = 4 if shortfall else requested_rows
                                actual_pool = pool_bytes - (512 if shortfall else 0)
                                self.assertEqual(stats["route_tile_rows"], rows)
                                self.assertEqual(stats["route_tiles"], (total + rows - 1) // rows)
                                self.assertEqual(stats["waves"], 1)
                                self.assertEqual(stats["activation_mode"], 1)
                                self.assertEqual(stats["num_banks"], 0)
                                self.assertEqual(stats["wave_capacity"], 0)
                                self.assertEqual(stats["cache_hits"], groups)
                                self.assertEqual(stats["cache_misses"], 0)
                                self.assertEqual(stats["resident_cache_bytes"], cache_bytes)
                                self.assertEqual(stats["resident_experts"], groups)
                                self.assertEqual(stats["resident_load_h2d_bytes"], cache_bytes)
                                self.assertEqual(stats["gpu_pool_bytes"], actual_pool)
                                self.assertEqual(stats["gpu_charged_bytes"], actual_pool + output_bytes + cache_bytes)
                                self.assertLessEqual(stats["gpu_charged_bytes"], budget - shortfall)
                                self.assertEqual(stats["pinned_pool_bytes"], counts_bytes)
                                self.assertEqual(stats["pinned_bytes"], counts_bytes)
                                for key in ("weight_h2d_bytes", "metadata_h2d_bytes", "dynamic_cache_bytes",
                                            "dynamic_experts", "cache_metadata_bytes", "cache_evictions",
                                            "cache_budget_evictions"):
                                    self.assertEqual(stats[key], 0, key)
                                del out

    def test_fp32_epilogue_same_context_clipped_full_wave_metadata_reuse(self):
        # Keep S/K and max_tile fixed: one context must switch between clipped
        # TileMeta and full-wave wave_meta, then recover after an empty call.
        # Static resident weights and miss banks both exercise this transition.
        for dtype in (torch.float16, torch.bfloat16):
            with mock.patch.dict(os.environ, {"FUSE_MOE_TILE_M": "64", "FUSE_MOE_TILE_N": "128"}):
                os.environ.pop("FUSE_MOE_TMA", None)
                x, w1, w2, dense_ids, scale = self.tensors(s=35, h=256, i=128, e=8, k=4, dtype=dtype)
                dense_ids.fill_(-1)
                dense_ids[0, 0] = 0
                dense_ids[:3, 1] = 3
                dense_ids[:, 2:] = 7
                sparse_ids = torch.full_like(dense_ids, -1)
                # Exactly one route per expert: even a capacity-3 miss wave has
                # only three rows, and every miss bank takes the full-wave path.
                sparse_ids[0, :3] = torch.tensor([7, 0, 3], device=x.device, dtype=torch.int32)
                masked_ids = torch.full_like(dense_ids, -1)
                changed_ids = torch.tensor([7, 0, 3, 7], device=x.device,
                                           dtype=torch.int32).expand_as(dense_ids).contiguous()
                scale.copy_(torch.linspace(-1.125, .875, scale.numel(), device=x.device).reshape_as(scale))
                changed_scale = (-scale * .75 + .125).contiguous()
                phases = (("clipped", dense_ids, scale, [1, 3, 70]),
                          ("full_wave", sparse_ids, scale, [1, 1, 1]),
                          ("clipped_again", dense_ids, scale, [1, 3, 70]),
                          ("masked", masked_ids, scale, [0, 0, 0]),
                          ("rescaled", dense_ids, changed_scale, [1, 3, 70]),
                          ("changed_routes", changed_ids, changed_scale, [35, 35, 70]))
                expert_bytes = 6 * x.shape[1] * w2.shape[2]
                device_w1, device_w2 = w1.to(x.device), w2.to(x.device)
                for resident in (True, False):
                    for banks in (1, 2):
                        with self.subTest(dtype=dtype, resident=resident, banks=banks), self.runtime(
                                w1, w2, banks=banks, capacity=3, activation_mode="fp32_accumulate",
                                max_route_tile_rows=7, tile_m=64, tile_n=128,
                                resident_cache_bytes=3 * expert_bytes if resident else 0,
                                resident_experts={0: [7, 0, 3]} if resident else None) as runtime:
                            highwater = None
                            for phase, ids, phase_scale, expert_counts in phases:
                                with self.subTest(phase=phase):
                                    counts = torch.bincount(ids[ids >= 0].cpu().long(), minlength=8)
                                    self.assertEqual(counts.tolist(), [expert_counts[0], 0, 0, expert_counts[1],
                                                                     0, 0, 0, expert_counts[2]])
                                    total = int(counts.sum().item())
                                    active = int(torch.count_nonzero(counts).item())
                                    routed_scale = phase_scale.clone()
                                    routed_scale[ids == -1] = float("nan")
                                    legacy = ops.fuse_moe(x, device_w1, device_w2, ids, routed_scale, backend="cuda")
                                    out = runtime.forward(x, ids, routed_scale)
                                    self.assertEqual(out.dtype, dtype)
                                    self.assertTrue(torch.isfinite(out).all().item())
                                    eps = torch.finfo(dtype).eps
                                    atol = max(1e-7, legacy.float().abs().max().item() * eps * 2)
                                    torch.testing.assert_close(out.float(), legacy.float(), atol=atol, rtol=eps * 2)
                                    empty_rows = (ids == -1).all(dim=1)
                                    torch.testing.assert_close(out[empty_rows], torch.zeros_like(out[empty_rows]),
                                                               atol=0, rtol=0)
                                    stats = runtime.stats()
                                    self.assertEqual(stats["activation_mode"], 1)
                                    self.assertEqual(stats["active_experts"], active)
                                    self.assertEqual(stats["waves"], 1 if active else 0)
                                    self.assertEqual(stats["route_tile_rows"], min(total, 7))
                                    self.assertEqual(stats["route_tiles"], (total + 6) // 7)
                                    self.assertEqual(stats["cache_hits"], active if resident else 0)
                                    self.assertEqual(stats["cache_misses"], 0 if resident else active)
                                    self.assertEqual(stats["num_banks"], banks if active and not resident else 0)
                                    self.assertEqual(stats["wave_capacity"], 3 if active and not resident else 0)
                                    self.assertEqual(stats["weight_h2d_bytes"], 0 if resident else active * expert_bytes)
                                    self.assertEqual(stats["metadata_h2d_bytes"],
                                                     (4 * active + 2) * 4 if active and not resident else 0)
                                    current = {key: stats[key] for key in (
                                        "gpu_pool_bytes", "pinned_pool_bytes", "gpu_charged_bytes",
                                        "resident_cache_bytes", "resident_experts")}
                                    if highwater is None:
                                        highwater = current
                                        self.assertGreater(stats["buffer_allocations"], 0)
                                    else:
                                        # All-masked configure retains high-water
                                        # Act/metadata/banks, despite zero execution
                                        # rows. Fresh-only exact Down accounting is
                                        # deliberately not applied to a reused pool.
                                        self.assertEqual(current, highwater)
                                        self.assertEqual(stats["buffer_allocations"], 0)
                                        # Active phases reuse TileMeta plus either
                                        # ResidentMeta or every bank's host/device
                                        # weights+metadata, not just route buffers.
                                        minimum_reuses = (7 if resident else 6 + 4 * banks) if active else 4
                                        self.assertGreaterEqual(stats["buffer_reuses"], minimum_reuses)
                                    for key in ("dynamic_cache_bytes", "dynamic_experts", "cache_admissions",
                                                "cache_evictions", "cache_budget_evictions"):
                                        self.assertEqual(stats[key], 0, key)
                                    # Release every phase's runtime output before
                                    # the next call so history cannot alter planning.
                                    del out, legacy


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GEGLUTests(unittest.TestCase):
    """GPU GEGLU/SwiGLU × fused/staged coverage (single class CLI selectable).

    Run one class::

        python tests/test_fuse_moe_offload.py GEGLUTests

    The staged reference is a per-expert CPU FP32 accumulation with the exact
    erf GELU and explicit dtype rounding at every materialization point. It is
    intentionally not bitwise: fp32_accumulate reduces in a nondeterministic
    order and router scales are caller float32.
    """

    default_async_staging = False

    @classmethod
    def setUpClass(cls):
        ops.ensure_loaded("fuse_moe")
        if not ops.is_available("fuse_moe"):
            raise unittest.SkipTest(ops.load_error("fuse_moe"))
        torch.manual_seed(19)

    def tensors(self, s=129, h=64, i=64, e=8, k=4, dtype=torch.float16, masked=False):
        x = (torch.randn(s, h, device="cuda") * .3).to(dtype)
        w1 = (torch.randn(e, 2 * i, h) * .05).to(dtype)
        w2 = (torch.randn(e, h, i) * .05).to(dtype)
        ids = torch.randint(e, (s, k), device="cuda", dtype=torch.int32)
        scale = torch.rand(s, k, device="cuda", dtype=torch.float32)
        if masked and s:
            ids[::3] = -1
            ids[0] = -1
            scale = scale.masked_fill(ids < 0, float("nan"))
        return x, w1, w2, ids, scale

    def runtime(self, w1, w2, **kwargs):
        async_staging = kwargs.setdefault("async_staging", self.default_async_staging)
        runtime = PrefillMoERuntime(
            ExpertStore({0: (w1, w2)}),
            gpu_budget_bytes=kwargs.pop("gpu_budget_bytes", 16 << 20),
            ram_budget_bytes=kwargs.pop("ram_budget_bytes", 64 << 20),
            pinned_budget_bytes=kwargs.pop("pinned_budget_bytes", 16 << 20),
            **kwargs)
        self.assertEqual(runtime.async_staging, async_staging)
        return runtime

    @staticmethod
    def modes():
        return (("silu", "fused"), ("gelu", "fused"),
                ("silu", "staged"), ("gelu", "staged"))

    def assert_modes_close(self, out, ref, label):
        torch.testing.assert_close(out.float().cpu(), ref, atol=.004, rtol=.06,
                                   msg=label)

    def test_full_gpu_four_mode_dtype_matrix(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(
                s=129, h=128, i=128, e=8, k=4, dtype=dtype)
            for expert_activation, numerical_mode in self.modes():
                with self.subTest(dtype=dtype, expert_activation=expert_activation,
                                  numerical_mode=numerical_mode):
                    out = ops.fuse_moe(
                        x, w1.cuda(), w2.cuda(), ids, scale, backend="cuda",
                        expert_activation=expert_activation,
                        numerical_mode=numerical_mode)
                    ref = reference(x, w1, w2, ids, scale,
                                    expert_activation, numerical_mode)
                    self.assertEqual(out.dtype, dtype)
                    self.assertEqual(tuple(out.shape), (129, 128))
                    self.assertTrue(torch.isfinite(out).all().item())
                    self.assert_modes_close(
                        out, ref,
                        f"{dtype} {expert_activation}/{numerical_mode}")

    def test_isolated_scalar_semantics_wide_n_atomic(self):
        # Share the CPU witness builder with the all-GPU test without duplicating
        # its cast oracle. Native GEMMs each have only one nonzero product.
        from test_fuse_moe import numerical_semantics_fixture

        for dtype in (torch.float16, torch.bfloat16):
            for numerical_mode in ("fused", "staged"):
                x, w1, w2, ids, scale, expected, alternatives = numerical_semantics_fixture(
                    dtype, numerical_mode, h=128, i=256)
                torch.testing.assert_close(
                    reference(x, w1, w2, ids, scale, "gelu", numerical_mode),
                    expected, atol=0, rtol=0)
                for activation_mode in ("route_buffer", "fp32_accumulate"):
                    with self.subTest(dtype=dtype, numerical_mode=numerical_mode,
                                      activation_mode=activation_mode), self.runtime(
                            w1, w2, activation_mode=activation_mode, tile_m=32,
                            tile_n=128, max_route_tile_rows=7,
                            expert_activation="gelu", numerical_mode=numerical_mode) as runtime:
                        self.assertEqual(runtime.tile_n, 128)
                        actual = runtime.forward(x.cuda(), ids.cuda(), scale.cuda()).float().cpu()
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        for name, wrong in alternatives.items():
                            self.assertFalse(torch.equal(actual, wrong),
                                             f"failed to discriminate {name}")
                        if activation_mode == "fp32_accumulate":
                            self.assertGreater(runtime.stats()["route_tiles"], 1)
                            self.assertEqual(runtime.stats()["activation_mode"], 1)

    def test_offload_route_fp32_cache_and_partial_tiles(self):
        for dtype in (torch.float16, torch.bfloat16):
            x, w1, w2, ids, scale = self.tensors(
                s=129, h=64, i=64, e=8, k=4, dtype=dtype)
            expert_bytes = 6 * 64 * 64
            for expert_activation, numerical_mode in self.modes():
                ref = reference(x, w1, w2, ids, scale,
                                expert_activation, numerical_mode)
                tag = f"{dtype} {expert_activation}/{numerical_mode}"
                # route_buffer never tiles.
                with self.subTest(tag=tag, mode="route_buffer"), self.runtime(
                        w1, w2, activation_mode="route_buffer", tile_m=32,
                        expert_activation=expert_activation,
                        numerical_mode=numerical_mode) as runtime:
                    out = runtime.forward(x, ids, scale)
                    self.assert_modes_close(out, ref, tag + " route_buffer")
                    self.assertEqual(runtime.stats()["activation_mode"], 0)
                    self.assertEqual(runtime.expert_activation, expert_activation)
                    self.assertEqual(runtime.numerical_mode, numerical_mode)
                # FP32 route accumulation with a tiny row cap forces many
                # partial tiles; repeat to cover pool reuse and warm metadata.
                with self.subTest(tag=tag, mode="fp32_partial"), self.runtime(
                        w1, w2, num_banks=2, activation_mode="fp32_accumulate",
                        max_route_tile_rows=7, tile_m=32,
                        expert_activation=expert_activation,
                        numerical_mode=numerical_mode) as runtime:
                    for call in range(2):
                        out = runtime.forward(x, ids, scale)
                        self.assert_modes_close(out, ref, tag + " fp32 partial")
                        stats = runtime.stats()
                        self.assertEqual(stats["activation_mode"], 1)
                        self.assertLessEqual(stats["route_tile_rows"], 7)
                        if call == 0:
                            self.assertGreater(stats["route_tiles"],
                                               (ids.numel() + 6) // 7 - 1)
                            self.assertGreater(stats["route_tiles"], 1)
                # Cache-warm invocation: the second call must hit promoted
                # experts without changing the numerical contract.
                with self.subTest(tag=tag, mode="cache_warm"), self.runtime(
                        w1, w2, num_banks=2, activation_mode="fp32_accumulate",
                        tile_m=32, dynamic_cache_bytes=4 * expert_bytes,
                        cache_policy="lru",
                        expert_activation=expert_activation,
                        numerical_mode=numerical_mode) as runtime:
                    first = runtime.forward(x, ids, scale)
                    self.assert_modes_close(first, ref, tag + " cache first")
                    second = runtime.forward(x, ids, scale)
                    self.assert_modes_close(second, ref, tag + " cache warm")
                    stats = runtime.stats()
                    self.assertGreater(stats["dynamic_cache_hits"], 0)
                    self.assertEqual(stats["cache_budget_evictions"], 0)
                    del first, second

    def test_multi_context_modes_do_not_leak(self):
        x, w1, w2, ids, scale = self.tensors(
            s=65, h=64, i=64, e=8, k=3, dtype=torch.bfloat16)
        refs = {(a, n): reference(x, w1, w2, ids, scale, a, n)
                for a, n in self.modes()}
        with self.runtime(w1, w2, activation_mode="route_buffer",
                          expert_activation="silu", numerical_mode="fused") as plain, \
                self.runtime(w1, w2, activation_mode="fp32_accumulate",
                             max_route_tile_rows=7, expert_activation="gelu",
                             numerical_mode="staged") as staged:
            for call in range(2):
                out_plain = plain.forward(x, ids, scale)
                out_staged = staged.forward(x, ids, scale)
                self.assert_modes_close(out_plain, refs[("silu", "fused")],
                                        "plain context")
                self.assert_modes_close(out_staged, refs[("gelu", "staged")],
                                        "staged context")
                self.assertEqual(plain.stats()["activation_mode"], 0)
                self.assertEqual(staged.stats()["activation_mode"], 1)
                self.assertEqual(plain.expert_activation, "silu")
                self.assertEqual(plain.numerical_mode, "fused")
                self.assertEqual(staged.expert_activation, "gelu")
                self.assertEqual(staged.numerical_mode, "staged")
                del out_plain, out_staged

    def test_native_v1_v2_contract_and_failure_before_token_mutation(self):
        x, w1, w2, ids, scale = self.tensors(
            s=65, h=64, i=64, e=8, k=2, dtype=torch.float16)
        gpu_weights = (w1.cuda(), w2.cuda())
        # Native calls bypass strict Python string validation; invalid integer
        # codes must still be rejected at op/create entry, before side effects.
        for field in ("expert_activation", "numerical_mode"):
            for code in (-1, 2):
                with self.subTest(field=field, code=code):
                    with self.assertRaisesRegex(RuntimeError, field):
                        torch.ops.recsys_ops.fuse_moe(
                            x, *gpu_weights, ids, scale, **{field: code})
                    with self.assertRaisesRegex(RuntimeError, field):
                        torch.ops.recsys_ops.fuse_moe_offload_create(
                            torch.cuda.current_device(), **{field: code})
        legacy = torch.ops.recsys_ops.fuse_moe(x, *gpu_weights, ids, scale)
        explicit = ops.fuse_moe(x, *gpu_weights, ids, scale,
                                expert_activation="silu", numerical_mode="fused")
        torch.testing.assert_close(legacy, explicit, atol=0, rtol=0)
        del gpu_weights, legacy, explicit
        store = ExpertStore({0: (w1, w2)})
        major, minor = torch.cuda.get_device_capability(x.device)
        defaults = OffloadConfig.from_store(
            store, gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20,
            pinned_budget_bytes=100000, target_sm=major * 10 + minor,
            async_staging=self.default_async_staging)
        staged_config = OffloadConfig.from_store(
            store, gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20,
            pinned_budget_bytes=100000, target_sm=major * 10 + minor,
            activation_mode="fp32_accumulate", async_staging=self.default_async_staging,
            expert_activation="gelu", numerical_mode="staged")
        with self.runtime(w1, w2, activation_mode="fp32_accumulate",
                          pinned_budget_bytes=100000,
                          expert_activation="gelu", numerical_mode="staged") as runtime:
            out = runtime.forward(x, ids, scale)
            self.assert_modes_close(out, reference(x, w1, w2, ids, scale,
                                                   "gelu", "staged"),
                                    "v2 valid forward")
            v1_vector = defaults.layer_config(0)
            v1_manifest = defaults.manifest()
            valid_vector = staged_config.layer_config(0)
            valid_manifest = staged_config.manifest()
            self.assertEqual((v1_vector[0], len(v1_vector)), (1, 23))
            self.assertEqual((valid_vector[0], len(valid_vector)), (2, 25))
            self.assertEqual(valid_manifest[0], 1)

            def assert_rejected(vector, manifest, message):
                before, version = runtime.token.clone(), runtime.token._version
                with self.assertRaisesRegex(RuntimeError, message):
                    torch.ops.recsys_ops.fuse_moe_offload(
                        x, w1, w2, ids, scale, runtime.token, 0, vector, manifest)
                self.assertEqual(runtime.token._version, version)
                torch.testing.assert_close(runtime.token, before, atol=0, rtol=0)

            # v1 records never imply GEGLU/staged.
            assert_rejected(v1_vector, v1_manifest, "version 1 configuration")
            # Config-only malformed v2: width and out-of-range mode codes.
            assert_rejected(valid_vector[:-1], (), "config length")
            assert_rejected((*valid_vector, 0), (), "config length")
            assert_rejected((2, *valid_vector[1:23], 2, 0, *valid_vector[25:]),
                            (), "expert_activation")
            assert_rejected((2, *valid_vector[1:23], 0, 2, *valid_vector[25:]),
                            (), "numerical_mode")
            # Config that disagrees with an otherwise valid manifest.
            assert_rejected(valid_vector[:-1], valid_manifest,
                            "differs from manifest|config length")
            assert_rejected((*valid_vector, 0), valid_manifest,
                            "differs from manifest|config length")
            # Manifest framing, inventory and truncation.
            assert_rejected(valid_vector, (2, *valid_manifest[1:]),
                            "manifest")
            assert_rejected(valid_vector, (1, 1, 1,
                                           len(valid_vector), *valid_vector),
                            "not bound|missing")
            assert_rejected(valid_vector, valid_manifest[:-1], "manifest")
            # A failed preflight must not consume the healthy context.
            recovered = runtime.forward(x, ids, scale)
            self.assert_modes_close(recovered, reference(x, w1, w2, ids, scale,
                                                         "gelu", "staged"),
                                    "recovery after rejected config")

        # Native accepts a v2 record whose codes are the defaults, even though
        # Python's canonical form for defaults is v1.
        with self.runtime(w1, w2, activation_mode="route_buffer",
                          pinned_budget_bytes=100000,
                          expert_activation="silu", numerical_mode="fused") as base:
            v1_vector = defaults.layer_config(0)
            v2_defaults = (2, *v1_vector[1:23], 0, 0, *v1_vector[23:])
            v2_manifest = (1, 1, 0, len(v2_defaults), *v2_defaults)
            with self.assertRaisesRegex(ValueError, "noncanonical"):
                OffloadConfig.from_layer_configs({0: v2_defaults})
            out = torch.ops.recsys_ops.fuse_moe_offload(
                x, w1, w2, ids, scale, base.token, 0, v2_defaults, v2_manifest)
            self.assert_modes_close(out, reference(x, w1, w2, ids, scale),
                                    "native v2 defaults")
            self.assertEqual(tuple(out.shape), (65, 64))

    def test_masked_topk_and_zero_sequence(self):
        x, w1, w2, ids, scale = self.tensors(
            s=129, h=64, i=64, e=8, k=4, dtype=torch.float16, masked=True)
        self.assertTrue((ids < 0).any().item())
        self.assertTrue(torch.isnan(scale).any().item())
        ref = reference(x, w1, w2, ids, scale, "gelu", "staged")
        with self.runtime(w1, w2, activation_mode="fp32_accumulate",
                          max_route_tile_rows=7, tile_m=32,
                          expert_activation="gelu",
                          numerical_mode="staged") as runtime:
            out = runtime.forward(x, ids, scale)
            self.assertTrue(torch.isfinite(out).all().item())
            self.assert_modes_close(out, ref, "masked -1")
            all_masked = (ids < 0).all(dim=1)
            if all_masked.any():
                torch.testing.assert_close(out[all_masked],
                                           torch.zeros_like(out[all_masked]),
                                           atol=0, rtol=0)
            stats = runtime.stats()
            self.assertEqual(stats["active_experts"],
                             len(set(ids.cpu().flatten().tolist()) - {-1}))
            del out

            # S=0 is a supported offload shape: no routes, no output, no error.
            empty_x = torch.empty(0, 64, device="cuda", dtype=torch.float16)
            empty_ids = torch.empty(0, 4, device="cuda", dtype=torch.int32)
            empty_scale = torch.empty(0, 4, device="cuda", dtype=torch.float32)
            empty = runtime.forward(empty_x, empty_ids, empty_scale)
            self.assertEqual(tuple(empty.shape), (0, 64))
            self.assertEqual(empty.dtype, torch.float16)
            self.assertEqual(runtime.stats()["active_experts"], 0)
            del empty
            # The same context still serves a nonempty invocation after S=0.
            again = runtime.forward(x, ids, scale)
            self.assert_modes_close(again, ref, "recovery after S=0")


class MLPStoreTests(unittest.TestCase):
    def store(self, expert_type="mlp"):
        return ExpertStore({layer: (torch.zeros(3, (1 if expert_type == "mlp" else 2) * i, 64, dtype=torch.float16),
                                    torch.zeros(3, 64, i, dtype=torch.float16))
                            for layer, i in ((0, 64), (5, 128))})

    def config(self, store, **kwargs):
        return OffloadConfig.from_store(store, target_sm=89, gpu_budget_bytes=2 << 20,
            ram_budget_bytes=4 << 20, pinned_budget_bytes=200000,
            resident_cache_bytes=200000, resident_experts={0: (0, 2), 5: (1,)}, **kwargs)

    def test_v3_all_modes_roundtrip_archive_and_runtime_options(self):
        store = self.store()
        with mock.patch.object(ops, "ensure_loaded") as load:
            for activation in ("silu", "gelu"):
                for numerical in ("fused", "staged"):
                    config = self.config(store, expert_type="mlp", expert_activation=activation, numerical_mode=numerical)
                    vectors = {layer: config.layer_config(layer) for layer in store.layers}
                    for layer, vector in vectors.items():
                        self.assertEqual((vector[0], len(vector)), (3, 26 + vector[21]))
                        self.assertEqual(vector[23:26], (int(activation == "gelu"), int(numerical == "staged"), 1))
                        self.assertEqual(vector[26:], (0, 2) if layer == 0 else (1,))
                    self.assertEqual(config.manifest()[0], 1)
                    self.assertEqual(config.runtime_options()["expert_type"], "mlp")
                    self.assertEqual(OffloadConfig.from_layer_configs(vectors), config)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "weights.pt"
                        store.save_cpp(path, weight_version=9, config=config)
                        module = torch.jit.load(str(path), map_location="cpu")
                        self.assertEqual(module.format_version.tolist(), [2])
                        restored = ExpertStore.load_cpp(path)
                        self.assertEqual(restored.deployment_config, config)
                        self.assertEqual(restored.weight_version, 9)
                        for layer in store.layers:
                            for actual, expected in zip(restored.layers[layer], store.layers[layer]):
                                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            load.assert_not_called()

    def test_type_and_mixed_shapes_reject_before_native_create(self):
        native = mock.Mock()
        with mock.patch.object(ops, "ensure_loaded") as load, mock.patch.object(torch.ops, "recsys_ops", native):
            for value in ("MLP", "mlp ", "", -1, 0, 1, 2, True, None, [], {}):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "expert_type"):
                    self.config(self.store(), expert_type=value)
            for architecture in ("glu", "mlp"):
                wrong = self.store("glu" if architecture == "mlp" else "mlp")
                with self.assertRaisesRegex(ValueError, "expert_type"):
                    self.config(wrong, expert_type=architecture)
                with self.assertRaisesRegex(ValueError, "expert_type"):
                    PrefillMoERuntime(wrong, device="cuda:0", expert_type=architecture)
            mixed = ExpertStore({0: self.store().layers[0], 5: self.store("glu").layers[5]})
            with self.assertRaisesRegex(ValueError, "expert_type"):
                self.config(mixed, expert_type="mlp")
            load.assert_not_called()
            native.fuse_moe_offload_create.assert_not_called()

    def test_v3_invalid_codes_crosslayer_and_manifest_rejection(self):
        config = self.config(self.store(), expert_type="mlp")
        vectors = {layer: config.layer_config(layer) for layer in (0, 5)}
        for position, value in ((0, 4), (23, -1), (23, 2), (24, 2), (25, -1), (25, 0), (25, 2), (25, True)):
            wrong = list(vectors[0])
            wrong[position] = value
            with self.subTest(position=position, value=value), self.assertRaises((ValueError, TypeError)):
                OffloadConfig.from_layer_configs({**vectors, 0: tuple(wrong)})
        other = self.config(self.store(), expert_type="mlp", expert_activation="gelu")
        with self.assertRaises(ValueError):
            OffloadConfig.from_layer_configs({0: vectors[0], 5: other.layer_config(5)})
        glu = self.config(self.store("glu"), expert_type="glu")
        with self.assertRaises(ValueError):
            OffloadConfig.from_layer_configs({0: vectors[0], 5: glu.layer_config(5)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            self.store().save_cpp(path, weight_version=1, config=config)
            module = torch.jit.load(str(path))
            module.offload_config_5[25] = 0
            torch.jit.save(module, str(path))
            with self.assertRaises(ValueError):
                ExpertStore.load_cpp(path)

    def test_create_twenty_args_explicit_codes(self):
        native = mock.Mock()
        native.fuse_moe_offload_create.return_value = torch.zeros(2, dtype=torch.int64)
        with (mock.patch.object(ops, "ensure_loaded"), mock.patch.object(ops, "is_available", return_value=True),
              mock.patch.object(torch.ops, "recsys_ops", native)):
            for architecture in ("glu", "mlp"):
                for activation in ("silu", "gelu"):
                    for numerical in ("fused", "staged"):
                        with PrefillMoERuntime(self.store(architecture), device="cuda:0", expert_type=architecture,
                                              expert_activation=activation, numerical_mode=numerical):
                            args = native.fuse_moe_offload_create.call_args.args
                            self.assertEqual(len(args), 20)
                            self.assertEqual(args[17:20], (int(activation == "gelu"), int(numerical == "staged"), int(architecture == "mlp")))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MLPTests(unittest.TestCase):
    """Independent MLP suite; never inherits/relabels the six GEGLU methods."""
    default_async_staging = False
    setUpClass = classmethod(GEGLUTests.setUpClass.__func__)
    assert_modes_close = GEGLUTests.assert_modes_close

    def test_dual_dtype_four_modes_sync_async_cache_mask_signed_duplicate_edges(self):
        cases = ((7, 64, 64, 8, 4), (129, 128, 192, 8, 4), (65, 192, 128, 8, 4),
                 (3, 64, 64, 8, 128), (3, 64, 64, 512, 2), (1, 64, 64, 1, 1))
        for dtype in (torch.float16, torch.bfloat16):
            for activation, numerical in GEGLUTests.modes():
                for s, h, i, e, k in cases:
                    x = (torch.randn(s, h, device="cuda") * .3).to(dtype)
                    w1 = (torch.randn(e, i, h) * .05).to(dtype)
                    w2 = (torch.randn(e, h, i) * .05).to(dtype)
                    ids = torch.arange(s, device="cuda", dtype=torch.int32).remainder(e)[:, None].expand(s, k).contiguous()
                    if s > 1:
                        ids[0] = -1
                    scale = torch.linspace(-.75, .9, s * k, device="cuda").reshape(s, k) / k
                    scale[ids < 0] = float("nan")
                    expected = reference(x, w1, w2, ids, scale, activation, numerical, "mlp")
                    for asynchronous in (False, True):
                        for cache in ("none", "static", "lru", "reuse"):
                            expert_bytes = 4 * h * i
                            with self.subTest(dtype=dtype, activation=activation, numerical=numerical,
                                              shape=(s,h,i,e,k), asynchronous=asynchronous, cache=cache):
                                config = OffloadConfig.from_store(ExpertStore({0: (w1, w2)}),
                                    target_sm=sum(v * f for v, f in zip(torch.cuda.get_device_capability(), (10, 1))),
                                    expert_type="mlp", expert_activation=activation, numerical_mode=numerical,
                                    gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20, pinned_budget_bytes=2 << 20,
                                    async_staging=asynchronous, activation_mode="fp32_accumulate", max_route_tile_rows=7,
                                    resident_cache_bytes=expert_bytes if cache == "static" else 0,
                                    resident_experts={0: (0,)} if cache == "static" else {},
                                    dynamic_cache_bytes=2 * expert_bytes if cache in ("lru", "reuse") else 0,
                                    cache_policy="lru" if cache == "lru" else "reuse")
                                with PrefillMoERuntime.from_config(ExpertStore({0: (w1,w2)}), config) as runtime:
                                    for _ in range(2):
                                        out = runtime.forward(x, ids, scale)
                                        self.assert_modes_close(out, expected, "MLP")
                                        self.assertTrue(torch.isfinite(out).all())
                                        del out
                                    self.assertLessEqual(runtime.stats()["gpu_charged_bytes"], config.gpu_budget_bytes)
                                    self.assertLessEqual(runtime.stats()["pinned_bytes"], config.pinned_budget_bytes)

    def test_scalar_activation_and_stage_witnesses(self):
        from test_fuse_moe import mlp_numerical_semantics_fixture
        for dtype in (torch.float16, torch.bfloat16):
            for activation, numerical in GEGLUTests.modes():
                x, w1, w2, ids, scale, expected, alternatives = mlp_numerical_semantics_fixture(dtype, numerical, activation)
                with PrefillMoERuntime(ExpertStore({0: (w1,w2)}), expert_type="mlp",
                        expert_activation=activation, numerical_mode=numerical, async_staging=self.default_async_staging,
                        gpu_budget_bytes=16 << 20, pinned_budget_bytes=2 << 20) as runtime:
                    actual = runtime.forward(x.cuda(), ids.cuda(), scale.cuda()).float().cpu()
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for wrong in alternatives.values():
                        self.assertFalse(torch.equal(actual, wrong))

    def test_v3_context_mismatch_before_token_mutation(self):
        store = MLPStoreTests().store()
        major, minor = torch.cuda.get_device_capability()
        config = OffloadConfig.from_store(store, expert_type="mlp", target_sm=major * 10 + minor,
            gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20, pinned_budget_bytes=2 << 20)
        with PrefillMoERuntime.from_config(store, config) as runtime:
            x = torch.zeros(7,64,device="cuda",dtype=torch.float16)
            ids = torch.zeros(7,2,device="cuda",dtype=torch.int32)
            scale = torch.ones(7,2,device="cuda")
            before, version, stats = runtime.token.clone(), runtime.token._version, runtime.stats()
            for position, code in ((23, 1), (24, 1), (25, 0), (25, -1), (25, 2)):
                vector = list(config.layer_config(0)); vector[position] = code
                manifest = (1, 2, 0, len(vector), *vector, 5, len(config.layer_config(5)), *config.layer_config(5))
                with self.assertRaises(RuntimeError):
                    torch.ops.recsys_ops.fuse_moe_offload(x, *store.layers[0], ids, scale, runtime.token, 0, vector, manifest)
                torch.testing.assert_close(runtime.token, before, atol=0, rtol=0)
                self.assertEqual(runtime.token._version, version)
                self.assertEqual(runtime.stats(), stats)


class InputBindingTests(unittest.TestCase):
    """CPU-only adapter contracts; never load native code or discover CUDA."""

    def setUp(self):
        self.binding = moe_offload.InputBinding
        pair = (torch.zeros(3, 128, 64, dtype=torch.float16),
                torch.zeros(3, 64, 64, dtype=torch.float16))
        self.store = ExpertStore({5: tuple(w.clone() for w in pair), 0: pair})
        self.config = OffloadConfig.from_store(self.store, gpu_budget_bytes=200000,
            ram_budget_bytes=300000, pinned_budget_bytes=100000, target_sm=89)
        native = mock.Mock()
        native.fuse_moe_offload_create.return_value = torch.zeros(2, dtype=torch.int64)
        native.fuse_moe_offload_status.return_value = torch.tensor(
            [1, 0, 0, 0, 0, 0, -1, 0, 0, -1], dtype=torch.int64)
        patches = (mock.patch.object(ops, "ensure_loaded"),
                   mock.patch.object(ops, "is_available", return_value=True),
                   mock.patch.object(torch.ops, "recsys_ops", native),
                   mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
                   mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError(
                       "CPU adapter tests must use an explicit device")))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.native = native
        self.runtime = moe_offload.PrefillMoERuntime.from_config(
            self.store, self.config, device="cuda:0")
        self.addCleanup(self.runtime.close)
        self.bindings = (self.binding(1, "w2", 5), self.binding(2, "token"),
                         self.binding(3, "w1", 0), self.binding(4, "w2", 0),
                         self.binding(5, "w1", 5))

    def test_t01_input_binding_frozen_placeholder_order_and_explicit_identity(self):
        b = self.binding(2, "token")
        with self.assertRaises(FrozenInstanceError):
            b.index = 3
        x = torch.zeros(2, 64, dtype=torch.float16)
        inputs = [x, None, None, None, None, None]
        prepared = self.runtime.prepare_inputs(inputs, self.bindings)
        self.assertIs(type(prepared), tuple)
        self.assertIs(prepared[0], x)
        self.assertEqual(inputs, [x, None, None, None, None, None])
        for index, expected in ((1, self.store.layers[5][1]), (2, self.runtime.token),
                                (3, self.store.layers[0][0]), (4, self.store.layers[0][1]),
                                (5, self.store.layers[5][0])):
            self.assertIs(prepared[index], expected)
        explicit = self.runtime.prepare_inputs(prepared, tuple(reversed(self.bindings)))
        self.assertTrue(all(a is b for a, b in zip(prepared, explicit)))

    def test_t01_input_binding_invalid_fields_before_native_load(self):
        for index, kind, layer in ((-1, "token", None), (True, "token", None),
                                   (1.0, "token", None), ("1", "token", None),
                                   (1, "runtime_token", None), (1, "gate_up", 0),
                                   (1, "down", 0), (1, "token", 0),
                                   (1, "w1", None), (1, "w2", True),
                                   (1, "w1", -1)):
            with self.subTest(index=index, kind=kind, layer=layer), self.assertRaises(
                    (TypeError, ValueError)):
                self.binding(index, kind, layer)

    def test_t01_binding_coverage_and_wrong_explicit_inputs_before_callable(self):
        inputs = [object(), None, None, None, None, None]
        prepared = self.runtime.prepare_inputs(inputs, self.bindings)
        callback = mock.Mock()
        invalid = (self.bindings[:-1], self.bindings + (self.bindings[0],),
                   self.bindings + (self.binding(0, "token"),),
                   (self.binding(6, "token"), *self.bindings[:1], *self.bindings[2:]),
                   (self.binding(1, "w2", 99), *self.bindings[1:]))
        before = self.runtime.status()
        for bindings in invalid:
            with self.subTest(bindings=bindings), self.assertRaises((ValueError, TypeError)):
                self.runtime.invoke(callback, inputs, artifact_config=self.config, bindings=bindings)
        for index in (1, 2, 3, 4, 5):
            wrong = list(prepared)
            wrong[index] = wrong[index].clone()
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                self.runtime.invoke(callback, wrong, artifact_config=self.config,
                                    bindings=self.bindings)
        callback.assert_not_called()
        self.native.fuse_moe_offload_begin.assert_not_called()
        self.assertEqual(self.runtime.status(), before)

    def test_t01_invoke_passed_artifact_validate_begin_callback_check_order(self):
        artifact = replace(self.config, num_banks=1)
        events = []
        self.native.fuse_moe_offload_validate.side_effect = lambda *args: events.append(
            ("validate", args[1]))
        self.native.fuse_moe_offload_begin.side_effect = lambda *_: events.append(("begin",))
        self.native.fuse_moe_offload_check.side_effect = lambda *_: events.append(("check",))
        result = object()

        def callback(*args):
            events.append(("callback",))
            self.assertIs(args[2], self.runtime.token)
            return result

        self.assertIs(self.runtime.invoke(callback, [None] * 6,
            artifact_config=artifact, bindings=self.bindings), result)
        self.assertEqual(events, [("validate", artifact.manifest()), ("begin",),
                                  ("callback",), ("check",)])
        self.assertIs(self.runtime.deployment_config, self.config)
        events.clear()
        self.native.fuse_moe_offload_validate.side_effect = RuntimeError("artifact mismatch")
        callback = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "artifact mismatch"):
            self.runtime.invoke(callback, [None] * 6, artifact_config=artifact,
                                bindings=self.bindings)
        callback.assert_not_called()
        self.assertEqual(events, [])

    def test_t02_callback_baseexception_still_ends_managed_scope(self):
        class Cancelled(BaseException):
            pass

        callback = mock.Mock(side_effect=Cancelled("cancelled"))
        with self.assertRaises(Cancelled):
            self.runtime.invoke(callback, [None] * 6, artifact_config=self.config,
                                bindings=self.bindings)
        self.native.fuse_moe_offload_begin.assert_called_once_with(self.runtime.token)
        self.native.fuse_moe_offload_check.assert_called_once_with(self.runtime.token)

    def test_t02_invoke_frame_output_and_callback_error_weakref_lifetime(self):
        # Retain the real invoke frame without installing a trace/profiler hook.
        # Returned results belong to the caller; failed results must not belong
        # to that retained frame. Exception tracebacks legitimately own callback
        # locals until their completed frames are explicitly cleared.
        for phase in ("success", "check_error", "callback_error"):
            with self.subTest(phase=phase):
                frames, refs = [], []
                self.native.fuse_moe_offload_check.side_effect = (
                    RuntimeError("native check failure") if phase == "check_error" else None)

                def callback(*args):
                    frames.append(sys._getframe(1))
                    output = torch.ones(3, dtype=torch.float32)
                    refs.append(weakref.ref(output))
                    if phase == "callback_error":
                        raise ValueError("callback failure")
                    return output

                if phase == "success":
                    output = self.runtime.invoke(callback, [None] * 6,
                        artifact_config=self.config, bindings=self.bindings)
                    self.assertIs(refs[0](), output)
                    self.assertIsNone(frames[0].f_locals["output"])
                    self.assertIsNone(frames[0].f_locals["callback_error"])
                    gc.collect()
                    self.assertIs(refs[0](), output)  # Caller ownership is intact.
                    del output
                else:
                    expected = RuntimeError if phase == "check_error" else ValueError
                    try:
                        self.runtime.invoke(callback, [None] * 6,
                            artifact_config=self.config, bindings=self.bindings)
                    except expected as error:
                        self.assertEqual(str(error), "native check failure" if phase == "check_error"
                                         else "callback failure")
                        self.assertIsNone(frames[0].f_locals["output"])
                        self.assertIsNone(frames[0].f_locals["callback_error"])
                        if phase == "check_error":
                            gc.collect()
                            self.assertIsNone(refs[0]())
                        else:
                            self.assertIsNotNone(refs[0]())  # Callback traceback owns it.
                        traceback.clear_frames(error.__traceback__)
                    else:
                        self.fail("invoke must preserve the original failure")
                gc.collect()
                self.assertIsNone(refs[0]())
                frames.clear()
        self.native.fuse_moe_offload_check.side_effect = None


class ArchiveResourceTests(unittest.TestCase):
    def test_t05_new_weight_generation_preserves_deterministic_config(self):
        generator = torch.Generator().manual_seed(5105)
        layers = {layer: (torch.randn(e, 2 * i, 64, generator=generator).half(),
                          torch.randn(e, 64, i, generator=generator).half())
                  for layer, e, i in ((5, 3, 128), (0, 7, 64))}
        store = ExpertStore(layers)
        config = OffloadConfig.from_store(store, gpu_budget_bytes=2 << 20,
            ram_budget_bytes=4 << 20, pinned_budget_bytes=200000, target_sm=89)
        other = OffloadConfig.from_store(ExpertStore(dict(reversed(list(layers.items())))),
            **config.runtime_options(), target_sm=89)
        self.assertEqual(config.manifest(), other.manifest())
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(ops, "ensure_loaded") as load:
            path = Path(temporary) / "synthetic.pt"
            store.save_cpp(path, weight_version=41, config=config)
            old = ExpertStore.load_cpp(path)
            replacement = ExpertStore({layer: tuple(w + .01 for w in pair)
                                       for layer, pair in layers.items()})
            replacement.save_cpp(path, weight_version=42, config=config)
            new = ExpertStore.load_cpp(path)
            self.assertEqual((old.weight_version, new.weight_version), (41, 42))
            self.assertEqual(old.deployment_config, new.deployment_config)
            for layer in layers:
                for a, b in zip(old.layers[layer], new.layers[layer]):
                    self.assertFalse(torch.equal(a, b))
                    self.assertNotEqual(a.untyped_storage()._cdata, b.untyped_storage()._cdata)
            load.assert_not_called()

    def test_t05_archive_load_peak_resource_observation_separate_process(self):
        # Isolate process high-water RSS from previous tests. This is synthetic
        # archive-load telemetry, NOT a bound on a business checkpoint peak.
        store = ExpertStore({0: (torch.zeros(32, 128, 64, dtype=torch.float16),
                                 torch.zeros(32, 64, 64, dtype=torch.float16))})
        script = (
            "import json,resource,sys,torch; "
            "from custom_ops.moe_offload import ExpertStore; "
            "before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
            "s=ExpertStore.load_cpp(sys.argv[1]); "
            "after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; "
            "print(json.dumps(dict(before_kib=before,peak_kib=after,"
            "weight_bytes=s.weight_bytes,layers=list(s.layers),"
            "cpu=all(w.device.type=='cpu' for p in s.layers.values() for w in p))))"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.pt"
            store.save_cpp(path, weight_version=1)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
            process = subprocess.run([sys.executable, "-c", script, str(path)],
                env=env, capture_output=True, text=True, timeout=20, check=True)
            report = json.loads(process.stdout.strip().splitlines()[-1])
        self.assertEqual(report["weight_bytes"], store.weight_bytes)
        self.assertEqual(report["layers"], [0])
        self.assertTrue(report["cpu"])
        self.assertGreater(report["peak_kib"], 0)
        self.assertGreaterEqual(report["peak_kib"], report["before_kib"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ManagedRuntimeTests(unittest.TestCase):
    # Reuse fixtures, not inheritance: inheriting OffloadTests reruns its entire
    # historical Cartesian suite in each new class.
    tensors = OffloadTests.tensors
    setUpClass = classmethod(OffloadTests.setUpClass.__func__)

    def run(self, result=None):
        guarded = self._testMethodName.startswith("test_t06_")
        if guarded:
            # Future.result cannot interrupt a native call holding the GIL, and
            # ThreadPoolExecutor joins at interpreter exit even with wait=False.
            # A process-level watchdog is required, not an unbounded helper join.
            faulthandler.dump_traceback_later(20, exit=True)
        try:
            return super().run(result)
        finally:
            if guarded and not getattr(self, "_native_timeout", False):
                faulthandler.cancel_dump_traceback_later()

    def configured(self, store, **options):
        major, minor = torch.cuda.get_device_capability()
        defaults = dict(gpu_budget_bytes=16 << 20, ram_budget_bytes=64 << 20,
                        pinned_budget_bytes=200000, target_sm=major * 10 + minor,
                        activation_mode="fp32_accumulate", max_route_tile_rows=17)
        config = OffloadConfig.from_store(store, **{**defaults, **options})
        runtime = moe_offload.PrefillMoERuntime.from_config(store, config)
        self.addCleanup(runtime.close)
        return runtime, config

    def bound_inputs(self, runtime, x, ids, scale):
        bindings = [moe_offload.InputBinding(3, "token")]
        inputs = [x, ids, scale, None]
        for layer in sorted(runtime.store.layers):
            bindings.extend((moe_offload.InputBinding(len(inputs), "w1", layer),
                             moe_offload.InputBinding(len(inputs) + 1, "w2", layer)))
            inputs.extend((None, None))
        return inputs, tuple(bindings)

    def graph(self, config, *, layer=0):
        adapter = OffloadedMoELayer(layer, config=config)
        return lambda x, ids, scale, token, w1, w2: adapter(x, w1, w2, ids, scale, token)

    def assert_status_schema(self, runtime):
        status = runtime.status()
        self.assertEqual(set(status), {"version", "state", "managed", "pending_error",
            "forward_sequence", "success_sequence", "failure_layer", "failure_sequence",
            "managed_sequence", "success_layer"})
        self.assertEqual(status["version"], 1)
        self.assertIn(status["state"], ("healthy", "recoverable", "poisoned"))
        raw = torch.ops.recsys_ops.fuse_moe_offload_status(runtime.token)
        self.assertEqual((raw.device.type, raw.dtype, tuple(raw.shape)),
                         ("cpu", torch.int64, (10,)))
        return status

    def test_t02_multilayer_caught_first_failure_blocks_downstream_and_stays_sticky(self):
        x, w1, w2, ids, scale = self.tensors(s=9, e=3, k=2)
        store = ExpertStore({0: (w1, w2), 5: (w1.clone(), w2.clone())})
        runtime, config = self.configured(store)
        inputs, bindings = self.bound_inputs(runtime, x, ids, scale)
        bad = ids.clone()
        bad[0, 0] = 3
        first = OffloadedMoELayer(0, config=config)
        second = OffloadedMoELayer(5, config=config)
        errors = []
        token, version = runtime.token.clone(), runtime.token._version

        def callback(x, ids, scale, token, a, b, c, d):
            for adapter, pair, routes in ((first, (a, b), bad), (second, (c, d), ids)):
                try:
                    adapter(x, *pair, routes, scale, token)
                except RuntimeError as error:
                    errors.append(str(error))
            return None  # Models a proxy which catches errors and continues.

        with self.assertRaises(RuntimeError) as caught:
            runtime.invoke(callback, inputs, artifact_config=config, bindings=bindings)
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0], errors[1])
        self.assertEqual(str(caught.exception), errors[0])
        failed = self.assert_status_schema(runtime)
        self.assertEqual((failed["state"], failed["managed"], failed["pending_error"]),
                         ("recoverable", 0, 1))
        self.assertEqual((failed["forward_sequence"], failed["failure_sequence"],
                          failed["failure_layer"], failed["success_sequence"]), (1, 1, 0, 0))
        torch.testing.assert_close(runtime.token, token, atol=0, rtol=0)
        self.assertEqual(runtime.token._version, version)
        blocked = mock.Mock()
        with self.assertRaises(RuntimeError):
            runtime.invoke(blocked, inputs, artifact_config=config, bindings=bindings)
        blocked.assert_not_called()
        self.assertEqual(runtime.status(), failed)
        # Read-only validation must neither clear nor replace a pending failure.
        runtime.validate_config(config)
        with self.assertRaises(RuntimeError):
            runtime.validate_config(replace(config, num_banks=1))
        self.assertEqual(runtime.status(), failed)
        torch.testing.assert_close(runtime.token, token, atol=0, rtol=0)
        self.assertEqual(runtime.token._version, version)
        # Eager retry works immediately, but does not silently acknowledge.
        out = runtime.forward(x, ids, scale, layer_id=5)
        torch.testing.assert_close(out.float().cpu(), reference(x, *store.layers[5], ids, scale),
                                   atol=.002, rtol=.04)
        sticky = runtime.status()
        self.assertEqual((sticky["failure_sequence"], sticky["failure_layer"]), (1, 0))
        self.assertEqual((sticky["success_sequence"], sticky["success_layer"]), (2, 5))
        for _ in range(2):
            with self.assertRaises(RuntimeError) as checked:
                runtime.check()
            self.assertEqual(str(checked.exception), errors[0])
            self.assertEqual(runtime.status(), sticky)
        runtime.acknowledge()
        cleared = runtime.status()
        self.assertEqual((cleared["state"], cleared["pending_error"], cleared["failure_layer"],
                          cleared["failure_sequence"]), ("healthy", 0, -1, 0))
        runtime.acknowledge()
        self.assertEqual(runtime.status(), cleared)
        runtime.check()
        recovered = runtime.invoke(
            lambda x, ids, scale, token, a, b, c, d: first(x, a, b, ids, scale, token),
            inputs, artifact_config=config, bindings=bindings)
        torch.testing.assert_close(recovered.float().cpu(), reference(x, w1, w2, ids, scale),
                                   atol=.002, rtol=.04)
        self.assertEqual(runtime.status()["state"], "healthy")

    def test_t01_preflight_wrong_artifact_weights_token_is_read_only(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=3, k=1)
        runtime, config = self.configured(ExpertStore({0: (w1, w2)}))
        inputs, bindings = self.bound_inputs(runtime, x, ids, scale)
        prepared = runtime.prepare_inputs(inputs, bindings)
        callback = mock.Mock()
        token, version = runtime.token.clone(), runtime.token._version
        before, stats = runtime.status(), runtime.stats()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        cases = [(prepared, replace(config, num_banks=1))]
        for index in (3, 4, 5):
            wrong = list(prepared)
            wrong[index] = wrong[index].clone()
            cases.append((wrong, config))
        for wrong, artifact in cases:
            with self.subTest(artifact=artifact.num_banks), self.assertRaises((ValueError, RuntimeError)):
                runtime.invoke(callback, wrong, artifact_config=artifact, bindings=bindings)
            self.assertEqual(runtime.status(), before)
            self.assertEqual(runtime.stats(), stats)
            torch.testing.assert_close(runtime.token, token, atol=0, rtol=0)
            self.assertEqual(runtime.token._version, version)
        callback.assert_not_called()
        torch.cuda.synchronize()
        self.assertEqual(torch.cuda.memory_allocated(), allocated)

    def test_t06_native_managed_owner_rejects_unauthorized_thread_read_only(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=3, k=1)
        runtime, config = self.configured(ExpertStore({0: (w1, w2)}))
        native = torch.ops.recsys_ops
        initial = native.fuse_moe_offload_status(runtime.token).clone()
        token, version = runtime.token.clone(), runtime.token._version
        executor = ThreadPoolExecutor(max_workers=1)
        futures = []
        try:
            for operation in ("forward", "check", "close"):
                with self.subTest(operation=operation):
                    native.fuse_moe_offload_begin(runtime.token)
                    before = native.fuse_moe_offload_status(runtime.token).clone()
                    self.assertEqual(before[1:8].tolist(), [0, 1, 0, 0, 0, -1, 0])

                    def unauthorized():
                        # Direct native calls bypass the Python serialization lock.
                        # No capture, CUDA work, or status polling on this thread.
                        if operation == "forward":
                            native.fuse_moe_offload(x, w1, w2, ids, scale, runtime.token,
                                0, config.layer_config(0), config.manifest())
                        elif operation == "check":
                            native.fuse_moe_offload_check(runtime.token)
                        else:
                            native.fuse_moe_offload_close(runtime.token)

                    future = executor.submit(unauthorized)
                    futures.append(future)
                    with self.assertRaisesRegex(RuntimeError, "another thread"):
                        future.result(timeout=20)
                    after = native.fuse_moe_offload_status(runtime.token)
                    torch.testing.assert_close(after, before, atol=0, rtol=0)
                    native.fuse_moe_offload_check(runtime.token)
                    native.fuse_moe_offload_acknowledge(runtime.token)
                    ended = native.fuse_moe_offload_status(runtime.token)
                    self.assertEqual(ended[:8].tolist(), initial[:8].tolist())
                    self.assertEqual(ended[9].item(), initial[9].item())
                    torch.testing.assert_close(runtime.token, token, atol=0, rtol=0)
                    self.assertEqual(runtime.token._version, version)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if not all(future.done() for future in futures):
                self._native_timeout = True
                self._cleanups.clear()

    def test_t06_same_thread_reentry_status_close_invoke_rejected(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=3, k=1)
        runtime, config = self.configured(ExpertStore({0: (w1, w2)}))
        inputs, bindings = self.bound_inputs(runtime, x, ids, scale)
        nested = mock.Mock()

        def callback(*args):
            for operation in (runtime.status, runtime.close,
                              lambda: runtime.invoke(nested, inputs,
                                  artifact_config=config, bindings=bindings)):
                with self.assertRaises(RuntimeError):
                    operation()
            return self.graph(config)(*args)

        out = runtime.invoke(callback, inputs, artifact_config=config, bindings=bindings)
        nested.assert_not_called()
        torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                   atol=.002, rtol=.04)
        self.assertEqual(runtime.status()["state"], "healthy")

    def test_t05_bounded_seed_expert_topk_edges_repeated_masked_legacy(self):
        # Four bounded cases, not E x K x dtype x mode x cache Cartesian growth.
        cases = ((501, torch.float16, 511, 127), (502, torch.bfloat16, 512, 128),
                 (503, torch.float16, 512, 128), (504, torch.bfloat16, 511, 127))
        for seed, dtype, e, k in cases:
            with self.subTest(seed=seed, dtype=dtype, e=e, k=k), torch.random.fork_rng():
                torch.manual_seed(seed)
                x, w1, w2, ids, scale = self.tensors(s=3, e=e, k=k, dtype=dtype)
                # Exercise distant experts, repeated routes and NaN masked scales.
                ids.copy_(torch.tensor([0, e - 1, e // 2, -1], device=x.device,
                                       dtype=torch.int32).repeat((k + 3) // 4)[:k].expand_as(ids))
                ids[0].fill_(-1)
                scale.div_(k)
                scale[ids == -1] = float("nan")
                runtime, config = self.configured(ExpertStore({0: (w1, w2)}),
                    pinned_budget_bytes=(e + 1) * 4 + 2 * (6 * 64 * 64 + 24),
                    max_route_tile_rows=7)
                expected = reference(x, w1, w2, ids, scale)
                legacy = ops.fuse_moe(x, w1.to(x.device), w2.to(x.device), ids, scale, backend="cuda")
                for _ in range(2):
                    out = runtime.forward(x, ids, scale)
                    torch.testing.assert_close(out.float().cpu(), expected, atol=.0002, rtol=.04)
                    torch.testing.assert_close(out.float(), legacy.float(), atol=.0002, rtol=.04)
                    self.assertTrue(torch.isfinite(out).all().item())
                    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=0, rtol=0)
                    self.assertGreater(runtime.stats()["waves"], 1)
                    del out
                runtime.close()

    def test_t05_large_s_budget_tiles_and_heterogeneous_layer_chain(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x, a, b, ids, scale = self.tensors(s=8193, e=7, k=2, dtype=dtype)
                _, c, d, ids5, scale5 = self.tensors(s=8193, e=3, i=128, k=2, dtype=dtype)
                ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
                          .reshape_as(ids).remainder(7))
                ids[::17, 1] = -1
                store = ExpertStore({0: (a, b), 5: (c, d)})
                # H stays 64 across the chain; I/E differ. Pinned cap forces
                # multiple miss waves, GPU budget excludes full S*K activations.
                runtime, _ = self.configured(store, gpu_budget_bytes=6 << 20,
                    pinned_budget_bytes=100000, max_route_tile_rows=127)
                first = runtime.forward(x, ids, scale)
                stats0 = runtime.stats()
                second = runtime.forward(first, ids5, scale5, layer_id=5)
                stats5 = runtime.stats()
                torch.testing.assert_close(first.float().cpu(), reference(x, a, b, ids, scale),
                                           atol=.002, rtol=.04)
                # Quantized first-layer output is the actual next-layer input.
                torch.testing.assert_close(second.float().cpu(), reference(first, c, d, ids5, scale5),
                                           atol=.0002, rtol=.04)
                for stats in (stats0, stats5):
                    self.assertGreater(stats["waves"], 1)
                    self.assertGreater(stats["route_tiles"], stats["waves"])
                    self.assertLessEqual(stats["route_tile_rows"], 127)
                    self.assertLessEqual(stats["gpu_charged_bytes"], 6 << 20)
                    self.assertLessEqual(stats["pinned_bytes"], 100000)
                runtime.close()

    def test_t06_concurrent_context_stream_budget_cache_failure_isolation(self):
        x, w1, w2, ids, scale = self.tensors(s=35, e=4, k=2)
        store = ExpertStore({0: (w1, w2)})
        left, lc = self.configured(store, gpu_budget_bytes=4 << 20, pinned_budget_bytes=50000)
        right, rc = self.configured(store, gpu_budget_bytes=8 << 20, pinned_budget_bytes=100000,
            resident_cache_bytes=6 * 64 * 64, resident_experts={0: [0]}, async_staging=True)
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        ready = torch.cuda.Event()
        ready.record()
        gate = threading.Barrier(2, timeout=20)
        expected = reference(x, w1, w2, ids, scale)

        def calls(runtime, config, stream, fail):
            retained = []
            with torch.cuda.device(x.device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                gate.wait()
                inputs, bindings = self.bound_inputs(runtime, x, ids, scale)
                if fail:
                    invalid = ids.clone()
                    invalid[0, 0] = 4
                    with self.assertRaises(RuntimeError):
                        runtime.forward(x, invalid, scale)
                    callback = mock.Mock()
                    with self.assertRaises(RuntimeError):
                        runtime.invoke(callback, inputs, artifact_config=config, bindings=bindings)
                    callback.assert_not_called()
                    runtime.acknowledge()
                for _ in range(3):
                    retained.append(runtime.invoke(self.graph(config), inputs,
                        artifact_config=config, bindings=bindings))
                stream.synchronize()
                return retained, runtime.status(), runtime.stats()

        executor = ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(calls, left, lc, streams[0], True),
                   executor.submit(calls, right, rc, streams[1], False)]
        try:
            results = [future.result(timeout=20) for future in futures]
        finally:
            gate.abort()
            executor.shutdown(wait=False, cancel_futures=True)
            # A timed-out native call must not trigger a blocking main-thread
            # cleanup close. The process runner must enforce its outer timeout.
            if not all(future.done() for future in futures):
                self._native_timeout = True
                self._cleanups.clear()
        for outputs, status, stats in results:
            self.assertEqual(status["state"], "healthy")
            self.assertFalse(status["pending_error"])
            self.assertEqual(len({out.data_ptr() for out in outputs}), 3)
            for out in outputs:
                torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
        self.assertEqual(results[0][1]["forward_sequence"], 4)
        self.assertEqual(results[1][1]["forward_sequence"], 3)
        self.assertEqual(results[0][2]["resident_experts"], 0)
        self.assertEqual(results[1][2]["resident_experts"], 1)
        self.assertLessEqual(results[0][2]["gpu_charged_bytes"], lc.gpu_budget_bytes)
        self.assertLessEqual(results[1][2]["gpu_charged_bytes"], rc.gpu_budget_bytes)
        left.close()
        # Closing one context cannot invalidate the other's historical outputs.
        for out in results[1][0]:
            torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
        right.check()

    def test_t06_close_serializes_with_invoke_using_deterministic_events(self):
        x, w1, w2, ids, scale = self.tensors(s=7, e=3, k=1)
        runtime, config = self.configured(ExpertStore({0: (w1, w2)}))
        inputs, bindings = self.bound_inputs(runtime, x, ids, scale)
        entered, attempted, release, closed = (threading.Event() for _ in range(4))
        stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        ready.record()

        def callback(*args):
            entered.set()
            if not release.wait(20):
                raise RuntimeError("close serialization release deadline")
            self.assertFalse(closed.is_set())
            return self.graph(config)(*args)

        def invoke():
            with torch.cuda.device(x.device), torch.cuda.stream(stream):
                stream.wait_event(ready)
                out = runtime.invoke(callback, inputs, artifact_config=config, bindings=bindings)
                stream.synchronize()
                return out

        def close():
            attempted.set()
            runtime.close()
            closed.set()

        executor = ThreadPoolExecutor(max_workers=2)
        futures = []
        try:
            futures.append(executor.submit(invoke))
            self.assertTrue(entered.wait(20))
            futures.append(executor.submit(close))
            self.assertTrue(attempted.wait(20))
            self.assertFalse(closed.is_set())
            release.set()
            out = futures[0].result(timeout=20)
            futures[1].result(timeout=20)
            self.assertTrue(closed.is_set())
            torch.testing.assert_close(out.float().cpu(), reference(x, w1, w2, ids, scale),
                                       atol=.002, rtol=.04)
            runtime.close()
        finally:
            release.set()
            executor.shutdown(wait=False, cancel_futures=True)
            if not all(future.done() for future in futures):
                self._native_timeout = True
                self._cleanups.clear()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FaultInjectionTests(unittest.TestCase):
    tensors = OffloadTests.tensors
    configured = ManagedRuntimeTests.configured
    bound_inputs = ManagedRuntimeTests.bound_inputs
    graph = ManagedRuntimeTests.graph
    assert_status_schema = ManagedRuntimeTests.assert_status_schema

    @classmethod
    def setUpClass(cls):
        OffloadTests.setUpClass.__func__(cls)
        if not all(hasattr(torch.ops.recsys_ops, name) for name in (
                "_fuse_moe_offload_test_inject", "_fuse_moe_offload_test_worker_states")):
            raise unittest.SkipTest("CUSTOM_OPS_OFFLOAD_TESTING schemas absent")

    def test_t03_worker_states_and_drain_capture_release(self):
        for _ in range(2):
            states = torch.ops.recsys_ops._fuse_moe_offload_test_worker_states()
            self.assertEqual((states.device.type, states.dtype), ("cpu", torch.int64))
            self.assertEqual(states.tolist(), [1, 1, 1, 1, 1, 1, 1, 1, 14, 5])

    def test_t03_fault_phase_sync_async_status_retry_poison_close_no_leaks(self):
        phases = (("gpu_pool_allocate", False), ("pinned_allocate", False),
                  ("thread_create", True), ("worker_running", True),
                  ("after_first_h2d", False), ("after_event_record", False),
                  ("after_kernel_submit", False), ("after_worker_ready", True),
                  ("before_queued_drain", True))
        x, w1, w2, ids, scale = self.tensors(s=35, e=4, k=2)
        ids.copy_(torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
                  .reshape_as(ids).remainder(4))
        store = ExpertStore({0: (w1, w2)})
        healthy, hc = self.configured(store)
        hi, hb = self.bound_inputs(healthy, x, ids, scale)
        warm = healthy.invoke(self.graph(hc), hi, artifact_config=hc, bindings=hb)
        expected = reference(x, w1, w2, ids, scale)
        del warm
        torch.cuda.synchronize()
        gc.collect()
        baseline = torch.cuda.memory_allocated()
        for point, async_only in phases:
            for asynchronous in ((True,) if async_only else (False, True)):
                for repetition in range(2):
                    with self.subTest(point=point, asynchronous=asynchronous, repetition=repetition):
                        runtime, config = self.configured(store, pinned_budget_bytes=50000,
                                                         async_staging=asynchronous)
                        token, version = runtime.token.clone(), runtime.token._version
                        torch.ops.recsys_ops._fuse_moe_offload_test_inject(runtime.token, point, 1)
                        inputs, bindings = self.bound_inputs(runtime, x, ids, scale)
                        with self.assertRaises(RuntimeError):
                            runtime.invoke(self.graph(config), inputs,
                                           artifact_config=config, bindings=bindings)
                        status = self.assert_status_schema(runtime)
                        recoverable = point in ("gpu_pool_allocate", "pinned_allocate")
                        self.assertEqual(status["state"], "recoverable" if recoverable else "poisoned")
                        self.assertEqual((status["managed"], status["pending_error"],
                                          status["forward_sequence"], status["failure_sequence"],
                                          status["failure_layer"], status["success_sequence"]),
                                         (0, 1, 1, 1, 0, 0))
                        torch.testing.assert_close(runtime.token, token, atol=0, rtol=0)
                        self.assertEqual(runtime.token._version, version)
                        for _ in range(2):
                            with self.assertRaises(RuntimeError):
                                runtime.check()
                            self.assertEqual(runtime.status(), status)
                        blocked = mock.Mock()
                        with self.assertRaises(RuntimeError):
                            runtime.invoke(blocked, inputs, artifact_config=config, bindings=bindings)
                        blocked.assert_not_called()
                        self.assertEqual(runtime.status(), status)
                        if recoverable:
                            recovered = runtime.forward(x, ids, scale)
                            torch.testing.assert_close(recovered.float().cpu(), expected, atol=.002, rtol=.04)
                            self.assertEqual(runtime.status()["failure_sequence"], 1)
                            del recovered
                            runtime.acknowledge()
                            self.assertEqual(runtime.status()["state"], "healthy")
                        else:
                            with self.assertRaises(RuntimeError):
                                runtime.acknowledge()
                            with self.assertRaises(RuntimeError):
                                runtime.forward(x, ids, scale)
                            self.assertEqual(runtime.status(), status)
                        # Partial H2D/event/kernel and queued worker failures must
                        # have drained before resources are released by close.
                        runtime.close()
                        runtime.close()
                        del inputs, runtime
                        torch.cuda.synchronize()
                        gc.collect()
                        self.assertEqual(torch.cuda.memory_allocated(), baseline)
                        out = healthy.invoke(self.graph(hc), hi, artifact_config=hc, bindings=hb)
                        torch.testing.assert_close(out.float().cpu(), expected, atol=.002, rtol=.04)
                        self.assertEqual(healthy.status()["state"], "healthy")
                        del out
                        torch.cuda.synchronize()


if __name__ == "__main__":
    allow_testing = "--allow-testing" in sys.argv
    if allow_testing:
        sys.argv.remove("--allow-testing")
    if "--ops-library" in sys.argv:
        index = sys.argv.index("--ops-library")
        if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
            raise SystemExit("--ops-library requires PATH")
        library = sys.argv[index + 1]
        del sys.argv[index:index + 2]
        ops.load_precompiled("fuse_moe", library, allow_testing=allow_testing)
    elif allow_testing:
        raise SystemExit("--allow-testing requires --ops-library PATH")
    if "--async-staging" in sys.argv:
        sys.argv.remove("--async-staging")
        OffloadTests.default_async_staging = True
        GEGLUTests.default_async_staging = True
        MLPTests.default_async_staging = True
    program = None
    try:
        program = unittest.main(exit=False)
    finally:
        # TestCase cleanups have already closed runtimes/released live results.
        # Release allocator reserve only here, never before a strict live-byte
        # assertion in FaultInjectionTests. CPU-only runs must not initialize CUDA.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    # Cleanup exceptions propagate above; failed tests never become exit code 0.
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
