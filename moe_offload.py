"""Graph-external RAM weight binding and graph-visible native MoE offload calls."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from contextlib import contextmanager
import threading
import torch


def _fake_offload(x, w1, w2, ids, scale, token, layer_id, deployment_config=(), deployment_manifest=()):
    torch._check(x.dim() == 2 and w1.dim() == 3 and w2.dim() == 3)
    torch._check(x.device.type == "cuda" and w1.device.type == "cpu" and w2.device.type == "cpu")
    torch._check(x.dtype in (torch.float16, torch.bfloat16) and w1.dtype == x.dtype and w2.dtype == x.dtype)
    torch._check(x.shape[1] == w1.shape[2])
    torch._check(w1.shape[0] == w2.shape[0])
    torch._check(w1.shape[1] == 2 * w2.shape[2] and w2.shape[1] == x.shape[1])
    torch._check(ids.dim() == 2 and scale.shape == ids.shape and ids.shape[0] == x.shape[0])
    torch._check(ids.dtype == torch.int32 and scale.dtype == torch.float32)
    torch._check(ids.device == x.device and scale.device == x.device)
    torch._check(token.device.type == "cpu" and token.dtype == torch.int64 and token.shape == (2,))
    torch._check(w1.shape[0] > 0 and w1.shape[0] <= 512)
    torch._check(x.shape[1] > 0 and x.shape[1] % 64 == 0)
    torch._check(w2.shape[2] > 0 and w2.shape[2] % 64 == 0)
    torch._check(ids.shape[1] > 0 and ids.shape[1] <= 128)
    if deployment_config:
        torch._check(len(deployment_config) >= 23 and deployment_config[0] == 1)
        torch._check(len(deployment_config) == 23 + deployment_config[21])
        torch._check(w1.shape[0] == deployment_config[17])
        torch._check(x.shape[1] == deployment_config[18])
        torch._check(w2.shape[2] == deployment_config[19])
        torch._check(x.dtype == (torch.float16 if deployment_config[20] == 0 else torch.bfloat16))
    return torch.empty_like(x)


@dataclass(frozen=True)
class InputBinding:
    """Explicit graph input slot: token, or a named layer's w1/w2.

    Other slots remain caller-owned. None in a bound slot is a placeholder;
    a supplied Tensor must already have the bound identity, never silently swap it.
    The graph must embed the same layer IDs and artifact configuration.
    """
    index: int
    kind: str
    layer_id: int | None = None

    def __post_init__(self):
        if type(self.index) is not int or not 0 <= self.index < (1 << 63):
            raise ValueError("binding index must be a nonnegative int64")
        if not isinstance(self.kind, str) or self.kind not in ("token", "w1", "w2"):
            raise ValueError("binding kind must be token, w1, or w2")
        if self.kind == "token":
            if self.layer_id is not None:
                raise ValueError("token binding must not have a layer_id")
        elif type(self.layer_id) is not int or not 0 <= self.layer_id < (1 << 63):
            raise ValueError("weight binding requires a nonnegative int64 layer_id")


@dataclass(frozen=True)
class _TensorIdentity:
    storage: object
    offset: int
    sizes: tuple
    strides: tuple
    dtype: torch.dtype
    device: torch.device
    version: int

    @classmethod
    def capture(cls, tensor):
        return cls(tensor.untyped_storage(), tensor.storage_offset(), tuple(tensor.shape),
                   tuple(tensor.stride()), tensor.dtype, tensor.device, tensor._version)

    def validate(self, tensor, *, weight):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or
                tensor.untyped_storage()._cdata != self.storage._cdata or
                tensor.storage_offset() != self.offset or tuple(tensor.shape) != self.sizes or
                tuple(tensor.stride()) != self.strides or tensor.dtype != self.dtype or
                tensor.device != self.device):
            raise ValueError("offload bound tensor identity/shape/stride/dtype changed")
        if weight and (tensor.is_inference() or tensor._version != self.version):
            raise ValueError("offload weight version changed; create a new runtime")


_RUNTIME_ACCESS = threading.local()
_STATUS = ("version", "state", "managed", "pending_error", "forward_sequence",
           "success_sequence", "failure_layer", "failure_sequence", "managed_sequence",
           "success_layer")


@dataclass(frozen=True)
class LayerKeys:
    """Explicit checkpoint layout: fused gate/up or separate gate + up, no guessing."""
    down: str
    gate_up: str | None = None
    gate: str | None = None
    up: str | None = None


class _ExpertArchive(torch.nn.Module):
    """Tensor-only archive consumed by Python or official C++ torch::jit::load."""

    def __init__(self, layers, weight_version, config=None):
        super().__init__()
        self.register_buffer("format_version", torch.tensor([1 if config is None else 2], dtype=torch.int64))
        self.register_buffer("weight_version", torch.tensor([weight_version], dtype=torch.int64))
        self.register_buffer("layer_ids", torch.tensor(sorted(layers), dtype=torch.int64))
        for layer_id in sorted(layers):
            if config is not None:
                self.register_buffer(f"offload_config_{layer_id}",
                                     torch.tensor(config.layer_config(layer_id), dtype=torch.int64))
            for name, tensor in zip(("w1", "w2"), layers[layer_id]):
                self.register_buffer(f"{name}_{layer_id}", tensor.detach().clone())

    def forward(self) -> torch.Tensor:
        return self.layer_ids


class ExpertStore:
    """Canonical CPU weights. Training updates require a new store/context."""

    def __init__(self, layers, *, ram_budget_bytes=0):
        if not isinstance(layers, Mapping):
            raise ValueError("layers must be a mapping of layer IDs to (w1,w2)")
        self.layers = dict(layers)
        self.ram_budget_bytes = ram_budget_bytes
        if type(ram_budget_bytes) is not int or not 0 <= ram_budget_bytes < (1 << 63):
            raise ValueError("ram_budget_bytes must be a nonnegative int64")
        storages = {}
        for layer_id, pair in self.layers.items():
            if (isinstance(layer_id, bool) or not isinstance(layer_id, int)
                    or not 0 <= layer_id < (1 << 63)
                    or not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray))
                    or len(pair) != 2):
                raise ValueError("layers must map nonnegative integer IDs to (w1,w2)")
            w1, w2 = pair
            if (not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor) or
                    w1.device.type != "cpu" or w2.device.type != "cpu" or
                    w1.dtype not in (torch.float16, torch.bfloat16) or w2.dtype != w1.dtype or
                    w1.ndim != 3 or w2.ndim != 3 or not w1.is_contiguous() or not w2.is_contiguous() or
                    w1.requires_grad or w2.requires_grad or w1.is_pinned() or w2.is_pinned() or
                    w1.is_inference() or w2.is_inference()):
                raise ValueError("canonical weights must be contiguous pageable CPU fp16/bf16 tensors with version counters")
            e, n, h = w1.shape
            if (e <= 0 or e > 512 or w2.shape[:2] != (e, h) or n != 2 * w2.shape[2] or
                    h <= 0 or h > _INT_MAX or w2.shape[2] <= 0 or
                    w2.shape[2] > _INT_MAX // 2 or h % 64 or w2.shape[2] % 64):
                raise ValueError("invalid canonical E/H/I or unsupported kernel alignment")
            for tensor in pair:
                storage = tensor.untyped_storage()
                storages[storage._cdata] = storage.nbytes()
        self.weight_bytes = sum(storages.values())
        if ram_budget_bytes and self.weight_bytes > ram_budget_bytes:
            raise ValueError("CPU backing storages exceed ram_budget_bytes")

    @classmethod
    @torch.inference_mode(False)
    def from_state_dict(cls, state_dict, layer_keys, *, dtype=None, ram_budget_bytes=0):
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("explicit target dtype must be fp16 or bf16")
        layers = {}
        for layer_id, keys in layer_keys.items():
            fused = keys.gate_up is not None
            if (fused and (keys.gate is not None or keys.up is not None)) or (
                    not fused and (keys.gate is None or keys.up is None)):
                raise ValueError("specify either fused gate_up or separate gate and up")

            def get(key):
                value = state_dict[key]
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.ndim != 3:
                    raise ValueError(f"{key}: expected CPU 3D tensor (explicitly adapt expert lists/transposes first)")
                if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f"{key}: quantized/unsupported representation")
                if dtype is None and value.dtype not in (torch.float16, torch.bfloat16):
                    raise ValueError(f"{key}: FP32 conversion requires explicit dtype")
                if value.is_inference():
                    value = value.clone()
                return value.detach().to(dtype=dtype or value.dtype).contiguous()

            down = get(keys.down)
            if fused:
                gate_up = get(keys.gate_up)
            else:
                gate, up = get(keys.gate), get(keys.up)
                if gate.shape != up.shape or gate.dtype != up.dtype:
                    raise ValueError("gate/up shape and dtype must match")
                gate_up = torch.cat((gate, up), dim=1)
            layers[layer_id] = (gate_up, down)
        return cls(layers, ram_budget_bytes=ram_budget_bytes)

    @classmethod
    @torch.inference_mode(False)
    def load(cls, path: str | Path, layer_keys, *, dtype=None, ram_budget_bytes=0,
             state_dict_field=None, mmap=False):
        """Trusted torch.save(state_dict) input, always CPU and weights_only=True.

        The RAM budget validates retained storages, not torch.load/packing peak RSS.
        Callers must reserve checkpoint/conversion peak before loading large files.
        mmap is optional serialization mapping, not an SSD execution backend.
        """
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
        if state_dict_field is not None:
            state = state[state_dict_field]
        return cls.from_state_dict(state, layer_keys, dtype=dtype, ram_budget_bytes=ram_budget_bytes)


    @torch.inference_mode(False)
    def save_cpp(self, path: str | Path, *, weight_version=0, config: OffloadConfig | None = None):
        """Save a CPU TorchScript buffer archive; no Python pickle loader in C++."""
        if (isinstance(weight_version, bool) or not isinstance(weight_version, int)
                or not 0 <= weight_version < (1 << 63)):
            raise ValueError("weight_version must be a nonnegative int64")
        # Revalidate publicly accessible layers before writing, and snapshot buffers.
        validated = type(self)(self.layers, ram_budget_bytes=self.ram_budget_bytes)
        if not validated.layers:
            raise ValueError("C++ archive requires at least one layer")
        if config is not None:
            if not isinstance(config, OffloadConfig):
                raise TypeError("config must be an OffloadConfig")
            config.validate_store(validated)
        snapshot = _ExpertArchive(validated.layers, weight_version, config)
        if config is not None:
            # Independent clones can expand shared/overlapping backing storages.
            # Validate retained snapshot bytes before opening the destination.
            snapshot_store = ExpertStore({
                layer: (getattr(snapshot, f"w1_{layer}"), getattr(snapshot, f"w2_{layer}"))
                for layer in validated.layers})
            config.validate_store(snapshot_store)
        torch.jit.save(torch.jit.script(snapshot), str(path))

    @classmethod
    @torch.inference_mode(False)
    def load_cpp(cls, path: str | Path, *, ram_budget_bytes=0):
        """Load the versioned standard archive, validating metadata and canonical tensors."""
        archive = torch.jit.load(str(path), map_location="cpu")
        buffers = dict(archive.named_buffers())
        for name in ("format_version", "weight_version", "layer_ids"):
            value = buffers.get(name)
            if (value is None or value.dtype != torch.int64 or value.ndim != 1
                    or value.device.type != "cpu"):
                raise ValueError(f"invalid archive metadata: {name}")
        format_version = buffers["format_version"].tolist()
        if format_version not in ([1], [2]):
            raise ValueError("unsupported archive format_version")
        versions = buffers["weight_version"].tolist()
        if len(versions) != 1 or versions[0] < 0:
            raise ValueError("invalid archive weight_version")
        ids = buffers["layer_ids"].tolist()
        if not ids or ids != sorted(set(ids)) or ids[0] < 0:
            raise ValueError("archive layer_ids must be nonempty, sorted, unique and nonnegative")
        expected = {"format_version", "weight_version", "layer_ids"}
        expected.update(f"{name}_{layer_id}" for layer_id in ids for name in ("w1", "w2"))
        if format_version == [2]:
            expected.update(f"offload_config_{layer_id}" for layer_id in ids)
        if set(buffers) != expected or list(archive.named_parameters()):
            raise ValueError("archive buffers do not match layer_ids")
        store = cls({layer_id: (buffers[f"w1_{layer_id}"], buffers[f"w2_{layer_id}"])
                     for layer_id in ids}, ram_budget_bytes=ram_budget_bytes)
        store.weight_version = versions[0]
        if format_version == [2]:
            configs = {}
            for layer_id in ids:
                value = buffers[f"offload_config_{layer_id}"]
                if value.device.type != "cpu" or value.dtype != torch.int64 or value.ndim != 1:
                    raise ValueError("invalid archive deployment configuration metadata")
                configs[layer_id] = tuple(value.tolist())
            store.deployment_config = OffloadConfig.from_layer_configs(configs)
            store.deployment_config.validate_store(store)
        return store


_STATS = ("active_experts", "waves", "weight_h2d_bytes", "metadata_h2d_bytes",
          "gpu_charged_bytes", "pinned_bytes", "ram_charged_bytes", "num_banks", "wave_capacity",
          "activation_mode", "route_tiles", "route_tile_rows", "buffer_allocations",
          "buffer_reuses", "gpu_pool_bytes", "pinned_pool_bytes",
          "cache_hits", "cache_misses", "cache_hit_weight_bytes", "resident_cache_bytes",
          "resident_experts", "resident_load_h2d_bytes",
          "dynamic_cache_bytes", "dynamic_experts", "dynamic_cache_hits",
          "cache_admissions", "cache_evictions", "cache_bypasses",
          "cache_promotion_d2d_bytes", "cache_budget_evictions", "cache_metadata_bytes")
_ACTIVATION_MODES = {"route_buffer": 0, "fp32_accumulate": 1, "auto": 2}
_CACHE_POLICIES = {"reuse": 0, "lru": 1}
_INT_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class OffloadConfig:
    """Versioned offline deployment contract; all budgets are explicit bytes.

    layer_specs contains immutable (layer_id, E, H, I, dtype_code, resident_ids)
    tuples. tile_m=0 fixes the deterministic shape-based policy, not an env override.
    CPU archives carry the same contract used as constant arguments in the AOTI graph.
    """
    gpu_budget_bytes: int
    ram_budget_bytes: int
    pinned_budget_bytes: int
    target_sm: int
    layer_specs: tuple
    num_banks: int = 2
    activation_mode: str = "route_buffer"
    max_route_tile_rows: int | str = 4096
    resident_cache_bytes: int = 0
    dynamic_cache_bytes: int = 0
    cache_policy: str = "reuse"
    cache_decay_interval: int = 32
    cache_min_reuse: int = 2
    cache_update_bytes: int = 0
    async_staging: bool = False
    host_profile: bool = False
    tile_m: int = 0
    tile_n: int = 128

    def __post_init__(self):
        def integer(name, value, minimum=0, maximum=(1 << 63) - 1):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        for name in ("gpu_budget_bytes", "ram_budget_bytes", "pinned_budget_bytes"):
            integer(name, getattr(self, name), 1)
        for name in ("resident_cache_bytes", "dynamic_cache_bytes", "cache_update_bytes"):
            integer(name, getattr(self, name))
        integer("target_sm", self.target_sm, 80, _INT_MAX)
        integer("num_banks", self.num_banks, 1, 2)
        for name in ("cache_decay_interval", "cache_min_reuse"):
            integer(name, getattr(self, name), 2, _INT_MAX)
        if not isinstance(self.activation_mode, str) or self.activation_mode not in _ACTIVATION_MODES:
            raise ValueError("invalid activation_mode")
        if not isinstance(self.cache_policy, str) or self.cache_policy not in _CACHE_POLICIES:
            raise ValueError("invalid cache_policy")
        if not (isinstance(self.max_route_tile_rows, str) and self.max_route_tile_rows == "auto"):
            integer("max_route_tile_rows", self.max_route_tile_rows, 1, _INT_MAX)
        for name in ("async_staging", "host_profile"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name, values in (("tile_m", (0, 32, 64, 128)), ("tile_n", (64, 128))):
            if type(getattr(self, name)) is not int or getattr(self, name) not in values:
                raise ValueError(f"invalid {name}")
        if self.resident_cache_bytes > self.gpu_budget_bytes:
            raise ValueError("resident_cache_bytes exceeds gpu_budget_bytes")
        if (not isinstance(self.layer_specs, Sequence) or
                isinstance(self.layer_specs, (str, bytes, bytearray))):
            raise ValueError("layer_specs must be a sequence of layer descriptions")
        specs = []
        resident_charge = 0
        for spec in self.layer_specs:
            if (not isinstance(spec, Sequence) or isinstance(spec, (str, bytes, bytearray))
                    or len(spec) != 6):
                raise ValueError("layer_specs must contain six-field layer descriptions")
            layer, e, h, i, dtype, ids = spec
            integer("layer_id", layer)
            integer("E", e, 1, 512)
            integer("H", h, 64, _INT_MAX)
            integer("I", i, 64, _INT_MAX // 2)
            integer("dtype_code", dtype, 0, 1)
            if h % 64 or i % 64:
                raise ValueError("H/I must be aligned to 64")
            if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
                raise ValueError("resident expert IDs must be a sequence of integers")
            ids = tuple(ids)
            for expert in ids:
                integer("resident expert", expert, 0, e - 1)
            if len(set(ids)) != len(ids) or (ids and not self.resident_cache_bytes):
                raise ValueError("invalid resident expert IDs or cache cap")
            resident_charge += ((6 * h * i + 511) // 512) * 512 * len(ids)
            specs.append((layer, e, h, i, dtype, tuple(sorted(ids))))
        if resident_charge > self.resident_cache_bytes:
            raise ValueError("static resident experts exceed resident_cache_bytes")
        if resident_charge > self.gpu_budget_bytes:
            raise ValueError("static resident experts exceed gpu_budget_bytes")
        specs.sort(key=lambda spec: spec[0])
        if not specs or len({s[0] for s in specs}) != len(specs):
            raise ValueError("layer_specs must be nonempty with unique layer IDs")
        object.__setattr__(self, "layer_specs", tuple(specs))

    @classmethod
    def from_store(cls, store: ExpertStore, *, resident_experts=None, **options):
        if not isinstance(store, ExpertStore):
            raise TypeError("store must be an ExpertStore")
        validated = ExpertStore(store.layers)
        if resident_experts is not None and not isinstance(resident_experts, Mapping):
            raise ValueError("resident_experts must be a mapping of layer IDs to ID sequences")
        resident = {} if resident_experts is None else dict(resident_experts)
        if any(type(layer) is not int or layer not in validated.layers for layer in resident):
            raise ValueError("resident_experts requires existing layer IDs")
        specs = tuple((layer, w1.shape[0], w1.shape[2], w2.shape[2],
                       int(w1.dtype == torch.bfloat16), resident.get(layer, ()))
                      for layer, (w1, w2) in validated.layers.items())
        config = cls(layer_specs=specs, **options)
        config.validate_store(validated)
        return config

    def validate_store(self, store: ExpertStore):
        if not isinstance(store, ExpertStore):
            raise TypeError("store must be an ExpertStore")
        # Public layers and cached accounting may have changed since construction.
        validated = ExpertStore(store.layers)
        actual = tuple(sorted((layer, w1.shape[0], w1.shape[2], w2.shape[2],
                               int(w1.dtype == torch.bfloat16))
                              for layer, (w1, w2) in validated.layers.items()))
        if actual != tuple(spec[:5] for spec in self.layer_specs):
            raise ValueError("deployment configuration mismatch: store layers/weights")
        if validated.weight_bytes > self.ram_budget_bytes:
            raise ValueError("CPU backing storages exceed ram_budget_bytes")

    def runtime_options(self):
        names = ("gpu_budget_bytes", "ram_budget_bytes", "pinned_budget_bytes", "num_banks",
                 "activation_mode", "max_route_tile_rows", "resident_cache_bytes",
                 "dynamic_cache_bytes", "cache_policy", "cache_decay_interval", "cache_min_reuse",
                 "cache_update_bytes", "async_staging", "host_profile", "tile_m", "tile_n")
        return {**{name: getattr(self, name) for name in names},
                "resident_experts": {spec[0]: spec[5] for spec in self.layer_specs if spec[5]}}

    def layer_config(self, layer_id):
        for layer, e, h, i, dtype, ids in self.layer_specs:
            if layer == layer_id:
                return (1, self.gpu_budget_bytes, self.ram_budget_bytes, self.pinned_budget_bytes,
                        self.num_banks, _ACTIVATION_MODES[self.activation_mode],
                        0 if self.max_route_tile_rows == "auto" else self.max_route_tile_rows,
                        self.resident_cache_bytes, self.dynamic_cache_bytes,
                        _CACHE_POLICIES[self.cache_policy], self.cache_decay_interval,
                        self.cache_min_reuse, self.cache_update_bytes, int(self.async_staging),
                        int(self.host_profile), self.tile_m, self.tile_n,
                        e, h, i, dtype, len(ids), self.target_sm, *ids)
        raise ValueError(f"layer {layer_id} is not in the deployment configuration")

    def manifest(self):
        """Complete model contract, checked before any layer executes."""
        values = [1, len(self.layer_specs)]
        for layer, *_ in self.layer_specs:
            config = self.layer_config(layer)
            values.extend((layer, len(config), *config))
        return tuple(values)

    @classmethod
    def from_layer_configs(cls, configs):
        if not isinstance(configs, Mapping) or not configs:
            raise ValueError("invalid deployment configuration format: expected nonempty mapping")
        normalized = {}
        for layer, value in configs.items():
            if (type(layer) is not int or not 0 <= layer < (1 << 63) or
                    not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray))):
                raise ValueError("invalid deployment configuration format: invalid layer or vector")
            config = tuple(value)
            if (len(config) < 23 or any(type(field) is not int or
                    not -(1 << 63) <= field < (1 << 63) for field in config)):
                raise ValueError("invalid deployment configuration format: expected int64 fields")
            if (config[0] != 1 or config[5] not in (0, 1, 2) or config[9] not in (0, 1) or
                    config[13] not in (0, 1) or config[14] not in (0, 1) or
                    config[6] < 0 or not 0 <= config[21] <= 512 or
                    len(config) != 23 + config[21]):
                raise ValueError("invalid deployment configuration format")
            normalized[layer] = config
        configs = normalized
        first = next(iter(configs.values()))
        specs = []
        for layer, config in configs.items():
            if (len(config) < 23 or config[:17] != first[:17] or config[22] != first[22]
                    or config[21] < 0 or len(config) != 23 + config[21]):
                raise ValueError("inconsistent deployment configuration format")
            specs.append((layer, *config[17:21], tuple(config[23:])))
        result = cls(first[1], first[2], first[3], first[22], tuple(specs),
                     num_banks=first[4], activation_mode=tuple(_ACTIVATION_MODES)[first[5]],
                     max_route_tile_rows=first[6] or "auto", resident_cache_bytes=first[7],
                     dynamic_cache_bytes=first[8], cache_policy=tuple(_CACHE_POLICIES)[first[9]],
                     cache_decay_interval=first[10], cache_min_reuse=first[11],
                     cache_update_bytes=first[12], async_staging=bool(first[13]),
                     host_profile=bool(first[14]), tile_m=first[15], tile_n=first[16])
        if any(result.layer_config(layer) != tuple(value) for layer, value in configs.items()):
            raise ValueError("noncanonical deployment configuration format")
        return result


class PrefillMoERuntime:
    """One serialized native context shared across layers; outputs never alias workspace.

    Budgets cover context buffers and live outputs, not unrelated model tensors,
    CUDA allocator reserve, driver/context memory, or other runtime instances.
    Invocations drain before returning; internal tensor pools are reused across
    calls and released by close(). Allocation/reuse stats exclude output tensors
    and count only internal tensors newly created/reused by the latest call.
    Pool high-water marks are trimmed to respect budgets, including live outputs.
    Zero budgets request initialization-time auto detection.
    async_staging=True enables one lazy persistent host producer, explicitly
    fixed at context creation; environment variables are ignored. The producer
    reuses existing banks, charges its bounded stack/bookkeeping to RAM, and is
    drained on failure and joined by close(). GPU/pinned budgets are unchanged.
    activation_mode selects route_buffer, fp32_accumulate, or budget-aware auto;
    max_route_tile_rows caps FP32 route tiles (the planner may choose fewer rows).
    'auto' removes the fixed cap and chooses the smallest aligned buffer attaining
    the budget-feasible tile count. It opportunistically leaves one eligible miss
    expert of cache headroom, without sacrificing banks/group capacity or evicting
    existing cache entries just to enlarge tiles. The default remains 4096.
    route_buffer does not tile; activation_mode='auto' still minimizes GPU charge.
    resident_experts selects a static deployment cache, loaded synchronously after
    all CPU weights are bound and before any forward. It is never changed by routing.
    resident_cache_bytes is a cap, not a reservation: only actual resident GPU
    allocations count against gpu_budget_bytes. Resident loads need no pinned budget.
    dynamic_cache_bytes independently caps evictable experts (zero disables it).
    reuse admission counts distinct demand experts once per successful layer call;
    heat halves every cache_decay_interval calls (>=2), including empty calls.
    cache_min_reuse is the admission threshold; lru instead admits every feasible
    miss as a baseline. Both policies bypass when staging/workspace, pinned/RAM,
    leases or the update cap prohibit promotion. Fixed residents are never evicted.
    cache_update_bytes limits promotion D2D per invocation; zero uses the dynamic
    cap. Promotion reuses an already loaded miss bank without extra weight H2D,
    drains before publication, and becomes a hit only on a subsequent invocation.
    Eviction uses byte-aware victim sets; reuse requires benefit to exceed victim
    benefit plus a 25% candidate-byte hysteresis. Under GPU pressure, the planner
    first reclaims inactive/cross-layer entries and commits evictions only after
    validating routes and proving the complete execution fits all budgets.
    Invalid routes and infeasible budget plans preserve cache entries and heat;
    workspace may still be trimmed during validation. This is not full-forward
    rollback for allocator OOM or asynchronous CUDA failures.
    Statistics distinguish static, dynamic, promotion D2D, and last-successful-call
    admission/eviction/bypass counters.
    """

    def __init__(self, store: ExpertStore, *, device=None, gpu_budget_bytes=0,
                 ram_budget_bytes=0, pinned_budget_bytes=0, num_banks=2,
                 activation_mode="route_buffer", max_route_tile_rows=4096,
                 resident_cache_bytes=0, resident_experts=None, dynamic_cache_bytes=0,
                 cache_policy="reuse", cache_decay_interval=32, cache_min_reuse=2,
                 cache_update_bytes=0, async_staging=False, host_profile=False,
                 tile_m=0, tile_n=128):
        self._lock = threading.Lock()
        self._closed = True
        if not isinstance(store, ExpertStore):
            raise TypeError("store must be an ExpertStore")
        # Own immutable binding inventory and identity snapshots even if callers
        # later replace the public store mapping or mutate aliases.
        validated = ExpertStore(store.layers, ram_budget_bytes=store.ram_budget_bytes)
        self._bound_layers = MappingProxyType(dict(validated.layers))
        self._weight_identities = {
            layer: tuple(_TensorIdentity.capture(w) for w in pair)
            for layer, pair in self._bound_layers.items()}
        for name, flag in (("async_staging", async_staging), ("host_profile", host_profile)):
            if type(flag) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name, value, allowed in (("tile_m", tile_m, (0, 32, 64, 128)),
                                     ("tile_n", tile_n, (64, 128))):
            if type(value) is not int or value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if not isinstance(activation_mode, str) or activation_mode not in _ACTIVATION_MODES:
            raise ValueError("activation_mode must be route_buffer, fp32_accumulate, or auto")
        if not (isinstance(max_route_tile_rows, str) and max_route_tile_rows == "auto"):
            if (isinstance(max_route_tile_rows, bool) or not isinstance(max_route_tile_rows, int)
                    or not 1 <= max_route_tile_rows <= _INT_MAX):
                raise ValueError("max_route_tile_rows must be an integer in [1, INT_MAX] or 'auto'")
        if (isinstance(resident_cache_bytes, bool) or not isinstance(resident_cache_bytes, int)
                or resident_cache_bytes < 0):
            raise ValueError("resident_cache_bytes must be a nonnegative integer")
        if (isinstance(dynamic_cache_bytes, bool) or not isinstance(dynamic_cache_bytes, int)
                or dynamic_cache_bytes < 0):
            raise ValueError("dynamic_cache_bytes must be a nonnegative integer")
        if not isinstance(cache_policy, str) or cache_policy not in _CACHE_POLICIES:
            raise ValueError("cache_policy must be reuse or lru")
        if (isinstance(cache_decay_interval, bool) or not isinstance(cache_decay_interval, int)
                or not 2 <= cache_decay_interval <= _INT_MAX):
            raise ValueError("cache_decay_interval must be an integer in [2, INT_MAX]")
        if (isinstance(cache_min_reuse, bool) or not isinstance(cache_min_reuse, int)
                or not 2 <= cache_min_reuse <= _INT_MAX):
            raise ValueError("cache_min_reuse must be an integer in [2, INT_MAX]")
        if (isinstance(cache_update_bytes, bool) or not isinstance(cache_update_bytes, int)
                or cache_update_bytes < 0):
            raise ValueError("cache_update_bytes must be a nonnegative integer")
        if resident_experts is not None and not isinstance(resident_experts, Mapping):
            raise ValueError("resident_experts must be a mapping of existing layer IDs to ID sequences")
        resident = {}
        for layer_id, expert_ids in (() if resident_experts is None else resident_experts.items()):
            if (isinstance(layer_id, bool) or not isinstance(layer_id, int)
                    or layer_id not in store.layers):
                raise ValueError("resident_experts requires existing integer store layer IDs")
            if not isinstance(expert_ids, Sequence) or isinstance(expert_ids, (str, bytes, bytearray)):
                raise ValueError("resident_experts values must be sequences of integer expert IDs")
            ids = tuple(expert_ids)
            expert_count = store.layers[layer_id][0].shape[0]
            if any(isinstance(e, bool) or not isinstance(e, int) or e < 0 or e >= expert_count for e in ids):
                raise ValueError("resident_experts IDs must be non-bool integers in range")
            if len(set(ids)) != len(ids):
                raise ValueError("resident_experts IDs must be unique within each layer")
            if ids and not resident_cache_bytes:
                raise ValueError("nonempty resident_experts requires positive resident_cache_bytes")
            resident[layer_id] = ids
        self.activation_mode = activation_mode
        self.max_route_tile_rows = max_route_tile_rows
        self.resident_cache_bytes = resident_cache_bytes
        self.resident_experts = MappingProxyType(resident)
        self.dynamic_cache_bytes = dynamic_cache_bytes
        self.cache_policy = cache_policy
        self.cache_decay_interval = cache_decay_interval
        self.cache_min_reuse = cache_min_reuse
        self.cache_update_bytes = cache_update_bytes
        self.async_staging = async_staging
        self.host_profile = host_profile
        self.tile_m = tile_m
        self.tile_n = tile_n
        self._deployment_contracts = {}
        self._deployment_manifest = ()
        from custom_ops import ops
        ops.ensure_loaded("fuse_moe")
        if not ops.is_available("fuse_moe"):
            raise RuntimeError(ops.load_error("fuse_moe"))
        self.store = store
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("offload requires CUDA")
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self.token = torch.ops.recsys_ops.fuse_moe_offload_create(
            index, gpu_budget_bytes, ram_budget_bytes or store.ram_budget_bytes,
            pinned_budget_bytes, num_banks, _ACTIVATION_MODES[activation_mode],
            0 if max_route_tile_rows == "auto" else max_route_tile_rows,
            resident_cache_bytes, dynamic_cache_bytes, _CACHE_POLICIES[cache_policy],
            cache_decay_interval, cache_min_reuse, cache_update_bytes,
            async_staging, host_profile, tile_m, tile_n)
        self._closed = False
        try:
            self._token_identity = _TensorIdentity.capture(self.token)
            for layer_id, (w1, w2) in self._bound_layers.items():
                torch.ops.recsys_ops.fuse_moe_offload_bind(self.token, layer_id, w1, w2)
            for layer_id, expert_ids in self.resident_experts.items():
                torch.ops.recsys_ops.fuse_moe_offload_resident_bind(self.token, layer_id, list(expert_ids))
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Initialization failure remains the primary error.
            raise

    @classmethod
    def from_config(cls, store: ExpertStore, config: OffloadConfig, *, device=None):
        """Initialize solely from the offline contract, without budget auto detection."""
        if not isinstance(config, OffloadConfig):
            raise TypeError("config must be an OffloadConfig")
        config.validate_store(store)
        selected = torch.device("cuda") if device is None else torch.device(device)
        if selected.type != "cuda":
            raise ValueError("offload requires CUDA")
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability(selected)
        actual_sm = major * 10 + minor
        if config.target_sm != actual_sm:
            raise ValueError("deployment configuration mismatch: target_sm "
                             f"{config.target_sm} does not match selected device SM{actual_sm}")
        runtime = cls(store, device=selected, **config.runtime_options())
        try:
            runtime.validate_config(config)
            runtime._deployment_contracts = {layer: config.layer_config(layer)
                                             for layer, *_ in config.layer_specs}
            runtime._deployment_manifest = config.manifest()
            runtime.deployment_config = config
        except BaseException:
            runtime.close()
            raise
        return runtime

    @contextmanager
    def _access(self):
        active = getattr(_RUNTIME_ACCESS, "active", ())
        if any(runtime is self for runtime in active):
            raise RuntimeError("reentrant offload runtime access is not supported")
        with self._lock:
            _RUNTIME_ACCESS.active = (*active, self)
            try:
                yield
            finally:
                _RUNTIME_ACCESS.active = active

    def _require_open(self):
        if self._closed:
            raise RuntimeError("offload runtime is closed")

    def _validate_bindings(self):
        self._token_identity.validate(self.token, weight=False)
        if set(self.store.layers) != set(self._bound_layers):
            raise ValueError("offload store layer inventory changed; create a new runtime")
        for layer, pair in self._bound_layers.items():
            current = self.store.layers[layer]
            if not isinstance(current, Sequence) or len(current) != 2:
                raise ValueError("offload store weight bindings changed")
            for identity, held, value in zip(self._weight_identities[layer], pair, current):
                identity.validate(held, weight=True)
                identity.validate(value, weight=True)

    def _validate_config(self, config):
        if not isinstance(config, OffloadConfig):
            raise TypeError("artifact_config must be an OffloadConfig")
        # Preserve canonical/store-shape diagnostics before identity checks.
        config.validate_store(self.store)
        self._token_identity.validate(self.token, weight=False)
        torch.ops.recsys_ops.fuse_moe_offload_validate(self.token, config.manifest())
        # Preserve native all-layer version diagnostics; detect public mapping
        # replacement or shape/storage alias changes before executing any graph.
        self._validate_bindings()

    def validate_config(self, config: OffloadConfig):
        """Read-only graph-external preflight using the ARTIFACT's frozen contract.

        Use invoke() to serialize preflight, execution and failure checking.
        This alone neither validates arbitrary graph bindings nor propagates
        unchecked AOTI errors. Compile artifacts with register_aoti_lowering().
        """
        with self._access():
            self._require_open()
            self._validate_config(config)

    def _prepare_inputs(self, inputs, bindings):
        if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes, bytearray)):
            raise TypeError("inputs must be an ordered sequence")
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes, bytearray)):
            raise TypeError("bindings must be a sequence of InputBinding")
        prepared = list(inputs)
        indices, roles = set(), set()
        for binding in bindings:
            if not isinstance(binding, InputBinding):
                raise TypeError("bindings must contain InputBinding values")
            if binding.index >= len(prepared) or binding.index in indices:
                raise ValueError("binding index out of range or duplicated")
            indices.add(binding.index)
            role = (binding.kind, binding.layer_id)
            if role in roles:
                raise ValueError("duplicate weight/token role binding")
            roles.add(role)
            if binding.kind == "token":
                tensor, identity, weight = self.token, self._token_identity, False
            else:
                if binding.layer_id not in self._bound_layers:
                    raise ValueError("binding layer is not in the runtime")
                slot = 0 if binding.kind == "w1" else 1
                tensor = self._bound_layers[binding.layer_id][slot]
                identity = self._weight_identities[binding.layer_id][slot]
                weight = True
            value = prepared[binding.index]
            if value is not None:
                identity.validate(value, weight=weight)
            prepared[binding.index] = tensor
        required = {("token", None)} | {
            (kind, layer) for layer in self._bound_layers for kind in ("w1", "w2")}
        if roles != required:
            raise ValueError("bindings require one token and every layer's w1/w2 exactly once")
        return tuple(prepared)

    def prepare_inputs(self, inputs, bindings):
        """Fill None placeholders without modifying the caller's input list.

        Explicit Tensor arguments are validated, not replaced silently. Non-bound
        inputs can be arbitrary objects accepted by the caller's official loader.
        """
        with self._access():
            self._require_open()
            self._validate_bindings()
            return self._prepare_inputs(inputs, bindings)

    def invoke(self, callable_, inputs, *, artifact_config: OffloadConfig, bindings):
        """Serialize preflight -> begin -> loader -> check, publishing only success.

        AOTI graphs MUST be built with this repository's checked lowering (see
        register_aoti_lowering). An unchecked old graph can continue after a
        swallowed error before check; this adapter cannot repair its codegen.
        Callable/graph identity, non-offload operators and external side effects
        belong to the caller. No automatic retries or full-graph rollback.
        """
        if not callable(callable_):
            raise TypeError("callable_ must be a callable loader entry")
        with self._access():
            self._require_open()
            self._validate_bindings()
            args = self._prepare_inputs(inputs, bindings)
            self._validate_config(artifact_config)
            torch.ops.recsys_ops.fuse_moe_offload_begin(self.token)
            callback_error, output = None, None
            try:
                try:
                    output = callable_(*args)
                except BaseException as error:
                    callback_error = error
                try:
                    torch.ops.recsys_ops.fuse_moe_offload_check(self.token)
                except BaseException as error:
                    if callback_error is not None:
                        raise error from callback_error
                    raise
                if callback_error is not None:
                    raise callback_error
                return output
            finally:
                # Profilers/traceback observers can retain returned Python frames.
                # Hand ownership only to the caller, never a stale frame-local
                # GPU result; failed partial results must be released as well.
                output = None
                callback_error = None

    def forward(self, x, topk_ids, topk_scale, *, layer_id=0):
        """Eager call; tracked failures remain pending until acknowledge()."""
        with self._access():
            self._require_open()
            w1, w2 = self._bound_layers[layer_id]
            return torch.ops.recsys_ops.fuse_moe_offload(
                x, w1, w2, topk_ids, topk_scale, self.token, layer_id,
                self._deployment_contracts.get(layer_id, ()), self._deployment_manifest)

    def stats(self):
        """Last successful forward statistics; not a success/failure signal."""
        with self._access():
            self._require_open()
            values = torch.ops.recsys_ops.fuse_moe_offload_stats(self.token).tolist()
            if len(values) != len(_STATS):
                raise RuntimeError(f"offload stats schema mismatch: expected {len(_STATS)} values, got {len(values)}")
            return dict(zip(_STATS, values))

    def status(self):
        """Versioned native status; first unacknowledged error survives later calls."""
        with self._access():
            self._require_open()
            raw = torch.ops.recsys_ops.fuse_moe_offload_status(self.token)
            if (raw.device.type != "cpu" or raw.dtype != torch.int64 or
                    tuple(raw.shape) != (len(_STATUS),)):
                raise RuntimeError("offload status schema mismatch")
            values = raw.tolist()
            if values[0] != 1 or values[1] not in (0, 1, 2):
                raise RuntimeError("unsupported offload status version/state")
            values[1] = ("healthy", "recoverable", "poisoned")[values[1]]
            return dict(zip(_STATUS, values))

    def check(self):
        """Observe pending failure without acknowledging it; use invoke for AOTI."""
        with self._access():
            self._require_open()
            torch.ops.recsys_ops.fuse_moe_offload_check(self.token)

    def acknowledge(self):
        """Explicitly acknowledge only a recoverable native failure, never retry."""
        with self._access():
            self._require_open()
            torch.ops.recsys_ops.fuse_moe_offload_acknowledge(self.token)

    def close(self):
        with self._access():
            if not self._closed:
                torch.ops.recsys_ops.fuse_moe_offload_close(self.token)
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception:
                pass


def compile_aoti(exported_program, *, package_path=None, inductor_configs=None):
    """Official offline compilation with checked failures and cache-safe defaults."""
    from custom_ops.moe_offload_aoti import compile_aoti as compile_package
    return compile_package(exported_program, package_path=package_path,
                           inductor_configs=inductor_configs)


def register_aoti_lowering():
    """Install the version-checked, fail-fast lowering BEFORE AOTI compilation.

    Import/build dependencies are compile-time only. Existing artifacts must be
    rebuilt; runtime execution remains entirely native with the official loader.
    """
    from custom_ops.moe_offload_aoti import register_aoti_lowering as register
    register()


class OffloadedMoELayer(torch.nn.Module):
    """Export adapter: CPU weights and deployment token stay external graph inputs."""

    def __init__(self, layer_id=0, *, config: OffloadConfig | None = None):
        super().__init__()
        self.layer_id = layer_id
        if config is not None and not isinstance(config, OffloadConfig):
            raise TypeError("config must be an OffloadConfig")
        # A Python tuple becomes a constant int[] in export/AOTI, not a graph input.
        self.deployment_config = () if config is None else config.layer_config(layer_id)
        self.deployment_manifest = () if config is None else config.manifest()

    def forward(self, x, w1, w2, topk_ids, topk_scale, runtime_token):
        return torch.ops.recsys_ops.fuse_moe_offload(
            x, w1, w2, topk_ids, topk_scale, runtime_token, self.layer_id,
            self.deployment_config, self.deployment_manifest)
