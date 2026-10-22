// Standalone AOTI RAM-offload acceptance. No Python headers or interpreter calls.
#include <ATen/ATen.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>
#include <torch/serialize.h>
#include "fuse_moe/fuse_moe_offload_runtime.h"
#include <sys/resource.h>
#include <cuda_runtime_api.h>
#include <climits>
#include <limits>
#include <chrono>
#include <map>
#include <memory>
#include <cmath>
#include <sched.h>
#include <future>
#include <iterator>
#include <mutex>
#include <type_traits>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <dlfcn.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t E = 4, H = 64, I = 64, K = 2;
constexpr int64_t expert_bytes = 6 * H * I;
constexpr int64_t gpu_budget = 128 * 1024;
constexpr int64_t cached_gpu_budget = 256 * 1024;
constexpr int64_t cache_cap = 128 * 1024;
constexpr int64_t dynamic_cap = ((expert_bytes + 511) / 512) * 512;
constexpr int64_t stats_fields = 31;
constexpr int64_t pinned_budget = 4 * (E + 1) + 2 * (expert_bytes + 24);
// Same total cap in sync/async: CPU weights, pinned/planning buffers,
// and the bounded worker's 1 MiB stack plus fixed bookkeeping allowance.
constexpr int64_t ram_budget = 4 * 1024 * 1024;
// Chosen once by CLI before any Runtime is constructed; never read from env.
bool config_async_staging = false;
bool config_host_profile = false;
// Wide-benchmark storage/activation; locked paths derive these from the archive.
std::string config_precision = "fp16";
int64_t config_expert_activation = 0;
int64_t config_numerical_mode = 0;
int64_t config_expert_type = 0;
bool precision_explicit = false, expert_activation_explicit = false, numerical_mode_explicit = false;
bool async_staging_explicit = false, host_profile_explicit = false;
std::filesystem::path output_directory;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}
long peak_rss_kib() {
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  return usage.ru_maxrss;
}

struct Weights {
  std::map<int64_t, std::pair<at::Tensor, at::Tensor>> layers;
  int64_t version;
  std::map<int64_t, std::vector<int64_t>> configs;
  explicit Weights(const std::string& path, bool wide = false, bool locked = false,
                   bool public_mode = false) {
    const auto started = Clock::now();
    // Official CPU archive reader only: do not execute the module's forward/JIT.
    auto archive = torch::jit::load(path, c10::Device(c10::kCPU));
    std::map<std::string, at::Tensor> buffers;
    for (const auto& buffer : archive.named_buffers()) {
      buffers.emplace(buffer.name, buffer.value);
    }
    require(archive.named_parameters().size() == 0, "Archive must not contain parameters");
    auto metadata = [&](const std::string& name) {
      require(buffers.count(name) == 1, "Missing archive metadata: " + name);
      auto value = buffers.at(name);
      require(value.device().is_cpu() && value.scalar_type() == at::kLong && value.dim() == 1,
              "Invalid archive metadata: " + name);
      return value;
    };
    auto format = metadata("format_version");
    require(format.numel() == 1 && format[0].item<int64_t>() == (locked ? 2 : 1),
            locked ? "LOCKED archive requires format2" : "Unsupported archive format_version");
    auto weight_version = metadata("weight_version");
    require(weight_version.numel() == 1 && weight_version[0].item<int64_t>() >= 0,
            "Invalid archive weight_version");
    version = weight_version[0].item<int64_t>();
    auto ids = metadata("layer_ids");
    require(ids.numel() > 0, "Empty archive layer_ids");
    std::set<std::string> expected{"format_version", "weight_version", "layer_ids"};
    int64_t previous = -1;
    for (int64_t j = 0; j < ids.numel(); ++j) {
      const int64_t layer = ids[j].item<int64_t>();
      require(layer > previous, "Archive layer_ids must be sorted, unique and nonnegative");
      previous = layer;
      const auto n1 = "w1_" + std::to_string(layer), n2 = "w2_" + std::to_string(layer);
      expected.insert(n1);
      expected.insert(n2);
      require(buffers.count(n1) && buffers.count(n2), "Archive buffers do not match layer_ids");
      const auto w1 = buffers.at(n1), w2 = buffers.at(n2);
      for (const auto& weight : {w1, w2}) {
        require(weight.device().is_cpu() && weight.is_contiguous() && !weight.is_pinned() &&
                    !weight.requires_grad() && !weight.is_inference() && weight.dim() == 3 &&
                    (weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kBFloat16),
                "Archive weight is not canonical pageable CPU fp16/bf16");
        (void)weight._version();
      }
      int64_t projection_factor = 2;
      if (locked) {
        const auto name = "offload_config_" + std::to_string(layer);
        expected.insert(name);
        auto vector = metadata(name).contiguous();
        configs.emplace(layer, std::vector<int64_t>(vector.const_data_ptr<int64_t>(),
                                                    vector.const_data_ptr<int64_t>() + vector.numel()));
        const auto& v = configs.at(layer);
        const size_t header = !v.empty() && v[0] == 1 ? 23 :
                              !v.empty() && v[0] == 2 ? 25 :
                              !v.empty() && v[0] == 3 ? 26 : 0;
        require(header > 0 && v.size() >= header && v[21] >= 0 && v[21] <= 512 &&
                    v.size() == header + static_cast<size_t>(v[21]),
                "LOCKED archive vector length/version");
        if (v[0] >= 2) {
          require((v[23] == 0 || v[23] == 1) && (v[24] == 0 || v[24] == 1),
                  "LOCKED archive invalid expert activation/numerical mode");
        }
        if (v[0] == 3) {
          require(v[25] == 1, "LOCKED archive invalid expert_type");
          projection_factor = 1;
        }
      }
      require(w1.scalar_type() == w2.scalar_type() && w1.size(0) > 0 && w1.size(0) <= 512 &&
                  w2.size(0) == w1.size(0) && w2.size(1) == w1.size(2) &&
                  w1.size(1) == projection_factor * w2.size(2) && w1.size(2) > 0 && w2.size(2) > 0 &&
                  w1.size(2) % 64 == 0 && w2.size(2) % 64 == 0,
              "Invalid canonical archive E/H/I");
      layers.emplace(layer, std::make_pair(w1, w2));
    }
    require(buffers.size() == expected.size(), "Unexpected archive buffers");
    for (const auto& buffer : buffers) {
      require(expected.count(buffer.first), "Unexpected archive buffer: " + buffer.first);
    }
    if (!public_mode) {
    require(locked ? (layers.count(0) && (layers.size() == 1 || (layers.size() == 2 && layers.count(5))))
                   : (wide ? (layers.size() == 1 && layers.count(0))
                           : (layers.size() == 2 && layers.count(0) && layers.count(5))),
            "Acceptance requires baseline layers 0/5 or wide layer 0");
    const int64_t expected_h = wide ? 128 : H;
    for (const auto& layer : layers) {
      const int64_t projection_factor = locked && configs.at(layer.first)[0] == 3 ? 1 : 2;
      require(layer.second.first.sizes() == at::IntArrayRef({E, projection_factor * I, expected_h}) &&
                  layer.second.second.sizes() == at::IntArrayRef({E, expected_h, I}) &&
                  (layer.second.first.scalar_type() == at::kHalf ||
                   ((locked || wide) && layer.second.first.scalar_type() == at::kBFloat16)),
              "Acceptance archive shape/dtype mismatch");
    }
    }
    std::cout << "ACCEPTANCE_LOAD {\"weight_version\":" << version
              << ",\"archive_load_ms\":" << elapsed_ms(started)
              << ",\"process_peak_rss_kib\":" << peak_rss_kib()
              << ",\"rss_note\":\"Linux lifetime high-water; not exact budget\"}" << std::endl;
  }
};

// Dispatcher boxed calls use precisely the installed schemas, including mutation.
std::vector<c10::IValue> call(const char* name, std::vector<c10::IValue> args) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
  op.callBoxed(&args);
  return args;
}

struct Runtime {
  at::Tensor token;
  bool closed = false;
  int64_t mode;
  std::vector<int64_t> resident_ids;
  int64_t dynamic_bytes, policy;
  int64_t budget, actual_cache_bytes = 0, resident_load_bytes = 0;
  int64_t backing_bytes;
  int64_t weight_bytes, pinned_limit, resident_limit, tile_limit;
  Runtime(int64_t device, const at::Tensor& w1, const at::Tensor& w2,
          int64_t activation_mode, const std::vector<int64_t>& expert_ids,
          int64_t dynamic_cache_bytes = 0, int64_t cache_policy = 0,
          const at::Tensor& w15 = {}, const at::Tensor& w25 = {},
          bool wide = false, int64_t wide_tile_rows = 3, int64_t benchmark_budget = 0,
          int64_t native_tile_override = -1,
          bool async_staging = config_async_staging, bool host_profile = config_host_profile,
          int64_t expert_activation = 0, int64_t numerical_mode = 0)
      : mode(activation_mode), resident_ids(expert_ids),
        dynamic_bytes(dynamic_cache_bytes), policy(cache_policy),
        budget(expert_ids.empty() && !dynamic_bytes ? gpu_budget : cached_gpu_budget) {
    const auto started = Clock::now();
    weight_bytes = wide ? 6 * w1.size(2) * w2.size(2) : expert_bytes;
    pinned_limit = wide ? 4 * (E + 1) + 2 * (2 * weight_bytes + 40) : pinned_budget;
    resident_limit = wide ? E * weight_bytes : cache_cap;
    tile_limit = wide ? wide_tile_rows : (mode == 1 ? 3 : 4096);
    if (native_tile_override >= 0) tile_limit = native_tile_override;
    if (wide) budget = 2 * 1024 * 1024;
    if (benchmark_budget) budget = benchmark_budget;
    backing_bytes = E * weight_bytes * (w15.defined() ? 2 : 1);
    token = call("recsys_ops::fuse_moe_offload_create",
                 {device, budget, ram_budget, pinned_limit, int64_t(2), mode,
                  tile_limit,
                  int64_t(resident_ids.empty() ? 0 : resident_limit), dynamic_bytes,
                  policy, int64_t(32), int64_t(2), int64_t(0),
                  async_staging, host_profile, int64_t(0), int64_t(128),
                  expert_activation, numerical_mode, int64_t(0)})[0].toTensor();
    require(token.device().is_cpu() && token.scalar_type() == at::kLong &&
                token.sizes() == at::IntArrayRef({2}), "Invalid deployment token");
    try {
      call("recsys_ops::fuse_moe_offload_bind", {token, int64_t(0), w1, w2});
      if (w15.defined()) {
        call("recsys_ops::fuse_moe_offload_bind", {token, int64_t(5), w15, w25});
      }
      if (!resident_ids.empty()) {
        // Construct a boxed int[] IValue, not a tensor, before any inference.
        call("recsys_ops::fuse_moe_offload_resident_bind",
             {token, int64_t(0), c10::IValue(std::vector<int64_t>(resident_ids))});
      }
      auto stats = call("recsys_ops::fuse_moe_offload_stats", {token})[0].toTensor();
      require(stats.device().is_cpu() && stats.scalar_type() == at::kLong &&
                  stats.numel() == stats_fields, "Initialization stats must contain 31 fields");
      const auto* st = stats.const_data_ptr<int64_t>();
      for (int j = 22; j < 30; ++j) {
        require(st[j] == 0, "Dynamic initialization fields must be zero");
      }
      require(dynamic_bytes ? st[30] > 0 : st[30] >= 0,
              "Invalid initialization cache metadata charge");
      const int64_t count = static_cast<int64_t>(resident_ids.size());
      actual_cache_bytes = st[19];
      resident_load_bytes = st[21];
      require(st[20] == count && resident_load_bytes == count * weight_bytes,
              "Resident initialization count/H2D mismatch");
      require(count ? (actual_cache_bytes >= count * weight_bytes &&
                       actual_cache_bytes <= resident_limit && actual_cache_bytes % 512 == 0)
                    : actual_cache_bytes == 0,
              "Resident initialization aligned charge mismatch");
      c10::cuda::getCurrentCUDAStream(device).synchronize();
      std::cout << "ACCEPTANCE_STARTUP {\"bind_startup_ms\":" << elapsed_ms(started)
                << ",\"process_peak_rss_kib\":" << peak_rss_kib()
                << ",\"rss_note\":\"Linux lifetime high-water; not exact budget\"}" << std::endl;
    } catch (...) {
      close();
      throw;
    }
  }
  void close() {
    if (!closed) {
      call("recsys_ops::fuse_moe_offload_close", {token});
      closed = true;
    }
  }
  ~Runtime() {
    if (!closed) {
      try {
        close();
      } catch (const std::exception& error) {
        std::cerr << "Runtime cleanup failed: " << error.what() << '\n';
      }
    }
  }
};

std::vector<at::Tensor> make_inputs(int64_t s, int64_t device,
                                   const at::Tensor& w1, const at::Tensor& w2,
                                   const at::Tensor& token) {
  auto fp32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto cuda = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device));
  auto x = (at::arange(s * H, fp32).cos() * .25).reshape({s, H}).to(cuda, w1.scalar_type());
  auto ids = at::arange(s * K, at::TensorOptions().dtype(at::kLong))
                 .remainder(E).reshape({s, K}).to(cuda, at::kInt);
  auto scale = at::full({s, K}, 1.0 / K, fp32).to(cuda);
  return {x, w1, w2, ids, scale, token};
}

at::Tensor reference(const std::vector<at::Tensor>& inputs, bool rounded = true,
                     int64_t expert_activation = 0, int64_t numerical_mode = 0, int64_t expert_type = 0) {
  require(expert_type == 0 || expert_type == 1, "Invalid expert_type");
  require(expert_activation == 0 || expert_activation == 1, "Invalid expert_activation");
  require(numerical_mode == 0 || numerical_mode == 1, "Invalid numerical_mode");
  require(inputs[1].size(1) == (expert_type == 0 ? 2 : 1) * inputs[2].size(2), "expert_type shape mismatch");
  auto x = inputs[0].to(at::kCPU).to(at::kFloat);
  auto w1 = inputs[1].to(at::kFloat);
  auto w2 = inputs[2].to(at::kFloat);
  auto ids = inputs[3].to(at::kCPU).to(at::kLong);
  auto scale = inputs[4].to(at::kCPU);
  auto out = at::zeros_like(x);
  const auto dtype = inputs[0].scalar_type();
  const auto round = [&](const at::Tensor& value) {
    return value.to(dtype).to(at::kFloat);
  };
  const bool staged = numerical_mode == 1;
  // Independent expert-by-expert ATen FP32 computation, not the fused GPU op.
  for (int64_t expert = 0; expert < w1.size(0); ++expert) {
    auto positions = at::nonzero(ids.eq(expert));
    auto rows = positions.select(1, 0);
    auto slots = positions.select(1, 1);
    auto projected = at::matmul(x.index_select(0, rows), w1[expert].transpose(0, 1));
    const int64_t intermediate = w2.size(2);
    auto gate = projected.narrow(1, 0, intermediate);
    at::Tensor up;
    if (expert_type == 0) up = projected.narrow(1, intermediate, intermediate);
    if (staged) {
      gate = round(gate);
      if (expert_type == 0) up = round(up);
    }
    // Exact erf GELU (never tanh) or the historical SiLU default.
    auto act = expert_activation == 1
        ? 0.5 * gate * (1.0 + at::erf(gate * 0.7071067811865476))
        : at::silu(gate);
    if (expert_type == 0) {
      if (staged) act = round(act);
      act = act * up;
      if (staged) act = round(act);
    }
    // Native fused gate/up retains FP32 accumulators, then stores Act in the
    // input dtype. Down is stored in that dtype before scale/FP32 reduction.
    if (rounded) act = round(act);
    auto values = at::matmul(act, w2[expert].transpose(0, 1));
    if (staged) values = round(values);
    if (rounded) values = round(values);
    auto route_scales = scale.reshape({-1}).index_select(0, rows * ids.size(1) + slots).unsqueeze(1);
    auto weighted = values * route_scales;
    if (staged) weighted = round(weighted);
    out.index_add_(0, rows, weighted);
  }
  return out;
}

void check(const std::vector<at::Tensor>& inputs, const at::Tensor& output,
           Runtime& runtime, int64_t before, int64_t id, const void* token_ptr,
           int64_t version,
           const std::string& label, bool check_reuse,
           const std::vector<int64_t>& dynamic_expected = {},
           const at::Tensor& chain_expected = {}, int64_t invocations = 1,
           bool wide = false) {
  c10::cuda::getCurrentCUDAStream(inputs[0].get_device()).synchronize();
  require(runtime.token.const_data_ptr() == token_ptr &&
              runtime.token[0].item<int64_t>() == id, label + ": token identity changed");
  require(runtime.token[1].item<int64_t>() == before + invocations,
          label + ": missing/doubled actual token mutation");
  require(runtime.token._version() == version + invocations,
          label + ": missing/doubled token version bump");
  require(output.device() == inputs[0].device() && output.scalar_type() == at::kHalf &&
              output.sizes() == inputs[0].sizes(), label + ": invalid output metadata");
  auto expected = chain_expected.defined() ? chain_expected : reference(inputs);
  auto actual = output.to(at::kCPU).to(at::kFloat);
  auto error = (actual - expected).abs();
  require(at::isfinite(actual).all().item<bool>() &&
              error.le(expected.abs() * .02 + (invocations == 1 ? .0002 : .000002)).all().item<bool>(),
          label + ": FP32 reference mismatch, max abs=" +
              std::to_string(error.max().item<float>()));
  if (invocations == 2) {
    require(expected.abs().max().item<float>() > .0001, label + ": real chain fixture underflowed");
    const float relative_l2 = error.norm().item<float>() / expected.norm().item<float>();
    require(relative_l2 < .01, label + ": chain relative L2 mismatch=" + std::to_string(relative_l2));
  }
  if (wide) {
    require(inputs[0].size(1) == 128 && inputs[2].size(2) == 64 && runtime.mode == 1,
            label + ": wide fused GEMM2 requires H128/I64 accumulate");
    require(expected.abs().max().item<float>() > .001, label + ": zero-output fixture");
    for (int64_t start : {int64_t(0), int64_t(64)}) {
      auto tile = expected.narrow(1, start, 64);
      require(tile.norm().item<float>() > .001 &&
                  error.narrow(1, start, 64).norm().item<float>() / tile.norm().item<float>() < .02,
              label + ": wide N tile reference mismatch at " + std::to_string(start));
    }
    auto masked = inputs[3].to(at::kCPU).lt(0).all(1);
    require(masked.any().item<bool>() && actual.index({masked}).eq(0).all().item<bool>(),
            label + ": fully masked rows must be exactly zero");
  }
  auto stats = call("recsys_ops::fuse_moe_offload_stats", {runtime.token})[0].toTensor();
  require(stats.device().is_cpu() && stats.scalar_type() == at::kLong &&
              stats.numel() == stats_fields, label + ": stats must contain all 31 fields");
  const auto* st = stats.const_data_ptr<int64_t>();
  const bool dynamic = runtime.dynamic_bytes > 0;
  require(dynamic == !dynamic_expected.empty(), label + ": missing dynamic expectations");
  const int64_t active = dynamic ? dynamic_expected.at(0) : E;
  const int64_t hits = dynamic ? dynamic_expected.at(1)
                             : static_cast<int64_t>(runtime.resident_ids.size());
  const int64_t misses = active - hits;
  // Baseline misses remain single-expert; wide banks hold two experts.
  const int64_t miss_capacity = wide ? std::min<int64_t>(misses, 2) : 1;
  int64_t miss_waves = misses;
  int64_t expected_waves = active;
  if (dynamic && dynamic_expected.size() == 7) expected_waves = dynamic_expected.at(6);
  if (!dynamic) {
    expected_waves = 0;
    miss_waves = 0;
    int64_t miss_run = 0;
    bool previous_hit = false;
    for (int64_t expert = 0; expert < E; ++expert) {
      const bool hit = std::find(runtime.resident_ids.begin(), runtime.resident_ids.end(), expert)
          != runtime.resident_ids.end();
      expected_waves += hit && !previous_hit;
      if (!hit) ++miss_run;
      if (hit || expert == E - 1) {
        if (miss_run) miss_waves += (miss_run + miss_capacity - 1) / miss_capacity;
        miss_run = 0;
      }
      previous_hit = hit;
    }
    expected_waves += miss_waves;
  }
  require(st[0] == active && st[1] == expected_waves && st[2] == misses * runtime.weight_bytes &&
              st[3] == (4 * misses + 2 * miss_waves) * 4,
          label + ": wave/miss transfer stats mismatch");
  require(st[16] == hits && st[17] == misses && st[18] == hits * runtime.weight_bytes &&
              st[19] == runtime.actual_cache_bytes &&
              st[20] == static_cast<int64_t>(runtime.resident_ids.size()) &&
              st[21] == runtime.resident_load_bytes, label + ": resident cache stats mismatch");
  if (dynamic) {
    const int64_t admissions = dynamic_expected.at(2);
    const int64_t evictions = dynamic_expected.at(3);
    const int64_t bypasses = dynamic_expected.at(4);
    const int64_t cached = dynamic_expected.at(5);
    require(st[22] == cached * dynamic_cap && st[23] == cached && st[24] == hits &&
                st[25] == admissions && st[26] == evictions && st[27] == bypasses &&
                st[28] == admissions * expert_bytes && st[29] == 0 && st[30] > 0,
            label + ": dynamic charge/hit/admission/eviction/D2D mismatch");
    // cache_update_bytes=0 caps total promotion traffic per call at dynamic capacity.
    require(st[28] >= 0 && st[28] <= runtime.dynamic_bytes,
            label + ": promotion exceeded the per-call dynamic update cap");
    require(st[19] == 0 && st[20] == 0 && st[21] == 0,
            label + ": dynamic cache contaminated static lifetime stats");
  } else {
    for (int j = 22; j < 30; ++j) {
      require(st[j] == 0, label + ": static baseline has dynamic activity");
    }
    require(st[30] >= 0, label + ": negative cache metadata bytes");
  }
  require(st[4] > 0 && st[4] <= runtime.budget &&
              st[5] > 0 && st[5] <= runtime.pinned_limit &&
              st[6] >= runtime.backing_bytes && st[6] <= ram_budget &&
              st[7] == (misses ? 2 : 0) && st[8] == (misses ? miss_capacity : 0),
          label + ": budget/bank stats mismatch");
  if (!wide && !dynamic && !hits) {
    require(st[1] > 1 && st[2] == E * expert_bytes && st[4] <= gpu_budget &&
                st[7] == 2 && st[8] == 1, label + ": disabled baseline regressed");
  }
  require(st[9] == runtime.mode && st[10] >= 0 && st[11] >= 0,
          label + ": activation mode/tile stats mismatch");
  if (runtime.mode == 1) {
    require(st[10] > 0 && st[11] > 0 &&
                (runtime.tile_limit == 0 || st[11] <= runtime.tile_limit) &&
                (wide || runtime.tile_limit != 3 || inputs[0].size(0) < 7 || st[10] > 1),
            label + ": accumulate route tile limit mismatch");
    if (wide) {
      auto host_ids = inputs[3].to(at::kCPU);
      int64_t wave_rows = 0, group_count = 0, max_wave = 0, expected_tiles = 0;
      bool previous_hit = false;
      auto add_wave = [&]() {
        max_wave = std::max(max_wave, wave_rows);
        expected_tiles += (wave_rows + st[11] - 1) / st[11];
      };
      for (int64_t expert = 0; expert < E; ++expert) {
        const bool hit = std::find(runtime.resident_ids.begin(), runtime.resident_ids.end(), expert)
                != runtime.resident_ids.end();
        const int64_t capacity = hit ? E : miss_capacity;
        if (expert && (hit != previous_hit || group_count == capacity)) {
          add_wave(); wave_rows = 0; group_count = 0;
        }
        wave_rows += host_ids.eq(expert).sum().item<int64_t>();
        ++group_count;
        previous_hit = hit;
      }
      add_wave();
      require(st[10] == expected_tiles, label + ": clipped wave tile count mismatch");
      if (runtime.tile_limit == 4096 || runtime.tile_limit == 0) {
        require(st[11] >= max_wave && st[10] == st[1], label + ": full-wave path not exercised");
        if (runtime.tile_limit == 0)
          require(st[11] < max_wave + 128, label + ": auto rows not minimized/aligned");
      } else if (max_wave > st[11]) {
        require(st[10] > st[1], label + ": small clipped tiles not exercised");
      } else {
        require(st[10] == st[1], label + ": full small waves must not be split");
      }
    }
  }
  require(st[12] >= 0 && st[13] >= 0 && st[14] > 0 &&
              st[14] + runtime.actual_cache_bytes + st[22] <= st[4] &&
              st[15] > 0 && st[15] <= st[5], label + ": pool stats mismatch");
  if (check_reuse) {
    require(st[12] == 0 && st[13] > 0,
            label + ": identical-shape second call must reuse internal tensors");
  }
  require(inputs[1].device().is_cpu() && inputs[2].device().is_cpu() &&
              !inputs[1].is_pinned() && !inputs[2].is_pinned(),
          label + ": canonical weights must remain pageable RAM");
  std::cout << "PASS " << label << ": token=" << id << ',' << before + invocations
            << " version=" << version << "->" << runtime.token._version()
            << " max_abs=" << error.max().item<float>() << " stats=";
  for (int j = 0; j < stats_fields; ++j) {
    std::cout << (j ? "," : "") << st[j];
  }
  std::cout << std::endl;
}

void check_dynamic_contexts(torch::inductor::AOTIModelPackageLoader& runner,
                            int64_t device, const at::Tensor& w1, const at::Tensor& w2,
                            const at::Tensor& w15, const at::Tensor& w25) {
  for (const auto& configuration : std::vector<std::pair<int64_t, int64_t>>{
           {0, 4096}, {1, 3}, {1, 0}}) {
    const auto mode = configuration.first, tile_rows = configuration.second;
    for (int64_t policy : {int64_t(0), int64_t(1)}) {
      // Separate cold contexts for dispatcher and AOTI, but one model loader.
      for (bool compiled : {false, true}) {
        Runtime runtime(device, w1, w2, mode, {}, dynamic_cap, policy, w15, w25,
                        false, 3, 0, tile_rows);
        auto inputs = make_inputs(7, device, w1, w2, runtime.token);
        auto hot = inputs;
        hot[3] = at::zeros_like(inputs[3]);
        const std::string label = std::string(compiled ? "C++ AOTI" : "C++ dispatcher") +
            " dynamic=" + (policy ? "lru" : "reuse") + " mode=" + std::to_string(mode) +
            " tile=" + std::to_string(tile_rows);
        auto invoke = [&](const std::vector<at::Tensor>& route, const std::string& stage,
                          const std::vector<int64_t>& expected) {
          const int64_t before = runtime.token[1].item<int64_t>();
          const int64_t id = runtime.token[0].item<int64_t>();
          const void* ptr = runtime.token.const_data_ptr();
          const int64_t version = runtime.token._version();
          std::vector<at::Tensor> outputs;
          if (compiled) {
            outputs = runner.run(route);
          } else {
            outputs = {call("recsys_ops::fuse_moe_offload",
                            {route[0], w1, w2, route[3], route[4], runtime.token, int64_t(0), c10::IValue(std::vector<int64_t>{}), c10::IValue(std::vector<int64_t>{})})[0]
                           .toTensor()};
          }
          require(outputs.size() == 1, label + ": expected one output");
          check(route, outputs[0], runtime, before, id, ptr, version,
                label + " " + stage, false, expected);
          // check() drains; outputs die here before the next serialized call.
        };
        // Duplicate slots/tokens must not count as multiple reuse observations.
        invoke(hot, "first-cold", {1, 0, policy, 0, 1 - policy, policy});
        invoke(hot, "second-admission", {1, policy, 1 - policy, 0, 0, 1});
        invoke(hot, "third-hit-zero-H2D", {1, 1, 0, 0, 0, 1});
        for (int warm = 0; warm < 5; ++warm) {
          invoke(hot, "warm-hot-" + std::to_string(warm), {1, 1, 0, 0, 0, 1});
        }
        // LRU promotes one cold key, then bypasses two when the update cap is spent.
        invoke(inputs, "distinct-scan", {E, 1, policy, policy, 3 - policy, 1});
        invoke(hot, "post-scan-hot", {1, 1 - policy, policy, policy, 0, 1});
        for (int64_t s : {int64_t(2), int64_t(64), int64_t(7)}) {
          auto resized = make_inputs(s, device, w1, w2, runtime.token);
          resized[3] = at::zeros_like(resized[3]);
          invoke(resized, "dynamic-S=" + std::to_string(s) + "-hot-hit", {1, 1, 0, 0, 0, 1});
        }
        runtime.close();
        Runtime grouped(device, w1, w2, mode, {}, 2 * dynamic_cap, policy, w15, w25,
                        false, 3, 0, tile_rows);
        for (int stage = 0; stage < 3; ++stage) {
          auto route = make_inputs(7, device, w1, w2, grouped.token);
          route[3] = route[3].remainder(2);
          const auto before = grouped.token[1].item<int64_t>();
          const auto id = grouped.token[0].item<int64_t>();
          const auto ptr = grouped.token.const_data_ptr();
          const auto version = grouped.token._version();
          std::vector<at::Tensor> outputs;
          if (compiled) outputs = runner.run(route);
          else outputs = {call("recsys_ops::fuse_moe_offload",
              {route[0], w1, w2, route[3], route[4], grouped.token, int64_t(0), c10::IValue(std::vector<int64_t>{}), c10::IValue(std::vector<int64_t>{})})[0].toTensor()};
          const int64_t hits = stage >= (policy ? 1 : 2) ? 2 : 0;
          const int64_t admissions = stage == (policy ? 0 : 1) ? 2 : 0;
          const int64_t cached = stage >= (policy ? 0 : 1) ? 2 : 0;
          check(route, outputs.at(0), grouped, before, id, ptr, version,
              label + " grouped-hot stage=" + std::to_string(stage), false,
              {2, hits, admissions, 0, stage == 0 && !policy ? 2 : 0, cached, hits ? 1 : 2});
        }
        grouped.close();
      }
    }
  }
}

void check_chain(torch::inductor::AOTIModelPackageLoader& runner, int64_t device,
                 const Weights& weights, int argc, char** argv) {
  const auto w1 = weights.layers.at(0).first, w2 = weights.layers.at(0).second;
  const auto w15 = weights.layers.at(5).first, w25 = weights.layers.at(5).second;
  for (int64_t mode : {int64_t(0), int64_t(1)}) {
    Runtime runtime(device, w1, w2, mode, {}, 0, 0, w15, w25);
    for (int arg = 6; arg <= argc; ++arg) {
      // Return to the initial shape after dynamic sizes using the same loader/token.
      const int64_t s = std::stoll(argv[arg == argc ? 6 : arg]);
      auto inputs = make_inputs(s, device, w1, w2, runtime.token);
      auto second = inputs;
      second[0] = reference(inputs).to(at::kHalf);  // Explicit FP16 inter-layer boundary.
      second[1] = w15;
      second[2] = w25;
      auto expected = reference(second);
      std::vector<at::Tensor> chain_inputs{inputs[0], w1, w2, w15, w25,
                                           inputs[3], inputs[4], runtime.token};
      for (bool compiled : {false, true}) {
        for (int repeat = 0; repeat < 2; ++repeat) {
          const int64_t before = runtime.token[1].item<int64_t>();
          const int64_t id = runtime.token[0].item<int64_t>();
          const void* ptr = runtime.token.const_data_ptr();
          const int64_t version = runtime.token._version();
          std::vector<at::Tensor> outputs;
          if (compiled) {
            outputs = runner.run(chain_inputs);
          } else {
            auto intermediate = call("recsys_ops::fuse_moe_offload",
                {inputs[0], w1, w2, inputs[3], inputs[4], runtime.token, int64_t(0), c10::IValue(std::vector<int64_t>{}), c10::IValue(std::vector<int64_t>{})})[0].toTensor();
            outputs = {call("recsys_ops::fuse_moe_offload",
                {intermediate, w15, w25, inputs[3], inputs[4], runtime.token, int64_t(5), c10::IValue(std::vector<int64_t>{}), c10::IValue(std::vector<int64_t>{})})[0].toTensor()};
          }
          require(outputs.size() == 1, "Real chain expected one flattened output");
          check(inputs, outputs[0], runtime, before, id, ptr, version,
                std::string(compiled ? "C++ AOTI" : "C++ dispatcher") +
                    " real-layer0->layer5 chain v=" + std::to_string(weights.version) +
                    " mode=" + std::to_string(mode) + " S=" + std::to_string(s) +
                    " repeat=" + std::to_string(repeat), false, {}, expected, 2);
        }
      }
    }
    runtime.close();
  }
}

std::vector<at::Tensor> make_wide_inputs(int64_t s, int64_t device,
                                       const at::Tensor& w1, const at::Tensor& w2,
                                       const at::Tensor& token) {
  auto fp32 = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto cuda = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device));
  auto x = (at::arange(s * w1.size(2), fp32).cos() * .25)
               .reshape({s, w1.size(2)}).to(cuda, w1.scalar_type());
  auto pattern = at::tensor({0, 0, 1, -1, 2, 3, -1, -1, 3, 1, 2, 2, 0, 3},
                            at::TensorOptions().dtype(at::kInt)).reshape({7, K});
  auto scales = at::tensor({.75f, -.25f, -.5f, 1.25f, .125f, -.875f, .625f, -1.5f,
                            -.375f, .5f, 1.125f, -.625f, -.75f, .25f}, fp32).reshape({7, K});
  auto rows = at::arange(s, at::TensorOptions().dtype(at::kLong)).remainder(7);
  auto ids = pattern.index_select(0, rows);
  auto scale = scales.index_select(0, rows);
  require(ids.lt(0).any().item<bool>() &&
              ids.select(1, 0).eq(ids.select(1, 1)).any().item<bool>() &&
              scale.lt(0).any().item<bool>() && scale.max().item<float>() > 1,
          "Wide fixture requires masked/repeated routes and signed nonuniform scales");
  std::set<int64_t> counts;
  for (int64_t expert = 0; expert < E; ++expert) {
    const auto count = ids.eq(expert).sum().item<int64_t>();
    require(count > 0, "Wide fixture must activate every expert");
    counts.insert(count);
  }
  require(counts.size() > 1, "Wide expert intervals must be nonuniform");
  return {x, w1, w2, ids.to(cuda), scale.to(cuda), token};
}

void check_wide(const std::string& package, const std::string& archive, int64_t device) {
  Weights weights(archive, true);
  const auto w1 = weights.layers.at(0).first, w2 = weights.layers.at(0).second;
  // One official loader for every tile size, resident layout, S and runtime token.
  torch::inductor::AOTIModelPackageLoader runner(package, "model");
  for (int64_t tile_rows : {int64_t(3), int64_t(4096), int64_t(0)}) {
    for (const auto& residents : std::vector<std::vector<int64_t>>{{}, {0, 2}, {0, 1}, {0, 1, 2, 3}}) {
      Runtime runtime(device, w1, w2, 1, residents, 0, 0, {}, {}, true, tile_rows);
      for (int64_t s : {int64_t(7), int64_t(64), int64_t(7)}) {
        auto inputs = make_wide_inputs(s, device, w1, w2, runtime.token);
        for (bool compiled : {false, true}) {
          for (int repeat = 0; repeat < 2; ++repeat) {
            const int64_t before = runtime.token[1].item<int64_t>();
            const int64_t id = runtime.token[0].item<int64_t>();
            const void* ptr = runtime.token.const_data_ptr();
            const int64_t version = runtime.token._version();
            std::vector<at::Tensor> outputs;
            if (compiled) outputs = runner.run(inputs);
            else outputs = {call("recsys_ops::fuse_moe_offload",
                {inputs[0], w1, w2, inputs[3], inputs[4], runtime.token, int64_t(0), c10::IValue(std::vector<int64_t>{}), c10::IValue(std::vector<int64_t>{})})[0].toTensor()};
            require(outputs.size() == 1, "Wide AOTI expected one output");
            check(inputs, outputs[0], runtime, before, id, ptr, version,
                  std::string(compiled ? "C++ wide AOTI" : "C++ wide dispatcher") +
                      " FP16 H128/I64 tile=" + std::to_string(tile_rows) +
                      " residents=" + std::to_string(residents.size()) +
                      " S=" + std::to_string(s) + " repeat=" + std::to_string(repeat),
                  repeat == 1, {}, {}, 1, true);
          }
        }
      }
      runtime.close();
    }
  }
  std::cout << "PASS standalone wide FP16 H128/I64 masked/repeated/signed-scale "
               "AOTI + dispatcher, small-tile + full-wave, miss + resident" << std::endl;
}

void benchmark_wide(const std::string& package, const std::string& archive,
                    int64_t device, int64_t iterations, const std::vector<int64_t>& sizes) {
  require(iterations > 0 && !sizes.empty(), "Benchmark needs positive iterations and shapes");
  Weights weights(archive, true);
  const auto w1 = weights.layers.at(0).first, w2 = weights.layers.at(0).second;
  require(w1.scalar_type() == (config_precision == "bf16" ? at::kBFloat16 : at::kHalf),
          "Wide benchmark --precision differs from archive dtype");
  torch::inductor::AOTIModelPackageLoader runner(package, "model");
  // Large-S benchmark overrides Runtime's 2 MiB small acceptance budget.
  // S=16384 alone needs a 4 MiB FP16 output and an 8 MiB FP32 accumulator;
  // leave room for route metadata, full-wave activations, banks and caches.
  // E4/H128/I64 CPU weights are 196608 bytes: the common 4 MiB RAM cap
  // also allows the bounded staging worker, pinned buffers and planning metadata.
  constexpr int64_t budget = 32 * 1024 * 1024;
  const int64_t weight_bytes = 6 * w1.size(2) * w2.size(2);
  const int64_t charge = ((weight_bytes + 511) / 512) * 512;
  auto stream = c10::cuda::getCurrentCUDAStream(device);
  for (const auto s : sizes) {
    require(s >= 7 && s <= 16384, "Benchmark S is outside exported [7,16384]");
    // Reference, archive/model load, context bind and route construction are untimed.
    auto reference_inputs = make_wide_inputs(s, device, w1, w2, at::Tensor());
    auto expected = reference(reference_inputs, true, config_expert_activation, config_numerical_mode)
                        .to(w1.scalar_type()).to(at::kFloat);
    for (const std::string policy : {"disabled", "static", "reuse"}) {
      int64_t capped_tiles = 0, capped_banks = 0, capped_capacity = 0;
      int64_t capped_hits = 0, capped_misses = 0;
      std::vector<int64_t> capped_traffic(stats_fields, 0);
      for (const int64_t tile_rows : {int64_t(4096), int64_t(0)}) {
        const std::vector<int64_t> residents = policy == "static"
            ? std::vector<int64_t>{0, 1, 2, 3} : std::vector<int64_t>{};
        Runtime runtime(device, w1, w2, 1, residents,
                        policy == "reuse" ? 2 * charge : 0, 0, {}, {}, true, tile_rows, budget, -1,
                        config_async_staging, config_host_profile,
                        config_expert_activation, config_numerical_mode);
        auto inputs = reference_inputs;
        inputs[5] = runtime.token;
        auto validate = [&](const at::Tensor& output, const at::Tensor& stats) {
          auto actual = output.to(at::kCPU).to(at::kFloat);
          auto error = (actual - expected).abs();
          require(output.sizes() == inputs[0].sizes() && output.scalar_type() == inputs[0].scalar_type() &&
                      output.device() == inputs[0].device() && at::isfinite(actual).all().item<bool>() &&
                      error.le(expected.abs() * .02 + .0002).all().item<bool>() &&
                      error.norm().item<float>() / expected.norm().item<float>() < .02,
                  "Benchmark independent FP32 reference mismatch");
          auto masked = inputs[3].to(at::kCPU).lt(0).all(1);
          require(actual.index({masked}).eq(0).all().item<bool>(), "Benchmark masked reference mismatch");
          const auto* st = stats.const_data_ptr<int64_t>();
          require(stats.numel() == stats_fields && st[0] == E && st[16] + st[17] == E &&
                      st[2] == st[17] * weight_bytes && st[18] == st[16] * weight_bytes &&
                      st[28] == st[25] * weight_bytes && st[28] <= runtime.dynamic_bytes &&
                      st[22] == st[23] * charge && st[22] <= runtime.dynamic_bytes &&
                      st[14] + st[19] + st[22] <= st[4] && st[4] <= budget &&
                      st[5] <= runtime.pinned_limit && st[6] <= ram_budget &&
                      st[9] == 1 && st[11] > 0 && (tile_rows == 0 || st[11] <= tile_rows),
                  "Benchmark traffic/cache/tile/budget mismatch");
          require(st[7] == (st[17] ? 2 : 0) && st[8] == (st[17] ? std::min<int64_t>(2, st[17]) : 0),
                  "Benchmark must not sacrifice banks/capacity for auto cache reservation");
          if (policy == "disabled") require(st[16] == 0 && st[23] == 0, "Disabled cache regressed");
          if (policy == "static") require(st[16] == E && st[2] == 0, "Static cache regressed");
        };
        int64_t priming_admissions = 0, priming_h2d = 0, priming_d2d = 0;
        for (int64_t observation = 0; observation < 3; ++observation) {
          auto outputs = runner.run(inputs);
          stream.synchronize();
          require(outputs.size() == 1, "Benchmark expected one output");
          auto stats = call("recsys_ops::fuse_moe_offload_stats", {runtime.token})[0].toTensor();
          validate(outputs[0], stats);
          priming_admissions += stats[25].item<int64_t>();
          priming_h2d += stats[2].item<int64_t>();
          priming_d2d += stats[28].item<int64_t>();
          if (policy == "reuse" && observation == 0)
            require(stats[25].item<int64_t>() == 0, "Reuse admission threshold regressed");
          if (policy == "reuse" && observation == 1)
            require(stats[25].item<int64_t>() > 0, "Auto plan must leave eligible promotion headroom");
        }
        double total_ms = 0;
        std::vector<double> times;
        std::vector<int64_t> peaks(stats_fields, 0), traffic(stats_fields, 0);
        int64_t allocator_peak = 0;
        for (int64_t repeat = 0; repeat < iterations; ++repeat) {
          stream.synchronize();
          c10::cuda::CUDACachingAllocator::resetPeakStats(device);
          const auto before = runtime.token[1].item<int64_t>();
          const auto version = runtime.token._version();
          const auto started = Clock::now();
          auto outputs = runner.run(inputs);
          stream.synchronize();
          const double ms = elapsed_ms(started);
          // Output destruction, stats/peak queries and numerical checks are not timed.
          require(outputs.size() == 1 && runtime.token[1].item<int64_t>() == before + 1 &&
                      runtime.token._version() == version + 1, "Benchmark AOTI token mutation mismatch");
          auto stats = call("recsys_ops::fuse_moe_offload_stats", {runtime.token})[0].toTensor();
          validate(outputs[0], stats);
          const auto* st = stats.const_data_ptr<int64_t>();
          for (int j = 0; j < stats_fields; ++j) {
            peaks[j] = std::max(peaks[j], st[j]);
            traffic[j] += st[j];
          }
          auto allocator = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
          allocator_peak = std::max<int64_t>(allocator_peak, allocator.allocated_bytes[0].peak);
          total_ms += ms;
          times.push_back(ms);
        }
        const auto samples_ms = times;
        std::sort(times.begin(), times.end());
        auto final_stats = call("recsys_ops::fuse_moe_offload_stats", {runtime.token})[0].toTensor();
        const auto* st = final_stats.const_data_ptr<int64_t>();
        if (tile_rows) {
          capped_tiles = st[10]; capped_banks = st[7]; capped_capacity = st[8];
          capped_hits = st[16]; capped_misses = st[17]; capped_traffic = traffic;
        } else {
          // Reuse may admit/evict different expert IDs under different tile
          // workspace high-water marks. Equal hit counts alone do not prove
          // equal wave boundaries; compare chunks only for fixed cache layouts.
          if (policy != "reuse") {
            require(st[10] <= capped_tiles && st[7] == capped_banks && st[8] == capped_capacity,
                    "Auto increased chunks or sacrificed banks/capacity against capped plan");
            require(traffic[2] == capped_traffic[2] && traffic[3] == capped_traffic[3] &&
                        traffic[28] == capped_traffic[28], "Fixed-cache tile comparison changed traffic");
          }
          require(st[10] == st[1], "Auto benchmark did not select feasible full-wave tiles");
          if (policy != "reuse") {
            auto ids = inputs[3].to(at::kCPU);
            int64_t max_wave_rows = 0;
            for (int64_t first = 0; first < E; first += policy == "static" ? E : 2) {
              const int64_t last = std::min<int64_t>(E, first + (policy == "static" ? E : 2));
              max_wave_rows = std::max(max_wave_rows,
                  (ids.ge(first) & ids.lt(last)).sum().item<int64_t>());
            }
            require(st[11] >= max_wave_rows && st[11] < max_wave_rows + 128,
                    "Auto full-wave rows were not minimized within tile_m alignment");
            if (s >= 8192) require(st[11] > 4096, "Auto sentinel still capped at 4096");
          }
        }
        std::cout << "BENCHMARK {\"backend\":\"standalone_cpp_aoti\",\"scenario\":\"skewed_masked_signed\","
                  << "\"synthetic_weights\":true,\"workload_note\":\"deployment performance boundary only; "
                     "small S is synthetic, not model speedup evidence\","
                  << "\"shape\":{\"S\":" << s << ",\"H\":128,\"I\":64,\"E\":4,\"K\":2},"
                  << "\"precision\":\"" << config_precision
                  << "\",\"expert_activation\":\"" << (config_expert_activation ? "gelu" : "silu")
                  << "\",\"numerical_mode\":\"" << (config_numerical_mode ? "staged" : "fused") << "\","
                  << "\"cache_policy\":\"" << policy << "\",\"max_route_tile_rows\":" << tile_rows
                  << ",\"tile_policy\":\"" << (tile_rows ? "4096" : "auto") << "\","
                  << "\"iterations\":" << iterations << ",\"latency_ms_total\":" << total_ms
                  << ",\"latency_ms_median\":" << (times[(iterations - 1) / 2] + times[iterations / 2]) / 2
                  << ",\"latency_ms_p95\":" << times[(95 * iterations + 99) / 100 - 1]
                  << ",\"tokens_per_second\":" << s * iterations * 1000.0 / total_ms
                  << ",\"timer\":\"steady_clock_forward_plus_stream_synchronize_only\","
                  << "\"gpu_budget_bytes\":" << budget << ",\"pinned_budget_bytes\":" << runtime.pinned_limit
                  << ",\"ram_budget_bytes\":" << ram_budget
                  << ",\"async_staging_enabled\":" << (config_async_staging ? "true" : "false")
                  << ",\"host_profile_enabled\":" << (config_host_profile ? "true" : "false")
                  << ",\"resident_cache_cap_bytes\":" << (residents.empty() ? 0 : runtime.resident_limit)
                  << ",\"dynamic_cache_cap_bytes\":" << runtime.dynamic_bytes
                  << ",\"route_tile_rows\":" << st[11] << ",\"route_tiles\":" << st[10]
                  << ",\"banks\":" << st[7] << ",\"capacity\":" << st[8]
                  << ",\"resident_experts\":" << st[20] << ",\"dynamic_experts\":" << st[23]
                  << ",\"resident_cache_bytes\":" << st[19] << ",\"dynamic_cache_bytes\":" << st[22]
                  << ",\"gpu_charged_peak_bytes\":" << peaks[4] << ",\"pinned_charged_peak_bytes\":" << peaks[5]
                  << ",\"ram_charged_peak_bytes\":" << peaks[6] << ",\"allocator_allocated_peak_bytes\":" << allocator_peak
                  << ",\"allocator_note\":\"process allocation high-water, not native charge\","
                  << "\"priming\":{\"observations\":3,\"cache_admissions\":" << priming_admissions
                  << ",\"weight_h2d_bytes\":" << priming_h2d << ",\"promotion_d2d_bytes\":" << priming_d2d << "},"
                  << "\"traffic_total\":{\"weight_h2d_bytes\":" << traffic[2]
                  << ",\"metadata_h2d_bytes\":" << traffic[3] << ",\"promotion_d2d_bytes\":" << traffic[28]
                  << ",\"cache_hits\":" << traffic[16] << ",\"cache_misses\":" << traffic[17] << "}"
                  << ",\"capped_comparison\":{\"fixed_cache_layout\":" << (policy == "reuse" ? "false" : "true")
                  << ",\"route_tiles\":" << capped_tiles << ",\"cache_hits_final\":" << capped_hits
                  << ",\"cache_misses_final\":" << capped_misses
                  << ",\"weight_h2d_delta_bytes\":" << traffic[2] - capped_traffic[2]
                  << ",\"metadata_h2d_delta_bytes\":" << traffic[3] - capped_traffic[3]
                  << ",\"promotion_d2d_delta_bytes\":" << traffic[28] - capped_traffic[28]
                  << ",\"traffic_note\":\"warm measured sequence only; reuse cache layouts may differ\"}"
                  << ",\"samples_ms\":[";
        for (size_t j = 0; j < samples_ms.size(); ++j)
          std::cout << (j ? "," : "") << std::setprecision(17) << samples_ms[j];
        cudaDeviceProp properties{};
        require(cudaGetDeviceProperties(&properties, static_cast<int>(device)) == cudaSuccess,
                "Benchmark environment query failed");
        std::cout << "],\"environment_signature\":{\"cxx_abi\":" << _GLIBCXX_USE_CXX11_ABI
                  << ",\"gpu\":\"" << properties.name << "\",\"target_sm\":"
                  << properties.major * 10 + properties.minor
                  << "},\"reference_checked\":true}" << std::endl;
      }
    }
  }
}

void validate_locked_archive(const Weights& weights, int64_t device) {
  require(!weights.configs.empty() && weights.configs.size() == weights.layers.size(),
          "LOCKED archive missing layer configuration");
  cudaDeviceProp properties{};
  require(cudaGetDeviceProperties(&properties, static_cast<int>(device)) == cudaSuccess,
          "LOCKED archive cannot validate target SM");
  const auto& first = weights.configs.begin()->second;
  int64_t backing = 0, resident_charge = 0;
  for (const auto& entry : weights.configs) {
    const auto& v = entry.second;
    // v1 = 23 fixed fields; v2 = the same prefix + activation@23 +
    // numerical@24, with static resident IDs from 25. The outer manifest and
    // archive format stay version 1/2 respectively.
    const size_t header = v.size() >= 1 && v[0] == 1 ? 23 : v.size() >= 1 && v[0] == 2 ? 25 : v.size() >= 1 && v[0] == 3 ? 26 : 0;
    require(header > 0 && v.size() >= header && v[21] >= 0 && v[21] <= 512 &&
                v.size() == header + static_cast<size_t>(v[21]),
            "LOCKED archive vector length/version");
    if (v[0] >= 2)
      require((v[23] == 0 || v[23] == 1) && (v[24] == 0 || v[24] == 1),
              "LOCKED archive invalid expert activation/numerical mode");
    if (v[0] == 3) require(v[25] == 1, "LOCKED archive invalid expert_type");
    require(v[1] > 0 && v[2] > 0 && v[3] > 0 && v[4] >= 1 && v[4] <= 2 &&
                v[5] >= 0 && v[5] <= 2 && v[6] >= 0 && v[6] <= INT_MAX &&
                v[7] >= 0 && v[7] <= v[1] && v[8] >= 0 && v[9] >= 0 && v[9] <= 1 &&
                v[10] >= 2 && v[10] <= INT_MAX && v[11] >= 2 && v[11] <= INT_MAX &&
                v[12] >= 0 && (v[13] == 0 || v[13] == 1) && (v[14] == 0 || v[14] == 1) &&
                (v[15] == 0 || v[15] == 32 || v[15] == 64 || v[15] == 128) &&
                (v[16] == 64 || v[16] == 128), "LOCKED archive invalid common fields");
    require(first.size() >= 23 && std::equal(v.begin(), v.begin() + 17, first.begin()) &&
                v[22] == first[22] && v[0] == first[0] &&
                (v[0] == 1 || (v[23] == first[23] && v[24] == first[24])) &&
                (v[0] != 3 || v[25] == first[25]),
            "LOCKED archive inconsistent common fields");
    require(v[22] >= 80 && v[22] == properties.major * 10 + properties.minor,
            "LOCKED archive target SM mismatch");
    require(v[17] > 0 && v[17] <= 512 && v[18] > 0 && v[18] <= INT_MAX && v[18] % 64 == 0 &&
                v[19] > 0 && v[19] <= INT_MAX / 2 && v[19] % 64 == 0 &&
                (v[20] == 0 || v[20] == 1) && v[21] <= v[17],
            "LOCKED archive invalid weight description");
    const auto& pair = weights.layers.at(entry.first);
    require(pair.first.sizes() == at::IntArrayRef({v[17], (v[0] == 3 ? 1 : 2) * v[19], v[18]}) &&
                pair.second.sizes() == at::IntArrayRef({v[17], v[18], v[19]}) &&
                pair.first.scalar_type() == (v[20] ? at::kBFloat16 : at::kHalf),
            "LOCKED archive actual weight/config mismatch");
    int64_t previous = -1;
    for (size_t j = header; j < v.size(); ++j) {
      require(v[j] > previous && v[j] < v[17] && v[7] > 0,
              "LOCKED archive invalid static resident IDs");
      previous = v[j];
    }
    const int64_t bytes = pair.first.nbytes() + pair.second.nbytes();
    require(bytes <= v[2] - backing, "LOCKED archive backing exceeds RAM cap");
    backing += bytes;
    const int64_t expert_charge = ((bytes / v[17] + 511) / 512) * 512;
    require(v[21] <= (v[7] - resident_charge) / expert_charge,
            "LOCKED archive static residents exceed cap");
    resident_charge += v[21] * expert_charge;
  }
}

std::vector<int64_t> locked_manifest(const Weights& weights) {
  std::vector<int64_t> manifest{1, static_cast<int64_t>(weights.configs.size())};
  for (const auto& entry : weights.configs) {
    manifest.push_back(entry.first);
    manifest.push_back(static_cast<int64_t>(entry.second.size()));
    manifest.insert(manifest.end(), entry.second.begin(), entry.second.end());
  }
  return manifest;
}

// Compatibility names for fixed-shape acceptance only. All context creation,
// identity validation, serialization and begin/check now belong to the PUBLIC
// header, not a private copy of the deployment adapter.
struct LockedRuntime : custom_ops::offload::Runtime {
  const at::Tensor token;
  const std::vector<int64_t> manifest;
  explicit LockedRuntime(const Weights& weights, int64_t device)
      : custom_ops::offload::Runtime(device, weights.layers, locked_manifest(weights)),
        token(custom_ops::offload::Runtime::token()),
        manifest(custom_ops::offload::Runtime::manifest()) {}
};

std::vector<custom_ops::offload::InputBinding> locked_bindings(bool chain) {
  using Binding = custom_ops::offload::InputBinding;
  std::vector<Binding> bindings{{chain ? 7 : 5, Binding::Token},
                                {1, Binding::GateUp, 0}, {2, Binding::Down, 0}};
  if (chain) {
    bindings.emplace_back(3, Binding::GateUp, 5);
    bindings.emplace_back(4, Binding::Down, 5);
  }
  return bindings;
}

std::vector<at::Tensor> locked_inputs(int64_t s, int64_t device, const Weights& weights,
                                     const at::Tensor& token, bool chain) {
  auto args = make_inputs(s, device, weights.layers.at(0).first, weights.layers.at(0).second, token);
  args[3] = at::tensor({0, 1}, at::TensorOptions().dtype(at::kInt))
                .to(args[0].device()).expand({s, K}).contiguous();
  if (chain) {
    args.insert(args.begin() + 3, weights.layers.at(5).second);
    args.insert(args.begin() + 3, weights.layers.at(5).first);
  }
  return args;
}

void check_locked_rejection(torch::inductor::AOTIModelPackageLoader& runner,
                            const std::vector<at::Tensor>& inputs, const at::Tensor& token,
                            const std::vector<int64_t>& artifact_manifest,
                            const std::string& label) {
  auto snapshot = token.clone();
  const auto version = token._version();
  const auto ptr = token.const_data_ptr();
  auto stats = call("recsys_ops::fuse_moe_offload_stats", {token})[0].toTensor().clone();
  bool rejected = false;
  bool model_invoked = false;
  try {
    call("recsys_ops::fuse_moe_offload_validate", {token, c10::IValue(artifact_manifest)});
    model_invoked = true;
    runner.run(inputs);
  }
  catch (const std::exception& error) {
    require(std::string(error.what()).find("deployment configuration mismatch") != std::string::npos,
            "LOCKED wrong rejection reason: " + std::string(error.what()));
    rejected = true;
  }
  require(rejected && !model_invoked,
          "LOCKED preflight must reject before invoking model: " + label);
  require(token.equal(snapshot) && token._version() == version && token.const_data_ptr() == ptr &&
              stats.equal(call("recsys_ops::fuse_moe_offload_stats", {token})[0].toTensor()),
          "LOCKED rejection mutated token/stats: " + label);
  std::cout << "PASS LOCKED C++ preflight reject-before-mutation " << label << std::endl;
}

void check_locked(const std::string& package, const std::string& archive, int64_t device, bool chain) {
  Weights weights(archive, false, true);
  require(weights.layers.size() == (chain ? 2 : 1), "LOCKED archive model layer count mismatch");
  validate_locked_archive(weights, device);
  const size_t header = weights.configs.begin()->second[0] == 3 ? 26 : weights.configs.begin()->second[0] == 2 ? 25 : 23;
  // Invalid vector validation is independently exercised before any context creation.
  for (int field : {-1, 0, 1, 13, 17, 21, 22, static_cast<int>(header)}) {
    auto invalid = weights;
    auto& v = invalid.configs.begin()->second;
    if (field == -1) v.pop_back();
    else if (field == static_cast<int>(header) && v.size() <= header) v.push_back(v[17]);
    else if (field == static_cast<int>(header)) v[field] = v[17];
    else v[field] = field == 22 ? v[field] + 1 : -1;
    bool rejected = false;
    try { validate_locked_archive(invalid, device); }
    catch (const std::exception& error) {
      require(std::string(error.what()).find("LOCKED archive") != std::string::npos,
              "LOCKED invalid archive wrong error");
      rejected = true;
    }
    require(rejected, "LOCKED invalid archive vector accepted");
  }
  const auto artifact_manifest = locked_manifest(weights);  // Original frozen artifact, not wrong context.
  torch::inductor::AOTIModelPackageLoader runner(package, "model");
  {
    LockedRuntime runtime(weights, device);
    int64_t call_index = 0;
    for (int64_t s : {7, 7, 7, 2, 7, 64, 2}) {
      auto inputs = locked_inputs(s, device, weights, runtime.token, chain);
      auto expected = inputs[0];
      // The LOCKED archive already contains the independently scaled layer5 down
      // weight. Use those exact weights for both graph inputs and CPU references;
      // never rescale here or change the legacy/checkpoint fixture.
      for (const auto& entry : weights.layers) {
        expected = reference({expected, entry.second.first, entry.second.second,
                              inputs[inputs.size() - 3], inputs[inputs.size() - 2], runtime.token}).to(at::kHalf);
      }
      if (chain) {
        const auto maximum = expected.abs().max().item<float>();
        require(at::isfinite(expected).all().item<bool>() && maximum > .0001,
                "LOCKED chain CPU fixture S=" + std::to_string(s) +
                    ": max abs=" + std::to_string(maximum) +
                    " must exceed 1e-4; check archive layer5 down-weight scaling");
      }
      auto snapshot = runtime.token.clone();
      const auto version = runtime.token._version();
      const auto ptr = runtime.token.const_data_ptr();
      // Graph guards are defense-in-depth: 2.6/2.11 fallback proxies discard
      // native error codes. Ordinary-dispatcher artifact preflight is mandatory per call.
      auto outputs = runtime.invoke(
          [&](const std::vector<at::Tensor>& bound) { return runner.run(bound); },
          inputs, locked_bindings(chain));
      c10::cuda::getCurrentCUDAStream(device).synchronize();
      const int64_t invocations = chain ? 2 : 1;
      require(outputs.size() == 1 && runtime.token[0].item<int64_t>() == snapshot[0].item<int64_t>() &&
                  runtime.token[1].item<int64_t>() == snapshot[1].item<int64_t>() + invocations &&
                  runtime.token._version() == version + invocations && runtime.token.const_data_ptr() == ptr,
              "LOCKED C++ compiled token mutation mismatch");
      auto actual = outputs[0].to(at::kCPU).to(at::kFloat);
      auto target = expected.to(at::kFloat);
      require(outputs[0].sizes() == inputs[0].sizes() && outputs[0].device() == inputs[0].device() &&
                  outputs[0].scalar_type() == inputs[0].scalar_type() && at::isfinite(actual).all().item<bool>() &&
                  (actual - target).abs().le(target.abs() * .02 + (chain ? .000002 : .0002)).all().item<bool>() &&
                  target.abs().max().item<float>() > (chain ? .0001 : .001) &&
                  (actual - target).norm().item<float>() / target.norm().item<float>() < .02,
              "LOCKED C++ iterative FP32 reference mismatch");
      auto stats = call("recsys_ops::fuse_moe_offload_stats", {runtime.token})[0].toTensor();
      const auto* st = stats.const_data_ptr<int64_t>();
      const auto& v = weights.configs.begin()->second;
      require(stats.numel() == stats_fields && st[0] == 2 && st[16] + st[17] == 2 &&
                  st[16] >= 1 && st[2] == st[17] * expert_bytes &&
                  st[20] == static_cast<int64_t>(weights.layers.size()) &&
                  st[21] == st[20] * expert_bytes && st[28] == st[25] * expert_bytes &&
                  st[14] + st[19] + st[22] <= st[4] && st[4] <= v[1] && st[5] <= v[3] && st[6] <= v[2] &&
                  st[9] == 1 && st[11] > 0, "LOCKED C++ cache/traffic/budget mismatch");
      require(st[24] == (call_index >= 2 ? 1 : 0) && st[25] == (call_index == 1 ? 1 : 0),
              "LOCKED C++ dynamic admission/hit progression mismatch");
      std::cout << "PASS LOCKED C++ " << (chain ? "true-chain" : "single")
                << " S=" << s << " observation=" << ++call_index << std::endl;
    }
    // The context is sealed by constructor preflight; explicit dispatcher calls must
    // supply both the per-layer contract and the same complete model manifest.
    auto inputs = locked_inputs(7, device, weights, runtime.token, false);
    const auto before = runtime.token.clone();
    const auto version = runtime.token._version();
    auto output = call("recsys_ops::fuse_moe_offload",
                       {inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], runtime.token,
                        int64_t(0), c10::IValue(weights.configs.at(0)), c10::IValue(runtime.manifest)})[0].toTensor();
    c10::cuda::getCurrentCUDAStream(device).synchronize();
    require(runtime.token[0].item<int64_t>() == before[0].item<int64_t>() &&
                runtime.token[1].item<int64_t>() == before[1].item<int64_t>() + 1 &&
                runtime.token._version() == version + 1 &&
                at::allclose(output.to(at::kCPU).to(at::kFloat), reference(inputs), .02, .0002),
            "LOCKED sealed direct forward with manifest failed");
    std::cout << "PASS LOCKED C++ sealed direct forward full manifest" << std::endl;
  }
  for (int field : {1, 13, 14, 15, 16, static_cast<int>(header)}) {
    auto wrong = weights;
    for (auto& entry : wrong.configs) {
      auto& v = entry.second;
      if (field == 1) v[field] += 512;
      else if (field == 13 || field == 14) v[field] = !v[field];
      else if (field == 15) v[field] = 64;
      else if (field == 16) v[field] = v[field] == 64 ? 128 : 64;
      else {
        require(v.size() == header + 1,
                "LOCKED archive acceptance requires one static resident per layer");
        v[field] = 2;
      }
    }
    LockedRuntime incompatible(wrong, device);
    check_locked_rejection(runner, locked_inputs(7, device, wrong, incompatible.token, chain),
                           incompatible.token, artifact_manifest, "context-field-" + std::to_string(field));
  }
  if (chain) {
    auto wrong_static = weights;
    require(wrong_static.configs.at(5).size() == header + 1,
            "LOCKED chain requires one static resident for layer5");
    wrong_static.configs.at(5)[header] = 1;
    require(wrong_static.configs.at(0) == weights.configs.at(0), "LOCKED layer0 config changed");
    {
      LockedRuntime incompatible(wrong_static, device);
      check_locked_rejection(runner, locked_inputs(7, device, wrong_static, incompatible.token, chain),
                             incompatible.token, artifact_manifest, "layer5-only-static-before-first-layer");
    }
    auto isolated = weights;
    for (auto& entry : isolated.layers) {
      entry.second.first = entry.second.first.clone();
      entry.second.second = entry.second.second.clone();
    }
    LockedRuntime incompatible(isolated, device);
    const auto first_v1 = isolated.layers.at(0).first._version();
    const auto first_v2 = isolated.layers.at(0).second._version();
    const auto second_v1 = isolated.layers.at(5).first._version();
    const auto second_v2 = isolated.layers.at(5).second._version();
    isolated.layers.at(5).first.add_(0);
    require(isolated.layers.at(0).first._version() == first_v1 &&
                isolated.layers.at(0).second._version() == first_v2 &&
                isolated.layers.at(5).first._version() == second_v1 + 1 &&
                isolated.layers.at(5).second._version() == second_v2,
            "LOCKED only layer5 weight version must change");
    check_locked_rejection(runner, locked_inputs(7, device, isolated, incompatible.token, chain),
                           incompatible.token, artifact_manifest, "layer5-only-weight-version-before-first-layer");
  }
  auto wrong_weights = weights;
  for (auto& entry : wrong_weights.layers) {
    entry.second.first = entry.second.first.narrow(0, 0, 3).clone();
    entry.second.second = entry.second.second.narrow(0, 0, 3).clone();
    wrong_weights.configs.at(entry.first)[17] = 3;
  }
  LockedRuntime incompatible(wrong_weights, device);
  check_locked_rejection(runner, locked_inputs(7, device, weights, incompatible.token, chain),
                         incompatible.token, artifact_manifest, "bound-weight-config");
}

using PublicRuntime = custom_ops::offload::Runtime;
using PublicBinding = custom_ops::offload::InputBinding;
static_assert(std::is_nothrow_destructible<PublicRuntime>::value,
              "Public runtime destructor must remain noexcept");

std::vector<PublicBinding> public_bindings(const Weights& weights) {
  std::vector<PublicBinding> bindings{{1, PublicBinding::Token}};
  int64_t n = 0;
  for (const auto& entry : weights.layers) {
    bindings.emplace_back(4 + 4 * n, PublicBinding::GateUp, entry.first);
    bindings.emplace_back(6 + 4 * n, PublicBinding::Down, entry.first);
    ++n;
  }
  return bindings;
}

std::vector<at::Tensor> public_inputs(int64_t s, int64_t device, const Weights& weights) {
  const auto cpu = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const auto cuda = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device));
  const int64_t h = weights.layers.begin()->second.first.size(2);
  std::vector<at::Tensor> inputs{
      (at::arange(s, cpu).sin() * .1).reshape({s, 1}).to(cuda, at::kHalf), {},
      (at::arange(s * h, cpu).cos() * .25).reshape({s, h}).to(cuda, at::kHalf)};
  for (const auto& entry : weights.layers) {
    auto ids = (at::arange(s * K, at::TensorOptions().dtype(at::kLong)) + entry.first)
                   .remainder(entry.second.first.size(0)).reshape({s, K}).to(at::kInt);
    for (int64_t row = 0; row < s; row += 7) ids[row][1].fill_(-1);
    auto scale = at::tensor({.75f, -.25f}, cpu).expand({s, K}).contiguous();
    inputs.insert(inputs.end(), {ids.to(cuda), {}, scale.to(cuda), {}});
  }
  return inputs;
}

at::Tensor public_reference(const std::vector<at::Tensor>& inputs, const Weights& weights,
                            bool rounded = true) {
  // Fused scalar arithmetic accumulates in FP32; cast only at custom-op
  // boundaries and at the final output, matching the compiled adapter.
  auto value = (inputs[2].to(at::kCPU).to(at::kFloat) * .875 + .03125).to(at::kHalf);
  int64_t n = 0;
  for (const auto& entry : weights.layers) {
    const int64_t h = entry.second.first.size(2);
    value = value.size(1) < h ? at::cat({value, value}, 1)
                             : value.narrow(1, 0, h).contiguous();
    // Independent per-expert reference with native FP16 output, followed by
    // FP32 scalar arithmetic fused through the final business add.
    value = reference({value, entry.second.first, entry.second.second,
                       inputs[3 + 4 * n], inputs[5 + 4 * n], inputs[1]}, rounded)
                .to(at::kHalf).to(at::kFloat) * .75;
    ++n;
    if (static_cast<size_t>(n) < weights.layers.size()) value = value.to(at::kHalf);
  }
  return (value + inputs[0].to(at::kCPU).to(at::kFloat) * .125).to(at::kHalf);
}

std::vector<int64_t> public_status(PublicRuntime& runtime) {
  auto tensor = runtime.status_tensor();
  require(tensor.device().is_cpu() && tensor.scalar_type() == at::kLong &&
              tensor.sizes() == at::IntArrayRef({10}), "PUBLIC status must be CPU int64[10]");
  const auto* data = tensor.const_data_ptr<int64_t>();
  require(data[0] == 1 && data[1] >= 0 && data[1] <= 2, "PUBLIC status v1 state mismatch");
  return std::vector<int64_t>(data, data + 10);
}

void use_artifact_config(const Weights& weights) {
  const auto& v = weights.configs.begin()->second;
  const std::string precision = v[20] ? "bf16" : "fp16";
  const int64_t activation = v[0] >= 2 ? v[23] : 0;
  const int64_t numerical = v[0] >= 2 ? v[24] : 0;
  config_expert_type = v[0] == 3 ? v[25] : 0;
  require(!precision_explicit || config_precision == precision,
          "--precision differs from frozen artifact dtype");
  require(!expert_activation_explicit || config_expert_activation == activation,
          "--expert-activation differs from frozen artifact config");
  require(!numerical_mode_explicit || config_numerical_mode == numerical,
          "--numerical-mode differs from frozen artifact config");
  require(!async_staging_explicit || v[13] == 1,
          "--async-staging differs from frozen artifact config");
  require(!host_profile_explicit || v[14] == 1,
          "--host-profile differs from frozen artifact config");
  config_precision = precision;
  config_expert_activation = activation;
  config_numerical_mode = numerical;
  config_async_staging = v[13] != 0;
  config_host_profile = v[14] != 0;
}

void assert_no_python_interpreter();

void check_geglu(const std::string& package, const std::string& archive,
                 int64_t device, const std::vector<int64_t>& sizes, int64_t expert_type = 0) {
  require(!sizes.empty(), "GEGLU requires dynamic sizes");
  for (const auto s : sizes) require(s >= 2 && s <= 64, "GEGLU S is outside exported [2,64]");
  for (int64_t s : {2, 7, 64})
    require(std::find(sizes.begin(), sizes.end(), s) != sizes.end(),
            "GEGLU requires S=2, S=7 and S=64");
  Weights weights(archive, false, true);
  require(weights.layers.size() == 2 && weights.layers.count(0) && weights.layers.count(5),
          "GEGLU requires the frozen two-layer 0/5 chain");
  validate_locked_archive(weights, device);
  use_artifact_config(weights);
  for (const auto& entry : weights.configs)
    require(entry.second[0] == (expert_type == 1 ? 3 : 2) &&
                (expert_type == 0 || entry.second[25] == 1) &&
                entry.second[23] == 1 && entry.second[24] == 1 &&
                entry.second[5] == 1 && entry.second[8] > 0,
            "GEGLU requires dynamic config-v2 GELU/staged accumulate artifact");
  const auto manifest = locked_manifest(weights);
  PublicRuntime runtime(device, weights.layers, manifest);
  require(runtime.manifest() == manifest, "GEGLU frozen manifest changed");
  require(runtime.expert_type() == expert_type, "Frozen expert_type mismatch");
  torch::inductor::AOTIModelPackageLoader loader(package, "model");
  const auto bindings = locked_bindings(true);
  auto observations = sizes;
  observations.push_back(sizes.front());  // Exercise reuse after dynamic shape changes.
  for (const auto s : observations) {
    auto inputs = locked_inputs(s, device, weights, runtime.token(), true);
    const auto chain_reference = [&](int64_t activation) {
      auto value = inputs[0];
      for (const auto& entry : weights.layers) {
        const auto& v = weights.configs.at(entry.first);
        value = reference({value, entry.second.first, entry.second.second,
                           inputs[5], inputs[6], inputs[7]}, true, activation, v[24], runtime.expert_type())
                    .to(inputs[0].scalar_type());
      }
      return value.to(at::kCPU).to(at::kFloat);
    };
    // Artifact fields are authoritative; GEGLU was checked above to be 1/1.
    const auto expected = chain_reference(config_expert_activation);
    const auto silu = chain_reference(0);
    const double norm = expected.to(at::kDouble).norm().item<double>();
    const double discrimination = (expected - silu).to(at::kDouble).norm().item<double>() /
                                  std::max(norm, 1e-12);
    require(at::isfinite(expected).all().item<bool>() &&
                expected.abs().max().item<float>() > .0001 && discrimination > .001,
            "GEGLU reference underflowed or cannot distinguish GELU from SiLU");
    const auto before = public_status(runtime);
    const auto token = runtime.token().clone();
    const auto version = runtime.token()._version();
    const auto ptr = runtime.token().const_data_ptr();
    auto outputs = runtime.invoke(
        [&](const std::vector<at::Tensor>& bound) { return loader.run(bound); }, inputs, bindings);
    c10::cuda::getCurrentCUDAStream(device).synchronize();
    const auto after = public_status(runtime);
    require(after[1] == 0 && after[2] == 0 && after[3] == 0 &&
                after[4] == before[4] + 2 && after[5] == after[4] &&
                after[8] == before[8] + 1 && after[9] == 5,
            "GEGLU managed status progression mismatch");
    require(runtime.token().const_data_ptr() == ptr && runtime.token()._version() == version + 2 &&
                runtime.token()[0].item<int64_t>() == token[0].item<int64_t>() &&
                runtime.token()[1].item<int64_t>() == token[1].item<int64_t>() + 2,
            "GEGLU compiled chain token identity/mutation mismatch");
    require(outputs.size() == 1 && outputs[0].defined() && outputs[0].sizes() == inputs[0].sizes() &&
                outputs[0].device() == inputs[0].device() &&
                outputs[0].scalar_type() == inputs[0].scalar_type(), "GEGLU output metadata mismatch");
    const auto actual = outputs[0].to(at::kCPU).to(at::kFloat);
    const auto error = actual.to(at::kDouble) - expected.to(at::kDouble);
    const double relative_l2 = error.norm().item<double>() / norm;
    const double reference_rms = norm / std::sqrt(static_cast<double>(expected.numel()));
    const double dtype_epsilon = inputs[0].scalar_type() == at::kHalf ? 1.0 / 1024 : 1.0 / 128;
    // MLP outputs have no second gate/up product suppressing their scale.
    // Use the same dtype/RMS budget as Python; legacy GLU stays at 2e-6.
    const double comparison_atol = .000002 + (expert_type == 1 ? 2 * dtype_epsilon * reference_rms : 0);
    require(at::isfinite(actual).all().item<bool>() &&
                (actual - expected).abs().le(expected.abs() * .02 + comparison_atol).all().item<bool>() &&
                relative_l2 < .02, "dtype-rounded staged expert-chain reference mismatch");
    auto stats = call("recsys_ops::fuse_moe_offload_stats", {runtime.token()})[0].toTensor();
    require(stats.device().is_cpu() && stats.scalar_type() == at::kLong && stats.numel() == stats_fields,
            "GEGLU stats schema mismatch");
    const auto* st = stats.const_data_ptr<int64_t>();
    const auto& v = weights.configs.begin()->second;
    require(st[0] == 2 && st[16] + st[17] == 2 && st[9] == 1 && st[11] > 0 &&
                st[14] + st[19] + st[22] <= st[4] && st[4] <= v[1] && st[5] <= v[3] && st[6] <= v[2],
            "GEGLU active routes/activation/budget mismatch");
    assert_no_python_interpreter();
    std::cout << std::setprecision(17) << (expert_type == 1 ? "ACCEPTANCE_MLP_CPP {\"S\":" : "ACCEPTANCE_GEGLU_CPP {\"S\":") << s
              << ",\"label\":\"C++ AOTI S=" << s << "\",\"precision\":\"" << config_precision
              << "\",\"expert_type\":\"" << (expert_type == 1 ? "mlp" : "glu")
              << "\",\"expert_activation\":\"gelu\",\"numerical_mode\":\"staged\""
              << ",\"comparison_atol\":" << comparison_atol
              << ",\"comparison_rtol\":0.02,\"reference_rms\":" << reference_rms
              << ",\"max_abs_error\":" << error.abs().max().item<double>()
              << ",\"rmse\":" << error.square().mean().sqrt().item<double>()
              << ",\"relative_l2\":" << relative_l2
              << ",\"exact_equal\":" << (actual.equal(expected) ? "true" : "false")
              << ",\"gelu_vs_silu_relative_l2\":" << discrimination
              << ",\"token_checked\":true,\"status_checked\":true,\"reference_checked\":true"
              << ",\"python_uninitialized\":true,\"strict_no_jit\":true"
              << ",\"async_staging_enabled\":" << (config_async_staging ? "true" : "false")
              << ",\"host_profile_enabled\":" << (config_host_profile ? "true" : "false")
              << ",\"status_values\":[";
    for (size_t j = 0; j < after.size(); ++j) std::cout << (j ? "," : "") << after[j];
    std::cout << "],\"stats_values\":[";
    for (int j = 0; j < stats_fields; ++j) std::cout << (j ? "," : "") << st[j];
    std::cout << "]}" << std::endl;
  }
  runtime.close();
}

void print_public_status(const std::vector<int64_t>& status, const std::string& label) {
  static std::mutex output_mutex;
  const std::lock_guard<std::mutex> output_lock(output_mutex);
  static const char* fields[] = {"version", "state", "managed", "pending_error",
      "forward_sequence", "success_sequence", "failure_layer", "failure_sequence",
      "managed_sequence", "success_layer"};
  static const char* states[] = {"healthy", "recoverable", "poisoned"};
  std::cout << "PUBLIC_STATUS {\"label\":\"" << label << "\",\"status\":{";
  for (size_t j = 0; j < status.size(); ++j) {
    std::cout << (j ? "," : "") << '"' << fields[j] << "\":";
    if (j == 1) std::cout << '"' << states[status[j]] << '"';
    else std::cout << status[j];
  }
  std::cout << "}}" << std::endl;
}

void public_check(torch::inductor::AOTIModelPackageLoader& loader, PublicRuntime& runtime,
                  const Weights& weights, const std::vector<at::Tensor>& inputs,
                  const std::vector<PublicBinding>& bindings, const std::string& label) {
  const auto before = public_status(runtime);
  const auto token = runtime.token().clone();
  const auto ptr = runtime.token().const_data_ptr();
  const auto version = runtime.token()._version();
  auto expected = public_reference(inputs, weights).to(at::kFloat);
  auto mathematical = public_reference(inputs, weights, false).to(at::kFloat);
  auto outputs = runtime.invoke(
      [&](const std::vector<at::Tensor>& bound) { return loader.run(bound); }, inputs, bindings);
  c10::cuda::getCurrentCUDAStream(runtime.device()).synchronize();
  const auto after = public_status(runtime);
  const int64_t count = static_cast<int64_t>(weights.layers.size());
  require(after[1] == 0 && after[2] == 0 && after[3] == 0 &&
              after[4] == before[4] + count && after[5] == after[4] &&
              after[8] == before[8] + 1 && after[9] == weights.layers.rbegin()->first,
          "PUBLIC managed status progression mismatch");
  require(runtime.token().const_data_ptr() == ptr && runtime.token()._version() == version + count &&
              runtime.token()[0].item<int64_t>() == token[0].item<int64_t>() &&
              runtime.token()[1].item<int64_t>() == token[1].item<int64_t>() + count,
          "PUBLIC token identity/mutation mismatch");
  require(outputs.size() == 1 && outputs[0].defined() && outputs[0].sizes() == expected.sizes() &&
              outputs[0].device() == inputs[2].device() && outputs[0].scalar_type() == at::kHalf,
          "PUBLIC output metadata mismatch");
  const auto actual = outputs[0].to(at::kCPU).to(at::kFloat);
  const auto error = (actual - expected).abs();
  const auto contribution = expected - (inputs[0].to(at::kCPU) * .125).to(at::kFloat);
  // Save numerical failures before asserting, using exactly the same limits.
  if (!output_directory.empty()) {
    torch::serialize::OutputArchive raw;
    for (size_t j = 0; j < inputs.size(); ++j)
      raw.write("input_" + std::to_string(j), inputs[j].to(at::kCPU));
    raw.write("output", outputs[0].to(at::kCPU));
    raw.write("expected", expected);
    raw.write("mathematical", mathematical);
    raw.write("status", runtime.status_tensor());
    raw.write("stats", call("recsys_ops::fuse_moe_offload_stats", {runtime.token()})[0].toTensor());
    raw.save_to((output_directory / (label + "_seq" + std::to_string(after[4]) + ".tensors.pt")).string());
  }
  require(at::isfinite(actual).all().item<bool>() &&
              error.le(expected.abs() * .02 + .00002).all().item<bool>() &&
              contribution.norm().item<float>() > .00001 &&
              error.norm().item<float>() / contribution.norm().item<float>() < .05,
          "PUBLIC arbitrary scalar business dtype-rounded FP32 reference mismatch: " + label +
              " error_norm=" + std::to_string(error.norm().item<float>()) +
              " contribution_norm=" + std::to_string(contribution.norm().item<float>()) +
              " relative_l2=" + std::to_string(
                  error.norm().item<float>() / contribution.norm().item<float>()));
  require((actual - mathematical).abs().le(mathematical.abs() * .02 + .00002).all().item<bool>(),
          "PUBLIC original mathematical pointwise reference mismatch: " + label);
  std::cout << "PUBLIC_NUMERICAL {\"label\":\"" << label
            << "\",\"rounded_relative_l2\":" << error.norm().item<float>() / contribution.norm().item<float>()
            << ",\"mathematical_max_abs\":" << (actual - mathematical).abs().max().item<float>()
            << ",\"mathematical_rmse\":" << (actual - mathematical).square().mean().sqrt().item<float>()
            << ",\"mathematical_contribution_relative_l2\":" << (actual - mathematical).norm().item<float>() /
                (mathematical - (inputs[0].to(at::kCPU) * .125).to(at::kFloat)).norm().item<float>()
            << "}" << std::endl;
  print_public_status(after, label);
}

void public_negative(const std::string& package, const Weights& weights, int64_t device,
                     int64_t error_layer) {
  require(error_layer >= 0 && static_cast<size_t>(error_layer) < weights.layers.size(),
          "PUBLIC negative layer index out of range");
  PublicRuntime runtime(device, weights.layers, locked_manifest(weights));
  torch::inductor::AOTIModelPackageLoader loader(package, "model");
  const auto bindings = public_bindings(weights);
  auto inputs = runtime.prepare(public_inputs(7, device, weights), bindings);
  auto invalid = inputs;
  const auto layer = std::next(weights.layers.begin(), error_layer);
  invalid[3 + 4 * error_layer] = invalid[3 + 4 * error_layer].clone();
  invalid[3 + 4 * error_layer][0][0].fill_(layer->second.first.size(0));
  // Establish the original native route diagnostic before the official AOTI.
  // Do not call runtime methods from its callback (reentrant access is forbidden).
  {
    PublicRuntime direct(device, weights.layers, locked_manifest(weights));
    auto args = direct.prepare(public_inputs(7, device, weights), bindings);
    auto x = at::zeros({7, layer->second.first.size(2)}, inputs[2].options());
    bool native_rejected = false;
    try {
      direct.invoke([&](const std::vector<at::Tensor>& bound) {
        return std::vector<at::Tensor>{call("recsys_ops::fuse_moe_offload",
            {x, bound[4 + 4 * error_layer], bound[6 + 4 * error_layer],
             invalid[3 + 4 * error_layer], bound[5 + 4 * error_layer], bound[1], layer->first,
             c10::IValue(weights.configs.at(layer->first)),
             c10::IValue(locked_manifest(weights))})[0].toTensor()};
      }, args, bindings);
    } catch (const std::exception& error) {
      native_rejected = true;
      require(std::string(error.what()).find("topk_ids must be -1 or in [0,E)") != std::string::npos,
              "PUBLIC direct native check lost original route diagnostic");
      std::cout << "PUBLIC_NATIVE_ERROR " << error.what() << std::endl;
    }
    require(native_rejected && public_status(direct)[6] == layer->first,
            "PUBLIC direct checked native route must reject at the requested layer");
    direct.close();
  }
  const auto before = public_status(runtime);
  if (!output_directory.empty()) {
    torch::serialize::OutputArchive raw;
    for (size_t j = 0; j < invalid.size(); ++j)
      raw.write("input_" + std::to_string(j), invalid[j].to(at::kCPU));
    raw.write("status", runtime.status_tensor());
    raw.save_to((output_directory / "negative.inputs.pt").string());
  }
  bool reached = false, published = false, rejected = false;
  try {
    auto outputs = runtime.invoke([&](const std::vector<at::Tensor>& bound) {
      reached = true;
      return loader.run(bound);  // No dispatcher replacement or fake output handle.
    }, invalid, bindings);
    published = true;
  } catch (const std::exception& error) {
    rejected = true;
    std::cout << "PUBLIC_ERROR " << error.what() << std::endl;
    require(std::string(error.what()).find("topk_ids must be -1 or in [0,E)") != std::string::npos,
            "PUBLIC native check must expose original route error, not generic proxy failure");
  }
  const auto after = public_status(runtime);
  print_public_status(after, "negative-first-failure");
  require(rejected && reached && !published && after[1] == 1 && after[2] == 0 && after[3] == 1 &&
              after[6] == layer->first && after[7] == before[4] + error_layer + 1 &&
              after[4] == after[7] && after[5] == before[4] + error_layer,
          "PUBLIC loaded AOTI must expose first failure and reject all partial outputs");
  reached = false;
  rejected = false;
  try {
    runtime.invoke([&](const std::vector<at::Tensor>& bound) {
      reached = true;
      return loader.run(bound);
    }, inputs, bindings);
  } catch (const std::exception&) { rejected = true; }
  const auto sticky = public_status(runtime);
  require(rejected && !reached && sticky[6] == after[6] && sticky[7] == after[7],
          "PUBLIC first error must remain sticky until acknowledge");
  runtime.acknowledge();
  const auto recovered = public_status(runtime);
  require(recovered[1] == 0 && recovered[2] == 0 && recovered[3] == 0,
          "PUBLIC recoverable acknowledge failed");
  public_check(loader, runtime, weights, inputs, bindings, "negative-recovered");
  runtime.close();
}

void public_concurrency(const std::string& package, const Weights& weights, int64_t device) {
  // Separate OFFICIAL loader per thread, immutable shared weights, disjoint safe
  // input vectors. No claim that one model_package_loader is thread safe.
  std::promise<void> ready;
  auto start = ready.get_future().share();
  std::vector<std::future<int64_t>> workers;
  std::vector<std::promise<void>> initialized(3);
  std::vector<std::future<void>> initialization;
  for (auto& signal : initialized) initialization.push_back(signal.get_future());
  for (int index = 0; index < 3; ++index) {
    workers.emplace_back(std::async(std::launch::async, [&, index, start] {
      c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
      at::NoGradGuard no_grad;
      PublicRuntime runtime(device, weights.layers, locked_manifest(weights));
      torch::inductor::AOTIModelPackageLoader loader(package, "model");
      auto bindings = public_bindings(weights);
      auto inputs = runtime.prepare(public_inputs(7, device, weights), bindings);
      initialized[index].set_value();
      start.wait();
      for (int repeat = 0; repeat < 2; ++repeat)
        public_check(loader, runtime, weights, inputs, bindings,
                     "thread" + std::to_string(index) + "-repeat" + std::to_string(repeat));
      const auto id = runtime.token()[0].item<int64_t>();
      runtime.close();
      return id;
    }));
  }
  bool all_initialized = true;
  for (auto& signal : initialization)
    all_initialized = (signal.wait_for(std::chrono::seconds(120)) == std::future_status::ready) && all_initialized;
  ready.set_value();  // Release successful threads even if another failed setup.
  require(all_initialized, "PUBLIC concurrent loader/context initialization timed out");
  std::set<int64_t> ids;
  for (auto& worker : workers) ids.insert(worker.get());
  require(ids.size() == 3, "PUBLIC concurrent contexts must have distinct tokens");

  PublicRuntime runtime(device, weights.layers, locked_manifest(weights));
  torch::inductor::AOTIModelPackageLoader loader(package, "model");
  auto bindings = public_bindings(weights);
  auto inputs = runtime.prepare(public_inputs(7, device, weights), bindings);
  std::promise<void> entered, release, closing;
  auto entered_signal = entered.get_future();
  auto release_signal = release.get_future().share();
  auto closing_signal = closing.get_future();
  auto invocation = std::async(std::launch::async, [&] {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    at::NoGradGuard no_grad;
    return runtime.invoke([&](const std::vector<at::Tensor>& bound) {
      entered.set_value();
      require(release_signal.wait_for(std::chrono::seconds(120)) == std::future_status::ready,
              "PUBLIC close/invoke coordination timed out");
      return loader.run(bound);
    }, inputs, bindings);
  });
  require(entered_signal.wait_for(std::chrono::seconds(120)) == std::future_status::ready,
          "PUBLIC callback not entered");
  auto closure = std::async(std::launch::async, [&] { closing.set_value(); runtime.close(); });
  closing_signal.wait();
  const bool close_blocked = closure.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;
  release.set_value();  // Always release before assertions/future destruction.
  require(invocation.get().size() == 1, "PUBLIC safe invoke lost output during close");
  closure.get();
  require(close_blocked && runtime.closed(), "PUBLIC close must serialize behind invocation");
  bool called = false, rejected = false;
  try {
    runtime.invoke([&](const std::vector<at::Tensor>& bound) {
      called = true; return loader.run(bound);
    }, inputs, bindings);
  } catch (const std::exception&) { rejected = true; }
  require(rejected && !called, "PUBLIC invoke after close must reject before callback");
  runtime.close();  // Idempotent safe close; no destructor exception.
}

void check_public(const std::string& package, const std::string& archive, int64_t device,
                  int64_t error_layer = -1) {
  Weights weights(archive, false, true, true);
  require(weights.layers.size() >= 1 && weights.layers.size() <= 3,
          "PUBLIC acceptance adapter supports one/two/three explicit layers only");
  if (error_layer >= 0) {
    public_negative(package, weights, device, error_layer);
    return;
  }
  PublicRuntime runtime(device, weights.layers, locked_manifest(weights));
  const auto bindings = public_bindings(weights);
  // Load the intact, prelinked official PT2 directly. Its generated .cpp basename
  // anchors load_metadata; metadata, compile/link flags and extern-kernel JSON
  // remain loader-owned inventory even when deployment never compiles anything.
  // T09 excludes an external source tree, not embedded immutable export sources.
  torch::inductor::AOTIModelPackageLoader loader(package, "model");
  for (int64_t s : {2, 7, 64, 2}) {
    auto raw = public_inputs(s, device, weights);
    auto inputs = runtime.prepare(raw, bindings);
    require(!raw[1].defined() && !raw[4].defined() && !raw[6].defined() &&
                raw[0].is_same(inputs[0]) && raw[2].is_same(inputs[2]),
            "PUBLIC prepare must fill only mapped undefined slots without mutating input vector");
    public_check(loader, runtime, weights, inputs, bindings, "S" + std::to_string(s));
  }
  auto inputs = runtime.prepare(public_inputs(7, device, weights), bindings);
  for (bool duplicate : {false, true}) {
    auto bad = bindings;
    if (duplicate) bad.push_back(bindings[0]); else bad.pop_back();
    bool called = false, rejected = false;
    const auto before = public_status(runtime);
    try {
      runtime.invoke([&](const std::vector<at::Tensor>&) {
        called = true; return std::vector<at::Tensor>{};
      }, inputs, bad);
    } catch (const std::exception&) { rejected = true; }
    require(rejected && !called && public_status(runtime) == before,
            "PUBLIC incomplete/duplicate bindings must reject before begin/callback");
  }
  struct CallbackError : std::runtime_error { using std::runtime_error::runtime_error; };
  bool rejected = false;
  try {
    runtime.invoke([](const std::vector<at::Tensor>&) -> std::vector<at::Tensor> {
      throw CallbackError("unknown business callback error");
    }, inputs, bindings);
  } catch (const CallbackError&) { rejected = true; }
  require(rejected && public_status(runtime)[2] == 0, "PUBLIC callback error/check cleanup failed");
  public_check(loader, runtime, weights, inputs, bindings, "callback-recovered");
  runtime.close();
  public_concurrency(package, weights, device);
  std::cout << "PASS PUBLIC official C++ AOTI heterogeneous dynamic-S scalar-business adapter" << std::endl;
}

void benchmark_locked(const std::string& package, const std::string& archive,
                      int64_t device, int64_t s, int64_t k, int64_t iterations,
                      const std::string& prime, const std::vector<std::string>& schedule) {
  require(s > 0 && k > 0 && k <= 128 && iterations > 0 &&
              schedule.size() == static_cast<size_t>(iterations), "Invalid locked benchmark schedule");
  const auto wall_start = Clock::now();
  rusage process_before{};
  require(getrusage(RUSAGE_SELF, &process_before) == 0, "getrusage failed");
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  require(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "sched_getaffinity failed");
  Weights weights(archive, false, true, true);
  require(weights.layers.size() == 1 && weights.layers.count(0), "Benchmark requires layer 0 only");
  const auto& pair = weights.layers.at(0);
  const int64_t e = pair.first.size(0), h = pair.first.size(2);
  validate_locked_archive(weights, device);
  use_artifact_config(weights);
  const auto dtype = pair.first.scalar_type();
  require(e == 16 && (dtype == at::kHalf || dtype == at::kBFloat16),
          "Locked synthetic fixture requires E16 fp16/bf16");
  const auto cpu = at::TensorOptions().device(at::kCPU);
  const auto gpu = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device));
  auto x = (at::arange(s * h, cpu.dtype(at::kFloat)).cos() * .25).reshape({s, h}).to(gpu, dtype);
  auto rows = at::arange(s, cpu.dtype(at::kLong)).reshape({s, 1});
  auto slots = at::arange(k, cpu.dtype(at::kLong)).reshape({1, k});
  std::map<std::string, at::Tensor> routes, scales, expected;
  routes["hot_a"] = (rows + slots).remainder(8).to(at::kInt);
  routes["hot_b"] = ((rows + slots).remainder(8) + 8).to(at::kInt);
  routes["uniform"] = (rows * k + slots).remainder(e).to(at::kInt);
  auto skewed = at::where((rows + slots).remainder(11).eq(0),
                          (rows + slots).remainder(6), at::zeros({s, k}, cpu.dtype(at::kLong)));
  skewed.masked_fill_((rows + 2 * slots).remainder(5).eq(0), -1);
  for (int64_t row = 0; row < s; row += 7) skewed[row].fill_(-1);
  routes["skewed_masked"] = skewed.to(at::kInt);
  for (auto& entry : routes) {
    auto scale = at::full({s, k}, 1.0 / k, cpu.dtype(at::kFloat));
    if (entry.first == "skewed_masked")
      scale = at::where((rows + slots).remainder(3).eq(0), -scale * .5, scale * 1.25);
    // Independent actual CPU ATen reference outside all invocation timers.
    expected[entry.first] = reference({x, pair.first, pair.second, entry.second, scale, {}},
                                      true, config_expert_activation, config_numerical_mode)
                                .to(dtype).to(at::kFloat);
    entry.second = entry.second.to(gpu);
    scales[entry.first] = scale.to(gpu);
  }
  require(routes.count(prime), "Unknown cold route");
  for (const auto& route : schedule) require(routes.count(route), "Unknown warm route");
  const auto load_start = Clock::now();
  torch::inductor::AOTIModelPackageLoader loader(package, "model");
  const double model_load_ms = elapsed_ms(load_start);
  const std::vector<PublicBinding> bindings{{1, PublicBinding::GateUp, 0},
                                           {2, PublicBinding::Down, 0}, {5, PublicBinding::Token}};
  auto stream = c10::cuda::getCurrentCUDAStream(device);
  c10::cuda::CUDACachingAllocator::resetPeakStats(device);
  struct Sample {
    std::string phase, route;
    double ms, cpu_ms;
    std::vector<int64_t> stats;
    int64_t rss, gpu_peak, gpu_reserved_peak;
  };
  std::vector<Sample> cold, priming, warm;
  std::vector<double> initialization;
  std::vector<std::vector<int64_t>> initialization_stats;
  auto stats_of = [&](PublicRuntime& runtime) {
    auto tensor = call("recsys_ops::fuse_moe_offload_stats", {runtime.token()})[0].toTensor();
    require(tensor.numel() == stats_fields, "Locked benchmark stats schema mismatch");
    const auto* ptr = tensor.const_data_ptr<int64_t>();
    return std::vector<int64_t>(ptr, ptr + stats_fields);
  };
  auto initialize = [&] {
    const auto start = Clock::now();
    auto runtime = std::make_unique<PublicRuntime>(device, weights.layers, locked_manifest(weights));
    stream.synchronize();
    initialization.push_back(elapsed_ms(start));
    initialization_stats.push_back(stats_of(*runtime));
    return runtime;
  };
  auto measure = [&](PublicRuntime& runtime, const std::string& phase, const std::string& route) {
    auto inputs = runtime.prepare({x, {}, {}, routes.at(route), scales.at(route), {}}, bindings);
    const auto status_before = public_status(runtime);
    const auto token_before = runtime.token().clone();
    const auto token_ptr = runtime.token().const_data_ptr();
    const auto token_version = runtime.token()._version();
    stream.synchronize();
    rusage before{}, after{};
    require(getrusage(RUSAGE_SELF, &before) == 0, "getrusage failed");
    const auto start = Clock::now();
    auto outputs = runtime.invoke([&](const std::vector<at::Tensor>& bound) { return loader.run(bound); },
                                  inputs, bindings);
    stream.synchronize();
    const double ms = elapsed_ms(start);
    require(getrusage(RUSAGE_SELF, &after) == 0, "getrusage failed");
    const auto cpu_ms = [](const rusage& value) {
      return 1000.0 * (value.ru_utime.tv_sec + value.ru_stime.tv_sec) +
             .001 * (value.ru_utime.tv_usec + value.ru_stime.tv_usec);
    };
    auto stats = stats_of(runtime);
    const auto status = public_status(runtime);
    require(status[1] == 0 && status[2] == 0 && status[3] == 0 &&
                status[4] == status_before[4] + 1 && status[5] == status[4] &&
                status[8] == status_before[8] + 1 && status[9] == 0,
            "Benchmark managed status progression mismatch");
    require(runtime.token().const_data_ptr() == token_ptr &&
                runtime.token()._version() == token_version + 1 &&
                runtime.token()[0].item<int64_t>() == token_before[0].item<int64_t>() &&
                runtime.token()[1].item<int64_t>() == token_before[1].item<int64_t>() + 1,
            "Benchmark token identity/mutation mismatch");
    auto allocator = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    require(outputs.size() == 1 && outputs[0].sizes() == x.sizes() && outputs[0].scalar_type() == dtype &&
                outputs[0].device() == x.device(), "Locked benchmark output metadata mismatch");
    auto actual = outputs[0].to(at::kCPU).to(at::kFloat);
    auto error = (actual - expected.at(route)).abs();
    require(at::isfinite(actual).all().item<bool>() &&
                error.le(expected.at(route).abs() * .02 + .00002).all().item<bool>(),
            "Locked benchmark independent reference mismatch");
    require(stats[2] == stats[17] * 6 * h * pair.second.size(2), "Locked benchmark H2D accounting mismatch");
    const auto& config = weights.configs.at(0);
    require(stats[4] <= config[1] && stats[6] <= config[2] && stats[5] <= config[3], "Budget exceeded");
    return Sample{phase, route, ms, cpu_ms(after) - cpu_ms(before), stats,
                  after.ru_maxrss * 1024, allocator.allocated_bytes[0].peak, allocator.reserved_bytes[0].peak};
  };
  for (int64_t n = 0; n < iterations; ++n) {
    auto runtime = initialize();
    cold.push_back(measure(*runtime, "cold", prime));
    runtime->close();
  }
  {
    auto runtime = initialize();
    for (int n = 0; n < 3; ++n) priming.push_back(measure(*runtime, "priming", prime));
    for (const auto& route : schedule) warm.push_back(measure(*runtime, "warm", route));
    runtime->close();
  }
  std::ostringstream report;
  report << std::setprecision(17);
  const auto vector_json = [&](const auto& values) {
    report << '[';
    for (size_t j = 0; j < values.size(); ++j) report << (j ? "," : "") << values[j];
    report << ']';
  };
  const auto samples_json = [&](const std::vector<Sample>& samples) {
    report << '[';
    for (size_t n = 0; n < samples.size(); ++n) {
      const auto& sample = samples[n];
      report << (n ? "," : "") << "{\"phase\":" << std::quoted(sample.phase)
             << ",\"route\":" << std::quoted(sample.route) << ",\"host_ms\":" << sample.ms
             << ",\"cpu_ms\":" << sample.cpu_ms << ",\"process_peak_rss_bytes\":" << sample.rss
             << ",\"gpu_max_allocated_bytes\":" << sample.gpu_peak
             << ",\"gpu_max_reserved_bytes\":" << sample.gpu_reserved_peak << ",\"stats_values\":";
      vector_json(sample.stats);
      report << '}';
    }
    report << ']';
  };
  const auto latency_json = [&](std::vector<double> times) {
    std::sort(times.begin(), times.end());
    const size_t n = times.size();
    report << "{\"samples\":" << n << ",\"latency_ms_median\":"
           << (n % 2 ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2)
           << ",\"latency_ms_p95\":" << times[static_cast<size_t>(std::ceil(.95 * n)) - 1] << '}';
  };
  report << "{\"schema_version\":3,\"case_version\":\"T04.frozen_config.v1\","
            "\"backend\":\"cpp_public_aoti\",\"final_frozen\":true,"
            "\"timer\":\"steady_clock_invoke_preflight_begin_loader_check_stream_sync\","
            "\"reference_checked\":true,\"model_load_ms\":" << model_load_ms
         << ",\"precision\":" << std::quoted(config_precision)
         << ",\"expert_activation\":" << std::quoted(config_expert_activation ? "gelu" : "silu")
         << ",\"numerical_mode\":" << std::quoted(config_numerical_mode ? "staged" : "fused")
         << ",\"async_staging_enabled\":" << (config_async_staging ? "true" : "false")
         << ",\"host_profile\":" << (weights.configs.at(0)[14] ? "true" : "false")
         << ",\"case_wall_ms\":" << elapsed_ms(wall_start)
         << ",\"process_peak_rss_bytes\":" << peak_rss_kib() * 1024
         << ",\"rss_scope\":\"Linux lifetime high-water\",\"manifest\":";
  vector_json(locked_manifest(weights));
  rusage process_after{};
  require(getrusage(RUSAGE_SELF, &process_after) == 0, "getrusage failed");
  const auto process_cpu_ms = [](const rusage& value) {
    return 1000.0 * (value.ru_utime.tv_sec + value.ru_stime.tv_sec) +
           .001 * (value.ru_utime.tv_usec + value.ru_stime.tv_usec);
  };
  report << ",\"case_cpu_ms\":" << process_cpu_ms(process_after) - process_cpu_ms(process_before)
         << ",\"cxx_abi\":" << _GLIBCXX_USE_CXX11_ABI << ",\"affinity\":[";
  bool first_cpu = true;
  for (int n = 0; n < CPU_SETSIZE; ++n) if (CPU_ISSET(n, &affinity)) {
    report << (first_cpu ? "" : ",") << n;
    first_cpu = false;
  }
  report << "],\"cpu_controls\":{";
  bool first_control = true;
  for (const char* path : {"/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat",
                          "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"}) {
    std::ifstream input(path);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    // std::quoted does not escape newlines for JSON.
    std::replace(text.begin(), text.end(), '\n', ' ');
    report << (first_control ? "" : ",") << std::quoted(path) << ':';
    if (input.is_open()) report << std::quoted(text); else report << "null";
    first_control = false;
  }
  report << "},\"timing_kind\":" << std::quoted(weights.configs.at(0)[14] ? "diagnostic_host_profile" : "normal_wall");
  report << ",\"initialization\":"; latency_json(initialization);
  for (const auto& entry : {std::make_pair("cold", &cold), std::make_pair("warm", &warm)}) {
    std::vector<double> times;
    for (const auto& sample : *entry.second) times.push_back(sample.ms);
    report << "," << std::quoted(entry.first) << ':'; latency_json(times);
  }
  report << ",\"raw\":{\"initialization_ms\":"; vector_json(initialization);
  report << ",\"initialization_stats_values\":[";
  for (size_t n = 0; n < initialization_stats.size(); ++n) {
    if (n) report << ',';
    vector_json(initialization_stats[n]);
  }
  report << "],\"cold\":"; samples_json(cold);
  report << ",\"priming\":"; samples_json(priming);
  report << ",\"warm\":"; samples_json(warm);
  report << "}}";
  if (!output_directory.empty()) {
    std::ofstream file(output_directory / "locked_benchmark.json");
    require(file.good(), "Cannot open locked benchmark evidence");
    file << report.str() << '\n';
  }
  std::cout << "BENCHMARK " << report.str() << std::endl;
}

// T03 independent lifecycle proof: no packages, Python frames, GPU reference
// matmul, or allocator substitutions. CPU snapshots outlive no CUDA output.
int64_t native_live_bytes(int64_t device) {
  return c10::cuda::CUDACachingAllocator::getDeviceStats(device).allocated_bytes[0].current;
}
void native_sync() { C10_CUDA_CHECK(cudaDeviceSynchronize()); }

template <class F>
void native_expect_error(F&& function, const std::string& diagnostic) {
  bool rejected = false;
  try { function(); }
  catch (const std::exception& error) {
    rejected = true;
    require(std::string(error.what()).find(diagnostic) != std::string::npos,
            "NATIVE unexpected rejection: " + std::string(error.what()));
  }
  require(rejected, "NATIVE expected rejection: " + diagnostic);
}

std::vector<int64_t> native_stats(PublicRuntime& runtime) {
  auto tensor = call("recsys_ops::fuse_moe_offload_stats", {runtime.token()})[0].toTensor();
  require(tensor.device().is_cpu() && tensor.scalar_type() == at::kLong &&
              tensor.numel() == stats_fields, "NATIVE stats schema mismatch");
  const auto* data = tensor.const_data_ptr<int64_t>();
  return {data, data + stats_fields};
}

void native_close_checked(PublicRuntime& runtime) {
  auto token = runtime.token(); // CPU token only; keep it to probe registry removal.
  runtime.close();
  require(runtime.closed(), "NATIVE close did not set adapter state");
  runtime.close();
  native_expect_error([&] { runtime.status_tensor(); }, "Runtime is closed");
  native_expect_error([&] { call("recsys_ops::fuse_moe_offload_status", {token}); },
                      "unknown");
  native_expect_error([&] { call("recsys_ops::fuse_moe_offload_check", {token}); },
                      "unknown");
}

void native_faults(int64_t device) {
  // Standalone ATen has no Python _lazy_init; initialize its official CUDA hooks
  // before querying allocator stats (otherwise no device stats table exists).
  at::globalContext().lazyInitDevice(at::kCUDA);
  for (const char* hook : {"recsys_ops::_fuse_moe_offload_test_inject",
                           "recsys_ops::_fuse_moe_offload_test_worker_states"}) {
    require(c10::Dispatcher::singleton().findSchema({hook, ""}).has_value(),
            std::string("--native-faults requires a testing library; missing hook ") + hook);
  }
  {
    auto states = call("recsys_ops::_fuse_moe_offload_test_worker_states", {})[0].toTensor();
    require(states.device().is_cpu() && states.scalar_type() == at::kLong &&
                states.sizes() == at::IntArrayRef({10}) &&
                at::equal(states, at::tensor({1, 1, 1, 1, 1, 1, 1, 1, 14, 5},
                                            at::TensorOptions().dtype(at::kLong))),
            "NATIVE queued/running/ready drain/shutdown/capture/TLS counts mismatch");
    std::cout << "NATIVE_WORKER_STATES [1,1,1,1,1,1,1,1,14,5]" << std::endl;
  }
  const int64_t initial_bytes = native_live_bytes(device);
  require(initial_bytes == 0, "NATIVE process already owns CUDA tensors");
  int cases = 0, concurrent_cases = 0, healthy_calls = 0;
  {
    const auto cpu = at::TensorOptions().device(at::kCPU);
    const auto gpu = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device));
    // Populate every element before copying; no random CUDA generator or cuBLAS.
    auto w1 = at::empty({E, 2 * I, H}, cpu.dtype(at::kHalf));
    auto w2 = at::empty({E, H, I}, cpu.dtype(at::kHalf));
    auto x_cpu = at::empty({35, H}, cpu.dtype(at::kHalf));
    auto ids_cpu = at::empty({35, K}, cpu.dtype(at::kInt));
    auto scales_cpu = at::empty({35, K}, cpu.dtype(at::kFloat));
    for (int64_t n = 0; n < w1.numel(); ++n)
      w1.data_ptr<at::Half>()[n] = at::Half(float(n % 29 - 14) / 256);
    for (int64_t n = 0; n < w2.numel(); ++n)
      w2.data_ptr<at::Half>()[n] = at::Half(float(n % 23 - 11) / 256);
    for (int64_t n = 0; n < x_cpu.numel(); ++n)
      x_cpu.data_ptr<at::Half>()[n] = at::Half(float(n % 31 - 15) / 64);
    for (int64_t n = 0; n < ids_cpu.numel(); ++n) {
      ids_cpu.data_ptr<int32_t>()[n] = n % E;
      scales_cpu.data_ptr<float>()[n] = n % K ? -.25f : .75f;
    }
    const auto w1_snapshot = w1.clone(), w2_snapshot = w2.clone();
    const auto x_snapshot = x_cpu.clone(), ids_snapshot = ids_cpu.clone();
    const auto scales_snapshot = scales_cpu.clone();
    custom_ops::offload::WeightMap weights{{0, {w1, w2}}};
    auto x = x_cpu.to(gpu), ids = ids_cpu.to(gpu), scales = scales_cpu.to(gpu);
    native_sync();
    const int64_t inputs_bytes = native_live_bytes(device);
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, static_cast<int>(device)));
    const std::vector<PublicBinding> bindings{{1, PublicBinding::GateUp, 0},
                                             {2, PublicBinding::Down, 0},
                                             {5, PublicBinding::Token}};
    const std::vector<std::string> faults{"gpu_pool_allocate", "pinned_allocate",
        "thread_create", "worker_running", "after_first_h2d", "after_event_record",
        "after_kernel_submit", "after_worker_ready", "before_queued_drain"};
    for (bool async : {false, true}) {
      // Exactly one expert per wave: pinned 50000 fits two banks, not capacity 2.
      const std::vector<int64_t> config{1, gpu_budget, ram_budget, 50000, 2, 0,
          4096, 0, 0, 0, 32, 2, 0, int64_t(async), 0, 0, 128,
          E, H, I, 0, 0, properties.major * 10 + properties.minor};
      std::vector<int64_t> manifest{1, 1, 0, int64_t(config.size())};
      manifest.insert(manifest.end(), config.begin(), config.end());
      const custom_ops::offload::Callable forward = [config, manifest](const std::vector<at::Tensor>& bound) {
        return std::vector<at::Tensor>{call("recsys_ops::fuse_moe_offload",
            {bound[0], bound[1], bound[2], bound[3], bound[4], bound[5], int64_t(0),
             c10::IValue(config), c10::IValue(manifest)})[0].toTensor()};
      };
      const auto raw = std::vector<at::Tensor>{x, {}, {}, ids, scales, {}};
      {
        PublicRuntime healthy(device, weights, manifest);
        auto healthy_inputs = healthy.prepare(raw, bindings);
        at::Tensor expected_cpu;
        auto verify_healthy = [&] {
          const auto before = public_status(healthy);
          {
            auto output = healthy.invoke(forward, healthy_inputs, bindings);
            native_sync();
            require(output.size() == 1 && output[0].sizes() == x.sizes() &&
                        output[0].scalar_type() == at::kHalf && output[0].device() == gpu,
                    "NATIVE healthy output metadata mismatch");
            auto actual = output[0].to(at::kCPU);
            if (!expected_cpu.defined()) expected_cpu = actual;
            else require(at::equal(actual, expected_cpu), "NATIVE healthy baseline changed");
          }
          native_sync();
          const auto after = public_status(healthy);
          require(after == std::vector<int64_t>({1, 0, 0, 0, before[4] + 1,
                      before[4] + 1, -1, 0, before[8] + 1, 0}),
                  "NATIVE healthy ten-field progression mismatch");
          const auto stats = native_stats(healthy);
          require(stats[0] == E && stats[1] == E && stats[7] == 2 && stats[8] == 1 &&
                      stats[5] <= 50000 && stats[4] <= gpu_budget,
                  "NATIVE expected all-four-experts/capacity-one/two-bank plan");
        };
        verify_healthy();
        ++healthy_calls;
        const int64_t held_bytes = native_live_bytes(device);
        const auto held_stats = native_stats(healthy);
        // Callback-only exceptions also end managed scopes. A completed native
        // output is destroyed during stack unwinding, never published or retained.
        {
          struct CallbackFailure : std::runtime_error { using std::runtime_error::runtime_error; };
          bool caught = false;
          try {
            healthy.invoke([&](const std::vector<at::Tensor>& bound) -> std::vector<at::Tensor> {
              auto outputs = forward(bound);
              throw CallbackFailure("native test business callback failure");
            }, healthy_inputs, bindings);
          } catch (const CallbackFailure&) { caught = true; }
          require(caught && public_status(healthy)[2] == 0 && public_status(healthy)[1] == 0,
                  "NATIVE business callback failure did not finish check");
          native_sync();
          require(native_live_bytes(device) == held_bytes, "NATIVE callback retained CUDA output");
        }
        for (const auto& fault : faults) {
          const bool allocation = fault == "gpu_pool_allocate" || fault == "pinned_allocate";
          const bool async_only = fault == "thread_create" || fault == "worker_running" ||
              fault == "after_worker_ready" || fault == "before_queued_drain";
          if (!async && async_only) continue;
          // Each fault uses a genuinely cold Runtime. The healthy context remains
          // alive and exercises a different caller compute stream concurrently.
          const auto healthy_before = public_status(healthy);
          std::promise<void> start;
          auto gate = start.get_future().share();
          auto independent = std::async(std::launch::async, [&, gate] {
            c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
            at::NoGradGuard no_grad;
            auto stream = c10::cuda::getStreamFromPool(false, device);
            c10::cuda::CUDAStreamGuard stream_guard(stream);
            gate.wait();
            for (int repeat = 0; repeat < 3; ++repeat) {
              auto output = healthy.invoke(forward, healthy_inputs, bindings);
              stream.synchronize();
              require(output.size() == 1 && at::equal(output[0].to(at::kCPU), expected_cpu),
                      "NATIVE independent context corrupted by concurrent fault");
            }
            stream.synchronize();
          });
          start.set_value(); // No future can block destruction on an unreleased gate.
          {
            PublicRuntime failed(device, weights, manifest);
            auto bound = failed.prepare(raw, bindings);
            require(public_status(failed) == std::vector<int64_t>({1, 0, 0, 0, 0, 0, -1, 0, 0, -1}),
                    "NATIVE fault context not cold");
            call("recsys_ops::_fuse_moe_offload_test_inject", {failed.token(), fault, int64_t(1)});
            bool caught = false, published = false;
            try {
              auto outputs = failed.invoke(forward, bound, bindings);
              published = true;
            } catch (const custom_ops::offload::InvocationError& error) {
              caught = true;
              require(error.native_error && error.callback_error,
                      "NATIVE invoke lost native or direct dispatcher exception");
              const auto diagnostic = fault == "thread_create" ? "injected staging thread creation" : fault;
              native_expect_error([&] { std::rethrow_exception(error.native_error); }, diagnostic);
              native_expect_error([&] { std::rethrow_exception(error.callback_error); }, diagnostic);
            }
            // The exception object/pointers are out of scope before any allocation snapshot.
            native_sync();
            const auto status = public_status(failed);
            require(caught && !published && status == std::vector<int64_t>({1,
                        allocation ? 1 : 2, 0, 1, 1, 0, 0, 1, 1, -1}),
                    "NATIVE fault status/check/output publication mismatch: " + fault);
            print_public_status(status, std::string("native-") + (async ? "async-" : "sync-") + fault);
            bool reached = false;
            native_expect_error([&] {
              failed.invoke([&](const std::vector<at::Tensor>& args) {
                reached = true; return forward(args);
              }, bound, bindings);
            }, allocation ? fault : "closed");
            require(!reached && public_status(failed) == status,
                    "NATIVE sticky failure changed/reentered callback");
            if (allocation) {
              failed.acknowledge();
              require(public_status(failed) == std::vector<int64_t>({1, 0, 0, 0, 1, 0, -1, 0, 1, -1}),
                      "NATIVE allocation acknowledge failed");
              {
                auto recovered = failed.invoke(forward, bound, bindings);
                native_sync();
                require(recovered.size() == 1 && at::equal(recovered[0].to(at::kCPU), expected_cpu),
                        "NATIVE acknowledged allocation recovery changed output");
              }
              require(public_status(failed) == std::vector<int64_t>({1, 0, 0, 0, 2, 2, -1, 0, 2, 0}),
                      "NATIVE recovery progression mismatch");
            } else {
              native_expect_error([&] { failed.acknowledge(); }, "poisoned");
              require(public_status(failed) == status, "NATIVE poison acknowledge changed status");
            }
            // All native work is complete BEFORE close, then registry must disappear.
            native_sync();
            native_close_checked(failed);
          }
          independent.get();
          native_sync();
          const auto healthy_after = public_status(healthy);
          require(healthy_after[4] == healthy_before[4] + 3 && healthy_after[5] == healthy_after[4] &&
                      healthy_after[8] == healthy_before[8] + 3 && healthy_after[1] == 0 &&
                      healthy_after[2] == 0 && healthy_after[3] == 0,
                  "NATIVE separate context status contaminated");
          const auto own_stats = native_stats(healthy);
          require(own_stats[14] == held_stats[14] && own_stats[15] == held_stats[15] &&
                      own_stats[19] == 0 && own_stats[22] == 0 &&
                      native_live_bytes(device) == held_bytes,
                  "NATIVE closed fault context retained live GPU allocations");
          verify_healthy();
          ++healthy_calls; ++cases; ++concurrent_cases;
          std::cout << "NATIVE_FAULT_PASS async=" << async << " fault=" << fault
                    << " live_bytes=" << held_bytes << " failed_context_registry=removed" << std::endl;
        }
        native_sync();
        native_close_checked(healthy);
      }
      native_sync();
      require(native_live_bytes(device) == inputs_bytes, "NATIVE healthy close retained GPU pool/output");
      require(at::equal(w1, w1_snapshot) && at::equal(w2, w2_snapshot) &&
                  at::equal(x_cpu, x_snapshot) && at::equal(ids_cpu, ids_snapshot) &&
                  at::equal(scales_cpu, scales_snapshot), "NATIVE immutable CPU fixtures changed");
    }
  } // All Runtime, Callable, futures, tensors (GPU AND CPU), snapshots destroyed.
  native_sync();
  require(native_live_bytes(device) == initial_bytes, "NATIVE final live tensor allocation leak");
  const auto before_trim = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  c10::cuda::CUDACachingAllocator::emptyCache();
  at::cuda::CachingHostAllocator_emptyCache();
  native_sync();
  const auto after_trim = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  require(after_trim.allocated_bytes[0].current == 0 && after_trim.active_bytes[0].current == 0 &&
              after_trim.reserved_bytes[0].current == 0, "NATIVE official allocator teardown incomplete");
  require(cases == 14 && concurrent_cases == 14, "NATIVE fault matrix incomplete");
  std::cout << "NATIVE_TEARDOWN {\"fault_cases\":" << cases
            << ",\"sync_fault_cases\":5,\"async_fault_cases\":9,\"worker_state_cases\":7"
            << ",\"concurrent_context_cases\":" << concurrent_cases
            << ",\"healthy_checks\":" << healthy_calls
            << ",\"concurrent_healthy_calls\":42,\"allocation_acknowledgements\":4"
            << ",\"poison_ack_rejections\":10,\"callback_output_unwind_cases\":2"
            << ",\"allocated_before_trim\":" << before_trim.allocated_bytes[0].current
            << ",\"reserved_before_trim\":" << before_trim.reserved_bytes[0].current
            << ",\"allocated_after_trim\":" << after_trim.allocated_bytes[0].current
            << ",\"reserved_after_trim\":" << after_trim.reserved_bytes[0].current
            << ",\"cuda_allocator_empty_cache\":true,\"pinned_allocator_empty_cache\":true}"
            << std::endl;
}

void assert_no_python_interpreter() {
  // libpython may be present solely to satisfy a pybind group's undefined symbols.
  using IsInitialized = int (*)();
  auto initialized = reinterpret_cast<IsInitialized>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
  require(!initialized || initialized() == 0, "Standalone runner initialized Python");
}
} // namespace

int run_main(int argc, char** argv) {
  try {
    const auto process_started = Clock::now();
    const std::string mode = argc >= 2 ? argv[1] : "";
    const std::set<std::string> modes{"--native-faults", "--locked-benchmark", "--public",
                                       "--locked", "--locked-chain", "--geglu", "--mlp"};
    require(mode.empty() || mode[0] != '-' || modes.count(mode), "Unknown runner mode: " + mode);
    const bool wide_benchmark = argc > 5 && std::string(argv[5]) == "--benchmark-wide";
    require(!(precision_explicit || expert_activation_explicit || numerical_mode_explicit) ||
                mode == "--geglu" || mode == "--mlp" || mode == "--locked-benchmark" || wide_benchmark,
            "--precision/--expert-activation/--numerical-mode require GEGLU or benchmark mode");
    for (int j = 2; j < argc; ++j) {
      const std::string option = argv[j];
      require(option.empty() || option[0] != '-' || option == "--negative-layer" ||
                  option == "--wide" || option == "--benchmark-wide",
              "Unknown runner argument: " + option);
    }
    if (mode == "--geglu" || mode == "--mlp") {
      require(argc >= 9, "Usage: runner --geglu model.pt2 ops.so device weights.pt S2 S7 S64 [S...] "
                         "[--precision fp16|bf16] [--async-staging] [--host-profile]");
      std::vector<int64_t> sizes;
      for (int j = 6; j < argc; ++j) {
        size_t consumed = 0;
        const std::string value = argv[j];
        const int64_t s = std::stoll(value, &consumed);
        require(consumed == value.size() && s >= 2 && s <= 64, "Invalid GEGLU S: " + value);
        sizes.push_back(s);
      }
      for (int64_t s : {2, 7, 64})
        require(std::find(sizes.begin(), sizes.end(), s) != sizes.end(), "GEGLU requires S=2,7,64");
      assert_no_python_interpreter();
      void* native = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
      if (!native) throw std::runtime_error(std::string("GEGLU ops dlopen failed: ") + dlerror());
      const int64_t device = std::stoll(argv[4]);
      require(device >= 0 && device <= std::numeric_limits<c10::DeviceIndex>::max(),
              "GEGLU device index out of range");
      c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
      at::NoGradGuard no_grad;
      check_geglu(argv[2], argv[5], device, sizes, mode == "--mlp" ? 1 : 0);
      assert_no_python_interpreter();
      std::cout << (mode == "--mlp" ?
                    "PASS MLP standalone frozen-manifest official AOTI strict_noJIT Python_uninitialized" :
                    "PASS GEGLU standalone frozen-manifest official AOTI strict_noJIT Python_uninitialized")
                << std::endl;
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--native-faults") {
      require(argc == 4, "Usage: runner --native-faults ops.so device [--output-dir path]");
      assert_no_python_interpreter();
      void* native = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
      if (!native) throw std::runtime_error(std::string("NATIVE ops dlopen failed: ") + dlerror());
      const int64_t device = std::stoll(argv[3]);
      require(device >= 0 && device <= std::numeric_limits<c10::DeviceIndex>::max(),
              "NATIVE device index out of range");
      {
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
        at::NoGradGuard no_grad;
        native_faults(device);
      }
      assert_no_python_interpreter();
      std::cout << "PASS NATIVE faults full allocator teardown Python_uninitialized" << std::endl;
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--locked-benchmark") {
      require(argc >= 11, "Usage: runner --locked-benchmark model.pt2 ops.so device weights.pt S K iterations cold_route warm_route... [--output-dir path]");
      assert_no_python_interpreter();
      void* native = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
      if (!native) throw std::runtime_error(std::string("Benchmark ops dlopen failed: ") + dlerror());
      const int64_t device = std::stoll(argv[4]);
      c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
      at::NoGradGuard no_grad;
      std::vector<std::string> schedule;
      for (int j = 10; j < argc; ++j) schedule.emplace_back(argv[j]);
      benchmark_locked(argv[2], argv[5], device, std::stoll(argv[6]), std::stoll(argv[7]),
                       std::stoll(argv[8]), argv[9], schedule);
      assert_no_python_interpreter();
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--public") {
      require(argc == 6 || (argc == 8 && std::string(argv[6]) == "--negative-layer"),
              "Usage: runner --public adapter.pt2 ops.so cuda_device weights.pt [--negative-layer index]");
      assert_no_python_interpreter();
      void* native = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
      if (!native) throw std::runtime_error(std::string("PUBLIC ops dlopen failed: ") + dlerror());
      const int64_t device = std::stoll(argv[4]);
      c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
      at::NoGradGuard no_grad;
      check_public(argv[2], argv[5], device, argc == 8 ? std::stoll(argv[7]) : -1);
      assert_no_python_interpreter();
      std::cout << "PASS PUBLIC standalone Python-free official AOTI" << std::endl;
      return 0;
    }
    if (argc >= 2 && (std::string(argv[1]) == "--locked" || std::string(argv[1]) == "--locked-chain")) {
      require(argc == 6, "Usage: runner --locked[-chain] model.pt2 ops.so cuda_device weights.pt");
      assert_no_python_interpreter();
      void* native = dlopen(argv[3], RTLD_NOW | RTLD_GLOBAL);
      if (!native) throw std::runtime_error(std::string("LOCKED ops dlopen failed: ") + dlerror());
      const int64_t device = std::stoll(argv[4]);
      c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
      at::NoGradGuard no_grad;
      check_locked(argv[2], argv[5], device, std::string(argv[1]) == "--locked-chain");
      assert_no_python_interpreter();
      std::cout << "PASS LOCKED standalone Python-free format2 AOTI" << std::endl;
      return 0;
    }
    // Common create-time CLI flags were validated and stripped in main for
    // every mode, including paths that return before legacy parsing.
    const bool benchmarking = argc >= 8 && std::string(argv[5]) == "--benchmark-wide";
    require(benchmarking || argc >= 9,
            "Usage: runner model.pt2 ops.so cuda_device weights.pt chain.pt2 S2 S7 S64 [S...] "
            "[--wide wide_fp16.pt2 wide_weights_fp16.pt]; or runner wide.pt2 ops.so cuda_device "
            "wide_weights.pt --benchmark-wide iterations S [S...]"
        " [--precision fp16|bf16] [--expert-activation silu|gelu] [--numerical-mode fused|staged]"
        " [--async-staging] [--host-profile]");
    int sizes_end = argc;
    std::string wide_package, wide_archive;
    for (int arg = 6; !benchmarking && arg < argc; ++arg) {
      if (std::string(argv[arg]) == "--wide") {
        require(arg >= 9 && arg + 3 == argc, "--wide requires exactly a package and archive after S values");
        sizes_end = arg;
        wide_package = argv[arg + 1];
        wide_archive = argv[arg + 2];
        break;
      }
    }
    assert_no_python_interpreter();
    // Never dlclose an operator registration library while dispatcher kernels live.
    void* native = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
    if (!native) {
      throw std::runtime_error(std::string("Native ops dlopen failed: ") + dlerror());
    }
    const int64_t device = std::stoll(argv[3]);
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    at::NoGradGuard no_grad;
    if (benchmarking) {
      std::vector<int64_t> sizes;
      for (int arg = 7; arg < argc; ++arg) sizes.push_back(std::stoll(argv[arg]));
      benchmark_wide(argv[1], argv[4], device, std::stoll(argv[6]), sizes);
      assert_no_python_interpreter();
      return 0;
    }
    Weights weights(argv[4]);
    const auto w1 = weights.layers.at(0).first, w2 = weights.layers.at(0).second;
    const auto w15 = weights.layers.at(5).first, w25 = weights.layers.at(5).second;
    // The official loader extracts the compiler-produced model and extern-kernel
    // JSON together, constructs RunnerCuda/ProxyExecutor, and owns cleanup.
    const auto model_started = Clock::now();
    torch::inductor::AOTIModelPackageLoader runner(argv[1], "model");
    torch::inductor::AOTIModelPackageLoader chain_runner(argv[5], "model");
    std::cout << "ACCEPTANCE_MODEL_LOAD {\"model_load_ms\":" << elapsed_ms(model_started)
              << ",\"startup_ms\":" << elapsed_ms(process_started)
              << ",\"process_peak_rss_kib\":" << peak_rss_kib()
              << ",\"rss_note\":\"Linux lifetime high-water; not exact budget\"}" << std::endl;
    for (int64_t mode : {int64_t(0), int64_t(1)}) {
      for (const std::string& preset : {std::string("disabled"), std::string("partial"),
                                        std::string("adjacent"), std::string("all")}) {
        const std::vector<int64_t> resident_ids = preset == "disabled" ? std::vector<int64_t>{}
            : preset == "partial" ? std::vector<int64_t>{0, 2}
            : preset == "adjacent" ? std::vector<int64_t>{0, 1}
                                  : std::vector<int64_t>{0, 1, 2, 3};
        Runtime runtime(device, w1, w2, mode, resident_ids, 0, 0, w15, w25);
        const std::string mode_label = " mode=" + std::to_string(mode) + " cache=" + preset;
        bool saw2 = false, saw7 = false, saw64 = false;
        for (int arg = 6; arg < sizes_end; ++arg) {
          int64_t s = std::stoll(argv[arg]);
          require(s >= 2 && s <= 64, "S is outside exported dynamic constraints");
          saw2 |= s == 2;
          saw7 |= s == 7;
          saw64 |= s == 64;
          auto inputs = make_inputs(s, device, w1, w2, runtime.token);
          // Validate the same context via native dispatcher, then the compiled model.
          // Destroy each output before the next call: live storage can force pool trim.
          for (int repeat = 0; repeat < 2; ++repeat) {
            const int64_t before = runtime.token[1].item<int64_t>();
            const int64_t id = runtime.token[0].item<int64_t>();
            const void* ptr = runtime.token.const_data_ptr();
            const int64_t version = runtime.token._version();
            auto out = call("recsys_ops::fuse_moe_offload",
                            {inputs[0], w1, w2, inputs[3], inputs[4], runtime.token, int64_t(0), c10::IValue(std::vector<int64_t>{}), c10::IValue(std::vector<int64_t>{})})[0]
                           .toTensor();
            check(inputs, out, runtime, before, id, ptr, version,
                  "C++ dispatcher" + mode_label + " S=" + std::to_string(s), repeat == 1);
          }
          for (int repeat = 0; repeat < 2; ++repeat) {
            const int64_t before = runtime.token[1].item<int64_t>();
            const int64_t id = runtime.token[0].item<int64_t>();
            const void* ptr = runtime.token.const_data_ptr();
            const int64_t version = runtime.token._version();
            auto outputs = runner.run(inputs);
            require(outputs.size() == 1, "Expected one flattened AOTI output");
            check(inputs, outputs[0], runtime, before, id, ptr, version,
                  "C++ AOTI" + mode_label + " S=" + std::to_string(s), repeat == 1);
          }
        }
        require(saw2 && saw7 && saw64, "Acceptance requires dynamic S=2, S=7 and S=64");
        runtime.close();
      }
    }
    check_dynamic_contexts(runner, device, w1, w2, w15, w25);
    check_chain(chain_runner, device, weights, sizes_end, argv);
    if (!wide_package.empty()) check_wide(wide_package, wide_archive, device);
    assert_no_python_interpreter();
    std::cout << "ACCEPTANCE_PROCESS {\"weight_version\":" << weights.version
              << ",\"process_wall_ms\":" << elapsed_ms(process_started)
              << ",\"process_peak_rss_kib\":" << peak_rss_kib()
              << ",\"rss_note\":\"Linux process lifetime high-water; not exact RAM budget\"}"
              << std::endl;
    std::cout << "PASS standalone C++ AOTI static + dynamic RAM offload + real two-layer chain (Python uninitialized)"
              << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL standalone C++ AOTI RAM offload: " << error.what() << std::endl;
    return 1;
  }
}

// Optional durable runner evidence. A killed/crashed process cannot finalize its
// own report: the Python parent's raw exit report records signal/returncode.
class EvidenceStream : public std::streambuf {
 public:
  EvidenceStream(std::streambuf* console, std::streambuf* archive)
      : console_(console), archive_(archive) {}
 protected:
  int_type overflow(int_type value) override {
    if (traits_type::eq_int_type(value, traits_type::eof())) return traits_type::not_eof(value);
    const std::lock_guard<std::mutex> lock(mutex_);
    const char c = traits_type::to_char_type(value);
    if (traits_type::eq_int_type(console_->sputc(c), traits_type::eof()) ||
        traits_type::eq_int_type(archive_->sputc(c), traits_type::eof())) return traits_type::eof();
    return value;
  }
  std::streamsize xsputn(const char* text, std::streamsize length) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    return std::min(console_->sputn(text, length), archive_->sputn(text, length));
  }
  int sync() override {
    const std::lock_guard<std::mutex> lock(mutex_);
    return console_->pubsync() == 0 && archive_->pubsync() == 0 ? 0 : -1;
  }
 private:
  std::streambuf* console_;
  std::streambuf* archive_;
  std::mutex mutex_;
};

int main(int argc, char** argv) {
  try {
    std::vector<char*> positional{argv[0]};
    for (int j = 1; j < argc; ++j) {
      const std::string option = argv[j];
      if (option == "--output-dir") {
        require(j + 1 < argc && output_directory.empty(), "--output-dir requires one unique path");
        output_directory = std::filesystem::absolute(argv[++j]);
      } else if (option == "--async-staging") {
        require(!async_staging_explicit, "Duplicate --async-staging");
        async_staging_explicit = config_async_staging = true;
      } else if (option == "--host-profile") {
        require(!host_profile_explicit, "Duplicate --host-profile");
        host_profile_explicit = config_host_profile = true;
      } else if (option == "--precision" || option == "--expert-activation" || option == "--numerical-mode") {
        require(j + 1 < argc, option + " requires a value");
        const std::string value = argv[++j];
        if (option == "--precision") {
          require(!precision_explicit && (value == "fp16" || value == "bf16"),
                  "--precision requires one unique fp16|bf16 value");
          precision_explicit = true;
          config_precision = value;
        } else if (option == "--expert-activation") {
          require(!expert_activation_explicit && (value == "silu" || value == "gelu"),
                  "--expert-activation requires one unique silu|gelu value");
          expert_activation_explicit = true;
          config_expert_activation = value == "gelu" ? 1 : 0;
        } else {
          require(!numerical_mode_explicit && (value == "fused" || value == "staged"),
                  "--numerical-mode requires one unique fused|staged value");
          numerical_mode_explicit = true;
          config_numerical_mode = value == "staged" ? 1 : 0;
        }
      } else positional.push_back(argv[j]);
    }
    const int positional_argc = static_cast<int>(positional.size());
    positional.push_back(nullptr);
    if (output_directory.empty()) return run_main(positional_argc, positional.data());
    std::filesystem::create_directories(output_directory);
    std::ofstream stdout_file(output_directory / "runner.stdout.txt");
    std::ofstream stderr_file(output_directory / "runner.stderr.txt");
    require(stdout_file.good() && stderr_file.good(), "Cannot open runner evidence streams");
    std::ostringstream arguments;
    for (int j = 0; j < argc; ++j) arguments << (j ? " " : "") << std::quoted(argv[j]);
    const auto write_exit = [&](const char* state, int code) {
      std::ofstream report(output_directory / "runner.exit.json");
      report << "{\"schema_version\":1,\"state\":" << std::quoted(state)
             << ",\"returncode\":" << code << ",\"command\":" << std::quoted(arguments.str()) << "}\n";
    };
    write_exit("running", -1);
    EvidenceStream stdout_tee(std::cout.rdbuf(), stdout_file.rdbuf());
    EvidenceStream stderr_tee(std::cerr.rdbuf(), stderr_file.rdbuf());
    auto* stdout_original = std::cout.rdbuf(&stdout_tee);
    auto* stderr_original = std::cerr.rdbuf(&stderr_tee);
    int code = 1;
    try { code = run_main(positional_argc, positional.data()); }
    catch (...) {
      std::cout.rdbuf(stdout_original);
      std::cerr.rdbuf(stderr_original);
      write_exit("failed", 1);
      throw;
    }
    std::cout.flush(); std::cerr.flush();
    std::cout.rdbuf(stdout_original);
    std::cerr.rdbuf(stderr_original);
    write_exit(code == 0 ? "passed" : "failed", code);
    return code;
  } catch (const std::exception& error) {
    std::cerr << "FAIL runner evidence: " << error.what() << std::endl;
    return 1;
  }
}
