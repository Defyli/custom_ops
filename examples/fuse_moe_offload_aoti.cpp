// Standalone functional example for the companion single/two-layer artifacts.
// Build with examples/moe_offload_cpp; no test code or Python execution at runtime.
// Offline build (use a compiler matching your LibTorch/CUDA installation):
//   cmake -S examples/moe_offload_cpp -B artifacts/moe_cpp_build \
//     -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release
//   cmake --build artifacts/moe_cpp_build --parallel 2
// If the pybind native extension needs unresolved Py* symbols, configure with
//   -DMOE_OFFLOAD_LIBPYTHON=/absolute/path/to/libpython.so
// This supplies symbols only; the executable never initializes an interpreter.
// Run with the companion Python example's artifacts:
//   artifacts/moe_cpp_build/fuse_moe_offload_aoti ./native.so model.pt2 weights.pt 0 7
#include "fuse_moe_offload_runtime.h"
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using custom_ops::offload::InputBinding;
using custom_ops::offload::Runtime;
using custom_ops::offload::WeightMap;
using Loader = torch::inductor::AOTIModelPackageLoader;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

int64_t integer(const char* text, const char* name) {
  const std::string value(text);
  require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
          std::string(name) + " must be a nonnegative decimal integer");
  size_t used = 0;
  const auto result = std::stoll(value, &used);
  require(used == value.size(), std::string("Invalid ") + name);
  return result;
}

struct Archive {
  WeightMap weights;
  std::vector<int64_t> manifest;
  int64_t weight_version;

  explicit Archive(const std::string& path) {
    // Read buffers on CPU with the official reader; never execute forward().
    auto module = torch::jit::load(path, c10::Device(c10::kCPU));
    require(module.named_parameters().size() == 0, "Archive must not contain parameters");
    std::map<std::string, at::Tensor> buffers;
    for (const auto& buffer : module.named_buffers()) {
      require(buffers.emplace(buffer.name, buffer.value).second, "Duplicate archive buffer");
    }
    auto metadata = [&](const std::string& name) {
      const auto it = buffers.find(name);
      require(it != buffers.end(), "Missing archive metadata: " + name);
      const auto& t = it->second;
      require(t.device().is_cpu() && t.layout() == at::kStrided &&
                  t.scalar_type() == at::kLong && t.dim() == 1,
              "Invalid archive metadata: " + name);
      auto contiguous = t.contiguous();
      const auto* data = contiguous.const_data_ptr<int64_t>();
      std::vector<int64_t> values;
      if (contiguous.numel()) values.assign(data, data + contiguous.numel());
      return values;
    };
    require(metadata("format_version") == std::vector<int64_t>{2},
            "Expected ExpertStore.save_cpp format2 with frozen offload configs");
    const auto version = metadata("weight_version");
    require(version.size() == 1 && version[0] >= 0, "Invalid archive weight_version");
    weight_version = version[0];
    const auto layers = metadata("layer_ids");
    require(layers == std::vector<int64_t>{0} || layers == std::vector<int64_t>({0, 5}),
            "This example expects single layer 0 or sorted two layers 0,5");
    manifest = {1, static_cast<int64_t>(layers.size())};
    std::set<std::string> expected{"format_version", "weight_version", "layer_ids"};
    for (const auto layer : layers) {
      const auto suffix = std::to_string(layer);
      const auto w1 = "w1_" + suffix, w2 = "w2_" + suffix;
      const auto config_name = "offload_config_" + suffix;
      expected.insert(w1);
      expected.insert(w2);
      expected.insert(config_name);
      require(buffers.count(w1) && buffers.count(w2), "Missing layer weights: " + suffix);
      weights.emplace(layer, std::make_pair(buffers.at(w1), buffers.at(w2)));
      const auto config = metadata(config_name);
      manifest.push_back(layer);
      manifest.push_back(static_cast<int64_t>(config.size()));
      manifest.insert(manifest.end(), config.begin(), config.end());
    }
    require(buffers.size() == expected.size(), "Unexpected archive buffers");
    for (const auto& entry : buffers) {
      require(expected.count(entry.first) != 0, "Unexpected archive buffer: " + entry.first);
    }
    // Runtime is the authority for config/context/SM/budget/canonical-weight
    // validation. Do not duplicate its config parser or access its detail namespace.
    // The caller must pair this archive with the graph that baked these vectors:
    // neither metadata nor weight_version authenticates business weight identity.
  }
};

// 2.11 accepts an explicit device index; installed 2.6 uses (path, model_name).
// Keep the discarded constructor dependent so both headers compile correctly.
template <typename T = Loader>
std::unique_ptr<T> load_package(const std::string& path, c10::DeviceIndex device) {
  if constexpr (std::is_constructible_v<T, const std::string&, const std::string&,
                                         bool, size_t, c10::DeviceIndex>) {
    return std::make_unique<T>(path, "model", false, size_t{1}, device);
  } else {
    return std::make_unique<T>(path, "model");
  }
}

void synchronize() {
  const auto result = cudaDeviceSynchronize();
  require(result == cudaSuccess, std::string("CUDA synchronization failed: ") +
                                    cudaGetErrorString(result));
}

// Declared after loader/Runtime: drain pending work before either is destroyed,
// including on exceptions. Cleanup must not replace the original exception.
struct DrainOnExit {
  ~DrainOnExit() noexcept {
    const auto result = cudaDeviceSynchronize();
    if (result != cudaSuccess)
      std::cerr << "CUDA cleanup synchronization: " << cudaGetErrorString(result) << '\n';
  }
};

int run(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    std::cerr << "Usage: " << argv[0]
              << " NATIVE_SO PACKAGE_PT2 WEIGHTS_PT [DEVICE=0] [TOKENS=7]\n";
    return 2;
  }
  const int64_t device = argc >= 5 ? integer(argv[4], "DEVICE") : 0;
  const int64_t tokens = argc >= 6 ? integer(argv[5], "TOKENS") : 7;
  require(device <= std::numeric_limits<c10::DeviceIndex>::max(), "DEVICE index out of range");
  require(tokens >= 2 && tokens <= 64, "TOKENS must be in the exported range [2,64]");

  // Operator registration and compiled proxies outlive all local objects. Never
  // dlclose: dispatcher registrations may still refer to extension code at exit.
  void* native = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if (!native) throw std::runtime_error(std::string("Native extension dlopen failed: ") + dlerror());
  require(device < c10::cuda::device_count(), "CUDA device is unavailable");
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  at::NoGradGuard no_grad;  // Not InferenceMode: canonical weights need version counters.
  Archive archive(argv[3]);
  auto loader = load_package(argv[2], static_cast<c10::DeviceIndex>(device));
  Runtime runtime(device, archive.weights, archive.manifest);
  DrainOnExit drain;
  const auto dtype = runtime.weights(0).first.scalar_type();
  for (const auto& layer : archive.weights) {
    const auto& w1 = layer.second.first;
    const auto& w2 = layer.second.second;
    require(w1.size(0) == 8 && w1.size(2) == 128 &&
                w2.size(2) == (layer.first == 0 ? 128 : 192) &&
                w1.scalar_type() == dtype,
            "Companion graph requires E8/H128/I128(layer0),I192(layer5), shared dtype");
  }
  constexpr int64_t top_k = 2;
  const auto cuda = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device));
  const auto cpu_float = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto x = (at::arange(tokens * 128, cpu_float).cos() * 0.25)
               .reshape({tokens, 128}).to(cuda, dtype);
  auto ids = at::arange(tokens * top_k, at::TensorOptions().dtype(at::kLong))
                 .remainder(8).reshape({tokens, top_k}).to(cuda, at::kInt);
  auto scale = at::full({tokens, top_k}, 1.0 / top_k, cpu_float).to(cuda);

  // Single: x,w10,w20,ids,scale,token. Two: x,w10,w20,w15,w25,ids,scale,token.
  // Undefined slots are filled exclusively by public Runtime.prepare().
  std::vector<at::Tensor> inputs{x};
  std::vector<InputBinding> bindings;
  for (const auto& layer : archive.weights) {
    bindings.emplace_back(static_cast<int64_t>(inputs.size()), InputBinding::GateUp, layer.first);
    inputs.emplace_back();
    bindings.emplace_back(static_cast<int64_t>(inputs.size()), InputBinding::Down, layer.first);
    inputs.emplace_back();
  }
  inputs.push_back(ids);  // Both layers have E8, so sharing ids/scale is legal.
  inputs.push_back(scale);
  bindings.emplace_back(static_cast<int64_t>(inputs.size()), InputBinding::Token);
  inputs.emplace_back();
  inputs = runtime.prepare(std::move(inputs), bindings);
  auto outputs = runtime.invoke(
      [&](const std::vector<at::Tensor>& bound) { return loader->run(bound); }, inputs, bindings);
  synchronize();
  require(outputs.size() == 1 && outputs[0].defined(), "Expected graph output tuple (y,)");
  const auto& y = outputs[0];
  require(y.device() == cuda && y.sizes() == x.sizes() && y.scalar_type() == dtype,
          "Output shape/device/dtype mismatch");
  const auto host = y.to(at::kCPU).to(at::kFloat);
  require(at::isfinite(host).all().item<bool>(), "Output contains non-finite values");
  runtime.close();  // Explicit close exposes failures; destructor is the fallback.
  std::cout << "OK layers=" << archive.weights.size() << " device=" << device
            << " tokens=" << tokens << " output=[" << tokens << ",128] dtype="
            << c10::toString(dtype) << " weight_version=" << archive.weight_version
            << " mean=" << host.mean().item<float>()
            << " max_abs=" << host.abs().max().item<float>() << '\n';
  return 0;  // Functional smoke only, not a numerical-reference or performance test.
}
}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "MoE offload example failed: " << error.what() << '\n';
    return 1;
  }
}
