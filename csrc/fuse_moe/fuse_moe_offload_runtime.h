#pragma once
// Public C++17, graph-external LibTorch adapter. Preload the registered extension
// and keep it loaded for this object's lifetime. No torch_python or loader copy.
// The caller supplies the official loader callback and the artifact's frozen
// manifest, and must bake that same manifest/layer IDs into the graph. Explicit
// input mapping does not authenticate arbitrary graphs or business weight releases.
// Compile AOTI artifacts with custom_ops.moe_offload.register_aoti_lowering():
// checked proxy returns stop the graph before downstream consumes a failed output.
// begin/check alone cannot repair an unchecked, previously compiled artifact.
// Native begin/check report call sequences and sticky/poisoned state;
// this header neither interprets status fields nor promises full-graph rollback or
// detection/recovery of every asynchronous CUDA error. No stable C ABI is promised.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/Storage.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace custom_ops::offload {
using WeightMap = std::map<int64_t, std::pair<at::Tensor, at::Tensor>>;
using Callable = std::function<std::vector<at::Tensor>(const std::vector<at::Tensor>&)>;

// Same argument order as Python InputBinding(index, kind, layer_id=None).
// Token has no layer; each weight binding must name an actual manifest layer.
struct InputBinding {
  enum Role { Token, GateUp, Down };
  int64_t index;
  Role kind;
  std::optional<int64_t> layer_id;
  InputBinding(int64_t index, Role kind, std::optional<int64_t> layer_id = std::nullopt)
      : index(index), kind(kind), layer_id(layer_id) {}
};

// Both original exceptions remain accessible when callback and native check fail.
// what() reports the native failure first; neither exception is replaced silently.
class InvocationError : public std::runtime_error {
 public:
  InvocationError(std::string message, std::exception_ptr native,
                  std::exception_ptr callback)
      : std::runtime_error(std::move(message)), native_error(std::move(native)),
        callback_error(std::move(callback)) {}
  const std::exception_ptr native_error;
  const std::exception_ptr callback_error;
};

namespace detail {
inline c10::OperatorHandle op(const char* name) {
  return c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
}
inline std::vector<c10::IValue> call(const char* name, std::vector<c10::IValue> args) {
  op(name).callBoxed(&args);
  return args;
}
inline at::Tensor tensor_result(const std::vector<c10::IValue>& result) {
  TORCH_CHECK(result.size() == 1 && result.front().isTensor(),
              "offload: expected one Tensor result from native operator");
  return result.front().toTensor();
}
inline void require_ops() {
  for (const char* name : {"recsys_ops::fuse_moe_offload_create",
       "recsys_ops::fuse_moe_offload_bind", "recsys_ops::fuse_moe_offload_resident_bind",
       "recsys_ops::fuse_moe_offload_validate", "recsys_ops::fuse_moe_offload_close",
       "recsys_ops::fuse_moe_offload_begin", "recsys_ops::fuse_moe_offload_check",
       "recsys_ops::fuse_moe_offload_status", "recsys_ops::fuse_moe_offload_acknowledge"})
    op(name);  // Missing schemas reject initialization before native create.
}

// Stack-linked TLS handles A -> B -> A reentry, not only immediate recursion.
// Every locking adapter entry checks BEFORE locking; different threads serialize.
struct Access {
  inline static thread_local Access* active = nullptr;
  const void* owner;
  Access* previous;
  std::unique_lock<std::mutex> lock;
  Access(const void* owner, std::mutex& mutex) : owner(owner), previous(active) {
    for (auto* entry = active; entry; entry = entry->previous)
      TORCH_CHECK(entry->owner != owner, "offload: same-thread Runtime reentry is unsupported");
    lock = std::unique_lock<std::mutex>(mutex);
    active = this;
  }
  ~Access() { active = previous; }
  Access(const Access&) = delete;
  Access& operator=(const Access&) = delete;
};

// Snapshot metadata separately: retaining a Tensor handle alone would not detect
// an external alias changing the same TensorImpl's storage/shape/strides.
struct Identity {
  c10::Storage storage;
  int64_t offset, version;
  std::vector<int64_t> sizes, strides;
  at::ScalarType dtype;
  c10::Device device;
  explicit Identity(const at::Tensor& t)
      : storage(t.storage()), offset(t.storage_offset()), version(t._version()),
        sizes(t.sizes().vec()), strides(t.strides().vec()), dtype(t.scalar_type()),
        device(t.device()) {}
  void check(const at::Tensor& t, bool check_version) const {
    TORCH_CHECK(t.defined() && t.layout() == at::kStrided && t.has_storage() &&
        t.storage().unsafeGetStorageImpl() == storage.unsafeGetStorageImpl() &&
        t.storage_offset() == offset && t.sizes() == at::IntArrayRef(sizes) &&
        t.strides() == at::IntArrayRef(strides) && t.scalar_type() == dtype &&
        t.device() == device, "offload: bound tensor identity/shape/stride/dtype changed");
    TORCH_CHECK(!check_version || (!t.is_inference() && t._version() == version),
                "offload: weight version changed; create a new Runtime");
  }
};

// v1 has 23 fixed fields and implies SiLU/fused (0/0); v2 has 25,
// with expert_activation/numerical_mode at 23/24 and resident IDs from 25.
// v3 has 26 fixed fields, adds expert_type at 25, and starts resident IDs at 26.
// v1/v2 imply GLU (type 0); native permits explicit v3 GLU as well as MLP.
// Native validate remains authoritative after binding (context, SM, residents).
inline size_t config_fixed_fields(const std::vector<int64_t>& c) {
  TORCH_CHECK(!c.empty() && (c[0] == 1 || c[0] == 2 || c[0] == 3),
              "offload: expected v1, v2 or v3 config");
  return c[0] == 1 ? 23 : (c[0] == 2 ? 25 : 26);
}
inline int64_t expert_activation(const std::vector<int64_t>& c) {
  return c[0] == 1 ? 0 : c[23];
}
inline int64_t numerical_mode(const std::vector<int64_t>& c) {
  return c[0] == 1 ? 0 : c[24];
}
inline int64_t expert_type(const std::vector<int64_t>& c) {
  return c[0] == 3 ? c[25] : 0;
}
inline void validate_config(const std::vector<int64_t>& c) {
  const auto fixed = config_fixed_fields(c);
  TORCH_CHECK(c.size() >= fixed, "offload: truncated config fixed fields");
  TORCH_CHECK((expert_activation(c) == 0 || expert_activation(c) == 1) &&
      (numerical_mode(c) == 0 || numerical_mode(c) == 1) &&
      (expert_type(c) == 0 || expert_type(c) == 1),
      "offload: invalid expert_activation/numerical_mode/expert_type enum");
  TORCH_CHECK(c[1] > 0 && c[2] > 0 && c[3] > 0 && (c[4] == 1 || c[4] == 2) &&
      c[5] >= 0 && c[5] <= 2 && c[6] >= 0 && c[6] <= INT_MAX &&
      c[7] >= 0 && c[7] <= c[1] && c[8] >= 0 && (c[9] == 0 || c[9] == 1) &&
      c[10] >= 2 && c[10] <= INT_MAX && c[11] >= 2 && c[11] <= INT_MAX &&
      c[12] >= 0 && (c[13] == 0 || c[13] == 1) && (c[14] == 0 || c[14] == 1) &&
      (c[15] == 0 || c[15] == 32 || c[15] == 64 || c[15] == 128) &&
      (c[16] == 64 || c[16] == 128), "offload: invalid budgets/context fields");
  TORCH_CHECK(c[17] > 0 && c[17] <= 512 && c[18] > 0 && c[18] <= INT_MAX &&
      c[18] % 64 == 0 && c[19] > 0 && c[19] <= INT_MAX / 2 && c[19] % 64 == 0 &&
      (c[20] == 0 || c[20] == 1) && c[21] >= 0 && c[21] <= c[17] &&
      c.size() - fixed == static_cast<size_t>(c[21]) && c[22] >= 80,
      "offload: invalid E/H/I/dtype/resident count/target SM");
  int64_t previous = -1;
  for (size_t j = fixed; j < c.size(); ++j) {
    TORCH_CHECK(c[j] > previous && c[j] < c[17], "offload: invalid sorted resident IDs");
    previous = c[j];
  }
}
inline std::map<int64_t, std::vector<int64_t>> parse(const std::vector<int64_t>& m) {
  TORCH_CHECK(m.size() >= 2 && m[0] == 1 && m[1] > 0 &&
      static_cast<uint64_t>(m[1]) <= (m.size() - 2) / 25,
      "offload: invalid v1 manifest header/count");
  std::map<int64_t, std::vector<int64_t>> configs;
  size_t offset = 2;
  int64_t previous = -1;
  for (int64_t n = 0; n < m[1]; ++n) {
    TORCH_CHECK(m.size() - offset >= 2, "offload: truncated layer header");
    const int64_t id = m[offset++], length = m[offset++];
    TORCH_CHECK(id > previous && length >= 23 &&
        static_cast<uint64_t>(length) <= m.size() - offset,
        "offload: unsorted/duplicate layer or invalid config length");
    previous = id;
    std::vector<int64_t> c(m.begin() + offset, m.begin() + offset + static_cast<size_t>(length));
    offset += static_cast<size_t>(length);
    validate_config(c);
    if (!configs.empty()) {
      const auto& first = configs.begin()->second;
      TORCH_CHECK(std::equal(c.begin() + 1, c.begin() + 17, first.begin() + 1) &&
          c[22] == first[22] && expert_activation(c) == expert_activation(first) &&
          numerical_mode(c) == numerical_mode(first) && expert_type(c) == expert_type(first),
                  "offload: inconsistent context configuration across layers");
    }
    configs.emplace(id, std::move(c));
  }
  TORCH_CHECK(offset == m.size(), "offload: trailing manifest fields");
  return configs;
}
inline int64_t multiply(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 && (!b || a <= std::numeric_limits<int64_t>::max() / b),
              "offload: weight size overflow");
  return a * b;
}
}  // namespace detail

class Runtime {
 public:
  Runtime(int64_t device, WeightMap weights, std::vector<int64_t> artifact_manifest)
      : device_(device), weights_(std::move(weights)), manifest_(std::move(artifact_manifest)) {
    detail::require_ops();
    const auto configs = detail::parse(manifest_);
    TORCH_CHECK(configs.size() == weights_.size(), "offload: manifest/weight inventory mismatch");
    std::set<c10::StorageImpl*> storages;
    int64_t backing = 0, resident = 0;
    const auto& first = configs.begin()->second;
    expert_activation_ = detail::expert_activation(first);
    numerical_mode_ = detail::numerical_mode(first);
    expert_type_ = detail::expert_type(first);
    for (const auto& entry : configs) {
      const auto it = weights_.find(entry.first);
      TORCH_CHECK(it != weights_.end(), "offload: manifest layer not bound: ", entry.first);
      const auto& c = entry.second;
      const auto& pair = it->second;
      for (const auto* t : {&pair.first, &pair.second}) {
        TORCH_CHECK(t->defined() && t->device().is_cpu() && t->layout() == at::kStrided &&
            t->is_contiguous() && !t->is_pinned() && !t->is_inference() && !t->requires_grad() &&
            t->scalar_type() == (c[20] == 0 ? at::kHalf : at::kBFloat16),
            "offload: expected canonical pageable CPU fp16/bf16 weights with version counters");
        if (storages.insert(t->storage().unsafeGetStorageImpl()).second) {
          const auto bytes = t->storage().nbytes();
          TORCH_CHECK(bytes <= static_cast<uint64_t>(first[2] - backing),
                      "offload: CPU backing exceeds RAM budget");
          backing += static_cast<int64_t>(bytes);
        }
      }
      const int64_t w1_rows = detail::multiply(c[19], detail::expert_type(c) == 0 ? 2 : 1);
      TORCH_CHECK(pair.first.sizes() == at::IntArrayRef({c[17], w1_rows, c[18]}) &&
          pair.second.sizes() == at::IntArrayRef({c[17], c[18], c[19]}),
          "offload: weight dimensions differ from manifest");
      const auto w1_bytes = detail::multiply(pair.first.numel() / c[17], pair.first.element_size());
      const auto w2_bytes = detail::multiply(pair.second.numel() / c[17], pair.second.element_size());
      TORCH_CHECK(w1_bytes <= std::numeric_limits<int64_t>::max() - w2_bytes,
                  "offload: weight size overflow");
      const auto bytes = w1_bytes + w2_bytes;
      TORCH_CHECK(bytes <= std::numeric_limits<int64_t>::max() - 511, "offload: size overflow");
      const auto charge = ((bytes + 511) / 512) * 512;
      TORCH_CHECK(c[21] <= (first[7] - resident) / charge,
                  "offload: static residents exceed cache budget");
      resident += c[21] * charge;
      identities_.emplace(entry.first, std::make_pair(detail::Identity(pair.first),
                                                     detail::Identity(pair.second)));
    }
    TORCH_CHECK(device >= 0 && device < c10::cuda::device_count(), "offload: invalid CUDA device");
    const auto* properties = at::cuda::getDeviceProperties(static_cast<c10::DeviceIndex>(device));
    TORCH_CHECK(properties && first[22] == int64_t(properties->major) * 10 + properties->minor,
                "offload: device target SM differs from artifact");
    std::vector<c10::IValue> args{device};
    for (size_t j = 1; j <= 16; ++j) {
      if (j == 13 || j == 14) args.emplace_back(bool(first[j]));
      else args.emplace_back(first[j]);
    }
    args.emplace_back(detail::expert_activation(first));
    args.emplace_back(detail::numerical_mode(first));
    args.emplace_back(detail::expert_type(first));
    // All 20 create arguments are explicit; frozen context/compute fields come
    // from the manifest. Boxed calls do not insert schema defaults; the extension
    // must be rebuilt with the 20-argument schema.
    token_ = detail::tensor_result(detail::call("recsys_ops::fuse_moe_offload_create", std::move(args)));
    try {
      TORCH_CHECK(token_.defined() && token_.device().is_cpu() && token_.scalar_type() == at::kLong &&
          token_.is_contiguous() && token_.sizes() == at::IntArrayRef({2}), "offload: invalid token");
      token_identity_.emplace(token_);
      detail::Access access(this, mutex_);
      for (const auto& entry : weights_)
        detail::call("recsys_ops::fuse_moe_offload_bind",
                     {token_, entry.first, entry.second.first, entry.second.second});
      for (const auto& entry : configs)
        if (entry.second[21])
          detail::call("recsys_ops::fuse_moe_offload_resident_bind", {token_, entry.first,
              c10::IValue(std::vector<int64_t>(
                  entry.second.begin() + detail::config_fixed_fields(entry.second), entry.second.end()))});
      validate_locked();  // Always uses the artifact manifest supplied by the caller.
    } catch (...) {
      cleanup();  // Constructor failure must not mask the original exception.
      throw;
    }
  }
  ~Runtime() noexcept { cleanup(); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

  // Handles are immutable in the adapter (including after close). External writes
  // through aliases are forbidden during invoke; snapshots catch tracked changes,
  // not unversioned memory writes. Native operators own token value/version changes.
  const at::Tensor& token() const noexcept { return token_; }
  const std::pair<at::Tensor, at::Tensor>& weights(int64_t layer) const { return weights_.at(layer); }
  const std::vector<int64_t>& manifest() const noexcept { return manifest_; }
  int64_t device() const noexcept { return device_; }
  // Frozen artifact contract: v1 implies SiLU/fused; v1/v2 imply GLU.
  int64_t expert_activation() const noexcept { return expert_activation_; }
  int64_t numerical_mode() const noexcept { return numerical_mode_; }
  int64_t expert_type() const noexcept { return expert_type_; }

  // Only designated placeholders are filled; arbitrary other arguments survive.
  std::vector<at::Tensor> prepare(std::vector<at::Tensor> inputs,
                                  const std::vector<InputBinding>& bindings) const {
    detail::Access access(this, mutex_);
    require_open();
    validate_store();
    for (const auto& binding : mapping(inputs.size(), bindings)) {
      auto& input = inputs[static_cast<size_t>(binding.first)];
      if (input.defined())
        binding.second.identity->check(input, binding.second.tensor != &token_);
      input = *binding.second.tensor;
    }
    return inputs;
  }
  std::vector<at::Tensor> invoke(const Callable& callable, const std::vector<at::Tensor>& inputs,
                                const std::vector<InputBinding>& bindings) {
    detail::Access access(this, mutex_);
    require_open();
    TORCH_CHECK(callable, "offload: empty loader callback");
    validate_store();
    for (const auto& binding : mapping(inputs.size(), bindings)) {
      const auto& actual = inputs[static_cast<size_t>(binding.first)];
      binding.second.identity->check(actual, binding.second.tensor != &token_);
    }
    validate_locked();
    detail::call("recsys_ops::fuse_moe_offload_begin", {token_});
    std::vector<at::Tensor> outputs;
    std::exception_ptr callback_error;
    try { outputs = callable(inputs); }
    catch (...) { callback_error = std::current_exception(); }
    try { detail::call("recsys_ops::fuse_moe_offload_check", {token_}); }
    catch (...) {
      if (!callback_error) throw;
      auto native_error = std::current_exception();
      std::string message = "offload: native check failed; loader callback also failed";
      try { std::rethrow_exception(native_error); }
      catch (const std::exception& e) { message = e.what(); }
      catch (...) {}
      throw InvocationError(std::move(message), std::move(native_error), std::move(callback_error));
    }
    if (callback_error) std::rethrow_exception(callback_error);
    return outputs;  // Never publish callback outputs before native check succeeds.
  }
  void validate() {
    detail::Access access(this, mutex_);
    require_open();
    validate_store();
    validate_locked();
  }
  at::Tensor status_tensor() const {
    detail::Access access(this, mutex_);
    require_open();
    return detail::tensor_result(detail::call("recsys_ops::fuse_moe_offload_status", {token_}));
  }
  void acknowledge() {
    detail::Access access(this, mutex_);
    require_open();
    detail::call("recsys_ops::fuse_moe_offload_acknowledge", {token_});
  }
  void close() {
    detail::Access access(this, mutex_);
    if (closed_) return;
    detail::call("recsys_ops::fuse_moe_offload_close", {token_});
    closed_ = true;  // Failed close stays observable and can be retried.
  }
  bool closed() const {
    detail::Access access(this, mutex_);
    return closed_;
  }

 private:
  void require_open() const { TORCH_CHECK(!closed_, "offload: Runtime is closed"); }
  void validate_locked() const {
    detail::call("recsys_ops::fuse_moe_offload_validate", {token_, c10::IValue(manifest_)});
  }
  void validate_store() const {
    token_identity_->check(token_, false);
    for (const auto& entry : weights_) {
      const auto& identity = identities_.at(entry.first);
      identity.first.check(entry.second.first, true);
      identity.second.check(entry.second.second, true);
    }
  }
  struct BoundInput {
    const at::Tensor* tensor;
    const detail::Identity* identity;
  };
  std::map<int64_t, BoundInput> mapping(
      size_t arity, const std::vector<InputBinding>& bindings) const {
    std::map<int64_t, BoundInput> slots;
    std::set<std::pair<int64_t, InputBinding::Role>> weights;
    size_t tokens = 0;
    for (const auto& b : bindings) {
      TORCH_CHECK(b.index >= 0 && static_cast<uint64_t>(b.index) < arity,
                  "offload: binding input index out of range");
      BoundInput input{nullptr, nullptr};
      if (b.kind == InputBinding::Token) {
        TORCH_CHECK(!b.layer_id && ++tokens == 1, "offload: exactly one shared token without layer ID required");
        input = {&token_, &*token_identity_};
      } else {
        TORCH_CHECK((b.kind == InputBinding::GateUp || b.kind == InputBinding::Down) &&
            b.layer_id && weights_.count(*b.layer_id), "offload: invalid weight role/layer binding");
        TORCH_CHECK(weights.emplace(*b.layer_id, b.kind).second, "offload: duplicate layer weight binding");
        const auto& pair = weights_.at(*b.layer_id);
        const auto& identity = identities_.at(*b.layer_id);
        input = b.kind == InputBinding::GateUp
            ? BoundInput{&pair.first, &identity.first} : BoundInput{&pair.second, &identity.second};
      }
      TORCH_CHECK(slots.emplace(b.index, input).second, "offload: duplicate input index");
    }
    TORCH_CHECK(tokens == 1 && weights.size() == 2 * weights_.size(),
                "offload: mapping requires one token and each manifest layer's w1/w2 exactly once");
    return slots;
  }
  void cleanup() noexcept {
    try { close(); }
    catch (const std::exception& e) { std::fprintf(stderr, "offload Runtime cleanup: %s\n", e.what()); }
    catch (...) { std::fprintf(stderr, "offload Runtime cleanup: unknown exception\n"); }
  }

  const int64_t device_;
  const WeightMap weights_;
  const std::vector<int64_t> manifest_;
  int64_t expert_activation_ = 0, numerical_mode_ = 0, expert_type_ = 0;  // Constructor-only writes.
  std::map<int64_t, std::pair<detail::Identity, detail::Identity>> identities_;
  at::Tensor token_;  // Assigned only during construction; never reset by close.
  std::optional<detail::Identity> token_identity_;
  mutable std::mutex mutex_;
  bool closed_ = false;  // Access only under mutex (no local sticky/poisoned mirror).
};
}  // namespace custom_ops::offload
