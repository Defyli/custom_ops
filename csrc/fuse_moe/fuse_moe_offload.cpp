#include "fuse_moe_offload.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/record_function.h>
#include <ATen/ThreadLocalState.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <charconv>
#include <climits>
#include <condition_variable>
#include <exception>
#include <functional>
#include <pthread.h>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <cerrno>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace {
using at::Tensor;

// Opt-in PyTorch observer scopes work with CPU profiling or emit_nvtx without
// adding a Python or NVTX link dependency to standalone C++ deployment.
// Disabled contexts never construct RecordFunction or consult callbacks.
struct HostStage {
    std::optional<at::RecordFunction> range;
    HostStage(bool enabled, const char* name) noexcept {
        if (!enabled) return;
        try {
            range.emplace(at::RecordScope::USER_SCOPE);
            if (range->isActive()) range->before(name);
        } catch (...) {
            // Diagnostic setup must not turn a valid inference into a failure.
            range.reset();
        }
    }
};
// Observer callbacks must not block by reentering their own locked context.
thread_local const void* active_context = nullptr;
struct ActiveContext {
    const void* previous = active_context;
    explicit ActiveContext(const void* context) { active_context = context; }
    ~ActiveContext() { active_context = previous; }
};

// One lazy, persistent producer, with at most one job per actual bank. Jobs
// borrow immutable weights/pool storage under the context's invocation lock.
// No tensor allocation or CUDA submission is performed by the producer.
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
enum class FaultPoint { GpuAllocate, PinnedAllocate, ThreadCreate, WorkerRunning,
    PartialH2D, EventRecord, KernelSubmit, WorkerReady, BeforeDrain, Count };
constexpr std::array<const char*, size_t(FaultPoint::Count)> fault_names{{
    "gpu_pool_allocate", "pinned_allocate", "thread_create", "worker_running",
    "after_first_h2d", "after_event_record", "after_kernel_submit",
    "after_worker_ready", "before_queued_drain"}};
struct Faults {
    std::mutex mutex;
    std::array<int64_t, size_t(FaultPoint::Count)> remaining{};
    void hit(FaultPoint point, bool* poison = nullptr) {
        bool fire = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto& count = remaining[size_t(point)];
            fire = count > 0 && --count == 0;
        }
        if (!fire) return;
        if (poison) *poison = true;
        if (point == FaultPoint::ThreadCreate)
            throw std::system_error(EAGAIN, std::generic_category(), "injected staging thread creation");
        TORCH_CHECK(false, "injected offload fault: ", fault_names[size_t(point)]);
    }
};
#define OFFLOAD_FAULT(c, point) (c).faults.hit(FaultPoint::point)
#define OFFLOAD_POISON_FAULT(c, point) (c).faults.hit(FaultPoint::point, &(c).closed)
#else
#define OFFLOAD_FAULT(c, point) ((void)0)
#define OFFLOAD_POISON_FAULT(c, point) ((void)0)
#endif

class StagingWorker {
    enum class State { Idle, Queued, Running, Ready };
    struct Slot {
        State state = State::Idle;
        std::function<void()> job;
        std::exception_ptr error;
    };
    std::array<Slot, 2> slots_;
    std::array<int, 2> queue_{};
    size_t head_ = 0, pending_ = 0;
    std::mutex mutex_;
    std::condition_variable changed_;
    pthread_t thread_{};
    bool started_ = false, stopping_ = false;
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
    Faults* faults_ = nullptr;
    bool test_hold_queue_ = false;
    std::function<void()> test_drain_observer_;
#endif

    static void* entry(void* self) noexcept {
        static_cast<StagingWorker*>(self)->run();
        return nullptr;
    }
    void run() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            changed_.wait(lock, [&] { return stopping_ || pending_; });
            if (stopping_) return;
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
            changed_.wait(lock, [&] { return stopping_ || !test_hold_queue_; });
            if (stopping_) return;
            if (!pending_) continue;
#endif
            const int bank = queue_[head_];
            head_ = (head_ + 1) % queue_.size();
            --pending_;
            auto& slot = slots_[bank];
            slot.state = State::Running;
            auto job = std::move(slot.job);
            lock.unlock();
            std::exception_ptr error;
            try { job(); } catch (...) { error = std::current_exception(); }
            // Drop diagnostic TLS/captures before publishing completion.
            job = nullptr;
            lock.lock();
            slot.error = std::move(error);
            slot.state = State::Ready;
            changed_.notify_all();
        }
    }
    void start() {
        if (started_) return;
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
        if (faults_) faults_->hit(FaultPoint::ThreadCreate);
#endif
        pthread_attr_t attr;
        int result = pthread_attr_init(&attr);
        if (result) throw std::system_error(result, std::generic_category(), "staging thread attributes");
        result = pthread_attr_setstacksize(&attr, stack_bytes());
        if (!result) result = pthread_attr_setguardsize(&attr, guard_bytes());
        if (!result) result = pthread_create(&thread_, &attr, entry, this);
        pthread_attr_destroy(&attr);
        if (result) throw std::system_error(result, std::generic_category(), "staging thread creation");
        started_ = true;
        // Best-effort OS diagnostic identity; no execution dependence on naming.
        pthread_setname_np(thread_, "moe_staging");
    }
public:
    // Explicit bounded stack, charged to RAM even before lazy startup. pthread
    // avoids an implementation-defined std::thread stack reservation. OS/libtorch
    // TLS and allocator RSS remain outside the measured buffer-accounting contract.
    static size_t stack_bytes() { return std::max<size_t>(1U << 20, PTHREAD_STACK_MIN); }
    static size_t guard_bytes() {
        const long page = sysconf(_SC_PAGESIZE);
        return page > 0 ? size_t(page) : 65536;
    }
    static int64_t ram_bytes() { return stack_bytes() + guard_bytes() + sizeof(StagingWorker) + 4096; }
    void submit(int bank, std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        TORCH_CHECK(!stopping_ && bank >= 0 && bank < 2 &&
            slots_[bank].state == State::Idle && pending_ < queue_.size(),
            "invalid staging bank lease");
        start();
        auto& slot = slots_[bank];
        slot.job = std::move(job);
        slot.state = State::Queued;
        queue_[(head_ + pending_) % queue_.size()] = bank;
        ++pending_;
        changed_.notify_all();
    }
    void wait(int bank) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& slot = slots_[bank];
        TORCH_CHECK(slot.state != State::Idle, "missing staging job");
        changed_.wait(lock, [&] { return slot.state == State::Ready; });
        auto error = std::move(slot.error);
        slot.error = nullptr;
        slot.state = State::Idle;
        lock.unlock();
        if (error) std::rethrow_exception(error);
    }
    // Cancel queued work, let the single running job finish, discard ready jobs.
    // Must precede releasing any borrowed weights, pool or per-call planning data.
    std::exception_ptr cancel_and_drain() noexcept {
        std::array<std::function<void()>, 2> discarded;
        std::array<std::exception_ptr, 2> errors;
        std::unique_lock<std::mutex> lock(mutex_);
        pending_ = 0;
        for (size_t bank = 0; bank < slots_.size(); ++bank) {
            auto& slot = slots_[bank];
            if (slot.state == State::Queued) {
                discarded[bank] = std::move(slot.job);
                slot.state = State::Idle;
            }
        }
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
        if (test_drain_observer_) test_drain_observer_();
#endif
        changed_.wait(lock, [&] {
            return std::none_of(slots_.begin(), slots_.end(),
                [](const Slot& slot) { return slot.state == State::Running; });
        });
        std::exception_ptr failed;
        for (size_t bank = 0; bank < slots_.size(); ++bank) {
            auto& slot = slots_[bank];
            if (!failed && slot.error) failed = slot.error;
            errors[bank] = std::move(slot.error);
            slot.error = nullptr;
            slot.state = State::Idle;
        }
        lock.unlock(); // Observer/TLS and exception destructors run outside the lock.
        return failed;
    }
    void shutdown() noexcept {
        cancel_and_drain();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            changed_.notify_all();
        }
        if (started_) { pthread_join(thread_, nullptr); started_ = false; }
    }
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
    void set_faults(Faults* faults) { faults_ = faults; }
    static Tensor exercise_states();
#endif
    ~StagingWorker() { shutdown(); }
};

#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
Tensor StagingWorker::exercise_states() {
    // CPU-only, deterministic: [version, queued drain, running drain, ready drain,
    // queued shutdown, running shutdown, ready shutdown, running error drain,
    // destroyed captures, destroyed TLS guards]. Values are counts, not timings.
    std::array<int64_t, 10> result{{1}};
    std::atomic<int64_t> destroyed{0}, tls_destroyed{0};
    struct Lease {
        std::atomic<int64_t>& count;
        explicit Lease(std::atomic<int64_t>& c) : count(c) {}
        ~Lease() { ++count; }
    };
    for (int test = 0; test < 7; ++test) {
        const int state = test == 6 ? 1 : test % 3;
        const bool shutdown = test >= 3 && test < 6;
        std::mutex gate_mutex;
        std::condition_variable gate_changed;
        bool entered = false, release = false;
        StagingWorker worker; // Destroy before gates captured by running jobs.
        worker.test_hold_queue_ = state == 0;
        // Drain itself releases the running job only after it observed Running.
        // No sleeps and no cancellation of executing work.
        worker.test_drain_observer_ = [&] {
            std::lock_guard<std::mutex> lock(gate_mutex);
            release = true;
            gate_changed.notify_all();
        };
        auto lease = std::make_shared<Lease>(destroyed);
        std::weak_ptr<Lease> weak = lease;
        at::ThreadLocalState tls;
        worker.submit(0, [&, lease, tls = std::move(tls)] {
            // Real PyTorch TLS install/restore plus an observable guard lifetime.
            Lease guard_lease(tls_destroyed);
            at::ThreadLocalStateGuard guard(tls);
            ActiveContext active(&worker);
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                entered = true;
                gate_changed.notify_all();
                if (state == 1) gate_changed.wait(lock, [&] { return release; });
            }
            if (test == 6) TORCH_CHECK(false, "injected running worker unit failure");
        });
        lease.reset();
        if (state == 1) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate_changed.wait(lock, [&] { return entered; });
        }
        {
            std::unique_lock<std::mutex> lock(worker.mutex_);
            if (state == 2)
                worker.changed_.wait(lock, [&] { return worker.slots_[0].state == State::Ready; });
            const auto expected = state == 0 ? State::Queued : state == 1 ? State::Running : State::Ready;
            TORCH_CHECK(worker.slots_[0].state == expected, "worker exercise state mismatch");
        }
        // Also retain a queued second-bank capture while the first bank is in
        // each tested state. Ready cannot race into bank 1: gate under slot lock.
        {
            std::lock_guard<std::mutex> lock(worker.mutex_);
            worker.test_hold_queue_ = true;
        }
        auto queued_lease = std::make_shared<Lease>(destroyed);
        std::weak_ptr<Lease> queued_weak = queued_lease;
        worker.submit(1, [queued_lease] {
            TORCH_CHECK(false, "queued exercise job must be cancelled, not executed");
        });
        queued_lease.reset();
        std::exception_ptr error;
        if (shutdown) worker.shutdown();
        else error = worker.cancel_and_drain();
        TORCH_CHECK(bool(error) == (test == 6), "worker exercise error propagation mismatch");
        TORCH_CHECK(weak.expired() && queued_weak.expired(),
            "worker retained a drained slot capture");
        {
            std::lock_guard<std::mutex> lock(worker.mutex_);
            TORCH_CHECK(worker.pending_ == 0 && std::all_of(worker.slots_.begin(), worker.slots_.end(),
                [](const Slot& slot) { return slot.state == State::Idle && !slot.job && !slot.error; }),
                "worker exercise drain retained a slot");
        }
        worker.shutdown();
        TORCH_CHECK(!worker.started_, "worker exercise shutdown did not join");
        ++result[1 + test];
    }
    result[8] = destroyed.load();
    result[9] = tls_destroyed.load();
    TORCH_CHECK(result[8] == 14 && result[9] == 5, "worker capture/TLS destruction mismatch");
    return at::tensor(at::IntArrayRef(result), at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}
#endif

bool cgroup_controller(const std::string& list, const std::string& name) {
    return ("," + list + ",").find("," + name + ",") != std::string::npos;
}
std::string mount_path(const std::string& field) {
    std::string path;
    for (size_t i = 0; i < field.size(); ++i) {
        if (field[i] == '\\' && i + 3 < field.size() &&
            field[i + 1] >= '0' && field[i + 1] <= '7' &&
            field[i + 2] >= '0' && field[i + 2] <= '7' &&
            field[i + 3] >= '0' && field[i + 3] <= '7') {
            path += char((field[i + 1] - '0') * 64 +
                         (field[i + 2] - '0') * 8 + field[i + 3] - '0');
            i += 3;
        } else {
            path += field[i];
        }
    }
    return path;
}
bool cgroup_path(const std::string& path) {
    if (path.empty() || path.front() != '/' || path.find('\0') != std::string::npos)
        return false;
    std::istringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/'))
        if (part == "." || part == "..") return false;
    return path == "/" || (path.back() != '/' && path.find("//") == std::string::npos);
}
bool cgroup_number(const std::string& path, uint64_t& value) {
    std::ifstream file(path);
    std::string text, extra;
    if (!(file >> text) || (file >> extra)) return false;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
}
void cgroup_remaining(const std::string& dir, bool v2, std::optional<int64_t>& remaining) {
    uint64_t limit, used;
    // v2 "max" is nonnumeric; v1's unlimited sentinel is LONG_MAX rounded to pages.
    if (!cgroup_number(dir + (v2 ? "/memory.max" : "/memory.limit_in_bytes"), limit) ||
        (!v2 && limit >= uint64_t(LONG_MAX) - (1ULL << 20)) ||
        !cgroup_number(dir + (v2 ? "/memory.current" : "/memory.usage_in_bytes"), used))
        return;
    int64_t bytes = static_cast<int64_t>(std::min<uint64_t>(
        used >= limit ? 0 : limit - used, INT64_MAX));
    remaining = remaining ? std::min(*remaining, bytes) : bytes;
}
int64_t auto_ram_budget() {
    std::optional<int64_t> available;
    long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages >= 0 && page > 0 && uint64_t(pages) <= uint64_t(INT64_MAX) / uint64_t(page))
        available = static_cast<int64_t>(pages) * static_cast<int64_t>(page);

    std::string v1_path, v2_path, line;
    std::ifstream groups("/proc/self/cgroup");
    while (std::getline(groups, line)) {
        auto first = line.find(':');
        if (first == std::string::npos) continue;
        auto second = line.find(':', first + 1);
        if (second == std::string::npos) continue;
        std::string controllers = line.substr(first + 1, second - first - 1);
        std::string path = line.substr(second + 1);
        if (!cgroup_path(path)) continue;
        if (line.substr(0, first) == "0" && controllers.empty()) v2_path = path;
        if (cgroup_controller(controllers, "memory")) v1_path = path;
    }
    std::optional<int64_t> remaining;
    std::ifstream mounts("/proc/self/mountinfo");
    while (std::getline(mounts, line)) {
        auto separator = line.find(" - ");
        if (separator == std::string::npos) continue;
        std::istringstream before(line.substr(0, separator)), after(line.substr(separator + 3));
        std::string id, parent, dev, root, mount, options, type, source, super;
        if (!(before >> id >> parent >> dev >> root >> mount >> options) ||
            !(after >> type >> source >> super)) continue;
        bool v2 = type == "cgroup2";
        if (!v2 && (type != "cgroup" || !cgroup_controller(super, "memory"))) continue;
        const std::string& path = v2 ? v2_path : v1_path;
        root = mount_path(root);
        mount = mount_path(mount);
        if (path.empty() || !cgroup_path(root) || !cgroup_path(mount)) continue;
        std::string suffix;
        if (root == "/") suffix = path == "/" ? "" : path;
        else if (path == root || path == "/") suffix = ""; // Namespace-relative root.
        else if (path.compare(0, root.size() + 1, root + "/") == 0) suffix = path.substr(root.size());
        else suffix = path; // cgroup namespaces may report paths relative to mountroot.
        std::string dir = (mount == "/" ? "" : mount) + suffix;
        if (dir.empty()) dir = "/";
        // Ancestors can impose a tighter quota; never walk above the visible mount.
        for (;;) {
            cgroup_remaining(dir == "/" ? "" : dir, v2, remaining);
            if (dir == mount) break;
            auto slash = dir.rfind('/');
            dir = slash == 0 ? "/" : dir.substr(0, slash);
        }
    }
    if (!remaining) {
        // Best effort when proc metadata is unavailable: only known standard roots.
        cgroup_remaining("/sys/fs/cgroup", true, remaining);
        cgroup_remaining("/sys/fs/cgroup/memory", false, remaining);
        cgroup_remaining("/sys/fs/cgroup", false, remaining);
    }
    if (remaining) available = available ? std::min(*available, *remaining) : *remaining;
    TORCH_CHECK(available, "cannot auto-detect RAM; supply ram_budget_bytes");
    return *available / 5 * 4 + (*available % 5) * 4 / 5;
}

int64_t mul(int64_t a, int64_t b) {
    TORCH_CHECK(a >= 0 && b >= 0 && (!b || a <= INT64_MAX / b), "offload size overflow");
    return a * b;
}
int64_t add(int64_t a, int64_t b) {
    TORCH_CHECK(a >= 0 && b >= 0 && a <= INT64_MAX - b, "offload size overflow");
    return a + b;
}
int64_t charge(int64_t n) {
    TORCH_CHECK(n <= INT64_MAX - 511, "offload size overflow");
    return n ? ((n + 511) / 512) * 512 : 0;
}
struct Heat {
    uint32_t frequency = 0;
    int64_t last_use = 0;
    bool dynamic = false;
};
struct Layer {
    Tensor w1, w2;
    uint32_t v1, v2;
    // Both fixed and dynamic entries directly address a complete w1+w2 expert.
    // Only dynamic entries may be evicted. Bound weights remain immutable.
    std::vector<Tensor> resident;
    std::vector<Heat> heat;
    int64_t calls = 0;
};
struct CacheActivity {
    int64_t admissions = 0, evictions = 0, bypasses = 0;
    int64_t promotion_bytes = 0, budget_evictions = 0;
};
struct Output {
    c10::weak_intrusive_ptr<c10::StorageImpl> storage;
    int64_t bytes;
};
// Persistent DMA resources. Each invocation drains before returning, so buffers
// can be reused on a different caller stream without relying on allocator events.
struct Transfers {
    int device;
    cudaStream_t copy = nullptr;
    std::vector<cudaEvent_t> ready, done;
    Transfers(int d, int banks) : device(d) {
        // Allocate host bookkeeping before creating CUDA resources: push_back
        // cannot then leak a just-created event if host allocation fails.
        ready.reserve(banks); done.reserve(banks);
        C10_CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        try {
            for (int b = 0; b < banks; ++b) {
                cudaEvent_t event;
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                ready.push_back(event);
                C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                done.push_back(event);
            }
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        if (copy) cudaStreamSynchronize(copy);
        for (auto event : ready) cudaEventDestroy(event);
        for (auto event : done) cudaEventDestroy(event);
        ready.clear(); done.clear();
        if (copy) cudaStreamDestroy(copy);
        copy = nullptr;
        cudaSetDevice(previous);
    }
    ~Transfers() { cleanup(); }
};
// Byte storage permits shape/dtype changes without accumulating per-layer pools.
// GPU charges include allocator alignment; pinned charges exclude allocator RSS.
enum Buffer { Routes, Counts, Down, Act, Accumulator, Scales, TileMeta,
    HostBank0, HostBank1, GpuBank0, GpuBank1, HostMeta0, HostMeta1,
    GpuMeta0, GpuMeta1, ResidentMeta, RouteSummary, BufferCount };
using Sizes = std::array<int64_t, BufferCount>;
bool host_buffer(int slot) {
    return slot == Counts || slot == HostBank0 || slot == HostBank1 ||
        slot == HostMeta0 || slot == HostMeta1;
}
struct Context {
    const int device;
    const int64_t gpu, ram, pinned, banks, mode, max_tile, cache_limit;
    int64_t cache_bytes = 0, cache_experts = 0, cache_load = 0;
    const int64_t dynamic_limit, policy, decay_interval, min_reuse, update_limit;
    int64_t dynamic_bytes = 0, dynamic_experts = 0;
    c10::weak_intrusive_ptr<c10::StorageImpl> token_storage;
    std::mutex mutex;
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
    Faults faults;
#endif
    bool closed = false; // Poisoned until explicit close removes the registry entry.
    bool managed = false;
    std::thread::id managed_owner;
    std::exception_ptr pending_error, drain_error;
    int64_t forward_sequence = 0, success_sequence = 0;
    int64_t failure_layer = -1, failure_sequence = 0;
    int64_t managed_sequence = 0, success_layer = -1;
    void record_error(std::exception_ptr error, int64_t layer, int64_t sequence) noexcept {
        if (!pending_error) {
            pending_error = std::move(error);
            failure_layer = layer;
            failure_sequence = sequence;
        }
    }
    void require_owner() const {
        TORCH_CHECK(!managed || managed_owner == std::this_thread::get_id(),
            "managed offload scope belongs to another thread");
    }
    bool sealed = false;
    bool seal_requires_manifest = false;
    const bool async_staging, host_profile;
    const int64_t tile_m, tile_n, target_sm;
    std::unordered_map<int64_t, Layer> layers;
    std::vector<Output> outputs;
    std::array<Tensor, BufferCount> pool;
    std::vector<int64_t> stats = std::vector<int64_t>(31, 0);
    Transfers transfers;
    // Destroy/join the producer before Transfers or pool/weights are destroyed.
    StagingWorker staging;
    Context(int d, int64_t g, int64_t r, int64_t p, int64_t b,
            int64_t m, int64_t tile, int64_t cache, int64_t dynamic,
            int64_t cache_policy, int64_t decay, int64_t reuse, int64_t update,
            bool async, bool profile, int64_t tm, int64_t tn, int64_t sm, const Tensor& t)
        : device(d), gpu(g), ram(r), pinned(p), banks(b), mode(m), max_tile(tile), cache_limit(cache),
          dynamic_limit(dynamic), policy(cache_policy), decay_interval(decay), min_reuse(reuse),
          update_limit(update), token_storage(t.storage().getWeakStorageImpl()),
          async_staging(async), host_profile(profile), tile_m(tm), tile_n(tn),
          target_sm(sm), transfers(d, b) {
#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
        staging.set_faults(&faults);
#endif
    }
};
// Version 1: [version, gpu, ram, pinned, banks, mode, max_tile, cache_limit,
// dynamic_limit, policy, decay_interval, min_reuse, update_limit, async, profile,
// tile_m, tile_n, E, H, I, dtype_code, resident_count, target_sm, sorted static IDs...].
// There are 23 fixed fields; target_sm is at 22 and static IDs start at 23.
// The half-open slice [1,17) matches immutable, resolved context values.
// Runtime device ordinal is deliberately absent: same-target replicas may use
// different local CUDA ordinals while sharing the graph's deployment contract.
void validate_deployment_config(at::IntArrayRef config, const Context& c,
                                const Layer& layer, const Tensor& w1, const Tensor& w2,
                                const char* format_error = "invalid deployment configuration format") {
    TORCH_CHECK(!c.sealed || !config.empty(),
        "sealed deployment context requires its compiled configuration");
    if (config.empty()) return;
    TORCH_CHECK(config.size() >= 23 && config[0] == 1,
        format_error, ": expected version 1 and 23 fixed fields");
    TORCH_CHECK(config[22] >= 80,
        format_error, ": target_sm must be SM80+");
    const int64_t experts = config[17], hidden = config[18], intermediate = config[19];
    const int64_t resident_count = config[21];
    TORCH_CHECK(config[1] > 0 && config[2] > 0 && config[3] > 0 &&
        (config[4] == 1 || config[4] == 2) && config[5] >= 0 && config[5] <= 2 &&
        config[6] >= 0 && config[6] <= INT_MAX && config[7] >= 0 && config[8] >= 0 &&
        (config[9] == 0 || config[9] == 1) &&
        config[10] >= 2 && config[10] <= INT_MAX &&
        config[11] >= 2 && config[11] <= INT_MAX && config[12] >= 0 &&
        (config[13] == 0 || config[13] == 1) && (config[14] == 0 || config[14] == 1) &&
        (config[15] == 0 || config[15] == 32 || config[15] == 64 || config[15] == 128) &&
        (config[16] == 64 || config[16] == 128),
        format_error, ": invalid budgets or context fields");
    TORCH_CHECK(experts > 0 && experts <= 512 && hidden > 0 && hidden <= INT_MAX &&
        hidden % 64 == 0 && intermediate > 0 && intermediate <= INT_MAX / 2 &&
        intermediate % 64 == 0 && (config[20] == 0 || config[20] == 1) &&
        resident_count >= 0 && resident_count <= experts &&
        config.size() - 23 == static_cast<size_t>(resident_count),
        format_error, ": invalid weights or resident count/length");
    int64_t previous = -1;
    for (size_t j = 23; j < config.size(); ++j) {
        TORCH_CHECK(config[j] > previous && config[j] < experts,
            format_error, ": static resident IDs must be sorted and unique in [0,E)");
        previous = config[j];
    }
    const std::array<int64_t, 16> expected{{c.gpu, c.ram, c.pinned, c.banks,
        c.mode, c.max_tile, c.cache_limit, c.dynamic_limit, c.policy, c.decay_interval,
        c.min_reuse, c.update_limit, int64_t(c.async_staging), int64_t(c.host_profile),
        c.tile_m, c.tile_n}};
    for (size_t j = 0; j < expected.size(); ++j)
        TORCH_CHECK(config[j + 1] == expected[j], "deployment configuration mismatch: context field ", j + 1);
    TORCH_CHECK(config[22] == c.target_sm, "deployment configuration mismatch: target_sm");
    auto matches_weights = [&](const Tensor& a, const Tensor& b) {
        return a.dim() == 3 && b.dim() == 3 && a.size(0) == experts && b.size(0) == experts &&
            a.size(1) == 2 * intermediate && a.size(2) == hidden &&
            b.size(1) == hidden && b.size(2) == intermediate &&
            a.scalar_type() == (config[20] == 0 ? at::kHalf : at::kBFloat16) &&
            b.scalar_type() == a.scalar_type();
    };
    TORCH_CHECK(matches_weights(w1, w2) && matches_weights(layer.w1, layer.w2),
        "deployment configuration mismatch: weight dimensions or dtype");
    size_t next = 23;
    for (size_t expert = 0; expert < layer.resident.size(); ++expert) {
        if (!layer.resident[expert].defined() ||
            (!layer.heat.empty() && layer.heat[expert].dynamic)) continue;
        TORCH_CHECK(next < config.size() && config[next] == static_cast<int64_t>(expert),
            "deployment configuration mismatch: static resident expert IDs");
        ++next;
    }
    TORCH_CHECK(next == config.size(), "deployment configuration mismatch: static resident expert IDs");
}

// Version 1 manifest: [1, layer_count, layer_id, config_length, *config,
// ...]. A sorted, complete inventory lets every graph invocation validate all
// layers before any token/cache mutation, allocation, or GPU submission.
void validate_deployment_manifest(at::IntArrayRef manifest, at::IntArrayRef config,
                                  const Context& c, int64_t current_layer) {
    TORCH_CHECK(!c.seal_requires_manifest || !manifest.empty(),
        "sealed deployment context requires its compiled manifest");
    if (manifest.empty()) return;
    TORCH_CHECK(manifest.size() >= 2 && manifest[0] == 1 && manifest[1] >= 0,
        "invalid deployment manifest format: expected version 1 and nonnegative layer count");
    const int64_t count = manifest[1];
    // Every record contains at least two header fields plus 23 config fields.
    TORCH_CHECK(static_cast<uint64_t>(count) <= (manifest.size() - 2) / 25,
        "invalid deployment manifest format: truncated layer records");
    TORCH_CHECK(static_cast<uint64_t>(count) == c.layers.size(),
        "deployment configuration mismatch: manifest must include every bound layer");
    size_t offset = 2;
    int64_t previous = -1;
    bool found_current = false;
    for (int64_t record = 0; record < count; ++record) {
        TORCH_CHECK(manifest.size() - offset >= 2,
            "invalid deployment manifest format: truncated layer header");
        const int64_t id = manifest[offset++];
        const int64_t length = manifest[offset++];
        TORCH_CHECK(id > previous,
            "invalid deployment manifest format: layer IDs must be sorted, unique and nonnegative");
        previous = id;
        TORCH_CHECK(length >= 23 && static_cast<uint64_t>(length) <= manifest.size() - offset,
            "invalid deployment manifest format: invalid config length");
        const auto segment = manifest.slice(offset, static_cast<size_t>(length));
        offset += static_cast<size_t>(length);
        const auto bound = c.layers.find(id);
        TORCH_CHECK(bound != c.layers.end(),
            "deployment configuration mismatch: manifest layer is not bound");
        const auto& layer = bound->second;
        validate_deployment_config(segment, c, layer, layer.w1, layer.w2,
            "invalid deployment manifest format");
        TORCH_CHECK(layer.w1._version() == layer.v1 && layer.w2._version() == layer.v2,
            "deployment configuration mismatch: weight version changed for layer ", id);
        if (id == current_layer) {
            TORCH_CHECK(config.size() == segment.size() &&
                std::equal(config.begin(), config.end(), segment.begin()),
                "deployment configuration mismatch: current layer config differs from manifest");
            found_current = true;
        }
    }
    TORCH_CHECK(offset == manifest.size(),
        "invalid deployment manifest format: trailing fields");
    TORCH_CHECK(found_current,
        "deployment configuration mismatch: current layer is missing from manifest");
}

int64_t pool_bytes(const Sizes& sizes, bool host) {
    int64_t bytes = 0;
    for (int slot = 0; slot < BufferCount; ++slot)
        if (host_buffer(slot) == host)
            bytes = add(bytes, host ? sizes[slot] : charge(sizes[slot]));
    return bytes;
}
Sizes pool_sizes(const Context& c) {
    Sizes sizes{};
    for (int slot = 0; slot < BufferCount; ++slot)
        if (c.pool[slot].defined()) sizes[slot] = c.pool[slot].numel();
    return sizes;
}
int64_t total_cache_bytes(const Context& c) {
    return add(c.cache_bytes, c.dynamic_bytes);
}
int64_t cache_metadata_bytes(const Context& c) {
    if (!c.dynamic_limit) return 0;
    int64_t bytes = 0;
    for (const auto& kv : c.layers) {
        // Heat plus bounded victim descriptors and per-layer bookkeeping. Like
        // host planning, this charge is not a promise about allocator/RSS overhead.
        bytes = add(bytes, add(64, mul(kv.second.heat.size(), sizeof(Heat) + 64)));
    }
    return bytes;
}
bool fits(const Context& c, const Sizes& sizes, int64_t outputs,
          int64_t backing, int64_t host_plan, int64_t reclaimed = 0) {
    return add(add(pool_bytes(sizes, false), outputs),
        total_cache_bytes(c) - reclaimed) <= c.gpu &&
        pool_bytes(sizes, true) <= c.pinned &&
        add(add(backing, pool_bytes(sizes, true)), host_plan) <= c.ram;
}
struct Victim {
    int64_t layer_id;
    int expert;
    Layer* layer;
    int64_t bytes;
};
int64_t expert_size(const Layer& layer) {
    return mul(mul(layer.w1.size(2), layer.w2.size(2)), 6);
}
uint32_t observed_frequency(const Layer& layer, int expert, const Layer* current,
                            const int* counts) {
    uint32_t f = layer.heat[expert].frequency;
    return f + (current == &layer && counts && counts[expert] > 0 && f < UINT32_MAX);
}
std::vector<Victim> cache_victims(Context& c, const Layer* current = nullptr,
        const int* counts = nullptr, const std::vector<bool>* consumed = nullptr) {
    std::vector<Victim> victims;
    for (auto& kv : c.layers) {
        auto& layer = kv.second;
        for (int e = 0; e < int(layer.heat.size()); ++e) {
            if (!layer.heat[e].dynamic) continue;
            // Planned resident waves hold a lease until their final kernel has
            // read the entry. Never evict a future wave's weight pointer.
            if (&layer == current && consumed && counts[e] && !(*consumed)[e]) continue;
            victims.push_back({kv.first, e, &layer, charge(expert_size(layer))});
        }
    }
    std::sort(victims.begin(), victims.end(), [&](const Victim& a, const Victim& b) {
        if (!c.policy) {
            auto fa = observed_frequency(*a.layer, a.expert, current, counts);
            auto fb = observed_frequency(*b.layer, b.expert, current, counts);
            // Transmission cost is proportional to bytes in this first version;
            // benefit/byte therefore orders by recent demand frequency.
            if (fa != fb) return fa < fb;
        }
        const auto& ha = a.layer->heat[a.expert];
        const auto& hb = b.layer->heat[b.expert];
        if (ha.last_use != hb.last_use) return ha.last_use < hb.last_use;
        return a.layer_id != b.layer_id ? a.layer_id < b.layer_id : a.expert < b.expert;
    });
    return victims;
}
void evict(Context& c, const Victim& v, CacheActivity& activity, bool budget) {
    // Caller has drained both streams. No D2H: canonical RAM is immutable.
    v.layer->resident[v.expert] = Tensor();
    v.layer->heat[v.expert].dynamic = false;
    c.dynamic_bytes -= v.bytes;
    --c.dynamic_experts;
    ++activity.evictions;
    activity.budget_evictions += budget;
}
struct Allocations {
    int64_t count = 0;
    std::array<bool, BufferCount> reused{};
};
void configure_pool(Context& c, const Sizes& required, int64_t outputs,
                    int64_t backing, int64_t host_plan, Allocations& allocations,
                    int64_t gpu_reserve = 0) {
    Sizes old = pool_sizes(c), target = old;
    for (int slot = 0; slot < BufferCount; ++slot)
        target[slot] = std::max(target[slot], required[slot]);
    // Routing is prepared before wave planning. Keep its exact request size so
    // the second configuration cannot trim or invalidate populated metadata.
    target[Routes] = required[Routes];
    target[Counts] = required[Counts];
    // Retain high-water capacities only if *all* budgets, including live outputs,
    // allow it. Trim before growing to avoid a transient old+new allocation peak.
    if (!fits(c, target, add(outputs, gpu_reserve), backing, host_plan)) target = required;
    TORCH_CHECK(fits(c, target, outputs, backing, host_plan),
        "pool/history exceeds gpu_budget_bytes or RAM/pinned budgets");
    for (int slot = 0; slot < BufferCount; ++slot)
        if (target[slot] != old[slot]) c.pool[slot] = Tensor();
    auto options = at::TensorOptions().dtype(at::kByte);
    for (int slot = 0; slot < BufferCount; ++slot) {
        if (!target[slot]) continue;
        if (!c.pool[slot].defined()) {
            c.pool[slot] = at::empty({target[slot]}, host_buffer(slot)
                ? options.device(at::kCPU).pinned_memory(true)
                : options.device(at::Device(at::kCUDA, c.device)));
            if (host_buffer(slot)) { OFFLOAD_FAULT(c, PinnedAllocate); }
            else { OFFLOAD_FAULT(c, GpuAllocate); }
            ++allocations.count;
        } else if (required[slot]) allocations.reused[slot] = true;
    }
}
template <typename T> T* pointer(Context& c, int slot) {
    return reinterpret_cast<T*>(c.pool[slot].data_ptr());
}
std::mutex registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<Context>> registry;
std::atomic<int64_t> next_id{1};
std::shared_ptr<Context> lookup(const Tensor& token) {
    TORCH_CHECK(token.device().is_cpu() && token.scalar_type() == at::kLong &&
        token.is_contiguous() && token.dim() == 1 && token.numel() == 2,
        "runtime_token must be contiguous CPU int64[2]");
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(token.data_ptr<int64_t>()[0]);
    TORCH_CHECK(it != registry.end(), "unknown or closed offload context; create a deployment token");
    TORCH_CHECK(active_context != it->second.get(),
        "reentrant offload context access from an observer is not supported");
    auto storage = it->second->token_storage.lock();
    TORCH_CHECK(storage && storage.get() == token.storage().unsafeGetStorageImpl(),
        "runtime_token identity mismatch (tokens cannot be cloned or serialized)");
    return it->second;
}
void weights(const Tensor& w1, const Tensor& w2) {
    TORCH_CHECK(w1.device().is_cpu() && w2.device().is_cpu(), "offload weights must remain on CPU");
    TORCH_CHECK(w1.dim() == 3 && w2.dim() == 3 && w1.is_contiguous() && w2.is_contiguous(),
        "offload weights must be canonical contiguous (E,2I,H)/(E,H,I)");
    TORCH_CHECK((w1.scalar_type() == at::kHalf || w1.scalar_type() == at::kBFloat16) &&
        w2.scalar_type() == w1.scalar_type(), "offload requires matching fp16/bf16 weights");
    TORCH_CHECK(w1.size(0) > 0 && w1.size(0) <= 512 && w2.size(0) == w1.size(0) &&
        w2.size(1) == w1.size(2) && w1.size(1) == 2 * w2.size(2) &&
        w1.size(2) > 0 && w2.size(2) > 0 && w1.size(2) % 64 == 0 && w2.size(2) % 64 == 0 &&
        w1.size(2) <= INT_MAX && w2.size(2) <= INT_MAX / 2,
        "invalid E/H/I or unsupported alignment (E<=512, H/I multiples of 64)");
    TORCH_CHECK(!w1.is_pinned() && !w2.is_pinned(),
        "canonical weights must be pageable RAM; pinned_budget_bytes covers staging only");
    TORCH_CHECK(!w1.is_inference() && !w2.is_inference(),
        "weights require version counters; create canonical weights outside inference_mode");
    TORCH_CHECK(!w1.requires_grad() && !w2.requires_grad(), "offload is inference-only");
}
int64_t ram_weights(const Context& c) {
    std::unordered_map<c10::StorageImpl*, int64_t> storages;
    for (const auto& kv : c.layers) {
        for (const Tensor* w : {&kv.second.w1, &kv.second.w2})
            storages.emplace(w->storage().unsafeGetStorageImpl(), w->storage().nbytes());
    }
    int64_t bytes = c.async_staging ? StagingWorker::ram_bytes() : 0;
    for (const auto& kv : storages) {
        TORCH_CHECK(kv.second <= INT64_MAX - bytes, "RAM size overflow");
        bytes += kv.second;
    }
    return bytes;
}
// Scope drains both streams even on validation/allocation/launch errors. No buffer
// may be freed while the copy stream or an expert GEMM still references it.
struct Flight {
    cudaStream_t compute, copy;
    bool& failed;
    bool drained = false;
    StagingWorker* staging;
    Context& context;
    explicit Flight(cudaStream_t s, Context& c, StagingWorker* worker = nullptr)
        : compute(s), copy(c.transfers.copy), failed(c.closed), staging(worker), context(c) {}
    void finish() {
        if (staging) {
            OFFLOAD_POISON_FAULT(context, BeforeDrain);
            auto error = staging->cancel_and_drain();
            if (error) {
                failed = true;
                std::rethrow_exception(error);
            }
        }
        auto copy_error = cudaStreamSynchronize(copy);
        auto compute_error = cudaStreamSynchronize(compute);
        drained = true;
        if (copy_error != cudaSuccess || compute_error != cudaSuccess) failed = true;
        C10_CUDA_CHECK(copy_error);
        C10_CUDA_CHECK(compute_error);
    }
    ~Flight() {
        // Also runs after finish(): planning captures cannot escape the call.
        if (staging) {
            auto error = staging->cancel_and_drain();
            if (error) {
                failed = true;
                // Outer catch records the initiating exception, not this secondary
                // drain failure. Retain it until then without allocating bookkeeping.
                if (!context.drain_error) context.drain_error = std::move(error);
            }
        }
        if (drained) return;
        if (cudaStreamSynchronize(copy) != cudaSuccess) failed = true;
        if (cudaStreamSynchronize(compute) != cudaSuccess) failed = true;
    }
};
bool promote(Context& c, Layer& layer, int expert, const uint8_t* gate_up,
        const uint8_t* down, const int* counts, const std::vector<bool>& consumed,
        int64_t outputs, cudaStream_t stream, CacheActivity& activity, int64_t timestamp) {
    if (!c.dynamic_limit) return false;
    const int64_t bytes = expert_size(layer), charged = charge(bytes);
    const int64_t limit = c.update_limit ? c.update_limit : c.dynamic_limit;
    const uint32_t frequency = observed_frequency(layer, expert, &layer, counts);
    if (charged > c.dynamic_limit || bytes > limit - activity.promotion_bytes ||
        (!c.policy && frequency < uint64_t(c.min_reuse))) return false;
    const int64_t base = add(add(outputs, pool_bytes(pool_sizes(c), false)), c.cache_bytes);
    // Admission never shrinks the current execution's banks/tiles or trims its
    // pool. Dynamic storage must fit in actual headroom after those allocations.
    if (base > c.gpu || charged > c.gpu - base) return false;
    const int64_t target = std::min(c.dynamic_limit, c.gpu - base);
    auto victims = cache_victims(c, &layer, counts, &consumed);
    int64_t freed = 0;
    size_t used = 0;
    long double victim_benefit = 0;
    while (c.dynamic_bytes - freed > target - charged && used < victims.size()) {
        const auto& v = victims[used++];
        freed = add(freed, v.bytes);
        victim_benefit += static_cast<long double>(observed_frequency(
            *v.layer, v.expert, &layer, counts)) * expert_size(*v.layer);
    }
    if (c.dynamic_bytes - freed > target - charged) return false;
    if (!c.policy && used && static_cast<long double>(frequency) * bytes <=
        victim_benefit + static_cast<long double>(bytes) / 4) return false;
    // Slow-path admission only: drain before freeing victims, and keep the source
    // bank leased until D2D completes. Ordinary bypass waves remain pipelined.
    Tensor candidate;
    Flight publication(stream, c);
    C10_CUDA_CHECK(cudaStreamSynchronize(c.transfers.copy));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t j = 0; j < used; ++j) evict(c, victims[j], activity, false);
    c10::InferenceMode normal_tensors(false);
    try {
        candidate = at::empty({bytes}, at::TensorOptions().dtype(at::kByte)
            .device(at::Device(at::kCUDA, c.device)));
    } catch (const c10::OutOfMemoryError&) {
        // Cache allocation is optional. Already loaded demand still computes.
        return false;
    }
    // Admission already drains: also complete the new allocation itself before
    // a different stream writes it (required by stream-ordered allocators).
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t gate_bytes = mul(layer.w1[0].numel(), 2);
    auto* dst = candidate.data_ptr<uint8_t>();
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, gate_up, gate_bytes, cudaMemcpyDeviceToDevice, publication.copy));
    C10_CUDA_CHECK(cudaMemcpyAsync(dst + gate_bytes, down, bytes - gate_bytes,
        cudaMemcpyDeviceToDevice, publication.copy));
    publication.finish();
    layer.resident[expert] = std::move(candidate);
    layer.heat[expert].dynamic = true;
    layer.heat[expert].last_use = timestamp;
    c.dynamic_bytes = add(c.dynamic_bytes, charged);
    ++c.dynamic_experts;
    ++activity.admissions;
    activity.promotion_bytes = add(activity.promotion_bytes, bytes);
    return true;
}
void commit_heat(Context& c, Layer& layer, const int* counts) {
    if (!c.dynamic_limit) return;
    // Successful invocations only, once per distinct expert, never per token or
    // route tile. Each layer has its own sample clock to avoid layer-order bias.
    for (int e = 0; e < int(layer.heat.size()); ++e)
        layer.heat[e].frequency = observed_frequency(layer, e, &layer, counts);
    if (++layer.calls == c.decay_interval) {
        for (auto& heat : layer.heat) heat.frequency /= 2;
        layer.calls = 0;
    }
}
struct Wave { std::vector<int> experts; int begin, rows; bool resident; };
void prepare_wave(const Wave& wave, const int* counts, int capacity, int tile_m,
        uint8_t* host, int* meta, const uint8_t* w1, const uint8_t* w2,
        int64_t w1_bytes, int64_t w2_bytes, bool profile) {
    const int groups = wave.experts.size();
    {
        HostStage stage(profile, "fuse_moe_offload::host_weight_copy");
        for (int g = 0; g < groups; ++g) {
            const int expert = wave.experts[g];
            std::memcpy(host + g * w1_bytes, w1 + expert * w1_bytes, w1_bytes);
            std::memcpy(host + capacity * w1_bytes + g * w2_bytes,
                w2 + expert * w2_bytes, w2_bytes);
        }
    }
    {
        HostStage stage(profile, "fuse_moe_offload::host_metadata");
        int* lens = meta; int* prefix = lens + groups; int* tiles = prefix + groups + 1;
        int* cu_tiles = tiles + groups;
        prefix[0] = 0; cu_tiles[0] = 0;
        for (int g = 0; g < groups; ++g) {
            lens[g] = counts[wave.experts[g]];
            prefix[g + 1] = prefix[g] + lens[g];
            tiles[g] = (lens[g] + tile_m - 1) / tile_m;
            cu_tiles[g + 1] = cu_tiles[g] + tiles[g];
        }
    }
}
std::vector<Wave> plan(const int* counts, int e, int capacity, const Layer& layer,
                      int resident_capacity, const std::vector<bool>* excluded = nullptr) {
    std::vector<Wave> result;
    int begin = 0;
    for (int i = 0; i < e; ++i) {
        if (!counts[i]) continue;
        const bool hit = layer.resident[i].defined() &&
            !(excluded && (*excluded)[i]);
        if (result.empty() || result.back().resident != hit ||
            int(result.back().experts.size()) == (hit ? resident_capacity : capacity))
            result.push_back({{}, begin, 0, hit});
        result.back().experts.push_back(i);
        result.back().rows += counts[i];
        begin += counts[i];
    }
    return result;
}
struct Execution {
    bool valid = false;
    int mode = 0, banks = 0, capacity = 0, rows = 0;
    int64_t peak = 0, cache_reserve = 0;
    Sizes sizes{};
    std::vector<Wave> waves;
};
Execution execution_plan(const Context& c, const int* counts, int experts,
    int active, int hits, const Layer& layer, int64_t s, int64_t h, int64_t i, int64_t t,
    const Sizes& routing, int64_t outputs, int64_t backing, int64_t host_plan,
    int mode, int tile_m, int64_t reclaimed = 0, const std::vector<bool>* excluded = nullptr) {
    Execution result;
    result.mode = mode;
    const int misses = active - hits;
    // Auto tiling must not spend all otherwise available admission headroom on
    // Act. Reserve at most one eligible expert, and only within the current
    // bank/group floor. Full caches are preserved, not evicted for larger tiles.
    int64_t reserve = 0;
    if (mode && !c.max_tile && misses && c.dynamic_limit) {
        const int64_t bytes = expert_size(layer), charged = charge(bytes);
        const int64_t limit = c.update_limit ? c.update_limit : c.dynamic_limit;
        if (charged <= c.dynamic_limit - (c.dynamic_bytes - reclaimed) && bytes <= limit) {
            for (int e = 0; e < experts; ++e) {
                const bool hit = layer.resident[e].defined() && !(excluded && (*excluded)[e]);
                if (counts[e] && !hit && (c.policy ||
                    observed_frequency(layer, e, &layer, counts) >= uint64_t(c.min_reuse))) {
                    reserve = charged;
                    break;
                }
            }
        }
    }
    auto buffers = [&](int banks, int capacity, int resident_groups, int rows) {
        Sizes sizes = routing;
        // Fused GEMM2 consumes rounded values directly: no Down tile write/read.
        sizes[Down] = mode ? 0 : mul(mul(t, h), 2);
        sizes[Act] = mul(mul(rows, i), 2);
        if (mode) {
            sizes[Accumulator] = mul(mul(s, h), 4);
            sizes[Scales] = mul(t, 4);
            if (active) sizes[TileMeta] = mul(4 * std::max({capacity, resident_groups, 1}) + 2, 4);
        }
        if (hits) {
            sizes[ResidentMeta] = mul(4 * resident_groups + 2, 4);
            if (resident_groups > 1)
                sizes[ResidentMeta] = add(sizes[ResidentMeta], mul(2 * resident_groups, sizeof(void*)));
        }
        for (int b = 0; b < banks; ++b) {
            sizes[HostBank0 + b] = sizes[GpuBank0 + b] = mul(mul(mul(capacity, h), i), 6);
            sizes[HostMeta0 + b] = sizes[GpuMeta0 + b] = mul(4 * capacity + 2, 4);
        }
        return sizes;
    };
    if (!active) {
        result.sizes = buffers(0, 0, 0, 0);
        result.valid = fits(c, result.sizes, outputs, backing, host_plan, reclaimed);
        result.peak = add(add(outputs, pool_bytes(result.sizes, false)), total_cache_bytes(c) - reclaimed);
        return result;
    }
    // Prefer double banking and the largest feasible expert group. A loaded
    // weight bank serves all its route tiles; tiling never repeats weight H2D.
    for (int banks = misses ? c.banks : 0; banks >= (misses ? 1 : 0); --banks) {
        for (int capacity = misses; capacity >= (misses ? 1 : 0); --capacity) {
            // Reject impossible bank/pinned/RAM floors before building up to
            // 32 resident alternatives; this lower bound cannot reject a fit.
            if (!fits(c, buffers(banks, capacity, hits ? 1 : 0, 1),
                      outputs, backing, host_plan, reclaimed)) continue;
            // Cache grouping is optional: shrink it before sacrificing miss-bank
            // capacity or evicting entries. Group=1 has the original memory floor.
            for (int resident_capacity = std::min(hits, kOffloadResidentGroups);
                 resident_capacity >= (hits ? 1 : 0); --resident_capacity) {
                auto waves = plan(counts, experts, capacity, layer, resident_capacity, excluded);
                int rows = 0, resident_groups = 0;
                for (const auto& wave : waves) {
                    rows = std::max(rows, wave.rows);
                    if (wave.resident)
                        resident_groups = std::max(resident_groups, int(wave.experts.size()));
                }
                int64_t headroom = 0;
                if (mode) {
                    if (c.max_tile) rows = std::min<int64_t>(rows, c.max_tile);
                    if (reserve && fits(c, buffers(banks, capacity, resident_groups, 1),
                                        add(outputs, reserve), backing, host_plan, reclaimed))
                        headroom = reserve;
                    // Monotone, alignment-aware search for the largest feasible tile.
                    int low = 1, high = rows, best = 0;
                    while (low <= high) {
                        int mid = low + (high - low) / 2;
                        if (fits(c, buffers(banks, capacity, resident_groups, mid),
                                 add(outputs, headroom), backing, host_plan, reclaimed)) {
                            best = mid; low = mid + 1;
                        } else high = mid - 1;
                    }
                    rows = best;
                    if (!rows) continue;
                    if (!c.max_tile) {
                        // All waves share Act. On the tile-count plateau, avoid
                        // spending bytes on rows that eliminate no kernel launches.
                        int minimum = 1;
                        for (const auto& wave : waves) {
                            const int tiles = 1 + (wave.rows - 1) / best;
                            minimum = std::max(minimum, 1 + (wave.rows - 1) / tiles);
                        }
                        const int64_t aligned = (int64_t(minimum) + tile_m - 1) / tile_m * tile_m;
                        rows = aligned <= best ? int(aligned) : minimum;
                    }
                }
                auto sizes = buffers(banks, capacity, resident_groups, rows);
                if (!fits(c, sizes, outputs, backing, host_plan, reclaimed)) continue;
                result.valid = true;
                result.banks = banks; result.capacity = capacity; result.rows = rows;
                result.sizes = sizes; result.waves = std::move(waves);
                result.cache_reserve = headroom;
                result.peak = add(add(outputs, pool_bytes(sizes, false)), total_cache_bytes(c) - reclaimed);
                return result;
            }
        }
    }
    return result;
}
}

Tensor fuse_moe_offload_create(int64_t device, int64_t gpu, int64_t ram,
                              int64_t pinned, int64_t banks, int64_t mode, int64_t max_tile,
                              int64_t cache_limit, int64_t dynamic_limit, int64_t policy,
                              int64_t decay_interval, int64_t min_reuse, int64_t update_limit,
                              bool async_staging, bool host_profile, int64_t tile_m, int64_t tile_n) {
    TORCH_CHECK(device >= 0 && device < at::cuda::device_count(), "invalid CUDA device");
    TORCH_CHECK(gpu >= 0 && ram >= 0 && pinned >= 0 && (banks == 1 || banks == 2),
        "budgets must be nonnegative; num_banks must be 1 or 2");
    TORCH_CHECK(mode >= 0 && mode <= 2 && max_tile >= 0 && max_tile <= INT_MAX,
        "activation_mode must be 0/1/2; max_route_tile_rows must be in [0,INT_MAX] (0=auto)");
    TORCH_CHECK(dynamic_limit >= 0 && update_limit >= 0 && (policy == 0 || policy == 1),
        "dynamic_cache_bytes/cache_update_bytes must be nonnegative; cache_policy must be 0/1");
    TORCH_CHECK(decay_interval >= 2 && decay_interval <= INT_MAX &&
        min_reuse >= 2 && min_reuse <= INT_MAX,
        "cache_decay_interval must be in [2,INT_MAX]; cache_min_reuse in [2,INT_MAX]");
    TORCH_CHECK(tile_m == 0 || tile_m == 32 || tile_m == 64 || tile_m == 128,
        "tile_m must be 0/32/64/128 (0=auto)");
    TORCH_CHECK(tile_n == 64 || tile_n == 128, "tile_n must be 64/128");
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    TORCH_CHECK(prop.major >= 8, "offload requires SM80+ (V100 unsupported)");
    size_t free, total;
    C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    if (!gpu) gpu = free / 5 * 4;
    if (!ram) ram = auto_ram_budget();
    if (!pinned) pinned = std::min<int64_t>(ram / 8, 256LL << 20);
    TORCH_CHECK(cache_limit >= 0 && cache_limit <= gpu,
        "resident_cache_bytes must be nonnegative and no greater than gpu_budget_bytes");
    c10::InferenceMode normal_tensors(false);
    Tensor token = at::zeros({2}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    int64_t id = next_id.fetch_add(1);
    token.data_ptr<int64_t>()[0] = id;
    auto context = std::make_shared<Context>(device, gpu, ram, pinned, banks, mode, max_tile,
        cache_limit, dynamic_limit, policy, decay_interval, min_reuse, update_limit,
        async_staging, host_profile, tile_m, tile_n,
        int64_t(prop.major) * 10 + prop.minor, token);
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.emplace(id, std::move(context));
    return token;
}

void fuse_moe_offload_bind(const Tensor& token, int64_t layer_id,
                          const Tensor& w1, const Tensor& w2) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    c->require_owner();
    TORCH_CHECK(!c->managed, "cannot bind inside a managed offload scope");
    TORCH_CHECK(!c->sealed, "deployment context is sealed");
    TORCH_CHECK(!c->closed && layer_id >= 0, "closed context or invalid layer_id");
    weights(w1, w2);
    TORCH_CHECK(!c->layers.count(layer_id), "layer already bound; use a new context for new weights");
    c->layers.emplace(layer_id, Layer{w1, w2, w1._version(), w2._version(),
        std::vector<Tensor>(w1.size(0)), c->dynamic_limit ? std::vector<Heat>(w1.size(0)) : std::vector<Heat>{}, 0});
    try {
        TORCH_CHECK(add(add(ram_weights(*c), pool_bytes(pool_sizes(*c), true)),
            cache_metadata_bytes(*c)) <= c->ram,
            "CPU weight backing and pinned pool exceed ram_budget_bytes");
    } catch (...) {
        c->layers.erase(layer_id);
        throw;
    }
}
void fuse_moe_offload_resident_bind(const Tensor& token, int64_t layer_id,
                                    at::IntArrayRef expert_ids) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    try {
    c->require_owner();
    TORCH_CHECK(!c->managed, "cannot bind inside a managed offload scope");
    TORCH_CHECK(!c->sealed, "deployment context is sealed");
    TORCH_CHECK(!c->closed, "offload context is closed");
    TORCH_CHECK(token.data_ptr<int64_t>()[1] == 0,
        "resident experts can only be bound before the first successful invocation");
    auto it = c->layers.find(layer_id);
    TORCH_CHECK(it != c->layers.end(), "layer must be bound before resident experts");
    auto& layer = it->second;
    TORCH_CHECK(layer.w1._version() == layer.v1 && layer.w2._version() == layer.v2,
        "weight version changed; bind a new context for updated weights");
    std::vector<bool> selected(layer.resident.size(), false);
    for (int64_t expert : expert_ids) {
        TORCH_CHECK(expert >= 0 && expert < int64_t(layer.resident.size()),
            "resident expert ID must be in [0,E)");
        TORCH_CHECK(!selected[expert] && !layer.resident[expert].defined(),
            "duplicate or already resident expert ID");
        selected[expert] = true;
    }
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(c->device));
    auto stream = at::cuda::getCurrentCUDAStream(c->device).stream();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "resident binding is not CUDA Graph capturable");
    if (expert_ids.empty()) return;
    const int64_t w1_bytes = mul(layer.w1[0].numel(), 2);
    const int64_t w2_bytes = mul(layer.w2[0].numel(), 2);
    const int64_t bytes = add(w1_bytes, w2_bytes);
    const int64_t additional = mul(expert_ids.size(), charge(bytes));
    const int64_t next = add(c->cache_bytes, additional);
    TORCH_CHECK(next <= c->cache_limit, "resident experts exceed resident_cache_bytes cap");
    TORCH_CHECK(add(next, pool_bytes(pool_sizes(*c), false)) <= c->gpu,
        "resident cache and pool exceed gpu_budget_bytes");
    const int64_t load = add(c->cache_load, mul(expert_ids.size(), bytes));
    // Transactional publication. Drain before any candidate storage can be freed
    // on allocation or copy errors; no partially loaded cache becomes visible.
    c10::InferenceMode normal_tensors(false);
    std::vector<Tensor> candidates;
    candidates.reserve(expert_ids.size());
    Flight flight(stream, *c);
    // A recycled allocation belongs to the current allocation stream. Complete
    // its previous users before writing from the independent DMA stream.
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto options = at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, c->device));
    for (size_t j = 0; j < expert_ids.size(); ++j)
        candidates.push_back(at::empty({bytes}, options));
    // The preceding sync covered recycled storage, not these new stream-ordered
    // allocations. Binding is graph-external and synchronous by contract.
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t j = 0; j < expert_ids.size(); ++j) {
        const int64_t expert = expert_ids[j];
        auto* dst = candidates[j].data_ptr<uint8_t>();
        C10_CUDA_CHECK(cudaMemcpyAsync(dst,
            static_cast<const uint8_t*>(layer.w1.data_ptr()) + expert * w1_bytes,
            w1_bytes, cudaMemcpyHostToDevice, flight.copy));
        C10_CUDA_CHECK(cudaMemcpyAsync(dst + w1_bytes,
            static_cast<const uint8_t*>(layer.w2.data_ptr()) + expert * w2_bytes,
            w2_bytes, cudaMemcpyHostToDevice, flight.copy));
    }
    flight.finish();
    for (size_t j = 0; j < expert_ids.size(); ++j)
        layer.resident[expert_ids[j]] = std::move(candidates[j]);
    c->cache_bytes = next;
    c->cache_experts += expert_ids.size();
    c->cache_load = load;
    } catch (const c10::OutOfMemoryError&) {
        throw; // Allocation-only binding failure remains transactional/recoverable.
    } catch (const c10::CUDAError&) {
        c->closed = true;
        throw;
    }
}

// Call outside the compiled graph: some AOTI fallback proxies ignore the
// executor error code and therefore cannot reliably propagate native guards.
// This preflight is read-only, including on failure; it does not seal context.
void fuse_moe_offload_validate(const Tensor& token, at::IntArrayRef deployment_manifest) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    ActiveContext executing(c.get());
    c->require_owner();
    TORCH_CHECK(!c->closed, "offload context is closed");
    TORCH_CHECK(deployment_manifest.size() >= 27 && deployment_manifest[0] == 1 &&
        deployment_manifest[1] > 0,
        "invalid deployment manifest format: expected version 1 and at least one complete layer record");
    const int64_t first_layer = deployment_manifest[2];
    const int64_t length = deployment_manifest[3];
    TORCH_CHECK(first_layer >= 0 && length >= 23 &&
        static_cast<uint64_t>(length) <= deployment_manifest.size() - 4,
        "invalid deployment manifest format: invalid first layer header or config length");
    const auto config = deployment_manifest.slice(4, static_cast<size_t>(length));
    validate_deployment_manifest(deployment_manifest, config, *c, first_layer);
}

#if defined(CUSTOM_OPS_OFFLOAD_TESTING) && CUSTOM_OPS_OFFLOAD_TESTING
void fuse_moe_offload_test_inject(const Tensor& token, const std::string& point, int64_t countdown) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    c->require_owner();
    TORCH_CHECK(!c->closed, "cannot inject into poisoned offload context");
    TORCH_CHECK(countdown >= 1, "fault countdown must be positive");
    auto found = std::find_if(fault_names.begin(), fault_names.end(),
        [&](const char* name) { return point == name; });
    TORCH_CHECK(found != fault_names.end(), "unknown offload fault point: ", point);
    std::lock_guard<std::mutex> fault_lock(c->faults.mutex);
    c->faults.remaining[size_t(found - fault_names.begin())] = countdown;
}
Tensor fuse_moe_offload_test_worker_states() { return StagingWorker::exercise_states(); }
#endif

void fuse_moe_offload_begin(const Tensor& token) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    TORCH_CHECK(!c->closed, "offload context is poisoned; close required");
    TORCH_CHECK(!c->managed, "managed offload scope is already active");
    if (c->pending_error) std::rethrow_exception(c->pending_error);
    TORCH_CHECK(c->managed_sequence < INT64_MAX, "offload managed sequence overflow");
    ++c->managed_sequence;
    c->managed_owner = std::this_thread::get_id();
    c->managed = true;
}
void fuse_moe_offload_check(const Tensor& token) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    c->require_owner();
    c->managed = false; // End scope even when the stored exception is rethrown.
    c->managed_owner = std::thread::id{};
    if (c->pending_error) std::rethrow_exception(c->pending_error);
    TORCH_CHECK(!c->closed, "offload context is poisoned; close required");
}
void fuse_moe_offload_acknowledge(const Tensor& token) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    TORCH_CHECK(!c->managed, "cannot acknowledge inside a managed offload scope");
    TORCH_CHECK(!c->closed, "cannot acknowledge poisoned offload context; close required");
    c->pending_error = nullptr;
    c->failure_layer = -1;
    c->failure_sequence = 0;
}
Tensor fuse_moe_offload_status(const Tensor& token) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    using fuse_moe::OffloadState;
    const auto state = c->closed ? OffloadState::Poisoned :
        c->pending_error ? OffloadState::Recoverable : OffloadState::Healthy;
    return at::tensor({fuse_moe::kOffloadStatusVersion, int64_t(state),
        int64_t(c->managed), int64_t(bool(c->pending_error)), c->forward_sequence,
        c->success_sequence, c->failure_layer, c->failure_sequence,
        c->managed_sequence, c->success_layer},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

void fuse_moe_offload_close(const Tensor& token) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    c->require_owner();
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(c->device));
    c->managed = false;
    c->managed_owner = std::thread::id{};
    c->closed = true;
    c->staging.shutdown();
    c->transfers.cleanup();
    for (auto& buffer : c->pool) buffer = Tensor();
    c->layers.clear();
    c->outputs.clear();
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    registry.erase(token.data_ptr<int64_t>()[0]);
}
Tensor fuse_moe_offload_stats(const Tensor& token) {
    auto c = lookup(token);
    std::lock_guard<std::mutex> lock(c->mutex);
    auto stats = c->stats;
    stats[19] = c->cache_bytes; stats[20] = c->cache_experts; stats[21] = c->cache_load;
    stats[22] = c->dynamic_bytes; stats[23] = c->dynamic_experts;
    stats[30] = cache_metadata_bytes(*c);
    return at::tensor(stats, at::TensorOptions().dtype(at::kLong));
}

Tensor fuse_moe_offload(const Tensor& x, const Tensor& w1, const Tensor& w2,
    const Tensor& ids, const Tensor& scale, Tensor token, int64_t layer_id,
    std::optional<at::IntArrayRef> config_argument,
    std::optional<at::IntArrayRef> manifest_argument) {
    const at::IntArrayRef deployment_config = config_argument.value_or(at::IntArrayRef{});
    const at::IntArrayRef deployment_manifest = manifest_argument.value_or(at::IntArrayRef{});
    auto c = lookup(token);
    std::unique_lock<std::mutex> lock(c->mutex);
    // Admission is not an owner invocation failure: an unauthorized thread must
    // not change the managed owner's sticky error, status or sequence.
    c->require_owner();
    ActiveContext executing(c.get());
    // Lookup/identity/admission failures are excluded; owner forward guards are
    // captured even when an AOTI proxy swallows their exceptions.
    // Eager retries may execute with a sticky recoverable error, but NEVER clear it.
    int64_t sequence = c->forward_sequence;
    try {
    TORCH_CHECK(!c->closed, "offload context is closed/poisoned; close required");
    if (c->managed && c->pending_error) std::rethrow_exception(c->pending_error);
    TORCH_CHECK(c->forward_sequence < INT64_MAX, "offload forward sequence overflow");
    sequence = ++c->forward_sequence;
    auto layer = c->layers.find(layer_id);
    TORCH_CHECK(layer != c->layers.end(), "layer must be bound before inference");
    validate_deployment_manifest(deployment_manifest, deployment_config, *c, layer_id);
    validate_deployment_config(deployment_config, *c, layer->second, w1, w2);
    auto same_weight = [](const Tensor& a, const Tensor& b) {
        return a.storage().unsafeGetStorageImpl() == b.storage().unsafeGetStorageImpl() &&
            a.storage_offset() == b.storage_offset() && a.sizes() == b.sizes() &&
            a.strides() == b.strides() && a.scalar_type() == b.scalar_type();
    };
    TORCH_CHECK(same_weight(w1, layer->second.w1) && same_weight(w2, layer->second.w2) &&
        w1._version() == layer->second.v1 && w2._version() == layer->second.v2 &&
        layer->second.w1._version() == layer->second.v1 &&
        layer->second.w2._version() == layer->second.v2,
        "weight identity/version changed; bind a new context for updated weights");
    TORCH_CHECK(x.is_cuda() && x.get_device() == c->device && x.dim() == 2 && x.is_contiguous() &&
        x.scalar_type() == w1.scalar_type() && x.size(1) == w1.size(2), "invalid offload x");
    TORCH_CHECK(ids.device() == x.device() && scale.device() == x.device() &&
        ids.scalar_type() == at::kInt && scale.scalar_type() == at::kFloat &&
        ids.dim() == 2 && ids.size(0) == x.size(0) && ids.size(1) > 0 && ids.size(1) <= 128 &&
        scale.sizes() == ids.sizes() && ids.is_contiguous() && scale.is_contiguous(),
        "routes must be contiguous CUDA int32/float32 (S,K), 1<=K<=128");
    TORCH_CHECK(!x.requires_grad() && !scale.requires_grad(), "offload is inference-only");
    int64_t s = x.size(0), h = x.size(1), k = ids.size(1), e = w1.size(0), i = w2.size(2);
    int64_t t = mul(s, k);
    TORCH_CHECK(t <= INT_MAX - 256 && s <= INT_MAX, "route count exceeds int32 kernel indexing");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(c->device).stream();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
        "offload host planning is not CUDA Graph capturable; disable cudagraphs for AOTi");
    TORCH_CHECK(token.data_ptr<int64_t>()[1] < INT64_MAX, "runtime invocation counter overflow");
    if (!deployment_config.empty()) c->sealed = true;
    if (!deployment_manifest.empty()) c->seal_requires_manifest = true;
    c->outputs.erase(std::remove_if(c->outputs.begin(), c->outputs.end(),
        [](const Output& o) { return o.storage.expired(); }), c->outputs.end());
    int64_t history = 0;
    for (const auto& o : c->outputs) history = add(history, o.bytes);
    int64_t output_bytes = charge(mul(mul(s, h), 2));
    int64_t outputs = add(history, output_bytes);
    Sizes routing{};
    routing[Routes] = mul(2 * t + 4 * e + 3, 4);
    routing[Counts] = mul(e + 1, 4);
    // Count mirror plus bounded C++ wave IDs/planning vectors (not allocator RSS).
    int64_t host_plan_bytes = add(4096 + mul(e, 128), cache_metadata_bytes(*c));
    int64_t backing = ram_weights(*c);
    CacheActivity activity;
    TORCH_CHECK(fits(*c, routing, outputs, backing, host_plan_bytes, c->dynamic_bytes),
        "pool/history exceeds gpu_budget_bytes or RAM/pinned budgets");
    Allocations allocations;
    // Validate and count using O(E) storage before allocating O(S*K) routes or
    // evicting anything. Failed ID/budget validation must not destroy hot entries.
    // Keep an existing routing pool when it already fits, avoiding churn on hits.
    const bool preflight = !fits(*c, routing, outputs, backing, host_plan_bytes);
    Sizes summary{};
    if (preflight) {
        summary[Counts] = routing[Counts];
        // One allocator quantum even for E=512: an existing successful dynamic
        // context always has at least its old Routes quantum to reuse as headroom.
        // Slice the histogram rather than assuming the full O(E) summary fits.
        summary[RouteSummary] = 512;
        configure_pool(*c, summary, history, backing, host_plan_bytes, allocations);
        Flight counting(stream, *c);
        for (int first = 0; first < e; first += 127) {
            const int bins = std::min<int64_t>(127, e - first);
            auto* gpu_counts = pointer<int>(*c, RouteSummary);
            auto* host_counts = pointer<int>(*c, Counts);
            moe_offload_count(ids.data_ptr<int>(), t, e, first, bins, gpu_counts, stream);
            C10_CUDA_CHECK(cudaMemcpyAsync(host_counts + first, gpu_counts,
                bins * 4, cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(host_counts + e, gpu_counts + bins,
                4, cudaMemcpyDeviceToHost, stream));
        }
        counting.finish();
    } else {
        configure_pool(*c, routing, outputs, backing, host_plan_bytes, allocations);
    }
    auto* r = preflight ? nullptr : pointer<int>(*c, Routes);
    auto* counts = pointer<int>(*c, Counts);
    fuse_moe::FMOE_params p{false};
    p.x_ptr = x.data_ptr(); p.topk_ids_ptr = ids.data_ptr<int>();
    p.topk_scale_ptr = scale.data_ptr<float>();
    auto route_pointers = [&] {
        p.row_indices_ptr = r; p.topk_pos_ptr = r + t;
        p.seqlens_ptr = r + 2 * t; p.cu_seqlens_ptr = p.seqlens_ptr + e;
        p.tiles_ptr = p.cu_seqlens_ptr + e + 1; p.cu_tiles_ptr = p.tiles_ptr + e;
    };
    p.num_seq = s; p.hidden = h; p.intermediate = i; p.num_expert = e;
    p.num_topk = k; p.total_num_seq = t; p.is_bf16 = x.scalar_type() == at::kBFloat16;
    p.tile_m = c->tile_m ? int(c->tile_m) : fuse_moe::moe_pick_tile_m_auto(t / e, true);
    p.use_pdl = false; p.use_vert_sched = false;
    p.use_wide_n = c->tile_n == 128;
    // Planning can trim unused buffers only after route work has drained.
    if (!preflight) {
        route_pointers();
        int* invalid = p.cu_tiles_ptr + e + 1;
        Flight routing_flight(stream, *c);
        if (t) {
            moe_offload_route(p, invalid, stream);
            C10_CUDA_CHECK(cudaMemcpyAsync(counts, p.seqlens_ptr, e * 4,
                cudaMemcpyDeviceToHost, stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(counts + e, invalid, 4,
                cudaMemcpyDeviceToHost, stream));
        } else std::memset(counts, 0, routing[Counts]);
        routing_flight.finish();
    }
    TORCH_CHECK(counts[e] == 0, "topk_ids must be -1 or in [0,E)");
    int active = 0, hits = 0, dynamic_hits = 0;
    for (int j = 0; j < e; ++j) active += counts[j] > 0;
    auto make_plan = [&](int64_t reclaimed = 0, const std::vector<bool>* excluded = nullptr) {
        int plan_hits = 0, plan_dynamic_hits = 0;
        for (int j = 0; j < e; ++j) {
            const bool hit = counts[j] > 0 && layer->second.resident[j].defined();
            const bool dynamic = hit && c->dynamic_limit && layer->second.heat[j].dynamic;
            const bool kept = !excluded || !(*excluded)[j];
            plan_hits += hit && kept;
            plan_dynamic_hits += dynamic && kept;
        }
        auto candidate = [&](int mode) {
            return execution_plan(*c, counts, e, active, plan_hits, layer->second, s, h, i, t, routing,
                outputs, backing, host_plan_bytes, mode, p.tile_m, reclaimed, excluded);
        };
        Execution result = candidate(c->mode == 1 ? 1 : 0);
        if (c->mode == 2) {
            auto accumulate = candidate(1);
            if (accumulate.valid && (!result.valid || accumulate.peak < result.peak))
                result = std::move(accumulate);
        }
        hits = plan_hits; dynamic_hits = plan_dynamic_hits;
        return result;
    };
    Execution execution = make_plan();
    std::vector<Victim> budget_victims;
    if (!execution.valid && c->dynamic_experts) {
        auto victims = cache_victims(*c);
        // Reclaim inactive/cross-layer entries first: dropping a live hit can
        // introduce staging and make pinned/RAM demand worse, not better.
        auto active_hit = [&](const Victim& v) {
            return v.layer == &layer->second && counts[v.expert] > 0;
        };
        std::stable_partition(victims.begin(), victims.end(),
            [&](const Victim& v) { return !active_hit(v); });
        std::vector<bool> excluded(e, false);
        int64_t reclaimed = 0;
        for (const auto& victim : victims) {
            budget_victims.push_back(victim);
            reclaimed = add(reclaimed, victim.bytes);
            if (victim.layer == &layer->second) excluded[victim.expert] = true;
            execution = make_plan(reclaimed, &excluded);
            if (execution.valid) break;
        }
    }
    TORCH_CHECK(execution.valid,
        "budgets cannot fit one expert and its activations/output/history; increase gpu_budget_bytes/RAM/pinned or reduce routes");
    if (execution.mode == 0)
        TORCH_CHECK(mul(s, (h + 2047) / 2048) <= INT_MAX,
            "route_buffer reduction exceeds int32 grid indexing; use fp32_accumulate");
    // Commit only the victim prefix proven feasible by the exact planner.
    for (const auto& victim : budget_victims) evict(*c, victim, activity, true);
    configure_pool(*c, execution.sizes, outputs, backing, host_plan_bytes, allocations,
                   execution.cache_reserve);
    if (preflight) {
        r = pointer<int>(*c, Routes);
        counts = pointer<int>(*c, Counts);
        route_pointers();
        Flight routing_flight(stream, *c);
        if (t) {
            moe_offload_route(p, p.cu_tiles_ptr + e + 1, stream);
        }
        routing_flight.finish();
    }
    const int banks = execution.banks, capacity = execution.capacity;
    const bool accumulate = execution.mode == 1;
    auto& waves = execution.waves;
    Tensor output = at::empty({s, h}, x.options());
    p.out_ptr = output.data_ptr();
    p.down_out_ptr = c->pool[Down].defined() ? c->pool[Down].data_ptr() : nullptr;
    int64_t route_tiles = 0;
    auto& transfers = c->transfers;
    std::vector<size_t> miss_indices;
    if (c->async_staging) {
        miss_indices.reserve(waves.size());
        for (size_t index = 0; index < waves.size(); ++index)
            if (!waves[index].resident) miss_indices.push_back(index);
    }
    // Defined after the borrowed plan, and before any producer submission.
    // Its destructor drains the producer before CUDA or plan storage is released.
    Flight flight(stream, *c, c->async_staging ? &c->staging : nullptr);
    if (banks) {
        // Pool growth may allocate on the caller stream (cudaMallocAsync is
        // stream ordered). Establish allocation -> first copy without host sync;
        // later ready records refer to DMA generations, after this wait enqueue.
        C10_CUDA_CHECK(cudaEventRecord(transfers.ready[0], stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(flight.copy, transfers.ready[0], 0));
    }
    if (accumulate && s) {
        C10_CUDA_CHECK(cudaMemsetAsync(pointer<float>(*c, Accumulator), 0,
            mul(mul(s, h), 4), stream));
        moe_offload_scales(p.topk_pos_ptr, p.topk_scale_ptr, pointer<float>(*c, Scales), t, stream);
    }
    int64_t w1_bytes = mul(mul(2 * i, h), 2), w2_bytes = mul(mul(h, i), 2);
    int64_t miss_waves = 0, metadata_bytes = 0;
    std::vector<bool> consumed(c->dynamic_limit ? e : 0, false);
    const int64_t timestamp = token.data_ptr<int64_t>()[1] + 1;
    auto enqueue = [&](size_t ordinal, int bank, bool reused) {
        auto* host = pointer<uint8_t>(*c, HostBank0 + bank);
        auto* meta = pointer<int>(*c, HostMeta0 + bank);
        const auto* wave = &waves[miss_indices[ordinal]];
        const auto* gate = static_cast<const uint8_t*>(w1.data_ptr());
        const auto* down = static_cast<const uint8_t*>(w2.data_ptr());
        const auto ready = transfers.ready[bank];
        const int device = c->device, tile_m = p.tile_m;
        const bool profile = c->host_profile;
        const void* context = c.get();
        // Propagate observer TLS per invocation, not at persistent thread startup.
        // No diagnostic TLS copies on the normal fast path.
        std::optional<at::ThreadLocalState> tls;
        if (profile) {
            try { tls.emplace(); } catch (...) { /* Diagnostics are best effort. */ }
        }
        try {
            c->staging.submit(bank, [=, tls = std::move(tls)] {
                ActiveContext executing(context);
                OFFLOAD_FAULT(*c, WorkerRunning);
                std::optional<at::ThreadLocalStateGuard> observer;
                if (tls) {
                    try { observer.emplace(*tls); } catch (...) { /* Continue without scopes. */ }
                }
                if (reused) {
                    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
                    HostStage stage(profile, "fuse_moe_offload::host_bank_wait");
                    C10_CUDA_CHECK(cudaEventSynchronize(ready));
                }
                prepare_wave(*wave, counts, capacity, tile_m, host, meta,
                    gate, down, w1_bytes, w2_bytes, profile);
            });
        } catch (...) { c->closed = true; throw; }
    };
    if (c->async_staging)
        for (int bank = 0; bank < std::min<size_t>(banks, miss_indices.size()); ++bank)
            enqueue(bank, bank, false);
    for (auto& wave : waves) {
        int groups = wave.experts.size();
        int b = -1;
        uint8_t* gpu;
        void* down_weight;
        int* wave_meta;
        if (wave.resident) {
            gpu = layer->second.resident[wave.experts[0]].data_ptr<uint8_t>();
            down_weight = gpu + w1_bytes;
            wave_meta = pointer<int>(*c, ResidentMeta);
            MoEResidentWave cached;
            for (int g = 0; g < groups; ++g) {
                cached.weights[g] = layer->second.resident[wave.experts[g]].data_ptr();
                cached.rows[g] = counts[wave.experts[g]];
            }
            moe_offload_resident_meta(wave_meta, groups, cached, w1_bytes, p.tile_m, stream);
        } else {
            b = miss_waves % banks;
            if (miss_waves >= banks) {
                HostStage stage(c->host_profile, "fuse_moe_offload::bank_wait");
                C10_CUDA_CHECK(cudaEventSynchronize(transfers.done[b]));
            }
            ++miss_waves;
            auto* host = pointer<uint8_t>(*c, HostBank0 + b);
            auto* meta = pointer<int>(*c, HostMeta0 + b);
            if (c->async_staging) {
                HostStage stage(c->host_profile, "fuse_moe_offload::host_ready_wait");
                try {
                    c->staging.wait(b);
                    OFFLOAD_FAULT(*c, WorkerReady);
                }
                catch (...) { c->closed = true; throw; }
            } else {
                prepare_wave(wave, counts, capacity, p.tile_m, host, meta,
                    static_cast<const uint8_t*>(w1.data_ptr()),
                    static_cast<const uint8_t*>(w2.data_ptr()), w1_bytes, w2_bytes, c->host_profile);
            }
            gpu = pointer<uint8_t>(*c, GpuBank0 + b);
            down_weight = gpu + capacity * w1_bytes;
            wave_meta = pointer<int>(*c, GpuMeta0 + b);
            {
                HostStage stage(c->host_profile, "fuse_moe_offload::h2d_submit");
                C10_CUDA_CHECK(cudaMemcpyAsync(gpu, host, groups * w1_bytes,
                    cudaMemcpyHostToDevice, flight.copy));
                OFFLOAD_POISON_FAULT(*c, PartialH2D);
                C10_CUDA_CHECK(cudaMemcpyAsync(gpu + capacity * w1_bytes,
                    host + capacity * w1_bytes, groups * w2_bytes, cudaMemcpyHostToDevice, flight.copy));
                C10_CUDA_CHECK(cudaMemcpyAsync(wave_meta, meta, (4 * groups + 2) * 4,
                    cudaMemcpyHostToDevice, flight.copy));
                C10_CUDA_CHECK(cudaEventRecord(transfers.ready[b], flight.copy));
                OFFLOAD_POISON_FAULT(*c, EventRecord);
                C10_CUDA_CHECK(cudaStreamWaitEvent(stream, transfers.ready[b], 0));
            }
            // The host bank is reusable after this DMA, independently of the
            // GPU bank's compute/promotion lease. Enqueue only its next miss.
            if (c->async_staging && size_t(miss_waves - 1 + banks) < miss_indices.size())
                enqueue(miss_waves - 1 + banks, b, true);
            metadata_bytes = add(metadata_bytes, (4 * groups + 2) * 4);
        }
        const int step = accumulate ? execution.rows : wave.rows;
        int first_group = 0, group_begin = 0;
        for (int begin = 0; begin < wave.rows;) {
            int rows = std::min(step, wave.rows - begin);
            int tile_groups = groups;
            if (accumulate) {
                // The full weight wave stays available, but GEMM need not scan empty groups
                // outside this tile. Counts and compact routes share expert order.
                while (group_begin + counts[wave.experts[first_group]] <= begin) {
                    group_begin += counts[wave.experts[first_group++]];
                }
                int last_group = first_group;
                int group_end = group_begin + counts[wave.experts[last_group]];
                while (group_end < begin + rows)
                    group_end += counts[wave.experts[++last_group]];
                tile_groups = last_group - first_group + 1;
            }
            auto q = p;
            q.num_expert = tile_groups; q.total_num_seq = rows;
            q.row_indices_ptr = p.row_indices_ptr + wave.begin + begin;
            q.w1_ptr = wave.resident ? gpu : gpu + int64_t(first_group) * w1_bytes;
            q.w2_ptr = wave.resident ? down_weight :
                static_cast<uint8_t*>(down_weight) + int64_t(first_group) * w2_bytes;
            if (wave.resident) {
                if (tile_groups > 1) {
                    auto weight_ptrs = reinterpret_cast<const void* const*>(wave_meta + 4 * groups + 2);
                    q.w1_group_ptrs = weight_ptrs + first_group;
                    q.w2_group_ptrs = weight_ptrs + groups + first_group;
                } else {
                    auto* weight = layer->second.resident[wave.experts[first_group]].data_ptr<uint8_t>();
                    q.w1_ptr = weight; q.w2_ptr = weight + w1_bytes;
                }
            }
            q.act_out_ptr = c->pool[Act].data_ptr();
            q.down_out_ptr = accumulate ? nullptr : static_cast<uint8_t*>(p.down_out_ptr) +
                int64_t(wave.begin + begin) * h * 2;
            const bool clipped = accumulate && (begin != 0 || rows != wave.rows);
            q.seqlens_ptr = clipped ? pointer<int>(*c, TileMeta) : wave_meta;
            if (clipped)
                moe_offload_tile_meta(wave_meta, q.seqlens_ptr, groups, first_group,
                    tile_groups, begin, rows, p.tile_m, stream);
            if (accumulate) {
                q.accumulator_ptr = pointer<float>(*c, Accumulator);
                q.sorted_scale_ptr = pointer<float>(*c, Scales) + wave.begin + begin;
            }
            q.cu_seqlens_ptr = q.seqlens_ptr + tile_groups;
            q.tiles_ptr = q.cu_seqlens_ptr + tile_groups + 1; q.cu_tiles_ptr = q.tiles_ptr + tile_groups;
            {
                HostStage stage(c->host_profile, "fuse_moe_offload::compute_submit");
                moe_offload_compute(q, stream);
                OFFLOAD_POISON_FAULT(*c, KernelSubmit);
            }
            ++route_tiles;
            begin += rows;
        }
        if (c->dynamic_limit) {
            for (int expert : wave.experts) {
                consumed[expert] = true;
                if (layer->second.heat[expert].dynamic)
                    layer->second.heat[expert].last_use = timestamp;
            }
            if (!wave.resident) {
                for (int g = 0; g < groups; ++g) {
                    if (!promote(*c, layer->second, wave.experts[g], gpu + g * w1_bytes,
                            static_cast<uint8_t*>(down_weight) + g * w2_bytes, counts, consumed,
                            outputs, stream, activity, timestamp)) ++activity.bypasses;
                }
            }
        }
        // Promotion also reads this bank; release it only after publication.
        if (b >= 0) C10_CUDA_CHECK(cudaEventRecord(transfers.done[b], stream));
    }
    if (accumulate && s)
        moe_offload_cast(pointer<float>(*c, Accumulator), p.out_ptr, s, h, p.is_bf16, stream);
    else if (t) moe_offload_reduce(p, stream);
    {
        HostStage stage(c->host_profile, "fuse_moe_offload::drain");
        flight.finish();
    }
    c->outputs.push_back({output.storage().getWeakStorageImpl(), output_bytes});
    auto sizes = pool_sizes(*c);
    int64_t gpu_pool = pool_bytes(sizes, false), pinned_pool = pool_bytes(sizes, true);
    int64_t reuses = std::count(allocations.reused.begin(), allocations.reused.end(), true);
    const int64_t expert_bytes = mul(mul(h, i), 6);
    c->stats = {active, int64_t(waves.size()), mul(active - hits, expert_bytes),
        metadata_bytes, add(add(outputs, gpu_pool), total_cache_bytes(*c)), pinned_pool,
        add(add(backing, pinned_pool), host_plan_bytes), banks, capacity,
        execution.mode, route_tiles, execution.rows, allocations.count, reuses, gpu_pool, pinned_pool,
        hits, active - hits, mul(hits, expert_bytes), c->cache_bytes, c->cache_experts, c->cache_load,
        c->dynamic_bytes, c->dynamic_experts, dynamic_hits, activity.admissions,
        activity.evictions, activity.bypasses, activity.promotion_bytes, activity.budget_evictions,
        cache_metadata_bytes(*c)};
    commit_heat(*c, layer->second, counts);
    token.select(0, 1).add_(1);
    c->success_sequence = sequence;
    c->success_layer = layer_id;
    return output;
    } catch (const c10::OutOfMemoryError&) {
        // Allocation-only failure is recoverable once all Flight leases drain.
        c->record_error(std::current_exception(), layer_id, sequence);
        c->drain_error = nullptr;
        throw;
    } catch (const c10::CUDAError&) {
        // Includes failures before a Flight exists (guard/capture/pool setup).
        c->closed = true;
        c->record_error(std::current_exception(), layer_id, sequence);
        c->drain_error = nullptr;
        throw;
    } catch (...) {
        c->record_error(std::current_exception(), layer_id, sequence);
        c->drain_error = nullptr;
        throw;
    }
}
