#include "einsum_define.h"
#include "einsum_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using pyqcd_einsum::ExecutionPlan;
using pyqcd_einsum::ExecutionStep;
using pyqcd_einsum::PairPlan;
using pyqcd_einsum::TensorMeta;
using pyqcd_einsum::UnaryPlan;

bool trace_enabled() {
    static const bool enabled = std::getenv("PYQCD_EINSUM_TRACE") != nullptr;
    return enabled;
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;

template <typename Type>
const Type* as_const_pointer(int64_t address) {
    if (address == 0) {
        return nullptr;
    }
    return reinterpret_cast<const Type*>(static_cast<std::intptr_t>(address));
}

template <typename Type>
Type* as_pointer(int64_t address) {
    if (address == 0) {
        return nullptr;
    }
    return reinterpret_cast<Type*>(static_cast<std::intptr_t>(address));
}

bool checked_mul(int64_t left, int64_t right, int64_t* result) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_product(const std::vector<int64_t>& shape, int64_t* result) {
    int64_t product = 1;
    for (int64_t extent : shape) {
        if (!checked_mul(product, extent, &product)) {
            return false;
        }
    }
    *result = product;
    return true;
}

bool checked_bytes(int64_t count, std::size_t element_size, std::size_t* bytes) {
    if (count < 0 || element_size == 0 ||
        static_cast<uint64_t>(count) >
            static_cast<uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
        return false;
    }
    *bytes = static_cast<std::size_t>(count) * element_size;
    return true;
}

bool valid_shape(const std::vector<int64_t>& shape) {
    return std::all_of(shape.begin(), shape.end(), [](int64_t extent) { return extent >= 0; });
}

std::size_t scalar_size(int32_t dtype_id) {
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
        case QCD_DTYPE_FLOAT64:
            return sizeof(double);
        case QCD_DTYPE_COMPLEX64:
        case QCD_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

std::size_t element_size(int32_t dtype_id) {
    const std::size_t scalar = scalar_size(dtype_id);
    if (dtype_id == QCD_DTYPE_COMPLEX128 || dtype_id == QCD_DTYPE_COMPLEX64) {
        return 2 * scalar;
    }
    return scalar;
}

bool tensors_match(const std::vector<int32_t>& left, const std::vector<int32_t>& right) {
    return left == right;
}

bool contains_label(const std::vector<int32_t>& labels, int32_t label) {
    return std::find(labels.begin(), labels.end(), label) != labels.end();
}

int64_t label_position(const std::vector<int32_t>& labels, int32_t label) {
    const auto found = std::find(labels.begin(), labels.end(), label);
    return found == labels.end() ? -1 : static_cast<int64_t>(found - labels.begin());
}

/* Recycles the large step buffers: a fresh 256 MiB mapping costs ~90 ms of
 * first-touch page faults, and a single E1 call maps about 1 GiB. */
class BufferPool {
  public:
    BufferPool() = default;
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    ~BufferPool() {
        for (Block& block : blocks_) {
            std::free(block.data);
        }
    }

    uint8_t* acquire(std::size_t bytes, std::size_t* capacity) {
        *capacity = bytes;
        if (bytes == 0) {
            return nullptr;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        std::size_t best = blocks_.size();
        for (std::size_t index = 0; index < blocks_.size(); ++index) {
            const std::size_t available = blocks_[index].bytes;
            if (available < bytes || available > bytes * 2) {
                continue;
            }
            if (best == blocks_.size() || available < blocks_[best].bytes) {
                best = index;
            }
        }
        if (best != blocks_.size()) {
            const Block block = blocks_[best];
            blocks_[best] = blocks_.back();
            blocks_.pop_back();
            cached_ -= block.bytes;
            *capacity = block.bytes;
            return block.data;
        }
        void* allocated = nullptr;
        if (posix_memalign(&allocated, 64, bytes) != 0) {
            return nullptr;
        }
        return static_cast<uint8_t*>(allocated);
    }

    void release(uint8_t* data, std::size_t bytes) {
        if (data == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        if (bytes >= kMinPooledBytes && cached_ + bytes <= kMaxPooledBytes) {
            blocks_.push_back(Block{data, bytes});
            cached_ += bytes;
            return;
        }
        std::free(data);
    }

  private:
    static constexpr std::size_t kMinPooledBytes = std::size_t(1) << 19;
    static constexpr std::size_t kMaxPooledBytes = std::size_t(1) << 30;

    struct Block {
        uint8_t* data;
        std::size_t bytes;
    };

    std::mutex mutex_;
    std::vector<Block> blocks_;
    std::size_t cached_ = 0;
};

BufferPool& buffer_pool() {
    static BufferPool pool;
    return pool;
}

class Buffer {
  public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.bytes_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            bytes_ = other.bytes_;
            other.data_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~Buffer() { release(); }

    bool allocate(std::size_t bytes) {
        release();
        if (bytes == 0) {
            return true;
        }
        data_ = buffer_pool().acquire(bytes, &bytes_);
        return data_ != nullptr;
    }

    uint8_t* data() {
        return data_;
    }

    const uint8_t* data() const {
        return data_;
    }

  private:
    void release() noexcept {
        buffer_pool().release(data_, bytes_);
        data_ = nullptr;
        bytes_ = 0;
    }

    uint8_t* data_ = nullptr;
    std::size_t bytes_ = 0;
};

struct SlotData {
    const uint8_t* external = nullptr;
    Buffer storage;
    bool owns_storage = false;
    std::vector<int32_t> labels;
    std::vector<int64_t> shape;
    bool has_labels = false;

    const uint8_t* data() const {
        return owns_storage ? storage.data() : external;
    }
};

using SgemmFn = void (*)(int, int, int, int, int, int, float, const void*, int, const void*, int,
                         float, void*, int);
using DgemmFn = void (*)(int, int, int, int, int, int, double, const void*, int, const void*,
                         int, double, void*, int);
using CgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);
using ZgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);
using Sgemm64Fn = void (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float,
                           const void*, int64_t, const void*, int64_t, float, void*, int64_t);
using Dgemm64Fn = void (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double,
                           const void*, int64_t, const void*, int64_t, double, void*, int64_t);
using Cgemm64Fn = void (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           const void*, const void*, int64_t, const void*, int64_t,
                           const void*, void*, int64_t);
using Zgemm64Fn = void (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           const void*, const void*, int64_t, const void*, int64_t,
                           const void*, void*, int64_t);

struct CblasApi {
    void* handle = nullptr;
    SgemmFn sgemm = nullptr;
    DgemmFn dgemm = nullptr;
    CgemmFn cgemm = nullptr;
    ZgemmFn zgemm = nullptr;
    Sgemm64Fn sgemm64 = nullptr;
    Dgemm64Fn dgemm64 = nullptr;
    Cgemm64Fn cgemm64 = nullptr;
    Zgemm64Fn zgemm64 = nullptr;
    bool ilp64 = false;

    bool complete() const {
        return sgemm64 != nullptr && dgemm64 != nullptr && cgemm64 != nullptr &&
                   zgemm64 != nullptr
               ? true
               : sgemm != nullptr && dgemm != nullptr && cgemm != nullptr &&
                     zgemm != nullptr;
    }
};

template <typename Function>
bool bind_symbol(void* handle, const char* name, Function* function) {
    *function = nullptr;
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return false;
    }
    static_assert(sizeof(Function) == sizeof(symbol), "dlsym function-pointer size mismatch");
    std::memcpy(function, &symbol, sizeof(Function));
    return true;
}

bool bind_cblas(void* handle, CblasApi* api) {
    return bind_symbol(handle, "cblas_sgemm", &api->sgemm) &&
           bind_symbol(handle, "cblas_dgemm", &api->dgemm) &&
           bind_symbol(handle, "cblas_cgemm", &api->cgemm) &&
           bind_symbol(handle, "cblas_zgemm", &api->zgemm);
}

bool bind_cblas64(void* handle, CblasApi* api) {
    const bool ready =
        bind_symbol(handle, "scipy_cblas_sgemm64_", &api->sgemm64) &&
        bind_symbol(handle, "scipy_cblas_dgemm64_", &api->dgemm64) &&
        bind_symbol(handle, "scipy_cblas_cgemm64_", &api->cgemm64) &&
        bind_symbol(handle, "scipy_cblas_zgemm64_", &api->zgemm64);
    api->ilp64 = ready;
    return ready;
}

CblasApi load_cblas() {
    CblasApi api;
    const char* requested = std::getenv("PYQCD_EINSUM_CBLAS_LIB");
    if (requested != nullptr && requested[0] != '\0') {
        void* handle = dlopen(requested, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            CblasApi requested_api;
            requested_api.handle = handle;
            if (bind_cblas64(handle, &requested_api) ||
                bind_cblas(handle, &requested_api)) {
                return requested_api;
            }
            dlclose(handle);
        }
    }
    if (bind_cblas(RTLD_DEFAULT, &api)) {
        return api;
    }

    constexpr std::array<const char*, 6> candidates = {
        "libcblas.so.3", "libcblas.so", "libopenblas.so.0", "libopenblas.so",
        "libopenblas.dylib", "libcblas.dylib"};
    for (const char* candidate : candidates) {
        void* handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            continue;
        }
        CblasApi candidate_api;
        candidate_api.handle = handle;
        if (bind_cblas(handle, &candidate_api)) {
            return candidate_api;
        }
        dlclose(handle);
    }
    return CblasApi{};
}

const CblasApi& cblas_api() {
    static CblasApi api;
    static std::once_flag initialized;
    std::call_once(initialized, [] { api = load_cblas(); });
    return api;
}

bool allocate_buffer(Buffer* buffer, int64_t count, int32_t dtype_id) {
    std::size_t bytes = 0;
    return checked_bytes(count, element_size(dtype_id), &bytes) && buffer->allocate(bytes);
}

bool shape_matches_symbols(const TensorMeta& tensor, const std::vector<int64_t>& sizes) {
    if (tensor.labels.size() != tensor.shape.size()) {
        return false;
    }
    for (std::size_t axis = 0; axis < tensor.labels.size(); ++axis) {
        const int32_t label = tensor.labels[axis];
        if (label < 0 || static_cast<std::size_t>(label) >= sizes.size() ||
            tensor.shape[axis] < 0 || tensor.shape[axis] != sizes[static_cast<std::size_t>(label)]) {
            return false;
        }
    }
    return true;
}

bool validate_actual_shape(const TensorMeta& tensor, const std::vector<int64_t>& sizes) {
    if (tensor.labels.size() != tensor.shape.size()) {
        return false;
    }
    for (std::size_t axis = 0; axis < tensor.labels.size(); ++axis) {
        const int32_t label = tensor.labels[axis];
        if (label < 0 || static_cast<std::size_t>(label) >= sizes.size()) {
            return false;
        }
        const int64_t expected = sizes[static_cast<std::size_t>(label)];
        const int64_t actual = tensor.shape[axis];
        if (actual < 0 || (actual != expected && actual != 1)) {
            return false;
        }
    }
    return true;
}

bool resolve_input(const SlotData& slot, const TensorMeta& expected,
                   const std::vector<int64_t>& sizes, TensorMeta* actual) {
    if (slot.has_labels && !tensors_match(slot.labels, expected.labels)) {
        return false;
    }
    if (slot.shape.size() != expected.labels.size()) {
        return false;
    }
    *actual = expected;
    actual->shape = slot.shape;
    return validate_actual_shape(*actual, sizes);
}

/* One loop level of a strided copy: ``destination`` advances by the element
 * size, ``source`` by ``source_step`` elements. */
struct CopyAxis {
    int64_t extent = 1;
    int64_t source_step = 0;
    int64_t dest_step = 0;
};

struct alignas(16) Word128 {
    uint64_t low;
    uint64_t high;
};

template <typename Word>
void copy_axes_fixed(uint8_t* output, const uint8_t* input, const std::vector<CopyAxis>& axes,
                     int64_t begin, int64_t end) {
    const std::size_t outer = axes.size() - 1;
    const int64_t item_size = static_cast<int64_t>(sizeof(Word));
    const CopyAxis inner = axes[outer];
    std::vector<int64_t> coordinates(outer, 0);
    int64_t remainder = begin;
    int64_t source_index = 0;
    int64_t dest_index = 0;
    for (std::size_t reverse = outer; reverse > 0; --reverse) {
        const std::size_t axis = reverse - 1;
        coordinates[axis] = remainder % axes[axis].extent;
        remainder /= axes[axis].extent;
        source_index += coordinates[axis] * axes[axis].source_step;
        dest_index += coordinates[axis] * axes[axis].dest_step;
    }
    const int64_t inner_step = inner.source_step * item_size;
    const int64_t inner_span = inner.extent * item_size;
    for (int64_t index = begin; index < end; ++index) {
        uint8_t* target = output + dest_index * item_size;
        const uint8_t* value = input + source_index * item_size;
        if (inner.source_step == 1) {
            std::memcpy(target, value, static_cast<std::size_t>(inner_span));
        } else if (inner_step == 0) {
            Word first;
            std::memcpy(&first, value, sizeof(Word));
            for (int64_t lane = 0; lane < inner.extent; ++lane) {
                std::memcpy(target + lane * item_size, &first, sizeof(Word));
            }
        } else {
            for (int64_t lane = 0; lane < inner.extent; ++lane) {
                Word element;
                std::memcpy(&element, value + lane * inner_step, sizeof(Word));
                std::memcpy(target + lane * item_size, &element, sizeof(Word));
            }
        }
        if (index + 1 == end) {
            break;
        }
        for (std::size_t reverse = outer; reverse > 0; --reverse) {
            const std::size_t axis = reverse - 1;
            if (++coordinates[axis] < axes[axis].extent) {
                source_index += axes[axis].source_step;
                dest_index += axes[axis].dest_step;
                break;
            }
            coordinates[axis] = 0;
            source_index -= axes[axis].source_step * (axes[axis].extent - 1);
            dest_index -= axes[axis].dest_step * (axes[axis].extent - 1);
        }
    }
}

void copy_axes_generic(uint8_t* output, const uint8_t* input,
                       const std::vector<CopyAxis>& axes, std::size_t item_size, int64_t begin,
                       int64_t end) {
    const std::size_t outer = axes.size() - 1;
    const CopyAxis inner = axes[outer];
    std::vector<int64_t> coordinates(outer, 0);
    int64_t remainder = begin;
    int64_t source_index = 0;
    int64_t dest_index = 0;
    for (std::size_t reverse = outer; reverse > 0; --reverse) {
        const std::size_t axis = reverse - 1;
        coordinates[axis] = remainder % axes[axis].extent;
        remainder /= axes[axis].extent;
        source_index += coordinates[axis] * axes[axis].source_step;
        dest_index += coordinates[axis] * axes[axis].dest_step;
    }
    for (int64_t index = begin; index < end; ++index) {
        uint8_t* target = output + dest_index * static_cast<int64_t>(item_size);
        const uint8_t* value = input + source_index * static_cast<int64_t>(item_size);
        for (int64_t lane = 0; lane < inner.extent; ++lane) {
            std::memcpy(target + lane * static_cast<int64_t>(item_size),
                        value + lane * inner.source_step * static_cast<int64_t>(item_size),
                        item_size);
        }
        if (index + 1 == end) {
            break;
        }
        for (std::size_t reverse = outer; reverse > 0; --reverse) {
            const std::size_t axis = reverse - 1;
            if (++coordinates[axis] < axes[axis].extent) {
                source_index += axes[axis].source_step;
                dest_index += axes[axis].dest_step;
                break;
            }
            coordinates[axis] = 0;
            source_index -= axes[axis].source_step * (axes[axis].extent - 1);
            dest_index -= axes[axis].dest_step * (axes[axis].extent - 1);
        }
    }
}

bool transpose_copy(const uint8_t* source, const std::vector<int64_t>& source_shape,
                    const std::vector<int64_t>& destination_shape,
                    const std::vector<int32_t>& destination_to_source, std::size_t item_size,
                    Buffer* destination) {
    if (source_shape.size() != destination_shape.size() ||
        source_shape.size() != destination_to_source.size() || !valid_shape(source_shape) ||
        !valid_shape(destination_shape)) {
        return false;
    }

    std::vector<bool> seen(source_shape.size(), false);
    for (int32_t source_axis : destination_to_source) {
        if (source_axis < 0 || static_cast<std::size_t>(source_axis) >= source_shape.size() ||
            seen[static_cast<std::size_t>(source_axis)]) {
            return false;
        }
        seen[static_cast<std::size_t>(source_axis)] = true;
    }

    int64_t source_count = 1;
    int64_t destination_count = 1;
    if (!checked_product(source_shape, &source_count) ||
        !checked_product(destination_shape, &destination_count)) {
        return false;
    }

    std::size_t destination_bytes = 0;
    if (!checked_bytes(destination_count, item_size, &destination_bytes)) {
        return false;
    }
    if (destination_bytes == 0) {
        destination->allocate(0);
        return true;
    }
    if (!destination->allocate(destination_bytes) || source == nullptr) {
        return false;
    }
    if (source_count == 0) {
        return false;
    }

    for (std::size_t destination_axis = 0; destination_axis < destination_shape.size();
         ++destination_axis) {
        const std::size_t source_axis =
            static_cast<std::size_t>(destination_to_source[destination_axis]);
        const int64_t source_extent = source_shape[source_axis];
        const int64_t destination_extent = destination_shape[destination_axis];
        if (source_extent != destination_extent && source_extent != 1) {
            return false;
        }
    }

    std::vector<int64_t> strides(source_shape.size(), 1);
    for (std::size_t axis = source_shape.size(); axis > 1; --axis) {
        strides[axis - 2] = strides[axis - 1] * source_shape[axis - 1];
    }

    const std::size_t rank = destination_shape.size();
    std::vector<CopyAxis> raw(rank);
    int64_t dest_step = 1;
    for (std::size_t reverse = rank; reverse > 0; --reverse) {
        const std::size_t axis = reverse - 1;
        const std::size_t source_axis =
            static_cast<std::size_t>(destination_to_source[axis]);
        raw[axis].extent = destination_shape[axis];
        raw[axis].source_step = source_shape[source_axis] == 1 ? 0 : strides[source_axis];
        raw[axis].dest_step = dest_step;
        dest_step *= destination_shape[axis];
    }

    std::vector<CopyAxis> axes;
    axes.reserve(rank);
    for (const CopyAxis& axis : raw) {
        if (axis.extent == 1) {
            continue;
        }
        if (!axes.empty() && axes.back().source_step == axis.source_step * axis.extent) {
            axes.back().extent *= axis.extent;
            axes.back().source_step = axis.source_step;
            axes.back().dest_step = axis.dest_step;
            continue;
        }
        axes.push_back(axis);
    }
    if (axes.empty()) {
        // 所有目标维都是 1：仍需拷贝唯一的那个元素，否则目标缓冲保持未初始化。
        if (destination_count != 1 || source_count != 1) {
            return false;
        }
        std::memcpy(destination->data(), source,
                    static_cast<std::size_t>(item_size));
        return true;
    }
    if (axes.size() > 1 && axes.back().source_step != 1) {
        std::size_t best = axes.size() - 1;
        for (std::size_t axis = 0; axis + 1 < axes.size(); ++axis) {
            if (axes[axis].source_step != 1) {
                continue;
            }
            if (best + 1 == axes.size() || axes[axis].extent > axes[best].extent) {
                best = axis;
            }
        }
        if (best + 1 != axes.size()) {
            std::swap(axes[best], axes[axes.size() - 2]);
        }
    }

    const std::size_t outer = axes.size() - 1;
    int64_t outer_count = 1;
    for (std::size_t axis = 0; axis < outer; ++axis) {
        outer_count *= axes[axis].extent;
    }

    uint8_t* output = destination->data();
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    int threads = 1;
    if (max_threads > 1 && destination_count >= 32768) {
        threads = static_cast<int>(
            std::min<int64_t>(max_threads, std::max<int64_t>(1, destination_count / 65536)));
    }
    const int64_t units = outer == 0 ? 1 : outer_count;
    const int64_t chunk = (units + threads - 1) / threads;

    const auto run_range = [&](int64_t begin, int64_t end) {
        if (item_size == 16) {
            copy_axes_fixed<Word128>(output, source, axes, begin, end);
        } else if (item_size == 8) {
            copy_axes_fixed<uint64_t>(output, source, axes, begin, end);
        } else if (item_size == 4) {
            copy_axes_fixed<uint32_t>(output, source, axes, begin, end);
        } else {
            copy_axes_generic(output, source, axes, item_size, begin, end);
        }
    };

    if (threads > 1 && outer > 0 && units >= 2) {
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int thread = 0; thread < threads; ++thread) {
            const int64_t begin = static_cast<int64_t>(thread) * chunk;
            const int64_t end = std::min(outer_count, begin + chunk);
            if (begin < end) {
                run_range(begin, end);
            }
        }
    } else if (threads > 1 && outer == 0) {
        const int64_t count = axes[0].extent;
        const int64_t step = (count + threads - 1) / threads;
        const int64_t source_step = axes[0].source_step * static_cast<int64_t>(item_size);
        const int64_t element = static_cast<int64_t>(item_size);
        const uint8_t* value = source;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int thread = 0; thread < threads; ++thread) {
            const int64_t begin = static_cast<int64_t>(thread) * step;
            const int64_t end = std::min(count, begin + step);
            if (begin >= end) {
                continue;
            }
            uint8_t* target = output + begin * element;
            if (source_step == 0) {
                for (int64_t lane = begin; lane < end; ++lane) {
                    std::memcpy(output + lane * element, value, item_size);
                }
            } else {
                std::memcpy(target, value + begin * source_step,
                            static_cast<std::size_t>((end - begin) * element));
            }
        }
    } else {
        run_range(0, units);
    }
    return true;
}

bool pack_tensor(const uint8_t* source, const std::vector<int64_t>& source_shape,
                 const std::vector<int64_t>& packed_shape,
                 const std::vector<int32_t>& packed_axes, std::size_t item_size,
                 Buffer* packed) {
    return transpose_copy(source, source_shape, packed_shape, packed_axes, item_size, packed);
}

template <typename Scalar, bool Complex>
void reduce_values(const uint8_t* source, uint8_t* destination,
                   const std::vector<int64_t>& source_shape,
                   const std::vector<int32_t>& output_to_source,
                   const std::vector<int64_t>& output_shape,
                   const std::vector<int32_t>& reduction_axes,
                   const std::vector<int64_t>& reduction_shape, int64_t output_count) {
    std::vector<int64_t> source_strides(source_shape.size(), 1);
    for (std::size_t axis = source_shape.size(); axis > 1; --axis) {
        source_strides[axis - 2] = source_strides[axis - 1] * source_shape[axis - 1];
    }
    int64_t reduction_count = 1;
    checked_product(reduction_shape, &reduction_count);

    for (int64_t output_index = 0; output_index < output_count; ++output_index) {
        std::vector<int64_t> source_coordinates(source_shape.size(), 0);
        int64_t remainder = output_index;
        for (std::size_t reverse = output_shape.size(); reverse > 0; --reverse) {
            const std::size_t output_axis = reverse - 1;
            const int64_t coordinate = remainder % output_shape[output_axis];
            remainder /= output_shape[output_axis];
            const int32_t source_axis = output_to_source[output_axis];
            source_coordinates[static_cast<std::size_t>(source_axis)] =
                source_shape[static_cast<std::size_t>(source_axis)] == 1 ? 0 : coordinate;
        }
        int64_t base_source_index = 0;
        for (std::size_t axis = 0; axis < source_shape.size(); ++axis) {
            base_source_index += source_coordinates[axis] * source_strides[axis];
        }

        for (int64_t reduction_index = 0; reduction_index < reduction_count; ++reduction_index) {
            int64_t reduction_remainder = reduction_index;
            int64_t source_index = base_source_index;
            for (std::size_t reverse = reduction_shape.size(); reverse > 0; --reverse) {
                const std::size_t reduction_axis = reverse - 1;
                const int64_t coordinate = reduction_remainder % reduction_shape[reduction_axis];
                reduction_remainder /= reduction_shape[reduction_axis];
                source_index += coordinate *
                                source_strides[static_cast<std::size_t>(reduction_axes[reduction_axis])];
            }

            const Scalar* source_values = reinterpret_cast<const Scalar*>(
                source + source_index * static_cast<int64_t>(sizeof(Scalar)) * (Complex ? 2 : 1));
            Scalar* destination_values = reinterpret_cast<Scalar*>(
                destination + output_index * static_cast<int64_t>(sizeof(Scalar)) * (Complex ? 2 : 1));
            if constexpr (Complex) {
                destination_values[0] += source_values[0];
                destination_values[1] += source_values[1];
            } else {
                destination_values[0] += source_values[0];
            }
        }
    }
}

bool execute_unary(const SlotData& slot, const TensorMeta& input, const TensorMeta& output,
                   const std::vector<int64_t>& sizes, int32_t dtype_id, SlotData* result) {
    UnaryPlan plan;
    if (!pyqcd_einsum::analyze_unary(input, output, sizes, &plan) ||
        !validate_actual_shape(input, sizes)) {
        return false;
    }

    int64_t source_count = 1;
    int64_t output_count = 1;
    if (!checked_product(input.shape, &source_count) ||
        !checked_product(output.shape, &output_count)) {
        return false;
    }

    Buffer destination;
    if (!allocate_buffer(&destination, output_count, dtype_id)) {
        return false;
    }
    if (output_count == 0) {
        result->owns_storage = true;
        result->storage = std::move(destination);
        result->labels = output.labels;
        result->shape = output.shape;
        result->has_labels = true;
        return true;
    }
    if (source_count != 0 && slot.data() == nullptr) {
        return false;
    }

    std::vector<int32_t> output_to_source(output.labels.size(), -1);
    for (std::size_t output_axis = 0; output_axis < output.labels.size(); ++output_axis) {
        const int64_t source_axis = label_position(input.labels, output.labels[output_axis]);
        if (source_axis < 0) {
            return false;
        }
        output_to_source[output_axis] = static_cast<int32_t>(source_axis);
        const int64_t source_extent = input.shape[static_cast<std::size_t>(source_axis)];
        const int64_t output_extent = output.shape[output_axis];
        if (source_extent != output_extent && source_extent != 1) {
            return false;
        }
    }

    std::vector<int32_t> reduction_axes;
    std::vector<int64_t> reduction_shape;
    for (std::size_t source_axis = 0; source_axis < input.labels.size(); ++source_axis) {
        if (!contains_label(output.labels, input.labels[source_axis])) {
            reduction_axes.push_back(static_cast<int32_t>(source_axis));
            reduction_shape.push_back(input.shape[source_axis]);
        }
    }
    int64_t reduction_count = 1;
    if (!checked_product(reduction_shape, &reduction_count)) {
        return false;
    }
    std::memset(destination.data(), 0,
                static_cast<std::size_t>(output_count) * element_size(dtype_id));
    if (reduction_count != 0) {
        switch (dtype_id) {
            case QCD_DTYPE_COMPLEX128:
                reduce_values<double, true>(slot.data(), destination.data(), input.shape,
                                            output_to_source, output.shape, reduction_axes,
                                            reduction_shape, output_count);
                break;
            case QCD_DTYPE_COMPLEX64:
                reduce_values<float, true>(slot.data(), destination.data(), input.shape,
                                           output_to_source, output.shape, reduction_axes,
                                           reduction_shape, output_count);
                break;
            case QCD_DTYPE_FLOAT64:
                reduce_values<double, false>(slot.data(), destination.data(), input.shape,
                                             output_to_source, output.shape, reduction_axes,
                                             reduction_shape, output_count);
                break;
            case QCD_DTYPE_FLOAT32:
                reduce_values<float, false>(slot.data(), destination.data(), input.shape,
                                            output_to_source, output.shape, reduction_axes,
                                            reduction_shape, output_count);
                break;
            default:
                return false;
        }
    }

    result->owns_storage = true;
    result->storage = std::move(destination);
    result->labels = output.labels;
    result->shape = output.shape;
    result->has_labels = true;
    return true;
}

bool batched_gemm(int32_t dtype_id, const PairPlan& plan, const uint8_t* left,
                  const uint8_t* right, Buffer* output) {
    const CblasApi& api = cblas_api();
    if (!api.complete() || plan.m > std::numeric_limits<int>::max() ||
        plan.n > std::numeric_limits<int>::max() || plan.k > std::numeric_limits<int>::max()) {
        return false;
    }

    int64_t m_k = 1;
    int64_t n_k = 1;
    int64_t m_n = 1;
    if (!checked_mul(plan.m, plan.k, &m_k) || !checked_mul(plan.n, plan.k, &n_k) ||
        !checked_mul(plan.m, plan.n, &m_n)) {
        return false;
    }

    const int m = static_cast<int>(plan.m);
    const int n = static_cast<int>(plan.n);
    const int k = static_cast<int>(plan.k);
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128: {
            const double alpha[2] = {1.0, 0.0};
            const double beta[2] = {0.0, 0.0};
            const double* a = reinterpret_cast<const double*>(left);
            const double* b = reinterpret_cast<const double*>(right);
            double* c = reinterpret_cast<double*>(output->data());
            for (int64_t batch = 0; batch < plan.batches; ++batch) {
                if (api.ilp64) {
                    api.zgemm64(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                                a + 2 * batch * m_k, k, b + 2 * batch * n_k, k, beta,
                                c + 2 * batch * m_n, n);
                } else {
                    api.zgemm(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                              a + 2 * batch * m_k, k, b + 2 * batch * n_k, k, beta,
                              c + 2 * batch * m_n, n);
                }
            }
            return true;
        }
        case QCD_DTYPE_COMPLEX64: {
            const float alpha[2] = {1.0F, 0.0F};
            const float beta[2] = {0.0F, 0.0F};
            const float* a = reinterpret_cast<const float*>(left);
            const float* b = reinterpret_cast<const float*>(right);
            float* c = reinterpret_cast<float*>(output->data());
            for (int64_t batch = 0; batch < plan.batches; ++batch) {
                if (api.ilp64) {
                    api.cgemm64(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                                a + 2 * batch * m_k, k, b + 2 * batch * n_k, k, beta,
                                c + 2 * batch * m_n, n);
                } else {
                    api.cgemm(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                              a + 2 * batch * m_k, k, b + 2 * batch * n_k, k, beta,
                              c + 2 * batch * m_n, n);
                }
            }
            return true;
        }
        case QCD_DTYPE_FLOAT64: {
            const double alpha = 1.0;
            const double beta = 0.0;
            const double* a = reinterpret_cast<const double*>(left);
            const double* b = reinterpret_cast<const double*>(right);
            double* c = reinterpret_cast<double*>(output->data());
            for (int64_t batch = 0; batch < plan.batches; ++batch) {
                if (api.ilp64) {
                    api.dgemm64(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                                a + batch * m_k, k, b + batch * n_k, k, beta,
                                c + batch * m_n, n);
                } else {
                    api.dgemm(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                              a + batch * m_k, k, b + batch * n_k, k, beta,
                              c + batch * m_n, n);
                }
            }
            return true;
        }
        case QCD_DTYPE_FLOAT32: {
            const float alpha = 1.0F;
            const float beta = 0.0F;
            const float* a = reinterpret_cast<const float*>(left);
            const float* b = reinterpret_cast<const float*>(right);
            float* c = reinterpret_cast<float*>(output->data());
            for (int64_t batch = 0; batch < plan.batches; ++batch) {
                if (api.ilp64) {
                    api.sgemm64(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                                a + batch * m_k, k, b + batch * n_k, k, beta,
                                c + batch * m_n, n);
                } else {
                    api.sgemm(kCblasRowMajor, kCblasNoTrans, kCblasTrans, m, n, k, alpha,
                              a + batch * m_k, k, b + batch * n_k, k, beta,
                              c + batch * m_n, n);
                }
            }
            return true;
        }
        default:
            return false;
    }
}

template <typename Real, bool Complex>
void batched_dot_values(const Real* left, const std::vector<int64_t>& left_shape,
                        const std::vector<int32_t>& left_perm,
                        const Real* right, const std::vector<int64_t>& right_shape,
                        const std::vector<int32_t>& right_perm, const PairPlan& plan,
                        Real* output) {
    const int64_t channels = Complex ? 2 : 1;
    const std::size_t batch_rank = plan.batch_labels.size();
    const std::size_t sum_rank = plan.sum_labels.size();
    const std::size_t left_free_rank = left_perm.size() - batch_rank - sum_rank;
    const std::size_t right_free_rank = right_perm.size() - batch_rank - sum_rank;
    const std::size_t left_sum_start = batch_rank + left_free_rank;
    const std::size_t right_sum_start = batch_rank + right_free_rank;
    const auto strides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> result(shape.size(), 1);
        for (std::size_t index = shape.size(); index > 1; --index) {
            result[index - 2] = result[index - 1] * shape[index - 1];
        }
        return result;
    };
    const std::vector<int64_t> left_strides = strides(left_shape);
    const std::vector<int64_t> right_strides = strides(right_shape);
    std::vector<int64_t> sum_coordinates(sum_rank, 0);

    for (int64_t batch = 0; batch < plan.batches; ++batch) {
        int64_t batch_remainder = batch;
        int64_t left_base = 0;
        int64_t right_base = 0;
        for (std::size_t reverse = batch_rank; reverse > 0; --reverse) {
            const std::size_t axis = reverse - 1;
            const int64_t extent = plan.left_packed_shape[axis];
            const int64_t coordinate = batch_remainder % extent;
            batch_remainder /= extent;
            left_base += coordinate * left_strides[
                static_cast<std::size_t>(left_perm[axis])];
            right_base += coordinate * right_strides[
                static_cast<std::size_t>(right_perm[axis])];
        }
        std::fill(sum_coordinates.begin(), sum_coordinates.end(), 0);
        int64_t left_sum_index = 0;
        int64_t right_sum_index = 0;
        Real real = Real(0);
        Real imag = Real(0);
        for (int64_t index = 0; index < plan.k; ++index) {
            const int64_t left_offset = (left_base + left_sum_index) * channels;
            const int64_t right_offset = (right_base + right_sum_index) * channels;
            if constexpr (Complex) {
                const Real lr = left[left_offset];
                const Real li = left[left_offset + 1];
                const Real rr = right[right_offset];
                const Real ri = right[right_offset + 1];
                real += lr * rr - li * ri;
                imag += lr * ri + li * rr;
            } else {
                real += left[left_offset] * right[right_offset];
            }
            if (index + 1 == plan.k) {
                break;
            }
            for (std::size_t reverse = sum_rank; reverse > 0; --reverse) {
                const std::size_t axis = reverse - 1;
                const int64_t extent = plan.left_packed_shape[left_sum_start + axis];
                if (++sum_coordinates[axis] < extent) {
                    left_sum_index += left_strides[
                        static_cast<std::size_t>(left_perm[left_sum_start + axis])];
                    right_sum_index += right_strides[
                        static_cast<std::size_t>(right_perm[right_sum_start + axis])];
                    break;
                }
                sum_coordinates[axis] = 0;
                left_sum_index -= left_strides[
                    static_cast<std::size_t>(left_perm[left_sum_start + axis])] * (extent - 1);
                right_sum_index -= right_strides[
                    static_cast<std::size_t>(right_perm[right_sum_start + axis])] * (extent - 1);
            }
        }
        output[batch * channels] = real;
        if constexpr (Complex) {
            output[batch * channels + 1] = imag;
        }
    }
}

bool execute_pair(const SlotData& left_slot, const SlotData& right_slot, const TensorMeta& left,
                  const TensorMeta& right, const TensorMeta& output,
                  const std::vector<int64_t>& sizes, int32_t dtype_id, SlotData* result) {
    PairPlan plan;
    const auto total_start = std::chrono::steady_clock::now();
    if (!validate_actual_shape(left, sizes) || !validate_actual_shape(right, sizes)) {
        return false;
    }

    const auto pack_score = [](const PairPlan& candidate) {
        return static_cast<int>(candidate.left_is_packed) +
               static_cast<int>(candidate.right_is_packed) +
               static_cast<int>(candidate.output_is_packed);
    };
    PairPlan direct_plan;
    const bool direct_ok = pyqcd_einsum::analyze_pair(left, right, output, sizes, &direct_plan);
    PairPlan swapped_plan;
    const bool swapped_ok = pyqcd_einsum::analyze_pair(right, left, output, sizes, &swapped_plan);
    bool swapped = false;
    if (direct_ok && swapped_ok) {
        swapped = pack_score(swapped_plan) < pack_score(direct_plan);
    } else if (!direct_ok && swapped_ok) {
        swapped = true;
    } else if (!direct_ok) {
        return false;
    }
    plan = swapped ? std::move(swapped_plan) : std::move(direct_plan);
    const SlotData& selected_left_slot = swapped ? right_slot : left_slot;
    const SlotData& selected_right_slot = swapped ? left_slot : right_slot;
    const TensorMeta& selected_left = swapped ? right : left;
    const TensorMeta& selected_right = swapped ? left : right;

    int64_t left_count = 1;
    int64_t right_count = 1;
    int64_t canonical_count = 1;
    int64_t output_count = 1;
    if (!checked_product(plan.left_packed_shape, &left_count) ||
        !checked_product(plan.right_packed_shape, &right_count) ||
        !checked_product(plan.output_packed_shape, &canonical_count) ||
        !checked_product(output.shape, &output_count) || canonical_count != output_count) {
        return false;
    }

    Buffer packed_left;
    Buffer packed_right;
    Buffer canonical;
    if (!allocate_buffer(&canonical, canonical_count, dtype_id)) {
        return false;
    }
    if (canonical_count != 0 && plan.k == 0) {
        std::memset(canonical.data(), 0,
                    static_cast<std::size_t>(canonical_count) * element_size(dtype_id));
    }
    const uint8_t* left_data = selected_left_slot.data();
    const uint8_t* right_data = selected_right_slot.data();
    if (plan.m == 1 && plan.n == 1) {
        if (canonical_count != 0 && plan.k != 0) {
            if (left_data == nullptr || right_data == nullptr) {
                return false;
            }
            switch (dtype_id) {
                case QCD_DTYPE_COMPLEX128:
                    batched_dot_values<double, true>(
                        reinterpret_cast<const double*>(left_data), selected_left.shape,
                        plan.left_perm, reinterpret_cast<const double*>(right_data),
                        selected_right.shape, plan.right_perm, plan,
                        reinterpret_cast<double*>(canonical.data()));
                    break;
                case QCD_DTYPE_COMPLEX64:
                    batched_dot_values<float, true>(
                        reinterpret_cast<const float*>(left_data), selected_left.shape,
                        plan.left_perm, reinterpret_cast<const float*>(right_data),
                        selected_right.shape, plan.right_perm, plan,
                        reinterpret_cast<float*>(canonical.data()));
                    break;
                case QCD_DTYPE_FLOAT64:
                    batched_dot_values<double, false>(
                        reinterpret_cast<const double*>(left_data), selected_left.shape,
                        plan.left_perm, reinterpret_cast<const double*>(right_data),
                        selected_right.shape, plan.right_perm, plan,
                        reinterpret_cast<double*>(canonical.data()));
                    break;
                case QCD_DTYPE_FLOAT32:
                    batched_dot_values<float, false>(
                        reinterpret_cast<const float*>(left_data), selected_left.shape,
                        plan.left_perm, reinterpret_cast<const float*>(right_data),
                        selected_right.shape, plan.right_perm, plan,
                        reinterpret_cast<float*>(canonical.data()));
                    break;
                default:
                    return false;
            }
        }
    } else {
        if (plan.left_is_packed) {
            if (!allocate_buffer(&packed_left, left_count, dtype_id) ||
                !pack_tensor(selected_left_slot.data(), selected_left.shape,
                             plan.left_packed_shape, plan.left_perm,
                             element_size(dtype_id), &packed_left)) {
                return false;
            }
            left_data = packed_left.data();
        }
        if (plan.right_is_packed) {
            if (!allocate_buffer(&packed_right, right_count, dtype_id) ||
                !pack_tensor(selected_right_slot.data(), selected_right.shape,
                             plan.right_packed_shape, plan.right_perm,
                             element_size(dtype_id), &packed_right)) {
                return false;
            }
            right_data = packed_right.data();
        }
        const auto pack_done = std::chrono::steady_clock::now();
        if (canonical_count != 0 && plan.k != 0) {
            if (left_data == nullptr || right_data == nullptr || canonical.data() == nullptr ||
                !batched_gemm(dtype_id, plan, left_data, right_data, &canonical)) {
                return false;
            }
        }
        const auto gemm_done = std::chrono::steady_clock::now();
        if (trace_enabled()) {
            std::fprintf(
                stderr,
                "PYQCD_EINSUM_KUPL_GEMM batch=%lld m=%lld n=%lld k=%lld "
                "pack_left=%d pack_right=%d pack_out=%d pack_ms=%.3f gemm_ms=%.3f\n",
                static_cast<long long>(plan.batches), static_cast<long long>(plan.m),
                static_cast<long long>(plan.n), static_cast<long long>(plan.k),
                plan.left_is_packed ? 1 : 0, plan.right_is_packed ? 1 : 0,
                plan.output_is_packed ? 1 : 0,
                std::chrono::duration<double, std::milli>(pack_done - total_start).count(),
                std::chrono::duration<double, std::milli>(gemm_done - pack_done).count());
        }
    }

    result->owns_storage = true;
    if (plan.output_is_packed) {
        Buffer destination;
        if (!transpose_copy(canonical.data(), plan.output_packed_shape, output.shape,
                            plan.output_perm, element_size(dtype_id), &destination)) {
            return false;
        }
        result->storage = std::move(destination);
    } else {
        result->storage = std::move(canonical);
    }
    result->labels = output.labels;
    result->shape = output.shape;
    result->has_labels = true;
    if (trace_enabled()) {
        std::fprintf(
            stderr,
            "PYQCD_EINSUM_KUPL_PAIR batch=%lld m=%lld n=%lld k=%lld "
            "pack_left=%d pack_right=%d pack_out=%d pack_ms=%.3f gemm_ms=%.3f total_ms=%.3f\n",
            static_cast<long long>(plan.batches), static_cast<long long>(plan.m),
            static_cast<long long>(plan.n), static_cast<long long>(plan.k),
            plan.left_is_packed ? 1 : 0, plan.right_is_packed ? 1 : 0,
            plan.output_is_packed ? 1 : 0,
            0.0, 0.0,
            elapsed_ms(total_start));
        std::fprintf(stderr, "  left_shape=");
        for (int64_t value : selected_left.shape) {
            std::fprintf(stderr, "%lld,", static_cast<long long>(value));
        }
        std::fprintf(stderr, " left_perm=");
        for (int32_t value : plan.left_perm) {
            std::fprintf(stderr, "%d,", value);
        }
        std::fprintf(stderr, " right_shape=");
        for (int64_t value : selected_right.shape) {
            std::fprintf(stderr, "%lld,", static_cast<long long>(value));
        }
        std::fprintf(stderr, " right_perm=");
        for (int32_t value : plan.right_perm) {
            std::fprintf(stderr, "%d,", value);
        }
        std::fprintf(stderr, " output_shape=");
        for (int64_t value : output.shape) {
            std::fprintf(stderr, "%lld,", static_cast<long long>(value));
        }
        std::fprintf(stderr, " output_perm=");
        for (int32_t value : plan.output_perm) {
            std::fprintf(stderr, "%d,", value);
        }
        std::fprintf(stderr, "\n");
    }
    return true;
}

bool valid_positions(const ExecutionStep& step, std::size_t active_count,
                     std::vector<int32_t>* descending) {
    *descending = step.positions;
    std::sort(descending->begin(), descending->end());
    if (std::adjacent_find(descending->begin(), descending->end()) != descending->end()) {
        return false;
    }
    for (int32_t position : *descending) {
        if (position < 0 || static_cast<std::size_t>(position) >= active_count) {
            return false;
        }
    }
    std::reverse(descending->begin(), descending->end());
    return true;
}

/* ------------------------------------------------------------------------- */
/* Gamma-sandwich fast path: ``ab,...bce,cd->...dae``                        */
/*                                                                           */
/* Two different contraction orders reach this shape in practice:            */
/*   * ``cd,...bce->...dbe`` first (the right matrix removes ``c``; this is    */
/*     what the ``dp`` planner used by the benchmark emits),                   */
/*   * ``ab,...bce->...cea`` first (the left matrix removes ``b``; the         */
/*     ``greedy`` planner emits this).                                        */
/* Both are algebraically the same two 4x4 matrix sandwiches, so the fused     */
/* kernel ignores the intermediate completely: it reads the propagation        */
/* tensor in its own layout ``[prefix][b][c][e]`` and writes the final         */
/* ``[prefix][d][a][e]`` output.  Matching therefore inspects operand roles    */
/* and the final output ordering only, mirroring the CUDA matcher.             */
/* ------------------------------------------------------------------------- */

struct SandwichGeometry {
    int32_t tensor_position = -1;
    /* gamma that contracts ``b`` and contributes ``a``. */
    int32_t left_matrix_position = -1;
    /* gamma that contracts ``c`` and contributes ``d``. */
    int32_t right_matrix_position = -1;
    std::vector<int64_t> tensor_shape;
    std::vector<int64_t> left_matrix_shape;
    std::vector<int64_t> right_matrix_shape;
    int64_t prefix = 1;
    int64_t left = 0;              /* |a| */
    int64_t contracted_left = 0;   /* |b| */
    int64_t contracted_right = 0;  /* |c| */
    int64_t right = 0;             /* |d| */
    int64_t lane = 0;              /* |e| */
};

bool same_label_set(const std::vector<int32_t>& left, const std::vector<int32_t>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (int32_t label : left) {
        if (std::find(right.begin(), right.end(), label) == right.end()) {
            return false;
        }
    }
    return true;
}

bool sandwich_extent(const std::vector<int64_t>& sizes, int32_t label, int64_t* extent) {
    if (label < 0 || static_cast<std::size_t>(label) >= sizes.size()) {
        return false;
    }
    const int64_t value = sizes[static_cast<std::size_t>(label)];
    if (value <= 0) {
        return false;
    }
    *extent = value;
    return true;
}

bool match_sandwich(const ExecutionPlan& execution, SandwichGeometry* geometry) {
    if (execution.num_inputs != 3 || execution.steps.size() != 2) {
        return false;
    }
    const ExecutionStep& first = execution.steps[0];
    const ExecutionStep& second = execution.steps[1];
    if (first.positions.size() != 2 || first.inputs.size() != 2 ||
        second.positions.size() != 2 || second.inputs.size() != 2) {
        return false;
    }

    for (int tensor_operand = 0; tensor_operand < 2; ++tensor_operand) {
        const int matrix_operand = 1 - tensor_operand;
        const TensorMeta& tensor = first.inputs[static_cast<std::size_t>(tensor_operand)];
        const TensorMeta& first_matrix = first.inputs[static_cast<std::size_t>(matrix_operand)];
        if (tensor.labels.size() < 3 || first_matrix.labels.size() != 2) {
            continue;
        }
        const std::size_t prefix_rank = tensor.labels.size() - 3;
        const int32_t b = tensor.labels[prefix_rank];
        const int32_t c = tensor.labels[prefix_rank + 1];
        const int32_t e = tensor.labels[prefix_rank + 2];

        int32_t a = -1;
        int32_t d = -1;
        bool left_matrix_first = false;
        if (first_matrix.labels[1] == b) {
            a = first_matrix.labels[0];
            left_matrix_first = true;
        } else if (first_matrix.labels[0] == c) {
            d = first_matrix.labels[1];
            left_matrix_first = false;
        } else {
            continue;
        }

        /* The second step must consume the intermediate plus the other gamma. */
        int intermediate_operand = -1;
        int remaining_matrix_operand = -1;
        bool unique = true;
        for (int operand = 0; operand < 2; ++operand) {
            const bool is_intermediate =
                second.inputs[static_cast<std::size_t>(operand)].labels == first.output.labels;
            int* slot = is_intermediate ? &intermediate_operand : &remaining_matrix_operand;
            if (*slot >= 0) {
                unique = false;
                break;
            }
            *slot = operand;
        }
        if (!unique || intermediate_operand < 0 || remaining_matrix_operand < 0) {
            continue;
        }
        const TensorMeta& other_matrix = second.inputs[static_cast<std::size_t>(remaining_matrix_operand)];
        if (other_matrix.labels.size() != 2) {
            continue;
        }
        if (left_matrix_first) {
            if (other_matrix.labels[0] != c) {
                continue;
            }
            d = other_matrix.labels[1];
        } else {
            if (other_matrix.labels[1] != b) {
                continue;
            }
            a = other_matrix.labels[0];
        }

        /* The first step has to remove exactly ``b`` (left matrix first) or ``c``
         * and keep everything else; the check rejects matrices that contract a
         * prefix axis. */
        std::vector<int32_t> prefix_labels(tensor.labels.begin(),
                                           tensor.labels.begin() +
                                               static_cast<std::ptrdiff_t>(prefix_rank));
        std::vector<int32_t> expected_first_output = prefix_labels;
        if (left_matrix_first) {
            expected_first_output.push_back(c);
            expected_first_output.push_back(e);
            expected_first_output.push_back(a);
        } else {
            expected_first_output.push_back(d);
            expected_first_output.push_back(b);
            expected_first_output.push_back(e);
        }
        if (!same_label_set(first.output.labels, expected_first_output)) {
            continue;
        }

        /* The final output has to be exactly ``[prefix..., d, a, e]`` because the
         * fused kernel writes that layout directly. */
        std::vector<int32_t> expected_output = prefix_labels;
        expected_output.push_back(d);
        expected_output.push_back(a);
        expected_output.push_back(e);
        if (second.output.labels != expected_output ||
            execution.output_labels != expected_output) {
            continue;
        }
        std::vector<int32_t> roles = expected_output;
        roles.push_back(b);
        roles.push_back(c);
        std::sort(roles.begin(), roles.end());
        if (std::adjacent_find(roles.begin(), roles.end()) != roles.end()) {
            continue;
        }

        /* Live slots after the first step: the survivor keeps the lowest index and
         * the freshly produced intermediate is appended last. */
        if (second.positions[static_cast<std::size_t>(intermediate_operand)] != 1 ||
            second.positions[static_cast<std::size_t>(remaining_matrix_operand)] != 0) {
            continue;
        }
        const int32_t tensor_position = first.positions[static_cast<std::size_t>(tensor_operand)];
        const int32_t first_matrix_position =
            first.positions[static_cast<std::size_t>(matrix_operand)];
        if (tensor_position < 0 || tensor_position >= execution.num_inputs ||
            first_matrix_position < 0 || first_matrix_position >= execution.num_inputs ||
            tensor_position == first_matrix_position) {
            continue;
        }
        int32_t remaining_matrix_input = -1;
        bool remaining_unique = true;
        for (int32_t input = 0; input < execution.num_inputs; ++input) {
            if (input != tensor_position && input != first_matrix_position) {
                if (remaining_matrix_input >= 0) {
                    remaining_unique = false;
                    break;
                }
                remaining_matrix_input = input;
            }
        }
        if (!remaining_unique || remaining_matrix_input < 0) {
            continue;
        }

        SandwichGeometry candidate;
        candidate.tensor_position = tensor_position;
        candidate.left_matrix_position =
            left_matrix_first ? first_matrix_position : remaining_matrix_input;
        candidate.right_matrix_position =
            left_matrix_first ? remaining_matrix_input : first_matrix_position;
        candidate.tensor_shape = tensor.shape;
        candidate.left_matrix_shape = first_matrix.shape;
        candidate.right_matrix_shape = other_matrix.shape;
        if (!left_matrix_first) {
            std::swap(candidate.left_matrix_shape, candidate.right_matrix_shape);
        }
        if (!sandwich_extent(execution.sizes, a, &candidate.left) ||
            !sandwich_extent(execution.sizes, b, &candidate.contracted_left) ||
            !sandwich_extent(execution.sizes, c, &candidate.contracted_right) ||
            !sandwich_extent(execution.sizes, d, &candidate.right) ||
            !sandwich_extent(execution.sizes, e, &candidate.lane)) {
            continue;
        }
        bool ok = true;
        candidate.prefix = pyqcd_einsum::checked_product(
            std::vector<int64_t>(tensor.shape.begin(),
                                 tensor.shape.begin() + static_cast<std::ptrdiff_t>(prefix_rank)),
            &ok);
        if (!ok || candidate.prefix <= 0) {
            continue;
        }
        const std::vector<int64_t>& tensor_shape = candidate.tensor_shape;
        if (tensor_shape.size() != prefix_rank + 3 ||
            tensor_shape[prefix_rank] != candidate.contracted_left ||
            tensor_shape[prefix_rank + 1] != candidate.contracted_right ||
            tensor_shape[prefix_rank + 2] != candidate.lane) {
            continue;
        }
        if (candidate.left_matrix_shape.size() != 2 ||
            candidate.left_matrix_shape[0] != candidate.left ||
            candidate.left_matrix_shape[1] != candidate.contracted_left ||
            candidate.right_matrix_shape.size() != 2 ||
            candidate.right_matrix_shape[0] != candidate.contracted_right ||
            candidate.right_matrix_shape[1] != candidate.right) {
            continue;
        }
        /* Only the 4x4 gamma sandwich has a dedicated kernel. */
        if (candidate.left != 4 || candidate.contracted_left != 4 ||
            candidate.contracted_right != 4 || candidate.right != 4) {
            continue;
        }
        *geometry = std::move(candidate);
        return true;
    }
    return false;
}

constexpr int kSandwichSize = 4;
/* Complex lanes handled by one inner block: 8 complexes = 128 B per row. */
constexpr int kSandwichBlock = 8;
/* Prefix counts below this stay serial to avoid OpenMP start-up cost. */
constexpr int64_t kSandwichParallelPrefix = 16;

/* ``Complex`` means ``Real`` scalars interleaved as (real, imaginary) pairs. */
template <typename Real, bool Complex>
void sandwich4_values(const Real* __restrict__ tensor,
                      const Real* __restrict__ left_matrix,
                      const Real* __restrict__ right_matrix,
                      Real* __restrict__ output, const SandwichGeometry& geometry) {
    constexpr int kSize = kSandwichSize;
    constexpr int kBlock = kSandwichBlock;
    constexpr int kChannels = Complex ? 2 : 1;
    constexpr int kMatrixValues = kSize * kSize;
    constexpr int kBlockValues = kMatrixValues * kBlock;

    Real left_real[kMatrixValues];
    Real left_imag[kMatrixValues];
    Real right_real[kMatrixValues];
    Real right_imag[kMatrixValues];
    for (int index = 0; index < kMatrixValues; ++index) {
        left_real[index] = left_matrix[index * kChannels];
        right_real[index] = right_matrix[index * kChannels];
        if constexpr (Complex) {
            left_imag[index] = left_matrix[index * kChannels + 1];
            right_imag[index] = right_matrix[index * kChannels + 1];
        } else {
            left_imag[index] = Real(0);
            right_imag[index] = Real(0);
        }
    }

    const int64_t lane = geometry.lane;
    /* Distance between two adjacent (b, c) rows of one prefix block. */
    const int64_t row_stride = lane * kChannels;
    const int64_t prefix_stride = static_cast<int64_t>(kMatrixValues) * row_stride;
    const int64_t block_count = (lane + kBlock - 1) / kBlock;

#pragma omp parallel for schedule(static) if(geometry.prefix >= kSandwichParallelPrefix)
    for (int64_t prefix = 0; prefix < geometry.prefix; ++prefix) {
        /* ``middle[b * 4 + d]`` holds sum_c M[b][c] * gamma2[c][d] for one lane block. */
        Real middle_real[kBlockValues];
        Real middle_imag[kBlockValues];
        for (int64_t block = 0; block < block_count; ++block) {
            const int64_t lane_start = block * kBlock;
            const int lanes = static_cast<int>(std::min<int64_t>(kBlock, lane - lane_start));
            const Real* tensor_block = tensor + prefix * prefix_stride + lane_start * kChannels;
            Real* output_block = output + prefix * prefix_stride + lane_start * kChannels;

            /* Stage 1: middle[b][d] = sum_c M[b][c] * gamma2[c][d]. */
            for (int first = 0; first < kSize; ++first) {
                Real row_real[kSize][kBlock];
                Real row_imag[kSize][kBlock];
                for (int contracted = 0; contracted < kSize; ++contracted) {
                    const Real* source =
                        tensor_block + (first * kSize + contracted) * row_stride;
                    for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                        row_real[contracted][lane_index] = source[lane_index * kChannels];
                        if constexpr (Complex) {
                            row_imag[contracted][lane_index] =
                                source[lane_index * kChannels + 1];
                        }
                    }
                }
                for (int last = 0; last < kSize; ++last) {
                    Real* target_real = middle_real + (first * kSize + last) * kBlock;
                    Real* target_imag = middle_imag + (first * kSize + last) * kBlock;
                    for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                        target_real[lane_index] = Real(0);
                        target_imag[lane_index] = Real(0);
                    }
                    for (int contracted = 0; contracted < kSize; ++contracted) {
                        const Real scale_real = right_real[contracted * kSize + last];
                        const Real scale_imag = right_imag[contracted * kSize + last];
                        const Real* source_real = row_real[contracted];
                        const Real* source_imag = row_imag[contracted];
                        if constexpr (Complex) {
#pragma omp simd
                            for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                                target_real[lane_index] +=
                                    source_real[lane_index] * scale_real -
                                    source_imag[lane_index] * scale_imag;
                                target_imag[lane_index] +=
                                    source_real[lane_index] * scale_imag +
                                    source_imag[lane_index] * scale_real;
                            }
                        } else {
#pragma omp simd
                            for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                                target_real[lane_index] +=
                                    source_real[lane_index] * scale_real;
                            }
                        }
                    }
                }
            }

            /* Stage 2: out[d][a] = sum_b gamma1[a][b] * middle[b][d]. */
            for (int last = 0; last < kSize; ++last) {
                for (int first = 0; first < kSize; ++first) {
                    Real accumulator_real[kBlock];
                    Real accumulator_imag[kBlock];
                    for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                        accumulator_real[lane_index] = Real(0);
                        accumulator_imag[lane_index] = Real(0);
                    }
                    for (int middle = 0; middle < kSize; ++middle) {
                        const Real scale_real = left_real[first * kSize + middle];
                        const Real scale_imag = left_imag[first * kSize + middle];
                        const Real* source_real =
                            middle_real + (middle * kSize + last) * kBlock;
                        const Real* source_imag =
                            middle_imag + (middle * kSize + last) * kBlock;
                        if constexpr (Complex) {
#pragma omp simd
                            for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                                accumulator_real[lane_index] +=
                                    source_real[lane_index] * scale_real -
                                    source_imag[lane_index] * scale_imag;
                                accumulator_imag[lane_index] +=
                                    source_real[lane_index] * scale_imag +
                                    source_imag[lane_index] * scale_real;
                            }
                        } else {
#pragma omp simd
                            for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                                accumulator_real[lane_index] +=
                                    source_real[lane_index] * scale_real;
                            }
                        }
                    }
                    Real* destination = output_block + (last * kSize + first) * row_stride;
                    for (int lane_index = 0; lane_index < lanes; ++lane_index) {
                        destination[lane_index * kChannels] = accumulator_real[lane_index];
                        if constexpr (Complex) {
                            destination[lane_index * kChannels + 1] =
                                accumulator_imag[lane_index];
                        }
                    }
                }
            }
        }
    }
}

enum class FastPathResult { not_matched, success, failure };

// specialize=False 时由 Python 侧设置该变量，强制走通用多步路径以便 A/B 对照。
bool fused_paths_disabled() {
    // 不缓存：A/B 对照会在同一进程内反复切换该变量。
    return std::getenv("PYQCD_EINSUM_DISABLE_FUSED") != nullptr;
}

FastPathResult try_sandwich_fast_path(const ExecutionPlan& execution,
                                      const std::vector<SlotData>& active,
                                      uint8_t* output) {
    SandwichGeometry geometry;
    if (fused_paths_disabled() || !match_sandwich(execution, &geometry)) {
        return FastPathResult::not_matched;
    }
    const auto lookup = [&](int32_t position) -> const SlotData* {
        if (position < 0 || static_cast<std::size_t>(position) >= active.size()) {
            return nullptr;
        }
        return &active[static_cast<std::size_t>(position)];
    };
    const SlotData* tensor = lookup(geometry.tensor_position);
    const SlotData* left = lookup(geometry.left_matrix_position);
    const SlotData* right = lookup(geometry.right_matrix_position);
    if (tensor == nullptr || left == nullptr || right == nullptr || output == nullptr) {
        return FastPathResult::failure;
    }
    /* Only raw, exactly shaped operands use the fused kernel; broadcast (size-1)
     * or already materialised operands fall back to pack/GEMM. */
    if (tensor->has_labels || left->has_labels || right->has_labels ||
        tensor->data() == nullptr || left->data() == nullptr || right->data() == nullptr ||
        tensor->shape != geometry.tensor_shape || left->shape != geometry.left_matrix_shape ||
        right->shape != geometry.right_matrix_shape) {
        return FastPathResult::not_matched;
    }
    const auto start = std::chrono::steady_clock::now();
    switch (execution.dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            sandwich4_values<double, true>(reinterpret_cast<const double*>(tensor->data()),
                                           reinterpret_cast<const double*>(left->data()),
                                           reinterpret_cast<const double*>(right->data()),
                                           reinterpret_cast<double*>(output), geometry);
            break;
        case QCD_DTYPE_COMPLEX64:
            sandwich4_values<float, true>(reinterpret_cast<const float*>(tensor->data()),
                                          reinterpret_cast<const float*>(left->data()),
                                          reinterpret_cast<const float*>(right->data()),
                                          reinterpret_cast<float*>(output), geometry);
            break;
        case QCD_DTYPE_FLOAT64:
            sandwich4_values<double, false>(reinterpret_cast<const double*>(tensor->data()),
                                            reinterpret_cast<const double*>(left->data()),
                                            reinterpret_cast<const double*>(right->data()),
                                            reinterpret_cast<double*>(output), geometry);
            break;
        case QCD_DTYPE_FLOAT32:
            sandwich4_values<float, false>(reinterpret_cast<const float*>(tensor->data()),
                                           reinterpret_cast<const float*>(left->data()),
                                           reinterpret_cast<const float*>(right->data()),
                                           reinterpret_cast<float*>(output), geometry);
            break;
        default:
            return FastPathResult::failure;
    }
    if (trace_enabled()) {
        std::fprintf(stderr,
                     "PYQCD_EINSUM_KUPL_SANDWICH prefix=%lld left=%lld middle=%lld "
                     "right=%lld lane=%lld kernel=1 total_ms=%.3f\n",
                     static_cast<long long>(geometry.prefix),
                     static_cast<long long>(geometry.left),
                     static_cast<long long>(geometry.contracted_left),
                     static_cast<long long>(geometry.right),
                     static_cast<long long>(geometry.lane), elapsed_ms(start));
    }
    return FastPathResult::success;
}

}  // namespace

namespace pyqcd_einsum {

int execute_parsed_contract(const ExecutionPlan& execution, long long tensors_address,
                            long long out_address) {
    const int64_t* tensor_addresses =
        as_const_pointer<int64_t>(static_cast<int64_t>(tensors_address));
    int64_t* output_addresses = as_pointer<int64_t>(static_cast<int64_t>(out_address));
    if (tensor_addresses == nullptr || output_addresses == nullptr) {
        return QCD_STATUS_INVALID;
    }
    if (execution.num_inputs <= 0 || execution.steps.empty() || scalar_size(execution.dtype_id) == 0) {
        return QCD_STATUS_INVALID;
    }

    std::vector<SlotData> active(static_cast<std::size_t>(execution.num_inputs));
    for (int32_t input_index = 0; input_index < execution.num_inputs; ++input_index) {
        const int64_t input_address = tensor_addresses[input_index];
        SlotData& slot = active[static_cast<std::size_t>(input_index)];
        slot.external = as_const_pointer<uint8_t>(input_address);
        slot.shape = execution.input_shapes[static_cast<std::size_t>(input_index)];
        if (!valid_shape(slot.shape)) {
            return QCD_STATUS_INVALID;
        }
        int64_t count = 1;
        if (!::checked_product(slot.shape, &count)) {
            return QCD_STATUS_INVALID;
        }
        if (count != 0 && slot.external == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }

    uint8_t* final_destination = as_pointer<uint8_t>(output_addresses[0]);
    const FastPathResult fast_result =
        try_sandwich_fast_path(execution, active, final_destination);
    if (fast_result == FastPathResult::success) {
        return QCD_STATUS_OK;
    }
    if (fast_result == FastPathResult::failure) {
        return QCD_STATUS_INVALID;
    }

    for (std::size_t step_index = 0; step_index < execution.steps.size(); ++step_index) {
        const ExecutionStep& step = execution.steps[step_index];
        std::vector<int32_t> descending_positions;
        if (!valid_positions(step, active.size(), &descending_positions) ||
            !shape_matches_symbols(step.output, execution.sizes)) {
            return QCD_STATUS_INVALID;
        }

        TensorMeta actual_input;
        SlotData intermediate;
        if (step.positions.size() == 1) {
            const SlotData& source = active[static_cast<std::size_t>(step.positions[0])];
            if (!resolve_input(source, step.inputs[0], execution.sizes, &actual_input) ||
                !execute_unary(source, actual_input, step.output, execution.sizes,
                               execution.dtype_id, &intermediate)) {
                return QCD_STATUS_INVALID;
            }
        } else {
            const SlotData& left = active[static_cast<std::size_t>(step.positions[0])];
            const SlotData& right = active[static_cast<std::size_t>(step.positions[1])];
            TensorMeta actual_left;
            TensorMeta actual_right;
            if (!resolve_input(left, step.inputs[0], execution.sizes, &actual_left) ||
                !resolve_input(right, step.inputs[1], execution.sizes, &actual_right) ||
                !execute_pair(left, right, actual_left, actual_right, step.output,
                              execution.sizes, execution.dtype_id, &intermediate)) {
                return QCD_STATUS_INVALID;
            }
        }

        for (int32_t position : descending_positions) {
            active.erase(active.begin() + position);
        }
        active.push_back(std::move(intermediate));
    }

    if (active.size() != 1 || !active[0].has_labels ||
        !tensors_match(active[0].labels, execution.output_labels)) {
        return QCD_STATUS_INVALID;
    }

    TensorMeta final_output;
    final_output.labels = execution.output_labels;
    final_output.shape.resize(final_output.labels.size());
    for (std::size_t axis = 0; axis < final_output.labels.size(); ++axis) {
        final_output.shape[axis] =
            execution.sizes[static_cast<std::size_t>(final_output.labels[axis])];
    }
    if (!tensors_match(active[0].labels, final_output.labels) ||
        active[0].shape != final_output.shape) {
        return QCD_STATUS_INVALID;
    }

    int64_t final_count = 1;
    std::size_t final_bytes = 0;
    if (!::checked_product(final_output.shape, &final_count) ||
        !checked_bytes(final_count, element_size(execution.dtype_id), &final_bytes)) {
        return QCD_STATUS_INVALID;
    }
    uint8_t* destination = final_destination;
    if (final_bytes != 0 && (destination == nullptr || active[0].data() == nullptr)) {
        return QCD_STATUS_INVALID;
    }
    if (final_bytes != 0) {
        std::memmove(destination, active[0].data(), final_bytes);
    }
    return QCD_STATUS_OK;
}

}  // namespace pyqcd_einsum

namespace {

int execute_contract(long long plan_address, long long tensors_address, long long out_address,
                     long long params_address) {
    const void* serialized = as_const_pointer<void>(static_cast<int64_t>(plan_address));
    const int32_t* params = as_const_pointer<int32_t>(static_cast<int64_t>(params_address));
    if (serialized == nullptr || params == nullptr) {
        return QCD_STATUS_INVALID;
    }
    (void)params;
    ExecutionPlan execution;
    std::string error;
    if (!pyqcd_einsum::parse_execution_plan(serialized, &execution, &error)) {
        return QCD_STATUS_INVALID;
    }
    return execute_parsed_contract(execution, tensors_address, out_address);
}

struct CompiledContext {
    uint64_t magic = 0x5143454b55504c58ULL;  // "QCEKUPLX", distinct from CUDA.
    ExecutionPlan execution;
};

std::shared_mutex context_lifecycle_mutex;

int compile_contract(long long plan_address, long long params_address, long long out_context) {
    const void* serialized = as_const_pointer<void>(static_cast<int64_t>(plan_address));
    const int32_t* params = as_const_pointer<int32_t>(static_cast<int64_t>(params_address));
    int64_t* destination = as_pointer<int64_t>(static_cast<int64_t>(out_context));
    if (serialized == nullptr || params == nullptr || destination == nullptr) {
        return QCD_STATUS_INVALID;
    }
    (void)params;
    auto context = std::make_unique<CompiledContext>();
    std::string error;
    if (!pyqcd_einsum::parse_execution_plan(serialized, &context->execution, &error) ||
        context->execution.num_inputs <= 0 || context->execution.steps.empty()) {
        return QCD_STATUS_INVALID;
    }
    *destination = static_cast<int64_t>(reinterpret_cast<std::intptr_t>(context.release()));
    return QCD_STATUS_OK;
}

int execute_compiled(long long context_address, long long tensors_address, long long out_address,
                     long long params_address) {
    std::shared_lock<std::shared_mutex> lifecycle(context_lifecycle_mutex);
    auto* context = as_pointer<CompiledContext>(static_cast<int64_t>(context_address));
    const int32_t* params = as_const_pointer<int32_t>(static_cast<int64_t>(params_address));
    if (context == nullptr || context->magic != CompiledContext{}.magic || params == nullptr) {
        return QCD_STATUS_INVALID;
    }
    (void)params;
    return execute_parsed_contract(context->execution, tensors_address, out_address);
}

void destroy_compiled(long long context_address) {
    std::unique_lock<std::shared_mutex> lifecycle(context_lifecycle_mutex);
    auto* context = as_pointer<CompiledContext>(static_cast<int64_t>(context_address));
    if (context == nullptr || context->magic != CompiledContext{}.magic) {
        return;
    }
    context->magic = 0;
    delete context;
}

}  // namespace

extern "C" QCD_EINSUM_API int applyContractQcd(long long plan, long long tensors, long long out,
                                                long long params) {
    try {
        return execute_contract(plan, tensors, out, params);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::length_error&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_EINSUM_API int applyCompileQcd(long long plan, long long params,
                                               long long out_context) {
    try {
        return compile_contract(plan, params, out_context);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_EINSUM_API int applyCompiledQcd(long long context, long long tensors,
                                                long long out, long long params) {
    try {
        return execute_compiled(context, tensors, out, params);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::length_error&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_EINSUM_API void applyDestroyQcd(long long context) {
    try {
        destroy_compiled(context);
    } catch (...) {
    }
}
