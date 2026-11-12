#include "einsum_dedicated.h"

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

#include "einsum_cuda.h"

/* numpy/cupy 的复数元素在内存里是交错的 [re, im] 对，与 cuComplex 布局一致。本 TU
 * 对复数只做指针算术（乘加交给 cuBLAS 的 Cgemm/Zgemm），按 cuFloatComplex /
 * cuDoubleComplex 取元素宽度即等价于按 numpy 的交错缓冲直传；布局前提编译期钉死。 */
static_assert(sizeof(cuFloatComplex) == 2 * sizeof(float),
              "cuFloatComplex must be an interleaved re/im pair");
static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double),
              "cuDoubleComplex must be an interleaved re/im pair");

/* CUDA 侧的固定步表执行器：格式见 include/einsum_dedicated.h。
 *
 * 与 KUPL `src/dedicated_apply.cpp` 的分工完全相同——步表已展开到
 * 「槽位 + 轴序 + batch/M/N/K」，本 TU 只做 pack → GEMM → unpack：
 *
 *   * pack / unpack 复用 `pack.cu` 的 `qcd_cuda_permute`（通用 strided gather，
 *     `shape` 给目标连续形状、`src_stride` 给源步长），不重复写搬运 kernel；
 *   * GEMM 用 cuBLAS：batch == 1 走 `cublasSgemm` / `cublasDgemm`，batch > 1 走
 *     对应的 `*StridedBatched`（单精度只进 32F 计算，双精度只进 64F，绝不混用）；
 *   * 中间槽与 pack 缓冲来自本 TU 的进程级设备块池：一次调用只在开头 cudaMalloc
 *     一次（单块 arena 切分），步内不再分配。
 *
 * 解析与校验口径逐条对齐 KUPL 侧（同一份 v1 格式、同一组返回码），因此 Python
 * 生成端产出的任何一条合法步表在两个后端上的可行性判定一致。
 */

namespace {

constexpr int kMaxRank = QCD_CUDA_DEDICATED_MAX_RANK;
constexpr int kMaxSlots = QCD_CUDA_DEDICATED_MAX_SLOTS;
constexpr int kMaxOperands = QCD_CUDA_DEDICATED_MAX_OPERANDS;
constexpr int kMaxSteps = QCD_CUDA_DEDICATED_MAX_STEPS;
constexpr std::uint64_t kMaxTableBytes = std::uint64_t(1) << 28;

/* 每个中间槽 / pack 缓冲在 arena 里的起点按 256 字节对齐：cuBLAS 的分块内核在
 * 16 字节对齐上性能最好，而 float32 的累计偏移只能保证 4 字节。 */
constexpr std::size_t kArenaAlignment = 256;
constexpr std::size_t kMinPooledBytes = std::size_t(1) << 20;   /* 1 MiB */
constexpr std::size_t kMaxPooledBytes = std::size_t(1) << 31;   /* 2 GiB */

bool trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("PYQCD_EINSUM_DEDICATED_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

using TraceClock = std::chrono::steady_clock;

double trace_ms(TraceClock::time_point begin, TraceClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/* ------------------------------------------------------------------ 读取器 */

class Reader {
  public:
    Reader(const void* data, std::size_t bytes)
        : base_(static_cast<const std::uint8_t*>(data)), bytes_(bytes) {}

    bool read_i32(std::int32_t* value) {
        if (position_ + sizeof(std::int32_t) > bytes_) {
            return false;
        }
        std::memcpy(value, base_ + position_, sizeof(std::int32_t));
        position_ += sizeof(std::int32_t);
        return true;
    }

    bool read_i64(std::int64_t* value) {
        if (position_ + sizeof(std::int64_t) > bytes_) {
            return false;
        }
        std::memcpy(value, base_ + position_, sizeof(std::int64_t));
        position_ += sizeof(std::int64_t);
        return true;
    }

    std::size_t position() const { return position_; }
    std::size_t bytes() const { return bytes_; }

  private:
    const std::uint8_t* base_;
    std::size_t bytes_;
    std::size_t position_ = 0;
};

/* ------------------------------------------------------------------ 基础 */

bool checked_mul(std::int64_t left, std::int64_t right, std::int64_t* result) {
    if (left < 0 || right < 0 ||
        (right != 0 && left > std::numeric_limits<std::int64_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_bytes(std::int64_t count, std::size_t element, std::size_t* bytes) {
    if (count < 0 || element == 0 ||
        static_cast<std::uint64_t>(count) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element)) {
        return false;
    }
    *bytes = static_cast<std::size_t>(count) * element;
    return true;
}

/* ------------------------------------------------------------------ 表结构 */

struct Shape {
    int rank = 0;
    std::int64_t extents[kMaxRank] = {};
};

struct Step {
    int a_slot = 0;
    int b_slot = 0;
    int dst_slot = -1;
    int rank_a = 0;
    int rank_b = 0;
    int rank_c = 0;
    int batch_rank = 0;
    int m_rank = 0;
    int k_rank = 0;
    std::int64_t batch = 0;
    std::int64_t m = 0;
    std::int64_t n = 0;
    std::int64_t k = 0;
    bool pack_a = false;
    bool pack_b = false;
    bool pack_c = false;
    /* 规范序各轴长（按本操作数自己的规范序）。 */
    std::int64_t a_extent[kMaxRank] = {};
    std::int64_t b_extent[kMaxRank] = {};
    std::int64_t c_extent[kMaxRank] = {};
    /* pack：源张量按规范序的步长（元素）。 */
    std::int64_t a_source_steps[kMaxRank] = {};
    std::int64_t b_source_steps[kMaxRank] = {};
    /* unpack：目的物理轴 p 的源步长（紧凑规范序）。 */
    std::int64_t c_source_steps[kMaxRank] = {};
    /* unpack：目的物理轴 p 的轴长（与 c_extent 的索引口径不同）。 */
    std::int64_t c_destination_extents[kMaxRank] = {};
};

struct Table {
    int num_operands = 0;
    int num_slots = 0;
    int out_rank = 0;
    int dtype_id = 0;
    std::vector<Shape> operands;
    std::vector<Shape> slots;
    Shape output;
    std::vector<Step> steps;
};

bool read_shape(Reader* reader, Shape* shape) {
    std::int32_t rank = 0;
    if (!reader->read_i32(&rank) || rank < 0 || rank > kMaxRank) {
        return false;
    }
    shape->rank = rank;
    for (int axis = 0; axis < rank; ++axis) {
        if (!reader->read_i64(&shape->extents[axis]) || shape->extents[axis] < 0) {
            return false;
        }
    }
    return true;
}

bool is_permutation(const std::int32_t* axes, int rank) {
    bool seen[kMaxRank] = {};
    for (int axis = 0; axis < rank; ++axis) {
        if (axes[axis] < 0 || axes[axis] >= rank || seen[axes[axis]]) {
            return false;
        }
        seen[axes[axis]] = true;
    }
    return true;
}

bool product_range(const std::int64_t* values, int begin, int end, std::int64_t* result) {
    std::int64_t value = 1;
    for (int index = begin; index < end; ++index) {
        if (!checked_mul(value, values[index], &value)) {
            return false;
        }
    }
    *result = value;
    return true;
}

bool product_shape(const Shape& shape, std::int64_t* result) {
    return product_range(shape.extents, 0, shape.rank, result);
}

bool suffix_steps(const std::int64_t* extents, int rank, std::int64_t* steps) {
    std::int64_t running = 1;
    for (int axis = rank - 1; axis >= 0; --axis) {
        steps[axis] = running;
        if (!checked_mul(running, extents[axis], &running)) {
            return false;
        }
    }
    return true;
}

bool shape_strides(const Shape& shape, std::int64_t* strides) {
    return suffix_steps(shape.extents, shape.rank, strides);
}

const Shape& slot_shape(const Table& table, int slot) {
    if (slot < table.num_operands) {
        return table.operands[static_cast<std::size_t>(slot)];
    }
    return table.slots[static_cast<std::size_t>(slot - table.num_operands)];
}

/* 解析 + 全量校验：任何不合规都返回非 0，绝不半途执行。 */
int parse_table(const void* data, long long length, int dtype_id, Table* table) {
    if (data == nullptr || table == nullptr || length <= 0 ||
        static_cast<std::uint64_t>(length) > kMaxTableBytes) {
        return QCD_STATUS_INVALID;
    }
    Reader reader(data, static_cast<std::size_t>(length));
    std::int32_t magic = 0;
    std::int32_t version = 0;
    std::int32_t total = 0;
    std::int32_t num_operands = 0;
    std::int32_t num_slots = 0;
    std::int32_t num_steps = 0;
    std::int32_t out_rank = 0;
    std::int32_t table_dtype = 0;
    if (!reader.read_i32(&magic) || !reader.read_i32(&version) || !reader.read_i32(&total) ||
        !reader.read_i32(&num_operands) || !reader.read_i32(&num_slots) ||
        !reader.read_i32(&num_steps) || !reader.read_i32(&out_rank) ||
        !reader.read_i32(&table_dtype)) {
        return QCD_STATUS_INVALID;
    }
    if (magic != QCD_CUDA_DEDICATED_MAGIC || version != QCD_CUDA_DEDICATED_VERSION ||
        total <= 0 || static_cast<std::uint64_t>(total) != static_cast<std::uint64_t>(length)) {
        return QCD_STATUS_INVALID;
    }
    if (table_dtype != dtype_id) {
        return QCD_STATUS_INVALID;
    }
    if (table_dtype != QCD_DTYPE_FLOAT32 && table_dtype != QCD_DTYPE_FLOAT64 &&
        table_dtype != QCD_DTYPE_COMPLEX64 && table_dtype != QCD_DTYPE_COMPLEX128) {
        /* 只有这四种 IEEE 精度有数值路径：其它 dtype 明确报「不支持」。 */
        return QCD_CUDA_DEDICATED_UNSUPPORTED;
    }
    if (num_operands < 1 || num_operands > kMaxOperands || num_slots < 0 ||
        num_slots > kMaxSlots || num_steps < 1 || num_steps > kMaxSteps || out_rank < 0 ||
        out_rank > kMaxRank) {
        return QCD_CUDA_DEDICATED_UNSUPPORTED;
    }

    table->num_operands = num_operands;
    table->num_slots = num_slots;
    table->out_rank = out_rank;
    table->dtype_id = table_dtype;
    table->operands.assign(static_cast<std::size_t>(num_operands), Shape{});
    table->slots.assign(static_cast<std::size_t>(num_slots), Shape{});
    table->output = Shape{};
    for (int index = 0; index < num_operands; ++index) {
        if (!read_shape(&reader, &table->operands[static_cast<std::size_t>(index)])) {
            return QCD_STATUS_INVALID;
        }
    }
    for (int index = 0; index < num_slots; ++index) {
        if (!read_shape(&reader, &table->slots[static_cast<std::size_t>(index)])) {
            return QCD_STATUS_INVALID;
        }
    }
    if (!read_shape(&reader, &table->output) || table->output.rank != out_rank) {
        return QCD_STATUS_INVALID;
    }

    table->steps.assign(static_cast<std::size_t>(num_steps), Step{});
    std::vector<char> written(static_cast<std::size_t>(num_slots), 0);
    const int total_slots = num_operands + num_slots;
    for (int index = 0; index < num_steps; ++index) {
        Step& step = table->steps[static_cast<std::size_t>(index)];
        std::int32_t a_slot = 0;
        std::int32_t b_slot = 0;
        std::int32_t dst_slot = 0;
        std::int32_t rank_a = 0;
        std::int32_t rank_b = 0;
        std::int32_t rank_c = 0;
        std::int32_t batch_rank = 0;
        std::int32_t m_rank = 0;
        std::int32_t k_rank = 0;
        if (!reader.read_i32(&a_slot) || !reader.read_i32(&b_slot) ||
            !reader.read_i32(&dst_slot) || !reader.read_i32(&rank_a) ||
            !reader.read_i32(&rank_b) || !reader.read_i32(&rank_c) ||
            !reader.read_i32(&batch_rank) || !reader.read_i32(&m_rank) ||
            !reader.read_i32(&k_rank)) {
            return QCD_STATUS_INVALID;
        }
        std::int64_t batch = 0;
        std::int64_t m = 0;
        std::int64_t n = 0;
        std::int64_t k = 0;
        if (!reader.read_i64(&batch) || !reader.read_i64(&m) || !reader.read_i64(&n) ||
            !reader.read_i64(&k)) {
            return QCD_STATUS_INVALID;
        }
        if (a_slot < 0 || a_slot >= total_slots || b_slot < 0 || b_slot >= total_slots ||
            a_slot == b_slot) {
            return QCD_STATUS_INVALID;
        }
        if (dst_slot < -1 || dst_slot >= num_slots) {
            return QCD_STATUS_INVALID;
        }
        if (dst_slot == -1 && index + 1 != num_steps) {
            /* 只有最后一步可以写最终输出。 */
            return QCD_STATUS_INVALID;
        }
        if (dst_slot >= 0 && written[static_cast<std::size_t>(dst_slot)] != 0) {
            return QCD_CUDA_DEDICATED_UNSUPPORTED;
        }
        if (rank_a < 0 || rank_a > kMaxRank || rank_b < 0 || rank_b > kMaxRank ||
            rank_c < 0 || rank_c > kMaxRank || batch_rank < 0 || m_rank < 0 || k_rank < 0) {
            return QCD_STATUS_INVALID;
        }
        if (batch_rank + m_rank + k_rank != rank_a || batch_rank + k_rank > rank_b) {
            return QCD_STATUS_INVALID;
        }
        const int n_rank = rank_b - batch_rank - k_rank;
        if (batch_rank + m_rank + n_rank != rank_c || batch < 0 || m < 0 || n < 0 || k < 0) {
            return QCD_STATUS_INVALID;
        }

        step.a_slot = a_slot;
        step.b_slot = b_slot;
        step.dst_slot = dst_slot;
        step.rank_a = rank_a;
        step.rank_b = rank_b;
        step.rank_c = rank_c;
        step.batch_rank = batch_rank;
        step.m_rank = m_rank;
        step.k_rank = k_rank;
        step.batch = batch;
        step.m = m;
        step.n = n;
        step.k = k;

        std::int32_t a_axes[kMaxRank] = {};
        std::int32_t b_axes[kMaxRank] = {};
        std::int32_t c_axes[kMaxRank] = {};
        for (int axis = 0; axis < rank_a; ++axis) {
            if (!reader.read_i32(&a_axes[axis])) {
                return QCD_STATUS_INVALID;
            }
        }
        for (int axis = 0; axis < rank_b; ++axis) {
            if (!reader.read_i32(&b_axes[axis])) {
                return QCD_STATUS_INVALID;
            }
        }
        for (int axis = 0; axis < rank_c; ++axis) {
            if (!reader.read_i32(&c_axes[axis])) {
                return QCD_STATUS_INVALID;
            }
        }
        if (!is_permutation(a_axes, rank_a) || !is_permutation(b_axes, rank_b) ||
            !is_permutation(c_axes, rank_c)) {
            return QCD_STATUS_INVALID;
        }

        const Shape& left = slot_shape(*table, a_slot);
        const Shape& right = slot_shape(*table, b_slot);
        if (left.rank != rank_a || right.rank != rank_b) {
            return QCD_STATUS_INVALID;
        }
        std::int64_t left_strides[kMaxRank] = {};
        std::int64_t right_strides[kMaxRank] = {};
        if (!shape_strides(left, left_strides) || !shape_strides(right, right_strides)) {
            return QCD_STATUS_INVALID;
        }

        /* 两侧规范序的轴长与源步长；batch/K 两组必须在 A、B 之间逐项相等。 */
        step.pack_a = false;
        step.pack_b = false;
        for (int axis = 0; axis < rank_a; ++axis) {
            step.a_extent[axis] = left.extents[a_axes[axis]];
            step.a_source_steps[axis] = left_strides[a_axes[axis]];
            if (a_axes[axis] != axis) {
                step.pack_a = true;
            }
        }
        for (int axis = 0; axis < rank_b; ++axis) {
            step.b_extent[axis] = right.extents[b_axes[axis]];
            step.b_source_steps[axis] = right_strides[b_axes[axis]];
            if (b_axes[axis] != axis) {
                step.pack_b = true;
            }
        }
        for (int axis = 0; axis < batch_rank; ++axis) {
            if (step.a_extent[axis] != step.b_extent[axis]) {
                return QCD_STATUS_INVALID;
            }
        }
        for (int axis = 0; axis < k_rank; ++axis) {
            if (step.a_extent[batch_rank + m_rank + axis] !=
                step.b_extent[batch_rank + axis]) {
                return QCD_STATUS_INVALID;
            }
        }
        std::int64_t product = 0;
        if (!product_range(step.a_extent, 0, batch_rank, &product) || product != batch ||
            !product_range(step.a_extent, batch_rank, batch_rank + m_rank, &product) ||
            product != m ||
            !product_range(step.a_extent, batch_rank + m_rank,
                           batch_rank + m_rank + k_rank, &product) ||
            product != k ||
            !product_range(step.b_extent, batch_rank + k_rank, rank_b, &product) ||
            product != n) {
            return QCD_STATUS_INVALID;
        }

        /* C 的规范序 = [batch..., M..., N...]；c_axes[p] 是结果物理轴 p 在其中的位置。 */
        for (int axis = 0; axis < batch_rank; ++axis) {
            step.c_extent[axis] = step.a_extent[axis];
        }
        for (int axis = 0; axis < m_rank; ++axis) {
            step.c_extent[batch_rank + axis] = step.a_extent[batch_rank + axis];
        }
        for (int axis = 0; axis < n_rank; ++axis) {
            step.c_extent[batch_rank + m_rank + axis] =
                step.b_extent[batch_rank + k_rank + axis];
        }
        std::int64_t canonical_steps[kMaxRank] = {};
        if (!suffix_steps(step.c_extent, rank_c, canonical_steps)) {
            return QCD_STATUS_INVALID;
        }
        const Shape& destination = dst_slot >= 0
                                       ? table->slots[static_cast<std::size_t>(dst_slot)]
                                       : table->output;
        if (destination.rank != rank_c) {
            return QCD_STATUS_INVALID;
        }
        step.pack_c = false;
        for (int axis = 0; axis < rank_c; ++axis) {
            const int position = c_axes[axis];
            if (step.c_extent[position] != destination.extents[axis]) {
                return QCD_STATUS_INVALID;
            }
            step.c_source_steps[axis] = canonical_steps[position];
            step.c_destination_extents[axis] = destination.extents[axis];
            if (position != axis) {
                step.pack_c = true;
            }
        }
        if (dst_slot >= 0) {
            written[static_cast<std::size_t>(dst_slot)] = 1;
        }
    }
    if (reader.position() != reader.bytes()) {
        /* 尾随字节：保守拒绝，避免格式版本错配被静默接受。 */
        return QCD_STATUS_INVALID;
    }
    return QCD_STATUS_OK;
}

/* ------------------------------------------------------------------ 设备块池 */

struct PoolBlock {
    void* data;
    std::size_t bytes;
};

/* 与 `apply_contract.cu` 的 CudaWorkspace / KUPL 侧 BlockPool 同策略：最佳适配、
 * 只在块大小落在 [bytes, 2*bytes] 时复用，小于 1 MiB 的块直接释放。 */
class DeviceBlockPool {
  public:
    DeviceBlockPool() = default;
    DeviceBlockPool(const DeviceBlockPool&) = delete;
    DeviceBlockPool& operator=(const DeviceBlockPool&) = delete;

    ~DeviceBlockPool() {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const PoolBlock& block : blocks_) {
            cudaFree(block.data);
        }
        blocks_.clear();
        cached_ = 0;
    }

    int acquire(std::size_t bytes, void** data, std::size_t* capacity) {
        *data = nullptr;
        *capacity = bytes;
        if (bytes == 0) {
            return QCD_STATUS_OK;
        }
        {
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
                const PoolBlock block = blocks_[best];
                blocks_[best] = blocks_.back();
                blocks_.pop_back();
                cached_ -= block.bytes;
                *data = block.data;
                *capacity = block.bytes;
                return QCD_STATUS_OK;
            }
        }
        void* allocated = nullptr;
        const cudaError_t error = cudaMalloc(&allocated, bytes);
        if (error != cudaSuccess) {
            return error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY
                                                      : QCD_STATUS_INTERNAL;
        }
        *data = allocated;
        *capacity = bytes;
        return QCD_STATUS_OK;
    }

    void release(void* data, std::size_t bytes) noexcept {
        if (data == nullptr || bytes == 0) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        if (bytes >= kMinPooledBytes && cached_ + bytes <= kMaxPooledBytes) {
            blocks_.push_back(PoolBlock{data, bytes});
            cached_ += bytes;
            return;
        }
        cudaFree(data);
    }

  private:
    std::mutex mutex_;
    std::vector<PoolBlock> blocks_;
    std::size_t cached_ = 0;
};

DeviceBlockPool& device_pool() {
    static DeviceBlockPool pool;
    return pool;
}

/* 一次调用独占一块 arena：中间槽与 pack 缓冲按 256 字节对齐切分，退出时整块归还
 * 池子（容量落在 [bytes, 2*bytes] 的下一次调用直接复用，不再 cudaMalloc）。 */
class CallArena {
  public:
    CallArena() = default;
    CallArena(const CallArena&) = delete;
    CallArena& operator=(const CallArena&) = delete;

    ~CallArena() { release(); }

    int allocate(std::size_t bytes) {
        release();
        if (bytes == 0) {
            return QCD_STATUS_OK;
        }
        return device_pool().acquire(bytes, &data_, &capacity_);
    }

    void* at(std::size_t offset) const {
        return static_cast<std::uint8_t*>(data_) + offset;
    }

  private:
    void release() noexcept {
        if (data_ != nullptr) {
            device_pool().release(data_, capacity_);
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    void* data_ = nullptr;
    std::size_t capacity_ = 0;
};

/* ------------------------------------------------------------------ 归约分支 */

/* 归约分支的累加精度：与 KUPL 侧 `Accumulator<T>` 逐项同口径 —— 单精度升到双精度
 * （复数同档：cuFloatComplex → cuDoubleComplex），双精度保持原位。
 *
 * 为什么要升档：M/N/K 任一为 1 时该步是纯缩放/广播归约，cuBLAS 会绕开分块内核直接
 * 走 gemv。gemv 在 32F 计算下按 float 累加，长链（k 大）上新引入的舍入显著劣于
 * cublasSgemm 的分块累加；KUPL 侧实测同一现象为 2.97e-04 vs 3.95e-05（差 7.5 倍）。
 * 这里把累加器统一提到双精度再舍入回存储精度，两边口径一致。 */
template <typename T>
struct AccumTraits;

template <>
struct AccumTraits<float> {
    using accum = double;
};

template <>
struct AccumTraits<double> {
    using accum = double;
};

template <>
struct AccumTraits<cuFloatComplex> {
    using accum = cuDoubleComplex;
};

template <>
struct AccumTraits<cuDoubleComplex> {
    using accum = cuDoubleComplex;
};

__device__ inline double to_accum(float value) { return static_cast<double>(value); }
__device__ inline double to_accum(double value) { return value; }

__device__ inline cuDoubleComplex to_accum(cuFloatComplex value) {
    return make_cuDoubleComplex(static_cast<double>(value.x), static_cast<double>(value.y));
}

__device__ inline cuDoubleComplex to_accum(cuDoubleComplex value) { return value; }

__device__ inline double zero_accum(double) { return 0.0; }
__device__ inline cuDoubleComplex zero_accum(cuDoubleComplex) {
    return make_cuDoubleComplex(0.0, 0.0);
}

/* 归约累加：实数与复数各一个重载，避免依赖 cuComplex 的 + / * 运算符重载差异。 */
__device__ inline double accum_add(double left, double right) { return left + right; }
__device__ inline cuDoubleComplex accum_add(cuDoubleComplex left, cuDoubleComplex right) {
    return make_cuDoubleComplex(left.x + right.x, left.y + right.y);
}

__device__ inline double accum_mul(double left, double right) { return left * right; }
__device__ inline cuDoubleComplex accum_mul(cuDoubleComplex left, cuDoubleComplex right) {
    return make_cuDoubleComplex(left.x * right.x - left.y * right.y,
                               left.x * right.y + left.y * right.x);
}

__device__ inline void from_accum(double value, float* destination) {
    *destination = static_cast<float>(value);
}

__device__ inline void from_accum(double value, double* destination) { *destination = value; }

__device__ inline void from_accum(cuDoubleComplex value, cuFloatComplex* destination) {
    *destination = make_cuFloatComplex(static_cast<float>(value.x),
                                       static_cast<float>(value.y));
}

__device__ inline void from_accum(cuDoubleComplex value, cuDoubleComplex* destination) {
    *destination = value;
}

/* K == 1 的缩放**不升档**：该步没有归约链，按存储精度乘一次即可，与 KUPL 侧
 * `target[column] = scale * right[column]` 逐位同口径。 */
__device__ inline float type_mul(float left, float right) { return left * right; }
__device__ inline double type_mul(double left, double right) { return left * right; }
__device__ inline cuFloatComplex type_mul(cuFloatComplex left, cuFloatComplex right) {
    return make_cuFloatComplex(left.x * right.x - left.y * right.y,
                               left.x * right.y + left.y * right.x);
}
__device__ inline cuDoubleComplex type_mul(cuDoubleComplex left, cuDoubleComplex right) {
    return make_cuDoubleComplex(left.x * right.x - left.y * right.y,
                                left.x * right.y + left.y * right.x);
}

constexpr int kReduceThreads = 128;

int reduce_blocks(std::int64_t elements) {
    const std::int64_t blocks = (elements + kReduceThreads - 1) / kReduceThreads;
    return static_cast<int>(std::min<std::int64_t>(blocks, 65535));
}

/* K == 1：纯缩放 C[b][m][n] = A[b][m] * B[b][n]（无归约，不进 BLAS）。 */
template <typename T>
__global__ void scale_k1_kernel(const T* a, const T* b, T* c, std::int64_t batch,
                                std::int64_t m, std::int64_t n) {
    const std::int64_t total = batch * m * n;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const std::int64_t group = index / (m * n);
        const std::int64_t rest = index % (m * n);
        const std::int64_t row = rest / n;
        const std::int64_t column = rest % n;
        const auto product =
            type_mul(a[group * m + row], b[group * n + column]);
        c[index] = product;
    }
}

/* N == 1：C[b][m] = Σ_k A[b][m][k] · B[b][k]，**双精度累加**（见 AccumTraits）。 */
template <typename T>
__global__ void reduce_n1_kernel(const T* a, const T* b, T* c, std::int64_t batch,
                                 std::int64_t m, std::int64_t k) {
    using Accum = typename AccumTraits<T>::accum;
    const std::int64_t total = batch * m;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const std::int64_t group = index / m;
        const std::int64_t row = index % m;
        const T* left = a + (group * m + row) * k;
        const T* right = b + group * k;
        Accum total_value = zero_accum(Accum{});
        for (std::int64_t inner = 0; inner < k; ++inner) {
            total_value = accum_add(
                total_value,
                accum_mul(to_accum(left[inner]), to_accum(right[inner])));
        }
        from_accum(total_value, c + index);
    }
}

/* M == 1：C[b][n] = Σ_k A[b][k] · B[b][k][n]，同样双精度累加。 */
template <typename T>
__global__ void reduce_m1_kernel(const T* a, const T* b, T* c, std::int64_t batch,
                                 std::int64_t n, std::int64_t k) {
    using Accum = typename AccumTraits<T>::accum;
    const std::int64_t total = batch * n;
    const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += stride) {
        const std::int64_t group = index / n;
        const std::int64_t column = index % n;
        const T* left = a + group * k;
        const T* right = b + group * k * n;
        Accum total_value = zero_accum(Accum{});
        for (std::int64_t inner = 0; inner < k; ++inner) {
            total_value = accum_add(
                total_value,
                accum_mul(to_accum(left[inner]),
                          to_accum(right[inner * n + column])));
        }
        from_accum(total_value, c + index);
    }
}

int map_launch(cudaError_t error) {
    return error == cudaSuccess
               ? QCD_STATUS_OK
               : (error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY
                                                     : QCD_STATUS_INTERNAL);
}

/* ------------------------------------------------------------------ BLAS 能力 */

/* dtype 白名单必须与**运行期**的 cuBLAS 能力一致：精简版 BLAS（某些容器镜像只保留
 * Sgemm/Dgemm）上没有 Cgemm/Zgemm 时，本入口必须把对应 dtype 判为不可执行并返回
 * QCD_CUDA_DEDICATED_UNSUPPORTED，让上层回落通用路径（fail open），而不是在
 * GEMM 调用点炸掉、也不是把复数降成实数算。
 *
 * 探测用 dlsym(RTLD_DEFAULT)：本 TU 已硬链 -lcublas，正常部署一律命中；探测的价值
 * 在于把"符号是否齐备"变成**数据**，而不是让缺符号在第一次调用时才暴露。
 * 每个 dtype 都要求 base 与 StridedBatched 两种变体齐备 —— 执行器在 batch>1 时
 * 只走后者，缺它就等于该 dtype 的完整路径不存在。 */
template <typename Function>
bool bind_gemm_symbol(const char* name, Function* function) {
    void* symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr) {
        *function = nullptr;
        return false;
    }
    static_assert(sizeof(Function) == sizeof(symbol), "function pointer size mismatch");
    std::memcpy(function, &symbol, sizeof(Function));
    return true;
}

struct GemmSymbols {
    bool float32 = false;
    bool float64 = false;
    bool complex64 = false;
    bool complex128 = false;
};

/* 探测只依赖符号表，不需要设备，因此可以安全地在无卡构建机上查询（全 false）。 */
const GemmSymbols& gemm_symbols() {
    static const GemmSymbols symbols = [] {
        GemmSymbols found;
        decltype(&cublasSgemm) sgemm = nullptr;
        decltype(&cublasDgemm) dgemm = nullptr;
        decltype(&cublasCgemm) cgemm = nullptr;
        decltype(&cublasZgemm) zgemm = nullptr;
        decltype(&cublasSgemmStridedBatched) sgemm_batched = nullptr;
        decltype(&cublasDgemmStridedBatched) dgemm_batched = nullptr;
        decltype(&cublasCgemmStridedBatched) cgemm_batched = nullptr;
        decltype(&cublasZgemmStridedBatched) zgemm_batched = nullptr;
        found.float32 = bind_gemm_symbol("cublasSgemm", &sgemm) &&
                        bind_gemm_symbol("cublasSgemmStridedBatched", &sgemm_batched);
        found.float64 = bind_gemm_symbol("cublasDgemm", &dgemm) &&
                        bind_gemm_symbol("cublasDgemmStridedBatched", &dgemm_batched);
        found.complex64 = bind_gemm_symbol("cublasCgemm", &cgemm) &&
                          bind_gemm_symbol("cublasCgemmStridedBatched", &cgemm_batched);
        found.complex128 = bind_gemm_symbol("cublasZgemm", &zgemm) &&
                           bind_gemm_symbol("cublasZgemmStridedBatched", &zgemm_batched);
        return found;
    }();
    return symbols;
}

bool gemm_available(int dtype_id) {
    const GemmSymbols& symbols = gemm_symbols();
    switch (dtype_id) {
        case QCD_DTYPE_FLOAT32:
            return symbols.float32;
        case QCD_DTYPE_FLOAT64:
            return symbols.float64;
        case QCD_DTYPE_COMPLEX64:
            return symbols.complex64;
        case QCD_DTYPE_COMPLEX128:
            return symbols.complex128;
        default:
            return false;
    }
}

/* bit(dtype_id) 置位 = 该 dtype 有完整数值路径；Python 侧据此裁剪 dtype 白名单。 */
int dtype_mask() {
    int mask = 0;
    for (int dtype_id = 0; dtype_id <= 3; ++dtype_id) {
        if (gemm_available(dtype_id)) {
            mask |= 1 << dtype_id;
        }
    }
    return mask;
}

/* ------------------------------------------------------------------ cuBLAS */

int map_cublas(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return QCD_STATUS_OK;
    }
    return status == CUBLAS_STATUS_ALLOC_FAILED ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

struct HandleState {
    std::mutex mutex;                 /* 串行化整次调用（arena/句柄共用） */
    cublasHandle_t handle = nullptr;
    int device = -1;

    ~HandleState() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

HandleState& handle_state() {
    static HandleState state;
    return state;
}

int ensure_handle(HandleState* state) {
    int device = -1;
    const cudaError_t device_error = cudaGetDevice(&device);
    if (device_error != cudaSuccess) {
        return device_error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY
                                                         : QCD_STATUS_INTERNAL;
    }
    if (state->handle != nullptr && state->device == device) {
        return QCD_STATUS_OK;
    }
    if (state->handle != nullptr) {
        cublasDestroy(state->handle);
        state->handle = nullptr;
    }
    cublasStatus_t status = cublasCreate(&state->handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        state->handle = nullptr;
        state->device = -1;
        return map_cublas(status);
    }
    status = cublasSetPointerMode(state->handle, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(state->handle);
        state->handle = nullptr;
        state->device = -1;
        return map_cublas(status);
    }
    state->device = device;
    return QCD_STATUS_OK;
}

/* 精度契约：float32 只进 cublasSgemm（32F 计算），float64 只进 cublasDgemm
 * （64F 计算），两条路径互不相通；batch > 1 用同名 *StridedBatched 变体。
 *
 * 行主序 → cuBLAS（列主序）的换法：与 `batched_gemm.cu` 同一条恒等式
 *
 *     C_rm(m×n) = A_rm(m×k) · B_rm(k×n)
 *  ⇔  C^T_col(n×m) = B^T_col(n×k) · A^T_col(k×m)
 *
 * 即把两侧操作数**对调**、m/n 对调，三个缓冲都按同一个基地址直传。这样
 * leading dimension 恰好等于各矩阵的行数（B_rm 当 n×k 用 → lda = n；
 * A_rm 当 k×m 用 → ldb = k；C_rm 当 n×m 用 → ldc = n），全部满足 cuBLAS 的
 * 校验。直接按「A 当 m×k、ldb = n」下发在 n < k 时会被 cuBLAS 以
 * "parameter 10 had an illegal value"（ldb < k）拒绝，这是本文件最初的坑。 */
template <typename T>
struct GemmDispatch;

template <>
struct GemmDispatch<float> {
    static int run(cublasHandle_t handle, std::int64_t batch, std::int64_t m, std::int64_t n,
                   std::int64_t k, const float* a, const float* b, float* c) {
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const int rows = static_cast<int>(n);
        const int cols = static_cast<int>(m);
        const int inner = static_cast<int>(k);
        const int leading_b = static_cast<int>(n);
        const int leading_a = static_cast<int>(k);
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[cuda-dedicated] sgemm batch=%lld C^T=%dx%d k=%d "
                         "lda=%d ldb=%d ldc=%d A=%p B=%p C=%p\n",
                         static_cast<long long>(batch), rows, cols, inner, leading_b,
                         leading_a, leading_b, static_cast<const void*>(b),
                         static_cast<const void*>(a), static_cast<void*>(c));
        }
        cublasStatus_t status = CUBLAS_STATUS_INVALID_VALUE;
        if (batch == 1) {
            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b,
                                 leading_b, a, leading_a, &beta, c, leading_b);
        } else if (batch > 1) {
            const long long stride_a = static_cast<long long>(m) * k;
            const long long stride_b = static_cast<long long>(k) * n;
            const long long stride_c = static_cast<long long>(m) * n;
            status = cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b, leading_b,
                stride_b, a, leading_a, stride_a, &beta, c, leading_b, stride_c,
                static_cast<int>(batch));
        }
        return map_cublas(status);
    }
};

template <>
struct GemmDispatch<double> {
    static int run(cublasHandle_t handle, std::int64_t batch, std::int64_t m, std::int64_t n,
                   std::int64_t k, const double* a, const double* b, double* c) {
        const double alpha = 1.0;
        const double beta = 0.0;
        const int rows = static_cast<int>(n);
        const int cols = static_cast<int>(m);
        const int inner = static_cast<int>(k);
        const int leading_b = static_cast<int>(n);
        const int leading_a = static_cast<int>(k);
        cublasStatus_t status = CUBLAS_STATUS_INVALID_VALUE;
        if (batch == 1) {
            status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b,
                                 leading_b, a, leading_a, &beta, c, leading_b);
        } else if (batch > 1) {
            const long long stride_a = static_cast<long long>(m) * k;
            const long long stride_b = static_cast<long long>(k) * n;
            const long long stride_c = static_cast<long long>(m) * n;
            status = cublasDgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b, leading_b,
                stride_b, a, leading_a, stride_a, &beta, c, leading_b, stride_c,
                static_cast<int>(batch));
        }
        return map_cublas(status);
    }
};

/* 复数两档：与实数版共用同一套「行主序 → 列主序」换法（两侧操作数对调、m/n
 * 对调、三个缓冲同基地址直传），alpha/beta 指向本精度的复常量。乘法是普通复数
 * 乘法（不做共轭）——einsum 的语义就是矩阵乘。 */
template <>
struct GemmDispatch<cuFloatComplex> {
    static int run(cublasHandle_t handle, std::int64_t batch, std::int64_t m, std::int64_t n,
                   std::int64_t k, const cuFloatComplex* a, const cuFloatComplex* b,
                   cuFloatComplex* c) {
        const cuFloatComplex alpha = make_cuFloatComplex(1.0F, 0.0F);
        const cuFloatComplex beta = make_cuFloatComplex(0.0F, 0.0F);
        const int rows = static_cast<int>(n);
        const int cols = static_cast<int>(m);
        const int inner = static_cast<int>(k);
        const int leading_b = static_cast<int>(n);
        const int leading_a = static_cast<int>(k);
        cublasStatus_t status = CUBLAS_STATUS_INVALID_VALUE;
        if (batch == 1) {
            status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b,
                                 leading_b, a, leading_a, &beta, c, leading_b);
        } else if (batch > 1) {
            const long long stride_a = static_cast<long long>(m) * k;
            const long long stride_b = static_cast<long long>(k) * n;
            const long long stride_c = static_cast<long long>(m) * n;
            status = cublasCgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b, leading_b,
                stride_b, a, leading_a, stride_a, &beta, c, leading_b, stride_c,
                static_cast<int>(batch));
        }
        return map_cublas(status);
    }
};

template <>
struct GemmDispatch<cuDoubleComplex> {
    static int run(cublasHandle_t handle, std::int64_t batch, std::int64_t m, std::int64_t n,
                   std::int64_t k, const cuDoubleComplex* a, const cuDoubleComplex* b,
                   cuDoubleComplex* c) {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        const int rows = static_cast<int>(n);
        const int cols = static_cast<int>(m);
        const int inner = static_cast<int>(k);
        const int leading_b = static_cast<int>(n);
        const int leading_a = static_cast<int>(k);
        cublasStatus_t status = CUBLAS_STATUS_INVALID_VALUE;
        if (batch == 1) {
            status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b,
                                 leading_b, a, leading_a, &beta, c, leading_b);
        } else if (batch > 1) {
            const long long stride_a = static_cast<long long>(m) * k;
            const long long stride_b = static_cast<long long>(k) * n;
            const long long stride_c = static_cast<long long>(m) * n;
            status = cublasZgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, inner, &alpha, b, leading_b,
                stride_b, a, leading_a, stride_a, &beta, c, leading_b, stride_c,
                static_cast<int>(batch));
        }
        return map_cublas(status);
    }
};

/* ------------------------------------------------------------------ 步执行 */

/* 一步的数值执行：先走缩放/归约短路径，其余交给 cuBLAS。
 *
 * pack 之后两侧缓冲区一律是规范序 [batch,M,K] / [batch,K,N] 的连续块（未 pack 时
 * 只可能是恒等轴序，同样是连续规范序），所以这里不需要 KUPL 侧那套 trans 判定。
 * 三条短路径与 KUPL `gemm_block` 逐条对齐：k==1 是纯缩放（按存储精度乘一次），
 * m==1 / n==1 是广播归约（**双精度累加**，见 AccumTraits），其余才进 GEMM。 */
template <typename T>
int run_step_math(cublasHandle_t handle, const Step& step, const T* left, const T* right,
                  T* result) {
    if (step.batch == 0) {
        return QCD_STATUS_OK;
    }
    if (step.k == 1) {
        const std::int64_t elements = step.batch * step.m * step.n;
        if (elements == 0) {
            return QCD_STATUS_OK;
        }
        scale_k1_kernel<T><<<reduce_blocks(elements), kReduceThreads>>>(
            left, right, result, step.batch, step.m, step.n);
        return map_launch(cudaGetLastError());
    }
    if (step.n == 1) {
        const std::int64_t elements = step.batch * step.m;
        if (elements == 0) {
            return QCD_STATUS_OK;
        }
        reduce_n1_kernel<T><<<reduce_blocks(elements), kReduceThreads>>>(
            left, right, result, step.batch, step.m, step.k);
        return map_launch(cudaGetLastError());
    }
    if (step.m == 1) {
        const std::int64_t elements = step.batch * step.n;
        if (elements == 0) {
            return QCD_STATUS_OK;
        }
        reduce_m1_kernel<T><<<reduce_blocks(elements), kReduceThreads>>>(
            left, right, result, step.batch, step.n, step.k);
        return map_launch(cudaGetLastError());
    }
    return GemmDispatch<T>::run(handle, step.batch, step.m, step.n, step.k, left, right,
                                result);
}

/* ------------------------------------------------------------------ 执行 */

bool make_copy_spec(const std::int64_t* extents, const std::int64_t* source_steps, int rank,
                    std::int64_t elements, qcd_cuda_copy_spec* spec) {
    if (rank < 0 || rank > QCD_CUDA_MAX_PACKED_RANK || elements < 0) {
        return false;
    }
    spec->rank = rank;
    spec->elements = elements;
    for (int axis = 0; axis < rank; ++axis) {
        if (extents[axis] < 0 || source_steps[axis] < 0) {
            return false;
        }
        spec->shape[axis] = extents[axis];
        spec->src_stride[axis] = source_steps[axis];
    }
    return true;
}

int zero_block(void* destination, std::int64_t elements, std::size_t element) {
    if (elements == 0) {
        return QCD_STATUS_OK;
    }
    const cudaError_t error = cudaMemsetAsync(
        destination, 0, static_cast<std::size_t>(elements) * element, nullptr);
    return error == cudaSuccess
               ? QCD_STATUS_OK
               : (error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY
                                                     : QCD_STATUS_INTERNAL);
}

template <typename T>
int run_typed(const Table& table, void* const* operands, void* out, cublasHandle_t handle) {
    const std::size_t element = sizeof(T);
    const bool trace = trace_enabled();
    const int total_slots = table.num_operands + table.num_slots;

    /* 中间槽：按全表形状一次算好 arena 偏移（256 字节对齐）。 */
    std::vector<std::size_t> slot_offsets(static_cast<std::size_t>(table.num_slots), 0);
    std::size_t used = 0;
    for (int index = 0; index < table.num_slots; ++index) {
        const Shape& shape = table.slots[static_cast<std::size_t>(index)];
        std::int64_t count = 0;
        std::size_t bytes = 0;
        if (!product_shape(shape, &count) || !checked_bytes(count, element, &bytes)) {
            return QCD_STATUS_INVALID;
        }
        used = align_up(used, kArenaAlignment);
        slot_offsets[static_cast<std::size_t>(index)] = used;
        if (bytes > std::numeric_limits<std::size_t>::max() - used) {
            return QCD_STATUS_INVALID;
        }
        used += bytes;
    }

    /* pack 缓冲按全表最大尺寸各一块。 */
    std::int64_t max_a = 0;
    std::int64_t max_b = 0;
    std::int64_t max_c = 0;
    for (const Step& step : table.steps) {
        std::int64_t elements = 0;
        if (!checked_mul(step.batch, step.m, &elements) ||
            !checked_mul(elements, step.k, &elements)) {
            return QCD_STATUS_INVALID;
        }
        max_a = std::max(max_a, step.pack_a ? elements : 0);
        if (!checked_mul(step.batch, step.k, &elements) ||
            !checked_mul(elements, step.n, &elements)) {
            return QCD_STATUS_INVALID;
        }
        max_b = std::max(max_b, step.pack_b ? elements : 0);
        if (!checked_mul(step.batch, step.m, &elements) ||
            !checked_mul(elements, step.n, &elements)) {
            return QCD_STATUS_INVALID;
        }
        max_c = std::max(max_c, step.pack_c ? elements : 0);
    }
    std::size_t pack_a_bytes = 0;
    std::size_t pack_b_bytes = 0;
    std::size_t pack_c_bytes = 0;
    if (!checked_bytes(max_a, element, &pack_a_bytes) ||
        !checked_bytes(max_b, element, &pack_b_bytes) ||
        !checked_bytes(max_c, element, &pack_c_bytes)) {
        return QCD_STATUS_INVALID;
    }
    used = align_up(used, kArenaAlignment);
    const std::size_t pack_a_offset = used;
    used += pack_a_bytes;
    used = align_up(used, kArenaAlignment);
    const std::size_t pack_b_offset = used;
    used += pack_b_bytes;
    used = align_up(used, kArenaAlignment);
    const std::size_t pack_c_offset = used;
    used += pack_c_bytes;

    const TraceClock::time_point allocation_begin =
        trace ? TraceClock::now() : TraceClock::time_point{};
    CallArena arena;
    int status = arena.allocate(used);
    if (status != QCD_STATUS_OK) {
        return status;
    }

    std::vector<void*> slot_data(static_cast<std::size_t>(total_slots), nullptr);
    std::vector<char> slot_live(static_cast<std::size_t>(total_slots), 0);
    for (int index = 0; index < table.num_operands; ++index) {
        slot_data[static_cast<std::size_t>(index)] = operands[index];
        slot_live[static_cast<std::size_t>(index)] = 1;
    }
    for (int index = 0; index < table.num_slots; ++index) {
        const std::size_t slot = static_cast<std::size_t>(table.num_operands + index);
        slot_data[slot] = arena.at(slot_offsets[static_cast<std::size_t>(index)]);
        slot_live[slot] = 0;
    }
    void* const pack_a = pack_a_bytes != 0 ? arena.at(pack_a_offset) : nullptr;
    void* const pack_b = pack_b_bytes != 0 ? arena.at(pack_b_offset) : nullptr;
    void* const pack_c = pack_c_bytes != 0 ? arena.at(pack_c_offset) : nullptr;

    if (trace) {
        std::fprintf(stderr,
                     "[cuda-dedicated] alloc %.3f ms | slots=%d steps=%d element=%zu "
                     "bytes: arena=%zu A=%zu B=%zu C=%zu\n",
                     trace_ms(allocation_begin, TraceClock::now()), table.num_slots,
                     static_cast<int>(table.steps.size()), element, used, pack_a_bytes,
                     pack_b_bytes, pack_c_bytes);
        for (int slot_index = 0; slot_index < table.num_slots; ++slot_index) {
            std::fprintf(stderr, "[cuda-dedicated] slot %d offset=%zu\n", slot_index,
                         slot_offsets[static_cast<std::size_t>(slot_index)]);
        }
    }

    for (std::size_t index = 0; index < table.steps.size(); ++index) {
        const Step& step = table.steps[index];
        const std::size_t left_index = static_cast<std::size_t>(step.a_slot);
        const std::size_t right_index = static_cast<std::size_t>(step.b_slot);
        if (slot_live[left_index] == 0 || slot_live[right_index] == 0) {
            return QCD_STATUS_INVALID;
        }
        void* destination = out;
        std::size_t destination_index = 0;
        if (step.dst_slot >= 0) {
            destination_index = static_cast<std::size_t>(table.num_operands + step.dst_slot);
            if (slot_live[destination_index] != 0) {
                return QCD_STATUS_INVALID;
            }
            destination = slot_data[destination_index];
        }
        if (step.m > std::numeric_limits<int>::max() ||
            step.n > std::numeric_limits<int>::max() ||
            step.k > std::numeric_limits<int>::max() ||
            step.batch > std::numeric_limits<int>::max()) {
            return QCD_STATUS_INVALID;
        }

        const T* left = static_cast<const T*>(slot_data[left_index]);
        const T* right = static_cast<const T*>(slot_data[right_index]);
        if ((step.m != 0 && step.n != 0 && step.k != 0) &&
            (left == nullptr || right == nullptr)) {
            return QCD_STATUS_INVALID;
        }

        const TraceClock::time_point step_begin =
            trace ? TraceClock::now() : TraceClock::time_point{};
        /* pack/unpack 的 dtype id 与步表同值：qcd_cuda_permute 对 QCD_DTYPE_* 四种
         * 精度都实例化了搬运 kernel（含复数按交错实虚对逐元素拷贝）。 */
        const int native_dtype = table.dtype_id;
        const T* packed_left = left;
        if (step.pack_a) {
            std::int64_t elements = 0;
            qcd_cuda_copy_spec spec{};
            if (!checked_mul(step.batch, step.m, &elements) ||
                !checked_mul(elements, step.k, &elements) ||
                !make_copy_spec(step.a_extent, step.a_source_steps, step.rank_a, elements,
                                &spec)) {
                return QCD_STATUS_INVALID;
            }
            status = qcd_cuda_permute(native_dtype, left, pack_a, spec, nullptr);
            if (status != QCD_STATUS_OK) {
                return status;
            }
            packed_left = static_cast<const T*>(pack_a);
        }
        const T* packed_right = right;
        if (step.pack_b) {
            std::int64_t elements = 0;
            qcd_cuda_copy_spec spec{};
            if (!checked_mul(step.batch, step.k, &elements) ||
                !checked_mul(elements, step.n, &elements) ||
                !make_copy_spec(step.b_extent, step.b_source_steps, step.rank_b, elements,
                                &spec)) {
                return QCD_STATUS_INVALID;
            }
            status = qcd_cuda_permute(native_dtype, right, pack_b, spec, nullptr);
            if (status != QCD_STATUS_OK) {
                return status;
            }
            packed_right = static_cast<const T*>(pack_b);
        }
        if (trace) {
            cudaDeviceSynchronize();
        }
        const TraceClock::time_point packed =
            trace ? TraceClock::now() : TraceClock::time_point{};

        T* result = step.pack_c ? static_cast<T*>(pack_c) : static_cast<T*>(destination);
        if (step.m != 0 && step.n != 0 && step.batch != 0) {
            if (step.k == 0) {
                std::int64_t elements = 0;
                if (!checked_mul(step.batch, step.m, &elements) ||
                    !checked_mul(elements, step.n, &elements)) {
                    return QCD_STATUS_INVALID;
                }
                status = zero_block(result, elements, element);
                if (status != QCD_STATUS_OK) {
                    return status;
                }
            } else {
                status = run_step_math<T>(handle, step, packed_left, packed_right, result);
                if (status != QCD_STATUS_OK) {
                    return status;
                }
            }
        }
        if (trace) {
            cudaDeviceSynchronize();
        }
        const TraceClock::time_point multiplied =
            trace ? TraceClock::now() : TraceClock::time_point{};

        if (step.pack_c) {
            qcd_cuda_copy_spec spec{};
            std::int64_t elements = 0;
            if (!checked_mul(step.batch, step.m, &elements) ||
                !checked_mul(elements, step.n, &elements) ||
                !make_copy_spec(step.c_destination_extents, step.c_source_steps, step.rank_c,
                                elements, &spec)) {
                return QCD_STATUS_INVALID;
            }
            status = qcd_cuda_permute(native_dtype, result, destination, spec, nullptr);
            if (status != QCD_STATUS_OK) {
                return status;
            }
        }
        if (trace) {
            cudaDeviceSynchronize();
            const TraceClock::time_point finished = TraceClock::now();
            std::fprintf(stderr,
                         "[cuda-dedicated] step %zu slots=%d,%d->%d "
                         "bmnk=%lld,%lld,%lld,%lld pack_a=%d pack_b=%d pack_c=%d | "
                         "pack %.3f ms gemm %.3f ms unpack %.3f ms total %.3f ms\n",
                         index, step.a_slot, step.b_slot, step.dst_slot,
                         static_cast<long long>(step.batch), static_cast<long long>(step.m),
                         static_cast<long long>(step.n), static_cast<long long>(step.k),
                         step.pack_a ? 1 : 0, step.pack_b ? 1 : 0, step.pack_c ? 1 : 0,
                         trace_ms(step_begin, packed), trace_ms(packed, multiplied),
                         trace_ms(multiplied, finished), trace_ms(step_begin, finished));
        }

        /* 中间体在收缩树里只被消费一次：读完即标记死亡。 */
        if (step.a_slot >= table.num_operands) {
            slot_live[left_index] = 0;
        }
        if (step.b_slot >= table.num_operands) {
            slot_live[right_index] = 0;
        }
        if (step.dst_slot >= 0) {
            slot_live[destination_index] = 1;
        }
    }
    return QCD_STATUS_OK;
}

/* ------------------------------------------------------------------ 入口 */

int apply_dedicated(const void* steps, long long steps_len, void* const* operands,
                    int n_operands, void* out, int dtype_id, void* stream) {
    if (steps == nullptr || operands == nullptr || out == nullptr || stream != nullptr) {
        return QCD_STATUS_INVALID;
    }
    Table table;
    const bool trace = trace_enabled();
    const TraceClock::time_point parse_begin =
        trace ? TraceClock::now() : TraceClock::time_point{};
    const int parsed = parse_table(steps, steps_len, dtype_id, &table);
    if (parsed != QCD_STATUS_OK) {
        return parsed;
    }
    if (trace) {
        std::fprintf(stderr, "[cuda-dedicated] parse %.3f ms\n",
                     trace_ms(parse_begin, TraceClock::now()));
    }
    if (n_operands != table.num_operands) {
        return QCD_STATUS_INVALID;
    }
    /* dtype 白名单与运行期 cuBLAS 能力必须一致：缺符号即判 UNSUPPORTED（fail open），
     * 由上层回落通用路径，绝不在 GEMM 调用点失败、也绝不降精度。 */
    if (!gemm_available(dtype_id)) {
        return QCD_CUDA_DEDICATED_UNSUPPORTED;
    }
    for (int index = 0; index < n_operands; ++index) {
        if (operands[index] == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    /* pack / unpack 复用 pack.cu 的实现；缺实现时明确失败，不伪造成功。 */
    if (qcd_cuda_permute == nullptr) {
        return QCD_STATUS_INTERNAL;
    }
    HandleState& state = handle_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const int handle_status = ensure_handle(&state);
    if (handle_status != QCD_STATUS_OK) {
        return handle_status;
    }
    switch (dtype_id) {
        case QCD_DTYPE_FLOAT32:
            return run_typed<float>(table, operands, out, state.handle);
        case QCD_DTYPE_FLOAT64:
            return run_typed<double>(table, operands, out, state.handle);
        case QCD_DTYPE_COMPLEX64:
            return run_typed<cuFloatComplex>(table, operands, out, state.handle);
        case QCD_DTYPE_COMPLEX128:
            return run_typed<cuDoubleComplex>(table, operands, out, state.handle);
        default:
            return QCD_CUDA_DEDICATED_UNSUPPORTED;
    }
}

}  // namespace

extern "C" QCD_CUDA_DEDICATED_API int qcd_cuda_dedicated_apply(
    const void* steps,
    long long steps_len,
    void* const* operands,
    int n_operands,
    void* out,
    int dtype_id,
    void* stream) {
    try {
        return apply_dedicated(steps, steps_len, operands, n_operands, out, dtype_id, stream);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_CUDA_DEDICATED_API int qcd_cuda_dedicated_dtypes(void) {
    try {
        return dtype_mask();
    } catch (...) {
        return 0;
    }
}
