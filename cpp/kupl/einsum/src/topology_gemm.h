#ifndef PYQCD_KUPL_EINSUM_TOPOLOGY_GEMM_H
#define PYQCD_KUPL_EINSUM_TOPOLOGY_GEMM_H

#include "einsum_define.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <vector>

/*
 * 参数化 GEMM 执行器：消费 topology_kernels.inc 里的 per-拓扑步表。
 *
 * 一条拓扑 = 一段 ``uint8`` 步序列（格式见 pyqcd/testing/_topology_kernel_gen.py）。
 * 每一步做三件事，**没有任何运行期符号解析**：
 *
 *   1. 打包：按 ``a_axes``/``b_axes`` 把两侧操作数 gather 成 GEMM 规范序
 *      ``[batch..., M..., K...]`` 与 ``[batch..., K..., N...]``；
 *   2. GEMM：``cblas_{s,d}gemm``（RowMajor，两侧 NoTrans）；
 *   3. 解包：按 ``c_axes`` 把 ``[batch..., M..., N...]`` 的批次结果 gather 成该步
 *      输出张量的下标顺序。
 *
 * 步长只由 ``dim`` 的幂次给出：轴长恒为 ``dim``，所以张量第 p 轴的步长是
 * ``dim^(rank-1-p)``。运行期只做整数幂与乘加，不解析符号、不查 stride 表。
 *
 * **精度**：float 与 double 各实例化一条路径（同一模板、同一张步表），调用方按
 * dtype 显式选择；complex 一律交回通用路径，绝不隐式降精度。
 *
 * **中间体**：每一步的输出是一块显式大小的缓冲区，来自与 ``src/apply_contract.cpp``
 * 同策略的进程级块池（acquire 最佳适配 / release 只回收大块），消费后立刻归还，
 * 因此峰值只占相邻两步的几块内存，不会每步 malloc。apply_contract.cpp 的
 * BufferPool 位于匿名命名空间、不可跨 TU 复用，这里按同样口径自带最小实现。
 */

namespace pyqcd_einsum {
namespace topology_gemm {

enum ElementKind : int { kFloat = 0, kDouble = 1 };

/* 步表里 rank 用 4 bit 存（生成期断言 <= 15），轴位置同样 4 bit。 */
enum : int { kMaxRank = 15 };
/* 单条拓扑的中间槽位数上限（语料实测 <= 16），用于栈上定长槽表。 */
enum : int { kMaxSlots = 64 };

struct Record {
    uint64_t hash;
    const char* signature;
    uint32_t code_offset;
    uint32_t code_bytes;
    uint16_t arity;
    uint8_t out_rank;
    uint8_t step_count;
};

/* ------------------------------------------------------------------ 块池 */

class BlockPool {
  public:
    BlockPool() = default;
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    ~BlockPool() {
        for (const Block& block : blocks_) {
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

inline BlockPool& block_pool() {
    static BlockPool pool;
    return pool;
}

/* ------------------------------------------------------------------ CBLAS */

using SgemmFn = void (*)(int, int, int, int, int, int, float, const void*, int, const void*, int,
                         float, void*, int);
using DgemmFn = void (*)(int, int, int, int, int, int, double, const void*, int, const void*, int,
                         double, void*, int);

struct CblasApi {
    SgemmFn sgemm = nullptr;
    DgemmFn dgemm = nullptr;

    bool complete() const { return sgemm != nullptr && dgemm != nullptr; }
};

template <typename Function>
inline bool bind_symbol(void* handle, const char* name, Function* function) {
    *function = nullptr;
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return false;
    }
    std::memcpy(function, &symbol, sizeof(Function));
    return true;
}

inline bool bind_pair(void* handle, CblasApi* api) {
    return bind_symbol(handle, "cblas_sgemm", &api->sgemm) &&
           bind_symbol(handle, "cblas_dgemm", &api->dgemm);
}

inline CblasApi load_cblas() {
    const char* requested = std::getenv("PYQCD_EINSUM_CBLAS_LIB");
    if (requested != nullptr && requested[0] != '\0') {
        void* handle = dlopen(requested, RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            CblasApi api;
            if (bind_pair(handle, &api)) {
                return api;
            }
            dlclose(handle);
        }
    }
    CblasApi api;
    if (bind_pair(RTLD_DEFAULT, &api)) {
        return api;
    }
    const char* candidates[] = {"libcblas.so.3", "libcblas.so", "libopenblas.so.0",
                                "libopenblas.so", "libcblas.dylib", "libopenblas.dylib"};
    for (const char* candidate : candidates) {
        void* handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            continue;
        }
        CblasApi local;
        if (bind_pair(handle, &local)) {
            return local;
        }
        dlclose(handle);
    }
    return CblasApi{};
}

inline const CblasApi& cblas_api() {
    static CblasApi api;
    static std::once_flag initialized;
    std::call_once(initialized, [] { api = load_cblas(); });
    return api;
}

/* ------------------------------------------------------------------ 基础 */

enum {
    /* CBLAS 枚举值，与 <cblas.h> 一致：make.sh 允许构建期找不到 OpenBLAS 头文件
     * （运行期 dlopen），因此这里不 include <cblas.h>。 */
    kCblasRowMajor = 101,
    kCblasNoTrans = 111
};

inline bool checked_power(int64_t dim, int exponent, int64_t* result) {
    int64_t value = 1;
    for (int step = 0; step < exponent; ++step) {
        if (value > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        value *= dim;
    }
    *result = value;
    return true;
}

inline int64_t power_or_zero(int64_t dim, int exponent) {
    int64_t value = 0;
    return checked_power(dim, exponent, &value) ? value : 0;
}

/* 沿最内层轴拷 dim 个元素；unit stride 时退化为整块 memcpy。 */
template <typename T>
inline void gather_axis(T* destination, const T* source, int64_t stride, int64_t dim) {
    if (stride == 1) {
        std::memcpy(destination, source, sizeof(T) * static_cast<std::size_t>(dim));
        return;
    }
    for (int64_t index = 0; index < dim; ++index) {
        destination[index] = source[index * stride];
    }
}

/* 把 source 按 ``strides``（每个目的轴的源步长，单位：元素）gather 成连续的
 * destination；destination 的元素数 = dim^rank。 */
template <typename T>
inline void gather(const T* source, const int64_t* strides, int rank, int64_t dim, T* destination,
                   int64_t count) {
    if (rank <= 0) {
        if (count > 0) {
            destination[0] = source[0];
        }
        return;
    }
    int64_t coordinate[kMaxRank];
    for (int axis = 0; axis < rank; ++axis) {
        coordinate[axis] = 0;
    }
    const int64_t inner_stride = strides[rank - 1];
    const int64_t rows = count / dim;
    int64_t offset = 0;
    T* cursor = destination;
    for (int64_t row = 0; row < rows; ++row) {
        gather_axis(cursor, source + offset, inner_stride, dim);
        cursor += dim;
        for (int axis = rank - 2; axis >= 0; --axis) {
            offset += strides[axis];
            if (++coordinate[axis] < dim) {
                break;
            }
            coordinate[axis] = 0;
            offset -= strides[axis] * dim;
        }
    }
}

/* 4 bit/轴 的轴列表读取器（高位在前）。 */
struct AxisReader {
    const uint8_t* data;
    int64_t position = 0;

    int next() {
        const uint8_t packed = data[position >> 1];
        const int value = ((position & 1) == 0) ? (packed >> 4) : (packed & 0x0F);
        ++position;
        return value;
    }
};

/* 步头：a_src/b_src/dst + 输入秩 + 分组秩。M/N 的秩由 rank 与 batch/inner 复原：
 *   rank_a = batch + M + K, rank_b = batch + K + N, rank_c = batch + M + N。 */
struct Step {
    int a_src = 0;
    int b_src = 0;
    int dst = -1;
    int rank_a = 0;
    int rank_b = 0;
    int rank_c = 0;
    int batch_rank = 0;
    int inner_rank = 0;
    /* 指向本条步自己的那 4 bit/轴 流：a_axes | b_axes | c_axes。 */
    const uint8_t* axes = nullptr;

    int row_rank() const { return rank_a - batch_rank - inner_rank; }
    int col_rank() const { return rank_b - batch_rank - inner_rank; }
};

/* 中间槽：操作数槽用 external 指向入参，其余槽的 owned 由本执行器分配。 */
struct Slot {
    const void* external = nullptr;
    void* owned = nullptr;
    std::size_t owned_bytes = 0;
    bool is_external = false;

    bool empty() const { return external == nullptr && owned == nullptr; }
    const void* read() const { return is_external ? external : owned; }

    void release_owned() {
        if (owned != nullptr) {
            block_pool().release(static_cast<uint8_t*>(owned), owned_bytes);
            owned = nullptr;
            owned_bytes = 0;
        }
    }
};

/* ------------------------------------------------------------------ GEMM */

inline void gemm_call(const CblasApi& api, int rows, int cols, int inner, const float* a,
                      const float* b, float* c) {
    api.sgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, rows, cols, inner, 1.0f, a, inner, b,
              cols, 0.0f, c, cols);
}

inline void gemm_call(const CblasApi& api, int rows, int cols, int inner, const double* a,
                      const double* b, double* c) {
    api.dgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, rows, cols, inner, 1.0, a, inner, b,
              cols, 0.0, c, cols);
}

/* 单步：打包 → 批量 GEMM，结果落在 ``packed_output``（[batch, M, N] 连续）。 */
template <typename T>
inline int run_step(const Step& step, const CblasApi& api, const Slot& left, const Slot& right,
                    int64_t dim, T* packed_output) {
    const int64_t count_batch = power_or_zero(dim, step.batch_rank);
    const int64_t count_rows = power_or_zero(dim, step.row_rank());
    const int64_t count_inner = power_or_zero(dim, step.inner_rank);
    const int64_t count_cols = power_or_zero(dim, step.col_rank());
    if (count_batch == 0 || count_rows == 0 || count_inner == 0 || count_cols == 0) {
        return QCD_STATUS_INVALID;
    }
    if (count_rows > std::numeric_limits<int>::max() ||
        count_cols > std::numeric_limits<int>::max() ||
        count_inner > std::numeric_limits<int>::max()) {
        return QCD_STATUS_INVALID;
    }
    if (left.empty() || right.empty() || packed_output == nullptr) {
        return QCD_STATUS_INVALID;
    }

    AxisReader reader{step.axes, 0};
    int a_axes[kMaxRank];
    int b_axes[kMaxRank];
    for (int axis = 0; axis < step.rank_a; ++axis) {
        a_axes[axis] = reader.next();
        if (a_axes[axis] >= step.rank_a) {
            return QCD_STATUS_INVALID;
        }
    }
    for (int axis = 0; axis < step.rank_b; ++axis) {
        b_axes[axis] = reader.next();
        if (b_axes[axis] >= step.rank_b) {
            return QCD_STATUS_INVALID;
        }
    }

    int64_t stride[kMaxRank];
    int64_t a_count = 0;
    int64_t b_count = 0;
    if (!checked_power(dim, step.rank_a, &a_count) || !checked_power(dim, step.rank_b, &b_count)) {
        return QCD_STATUS_INVALID;
    }
    for (int axis = 0; axis < step.rank_a; ++axis) {
        if (!checked_power(dim, step.rank_a - 1 - a_axes[axis], &stride[axis])) {
            return QCD_STATUS_INVALID;
        }
    }
    std::size_t a_bytes = static_cast<std::size_t>(a_count) * sizeof(T);
    std::size_t b_bytes = static_cast<std::size_t>(b_count) * sizeof(T);
    T* packed_a = reinterpret_cast<T*>(block_pool().acquire(a_bytes, &a_bytes));
    if (packed_a == nullptr) {
        return QCD_STATUS_MEMORY;
    }
    gather<T>(static_cast<const T*>(left.read()), stride, step.rank_a, dim, packed_a, a_count);

    for (int axis = 0; axis < step.rank_b; ++axis) {
        if (!checked_power(dim, step.rank_b - 1 - b_axes[axis], &stride[axis])) {
            block_pool().release(reinterpret_cast<uint8_t*>(packed_a), a_bytes);
            return QCD_STATUS_INVALID;
        }
    }
    T* packed_b = reinterpret_cast<T*>(block_pool().acquire(b_bytes, &b_bytes));
    if (packed_b == nullptr) {
        block_pool().release(reinterpret_cast<uint8_t*>(packed_a), a_bytes);
        return QCD_STATUS_MEMORY;
    }
    gather<T>(static_cast<const T*>(right.read()), stride, step.rank_b, dim, packed_b, b_count);

    /* 批量 GEMM：C_b = A_b (M x K) · B_b (K x N)。 */
    for (int64_t batch = 0; batch < count_batch; ++batch) {
        const T* a_block = packed_a + batch * count_rows * count_inner;
        const T* b_block = packed_b + batch * count_inner * count_cols;
        T* c_block = packed_output + batch * count_rows * count_cols;
        gemm_call(api, static_cast<int>(count_rows), static_cast<int>(count_cols),
                  static_cast<int>(count_inner), a_block, b_block, c_block);
    }
    block_pool().release(reinterpret_cast<uint8_t*>(packed_a), a_bytes);
    block_pool().release(reinterpret_cast<uint8_t*>(packed_b), b_bytes);
    return QCD_STATUS_OK;
}

/* 规范序 [batch..., M..., N...] 里第 ``canonical`` 轴在打包缓冲里的步长。
 * 组内还要按局部序号取 dim 的幂：同组内相邻轴的步长差一个 dim。 */
inline bool packed_stride(int canonical, const Step& step, int64_t dim, int64_t count_rows,
                          int64_t count_cols, int64_t* result) {
    int64_t local = 0;
    if (canonical < step.batch_rank) {
        if (!checked_power(dim, step.batch_rank - 1 - canonical, &local)) {
            return false;
        }
        *result = local * count_rows * count_cols;
        return true;
    }
    if (canonical < step.batch_rank + step.row_rank()) {
        if (!checked_power(dim, step.row_rank() - 1 - (canonical - step.batch_rank), &local)) {
            return false;
        }
        *result = local * count_cols;
        return true;
    }
    if (!checked_power(dim, step.col_rank() - 1 - (canonical - step.batch_rank - step.row_rank()),
                       &local)) {
        return false;
    }
    *result = local;
    return true;
}

/* 解包：``source`` 里的 [batch, M, N] 结果按 c_axes gather 成输出下标顺序。 */
template <typename T>
inline int unpack_step(const Step& step, int64_t dim, const T* source, T* destination) {
    const int64_t count_rows = power_or_zero(dim, step.row_rank());
    const int64_t count_cols = power_or_zero(dim, step.col_rank());
    const int64_t count_output = power_or_zero(dim, step.rank_c);
    if (count_rows == 0 || count_cols == 0 || count_output == 0) {
        return QCD_STATUS_INVALID;
    }
    AxisReader reader{step.axes, step.rank_a + step.rank_b};
    int64_t stride[kMaxRank];
    for (int axis = 0; axis < step.rank_c; ++axis) {
        const int canonical = reader.next();
        if (canonical >= step.rank_c ||
            !packed_stride(canonical, step, dim, count_rows, count_cols, &stride[axis])) {
            return QCD_STATUS_INVALID;
        }
    }
    gather<T>(source, stride, step.rank_c, dim, destination, count_output);
    return QCD_STATUS_OK;
}

/* 一步是否需要解包（c_axes 恒等于规范序时可以直接落在目的缓冲区）。 */
inline bool step_output_is_canonical(const Step& step) {
    AxisReader reader{step.axes, step.rank_a + step.rank_b};
    for (int axis = 0; axis < step.rank_c; ++axis) {
        if (reader.next() != axis) {
            return false;
        }
    }
    return true;
}

/* 一条拓扑：依次执行步表。``code`` 指向该拓扑自己的那一段（已按 offset 偏移）。 */
template <typename T>
inline int run_typed(const Record& record, const uint8_t* code, const void* const* operands,
                     void* out, int64_t dim) {
    if (code == nullptr || operands == nullptr || out == nullptr || dim <= 0 ||
        record.arity == 0 || record.step_count == 0 || record.step_count > 200) {
        return QCD_STATUS_INVALID;
    }
    if (record.arity > kMaxSlots) {
        return QCD_STATUS_INVALID;
    }
    const CblasApi& api = cblas_api();
    if (!api.complete()) {
        return QCD_STATUS_INVALID;
    }

    Slot slots[kMaxSlots];
    for (int index = 0; index < record.arity; ++index) {
        if (operands[index] == nullptr) {
            return QCD_STATUS_INVALID;
        }
        slots[index].external = operands[index];
        slots[index].is_external = true;
    }

    const uint8_t* const code_end = code + record.code_bytes;
    const uint8_t* cursor = code;
    int status = QCD_STATUS_OK;
    for (int step_index = 0; step_index < record.step_count && status == QCD_STATUS_OK;
         ++step_index) {
        if (cursor + 6 > code_end) {
            status = QCD_STATUS_INVALID;
            break;
        }
        Step step;
        step.a_src = static_cast<int8_t>(cursor[0]);
        step.b_src = static_cast<int8_t>(cursor[1]);
        step.dst = static_cast<int8_t>(cursor[2]);
        step.rank_a = cursor[3] >> 4;
        step.rank_b = cursor[3] & 0x0F;
        step.rank_c = cursor[4];
        step.batch_rank = cursor[5] >> 4;
        step.inner_rank = cursor[5] & 0x0F;
        if (step.rank_a < 0 || step.rank_b < 0 || step.rank_c < 0 || step.rank_a > kMaxRank ||
            step.rank_b > kMaxRank || step.rank_c > kMaxRank) {
            status = QCD_STATUS_INVALID;
            break;
        }
        if (step.batch_rank + step.inner_rank > step.rank_a ||
            step.batch_rank + step.inner_rank > step.rank_b ||
            step.row_rank() < 0 || step.col_rank() < 0 ||
            step.row_rank() + step.col_rank() + step.batch_rank != step.rank_c) {
            status = QCD_STATUS_INVALID;
            break;
        }
        const int axes_bytes = (step.rank_a + step.rank_b + step.rank_c + 1) / 2;
        if (cursor + 6 + axes_bytes > code_end) {
            status = QCD_STATUS_INVALID;
            break;
        }
        step.axes = cursor + 6;
        cursor += 6 + axes_bytes;

        /* 槽位映射：步表里 0..arity-1 是操作数、-1-slot 是中间体；运行期数组把
         * 中间体放在 arity 之后，两段不重叠。 */
        const int slot_a = step.a_src >= 0 ? step.a_src
                                           : record.arity + (-1 - static_cast<int>(step.a_src));
        const int slot_b = step.b_src >= 0 ? step.b_src
                                           : record.arity + (-1 - static_cast<int>(step.b_src));
        const int slot_c = step.dst >= 0 ? record.arity + static_cast<int>(step.dst) : -1;
        if (slot_a >= kMaxSlots || slot_b >= kMaxSlots || slot_c >= kMaxSlots ||
            slots[slot_a].empty() || slots[slot_b].empty()) {
            status = QCD_STATUS_INVALID;
            break;
        }

        /* 目的缓冲区：中间槽（每步一块，消费后立刻归还）或最终输出。 */
        void* destination = out;
        if (slot_c >= 0) {
            if (!slots[slot_c].empty()) {
                status = QCD_STATUS_INVALID;
                break;
            }
            const int64_t elements = power_or_zero(dim, step.rank_c);
            if (elements == 0) {
                status = QCD_STATUS_INVALID;
                break;
            }
            std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
            void* buffer = block_pool().acquire(bytes, &bytes);
            if (buffer == nullptr) {
                status = QCD_STATUS_MEMORY;
                break;
            }
            slots[slot_c].owned = buffer;
            slots[slot_c].owned_bytes = bytes;
            slots[slot_c].is_external = false;
            destination = buffer;
        }

        /* 输出轴序恰好等于规范序时省掉解包（GEMM 直接落盘）。 */
        const bool direct = step_output_is_canonical(step);
        std::size_t packed_bytes = static_cast<std::size_t>(power_or_zero(dim, step.rank_c)) *
                                   sizeof(T);
        T* packed_output = static_cast<T*>(destination);
        if (!direct) {
            packed_output = reinterpret_cast<T*>(block_pool().acquire(packed_bytes, &packed_bytes));
            if (packed_output == nullptr) {
                status = QCD_STATUS_MEMORY;
            }
        }
        if (status == QCD_STATUS_OK) {
            status = run_step<T>(step, api, slots[slot_a], slots[slot_b], dim, packed_output);
            if (status == QCD_STATUS_OK && !direct) {
                status = unpack_step<T>(step, dim, packed_output, static_cast<T*>(destination));
            }
        }
        if (!direct) {
            block_pool().release(reinterpret_cast<uint8_t*>(packed_output), packed_bytes);
        }

        /* 一步的输入中间体到此为止（收缩树里每个中间体只被消费一次）。 */
        if (slot_a >= record.arity) {
            slots[slot_a].release_owned();
        }
        if (slot_b >= record.arity) {
            slots[slot_b].release_owned();
        }
    }
    return status;
}

inline int run(const Record& record, const uint8_t* code, const void* const* operands, void* out,
               int elem_kind, int64_t dim) {
    if (elem_kind == kFloat) {
        return run_typed<float>(record, code, operands, out, dim);
    }
    if (elem_kind == kDouble) {
        return run_typed<double>(record, code, operands, out, dim);
    }
    return QCD_STATUS_INVALID;
}

}  // namespace topology_gemm
}  // namespace pyqcd_einsum

#endif /* PYQCD_KUPL_EINSUM_TOPOLOGY_GEMM_H */
