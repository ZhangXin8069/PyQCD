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
 * **精度**：四类元素类型各实例化一条路径（同一模板、同一张步表）——``float`` /
 * ``double`` / ``ComplexElement<float>`` / ``ComplexElement<double>``，与
 * ``QCD_DTYPE_*`` 的 0..3 一一对应。调用方按 dtype 显式选择，**不设主精度**：
 * 哪一类能用只取决于运行期 CBLAS 是否导出对应的 ``{s,d,c,z}gemm``；缺符号时
 * 该类返回不匹配状态由上层回落，绝不隐式降精度。复数按交错 re/im 布局直传，
 * 正是 CBLAS 的复数口径，因此打包/转置仍是纯字节搬运。
 *
 * **中间体**：每一步的输出是一块显式大小的缓冲区，来自与 ``src/apply_contract.cpp``
 * 同策略的进程级块池（acquire 最佳适配 / release 只回收大块），消费后立刻归还，
 * 因此峰值只占相邻两步的几块内存，不会每步 malloc。apply_contract.cpp 的
 * BufferPool 位于匿名命名空间、不可跨 TU 复用，这里按同样口径自带最小实现。
 */

namespace pyqcd_einsum {
namespace topology_gemm {

// 元素种类与 QCD_DTYPE_* 逐项同值（complex128=0, complex64=1, float64=2, float32=3），
// 上层因此可以直接把 dtype_id 当下发参数。kFloat/kDouble 是历史名字，保留以便
// 生成的步表（topology_kernels.inc 的 topo_run_f/topo_run_d）继续编译。
enum ElementKind : int {
    kComplex128 = QCD_DTYPE_COMPLEX128,
    kComplex64 = QCD_DTYPE_COMPLEX64,
    kDouble = QCD_DTYPE_FLOAT64,
    kFloat = QCD_DTYPE_FLOAT32
};

/* 交错 re/im 的 POD 复数元素：与 CBLAS 的复数口径、numpy 的 complex 内存布局
 * 完全一致（``reinterpret_cast`` 即用），也与 PyQCU ``LatticeComplex<T>`` 的
 * ``float2``/``double2`` 存储同构。必须保持平凡可拷贝，打包/转置才能退化为 memcpy。 */
template <typename T>
struct ComplexElement {
    T real;
    T imag;
};

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
    /* 步表格式版本：2 = 只存轴序（轴长靠 dim 幂次，仅等长超立方体）；
     * 3 = 段首带 label 块、每步带 c_labels（轴长来自 label→size，支持异形轴）。 */
    uint8_t version = 2;
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
/* 复数 GEMM 的 alpha/beta 是**指针**（CBLAS 口径），元素按交错 re/im 直传。 */
using CgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);
using ZgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);

struct CblasApi {
    SgemmFn sgemm = nullptr;
    DgemmFn dgemm = nullptr;
    CgemmFn cgemm = nullptr;
    ZgemmFn zgemm = nullptr;

    bool complete() const { return sgemm != nullptr && dgemm != nullptr; }
    bool complex_complete() const { return cgemm != nullptr && zgemm != nullptr; }
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
    const bool real = bind_symbol(handle, "cblas_sgemm", &api->sgemm) &&
                      bind_symbol(handle, "cblas_dgemm", &api->dgemm);
    /* 复数符号是**可选**的：只提供实数的 BLAS 仍服务 float/double，复数 dtype
     * 由 run() 按自己的符号判定不可用并回落 —— 不让缺 zgemm 连累实数路径。 */
    bind_symbol(handle, "cblas_cgemm", &api->cgemm);
    bind_symbol(handle, "cblas_zgemm", &api->zgemm);
    return real;
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

/* 转置专用的分块搬运：源与目的都是**超立方体**的轴重排时，用二维分块把源/目的
 * 两侧的缓存行都吃满。
 *
 * 实测动机：14 算子路径的最后一步是 ``k=1`` 的「外积」（2500x2500 = 6.25M 元素），
 * 解包要把它重排成 ``(50,50,50,50)``。逐行 gather 在这步花 87 ms，占总时间 37%；
 * 分块转置把同一批 cache line 的读写集中，实测降到 20 ms 量级。
 *
 * 做法：把 rank 轴分成「外层若干轴」与「最内 two 轴」，外层按块步进，内层做
 * 经典 2D 分块转置（BLOCK x BLOCK 的临时缓冲，两个方向都连续写）。
 */
constexpr int64_t kTransposeBlock = 32;

template <typename T>
inline void blocked_transpose_2d(const T* source, T* destination, int64_t rows, int64_t cols,
                                 int64_t source_row_stride, int64_t source_col_stride,
                                 int64_t destination_row_stride,
                                 int64_t destination_col_stride) {
    T tile[kTransposeBlock * kTransposeBlock];
    for (int64_t row = 0; row < rows; row += kTransposeBlock) {
        const int64_t row_end = row + kTransposeBlock < rows ? row + kTransposeBlock : rows;
        for (int64_t col = 0; col < cols; col += kTransposeBlock) {
            const int64_t col_end = col + kTransposeBlock < cols ? col + kTransposeBlock : cols;
            /* 先按源侧连续方向读进 tile（源列方向步长通常为 1）。 */
            for (int64_t r = row; r < row_end; ++r) {
                const T* from = source + r * source_row_stride;
                T* buffer = tile + (r - row) * kTransposeBlock;
                for (int64_t c = col; c < col_end; ++c) {
                    buffer[c - col] = from[c * source_col_stride];
                }
            }
            /* 再按目的侧连续方向写出。 */
            for (int64_t c = col; c < col_end; ++c) {
                T* to = destination + c * destination_col_stride;
                const T* buffer = tile + (c - col);
                for (int64_t r = row; r < row_end; ++r) {
                    to[r * destination_row_stride] = buffer[(r - row) * kTransposeBlock];
                }
            }
        }
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

/* 与 ``gather`` 同语义，但每轴长度来自 ``extents``（真实轴长，可互不相同）。
 * v3 执行器用它；v2 的 ``gather`` 仍按标量 ``dim`` 走，两者互不影响。 */
template <typename T>
inline void gather_extents(const T* source, const int64_t* strides, int rank,
                           const int64_t* extents, T* destination, int64_t count) {
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
    const int64_t inner_extent = extents[rank - 1];
    const int64_t inner_stride = strides[rank - 1];
    const int64_t rows = inner_extent > 0 ? count / inner_extent : 0;
    int64_t offset = 0;
    T* cursor = destination;
    for (int64_t row = 0; row < rows; ++row) {
        gather_axis(cursor, source + offset, inner_stride, inner_extent);
        cursor += inner_extent;
        for (int axis = rank - 2; axis >= 0; --axis) {
            offset += strides[axis];
            if (++coordinate[axis] < extents[axis]) {
                break;
            }
            coordinate[axis] = 0;
            offset -= strides[axis] * extents[axis];
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

inline void gemm_call(const CblasApi& api, int rows, int cols, int inner,
                      const ComplexElement<float>* a, const ComplexElement<float>* b,
                      ComplexElement<float>* c) {
    const float alpha[2] = {1.0f, 0.0f};
    const float beta[2] = {0.0f, 0.0f};
    api.cgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, rows, cols, inner,
              static_cast<const void*>(alpha), static_cast<const void*>(a), inner,
              static_cast<const void*>(b), cols, static_cast<const void*>(beta),
              static_cast<void*>(c), cols);
}

inline void gemm_call(const CblasApi& api, int rows, int cols, int inner,
                      const ComplexElement<double>* a, const ComplexElement<double>* b,
                      ComplexElement<double>* c) {
    const double alpha[2] = {1.0, 0.0};
    const double beta[2] = {0.0, 0.0};
    api.zgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, rows, cols, inner,
              static_cast<const void*>(alpha), static_cast<const void*>(a), inner,
              static_cast<const void*>(b), cols, static_cast<const void*>(beta),
              static_cast<void*>(c), cols);
}

/* 元素类型 → 该类型可用的 GEMM 符号。四类 dtype 一视同仁：谁缺符号谁回落，
 * 没有任何一类被当成"默认精度"而豁免检查。 */
inline bool gemm_available(const CblasApi& api, const float*) { return api.sgemm != nullptr; }
inline bool gemm_available(const CblasApi& api, const double*) { return api.dgemm != nullptr; }
inline bool gemm_available(const CblasApi& api, const ComplexElement<float>*) {
    return api.cgemm != nullptr;
}
inline bool gemm_available(const CblasApi& api, const ComplexElement<double>*) {
    return api.zgemm != nullptr;
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
    int canonical_of[kMaxRank];
    for (int axis = 0; axis < step.rank_c; ++axis) {
        const int canonical = reader.next();
        canonical_of[axis] = canonical;
        if (canonical >= step.rank_c ||
            !packed_stride(canonical, step, dim, count_rows, count_cols, &stride[axis])) {
            return QCD_STATUS_INVALID;
        }
    }
    /* 纯二维互换：输出恰好是「前 h 个轴 <-> 后 (rank-h) 个轴」两组整体交换时，
     * 用分块转置代替逐行 gather。这是 14-16 算子路径最后一步（k=1 外积）的
     * 常见形状，实测占该路径 37% 的时间。 */
    if (step.rank_c >= 2) {
        int split = -1;
        for (int axis = 1; axis < step.rank_c; ++axis) {
            /* 目的轴 [0,axis) 对应源轴 [rank-axis, rank)，两者各自连续递增。 */
            bool swap = true;
            for (int index = 0; index < axis && swap; ++index) {
                swap = canonical_of[index] == step.rank_c - axis + index;
            }
            for (int index = axis; index < step.rank_c && swap; ++index) {
                swap = canonical_of[index] == index - axis;
            }
            if (swap) {
                split = axis;
                break;
            }
        }
        if (split > 0 && split < step.rank_c) {
            const int64_t left = power_or_zero(dim, split);
            const int64_t right = power_or_zero(dim, step.rank_c - split);
            if (left > 0 && right > 0 &&
                left * right == count_output) {
                /* 源布局：left 为行（步长 right）、right 为列（步长 1）；
                 * 目的布局：right 为行（步长 left）、left 为列（步长 1）。 */
                blocked_transpose_2d<T>(source, destination, left, right,
                                        right, 1, 1, left);
                return QCD_STATUS_OK;
            }
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
    if (!gemm_available(api, static_cast<const T*>(nullptr))) {
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
    switch (elem_kind) {
        case kComplex128:
            return run_typed<ComplexElement<double>>(record, code, operands, out, dim);
        case kComplex64:
            return run_typed<ComplexElement<float>>(record, code, operands, out, dim);
        case kDouble:
            return run_typed<double>(record, code, operands, out, dim);
        case kFloat:
            return run_typed<float>(record, code, operands, out, dim);
        default:
            return QCD_STATUS_INVALID;
    }
}

/* ============================ 步表 v3 执行器 ================================
 *
 * 与上面的 v2 执行器并存，互不影响：v2 只用``dim``幂次（仅等长超立方体），
 * v3 用 label→size 表（支持生产异形轴）。调用方按 ``Record.version`` 选择。
 *
 * v3 字节流（生成端：``pyqcd/testing/_topology_kernel_gen.encode_steps``）：
 *
 *   段首  uint8 arity
 *         逐操作数: uint8 rank + rank×uint8 label_id（canonical 编号）
 *   每步  v2 的 6 字节头 + 轴序 nibble 流 + uint8 c_labels[rank_c]
 *
 * ``c_labels`` 是该步输出**目标轴序**的 label；中间体 unpack 后按此序存放，
 * 下一步按位置引用。运行期只需 ``sizes[label]`` 就能拿到逐轴真实长度。
 * ========================================================================= */

/* 逐步的真实轴长推演结果，供 v3 执行器内部使用。 */
struct V3StepInfo {
    int64_t batch_extents[kMaxRank];
    int64_t row_extents[kMaxRank];
    int64_t col_extents[kMaxRank];
    int64_t output_extents[kMaxRank];
    int64_t count_batch = 1;
    int64_t count_rows = 1;
    int64_t count_inner = 1;
    int64_t count_cols = 1;
    int slot_a = -1;
    int slot_b = -1;
    int slot_c = -1;
};

/* 一段 v3 段首的 label 块解码结果。 */
struct V3Header {
    int arity = 0;
    int ranks[kMaxSlots];
    uint8_t labels[kMaxSlots][kMaxRank];
    int consumed = 0;
};

inline bool v3_parse_header(const uint8_t* code, std::size_t code_bytes, V3Header* header) {
    if (code == nullptr || header == nullptr || code_bytes < 1) {
        return false;
    }
    header->arity = code[0];
    if (header->arity <= 0 || header->arity > kMaxSlots) {
        return false;
    }
    std::size_t position = 1;
    for (int index = 0; index < header->arity; ++index) {
        if (position >= code_bytes) {
            return false;
        }
        const int rank = code[position++];
        if (rank < 0 || rank > kMaxRank) {
            return false;
        }
        header->ranks[index] = rank;
        for (int axis = 0; axis < rank; ++axis) {
            if (position >= code_bytes) {
                return false;
            }
            header->labels[index][axis] = code[position++];
        }
    }
    header->consumed = static_cast<int>(position);
    return true;
}

/* 按 label 取长度并填好一步的分组长度。 */
inline bool v3_resolve_step(const Step& step, const uint8_t* a_labels,
                            const uint8_t* b_labels, const uint8_t* c_labels,
                            const int64_t* label_sizes, int num_labels,
                            V3StepInfo* info) {
    const int batch = step.batch_rank;
    const int inner = step.inner_rank;
    const int row_rank = step.row_rank();
    const int col_rank = step.col_rank();
    auto extent = [&](uint8_t label) -> int64_t {
        if (static_cast<int>(label) >= num_labels) {
            return 0;
        }
        return label_sizes[label];
    };
    for (int axis = 0; axis < batch; ++axis) {
        const int64_t left = extent(a_labels[axis]);
        const int64_t right = extent(b_labels[axis]);
        if (left <= 0 || left != right) {
            return false;
        }
        info->batch_extents[axis] = left;
        info->count_batch *= left;
    }
    for (int axis = 0; axis < row_rank; ++axis) {
        const int64_t value = extent(a_labels[batch + axis]);
        if (value <= 0) {
            return false;
        }
        info->row_extents[axis] = value;
        info->count_rows *= value;
    }
    for (int axis = 0; axis < inner; ++axis) {
        const int64_t left = extent(a_labels[batch + row_rank + axis]);
        const int64_t right = extent(b_labels[batch + axis]);
        if (left <= 0 || left != right) {
            return false;
        }
        info->count_inner *= left;
    }
    for (int axis = 0; axis < col_rank; ++axis) {
        const int64_t value = extent(b_labels[batch + inner + axis]);
        if (value <= 0) {
            return false;
        }
        info->col_extents[axis] = value;
        info->count_cols *= value;
    }
    for (int axis = 0; axis < step.rank_c; ++axis) {
        const int64_t value = extent(c_labels[axis]);
        if (value <= 0) {
            return false;
        }
        info->output_extents[axis] = value;
    }
    return info->count_batch > 0 && info->count_rows > 0 && info->count_inner > 0 &&
           info->count_cols > 0;
}

/* 打包 + 批量 GEMM（v3：长度全部来自 ``V3StepInfo``）。 */
/* ``a_source_len`` / ``b_source_len`` 是 A/B 操作数**按自身轴序**的逐轴长度
 * （即 ``slot_lengths[slot]``）；``a_labels_len`` / ``b_labels_len`` 是它们在
 * **规范序**（batch|M|K / batch|K|N）下重排后的长度。
 *
 * 必须区分这两组：源张量按自身轴序存放，取其步长要用 ``a_source_len`` 的后缀积；
 * 曾经误用规范序长度算后缀积，等长时两者数值相同（都等于 ``dim^k``）所以测不出，
 * 异形轴时错位 → 越界读 → float64 上表现为 nan 或段错误。 */
template <typename T>
inline int v3_run_step(const Step& step, const CblasApi& api, const Slot& left,
                       const Slot& right, const int64_t* a_source_len, int a_source_rank,
                       const int64_t* b_source_len, int b_source_rank,
                       const int64_t* a_labels_len, const int64_t* b_labels_len,
                       const V3StepInfo& info, T* packed_output) {
    if (info.count_rows > std::numeric_limits<int>::max() ||
        info.count_cols > std::numeric_limits<int>::max() ||
        info.count_inner > std::numeric_limits<int>::max()) {
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
    /* 源侧后缀积必须按**源自身轴序**的长度算。 */
    int64_t source_stride[kMaxRank];
    int64_t total = 1;
    for (int axis = a_source_rank - 1; axis >= 0; --axis) {
        if (a_source_len[axis] <= 0) {
            return QCD_STATUS_INVALID;
        }
        source_stride[axis] = total;
        total *= a_source_len[axis];
    }
    int64_t stride[kMaxRank];
    int64_t a_count = 1;
    for (int axis = 0; axis < step.rank_a; ++axis) {
        const int source_axis = a_axes[axis];
        if (source_axis < 0 || source_axis >= a_source_rank) {
            return QCD_STATUS_INVALID;
        }
        stride[axis] = source_stride[source_axis];
        a_count *= a_labels_len[axis];
    }
    std::size_t a_bytes = static_cast<std::size_t>(a_count) * sizeof(T);
    T* packed_a = reinterpret_cast<T*>(block_pool().acquire(a_bytes, &a_bytes));
    if (packed_a == nullptr) {
        return QCD_STATUS_MEMORY;
    }
    gather_extents<T>(static_cast<const T*>(left.read()), stride, step.rank_a,
                      a_labels_len, packed_a, a_count);

    total = 1;
    for (int axis = b_source_rank - 1; axis >= 0; --axis) {
        if (b_source_len[axis] <= 0) {
            return QCD_STATUS_INVALID;
        }
        source_stride[axis] = total;
        total *= b_source_len[axis];
    }
    int64_t b_count = 1;
    for (int axis = 0; axis < step.rank_b; ++axis) {
        const int source_axis = b_axes[axis];
        if (source_axis < 0 || source_axis >= b_source_rank) {
            return QCD_STATUS_INVALID;
        }
        stride[axis] = source_stride[source_axis];
        b_count *= b_labels_len[axis];
    }
    std::size_t b_bytes = static_cast<std::size_t>(b_count) * sizeof(T);
    T* packed_b = reinterpret_cast<T*>(block_pool().acquire(b_bytes, &b_bytes));
    if (packed_b == nullptr) {
        block_pool().release(reinterpret_cast<uint8_t*>(packed_a), a_bytes);
        return QCD_STATUS_MEMORY;
    }
    gather_extents<T>(static_cast<const T*>(right.read()), stride, step.rank_b,
                      b_labels_len, packed_b, b_count);

    for (int64_t batch = 0; batch < info.count_batch; ++batch) {
        const T* a_block = packed_a + batch * info.count_rows * info.count_inner;
        const T* b_block = packed_b + batch * info.count_inner * info.count_cols;
        T* c_block = packed_output + batch * info.count_rows * info.count_cols;
        gemm_call(api, static_cast<int>(info.count_rows), static_cast<int>(info.count_cols),
                  static_cast<int>(info.count_inner), a_block, b_block, c_block);
    }
    block_pool().release(reinterpret_cast<uint8_t*>(packed_a), a_bytes);
    block_pool().release(reinterpret_cast<uint8_t*>(packed_b), b_bytes);
    return QCD_STATUS_OK;
}

/* 解包（v3）：[batch, M, N] → 目的轴序。 */
template <typename T>
inline int v3_unpack_step(const Step& step, const V3StepInfo& info,
                          const T* source, T* destination) {
    const int batch = step.batch_rank;
    const int row_rank = step.row_rank();
    const int col_rank = step.col_rank();
    for (int axis = 0; axis < step.rank_c; ++axis) {
        if (!step_output_is_canonical(step)) {
            break;
        }
    }
    AxisReader reader{step.axes, step.rank_a + step.rank_b};
    int64_t stride[kMaxRank];
    for (int axis = 0; axis < step.rank_c; ++axis) {
        const int canonical = reader.next();
        if (canonical < 0 || canonical >= step.rank_c) {
            return QCD_STATUS_INVALID;
        }
        if (canonical < batch) {
            int64_t tail = 1;
            for (int index = canonical + 1; index < batch; ++index) {
                tail *= info.batch_extents[index];
            }
            stride[axis] = tail * info.count_rows * info.count_cols;
        } else if (canonical < batch + row_rank) {
            const int local = canonical - batch;
            int64_t tail = 1;
            for (int index = local + 1; index < row_rank; ++index) {
                tail *= info.row_extents[index];
            }
            stride[axis] = tail * info.count_cols;
        } else {
            const int local = canonical - batch - row_rank;
            int64_t tail = 1;
            for (int index = local + 1; index < col_rank; ++index) {
                tail *= info.col_extents[index];
            }
            stride[axis] = tail;
        }
    }
    int64_t count = 1;
    for (int axis = 0; axis < step.rank_c; ++axis) {
        count *= info.output_extents[axis];
    }
    gather_extents<T>(source, stride, step.rank_c, info.output_extents, destination, count);
    return QCD_STATUS_OK;
}

/* v3 主执行器。 */
template <typename T>
inline int run_typed_v3(const Record& record, const uint8_t* code,
                        const void* const* operands, void* out,
                        const int64_t* label_sizes, int num_labels) {
    if (code == nullptr || operands == nullptr || out == nullptr || label_sizes == nullptr ||
        num_labels <= 0 || record.code_bytes == 0) {
        return QCD_STATUS_INVALID;
    }
    const CblasApi& api = cblas_api();
    if (!gemm_available(api, static_cast<const T*>(nullptr))) {
        return QCD_STATUS_INVALID;
    }
    V3Header header;
    if (!v3_parse_header(code, record.code_bytes, &header)) {
        return QCD_STATUS_INVALID;
    }
    if (header.arity != static_cast<int>(record.arity)) {
        return QCD_STATUS_INVALID;
    }

    Slot slots[kMaxSlots];
    uint8_t slot_labels[kMaxSlots][kMaxRank];
    int slot_rank[kMaxSlots];
    int64_t slot_lengths[kMaxSlots][kMaxRank];
    for (int index = 0; index < kMaxSlots; ++index) {
        slot_rank[index] = 0;
    }
    for (int index = 0; index < header.arity; ++index) {
        if (operands[index] == nullptr) {
            return QCD_STATUS_INVALID;
        }
        slots[index].external = operands[index];
        slots[index].is_external = true;
        slot_rank[index] = header.ranks[index];
        for (int axis = 0; axis < header.ranks[index]; ++axis) {
            const uint8_t label = header.labels[index][axis];
            if (static_cast<int>(label) >= num_labels) {
                return QCD_STATUS_INVALID;
            }
            slot_labels[index][axis] = label;
            slot_lengths[index][axis] = label_sizes[label];
        }
    }

    const uint8_t* cursor = code + header.consumed;
    const uint8_t* const code_end = code + record.code_bytes;
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
        if (step.rank_a < 0 || step.rank_b < 0 || step.rank_c < 0 ||
            step.rank_a > kMaxRank || step.rank_b > kMaxRank || step.rank_c > kMaxRank) {
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
        if (cursor + 6 + axes_bytes + step.rank_c > code_end) {
            status = QCD_STATUS_INVALID;
            break;
        }
        step.axes = cursor + 6;
        const uint8_t* c_labels = cursor + 6 + axes_bytes;
        cursor += 6 + axes_bytes + step.rank_c;

        const int slot_a = step.a_src >= 0 ? step.a_src
                                           : header.arity + (-1 - static_cast<int>(step.a_src));
        const int slot_b = step.b_src >= 0 ? step.b_src
                                           : header.arity + (-1 - static_cast<int>(step.b_src));
        const int slot_c = step.dst >= 0 ? header.arity + static_cast<int>(step.dst) : -1;
        if (slot_a >= kMaxSlots || slot_b >= kMaxSlots || slot_c >= kMaxSlots ||
            slots[slot_a].empty() || slots[slot_b].empty()) {
            status = QCD_STATUS_INVALID;
            break;
        }
        if (slot_rank[slot_a] != step.rank_a || slot_rank[slot_b] != step.rank_b) {
            status = QCD_STATUS_INVALID;
            break;
        }

        /* 按轴序把「槽的逐轴长度」翻成规范分组顺序。 */
        AxisReader order_reader{step.axes, 0};
        int64_t a_len[kMaxRank];
        int64_t b_len[kMaxRank];
        uint8_t a_labels[kMaxRank];
        uint8_t b_labels[kMaxRank];
        for (int axis = 0; axis < step.rank_a; ++axis) {
            const int source_axis = order_reader.next();
            if (source_axis < 0 || source_axis >= step.rank_a) {
                status = QCD_STATUS_INVALID;
                break;
            }
            a_len[axis] = slot_lengths[slot_a][source_axis];
            a_labels[axis] = slot_labels[slot_a][source_axis];
        }
        if (status != QCD_STATUS_OK) {
            break;
        }
        for (int axis = 0; axis < step.rank_b; ++axis) {
            const int source_axis = order_reader.next();
            if (source_axis < 0 || source_axis >= step.rank_b) {
                status = QCD_STATUS_INVALID;
                break;
            }
            b_len[axis] = slot_lengths[slot_b][source_axis];
            b_labels[axis] = slot_labels[slot_b][source_axis];
        }
        if (status != QCD_STATUS_OK) {
            break;
        }

        V3StepInfo info;
        info.slot_a = slot_a;
        info.slot_b = slot_b;
        info.slot_c = slot_c;
        if (!v3_resolve_step(step, a_labels, b_labels, c_labels, label_sizes, num_labels,
                             &info)) {
            status = QCD_STATUS_INVALID;
            break;
        }

        void* destination = out;
        if (slot_c >= 0) {
            if (!slots[slot_c].empty()) {
                status = QCD_STATUS_INVALID;
                break;
            }
            int64_t elements = 1;
            for (int axis = 0; axis < step.rank_c; ++axis) {
                elements *= info.output_extents[axis];
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
            slot_rank[slot_c] = step.rank_c;
            for (int axis = 0; axis < step.rank_c; ++axis) {
                slot_labels[slot_c][axis] = c_labels[axis];
                slot_lengths[slot_c][axis] = info.output_extents[axis];
            }
            destination = buffer;
        }

        const bool direct = step_output_is_canonical(step);
        int64_t packed_elements = info.count_batch * info.count_rows * info.count_cols;
        std::size_t packed_bytes = static_cast<std::size_t>(packed_elements) * sizeof(T);
        T* packed_output = static_cast<T*>(destination);
        if (!direct) {
            packed_output = reinterpret_cast<T*>(block_pool().acquire(packed_bytes, &packed_bytes));
            if (packed_output == nullptr) {
                status = QCD_STATUS_MEMORY;
            }
        }
        if (status == QCD_STATUS_OK) {
            status = v3_run_step<T>(step, api, slots[slot_a], slots[slot_b],
                                    slot_lengths[slot_a], step.rank_a,
                                    slot_lengths[slot_b], step.rank_b,
                                    a_len, b_len, info, packed_output);
            if (status == QCD_STATUS_OK && !direct) {
                status = v3_unpack_step<T>(step, info, packed_output,
                                           static_cast<T*>(destination));
            }
        }
        if (!direct) {
            block_pool().release(reinterpret_cast<uint8_t*>(packed_output), packed_bytes);
        }
        if (slot_a >= header.arity) {
            slots[slot_a].release_owned();
            slot_rank[slot_a] = 0;
        }
        if (slot_b >= header.arity) {
            slots[slot_b].release_owned();
            slot_rank[slot_b] = 0;
        }
    }
    return status;
}

inline int run_v3(const Record& record, const uint8_t* code, const void* const* operands,
                  void* out, int elem_kind, const int64_t* label_sizes, int num_labels) {
    switch (elem_kind) {
        case kComplex128:
            return run_typed_v3<ComplexElement<double>>(record, code, operands, out,
                                                        label_sizes, num_labels);
        case kComplex64:
            return run_typed_v3<ComplexElement<float>>(record, code, operands, out,
                                                       label_sizes, num_labels);
        case kDouble:
            return run_typed_v3<double>(record, code, operands, out, label_sizes, num_labels);
        case kFloat:
            return run_typed_v3<float>(record, code, operands, out, label_sizes, num_labels);
        default:
            return QCD_STATUS_INVALID;
    }
}

}  // namespace topology_gemm
}  // namespace pyqcd_einsum

#endif /* PYQCD_KUPL_EINSUM_TOPOLOGY_GEMM_H */
