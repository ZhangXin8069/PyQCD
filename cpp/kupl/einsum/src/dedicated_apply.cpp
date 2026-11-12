#include "dedicated_apply.h"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <vector>

/* 固定步表执行器：格式见 include/dedicated_apply.h。
 *
 * 设计取舍：
 *   * 步表由 Python 生成端展开到「batch/M/N/K + 三张轴序表」的粒度，原生侧
 *     不做符号解析、不重排收缩顺序，逐步只做 pack → GEMM → unpack；
 *   * 每个操作数/中间槽的源步长、pack 后连续步长、unpack 的源/目的步长都在
 *     parse 阶段一次算好，执行期只做整数乘加；
 *   * 中间槽与 pack 缓冲都来自本 TU 的进程级块池：一次调用**只在开头**分配
 *     （每个中间槽一块 + 3 块 pack 缓冲），步内不再分配。apply_contract.cpp 的
 *     BufferPool 在匿名命名空间里不可跨 TU 复用，这里按同一策略自带最小实现。
 */

namespace {

constexpr std::size_t kHeaderBytes = QCD_DEDICATED_HEADER_BYTES;
constexpr int kMaxRank = QCD_DEDICATED_MAX_RANK;
constexpr int kMaxSlots = QCD_DEDICATED_MAX_SLOTS;
constexpr int kMaxOperands = QCD_DEDICATED_MAX_OPERANDS;
constexpr int kMaxSteps = QCD_DEDICATED_MAX_STEPS;
constexpr std::uint64_t kMaxTableBytes = std::uint64_t(1) << 28;

constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;

/* 一步里某一侧是否可以用 BLAS 的转置参数直接吃原缓冲：
 *   kDirectNone  源轴序既不是恒等也不是组交换 → 必须 pack；
 *   kDirectPlain 源就是 [batch][M][K]（A）或 [batch][K][N]（B）→ trans=0；
 *   kDirectTrans 源是 [batch][K][M]（A）或 [batch][N][K]（B）→ trans=1。
 * 两种 direct 布局的每 batch 元素数都是 m*k（A）/ k*n（B），与 pack 后一致，
 * 因此批循环的偏移公式不变。 */
enum { kDirectNone = -1, kDirectPlain = 0, kDirectTrans = 1 };

/* 诊断开关：PYQCD_EINSUM_DEDICATED_TRACE=1 时逐步打印 pack/GEMM/unpack 耗时与
 * 中间体字节数（stderr）。默认关闭，关闭时不进入任何计时分支。 */
bool trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("PYQCD_EINSUM_DEDICATED_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

/* A/B 基准开关：PYQCD_EINSUM_DEDICATED_BASELINE=1 时退回两条旧口径——不做
 * transA/transB 直通（非恒等轴序一律 pack）、gather 只按规范序从最内轴向外
 * 合并。只用于同一进程内对照新旧路径；每次调用现读环境变量，不缓存。 */
bool baseline_enabled() {
    const char* value = std::getenv("PYQCD_EINSUM_DEDICATED_BASELINE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

using TraceClock = std::chrono::steady_clock;

double trace_ms(TraceClock::time_point begin, TraceClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
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

/* ------------------------------------------------------------------ 块池 */

/* 与 src/apply_contract.cpp 的 BufferPool 同策略：最佳适配、只在块大小落在
 * [bytes, 2*bytes] 时复用，释放小于 512 KiB 的块直接归还系统。 */
class BlockPool {
  public:
    BlockPool() = default;
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    ~BlockPool() {
        for (Block& block : blocks_) {
            std::free(block.data);
        }
    }

    std::uint8_t* acquire(std::size_t bytes, std::size_t* capacity) {
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
        return static_cast<std::uint8_t*>(allocated);
    }

    void release(std::uint8_t* data, std::size_t bytes) {
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
        std::uint8_t* data;
        std::size_t bytes;
    };

    std::mutex mutex_;
    std::vector<Block> blocks_;
    std::size_t cached_ = 0;
};

BlockPool& block_pool() {
    static BlockPool pool;
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
        data_ = block_pool().acquire(bytes, &bytes_);
        return data_ != nullptr;
    }

    std::uint8_t* data() { return data_; }
    const std::uint8_t* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }

  private:
    void release() noexcept {
        block_pool().release(data_, bytes_);
        data_ = nullptr;
        bytes_ = 0;
    }

    std::uint8_t* data_ = nullptr;
    std::size_t bytes_ = 0;
};

/* 单个元素的字节数。复数按交错实虚布局，等于 2× 实部宽度。 */
std::size_t element_size(int dtype_id) {
    switch (dtype_id) {
        case QCD_DTYPE_FLOAT32:
            return sizeof(float);
        case QCD_DTYPE_FLOAT64:
            return sizeof(double);
        case QCD_DTYPE_COMPLEX64:
            return 2 * sizeof(float);
        case QCD_DTYPE_COMPLEX128:
            return 2 * sizeof(double);
        default:
            return 0;
    }
}

/* numpy 的复数在内存里是交错的 [re, im] 对，与 C99 复数布局一致；这里直接按
 * std::complex<T> 读写源缓冲。布局前提用 static_assert 钉死：一旦某个标准库
 * 实现换了表示，编译期就失败，而不是静默按错位解释数据。 */
static_assert(sizeof(std::complex<float>) == 2 * sizeof(float),
              "std::complex<float> must be an interleaved re/im pair");
static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
              "std::complex<double> must be an interleaved re/im pair");

/* 归约分支的累加精度：实数 float 升到 double（与 cblas_sgemm 内部口径一致），
 * 复数同精度升一档（complex<float> → complex<double>），其余保持原位。
 * cblas_cgemm / zgemm 自身就是按各自精度累加的，所以只有这处手工归约需要升档。 */
template <typename T>
struct Accumulator {
    using type = T;
};
template <>
struct Accumulator<float> {
    using type = double;
};
template <>
struct Accumulator<std::complex<float>> {
    using type = std::complex<double>;
};

/* ------------------------------------------------------------------ CBLAS */

using SgemmFn = void (*)(int, int, int, int, int, int, float, const void*, int, const void*, int,
                         float, void*, int);
using DgemmFn = void (*)(int, int, int, int, int, int, double, const void*, int, const void*,
                         int, double, void*, int);
/* 复数走 CBLAS 的 "c"/"z" 版本：元素是**交错实虚布局**（与 numpy 一致），
 * alpha/beta 用 void* 指向对应精度的复数常量。 */
using CgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);
using ZgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);

struct CblasApi {
    SgemmFn sgemm = nullptr;
    DgemmFn dgemm = nullptr;
    CgemmFn cgemm = nullptr;
    ZgemmFn zgemm = nullptr;

    /* 实数两档必须齐备；复数两档按需（缺了就让该 dtype 回落，不影响实数）。 */
    bool complete() const { return sgemm != nullptr && dgemm != nullptr; }
    bool complex_complete() const { return cgemm != nullptr && zgemm != nullptr; }
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

bool bind_pair(void* handle, CblasApi* api) {
    const bool real = bind_symbol(handle, "cblas_sgemm", &api->sgemm) &&
                      bind_symbol(handle, "cblas_dgemm", &api->dgemm);
    /* 复数符号是**可选**的：某些精简 BLAS 只导出实数入口。缺失时
     * complex_complete() 为 false，复数 dtype 走回落路径。 */
    bind_symbol(handle, "cblas_cgemm", &api->cgemm);
    bind_symbol(handle, "cblas_zgemm", &api->zgemm);
    return real;
}

CblasApi load_cblas() {
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

const CblasApi& cblas_api() {
    static CblasApi api;
    static std::once_flag initialized;
    std::call_once(initialized, [] { api = load_cblas(); });
    return api;
}

/* 每个 dtype 只依赖自己那一个 GEMM 符号：实数两档沿用原来的「sgemm+dgemm 齐备」
 * 口径（保持既有行为），复数档只要求本精度的 cgemm / zgemm。复数符号缺失时
 * parse_table 已经把该 dtype 判为 UNSUPPORTED，这里只是执行的二次守卫。 */
template <typename T>
struct BlasGate {
    static bool ready(const CblasApi& api) { return api.complete(); }
};
template <>
struct BlasGate<std::complex<float>> {
    static bool ready(const CblasApi& api) { return api.cgemm != nullptr; }
};
template <>
struct BlasGate<std::complex<double>> {
    static bool ready(const CblasApi& api) { return api.zgemm != nullptr; }
};

/* ------------------------------------------------------------------ 解析 */

/* 一次 gather 的循环拆分（解析期算好，执行期只跑整数加法）。
 *
 * 旧实现只从**规范序最内轴**向外合并，因此源里步长 1 的轴恰好不是规范序最内轴
 * 时，内层退化成逐元素 strided 读（每次读基本是一行新的 cache line）。这里改成
 * **按源步长从大到小**排序再合并：源永远是连续张量，步长 1 的物理轴一定在内，
 * 于是内层 run 的源步长恒为 1（顺序读），只有目的侧可能带步长。合并规则不变：
 * 相邻轴两侧步长都接得上就并进同一个 run。 */
struct CopyPlan {
    int outer_rank = 0;                    /* 外层独立迭代的轴数 */
    int order[kMaxRank] = {};              /* 外层轴的规范序下标，由内向外 */
    std::int64_t inner = 0;                /* 内层 run 的元素数 */
    std::int64_t inner_source_step = 0;
    std::int64_t inner_destination_step = 0;
    bool memcpy_run = false;               /* 两侧内层步长都是 1：整段 memcpy */
};

void build_copy_plan(const std::int64_t* source_steps, const std::int64_t* destination_steps,
                     const std::int64_t* extents, int rank, bool legacy, CopyPlan* plan);
bool copy_plan_valid(const CopyPlan& plan, int rank);
double estimate_copy_lines(const CopyPlan& plan, const std::int64_t* extents,
                           std::size_t element);
void choose_copy_plan(const std::int64_t* source_steps,
                      const std::int64_t* destination_steps, const std::int64_t* extents,
                      int rank, bool baseline, std::size_t element, CopyPlan* plan);

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
    bool pack_a = false;   /* a_axes 非恒等：需要 pack */
    bool pack_b = false;
    bool pack_c = false;   /* c_axes 非恒等：需要 unpack */
    /* 直接吃原缓冲的布局判定（见 kDirect*）；kDirectNone 时才真 pack。 */
    int a_direct = kDirectNone;
    int b_direct = kDirectNone;
    CopyPlan copy_a;
    CopyPlan copy_b;
    CopyPlan copy_c;
    /* 规范序各轴长（按本操作数自己的规范序排列）。 */
    std::int64_t a_extent[kMaxRank] = {};
    std::int64_t b_extent[kMaxRank] = {};
    std::int64_t c_extent[kMaxRank] = {};
    /* 打包：源张量按规范序的步长，与紧凑规范序的步长。 */
    std::int64_t a_source_steps[kMaxRank] = {};
    std::int64_t a_packed_steps[kMaxRank] = {};
    std::int64_t b_source_steps[kMaxRank] = {};
    std::int64_t b_packed_steps[kMaxRank] = {};
    /* 解包：目的物理轴 p 的源步长（紧凑规范序）与目的物理步长。 */
    std::int64_t c_source_steps[kMaxRank] = {};
    std::int64_t c_destination_steps[kMaxRank] = {};
    /* 目的物理轴 p 的轴长（c_extent 按 C 规范序索引，两者不同）。 */
    std::int64_t c_destination_extents[kMaxRank] = {};
};

struct Table {
    int num_operands = 0;
    int num_slots = 0;
    int out_rank = 0;
    int dtype_id = 0;
    bool baseline = false;   /* 旧口径 A/B 开关，只影响 trace 打印 */
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
int parse_table(const void* data, long long length, int dtype_id, bool baseline, Table* table) {
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
    if (magic != QCD_DEDICATED_MAGIC || version != QCD_DEDICATED_VERSION ||
        total <= 0 || static_cast<std::uint64_t>(total) != static_cast<std::uint64_t>(length)) {
        return QCD_STATUS_INVALID;
    }
    if (table_dtype != dtype_id) {
        return QCD_STATUS_INVALID;
    }
    if (table_dtype != QCD_DTYPE_FLOAT32 && table_dtype != QCD_DTYPE_FLOAT64 &&
        table_dtype != QCD_DTYPE_COMPLEX64 && table_dtype != QCD_DTYPE_COMPLEX128) {
        /* 只有这四种 IEEE 精度有数值路径：其它 dtype 明确报「不支持」。 */
        return QCD_DEDICATED_UNSUPPORTED;
    }
    if ((table_dtype == QCD_DTYPE_COMPLEX64 || table_dtype == QCD_DTYPE_COMPLEX128) &&
        !cblas_api().complex_complete()) {
        /* 复数走 cgemm / zgemm。运行期 BLAS 只导出实数入口时，复数档**不硬错**、
         * 也不隐式降精度，而是报 UNSUPPORTED 让上层回落到通用路径。 */
        return QCD_DEDICATED_UNSUPPORTED;
    }
    if (num_operands < 1 || num_operands > kMaxOperands || num_slots < 0 ||
        num_slots > kMaxSlots || num_steps < 1 || num_steps > kMaxSteps || out_rank < 0 ||
        out_rank > kMaxRank) {
        return QCD_DEDICATED_UNSUPPORTED;
    }

    table->num_operands = num_operands;
    table->num_slots = num_slots;
    table->out_rank = out_rank;
    table->dtype_id = table_dtype;
    table->baseline = baseline;
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
            return QCD_DEDICATED_UNSUPPORTED;
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

        /* 两侧规范序的轴长与源步长；batch/K 两组必须在 A、B 之间逐项相等。
         *
         * 只有「恒等」和「组交换」两种轴序可以直接交给 BLAS（后者用 transA /
         * transB 吸收），其余轴序必须 pack。判据只看 a_axes/b_axes 与两种期望
         * 排列是否逐位相等。 */
        step.a_direct = kDirectPlain;
        step.b_direct = kDirectPlain;
        for (int axis = 0; axis < rank_a; ++axis) {
            step.a_extent[axis] = left.extents[a_axes[axis]];
            step.a_source_steps[axis] = left_strides[a_axes[axis]];
            if (a_axes[axis] != axis) {
                step.a_direct = kDirectNone;
            }
        }
        if (step.a_direct == kDirectNone) {
            step.a_direct = kDirectTrans;
            for (int axis = 0; axis < rank_a; ++axis) {
                const int transposed =
                    axis < batch_rank
                        ? axis
                        : (axis < batch_rank + m_rank ? axis + k_rank : axis - m_rank);
                if (a_axes[axis] != transposed) {
                    step.a_direct = kDirectNone;
                    break;
                }
            }
        }
        for (int axis = 0; axis < rank_b; ++axis) {
            step.b_extent[axis] = right.extents[b_axes[axis]];
            step.b_source_steps[axis] = right_strides[b_axes[axis]];
            if (b_axes[axis] != axis) {
                step.b_direct = kDirectNone;
            }
        }
        if (step.b_direct == kDirectNone) {
            step.b_direct = kDirectTrans;
            for (int axis = 0; axis < rank_b; ++axis) {
                const int transposed =
                    axis < batch_rank
                        ? axis
                        : (axis < batch_rank + k_rank ? axis + n_rank : axis - k_rank);
                if (b_axes[axis] != transposed) {
                    step.b_direct = kDirectNone;
                    break;
                }
            }
        }
        step.pack_a = step.a_direct == kDirectNone;
        step.pack_b = step.b_direct == kDirectNone;
        if (baseline) {
            /* 旧口径：非恒等轴序一律 pack，不用 BLAS 的转置参数。 */
            if (step.a_direct != kDirectPlain) {
                step.a_direct = kDirectNone;
                step.pack_a = true;
            }
            if (step.b_direct != kDirectPlain) {
                step.b_direct = kDirectNone;
                step.pack_b = true;
            }
        }
        if (!suffix_steps(step.a_extent, rank_a, step.a_packed_steps) ||
            !suffix_steps(step.b_extent, rank_b, step.b_packed_steps)) {
            return QCD_STATUS_INVALID;
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
        const Shape& destination = dst_slot >= 0 ? table->slots[static_cast<std::size_t>(dst_slot)]
                                                 : table->output;
        if (destination.rank != rank_c) {
            return QCD_STATUS_INVALID;
        }
        std::int64_t destination_strides[kMaxRank] = {};
        if (!shape_strides(destination, destination_strides)) {
            return QCD_STATUS_INVALID;
        }
        step.pack_c = false;
        for (int axis = 0; axis < rank_c; ++axis) {
            const int position = c_axes[axis];
            if (step.c_extent[position] != destination.extents[axis]) {
                return QCD_STATUS_INVALID;
            }
            step.c_source_steps[axis] = canonical_steps[position];
            step.c_destination_steps[axis] = destination_strides[axis];
            step.c_destination_extents[axis] = destination.extents[axis];
            if (position != axis) {
                step.pack_c = true;
            }
        }
        /* gather 的循环拆分在解析期一次算好：执行期只按 order[] 走里程计。
         * 本步不需要拷贝的一侧不建 plan（数据不进任何缓冲）。 */
        const std::size_t element = element_size(table_dtype);
        if (step.pack_a) {
            choose_copy_plan(step.a_source_steps, step.a_packed_steps, step.a_extent, rank_a,
                             baseline, element, &step.copy_a);
            if (!copy_plan_valid(step.copy_a, rank_a)) {
                return QCD_STATUS_INVALID;
            }
        }
        if (step.pack_b) {
            choose_copy_plan(step.b_source_steps, step.b_packed_steps, step.b_extent, rank_b,
                             baseline, element, &step.copy_b);
            if (!copy_plan_valid(step.copy_b, rank_b)) {
                return QCD_STATUS_INVALID;
            }
        }
        if (step.pack_c) {
            choose_copy_plan(step.c_source_steps, step.c_destination_steps,
                             step.c_destination_extents, rank_c, baseline, element,
                             &step.copy_c);
            if (!copy_plan_valid(step.copy_c, rank_c)) {
                return QCD_STATUS_INVALID;
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

/* ------------------------------------------------------------------ 拷贝 */

/* 把 (源步长表, 目的步长表, 轴长表) 编译成 CopyPlan。
 *
 * 步骤：把轴按**源步长从大到小**排序（源是连续张量 → 一定存在步长 1 的轴，
 * 它被排到最内层，内层读因此恒为顺序），再从内向外把两侧步长都接得上的相邻轴
 * 合并进同一个 run。合并后的内层 run 用「源步长 / 目的步长」两个常数描述：
 *   * 两侧都是 1 → 整段 memcpy；
 *   * 只有源是 1 → 顺序读、按目的步长写（目的缓冲是我们自己的，通常留在
 *     L2/L3，比按大源步长逐元素读便宜得多）。
 * 其余轴按从内到外的次序进 order[]，执行期用里程计推进偏移。 */
void build_copy_plan(const std::int64_t* source_steps, const std::int64_t* destination_steps,
                     const std::int64_t* extents, int rank, bool legacy, CopyPlan* plan) {
    *plan = CopyPlan{};
    if (rank <= 0) {
        plan->inner = 1;
        plan->memcpy_run = true;
        return;
    }
    int sorted[kMaxRank] = {};
    int count = 0;
    for (int axis = 0; axis < rank; ++axis) {
        if (extents[axis] == 0) {
            return;                            /* 空张量：不需要搬 */
        }
        int position = count;
        if (legacy) {
            /* 旧口径：按规范序从最内轴向外，不考虑源侧步长。 */
            sorted[count] = axis;
        } else {
            while (position > 0 &&
                   source_steps[sorted[position - 1]] < source_steps[axis]) {
                sorted[position] = sorted[position - 1];
                --position;
            }
            sorted[position] = axis;
        }
        ++count;
    }

    const int innermost = sorted[count - 1];
    plan->inner = extents[innermost];
    plan->inner_source_step = source_steps[innermost];
    plan->inner_destination_step = destination_steps[innermost];
    /* 逐轴向外合并：合并后的 run 在源侧必须仍然连续（步长 1），否则 memcpy /
     * 单步长内层循环的前提不成立。旧口径额外要求目的侧也连续（两侧步长都等于
     * 已累计的元素数），与新口径的「目的侧固定步长」不同。 */
    const bool mergeable = plan->inner_source_step == 1;
    /* 旧口径只有在内层两侧都连续时才继续合并（与原实现的 run 计数等价）。 */
    const bool legacy_mergeable =
        plan->inner_source_step == 1 && plan->inner_destination_step == 1;
    for (int index = count - 2; index >= 0; --index) {
        const int axis = sorted[index];
        const bool merges =
            legacy
                ? (legacy_mergeable && source_steps[axis] == plan->inner &&
                   destination_steps[axis] == plan->inner)
                : (mergeable && source_steps[axis] == plan->inner &&
                   destination_steps[axis] == plan->inner_destination_step * plan->inner);
        if (merges) {
            plan->inner *= extents[axis];
            continue;
        }
        plan->order[plan->outer_rank++] = axis;
    }
    plan->memcpy_run = plan->inner_source_step == 1 && plan->inner_destination_step == 1;
}

bool copy_plan_valid(const CopyPlan& plan, int rank) {
    if (plan.inner < 0 || plan.outer_rank < 0 || plan.outer_rank > rank) {
        return false;
    }
    bool seen[kMaxRank] = {};
    for (int index = 0; index < plan.outer_rank; ++index) {
        const int axis = plan.order[index];
        if (axis < 0 || axis >= rank || seen[axis]) {
            return false;
        }
        seen[axis] = true;
    }
    return true;
}

/* 估算一次 gather 碰到的 cache line 数，用来在「规范序合并」与「源步长合并」
 * 两种循环次序之间择优（两者搬的元素数完全相同，差异只在局部性）。
 *
 * 内层 run 的每个外层迭代读/写各是一段固定步长的访问：
 *   * 步长 1（或步长*元素 < 64 字节）→ 一个 run 只占
 *     ceil((inner-1)*step*element + element) / 64 行；
 *   * 大跳步 → 每个元素一行。
 * 两侧取较大者作为该次迭代的行数，再乘外层迭代数。 */
double estimate_copy_lines(const CopyPlan& plan, const std::int64_t* extents,
                           std::size_t element) {
    if (plan.inner <= 0) {
        return 0.0;
    }
    double outer = 1.0;
    for (int index = 0; index < plan.outer_rank; ++index) {
        outer *= static_cast<double>(extents[plan.order[index]]);
    }
    const double line = 64.0;
    const auto lines_for = [&](std::int64_t step) {
        const double bytes = static_cast<double>((plan.inner - 1) * step + 1) *
                             static_cast<double>(element);
        return bytes <= line ? 1.0 : bytes / line;
    };
    const double reads = lines_for(plan.inner_source_step);
    const double writes = lines_for(plan.inner_destination_step);
    return outer * (reads > writes ? reads : writes);
}

/* 两种循环次序都建一遍，取估算 cache line 更少的那种；估算相等时用规范序
 * （旧口径），因为它在实测里对大张量更稳。baseline 模式只用规范序。 */
void choose_copy_plan(const std::int64_t* source_steps,
                      const std::int64_t* destination_steps, const std::int64_t* extents,
                      int rank, bool baseline, std::size_t element, CopyPlan* plan) {
    CopyPlan canonical;
    build_copy_plan(source_steps, destination_steps, extents, rank, true, &canonical);
    if (baseline || !copy_plan_valid(canonical, rank)) {
        *plan = canonical;
        return;
    }
    CopyPlan by_source;
    build_copy_plan(source_steps, destination_steps, extents, rank, false, &by_source);
    if (!copy_plan_valid(by_source, rank) ||
        estimate_copy_lines(canonical, extents, element) <=
            estimate_copy_lines(by_source, extents, element)) {
        *plan = canonical;
        return;
    }
    *plan = by_source;
}

/* 按 CopyPlan 执行 gather：外层里程计 + 内层 run。 */
template <typename T>
void run_copy_plan(const T* source, T* destination, const CopyPlan& plan,
                   const std::int64_t* source_steps, const std::int64_t* destination_steps,
                   const std::int64_t* extents) {
    if (plan.inner == 0) {
        return;
    }
    if (plan.outer_rank == 0) {
        if (plan.memcpy_run) {
            std::memcpy(destination, source,
                        sizeof(T) * static_cast<std::size_t>(plan.inner));
            return;
        }
        if (plan.inner_source_step == 1) {
            for (std::int64_t lane = 0; lane < plan.inner; ++lane) {
                destination[lane * plan.inner_destination_step] = source[lane];
            }
        } else {
            for (std::int64_t lane = 0; lane < plan.inner; ++lane) {
                destination[lane * plan.inner_destination_step] =
                    source[lane * plan.inner_source_step];
            }
        }
        return;
    }
    std::int64_t coordinates[kMaxRank] = {};
    std::int64_t source_offset = 0;
    std::int64_t destination_offset = 0;
    std::int64_t outer = 1;
    for (int index = 0; index < plan.outer_rank; ++index) {
        outer *= extents[plan.order[index]];
    }
    for (std::int64_t index = 0; index < outer; ++index) {
        const T* from = source + source_offset;
        T* to = destination + destination_offset;
        if (plan.memcpy_run) {
            std::memcpy(to, from, sizeof(T) * static_cast<std::size_t>(plan.inner));
        } else if (plan.inner_source_step == 1) {
            for (std::int64_t lane = 0; lane < plan.inner; ++lane) {
                to[lane * plan.inner_destination_step] = from[lane];
            }
        } else {
            for (std::int64_t lane = 0; lane < plan.inner; ++lane) {
                to[lane * plan.inner_destination_step] = from[lane * plan.inner_source_step];
            }
        }
        if (index + 1 == outer) {
            break;
        }
        for (int axis = 0; axis < plan.outer_rank; ++axis) {
            const int position = plan.order[axis];
            source_offset += source_steps[position];
            destination_offset += destination_steps[position];
            if (++coordinates[axis] < extents[position]) {
                break;
            }
            coordinates[axis] = 0;
            source_offset -= source_steps[position] * extents[position];
            destination_offset -= destination_steps[position] * extents[position];
        }
    }
}

/* ------------------------------------------------------------------ GEMM */

/* lda / ldb 由布局决定：NoTrans 时矩阵按 [M][K]（A）或 [K][N]（B）连续存放，
 * 主维分别是 k / n；转置时按 [K][M]（A）或 [N][K]（B）存放，主维是 m / k。 */
void gemm_call(const CblasApi& api, int trans_a, int trans_b, int rows, int cols, int inner,
               const float* a, const float* b, float* c) {
    const int lda = trans_a != 0 ? rows : inner;
    const int ldb = trans_b != 0 ? inner : cols;
    api.sgemm(kCblasRowMajor, trans_a != 0 ? kCblasTrans : kCblasNoTrans,
              trans_b != 0 ? kCblasTrans : kCblasNoTrans, rows, cols, inner, 1.0F, a, lda, b,
              ldb, 0.0F, c, cols);
}

void gemm_call(const CblasApi& api, int trans_a, int trans_b, int rows, int cols, int inner,
               const double* a, const double* b, double* c) {
    const int lda = trans_a != 0 ? rows : inner;
    const int ldb = trans_b != 0 ? inner : cols;
    api.dgemm(kCblasRowMajor, trans_a != 0 ? kCblasTrans : kCblasNoTrans,
              trans_b != 0 ? kCblasTrans : kCblasNoTrans, rows, cols, inner, 1.0, a, lda, b,
              ldb, 0.0, c, cols);
}

/* 复数走 CBLAS 的 cgemm / zgemm：元素是交错实虚布局（与 numpy 一致），
 * alpha/beta 是指向**本精度复常量**的指针，其余参数与实数版逐项相同。 */
void gemm_call(const CblasApi& api, int trans_a, int trans_b, int rows, int cols, int inner,
               const std::complex<float>* a, const std::complex<float>* b,
               std::complex<float>* c) {
    const int lda = trans_a != 0 ? rows : inner;
    const int ldb = trans_b != 0 ? inner : cols;
    const std::complex<float> alpha(1.0F, 0.0F);
    const std::complex<float> beta(0.0F, 0.0F);
    api.cgemm(kCblasRowMajor, trans_a != 0 ? kCblasTrans : kCblasNoTrans,
              trans_b != 0 ? kCblasTrans : kCblasNoTrans, rows, cols, inner, &alpha, a, lda, b,
              ldb, &beta, c, cols);
}

void gemm_call(const CblasApi& api, int trans_a, int trans_b, int rows, int cols, int inner,
               const std::complex<double>* a, const std::complex<double>* b,
               std::complex<double>* c) {
    const int lda = trans_a != 0 ? rows : inner;
    const int ldb = trans_b != 0 ? inner : cols;
    const std::complex<double> alpha(1.0, 0.0);
    const std::complex<double> beta(0.0, 0.0);
    api.zgemm(kCblasRowMajor, trans_a != 0 ? kCblasTrans : kCblasNoTrans,
              trans_b != 0 ? kCblasTrans : kCblasNoTrans, rows, cols, inner, &alpha, a, lda, b,
              ldb, &beta, c, cols);
}

/* 一组 batch 的 ``C[b] = A[b] * B[b]``（A 为 m×k、B 为 k×n、C 为 m×n，行主序）。
 *
 * M/N/K 任一为 1 时这一步本质是纯缩放/广播归约：cblas 的分块内核在这种形状上
 * 只是调用开销（K=1 时是一个 rank-1 更新，M=1/N=1 时是 gemv），因此直接按定义
 * 求和，不进入 BLAS。这些短路径只接受两侧都已按 [M][K]/[K][N] 连续存放
 * （trans 为 0）；需要转置时直接走 BLAS。 */
template <typename T>
void gemm_block(const CblasApi& api, int trans_a, int trans_b, std::int64_t batch,
                std::int64_t m, std::int64_t n, std::int64_t k, const T* a, const T* b, T* c) {
    if (trans_a == 0 && trans_b == 0) {
        if (k == 1) {
            for (std::int64_t group = 0; group < batch; ++group) {
                const T* left = a + group * m;
                const T* right = b + group * n;
                T* out = c + group * m * n;
                for (std::int64_t row = 0; row < m; ++row) {
                    const T scale = left[row];
                    T* target = out + row * n;
                    for (std::int64_t column = 0; column < n; ++column) {
                        target[column] = scale * right[column];
                    }
                }
            }
            return;
        }
        if (n == 1) {
            /* 归约轴用**双精度累加器**：cblas_sgemm 内部按块做 float 乘 + 更高精度求和，
             * 而这里若直接 `float total += ...`，长链（k 大）上会明显劣于 BLAS。
             * 实测：某 16 算子方程用 float 累加时 float32 相对偏差 2.97e-04，
             * 而 BLAS 路径只有 3.95e-05 —— 差 7.5 倍，全来自这一处。
             * 双精度累加后再舍入回 T，与 BLAS 的口径一致，且不改变存储精度。 */
            using Accum = typename Accumulator<T>::type;
            for (std::int64_t group = 0; group < batch; ++group) {
                const T* left = a + group * m * k;
                const T* right = b + group * k;
                T* out = c + group * m;
                for (std::int64_t row = 0; row < m; ++row) {
                    Accum total = Accum(0);
                    for (std::int64_t inner = 0; inner < k; ++inner) {
                        total += Accum(left[row * k + inner]) * Accum(right[inner]);
                    }
                    out[row] = static_cast<T>(total);
                }
            }
            return;
        }
        if (m == 1) {
            /* 与 n==1 同理：归约轴用双精度累加，避免 float 长链求和劣于 BLAS。 */
            using Accum = typename Accumulator<T>::type;
            std::vector<Accum> accumulator(static_cast<std::size_t>(n));
            for (std::int64_t group = 0; group < batch; ++group) {
                const T* left = a + group * k;
                const T* right = b + group * k * n;
                T* out = c + group * n;
                std::fill(accumulator.begin(), accumulator.end(), Accum(0));
                for (std::int64_t inner = 0; inner < k; ++inner) {
                    const Accum scale = Accum(left[inner]);
                    const T* row = right + inner * n;
                    for (std::int64_t column = 0; column < n; ++column) {
                        accumulator[static_cast<std::size_t>(column)] +=
                            scale * Accum(row[column]);
                    }
                }
                for (std::int64_t column = 0; column < n; ++column) {
                    out[column] = static_cast<T>(accumulator[static_cast<std::size_t>(column)]);
                }
            }
            return;
        }
    }
    for (std::int64_t group = 0; group < batch; ++group) {
        gemm_call(api, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
                  static_cast<int>(k), a + group * m * k, b + group * k * n,
                  c + group * m * n);
    }
}

/* ------------------------------------------------------------------ 执行 */

struct Slot {
    const std::uint8_t* external = nullptr;
    Buffer storage;
    bool owned = false;
    bool live = false;

    const std::uint8_t* data() const { return owned ? storage.data() : external; }
};

template <typename T>
int run_typed(const Table& table, void* const* operands, void* out) {
    const CblasApi& api = cblas_api();
    if (!BlasGate<T>::ready(api)) {
        return QCD_STATUS_INVALID;
    }
    const bool trace = trace_enabled();
    const TraceClock::time_point allocation_begin =
        trace ? TraceClock::now() : TraceClock::time_point{};
    const std::size_t element = sizeof(T);
    const int total_slots = table.num_operands + table.num_slots;

    std::vector<Slot> slots(static_cast<std::size_t>(total_slots));
    for (int index = 0; index < table.num_operands; ++index) {
        const Shape& shape = table.operands[static_cast<std::size_t>(index)];
        std::int64_t count = 0;
        if (!product_shape(shape, &count)) {
            return QCD_STATUS_INVALID;
        }
        slots[static_cast<std::size_t>(index)].external =
            static_cast<const std::uint8_t*>(operands[index]);
        slots[static_cast<std::size_t>(index)].live = true;
        if (count != 0 && slots[static_cast<std::size_t>(index)].external == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    /* 中间槽一次分配到位：执行期不再申请内存。 */
    for (int index = 0; index < table.num_slots; ++index) {
        const Shape& shape = table.slots[static_cast<std::size_t>(index)];
        std::int64_t count = 0;
        std::size_t bytes = 0;
        if (!product_shape(shape, &count) || !checked_bytes(count, element, &bytes)) {
            return QCD_STATUS_INVALID;
        }
        Slot& slot = slots[static_cast<std::size_t>(table.num_operands + index)];
        if (!slot.storage.allocate(bytes)) {
            return QCD_STATUS_MEMORY;
        }
        slot.owned = true;
    }

    /* pack 缓冲按全表最大尺寸一次分配。 */
    std::int64_t max_a = 0;
    std::int64_t max_b = 0;
    std::int64_t max_c = 0;
    for (const Step& step : table.steps) {
        std::int64_t elements_a = 0;
        std::int64_t elements_b = 0;
        std::int64_t elements_c = 0;
        if (!checked_mul(step.batch, step.m, &elements_a) ||
            !checked_mul(elements_a, step.k, &elements_a) ||
            !checked_mul(step.batch, step.k, &elements_b) ||
            !checked_mul(elements_b, step.n, &elements_b) ||
            !checked_mul(step.batch, step.m, &elements_c) ||
            !checked_mul(elements_c, step.n, &elements_c)) {
            return QCD_STATUS_INVALID;
        }
        max_a = std::max(max_a, step.pack_a ? elements_a : 0);
        max_b = std::max(max_b, step.pack_b ? elements_b : 0);
        max_c = std::max(max_c, step.pack_c ? elements_c : 0);
    }
    Buffer packed_a;
    Buffer packed_b;
    Buffer packed_c;
    std::size_t packed_a_bytes = 0;
    std::size_t packed_b_bytes = 0;
    std::size_t packed_c_bytes = 0;
    if (!checked_bytes(max_a, element, &packed_a_bytes) ||
        !checked_bytes(max_b, element, &packed_b_bytes) ||
        !checked_bytes(max_c, element, &packed_c_bytes)) {
        return QCD_STATUS_INVALID;
    }
    if ((packed_a_bytes != 0 && !packed_a.allocate(packed_a_bytes)) ||
        (packed_b_bytes != 0 && !packed_b.allocate(packed_b_bytes)) ||
        (packed_c_bytes != 0 && !packed_c.allocate(packed_c_bytes))) {
        return QCD_STATUS_MEMORY;
    }

    std::uint8_t* const final_output = static_cast<std::uint8_t*>(out);
    if (trace) {
        std::fprintf(stderr,
                     "[dedicated] alloc %.3f ms | slots=%d steps=%d element=%zu mode=%s "
                     "bytes: A=%zu B=%zu C=%zu\n",
                     trace_ms(allocation_begin, TraceClock::now()), table.num_slots,
                     static_cast<int>(table.steps.size()), element,
                     table.baseline ? "baseline" : "direct", packed_a_bytes, packed_b_bytes,
                     packed_c_bytes);
        for (int slot_index = 0; slot_index < table.num_slots; ++slot_index) {
            std::fprintf(stderr, "[dedicated] slot %d bytes=%zu\n", slot_index,
                         slots[static_cast<std::size_t>(table.num_operands + slot_index)]
                             .storage.bytes());
        }
    }
    bool wrote_output = false;
    long long trace_pack_elements = 0;
    double trace_pack_ms = 0.0;
    double trace_gemm_ms = 0.0;
    double trace_unpack_ms = 0.0;
    for (std::size_t index = 0; index < table.steps.size(); ++index) {
        const Step& step = table.steps[index];
        Slot& left_slot = slots[static_cast<std::size_t>(step.a_slot)];
        Slot& right_slot = slots[static_cast<std::size_t>(step.b_slot)];
        if (!left_slot.live || !right_slot.live) {
            return QCD_STATUS_INVALID;
        }

        std::uint8_t* destination = final_output;
        Slot* destination_slot = nullptr;
        if (step.dst_slot >= 0) {
            destination_slot =
                &slots[static_cast<std::size_t>(table.num_operands + step.dst_slot)];
            if (destination_slot->live) {
                return QCD_STATUS_INVALID;
            }
            destination = destination_slot->storage.data();
        } else {
            wrote_output = true;
        }

        const std::int64_t batch = step.batch;
        const std::int64_t m = step.m;
        const std::int64_t n = step.n;
        const std::int64_t k = step.k;
        if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
            k > std::numeric_limits<int>::max()) {
            return QCD_STATUS_INVALID;
        }

        const T* left = reinterpret_cast<const T*>(left_slot.data());
        const T* right = reinterpret_cast<const T*>(right_slot.data());
        if ((m != 0 && n != 0 && k != 0) && (left == nullptr || right == nullptr)) {
            return QCD_STATUS_INVALID;
        }

        const TraceClock::time_point step_begin =
            trace ? TraceClock::now() : TraceClock::time_point{};
        const T* packed_left = left;
        if (step.pack_a) {
            T* buffer = reinterpret_cast<T*>(packed_a.data());
            run_copy_plan(left, buffer, step.copy_a, step.a_source_steps,
                          step.a_packed_steps, step.a_extent);
            packed_left = buffer;
        }
        const T* packed_right = right;
        if (step.pack_b) {
            T* buffer = reinterpret_cast<T*>(packed_b.data());
            run_copy_plan(right, buffer, step.copy_b, step.b_source_steps,
                          step.b_packed_steps, step.b_extent);
            packed_right = buffer;
        }
        const TraceClock::time_point packed =
            trace ? TraceClock::now() : TraceClock::time_point{};

        T* result = step.pack_c ? reinterpret_cast<T*>(packed_c.data())
                                : reinterpret_cast<T*>(destination);
        if (m != 0 && n != 0) {
            if (k == 0) {
                /* 走 void* 是为了让 -Wclass-memaccess 不再对 std::complex 报「清空
                 * 非平凡类型」：复数与实数一样是交错标量对，全零字节就是数值 0。 */
                std::memset(static_cast<void*>(result), 0,
                            static_cast<std::size_t>(batch * m * n) * sizeof(T));
            } else {
                /* 两侧 direct 布局各自决定 trans 位：kDirectPlain → 0，
                 * kDirectTrans → 1，kDirectNone 说明该侧已 pack 成 NoTrans。 */
                const int trans_a = step.a_direct == kDirectTrans ? 1 : 0;
                const int trans_b = step.b_direct == kDirectTrans ? 1 : 0;
                gemm_block(api, trans_a, trans_b, batch, m, n, k, packed_left, packed_right,
                           result);
            }
        }
        const TraceClock::time_point multiplied =
            trace ? TraceClock::now() : TraceClock::time_point{};
        if (step.pack_c) {
            run_copy_plan(result, reinterpret_cast<T*>(destination), step.copy_c,
                          step.c_source_steps, step.c_destination_steps,
                          step.c_destination_extents);
        }
        if (trace) {
            const TraceClock::time_point finished = TraceClock::now();
            const long long pack_elements =
                (step.pack_a ? static_cast<long long>(batch * m * k) : 0) +
                (step.pack_b ? static_cast<long long>(batch * k * n) : 0) +
                (step.pack_c ? static_cast<long long>(batch * m * n) : 0);
            std::fprintf(stderr,
                         "[dedicated] step %zu slots=%d,%d->%d bmnk=%lld,%lld,%lld,%lld "
                         "pack=%d%d%d trans=%d%d elements=%lld | pack %.3f ms gemm %.3f ms "
                         "unpack %.3f ms total %.3f ms\n",
                         index, step.a_slot, step.b_slot, step.dst_slot,
                         static_cast<long long>(batch), static_cast<long long>(m),
                         static_cast<long long>(n), static_cast<long long>(k),
                         step.pack_a ? 1 : 0, step.pack_b ? 1 : 0, step.pack_c ? 1 : 0,
                         step.a_direct == kDirectTrans ? 1 : 0,
                         step.b_direct == kDirectTrans ? 1 : 0, pack_elements,
                         trace_ms(step_begin, packed), trace_ms(packed, multiplied),
                         trace_ms(multiplied, finished), trace_ms(step_begin, finished));
            trace_pack_elements += pack_elements;
            trace_pack_ms += trace_ms(step_begin, packed);
            trace_gemm_ms += trace_ms(packed, multiplied);
            trace_unpack_ms += trace_ms(multiplied, finished);
        }

        /* 中间体在收缩树里只被消费一次：读完即标记死亡。 */
        if (step.a_slot >= table.num_operands) {
            left_slot.live = false;
        }
        if (step.b_slot >= table.num_operands) {
            right_slot.live = false;
        }
        if (destination_slot != nullptr) {
            destination_slot->live = true;
        }
    }
    if (trace) {
        std::fprintf(stderr,
                     "[dedicated] summary steps=%zu pack_elements=%lld | pack %.3f ms "
                     "gemm %.3f ms unpack %.3f ms\n",
                     table.steps.size(), trace_pack_elements, trace_pack_ms, trace_gemm_ms,
                     trace_unpack_ms);
    }
    return wrote_output ? QCD_STATUS_OK : QCD_STATUS_INVALID;
}

int apply_dedicated(const void* steps, long long steps_len, void* const* operands,
                    int n_operands, void* out, int dtype_id) {
    if (operands == nullptr || out == nullptr) {
        return QCD_STATUS_INVALID;
    }
    Table table;
    const bool trace = trace_enabled();
    const bool baseline = baseline_enabled();
    const TraceClock::time_point parse_begin =
        trace ? TraceClock::now() : TraceClock::time_point{};
    const int parsed = parse_table(steps, steps_len, dtype_id, baseline, &table);
    if (parsed != QCD_STATUS_OK) {
        return parsed;
    }
    if (trace) {
        std::fprintf(stderr, "[dedicated] parse %.3f ms\n",
                     trace_ms(parse_begin, TraceClock::now()));
    }
    if (n_operands != table.num_operands) {
        return QCD_STATUS_INVALID;
    }
    switch (dtype_id) {
        case QCD_DTYPE_FLOAT32:
            return run_typed<float>(table, operands, out);
        case QCD_DTYPE_FLOAT64:
            return run_typed<double>(table, operands, out);
        case QCD_DTYPE_COMPLEX64:
            return run_typed<std::complex<float>>(table, operands, out);
        case QCD_DTYPE_COMPLEX128:
            return run_typed<std::complex<double>>(table, operands, out);
        default:
            return QCD_DEDICATED_UNSUPPORTED;
    }
}

}  // namespace

extern "C" QCD_EINSUM_API int qcd_kupl_dedicated_apply(const void* steps, long long steps_len,
                                                       void* const* operands, int n_operands,
                                                       void* out, int dtype_id) {
    try {
        return apply_dedicated(steps, steps_len, operands, n_operands, out, dtype_id);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::length_error&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}
