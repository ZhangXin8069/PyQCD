#ifndef PYQCD_KUPL_EINSUM_FAMILY_TEMPLATES_H
#define PYQCD_KUPL_EINSUM_FAMILY_TEMPLATES_H

#include "family_apply.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <vector>

/*
 * 25 族「可套用模板」执行器。
 *
 * 模板参数（编译期常量）::
 *
 *     template <typename T, int ARITY, int OUT_RANK, int... RANKS>
 *     int run_family(const Plan& plan, void* const* operands, void* out);
 *
 *     ARITY      算子元数（6/7/9/10/12/13/14/16）
 *     OUT_RANK   输出 rank（0..6）
 *     RANKS...   各操作数 rank 的**升序多重集**（2/3/4 的组合）
 *
 * 这三个量正是 docs/wick_kernel_families_20261111.md 的分组键，25 族因此是 25 份
 * 实例化（见 src/family_apply.cpp 的 kFamilies 表）。编译期常量直接决定：
 *
 *   * 步数 kSteps = ARITY - 1、中间槽 kSlots = ARITY - 2（二元收缩树），
 *     执行循环与槽位表都是定长数组，编译器可展开；
 *   * 每个 Step 的轴表上界 kMaxRankHere = max(RANKS..., OUT_RANK)，
 *     所有 per-step 坐标/步长数组都不需要堆分配；
 *   * 计划是否合法：元数、输出 rank、rank 多重集必须逐项相等，否则本实例化
 *     拒绝执行（返回 1），不会"看起来能跑"地算错。
 *
 * 运行期数据只有步表：每步的两个槽、保留的指标、各轴长度，以及源步长/紧凑步长/
 * 解包步长（全部由 src/family_apply.cpp 的解析器一次算好）。执行期只做
 * pack → cblas_{s,d}gemm → unpack。
 *
 * 精度：float / double 各一条路径（模板 T），由 dtype_id 显式选择；complex
 * 在解析阶段就返回 4，绝不隐式降精度。
 */

namespace pyqcd_einsum {
namespace family {

enum : int { kMaxRank = QCD_FAMILY_MAX_RANK, kMaxArity = QCD_FAMILY_MAX_ARITY };

/* ------------------------------------------------------------------ 数据 */

struct Shape {
    int rank = 0;
    std::int64_t extents[kMaxRank] = {};
};

/* 一步的全部执行期常量：槽位、轴序、以及预先算好的步长（单位：元素）。
 * 语义见 include/family_apply.h 的“步记录”一节。 */
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
    bool pack_a = false; /* a_axes 非恒等：需要 pack */
    bool pack_b = false;
    bool pack_c = false; /* c_axes 非恒等：需要 unpack */
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
};

/* 一份已解析的计划。所有数组都是定长：元数 ≤ 16、步数 ≤ 15、中间槽 ≤ 14。 */
struct Plan {
    int num_operands = 0;
    int num_slots = 0;
    int num_steps = 0;
    int out_rank = 0;
    int dtype_id = 0;
    Shape operands[kMaxArity];
    Shape slots[kMaxArity];
    Shape output;
    Step steps[kMaxArity];
};

/* ------------------------------------------------------------------ 块池 */

/* 与 src/apply_contract.cpp 的 BufferPool 同策略：最佳适配、只在块大小落在
 * [bytes, 2*bytes] 时复用，释放小于 512 KiB 的块直接归还系统。该池在
 * apply_contract.cpp 的匿名命名空间里，无法跨 TU 复用，这里按同一口径自带最小
 * 实现（与 topology_gemm.h 同一取舍）。 */
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

inline BlockPool& block_pool() {
    static BlockPool pool;
    return pool;
}

/* 块池的 RAII 句柄：析构即归还，异常路径也不会漏。 */
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

  private:
    void release() noexcept {
        block_pool().release(data_, bytes_);
        data_ = nullptr;
        bytes_ = 0;
    }

    std::uint8_t* data_ = nullptr;
    std::size_t bytes_ = 0;
};

/* ------------------------------------------------------------------ CBLAS */

using SgemmFn = void (*)(int, int, int, int, int, int, float, const void*, int, const void*, int,
                         float, void*, int);
using DgemmFn = void (*)(int, int, int, int, int, int, double, const void*, int, const void*,
                         int, double, void*, int);

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
    static_assert(sizeof(Function) == sizeof(symbol), "dlsym function-pointer size mismatch");
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

enum {
    /* CBLAS 枚举值，与 <cblas.h> 一致：make.sh 允许构建期找不到 OpenBLAS 头文件
     * （运行期 dlopen），因此这里不 include <cblas.h>。 */
    kCblasRowMajor = 101,
    kCblasNoTrans = 111
};

/* ------------------------------------------------------------------ 拷贝 */

/* 同一逻辑坐标下的 strided copy：source/destination 各自按 ``steps``（元素）
 * 沿逻辑轴前进，``extents`` 是逻辑各轴长度。
 *
 * rank 0..4 各自展开：语料的 pack/unpack 绝大多数是 rank ≤ 4 的小转置，
 * 通用坐标回溯在 dim=2/3 时反而是主要成本，展开后内层是连续循环。 */
template <typename T>
inline void copy_axes(const T* source, const std::int64_t* source_steps, T* destination,
                      const std::int64_t* destination_steps, const std::int64_t* extents,
                      int rank) {
    switch (rank) {
        case 0:
            *destination = *source;
            return;
        case 1: {
            const std::int64_t n0 = extents[0];
            const std::int64_t s0 = source_steps[0];
            const std::int64_t d0 = destination_steps[0];
            if (s0 == 1 && d0 == 1) {
                std::memcpy(destination, source, sizeof(T) * static_cast<std::size_t>(n0));
                return;
            }
            for (std::int64_t i0 = 0; i0 < n0; ++i0) {
                destination[i0 * d0] = source[i0 * s0];
            }
            return;
        }
        case 2: {
            const std::int64_t n0 = extents[0];
            const std::int64_t n1 = extents[1];
            for (std::int64_t i0 = 0; i0 < n0; ++i0) {
                const T* from = source + i0 * source_steps[0];
                T* to = destination + i0 * destination_steps[0];
                const std::int64_t s1 = source_steps[1];
                const std::int64_t d1 = destination_steps[1];
                if (s1 == 1 && d1 == 1) {
                    std::memcpy(to, from, sizeof(T) * static_cast<std::size_t>(n1));
                    continue;
                }
                for (std::int64_t i1 = 0; i1 < n1; ++i1) {
                    to[i1 * d1] = from[i1 * s1];
                }
            }
            return;
        }
        case 3: {
            const std::int64_t n0 = extents[0];
            const std::int64_t n1 = extents[1];
            const std::int64_t n2 = extents[2];
            for (std::int64_t i0 = 0; i0 < n0; ++i0) {
                for (std::int64_t i1 = 0; i1 < n1; ++i1) {
                    const T* from = source + i0 * source_steps[0] + i1 * source_steps[1];
                    T* to = destination + i0 * destination_steps[0] +
                            i1 * destination_steps[1];
                    const std::int64_t s2 = source_steps[2];
                    const std::int64_t d2 = destination_steps[2];
                    if (s2 == 1 && d2 == 1) {
                        std::memcpy(to, from, sizeof(T) * static_cast<std::size_t>(n2));
                        continue;
                    }
                    for (std::int64_t i2 = 0; i2 < n2; ++i2) {
                        to[i2 * d2] = from[i2 * s2];
                    }
                }
            }
            return;
        }
        case 4: {
            const std::int64_t n0 = extents[0];
            const std::int64_t n1 = extents[1];
            const std::int64_t n2 = extents[2];
            const std::int64_t n3 = extents[3];
            for (std::int64_t i0 = 0; i0 < n0; ++i0) {
                for (std::int64_t i1 = 0; i1 < n1; ++i1) {
                    for (std::int64_t i2 = 0; i2 < n2; ++i2) {
                        const T* from = source + i0 * source_steps[0] +
                                        i1 * source_steps[1] + i2 * source_steps[2];
                        T* to = destination + i0 * destination_steps[0] +
                                i1 * destination_steps[1] + i2 * destination_steps[2];
                        const std::int64_t s3 = source_steps[3];
                        const std::int64_t d3 = destination_steps[3];
                        if (s3 == 1 && d3 == 1) {
                            std::memcpy(to, from, sizeof(T) * static_cast<std::size_t>(n3));
                            continue;
                        }
                        for (std::int64_t i3 = 0; i3 < n3; ++i3) {
                            to[i3 * d3] = from[i3 * s3];
                        }
                    }
                }
            }
            return;
        }
        default:
            break;
    }
    /* rank > 4：通用坐标回溯（语料不出现，保留正确性）。 */
    std::int64_t coordinates[kMaxRank] = {};
    std::int64_t source_offset = 0;
    std::int64_t destination_offset = 0;
    std::int64_t outer = 1;
    for (int axis = 0; axis < rank - 1; ++axis) {
        outer *= extents[axis];
    }
    const std::int64_t inner = extents[rank - 1];
    const std::int64_t source_inner = source_steps[rank - 1];
    const std::int64_t destination_inner = destination_steps[rank - 1];
    for (std::int64_t index = 0; index < outer; ++index) {
        const T* from = source + source_offset;
        T* to = destination + destination_offset;
        if (source_inner == 1 && destination_inner == 1) {
            std::memcpy(to, from, sizeof(T) * static_cast<std::size_t>(inner));
        } else {
            for (std::int64_t lane = 0; lane < inner; ++lane) {
                to[lane * destination_inner] = from[lane * source_inner];
            }
        }
        for (int axis = rank - 2; axis >= 0; --axis) {
            source_offset += source_steps[axis];
            destination_offset += destination_steps[axis];
            if (++coordinates[axis] < extents[axis]) {
                break;
            }
            coordinates[axis] = 0;
            source_offset -= source_steps[axis] * extents[axis];
            destination_offset -= destination_steps[axis] * extents[axis];
        }
    }
}

/* ------------------------------------------------------------------ GEMM */

inline void gemm_call(const CblasApi& api, int rows, int cols, int inner, const float* a,
                      const float* b, float* c) {
    api.sgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, rows, cols, inner, 1.0F, a, inner, b,
              cols, 0.0F, c, cols);
}

inline void gemm_call(const CblasApi& api, int rows, int cols, int inner, const double* a,
                      const double* b, double* c) {
    api.dgemm(kCblasRowMajor, kCblasNoTrans, kCblasNoTrans, rows, cols, inner, 1.0, a, inner, b,
              cols, 0.0, c, cols);
}

/* ------------------------------------------------------------------ 执行 */

struct Slot {
    const std::uint8_t* external = nullptr;
    Buffer storage;
    bool owned = false;
    bool live = false;

    const std::uint8_t* data() const { return owned ? storage.data() : external; }
};

inline bool checked_mul(std::int64_t left, std::int64_t right, std::int64_t* result) {
    if (left < 0 || right < 0 ||
        (right != 0 && left > std::numeric_limits<std::int64_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

inline bool checked_bytes(std::int64_t count, std::size_t element, std::size_t* bytes) {
    if (count < 0 || element == 0 ||
        static_cast<std::uint64_t>(count) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element)) {
        return false;
    }
    *bytes = static_cast<std::size_t>(count) * element;
    return true;
}

inline bool product_shape(const Shape& shape, std::int64_t* result) {
    std::int64_t value = 1;
    for (int axis = 0; axis < shape.rank; ++axis) {
        if (!checked_mul(value, shape.extents[axis], &value)) {
            return false;
        }
    }
    *result = value;
    return true;
}

/* 族的编译期常量：元数、输出阶、rank 多重集。25 份实例化各自携带一套。 */
template <int ARITY_, int OUT_RANK_, int... RANKS_>
struct Family {
    static constexpr int kArity = ARITY_;
    static constexpr int kOutRank = OUT_RANK_;
    static constexpr int kRankCount = static_cast<int>(sizeof...(RANKS_));
    static constexpr int kSteps = ARITY_ - 1;                          /* 全二元步 */
    static constexpr int kSlots = ARITY_ > 2 ? ARITY_ - 2 : 0;
    static constexpr int kTotalSlots = ARITY_ + kSlots;

    static constexpr int ranks[kRankCount] = {RANKS_...};
    static constexpr int max_rank() {
        int value = OUT_RANK_;
        for (int index = 0; index < kRankCount; ++index) {
            value = ranks[index] > value ? ranks[index] : value;
        }
        return value;
    }
};

/* 计划的元数、输出阶、rank 多重集必须与模板参数逐项相等——这是“同族”的定义，
 * 也是本入口唯一的结构性准入条件。 */
template <int ARITY, int OUT_RANK, int... RANKS>
inline int check_family(const Plan& plan) {
    using Traits = Family<ARITY, OUT_RANK, RANKS...>;
    if (plan.num_operands != Traits::kArity || plan.out_rank != Traits::kOutRank ||
        plan.num_steps != Traits::kSteps || plan.num_slots != Traits::kSlots) {
        return QCD_STATUS_INVALID;
    }
    int sorted[kMaxArity] = {};
    for (int index = 0; index < ARITY; ++index) {
        const int rank = plan.operands[index].rank;
        if (rank < 0 || rank > kMaxRank) {
            return QCD_STATUS_INVALID;
        }
        int position = index;
        while (position > 0 && sorted[position - 1] > rank) {
            sorted[position] = sorted[position - 1];
            --position;
        }
        sorted[position] = rank;
    }
    for (int index = 0; index < Traits::kRankCount; ++index) {
        if (sorted[index] != Traits::ranks[index]) {
            return QCD_STATUS_INVALID;
        }
    }
    return QCD_STATUS_OK;
}

template <typename T, int ARITY, int OUT_RANK, int... RANKS>
int run_family(const Plan& plan, void* const* operands, void* out) {
    using Traits = Family<ARITY, OUT_RANK, RANKS...>;
    static_assert(Traits::max_rank() <= kMaxRank,
                  "family rank exceeds the fixed axis-table bound");
    const int prepared = check_family<ARITY, OUT_RANK, RANKS...>(plan);
    if (prepared != QCD_STATUS_OK) {
        return prepared;
    }
    const CblasApi& api = cblas_api();
    if (!api.complete()) {
        return QCD_STATUS_INVALID;
    }
    const std::size_t element = sizeof(T);

    Slot slots[Traits::kTotalSlots];
    for (int index = 0; index < ARITY; ++index) {
        const Shape& shape = plan.operands[index];
        std::int64_t count = 0;
        if (!product_shape(shape, &count)) {
            return QCD_STATUS_INVALID;
        }
        slots[index].external = static_cast<const std::uint8_t*>(operands[index]);
        slots[index].live = true;
        if (count != 0 && slots[index].external == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    /* 中间槽一次分配到位：执行期不再申请内存。 */
    for (int index = 0; index < Traits::kSlots; ++index) {
        const Shape& shape = plan.slots[index];
        std::int64_t count = 0;
        std::size_t bytes = 0;
        if (!product_shape(shape, &count) || !checked_bytes(count, element, &bytes)) {
            return QCD_STATUS_INVALID;
        }
        Slot& slot = slots[ARITY + index];
        if (!slot.storage.allocate(bytes)) {
            return QCD_STATUS_MEMORY;
        }
        slot.owned = true;
    }

    /* pack 缓冲按全表最大尺寸一次分配（只算真正需要打包的步）。 */
    std::int64_t max_a = 0;
    std::int64_t max_b = 0;
    std::int64_t max_c = 0;
    for (int index = 0; index < Traits::kSteps; ++index) {
        const Step& step = plan.steps[index];
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
        if (step.pack_a && elements_a > max_a) {
            max_a = elements_a;
        }
        if (step.pack_b && elements_b > max_b) {
            max_b = elements_b;
        }
        if (step.pack_c && elements_c > max_c) {
            max_c = elements_c;
        }
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
    bool wrote_output = false;
    for (int index = 0; index < Traits::kSteps; ++index) {
        const Step& step = plan.steps[index];
        Slot& left_slot = slots[step.a_slot];
        Slot& right_slot = slots[step.b_slot];
        if (!left_slot.live || !right_slot.live) {
            return QCD_STATUS_INVALID;
        }
        std::uint8_t* destination = final_output;
        Slot* destination_slot = nullptr;
        if (step.dst_slot >= 0) {
            destination_slot = &slots[ARITY + step.dst_slot];
            if (destination_slot->live) {
                return QCD_STATUS_INVALID;
            }
            destination = destination_slot->storage.data();
        } else {
            wrote_output = true;
        }
        if (step.m > std::numeric_limits<int>::max() ||
            step.n > std::numeric_limits<int>::max() ||
            step.k > std::numeric_limits<int>::max()) {
            return QCD_STATUS_INVALID;
        }

        const T* left = reinterpret_cast<const T*>(left_slot.data());
        const T* right = reinterpret_cast<const T*>(right_slot.data());
        if (step.m != 0 && step.n != 0 && step.k != 0 &&
            (left == nullptr || right == nullptr)) {
            return QCD_STATUS_INVALID;
        }

        const T* packed_left = left;
        if (step.pack_a) {
            T* buffer = reinterpret_cast<T*>(packed_a.data());
            copy_axes(left, step.a_source_steps, buffer, step.a_packed_steps, step.a_extent,
                      step.rank_a);
            packed_left = buffer;
        }
        const T* packed_right = right;
        if (step.pack_b) {
            T* buffer = reinterpret_cast<T*>(packed_b.data());
            copy_axes(right, step.b_source_steps, buffer, step.b_packed_steps, step.b_extent,
                      step.rank_b);
            packed_right = buffer;
        }

        T* result = step.pack_c ? reinterpret_cast<T*>(packed_c.data())
                                : reinterpret_cast<T*>(destination);
        if (step.m != 0 && step.n != 0) {
            if (step.k == 0) {
                std::memset(result, 0,
                            static_cast<std::size_t>(step.batch * step.m * step.n) * sizeof(T));
            } else {
                for (std::int64_t group = 0; group < step.batch; ++group) {
                    gemm_call(api, static_cast<int>(step.m), static_cast<int>(step.n),
                              static_cast<int>(step.k), packed_left + group * step.m * step.k,
                              packed_right + group * step.k * step.n,
                              result + group * step.m * step.n);
                }
            }
        }
        if (step.pack_c) {
            copy_axes(result, step.c_source_steps, reinterpret_cast<T*>(destination),
                      step.c_destination_steps, step.c_extent, step.rank_c);
        }

        /* 收缩树里每个中间体只被消费一次：读完即标记死亡。 */
        if (step.a_slot >= ARITY) {
            left_slot.live = false;
        }
        if (step.b_slot >= ARITY) {
            right_slot.live = false;
        }
        if (destination_slot != nullptr) {
            destination_slot->live = true;
        }
    }
    return wrote_output ? QCD_STATUS_OK : QCD_STATUS_INVALID;
}

}  // namespace family
}  // namespace pyqcd_einsum

#endif /* PYQCD_KUPL_EINSUM_FAMILY_TEMPLATES_H */
