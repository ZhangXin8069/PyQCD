#include "family_apply.h"

#include "family_templates.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

/* 25 族「可套用模板」入口：格式见 include/family_apply.h。
 *
 * 本 TU 只做三件事：
 *   1. 把步表字节流解析成 family::Plan（形状、轴序、预计算步长，全量校验）；
 *   2. 按 (元数, 输出 rank, 各操作数 rank 多重集) 找到对应的模板实例化；
 *   3. 交给 src/family_templates.h 的 run_family<float/double> 执行。
 *
 * 数值路径与 workspace 策略全在模板头里：pack → cblas_{s,d}gemm → unpack，
 * 中间体与 pack 缓冲来自进程级块池。这里不出现任何 dtype 转换：dtype_id 与
 * 表头不一致、或不是 float32/float64，一律在解析阶段返回 1/4。
 */

namespace {

using pyqcd_einsum::family::Plan;
using pyqcd_einsum::family::Shape;
using pyqcd_einsum::family::Step;
using pyqcd_einsum::family::kMaxArity;
using pyqcd_einsum::family::kMaxRank;
using pyqcd_einsum::family::checked_bytes;
using pyqcd_einsum::family::checked_mul;
using pyqcd_einsum::family::product_shape;

constexpr std::uint64_t kMaxTableBytes = std::uint64_t(1) << 26;

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

/* 行主序紧凑步长（单位：元素）。 */
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

const Shape& slot_shape(const Plan& plan, int slot) {
    if (slot < plan.num_operands) {
        return plan.operands[slot];
    }
    return plan.slots[slot - plan.num_operands];
}

/* ------------------------------------------------------------------ 解析 */

/* 解析 + 全量校验：任何不合规都返回非 0，绝不半途执行。
 *
 * 除通用一致性（形状/槽位/轴排列/乘积）外，本入口还强制二元收缩树形状：
 *   num_steps == num_operands - 1、num_slots == num_steps - 1、每步 arity 2、
 *   只有最后一步 dst_slot == -1。这三点是“同族可套用同一模板”的前提。 */
int parse_plan(const void* data, long long length, int dtype_id, Plan* plan) {
    if (data == nullptr || plan == nullptr || length <= 0 ||
        static_cast<std::uint64_t>(length) > kMaxTableBytes) {
        return QCD_STATUS_INVALID;
    }
    Reader reader(data, static_cast<std::size_t>(length));
    std::int32_t header[10] = {};
    for (int index = 0; index < 10; ++index) {
        if (!reader.read_i32(&header[index])) {
            return QCD_STATUS_INVALID;
        }
    }
    const std::int32_t magic = header[0];
    const std::int32_t version = header[1];
    const std::int32_t total = header[2];
    const std::int32_t num_operands = header[3];
    const std::int32_t num_slots = header[4];
    const std::int32_t num_steps = header[5];
    const std::int32_t out_rank = header[6];
    const std::int32_t table_dtype = header[7];
    if (magic != QCD_FAMILY_MAGIC || version != QCD_FAMILY_VERSION || total <= 0 ||
        static_cast<std::uint64_t>(total) != static_cast<std::uint64_t>(length)) {
        return QCD_STATUS_INVALID;
    }
    if (header[8] != 0 || header[9] != 0) {
        return QCD_STATUS_INVALID;
    }
    if (table_dtype != dtype_id) {
        return QCD_STATUS_INVALID;
    }
    if (table_dtype != QCD_DTYPE_FLOAT32 && table_dtype != QCD_DTYPE_FLOAT64) {
        /* complex 与未知 dtype 没有本入口的数值路径：明确报“不支持”，
         * 绝不隐式降精度。 */
        return QCD_FAMILY_UNSUPPORTED;
    }
    if (num_operands < QCD_FAMILY_MIN_ARITY || num_operands > kMaxArity ||
        out_rank < 0 || out_rank > kMaxRank ||
        num_steps != num_operands - 1 || num_slots != num_steps - 1) {
        return QCD_FAMILY_UNSUPPORTED;
    }

    plan->num_operands = num_operands;
    plan->num_slots = num_slots;
    plan->num_steps = num_steps;
    plan->out_rank = out_rank;
    plan->dtype_id = table_dtype;

    for (int index = 0; index < num_operands; ++index) {
        if (!read_shape(&reader, &plan->operands[index])) {
            return QCD_STATUS_INVALID;
        }
    }
    for (int index = 0; index < num_slots; ++index) {
        if (!read_shape(&reader, &plan->slots[index])) {
            return QCD_STATUS_INVALID;
        }
    }
    if (!read_shape(&reader, &plan->output) || plan->output.rank != out_rank) {
        return QCD_STATUS_INVALID;
    }

    bool written[kMaxArity] = {};
    const int total_slots = num_operands + num_slots;  /* <= 16 + 14，见上面的族形状约束 */
    for (int index = 0; index < num_steps; ++index) {
        Step& step = plan->steps[index];
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
        if (dst_slot >= 0 && written[dst_slot]) {
            return QCD_FAMILY_UNSUPPORTED;
        }
        if (rank_a < 0 || rank_a > kMaxRank || rank_b < 0 || rank_b > kMaxRank ||
            rank_c < 0 || rank_c > kMaxRank || batch_rank < 0 || m_rank < 0 ||
            k_rank < 0) {
            return QCD_STATUS_INVALID;
        }
        if (batch_rank + m_rank + k_rank != rank_a || batch_rank + k_rank > rank_b) {
            return QCD_STATUS_INVALID;
        }
        const int n_rank = rank_b - batch_rank - k_rank;
        if (batch_rank + m_rank + n_rank != rank_c || batch < 0 || m < 0 || n < 0 ||
            k < 0) {
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

        const Shape& left = slot_shape(*plan, a_slot);
        const Shape& right = slot_shape(*plan, b_slot);
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
        const Shape& destination =
            dst_slot >= 0 ? plan->slots[dst_slot] : plan->output;
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
            if (position != axis) {
                step.pack_c = true;
            }
        }
        if (dst_slot >= 0) {
            written[dst_slot] = true;
        }
    }
    if (reader.position() != reader.bytes()) {
        /* 尾随字节：保守拒绝，避免格式版本错配被静默接受。 */
        return QCD_STATUS_INVALID;
    }
    return QCD_STATUS_OK;
}

/* ------------------------------------------------------------------ 族派发 */

using RunFn = int (*)(const Plan&, void* const*, void*, int dtype_id);

/* 一族的全部编译期常量都在模板实参里；运行期只按 (元数, 输出 rank, rank 多重集)
 * 查表。dtype 在实例化内部再分 float/double 两条路径。 */
template <int ARITY, int OUT_RANK, int... RANKS>
int run_family_entry(const Plan& plan, void* const* operands, void* out, int dtype_id) {
    if (dtype_id == QCD_DTYPE_FLOAT32) {
        return pyqcd_einsum::family::run_family<float, ARITY, OUT_RANK, RANKS...>(
            plan, operands, out);
    }
    if (dtype_id == QCD_DTYPE_FLOAT64) {
        return pyqcd_einsum::family::run_family<double, ARITY, OUT_RANK, RANKS...>(
            plan, operands, out);
    }
    return QCD_FAMILY_UNSUPPORTED;
}

template <int... RANKS>
constexpr int count_ranks() {
    return static_cast<int>(sizeof...(RANKS));
}

struct FamilyEntry {
    int arity;
    int out_rank;
    int n_ranks;
    int ranks[kMaxArity];
    RunFn run;
};

/* 编号与 docs/wick_kernel_families_20261111.md 的 F01..F25 逐行对应
 * （该文档按条数降序排列）。 */
#define QCD_FAMILY_ENTRY(ARITY, OUT_RANK, ...)                                            \
    FamilyEntry {                                                                         \
        ARITY, OUT_RANK, count_ranks<__VA_ARGS__>(), {__VA_ARGS__},                       \
            &run_family_entry<ARITY, OUT_RANK, __VA_ARGS__>                               \
    }

const FamilyEntry kFamilies[] = {
    QCD_FAMILY_ENTRY(16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4),   /* F01 */
    QCD_FAMILY_ENTRY(16, 4, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4),   /* F02 */
    QCD_FAMILY_ENTRY(16, 5, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4),   /* F03 */
    QCD_FAMILY_ENTRY(14, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4),         /* F04 */
    QCD_FAMILY_ENTRY(14, 6, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),         /* F05 */
    QCD_FAMILY_ENTRY(16, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4),   /* F06 */
    QCD_FAMILY_ENTRY(16, 5, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4),   /* F07 */
    QCD_FAMILY_ENTRY(13, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4),            /* F08 */
    QCD_FAMILY_ENTRY(13, 4, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4),            /* F09 */
    QCD_FAMILY_ENTRY(13, 5, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4),            /* F10 */
    QCD_FAMILY_ENTRY(13, 3, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4),            /* F11 */
    QCD_FAMILY_ENTRY(13, 5, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4),            /* F12 */
    QCD_FAMILY_ENTRY(10, 4, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4),                     /* F13 */
    QCD_FAMILY_ENTRY(10, 5, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4),                     /* F14 */
    QCD_FAMILY_ENTRY(10, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4),                     /* F15 */
    QCD_FAMILY_ENTRY(12, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4),               /* F16 */
    QCD_FAMILY_ENTRY(12, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4),               /* F17 */
    QCD_FAMILY_ENTRY(12, 3, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4),               /* F18 */
    QCD_FAMILY_ENTRY(9, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4),                         /* F19 */
    QCD_FAMILY_ENTRY(9, 3, 2, 2, 3, 3, 3, 3, 4, 4, 4),                         /* F20 */
    QCD_FAMILY_ENTRY(7, 2, 2, 2, 3, 3, 4, 4, 4),                               /* F21 */
    QCD_FAMILY_ENTRY(9, 0, 2, 2, 2, 2, 2, 2, 4, 4, 4),                         /* F22 */
    QCD_FAMILY_ENTRY(7, 4, 2, 2, 4, 4, 4, 4, 4),                               /* F23 */
    QCD_FAMILY_ENTRY(6, 0, 2, 2, 2, 2, 4, 4),                                  /* F24 */
    QCD_FAMILY_ENTRY(6, 2, 2, 2, 3, 3, 4, 4),                                  /* F25 */
};

#undef QCD_FAMILY_ENTRY

constexpr int kFamilyCount = static_cast<int>(sizeof(kFamilies) / sizeof(kFamilies[0]));

const FamilyEntry* find_family(const Plan& plan) {
    int sorted[kMaxArity] = {};
    for (int index = 0; index < plan.num_operands; ++index) {
        const int rank = plan.operands[index].rank;
        int position = index;
        while (position > 0 && sorted[position - 1] > rank) {
            sorted[position] = sorted[position - 1];
            --position;
        }
        sorted[position] = rank;
    }
    for (int index = 0; index < kFamilyCount; ++index) {
        const FamilyEntry& entry = kFamilies[index];
        if (entry.arity != plan.num_operands || entry.out_rank != plan.out_rank ||
            entry.n_ranks != plan.num_operands) {
            continue;
        }
        bool match = true;
        for (int axis = 0; axis < entry.n_ranks; ++axis) {
            if (entry.ranks[axis] != sorted[axis]) {
                match = false;
                break;
            }
        }
        if (match) {
            return &entry;
        }
    }
    return nullptr;
}

int apply_family(const void* plan_bytes, long long plan_len, void* const* operands,
                 int n_operands, void* out, int dtype_id) {
    if (operands == nullptr || out == nullptr || n_operands <= 0) {
        return QCD_STATUS_INVALID;
    }
    Plan plan;
    const int parsed = parse_plan(plan_bytes, plan_len, dtype_id, &plan);
    if (parsed != QCD_STATUS_OK) {
        return parsed;
    }
    if (n_operands != plan.num_operands) {
        return QCD_STATUS_INVALID;
    }
    const FamilyEntry* family = find_family(plan);
    if (family == nullptr) {
        /* 结构合法但不属于 25 族：明确报“计划不认”，让上层回落而不是猜。 */
        return QCD_FAMILY_UNSUPPORTED;
    }
    return family->run(plan, operands, out, dtype_id);
}

}  // namespace

extern "C" QCD_EINSUM_API int qcd_kupl_family_apply(const void* plan, long long plan_len,
                                                    void* const* operands, int n_operands,
                                                    void* out, int dtype_id) {
    try {
        return apply_family(plan, plan_len, operands, n_operands, out, dtype_id);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::length_error&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}
