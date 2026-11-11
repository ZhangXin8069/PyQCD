#include "einsum_topology.h"

#include "einsum_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kHeaderBytes = 9 * sizeof(int32_t);
constexpr uint64_t kMaxPlanBytes = uint64_t(1) << 30;
constexpr int32_t kMaxInputs = 1 << 16;

struct PlanHeader {
    int32_t magic;
    int32_t version;
    int32_t total_bytes;
    int32_t num_inputs;
    int32_t num_steps;
    int32_t num_symbols;
    int32_t output_ndim;
    int32_t dtype_id;
    int32_t reserved;
};

static_assert(sizeof(PlanHeader) == kHeaderBytes, "version-2 header is nine int32 words");

bool read_header(const void* plan, std::size_t plan_len, PlanHeader* header) {
    if (plan == nullptr || header == nullptr || plan_len < kHeaderBytes) {
        return false;
    }
    std::memcpy(header, plan, kHeaderBytes);
    return true;
}

/* 头部是全量校验的第一道闸：长度、dtype 与操作数个数必须和实际入参逐项一致，
 * 否则连 parse_execution_plan 都不必进。 */
bool header_matches(const PlanHeader& header, long long plan_len, int n_operands,
                    int dtype_id) {
    if (header.magic != static_cast<int32_t>(pyqcd_einsum::kExecutionMagic)) {
        return false;
    }
    if (header.version != static_cast<int32_t>(pyqcd_einsum::kExecutionVersion)) {
        return false;
    }
    if (header.total_bytes <= 0 ||
        static_cast<uint64_t>(header.total_bytes) > kMaxPlanBytes ||
        static_cast<uint64_t>(header.total_bytes) != static_cast<uint64_t>(plan_len)) {
        return false;
    }
    if (header.num_inputs != n_operands) {
        return false;
    }
    /* 精度口径：dtype 由调用方显式给出，必须与计划头一致，绝不隐式转换。 */
    if (header.dtype_id != dtype_id) {
        return false;
    }
    if (header.reserved != 0 || header.num_symbols < 0 || header.output_ndim < 0) {
        return false;
    }
    return true;
}

bool add_offset(int64_t base, int64_t offset, int64_t* result) {
    if (offset > 0 && base > std::numeric_limits<int64_t>::max() - offset) {
        return false;
    }
    if (offset < 0 && base < std::numeric_limits<int64_t>::min() - offset) {
        return false;
    }
    *result = base + offset;
    return true;
}

int apply_topology(const void* plan, long long plan_len, void* const* operands, int n_operands,
                   const long long* operand_offsets, void* out, int dtype_id) {
    if (plan == nullptr || operands == nullptr || out == nullptr) {
        return QCD_STATUS_INVALID;
    }
    if (n_operands <= 0 || n_operands > kMaxInputs) {
        return QCD_STATUS_INVALID;
    }
    if (plan_len <= 0 || static_cast<uint64_t>(plan_len) > kMaxPlanBytes) {
        return QCD_STATUS_INVALID;
    }
    if (dtype_id < QCD_DTYPE_COMPLEX128 || dtype_id > QCD_DTYPE_FLOAT32) {
        return QCD_STATUS_INVALID;
    }

    PlanHeader header;
    if (!read_header(plan, static_cast<std::size_t>(plan_len), &header) ||
        !header_matches(header, plan_len, n_operands, dtype_id)) {
        return QCD_STATUS_INVALID;
    }
    /* 空输入/空步骤不需要执行器：明确归入"本入口不支持"，而不是让执行器在
     * 半途返回一个语义含糊的 1。 */
    if (header.num_inputs <= 0 || header.num_steps <= 0) {
        return QCD_TOPOLOGY_UNSUPPORTED;
    }

    pyqcd_einsum::ExecutionPlan execution;
    std::string error;
    if (!pyqcd_einsum::parse_execution_plan(plan, &execution, &error)) {
        /* 结构层面超出 version-2 的单步 arity ≤ 2 模型（arity > 2、同一操作数
         * 内重复下标、尾随字节等）都归入"本入口不支持"：fail closed，不猜测。 */
        return QCD_TOPOLOGY_UNSUPPORTED;
    }

    std::vector<int64_t> addresses(static_cast<std::size_t>(n_operands), 0);
    for (int index = 0; index < n_operands; ++index) {
        const int64_t base =
            static_cast<int64_t>(reinterpret_cast<std::intptr_t>(operands[index]));
        const int64_t offset = operand_offsets == nullptr ? 0 : operand_offsets[index];
        if (!add_offset(base, offset, &addresses[static_cast<std::size_t>(index)])) {
            return QCD_STATUS_INVALID;
        }
    }
    int64_t output_address = static_cast<int64_t>(reinterpret_cast<std::intptr_t>(out));
    return pyqcd_einsum::execute_parsed_contract(
        execution, static_cast<long long>(reinterpret_cast<std::intptr_t>(addresses.data())),
        static_cast<long long>(reinterpret_cast<std::intptr_t>(&output_address)));
}

}  // namespace

extern "C" QCD_EINSUM_API int qcd_kupl_topology_apply(
    const void* plan, long long plan_len, void* const* operands, int n_operands,
    const long long* operand_offsets, void* out, int dtype_id) {
    try {
        return apply_topology(plan, plan_len, operands, n_operands, operand_offsets, out,
                              dtype_id);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::length_error&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}
