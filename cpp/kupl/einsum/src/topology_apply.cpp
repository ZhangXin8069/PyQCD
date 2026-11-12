#include "einsum_topology.h"

#include "einsum_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

/* per-拓扑 GEMM 步表（由 pyqcd/testing/_topology_kernel_gen.py 生成）。生成物自带
 * 执行器头，因此这里只需要在 parse 之后按"规范指纹"查表即可。
 *
 * 生成物被 .gitignore 忽略（约 2.5 MB 源码、40 万字节步表），干净树里可能不存在：
 * 此时保持旧行为（整条计划交给 version-2 通用执行器），编译与数值语义都不变。
 * 生成命令：``python -m pyqcd.testing._topology_kernel_gen``。 */
#if defined(__has_include)
#if __has_include("topology_kernels.inc")
#include "topology_kernels.inc"
#define PYQCD_TOPO_STEP_TABLE 1
#endif
#endif
#ifndef PYQCD_TOPO_STEP_TABLE
#define PYQCD_TOPO_STEP_TABLE 0
#endif

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

/* ---------------------------------------------------------------- 特化派发
 *
 * 计划字节流里没有方程字面量，但**结构**是完整的：每个操作数的指标序列加上
 * 输出指标序列。把指标按"全式首次出现顺序"重新编号就得到与生成期一致的规范
 * 签名（``0123,4567,01,45,23,67->``），字母改名等价式因此自动命中；操作数顺序
 * 不同则签名不同，保守回落到通用路径。
 *
 * 只在"全部轴长相同（= dim）"时才走特化：步表把每根轴的步长写成 dim 的幂，
 * 广播（轴长 1）或不等长输入都不在这个前提内。
 */
constexpr int kSpecializedMiss = -1;
#if PYQCD_TOPO_STEP_TABLE
constexpr char kSignatureDigits[] = "0123456789abcdefghijklmnopqrstuvwxyz";

uint64_t signature_hash(const std::string& text) {
    uint64_t value = 0xCBF29CE484222325ULL;
    for (char ch : text) {
        value ^= static_cast<uint8_t>(ch);
        value *= 0x100000001B3ULL;
    }
    return value;
}

bool uniform_dimension(const pyqcd_einsum::ExecutionPlan& execution, int64_t* dim) {
    if (execution.sizes.empty() || execution.input_shapes.empty()) {
        return false;
    }
    const int64_t first = execution.sizes[0];
    if (first <= 0) {
        return false;
    }
    for (int64_t size : execution.sizes) {
        if (size != first) {
            return false;
        }
    }
    for (const std::vector<int64_t>& shape : execution.input_shapes) {
        for (int64_t extent : shape) {
            if (extent != first) {
                return false;
            }
        }
    }
    *dim = first;
    return true;
}

bool build_canonical_signature(const pyqcd_einsum::ExecutionPlan& execution,
                               std::string* signature) {
    const std::size_t inputs = execution.input_shapes.size();
    std::vector<std::vector<int32_t>> labels(inputs);
    std::vector<char> seen(inputs, 0);
    /* 计划里的 positions 是**活表下标**：每一步先按降序弹掉被收缩的两个槽，再把
     * 结果压回表尾，所以同一个下标在不同步骤里指向不同张量。这里先按同样的
     * pop/append 序列把它们翻译回稳定的槽号（>= 0 是输入，< 0 是中间结果），
     * 否则第二步起就会把中间结果的指标当成某个输入项的指标。
     *
     * 输入项的指标序列必须逐位一致：plan 的 symbol 编号按"输入项首次出现顺序"
     * 分配，因此这个签名与生成期（pyqcd.testing._topology_kernel_gen）的规范签名
     * 逐字节相同，字母改名等价式也会命中同一条记录。 */
    std::vector<int32_t> live(inputs, 0);
    for (std::size_t index = 0; index < inputs; ++index) {
        live[index] = static_cast<int32_t>(index);
    }
    for (const pyqcd_einsum::ExecutionStep& step : execution.steps) {
        if (step.positions.size() != step.inputs.size()) {
            return false;
        }
        for (std::size_t index = 0; index < step.positions.size(); ++index) {
            const int32_t position = step.positions[index];
            if (position < 0 || static_cast<std::size_t>(position) >= live.size()) {
                return false;
            }
            const pyqcd_einsum::TensorMeta& meta = step.inputs[index];
            if (meta.labels.size() != meta.shape.size()) {
                return false;
            }
            const int32_t slot = live[static_cast<std::size_t>(position)];
            if (slot < 0) {
                continue; /* 中间结果不参与输入签名 */
            }
            const std::size_t stable = static_cast<std::size_t>(slot);
            if (seen[stable] != 0) {
                if (labels[stable] != meta.labels) {
                    return false;
                }
                continue;
            }
            labels[stable] = meta.labels;
            seen[stable] = 1;
        }
        std::vector<int32_t> order(step.positions.begin(), step.positions.end());
        std::sort(order.begin(), order.end(), std::greater<int32_t>());
        for (int32_t position : order) {
            live.erase(live.begin() + position);
        }
        live.push_back(-1);
    }
    for (std::size_t index = 0; index < inputs; ++index) {
        if (seen[index] == 0 || labels[index].size() != execution.input_shapes[index].size()) {
            return false;
        }
    }

    signature->clear();
    for (std::size_t index = 0; index < inputs; ++index) {
        if (index != 0) {
            signature->push_back(',');
        }
        for (int32_t label : labels[index]) {
            if (label < 0 || label >= static_cast<int32_t>(sizeof(kSignatureDigits) - 1)) {
                return false;
            }
            signature->push_back(kSignatureDigits[label]);
        }
    }
    signature->append("->");
    for (int32_t label : execution.output_labels) {
        if (label < 0 || label >= static_cast<int32_t>(sizeof(kSignatureDigits) - 1)) {
            return false;
        }
        signature->push_back(kSignatureDigits[label]);
    }
    return true;
}

bool trace_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("PYQCD_EINSUM_TOPOLOGY_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}
#endif  /* PYQCD_TOPO_STEP_TABLE */

/* 命中步表就把整条收缩交给参数化 GEMM 执行器；任何不适配都返回 kSpecializedMiss
 * 让调用方走 version-2 通用执行器（fail closed，绝不猜测）。 */
int run_specialized(const pyqcd_einsum::ExecutionPlan& execution,
                    const std::vector<int64_t>& addresses, void* out, int dtype_id) {
#if !PYQCD_TOPO_STEP_TABLE
    /* 干净树（没有生成的步表）保持旧行为。 */
    (void)execution;
    (void)addresses;
    (void)out;
    (void)dtype_id;
    return kSpecializedMiss;
#else
    int elem_kind = 0;
    if (dtype_id == QCD_DTYPE_FLOAT32) {
        elem_kind = pyqcd_einsum::topology_gemm::kFloat;
    } else if (dtype_id == QCD_DTYPE_FLOAT64) {
        elem_kind = pyqcd_einsum::topology_gemm::kDouble;
    } else {
        /* 复数没有特化步表：绝不隐式降精度，直接回落通用路径。 */
        return kSpecializedMiss;
    }

    int64_t dim = 0;
    if (!uniform_dimension(execution, &dim)) {
        return kSpecializedMiss;
    }
    std::string signature;
    if (!build_canonical_signature(execution, &signature)) {
        return kSpecializedMiss;
    }
    const pyqcd_einsum::topology_gemm::Record* record =
        pyqcd_einsum::topology_table::topo_find(signature_hash(signature));
    if (record == nullptr || signature != record->signature) {
        return kSpecializedMiss;
    }
    if (record->arity != execution.input_shapes.size() ||
        record->out_rank != execution.output_labels.size()) {
        return kSpecializedMiss;
    }
    int64_t output_elements = 0;
    if (!pyqcd_einsum::topology_gemm::checked_power(
            dim, static_cast<int>(record->out_rank), &output_elements)) {
        return kSpecializedMiss;
    }
    std::vector<const void*> operands(addresses.size(), nullptr);
    for (std::size_t index = 0; index < addresses.size(); ++index) {
        operands[index] = reinterpret_cast<const void*>(
            static_cast<std::intptr_t>(addresses[index]));
    }
    const uint8_t* code =
        reinterpret_cast<const uint8_t*>(pyqcd_einsum::topology_table::kTopoCode) +
        record->code_offset;
    const int status = pyqcd_einsum::topology_gemm::run(
        *record, code, operands.data(), out, elem_kind, dim);
    if (status != QCD_STATUS_OK) {
        return kSpecializedMiss;
    }
    if (trace_enabled()) {
        std::fprintf(stderr,
                     "[pyqcd.einsum] topology-specialized: %s dim=%lld dtype=%d "
                     "steps=%d\n",
                     record->signature, static_cast<long long>(dim), dtype_id,
                     static_cast<int>(record->step_count));
    }
    (void)output_elements;
    return QCD_STATUS_OK;
#endif
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
    /* 语料拓扑优先走编译期步表（per-拓扑 GEMM 分步）；不适配就保持原有语义，
     * 交给 version-2 通用执行器。 */
    const int specialized = run_specialized(execution, addresses, out, dtype_id);
    if (specialized == QCD_STATUS_OK) {
        return QCD_STATUS_OK;
    }
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
