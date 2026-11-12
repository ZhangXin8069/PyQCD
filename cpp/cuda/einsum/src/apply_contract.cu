#include "einsum_cuda.h"
#include "einsum_runtime.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/einsum_topology.h"

#include "topology_steps.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

bool contains(const std::vector<int32_t>& values, int32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

int32_t position_of(const std::vector<int32_t>& labels, int32_t label) {
    const auto found = std::find(labels.begin(), labels.end(), label);
    return found == labels.end() ? -1 : static_cast<int32_t>(found - labels.begin());
}

bool decode_execution_plan(const void* data, pyqcd_einsum::ExecutionPlan* plan) {
    if (data == nullptr || plan == nullptr) {
        return false;
    }
    std::string error;
    pyqcd_einsum::ExecutionPlan parsed;
    if (!pyqcd_einsum::parse_execution_plan(data, &parsed, &error) ||
        parsed.num_inputs <= 0 || parsed.steps.empty() ||
        parsed.steps.back().output.labels != parsed.output_labels) {
        return false;
    }
    *plan = std::move(parsed);
    return true;
}

template <typename Type>
Type* pointer_from_address(long long address) {
    if (address <= 0) {
        return nullptr;
    }
    return reinterpret_cast<Type*>(static_cast<std::uintptr_t>(address));
}

bool multiply_checked(int64_t left, int64_t right, int64_t* product) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool checked_product(const std::vector<int64_t>& shape, int64_t* product) {
    int64_t value = 1;
    for (int64_t extent : shape) {
        if (!multiply_checked(value, extent, &value)) {
            return false;
        }
    }
    *product = value;
    return true;
}

bool checked_product(const int64_t* shape, int32_t rank, int64_t* product) {
    int64_t value = 1;
    for (int32_t axis = 0; axis < rank; ++axis) {
        if (!multiply_checked(value, shape[axis], &value)) {
            return false;
        }
    }
    *product = value;
    return true;
}

int64_t dtype_size(int dtype_id) {
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            return 16;
        case QCD_DTYPE_COMPLEX64:
            return 8;
        case QCD_DTYPE_FLOAT64:
            return 8;
        case QCD_DTYPE_FLOAT32:
            return 4;
        default:
            return 0;
    }
}

bool allocation_bytes(int64_t elements, int dtype_id, std::size_t* bytes) {
    const int64_t element_size = dtype_size(dtype_id);
    int64_t total = 0;
    if (element_size <= 0 || !multiply_checked(elements, element_size, &total) ||
        static_cast<uint64_t>(total) > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    *bytes = total == 0 ? 1 : static_cast<std::size_t>(total);
    return true;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        strides[index - 2] = strides[index - 1] * shape[index - 1];
    }
    return strides;
}

bool target_shape(
    const std::vector<int32_t>& labels,
    const std::vector<int64_t>& sizes,
    std::vector<int64_t>* shape) {
    shape->resize(labels.size());
    for (std::size_t axis = 0; axis < labels.size(); ++axis) {
        const int32_t label = labels[axis];
        if (label < 0 || static_cast<std::size_t>(label) >= sizes.size()) {
            return false;
        }
        (*shape)[axis] = sizes[static_cast<std::size_t>(label)];
    }
    return true;
}

bool make_copy_spec(
    const std::vector<int32_t>& source_labels,
    const std::vector<int64_t>& source_shape,
    const std::vector<int32_t>& destination_labels,
    const std::vector<int64_t>& destination_shape,
    qcd_cuda_copy_spec* spec) {
    if (source_labels.size() != source_shape.size() ||
        destination_labels.size() != destination_shape.size() ||
        destination_labels.size() > static_cast<std::size_t>(QCD_CUDA_MAX_PACKED_RANK)) {
        return false;
    }
    int64_t elements = 0;
    if (!checked_product(destination_shape, &elements)) {
        return false;
    }
    const std::vector<int64_t> source_strides = contiguous_strides(source_shape);
    *spec = qcd_cuda_copy_spec{};
    spec->rank = static_cast<int32_t>(destination_labels.size());
    spec->elements = elements;
    for (std::size_t axis = 0; axis < destination_labels.size(); ++axis) {
        spec->shape[axis] = destination_shape[axis];
        const int32_t source_axis = position_of(source_labels, destination_labels[axis]);
        if (source_axis < 0) {
            spec->src_stride[axis] = 0;
            continue;
        }
        const int64_t source_extent = source_shape[static_cast<std::size_t>(source_axis)];
        const int64_t destination_extent = destination_shape[axis];
        if (source_extent == destination_extent) {
            spec->src_stride[axis] = source_strides[static_cast<std::size_t>(source_axis)];
        } else if (source_extent == 1 && destination_extent > 1) {
            spec->src_stride[axis] = 0;
        } else {
            return false;
        }
    }
    return true;
}

bool make_hadamard_spec(
    const std::vector<int32_t>& left_labels,
    const std::vector<int64_t>& left_shape,
    const std::vector<int32_t>& right_labels,
    const std::vector<int64_t>& right_shape,
    const std::vector<int32_t>& output_labels,
    const std::vector<int64_t>& output_shape,
    qcd_cuda_hadamard_spec* spec) {
    if (left_labels.size() != left_shape.size() || right_labels.size() != right_shape.size() ||
        output_labels.size() != output_shape.size() ||
        output_labels.size() > static_cast<std::size_t>(QCD_CUDA_MAX_RANK)) {
        return false;
    }
    int64_t elements = 0;
    if (!checked_product(output_shape, &elements)) {
        return false;
    }
    const std::vector<int64_t> left_strides = contiguous_strides(left_shape);
    const std::vector<int64_t> right_strides = contiguous_strides(right_shape);
    *spec = qcd_cuda_hadamard_spec{};
    spec->rank = static_cast<int32_t>(output_labels.size());
    spec->elements = elements;
    for (std::size_t axis = 0; axis < output_labels.size(); ++axis) {
        spec->shape[axis] = output_shape[axis];
        const int32_t left_axis = position_of(left_labels, output_labels[axis]);
        const int32_t right_axis = position_of(right_labels, output_labels[axis]);
        if (left_axis < 0) {
            spec->left_stride[axis] = 0;
        } else {
            const int64_t extent = left_shape[static_cast<std::size_t>(left_axis)];
            if (extent == output_shape[axis]) {
                spec->left_stride[axis] = left_strides[static_cast<std::size_t>(left_axis)];
            } else if (extent == 1 && output_shape[axis] > 1) {
                spec->left_stride[axis] = 0;
            } else {
                return false;
            }
        }
        if (right_axis < 0) {
            spec->right_stride[axis] = 0;
        } else {
            const int64_t extent = right_shape[static_cast<std::size_t>(right_axis)];
            if (extent == output_shape[axis]) {
                spec->right_stride[axis] = right_strides[static_cast<std::size_t>(right_axis)];
            } else if (extent == 1 && output_shape[axis] > 1) {
                spec->right_stride[axis] = 0;
            } else {
                return false;
            }
        }
    }
    return true;
}

bool make_reduce_spec(
    const std::vector<int32_t>& input_labels,
    const std::vector<int64_t>& input_shape,
    const std::vector<int32_t>& output_labels,
    const std::vector<int64_t>& output_shape,
    qcd_cuda_reduce_spec* spec) {
    if (input_labels.size() != input_shape.size() || output_labels.size() != output_shape.size() ||
        output_labels.size() > static_cast<std::size_t>(QCD_CUDA_MAX_RANK)) {
        return false;
    }
    const std::vector<int64_t> input_strides = contiguous_strides(input_shape);
    *spec = qcd_cuda_reduce_spec{};
    spec->output_rank = static_cast<int32_t>(output_labels.size());
    for (std::size_t axis = 0; axis < output_labels.size(); ++axis) {
        const int32_t input_axis = position_of(input_labels, output_labels[axis]);
        if (input_axis < 0) {
            return false;
        }
        spec->output_shape[axis] = output_shape[axis];
        const int64_t extent = input_shape[static_cast<std::size_t>(input_axis)];
        if (extent == output_shape[axis]) {
            spec->output_src_stride[axis] = input_strides[static_cast<std::size_t>(input_axis)];
        } else if (extent == 1 && output_shape[axis] > 1) {
            spec->output_src_stride[axis] = 0;
        } else {
            return false;
        }
    }
    for (std::size_t input_axis = 0; input_axis < input_labels.size(); ++input_axis) {
        const int32_t label = input_labels[input_axis];
        if (contains(output_labels, label)) {
            continue;
        }
        const int32_t reduction_axis = spec->reduction_rank++;
        spec->reduction_extent[reduction_axis] = input_shape[input_axis];
        spec->reduction_src_stride[reduction_axis] = input_strides[input_axis];
    }
    return checked_product(output_shape, &spec->output_elements) &&
           checked_product(spec->reduction_extent, spec->reduction_rank,
                           &spec->reduction_elements);
}

struct NodeMeta {
    std::vector<int32_t> labels;
    std::vector<int64_t> shape;
    bool fixed_labels = false;
};

struct ResolvedStep {
    std::vector<int32_t> operands;
    int32_t output = -1;
};

bool validate_operand(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<NodeMeta>& nodes,
    int32_t node_id,
    const pyqcd_einsum::TensorMeta& expected) {
    if (node_id < 0 || static_cast<std::size_t>(node_id) >= nodes.size()) {
        return false;
    }
    const NodeMeta& node = nodes[static_cast<std::size_t>(node_id)];
    if (node.shape.size() != expected.labels.size()) {
        return false;
    }
    if (node.fixed_labels && node.labels != expected.labels) {
        return false;
    }
    for (std::size_t axis = 0; axis < expected.labels.size(); ++axis) {
        const int32_t label = expected.labels[axis];
        if (label < 0 || static_cast<std::size_t>(label) >= plan.sizes.size()) {
            return false;
        }
        const int64_t target = plan.sizes[static_cast<std::size_t>(label)];
        const int64_t extent = node.shape[axis];
        if (extent != target && !(extent == 1 && target > 1)) {
            return false;
        }
    }
    return true;
}

bool resolve_plan(
    const pyqcd_einsum::ExecutionPlan& plan,
    std::vector<ResolvedStep>* resolved) {
    const int32_t num_nodes = plan.num_inputs + static_cast<int32_t>(plan.steps.size());
    std::vector<NodeMeta> nodes(static_cast<std::size_t>(num_nodes));
    std::vector<bool> ready(static_cast<std::size_t>(num_nodes), false);
    std::vector<int32_t> slots(static_cast<std::size_t>(plan.num_inputs));
    for (int32_t input = 0; input < plan.num_inputs; ++input) {
        nodes[static_cast<std::size_t>(input)].shape =
            plan.input_shapes[static_cast<std::size_t>(input)];
        slots[static_cast<std::size_t>(input)] = input;
        ready[static_cast<std::size_t>(input)] = true;
    }

    std::vector<ResolvedStep> candidate;
    candidate.reserve(plan.steps.size());
    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const pyqcd_einsum::ExecutionStep& step = plan.steps[step_index];
        ResolvedStep step_result;
        std::vector<int32_t> unique_positions = step.positions;
        std::sort(unique_positions.begin(), unique_positions.end());
        if (std::adjacent_find(unique_positions.begin(), unique_positions.end()) !=
            unique_positions.end()) {
            return false;
        }
        for (int32_t position : step.positions) {
            if (position < 0 || static_cast<std::size_t>(position) >= slots.size()) {
                return false;
            }
            step_result.operands.push_back(slots[static_cast<std::size_t>(position)]);
        }

        for (std::size_t operand = 0; operand < step_result.operands.size(); ++operand) {
            if (!validate_operand(plan, nodes, step_result.operands[operand],
                                  step.inputs[operand])) {
                return false;
            }
        }

        step_result.output = plan.num_inputs + static_cast<int32_t>(step_index);
        if (step_result.output < 0 || static_cast<std::size_t>(step_result.output) >= nodes.size() ||
            ready[static_cast<std::size_t>(step_result.output)]) {
            return false;
        }
        NodeMeta& output = nodes[static_cast<std::size_t>(step_result.output)];
        output.labels = step.output.labels;
        if (!target_shape(output.labels, plan.sizes, &output.shape)) {
            return false;
        }
        output.fixed_labels = true;
        ready[static_cast<std::size_t>(step_result.output)] = true;

        std::sort(unique_positions.begin(), unique_positions.end(), std::greater<int32_t>());
        for (int32_t position : unique_positions) {
            slots.erase(slots.begin() + position);
        }
        slots.push_back(step_result.output);
        candidate.push_back(std::move(step_result));
    }
    if (slots.size() != 1) {
        return false;
    }
    *resolved = std::move(candidate);
    return true;
}

enum class OperationKind {
    kCopy,
    kReduce,
    kHadamard,
    kGemm,
};

struct PreparedStep {
    OperationKind kind = OperationKind::kCopy;
    int32_t left = -1;
    int32_t right = -1;
    int32_t output = -1;
    int64_t output_elements = 0;
    qcd_cuda_copy_spec copy{};
    qcd_cuda_reduce_spec reduce{};
    qcd_cuda_hadamard_spec hadamard{};
    qcd_cuda_copy_spec left_pack{};
    qcd_cuda_copy_spec right_pack{};
    qcd_cuda_copy_spec output_permute{};
    int64_t m = 1;
    int64_t n = 1;
    int64_t k = 1;
    int64_t batch = 1;
    bool permute_output = false;
    /* pack 恒等：操作数已在所需 pack 序上且形状一致，无需再拷一次。
     * 这是 per-拓扑步表最常命中的优化——中间体本来就是上一步的规范输出。 */
    bool left_pack_identity = false;
    bool right_pack_identity = false;
};

bool same_labels(const std::vector<int32_t>& left, const std::vector<int32_t>& right) {
    return left == right;
}

bool prepare_binary(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<NodeMeta>& nodes,
    const ResolvedStep& resolved,
    const pyqcd_einsum::ExecutionStep& step,
    PreparedStep* prepared) {
    const std::vector<int32_t>& left_labels = step.inputs[0].labels;
    const std::vector<int32_t>& right_labels = step.inputs[1].labels;
    const std::vector<int64_t>& left_shape = nodes[resolved.operands[0]].shape;
    const std::vector<int64_t>& right_shape = nodes[resolved.operands[1]].shape;
    const std::vector<int32_t>& output_labels = step.output.labels;
    std::vector<int64_t> output_shape;
    if (!target_shape(output_labels, plan.sizes, &output_shape)) {
        return false;
    }
    for (int32_t label : output_labels) {
        if (!contains(left_labels, label) && !contains(right_labels, label)) {
            return false;
        }
    }

    std::vector<int32_t> batch_labels;
    std::vector<int32_t> left_free;
    std::vector<int32_t> right_free;
    std::vector<int32_t> reduction_labels;
    for (int32_t label : left_labels) {
        if (contains(right_labels, label)) {
            if (contains(output_labels, label)) {
                batch_labels.push_back(label);
            } else {
                reduction_labels.push_back(label);
            }
        } else if (contains(output_labels, label)) {
            left_free.push_back(label);
        } else {
            reduction_labels.push_back(label);
        }
    }
    for (int32_t label : right_labels) {
        if (!contains(left_labels, label)) {
            if (contains(output_labels, label)) {
                right_free.push_back(label);
            } else {
                reduction_labels.push_back(label);
            }
        }
    }

    prepared->left = resolved.operands[0];
    prepared->right = resolved.operands[1];
    prepared->output = resolved.output;
    if (reduction_labels.empty()) {
        prepared->kind = OperationKind::kHadamard;
        if (!make_hadamard_spec(left_labels, left_shape, right_labels, right_shape,
                                output_labels, output_shape, &prepared->hadamard) ||
            !checked_product(output_shape, &prepared->output_elements)) {
            return false;
        }
        return true;
    }

    prepared->kind = OperationKind::kGemm;
    std::vector<int32_t> left_pack_labels = batch_labels;
    left_pack_labels.insert(left_pack_labels.end(), left_free.begin(), left_free.end());
    left_pack_labels.insert(left_pack_labels.end(), reduction_labels.begin(), reduction_labels.end());
    std::vector<int32_t> right_pack_labels = batch_labels;
    right_pack_labels.insert(right_pack_labels.end(), right_free.begin(), right_free.end());
    right_pack_labels.insert(right_pack_labels.end(), reduction_labels.begin(), reduction_labels.end());
    std::vector<int32_t> canonical_labels = batch_labels;
    canonical_labels.insert(canonical_labels.end(), left_free.begin(), left_free.end());
    canonical_labels.insert(canonical_labels.end(), right_free.begin(), right_free.end());

    std::vector<int64_t> left_pack_shape;
    std::vector<int64_t> right_pack_shape;
    std::vector<int64_t> canonical_shape;
    if (!target_shape(left_pack_labels, plan.sizes, &left_pack_shape) ||
        !target_shape(right_pack_labels, plan.sizes, &right_pack_shape) ||
        !target_shape(canonical_labels, plan.sizes, &canonical_shape) ||
        !make_copy_spec(left_labels, left_shape, left_pack_labels, left_pack_shape,
                        &prepared->left_pack) ||
        !make_copy_spec(right_labels, right_shape, right_pack_labels, right_pack_shape,
                        &prepared->right_pack)) {
        return false;
    }

    std::vector<int64_t> batch_shape;
    std::vector<int64_t> left_free_shape;
    std::vector<int64_t> right_free_shape;
    std::vector<int64_t> reduction_shape;
    if (!target_shape(batch_labels, plan.sizes, &batch_shape) ||
        !target_shape(left_free, plan.sizes, &left_free_shape) ||
        !target_shape(right_free, plan.sizes, &right_free_shape) ||
        !target_shape(reduction_labels, plan.sizes, &reduction_shape) ||
        !checked_product(batch_shape, &prepared->batch) ||
        !checked_product(left_free_shape, &prepared->m) ||
        !checked_product(right_free_shape, &prepared->n) ||
        !checked_product(reduction_shape, &prepared->k) ||
        !checked_product(canonical_shape, &prepared->output_elements)) {
        return false;
    }
    if (prepared->m > std::numeric_limits<int>::max() ||
        prepared->n > std::numeric_limits<int>::max() ||
        prepared->k > std::numeric_limits<int>::max() ||
        prepared->batch > std::numeric_limits<int>::max()) {
        return false;
    }

    prepared->permute_output = !same_labels(canonical_labels, output_labels);
    if (prepared->permute_output &&
        !make_copy_spec(canonical_labels, canonical_shape, output_labels, output_shape,
                        &prepared->output_permute)) {
        return false;
    }
    /* 只有在标签序与形状都逐位一致时，pack 才是恒等拷贝：
     * 此时 src_stride 恰好等于目标连续步长，跳过拷贝不影响任何元素。 */
    prepared->left_pack_identity =
        same_labels(left_pack_labels, left_labels) && left_pack_shape == left_shape;
    prepared->right_pack_identity =
        same_labels(right_pack_labels, right_labels) && right_pack_shape == right_shape;
    return true;
}

bool prepare_execution(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<ResolvedStep>& resolved,
    std::vector<PreparedStep>* prepared) {
    std::vector<NodeMeta> nodes(
        static_cast<std::size_t>(plan.num_inputs + static_cast<int32_t>(plan.steps.size())));
    for (int32_t input = 0; input < plan.num_inputs; ++input) {
        nodes[static_cast<std::size_t>(input)].shape =
            plan.input_shapes[static_cast<std::size_t>(input)];
    }
    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const pyqcd_einsum::ExecutionStep& step = plan.steps[step_index];
        const ResolvedStep& step_resolved = resolved[step_index];
        NodeMeta& output = nodes[static_cast<std::size_t>(step_resolved.output)];
        output.labels = step.output.labels;
        output.fixed_labels = true;
        if (!target_shape(output.labels, plan.sizes, &output.shape)) {
            return false;
        }
    }

    prepared->clear();
    prepared->reserve(plan.steps.size());
    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const pyqcd_einsum::ExecutionStep& step = plan.steps[step_index];
        const ResolvedStep& step_resolved = resolved[step_index];
        PreparedStep item;
        item.output = step_resolved.output;
        if (step.inputs.size() == 1) {
            const std::vector<int32_t>& input_labels = step.inputs[0].labels;
            const std::vector<int64_t>& input_shape = nodes[step_resolved.operands[0]].shape;
            const std::vector<int32_t>& output_labels = step.output.labels;
            std::vector<int64_t> output_shape;
            if (!target_shape(output_labels, plan.sizes, &output_shape)) {
                return false;
            }
            for (int32_t label : output_labels) {
                if (!contains(input_labels, label)) {
                    return false;
                }
            }
            item.left = step_resolved.operands[0];
            std::vector<int32_t> reduction_labels;
            for (int32_t label : input_labels) {
                if (!contains(output_labels, label)) {
                    reduction_labels.push_back(label);
                }
            }
            if (reduction_labels.empty()) {
                item.kind = OperationKind::kCopy;
                if (!make_copy_spec(input_labels, input_shape, output_labels, output_shape,
                                    &item.copy) ||
                    !checked_product(output_shape, &item.output_elements)) {
                    return false;
                }
            } else {
                item.kind = OperationKind::kReduce;
                if (!make_reduce_spec(input_labels, input_shape, output_labels, output_shape,
                                      &item.reduce)) {
                    return false;
                }
                item.output_elements = item.reduce.output_elements;
            }
        } else {
            if (!prepare_binary(plan, nodes, step_resolved, step, &item)) {
                return false;
            }
        }
        prepared->push_back(std::move(item));
    }
    return true;
}

/* -------------------------------------------------------------------------
 * 拓扑描述符 → 预编译步表 → 执行计划
 *
 * 描述符（由 `pyqcd/einsum/_kernel_cuda.py` 生产）只带**结构**：dtype、每个
 * 操作数的 rank/标签/形状、输出标签，以及拓扑签名摘要。收缩顺序不在描述符里，
 * 而是取 `topology_steps.h` 的编译期常量步表（每拓扑一条紧凑步表）——运行期
 * 不做 path 搜索，Python 也不逐步调度。
 *
 * 步表只存"位置对"（相对活表），标签在原生侧按 einsum 语义展开：
 *   - 中间体按 (与另一操作数共享的批次标, 本侧自由标, 对侧自由标) 的规范序排布，
 *     与 `prepare_binary` 建 pack 序的口径一致，可让下一步的 pack 判定为恒等；
 *   - 末步输出直接取描述符给的输出标签顺序，省掉一次输出 permute；
 *   - 任何越界/标签失配/尺寸超限都 fail closed（返回 UNSUPPORTED，由上层回落）。
 *
 * 摘要 key 用签名 blake2b-128 的前 8 字节（与 `_topology_registry` 同一口径），
 * 另加规范标签流的 64 位 FNV 指纹由原生侧重算并比对：表项、摘要、实际结构三者
 * 必须同时吻合，否则这条计划本入口不认。
 * ---------------------------------------------------------------------- */

constexpr uint32_t kTopologyDescriptorMagic = 0x51435444u; /* 'DTCQ' */
constexpr uint32_t kTopologyDescriptorVersion = 1u;
constexpr int32_t kTopologyDescriptorHeaderWords = 8;
constexpr int32_t kTopologyDescriptorDigestWords = 4;
constexpr int32_t kTopologyDescriptorMaxInputs = 64;
constexpr int32_t kTopologyDescriptorMaxLabels = 200;
constexpr int64_t kTopologyMaxIntermediateElements = int64_t(1) << 31;

bool topology_dtype_supported(int dtype_id) {
    return dtype_id >= QCD_DTYPE_COMPLEX128 && dtype_id <= QCD_DTYPE_FLOAT32;
}

struct TopologyDescriptor {
    int32_t dtype_id = -1;
    int32_t num_inputs = 0;
    int32_t num_labels = 0;
    uint64_t key = 0;
    std::vector<int64_t> sizes;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int32_t>> input_labels;
    std::vector<int32_t> output_labels;
};

constexpr uint64_t kFnvOffsetBasis = 0xCBF29CE484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001B3ULL;

struct FnvHasher {
    uint64_t state = kFnvOffsetBasis;
    void byte(uint8_t value) {
        state = (state ^ value) * kFnvPrime;
    }
};

uint64_t topology_key_from_digest(const uint8_t* digest) {
    uint64_t value = 0;
    for (int32_t index = 0; index < 8; ++index) {
        value = (value << 8) | static_cast<uint64_t>(digest[index]);
    }
    return value;
}

/* 与 `pyqcd/testing` 侧生成脚本同一口径：每个操作数 rank 字节 + 标签字节，
 * 0xFF 分隔后接输出标签。 */
uint64_t topology_structure_fingerprint(
    const std::vector<std::vector<int32_t>>& inputs,
    const std::vector<int32_t>& output) {
    FnvHasher hasher;
    for (const std::vector<int32_t>& labels : inputs) {
        hasher.byte(static_cast<uint8_t>(labels.size()));
        for (int32_t label : labels) {
            hasher.byte(static_cast<uint8_t>(label));
        }
    }
    hasher.byte(0xFF);
    for (int32_t label : output) {
        hasher.byte(static_cast<uint8_t>(label));
    }
    return hasher.state;
}

const qcd_topology_entry* find_topology_entry(uint64_t key, uint64_t fingerprint) {
    int32_t low = 0;
    int32_t high = kQcdTopoEntryCount - 1;
    while (low <= high) {
        const int32_t middle = low + (high - low) / 2;
        const qcd_topology_entry& entry = kQcdTopoEntries[middle];
        if (entry.key == key) {
            /* 指纹与元数必须同时吻合：摘要撞车或表/代码漂移都不得静默执行。 */
            return entry.fnv == fingerprint ? &entry : nullptr;
        }
        if (entry.key < key) {
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    return nullptr;
}

bool is_topology_descriptor(const void* data, long long bytes) {
    if (data == nullptr ||
        bytes < static_cast<long long>(kTopologyDescriptorHeaderWords * sizeof(int32_t))) {
        return false;
    }
    int32_t magic = 0;
    std::memcpy(&magic, data, sizeof(magic));
    return magic == static_cast<int32_t>(kTopologyDescriptorMagic);
}

class DescriptorReader {
  public:
    DescriptorReader(const void* data, long long bytes)
        : data_(static_cast<const uint8_t*>(data)), bytes_(bytes) {}

    bool read_i32(int32_t* value) {
        if (offset_ + static_cast<long long>(sizeof(int32_t)) > bytes_) {
            return false;
        }
        std::memcpy(value, data_ + offset_, sizeof(int32_t));
        offset_ += static_cast<long long>(sizeof(int32_t));
        return true;
    }

    bool at_end() const { return offset_ == bytes_; }
  private:
    const uint8_t* data_;
    long long bytes_;
    long long offset_ = 0;
};

bool parse_topology_descriptor(
    const void* data, long long bytes, TopologyDescriptor* descriptor) {
    if (!is_topology_descriptor(data, bytes) || descriptor == nullptr) {
        return false;
    }
    DescriptorReader reader(data, bytes);
    int32_t header[kTopologyDescriptorHeaderWords] = {0};
    for (int32_t index = 0; index < kTopologyDescriptorHeaderWords; ++index) {
        if (!reader.read_i32(&header[index])) {
            return false;
        }
    }
    const int32_t version = header[1];
    const int32_t total_bytes = header[2];
    const int32_t dtype_id = header[3];
    const int32_t num_inputs = header[4];
    const int32_t num_labels = header[5];
    const int32_t num_output = header[6];
    if (version != static_cast<int32_t>(kTopologyDescriptorVersion) ||
        total_bytes != bytes || !topology_dtype_supported(dtype_id) ||
        num_inputs <= 0 || num_inputs > kTopologyDescriptorMaxInputs ||
        num_labels < 0 || num_labels > kTopologyDescriptorMaxLabels ||
        num_output < 0 || num_output > num_labels) {
        return false;
    }

    uint8_t digest[16] = {0};
    int32_t digest_words[kTopologyDescriptorDigestWords] = {0};
    for (int32_t index = 0; index < kTopologyDescriptorDigestWords; ++index) {
        if (!reader.read_i32(&digest_words[index])) {
            return false;
        }
    }
    std::memcpy(digest, digest_words, sizeof(digest));

    TopologyDescriptor candidate;
    candidate.dtype_id = dtype_id;
    candidate.num_inputs = num_inputs;
    candidate.num_labels = num_labels;
    candidate.key = topology_key_from_digest(digest);
    candidate.sizes.assign(static_cast<std::size_t>(num_labels), -1);
    candidate.input_labels.resize(static_cast<std::size_t>(num_inputs));
    candidate.input_shapes.resize(static_cast<std::size_t>(num_inputs));

    for (int32_t input = 0; input < num_inputs; ++input) {
        int32_t rank = 0;
        if (!reader.read_i32(&rank) || rank <= 0 ||
            rank > static_cast<int32_t>(QCD_CUDA_MAX_RANK)) {
            return false;
        }
        std::vector<int32_t> labels;
        labels.reserve(static_cast<std::size_t>(rank));
        for (int32_t axis = 0; axis < rank; ++axis) {
            int32_t label = 0;
            if (!reader.read_i32(&label) || label < 0 || label >= num_labels ||
                contains(labels, label)) {
                return false;
            }
            labels.push_back(label);
        }
        std::vector<int64_t> shape(static_cast<std::size_t>(rank), 0);
        for (int32_t axis = 0; axis < rank; ++axis) {
            int32_t extent = 0;
            if (!reader.read_i32(&extent) || extent <= 0) {
                return false;
            }
            shape[static_cast<std::size_t>(axis)] = extent;
            const int32_t label = labels[static_cast<std::size_t>(axis)];
            int64_t& size = candidate.sizes[static_cast<std::size_t>(label)];
            /* 同一标签在多个操作数里出现：允许 1 对 N 广播，其余一律拒绝。 */
            if (size < 0) {
                size = extent;
            } else if (size == extent) {
                continue;
            } else if (extent == 1) {
                continue;
            } else if (size == 1) {
                size = extent;
            } else {
                return false;
            }
        }
        candidate.input_labels[static_cast<std::size_t>(input)] = std::move(labels);
        candidate.input_shapes[static_cast<std::size_t>(input)] = std::move(shape);
    }

    std::vector<int32_t> output;
    output.reserve(static_cast<std::size_t>(num_output));
    for (int32_t axis = 0; axis < num_output; ++axis) {
        int32_t label = 0;
        if (!reader.read_i32(&label) || label < 0 || label >= num_labels ||
            contains(output, label)) {
            return false;
        }
        output.push_back(label);
    }
    candidate.output_labels = std::move(output);
    if (!reader.at_end()) {
        return false;
    }
    for (int64_t size : candidate.sizes) {
        if (size <= 0) {
            return false;
        }
    }
    *descriptor = std::move(candidate);
    return true;
}

struct LiveOperand {
    std::vector<int32_t> labels;
};

bool topology_step_output_labels(
    const std::vector<int32_t>& left,
    const std::vector<int32_t>& right,
    const std::vector<LiveOperand>& live,
    std::size_t left_position,
    std::size_t right_position,
    const std::vector<int32_t>& final_labels,
    std::vector<int32_t>* output) {
    output->clear();
    const auto survives = [&live, &final_labels, left_position, right_position](int32_t label) {
        if (contains(final_labels, label)) {
            return true;
        }
        for (std::size_t index = 0; index < live.size(); ++index) {
            if (index == left_position || index == right_position) {
                continue;
            }
            if (contains(live[index].labels, label)) {
                return true;
            }
        }
        return false;
    };
    /* 规范序：批次标（两侧共享）→ 左自由标 → 右自由标，与 `prepare_binary`
     * 构造 pack 序的顺序一致，后续步骤的 pack 更容易判定为恒等。 */
    for (int32_t label : left) {
        if (contains(right, label) && survives(label) && !contains(*output, label)) {
            output->push_back(label);
        }
    }
    for (int32_t label : left) {
        if (!contains(right, label) && survives(label) && !contains(*output, label)) {
            output->push_back(label);
        }
    }
    for (int32_t label : right) {
        if (!contains(left, label) && survives(label) && !contains(*output, label)) {
            output->push_back(label);
        }
    }
    return true;
}

bool topology_plan_from_table(
    const TopologyDescriptor& descriptor,
    const qcd_topology_order& order,
    pyqcd_einsum::ExecutionPlan* plan) {
    if (order.pairs == nullptr || order.steps <= 0 ||
        order.steps != descriptor.num_inputs - 1) {
        return false;
    }
    pyqcd_einsum::ExecutionPlan candidate;
    candidate.dtype_id = descriptor.dtype_id;
    candidate.num_inputs = descriptor.num_inputs;
    candidate.output_ndim = static_cast<int32_t>(descriptor.output_labels.size());
    candidate.sizes = descriptor.sizes;
    candidate.input_shapes = descriptor.input_shapes;
    candidate.output_labels = descriptor.output_labels;

    std::vector<LiveOperand> live(static_cast<std::size_t>(descriptor.num_inputs));
    for (int32_t input = 0; input < descriptor.num_inputs; ++input) {
        live[static_cast<std::size_t>(input)].labels =
            descriptor.input_labels[static_cast<std::size_t>(input)];
    }

    const auto shape_of = [&descriptor](const std::vector<int32_t>& labels) {
        std::vector<int64_t> shape(labels.size(), 0);
        for (std::size_t axis = 0; axis < labels.size(); ++axis) {
            shape[axis] = descriptor.sizes[static_cast<std::size_t>(labels[axis])];
        }
        return shape;
    };

    candidate.steps.reserve(static_cast<std::size_t>(order.steps));
    for (int32_t step_index = 0; step_index < order.steps; ++step_index) {
        const int32_t left_position = order.pairs[2 * step_index];
        const int32_t right_position = order.pairs[2 * step_index + 1];
        const std::size_t live_size = live.size();
        if (left_position < 0 || right_position < 0 ||
            static_cast<std::size_t>(left_position) >= live_size ||
            static_cast<std::size_t>(right_position) >= live_size ||
            left_position == right_position) {
            return false;
        }
        /* 位置对按升序记账：先擦大的，再擦小的（步表位置指的是紧凑活表）。 */
        const std::size_t first = static_cast<std::size_t>(
            std::min(left_position, right_position));
        const std::size_t second = static_cast<std::size_t>(
            std::max(left_position, right_position));

        pyqcd_einsum::ExecutionStep step;
        step.positions = {static_cast<int32_t>(first), static_cast<int32_t>(second)};
        const std::vector<int32_t>& first_labels = live[first].labels;
        const std::vector<int32_t>& second_labels = live[second].labels;
        const bool last_step = step_index + 1 == order.steps;
        std::vector<int32_t> output_labels;
        if (last_step) {
            output_labels = descriptor.output_labels;
        } else if (!topology_step_output_labels(
                       first_labels, second_labels, live, first, second,
                       descriptor.output_labels, &output_labels)) {
            return false;
        }
        step.inputs.resize(2);
        step.inputs[0].labels = first_labels;
        step.inputs[0].shape = shape_of(first_labels);
        step.inputs[1].labels = second_labels;
        step.inputs[1].shape = shape_of(second_labels);
        step.output.labels = output_labels;
        step.output.shape = shape_of(output_labels);
        candidate.steps.push_back(std::move(step));

        int64_t elements = 0;
        if (!checked_product(step.output.shape, &elements) ||
            elements > kTopologyMaxIntermediateElements) {
            return false;
        }

        LiveOperand merged;
        merged.labels = candidate.steps.back().output.labels;
        live.erase(live.begin() + static_cast<std::ptrdiff_t>(second));
        live.erase(live.begin() + static_cast<std::ptrdiff_t>(first));
        live.push_back(std::move(merged));
    }
    if (candidate.steps.size() != static_cast<std::size_t>(order.steps) ||
        candidate.steps.back().output.labels != descriptor.output_labels) {
        return false;
    }
    *plan = std::move(candidate);
    return true;
}

/* 不在语料步表里的拓扑只接受极小的链式收缩（左到右），其余一律 UNSUPPORTED
 * 让上层回落到带真实 path 搜索的通用路径。 */
bool topology_plan_chain(
    const TopologyDescriptor& descriptor, pyqcd_einsum::ExecutionPlan* plan) {
    if (descriptor.num_inputs < 1 || descriptor.num_inputs > 6) {
        return false;
    }
    if (descriptor.num_inputs == 1) {
        pyqcd_einsum::ExecutionPlan candidate;
        candidate.dtype_id = descriptor.dtype_id;
        candidate.num_inputs = 1;
        candidate.output_ndim = static_cast<int32_t>(descriptor.output_labels.size());
        candidate.sizes = descriptor.sizes;
        candidate.input_shapes = descriptor.input_shapes;
        candidate.output_labels = descriptor.output_labels;
        pyqcd_einsum::ExecutionStep step;
        step.positions = {0};
        step.inputs.resize(1);
        step.inputs[0].labels = descriptor.input_labels[0];
        step.inputs[0].shape = descriptor.input_shapes[0];
        step.output.labels = descriptor.output_labels;
        step.output.shape.resize(descriptor.output_labels.size(), 0);
        for (std::size_t axis = 0; axis < descriptor.output_labels.size(); ++axis) {
            step.output.shape[axis] =
                descriptor.sizes[static_cast<std::size_t>(descriptor.output_labels[axis])];
        }
        candidate.steps.push_back(std::move(step));
        *plan = std::move(candidate);
        return true;
    }
    std::vector<int8_t> pairs;
    pairs.reserve(static_cast<std::size_t>(2 * (descriptor.num_inputs - 1)));
    for (int32_t step = 0; step + 1 < descriptor.num_inputs; ++step) {
        pairs.push_back(0);
        pairs.push_back(1);
    }
    const qcd_topology_order order{pairs.data(),
                                   descriptor.num_inputs - 1};
    return topology_plan_from_table(descriptor, order, plan);
}

bool build_topology_plan(
    const TopologyDescriptor& descriptor, pyqcd_einsum::ExecutionPlan* plan) {
    const uint64_t fingerprint =
        topology_structure_fingerprint(descriptor.input_labels, descriptor.output_labels);
    const qcd_topology_entry* entry =
        find_topology_entry(descriptor.key, fingerprint);
    if (entry == nullptr) {
        return topology_plan_chain(descriptor, plan);
    }
    if (entry->order >= static_cast<uint16_t>(sizeof(kQcdTopoOrders) /
                                              sizeof(kQcdTopoOrders[0]))) {
        return false;
    }
    return topology_plan_from_table(
        descriptor, kQcdTopoOrders[entry->order], plan);
}

struct SandwichPlan {
    bool matched = false;
    int32_t tensor_input = -1;
    int32_t left_matrix_input = -1;
    int32_t right_matrix_input = -1;
    qcd_cuda_sandwich_spec spec{};
};

bool all_distinct(const std::vector<int32_t>& values) {
    for (std::size_t left = 0; left < values.size(); ++left) {
        for (std::size_t right = left + 1; right < values.size(); ++right) {
            if (values[left] == values[right]) {
                return false;
            }
        }
    }
    return true;
}

bool same_label_set(
    const std::vector<int32_t>& left,
    const std::vector<int32_t>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (int32_t label : left) {
        if (!contains(right, label)) {
            return false;
        }
    }
    return true;
}

bool sandwich_dimensions(
    const pyqcd_einsum::ExecutionPlan& plan,
    int32_t a,
    int32_t b,
    int32_t c,
    int32_t d,
    int32_t e,
    const std::vector<int32_t>& prefix_labels,
    qcd_cuda_sandwich_spec* spec) {
    std::vector<int32_t> role_labels = prefix_labels;
    role_labels.insert(role_labels.end(), {a, b, c, d, e});
    if (!all_distinct(role_labels)) {
        return false;
    }
    int64_t prefix = 1;
    for (int32_t label : prefix_labels) {
        if (label < 0 || static_cast<std::size_t>(label) >= plan.sizes.size() ||
            !multiply_checked(prefix, plan.sizes[static_cast<std::size_t>(label)],
                              &prefix)) {
            return false;
        }
    }
    for (int32_t label : {a, b, c, d, e}) {
        if (label < 0 || static_cast<std::size_t>(label) >= plan.sizes.size() ||
            plan.sizes[static_cast<std::size_t>(label)] <= 0) {
            return false;
        }
    }
    qcd_cuda_sandwich_spec candidate{};
    candidate.prefix = prefix;
    candidate.left = plan.sizes[static_cast<std::size_t>(a)];
    candidate.contracted_left = plan.sizes[static_cast<std::size_t>(b)];
    candidate.contracted_right = plan.sizes[static_cast<std::size_t>(c)];
    candidate.right = plan.sizes[static_cast<std::size_t>(d)];
    candidate.lane = plan.sizes[static_cast<std::size_t>(e)];
    int64_t elements = 1;
    int64_t temporary = 1;
    if (!multiply_checked(candidate.prefix, candidate.contracted_left, &temporary) ||
        !multiply_checked(temporary, candidate.contracted_right, &elements) ||
        !multiply_checked(elements, candidate.lane, &temporary) ||
        !multiply_checked(candidate.left, candidate.contracted_left, &elements) ||
        !multiply_checked(candidate.contracted_right, candidate.right, &elements) ||
        !multiply_checked(candidate.prefix, candidate.right, &elements) ||
        !multiply_checked(elements, candidate.left, &temporary) ||
        !multiply_checked(temporary, candidate.lane, &elements)) {
        return false;
    }
    *spec = candidate;
    return true;
}

bool match_sandwich(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<ResolvedStep>& resolved,
    SandwichPlan* sandwich) {
    *sandwich = SandwichPlan{};
    if (plan.num_inputs != 3 || plan.steps.size() != 2 || resolved.size() != 2) {
        return false;
    }
    const pyqcd_einsum::ExecutionStep& first = plan.steps[0];
    const pyqcd_einsum::ExecutionStep& second = plan.steps[1];
    if (first.inputs.size() != 2 || second.inputs.size() != 2) {
        return false;
    }

    for (int tensor_operand = 0; tensor_operand < 2; ++tensor_operand) {
        const int matrix_operand = 1 - tensor_operand;
        const int32_t tensor_input = resolved[0].operands[tensor_operand];
        const int32_t first_matrix_input = resolved[0].operands[matrix_operand];
        if (tensor_input < 0 || tensor_input >= plan.num_inputs ||
            first_matrix_input < 0 || first_matrix_input >= plan.num_inputs ||
            tensor_input == first_matrix_input) {
            continue;
        }
        int32_t remaining_matrix_input = -1;
        for (int32_t input = 0; input < plan.num_inputs; ++input) {
            if (input != tensor_input && input != first_matrix_input) {
                remaining_matrix_input = input;
                break;
            }
        }
        if (remaining_matrix_input < 0) {
            continue;
        }

        const std::vector<int32_t>& tensor_labels =
            first.inputs[tensor_operand].labels;
        const std::vector<int32_t>& first_matrix_labels =
            first.inputs[matrix_operand].labels;
        if (tensor_labels.size() < 3 || first_matrix_labels.size() != 2) {
            continue;
        }
        const std::size_t prefix_rank = tensor_labels.size() - 3;
        const int32_t b = tensor_labels[prefix_rank];
        const int32_t c = tensor_labels[prefix_rank + 1];
        const int32_t e = tensor_labels[prefix_rank + 2];
        std::vector<int32_t> prefix_labels(tensor_labels.begin(),
                                           tensor_labels.begin() +
                                               static_cast<std::ptrdiff_t>(prefix_rank));

        bool left_matrix_first = false;
        bool right_matrix_first = false;
        int32_t a = -1;
        int32_t d = -1;
        std::vector<int32_t> first_output;
        std::vector<int32_t> other_matrix_labels;
        if (first_matrix_labels[1] == b) {
            a = first_matrix_labels[0];
            left_matrix_first = true;
            first_output = prefix_labels;
            first_output.insert(first_output.end(), {c, e, a});
        } else if (first_matrix_labels[0] == c) {
            d = first_matrix_labels[1];
            right_matrix_first = true;
            first_output = prefix_labels;
            first_output.insert(first_output.end(), {d, b, e});
        } else {
            continue;
        }
        if (!same_label_set(first.output.labels, first_output)) {
            continue;
        }

        int intermediate_operand = -1;
        int remaining_matrix_operand = -1;
        for (int operand = 0; operand < 2; ++operand) {
            const int32_t node = resolved[1].operands[operand];
            if (node == plan.num_inputs) {
                intermediate_operand = operand;
            } else if (node == remaining_matrix_input) {
                remaining_matrix_operand = operand;
            }
        }
        if (intermediate_operand < 0 || remaining_matrix_operand < 0 ||
            intermediate_operand == remaining_matrix_operand) {
            continue;
        }
        other_matrix_labels =
            second.inputs[remaining_matrix_operand].labels;
        if (other_matrix_labels.size() != 2 ||
            second.inputs[intermediate_operand].labels != first.output.labels ||
            (left_matrix_first && other_matrix_labels[0] != c) ||
            (right_matrix_first && other_matrix_labels[1] != b)) {
            continue;
        }
        if (left_matrix_first) {
            d = other_matrix_labels[1];
        } else {
            a = other_matrix_labels[0];
        }

        std::vector<int32_t> expected_output = prefix_labels;
        expected_output.insert(expected_output.end(), {d, a, e});
        if (second.output.labels != expected_output ||
            plan.output_labels != expected_output) {
            continue;
        }

        SandwichPlan candidate;
        if (!sandwich_dimensions(plan, a, b, c, d, e, prefix_labels,
                                 &candidate.spec)) {
            continue;
        }
        candidate.matched = true;
        candidate.tensor_input = tensor_input;
        candidate.left_matrix_input =
            left_matrix_first ? first_matrix_input : remaining_matrix_input;
        candidate.right_matrix_input =
            left_matrix_first ? remaining_matrix_input : first_matrix_input;
        *sandwich = candidate;
        return true;
    }
    return false;
}

struct E4FusedPlan {
    bool matched = false;
    qcd_cuda_e4_spec spec{};
};

bool exact_input_shape(
    const std::vector<int32_t>& labels,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& sizes) {
    if (labels.size() != shape.size()) {
        return false;
    }
    for (std::size_t axis = 0; axis < labels.size(); ++axis) {
        const int32_t label = labels[axis];
        if (label < 0 || static_cast<std::size_t>(label) >= sizes.size() ||
            shape[axis] != sizes[static_cast<std::size_t>(label)]) {
            return false;
        }
    }
    return true;
}

bool recover_input_terms(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<ResolvedStep>& resolved,
    std::vector<std::vector<int32_t>>* terms) {
    if (resolved.size() != plan.steps.size()) {
        return false;
    }
    terms->assign(static_cast<std::size_t>(plan.num_inputs), {});
    std::vector<int32_t> uses(static_cast<std::size_t>(plan.num_inputs), 0);
    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        const pyqcd_einsum::ExecutionStep& step = plan.steps[step_index];
        if (step.inputs.size() != resolved[step_index].operands.size()) {
            return false;
        }
        for (std::size_t operand = 0; operand < step.inputs.size(); ++operand) {
            const int32_t node = resolved[step_index].operands[operand];
            if (node < 0) {
                return false;
            }
            if (node >= plan.num_inputs) {
                continue;
            }
            const std::vector<int32_t>& labels = step.inputs[operand].labels;
            if (uses[static_cast<std::size_t>(node)] != 0 &&
                (*terms)[static_cast<std::size_t>(node)] != labels) {
                return false;
            }
            (*terms)[static_cast<std::size_t>(node)] = labels;
            ++uses[static_cast<std::size_t>(node)];
        }
    }
    return std::all_of(uses.begin(), uses.end(), [](int32_t count) {
        return count == 1;
    });
}

constexpr int64_t kE4FusedMaxSpatialVolume = 1;

bool match_e4_fused(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<ResolvedStep>& resolved,
    E4FusedPlan* fused) {
    *fused = E4FusedPlan{};
    if (plan.num_inputs != 4 || plan.steps.size() != 3 || resolved.size() != 3) {
        return false;
    }
    for (std::size_t step_index = 0; step_index < plan.steps.size(); ++step_index) {
        if (plan.steps[step_index].inputs.size() != 2 ||
            resolved[step_index].operands.size() != 2) {
            return false;
        }
    }

    std::vector<std::vector<int32_t>> terms;
    if (!recover_input_terms(plan, resolved, &terms)) {
        return false;
    }
    const std::vector<int32_t>& a_labels = terms[0];
    const std::vector<int32_t>& b_labels = terms[1];
    const std::vector<int32_t>& gamma_labels = terms[2];
    const std::vector<int32_t>& c_labels = terms[3];
    if (a_labels.size() != 8 || b_labels.size() != 8 ||
        gamma_labels.size() != 3 || c_labels.size() != 6) {
        return false;
    }

    const int32_t z = a_labels[0];
    const int32_t y = a_labels[1];
    const int32_t x = a_labels[2];
    const int32_t c = a_labels[3];
    const int32_t t = a_labels[4];
    const int32_t a = a_labels[5];
    const int32_t b = a_labels[6];
    const int32_t n = a_labels[7];
    const int32_t input_g = b_labels[3];
    const int32_t d = b_labels[5];
    const int32_t e = b_labels[6];
    const int32_t f = b_labels[7];
    const int32_t output_g = gamma_labels[0];

    if (b_labels != std::vector<int32_t>{z, y, x, input_g, t, d, e, f} ||
        gamma_labels != std::vector<int32_t>{output_g, a, e} ||
        c_labels != std::vector<int32_t>{t, z, y, x, input_g, c}) {
        return false;
    }
    const std::vector<int32_t> output_labels{t, output_g, d, b, f, n};
    if (plan.output_labels != output_labels ||
        !all_distinct({z, y, x, c, t, a, b, n, input_g, output_g, d, e, f})) {
        return false;
    }
    if (!exact_input_shape(a_labels, plan.input_shapes[0], plan.sizes) ||
        !exact_input_shape(b_labels, plan.input_shapes[1], plan.sizes) ||
        !exact_input_shape(gamma_labels, plan.input_shapes[2], plan.sizes) ||
        !exact_input_shape(c_labels, plan.input_shapes[3], plan.sizes)) {
        return false;
    }

    E4FusedPlan candidate;
    candidate.spec.z = plan.sizes[static_cast<std::size_t>(z)];
    candidate.spec.y = plan.sizes[static_cast<std::size_t>(y)];
    candidate.spec.x = plan.sizes[static_cast<std::size_t>(x)];
    candidate.spec.c = plan.sizes[static_cast<std::size_t>(c)];
    candidate.spec.t = plan.sizes[static_cast<std::size_t>(t)];
    candidate.spec.a = plan.sizes[static_cast<std::size_t>(a)];
    candidate.spec.b = plan.sizes[static_cast<std::size_t>(b)];
    candidate.spec.n = plan.sizes[static_cast<std::size_t>(n)];
    candidate.spec.input_g = plan.sizes[static_cast<std::size_t>(input_g)];
    candidate.spec.output_g = plan.sizes[static_cast<std::size_t>(output_g)];
    candidate.spec.d = plan.sizes[static_cast<std::size_t>(d)];
    candidate.spec.e = plan.sizes[static_cast<std::size_t>(e)];
    candidate.spec.f = plan.sizes[static_cast<std::size_t>(f)];
    const std::vector<int64_t> output_shape{
        candidate.spec.t, candidate.spec.output_g, candidate.spec.d,
        candidate.spec.b, candidate.spec.f, candidate.spec.n};
    if (!checked_product(output_shape, &candidate.spec.output_elements)) {
        return false;
    }
    // 收益判据（实测，2026-11-08）：staged 融合 kernel 只在空间体积为 1 时占优
    // （z*y*x=1 → 快的 3.6x；z*y*x=8/27/64/512 → 慢 1.4x/9.7x/22x/26x；
    //  L24 的 z*y*x=13824 → 慢 33x）。超出阈值即回落通用多步，避免灾难性退化。
    if (candidate.spec.z * candidate.spec.y * candidate.spec.x >
        kE4FusedMaxSpatialVolume) {
        return false;
    }
    candidate.matched = true;
    *fused = candidate;
    return true;
}

int map_cuda_error(cudaError_t error) {
    if (error == cudaSuccess) {
        return QCD_STATUS_OK;
    }
    return error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

int map_cublas_status(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return QCD_STATUS_OK;
    }
    return status == CUBLAS_STATUS_ALLOC_FAILED ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

constexpr uint64_t kCompiledContextMagic = 0x514345434f4e5458ULL;
constexpr int32_t kMaxCudaStreams = 32;
std::shared_mutex context_lifecycle_mutex;

// TODO: streams are scoped to one CUDA device; multi-device scheduling is not implemented.
bool trace_enabled() {
    static const bool enabled = std::getenv("PYQCD_EINSUM_TRACE") != nullptr;
    return enabled;
}

int requested_stream_count() {
    const char* value = std::getenv("PYQCD_EINSUM_CUDA_STREAMS");
    if (value == nullptr || *value == '\0') {
        return 1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        return 1;
    }
    return static_cast<int>(std::min<long>(parsed, kMaxCudaStreams));
}

bool needs_gemm(const std::vector<PreparedStep>& steps) {
    for (const PreparedStep& step : steps) {
        if (step.kind == OperationKind::kGemm) {
            return true;
        }
    }
    return false;
}

void* offset_pointer(void* base, int64_t elements, int dtype_id) {
    const int64_t size = dtype_size(dtype_id);
    return static_cast<uint8_t*>(base) + elements * size;
}

constexpr uint64_t kLayoutFnvOffset = 0xCBF29CE484222325ULL;
constexpr uint64_t kLayoutFnvPrime = 0x100000001B3ULL;

/* 步骤布局指纹：只覆盖决定 arena 尺寸与偏移的字段。布局相同即可直接复用上一次
 * 的 arena 与偏移表；布局不同但容量够时只重算偏移，不重新分配。 */
uint64_t layout_signature_of(const std::vector<PreparedStep>& steps) {
    uint64_t hash = kLayoutFnvOffset;
    const auto mix = [&hash](uint64_t value) {
        hash = (hash ^ value) * kLayoutFnvPrime;
    };
    mix(steps.size());
    for (const PreparedStep& step : steps) {
        mix(static_cast<uint64_t>(step.kind));
        mix(static_cast<uint64_t>(step.output_elements));
        mix(static_cast<uint64_t>(step.m));
        mix(static_cast<uint64_t>(step.n));
        mix(static_cast<uint64_t>(step.k));
        mix(static_cast<uint64_t>(step.batch));
        mix(step.permute_output ? 1u : 2u);
        mix(step.left_pack_identity ? 3u : 4u);
        mix(step.right_pack_identity ? 5u : 6u);
    }
    return hash;
}

class CudaWorkspace {
  public:
    CudaWorkspace() = default;
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

    ~CudaWorkspace() { release(); }

    /* 换一条拓扑只换布局，不换池子：arena 仅在容量不够时才重新 cudaMalloc。
     * `layout_signature_` 是步骤布局的指纹（步种类 + 各步 m/n/k/batch + 中间体
     * 元素数 + pack 恒等标志），布局变了就必须重算偏移；容量不足才扩容。
     * 这样同一进程里连续跑多条拓扑不再反复 free/alloc，也不再重建 cuBLAS 句柄。 */
    int prepare(const std::vector<PreparedStep>& prepared, int dtype_id) {
        int device = -1;
        int status = map_cuda_error(cudaGetDevice(&device));
        if (status != QCD_STATUS_OK) {
            return status;
        }
        const uint64_t signature = layout_signature_of(prepared);
        const bool reusable = initialized_ && !sandwich_storage_ && !e4_storage_ &&
                              device_ == device && dtype_id_ == dtype_id &&
                              capacity_elements_ > 0;
        if (reusable && signature == layout_signature_) {
            return QCD_STATUS_OK;
        }

        std::vector<int64_t> output_offsets(prepared.size(), 0);
        int64_t output_elements = 0;
        int64_t max_left_pack = 0;
        int64_t max_right_pack = 0;
        int64_t max_gemm_output = 0;
        int64_t max_batch = 1;
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            const PreparedStep& step = prepared[index];
            output_offsets[index] = output_elements;
            if (!add_checked(output_elements, step.output_elements, &output_elements)) {
                return QCD_STATUS_INVALID;
            }
            if (step.kind != OperationKind::kGemm) {
                continue;
            }
            int64_t elements = 0;
            int64_t temporary = 0;
            if (!multiply_checked(step.batch, step.m, &temporary) ||
                !multiply_checked(temporary, step.k, &elements)) {
                return QCD_STATUS_INVALID;
            }
            max_left_pack = std::max(max_left_pack, elements);
            if (!multiply_checked(step.batch, step.n, &temporary) ||
                !multiply_checked(temporary, step.k, &elements)) {
                return QCD_STATUS_INVALID;
            }
            max_right_pack = std::max(max_right_pack, elements);
            if (!multiply_checked(step.batch, step.m, &temporary) ||
                !multiply_checked(temporary, step.n, &elements)) {
                return QCD_STATUS_INVALID;
            }
            max_gemm_output = std::max(max_gemm_output, elements);
            const int64_t parallel_units = step.batch > 1 ? step.batch : step.m;
            max_batch = std::max(max_batch, parallel_units);
        }

        int64_t total_elements = output_elements;
        if (!add_checked(total_elements, max_left_pack, &total_elements) ||
            !add_checked(total_elements, max_right_pack, &total_elements) ||
            !add_checked(total_elements, max_gemm_output, &total_elements)) {
            return QCD_STATUS_INVALID;
        }

        const int wanted_streams = std::max(
            1, std::min(requested_stream_count(),
                        static_cast<int>(std::min<int64_t>(
                            max_batch, std::numeric_limits<int>::max()))));
        const bool reuse_arena = reusable && total_elements <= capacity_elements_;

        output_offsets_ = std::move(output_offsets);
        output_elements_ = output_elements;
        max_left_pack_ = max_left_pack;
        max_right_pack_ = max_right_pack;
        max_gemm_output_ = max_gemm_output;
        max_batch_ = max_batch;
        dtype_id_ = dtype_id;
        device_ = device;
        sandwich_storage_ = false;
        layout_signature_ = signature;

        if (reuse_arena) {
            storage_elements_ = capacity_elements_;
            status = ensure_workers(wanted_streams, needs_gemm(prepared));
            if (status != QCD_STATUS_OK) {
                release();
                return status;
            }
            initialized_ = true;
            if (trace_enabled()) {
                std::fprintf(
                    stderr,
                    "PYQCD_EINSUM_CUDA_WORKSPACE elements=%lld streams=%d "
                    "mallocs=0 reused=1 needed=%lld\n",
                    static_cast<long long>(storage_elements_), stream_count_,
                    static_cast<long long>(total_elements));
            }
            return QCD_STATUS_OK;
        }

        release();
        sandwich_storage_ = false;
        dtype_id_ = dtype_id;
        device_ = device;
        output_offsets_.resize(prepared.size(), 0);
        int64_t running = 0;
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            output_offsets_[index] = running;
            running += prepared[index].output_elements;
        }
        output_elements_ = output_elements;
        max_left_pack_ = max_left_pack;
        max_right_pack_ = max_right_pack;
        max_gemm_output_ = max_gemm_output;
        max_batch_ = max_batch;
        layout_signature_ = signature;
        storage_elements_ = total_elements;
        capacity_elements_ = total_elements;
        std::size_t bytes = 0;
        if (!allocation_bytes(storage_elements_, dtype_id_, &bytes)) {
            release();
            return QCD_STATUS_INVALID;
        }
        status = map_cuda_error(cudaMalloc(&storage_, bytes));
        if (status != QCD_STATUS_OK) {
            release();
            return status;
        }
        status = ensure_workers(wanted_streams, needs_gemm(prepared));
        if (status != QCD_STATUS_OK) {
            release();
            return status;
        }
        initialized_ = true;
        if (trace_enabled()) {
            std::fprintf(
                stderr,
                "PYQCD_EINSUM_CUDA_WORKSPACE elements=%lld streams=%d mallocs=1\n",
                static_cast<long long>(storage_elements_), stream_count_);
        }
        return QCD_STATUS_OK;
    }

    int prepare_sandwich(int64_t output_elements, int dtype_id) {
        if (output_elements <= 0) {
            return QCD_STATUS_INVALID;
        }
        int device = -1;
        int status = map_cuda_error(cudaGetDevice(&device));
        if (status != QCD_STATUS_OK) {
            return status;
        }
        if (initialized_ && sandwich_storage_ && device_ == device &&
            dtype_id_ == dtype_id && storage_elements_ == output_elements) {
            return QCD_STATUS_OK;
        }

        release();
        dtype_id_ = dtype_id;
        device_ = device;
        sandwich_storage_ = true;
        storage_elements_ = output_elements;
        capacity_elements_ = output_elements;
        output_elements_ = 0;
        output_offsets_.assign(1, 0);
        std::size_t bytes = 0;
        if (!allocation_bytes(storage_elements_, dtype_id_, &bytes)) {
            release();
            return QCD_STATUS_INVALID;
        }
        status = map_cuda_error(cudaMalloc(&storage_, bytes));
        if (status != QCD_STATUS_OK) {
            release();
            return status;
        }
        initialized_ = true;
        if (trace_enabled()) {
            std::fprintf(
                stderr,
                "PYQCD_EINSUM_CUDA_WORKSPACE mode=sandwich elements=%lld "
                "streams=1 mallocs=1\n",
                static_cast<long long>(storage_elements_));
        }
        return QCD_STATUS_OK;
    }

    int prepare_e4(int dtype_id) {
        int device = -1;
        int status = map_cuda_error(cudaGetDevice(&device));
        if (status != QCD_STATUS_OK) {
            return status;
        }
        if (initialized_ && e4_storage_ && device_ == device &&
            dtype_id_ == dtype_id) {
            return QCD_STATUS_OK;
        }

        release();
        dtype_id_ = dtype_id;
        device_ = device;
        e4_storage_ = true;
        capacity_elements_ = 0;
        initialized_ = true;
        if (trace_enabled()) {
            std::fprintf(
                stderr,
                "PYQCD_EINSUM_CUDA_WORKSPACE mode=e4 elements=0 "
                "streams=1 mallocs=0\n");
        }
        return QCD_STATUS_OK;
    }

    void* output(std::size_t index) {
        return offset_pointer(storage_, output_offsets_[index], dtype_id_);
    }

    void* left_pack() {
        return offset_pointer(storage_, output_elements_, dtype_id_);
    }

    void* right_pack() {
        return offset_pointer(
            storage_, output_elements_ + max_left_pack_, dtype_id_);
    }

    void* gemm_output() {
        return offset_pointer(
            storage_, output_elements_ + max_left_pack_ + max_right_pack_, dtype_id_);
    }

    int workers(int64_t batch) const {
        return std::max(1, std::min(stream_count_, static_cast<int>(batch)));
    }

    cudaStream_t stream(int worker, int workers) const {
        return workers > 1 ? streams_[static_cast<std::size_t>(worker)] : nullptr;
    }

    cublasHandle_t handle(int worker) const {
        return handles_[static_cast<std::size_t>(worker)];
    }

    int begin_batch_split(int workers) {
        if (workers <= 1) {
            return QCD_STATUS_OK;
        }
        const cudaError_t record_error = cudaEventRecord(pack_event_, nullptr);
        if (record_error != cudaSuccess) {
            return map_cuda_error(record_error);
        }
        for (int worker = 0; worker < workers; ++worker) {
            const cudaError_t wait_error = cudaStreamWaitEvent(
                streams_[static_cast<std::size_t>(worker)], pack_event_, 0);
            if (wait_error != cudaSuccess) {
                return map_cuda_error(wait_error);
            }
        }
        return QCD_STATUS_OK;
    }

    int end_batch_split(int workers) {
        if (workers <= 1) {
            return QCD_STATUS_OK;
        }
        for (int worker = 0; worker < workers; ++worker) {
            const cudaError_t record_error = cudaEventRecord(
                done_events_[static_cast<std::size_t>(worker)],
                streams_[static_cast<std::size_t>(worker)]);
            if (record_error != cudaSuccess) {
                return map_cuda_error(record_error);
            }
        }
        for (int worker = 0; worker < workers; ++worker) {
            const cudaError_t wait_error = cudaStreamWaitEvent(
                nullptr, done_events_[static_cast<std::size_t>(worker)], 0);
            if (wait_error != cudaSuccess) {
                return map_cuda_error(wait_error);
            }
        }
        return QCD_STATUS_OK;
    }

    /* 让流/句柄池至少容纳 wanted 个 worker；只增不减，复用时不重建。 */
    int ensure_workers(int wanted, bool with_gemm) {
        if (wanted < 1) {
            wanted = 1;
        }
        const std::size_t size = static_cast<std::size_t>(wanted);
        if (streams_.size() < size) {
            streams_.resize(size, nullptr);
        }
        if (handles_.size() < size) {
            handles_.resize(size, nullptr);
        }
        if (done_events_.size() < size) {
            done_events_.resize(size, nullptr);
        }
        for (int index = 0; index < wanted; ++index) {
            const std::size_t slot = static_cast<std::size_t>(index);
            if (wanted > 1 && streams_[slot] == nullptr) {
                const cudaError_t error = cudaStreamCreateWithFlags(
                    &streams_[slot], cudaStreamNonBlocking);
                if (error != cudaSuccess) {
                    return map_cuda_error(error);
                }
            }
            if (wanted > 1 && done_events_[slot] == nullptr) {
                const cudaError_t error = cudaEventCreateWithFlags(
                    &done_events_[slot], cudaEventDisableTiming);
                if (error != cudaSuccess) {
                    return map_cuda_error(error);
                }
            }
            if (with_gemm && handles_[slot] == nullptr) {
                cublasHandle_t handle = nullptr;
                int status = map_cublas_status(cublasCreate(&handle));
                if (status != QCD_STATUS_OK) {
                    return status;
                }
                status = map_cublas_status(
                    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
                if (status != QCD_STATUS_OK) {
                    cublasDestroy(handle);
                    return status;
                }
                handles_[slot] = handle;
            }
        }
        if (wanted > 1 && pack_event_ == nullptr) {
            const cudaError_t error = cudaEventCreateWithFlags(
                &pack_event_, cudaEventDisableTiming);
            if (error != cudaSuccess) {
                return map_cuda_error(error);
            }
        }
        stream_count_ = wanted;
        return QCD_STATUS_OK;
    }

    /* 计划换了：偏移表作废，但 arena 留着（下一次 prepare 按新布局决定复用或扩容）。 */
    void invalidate_layout() { layout_signature_ = 0; }

    void release() {
        if (device_ >= 0) {
            int current_device = -1;
            if (cudaGetDevice(&current_device) == cudaSuccess) {
                cudaSetDevice(device_);
            }
            cudaDeviceSynchronize();
            if (pack_event_ != nullptr) {
                cudaEventDestroy(pack_event_);
            }
            for (cudaEvent_t event : done_events_) {
                if (event != nullptr) {
                    cudaEventDestroy(event);
                }
            }
            for (cublasHandle_t handle : handles_) {
                if (handle != nullptr) {
                    cublasDestroy(handle);
                }
            }
            for (cudaStream_t stream : streams_) {
                if (stream != nullptr) {
                    cudaStreamDestroy(stream);
                }
            }
            if (storage_ != nullptr) {
                cudaFree(storage_);
            }
            if (current_device >= 0 && current_device != device_) {
                cudaSetDevice(current_device);
            }
        }
        storage_ = nullptr;
        storage_elements_ = 0;
        capacity_elements_ = 0;
        layout_signature_ = 0;
        output_elements_ = 0;
        max_left_pack_ = 0;
        max_right_pack_ = 0;
        max_gemm_output_ = 0;
        max_batch_ = 1;
        stream_count_ = 1;
        device_ = -1;
        dtype_id_ = -1;
        sandwich_storage_ = false;
        e4_storage_ = false;
        pack_event_ = nullptr;
        output_offsets_.clear();
        done_events_.clear();
        handles_.clear();
        streams_.clear();
        initialized_ = false;
    }

  private:
    static bool add_checked(int64_t left, int64_t right, int64_t* result) {
        if (left < 0 || right < 0 ||
            left > std::numeric_limits<int64_t>::max() - right) {
            return false;
        }
        *result = left + right;
        return true;
    }

    void* storage_ = nullptr;
    int64_t storage_elements_ = 0;
    /* arena 的实际容量（元素）；`storage_elements_` 只用于 trace。 */
    int64_t capacity_elements_ = 0;
    uint64_t layout_signature_ = 0;
    int64_t output_elements_ = 0;
    int64_t max_left_pack_ = 0;
    int64_t max_right_pack_ = 0;
    int64_t max_gemm_output_ = 0;
    int64_t max_batch_ = 1;
    int dtype_id_ = -1;
    int device_ = -1;
    int stream_count_ = 1;
    bool sandwich_storage_ = false;
    bool e4_storage_ = false;
    bool initialized_ = false;
    cudaEvent_t pack_event_ = nullptr;
    std::vector<int64_t> output_offsets_;
    std::vector<cudaStream_t> streams_;
    std::vector<cublasHandle_t> handles_;
    std::vector<cudaEvent_t> done_events_;
};

// ``PYQCD_EINSUM_TRACE=2`` adds per-step CUDA-event timing to the trace output.
bool step_timing_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("PYQCD_EINSUM_TRACE");
        if (value == nullptr) {
            return false;
        }
        return std::atoi(value) >= 2;
    }();
    return enabled;
}

const char* operation_kind_name(OperationKind kind) {
    switch (kind) {
        case OperationKind::kCopy:
            return "copy";
        case OperationKind::kReduce:
            return "reduce";
        case OperationKind::kHadamard:
            return "hadamard";
        case OperationKind::kGemm:
            return "gemm";
    }
    return "?";
}

int execute_prepared(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<PreparedStep>& prepared,
    CudaWorkspace* workspace,
    const std::vector<void*>& input_data,
    void* output_data) {
    if (workspace == nullptr || prepared.empty() || output_data == nullptr) {
        return QCD_STATUS_INVALID;
    }
    const int dtype_id = plan.dtype_id;
    int status = workspace->prepare(prepared, dtype_id);
    if (status != QCD_STATUS_OK) {
        return status;
    }
    const int32_t num_nodes = plan.num_inputs + static_cast<int32_t>(prepared.size());
    std::vector<void*> nodes(static_cast<std::size_t>(num_nodes), nullptr);
    for (int32_t input = 0; input < plan.num_inputs; ++input) {
        nodes[static_cast<std::size_t>(input)] = input_data[static_cast<std::size_t>(input)];
    }

    const bool step_timing = step_timing_enabled();
    const std::size_t timing_count = step_timing ? prepared.size() : 0;
    std::vector<cudaEvent_t> timing_start(timing_count, nullptr);
    std::vector<cudaEvent_t> timing_stop(timing_count, nullptr);
    std::vector<int> timing_skipped(timing_count, 0);  // reserved: pack elision off
    if (step_timing) {
        for (std::size_t index = 0; index < timing_count; ++index) {
            if (cudaEventCreate(&timing_start[index]) != cudaSuccess ||
                cudaEventCreate(&timing_stop[index]) != cudaSuccess) {
                for (cudaEvent_t event : timing_start) {
                    if (event != nullptr) {
                        cudaEventDestroy(event);
                    }
                }
                for (cudaEvent_t event : timing_stop) {
                    if (event != nullptr) {
                        cudaEventDestroy(event);
                    }
                }
                timing_start.clear();
                timing_stop.clear();
                return QCD_STATUS_INTERNAL;
            }
        }
    }
    const auto destroy_timing = [&timing_start, &timing_stop] {
        for (cudaEvent_t event : timing_start) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
        for (cudaEvent_t event : timing_stop) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
    };
    if (step_timing) {
        cudaEventRecord(timing_start[0], nullptr);
    }

    for (std::size_t step_index = 0; step_index < prepared.size(); ++step_index) {
        const PreparedStep& step = prepared[step_index];
        void* step_output = workspace->output(step_index);
        nodes[static_cast<std::size_t>(step.output)] = step_output;

        switch (step.kind) {
            case OperationKind::kCopy:
                if (qcd_cuda_permute == nullptr) {
                    return QCD_STATUS_INTERNAL;
                }
                status = qcd_cuda_permute(
                    dtype_id, nodes[static_cast<std::size_t>(step.left)], step_output,
                    step.copy, nullptr);
                break;
            case OperationKind::kReduce:
                if (qcd_cuda_reduce == nullptr) {
                    return QCD_STATUS_INTERNAL;
                }
                status = qcd_cuda_reduce(
                    dtype_id, nodes[static_cast<std::size_t>(step.left)], step_output,
                    step.reduce, nullptr);
                break;
            case OperationKind::kHadamard:
                if (qcd_cuda_hadamard == nullptr) {
                    return QCD_STATUS_INTERNAL;
                }
                status = qcd_cuda_hadamard(
                    dtype_id,
                    nodes[static_cast<std::size_t>(step.left)],
                    nodes[static_cast<std::size_t>(step.right)],
                    step_output,
                    step.hadamard,
                    nullptr);
                break;
            case OperationKind::kGemm: {
                if (qcd_cuda_permute == nullptr || qcd_cuda_batched_gemm == nullptr) {
                    return QCD_STATUS_INTERNAL;
                }
                void* left_pack = workspace->left_pack();
                void* right_pack = workspace->right_pack();
                /* pack 恒等就不下发拷贝 kernel：操作数已经在 [batch,free,reduce]
                 * 的连续布局上，直接把它当 GEMM 的 A/B 用。 */
                void* left_data = nodes[static_cast<std::size_t>(step.left)];
                if (step.left_pack_identity) {
                    left_pack = left_data;
                } else {
                    status = qcd_cuda_permute(
                        dtype_id, left_data, left_pack, step.left_pack, nullptr);
                }
                if (status != QCD_STATUS_OK) {
                    destroy_timing();
                    return status;
                }
                void* right_data = nodes[static_cast<std::size_t>(step.right)];
                if (step.right_pack_identity) {
                    right_pack = right_data;
                } else {
                    status = qcd_cuda_permute(
                        dtype_id, right_data, right_pack, step.right_pack, nullptr);
                }
                if (status != QCD_STATUS_OK) {
                    destroy_timing();
                    return status;
                }
                if (step_timing) {
                    timing_skipped[step_index] =
                        (step.left_pack_identity ? 1 : 0) + (step.right_pack_identity ? 1 : 0);
                }
                void* gemm_output = step.permute_output ? workspace->gemm_output() : step_output;
                const bool split_batch = step.batch > 1;
                const int64_t parallel_units = split_batch ? step.batch : step.m;
                const int workers = workspace->workers(parallel_units);
                status = workspace->begin_batch_split(workers);
                for (int worker = 0; worker < workers && status == QCD_STATUS_OK; ++worker) {
                    const int64_t begin = parallel_units * worker / workers;
                    const int64_t end = parallel_units * (worker + 1) / workers;
                    const int64_t count = end - begin;
                    int64_t worker_m = step.m;
                    int64_t worker_batch = step.batch;
                    void* worker_left = left_pack;
                    void* worker_right = right_pack;
                    void* worker_output = gemm_output;
                    if (split_batch) {
                        worker_batch = count;
                        worker_left = offset_pointer(
                            left_pack, begin * step.m * step.k, dtype_id);
                        worker_right = offset_pointer(
                            right_pack, begin * step.n * step.k, dtype_id);
                        worker_output = offset_pointer(
                            gemm_output, begin * step.m * step.n, dtype_id);
                    } else {
                        worker_m = count;
                        worker_left = offset_pointer(
                            left_pack, begin * step.k, dtype_id);
                        worker_output = offset_pointer(
                            gemm_output, begin * step.n, dtype_id);
                    }
                    status = qcd_cuda_batched_gemm(
                        dtype_id, workspace->handle(worker), worker_left, worker_right,
                        worker_output, worker_m, step.n, step.k, worker_batch,
                        workspace->stream(worker, workers));
                }
                if (status == QCD_STATUS_OK) {
                    status = workspace->end_batch_split(workers);
                }
                if (trace_enabled() && status == QCD_STATUS_OK) {
                    std::fprintf(
                        stderr,
                        "PYQCD_EINSUM_CUDA_GEMM batch=%lld m=%lld n=%lld k=%lld streams=%d\n",
                        static_cast<long long>(step.batch), static_cast<long long>(step.m),
                        static_cast<long long>(step.n), static_cast<long long>(step.k), workers);
                }
                if (status == QCD_STATUS_OK && step.permute_output) {
                    status = qcd_cuda_permute(
                        dtype_id, gemm_output, step_output, step.output_permute, nullptr);
                }
                break;
            }
        }
        if (status != QCD_STATUS_OK) {
            cudaDeviceSynchronize();
            destroy_timing();
            return status;
        }
        if (step_timing) {
            cudaEventRecord(timing_stop[step_index], nullptr);
            if (step_index + 1 < prepared.size()) {
                cudaEventRecord(timing_start[step_index + 1], nullptr);
            }
        }
    }

    const int32_t final_node = prepared.back().output;
    int64_t final_elements = prepared.back().output_elements;
    std::size_t final_bytes = 0;
    if (!allocation_bytes(final_elements, dtype_id, &final_bytes)) {
        destroy_timing();
        return QCD_STATUS_INVALID;
    }
    if (final_elements > 0) {
        const cudaError_t copy_error = cudaMemcpyAsync(
            output_data, nodes[static_cast<std::size_t>(final_node)], final_bytes,
            cudaMemcpyDeviceToDevice, nullptr);
        if (copy_error != cudaSuccess) {
            destroy_timing();
            return map_cuda_error(copy_error);
        }
    }
    const cudaError_t sync_error = cudaDeviceSynchronize();
    if (step_timing) {
        for (std::size_t index = 0; index < timing_count; ++index) {
            float milliseconds = 0.0f;
            const bool measured =
                cudaEventElapsedTime(&milliseconds, timing_start[index],
                                     timing_stop[index]) == cudaSuccess;
            const PreparedStep& step = prepared[index];
            std::fprintf(
                stderr,
                "PYQCD_EINSUM_CUDA_STEP index=%zu kind=%s m=%lld n=%lld k=%lld "
                "batch=%lld pack_skipped=%d ms=%.4f\n",
                index, operation_kind_name(step.kind),
                static_cast<long long>(step.m), static_cast<long long>(step.n),
                static_cast<long long>(step.k), static_cast<long long>(step.batch),
                timing_skipped[index], measured ? milliseconds : -1.0f);
        }
    }
    destroy_timing();
    return map_cuda_error(sync_error);
}

int execute_sandwich(
    const pyqcd_einsum::ExecutionPlan& plan,
    const std::vector<PreparedStep>& prepared,
    const SandwichPlan& sandwich,
    CudaWorkspace* workspace,
    const std::vector<void*>& input_data,
    void* output_data) {
    if (workspace == nullptr || prepared.empty() || qcd_cuda_sandwich == nullptr) {
        return QCD_STATUS_INTERNAL;
    }
    const int64_t output_elements = prepared.back().output_elements;
    int status = workspace->prepare_sandwich(output_elements, plan.dtype_id);
    if (status != QCD_STATUS_OK) {
        return status;
    }
    void* workspace_output = workspace->output(0);
    status = qcd_cuda_sandwich(
        plan.dtype_id,
        input_data[static_cast<std::size_t>(sandwich.left_matrix_input)],
        input_data[static_cast<std::size_t>(sandwich.tensor_input)],
        input_data[static_cast<std::size_t>(sandwich.right_matrix_input)],
        workspace_output,
        sandwich.spec,
        nullptr);
    if (status != QCD_STATUS_OK) {
        cudaDeviceSynchronize();
        return status;
    }

    std::size_t output_bytes = 0;
    if (!allocation_bytes(output_elements, plan.dtype_id, &output_bytes)) {
        return QCD_STATUS_INVALID;
    }
    const cudaError_t copy_error = cudaMemcpyAsync(
        output_data, workspace_output, output_bytes,
        cudaMemcpyDeviceToDevice, nullptr);
    if (copy_error != cudaSuccess) {
        return map_cuda_error(copy_error);
    }
    if (trace_enabled()) {
        std::fprintf(
            stderr,
            "PYQCD_EINSUM_CUDA_SANDWICH prefix=%lld left=%lld middle=%lld "
            "right=%lld lane=%lld kernel=1\n",
            static_cast<long long>(sandwich.spec.prefix),
            static_cast<long long>(sandwich.spec.left),
            static_cast<long long>(sandwich.spec.contracted_left),
            static_cast<long long>(sandwich.spec.right),
            static_cast<long long>(sandwich.spec.lane));
    }
    return map_cuda_error(cudaDeviceSynchronize());
}

int execute_e4_fused(
    const pyqcd_einsum::ExecutionPlan& plan,
    const E4FusedPlan& fused,
    CudaWorkspace* workspace,
    const std::vector<void*>& input_data,
    void* output_data) {
    if (workspace == nullptr || !fused.matched || qcd_cuda_e4_fused == nullptr ||
        input_data.size() != 4) {
        return QCD_STATUS_INTERNAL;
    }
    int status = workspace->prepare_e4(plan.dtype_id);
    if (status != QCD_STATUS_OK) {
        return status;
    }
    status = qcd_cuda_e4_fused(
        plan.dtype_id,
        input_data[0],
        input_data[1],
        input_data[2],
        input_data[3],
        output_data,
        fused.spec,
        nullptr);
    if (status != QCD_STATUS_OK) {
        cudaDeviceSynchronize();
        return status;
    }
    if (trace_enabled()) {
        std::fprintf(
            stderr,
            "PYQCD_EINSUM_CUDA_E4 z=%lld y=%lld x=%lld c=%lld t=%lld "
            "a=%lld b=%lld n=%lld g=%lld G=%lld d=%lld e=%lld f=%lld kernel=1\n",
            static_cast<long long>(fused.spec.z),
            static_cast<long long>(fused.spec.y),
            static_cast<long long>(fused.spec.x),
            static_cast<long long>(fused.spec.c),
            static_cast<long long>(fused.spec.t),
            static_cast<long long>(fused.spec.a),
            static_cast<long long>(fused.spec.b),
            static_cast<long long>(fused.spec.n),
            static_cast<long long>(fused.spec.input_g),
            static_cast<long long>(fused.spec.output_g),
            static_cast<long long>(fused.spec.d),
            static_cast<long long>(fused.spec.e),
            static_cast<long long>(fused.spec.f));
    }
    return map_cuda_error(cudaDeviceSynchronize());
}

struct CompiledContext {
    uint64_t magic = kCompiledContextMagic;
    pyqcd_einsum::ExecutionPlan plan;
    std::vector<ResolvedStep> resolved;
    std::vector<PreparedStep> prepared;
    SandwichPlan sandwich;
    E4FusedPlan e4_fused;
    CudaWorkspace workspace;
    std::mutex mutex;

    ~CompiledContext() {
        workspace.release();
        magic = 0;
    }
};

bool initialize_context(const void* plan_data, CompiledContext* context) {
    if (plan_data == nullptr || context == nullptr ||
        !decode_execution_plan(plan_data, &context->plan)) {
        return false;
    }
    if (!resolve_plan(context->plan, &context->resolved) ||
        context->resolved.size() != context->plan.steps.size()) {
        return false;
    }
    if (!prepare_execution(context->plan, context->resolved, &context->prepared) ||
        context->prepared.size() != context->plan.steps.size()) {
        return false;
    }
    // specialize=False 时由 Python 侧设置该变量，强制走通用多步路径以便 A/B 对照。
    if (std::getenv("PYQCD_EINSUM_DISABLE_FUSED") == nullptr) {
        match_sandwich(context->plan, context->resolved, &context->sandwich);
        match_e4_fused(context->plan, context->resolved, &context->e4_fused);
    } else if (trace_enabled()) {
        std::fprintf(stderr, "PYQCD_EINSUM_FUSED_DISABLED\n");
    }
    return true;
}

int execute_context(
    CompiledContext* context,
    long long tensors_address,
    long long out_address) {
    if (context == nullptr || context->magic != kCompiledContextMagic) {
        return QCD_STATUS_INVALID;
    }
    const int64_t* tensor_addresses = pointer_from_address<const int64_t>(tensors_address);
    const int64_t* out_addresses = pointer_from_address<const int64_t>(out_address);
    if (tensor_addresses == nullptr || out_addresses == nullptr) {
        return QCD_STATUS_INVALID;
    }

    std::vector<void*> input_data(
        static_cast<std::size_t>(context->plan.num_inputs), nullptr);
    for (int32_t input = 0; input < context->plan.num_inputs; ++input) {
        input_data[static_cast<std::size_t>(input)] =
            pointer_from_address<void>(tensor_addresses[input]);
        if (input_data[static_cast<std::size_t>(input)] == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    void* output_data = pointer_from_address<void>(out_addresses[0]);
    if (output_data == nullptr) {
        return QCD_STATUS_INVALID;
    }

    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->e4_fused.matched && qcd_cuda_e4_fused != nullptr) {
        return execute_e4_fused(
            context->plan, context->e4_fused, &context->workspace,
            input_data, output_data);
    }
    if (context->sandwich.matched && qcd_cuda_sandwich != nullptr) {
        return execute_sandwich(
            context->plan, context->prepared, context->sandwich,
            &context->workspace, input_data, output_data);
    }
    return execute_prepared(
        context->plan, context->prepared, &context->workspace, input_data, output_data);
}

int compile_entry(long long plan_address, long long params_address, long long out_context) {
    const void* plan_data = pointer_from_address<const void>(plan_address);
    const int32_t* params = pointer_from_address<const int32_t>(params_address);
    int64_t* destination = pointer_from_address<int64_t>(out_context);
    if (plan_data == nullptr || params == nullptr || destination == nullptr) {
        return QCD_STATUS_INVALID;
    }
    (void)params;
    auto context = std::make_unique<CompiledContext>();
    if (!initialize_context(plan_data, context.get())) {
        return QCD_STATUS_INVALID;
    }
    *destination = static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(context.release()));
    return QCD_STATUS_OK;
}

int contract_entry(long long plan_address, long long tensors_address, long long out_address,
                   long long params_address) {
    const void* plan_data = pointer_from_address<const void>(plan_address);
    const int32_t* params = pointer_from_address<const int32_t>(params_address);
    if (plan_data == nullptr || params == nullptr) {
        return QCD_STATUS_INVALID;
    }
    (void)params;
    CompiledContext context;
    if (!initialize_context(plan_data, &context)) {
        return QCD_STATUS_INVALID;
    }
    return execute_context(&context, tensors_address, out_address);
}

int compiled_entry(long long context_address, long long tensors_address, long long out_address,
                   long long params_address) {
    std::shared_lock<std::shared_mutex> lifecycle(context_lifecycle_mutex);
    CompiledContext* context = pointer_from_address<CompiledContext>(context_address);
    const int32_t* params = pointer_from_address<const int32_t>(params_address);
    if (context == nullptr || params == nullptr) {
        return QCD_STATUS_INVALID;
    }
    (void)params;
    return execute_context(context, tensors_address, out_address);
}

void destroy_entry(long long context_address) {
    std::unique_lock<std::shared_mutex> lifecycle(context_lifecycle_mutex);
    CompiledContext* context = pointer_from_address<CompiledContext>(context_address);
    if (context == nullptr || context->magic != kCompiledContextMagic) {
        return;
    }
    context->magic = 0;
    delete context;
}

bool providers_available(const std::vector<PreparedStep>& prepared) {
    bool needs_permute = false;
    bool needs_reduce = false;
    bool needs_hadamard = false;
    bool needs_gemm = false;
    for (const PreparedStep& step : prepared) {
        switch (step.kind) {
            case OperationKind::kCopy:
                needs_permute = true;
                break;
            case OperationKind::kReduce:
                needs_reduce = true;
                break;
            case OperationKind::kHadamard:
                needs_hadamard = true;
                break;
            case OperationKind::kGemm:
                needs_permute = true;
                needs_gemm = true;
                break;
        }
    }
    return (!needs_permute || qcd_cuda_permute != nullptr) &&
           (!needs_reduce || qcd_cuda_reduce != nullptr) &&
           (!needs_hadamard || qcd_cuda_hadamard != nullptr) &&
           (!needs_gemm || qcd_cuda_batched_gemm != nullptr);
}

struct TopologyProgram {
    std::mutex mutex;
    std::vector<uint8_t> bytes;
    int dtype_id = -1;
    pyqcd_einsum::ExecutionPlan plan;
    std::vector<ResolvedStep> resolved;
    std::vector<PreparedStep> prepared;
    CudaWorkspace workspace;

    bool matches(const void* plan_data, long long plan_len, int requested_dtype) {
        if (dtype_id != requested_dtype || bytes.empty() ||
            plan_len != static_cast<long long>(bytes.size())) {
            return false;
        }
        return std::memcmp(bytes.data(), plan_data, bytes.size()) == 0;
    }

    int load(const void* plan_data, long long plan_len, int requested_dtype) {
        pyqcd_einsum::ExecutionPlan parsed;
        if (is_topology_descriptor(plan_data, plan_len)) {
            /* 描述符：结构 + 拓扑摘要 → 预编译步表 → 执行计划。
             * 一条 path 搜索都没有：顺序来自 topology_steps.h 的常量表。 */
            TopologyDescriptor descriptor;
            if (!parse_topology_descriptor(plan_data, plan_len, &descriptor)) {
                if (trace_enabled()) {
                    std::fprintf(stderr, "PYQCD_EINSUM_CUDA_TOPOLOGY reject=descriptor\n");
                }
                return QCD_TOPOLOGY_UNSUPPORTED;
            }
            if (descriptor.dtype_id != requested_dtype) {
                return QCD_STATUS_INVALID;
            }
            if (!build_topology_plan(descriptor, &parsed)) {
                if (trace_enabled()) {
                    std::fprintf(
                        stderr,
                        "PYQCD_EINSUM_CUDA_TOPOLOGY reject=table key=0x%016llx "
                        "arity=%d\n",
                        static_cast<unsigned long long>(descriptor.key),
                        descriptor.num_inputs);
                }
                return QCD_TOPOLOGY_UNSUPPORTED;
            }
        } else if (!decode_execution_plan(plan_data, &parsed)) {
            return QCD_TOPOLOGY_UNSUPPORTED;
        }
        if (parsed.dtype_id != requested_dtype) {
            return QCD_STATUS_INVALID;
        }
        std::vector<ResolvedStep> candidate_resolved;
        std::vector<PreparedStep> candidate_prepared;
        if (!resolve_plan(parsed, &candidate_resolved) ||
            candidate_resolved.size() != parsed.steps.size() ||
            !prepare_execution(parsed, candidate_resolved, &candidate_prepared) ||
            candidate_prepared.size() != parsed.steps.size()) {
            return QCD_TOPOLOGY_UNSUPPORTED;
        }
        if (!providers_available(candidate_prepared)) {
            return QCD_STATUS_INTERNAL;
        }
        std::vector<uint8_t> candidate_bytes(
            static_cast<const uint8_t*>(plan_data),
            static_cast<const uint8_t*>(plan_data) + plan_len);
        /* 计划换了：偏移表作废，但 arena 留在池子里复用（容量够就不重新 malloc）。 */
        workspace.invalidate_layout();
        bytes = std::move(candidate_bytes);
        dtype_id = requested_dtype;
        plan = std::move(parsed);
        resolved = std::move(candidate_resolved);
        prepared = std::move(candidate_prepared);
        return QCD_STATUS_OK;
    }
};

TopologyProgram& topology_program() {
    static TopologyProgram program;
    return program;
}

bool offset_pointer_checked(
    void* base, long long offset, void** result) {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
    if (offset < 0 ||
        static_cast<std::uint64_t>(offset) >
            std::numeric_limits<std::uintptr_t>::max() - address) {
        return false;
    }
    *result = reinterpret_cast<void*>(address + static_cast<std::uintptr_t>(offset));
    return true;
}

int topology_entry(
    const void* plan_data,
    long long plan_len,
    void* const* operands,
    int n_operands,
    const long long* operand_offsets,
    void* output,
    int dtype_id,
    void* stream) {
    if (plan_data == nullptr || plan_len <= 0 || operands == nullptr ||
        output == nullptr || stream != nullptr || n_operands <= 0 ||
        !topology_dtype_supported(dtype_id)) {
        return QCD_STATUS_INVALID;
    }
    const bool descriptor = is_topology_descriptor(plan_data, plan_len);
    const long long minimum_bytes =
        static_cast<long long>((descriptor ? kTopologyDescriptorHeaderWords
                                           : 9) * sizeof(int32_t));
    if (plan_len < minimum_bytes ||
        static_cast<long long>(static_cast<const int32_t*>(plan_data)[2]) != plan_len) {
        return QCD_STATUS_INVALID;
    }
    TopologyProgram& program = topology_program();
    std::lock_guard<std::mutex> lock(program.mutex);
    if (!program.matches(plan_data, plan_len, dtype_id)) {
        const int status = program.load(plan_data, plan_len, dtype_id);
        if (status != QCD_STATUS_OK) {
            return status;
        }
    }
    if (n_operands != program.plan.num_inputs) {
        return QCD_STATUS_INVALID;
    }
    std::vector<void*> input_data(
        static_cast<std::size_t>(program.plan.num_inputs), nullptr);
    for (int32_t index = 0; index < program.plan.num_inputs; ++index) {
        void* base = operands[index];
        if (base == nullptr) {
            return QCD_STATUS_INVALID;
        }
        const long long offset =
            operand_offsets == nullptr ? 0 : operand_offsets[index];
        if (!offset_pointer_checked(base, offset, &input_data[static_cast<std::size_t>(index)])) {
            return QCD_STATUS_INVALID;
        }
    }
    return execute_prepared(
        program.plan, program.prepared, &program.workspace, input_data, output);
}

}  // namespace

extern "C" QCD_EINSUM_API int applyContractQcd(
    long long plan, long long tensors, long long out, long long params) {
    try {
        return contract_entry(plan, tensors, out, params);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_EINSUM_API int applyContractMpiQcd(
    long long plan, long long tensors, long long out, long long params) {
    (void)plan;
    (void)tensors;
    (void)out;
    (void)params;
    return QCD_STATUS_INVALID;
}

extern "C" QCD_EINSUM_API int applyCompileQcd(
    long long plan, long long params, long long out_context) {
    try {
        return compile_entry(plan, params, out_context);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_EINSUM_API int applyCompiledQcd(
    long long context, long long tensors, long long out, long long params) {
    try {
        return compiled_entry(context, tensors, out, params);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

extern "C" QCD_EINSUM_API void applyDestroyQcd(long long context) {
    try {
        destroy_entry(context);
    } catch (...) {
    }
}

extern "C" QCD_CUDA_TOPOLOGY_API int qcd_cuda_topology_apply(
    const void* plan,
    long long plan_len,
    void* const* operands,
    int n_operands,
    const long long* operand_offsets,
    void* out,
    int dtype_id,
    void* stream) {
    try {
        return topology_entry(
            plan, plan_len, operands, n_operands, operand_offsets, out, dtype_id,
            stream);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}
