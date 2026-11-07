#include "einsum_cuda.h"
#include "einsum_runtime.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

// TODO: streams are scoped to one CUDA device; MPI and multi-device scheduling are not implemented.
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

class CudaWorkspace {
  public:
    CudaWorkspace() = default;
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

    ~CudaWorkspace() { release(); }

    int prepare(const std::vector<PreparedStep>& prepared, int dtype_id) {
        int device = -1;
        int status = map_cuda_error(cudaGetDevice(&device));
        if (status != QCD_STATUS_OK) {
            return status;
        }
        if (initialized_ && !sandwich_storage_ && !e4_storage_ &&
            device_ == device && dtype_id_ == dtype_id) {
            return QCD_STATUS_OK;
        }

        release();
        sandwich_storage_ = false;
        dtype_id_ = dtype_id;
        device_ = device;
        output_offsets_.resize(prepared.size(), 0);
        int64_t output_elements = 0;
        max_left_pack_ = 0;
        max_right_pack_ = 0;
        max_gemm_output_ = 0;
        max_batch_ = 1;
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            const PreparedStep& step = prepared[index];
            output_offsets_[index] = output_elements;
            if (!add_checked(output_elements, step.output_elements, &output_elements)) {
                release();
                return QCD_STATUS_INVALID;
            }
            if (step.kind != OperationKind::kGemm) {
                continue;
            }
            int64_t elements = 0;
            int64_t temporary = 0;
            if (!multiply_checked(step.batch, step.m, &temporary) ||
                !multiply_checked(temporary, step.k, &elements)) {
                release();
                return QCD_STATUS_INVALID;
            }
            max_left_pack_ = std::max(max_left_pack_, elements);
            if (!multiply_checked(step.batch, step.n, &temporary) ||
                !multiply_checked(temporary, step.k, &elements)) {
                release();
                return QCD_STATUS_INVALID;
            }
            max_right_pack_ = std::max(max_right_pack_, elements);
            if (!multiply_checked(step.batch, step.m, &temporary) ||
                !multiply_checked(temporary, step.n, &elements)) {
                release();
                return QCD_STATUS_INVALID;
            }
            max_gemm_output_ = std::max(max_gemm_output_, elements);
            const int64_t parallel_units = step.batch > 1 ? step.batch : step.m;
            max_batch_ = std::max(max_batch_, parallel_units);
        }

        int64_t total_elements = output_elements;
        if (!add_checked(total_elements, max_left_pack_, &total_elements) ||
            !add_checked(total_elements, max_right_pack_, &total_elements) ||
            !add_checked(total_elements, max_gemm_output_, &total_elements)) {
            release();
            return QCD_STATUS_INVALID;
        }
        storage_elements_ = total_elements;
        output_elements_ = output_elements;
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

        stream_count_ = std::max(1, std::min(requested_stream_count(),
                                             static_cast<int>(std::min<int64_t>(
                                                 max_batch_, std::numeric_limits<int>::max()))));
        streams_.assign(static_cast<std::size_t>(stream_count_), nullptr);
        handles_.assign(static_cast<std::size_t>(stream_count_), nullptr);
        if (stream_count_ > 1) {
            done_events_.assign(static_cast<std::size_t>(stream_count_), nullptr);
            for (int index = 0; index < stream_count_; ++index) {
                const cudaError_t stream_error = cudaStreamCreateWithFlags(
                    &streams_[static_cast<std::size_t>(index)], cudaStreamNonBlocking);
                if (stream_error != cudaSuccess) {
                    status = map_cuda_error(stream_error);
                    release();
                    return status;
                }
                const cudaError_t event_error = cudaEventCreateWithFlags(
                    &done_events_[static_cast<std::size_t>(index)], cudaEventDisableTiming);
                if (event_error != cudaSuccess) {
                    status = map_cuda_error(event_error);
                    release();
                    return status;
                }
            }
            const cudaError_t event_error =
                cudaEventCreateWithFlags(&pack_event_, cudaEventDisableTiming);
            if (event_error != cudaSuccess) {
                status = map_cuda_error(event_error);
                release();
                return status;
            }
        }

        if (needs_gemm(prepared)) {
            for (int index = 0; index < stream_count_; ++index) {
                cublasHandle_t handle = nullptr;
                status = map_cublas_status(cublasCreate(&handle));
                if (status != QCD_STATUS_OK) {
                    release();
                    return status;
                }
                handles_[static_cast<std::size_t>(index)] = handle;
                status = map_cublas_status(
                    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
                if (status != QCD_STATUS_OK) {
                    release();
                    return status;
                }
            }
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

// A pack whose destination strides already equal the source strides copies each
// element onto itself; the GEMM can consume the operand in place.
// ``PYQCD_EINSUM_CUDA_PACK_ELISION=0`` restores the unconditional pack.
bool pack_elision_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("PYQCD_EINSUM_CUDA_PACK_ELISION");
        if (value == nullptr || *value == '\0') {
            return true;
        }
        return std::atoi(value) != 0;
    }();
    return enabled;
}

bool copy_spec_is_identity(const qcd_cuda_copy_spec& spec) {
    if (spec.elements <= 0 || spec.rank < 0) {
        return false;
    }
    int64_t expected = 1;
    for (int32_t axis = spec.rank - 1; axis >= 0; --axis) {
        if (spec.shape[axis] <= 1) {
            continue;
        }
        if (spec.src_stride[axis] != expected) {
            return false;
        }
        expected *= spec.shape[axis];
    }
    return true;
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
    if (workspace == nullptr) {
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
    std::vector<int> timing_skipped(timing_count, 0);
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
                const bool elide = pack_elision_enabled();
                const bool left_in_place =
                    elide && copy_spec_is_identity(step.left_pack);
                const bool right_in_place =
                    elide && copy_spec_is_identity(step.right_pack);
                if (left_in_place) {
                    left_pack = nodes[static_cast<std::size_t>(step.left)];
                    if (step_timing) {
                        ++timing_skipped[step_index];
                    }
                } else {
                    status = qcd_cuda_permute(
                        dtype_id, nodes[static_cast<std::size_t>(step.left)], left_pack,
                        step.left_pack, nullptr);
                    if (status != QCD_STATUS_OK) {
                        destroy_timing();
                        return status;
                    }
                }
                if (right_in_place) {
                    right_pack = nodes[static_cast<std::size_t>(step.right)];
                    if (step_timing) {
                        ++timing_skipped[step_index];
                    }
                } else {
                    status = qcd_cuda_permute(
                        dtype_id, nodes[static_cast<std::size_t>(step.right)], right_pack,
                        step.right_pack, nullptr);
                    if (status != QCD_STATUS_OK) {
                        destroy_timing();
                        return status;
                    }
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
