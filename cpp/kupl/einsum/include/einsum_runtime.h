#ifndef PYQCD_EINSUM_RUNTIME_H
#define PYQCD_EINSUM_RUNTIME_H

#include "einsum_define.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pyqcd_einsum {

constexpr uint32_t kExecutionMagic = 0x51434532u;
constexpr uint32_t kExecutionVersion = 2u;

struct TensorMeta {
    std::vector<int32_t> labels;
    std::vector<int64_t> shape;
};

struct ExecutionStep {
    std::vector<int32_t> positions;
    std::vector<TensorMeta> inputs;
    TensorMeta output;
};

struct ExecutionPlan {
    int32_t dtype_id = -1;
    int32_t num_inputs = 0;
    int32_t output_ndim = 0;
    std::vector<int64_t> sizes;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int32_t> output_labels;
    std::vector<ExecutionStep> steps;
};

struct PairPlan {
    std::vector<int32_t> batch_labels;
    std::vector<int32_t> sum_labels;
    std::vector<int32_t> left_labels;
    std::vector<int32_t> right_labels;
    std::vector<int32_t> output_labels;
    std::vector<int32_t> left_perm;
    std::vector<int32_t> right_perm;
    std::vector<int32_t> output_perm;
    std::vector<int64_t> left_packed_shape;
    std::vector<int64_t> right_packed_shape;
    std::vector<int64_t> output_packed_shape;
    int64_t batches = 1;
    int64_t m = 1;
    int64_t n = 1;
    int64_t k = 1;
    bool left_is_packed = false;
    bool right_is_packed = false;
    bool output_is_packed = false;
    std::string reason;
};

struct UnaryPlan {
    std::vector<int32_t> input_labels;
    std::vector<int32_t> output_labels;
    std::vector<int32_t> input_perm;
    std::vector<int64_t> output_shape;
    bool is_permutation = false;
    bool is_reduction = false;
    std::string reason;
};

bool parse_execution_plan(const void* data, ExecutionPlan* plan, std::string* error);

bool analyze_pair(
    const TensorMeta& left,
    const TensorMeta& right,
    const TensorMeta& output,
    const std::vector<int64_t>& sizes,
    PairPlan* plan);

bool analyze_unary(
    const TensorMeta& input,
    const TensorMeta& output,
    const std::vector<int64_t>& sizes,
    UnaryPlan* plan);

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape);
int64_t checked_product(const std::vector<int64_t>& shape, bool* ok);
int64_t element_count(const TensorMeta& tensor, bool* ok);

}  // namespace pyqcd_einsum

#endif
