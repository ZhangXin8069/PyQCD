#include "einsum_runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace pyqcd_einsum {
namespace {

constexpr int32_t kMaxRank = 64;
constexpr int32_t kMaxInputs = 1 << 16;
constexpr int32_t kMaxSteps = 1 << 20;
constexpr int32_t kMaxSymbols = 1 << 20;
constexpr uint64_t kMaxPlanBytes = uint64_t(1) << 30;

class Reader {
  public:
    Reader(const void* data, std::size_t limit) : data_(static_cast<const uint8_t*>(data)), limit_(limit) {}

    bool read_i32(int32_t* value) {
        if (offset_ + sizeof(int32_t) > limit_) {
            return false;
        }
        std::memcpy(value, data_ + offset_, sizeof(int32_t));
        offset_ += sizeof(int32_t);
        return true;
    }

    bool read_i64(int64_t* value) {
        const std::size_t aligned = (offset_ + alignof(int64_t) - 1) & ~(alignof(int64_t) - 1);
        if (aligned + sizeof(int64_t) > limit_) {
            return false;
        }
        std::memcpy(value, data_ + aligned, sizeof(int64_t));
        offset_ = aligned + sizeof(int64_t);
        return true;
    }

    std::size_t offset() const { return offset_; }

  private:
    const uint8_t* data_;
    std::size_t limit_;
    std::size_t offset_ = 0;
};

bool read_labels(Reader* reader, std::vector<int32_t>* labels, int32_t num_symbols) {
    int32_t rank = 0;
    if (!reader->read_i32(&rank) || rank < 0 || rank > kMaxRank) {
        return false;
    }
    labels->assign(static_cast<std::size_t>(rank), 0);
    for (int32_t index = 0; index < rank; ++index) {
        int32_t label = 0;
        if (!reader->read_i32(&label) || label < 0 || label >= num_symbols) {
            return false;
        }
        (*labels)[static_cast<std::size_t>(index)] = label;
    }
    return true;
}

bool unique_labels(const std::vector<int32_t>& labels) {
    return std::unordered_set<int32_t>(labels.begin(), labels.end()).size() == labels.size();
}

bool contains(const std::vector<int32_t>& labels, int32_t value) {
    return std::find(labels.begin(), labels.end(), value) != labels.end();
}

int32_t position_of(const std::vector<int32_t>& labels, int32_t value) {
    const auto found = std::find(labels.begin(), labels.end(), value);
    return found == labels.end() ? -1 : static_cast<int32_t>(found - labels.begin());
}

bool valid_symbol(int32_t label, const std::vector<int64_t>& sizes) {
    return label >= 0 && static_cast<std::size_t>(label) < sizes.size();
}

}  // namespace

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
        strides[index - 2] = strides[index - 1] * shape[index - 1];
    }
    return strides;
}

int64_t checked_product(const std::vector<int64_t>& shape, bool* ok) {
    int64_t product = 1;
    for (int64_t extent : shape) {
        if (extent < 0 || (extent != 0 && product > std::numeric_limits<int64_t>::max() / extent)) {
            *ok = false;
            return 0;
        }
        product *= extent;
    }
    return product;
}

int64_t element_count(const TensorMeta& tensor, bool* ok) {
    if (tensor.shape.size() != tensor.labels.size()) {
        *ok = false;
        return 0;
    }
    return checked_product(tensor.shape, ok);
}

bool parse_execution_plan(const void* data, ExecutionPlan* plan, std::string* error) {
    if (data == nullptr || plan == nullptr) {
        if (error != nullptr) {
            *error = "plan pointer is null";
        }
        return false;
    }
    const int32_t* header = static_cast<const int32_t*>(data);
    if (header[0] != static_cast<int32_t>(kExecutionMagic)) {
        if (error != nullptr) {
            *error = "bad execution-plan magic";
        }
        return false;
    }
    if (header[1] != static_cast<int32_t>(kExecutionVersion)) {
        if (error != nullptr) {
            *error = "unsupported execution-plan version";
        }
        return false;
    }
    const int64_t total_bytes = header[2];
    if (total_bytes <= 0 || static_cast<uint64_t>(total_bytes) > kMaxPlanBytes) {
        if (error != nullptr) {
            *error = "invalid execution-plan byte count";
        }
        return false;
    }
    const int32_t num_inputs = header[3];
    const int32_t num_steps = header[4];
    const int32_t num_symbols = header[5];
    const int32_t output_ndim = header[6];
    const int32_t dtype_id = header[7];
    if (num_inputs < 0 || num_inputs > kMaxInputs || num_steps < 0 || num_steps > kMaxSteps ||
        num_symbols < 0 || num_symbols > kMaxSymbols || output_ndim < 0 || output_ndim > kMaxRank ||
        dtype_id < QCD_DTYPE_COMPLEX128 || dtype_id > QCD_DTYPE_FLOAT32) {
        if (error != nullptr) {
            *error = "execution-plan header is out of range";
        }
        return false;
    }

    Reader reader(data, static_cast<std::size_t>(total_bytes));
    for (int32_t index = 0; index < 9; ++index) {
        int32_t ignored = 0;
        if (!reader.read_i32(&ignored)) {
            return false;
        }
    }
    const std::size_t aligned_sizes = (reader.offset() + alignof(int64_t) - 1) &
                                      ~(alignof(int64_t) - 1);
    if (aligned_sizes > static_cast<std::size_t>(total_bytes)) {
        return false;
    }
    Reader sizes_reader(static_cast<const uint8_t*>(data) + aligned_sizes,
                        static_cast<std::size_t>(total_bytes) - aligned_sizes);
    std::vector<int64_t> sizes(static_cast<std::size_t>(num_symbols), 0);
    for (int32_t index = 0; index < num_symbols; ++index) {
        if (!sizes_reader.read_i64(&sizes[static_cast<std::size_t>(index)]) || sizes[static_cast<std::size_t>(index)] < 0) {
            if (error != nullptr) {
                *error = "invalid symbol-size table";
            }
            return false;
        }
    }
    const std::size_t after_sizes = aligned_sizes + sizes_reader.offset();
    Reader body(static_cast<const uint8_t*>(data) + after_sizes,
                static_cast<std::size_t>(total_bytes) - after_sizes);

    ExecutionPlan parsed;
    parsed.dtype_id = dtype_id;
    parsed.num_inputs = num_inputs;
    parsed.output_ndim = output_ndim;
    parsed.sizes = std::move(sizes);
    parsed.input_shapes.resize(static_cast<std::size_t>(num_inputs));
    for (int32_t input = 0; input < num_inputs; ++input) {
        int32_t rank = 0;
        if (!body.read_i32(&rank) || rank < 0 || rank > kMaxRank) {
            if (error != nullptr) {
                *error = "invalid input rank";
            }
            return false;
        }
        parsed.input_shapes[static_cast<std::size_t>(input)].resize(static_cast<std::size_t>(rank));
        for (int32_t axis = 0; axis < rank; ++axis) {
            int64_t extent = 0;
            if (!body.read_i64(&extent) || extent < 0) {
                if (error != nullptr) {
                    *error = "invalid input shape";
                }
                return false;
            }
            parsed.input_shapes[static_cast<std::size_t>(input)][static_cast<std::size_t>(axis)] = extent;
        }
    }
    parsed.output_labels.resize(static_cast<std::size_t>(output_ndim));
    for (int32_t axis = 0; axis < output_ndim; ++axis) {
        int32_t label = 0;
        if (!body.read_i32(&label) || !valid_symbol(label, parsed.sizes)) {
            if (error != nullptr) {
                *error = "invalid output label";
            }
            return false;
        }
        parsed.output_labels[static_cast<std::size_t>(axis)] = label;
    }
    if (!unique_labels(parsed.output_labels)) {
        if (error != nullptr) {
            *error = "duplicate output label";
        }
        return false;
    }

    parsed.steps.resize(static_cast<std::size_t>(num_steps));
    for (int32_t step_index = 0; step_index < num_steps; ++step_index) {
        ExecutionStep& step = parsed.steps[static_cast<std::size_t>(step_index)];
        int32_t arity = 0;
        if (!body.read_i32(&arity) || arity < 1 || arity > 2) {
            if (error != nullptr) {
                *error = "execution step arity must be one or two";
            }
            return false;
        }
        step.positions.resize(static_cast<std::size_t>(arity));
        for (int32_t operand = 0; operand < arity; ++operand) {
            if (!body.read_i32(&step.positions[static_cast<std::size_t>(operand)]) ||
                step.positions[static_cast<std::size_t>(operand)] < 0) {
                if (error != nullptr) {
                    *error = "invalid operand position";
                }
                return false;
            }
        }
        step.inputs.resize(static_cast<std::size_t>(arity));
        for (int32_t operand = 0; operand < arity; ++operand) {
            TensorMeta& input = step.inputs[static_cast<std::size_t>(operand)];
            if (!read_labels(&body, &input.labels, num_symbols) || !unique_labels(input.labels)) {
                if (error != nullptr) {
                    *error = "native execution does not support repeated labels within an operand";
                }
                return false;
            }
            input.shape.resize(input.labels.size());
            for (std::size_t axis = 0; axis < input.labels.size(); ++axis) {
                input.shape[axis] = parsed.sizes[static_cast<std::size_t>(input.labels[axis])];
            }
        }
        if (!read_labels(&body, &step.output.labels, num_symbols) || !unique_labels(step.output.labels)) {
            if (error != nullptr) {
                *error = "invalid step output labels";
            }
            return false;
        }
        step.output.shape.resize(step.output.labels.size());
        for (std::size_t axis = 0; axis < step.output.labels.size(); ++axis) {
            step.output.shape[axis] = parsed.sizes[static_cast<std::size_t>(step.output.labels[axis])];
        }
    }
    if (body.offset() != static_cast<std::size_t>(total_bytes) - after_sizes) {
        if (error != nullptr) {
            *error = "execution plan has trailing bytes";
        }
        return false;
    }
    *plan = std::move(parsed);
    return true;
}

bool analyze_pair(
    const TensorMeta& left,
    const TensorMeta& right,
    const TensorMeta& output,
    const std::vector<int64_t>& sizes,
    PairPlan* plan) {
    if (plan == nullptr || !unique_labels(left.labels) || !unique_labels(right.labels) ||
        !unique_labels(output.labels)) {
        return false;
    }
    PairPlan result;
    result.left_labels = left.labels;
    result.right_labels = right.labels;
    result.output_labels = output.labels;

    std::unordered_set<int32_t> left_set(left.labels.begin(), left.labels.end());
    std::unordered_set<int32_t> right_set(right.labels.begin(), right.labels.end());
    std::unordered_set<int32_t> output_set(output.labels.begin(), output.labels.end());

    for (int32_t label : left.labels) {
        if (!valid_symbol(label, sizes)) {
            result.reason = "invalid left label";
            return false;
        }
        const bool in_right = right_set.count(label) != 0;
        const bool in_output = output_set.count(label) != 0;
        if (!in_output && !in_right) {
            result.reason = "left-only reduction requires a dedicated reduce kernel";
            return false;
        }
    }
    for (int32_t label : right.labels) {
        if (!valid_symbol(label, sizes)) {
            result.reason = "invalid right label";
            return false;
        }
        const bool in_left = left_set.count(label) != 0;
        const bool in_output = output_set.count(label) != 0;
        if (!in_output && !in_left) {
            result.reason = "right-only reduction requires a dedicated reduce kernel";
            return false;
        }
    }
    for (int32_t label : output.labels) {
        if (!valid_symbol(label, sizes) || (left_set.count(label) == 0 && right_set.count(label) == 0)) {
            result.reason = "output label is absent from both operands";
            return false;
        }
    }

    for (int32_t label : output.labels) {
        if (left_set.count(label) != 0 && right_set.count(label) != 0) {
            result.batch_labels.push_back(label);
        }
    }
    for (int32_t label : left.labels) {
        if (right_set.count(label) != 0 && output_set.count(label) == 0) {
            result.sum_labels.push_back(label);
        }
    }

    std::vector<int32_t> left_free;
    for (int32_t label : left.labels) {
        if (right_set.count(label) == 0) {
            left_free.push_back(label);
            result.left_perm.push_back(position_of(left.labels, label));
        }
    }
    std::vector<int32_t> right_free;
    for (int32_t label : right.labels) {
        if (left_set.count(label) == 0) {
            right_free.push_back(label);
            result.right_perm.push_back(position_of(right.labels, label));
        }
    }

    std::vector<int32_t> left_ordered = result.batch_labels;
    left_ordered.insert(left_ordered.end(), left_free.begin(), left_free.end());
    left_ordered.insert(left_ordered.end(), result.sum_labels.begin(), result.sum_labels.end());
    result.left_perm.clear();
    for (int32_t label : left_ordered) {
        result.left_perm.push_back(position_of(left.labels, label));
        result.left_packed_shape.push_back(sizes[static_cast<std::size_t>(label)]);
    }

    std::vector<int32_t> right_ordered = result.batch_labels;
    right_ordered.insert(right_ordered.end(), right_free.begin(), right_free.end());
    right_ordered.insert(right_ordered.end(), result.sum_labels.begin(), result.sum_labels.end());
    result.right_perm.clear();
    for (int32_t label : right_ordered) {
        result.right_perm.push_back(position_of(right.labels, label));
        result.right_packed_shape.push_back(sizes[static_cast<std::size_t>(label)]);
    }

    std::vector<int32_t> canonical_labels = result.batch_labels;
    canonical_labels.insert(canonical_labels.end(), left_free.begin(), left_free.end());
    canonical_labels.insert(canonical_labels.end(), right_free.begin(), right_free.end());
    for (int32_t label : canonical_labels) {
        result.output_packed_shape.push_back(sizes[static_cast<std::size_t>(label)]);
    }
    for (int32_t label : output.labels) {
        result.output_perm.push_back(position_of(canonical_labels, label));
    }

    result.batches = 1;
    for (int32_t label : result.batch_labels) {
        result.batches *= sizes[static_cast<std::size_t>(label)];
    }
    result.m = 1;
    for (int32_t label : left_free) {
        result.m *= sizes[static_cast<std::size_t>(label)];
    }
    result.n = 1;
    for (int32_t label : right_free) {
        result.n *= sizes[static_cast<std::size_t>(label)];
    }
    result.k = 1;
    for (int32_t label : result.sum_labels) {
        result.k *= sizes[static_cast<std::size_t>(label)];
    }
    std::vector<int32_t> identity_left;
    for (std::size_t axis = 0; axis < left.labels.size(); ++axis) {
        identity_left.push_back(static_cast<int32_t>(axis));
    }
    result.left_is_packed = result.left_perm != identity_left;
    std::vector<int32_t> identity_right;
    for (std::size_t axis = 0; axis < right.labels.size(); ++axis) {
        identity_right.push_back(static_cast<int32_t>(axis));
    }
    result.right_is_packed = result.right_perm != identity_right;
    std::vector<int32_t> identity_output;
    for (std::size_t axis = 0; axis < output.labels.size(); ++axis) {
        identity_output.push_back(static_cast<int32_t>(axis));
    }
    result.output_is_packed = result.output_perm != identity_output;
    *plan = std::move(result);
    return true;
}

bool analyze_unary(
    const TensorMeta& input,
    const TensorMeta& output,
    const std::vector<int64_t>& sizes,
    UnaryPlan* plan) {
    if (plan == nullptr || !unique_labels(input.labels) || !unique_labels(output.labels)) {
        return false;
    }
    UnaryPlan result;
    result.input_labels = input.labels;
    result.output_labels = output.labels;
    for (int32_t label : input.labels) {
        if (!valid_symbol(label, sizes)) {
            result.reason = "invalid unary input label";
            return false;
        }
        if (contains(output.labels, label)) {
            result.input_perm.push_back(position_of(input.labels, label));
        }
    }
    for (int32_t label : output.labels) {
        if (!contains(input.labels, label) || !valid_symbol(label, sizes)) {
            result.reason = "unary output label is absent from its input";
            return false;
        }
        result.output_shape.push_back(sizes[static_cast<std::size_t>(label)]);
    }
    result.is_reduction = output.labels.size() != input.labels.size();
    std::vector<int32_t> identity;
    for (std::size_t axis = 0; axis < input.labels.size(); ++axis) {
        identity.push_back(static_cast<int32_t>(axis));
    }
    result.is_permutation = !result.is_reduction && result.input_perm != identity;
    *plan = std::move(result);
    return true;
}

}  // namespace pyqcd_einsum
