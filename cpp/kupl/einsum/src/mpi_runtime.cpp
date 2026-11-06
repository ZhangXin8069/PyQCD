#include "einsum_define.h"

#if defined(PYQCD_EINSUM_ENABLE_MPI)

#include "einsum_mpi.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pyqcd_einsum::ExecutionPlan;
using pyqcd_einsum::ExecutionStep;
using pyqcd_einsum::TensorMeta;

template <typename Type>
const Type* as_const_pointer(long long address) {
    return address == 0 ? nullptr
                        : reinterpret_cast<const Type*>(static_cast<std::intptr_t>(address));
}

template <typename Type>
Type* as_pointer(long long address) {
    return address == 0 ? nullptr
                        : reinterpret_cast<Type*>(static_cast<std::intptr_t>(address));
}

bool contains(const std::vector<int32_t>& labels, int32_t symbol) {
    return std::find(labels.begin(), labels.end(), symbol) != labels.end();
}

int32_t position_of(const std::vector<int32_t>& labels, int32_t symbol) {
    const auto found = std::find(labels.begin(), labels.end(), symbol);
    return found == labels.end() ? -1 : static_cast<int32_t>(found - labels.begin());
}

bool checked_mul(int64_t left, int64_t right, int64_t* result) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_product(const std::vector<int64_t>& shape, int64_t* result) {
    int64_t product = 1;
    for (const int64_t extent : shape) {
        if (!checked_mul(product, extent, &product)) {
            return false;
        }
    }
    *result = product;
    return true;
}

std::size_t scalar_size(int32_t dtype_id) {
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
        case QCD_DTYPE_FLOAT64:
            return sizeof(double);
        case QCD_DTYPE_COMPLEX64:
        case QCD_DTYPE_FLOAT32:
            return sizeof(float);
        default:
            return 0;
    }
}

std::size_t element_size(int32_t dtype_id) {
    const std::size_t scalar = scalar_size(dtype_id);
    if (dtype_id == QCD_DTYPE_COMPLEX128 || dtype_id == QCD_DTYPE_COMPLEX64) {
        return 2 * scalar;
    }
    return scalar;
}

bool checked_bytes(int64_t count, std::size_t size, int64_t* bytes) {
    if (count < 0 || size == 0 ||
        static_cast<uint64_t>(count) >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max() / size)) {
        return false;
    }
    *bytes = count * static_cast<int64_t>(size);
    return true;
}

struct LiveOperand {
    bool initial = false;
    std::vector<int32_t> labels;
};

bool infer_initial_labels(const ExecutionPlan& execution,
                          std::vector<std::vector<int32_t>>* initial_labels) {
    std::vector<LiveOperand> active(static_cast<std::size_t>(execution.num_inputs));
    initial_labels->assign(static_cast<std::size_t>(execution.num_inputs), {});
    std::vector<uint8_t> assigned(static_cast<std::size_t>(execution.num_inputs), 0);
    int32_t assigned_inputs = 0;
    for (LiveOperand& operand : active) {
        operand.initial = true;
    }

    for (const ExecutionStep& step : execution.steps) {
        if (step.positions.size() != step.inputs.size() || step.inputs.empty()) {
            return false;
        }
        std::vector<int32_t> positions = step.positions;
        std::vector<std::size_t> operand_order(positions.size());
        for (std::size_t index = 0; index < operand_order.size(); ++index) {
            operand_order[index] = index;
        }
        std::sort(operand_order.begin(), operand_order.end(),
                  [&](std::size_t left, std::size_t right) {
                      return positions[left] < positions[right];
                  });
        std::sort(positions.begin(), positions.end());
        if (std::adjacent_find(positions.begin(), positions.end()) != positions.end()) {
            return false;
        }
        for (std::size_t order = 0; order < positions.size(); ++order) {
            const std::size_t operand = operand_order[order];
            const int32_t position = positions[order];
            if (position < 0 || static_cast<std::size_t>(position) >= active.size()) {
                return false;
            }
            LiveOperand& live = active[static_cast<std::size_t>(position)];
            if (live.initial) {
                if (!assigned[static_cast<std::size_t>(position)]) {
                    live.labels = step.inputs[operand].labels;
                    (*initial_labels)[static_cast<std::size_t>(position)] = live.labels;
                    assigned[static_cast<std::size_t>(position)] = 1;
                    ++assigned_inputs;
                } else if (live.labels != step.inputs[operand].labels) {
                    return false;
                }
            } else if (live.labels != step.inputs[operand].labels) {
                return false;
            }
        }

        std::reverse(positions.begin(), positions.end());
        for (const int32_t position : positions) {
            active.erase(active.begin() + position);
            assigned.erase(assigned.begin() + position);
        }
        LiveOperand output;
        output.labels = step.output.labels;
        active.push_back(std::move(output));
        assigned.push_back(0);
    }

    if (active.size() != 1) {
        return false;
    }
    return assigned_inputs == execution.num_inputs;
}

void replace_tensor_extent(TensorMeta* tensor, int32_t symbol, int64_t extent) {
    const int32_t axis = position_of(tensor->labels, symbol);
    if (axis >= 0 && static_cast<std::size_t>(axis) < tensor->shape.size()) {
        tensor->shape[static_cast<std::size_t>(axis)] = extent;
    }
}

int64_t shard_extent(int64_t extent, int size, int rank) {
    const int64_t base = extent / size;
    const int64_t remainder = extent % size;
    return base + (rank < remainder ? 1 : 0);
}

int64_t shard_start(int64_t extent, int size, int rank) {
    const int64_t base = extent / size;
    const int64_t remainder = extent % size;
    return static_cast<int64_t>(rank) * base + std::min<int64_t>(rank, remainder);
}

class AlignedBuffer {
  public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    ~AlignedBuffer() { std::free(data_); }

    bool allocate(std::size_t bytes) {
        std::free(data_);
        data_ = nullptr;
        if (bytes == 0) {
            return true;
        }
        void* allocated = nullptr;
        if (posix_memalign(&allocated, 64, bytes) != 0) {
            return false;
        }
        data_ = static_cast<uint8_t*>(allocated);
        return true;
    }

    uint8_t* data() const { return data_; }

  private:
    uint8_t* data_ = nullptr;
};

bool allreduce_error(int local_error) {
    int global_error = 0;
    return MPI_Allreduce(&local_error, &global_error, 1, MPI_INT, MPI_MAX,
                         MPI_COMM_WORLD) == MPI_SUCCESS &&
           global_error == 0;
}

int allreduce_status(int local_status) {
    int global_status = QCD_STATUS_OK;
    if (MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        return QCD_STATUS_INTERNAL;
    }
    return global_status;
}

}  // namespace

namespace {

int execute_mpi_contract(long long plan_address, long long tensors_address,
                         long long out_address, long long params_address) {
    const int32_t* params = as_const_pointer<int32_t>(params_address);
    const int32_t requested_mode =
        params == nullptr ? QCD_MPI_MODE_OFF : params[QCD_PARAM_MPI_MODE];

    // Mode off is the serial fast path and must not touch MPI at all.
    if (requested_mode == QCD_MPI_MODE_OFF) {
        const void* serialized = as_const_pointer<void>(plan_address);
        if (serialized == nullptr || params == nullptr) {
            return QCD_STATUS_INVALID;
        }
        ExecutionPlan execution;
        std::string error;
        if (!pyqcd_einsum::parse_execution_plan(serialized, &execution, &error)) {
            return QCD_STATUS_INVALID;
        }
        return pyqcd_einsum::execute_parsed_contract(execution, tensors_address, out_address);
    }

    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized) {
        return QCD_STATUS_INVALID;
    }
    int thread_level = MPI_THREAD_SINGLE;
    if (MPI_Query_thread(&thread_level) != MPI_SUCCESS || thread_level < MPI_THREAD_FUNNELED) {
        return QCD_STATUS_INVALID;
    }

    int rank = 0;
    int size = 1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
        MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size <= 0) {
        return QCD_STATUS_INTERNAL;
    }

    const int32_t shard_symbol = params == nullptr ? -1 : params[QCD_PARAM_MPI_SHARD_SYMBOL];
    const int32_t shard_axis = params == nullptr ? -1 : params[QCD_PARAM_MPI_SHARD_AXIS];
    const int32_t output_mode = params == nullptr ? -1 : params[QCD_PARAM_MPI_OUTPUT];
    const void* serialized = as_const_pointer<void>(plan_address);
    ExecutionPlan execution;
    std::string parse_error;
    const bool parsed = serialized != nullptr &&
                        pyqcd_einsum::parse_execution_plan(serialized, &execution, &parse_error);
    const int64_t global_extent =
        parsed && shard_symbol >= 0 && static_cast<std::size_t>(shard_symbol) < execution.sizes.size()
            ? execution.sizes[static_cast<std::size_t>(shard_symbol)]
            : -1;

    // A single process uses the serial call path and performs no collectives.
    if (size == 1) {
        if (!parsed) {
            return QCD_STATUS_INVALID;
        }
        return pyqcd_einsum::execute_parsed_contract(execution, tensors_address, out_address);
    }

    int configuration[6] = {
        requested_mode,
        parsed ? execution.dtype_id : -1,
        shard_symbol,
        shard_axis,
        output_mode,
        parsed ? static_cast<int>(global_extent) : -1,
    };
    int configuration_min[6] = {};
    int configuration_max[6] = {};
    if (MPI_Allreduce(configuration, configuration_min, 6, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS ||
        MPI_Allreduce(configuration, configuration_max, 6, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        return QCD_STATUS_INTERNAL;
    }
    for (int index = 0; index < 6; ++index) {
        if (configuration_min[index] != configuration_max[index]) {
            return QCD_STATUS_INVALID;
        }
    }

    int local_validation_error = 0;
    std::vector<std::vector<int32_t>> initial_labels;
    int64_t prefix = 1;
    int64_t suffix = 1;
    int64_t local_extent = 0;
    int64_t local_output_elements = 0;
    int64_t local_output_bytes = 0;
    const int64_t* tensor_addresses =
        as_const_pointer<int64_t>(static_cast<int64_t>(tensors_address));
    uint8_t* destination = as_pointer<uint8_t>(static_cast<int64_t>(out_address));

    if (!parsed || params == nullptr || tensor_addresses == nullptr ||
        (requested_mode != QCD_MPI_MODE_BATCH && requested_mode != QCD_MPI_MODE_FREE &&
         requested_mode != QCD_MPI_MODE_REQUIRE) ||
        (output_mode != QCD_MPI_OUTPUT_GATHER && output_mode != QCD_MPI_OUTPUT_LOCAL) ||
        shard_axis < 0 || static_cast<std::size_t>(shard_axis) >= execution.output_labels.size() ||
        execution.output_labels[static_cast<std::size_t>(shard_axis)] != shard_symbol ||
        global_extent < 0 || execution.num_inputs <= 0 ||
        !infer_initial_labels(execution, &initial_labels)) {
        local_validation_error = 1;
    }

    int initial_occurrences = 0;
    if (!local_validation_error) {
        for (std::size_t input = 0; input < initial_labels.size(); ++input) {
            const std::vector<int32_t>& labels = initial_labels[input];
            const int32_t axis = position_of(labels, shard_symbol);
            if (axis < 0) {
                continue;
            }
            ++initial_occurrences;
            const auto& shape = execution.input_shapes[input];
            if (static_cast<std::size_t>(axis) >= shape.size() ||
                shape[static_cast<std::size_t>(axis)] != global_extent) {
                local_validation_error = 1;
            }
        }
        if (initial_occurrences == 0 ||
            (requested_mode == QCD_MPI_MODE_BATCH && initial_occurrences < 2) ||
            (requested_mode == QCD_MPI_MODE_FREE && initial_occurrences != 1)) {
            local_validation_error = 1;
        }
    }

    if (!local_validation_error) {
        for (const ExecutionStep& step : execution.steps) {
            bool appears_in_input = false;
            for (const TensorMeta& input : step.inputs) {
                if (!contains(input.labels, shard_symbol)) {
                    continue;
                }
                appears_in_input = true;
                const int32_t axis = position_of(input.labels, shard_symbol);
                if (axis < 0 || static_cast<std::size_t>(axis) >= input.shape.size() ||
                    input.shape[static_cast<std::size_t>(axis)] != global_extent) {
                    local_validation_error = 1;
                    break;
                }
            }
            const bool appears_in_output = contains(step.output.labels, shard_symbol);
            if (appears_in_input != appears_in_output) {
                local_validation_error = 1;
            }
            if (appears_in_output) {
                const int32_t axis = position_of(step.output.labels, shard_symbol);
                if (axis < 0 || static_cast<std::size_t>(axis) >= step.output.shape.size() ||
                    step.output.shape[static_cast<std::size_t>(axis)] != global_extent) {
                    local_validation_error = 1;
                }
            }
            if (local_validation_error) {
                break;
            }
        }
    }

    local_extent = shard_extent(global_extent, size, rank);
    std::vector<int64_t> global_output_shape;
    std::vector<int64_t> local_output_shape;
    if (!local_validation_error) {
        global_output_shape.resize(execution.output_labels.size());
        local_output_shape.resize(execution.output_labels.size());
        for (std::size_t axis = 0; axis < execution.output_labels.size(); ++axis) {
            global_output_shape[axis] =
                execution.sizes[static_cast<std::size_t>(execution.output_labels[axis])];
            local_output_shape[axis] = global_output_shape[axis];
        }
        local_output_shape[static_cast<std::size_t>(shard_axis)] = local_extent;

        int64_t before = 1;
        int64_t after = 1;
        for (int32_t axis = 0; axis < shard_axis; ++axis) {
            if (!checked_mul(before, global_output_shape[static_cast<std::size_t>(axis)], &before)) {
                local_validation_error = 1;
                break;
            }
        }
        for (std::size_t axis = static_cast<std::size_t>(shard_axis + 1);
             axis < global_output_shape.size(); ++axis) {
            if (!checked_mul(after, global_output_shape[axis], &after)) {
                local_validation_error = 1;
                break;
            }
        }
        prefix = before;
        suffix = after;
        if (!local_validation_error &&
            (!checked_mul(prefix, local_extent, &local_output_elements) ||
             !checked_mul(local_output_elements, suffix, &local_output_elements) ||
             !checked_bytes(local_output_elements, element_size(execution.dtype_id),
                            &local_output_bytes))) {
            local_validation_error = 1;
        }
        if (!local_validation_error && output_mode == QCD_MPI_OUTPUT_GATHER &&
            local_output_bytes > std::numeric_limits<int>::max()) {
            local_validation_error = 1;
        }
    }

    if (!local_validation_error) {
        for (int32_t input = 0; input < execution.num_inputs; ++input) {
            const auto& labels = initial_labels[static_cast<std::size_t>(input)];
            const int32_t axis = position_of(labels, shard_symbol);
            if (axis < 0) {
                continue;
            }
            std::vector<int64_t> local_shape =
                execution.input_shapes[static_cast<std::size_t>(input)];
            local_shape[static_cast<std::size_t>(axis)] = local_extent;
            int64_t elements = 0;
            if (!checked_product(local_shape, &elements) ||
                (elements != 0 && tensor_addresses[input] == 0)) {
                local_validation_error = 1;
                break;
            }
        }
        if (local_output_bytes != 0 && destination == nullptr) {
            local_validation_error = 1;
        }
    }

    if (!allreduce_error(local_validation_error)) {
        return QCD_STATUS_INVALID;
    }

    ExecutionPlan local_execution = execution;
    local_execution.sizes[static_cast<std::size_t>(shard_symbol)] = local_extent;
    for (int32_t input = 0; input < execution.num_inputs; ++input) {
        const auto& labels = initial_labels[static_cast<std::size_t>(input)];
        const int32_t axis = position_of(labels, shard_symbol);
        if (axis >= 0) {
            local_execution.input_shapes[static_cast<std::size_t>(input)]
                                       [static_cast<std::size_t>(axis)] = local_extent;
        }
    }
    for (ExecutionStep& step : local_execution.steps) {
        for (TensorMeta& input : step.inputs) {
            replace_tensor_extent(&input, shard_symbol, local_extent);
        }
        replace_tensor_extent(&step.output, shard_symbol, local_extent);
    }

    AlignedBuffer local_storage;
    uint8_t* local_destination = destination;
    if (output_mode == QCD_MPI_OUTPUT_GATHER) {
        if (!local_storage.allocate(static_cast<std::size_t>(local_output_bytes))) {
            return allreduce_status(QCD_STATUS_MEMORY);
        }
        local_destination = local_storage.data();
    }
    int64_t local_pointer = static_cast<int64_t>(
        reinterpret_cast<std::intptr_t>(local_destination));
    int local_status = pyqcd_einsum::execute_parsed_contract(
        local_execution, tensors_address,
        static_cast<long long>(reinterpret_cast<std::intptr_t>(&local_pointer)));
    const int status = allreduce_status(local_status);
    if (status != QCD_STATUS_OK || output_mode == QCD_MPI_OUTPUT_LOCAL) {
        return status;
    }

    std::vector<int> counts(static_cast<std::size_t>(size));
    std::vector<int> displacements(static_cast<std::size_t>(size));
    int64_t total_elements = 0;
    for (int owner = 0; owner < size; ++owner) {
        int64_t owner_extent = shard_extent(global_extent, size, owner);
        int64_t owner_elements = 0;
        int64_t owner_bytes = 0;
        if (!checked_mul(prefix, owner_extent, &owner_elements) ||
            !checked_mul(owner_elements, suffix, &owner_elements) ||
            !checked_bytes(owner_elements, element_size(execution.dtype_id), &owner_bytes) ||
            owner_bytes > std::numeric_limits<int>::max() ||
            total_elements > std::numeric_limits<int>::max() - owner_bytes) {
            return allreduce_status(QCD_STATUS_MEMORY);
        }
        counts[static_cast<std::size_t>(owner)] = static_cast<int>(owner_bytes);
        displacements[static_cast<std::size_t>(owner)] = static_cast<int>(total_elements);
        total_elements += owner_bytes;
    }

    std::vector<uint8_t> packed(static_cast<std::size_t>(total_elements));
    if (MPI_Allgatherv(local_storage.data(), static_cast<int>(local_output_bytes), MPI_BYTE,
                       packed.data(), counts.data(), displacements.data(), MPI_BYTE,
                       MPI_COMM_WORLD) != MPI_SUCCESS) {
        return QCD_STATUS_INTERNAL;
    }

    for (int owner = 0; owner < size; ++owner) {
        const int64_t owner_extent = shard_extent(global_extent, size, owner);
        const int64_t owner_start = shard_start(global_extent, size, owner);
        const int64_t owner_block_bytes = owner_extent * suffix *
                                          static_cast<int64_t>(element_size(execution.dtype_id));
        const uint8_t* source = packed.data() + displacements[static_cast<std::size_t>(owner)];
        for (int64_t outer = 0; outer < prefix; ++outer) {
            uint8_t* target = destination +
                              (outer * global_extent + owner_start) * suffix *
                                  static_cast<int64_t>(element_size(execution.dtype_id));
            std::memcpy(target, source + outer * owner_block_bytes,
                        static_cast<std::size_t>(owner_block_bytes));
        }
    }
    return QCD_STATUS_OK;
}

}  // namespace

extern "C" QCD_EINSUM_API int applyContractMpiQcd(long long plan_address,
                                                   long long tensors_address,
                                                   long long out_address,
                                                   long long params_address) {
    try {
        return execute_mpi_contract(
            plan_address, tensors_address, out_address, params_address
        );
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::length_error&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

#endif  // PYQCD_EINSUM_ENABLE_MPI
