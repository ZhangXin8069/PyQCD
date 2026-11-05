/* C ABI marshalling for the planning kernel.
 *
 * Everything that crosses this file is a raw address, so every length the
 * planner cannot derive from the data is validated before a single byte is
 * written: the kernel either fills ``set_ptrs`` exactly as documented in
 * einsum_define.h or reports a status and leaves the output untouched.
 */

#include "einsum_define.h"
#include "einsum_plan.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <vector>

namespace {

constexpr int32_t kMaxTensors = 1 << 16;
constexpr int32_t kMaxSymbols = 1 << 18;
constexpr int64_t kMaxFlatIndices = int64_t(1) << 26;

template <typename Type>
const Type* as_const_pointer(int64_t address) {
    if (address == 0) {
        return nullptr;
    }
    return reinterpret_cast<const Type*>(static_cast<std::intptr_t>(address));
}

template <typename Type>
Type* as_pointer(int64_t address) {
    if (address == 0) {
        return nullptr;
    }
    return reinterpret_cast<Type*>(static_cast<std::intptr_t>(address));
}

int plan_entry(long long set_ptrs, long long params) {
    const int64_t* set = as_const_pointer<int64_t>(static_cast<int64_t>(set_ptrs));
    const int32_t* param = as_const_pointer<int32_t>(static_cast<int64_t>(params));
    if (set == nullptr || param == nullptr) {
        return QCD_STATUS_INVALID;
    }

    const int32_t num_ops = param[QCD_PARAM_NUM_OPS];
    const int32_t max_ndim = param[QCD_PARAM_MAX_NDIM];
    const int32_t planner_id = param[QCD_PARAM_PLANNER_ID];
    const int32_t mem_hi = param[QCD_PARAM_MEM_HI];
    const int32_t mem_lo = param[QCD_PARAM_MEM_LO];
    if (num_ops < 0 || num_ops > kMaxTensors || max_ndim < 0 || planner_id < 0 ||
        planner_id > QCD_PLANNER_BRANCH) {
        return QCD_STATUS_INVALID;
    }

    const int64_t num_symbols = set[QCD_SET_NUM_SYMBOLS];
    const int64_t output_ndim = set[QCD_SET_OUTPUT_NDIM];
    const int64_t path_capacity = set[QCD_SET_PATH_CAPACITY];
    if (num_symbols < 0 || num_symbols > kMaxSymbols) {
        return QCD_STATUS_INVALID;
    }
    if (output_ndim < 0 || output_ndim > num_symbols) {
        return QCD_STATUS_INVALID;
    }
    if (path_capacity < 0) {
        return QCD_STATUS_INVALID;
    }
    if (set[QCD_SET_SCRATCH_SIZE] > 0 && set[QCD_SET_SCRATCH] == 0) {
        return QCD_STATUS_INVALID;
    }

    const int64_t* sizes = as_const_pointer<int64_t>(set[QCD_SET_SIZES]);
    if (num_symbols > 0) {
        if (sizes == nullptr) {
            return QCD_STATUS_INVALID;
        }
        for (int64_t symbol = 0; symbol < num_symbols; ++symbol) {
            if (sizes[symbol] < 0) {
                return QCD_STATUS_INVALID;
            }
        }
    }

    const int64_t* offsets = nullptr;
    const int32_t* flat = nullptr;
    int64_t flat_count = 0;
    if (num_ops > 0) {
        offsets = as_const_pointer<int64_t>(set[QCD_SET_INPUT_OFFSETS]);
        if (offsets == nullptr || offsets[0] < 0) {
            return QCD_STATUS_INVALID;
        }
        for (int32_t index = 0; index < num_ops; ++index) {
            if (offsets[index + 1] < offsets[index]) {
                return QCD_STATUS_INVALID;
            }
        }
        flat_count = offsets[num_ops];
        if (flat_count > kMaxFlatIndices) {
            return QCD_STATUS_INVALID;
        }
        if (flat_count > 0) {
            flat = as_const_pointer<int32_t>(set[QCD_SET_INPUTS_FLAT]);
            if (flat == nullptr) {
                return QCD_STATUS_INVALID;
            }
            for (int64_t index = 0; index < flat_count; ++index) {
                if (flat[index] < 0 || flat[index] >= num_symbols) {
                    return QCD_STATUS_INVALID;
                }
            }
        }
    }

    const int32_t* output = nullptr;
    if (output_ndim > 0) {
        output = as_const_pointer<int32_t>(set[QCD_SET_OUTPUT]);
        if (output == nullptr) {
            return QCD_STATUS_INVALID;
        }
        for (int64_t index = 0; index < output_ndim; ++index) {
            if (output[index] < 0 || output[index] >= num_symbols) {
                return QCD_STATUS_INVALID;
            }
        }
    }

    int64_t* out_cost = as_pointer<int64_t>(set[QCD_SET_OUT_COST]);
    if (out_cost == nullptr) {
        return QCD_STATUS_INVALID;
    }
    const int64_t required_path = num_ops >= 2 ? 2 * (static_cast<int64_t>(num_ops) - 1) : 0;
    int32_t* out_path = nullptr;
    if (required_path > 0) {
        if (path_capacity < required_path) {
            return QCD_STATUS_INVALID;
        }
        out_path = as_pointer<int32_t>(set[QCD_SET_OUT_PATH]);
        if (out_path == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }

    std::vector<pyqcd_einsum::TensorView> tensors(static_cast<std::size_t>(num_ops > 0 ? num_ops : 0));
    for (int32_t index = 0; index < num_ops; ++index) {
        tensors[static_cast<std::size_t>(index)] = pyqcd_einsum::TensorView{
            flat == nullptr ? nullptr : flat + offsets[index],
            static_cast<int32_t>(offsets[index + 1] - offsets[index])};
    }

    pyqcd_einsum::PlanSpec spec{};
    spec.tensors = tensors.empty() ? nullptr : tensors.data();
    spec.num_tensors = num_ops;
    spec.output = output;
    spec.output_count = static_cast<int32_t>(output_ndim);
    spec.sizes = sizes;
    spec.num_symbols = static_cast<int32_t>(num_symbols);
    spec.planner_id = planner_id;
    spec.memory_limited = mem_hi >= 0;
    spec.memory_limit =
        mem_hi < 0 ? -1
                  : ((static_cast<int64_t>(mem_hi) << 32) |
                     static_cast<int64_t>(static_cast<uint32_t>(mem_lo)));

    pyqcd_einsum::PlanOutcome outcome;
    const int status = pyqcd_einsum::plan_contract(spec, &outcome);
    if (status != QCD_STATUS_OK) {
        return status;
    }
    if (static_cast<int64_t>(outcome.path.size()) != required_path) {
        return QCD_STATUS_INTERNAL;
    }
    if (required_path > 0) {
        std::memcpy(out_path, outcome.path.data(), static_cast<std::size_t>(required_path) * sizeof(int32_t));
    }
    out_cost[0] = outcome.opt_cost;
    out_cost[1] = outcome.largest_intermediate;
    return QCD_STATUS_OK;
}

}  // namespace

extern "C" QCD_EINSUM_API int applyPlanQcd(long long set_ptrs, long long params) {
    try {
        return plan_entry(set_ptrs, params);
    } catch (const pyqcd_einsum::PlanBudgetExceeded&) {
        return QCD_STATUS_MEMORY;
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}

/* The planning kernel keeps no session state. These stubs only exist so a
 * planning-only build still exports the frozen ABI; the numeric domain owner
 * replaces them with strong definitions and the linker prefers those. */
extern "C" QCD_EINSUM_API __attribute__((weak)) void applyInitQcd(long long set_ptrs, long long params) {
    (void)set_ptrs;
    (void)params;
}

extern "C" QCD_EINSUM_API __attribute__((weak)) void applyEndQcd(long long set_ptrs, long long params) {
    (void)set_ptrs;
    (void)params;
}
