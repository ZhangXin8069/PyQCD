#ifndef PYQCD_EINSUM_PLAN_H
#define PYQCD_EINSUM_PLAN_H

#include "einsum_define.h"

#include <cstdint>
#include <vector>

namespace pyqcd_einsum {

struct TensorView {
    const int32_t* symbols;
    int32_t count;
};

struct PlanSpec {
    const TensorView* tensors;
    int32_t num_tensors;
    const int32_t* output;
    int32_t output_count;
    const int64_t* sizes;
    int32_t num_symbols;
    int32_t planner_id;
    bool memory_limited;
    int64_t memory_limit;
};

struct PlanOutcome {
    std::vector<int32_t> path;
    int64_t opt_cost = 0;
    int64_t largest_intermediate = 1;
};

class PlanBudgetExceeded final {};

int plan_contract(const PlanSpec& spec, PlanOutcome* outcome);

}  // namespace pyqcd_einsum

#endif
