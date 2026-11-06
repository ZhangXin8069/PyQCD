#include "einsum_runtime.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using pyqcd_einsum::analyze_pair;
    using pyqcd_einsum::PairPlan;
    using pyqcd_einsum::TensorMeta;

    TensorMeta left{{0, 1, 2}, {2, 3, 4}};
    TensorMeta right{{1, 3, 2}, {3, 5, 4}};
    TensorMeta output{{0, 3, 1}, {2, 5, 3}};
    const std::vector<int64_t> sizes{2, 3, 4, 5};

    PairPlan plan;
    require(analyze_pair(left, right, output, sizes, &plan), plan.reason.c_str());

    require(plan.batch_labels == std::vector<int32_t>({1}), "batch label order");
    require(plan.sum_labels == std::vector<int32_t>({2}), "sum label order");
    require(plan.left_perm == std::vector<int32_t>({1, 0, 2}), "left K-last permutation");
    require(plan.right_perm == std::vector<int32_t>({0, 1, 2}), "right K-last permutation");
    require(plan.output_packed_shape == std::vector<int64_t>({3, 2, 5}), "packed output shape");
    require(plan.output_perm == std::vector<int32_t>({1, 2, 0}), "output permutation");
    require(plan.batches == 3 && plan.m == 2 && plan.n == 5 && plan.k == 4, "GEMM dimensions");
    require(plan.left_is_packed, "left input requires K-last packing");
    require(!plan.right_is_packed, "right input is already K-last in this case");

    TensorMeta mm_left{{0, 1}, {512, 512}};
    TensorMeta mm_right{{1, 2}, {512, 512}};
    TensorMeta mm_output{{0, 2}, {512, 512}};
    const std::vector<int64_t> mm_sizes{512, 512, 512};
    PairPlan mm_plan;
    require(analyze_pair(mm_left, mm_right, mm_output, mm_sizes, &mm_plan), mm_plan.reason.c_str());
    require(!mm_plan.left_is_packed && mm_plan.right_is_packed && !mm_plan.output_is_packed,
            "right K-last layout may require one transpose for plain matrix multiplication");

    std::cout << "test_runtime: PASS\n";
    return 0;
}
