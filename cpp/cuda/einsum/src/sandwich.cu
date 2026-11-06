#include "einsum_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kSandwichThreads = 256;
constexpr int kWarpSize = 32;
constexpr int64_t kMaxGridBlocks = 65535;

int map_cuda_error(cudaError_t error) {
    if (error == cudaSuccess) {
        return QCD_STATUS_OK;
    }
    return error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

bool multiply_checked(int64_t left, int64_t right, int64_t* product) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool valid_spec(const qcd_cuda_sandwich_spec& spec) {
    if (spec.prefix <= 0 || spec.left <= 0 || spec.contracted_left <= 0 ||
        spec.contracted_right <= 0 || spec.right <= 0 || spec.lane <= 0) {
        return false;
    }
    int64_t elements = 1;
    int64_t temporary = 0;
    if (!multiply_checked(spec.prefix, spec.contracted_left, &elements) ||
        !multiply_checked(elements, spec.contracted_right, &temporary) ||
        !multiply_checked(temporary, spec.lane, &elements) ||
        !multiply_checked(spec.left, spec.contracted_left, &elements) ||
        !multiply_checked(spec.contracted_right, spec.right, &elements) ||
        !multiply_checked(spec.prefix, spec.right, &elements) ||
        !multiply_checked(elements, spec.left, &temporary)) {
        return false;
    }
    return multiply_checked(temporary, spec.lane, &elements);
}

template <typename Type, bool Complex>
__global__ void sandwich_kernel(
    const Type* left_matrix,
    const Type* tensor,
    const Type* right_matrix,
    Type* dst,
    qcd_cuda_sandwich_spec spec,
    int64_t tasks,
    int64_t lane_blocks) {
    constexpr int kChannels = Complex ? 2 : 1;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
    int64_t task = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
    const int64_t task_stride =
        static_cast<int64_t>(gridDim.x) * warps_per_block;

    for (; task < tasks; task += task_stride) {
        const int64_t lane_block = task % lane_blocks;
        int64_t group = task / lane_blocks;
        const int64_t left_index = group % spec.left;
        group /= spec.left;
        const int64_t right_index = group % spec.right;
        const int64_t prefix_index = group / spec.right;
        const int64_t lane_index = lane_block * kWarpSize + lane;
        if (lane_index >= spec.lane || prefix_index >= spec.prefix) {
            continue;
        }

        Type accumulated_real = Type(0);
        Type accumulated_imag = Type(0);
        for (int64_t right_middle = 0; right_middle < spec.contracted_right;
             ++right_middle) {
            Type inner_real = Type(0);
            Type inner_imag = Type(0);
            for (int64_t left_middle = 0; left_middle < spec.contracted_left;
                 ++left_middle) {
                const int64_t tensor_index =
                    (((prefix_index * spec.contracted_left + left_middle) *
                          spec.contracted_right +
                      right_middle) *
                         spec.lane +
                     lane_index) *
                    kChannels;
                const int64_t matrix_index =
                    (left_index * spec.contracted_left + left_middle) * kChannels;
                if constexpr (Complex) {
                    const Type xr = tensor[tensor_index];
                    const Type xi = tensor[tensor_index + 1];
                    const Type mr = left_matrix[matrix_index];
                    const Type mi = left_matrix[matrix_index + 1];
                    inner_real += xr * mr - xi * mi;
                    inner_imag += xr * mi + xi * mr;
                } else {
                    inner_real +=
                        tensor[tensor_index] * left_matrix[matrix_index];
                }
            }

            const int64_t right_matrix_index =
                (right_middle * spec.right + right_index) * kChannels;
            if constexpr (Complex) {
                const Type nr = right_matrix[right_matrix_index];
                const Type ni = right_matrix[right_matrix_index + 1];
                accumulated_real += inner_real * nr - inner_imag * ni;
                accumulated_imag += inner_real * ni + inner_imag * nr;
            } else {
                accumulated_real +=
                    inner_real * right_matrix[right_matrix_index];
            }
        }

        const int64_t output_index =
            (((prefix_index * spec.right + right_index) * spec.left + left_index) *
                 spec.lane +
             lane_index) *
            kChannels;
        dst[output_index] = accumulated_real;
        if constexpr (Complex) {
            dst[output_index + 1] = accumulated_imag;
        }
    }
}

template <typename Type, bool Complex>
int launch_sandwich(
    const void* left_matrix,
    const void* tensor,
    const void* right_matrix,
    void* dst,
    qcd_cuda_sandwich_spec spec,
    cudaStream_t stream) {
    int64_t groups = 1;
    int64_t output_elements = 1;
    const int64_t lane_blocks = (spec.lane - 1) / kWarpSize + 1;
    int64_t tasks = 1;
    if (!multiply_checked(groups, spec.prefix, &groups) ||
        !multiply_checked(groups, spec.right, &groups) ||
        !multiply_checked(groups, spec.left, &groups)) {
        return QCD_STATUS_INVALID;
    }
    if (!multiply_checked(groups, lane_blocks, &tasks) ||
        !multiply_checked(groups, spec.lane, &output_elements)) {
        return QCD_STATUS_INVALID;
    }
    if (output_elements == 0) {
        return QCD_STATUS_OK;
    }

    constexpr int kWarpsPerBlock = kSandwichThreads / kWarpSize;
    const int64_t required_blocks = (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int grid_blocks = static_cast<int>(
        std::min<int64_t>(required_blocks, kMaxGridBlocks));
    sandwich_kernel<Type, Complex><<<grid_blocks, kSandwichThreads, 0, stream>>>(
        static_cast<const Type*>(left_matrix),
        static_cast<const Type*>(tensor),
        static_cast<const Type*>(right_matrix),
        static_cast<Type*>(dst),
        spec,
        tasks,
        lane_blocks);
    return map_cuda_error(cudaGetLastError());
}

}  // namespace

extern "C" QCD_CUDA_LOCAL int qcd_cuda_sandwich(
    int dtype_id,
    const void* left_matrix,
    const void* tensor,
    const void* right_matrix,
    void* dst,
    qcd_cuda_sandwich_spec spec,
    cudaStream_t stream) {
    if (!valid_spec(spec) || left_matrix == nullptr || tensor == nullptr ||
        right_matrix == nullptr || dst == nullptr) {
        return QCD_STATUS_INVALID;
    }
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            return launch_sandwich<double, true>(
                left_matrix, tensor, right_matrix, dst, spec, stream);
        case QCD_DTYPE_COMPLEX64:
            return launch_sandwich<float, true>(
                left_matrix, tensor, right_matrix, dst, spec, stream);
        case QCD_DTYPE_FLOAT64:
            return launch_sandwich<double, false>(
                left_matrix, tensor, right_matrix, dst, spec, stream);
        case QCD_DTYPE_FLOAT32:
            return launch_sandwich<float, false>(
                left_matrix, tensor, right_matrix, dst, spec, stream);
        default:
            return QCD_STATUS_INVALID;
    }
}
