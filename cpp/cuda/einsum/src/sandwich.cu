#include "einsum_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kSandwichThreads = 256;
constexpr int kWarpSize = 32;
constexpr int64_t kMaxGridBlocks = 65535;
/* Gamma matrices are 4x4 for every physical sandwich, so the hot kernel is
 * specialised for that shape and the generic kernel stays as a fallback. */
constexpr int kSandwichSize = 4;
constexpr int kSandwichMatrixValues = kSandwichSize * kSandwichSize;

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

/* Generic kernel: one warp per (prefix, d, a, lane block).  Every lane of the
 * warp owns one lane index and walks the two matrix products itself. */
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
                    const Type xr = __ldg(tensor + tensor_index);
                    const Type xi = __ldg(tensor + tensor_index + 1);
                    const Type mr = __ldg(left_matrix + matrix_index);
                    const Type mi = __ldg(left_matrix + matrix_index + 1);
                    inner_real += xr * mr - xi * mi;
                    inner_imag += xr * mi + xi * mr;
                } else {
                    inner_real += __ldg(tensor + tensor_index) *
                                  __ldg(left_matrix + matrix_index);
                }
            }

            const int64_t right_matrix_index =
                (right_middle * spec.right + right_index) * kChannels;
            if constexpr (Complex) {
                const Type nr = __ldg(right_matrix + right_matrix_index);
                const Type ni = __ldg(right_matrix + right_matrix_index + 1);
                accumulated_real += inner_real * nr - inner_imag * ni;
                accumulated_imag += inner_real * ni + inner_imag * nr;
            } else {
                accumulated_real += inner_real * __ldg(right_matrix + right_matrix_index);
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

/* Specialised 4x4 kernel: one warp per (prefix, lane block) computes all sixteen
 * (d, a) outputs of that lane, so every propagation element is read exactly
 * once instead of once per output.  ``left_matrix`` is ``[a][b]`` and
 * ``right_matrix`` is ``[c][d]``. */
template <typename Type, bool Complex>
__global__ void sandwich4_kernel(
    const Type* __restrict__ left_matrix,
    const Type* __restrict__ tensor,
    const Type* __restrict__ right_matrix,
    Type* __restrict__ dst,
    qcd_cuda_sandwich_spec spec,
    int64_t tasks,
    int64_t lane_blocks) {
    constexpr int kChannels = Complex ? 2 : 1;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
    const int warps_per_block = static_cast<int>(blockDim.x) / kWarpSize;
    int64_t task = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
    const int64_t task_stride = static_cast<int64_t>(gridDim.x) * warps_per_block;

    /* The two gamma matrices are warp uniform; broadcast them through shared
     * memory once per block instead of re-reading them for every output. */
    constexpr int kMatrixChannels = kSandwichMatrixValues * kChannels;
    __shared__ Type shared_left[kMatrixChannels];
    __shared__ Type shared_right[kMatrixChannels];
    if (threadIdx.x < kMatrixChannels) {
        shared_left[threadIdx.x] = left_matrix[threadIdx.x];
    } else if (threadIdx.x < 2 * kMatrixChannels) {
        const int index = static_cast<int>(threadIdx.x) - kMatrixChannels;
        shared_right[index] = right_matrix[index];
    }
    __syncthreads();

    Type left_real[kSandwichMatrixValues];
    Type left_imag[kSandwichMatrixValues];
    Type right_real[kSandwichMatrixValues];
    Type right_imag[kSandwichMatrixValues];
#pragma unroll
    for (int index = 0; index < kSandwichMatrixValues; ++index) {
        left_real[index] = shared_left[index * kChannels];
        right_real[index] = shared_right[index * kChannels];
        if constexpr (Complex) {
            left_imag[index] = shared_left[index * kChannels + 1];
            right_imag[index] = shared_right[index * kChannels + 1];
        } else {
            left_imag[index] = Type(0);
            right_imag[index] = Type(0);
        }
    }

    for (; task < tasks; task += task_stride) {
        const int64_t lane_block = task % lane_blocks;
        const int64_t prefix_index = task / lane_blocks;
        const int64_t lane_index = lane_block * kWarpSize + lane;
        if (lane_index >= spec.lane || prefix_index >= spec.prefix) {
            continue;
        }
        const int64_t base =
            (prefix_index * kSandwichMatrixValues) * spec.lane * kChannels +
            lane_index * kChannels;

        /* Accumulators for the sixteen (d, a) outputs of this lane. */
        Type output_real[kSandwichMatrixValues];
        Type output_imag[kSandwichMatrixValues];
#pragma unroll
        for (int index = 0; index < kSandwichMatrixValues; ++index) {
            output_real[index] = Type(0);
            output_imag[index] = Type(0);
        }

#pragma unroll
        for (int first = 0; first < kSandwichSize; ++first) {
            /* middle[d] = sum_c M[first][c] * gamma2[c][d] */
            Type middle_real[kSandwichSize];
            Type middle_imag[kSandwichSize];
#pragma unroll
            for (int last = 0; last < kSandwichSize; ++last) {
                Type sum_real = Type(0);
                Type sum_imag = Type(0);
#pragma unroll
                for (int contracted = 0; contracted < kSandwichSize; ++contracted) {
                    const int64_t offset =
                        base + (first * kSandwichSize + contracted) * spec.lane * kChannels;
                    const int matrix = contracted * kSandwichSize + last;
                    if constexpr (Complex) {
                        const Type xr = __ldg(tensor + offset);
                        const Type xi = __ldg(tensor + offset + 1);
                        sum_real += xr * right_real[matrix] - xi * right_imag[matrix];
                        sum_imag += xr * right_imag[matrix] + xi * right_real[matrix];
                    } else {
                        sum_real += __ldg(tensor + offset) * right_real[matrix];
                    }
                }
                middle_real[last] = sum_real;
                middle_imag[last] = sum_imag;
            }
#pragma unroll
            for (int left_index = 0; left_index < kSandwichSize; ++left_index) {
                const int matrix = left_index * kSandwichSize + first;
                const Type left_lr = left_real[matrix];
                const Type left_li = left_imag[matrix];
#pragma unroll
                for (int last = 0; last < kSandwichSize; ++last) {
                    const int target = last * kSandwichSize + left_index;
                    if constexpr (Complex) {
                        output_real[target] += left_lr * middle_real[last] -
                                               left_li * middle_imag[last];
                        output_imag[target] += left_lr * middle_imag[last] +
                                               left_li * middle_real[last];
                    } else {
                        output_real[target] += left_lr * middle_real[last];
                    }
                }
            }
        }

#pragma unroll
        for (int last = 0; last < kSandwichSize; ++last) {
#pragma unroll
            for (int left_index = 0; left_index < kSandwichSize; ++left_index) {
                const int source = last * kSandwichSize + left_index;
                const int64_t offset =
                    base + (last * kSandwichSize + left_index) * spec.lane * kChannels;
                dst[offset] = output_real[source];
                if constexpr (Complex) {
                    dst[offset + 1] = output_imag[source];
                }
            }
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
    if (!multiply_checked(spec.prefix, spec.left, &groups) ||
        !multiply_checked(groups, spec.right, &groups)) {
        return QCD_STATUS_INVALID;
    }
    if (!multiply_checked(groups, spec.lane, &output_elements)) {
        return QCD_STATUS_INVALID;
    }
    if (output_elements == 0) {
        return QCD_STATUS_OK;
    }

    constexpr int kWarpsPerBlock = kSandwichThreads / kWarpSize;
    const bool specialised = spec.left == kSandwichSize &&
                             spec.contracted_left == kSandwichSize &&
                             spec.contracted_right == kSandwichSize &&
                             spec.right == kSandwichSize;
    if (specialised) {
        /* One warp covers all sixteen outputs of a (prefix, lane block) pair. */
        const int64_t lane_blocks = (spec.lane - 1) / kWarpSize + 1;
        int64_t tasks = 1;
        if (!multiply_checked(spec.prefix, lane_blocks, &tasks)) {
            return QCD_STATUS_INVALID;
        }
        const int64_t required_blocks = (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock;
        const int grid_blocks = static_cast<int>(
            std::min<int64_t>(required_blocks, kMaxGridBlocks));
        sandwich4_kernel<Type, Complex>
            <<<grid_blocks, kSandwichThreads, 0, stream>>>(
                static_cast<const Type*>(left_matrix),
                static_cast<const Type*>(tensor),
                static_cast<const Type*>(right_matrix),
                static_cast<Type*>(dst),
                spec,
                tasks,
                lane_blocks);
        return map_cuda_error(cudaGetLastError());
    }

    const int64_t lane_blocks = (spec.lane - 1) / kWarpSize + 1;
    int64_t tasks = 1;
    if (!multiply_checked(groups, lane_blocks, &tasks)) {
        return QCD_STATUS_INVALID;
    }
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
