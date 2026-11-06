#include "einsum_cuda.h"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kPackThreads = 256;
constexpr int64_t kMaxGridBlocks = 65535;
constexpr int kTransposeTile = 32;

struct TransposeAxes {
    int32_t inner = -1;
    int32_t outer = -1;
};

template <typename Type>
struct ScalarTraits;

template <>
struct ScalarTraits<float> {
    __device__ static __forceinline__ float zero() { return 0.0f; }
    __device__ static __forceinline__ float mul(float left, float right) { return left * right; }
    __device__ static __forceinline__ float add(float left, float right) { return left + right; }
};

template <>
struct ScalarTraits<double> {
    __device__ static __forceinline__ double zero() { return 0.0; }
    __device__ static __forceinline__ double mul(double left, double right) { return left * right; }
    __device__ static __forceinline__ double add(double left, double right) { return left + right; }
};

template <>
struct ScalarTraits<cuFloatComplex> {
    __device__ static __forceinline__ cuFloatComplex zero() { return make_cuFloatComplex(0.0f, 0.0f); }
    __device__ static __forceinline__ cuFloatComplex mul(cuFloatComplex left, cuFloatComplex right) {
        return cuCmulf(left, right);
    }
    __device__ static __forceinline__ cuFloatComplex add(cuFloatComplex left, cuFloatComplex right) {
        return cuCaddf(left, right);
    }
};

template <>
struct ScalarTraits<cuDoubleComplex> {
    __device__ static __forceinline__ cuDoubleComplex zero() {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    __device__ static __forceinline__ cuDoubleComplex mul(cuDoubleComplex left, cuDoubleComplex right) {
        return cuCmul(left, right);
    }
    __device__ static __forceinline__ cuDoubleComplex add(cuDoubleComplex left, cuDoubleComplex right) {
        return cuCadd(left, right);
    }
};

__device__ __forceinline__ int64_t source_offset(
    int64_t linear, const int64_t* shape, const int64_t* stride, int32_t rank) {
    int64_t offset = 0;
    for (int32_t axis = rank - 1; axis >= 0; --axis) {
        const int64_t extent = shape[axis];
        const int64_t coordinate = linear % extent;
        linear /= extent;
        offset += coordinate * stride[axis];
    }
    return offset;
}

template <typename Type>
__global__ void pack_kernel(
    const Type* src, Type* dst, qcd_cuda_copy_spec spec) {
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < spec.elements; index += step) {
        dst[index] = src[source_offset(index, spec.shape, spec.src_stride, spec.rank)];
    }
}

template <typename Type>
__global__ void transpose_pack_kernel(
    const Type* src,
    Type* dst,
    qcd_cuda_copy_spec spec,
    TransposeAxes axes) {
    __shared__ Type tile[kTransposeTile][kTransposeTile + 1];

    int64_t inner_stride = 1;
    int64_t outer_stride = 1;
    for (int32_t axis = axes.inner + 1; axis < spec.rank; ++axis) {
        inner_stride *= spec.shape[axis];
    }
    for (int32_t axis = axes.outer + 1; axis < spec.rank; ++axis) {
        outer_stride *= spec.shape[axis];
    }

    int64_t batch_index = blockIdx.z;
    int64_t source_base = 0;
    int64_t destination_base = 0;
    int64_t destination_stride = 1;
    for (int32_t axis = spec.rank - 1; axis >= 0; --axis) {
        if (axis == axes.inner || axis == axes.outer) {
            destination_stride *= spec.shape[axis];
            continue;
        }
        const int64_t coordinate = batch_index % spec.shape[axis];
        batch_index /= spec.shape[axis];
        source_base += coordinate * spec.src_stride[axis];
        destination_base += coordinate * destination_stride;
        destination_stride *= spec.shape[axis];
    }

    const int64_t source_inner = static_cast<int64_t>(blockIdx.x) * kTransposeTile +
                                 threadIdx.x;
    for (int32_t local_outer = threadIdx.y; local_outer < kTransposeTile;
         local_outer += blockDim.y) {
        const int64_t source_outer =
            static_cast<int64_t>(blockIdx.y) * kTransposeTile + local_outer;
        if (source_inner < spec.shape[axes.inner] &&
            source_outer < spec.shape[axes.outer]) {
            const int64_t offset = source_base +
                                   source_inner * spec.src_stride[axes.inner] +
                                   source_outer * spec.src_stride[axes.outer];
            tile[local_outer][threadIdx.x] = src[offset];
        }
    }
    __syncthreads();

    const int64_t destination_outer = static_cast<int64_t>(blockIdx.y) *
                                      kTransposeTile + threadIdx.x;
    for (int32_t local_inner = threadIdx.y; local_inner < kTransposeTile;
         local_inner += blockDim.y) {
        const int64_t destination_inner =
            static_cast<int64_t>(blockIdx.x) * kTransposeTile + local_inner;
        if (destination_inner < spec.shape[axes.inner] &&
            destination_outer < spec.shape[axes.outer]) {
            const int64_t offset = destination_base +
                                   destination_inner * inner_stride +
                                   destination_outer * outer_stride;
            dst[offset] = tile[threadIdx.x][local_inner];
        }
    }
}

template <typename Type>
__global__ void hadamard_kernel(
    const Type* left, const Type* right, Type* dst, qcd_cuda_hadamard_spec spec) {
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < spec.elements; index += step) {
        int64_t linear = index;
        int64_t left_offset = 0;
        int64_t right_offset = 0;
        for (int32_t axis = spec.rank - 1; axis >= 0; --axis) {
            const int64_t coordinate = linear % spec.shape[axis];
            linear /= spec.shape[axis];
            left_offset += coordinate * spec.left_stride[axis];
            right_offset += coordinate * spec.right_stride[axis];
        }
        dst[index] = ScalarTraits<Type>::mul(left[left_offset], right[right_offset]);
    }
}

template <typename Type>
__global__ void reduce_kernel(
    const Type* src, Type* dst, qcd_cuda_reduce_spec spec) {
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_index < spec.output_elements; output_index += step) {
        int64_t linear = output_index;
        int64_t base = 0;
        for (int32_t axis = spec.output_rank - 1; axis >= 0; --axis) {
            const int64_t coordinate = linear % spec.output_shape[axis];
            linear /= spec.output_shape[axis];
            base += coordinate * spec.output_src_stride[axis];
        }

        Type sum = ScalarTraits<Type>::zero();
        for (int64_t reduction_index = 0; reduction_index < spec.reduction_elements;
             ++reduction_index) {
            int64_t reduction_linear = reduction_index;
            int64_t offset = base;
            for (int32_t axis = spec.reduction_rank - 1; axis >= 0; --axis) {
                const int64_t coordinate = reduction_linear % spec.reduction_extent[axis];
                reduction_linear /= spec.reduction_extent[axis];
                offset += coordinate * spec.reduction_src_stride[axis];
            }
            sum = ScalarTraits<Type>::add(sum, src[offset]);
        }
        dst[output_index] = sum;
    }
}

bool checked_product(const int64_t* values, int32_t count, int64_t* product) {
    int64_t total = 1;
    for (int32_t index = 0; index < count; ++index) {
        if (values[index] < 0 || (values[index] != 0 &&
                                  total > std::numeric_limits<int64_t>::max() / values[index])) {
            return false;
        }
        total *= values[index];
    }
    *product = total;
    return true;
}

bool valid_copy_spec(const qcd_cuda_copy_spec& spec) {
    if (spec.rank < 0 || spec.rank > QCD_CUDA_MAX_PACKED_RANK || spec.elements < 0) {
        return false;
    }
    int64_t product = 0;
    if (!checked_product(spec.shape, spec.rank, &product) || product != spec.elements) {
        return false;
    }
    for (int32_t index = 0; index < spec.rank; ++index) {
        if (spec.src_stride[index] < 0) {
            return false;
        }
    }
    return true;
}

bool valid_hadamard_spec(const qcd_cuda_hadamard_spec& spec) {
    if (spec.rank < 0 || spec.rank > QCD_CUDA_MAX_RANK || spec.elements < 0) {
        return false;
    }
    int64_t product = 0;
    if (!checked_product(spec.shape, spec.rank, &product) || product != spec.elements) {
        return false;
    }
    for (int32_t index = 0; index < spec.rank; ++index) {
        if (spec.left_stride[index] < 0 || spec.right_stride[index] < 0) {
            return false;
        }
    }
    return true;
}

bool valid_reduce_spec(const qcd_cuda_reduce_spec& spec) {
    if (spec.output_rank < 0 || spec.output_rank > QCD_CUDA_MAX_RANK ||
        spec.reduction_rank < 0 || spec.reduction_rank > QCD_CUDA_MAX_RANK ||
        spec.output_elements < 0 || spec.reduction_elements < 0) {
        return false;
    }
    int64_t product = 0;
    if (!checked_product(spec.output_shape, spec.output_rank, &product) ||
        product != spec.output_elements ||
        !checked_product(spec.reduction_extent, spec.reduction_rank, &product) ||
        product != spec.reduction_elements) {
        return false;
    }
    for (int32_t index = 0; index < spec.output_rank; ++index) {
        if (spec.output_src_stride[index] < 0) {
            return false;
        }
    }
    for (int32_t index = 0; index < spec.reduction_rank; ++index) {
        if (spec.reduction_src_stride[index] < 0) {
            return false;
        }
    }
    return true;
}

unsigned int grid_size(int64_t elements) {
    const int64_t blocks = (elements + kPackThreads - 1) / kPackThreads;
    return static_cast<unsigned int>(blocks > kMaxGridBlocks ? kMaxGridBlocks : blocks);
}

bool find_transpose_axes(const qcd_cuda_copy_spec& spec, TransposeAxes* axes) {
    int32_t inner = -1;
    int64_t inner_stride = 1;
    for (int32_t axis = 0; axis < spec.rank; ++axis) {
        if (spec.shape[axis] > 1 && spec.src_stride[axis] == 1) {
            inner = axis;
            inner_stride = spec.src_stride[axis];
            break;
        }
    }
    if (inner < 0) {
        return false;
    }

    int32_t outer = -1;
    int64_t best_stride = 0;
    for (int32_t axis = inner + 1; axis < spec.rank; ++axis) {
        const int64_t stride = spec.src_stride[axis];
        if (spec.shape[axis] <= 1 || stride <= inner_stride) {
            continue;
        }
        if (outer < 0 || stride < best_stride) {
            outer = axis;
            best_stride = stride;
        }
    }
    if (outer < 0) {
        return false;
    }
    axes->inner = inner;
    axes->outer = outer;
    return true;
}

int cuda_status(cudaError_t error) {
    return error == cudaSuccess ? QCD_STATUS_OK : QCD_STATUS_INTERNAL;
}

template <typename Type>
int launch_pack(
    const void* src, void* dst, qcd_cuda_copy_spec spec, cudaStream_t stream) {
    if (spec.elements == 0) {
        return QCD_STATUS_OK;
    }
    TransposeAxes axes;
    if (find_transpose_axes(spec, &axes)) {
        const int64_t inner_extent = spec.shape[axes.inner];
        const int64_t outer_extent = spec.shape[axes.outer];
        const int64_t matrix_elements = inner_extent * outer_extent;
        const int64_t batches = spec.elements / matrix_elements;
        const unsigned int tiles_inner = static_cast<unsigned int>(
            (inner_extent + kTransposeTile - 1) / kTransposeTile);
        const unsigned int tiles_outer = static_cast<unsigned int>(
            (outer_extent + kTransposeTile - 1) / kTransposeTile);
        if (matrix_elements > 1 && batches > 1 && batches <= kMaxGridBlocks &&
            tiles_inner <= kMaxGridBlocks && tiles_outer <= kMaxGridBlocks) {
            const dim3 threads(kTransposeTile, kTransposeTile / 4);
            const dim3 blocks(tiles_inner, tiles_outer, static_cast<unsigned int>(batches));
            transpose_pack_kernel<Type><<<blocks, threads, 0, stream>>>(
                static_cast<const Type*>(src), static_cast<Type*>(dst), spec, axes);
            return cuda_status(cudaGetLastError());
        }
    }
    pack_kernel<Type><<<grid_size(spec.elements), kPackThreads, 0, stream>>>(
        static_cast<const Type*>(src), static_cast<Type*>(dst), spec);
    return cuda_status(cudaGetLastError());
}

template <typename Type>
int launch_hadamard(
    const void* left, const void* right, void* dst, qcd_cuda_hadamard_spec spec,
    cudaStream_t stream) {
    if (spec.elements == 0) {
        return QCD_STATUS_OK;
    }
    hadamard_kernel<Type><<<grid_size(spec.elements), kPackThreads, 0, stream>>>(
        static_cast<const Type*>(left), static_cast<const Type*>(right), static_cast<Type*>(dst), spec);
    return cuda_status(cudaGetLastError());
}

template <typename Type>
int launch_reduce(
    const void* src, void* dst, qcd_cuda_reduce_spec spec, cudaStream_t stream) {
    if (spec.output_elements == 0) {
        return QCD_STATUS_OK;
    }
    reduce_kernel<Type><<<grid_size(spec.output_elements), kPackThreads, 0, stream>>>(
        static_cast<const Type*>(src), static_cast<Type*>(dst), spec);
    return cuda_status(cudaGetLastError());
}

}  // namespace

extern "C" QCD_CUDA_LOCAL int qcd_cuda_permute(
    int dtype_id, const void* src, void* dst, qcd_cuda_copy_spec spec,
    cudaStream_t stream) {
    if (!valid_copy_spec(spec)) {
        return QCD_STATUS_INVALID;
    }
    if (spec.elements > 0 && (src == nullptr || dst == nullptr)) {
        return QCD_STATUS_INVALID;
    }
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            return launch_pack<cuDoubleComplex>(src, dst, spec, stream);
        case QCD_DTYPE_COMPLEX64:
            return launch_pack<cuFloatComplex>(src, dst, spec, stream);
        case QCD_DTYPE_FLOAT64:
            return launch_pack<double>(src, dst, spec, stream);
        case QCD_DTYPE_FLOAT32:
            return launch_pack<float>(src, dst, spec, stream);
        default:
            return QCD_STATUS_INVALID;
    }
}

extern "C" QCD_CUDA_LOCAL int qcd_cuda_hadamard(
    int dtype_id, const void* left, const void* right, void* dst,
    qcd_cuda_hadamard_spec spec, cudaStream_t stream) {
    if (!valid_hadamard_spec(spec)) {
        return QCD_STATUS_INVALID;
    }
    if (spec.elements > 0 && (left == nullptr || right == nullptr || dst == nullptr)) {
        return QCD_STATUS_INVALID;
    }
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            return launch_hadamard<cuDoubleComplex>(left, right, dst, spec, stream);
        case QCD_DTYPE_COMPLEX64:
            return launch_hadamard<cuFloatComplex>(left, right, dst, spec, stream);
        case QCD_DTYPE_FLOAT64:
            return launch_hadamard<double>(left, right, dst, spec, stream);
        case QCD_DTYPE_FLOAT32:
            return launch_hadamard<float>(left, right, dst, spec, stream);
        default:
            return QCD_STATUS_INVALID;
    }
}

extern "C" QCD_CUDA_LOCAL int qcd_cuda_reduce(
    int dtype_id, const void* src, void* dst, qcd_cuda_reduce_spec spec,
    cudaStream_t stream) {
    if (!valid_reduce_spec(spec)) {
        return QCD_STATUS_INVALID;
    }
    if (spec.output_elements > 0 && (src == nullptr || dst == nullptr)) {
        return QCD_STATUS_INVALID;
    }
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            return launch_reduce<cuDoubleComplex>(src, dst, spec, stream);
        case QCD_DTYPE_COMPLEX64:
            return launch_reduce<cuFloatComplex>(src, dst, spec, stream);
        case QCD_DTYPE_FLOAT64:
            return launch_reduce<double>(src, dst, spec, stream);
        case QCD_DTYPE_FLOAT32:
            return launch_reduce<float>(src, dst, spec, stream);
        default:
            return QCD_STATUS_INVALID;
    }
}
