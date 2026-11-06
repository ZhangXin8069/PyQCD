#include "einsum_cuda.h"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kE4Threads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxValuesPerLane = 4;
constexpr int64_t kMaxGridBlocks = 65535;

template <typename Type>
struct E4Ops;

template <>
struct E4Ops<float> {
    __device__ static __forceinline__ float zero() { return 0.0f; }
    __device__ static __forceinline__ float add(float left, float right) {
        return left + right;
    }
    __device__ static __forceinline__ float mul(float left, float right) {
        return left * right;
    }
};

template <>
struct E4Ops<double> {
    __device__ static __forceinline__ double zero() { return 0.0; }
    __device__ static __forceinline__ double add(double left, double right) {
        return left + right;
    }
    __device__ static __forceinline__ double mul(double left, double right) {
        return left * right;
    }
};

template <>
struct E4Ops<cuFloatComplex> {
    __device__ static __forceinline__ cuFloatComplex zero() {
        return make_cuFloatComplex(0.0f, 0.0f);
    }
    __device__ static __forceinline__ cuFloatComplex add(
        cuFloatComplex left, cuFloatComplex right) {
        return make_cuFloatComplex(left.x + right.x, left.y + right.y);
    }
    __device__ static __forceinline__ cuFloatComplex mul(
        cuFloatComplex left, cuFloatComplex right) {
        return make_cuFloatComplex(
            left.x * right.x - left.y * right.y,
            left.x * right.y + left.y * right.x);
    }
};

template <>
struct E4Ops<cuDoubleComplex> {
    __device__ static __forceinline__ cuDoubleComplex zero() {
        return make_cuDoubleComplex(0.0, 0.0);
    }
    __device__ static __forceinline__ cuDoubleComplex add(
        cuDoubleComplex left, cuDoubleComplex right) {
        return make_cuDoubleComplex(left.x + right.x, left.y + right.y);
    }
    __device__ static __forceinline__ cuDoubleComplex mul(
        cuDoubleComplex left, cuDoubleComplex right) {
        return make_cuDoubleComplex(
            left.x * right.x - left.y * right.y,
            left.x * right.y + left.y * right.x);
    }
};

__device__ __forceinline__ float broadcast_warp(float value) {
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ double broadcast_warp(double value) {
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ cuFloatComplex broadcast_warp(cuFloatComplex value) {
    return make_cuFloatComplex(
        __shfl_sync(0xffffffffu, value.x, 0),
        __shfl_sync(0xffffffffu, value.y, 0));
}

__device__ __forceinline__ cuDoubleComplex broadcast_warp(cuDoubleComplex value) {
    return make_cuDoubleComplex(
        __shfl_sync(0xffffffffu, value.x, 0),
        __shfl_sync(0xffffffffu, value.y, 0));
}

template <typename Type, int ValuesPerLane>
__global__ __launch_bounds__(kE4Threads) void e4_fused_kernel(
    const Type* a,
    const Type* b,
    const Type* gamma,
    const Type* c,
    Type* dst,
    qcd_cuda_e4_spec spec,
    int64_t tasks,
    int64_t n_tiles) {
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp_in_block = static_cast<int>(threadIdx.x) / kWarpSize;
    constexpr int kWarpsPerBlock = kE4Threads / kWarpSize;
    const int64_t task_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    for (int64_t task = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock +
                        warp_in_block;
         task < tasks; task += task_stride) {
        int64_t row = task / n_tiles;
        const int64_t n_tile = task - row * n_tiles;

        const int64_t f_index = row % spec.f;
        row /= spec.f;
        const int64_t b_index = row % spec.b;
        row /= spec.b;
        const int64_t d_index = row % spec.d;
        row /= spec.d;
        const int64_t output_g_index = row % spec.output_g;
        const int64_t t_index = row / spec.output_g;
        const int64_t n_base =
            n_tile * (kWarpSize * ValuesPerLane) + lane;

        Type sums[ValuesPerLane];
#pragma unroll
        for (int item = 0; item < ValuesPerLane; ++item) {
            sums[item] = E4Ops<Type>::zero();
        }

        for (int64_t z = 0; z < spec.z; ++z) {
            for (int64_t y = 0; y < spec.y; ++y) {
                for (int64_t x = 0; x < spec.x; ++x) {
                    for (int64_t c_index = 0; c_index < spec.c; ++c_index) {
                        const int64_t c_base =
                            (((((t_index * spec.z + z) * spec.y + y) * spec.x + x) *
                                   spec.input_g) *
                                  spec.c) +
                             c_index;
                        for (int64_t input_g_index = 0;
                             input_g_index < spec.input_g; ++input_g_index) {
                            const int64_t c_offset =
                                c_base + input_g_index * spec.c;
                            const int64_t b_base =
                                (((((((z * spec.y + y) * spec.x + x) * spec.input_g +
                                      input_g_index) *
                                         spec.t +
                                     t_index) *
                                        spec.d +
                                    d_index) *
                                   spec.e) *
                                  spec.f) +
                                f_index;
                            for (int64_t a_index = 0; a_index < spec.a; ++a_index) {
                                const int64_t a_base =
                                    (((((((z * spec.y + y) * spec.x + x) * spec.c +
                                          c_index) *
                                             spec.t +
                                         t_index) *
                                            spec.a +
                                        a_index) *
                                           spec.b +
                                       b_index) *
                                      spec.n);
                                const int64_t gamma_base =
                                    (output_g_index * spec.a + a_index) * spec.e;
                                for (int64_t e_index = 0; e_index < spec.e; ++e_index) {
                                    Type factor = E4Ops<Type>::zero();
                                    if (lane == 0) {
                                        const int64_t b_offset =
                                            b_base + e_index * spec.f;
                                        const int64_t gamma_offset =
                                            gamma_base + e_index;
                                        factor = E4Ops<Type>::mul(
                                            E4Ops<Type>::mul(
                                                b[b_offset], gamma[gamma_offset]),
                                            c[c_offset]);
                                    }
                                    factor = broadcast_warp(factor);

#pragma unroll
                                    for (int item = 0; item < ValuesPerLane; ++item) {
                                        const int64_t n_index =
                                            n_base + item * kWarpSize;
                                        if (n_index < spec.n) {
                                            sums[item] = E4Ops<Type>::add(
                                                sums[item],
                                                E4Ops<Type>::mul(
                                                    a[a_base + n_index], factor));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        const int64_t output_row = task / n_tiles;
        const int64_t output_base = output_row * spec.n;
#pragma unroll
        for (int item = 0; item < ValuesPerLane; ++item) {
            const int64_t n_index = n_base + item * kWarpSize;
            if (n_index < spec.n) {
                dst[output_base + n_index] = sums[item];
            }
        }
    }
}

bool multiply_checked(int64_t left, int64_t right, int64_t* product) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool checked_product(const int64_t* values, int count, int64_t* product) {
    int64_t total = 1;
    for (int index = 0; index < count; ++index) {
        if (!multiply_checked(total, values[index], &total)) {
            return false;
        }
    }
    *product = total;
    return true;
}

bool valid_spec(const qcd_cuda_e4_spec& spec) {
    const int64_t dimensions[] = {
        spec.z, spec.y, spec.x, spec.c, spec.t, spec.a, spec.b, spec.n,
        spec.input_g, spec.output_g, spec.d, spec.e, spec.f};
    for (int64_t extent : dimensions) {
        if (extent < 0) {
            return false;
        }
    }

    int64_t elements = 0;
    const int64_t a_shape[] = {
        spec.z, spec.y, spec.x, spec.c, spec.t, spec.a, spec.b, spec.n};
    const int64_t b_shape[] = {
        spec.z, spec.y, spec.x, spec.input_g, spec.t, spec.d, spec.e, spec.f};
    const int64_t gamma_shape[] = {spec.output_g, spec.a, spec.e};
    const int64_t c_shape[] = {
        spec.t, spec.z, spec.y, spec.x, spec.input_g, spec.c};
    const int64_t output_shape[] = {
        spec.t, spec.output_g, spec.d, spec.b, spec.f, spec.n};
    if (!checked_product(a_shape, 8, &elements) ||
        !checked_product(b_shape, 8, &elements) ||
        !checked_product(gamma_shape, 3, &elements) ||
        !checked_product(c_shape, 6, &elements) ||
        !checked_product(output_shape, 6, &elements) ||
        elements != spec.output_elements) {
        return false;
    }
    return true;
}

int map_cuda_error(cudaError_t error) {
    if (error == cudaSuccess) {
        return QCD_STATUS_OK;
    }
    return error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

template <typename Type, int ValuesPerLane>
int launch_e4_tiled(
    const void* a,
    const void* b,
    const void* gamma,
    const void* c,
    void* dst,
    qcd_cuda_e4_spec spec,
    cudaStream_t stream) {
    int64_t rows = 1;
    int64_t temporary = 1;
    if (!multiply_checked(spec.t, spec.output_g, &rows) ||
        !multiply_checked(rows, spec.d, &temporary) ||
        !multiply_checked(temporary, spec.b, &rows) ||
        !multiply_checked(rows, spec.f, &temporary)) {
        return QCD_STATUS_INVALID;
    }
    rows = temporary;
    const int64_t n_tiles =
        spec.n / (kWarpSize * ValuesPerLane) +
        (spec.n % (kWarpSize * ValuesPerLane) != 0 ? 1 : 0);
    int64_t tasks = 0;
    if (!multiply_checked(rows, n_tiles, &tasks)) {
        return QCD_STATUS_INVALID;
    }
    if (tasks == 0) {
        return QCD_STATUS_OK;
    }

    constexpr int kWarpsPerBlock = kE4Threads / kWarpSize;
    const int64_t required_blocks = (tasks + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int grid_blocks = static_cast<int>(
        std::min<int64_t>(required_blocks, kMaxGridBlocks));
    e4_fused_kernel<Type, ValuesPerLane><<<grid_blocks, kE4Threads, 0, stream>>>(
        static_cast<const Type*>(a),
        static_cast<const Type*>(b),
        static_cast<const Type*>(gamma),
        static_cast<const Type*>(c),
        static_cast<Type*>(dst),
        spec,
        tasks,
        n_tiles);
    return map_cuda_error(cudaGetLastError());
}

template <typename Type>
int launch_e4(
    const void* a,
    const void* b,
    const void* gamma,
    const void* c,
    void* dst,
    qcd_cuda_e4_spec spec,
    cudaStream_t stream) {
    if (spec.n <= kWarpSize) {
        return launch_e4_tiled<Type, 1>(a, b, gamma, c, dst, spec, stream);
    }
    if (spec.n <= 2 * kWarpSize) {
        return launch_e4_tiled<Type, 2>(a, b, gamma, c, dst, spec, stream);
    }
    if (spec.n <= 3 * kWarpSize) {
        return launch_e4_tiled<Type, 3>(a, b, gamma, c, dst, spec, stream);
    }
    static_assert(kMaxValuesPerLane == 4, "E4 lane specialization count drifted");
    return launch_e4_tiled<Type, kMaxValuesPerLane>(
        a, b, gamma, c, dst, spec, stream);
}

}  // namespace

extern "C" QCD_CUDA_LOCAL int qcd_cuda_e4_fused(
    int dtype_id,
    const void* a,
    const void* b,
    const void* gamma,
    const void* c,
    void* dst,
    qcd_cuda_e4_spec spec,
    cudaStream_t stream) {
    if (!valid_spec(spec)) {
        return QCD_STATUS_INVALID;
    }
    if (spec.output_elements > 0 &&
        (a == nullptr || b == nullptr || gamma == nullptr || c == nullptr ||
         dst == nullptr)) {
        return QCD_STATUS_INVALID;
    }
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            return launch_e4<cuDoubleComplex>(a, b, gamma, c, dst, spec, stream);
        case QCD_DTYPE_COMPLEX64:
            return launch_e4<cuFloatComplex>(a, b, gamma, c, dst, spec, stream);
        case QCD_DTYPE_FLOAT64:
            return launch_e4<double>(a, b, gamma, c, dst, spec, stream);
        case QCD_DTYPE_FLOAT32:
            return launch_e4<float>(a, b, gamma, c, dst, spec, stream);
        default:
            return QCD_STATUS_INVALID;
    }
}
