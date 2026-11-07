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
constexpr int kMaxAccumulators = 4;
constexpr int kE4MinBlocks = 2;
constexpr int64_t kStagedSharedBudget = 48 * 1024;
constexpr int64_t kStagedExtentLimit = int64_t(1) << 30;

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
    __device__ static __forceinline__ void madd(
        float& acc, float left, float right) {
        acc = __fmaf_rn(left, right, acc);
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
    __device__ static __forceinline__ void madd(
        double& acc, double left, double right) {
        acc = __fma_rn(left, right, acc);
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
    __device__ static __forceinline__ void madd(
        cuFloatComplex& acc, cuFloatComplex left, cuFloatComplex right) {
        acc.x = __fmaf_rn(left.x, right.x, __fmaf_rn(-left.y, right.y, acc.x));
        acc.y = __fmaf_rn(left.x, right.y, __fmaf_rn(left.y, right.x, acc.y));
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
    __device__ static __forceinline__ void madd(
        cuDoubleComplex& acc, cuDoubleComplex left, cuDoubleComplex right) {
        acc.x = __fma_rn(left.x, right.x, __fma_rn(-left.y, right.y, acc.x));
        acc.y = __fma_rn(left.x, right.y, __fma_rn(left.y, right.x, acc.y));
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

// One warp per output row: reads every factor through a lane-0 broadcast and
// re-reads the left operand for each (c, g, a, e) step. Fallback path.
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

// Block = (t, b, f-tile). Folds c into a per-block "sums_n[v][g][a][n]" table and
// e into "factors[v][G][g][a][d][f]", then runs one flat (f, n) contraction pass.
template <typename Type, int MaxAcc>
__global__ __launch_bounds__(kE4Threads, kE4MinBlocks) void e4_staged_kernel(
    const Type* __restrict__ a,
    const Type* __restrict__ b,
    const Type* __restrict__ gamma,
    const Type* __restrict__ c,
    Type* __restrict__ dst,
    qcd_cuda_e4_spec spec,
    int nl,
    int fc,
    int nf) {
    extern __shared__ __align__(16) unsigned char raw_shared[];
    Type* shared = reinterpret_cast<Type*>(raw_shared);

    const int volume = static_cast<int>(spec.z * spec.y * spec.x);
    const int rank = static_cast<int>(spec.input_g * spec.a);
    const int ig = static_cast<int>(spec.input_g);
    const int og = static_cast<int>(spec.output_g);
    const int dim_a = static_cast<int>(spec.a);
    const int dim_b = static_cast<int>(spec.b);
    const int dim_c = static_cast<int>(spec.c);
    const int dim_d = static_cast<int>(spec.d);
    const int dim_e = static_cast<int>(spec.e);
    const int dim_f = static_cast<int>(spec.f);
    const int dim_n = static_cast<int>(spec.n);
    const int dim_t = static_cast<int>(spec.t);

    Type* sums_n = shared;
    Type* factors = shared + static_cast<int64_t>(volume) * rank * nl;

    const int f_tile_index = static_cast<int>(blockIdx.x) % nf;
    const int row_unit = static_cast<int>(blockIdx.x) / nf;
    const int b_index = row_unit % dim_b;
    const int t_index = row_unit / dim_b;

    const int64_t a_b = dim_n;
    const int64_t a_a = static_cast<int64_t>(dim_b) * a_b;
    const int64_t a_t = static_cast<int64_t>(dim_a) * a_a;
    const int64_t a_c = static_cast<int64_t>(dim_t) * a_t;
    const int64_t a_v = static_cast<int64_t>(dim_c) * a_c;
    const int64_t c_c = 1;
    const int64_t c_g = dim_c;
    const int64_t c_v = static_cast<int64_t>(ig) * c_g;
    const int64_t c_t = static_cast<int64_t>(volume) * c_v;
    const int64_t b_e = dim_f;
    const int64_t b_d = static_cast<int64_t>(dim_e) * b_e;
    const int64_t b_t = static_cast<int64_t>(dim_d) * b_d;
    const int64_t b_g = static_cast<int64_t>(dim_t) * b_t;
    const int64_t b_v = static_cast<int64_t>(ig) * b_g;
    const int64_t o_f = dim_n;
    const int64_t o_b = static_cast<int64_t>(dim_f) * o_f;
    const int64_t o_d = static_cast<int64_t>(dim_b) * o_b;
    const int64_t o_g = static_cast<int64_t>(dim_d) * o_d;
    const int64_t o_t = static_cast<int64_t>(og) * o_g;

    const int f_base = f_tile_index * fc;
    const int f_remaining = dim_f - f_base;
    const int f_count = fc < f_remaining ? fc : f_remaining;
    const int tid = static_cast<int>(threadIdx.x);

    const int factor_count = volume * og * rank * dim_d * f_count;
    for (int item = tid; item < factor_count; item += kE4Threads) {
        int rest = item;
        const int f_index = rest % f_count;
        rest /= f_count;
        const int d_index = rest % dim_d;
        rest /= dim_d;
        const int a_index = rest % dim_a;
        rest /= dim_a;
        const int g_index = rest % ig;
        rest /= ig;
        const int og_index = rest % og;
        const int v_index = rest / og;

        const int64_t b_offset = static_cast<int64_t>(v_index) * b_v +
                                 static_cast<int64_t>(g_index) * b_g +
                                 static_cast<int64_t>(t_index) * b_t +
                                 static_cast<int64_t>(d_index) * b_d +
                                 (f_base + f_index);
        const int64_t gamma_offset =
            (static_cast<int64_t>(og_index) * dim_a + a_index) * dim_e;
        Type acc = E4Ops<Type>::zero();
        for (int e_index = 0; e_index < dim_e; ++e_index) {
            E4Ops<Type>::madd(acc, gamma[gamma_offset + e_index],
                              b[b_offset + e_index * b_e]);
        }
        const int64_t factor_offset =
            ((((static_cast<int64_t>(v_index) * og + og_index) * ig + g_index) *
                  dim_a +
              a_index) * dim_d + d_index) * f_count + f_index;
        factors[factor_offset] = acc;
    }
    __syncthreads();

    for (int n_base = 0; n_base < dim_n; n_base += nl) {
        const int n_remaining = dim_n - n_base;
        const int n_count = nl < n_remaining ? nl : n_remaining;
        if (tid < n_count) {
            const int n_index = n_base + tid;
            for (int v_index = 0; v_index < volume; ++v_index) {
                const int64_t a_offset_base = static_cast<int64_t>(v_index) * a_v +
                                              static_cast<int64_t>(t_index) * a_t +
                                              static_cast<int64_t>(b_index) * a_b +
                                              n_index;
                for (int g_index = 0; g_index < ig; ++g_index) {
                    const int64_t c_offset = static_cast<int64_t>(t_index) * c_t +
                                             static_cast<int64_t>(v_index) * c_v +
                                             static_cast<int64_t>(g_index) * c_g;
                    for (int a_index = 0; a_index < dim_a; ++a_index) {
                        const int64_t a_offset = a_offset_base +
                                                 static_cast<int64_t>(a_index) * a_a;
                        Type acc = E4Ops<Type>::zero();
                        for (int c_index = 0; c_index < dim_c; ++c_index) {
                            E4Ops<Type>::madd(acc, a[a_offset + c_index * a_c],
                                              c[c_offset + c_index * c_c]);
                        }
                        sums_n[((static_cast<int64_t>(v_index) * ig + g_index) * dim_a +
                                a_index) * nl + tid] = acc;
                    }
                }
            }
        }
        __syncthreads();

        int f_index = tid / n_count;
        int n_lane = tid - f_index * n_count;
        const int factor_stride = dim_d * f_count;
        const int64_t output_base = static_cast<int64_t>(t_index) * o_t +
                                    static_cast<int64_t>(b_index) * o_b + n_base;
        for (; f_index < f_count;) {
            const int64_t output_tile = output_base +
                                        static_cast<int64_t>(f_base + f_index) * o_f +
                                        n_lane;
            for (int og_index = 0; og_index < og; ++og_index) {
                Type acc[MaxAcc];
#pragma unroll
                for (int d_index = 0; d_index < MaxAcc; ++d_index) {
                    acc[d_index] = E4Ops<Type>::zero();
                }
                for (int v_index = 0; v_index < volume; ++v_index) {
                    int sum_offset = (v_index * rank) * nl + n_lane;
                    int factor_offset =
                        ((v_index * og + og_index) * rank) * factor_stride + f_index;
                    for (int pair = 0; pair < rank; ++pair) {
                        const Type value = sums_n[sum_offset];
                        const Type* factor_row = factors + factor_offset;
                        for (int d_index = 0; d_index < dim_d; ++d_index) {
                            E4Ops<Type>::madd(acc[d_index],
                                              factor_row[d_index * f_count], value);
                        }
                        sum_offset += nl;
                        factor_offset += factor_stride;
                    }
                }
                const int64_t output_offset =
                    output_tile + static_cast<int64_t>(og_index) * o_g;
                for (int d_index = 0; d_index < dim_d; ++d_index) {
                    dst[output_offset + static_cast<int64_t>(d_index) * o_d] =
                        acc[d_index];
                }
            }
            n_lane += kE4Threads;
            while (n_lane >= n_count) {
                n_lane -= n_count;
                ++f_index;
            }
        }
        __syncthreads();
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

int multiprocessor_count() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 0;
    }
    static int cached_device = -1;
    static int cached_count = 0;
    if (cached_device == device) {
        return cached_count;
    }
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess ||
        count <= 0) {
        count = 0;
    }
    cached_count = count;
    cached_device = device;
    return count;
}

struct StagedTiles {
    int nl = 0;
    int fc = 0;
    int nf = 0;
};

bool plan_staged_tiles(
    const qcd_cuda_e4_spec& spec,
    int64_t volume,
    int64_t element_bytes,
    StagedTiles* tiles) {
    const int64_t extents[] = {
        spec.z, spec.y, spec.x, spec.c, spec.t, spec.a, spec.b, spec.n,
        spec.input_g, spec.output_g, spec.d, spec.e, spec.f};
    for (int64_t extent : extents) {
        if (extent > kStagedExtentLimit) {
            return false;
        }
    }
    if (spec.d < 1 || spec.d > kMaxAccumulators || spec.a < 1 ||
        spec.input_g < 1 || spec.output_g < 1 || spec.n < 1 || spec.f < 1 ||
        spec.t < 1 || spec.b < 1 || spec.c < 1 || spec.e < 1) {
        return false;
    }
    const int64_t rank = spec.input_g * spec.a;
    if (volume * rank > kStagedExtentLimit) {
        return false;
    }

    int64_t nl = std::min<int64_t>(spec.n, kE4Threads);
    int64_t sums_bytes = volume * rank * nl * element_bytes;
    if (sums_bytes > kStagedSharedBudget / 2) {
        nl = (kStagedSharedBudget / 2) / (volume * rank * element_bytes);
        if (nl < 1) {
            return false;
        }
        nl = std::min<int64_t>(nl, spec.n);
        sums_bytes = volume * rank * nl * element_bytes;
    }
    const int64_t factor_bytes =
        volume * spec.output_g * rank * spec.d * element_bytes;
    int64_t fc_max = (kStagedSharedBudget - sums_bytes) / factor_bytes;
    fc_max = std::min<int64_t>(fc_max, spec.f);
    if (fc_max < 1) {
        return false;
    }

    const int64_t units = spec.t * spec.b;
    const int sm_count = multiprocessor_count();
    const int64_t target_blocks = sm_count > 0 ? 10 * static_cast<int64_t>(sm_count) : 240;
    const int64_t target_tiles = std::max<int64_t>(1, (target_blocks + units - 1) / units);
    int64_t fc = (spec.f + target_tiles - 1) / target_tiles;
    fc = std::max<int64_t>(1, std::min(fc, fc_max));
    const int64_t nf = (spec.f + fc - 1) / fc;
    if (units * nf > kStagedExtentLimit) {
        return false;
    }
    tiles->nl = static_cast<int>(nl);
    tiles->fc = static_cast<int>(fc);
    tiles->nf = static_cast<int>(nf);
    return true;
}

template <typename Type>
int launch_e4_staged(
    const void* a,
    const void* b,
    const void* gamma,
    const void* c,
    void* dst,
    qcd_cuda_e4_spec spec,
    cudaStream_t stream,
    bool* launched) {
    *launched = false;
    const int64_t volume = spec.z * spec.y * spec.x;
    StagedTiles tiles;
    if (!plan_staged_tiles(spec, volume, static_cast<int64_t>(sizeof(Type)), &tiles)) {
        return QCD_STATUS_OK;
    }
    const int64_t shared_elements =
        volume * (static_cast<int64_t>(spec.input_g) * spec.a * tiles.nl +
                  static_cast<int64_t>(spec.output_g) * spec.input_g * spec.a *
                      spec.d * tiles.fc);
    const int64_t blocks = static_cast<int64_t>(spec.t) * spec.b * tiles.nf;
    if (blocks <= 0) {
        return QCD_STATUS_OK;
    }
    const std::size_t shared_bytes =
        static_cast<std::size_t>(shared_elements) * sizeof(Type);
    if (spec.d > 1) {
        e4_staged_kernel<Type, kMaxAccumulators>
            <<<static_cast<unsigned int>(blocks), kE4Threads, shared_bytes, stream>>>(
                static_cast<const Type*>(a), static_cast<const Type*>(b),
                static_cast<const Type*>(gamma), static_cast<const Type*>(c),
                static_cast<Type*>(dst), spec, tiles.nl, tiles.fc, tiles.nf);
    } else {
        e4_staged_kernel<Type, 1>
            <<<static_cast<unsigned int>(blocks), kE4Threads, shared_bytes, stream>>>(
                static_cast<const Type*>(a), static_cast<const Type*>(b),
                static_cast<const Type*>(gamma), static_cast<const Type*>(c),
                static_cast<Type*>(dst), spec, tiles.nl, tiles.fc, tiles.nf);
    }
    const int status = map_cuda_error(cudaGetLastError());
    *launched = status == QCD_STATUS_OK;
    return status;
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
int launch_e4_fallback(
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

template <typename Type>
int launch_e4(
    const void* a,
    const void* b,
    const void* gamma,
    const void* c,
    void* dst,
    qcd_cuda_e4_spec spec,
    cudaStream_t stream) {
    bool launched = false;
    const int status =
        launch_e4_staged<Type>(a, b, gamma, c, dst, spec, stream, &launched);
    if (launched || status != QCD_STATUS_OK) {
        return status;
    }
    return launch_e4_fallback<Type>(a, b, gamma, c, dst, spec, stream);
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
