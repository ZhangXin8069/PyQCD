/*
 * Pattern-specific CUDA kernel for the sush distillation 3pt main term.
 *
 *   variant 0 (P1)  agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy
 *   variant 1 (P2)  agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy
 *                   -> MNGLxy
 *
 * Why this file exists
 * --------------------
 * The generic path executes the 14 pairwise steps of the dp plan: every step
 * packs (transposes) both operands and then calls a batched GEMM, so the 4x4
 * gamma contractions become K=4 GEMMs and the floating point work is dominated
 * by packing traffic and launch latency rather than by arithmetic.
 *
 * This file executes one fixed fusion plan per variant instead:
 *
 *   1. the projectors and the gamma matrices are folded into the five
 *      perambulator slices once (six tiny kernels),
 *   2. the four propagator vertices (sink_VVV, VdV_link, source_VdV,
 *      source_VVV) are absorbed into a single running "carrier" tensor in the
 *      cost-optimal order,
 *   3. the last stage (only spin*proj^2*mom^2*link outputs) is a block-wide
 *      reduction because the nev^2 reduction is far longer than the output.
 *
 * The walk order is the dp order of the same equation, so the plan does not add
 * arithmetic - it removes packing, K=4 GEMMs and intermediate round trips:
 *
 *   P1 (open Dirac chain x-A-g5-E-g7kq-...-D-W-C-y):
 *     folds -> S -> E -> g7kq -> T -> B -> V -> D -> W -> C(final)
 *       cost  8N^3 + 32N^3 + 8N^3 + 8N^4 + 32N^4 + 8N^4 + 32N^3 + 16N^3
 *   P2 (the g7/g7 loop B-D has to be closed first, so the walk starts at D):
 *     folds -> T -> B -> V -> A -> S -> E -> W -> C(final)
 *       cost 16N^4 + 16N^4 + N^4 + 8N^3 + 8N^3 + 32N^3 + 16N^3
 *
 * at N=nev=50, mom=1, link=2 that is 3.1e8 (P1) / 2.2e8 (P2) complex MACs,
 * identical to the dp plan cost of the same 15-operand equation.
 *
 * Every walk step is "out[batch,j] = sum_k left[batch,k] * right[batch,k,j]"
 * with arbitrary per-axis strides, which is implemented once (gc_kernel) and
 * reused by all 15/14 stages.  Layouts are declared next to each stage in
 * [slowest]..[fastest] order.
 */

/*
 * NOTE on the include below: `make-nv.sh` passes the KUPL include directory
 * *before* this backend's include directory, and `cpp/kupl/einsum/include`
 * contains a different `pattern_3pt.h` (the CPU/KUPL ABI: no
 * QCD_PATTERN_3PT_API and no qcd_cuda_pattern_3pt).  A plain
 * #include "../include/pattern_3pt.h" therefore resolves to the KUPL header and the whole
 * translation unit fails to parse at the first QCD_PATTERN_3PT_API token.
 * Quoting the path relative to this file pins the lookup to
 * cpp/cuda/einsum/include/pattern_3pt.h regardless of the -I order.
 */
#include "../include/pattern_3pt.h"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBatch = 6;
constexpr int kMaxK = 3;
constexpr int kMaxJ = 3;

using cplx = double2;

__device__ __forceinline__ void cmac(cplx& acc, cplx a, cplx b) {
    acc.x = fma(a.x, b.x, fma(-a.y, b.y, acc.x));
    acc.y = fma(a.x, b.y, fma(a.y, b.x, acc.y));
}

/*
 * One axis of a contraction stage.
 *
 *   batch axes live in both operands (a stride of 0 means "absent") and in the
 *   output, ``k`` axes are summed, ``j`` axes are free and come from the right
 *   operand only.
 */
struct gc_axis {
    long long extent;
    long long sa;   /* stride in the left operand  (batch/k)  */
    long long sb;   /* stride in the right operand (batch/k/j) */
    long long so;   /* stride in the output        (batch/j)  */
};

struct gc_spec {
    int nb;
    int nk;
    int nj;
    gc_axis batch[kMaxBatch];
    gc_axis k[kMaxK];
    gc_axis j[kMaxJ];
};

struct gc_builder {
    gc_spec spec;

    gc_builder() {
        spec.nb = 0;
        spec.nk = 0;
        spec.nj = 0;
    }
    void batch(long long extent, long long sa, long long sb, long long so) {
        spec.batch[spec.nb++] = gc_axis{extent, sa, sb, so};
    }
    void k(long long extent, long long sa, long long sb) {
        spec.k[spec.nk++] = gc_axis{extent, sa, sb, 0};
    }
    void j(long long extent, long long sb, long long so) {
        spec.j[spec.nj++] = gc_axis{extent, 0, sb, so};
    }
};

long long product_of_batch(const gc_spec& spec) {
    long long total = 1;
    for (int i = 0; i < spec.nb; ++i) {
        total *= spec.batch[i].extent;
    }
    return total;
}

long long product_of_j(const gc_spec& spec) {
    long long total = 1;
    for (int i = 0; i < spec.nj; ++i) {
        total *= spec.j[i].extent;
    }
    return total;
}

/*
 * out[batch, j] = sum_k left[batch, k] * right[batch, k, j]
 *
 * One thread per output element; the k loop is a compile-time sized nest so the
 * inner loop is a plain pointer walk.  Axes are declared slowest first, which
 * also fixes the linearisation used for the thread mapping.  Because complex128
 * runs at 1/64 of the fp64 rate on this class of GPU the loop is arithmetically
 * bound even with two loads per MAC, so no shared-memory tiling is needed.
 */
template <int NK>
__global__ __launch_bounds__(kThreads) void gc_kernel(
    const cplx* __restrict__ left,
    const cplx* __restrict__ right,
    cplx* __restrict__ out,
    gc_spec spec,
    long long total_left,
    long long total_right) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         index < total_left * total_right; index += stride) {
        long long j_index = index % total_right;
        long long b_index = index / total_right;

        long long offset_left = 0;
        long long offset_right = 0;
        long long offset_out = 0;
        for (int axis = spec.nb - 1; axis >= 0; --axis) {
            const long long extent = spec.batch[axis].extent;
            const long long digit = b_index % extent;
            b_index /= extent;
            offset_left += digit * spec.batch[axis].sa;
            offset_right += digit * spec.batch[axis].sb;
            offset_out += digit * spec.batch[axis].so;
        }
        for (int axis = spec.nj - 1; axis >= 0; --axis) {
            const long long extent = spec.j[axis].extent;
            const long long digit = j_index % extent;
            j_index /= extent;
            offset_right += digit * spec.j[axis].sb;
            offset_out += digit * spec.j[axis].so;
        }

        cplx acc = make_double2(0.0, 0.0);
        if (NK == 1) {
            long long index_left = offset_left;
            long long index_right = offset_right;
            const long long n0 = spec.k[0].extent;
            const long long s0a = spec.k[0].sa;
            const long long s0b = spec.k[0].sb;
            for (long long i0 = 0; i0 < n0; ++i0) {
                cmac(acc, left[index_left], right[index_right]);
                index_left += s0a;
                index_right += s0b;
            }
        } else if (NK == 2) {
            long long base_left = offset_left;
            long long base_right = offset_right;
            const long long n0 = spec.k[0].extent;
            const long long n1 = spec.k[1].extent;
            const long long s0a = spec.k[0].sa, s0b = spec.k[0].sb;
            const long long s1a = spec.k[1].sa, s1b = spec.k[1].sb;
            for (long long i0 = 0; i0 < n0; ++i0) {
                long long index_left = base_left;
                long long index_right = base_right;
                for (long long i1 = 0; i1 < n1; ++i1) {
                    cmac(acc, left[index_left], right[index_right]);
                    index_left += s1a;
                    index_right += s1b;
                }
                base_left += s0a;
                base_right += s0b;
            }
        } else {
            long long base_left = offset_left;
            long long base_right = offset_right;
            const long long n0 = spec.k[0].extent;
            const long long n1 = spec.k[1].extent;
            const long long n2 = spec.k[2].extent;
            const long long s0a = spec.k[0].sa, s0b = spec.k[0].sb;
            const long long s1a = spec.k[1].sa, s1b = spec.k[1].sb;
            const long long s2a = spec.k[2].sa, s2b = spec.k[2].sb;
            for (long long i0 = 0; i0 < n0; ++i0) {
                long long mid_left = base_left;
                long long mid_right = base_right;
                for (long long i1 = 0; i1 < n1; ++i1) {
                    long long index_left = mid_left;
                    long long index_right = mid_right;
                    for (long long i2 = 0; i2 < n2; ++i2) {
                        cmac(acc, left[index_left], right[index_right]);
                        index_left += s2a;
                        index_right += s2b;
                    }
                    mid_left += s1a;
                    mid_right += s1b;
                }
                base_left += s0a;
                base_right += s0b;
            }
        }
        out[offset_out] = acc;
    }
}

/*
 * Final stage: same contraction, but the output is tiny (mom^2 * spin * link *
 * proj^2 = 32 elements in production) while the reduction is nev^2 * spin, so
 * one block owns one output element and splits the reduction across the block.
 */
__global__ __launch_bounds__(kThreads) void final_kernel(
    const cplx* __restrict__ left,
    const cplx* __restrict__ right,
    cplx* __restrict__ out,
    gc_spec spec,
    long long total_left,
    long long total_right) {
    __shared__ cplx partial[kThreads];
    const long long index = blockIdx.x;
    if (index >= total_left * total_right) {
        return;
    }
    long long j_index = index % total_right;
    long long b_index = index / total_right;
    long long offset_left = 0;
    long long offset_right = 0;
    long long offset_out = 0;
    for (int axis = spec.nb - 1; axis >= 0; --axis) {
        const long long extent = spec.batch[axis].extent;
        const long long digit = b_index % extent;
        b_index /= extent;
        offset_left += digit * spec.batch[axis].sa;
        offset_right += digit * spec.batch[axis].sb;
        offset_out += digit * spec.batch[axis].so;
    }
    for (int axis = spec.nj - 1; axis >= 0; --axis) {
        const long long extent = spec.j[axis].extent;
        const long long digit = j_index % extent;
        j_index /= extent;
        offset_right += digit * spec.j[axis].sb;
        offset_out += digit * spec.j[axis].so;
    }

    long long reduction = 1;
    for (int axis = 0; axis < spec.nk; ++axis) {
        reduction *= spec.k[axis].extent;
    }

    cplx acc = make_double2(0.0, 0.0);
    for (long long flat = threadIdx.x; flat < reduction; flat += blockDim.x) {
        long long rest = flat;
        long long index_left = offset_left;
        long long index_right = offset_right;
        for (int axis = spec.nk - 1; axis >= 0; --axis) {
            const long long extent = spec.k[axis].extent;
            const long long digit = rest % extent;
            rest /= extent;
            index_left += digit * spec.k[axis].sa;
            index_right += digit * spec.k[axis].sb;
        }
        cmac(acc, left[index_left], right[index_right]);
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (threadIdx.x < step) {
            const cplx other = partial[threadIdx.x + step];
            partial[threadIdx.x].x += other.x;
            partial[threadIdx.x].y += other.y;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[offset_out] = partial[0];
    }
}

int map_cuda(cudaError_t error) {
    if (error == cudaSuccess) {
        return QCD_STATUS_OK;
    }
    return (error == cudaErrorMemoryAllocation) ? QCD_STATUS_MEMORY
                                                : QCD_STATUS_INTERNAL;
}

int launch_gc(const gc_spec& spec, const void* left, const void* right,
              void* out, cudaStream_t stream) {
    const long long total_left = product_of_batch(spec);
    const long long total_right = product_of_j(spec);
    const long long total = total_left * total_right;
    if (total <= 0) {
        return QCD_STATUS_OK;
    }
    const long long blocks = (total + kThreads - 1) / kThreads;
    const unsigned grid =
        (unsigned)(blocks > 2147483647LL ? 2147483647LL : blocks);
    const cplx* l = static_cast<const cplx*>(left);
    const cplx* r = static_cast<const cplx*>(right);
    cplx* o = static_cast<cplx*>(out);
    switch (spec.nk) {
        case 1:
            gc_kernel<1><<<grid, kThreads, 0, stream>>>(
                l, r, o, spec, total_left, total_right);
            break;
        case 2:
            gc_kernel<2><<<grid, kThreads, 0, stream>>>(
                l, r, o, spec, total_left, total_right);
            break;
        case 3:
            gc_kernel<3><<<grid, kThreads, 0, stream>>>(
                l, r, o, spec, total_left, total_right);
            break;
        default:
            return QCD_STATUS_INVALID;
    }
    return map_cuda(cudaGetLastError());
}

int launch_final(const gc_spec& spec, const void* left, const void* right,
                 void* out, cudaStream_t stream) {
    const long long total_left = product_of_batch(spec);
    const long long total_right = product_of_j(spec);
    const long long total = total_left * total_right;
    if (total <= 0) {
        return QCD_STATUS_OK;
    }
    final_kernel<<<(unsigned)total, kThreads, 0, stream>>>(
        static_cast<const cplx*>(left), static_cast<const cplx*>(right),
        static_cast<cplx*>(out), spec, total_left, total_right);
    return map_cuda(cudaGetLastError());
}

/* ------------------------------------------------------------------ */
/* device arena: one grow-only allocation, reused by every call         */
/* ------------------------------------------------------------------ */

struct arena_state {
    void* pointer = nullptr;
    std::size_t capacity = 0;
    int device = -1;
};

arena_state g_arena;
std::mutex g_arena_mutex;

void* arena_reserve(std::size_t bytes, int* status) {
    if (bytes == 0) {
        *status = QCD_STATUS_INVALID;
        return nullptr;
    }
    int device = 0;
    const cudaError_t device_error = cudaGetDevice(&device);
    if (device_error != cudaSuccess) {
        *status = map_cuda(device_error);
        return nullptr;
    }
    if (g_arena.pointer != nullptr && g_arena.device != device) {
        cudaFree(g_arena.pointer);
        g_arena.pointer = nullptr;
        g_arena.capacity = 0;
    }
    if (g_arena.pointer == nullptr || g_arena.capacity < bytes) {
        void* fresh = nullptr;
        const cudaError_t error = cudaMalloc(&fresh, bytes);
        if (error != cudaSuccess) {
            *status = map_cuda(error);
            return nullptr;
        }
        if (g_arena.pointer != nullptr) {
            cudaFree(g_arena.pointer);
        }
        g_arena.pointer = fresh;
        g_arena.capacity = bytes;
        g_arena.device = device;
    }
    *status = QCD_STATUS_OK;
    return g_arena.pointer;
}

#ifdef QCD_PATTERN_3PT_DEBUG_DUMP
void debug_dump(const char* name, const void* device_pointer,
                long long elements, cudaStream_t stream) {
    const char* root = std::getenv("QCD_PATTERN_3PT_DUMP");
    if (root == nullptr || root[0] == '\0' || device_pointer == nullptr ||
        elements <= 0) {
        return;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s.%s.bin", root, name);
    FILE* handle = std::fopen(path, "wb");
    if (handle == nullptr) {
        return;
    }
    void* host = std::malloc((std::size_t)elements * sizeof(cplx));
    if (host == nullptr) {
        std::fclose(handle);
        return;
    }
    if (cudaMemcpyAsync(host, device_pointer,
                        (std::size_t)elements * sizeof(cplx),
                        cudaMemcpyDeviceToHost, stream) == cudaSuccess) {
        cudaStreamSynchronize(stream);
        std::fwrite(host, sizeof(cplx), (std::size_t)elements, handle);
    }
    std::free(host);
    std::fclose(handle);
}
#define QCD_DUMP(name, pointer, elements, stream) \
    debug_dump(name, pointer, elements, stream)
#else
#define QCD_DUMP(name, pointer, elements, stream) ((void)0)
#endif

#ifdef QCD_PATTERN_3PT_STAGE_TIMING
/* Development-only per-stage profile: one CUDA event pair per stage. */
struct stage_timer {
    const char* name;
    cudaStream_t stream;
    cudaEvent_t start;
    cudaEvent_t stop;

    stage_timer(const char* stage_name, cudaStream_t s)
        : name(stage_name), stream(s), start(nullptr), stop(nullptr) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }
    ~stage_timer() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::printf("[stage] %-8s %9.3f ms\n", name, milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};
/* Each stage lives in its own block scope, so a fixed local name is safe. */
#define QCD_STAGE_TIMER_1(name, stream) stage_timer qcd_stage_timer(name, stream)
#else
#define QCD_STAGE_TIMER_1(name, stream) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* fusion plans                                                        */
/* ------------------------------------------------------------------ */

struct dims_t {
    long long sp;   /* 4 */
    long long pj;   /* 2 */
    long long n;    /* nev */
    long long mo;   /* mom */
    long long lk;   /* link */
};

struct buffers {
    cplx* ta;
    cplx* a1;
    cplx* tc;
    cplx* c1;
    cplx* b1;
    cplx* d1;
    cplx* s0;
    cplx* s1;
    cplx* s2;
    cplx* s3;
    cplx* big0;
    cplx* big1;
    cplx* tail;
    cplx* out;
};

/*
 * Folds shared by both variants (everything that only involves one Dirac pair):
 *   ta[a][i][b][h]    = sum_g A[a][g][b][h] gamma5[g][i]
 *   a1[x][i][b][h]    = sum_a ta[a][i][b][h] proj_ax[a][x]
 *   tc[o][y][p][t]    = sum_s C[o][s][p][t] proj_sy[s][y]
 *   c1[G][m][y][p][t] = sum_o gamma_curr[G][m][o] tc[o][y][p][t]
 *   b1[e][k][f][r]    = sum_q B[e][q][f][r] gamma7_kq[k][q]
 *   d1[e][s][d][n]    = sum_c gamma7_ce[c][e] D[c][s][d][n]
 */
int run_folds(const dims_t& d, const void* const* in, buffers& b,
              cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj;
    int status = QCD_STATUS_OK;

    {
        gc_builder g;
        QCD_STAGE_TIMER_1("ta", stream);
        g.batch(SP, SP * N * N, 0, SP * N * N);   /* a */
        g.batch(N, N, 0, N);                      /* b */
        g.batch(N, 1, 0, 1);                      /* h */
        g.k(SP, N * N, SP);                       /* g */
        g.j(SP, 1, N * N);                        /* i */
        status = launch_gc(g.spec, in[0], in[7], b.ta, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("ta", b.ta, SP * SP * N * N, stream);
    }
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("a1", stream);
        g.batch(SP, N * N, 0, N * N);             /* i */
        g.batch(N, N, 0, N);                      /* b */
        g.batch(N, 1, 0, 1);                      /* h */
        g.k(SP, SP * N * N, PJ);                  /* a */
        g.j(PJ, 1, SP * N * N);                   /* x */
        status = launch_gc(g.spec, b.ta, in[13], b.a1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("a1", b.a1, PJ * SP * N * N, stream);
    }
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("tc", stream);
        g.batch(SP, SP * N * N, 0, PJ * N * N);   /* o */
        g.batch(N, N, 0, N);                      /* p */
        g.batch(N, 1, 0, 1);                      /* t */
        g.k(SP, N * N, PJ);                       /* s */
        g.j(PJ, 1, N * N);                        /* y */
        status = launch_gc(g.spec, in[2], in[14], b.tc, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("tc", b.tc, SP * PJ * N * N, stream);
    }
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("c1", stream);
        g.batch(PJ, N * N, 0, N * N);             /* y */
        g.batch(N, N, 0, N);                      /* p */
        g.batch(N, 1, 0, 1);                      /* t */
        g.k(SP, PJ * N * N, 1);                   /* o */
        g.j(SP, SP * SP, SP * PJ * N * N);        /* G */
        g.j(SP, SP, PJ * N * N);                  /* m */
        status = launch_gc(g.spec, b.tc, in[6], b.c1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("c1", b.c1, SP * SP * PJ * N * N, stream);
    }
    {
        /* b1[e][k][r][f]: r/f swapped so that f - the axis the B absorb walks
         * with consecutive threads - is contiguous. */
        gc_builder g;
        QCD_STAGE_TIMER_1("b1", stream);
        g.batch(SP, SP * N * N, 0, SP * N * N);   /* e */
        g.batch(N, N, 0, 1);                      /* f  -> out stride 1 */
        g.batch(N, 1, 0, N);                      /* r  -> out stride N */
        g.k(SP, N * N, 1);                        /* q */
        g.j(SP, SP, N * N);                       /* k */
        status = launch_gc(g.spec, in[1], in[8], b.b1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("b1", b.b1, SP * SP * N * N, stream);
    }
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("d1", stream);
        g.batch(SP, N * N, 0, N * N);             /* s (cmdn m / ckdl k) */
        g.batch(N, N, 0, N);                      /* d */
        g.batch(N, 1, 0, 1);                      /* n (n / l) */
        g.k(SP, SP * N * N, SP);                  /* c */
        g.j(SP, 1, SP * N * N);                   /* e */
        status = launch_gc(g.spec, in[3], in[5], b.d1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("d1", b.d1, SP * SP * N * N, stream);
    }
    return status;
}

/* shared tail: W absorb + final projection; identical for both variants */
int run_tail(const dims_t& d, const void* const* in, const cplx* carrier,
             buffers& b, cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo, LK = d.lk;
    int status = QCD_STATUS_OK;

    /* car[M][x][t][N][m][L][p] = sum_n carrier[M][x][t][N][m][n] W[N][L][n][p] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("car", stream);
        g.batch(MO, PJ * N * MO * SP * N, 0, PJ * N * MO * SP * LK * N); /* M */
        g.batch(PJ, N * MO * SP * N, 0, N * MO * SP * LK * N);           /* x */
        g.batch(N, MO * SP * N, 0, MO * SP * LK * N);                    /* t */
        g.batch(MO, SP * N, LK * N * N, SP * LK * N);                    /* N */
        g.batch(SP, N, 0, LK * N);                                       /* m */
        g.k(N, 1, N);                                                    /* n */
        g.j(LK, N * N, N);                                               /* L */
        g.j(N, 1, 1);                                                    /* p */
        status = launch_gc(g.spec, carrier, in[10], b.tail, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("car", b.tail, MO * PJ * N * MO * SP * LK * N, stream);
    }
    /* out[M][N][G][L][x][y] = sum_{m,p,t} car[..] c1[G][m][y][p][t] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("out", stream);
        g.batch(MO, PJ * N * MO * SP * LK * N, 0, MO * SP * LK * PJ * PJ); /* M */
        g.batch(MO, SP * LK * N, 0, SP * LK * PJ * PJ);                    /* N */
        g.batch(SP, 0, SP * PJ * N * N, LK * PJ * PJ);                     /* G */
        g.batch(PJ, N * MO * SP * LK * N, 0, PJ);                          /* x */
        g.batch(LK, N, 0, PJ * PJ);                                        /* L */
        g.k(SP, LK * N, PJ * N * N);                                       /* m */
        g.k(N, 1, N);                                                      /* p */
        g.k(N, MO * SP * LK * N, 1);                                       /* t */
        g.j(PJ, N * N, 1);                                                 /* y */
        status = launch_final(g.spec, b.tail, b.c1, b.out, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("out", b.out, MO * MO * SP * LK * PJ * PJ, stream);
    }
    return status;
}

/* variant 0: agbh,eqfr,ospt,cmdn,ikjl,... */
int run_p1(const dims_t& d, const void* const* in, buffers& b,
           cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo;
    int status = QCD_STATUS_OK;

    /* a2[M][x][i][b][j] = sum_h a1[x][i][b][h] S[M][h][j] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p1_a2", stream);
        g.batch(MO, 0, N * N, PJ * SP * N * N);   /* M */
        g.batch(PJ, SP * N * N, 0, SP * N * N);   /* x */
        g.batch(SP, N * N, 0, N * N);             /* i */
        g.batch(N, N, 0, N);                      /* b */
        g.k(N, 1, N);                             /* h */
        g.j(N, 1, 1);                             /* j */
        status = launch_gc(g.spec, b.a1, in[11], b.s0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a2", b.s0, MO * PJ * SP * N * N, stream);
    }
    /* a3[M][x][b][s2][l] = sum_{i,j} a2[M][x][i][b][j] E[i][s2][j][l] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p1_a3", stream);
        g.batch(MO, PJ * SP * N * N, 0, PJ * N * SP * N); /* M */
        g.batch(PJ, SP * N * N, 0, N * SP * N);           /* x */
        g.batch(N, N, 0, SP * N);                         /* b */
        g.k(SP, N * N, SP * N * N);                       /* i */
        g.k(N, 1, N);                                     /* j */
        g.j(SP, N * N, N);                                /* s2 */
        g.j(N, 1, 1);                                     /* l */
        status = launch_gc(g.spec, b.s0, in[4], b.s1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a3", b.s1, MO * PJ * N * SP * N, stream);
    }
    /* a5[M][x][b][k][r][t] = sum_l a3[M][x][b][k][l] T[M][l][r][t]
     *
     * note: gamma7_kq is folded into b1 (b1[e][k][f][r] = sum_q B[e][q][f][r]
     * gamma7[k][q]), so the carrier keeps E's raw k here and the B absorb below
     * contracts (k, r) - applying the gamma on both sides would square it. */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p1_a5", stream);
        g.batch(MO, PJ * N * SP * N, N * N * N, PJ * N * SP * N * N); /* M */
        g.batch(PJ, N * SP * N, 0, N * SP * N * N);                   /* x */
        g.batch(N, SP * N, 0, SP * N * N);                            /* b */
        g.batch(SP, N, 0, N * N);                                     /* k */
        g.k(N, 1, N * N);                                             /* l */
        g.j(N, N, N);                                                 /* r */
        g.j(N, 1, 1);                                                 /* t */
        status = launch_gc(g.spec, b.s1, in[12], b.big0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a5", b.big0, MO * PJ * N * SP * N * N, stream);
    }
    /* a6[M][x][b][t][e][f] = sum_{k,r} a5[M][x][b][k][r][t] b1[e][k][f][r] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p1_a6", stream);
        g.batch(MO, PJ * N * SP * N * N, 0, PJ * N * N * SP * N); /* M */
        g.batch(PJ, N * SP * N * N, 0, N * N * SP * N);           /* x */
        g.batch(N, SP * N * N, 0, N * SP * N);                    /* b */
        g.batch(N, 1, 0, SP * N);                                 /* t */
        g.k(SP, N * N, N * N);                                    /* k */
        g.k(N, N, N);                                             /* r */
        g.j(SP, SP * N * N, N);                                   /* e */
        g.j(N, 1, 1);                                             /* f */
        status = launch_gc(g.spec, b.big0, b.b1, b.big1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a6", b.big1, MO * PJ * N * N * SP * N, stream);
    }
    /* a7[M][x][t][e][N][d] = sum_{b,f} a6[M][x][b][t][e][f] V[N][b][d][f] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p1_a7", stream);
        g.batch(MO, PJ * N * N * SP * N, 0, PJ * N * SP * MO * N); /* M */
        g.batch(PJ, N * N * SP * N, 0, N * SP * MO * N);           /* x */
        g.batch(N, SP * N, 0, SP * MO * N);                        /* t */
        g.batch(SP, N, 0, MO * N);                                 /* e */
        g.batch(MO, 0, N * N * N, N);                              /* N */
        g.k(N, N * SP * N, N * N);                                 /* b */
        g.k(N, 1, 1);                                              /* f */
        g.j(N, N, 1);                                              /* d */
        status = launch_gc(g.spec, b.big1, in[9], b.s3, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a7", b.s3, MO * PJ * N * SP * MO * N, stream);
    }
    /* a8[M][x][t][N][m][n] = sum_{e,d} a7[M][x][t][e][N][d] d1[e][m][d][n] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p1_a8", stream);
        g.batch(MO, PJ * N * SP * MO * N, 0, PJ * N * MO * SP * N); /* M */
        g.batch(PJ, N * SP * MO * N, 0, N * MO * SP * N);           /* x */
        g.batch(N, SP * MO * N, 0, MO * SP * N);                    /* t */
        g.batch(MO, N, 0, SP * N);                                  /* N */
        g.k(SP, MO * N, SP * N * N);                                /* e */
        g.k(N, 1, N);                                               /* d */
        g.j(SP, N * N, N);                                          /* m */
        g.j(N, 1, 1);                                               /* n */
        status = launch_gc(g.spec, b.s3, b.d1, b.s0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a8", b.s0, MO * PJ * N * MO * SP * N, stream);
    }
    return status;
}

/* variant 1: agbh,eqfr,ospt,ckdl,imjn,... */
int run_p2(const dims_t& d, const void* const* in, buffers& b,
           cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo;
    int status = QCD_STATUS_OK;

    /* u1[M][e][k][d][r][t] = sum_l d1[e][k][d][l] T[M][l][r][t] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p2_u1", stream);
        g.batch(MO, 0, N * N * N, SP * SP * N * N * N); /* M */
        g.batch(SP, SP * N * N, 0, SP * N * N * N);     /* e */
        g.batch(SP, N * N, 0, N * N * N);               /* k */
        g.batch(N, N, 0, N * N);                        /* d */
        g.k(N, 1, N * N);                               /* l */
        g.j(N, N, N);                                   /* r */
        g.j(N, 1, 1);                                   /* t */
        status = launch_gc(g.spec, b.d1, in[12], b.big0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u1", b.big0, MO * SP * SP * N * N * N, stream);
    }
    /* u2[M][d][t][f] = sum_{e,k,r} u1[M][e][k][d][r][t] b1[e][k][f][r] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p2_u2", stream);
        g.batch(MO, SP * SP * N * N * N, 0, N * N * N); /* M */
        g.batch(N, N * N, 0, N * N);                    /* d */
        g.batch(N, 1, 0, N);                            /* t */
        g.k(SP, SP * N * N * N, SP * N * N);            /* e */
        g.k(SP, N * N * N, N * N);                      /* k */
        g.k(N, N, N);                                   /* r */
        g.j(N, 1, 1);                                   /* f */
        status = launch_gc(g.spec, b.big0, b.b1, b.s0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u2", b.s0, MO * N * N * N, stream);
    }
    /* u3[M][t][N][b] = sum_{d,f} u2[M][d][t][f] V[N][b][d][f] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p2_u3", stream);
        g.batch(MO, N * N * N, 0, N * MO * N);   /* M */
        g.batch(N, N, 0, MO * N);                /* t */
        g.batch(MO, 0, N * N * N, N);            /* N */
        g.k(N, N * N, N);                        /* d */
        g.k(N, 1, 1);                            /* f */
        g.j(N, N * N, 1);                        /* b */
        status = launch_gc(g.spec, b.s0, in[9], b.s1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u3", b.s1, MO * N * MO * N, stream);
    }
    /* u4[M][t][N][x][i][h] = sum_b u3[M][t][N][b] a1[x][i][b][h] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p2_u4", stream);
        g.batch(MO, N * MO * N, 0, N * MO * PJ * SP * N); /* M */
        g.batch(N, MO * N, 0, MO * PJ * SP * N);          /* t */
        g.batch(MO, N, 0, PJ * SP * N);                   /* N */
        g.k(N, 1, N);                                     /* b */
        g.j(PJ, SP * N * N, SP * N);                      /* x */
        g.j(SP, N * N, N);                                /* i */
        g.j(N, 1, 1);                                     /* h */
        status = launch_gc(g.spec, b.s1, b.a1, b.s2, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u4", b.s2, MO * N * MO * PJ * SP * N, stream);
    }
    /* u5[M][t][N][x][i][j] = sum_h u4[M][t][N][x][i][h] S[M][h][j] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p2_u5", stream);
        g.batch(MO, N * MO * PJ * SP * N, N * N, N * MO * PJ * SP * N); /* M */
        g.batch(N, MO * PJ * SP * N, 0, MO * PJ * SP * N);              /* t */
        g.batch(MO, PJ * SP * N, 0, PJ * SP * N);                       /* N */
        g.batch(PJ, SP * N, 0, SP * N);                                 /* x */
        g.batch(SP, N, 0, N);                                           /* i */
        g.k(N, 1, N);                                                   /* h */
        g.j(N, 1, 1);                                                   /* j */
        status = launch_gc(g.spec, b.s2, in[11], b.s3, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u5", b.s3, MO * N * MO * PJ * SP * N, stream);
    }
    /* u6[M][x][t][N][m][n] = sum_{i,j} u5[M][t][N][x][i][j] E2[i][m][j][n] */
    {
        gc_builder g;
        QCD_STAGE_TIMER_1("p2_u6", stream);
        g.batch(MO, N * MO * PJ * SP * N, 0, PJ * N * MO * SP * N); /* M */
        g.batch(PJ, SP * N, 0, N * MO * SP * N);                    /* x */
        g.batch(N, MO * PJ * SP * N, 0, MO * SP * N);               /* t */
        g.batch(MO, PJ * SP * N, 0, SP * N);                        /* N */
        g.k(SP, N, SP * N * N);                                     /* i */
        g.k(N, 1, N);                                               /* j */
        g.j(SP, N * N, N);                                          /* m */
        g.j(N, 1, 1);                                               /* n */
        /* u6 goes to big1 so that the shared tail can read it while writing its
         * own carrier into the tail slot. */
        status = launch_gc(g.spec, b.s3, in[4], b.big1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u6", b.big1, MO * PJ * N * MO * SP * N, stream);
    }
    return status;
}

long long max_of(long long a, long long b) { return a > b ? a : b; }

}  // namespace

extern "C" QCD_PATTERN_3PT_API int qcd_cuda_pattern_3pt(
    int dtype_id,
    const void* const* inputs,
    void* output,
    const long long* dims,
    int variant,
    void* stream_pointer) {
    if (dtype_id != QCD_DTYPE_COMPLEX128) {
        /* complex64 is deliberately not claimed: the documented 1e-12 relative
         * accuracy contract of this pattern only holds in double precision. */
        return QCD_STATUS_INVALID;
    }
    if (inputs == nullptr || output == nullptr || dims == nullptr) {
        return QCD_STATUS_INVALID;
    }
    if (variant != 0 && variant != 1) {
        return QCD_STATUS_INVALID;
    }
    for (int i = 0; i < 15; ++i) {
        if (inputs[i] == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    dims_t d;
    d.n = dims[0];
    d.mo = dims[1];
    d.lk = dims[2];
    d.sp = dims[3];
    d.pj = dims[4];
    if ((d.n <= 0) || (d.mo <= 0) || (d.lk <= 0) || (d.sp != 4) ||
        (d.pj != 2)) {
        return QCD_STATUS_INVALID;
    }
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo, LK = d.lk;

    const long long n_ta = SP * SP * N * N;
    const long long n_a1 = PJ * SP * N * N;
    const long long n_tc = SP * PJ * N * N;
    const long long n_c1 = SP * SP * PJ * N * N;
    const long long n_b1 = SP * SP * N * N;
    const long long n_d1 = SP * SP * N * N;
    const long long n_folds = n_ta + n_a1 + n_tc + n_c1 + n_b1 + n_d1;

    /* stages that live in one of the four "small" slots */
    const long long n_mid = MO * PJ * N * SP * N;             /* a2     */
    const long long n_car = MO * PJ * N * MO * SP * N;        /* a7/a8  */
    const long long n_u2 = MO * N * N * N;                    /* u2     */
    const long long n_u3 = MO * N * MO * N;                   /* u3     */
    const long long n_u4 = MO * N * MO * PJ * SP * N;         /* u4/u5  */
    const long long n_small =
        max_of(max_of(n_mid, n_car), max_of(max_of(n_u2, n_u3), n_u4));
    /* the two "big" slots */
    const long long n_big_a = MO * PJ * N * SP * N * N;       /* a5     */
    const long long n_big_b = MO * PJ * N * SP * N * N;       /* a6     */
    const long long n_u1 = MO * SP * SP * N * N * N;          /* u1     */
    const long long n_tail = MO * PJ * N * MO * SP * LK * N;  /* a9/u6  */

    long long total = n_folds + 4 * n_small + n_tail;
    long long big0 = 0, big1 = 0;
    if (variant == 0) {
        big0 = n_big_a;
        big1 = n_big_b;
    } else {
        big0 = n_u1;
        big1 = max_of(n_big_a, n_car); /* holds u6 for P2 */
    }
    total += big0 + big1;

    std::lock_guard<std::mutex> guard(g_arena_mutex);
    int status = QCD_STATUS_OK;
    void* arena = arena_reserve((std::size_t)total * sizeof(cplx), &status);
    if (status != QCD_STATUS_OK) {
        return status;
    }

    cplx* cursor = static_cast<cplx*>(arena);
    const cplx* const arena_end = cursor + total;
    buffers b{};
    b.ta = cursor; cursor += n_ta;
    b.a1 = cursor; cursor += n_a1;
    b.tc = cursor; cursor += n_tc;
    b.c1 = cursor; cursor += n_c1;
    b.b1 = cursor; cursor += n_b1;
    b.d1 = cursor; cursor += n_d1;
    b.s0 = cursor; cursor += n_small;
    b.s1 = cursor; cursor += n_small;
    b.s2 = cursor; cursor += n_small;
    b.s3 = cursor; cursor += n_small;
    b.big0 = cursor; cursor += big0;
    b.big1 = cursor; cursor += big1;
    b.tail = cursor; cursor += n_tail;
    b.out = static_cast<cplx*>(output);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    status = run_folds(d, inputs, b, stream);
    if (status == QCD_STATUS_OK) {
        if (variant == 0) {
            status = run_p1(d, inputs, b, stream);
            if (status == QCD_STATUS_OK) {
                /* a8 landed in s0 */
                status = run_tail(d, inputs, b.s0, b, stream);
            }
        } else {
            status = run_p2(d, inputs, b, stream);
            if (status == QCD_STATUS_OK) {
                /* u6 landed in big1 */
                status = run_tail(d, inputs, b.big1, b, stream);
            }
        }
    }
    if (status != QCD_STATUS_OK) {
        return status;
    }
    if (cursor > arena_end) {
        return QCD_STATUS_INTERNAL; /* layout bookkeeping drift */
    }
    return map_cuda(cudaGetLastError());
}

extern "C" QCD_PATTERN_3PT_API void qcd_cuda_pattern_3pt_free(void) {
    std::lock_guard<std::mutex> guard(g_arena_mutex);
    if (g_arena.pointer != nullptr) {
        cudaFree(g_arena.pointer);
        g_arena.pointer = nullptr;
        g_arena.capacity = 0;
        g_arena.device = -1;
    }
}
