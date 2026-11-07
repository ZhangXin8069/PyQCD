/*
 * Pattern-specific CUDA kernel for the sush distillation 3pt main term.
 *
 *   variant 0 (P1)  agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy
 *   variant 1 (P2)  agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy
 *                   -> MNGLxy
 *
 * The generic path spends its time in the 14 pairwise steps of the dp plan: it
 * transposes (packs) both operands of every step and then calls a batched GEMM
 * whose shapes are tiny on the spin side (4x4 gamma contractions become K=4
 * GEMMs).  This file instead executes a *fixed* fusion plan for the pattern.
 *
 * Fusion plan
 * -----------
 * Everything is expressed in the eigenvector ("nev") indices, the spin indices
 * are folded into the perambulators once up front, and the four propagator
 * vertices (sink_VVV, VdV_link, source_VdV, source_VVV) are absorbed into a
 * single running "carrier" tensor in the cost-optimal order.  For P1 that order
 * is the open Dirac chain  x-A-g5-E-g7kq-(T,B,V)-g7ce-D-W-C-y ; for P2 the
 * closed g7/g7 loop (B,D) has to be closed first, so the same carrier walk runs
 * from the D end:
 *
 *   P1: fold A/B/C/D -> S -> E -> g7kq -> T -> B -> V -> D -> W -> C
 *   P2: fold A/B/C/D -> T -> B -> V -> A -> S -> E -> W -> C
 *
 * Both walks touch the same 15 operands; the only difference is which Dirac leg
 * of the 4th/5th perambulator carries which gamma.  Every walk step is a tensor
 * contraction with an arbitrary strided layout, so one primitive
 * (``gc_kernel``) implements all of them; the final step, whose output has only
 * spin*proj*proj*mom*mom*link elements, is a dedicated reduction kernel that
 * splits the nev^2 reduction across the block.
 *
 * Cost of the chosen order (complex MACs, nev=N, mom=1, link=2, spins=4):
 *   P1: 8N^4 (T) + 32N^4 (B) + 8N^4 (V) + O(N^3)   = 3.1e8 at N=50
 *   P2: 16N^4 (T) + 16N^4 (B) + N^4 (V) + O(N^3)   = 2.2e8 at N=50
 * which is exactly the dp plan cost of the same equation, i.e. the fusion does
 * not add arithmetic - it removes the packing, the 4x4 GEMMs and the
 * intermediate round trips.
 */

#include "pattern_3pt.h"

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

__device__ __forceinline__ cplx cmul(cplx a, cplx b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ void cmac(cplx& acc, cplx a, cplx b) {
    acc.x = fma(a.x, b.x, fma(-a.y, b.y, acc.x));
    acc.y = fma(a.x, b.y, fma(a.y, b.x, acc.y));
}

/*
 * One axis of a contraction stage.
 *
 *   batch axes live in both operands (a stride of 0 means "absent") and in the
 *   output; ``k`` axes are summed and appear in neither output nor as batch;
 *   ``j`` axes are free and come from the right operand only.
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
        spec.j[spec.nj++] = gc_axis{0, sb, so};
        spec.j[spec.nj - 1].extent = extent;
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
 * inner loop is a plain pointer walk.  Nested axes are declared slowest first,
 * which also fixes the linearisation used for the thread mapping.
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
 * Final stage: out[batch, j] = sum_k left[batch, k] * right[batch, k, j].
 *
 * The output is tiny (mom^2 * spin * link * proj^2 elements, 32 in production)
 * while the reduction is nev^2 * spin long, so one block owns one output
 * element and splits the reduction across the block.
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
    const unsigned grid = (unsigned)(blocks > 2147483647LL ? 2147483647LL : blocks);
    const cplx* l = static_cast<const cplx*>(left);
    const cplx* r = static_cast<const cplx*>(right);
    cplx* o = static_cast<cplx*>(out);
    switch (spec.nk) {
        case 1:
            gc_kernel<1><<<grid, kThreads, 0, stream>>>(l, r, o, spec,
                                                        total_left, total_right);
            break;
        case 2:
            gc_kernel<2><<<grid, kThreads, 0, stream>>>(l, r, o, spec,
                                                        total_left, total_right);
            break;
        case 3:
            gc_kernel<3><<<grid, kThreads, 0, stream>>>(l, r, o, spec,
                                                        total_left, total_right);
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

std::size_t verify_integer(long long value, long long limit) {
    if (value <= 0 || value > limit) {
        return 0;
    }
    return 1;
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

/* ------------------------------------------------------------------ */
/* the two fusion plans                                                */
/* ------------------------------------------------------------------ */

/*
 * Layouts are written as [axis][axis]...[axis] with the last one contiguous.
 * ``sp`` = Dirac extent 4, ``pj`` = projector extent 2, ``N`` = nev,
 * ``mo`` = mom, ``lk`` = link.
 */
struct dims_t {
    long long sp;
    long long pj;
    long long n;
    long long mo;
    long long lk;
};

long long elem(long long a) { return a; }
long long elem(long long a, long long b) { return a * b; }
long long elem(long long a, long long b, long long c) { return a * b * c; }
long long elem(long long a, long long b, long long c, long long d) {
    return a * b * c * d;
}
long long elem(long long a, long long b, long long c, long long d, long long e) {
    return a * b * c * d * e;
}
long long elem(long long a, long long b, long long c, long long d, long long e,
               long long f) {
    return a * b * c * d * e * f;
}

/*
 * Folds shared by both variants:
 *   ta[a][i][b][h] = sum_g A[a][g][b][h] gamma5[g][i]
 *   a1[x][i][b][h] = sum_a ta[a][i][b][h] P[a][x]
 *   tc[o][y][p][t] = sum_s C[o][s][p][t] P[s][y]
 *   c1[G][m][y][p][t] = sum_o gamma_curr[G][m][o] tc[o][y][p][t]
 *   b1[e][k][f][r] = sum_q B[e][q][f][r] gamma7_kq[k][q]
 *   d1[e][s][d][n] = sum_c gamma7_ce[c][e] D[c][s][d][n]
 */
struct scratch {
    cplx* ta;
    cplx* a1;
    cplx* tc;
    cplx* c1;
    cplx* b1;
    cplx* d1;
    cplx* big_a;   /* P1: a5 */ cplx* big_b; /* P1: a6 */
    cplx* car0;    /* P1: a2/a3/a4 ; P2: u2/u3 */
    cplx* car1;    /* P1: a7/a8    ; P2: u4/u5 */
    cplx* car2;    /* P1: a9       ; P2: u6 */
    cplx* out;
};

int run_shared_folds(const dims_t& d, const void* const* in, scratch& s,
                     cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj;
    int status = QCD_STATUS_OK;

    /* ta[a][i][b][h] */
    {
        gc_builder b;
        b.batch(SP, SP * N * N, 0, SP * N * N);           /* a */
        b.batch(N, N, 0, N);                              /* b */
        b.batch(N, 1, 0, 1);                              /* h */
        b.k(SP, N * N, SP);                               /* g */
        b.j(SP, 1, N * N);                                /* i */
        status = launch_gc(b.spec, in[0], in[7], s.ta, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("ta", s.ta, elem(SP, SP, N, N), stream);
    }
    /* a1[x][i][b][h] */
    {
        gc_builder b;
        b.batch(SP, N * N, 0, N * N);                     /* i */
        b.batch(N, N, 0, N);                              /* b */
        b.batch(N, 1, 0, 1);                              /* h */
        b.k(SP, SP * N * N, PJ);                          /* a */
        b.j(PJ, 1, SP * N * N);                           /* x */
        status = launch_gc(b.spec, s.ta, in[13], s.a1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("a1", s.a1, elem(PJ, SP, N, N), stream);
    }
    /* tc[o][y][p][t] */
    {
        gc_builder b;
        b.batch(SP, SP * N * N, 0, PJ * N * N);           /* o */
        b.batch(N, N, 0, N);                              /* p */
        b.batch(N, 1, 0, 1);                              /* t */
        b.k(SP, N * N, PJ);                               /* s */
        b.j(PJ, 1, N * N);                                /* y */
        status = launch_gc(b.spec, in[2], in[14], s.tc, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("tc", s.tc, elem(SP, PJ, N, N), stream);
    }
    /* c1[G][m][y][p][t] */
    {
        gc_builder b;
        b.batch(N, N * N, 0, N * N);                      /* y */
        b.batch(N, N, 0, N);                              /* p */
        b.batch(N, 1, 0, 1);                              /* t */
        b.k(SP, PJ * N * N, 1);                           /* o */
        b.j(SP, SP * SP, SP * PJ * N * N);                /* G */
        b.j(SP, SP, PJ * N * N);                          /* m */
        status = launch_gc(b.spec, s.tc, in[6], s.c1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("c1", s.c1, elem(SP, SP, PJ, N, N), stream);
    }
    /* b1[e][k][f][r] */
    {
        gc_builder b;
        b.batch(SP, SP * N * N, 0, SP * N * N);           /* e */
        b.batch(N, N, 0, N);                              /* f */
        b.batch(N, 1, 0, 1);                              /* r */
        b.k(SP, N * N, 1);                                /* q */
        b.j(SP, SP, N * N);                               /* k */
        status = launch_gc(b.spec, in[1], in[8], s.b1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("b1", s.b1, elem(SP, SP, N, N), stream);
    }
    /* d1[e][s][d][n] */
    {
        gc_builder b;
        b.batch(SP, N * N, 0, N * N);                     /* s (= cmdn's m / ckdl's k) */
        b.batch(N, N, 0, N);                              /* d */
        b.batch(N, 1, 0, 1);                              /* n (= n / l) */
        b.k(SP, SP * N * N, SP);                          /* c */
        b.j(SP, 1, SP * N * N);                           /* e */
        status = launch_gc(b.spec, in[3], in[5], s.d1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("d1", s.d1, elem(SP, SP, N, N), stream);
    }
    return status;
}

/* shared tail: W absorb + final projection (identical for both variants) */
int run_shared_tail(const dims_t& d, const void* const* in, scratch& s,
                    cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo, LK = d.lk;
    int status = QCD_STATUS_OK;

    /* car2[M][x][t][N][m][L][p] = sum_n car1[M][x][t][N][m][n] W[N][L][n][p] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * MO * SP * N, 0, PJ * N * MO * SP * LK * N); /* M */
        b.batch(PJ, N * MO * SP * N, 0, N * MO * SP * LK * N);           /* x */
        b.batch(N, MO * SP * N, 0, MO * SP * LK * N);                    /* t */
        b.batch(MO, SP * N, LK * N * N, SP * LK * N);                    /* N */
        b.batch(SP, N, 0, LK * N);                                       /* m */
        b.k(N, 1, N);                                                    /* n */
        b.j(LK, N * N, N);                                               /* L */
        b.j(N, 1, 1);                                                    /* p */
        status = launch_gc(b.spec, s.car1, in[10], s.car2, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("car2", s.car2, elem(MO, PJ, N, MO, SP, LK, N), stream);
    }
    /* out[M][N][G][L][x][y] = sum_{m,p,t} car2[M][x][t][N][m][L][p] c1[G][m][y][p][t] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * MO * SP * LK * N, 0, MO * SP * LK * PJ * PJ); /* M */
        b.batch(MO, SP * LK * N, 0, SP * LK * PJ * PJ);                    /* N */
        b.batch(SP, 0, SP * PJ * N * N, LK * PJ * PJ);                     /* G */
        b.batch(PJ, N * MO * SP * LK * N, 0, PJ);                          /* x */
        b.batch(LK, N, 0, PJ * PJ);                                        /* L */
        b.k(SP, LK * N, PJ * N * N);                                       /* m */
        b.k(N, 1, N);                                                      /* p */
        b.k(N, MO * SP * LK * N, 1);                                       /* t */
        b.j(PJ, N * N, 1);                                                 /* y */
        status = launch_final(b.spec, s.car2, s.c1, s.out, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("out", s.out, elem(MO, MO, SP, LK, PJ, PJ), stream);
    }
    return status;
}

/* variant 0: agbh,eqfr,ospt,cmdn,ikjl,... */
int run_p1(const dims_t& d, const void* const* in, scratch& s,
           cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo;
    int status = QCD_STATUS_OK;

    /* a2[M][x][i][b][j] = sum_h a1[x][i][b][h] S[M][h][j] */
    {
        gc_builder b;
        b.batch(MO, 0, N * N, PJ * SP * N * N);           /* M */
        b.batch(PJ, SP * N * N, 0, SP * N * N);           /* x */
        b.batch(SP, N * N, 0, N * N);                     /* i */
        b.batch(N, N, 0, N);                              /* b */
        b.k(N, 1, N);                                     /* h */
        b.j(N, 1, 1);                                     /* j */
        status = launch_gc(b.spec, s.a1, in[11], s.car0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a2", s.car0, elem(MO, PJ, SP, N, N), stream);
    }
    /* a3[M][x][b][s2][l] = sum_{i,j} a2[M][x][i][b][j] E[i][s2][j][l] */
    {
        gc_builder b;
        b.batch(MO, PJ * SP * N * N, 0, PJ * N * SP * N); /* M */
        b.batch(PJ, SP * N * N, 0, N * SP * N);           /* x */
        b.batch(N, N, 0, SP * N);                         /* b */
        b.k(SP, N * N, SP * N * N);                       /* i */
        b.k(N, 1, N);                                     /* j */
        b.j(SP, N * N, N);                                /* s2 */
        b.j(N, 1, 1);                                     /* l */
        status = launch_gc(b.spec, s.car0, in[4], s.big_b, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a3", s.big_b, elem(MO, PJ, N, SP, N), stream);
    }
    /* a4[M][x][b][q][l] = sum_{s2} a3[M][x][b][s2][l] gamma7_kq[s2][q] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * SP * N, 0, PJ * N * SP * N); /* M */
        b.batch(PJ, N * SP * N, 0, N * SP * N);           /* x */
        b.batch(N, SP * N, 0, SP * N);                    /* b */
        b.batch(N, 1, 0, 1);                              /* l */
        b.k(SP, N, SP);                                   /* s2 */
        b.j(SP, 1, N);                                    /* q */
        status = launch_gc(b.spec, s.big_b, in[8], s.car0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a4", s.car0, elem(MO, PJ, N, SP, N), stream);
    }
    /* a5[M][x][b][q][r][t] = sum_l a4[M][x][b][q][l] T[M][l][r][t] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * SP * N, N * N * N, PJ * N * SP * N * N); /* M */
        b.batch(PJ, N * SP * N, 0, N * SP * N * N);                   /* x */
        b.batch(N, SP * N, 0, SP * N * N);                            /* b */
        b.batch(SP, N, 0, N * N);                                     /* q */
        b.k(N, 1, N * N);                                             /* l */
        b.j(N, N, N);                                                 /* r */
        b.j(N, 1, 1);                                                 /* t */
        status = launch_gc(b.spec, s.car0, in[12], s.big_a, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a5", s.big_a, elem(MO, PJ, N, SP, N, N), stream);
    }
    /* a6[M][x][b][t][e][f] = sum_{q,r} a5[M][x][b][q][r][t] b1[e][q][f][r] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * SP * N * N, 0, PJ * N * N * SP * N); /* M */
        b.batch(PJ, N * SP * N * N, 0, N * N * SP * N);           /* x */
        b.batch(N, SP * N * N, 0, N * SP * N);                    /* b */
        b.batch(N, 1, 0, SP * N);                                 /* t */
        b.k(SP, N * N, N * N);                                    /* q */
        b.k(N, N, 1);                                             /* r */
        b.j(SP, SP * N * N, N);                                   /* e */
        b.j(N, N, 1);                                             /* f */
        status = launch_gc(b.spec, s.big_a, s.b1, s.big_b, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a6", s.big_b, elem(MO, PJ, N, N, SP, N), stream);
    }
    /* a7[M][x][t][e][N][d] = sum_{b,f} a6[M][x][b][t][e][f] V[N][b][d][f] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * N * SP * N, 0, PJ * N * SP * MO * N); /* M */
        b.batch(PJ, N * N * SP * N, 0, N * SP * MO * N);           /* x */
        b.batch(N, SP * N, 0, SP * MO * N);                        /* t */
        b.batch(SP, N, 0, MO * N);                                 /* e */
        b.batch(MO, 0, N * N * N, N);                              /* N */
        b.k(N, N * SP * N, N * N);                                 /* b */
        b.k(N, 1, 1);                                              /* f */
        b.j(N, N, 1);                                              /* d */
        status = launch_gc(b.spec, s.big_b, in[9], s.car1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a7", s.car1, elem(MO, PJ, N, SP, MO, N), stream);
    }
    /* a8[M][x][t][N][m][n] = sum_{e,d} a7[M][x][t][e][N][d] d1[e][m][d][n] */
    {
        gc_builder b;
        b.batch(MO, PJ * N * SP * MO * N, 0, PJ * N * MO * SP * N); /* M */
        b.batch(PJ, N * SP * MO * N, 0, N * MO * SP * N);           /* x */
        b.batch(N, SP * MO * N, 0, MO * SP * N);                    /* t */
        b.batch(MO, N, 0, SP * N);                                  /* N */
        b.k(SP, MO * N, SP * N * N);                                /* e */
        b.k(N, 1, N);                                               /* d */
        b.j(SP, N * N, N);                                          /* m */
        b.j(N, 1, 1);                                               /* n */
        status = launch_gc(b.spec, s.car1, s.d1, s.car1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p1_a8", s.car1, elem(MO, PJ, N, MO, SP, N), stream);
    }
    return status;
}

/* variant 1: agbh,eqfr,ospt,ckdl,imjn,... */
int run_p2(const dims_t& d, const void* const* in, scratch& s,
           cudaStream_t stream) {
    const long long N = d.n, SP = d.sp, PJ = d.pj, MO = d.mo;
    int status = QCD_STATUS_OK;

    /* u1[M][e][k][d][r][t] = sum_l d1[e][k][d][l] T[M][l][r][t] */
    {
        gc_builder b;
        b.batch(MO, 0, N * N * N, SP * SP * N * N * N); /* M */
        b.batch(SP, SP * N * N, 0, SP * N * N * N);     /* e */
        b.batch(SP, N * N, 0, N * N * N);               /* k */
        b.batch(N, N, 0, N * N);                        /* d */
        b.k(N, 1, N * N);                               /* l */
        b.j(N, N, N);                                   /* r */
        b.j(N, 1, 1);                                   /* t */
        status = launch_gc(b.spec, s.d1, in[12], s.big_a, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u1", s.big_a, elem(MO, SP, SP, N, N, N), stream);
    }
    /* u2[M][d][t][f] = sum_{e,k,r} u1[M][e][k][d][r][t] b1[e][k][f][r] */
    {
        gc_builder b;
        b.batch(MO, SP * SP * N * N * N, 0, N * N * N); /* M */
        b.batch(N, N * N, 0, N * N);                    /* d */
        b.batch(N, 1, 0, N);                            /* t */
        b.k(SP, SP * N * N * N, SP * N * N);            /* e */
        b.k(SP, N * N * N, N * N);                      /* k */
        b.k(N, N, 1);                                   /* r */
        b.j(N, N, 1);                                   /* f */
        status = launch_gc(b.spec, s.big_a, s.b1, s.car0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u2", s.car0, elem(MO, N, N, N), stream);
    }
    /* u3[M][t][N][b] = sum_{d,f} u2[M][d][t][f] V[N][b][d][f] */
    {
        gc_builder b;
        b.batch(MO, N * N * N, 0, N * MO * N);   /* M */
        b.batch(N, N, 0, MO * N);                /* t */
        b.batch(MO, 0, N * N * N, N);            /* N */
        b.k(N, N * N, N);                        /* d */
        b.k(N, 1, 1);                            /* f */
        b.j(N, N * N, 1);                        /* b */
        status = launch_gc(b.spec, s.car0, in[9], s.car0, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u3", s.car0, elem(MO, N, MO, N), stream);
    }
    /* u4[M][t][N][x][i][h] = sum_b u3[M][t][N][b] a1[x][i][b][h] */
    {
        gc_builder b;
        b.batch(MO, N * MO * N, 0, N * MO * PJ * SP * N); /* M */
        b.batch(N, MO * N, 0, MO * PJ * SP * N);          /* t */
        b.batch(MO, N, 0, PJ * SP * N);                   /* N */
        b.k(N, 1, N);                                     /* b */
        b.j(PJ, SP * N * N, SP * N);                      /* x */
        b.j(SP, N * N, N);                                /* i */
        b.j(N, 1, 1);                                     /* h */
        status = launch_gc(b.spec, s.car0, s.a1, s.car1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u4", s.car1, elem(MO, N, MO, PJ, SP, N), stream);
    }
    /* u5[M][t][N][x][i][j] = sum_h u4[M][t][N][x][i][h] S[M][h][j] */
    {
        gc_builder b;
        b.batch(MO, N * MO * PJ * SP * N, N * N, N * MO * PJ * SP * N); /* M */
        b.batch(N, MO * PJ * SP * N, 0, MO * PJ * SP * N);              /* t */
        b.batch(MO, PJ * SP * N, 0, PJ * SP * N);                       /* N */
        b.batch(PJ, SP * N, 0, SP * N);                                 /* x */
        b.batch(SP, N, 0, N);                                           /* i */
        b.k(N, 1, N);                                                   /* h */
        b.j(N, 1, 1);                                                   /* j */
        status = launch_gc(b.spec, s.car1, in[11], s.car1, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u5", s.car1, elem(MO, N, MO, PJ, SP, N), stream);
    }
    /* u6[M][x][t][N][m][n] = sum_{i,j} u5[M][t][N][x][i][j] E2[i][m][j][n] */
    {
        gc_builder b;
        b.batch(MO, N * MO * PJ * SP * N, 0, PJ * N * MO * SP * N); /* M */
        b.batch(PJ, SP * N, 0, N * MO * SP * N);                    /* x */
        b.batch(N, MO * PJ * SP * N, 0, MO * SP * N);               /* t */
        b.batch(MO, PJ * SP * N, 0, SP * N);                        /* N */
        b.k(SP, N, SP * N * N);                                     /* i */
        b.k(N, 1, N);                                               /* j */
        b.j(SP, N * N, N);                                          /* m */
        b.j(N, 1, 1);                                               /* n */
        status = launch_gc(b.spec, s.car1, in[4], s.car2, stream);
        if (status != QCD_STATUS_OK) return status;
        QCD_DUMP("p2_u6", s.car2, elem(MO, PJ, N, MO, SP, N), stream);
    }
    /* u6 is the carrier the shared tail expects, but the tail reads car1 */
    s.car1 = s.car2;
    return status;
}

int run_variant(int variant, const dims_t& d, const void* const* in, scratch& s,
                cudaStream_t stream) {
    int status = run_shared_folds(d, in, s, stream);
    if (status != QCD_STATUS_OK) {
        return status;
    }
    if (variant == 0) {
        /*
         * P1 needs three live carriers (a4 -> a5 -> a6) so big_a/big_b hold the
         * two nev^3 tensors and car0 carries the nev^2 stages.
         */
        status = run_p1(d, in, s, stream);
        if (status != QCD_STATUS_OK) {
            return status;
        }
        s.car1 = nullptr; /* a8 is written into car1 by run_p1 */
    } else {
        status = run_p2(d, in, s, stream);
        if (status != QCD_STATUS_OK) {
            return status;
        }
    }
    return run_shared_tail(d, in, s, stream);
}

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

    /* Intermediate sizes (elements).  Both plans are laid out so that the
     * offsets can be computed without a general allocator. */
    const long long n_ta = elem(SP, SP, N, N);
    const long long n_a1 = elem(PJ, SP, N, N);
    const long long n_tc = elem(SP, PJ, N, N);
    const long long n_c1 = elem(SP, SP, PJ, N, N);
    const long long n_b1 = elem(SP, SP, N, N);
    const long long n_d1 = elem(SP, SP, N, N);
    const long long n_mid = elem(MO, PJ, N, SP, N);              /* a2..a4 */
    const long long n_big = elem(MO, PJ, N, SP, N, N);           /* a5/a6 */
    const long long n_car = elem(MO, PJ, N, MO, SP, N);          /* a7/a8 */
    const long long n_tail = elem(MO, PJ, N, MO, SP, LK, N);     /* a9/u6-u7 */
    const long long n_u1 = elem(MO, SP, SP, N, N, N);
    const long long n_u2 = elem(MO, N, N, N);
    const long long n_u4 = elem(MO, N, MO, PJ, SP, N);

    long long total = n_ta + n_a1 + n_tc + n_c1 + n_b1 + n_d1;
    if (variant == 0) {
        total += n_mid * 2 + n_big * 2 + n_car + n_tail;
    } else {
        total += n_u1 + n_u2 + n_u4 * 2 + n_tail;
    }

    std::lock_guard<std::mutex> guard(g_arena_mutex);
    int status = QCD_STATUS_OK;
    void* arena = arena_reserve((std::size_t)total * sizeof(cplx), &status);
    if (status != QCD_STATUS_OK) {
        return status;
    }

    cplx* cursor = static_cast<cplx*>(arena);
    scratch s{};
    s.ta = cursor; cursor += n_ta;
    s.a1 = cursor; cursor += n_a1;
    s.tc = cursor; cursor += n_tc;
    s.c1 = cursor; cursor += n_c1;
    s.b1 = cursor; cursor += n_b1;
    s.d1 = cursor; cursor += n_d1;
    if (variant == 0) {
        s.car0 = cursor; cursor += n_mid;    /* a2/a3/a4 share (sequential) */
        s.big_a = cursor; cursor += n_big;   /* a5 */
        s.big_b = cursor; cursor += n_big;   /* a6, then a3 lives in big_b */
        s.car1 = cursor; cursor += n_car;    /* a7/a8 */
        s.car2 = cursor; cursor += n_tail;   /* a9 */
    } else {
        s.big_a = cursor; cursor += n_u1;    /* u1 */
        s.car0 = cursor; cursor += n_u2;     /* u2/u3 (u2 is nev^3 * mom)  */
        cursor += (n_u2 > n_mid ? n_u2 - n_mid : 0); /* headroom for u3 */
        s.car1 = cursor; cursor += n_u4;     /* u4/u5 */
        s.car2 = cursor; cursor += n_tail;   /* u6 -> u7 */
    }
    s.out = static_cast<cplx*>(output);
    if (cursor > static_cast<cplx*>(arena) + total) {
        return QCD_STATUS_INTERNAL; /* layout bookkeeping drift */
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    status = run_variant(variant, d, inputs, s, stream);
    if (status != QCD_STATUS_OK) {
        return status;
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
