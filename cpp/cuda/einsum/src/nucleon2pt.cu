/* PyQCD — nucleon 2pt 专用融合 kernel（nucleon2pt.cu）
 *
 * 契约、dims 约定与轴序说明见 include/nucleon2pt.h；本文件只放实现。
 *
 * 设计要点（与通用 einsum 路径的差别）
 * ------------------------------------------------------------------
 * 1. **FLOP 最优**：按 96N⁴ 的路径算（= opt_einsum optimal/dp 的 opt_cost/2），
 *    不用通用路径的分步 tensordot + transpose。
 * 2. **单流、一次 workspace 申请**：Abuf/Bbuf 各 16N³ 元素，一次 cudaMalloc
 *    复用；没有多流/多卡并行，所有倍率都来自算法与 kernel 本身。
 * 3. **布局为 big-GEMM 而设**：
 *      Bbuf[c][m][f][n][b]  —— 收缩指标 (n,b) 连续，列 (m,f) 各一个基址；
 *      Abuf[c][g][p][n][b]  —— 收缩指标 (n,b) 连续，行 (g,p) 各一个基址；
 *    于是 64N⁴ 的大收缩（K = 4N²）在两侧都能**连续**读取，退化成
 *    一个 (4N)×(4N)、K=4N² 的分块 GEMM；两个 16N⁴ 的展开步骤是
 *    16N 个 (N×N×N) 的小 GEMM，按 (i,k) 连续的方式分块。
 * 4. 自研共享内存分块 GEMM（double2/float2），不走 cuBLAS：cuBLAS ZGEMM
 *    在 (400×400×40000) 实测只有 ~83 GFLOP/s，本机真实 FP64 上限约
 *    ~120 GFLOP/s（详见交付报告 §测量）。
 */

#include "nucleon2pt.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

// ---------------------------------------------------------------- 基本形状
constexpr int kSpin = 4;         // 自旋/色指标长度（g,a,h,b / m,c,n,d / o,e,p,f）
constexpr int kThreadTile = 4;   // 每线程 4x4 复数累加器
constexpr int kBlockSide = 16;   // 16x16 = 256 线程
constexpr int kThreads = kBlockSide * kBlockSide;
constexpr int kTileM = kBlockSide * kThreadTile;   // 64 行 / 块
constexpr int kTileN = kBlockSide * kThreadTile;   // 64 列 / 块
constexpr int kTileK = 16;                         // K 方向分块
constexpr long long kMaxN = 4096;
constexpr long long kWorkspaceLimitBytes = 3ll << 30;  // 3 GiB 上限（fail closed）

// ---------------------------------------------------------------- 复数运算
template <typename T>
struct Scalar;
template <>
struct Scalar<double2> {
    using type = double;
};
template <>
struct Scalar<float2> {
    using type = float;
};

__device__ __forceinline__ void cmadd(double2& acc, const double2 a, const double2 b) {
    acc.x = __fma_rn(a.x, b.x, acc.x);
    acc.x = __fma_rn(-a.y, b.y, acc.x);
    acc.y = __fma_rn(a.x, b.y, acc.y);
    acc.y = __fma_rn(a.y, b.x, acc.y);
}

__device__ __forceinline__ void cmadd(float2& acc, const float2 a, const float2 b) {
    acc.x = __fmaf_rn(a.x, b.x, acc.x);
    acc.x = __fmaf_rn(-a.y, b.y, acc.x);
    acc.y = __fmaf_rn(a.x, b.y, acc.y);
    acc.y = __fmaf_rn(a.y, b.x, acc.y);
}

__device__ __forceinline__ void cadd(double2& acc, const double2 a) {
    acc.x += a.x;
    acc.y += a.y;
}
__device__ __forceinline__ void cadd(float2& acc, const float2 a) {
    acc.x += a.x;
    acc.y += a.y;
}

__device__ __forceinline__ double2 cmul(const double2 a, const double2 b) {
    double2 r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

__device__ __forceinline__ float2 cmul(const float2 a, const float2 b) {
    float2 r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

template <typename T>
__device__ __forceinline__ T zero_of();
template <>
__device__ __forceinline__ double2 zero_of<double2>() {
    return make_double2(0.0, 0.0);
}
template <>
__device__ __forceinline__ float2 zero_of<float2>() {
    return make_float2(0.0f, 0.0f);
}

// ------------------------------------------------------------- perm 码解码
// code = p0*1000 + p1*100 + p2*10 + p3；pi = 规范轴 i 对应的输入轴下标。
__host__ __device__ __forceinline__ void perm_digits(int code, int p[4]) {
    p[0] = code / 1000;
    p[1] = (code / 100) % 10;
    p[2] = (code / 10) % 10;
    p[3] = code % 10;
}

__host__ __device__ __forceinline__ bool perm_valid(int code, int rank) {
    if (code < 0 || code > 3333) {
        return false;
    }
    int p[4];
    perm_digits(code, p);
    int seen = 0;
    for (int i = 0; i < 4; ++i) {
        if (i >= rank) {
            if (p[i] != i) {
                return false;
            }
            continue;
        }
        if (p[i] < 0 || p[i] >= rank) {
            return false;
        }
        if (seen & (1 << p[i])) {
            return false;
        }
        seen |= 1 << p[i];
    }
    return true;
}

// ======================================================= stage 0: 归一化 (4,4,N,N)
// dst[i0][i1][i2][i3] （规范序、紧致） = src[Σ i_k * stride[k]]
template <typename T>
__global__ void normalize_peram_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                       long long n, long long total, long long s0,
                                       long long s1, long long s2, long long s3) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long i3 = idx % n;
    const long long i2 = (idx / n) % n;
    const long long i1 = (idx / (n * n)) % kSpin;
    const long long i0 = idx / (n * n * kSpin);
    dst[idx] = src[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
}

// ============================================================== stage 1a: U1T
// U1T[c][g][b][h] = Σ_a GX[a][c] A[g][a][h][b]      （A 为规范紧致序）
template <typename T>
__global__ void fold_gx_kernel(const T* __restrict__ a, const T* __restrict__ gx,
                               T* __restrict__ u1t, long long n, long long total,
                               int gx_transposed) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long h = idx % n;
    const long long b = (idx / n) % n;
    const long long g = (idx / (n * n)) % kSpin;
    const long long c = idx / (n * n * kSpin);

    T acc = zero_of<T>();
    for (int ai = 0; ai < kSpin; ++ai) {
        const T av = a[((g * kSpin + ai) * n + h) * n + b];
        const T gv = gx_transposed ? gx[c * kSpin + ai] : gx[ai * kSpin + c];
        cmadd(acc, av, gv);
    }
    u1t[idx] = acc;
}

// ============================================================ stage 1b: U2
// U2[g][m][p][f] = Σ_{o,e} GY[o][g] PR[e][m] C[o][e][p][f]   （C 为规范紧致序）
template <typename T>
__global__ void fold_gy_pr_kernel(const T* __restrict__ c, const T* __restrict__ gy,
                                  const T* __restrict__ pr, T* __restrict__ u2,
                                  long long n, long long total, int gy_transposed,
                                  int pr_transposed) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long f = idx % n;
    const long long p = (idx / n) % n;
    const long long m = (idx / (n * n)) % kSpin;
    const long long g = idx / (n * n * kSpin);

    T acc = zero_of<T>();
    for (int o = 0; o < kSpin; ++o) {
        const T gyv = gy_transposed ? gy[g * kSpin + o] : gy[o * kSpin + g];
        for (int e = 0; e < kSpin; ++e) {
            const T prv = pr_transposed ? pr[m * kSpin + e] : pr[e * kSpin + m];
            cmadd(acc, cmul(gyv, prv), c[((o * kSpin + e) * n + p) * n + f]);
        }
    }
    u2[idx] = acc;
}

// ============================================================ V1 -> V1T
// V1T[f][b][d] = V1[mx][b][d][f]（任意轴序，动量轴固定输入第 0 维）
template <typename T>
__global__ void v1_to_transposed_kernel(const T* __restrict__ v1,
                                        T* __restrict__ v1t, long long n,
                                        long long momentum_offset, long long sb,
                                        long long sd, long long sf) {
    const long long total = n * n * n;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long d = idx % n;
    const long long b = (idx / n) % n;
    const long long f = idx / (n * n);
    v1t[(f * n + b) * n + d] = v1[momentum_offset + b * sb + d * sd + f * sf];
}

// ============================================================ V2 -> canonical
// V2n[n][p][h] = V2[my][n][p][h]（任意轴序）
template <typename T>
__global__ void v2_to_canonical_kernel(const T* __restrict__ v2,
                                       T* __restrict__ v2n, long long n,
                                       long long momentum_offset, long long sn,
                                       long long sp, long long sh) {
    const long long total = n * n * n;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long h = idx % n;
    const long long p = (idx / n) % n;
    const long long nn = idx / (n * n);
    v2n[idx] = v2[momentum_offset + nn * sn + p * sp + h * sh];
}

// ================================================== 通用 batched TN GEMM
// out[i][j] += Σ_k A[i][k] B[j][k]（两侧 K 连续）；每个 batch 的基址由
// BatchMap 的 3 轴仿射偏移给出。
struct BatchMap {
    long long size[3];
    long long stride[3];
};

__device__ __forceinline__ long long batch_offset(const BatchMap& map, int id) {
    long long base = 0;
    int rem = id;
    for (int d = 2; d >= 0; --d) {
        const int extent = static_cast<int>(map.size[d]);
        const int index = (extent > 0) ? (rem % extent) : 0;
        if (extent > 0) {
            rem /= extent;
        }
        base += static_cast<long long>(index) * map.stride[d];
    }
    return base;
}

template <typename T>
__global__ __launch_bounds__(kThreads) void gemm_tn_batched_kernel(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, int rows,
    int cols, int kk, long long a_row, long long b_row, long long c_row,
    long long c_col, BatchMap amap, BatchMap bmap, BatchMap cmap) {
    __shared__ T sh_a[kTileM][kTileK + 1];
    __shared__ T sh_b[kTileN][kTileK + 1];

    const int tid = threadIdx.x;
    const int tx = tid % kBlockSide;
    const int ty = tid / kBlockSide;
    const int i_tile = static_cast<int>(blockIdx.y) * kTileM;
    const int j_tile = static_cast<int>(blockIdx.x) * kTileN;
    const int row_lo = i_tile + ty * kThreadTile;
    const int col_lo = j_tile + tx * kThreadTile;
    const bool active = (row_lo < rows) && (col_lo < cols) && (row_lo + kThreadTile > 0) &&
                        (col_lo + kThreadTile > 0);

    const long long abase = batch_offset(amap, static_cast<int>(blockIdx.z));
    const long long bbase = batch_offset(bmap, static_cast<int>(blockIdx.z));
    const long long cbase = batch_offset(cmap, static_cast<int>(blockIdx.z));

    T acc[kThreadTile][kThreadTile];
#pragma unroll
    for (int r = 0; r < kThreadTile; ++r) {
#pragma unroll
        for (int s = 0; s < kThreadTile; ++s) {
            acc[r][s] = zero_of<T>();
        }
    }

    for (int k0 = 0; k0 < kk; k0 += kTileK) {
        for (int idx = tid; idx < kTileM * kTileK; idx += kThreads) {
            const int r = idx / kTileK;
            const int t = idx - r * kTileK;
            const int gi = i_tile + r;
            const int gk = k0 + t;
            T v = zero_of<T>();
            if (gi < rows && gk < kk) {
                v = a[abase + static_cast<long long>(gi) * a_row + gk];
            }
            sh_a[r][t] = v;
        }
        for (int idx = tid; idx < kTileN * kTileK; idx += kThreads) {
            const int r = idx / kTileK;
            const int t = idx - r * kTileK;
            const int gj = j_tile + r;
            const int gk = k0 + t;
            T v = zero_of<T>();
            if (gj < cols && gk < kk) {
                v = b[bbase + static_cast<long long>(gj) * b_row + gk];
            }
            sh_b[r][t] = v;
        }
        __syncthreads();

        if (active) {
            for (int t = 0; t < kTileK; ++t) {
                T av[kThreadTile];
                T bv[kThreadTile];
#pragma unroll
                for (int r = 0; r < kThreadTile; ++r) {
                    av[r] = sh_a[ty * kThreadTile + r][t];
                }
#pragma unroll
                for (int s = 0; s < kThreadTile; ++s) {
                    bv[s] = sh_b[tx * kThreadTile + s][t];
                }
#pragma unroll
                for (int r = 0; r < kThreadTile; ++r) {
#pragma unroll
                    for (int s = 0; s < kThreadTile; ++s) {
                        cmadd(acc[r][s], av[r], bv[s]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }
#pragma unroll
    for (int r = 0; r < kThreadTile; ++r) {
        const int gi = row_lo + r;
        if (gi >= rows) {
            continue;
        }
#pragma unroll
        for (int s = 0; s < kThreadTile; ++s) {
            const int gj = col_lo + s;
            if (gj >= cols) {
                continue;
            }
            c[cbase + static_cast<long long>(gi) * c_row +
              static_cast<long long>(gj) * c_col] = acc[r][s];
        }
    }
}

// ================================================================ stage 5: 星形收缩
// Cout[(g,p)][(m,f)] = Σ_{c,(n,b)} Abuf[c][g][p][n][b] Bbuf[c][m][f][n][b]
template <typename T>
__global__ __launch_bounds__(kThreads) void star_kernel(
    const T* __restrict__ abuf, const T* __restrict__ bbuf, T* __restrict__ cout,
    int n) {
    __shared__ T sh_a[kTileM][kTileK + 1];
    __shared__ T sh_b[kTileN][kTileK + 1];
    __shared__ long long sh_rowbase[kTileM];
    __shared__ long long sh_colbase[kTileN];

    const int tid = threadIdx.x;
    const int tx = tid % kBlockSide;
    const int ty = tid / kBlockSide;
    const int rows = kSpin * n;
    const int cols = kSpin * n;
    const long long n2 = static_cast<long long>(n) * n;
    const long long n3 = n2 * n;
    const int kk = static_cast<int>(n2);
    const int i_tile = static_cast<int>(blockIdx.y) * kTileM;
    const int j_tile = static_cast<int>(blockIdx.x) * kTileN;
    const int row_lo = i_tile + ty * kThreadTile;
    const int col_lo = j_tile + tx * kThreadTile;
    const bool active = (row_lo < rows) && (col_lo < cols);

    if (tid < kTileM) {
        const int i = i_tile + tid;
        sh_rowbase[tid] = (i < rows) ? (static_cast<long long>(i / n) * n3 +
                                        static_cast<long long>(i % n) * n2)
                                    : -1;
    }
    if (tid < kTileN) {
        const int j = j_tile + tid;
        sh_colbase[tid] = (j < cols) ? (static_cast<long long>(j / n) * n3 +
                                        static_cast<long long>(j % n) * n2)
                                    : -1;
    }
    __syncthreads();

    T acc[kThreadTile][kThreadTile];
#pragma unroll
    for (int r = 0; r < kThreadTile; ++r) {
#pragma unroll
        for (int s = 0; s < kThreadTile; ++s) {
            acc[r][s] = zero_of<T>();
        }
    }

    for (int c = 0; c < kSpin; ++c) {
        const long long cbase = static_cast<long long>(c) * kSpin * n3;
        const T* a_ptr = abuf + cbase;
        const T* b_ptr = bbuf + cbase;
        for (int k0 = 0; k0 < kk; k0 += kTileK) {
            for (int idx = tid; idx < kTileM * kTileK; idx += kThreads) {
                const int r = idx / kTileK;
                const int t = idx - r * kTileK;
                const long long rb = sh_rowbase[r];
                const int gk = k0 + t;
                T v = zero_of<T>();
                if (rb >= 0 && gk < kk) {
                    v = a_ptr[rb + gk];
                }
                sh_a[r][t] = v;
            }
            for (int idx = tid; idx < kTileN * kTileK; idx += kThreads) {
                const int r = idx / kTileK;
                const int t = idx - r * kTileK;
                const long long cb = sh_colbase[r];
                const int gk = k0 + t;
                T v = zero_of<T>();
                if (cb >= 0 && gk < kk) {
                    v = b_ptr[cb + gk];
                }
                sh_b[r][t] = v;
            }
            __syncthreads();

            if (active) {
                for (int t = 0; t < kTileK; ++t) {
                    T av[kThreadTile];
                    T bv[kThreadTile];
#pragma unroll
                    for (int r = 0; r < kThreadTile; ++r) {
                        av[r] = sh_a[ty * kThreadTile + r][t];
                    }
#pragma unroll
                    for (int s = 0; s < kThreadTile; ++s) {
                        bv[s] = sh_b[tx * kThreadTile + s][t];
                    }
#pragma unroll
                    for (int r = 0; r < kThreadTile; ++r) {
#pragma unroll
                        for (int s = 0; s < kThreadTile; ++s) {
                            cmadd(acc[r][s], av[r], bv[s]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    if (!active) {
        return;
    }
    const long long ld = static_cast<long long>(kSpin) * n;
#pragma unroll
    for (int r = 0; r < kThreadTile; ++r) {
        const int i = row_lo + r;
        if (i >= rows) {
            continue;
        }
#pragma unroll
        for (int s = 0; s < kThreadTile; ++s) {
            const int j = col_lo + s;
            if (j >= cols) {
                continue;
            }
            cout[static_cast<long long>(i) * ld + j] = acc[r][s];
        }
    }
}

// ============================================================ stage 6: 收尾点积
// out += Σ_{g,m,p,f} U2[(g*4+m)*N²+p*N+f] * Cout[(g*N+p)*(4N) + m*N+f]
template <typename T>
__global__ void final_dot_kernel(const T* __restrict__ u2, const T* __restrict__ cout,
                                 T* __restrict__ out, long long n, long long total,
                                 int blocks) {
    const long long start = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long step = static_cast<long long>(blocks) * blockDim.x;
    const long long n2 = n * n;
    const long long ld = static_cast<long long>(kSpin) * n;
    T acc = zero_of<T>();
    for (long long idx = start; idx < total; idx += step) {
        const long long f = idx % n;
        const long long p = (idx / n) % n;
        const long long m = (idx / n2) % kSpin;
        const long long g = idx / (n2 * kSpin);
        const long long row = g * n + p;
        cmadd(acc, u2[idx], cout[row * ld + m * n + f]);
    }
    __shared__ T sh[kThreads];
    const int lane = static_cast<int>(threadIdx.x);
    sh[lane] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            cadd(sh[lane], sh[lane + offset]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        using R = typename Scalar<T>::type;
        atomicAdd(reinterpret_cast<R*>(out), sh[0].x);
        atomicAdd(reinterpret_cast<R*>(out) + 1, sh[0].y);
    }
}

// ================================================================== host 侧
struct Params {
    long long n = 0;
    long long mx = 0;
    long long my = 0;
    int mode = 0;
    int matflags = 0;
    int perm[5] = {0, 0, 0, 0, 0};  // A, B, C, V1, V2
};

struct Workspace {
    void* ptr = nullptr;
    size_t bytes = 0;
    int device = -1;
};

std::mutex g_mutex;
Workspace g_workspace;

int map_cuda(cudaError_t err) {
    if (err == cudaSuccess) {
        return QCD_STATUS_OK;
    }
    return (err == cudaErrorMemoryAllocation || err == cudaErrorDevicesUnavailable)
               ? QCD_STATUS_MEMORY
               : QCD_STATUS_INTERNAL;
}

void permuted_strides(const long long src[4], int code, long long out[4]) {
    int p[4];
    perm_digits(code, p);
    for (int i = 0; i < 4; ++i) {
        out[i] = src[p[i]];
    }
}

bool get_workspace(size_t bytes, void** ptr) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    std::lock_guard<std::mutex> guard(g_mutex);
    if (g_workspace.ptr != nullptr && g_workspace.bytes >= bytes &&
        g_workspace.device == device) {
        *ptr = g_workspace.ptr;
        return true;
    }
    if (g_workspace.ptr != nullptr) {
        cudaFree(g_workspace.ptr);
        g_workspace.ptr = nullptr;
        g_workspace.bytes = 0;
        g_workspace.device = -1;
    }
    void* fresh = nullptr;
    if (cudaMalloc(&fresh, bytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    g_workspace.ptr = fresh;
    g_workspace.bytes = bytes;
    g_workspace.device = device;
    *ptr = fresh;
    return true;
}

struct Buffers {
    void* an = nullptr;    // A 规范副本
    void* bn = nullptr;    // B 规范副本
    void* cn = nullptr;    // C 规范副本
    void* u1t = nullptr;
    void* u2 = nullptr;
    void* v1t = nullptr;
    void* v2n = nullptr;
    void* cout = nullptr;
    void* abuf = nullptr;
    void* bbuf = nullptr;
};

bool layout_buffers(size_t elem, long long n, void* base, Buffers* out) {
    if (n <= 0 || n > kMaxN) {
        return false;
    }
    const long long n2 = n * n;
    const long long n3 = n2 * n;
    const long long part4 = kSpin * kSpin * n2;    // 16N²
    const long long part16n3 = kSpin * kSpin * n3; // 16N³
    const long long elems = 6 * part4 + 2 * n3 + 2 * part16n3;
    if (elems > static_cast<long long>(kWorkspaceLimitBytes / static_cast<long long>(elem))) {
        return false;
    }
    char* cursor = static_cast<char*>(base);
    auto take = [&](long long count) {
        void* p = cursor;
        cursor += static_cast<size_t>(count) * elem;
        return p;
    };
    out->an = take(part4);
    out->bn = take(part4);
    out->cn = take(part4);
    out->u1t = take(part4);
    out->u2 = take(part4);
    out->v1t = take(n3);
    out->v2n = take(n3);
    out->cout = take(part4);
    out->abuf = take(part16n3);
    out->bbuf = take(part16n3);
    return true;
}

template <typename T>
void launch_normalize(const T* src, T* dst, long long n, const long long s[4],
                      cudaStream_t stream) {
    const long long total = kSpin * kSpin * n * n;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    normalize_peram_kernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        src, dst, n, total, s[0], s[1], s[2], s[3]);
}

template <typename T>
int launch_all(const void* const* in, void* out, const Params& p, cudaStream_t stream) {
    const long long n = p.n;
    const long long n2 = n * n;
    const long long n3 = n2 * n;
    const long long part4 = kSpin * kSpin * n2;
    const size_t elem = sizeof(T);

    void* base = nullptr;
    const long long elems = 6 * part4 + 2 * n3 + 2 * (kSpin * kSpin * n3);
    if (elems > static_cast<long long>(kWorkspaceLimitBytes / static_cast<long long>(elem))) {
        return QCD_STATUS_MEMORY;
    }
    if (!get_workspace(static_cast<size_t>(elems) * elem, &base)) {
        return QCD_STATUS_MEMORY;
    }
    Buffers buf;
    if (!layout_buffers(elem, n, base, &buf)) {
        return QCD_STATUS_MEMORY;
    }

    const T* a_in = static_cast<const T*>(in[0]);
    const T* b_in = static_cast<const T*>(in[1]);
    const T* c_in = static_cast<const T*>(in[2]);
    const T* gx = static_cast<const T*>(in[3]);
    const T* gy = static_cast<const T*>(in[4]);
    const T* pr = static_cast<const T*>(in[5]);
    const T* v1 = static_cast<const T*>(in[6]);
    const T* v2 = static_cast<const T*>(in[7]);
    T* an = static_cast<T*>(buf.an);
    T* bn = static_cast<T*>(buf.bn);
    T* cn = static_cast<T*>(buf.cn);
    T* u1t = static_cast<T*>(buf.u1t);
    T* u2 = static_cast<T*>(buf.u2);
    T* v1t = static_cast<T*>(buf.v1t);
    T* v2n = static_cast<T*>(buf.v2n);
    T* cout = static_cast<T*>(buf.cout);
    T* abuf = static_cast<T*>(buf.abuf);
    T* bbuf = static_cast<T*>(buf.bbuf);

    const long long peram_src[4] = {kSpin * n2, n2, n, 1};
    long long sa[4], sb[4], sc[4];
    permuted_strides(peram_src, p.perm[0], sa);
    permuted_strides(peram_src, p.perm[1], sb);
    permuted_strides(peram_src, p.perm[2], sc);
    const long long vvv_src[4] = {n3, n2, n, 1};
    long long sv1[4], sv2[4];
    permuted_strides(vvv_src, p.perm[3], sv1);
    permuted_strides(vvv_src, p.perm[4], sv2);

    // ---- stage 0：三根 peram 归一化成规范紧致序（恒等轴序时直接复用）
    const T* a = (p.perm[0] == QCD_NUCLEON2PT_PERM_IDENTITY) ? a_in : an;
    const T* b = (p.perm[1] == QCD_NUCLEON2PT_PERM_IDENTITY) ? b_in : bn;
    const T* c = (p.perm[2] == QCD_NUCLEON2PT_PERM_IDENTITY) ? c_in : cn;
    if (a != a_in) {
        launch_normalize<T>(a_in, an, n, sa, stream);
    }
    if (b != b_in) {
        launch_normalize<T>(b_in, bn, n, sb, stream);
    }
    if (c != c_in) {
        launch_normalize<T>(c_in, cn, n, sc, stream);
    }

    // ---- stage 1a / 1b
    {
        const int threads = 256;
        const long long blocks = (part4 + threads - 1) / threads;
        fold_gx_kernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            a, gx, u1t, n, part4, (p.matflags & 1) ? 1 : 0);
        fold_gy_pr_kernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            c, gy, pr, u2, n, part4, (p.matflags & 2) ? 1 : 0, (p.matflags & 4) ? 1 : 0);
    }
    int status = map_cuda(cudaGetLastError());
    if (status != QCD_STATUS_OK) {
        return status;
    }

    const int exp_batch = static_cast<int>(kSpin * kSpin * n);
    const int exp_grid_x = static_cast<int>((n + kTileN - 1) / kTileN);
    const int exp_grid_y = static_cast<int>((n + kTileM - 1) / kTileM);
    const int star_grid = static_cast<int>((kSpin * n + kTileN - 1) / kTileN);

    const long long pairs = (p.mode == 0) ? p.mx : (p.mx * p.my);
    for (long long pair = 0; pair < pairs; ++pair) {
        long long my_index = 0;
        long long mx_index = 0;
        if (p.mode == 0) {
            my_index = pair;
            mx_index = pair;
        } else {
            my_index = pair / p.mx;
            mx_index = pair % p.mx;
        }

        // ---- V1 -> V1T
        {
            const long long total = n3;
            const int threads = 256;
            const long long blocks = (total + threads - 1) / threads;
            v1_to_transposed_kernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
                v1, v1t, n, mx_index * vvv_src[0], sv1[1], sv1[2], sv1[3]);
        }
        // ---- V2 -> canonical（非恒等轴序时才复制）
        const T* v2src = v2 + my_index * vvv_src[0];
        if (p.perm[4] != QCD_NUCLEON2PT_PERM_IDENTITY) {
            const long long total = n3;
            const int threads = 256;
            const long long blocks = (total + threads - 1) / threads;
            v2_to_canonical_kernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
                v2, v2n, n, my_index * vvv_src[0], sv2[1], sv2[2], sv2[3]);
            v2src = v2n;
        }

        // ---- Bbuf[c][m][f][n][b] = Σ_d B[m][c][n][d] V1T[f][b][d]
        {
            BatchMap amap, bmap, cmap;
            amap.size[0] = kSpin; amap.size[1] = kSpin; amap.size[2] = n;
            amap.stride[0] = n2;         // c  (B 的规范 c 步长 = N²)
            amap.stride[1] = kSpin * n2; // m
            amap.stride[2] = 0;          // f
            bmap.size[0] = kSpin; bmap.size[1] = kSpin; bmap.size[2] = n;
            bmap.stride[0] = 0;
            bmap.stride[1] = 0;
            bmap.stride[2] = n2;         // f
            cmap.size[0] = kSpin; cmap.size[1] = kSpin; cmap.size[2] = n;
            cmap.stride[0] = kSpin * n3; // c
            cmap.stride[1] = n3;         // m
            cmap.stride[2] = n2;         // f
            dim3 grid(static_cast<unsigned>(exp_grid_x), static_cast<unsigned>(exp_grid_y),
                      static_cast<unsigned>(exp_batch));
            gemm_tn_batched_kernel<T><<<grid, kThreads, 0, stream>>>(
                b, v1t, bbuf, static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                n, n, n, 1, amap, bmap, cmap);
        }
        // ---- Abuf[c][g][p][n][b] = Σ_h V2[n][p][h] U1T[c][g][b][h]
        {
            BatchMap amap, bmap, cmap;
            amap.size[0] = kSpin; amap.size[1] = kSpin; amap.size[2] = n;
            amap.stride[0] = 0;
            amap.stride[1] = 0;
            amap.stride[2] = n2;         // n
            bmap.size[0] = kSpin; bmap.size[1] = kSpin; bmap.size[2] = n;
            bmap.stride[0] = kSpin * n2; // c  (U1T[c][g][b][h] 的 c 步长 = 4N²)
            bmap.stride[1] = n2;         // g
            bmap.stride[2] = 0;
            cmap.size[0] = kSpin; cmap.size[1] = kSpin; cmap.size[2] = n;
            cmap.stride[0] = kSpin * n3; // c
            cmap.stride[1] = n3;         // g
            cmap.stride[2] = n;          // n
            dim3 grid(static_cast<unsigned>(exp_grid_x), static_cast<unsigned>(exp_grid_y),
                      static_cast<unsigned>(exp_batch));
            gemm_tn_batched_kernel<T><<<grid, kThreads, 0, stream>>>(
                v2src, u1t, abuf, static_cast<int>(n), static_cast<int>(n),
                static_cast<int>(n), n, n, n2, 1, amap, bmap, cmap);
        }
        // ---- Cout
        {
            dim3 grid(static_cast<unsigned>(star_grid), static_cast<unsigned>(star_grid));
            star_kernel<T><<<grid, kThreads, 0, stream>>>(abuf, bbuf, cout,
                                                          static_cast<int>(n));
        }
        // ---- 收尾点积
        {
            T* out_slot = (p.mode == 0)
                              ? (static_cast<T*>(out) + mx_index)
                              : (static_cast<T*>(out) + (my_index * p.mx + mx_index));
            status = map_cuda(cudaMemsetAsync(out_slot, 0, elem, stream));
            if (status != QCD_STATUS_OK) {
                return status;
            }
            const int blocks = 24;
            final_dot_kernel<T><<<blocks, 256, 0, stream>>>(u2, cout, out_slot, n, part4,
                                                            blocks);
        }
        status = map_cuda(cudaGetLastError());
        if (status != QCD_STATUS_OK) {
            return status;
        }
    }
    return map_cuda(cudaGetLastError());
}

}  // namespace

extern "C" QCD_NUCLEON2PT_API int qcd_cuda_nucleon_2pt(int dtype_id,
                                                       const void* const* inputs,
                                                       void* output,
                                                       const long long* dims,
                                                       void* stream) {
    if (inputs == nullptr || dims == nullptr || output == nullptr) {
        return QCD_STATUS_INVALID;
    }
    if (dtype_id != QCD_DTYPE_COMPLEX128 && dtype_id != QCD_DTYPE_COMPLEX64) {
        return QCD_STATUS_INVALID;
    }
    Params p;
    p.n = dims[0];
    p.mx = dims[1];
    p.my = dims[2];
    p.mode = static_cast<int>(dims[3]);
    p.perm[0] = static_cast<int>(dims[4]);
    p.perm[1] = static_cast<int>(dims[5]);
    p.perm[2] = static_cast<int>(dims[6]);
    p.perm[3] = static_cast<int>(dims[7]);
    p.perm[4] = static_cast<int>(dims[8]);
    p.matflags = static_cast<int>(dims[9]);

    if (p.n < 1 || p.n > kMaxN || p.mx < 0 || p.my < 0) {
        return QCD_STATUS_INVALID;
    }
    if (p.mode != 0 && p.mode != 1) {
        return QCD_STATUS_INVALID;
    }
    if (p.mode == 0) {
        if (p.mx != p.my || p.mx < 1) {
            return QCD_STATUS_INVALID;
        }
    } else if (p.mx < 1 || p.my < 1) {
        return QCD_STATUS_INVALID;
    }
    if (p.matflags < 0 || p.matflags > 7) {
        return QCD_STATUS_INVALID;
    }
    for (int i = 0; i < 5; ++i) {
        if (!perm_valid(p.perm[i], 4)) {
            return QCD_STATUS_INVALID;
        }
        int digits[4];
        perm_digits(p.perm[i], digits);
        if (i >= 3 && digits[0] != 0) {  // V1/V2 的动量轴必须留在第 0 维
            return QCD_STATUS_INVALID;
        }
    }
    for (int i = 0; i < QCD_NUCLEON2PT_INPUTS; ++i) {
        if (inputs[i] == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }

    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    if (dtype_id == QCD_DTYPE_COMPLEX128) {
        return launch_all<double2>(inputs, output, p, s);
    }
    return launch_all<float2>(inputs, output, p, s);
}
