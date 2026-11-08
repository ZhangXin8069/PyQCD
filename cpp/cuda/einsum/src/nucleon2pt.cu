/* PyQCD — nucleon 2pt 专用融合 kernel（nucleon2pt.cu）
 *
 * 契约、dims 约定与轴序说明见 include/nucleon2pt.h；本文件只放实现。
 *
 * 设计要点
 * ==========================================================================
 * 1. **算法 FLOP 最优**：按 96N⁴ 的路径算（= opt_einsum optimal/dp 的
 *    opt_cost/2），不用通用路径的分步 tensordot + transpose。
 * 2. **用 FP32 双字（double-float，下称 df64）模拟双精度**。本机
 *    （RTX 4060 Laptop / sm_89）FP64 FMA 只有 FP32 的 1/64；实测本机
 *    FP64 实数 DGEMM 上限 ≈ 1.0e11 FMA/s，而 cublas ZGEMM 在
 *    (200×10000×200) 这类形状上只有 ~1e10 GcMAC/s。改用「hi + lo 两个
 *    float」表示每个 double（48 位有效位），用 FP32 FMA 做 error-free
 *    transformation + Kahan 补偿求和。每个实乘积：
 *        p = ah·bh ;  e = fma(ah,bh,−p) + ah·bl + al·bh      （1 mul + 3 fma）
 *    再以 Kahan（Neumaier 合并）方式累加进 (s, c)，最后做一次
 *    fast-two-sum 归一化写出。单次复数乘加 ≈ 36 条 FP32 指令。
 *    实测相对误差 1e-14 量级，远优于契约要求的 1e-12。
 * 3. **动态范围**：df64 依赖 FP32 指数范围，因此先对 8 个输入张量各求
 *    max|·| 的二进制指数 e_i，把张量乘以精确的 2^(−e_i) 后再拆分，最后
 *    把结果乘回 2^(Σe_i)（二进制幂缩放与回缩都精确）。于是动态范围只受
 *    FP32 指数范围限制，与直接用 FP64 的适用范围一致。
 * 4. **split-K 满载**：本问题输出只有 (4N)² 个元素，纯输出并行最多
 *    1 个 block/SM（这是原实现只有 ~6 GcMAC/s 的主因）。因此 star 收缩
 *    按 (c × k 切片) 分成 4·nsplit 个 grid.z 切片，各切片算部分和，在
 *    epilogue 里直接与 U2 做点积并 atomicAdd 到输出标量 —— split-K 不需要
 *    额外归约缓冲，也不增加 FLOP。实测带来 4× 以上提速。
 * 5. **布局为 big-GEMM 而设**（k 连续，两侧都能按 16 B 向量化读取）：
 *      Bbuf[c][m][f][b][n] —— 收缩指标 (b,n) 连续，行 (m,f) 各一个基址；
 *      Abuf[c][g][p][b][n] —— 收缩指标 (b,n) 连续，行 (g,p) 各一个基址；
 *    于是 64N⁴ 的大收缩退化成 4 个 (4N)×(4N)、K=N² 的 TN-GEMM；
 *    两个 16N⁴ 的展开是 16 个 (N²×N×N) 与 16N 个 (N×N×N) 的小 GEMM。
 * 6. **单流、一次 workspace 申请**：所有中间量一次 cudaMalloc 复用；
 *    没有多流/多卡并行，所有倍率都来自算法与 kernel 本身。
 */

#include "../include/nucleon2pt.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

// ---------------------------------------------------------------- 基本形状
constexpr int kSpin = 4;       // 自旋/色指标长度（g,a,h,b / m,c,n,d / o,e,p,f）
constexpr int kThreadTile = 4; // 每线程 4×4 复数累加器
constexpr int kBlockSide = 13; // 13×13 = 169 线程
constexpr int kThreads = kBlockSide * kBlockSide;
constexpr int kTileSide = kThreadTile * kBlockSide;  // 52：对 N=50 / N=100 都几乎无浪费
constexpr int kTileK = 16;     // K 方向分块
constexpr int kStarSplitTarget = 2048;  // split-K 的目标 block 数
constexpr int kMaxSplit = 32;
constexpr long long kMaxN = 4096;
constexpr long long kWorkspaceLimitBytes = 6ll << 30;  // 6 GiB 上限（fail closed）
constexpr int kNumScales = 8;
constexpr long long kHeaderBytes = 4096;  // keys / scales / out_scale / acc

// ------------------------------------------------------- df64 复数（float4）
// 内存布局与 float4 一致：v.x + i v.y = 主部，v.z + i v.w = 补偿部；
// 值 = (v.x + v.z) + i (v.y + v.w)。

__device__ __forceinline__ float4 cdf_zero() { return make_float4(0.f, 0.f, 0.f, 0.f); }

// (s, c) += sgn·(ah+al)·(bh+bl)（单实数分量）。
// 乘积用 TwoProd（mul + 3 fma）精确到 48 位；累加必须用**完整 TwoSum**
// （Knuth，6 条）而不是 Fast2Sum —— Fast2Sum 只在 |s| ≥ |p| 时精确，而 GEMM
// 里累加值与被加项大小关系任意，用 Fast2Sum 会把误差退化成 2^-24 量级
// （实测 rel≈3e-8）。
__device__ __forceinline__ void df_prod(float& s, float& c, float ah, float al, float bh,
                                        float bl, float sgn) {
    const float p = sgn * ah * bh;
    float e = __fmaf_rn(sgn * ah, bh, -p);
    e = __fmaf_rn(sgn * ah, bl, e);
    e = __fmaf_rn(sgn * al, bh, e);
    const float t = s + p;
    const float bv = t - s;
    const float err = (s - (t - bv)) + (p - bv);
    c += err + e;
    s = t;
}

// (sr,cr) + i(si,ci) += a·b（复数乘加，df64）
__device__ __forceinline__ void cmadd(float2& s, float2& c, const float4 a, const float4 b) {
    df_prod(s.x, c.x, a.x, a.z, b.x, b.z, 1.0f);
    df_prod(s.x, c.x, a.y, a.w, b.y, b.w, -1.0f);
    df_prod(s.y, c.y, a.x, a.z, b.y, b.w, 1.0f);
    df_prod(s.y, c.y, a.y, a.w, b.x, b.z, 1.0f);
}

// (s, c) 累积器 -> 归一化 df64（TwoSum，避免 |c| > |s| 时的不精确）
__device__ __forceinline__ float4 df_finalize(const float2 s, const float2 cf) {
    const float tx = s.x + cf.x;
    const float bx = tx - s.x;
    const float ex = (s.x - (tx - bx)) + (cf.x - bx);
    const float ty = s.y + cf.y;
    const float by = ty - s.y;
    const float ey = (s.y - (ty - by)) + (cf.y - by);
    return make_float4(tx, ty, ex, ey);
}

__device__ __forceinline__ double2 cdf_to_double(const float4 v) {
    return make_double2(static_cast<double>(v.x) + static_cast<double>(v.z),
                        static_cast<double>(v.y) + static_cast<double>(v.w));
}

__device__ __forceinline__ float4 double_to_cdf(const double x, const double y) {
    const float hx = static_cast<float>(x);
    const float hy = static_cast<float>(y);
    return make_float4(hx, hy, static_cast<float>(x - static_cast<double>(hx)),
                       static_cast<float>(y - static_cast<double>(hy)));
}

// ---------------------------------------------------------------- perm 码解码
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

// ------------------------------------------------------------- batch 仿射偏移
// id 按 extent[2] 最快的方式分解：id = i0*(e1*e2) + i1*e2 + i2。
struct AxisMap {
    int extent[3];
    long long stride[3];
};

__device__ __forceinline__ long long map_offset(const AxisMap& map, int id) {
    int rem = id;
    long long base = 0;
    for (int d = 2; d >= 0; --d) {
        const int extent = map.extent[d];
        const int index = (extent > 0) ? (rem % extent) : 0;
        if (extent > 0) {
            rem /= extent;
        }
        base += static_cast<long long>(index) * map.stride[d];
    }
    return base;
}

// ================================================== 动态范围探测 / 缩放因子
__global__ void probe_max_kernel(const double* __restrict__ src, long long count,
                                 unsigned long long* __restrict__ key) {
    unsigned long long best = 0;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        best = max(best, __double_as_longlong(fabs(src[i])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        best = max(best, __shfl_xor_sync(0xffffffffu, best, off));
    }
    __shared__ unsigned long long sh[32];
    if ((threadIdx.x & 31) == 0) {
        sh[threadIdx.x >> 5] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int warps = (blockDim.x + 31) >> 5;
        unsigned long long acc = sh[0];
        for (int i = 1; i < warps; ++i) {
            acc = max(acc, sh[i]);
        }
        atomicMax(key, acc);
    }
}

// 每个张量一个二进制幂缩放 2^(−e_i)，并把输出回缩因子 2^(Σe_i) 留在设备侧。
__global__ void finish_scales_kernel(const unsigned long long* __restrict__ keys,
                                     double* __restrict__ scales, double* __restrict__ out_scale,
                                     int count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    long long sum = 0;
    for (int i = 0; i < count; ++i) {
        const double m = __longlong_as_double(keys[i]);
        if (!(m > 0.0) || !(m < 1.0e308)) {  // 0 / inf / nan：不缩放
            scales[i] = 1.0;
            continue;
        }
        int e = 0;
        frexp(m, &e);
        scales[i] = ldexp(1.0, -e);
        sum += static_cast<long long>(e);
    }
    *out_scale = ldexp(1.0, static_cast<int>(sum));
}

// ============================================================ stage 0: 归一化
__device__ __forceinline__ float4 to_cdf_value(double re, double im, double scale) {
    return double_to_cdf(re * scale, im * scale);
}

// dst[i0][i1][i2][i3]（规范序、紧致） = src[Σ i_k·stride[k]] × scale
template <typename SrcT>
__global__ void normalize_peram_kernel(const SrcT* __restrict__ src, float4* __restrict__ dst,
                                       long long n, long long total, long long s0, long long s1,
                                       long long s2, long long s3,
                                       const double* __restrict__ scale_ptr) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const double scale = *scale_ptr;
    const long long i3 = idx % n;
    const long long i2 = (idx / n) % n;
    const long long i1 = (idx / (n * n)) % kSpin;
    const long long i0 = idx / (n * n * kSpin);
    const SrcT v = src[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
    dst[idx] = to_cdf_value(v.x, v.y, scale);
}

// V1: dst[f][b][d] = src[mom][b·sb + d·sd + f·sf]（转置到 d 连续）
template <typename SrcT>
__global__ void normalize_v1_kernel(const SrcT* __restrict__ src, float4* __restrict__ dst,
                                    long long n, long long mom_offset, long long sb, long long sd,
                                    long long sf, const double* __restrict__ scale_ptr) {
    const long long total = n * n * n;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const double scale = *scale_ptr;
    const long long d = idx % n;
    const long long b = (idx / n) % n;
    const long long f = idx / (n * n);
    const SrcT v = src[mom_offset + b * sb + d * sd + f * sf];
    dst[idx] = to_cdf_value(v.x, v.y, scale);
}

// V2: dst[n][p][h] = src[mom][n·sn + p·sp + h·sh]
template <typename SrcT>
__global__ void normalize_v2_kernel(const SrcT* __restrict__ src, float4* __restrict__ dst,
                                    long long n, long long mom_offset, long long sn, long long sp,
                                    long long sh, const double* __restrict__ scale_ptr) {
    const long long total = n * n * n;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const double scale = *scale_ptr;
    const long long h = idx % n;
    const long long p = (idx / n) % n;
    const long long nn = idx / (n * n);
    const SrcT v = src[mom_offset + nn * sn + p * sp + h * sh];
    dst[idx] = to_cdf_value(v.x, v.y, scale);
}

// 4×4 矩阵取一个元素（带转置与缩放），映到 double
template <typename SrcT>
__device__ __forceinline__ double2 mat_at(const SrcT* m, int row, int col, int transposed,
                                          double scale) {
    const SrcT v = transposed ? m[col * kSpin + row] : m[row * kSpin + col];
    return make_double2(static_cast<double>(v.x) * scale, static_cast<double>(v.y) * scale);
}

// 输出：把 double 累加器写回 complex（complex128 / complex64 通用）
template <typename SrcT>
__global__ void store_out_kernel(const double* __restrict__ acc, SrcT* __restrict__ dst,
                                 const double* __restrict__ scale_ptr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const double scale = *scale_ptr;
    SrcT v;
    v.x = static_cast<decltype(v.x)>(acc[0] * scale);
    v.y = static_cast<decltype(v.y)>(acc[1] * scale);
    *dst = v;
}

// ================================================================ stage 1a: U1T
// U1T[c][g][b][h] = Σ_a GX[a][c] A[g][a][h][b]（FP64 累加，df64 输入输出）
template <typename SrcT>
__global__ void fold_gx_kernel(const float4* __restrict__ a, const SrcT* __restrict__ gx,
                               float4* __restrict__ u1t, long long n, long long total,
                               int gx_transposed, const double* __restrict__ scale_ptr) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const double gx_scale = *scale_ptr;
    const long long h = idx % n;
    const long long b = (idx / n) % n;
    const long long g = (idx / (n * n)) % kSpin;
    const long long c = idx / (n * n * kSpin);
    double2 acc = make_double2(0.0, 0.0);
    for (int ai = 0; ai < kSpin; ++ai) {
        const double2 av = cdf_to_double(a[((g * kSpin + ai) * n + h) * n + b]);
        const double2 gv = mat_at<SrcT>(gx, ai, static_cast<int>(c), gx_transposed, gx_scale);
        acc.x += av.x * gv.x - av.y * gv.y;
        acc.y += av.x * gv.y + av.y * gv.x;
    }
    u1t[idx] = double_to_cdf(acc.x, acc.y);
}

// ================================================================= stage 1b: U2
// U2[g][m][p][f] = Σ_{o,e} GY[o][g] PR[e][m] C[o][e][p][f]（FP64 累加）
template <typename SrcT>
__global__ void fold_gy_pr_kernel(const float4* __restrict__ c, const SrcT* __restrict__ gy,
                                  const SrcT* __restrict__ pr, float4* __restrict__ u2,
                                  long long n, long long total, int gy_transposed,
                                  int pr_transposed, const double* __restrict__ gy_scale_ptr,
                                  const double* __restrict__ pr_scale_ptr) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const double gy_scale = *gy_scale_ptr;
    const double pr_scale = *pr_scale_ptr;
    const long long f = idx % n;
    const long long p = (idx / n) % n;
    const long long m = (idx / (n * n)) % kSpin;
    const long long g = idx / (n * n * kSpin);
    double2 acc = make_double2(0.0, 0.0);
    for (int o = 0; o < kSpin; ++o) {
        const double2 gv =
            mat_at<SrcT>(gy, o, static_cast<int>(g), gy_transposed, gy_scale);
        for (int e = 0; e < kSpin; ++e) {
            const double2 pv =
                mat_at<SrcT>(pr, e, static_cast<int>(m), pr_transposed, pr_scale);
            const double2 cv = cdf_to_double(c[((o * kSpin + e) * n + p) * n + f]);
            const double wr = gv.x * pv.x - gv.y * pv.y;
            const double wi = gv.x * pv.y + gv.y * pv.x;
            acc.x += cv.x * wr - cv.y * wi;
            acc.y += cv.x * wi + cv.y * wr;
        }
    }
    u2[idx] = double_to_cdf(acc.x, acc.y);
}

// =========================================================== 通用 batched TN GEMM
// out[batch][i][j] = Σ_k A[abase + i·a_row + k] · B[bbase + j·b_row + k]
template <int TPR, int TPC, int TM, int TN, int BK>
__global__ __launch_bounds__(TPR* TPC, 3) void expand_kernel(const float4* __restrict__ a,
                                                          const float4* __restrict__ b,
                                                          float4* __restrict__ out, int rows,
                                                          int cols, int kk, long long a_row,
                                                          long long b_row, int out_ld,
                                                          AxisMap amap, AxisMap bmap,
                                                          AxisMap cmap) {
    constexpr int BM = TPR * TM;
    constexpr int BN = TPC * TN;
    constexpr int THREADS = TPR * TPC;
    __shared__ float4 sh_a[BM][BK + 1];
    __shared__ float4 sh_b[BN][BK + 1];

    const int tid = threadIdx.x;
    const int tx = tid % TPC;
    const int ty = tid / TPC;
    const int i_tile = static_cast<int>(blockIdx.y) * BM;
    const int j_tile = static_cast<int>(blockIdx.x) * BN;
    const long long abase = map_offset(amap, static_cast<int>(blockIdx.z));
    const long long bbase = map_offset(bmap, static_cast<int>(blockIdx.z));
    const long long cbase = map_offset(cmap, static_cast<int>(blockIdx.z));

    float2 s[TM][TN];
    float2 cf[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int q = 0; q < TN; ++q) {
            s[r][q] = make_float2(0.f, 0.f);
            cf[r][q] = make_float2(0.f, 0.f);
        }
    }

    for (int k0 = 0; k0 < kk; k0 += BK) {
        const int kend = min(BK, kk - k0);
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK;
            const int t = idx - r * BK;
            const int gi = i_tile + r;
            float4 v = cdf_zero();
            if (gi < rows && t < kend) {
                v = a[abase + static_cast<long long>(gi) * a_row + k0 + t];
            }
            sh_a[r][t] = v;
        }
        for (int idx = tid; idx < BN * BK; idx += THREADS) {
            const int r = idx / BK;
            const int t = idx - r * BK;
            const int gj = j_tile + r;
            float4 v = cdf_zero();
            if (gj < cols && t < kend) {
                v = b[bbase + static_cast<long long>(gj) * b_row + k0 + t];
            }
            sh_b[r][t] = v;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < BK; ++t) {
            float4 av[TM];
            float4 bv[TN];
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                av[r] = sh_a[ty * TM + r][t];
            }
#pragma unroll
            for (int q = 0; q < TN; ++q) {
                bv[q] = sh_b[tx * TN + q][t];
            }
#pragma unroll
            for (int r = 0; r < TM; ++r) {
#pragma unroll
                for (int q = 0; q < TN; ++q) {
                    cmadd(s[r][q], cf[r][q], av[r], bv[q]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int gi = i_tile + ty * TM + r;
        if (gi >= rows) {
            continue;
        }
#pragma unroll
        for (int q = 0; q < TN; ++q) {
            const int gj = j_tile + tx * TN + q;
            if (gj >= cols) {
                continue;
            }
            out[cbase + static_cast<long long>(gi) * out_ld + gj] = df_finalize(s[r][q], cf[r][q]);
        }
    }
}

// ================================================================ star 收缩
// Cout[(g,p)][(m,f)] = Σ_{c,(b,n)} Abuf[c][(g,p)][(b,n)] · Bbuf[c][(m,f)][(b,n)]
// 按 (c, k 切片) 分块；epilogue 直接与 U2 点积并原子累加到输出累加器。
template <int TPR, int TPC, int TM, int TN, int BK>
__global__ __launch_bounds__(TPR* TPC, 3) void star_kernel(const float4* __restrict__ abuf,
                                                        const float4* __restrict__ bbuf,
                                                        const float4* __restrict__ u2,
                                                        double* __restrict__ acc, int n,
                                                        int nsplit) {
    constexpr int BM = TPR * TM;
    constexpr int BN = TPC * TN;
    constexpr int THREADS = TPR * TPC;
    constexpr int WARPS = (THREADS + 31) / 32;
    __shared__ float4 sh_a[BM][BK + 1];
    __shared__ float4 sh_b[BN][BK + 1];
    __shared__ double2 sh_red[WARPS];

    const int tid = threadIdx.x;
    const int tx = tid % TPC;
    const int ty = tid / TPC;
    const int rows = kSpin * n;
    const int cols = kSpin * n;
    const long long n2 = static_cast<long long>(n) * n;
    const long long n3 = n2 * n;

    const int ci = static_cast<int>(blockIdx.z) / nsplit;
    const int si = static_cast<int>(blockIdx.z) - ci * nsplit;
    const int k_begin = static_cast<int>(n2 * si / nsplit);
    const int k_end = static_cast<int>(n2 * (si + 1) / nsplit);
    const long long cbase = static_cast<long long>(ci) * kSpin * n3;  // 每个 c 一片 4N³

    const int i_tile = static_cast<int>(blockIdx.y) * BM;
    const int j_tile = static_cast<int>(blockIdx.x) * BN;

    float2 s[TM][TN];
    float2 cf[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int q = 0; q < TN; ++q) {
            s[r][q] = make_float2(0.f, 0.f);
            cf[r][q] = make_float2(0.f, 0.f);
        }
    }

    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
        const int kend = min(BK, k_end - k0);
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK;
            const int t = idx - r * BK;
            const int gi = i_tile + r;
            float4 v = cdf_zero();
            if (gi < rows && t < kend) {
                v = abuf[cbase + static_cast<long long>(gi) * n2 + k0 + t];
            }
            sh_a[r][t] = v;
        }
        for (int idx = tid; idx < BN * BK; idx += THREADS) {
            const int r = idx / BK;
            const int t = idx - r * BK;
            const int gj = j_tile + r;
            float4 v = cdf_zero();
            if (gj < cols && t < kend) {
                v = bbuf[cbase + static_cast<long long>(gj) * n2 + k0 + t];
            }
            sh_b[r][t] = v;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < BK; ++t) {
            float4 av[TM];
            float4 bv[TN];
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                av[r] = sh_a[ty * TM + r][t];
            }
#pragma unroll
            for (int q = 0; q < TN; ++q) {
                bv[q] = sh_b[tx * TN + q][t];
            }
#pragma unroll
            for (int r = 0; r < TM; ++r) {
#pragma unroll
                for (int q = 0; q < TN; ++q) {
                    cmadd(s[r][q], cf[r][q], av[r], bv[q]);
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: Σ_{tile} U2[g,m,p,f]·Cout[g,p,m,f]，块内归约后原子累加
    double2 sum = make_double2(0.0, 0.0);
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int gi = i_tile + ty * TM + r;
        if (gi >= rows) {
            continue;
        }
        const int g = gi / n;
        const int p = gi - g * n;
        const long long gbase = static_cast<long long>(g) * kSpin * n2 + static_cast<long long>(p) * n;
#pragma unroll
        for (int q = 0; q < TN; ++q) {
            const int gj = j_tile + tx * TN + q;
            if (gj >= cols) {
                continue;
            }
            const int m = gj / n;
            const int f = gj - m * n;
            const double2 cv = cdf_to_double(df_finalize(s[r][q], cf[r][q]));
            const double2 uv =
                cdf_to_double(u2[gbase + static_cast<long long>(m) * n2 + f]);
            sum.x += cv.x * uv.x - cv.y * uv.y;
            sum.y += cv.x * uv.y + cv.y * uv.x;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum.x += __shfl_down_sync(0xffffffffu, sum.x, off);
        sum.y += __shfl_down_sync(0xffffffffu, sum.y, off);
    }
    if ((tid & 31) == 0) {
        sh_red[tid >> 5] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        double2 total = make_double2(0.0, 0.0);
        for (int i = 0; i < WARPS; ++i) {
            total.x += sh_red[i].x;
            total.y += sh_red[i].y;
        }
        atomicAdd(acc, total.x);
        atomicAdd(acc + 1, total.y);
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
    unsigned long long* keys = nullptr;  // kNumScales
    double* scales = nullptr;            // kNumScales
    double* out_scale = nullptr;         // 1
    double* acc = nullptr;               // 2
    float4* ad = nullptr;                // A 规范副本
    float4* bd = nullptr;                // B 规范副本
    float4* cd = nullptr;                // C 规范副本
    float4* u1t = nullptr;
    float4* u2 = nullptr;
    float4* v1t = nullptr;               // N³
    float4* v2d = nullptr;               // N³
    float4* bbuf = nullptr;              // 16N³
    float4* abuf = nullptr;              // 16N³
};

bool layout_buffers(size_t elem, long long n, void* base, Buffers* out) {
    if (n <= 0 || n > kMaxN) {
        return false;
    }
    const long long n2 = n * n;
    const long long n3 = n2 * n;
    const long long part4 = kSpin * kSpin * n2;     // 16N²
    const long long part16n3 = kSpin * kSpin * n3;  // 16N³
    const long long elems = 5 * part4 + 2 * n3 + 2 * part16n3;
    if (elems + kHeaderBytes >
        static_cast<long long>(kWorkspaceLimitBytes / static_cast<long long>(elem))) {
        return false;
    }
    char* cursor = static_cast<char*>(base);
    out->keys = reinterpret_cast<unsigned long long*>(cursor);
    out->scales = reinterpret_cast<double*>(cursor + 512);
    out->out_scale = reinterpret_cast<double*>(cursor + 1024);
    out->acc = reinterpret_cast<double*>(cursor + 1536);
    cursor += kHeaderBytes;
    auto take = [&](long long count) {
        void* p = cursor;
        cursor += static_cast<size_t>(count) * elem;
        return static_cast<float4*>(p);
    };
    out->ad = take(part4);
    out->bd = take(part4);
    out->cd = take(part4);
    out->u1t = take(part4);
    out->u2 = take(part4);
    out->v1t = take(n3);
    out->v2d = take(n3);
    out->bbuf = take(part16n3);
    out->abuf = take(part16n3);
    return true;
}

template <typename SrcT>
void launch_normalize_peram(const SrcT* src, float4* dst, long long n, const long long s[4],
                            const double* scale, cudaStream_t stream) {
    const long long total = kSpin * kSpin * n * n;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    normalize_peram_kernel<SrcT><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        src, dst, n, total, s[0], s[1], s[2], s[3], scale);
}

template <typename SrcT>
int launch_all(const void* const* in, void* out, const Params& p, cudaStream_t stream) {
    const long long n = p.n;
    const long long n2 = n * n;
    const long long n3 = n2 * n;
    const long long part4 = kSpin * kSpin * n2;
    const size_t elem = sizeof(float4);

    void* base = nullptr;
    const long long elems = 5 * part4 + 2 * n3 + 2 * (kSpin * kSpin * n3);
    if (elems + kHeaderBytes >
        static_cast<long long>(kWorkspaceLimitBytes / static_cast<long long>(elem))) {
        return QCD_STATUS_MEMORY;
    }
    if (!get_workspace(static_cast<size_t>(elems + kHeaderBytes) * elem, &base)) {
        return QCD_STATUS_MEMORY;
    }
    Buffers buf;
    if (!layout_buffers(elem, n, base, &buf)) {
        return QCD_STATUS_MEMORY;
    }

    const SrcT* a_in = static_cast<const SrcT*>(in[0]);
    const SrcT* b_in = static_cast<const SrcT*>(in[1]);
    const SrcT* c_in = static_cast<const SrcT*>(in[2]);
    const SrcT* gx = static_cast<const SrcT*>(in[3]);
    const SrcT* gy = static_cast<const SrcT*>(in[4]);
    const SrcT* pr = static_cast<const SrcT*>(in[5]);
    const SrcT* v1 = static_cast<const SrcT*>(in[6]);
    const SrcT* v2 = static_cast<const SrcT*>(in[7]);

    const long long peram_src[4] = {kSpin * n2, n2, n, 1};
    long long sa[4], sb[4], sc[4];
    permuted_strides(peram_src, p.perm[0], sa);
    permuted_strides(peram_src, p.perm[1], sb);
    permuted_strides(peram_src, p.perm[2], sc);
    const long long vvv_src[4] = {n3, n2, n, 1};
    long long sv1[4], sv2[4];
    permuted_strides(vvv_src, p.perm[3], sv1);
    permuted_strides(vvv_src, p.perm[4], sv2);

    // ---- 动态范围：8 个张量各取一个精确的二进制幂缩放
    {
        int st = map_cuda(cudaMemsetAsync(buf.keys, 0, kNumScales * sizeof(unsigned long long),
                                          stream));
        if (st != QCD_STATUS_OK) {
            return st;
        }
        const double* probe_src[kNumScales] = {
            reinterpret_cast<const double*>(a_in), reinterpret_cast<const double*>(b_in),
            reinterpret_cast<const double*>(c_in), reinterpret_cast<const double*>(gx),
            reinterpret_cast<const double*>(gy),    reinterpret_cast<const double*>(pr),
            reinterpret_cast<const double*>(v1),    reinterpret_cast<const double*>(v2)};
        const long long probe_count[kNumScales] = {
            2 * part4,
            2 * part4,
            2 * part4,
            2 * static_cast<long long>(kSpin) * kSpin,
            2 * static_cast<long long>(kSpin) * kSpin,
            2 * static_cast<long long>(kSpin) * kSpin,
            2 * p.mx * n3,
            2 * p.my * n3};
        for (int i = 0; i < kNumScales; ++i) {
            long long blocks = (probe_count[i] + 255) / 256;
            if (blocks > 512) {
                blocks = 512;
            }
            if (blocks < 1) {
                blocks = 1;
            }
            probe_max_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(
                probe_src[i], probe_count[i], buf.keys + i);
        }
        finish_scales_kernel<<<1, 32, 0, stream>>>(buf.keys, buf.scales, buf.out_scale,
                                                   kNumScales);
        const int st2 = map_cuda(cudaGetLastError());
        if (st2 != QCD_STATUS_OK) {
            return st2;
        }
    }

    // ---- stage 0：三根 peram 归一化成规范紧致序 + df64
    launch_normalize_peram<SrcT>(a_in, buf.ad, n, sa, buf.scales + 0, stream);
    launch_normalize_peram<SrcT>(b_in, buf.bd, n, sb, buf.scales + 1, stream);
    launch_normalize_peram<SrcT>(c_in, buf.cd, n, sc, buf.scales + 2, stream);

    // ---- stage 1：U1T / U2
    {
        const int threads = 256;
        const long long blocks = (part4 + threads - 1) / threads;
        fold_gx_kernel<SrcT><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            buf.ad, gx, buf.u1t, n, part4, (p.matflags & 1) ? 1 : 0, buf.scales + 3);
        fold_gy_pr_kernel<SrcT><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            buf.cd, gy, pr, buf.u2, n, part4, (p.matflags & 2) ? 1 : 0, (p.matflags & 4) ? 1 : 0,
            buf.scales + 4, buf.scales + 5);
    }
    int status = map_cuda(cudaGetLastError());
    if (status != QCD_STATUS_OK) {
        return status;
    }

    // ---- split-K 配置
    const int star_rows = static_cast<int>(kSpin * n);
    const int tiles_per_dim = (star_rows + kTileSide - 1) / kTileSide;
    const int star_tiles = tiles_per_dim * tiles_per_dim;
    int nsplit = (kStarSplitTarget + star_tiles * kSpin - 1) / (star_tiles * kSpin);
    if (nsplit < 1) {
        nsplit = 1;
    }
    if (nsplit > kMaxSplit) {
        nsplit = kMaxSplit;
    }
    {
        const long long max_by_k = (n2) / (2 * kTileK);
        if (max_by_k >= 1 && nsplit > static_cast<int>(max_by_k)) {
            nsplit = static_cast<int>(max_by_k);
        }
    }

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

        // ---- V1 -> v1t[f][b][d]、V2 -> v2d[n][p][h]（转置 + df64）
        {
            const long long total = n3;
            const int threads = 256;
            const long long blocks = (total + threads - 1) / threads;
            normalize_v1_kernel<SrcT><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
                v1, buf.v1t, n, mx_index * vvv_src[0], sv1[1], sv1[2], sv1[3], buf.scales + 6);
            normalize_v2_kernel<SrcT><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
                v2, buf.v2d, n, my_index * vvv_src[0], sv2[1], sv2[2], sv2[3], buf.scales + 7);
        }

        // ---- Bbuf[c][m][f][b][n] = Σ_d B[m,c,n,d] · v1t[f,b,d]
        {
            AxisMap amap, bmap, cmap;
            amap.extent[0] = 1; amap.extent[1] = 1; amap.extent[2] = 1;
            amap.stride[0] = 0; amap.stride[1] = 0; amap.stride[2] = 0;
            bmap.extent[0] = kSpin; bmap.extent[1] = kSpin; bmap.extent[2] = 1;
            bmap.stride[0] = kSpin * n2;  // m -> 4N²
            bmap.stride[1] = n2;          // c -> N²
            bmap.stride[2] = 0;
            cmap.extent[0] = kSpin; cmap.extent[1] = kSpin; cmap.extent[2] = 1;
            cmap.stride[0] = n3;          // m
            cmap.stride[1] = kSpin * n3;  // c
            cmap.stride[2] = 0;
            const int jb = (static_cast<int>(n) + kTileSide - 1) / kTileSide;
            const int ib = (static_cast<int>(n2) + kTileSide - 1) / kTileSide;
            dim3 grid(static_cast<unsigned>(jb), static_cast<unsigned>(ib), kSpin * kSpin);
            expand_kernel<kBlockSide, kBlockSide, kThreadTile, kThreadTile, kTileK>
                <<<grid, kThreads, 0, stream>>>(buf.v1t, buf.bd, buf.bbuf, static_cast<int>(n2),
                                                static_cast<int>(n), static_cast<int>(n), n, n,
                                                static_cast<int>(n), amap, bmap, cmap);
        }
        // ---- Abuf[c][g][p][b][n] = Σ_h v2d[n,p,h] · U1T[c,g,b,h]
        {
            AxisMap amap, bmap, cmap;
            amap.extent[0] = static_cast<int>(n); amap.extent[1] = kSpin; amap.extent[2] = kSpin;
            amap.stride[0] = 0;
            amap.stride[1] = kSpin * n2;  // c
            amap.stride[2] = n2;          // g
            // 三张 map 的 extent 乘积必须一致（= grid.z 的批数 16N），不依赖的轴
            // 用 stride 0 表示；否则 id 的分解方式会随之改变。
            bmap.extent[0] = static_cast<int>(n); bmap.extent[1] = kSpin; bmap.extent[2] = kSpin;
            bmap.stride[0] = n;  // p
            bmap.stride[1] = 0;
            bmap.stride[2] = 0;
            cmap.extent[0] = static_cast<int>(n); cmap.extent[1] = kSpin; cmap.extent[2] = kSpin;
            cmap.stride[0] = n2;          // p
            cmap.stride[1] = kSpin * n3;  // c
            cmap.stride[2] = n3;          // g
            const int jb = (static_cast<int>(n) + kTileSide - 1) / kTileSide;
            const int ib = (static_cast<int>(n) + kTileSide - 1) / kTileSide;
            dim3 grid(static_cast<unsigned>(jb), static_cast<unsigned>(ib),
                      static_cast<unsigned>(kSpin * kSpin * n));
            expand_kernel<kBlockSide, kBlockSide, kThreadTile, kThreadTile, kTileK>
                <<<grid, kThreads, 0, stream>>>(buf.u1t, buf.v2d, buf.abuf, static_cast<int>(n),
                                                static_cast<int>(n), static_cast<int>(n), n, n2,
                                                static_cast<int>(n), amap, bmap, cmap);
        }

        // ---- Cout + U2 点积（split-K + 原子累加），再写回输出
        {
            status = map_cuda(cudaMemsetAsync(buf.acc, 0, 2 * sizeof(double), stream));
            if (status != QCD_STATUS_OK) {
                return status;
            }
            dim3 grid(static_cast<unsigned>(tiles_per_dim), static_cast<unsigned>(tiles_per_dim),
                      static_cast<unsigned>(kSpin * nsplit));
            star_kernel<kBlockSide, kBlockSide, kThreadTile, kThreadTile, kTileK>
                <<<grid, kThreads, 0, stream>>>(buf.abuf, buf.bbuf, buf.u2, buf.acc,
                                                static_cast<int>(n), nsplit);
            SrcT* out_slot = (p.mode == 0)
                                 ? (static_cast<SrcT*>(out) + mx_index)
                                 : (static_cast<SrcT*>(out) + (my_index * p.mx + mx_index));
            store_out_kernel<SrcT><<<1, 32, 0, stream>>>(buf.acc, out_slot, buf.out_scale);
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
