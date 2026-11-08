/* PyQCD — nucleon 2pt 专用融合 kernel（nucleon2pt.cu）
 *
 * 契约、dims 约定与轴序说明见 include/nucleon2pt.h；本文件只放实现。
 *
 * 规范方程（唯一形态）
 * ------------------------------------------------------------------
 *   out[my][mx] = Σ A[g,a,h,b] B[m,c,n,d] C[o,e,p,f] GX[a,c] GY[o,g]
 *                   PR[e,m] V1[mx,b,d,f] V2[my,n,p,h]
 *
 * 收缩序列（FLOP 数 = 96N⁴ = opt_einsum optimal/dp 的 opt_cost/2）
 * ------------------------------------------------------------------
 *   U1T[c,g,b,h]    = Σ_a GX[a,c] A[g,a,h,b]                       64N²
 *   U2[g,m,p,f]     = Σ_{o,e} GY[o,g] PR[e,m] C[o,e,p,f]          256N²
 *   Bbuf[m,f,c,n,b] = Σ_d B[m,c,n,d] V1[b,d,f]                     16N⁴
 *   Abuf[c,g,p,n,b] = Σ_h V2[n,p,h] U1T[c,g,b,h]                   16N⁴
 *   Cout[g,p,m,f]   = Σ_{c,n,b} Abuf[c,g,p,n,b] Bbuf[m,f,c,n,b]    64N⁴
 *   out             = Σ_{g,m,p,f} U2[g,m,p,f] Cout[g,p,m,f]        16N²
 *
 * 为什么能比 opt_einsum 快：同样是 96N⁴，但
 *   1. 三个大 GEMM 的 K 维在两侧都连续（TN 形式），没有 pack / transpose；
 *   2. 大 GEMM 用**实 DP FMA** 而不是 complex GEMM：复数乘法用
 *      Karatsuba / 3M 恒等式
 *          R = Σ Ar·Br − Σ Ai·Bi
 *          I = Σ (Ar+Ai)·(Br+Bi) − Σ Ar·Br − Σ Ai·Bi
 *      即 3 个实乘加 / 复数乘加（朴素 4M 需要 4 个）。sm_89 的 DP 比例是
 *      1/64，本机 DP FMA 上限实测 ≈ 1.02e11 FMA/s；96N⁴ 个复数乘加
 *      = 2.88e10 个实 FMA（3M），而 4M 的 3.84e10 会直接输给 opt_einsum。
 *      3M 的额外舍入最多把 2⁻⁵³ 项放大 4×，N=100 实测 rel ≈ 5e-15。
 *   3. 一次 workspace 申请、单流执行、无多卡多流；Abuf/Bbuf 按 n 分块，
 *      峰值显存按 cudaMemGetInfo 自适应（上限 768 MiB）。
 *
 * 布局（全部由本文件决定；host 侧算 batch 基址）
 * ------------------------------------------------------------------
 *   BbufX[m][f][c][n][b]   行 (m,f) 内 (c,n,b) 连续（c 段长 Nc·N）
 *   AbufX[c][g][p][n][b]   行 (g,p) 内 (n,b) 连续，c 段长 16·Nc·N
 *   X ∈ {R, I, S = R+I}：三份实数组，编码见 kLR / kLI / kLS
 *   V1X[f][b][d]           f 外层，便于 B[m,c,n,d] × V1[f,b,d]
 *   V2X[p][n][h]           p 外层，便于 V2[p,n,h] × U1T[c,g,b,h]
 *   U1TX[c][g][b][h]
 *   U2[g][m][p][f]（复数）
 *
 * star 步骤
 * ------------------------------------------------------------------
 *   对每个 c：T1 += AbufR(c)·BbufR(c)ᵀ
 *              T2 += AbufI(c)·BbufI(c)ᵀ
 *              T3 += AbufS(c)·BbufS(c)ᵀ      （4N × 4N，K = Nc·N）
 *   然后 Cout = (T1 − T2) + i·(T3 − T1 − T2)，与 U2 点积得标量。
 *
 * 线程/并发
 * ------------------------------------------------------------------
 *   设备 workspace 静态缓存复用，同一设备上的并发调用必须由调用方串行化。
 *   流由 `stream` 参数给出（0 = legacy default stream）。
 */

/* 注意：-I 顺序是 cpp/kupl/einsum/include 在前，那里有一个同名
 * nucleon2pt.h（KUPL/CPU 版，声明 qcd_kupl_nucleon_2pt）。用引号 + 相对
 * 路径强制选中本目录的 CUDA 头，避免 ABI 互相覆盖。 */
#include "../include/nucleon2pt.h"

#ifndef QCD_NUCLEON2PT_PERM_IDENTITY
#  error "wrong nucleon2pt.h resolved (kupl shadow?); expected cpp/cuda/einsum/include/nucleon2pt.h"
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

// ---------------------------------------------------------------- 基本常量
constexpr int kSpin = 4;                               // 自旋/色指标长度
constexpr long long kMaxN = 2048;
constexpr size_t kWorkspaceLimitBytes = 2ull << 30;    // 2 GiB，fail closed
// Abuf + Bbuf（6 份实数组）的目标上限；实际取 min(此值, 空闲显存/2)
constexpr size_t kBigBufferBudget = 768ull << 20;

// ------------------------------------------------------------- perm 码解码
// code = p0*1000 + p1*100 + p2*10 + p3；p[i] = 规范轴 i 对应的输入轴下标
__host__ __device__ __forceinline__ void perm_digits(int code, int p[4]) {
    p[0] = code / 1000;
    p[1] = (code / 100) % 10;
    p[2] = (code / 10) % 10;
    p[3] = code % 10;
}

__host__ __device__ __forceinline__ bool perm_valid(int code) {
    if (code < 0 || code > 3333) {
        return false;
    }
    int p[4];
    perm_digits(code, p);
    int seen = 0;
    for (int i = 0; i < 4; ++i) {
        if (p[i] < 0 || p[i] >= 4 || (seen & (1 << p[i])) != 0) {
            return false;
        }
        seen |= 1 << p[i];
    }
    return true;
}

template <typename R>
__device__ __forceinline__ R zero_r();
template <>
__device__ __forceinline__ double zero_r<double>() {
    return 0.0;
}
template <>
__device__ __forceinline__ float zero_r<float>() {
    return 0.0f;
}

// ------------------------------------------------------- stage 0: 归一化复数
// dst[i0][i1][i2][i3]（规范序、紧致） = src[Σ i_k * stride[k]]
template <typename R>
__global__ void normalize4_complex_kernel(const R* __restrict__ src, R* __restrict__ dst,
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

// 同上，但拆成 (R, I, S = R+I) 三份实数组
template <typename R>
__global__ void normalize4_split_kernel(const R* __restrict__ src, R* __restrict__ dr,
                                        R* __restrict__ di, R* __restrict__ ds, long long n,
                                        long long total, long long s0, long long s1,
                                        long long s2, long long s3) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long i3 = idx % n;
    const long long i2 = (idx / n) % n;
    const long long i1 = (idx / (n * n)) % kSpin;
    const long long i0 = idx / (n * n * kSpin);
    const long long o = 2 * (i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3);
    const R re = src[o];
    const R im = src[o + 1];
    dr[idx] = re;
    di[idx] = im;
    ds[idx] = re + im;
}

// ------------------------------------------------------------ V1 -> [f][b][d]
template <typename R>
__global__ void norm_v1_split_kernel(const R* __restrict__ src, R* __restrict__ dr,
                                     R* __restrict__ di, R* __restrict__ ds, long long n,
                                     long long total, long long mom_off, long long sb,
                                     long long sd, long long sf) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long d = idx % n;
    const long long b = (idx / n) % n;
    const long long f = idx / (n * n);
    const long long o = 2 * (mom_off + b * sb + d * sd + f * sf);
    const R re = src[o];
    const R im = src[o + 1];
    const long long dst = (f * n + b) * n + d;
    dr[dst] = re;
    di[dst] = im;
    ds[dst] = re + im;
}

// ------------------------------------------------------------ V2 -> [p][n][h]
template <typename R>
__global__ void norm_v2_split_kernel(const R* __restrict__ src, R* __restrict__ dr,
                                     R* __restrict__ di, R* __restrict__ ds, long long n,
                                     long long total, long long mom_off, long long sn,
                                     long long sp, long long sh) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long h = idx % n;
    const long long p = (idx / n) % n;
    const long long nn = idx / (n * n);
    const long long o = 2 * (mom_off + nn * sn + p * sp + h * sh);
    const R re = src[o];
    const R im = src[o + 1];
    const long long dst = (p * n + nn) * n + h;
    dr[dst] = re;
    di[dst] = im;
    ds[dst] = re + im;
}

// ============================================================== U1T（复数→拆分）
// U1T[c][g][b][h] = Σ_a GX[a,c] A[g,a,h,b]，A 为规范紧致序 [g][a][h][b]
template <typename R>
__global__ void fold_gx_kernel(const R* __restrict__ a, const R* __restrict__ gx,
                               R* __restrict__ ur, R* __restrict__ ui, R* __restrict__ us,
                               long long n, long long total, int gx_transposed) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long h = idx % n;
    const long long b = (idx / n) % n;
    const long long g = (idx / (n * n)) % kSpin;
    const long long c = idx / (n * n * kSpin);

    R ar = zero_r<R>(), ai = zero_r<R>();
    for (int k = 0; k < kSpin; ++k) {
        const long long o = 2 * (((g * kSpin + k) * n + h) * n + b);
        const R xr = a[o];
        const R xi = a[o + 1];
        const long long gi = (gx_transposed ? (c * kSpin + k) : (k * kSpin + c)) * 2;
        const R gr = gx[gi];
        const R gim = gx[gi + 1];
        ar += xr * gr - xi * gim;
        ai += xr * gim + xi * gr;
    }
    ur[idx] = ar;
    ui[idx] = ai;
    us[idx] = ar + ai;
}

// =============================================================== U2（保持复数）
// U2[g][m][p][f] = Σ_{o,e} GY[o,g] PR[e,m] C[o,e,p,f]
template <typename R>
__global__ void fold_gy_pr_kernel(const R* __restrict__ c, const R* __restrict__ gy,
                                  const R* __restrict__ pr, R* __restrict__ u2, long long n,
                                  long long total, int gy_transposed, int pr_transposed) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const long long f = idx % n;
    const long long p = (idx / n) % n;
    const long long m = (idx / (n * n)) % kSpin;
    const long long g = idx / (n * n * kSpin);

    R ar = zero_r<R>(), ai = zero_r<R>();
    for (int o = 0; o < kSpin; ++o) {
        const long long gi = (gy_transposed ? (g * kSpin + o) : (o * kSpin + g)) * 2;
        const R gyr = gy[gi];
        const R gyi = gy[gi + 1];
        for (int e = 0; e < kSpin; ++e) {
            const long long pi = (pr_transposed ? (m * kSpin + e) : (e * kSpin + m)) * 2;
            const R prr = pr[pi];
            const R pri = pr[pi + 1];
            const R wr = gyr * prr - gyi * pri;
            const R wi = gyr * pri + gyi * prr;
            const long long oo = 2 * (((o * kSpin + e) * n + p) * n + f);
            const R cr = c[oo];
            const R ci = c[oo + 1];
            ar += wr * cr - wi * ci;
            ai += wr * ci + wi * cr;
        }
    }
    u2[2 * idx] = ar;
    u2[2 * idx + 1] = ai;
}

// ===================================================== 3M 原地合成（就地）
// 输入 t1 = ΣAr·Br, t2 = ΣAi·Bi, t3 = ΣAs·Bs
//   R = t1 − t2,  I = t3 − t1 − t2,  S = R + I = t3 − 2·t2
// dr/di/ds 可以与 t1/t2/t3 相同（逐元素、同下标先读后写）。
template <typename R>
__global__ void combine3_kernel(const R* __restrict__ t1, const R* __restrict__ t2,
                                const R* __restrict__ t3, R* __restrict__ dr,
                                R* __restrict__ di, R* __restrict__ ds, long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const R a1 = t1[idx];
    const R a2 = t2[idx];
    const R a3 = t3[idx];
    dr[idx] = a1 - a2;
    di[idx] = a3 - a1 - a2;
    ds[idx] = a3 - a2 - a2;
}

// ============================================================== 通用 TN GEMM
// C[cbase+i·ldc+j] = accumulate ? C + Σ_k A[abase+i·lda+k]·B[bbase+j·ldb+k] : Σ_k ...
// 两侧 K 连续；batch 基址由 Batch3 三轴仿射偏移给出，另加一个标量基址。
struct Batch3 {
    long long size[3];
    long long stride[3];
};

__device__ __forceinline__ long long batch_offset(const Batch3& m, int id) {
    long long base = 0;
    int rem = id;
    for (int d = 2; d >= 0; --d) {
        const int extent = static_cast<int>(m.size[d]);
        if (extent <= 0) {
            continue;
        }
        const int index = rem % extent;
        rem /= extent;
        base += static_cast<long long>(index) * m.stride[d];
    }
    return base;
}

template <typename R, int TM, int TN, int BX, int BY, int BK>
__global__ __launch_bounds__(BX* BY) void gemm_tn_kernel(
    int M, int Ncol, int K, const R* __restrict__ A, long long lda, Batch3 amap,
    long long abase0, const R* __restrict__ B, long long ldb, Batch3 bmap, long long bbase0,
    R* __restrict__ C, long long ldc, Batch3 cmap, long long cbase0, int accumulate) {
    constexpr int BM = BY * TM;
    constexpr int BN = BX * TN;
    constexpr int kThreadsHere = BX * BY;
    __shared__ R sa[BM][BK + 1];
    __shared__ R sb[BN][BK + 1];

    const int tid = threadIdx.x;
    const int tx = tid % BX;
    const int ty = tid / BX;
    const int i_tile = static_cast<int>(blockIdx.y) * BM;
    const int j_tile = static_cast<int>(blockIdx.x) * BN;

    const long long abase = abase0 + batch_offset(amap, static_cast<int>(blockIdx.z));
    const long long bbase = bbase0 + batch_offset(bmap, static_cast<int>(blockIdx.z));
    const long long cbase = cbase0 + batch_offset(cmap, static_cast<int>(blockIdx.z));

    R acc[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int s = 0; s < TN; ++s) {
            acc[r][s] = zero_r<R>();
        }
    }

    const int avalid = M - i_tile;
    const int bvalid = Ncol - j_tile;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += kThreadsHere) {
            const int r = idx / BK;
            const int t = idx - r * BK;
            const int gk = k0 + t;
            R v = zero_r<R>();
            if (r < avalid && gk < K) {
                v = A[abase + static_cast<long long>(i_tile + r) * lda + gk];
            }
            sa[r][t] = v;
        }
        for (int idx = tid; idx < BN * BK; idx += kThreadsHere) {
            const int r = idx / BK;
            const int t = idx - r * BK;
            const int gk = k0 + t;
            R v = zero_r<R>();
            if (r < bvalid && gk < K) {
                v = B[bbase + static_cast<long long>(j_tile + r) * ldb + gk];
            }
            sb[r][t] = v;
        }
        __syncthreads();

        const int kmax = (K - k0 < BK) ? (K - k0) : BK;
#pragma unroll 1
        for (int t = 0; t < kmax; ++t) {
            R av[TM];
            R bv[TN];
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                av[r] = sa[ty * TM + r][t];
            }
#pragma unroll
            for (int s = 0; s < TN; ++s) {
                bv[s] = sb[tx * TN + s][t];
            }
#pragma unroll
            for (int r = 0; r < TM; ++r) {
#pragma unroll
                for (int s = 0; s < TN; ++s) {
                    acc[r][s] += av[r] * bv[s];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int gi = i_tile + ty * TM + r;
        if (gi >= M) {
            continue;
        }
#pragma unroll
        for (int s = 0; s < TN; ++s) {
            const int gj = j_tile + tx * TN + s;
            if (gj >= Ncol) {
                continue;
            }
            R* q = C + cbase + static_cast<long long>(gi) * ldc + gj;
            if (accumulate) {
                *q += acc[r][s];
            } else {
                *q = acc[r][s];
            }
        }
    }
}

// ---------------------------------------------------------- tile 选择与派发
struct TileCfg {
    int tm;
    int tn;
    int bx;
    int by;
    int bk;
};

constexpr TileCfg kTileTable[] = {
    {5, 5, 20, 20, 16},   // 100x100, 400 threads
    {5, 5, 16, 16, 32},   //  80x80,  256 threads
    {5, 5, 8, 8, 32},     //  40x40,   64 threads
    {4, 4, 16, 16, 32},   //  64x64,  256 threads
    {2, 2, 16, 16, 32},   //  32x32,  256 threads
    {1, 1, 16, 16, 32},   //  16x16,  256 threads
};
constexpr int kTileCount = static_cast<int>(sizeof(kTileTable) / sizeof(kTileTable[0]));

int choose_tile(long long M, long long Ncol, long long K, long long nbatch, int sm_count) {
    int best = kTileCount - 1;
    double best_score = 1e300;
    for (int t = 0; t < kTileCount; ++t) {
        const TileCfg& cfg = kTileTable[t];
        const long long bm = static_cast<long long>(cfg.by) * cfg.tm;
        const long long bn = static_cast<long long>(cfg.bx) * cfg.tn;
        const long long tm = (M + bm - 1) / bm;
        const long long tn = (Ncol + bn - 1) / bn;
        const long long r = tm * bm;
        const long long c = tn * bn;
        const long long kk = ((K + cfg.bk - 1) / cfg.bk) * cfg.bk;
        double score = static_cast<double>(r) * static_cast<double>(c) *
                       static_cast<double>(kk);
        // 块数不足以铺满 SM 时按占用率惩罚（每 SM 至少 1 块且尽量 ≥ SM 数）
        const long long blocks = tm * tn * nbatch;
        if (blocks < sm_count) {
            score *= static_cast<double>(sm_count) / static_cast<double>(blocks > 0 ? blocks : 1);
        }
        if (score < best_score) {
            best_score = score;
            best = t;
        }
    }
    return best;
}

template <typename R>
void launch_gemm(int tile, int M, int Ncol, int K, const R* A, long long lda,
                 const Batch3& amap, long long abase0, const R* B, long long ldb,
                 const Batch3& bmap, long long bbase0, R* C, long long ldc, const Batch3& cmap,
                 int nbatch, int accumulate, cudaStream_t stream) {
    const TileCfg& cfg = kTileTable[tile];
    const int bm = cfg.by * cfg.tm;
    const int bn = cfg.bx * cfg.tn;
    dim3 grid(static_cast<unsigned>((Ncol + bn - 1) / bn),
              static_cast<unsigned>((M + bm - 1) / bm), static_cast<unsigned>(nbatch));
    const int threads = cfg.bx * cfg.by;
    Batch3 zero_map;
    zero_map.size[0] = zero_map.size[1] = zero_map.size[2] = 1;
    zero_map.stride[0] = zero_map.stride[1] = zero_map.stride[2] = 0;
    const Batch3& cm = (nbatch > 1) ? cmap : zero_map;
#define PYQCD_LAUNCH_GEMM(TMv, TNv, BXv, BYv, BKv)                                            \
    gemm_tn_kernel<R, TMv, TNv, BXv, BYv, BKv><<<grid, threads, 0, stream>>>(                  \
        M, Ncol, K, A, lda, amap, abase0, B, ldb, bmap, bbase0, C, ldc, cm, 0, accumulate)
    switch (tile) {
        case 0: PYQCD_LAUNCH_GEMM(5, 5, 20, 20, 16); break;
        case 1: PYQCD_LAUNCH_GEMM(5, 5, 16, 16, 32); break;
        case 2: PYQCD_LAUNCH_GEMM(5, 5, 8, 8, 32); break;
        case 3: PYQCD_LAUNCH_GEMM(4, 4, 16, 16, 32); break;
        case 4: PYQCD_LAUNCH_GEMM(2, 2, 16, 16, 32); break;
        default: PYQCD_LAUNCH_GEMM(1, 1, 16, 16, 32); break;
    }
#undef PYQCD_LAUNCH_GEMM
}

// ============================================================ 收尾点积
// out += Σ_{i,j} U2'(i,j)·Cout(i,j)，Cout = (t1 − t2) + i(t3 − t1 − t2)
// i = g·N + p, j = m·N + f
template <typename R>
__global__ void final_dot_kernel(const R* __restrict__ u2, const R* __restrict__ t1,
                                 const R* __restrict__ t3, const R* __restrict__ t2,
                                 R* __restrict__ out, long long total, long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    R ar = zero_r<R>(), ai = zero_r<R>();
    const long long row = 4 * n;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const long long jj = idx % row;
        const long long ii = idx / row;
        const long long g = ii / n;
        const long long p = ii - g * n;
        const long long m = jj / n;
        const long long f = jj - m * n;
        const R cr = t1[idx] - t2[idx];
        const R ci = t3[idx] - t1[idx] - t2[idx];
        const long long uo = 2 * (((g * 4 + m) * n + p) * n + f);
        const R ur = u2[uo];
        const R ui = u2[uo + 1];
        ar += ur * cr - ui * ci;
        ai += ur * ci + ui * cr;
    }
    __shared__ R shr[256];
    __shared__ R shi[256];
    const int lane = static_cast<int>(threadIdx.x);
    shr[lane] = ar;
    shi[lane] = ai;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (lane < off) {
            shr[lane] += shr[lane + off];
            shi[lane] += shi[lane + off];
        }
        __syncthreads();
    }
    if (lane == 0) {
        atomicAdd(out, shr[0]);
        atomicAdd(out + 1, shi[0]);
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

template <typename R>
struct Buffers {
    R* a_can = nullptr;
    R* c_can = nullptr;
    R* u2 = nullptr;
    R* bsp = nullptr;
    R* u1t = nullptr;
    R* v1s = nullptr;
    R* v2s = nullptr;
    R* abuf = nullptr;
    R* bbuf = nullptr;
    R* tmp = nullptr;
};

template <typename R>
size_t buffer_elems(long long n, long long nc) {
    const long long n2 = n * n;
    const long long part4 = 16 * n2;
    const long long n3 = n2 * n;
    const long long partbig = 16 * n * nc * n;
    long long total = 0;
    total += 2 * part4;    // a_can（复数 = 2R）
    total += 2 * part4;    // c_can
    total += 2 * part4;    // u2
    total += 3 * part4;    // bsp
    total += 3 * part4;    // u1t
    total += 3 * n3;       // v1s
    total += 3 * n3;       // v2s
    total += 3 * partbig;  // abuf
    total += 3 * partbig;  // bbuf
    total += 3 * part4;    // tmp
    return static_cast<size_t>(total);
}

template <typename R>
void layout_buffers(R* base, long long n, long long nc, Buffers<R>* out) {
    const long long n2 = n * n;
    const long long part4 = 16 * n2;
    const long long n3 = n2 * n;
    const long long partbig = 16 * n * nc * n;
    R* p = base;
    out->a_can = p; p += 2 * part4;
    out->c_can = p; p += 2 * part4;
    out->u2 = p; p += 2 * part4;
    out->bsp = p; p += 3 * part4;
    out->u1t = p; p += 3 * part4;
    out->v1s = p; p += 3 * n3;
    out->v2s = p; p += 3 * n3;
    out->abuf = p; p += 3 * partbig;
    out->bbuf = p; p += 3 * partbig;
    out->tmp = p; p += 3 * part4;
}

template <typename R>
void launch_normalize_complex(const R* src, R* dst, long long n, const long long s[4],
                              cudaStream_t stream) {
    const long long total = kSpin * kSpin * n * n;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    normalize4_complex_kernel<R><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        src, dst, n, total, s[0], s[1], s[2], s[3]);
}

template <typename R>
void launch_normalize_split(const R* src, R* dr, R* di, R* ds, long long n,
                            const long long s[4], cudaStream_t stream) {
    const long long total = kSpin * kSpin * n * n;
    const int threads = 256;
    const long long blocks = (total + threads - 1) / threads;
    normalize4_split_kernel<R><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
        src, dr, di, ds, n, total, s[0], s[1], s[2], s[3]);
}

// ------------------------------------------------------------------ 主流程
template <typename R>
int launch_all(const void* const* in, void* out, const Params& p, cudaStream_t stream) {
    const long long n = p.n;
    const long long n2 = n * n;
    const long long n3 = n2 * n;

    int sm_count = 24;
    {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) {
            return QCD_STATUS_INTERNAL;
        }
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess) {
            sm_count = v;
        }
    }

    // ---- n 方向分块，控制 Abuf/Bbuf 峰值显存
    long long nc = n;
    {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            free_bytes = kBigBufferBudget;
        }
        size_t budget = kBigBufferBudget;
        if (free_bytes / 2 < budget) {
            budget = free_bytes / 2;
        }
        const long long per_n = 6 * 16 * n * n * static_cast<long long>(sizeof(R));
        long long cap = static_cast<long long>(budget) / per_n;
        if (cap < 1) {
            cap = 1;
        }
        if (cap < nc) {
            nc = cap;
        }
    }
    if (nc < 1) {
        nc = 1;
    }
    long long nchunks = (n + nc - 1) / nc;
    nc = (n + nchunks - 1) / nchunks;
    nchunks = (n + nc - 1) / nc;

    const size_t elems = buffer_elems<R>(n, nc);
    if (elems * sizeof(R) > kWorkspaceLimitBytes) {
        return QCD_STATUS_MEMORY;
    }
    void* base_raw = nullptr;
    if (!get_workspace(elems * sizeof(R), &base_raw)) {
        return QCD_STATUS_MEMORY;
    }
    Buffers<R> buf;
    layout_buffers<R>(static_cast<R*>(base_raw), n, nc, &buf);

    const R* a_in = static_cast<const R*>(in[0]);
    const R* b_in = static_cast<const R*>(in[1]);
    const R* c_in = static_cast<const R*>(in[2]);
    const R* gx = static_cast<const R*>(in[3]);
    const R* gy = static_cast<const R*>(in[4]);
    const R* pr = static_cast<const R*>(in[5]);
    const R* v1 = static_cast<const R*>(in[6]);
    const R* v2 = static_cast<const R*>(in[7]);

    const long long peram_src[4] = {kSpin * n2, n2, n, 1};
    long long sa[4], sb[4], sc[4];
    permuted_strides(peram_src, p.perm[0], sa);
    permuted_strides(peram_src, p.perm[1], sb);
    permuted_strides(peram_src, p.perm[2], sc);
    const long long vvv_src[4] = {n3, n2, n, 1};
    long long sv1[4], sv2[4];
    permuted_strides(vvv_src, p.perm[3], sv1);
    permuted_strides(vvv_src, p.perm[4], sv2);

    // ---- stage 0：A / C 归一化成规范紧致序（恒等轴序时直接复用输入）
    const R* a = (p.perm[0] == QCD_NUCLEON2PT_PERM_IDENTITY) ? a_in : buf.a_can;
    const R* c = (p.perm[2] == QCD_NUCLEON2PT_PERM_IDENTITY) ? c_in : buf.c_can;
    if (a != a_in) {
        launch_normalize_complex<R>(a_in, buf.a_can, n, sa, stream);
    }
    if (c != c_in) {
        launch_normalize_complex<R>(c_in, buf.c_can, n, sc, stream);
    }
    // ---- B 永远拆成 (R,I,S)（规范序 [m][c][n][d]）
    launch_normalize_split<R>(b_in, buf.bsp, buf.bsp + 32 * n2, buf.bsp + 16 * n2, n, sb,
                              stream);

    int status = map_cuda(cudaGetLastError());
    if (status != QCD_STATUS_OK) {
        return status;
    }

    // ---- U1T / U2（与动量无关，算一次）
    {
        const long long total = kSpin * kSpin * n2;
        const int threads = 256;
        const long long blocks = (total + threads - 1) / threads;
        // u1t 布局：[0]=R, [16N²]=S, [32N²]=I
        fold_gx_kernel<R><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            a, gx, buf.u1t, buf.u1t + 32 * n2, buf.u1t + 16 * n2, n, total,
            (p.matflags & 1) ? 1 : 0);
        fold_gy_pr_kernel<R><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
            c, gy, pr, buf.u2, n, total, (p.matflags & 2) ? 1 : 0, (p.matflags & 4) ? 1 : 0);
    }
    status = map_cuda(cudaGetLastError());
    if (status != QCD_STATUS_OK) {
        return status;
    }

    // bsp / u1t / v1s / v2s 的 (R,S,I) 分组布局：[0]=R, [16N² or N³]=S, [32N² or 2N³]=I
    const R* B_R = buf.bsp;
    const R* B_S = buf.bsp + 16 * n2;
    const R* B_I = buf.bsp + 32 * n2;
    const R* U1_R = buf.u1t;
    const R* U1_S = buf.u1t + 16 * n2;
    const R* U1_I = buf.u1t + 32 * n2;

    const int star_m = static_cast<int>(4 * n);
    const int nbatchA = static_cast<int>(kSpin * n * kSpin);
    const int nbatchB = static_cast<int>(kSpin * kSpin * n);

    const long long pairs = (p.mode == 0) ? p.mx : (p.mx * p.my);
    for (long long pair = 0; pair < pairs; ++pair) {
        long long mx_index = 0;
        long long my_index = 0;
        if (p.mode == 0) {
            mx_index = pair;
            my_index = pair;
        } else {
            my_index = pair / p.mx;
            mx_index = pair % p.mx;
        }

        // ---- V1[mx] -> [f][b][d]（拆分）；V2[my] -> [p][n][h]（拆分）
        {
            const long long total = n3;
            const int threads = 256;
            const long long blocks = (total + threads - 1) / threads;
            const unsigned g = static_cast<unsigned>(blocks);
            // 布局：v1s [0]=R, [N³]=S, [2N³]=I
            norm_v1_split_kernel<R><<<g, threads, 0, stream>>>(
                v1, buf.v1s, buf.v1s + 2 * n3, buf.v1s + n3, n, total,
                mx_index * vvv_src[0], sv1[1], sv1[2], sv1[3]);
            norm_v2_split_kernel<R><<<g, threads, 0, stream>>>(
                v2, buf.v2s, buf.v2s + 2 * n3, buf.v2s + n3, n, total,
                my_index * vvv_src[0], sv2[1], sv2[2], sv2[3]);
        }
        status = map_cuda(cudaGetLastError());
        if (status != QCD_STATUS_OK) {
            return status;
        }

        const R* V1_R = buf.v1s;
        const R* V1_S = buf.v1s + n3;
        const R* V1_I = buf.v1s + 2 * n3;
        const R* V2_R = buf.v2s;
        const R* V2_S = buf.v2s + n3;
        const R* V2_I = buf.v2s + 2 * n3;

        R* out_slot = static_cast<R*>(out) +
                      ((p.mode == 0) ? mx_index : (my_index * p.mx + mx_index)) * 2;
        status = map_cuda(cudaMemsetAsync(out_slot, 0, 2 * sizeof(R), stream));
        if (status != QCD_STATUS_OK) {
            return status;
        }

        bool first_star = true;
        for (long long chunk = 0; chunk < nchunks; ++chunk) {
            const long long n0 = chunk * nc;
            const long long ncur = (n - n0 < nc) ? (n - n0) : nc;
            const long long ncn = ncur * n;
            const int nc_i = static_cast<int>(ncur);
            const int n_i = static_cast<int>(n);
            const int ncn_i = static_cast<int>(ncn);
            // Abuf 总大小 = 4(c)·4(g)·n(p)·ncur·n(b)；Abuf 的 c 步长是其 1/4
            const long long pbig = 16 * n * ncn;
            const long long cbig = 4 * n * ncn;
            R* Ab_R = buf.abuf;
            R* Ab_S = buf.abuf + pbig;
            R* Ab_I = buf.abuf + 2 * pbig;
            R* Bb_R = buf.bbuf;
            R* Bb_S = buf.bbuf + pbig;
            R* Bb_I = buf.bbuf + 2 * pbig;

            // ---------------- stage A: Bbuf[m][f][c][n][b] = Σ_d B[m,c,n,d]·V1[f,b,d]
            {
                Batch3 amap, bmap, cmap;
                amap.size[0] = kSpin; amap.size[1] = n_i; amap.size[2] = kSpin;
                amap.stride[0] = kSpin * n2;   // m
                amap.stride[1] = 0;            // f
                amap.stride[2] = n2;           // c
                bmap.size[0] = kSpin; bmap.size[1] = n_i; bmap.size[2] = kSpin;
                bmap.stride[0] = 0;
                bmap.stride[1] = n2;           // f
                bmap.stride[2] = 0;
                cmap.size[0] = kSpin; cmap.size[1] = n_i; cmap.size[2] = kSpin;
                cmap.stride[0] = 4 * n * ncn;  // m
                cmap.stride[1] = 4 * ncn;      // f
                cmap.stride[2] = ncn;          // c
                const long long aoff = n0 * n; // B: n 偏移
                const long long boff = n0 * n; // V1: b 偏移
                const R* aops[3] = {B_R, B_I, B_S};
                const R* bops[3] = {V1_R, V1_I, V1_S};
                R* cops[3] = {Bb_R, Bb_I, Bb_S};
                const int tile = choose_tile(ncur, n, n, nbatchA, sm_count);
                for (int k = 0; k < 3; ++k) {
                    launch_gemm<R>(tile, nc_i, n_i, n_i, aops[k], n, amap, aoff, bops[k], n,
                                   bmap, boff, cops[k], n, cmap, nbatchA, 0, stream);
                }
            }
            // ---------------- stage B: Abuf[c][g][p][n][b] = Σ_h V2[p,n,h]·U1T[c,g,b,h]
            {
                Batch3 amap, bmap, cmap;
                amap.size[0] = kSpin; amap.size[1] = kSpin; amap.size[2] = n_i;
                amap.stride[0] = 0;
                amap.stride[1] = 0;
                amap.stride[2] = n2;           // p
                bmap.size[0] = kSpin; bmap.size[1] = kSpin; bmap.size[2] = n_i;
                bmap.stride[0] = kSpin * n2;   // c
                bmap.stride[1] = n2;           // g
                bmap.stride[2] = 0;
                cmap.size[0] = kSpin; cmap.size[1] = kSpin; cmap.size[2] = n_i;
                cmap.stride[0] = 4 * n * ncn;  // c
                cmap.stride[1] = n * ncn;      // g
                cmap.stride[2] = ncn;          // p
                const long long aoff = n0 * n; // V2: [p][n][h] 的 n 偏移
                const R* aops[3] = {V2_R, V2_I, V2_S};
                const R* bops[3] = {U1_R, U1_I, U1_S};
                R* cops[3] = {Ab_R, Ab_I, Ab_S};
                const int tile = choose_tile(ncur, n, n, nbatchB, sm_count);
                for (int k = 0; k < 3; ++k) {
                    launch_gemm<R>(tile, nc_i, n_i, n_i, aops[k], n, amap, aoff, bops[k], n,
                                   bmap, 0, cops[k], n, cmap, nbatchB, 0, stream);
                }
            }
            // ---------------- 3M 合成（就地，逐元素同下标先读后写）
            {
                const int threads = 256;
                const long long total = 16 * n * ncn;
                const long long blocks = (total + threads - 1) / threads;
                const unsigned g = static_cast<unsigned>(blocks);
                combine3_kernel<R><<<g, threads, 0, stream>>>(Bb_R, Bb_I, Bb_S, Bb_R, Bb_I,
                                                             Bb_S, total);
                combine3_kernel<R><<<g, threads, 0, stream>>>(Ab_R, Ab_I, Ab_S, Ab_R, Ab_I,
                                                             Ab_S, total);
            }
            // ---------------- stage C: star（按 c 分 4 次，累加到 tmp）
            {
                Batch3 one;
                one.size[0] = 1; one.size[1] = 1; one.size[2] = 1;
                one.stride[0] = one.stride[1] = one.stride[2] = 0;
                R* T1b = buf.tmp;             // Σ R·R
                R* T3b = buf.tmp + 16 * n2;   // Σ S·S
                R* T2b = buf.tmp + 32 * n2;   // Σ I·I
                const R* aops[3] = {Ab_R, Ab_I, Ab_S};
                const R* bops[3] = {Bb_R, Bb_I, Bb_S};
                R* tops[3] = {T1b, T2b, T3b};
                const int tile = choose_tile(star_m, star_m, ncn, 1, sm_count);
                for (int cc = 0; cc < kSpin; ++cc) {
                    const long long aoff = static_cast<long long>(cc) * cbig;
                    const long long boff = static_cast<long long>(cc) * ncn;
                    const int acc = (first_star && cc == 0) ? 0 : 1;
                    for (int k = 0; k < 3; ++k) {
                        launch_gemm<R>(tile, star_m, star_m, ncn_i, aops[k], ncn, one, aoff,
                                       bops[k], 4 * ncn, one, boff, tops[k], star_m, one, 1,
                                       acc, stream);
                    }
                }
                first_star = false;
            }
            status = map_cuda(cudaGetLastError());
            if (status != QCD_STATUS_OK) {
                return status;
            }
        }

        // ---- 收尾点积：Cout = (T1b − T2b) + i(T3b − T1b − T2b)
        {
            const int threads = 256;
            const long long total = 16 * n2;
            final_dot_kernel<R><<<48, threads, 0, stream>>>(
                buf.u2, buf.tmp, buf.tmp + 16 * n2, buf.tmp + 32 * n2, out_slot, total, n);
        }
        status = map_cuda(cudaGetLastError());
        if (status != QCD_STATUS_OK) {
            return status;
        }
    }
    return map_cuda(cudaGetLastError());
}

}  // namespace

/* --- debug-only workspace dump (removed from the delivered file) --- */
extern "C" QCD_NUCLEON2PT_API int qcd_cuda_n2pt_dbg(double* dst, long long elem_off,
                                                   long long count) {
    if (g_workspace.ptr == nullptr) {
        return 1;
    }
    cudaError_t e = cudaMemcpy(dst, static_cast<char*>(g_workspace.ptr) + elem_off * 8,
                               count * 8, cudaMemcpyDeviceToHost);
    return (e == cudaSuccess) ? 0 : 2;
}

extern "C" QCD_NUCLEON2PT_API long long qcd_cuda_n2pt_ws_elems() {
    return static_cast<long long>(g_workspace.bytes / 8);
}

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

    if (p.n < 1 || p.n > kMaxN) {
        return QCD_STATUS_INVALID;
    }
    if (p.mode != 0 && p.mode != 1) {
        return QCD_STATUS_INVALID;
    }
    if (p.mx < 1 || p.my < 1) {
        return QCD_STATUS_INVALID;
    }
    if (p.mode == 0 && p.mx != p.my) {
        return QCD_STATUS_INVALID;
    }
    if (p.matflags < 0 || p.matflags > 7) {
        return QCD_STATUS_INVALID;
    }
    for (int i = 0; i < 5; ++i) {
        if (!perm_valid(p.perm[i])) {
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
        return launch_all<double>(inputs, output, p, s);
    }
    return launch_all<float>(inputs, output, p, s);
}
