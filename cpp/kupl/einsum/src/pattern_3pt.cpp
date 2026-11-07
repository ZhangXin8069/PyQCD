/* 格点 QCD 3pt 主项（P1/P2）专用 CPU 执行 kernel。
 *
 * 目的：把 15 算子 / 14 步的通用执行（每步 pack + GEMM + 输出 permute + 新建
 * 中间缓冲）替换成**零 pack、零输出 permute、缓冲全复用**的一次调用。
 *
 * 关键设计（与通用执行层的差异）：
 *   1. 所有 GEMM 的操作数都是**现成布局**：中间结果的 layout 在写出时就按下一
 *      步的 (m,k) / (k,n) 要求排好，只有少数小张量（≤ n^2 或 n^3 一次）需要
 *      显式置换，置换是连续 run 的拷贝，不是通用 odometer pack。
 *   2. 4x4 Dirac / 4x2 投影的哑指标全部用**按最内维遍历 + 寄存器复用**的直接
 *      循环（peram 预处理、gamma 夹心、尾部小收缩），不走 pack/GEMM。
 *   3. 大 GEMM 只有 3 个（P1: 5e7 / 2e8 / 5e7；P2: 1e8 / 1e8 + 6e6），
 *      全部是 (m,n,k) 形状友好、B 侧 k 维连续的 row-major GEMM。
 *   4. 中间缓冲通过小型 pool 复用，避免每次调用首触页缺页（通用执行层在
 *      step6 上正是被 16 MB 新映射的首触吃掉约 20 ms）。
 *
 * 线程口径：GEMM 用进程内 CBLAS（OpenBLAS），线程数由 OPENBLAS_NUM_THREADS
 * 决定；本文件自己的 OpenMP 循环由 OMP_NUM_THREADS 决定；二者不嵌套。
 */

#include "pattern_3pt.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int64_t kDirac = 4; /* 固定 4x4 Dirac 结构 */
constexpr int64_t kProj = 2;  /* 投影 (4,2) 的列数 */
constexpr int64_t kSmallTile = 64;

constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;
constexpr int kCblasTrans = 112;

struct Cplx {
    double re;
    double im;
};

static_assert(sizeof(Cplx) == 16, "complex128 must be 16 bytes");

inline void madd(Cplx* acc, const Cplx& a, const Cplx& b) {
    acc->re += a.re * b.re - a.im * b.im;
    acc->im += a.re * b.im + a.im * b.re;
}

/* ------------------------------------------------------------------------- */
/* 可复用缓冲池：避免每次调用重新 mmap/首触 16~32 MB 中间缓冲                  */
/* ------------------------------------------------------------------------- */

class BufferPool {
  public:
    static BufferPool& instance() {
        static BufferPool pool;
        return pool;
    }

    void* acquire(std::size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            for (int index = 0; index < kSlots; ++index) {
                if (slots_[index].data != nullptr && slots_[index].bytes >= bytes &&
                    slots_[index].bytes <= bytes * 2) {
                    void* data = slots_[index].data;
                    slots_[index].data = nullptr;
                    return data;
                }
            }
        }
        void* data = nullptr;
        if (posix_memalign(&data, 64, bytes) != 0) {
            return nullptr;
        }
        return data;
    }

    void release(void* data, std::size_t bytes) {
        if (data == nullptr) {
            return;
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            for (int index = 0; index < kSlots; ++index) {
                if (slots_[index].data == nullptr) {
                    slots_[index].data = data;
                    slots_[index].bytes = bytes;
                    return;
                }
            }
        }
        std::free(data);
    }

  private:
    static constexpr int kSlots = 24;

    struct Slot {
        void* data = nullptr;
        std::size_t bytes = 0;
    };

    ~BufferPool() {
        for (int index = 0; index < kSlots; ++index) {
            std::free(slots_[index].data);
        }
    }

    std::mutex mutex_;
    Slot slots_[kSlots];
};

/* RAII 持有：acquire 失败时 fail-closed。 */
class Lease {
  public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    ~Lease() { reset(); }

    bool take(std::size_t bytes) {
        reset();
        if (bytes == 0) {
            return true;
        }
        data_ = BufferPool::instance().acquire(bytes);
        bytes_ = bytes;
        return data_ != nullptr;
    }

    void reset() {
        if (data_ != nullptr) {
            BufferPool::instance().release(data_, bytes_);
        }
        data_ = nullptr;
        bytes_ = 0;
    }

    template <typename Type>
    Type* as() const {
        return reinterpret_cast<Type*>(data_);
    }

    bool ok() const { return data_ != nullptr || bytes_ == 0; }

  private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

/* ------------------------------------------------------------------------- */
/* CBLAS zgemm 绑定（与 apply_contract.cpp 相同的探测顺序，独立实现）          */
/* ------------------------------------------------------------------------- */

using ZgemmLp64 = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                           const void*, int, const void*, void*, int);
using ZgemmIlp64 = void (*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, const void*,
                            const void*, int64_t, const void*, int64_t, const void*, void*, int64_t);

struct Blas {
    ZgemmLp64 lp64 = nullptr;
    ZgemmIlp64 ilp64 = nullptr;

    bool ready() const { return lp64 != nullptr || ilp64 != nullptr; }
};

template <typename Function>
bool bind_symbol(void* handle, const char* name, Function* function) {
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return false;
    }
    std::memcpy(function, &symbol, sizeof(Function));
    return true;
}

Blas load_blas() {
    Blas api;
    const char* candidates[] = {"libcblas.so.3", "libcblas.so", "libopenblas.so.0",
                                "libopenblas.so", "libscipy_openblas64_.so"};
    void* handles[8];
    int handle_count = 0;
    handles[handle_count++] = RTLD_DEFAULT;
    if (const char* override_path = std::getenv("PYQCD_EINSUM_CBLAS_LIB")) {
        if (override_path[0] != '\0' && handle_count < 8) {
            if (void* handle = dlopen(override_path, RTLD_LAZY | RTLD_LOCAL)) {
                handles[handle_count++] = handle;
            }
        }
    }
    for (const char* name : candidates) {
        if (handle_count >= 8) {
            break;
        }
        if (void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL)) {
            handles[handle_count++] = handle;
        }
    }
    for (int index = 0; index < handle_count; ++index) {
        if (bind_symbol(handles[index], "cblas_zgemm64_", &api.ilp64) ||
            bind_symbol(handles[index], "cblas_zgemm", &api.lp64)) {
            if (api.ilp64 != nullptr) {
                api.lp64 = nullptr;
                return api;
            }
            return api;
        }
    }
    return api;
}

const Blas& blas() {
    static const Blas api = load_blas();
    return api;
}

/* C = alpha*A*B + beta*C（alpha=1；beta 由 accumulate 决定）。 */
bool gemm(int trans_a, int trans_b, int64_t m, int64_t n, int64_t k, const Cplx* a, int64_t lda,
          const Cplx* b, int64_t ldb, bool accumulate, Cplx* c, int64_t ldc) {
    if (m <= 0 || n <= 0) {
        return true;
    }
    if (k <= 0) {
        if (!accumulate) {
            std::memset(c, 0, static_cast<std::size_t>(m * n) * sizeof(Cplx));
        }
        return true;
    }
    const Blas& api = blas();
    if (!api.ready()) {
        return false;
    }
    const Cplx alpha{1.0, 0.0};
    const Cplx beta = accumulate ? Cplx{1.0, 0.0} : Cplx{0.0, 0.0};
    if (api.ilp64 != nullptr) {
        api.ilp64(kCblasRowMajor, trans_a, trans_b, m, n, k, &alpha, a, lda, b, ldb, &beta, c,
                  ldc);
        return true;
    }
    if (m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max() || lda > std::numeric_limits<int>::max() ||
        ldb > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) {
        return false;
    }
    api.lp64(kCblasRowMajor, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
             static_cast<int>(k), &alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
             &beta, c, static_cast<int>(ldc));
    return true;
}

/* ------------------------------------------------------------------------- */
/* 小型置换：全部按“连续 run 拷贝”实现，并用 OpenMP 摊到 OMP_NUM_THREADS 上     */
/* ------------------------------------------------------------------------- */

inline bool parallel_worth(int64_t work, int64_t minimum) { return work >= minimum; }

/* dst[(i_j, j), (k, l)] <- src[(i, k, j, l)]（P1: gamma7b 之后的 ikjl / P2: imjn） */
void permute_source_pair(const Cplx* src, int64_t n, Cplx* dst) {
    /* src[((i*4 + x)*n + j)*n + l]，dst[((i*n + j)*4 + x)*n + l] */
    const int64_t total = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(total, 64)) schedule(static)
#endif
    for (int64_t row = 0; row < total; ++row) {
        const int64_t i = row / n;
        const int64_t j = row % n;
        for (int64_t x = 0; x < kDirac; ++x) {
            std::memcpy(dst + (row * kDirac + x) * n,
                        src + ((i * kDirac + x) * n + j) * n,
                        static_cast<std::size_t>(n) * sizeof(Cplx));
        }
    }
}

/* dst[(e*n + f), (q*n + r)] <- src[((e*4 + q)*n + f)*n + r]（eqfr） */
void permute_middle_pair(const Cplx* src, int64_t n, Cplx* dst) {
    const int64_t total = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(total, 64)) schedule(static)
#endif
    for (int64_t row = 0; row < total; ++row) {
        const int64_t e = row / n;
        const int64_t f = row % n;
        for (int64_t q = 0; q < kDirac; ++q) {
            std::memcpy(dst + (row * kDirac + q) * n,
                        src + ((e * kDirac + q) * n + f) * n,
                        static_cast<std::size_t>(n) * sizeof(Cplx));
        }
    }
}

/* 2D 转置：dst[dst_row_stride*i + j] = src[src_row_stride*j + i]，8x8 分块。 */
void transpose_2d(const Cplx* src, Cplx* dst, int64_t rows, int64_t cols,
                  int64_t src_row_stride, int64_t dst_row_stride) {
    constexpr int64_t kTile = 8;
    const int64_t tile_rows = (rows + kTile - 1) / kTile;
    const int64_t tile_cols = (cols + kTile - 1) / kTile;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (parallel_worth(tile_rows * tile_cols, 8)) schedule(static)
#endif
    for (int64_t tr = 0; tr < tile_rows; ++tr) {
        for (int64_t tc = 0; tc < tile_cols; ++tc) {
            const int64_t r0 = tr * kTile;
            const int64_t c0 = tc * kTile;
            const int64_t r1 = r0 + kTile < rows ? r0 + kTile : rows;
            const int64_t c1 = c0 + kTile < cols ? c0 + kTile : cols;
            for (int64_t i = r0; i < r1; ++i) {
                const Cplx* line = src + c0 * src_row_stride + i;
                for (int64_t j = c0; j < c1; ++j) {
                    dst[j * dst_row_stride + i] = line[(j - c0) * src_row_stride];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* peram 预处理（4x4 / 4x2 哑指标）：按最内维遍历 + 寄存器复用                */
/* ------------------------------------------------------------------------- */

/* a2[x,b,i,h] = sum_{a,g} proj[a,x] A[a,g,b,h] g5[g,i]
 * t1[x,g,b,h] 为中转。 */
bool dress_source_wedge(const Cplx* a, const Cplx* proj, const Cplx* g5, int64_t n, Cplx* t1,
                        Cplx* a2) {
    const int64_t nn = n * n;
    /* (a) t1[x,g,b,h] = sum_a proj[a,x] A[a,g,b,h]：每个 g 一次 (2 x 4) x (4 x n^2) GEMM。 */
    Cplx proj_t[kProj * kDirac];
    for (int64_t x = 0; x < kProj; ++x) {
        for (int64_t d = 0; d < kDirac; ++d) {
            proj_t[x * kDirac + d] = proj[d * kProj + x];
        }
    }
    for (int64_t g = 0; g < kDirac; ++g) {
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj, nn, kDirac, proj_t, kDirac,
                  a + g * nn, kDirac * nn, false, t1 + g * nn, kDirac * nn)) {
            return false;
        }
    }
    /* (b) a2[x,b,i,h] = sum_g t1[x,g,b,h] g5[g,i]：每次 8 个输出、16 次乘加。 */
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(nn, 4096)) schedule(static)
#endif
    for (int64_t bh = 0; bh < nn; ++bh) {
        Cplx t[kProj][kDirac];
        for (int64_t x = 0; x < kProj; ++x) {
            for (int64_t g = 0; g < kDirac; ++g) {
                t[x][g] = t1[(x * kDirac + g) * nn + bh];
            }
        }
        const int64_t b = bh / n;
        const int64_t h = bh - b * n;
        for (int64_t x = 0; x < kProj; ++x) {
            for (int64_t i = 0; i < kDirac; ++i) {
                Cplx acc{0.0, 0.0};
                for (int64_t g = 0; g < kDirac; ++g) {
                    madd(&acc, t[x][g], g5[g * kDirac + i]);
                }
                a2[((x * n + b) * kDirac + i) * n + h] = acc;
            }
        }
    }
    return true;
}

/* c2[o,y,p,t] = sum_s C[o,s,p,t] proj[s,y] */
bool dress_sink_wedge(const Cplx* c, const Cplx* proj, int64_t n, Cplx* c2) {
    const int64_t nn = n * n;
    Cplx proj_t[kProj * kDirac];
    for (int64_t y = 0; y < kProj; ++y) {
        for (int64_t s = 0; s < kDirac; ++s) {
            proj_t[y * kDirac + s] = proj[s * kProj + y];
        }
    }
    for (int64_t o = 0; o < kDirac; ++o) {
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj, nn, kDirac, proj_t, kDirac,
                  c + o * kDirac * nn, nn, false, c2 + o * kProj * nn, nn)) {
            return false;
        }
    }
    return true;
}

/* P1: d3t[(m,n),(d,e)] = sum_c gamma7a[c,e] cmdn[c,m,d,n] */
void dress_p1_sink_peram(const Cplx* d, const Cplx* g7a, int64_t n, Cplx* d3t) {
    const int64_t nn = n * n;
    const int64_t dd = kDirac * kDirac;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(nn, 4096)) schedule(static)
#endif
    for (int64_t mn = 0; mn < nn; ++mn) {
        const int64_t m = mn / n;
        const int64_t nn_i = mn - m * n;
        Cplx t[kDirac][kDirac];
        for (int64_t c = 0; c < kDirac; ++c) {
            for (int64_t dd_i = 0; dd_i < kDirac; ++dd_i) {
                t[c][dd_i] = d[((c * kDirac + m) * n + dd_i) * n + nn_i];
            }
        }
        for (int64_t dd_i = 0; dd_i < kDirac; ++dd_i) {
            for (int64_t e = 0; e < kDirac; ++e) {
                Cplx acc{0.0, 0.0};
                for (int64_t c = 0; c < kDirac; ++c) {
                    madd(&acc, g7a[c * kDirac + e], t[c][dd_i]);
                }
                d3t[mn * dd + dd_i * kDirac + e] = acc;
            }
        }
    }
}

/* P1: e3t[((i*n + j)*4 + q)*n + l] = sum_k ikjl[i,k,j,l] gamma7b[k,q] */
void dress_p1_source_peram(const Cplx* e, const Cplx* g7b, int64_t n, Cplx* e3t) {
    const int64_t total = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(total, 64)) schedule(static)
#endif
    for (int64_t row = 0; row < total; ++row) {
        const int64_t i = row / n;
        const int64_t j = row - i * n;
        Cplx* out = e3t + row * kDirac * n;
        for (int64_t l0 = 0; l0 < n; l0 += kSmallTile) {
            const int64_t l1 = l0 + kSmallTile < n ? l0 + kSmallTile : n;
            Cplx t[kDirac][kSmallTile];
            for (int64_t k = 0; k < kDirac; ++k) {
                const Cplx* line = e + ((i * kDirac + k) * n + j) * n + l0;
                for (int64_t l = l0; l < l1; ++l) {
                    t[k][l - l0] = line[l - l0];
                }
            }
            for (int64_t q = 0; q < kDirac; ++q) {
                for (int64_t l = l0; l < l1; ++l) {
                    Cplx acc{0.0, 0.0};
                    for (int64_t k = 0; k < kDirac; ++k) {
                        madd(&acc, g7b[k * kDirac + q], t[k][l - l0]);
                    }
                    out[q * n + l] = acc;
                }
            }
        }
    }
}

/* P2: fp[(l),(e,q,d)] = sum_{c,k} gamma7a[c,e] ckdl[c,k,d,l] gamma7b[k,q] */
void dress_p2_middle_peram(const Cplx* d, const Cplx* g7a, const Cplx* g7b, int64_t n,
                           Cplx* fp) {
    /* 先把 (c,k) 折成 (e,q)：对固定 (d,l) 做 4x4 双线性型。 */
    const int64_t nn = n * n;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(nn, 4096)) schedule(static)
#endif
    for (int64_t dl = 0; dl < nn; ++dl) {
        const int64_t dd = dl / n;
        const int64_t l = dl - dd * n;
        Cplx t[kDirac][kDirac];
        for (int64_t c = 0; c < kDirac; ++c) {
            for (int64_t k = 0; k < kDirac; ++k) {
                t[c][k] = d[((c * kDirac + k) * n + dd) * n + l];
            }
        }
        for (int64_t e = 0; e < kDirac; ++e) {
            for (int64_t q = 0; q < kDirac; ++q) {
                Cplx acc{0.0, 0.0};
                for (int64_t c = 0; c < kDirac; ++c) {
                    for (int64_t k = 0; k < kDirac; ++k) {
                        const Cplx g = Cplx{g7a[c * kDirac + e].re * g7b[k * kDirac + q].re -
                                                g7a[c * kDirac + e].im * g7b[k * kDirac + q].im,
                                            g7a[c * kDirac + e].re * g7b[k * kDirac + q].im +
                                                g7a[c * kDirac + e].im * g7b[k * kDirac + q].re};
                        madd(&acc, g, t[c][k]);
                    }
                }
                fp[(l * kDirac + e) * kDirac * n + q * n + dd] = acc;
            }
        }
    }
}

/* P1: vvv_perm[(d),(f,b)] <- vvv[b,d,f] */
void permute_sink_vvv(const Cplx* vvv, int64_t n, Cplx* out) {
    const int64_t nn = n * n;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(n, 32)) schedule(static)
#endif
    for (int64_t dd = 0; dd < n; ++dd) {
        Cplx* row = out + dd * nn;
        for (int64_t f = 0; f < n; ++f) {
            Cplx* dst = row + f * n;
            const Cplx* src = vvv + dd * n + f;
            for (int64_t b = 0; b < n; ++b) {
                dst[b] = src[b * nn];
            }
        }
    }
}

/* P1: [q,b,x,r,t] -> [q,r,b,t,x]，按 (q,r,b) 的 (t,x) 交错写；只处理 b in [b0,b0+bb)。 */
void permute_p1_big_block(const Cplx* src, int64_t n, int64_t b0, int64_t bb, Cplx* dst) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (parallel_worth(kDirac * n, 16)) schedule(static)
#endif
    for (int64_t q = 0; q < kDirac; ++q) {
        for (int64_t r = 0; r < n; ++r) {
            Cplx* dbase = dst + (q * n + r) * (bb * n * kProj);
            for (int64_t b = 0; b < bb; ++b) {
                const Cplx* s0 = src + ((((q * n + (b0 + b)) * kProj) * n + r) * n);
                const Cplx* s1 = s0 + n * n;
                Cplx* d = dbase + b * n * kProj;
                for (int64_t t = 0; t < n; ++t) {
                    d[t * kProj] = s0[t];
                    d[t * kProj + 1] = s1[t];
                }
            }
        }
    }
}

/* P1: t2p[((q*n + b)*2 + x)*n + l] <- t2[((x*n + b)*4 + q)*n + l] */
void permute_p1_middle(const Cplx* t2, int64_t n, Cplx* t2p) {
    const int64_t total = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (parallel_worth(total, 128)) schedule(static)
#endif
    for (int64_t q = 0; q < kDirac; ++q) {
        for (int64_t b = 0; b < n; ++b) {
            for (int64_t x = 0; x < kProj; ++x) {
                std::memcpy(t2p + ((q * n + b) * kProj + x) * n,
                            t2 + ((x * n + b) * kDirac + q) * n,
                            static_cast<std::size_t>(n) * sizeof(Cplx));
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* 公共尾部：W[n,(o,y,t)] = sum_p VdV[N,L,n,p] c2[o,y,p,t]                     */
/* ------------------------------------------------------------------------- */
/* c2p[(p),(o,y,t)] <- c2[o,y,p,t] */
void permute_sink_wedge(const Cplx* c2, int64_t n, Cplx* c2p) {
    const int64_t col = kDirac * kProj * n;
#ifdef _OPENMP
#pragma omp parallel for if (parallel_worth(n, 32)) schedule(static)
#endif
    for (int64_t p = 0; p < n; ++p) {
        Cplx* dst = c2p + p * col;
        for (int64_t o = 0; o < kDirac; ++o) {
            for (int64_t y = 0; y < kProj; ++y) {
                const Cplx* src = c2 + ((o * kProj + y) * n + p) * n;
                std::memcpy(dst + (o * kProj + y) * n, src, static_cast<std::size_t>(n) *
                                                                   sizeof(Cplx));
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* 一次 arena：所有中间缓冲从同一块申请，复用 pool 里的同一块内存，            */
/* 避免每次调用重新映射 + 首触页缺页。                                        */
/* ------------------------------------------------------------------------- */

class Arena {
  public:
    bool reserve(std::size_t elements) {
        return lease_.take(elements * sizeof(Cplx));
    }

    Cplx* take(std::size_t elements) {
        const std::size_t bytes = elements * sizeof(Cplx);
        const std::size_t aligned = (bytes + 63u) & ~static_cast<std::size_t>(63u);
        uint8_t* pointer = static_cast<uint8_t*>(lease_.raw()) + used_;
        used_ += aligned;
        return reinterpret_cast<Cplx*>(pointer);
    }

    bool fits() const { return used_ <= lease_.bytes(); }

  private:
    Lease lease_{};
    std::size_t used_ = 0;
};

/* ------------------------------------------------------------------------- */
/* P1 主体                                                                   */
/* ------------------------------------------------------------------------- */

struct P1Buffers {
    Cplx* a2 = nullptr;      /* [x,b,i,h]                      8n^2 */
    Cplx* c2 = nullptr;      /* [o,y,p,t]                      8n^2 */
    Cplx* c2p = nullptr;     /* [p,(o,y,t)]                    8n^2 */
    Cplx* d3t = nullptr;     /* [(m,n),(d,e)]                 16n^2 */
    Cplx* e3t = nullptr;     /* [(i,j),(q,l)]                 16n^2 */
    Cplx* b2t = nullptr;     /* [(e,f),(q,r)]                 16n^2 */
    Cplx* t1 = nullptr;      /* [x,b,i,j]（复用于 [x,g,b,h]）   8n^2 */
    Cplx* t2 = nullptr;      /* [x,b,q,l]                      8n^2 */
    Cplx* t2p = nullptr;     /* [(q,b,x),(l)]                  8n^2 */
    Cplx* t3 = nullptr;      /* [(q,b,x),(r,t)]                8n^3 */
    Cplx* t3p = nullptr;     /* [(q,r),(b,t,x)]（按 b 分块）    8n^3 / 块 */
    Cplx* o = nullptr;       /* [(e,f),(b,t,x)]                8n^3 */
    Cplx* s = nullptr;       /* [(d,e),(t,x)]                  8n^2 */
    Cplx* vvvp = nullptr;    /* [(d),(f,b)]                    n^3  */
    Cplx* w = nullptr;       /* [L][n,(o,y,t)]                 8L n^2 */
    Cplx* out1 = nullptr;    /* [(m,n),(t,x)]                  8n^2 */
    Cplx* z = nullptr;       /* [m,o,L,y,x]                    64L  */
};

std::size_t p1_permute_block(int64_t n) {
    /* t3p 分块列数上限按 ~2 Mi 元素控制，n 不大时一块用完。 */
    const int64_t per_b = kDirac * n * n * kProj;
    int64_t block = 2097152 / (per_b > 0 ? per_b : 1);
    if (block < 1) {
        block = 1;
    }
    if (block > n) {
        block = n;
    }
    return static_cast<std::size_t>(block);
}

bool run_p1(const Cplx* const* in, int64_t n, int64_t mm, int64_t mn, int64_t link, Cplx* out) {
    const int64_t nn = n * n;
    const int64_t nnn = nn * n;
    const int64_t proj_cols = n * kProj;
    const int64_t wedge_cols = kDirac * kProj * n;
    const int64_t block = static_cast<int64_t>(p1_permute_block(n));

    Arena arena;
    const std::size_t elements = static_cast<std::size_t>(kDirac * kProj * nnn) * 2 +
                                 static_cast<std::size_t>(kDirac * n * block * n * kProj) +
                                 static_cast<std::size_t>(16 * nn) * 3 +
                                 static_cast<std::size_t>(8 * nn) * 6 +
                                 static_cast<std::size_t>(nnn) +
                                 static_cast<std::size_t>(8 * link * nn) +
                                 static_cast<std::size_t>(64 * link) + 4096;
    if (!arena.reserve(elements)) {
        return false;
    }
    P1Buffers buf;
    buf.a2 = arena.take(8 * nn);
    buf.c2 = arena.take(8 * nn);
    buf.c2p = arena.take(8 * nn);
    buf.d3t = arena.take(16 * nn);
    buf.e3t = arena.take(16 * nn);
    buf.b2t = arena.take(16 * nn);
    buf.t1 = arena.take(8 * nn);
    buf.t2 = arena.take(8 * nn);
    buf.t2p = arena.take(8 * nn);
    buf.t3 = arena.take(kDirac * kProj * nnn);
    buf.t3p = arena.take(kDirac * n * block * n * kProj);
    buf.o = arena.take(kDirac * kProj * nnn);
    buf.s = arena.take(8 * nn);
    buf.vvvp = arena.take(nnn);
    buf.w = arena.take(8 * link * nn);
    buf.out1 = arena.take(8 * nn);
    buf.z = arena.take(64 * link);
    if (!arena.fits()) {
        return false;
    }

    /* ---- 与 M/N 无关的预处理（每次调用一次） ---- */
    if (!dress_sink_wedge(in[2], in[14], n, buf.c2)) {
        return false;
    }
    permute_sink_wedge(buf.c2, n, buf.c2p);
    if (!dress_source_wedge(in[0], in[13], in[7], n, buf.t1, buf.a2)) {
        return false;
    }
    dress_p1_sink_peram(in[3], in[5], n, buf.d3t);
    dress_p1_source_peram(in[4], in[8], n, buf.e3t);
    permute_middle_pair(in[1], n, buf.b2t);

    for (int64_t m = 0; m < mm; ++m) {
        /* S1: t1[(x,b,i),(j)] = a2[(x,b,i),(h)] * srcVdV[m][(h),(j)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * n * kDirac, n, n, buf.a2, n,
                  in[11] + m * nn, n, false, buf.t1, n)) {
            return false;
        }
        /* S2: t2[(x,b),(q,l)] = t1[(x,b),(i,j)] * e3t[(i,j),(q,l)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * n, kDirac * n, kDirac * n, buf.t1,
                  kDirac * n, buf.e3t, kDirac * n, false, buf.t2, kDirac * n)) {
            return false;
        }
        permute_p1_middle(buf.t2, n, buf.t2p);
        /* S3: t3[(q,b,x),(r,t)] = t2p[(q,b,x),(l)] * srcVVV[m][(l),(r,t)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kDirac * kProj * n, nn, n, buf.t2p, n,
                  in[12] + m * nnn, nn, false, buf.t3, nn)) {
            return false;
        }
        /* S5: o[(e,f),(b,t,x)] += b2t[(e,f),(q,r)] * t3p[(q,r),(b,t,x)]（按 b 分块） */
        bool first_block = true;
        for (int64_t b0 = 0; b0 < n; b0 += block) {
            const int64_t bb = (b0 + block < n) ? block : (n - b0);
            permute_p1_big_block(buf.t3, n, b0, bb, buf.t3p);
            if (!gemm(kCblasNoTrans, kCblasNoTrans, kDirac * n, bb * n * kProj, kDirac * n,
                      buf.b2t, kDirac * n, buf.t3p, bb * n * kProj, !first_block,
                      buf.o + b0 * proj_cols, nnn * kDirac)) {
                return false;
            }
            first_block = false;
        }
        for (int64_t ns = 0; ns < mn; ++ns) {
            /* S6: s[(d,e),(t,x)] = vvvp[(d),(f,b)] * o[(e,f),(b,t,x)]（逐 e） */
            permute_sink_vvv(in[9] + ns * nnn, n, buf.vvvp);
            for (int64_t e = 0; e < kDirac; ++e) {
                if (!gemm(kCblasNoTrans, kCblasNoTrans, n, proj_cols, nn, buf.vvvp, nn,
                          buf.o + e * n * nn * kProj, proj_cols, false,
                          buf.s + e * proj_cols, kDirac * proj_cols)) {
                    return false;
                }
            }
            /* 尾部 step10：out1[(m,n),(t,x)] = d3t[(m,n),(d,e)] * s[(d,e),(t,x)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, kDirac * n, proj_cols, kDirac * n, buf.d3t,
                      kDirac * n, buf.s, proj_cols, false, buf.out1, proj_cols)) {
                return false;
            }
            /* W[L][(n),(o,y,t)] = VdVL[ns,L][(n),(p)] * c2p[(p),(o,y,t)] */
            for (int64_t l = 0; l < link; ++l) {
                if (!gemm(kCblasNoTrans, kCblasNoTrans, n, wedge_cols, n,
                          in[10] + (ns * link + l) * nn, n, buf.c2p, wedge_cols, false,
                          buf.w + l * n * wedge_cols, wedge_cols)) {
                    return false;
                }
            }
            /* z[m,o,L,y,x] = sum_{n,t} out1[(m,n),(t,x)] * W[L][(n),(o,y,t)] */
            std::memset(buf.z, 0, static_cast<std::size_t>(64 * link) * sizeof(Cplx));
            for (int64_t mi = 0; mi < kDirac; ++mi) {
                for (int64_t ni = 0; ni < n; ++ni) {
                    const Cplx* row = buf.out1 + (mi * n + ni) * proj_cols;
                    for (int64_t l = 0; l < link; ++l) {
                        const Cplx* wrow = buf.w + (l * n + ni) * wedge_cols;
                        for (int64_t o = 0; o < kDirac; ++o) {
                            for (int64_t y = 0; y < kProj; ++y) {
                                const Cplx* wt = wrow + (o * kProj + y) * n;
                                Cplx* za = buf.z + ((((mi * kDirac + o) * link + l) * kProj + y) *
                                                    kProj);
                                for (int64_t t = 0; t < n; ++t) {
                                    const Cplx wv = wt[t];
                                    madd(&za[0], row[t * kProj], wv);
                                    madd(&za[1], row[t * kProj + 1], wv);
                                }
                            }
                        }
                    }
                }
            }
            /* out[M,N,G,L,x,y] = sum_{m,o} gcur[G,m,o] z[m,o,L,y,x] */
            for (int64_t g = 0; g < kDirac; ++g) {
                for (int64_t l = 0; l < link; ++l) {
                    for (int64_t xx = 0; xx < kProj; ++xx) {
                        for (int64_t yy = 0; yy < kProj; ++yy) {
                            Cplx acc{0.0, 0.0};
                            for (int64_t mi = 0; mi < kDirac; ++mi) {
                                for (int64_t o = 0; o < kDirac; ++o) {
                                    madd(&acc, in[6][(g * kDirac + mi) * kDirac + o],
                                         buf.z[((((mi * kDirac + o) * link + l) * kProj + yy) *
                                                    kProj +
                                                xx)]);
                                }
                            }
                            out[(((m * mn + ns) * kDirac + g) * link + l) * kProj * kProj +
                                xx * kProj + yy] = acc;
                        }
                    }
                }
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------------- */
/* P2 主体                                                                   */
/* ------------------------------------------------------------------------- */

struct P2Buffers {
    Cplx* a2 = nullptr;   /* [x,b,i,h]        8n^2 */
    Cplx* c2 = nullptr;   /* [o,y,p,t]        8n^2 */
    Cplx* c2p = nullptr;  /* [p,(o,y,t)]      8n^2 */
    Cplx* fp = nullptr;   /* [(l),(e,q,d)]   16n^2 */
    Cplx* m4t = nullptr;  /* [(i,j),(m,n)]   16n^2 */
    Cplx* t1 = nullptr;   /* [x,b,i,j]        8n^2 */
    Cplx* t2 = nullptr;   /* [x,b,m,n]        8n^2 */
    Cplx* svvv = nullptr; /* [(r,t),(l)]      n^3  */
    Cplx* t3 = nullptr;   /* [(r,t),(e,q,d)] 16n^3 */
    Cplx* out4 = nullptr; /* [M][f][(t,d)]   4n^3 */
    Cplx* out4t = nullptr;/* [(d,f)][M][t]   4n^3 */
    Cplx* out2 = nullptr; /* [N][b][M][t]     n^2 M */
    Cplx* tjoin = nullptr;/* [x,m,n][M][t]    8 n^2 M */
    Cplx* y1 = nullptr;   /* [x,o,n][M][t][G]  32 n^2 M */
    Cplx* w = nullptr;    /* [L][n,(o,y,t)]   8L n^2 */
    Cplx* z2 = nullptr;   /* [x,G,L,y]        8L */
};

bool run_p2(const Cplx* const* in, int64_t n, int64_t mm, int64_t mn, int64_t link, Cplx* out) {
    const int64_t nn = n * n;
    const int64_t nnn = nn * n;
    const int64_t proj_cols = n * kProj;
    const int64_t wedge_cols = kDirac * kProj * n;
    const int64_t mid_cols = kDirac * kDirac * n; /* (e,q,d) = 16n */
    const int64_t f_cols = mm * n;                /* (M,t) */

    Arena arena;
    const std::size_t elements = static_cast<std::size_t>(16 * nnn) * 2 +
                                 static_cast<std::size_t>(4 * nnn) * 2 +
                                 static_cast<std::size_t>(16 * nn) * 4 +
                                 static_cast<std::size_t>(8 * nn) * 2 +
                                 static_cast<std::size_t>(nnn) +
                                 static_cast<std::size_t>(n * f_cols) +
                                 static_cast<std::size_t>(8 * n * f_cols) +
                                 static_cast<std::size_t>(32 * n * f_cols) +
                                 static_cast<std::size_t>(8 * link * nn) +
                                 static_cast<std::size_t>(8 * link) + 4096;
    if (!arena.reserve(elements)) {
        return false;
    }
    P2Buffers buf;
    buf.a2 = arena.take(8 * nn);
    buf.c2 = arena.take(8 * nn);
    buf.c2p = arena.take(8 * nn);
    buf.fp = arena.take(16 * nn);
    buf.m4t = arena.take(16 * nn);
    buf.t1 = arena.take(8 * nn);
    buf.t2 = arena.take(8 * nn);
    buf.svvv = arena.take(nnn);
    buf.t3 = arena.take(mid_cols * nn);
    buf.out4 = arena.take(n * nn * mm);
    buf.out4t = arena.take(nn * f_cols);
    buf.out2 = arena.take(n * f_cols);
    buf.tjoin = arena.take(kProj * kDirac * n * f_cols);
    buf.y1 = arena.take(kProj * kDirac * n * f_cols * kDirac);
    buf.w = arena.take(8 * link * nn);
    buf.z2 = arena.take(8 * link);
    if (!arena.fits()) {
        return false;
    }

    /* ---- 与 M/N 无关的预处理 ---- */
    if (!dress_sink_wedge(in[2], in[14], n, buf.c2)) {
        return false;
    }
    permute_sink_wedge(buf.c2, n, buf.c2p);
    if (!dress_source_wedge(in[0], in[13], in[7], n, buf.t1, buf.a2)) {
        return false;
    }
    dress_p2_middle_peram(in[3], in[5], in[8], n, buf.fp);
    permute_source_pair(in[4], n, buf.m4t);

    for (int64_t m = 0; m < mm; ++m) {
        /* S1: t1[(x,b,i),(j)] = a2[(x,b,i),(h)] * srcVdV[m][(h),(j)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * n * kDirac, n, n, buf.a2, n,
                  in[11] + m * nn, n, false, buf.t1, n)) {
            return false;
        }
        /* S2: t2[(x,b),(m,n)] = t1[(x,b),(i,j)] * m4t[(i,j),(m,n)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * n, kDirac * n, kDirac * n, buf.t1,
                  kDirac * n, buf.m4t, kDirac * n, false, buf.t2, kDirac * n)) {
            return false;
        }
        /* S3: t3[(r,t),(e,q,d)] = srcVVV[m][(l),(r,t)]^T * fp[(l),(e,q,d)] */
        transpose_2d(in[12] + m * nnn, buf.svvv, n, nn, nn, n);
        if (!gemm(kCblasNoTrans, kCblasNoTrans, nn, mid_cols, n, buf.svvv, n, buf.fp, mid_cols,
                  false, buf.t3, mid_cols)) {
            return false;
        }
        /* S4: out4[M][(f),(t,d)] = sum_{e,q} eqfr[e,q][(f),(r)] * t3[(r),(t,d)] */
        for (int64_t e = 0; e < kDirac; ++e) {
            for (int64_t q = 0; q < kDirac; ++q) {
                const bool accumulate = (e != 0) || (q != 0);
                if (!gemm(kCblasNoTrans, kCblasNoTrans, n, nn, n,
                          in[1] + (e * kDirac + q) * nn, n, buf.t3 + (e * kDirac + q) * n,
                          mid_cols, accumulate, buf.out4 + m * n * nn, nn)) {
                    return false;
                }
            }
        }
        /* out4[M][f][(t,d)] -> out4t[(d,f)][(M,t)] */
        for (int64_t mi = 0; mi < mm; ++mi) {
            for (int64_t f = 0; f < n; ++f) {
                transpose_2d(buf.out4 + (mi * n + f) * nn, buf.out4t + f * f_cols + mi * n, n, n,
                             n, n * f_cols);
            }
        }
        for (int64_t ns = 0; ns < mn; ++ns) {
            /* S5: out2[(b),(M,t)] = VVV[ns][(b),(d,f)] * out4t[(d,f),(M,t)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, n, f_cols, nn,
                      in[9] + ns * nnn, nn, buf.out4t, f_cols, false, buf.out2, f_cols)) {
                return false;
            }
            /* S6: tjoin[(x,m,n),(M,t)] = t2[(x,b),(m,n)]^T * out2[(b),(M,t)] */
            for (int64_t x = 0; x < kProj; ++x) {
                if (!gemm(kCblasTrans, kCblasNoTrans, kDirac * n, f_cols, n,
                          buf.t2 + x * kDirac * n * n, kDirac * n, buf.out2, f_cols, false,
                          buf.tjoin + x * kDirac * n * f_cols, f_cols)) {
                    return false;
                }
            }
            /* W[L][(n),(o,y,t)] = VdVL[ns,L][(n),(p)] * c2p[(p),(o,y,t)] */
            for (int64_t l = 0; l < link; ++l) {
                if (!gemm(kCblasNoTrans, kCblasNoTrans, n, wedge_cols, n,
                          in[10] + (ns * link + l) * nn, n, buf.c2p, wedge_cols, false,
                          buf.w + l * n * wedge_cols, wedge_cols)) {
                    return false;
                }
            }
            /* y1[x,o,n][(M,t)][G] = sum_m tjoin[x,m,n][(M,t)] gcur[G,m,o] */
            for (int64_t x = 0; x < kProj; ++x) {
                for (int64_t o = 0; o < kDirac; ++o) {
                    for (int64_t ni = 0; ni < n; ++ni) {
                        for (int64_t mt = 0; mt < f_cols; ++mt) {
                            Cplx v[kDirac];
                            for (int64_t mi = 0; mi < kDirac; ++mi) {
                                v[mi] = buf.tjoin[((x * kDirac + mi) * n + ni) * f_cols + mt];
                            }
                            Cplx* dst = buf.y1 + (((x * kDirac + o) * n + ni) * f_cols + mt) *
                                                        kDirac;
                            for (int64_t g = 0; g < kDirac; ++g) {
                                Cplx acc{0.0, 0.0};
                                for (int64_t mi = 0; mi < kDirac; ++mi) {
                                    madd(&acc, v[mi], in[6][(g * kDirac + mi) * kDirac + o]);
                                }
                                dst[g] = acc;
                            }
                        }
                    }
                }
            }
            /* z2[x,G,L,y] = sum_{n,o,t} y1[x,o,n][(M,t)][G] * W[L][(n),(o,y,t)] */
            std::memset(buf.z2, 0, static_cast<std::size_t>(8 * link) * sizeof(Cplx));
            for (int64_t x = 0; x < kProj; ++x) {
                for (int64_t o = 0; o < kDirac; ++o) {
                    for (int64_t ni = 0; ni < n; ++ni) {
                        const Cplx* ybase =
                            buf.y1 + (((x * kDirac + o) * n + ni) * f_cols) * kDirac;
                        for (int64_t l = 0; l < link; ++l) {
                            const Cplx* wrow = buf.w + (l * n + ni) * wedge_cols;
                            for (int64_t y = 0; y < kProj; ++y) {
                                const Cplx* wt = wrow + (o * kProj + y) * n;
                                Cplx* zx = buf.z2 + (((x * kDirac) * link + l) * kProj + y) *
                                                       kProj;
                                for (int64_t t = 0; t < n; ++t) {
                                    const Cplx wv = wt[t];
                                    const Cplx* yv = ybase + t * kDirac;
                                    for (int64_t g = 0; g < kDirac; ++g) {
                                        madd(&zx[g * link * kProj * kProj], yv[g], wv);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (int64_t g = 0; g < kDirac; ++g) {
                for (int64_t l = 0; l < link; ++l) {
                    for (int64_t xx = 0; xx < kProj; ++xx) {
                        for (int64_t yy = 0; yy < kProj; ++yy) {
                            out[(((m * mn + ns) * kDirac + g) * link + l) * kProj * kProj +
                                xx * kProj + yy] =
                                buf.z2[(((xx * kDirac + g) * link + l) * kProj + yy)];
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool supported(int dtype_id, const int64_t* dims, int variant) {
    if (dims == nullptr) {
        return false;
    }
    if (dtype_id != QCD_DTYPE_COMPLEX128) {
        return false;
    }
    if (variant != QCD_PATTERN_3PT_P1 && variant != QCD_PATTERN_3PT_P2) {
        return false;
    }
    const int64_t n = dims[QCD_3PT_DIM_NVEC];
    const int64_t mm = dims[QCD_3PT_DIM_MOM_SOURCE];
    const int64_t mn = dims[QCD_3PT_DIM_MOM_SINK];
    const int64_t link = dims[QCD_3PT_DIM_LINK];
    if (n < 1 || mm < 1 || mn < 1 || link < 1) {
        return false;
    }
    if (dims[QCD_3PT_DIM_DIRAC] != kDirac || dims[QCD_3PT_DIM_PROJ] != kProj) {
        return false;
    }
    /* 所有 GEMM 维度都要落在 int32 内（LP64 CBLAS 路径）。 */
    const int64_t limit = std::numeric_limits<int>::max();
    if (n > limit || mm > limit || mn > limit || link > limit) {
        return false;
    }
    if (n * n > limit || n * n * n > limit || n * n * kDirac * kDirac > limit) {
        return false;
    }
    return true;
}

}  // namespace

extern "C" QCD_EINSUM_API int qcd_kupl_pattern_3pt_supported(int dtype_id, const int64_t* dims,
                                                            int variant) {
    return supported(dtype_id, dims, variant) ? QCD_STATUS_OK : QCD_STATUS_INVALID;
}

extern "C" QCD_EINSUM_API int qcd_kupl_pattern_3pt(int dtype_id, const void* const* inputs,
                                                   void* output, const int64_t* dims,
                                                   int variant) {
    if (!supported(dtype_id, dims, variant)) {
        return QCD_STATUS_INVALID;
    }
    if (inputs == nullptr || output == nullptr) {
        return QCD_STATUS_INVALID;
    }
    for (int index = 0; index < QCD_3PT_INPUT_COUNT; ++index) {
        if (inputs[index] == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    if (!blas().ready()) {
        return QCD_STATUS_INTERNAL;
    }
    const int64_t n = dims[QCD_3PT_DIM_NVEC];
    const int64_t mm = dims[QCD_3PT_DIM_MOM_SOURCE];
    const int64_t mn = dims[QCD_3PT_DIM_MOM_SINK];
    const int64_t link = dims[QCD_3PT_DIM_LINK];
    const Cplx* const* typed = reinterpret_cast<const Cplx* const*>(inputs);
    Cplx* out = reinterpret_cast<Cplx*>(output);
    try {
        const bool ok = (variant == QCD_PATTERN_3PT_P1) ? run_p1(typed, n, mm, mn, link, out)
                                                        : run_p2(typed, n, mm, mn, link, out);
        return ok ? QCD_STATUS_OK : QCD_STATUS_INTERNAL;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}
