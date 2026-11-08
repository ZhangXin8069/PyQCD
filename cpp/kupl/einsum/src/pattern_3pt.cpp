/* 格点 QCD 3pt 主项（P1/P2）专用 CPU 执行 kernel。
 *
 * 目标：把 15 算子 / 14 步的通用执行（每步 pack + GEMM + 输出 permute + 新建
 * 中间缓冲）替换成**固定 14 步、零通用 pack、缓冲全复用**的一次调用。
 *
 * 设计要点（与通用执行层的差异）
 * ------------------------------
 * 1. 收缩顺序由方程结构手工推导（Gmo 的 mu 与 NLnp 的 n 分居两端，Mhj/gi/ax 在源端，
 *    ospt/sy 在汇端，eqfr 与两条 peram 切片构成一个"环"），总乘加数与 opt_einsum 的
 *    `dp` 最优路径同阶（P1 ≈ 3.1e8、P2 ≈ 2.1e8 复数乘加，n = Nev = 50），但：
 *      - 每个中间张量的 layout 在写出时就按下一步的 (m,k)/(k,n) 排好，只有 <= 8n^2
 *        或按 b 分块的小张量需要显式置换，置换是连续 run 拷贝；
 *      - 大 GEMM 只有 3~4 个，形状分别为 (8n x n^2 x n)、(4n x 2n^2 x 4n)、
 *        (n x 2n x n^2)，全部 row-major、B 侧 k/n 连续。
 * 2. Dirac(4)/投影(4,2) 哑指标的预处理（agbh x ax x gi、ospt x sy、ikjl x kq、
 *    cmdn x ce、ckdl x ce x kq）都是小 GEMM 或直循环，不进通用 pack。
 * 3. 中间缓冲通过 pool 复用（避免每次调用重新 mmap + 首触页缺页）。
 *
 * 线程口径：GEMM 全部交给进程内已加载的 CBLAS（OpenBLAS），线程数由
 * `OPENBLAS_NUM_THREADS` 决定；本文件自己的 OpenMP 循环（置换/小预处理）只受
 * `OMP_NUM_THREADS` 约束，且**不与任何 CBLAS 调用嵌套**，因此不存在
 * OMP 线程 x BLAS 线程的乘性超订。
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

inline int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

/* ------------------------------------------------------------------------- */
/* 可复用缓冲池：避免每次调用重新 mmap / 首触 16~64 MB 中间缓冲               */
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
            int best = -1;
            for (int index = 0; index < kSlots; ++index) {
                if (slots_[index].data != nullptr && slots_[index].bytes >= bytes) {
                    if (best < 0 || slots_[best].bytes > slots_[index].bytes) {
                        best = index;
                    }
                }
            }
            if (best >= 0) {
                void* data = slots_[best].data;
                slots_[best].data = nullptr;
                slots_[best].bytes = 0;
                return data;
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
    static constexpr int kSlots = 8;

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

    bool ok() const { return data_ != nullptr || bytes_ == 0; }

    std::size_t bytes() const { return bytes_; }

    void* raw() const { return data_; }

  private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

/* 从同一块 slab 上按 64B 对齐切分，越界即 fail-closed。 */
class Arena {
  public:
    bool reserve(std::size_t bytes) { return lease_.take(bytes); }

    template <typename Type>
    Type* take(std::size_t elements) {
        const std::size_t bytes = elements * sizeof(Type);
        const std::size_t aligned = (bytes + 63u) & ~static_cast<std::size_t>(63u);
        if (used_ + aligned > lease_.bytes()) {
            overflow_ = true;
            return nullptr;
        }
        Type* pointer = reinterpret_cast<Type*>(static_cast<uint8_t*>(lease_.raw()) + used_);
        used_ += aligned;
        return pointer;
    }

    bool ok() const { return lease_.ok() && !overflow_; }

  private:
    Lease lease_{};
    std::size_t used_ = 0;
    bool overflow_ = false;
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

/* C = alpha*A*B + beta*C，alpha=1；beta = accumulate ? 1 : 0。 */
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
    const int64_t limit = std::numeric_limits<int>::max();
    if (m > limit || n > limit || k > limit || lda > limit || ldb > limit || ldc > limit) {
        return false;
    }
    api.lp64(kCblasRowMajor, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
             static_cast<int>(k), &alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
             &beta, c, static_cast<int>(ldc));
    return true;
}

/* ------------------------------------------------------------------------- */
/* 预处理：与 M/N 无关的小收缩                                                 */
/* ------------------------------------------------------------------------- */

/* a3[(x,i)][(b,h)] = sum_{a,g} ax[a,x] agbh[a,g,b,h] gi[g,i]（8 x n^2） */
bool dress_source(const Cplx* agbh, const Cplx* gi, const Cplx* ax, int64_t n, Cplx* a3) {
    Cplx weight[kProj * kDirac * kDirac * kDirac]; /* [(x,i)][(a,g)] = 8 x 16 */
    for (int64_t x = 0; x < kProj; ++x) {
        for (int64_t i = 0; i < kDirac; ++i) {
            for (int64_t a = 0; a < kDirac; ++a) {
                for (int64_t g = 0; g < kDirac; ++g) {
                    const Cplx p = ax[a * kProj + x];
                    const Cplx q = gi[g * kDirac + i];
                    weight[((x * kDirac + i) * kDirac + a) * kDirac + g] =
                        Cplx{p.re * q.re - p.im * q.im, p.re * q.im + p.im * q.re};
                }
            }
        }
    }
    const int64_t nn = n * n;
    return gemm(kCblasNoTrans, kCblasNoTrans, kProj * kDirac, nn, kDirac * kDirac, weight,
                kDirac * kDirac, agbh, nn, false, a3, nn);
}

/* c2[(o,y)][(p,t)] = sum_s ospt[o,s,p,t] sy[s,y]（8 x n^2） */
bool dress_sink(const Cplx* ospt, const Cplx* sy, int64_t n, Cplx* c2) {
    const int64_t nn = n * n;
    Cplx sy_t[kProj * kDirac]; /* [(y)][(s)] */
    for (int64_t y = 0; y < kProj; ++y) {
        for (int64_t s = 0; s < kDirac; ++s) {
            sy_t[y * kDirac + s] = sy[s * kProj + y];
        }
    }
    for (int64_t o = 0; o < kDirac; ++o) {
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj, nn, kDirac, sy_t, kDirac,
                  ospt + o * kDirac * nn, nn, false, c2 + o * kProj * nn, nn)) {
            return false;
        }
    }
    return true;
}

/* c2p[(p)][(o,y,t)] <- c2[(o,y)][(p,t)]（8n^2） */
void permute_sink_to_p(const Cplx* c2, int64_t n, Cplx* c2p) {
    const int64_t nn = n * n;
    const int64_t rows = kDirac * kProj; /* (o,y) */
#ifdef _OPENMP
#pragma omp parallel for if (rows * n > 256) schedule(static)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        const Cplx* src = c2 + row * nn;
        Cplx* dst = c2p + row * n;
        for (int64_t p = 0; p < n; ++p) {
            std::memcpy(dst + p * rows * n, src + p * n,
                        static_cast<std::size_t>(n) * sizeof(Cplx));
        }
    }
}

/* c2t[(p,t)][(o,y)] <- c2[(o,y)][(p,t)]（8n^2） */
void permute_sink_to_pt(const Cplx* c2, int64_t n, Cplx* c2t) {
    const int64_t nn = n * n;
    const int64_t rows = kDirac * kProj;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (nn * rows > 4096) schedule(static)
#endif
    for (int64_t p = 0; p < n; ++p) {
        for (int64_t t = 0; t < n; ++t) {
            Cplx* dst = c2t + (p * n + t) * rows;
            for (int64_t row = 0; row < rows; ++row) {
                dst[row] = c2[row * nn + p * n + t];
            }
        }
    }
}

/* P1: e4[(i,j)][(q,l)] = sum_k ikjl[i,k,j,l] kq[k,q]（16n^2） */
void dress_source_pair(const Cplx* ikjl, const Cplx* kq, int64_t n, Cplx* e4) {
    const int64_t nn = n * n;
    const int64_t stride = kDirac * n; /* 4n */
    Cplx kq_t[kDirac * kDirac];
    for (int64_t q = 0; q < kDirac; ++q) {
        for (int64_t k = 0; k < kDirac; ++k) {
            kq_t[q * kDirac + k] = kq[k * kDirac + q];
        }
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kDirac * n > 128) schedule(static)
#endif
    for (int64_t i = 0; i < kDirac; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const Cplx* base = ikjl + (i * kDirac * n + j) * n;
            Cplx* out = e4 + (i * n + j) * stride;
            for (int64_t q = 0; q < kDirac; ++q) {
                Cplx* dst = out + q * n;
                for (int64_t l = 0; l < n; ++l) {
                    Cplx acc{0.0, 0.0};
                    for (int64_t k = 0; k < kDirac; ++k) {
                        madd(&acc, kq_t[q * kDirac + k], base[k * nn + l]);
                    }
                    dst[l] = acc;
                }
            }
        }
    }
}

/* P1: d3[(mu,n)][(d,e)] = sum_c cmdn[c,mu,d,n] ce[c,e]（16n^2） */
void dress_sink_peram(const Cplx* cmdn, const Cplx* ce, int64_t n, Cplx* d3) {
    const int64_t cols = kDirac * n; /* 4n = (d,e) */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kDirac * kDirac * n > 256) schedule(static)
#endif
    for (int64_t mu = 0; mu < kDirac; ++mu) {
        for (int64_t d = 0; d < n; ++d) {
            for (int64_t nv = 0; nv < n; ++nv) {
                Cplx acc[kDirac];
                for (int64_t e = 0; e < kDirac; ++e) {
                    acc[e] = Cplx{0.0, 0.0};
                }
                for (int64_t c = 0; c < kDirac; ++c) {
                    const Cplx val = cmdn[((c * kDirac + mu) * n + d) * n + nv];
                    for (int64_t e = 0; e < kDirac; ++e) {
                        madd(&acc[e], val, ce[c * kDirac + e]);
                    }
                }
                Cplx* dst = d3 + (mu * n + nv) * cols + d * kDirac;
                for (int64_t e = 0; e < kDirac; ++e) {
                    dst[e] = acc[e];
                }
            }
        }
    }
}

/* P2: d2[(e,q)][(d,l)] = sum_{c,k} ckdl[c,k,d,l] ce[c,e] kq[k,q]（16 x n^2） */
bool dress_p2_middle(const Cplx* ckdl, const Cplx* ce, const Cplx* kq, int64_t n, Cplx* d2) {
    Cplx weight[kDirac * kDirac * kDirac * kDirac]; /* [(e,q)][(c,k)] */
    for (int64_t e = 0; e < kDirac; ++e) {
        for (int64_t q = 0; q < kDirac; ++q) {
            for (int64_t c = 0; c < kDirac; ++c) {
                for (int64_t k = 0; k < kDirac; ++k) {
                    const Cplx a = ce[c * kDirac + e];
                    const Cplx b = kq[k * kDirac + q];
                    weight[((e * kDirac + q) * kDirac + c) * kDirac + k] =
                        Cplx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
                }
            }
        }
    }
    const int64_t nn = n * n;
    return gemm(kCblasNoTrans, kCblasNoTrans, kDirac * kDirac, nn, kDirac * kDirac, weight,
                kDirac * kDirac, ckdl, nn, false, d2, nn);
}

/* P1: b2[(e,f)][(q,r)] <- eqfr[e,q,f,r]（16n^2） */
void permute_middle_p1(const Cplx* eqfr, int64_t n, Cplx* b2) {
    const int64_t nn = n * n;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kDirac * n > 128) schedule(static)
#endif
    for (int64_t e = 0; e < kDirac; ++e) {
        for (int64_t f = 0; f < n; ++f) {
            Cplx* dst = b2 + (e * n + f) * kDirac * n;
            for (int64_t q = 0; q < kDirac; ++q) {
                std::memcpy(dst + q * n, eqfr + (e * kDirac + q) * nn + f * n,
                            static_cast<std::size_t>(n) * sizeof(Cplx));
            }
        }
    }
}

/* P2: b2f[(f)][(e,q,r)] <- eqfr[e,q,f,r]（n x 16n） */
void permute_middle_p2(const Cplx* eqfr, int64_t n, Cplx* b2f) {
    const int64_t nn = n * n;
    const int64_t rows = kDirac * kDirac * n; /* 16n = (e,q,r) */
#ifdef _OPENMP
#pragma omp parallel for if (n * rows > 1024) schedule(static)
#endif
    for (int64_t f = 0; f < n; ++f) {
        Cplx* dst = b2f + f * rows;
        for (int64_t eq = 0; eq < kDirac * kDirac; ++eq) {
            std::memcpy(dst + eq * n, eqfr + eq * nn + f * n,
                        static_cast<std::size_t>(n) * sizeof(Cplx));
        }
    }
}

/* P2: e2[(i,j)][(mu,n)] <- imjn[i,mu,j,n]（16n^2） */
void permute_source_pair_p2(const Cplx* imjn, int64_t n, Cplx* e2) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kDirac * n > 128) schedule(static)
#endif
    for (int64_t i = 0; i < kDirac; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            Cplx* dst = e2 + (i * n + j) * kDirac * n;
            for (int64_t mu = 0; mu < kDirac; ++mu) {
                std::memcpy(dst + mu * n, imjn + ((i * kDirac + mu) * n + j) * n,
                            static_cast<std::size_t>(n) * sizeof(Cplx));
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* 中间张量的换序（每个都按连续 run 拷贝实现）                                 */
/* ------------------------------------------------------------------------- */

/* t1[(x,i,b)][(j)] -> t1p[(x,b)][(i,j)]（8n^2） */
void pack_t1(const Cplx* t1, int64_t n, Cplx* t1p) {
    const int64_t stride = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kProj * kDirac * n > 512) schedule(static)
#endif
    for (int64_t x = 0; x < kProj; ++x) {
        for (int64_t b = 0; b < n; ++b) {
            Cplx* dst = t1p + (x * n + b) * stride;
            for (int64_t i = 0; i < kDirac; ++i) {
                std::memcpy(dst + i * n, t1 + ((x * kDirac + i) * n + b) * n,
                            static_cast<std::size_t>(n) * sizeof(Cplx));
            }
        }
    }
}

/* P1: t2[(x,b)][(q,l)] -> t2p[(q,b,x)][(l)]（8n^2） */
void pack_t2_p1(const Cplx* t2, int64_t n, Cplx* t2p) {
    const int64_t stride = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kProj * kDirac * n > 512) schedule(static)
#endif
    for (int64_t x = 0; x < kProj; ++x) {
        for (int64_t b = 0; b < n; ++b) {
            const Cplx* src = t2 + (x * n + b) * stride;
            for (int64_t q = 0; q < kDirac; ++q) {
                std::memcpy(t2p + ((q * n + b) * kProj + x) * n, src + q * n,
                            static_cast<std::size_t>(n) * sizeof(Cplx));
            }
        }
    }
}

/* P2: t2[(x,b)][(mu,n)] -> t2p[(x,mu,n)][(b)]（8n^2，分块转置） */
void pack_t2_p2(const Cplx* t2, int64_t n, Cplx* t2p) {
    const int64_t stride = kDirac * n;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kProj * n * kDirac > 512) schedule(static)
#endif
    for (int64_t x = 0; x < kProj; ++x) {
        for (int64_t b0 = 0; b0 < n; b0 += 8) {
            const int64_t b1 = min64(b0 + 8, n);
            for (int64_t mu = 0; mu < kDirac; ++mu) {
                for (int64_t nv0 = 0; nv0 < n; nv0 += 8) {
                    const int64_t nv1 = min64(nv0 + 8, n);
                    for (int64_t b = b0; b < b1; ++b) {
                        const Cplx* src = t2 + (x * n + b) * stride + mu * n;
                        for (int64_t nv = nv0; nv < nv1; ++nv) {
                            t2p[((x * kDirac + mu) * n + nv) * n + b] = src[nv];
                        }
                    }
                }
            }
        }
    }
}

/* P1: t3[(q,b,x)][(r,t)]（b 分块）-> t3p[(q,r)][(b,t,x)] */
void pack_t3(const Cplx* t3, int64_t n, int64_t bb, Cplx* t3p) {
    const int64_t nn = n * n;
    const int64_t cols = bb * kProj * n;
    const int64_t pitch = kProj * n;
    const int64_t total = kDirac * n; /* (q,r) */
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (total * bb * n > 4096) schedule(static)
#endif
    for (int64_t q = 0; q < kDirac; ++q) {
        for (int64_t r = 0; r < n; ++r) {
            Cplx* dst = t3p + (q * n + r) * cols;
            for (int64_t b = 0; b < bb; ++b) {
                const Cplx* s0 = t3 + ((q * bb * kProj) + b * kProj) * nn + r * n;
                const Cplx* s1 = s0 + nn;
                Cplx* d = dst + b * pitch;
                for (int64_t t = 0; t < n; ++t) {
                    d[2 * t] = s0[t];
                    d[2 * t + 1] = s1[t];
                }
            }
        }
    }
}

/* P1: Nbdf[b,d,f] -> nbp[(d)][(f,b)]（n^3） */
void permute_sink_vvv_p1(const Cplx* vvv, int64_t n, Cplx* nbp) {
    const int64_t nn = n * n;
    constexpr int64_t kTile = 16;
    const int64_t tiles = (n + kTile - 1) / kTile;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (tiles * tiles > 2) schedule(static)
#endif
    for (int64_t td = 0; td < tiles; ++td) {
        for (int64_t tf = 0; tf < tiles; ++tf) {
            const int64_t d0 = td * kTile;
            const int64_t f0 = tf * kTile;
            const int64_t d1 = min64(d0 + kTile, n);
            const int64_t f1 = min64(f0 + kTile, n);
            for (int64_t b = 0; b < n; ++b) {
                const Cplx* src = vvv + b * nn;
                for (int64_t d = d0; d < d1; ++d) {
                    Cplx* dst = nbp + (d * n + f0) * n + b;
                    const Cplx* line = src + d * n;
                    for (int64_t f = f0; f < f1; ++f) {
                        dst[(f - f0) * n] = line[f];
                    }
                }
            }
        }
    }
}

/* P1: out1[(mu,n)][(t,x)] -> out1t[(mu,x)][(n,t)]（8n^2） */
void pack_out1(const Cplx* out1, int64_t n, Cplx* out1t) {
    const int64_t nn = n * n;
    const int64_t pitch = kProj * n;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (kDirac * n > 128) schedule(static)
#endif
    for (int64_t mu = 0; mu < kDirac; ++mu) {
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t nv = 0; nv < n; ++nv) {
                const Cplx* src = out1 + (mu * n + nv) * pitch + t * kProj;
                Cplx* dst = out1t + (mu * kProj) * nn + nv * n + t;
                dst[0] = src[0];
                dst[nn] = src[1];
            }
        }
    }
}

/* P1: w[L][(n)][(o,y,t)] -> wt[L][(o,y)][(n,t)]（8L n^2） */
void pack_w(const Cplx* w, int64_t n, int64_t link, Cplx* wt) {
    const int64_t nn = n * n;
    const int64_t rows = kDirac * kProj; /* 8 = (o,y) */
    const int64_t pitch = rows * n;      /* 8n */
#ifdef _OPENMP
#pragma omp parallel for collapse(3) if (link * rows * n > 1024) schedule(static)
#endif
    for (int64_t l = 0; l < link; ++l) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t nv = 0; nv < n; ++nv) {
                std::memcpy(wt + (l * rows + row) * nn + nv * n,
                            w + l * n * pitch + nv * pitch + row * n,
                            static_cast<std::size_t>(n) * sizeof(Cplx));
            }
        }
    }
}

/* P2: 单个 (e,q) 块的 Z[(d)][(r,t)] -> Zt[(r)][(d,t)]（n^3） */
void permute_p2_block(const Cplx* z, int64_t n, Cplx* zt) {
    const int64_t nn = n * n;
    constexpr int64_t kTile = 16;
    const int64_t tiles = (n + kTile - 1) / kTile;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (tiles * tiles > 2) schedule(static)
#endif
    for (int64_t td = 0; td < tiles; ++td) {
        for (int64_t tr = 0; tr < tiles; ++tr) {
            const int64_t d0 = td * kTile;
            const int64_t r0 = tr * kTile;
            const int64_t d1 = min64(d0 + kTile, n);
            const int64_t r1 = min64(r0 + kTile, n);
            for (int64_t d = d0; d < d1; ++d) {
                const Cplx* src = z + d * nn;
                for (int64_t r = r0; r < r1; ++r) {
                    std::memcpy(zt + r * nn + d * n, src + r * n,
                                static_cast<std::size_t>(n) * sizeof(Cplx));
                }
            }
        }
    }
}

/* P2: Nbdf[b,d,f] -> nbp[(b)][(f,d)]（n^3） */
void permute_sink_vvv_p2(const Cplx* vvv, int64_t n, Cplx* nbp) {
    const int64_t nn = n * n;
    constexpr int64_t kTile = 16;
    const int64_t tiles = (n + kTile - 1) / kTile;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (tiles * tiles > 2) schedule(static)
#endif
    for (int64_t td = 0; td < tiles; ++td) {
        for (int64_t tf = 0; tf < tiles; ++tf) {
            const int64_t d0 = td * kTile;
            const int64_t f0 = tf * kTile;
            const int64_t d1 = min64(d0 + kTile, n);
            const int64_t f1 = min64(f0 + kTile, n);
            for (int64_t b = 0; b < n; ++b) {
                const Cplx* src = vvv + b * nn;
                Cplx* dst = nbp + b * nn;
                for (int64_t f = f0; f < f1; ++f) {
                    Cplx* line = dst + f * n;
                    const Cplx* col = src + f;
                    for (int64_t d = d0; d < d1; ++d) {
                        line[d] = col[d * n];
                    }
                }
            }
        }
    }
}

/* P2: u[(x,mu,n)][(t)] -> up[(x,mu,t)][(n)]（8n^2） */
void pack_u(const Cplx* u, int64_t n, Cplx* up) {
    const int64_t nn = n * n;
    const int64_t rows = kProj * kDirac; /* 8 = (x,mu) */
#ifdef _OPENMP
#pragma omp parallel for if (rows * n > 128) schedule(static)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        const Cplx* src = u + row * nn;
        for (int64_t t = 0; t < n; ++t) {
            Cplx* dst = up + row * nn + t * n;
            for (int64_t nv = 0; nv < n; ++nv) {
                dst[nv] = src[nv * n + t];
            }
        }
    }
}

/* P2: v[(x,mu,t)][(p)] -> v3[(x,mu)][(p,t)]（8n^2） */
void pack_v(const Cplx* v, int64_t n, Cplx* v3) {
    const int64_t nn = n * n;
    const int64_t rows = kProj * kDirac; /* 8 */
#ifdef _OPENMP
#pragma omp parallel for if (rows * n > 128) schedule(static)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        const Cplx* src = v + row * nn;
        Cplx* dst = v3 + row * nn;
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t p = 0; p < n; ++p) {
                dst[p * n + t] = src[t * n + p];
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* P1 主体                                                                   */
/* ------------------------------------------------------------------------- */

bool run_p1(const Cplx* const* in, int64_t n, int64_t mm, int64_t mn, int64_t link, Cplx* out) {
    const int64_t nn = n * n;
    const int64_t nnn = nn * n;
    const int64_t mid_rows = kDirac * n;          /* 4n = (e,f) / (q,r) / (mu,n) */
    const int64_t wedge_cols = kProj * n;         /* 2n = (t,x) */
    const int64_t sink_cols = kDirac * kProj * n; /* 8n = (o,y,t) */

    /* b 分块：让 t3 / t3p 留在末级缓存（每块 <= 4 Mi 元素） */
    int64_t bb = n;
    {
        const int64_t per_b = kProj * kDirac * nn; /* 8 n^2 */
        const int64_t cap = 262144;
        if (n * per_b > cap) {
            bb = cap / per_b;
            if (bb < 1) {
                bb = 1;
            }
        }
    }

    const std::size_t sizes[] = {
        static_cast<std::size_t>(kProj * kDirac * nn),         /* a3 */
        static_cast<std::size_t>(kDirac * kProj * nn),         /* c2 */
        static_cast<std::size_t>(kDirac * kProj * nn),         /* c2p */
        static_cast<std::size_t>(kDirac * kDirac * nn),        /* e4 */
        static_cast<std::size_t>(kDirac * kDirac * nn),        /* d3 */
        static_cast<std::size_t>(kDirac * kDirac * nn),        /* b2 */
        static_cast<std::size_t>(kProj * kDirac * nn),         /* t1 */
        static_cast<std::size_t>(kProj * kDirac * nn),         /* t1p */
        static_cast<std::size_t>(kProj * kDirac * nn),         /* t2 */
        static_cast<std::size_t>(kProj * kDirac * nn),         /* t2p */
        static_cast<std::size_t>(kDirac * bb * kProj * nn),    /* t3 */
        static_cast<std::size_t>(kDirac * n * bb * kProj * n), /* t3p */
        static_cast<std::size_t>(kDirac * kProj * nnn),        /* o */
        static_cast<std::size_t>(nnn),                         /* nbp */
        static_cast<std::size_t>(kDirac * kProj * nn),         /* s */
        static_cast<std::size_t>(kDirac * kProj * nn),         /* out1 */
        static_cast<std::size_t>(kDirac * kProj * nn),         /* out1t */
        static_cast<std::size_t>(sink_cols * n * link),        /* w */
        static_cast<std::size_t>(kDirac * kProj * nn * link),  /* wt */
        static_cast<std::size_t>(kDirac * kDirac * link * kProj * kProj) /* z */
    };
    const int count = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));
    std::size_t bytes = 0;
    for (int index = 0; index < count; ++index) {
        bytes += (sizes[index] * sizeof(Cplx) + 63u) & ~static_cast<std::size_t>(63u);
    }
    Arena arena;
    if (!arena.reserve(bytes)) {
        return false;
    }
    Cplx* a3 = arena.take<Cplx>(sizes[0]);
    Cplx* c2 = arena.take<Cplx>(sizes[1]);
    Cplx* c2p = arena.take<Cplx>(sizes[2]);
    Cplx* e4 = arena.take<Cplx>(sizes[3]);
    Cplx* d3 = arena.take<Cplx>(sizes[4]);
    Cplx* b2 = arena.take<Cplx>(sizes[5]);
    Cplx* t1 = arena.take<Cplx>(sizes[6]);
    Cplx* t1p = arena.take<Cplx>(sizes[7]);
    Cplx* t2 = arena.take<Cplx>(sizes[8]);
    Cplx* t2p = arena.take<Cplx>(sizes[9]);
    Cplx* t3 = arena.take<Cplx>(sizes[10]);
    Cplx* t3p = arena.take<Cplx>(sizes[11]);
    Cplx* o = arena.take<Cplx>(sizes[12]);
    Cplx* nbp = arena.take<Cplx>(sizes[13]);
    Cplx* s = arena.take<Cplx>(sizes[14]);
    Cplx* out1 = arena.take<Cplx>(sizes[15]);
    Cplx* out1t = arena.take<Cplx>(sizes[16]);
    Cplx* w = arena.take<Cplx>(sizes[17]);
    Cplx* wt = arena.take<Cplx>(sizes[18]);
    Cplx* z = arena.take<Cplx>(sizes[19]);
    if (!arena.ok()) {
        return false;
    }

    /* ---- 与 M/N 无关的预处理 ---- */
    if (!dress_source(in[0], in[7], in[13], n, a3)) {
        return false;
    }
    if (!dress_sink(in[2], in[14], n, c2)) {
        return false;
    }
    permute_sink_to_p(c2, n, c2p);
    dress_source_pair(in[4], in[8], n, e4);
    dress_sink_peram(in[3], in[5], n, d3);
    permute_middle_p1(in[1], n, b2);

    const Cplx* gm = in[6]; /* Gmo[G][mu][o] */
    const int64_t group = kDirac * kProj; /* 8 */

    for (int64_t m = 0; m < mm; ++m) {
        /* S1: t1[(x,i,b)][(j)] = sum_h a3[(x,i)][(b,h)] Mhj[m][(h)][(j)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * kDirac * n, n, n, a3, n,
                  in[11] + m * nn, n, false, t1, n)) {
            return false;
        }
        pack_t1(t1, n, t1p);
        /* S2: t2[(x,b)][(q,l)] = sum_{(i,j)} t1p[(x,b)][(i,j)] e4[(i,j)][(q,l)] */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * n, mid_rows, mid_rows, t1p, mid_rows,
                  e4, mid_rows, false, t2, mid_rows)) {
            return false;
        }
        pack_t2_p1(t2, n, t2p);
        /* S3/S4：按 b 分块 */
        for (int64_t b0 = 0; b0 < n; b0 += bb) {
            const int64_t cur = min64(bb, n - b0);
            /* S3: t3[(q,b,x)][(r,t)] = sum_l t2p[(q,b,x)][(l)] Mlrt[m][(l)][(r,t)] */
            if (cur == n) {
                if (!gemm(kCblasNoTrans, kCblasNoTrans, kDirac * kProj * n, nn, n, t2p, n,
                          in[12] + m * nnn, nn, false, t3, nn)) {
                    return false;
                }
            } else {
                for (int64_t q = 0; q < kDirac; ++q) {
                    if (!gemm(kCblasNoTrans, kCblasNoTrans, cur * kProj, nn, n,
                              t2p + (q * n + b0) * kProj * n, n, in[12] + m * nnn, nn, false,
                              t3 + q * cur * kProj * nn, nn)) {
                        return false;
                    }
                }
            }
            pack_t3(t3, n, cur, t3p);
            /* S4: o[(e,f)][(b,t,x)] = sum_{(q,r)} b2[(e,f)][(q,r)] t3p[(q,r)][(b,t,x)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, mid_rows, cur * wedge_cols, mid_rows, b2,
                      mid_rows, t3p, cur * wedge_cols, false, o + b0 * wedge_cols,
                      n * wedge_cols)) {
                return false;
            }
        }

        for (int64_t ns = 0; ns < mn; ++ns) {
            /* S6: s[(d,e)][(t,x)] = sum_{f,b} Nbdf[ns][b,d,f] o[(e,f)][(b,t,x)] */
            permute_sink_vvv_p1(in[9] + ns * nnn, n, nbp);
            for (int64_t e = 0; e < kDirac; ++e) {
                if (!gemm(kCblasNoTrans, kCblasNoTrans, n, wedge_cols, nn, nbp, nn,
                          o + e * wedge_cols * nn, wedge_cols, false, s + e * wedge_cols,
                          sink_cols)) {
                    return false;
                }
            }
            /* S7: out1[(mu,n)][(t,x)] = sum_{(d,e)} d3[(mu,n)][(d,e)] s[(d,e)][(t,x)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, mid_rows, wedge_cols, mid_rows, d3,
                      mid_rows, s, wedge_cols, false, out1, wedge_cols)) {
                return false;
            }
            pack_out1(out1, n, out1t);
            /* S8: w[L][(n)][(o,y,t)] = NLnp[ns,L][(n)][(p)] c2p[(p)][(o,y,t)] */
            for (int64_t l = 0; l < link; ++l) {
                if (!gemm(kCblasNoTrans, kCblasNoTrans, n, sink_cols, n,
                          in[10] + (ns * link + l) * nn, n, c2p, sink_cols, false,
                          w + l * n * sink_cols, sink_cols)) {
                    return false;
                }
            }
            pack_w(w, n, link, wt);
            for (int64_t l = 0; l < link; ++l) {
                /* S9: z[(mu,x),(o,y)] = sum_{(n,t)} out1t[(mu,x)][(n,t)] wt[l][(o,y)][(n,t)] */
                if (!gemm(kCblasNoTrans, kCblasTrans, group, group, nn, out1t, nn,
                          wt + l * group * nn, nn, false, z, group)) {
                    return false;
                }
                /* S10: out[G][l][x][y] = sum_{mu,o} Gmo[G][mu][o] z[(mu,x),(o,y)] */
                Cplx* dst =
                    out + ((m * mn + ns) * kDirac) * link * kProj * kProj + l * kProj * kProj;
                for (int64_t G = 0; G < kDirac; ++G) {
                    for (int64_t x = 0; x < kProj; ++x) {
                        for (int64_t y = 0; y < kProj; ++y) {
                            Cplx acc{0.0, 0.0};
                            for (int64_t mu = 0; mu < kDirac; ++mu) {
                                for (int64_t oo = 0; oo < kDirac; ++oo) {
                                    madd(&acc, gm[(G * kDirac + mu) * kDirac + oo],
                                         z[(mu * kProj + x) * group + (oo * kProj + y)]);
                                }
                            }
                            dst[G * link * kProj * kProj + x * kProj + y] = acc;
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

bool run_p2(const Cplx* const* in, int64_t n, int64_t mm, int64_t mn, int64_t link, Cplx* out) {
    const int64_t nn = n * n;
    const int64_t nnn = nn * n;
    const int64_t pair_cols = kDirac * kDirac * n; /* 16n = (e,q,r) */
    const int64_t mid_rows = kDirac * n;           /* 4n = (x,mu,n) / (mu,n) */
    const int64_t group = kDirac * kProj;          /* 8 = (x,mu) / (o,y) */

    const std::size_t sizes[] = {
        static_cast<std::size_t>(kProj * kDirac * nn),  /* a3 */
        static_cast<std::size_t>(kDirac * kProj * nn),  /* c2 */
        static_cast<std::size_t>(kDirac * kProj * nn),  /* c2t */
        static_cast<std::size_t>(kDirac * kDirac * nn), /* e2 */
        static_cast<std::size_t>(kDirac * kDirac * nn), /* d2 */
        static_cast<std::size_t>(pair_cols * n),        /* b2f */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* t1 */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* t1p */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* t2 */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* t2p */
        static_cast<std::size_t>(nnn),                  /* zbuf */
        static_cast<std::size_t>(nnn),                  /* zt */
        static_cast<std::size_t>(nnn),                  /* y */
        static_cast<std::size_t>(nnn),                  /* nbp */
        static_cast<std::size_t>(nn),                   /* tot */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* u */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* up */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* v */
        static_cast<std::size_t>(kProj * kDirac * nn),  /* v3 */
        static_cast<std::size_t>(group * group)         /* z2 */
    };
    const int count = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));
    std::size_t bytes = 0;
    for (int index = 0; index < count; ++index) {
        bytes += (sizes[index] * sizeof(Cplx) + 63u) & ~static_cast<std::size_t>(63u);
    }
    Arena arena;
    if (!arena.reserve(bytes)) {
        return false;
    }
    Cplx* a3 = arena.take<Cplx>(sizes[0]);
    Cplx* c2 = arena.take<Cplx>(sizes[1]);
    Cplx* c2t = arena.take<Cplx>(sizes[2]);
    Cplx* e2 = arena.take<Cplx>(sizes[3]);
    Cplx* d2 = arena.take<Cplx>(sizes[4]);
    Cplx* b2f = arena.take<Cplx>(sizes[5]);
    Cplx* t1 = arena.take<Cplx>(sizes[6]);
    Cplx* t1p = arena.take<Cplx>(sizes[7]);
    Cplx* t2 = arena.take<Cplx>(sizes[8]);
    Cplx* t2p = arena.take<Cplx>(sizes[9]);
    Cplx* zbuf = arena.take<Cplx>(sizes[10]);
    Cplx* zt = arena.take<Cplx>(sizes[11]);
    Cplx* y = arena.take<Cplx>(sizes[12]);
    Cplx* nbp = arena.take<Cplx>(sizes[13]);
    Cplx* tot = arena.take<Cplx>(sizes[14]);
    Cplx* u = arena.take<Cplx>(sizes[15]);
    Cplx* up = arena.take<Cplx>(sizes[16]);
    Cplx* v = arena.take<Cplx>(sizes[17]);
    Cplx* v3 = arena.take<Cplx>(sizes[18]);
    Cplx* z2 = arena.take<Cplx>(sizes[19]);
    if (!arena.ok()) {
        return false;
    }

    if (!dress_source(in[0], in[7], in[13], n, a3)) {
        return false;
    }
    if (!dress_sink(in[2], in[14], n, c2)) {
        return false;
    }
    permute_sink_to_pt(c2, n, c2t);
    permute_source_pair_p2(in[4], n, e2);
    if (!dress_p2_middle(in[3], in[5], in[8], n, d2)) {
        return false;
    }
    permute_middle_p2(in[1], n, b2f);

    const Cplx* gm = in[6];

    for (int64_t m = 0; m < mm; ++m) {
        /* S1/S2：源端夹心（与 P1 相同） */
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * kDirac * n, n, n, a3, n,
                  in[11] + m * nn, n, false, t1, n)) {
            return false;
        }
        pack_t1(t1, n, t1p);
        if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * n, mid_rows, mid_rows, t1p, mid_rows,
                  e2, mid_rows, false, t2, mid_rows)) {
            return false;
        }
        pack_t2_p2(t2, n, t2p);
        /* 环 (d2, Mlrt, eqfr, Nbdf)：逐 (e,q) 块求值，块内中间量 n^3 可留缓存 */
        for (int64_t eq = 0; eq < kDirac * kDirac; ++eq) {
            /* Z[(d)][(r,t)] = sum_l d2[(e,q)][(d,l)] Mlrt[m][(l)][(r,t)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, n, nn, n, d2 + eq * nn, n,
                      in[12] + m * nnn, nn, false, zbuf, nn)) {
                return false;
            }
            permute_p2_block(zbuf, n, zt);
            /* Y[(f)][(d,t)] += sum_r b2f[(f)][(e,q,r)] zt[(r)][(d,t)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, n, nn, n, b2f + eq * n, pair_cols, zt, nn,
                      eq != 0, y, nn)) {
                return false;
            }
        }

        for (int64_t ns = 0; ns < mn; ++ns) {
            /* T[(b)][(t)] = sum_{(f,d)} Nbdf[ns][b,d,f] Y[(f),(d,t)] */
            permute_sink_vvv_p2(in[9] + ns * nnn, n, nbp);
            if (!gemm(kCblasNoTrans, kCblasNoTrans, n, n, nn, nbp, nn, y, n, false, tot, n)) {
                return false;
            }
            /* U[(x,mu,n)][(t)] = sum_b t2p[(x,mu,n)][(b)] T[(b)][(t)] */
            if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * mid_rows, n, n, t2p, n, tot, n,
                      false, u, n)) {
                return false;
            }
            pack_u(u, n, up);
            for (int64_t l = 0; l < link; ++l) {
                /* V[(x,mu,t)][(p)] = sum_n up[(x,mu,t)][(n)] NLnp[ns,l][(n)][(p)] */
                if (!gemm(kCblasNoTrans, kCblasNoTrans, kProj * mid_rows, n, n, up, n,
                          in[10] + (ns * link + l) * nn, n, false, v, n)) {
                    return false;
                }
                pack_v(v, n, v3);
                /* Z2[(x,mu)][(o,y)] = sum_{(p,t)} v3[(x,mu)][(p,t)] c2t[(p,t)][(o,y)] */
                if (!gemm(kCblasNoTrans, kCblasNoTrans, group, group, nn, v3, nn, c2t, group,
                          false, z2, group)) {
                    return false;
                }
                Cplx* dst =
                    out + ((m * mn + ns) * kDirac) * link * kProj * kProj + l * kProj * kProj;
                for (int64_t G = 0; G < kDirac; ++G) {
                    for (int64_t x = 0; x < kProj; ++x) {
                        for (int64_t y = 0; y < kProj; ++y) {
                            Cplx acc{0.0, 0.0};
                            for (int64_t mu = 0; mu < kDirac; ++mu) {
                                for (int64_t oo = 0; oo < kDirac; ++oo) {
                                    madd(&acc, gm[(G * kDirac + mu) * kDirac + oo],
                                         z2[(x * kDirac + mu) * group + (oo * kProj + y)]);
                                }
                            }
                            dst[G * link * kProj * kProj + x * kProj + y] = acc;
                        }
                    }
                }
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------------- */
/* 参数校验                                                                  */
/* ------------------------------------------------------------------------- */

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
    /* 所有 GEMM 维度都要落在 LP64 CBLAS 的 int32 范围内。 */
    const int64_t limit = std::numeric_limits<int>::max();
    if (n > limit || mm > limit || mn > limit || link > limit) {
        return false;
    }
    if (n * n > limit || n * n * n > limit) {
        return false;
    }
    if (n * n * (kDirac * kProj) > limit) {
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
