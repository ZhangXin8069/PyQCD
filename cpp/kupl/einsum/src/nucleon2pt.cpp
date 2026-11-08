// KUPL(CPU) 专用核子/sush 2pt 收缩 kernel。
//
// 覆盖两条生产方程（dims/输入输出约定见 include/nucleon2pt.h）：
//   variant 0  E1  gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M
//   variant 1  S1  eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN
//
// ===========================================================================
// 1. 统一范式（canonical form）
// ===========================================================================
// 两条方程是同一个张量网络的两种下标命名。记 inputs[0..2] 为三个 peram
// P0,P1,P2，每个带一对 spinor 下标 (u_i,v_i) 与两条矢量腿（一条连 VVV#1 =
// inputs[5]、一条连 VVV#2 = inputs[6]）；三个 4x4 gamma 两两配对 peram 的
// spinor，构成三角形：
//
//   G20  配对 P2.u2 <-> P0.u0                 （E1 = inputs[4]，S1 = inputs[3]）
//   G01  配对 P0.v0 <-> P1.v1                 （E1 = inputs[3]，S1 = inputs[4]）
//   G12  配对 P2.v2 <-> P1.u1                 （两条方程都是 inputs[7]）
//
// G12 在生产数据里是宇称投影 P+ = (1+gamma4)/2（秩 2）。kernel 对 G12 做
// 秩 r 分解 G12[v2,u1] = sum_{k<r} u_k[v2] * conj(v_k[u1])（列主元 MGS-QR，
// 判据 rel_tol = 1e-14，r 取 1..4），于是：
//
//   PA [u2,v0][a1,a2]  = sum_u0 G20[u2,u0] P0[u0,v0][a1,a2]
//   PCp[k][c1][u2,c2]  = sum_v2 u_k[v2] P2[u2,v2][c1,c2]
//   PBB[v0,k][b1,b2]   = sum_{u1,v1} G01[v0,v1] conj(v_k[u1]) P1[u1,v1][b1,b2]
//
//   X[k,b1][a1][u2,c2] = sum_c1 PCp[k][c1][u2,c2] V1[m1,a1,b1,c1]     r*N 个 GEMM
//   Y[v0,b2][a1][u2,c2] = sum_a2 PA[u2,v0][a1,a2] V2[m2,a2,b2,c2]     4*N 个 GEMM
//   Z[(k,b1),(v0,b2)]  += sum_{a1,u2,c2} X_a1 * Y_a1^T                 N   个 GEMM
//   out[m2,m1]          = sum_{k,v0,b1,b2} Z[(k,b1),(v0,b2)] PBB[v0,k][b1,b2]
//
// 乘加量（每个动量对）：X 半 r*4*N^4，Y 半 4*4*N^4，join r*16*N^4。
//   r = 4（一般 4x4 gamma）：96 N^4 —— 与 opt_einsum dp 路径的 9.60e9 @ N=100 相同
//   r = 2（生产投影 P+）    ：56 N^4 —— 5.60e9 @ N=100（省掉 42% 乘加）
// 秩 r=4 时该算式与直接改写逐位等价（只是换了收缩次序），r<4 时截断误差
// <= rel_tol * |G12|，对 1e-12 的验收口径安全。
//
// 这个次序比通用路径的 DP 最优更省：把 canonical 网络（PA/PCp/PBB/V1/V2）
// 交给 contract_path(optimize='optimal') 得到 96 N^4（r=2），本 kernel 是 56 N^4。
//
// ===========================================================================
// 2. 布局：三段 GEMM 全部原地命中源张量，无 256MB 级 pack/transpose
// ===========================================================================
// 每个 GEMM 的 A/B 都直接取源张量里"最内维连续"的二维块：
//   * X 半：A = V1 的 (a1,c1) 块。S1 源布局 (b1,c1,a1) -> 用 Trans（lda = n²，
//            M 维 a1 连续）；E1 源布局 (a1,b1,c1) -> NoTrans（lda = n²）。
//   * Y 半：B = V2 的 (a2,c2) 块。两条方程源布局都是 (b2,?,?) 且 a2 连续
//            （S1: (b2,a2,c2)；E1: (b2,c2,a2)）-> Trans/NoTrans 由源布局决定。
//   * join：X/Y 的中间缓冲布局挑成 [row][a1|][u2,c2]，使 (k,b1)/(v0,b2) 有统一
//            行 stride、a1 的 stride 恰好是 block = 4N，join 用同一块内存做
//            (rN x block) x (block x block)^T。
// 对照：通用路径（以及本文件的 steps 参考路径）会先物化自然布局、再 pack 成
// K-major，pack 本身要额外 1GB 级读写；本 kernel 把它消掉了。
//
// ===========================================================================
// 3. 分块、工作区与三段 GEMM 的 M 维（2026-11 这一轮的主要优化点）
// ===========================================================================
// 中间缓冲 x / y / acc / z 与四个小张量（PA/PCp/PBB/PA 分块）全部放进
// thread_local 工作区复用：
//   * 不再每调用 malloc + 清零 384MB（N=100 时实测这一步占 107ms / 322ms）；
//     x/y 由 beta=0 的 GEMM 整体覆盖写，**不需要清零**；
//   * 工作区只涨不缩；任何一次调用要读的位置都在本次调用内被写过
//     （GEMM beta=0 全覆盖 + acc 显式清零），不存在跨调用残留
//     （PYQCD_KUPL_N2PT_POISON=1 可自检：全缓冲先填 NaN，输出仍有限且正确）；
//   * 默认 nb = n（不分块），只在 x+y 超出预算时按预算切小
//     （PYQCD_KUPL_N2PT_BUDGET_MB，默认 512MB；PYQCD_KUPL_N2PT_CHUNK 可强制）。
// 三段 GEMM 的 M 维都做到 O(N) 以上，避免小 GEMM 的 pack 开销：
//   * X 半：每个 (k,b1) 一个 (nb x 4N) GEMM，M = nb；
//   * Y 半：把 (v0,a1L,u2) 合成一个行索引 -> 每个 b2 只有**一个**
//     (16*nb x N x N) GEMM，B（V2 的 b2 切片）每 b2 只 pack 一次。为此每个
//     a1 分块先把 PA 的一个分块重排成 [v0][a1L][u2][a2]（v0 stride = nb*block，
//     整轮下来搬运量≈PA 大小，可忽略）；相应地 y 按 [b2][v0][a1L][u2,c2] 存放。
//   * join：每个 a1 一个 (rN x 4N x 4N) GEMM，累加到 per-thread Z。
// build_core（PA/PCp/PBB）与动量无关，已提到动量循环之外，每次调用只算一次
// （旧实现每对 (m1,m2) 重算一次）。
// 旧实现在 nb < n（预算不足）时索引仍按 chunk = n 计算，会越界写坏堆；这次
// 一并修掉（nb 可以自由小于 n）。
//
// ===========================================================================
// 4. 线程口径
// ===========================================================================
// 并发只发生在"独立 GEMM 批"这一层：`omp parallel for` 的线程数由
// omp_get_max_threads()（= OMP_NUM_THREADS）给出；进入并行批之前把 CBLAS
// 线程数压到 1、退出后恢复，所以总并发 = OMP_NUM_THREADS，绝不 OMP x BLAS
// 超订。若 CBLAS 不提供 openblas_{get,set}_num_threads，则所有批退化为串行
// 调用（并发由 OPENBLAS_NUM_THREADS 决定），同样不超订。
//
// ===========================================================================
// 5. 诊断开关（只影响诊断，不影响数值）
// ===========================================================================
//   PYQCD_KUPL_N2PT_MODE=fused|steps    steps = 显式物化 + pack 的 7 阶段参考路径
//   PYQCD_KUPL_N2PT_PROFILE=1           逐阶段耗时打到 stderr
//   PYQCD_KUPL_N2PT_BUDGET_MB=<int>     x+y 中间缓冲预算，默认 512
//   PYQCD_KUPL_N2PT_CHUNK=<int>         a1 分块大小 nb
//   PYQCD_KUPL_N2PT_POISON=1            每次调用前把中间缓冲填 NaN，自检
//                                       "未写就读 / 跨调用残留"

#include "nucleon2pt.h"

#include <dlfcn.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {

using c128 = std::complex<double>;

constexpr int kSpin = 4;               // Dirac spinor 维度（必须为 4）
constexpr double kRankRelTol = 1e-14;  // G12 数值秩判据（相对 Frobenius 范数）

// CBLAS 行主序常量（不依赖 cblas.h）
enum { kRowMajor = 101, kNoTrans = 111, kTrans = 112 };

// --------------------------------------------------------------------------
// BLAS 动态绑定（与 cpp/kupl/einsum/src/apply_contract.cpp 同款做法）
// --------------------------------------------------------------------------
using ZgemmFn = void (*)(int, int, int, int, int, int, const void*, const void*, int,
                         const void*, int, const void*, void*, int);
using SetThreadsFn = int (*)(int);
using GetThreadsFn = int (*)(void);

struct BlasApi {
    ZgemmFn zgemm = nullptr;
    SetThreadsFn set_threads = nullptr;
    GetThreadsFn get_threads = nullptr;
};

template <typename Function>
void bind_symbol(void* handle, const char* name, Function* function) {
    *function = nullptr;
    if (handle == nullptr) {
        return;
    }
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return;
    }
    static_assert(sizeof(Function) == sizeof(symbol), "dlsym function-pointer size mismatch");
    std::memcpy(function, &symbol, sizeof(Function));
}

void bind_all(void* handle, BlasApi* api) {
    bind_symbol(handle, "cblas_zgemm", &api->zgemm);
    bind_symbol(handle, "openblas_set_num_threads", &api->set_threads);
    bind_symbol(handle, "openblas_get_num_threads", &api->get_threads);
}

BlasApi load_blas() {
    BlasApi api;
    const char* requested = std::getenv("PYQCD_EINSUM_CBLAS_LIB");
    if (requested != nullptr && requested[0] != '\0') {
        void* handle = dlopen(requested, RTLD_NOW | RTLD_LOCAL);
        bind_all(handle, &api);
        if (api.zgemm != nullptr) {
            return api;
        }
        if (handle != nullptr) {
            dlclose(handle);
        }
    }
    bind_all(RTLD_DEFAULT, &api);  // 进程内已载入（Python/numpy）的情形
    if (api.zgemm != nullptr) {
        return api;
    }
    static const char* kCandidates[] = {"libcblas.so.3", "libcblas.so", "libopenblas.so.0",
                                        "libopenblas.so", "libcblas.dylib", "libopenblas.dylib"};
    for (const char* candidate : kCandidates) {
        void* handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            continue;
        }
        bind_all(handle, &api);
        if (api.zgemm != nullptr) {
            return api;
        }
        dlclose(handle);
    }
    return BlasApi{};
}

const BlasApi& blas() {
    static BlasApi api = load_blas();
    return api;
}

// 内部并行批期间把 CBLAS 压到 1 线程；析构恢复，避免 OMP x BLAS 超订。
class BlasSerialGuard {
  public:
    BlasSerialGuard() {
        const BlasApi& api = blas();
        if (api.set_threads != nullptr && api.get_threads != nullptr) {
            saved_ = api.get_threads();
            api.set_threads(1);
        }
    }
    ~BlasSerialGuard() {
        if (saved_ > 0) {
            blas().set_threads(saved_);
        }
    }
    BlasSerialGuard(const BlasSerialGuard&) = delete;
    BlasSerialGuard& operator=(const BlasSerialGuard&) = delete;
    bool active() const { return saved_ > 0; }

  private:
    int saved_ = -1;
};

// accumulate=false 时 C 被覆盖（beta = 0），true 时累加（beta = 1）。
inline void gemm(int m, int n, int k, const c128* a, int lda, const c128* b, int ldb, c128* c,
                 int ldc, bool trans_a, bool trans_b, bool accumulate = true) {
    static const c128 one(1.0, 0.0);
    static const c128 zero(0.0, 0.0);
    blas().zgemm(kRowMajor, trans_a ? kTrans : kNoTrans, trans_b ? kTrans : kNoTrans, m, n, k,
                 &one, a, lda, b, ldb, accumulate ? &one : &zero, c, ldc);
}

// --------------------------------------------------------------------------
// G12 的秩 r 分解：G12[i,j] ~= sum_{k<r} u[k][i] * conj(v[k][j])
// 列主元修正 Gram-Schmidt（kSpin = 4，带一次再正交化）。
// --------------------------------------------------------------------------
struct RankFactor {
    int rank = 0;
    c128 u[kSpin][kSpin]{};  // u[k][i]
    c128 v[kSpin][kSpin]{};  // 参与 conj(v[k][j])
};

RankFactor factor_rank(const c128 a[kSpin][kSpin]) {
    RankFactor out;
    double norm0 = 0.0;
    for (int i = 0; i < kSpin; ++i) {
        for (int j = 0; j < kSpin; ++j) {
            norm0 += std::norm(a[i][j]);
        }
    }
    norm0 = std::sqrt(norm0);
    if (norm0 == 0.0) {
        return out;
    }
    bool used[kSpin] = {false, false, false, false};
    for (int step = 0; step < kSpin; ++step) {
        // 对每个未用列算正交化后的残差范数，挑最大者（列主元）
        int best = -1;
        double best_norm = -1.0;
        c128 work[kSpin][kSpin];
        for (int j = 0; j < kSpin; ++j) {
            if (used[j]) {
                continue;
            }
            c128 w[kSpin];
            for (int i = 0; i < kSpin; ++i) {
                w[i] = a[i][j];
            }
            for (int pass = 0; pass < 2; ++pass) {
                for (int k = 0; k < out.rank; ++k) {
                    c128 acc(0.0, 0.0);
                    for (int i = 0; i < kSpin; ++i) {
                        acc += std::conj(out.u[k][i]) * w[i];
                    }
                    for (int i = 0; i < kSpin; ++i) {
                        w[i] -= acc * out.u[k][i];
                    }
                }
            }
            double nrm = 0.0;
            for (int i = 0; i < kSpin; ++i) {
                nrm += std::norm(w[i]);
            }
            nrm = std::sqrt(nrm);
            for (int i = 0; i < kSpin; ++i) {
                work[j][i] = w[i];
            }
            if (nrm > best_norm) {
                best_norm = nrm;
                best = j;
            }
        }
        if (best < 0 || best_norm <= kRankRelTol * norm0) {
            break;
        }
        used[best] = true;
        for (int i = 0; i < kSpin; ++i) {
            out.u[out.rank][i] = work[best][i] / best_norm;
        }
        // v_k[j] = conj(<u_k, A[:,j]>)
        for (int j = 0; j < kSpin; ++j) {
            c128 acc(0.0, 0.0);
            for (int i = 0; i < kSpin; ++i) {
                acc += std::conj(out.u[out.rank][i]) * a[i][j];
            }
            out.v[out.rank][j] = std::conj(acc);
        }
        ++out.rank;
    }
    return out;
}

// --------------------------------------------------------------------------
// 变体描述表
// --------------------------------------------------------------------------
struct GammaDesc {
    int op;
    int peram0;
    int slot0;
    int peram1;
    int slot1;
};

struct VariantDesc {
    int spin_axis[3][2];  // peram i 的 (u,v) 在源张量里的轴号
    int v1_axis[3];       // peram i 连 VVV#1 的轴号
    int v2_axis[3];       // peram i 连 VVV#2 的轴号
    GammaDesc g20;
    GammaDesc g01;
    GammaDesc g12;
    int v1_perm[3];  // canonical (a1,b1,c1) -> VVV#1 的源轴号（1..3）
    int v2_perm[3];
    bool v1_trans;   // X 半 GEMM 中 VVV#1 用转置形式
    bool v2_trans;   // Y 半 GEMM 中 VVV#2 用转置形式
    bool tied_momentum;
    int out_ndim;
};

// S1: eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN
const VariantDesc kS1 = {
    {{0, 1}, {0, 1}, {0, 1}},
    {2, 2, 2},
    {3, 3, 3},
    {3, 2, 0, 0, 0},     // G20 = ce
    {4, 1, 1, 0, 1},     // G01 = mo
    {7, 2, 1, 1, 0},     // G12 = ga
    {3, 1, 2},           // V1 = Nbdf: a1=f(轴3), b1=b(轴1), c1=d(轴2)
    {2, 1, 3},           // V2 = Mnph: a2=p(轴2), b2=n(轴1), c2=h(轴3)
    true,
    false,
    false,
    2,
};

// E1: gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M
const VariantDesc kE1 = {
    {{0, 1}, {0, 1}, {0, 1}},
    {3, 3, 3},
    {2, 2, 2},
    {4, 2, 0, 0, 0},     // G20 = og
    {3, 0, 1, 1, 1},     // G01 = ac
    {7, 2, 1, 1, 0},     // G12 = em
    {1, 2, 3},           // V1 = Mbdf: a1=b(轴1), b1=d(轴2), c1=f(轴3)
    {3, 1, 2},           // V2 = Mnph: a2=h(轴3), b2=n(轴1), c2=p(轴2)
    false,
    true,
    true,
    1,
};

void canonical_gamma(const c128* src, const GammaDesc& desc, int canon_peram0, int canon_slot0,
                     int canon_peram1, int canon_slot1, c128 out[kSpin][kSpin]) {
    const bool same = desc.peram0 == canon_peram0 && desc.slot0 == canon_slot0 &&
                      desc.peram1 == canon_peram1 && desc.slot1 == canon_slot1;
    const bool swapped = desc.peram0 == canon_peram1 && desc.slot0 == canon_slot1 &&
                         desc.peram1 == canon_peram0 && desc.slot1 == canon_slot0;
    for (int i = 0; i < kSpin; ++i) {
        for (int j = 0; j < kSpin; ++j) {
            out[i][j] = same ? src[i * kSpin + j]
                             : (swapped ? src[j * kSpin + i] : c128(0.0, 0.0));
        }
    }
}

// --------------------------------------------------------------------------
// 访问器与工具
// --------------------------------------------------------------------------
bool env_flag(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool profile_enabled() { return env_flag("PYQCD_KUPL_N2PT_PROFILE"); }
bool poison_enabled() { return env_flag("PYQCD_KUPL_N2PT_POISON"); }

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t budget_bytes() {
    const char* env = std::getenv("PYQCD_KUPL_N2PT_BUDGET_MB");
    int64_t mb = 512;
    if (env != nullptr && env[0] != '\0') {
        char* end = nullptr;
        const long long value = std::strtoll(env, &end, 10);
        if (end != env && value > 0) {
            mb = value;
        }
    }
    return mb * 1024LL * 1024LL;
}

struct PeramView {
    const c128* base = nullptr;
    int64_t stride_u = 0;
    int64_t stride_v = 0;
    int64_t stride_1 = 0;
    int64_t stride_2 = 0;
};

PeramView make_peram_view(const void* ptr, int64_t nvec, const VariantDesc& desc, int index) {
    PeramView view;
    view.base = static_cast<const c128*>(ptr);
    int64_t dims[4] = {0, 0, 0, 0};
    dims[desc.spin_axis[index][0]] = kSpin;
    dims[desc.spin_axis[index][1]] = kSpin;
    dims[desc.v1_axis[index]] = nvec;
    dims[desc.v2_axis[index]] = nvec;
    int64_t stride[4];
    stride[3] = 1;
    for (int k = 2; k >= 0; --k) {
        stride[k] = stride[k + 1] * dims[k + 1];
    }
    view.stride_u = stride[desc.spin_axis[index][0]];
    view.stride_v = stride[desc.spin_axis[index][1]];
    view.stride_1 = stride[desc.v1_axis[index]];
    view.stride_2 = stride[desc.v2_axis[index]];
    return view;
}

struct VvvView {
    const c128* base = nullptr;
    int64_t stride_m = 0;
    int64_t stride_leg[3] = {0, 0, 0};
};

VvvView make_vvv_view(const void* ptr, int64_t nvec, int64_t nmom, const int perm[3]) {
    VvvView view;
    view.base = static_cast<const c128*>(ptr);
    const int64_t dims[4] = {nmom, nvec, nvec, nvec};
    int64_t stride[4];
    stride[3] = 1;
    for (int k = 2; k >= 0; --k) {
        stride[k] = stride[k + 1] * dims[k + 1];
    }
    view.stride_m = stride[0];
    for (int leg = 0; leg < 3; ++leg) {
        view.stride_leg[leg] = stride[perm[leg]];
    }
    return view;
}

// --------------------------------------------------------------------------
// 工作区：跨调用复用（thread_local），只涨不缩。
// x / y 由 GEMM(beta=0) 与显式 setup 完整覆盖，因此不清零；任何一次调用都不会
// 读到上一次调用留下的内容（见 PYQCD_KUPL_N2PT_POISON 自检）。
// --------------------------------------------------------------------------
struct RawBuf {
    c128* ptr = nullptr;
    size_t cap = 0;

    bool ensure(size_t count) {
        if (count <= cap) {
            return true;
        }
        c128* fresh = static_cast<c128*>(std::malloc(count * sizeof(c128)));
        if (fresh == nullptr) {
            return false;
        }
        std::free(ptr);
        ptr = fresh;
        cap = count;
        return true;
    }
    ~RawBuf() { std::free(ptr); }
    RawBuf() = default;
    RawBuf(const RawBuf&) = delete;
    RawBuf& operator=(const RawBuf&) = delete;
};

struct Workspace {
    RawBuf pa;    // [v0][a1][u2][a2]       16 n^2
    RawBuf pac;   // [v0][a1L][u2][a2]      4*nb*block（PA 的一个 a1 分块，见下）
    RawBuf pcp;   // [k][c1][u2,c2]         r*4 n^2
    RawBuf pbb;   // [v0*r+k][b1][b2]       4r n^2
    RawBuf x;     // [(k,b1)][a1L][u2,c2]   rows_x * nb * block
    RawBuf y;     // [(v0,b2)][a1L][u2,c2]  rows_y * nb * block
    RawBuf acc;   // [tid][(k,b1)][u2,c2]   threads * rows_x * block
    RawBuf z;     // [(k,b1)][u2,c2]        rows_x * block
};

Workspace& workspace() {
    thread_local Workspace ws;
    return ws;
}

// --------------------------------------------------------------------------
// 三个"禁闭"小步：PA / PCp / PBB。复杂度都是 O(4*r*N^2) 量级，且与动量无关，
// 每次调用只算一次。
//   PA  布局 [v0][a1][u2][a2]：a1 stride = block、u2 stride = n，直接给 Y 半
//       GEMM 当 (a1,u2) x a2 的 A 矩阵（lda = n）。
//   PCp 布局 [k][c1][u2,c2]  ：给 X 半 GEMM 当 c1 x (u2,c2) 的 B（ldb = block）。
//   PBB 布局 [v0*r+k][b1][b2]：最后一步收缩用。
// --------------------------------------------------------------------------
void build_core(const PeramView& p0, const PeramView& p1, const PeramView& p2,
                const c128 g20[kSpin][kSpin], const c128 g01[kSpin][kSpin], const RankFactor& rf,
                int64_t n, int threads, c128* pa, c128* pcp, c128* pbb) {
    const int r = rf.rank;
    const int64_t block = kSpin * n;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t idx = 0; idx < static_cast<int64_t>(kSpin) * kSpin; ++idx) {
        const int u2 = static_cast<int>(idx) / kSpin;
        const int v0 = static_cast<int>(idx) % kSpin;
        c128* dst = pa + static_cast<int64_t>(v0) * kSpin * n * n + static_cast<int64_t>(u2) * n;
        for (int64_t a1 = 0; a1 < n; ++a1) {
            for (int64_t a2 = 0; a2 < n; ++a2) {
                c128 acc(0.0, 0.0);
                for (int u0 = 0; u0 < kSpin; ++u0) {
                    const c128 g = g20[u2][u0];
                    if (g == c128(0.0, 0.0)) {
                        continue;
                    }
                    const int64_t off = static_cast<int64_t>(u0) * p0.stride_u +
                                        static_cast<int64_t>(v0) * p0.stride_v +
                                        a1 * p0.stride_1 + a2 * p0.stride_2;
                    acc += g * p0.base[off];
                }
                dst[a1 * block + a2] = acc;
            }
        }
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t idx = 0; idx < static_cast<int64_t>(r) * n; ++idx) {
        const int k = static_cast<int>(idx / n);
        const int64_t c1 = idx % n;
        c128* dst = pcp + (static_cast<int64_t>(k) * n + c1) * block;
        for (int u2 = 0; u2 < kSpin; ++u2) {
            for (int64_t c2 = 0; c2 < n; ++c2) {
                c128 acc(0.0, 0.0);
                for (int v2 = 0; v2 < kSpin; ++v2) {
                    const c128 g = rf.u[k][v2];
                    if (g == c128(0.0, 0.0)) {
                        continue;
                    }
                    const int64_t off = static_cast<int64_t>(u2) * p2.stride_u +
                                        static_cast<int64_t>(v2) * p2.stride_v +
                                        c1 * p2.stride_1 + c2 * p2.stride_2;
                    acc += g * p2.base[off];
                }
                dst[static_cast<int64_t>(u2) * n + c2] = acc;
            }
        }
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t idx = 0; idx < static_cast<int64_t>(kSpin) * r; ++idx) {
        const int v0 = static_cast<int>(idx) / r;
        const int k = static_cast<int>(idx) % r;
        c128* dst = pbb + (static_cast<int64_t>(v0) * r + k) * n * n;
        for (int64_t b1 = 0; b1 < n; ++b1) {
            for (int64_t b2 = 0; b2 < n; ++b2) {
                c128 acc(0.0, 0.0);
                for (int u1 = 0; u1 < kSpin; ++u1) {
                    const c128 w = std::conj(rf.v[k][u1]);
                    for (int v1 = 0; v1 < kSpin; ++v1) {
                        const c128 g = g01[v0][v1];
                        if (g == c128(0.0, 0.0)) {
                            continue;
                        }
                        const int64_t off = static_cast<int64_t>(u1) * p1.stride_u +
                                            static_cast<int64_t>(v1) * p1.stride_v +
                                            b1 * p1.stride_1 + b2 * p1.stride_2;
                        acc += w * g * p1.base[off];
                    }
                }
                dst[b1 * n + b2] = acc;
            }
        }
    }
}

// out = sum_{k,v0,b1,b2} Z[(k,b1),(b2,v0)] PBB[v0,k][b1,b2]
// Z 布局 [(k,b1)][(b2,v0)]（block = 4n 列，列序 r = b2*kSpin + v0），
// PBB 布局 [(v0*r+k)][b1][b2]。4*r*N^3 次乘加（N=100 时 8e6），按 v0 并行。
c128 contract_final(const c128* z, const c128* pbb, int64_t n, int r, int threads) {
    const int64_t block = kSpin * n;
    c128 partial[kSpin];
    const int nthreads = static_cast<int>(std::min<int64_t>(threads, kSpin));
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int v0 = 0; v0 < kSpin; ++v0) {
        c128 total(0.0, 0.0);
        for (int k = 0; k < r; ++k) {
            const c128* w_block = pbb + (static_cast<int64_t>(v0) * r + k) * n * n;
            for (int64_t b1 = 0; b1 < n; ++b1) {
                const c128* row = z + (static_cast<int64_t>(k) * n + b1) * block + v0;
                const c128* w = w_block + b1 * n;
                c128 local(0.0, 0.0);
                for (int64_t b2 = 0; b2 < n; ++b2) {
                    local += row[b2 * kSpin] * w[b2];
                }
                total += local;
            }
        }
        partial[v0] = total;
    }
    c128 result(0.0, 0.0);
    for (int i = 0; i < kSpin; ++i) {
        result += partial[i];
    }
    return result;
}

struct Setup {
    c128 g20[kSpin][kSpin];
    c128 g01[kSpin][kSpin];
    RankFactor rf;
    PeramView p0, p1, p2;
    VvvView v1, v2;
    int r = 0;
};

Setup make_setup(const c128* const* in, int64_t nvec, int64_t nmom, const VariantDesc& desc) {
    Setup s;
    canonical_gamma(in[desc.g20.op], desc.g20, 2, 0, 0, 0, s.g20);
    canonical_gamma(in[desc.g01.op], desc.g01, 0, 1, 1, 1, s.g01);
    c128 g12[kSpin][kSpin];
    canonical_gamma(in[desc.g12.op], desc.g12, 2, 1, 1, 0, g12);
    s.rf = factor_rank(g12);
    s.r = s.rf.rank;
    s.p0 = make_peram_view(in[0], nvec, desc, 0);
    s.p1 = make_peram_view(in[1], nvec, desc, 1);
    s.p2 = make_peram_view(in[2], nvec, desc, 2);
    s.v1 = make_vvv_view(in[5], nvec, nmom, desc.v1_perm);
    s.v2 = make_vvv_view(in[6], nvec, nmom, desc.v2_perm);
    return s;
}

// VVV 的 GEMM 形式与实际 leading dimension（fail closed 校验）。
struct VvvForm {
    bool trans = false;
    int64_t ld = 0;
};

bool resolve_vvv(const VvvView& view, int64_t n, VvvForm* form) {
    form->trans = view.stride_leg[0] == 1;
    form->ld = form->trans ? view.stride_leg[2] : view.stride_leg[0];
    if (form->trans) {
        return view.stride_leg[2] >= n;
    }
    return view.stride_leg[2] == 1 && view.stride_leg[0] >= n;
}

// a1 分块大小：默认尽量不分块（nb = n），只在 x+y 会超出预算
// （PYQCD_KUPL_N2PT_BUDGET_MB，默认 512MB）时才切小。实测 N=100 时 nb = n
// 最快（X 半 M = n、Y 半 M = 16n、join N = 4n 三段 GEMM 都到本机 zgemm 峰值）。
int64_t default_chunk(int64_t n, int r) {
    const int64_t block = kSpin * n;
    const int64_t per_a1_bytes =
        (static_cast<int64_t>(r) * n + kSpin * n) * block * static_cast<int64_t>(sizeof(c128));
    int64_t nb = n;
    if (per_a1_bytes > 0) {
        const int64_t by_budget = budget_bytes() / per_a1_bytes;
        if (nb > by_budget) {
            nb = by_budget;
        }
    }
    const char* env = std::getenv("PYQCD_KUPL_N2PT_CHUNK");
    if (env != nullptr && env[0] != '\0') {
        char* end = nullptr;
        const long long value = std::strtoll(env, &end, 10);
        if (end != env && value > 0) {
            nb = value;
        }
    }
    if (nb < 1) {
        nb = 1;
    }
    if (nb > n) {
        nb = n;
    }
    return nb;
}

// --------------------------------------------------------------------------
// fused 主路径
// --------------------------------------------------------------------------
int run_fused(const c128* const* in, c128* out, int64_t n, int64_t nmom,
              const VariantDesc& desc) {
    const Setup setup = make_setup(in, n, nmom, desc);
    if (setup.r == 0) {
        const int64_t total = desc.tied_momentum ? nmom : nmom * nmom;
        for (int64_t i = 0; i < total; ++i) {
            out[i] = c128(0.0, 0.0);
        }
        return QCD_STATUS_OK;
    }
    const int r = setup.r;
    const int64_t block = kSpin * n;                         // (u2,c2) = 4N 连续块
    const int64_t rows_x = static_cast<int64_t>(r) * n;      // (k,b1)
    const int64_t rows_y = static_cast<int64_t>(kSpin) * n;  // (v0,b2)
    // 由源布局判定 VVV 的 GEMM 形式与实际 leading dimension（fail closed 校验）
    VvvForm v1f, v2f;
    if (!resolve_vvv(setup.v1, n, &v1f) || !resolve_vvv(setup.v2, n, &v2f)) {
        return QCD_STATUS_INTERNAL;  // 描述表与源布局不一致
    }
    const bool v1_trans = v1f.trans;
    const bool v2_trans = v2f.trans;
    const int64_t v1_ld = v1f.ld;
    const int64_t v2_ld = v2f.ld;
    // BLAS 接口用 int：N 极大时直接失败，避免静默截断。
    if (n > 4096 || block * block > INT32_MAX || rows_x * block > INT32_MAX) {
        return QCD_STATUS_INVALID;
    }
    const int64_t a1_stride = setup.v1.stride_leg[0];  // V1 里 a1 的 stride

    int threads = omp_get_max_threads();
    if (threads < 1) {
        threads = 1;
    }
    const BlasSerialGuard serial_guard;
    if (!serial_guard.active()) {
        threads = 1;
    }

    const int64_t nb = default_chunk(n, r);

    Workspace& ws = workspace();
    const size_t pa_elems = static_cast<size_t>(kSpin) * kSpin * n * n;
    const size_t pac_elems = static_cast<size_t>(kSpin) * nb * block;
    const size_t pcp_elems = static_cast<size_t>(r) * kSpin * n * n;
    const size_t pbb_elems = static_cast<size_t>(kSpin) * r * n * n;
    const size_t x_elems = static_cast<size_t>(rows_x) * nb * block;
    const size_t y_elems = static_cast<size_t>(rows_y) * nb * block;
    const size_t acc_elems = static_cast<size_t>(threads) * rows_x * block;
    const size_t z_elems = static_cast<size_t>(rows_x) * block;
    if (!ws.pa.ensure(pa_elems) || !ws.pac.ensure(pac_elems) || !ws.pcp.ensure(pcp_elems) ||
        !ws.pbb.ensure(pbb_elems) || !ws.x.ensure(x_elems) || !ws.y.ensure(y_elems) ||
        !ws.acc.ensure(acc_elems) || !ws.z.ensure(z_elems)) {
        return QCD_STATUS_MEMORY;
    }
    c128* pa = ws.pa.ptr;
    c128* pac = ws.pac.ptr;
    c128* pcp = ws.pcp.ptr;
    c128* pbb = ws.pbb.ptr;
    c128* x = ws.x.ptr;
    c128* y = ws.y.ptr;
    c128* acc = ws.acc.ptr;
    c128* z = ws.z.ptr;

    if (poison_enabled()) {
        // 自检用：把所有中间缓冲填 NaN。若存在"未写就读"或跨调用残留，输出会变 NaN。
        const c128 poison(std::nan(""), std::nan(""));
        c128* bufs[8] = {pa, pac, pcp, pbb, x, y, acc, z};
        const size_t sizes[8] = {pa_elems, pac_elems, pcp_elems, pbb_elems, x_elems, y_elems,
                                 acc_elems, z_elems};
        for (int b = 0; b < 8; ++b) {
            std::fill(bufs[b], bufs[b] + sizes[b], poison);
        }
    }

    const double t_all0 = now_seconds();
    double t_core = 0.0, t_x = 0.0, t_y = 0.0, t_join = 0.0, t_final = 0.0;
    double t0 = t_all0;

    build_core(setup.p0, setup.p1, setup.p2, setup.g20, setup.g01, setup.rf, n, threads, pa, pcp,
               pbb);
    t_core += now_seconds() - t0;

    const int64_t acc_stride = rows_x * block;
    const int64_t x_row_stride = nb * block;
    const int64_t y_row_stride = nb * block;

    for (int64_t m1 = 0; m1 < nmom; ++m1) {
        for (int64_t m2i = 0; m2i < (desc.tied_momentum ? 1 : nmom); ++m2i) {
            const int64_t m2 = desc.tied_momentum ? m1 : m2i;
            const c128* v1m = setup.v1.base + m1 * setup.v1.stride_m;
            const c128* v2m = setup.v2.base + m2 * setup.v2.stride_m;

            t0 = now_seconds();
            // 累加器清零（per-thread 各一份，最后归约）
            const int64_t acc_total = static_cast<int64_t>(threads) * acc_stride;
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t i = 0; i < acc_total; ++i) {
                acc[i] = c128(0.0, 0.0);
            }

            for (int64_t a1_begin = 0; a1_begin < n; a1_begin += nb) {
                const int64_t nb_cur = std::min(nb, n - a1_begin);

                // ---- X 半：X[k,b1][a1|][u2,c2] ----
                // 每个 (k,b1) 一个 (nb_cur x block) GEMM；k 在外层，pcp[k] 驻留。
                t0 = now_seconds();
#pragma omp parallel for schedule(static) num_threads(threads)
                for (int64_t kb = 0; kb < rows_x; ++kb) {
                    const int k = static_cast<int>(kb / n);
                    const int64_t b1 = kb % n;
                    const c128* bmat = pcp + (static_cast<int64_t>(k) * n) * block;
                    c128* cmat = x + kb * nb * block;
                    const c128* amat = v1m + b1 * setup.v1.stride_leg[1] + a1_begin * a1_stride;
                    gemm(static_cast<int>(nb_cur), static_cast<int>(block), static_cast<int>(n),
                         amat, static_cast<int>(v1_ld), bmat, static_cast<int>(block), cmat,
                         static_cast<int>(block), v1_trans, false, false);
                }
                t_x += now_seconds() - t0;
                t0 = now_seconds();

                // ---- Y 半：Y[b2][v0][a1|][u2,c2] ----
                // 关键：把 (v0,a1L,u2) 合成一个行索引，于是每个 b2 只需要**一个**
                // GEMM（M = 4*4*nb，N = n，K = n），B（V2 的 b2 切片）每 b2 只
                // pack 一次。行 stride = n 要求 A 的行空间是 [v0][a1L][u2][a2]
                // 且 v0 stride = nb*block；PA 本身是 [v0][a1][u2][a2]
                // （v0 stride = n*block），所以每个 a1 分块先做一次 4*nb*block 的
                // 重排（每个分块搬 16*nb*n 个元素，整轮下来约等于 PA 大小）。
                // y 因此按 [b2][v0][a1L][u2,c2] 存放：(v0,b2) 行 stride = nb*block，
                // 行序是 b2*4+v0，contract_final 按这个序配对 PBB。
                for (int v0 = 0; v0 < kSpin; ++v0) {
                    c128* dst = pac + static_cast<int64_t>(v0) * nb * block;
                    const c128* src = pa + static_cast<int64_t>(v0) * n * block + a1_begin * block;
                    std::memcpy(dst, src, static_cast<size_t>(nb_cur) * block * sizeof(c128));
                    if (nb_cur < nb) {
                        // 尾分块：把未用的 a1L 行清零，保证 GEMM 不会把脏数据写进 y
                        std::fill(dst + nb_cur * block, dst + nb * block, c128(0.0, 0.0));
                    }
                }
                t_y += now_seconds() - t0;
                t0 = now_seconds();
#pragma omp parallel for schedule(static) num_threads(threads)
                for (int64_t b2 = 0; b2 < n; ++b2) {
                    const c128* bmat = v2m + b2 * setup.v2.stride_leg[1];
                    c128* cmat = y + b2 * kSpin * nb * block;
                    gemm(static_cast<int>(kSpin * kSpin * nb), static_cast<int>(n),
                         static_cast<int>(n), pac, static_cast<int>(n), bmat,
                         static_cast<int>(v2_ld), cmat, static_cast<int>(n), false, v2_trans,
                         false);
                }
                t_y += now_seconds() - t0;
                t0 = now_seconds();

                // ---- join：逐 a1 累加 X_a1 * Y_a1^T ----
                t0 = now_seconds();
#pragma omp parallel for schedule(static) num_threads(threads)
                for (int64_t a1 = 0; a1 < nb_cur; ++a1) {
                    const int tid = omp_get_thread_num();
                    c128* target = acc + static_cast<int64_t>(tid) * acc_stride;
                    gemm(static_cast<int>(rows_x), static_cast<int>(block),
                         static_cast<int>(block), x + a1 * block,
                         static_cast<int>(x_row_stride), y + a1 * block,
                         static_cast<int>(y_row_stride), target, static_cast<int>(block), false,
                         true);
                }
                t_join += now_seconds() - t0;
            }

            // ---- 归约 + 与 PBB 收缩 ----
            t0 = now_seconds();
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t i = 0; i < acc_stride; ++i) {
                c128 sum(0.0, 0.0);
                for (int t = 0; t < threads; ++t) {
                    sum += acc[static_cast<int64_t>(t) * acc_stride + i];
                }
                z[i] = sum;
            }
            const c128 result = contract_final(z, pbb, n, r, threads);
            if (desc.out_ndim == 2) {
                out[m2 * nmom + m1] = result;
            } else {
                out[m1] = result;
            }
            t_final += now_seconds() - t0;
        }
    }

    if (profile_enabled()) {
        const double elapsed = now_seconds() - t_all0;
        std::fprintf(stderr,
                     "[kupl-n2pt:fused] n=%lld nmom=%lld r=%d threads=%d nb=%lld "
                     "total=%.3f s | s0_core=%.3f s1_Xhalf=%.3f s2_Yhalf=%.3f s3_join=%.3f "
                     "s4_final=%.3f\n",
                     static_cast<long long>(n), static_cast<long long>(nmom), r, threads,
                     static_cast<long long>(nb), elapsed, t_core, t_x, t_y, t_join, t_final);
    }
    return QCD_STATUS_OK;
}

// --------------------------------------------------------------------------
// 参考路径（steps）：显式物化自然布局 + pack + 单个大 GEMM。
// 语义与 fused 完全一致，只用于诊断/对照（PYQCD_KUPL_N2PT_MODE=steps）。
// 7 个阶段与 dp 路径的对应关系：
//   s0 禁闭/投影（= dp step0 + step4 的 gamma 吸收）
//   s1 X 自然布局   (= dp step2: V1 x P2)
//   s2 Y 自然布局   (= dp step1: V2 x P0)
//   s3 pack X       (= 无对应，通用路径的隐式转置)
//   s4 pack Y
//   s5 单个大 GEMM  (= dp step3: 主收缩)
//   s6 与 PBB 收缩  (= dp step5+6)
// --------------------------------------------------------------------------
int run_steps(const c128* const* in, c128* out, int64_t n, int64_t nmom,
              const VariantDesc& desc) {
    const Setup setup = make_setup(in, n, nmom, desc);
    if (setup.r == 0) {
        const int64_t total = desc.tied_momentum ? nmom : nmom * nmom;
        for (int64_t i = 0; i < total; ++i) {
            out[i] = c128(0.0, 0.0);
        }
        return QCD_STATUS_OK;
    }
    const int r = setup.r;
    const int64_t block = kSpin * n;
    VvvForm v1f, v2f;
    if (!resolve_vvv(setup.v1, n, &v1f) || !resolve_vvv(setup.v2, n, &v2f)) {
        return QCD_STATUS_INTERNAL;
    }
    const bool v1_trans = v1f.trans;
    const bool v2_trans = v2f.trans;
    const int64_t v1_ld = v1f.ld;
    const int64_t v2_ld = v2f.ld;
    if (n > 4096 || block * block > INT32_MAX) {
        return QCD_STATUS_INVALID;
    }

    int threads = omp_get_max_threads();
    if (threads < 1) {
        threads = 1;
    }
    const BlasSerialGuard serial_guard;
    if (!serial_guard.active()) {
        threads = 1;
    }

    Workspace& ws = workspace();
    const size_t pa_elems = static_cast<size_t>(kSpin) * kSpin * n * n;
    const size_t pcp_elems = static_cast<size_t>(r) * kSpin * n * n;
    const size_t pbb_elems = static_cast<size_t>(kSpin) * r * n * n;
    const size_t z_elems = static_cast<size_t>(r) * n * block;
    const size_t xnat_elems = static_cast<size_t>(kSpin) * r * n * n * n;
    const size_t ynat_elems = static_cast<size_t>(kSpin) * kSpin * n * n * n;
    const size_t xkm_elems = static_cast<size_t>(kSpin) * n * n * (r * n);
    const size_t ykm_elems = static_cast<size_t>(kSpin) * n * n * block;
    if (!ws.pa.ensure(pa_elems) || !ws.pcp.ensure(pcp_elems) || !ws.pbb.ensure(pbb_elems) ||
        !ws.x.ensure(xnat_elems + xkm_elems) || !ws.y.ensure(ynat_elems + ykm_elems) ||
        !ws.z.ensure(z_elems)) {
        return QCD_STATUS_MEMORY;
    }
    c128* pa = ws.pa.ptr;
    c128* pcp = ws.pcp.ptr;
    c128* pbb = ws.pbb.ptr;
    c128* xnat = ws.x.ptr;
    c128* xkm = xnat + xnat_elems;
    c128* ynat = ws.y.ptr;
    c128* ykm = ynat + ynat_elems;
    c128* z = ws.z.ptr;

    double t[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double t_all0 = now_seconds();
    double t0 = t_all0;

    build_core(setup.p0, setup.p1, setup.p2, setup.g20, setup.g01, setup.rf, n, threads, pa, pcp,
               pbb);
    t[0] += now_seconds() - t0;

    for (int64_t m1 = 0; m1 < nmom; ++m1) {
        for (int64_t m2i = 0; m2i < (desc.tied_momentum ? 1 : nmom); ++m2i) {
            const int64_t m2 = desc.tied_momentum ? m1 : m2i;
            const c128* v1m = setup.v1.base + m1 * setup.v1.stride_m;
            const c128* v2m = setup.v2.base + m2 * setup.v2.stride_m;

            t0 = now_seconds();
            // ---- s1: X 自然布局 [u2][k][a1,b1][c2] ----
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t idx = 0; idx < static_cast<int64_t>(kSpin) * r; ++idx) {
                const int u2 = static_cast<int>(idx / r);
                const int k = static_cast<int>(idx) % r;
                // PCp 的 (c1,c2) 子块：c1 行 stride = block
                const c128* bmat = pcp + (static_cast<int64_t>(k) * n) * block + u2 * n;
                c128* cmat = xnat + idx * n * n * n;
                if (v1_trans) {
                    for (int64_t b1 = 0; b1 < n; ++b1) {
                        const c128* amat = v1m + b1 * setup.v1.stride_leg[1];
                        gemm(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), amat,
                             static_cast<int>(v1_ld), bmat, static_cast<int>(block),
                             cmat + b1 * n, static_cast<int>(n * n), true, false, false);
                    }
                } else {
                    // 行是 (a1,b1)，a1 慢：E1 源布局里 a1 stride = n²、b1 stride = n，
                    // 所以 lda 取 b1 的 stride（= n）时行距正好等于 n。
                    gemm(static_cast<int>(n * n), static_cast<int>(n), static_cast<int>(n), v1m,
                         static_cast<int>(setup.v1.stride_leg[1]), bmat,
                         static_cast<int>(block), cmat, static_cast<int>(n), false, false, false);
                }
            }
            t[1] += now_seconds() - t0;
            t0 = now_seconds();

            // ---- s2: Y 自然布局 [u2][v0][a1][b2,c2] ----
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t idx = 0; idx < static_cast<int64_t>(kSpin) * kSpin; ++idx) {
                const int u2 = static_cast<int>(idx) / kSpin;
                const int v0 = static_cast<int>(idx) % kSpin;
                const c128* amat =
                    pa + static_cast<int64_t>(v0) * kSpin * n * n + static_cast<int64_t>(u2) * n;
                c128* cmat = ynat + idx * n * n * n;
                for (int64_t b2 = 0; b2 < n; ++b2) {
                    const c128* bmat = v2m + b2 * setup.v2.stride_leg[1];
                    gemm(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n), amat,
                         static_cast<int>(block), bmat, static_cast<int>(v2_ld),
                         cmat + b2 * n, static_cast<int>(n * n), false, v2_trans, false);
                }
            }
            t[2] += now_seconds() - t0;
            t0 = now_seconds();

            // ---- s3: pack X -> [u2,a1,c2][k,b1] ----
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t row = 0; row < static_cast<int64_t>(kSpin) * n * n; ++row) {
                const int u2 = static_cast<int>(row / (n * n));
                const int64_t rem = row % (n * n);
                const int64_t a1 = rem / n;
                const int64_t c2 = rem % n;
                c128* dst = xkm + row * (r * n);
                for (int k = 0; k < r; ++k) {
                    const c128* src = xnat + (static_cast<int64_t>(u2) * r + k) * n * n * n +
                                      a1 * n * n + c2;
                    for (int64_t b1 = 0; b1 < n; ++b1) {
                        dst[k * n + b1] = src[b1 * n];
                    }
                }
            }
            t[3] += now_seconds() - t0;
            t0 = now_seconds();

            // ---- s4: pack Y -> [u2,a1,c2][v0,b2] ----
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t row = 0; row < static_cast<int64_t>(kSpin) * n * n; ++row) {
                const int u2 = static_cast<int>(row / (n * n));
                const int64_t rem = row % (n * n);
                const int64_t a1 = rem / n;
                const int64_t c2 = rem % n;
                c128* dst = ykm + row * block;
                for (int v0 = 0; v0 < kSpin; ++v0) {
                    const c128* src = ynat + (static_cast<int64_t>(u2) * kSpin + v0) * n * n * n +
                                      a1 * n * n + c2;
                    for (int64_t b2 = 0; b2 < n; ++b2) {
                        // 列序必须与 fused 路径一致：(b2,v0)，见 contract_final
                        dst[b2 * kSpin + v0] = src[b2 * n];
                    }
                }
            }
            t[4] += now_seconds() - t0;
            t0 = now_seconds();

            // ---- s5: 单个大 GEMM Z = Xkm^T * Ykm ----
            gemm(static_cast<int>(r * n), static_cast<int>(block),
                 static_cast<int>(kSpin * n * n), xkm, static_cast<int>(r * n), ykm,
                 static_cast<int>(block), z, static_cast<int>(block), true, false, false);
            t[5] += now_seconds() - t0;
            t0 = now_seconds();

            // ---- s6: 与 PBB 收缩 ----
            const c128 result = contract_final(z, pbb, n, r, threads);
            if (desc.out_ndim == 2) {
                out[m2 * nmom + m1] = result;
            } else {
                out[m1] = result;
            }
            t[6] += now_seconds() - t0;
        }
    }

    if (profile_enabled()) {
        const double elapsed = now_seconds() - t_all0;
        std::fprintf(stderr,
                     "[kupl-n2pt:steps] n=%lld nmom=%lld r=%d threads=%d total=%.3f s | "
                     "s0_core=%.3f s1_Xnat=%.3f s2_Ynat=%.3f s3_packX=%.3f s4_packY=%.3f "
                     "s5_join=%.3f s6_final=%.3f\n",
                     static_cast<long long>(n), static_cast<long long>(nmom), r, threads, elapsed,
                     t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
    }
    return QCD_STATUS_OK;
}

bool env_is(const char* name, const char* value) {
    const char* env = std::getenv(name);
    return env != nullptr && std::strcmp(env, value) == 0;
}

}  // namespace

extern "C" QCD_EINSUM_API int qcd_kupl_nucleon_2pt(int dtype_id, const void* const* inputs,
                                                  void* output, const int64_t* dims, int variant) {
    if (dtype_id != QCD_DTYPE_COMPLEX128) {
        return QCD_STATUS_INVALID;  // complex64 等明确不支持
    }
    if (inputs == nullptr || output == nullptr || dims == nullptr) {
        return QCD_STATUS_INVALID;
    }
    if (variant != 0 && variant != 1) {
        return QCD_STATUS_INVALID;
    }
    for (int i = 0; i < 8; ++i) {
        if (inputs[i] == nullptr) {
            return QCD_STATUS_INVALID;
        }
    }
    const int64_t nvec = dims[0];
    const int64_t nmom = dims[1];
    const int64_t nspin = dims[2];
    if (nvec < 1 || nmom < 1 || nspin != kSpin) {
        return QCD_STATUS_INVALID;
    }
    if (nvec > INT32_MAX || nmom > INT32_MAX) {
        return QCD_STATUS_INVALID;
    }
    for (int i = 3; i < 8; ++i) {
        if (dims[i] != 0) {
            return QCD_STATUS_INVALID;  // 保留项必须为 0
        }
    }
    const c128* in[8];
    for (int i = 0; i < 8; ++i) {
        in[i] = static_cast<const c128*>(inputs[i]);
    }
    c128* out = static_cast<c128*>(output);
    const VariantDesc& desc = (variant == 0) ? kE1 : kS1;
    try {
        return env_is("PYQCD_KUPL_N2PT_MODE", "steps")
                   ? run_steps(in, out, nvec, nmom, desc)
                   : run_fused(in, out, nvec, nmom, desc);
    } catch (const std::bad_alloc&) {
        return QCD_STATUS_MEMORY;
    } catch (...) {
        return QCD_STATUS_INTERNAL;
    }
}
