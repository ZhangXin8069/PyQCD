/* PyQCD — nucleon 2pt 收缩的 CUDA 专用融合入口（nucleon2pt.h）
 *
 * 本文件只声明**一个**自包含 C 入口，实现位于
 * `cpp/cuda/einsum/src/nucleon2pt.cu`。它覆盖两条真实生产方程
 * （`pyqcd/einsum/_specialize.py` 的 E1 与 S1），两者在张量网络层是
 * 同一个收缩，只是轴序/算子顺序不同：
 *
 *   E1  gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M
 *   S1  eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN
 *
 * 规范方程（本入口实现的唯一形态）
 * ------------------------------------------------------------------
 *   out[my][mx] = Σ A[g,a,h,b] B[m,c,n,d] C[o,e,p,f] GX[a,c] GY[o,g]
 *                   PR[e,m] V1[mx,b,d,f] V2[my,n,p,h]
 *
 * 自由指标只有两个动量指标（V1 的 mx、V2 的 my）；模式 0 下二者取同一个
 * 指标（对角，输出长度 mx），模式 1 下输出 (my,mx) 矩阵。
 *
 * 归约结构（FLOP 最优，等于 opt_einsum `optimal`/`dp` 路径的 opt_cost）
 * ------------------------------------------------------------------
 *   U1T[c,g,b,h]      = Σ_a  GX[a,c] A[g,a,h,b]
 *   U2[g,m,p,f]       = Σ_o,e GY[o,g] PR[e,m] C[o,e,p,f]
 *   Bbuf[c,m,f,n,b]   = Σ_d  B[m,c,n,d] V1[mx,b,d,f]
 *   Abuf[c,g,p,n,b]   = Σ_h  V2[my,n,p,h] U1T[c,g,b,h]
 *   Cout[g,p,m,f]     = Σ_c,n,b Abuf[c,g,p,n,b] Bbuf[c,m,f,n,b]
 *   out[my][mx]       = Σ_g,m,p,f U2[g,m,p,f] Cout[g,p,m,f]
 * 乘加次数 = 16N⁴ + 16N⁴ + 64N⁴ = 96N⁴（N=100 时 9.6e9），与
 * `opt_einsum.contract_path(..., optimize='optimal').opt_cost / 2` 一致。
 *
 * 规范轴序（canonical roles）——`inputs[]` 必须按下面的顺序与轴序给出
 * ------------------------------------------------------------------
 *   idx  operand  规范轴序（canonical）                E1 里对应
 *   ---  -------  -----------------------------------  -------------------
 *    0   A       (g, a, h, b)   4×4×N×N                gahb
 *           轴角色: g=GY 自旋, a=GX 自旋, h=V2 的 N, b=V1 的 N
 *    1   B       (m, c, n, d)   4×4×N×N                mcnd
 *           轴角色: m=PR 自旋, c=GX 自旋, n=V2 的 N, d=V1 的 N
 *    2   C       (o, e, p, f)   4×4×N×N                oepf
 *           轴角色: o=GY 自旋, e=PR 自旋, p=V2 的 N, f=V1 的 N
 *    3   GX      (a, c)         4×4                    ac
 *    4   GY      (o, g)         4×4                    og
 *    5   PR      (e, m)         4×4                    em
 *    6   V1      (mx, b, d, f)  mx×N×N×N                M bdf
 *           轴角色: b=A 的 V1-N, d=B 的 V1-N, f=C 的 V1-N
 *    7   V2      (my, n, p, h)  my×N×N×N                M nph
 *           轴角色: n=B 的 V2-N, p=C 的 V2-N, h=A 的 V2-N
 *
 * 允许的输入变体（`dims[]` 的 perm 码）
 * ------------------------------------------------------------------
 * 生产脚本里 A/B/C/V1/V2 的轴序与上表并不总是一致（见下面 S1 的例子），
 * 因此 dims[] 里给了 5 个"源轴序"码，把**规范轴 i 对应输入的哪一根轴**
 * 写清楚；码是 4 个十进制数字 (p0,p1,p2,p3)，值 = p0*1000+p1*100+p2*10+p3，
 * 恒等轴序写作 123：
 *
 *   permA / permB / permC   三根张量的 (4,4,N,N) 轴序
 *   permV1 / permV2         三根量的 (mom,N,N,N) 轴序；数字 p0 必须是 0
 *                           （动量轴固定在输入的第 0 维）
 *
 * 矩阵类输入（GX/GY/PR）只允许转置，用 dims[9] 的 3 个 bit 表示
 * （bit0=GX 转置, bit1=GY 转置, bit2=PR 转置）。
 *
 * dims[] 约定（长度 QCD_NUCLEON2PT_DIMS = 10，全部是 long long）
 * ------------------------------------------------------------------
 *   dims[0] = N       每根 (4,4,N,N) 张量的 N；必须 ≥ 1
 *   dims[1] = mx      V1 第 0 维长度（V1 的动量个数）
 *   dims[2] = my      V2 第 0 维长度（V2 的动量个数）
 *   dims[3] = mode    0 = 对角（要求 mx == my，用 V1[k] 配 V2[k]，输出长 mx）
 *                     1 = 外积（输出 (my,mx) 行主序，第 M 维=my 在外）
 *   dims[4] = permA   A 的源轴序码（恒等 = 123）
 *   dims[5] = permB   B 的源轴序码
 *   dims[6] = permC   C 的源轴序码
 *   dims[7] = permV1  V1 的源轴序码（p0 = 0）
 *   dims[8] = permV2  V2 的源轴序码（p0 = 0）
 *   dims[9] = matflags  bit0 GX 转置 / bit1 GY 转置 / bit2 PR 转置
 *
 * output 约定
 * ------------------------------------------------------------------
 *   mode 0: 长度 mx 的连续数组
 *   mode 1: 长度 my*mx 的连续数组，out[my_i*mx + mx_i]
 *
 * dtype_id（与 einsum_define.h 一致）
 * ------------------------------------------------------------------
 *   0 = complex128（主路径，FP64 累加）
 *   1 = complex64
 *   其余值返回 QCD_STATUS_INVALID。
 *
 * 返回值（与 einsum_define.h 一致）
 * ------------------------------------------------------------------
 *   0 = QCD_STATUS_OK
 *   1 = QCD_STATUS_INVALID   参数/dim 非法、dtype 不支持、perm 码非法
 *   2 = QCD_STATUS_MEMORY    workspace 申请失败或超出上限
 *   3 = QCD_STATUS_INTERNAL   CUDA 运行期错误
 * 任何错误都不会静默返回成功，也不会写半个输出（除 CUDA 运行期错误）。
 *
 * 例子
 * ------------------------------------------------------------------
 *  E1（`gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M`，N=100, M=1）
 *      inputs = {P1, P2, P3, G1, G2, pr, V1, V2}   （顺序即规范顺序）
 *      dims   = {100, 1, 1, 0, 123,123,123,123,123, 0}
 *      output = 长度 1 的复杂标量
 *
 *  S1（`eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN`，N=100，两动量各 1）
 *      inputs = {t3(cgdh), t1(eofp), t2(ambn), ce, ga, mo, Nbdf, Mnph}
 *      dims   = {100, 1, 1, 1, 1032, 1032, 132, 231, 213, 2}
 *      即 permA=1032（A=t3 的轴序 g,a,h,b ← 输入 (c,g,d,h) 的 1,0,3,2）
 *         permB=1032（B=t1 的轴序 ← 输入 (e,o,f,p) 的 1,0,3,2）
 *         permC=132 （C=t2 的轴序 ← 输入 (a,m,b,n) 的 0,1,3,2）
 *         permV1=231（V1=Nbdf ← 规范 (mom,A,B,C) = 源 0,2,3,1）
 *         permV2=213（V2=Mnph ← 规范 (mom,B,C,A) = 源 0,2,1,3）
 *         matflags=2（ga 需要转置成 (a,g)）
 *      output = 长度 1 的复杂标量（形状 (my,mx) = (1,1)）
 *
 * 线程/并发
 * ------------------------------------------------------------------
 * 设备 workspace 以静态缓存复用（与 `compile`d-context 同一思路），
 * 因此**同一设备上的并发调用必须由调用方串行化**（单流单调用是安全的）。
 * 流由 `stream` 参数给出（可传 0 = legacy default stream）。
 */

#ifndef PYQCD_NUCLEON2PT_H
#define PYQCD_NUCLEON2PT_H

#include <stdint.h>

#if defined(__has_include)
#  if __has_include("einsum_define.h")
#    include "einsum_define.h"
#    define QCD_NUCLEON2PT_HAVE_DTYPE_ENUM 1
#  endif
#endif

#ifndef QCD_NUCLEON2PT_HAVE_DTYPE_ENUM
/* 与 cpp/kupl/einsum/include/einsum_define.h 的冻结 ABI 对齐的本地镜像，
 * 仅在编译时找不到 einsum_define.h 的情况下使用。 */
enum {
    QCD_STATUS_OK = 0,
    QCD_STATUS_INVALID = 1,
    QCD_STATUS_MEMORY = 2,
    QCD_STATUS_INTERNAL = 3
};
enum {
    QCD_DTYPE_COMPLEX128 = 0,
    QCD_DTYPE_COMPLEX64 = 1,
    QCD_DTYPE_FLOAT64 = 2,
    QCD_DTYPE_FLOAT32 = 3
};
#endif

#if defined(_WIN32)
#  define QCD_NUCLEON2PT_API __declspec(dllexport)
#else
#  define QCD_NUCLEON2PT_API __attribute__((visibility("default")))
#endif

#define QCD_NUCLEON2PT_DIMS 10
#define QCD_NUCLEON2PT_INPUTS 8
#define QCD_NUCLEON2PT_PERM_IDENTITY 123

#ifdef __cplusplus
extern "C" {
#endif

/* dtype_id: 0 = complex128, 1 = complex64
 * inputs   : 长度 8 的设备指针数组，顺序见文件头
 * output   : 设备输出指针
 * dims     : 长度 10 的 long long 数组，见文件头
 * stream   : cudaStream_t（以 void* 传入，0 表示 legacy default stream）
 */
QCD_NUCLEON2PT_API int qcd_cuda_nucleon_2pt(
    int dtype_id,
    const void* const* inputs,
    void* output,
    const long long* dims,
    void* stream);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_NUCLEON2PT_H */
