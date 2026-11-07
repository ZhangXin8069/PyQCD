#ifndef PYQCD_KUPL_NUCLEON2PT_H
#define PYQCD_KUPL_NUCLEON2PT_H

#include "einsum_define.h"

#include <stdint.h>

/*
 * KUPL(CPU) 格点 QCD 2pt 收缩专用 kernel（E1 / S1）。
 *
 * 覆盖 pyqcd/testing/einsum/bench_distillation.py 的两条生产方程：
 *
 *   variant = 0  E1  gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M
 *   variant = 1  S1  eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN
 *
 * 两条方程是同一个张量网络的两种下标命名：三个 peram 切片（每个带
 * 1 对 spinor 下标 + 1 条连到 VVV#1 的矢量腿 + 1 条连到 VVV#2 的矢量腿）、
 * 两个 VVV 三点张量、三个 4x4 gamma/投影矩阵（两两配对三个 peram 的 spinor
 * 下标，构成三角形）。kernel 内部按统一范式（canonical form）计算，见
 * cpp/kupl/einsum/src/nucleon2pt.cpp 顶部注释。
 *
 * ---------------------------------------------------------------------------
 * 入口
 * ---------------------------------------------------------------------------
 *   int qcd_kupl_nucleon_2pt(int dtype_id, const void* const* inputs,
 *                            void* output, const int64_t* dims, int variant);
 *
 * 返回值：0 成功；1 参数非法/ dtype 不支持（fail closed）；2 内存不足；
 *         3 其它内部错误。
 * 任何非 0 返回都不写 output。
 *
 * ---------------------------------------------------------------------------
 * dtype_id（见 einsum_define.h）
 * ---------------------------------------------------------------------------
 *   0 = QCD_DTYPE_COMPLEX128 : 支持（唯一支持的 dtype）
 *   1 = QCD_DTYPE_COMPLEX64  : **明确不支持**，立即返回 1
 *   其它                      : 返回 1
 *
 * ---------------------------------------------------------------------------
 * dims 约定（8 项定长，按顺序逐项说明；本 kernel 全部读取，越界即返回 1）
 * ---------------------------------------------------------------------------
 *   dims[0] = nvec   : 矢量空间维度 N（peram 的矢量腿与 VVV 的后三维都是它），>= 1
 *   dims[1] = nmom   : 动量指标个数 M（两个 VVV 的首轴都是它），>= 1
 *   dims[2] = nspin  : Dirac spinor 维度，**必须等于 4**（gamma 为 4x4）
 *   dims[3]          : 保留，必须为 0
 *   dims[4]          : 保留，必须为 0
 *   dims[5]          : 保留，必须为 0
 *   dims[6]          : 保留，必须为 0
 *   dims[7]          : 保留，必须为 0
 *
 * inputs/output 均为复数数组指针（complex128 = 16 字节的 {re,im}），
 * 全部按 C 行主序连续存储（即 numpy 的 C order，无 stride 参数）。
 *
 * ---------------------------------------------------------------------------
 * variant = 0：E1  gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M
 * ---------------------------------------------------------------------------
 *   inputs[0]  (nspin,nspin,nvec,nvec)  gahb  peram 切片 1
 *   inputs[1]  (nspin,nspin,nvec,nvec)  mcnd  peram 切片 2
 *   inputs[2]  (nspin,nspin,nvec,nvec)  oepf  peram 切片 3
 *   inputs[3]  (nspin,nspin)            ac    gamma（配对 op0.a - op1.c）
 *   inputs[4]  (nspin,nspin)            og    gamma（配对 op2.o - op0.g）
 *   inputs[5]  (nmom,nvec,nvec,nvec)    Mbdf  VVV#1
 *   inputs[6]  (nmom,nvec,nvec,nvec)    Mnph  VVV#2
 *   inputs[7]  (nspin,nspin)            em    投影/gamma（配对 op2.e - op1.m）
 *   output     (nmom)                   M：两个 VVV 共用同一个动量下标 M，
 *                                       即 out[M] 由 inputs[5][M] 与 inputs[6][M]
 *                                       同时取第 M 片构成
 *
 * ---------------------------------------------------------------------------
 * variant = 1：S1  eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN
 * ---------------------------------------------------------------------------
 *   inputs[0]  (nspin,nspin,nvec,nvec)  eofp  peram 切片 1
 *   inputs[1]  (nspin,nspin,nvec,nvec)  ambn  peram 切片 2
 *   inputs[2]  (nspin,nspin,nvec,nvec)  cgdh  peram 切片 3
 *   inputs[3]  (nspin,nspin)            ce    gamma（配对 op2.c - op0.e）
 *   inputs[4]  (nspin,nspin)            mo    gamma（配对 op1.m - op0.o）
 *   inputs[5]  (nmom,nvec,nvec,nvec)    Nbdf  VVV#1
 *   inputs[6]  (nmom,nvec,nvec,nvec)    Mnph  VVV#2
 *   inputs[7]  (nspin,nspin)            ga    投影/gamma（配对 op2.g - op1.a）
 *   output     (nmom,nmom)              MN：M 取 inputs[6] 的动量轴、N 取
 *                                       inputs[5] 的动量轴，按 C 序
 *                                       out[M*nmom + N] 存放（与 numpy
 *                                       `...->MN` 的形状 (nmom,nmom) 一致）
 *
 * 两个 variant 的 inputs[i] 顺序都与方程里的操作数顺序逐位对应，
 * 允许指针别名（例如 E1 的真实用法里三个 peram 切片是同一块内存）。
 *
 * ---------------------------------------------------------------------------
 * 线程口径
 * ---------------------------------------------------------------------------
 * 只使用调用方给定的 OMP_NUM_THREADS 与 OPENBLAS_NUM_THREADS：
 *   - kernel 内部并行只发生在"独立 GEMM 批"这一层，线程数取
 *     omp_get_max_threads()（即 OMP_NUM_THREADS），不会超过它；
 *   - 进入内部并行批之前把 CBLAS 的线程数压到 1（退出时恢复原值），
 *     因此总并发 = min(OMP_NUM_THREADS, ...)，不会出现 OMP×BLAS 超订；
 *   - 若 CBLAS 不提供 openblas_{get,set}_num_threads，则退化为串行调用
 *     GEMM（此时并发由 OPENBLAS_NUM_THREADS 决定），同样不超订。
 *
 * ---------------------------------------------------------------------------
 * 可选调试环境变量（仅在诊断时使用，不影响数值）
 * ---------------------------------------------------------------------------
 *   PYQCD_KUPL_N2PT_MODE=steps|fused   steps = 显式物化 + pack 的 7 步参考路径
 *   PYQCD_KUPL_N2PT_PROFILE=1          逐阶段耗时打印到 stderr
 *   PYQCD_KUPL_N2PT_BUDGET_MB=<int>    中间缓冲预算（默认 512）
 */

#ifdef __cplusplus
extern "C" {
#endif

QCD_EINSUM_API int qcd_kupl_nucleon_2pt(int dtype_id, const void* const* inputs, void* output,
                                       const int64_t* dims, int variant);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_KUPL_NUCLEON2PT_H */
