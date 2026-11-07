#ifndef PYQCD_EINSUM_PATTERN_3PT_H
#define PYQCD_EINSUM_PATTERN_3PT_H

/* 格点 QCD 3pt 主项专用 CPU（KUPL）执行 kernel。
 *
 * 对应真实生产热点（sush distillation 3pt，15 个操作数、14 步收缩）：
 *
 *   P1: agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy
 *   P2: agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy
 *
 * 本文件只声明 ABI；接线（识别方程 -> 调用）由执行层完成，本 kernel 不参与
 * 方程匹配，也不改变 `applyContractQcd` 的语义。
 *
 * 约定：
 *   - 只支持 complex128（`QCD_DTYPE_COMPLEX128`）；complex64 / 实数返回非 0。
 *   - 所有操作数与输出必须是 **C 连续**（行主序），不支持步长视图。
 *   - `dims` 给出逻辑尺寸，kernel 按下面的固定槽位读取（`dims[0..5]` 必填，
 *     其余槽位不读）：
 *       dims[0] = nvec            （颜色/特征向量维 N）
 *       dims[1] = mom_source      （M，source 端动量分量数，>= 1）
 *       dims[2] = mom_sink        （N，sink   端动量分量数，>= 1）
 *       dims[3] = link            （L，插入算符的链接数，>= 1）
 *       dims[4] = dirac           （必须等于 4）
 *       dims[5] = proj            （必须等于 2，投影 `(4,2)` 的列数）
 *   - `inputs` 固定 15 槽，顺序与方程算子顺序完全一致（见
 *     `QCD_3PT_IN_*`）。每个槽的期望形状见下表（n=nvec, M=mom_source,
 *     N=mom_sink, L=link）；
 *
 *     槽  变体 下标   形状        说明
 *     --- ----- ---- ------------- -------------------------------------------
 *      0  P1/P2 agbh (4,4,n,n)     peram 切片 0
 *      1  P1/P2 eqfr (4,4,n,n)     peram 切片 1（中间 peram）
 *      2  P1/P2 ospt (4,4,n,n)     peram 切片 2（sink）
 *      3  P1    cmdn (4,4,n,n)     peram 切片 3
 *      3  P2    ckdl (4,4,n,n)     peram 切片 3（置换型）
 *      4  P1    ikjl (4,4,n,n)     peram 切片 4
 *      4  P2    imjn (4,4,n,n)     peram 切片 4（置换型）
 *      5  P1/P2 ce   (4,4)         gamma(7)
 *      6  P1/P2 Gmo  (4,4,4)       gamma_curr
 *      7  P1/P2 gi   (4,4)         gamma(5)
 *      8  P1/P2 kq   (4,4)         gamma(7)
 *      9  P1/P2 Nbdf (N,n,n,n)     sink VVV
 *     10  P1/P2 NLnp (N,L,n,n)     VdV link
 *     11  P1/P2 Mhj  (M,n,n)       source VdV
 *     12  P1/P2 Mlrt (M,n,n,n)     source VVV
 *     13  P1/P2 ax   (4,2)         投影（P0 的 Dirac 腿）
 *     14  P1/P2 sy   (4,2)         投影（P2 的 Dirac 腿）
 *
 *   - `output` 形状固定为 `(M, N, 4, L, 2, 2)`，C 连续。
 *   - `variant`：0 = P1，1 = P2，其它值返回 `QCD_STATUS_INVALID`。
 *   - 线程口径：GEMM 交给进程内已加载的 CBLAS（OpenBLAS）实现，线程数由
 *     `OPENBLAS_NUM_THREADS`（或 OpenBLAS 自身的默认）决定；本 kernel 自己的
 *     OpenMP 循环（置换/小收缩）只受 `OMP_NUM_THREADS` 约束。两者互不嵌套，
 *     因此不会出现 OMP 线程 × BLAS 线程的乘性超订。
 *   - 失败语义 fail-closed：任何校验失败都在写 `output` 之前返回非 0。
 */

#include <stdint.h>

#include "einsum_define.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QCD_PATTERN_3PT_P1 = 0, /* agbh,eqfr,ospt,cmdn,ikjl,... */
    QCD_PATTERN_3PT_P2 = 1  /* agbh,eqfr,ospt,ckdl,imjn,... */
};

enum {
    QCD_3PT_DIM_NVEC = 0,
    QCD_3PT_DIM_MOM_SOURCE = 1,
    QCD_3PT_DIM_MOM_SINK = 2,
    QCD_3PT_DIM_LINK = 3,
    QCD_3PT_DIM_DIRAC = 4,
    QCD_3PT_DIM_PROJ = 5,
    QCD_3PT_DIM_COUNT = 6
};

enum {
    QCD_3PT_IN_PERAM0 = 0,
    QCD_3PT_IN_PERAM1 = 1,
    QCD_3PT_IN_PERAM2 = 2,
    QCD_3PT_IN_PERAM3 = 3,
    QCD_3PT_IN_PERAM4 = 4,
    QCD_3PT_IN_GAMMA7A = 5,
    QCD_3PT_IN_GAMMA_CURR = 6,
    QCD_3PT_IN_GAMMA5 = 7,
    QCD_3PT_IN_GAMMA7B = 8,
    QCD_3PT_IN_SINK_VVV = 9,
    QCD_3PT_IN_VDV_LINK = 10,
    QCD_3PT_IN_SOURCE_VDV = 11,
    QCD_3PT_IN_SOURCE_VVV = 12,
    QCD_3PT_IN_PROJ_SOURCE = 13,
    QCD_3PT_IN_PROJ_SINK = 14,
    QCD_3PT_INPUT_COUNT = 15
};

/* 执行 P1/P2 的专用 kernel。
 * 返回 0 成功；非 0 见 einsum_define.h 的状态码（1=非法参数/不支持，
 * 2=内存不足，3=内部错误，例如进程内找不到可用的 CBLAS）。 */
QCD_EINSUM_API int qcd_kupl_pattern_3pt(int dtype_id, const void* const* inputs, void* output,
                                        const int64_t* dims, int variant);

/* 与 qcd_kupl_pattern_3pt 相同的入参校验，只做判定不执行；供接线方在调用前
 * 预筛（例如把不支持的形状留给通用执行层）。返回 0 表示可执行。 */
QCD_EINSUM_API int qcd_kupl_pattern_3pt_supported(int dtype_id, const int64_t* dims, int variant);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_EINSUM_PATTERN_3PT_H */
