#ifndef PYQCD_EINSUM_CUDA_PATTERN_3PT_H
#define PYQCD_EINSUM_CUDA_PATTERN_3PT_H

#include "einsum_define.h"

/*
 * Pattern-specific CUDA kernel for the sush distillation 3pt main term.
 *
 *   variant 0 (P1):
 *     agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy
 *   variant 1 (P2):
 *     agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy
 *
 * Both variants share the same operand shapes and the same eigen-space
 * network; they differ only in which Dirac leg of the fourth/fifth
 * perambulator slice carries which gamma structure (see the stage tables in
 * pattern_3pt.cu).
 *
 * ---------------------------------------------------------------------------
 * Operand order (``inputs[0] .. inputs[14]``) -- exactly the einsum order
 * ---------------------------------------------------------------------------
 *   [ 0] perambulator slice  agbh  [spin][spin][nev][nev]   (a = projected, g = gamma5)
 *   [ 1] perambulator slice  eqfr  [spin][spin][nev][nev]   (e = gamma7_ce, q = gamma7_kq)
 *   [ 2] perambulator slice  ospt  [spin][spin][nev][nev]   (o = gamma_curr, s = projected)
 *   [ 3] perambulator slice  cmdn  (P1) / ckdl (P2)         [spin][spin][nev][nev]
 *   [ 4] perambulator slice  ikjl  (P1) / imjn (P2)         [spin][spin][nev][nev]
 *   [ 5] gamma7   ce   [spin][spin]
 *   [ 6] gamma_curr Gmo [spin][spin][spin]                  (first axis reaches the output)
 *   [ 7] gamma5   gi   [spin][spin]
 *   [ 8] gamma7   kq   [spin][spin]
 *   [ 9] sink_VVV   Nbdf [mom][nev][nev][nev]
 *   [10] VdV_link   NLnp [mom][link][nev][nev]
 *   [11] source_VdV Mhj  [mom][nev][nev]
 *   [12] source_VVV Mlrt [mom][nev][nev][nev]
 *   [13] projection ax  [spin][proj]
 *   [14] projection sy  [spin][proj]
 *
 * All operands are row-major (C order) and contiguous, complex128.
 *
 * ---------------------------------------------------------------------------
 * Output
 * ---------------------------------------------------------------------------
 *   MNGLxy, row-major with ``y`` fastest:
 *     [mom][mom][spin][link][proj][proj]
 *   i.e. out[(M*mom + N)*spin*link*proj*proj + ((G*link + L)*proj + x)*proj + y]
 *
 * ---------------------------------------------------------------------------
 * dims (exactly 5 entries, all read as long long)
 * ---------------------------------------------------------------------------
 *   dims[0] nev   eigenvector extent: the peram eig legs and every nev axis of
 *                 sink_VVV / VdV_link / source_VdV / source_VVV
 *   dims[1] mom   momentum extent: axis 0 of sink_VVV, VdV_link, source_VdV,
 *                 source_VVV, and both output momentum axes
 *   dims[2] link  link extent: axis 1 of VdV_link and the ``L`` output axis
 *   dims[3] spin  Dirac extent, must be 4
 *   dims[4] proj  projector extent, must be 2
 *
 * The five numbers fully determine every operand shape, so the implementation
 * validates nothing else about the caller's buffers.
 *
 * ---------------------------------------------------------------------------
 * Supported dtypes
 * ---------------------------------------------------------------------------
 *   QCD_DTYPE_COMPLEX128   fully supported
 *   anything else          QCD_STATUS_INVALID (complex64 is intentionally not
 *                          claimed: the 1e-12 relative-accuracy contract of
 *                          this pattern only holds in double precision)
 *
 * ---------------------------------------------------------------------------
 * Execution model
 * ---------------------------------------------------------------------------
 *   Single device, single stream: only ``stream`` is used, no other stream is
 *   created, and the intermediates live in one reusable device arena owned by
 *   this translation unit (grown on demand, freed by qcd_cuda_pattern_3pt_free).
 */

#if defined(_WIN32)
#define QCD_PATTERN_3PT_API __declspec(dllexport)
#else
#define QCD_PATTERN_3PT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Evaluate the 15-operand 3pt main term.
 *
 *   dtype_id  QCD_DTYPE_COMPLEX128
 *   inputs    15 device pointers, order above
 *   output    device pointer to MNGLxy
 *   dims      5 entries, order above
 *   variant   0 -> P1, 1 -> P2
 *   stream    cudaStream_t (nullptr means the default stream)
 *
 * Returns QCD_STATUS_OK, QCD_STATUS_INVALID (bad descriptor/pointers/dtype) or
 * QCD_STATUS_MEMORY (workspace allocation failed) or QCD_STATUS_INTERNAL
 * (a kernel launch failed).
 */
QCD_PATTERN_3PT_API int qcd_cuda_pattern_3pt(
    int dtype_id,
    const void* const* inputs,
    void* output,
    const long long* dims,
    int variant,
    void* stream);

/*
 * Release the cached device workspace.  Optional: the arena is otherwise kept
 * for the next call (steady-state calls do not call cudaMalloc).
 */
QCD_PATTERN_3PT_API void qcd_cuda_pattern_3pt_free(void);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_EINSUM_CUDA_PATTERN_3PT_H */
