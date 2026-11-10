#ifndef PYQCD_CPU_EINSUMS_H
#define PYQCD_CPU_EINSUMS_H

#include <stdint.h>

#if defined(_WIN32)
#define PYQCEINSUMS_API __declspec(dllexport)
#else
#define PYQCEINSUMS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Einsums string `einsum` bridge ABI, version 1.
 *
 * Layout is a byte-for-byte mirror of the MatX bridge
 * (`cpp/cuda/matx/include/matx_einsum.h`): pointers and counts travel as flat
 * integer arrays so Python can marshal them with ctypes without any struct
 * packing agreement. The only difference is the address space: every pointer
 * carried by `set_ptrs` is a CPU host pointer, not a device pointer.
 *
 * `set_ptrs` is int64_t[PYQCEINSUMS_SET_SIZE]:
 *   [OUTPUT]        host pointer of the output tensor
 *   [INPUTS]        host pointer to int64_t[nops] of operand host pointers
 *   [INPUT_SHAPES]  host pointer to int64_t[nops * INPUT_RANK], row-major
 *   [OUTPUT_SHAPE]  host pointer to int64_t[OUT_NDIM]
 *
 * `params` is int32_t[PYQCEINSUMS_PARAM_SIZE]:
 *   [NUM_OPS], [INPUT_RANK], [DTYPE_ID], [OUT_NDIM]
 *
 * `spec` is passed to Einsums verbatim: `"out <- in1 ; in2"`, output first,
 * inputs separated by semicolons -- the Einsums ComputeGraph spelling of
 * numpy's `"in1,in2->out"`. The bridge does not translate between the two.
 *
 * The bridge is deliberately bounded: it accepts exactly two operands with
 * rank 1..4 and an output rank of 1..4, and both operands must share a rank.
 * Einsums' `compute_graph::einsum` has only two-operand and three-or-more
 * n-ary overloads, so a unary transpose/reduction must be written with
 * `cg::permute`; rank-0 tensors are not valid einsum operands or outputs in
 * this version (Einsums' own EagerParityGaps test records that limitation), so
 * scalar inner products go through `cg::dot`. Everything outside the envelope
 * is rejected with PYQCEINSUMS_STATUS_INVALID before any Einsums call,
 * matching the fail-closed rule of the einsum ABI.
 */

enum {
    PYQCEINSUMS_STATUS_OK = 0,
    PYQCEINSUMS_STATUS_INVALID = 1,
    PYQCEINSUMS_STATUS_MEMORY = 2,
    PYQCEINSUMS_STATUS_INTERNAL = 3,
    PYQCEINSUMS_STATUS_UNAVAILABLE = 4
};

enum {
    PYQCEINSUMS_DTYPE_COMPLEX128 = 0,
    PYQCEINSUMS_DTYPE_COMPLEX64 = 1,
    PYQCEINSUMS_DTYPE_FLOAT64 = 2,
    PYQCEINSUMS_DTYPE_FLOAT32 = 3
};

enum {
    PYQCEINSUMS_PARAM_NUM_OPS = 0,
    PYQCEINSUMS_PARAM_INPUT_RANK = 1,
    PYQCEINSUMS_PARAM_DTYPE_ID = 2,
    PYQCEINSUMS_PARAM_OUT_NDIM = 3,
    PYQCEINSUMS_PARAM_SIZE = 8
};

enum {
    PYQCEINSUMS_SET_OUTPUT = 0,
    PYQCEINSUMS_SET_INPUTS = 1,
    PYQCEINSUMS_SET_INPUT_SHAPES = 2,
    PYQCEINSUMS_SET_OUTPUT_SHAPE = 3,
    PYQCEINSUMS_SET_SIZE = 8
};

enum {
    PYQCEINSUMS_ABI_VERSION = 1,
    PYQCEINSUMS_MAX_OPERANDS = 2,
    PYQCEINSUMS_MAX_RANK = 4
};

/* Returns PYQCEINSUMS_ABI_VERSION of the loaded archive. */
PYQCEINSUMS_API int pyqceinsums_abi_version(void);

/*
 * 1 only when this archive was compiled with PYQCEINSUMS_HAVE_EINSUMS=1 **and**
 * Einsums headers were found at build time; 0 for the always-built stub, whose
 * pyqceinsums_einsum() returns PYQCEINSUMS_STATUS_UNAVAILABLE.
 */
PYQCEINSUMS_API int pyqceinsums_backend_enabled(void);

/*
 * Evaluates the Einsums `spec` on the operands described by `set_ptrs` and
 * writes into the output tensor. Returns a PYQCEINSUMS_STATUS_* code; the
 * message of the last failure is available through pyqceinsums_last_error().
 */
PYQCEINSUMS_API int pyqceinsums_einsum(
    const char* spec, long long set_ptrs, long long params);

/* Thread-local diagnostic of the last non-OK pyqceinsums_einsum() call. */
PYQCEINSUMS_API const char* pyqceinsums_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
