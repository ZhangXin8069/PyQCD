#ifndef PYQCD_MATX_EINSUM_H
#define PYQCD_MATX_EINSUM_H

#include <stdint.h>

#if defined(_WIN32)
#define PYQCMATX_API __declspec(dllexport)
#else
#define PYQCMATX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MatX `einsum` bridge ABI, version 1.
 *
 * Layout follows `cpp/cuda/einsum/include/einsum_define.h`: pointers and
 * counts travel as flat integer arrays so Python can marshal them with
 * ctypes without any struct packing agreement.
 *
 * `set_ptrs` is int64_t[PYQCMATX_SET_SIZE]:
 *   [OUTPUT]        device pointer of the output tensor
 *   [INPUTS]        host pointer to int64_t[nops] of device pointers
 *   [INPUT_SHAPES]  host pointer to int64_t[nops * INPUT_RANK], row-major
 *   [OUTPUT_SHAPE]  host pointer to int64_t[OUT_NDIM]
 *
 * `params` is int32_t[PYQCMATX_PARAM_SIZE]:
 *   [NUM_OPS], [INPUT_RANK], [DTYPE_ID], [OUT_NDIM]
 *
 * The bridge is deliberately bounded: it accepts one or two operands, ranks
 * up to 4, and requires both operands of a binary contraction to share a rank.
 * Everything outside that envelope is rejected with PYQCMATX_STATUS_INVALID
 * before any cuTENSOR call, matching the fail-closed rule of the einsum ABI.
 */

enum {
    PYQCMATX_STATUS_OK = 0,
    PYQCMATX_STATUS_INVALID = 1,
    PYQCMATX_STATUS_MEMORY = 2,
    PYQCMATX_STATUS_INTERNAL = 3,
    PYQCMATX_STATUS_UNAVAILABLE = 4
};

enum {
    PYQCMATX_DTYPE_COMPLEX128 = 0,
    PYQCMATX_DTYPE_COMPLEX64 = 1,
    PYQCMATX_DTYPE_FLOAT64 = 2,
    PYQCMATX_DTYPE_FLOAT32 = 3
};

enum {
    PYQCMATX_PARAM_NUM_OPS = 0,
    PYQCMATX_PARAM_INPUT_RANK = 1,
    PYQCMATX_PARAM_DTYPE_ID = 2,
    PYQCMATX_PARAM_OUT_NDIM = 3,
    PYQCMATX_PARAM_SIZE = 8
};

enum {
    PYQCMATX_SET_OUTPUT = 0,
    PYQCMATX_SET_INPUTS = 1,
    PYQCMATX_SET_INPUT_SHAPES = 2,
    PYQCMATX_SET_OUTPUT_SHAPE = 3,
    PYQCMATX_SET_SIZE = 8
};

enum {
    PYQCMATX_ABI_VERSION = 1,
    PYQCMATX_MAX_OPERANDS = 2,
    PYQCMATX_MAX_RANK = 4
};

/* Returns PYQCMATX_ABI_VERSION of the loaded archive. */
PYQCMATX_API int pyqcmatx_abi_version(void);

/* 1 when built with -DMATX_EN_CUTENSOR=ON against MatX, else 0. */
PYQCMATX_API int pyqcmatx_cutensor_enabled(void);

/*
 * Evaluates `subscripts` on the operands described by `set_ptrs` and writes
 * into the output tensor. Returns a PYQCMATX_STATUS_* code; the message of the
 * last failure is available through pyqcmatx_last_error().
 */
PYQCMATX_API int pyqcmatx_einsum(
    const char* subscripts, long long set_ptrs, long long params);

/* Thread-local diagnostic of the last non-OK pyqcmatx_einsum() call. */
PYQCMATX_API const char* pyqcmatx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
