#ifndef PYQCD_CPU_EINSUMS_PYTHON_H
#define PYQCD_CPU_EINSUMS_PYTHON_H

/*
 * Flat C declaration surface for the ctypes/Cython side.
 *
 * Mirrors `cpp/cuda/matx/python/pyqcmatx.h`: no structs, no enums repeated --
 * the loader reads the status/dtype/slot numbers from `cpu_einsums.h`, so the
 * Python side and the C++ side can never disagree about a slot index.
 */

#include "cpu_einsums.h"

#ifdef __cplusplus
extern "C" {
#endif

PYQCEINSUMS_API int pyqceinsums_abi_version(void);
PYQCEINSUMS_API int pyqceinsums_backend_enabled(void);
PYQCEINSUMS_API int pyqceinsums_einsum(
    const char* spec, long long set_ptrs, long long params);
PYQCEINSUMS_API const char* pyqceinsums_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
