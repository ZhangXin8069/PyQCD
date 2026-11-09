#ifndef PYQCD_MATX_PYTHON_H
#define PYQCD_MATX_PYTHON_H

#include "matx_einsum.h"

#ifdef __cplusplus
extern "C" {
#endif

PYQCMATX_API int pyqcmatx_abi_version(void);
PYQCMATX_API int pyqcmatx_cutensor_enabled(void);
PYQCMATX_API int pyqcmatx_einsum(
    const char* subscripts, long long set_ptrs, long long params);
PYQCMATX_API const char* pyqcmatx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
