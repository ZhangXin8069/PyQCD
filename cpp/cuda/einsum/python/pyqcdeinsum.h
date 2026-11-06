#ifndef PYQCD_EINSUM_PYTHON_H
#define PYQCD_EINSUM_PYTHON_H

#include "einsum_define.h"

#ifdef __cplusplus
extern "C" {
#endif

QCD_EINSUM_API int applyPlanQcd(long long set_ptrs, long long params);
QCD_EINSUM_API int applyContractQcd(
    long long plan, long long tensors, long long out, long long params);
QCD_EINSUM_API void applyInitQcd(long long set_ptrs, long long params);
QCD_EINSUM_API void applyEndQcd(long long set_ptrs, long long params);

#ifdef __cplusplus
}
#endif

#endif
