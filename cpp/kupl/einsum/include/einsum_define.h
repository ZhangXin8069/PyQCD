#ifndef PYQCD_EINSUM_DEFINE_H
#define PYQCD_EINSUM_DEFINE_H

#include <stdint.h>

#if defined(_WIN32)
#define QCD_EINSUM_API __declspec(dllexport)
#else
#define QCD_EINSUM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QCD_STATUS_OK = 0,
    QCD_STATUS_INVALID = 1,
    QCD_STATUS_MEMORY = 2,
    QCD_STATUS_INTERNAL = 3
};

enum {
    QCD_PLANNER_GREEDY = 0,
    QCD_PLANNER_OPTIMAL = 1,
    QCD_PLANNER_DP = 2,
    QCD_PLANNER_BRANCH = 3
};

enum {
    QCD_DTYPE_COMPLEX128 = 0,
    QCD_DTYPE_COMPLEX64 = 1,
    QCD_DTYPE_FLOAT64 = 2,
    QCD_DTYPE_FLOAT32 = 3
};

enum {
    QCD_PARAM_NUM_OPS = 0,
    QCD_PARAM_MAX_NDIM = 1,
    QCD_PARAM_PLANNER_ID = 2,
    QCD_PARAM_MEM_HI = 3,
    QCD_PARAM_MEM_LO = 4,
    QCD_PARAM_SEED = 5,
    QCD_PARAM_VERBOSE = 6,
    QCD_PARAM_USE_BLAS = 7,
    QCD_PARAM_DTYPE_ID = 8,
    QCD_PARAM_LAYOUT_ID = 9,
    QCD_PARAM_SIZE = 32
};

enum {
    QCD_SET_INPUTS_FLAT = 0,
    QCD_SET_INPUT_OFFSETS = 1,
    QCD_SET_OUTPUT = 2,
    QCD_SET_SIZES = 3,
    QCD_SET_OUT_PATH = 4,
    QCD_SET_OUT_COST = 5,
    QCD_SET_SCRATCH = 6,
    QCD_SET_SCRATCH_SIZE = 7,
    QCD_SET_NUM_SYMBOLS = 8,
    QCD_SET_OUTPUT_NDIM = 9,
    QCD_SET_PATH_CAPACITY = 10,
    QCD_SET_SIZE = 16
};

QCD_EINSUM_API int applyPlanQcd(long long set_ptrs, long long params);
QCD_EINSUM_API int applyContractQcd(
    long long plan, long long tensors, long long out, long long params);
QCD_EINSUM_API int applyCompileQcd(
    long long plan, long long params, long long out_context);
QCD_EINSUM_API int applyCompiledQcd(
    long long context, long long tensors, long long out, long long params);
QCD_EINSUM_API void applyDestroyQcd(long long context);
QCD_EINSUM_API void applyInitQcd(long long set_ptrs, long long params);
QCD_EINSUM_API void applyEndQcd(long long set_ptrs, long long params);

#ifdef __cplusplus
}
#endif

#endif
