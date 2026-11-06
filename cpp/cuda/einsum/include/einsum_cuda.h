#ifndef PYQCD_EINSUM_CUDA_H
#define PYQCD_EINSUM_CUDA_H

#include "einsum_define.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>

constexpr int32_t QCD_CUDA_MAX_RANK = 64;
constexpr int32_t QCD_CUDA_MAX_PACKED_RANK = 128;

struct qcd_cuda_copy_spec {
    int32_t rank;
    int64_t elements;
    int64_t shape[QCD_CUDA_MAX_PACKED_RANK];
    int64_t src_stride[QCD_CUDA_MAX_PACKED_RANK];
};

struct qcd_cuda_hadamard_spec {
    int32_t rank;
    int64_t elements;
    int64_t shape[QCD_CUDA_MAX_RANK];
    int64_t left_stride[QCD_CUDA_MAX_RANK];
    int64_t right_stride[QCD_CUDA_MAX_RANK];
};

struct qcd_cuda_reduce_spec {
    int32_t output_rank;
    int32_t reduction_rank;
    int64_t output_elements;
    int64_t reduction_elements;
    int64_t output_shape[QCD_CUDA_MAX_RANK];
    int64_t output_src_stride[QCD_CUDA_MAX_RANK];
    int64_t reduction_extent[QCD_CUDA_MAX_RANK];
    int64_t reduction_src_stride[QCD_CUDA_MAX_RANK];
};

struct qcd_cuda_sandwich_spec {
    int64_t prefix;
    int64_t left;
    int64_t contracted_left;
    int64_t contracted_right;
    int64_t right;
    int64_t lane;
};

struct qcd_cuda_e4_spec {
    int64_t z;
    int64_t y;
    int64_t x;
    int64_t c;
    int64_t t;
    int64_t a;
    int64_t b;
    int64_t n;
    int64_t input_g;
    int64_t output_g;
    int64_t d;
    int64_t e;
    int64_t f;
    int64_t output_elements;
};

#if defined(_WIN32)
#define QCD_CUDA_LOCAL
#define QCD_CUDA_WEAK
#else
#define QCD_CUDA_LOCAL __attribute__((visibility("hidden")))
#define QCD_CUDA_WEAK __attribute__((weak, visibility("hidden")))
#endif

extern "C" QCD_CUDA_WEAK int qcd_cuda_permute(
    int dtype_id, const void* src, void* dst, qcd_cuda_copy_spec spec,
    cudaStream_t stream);

extern "C" QCD_CUDA_WEAK int qcd_cuda_hadamard(
    int dtype_id, const void* left, const void* right, void* dst,
    qcd_cuda_hadamard_spec spec, cudaStream_t stream);

extern "C" QCD_CUDA_WEAK int qcd_cuda_reduce(
    int dtype_id, const void* src, void* dst, qcd_cuda_reduce_spec spec,
    cudaStream_t stream);

extern "C" QCD_CUDA_WEAK int qcd_cuda_sandwich(
    int dtype_id,
    const void* left_matrix,
    const void* tensor,
    const void* right_matrix,
    void* dst,
    qcd_cuda_sandwich_spec spec,
    cudaStream_t stream);

extern "C" QCD_CUDA_WEAK int qcd_cuda_e4_fused(
    int dtype_id,
    const void* a,
    const void* b,
    const void* gamma,
    const void* c,
    void* dst,
    qcd_cuda_e4_spec spec,
    cudaStream_t stream);

extern "C" QCD_CUDA_WEAK int qcd_cuda_batched_gemm(
    int dtype_id,
    cublasHandle_t handle,
    const void* left,
    const void* right,
    void* dst,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch,
    cudaStream_t stream);

#endif
