#include "einsum_cuda.h"

#include <cublasLt.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

int map_cuda_error(cudaError_t error) {
    if (error == cudaSuccess) {
        return QCD_STATUS_OK;
    }
    return error == cudaErrorMemoryAllocation ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

int map_cublas_status(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return QCD_STATUS_OK;
    }
    return status == CUBLAS_STATUS_ALLOC_FAILED ? QCD_STATUS_MEMORY : QCD_STATUS_INTERNAL;
}

template <typename Function>
bool bind_lt_symbol(void* library, const char* name, Function* function) {
    void* symbol = dlsym(library, name);
    if (symbol == nullptr) {
        return false;
    }
    static_assert(sizeof(Function) == sizeof(symbol), "function pointer size mismatch");
    std::memcpy(function, &symbol, sizeof(Function));
    return true;
}

struct LtApi {
    void* library = nullptr;
    decltype(&cublasLtCreate) create = nullptr;
    decltype(&cublasLtDestroy) destroy = nullptr;
    decltype(&cublasLtMatmulDescCreate) create_operation = nullptr;
    decltype(&cublasLtMatmulDescDestroy) destroy_operation = nullptr;
    decltype(&cublasLtMatmulDescSetAttribute) set_operation_attribute = nullptr;
    decltype(&cublasLtMatrixLayoutCreate) create_layout = nullptr;
    decltype(&cublasLtMatrixLayoutDestroy) destroy_layout = nullptr;
    decltype(&cublasLtMatrixLayoutSetAttribute) set_layout_attribute = nullptr;
    decltype(&cublasLtMatmul) matmul = nullptr;

    bool available() const {
        return create != nullptr && destroy != nullptr && create_operation != nullptr &&
               destroy_operation != nullptr && set_operation_attribute != nullptr &&
               create_layout != nullptr &&
               destroy_layout != nullptr && set_layout_attribute != nullptr && matmul != nullptr;
    }
};

const LtApi& lt_api() {
    static const LtApi api = [] {
        LtApi loaded;
        loaded.library = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_LOCAL);
        if (loaded.library == nullptr) {
            loaded.library = dlopen("libcublasLt.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (loaded.library == nullptr) {
            return loaded;
        }
        bind_lt_symbol(loaded.library, "cublasLtCreate", &loaded.create);
        bind_lt_symbol(loaded.library, "cublasLtDestroy", &loaded.destroy);
        bind_lt_symbol(loaded.library, "cublasLtMatmulDescCreate", &loaded.create_operation);
        bind_lt_symbol(loaded.library, "cublasLtMatmulDescDestroy", &loaded.destroy_operation);
        bind_lt_symbol(loaded.library, "cublasLtMatmulDescSetAttribute",
                       &loaded.set_operation_attribute);
        bind_lt_symbol(loaded.library, "cublasLtMatrixLayoutCreate", &loaded.create_layout);
        bind_lt_symbol(loaded.library, "cublasLtMatrixLayoutDestroy", &loaded.destroy_layout);
        bind_lt_symbol(loaded.library, "cublasLtMatrixLayoutSetAttribute",
                       &loaded.set_layout_attribute);
        bind_lt_symbol(loaded.library, "cublasLtMatmul", &loaded.matmul);
        return loaded;
    }();
    return api;
}

cublasStatus_t set_batch_layout(
    const LtApi& api,
    cublasLtMatrixLayout_t layout,
    int32_t batch_count,
    int64_t stride) {
    cublasStatus_t status = api.set_layout_attribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    return api.set_layout_attribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
}

cublasStatus_t cublas_lt_batched_gemm(
    int dtype_id,
    const void* left,
    const void* right,
    void* dst,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch,
    int64_t stride_left,
    int64_t stride_right,
    int64_t stride_dst,
    cudaStream_t stream) {
    const LtApi& api = lt_api();
    if (!api.available()) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    cudaDataType_t data_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            data_type = CUDA_C_64F;
            compute_type = CUBLAS_COMPUTE_64F;
            break;
        case QCD_DTYPE_COMPLEX64:
            data_type = CUDA_C_32F;
            compute_type = CUBLAS_COMPUTE_32F;
            break;
        case QCD_DTYPE_FLOAT64:
            data_type = CUDA_R_64F;
            compute_type = CUBLAS_COMPUTE_64F;
            break;
        case QCD_DTYPE_FLOAT32:
            data_type = CUDA_R_32F;
            compute_type = CUBLAS_COMPUTE_32F;
            break;
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }

    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t right_layout = nullptr;
    cublasLtMatrixLayout_t left_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasStatus_t status = api.create(&handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = api.create_operation(&operation, compute_type, data_type);
    }
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = api.set_operation_attribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = api.set_operation_attribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose, sizeof(no_transpose));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = api.create_layout(
            &right_layout, data_type, static_cast<uint64_t>(k), static_cast<uint64_t>(n),
            static_cast<int64_t>(k));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = api.create_layout(
            &left_layout, data_type, static_cast<uint64_t>(k), static_cast<uint64_t>(m),
            static_cast<int64_t>(k));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = api.create_layout(
            &output_layout, data_type, static_cast<uint64_t>(n), static_cast<uint64_t>(m),
            static_cast<int64_t>(n));
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = set_batch_layout(api, right_layout, static_cast<int32_t>(batch), stride_right);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = set_batch_layout(api, left_layout, static_cast<int32_t>(batch), stride_left);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = set_batch_layout(api, output_layout, static_cast<int32_t>(batch), stride_dst);
    }

    if (status == CUBLAS_STATUS_SUCCESS) {
        if (dtype_id == QCD_DTYPE_COMPLEX128) {
            const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            status = api.matmul(
                handle, operation, &alpha, right, right_layout, left, left_layout, &beta,
                dst, output_layout, dst, output_layout, nullptr, nullptr, 0, stream);
        } else if (dtype_id == QCD_DTYPE_COMPLEX64) {
            const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
            const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
            status = api.matmul(
                handle, operation, &alpha, right, right_layout, left, left_layout, &beta,
                dst, output_layout, dst, output_layout, nullptr, nullptr, 0, stream);
        } else if (dtype_id == QCD_DTYPE_FLOAT64) {
            const double alpha = 1.0;
            const double beta = 0.0;
            status = api.matmul(
                handle, operation, &alpha, right, right_layout, left, left_layout, &beta,
                dst, output_layout, dst, output_layout, nullptr, nullptr, 0, stream);
        } else {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            status = api.matmul(
                handle, operation, &alpha, right, right_layout, left, left_layout, &beta,
                dst, output_layout, dst, output_layout, nullptr, nullptr, 0, stream);
        }
    }

    if (output_layout != nullptr) {
        api.destroy_layout(output_layout);
    }
    if (left_layout != nullptr) {
        api.destroy_layout(left_layout);
    }
    if (right_layout != nullptr) {
        api.destroy_layout(right_layout);
    }
    if (operation != nullptr) {
        api.destroy_operation(operation);
    }
    if (handle != nullptr) {
        api.destroy(handle);
    }
    return status;
}

bool multiply_checked(int64_t left, int64_t right, int64_t* product) {
    if (left < 0 || right < 0 ||
        (left != 0 && right > std::numeric_limits<int64_t>::max() / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

int zero_output(
    int dtype_id, void* dst, int64_t m, int64_t n, int64_t batch,
    cudaStream_t stream) {
    int64_t elements = 0;
    int64_t matrices = 0;
    if (!multiply_checked(m, n, &elements) || !multiply_checked(elements, batch, &matrices)) {
        return QCD_STATUS_INVALID;
    }
    std::size_t element_size = 0;
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128:
            element_size = sizeof(cuDoubleComplex);
            break;
        case QCD_DTYPE_COMPLEX64:
            element_size = sizeof(cuFloatComplex);
            break;
        case QCD_DTYPE_FLOAT64:
            element_size = sizeof(double);
            break;
        case QCD_DTYPE_FLOAT32:
            element_size = sizeof(float);
            break;
        default:
            return QCD_STATUS_INVALID;
    }
    if (matrices == 0) {
        return QCD_STATUS_OK;
    }
    if (static_cast<uint64_t>(matrices) > std::numeric_limits<std::size_t>::max() / element_size) {
        return QCD_STATUS_INVALID;
    }
    return map_cuda_error(cudaMemsetAsync(
        dst, 0, static_cast<std::size_t>(matrices) * element_size, stream));
}

}  // namespace

extern "C" QCD_CUDA_LOCAL int qcd_cuda_batched_gemm(
    int dtype_id,
    cublasHandle_t handle,
    const void* left,
    const void* right,
    void* dst,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch,
    cudaStream_t stream) {
    if (handle == nullptr || m < 0 || n < 0 || k < 0 || batch < 0 ||
        m > std::numeric_limits<int>::max() || n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max() || batch > std::numeric_limits<int>::max()) {
        return QCD_STATUS_INVALID;
    }
    if (m == 0 || n == 0 || batch == 0) {
        return QCD_STATUS_OK;
    }
    if (k == 0) {
        if (dst == nullptr) {
            return QCD_STATUS_INVALID;
        }
        return zero_output(dtype_id, dst, m, n, batch, stream);
    }
    if (left == nullptr || right == nullptr || dst == nullptr) {
        return QCD_STATUS_INVALID;
    }

    const int cublas_m = static_cast<int>(n);
    const int cublas_n = static_cast<int>(m);
    const int cublas_k = static_cast<int>(k);
    const int cublas_batch = static_cast<int>(batch);
    int64_t stride_left = 0;
    int64_t stride_right = 0;
    int64_t stride_dst = 0;
    if (!multiply_checked(m, k, &stride_left) ||
        !multiply_checked(k, n, &stride_right) ||
        !multiply_checked(m, n, &stride_dst)) {
        return QCD_STATUS_INVALID;
    }

    int status = map_cublas_status(cublasSetStream(handle, stream));
    if (status != QCD_STATUS_OK) {
        return status;
    }
    status = map_cublas_status(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    if (status != QCD_STATUS_OK) {
        return status;
    }

    cublasStatus_t result = CUBLAS_STATUS_NOT_INITIALIZED;
    switch (dtype_id) {
        case QCD_DTYPE_COMPLEX128: {
            const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            result = cublasZgemmStridedBatched(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                cublas_m,
                cublas_n,
                cublas_k,
                &alpha,
                static_cast<const cuDoubleComplex*>(right),
                cublas_k,
                stride_right,
                static_cast<const cuDoubleComplex*>(left),
                cublas_k,
                stride_left,
                &beta,
                static_cast<cuDoubleComplex*>(dst),
                cublas_m,
                stride_dst,
                cublas_batch);
            break;
        }
        case QCD_DTYPE_COMPLEX64: {
            const cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
            const cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
            result = cublasCgemmStridedBatched(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                cublas_m,
                cublas_n,
                cublas_k,
                &alpha,
                static_cast<const cuFloatComplex*>(right),
                cublas_k,
                stride_right,
                static_cast<const cuFloatComplex*>(left),
                cublas_k,
                stride_left,
                &beta,
                static_cast<cuFloatComplex*>(dst),
                cublas_m,
                stride_dst,
                cublas_batch);
            break;
        }
        case QCD_DTYPE_FLOAT64: {
            const double alpha = 1.0;
            const double beta = 0.0;
            result = cublasDgemmStridedBatched(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                cublas_m,
                cublas_n,
                cublas_k,
                &alpha,
                static_cast<const double*>(right),
                cublas_k,
                stride_right,
                static_cast<const double*>(left),
                cublas_k,
                stride_left,
                &beta,
                static_cast<double*>(dst),
                cublas_m,
                stride_dst,
                cublas_batch);
            break;
        }
        case QCD_DTYPE_FLOAT32: {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            result = cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                cublas_m,
                cublas_n,
                cublas_k,
                &alpha,
                static_cast<const float*>(right),
                cublas_k,
                stride_right,
                static_cast<const float*>(left),
                cublas_k,
                stride_left,
                &beta,
                static_cast<float*>(dst),
                cublas_m,
                stride_dst,
                cublas_batch);
            break;
        }
        default:
            return QCD_STATUS_INVALID;
    }
    if (result == CUBLAS_STATUS_NOT_SUPPORTED) {
        result = cublas_lt_batched_gemm(
            dtype_id, left, right, dst, m, n, k, batch,
            stride_left, stride_right, stride_dst, stream);
    }
    return map_cublas_status(result);
}
