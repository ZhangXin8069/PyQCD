#include <cstdint>
#include <string>

#include "matx_einsum.h"

#if defined(PYQCMATX_HAVE_MATX) && PYQCMATX_HAVE_MATX
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <matx.h>
#endif

namespace {

thread_local std::string g_last_error;

int reject(const std::string& message) {
    g_last_error = message;
    return PYQCMATX_STATUS_INVALID;
}

int unavailable(const char* reason) {
    g_last_error = reason;
    return PYQCMATX_STATUS_UNAVAILABLE;
}

constexpr const char* kNoMatX =
    "MatX einsum bridge was built without MatX headers; set PYQCD_MATX_ROOT "
    "(or MATX_ROOT) to a MatX checkout and rebuild cpp/cuda/matx/make-nv.sh";

constexpr const char* kNoCutensor =
    "MatX einsum bridge was built with MatX but without -DMATX_EN_CUTENSOR=ON; "
    "cutensor::einsum requires cuTENSOR>=1.7.0 and cuTensorNet>=23.03.0.20";

#if defined(PYQCMATX_HAVE_MATX) && PYQCMATX_HAVE_MATX && \
    defined(PYQCMATX_ENABLE_CUTENSOR) && PYQCMATX_ENABLE_CUTENSOR

// matx::make_tensor(T*, const index_t (&shape)[RANK], bool owning=false) takes a
// C array reference, so the shape must live in a wrapper whose member is the
// array; an expression returning cuda::std::array would not bind.
template <int R>
struct Shape {
    matx::index_t values[R];
};

template <int R>
Shape<R> to_shape(const int64_t* raw) {
    Shape<R> shape{};
    for (int axis = 0; axis < R; ++axis) {
        shape.values[axis] = static_cast<matx::index_t>(raw[axis]);
    }
    return shape;
}

template <typename T, int RIN, int ROUT>
int run_unary(const char* subscripts, void* out, void* const* inputs,
              const int64_t* input_shapes, const int64_t* out_shape) {
    if constexpr (RIN == 0) {
        return reject("rank-0 input operands are not supported by the bridge");
    } else {
        auto shape = to_shape<RIN>(input_shapes);
        auto a = matx::make_tensor<T>(static_cast<T*>(inputs[0]),
                                      shape.values);
        matx::CUDAExecutor exec;
        if constexpr (ROUT == 0) {
            // A zero-length C array is not valid C++, so wrap the scalar output
            // by allocating a rank-0 tensor and copying the single element back.
            auto c = matx::make_tensor<T>(cuda::std::array<matx::index_t, 0>{});
            (c = matx::cutensor::einsum(subscripts, a)).run(exec);
            cudaMemcpy(out, c.Data(), sizeof(T), cudaMemcpyDeviceToDevice);
        } else {
            auto shape_out = to_shape<ROUT>(out_shape);
            auto c = matx::make_tensor<T>(static_cast<T*>(out),
                                          shape_out.values);
            (c = matx::cutensor::einsum(subscripts, a)).run(exec);
        }
        return PYQCMATX_STATUS_OK;
    }
}

template <typename T, int RIN, int ROUT>
int run_binary(const char* subscripts, void* out, void* const* inputs,
               const int64_t* input_shapes, const int64_t* out_shape) {
    if constexpr (RIN == 0) {
        return reject("rank-0 input operands are not supported by the bridge");
    } else {
        auto shape_a = to_shape<RIN>(input_shapes);
        auto shape_b = to_shape<RIN>(input_shapes + RIN);
        auto a = matx::make_tensor<T>(static_cast<T*>(inputs[0]),
                                      shape_a.values);
        auto b = matx::make_tensor<T>(static_cast<T*>(inputs[1]),
                                      shape_b.values);
        matx::CUDAExecutor exec;
        if constexpr (ROUT == 0) {
            auto c = matx::make_tensor<T>(cuda::std::array<matx::index_t, 0>{});
            (c = matx::cutensor::einsum(subscripts, a, b)).run(exec);
            cudaMemcpy(out, c.Data(), sizeof(T), cudaMemcpyDeviceToDevice);
        } else {
            auto shape_out = to_shape<ROUT>(out_shape);
            auto c = matx::make_tensor<T>(static_cast<T*>(out),
                                          shape_out.values);
            (c = matx::cutensor::einsum(subscripts, a, b)).run(exec);
        }
        return PYQCMATX_STATUS_OK;
    }
}

template <typename T>
int dispatch_unary(const char* subscripts, int rank, int out_ndim, void* out,
                   void* const* inputs, const int64_t* input_shapes,
                   const int64_t* out_shape) {
    switch (rank) {
    case 0:
        switch (out_ndim) {
        case 0: return run_unary<T, 0, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_unary<T, 0, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_unary<T, 0, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_unary<T, 0, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_unary<T, 0, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 1:
        switch (out_ndim) {
        case 0: return run_unary<T, 1, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_unary<T, 1, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_unary<T, 1, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_unary<T, 1, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_unary<T, 1, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 2:
        switch (out_ndim) {
        case 0: return run_unary<T, 2, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_unary<T, 2, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_unary<T, 2, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_unary<T, 2, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_unary<T, 2, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 3:
        switch (out_ndim) {
        case 0: return run_unary<T, 3, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_unary<T, 3, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_unary<T, 3, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_unary<T, 3, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_unary<T, 3, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 4:
        switch (out_ndim) {
        case 0: return run_unary<T, 4, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_unary<T, 4, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_unary<T, 4, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_unary<T, 4, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_unary<T, 4, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    default:
        return reject("input rank out of range");
    }
}

template <typename T>
int dispatch_binary(const char* subscripts, int rank, int out_ndim, void* out,
                    void* const* inputs, const int64_t* input_shapes,
                    const int64_t* out_shape) {
    switch (rank) {
    case 0:
        switch (out_ndim) {
        case 0: return run_binary<T, 0, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_binary<T, 0, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_binary<T, 0, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_binary<T, 0, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_binary<T, 0, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 1:
        switch (out_ndim) {
        case 0: return run_binary<T, 1, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_binary<T, 1, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_binary<T, 1, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_binary<T, 1, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_binary<T, 1, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 2:
        switch (out_ndim) {
        case 0: return run_binary<T, 2, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_binary<T, 2, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_binary<T, 2, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_binary<T, 2, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_binary<T, 2, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 3:
        switch (out_ndim) {
        case 0: return run_binary<T, 3, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_binary<T, 3, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_binary<T, 3, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_binary<T, 3, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_binary<T, 3, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    case 4:
        switch (out_ndim) {
        case 0: return run_binary<T, 4, 0>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 1: return run_binary<T, 4, 1>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 2: return run_binary<T, 4, 2>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 3: return run_binary<T, 4, 3>(
            subscripts, out, inputs, input_shapes, out_shape);
        case 4: return run_binary<T, 4, 4>(
            subscripts, out, inputs, input_shapes, out_shape);
        default: return reject("out_ndim out of range");
        }
    default:
        return reject("input rank out of range");
    }
}

template <typename T>
int dispatch(const char* subscripts, int num_ops, int rank, int out_ndim,
             void* out, void* const* inputs, const int64_t* input_shapes,
             const int64_t* out_shape) {
    if (num_ops == 1) {
        return dispatch_unary<T>(subscripts, rank, out_ndim, out, inputs,
                                 input_shapes, out_shape);
    }
    return dispatch_binary<T>(subscripts, rank, out_ndim, out, inputs,
                              input_shapes, out_shape);
}

#endif

}  // namespace

extern "C" {

int pyqcmatx_abi_version(void) { return PYQCMATX_ABI_VERSION; }

int pyqcmatx_cutensor_enabled(void) {
#if defined(PYQCMATX_HAVE_MATX) && PYQCMATX_HAVE_MATX && \
    defined(PYQCMATX_ENABLE_CUTENSOR) && PYQCMATX_ENABLE_CUTENSOR
    return 1;
#else
    return 0;
#endif
}

const char* pyqcmatx_last_error(void) { return g_last_error.c_str(); }

int pyqcmatx_einsum(const char* subscripts, long long set_ptrs,
                    long long params) {
    g_last_error.clear();
    if (subscripts == nullptr || subscripts[0] == '\0') {
        return reject("subscripts is empty");
    }
    if (set_ptrs == 0 || params == 0) {
        return reject("set_ptrs/params must point to caller-owned arrays");
    }
    const auto* slots = reinterpret_cast<const int64_t*>(set_ptrs);
    const auto* controls = reinterpret_cast<const int32_t*>(params);
    const int num_ops = controls[PYQCMATX_PARAM_NUM_OPS];
    const int rank = controls[PYQCMATX_PARAM_INPUT_RANK];
    const int dtype_id = controls[PYQCMATX_PARAM_DTYPE_ID];
    const int out_ndim = controls[PYQCMATX_PARAM_OUT_NDIM];
    if (num_ops < 1 || num_ops > PYQCMATX_MAX_OPERANDS) {
        return reject("num_ops must be 1 or 2");
    }
    if (rank < 0 || rank > PYQCMATX_MAX_RANK) {
        return reject("input rank out of range");
    }
    if (out_ndim < 0 || out_ndim > PYQCMATX_MAX_RANK) {
        return reject("out_ndim out of range");
    }
    if (dtype_id < PYQCMATX_DTYPE_COMPLEX128 ||
        dtype_id > PYQCMATX_DTYPE_FLOAT32) {
        return reject("dtype_id out of range");
    }
    void* out = reinterpret_cast<void*>(slots[PYQCMATX_SET_OUTPUT]);
    const auto* input_ptrs =
        reinterpret_cast<const int64_t*>(slots[PYQCMATX_SET_INPUTS]);
    const auto* input_shapes =
        reinterpret_cast<const int64_t*>(slots[PYQCMATX_SET_INPUT_SHAPES]);
    const auto* out_shape =
        reinterpret_cast<const int64_t*>(slots[PYQCMATX_SET_OUTPUT_SHAPE]);
    if (out == nullptr) {
        return reject("output device pointer is null");
    }
    if (input_ptrs == nullptr || input_shapes == nullptr ||
        out_shape == nullptr) {
        return reject("input pointer/shape arrays are null");
    }
    void* inputs[PYQCMATX_MAX_OPERANDS] = {nullptr, nullptr};
    for (int index = 0; index < num_ops; ++index) {
        if (input_ptrs[index] == 0) {
            return reject("input device pointer is null");
        }
        inputs[index] = reinterpret_cast<void*>(input_ptrs[index]);
    }
    for (int index = 0; index < num_ops * rank; ++index) {
        if (input_shapes[index] < 0) {
            return reject("input shape has a negative extent");
        }
    }
    for (int axis = 0; axis < out_ndim; ++axis) {
        if (out_shape[axis] < 0) {
            return reject("output shape has a negative extent");
        }
    }

#if defined(PYQCMATX_HAVE_MATX) && PYQCMATX_HAVE_MATX && \
    defined(PYQCMATX_ENABLE_CUTENSOR) && PYQCMATX_ENABLE_CUTENSOR
    try {
        switch (dtype_id) {
        case PYQCMATX_DTYPE_COMPLEX128:
            return dispatch<cuda::std::complex<double>>(
                subscripts, num_ops, rank, out_ndim, out, inputs,
                input_shapes, out_shape);
        case PYQCMATX_DTYPE_COMPLEX64:
            return dispatch<cuda::std::complex<float>>(
                subscripts, num_ops, rank, out_ndim, out, inputs,
                input_shapes, out_shape);
        case PYQCMATX_DTYPE_FLOAT64:
            return dispatch<double>(subscripts, num_ops, rank, out_ndim, out,
                                    inputs, input_shapes, out_shape);
        case PYQCMATX_DTYPE_FLOAT32:
            return dispatch<float>(subscripts, num_ops, rank, out_ndim, out,
                                   inputs, input_shapes, out_shape);
        default:
            return reject("dtype_id out of range");
        }
    } catch (const std::exception& error) {
        g_last_error = std::string("MatX einsum threw: ") + error.what();
        return PYQCMATX_STATUS_INTERNAL;
    } catch (...) {
        g_last_error = "MatX einsum threw a non-standard exception";
        return PYQCMATX_STATUS_INTERNAL;
    }
#elif defined(PYQCMATX_HAVE_MATX) && PYQCMATX_HAVE_MATX
    return unavailable(kNoCutensor);
#else
    return unavailable(kNoMatX);
#endif
}

}  // extern "C"
