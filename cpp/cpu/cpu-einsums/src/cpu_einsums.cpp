// Einsums string-`einsum` C ABI bridge (CPU, host pointers).
//
// Two build modes, selected by -DPYQCEINSUMS_HAVE_EINSUMS=0|1 from make.sh:
//
//   * HAVE_EINSUMS=0  -- the always-buildable stub. The archive still exports
//                        the frozen four-entry ABI, but every einsum call
//                        returns PYQCEINSUMS_STATUS_UNAVAILABLE together with an
//                        actionable reason. Nothing here fakes a result.
//   * HAVE_EINSUMS=1  -- the real path: materialise einsums::Tensor objects,
//                        std::copy the caller's host buffers in, call
//                        einsums::compute_graph::einsum(spec, &out, a[, b]) and
//                        std::copy the result back. **Never compiled on this
//                        machine** (no Einsums headers) -- see AGENTS.md §6.
//
// Validation is shared by both modes and runs before anything else, exactly in
// the order documented in include/cpu_einsums.h.

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>

#include "cpu_einsums.h"

#if defined(PYQCEINSUMS_HAVE_EINSUMS) && PYQCEINSUMS_HAVE_EINSUMS

// Header paths verified against refer/git-rep/Einsums @ 628971c:
//   libs/Einsums/Tensor/include/Einsums/Tensor/Tensor.hpp      (create_tensor, Tensor<T,R>)
//   libs/Einsums/ComputeGraph/include/Einsums/ComputeGraph/Operations.hpp  (cg::einsum)
// Both stay overridable for a different Einsums version.
#ifndef PYQCEINSUMS_TENSOR_HEADER
#define PYQCEINSUMS_TENSOR_HEADER <Einsums/Tensor/Tensor.hpp>
#endif
#ifndef PYQCEINSUMS_COMPUTE_GRAPH_HEADER
#define PYQCEINSUMS_COMPUTE_GRAPH_HEADER <Einsums/ComputeGraph/Operations.hpp>
#endif

#include PYQCEINSUMS_TENSOR_HEADER
#include PYQCEINSUMS_COMPUTE_GRAPH_HEADER

#include <algorithm>
#include <complex>
#include <exception>
#include <mutex>
#include <utility>

#endif

namespace {

thread_local std::string g_last_error;

int reject(const std::string& message) {
    g_last_error = message;
    return PYQCEINSUMS_STATUS_INVALID;
}

#if !defined(PYQCEINSUMS_HAVE_EINSUMS) || !PYQCEINSUMS_HAVE_EINSUMS

int unavailable(const char* reason) {
    g_last_error = reason;
    return PYQCEINSUMS_STATUS_UNAVAILABLE;
}

// Reason strings must stay actionable: a user reading pyqceinsums_last_error()
// has to learn which knob to turn, not just that something is missing.
constexpr const char* kNoEinsums =
    "Einsums string-einsum bridge was built WITHOUT Einsums "
    "(PYQCEINSUMS_HAVE_EINSUMS=0), so pyqceinsums_einsum() is unavailable. "
    "Fix: point PYQCD_EINSUMS_ROOT (or EINSUMS_ROOT) at an Einsums install "
    "prefix containing include/Einsums/*.hpp, or at a source checkout whose "
    "libs/Einsums/Tensor/include/Einsums/Tensor/Tensor.hpp exists; "
    "optionally set PYQCD_EINSUMS_LIBS (e.g. the Einsums CMake targets' "
    "archive names: -lEinsums-abi<N> for the umbrella, or "
    "-lEinsums_Tensor -lEinsums_ComputeGraph ... for the module libraries) plus "
    "PYQCD_EINSUMS_FLAGS, then rerun `bash cpp/cpu/cpu-einsums/make.sh`. The "
    "einsums python module is only needed for the pybind11 face and requires an "
    "Einsums configured with -DEINSUMS_BUILD_PYTHON=ON (pybind11>=2.11); this C "
    "ABI does not import it.";

#endif

#if defined(PYQCEINSUMS_HAVE_EINSUMS) && PYQCEINSUMS_HAVE_EINSUMS

// ---------------------------------------------------------------- real path

// Einsums initialises its modules through registered startup hooks
// (register_startup_function in libs/Einsums/*/src/InitModule.cpp); there is no
// top-level einsums::initialize(). The only symbol with that name is
// einsums::comm::initialize(int*, char***), the MPI layer, which the CPU eager
// path does not need. Nothing to call here.

template <int R>
std::size_t element_count(const int64_t* shape) {
    std::size_t count = 1;
    for (int axis = 0; axis < R; ++axis) {
        count *= static_cast<std::size_t>(shape[axis]);
    }
    return count;
}

// create_tensor<Type>(std::string const&, Args...):
//   requires(std::is_convertible_v<Args, std::size_t> && ...)
// and returns Tensor<Type, sizeof...(Args)>. Dimensions are cast to size_t to
// match that constraint exactly.
template <typename T, int R, std::size_t... I>
auto make_tensor_impl(const char* name, const int64_t* shape,
                      std::index_sequence<I...>) {
    return einsums::create_tensor<T>(name,
                                     static_cast<std::size_t>(shape[I])...);
}

template <typename T, int R>
auto make_tensor(const char* name, const int64_t* shape) {
    return make_tensor_impl<T, R>(name, shape, std::make_index_sequence<R>{});
}

template <typename T, int RIN, int ROUT>
int run_unary(const char* spec, void* out, void* const* inputs,
              const int64_t* input_shapes, const int64_t* out_shape) {
    auto a = make_tensor<T, RIN>("A", input_shapes);
    auto c = make_tensor<T, ROUT>("C", out_shape);
    std::copy(static_cast<const T*>(inputs[0]),
              static_cast<const T*>(inputs[0]) + element_count<RIN>(input_shapes),
              a.data());
    einsums::compute_graph::einsum(spec, &c, a);
    std::copy(c.data(),
              c.data() + element_count<ROUT>(out_shape),
              static_cast<T*>(out));
    return PYQCEINSUMS_STATUS_OK;
}

template <typename T, int RIN, int ROUT>
int run_binary(const char* spec, void* out, void* const* inputs,
               const int64_t* input_shapes, const int64_t* out_shape) {
    auto a = make_tensor<T, RIN>("A", input_shapes);
    auto b = make_tensor<T, RIN>("B", input_shapes + RIN);
    auto c = make_tensor<T, ROUT>("C", out_shape);
    const std::size_t operand_count = element_count<RIN>(input_shapes);
    std::copy(static_cast<const T*>(inputs[0]),
              static_cast<const T*>(inputs[0]) + operand_count, a.data());
    std::copy(static_cast<const T*>(inputs[1]),
              static_cast<const T*>(inputs[1]) + operand_count, b.data());
    einsums::compute_graph::einsum(spec, &c, a, b);
    std::copy(c.data(),
              c.data() + element_count<ROUT>(out_shape),
              static_cast<T*>(out));
    return PYQCEINSUMS_STATUS_OK;
}

template <typename T, int RIN, int ROUT>
int run(const char* spec, int num_ops, void* out, void* const* inputs,
        const int64_t* input_shapes, const int64_t* out_shape) {
    if (num_ops == 1) {
        return run_unary<T, RIN, ROUT>(spec, out, inputs, input_shapes,
                                       out_shape);
    }
    return run_binary<T, RIN, ROUT>(spec, out, inputs, input_shapes, out_shape);
}

template <typename T, int RIN>
int dispatch_out_ndim(const char* spec, int num_ops, int out_ndim, void* out,
                      void* const* inputs, const int64_t* input_shapes,
                      const int64_t* out_shape) {
    switch (out_ndim) {
    case 0: return run<T, RIN, 0>(spec, num_ops, out, inputs, input_shapes,
                                  out_shape);
    case 1: return run<T, RIN, 1>(spec, num_ops, out, inputs, input_shapes,
                                  out_shape);
    case 2: return run<T, RIN, 2>(spec, num_ops, out, inputs, input_shapes,
                                  out_shape);
    case 3: return run<T, RIN, 3>(spec, num_ops, out, inputs, input_shapes,
                                  out_shape);
    case 4: return run<T, RIN, 4>(spec, num_ops, out, inputs, input_shapes,
                                  out_shape);
    default:
        // Unreachable: the ABI layer already rejected out_ndim outside 0..4.
        return reject("out_ndim out of range");
    }
}

template <typename T>
int dispatch(const char* spec, int num_ops, int rank, int out_ndim, void* out,
             void* const* inputs, const int64_t* input_shapes,
             const int64_t* out_shape) {
    switch (rank) {
    case 0: return dispatch_out_ndim<T, 0>(spec, num_ops, out_ndim, out, inputs,
                                           input_shapes, out_shape);
    case 1: return dispatch_out_ndim<T, 1>(spec, num_ops, out_ndim, out, inputs,
                                           input_shapes, out_shape);
    case 2: return dispatch_out_ndim<T, 2>(spec, num_ops, out_ndim, out, inputs,
                                           input_shapes, out_shape);
    case 3: return dispatch_out_ndim<T, 3>(spec, num_ops, out_ndim, out, inputs,
                                           input_shapes, out_shape);
    case 4: return dispatch_out_ndim<T, 4>(spec, num_ops, out_ndim, out, inputs,
                                           input_shapes, out_shape);
    default:
        // Unreachable: the ABI layer already rejected rank outside 0..4.
        return reject("input rank out of range");
    }
}

#endif  // PYQCEINSUMS_HAVE_EINSUMS

}  // namespace

extern "C" {

int pyqceinsums_abi_version(void) { return PYQCEINSUMS_ABI_VERSION; }

int pyqceinsums_backend_enabled(void) {
#if defined(PYQCEINSUMS_HAVE_EINSUMS) && PYQCEINSUMS_HAVE_EINSUMS
    return 1;
#else
    return 0;
#endif
}

const char* pyqceinsums_last_error(void) { return g_last_error.c_str(); }

int pyqceinsums_einsum(const char* spec, long long set_ptrs, long long params) {
    g_last_error.clear();
    // ---- fail-closed validation; order is part of the contract -------------
    if (spec == nullptr || spec[0] == '\0') {
        return reject("spec is empty");
    }
    if (set_ptrs == 0 || params == 0) {
        return reject("set_ptrs/params must point to caller-owned arrays");
    }
    const auto* slots = reinterpret_cast<const int64_t*>(set_ptrs);
    const auto* controls = reinterpret_cast<const int32_t*>(params);
    const int num_ops = controls[PYQCEINSUMS_PARAM_NUM_OPS];
    const int rank = controls[PYQCEINSUMS_PARAM_INPUT_RANK];
    const int dtype_id = controls[PYQCEINSUMS_PARAM_DTYPE_ID];
    const int out_ndim = controls[PYQCEINSUMS_PARAM_OUT_NDIM];
    if (num_ops != 2) {
        return reject("Einsums cg::einsum has only two-operand overloads "
                      "(Operations.hpp); a unary transpose/reduction must use "
                      "cg::permute or cg::dot instead");
    }
    if (rank < 0 || rank > PYQCEINSUMS_MAX_RANK) {
        return reject("input rank out of range");
    }
    if (out_ndim < 0 || out_ndim > PYQCEINSUMS_MAX_RANK) {
        return reject("out_ndim out of range");
    }
    if (dtype_id < PYQCEINSUMS_DTYPE_COMPLEX128 ||
        dtype_id > PYQCEINSUMS_DTYPE_FLOAT32) {
        return reject("dtype_id out of range");
    }
    void* out = reinterpret_cast<void*>(slots[PYQCEINSUMS_SET_OUTPUT]);
    const auto* input_ptrs =
        reinterpret_cast<const int64_t*>(slots[PYQCEINSUMS_SET_INPUTS]);
    const auto* input_shapes =
        reinterpret_cast<const int64_t*>(slots[PYQCEINSUMS_SET_INPUT_SHAPES]);
    const auto* out_shape =
        reinterpret_cast<const int64_t*>(slots[PYQCEINSUMS_SET_OUTPUT_SHAPE]);
    if (out == nullptr) {
        return reject("output host pointer is null");
    }
    if (input_ptrs == nullptr || input_shapes == nullptr ||
        out_shape == nullptr) {
        return reject("input pointer/shape arrays are null");
    }
    void* inputs[PYQCEINSUMS_MAX_OPERANDS] = {nullptr, nullptr};
    for (int index = 0; index < num_ops; ++index) {
        if (input_ptrs[index] == 0) {
            return reject("input host pointer is null");
        }
        inputs[index] = reinterpret_cast<void*>(input_ptrs[index]);
    }
    // INPUT_RANK is a single slot, so a binary contraction is same-rank by
    // construction; anything else is rejected before we get here.
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
    // ---- backend dispatch --------------------------------------------------
#if defined(PYQCEINSUMS_HAVE_EINSUMS) && PYQCEINSUMS_HAVE_EINSUMS
    try {
        switch (dtype_id) {
        case PYQCEINSUMS_DTYPE_COMPLEX128:
            return dispatch<std::complex<double>>(
                spec, num_ops, rank, out_ndim, out, inputs, input_shapes,
                out_shape);
        case PYQCEINSUMS_DTYPE_COMPLEX64:
            return dispatch<std::complex<float>>(
                spec, num_ops, rank, out_ndim, out, inputs, input_shapes,
                out_shape);
        case PYQCEINSUMS_DTYPE_FLOAT64:
            return dispatch<double>(spec, num_ops, rank, out_ndim, out, inputs,
                                    input_shapes, out_shape);
        case PYQCEINSUMS_DTYPE_FLOAT32:
            return dispatch<float>(spec, num_ops, rank, out_ndim, out, inputs,
                                   input_shapes, out_shape);
        default:
            return reject("dtype_id out of range");
        }
    } catch (const std::bad_alloc&) {
        g_last_error = "Einsums tensor allocation failed (host memory)";
        return PYQCEINSUMS_STATUS_MEMORY;
    } catch (const std::exception& error) {
        g_last_error = std::string("Einsums einsum threw: ") + error.what();
        return PYQCEINSUMS_STATUS_INTERNAL;
    } catch (...) {
        g_last_error = "Einsums einsum threw a non-standard exception";
        return PYQCEINSUMS_STATUS_INTERNAL;
    }
#else
    // The stub validates everything above (that is the point: the ABI contract
    // is provable without Einsums) and then refuses to compute.
    (void)inputs;
    (void)num_ops;
    (void)rank;
    (void)out_ndim;
    return unavailable(kNoEinsums);
#endif
}

}  // extern "C"
