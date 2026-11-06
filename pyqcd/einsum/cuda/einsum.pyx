# cython: language_level=3
# cython: binding=True
"""Thin Cython bridge to ``libpyqcdeinsum``.

The raw planner and contract entry points exchange addresses with the C ABI.
Python callers own the CPU control tables and array lifetimes. ``contract_plan``
only keeps temporary contiguous arrays alive for the duration of the call.
"""

import operator

from pyqcd.einsum.cuda.einsum_api cimport (
    applyCompileQcd as _c_applyCompileQcd,
    applyCompiledQcd as _c_applyCompiledQcd,
    applyContractQcd as _c_applyContractQcd,
    applyDestroyQcd as _c_applyDestroyQcd,
    applyEndQcd as _c_applyEndQcd,
    applyInitQcd as _c_applyInitQcd,
    applyPlanQcd as _c_applyPlanQcd,
)

__all__ = [
    "applyCompileQcd",
    "applyCompiledQcd",
    "applyContractQcd",
    "applyDestroyQcd",
    "applyEndQcd",
    "applyInitQcd",
    "applyPlanQcd",
    "backend_info",
    "contract_plan",
]

QCD_STATUS_OK = 0
QCD_STATUS_INVALID = 1
QCD_STATUS_MEMORY = 2
QCD_STATUS_INTERNAL = 3

STATUS_NAMES = (
    "QCD_STATUS_OK",
    "QCD_STATUS_INVALID",
    "QCD_STATUS_MEMORY",
    "QCD_STATUS_INTERNAL",
)

_MAX_ADDRESS = (1 << 63) - 1


def _address(value, str name):
    """Coerce one ABI argument to a non-null signed 64-bit address."""
    try:
        address = operator.index(value)
    except TypeError:
        raise TypeError(
            f"{name} must be an integer address, got {type(value).__name__}") from None
    if address <= 0:
        raise ValueError(f"{name} must be a non-null address, got {address}")
    if address > _MAX_ADDRESS:
        raise OverflowError(f"{name} does not fit in a signed 64-bit address: {address}")
    return address


def applyPlanQcd(set_ptrs, params):
    """Plan the contraction described by ``set_ptrs``; return ``QCD_STATUS_*``."""
    cdef long long c_set_ptrs = _address(set_ptrs, "set_ptrs")
    cdef long long c_params = _address(params, "params")
    cdef int status
    with nogil:
        status = _c_applyPlanQcd(c_set_ptrs, c_params)
    return status


def applyContractQcd(plan, tensors, out, params):
    """Execute an encoded kernel plan; return ``QCD_STATUS_*``."""
    cdef long long c_plan = _address(plan, "plan")
    cdef long long c_tensors = _address(tensors, "tensors")
    cdef long long c_out = _address(out, "out")
    cdef long long c_params = _address(params, "params")
    cdef int status
    with nogil:
        status = _c_applyContractQcd(c_plan, c_tensors, c_out, c_params)
    return status


def applyCompileQcd(plan, params, out_context):
    """Compile a plan into a reusable native CUDA execution context."""
    cdef long long c_plan = _address(plan, "plan")
    cdef long long c_params = _address(params, "params")
    cdef long long c_out_context = _address(out_context, "out_context")
    cdef int status
    with nogil:
        status = _c_applyCompileQcd(c_plan, c_params, c_out_context)
    return status


def applyCompiledQcd(context, tensors, out, params):
    """Execute a plan through a context returned by :func:`applyCompileQcd`."""
    cdef long long c_context = _address(context, "context")
    cdef long long c_tensors = _address(tensors, "tensors")
    cdef long long c_out = _address(out, "out")
    cdef long long c_params = _address(params, "params")
    cdef int status
    with nogil:
        status = _c_applyCompiledQcd(c_context, c_tensors, c_out, c_params)
    return status


def applyDestroyQcd(context):
    """Destroy a native CUDA execution context; zero is accepted as a no-op."""
    try:
        address = operator.index(context)
    except TypeError:
        raise TypeError(
            f"context must be an integer address, got {type(context).__name__}") from None
    if address < 0:
        raise ValueError(f"context must be a non-negative address, got {address}")
    if address > _MAX_ADDRESS:
        raise OverflowError(f"context does not fit in a signed 64-bit address: {address}")
    cdef long long c_context = address
    with nogil:
        _c_applyDestroyQcd(c_context)


def applyInitQcd(set_ptrs, params):
    """Call the backend session-initialisation hook."""
    cdef long long c_set_ptrs = _address(set_ptrs, "set_ptrs")
    cdef long long c_params = _address(params, "params")
    with nogil:
        _c_applyInitQcd(c_set_ptrs, c_params)


def applyEndQcd(set_ptrs, params):
    """Call the backend session-teardown hook."""
    cdef long long c_set_ptrs = _address(set_ptrs, "set_ptrs")
    cdef long long c_params = _address(params, "params")
    with nogil:
        _c_applyEndQcd(c_set_ptrs, c_params)


def backend_info():
    """Identify the compiled backend without touching the native library."""
    return {"backend": "cuda", "compiled": True}


def _host_plan(plan_data):
    import numpy as np

    if isinstance(plan_data, (bytes, bytearray, memoryview)):
        return np.frombuffer(plan_data, dtype=np.uint8)
    array = np.asarray(plan_data)
    if array.dtype == np.dtype(np.uint8):
        return np.ascontiguousarray(array).reshape(-1)
    if array.dtype.kind in "iu" and array.dtype.itemsize in (1, 2, 4, 8):
        return np.ascontiguousarray(array).view(np.uint8).reshape(-1)
    raise TypeError("execution plan must be bytes-like or an integer array")


def contract_plan(plan_data, arrays, out=None):
    """Execute a version-2 plan with CuPy device-buffer raw pointers."""
    import numpy as np
    from pyqcd.einsum._native_plan import decode_execution_plan, dtype_from_id

    try:
        import cupy
    except ImportError as error:
        raise ImportError("contract_plan requires CuPy") from error

    info = decode_execution_plan(plan_data)
    if len(arrays) != len(info.input_shapes):
        raise ValueError(
            f"plan expects {len(info.input_shapes)} arrays, got {len(arrays)}")

    dtype = dtype_from_id(info.dtype_id)
    device_arrays = []
    for index, (array, shape) in enumerate(zip(arrays, info.input_shapes)):
        device_array = cupy.asarray(array)
        if device_array.dtype != dtype:
            raise ValueError(
                f"input {index} dtype {device_array.dtype} does not match plan dtype {dtype}")
        if tuple(device_array.shape) != shape:
            raise ValueError(
                f"input {index} shape {device_array.shape} does not match plan shape {shape}")
        device_arrays.append(cupy.ascontiguousarray(device_array))

    output_shape = tuple(info.sizes[label] for label in info.output_labels)
    output = cupy.empty(output_shape, dtype=dtype) if out is None else cupy.asarray(out)
    if output.dtype != dtype:
        raise ValueError(
            f"output dtype {output.dtype} does not match plan dtype {dtype}")
    if tuple(output.shape) != output_shape:
        raise ValueError(
            f"output shape {output.shape} does not match plan shape {output_shape}")
    output_contiguous = cupy.ascontiguousarray(output)

    plan_buffer = _host_plan(plan_data)
    pointer_table = np.ascontiguousarray(
        [int(array.data.ptr) for array in device_arrays], dtype=np.int64)
    output_pointer = np.ascontiguousarray([int(output_contiguous.data.ptr)], dtype=np.int64)
    params = np.zeros(32, dtype=np.int32)
    params[0] = len(device_arrays)
    params[1] = max((len(shape) for shape in info.input_shapes), default=0)
    params[8] = info.dtype_id

    keepalive = (plan_buffer, device_arrays, pointer_table, output_pointer, params,
                 output_contiguous)
    cupy.cuda.Device().synchronize()
    try:
        status = applyContractQcd(
            _address(plan_buffer.ctypes.data, "plan"),
            _address(pointer_table.ctypes.data, "tensors"),
            _address(output_pointer.ctypes.data, "out"),
            _address(params.ctypes.data, "params"),
        )
    finally:
        cupy.cuda.Device().synchronize()
    if status != QCD_STATUS_OK:
        from pyqcd.einsum.kupl.einsum import NativeUnavailable
        raise NativeUnavailable(f"native CUDA contraction failed with status {status}")

    if output_contiguous is not output:
        output[...] = output_contiguous
    del keepalive
    return output
