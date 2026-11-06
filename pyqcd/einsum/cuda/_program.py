"""Reusable CUDA execution context for version-2 einsum plans."""

from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

from .._native_plan import decode_execution_plan, dtype_from_id


class CudaProgram:
    """Compile a plan once and reuse its CUDA workspace and control buffers."""

    def __init__(self, plan_data: bytes) -> None:
        import cupy

        from . import einsum

        self._cupy = cupy
        self._einsum = einsum
        self.device_id = int(cupy.cuda.Device().id)
        self.plan_data = np.frombuffer(plan_data, dtype=np.uint8)
        self.info = decode_execution_plan(plan_data)
        self.dtype = dtype_from_id(self.info.dtype_id)
        self.input_shapes = tuple(self.info.input_shapes)
        self.output_shape = tuple(
            self.info.sizes[label] for label in self.info.output_labels
        )
        self.params = np.zeros(32, dtype=np.int32)
        self.params[0] = len(self.input_shapes)
        self.params[1] = max(map(len, self.input_shapes), default=0)
        self.params[8] = self.info.dtype_id
        self.context = np.empty(1, dtype=np.int64)
        self._local = threading.local()
        with cupy.cuda.Device(self.device_id):
            status = einsum.applyCompileQcd(
                self.plan_data.ctypes.data,
                self.params.ctypes.data,
                self.context.ctypes.data,
            )
        if status != 0 or int(self.context[0]) == 0:
            from ..kupl.einsum import NativeUnavailable

            raise NativeUnavailable(f"native CUDA compile failed with status {status}")

    def __del__(self) -> None:
        try:
            if getattr(self, "context", None) is not None and int(self.context[0]) != 0:
                with self._cupy.cuda.Device(self.device_id):
                    self._einsum.applyDestroyQcd(int(self.context[0]))
                self.context[0] = 0
        except Exception:
            pass

    def __call__(self, arrays: Sequence[Any], out: Any = None) -> Any:
        with self._cupy.cuda.Device(self.device_id):
            return self._call(arrays, out=out)

    def _call(self, arrays: Sequence[Any], out: Any = None) -> Any:
        cupy = self._cupy
        if len(arrays) != len(self.input_shapes):
            raise ValueError(
                f"plan expects {len(self.input_shapes)} arrays, got {len(arrays)}"
            )

        array_key = (
            tuple(id(array) for array in arrays),
            tuple(getattr(array, "shape", None) for array in arrays),
            tuple(getattr(array, "dtype", None) for array in arrays),
        )
        state = getattr(self._local, "state", None)
        if state is None or state[0] != array_key:
            device_arrays = []
            for index, (array, shape) in enumerate(zip(arrays, self.input_shapes)):
                device_array = cupy.asarray(array)
                if int(device_array.device.id) != self.device_id:
                    raise ValueError(
                        f"input {index} is on CUDA device {device_array.device.id}, "
                        f"expected {self.device_id}"
                    )
                if device_array.dtype != self.dtype or tuple(device_array.shape) != shape:
                    raise ValueError(
                        f"input {index} does not match dtype/shape {self.dtype}/{shape}"
                    )
                device_arrays.append(cupy.ascontiguousarray(device_array))
            pointers = np.empty(len(device_arrays), dtype=np.int64)
            for index, array in enumerate(device_arrays):
                pointers[index] = int(array.data.ptr)
            output_pointers = np.empty(1, dtype=np.int64)
            state = (array_key, device_arrays, pointers, output_pointers)
            self._local.state = state
        else:
            _, device_arrays, pointers, output_pointers = state

        output = cupy.empty(self.output_shape, dtype=self.dtype) if out is None else cupy.asarray(out)
        if output.dtype != self.dtype or tuple(output.shape) != self.output_shape:
            raise ValueError("output does not match compiled CUDA plan")
        output_contiguous = cupy.ascontiguousarray(output)
        output_pointers[0] = int(output_contiguous.data.ptr)
        stream = cupy.cuda.get_current_stream()
        stream.synchronize()
        status = self._einsum.applyCompiledQcd(
            int(self.context[0]),
            pointers.ctypes.data,
            output_pointers.ctypes.data,
            self.params.ctypes.data,
        )
        stream.synchronize()
        if status != 0:
            from ..kupl.einsum import NativeUnavailable

            raise NativeUnavailable(f"native CUDA execution failed with status {status}")
        if output_contiguous is not output:
            output[...] = output_contiguous
        return output
