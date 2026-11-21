"""CUDA topology entry: descriptor decoding and the ctypes bridge.

Two byte streams reach ``qcd_cuda_topology_apply``:

* a **version-2 execution plan** (``pyqcd/einsum/_native_plan.py``): full step list,
  used by the frozen entries and by callers that already have a path;
* a **topology descriptor** (``_kernel_cuda._topology_descriptor``): structure only —
  per-operand labels/shapes, output labels and the 128-bit signature digest. The
  contraction order comes from the native precompiled step table
  (``cpp/cuda/einsum/src/topology_steps.h``), so no path search runs at all.
"""

from __future__ import annotations

import ctypes
import os
import stat
import threading
from collections import namedtuple
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .._native_plan import decode_execution_plan, dtype_from_id

_CUDA_LIBRARY_ENV = "PYQCD_EINSUM_CUDA_LIB"

# Frozen status codes plus the topology entry's own QCD_TOPOLOGY_UNSUPPORTED,
# kept numerically identical to ``cpp/cuda/einsum/include/einsum_topology.h``.
_STATUS_OK = 0
_TOPOLOGY_UNSUPPORTED = 4

# Descriptor layout; must stay in sync with ``apply_contract.cu``.
_DESCRIPTOR_MAGIC = 0x51435444
_DESCRIPTOR_VERSION = 1
_DESCRIPTOR_HEADER_WORDS = 8
_DESCRIPTOR_DIGEST_WORDS = 4
_REQUIRED_CUDA_EXPORTS = (
    "applyPlanQcd",
    "applyContractQcd",
    "applyCompileQcd",
    "applyCompiledQcd",
    "applyDestroyQcd",
    "applyInitQcd",
    "applyEndQcd",
    "qcd_cuda_topology_apply",
    "qcd_cuda_dedicated_apply",
    "qcd_cuda_dedicated_dtypes",
)

DescriptorView = namedtuple(
    "DescriptorView",
    "dtype_id input_shapes input_labels output_labels sizes steps")


def _library_path() -> Path:
    """Return the CUDA archive to load (``PYQCD_EINSUM_CUDA_LIB`` wins)."""
    override = os.environ.get(_CUDA_LIBRARY_ENV, "").strip()
    if override:
        return Path(override)
    from .._native import _cuda_library_path

    return Path(_cuda_library_path())


def _bind_host_cublas() -> None:
    """让宿主（cupy / torch）先把它自己的 ``libcublas`` 绑进本进程。

    背景：CUDA 归档带绝对 RUNPATH（conda 的编译器 wrapper 注入），其中 ``NEEDED``
    是 ``libcublas.so.12``。如果先 dlopen 归档，动态加载器会按 RUNPATH 找到**构建
    机那份**（例如 12.4），于是进程里同时存在两份 cuBLAS。实测症状：调用过拓扑/
    步表入口之后，cupy 的 **complex64** GEMM 恒报 ``CUBLAS_STATUS_NOT_SUPPORTED``
    （c128/f32/f64 不受影响），看起来像"内核算错"，其实是两份 cuBLAS 抢同一份
    进程级状态。

    先让宿主加载它自己那份，``libcublas.so.12`` 这个 soname 就已有映射，归档的
    ``NEEDED`` 会复用它，不会拉第二份。任何一步失败都静默跳过——这只是"尽量避开
    双份"，不是正确性依赖。
    """
    for importer in (
        lambda: __import__("cupy_backends.cuda.libs.cublas", fromlist=["cublas"]),
        lambda: __import__("torch", fromlist=["torch"]),
    ):
        try:
            importer()
            return
        except Exception:  # noqa: BLE001 - 宿主不存在就换下一个
            continue


class TopologyBindingError(RuntimeError):
    """Expected binding failure carrying the exact missing export names."""

    def __init__(self, path: Path, missing: Sequence[str]) -> None:
        self.path = Path(path)
        self.missing = tuple(str(name) for name in missing)
        super().__init__(
            f"CUDA topology archive is missing required exports at {self.path}: "
            f"{', '.join(self.missing)}")


class TopologyBinding:
    """ctypes bridge for ``qcd_cuda_topology_apply``.

    Deliberately outside the Cython extension: the topology entry is an extra
    exported symbol (like ``qcd_cuda_dedicated_apply``), so no rebuild of
    ``pyqcd.einsum.cuda.einsum`` is required for it to become usable.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else _library_path()
        _bind_host_cublas()
        self.library = ctypes.CDLL(str(self.path))
        missing = []
        for name in _REQUIRED_CUDA_EXPORTS:
            try:
                getattr(self.library, name)
            except AttributeError:
                missing.append(name)
        if missing:
            raise TopologyBindingError(self.path, missing)
        self.apply = self.library.qcd_cuda_topology_apply
        self.apply.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.apply.restype = ctypes.c_int


_binding_lock = threading.Lock()
_binding_stamps: dict[str, tuple[Any, ...]] = {}
_bindings: dict[str, TopologyBinding] = {}
_restart_required: dict[str, dict[str, Any]] = {}


def _absolute_path(path: Path | str) -> Path:
    """Return a stable absolute path for diagnostics and cache identity."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _binding_for_target_locked(
        target: Path) -> tuple[TopologyBinding | None, dict[str, Any], bool]:
    """Resolve one path while holding ``_binding_lock`` and return its snapshot."""
    key = str(target)
    try:
        file_stat = os.stat(target)
    except FileNotFoundError:
        stamp = (key, "missing")
        present = False
    except OSError as error:
        _bindings.pop(key, None)
        _binding_stamps.pop(key, None)
        failure = {
            "path": key,
            "present": False,
            "status": "load-failed",
            "detail": (
                f"CUDA topology archive could not be inspected at {target}: "
                f"{type(error).__name__}: {error}"
            ),
        }
        return None, failure, False
    else:
        stamp = (
            key,
            int(file_stat.st_dev),
            int(file_stat.st_ino),
            int(file_stat.st_size),
            int(file_stat.st_mtime_ns),
        )
        present = stat.S_ISREG(file_stat.st_mode)

    restarted = _restart_required.get(key)
    if restarted is not None:
        report = dict(restarted)
        report["present"] = present
        return None, report, present

    binding = _bindings.get(key)
    previous_stamp = _binding_stamps.get(key)
    if binding is not None:
        if previous_stamp == stamp:
            return binding, {
                "path": key,
                "present": True,
                "status": "ready",
                "detail": f"CUDA topology archive loaded from {key}",
            }, present
        # dlopen normally reuses a library already loaded by absolute path.
        # Never keep reporting success if this path no longer names its archive.
        _bindings.pop(key, None)
        _binding_stamps.pop(key, None)
        failure = {
            "path": key,
            "present": present,
            "status": "load-failed",
            "detail": (
                f"CUDA topology archive changed at {target}; restart the Python "
                "process before using the replacement archive"
            ),
        }
        _restart_required[key] = dict(failure)
        return None, failure, present

    _binding_stamps[key] = stamp
    failure = {
        "path": key,
        "present": present,
        "status": "checking",
        "detail": f"checking CUDA topology archive at {target}",
    }
    if not present:
        failure = {
            "path": key,
            "present": False,
            "status": "missing",
            "detail": (
                f"CUDA topology archive is missing at {target}; "
                "reinstall pyqcd or set PYQCD_EINSUM_CUDA_LIB"
            ),
        }
        return None, failure, present
    try:
        loaded = TopologyBinding(target)
    except OSError as error:
        failure = {
            "path": key,
            "present": True,
            "status": "load-failed",
            "detail": (
                f"CUDA topology archive could not be loaded at {target}: "
                f"{type(error).__name__}: {error}"
            ),
        }
    except TopologyBindingError as error:
        failure = {
            "path": key,
            "present": True,
            "status": "symbol-missing",
            "detail": (
                f"{error}; rebuild pyqcd with "
                "cpp/cuda/einsum/make-nv.sh"
            ),
        }
    else:
        _bindings[key] = loaded
        failure = {
            "path": key,
            "present": True,
            "status": "ready",
            "detail": f"CUDA topology archive loaded from {target}",
        }
    return _bindings.get(key), failure, present


def topology_binding_snapshot(
        path: Path | None = None) -> tuple[TopologyBinding | None, dict[str, Any]]:
    """Return a binding and its diagnostic snapshot from one locked check."""
    target = _absolute_path(path if path is not None else _library_path())
    with _binding_lock:
        binding, report, _present = _binding_for_target_locked(target)
    report["exported"] = binding is not None
    return binding, report


def topology_binding(path: Path | None = None) -> TopologyBinding | None:
    """Return the cached binding, or ``None`` when the archive lacks the symbol."""
    return topology_binding_snapshot(path)[0]


def topology_binding_report(path: Path | None = None) -> dict[str, Any]:
    """Describe whether the selected archive exists, loads, and exports the entry."""
    _binding, report = topology_binding_snapshot(path)
    return report


_PLAN_VIEW_LOCK = threading.Lock()
_PLAN_VIEW_CACHE: dict[bytes, tuple[Any, Any, tuple, tuple]] = {}
_PLAN_VIEW_CACHE_LIMIT = 32


def _plan_buffer(plan: Any) -> np.ndarray:
    if isinstance(plan, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(plan, dtype=np.uint8)
    else:
        array = np.asarray(plan)
        if array.dtype == np.dtype(np.uint8):
            buffer = np.ascontiguousarray(array).reshape(-1)
        elif array.dtype.kind in "iu":
            buffer = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        else:
            raise TypeError("execution plan must be a bytes-like or integer array")
    return np.ascontiguousarray(buffer)


def _decode_descriptor(plan_data: np.ndarray) -> DescriptorView:
    """Decode the structure-only descriptor (see ``apply_contract.cu``)."""
    words = np.frombuffer(plan_data, dtype='<i4')
    if words.size < _DESCRIPTOR_HEADER_WORDS + _DESCRIPTOR_DIGEST_WORDS:
        raise ValueError("topology descriptor is truncated")
    magic, version, total_bytes, local_dtype, num_inputs, num_labels, num_output, _ = (
        int(value) for value in words[:8])
    if magic != _DESCRIPTOR_MAGIC:
        raise ValueError(f"unexpected topology descriptor magic 0x{magic:08x}")
    if version != _DESCRIPTOR_VERSION:
        raise ValueError(f"unsupported topology descriptor version {version}")
    if total_bytes != plan_data.size:
        raise ValueError(
            f"topology descriptor length {total_bytes} != {plan_data.size} bytes")
    if num_inputs <= 0 or num_labels < 0 or not 0 <= num_output <= num_labels:
        raise ValueError("topology descriptor header is inconsistent")
    offset = _DESCRIPTOR_HEADER_WORDS + _DESCRIPTOR_DIGEST_WORDS
    input_labels: list[tuple[int, ...]] = []
    input_shapes: list[tuple[int, ...]] = []
    for _ in range(num_inputs):
        if offset >= words.size:
            raise ValueError("topology descriptor ends before its operands")
        rank = int(words[offset])
        offset += 1
        if rank <= 0 or offset + 2 * rank > words.size:
            raise ValueError(f"topology descriptor rank {rank} is out of range")
        labels = tuple(int(value) for value in words[offset:offset + rank])
        offset += rank
        shape = tuple(int(value) for value in words[offset:offset + rank])
        offset += rank
        if any(label < 0 or label >= num_labels for label in labels):
            raise ValueError("topology descriptor carries an out-of-range label")
        if any(extent <= 0 for extent in shape):
            raise ValueError("topology descriptor carries a non-positive extent")
        input_labels.append(labels)
        input_shapes.append(shape)
    if offset + num_output != words.size:
        raise ValueError("topology descriptor has trailing bytes")
    output_labels = tuple(int(value) for value in words[offset:offset + num_output])

    sizes = [-1] * num_labels
    for labels, shape in zip(input_labels, input_shapes):
        for label, extent in zip(labels, shape):
            sizes[label] = max(sizes[label], extent)
    if any(value <= 0 for value in sizes):
        raise ValueError("topology descriptor omits the extent of a label")
    return DescriptorView(
        dtype_id=local_dtype,
        input_shapes=tuple(input_shapes),
        input_labels=tuple(input_labels),
        output_labels=output_labels,
        sizes=tuple(sizes),
        steps=num_inputs - 1,
    )


def _plan_view(plan_data: np.ndarray) -> tuple[Any, Any, tuple, tuple]:
    """Decode and validate a plan or descriptor once per distinct byte stream."""
    key = plan_data.tobytes()
    with _PLAN_VIEW_LOCK:
        cached = _PLAN_VIEW_CACHE.get(key)
    if cached is not None:
        return cached
    words = np.frombuffer(plan_data, dtype='<i4')
    if words.size and int(words[0]) == _DESCRIPTOR_MAGIC:
        descriptor = _decode_descriptor(plan_data)
        view = (
            descriptor,
            dtype_from_id(descriptor.dtype_id),
            tuple(descriptor.input_shapes),
            tuple(descriptor.sizes[label] for label in descriptor.output_labels),
        )
    else:
        info = decode_execution_plan(plan_data)
        view = (
            info,
            dtype_from_id(info.dtype_id),
            tuple(info.input_shapes),
            tuple(info.sizes[label] for label in info.output_labels),
        )
    with _PLAN_VIEW_LOCK:
        if len(_PLAN_VIEW_CACHE) >= _PLAN_VIEW_CACHE_LIMIT:
            _PLAN_VIEW_CACHE.pop(next(iter(_PLAN_VIEW_CACHE)))
        _PLAN_VIEW_CACHE[key] = view
    return view


def topology_apply(
    plan: Any,
    operands: Sequence[Any],
    out: Any = None,
    offsets: Sequence[int] | None = None,
    library_path: Path | str | None = None,
    info: dict[str, Any] | None = None,
) -> Any:
    """Run one native call for a version-2 plan **or** a topology descriptor.

    ``operands`` are CuPy arrays already on the compute device, in plan order;
    ``offsets`` are optional **byte** offsets added to each device pointer so a
    slice of a big tensor can be addressed without copying it. One native call
    runs every step: the entry only reads the device pointers it is given and
    writes the result once, so no host staging happens here either.

    dtype, shape and device must match the plan exactly — nothing is converted
    implicitly. A non-contiguous operand is staged into a contiguous device copy
    (the index is recorded in ``info``) because the entry never interprets
    strides; ``out`` must already be contiguous, otherwise the native write
    would land somewhere the caller cannot see.

    ``info`` (optional dict) is cleared and filled with what was actually sent
    down: plan byte count, step count, dtype id, operand count, output shape,
    whether ``out`` was caller-provided, which operands had to be copied and the
    raw ``offsets``.

    Raises ``NativeUnavailable`` when the entry reports a non-zero status;
    ``status == _TOPOLOGY_UNSUPPORTED`` means the plan itself (not the arguments)
    is outside what the entry can execute.
    """
    import cupy

    from ..kupl.einsum import NativeUnavailable

    binding, report = topology_binding_snapshot(
        None if library_path is None else Path(library_path))
    if binding is None:
        raise NativeUnavailable(report["detail"])

    plan_data = _plan_buffer(plan)
    decoded, native_dtype, input_shapes, output_shape = _plan_view(plan_data)
    arrays = list(operands)
    if len(arrays) != len(input_shapes):
        raise ValueError(f"plan expects {len(input_shapes)} operands, got {len(arrays)}")
    device_id = int(cupy.cuda.Device().id)
    device_arrays = []
    copied = []
    for index, (operand, shape) in enumerate(zip(arrays, input_shapes)):
        array = cupy.asarray(operand)
        if array.dtype != native_dtype:
            raise ValueError(
                f"operand {index} dtype {array.dtype} does not match plan dtype {native_dtype}")
        if tuple(array.shape) != tuple(shape):
            raise ValueError(
                f"operand {index} shape {tuple(array.shape)} does not match plan shape {tuple(shape)}")
        if int(array.device.id) != device_id:
            raise ValueError(
                f"operand {index} is on CUDA device {int(array.device.id)}, expected {device_id}")
        if not array.flags['C_CONTIGUOUS']:
            copied.append(index)
        device_arrays.append(cupy.ascontiguousarray(array))

    if out is None:
        output = cupy.empty(output_shape, dtype=native_dtype)
    else:
        output = cupy.asarray(out)
        if output.dtype != native_dtype:
            raise ValueError(f"output dtype {output.dtype} does not match plan dtype {native_dtype}")
        if tuple(output.shape) != output_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} does not match plan shape {output_shape}")
        if not output.flags['C_CONTIGUOUS']:
            raise ValueError("output buffer must be C-contiguous")
        if int(output.device.id) != device_id:
            raise ValueError(
                f"output is on CUDA device {int(output.device.id)}, expected {device_id}")

    pointers = (ctypes.c_void_p * len(device_arrays))(
        *[int(array.data.ptr) for array in device_arrays])
    offset_array = None
    raw_offsets = None
    if offsets is not None:
        raw_offsets = tuple(int(value) for value in offsets)
        if len(raw_offsets) != len(device_arrays):
            raise ValueError(
                f"offsets must have {len(device_arrays)} entries, got {len(raw_offsets)}")
        offset_array = (ctypes.c_int64 * len(raw_offsets))(*raw_offsets)

    if info is not None:
        step_count = (int(decoded.steps) if isinstance(decoded.steps, (int, np.integer))
                      else len(decoded.steps))
        info.clear()
        info.update(
            dtype=str(native_dtype),
            dtype_id=int(decoded.dtype_id),
            n_operands=len(device_arrays),
            plan_bytes=int(plan_data.size),
            steps=step_count,
            output_shape=tuple(output_shape),
            out="allocated" if out is None else "provided",
            copied=tuple(copied),
            offsets=raw_offsets,
        )

    # CuPy issues conversions on its own stream while the entry launches on the
    # default stream; wait for that stream so no operand is still being written.
    cupy.cuda.get_current_stream().synchronize()
    status = int(
        binding.apply(
            int(plan_data.ctypes.data),
            int(plan_data.size),
            pointers,
            len(device_arrays),
            offset_array,
            int(output.data.ptr),
            int(decoded.dtype_id),
            None,
        )
    )
    if status != _STATUS_OK:
        reason = (
            " (plan not executable as version-2 steps)"
            if status == _TOPOLOGY_UNSUPPORTED
            else ""
        )
        # 状态码随身带出（1/3 = 参数或内部错误，4 = 这条计划本入口不认）。
        error = NativeUnavailable(
            f"qcd_cuda_topology_apply failed with status {status}{reason}")
        error.status = status
        raise error
    return output
