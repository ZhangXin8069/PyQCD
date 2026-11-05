"""ctypes binding for the native einsum planning kernel.

``params`` is ``int32[32]`` and ``set_ptrs`` is ``int64[16]``; both mirror
``cpp/einsum/include/einsum_define.h``. The kernel receives one flat int32 index
array per call plus an int64 offset table, the output index list, the per-symbol
size table, and two output arrays: the linear path as ``(i, j)`` pairs and
``{opt_cost, largest_intermediate}``.

Loading is lazy: importing this module does not touch the filesystem beyond
resolving the default path, and the first :func:`plan` call performs the
``CDLL``. A missing or unloadable archive raises the private
:class:`_NativeUnavailable`, which :func:`plan` turns into ``None`` so the
caller can fall back to the pure Python planner -- no diagnostics are printed.

Index symbols are non-negative integers that index ``sizes`` (a mapping is
accepted too and renamed to dense ids); the array lengths the kernel cannot
infer travel in the reserved ``set_ptrs`` slots ``[8:11]``.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_LIB = _REPO / "cpp" / "kupl" / "einsum" / "libpyqcdeinsum.so"

_PARAM_SLOTS = 32
_SET_SLOTS = 16
_STATUS_OK = 0
_MAX_TENSORS = 1 << 16

# set_ptrs / params slot indices, kept in sync with einsum_define.h.
_SET_INPUTS_FLAT = 0
_SET_INPUT_OFFSETS = 1
_SET_OUTPUT = 2
_SET_SIZES = 3
_SET_OUT_PATH = 4
_SET_OUT_COST = 5
_SET_SCRATCH = 6
_SET_SCRATCH_SIZE = 7
_SET_NUM_SYMBOLS = 8
_SET_OUTPUT_NDIM = 9
_SET_PATH_CAPACITY = 10
_PARAM_NUM_OPS = 0
_PARAM_MAX_NDIM = 1
_PARAM_PLANNER_ID = 2
_PARAM_MEM_HI = 3
_PARAM_MEM_LO = 4
_PARAM_MEM_UNLIMITED = -1


class _NativeUnavailable(RuntimeError):
    """Raised when the native archive cannot be loaded; never user facing."""


def library_path() -> Path:
    """Return the archive this module would load (``PYQCD_EINSUM_LIB`` wins)."""
    requested = os.environ.get("PYQCD_EINSUM_LIB")
    return Path(requested).expanduser() if requested else _DEFAULT_LIB


_LL = ctypes.c_longlong
_I = ctypes.c_int

_SIGNATURES = {
    "applyPlanQcd": (_LL, _LL),
    "applyContractQcd": (_LL, _LL, _LL, _LL),
    "applyInitQcd": (_LL, _LL),
    "applyEndQcd": (_LL, _LL),
}

_STATUS_FUNCTIONS = {"applyPlanQcd", "applyContractQcd"}

_LIBRARY: Any = None
_LOAD_FAILURE: str | None = None


def _load_library() -> Any:
    """Load the archive once and bind every signature the build exports."""
    global _LIBRARY, _LOAD_FAILURE
    if _LIBRARY is not None:
        return _LIBRARY
    if _LOAD_FAILURE is not None:
        raise _NativeUnavailable(_LOAD_FAILURE)
    path = library_path()
    if not path.is_file():
        _LOAD_FAILURE = f"native einsum library not found at {path}"
        raise _NativeUnavailable(_LOAD_FAILURE)
    try:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    except OSError as exc:
        _LOAD_FAILURE = f"native einsum library failed to load: {exc}"
        raise _NativeUnavailable(_LOAD_FAILURE) from exc
    for name, argtypes in _SIGNATURES.items():
        try:
            function = getattr(library, name)
        except AttributeError:
            continue
        function.argtypes = argtypes
        function.restype = ctypes.c_int if name in _STATUS_FUNCTIONS else None
    _LIBRARY = library
    return library


def available() -> bool:
    """Return whether the native planner can be loaded; never raises."""
    try:
        _load_library()
    except _NativeUnavailable:
        return False
    return True


def _address(buffer: np.ndarray) -> int:
    """Address of an already C-contiguous buffer for a raw ctypes call."""
    return int(np.ascontiguousarray(buffer).ctypes.data)


def _encode_indices(values: Sequence[int]) -> np.ndarray:
    return np.ascontiguousarray(list(values), dtype=np.int32).reshape(-1)


def _dense_symbols(
    inputs: Any, output: Sequence[str], sizes: Mapping[str, int]
) -> tuple[list, list, list]:
    """Rename arbitrary symbol names to dense int ids in first-appearance order."""
    order: dict[str, int] = {}
    for term in list(inputs) + [output]:
        for symbol in term:
            if symbol not in order:
                if symbol not in sizes:
                    raise KeyError(symbol)
                order[symbol] = len(order)
    return (
        [[order[symbol] for symbol in term] for term in inputs],
        [order[symbol] for symbol in output],
        [sizes[symbol] for symbol in order],
    )


def _encode_memory_limit(memory_limit: int | None) -> tuple[int, int]:
    if memory_limit is None or memory_limit == -1:
        return _PARAM_MEM_UNLIMITED, 0
    limit = int(memory_limit)
    if limit < 0:
        return _PARAM_MEM_UNLIMITED, 0
    high = min(limit >> 32, 0x7FFFFFFF)
    return high, limit & 0xFFFFFFFF


def _plan_with(
    function,
    inputs: Sequence[Sequence[int]],
    output: Sequence[int],
    sizes: Sequence[int] | Mapping[Any, int],
    planner_id: int = 0,
    memory_limit: int | None = None,
) -> tuple[tuple[tuple[int, ...], ...], int, int] | None:
    """Plan with an already-resolved native entry point (ctypes or Cython)."""
    try:
        if isinstance(sizes, Mapping):
            inputs, output, sizes = _dense_symbols(inputs, output, sizes)
        num_ops = len(inputs)
        if num_ops > _MAX_TENSORS:
            return None
        tensors = [_encode_indices(item) for item in inputs]
        flat = (
            np.ascontiguousarray(np.concatenate(tensors), dtype=np.int32)
            if tensors
            else np.empty(0, dtype=np.int32)
        )
        offsets = np.zeros(num_ops + 1, dtype=np.int64)
        if num_ops:
            ndims = np.fromiter((item.size for item in tensors), dtype=np.int64, count=num_ops)
            np.cumsum(ndims, out=offsets[1:])
        output_index = _encode_indices(output)
        symbol_sizes = np.ascontiguousarray(sizes, dtype=np.int64).reshape(-1)
        num_symbols = int(symbol_sizes.size)
        if output_index.size > num_symbols:
            return None

        path_capacity = 2 * (num_ops - 1) if num_ops >= 2 else 0
        out_path = np.zeros(max(path_capacity, 1), dtype=np.int32)
        out_cost = np.zeros(2, dtype=np.int64)

        set_ptrs = np.zeros(_SET_SLOTS, dtype=np.int64)
        set_ptrs[_SET_INPUTS_FLAT] = _address(flat)
        set_ptrs[_SET_INPUT_OFFSETS] = _address(offsets)
        set_ptrs[_SET_OUTPUT] = _address(output_index)
        set_ptrs[_SET_SIZES] = _address(symbol_sizes)
        set_ptrs[_SET_OUT_PATH] = _address(out_path)
        set_ptrs[_SET_OUT_COST] = _address(out_cost)
        set_ptrs[_SET_NUM_SYMBOLS] = num_symbols
        set_ptrs[_SET_OUTPUT_NDIM] = int(output_index.size)
        set_ptrs[_SET_PATH_CAPACITY] = path_capacity

        params = np.zeros(_PARAM_SLOTS, dtype=np.int32)
        params[_PARAM_NUM_OPS] = num_ops
        params[_PARAM_MAX_NDIM] = max((int(item.size) for item in tensors), default=0)
        params[_PARAM_PLANNER_ID] = int(planner_id)
        params[_PARAM_MEM_HI], params[_PARAM_MEM_LO] = _encode_memory_limit(memory_limit)

        status = int(function(_address(set_ptrs), _address(params)))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if status != _STATUS_OK:
        return None

    if path_capacity:
        pairs = out_path[:path_capacity].reshape(-1, 2)
        path = tuple((int(first), int(second)) for first, second in pairs)
    elif num_ops == 1:
        path = ((0,),)
    else:
        path = ()
    return path, int(out_cost[0]), int(out_cost[1])
