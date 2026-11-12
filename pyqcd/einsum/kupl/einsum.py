"""ctypes bindings for the native einsum planning and execution kernels.

``params`` is ``int32[32]`` and ``set_ptrs`` is ``int64[16]``; both mirror
``cpp/einsum/include/einsum_define.h``. The kernel receives one flat int32 index
array per call plus an int64 offset table, the output index list, the per-symbol
size table, and two output arrays: the linear path as ``(i, j)`` pairs and
``{opt_cost, largest_intermediate}``.

Loading is lazy: importing this module does not touch the filesystem, and the
first planning or execution call performs the ``CDLL``. A missing archive or
symbol raises :class:`NativeUnavailable`; callers that support a pure Python
fallback catch that exception and continue.

An archive built against MPI needs that runtime in the process first, otherwise
``dlopen`` fails with ``undefined symbol: ompi_mpi_byte``. When the error is an
undefined symbol the loader preloads an MPI runtime once and retries, so the
installation order of the MPI and einsum libraries does not matter;
:func:`library_report` records that retry under ``load_retry``.

Index symbols are non-negative integers that index ``sizes`` (a mapping is
accepted too and renamed to dense ids); the array lengths the kernel cannot
infer travel in the reserved ``set_ptrs`` slots ``[8:11]``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .._native_plan import decode_execution_plan, dtype_from_id
from .mpi import MpiConfigError, make_mpi_executor, mpi_cache_key, normalize_mpi_config
from .multiprocess import (
    MpConfig,
    MpConfigError,
    MpSession,
    MpUnavailable,
    make_mp_executor,
    mp_cache_key,
    normalize_mp_config,
)

_REPO = Path(__file__).resolve().parents[3]
_PACKAGED_LIB = Path(__file__).resolve().with_name("libpyqcdeinsum_kupl.so")
_REPO_LIB = _REPO / "cpp" / "kupl" / "einsum" / "libpyqcdeinsum_kupl.so"

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
_PARAM_DTYPE_ID = 8
_PARAM_LAYOUT_ID = 9
_PARAM_MPI_MODE = 10
_PARAM_MPI_SHARD_SYMBOL = 11
_PARAM_MPI_SHARD_AXIS = 12
_PARAM_MPI_OUTPUT = 13


class NativeUnavailable(RuntimeError):
    """Raised when a requested native einsum operation cannot be executed."""


_NativeUnavailable = NativeUnavailable


def library_path() -> Path:
    """Return the archive this module would load (``PYQCD_EINSUM_LIB`` wins)."""
    requested = os.environ.get("PYQCD_EINSUM_LIB")
    if requested:
        return Path(requested).expanduser()
    if _PACKAGED_LIB.is_file():
        return _PACKAGED_LIB
    if _REPO_LIB.is_file():
        return _REPO_LIB
    # 都没编出来时报告包内规范路径，而不是安装后不存在的 site-packages/cpp 布局。
    return _PACKAGED_LIB


def library_report() -> dict[str, Any]:
    """Describe the archive this module loads, for actionable diagnostics.

    ``status`` names *why* a native call has no reusable compiler context:
    ``library missing`` (nothing at the reported path), ``library failed to
    load`` (present but ``dlopen`` rejected it, e.g. wrong architecture or a
    missing dependency) or ``compiler context unavailable`` (the archive loads
    but exports no usable compiler entry point, or compiling that plan failed).
    The three need different repairs, so callers must not collapse them.

    ``load_retry`` is set when a failed ``dlopen`` was rescued by preloading an
    MPI runtime (``"preloaded libmpi.so.40"``); ``mpi_preload`` describes that
    preload attempt whether it helped or not, and is ``None`` when the archive
    never needed it.
    """
    path = library_path()
    present = path.is_file()
    if not present:
        status = "library missing"
    elif _LOAD_FAILURE_KIND == "load":
        status = "library failed to load"
    else:
        status = "compiler context unavailable"
    return {
        "path": os.path.abspath(str(path)),
        "present": present,
        "status": status,
        "load_failure": _LOAD_FAILURE,
        "load_retry": _LOAD_RETRY,
        "mpi_preload": _mpi_preload_report(),
    }


_LL = ctypes.c_longlong
_I = ctypes.c_int

# qcd_kupl_topology_apply 的补充状态码：计划结构超出 version-2 单步 arity ≤ 2 模型。
# 与 _STATUS_OK(=0) 之外的其它代码一样，调用方一律回落现有路径。
_TOPOLOGY_UNSUPPORTED = 4

_SIGNATURES = {
    "applyPlanQcd": (_LL, _LL),
    "applyContractQcd": (_LL, _LL, _LL, _LL),
    "applyContractMpiQcd": (_LL, _LL, _LL, _LL),
    "applyCompileQcd": (_LL, _LL, _LL),
    "applyCompiledQcd": (_LL, _LL, _LL, _LL),
    "applyDestroyQcd": (_LL,),
    "applyInitQcd": (_LL, _LL),
    "applyEndQcd": (_LL, _LL),
    "qcd_kupl_topology_apply": (
        ctypes.c_void_p,
        _LL,
        ctypes.POINTER(ctypes.c_void_p),
        _I,
        ctypes.POINTER(_LL),
        ctypes.c_void_p,
        _I,
    ),
}

_STATUS_FUNCTIONS = {
    "applyPlanQcd",
    "applyContractQcd",
    "applyContractMpiQcd",
    "applyCompileQcd",
    "applyCompiledQcd",
    "qcd_kupl_topology_apply",
}

_LIBRARY: Any = None
_LOAD_FAILURE: str | None = None
# ``"missing"`` / ``"load"``: 缺库与"库在但 dlopen 失败"的修法不同，报告要能区分。
_LOAD_FAILURE_KIND: str | None = None
_LIBRARY_LOCK = threading.Lock()

# dlopen 只会在两种情况下"装不进去"：归档缺失，或归档依赖的库没进进程。后者最常见的
# 报错是 ``undefined symbol: ompi_mpi_byte``（归档链了 OpenMPI，运行时却不在进程里）。
# 把 MPI 运行时先带进全局作用域再重试一次 dlopen 就能装上，所以这里保留这条最小韧性，
# 其余失败照旧 fail-open：预加载只允许"帮忙"，不允许改变结果语义。
_UNDEFINED_SYMBOL = "undefined symbol"
_PRELOAD_ATTEMPTED = False
_PRELOAD_RESULT: str | None = None
_PRELOAD_FAILURE: str | None = None
# 每个候选库路径只 dlopen 一次（幂等），结果按路径缓存。
_PRELOAD_TRIED: dict[str, str | None] = {}
_LOAD_RETRY: str | None = None
# ``ctypes.CDLL`` 走模块属性，契约测试才能只替换本模块的加载器，而不去补丁 stdlib 里
# 的全局符号（那会波及同一进程里无关的 ctypes 使用者）。
_dlopen = ctypes.CDLL


def _mpi4py_candidate() -> list[str]:
    """MPI runtime already dragged into the process by an imported ``mpi4py.MPI``."""
    origin = getattr(sys.modules.get("mpi4py.MPI"), "__file__", None)
    return [str(origin)] if origin else []


def _find_library_candidate() -> list[str]:
    """Whatever the dynamic linker cache calls the MPI runtime."""
    found = ctypes.util.find_library("mpi")
    return [found] if found else []


def _mpicxx_candidates() -> list[str]:
    """MPI runtimes named by ``mpicxx --showme:link`` (OpenMPI/MPICH conventions)."""
    for compiler in ("mpicxx", "mpic++", "mpicc"):
        executable = shutil.which(compiler)
        if not executable:
            continue
        try:
            completed = subprocess.run(
                [executable, "--showme:link"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        tokens = completed.stdout.split()
        directories = [token[2:] for token in tokens if token.startswith("-L")]
        names = [token[2:] for token in tokens if token.startswith("-l")]
        found: list[str] = []
        for directory in directories:
            for name in names:
                found.extend(
                    sorted(glob.glob(os.path.join(directory, f"lib{name}.so*"))))
            found.extend(sorted(glob.glob(os.path.join(directory, "libmpi.so*"))))
        if found:
            return found
    return []


def _ld_library_path_candidates() -> list[str]:
    """``libmpi.so*`` reachable through ``LD_LIBRARY_PATH``."""
    found: list[str] = []
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if entry:
            found.extend(sorted(glob.glob(os.path.join(entry, "libmpi.so*"))))
    return found


def _mpi_candidate_paths() -> list[str]:
    """MPI runtimes worth preloading, cheapest first and de-duplicated."""
    candidates: list[str] = []
    for group in (
        _mpi4py_candidate(),
        _find_library_candidate(),
        _mpicxx_candidates(),
        _ld_library_path_candidates(),
    ):
        for candidate in group:
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _preload_mpi_runtime() -> str | None:
    """Bring an MPI runtime into the global scope, reporting what loaded.

    Returns a short description of the runtime now in the process (for example
    ``"libmpi.so.40"``) or ``None`` when no candidate could be loaded. Every
    path is tried at most once, and any failure is recorded instead of raised:
    the preload may only help the retry, never change its outcome.
    """
    global _PRELOAD_ATTEMPTED, _PRELOAD_RESULT, _PRELOAD_FAILURE

    if _PRELOAD_ATTEMPTED:
        return _PRELOAD_RESULT
    _PRELOAD_ATTEMPTED = True
    candidates = _mpi_candidate_paths()
    failures: list[str] = []
    for candidate in candidates:
        if candidate not in _PRELOAD_TRIED:
            try:
                _dlopen(candidate, mode=ctypes.RTLD_GLOBAL)
            except Exception as exc:  # noqa: BLE001 - 预加载只是尽力而为
                _PRELOAD_TRIED[candidate] = None
                failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
            else:
                _PRELOAD_TRIED[candidate] = os.path.basename(candidate)
        description = _PRELOAD_TRIED[candidate]
        if description:
            _PRELOAD_RESULT = description
            return description
    _PRELOAD_FAILURE = (
        "; ".join(failures)
        or f"no candidate among {len(candidates)} MPI library path(s)"
    )
    return None


def _mpi_preload_report() -> str | None:
    """One-line outcome of the MPI preload attempt (``None`` when never attempted)."""
    if not _PRELOAD_ATTEMPTED:
        return None
    if _PRELOAD_RESULT:
        return f"preloaded {_PRELOAD_RESULT}"
    return f"no usable MPI runtime ({_PRELOAD_FAILURE or 'unknown reason'})"


def _load_failure_text(error: OSError) -> str:
    """``library failed to load`` text, carrying any MPI preload attempt."""
    text = f"native einsum library failed to load: {error}"
    if _PRELOAD_ATTEMPTED:
        text += (
            f"; MPI preload attempted: {_mpi_preload_report()}; "
            "dlopen after preload failed"
        )
    return text


def _retry_after_mpi_preload(path: Path, error: OSError) -> Any:
    """Preload an MPI runtime once and retry ``dlopen`` for an unresolved symbol.

    Only an ``undefined symbol`` failure is retried: the archive is on disk and
    well formed, it just needs its MPI runtime loaded first. Every other
    ``dlopen`` failure keeps the plain fail-open path, and a preload that raises
    is recorded rather than propagated — the retry may never turn a load failure
    into a different failure.
    """
    global _LOAD_RETRY, _PRELOAD_ATTEMPTED, _PRELOAD_FAILURE

    if _UNDEFINED_SYMBOL not in str(error):
        return None
    try:
        preloaded = _preload_mpi_runtime()
    except Exception as exc:  # noqa: BLE001 - 预加载失败只记录原因，不改结果
        _PRELOAD_ATTEMPTED = True
        _PRELOAD_FAILURE = f"{type(exc).__name__}: {exc}"
        preloaded = None
    if not preloaded:
        return None
    try:
        library = _dlopen(str(path), mode=ctypes.RTLD_LOCAL)
    except OSError:
        return None
    _LOAD_RETRY = f"preloaded {preloaded}"
    return library


def _load_library() -> Any:
    """Load the archive once and bind every signature the build exports."""
    global _LIBRARY, _LOAD_FAILURE, _LOAD_FAILURE_KIND
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if _LOAD_FAILURE is not None:
            raise _NativeUnavailable(_LOAD_FAILURE)
        if not os.environ.get("PYQCD_EINSUM_CBLAS_LIB"):
            numpy_libs = Path(np.__file__).resolve().parent.parent / "numpy.libs"
            candidates = sorted(glob.glob(str(numpy_libs / "libscipy_openblas64_*.so")))
            if candidates:
                os.environ["PYQCD_EINSUM_CBLAS_LIB"] = candidates[0]
        path = library_path()
        if not path.is_file():
            _LOAD_FAILURE_KIND = "missing"
            _LOAD_FAILURE = f"native einsum library not found at {path}"
            raise _NativeUnavailable(_LOAD_FAILURE)
        try:
            library = _dlopen(str(path), mode=ctypes.RTLD_LOCAL)
        except OSError as exc:
            library = _retry_after_mpi_preload(path, exc)
            if library is None:
                _LOAD_FAILURE_KIND = "load"
                _LOAD_FAILURE = _load_failure_text(exc)
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


def _load_symbol(name: str) -> Any:
    library = _load_library()
    try:
        return getattr(library, name)
    except AttributeError as exc:
        raise NativeUnavailable(f"native einsum library does not export {name}") from exc


def load_plan() -> Any:
    """Return the ctypes ``applyPlanQcd`` entry point."""
    return _load_symbol("applyPlanQcd")


def load_contract() -> Any:
    """Return the ctypes ``applyContractQcd`` entry point."""
    return _load_symbol("applyContractQcd")


def load_mpi_contract() -> Any:
    """Return the ctypes ``applyContractMpiQcd`` entry point."""
    return _load_symbol("applyContractMpiQcd")


def load_topology() -> Any:
    """Return the ctypes ``qcd_kupl_topology_apply`` entry point."""
    return _load_symbol("qcd_kupl_topology_apply")


def available() -> bool:
    """Return whether the native planner can be loaded; never raises."""
    try:
        load_plan()
    except NativeUnavailable:
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


def _plan_buffer(plan: Any) -> np.ndarray:
    if isinstance(plan, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(plan, dtype=np.uint8)
    else:
        array = np.asarray(plan)
        if array.dtype == np.dtype(np.uint8):
            buffer = np.ascontiguousarray(array).reshape(-1)
        elif array.dtype.kind in "iu" and array.dtype.itemsize in (1, 2, 4, 8):
            buffer = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        else:
            raise TypeError("execution plan must be a bytes-like or integer array")
    return np.ascontiguousarray(buffer)


def _host_buffer(value: Any, dtype: np.dtype, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise ValueError(f"{label} dtype {array.dtype} does not match plan dtype {dtype}")
    if tuple(array.shape) != shape:
        raise ValueError(f"{label} shape {array.shape} does not match plan shape {shape}")
    return np.ascontiguousarray(array)


def _backend_module(backend: Any) -> Any:
    if backend is None:
        return np
    if isinstance(backend, str):
        if backend.lower() in {"numpy", "cpu"}:
            return np
        raise ValueError(f"KUPL execution requires a numpy backend, got {backend!r}")
    if not hasattr(backend, "empty"):
        raise TypeError("backend must expose an empty(shape, dtype=...) function")
    return backend


class KuplProgram:
    """A decoded execution plan with fixed shapes and reusable control buffers."""

    def __init__(
        self,
        plan_data: np.ndarray,
        backend: Any = np,
        mpi_config: Any = None,
        mp_config: Any = None,
    ) -> None:
        if mpi_config is not None and mp_config is not None:
            raise TypeError("pass either mpi_config or mp_config, not both")
        self.plan_data = plan_data
        self.info = decode_execution_plan(plan_data)
        self.dtype = dtype_from_id(self.info.dtype_id)
        self.input_shapes = tuple(self.info.input_shapes)
        self.output_shape = tuple(
            self.info.sizes[label] for label in self.info.output_labels
        )
        self.backend_module = _backend_module(backend)
        self.function = load_contract()
        self.params = np.zeros(_PARAM_SLOTS, dtype=np.int32)
        self.params[_PARAM_NUM_OPS] = len(self.input_shapes)
        self.params[_PARAM_MAX_NDIM] = max(map(len, self.input_shapes), default=0)
        self.params[_PARAM_DTYPE_ID] = self.info.dtype_id
        self.params[_PARAM_LAYOUT_ID] = 0
        self.mpi_config = normalize_mpi_config(mpi_config)
        self.mp_config = normalize_mp_config(mp_config)
        self.params[_PARAM_MPI_MODE] = self.mpi_config.mode_code
        self.params[_PARAM_MPI_SHARD_AXIS] = (
            -1 if self.mpi_config.shard_axis is None else self.mpi_config.shard_axis
        )
        self.params[_PARAM_MPI_OUTPUT] = self.mpi_config.output_code
        self.params_address = _address(self.params)
        self._local = threading.local()
        self.compiled_function = None
        self.destroy_function = None
        self.context = None
        self.mpi_executor = make_mpi_executor(
            self.info, self.dtype, self.mpi_config, load_mpi_contract
        )
        self.mp_executor = make_mp_executor(
            self.info, self.dtype, self.mp_config, self.backend_module
        )
        if self.mpi_executor is not None and self.mp_executor is not None:
            raise ValueError("MPI and multiprocessing execution are mutually exclusive")
        if self.mpi_executor is not None:
            self.params[_PARAM_MPI_SHARD_SYMBOL] = self.mpi_executor.shard_symbol
        elif self.mp_executor is None:
            try:
                compile_function = _load_symbol("applyCompileQcd")
                self.compiled_function = _load_symbol("applyCompiledQcd")
                self.destroy_function = _load_symbol("applyDestroyQcd")
            except NativeUnavailable:
                compile_function = None
            if compile_function is not None and self.compiled_function is not None:
                context = np.empty(1, dtype=np.int64)
                status = int(
                    compile_function(
                        _address(self.plan_data), self.params_address, _address(context)
                    )
                )
                if status != _STATUS_OK or int(context[0]) == 0:
                    raise NativeUnavailable(
                        f"native einsum compile failed with status {status}"
                    )
                self.context = context

    def __del__(self) -> None:
        try:
            if self.mp_executor is not None:
                self.mp_executor.close()
            if self.context is not None and self.destroy_function is not None:
                self.destroy_function(int(self.context[0]))
                self.context[0] = 0
        except Exception:
            pass

    def __call__(self, arrays: Sequence[Any], out: Any = None) -> Any:
        if self.mpi_executor is not None:
            return self.mpi_executor.execute(self, arrays, out=out)
        if self.mp_executor is not None:
            return self.mp_executor.execute(arrays, out=out)
        if len(arrays) != len(self.input_shapes):
            raise ValueError(
                f"plan expects {len(self.input_shapes)} arrays, got {len(arrays)}"
            )

        cacheable = all(
            isinstance(array, np.ndarray) and array.flags.c_contiguous
            for array in arrays
        )
        array_key = (
            tuple(id(array) for array in arrays),
            tuple(getattr(array, "shape", None) for array in arrays),
            tuple(getattr(array, "dtype", None) for array in arrays),
            tuple(getattr(array, "strides", None) for array in arrays),
            tuple(
                int(array.__array_interface__["data"][0])
                if isinstance(array, np.ndarray)
                else None
                for array in arrays
            ),
        )
        state = getattr(self._local, "state", None) if cacheable else None
        if state is None or state[0] != array_key:
            inputs = []
            for index, (array, shape) in enumerate(zip(arrays, self.input_shapes)):
                if (
                    isinstance(array, np.ndarray)
                    and array.dtype == self.dtype
                    and array.shape == shape
                    and array.flags.c_contiguous
                ):
                    inputs.append(array)
                else:
                    inputs.append(_host_buffer(array, self.dtype, shape, f"input {index}"))
            pointers = np.empty(len(inputs), dtype=np.int64)
            for index, array in enumerate(inputs):
                pointers[index] = int(array.ctypes.data)
            output_pointers = np.empty(1, dtype=np.int64)
            state = (array_key, inputs, pointers, output_pointers)
            if cacheable:
                self._local.state = state
        else:
            _, inputs, pointers, output_pointers = state

        output = (
            self.backend_module.empty(self.output_shape, dtype=self.dtype)
            if out is None
            else out
        )
        if out is None and self.backend_module is np:
            output_view = output
            output_array = output
        else:
            output_view = np.asarray(output)
            if output_view.dtype != self.dtype:
                raise ValueError(
                    f"output dtype {output_view.dtype} does not match plan dtype {self.dtype}"
                )
            if tuple(output_view.shape) != self.output_shape:
                raise ValueError(
                    f"output shape {output_view.shape} does not match plan shape {self.output_shape}"
                )
            output_array = np.ascontiguousarray(output_view)
        output_pointers[0] = int(output_array.ctypes.data)
        try:
            if self.context is not None and self.compiled_function is not None:
                status = int(
                    self.compiled_function(
                        int(self.context[0]),
                        _address(pointers),
                        _address(output_pointers),
                        self.params_address,
                    )
                )
            else:
                status = int(
                    self.function(
                        _address(self.plan_data),
                        _address(pointers),
                        _address(output_pointers),
                        self.params_address,
                    )
                )
        except Exception as exc:
            raise NativeUnavailable(f"native einsum execution call failed: {exc}") from exc
        if status != _STATUS_OK:
            raise NativeUnavailable(f"native einsum execution failed with status {status}")
        if output_array is not output_view:
            np.copyto(output_view, output_array)
        return output

    def _execute_local(self, arrays: Sequence[Any], out: Any = None) -> Any:
        """Execute rank-local slabs through the native MPI collective path."""
        if self.mpi_executor is None:
            raise RuntimeError("rank-local execution requires an active MPI session")
        if len(arrays) != len(self.input_shapes):
            raise ValueError(
                f"plan expects {len(self.input_shapes)} arrays, got {len(arrays)}"
            )

        inputs = []
        for index, array in enumerate(arrays):
            value = np.asarray(array)
            if value.dtype != self.dtype:
                raise ValueError(
                    f"input {index} dtype {value.dtype} does not match plan dtype {self.dtype}"
                )
            inputs.append(np.ascontiguousarray(value))
        expected_output_shape = (
            self.output_shape
            if self.mpi_executor.output_mode == 0
            else self.mpi_executor._local_shape(self.output_shape)
        )
        output = (
            self.backend_module.empty(expected_output_shape, dtype=self.dtype)
            if out is None
            else out
        )
        output_view = np.asarray(output)
        if output_view.dtype != self.dtype:
            raise ValueError(
                f"output dtype {output_view.dtype} does not match plan dtype {self.dtype}"
            )
        if tuple(output_view.shape) != expected_output_shape:
            raise ValueError(
                f"output shape {output_view.shape} does not match MPI shape {expected_output_shape}"
            )
        output_array = np.ascontiguousarray(output_view)

        pointers = np.empty(len(inputs), dtype=np.int64)
        for index, array in enumerate(inputs):
            pointers[index] = int(array.ctypes.data)
        try:
            status = int(
                load_mpi_contract()(
                    _address(self.plan_data),
                    _address(pointers),
                    int(output_array.ctypes.data),
                    self.params_address,
                )
            )
        except Exception as exc:
            raise NativeUnavailable(f"native MPI einsum execution call failed: {exc}") from exc
        if status == 1:
            raise MpiConfigError("native MPI einsum configuration or plan is invalid")
        if status != _STATUS_OK:
            raise NativeUnavailable(f"native MPI einsum execution failed with status {status}")
        if output_array is not output_view:
            np.copyto(output_view, output_array)
        return output


_PROGRAM_CACHE: dict[
    tuple[
        bytes,
        tuple[int, int | None, int],
        tuple[int, int | None, int, int, int],
    ],
    KuplProgram,
] = {}
_PROGRAM_CACHE_LIMIT = 64
_PROGRAM_CACHE_LOCK = threading.Lock()


def compile_program(
    plan: Any,
    backend: Any = np,
    mpi_config: Any = None,
    mp_config: Any = None,
    mpi: Any = None,
    mp: Any = None,
) -> KuplProgram:
    """Compile a plan once and return its reusable native callable."""
    if mpi is not None:
        if mpi_config is not None:
            raise TypeError("pass either mpi_config or mpi, not both")
        mpi_config = mpi
    if mp is not None:
        if mp_config is not None:
            raise TypeError("pass either mp_config or mp, not both")
        mp_config = mp
    if mpi_config is not None and mp_config is not None:
        raise TypeError("pass either mpi_config or mp_config, not both")
    plan_data = _plan_buffer(plan)
    key = (
        plan_data.tobytes(),
        mpi_cache_key(mpi_config),
        mp_cache_key(mp_config),
    )
    with _PROGRAM_CACHE_LOCK:
        program = _PROGRAM_CACHE.get(key)
    if program is None:
        candidate = KuplProgram(
            plan_data,
            backend=backend,
            mpi_config=mpi_config,
            mp_config=mp_config,
        )
        with _PROGRAM_CACHE_LOCK:
            program = _PROGRAM_CACHE.get(key)
            if program is None:
                if len(_PROGRAM_CACHE) >= _PROGRAM_CACHE_LIMIT:
                    _PROGRAM_CACHE.pop(next(iter(_PROGRAM_CACHE)))
                _PROGRAM_CACHE[key] = candidate
                program = candidate
    return program


def execute(
    plan: Any,
    arrays: Sequence[Any],
    out: Any = None,
    backend: Any = np,
    mpi_config: Any = None,
    mp_config: Any = None,
    mpi: Any = None,
    mp: Any = None,
) -> Any:
    """Execute a version-2 plan with a cached KUPL native program."""
    return compile_program(
        plan,
        backend=backend,
        mpi_config=mpi_config,
        mp_config=mp_config,
        mpi=mpi,
        mp=mp,
    )(arrays, out=out)


def mpi_enabled() -> bool:
    """返回当前 KUPL 二进制是否编入真实 MPI 分片实现。

    非 MPI 构建下 ``applyContractMpiQcd`` 是 fail-closed stub（返回 1），
    本函数返回 ``False``，供上层显式标注"非 MPI 版本"。
    """
    try:
        library = _load_library()
        entry = library.qcd_kupl_mpi_enabled
    except Exception:  # noqa: BLE001 - 缺库/缺符号视为非 MPI
        return False
    entry.argtypes = []
    entry.restype = ctypes.c_int
    return entry() == 1


_PLAN_VIEW_CACHE: dict[
    bytes, tuple[Any, np.dtype, tuple[tuple[int, ...], ...], tuple[int, ...]]
] = {}
_PLAN_VIEW_CACHE_LIMIT = 64
_PLAN_VIEW_LOCK = threading.Lock()


def _plan_view(
    plan_data: np.ndarray,
) -> tuple[Any, np.dtype, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Decode and validate a plan once per distinct byte stream.

    A decoded plan is a pure function of its bytes, so the second call of the
    same plan must not pay the decode again: on a 1.7 KB / 15-step plan that is
    ~120 us, i.e. more than the native execution itself.
    """
    key = plan_data.tobytes()
    with _PLAN_VIEW_LOCK:
        cached = _PLAN_VIEW_CACHE.get(key)
    if cached is not None:
        return cached
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
    info: dict[str, Any] | None = None,
) -> Any:
    """Run a whole version-2 plan with one native call.

    ``operands`` are the plan inputs in plan order; the destination (``out``, or
    a freshly allocated C-contiguous array when it is ``None``) is returned.
    ``offsets`` are optional **byte** offsets added to each operand pointer, so a
    slice of a big tensor can be addressed without copying it.

    dtype and shape must match the plan exactly — nothing is converted
    implicitly — and ``out`` must be C-contiguous, otherwise the native write
    would land somewhere the caller cannot see. A mis-sized or non-contiguous
    input is copied into a contiguous one first, exactly like
    :meth:`KuplProgram.__call__`.

    ``info`` (optional dict) is cleared and filled with what was actually sent
    down: plan byte count, step count, dtype id, operand count, output shape,
    whether ``out`` was caller-provided, which operands had to be copied into a
    contiguous buffer (``copied``) and the raw ``offsets``. Callers use it to
    report a copy instead of silently paying for one.

    Raises :class:`NativeUnavailable` when the kernel reports a non-zero status;
    ``status == _TOPOLOGY_UNSUPPORTED`` means the plan itself (not the arguments)
    is outside what this entry can execute.
    """
    plan_data = _plan_buffer(plan)
    decoded, dtype, input_shapes, output_shape = _plan_view(plan_data)
    if len(operands) != len(input_shapes):
        raise ValueError(f"plan expects {len(input_shapes)} operands, got {len(operands)}")
    inputs = []
    copied = []
    for index, (operand, shape) in enumerate(zip(operands, input_shapes)):
        inputs.append(_host_buffer(operand, dtype, shape, f"operand {index}"))
        # ``_host_buffer`` 用 np.ascontiguousarray 兜底；非连续输入因此被静默复制。
        # 把下标记进 info，调用方才能发现这块额外拷贝。
        if not (isinstance(operand, np.ndarray) and operand.flags.c_contiguous):
            copied.append(index)
    if out is None:
        output = np.empty(output_shape, dtype=dtype)
    else:
        output = np.asarray(out)
        if output.dtype != dtype:
            raise ValueError(f"output dtype {output.dtype} does not match plan dtype {dtype}")
        if tuple(output.shape) != output_shape:
            raise ValueError(
                f"output shape {output.shape} does not match plan shape {output_shape}"
            )
        if not output.flags.c_contiguous:
            raise ValueError("output buffer must be C-contiguous")

    pointers = (ctypes.c_void_p * len(inputs))(
        *[int(array.ctypes.data) for array in inputs]
    )
    offset_array = None
    raw_offsets = None
    if offsets is not None:
        raw_offsets = tuple(int(value) for value in offsets)
        if len(raw_offsets) != len(inputs):
            raise ValueError(
                f"offsets must have {len(inputs)} entries, got {len(raw_offsets)}")
        offset_array = (_LL * len(raw_offsets))(*raw_offsets)

    if info is not None:
        info.clear()
        info.update(
            dtype=str(dtype),
            dtype_id=int(decoded.dtype_id),
            n_operands=len(inputs),
            plan_bytes=int(plan_data.size),
            steps=len(decoded.steps),
            output_shape=tuple(output_shape),
            out="allocated" if out is None else "provided",
            copied=tuple(copied),
            offsets=raw_offsets,
        )

    status = int(
        load_topology()(
            int(plan_data.ctypes.data),
            int(plan_data.size),
            pointers,
            len(inputs),
            offset_array,
            int(output.ctypes.data),
            int(decoded.dtype_id),
        )
    )
    if status != _STATUS_OK:
        reason = (
            " (plan not executable as version-2 steps)"
            if status == _TOPOLOGY_UNSUPPORTED
            else ""
        )
        # 状态码随身带出：1/3 是"调用参数或原生内部有问题"，4 是"这条计划本入口
        # 不认"，上层据此给出不同的诊断并决定是否回落。
        error = NativeUnavailable(
            f"native topology execution failed with status {status}{reason}")
        error.status = status
        raise error
    return output
