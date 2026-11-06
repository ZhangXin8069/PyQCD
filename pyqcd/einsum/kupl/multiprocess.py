"""Spawn-based multiprocessing fallback for KUPL execution.

The fallback shards one output axis that is retained by every execution step.
Workers read and write host shared memory, while the parent process validates
and assembles results in shard order. MPI is never initialized here.
"""

from __future__ import annotations

import ctypes
import mmap
import multiprocessing
import operator
import os
import sys
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from .._native_plan import dtype_from_id, encode_execution_plan
from .mpi import (
    MPI_MODE_BATCH,
    MPI_MODE_FREE,
    MpiConfig,
    MpiConfigError,
    _initial_labels,
    _validate_plan,
)

MP_MODE_OFF = 0
MP_MODE_AUTO = 1
MP_MODE_REQUIRE = 2

MP_OUTPUT_GATHER = 0
MP_OUTPUT_LOCAL = 1

_MODE_NAMES = {
    "off": MP_MODE_OFF,
    "disabled": MP_MODE_OFF,
    "serial": MP_MODE_OFF,
    "auto": MP_MODE_AUTO,
    "require": MP_MODE_REQUIRE,
    "required": MP_MODE_REQUIRE,
}
_OUTPUT_NAMES = {
    "gather": MP_OUTPUT_GATHER,
    "global": MP_OUTPUT_GATHER,
    "local": MP_OUTPUT_LOCAL,
}
_MPI_ENV_MARKERS = (
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_UNIVERSE_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "SLURM_NTASKS",
)
_BLAS_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class MpConfigError(RuntimeError):
    """The multiprocessing configuration or execution plan is invalid."""


class MpUnavailable(RuntimeError):
    """Multiprocessing was required but cannot be enabled safely."""


@dataclass(frozen=True)
class MpConfig:
    """Configuration for one-axis KUPL multiprocessing."""

    mode: str | int = "off"
    shard_axis: int | None = None
    output: str | int = "gather"
    workers: int = 1
    threads_per_worker: int = 1

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, str):
            try:
                mode = _MODE_NAMES[mode.strip().lower()]
            except KeyError as exc:
                raise ValueError(f"unknown multiprocessing mode: {self.mode!r}") from exc
        else:
            mode = operator.index(mode)
        if mode not in (MP_MODE_OFF, MP_MODE_AUTO, MP_MODE_REQUIRE):
            raise ValueError(f"invalid multiprocessing mode id: {mode}")

        output = self.output
        if isinstance(output, str):
            try:
                output = _OUTPUT_NAMES[output.strip().lower()]
            except KeyError as exc:
                raise ValueError(f"unknown multiprocessing output mode: {self.output!r}") from exc
        else:
            output = operator.index(output)
        if output not in (MP_OUTPUT_GATHER, MP_OUTPUT_LOCAL):
            raise ValueError(f"invalid multiprocessing output mode id: {output}")

        axis = self.shard_axis
        if axis is not None:
            axis = operator.index(axis)
            if axis < 0:
                raise ValueError("shard_axis must be non-negative")

        workers = operator.index(self.workers)
        if workers < 1:
            raise ValueError("workers must be at least 1")
        threads = operator.index(self.threads_per_worker)
        if threads != 1:
            raise ValueError("threads_per_worker must be 1 in the current KUPL MP fallback")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "shard_axis", axis)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "threads_per_worker", threads)

    @property
    def enabled(self) -> bool:
        return self.mode != MP_MODE_OFF and self.workers > 1

    @property
    def mode_code(self) -> int:
        return int(self.mode)

    @property
    def output_code(self) -> int:
        return int(self.output)

    @property
    def cache_key(self) -> tuple[int, int | None, int, int, int]:
        return (
            int(self.mode),
            self.shard_axis,
            int(self.output),
            self.workers,
            self.threads_per_worker,
        )


def normalize_mp_config(value: Any = None) -> MpConfig:
    """Normalize ``None``, strings, mappings, and :class:`MpConfig` values."""
    if isinstance(value, MpConfig):
        return value
    if value is None:
        return MpConfig()
    if isinstance(value, str):
        return MpConfig(mode=value)
    if isinstance(value, dict):
        return MpConfig(**value)
    raise TypeError(f"mp_config must be MpConfig-compatible, got {type(value).__name__}")


def mp_cache_key(value: Any = None) -> tuple[int, int | None, int, int, int]:
    """Return a cache component covering every multiprocessing execution knob."""
    return normalize_mp_config(value).cache_key


def _mpi_active() -> bool:
    """Detect an existing MPI context without importing mpi4py or initializing it."""
    for module_name in ("mpi4py.MPI", "MPI"):
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, "Is_initialized"):
            continue
        try:
            if bool(module.Is_initialized()):
                return True
        except Exception:
            return True
    try:
        function = ctypes.CDLL(None).MPI_Initialized
        function.argtypes = (ctypes.POINTER(ctypes.c_int),)
        function.restype = ctypes.c_int
        initialized = ctypes.c_int()
        if function(ctypes.byref(initialized)) == 0 and initialized.value:
            return True
    except (AttributeError, OSError):
        pass
    return any(name in os.environ for name in _MPI_ENV_MARKERS)


def _partition(extent: int, workers: int, worker: int) -> tuple[int, int]:
    base, remainder = divmod(extent, workers)
    count = base + (1 if worker < remainder else 0)
    start = worker * base + min(worker, remainder)
    return start, count


@dataclass(frozen=True)
class _Partition:
    worker: int
    start: int
    count: int


@dataclass(frozen=True)
class _PlanOperand:
    shape: tuple[int, ...]
    dtype: np.dtype


def _local_plan(
    info: Any,
    labels: Sequence[tuple[int, ...]],
    shard_symbol: int,
    count: int,
) -> bytes:
    """Re-encode the same plan with the retained shard extent made local."""
    symbols = tuple(chr(0x100 + index) for index in range(len(info.sizes)))
    if any(len(symbol) != 1 for symbol in symbols):
        raise MpConfigError("execution plan has too many symbols for MP plan rewriting")

    local_shapes: list[tuple[int, ...]] = []
    for shape, term in zip(info.input_shapes, labels):
        local_shape = list(shape)
        if shard_symbol in term:
            local_shape[term.index(shard_symbol)] = count
        local_shapes.append(tuple(local_shape))
    contraction_list = []
    for step in info.steps:
        used: set[int] = set()
        for term in step.inputs:
            used.update(term)
        removed = frozenset(symbols[label] for label in used.difference(step.output))
        equation = (
            ",".join("".join(symbols[label] for label in term) for term in step.inputs)
            + "->"
            + "".join(symbols[label] for label in step.output)
        )
        contraction_list.append((step.positions, removed, equation, None, False))

    # The encoder only reads shape and dtype; the original dtype is required for
    # the rewritten plan header even though no parent-side allocation is made.
    dtype = dtype_from_id(info.dtype_id)
    operands = tuple(_PlanOperand(shape, dtype) for shape in local_shapes)
    return encode_execution_plan(contraction_list, operands)


def _resolve_axis(
    config: MpConfig, info: Any, labels: Sequence[tuple[int, ...]]
) -> tuple[int, int, int]:
    axes = (
        (config.shard_axis,)
        if config.shard_axis is not None
        else tuple(range(len(info.output_labels)))
    )
    failures = []
    for axis in axes:
        if axis >= len(info.output_labels):
            failures.append("shard_axis does not identify an output axis")
            continue
        symbol = info.output_labels[axis]
        occurrences = sum(symbol in term for term in labels)
        if occurrences == 1:
            semantic_mode = MPI_MODE_FREE
        elif occurrences >= 2:
            semantic_mode = MPI_MODE_BATCH
        else:
            failures.append("shard symbol is absent from all inputs")
            continue
        try:
            _validate_plan(MpiConfig(semantic_mode, axis, "gather"), info, labels, axis)
        except MpiConfigError as exc:
            failures.append(str(exc))
            continue
        return axis, symbol, semantic_mode

    detail = failures[-1] if failures else "shard_axis does not identify an output axis"
    raise MpConfigError(f"no safe retained output axis for MP sharding: {detail}")


class _SharedBuffer:
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = elements * dtype.itemsize
        self.memory = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        self.value = np.frombuffer(self.memory.buf, dtype=dtype, count=elements).reshape(shape)

    @property
    def name(self) -> str:
        return self.memory.name

    def close(self) -> None:
        value = getattr(self, "value", None)
        if value is not None:
            del value
            self.value = None
        memory = getattr(self, "memory", None)
        if memory is None:
            return
        self.memory = None
        memory.close()
        try:
            memory.unlink()
        except FileNotFoundError:
            pass


def _new_shared(shape: tuple[int, ...], dtype: np.dtype) -> _SharedBuffer:
    return _SharedBuffer(shape, dtype)


class _SharedAttachment:
    def __init__(self, mapping: mmap.mmap) -> None:
        self.mapping = mapping

    def close(self) -> None:
        mapping = self.mapping
        if mapping is not None:
            self.mapping = None
            mapping.close()


def _attach_shared(
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    offset: int = 0,
    strides: tuple[int, ...] | None = None,
) -> tuple[_SharedAttachment, np.ndarray]:
    shm_dir = getattr(shared_memory, "_SHM_DIR", "/dev/shm")
    path = os.path.join(shm_dir, name.lstrip("/"))
    file_descriptor = os.open(path, os.O_RDWR)
    try:
        mapping = mmap.mmap(file_descriptor, 0)
    finally:
        os.close(file_descriptor)
    value_dtype = np.dtype(dtype)
    view = np.ndarray(
        shape=shape,
        dtype=value_dtype,
        buffer=mapping,
        offset=offset,
        strides=strides,
    )
    return _SharedAttachment(mapping), view


def _limit_worker_blas() -> None:
    for name in _BLAS_ENV:
        os.environ[name] = "1"

    numpy_libs = os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs")
    try:
        candidates = sorted(
            os.path.join(numpy_libs, name)
            for name in os.listdir(numpy_libs)
            if name.startswith("libscipy_openblas64_") and name.endswith(".so")
        )
    except OSError:
        candidates = []
    for path in candidates:
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        for symbol in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
            function = getattr(library, symbol, None)
            if function is None:
                continue
            try:
                function(1)
            except Exception:
                pass


def _worker_initialize(threads_per_worker: int) -> None:
    if threads_per_worker != 1:
        raise RuntimeError("KUPL MP workers currently require one BLAS thread")
    _limit_worker_blas()


_WORKER_PROGRAMS: dict[bytes, Any] = {}
_WORKER_PROGRAM_LIMIT = 16


@dataclass(frozen=True)
class _WorkerInput:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int = 0
    strides: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _WorkerTask:
    plan_data: bytes
    inputs: tuple[_WorkerInput, ...]
    output_name: str | None
    output_shape: tuple[int, ...]
    output_dtype: str


def _worker_execute(task: _WorkerTask) -> str | None:
    attached: list[_SharedAttachment] = []
    inputs: tuple[np.ndarray, ...] = ()
    output: np.ndarray | None = None
    input_values: list[np.ndarray] = []
    program = None
    try:
        if any(extent == 0 for extent in task.output_shape):
            return None
        if task.output_name is None:
            return "worker received no shared output buffer"
        output_memory, output = _attach_shared(
            task.output_name, task.output_shape, task.output_dtype
        )
        attached.append(output_memory)
        for item in task.inputs:
            memory, view = _attach_shared(
                item.name, item.shape, item.dtype, item.offset, item.strides
            )
            attached.append(memory)
            input_values.append(view)
        inputs = tuple(input_values)

        from .einsum import compile_program

        program = _WORKER_PROGRAMS.get(task.plan_data)
        if program is None:
            program = compile_program(task.plan_data, mp_config=MpConfig("off"))
            if len(_WORKER_PROGRAMS) >= _WORKER_PROGRAM_LIMIT:
                _WORKER_PROGRAMS.pop(next(iter(_WORKER_PROGRAMS)))
            _WORKER_PROGRAMS[task.plan_data] = program
        program(inputs, out=output)
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        if program is not None:
            program._local.state = None
        del inputs
        del output
        del input_values
        try:
            del view
        except UnboundLocalError:
            pass
        for attachment in attached:
            attachment.close()


class MpSession:
    """Reusable spawn worker pool for one fixed execution plan."""

    def __init__(
        self,
        config: MpConfig,
        info: Any,
        dtype: np.dtype,
        labels: Sequence[tuple[int, ...]],
        shard_axis: int,
        shard_symbol: int,
        semantic_mode: int,
        context: Any,
    ) -> None:
        self.config = config
        self.info = info
        self.dtype = np.dtype(dtype)
        self.initial_labels = tuple(tuple(term) for term in labels)
        self.shard_axis = shard_axis
        self.shard_symbol = shard_symbol
        self.semantic_mode = semantic_mode
        extent = info.sizes[shard_symbol]
        self.partitions = tuple(
            _Partition(worker, *_partition(extent, config.workers, worker))
            for worker in range(config.workers)
        )
        self._executor = ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=context,
            initializer=_worker_initialize,
            initargs=(config.threads_per_worker,),
        )
        counts = {partition.count for partition in self.partitions}
        self._local_plans = {
            count: _local_plan(info, self.initial_labels, shard_symbol, count)
            for count in counts
        }

    @property
    def workers(self) -> int:
        return self.config.workers

    @property
    def output_mode(self) -> int:
        return int(self.config.output)

    def local_shape(self, count: int) -> tuple[int, ...]:
        shape = list(self.info.sizes[label] for label in self.info.output_labels)
        shape[self.shard_axis] = count
        return tuple(shape)

    def _local_input_shapes(self, count: int) -> tuple[tuple[int, ...], ...]:
        result = []
        for shape, labels in zip(self.info.input_shapes, self.initial_labels):
            local_shape = list(shape)
            if self.shard_symbol in labels:
                local_shape[labels.index(self.shard_symbol)] = count
            result.append(tuple(local_shape))
        return tuple(result)

    def _input_layouts(
        self, partition: _Partition
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        layouts = []
        for shape, labels in zip(self.info.input_shapes, self.initial_labels):
            strides = [self.dtype.itemsize] * len(shape)
            for axis in range(len(shape) - 2, -1, -1):
                strides[axis] = strides[axis + 1] * shape[axis + 1]
            if self.shard_symbol not in labels:
                layouts.append((0, tuple(strides)))
                continue
            axis = labels.index(self.shard_symbol)
            layouts.append((partition.start * strides[axis], tuple(strides)))
        return tuple(layouts)

    def _finish_gather(self, locals_: Sequence[np.ndarray], out: Any) -> Any:
        global_shape = tuple(self.info.sizes[label] for label in self.info.output_labels)
        target = np.empty(global_shape, dtype=self.dtype) if out is None else out
        target_view = np.asarray(target)
        if target_view.dtype != self.dtype:
            raise MpConfigError(
                f"output dtype {target_view.dtype} does not match plan dtype {self.dtype}"
            )
        if tuple(target_view.shape) != global_shape:
            raise MpConfigError(
                f"output shape {target_view.shape} does not match MP shape {global_shape}"
            )
        for partition, local in zip(self.partitions, locals_):
            slices = [slice(None)] * len(global_shape)
            slices[self.shard_axis] = slice(
                partition.start, partition.start + partition.count
            )
            np.copyto(target_view[tuple(slices)], local)
        return target

    def _finish_local(self, locals_: Sequence[np.ndarray], out: Any) -> Any:
        if out is None:
            return tuple(locals_)
        if isinstance(out, np.ndarray):
            raise MpConfigError("local MP output requires one output array per worker")
        try:
            targets = tuple(out)
        except TypeError as exc:
            raise MpConfigError("local MP output requires one output array per worker") from exc
        if len(targets) != self.workers:
            raise MpConfigError(
                f"local MP output requires {self.workers} arrays, got {len(targets)}"
            )
        views = []
        for worker, (target, local) in enumerate(zip(targets, locals_)):
            view = np.asarray(target)
            if view.dtype != self.dtype or tuple(view.shape) != local.shape:
                raise MpConfigError(
                    f"local output {worker} must have dtype {self.dtype} and shape {local.shape}"
                )
            views.append(view)
        for view, local in zip(views, locals_):
            np.copyto(view, local)
        return out

    def execute(self, arrays: Sequence[Any], out: Any = None) -> Any:
        if self._executor is None:
            raise MpUnavailable("KUPL MP session is closed")
        validated: list[np.ndarray] = []
        if len(arrays) != len(self.info.input_shapes):
            raise MpConfigError(
                f"plan expects {len(self.info.input_shapes)} arrays, got {len(arrays)}"
            )
        for index, (array, shape) in enumerate(zip(arrays, self.info.input_shapes)):
            value = np.asarray(array)
            if value.dtype != self.dtype:
                raise MpConfigError(
                    f"input {index} dtype {value.dtype} does not match plan dtype {self.dtype}"
                )
            if tuple(value.shape) != tuple(shape):
                raise MpConfigError(
                    f"input {index} shape {value.shape} does not match plan shape {shape}"
                )
            validated.append(value)

        shared_inputs: list[_SharedBuffer] = []
        shared_outputs: list[_SharedBuffer | None] = []
        tasks: list[_WorkerTask] = []
        try:
            for value in validated:
                shared = _new_shared(tuple(value.shape), self.dtype)
                np.copyto(shared.value, value)
                shared_inputs.append(shared)

            for partition in self.partitions:
                local_shapes = self._local_input_shapes(partition.count)
                layouts = self._input_layouts(partition)
                worker_inputs = tuple(
                    _WorkerInput(
                        shared.name, shape, self.dtype.str, offset, strides
                    )
                    for shared, shape, (offset, strides) in zip(
                        shared_inputs, local_shapes, layouts
                    )
                )
                output_shape = self.local_shape(partition.count)
                if any(extent == 0 for extent in output_shape):
                    shared_outputs.append(None)
                    output_name = None
                else:
                    output_shared = _new_shared(output_shape, self.dtype)
                    shared_outputs.append(output_shared)
                    output_name = output_shared.name
                tasks.append(
                    _WorkerTask(
                        self._local_plans[partition.count],
                        worker_inputs,
                        output_name,
                        output_shape,
                        self.dtype.str,
                    )
                )

            futures = [self._executor.submit(_worker_execute, task) for task in tasks]
            errors = []
            for partition, future in zip(self.partitions, futures):
                error = future.result()
                if error is not None:
                    errors.append(f"worker {partition.worker}:\n{error}")
            if errors:
                raise MpConfigError("KUPL MP worker failed:\n" + "\n".join(errors))

            locals_: list[np.ndarray] = []
            for partition, shared in zip(self.partitions, shared_outputs):
                output_shape = self.local_shape(partition.count)
                if shared is None:
                    locals_.append(np.empty(output_shape, dtype=self.dtype))
                else:
                    locals_.append(
                        np.array(shared.value, dtype=self.dtype, copy=True, order="C")
                    )

            if self.output_mode == MP_OUTPUT_LOCAL:
                return self._finish_local(locals_, out)
            return self._finish_gather(locals_, out)
        finally:
            # ProcessPool tasks have completed before this point; releasing the
            # named mappings here makes cleanup deterministic on every failure path.
            del tasks
            for shared in shared_outputs:
                if shared is not None:
                    shared.close()
            for shared in shared_inputs:
                shared.close()

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            self._executor = None
            executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_mp_executor(
    info: Any,
    dtype: Any,
    config: Any,
    backend_module: Any = np,
) -> MpSession | None:
    """Create a validated MP session, or preserve the serial fast path."""
    normalized = normalize_mp_config(config)
    if not normalized.enabled:
        return None
    if backend_module is not np:
        if normalized.mode == MP_MODE_REQUIRE:
            raise MpUnavailable("KUPL MP currently supports only the NumPy host backend")
        return None
    if _mpi_active():
        if normalized.mode == MP_MODE_REQUIRE:
            raise MpUnavailable("KUPL MP cannot run while MPI is initialized")
        return None
    if multiprocessing.current_process().daemon:
        if normalized.mode == MP_MODE_REQUIRE:
            raise MpUnavailable("daemonic processes cannot create KUPL MP workers")
        return None
    shm_dir = getattr(shared_memory, "_SHM_DIR", "/dev/shm")
    if os.name != "posix" or not shm_dir or not os.path.isdir(shm_dir):
        if normalized.mode == MP_MODE_REQUIRE:
            raise MpUnavailable("KUPL MP requires a POSIX shared-memory directory")
        return None

    try:
        labels = _initial_labels(info)
        shard_axis, shard_symbol, semantic_mode = _resolve_axis(normalized, info, labels)
    except MpiConfigError as exc:
        if normalized.mode == MP_MODE_AUTO:
            return None
        raise MpConfigError(f"MP execution plan validation failed: {exc}") from exc
    except MpConfigError:
        if normalized.mode == MP_MODE_AUTO:
            return None
        raise
    context = multiprocessing.get_context("spawn")
    try:
        return MpSession(
            normalized,
            info,
            np.dtype(dtype),
            labels,
            shard_axis,
            shard_symbol,
            semantic_mode,
            context,
        )
    except MpConfigError:
        raise
    except (OSError, RuntimeError) as exc:
        if normalized.mode == MP_MODE_REQUIRE:
            raise MpUnavailable(f"KUPL MP worker setup failed: {exc}") from exc
        return None


__all__ = [
    "MP_MODE_AUTO",
    "MP_MODE_OFF",
    "MP_MODE_REQUIRE",
    "MP_OUTPUT_GATHER",
    "MP_OUTPUT_LOCAL",
    "MpConfig",
    "MpConfigError",
    "MpSession",
    "MpUnavailable",
    "make_mp_executor",
    "mp_cache_key",
    "normalize_mp_config",
]
