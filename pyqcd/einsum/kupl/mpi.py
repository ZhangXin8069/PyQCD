"""MPI sharding support for the KUPL execution backend.

Stage one shards exactly one output axis retained by every execution step.
The native runtime owns plan cloning and the final ``MPI_Allgatherv``; this
module lazily resolves mpi4py, prepares rank-local slabs, and keeps failures
symmetric across ranks.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

MPI_MODE_OFF = 0
MPI_MODE_BATCH = 1
MPI_MODE_FREE = 2
MPI_MODE_REQUIRE = 3

MPI_OUTPUT_GATHER = 0
MPI_OUTPUT_LOCAL = 1

_MODE_NAMES = {
    "off": MPI_MODE_OFF,
    "disabled": MPI_MODE_OFF,
    "serial": MPI_MODE_OFF,
    "batch": MPI_MODE_BATCH,
    "free": MPI_MODE_FREE,
    "require": MPI_MODE_REQUIRE,
    "required": MPI_MODE_REQUIRE,
    "mpi": MPI_MODE_REQUIRE,
    "on": MPI_MODE_REQUIRE,
}
_OUTPUT_NAMES = {
    "gather": MPI_OUTPUT_GATHER,
    "global": MPI_OUTPUT_GATHER,
    "local": MPI_OUTPUT_LOCAL,
}


class MpiConfigError(RuntimeError):
    """The MPI einsum configuration or plan is invalid on at least one rank."""


class MpiUnavailable(RuntimeError):
    """MPI execution was requested but mpi4py or a compatible thread level is absent."""


@dataclass(frozen=True)
class MpiConfig:
    """Configuration for one-axis KUPL MPI sharding."""

    mode: str | int = "off"
    shard_axis: int | None = None
    output: str | int = "gather"

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, str):
            try:
                mode = _MODE_NAMES[mode.strip().lower()]
            except KeyError as exc:
                raise ValueError(f"unknown MPI einsum mode: {self.mode!r}") from exc
        else:
            mode = operator.index(mode)
        if mode not in (MPI_MODE_OFF, MPI_MODE_BATCH, MPI_MODE_FREE, MPI_MODE_REQUIRE):
            raise ValueError(f"invalid MPI einsum mode id: {mode}")

        output = self.output
        if isinstance(output, str):
            try:
                output = _OUTPUT_NAMES[output.strip().lower()]
            except KeyError as exc:
                raise ValueError(f"unknown MPI output mode: {self.output!r}") from exc
        else:
            output = operator.index(output)
        if output not in (MPI_OUTPUT_GATHER, MPI_OUTPUT_LOCAL):
            raise ValueError(f"invalid MPI output mode id: {output}")

        axis = self.shard_axis
        if axis is not None:
            axis = operator.index(axis)
            if axis < 0:
                raise ValueError("shard_axis must be non-negative")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "shard_axis", axis)
        object.__setattr__(self, "output", output)

    @property
    def enabled(self) -> bool:
        return self.mode != MPI_MODE_OFF

    @property
    def mode_code(self) -> int:
        return int(self.mode)

    @property
    def output_code(self) -> int:
        return int(self.output)

    @property
    def cache_key(self) -> tuple[int, int | None, int]:
        return self.mode, self.shard_axis, self.output


def normalize_mpi_config(value: Any = None) -> MpiConfig:
    """Normalize ``None``, strings, mappings, and :class:`MpiConfig` values."""
    if isinstance(value, MpiConfig):
        return value
    if value is None:
        return MpiConfig()
    if isinstance(value, str):
        return MpiConfig(mode=value)
    if isinstance(value, dict):
        return MpiConfig(**value)
    raise TypeError(f"mpi_config must be MpiConfig-compatible, got {type(value).__name__}")


def mpi_cache_key(value: Any = None) -> tuple[int, int | None, int]:
    """Return a stable cache component that includes every MPI execution knob."""
    return normalize_mpi_config(value).cache_key


@dataclass(frozen=True)
class _Partition:
    rank: int
    size: int
    start: int
    count: int


def _partition(extent: int, size: int, rank: int) -> _Partition:
    base, remainder = divmod(extent, size)
    count = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
    return _Partition(rank, size, start, count)


def _load_mpi() -> Any:
    try:
        from mpi4py import MPI
    except Exception as exc:  # noqa: BLE001 - import failures are an optional dependency boundary
        raise MpiUnavailable(f"mpi4py is unavailable: {type(exc).__name__}: {exc}") from exc
    if not MPI.Is_initialized():
        raise MpiUnavailable("MPI is not initialized; KUPL does not call MPI_Init")
    if MPI.Query_thread() < MPI.THREAD_FUNNELED:
        raise MpiUnavailable("KUPL MPI requires MPI_THREAD_FUNNELED or stronger")
    return MPI


def _initial_labels(info: Any) -> tuple[tuple[int, ...], ...]:
    active: list[tuple[bool, tuple[int, ...] | None]] = [
        (True, None) for _ in info.input_shapes
    ]
    initial: list[tuple[int, ...] | None] = [None] * len(info.input_shapes)
    for step in info.steps:
        for position, labels in zip(step.positions, step.inputs):
            if position < 0 or position >= len(active):
                raise MpiConfigError("execution plan references an invalid live operand")
            is_initial, known = active[position]
            if known is not None and known != labels:
                raise MpiConfigError("execution plan has inconsistent live labels")
            if is_initial and initial[position] is None:
                initial[position] = labels
            active[position] = (is_initial, labels)
        for position in sorted(step.positions, reverse=True):
            active.pop(position)
        active.append((False, step.output))
    if len(active) != 1 or any(labels is None for labels in initial):
        raise MpiConfigError("execution plan does not expose every initial input label")
    return tuple(labels for labels in initial if labels is not None)


def _select_axis(config: MpiConfig, info: Any, labels: Sequence[tuple[int, ...]]) -> int:
    if config.shard_axis is not None:
        axis = config.shard_axis
    else:
        axis = next(
            (index for index, symbol in enumerate(info.output_labels)
             if (config.mode == MPI_MODE_FREE and sum(symbol in term for term in labels) == 1)
             or (config.mode == MPI_MODE_BATCH and sum(symbol in term for term in labels) >= 2)
             or config.mode == MPI_MODE_REQUIRE),
            -1,
        )
    if axis < 0 or axis >= len(info.output_labels):
        raise MpiConfigError("shard_axis does not identify an output axis")
    return axis


def _validate_plan(config: MpiConfig, info: Any, labels: Sequence[tuple[int, ...]], axis: int) -> int:
    symbol = info.output_labels[axis]
    occurrences = sum(symbol in term for term in labels)
    if occurrences == 0:
        raise MpiConfigError("shard symbol is absent from all inputs")
    if config.mode == MPI_MODE_BATCH and occurrences < 2:
        raise MpiConfigError("batch mode requires the shard symbol in at least two inputs")
    if config.mode == MPI_MODE_FREE and occurrences != 1:
        raise MpiConfigError("free mode requires the shard symbol in exactly one input")

    extent = info.sizes[symbol]
    for input_index, term in enumerate(labels):
        if symbol not in term:
            continue
        input_axis = term.index(symbol)
        if info.input_shapes[input_index][input_axis] != extent:
            raise MpiConfigError("broadcasting is not supported on the MPI shard axis")

    for step_number, step in enumerate(info.steps):
        appears_in = [symbol in term for term in step.inputs]
        appears_out = symbol in step.output
        if any(appears_in) != appears_out:
            raise MpiConfigError(
                f"shard symbol is eliminated or introduced at execution step {step_number}"
            )
    return symbol


def _symmetric_error(comm: Any, MPI: Any, error: bool, message: str) -> None:
    failed = comm.allreduce(1 if error else 0, op=MPI.MAX)
    if failed:
        raise MpiConfigError(message)


@dataclass(frozen=True)
class MpiSession:
    config: MpiConfig
    communicator: Any
    mpi: Any
    initial_labels: tuple[tuple[int, ...], ...]
    shard_axis: int
    shard_symbol: int
    partition: _Partition

    @property
    def output_mode(self) -> int:
        return int(self.config.output)

    def _local_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        result = list(shape)
        result[self.shard_axis] = self.partition.count
        return tuple(result)

    def _slice_inputs(self, arrays: Sequence[Any], program: Any) -> list[np.ndarray]:
        if len(arrays) != len(program.input_shapes):
            raise MpiConfigError(
                f"plan expects {len(program.input_shapes)} arrays, got {len(arrays)}"
            )
        local: list[np.ndarray] = []
        for index, (array, global_shape, labels) in enumerate(
            zip(arrays, program.input_shapes, self.initial_labels)
        ):
            value = np.asarray(array)
            if value.dtype != program.dtype:
                raise MpiConfigError(
                    f"input {index} dtype {value.dtype} does not match plan dtype {program.dtype}"
                )
            if tuple(value.shape) != tuple(global_shape):
                raise MpiConfigError(
                    f"input {index} shape {value.shape} does not match plan shape {global_shape}"
                )
            if self.shard_symbol not in labels:
                local.append(np.ascontiguousarray(value))
                continue
            axis = labels.index(self.shard_symbol)
            slices = [slice(None)] * value.ndim
            slices[axis] = slice(
                self.partition.start, self.partition.start + self.partition.count
            )
            slab = np.ascontiguousarray(value[tuple(slices)])
            expected = list(global_shape)
            expected[axis] = self.partition.count
            if slab.shape != tuple(expected):
                raise MpiConfigError(f"input {index} produced an invalid rank-local slab")
            local.append(slab)
        return local

    def _prepare_output(self, out: Any, program: Any) -> tuple[Any, np.ndarray, bool]:
        global_shape = tuple(program.output_shape)
        shape = global_shape if self.output_mode == MPI_OUTPUT_GATHER else self._local_shape(global_shape)
        output = program.backend_module.empty(shape, dtype=program.dtype) if out is None else out
        view = np.asarray(output)
        if view.dtype != program.dtype:
            raise MpiConfigError(
                f"output dtype {view.dtype} does not match plan dtype {program.dtype}"
            )
        if tuple(view.shape) != shape:
            raise MpiConfigError(f"output shape {view.shape} does not match MPI shape {shape}")
        work = np.ascontiguousarray(view)
        return output, work, view is work

    def execute(self, program: Any, arrays: Sequence[Any], out: Any = None) -> Any:
        local_inputs: list[np.ndarray] = []
        output: Any = None
        output_work: np.ndarray | None = None
        output_direct = False
        local_error = False
        try:
            local_inputs = self._slice_inputs(arrays, program)
            output, output_work, output_direct = self._prepare_output(out, program)
        except Exception:  # noqa: BLE001 - rank-local failures must be symmetrized
            local_error = True
        _symmetric_error(
            self.communicator,
            self.mpi,
            local_error,
            "rank-local MPI einsum input/output validation failed",
        )
        assert output_work is not None

        program._execute_local(local_inputs, output_work)
        if not output_direct:
            np.copyto(np.asarray(output), output_work)
        return output


def prepare(config: MpiConfig, info: Any, communicator: Any = None) -> MpiSession | None:
    """Resolve MPI lazily; return ``None`` for the serial fast path."""
    if not config.enabled:
        return None
    MPI = _load_mpi()
    comm = MPI.COMM_WORLD if communicator is None else communicator
    if comm.Compare(MPI.COMM_WORLD) != MPI.IDENT:
        raise MpiUnavailable(
            "native KUPL MPI currently requires MPI_COMM_WORLD"
        )
    if comm.Get_size() == 1:
        return None

    local_error = False
    session: MpiSession | None = None
    try:
        labels = _initial_labels(info)
        axis = _select_axis(config, info, labels)
        symbol = _validate_plan(config, info, labels, axis)
        rank = comm.Get_rank()
        size = comm.Get_size()
        partition = _partition(info.sizes[symbol], size, rank)
        session = MpiSession(config, comm, MPI, labels, axis, symbol, partition)
    except Exception:  # noqa: BLE001 - make plan-validation failure rank symmetric
        local_error = True
    _symmetric_error(comm, MPI, local_error, "MPI einsum plan validation failed")
    assert session is not None
    return session


def make_mpi_executor(
    info: Any,
    dtype: Any,
    config: Any,
    native_loader: Any,
    communicator: Any = None,
) -> MpiSession | None:
    """Compatibility constructor used by the ctypes KUPL program wrapper."""
    del dtype, native_loader
    return prepare(normalize_mpi_config(config), info, communicator=communicator)


__all__ = [
    "MPI_MODE_BATCH",
    "MPI_MODE_FREE",
    "MPI_MODE_OFF",
    "MPI_MODE_REQUIRE",
    "MPI_OUTPUT_GATHER",
    "MPI_OUTPUT_LOCAL",
    "MpiConfig",
    "MpiConfigError",
    "MpiSession",
    "MpiUnavailable",
    "make_mpi_executor",
    "mpi_cache_key",
    "normalize_mpi_config",
]
