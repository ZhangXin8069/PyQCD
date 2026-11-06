"""MPI reduction sharding for the CUDA einsum backend.

The local contraction remains the native compiled CUDA path. MPI only slices
one equation label, executes rank-local plans, and combines the result:

* ``reduce`` shards an eliminated label and sums the small output with
  ``Allreduce``;
* ``batch`` and ``free`` shard an output label and gather the result.

This layer does not initialize MPI and never starts subprocesses. Callers are
expected to launch one process per GPU with ``mpirun`` and select devices
before constructing CuPy arrays.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..kupl.mpi import (
    MpiConfigError,
    _load_mpi,
    _partition,
)

_MODES = {
    "reduce": "reduce",
    "reduction": "reduce",
    "batch": "batch",
    "free": "free",
}
_OUTPUTS = {
    "gather": "gather",
    "global": "gather",
    "local": "local",
}
_PROGRAM_CACHE: OrderedDict[tuple[Any, ...], "_MpiProgram"] = OrderedDict()
_PROGRAM_CACHE_LIMIT = 4


@dataclass(frozen=True)
class CudaMpiConfig:
    """Configuration for one-axis CUDA MPI sharding."""

    mode: str = "reduce"
    shard_label: str | None = None
    output: str = "gather"
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        try:
            mode = _MODES[str(self.mode).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown CUDA MPI mode: {self.mode!r}") from exc
        try:
            output = _OUTPUTS[str(self.output).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown CUDA MPI output mode: {self.output!r}") from exc
        label = self.shard_label
        if label is not None:
            if not isinstance(label, str) or len(label) != 1:
                raise ValueError("shard_label must be one einsum label")
            if label in ",->" or label.isspace():
                raise ValueError(f"invalid shard_label: {label!r}")
        weights = self.weights
        if weights is not None:
            weights = tuple(float(value) for value in weights)
            if not weights or any(
                not math.isfinite(value) or value <= 0.0 for value in weights
            ):
                raise ValueError("MPI weights must be finite positive values")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True)
class _ArrayMeta:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


@dataclass(frozen=True)
class _WeightedPartition:
    rank: int
    size: int
    start: int
    count: int


def _partition_for(
    extent: int,
    weights: tuple[float, ...] | None,
    size: int,
    rank: int,
) -> Any:
    if weights is None:
        return _partition(extent, size, rank)
    if len(weights) != size:
        raise MpiConfigError(
            f"MPI weights count {len(weights)} does not match communicator size {size}"
        )
    total = math.fsum(weights)
    before = math.fsum(weights[:rank])
    through = math.fsum(weights[:rank + 1])
    start = min(extent, int(math.floor(extent * before / total)))
    end = extent if rank + 1 == size else min(
        extent, int(math.floor(extent * through / total))
    )
    return _WeightedPartition(rank, size, start, max(0, end - start))


def _validate_config(config: CudaMpiConfig, path_info: Any) -> str:
    output_labels = set(path_info.output_subscript)
    label = config.shard_label
    if label is None:
        raise MpiConfigError("no eligible CUDA MPI shard label")

    occurrences = sum(label in term for term in path_info.input_subscripts.split(","))
    if occurrences == 0:
        raise MpiConfigError(f"shard label {label!r} is absent from all inputs")
    if config.mode == "reduce" and label in output_labels:
        raise MpiConfigError("reduce mode requires a label absent from final output")
    if config.mode in ("batch", "free") and label not in output_labels:
        raise MpiConfigError("output-shard mode requires a label in final output")
    if config.mode == "batch" and occurrences < 2:
        raise MpiConfigError("batch mode requires at least two input occurrences")
    if config.mode == "free" and occurrences != 1:
        raise MpiConfigError("free mode requires exactly one input occurrence")
    return label


def _choose_reduction_label(path_info: Any) -> str:
    """Choose the eliminated label touching the largest planned work."""
    from ..helpers import flop_count

    output_labels = set(path_info.output_subscript)
    scores: dict[str, int] = {label: 0 for label in path_info.indices}
    for step in path_info.contraction_list:
        positions, removed, equation, _remaining, _blas = step
        left = equation.partition("->")[0]
        labels = set(left.replace(",", ""))
        cost = flop_count(
            labels, bool(removed), len(positions), path_info.size_dict
        )
        for label in labels & set(removed):
            scores[label] += int(cost)
    candidates = [
        label for label in path_info.indices
        if label not in output_labels and path_info.size_dict[label] > 1
    ]
    if not candidates:
        raise MpiConfigError("no eligible eliminated label for reduce sharding")
    return max(
        candidates,
        key=lambda label: (scores[label], -path_info.input_subscripts.index(label)),
    )


def _resolve_label(config: CudaMpiConfig, path_info: Any) -> str:
    if config.shard_label is not None:
        return _validate_config(config, path_info)
    output_labels = set(path_info.output_subscript)
    if config.mode == "reduce":
        label = _choose_reduction_label(path_info)
        resolved = CudaMpiConfig(config.mode, label, config.output)
    else:
        candidates = [
            label for label in path_info.output_subscript
            if path_info.size_dict[label] > 1
        ]
        if not candidates:
            raise MpiConfigError("no eligible output label for sharding")
        resolved = CudaMpiConfig(config.mode, candidates[0], config.output)
    return _validate_config(resolved, path_info)


def _slice_operands(
    arrays: Sequence[Any],
    terms: Sequence[str],
    label: str,
    start: int,
    count: int,
) -> list[Any]:
    import cupy

    local: list[Any] = []
    for index, (array, term) in enumerate(zip(arrays, terms)):
        value = cupy.asarray(array)
        _validate_term_rank(term, value)
        if label not in term:
            local.append(cupy.ascontiguousarray(value))
            continue
        axis = term.index(label)
        if value.shape[axis] == 1:
            local.append(cupy.ascontiguousarray(value))
            continue
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(start, start + count)
        local.append(cupy.ascontiguousarray(value[tuple(slices)]))
    return local


def _validate_term_rank(term: str, array: Any) -> None:
    """Reject a term whose equation rank differs from the array rank."""
    if len(term) != array.ndim:
        raise MpiConfigError(
            f"term {term!r} rank {len(term)} does not match input rank {array.ndim}"
        )


def _symmetric_errors(comm: Any, errors: Sequence[str | None], message: str) -> None:
    gathered = comm.allgather(errors[0])
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(gathered) if error]
    if failures:
        raise MpiConfigError(f"{message}: " + "; ".join(failures))


@dataclass
class _MpiProgram:
    config: CudaMpiConfig
    communicator: Any
    mpi: Any
    path_info: Any
    terms: tuple[str, ...]
    label: str
    partition: Any
    program: Any

    def execute(self, arrays: Sequence[Any]) -> Any:
        import cupy

        local_arrays = _slice_operands(
            arrays,
            self.terms,
            self.label,
            self.partition.start,
            self.partition.count,
        )
        local_result = self.program(local_arrays)
        cupy.cuda.Device().synchronize()
        if self.config.mode == "reduce":
            local_host = cupy.asnumpy(local_result)
            reduced = self.communicator.allreduce(local_host, op=self.mpi.SUM)
            return cupy.asarray(reduced)
        if self.config.output == "local":
            return local_result
        local_host = cupy.asnumpy(local_result)
        gathered = self.communicator.allgather(local_host)
        axis = self.path_info.output_subscript.index(self.label)
        return cupy.asarray(np.concatenate(gathered, axis=axis))


def _cache_key(
    subscripts: str,
    arrays: Sequence[Any],
    config: CudaMpiConfig,
    optimize: Any,
    communicator: Any,
) -> tuple[Any, ...]:
    import cupy

    return (
        subscripts,
        tuple(tuple(int(value) for value in array.shape) for array in arrays),
        tuple(str(array.dtype) for array in arrays),
        config,
        repr(optimize),
        id(communicator),
        int(communicator.Get_rank()),
        int(communicator.Get_size()),
        int(cupy.cuda.Device().id),
    )


def _build_program(
    subscripts: str,
    arrays: Sequence[Any],
    terms: tuple[str, ...],
    config: CudaMpiConfig,
    optimize: Any,
    communicator: Any,
    mpi: Any,
) -> _MpiProgram:
    from .._api import contract_path
    from .._native_plan import encode_execution_plan
    from ._program import CudaProgram

    _path, path_info = contract_path(subscripts, *arrays, optimize=optimize)
    label = _resolve_label(config, path_info)
    extent = path_info.size_dict[label]
    for index, (term, value) in enumerate(zip(terms, arrays)):
        _validate_term_rank(term, value)
        if label not in term:
            continue
        axis = term.index(label)
        if value.shape[axis] not in (1, extent):
            raise MpiConfigError(
                f"input {index} shard extent {value.shape[axis]} "
                f"does not match {extent}"
            )

    partition = _partition_for(
        extent, config.weights, communicator.Get_size(), communicator.Get_rank()
    )
    local_arrays = _slice_operands(
        arrays, terms, label, partition.start, partition.count
    )
    local_plan = encode_execution_plan(
        path_info.contraction_list,
        [_ArrayMeta(tuple(value.shape), value.dtype) for value in local_arrays],
    )
    return _MpiProgram(
        config,
        communicator,
        mpi,
        path_info,
        terms,
        label,
        partition,
        CudaProgram(local_plan),
    )


def clear_mpi_cache() -> None:
    """Release cached rank-local CUDA programs and their workspaces."""
    _PROGRAM_CACHE.clear()


def contract_mpi(
    subscripts: str,
    *operands: Any,
    mpi_config: CudaMpiConfig | None = None,
    optimize: Any = "greedy",
    communicator: Any = None,
) -> Any:
    """Execute one CUDA einsum with rank-local slicing and MPI combination."""
    import cupy

    config = mpi_config or CudaMpiConfig()
    MPI = _load_mpi()
    comm = MPI.COMM_WORLD if communicator is None else communicator
    if comm.Get_size() == 1:
        from .._api import contract

        return contract(subscripts, *operands, optimize=optimize, backend="cupy")

    local_error: str | None = None
    result = None
    try:
        arrays = [cupy.asarray(value) for value in operands]
        if not arrays:
            raise MpiConfigError("at least one operand is required")
        if any(value.dtype != arrays[0].dtype for value in arrays):
            raise MpiConfigError("mixed precision is not supported")
        terms = tuple(subscripts.partition("->")[0].split(","))
        key = _cache_key(subscripts, arrays, config, optimize, comm)
        program = _PROGRAM_CACHE.pop(key, None)
        if program is None:
            program = _build_program(
                subscripts, arrays, terms, config, optimize, comm, MPI
            )
        _PROGRAM_CACHE[key] = program
        while len(_PROGRAM_CACHE) > _PROGRAM_CACHE_LIMIT:
            _PROGRAM_CACHE.popitem(last=False)
        result = program.execute(arrays)
    except Exception as exc:  # noqa: BLE001 - all ranks must observe the same failure
        local_error = f"{type(exc).__name__}: {exc}"

    _symmetric_errors(comm, [local_error], "CUDA MPI einsum failed")
    if result is None:
        raise MpiConfigError("CUDA MPI einsum produced no result")
    return result


__all__ = ["CudaMpiConfig", "clear_mpi_cache", "contract_mpi"]
