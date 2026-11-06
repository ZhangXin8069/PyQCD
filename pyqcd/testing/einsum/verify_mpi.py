"""Exercise the KUPL MPI sharding path with real MPI ranks.

Examples::

    python -m pyqcd.testing.einsum.verify_mpi --single
    mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Callable

import numpy as np

from pyqcd.einsum import contract_path
from pyqcd.einsum._native_plan import encode_execution_plan
from pyqcd.einsum.kupl.einsum import compile_program
from pyqcd.einsum.kupl.mpi import (
    MPI_MODE_BATCH,
    MPI_MODE_FREE,
    MPI_MODE_REQUIRE,
    MPI_OUTPUT_LOCAL,
    MpiConfig,
    MpiConfigError,
    MpiUnavailable,
)

_DTYPES = (np.complex128, np.complex64, np.float64, np.float32)


def _operands(
    shapes: tuple[tuple[int, ...], ...], dtype: type[np.generic], seed: int
) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    values = []
    for shape in shapes:
        value = generator.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            value = value + 1j * generator.standard_normal(shape)
        values.append(np.asarray(value, dtype=dtype))
    return tuple(values)


def _plan(equation: str, operands: tuple[np.ndarray, ...]) -> bytes:
    _, path_info = contract_path(equation, *operands, optimize="greedy")
    return encode_execution_plan(path_info.contraction_list, operands)


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    dtype = np.dtype(actual.dtype)
    tolerance = 1e-10 if dtype in (np.dtype(np.complex128), np.dtype(np.float64)) else 2e-5
    if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
        difference = np.max(np.abs(actual - expected)) if actual.size else 0.0
        raise AssertionError(f"{label}: max|delta|={difference:.3e}")


def _partition(extent: int, size: int, rank: int) -> tuple[int, int]:
    base, remainder = divmod(extent, size)
    count = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
    return start, count


def _serial_checks() -> None:
    cases = (
        ("bij,bjk->bik", ((3, 2, 4), (3, 4, 5))),
        ("ij,jk->ik", ((4, 3), (3, 5))),
    )
    for dtype_index, dtype in enumerate(_DTYPES):
        for case_index, (equation, shapes) in enumerate(cases):
            operands = _operands(shapes, dtype, 20261106 + dtype_index * 10 + case_index)
            program = compile_program(_plan(equation, operands), mpi_config=MpiConfig("off"))
            actual = program(operands)
            expected = np.einsum(equation, *operands)
            _assert_close(actual, expected, f"serial {np.dtype(dtype).name} {equation}")


def _mpi_case(
    comm: Any,
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
    dtype: type[np.generic],
    config: MpiConfig,
    seed: int,
) -> Any:
    rank = int(comm.Get_rank())
    operands = _operands(shapes, dtype, seed)
    program = compile_program(_plan(equation, operands), mpi_config=config)
    session = program.mpi_executor
    if session is None:
        raise AssertionError("MPI configuration unexpectedly selected the serial path")
    output = program(operands)
    expected = np.einsum(equation, *operands)
    if int(config.output) == MPI_OUTPUT_LOCAL:
        start, count = _partition(expected.shape[session.shard_axis], comm.Get_size(), rank)
        slices = [slice(None)] * expected.ndim
        slices[session.shard_axis] = slice(start, start + count)
        target = expected[tuple(slices)]
        if output.shape != target.shape:
            raise AssertionError(f"local output shape {output.shape} != {target.shape}")
        _assert_close(output, target, f"local {np.dtype(dtype).name} {equation}")
    else:
        if output.shape != expected.shape:
            raise AssertionError(f"gathered output shape {output.shape} != {expected.shape}")
        _assert_close(output, expected, f"gather {np.dtype(dtype).name} {equation}")
    return output


def _expect_symmetric_failure(comm: Any, MPI: Any, action: Callable[[], Any], label: str) -> None:
    caught = 0
    try:
        action()
    except (MpiConfigError, MpiUnavailable):
        caught = 1
    all_caught = comm.allreduce(caught, op=MPI.MIN)
    if all_caught != 1:
        raise AssertionError(f"{label}: failure was not symmetric across ranks")


def _mpi_checks(comm: Any, MPI: Any) -> None:
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())
    if size != 2:
        raise AssertionError(f"verify_mpi requires exactly 2 ranks, got {size}")

    for dtype_index, dtype in enumerate(_DTYPES):
        _mpi_case(
            comm,
            "bij,bjk->bik",
            ((3, 2, 4), (3, 4, 5)),
            dtype,
            MpiConfig(MPI_MODE_BATCH, 0, "gather"),
            1000 + dtype_index,
        )
        for axis in (0, 1):
            _mpi_case(
                comm,
                "ij,jk->ik",
                ((4, 3), (3, 5)),
                dtype,
                MpiConfig(MPI_MODE_FREE, axis, "gather"),
                2000 + dtype_index * 10 + axis,
            )

    _mpi_case(
        comm,
        "ij,jk->ik",
        ((5, 3), (3, 4)),
        np.float64,
        MpiConfig(MPI_MODE_FREE, 0, "gather"),
        3001,
    )
    empty_operands = _operands(((1, 3), (3, 4)), np.float32, 3002)
    empty_plan = _plan("ij,jk->ik", empty_operands)
    empty_program = compile_program(
        empty_plan, mpi_config=MpiConfig(MPI_MODE_REQUIRE, 0, "gather")
    )
    _expected_start, expected_count = _partition(1, size, rank)
    if empty_program.mpi_executor is None or empty_program.mpi_executor.partition.count != expected_count:
        raise AssertionError("extent-1 partition is not 1/0 across two ranks")
    _assert_close(
        empty_program(empty_operands),
        np.einsum("ij,jk->ik", *empty_operands),
        "extent-1 empty rank",
    )

    _mpi_case(
        comm,
        "ij,jk->ik",
        ((4, 3), (3, 5)),
        np.float64,
        MpiConfig(MPI_MODE_FREE, 0, "local"),
        3003,
    )

    mismatch_operands = _operands(((4, 3), (3, 5)), np.float64, 4001)
    mismatch_plan = _plan("ij,jk->ik", mismatch_operands)
    _expect_symmetric_failure(
        comm,
        MPI,
        lambda: compile_program(
            mismatch_plan, mpi_config=MpiConfig(MPI_MODE_FREE, rank, "gather")
        )(mismatch_operands),
        "configuration mismatch",
    )

    bad_operands = _operands(((4, 3), (3, 5)), np.float64, 4002)
    bad_plan = _plan("ij,jk->ik", bad_operands)
    bad_program = compile_program(
        bad_plan, mpi_config=MpiConfig(MPI_MODE_FREE, 0, "gather")
    )
    bad_input = bad_operands[0][:-1] if rank == 1 else bad_operands[0]
    _expect_symmetric_failure(
        comm,
        MPI,
        lambda: bad_program((bad_input, bad_operands[1])),
        "bad rank-local shape",
    )


def _single_rank_checks(comm: Any, MPI: Any) -> None:
    _serial_checks()
    operands = _operands(((4, 3), (3, 5)), np.complex128, 5001)
    plan = _plan("ij,jk->ik", operands)
    serial = compile_program(plan, mpi_config=MpiConfig("off"))
    enabled = compile_program(
        plan, mpi_config=MpiConfig(MPI_MODE_FREE, 0, "gather")
    )
    if enabled is serial:
        raise AssertionError("program cache did not include the MPI configuration")
    if enabled.mpi_executor is not None:
        raise AssertionError("one-rank MPI configuration did not preserve the serial path")
    serial_output = serial(operands)
    enabled_output = enabled(operands)
    if not np.array_equal(serial_output, enabled_output):
        raise AssertionError("one-rank MPI path changed serial output bits")

    def median_seconds(program: Any) -> float:
        for _ in range(3):
            program(operands)
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            program(operands)
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    serial_time = median_seconds(serial)
    enabled_time = median_seconds(enabled)
    if comm.Get_rank() == 0:
        print(
            f"np=1 serial={serial_time * 1e3:.3f} ms "
            f"mpi-enabled={enabled_time * 1e3:.3f} ms"
        )
    del MPI


def main() -> int:
    parser = argparse.ArgumentParser(description="verify KUPL MPI axis sharding")
    parser.add_argument("--single", action="store_true", help="run without starting MPI ranks")
    args = parser.parse_args()

    if args.single:
        _serial_checks()
        print("PASS verify_mpi --single")
        return 0

    from mpi4py import MPI

    if not MPI.Is_initialized():
        MPI.Init_thread(MPI.THREAD_FUNNELED)
    if MPI.Query_thread() < MPI.THREAD_FUNNELED:
        raise RuntimeError("MPI_THREAD_FUNNELED is required")
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        _single_rank_checks(comm, MPI)
    else:
        _mpi_checks(comm, MPI)
    if comm.Get_rank() == 0:
        print(f"PASS verify_mpi ranks={comm.Get_size()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
