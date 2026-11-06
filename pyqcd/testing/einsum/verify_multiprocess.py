"""Verify the spawn-based KUPL multiprocessing fallback."""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

from pyqcd.einsum import contract_path
from pyqcd.einsum._native_plan import encode_execution_plan
from pyqcd.einsum.kupl.einsum import compile_program
from pyqcd.einsum.kupl.mpi import MpiConfig
from pyqcd.einsum.kupl.multiprocess import (
    MP_OUTPUT_LOCAL,
    MpConfig,
    MpConfigError,
    MpUnavailable,
)

_DTYPES = (np.complex128, np.complex64, np.float64, np.float32)


class Tally:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"PASS {label}")
        else:
            self.failed += 1
            print(f"FAIL {label}: {detail}")


def _operands(
    shapes: tuple[tuple[int, ...], ...], dtype: type[np.generic], seed: int
) -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(seed)
    result = []
    for shape in shapes:
        value = generator.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            value = value + 1j * generator.standard_normal(shape)
        result.append(np.asarray(value, dtype=dtype))
    return tuple(result)


def _plan(equation: str, operands: tuple[np.ndarray, ...]) -> bytes:
    _, path_info = contract_path(equation, *operands, optimize="greedy")
    return encode_execution_plan(path_info.contraction_list, operands)


def _serial(equation: str, operands: tuple[np.ndarray, ...]) -> tuple[Any, bytes]:
    plan = _plan(equation, operands)
    return compile_program(plan, mp_config=MpConfig("off"))(operands), plan


def _close(program: Any) -> None:
    session = getattr(program, "mp_executor", None)
    if session is not None:
        session.close()


def _assert_case(
    tally: Tally,
    label: str,
    equation: str,
    operands: tuple[np.ndarray, ...],
    config: MpConfig,
) -> np.ndarray:
    expected, plan = _serial(equation, operands)
    program = compile_program(plan, mp_config=config)
    try:
        actual = program(operands)
        if config.output_code == MP_OUTPUT_LOCAL:
            actual_gather = np.concatenate(actual, axis=program.mp_executor.shard_axis)
        else:
            actual_gather = actual
        reference = np.einsum(equation, *operands)
        error = float(np.max(np.abs(actual_gather - reference), initial=0.0))
        tally.check(
            np.array_equal(actual_gather, expected),
            label,
            f"not bitwise identical to serial output; max|delta|={error:.3e}",
        )
        return actual
    finally:
        _close(program)


def _check_config_and_serial(tally: Tally) -> None:
    operands = _operands(((3, 2, 4), (3, 4, 5)), np.complex128, 100)
    plan = _plan("bij,bjk->bik", operands)
    serial = compile_program(plan, mp_config=MpConfig("off"))
    workers_one = compile_program(
        plan, mp_config=MpConfig("auto", 0, "gather", workers=1)
    )
    off_two = compile_program(
        plan, mp_config=MpConfig("off", 0, "gather", workers=2)
    )
    expected = serial(operands)
    tally.check(
        serial.mp_executor is None
        and workers_one.mp_executor is None
        and off_two.mp_executor is None,
        "off/workers=1 preserve serial path",
    )
    tally.check(
        np.array_equal(workers_one(operands), expected)
        and np.array_equal(off_two(operands), expected),
        "serial fallback is bitwise identical",
    )
    tally.check(
        serial is compile_program(plan, mp_config=MpConfig("off")),
        "MP configuration participates in program cache key",
    )

    failed_closed = False
    try:
        compile_program(
            plan,
            mpi_config=MpiConfig("off"),
            mp_config=MpConfig("off"),
        )
    except TypeError:
        failed_closed = True
    tally.check(failed_closed, "MPI/MP configuration mutual exclusion")

    invalid_threads = False
    try:
        MpConfig("auto", 0, "gather", workers=2, threads_per_worker=2)
    except ValueError:
        invalid_threads = True
    tally.check(invalid_threads, "unsupported nested worker threading fails closed")


def _check_four_dtypes(tally: Tally) -> None:
    for index, dtype in enumerate(_DTYPES):
        dtype_name = np.dtype(dtype).name
        batch = _operands(
            ((3, 2, 4), (3, 4, 5)), dtype, 200 + index
        )
        _assert_case(
            tally,
            f"batch gather {dtype_name}",
            "bij,bjk->bik",
            batch,
            MpConfig("require", 0, "gather", workers=2),
        )

        free = _operands(((5, 3), (3, 4)), dtype, 300 + index)
        _assert_case(
            tally,
            f"free extent5 {dtype_name}",
            "ij,jk->ik",
            free,
            MpConfig("require", 0, "gather", workers=2),
        )


def _check_partitions_and_empty(tally: Tally) -> None:
    free = _operands(((5, 2), (2, 4)), np.float64, 400)
    plan = _plan("ij,jk->ik", free)
    program = compile_program(
        plan, mp_config=MpConfig("require", 0, "gather", workers=2)
    )
    try:
        counts = tuple(part.count for part in program.mp_executor.partitions)
        starts = tuple(part.start for part in program.mp_executor.partitions)
        tally.check(counts == (3, 2) and starts == (0, 3), "extent5 partitions are 3/2")
        program(free)
    finally:
        _close(program)

    one = _operands(((1, 2, 3), (1, 3, 4)), np.float64, 401)
    plan = _plan("bij,bjk->bik", one)
    program = compile_program(
        plan, mp_config=MpConfig("require", 0, "gather", workers=2)
    )
    try:
        counts = tuple(part.count for part in program.mp_executor.partitions)
        expected, serial_plan = _serial("bij,bjk->bik", one)
        del serial_plan
        actual = program(one)
        tally.check(
            counts == (1, 0) and np.array_equal(actual, expected),
            "extent1 creates one empty worker without changing output",
        )
    finally:
        _close(program)


def _check_output_modes(tally: Tally) -> None:
    operands = _operands(((5, 2), (2, 4)), np.complex128, 500)
    expected, plan = _serial("ij,jk->ik", operands)

    gather = compile_program(
        plan, mp_config=MpConfig("require", 0, "gather", workers=2)
    )
    try:
        tally.check(
            np.array_equal(gather(operands), expected),
            "gather returns ordered global output",
        )
    finally:
        _close(gather)

    local = compile_program(
        plan, mp_config=MpConfig("require", 0, "local", workers=2)
    )
    try:
        parts = local(operands)
        shapes = tuple(part.shape for part in parts)
        tally.check(
            shapes == ((3, 4), (2, 4))
            and np.array_equal(np.concatenate(parts, axis=0), expected),
            "local returns ordered worker shards",
        )
    finally:
        _close(local)

    lower_axis = _operands(((3, 5), (5, 4)), np.float64, 501)
    _assert_case(
        tally,
        "free shard can be a non-leading output axis",
        "ij,jk->ik",
        lower_axis,
        MpConfig("require", 1, "gather", workers=2),
    )


def _check_fail_closed(tally: Tally) -> None:
    operands = _operands(((4, 3), (3, 5)), np.float64, 600)
    plan = _plan("ij,jk->ik", operands)
    program = compile_program(
        plan, mp_config=MpConfig("require", 0, "gather", workers=2)
    )
    try:
        bad_shape = False
        try:
            program((operands[0][:-1], operands[1]))
        except MpConfigError:
            bad_shape = True
        tally.check(bad_shape, "bad input shape fails closed")

        bad_dtype = False
        try:
            program((operands[0].astype(np.float32), operands[1]))
        except MpConfigError:
            bad_dtype = True
        tally.check(bad_dtype, "mixed input dtype fails closed")
    finally:
        _close(program)

    bad_axis = False
    try:
        compile_program(
            plan, mp_config=MpConfig("require", 2, "gather", workers=2)
        )
    except MpConfigError:
        bad_axis = True
    tally.check(bad_axis, "invalid retained output axis fails closed")

    scalar = _operands(((3,), (3,)), np.float64, 601)
    scalar_plan = _plan("i,i->", scalar)
    scalar_failed = False
    try:
        compile_program(
            scalar_plan,
            mp_config=MpConfig("require", 0, "gather", workers=2),
        )
    except MpConfigError:
        scalar_failed = True
    tally.check(scalar_failed, "scalar output cannot be sharded")

    auto_scalar = compile_program(
        scalar_plan,
        mp_config=MpConfig("auto", 0, "gather", workers=2),
    )
    tally.check(
        auto_scalar.mp_executor is None
        and np.array_equal(auto_scalar(scalar), np.einsum("i,i->", *scalar)),
        "auto falls back when no safe MP axis exists",
    )


def _median_seconds(program: Any, operands: tuple[np.ndarray, ...], repeats: int) -> float:
    for _ in range(2):
        program(operands)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        program(operands)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _check_performance(tally: Tally) -> None:
    operands = _operands(((32, 64, 64), (32, 64, 64)), np.complex128, 700)
    expected, plan = _serial("bij,bjk->bik", operands)
    serial = compile_program(plan, mp_config=MpConfig("off"))
    parallel = compile_program(
        plan, mp_config=MpConfig("require", 0, "gather", workers=2)
    )
    try:
        parallel_output = parallel(operands)
        serial_time = _median_seconds(serial, operands, 5)
        parallel_time = _median_seconds(parallel, operands, 5)
        speedup = serial_time / parallel_time
        parent_threads = os.environ.get("OPENBLAS_NUM_THREADS", "default")
        print(
            "INFO performance "
            f"serial={serial_time * 1e3:.3f} ms "
            f"mp2={parallel_time * 1e3:.3f} ms speedup={speedup:.3f}x "
            f"parent_blas={parent_threads} mp_worker_blas=1"
        )
        tally.check(
            np.array_equal(parallel_output, expected),
            "performance case remains bitwise identical",
        )
    finally:
        _close(parallel)


def _check_mpi_guard(tally: Tally) -> None:
    code = r"""
import numpy as np
from mpi4py import MPI
from pyqcd.einsum import contract_path
from pyqcd.einsum._native_plan import encode_execution_plan
from pyqcd.einsum.kupl.einsum import compile_program
from pyqcd.einsum.kupl.multiprocess import MpConfig, MpUnavailable

assert MPI.Is_initialized()
left = np.ones((4, 3), dtype=np.float64)
right = np.ones((3, 5), dtype=np.float64)
_, info = contract_path("ij,jk->ik", left, right, optimize="greedy")
plan = encode_execution_plan(info.contraction_list, (left, right))
try:
    compile_program(plan, mp_config=MpConfig("require", 0, "gather", workers=2))
except MpUnavailable:
    pass
else:
    raise AssertionError("MP was enabled after MPI initialization")
program = compile_program(plan, mp_config=MpConfig("auto", 0, "gather", workers=2))
assert program.mp_executor is None
assert np.array_equal(program((left, right)), left @ right)
"""
    environment = os.environ.copy()
    environment["MPI4PY_RC_INITIALIZE"] = "true"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = result.stderr.strip() or result.stdout.strip()
    tally.check(result.returncode == 0, "MPI-initialized guard", detail)


def main() -> int:
    tally = Tally()
    _check_config_and_serial(tally)
    _check_four_dtypes(tally)
    _check_partitions_and_empty(tally)
    _check_output_modes(tally)
    _check_fail_closed(tally)
    _check_performance(tally)
    _check_mpi_guard(tally)
    print(f"multiprocess: {tally.passed} passed, {tally.failed} failed")
    return 0 if tally.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
