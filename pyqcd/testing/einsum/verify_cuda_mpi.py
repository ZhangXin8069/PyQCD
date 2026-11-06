"""Verify CUDA MPI reduction and output sharding with real collectives."""

from __future__ import annotations

import argparse

import numpy as np


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    return float(np.max(difference / scale, initial=0.0))


def run() -> int:
    from mpi4py import MPI
    import cupy

    from pyqcd.einsum.cuda.mpi import CudaMpiConfig, contract_mpi

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise RuntimeError("CUDA MPI verification requires at least two ranks")

    device_count = cupy.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RuntimeError("CUDA MPI verification requires a visible CUDA device")
    device = rank % device_count
    cupy.cuda.Device(device).use()

    rng = np.random.default_rng(20261106)
    checks = []

    def check(name: str, equation: str, arrays: list[np.ndarray], config: object) -> None:
        expected = np.einsum(equation, *arrays)
        gpu_arrays = [cupy.asarray(array) for array in arrays]
        actual_device = contract_mpi(
            equation,
            *gpu_arrays,
            mpi_config=config,
            optimize="greedy",
            communicator=comm,
        )
        actual = cupy.asnumpy(actual_device)
        error = _relative_error(actual, expected)
        checks.append((name, error))
        if error > 1e-12:
            if rank == 0:
                print(f"DEBUG {name} expected_shape={expected.shape} actual_shape={actual.shape}")
                print(f"DEBUG {name} expected_norm={np.linalg.norm(expected):.6e}")
                print(f"DEBUG {name} actual_norm={np.linalg.norm(actual):.6e}")
                print(f"DEBUG {name} expected[0]={expected.reshape(-1)[0]}")
                print(f"DEBUG {name} actual[0]={actual.reshape(-1)[0]}")
            raise AssertionError(f"{name}: relative error {error:.3e}")

    a = rng.normal(size=(8, 12)) + 1j * rng.normal(size=(8, 12))
    b = rng.normal(size=(12, 6)) + 1j * rng.normal(size=(12, 6))
    check(
        "reduce-matmul",
        "ab,bc->ac",
        [a, b],
        CudaMpiConfig(mode="reduce", shard_label="b"),
    )

    c = rng.normal(size=(3, 7, 10)) + 1j * rng.normal(size=(3, 7, 10))
    d = rng.normal(size=(10, 5, 4)) + 1j * rng.normal(size=(10, 5, 4))
    check(
        "reduce-chain",
        "abc,cde->abde",
        [c, d],
        CudaMpiConfig(mode="reduce", shard_label="c"),
    )

    e = rng.normal(size=(8, 2, 3)) + 1j * rng.normal(size=(8, 2, 3))
    f = rng.normal(size=(8, 3, 4)) + 1j * rng.normal(size=(8, 3, 4))
    check(
        "gather-batch",
        "bij,bjk->bik",
        [e, f],
        CudaMpiConfig(mode="batch", shard_label="b", output="gather"),
    )
    comm.Barrier()
    if rank == 0:
        for name, error in checks:
            print(f"PASS {name} max_rel={error:.3e}")
        print(f"cuda-mpi: {len(checks)} passed, 0 failed, ranks={size}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
