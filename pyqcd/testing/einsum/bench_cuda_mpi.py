"""Benchmark CUDA MPI reduction sharding on the distillation equations."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

from . import bench_distillation as bench

SHARD_LABELS = {"E1": "n", "E2": "b"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="T1", choices=("T1", "T2"))
    parser.add_argument("--equations", default="E1,E2")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--outdir")
    return parser.parse_args()


def _run() -> int:
    from mpi4py import MPI
    import cupy
    import opt_einsum

    from pyqcd.einsum.cuda.mpi import CudaMpiConfig, clear_mpi_cache, contract_mpi

    args = _arguments()
    if args.iters < 1 or args.warmup < 0:
        raise ValueError("iters must be positive and warmup non-negative")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        raise RuntimeError("CUDA MPI benchmark requires at least two ranks")
    device_count = cupy.cuda.runtime.getDeviceCount()
    if device_count < 1:
        raise RuntimeError("no CUDA device is visible")
    configured_devices = os.environ.get("PYQCD_CUDA_MPI_DEVICES", "")
    if configured_devices:
        device_ids = [
            int(value) for value in configured_devices.split(",") if value.strip()
        ]
        if not device_ids or any(value < 0 or value >= device_count for value in device_ids):
            raise ValueError("PYQCD_CUDA_MPI_DEVICES contains an invalid device id")
    else:
        device_ids = list(range(device_count))
    cupy.cuda.Device(device_ids[rank % len(device_ids)]).use()
    configured_weights = os.environ.get("PYQCD_CUDA_MPI_WEIGHTS", "")
    weights = (
        tuple(float(value) for value in configured_weights.split(",") if value.strip())
        if configured_weights
        else None
    )
    device_name = cupy.cuda.runtime.getDeviceProperties(
        cupy.cuda.Device().id
    )["name"].decode()

    profile = bench.PROFILES[args.profile]
    equations = tuple(name.strip() for name in args.equations.split(",") if name.strip())
    unknown = sorted(set(equations) - set(SHARD_LABELS))
    if unknown:
        raise ValueError(f"unsupported CUDA MPI benchmark equations: {unknown}")

    rows = []
    for t_src in profile["t_src"]:
        peram = bench.load_peram(t_src, profile["nvec"])
        for name in equations:
            meta = bench.EQUATION_BY_NAME[name]
            arrays_host = bench.build_operands(name, peram, profile, seed=1000 + 17 * t_src)
            arrays = [cupy.asarray(array) for array in arrays_host]
            config = CudaMpiConfig(
                mode="reduce",
                shard_label=SHARD_LABELS[name],
                output="gather",
                weights=weights,
            )

            reference = opt_einsum.contract(
                meta["eq"], *arrays, optimize="greedy", backend="cupy"
            )
            cupy.cuda.Device().synchronize()

            for _ in range(args.warmup):
                warm = contract_mpi(
                    meta["eq"], *arrays, mpi_config=config, optimize="greedy", communicator=comm
                )
                comm.Barrier()
                del warm

            mpi_samples = []
            for _ in range(args.iters):
                comm.Barrier()
                start = time.perf_counter()
                result = contract_mpi(
                    meta["eq"], *arrays, mpi_config=config, optimize="greedy", communicator=comm
                )
                comm.Barrier()
                mpi_samples.append(time.perf_counter() - start)
                del result

            opt_samples = []
            if rank == 0:
                for _ in range(args.iters):
                    cupy.cuda.Device().synchronize()
                    start = time.perf_counter()
                    output = opt_einsum.contract(
                        meta["eq"], *arrays, optimize="greedy", backend="cupy"
                    )
                    cupy.cuda.Device().synchronize()
                    opt_samples.append(time.perf_counter() - start)
                    del output

            actual = contract_mpi(
                meta["eq"], *arrays, mpi_config=config, optimize="greedy", communicator=comm
            )
            actual_host = cupy.asnumpy(actual)
            reference_host = cupy.asnumpy(reference)
            relative_error = bench._max_relative_error(actual_host, reference_host)
            if relative_error > bench.RELATIVE_ERROR_TOLERANCE:
                raise AssertionError(f"{name}: relative error {relative_error:.3e}")

            rank_medians = comm.allgather(statistics.median(mpi_samples))
            device_names = comm.allgather(device_name)
            if rank == 0:
                opt_median = statistics.median(opt_samples)
                mpi_median = statistics.median(mpi_samples)
                speedup = opt_median / mpi_median
                rows.append(
                    {
                        "profile": args.profile,
                        "eq": name,
                        "ranks": size,
                        "device_count": device_count,
                        "devices": device_names,
                        "rank_median_s": rank_medians,
                        "shard_label": SHARD_LABELS[name],
                        "weights": weights,
                        "opt_median_s": opt_median,
                        "qcd_mpi_median_s": mpi_median,
                        "speedup": speedup,
                        "max_relative_error": relative_error,
                        "pass": speedup >= bench.SPEEDUP_TARGET,
                    }
                )
            clear_mpi_cache()
            cupy.get_default_memory_pool().free_all_blocks()
            del arrays, arrays_host, reference, actual
        del peram

    passed = all(row["pass"] for row in rows)
    if rank == 0:
        outdir = Path(args.outdir or f"/tmp/pyqcd-cuda-mpi-{datetime.now():%Y%m%d%H%M%S}")
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "benchmark": "qcd-einsum CUDA MPI vs single-GPU opt_einsum",
            "provisional": True,
            "target_speedup": bench.SPEEDUP_TARGET,
            "rows": rows,
            "passed": passed,
        }
        (outdir / "bench.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for row in rows:
            state = "PASS" if row["pass"] else "FAIL"
            print(
                f"{row['eq']}/gpu-mpi ranks={size}: "
                f"opt={row['opt_median_s']:.6f}s "
                f"qcd={row['qcd_mpi_median_s']:.6f}s "
                f"speedup={row['speedup']:.3f}x "
                f"ranks_ms={[round(value * 1000, 3) for value in row['rank_median_s']]} "
                f"max_rel={row['max_relative_error']:.3e} {state}"
            )
        print(f"artifact: {outdir / 'bench.json'}")
    comm.Barrier()
    return 0 if passed else 1


def main() -> int:
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
