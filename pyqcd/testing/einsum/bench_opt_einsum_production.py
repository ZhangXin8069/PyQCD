"""Compare PyQCD specialized/generic paths with opt_einsum on real equations."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import bench_specialized
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum._bench_scene import dims
from pyqcd.testing.einsum import _oracle


def _path_from_output(text: str) -> str:
    if "kernel:" in text:
        return text.split("kernel:", 1)[1].split("]", 1)[0].strip()
    if "plan-cache" in text:
        return "plan-cache"
    return "unknown"


def _as_numpy(value):
    if type(value).__module__.split(".", 1)[0] == "cupy":
        import cupy

        cupy.cuda.Stream.null.synchronize()
        return cupy.asnumpy(value)
    return np.asarray(value)


def _relative_error(actual, expected) -> float:
    got = _as_numpy(actual)
    ref = _as_numpy(expected)
    scale = max(float(np.max(np.abs(ref), initial=0.0)), 1e-30)
    return float(np.max(np.abs(got - ref), initial=0.0) / scale)


def _elapsed(call, sync) -> float:
    sync()
    started = time.perf_counter()
    result = call()
    sync()
    elapsed = time.perf_counter() - started
    return elapsed, result


def _measure_arms(calls, sync, release, warmup: int, rounds: int):
    for call in calls.values():
        output = call()
        del output
        release()
    for _ in range(warmup):
        for call in calls.values():
            output = call()
            del output
            release()

    times = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(rounds):
        order = names[round_index % len(names):] + names[:round_index % len(names)]
        for name in order:
            elapsed, output = _elapsed(calls[name], sync)
            times[name].append(elapsed)
            del output
            release()
    return times


def run(device: str, seqs: list[int], precision: str, rounds: int,
        warmup: int, strategies: list[str], out: Path) -> int:
    oracle = _oracle.oracle()
    oracle_version, oracle_path = _oracle.oracle_version()
    real, complex_, sync, release, device_name, configure = (
        bench_specialized._builders(device))
    configure(precision)
    scene = dims("full")
    by_seq = tasklist.by_seq()
    rtol = 1e-5 if precision == "single" else 1e-10
    payload = {
        "device": device_name,
        "scene": "full",
        "dimensions": scene,
        "precision": precision,
        "rounds": rounds,
        "warmup": warmup,
        "opt_einsum_version": oracle_version,
        "opt_einsum_path": oracle_path,
        "strategies": strategies,
        "records": [],
    }
    failures = 0
    for seq in seqs:
        task = by_seq[seq]
        builder = complex_ if seq % 2 else real
        operands = bench_specialized.operands_for(
            task.equation, scene, builder, task.roles)

        reference = pe.contract(
            task.equation, *operands, specialize=False, optimize="auto")
        sync()

        calls = {
            "pyqcd_generic": lambda: pe.contract(
                task.equation, *operands, specialize=False, optimize="auto"),
            "qcd_specialized": lambda: pe.contract(
                task.equation, *operands, specialize=True, optimize="auto"),
        }
        for strategy in strategies:
            calls[f"opt_einsum_{strategy}"] = (
                lambda strategy=strategy: oracle.contract(
                    task.equation, *operands, optimize=strategy))

        errors = {}
        specialized_output = io.StringIO()
        for name, call in calls.items():
            if name == "qcd_specialized":
                with contextlib.redirect_stdout(specialized_output):
                    output = call()
            else:
                output = call()
            sync()
            errors[name] = _relative_error(output, reference)
            del output
            release()
        del reference
        release()

        try:
            times = _measure_arms(calls, sync, release, warmup, rounds)
        except Exception as error:  # noqa: BLE001 - preserve per-path failure
            print(
                "OPT_EINSUM_FAILURE_JSON="
                + json.dumps({
                    "seq": seq,
                    "error": f"{type(error).__name__}: {error}",
                }, sort_keys=True),
                flush=True,
            )
            failures += 1
            del operands
            release()
            continue

        medians = {name: statistics.median(values)
                   for name, values in times.items()}
        path = _path_from_output(specialized_output.getvalue())
        record = {
            "seq": seq,
            "arity": task.arity,
            "equation": task.equation,
            "path": path,
            "times_s": times,
            "medians_s": medians,
            "relative_errors": errors,
        }
        for strategy in strategies:
            opt_name = f"opt_einsum_{strategy}"
            record[f"qcd_specialized_over_{opt_name}"] = (
                medians["qcd_specialized"] / medians[opt_name])
            record[f"pyqcd_generic_over_{opt_name}"] = (
                medians["pyqcd_generic"] / medians[opt_name])
        payload["records"].append(record)
        print("OPT_EINSUM_RESULT_JSON=" + json.dumps(record, sort_keys=True),
              flush=True)
        if max(errors.values()) > rtol:
            failures += 1
        del operands
        release()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}: {len(payload['records'])} records", flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--precision", choices=("single", "double"),
                        default="double")
    parser.add_argument("--seqs", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--strategies", default="dp,greedy")
    parser.add_argument("--oracle", choices=("frozen", "installed"),
                        default="frozen")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PYQCD_EINSUM_ORACLE"] = args.oracle
    seqs = [int(item) for item in args.seqs.split(",") if item]
    strategies = [item for item in args.strategies.split(",") if item]
    return run(args.device, seqs, args.precision, args.rounds, args.warmup,
               strategies, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
