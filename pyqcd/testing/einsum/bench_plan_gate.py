"""Full-scene plan metrics and same-version A/B for the dedicated plan gate."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _kernel_cuda, _kernel_kupl
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum import bench_specialized
from pyqcd.einsum._bench_scene import dims


_NEGATIVE_CACHE_ENV = "PYQCD_EINSUM_NEGATIVE_CACHE"
_DEDICATED_PLAN_ENV = "PYQCD_EINSUM_DEDICATED_PLAN"
_DTYPE_IDS = {
    "complex128": 0,
    "complex64": 1,
    "float64": 2,
    "float32": 3,
}


def _elapsed(function, sync):
    sync()
    started = time.perf_counter()
    value = function()
    sync()
    return time.perf_counter() - started, value


def _relative_error(actual, expected, as_numpy):
    got = as_numpy(actual)
    ref = as_numpy(expected)
    scale = max(float(np.max(np.abs(ref), initial=0.0)), 1e-30)
    return float(np.max(np.abs(got - ref), initial=0.0) / scale)


def _kernel_module(device):
    return _kernel_cuda if device == "gpu" else _kernel_kupl


def _kernel_snapshot(device):
    module = _kernel_module(device)
    return {
        "info": dict(module.LAST_INFO),
        "failure": dict(module.LAST_FAILURE),
        "negative_cache": _kernel_kupl.negative_cache_info(),
    }


def command_metrics(args):
    by_seq = tasklist.by_seq()
    print(
        "PLAN_METRICS_HEADER="
        + json.dumps({
            "scene": "full",
            "dimensions": dims("full"),
            "seqs": args.seqs,
            "optimizers": args.optimizers,
            "dtype": args.dtype,
        }, sort_keys=True),
        flush=True,
    )
    failures = 0
    dtype = np.dtype(args.dtype)
    for seq in args.seqs:
        task = by_seq[seq]
        terms = task.equation.split("->", 1)[0].split(",")
        sizes = bench_specialized._label_sizes(terms, task.roles, dims("full"))
        shapes = [tuple(sizes[label] for label in term) for term in terms]
        for optimizer in args.optimizers:
            encoded = _kernel_kupl._dedicated_plan(
                task.equation, shapes, _DTYPE_IDS[args.dtype], optimizer)
            if encoded is None:
                row = {
                    "seq": seq,
                    "arity": task.arity,
                    "optimizer": optimizer,
                    "available": False,
                }
                failures += 1
            else:
                stats = _kernel_kupl._plan_statistics(encoded[0])
                row = {
                    "seq": seq,
                    "arity": task.arity,
                    "optimizer": optimizer,
                    "available": True,
                    "plan_bytes": len(encoded[0]),
                    **stats,
                    "complex128_cuda_gate": (
                        dtype == np.dtype(np.complex128)
                        and stats["plan_max_step_macs"] > 40_000_000_000
                    ),
                }
            print("PLAN_JSON=" + json.dumps(row, sort_keys=True), flush=True)
    return 1 if failures else 0


def _build_bench(device, precision):
    real, complex_, sync, release, device_name, configure = (
        bench_specialized._builders(device))
    configure(precision)
    as_numpy = (lambda value: value.get()) if device == "gpu" else np.asarray
    return real, complex_, sync, release, device_name, as_numpy


def _operands_for_seq(seq, real, complex_):
    task = tasklist.by_seq()[seq]
    builder = complex_ if seq % 2 else real
    operands = bench_specialized.operands_for(
        task.equation, dims("full"), builder, task.roles)
    return task, operands


def _set_negative_cache(enabled):
    os.environ[_NEGATIVE_CACHE_ENV] = "1" if enabled else "0"


def _set_dedicated_plan(optimizer):
    if optimizer is None:
        os.environ.pop(_DEDICATED_PLAN_ENV, None)
    else:
        os.environ[_DEDICATED_PLAN_ENV] = optimizer


def command_negative_cache_ab(args):
    real, complex_, sync, release, device_name, as_numpy = _build_bench(
        args.device, args.precision)
    by_seq = tasklist.by_seq()
    print(
        "NEGATIVE_CACHE_AB_HEADER="
        + json.dumps({
            "device": device_name,
            "scene": "full",
            "dimensions": dims("full"),
            "precision": args.precision,
            "seqs": args.seqs,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "optimize": "auto",
            "oracle": "pyqcd_generic_specialize_false",
        }, sort_keys=True),
        flush=True,
    )
    failures = 0
    try:
        for seq in args.seqs:
            task = by_seq[seq]
            builder = complex_ if seq % 2 else real
            operands = bench_specialized.operands_for(
                task.equation, dims("full"), builder, task.roles)
            reference = pe.contract(
                task.equation, *operands, specialize=False, optimize="auto")
            sync()

            arm_errors = {}
            arm_snapshots = {}
            for enabled in (False, True):
                _set_negative_cache(enabled)
                if enabled:
                    _kernel_kupl._negative_cache_clear()
                    pe.contract(task.equation, *operands, specialize=True,
                                optimize="auto")
                    sync()
                output = pe.contract(
                    task.equation, *operands, specialize=True, optimize="auto")
                sync()
                arm_errors[enabled] = _relative_error(output, reference, as_numpy)
                arm_snapshots[enabled] = _kernel_snapshot(args.device)
                del output
                release()

            del reference
            for enabled in (False, True):
                _set_negative_cache(enabled)
                for _ in range(args.warmup):
                    _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                    release()

            off_times = []
            on_times = []
            for round_index in range(args.rounds):
                if round_index % 2 == 0:
                    _set_negative_cache(False)
                    off, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                    _set_negative_cache(True)
                    on, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                else:
                    _set_negative_cache(True)
                    on, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                    _set_negative_cache(False)
                    off, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                off_times.append(off)
                on_times.append(on)
                release()

            off_median = statistics.median(off_times)
            on_median = statistics.median(on_times)
            paired = [on / off for on, off in zip(on_times, off_times)]
            row = {
                "seq": seq,
                "arity": task.arity,
                "cache_off_times_s": off_times,
                "cache_on_times_s": on_times,
                "cache_off_median_s": off_median,
                "cache_on_median_s": on_median,
                "ratio_on_over_off": on_median / off_median,
                "paired_ratio_median": statistics.median(paired),
                "improvement_fraction": 1.0 - on_median / off_median,
                "relative_errors": arm_errors,
                "snapshots": arm_snapshots,
            }
            print("NEGATIVE_CACHE_RESULT_JSON="
                  + json.dumps(row, sort_keys=True), flush=True)
            if max(arm_errors.values()) > (1e-5 if args.precision == "single" else 1e-10):
                failures += 1
            del operands
            release()
    finally:
        os.environ.pop(_NEGATIVE_CACHE_ENV, None)
    return 1 if failures else 0


def command_branch2_ab(args):
    real, complex_, sync, release, device_name, as_numpy = _build_bench(
        args.device, args.precision)
    by_seq = tasklist.by_seq()
    print(
        "BRANCH2_AB_HEADER="
        + json.dumps({
            "device": device_name,
            "scene": "full",
            "dimensions": dims("full"),
            "precision": args.precision,
            "seqs": args.seqs,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "baseline": "auto",
            "candidate": "branch-2",
            "oracle": "pyqcd_generic_specialize_false",
        }, sort_keys=True),
        flush=True,
    )
    failures = 0
    _set_negative_cache(True)
    try:
        for seq in args.seqs:
            task = by_seq[seq]
            builder = complex_ if seq % 2 else real
            operands = bench_specialized.operands_for(
                task.equation, dims("full"), builder, task.roles)
            reference = pe.contract(
                task.equation, *operands, specialize=False, optimize="auto")
            sync()

            arm_errors = {}
            arm_snapshots = {}
            for label, override in (("auto", None), ("branch-2", "branch-2")):
                _set_dedicated_plan(override)
                _kernel_kupl._negative_cache_clear()
                output = pe.contract(
                    task.equation, *operands, specialize=True,
                    optimize="auto")
                sync()
                arm_errors[label] = _relative_error(output, reference, as_numpy)
                arm_snapshots[label] = _kernel_snapshot(args.device)
                del output
                release()
            del reference

            for label, override in (("auto", None), ("branch-2", "branch-2")):
                _set_dedicated_plan(override)
                for _ in range(args.warmup):
                    _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                    release()

            auto_times = []
            branch2_times = []
            for round_index in range(args.rounds):
                if round_index % 2 == 0:
                    _set_dedicated_plan(None)
                    auto, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                    _set_dedicated_plan("branch-2")
                    branch2, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                else:
                    _set_dedicated_plan("branch-2")
                    branch2, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                    _set_dedicated_plan(None)
                    auto, _ = _elapsed(
                        lambda: pe.contract(
                            task.equation, *operands, specialize=True,
                            optimize="auto"),
                        sync,
                    )
                auto_times.append(auto)
                branch2_times.append(branch2)
                release()

            auto_median = statistics.median(auto_times)
            branch2_median = statistics.median(branch2_times)
            paired = [branch2 / auto for branch2, auto
                      in zip(branch2_times, auto_times)]
            row = {
                "seq": seq,
                "arity": task.arity,
                "auto_times_s": auto_times,
                "branch2_times_s": branch2_times,
                "auto_median_s": auto_median,
                "branch2_median_s": branch2_median,
                "ratio_branch2_over_auto": branch2_median / auto_median,
                "paired_ratio_median": statistics.median(paired),
                "improvement_fraction": 1.0 - branch2_median / auto_median,
                "relative_errors": arm_errors,
                "snapshots": arm_snapshots,
            }
            print("BRANCH2_RESULT_JSON="
                  + json.dumps(row, sort_keys=True), flush=True)
            if max(arm_errors.values()) > (1e-5 if args.precision == "single" else 1e-10):
                failures += 1
            del operands
            release()
    finally:
        os.environ.pop(_NEGATIVE_CACHE_ENV, None)
        os.environ.pop(_DEDICATED_PLAN_ENV, None)
    return 1 if failures else 0


def _seqs(value):
    return [int(item) for item in value.split(",") if item]


def _optimizers(value):
    return [item for item in value.split(",") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--seqs", type=_seqs, default=[6585])
    metrics.add_argument("--dtype", choices=tuple(_DTYPE_IDS), default="complex128")
    metrics.add_argument("--optimizers", type=_optimizers,
                         default=["auto", "greedy", "branch-2"])
    metrics.set_defaults(func=command_metrics)

    negative = subparsers.add_parser("negative-cache-ab")
    negative.add_argument("--device", choices=("cpu", "gpu"), required=True)
    negative.add_argument("--precision", choices=("single", "double"), default="double")
    negative.add_argument("--seqs", type=_seqs, default=[2584, 866, 1863, 6585])
    negative.add_argument("--rounds", type=int, default=10)
    negative.add_argument("--warmup", type=int, default=1)
    negative.set_defaults(func=command_negative_cache_ab)

    branch2 = subparsers.add_parser("branch2-ab")
    branch2.add_argument("--device", choices=("cpu", "gpu"), required=True)
    branch2.add_argument("--precision", choices=("single", "double"), default="double")
    branch2.add_argument("--seqs", type=_seqs, default=[6585])
    branch2.add_argument("--rounds", type=int, default=10)
    branch2.add_argument("--warmup", type=int, default=1)
    branch2.set_defaults(func=command_branch2_ab)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
