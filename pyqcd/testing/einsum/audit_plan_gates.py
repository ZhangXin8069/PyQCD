"""Audit CPU/CUDA dedicated plan gates over the complete production corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time

import numpy as np

from pyqcd.einsum import _kernel_cuda
from pyqcd.einsum import _kernel_kupl
from pyqcd.einsum import bench_specialized
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum._bench_scene import dims


def _stats(plan):
    return _kernel_kupl._plan_statistics(plan[0])


def audit(limit: int | None = None) -> int:
    tasks = tasklist.TASKS if limit is None else tasklist.TASKS[:limit]
    scene = dims("full")
    cpu_reasons: Counter[str] = Counter()
    cuda_reasons: Counter[str] = Counter()
    unavailable = 0
    started = time.perf_counter()
    max_cpu_arena = (0, 0)
    max_cuda_arena = (0, 0)
    max_cpu_step = (0, 0)
    max_cuda_step = (0, 0)

    print(
        "PLAN_GATE_AUDIT_HEADER="
        + json.dumps({
            "scene": "full",
            "dimensions": scene,
            "tasks": len(tasks),
        }, sort_keys=True),
        flush=True,
    )

    for task in tasks:
        terms = task.equation.split("->", 1)[0].split(",")
        sizes = bench_specialized._label_sizes(terms, task.roles, scene)
        shapes = [tuple(sizes[label] for label in term) for term in terms]
        dtype_id = 0 if task.seq % 2 else 2
        dtype = np.dtype(np.complex128 if dtype_id == 0 else np.float64)

        cpu_plan = _kernel_kupl._cheapest_dedicated_plan(
            task.equation, shapes, dtype_id, True)
        cuda_plan = _kernel_kupl._cheapest_dedicated_plan(
            task.equation, shapes, dtype_id, False)
        if cpu_plan is None or cuda_plan is None:
            unavailable += 1
            continue

        cpu_stats = _stats(cpu_plan)
        cuda_stats = _stats(cuda_plan)
        cpu_reason = _kernel_kupl._cpu_plan_fallback_reason(cpu_stats)
        cuda_reason = _kernel_cuda._plan_fallback_reason(dtype, cuda_stats)
        cpu_key = cpu_reason.split(" (", 1)[0] if cpu_reason else "allowed"
        cuda_key = cuda_reason.split(" (", 1)[0] if cuda_reason else "allowed"
        cpu_reasons[cpu_key] += 1
        cuda_reasons[cuda_key] += 1
        max_cpu_arena = max(max_cpu_arena, (cpu_stats["plan_arena_bytes"], task.seq))
        max_cuda_arena = max(max_cuda_arena, (cuda_stats["plan_arena_bytes"], task.seq))
        max_cpu_step = max(max_cpu_step, (cpu_stats["plan_max_step_macs"], task.seq))
        max_cuda_step = max(max_cuda_step, (cuda_stats["plan_max_step_macs"], task.seq))

        if cpu_reason or cuda_reason:
            print(
                "PLAN_GATE_BLOCK_JSON="
                + json.dumps({
                    "seq": task.seq,
                    "arity": task.arity,
                    "dtype": dtype.name,
                    "cpu": cpu_reason,
                    "cuda": cuda_reason,
                    "cpu_stats": cpu_stats,
                    "cuda_stats": cuda_stats,
                }, sort_keys=True),
                flush=True,
            )

    summary = {
        "tasks": len(tasks),
        "unavailable": unavailable,
        "cpu_blocked": sum(
            count for reason, count in cpu_reasons.items() if reason != "allowed"),
        "cuda_blocked": sum(
            count for reason, count in cuda_reasons.items() if reason != "allowed"),
        "cpu_reasons": dict(sorted(cpu_reasons.items())),
        "cuda_reasons": dict(sorted(cuda_reasons.items())),
        "max_cpu_arena_bytes": {
            "value": max_cpu_arena[0],
            "seq": max_cpu_arena[1],
        },
        "max_cuda_arena_bytes": {
            "value": max_cuda_arena[0],
            "seq": max_cuda_arena[1],
        },
        "max_cpu_step_macs": {
            "value": max_cpu_step[0],
            "seq": max_cpu_step[1],
        },
        "max_cuda_step_macs": {
            "value": max_cuda_step[0],
            "seq": max_cuda_step[1],
        },
        "elapsed_s": time.perf_counter() - started,
    }
    print("PLAN_GATE_AUDIT_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True),
          flush=True)
    return 0 if unavailable == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return audit(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
