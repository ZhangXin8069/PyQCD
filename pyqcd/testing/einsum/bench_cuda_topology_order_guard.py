"""按生产尺寸（``distill``）实测 CUDA 拓扑路径的端到端耗时与路由此举。

用途：量化「表内序病态即 fail closed」守卫的**生产口径**收益 —— 同一条语料方程，
守卫放行时由原生步表执行（表内序），守卫拒绝时回落带真实 path 搜索的通用路径。
两类格子都如实标出实际路由，不剔除、不换小尺寸代替。

口径
----
``dims("distill")``（``Nev=100`` / ``Ns=4`` / ``Mom=1``），轴长由语料 ``roles``
经 :func:`pyqcd.einsum.bench_specialized._label_sizes` 反推。这是本库对蒸馏收缩
唯一认可的性能口径；**不**用 smoke / 等长超立方体。

``--label`` 只写进 JSON，用来区分「同一脚本在不同构建上跑的两次测量」（例如
把守卫阈值临时放大以复现修复前的行为）；数字本身不做任何归一化。

用法::

    python -m pyqcd.testing.einsum.bench_cuda_topology_order_guard
    python -m pyqcd.testing.einsum.bench_cuda_topology_order_guard \
        --seqs 510,2565,869,1859,3,853 --rounds 3 --label guard-off

单条方程的排查请用 ``python -m pyqcd.einsum.test --help``（2026-11-15 起的唯一合法
入口，口径以它的缺省值为准）；本脚本负责守卫开关的批量端到端对比。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np

from pyqcd.einsum import contract
from pyqcd.einsum import _kernel_cuda
from pyqcd.einsum import _kernel_kupl
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum._bench_scene import dims as scene_dims
from pyqcd.einsum.bench_specialized import _label_sizes
from pyqcd.testing._topology_kernel_gen import parse

ARTIFACTS = pathlib.Path(__file__).resolve().parent / "artifacts"

#: 默认样本：4 条病态（守卫应拒绝）+ 2 条良性（守卫应放行），元数 9..16 覆盖。
DEFAULT_SEQS = (510, 2565, 869, 1859, 3, 853)


def latest_artifact() -> pathlib.Path:
    candidates = sorted(ARTIFACTS.glob("cuda_topology_order_*"))
    if not candidates:
        raise SystemExit("找不到量化产物；先运行 "
                         "python -m pyqcd.testing.einsum.analyze_cuda_topology_order")
    return candidates[-1]


def analysis_rows(directory: pathlib.Path) -> dict[int, dict[str, Any]]:
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    return {int(row["seq"]): row for row in payload["rows"] if row.get("ratio")}


def operands_for(seq: int, dtype: str, dims: dict[str, int]):
    task = tasklist.lookup(seq)
    if task is None:
        raise SystemExit(f"unknown seq {seq}")
    terms, _output = parse(task.equation)
    sizes = _label_sizes(terms, task.roles, dims)
    shapes = [tuple(sizes[label] for label in term) for term in terms]
    rng = np.random.default_rng(seq)
    if np.dtype(dtype).kind == "c":
        return [(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                .astype(dtype) for shape in shapes]
    return [rng.standard_normal(shape).astype(dtype) for shape in shapes]


def run(seqs: Sequence[int], rounds: int, dtype: str, label: str,
        directory: pathlib.Path, quiet: bool) -> int:
    try:
        import cupy
    except ImportError:
        print("SKIP: cupy 不可用")
        return 0
    table = analysis_rows(directory)
    dims = scene_dims("distill")
    cells: list[dict[str, Any]] = []
    for seq in seqs:
        task = tasklist.lookup(seq)
        assert task is not None
        operands = operands_for(seq, dtype, dims)
        devices = [cupy.asarray(item) for item in operands]
        times: list[float] = []
        routes: list[str] = []
        reason = ""
        try:
            _kernel_kupl._NEGATIVE_CACHE.clear()
            _kernel_cuda.LAST_INFO.clear()
            _kernel_cuda.LAST_FAILURE.update({"reason": "", "status": None})
            contract(task.equation, *devices)          # 首调：含计划构建/编码，丢弃
            for _ in range(rounds):
                _kernel_cuda.LAST_INFO.clear()
                _kernel_cuda.LAST_FAILURE.update({"reason": "", "status": None})
                started = time.perf_counter()
                contract(task.equation, *devices)
                times.append(time.perf_counter() - started)
                routes.append(str(_kernel_cuda.LAST_INFO.get("path")
                                  or _kernel_cuda.LAST_FAILURE.get("reason") or "?"))
            reason = routes[-1] if routes else "?"
        except Exception as error:                     # noqa: BLE001
            reason = f"{type(error).__name__}: {error}"
        row = table.get(seq, {})
        cell = {
            "seq": seq,
            "arity": row.get("arity"),
            "equation": task.equation,
            "route": reason,
            "median_seconds": statistics.median(times) if times else None,
            "rounds": len(times),
            "table_mac": row.get("table_mac"),
            "auto_mac": row.get("auto_mac"),
            "ratio_vs_greedy": row.get("ratio_vs_greedy"),
            "table_largest": row.get("table_largest"),
        }
        cells.append(cell)
        if not quiet:
            seconds = (f"{cell['median_seconds']:.4f}s"
                       if cell["median_seconds"] is not None else "failed")
            print(f"seq={seq:5d} arity={cell['arity']:2d} {seconds:>9s} "
                  f"route={str(reason)[:48]:48s} "
                  f"table/auto={row.get('ratio', float('nan')):.4g}")
        del operands, devices
        cupy.get_default_memory_pool().free_all_blocks()
    report = {
        "label": label,
        "stamp": time.strftime("%Y%m%d%H%M%S"),
        "dtype": dtype,
        "dims": {key: dims[key] for key in ("Nev", "Ns", "Mom", "Lt", "Nx")},
        "rounds": rounds,
        "cells": cells,
    }
    target = directory / f"bench_guard_{label}_{report['stamp']}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        print(f"report: {target}")
    return 0 if all(cell["median_seconds"] is not None for cell in cells) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default=",".join(str(item) for item in DEFAULT_SEQS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dtype", default="complex128")
    parser.add_argument("--label", default="guard-on")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    directory = (pathlib.Path(args.artifact) if args.artifact else latest_artifact())
    seqs = tuple(int(item) for item in args.seqs.split(",") if item.strip())
    return run(seqs, args.rounds, args.dtype, args.label, directory, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
