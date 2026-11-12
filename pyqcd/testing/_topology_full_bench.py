"""全量 8638 条语料的一次性基准与数据快照（供第三方 einsum 抽样复验）。

设计约束：

* **只测 ``pyqcd.einsum.contract``**，不掺原生 kernel 分支以外的东西；
* 单路径目标 ~100 ms，总时限 **1000 s** 硬闸门（超时即停止并如实报告进度）；
* 每条记录耗时分位（min/median/max）；
* **保存输入与输出**（``.npz``，float32），供 ``numpy``/``opt_einsum``/``Einsums``
  等第三方实现读回复验。

用法::

    python -m pyqcd.testing._topology_full_bench                  # 全量，限时 1000 s
    python -m pyqcd.testing._topology_full_bench --limit 100      # 先跑 100 条
    python -m pyqcd.testing._topology_full_bench --dim 3 --time-limit 600
    python -m pyqcd.testing._topology_full_bench --out /tmp/snap   # 指定输出目录

产物（默认 ``<repo>/data/runs/topology-full-bench-<时间戳>/``）：

    manifest.json     每条：seq/tid/line/case/equation/shapes/dtype/耗时/输出文件
    inputs/<seq>.npz  该条的操作数（按 equation 顺序，键 a0,a1,…）
    outputs/<seq>.npz 该条的结果（键 out）
    summary.json      总计/成功/失败/超时/耗时统计
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np

DEFAULT_DIM = 3
DEFAULT_TIME_LIMIT = 1000.0
DTYPE = np.float32


def _shapes_for(equation: str, dim: int) -> list[tuple[int, ...]]:
    """按指标大小写给形状：大写=外置轴取 2，小写=dim。"""
    return [tuple(2 if ch.isupper() else dim for ch in term)
            for term in equation.split("->")[0].split(",")]


def _make_operands(shapes, rng) -> list[np.ndarray]:
    return [rng.standard_normal(shape).astype(DTYPE) for shape in shapes]


def run(limit: int | None = None, dim: int = DEFAULT_DIM,
        time_limit: float = DEFAULT_TIME_LIMIT, out: pathlib.Path | None = None,
        save: bool = True, quiet: bool = False) -> dict:
    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasklist

    tasks = list(tasklist.TASKS)
    if limit is not None:
        tasks = tasks[:limit]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if out is None:
        root = pathlib.Path(__file__).resolve().parents[2]
        out = root / "data" / "runs" / f"topology-full-bench-{stamp}"
    out = pathlib.Path(out)
    if save:
        (out / "inputs").mkdir(parents=True, exist_ok=True)
        (out / "outputs").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    durations: list[float] = []
    failures: list[dict] = []
    started = time.perf_counter()
    timed_out = False

    for position, task in enumerate(tasks):
        elapsed = time.perf_counter() - started
        if elapsed > time_limit:
            timed_out = True
            if not quiet:
                print(f"[timeout] 已用 {elapsed:.0f}s > {time_limit:.0f}s，"
                      f"在 seq={task.seq} 处停止（完成 {position}/{len(tasks)}）",
                      flush=True)
            break

        shapes = _shapes_for(task.equation, dim)
        rng = np.random.default_rng(task.seq)
        operands = _make_operands(shapes, rng)

        begin = time.perf_counter()
        try:
            result = pe.contract(task.equation, *operands)
            seconds = time.perf_counter() - begin
            ok = True
            error = ""
        except Exception as exc:                      # noqa: BLE001 - 记录而非抛出
            seconds = time.perf_counter() - begin
            result = None
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"seq": task.seq, "tid": task.tid,
                             "equation": task.equation, "error": error})

        record = {
            "seq": task.seq,
            "tid": task.tid,
            "line": task.line,
            "case": task.case,
            "subtable": task.subtable,
            "equation": task.equation,
            "arity": len(operands),
            "shapes": [list(shape) for shape in shapes],
            "dtype": np.dtype(DTYPE).name,
            "dim": dim,
            "seconds": round(seconds, 6),
            "ok": ok,
        }
        if ok:
            durations.append(seconds)
            record["output_shape"] = list(np.shape(result))
            if save:
                input_path = out / "inputs" / f"{task.seq:05d}.npz"
                output_path = out / "outputs" / f"{task.seq:05d}.npz"
                np.savez_compressed(
                    input_path, **{f"a{i}": operand
                                   for i, operand in enumerate(operands)})
                np.savez_compressed(output_path, out=result)
                record["input_file"] = str(input_path.relative_to(out))
                record["output_file"] = str(output_path.relative_to(out))
        manifest.append(record)

        if not quiet and (position + 1) % 500 == 0:
            print(f"  {position + 1}/{len(tasks)} 已完成，"
                  f"累计 {time.perf_counter() - started:.0f}s", flush=True)

    total_seconds = time.perf_counter() - started
    summary = {
        "dim": dim,
        "dtype": np.dtype(DTYPE).name,
        "requested": len(tasks),
        "completed": len(manifest),
        "ok": sum(1 for item in manifest if item["ok"]),
        "failed": len(failures),
        "timed_out": timed_out,
        "time_limit_s": time_limit,
        "total_seconds": round(total_seconds, 3),
        "per_path_seconds": {
            "min": round(min(durations), 6) if durations else None,
            "median": round(statistics.median(durations), 6) if durations else None,
            "max": round(max(durations), 6) if durations else None,
            "mean": round(statistics.fmean(durations), 6) if durations else None,
        },
        "pyqcd_version": __import__("pyqcd").__version__,
        "backend": pe.build_info().get("backend"),
        "created_at": stamp,
    }
    if save:
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "summary.json").write_text(
            json.dumps({"summary": summary, "failures": failures},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        summary["out_dir"] = str(out)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条（按 seq）")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"小写指标维度（默认 {DEFAULT_DIM}）")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                        help=f"总时限秒（默认 {DEFAULT_TIME_LIMIT:.0f}）")
    parser.add_argument("--out", default=None, help="输出目录")
    parser.add_argument("--no-save", action="store_true",
                        help="只测时间，不落盘输入/输出")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    summary = run(limit=args.limit, dim=args.dim, time_limit=args.time_limit,
                  out=pathlib.Path(args.out) if args.out else None,
                  save=not args.no_save, quiet=args.quiet)

    print()
    print(f"请求条数      : {summary['requested']}")
    print(f"实际完成      : {summary['completed']}")
    print(f"成功 / 失败   : {summary['ok']} / {summary['failed']}")
    print(f"是否超时中止  : {summary['timed_out']}")
    print(f"总耗时        : {summary['total_seconds']:.1f} s "
          f"(限 {summary['time_limit_s']:.0f} s)")
    per = summary["per_path_seconds"]
    if per["median"] is not None:
        print(f"单条耗时      : min {per['min']*1e3:.2f} ms / "
              f"median {per['median']*1e3:.2f} ms / "
              f"max {per['max']*1e3:.2f} ms")
    if summary.get("out_dir"):
        print(f"产物目录      : {summary['out_dir']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
