"""GPU（CuPy）性能对照：CUDA 特化 vs 通用路径 vs cupy.einsum。

与 CPU 版同一套纪律（见 ``measured-pitfalls.md`` §14）：

* **交错计时**：每轮三臂相邻各测一次，报每轮比值中位；
* **同步计时**：每次调用前后 `cp.cuda.Device(0).synchronize()`，否则只测到 kernel launch；
* **预热**：特化建步表、cupy.einsum 建计划缓存、通用路径建 path；
* **opt 臂只计执行**：cupy.einsum 不传优化器（它自身无 path 搜索），
  对照臂用 ``pe.contract(..., specialize=False)``（CPU/numpy 语义）与
  ``cp.einsum``（cuBLAS）。

用法::

    python -m pyqcd.testing._topology_perf_compare_gpu --sample 100 --dim 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np

REPEAT_BUDGET_SECONDS = 0.001
MAX_REPEAT = 50


def _shapes(equation: str, dim: int) -> list[tuple[int, ...]]:
    return [tuple(2 if ch.isupper() else dim for term in [t] for ch in term)
            for t in equation.split("->")[0].split(",")]


def _timed(call, cp) -> float:
    """一次调用的平均耗时（含同步），按预算自适应重复。"""
    cp.cuda.Device(0).synchronize()
    start = time.perf_counter()
    call()
    cp.cuda.Device(0).synchronize()
    probe = max(time.perf_counter() - start, 1e-9)
    repeat = max(1, min(MAX_REPEAT, int(REPEAT_BUDGET_SECONDS / probe)))
    start = time.perf_counter()
    for _ in range(repeat):
        call()
    cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - start) / repeat


def compare(sample: int | None = 100, dim: int = 4, rounds: int = 3,
            seed: int = 0, dtype_name: str = "float32",
            out: pathlib.Path | None = None) -> dict:
    import cupy as cp

    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasklist

    dtype = {"float32": np.float32, "float64": np.float64,
             "complex64": np.complex64, "complex128": np.complex128}[dtype_name]

    tasks = list(tasklist.TASKS)
    if sample is not None and sample < len(tasks):
        picked = np.random.default_rng(seed).choice(
            len(tasks), size=sample, replace=False)
        tasks = [tasks[index] for index in sorted(picked)]

    rows: list[dict] = []
    for task in tasks:
        rng = np.random.default_rng(task.seq)
        host = [rng.standard_normal(shape).astype(dtype)
                for shape in _shapes(task.equation, dim)]
        operands = [cp.asarray(item) for item in host]
        try:
            # 对照臂用 **opt_einsum**（backed by cupy），并且 path 事先求出复用：
            # opt_einsum 在 13–16 算子上的 greedy 搜索本身可达数十毫秒，混进计时
            # 会把"执行快"变成"规划慢"。cupy.einsum 不用：它对多算子是逐对朴素
            # 收缩（实测 13 算子 80+ ms/次），拿它当基线没有意义。
            import opt_einsum as oe

            path, _ = oe.contract_path(task.equation, *operands, optimize="greedy")
            reference_call = lambda: oe.contract(
                task.equation, *operands, optimize=path, backend="cupy")
            # 预热两次：cuBLAS handle 与 kernel 首次编译可达秒级。
            for _ in range(2):
                pe.contract(task.equation, *operands)
                pe.contract(task.equation, *operands, specialize=False)
                reference_call()
            cp.cuda.Device(0).synchronize()
        except Exception:                              # noqa: BLE001
            continue

        arms = (
            ("specialized", lambda: pe.contract(task.equation, *operands)),
            ("generic", lambda: pe.contract(task.equation, *operands,
                                           specialize=False)),
            ("opt_einsum", reference_call),
        )
        pairs_sg, pairs_sc = [], []
        measured = {}
        for _ in range(rounds):
            for name, call in arms:
                measured[name] = _timed(call, cp)
            pairs_sg.append(measured["specialized"] / measured["generic"])
            pairs_sc.append(measured["specialized"] / measured["opt_einsum"])

        # 正确性：与通用路径对照（同 dtype）
        expected = pe.contract(task.equation, *operands, specialize=False)
        got = pe.contract(task.equation, *operands)
        scale = max(float(cp.max(cp.abs(expected)).get()), 1e-300)
        deviation = float(cp.max(cp.abs(got - expected)).get()) / scale

        rows.append({
            "seq": task.seq, "tid": task.tid, "arity": task.arity,
            "specialized_us": measured["specialized"] * 1e6,
            "generic_us": measured["generic"] * 1e6,
            "opt_einsum_us": measured["opt_einsum"] * 1e6,
            "specialized_vs_generic": statistics.median(pairs_sg),
            "specialized_vs_opt_einsum": statistics.median(pairs_sc),
            "deviation": deviation,
        })

    def stat(key):
        values = [row[key] for row in rows]
        return {"median": statistics.median(values),
                "min": min(values), "max": max(values)} if values else {}

    by_arity: dict[int, list[dict]] = {}
    for row in rows:
        by_arity.setdefault(row["arity"], []).append(row)

    summary = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "dtype": dtype_name, "dim": dim, "rounds": rounds,
        "checked": len(rows),
        "specialized_vs_generic": stat("specialized_vs_generic"),
        "specialized_vs_opt_einsum": stat("specialized_vs_opt_einsum"),
        "absolute_us": {
            "specialized_median":
                statistics.median([r["specialized_us"] for r in rows]),
            "generic_median":
                statistics.median([r["generic_us"] for r in rows]),
            "opt_einsum_median":
                statistics.median([r["opt_einsum_us"] for r in rows]),
        } if rows else {},
        "max_deviation": max((r["deviation"] for r in rows), default=None),
        "by_arity": {
            str(arity): {
                "n": len(items),
                "vs_generic": statistics.median(
                    [i["specialized_vs_generic"] for i in items]),
                "vs_opt_einsum": statistics.median(
                    [i["specialized_vs_opt_einsum"] for i in items]),
                "specialized_us": statistics.median(
                    [i["specialized_us"] for i in items]),
            } for arity, items in sorted(by_arity.items())
        },
        "slowest": sorted(rows, key=lambda r: -r["specialized_vs_generic"])[:5],
    }
    if out is not None:
        target = pathlib.Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"summary": summary, "rows": rows},
                                     ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float64", "complex64", "complex128"])
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    sample = None if args.sample == 0 else args.sample
    summary = compare(sample=sample, dim=args.dim, rounds=args.rounds,
                      dtype_name=args.dtype,
                      out=pathlib.Path(args.json) if args.json else None)

    if not summary["checked"]:
        print("没有可测条目")
        return 1

    print(f"device={summary['device']}  dtype={summary['dtype']}  "
          f"dim={summary['dim']}  条数={summary['checked']}  "
          f"rounds={summary['rounds']}（含 CUDA 同步）")
    absolute = summary["absolute_us"]
    print()
    print(f"{'臂':26s} {'中位(us)':>10s}")
    print(f"{'pyqcd CUDA 特化':26s} {absolute['specialized_median']:10.1f}")
    print(f"{'pyqcd 通用':26s} {absolute['generic_median']:10.1f}")
    print(f"{'opt_einsum (cupy)':26s} {absolute['opt_einsum_median']:10.1f}")
    print()
    for key, label in (("specialized_vs_generic", "特化 / 通用"),
                       ("specialized_vs_opt_einsum", "特化 / opt_einsum")):
        item = summary[key]
        print(f"{label:22s} 中位 {item['median']:5.2f}x  "
              f"min {item['min']:5.2f}x  max {item['max']:5.2f}x")
    print(f"最大偏差（vs 通用）: {summary['max_deviation']:.2e}")
    print()
    print(f"{'元数':>4s} {'条数':>6s} {'vs 通用':>9s} {'vs opt_ens':>10s} "
          f"{'特化中位(us)':>13s}")
    for arity, item in summary["by_arity"].items():
        print(f"{arity:>4s} {item['n']:6d} {item['vs_generic']:8.2f}x "
              f"{item['vs_opt_einsum']:9.2f}x {item['specialized_us']:13.1f}")
    if summary["slowest"]:
        print()
        print("最慢 5 条：")
        for row in summary["slowest"]:
            print(f"  {row['specialized_vs_generic']:5.2f}x seq={row['seq']:5d} "
                  f"arity={row['arity']:2d} 特化 {row['specialized_us']:.1f}us / "
                  f"通用 {row['generic_us']:.1f}us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
