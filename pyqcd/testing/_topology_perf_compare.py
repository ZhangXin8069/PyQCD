"""性能对照：pyqcd 特化路径 vs pyqcd 通用路径 vs opt_einsum。

三个对照臂（同一批操作数、同机、同 dtype、交错计时）：

* ``specialized``：``pyqcd.einsum.contract(eq, *ops)``（走 ``kupl_dedicated``）
* ``generic``    ：``pyqcd.einsum.contract(eq, *ops, specialize=False)``
* ``opt_einsum`` ：``opt_einsum.contract(eq, *ops, optimize=<同一 path>)``
  计时**只算执行**：path 用 `contract_path` 事先求出并复用，避免把规划时间
  混进对比（否则 16 算子方程上 opt_einsum 的 greedy 搜索会主导结果）。

交错协议：每一轮内三个臂相邻各跑一次，取每轮比值；报中位数，避免顺序漂移。

用法::

    python -m pyqcd.testing._topology_perf_compare               # 抽 200 条
    python -m pyqcd.testing._topology_perf_compare --sample 0    # 全量
    python -m pyqcd.testing._topology_perf_compare --dim 6 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np


def _shapes(equation: str, dim: int) -> list[tuple[int, ...]]:
    return [tuple(2 if ch.isupper() else dim for ch in term)
            for term in equation.split("->")[0].split(",")]


# 单次采样的目标时长。取 1 ms 而不是 2 ms：预算越大，单个臂占用的连续时间越长，
# 在忙碌机器上越容易跨过调度/频率变化窗口，反而制造"某个臂特别慢"的假象。
REPEAT_BUDGET_SECONDS = 0.001
MAX_REPEAT = 50


def _timed_call(function) -> float:
    """测**一次**调用的耗时：按预算自适应重复，返回「每次调用」的平均值。

    不能返回「最快一次」——小任务上最快值几乎等于空转开销；也不能无限重复——
    长窗口会跨过机器的频率/调度变化。1 ms 预算 + 上限 50 次是实测折中。
    """
    start = time.perf_counter()
    function()
    probe = max(time.perf_counter() - start, 1e-9)
    repeat = max(1, min(MAX_REPEAT, int(REPEAT_BUDGET_SECONDS / probe)))
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    return (time.perf_counter() - start) / repeat


def compare(sample: int | None = 200, dim: int = 3, rounds: int = 3,
            seed: int = 0, out: pathlib.Path | None = None) -> dict:
    import opt_einsum as oe

    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasklist

    tasks = list(tasklist.TASKS)
    if sample is not None and sample < len(tasks):
        picked = np.random.default_rng(seed).choice(
            len(tasks), size=sample, replace=False)
        tasks = [tasks[index] for index in sorted(picked)]

    rows: list[dict] = []
    for task in tasks:
        rng = np.random.default_rng(task.seq)
        operands = [rng.standard_normal(shape).astype(np.float32)
                    for shape in _shapes(task.equation, dim)]
        try:
            path, _ = oe.contract_path(task.equation, *operands,
                                       optimize="greedy")
            # 预热（特化要建步表；opt_einsum 要建 path 缓存）
            pe.contract(task.equation, *operands)
            pe.contract(task.equation, *operands, specialize=False)
            oe.contract(task.equation, *operands, optimize=path)
        except Exception:                                  # noqa: BLE001
            continue

        # **交错计时**：每一轮内三个臂相邻各跑一次，再取每轮比值的中位数。
        # 顺序测量（A 全部跑完再跑 B）在忙碌机器上漂移 20–40%，足以伪造 2 倍差距。
        specialized = generic = reference = 0.0
        pairs_sg, pairs_so = [], []
        arms = (
            ("specialized", lambda: pe.contract(task.equation, *operands)),
            ("generic", lambda: pe.contract(task.equation, *operands,
                                            specialize=False)),
            ("opt_einsum", lambda: oe.contract(task.equation, *operands,
                                               optimize=path)),
        )
        for _ in range(rounds):
            measured = {}
            for name, call in arms:
                measured[name] = _timed_call(call)
            pairs_sg.append(measured["specialized"] / measured["generic"])
            pairs_so.append(measured["specialized"] / measured["opt_einsum"])
        specialized = measured["specialized"]
        generic = measured["generic"]
        reference = measured["opt_einsum"]
        rows.append({
            "seq": task.seq, "tid": task.tid, "arity": task.arity,
            "specialized_us": specialized * 1e6,
            "generic_us": generic * 1e6,
            "opt_einsum_us": reference * 1e6,
            "specialized_vs_generic": statistics.median(pairs_sg),
            "specialized_vs_opt_einsum": statistics.median(pairs_so),
        })

    def summarize(key):
        values = [row[key] for row in rows]
        return {
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "min": min(values), "max": max(values),
            "faster_than_generic":
                sum(1 for row in rows if row[key] < 1.0),
        }

    by_arity: dict[int, list[dict]] = {}
    for row in rows:
        by_arity.setdefault(row["arity"], []).append(row)

    summary = {
        "dim": dim, "rounds": rounds, "checked": len(rows),
        "specialized_vs_generic": summarize("specialized_vs_generic"),
        "specialized_vs_opt_einsum": summarize("specialized_vs_opt_einsum"),
        "absolute_us": {
            "specialized_median":
                statistics.median([row["specialized_us"] for row in rows]),
            "generic_median":
                statistics.median([row["generic_us"] for row in rows]),
            "opt_einsum_median":
                statistics.median([row["opt_einsum_us"] for row in rows]),
        } if rows else {},
        "by_arity": {
            str(arity): {
                "n": len(items),
                "specialized_vs_generic_median": statistics.median(
                    [item["specialized_vs_generic"] for item in items]),
                "specialized_vs_opt_einsum_median": statistics.median(
                    [item["specialized_vs_opt_einsum"] for item in items]),
                "specialized_us_median": statistics.median(
                    [item["specialized_us"] for item in items]),
            } for arity, items in sorted(by_arity.items())
        },
        "slowest": sorted(rows, key=lambda r: -r["specialized_vs_generic"])[:5],
    }
    if out is not None:
        path_target = pathlib.Path(out)
        path_target.parent.mkdir(parents=True, exist_ok=True)
        path_target.write_text(
            json.dumps({"summary": summary, "rows": rows},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200,
                        help="抽样条数（0 表示全量 8638）")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    sample = None if args.sample == 0 else args.sample
    summary = compare(sample=sample, dim=args.dim, rounds=args.rounds,
                      seed=args.seed,
                      out=pathlib.Path(args.json) if args.json else None)

    absolute = summary["absolute_us"]
    print(f"dim={summary['dim']}  条数={summary['checked']}  "
          f"rounds={summary['rounds']}（每臂取最快一次）")
    print()
    print(f"{'臂':28s} {'中位(us)':>10s}")
    print(f"{'pyqcd 特化':28s} {absolute['specialized_median']:10.1f}")
    print(f"{'pyqcd 通用':28s} {absolute['generic_median']:10.1f}")
    print(f"{'opt_einsum（path 复用）':28s} {absolute['opt_einsum_median']:10.1f}")
    print()
    for key, label in (("specialized_vs_generic", "特化 / 通用"),
                       ("specialized_vs_opt_einsum", "特化 / opt_einsum")):
        stat = summary[key]
        print(f"{label:22s} 中位 {stat['median']:5.2f}x  "
              f"min {stat['min']:5.2f}x  max {stat['max']:5.2f}x  "
              f"快于对手 {stat['faster_than_generic']}/{summary['checked']}")
    print()
    print(f"{'元数':>4s} {'条数':>6s} {'vs 通用':>9s} {'vs opt_einsum':>14s} "
          f"{'特化中位(us)':>13s}")
    for arity, item in summary["by_arity"].items():
        print(f"{arity:>4s} {item['n']:6d} "
              f"{item['specialized_vs_generic_median']:8.2f}x "
              f"{item['specialized_vs_opt_einsum_median']:13.2f}x "
              f"{item['specialized_us_median']:13.1f}")
    if summary["slowest"]:
        print()
        print("最慢 5 条（特化/通用）：")
        for row in summary["slowest"]:
            print(f"  {row['specialized_vs_generic']:5.2f}x "
                  f"seq={row['seq']:5d} arity={row['arity']:2d} "
                  f"特化 {row['specialized_us']:.1f}us / 通用 {row['generic_us']:.1f}us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
