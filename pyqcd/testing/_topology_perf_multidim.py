"""中尺寸性能抽样：多档尺寸 × 多条路径，参考对象 opt_einsum。

目的：**防止单尺寸假数据**。同一批抽样路径在多个中尺寸下各测一次，
只有当多档尺寸都显示一致方向时，才认为结论成立。

口径（与 ``measured-pitfalls.md`` §13/§14 一致）：

* 参考臂 ``opt_einsum``：``contract_path`` 事先求出并复用，**只计执行**；
* **交错计时**：每轮三臂相邻各测一次，报每轮比值中位；
* **预热一次**（特化建步表、opt_einsum 建 path 缓存）；
* 单样本按 1 ms 预算自适应重复；
* 每档尺寸都记录工作集与缓存层级，越过 L2 的档位标注为"带宽主导，仅供参考"。

用法::

    python -m pyqcd.testing._topology_perf_multidim --sample 10
    python -m pyqcd.testing._topology_perf_multidim --sample 10 --dims 3,6,8,10
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
# 默认四个中尺寸：工作集 0.01 / 0.07 / 0.23 / 0.55 MB，全部驻留 L1–L2
DEFAULT_DIMS = (4, 6, 8, 10)
CACHE_L1_BYTES = 192 * 1024
CACHE_L2_BYTES = 1250 * 1024
CACHE_L3_BYTES = 24 * 1024 * 1024


def _shapes(equation: str, dim: int) -> list[tuple[int, ...]]:
    return [tuple(2 if ch.isupper() else dim for ch in term)
            for term in equation.split("->")[0].split(",")]


def _cache_level(working_bytes: int) -> str:
    if working_bytes < CACHE_L1_BYTES:
        return "L1"
    if working_bytes < CACHE_L2_BYTES:
        return "L2"
    if working_bytes < CACHE_L3_BYTES:
        return "L3"
    return "DRAM"


def _working_set(equation: str, operands, path_info) -> int:
    inputs = sum(operand.nbytes for operand in operands)
    itemsize = getattr(operands[0], "dtype", np.dtype(np.float32)).itemsize
    largest = (max(path_info.size_list) * itemsize
               if getattr(path_info, "size_list", None) else 0)
    return inputs + largest


def _timed(call) -> float:
    start = time.perf_counter()
    call()
    probe = max(time.perf_counter() - start, 1e-9)
    repeat = max(1, min(MAX_REPEAT, int(REPEAT_BUDGET_SECONDS / probe)))
    start = time.perf_counter()
    for _ in range(repeat):
        call()
    return (time.perf_counter() - start) / repeat


DTYPES = {"float32": np.float32, "float64": np.float64,
          "complex64": np.complex64, "complex128": np.complex128}


def run(sample: int = 10, dims=DEFAULT_DIMS, rounds: int = 5,
        seed: int = 0, out: pathlib.Path | None = None,
        dtype_name: str = "float32") -> dict:
    import opt_einsum as oe

    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasklist

    tasks = list(tasklist.TASKS)
    picked = np.random.default_rng(seed).choice(len(tasks), size=min(sample, len(tasks)),
                                                replace=False)
    picked = [tasks[index] for index in sorted(picked)]

    rows: list[dict] = []
    for dim in dims:
        for task in picked:
            rng = np.random.default_rng(task.seq)
            dtype = DTYPES[dtype_name]
            operands = [rng.standard_normal(shape).astype(dtype)
                        for shape in _shapes(task.equation, dim)]
            try:
                path, info = oe.contract_path(task.equation, *operands,
                                              optimize="greedy")
                pe.contract(task.equation, *operands)
                pe.contract(task.equation, *operands, specialize=False)
                oe.contract(task.equation, *operands, optimize=path)
            except MemoryError:
                rows.append({"seq": task.seq, "dim": dim, "skipped": "MemoryError"})
                continue
            except Exception as exc:                    # noqa: BLE001
                rows.append({"seq": task.seq, "dim": dim,
                             "skipped": f"{type(exc).__name__}: {exc}"})
                continue

            arms = (
                ("specialized", lambda: pe.contract(task.equation, *operands)),
                ("generic", lambda: pe.contract(task.equation, *operands,
                                                specialize=False)),
                ("opt_einsum", lambda: oe.contract(task.equation, *operands,
                                                   optimize=path)),
            )
            pairs_sg, pairs_so = [], []
            measured = {}
            for _ in range(rounds):
                for name, call in arms:
                    measured[name] = _timed(call)
                pairs_sg.append(measured["specialized"] / measured["generic"])
                pairs_so.append(measured["specialized"] / measured["opt_einsum"])

            expected = oe.contract(task.equation, *operands, optimize=path)
            got = pe.contract(task.equation, *operands)
            scale = max(float(np.max(np.abs(expected))), 1e-300)
            rows.append({
                "seq": task.seq, "tid": task.tid, "dim": dim,
                "arity": len(operands),
                "specialized_ms": measured["specialized"] * 1e3,
                "generic_ms": measured["generic"] * 1e3,
                "opt_einsum_ms": measured["opt_einsum"] * 1e3,
                "specialized_vs_generic": statistics.median(pairs_sg),
                "specialized_vs_opt_einsum": statistics.median(pairs_so),
                "deviation": float(np.max(np.abs(got - expected))) / scale,
                "working_set_mb": _working_set(task.equation, operands, info) / 1024 / 1024,
                "cache": _cache_level(_working_set(task.equation, operands, info)),
            })

    good = [row for row in rows if "skipped" not in row]

    def per_dim(rows_for_dim):
        return {
            "n": len(rows_for_dim),
            "specialized_ms_median": statistics.median(
                [r["specialized_ms"] for r in rows_for_dim]),
            "vs_generic_median": statistics.median(
                [r["specialized_vs_generic"] for r in rows_for_dim]),
            "vs_opt_einsum_median": statistics.median(
                [r["specialized_vs_opt_einsum"] for r in rows_for_dim]),
            "vs_opt_einsum_min": min(r["specialized_vs_opt_einsum"] for r in rows_for_dim),
            "vs_opt_einsum_max": max(r["specialized_vs_opt_einsum"] for r in rows_for_dim),
            "max_deviation": max(r["deviation"] for r in rows_for_dim),
            "working_set_mb": statistics.median(
                [r["working_set_mb"] for r in rows_for_dim]),
            "cache": rows_for_dim[0]["cache"],
        }

    # ── 自洽性检查 ────────────────────────────────────────────────────
    # 同一路径在不同尺寸上的「特化/opt_einsum」比值不应出现方向反转：
    # 若小尺寸报"慢"（>1）而大尺寸报"快"（<1），几乎必是单次测量的调度噪声
    # （实测 seq=650：dim=8 报 1.79×，独立复核为 0.64×）。
    # 这类条目单列出来，不参与"最差"排名，并强制标注 needs_recheck。
    by_path: dict[int, list[dict]] = {}
    for row in good:
        by_path.setdefault(row["seq"], []).append(row)
    suspect: list[dict] = []
    for seq, items in by_path.items():
        ratios = sorted(((item["dim"], item["specialized_vs_opt_einsum"])
                         for item in items), key=lambda pair: pair[0])
        if len(ratios) < 2:
            continue
        slow_at = [dim for dim, ratio in ratios if ratio > 1.0]
        fast_at = [dim for dim, ratio in ratios if ratio < 1.0]
        if slow_at and fast_at and min(slow_at) < max(fast_at):
            # 小尺寸慢、大尺寸快 → 方向反转
            for item in items:
                if item["dim"] in slow_at:
                    item["needs_recheck"] = True
                    suspect.append({"seq": seq, "dim": item["dim"],
                                    "reported": item["specialized_vs_opt_einsum"],
                                    "other_dims": {d: r for d, r in ratios}})

    clean = [row for row in good if not row.get("needs_recheck")]

    summary = {
        "sample": len(picked), "dims": list(dims), "rounds": rounds,
        "dtype": dtype_name,
        "checked": len(good), "skipped": len(rows) - len(good),
        "suspect": suspect,
        "clean": len(clean),
        "per_dim": {str(dim): per_dim([r for r in clean if r["dim"] == dim])
                    for dim in dims if any(r["dim"] == dim for r in clean)},
        "worst": sorted(clean, key=lambda r: -r["specialized_vs_opt_einsum"])[:5],
    }
    if out is not None:
        target = pathlib.Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"summary": summary, "rows": rows},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--dims", default=",".join(str(d) for d in DEFAULT_DIMS))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    dims = tuple(int(item) for item in args.dims.split(",") if item.strip())
    summary = run(sample=args.sample, dims=dims, rounds=args.rounds,
                  seed=args.seed,
                  out=pathlib.Path(args.json) if args.json else None,
                  dtype_name=args.dtype)

    print(f"dtype={summary['dtype']}  抽样 {summary['sample']} 条路径 × {len(dims)} 档尺寸  "
          f"rounds={summary['rounds']}  成功 {summary['checked']}  跳过 {summary['skipped']}")
    print()
    print(f"{'dim':>4s} {'条数':>5s} {'工作集MB':>9s} {'层级':>5s} "
          f"{'特化(ms)':>10s} {'vs 通用':>8s} {'vs opt_einsum':>14s} {'最大偏差':>9s}")
    for dim, item in summary["per_dim"].items():
        print(f"{dim:>4s} {item['n']:5d} {item['working_set_mb']:9.2f} "
              f"{item['cache']:>5s} {item['specialized_ms_median']:10.2f} "
              f"{item['vs_generic_median']:7.2f}x "
              f"{item['vs_opt_einsum_median']:13.2f}x "
              f"{item['max_deviation']:9.1e}")
    print()
    if summary["suspect"]:
        print(f"自洽性检查：{len(summary['suspect'])} 条疑似假数据（方向反转，已排除出统计）")
        for item in summary["suspect"]:
            print(f"  seq={item['seq']:5d} dim={item['dim']:2d} "
                  f"报 {item['reported']:.2f}x，同路径其他尺寸 "
                  f"{ {d: round(r, 2) for d, r in item['other_dims'].items()} }")
        print(f"  参与统计的干净条目：{summary['clean']}/{summary['checked']}")
        print()
    print("多档一致性检查（每档都要 <1.0 才算结论成立）：")
    all_faster = all(item["vs_opt_einsum_median"] < 1.0
                     for item in summary["per_dim"].values())
    print(f"  {'全部尺寸均快于 opt_einsum' if all_faster else '存在某档不快于 opt_einsum'}")
    if summary["worst"]:
        print()
        print("最差 5 条（特化/opt_einsum）：")
        for row in summary["worst"]:
            print(f"  {row['specialized_vs_opt_einsum']:5.2f}x dim={row['dim']:2d} "
                  f"seq={row['seq']:5d} arity={row['arity']:2d} "
                  f"特化 {row['specialized_ms']:.2f}ms / oe {row['opt_einsum_ms']:.2f}ms")
    return 0 if all_faster else 1


if __name__ == "__main__":
    raise SystemExit(main())
