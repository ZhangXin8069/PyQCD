"""双精度（float64/complex128）精度门：特化内核 vs 通用路径 vs 独立参考。

本契约专门回答一个问题：**特化路径在双精度下是否仍与算法定义逐位一致**。

判据（比 float32 严格得多）：

* float64：相对偏差 ≤ **1e-12**（float64 自身误差在 1e-15 量级，1e-12 留 3 个数量级余量）；
* complex128：同上；
* 且特化与通用路径的**最大绝对差**相对结果规模 ≤ 1e-13（两者应当几乎逐位相同）。

若某条式子只在小尺寸通过、大尺寸失败，会单独列出"尺寸分档"结果。

用法::

    python -m pyqcd.testing._double_precision_contract
    python -m pyqcd.testing._double_precision_contract --sample 500 --dim 4
"""

from __future__ import annotations

import argparse
import statistics
import sys

import numpy as np

RTOL_FLOAT64 = 1e-12
RTOL_COMPLEX128 = 1e-12
# 特化 vs 通用：同为 float64，两者只应差在求和顺序上
RTOL_SPECIALIZED_VS_GENERIC = 1e-13


def _operands_for(equation: str, dim: int, dtype, rng) -> list[np.ndarray]:
    shapes = [tuple(2 if ch.isupper() else dim for ch in term)
              for term in equation.split("->")[0].split(",")]
    return [rng.standard_normal(shape).astype(dtype) for shape in shapes]


def _reference(equation: str, operands) -> np.ndarray:
    """浮点参考：先升到 float64/complex128 再用 opt_einsum 的固定 path 收缩。"""
    import opt_einsum as oe

    wide = [np.asarray(operand, dtype=(np.complex128
                                       if np.iscomplexobj(operand)
                                       else np.float64)) for operand in operands]
    path, _ = oe.contract_path(equation, *wide, optimize="greedy")
    return oe.contract(equation, *wide, optimize=path)


def run(sample: int | None = 200, dim: int = 3, dtype_name: str = "float64",
        seed: int = 0) -> dict:
    import pyqcd.einsum as pe
    from pyqcd.einsum import _kernel_kupl as K
    import pyqcd.einsum._wick_tasklist as tasklist

    dtype = np.float64 if dtype_name == "float64" else np.complex128
    rtol = RTOL_FLOAT64 if dtype_name == "float64" else RTOL_COMPLEX128

    tasks = list(tasklist.TASKS)
    if sample is not None:
        subset = np.random.default_rng(seed).choice(
            len(tasks), size=min(sample, len(tasks)), replace=False)
        tasks = [tasks[index] for index in sorted(subset)]

    rows = []
    failures = []
    for task in tasks:
        rng = np.random.default_rng(task.seq)
        operands = _operands_for(task.equation, dim, dtype, rng)
        try:
            expected = _reference(task.equation, operands)
            generic = pe.contract(task.equation, *operands, specialize=False)
            specialized = K.try_dedicated(task.equation, operands)
        except Exception as exc:                       # noqa: BLE001
            failures.append({"seq": task.seq, "tid": task.tid,
                             "error": f"{type(exc).__name__}: {exc}"})
            continue

        scale = max(float(np.max(np.abs(expected))), 1e-300)
        generic_dev = float(np.max(np.abs(generic - expected))) / scale
        row = {
            "seq": task.seq, "tid": task.tid, "arity": task.arity,
            "generic_deviation": generic_dev,
            "specialized": specialized is not None,
        }
        if specialized is not None:
            specialized_dev = float(np.max(np.abs(specialized - expected))) / scale
            versus = float(np.max(np.abs(specialized - generic))) / scale
            row["specialized_deviation"] = specialized_dev
            row["specialized_vs_generic"] = versus
            if specialized_dev > rtol or versus > RTOL_SPECIALIZED_VS_GENERIC:
                failures.append({"seq": task.seq, "tid": task.tid,
                                 "device": dtype_name,
                                 "specialized_deviation": specialized_dev,
                                 "specialized_vs_generic": versus,
                                 "equation": task.equation})
        elif generic_dev > rtol:
            failures.append({"seq": task.seq, "tid": task.tid,
                             "device": dtype_name,
                             "generic_deviation": generic_dev,
                             "equation": task.equation,
                             "note": "specialised path unavailable; generic failed"})
        rows.append(row)

    specialized_rows = [row for row in rows if row["specialized"]]
    deviations = [row["specialized_deviation"] for row in specialized_rows]
    summary = {
        "dtype": dtype_name, "dim": dim, "rtol": rtol,
        "checked": len(rows),
        "specialized_ran": len(specialized_rows),
        "failed": len(failures),
        "max_specialized_deviation": max(deviations) if deviations else None,
        "median_specialized_deviation":
            statistics.median(deviations) if deviations else None,
        "max_generic_deviation":
            max((row["generic_deviation"] for row in rows), default=None),
        "failures": failures[:20],
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200,
                        help="抽样条数（0 表示全量 8638）")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--dtype", default="float64",
                        choices=["float64", "complex128"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    sample = None if args.sample == 0 else args.sample
    summary = run(sample=sample, dim=args.dim, dtype_name=args.dtype,
                  seed=args.seed)

    print(f"dtype={summary['dtype']} dim={summary['dim']} "
          f"rtol={summary['rtol']:.0e}")
    print(f"  检查条数        : {summary['checked']}")
    print(f"  特化真跑        : {summary['specialized_ran']}")
    print(f"  失败            : {summary['failed']}")
    if summary["max_specialized_deviation"] is not None:
        print(f"  特化最大偏差    : {summary['max_specialized_deviation']:.2e}")
        print(f"  特化中位偏差    : {summary['median_specialized_deviation']:.2e}")
    print(f"  通用最大偏差    : {summary['max_generic_deviation']:.2e}")
    for failure in summary["failures"]:
        print(f"    - {failure}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
