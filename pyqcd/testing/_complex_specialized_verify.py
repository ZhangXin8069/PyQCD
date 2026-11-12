"""复数特化验收：**始终以 complex128 真值为参考**。

为什么不能用"通用路径的 complex64 结果"当参考：实测该 dtype 在 13–16 算子链上
自身误差就达 1.5e-04（与 opt_einsum 的 complex64 同量级），拿它当基准会把
"特化更准"判成"特化超差"——本文件早期版本正是这么误判的。

参考实现：把操作数升到 complex128 后，用 opt_einsum 的固定 path 收缩。
判据：complex128 ≤ 1e-12；complex64 ≤ 1e-4（该 dtype 的固有极限）。

用法::

    python -m pyqcd.testing._complex_specialized_verify --dtype complex128
    python -m pyqcd.testing._complex_specialized_verify --dtype complex64 --limit 500
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

TRUTH_DTYPE = np.complex128
TOLERANCE = {"complex128": 1e-12, "complex64": 1e-4}


def _operands(equation: str, dim: int, dtype, seed: int) -> list[np.ndarray]:
    return [np.random.default_rng(seed).standard_normal(
                tuple(2 if ch.isupper() else dim for ch in term)).astype(dtype)
            for term in equation.split("->")[0].split(",")]


def _truth(equation: str, operands) -> np.ndarray:
    """complex128 真值：升精度 + opt_einsum 固定 path。"""
    import opt_einsum as oe

    wide = [np.asarray(operand, dtype=TRUTH_DTYPE) for operand in operands]
    path, _ = oe.contract_path(equation, *wide, optimize="greedy")
    return oe.contract(equation, *wide, optimize=path)


def verify(dtype_name: str = "complex128", limit: int | None = None,
           dim: int = 3) -> dict:
    import pyqcd.einsum as pe
    import pyqcd.einsum._wick_tasklist as tasklist

    dtype = np.complex128 if dtype_name == "complex128" else np.complex64
    tolerance = TOLERANCE[dtype_name]

    tasks = list(tasklist.TASKS)
    if limit is not None:
        tasks = tasks[:limit]

    ran = passed = 0
    worst_specialized = 0.0
    worst_generic = 0.0
    specialized_better = generic_better = 0
    failures: list[dict] = []
    started = time.perf_counter()

    for task in tasks:
        operands = _operands(task.equation, dim, dtype, task.seq)
        got = pe.contract(task.equation, *operands)
        if got is None:
            continue
        ran += 1
        generic = pe.contract(task.equation, *operands, specialize=False)
        truth = _truth(task.equation, operands)
        scale = max(float(np.max(np.abs(truth))), 1e-300)
        dev_specialized = float(np.max(np.abs(got - truth))) / scale
        dev_generic = float(np.max(np.abs(generic - truth))) / scale
        worst_specialized = max(worst_specialized, dev_specialized)
        worst_generic = max(worst_generic, dev_generic)
        if dev_specialized < dev_generic:
            specialized_better += 1
        elif dev_generic < dev_specialized:
            generic_better += 1
        if dev_specialized <= tolerance:
            passed += 1
        elif len(failures) < 10:
            failures.append({
                "seq": task.seq, "tid": task.tid, "arity": task.arity,
                "specialized": dev_specialized, "generic": dev_generic,
            })

    return {
        "dtype": dtype_name, "dim": dim, "tolerance": tolerance,
        "checked": len(tasks), "specialised_ran": ran, "passed": passed,
        "failed": len(tasks) - passed,
        "worst_specialized": worst_specialized,
        "worst_generic": worst_generic,
        "specialized_better": specialized_better,
        "generic_better": generic_better,
        "seconds": round(time.perf_counter() - started, 1),
        "failures": failures,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", default="complex128",
                        choices=["complex64", "complex128"])
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条（默认全量 8638）")
    parser.add_argument("--dim", type=int, default=3)
    args = parser.parse_args(argv)

    summary = verify(dtype_name=args.dtype, limit=args.limit, dim=args.dim)
    print(f"dtype={summary['dtype']} dim={summary['dim']} "
          f"判据={summary['tolerance']:.0e}（参考 = complex128 真值）")
    print(f"  检查条数        : {summary['checked']}")
    print(f"  特化真跑        : {summary['specialised_ran']}")
    print(f"  通过 / 失败     : {summary['passed']} / {summary['failed']}")
    print(f"  特化最大偏差    : {summary['worst_specialized']:.2e}")
    print(f"  通用最大偏差    : {summary['worst_generic']:.2e}")
    print(f"  特化更准 / 通用更准 : {summary['specialized_better']} / "
          f"{summary['generic_better']}")
    print(f"  耗时            : {summary['seconds']} s")
    for failure in summary["failures"]:
        print(f"    - {failure}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
