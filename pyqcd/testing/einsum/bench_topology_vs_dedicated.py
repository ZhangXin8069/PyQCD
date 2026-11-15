"""按生产尺寸口径对比 ``topology`` 与 ``dedicated``（四类 dtype）。

口径
----
``distill``（``Nev=100``、``Mom=1``，见 :data:`pyqcd.einsum._bench_scene.PRODUCTION_DISTILL`）
——蒸馏收缩的性能结论只能用这一条，**不得**用 smoke／等长超立方体扫描替代。
轴长由语料的 ``roles`` 反推（:func:`pyqcd.einsum.bench_specialized._label_sizes`），
不使用 ``_label_sizes`` 的等长假设。

为什么需要这个脚本
------------------
``_kernel_kupl`` 里关于复数性能的旧注释（"6–7×、seq=1700 complex64 0.15×"）测的是
**v2 通用执行器**：当时 ``topology_apply.cpp`` 对复数直接 ``kSpecializedMiss``，
``LAST_INFO['path']`` 却仍写 ``kupl_topology``，所以那条数据把 v2 的耗时记在了
v3 头上。复数接通 v3（cgemm/zgemm）后必须重测才能用于选路。

计时协议
--------
首调单独丢弃（含路径搜索、plan 编码与 pack-bound 守卫的一次
``_simulate_plan``），随后取 ``--rounds`` 轮的中位数。被
:func:`pyqcd.einsum._kernel_kupl._topology_pack_bound_reason` 或
:func:`_cpu_plan_fallback_reason` 拒绝的格子如实标 ``declined`` 并写明理由，
**不剔除、不换小尺寸代替**。

用法::

    python -m pyqcd.testing.einsum.bench_topology_vs_dedicated
    PYQCD_EINSUM_TOPOLOGY_REAL=1 \
        python -m pyqcd.testing.einsum.bench_topology_vs_dedicated --dtypes complex128
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Any

import numpy as np

from pyqcd.einsum import _kernel_kupl as kupl
from pyqcd.einsum import _wick_tasklist as tasklist
from pyqcd.einsum._bench_scene import dims as scene_dims
from pyqcd.einsum.bench_specialized import _label_sizes
from pyqcd.testing._topology_kernel_gen import parse

#: 代表 seq：按语料的 (arity, rank 多重集) 签名覆盖主要规模档。
DEFAULT_SEQS = (1, 3, 9, 129, 849, 855, 1599, 2583, 2591, 3431, 3461,
                4301, 4331, 5051, 6911, 6935, 7055, 7775, 7791, 7919)
#: 容差按 dtype 分档：单精度/半精度量级的累加误差不同，统一阈值会错杀正确的单精度。
TOLERANCE = {np.dtype(np.complex128): 1e-9, np.dtype(np.complex64): 1e-4,
             np.dtype(np.float64): 1e-9, np.dtype(np.float32): 1e-4}
DEFAULT_DTYPES = ("complex128", "complex64", "float64", "float32")


def build_operands(seq: int, dtype: Any, table: dict[str, int]):
    """按语料 roles 构造真实生产尺寸的操作数。"""
    task = tasklist.lookup(seq)
    if task is None:
        raise SystemExit(f"unknown seq {seq}")
    terms, _out = parse(task.equation)
    sizes = _label_sizes(terms, task.roles, table)
    shapes = [tuple(sizes[label] for label in term) for term in terms]
    rng = np.random.default_rng(seq)
    if np.dtype(dtype).kind == "c":
        operands = [(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                    .astype(dtype) for shape in shapes]
    else:
        operands = [rng.standard_normal(shape).astype(dtype) for shape in shapes]
    return task, operands


def measure(call, rounds: int):
    """首调丢弃，返回 (中位耗时, 结果, 拒绝理由)。"""
    kupl.LAST_INFO.clear()
    kupl.LAST_FAILURE.update({"reason": "", "status": None})
    first = call()
    reason = str(kupl.LAST_FAILURE.get("reason", ""))
    if first is None:
        return None, None, reason
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        out = call()
        samples.append(time.perf_counter() - start)
        if out is None:
            return None, None, str(kupl.LAST_FAILURE.get("reason", ""))
    return statistics.median(samples), first, reason


def run(scene: str, seqs, dtypes, rounds: int) -> dict:
    table = scene_dims(scene)
    rows = []
    for name in dtypes:
        dtype = np.dtype(name)
        for seq in seqs:
            task, operands = build_operands(seq, dtype, table)
            topo_time, topo_out, topo_reason = measure(
                lambda: kupl.try_topology_apply(task.equation, operands), rounds)
            ded_time, ded_out, ded_reason = measure(
                lambda: kupl.try_dedicated(task.equation, operands), rounds)
            rel = None
            within = None
            if topo_out is not None and ded_out is not None:
                expected = np.einsum(task.equation, *operands, optimize="greedy")
                scale = float(np.max(np.abs(expected))) or 1.0
                rel = float(np.max(np.abs(topo_out - expected))) / scale
                within = rel <= TOLERANCE[dtype]
            rows.append({
                "seq": seq, "dtype": dtype.name, "arity": len(operands),
                "input_gib": round(sum(o.nbytes for o in operands) / 2 ** 30, 3),
                "topology_ms": None if topo_time is None else round(topo_time * 1000, 3),
                "dedicated_ms": None if ded_time is None else round(ded_time * 1000, 3),
                "ratio": (None if (topo_time is None or ded_time is None or ded_time <= 0)
                          else round(topo_time / ded_time, 3)),
                "rel": None if rel is None else float(f"{rel:.3e}"),
                "within_tolerance": within,
                "topology_declined": None if topo_out is not None else topo_reason[:90],
                "dedicated_declined": None if ded_out is not None else ded_reason[:90],
            })
    return {
        "scene": scene,
        "table": table,
        "rounds": rounds,
        "timing": "topology_ms/dedicated_ms 为热态中位数，首调已丢弃",
        "real_dtype_override": bool(kupl._topology_real_override()),
        "rows": rows,
    }


def summarize(result: dict) -> None:
    rows = [row for row in result["rows"] if row["ratio"] is not None]
    print(f"\n{'dtype':>11} {'n':>3} {'中位':>8} {'最优':>8} {'最差':>8} {'快于 dedicated':>16}")
    for name in sorted({row["dtype"] for row in result["rows"]}):
        ratios = sorted(row["ratio"] for row in rows if row["dtype"] == name)
        if not ratios:
            print(f"{name:>11}   0  （全部被拒）")
            continue
        faster = sum(1 for value in ratios if value < 1)
        print(f"{name:>11} {len(ratios):>3} {statistics.median(ratios):>7.2f}x "
              f"{min(ratios):>7.2f}x {max(ratios):>7.2f}x {faster:>8}/{len(ratios):<7}")
    bad = [row for row in result["rows"] if row["within_tolerance"] is False]
    print(f"\n数值超容差: {len(bad)} 格" if bad else "\n数值全部在容差内")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="distill", choices=("distill", "distill_curr"))
    parser.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES))
    parser.add_argument("--seqs", default=",".join(str(s) for s in DEFAULT_SEQS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", default=None, help="落盘路径（默认 artifacts/ 下带时间戳）")
    args = parser.parse_args(argv)

    dtypes = tuple(item.strip() for item in args.dtypes.split(",") if item.strip())
    seqs = tuple(int(item) for item in args.seqs.split(",") if item.strip())
    result = run(args.scene, seqs, dtypes, args.rounds)

    if args.json:
        target = pathlib.Path(args.json)
    else:
        stamp = time.strftime("%Y%m%d%H%M%S")
        target = (pathlib.Path(__file__).resolve().parent / "artifacts"
                  / f"topo_vs_dedicated_{args.scene}_{stamp}" / "result.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    summarize(result)
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
