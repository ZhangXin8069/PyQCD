"""全量验证 ``_dedicated_step_tables`` 的固定步表内核。

对每条收缩式：随机操作数（小写指标取 ``dim``、大写取 ``external``）→ 固定步表执行器 →
与 numpy 对照，逐条记录偏差与耗时；失败立即打印行号与式子，不吞掉。

两条对照口径
------------
``A``：``np.einsum(eq, *ops, optimize=["einsum_path", *pairs])``，把本步表反解成
numpy 的显式路径，用 numpy 自己的 C 内核逐步执行。覆盖全部条目，验证步表与执行器。
``B``：``np.einsum(eq, *ops, optimize="greedy")``，numpy 自己的 greedy 路径（独立
规划）。numpy 会把末尾若干步合并成一个多操作数 ``c_einsum`` 调用，合并步的朴素
工作集实测可达 ``2.8e10``（单条 >90 s 不返回），因此只在该工作集不超过
``--greedy-max-elements`` 时运行，跳过数如实计入汇总。
``C``：numpy greedy 路径分步执行——沿用 ``B`` 的路径，但把多操作数合并步按
“中间张量最小”拆成两两 ``np.einsum``，因此**每条**都有独立 greedy 对照，
且代价与条数线性（实测约 1–2 ms/条）。``--no-greedy-decomposed`` 可关闭。

用法::

    python -m pyqcd.testing._dedicated_kernel_batch --limit 50
    python -m pyqcd.testing._dedicated_kernel_batch --limit 500 --json /tmp/k2_500.json
    python -m pyqcd.testing._dedicated_kernel_batch --start 500 --limit 500
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import Counter
from contextlib import contextmanager

import numpy as np

from ..einsum import _dedicated_step_tables as ST
from ..einsum import _wick_tasklist as TASKS


class _Timeout(Exception):
    pass


@contextmanager
def _deadline(seconds: float):
    """只对 Python 侧执行器生效：numpy 的 C 内核不会响应信号。"""
    import signal

    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(_signum, _frame):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _operand_shapes(terms, dim: int, external: int):
    return [tuple(external if ch.isupper() else dim for ch in term) for term in terms]


def _make_operands(index: int, terms, cfg) -> list[np.ndarray]:
    rng = np.random.default_rng([cfg.seed, index])
    return [
        rng.standard_normal(shape).astype(cfg.dtype, copy=False)
        for shape in _operand_shapes(terms, cfg.dim, cfg.external)
    ]


def _max_intermediate(table, dim: int, external: int) -> int:
    """该步表在给定维度配置下的最大中间量元素数（含最终输出）。"""
    largest = 1
    for _left, _right, keep in table:
        size = 1
        for label in keep:
            size *= external if label.isupper() else dim
        largest = max(largest, size)
    return largest


def _greedy_working_set(equation: str, terms, output: str, cfg) -> tuple[float, list | None]:
    """numpy 自己 greedy 路径上的最大朴素工作集（元素数）与路径本身。"""
    dummies = [np.empty(shape, dtype=np.int8)
               for shape in _operand_shapes(terms, cfg.dim, cfg.external)]
    try:
        path, contraction_list = np.einsum_path(
            equation, *dummies, optimize="greedy", einsum_call=True
        )
    except Exception:
        return float("inf"), None
    sizes = {ch: cfg.external if ch.isupper() else cfg.dim for term in terms for ch in term}
    largest = 1
    last = len(contraction_list) - 1
    for number, (_inds, einsum_str, _remaining) in enumerate(contraction_list):
        labels = set(einsum_str.partition("->")[0].replace(",", ""))
        if number == last:
            labels |= set(output)
        largest = max(largest, math.prod(sizes.get(label, 1) for label in labels))
    steps = [tuple(int(i) for i in entry[0]) for entry in contraction_list]
    return largest, steps


def _keep_of(labels, ids, output: str, cfg) -> str:
    group = "".join(labels[i] for i in ids)
    rest = "".join(labels[i] for i in range(len(labels)) if i not in ids)
    return "".join(dict.fromkeys(ch for ch in group if ch in output or ch in rest))


def _merge_group(labels, arrays, target: str, rest: str, cfg) -> np.ndarray:
    """把一组操作数按“中间张量最小”贪心合并到 ``target`` 指标。"""
    sizes = {ch: cfg.external if ch.isupper() else cfg.dim
             for label in labels for ch in label}
    work = [[label, array] for label, array in zip(labels, arrays)]
    while len(work) > 1:
        best_key = None
        best_pair = None
        best_keep = ""
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                others = "".join(w[0] for k, w in enumerate(work) if k not in (i, j)) + rest
                keep = "".join(dict.fromkeys(
                    ch for ch in work[i][0] + work[j][0] if ch in target or ch in others))
                size = math.prod(sizes.get(ch, 1) for ch in keep) if keep else 1
                key = (size, -len(set(work[i][0]) & set(work[j][0])))
                if best_key is None or key < best_key:
                    best_key, best_pair, best_keep = key, (i, j), keep
        i, j = best_pair
        merged = np.einsum(f"{work[i][0]},{work[j][0]}->{best_keep}",
                           work[i][1], work[j][1], optimize=False)
        for slot in sorted(best_pair, reverse=True):
            del work[slot]
        work.append([best_keep, merged])
    label, array = work[0]
    if label != target:
        array = np.transpose(array, [label.index(ch) for ch in target])
    return array


def _greedy_decomposed(equation: str, operands, path, cfg) -> np.ndarray:
    """按 numpy greedy 路径逐步求值；多操作数步骤两两拆分，避免朴素全枚举。"""
    terms, _, output = equation.partition("->")
    pool = list(operands)
    labels = terms.split(",")
    for step in path:
        ids = list(step)
        target = _keep_of(labels, ids, output, cfg)
        rest = "".join(labels[i] for i in range(len(labels)) if i not in ids)
        merged = _merge_group([labels[i] for i in ids], [pool[i] for i in ids],
                              target, rest, cfg)
        for slot in sorted(ids, reverse=True):
            del pool[slot]
            del labels[slot]
        pool.append(merged)
        labels.append(target)
    result, label = pool[0], labels[0]
    if label != output:
        result = np.transpose(result, [label.index(ch) for ch in output])
    return result


def _deviation(got: np.ndarray, expected: np.ndarray) -> float | None:
    if got.shape != expected.shape:
        return None
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    return float(np.max(np.abs(got - expected))) / max(scale, 1e-30)


def _check_one(index: int, task, cfg) -> dict:
    equation = task.equation
    terms, _, output = equation.partition("->")
    terms = terms.split(",")
    table = ST.steps(index)
    record = {
        "index": index,
        "line": task.line,
        "tid": task.tid,
        "case": task.case,
        "subtable": task.subtable,
        "equation": equation,
        "arity": len(terms),
        "steps": len(table),
        "kind": "ok",
        "detail": "",
        "deviation": None,
        "greedy_deviation": None,
        "decomposed_deviation": None,
        "greedy_reference": False,
        "greedy_working_set": None,
        "max_intermediate": _max_intermediate(table, cfg.dim, cfg.external),
        "kernel_s": 0.0,
        "reference_s": 0.0,
    }
    if record["max_intermediate"] > cfg.max_elements:
        record["kind"] = "memory_guard"
        record["detail"] = f"max intermediate {record['max_intermediate']} > {cfg.max_elements}"
        return record

    operands = _make_operands(index, terms, cfg)
    try:
        with _deadline(cfg.task_timeout):
            started = time.perf_counter()
            got = ST.execute(equation, table, operands)
            record["kernel_s"] = time.perf_counter() - started
    except _Timeout:
        record["kind"] = "timeout"
        return record
    except MemoryError:
        record["kind"] = "oom"
        return record
    except Exception as exc:
        record["kind"] = "error"
        record["detail"] = f"{type(exc).__name__}: {exc}"
        return record

    pairs = [(left, right) for left, right, _keep in table]
    try:
        started = time.perf_counter()
        expected = np.einsum(equation, *operands, optimize=["einsum_path", *pairs])
        record["reference_s"] = time.perf_counter() - started
    except Exception as exc:
        record["kind"] = "reference_error"
        record["detail"] = f"{type(exc).__name__}: {exc}"
        return record

    deviation = _deviation(got, expected)
    record["deviation"] = deviation
    if deviation is None:
        record["kind"] = "shape_mismatch"
        record["detail"] = f"kernel {got.shape} vs reference {expected.shape}"
        return record
    if deviation > cfg.tolerance:
        record["kind"] = "deviation"
        return record

    working_set, greedy_path = _greedy_working_set(equation, terms, output, cfg)
    record["greedy_working_set"] = working_set
    if cfg.greedy_max_elements and working_set <= cfg.greedy_max_elements:
        try:
            started = time.perf_counter()
            greedy = np.einsum(equation, *operands, optimize="greedy")
            record["reference_s"] += time.perf_counter() - started
        except Exception as exc:
            record["kind"] = "greedy_error"
            record["detail"] = f"{type(exc).__name__}: {exc}"
            return record
        greedy_deviation = _deviation(got, greedy)
        record["greedy_deviation"] = greedy_deviation
        record["greedy_reference"] = True
        if greedy_deviation is None or greedy_deviation > cfg.tolerance:
            record["kind"] = "greedy_deviation"
            return record
    if cfg.greedy_decomposed:
        if greedy_path is None:
            record["kind"] = "greedy_path_error"
            record["detail"] = "numpy greedy 路径不可用"
            return record
        try:
            started = time.perf_counter()
            decomposed = _greedy_decomposed(equation, operands, greedy_path, cfg)
            record["reference_s"] += time.perf_counter() - started
        except Exception as exc:
            record["kind"] = "greedy_decomposed_error"
            record["detail"] = f"{type(exc).__name__}: {exc}"
            return record
        decomposed_deviation = _deviation(got, decomposed)
        record["decomposed_deviation"] = decomposed_deviation
        if decomposed_deviation is None or decomposed_deviation > cfg.tolerance:
            record["kind"] = "greedy_decomposed_deviation"
            return record
    return record


def _peak_rss_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="批量验证固定步表内核")
    parser.add_argument("--limit", type=int, default=None, help="抽检条数（默认全部 8638）")
    parser.add_argument("--start", type=int, default=0, help="起始条目下标")
    parser.add_argument("--dim", type=int, default=3, help="小写指标维度")
    parser.add_argument("--external", type=int, default=2, help="大写（外置）指标维度")
    parser.add_argument("--dtype", default="float64", help="操作数 dtype")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（每条再混入条目下标）")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="范数相对偏差阈值")
    parser.add_argument("--task-timeout", type=float, default=30.0, help="单条执行器超时（秒，0 关闭）")
    parser.add_argument("--max-elements", type=int, default=1 << 28, help="单条中间量元素数上限")
    parser.add_argument("--greedy-max-elements", type=float, default=1e7,
                        help="运行 numpy 独立 greedy 对照的工作集上限（0 关闭）")
    parser.add_argument("--greedy-decomposed", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="用 numpy greedy 路径分步执行做逐条对照（默认开启）")
    parser.add_argument("--progress-every", type=int, default=500, help="进度打印间隔（0 关闭）")
    parser.add_argument("--json", default=None, help="把逐条记录写入该 JSON 文件")
    parser.add_argument("--quiet", action="store_true", help="不打印进度与汇总，只打印失败")
    cfg = parser.parse_args(argv)

    items = TASKS.tasks()
    total = len(items)
    start = max(0, min(cfg.start, total))
    limit = total - start if cfg.limit is None else max(0, min(cfg.limit, total - start))

    aligned = ST.verify_alignment()
    if not aligned:
        print("步表对齐检查失败：行数/指纹与任务清单不一致，先重生成步表")
        return 2
    peak_before = _peak_rss_mib()

    if not cfg.quiet:
        print(f"=== 固定步表批量验证：索引 {start}..{start + limit - 1}（{limit} 条 / 共 {total} 条）")
        print(f"    配置 dim={cfg.dim} external={cfg.external} dtype={cfg.dtype} "
              f"seed={cfg.seed} tol={cfg.tolerance:g} "
              f"greedy上限={cfg.greedy_max_elements:g}")
        print(f"    对齐检查 {ST.count()} 条 / {total} 条，指纹 0x{ST.FINGERPRINT:08x}，通过")

    counters: Counter = Counter()
    failures: list[dict] = []
    slowest: list[tuple[float, int, float, float | None]] = []
    records: list[dict] = []
    kernel_time = 0.0
    reference_time = 0.0
    worst_deviation = 0.0
    worst_greedy_deviation = 0.0
    worst_decomposed_deviation = 0.0
    greedy_skipped = 0
    greedy_skip_worst = 0.0

    wall_started = time.perf_counter()
    for offset in range(limit):
        index = start + offset
        record = _check_one(index, items[index], cfg)
        counters[record["kind"]] += 1
        kernel_time += record["kernel_s"]
        reference_time += record["reference_s"]
        if record["deviation"] is not None:
            worst_deviation = max(worst_deviation, record["deviation"])
        if record["greedy_deviation"] is not None:
            worst_greedy_deviation = max(worst_greedy_deviation, record["greedy_deviation"])
        if record["decomposed_deviation"] is not None:
            worst_decomposed_deviation = max(worst_decomposed_deviation,
                                             record["decomposed_deviation"])
            counters["decomposed_checked"] += 1
        if record["greedy_reference"]:
            counters["greedy_checked"] += 1
        elif record["greedy_working_set"] is not None:
            greedy_skipped += 1
            greedy_skip_worst = max(greedy_skip_worst, record["greedy_working_set"])
        heapq.heappush(slowest, (record["kernel_s"], index,
                                 record["reference_s"], record["deviation"]))
        if len(slowest) > 10:
            heapq.heappop(slowest)
        if cfg.json:
            records.append(record)
        if record["kind"] != "ok":
            failures.append(record)
            print(f"FAIL {record['kind']} {record['tid']} line={record['line']} "
                  f"[{record['subtable']}] {record['equation']} "
                  f"dev={record['deviation']} {record['detail']}")
        if cfg.progress_every and not cfg.quiet and (offset + 1) % cfg.progress_every == 0:
            elapsed = time.perf_counter() - wall_started
            print(f"    ... {offset + 1}/{limit} 条，{elapsed:.1f}s", file=sys.stderr)
    wall = time.perf_counter() - wall_started

    passed = counters["ok"]
    peak_after = _peak_rss_mib()
    ms_per_task = wall / passed * 1e3 if passed else float("nan")
    seconds_per_thousand = ms_per_task * 1000 / 1e3 if passed else float("nan")
    remaining = total - (start + limit)

    if not cfg.quiet:
        print()
        print(f"条目范围    : {start}..{start + limit - 1}（{limit} 条 / 共 {total} 条）")
        print(f"通过 / 失败 : {passed} / {len(failures)}")
        print(f"对照 A      : numpy 同路径 optimize=[einsum_path,...]  {limit - counters['memory_guard']} 条覆盖")
        checked = counters["greedy_checked"]
        skipped_note = (f"实测最大 {greedy_skip_worst:.3g}" if greedy_skipped else "无跳过")
        print(f"对照 B      : numpy 独立 optimize='greedy'  {checked} 条运行，"
              f"{greedy_skipped} 条跳过（合并步工作集 > {cfg.greedy_max_elements:g}，"
              f"{skipped_note}）")
        decomposed_checked = counters["decomposed_checked"]
        print(f"对照 C      : numpy greedy 路径分步（合并步两两拆分）  "
              f"{decomposed_checked} 条覆盖"
              + (f"（占 {decomposed_checked / max(limit, 1):.0%}）" if cfg.greedy_decomposed else "（已关闭）"))
        print(f"最大偏差    : A {worst_deviation:.3e}"
              + (f" / B {worst_greedy_deviation:.3e}" if checked else "")
              + (f" / C {worst_decomposed_deviation:.3e}" if decomposed_checked else ""))
        print(f"耗时        : 总 {wall:.3f} s，"
              + (f"{ms_per_task:.3f} ms/条，{seconds_per_thousand:.2f} s/千条"
                 if passed else "n/a（无通过条目）"))
        print(f"内核 / 对照 : 均值 {kernel_time / max(limit, 1) * 1e3:.3f} ms / "
              f"{reference_time / max(limit, 1) * 1e3:.3f} ms")
        if peak_after is not None:
            print(f"峰值内存    : {peak_after:.1f} MiB（ru_maxrss），入口 {peak_before:.1f} MiB")
        for kind in ("memory_guard", "timeout", "oom", "error",
                     "reference_error", "shape_mismatch", "deviation",
                     "greedy_error", "greedy_deviation",
                     "greedy_path_error", "greedy_decomposed_error",
                     "greedy_decomposed_deviation"):
            if counters[kind]:
                print(f"计数 {kind:16s}: {counters[kind]}")
        print("最慢 10 条  :")
        for seconds, index, reference_seconds, deviation in sorted(slowest, reverse=True):
            task = items[index]
            print(f"  line={task.line:5d} {task.tid} [{task.subtable}] "
                  f"arity={len(task.equation.partition('->')[0].split(',')):2d} "
                  f"steps={len(ST.steps(index)):2d} "
                  f"kernel={seconds * 1e3:7.3f} ms ref={reference_seconds * 1e3:8.3f} ms "
                  + (f"dev={deviation:.2e}" if deviation is not None else "dev=n/a"))
        if remaining:
            print(f"外推        : 剩余 {remaining} 条 × {ms_per_task:.3f} ms ≈ "
                  f"{remaining * ms_per_task / 1e3:.1f} s")

    if cfg.json:
        with open(cfg.json, "w", encoding="utf-8") as handle:
            json.dump({"config": vars(cfg), "counters": counters,
                       "failures": failures, "records": records}, handle,
                      ensure_ascii=False, indent=1)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
