"""8638 条收缩式的"专用 vs 通用"全量普查，并生成 ``_dedicated_allowlist``。

测什么
------
对 ``_wick_tasklist`` 的每一条方程、每个 ``dim ∈ {4, 6, 8}``：

* 专用：``contract(eq, *ops)`` —— 默认路径，命中 ``_specialize.match`` 后交给
  原生拓扑入口（本机落在 ``kupl_topology``）；
* 通用：``contract(eq, *ops, specialize=False)`` —— 同一次调用关闭专用执行。

每条路径**先预热一次**（建计划表、编码步表、填形状缓存），再**交错 A/B 各 5 轮**
取中位。``ratio = 专用中位 / 通用中位``，``ratio < 1`` 表示专用更快。

为什么需要它
------------
6 元方程与 16 元方程的单次代价差 4–5 个数量级（dim=8 下 1e4 对 1e8 FLOP），
"专用路径更快"不是全局结论，只能逐条判定。本脚本因此把结论落成一张以
``seq`` 为键的白名单（:mod:`pyqcd.einsum._dedicated_allowlist`），运行期只查表。

预算与 TBD
----------
``opt_cost``（``contract_path`` 给出的优化后 FLOP 数）超过 ``--max-cost`` 的
单元格不测，记 ``over-budget``。dim=8 全量 5 轮按实测外推约 17 小时，而
1e8 FLOP 的单次调用已约 0.1 s；``--max-cost 3e8`` 覆盖 dim=4 全部、dim=6 的
97.1%、dim=8 的 76.4%。未测单元格一律不进白名单，报告逐条给出 TBD 原因。

看门狗
------
原生入口存在"方程相关的尺寸断崖"（``docs/topology_kernels_20261111.md`` 第 6 节）。
测量在子进程里做，父进程逐单元格等回执；超时就杀掉子进程、把该单元格记为
``timeout`` 并从下一格重启。单个病态方程不会拖垮整批，也不会被误写成"更慢"。

用法::

    python -m pyqcd.testing._dedicated_census                 # 全量（可断点续跑）
    python -m pyqcd.testing._dedicated_census --limit 200 --dims 4
    python -m pyqcd.testing._dedicated_census --report-only --raw <路径>

``--raw`` 默认放在系统临时目录；仓库里只落生成物 ``_dedicated_allowlist.py``。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np

from ..einsum import contract, contract_path

MODULE = "pyqcd.testing._dedicated_census"

DIMS = (4, 6, 8)
ROUNDS = 5
DTYPE = "complex128"
MAX_COST = 3.0e8
CELL_TIMEOUT = 30.0
REL_TOL = 1e-9
AA_EVERY = 64                 # 每 64 个单元格做一次 A/A 对照（同路径复测）

DIM, SEQ, EQUATION = "dim", "seq", "equation"

_OUT = Path(__file__).resolve().parents[1] / "einsum" / "_dedicated_allowlist.py"
_RAW = Path(tempfile.gettempdir()) / "pyqcd_dedicated_census.jsonl"

# 操作数按 (dim, rank, dtype) 缓存：形状只由 rank 决定，分配不该计进计时区间。
_OPERANDS: dict[tuple[int, int, str], np.ndarray] = {}


def _operand(dim: int, rank: int, dtype: np.dtype) -> np.ndarray:
    key = (dim, rank, dtype.str)
    array = _OPERANDS.get(key)
    if array is None:
        rng = np.random.default_rng(1000 * dim + rank + dtype.itemsize)
        shape = (dim,) * rank
        real = rng.standard_normal(shape)
        data = real + 1j * rng.standard_normal(shape) if dtype.kind == "c" else real
        array = np.ascontiguousarray(data.astype(dtype))
        _OPERANDS[key] = array
    return array


def _shapes(equation: str, dim: int) -> list[tuple[int, ...]]:
    terms = equation.split("->")[0].split(",")
    return [tuple([dim] * len(term)) for term in terms]


def _relative_error(actual, expected) -> float:
    left, right = np.asarray(actual), np.asarray(expected)
    if left.shape != right.shape:
        return float("inf")
    if right.size == 0:
        return 0.0
    scale = float(np.max(np.abs(right)))
    difference = float(np.max(np.abs(left - right)))
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / scale


def _kernel_label(text: str) -> str:
    """从 ``_specialize.announce`` 的提示里取出实际执行的标签（空串=没打印）。"""
    for line in text.splitlines():
        if "[kernel:" in line:
            return line.split("[kernel:", 1)[1].split("]", 1)[0].strip()
        if "[plan-cache]" in line:
            return "plan-cache"
    return ""


def _native_ran() -> bool:
    """原生入口是否真的执行过：成功下发会留下 ``LAST_INFO``，回落时为空。

    ``_specialize.announce`` 每个形态名每进程只打印一次，所以换 dim 复用同一批
    方程时没有提示可读；这里直接看执行侧留下的证据。
    """
    from ..einsum import _kernel_cuda, _kernel_kupl

    return bool(_kernel_kupl.LAST_INFO or _kernel_cuda.LAST_INFO)


# ── 单单元格测量（只在 worker 里跑） ────────────────────────────────────────

def measure_cell(job: dict) -> dict:
    """测一条 (eq, dim)：预热一次 + 交错 A/B 各 ``rounds`` 轮，取中位。"""
    equation = job[EQUATION]
    dim = int(job[DIM])
    seq = int(job[SEQ])
    rounds = int(job.get("rounds", ROUNDS))
    dtype = np.dtype(job.get("dtype", DTYPE))

    terms = equation.split("->")[0].split(",")
    operands = [_operand(dim, len(term), dtype) for term in terms]
    buffer = io.StringIO()
    spec_times: list[float] = []
    gen_times: list[float] = []
    aa_ratio = None
    reldiff = float("nan")
    status, error = "ok", ""
    ran_natively = False

    with contextlib.redirect_stdout(buffer):
        try:
            specialized = contract(equation, *operands)          # 预热：建表 + 缓存
            ran_natively = _native_ran()
            generic = contract(equation, *operands, specialize=False)
            reldiff = _relative_error(specialized, generic)
            for index in range(rounds):
                # 轮内顺序按奇偶交替：同一格内"后测的更快"的漂移对两条路径对称。
                if index % 2 == 0:
                    start = time.perf_counter()
                    contract(equation, *operands)
                    middle = time.perf_counter()
                    contract(equation, *operands, specialize=False)
                    end = time.perf_counter()
                    spec_times.append(middle - start)
                    gen_times.append(end - middle)
                else:
                    start = time.perf_counter()
                    contract(equation, *operands, specialize=False)
                    middle = time.perf_counter()
                    contract(equation, *operands)
                    end = time.perf_counter()
                    gen_times.append(middle - start)
                    spec_times.append(end - middle)
            if job.get("aa_every") and seq % int(job["aa_every"]) == 0:
                repeat = []
                for _ in range(rounds):
                    start = time.perf_counter()
                    contract(equation, *operands, specialize=False)
                    repeat.append(time.perf_counter() - start)
                baseline = statistics.median(gen_times)
                if baseline > 0:
                    aa_ratio = statistics.median(repeat) / baseline
        except Exception as exc:                                  # noqa: BLE001 - 逐格记录
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

    record = dict(job)
    record.pop("rounds", None)
    record.pop("aa_every", None)
    record.update({
        "status": status,
        "error": error,
        "kernel": _kernel_label(buffer.getvalue()) or (
            "native" if ran_natively else ""),
        "reldiff": reldiff,
        "aa_ratio": aa_ratio,
        "rounds": rounds,
    })
    if status != "ok" or not spec_times or not gen_times:
        return record
    spec_median = statistics.median(spec_times)
    gen_median = statistics.median(gen_times)
    record["spec_s"] = spec_median
    record["gen_s"] = gen_median
    record["ratio"] = (spec_median / gen_median) if gen_median > 0 else float("inf")
    return record


def worker_loop(args) -> int:
    """stdin 逐行收 job、stdout 逐行回 record（stdout 只走协议，提示已重定向）。"""
    protocol = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        job.setdefault("dtype", args.dtype)
        job["rounds"] = args.rounds
        job["aa_every"] = args.aa_every
        record = measure_cell(job)
        protocol.write(json.dumps(record, ensure_ascii=False) + "\n")
        protocol.flush()
    return 0


# ── 计划与执行（父进程） ────────────────────────────────────────────────────

def plan_cells(dims, max_cost: float, limit: int | None) -> list[dict]:
    """全部 (dim, seq) 单元格 + ``opt_cost``；按代价升序（先测便宜的）。"""
    from ..einsum import _wick_tasklist as tasks

    cells: list[dict] = []
    for dim in dims:
        started = time.perf_counter()
        costs = []
        for task in tasks.TASKS:
            _, info = contract_path(
                task.equation, *_shapes(task.equation, dim), shapes=True)
            costs.append(float(info.opt_cost))
        order = sorted(range(len(costs)), key=lambda index: costs[index])
        affordable = [index for index in order if costs[index] <= max_cost]
        if limit is not None:
            affordable = affordable[:limit]
        for index in affordable:
            task = tasks.TASKS[index]
            cells.append({DIM: dim, SEQ: task.seq, EQUATION: task.equation,
                          "cost": costs[index]})
        inside = sum(1 for cost in costs if cost <= max_cost)
        print(f"[plan] dim={dim} 规划 {time.perf_counter() - started:.1f}s，"
              f"预算内 {inside}/{len(costs)}，本次 {len(affordable)}",
              file=sys.stderr, flush=True)
    return cells


def _spawn(args) -> subprocess.Popen:
    command = [sys.executable, "-B", "-m", MODULE, "--worker",
               "--dtype", args.dtype, "--rounds", str(args.rounds),
               "--aa-every", str(args.aa_every)]
    return subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1)


def _pump(stream, sink: queue.Queue) -> None:
    """把子进程 stdout 的每一行送进队列；EOF 送 ``None``。"""
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put(None)


def _stop(proc: subprocess.Popen) -> None:
    try:
        if proc.stdin:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _failure(cell: dict, status: str, error: str) -> dict:
    record = dict(cell)
    record.update({"status": status, "error": error, "kernel": "",
                   "reldiff": float("nan"), "aa_ratio": None, "rounds": 0})
    return record


def _start_worker(args) -> tuple[subprocess.Popen, queue.Queue]:
    proc = _spawn(args)
    sink: queue.Queue = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, sink), daemon=True).start()
    return proc, sink


def run_cells(cells: list[dict], args, ledger_path: Path) -> None:
    """逐单元格下发；父进程等回执，超时/崩溃即杀子进程并记 TBD。

    换 ``dim`` 时主动重启子进程：``_specialize.announce`` 每个形态名每进程只打印
    一次，同一批方程到第二个 dim 就没有提示可读了（``kernel`` 会记空）。
    """
    if not cells:
        return
    proc, sink = _start_worker(args)
    current_dim = None
    failures = 0
    started = time.perf_counter()
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for count, cell in enumerate(cells, 1):
            if current_dim is not None and cell[DIM] != current_dim:
                _stop(proc)
                proc, sink = _start_worker(args)
                failures = 0
            current_dim = cell[DIM]
            record = None
            try:
                proc.stdin.write(json.dumps(cell, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                record = _failure(cell, "crash", "worker exited before the cell")
            if record is None:
                try:
                    line = sink.get(timeout=args.cell_timeout)
                except queue.Empty:
                    _stop(proc)
                    record = _failure(
                        cell, "timeout",
                        f"no reply within {args.cell_timeout:.0f}s")
                    proc, sink = _start_worker(args)
                else:
                    if line is None:
                        _stop(proc)
                        record = _failure(cell, "crash", "worker died")
                        proc, sink = _start_worker(args)
                    else:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            record = _failure(cell, "crash", f"bad reply: {exc}")
            if record.get("status") in ("timeout", "crash"):
                failures += 1
                if failures >= 5:
                    ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ledger.flush()
                    raise SystemExit(
                        f"worker 连续失败 {failures} 次（最后一格 seq={cell[SEQ]} "
                        f"dim={cell[DIM]}: {record.get('error')}）")
            else:
                failures = 0
            ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
            ledger.flush()
            if count % 25 == 0 or count == len(cells):
                elapsed = time.perf_counter() - started
                rate = count / elapsed if elapsed > 0 else 0.0
                tail = (f"  剩余 {(len(cells) - count) / rate / 60:.1f} min"
                        if rate else "")
                print(f"[run] {count}/{len(cells)}  {rate:.2f} cell/s{tail}",
                      file=sys.stderr, flush=True)
    _stop(proc)


def load_ledger(path: Path) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[(int(record[DIM]), int(record[SEQ]))] = record
    return records


# ── 报告与生成物 ────────────────────────────────────────────────────────────

def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _is_hit(record: dict) -> bool:
    kernel = str(record.get("kernel", ""))
    return bool(kernel) and kernel != "plan-cache"


def _allowable(record: dict) -> bool:
    return (record.get("status") == "ok"
            and _is_hit(record)
            and float(record.get("ratio", 2.0)) < 1.0
            and float(record.get("reldiff", 1.0)) <= REL_TOL)


def summarise(records: list[dict], n_tasks: int, dims) -> str:
    lines: list[str] = []
    for dim in dims:
        group = [r for r in records if int(r[DIM]) == dim]
        ok = [r for r in group if r.get("status") == "ok"]
        over = [r for r in group if r.get("status") == "over-budget"]
        bad = [r for r in group if r.get("status") in ("timeout", "crash", "error")]
        hitless = [r for r in ok if not _is_hit(r)]
        ratios = [float(r["ratio"]) for r in ok]
        fast = [r for r in ok if _allowable(r)]
        slower = sum(1 for r in ratios if r >= 1.0)
        lines.append(
            f"== dim={dim} ==\n"
            f"  实测 ok {len(ok)} / 任务 {n_tasks}"
            f"（over-budget {len(over)}，timeout/crash/error {len(bad)}，"
            f"专用未执行 {len(hitless)}，未进入计划 {n_tasks - len(group)}）\n"
            f"  特化更快 {len(fast)}；其余 {len(ok) - len(fast)}"
            f"（其中 ratio>=1.0 的 {slower} 条）"
        )
        if ratios:
            lines.append(
                f"  ratio 中位 {statistics.median(ratios):.3f}"
                f"  p10 {_quantile(ratios, 0.10):.3f}"
                f"  p90 {_quantile(ratios, 0.90):.3f}"
                f"  min {min(ratios):.3f}  max {max(ratios):.3f}\n"
                f"  边际：ratio<0.90 {sum(1 for r in ratios if r < 0.90)}，"
                f"<0.75 {sum(1 for r in ratios if r < 0.75)}，"
                f"<0.50 {sum(1 for r in ratios if r < 0.50)}")
        aa = [float(r["aa_ratio"]) for r in group if r.get("aa_ratio")]
        if aa:
            lines.append(
                f"  A/A 噪声对照 {len(aa)} 格：ratio 中位 {statistics.median(aa):.3f}，"
                f"|ratio-1| p95 {_quantile([abs(v - 1) for v in aa], 0.95):.3f}")
        kernels: dict[str, int] = {}
        for record in ok:
            label = str(record.get("kernel", "")) or "(none)"
            kernels[label] = kernels.get(label, 0) + 1
        if kernels:
            lines.append("  实际执行标签：" + "，".join(
                f"{name} {count}" for name, count in
                sorted(kernels.items(), key=lambda item: -item[1])))
        worst = sorted(ok, key=lambda r: float(r["ratio"]), reverse=True)[:10]
        if worst:
            lines.append("  最差 10（专用相对最慢）：")
            for record in worst:
                lines.append(f"    seq={int(record[SEQ]):5d} "
                             f"ratio={float(record['ratio']):7.3f} "
                             f"spec={float(record['spec_s']) * 1e3:8.2f}ms "
                             f"gen={float(record['gen_s']) * 1e3:8.2f}ms  "
                             f"{str(record[EQUATION])[:60]}")
        lines.append("")
    return "\n".join(lines)


def _wrap(tokens: list[str], width: int = 16) -> str:
    return "\n".join(" ".join(tokens[i:i + width])
                     for i in range(0, len(tokens), width))


def _header(dims, allowed, coverage, meta) -> str:
    n_tasks = meta["n_tasks"]
    text = f'''"""``seq`` → "专用路径实测更快"的白名单（生成物，勿手改）。

由 :mod:`pyqcd.testing._dedicated_census` 生成于 {meta["stamp"]}。

口径
----
* 数据：``refer/sush/wick_12cases.md`` 的 {n_tasks} 条收缩式，主键是
  ``_wick_tasklist`` 的连续 ``seq``（不是 T 号，T 号有 720 个重复）；
* 对比：``contract(eq, *ops)``（默认，命中拓扑入口）与
  ``contract(eq, *ops, specialize=False)``；每条路径预热 1 次，再交错 A/B
  各 {meta["rounds"]} 轮取中位；
* dtype ``{meta["dtype"]}``；``dim ∈ {tuple(dims)}``；每个指标轴长 = dim；
* 判据：``ratio = 专用中位 / 通用中位 < 1.0``，且该单元格的专用路径**确实执行**
  （命中提示为 ``[kernel: ...]``，不是 ``[plan-cache]``），且两条路径数值一致
  （相对最大偏差 <= {REL_TOL:g}）；
* 预算：``opt_cost > {meta["max_cost"]:.1e}`` 的单元格未测（TBD），一律不进白名单；
  其余未测原因（timeout/crash/error）同样不收录。

覆盖（实测 ok / 全部 {n_tasks} 条）
----
'''
    for dim in dims:
        measured = coverage.get(dim, 0)
        text += (f"* dim={dim}: 实测 {measured} 条，白名单 {len(allowed[dim])} 条，"
                 f"未测（TBD）{n_tasks - measured} 条\n")
    text += '''
用法
----
``ALLOW[dim]`` 的成员是 ``seq`` 的十进制字符串（``_BLOCKS[dim].split()`` 直接
得到，import 期不解析 8000+ 个整数）。按 ``seq: int`` 查询用 :func:`allow`；
需要整数序列用 :func:`seqs`。

本表只对**上述 dtype 与后端**成立：换 dtype（float32/float64 走
``kupl_dedicated`` 步表路径）或换后端必须重测，不能照搬。
"""

from __future__ import annotations

'''
    text += "_DIMS = " + repr(tuple(dims)) + "\n\n"
    text += '''_BLOCKS: dict[int, str] = {
'''
    for dim in dims:
        text += f'    {dim}: """\\\n{_wrap([str(seq) for seq in allowed[dim]])}\n""",\n'
    text += '''}

ALLOW: dict[int, frozenset[str]]


def __getattr__(name: str):
    """惰性解析：只有真正访问 ``ALLOW`` 时才 split 数据块。"""
    if name == "ALLOW":
        value = {dim: frozenset(block.split()) for dim, block in _BLOCKS.items()}
        globals()["ALLOW"] = value
        return value
    raise AttributeError(name)


def allow(dim: int, seq: int) -> bool:
    """该 dim 下 ``seq`` 是否实测"专用更快"。"""
    return str(int(seq)) in ALLOW[int(dim)]


def seqs(dim: int) -> tuple[int, ...]:
    """该 dim 下全部白名单 ``seq``（升序，整数视图）。"""
    return tuple(sorted(int(token) for token in ALLOW[int(dim)]))
'''
    return text


def write_allowlist(records: list[dict], path: Path, dims, meta: dict):
    allowed = {dim: sorted(int(r[SEQ]) for r in records
                           if int(r[DIM]) == dim and _allowable(r))
               for dim in dims}
    coverage = meta["coverage"]
    path.write_text(_header(dims, allowed, coverage, meta), encoding="utf-8")
    return allowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dims", default=",".join(map(str, DIMS)))
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--max-cost", type=float, default=MAX_COST)
    parser.add_argument("--cell-timeout", type=float, default=CELL_TIMEOUT)
    parser.add_argument("--aa-every", type=int, default=AA_EVERY)
    parser.add_argument("--limit", type=int, default=None,
                        help="每个 dim 只测最便宜的 N 条（冒烟用）")
    parser.add_argument("--raw", default=str(_RAW))
    parser.add_argument("--out", default=str(_OUT))
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--retry-failures", action="store_true",
                        help="重测 timeout/crash/error 的单元格")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        return worker_loop(args)

    dims = tuple(int(part) for part in args.dims.split(",") if part.strip())
    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    from ..einsum import _wick_tasklist as tasks

    ledger = load_ledger(raw_path)
    if not args.report_only:
        planned = plan_cells(dims, args.max_cost, args.limit)
        todo: list[dict] = []
        with raw_path.open("a", encoding="utf-8") as handle:
            for cell in planned:
                key = (cell[DIM], cell[SEQ])
                previous = ledger.get(key)
                if previous is not None and not (
                        args.retry_failures
                        and previous.get("status") in ("timeout", "crash", "error")) \
                        and not (previous.get("status") == "over-budget"
                                 and cell["cost"] <= args.max_cost):
                    continue
                if cell["cost"] > args.max_cost:
                    record = dict(cell, status="over-budget", error="", kernel="",
                                  reldiff=float("nan"), aa_ratio=None, rounds=0)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ledger[key] = record
                    continue
                todo.append(cell)
            handle.flush()
        run_cells(todo, args, raw_path)
        ledger = load_ledger(raw_path)

    records = list(ledger.values())
    print(summarise(records, tasks.N_TASKS, dims))

    coverage = {dim: sum(1 for r in records
                         if int(r[DIM]) == dim and r.get("status") == "ok")
                for dim in dims}
    if not args.no_write:
        allowed = write_allowlist(records, Path(args.out), dims, {
            "n_tasks": tasks.N_TASKS, "dtype": args.dtype,
            "max_cost": args.max_cost, "coverage": coverage,
            "rounds": args.rounds, "stamp": args.stamp})
        print(f"[out] {args.out}：" +
              "，".join(f"dim={dim} {len(allowed[dim])} 条" for dim in dims))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
