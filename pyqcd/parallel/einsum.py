"""MPI 任务级并行执行多个相互独立的 :mod:`pyqcd.einsum` 收缩。

本模块只做「分发与汇总」，不改变 ``pyqcd.einsum`` 的任何行为：每个子任务仍是
一次普通的 ``contract(...)`` 调用，参数与返回值语义完全一致。

适用场景（也是本功能的边界）：单队列里排着大批量、但每个任务都很小的 einsum。
这类任务无法把单次收缩拆到多进程（单个收缩的并行度受限），只能做**任务级**并行。

两种运行方式：

* 已在 MPI 下（``mpirun -np N python your_script.py``）：各 rank 都调用
  ``einsums(...)``，rank 0 规划后广播分箱，各 rank 算自己那箱，最后 gather 回
  rank 0。此时数据不动，零序列化开销。
* 不在 MPI 下：``einsums(...)`` 自己 ``mpirun -np K python -m
  pyqcd.parallel.einsum --worker <dir>`` 拉起 K 个进程，任务与结果经临时目录
  传递。此时每个操作数会被序列化一次、结果反序列化一次。

工作量分配：按 ``contract_path`` 的优化 FLOP 数（并以输入/输出元素数兜底，覆盖
零求和的逐元素方程）做工时估计，再用 LPT（longest-processing-time-first）装箱，
保证最慢进程的负载不超过最优解的 4/3-1/(3K)。进程数默认取 ``min(子任务数,
CPU 线程数, 可用内存允许值)``，不会凭空超出硬件极限。
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np

from ..einsum import contract, contract_path

__all__ = ["einsums", "plan_einsums", "REMINDER_TEXT",
           "COLLECTIVE_MEMORY_WARNING"]

REMINDER_TEXT = (
    "使用场景提醒：本功能针对单队列执行的大批量但小单量的 einsum 任务，"
    "解决此类任务不能充分利用计算资源的痛点；但是此功能也存在超出硬件资源极限的可能，"
    "注意可信性分析与功能实测。"
    "（具体代价：① 进程启动与任务/结果传输是固定开销，单任务越大越不划算；"
    "② 并发进程数受 CPU 线程数与可用内存约束，但仍可能超订；"
    "③ 结果需回传 rank 0 / 父进程，回传量约等于全部输出字节数。"
    "实测口径见 `python -m pyqcd.testing.parallel_einsum bench`。）"
)

# collective 模式的内存警告：实测（`resource.ru_maxrss`，float32 矩阵乘）表明
# 输入被每个 rank 各持一份 —— 64 个 512² 任务（整批输入 134 MB）时，4 rank 在开始
# 收缩前就各占 223 MB，串行只有一份；结果在 rank 0 再叠一层，`broadcast=True` 还会
# 让每个 rank 各持一份全部结果。这条单独成句而不是塞进上面的通用提醒，因为它是最
# 容易被忽略、且在 GPU 上会直接触发 OOM 的代价。
COLLECTIVE_MEMORY_WARNING = (
    "collective 模式内存提醒：每个 rank 都持有**整批输入**（输入占用 ≈ 单份 × rank 数），"
    "rank 0 另持有**全部结果**；传 broadcast=True 时每个 rank 都会各持一份全部结果。"
    "实测（float32，64 个 512² 任务，整批输入 134 MB）：串行峰值 234 MB，"
    "collective 4 rank 聚合约 1.15 GB。GPU/显存上这直接意味着 rank 数张卡各放一份整批输入，"
    "多卡时请让 rank 数与可见卡数对齐，否则显存会成倍放大。"
    "输入大到放不下时改用自建进程池（任务经磁盘分发，worker 只读自己那一箱），"
    "代价是每次调用多 0.15–0.3 s 固定开销。"
)

_REMINDER_SHOWN = False
_COLLECTIVE_WARNING_SHOWN = False
_RANK_ENV = (
    ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK"),
    ("PMI_SIZE", "PMI_RANK"),
    ("SLURM_NTASKS", "SLURM_PROCID"),
    ("MV2_COMM_WORLD_SIZE", "MV2_COMM_WORLD_RANK"),
)
_TASK_KEYS = frozenset(
    ("subscripts", "equation", "eq", "operands", "kwargs"))


# ═══════════════════════════════════════════════════════════════════
# 任务规范化
# ═══════════════════════════════════════════════════════════════════

def _normalize_one(spec, index):
    """把一个任务描述规范化为 ``{'subscripts', 'operands', 'kwargs'}``。"""
    if isinstance(spec, dict):
        unknown = set(spec) - _TASK_KEYS
        if unknown:
            raise ValueError(
                f"task {index!r} has unsupported key(s): {sorted(unknown)}; "
                f"allowed: {sorted(_TASK_KEYS)}")
        subscripts = spec.get("subscripts", spec.get("equation", spec.get("eq")))
        operands = spec.get("operands")
        kwargs = dict(spec.get("kwargs") or {})
    elif isinstance(spec, (tuple, list)) and len(spec) in (2, 3):
        subscripts, operands = spec[0], spec[1]
        kwargs = dict(spec[2]) if len(spec) == 3 else {}
    else:
        raise TypeError(
            f"task {index!r} must be a dict or a "
            f"(subscripts, operands[, kwargs]) tuple, "
            f"got {type(spec).__name__}")
    if not isinstance(subscripts, str):
        raise TypeError(f"task {index!r} subscripts must be str, "
                        f"got {type(subscripts).__name__}")
    if not isinstance(operands, (tuple, list)):
        raise TypeError(f"task {index!r} operands must be a list/tuple, "
                        f"got {type(operands).__name__}")
    if not operands:
        raise ValueError(f"task {index!r} has no operands")
    return {"subscripts": subscripts,
            "operands": list(operands),
            "kwargs": kwargs}


def _normalize_tasks(tasks):
    """返回 ``(kind, keys, specs)``；``kind`` 为 ``'list'`` 或 ``'dict'``。"""
    if isinstance(tasks, dict):
        if not tasks:
            raise ValueError("tasks mapping must not be empty")
        keys = list(tasks.keys())
        return "dict", keys, [_normalize_one(tasks[key], key) for key in keys]
    if isinstance(tasks, (list, tuple)):
        if not tasks:
            raise ValueError("tasks sequence must not be empty")
        return "list", list(range(len(tasks))), [
            _normalize_one(spec, index) for index, spec in enumerate(tasks)]
    raise TypeError(
        "tasks must be a list/tuple of task specs or a mapping of "
        f"name -> task spec, got {type(tasks).__name__}")


def _package(kind, keys, ordered):
    """按输入容器类型重装结果，保持一一对应。"""
    if kind == "dict":
        return {key: ordered[index] for index, key in enumerate(keys)}
    return ordered


# ═══════════════════════════════════════════════════════════════════
# 代价估计与负载分配
# ═══════════════════════════════════════════════════════════════════

def _shape_of(operand):
    shape = getattr(operand, "shape", None)
    if shape is not None:
        return tuple(int(size) for size in shape)
    return tuple(int(size) for size in np.asarray(operand).shape)


def _bytes_of(operand):
    for attribute in ("nbytes",):
        value = getattr(operand, attribute, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    try:
        return int(operand.element_size()) * int(operand.nelement())
    except Exception:
        pass
    return int(np.prod(_shape_of(operand), dtype=np.int64)) * 16


def _work_and_bytes(spec, cache):
    """单任务的 ``(工时估计, 输入字节数)``；工时按方程+形状缓存。"""
    shapes = tuple(_shape_of(op) for op in spec["operands"])
    total_bytes = sum(_bytes_of(op) for op in spec["operands"])
    key = (spec["subscripts"], shapes)
    cached = cache.get(key)
    if cached is None:
        io_elements = sum(int(np.prod(shape, dtype=np.int64))
                          for shape in shapes if shape)
        try:
            _, info = contract_path(spec["subscripts"], *shapes, shapes=True,
                                    optimize="greedy")
            flops = float(info.opt_cost)
        except Exception:
            # 非法方程在这里静默退回元素数代理，真正的错误留到执行期抛出，
            # 这样规划失败不会抢在结果异常之前改变错误类型。
            flops = float(io_elements)
        cached = max(flops, float(io_elements), 1.0)
        cache[key] = cached
    return cached, total_bytes


def _assign_lpt(work, workers):
    """LPT 装箱：按工时降序放入当前最空的箱子（堆实现，O(m log m)）。"""
    bins = [[] for _ in range(workers)]
    heap = [(0.0, index) for index in range(workers)]
    heapq.heapify(heap)
    for index in sorted(range(len(work)), key=lambda i: (-work[i], i)):
        load, slot = heapq.heappop(heap)
        bins[slot].append(index)
        heapq.heappush(heap, (load + work[index], slot))
    for entries in bins:
        entries.sort()
    return bins


def _choose_workers(work, task_bytes, max_procs, resources):
    """进程数 = min(子任务数, 上限)；默认上限由 CPU 线程数与可用内存决定。"""
    cpu_limit = max(1, int(resources.get("cpu_threads") or 1))
    if max_procs is None:
        limit = cpu_limit
        capped_by = "cpu_threads"
        available_mb = resources.get("mem_avail_mb")
        heaviest_mb = 2.0 * max(task_bytes) / 1e6 if task_bytes else 0.0
        if available_mb is not None and heaviest_mb > 0:
            mem_limit = int(float(available_mb) * 0.8 / heaviest_mb)
            if mem_limit < limit:
                limit = max(1, mem_limit)
                capped_by = "mem_avail_mb"
    else:
        if isinstance(max_procs, bool) or not isinstance(max_procs, int):
            raise TypeError("max_procs must be a positive integer or None")
        if max_procs < 1:
            raise ValueError("max_procs must be positive")
        limit = max_procs
        capped_by = "max_procs"
    workers = max(1, min(len(work), limit))
    return workers, {"limit": limit, "capped_by": capped_by,
                     "cpu_threads": cpu_limit}


def _light_resources():
    """只取批量调度需要的两项，**不探测 GPU**。

    ``detect_resources`` 会为了拿显存去 ``import torch``（本机实测 1.0 s），
    而任务级并行只需要「能用几个进程」和「内存还剩多少」；当单任务耗时才几十
    毫秒时，这 1 秒会把并行收益完全吃掉。
    """
    from ._resources import available_memory_mb, cpu_threads

    return {"cpu_threads": cpu_threads(), "mem_avail_mb": available_memory_mb()}


def _plan_specs(specs, max_procs=None, resources=None):
    if resources is None:
        resources = _light_resources()
    elif "cpu_threads" not in resources:
        resources = dict(resources, **_light_resources())
    cache = {}
    work, task_bytes = [], []
    for spec in specs:
        estimate, nbytes = _work_and_bytes(spec, cache)
        work.append(estimate)
        task_bytes.append(nbytes)
    workers, limits = _choose_workers(work, task_bytes, max_procs, resources)
    bins = _assign_lpt(work, workers)
    loads = [float(sum(work[index] for index in entries)) for entries in bins]
    total = float(sum(work))
    mean = total / workers
    return {
        "tasks": len(specs), "workers": workers, "bins": bins, "work": work,
        "loads": loads, "total_work": total, "mean_load": mean,
        "max_load": max(loads), "min_load": min(loads),
        "skew": (max(loads) / mean) if mean > 0 else 1.0,
        "limits": limits,
    }


def plan_einsums(tasks, max_procs=None, resources=None):
    """只做规划、不执行：返回进程数、分箱与负载统计（便于自查与复现）。"""
    _kind, _keys, specs = _normalize_tasks(tasks)
    plan = _plan_specs(specs, max_procs=max_procs, resources=resources)
    plan["bins"] = [list(entries) for entries in plan["bins"]]
    return plan


def _format_plan(plan, mode):
    limits = plan["limits"]
    return (f"[pyqcd.parallel.einsum] mode={mode} tasks={plan['tasks']} "
            f"workers={plan['workers']} "
            f"({limits['capped_by']}<={limits['limit']}) "
            f"skew={plan['skew']:.2f}x "
            f"bins={[len(entries) for entries in plan['bins']]}")


# ═══════════════════════════════════════════════════════════════════
# 单个任务执行与失败汇总
# ═══════════════════════════════════════════════════════════════════

def _call(spec, optimize, backend):
    kwargs = dict(spec["kwargs"])
    kwargs.setdefault("optimize", optimize)
    if backend is not None:
        kwargs.setdefault("backend", backend)
    return contract(spec["subscripts"], *spec["operands"], **kwargs)


def _failure(index, exc):
    return {"index": int(index), "type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(limit=6)}


def _raise_failures(failures, keys):
    details = "; ".join(
        f"task {keys[item['index']]!r} ({item['type']}): {item['message']}"
        for item in failures)
    raise RuntimeError(f"einsums: {len(failures)} task(s) failed: {details}")


# ═══════════════════════════════════════════════════════════════════
# MPI 上下文（轻量版：不 import pyqcd.pipeline）
# ═══════════════════════════════════════════════════════════════════

def _launcher_env_world():
    """从启动器环境变量读 ``(rank, size)``；无标记时返回 ``None``。

    先查环境再考虑 mpi4py 是有意为之：``from mpi4py import MPI`` 在本机要
    **1.6 s**（实测），而 worker 只需要 rank/size。每个 worker 都付这份开销，
    会把小任务的并行收益全部吃掉。
    """
    for size_key, rank_key in _RANK_ENV:
        size, rank = os.environ.get(size_key), os.environ.get(rank_key)
        if (size and rank and size.isdigit() and rank.isdigit()
                and int(size) > 0):
            return int(rank), int(size)
    return None


def _mpi_world():
    """返回 ``(comm, rank, size)``；无 MPI 时为 ``(None, 0, 1)``。"""
    if _launcher_env_world() is None and "mpi4py" not in sys.modules:
        # 既无启动器标记、mpi4py 也还没被导入：确定是单进程，省掉 1.6 s 导入。
        return None, 0, 1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception as exc:
        markers = [name for pair in _RANK_ENV for name in pair
                   if name in os.environ]
        if not markers:
            return None, 0, 1
        raise RuntimeError(
            f"MPI launcher detected via {', '.join(markers)}, but mpi4py "
            "initialization failed") from exc


def _worker_identity():
    """worker 的 ``(rank, size)``：启动器环境变量优先，其次 mpi4py。"""
    identity = _launcher_env_world()
    if identity is not None:
        return identity
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except Exception:
        return 0, 1


# ═══════════════════════════════════════════════════════════════════
# 三条执行路径
# ═══════════════════════════════════════════════════════════════════

def _run_serial(kind, keys, specs, optimize, backend):
    ordered, failures = [], []
    for index, spec in enumerate(specs):
        try:
            ordered.append(_call(spec, optimize, backend))
        except Exception as exc:
            failures.append(_failure(index, exc))
    if failures:
        _raise_failures(failures, keys)
    return _package(kind, keys, ordered)


def _slim_plan(plan, size):
    """只保留各 rank 需要的信息：分箱与统计，不含逐任务工时数组。

    进程数被实际 rank 数压缩时必须**重新装箱**——直接截断原来的箱子会把多出来
    的那几箱任务丢掉（曾导致结果缺项 KeyError）。
    """
    workers = min(plan["workers"], size)
    bins = (plan["bins"] if workers == plan["workers"]
            else _assign_lpt(plan["work"], workers))
    loads = [float(sum(plan["work"][index] for index in entries))
             for entries in bins]
    mean = plan["total_work"] / workers
    return {"planned_workers": plan["workers"], "workers": workers,
            "tasks": plan["tasks"],
            "bins": [list(entries) for entries in bins],
            "skew": (max(loads) / mean) if mean > 0 else 1.0,
            "limits": plan["limits"]}


def _run_collective(comm, rank, size, kind, keys, specs, slim, optimize,
                    backend, broadcast, verbose):
    """已在 MPI 下：各 rank 分担分箱，结果 gather 回 rank 0。

    规划只在 rank 0 做一次再广播：``contract_path`` 的代价在小任务上占比很高
    （31–84%），让每个 rank 各自重算会白白吃掉并行收益。
    """
    workers = slim["workers"]
    if rank == 0:
        if verbose and workers < slim["planned_workers"]:
            print(f"[pyqcd.parallel.einsum] only {size} MPI rank(s); "
                  f"workers reduced {slim['planned_workers']} -> {workers}")
        if verbose and workers < size:
            print(f"[pyqcd.parallel.einsum] {size - workers} idle rank(s) "
                  f"(fewer tasks than ranks)")
    bins = slim["bins"]
    mine = bins[rank] if rank < len(bins) else []
    local_results, local_failures = [], []
    compute_seconds = 0.0
    for index in mine:
        try:
            started = time.perf_counter()
            local_results.append((index, _call(specs[index], optimize, backend)))
            compute_seconds += time.perf_counter() - started
        except Exception as exc:
            local_failures.append(_failure(index, exc))
    gathered = comm.gather((local_results, local_failures, compute_seconds),
                           root=0)

    if rank != 0:
        return (comm.bcast(None, root=0) if broadcast else None), None

    results, failures, per_rank = {}, [], []
    for chunk_results, chunk_failures, chunk_seconds in gathered:
        results.update(chunk_results)
        failures.extend(chunk_failures)
        per_rank.append(chunk_seconds)
    if failures:
        failures.sort(key=lambda item: item["index"])
        _raise_failures(failures, keys)
    ordered = [results[index] for index in range(len(specs))]
    payload = _package(kind, keys, ordered)
    timings = {"per_rank_seconds": per_rank,
               "compute_seconds": max(per_rank) if per_rank else 0.0,
               "pool_overhead_seconds": 0.0}   # collective 没有进程池与落盘
    return (comm.bcast(payload, root=0) if broadcast else payload), timings


def _to_numpy(value):
    """把后端数组转成 numpy（自建进程池路径要落盘交换）。"""
    if isinstance(value, np.ndarray):
        return value
    for getter in ("get", "detach", "cpu"):
        method = getattr(value, getter, None)
        if callable(method):
            try:
                value = method()
            except Exception:
                continue
    return np.asarray(value)


def _write_tasks(run_dir, specs):
    task_dir = os.path.join(run_dir, "tasks")
    os.makedirs(task_dir, exist_ok=True)
    meta = []
    for index, spec in enumerate(specs):
        arrays = {f"operand_{num}": _to_numpy(operand)
                  for num, operand in enumerate(spec["operands"])}
        np.savez(os.path.join(task_dir, f"task_{index:05d}.npz"), **arrays)
        meta.append({"subscripts": spec["subscripts"],
                     "kwargs": spec["kwargs"],
                     "n_operands": len(arrays)})
    with open(os.path.join(run_dir, "tasks.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)


def _read_task(run_dir, index, meta_cache):
    meta = meta_cache
    if meta is None:
        with open(os.path.join(run_dir, "tasks.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
    entry = meta[index]
    with np.load(os.path.join(run_dir, "tasks", f"task_{index:05d}.npz")) as data:
        operands = [data[f"operand_{num}"] for num in range(entry["n_operands"])]
    return {"subscripts": entry["subscripts"], "operands": operands,
            "kwargs": entry["kwargs"]}


def _default_launcher():
    """定位 MPI 启动器；root 下 OpenMPI 需要显式放行。"""
    for name in ("mpirun", "mpiexec"):
        path = shutil.which(name)
        if path is None:
            continue
        argv = [path]
        if getattr(os, "geteuid", lambda: -1)() == 0:
            argv.append("--allow-run-as-root")
        return argv
    return None


def _package_root():
    """``pyqcd`` 包的父目录（worker 需要它进 ``PYTHONPATH`` 才能 ``-m``）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def _collect_worker_errors(results_dir):
    errors = []
    if not os.path.isdir(results_dir):
        return errors
    for name in sorted(os.listdir(results_dir)):
        if name.startswith("error_") and name.endswith(".json"):
            with open(os.path.join(results_dir, name), encoding="utf-8") as handle:
                errors.append(json.load(handle))
    return errors


def _collect_worker_timings(results_dir):
    """读回各 rank 自报的纯计算耗时；缺失时返回 ``None``。

    进程池的墙钟时间包含 mpirun 启动与落盘/回读，二者对"这批任务值不值得并行"
    是两个不同问题。这里把 worker 内部实测的计算时间单独取出，供调用方与基准
    分别统计"计入初始化"和"不计入初始化"两种口径。
    """
    if not os.path.isdir(results_dir):
        return None
    timings = []
    for name in sorted(os.listdir(results_dir)):
        if name.startswith("timing_") and name.endswith(".json"):
            with open(os.path.join(results_dir, name), encoding="utf-8") as handle:
                timings.append(json.load(handle))
    if not timings:
        return None
    per_worker = [item["compute_seconds"] for item in timings]
    return {
        "per_worker_seconds": per_worker,
        "tasks_per_worker": [item["tasks"] for item in timings],
        "compute_seconds": max(per_worker),
        "total_task_seconds": sum(per_worker),
        "workers": len(per_worker),
    }


def _missing_results(results_dir, total):
    if not os.path.isdir(results_dir):
        return list(range(total))
    return [index for index in range(total)
            if not os.path.exists(
                os.path.join(results_dir, f"result_{index:05d}.npy"))]


def _worker_env(workers, worker_threads=None):
    """给 worker 分配线程预算，避免 K 进程 × 全线程造成超订。

    不能靠"环境变量与 ``_threads.exported_env`` 相同与否"来判断用户是否显式设置：
    任何在导入 ``pyqcd.einsum`` **之后**才加载的库（本机实测 `torch`，经由
    `pyqcd.testing.einsum.bench_distillation`）会把 ``OMP_NUM_THREADS`` 等改写掉，
    旧启发式随即失效，4 进程各拿 4 线程 → 400 个小任务从 0.30 s 涨到 1.79 s。
    启动器知道自己要起几个进程，因此这里直接给出预算；需要别的值请传
    ``worker_threads``。
    """
    env = os.environ.copy()
    try:
        from ..einsum import _threads
        cores = max(1, _threads.physical_cores())
        names = (_threads.STANDARD_OMP_ENV, _threads.STANDARD_BLAS_ENV)
    except Exception:
        return env, None
    budget = max(1, int(worker_threads)) if worker_threads else \
        max(1, cores // max(1, workers))
    for name in names:
        env[name] = str(budget)
    return env, {"cores": cores, "per_worker": budget,
                 "threads": {name: env[name] for name in names}}


def _run_launched(kind, keys, specs, plan, optimize, backend, workdir,
                  keep_workdir, launcher, worker_threads=None):
    """不在 MPI 下：自建 mpirun 进程池，任务与结果经临时目录交换。"""
    argv_prefix = (list(launcher) if launcher is not None
                   else _default_launcher())
    if not argv_prefix:
        warnings.warn(
            "mpirun/mpiexec not found; falling back to in-process serial "
            "execution (no task-level parallelism)",
            RuntimeWarning, stacklevel=3)
        return _run_serial(kind, keys, specs, optimize, backend), {
            "mode": "serial_fallback", "workers": 1}

    run_dir = tempfile.mkdtemp(prefix="pyqcd-einsums-", dir=workdir)
    results_dir = os.path.join(run_dir, "results")
    try:
        _write_tasks(run_dir, specs)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(run_dir, "assign.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"bins": plan["bins"], "workers": plan["workers"]},
                      handle)
        command = argv_prefix + [
            "-np", str(plan["workers"]), sys.executable,
            "-m", "pyqcd.parallel.einsum", "--worker", run_dir,
            "--optimize", str(optimize), "--backend", str(backend)]
        env, thread_budget = _worker_env(plan["workers"], worker_threads)
        env["PYTHONPATH"] = os.pathsep.join(
            [_package_root(), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env=env, cwd=workdir or None)
        errors = _collect_worker_errors(results_dir)
        missing = _missing_results(results_dir, len(specs))
        if completed.returncode != 0 or errors or missing:
            tail = (completed.stderr or completed.stdout or "").strip()[-1500:]
            if errors or missing:
                detail = "; ".join(
                    f"task {keys[item['index']]!r} ({item['type']}): "
                    f"{item['message']}" for item in errors)
                if missing:
                    detail += ("; " if detail else "")
                    detail += f"missing results for {missing[:8]}"
                raise RuntimeError(
                    f"einsums: {len(errors) or len(missing)} task(s) failed: "
                    f"{detail}\n--- launcher output ---\n{tail}")
            raise RuntimeError(
                f"einsums: MPI launch failed (exit={completed.returncode}); "
                f"no task reported an error\n"
                f"--- launcher output ---\n{tail}")
        ordered = [np.load(os.path.join(results_dir, f"result_{index:05d}.npy"))
                   for index in range(len(specs))]
        payload = _package(kind, keys, ordered)
        timings = _collect_worker_timings(results_dir)
        return payload, {"mode": "mpi_launch", "workers": plan["workers"],
                         "threads": thread_budget,
                         "timings": timings,
                         "workdir": run_dir if keep_workdir else None}
    finally:
        if not keep_workdir:
            shutil.rmtree(run_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# 公共入口
# ═══════════════════════════════════════════════════════════════════

def einsums(tasks, max_procs=None, optimize="auto", backend="auto",
            reminder=True, verbose=True, workdir=None, keep_workdir=False,
            launcher=None, broadcast=False, worker_threads=None,
            return_info=False):
    """并行执行多个相互独立的 einsum 任务，结果与串行调用逐一对应。

    Parameters
    ----------
    tasks : list | dict
        每个元素是一次 ``pyqcd.einsum.contract`` 的参数集合：``{'subscripts':
        str, 'operands': [...], 'kwargs': {...}}``（``subscripts`` 亦可写作
        ``equation``/``eq``，``kwargs`` 可省略），或等价的
        ``(subscripts, operands[, kwargs])`` 元组。传 dict 返回同键 dict，
        传 list 返回同序 list。
    max_procs : int, optional
        MPI 进程数上限；``None``（默认）表示取系统允许的最大值
        ``min(子任务数, CPU 线程数, 可用内存允许的进程数)``。
    optimize, backend : str
        传给 ``contract`` 的默认值；单个任务的 ``kwargs`` 可覆盖。
    reminder : bool
        首次调用打印「使用场景提醒」（每个进程一次）。
    verbose : bool
        打印规划摘要（rank 0 / 父进程）。
    workdir : str, optional
        自建进程池时临时目录的父目录（默认系统临时目录）。
    keep_workdir : bool
        保留临时目录（调试用）。
    launcher : list, optional
        覆盖 MPI 启动命令，如 ``['mpiexec', '--allow-run-as-root']``。
    worker_threads : int, optional
        自建进程池里每个 worker 的 OMP/BLAS 线程数；默认 ``物理核数 // 进程数``。
        小任务批次建议显式写 1（线程创建开销会盖过并行收益）。
    broadcast : bool
        仅 collective 模式有意义：``True`` 时把完整结果广播给所有 rank
        （默认只在 rank 0 返回结果，其他 rank 返回 ``None``）。
    return_info : bool
        ``True`` 时返回 ``(payload, info)``。``info`` 里带 ``timings``：
        自建进程池时各 worker 自报的**纯计算**耗时（不含 mpirun 启动与落盘），
        用于把"计入初始化"和"不计入初始化"两种口径分开统计。

    Returns
    -------
    list | dict | None
        与输入一一对应的结果；collective 模式下非 rank 0 默认得到 ``None``。
        传 ``return_info=True`` 时返回 ``(payload, info)``。
    """
    global _REMINDER_SHOWN
    kind, keys, specs = _normalize_tasks(tasks)
    comm, rank, size = _mpi_world()

    if reminder and not _REMINDER_SHOWN and rank == 0:
        print(f"[pyqcd.parallel.einsum] {REMINDER_TEXT}", flush=True)
        _REMINDER_SHOWN = True
    elif reminder:
        _REMINDER_SHOWN = True

    if comm is not None and size > 1:
        # 进入 collective 就必然复制输入：这条警告与 reminder 分开，因为它是模式
        # 本身的内存代价，不看用户是否要求打印通用提醒。每个进程只打一次。
        global _COLLECTIVE_WARNING_SHOWN
        if rank == 0:
            if not _COLLECTIVE_WARNING_SHOWN:
                print(f"[pyqcd.parallel.einsum] {COLLECTIVE_MEMORY_WARNING}",
                      flush=True)
            _COLLECTIVE_WARNING_SHOWN = True
        # 规划只在 rank 0 做一次：小任务上 contract_path 占总耗时的 31–84%，
        # 每个 rank 各自重算会把并行收益直接吃掉。
        slim = _slim_plan(_plan_specs(specs, max_procs=max_procs), size) \
            if rank == 0 else None
        slim = comm.bcast(slim, root=0)
        if verbose and rank == 0:
            print(_format_plan(slim, "mpi-collective"), flush=True)
        payload, timings = _run_collective(comm, rank, size, kind, keys, specs,
                                           slim, optimize, backend, broadcast,
                                           verbose)
        if return_info:
            return payload, {"mode": "mpi_collective",
                             "workers": slim["workers"], "timings": timings}
        return payload

    plan = _plan_specs(specs, max_procs=max_procs)
    if plan["workers"] > 1:
        if verbose:
            print(_format_plan(plan, "mpi-launch"), flush=True)
        payload, info = _run_launched(kind, keys, specs, plan, optimize,
                                     backend, workdir, keep_workdir, launcher,
                                     worker_threads)
        return (payload, info) if return_info else payload

    if verbose:
        print(_format_plan(plan, "serial"), flush=True)
    payload = _run_serial(kind, keys, specs, optimize, backend)
    if return_info:
        return payload, {"mode": "serial", "workers": plan["workers"],
                         "timings": None}
    return payload


# ═══════════════════════════════════════════════════════════════════
# worker 入口：由 mpirun 拉起，各进程按 assign.json 领取自己的箱
# ═══════════════════════════════════════════════════════════════════

def _worker_main(run_dir, optimize="auto", backend="auto"):
    rank, _size = _worker_identity()
    with open(os.path.join(run_dir, "assign.json"), encoding="utf-8") as handle:
        bins = json.load(handle)["bins"]
    mine = bins[rank] if rank < len(bins) else []
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    if mine:
        with open(os.path.join(run_dir, "tasks.json"), encoding="utf-8") as handle:
            meta_cache = json.load(handle)
    else:
        meta_cache = []
    compute_seconds = 0.0          # 只累计真正做收缩的时间，不含进程启动与读盘
    for index in mine:
        try:
            # 读盘与计算分开计时：读盘属于进程池的搬运成本，不是收缩本身的开销。
            spec = _read_task(run_dir, index, meta_cache)
            started = time.perf_counter()
            result = _call(spec, optimize, backend)
            compute_seconds += time.perf_counter() - started
        except Exception as exc:
            info = _failure(index, exc)
            with open(os.path.join(results_dir, f"error_{rank:03d}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(info, handle, ensure_ascii=False)
            print(f"[worker {rank}] task {index} failed: {info['type']}: "
                  f"{info['message']}", file=sys.stderr, flush=True)
            return 1
        np.save(os.path.join(results_dir, f"result_{index:05d}.npy"),
                _to_numpy(result))
    # 每 rank 的纯计算耗时：父进程据此把"进程池启动"与"实际计算"分开统计。
    with open(os.path.join(results_dir, f"timing_{rank:03d}.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"tasks": len(mine), "compute_seconds": compute_seconds},
                  handle)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="pyqcd.parallel.einsum worker / planning entry")
    parser.add_argument("--worker", metavar="RUN_DIR",
                        help="执行已分箱的批量任务目录（由 einsums 内部使用）")
    parser.add_argument("--optimize", default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="打印一次最小示例的规划结果")
    args = parser.parse_args(argv)

    if args.worker:
        return _worker_main(args.worker, args.optimize, args.backend)

    if args.dry_run:
        rng = np.random.default_rng(0)
        demo = [{"subscripts": "ij,jk->ik",
                 "operands": [rng.standard_normal((128, 128)),
                              rng.standard_normal((128, 128))]}
                for _ in range(8)]
        plan = plan_einsums(demo)
        print(_format_plan(plan, "dry-run"))
        print(REMINDER_TEXT)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
