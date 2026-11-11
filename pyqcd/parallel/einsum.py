"""MPI 任务级并行执行多个相互独立的 :mod:`pyqcd.einsum` 收缩。

本模块只做「分发与汇总」，不改变 ``pyqcd.einsum`` 的任何行为：每个子任务仍是
一次普通的 ``contract(...)`` 调用，参数与返回值语义完全一致。

适用场景（也是本功能的边界）：单队列里排着大批量、但每个任务都很小的 einsum。
这类任务无法把单次收缩拆到多进程（单个收缩的并行度受限），只能做**任务级**并行。

公开接口只有两步（不提供一体化包装，避免与第三方 Einsums 库的名字混淆）：

* :func:`init` —— 只接受列式字典 ``{task_id, equation, operands, results}``，
  校验后做 LPT 规划，返回句柄（不执行计算）；
* :func:`run` —— 启动器语义，无返回值，结果**原地**写入 ``init`` 时传入的
  ``results`` 缓冲区，不重新分配内存或显存。

两种运行方式：

* 已在 MPI 下（``mpirun -np N python your_script.py``）：各 rank 都调用
  ``init`` + ``run``，默认只由 rank 0 串行执行整批、其余 rank 立即返回。
  ``collective=True`` 会让各 rank 分摊任务（零序列化），但**该模式已弃用**：
  每个 rank 都要持有整批输入，内存/显存代价随 rank 数线性增长，且不再纳入测试。
* 不在 MPI 下：自己 ``mpirun -np K python -m pyqcd.parallel.einsum --worker
  <dir>`` 拉起 K 个进程，任务与结果经临时目录传递。

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

__all__ = ["init", "run", "plan_tasks", "REMINDER_TEXT",
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
    """返回 ``(keys, specs)``：任务名（list 用下标）与规范化后的任务描述。"""
    if isinstance(tasks, dict):
        if not tasks:
            raise ValueError("tasks mapping must not be empty")
        keys = list(tasks.keys())
        return keys, [_normalize_one(tasks[key], key) for key in keys]
    if isinstance(tasks, (list, tuple)):
        if not tasks:
            raise ValueError("tasks sequence must not be empty")
        return list(range(len(tasks))), [
            _normalize_one(spec, index) for index, spec in enumerate(tasks)]
    raise TypeError(
        "tasks must be a list/tuple of task specs or a mapping of "
        f"name -> task spec, got {type(tasks).__name__}")


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


def plan_tasks(tasks, operands=None, max_procs=None, resources=None):
    """只做规划、不执行：返回进程数、分箱与负载统计（便于自查与复现）。

    两种调用方式：

    * ``plan_tasks(tasks)`` —— ``tasks`` 是 dict 或 list 的任务描述；
    * ``plan_tasks(equations, operands)`` —— 两个等长序列（与 ``init`` 的列式字典
      同形），便于在 ``init`` 之前单独看分箱。
    """
    if operands is not None:
        tasks = list(zip(tasks, operands))
    _keys, specs = _normalize_tasks(tasks)
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
    raise RuntimeError(f"pyqcd.parallel.einsum: {len(failures)} "
                       f"task(s) failed: {details}")


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

def _run_serial(specs, optimize, backend, keys=None):
    """逐个执行，返回与 ``specs`` 同序的结果列表；任何失败都会整体抛错。

    ``keys`` 只用于错误信息（``task_id`` 或下标），不影响执行。
    """
    ordered, failures = [], []
    for index, spec in enumerate(specs):
        try:
            ordered.append(_call(spec, optimize, backend))
        except Exception as exc:
            failures.append(_failure(index, exc))
    if failures:
        _raise_failures(failures, keys if keys is not None
                        else list(range(len(specs))))
    return ordered


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


def _run_collective(comm, rank, size, keys, specs, slim, optimize,
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
    timings = {"per_rank_seconds": per_rank,
               "compute_seconds": max(per_rank) if per_rank else 0.0,
               "pool_overhead_seconds": 0.0}   # collective 没有进程池与落盘
    if broadcast:
        ordered = comm.bcast(ordered, root=0)
    return ordered, timings


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
        return _run_serial(specs, optimize, backend), {
            "mode": "serial_fallback", "workers": 1}

    run_dir = tempfile.mkdtemp(prefix="pyqcd-einsum-", dir=workdir)
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
                    f"pyqcd.parallel.einsum: {len(errors) or len(missing)} "
                    f"task(s) failed: "
                    f"{detail}\n--- launcher output ---\n{tail}")
            raise RuntimeError(
                f"pyqcd.parallel.einsum: MPI launch failed "
                f"(exit={completed.returncode}); "
                f"no task reported an error\n"
                f"--- launcher output ---\n{tail}")
        ordered = [np.load(os.path.join(results_dir, f"result_{index:05d}.npy"))
                   for index in range(len(specs))]
        timings = _collect_worker_timings(results_dir)
        return ordered, {"mode": "mpi_launch", "workers": plan["workers"],
                         "threads": thread_budget,
                         "timings": timings,
                         "workdir": run_dir if keep_workdir else None}
    finally:
        if not keep_workdir:
            shutil.rmtree(run_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# 公共入口：init（规划）/ run（启动器，结果原位写入）
# ═══════════════════════════════════════════════════════════════════

def _execute(specs, keys, plan, optimize, backend, collective, workdir,
             keep_workdir, launcher, worker_threads, broadcast, verbose):
    """按 collective → 自建池 → 串行 的顺序执行，返回 ``(ordered, info)``。

    ``collective`` 模式**已弃用**且不再纳入测试：每个 rank 都要持有整批输入，
    内存/显存代价随 rank 数线性增长。默认（``collective=False``）在 ``mpirun``
    内只由 rank 0 执行、其余 rank 直接返回 ``None``，不做任何复制。
    """
    comm, rank, size = _mpi_world()

    if collective:
        warnings.warn(
            "pyqcd.parallel.einsum: collective mode is deprecated and is no "
            "longer tested; every rank holds a full copy of the batch. Use the "
            "default (rank 0 only) or a self-launched pool instead.",
            DeprecationWarning, stacklevel=2)

    if comm is not None and size > 1 and not collective:
        if verbose and rank == 0:
            print("[pyqcd.parallel.einsum] collective mode is deprecated and "
                  "off by default (every rank would hold a full copy of the "
                  "batch). Running on rank 0 only.", flush=True)
        if rank != 0:
            return None, {"mode": "rank0_only", "workers": 1, "timings": None}
        ordered = _run_serial(specs, optimize, backend, keys)
        return ordered, {"mode": "rank0_only", "workers": 1, "timings": None}

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
        slim = _slim_plan(plan, size) if rank == 0 else None
        slim = comm.bcast(slim, root=0)
        if verbose and rank == 0:
            print(_format_plan(slim, "mpi-collective"), flush=True)
        ordered, timings = _run_collective(comm, rank, size, keys, specs, slim,
                                           optimize, backend, broadcast, verbose)
        return ordered, {"mode": "mpi_collective", "workers": slim["workers"],
                         "timings": timings}

    if plan["workers"] > 1:
        if verbose:
            print(_format_plan(plan, "mpi-launch"), flush=True)
        payload, info = _run_launched("list", keys, specs, plan, optimize,
                                      backend, workdir, keep_workdir, launcher,
                                      worker_threads)
        return payload, info

    if verbose:
        print(_format_plan(plan, "serial"), flush=True)
    ordered = _run_serial(specs, optimize, backend, keys)
    return ordered, {"mode": "serial", "workers": plan["workers"],
                     "timings": None}


def _show_reminder(reminder):
    global _REMINDER_SHOWN
    if reminder and not _REMINDER_SHOWN:
        print(f"[pyqcd.parallel.einsum] {REMINDER_TEXT}", flush=True)
    _REMINDER_SHOWN = True


_CONFIG_KEYS = frozenset(("task_id", "equation", "operands", "results"))


def init(config, max_proc=None, verbose=False, reminder=True,
         optimize="auto", backend="auto", collective=False, workdir=None,
         keep_workdir=False, launcher=None, worker_threads=None,
         broadcast=False):
    """校验并规划一批任务，返回可直接交给 :func:`run` 的句柄（不执行计算）。

    只接受**列式字典**，且结果写进调用方预先分配好的缓冲区，因此可以先 ``init``
    计时、再 ``run`` 计时，把规划与执行分开测量。

    Parameters
    ----------
    config : dict
        四个键，前三个必需::

            {
              "task_id":  ["t0", "t1", ...],          # 可选，缺省用下标
              "equation": ["ij,jk->ik", ...],         # 与任务一一对应
              "operands": [[a0, b0], [a1, b1], ...],  # 只读，不会被复制或改动
              "results":  [out0, out1, ...],          # 可写，结果原地写入
            }

        ``operands`` 里的数组**只读**：``init``/``run`` 不复制、不改写它们。
        ``results`` 里的数组由调用方分配，``run`` 直接把结果写进这些缓冲区
        （``results[i][...] = value``），不重新分配内存或显存。缓冲区必须与操作数
        **同设备**：GPU 上要给设备数组，给 numpy 缓冲区会触发 CuPy 的
        “implicit conversion to a NumPy array is not allowed” 而报 ``TypeError``。
    max_proc : int, optional
        进程数上限；``None`` 表示 ``min(子任务数, CPU 线程数, 可用内存允许值)``。
    verbose : bool
        调试输出，默认 ``False``；打开后每条都带 ``task_id``，便于定位。
    reminder : bool
        首次调用打印使用场景提醒。
    其余参数（``optimize`` / ``backend`` / ``workdir`` / ``keep_workdir`` /
    ``launcher`` / ``worker_threads`` / ``broadcast``）含义与 ``contract`` /
    进程池设置一致。``collective`` **已弃用**且不再纳入测试：打开后每个 rank
    都会持有整批输入，内存/显存随 rank 数线性增长；默认只由 rank 0 执行。

    Returns
    -------
    dict
        句柄，含 ``specs``（内部任务描述）、``plan``（分箱与负载）、``keys``
        （``task_id``）与 ``results``（调用方的输出缓冲区引用）。
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict with keys "
                        f"{sorted(_CONFIG_KEYS)}; got {type(config).__name__}")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"config has unsupported key(s): {sorted(unknown)}; "
                         f"allowed: {sorted(_CONFIG_KEYS)}")
    for required in ("equation", "operands", "results"):
        if required not in config:
            raise ValueError(f"config is missing required key {required!r}")
    equations = config["equation"]
    operands = config["operands"]
    results = config["results"]
    for name, value in (("equation", equations), ("operands", operands),
                        ("results", results)):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"config[{name!r}] must be a list, "
                            f"got {type(value).__name__}")
    count = len(equations)
    if count == 0:
        raise ValueError("config must describe at least one task")
    if len(operands) != count or len(results) != count:
        raise ValueError(
            f"config lengths differ: equation={count} operands={len(operands)} "
            f"results={len(results)}")
    ids = config.get("task_id")
    if ids is None:
        ids = list(range(count))
    elif not isinstance(ids, (list, tuple)) or len(ids) != count:
        raise ValueError("config['task_id'] must be a list with one entry per "
                         "task (or omitted)")

    specs, cache = [], {}
    for index in range(count):
        equation = equations[index]
        task_operands = operands[index]
        if not isinstance(equation, str):
            raise TypeError(f"task {ids[index]!r} equation must be str, "
                            f"got {type(equation).__name__}")
        if not isinstance(task_operands, (list, tuple)) or not task_operands:
            raise TypeError(f"task {ids[index]!r} operands must be a non-empty "
                            f"list, got {type(task_operands).__name__}")
        target = results[index]
        # 两层 getattr：设备数组的 flags 未必提供 writeable（cupy 就没有），
        # 直接取属性会抛 AttributeError，而这只是"能否提前发现只读缓冲区"的
        # 便利检查，不该成为缓冲区类型的准入门槛。
        writeable = getattr(getattr(target, "flags", None), "writeable", None)
        if writeable is not None and not writeable:
            raise ValueError(f"task {ids[index]!r} result buffer is read-only; "
                             "run() writes results in place")
        specs.append({"subscripts": equation,
                      "operands": list(task_operands),
                      "kwargs": {}})

    plan = _plan_specs(specs, max_procs=max_proc)
    # 输出形状校验：提前失败好过算完才发现缓冲区尺寸不对。
    for index, spec in enumerate(specs):
        expected = _expected_output_shape(spec, cache)
        actual = tuple(getattr(results[index], "shape", ()))
        if expected is not None and actual != expected:
            raise ValueError(
                f"task {ids[index]!r} result buffer shape {actual} does not "
                f"match the contraction output {expected}")

    if verbose:
        print(_format_plan(plan, "init"), flush=True)
    return {"keys": list(ids), "specs": specs, "plan": plan,
            "results": list(results), "verbose": bool(verbose),
            "reminder": bool(reminder), "optimize": optimize,
            "backend": backend, "collective": bool(collective),
            "workdir": workdir, "keep_workdir": keep_workdir,
            "launcher": launcher, "worker_threads": worker_threads,
            "broadcast": broadcast}


def run(state, max_proc=None):
    """执行 :func:`init` 规划好的批次，结果**原地**写入 ``state["results"]``。

    启动器语义：返回 ``None``；数据通过调用方传入的缓冲区交换，不产生新的返回值，
    因此调用前后可以做精确的耗时/内存对照——`init` 计时得到规划开销，`run` 计时
    得到真实执行开销。在 ``mpirun`` 内默认只由 rank 0 执行，其余 rank 立即返回。

    Parameters
    ----------
    state : dict
        :func:`init` 的返回值。
    max_proc : int, optional
        覆盖 ``init`` 时的进程数上限（一般不用）。
    """
    if not isinstance(state, dict) or "specs" not in state:
        raise TypeError("run() expects the dict returned by init()")
    specs = state["specs"]
    plan = state["plan"]
    if max_proc is not None and max_proc != state["plan"]["workers"]:
        plan = _plan_specs(specs, max_procs=max_proc)
    _show_reminder(state.get("reminder", True))
    ordered, info = _execute(
        specs, state["keys"], plan, state["optimize"], state["backend"],
        state["collective"], state["workdir"], state["keep_workdir"],
        state["launcher"], state["worker_threads"], state["broadcast"],
        state["verbose"])
    # 启动器语义：返回 None，但把这次执行的统计写回 state，供调用方分段计时
    # （`run` 没有返回值，信息不能丢）。
    state["mode"] = info.get("mode")
    state["timings"] = info.get("timings")
    if ordered is None:
        return None
    for target, value in zip(state["results"], ordered):
        # 直接写进调用方的缓冲区：不分配新数组，也不改 dtype/形状。
        target[...] = value
    return None


def _expected_output_shape(spec, cache):
    """按 ``contract_path`` 的 size_dict 推出输出形状；失败时返回 ``None``。"""
    key = (spec["subscripts"], tuple(_shape_of(op) for op in spec["operands"]))
    if key in cache:
        return cache[key]
    shape = None
    try:
        _, info = contract_path(spec["subscripts"], *key[1], shapes=True,
                                optimize="greedy")
        shape = tuple(info.size_dict[label] for label in info.output_subscript)
    except Exception:
        shape = None
    cache[key] = shape
    return shape



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
                        help="执行已分箱的批量任务目录（由 init/run 内部使用）")
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
        plan = plan_tasks(demo)
        print(_format_plan(plan, "dry-run"))
        print(REMINDER_TEXT)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
