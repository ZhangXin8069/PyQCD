"""``--tune`` 的**组级硬超时**：父进程起子进程跑一组，超时由父进程 kill。

为什么必须换掉 ``SIGALRM``
--------------------------
``SIGALRM`` 的处理器只在**回到解释器**时才派发。一组里只要有一次 ``contract()``
落进原生 C/CUDA 代码不返回，闹钟到期也只是一次"迟到的配送"——实测
（``docs/einsum_tune_timeout_20261119.md`` §2）：单次 2.34 s 的原生 BLAS、上限 0.3 s，
实际 2.29 s 才抛（超限 1.99 s）；用户那条 10 算子生产方程在 ``STAT=Rl+``、
``%CPU=100``、``GPU 0%`` 状态下挂了 3 分钟以上。

能真正打断原生代码的只有**父进程 kill 子进程**。这里就是那条路线的编排层：
:mod:`pyqcd.einsum._tune_worker` 在新解释器里跑一组的 ``warmup + rounds`` 次
``contract()``，本模块负责起进程、等结果、到点强杀。

为什么是**新解释器**而不是 ``fork``
-----------------------------------
CUDA 上下文不跨 ``fork`` 复用：父进程只要碰过 CUDA，``fork`` 出来的子进程第一次
``cupy.asarray`` 就是 ``cudaErrorInitializationError``（本机 RTX 4060 + cupy 14.2
实测；用户复测时是 22 组全 error）。``subprocess`` 起全新解释器则完全没有这个问题。

操作数怎么传
------------
父进程按固定 ``seed`` 生成**一次**、``np.save`` 落盘到**一个**暂存目录，所有组共用
（生成 3.81 GiB 实测 8.8 s、tmpfs 落盘 1.7 s、读回 1.2 s——**一次**摊到 21 组上可以
接受，每组各落一份则不可接受）。``np.save``/``np.load`` 逐位无损，所以"子进程看到的
数"与"同进程直接调用看到的数"是同一批。

落点由 ``PYQCD_EINSUM_TUNE_STAGE_DIR`` 指定；未设时用系统临时目录。需要更快的
I/O 时把它指到 tmpfs（本机 ``/dev/shm`` 实测 2243 MiB/s），但要注意 tmpfs 占的是
内存——操作数总字节超过可用内存时会 OOM。

启动开销单列
------------
每组要付一次解释器启动 + ``import numpy``/``import cupy`` + 操作数载入（本机实测
``import pyqcd.einsum`` 中位 **126 ms**、``import cupy`` 约 **244 ms**）。这段耗时
**不计入** ``wall_ms``：``wall_ms`` 量的是"这一组配置下的 ``contract()`` 要多久"，
进程启动不是被测对象。启动耗时单独累计进 ``startup_ms`` 并在报告里单列一行。

落回进程内
----------
``PYQCD_EINSUM_TUNE_INPROCESS=1``（或 ``true``/``yes``/``on``）显式退回旧的进程内
``SIGALRM`` 路线：只为"临时确认某事"或对照实验，代价是原生调用打不断。
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

__all__ = [
    "INPROCESS_ENV", "STAGE_ENV", "READY_TIMEOUT_S", "OUTPUT_MAX_BYTES",
    "isolation_enabled", "stage_dir", "stage_operands", "run_group", "cleanup",
    "summarize_startup",
]

#: 显式退回进程内 ``SIGALRM`` 路线的开关（``1``/``true``/``yes``/``on`` = 退回）。
INPROCESS_ENV = "PYQCD_EINSUM_TUNE_INPROCESS"

#: 操作数暂存目录的覆盖点；未设时用系统临时目录。
STAGE_ENV = "PYQCD_EINSUM_TUNE_STAGE_DIR"

#: 等子进程 ``ready``（解释器启动 + 导入 + 操作数载入）的上限秒数。
#:
#: 它与 ``--tune-timeout`` 是**两件事**：那个上限量的是 ``contract()``，而进程启动不是
#: ``contract()``。这里给一个足够宽的上限（GPU 上首次建 CUDA 上下文可能上秒级），
#: 超了按 ``error`` 记账并如实报出来，不冒充 timeout。
READY_TIMEOUT_S = 300.0

#: 轮询文件系统的间隔（秒）。``ready``/结果文件都很小，0.005 s 足够灵敏又不烧 CPU。
_POLL_S = 0.005

#: 结果的输出张量超过这个字节数就不落盘：``rel`` 校验会退化成"未校验"并如实标注。
#: 与参照相比它只是"多一次可选的 I/O"，不该因为一个巨大的输出把磁盘写满。
OUTPUT_MAX_BYTES = 256 * 1024 * 1024


class GroupTimeout(RuntimeError):
    """父进程在组级上限处强杀了子进程（``status=timeout``）。"""


def isolation_enabled() -> bool:
    """本次是否走**子进程隔离**（缺省是；``PYQCD_EINSUM_TUNE_INPROCESS=1`` 才退回）。"""
    text = str(os.environ.get(INPROCESS_ENV, "")).strip().lower()
    return text not in ("1", "true", "yes", "on")


def stage_dir() -> str:
    """操作数暂存目录：``PYQCD_EINSUM_TUNE_STAGE_DIR`` 或系统临时目录下的新目录。"""
    configured = str(os.environ.get(STAGE_ENV, "")).strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return tempfile.mkdtemp(prefix="pyqcd-einsum-tune-", dir=configured)
    return tempfile.mkdtemp(prefix="pyqcd-einsum-tune-")


def stage_operands(operands: Sequence[Any], directory: str) -> dict[str, Any]:
    """把操作数落盘到 ``directory``（``operand_NN.npy``），返回落盘统计。

    文件名**定长补零**：子进程按文件名排序读回，轴序因此不依赖调用方传参顺序。
    ``np.save`` 对 numpy / cupy / torch 都接受数组协议，cupy 张量由 ``np.save`` 自己
    搬到主机（这里不自己 ``.get()``，避免多一次拷贝）。
    """
    import numpy as np

    written = 0
    started = time.perf_counter()
    for index, operand in enumerate(operands):
        path = os.path.join(directory, f"operand_{index:02d}.npy")
        np.save(path, operand)
        try:
            written += int(os.path.getsize(path))
        except OSError:                       # pragma: no cover - 竞态，不影响正确性
            written += int(getattr(operand, "nbytes", 0) or 0)
    return {
        "dir": directory,
        "count": len(operands),
        "bytes": written,
        "seconds": time.perf_counter() - started,
    }


def _wait_for(path: str, deadline: float | None) -> bool:
    """等 ``path`` 出现；``deadline`` 到点仍未出现则返回 ``False``。"""
    while True:
        if os.path.exists(path):
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)


def _read_json(path: str) -> dict[str, Any] | None:
    """读结果文件；缺失或半截（父进程强杀时会这样）都返回 ``None``。"""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _kill_group(process: subprocess.Popen) -> None:
    """强杀**整个进程组**（子进程可能再起线程/孙进程，只杀 pid 会留孤儿）。

    ``start_new_session=True`` 让子进程成为新会话的首进程，于是 ``getpgid(pid)`` 就是
    它自己的组，``killpg`` 打不到本进程。
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):   # pragma: no cover - 已自行退出
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:                # pragma: no cover - 不该发生
        pass


def run_group(job: dict[str, Any], *, workdir: str,
              group_limit_s: float | None,
              deadline: float | None = None) -> dict[str, Any]:
    """在新解释器里跑一组，返回观察结果；超时强杀并按 ``timeout`` 记账。

    ``group_limit_s`` 从子进程写下 ``ready`` 那一刻开始计时——``--tune-timeout`` 说的是
    ``contract()`` 的上限，进程启动与操作数载入不是 ``contract()``，各自另有上界
    （:data:`READY_TIMEOUT_S`）。``group_limit_s is None`` = 不设上限（只受 ``deadline``
    约束）。

    ``deadline`` 是**整轮**的 ``time.monotonic()`` 上限（``--tune-budget``），比组级上限
    更紧时按它截断，并记 ``budget``——那是"整轮没时间了"，不是"这一组太慢"。

    返回的 ``status`` 取 ``ok`` / ``timeout`` / ``budget`` / ``error`` / ``pruned``；
    ``startup_s`` 与 ``wall_s`` 分开给（前者是进程启动 + 载入，后者是受测的时间区）。
    """
    os.makedirs(workdir, exist_ok=True)
    ready_path = os.path.join(workdir, "ready")
    result_path = os.path.join(workdir, "result.json")
    job_path = os.path.join(workdir, "job.json")
    for stale in (ready_path, result_path):
        try:
            os.remove(stale)
        except OSError:
            pass
    payload = dict(job)
    payload["ready_path"] = ready_path
    payload["result_path"] = result_path
    with open(job_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, default=str)

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    # 子进程不继承父进程的调优开关：它跑的是**一组**，不是一整轮。
    environment.pop(INPROCESS_ENV, None)
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", "pyqcd.einsum._tune_worker", job_path],
        cwd=None, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True, text=True)

    ready_deadline = started + READY_TIMEOUT_S
    if deadline is not None:
        ready_deadline = min(ready_deadline, deadline)
    became_ready = _wait_for(ready_path, ready_deadline)
    if not became_ready:
        _kill_group(process)
        stdout, stderr = _drain(process)
        exhausted = deadline is not None and time.monotonic() >= deadline
        return {
            "status": "budget" if exhausted else "error",
            "reason": (("整轮 --tune-budget 在子进程启动阶段用尽" if exhausted else
                        f"子进程 {READY_TIMEOUT_S:g}s 内没有写下 ready（启动或操作数载入"
                        f"失败）；stderr={stderr.strip()[:800] or '-'}")),
            "startup_s": time.monotonic() - started,
            "wall_s": None, "stdout": stdout, "stderr": stderr,
        }
    ready_at = time.monotonic()
    startup_s = ready_at - started

    limit = group_limit_s
    status_override: str | None = None
    if deadline is not None:
        left = deadline - ready_at
        if left <= 0.0:
            _kill_group(process)
            return {
                "status": "budget",
                "reason": "整轮 --tune-budget 在首组调用之前用尽",
                "startup_s": startup_s, "wall_s": None,
            }
        if limit is None or left < limit:
            limit, status_override = left, "budget"
    hard_deadline = None if limit is None else ready_at + float(limit)

    while True:
        if os.path.exists(result_path):
            break
        if process.poll() is not None:
            break
        if hard_deadline is not None and time.monotonic() >= hard_deadline:
            _kill_group(process)
            stdout, stderr = _drain(process)
            return {
                "status": status_override or "timeout",
                "reason": (f"组级上限 {float(limit):g}s 已到，父进程强杀子进程"
                           f"（原生 C/CUDA 代码打不断，只能这样）"),
                "startup_s": startup_s,
                "wall_s": time.monotonic() - ready_at,
                "stdout": stdout, "stderr": stderr,
            }
        time.sleep(_POLL_S)

    wall_s = time.monotonic() - ready_at
    stdout, stderr = _drain(process)
    observation = _read_json(result_path)
    if observation is None:
        code = process.returncode
        return {
            "status": "error",
            "reason": (f"子进程没有写出结果文件（exit={code}）；"
                       f"stderr={stderr.strip()[:800] or '-'}"),
            "startup_s": startup_s, "wall_s": wall_s,
            "stdout": stdout, "stderr": stderr,
        }
    observation.setdefault("status", "ok")
    observation["startup_s"] = startup_s
    # ``wall_s``：子进程自己测的那个才是受测时间区的真实墙钟（父进程只能看见
    # ``ready`` 出现的时刻，轮询粒度 5 ms——一组比这更快时父进程量出来的会是假数）。
    # 子进程没给（旧版 worker / 异常路径）才退回父进程口径。
    if observation.get("wall_s") is None:
        observation["wall_s"] = wall_s
    if stderr:
        observation.setdefault("stderr", stderr.strip()[:800])
    return observation


def _drain(process: subprocess.Popen) -> tuple[str, str]:
    """收完子进程的 stdout/stderr（进程已退出或已强杀；带上限避免阻塞）。"""
    try:
        stdout, stderr = process.communicate(timeout=10.0)
    except subprocess.TimeoutExpired:                # pragma: no cover - 已被强杀
        _kill_group(process)
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            return "", ""
    except ValueError:                               # pragma: no cover - 流已关闭
        return "", ""
    return stdout or "", stderr or ""


def cleanup(directory: str) -> None:
    """删掉暂存目录；失败不抛（临时目录残留不该让一次成功的遍历变失败）。"""
    shutil.rmtree(directory, ignore_errors=True)


def summarize_startup(values: Iterable[float]) -> dict[str, Any]:
    """启动耗时的报告摘要（毫秒）：次数、总计、中位、最大。

    单列的理由：启动开销是**测量开销**，不是被测对象的耗时。把它混进 ``wall_ms``
    会让"这一组配置要多久"凭空多出 0.13–0.5 s，而缓存/内核选路的结论正是基于这个数。
    """
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "total_ms": 0.0, "median_ms": None, "max_ms": None}
    ordered = sorted(items)
    middle = len(ordered) // 2
    median = (ordered[middle] if len(ordered) % 2 else
              0.5 * (ordered[middle - 1] + ordered[middle]))
    return {
        "count": len(items),
        "total_ms": round(sum(items) * 1000.0, 3),
        "median_ms": round(median * 1000.0, 3),
        "max_ms": round(ordered[-1] * 1000.0, 3),
    }
