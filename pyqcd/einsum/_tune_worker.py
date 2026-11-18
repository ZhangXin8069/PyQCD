"""``--tune`` 的**子进程执行入口**：跑一组 ``contract()``，把观察结果回传父进程。

为什么需要它
------------
``--tune-timeout`` 以前用 ``signal.setitimer`` + ``SIGALRM`` 实现，而信号处理器只在
**回到解释器**时才派发：一旦某次 ``contract()`` 进入原生 C/CUDA 代码就不再回来，
闹钟等于没装。2026-11-18 用户实测：10 算子方程的 ``--tune`` 在 ``STAT=Rl+``、
``%CPU=100``、``GPU 0%`` 状态下挂 3 分钟以上，``--tune-timeout`` 完全没生效。
能真正打断原生代码的只有**父进程 kill 子进程**，所以受测调用搬到子进程里跑，
超时由父进程强制生效（:func:`pyqcd.einsum.test._run_group_in_subprocess`）。

为什么是**新解释器**而不是 ``fork``
-----------------------------------
``fork`` 更便宜，但 CUDA 上下文不能跨 ``fork`` 复用：父进程只要碰过 CUDA
（``getDeviceCount()`` 或任意一次分配就够），子进程里第一次 ``cupy.asarray`` 就是
``cudaErrorInitializationError``（本机 RTX 4060 + cupy 14.2 实测；反之父进程从未
``import cupy`` 时 fork 出来的子进程能正常初始化 CUDA）。而父进程必须碰 CUDA——它要
建操作数、并从**设备对象**上读出 ``device='cupy:0'`` 这类键列。所以这里用
``subprocess`` 起全新解释器：每组付一次进程启动（实测中位 ~250 ms，含 ``import cupy``），
换来"父进程的 CUDA 状态一律不参与"。

子进程**只做一件事**
--------------------
载入操作数 → 跑 ``warmup`` + ``rounds`` 次 ``contract()`` → 把观察结果（耗时样本、
结果形状/dtype、是否异常、最近一次选路记录）写进结果文件。**写配置表、打印表格、
选优、剪枝一律在这里不做**：那是父进程的编排工作，而且父进程才是能杀死本进程的一方。

通信方式（三个文件，都在父进程给的临时目录里）
----------------------------------------------
* ``job.json``：父进程 → 子进程的任务单（方程、形状、dtype、backend、三轴取值、
  ``warmup``/``rounds``、操作数暂存路径、结果路径）。
* ``operands/*.npy``：父进程按固定 ``seed`` 生成**一次**、落盘，子进程按原样读回
  （``np.save``/``np.load`` 逐位无损，所以"子进程看到的数"与"同进程直接调用看到的数"
  是同一批）。**不从父进程搬设备内存**：父进程的 CUDA 状态一概不参与。
* ``ready``：子进程在**第一次调用之前**创建它。父进程的 ``--tune-timeout`` 从这一刻
  开始计时——--tune-timeout 说的是"每一次 ``contract()`` 的上限"，进程启动与操作数
  载入不是 ``contract()``，不该占它的额度（各自另有上界，超了如实报 ``error``）。
* ``result.json``：子进程 → 父进程的观察结果。父进程读不到就按 ``timeout``/``error``
  记账，**不回退成"再看一眼耗时"**。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any

__all__ = ["run_job", "main"]

#: 让受测调用**人为卡住**的调试开关（秒，缺省未设＝不启用）。
#:
#: 只给契约用：在 ``ready`` 之后、第一次 ``contract()`` 之前睡这么久，模拟"进去了就
#: 不返回"的原生调用。生产路径不设它时一个字节都不走这段代码。
SLEEP_ENV = "PYQCD_EINSUM_TUNE_WORKER_SLEEP"

#: 让子进程**立刻崩溃**的调试开关（退出码，缺省未设＝不启用）。
#:
#: 只给契约用：在 ``ready`` 之后直接 ``os._exit(<code>)``，模拟段错误/OOM 这一类
#: "连结果文件都写不出来"的崩溃。
EXIT_ENV = "PYQCD_EINSUM_TUNE_WORKER_EXIT"


def _load_operands(directory: str, device: str) -> list[Any]:
    """按文件名顺序读回父进程暂存的操作数；非 CPU 设备上再搬到显存。"""
    import numpy as np

    paths = sorted(glob.glob(os.path.join(directory, "operand_*.npy")))
    host = [np.load(path) for path in paths]
    if str(device or "cpu").strip().lower() == "cpu":
        return host
    import cupy

    return [cupy.asarray(item) for item in host]


def _filter_profile_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """只保留底层 ``contract()`` 真的接受的关键字（旧版库上退化，不伪造）。"""
    import inspect

    import pyqcd.einsum as pe

    try:
        names = inspect.signature(pe.contract).parameters
    except (TypeError, ValueError):             # pragma: no cover - 装饰器/扩展
        return {}
    return {key: value for key, value in kwargs.items() if key in names}


def run_job(job: dict[str, Any], mark_ready) -> dict[str, Any]:
    """跑一组 ``warmup`` + ``rounds`` 次 ``contract()``，返回观察结果。

    ``mark_ready`` 在**第一次调用之前**被调用一次（父进程据此开始计 ``--tune-timeout``）。
    返回的字典就是写进 ``result.json`` 的内容；父进程只认这几个键：``median_s`` /
    ``samples`` / ``shape`` / ``dtype`` / ``last_kernel`` / ``kernel_info``。
    """
    import statistics

    import numpy as np

    import pyqcd.einsum as pe
    from pyqcd.einsum import _api

    equation = str(job["equation"])
    device = str(job.get("device") or "cpu")
    backend = job.get("backend")
    warmup = max(int(job.get("warmup") or 0), 0)
    rounds = max(int(job.get("rounds") or 1), 1)
    operands = _load_operands(str(job["operands_dir"]), device)
    kwargs = dict(job.get("contract_kwargs") or {})
    kwargs.update(_filter_profile_kwargs(dict(job.get("profile_kwargs") or {})))
    kwargs["backend"] = backend
    # 空串 = 调用方没指定（``None`` 一路传到 ``contract()``，由 ``_api`` 自己回落）。
    if job.get("kernel_force") is not None:
        kwargs["kernel_force"] = job["kernel_force"]
    kwargs["as_equivalence"] = job.get("as_equivalence")
    kwargs["optimize"] = job.get("optimize")

    def clear_records() -> None:
        """清掉父进程留不下来的选路记录，让观察只反映**本组**。"""
        if hasattr(_api, "_LAST_KERNEL"):
            _api._LAST_KERNEL = None
        for name in ("_kernel_cuda", "_kernel_kupl"):
            try:
                module = __import__(f"pyqcd.einsum.{name}", fromlist=[name])
            except ImportError:                 # pragma: no cover - 旧版库
                continue
            info = getattr(module, "LAST_INFO", None)
            if isinstance(info, dict):
                info.clear()

    sleep = os.environ.get(SLEEP_ENV, "").strip()
    crash = os.environ.get(EXIT_ENV, "").strip()
    clear_records()
    mark_ready()
    if crash:
        os._exit(int(crash))
    if sleep:
        # 模拟"进入原生代码后不返回"：就在计时区里睡，父进程只能靠 kill 打断。
        time.sleep(float(sleep))
    for _ in range(warmup):
        pe.contract(equation, *operands, **kwargs)
    samples = []
    result = None
    for _ in range(rounds):
        started = time.perf_counter()
        result = pe.contract(equation, *operands, **kwargs)
        samples.append(time.perf_counter() - started)
    summary: dict[str, Any] = {
        "median_s": float(statistics.median(samples)),
        "samples": [float(item) for item in samples],
        "shape": list(np.shape(result)),
        "dtype": str(getattr(result, "dtype", "")),
        "last_kernel": _api.last_kernel() if hasattr(_api, "last_kernel") else None,
        "kernel_info": _kernel_info(),
    }
    output = job.get("output_operand")
    if output:
        np.save(str(output), _to_host(result))
    return summary


def _kernel_info() -> dict[str, Any]:
    """KUPL/CUDA 入口最近一次调用的路线信息（与 ``test._kernel_info`` 同口径）。"""
    info: dict[str, Any] = {}
    for name in ("kupl", "cuda"):
        try:
            module = __import__(f"pyqcd.einsum._kernel_{name}", fromlist=[name])
        except ImportError:                     # pragma: no cover - 旧版库
            continue
        last = getattr(module, "LAST_INFO", None)
        if last:
            info[name] = {key: last[key] for key in
                          ("path", "specialized", "steps", "dtype", "copied")
                          if key in last}
    return info


def _to_host(value):
    """cupy/torch 张量 → numpy（只用于落盘比对）。"""
    import numpy as np

    module = type(value).__module__
    if module.startswith("cupy"):
        return value.get()
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main(argv: list[str] | None = None) -> int:
    """``python -m pyqcd.einsum._tune_worker <job.json>`` 的入口。"""
    items = list(sys.argv[1:] if argv is None else argv)
    if len(items) != 1:
        print("usage: python -m pyqcd.einsum._tune_worker <job.json>",
              file=sys.stderr)
        return 2
    with open(items[0], encoding="utf-8") as handle:
        job = json.load(handle)
    ready_path = str(job["ready_path"])
    result_path = str(job["result_path"])

    def mark_ready() -> None:
        with open(ready_path, "w", encoding="utf-8") as handle:
            handle.write(str(time.time()))

    try:
        summary = run_job(job, mark_ready)
    except BaseException as exc:                # noqa: BLE001 - 回传给父进程记账
        _write(result_path, {"error": f"{type(exc).__name__}: {exc}",
                             "error_type": type(exc).__name__})
        return 1
    _write(result_path, summary)
    return 0


def _write(path: str, payload: dict[str, Any]) -> None:
    """原子写结果文件：父进程只会看到"完整"或"不存在"两种状态。"""
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, default=str)
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
