"""``--tune`` 的**子进程执行入口**：跑一组 ``contract()``，把观察结果回传父进程。

为什么需要它
------------
``--tune-timeout`` 以前用 ``signal.setitimer`` + ``SIGALRM`` 实现，而信号处理器只在
**回到解释器**时才派发：一旦某次 ``contract()`` 进入原生 C/CUDA 代码就不再回来，
闹钟等于没装。2026-11-18 用户实测：10 算子方程的 ``--tune`` 在 ``STAT=Rl+``、
``%CPU=100``、``GPU 0%`` 状态下挂 3 分钟以上，``--tune-timeout`` 完全没生效。
能真正打断原生代码的只有**父进程 kill 子进程**，所以受测调用搬到子进程里跑，
超时由父进程强制生效（:func:`pyqcd.einsum._tune_runner.run_group`）。

为什么是**新解释器**而不是 ``fork``
-----------------------------------
``fork`` 更便宜，但 CUDA 上下文不能跨 ``fork`` 复用：父进程只要碰过 CUDA
（``getDeviceCount()`` 或任意一次分配就够），子进程里第一次 ``cupy.asarray`` 就是
``cudaErrorInitializationError``（本机 RTX 4060 + cupy 14.2 实测；反之父进程从未
``import cupy`` 时 fork 出来的子进程能正常初始化 CUDA）。而父进程必须碰 CUDA——它要
建操作数、并从**设备对象**上读出 ``device='cupy:0'`` 这类键列。所以这里用
``subprocess`` 起全新解释器：每组付一次进程启动（本机实测 ``import pyqcd.einsum``
中位 **126 ms**、``import cupy`` 约 **244 ms**），换来"父进程的 CUDA 状态一律不参与"。

子进程**只做一件事**
--------------------
载入操作数 → 跑 ``warmup`` + ``rounds`` 次 ``contract()`` → 把观察结果（耗时样本、
结果形状/dtype、是否异常、最近一次选路记录）写进结果文件。**写配置表、打印表格、
选优一律在这里不做**：那是父进程的编排工作，而且父进程才是能杀死本进程的一方。

组内 ``min_wall_ms`` 剪枝仍在**这里**做：规则是"单次调用耗时超过当前上界即终止该组"，
必须能在两次调用之间生效，否则一组 7 次调用会先跑完才被父进程判死。命中时写
``status=pruned`` 与两个数值，父进程照原文 ``耗时大于min_wall_ms,此组被终止`` 打出。

通信方式（三个文件，都在父进程给的临时目录里）
----------------------------------------------
* ``job.json``：父进程 → 子进程的任务单（方程、dtype、backend、三轴取值、
  ``warmup``/``rounds``、``min_wall_ms``、操作数目录、结果路径、可选输出路径）。
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
from typing import Any, Sequence

__all__ = ["run_job", "main", "Pruned", "SLEEP_ENV", "EXIT_ENV"]

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


class Pruned(RuntimeError):
    """组内 ``min_wall_ms`` 剪枝：某一次调用比当前上界还慢，该组就此终止。

    继承 ``RuntimeError`` 是有意的：它必须穿透库里那些 ``except Exception`` 的
    "丢掉这次尝试、回退再跑一遍"，否则被终止的那一组会以通用路径重跑一遍，
    "终止"就成了空话。父进程据此记 ``status=pruned``（不是 ``timeout``：这一组
    并没有跑到时间上限，它只是比当前最快的那一组慢）。
    """

    def __init__(self, wall_ms: float, min_wall_ms: float | None) -> None:
        super().__init__(f"pruned at {float(wall_ms):.6f}ms")
        self.wall_ms = float(wall_ms)
        self.min_wall_ms = None if min_wall_ms is None else float(min_wall_ms)


def _load_operands(directory: str, device: str, backend: Any = None) -> list[Any]:
    """按文件名顺序读回父进程暂存的操作数，并**按 backend 转成该后端的数组**。

    落盘一律是 ``.npy``（父进程用 ``np.save``），所以读回来必然是 numpy 家族；能不能
    直接交给 ``contract()`` 取决于**后端**，不是设备：

    * ``numpy`` / ``auto`` / 空 → 原样返回；
    * ``torch`` → ``torch.from_numpy``（**不能**把 memmap 直接喂给 torch：
      ``torch.tensordot`` 会报 ``argument 'input' must be Tensor, not memmap``，
      2026-11-20 实测 tune 在 ``--backend torch`` 上 21 组全 ``error`` 正是这条）；
    * 其余（``cupy``）→ ``cupy.asarray``。

    ``mmap`` 只在 numpy 后端上开：它是"父进程与全部组共用同一份只读数据"的省钱手段，
    而 torch/cupy 需要一个**真**数组（``torch.from_numpy`` 接受 ndarray 子类但会为
    非可写缓冲发警告，而 memmap 恰好不可写；``cupy.asarray`` 更是要真实主机缓冲）。
    """
    import numpy as np

    paths = sorted(glob.glob(os.path.join(directory, "operand_*.npy")))
    device_text = str(device or "cpu").strip().lower()
    backend_text = str(backend or "").strip().lower()
    cpu = device_text == "cpu"
    numpy_like = backend_text in ("", "auto", "numpy")
    use_mmap = cpu and numpy_like
    host = [np.load(path, mmap_mode="r" if use_mmap else None) for path in paths]
    if numpy_like:
        return host
    if backend_text == "torch":
        import torch

        tensors = [torch.from_numpy(np.ascontiguousarray(item)) for item in host]
        if not cpu:
            tensors = [item.to("cuda") for item in tensors]
        return tensors
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


def _sample_minimum(samples: Sequence[float]) -> float:
    """Return the fastest observed call time."""
    return min(float(item) for item in samples)


def run_job(job: dict[str, Any], mark_ready) -> dict[str, Any]:
    """跑一组 ``warmup`` + ``rounds`` 次 ``contract()``，返回观察结果。

    ``mark_ready`` 在**第一次调用之前**被调用一次（父进程据此开始计 ``--tune-timeout``）。
    返回的字典就是写进 ``result.json`` 的内容；父进程只认这几个键：``minimum_s`` /
    ``samples`` / ``shape`` / ``dtype`` / ``last_kernel`` / ``kernel_info`` /
    ``saved_output`` / ``output_skipped_bytes``。

    ``min_wall_ms``（可选）是组内剪枝**截止**（父进程按 ``MIN_WALL_FACTOR`` 乘过
    当前最快最小值）：**每一次**调用测到的墙钟超过它就抛
    :class:`Pruned`，不再跑完这一组。用户规则说的就是这个，只是执行位置在子进程里
    ——父进程没有办法打断正在进行的那一次调用。
    """
    import numpy as np

    import pyqcd.einsum as pe
    from pyqcd.einsum import _api, _sync

    equation = str(job["equation"])
    device = str(job.get("device") or "cpu")
    backend = job.get("backend")
    warmup = max(int(job.get("warmup") or 0), 0)
    rounds = max(int(job.get("rounds") or 1), 1)
    raw_bound = job.get("min_wall_ms")
    min_wall_ms = None if raw_bound is None else float(raw_bound)
    # backend 必须传下去：操作数的**类型**由后端决定（torch 要 Tensor，不是 memmap）。
    operands = _load_operands(str(job["operands_dir"]), device,
                              job.get("backend"))
    kwargs = dict(job.get("contract_kwargs") or {})
    kwargs.update(_filter_profile_kwargs(dict(job.get("profile_kwargs") or {})))
    kwargs["backend"] = backend
    # ``None`` = 调用方没指定（一路传到 ``contract()``，由 ``_api`` 自己回落）。
    if job.get("kernel_force") is not None:
        kwargs["kernel_force"] = job["kernel_force"]
    kwargs["as_equivalence"] = job.get("as_equivalence")
    kwargs["optimize"] = job.get("optimize")
    if job.get("profile_off"):
        # 遍历**必须**量"被测配置本身"：读表会让命中行把 ``optimize`` / ``kernel``
        # 换掉，于是量到的不是这一格；更糟的是父进程按格显式下发的三轴会与命中行
        # 冲突，``_api`` 抛 ``ProfileConflict``，每一格都变成 ``error``。
        #
        # 进程内路线靠 ``_profile_cache.tune_session()`` 的 ``_TUNING`` 实现同一件事，
        # 但那是一个**进程内**计数器：子进程里它恒为 0，读表照旧发生（2026-11-19 用户
        # 实测：9 组 ProfileConflict）。所以这里显式关掉本次调用的画像读写。
        profile_name = "profile"
        try:
            import inspect

            if profile_name in inspect.signature(pe.contract).parameters:
                kwargs[profile_name] = False
        except (TypeError, ValueError):         # pragma: no cover - 装饰器/扩展
            kwargs.pop(profile_name, None)

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
    # ``wall_s`` 必须由**本进程**测：父进程只看得见 ``ready`` 文件什么时候出现，而轮询
    # 粒度是毫秒级——一组只要跑得比这个粗，父进程量出来的"从 ready 到现在"就可能是
    # 微秒级的假数（甚至比 ``minimum_s`` 还小）。这里从 mark_ready 之前起表，到样本
    # 全部跑完为止，是**受测时间区**的真实墙钟。
    # 峰值内存/显存：在**本组**第一次调用之前立基线（``reset_peak_memory``），
    # 组末读一次，然后 ``release_memory`` 把池/缓存还回去。逐组重置是必须的：
    # 累加的峰值会把上一组的占用算到本组头上，剪优与剪枝都跟着错。
    #
    # **不调 ``memory_probe_available()``**：那是唯一会 ``import torch`` 的入口
    # （本机 1.4 s）。子进程路线每组一个全新解释器，每组付一次 1.4 s 就是 29 × 1.4 s
    # 的纯浪费，且它测的全是"这台机器有什么"（与这一组的耗时无关）。
    # ``_memory`` 的采样用 ``_torch_if_ready`` / ``_cupy_if_ready``：只在模块已在
    # ``sys.modules`` 里时才取句柄——**不为测量而 import**。
    from pyqcd.einsum import _memory

    # ``exclude_mb``：这批操作数**本来就要占**的那部分。它们由父进程暂存、本进程用
    # mmap 读（CPU）或整份拷进显存（GPU），29 组看到的是同一份数据——不扣的话每组
    # 都报同一个大数（生产形状实测 ~3923 MB），那个数只回答"输入有多大"。
    # ``operands`` 的字节数就是这份常驻量最直接的量度。
    resident_mb = 0.0
    for operand in operands:
        try:
            resident_mb += float(operand.nbytes) / _memory.MB
        except (AttributeError, TypeError):           # pragma: no cover - 非数组
            continue
    _memory.reset_peak_memory(exclude_mb=resident_mb)
    ready_started = time.perf_counter()
    mark_ready()
    if crash:
        os._exit(int(crash))
    if sleep:
        # 模拟"进入原生代码后不返回"：就在计时区里睡，父进程只能靠 kill 打断。
        time.sleep(float(sleep))

    def measured() -> float:
        """一次受测调用 → 秒（**含设备同步**）；超过 ``min_wall_ms`` 抛 :class:`Pruned`。

        同步是硬要求：``cupy`` / ``torch(cuda)`` 异步下发，不同步就只量到入队时间
        （2026-11-19 用户实测：同一组 GPU 收缩脚本里约 900 ms、这里 0.32 ms，还被
        当成 ``best`` 写进了配置表）。CPU 上 ``_sync.synchronize`` 是空操作。
        """
        _sync.synchronize(*operands)
        started = time.perf_counter()
        got = pe.contract(equation, *operands, **kwargs)
        _sync.synchronize(*operands)
        wall_s = time.perf_counter() - started
        if min_wall_ms is not None and wall_s * 1000.0 > min_wall_ms:
            # ``min_wall_ms`` 在这里收到的是**截止**（父进程已按
            # ``_tune_axes.MIN_WALL_FACTOR`` 乘过），本进程不再乘第二遍。
            raise Pruned(wall_s * 1000.0, min_wall_ms)
        measured.result = got                         # type: ignore[attr-defined]
        return wall_s

    for _ in range(warmup):
        measured()
    samples: list[float] = []
    for _ in range(rounds):
        samples.append(measured())
    result = getattr(measured, "result", None)
    minimum_s = _sample_minimum(samples)

    summary: dict[str, Any] = {
        "status": "ok",
        "minimum_s": float(minimum_s),
        "samples": [float(item) for item in samples],
        "shape": list(np.shape(result)),
        "dtype": str(getattr(result, "dtype", "")),
        # 受测时间区的**真实**墙钟（本进程测，父进程的轮询粒度量不准这个量）。
        "wall_s": time.perf_counter() - ready_started,
        # 本组峰值（MB，1024² 字节）：组内测量、组末释放。``0.0`` 表示"没测到"。
        "peak_memory_mb": float(_memory.peak_memory_mb()),
        # 扣掉的那部分如实写回来：读者据此知道"这个峰值不含输入本身"。
        "peak_memory_excluded_mb": round(resident_mb, 1),
        "peak_memory_notes": dict(_memory.LAST_ERROR) if _memory.LAST_ERROR else {},
        "last_kernel": _api.last_kernel() if hasattr(_api, "last_kernel") else None,
        "kernel_info": _kernel_info(),
        "saved_output": False,
    }
    output = job.get("output_operand")
    if output:
        # 只有**小**输出才落盘：``rel`` 校验是可选的加分项，不该为它把磁盘写满。
        limit = int(job.get("output_max_bytes") or 0)
        host_result = _to_host(result)
        if limit > 0 and int(host_result.nbytes) > limit:
            summary["output_skipped_bytes"] = int(host_result.nbytes)
        else:
            try:
                np.save(str(output), host_result)
                summary["saved_output"] = True
            except OSError as exc:              # pragma: no cover - 磁盘满/只读
                summary["output_error"] = f"{type(exc).__name__}: {exc}"
    # 本组结束：把池/缓存还给设备。放在 ``return`` 之前而不是父进程里——显存属于
    # **本进程**，父进程清不到它。
    _memory.release_memory(*operands)
    summary["release_s"] = _memory.last_release_seconds()
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
                          ("path", "specialized", "steps", "dtype", "copied",
                           "budget_gate", "budget_gate_bypassed")
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
    except Pruned as exc:
        # 剪枝是**判定**不是意外：父进程照 ``pruned`` 记账并原样打出那两个数值。
        _write(result_path, {"status": "pruned",
                             "pruned_wall_ms": exc.wall_ms,
                             "pruned_min_wall_ms": exc.min_wall_ms})
        return 0
    except BaseException as exc:                # noqa: BLE001 - 回传给父进程记账
        _write(result_path, {"status": "error",
                             "error": f"{type(exc).__name__}: {exc}",
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
