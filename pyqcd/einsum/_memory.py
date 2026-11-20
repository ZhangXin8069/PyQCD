"""每组 ``--tune`` 的**峰值内存 / 显存**测量与释放。

用户的规则是"每一项测试组都要记录峰值内存或显存占用（MB），并且进下一组前释放上一组的
占用"——后者是重点：不释放就会把上一组残留算到本组头上，报告出来的数就是累积量而不是
本组量。本模块只提供三件事：**取样**（:func:`peak_memory_mb`）、**立基线**（
:func:`reset_peak_memory`）、**归还占用**（:func:`release_memory`），外加一个
:func:`memory_probe_available` 让调用方知道本机能测到哪几路。

单位口径（重要）
----------------
本模块所有 ``*_mb`` 都是 **MB = 1024² 字节（即 MiB）**，不是 1e6 字节的十进制 MB：
``/proc/self/status`` 的 kB 列、``resource.ru_maxrss``（Linux 为 kB）、cupy 的
``used_bytes()``、torch 的 ``*_memory_allocated()`` 都按 1024 进制换算，混用会差 4.9%。
用户要求的"单位 MB"按此口径满足；:data:`LAST_ERROR` 里记录每条原因时都会带
``MB = 1024² 字节`` 这句后缀（见 :data:`MB_NOTE`），以免下游误读。

三路（CPU / cupy / torch）为什么都要覆盖
----------------------------------------
``--tune`` 的 ``backend`` 轴是 ``numpy`` / ``cupy`` / ``torch`` 三选一：

* **CPU（numpy 与一切 host 侧张量）**：``resource.getrusage(RUSAGE_SELF).ru_maxrss``
  是**进程生命周期**的峰值，做不到"清零"。所以 :func:`reset_peak_memory` 记一个
  ``_BASELINE['maxrss']``，读数取 ``max(0, 当前峰值 - 基线)``。它的含义是
  "**自重置点起的峰值增量**"，并且**受生命周期历史峰值影响**：如果本进程早先
  （上一组、或 fixtures 载入数据）已经冲到过 3 GiB，那么重置后的基线就是 3 GiB，
  本组再涨到 1 GiB 也读不出增量——**这是下界估计，不是精确的组内峰值**。
  为了补上这一点，重置后起一个 20 ms 的采样线程轮询 ``/proc/self/status`` 的
  ``VmRSS``（瞬时值，**可以下降**），取其相对基线的最大增量，与 ``VmHWM`` 的增量
  一起取 ``max``。``VmHWM`` 与 ``ru_maxrss`` 是同一个量（High Water Mark），两路都读
  是为了非 Linux 平台没有 ``/proc`` 时不至于全瞎。
* **cupy**：``cupy.cuda.Device().mem_info`` 给 ``(free, total)`` 瞬时值，峰值用
  ``total - min(free)`` 的抽样峰值；同时每秒多次读 ``get_default_memory_pool()
  .used_bytes()``（纯 Python 计数，无 CUDA 调用）。注意 ``used_bytes()`` 只算"正在
  使用"的块——数组一 ``del`` 就回收到池里，``used_bytes()`` 归零而 VRAM **没有**
  归还给驱动，那部分体现在 ``total_bytes()``（池的预留量）。所以
  :func:`release_memory` 调的是 ``free_all_blocks()``，它才真正把块还给驱动。
* **torch**：``torch.cuda.max_memory_allocated()`` 是唯一**真能清零**的峰值计数器
  （``reset_peak_memory_stats()``）。``import torch`` 实测约 0.85 s，因此**严格惰性**：
  只在 :func:`memory_probe_available` 里尝试一次并缓存结果（失败也缓存），
  :func:`peak_memory_mb` / :func:`reset_peak_memory` **不会**为了测量而 import torch。

调用顺序（缺了第一步就会漏掉 torch 那一路）
-------------------------------------------
::

    available = memory_probe_available()   # 一次，昂贵，把 cupy/torch 探测并缓存
    reset_peak_memory()                    # 每组开始前：立基线 + 起采样线程
    ...                                     # 跑这一组
    peak_memory_mb() / format_peak_mb(...) # 这一组的峰值（维护期间可多次读）
    release_memory()                       # 每组结束后：停采样 + 清池 + gc

``memory_probe_available()`` 之所以要在第一次 ``reset_peak_memory()`` **之前**调用：
本机上 ``torch.cuda.is_available()``/首次 CUDA 触达会建主上下文，实测多吃约 100-200 MiB
显存（见下条），这笔开销不该记到第一组的账上。

为什么不为了"测一下显存"就建 CUDA 上下文
----------------------------------------
本机实测（RTX 4060 Laptop + cupy 14.2）：进程只要还没有 CUDA 主上下文，
``cupy.cuda.Device(0).mem_info`` 的**第一次**调用就会建上下文，
``nvidia-smi`` 的 ``memory.used`` 从 995 MiB 跳到 1107 MiB（**+112 MiB**）。
那笔开销如果发生在 reset 之后，就会被当成本组的占用——正好违背本模块的目的。
所以设备级采样前先用驱动 API ``cuDevicePrimaryCtxGetState``（0.35 µs/次，
**只读、不建上下文**）问一句"已经有主上下文了吗"，没有就不采样
（此时本组的显存在任何一路眼里都还是 0，答案本身正确）。

同样地，:func:`memory_probe_available` 判定 torch 可用性用
``torch.cuda.device_count() > 0`` 而不是 ``torch.cuda.is_available()``：后者实测
从 806 MiB 走到 1000 MiB（+194 MiB，建上下文），前者只 +4 MiB。

采样线程会不会污染计时
----------------------
``--tune`` 同时量耗时，所以采样的代价要写在明面上：采样线程是 daemon，每
``PYQCD_EINSUM_MEMORY_SAMPLE_MS``（缺省 20 ms）醒一次，读 ``/proc/self/status``
约 30-60 µs（``open`` 期间释放 GIL，约占有空比的 0.3%）；设备级 ``memGetInfo``
（约 40 µs）默认每 5 个 tick 一次（即 ~100 ms），它**可能**引入一次驱动级同步，
对亚毫秒 GPU 调用是可见的噪声。两条都可以关：``PYQCD_EINSUM_MEMORY_SAMPLE_MS=0``
关掉整个采样线程、``PYQCD_EINSUM_MEMORY_DEVICE_PROBE=0`` 只关设备级那一路
（此时 cupy 池与 torch 计数这两条**无 CUDA 调用**的 host 侧计数照常工作）。

绝不抛异常，也绝不静默
---------------------
五个公开函数（``peak_memory_mb`` / ``reset_peak_memory`` / ``release_memory`` /
``memory_probe_available`` / ``format_peak_mb``，外加只读的
``peak_memory_breakdown``）都在 ``--tune`` 的记账路径上，测量失败不该让一组配置崩掉：
每一个入口都自带 ``try/except``，失败时返回 0.0 / 空操作，同时把
``"<来源>: <异常类型>: <异常信息> [MB = 1024² 字节]"`` 写进模块级 :data:`LAST_ERROR`
（按来源去重，``reset_peak_memory()`` 每组清一次，好让报告对着某一组解释）。

"没测到"与"测挂了"分两个账本：

* :data:`LAST_ERROR` —— 真的出错了（读 ``/proc`` 抛异常、``cuInit`` 失败、
  ``torch`` import 失败……）。**正常机器上应该是空 dict**。
* :data:`LAST_NOTE` —— 已知的、非异常的降级（本机没有 ``libcuda.so.1``、
  ``cupy.is_available()`` 为 False、``SAMPLE_MS_ENV=0`` 关掉了采样线程……）。
  把这类情况写进 ``LAST_ERROR`` 会让每台无 GPU 的机器每组都"报错"，
  真正的问题反而被淹掉。

探测期（:func:`memory_probe_available`）产生的两本账会另存一份，之后每组
:func:`reset_peak_memory` 清账时再放回去——"这台机器缺什么"是机器属性，不该被
某一组的清零动作抹掉。
"""
from __future__ import annotations

import gc
import os
import resource
import sys
import threading
import time
from typing import Any

__all__ = [
    "LAST_ERROR",
    "LAST_NOTE",
    "MB",
    "MB_NOTE",
    "format_peak_mb",
    "last_release_seconds",
    "memory_probe_available",
    "peak_memory_breakdown",
    "peak_memory_mb",
    "release_memory",
    "reset_peak_memory",
]

#: 本模块 MB 的口径：**1024² 字节（MiB）**。所有 ``*_mb`` 换算都用它。
MB = 1024 * 1024

#: 写进 :data:`LAST_ERROR` 每条原因尾巴上的单位说明（用户要求"单位 MB"，
#: 这里如实写明是 1024² 而不是 1e6，免得下游把 1 GiB 读成 1.07 GB）。
MB_NOTE = "MB = 1024² 字节（MiB）, 非 1e6 字节"

#: 失败原因账本：``{来源: "异常类型: 信息 [MB = 1024² 字节]"}``。
#: ``reset_peak_memory()`` 每组清零；**没有任何失败时是空 dict**。
LAST_ERROR: dict[str, str] = {}

#: 非失败的状态说明（例如"CUDA 尚未初始化，显存计数按 0 记"），同样按来源去重、
#: 每组清零。与 :data:`LAST_ERROR` 分开，避免把"没测到"说成"测挂了"。
LAST_NOTE: dict[str, str] = {}

#: **探测期**的失败/说明（:func:`memory_probe_available` 那一次 import/查设备的
#: 结果）。它描述的是"这台机器有什么"，与"这一组出了什么事"无关，所以
#: :func:`reset_peak_memory` 清账之后会把这几条**重新放回**
#: :data:`LAST_ERROR` / :data:`LAST_NOTE`——否则"torch 装坏了"这种事实会被第一组的
#: 清零动作抹掉，报告里再也看不到。
_PROBE_ERROR: dict[str, str] = {}
_PROBE_NOTE: dict[str, str] = {}

#: 采样线程的缺省周期（秒）= 20 ms；环境变量 ``PYQCD_EINSUM_MEMORY_SAMPLE_MS``
#: 可覆盖（``0`` = 不起线程，只靠同步取样）。
DEFAULT_SAMPLE_MS = 20.0
SAMPLE_MS_ENV = "PYQCD_EINSUM_MEMORY_SAMPLE_MS"

#: 设备级（``memGetInfo``）采样开关；``0`` 关闭，见模块文档"采样线程会不会污染计时"。
DEVICE_PROBE_ENV = "PYQCD_EINSUM_MEMORY_DEVICE_PROBE"

#: 设备级采样每隔几个 tick 做一次（20 ms × 5 = 100 ms）。
_DEVICE_TICKS = 5

#: 惰性 import 的缓存哨兵：``_MISSING`` = 还没试过，``None`` = 试过且不可用。
_MISSING = object
_cupy_cache: Any = _MISSING
_torch_cache: Any = _MISSING
_cuda_lib_cache: Any = _MISSING

#: 本组已经记过的来源（去重，避免 20 ms 一条把 ``LAST_ERROR`` 刷成日志）。
_RECORDED: set[str] = set()

#: 全局状态锁：采样线程与主线程都会碰 :data:`_PEAK` / :data:`_BASELINE`。
_LOCK = threading.RLock()

#: 重置点那一刻的**绝对占用**（MB），之后的读数一律减它。
#: 为什么用绝对基线而不是"清零计数器"：``ru_maxrss`` / ``VmHWM`` 降不下来。
_BASELINE: dict[str, float] = {
    "maxrss": 0.0,   # ru_maxrss（Linux kB / macOS 字节 → MB）
    "hwm": 0.0,      # /proc/self/status 的 VmHWM（与 ru_maxrss 同量）
    "rss": 0.0,      # /proc/self/status 的 VmRSS（瞬时，可下降）
    "cupy_pool": 0.0,   # cupy 池 used_bytes
    "device": 0.0,      # 设备已用显存 total - free
}

#: 被排除在峰值之外的**已知常驻占用**（MB）。由 :func:`reset_peak_memory` 的
#: ``exclude_mb`` 写入：调用方知道"这批操作数本来就要占 X MB"，那是**输入**不是
#: "这一组配置的开销"，混进峰值之后所有组的数字都一样大，比较就失去意义。
#:
#: 为什么必须能扣：生产形状上操作数 1953 MiB，tune 的 29 组每组都要触碰同一份 mmap
#: 页（CPU）或持有同一份显存副本（GPU），于是每组都报 ~3923 MB——那个数回答的是
#: "输入有多大"，不是"这一组额外吃了多少"（2026-11-20 用户实测发现这条口径不可用）。
_EXCLUDE_MB = 0.0

#: 自重置点以来观察到的**增量**峰值（MB，全部 >= 0）。
_PEAK: dict[str, float] = {
    "rss": 0.0,       # VmRSS 抽样峰值增量（这一路不受 ru_maxrss 历史峰值影响）
    "cupy": 0.0,      # cupy 池 used_bytes 增量峰值
    "device": 0.0,    # 设备已用显存增量峰值（含同卡其它进程的增长，故是上界）
    "torch": 0.0,     # torch max_memory_allocated（绝对值；它自己能清零）
}

#: 当前采样线程（``None`` = 没在采）。
_SAMPLER: "_Sampler | None" = None

#: 最近一次 ``release_memory`` 的耗时（秒），仅供报告参考。
_LAST_RELEASE_S = 0.0


# ---------------------------------------------------------------------------
# 记账
# ---------------------------------------------------------------------------
#: 正在做"这台机器有什么"的探测时为 True：此期间的记录另外抄一份进
#: :data:`_PROBE_ERROR` / :data:`_PROBE_NOTE`，好在每组清零之后恢复。
_IN_PROBE = False


def _record(source: str, exc: BaseException) -> None:
    """把一次失败写进 :data:`LAST_ERROR`（首次出现即记，之后去重）。"""
    key = str(source)
    if key in _RECORDED:
        return
    _RECORDED.add(key)
    message = f"{type(exc).__name__}: {exc} [{MB_NOTE}]"
    LAST_ERROR[key] = message
    if _IN_PROBE:
        _PROBE_ERROR[key] = message


def _note(source: str, message: str) -> None:
    """把一条非失败状态写进 :data:`LAST_NOTE`（按来源去重）。"""
    key = "note:" + str(source)
    if key in _RECORDED:
        return
    _RECORDED.add(key)
    text = f"{message} [{MB_NOTE}]"
    LAST_NOTE[str(source)] = text
    if _IN_PROBE:
        _PROBE_NOTE[str(source)] = text


# ---------------------------------------------------------------------------
# 惰性模块句柄
# ---------------------------------------------------------------------------
def _cupy_module() -> Any:
    """cupy 模块；不可用（未安装 / 无可见设备）返回 ``None``。

    只在这里 import：cupy 的 import 实测约 50 ms / +4 MiB 显存（``cuInit`` 不算
    建上下文），比 torch 便宜得多，但仍按需进行并缓存结果。
    """
    global _cupy_cache
    if _cupy_cache is _MISSING:
        try:
            import cupy  # noqa: PLC0415 - 惰性导入，见模块文档
            _cupy_cache = cupy if cupy.is_available() else None
            if _cupy_cache is None:
                _note("cupy", "cupy 可导入但看不到可用 GPU（cupy.is_available() 为 False）")
        except Exception as exc:                     # noqa: BLE001 - 记账后继续
            _record("cupy", exc)
            _cupy_cache = None
    return _cupy_cache


def _torch_module() -> Any:
    """torch 模块；不可用返回 ``None``。

    ``import torch`` 本机实测约 0.85 s——**只在** :func:`memory_probe_available`
    里尝试一次，结果（含失败）缓存在模块级，后续调用零成本。
    """
    global _torch_cache
    if _torch_cache is _MISSING:
        try:
            import torch  # noqa: PLC0415 - 惰性导入，见模块文档
            _torch_cache = torch
        except Exception as exc:                     # noqa: BLE001 - 记账后继续
            _record("torch", exc)
            _torch_cache = None
    return _torch_cache


def _cupy_if_ready() -> Any:
    """cupy 句柄，但**不为此 import**：只有探过或已在 ``sys.modules`` 里才返回。

    这条规矩让 ``peak_memory_mb()`` 能安心待在计时路径旁边——它不会突然花 50 ms
    去 import 一个模块（那会污染这一组的耗时）。
    """
    if _cupy_cache is _MISSING and "cupy" not in sys.modules:
        return None
    return _cupy_module()


def _torch_if_ready() -> Any:
    """torch 句柄，但**不为此 import**（理由同 :func:`_cupy_if_ready`）。"""
    if _torch_cache is _MISSING and "torch" not in sys.modules:
        return None
    return _torch_module()


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
def _maxrss_mb() -> float:
    """``ru_maxrss`` → MB。

    单位依据：Linux 的 ``getrusage(2)`` 明确写 ``ru_maxrss`` 以 **kilobytes** 计；
    macOS/BSD（``sys.platform == 'darwin'``）以 **bytes** 计。所以按平台分支，别用
    一个常数糊过去——在 macOS 上按 kB 读会把 100 MiB 报成 100 GB 量级的假数。
    """
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / MB
    return raw / 1024.0


def _read_status_kb(field: str) -> float | None:
    """读 ``/proc/self/status`` 里某一行的第一个数（单位 kB）；没有该行返回 None。"""
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return float(line.split()[1])
    return None


def _vm_kb(field: str) -> float | None:
    """``VmHWM`` / ``VmRSS``（kB）；非 Linux 无 ``/proc`` 时返回 ``None`` 并记账。"""
    try:
        value = _read_status_kb(field)
    except Exception as exc:                          # noqa: BLE001
        _record("cpu_" + field.lower(), exc)
        return None
    if value is None:
        _note("cpu_" + field.lower(), f"/proc/self/status 里没有 {field} 行，该路不参与取峰值")
    return value


def _cpu_baseline() -> dict[str, float]:
    """取 CPU 三路的重置基线（失败的那一路按 0 记，并进 LAST_ERROR/LAST_NOTE）。"""
    out = {"maxrss": 0.0, "hwm": 0.0, "rss": 0.0}
    try:
        out["maxrss"] = _maxrss_mb()
    except Exception as exc:                          # noqa: BLE001
        _record("cpu_maxrss", exc)
    for field, key in (("VmHWM", "hwm"), ("VmRSS", "rss")):
        kb = _vm_kb(field)
        if kb is not None:
            out[key] = kb / 1024.0
    return out


# ---------------------------------------------------------------------------
# CUDA 驱动：只读的"有没有主上下文"探针
# ---------------------------------------------------------------------------
def _cuda_driver() -> Any:
    """``libcuda.so.1`` 句柄（已 ``cuInit``）；拿不到返回 ``None``。

    只用它的 ``cuDevicePrimaryCtxGetState``：0.35 µs/次，**只读**，不建上下文。
    ``cuInit(0)`` 只做驱动初始化（本机实测 +4 MiB，不建上下文），是调用
    ``cuDevicePrimaryCtxGetState`` 的前提。

    两种"拿不到"分开记：**没有 ``libcuda.so.1``**（纯 CPU 机器，属已知情况）进
    :data:`LAST_NOTE`——否则每台无 GPU 的机器每组都会报一条"错误"，把报告噪音化；
    ``cuInit`` 返回非零（装了驱动却初始化失败，真异常）进 :data:`LAST_ERROR`。
    两条都不静默，只是分到不同的账本里。
    """
    global _cuda_lib_cache
    if _cuda_lib_cache is _MISSING:
        try:
            import ctypes  # noqa: PLC0415 - 只有设备级采样才需要
            try:
                lib = ctypes.CDLL("libcuda.so.1")
            except OSError as exc:
                _note("cuda_driver",
                      f"没有 libcuda.so.1（{type(exc).__name__}: {exc}）："
                      "设备级显存采样关闭，CPU 内存与 cupy/torch 池计数不受影响")
                _cuda_lib_cache = None
                return _cuda_lib_cache
            rc = lib.cuInit(0)
            if rc != 0:
                raise RuntimeError(f"cuInit(0) 返回 {rc}")
            lib.cuDevicePrimaryCtxGetState.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
            lib.cuDevicePrimaryCtxGetState.restype = ctypes.c_int
            _cuda_lib_cache = lib
        except Exception as exc:                      # noqa: BLE001 - 记账后关掉该路
            _record("cuda_driver", exc)
            _cuda_lib_cache = None
    return _cuda_lib_cache


def _cuda_context_active() -> bool:
    """device 0 上是否已有 CUDA 主上下文（已有才允许做设备级采样）。"""
    lib = _cuda_driver()
    if lib is None:
        return False
    try:
        import ctypes  # noqa: PLC0415
        flags = ctypes.c_uint(0)
        active = ctypes.c_int(0)
        rc = lib.cuDevicePrimaryCtxGetState(
            ctypes.c_int(0), ctypes.byref(flags), ctypes.byref(active))
        if rc != 0:
            raise RuntimeError(f"cuDevicePrimaryCtxGetState 返回 {rc}")
        return bool(active.value)
    except Exception as exc:                          # noqa: BLE001
        _record("cuda_context", exc)
        return False


def _device_used_mb() -> float:
    """设备已用显存（``total - free``，MB）。调用前必须确认上下文已存在。"""
    cupy = _cupy_if_ready()
    if cupy is not None:
        free, total = cupy.cuda.runtime.memGetInfo()
        return (float(total) - float(free)) / MB
    torch = _torch_if_ready()
    if torch is not None and torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        return (float(total) - float(free)) / MB
    raise RuntimeError("没有可用的 CUDA 读接口（cupy 与 torch 都未就绪）")


# ---------------------------------------------------------------------------
# 取样
# ---------------------------------------------------------------------------
def _sample_cpu_locked() -> None:
    """同步取一次 CPU 瞬时 RSS，更新"自重置点起"的抽样峰值增量。"""
    kb = _vm_kb("VmRSS")
    if kb is None:
        return
    delta = kb / 1024.0 - _BASELINE["rss"]
    if delta > _PEAK["rss"]:
        _PEAK["rss"] = delta


def _sample_gpu_locked(device: bool) -> None:
    """取一次 GPU 三路（cupy 池 / torch 计数 / 设备总量）里的可用项。"""
    cupy = _cupy_if_ready()
    if cupy is not None:
        try:
            used = float(cupy.get_default_memory_pool().used_bytes()) / MB
            delta = used - _BASELINE["cupy_pool"]
            if delta > _PEAK["cupy"]:
                _PEAK["cupy"] = delta
        except Exception as exc:                      # noqa: BLE001
            _record("cupy_pool", exc)
    torch = _torch_if_ready()
    if torch is not None:
        try:
            # 判据只用 is_initialized()：``torch.cuda.is_available()`` 会建主上下文
            # （本机实测 +194 MiB），测量代码绝不能因为"看一眼"就改设备状态。
            # 未初始化时 torch 显存**必然**是 0（本进程还没在 torch 上分配过），
            # 读 0 就是正确答案，不必每轮记一条说明——那会把报告刷成噪音。
            # "torch 有没有 GPU"这件事由 memory_probe_available() 一次性记在账上。
            if torch.cuda.is_initialized():
                # 这一路是**绝对值**：reset_peak_memory_stats() 真能清零，
                # 所以不需要（也不该）再减基线。
                peak = float(torch.cuda.max_memory_allocated()) / MB
                if peak > _PEAK["torch"]:
                    _PEAK["torch"] = peak
        except Exception as exc:                      # noqa: BLE001
            _record("torch_peak", exc)
    if not device:
        return
    if not _cuda_context_active():
        # 没有上下文 = 还没有任何 CUDA 分配。**不为测量去建上下文**（会凭空 +112 MiB，
        # 见模块文档），直接按 0 处理。
        return
    try:
        used = _device_used_mb()
        delta = used - _BASELINE["device"]
        if delta > _PEAK["device"]:
            _PEAK["device"] = delta
    except Exception as exc:                          # noqa: BLE001
        _record("cuda_device", exc)


def _take_sample(device: bool = True) -> None:
    """同步取一次样本（供采样线程与 ``peak_memory_mb`` 共用）。"""
    with _LOCK:
        _sample_cpu_locked()
        _sample_gpu_locked(device)


class _Sampler(threading.Thread):
    """20 ms 轮询的采样线程：daemon，绝不抛异常，可被 :meth:`stop` 立即停掉。

    为什么需要它：``ru_maxrss``/``VmHWM`` 只记"这辈子最高过多少"，量不出"本组涨了
    多少"；``VmRSS`` 会随释放下降，只有**持续观察**才能抓住两次观察之间的峰值
    （一组里操作数建好、临时缓冲、结果三者叠加的那个瞬间，单点取样经常错过）。
    """

    def __init__(self, interval_s: float, device_probe: bool) -> None:
        super().__init__(name="pyqcd-peak-memory", daemon=True)
        self._interval_s = max(1e-3, float(interval_s))
        self._device_probe = bool(device_probe)
        self._stop_event = threading.Event()
        self.ticks = 0
        self.errors = 0

    @property
    def interval_s(self) -> float:
        return self._interval_s

    def run(self) -> None:                            # pragma: no cover - 线程体
        while not self._stop_event.wait(self._interval_s):
            try:
                self.ticks += 1
                device = self._device_probe and (self.ticks % _DEVICE_TICKS == 0)
                _take_sample(device=device)
            except Exception as exc:                  # noqa: BLE001 - 线程不能死
                self.errors += 1
                _record("sampler", exc)

    def stop(self, timeout: float = 1.0) -> None:
        """请求停止并等待线程退出（在同一个线程里调用时只置标志，不 join）。"""
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)


def _sample_interval_s() -> float:
    """采样周期（秒）：环境变量 ``PYQCD_EINSUM_MEMORY_SAMPLE_MS``，``0`` 关线程。"""
    raw = os.environ.get(SAMPLE_MS_ENV)
    if raw is None or raw == "":
        return DEFAULT_SAMPLE_MS / 1000.0
    try:
        return max(0.0, float(raw)) / 1000.0
    except Exception as exc:                          # noqa: BLE001
        _record("env_sample_ms", exc)
        return DEFAULT_SAMPLE_MS / 1000.0


def _device_probe_enabled() -> bool:
    """设备级（``memGetInfo``）采样是否开启；缺省开，``=0`` 关。"""
    raw = os.environ.get(DEVICE_PROBE_ENV)
    return not (raw is not None and raw.strip() in ("0", "off", "false", "False"))


def _start_sampler() -> None:
    """起采样线程（周期 <= 0 时不起，只靠同步取样）。"""
    global _SAMPLER
    interval = _sample_interval_s()
    if interval <= 0.0:
        _note("sampler", f"{SAMPLE_MS_ENV}=0：不启采样线程，只做同步取样")
        return
    try:
        sampler = _Sampler(interval, _device_probe_enabled())
        sampler.start()
        _SAMPLER = sampler
    except Exception as exc:                          # noqa: BLE001
        _SAMPLER = None
        _record("sampler_start", exc)


def _stop_sampler() -> None:
    """停采样线程（幂等）。"""
    global _SAMPLER
    sampler, _SAMPLER = _SAMPLER, None
    if sampler is None:
        return
    try:
        sampler.stop()
    except Exception as exc:                          # noqa: BLE001
        _record("sampler_stop", exc)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def memory_probe_available() -> dict[str, bool]:
    """各来源是否可用：``{"cpu": True, "cupy": bool, "torch": bool}``。

    这是**唯一**会为了探测而 import cupy/torch 的入口（torch 约 0.85 s），结果
    （包括"不可用"和 import 失败）都缓存在模块级，重复调用零成本。

    判据与代价（本机实测）：

    * ``cpu``：``resource.getrusage`` 能调通即 True（POSIX 必有；不成立说明环境异常）。
    * ``cupy``：``cupy.is_available()``（+4 MiB 显存，不建上下文）。
    * ``torch``：``torch.cuda.device_count() > 0``。**故意不用
      ``torch.cuda.is_available()``**：后者实测 +194 MiB（建主上下文），而本函数
      通常在第一组之前调用，那笔开销会污染基线。

    值一定是 ``bool``（不是 ``numpy.bool_`` 之类），键恰好是这三个。探测过程中产生的
    失败/说明会另外存一份进 :data:`_PROBE_ERROR` / :data:`_PROBE_NOTE`，使得之后
    每组 :func:`reset_peak_memory` 清账时仍能把"这台机器缺什么"带回来。
    """
    global _IN_PROBE
    result = {"cpu": True, "cupy": False, "torch": False}
    _IN_PROBE = True
    try:
        try:
            resource.getrusage(resource.RUSAGE_SELF)
        except Exception as exc:                      # noqa: BLE001
            _record("cpu", exc)
            result["cpu"] = False
        result["cupy"] = _cupy_module() is not None
        torch = _torch_module()
        if torch is None:
            result["torch"] = False
        else:
            try:
                result["torch"] = bool(torch.cuda.device_count() > 0)
            except Exception as exc:                  # noqa: BLE001
                _record("torch_device_count", exc)
                result["torch"] = False
            if not result["torch"]:
                _note("torch", "torch.cuda.device_count() == 0：无可见 CUDA 设备")
    finally:
        _IN_PROBE = False
    return {key: bool(value) for key, value in result.items()}


def reset_peak_memory(exclude_mb: float = 0.0) -> None:
    """把峰值计数器清零：立新基线、清错误账、起采样线程。**绝不抛异常**。

    ``exclude_mb`` = **已知常驻占用**（MB），从各路读数里扣掉。它回答的是
    "这批操作数本来就要占多少"——那是**输入**，不是"这一组配置的开销"。
    生产形状上操作数 1953 MiB、每组都要触碰同一份 mmap 页，不扣的话 29 组全报
    ~3923 MB，比较"哪组更省内存"就失去意义（2026-11-20 用户实测）。

    真正能"清零"的只有一路（torch 的 ``reset_peak_memory_stats()``）；其余各路是
    **记基线 + 之后只看增量**：

    * ``ru_maxrss`` / ``VmHWM`` 是生命周期峰值，降不下来——基线就是"重置那一刻的
      历史最高"，所以读数含义是"自重置点起的增量，**受历史峰值影响，是下界估计**"；
    * ``VmRSS`` 可下降，重置后由采样线程持续取瞬时值的最大增量；
    * cupy 池 ``used_bytes`` 与设备已用显存同样按增量算。

    幂等：上一组的采样线程会先被停掉（daemon，且 join 有 1 s 上限）。错误账
    :data:`LAST_ERROR` / :data:`LAST_NOTE` 在这里清空，一组一份——但
    :func:`memory_probe_available` 探到的"这台机器缺什么"（:data:`_PROBE_ERROR` /
    :data:`_PROBE_NOTE`）会重新放回，否则清账动作会把"torch 装坏了"这种事实抹掉。
    """
    global _LAST_RELEASE_S, _EXCLUDE_MB
    try:
        _EXCLUDE_MB = max(0.0, float(exclude_mb or 0.0))
    except (TypeError, ValueError):                   # noqa: BLE001 - 测量不许抛
        _EXCLUDE_MB = 0.0
    try:
        _stop_sampler()
    except Exception as exc:                          # noqa: BLE001
        _record("sampler_stop", exc)
    with _LOCK:
        LAST_ERROR.clear()
        LAST_NOTE.clear()
        _RECORDED.clear()
        LAST_ERROR.update(_PROBE_ERROR)
        LAST_NOTE.update(_PROBE_NOTE)
        for key in _PROBE_ERROR:
            _RECORDED.add(key)
        for key in _PROBE_NOTE:
            _RECORDED.add("note:" + key)
        # 设备已用显存的基线：只有上下文已存在时才取，避免为取基线建上下文。
        device_baseline = 0.0
        if _cuda_context_active():
            try:
                device_baseline = _device_used_mb()
            except Exception as exc:                  # noqa: BLE001
                _record("cuda_device", exc)
        _BASELINE.update(_cpu_baseline())
        _BASELINE["cupy_pool"] = 0.0
        try:
            cupy = _cupy_if_ready()
            if cupy is not None:
                _BASELINE["cupy_pool"] = float(
                    cupy.get_default_memory_pool().used_bytes()) / MB
        except Exception as exc:                      # noqa: BLE001
            _record("cupy_pool", exc)
        _BASELINE["device"] = device_baseline
        for key in _PEAK:
            _PEAK[key] = 0.0
        # torch 是唯一能真清零的：只在 CUDA 已初始化时才调，否则会为了"重置"
        # 去建上下文（实测 +194 MiB 级别，正好是我们要避免的污染）。未初始化时
        # torch 的显存计数必然是 0，"不重置"与"重置"读数相同，所以这不算降级、
        # 也不进 LAST_NOTE——否则每一组都会多一条无信息量的说明。
        torch = _torch_if_ready()
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.reset_peak_memory_stats()
            except Exception as exc:                  # noqa: BLE001
                _record("torch_reset", exc)
        _LAST_RELEASE_S = 0.0
    _start_sampler()


def peak_memory_mb() -> float:
    """当前进程/设备**自重置点以来**的峰值占用（MB，1024² 字节），取不到返回 ``0.0``。

    口径（务必照实理解）：

    * CPU 这一路是 ``max(0, ru_maxrss - 基线, VmHWM - 基线, VmRSS抽样峰值 - 基线)``。
      ``ru_maxrss``/``VmHWM`` 是**进程生命周期**的峰值、**无法清零**，所以这个值
      对第一组之后的组是"自重置点起的峰值增量，受生命周期历史峰值影响，是**下界
      估计**"，不是精确的组内峰值；``VmRSS`` 抽样那一路才是能反映"本组涨了多少"的。
    * GPU 这一路取三者的最大值：cupy 池增量、torch ``max_memory_allocated()``（绝对值，
      因为它自己的计数器能清零）、以及**存在 CUDA 上下文时**设备已用显存的增量
      （这一路含同卡其它进程的增长，因此是上界；上下文不存在时不采样，见模块文档）。
    * 最终返回 CPU 与 GPU 的**较大者**，即"这一组的峰值占用"。要分量看
      :func:`peak_memory_breakdown`。

    调用本身会同步取一次样（约 40-90 µs，含一次 ``/proc`` 读），所以"分配完立刻读"
    就能看到刚发生的增长，不必等采样线程。**绝不抛异常**；失败原因在
    :data:`LAST_ERROR`。
    """
    try:
        _take_sample(device=True)
    except Exception as exc:                          # noqa: BLE001
        _record("sample", exc)
    try:
        with _LOCK:
            return float(max(_cpu_peak_locked(), _gpu_peak_locked()))
    except Exception as exc:                          # noqa: BLE001
        _record("peak", exc)
        return 0.0


def peak_memory_breakdown() -> dict[str, float]:
    """分路峰值（MB）：``cpu`` / ``cupy`` / ``torch`` / ``device`` / ``total``。

    报告里"这组吃了 1234MB"用 ``total``；要解释它是显存还是内存看其余四项。
    ``device`` 与 ``cupy``/``torch`` 会重叠（设备总量包含池里那块），不要相加，
    取它们是同一件事的不同视角。**绝不抛异常**（失败时各项为 0.0）。
    """
    try:
        _take_sample(device=True)
    except Exception as exc:                          # noqa: BLE001
        _record("sample", exc)
    try:
        with _LOCK:
            out = {
                "cpu": _cpu_peak_locked(),
                "cupy": max(0.0, float(_PEAK["cupy"]) - _EXCLUDE_MB),
                "torch": max(0.0, float(_PEAK["torch"]) - _EXCLUDE_MB),
                "device": max(0.0, float(_PEAK["device"]) - _EXCLUDE_MB),
            }
            out["total"] = float(max(out["cpu"], out["cupy"], out["torch"], out["device"]))
            return out
    except Exception as exc:                          # noqa: BLE001
        _record("peak", exc)
        return {"cpu": 0.0, "cupy": 0.0, "torch": 0.0, "device": 0.0, "total": 0.0}


def _cpu_peak_locked() -> float:
    """CPU 峰值（MB）：HWM 增量、ru_maxrss 增量、VmRSS 抽样峰值三者取最大。

    三路各自 ``try``：任何一路挂掉只丢那一路（并进 :data:`LAST_ERROR`），
    不影响其余两路——这也是"测量失败不该让 tune 崩"的具体落点。
    """
    candidates = [0.0, float(_PEAK["rss"])]
    try:
        maxrss = _maxrss_mb()
        candidates.append(max(0.0, maxrss - _BASELINE["maxrss"]))
    except Exception as exc:                          # noqa: BLE001
        _record("cpu_maxrss", exc)
    try:
        hwm_kb = _vm_kb("VmHWM")
        if hwm_kb is not None:
            candidates.append(max(0.0, hwm_kb / 1024.0 - _BASELINE["hwm"]))
    except Exception as exc:                          # noqa: BLE001
        _record("cpu_hwm", exc)
    return max(0.0, max(candidates) - _EXCLUDE_MB)


def _gpu_peak_locked() -> float:
    """GPU 峰值（MB）：cupy 池 / torch 计数 / 设备总量三者取最大，再扣掉已知常驻。

    ``torch`` 那一路是**绝对值**（它的计数器能清零），所以它自带"这一组额外吃了
    多少"的语义；扣掉 ``_EXCLUDE_MB`` 只影响"操作数本来就在显存里"的那部分，
    让三路口径一致。
    """
    try:
        peak = max(float(_PEAK["cupy"]), float(_PEAK["torch"]),
                   float(_PEAK["device"]))
        return max(0.0, peak - _EXCLUDE_MB)
    except Exception as exc:                          # noqa: BLE001
        _record("gpu_peak", exc)
        return 0.0


def release_memory(*operands: Any) -> None:
    """释放本组占用：停采样、清 cupy 池、清 torch 缓存、对 CPU 侧做一次 gc。

    参数 ``operands`` 只是 API 对称性（调用方可以把本组的操作数一起传进来，
    让人一眼看出"这一组释放的是这些"）；本函数**不持有**它们，也不改调用方的引用
    ——真正的释放靠调用方 ``del`` + 本函数的池/缓存清理。返回 ``None``；
    要报"释放花了多久"读 :func:`last_release_seconds`。

    做的四件事：

    1. 停采样线程（daemon，``join`` 上限 1 s）；
    2. ``cupy.get_default_memory_pool().free_all_blocks()`` 与
       ``cupy.get_default_pinned_memory_pool().free_all_blocks()``：``used_bytes()``
       归零只是"回收到池"，VRAM 仍在池里（``total_bytes()`` 才是占卡量），必须
       ``free_all_blocks()`` 才真正还给驱动；
    3. ``torch.cuda.empty_cache()``：把 caching allocator 缓存的块还给驱动（同样，
       ``memory_allocated()`` 归零 ≠ 归还）；
    4. ``gc.collect()``：让 numpy 数组的引用环、以及 ``_execute`` 的缓冲池之外的对象
       真正销毁，RSS 才降得下来。

    **绝不抛异常**；失败原因进 :data:`LAST_ERROR`。峰值**不会被清零**——读数仍然
    是本组的（先读峰值再释放是正常顺序，反过来读也不会变成 0，见模块文档的调用顺序）。
    """
    global _LAST_RELEASE_S
    started = time.perf_counter()
    try:
        _stop_sampler()
    except Exception as exc:                          # noqa: BLE001
        _record("sampler_stop", exc)
    cupy = _cupy_if_ready()
    if cupy is not None:
        for label, getter in (
            ("cupy_pool", lambda: cupy.get_default_memory_pool()),
            ("cupy_pinned", lambda: cupy.get_default_pinned_memory_pool()),
        ):
            try:
                getter().free_all_blocks()
            except Exception as exc:                  # noqa: BLE001
                _record(label, exc)
    torch = _torch_if_ready()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception as exc:                      # noqa: BLE001
            _record("torch_empty_cache", exc)
    try:
        gc.collect()
    except Exception as exc:                          # noqa: BLE001
        _record("gc", exc)
    _LAST_RELEASE_S = time.perf_counter() - started
    return None


def format_peak_mb(value: float) -> str:
    """峰值渲染成机器可读的 ``"1234MB"``；非正数/不可用返回 ``"-"``。

    * 整数、无小数、无千分位（``1234.4 -> "1234MB"``，``1e6 -> "1000000MB"``）；
    * ``0.0``（以及负数、``None``、``nan``、``inf``）-> ``"-"``，表示"没测到"，与
      真的用满 0 MB 区分开；
    * 正的但四舍五入到 0（例如 ``0.4``）-> ``"0MB"``：那是"很小"，不是"没测到"。
    """
    try:
        number = float(value)
    except Exception:                                 # noqa: BLE001
        return "-"
    if number != number or number in (float("inf"), float("-inf")):
        return "-"
    if number <= 0.0:
        return "-"
    return f"{int(round(number))}MB"


def last_release_seconds() -> float:
    """最近一次 :func:`release_memory` 的耗时（秒）；没调过为 0.0。"""
    return float(_LAST_RELEASE_S)
