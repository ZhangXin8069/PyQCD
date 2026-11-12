"""pyqcd 并行子包：本机资源探测（CPU 线程 / 内存 / GPU / 磁盘）。

只提供只读的资源快照，不做任务调度、不启动子进程、不依赖任何分布式运行时。
``detect_resources`` 汇总全部字段，``cpu_threads`` / ``gpu_info`` 等可按需单独调用。
"""
from ._resources import detect_resources, cpu_threads, gpu_info

__all__ = [
    "detect_resources", "cpu_threads", "gpu_info", "total_memory_mb",
    "available_memory_mb", "free_disk_mb",
]


def __getattr__(name):
    """惰性解析：探测函数都在 ``_resources`` 里，导入本包不触发任何探测。"""
    from . import _resources

    try:
        value = getattr(_resources, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
