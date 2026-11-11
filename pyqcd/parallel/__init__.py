"""pyqcd 并行子包：资源检测 + MPI 元任务调度（torch+mpi4py+h5py）。

``pyqcd.parallel.einsum.einsums`` 把多个**相互独立**的 einsum 任务按工作量分箱
后分给多个 MPI 进程执行（任务级并行，不改动 ``pyqcd.einsum`` 行为）。

``_mpi`` 侧会经 ``pyqcd.pipeline`` 拉入 h5py/cupy（实测约 0.6 s）；批量 einsum 的
worker 进程只需要轻量路径，所以这里全部改惰性解析，import 本包不再付出那份代价。
"""
from ._resources import detect_resources, cpu_threads, gpu_info

__all__ = [
    "detect_resources", "cpu_threads", "gpu_info",
    "get_mpi_context", "plan_parallel", "format_plan",
    "run_meta_task", "run_parallel_pipeline", "main",
    "einsums", "plan_einsums",
]

_MPI_NAMES = frozenset((
    "get_mpi_context", "plan_parallel", "format_plan",
    "run_meta_task", "run_parallel_pipeline", "main"))


def __getattr__(name):
    """惰性解析：``_mpi`` 的重依赖与批量 einsum 入口都不在包导入时加载。"""
    if name in _MPI_NAMES:
        from . import _mpi

        value = getattr(_mpi, name)
        globals()[name] = value
        return value
    if name in ("einsums", "plan_einsums"):
        from . import einsum as _parallel_einsum

        value = getattr(_parallel_einsum, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
