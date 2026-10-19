"""
功能: 性能监测工具包, 提供内存和显存占用监测函数.
输出: 打印 CPU 进程峰值内存和 GPU 显存占用.
说明: 内存和显存统一换算为 GB, 数组大小按 nbytes 计算.
"""
import resource

import cupy as cp


# ==================== 内存与显存监测 ====================
def print_mem_peak(tag, use_gpu=True):
    """
    打印 CPU 进程峰值内存, 按需打印 GPU 显存占用, 单位 GB.

    参数:
    -----
    tag: 输出标签, 用于区分调用位置.

    可选项:
    -----
    use_gpu: 是否打印 GPU 显存占用, 默认 True 打印 CPU 峰值和 GPU 显存; False 只打印 CPU 峰值.

    输出:
    -----
    无返回值, 直接打印 CPU 进程 RSS 峰值, 以及 GPU 已用显存和保留显存.
    """
    # ru_maxrss 在 Linux 上单位为 KB, 换算为 GB 需除以 1024**2.
    cpu_peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    message = f"{tag}: cpu rss peak {cpu_peak_gb / 1024**2:.1f} GB"
    if use_gpu:
        pool = cp.get_default_memory_pool()
        message += (
            f", gpu used {pool.used_bytes() / 1024**3:.1f} GB"
            f", gpu reserved {pool.total_bytes() / 1024**3:.1f} GB"
        )
    print(message)


def print_var_size(tag, **variables):
    """
    打印若干变量所占内存/显存大小, 单位 GB.

    参数:
    -----
    tag: 输出标签, 用于区分调用位置.

    可变关键字参数:
    -----
    variables: 变量名和变量本身, 例如 print_var_size("2pt", vvv=vvv, peram=peram).

    输出:
    -----
    无返回值, 直接打印每个变量的名称和大小.
    """
    message = f"{tag}:"
    for name, var in variables.items():
        message += f" {name} {var.nbytes / 1024**3:.1f} GB"
    print(message)
