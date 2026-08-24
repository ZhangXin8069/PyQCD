#!/public/home/huangcl/.venv/bin/python
"""
比较两个 ratio 数据文件 (格式相同) 的均值与 sem 的绝对误差.

这两个文件都是用户自己的数据, 分别由两种不同方法计算得到:
    方法1 (mine): 04_gluon_unpolarized_PDF/check_for_zch_centered/1_result/L24x72/P4/ratio/ratio_ave.npy
    方法2 (zch):  04_gluon_unpolarized_PDF/check_for_zch/1_result/L24x72/P4/ave/ratio.npy

输出:
    打印两个文件在有效区域 (tins <= tsep) 内均值与 sem 的绝对误差统计
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import sem  # noqa: E402


# ===== 独立开关, 方便调试时修改 =====
DEBUG = False
# ===================================


# ===== 定义 dataclass =====

@dataclass
class CompareParams:
    """比较参数"""
    dt_max: int = 20            # tsep 维度大小
    Nx: int = 24                # z 维度大小
    tsep_min: int = 0           # tsep 统计最小值
    tsep_max: int = 20          # tsep 统计最大值
    abs_threshold: float = 1e-2  # 绝对误差统计阈值


@dataclass
class PathConfig:
    """路径配置"""
    mine_path: str = ""
    zch_path: str = ""


# ===== 初始化函数 =====

def init_config():
    """初始化配置

    输入:
        无 (配置硬编码在函数内)

    输出:
        cmppa: CompareParams 实例
        pathpa: PathConfig 实例
    """
    # 脚本所在目录 (04_gluon_unpolarized_PDF/check_for_zch_centered)
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    cmppa = CompareParams()
    pathpa = PathConfig(
        mine_path=os.path.join(
            _script_dir, "1_result", "L24x72", "P4", "ratio", "ratio_ave.npy"),
        zch_path=os.path.join(
            _script_dir, "..", "check_for_zch",
            "1_result", "L24x72", "P4", "ave", "ratio.npy"),
    )
    return cmppa, pathpa


# ===== 功能函数 =====

def load_ratio(file_path: str):
    """加载 ratio 数据

    输入:
        file_path: ratio 文件路径

    输出:
        ratio: 数组, para 依次为 Nsample, tsep, tins, z
    """
    ratio = np.load(file_path)
    print(f"  loaded: {file_path}")
    print(f"  ratio shape: {ratio.shape}")
    return ratio


def compare_ratio(mine, zch, cmppa: CompareParams):
    """比较 mine 与 zch 的 ratio 的均值与 sem 的绝对误差

    输入:
        mine: 方法1的 ratio 数据, para 依次为 Nsample, tsep, tins, z
        zch: 方法2的 ratio 数据, para 依次为 Nsample, tsep, tins, z
        cmppa: CompareParams 实例

    输出:
        打印有效区域 (tins <= tsep) 内均值与 sem 的绝对误差统计
    """
    if mine.shape != zch.shape:
        print(f"  shape mismatch! mine={mine.shape}, zch={zch.shape}")
        return

    dt_max = cmppa.dt_max
    Nx = cmppa.Nx

    # 有效区域: tins <= tsep, tsep_min <= tsep <= tsep_max
    tsep_idx = np.arange(dt_max)[:, None, None]   # (tsep, 1, 1)
    tins_idx = np.arange(dt_max)[None, :, None]   # (1, tins, 1)
    valid = (tins_idx <= tsep_idx) & (tsep_idx >= cmppa.tsep_min) & (
        tsep_idx <= cmppa.tsep_max)  # (tsep, tins)
    mask = np.broadcast_to(valid, (dt_max, dt_max, Nx))  # (tsep, tins, z)

    # 均值与 sem, 沿 Nsample 轴
    mine_mean = mine.mean(0)  # (tsep, tins, z)
    mine_sem = sem(mine, jackknife=False)  # (tsep, tins, z)
    zch_mean = zch.mean(0)  # (tsep, tins, z)
    zch_sem = sem(zch, jackknife=False)  # (tsep, tins, z)

    # 绝对误差
    mean_abs_diff = np.abs(mine_mean - zch_mean)
    sem_abs_diff = np.abs(mine_sem - zch_sem)

    # 只统计有效区域
    mean_abs_masked = np.where(mask, mean_abs_diff, np.nan)
    sem_abs_masked = np.where(mask, sem_abs_diff, np.nan)

    n_valid = int(np.sum(mask))
    th = cmppa.abs_threshold

    print(
        f"===== mean abs_diff statistics (tins <= tsep, "
        f"{cmppa.tsep_min} <= tsep <= {cmppa.tsep_max}) =====")
    print(f"  valid points: {n_valid}")
    print(f"  mean abs_diff: mean = {np.nanmean(mean_abs_masked):.6e}, "
          f"max = {np.nanmax(mean_abs_masked):.6e}")
    print(f"  mean abs_diff > {th}: {int(np.sum(mean_abs_masked > th))}")
    print(
        f"  mean abs_diff > {th / 10}: {int(np.sum(mean_abs_masked > th / 10))}")
    print(
        f"  mean abs_diff > {th / 100}: {int(np.sum(mean_abs_masked > th / 100))}")

    print(
        f"===== sem abs_diff statistics (tins <= tsep, "
        f"{cmppa.tsep_min} <= tsep <= {cmppa.tsep_max}) =====")
    print(f"  valid points: {n_valid}")
    print(f"  sem abs_diff: mean = {np.nanmean(sem_abs_masked):.6e}, "
          f"max = {np.nanmax(sem_abs_masked):.6e}")
    print(f"  sem abs_diff > {th}: {int(np.sum(sem_abs_masked > th))}")
    print(
        f"  sem abs_diff > {th / 10}: {int(np.sum(sem_abs_masked > th / 10))}")
    print(
        f"  sem abs_diff > {th / 100}: {int(np.sum(sem_abs_masked > th / 100))}")


# ===== 主函数 =====

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Compare two ratio files (mean and sem abs diff)")
    parser.add_argument("-m", type=str, default="",
                        help="mine ratio file path (default: use init_config)")
    parser.add_argument("-z", type=str, default="",
                        help="zch ratio file path (default: use init_config)")
    args = parser.parse_args()

    time0 = time.perf_counter()

    # 初始化配置
    cmppa, pathpa = init_config()

    # 命令行参数覆盖
    if args.m:
        pathpa.mine_path = args.m
    if args.z:
        pathpa.zch_path = args.z

    print("mine path:", pathpa.mine_path)
    print("zch path:", pathpa.zch_path)

    # 加载数据
    print("\nLoading mine ratio ...")
    mine = load_ratio(pathpa.mine_path)
    print("\nLoading zch ratio ...")
    zch = load_ratio(pathpa.zch_path)

    # 比较
    print("\n===== compare ratio =====")
    compare_ratio(mine, zch, cmppa)

    time1 = time.perf_counter()
    print(f"\nspend time: {(time1 - time0):.2f}s")
    print("job finish")
