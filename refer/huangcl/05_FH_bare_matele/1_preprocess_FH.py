#!/public/home/huangcl/.venv/bin/python
"""
preprocess_FH.py

从 02_ratio 的结果目录读取全部 6 个方向的 ratio 数据, 求平均后做 FH 变换,
并画 FH 图.

用法:
    python preprocess_FH.py -c L24x72 -p 4

输出:
    1_result/{conf_short}/P{P}/fh/
        FH_nex{N}.npy      # FH 变换结果
        z{iz}.png          # FH 图 (多个 nex 对比)
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import sem, plot_errbar  # noqa: E402


# ===== 定义 dataclass =====

@dataclass
class ReadConfig:
    """读取与计算参数"""
    conf_short: str
    P: int
    nexmax: int                     # FH 时循环 nex=0..nexmax
    ave_dirs: list[str]             # 需要平均的 dir 列表
    z_list: list[int]               # 要画的 z 列表


@dataclass
class PathConfig:
    """路径配置"""
    ratio_dir: str                  # 02_ratio 结果目录 (读取)
    fh_dir: str                     # FH 结果保存目录 (输出)


@dataclass
class PlotConfig:
    """画图参数"""
    fh_xlim: list[float]            # FH 图横轴范围
    fh_ylim: list[float]            # FH 图纵轴范围
    xoffset: float = 0.2            # 多条曲线时横坐标错开量


# ===== 初始化配置 (返回 cfg 和路径) =====

def init_config(conf_short: str, P: int):
    """
    根据 conf_short 和 P 初始化配置, 创建输出目录.

    返回:
        rcfg: ReadConfig
        pcfg: PathConfig
        pltcfg: PlotConfig
    """
    # ===================================================
    # ===== 配置参数 (P 直接在下面修改) =====
    # ===================================================
    if conf_short == "L24x72":
        rcfg = ReadConfig(
            conf_short="L24x72",
            P=P,
            nexmax=2,   # FH 时循环 nex=0..nexmax (设为 0 则只算 nex=0)
            ave_dirs=["pos_x", "pos_y", "pos_z",
                       "neg_x", "neg_y", "neg_z"],
            z_list=list(range(3)),
        )
        pltcfg = PlotConfig(
            fh_xlim=[2.5, 20.5],
            fh_ylim=[-0.1, 1.1],
            xoffset=0.2,
        )
    else:
        print(f"conf {conf_short} not exist.")
        sys.exit()
    # ===================================================

    # 路径配置 (统一在 init_config 中管理)
    _project_dir = Path(__file__).resolve().parent.parent
    pcfg = PathConfig(
        ratio_dir=os.path.join(
            str(_project_dir), "02_ratio", "1_result",
            conf_short, f"P{P}"),
        fh_dir=os.path.join(
            os.getcwd(), "01_result", conf_short, f"P{P}", "fh"),
    )
    os.makedirs(pcfg.fh_dir, exist_ok=True)

    return rcfg, pcfg, pltcfg


# ===== 函数定义 =====

def load_one_ratio(pcfg: PathConfig, direction: str) -> np.ndarray:
    """
    读取指定方向的 ratio 数据.

    返回: ratio_array, shape (Nsample, dt, dtau, z)
    """
    _load_path = os.path.join(pcfg.ratio_dir, direction, "ratio.npy")

    print(f"  loading ratio ({direction}) from: {_load_path}")
    if not os.path.exists(_load_path):
        print(f"    Error: file not found: {_load_path}")
        sys.exit(1)

    ratio = np.load(_load_path)
    print(f"    loaded shape: (Nsample, dt, dtau, z) = {ratio.shape}")
    return ratio


def calc_fh(ratio: np.ndarray, save_path: str, nex: int = 0) -> np.ndarray:
    """
    计算 FH_n(t) = \sum_{\tau=nex}^{t+1-nex} R(t+1, \tau) - \sum_{\tau=nex}^{t-nex} R(t, \tau)

    对每个 dt, 在 \tau 方向去掉两端各 nex 个点后求和, 再做差分.

    参数:
        ratio: 输入 ratio 数组
        save_path: 保存路径
        nex: 两端各去掉的点数 (默认 0)

    返回: FH 数组
    """
    print(f"  computing FH (nex={nex})")

    Nsample, dtmax, _, Nz = ratio.shape

    # temp(t) = \sum_{\tau=nex}^{t-nex} R(t, \tau)
    temp = np.zeros((Nsample, dtmax, Nz))
    for dt in range(2 * nex, dtmax):  # 确保求和区间非空
        temp[:, dt] = ratio[:, dt, nex:dt-nex+1, :].sum(axis=1)

    # FH(t) = temp(t) - temp(t-1)
    fh = temp - np.roll(temp, 1, axis=1)

    print(f"    FH shape: (Nsample, dt, z) = {fh.shape}")

    # 保存 FH 结果
    np.save(save_path, fh)
    print(f"    FH saved to: {save_path}")

    return fh


def plot_fh(all_fh: dict, save_dir: str, conf_short: str, P: int,
            rcfg: ReadConfig, pltcfg: PlotConfig):
    """
    画 FH 图, 多个 nex 画在同一张图上对比.

    参数:
        all_fh: {nex: fh_array}, 每个 fh_array shape (Nsample, dt, z)
        save_dir: 保存目录
        conf_short: 配置名
        P: 动量
        rcfg: ReadConfig
        pltcfg: PlotConfig
    """
    print(f"  plotting FH to: {save_dir}")

    _first_nex = next(iter(all_fh.keys()))
    _, dt_max, Nz = all_fh[_first_nex].shape
    t_vals = np.arange(dt_max)

    for _iz in rcfg.z_list:
        if _iz >= Nz:
            print(f"    Warning: z={_iz} exceeds Nz={Nz}, skipping")
            continue

        # 收集所有 nex 的数据
        _data = {}
        for _nex, _fh in sorted(all_fh.items()):
            _mean = _fh.mean(0)[:, _iz]       # (dt,)
            _err = sem(_fh, jackknife=False)[:, _iz]  # (dt,)
            _data[f"nex={_nex}"] = (_mean, _err)

        save_path = os.path.join(save_dir, f"z{_iz}.png")
        plot_errbar(
            t_vals, _data, save_path,
            xlabel="t", ylabel="FH",
            xlim=pltcfg.fh_xlim, ylim=pltcfg.fh_ylim,
            x_offset=pltcfg.xoffset,
            title=f"{conf_short}, P={P}, z={_iz}",
        )


# ===== 主函数 =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess: load ratio → FH transform → plot FH")
    parser.add_argument("-c", type=str, default="L24x72",
                        help="conf_short (default: L24x72)")
    parser.add_argument("-p", type=int, default=2,
                        help="momentum P (default: 2)")
    args = parser.parse_args()

    conf_short = args.c
    P = args.p

    # 初始化配置和路径
    rcfg, pcfg, pltcfg = init_config(conf_short, P)

    print(f"conf_short: {conf_short}, P: {P}")
    print(f"ave_dirs: {rcfg.ave_dirs}")

    _t0 = time.perf_counter()

    # ---- 1. 加载各方向 ratio 并求平均 ----
    print(f"\n{'='*60}")
    print(f"  loading ratios from ave_dirs: {rcfg.ave_dirs}")
    _ratios = []
    for _dir in rcfg.ave_dirs:
        _r = load_one_ratio(pcfg, _dir)
        _ratios.append(_r)
    ratio_ave = np.mean(_ratios, axis=0)
    print(f"  ratio_ave shape: {ratio_ave.shape}")

    # ---- 2. 对平均后的 ratio 做 FH 变换 ----
    all_fh = {}
    for _nex in range(rcfg.nexmax + 1):
        _fh_path = os.path.join(pcfg.fh_dir, f"FH_nex{_nex}.npy")
        fh = calc_fh(ratio_ave, _fh_path, nex=_nex)
        all_fh[_nex] = fh

    # ---- 3. 画 FH 对比图 ----
    plot_fh(all_fh, pcfg.fh_dir, conf_short, P, rcfg, pltcfg)

    _t1 = time.perf_counter()
    print(f"\nTotal time: {(_t1 - _t0):.2f}s")
    print("preprocess_FH done.")
