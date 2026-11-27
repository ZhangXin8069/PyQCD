#!/public/home/huangcl/.venv/bin/python
"""
功能: 读取 cg5/cg5gt 的 NP 2pt 数据, 逐方向投影得到氘核关联函数, 做 jackknife
      重采样并计算有效质量, 把同一 |P| 下各方向的有效质量画到同一张图上.
输入: init() 中的组态, 动量, 方向列表和画图参数.
输出: 03_fig_rep/<conf_short>/effmass_p<mom>_dir_<conf_short>.png.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from effmass_tools import calc_effective_mass

# ---- 从项目目录 98_tools, 07_get_channel_2pt 和本目录导入函数 ----
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import (  # noqa: E402  # type: ignore
    calc_momentum_GeV, resample, sem,
)
from figure_tools import plot_errbar  # noqa: E402  # type: ignore
from snsc_params import get_conf  # noqa: E402  # type: ignore

sys.path.insert(0, str(project_dir / "07_get_channel_2pt"))
from get_channel_2pt import load_NPcorr, proj_deuteron_chan  # noqa: E402  # type: ignore

# ==================== 参数定义 ====================


@dataclass
class ReadConfig:
    """数据读取参数."""
    conf_short: str
    conf_num: int
    exclude_ids: list
    momentum: int
    mom_dirs: list


@dataclass
class PlotConfig:
    """画图参数."""
    xlim: list
    ylim: list
    energy_unit: float
    title: str
    figure_file: Path


# ==================== 初始化 ====================
def init() -> Tuple[ReadConfig, PlotConfig]:
    """
    初始化 cg5/cg5gt 数据路径和画图参数.

    输出:
    -----
    Read: 数据读取参数, 包含 conf_short, conf_num, exclude_ids,
          momentum, mom_dirs.
    Plot: 画图参数, 包含 xlim, ylim, energy_unit, title, figure_file.
    """
    CONF_SHORT = "c24p29"
    MOMENTUM = 0
    # 需要画出的总动量方向, 正负方向各一条曲线.
    MOM_DIRS = ["x"]

    MASS_GEV = 2.0

    # 各组态的读取参数, 键为 conf_short.
    CONF_DICT = {
        "c24p29": {"conf_num": 299, "exclude_ids": [12300]},
        "c32p29": {"conf_num": 149, "exclude_ids": [18350]},
    }

    CONF_NUM = CONF_DICT[CONF_SHORT]["conf_num"]
    EXCLUDE_IDS = CONF_DICT[CONF_SHORT]["exclude_ids"]

    conf_data = get_conf(CONF_SHORT)
    energy_unit = 0.1973 / conf_data["lattice_spacing"]
    momentum_gev = calc_momentum_GeV(MOMENTUM, CONF_SHORT)
    # 由 E^2 = p^2 + m^2 估计 E0, 单位 GeV, 用于确定纵轴范围.
    e0_gev = np.sqrt(momentum_gev ** 2 + MASS_GEV ** 2)

    xlim = [3.5, 16.5]
    # 纵轴取 E0 估计值的正负 1 GeV.
    ylim = [e0_gev - 1.0, e0_gev + 1.0]
    title = (
        f"deuteron effective mass, |P|={momentum_gev:.3g} GeV, "
        f"{CONF_SHORT}, Nconf={CONF_NUM}"
    )

    figure_file = (
        script_dir / "03_fig_rep" / CONF_SHORT
        / f"effmass_p{MOMENTUM}_{CONF_SHORT}.png"
    )

    Read = ReadConfig(
        CONF_SHORT, CONF_NUM, EXCLUDE_IDS, MOMENTUM, MOM_DIRS)
    Plot = PlotConfig(xlim, ylim, energy_unit, title, figure_file)
    return Read, Plot


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Read, Plot = init()

    # 键为方向标签, 值为该方向的有效质量 (均值, 误差).
    effmass_data = {}
    for mom_dir in Read.mom_dirs:
        # para: conf, dt, t_src, r, q, z, y
        correlators = load_NPcorr(
            Read.conf_short, Read.conf_num, Read.momentum, mom_dir,
            Read.exclude_ids)
        # para: conf, dt, t_src
        _deuteron_2pt = np.real(proj_deuteron_chan(correlators).mean(-1))
        # para: conf, dt
        _deuteron_2pt = _deuteron_2pt.mean(-1)

        # para: sample, dt
        corr_samples = resample(_deuteron_2pt, jackknife=True)
        # para: sample, dt
        effmass_samples = calc_effective_mass(corr_samples, Plot.energy_unit)

        effmass_data[f"p{mom_dir}{Read.momentum}"] = (
            np.arange(effmass_samples.shape[-1]),
            effmass_samples.mean(axis=0),
            sem(effmass_samples, jackknife=True)
        )

    plot_errbar(
        effmass_data, Plot.figure_file,
        xlabel="t/a",
        ylabel="effective mass (GeV)",
        xlim=Plot.xlim,
        ylim=Plot.ylim,
        title=Plot.title,
        x_offset=0.15,
    )

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
