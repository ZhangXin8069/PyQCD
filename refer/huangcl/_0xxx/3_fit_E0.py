#!/public/home/huangcl/.venv/bin/python
"""
功能: 读取 cg5/cg5gt 的 NP 2pt 数据, 逐方向投影得到氘核关联函数,
      对方向做等权平均后再做 jackknife 重采样和有效质量双态拟合.
输入: init() 中的组态, 动量, 方向列表, 拟合窗口和画图参数.
输出: 02_data/<conf_short>/fit_p<mom>.npz, 方向平均后的双态模型参数结果;
      03_fig_rep/<conf_short>/effmass_p<mom>_<conf_short>.png.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from effmass_tools import fit_correlator, plot_effmass

# ---- 从项目目录 98_tools, 07_get_channel_2pt 和本目录导入函数 ----
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import calc_momentum_GeV, resample  # noqa: E402  # type: ignore
from output_tools import save_npz  # noqa: E402  # type: ignore
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
class FitConfig:
    """关联函数拟合参数, 全部使用格点单位."""
    fit_t_min: int
    fit_t_max: int
    p0: dict
    rol: dict
    data_dir: Path


@dataclass
class PlotConfig:
    """画图参数."""
    xlim: list
    ylim: list
    energy_unit: float
    conf_short: str
    title: str
    figure_file: Path


# ==================== 初始化 ====================
def init() -> Tuple[ReadConfig, FitConfig, PlotConfig]:
    """
    初始化 cg5/cg5gt 数据路径, 拟合参数, 保存路径和画图参数.

    输出:
    -----
    Read: 数据读取参数, 包含 conf_short, conf_num, exclude_ids,
          momentum, mom_dirs.
    Fit: 关联函数拟合参数, 包含 fit_t_min, fit_t_max, p0, rol,
         data_dir.
    Plot: 画图参数, 包含 xlim, ylim, energy_unit, conf_short, title,
          figure_file.
    """
    CONF_SHORT = "c24p29"
    MOMENTUM = 2
    # 读取的总动量方向, 含正负两个方向, 平均时全部使用.
    MOM_DIRS = ["x"]

    MASS_GEV = 2.0
    FIT_ROL = {"A0": 1.0, "E0": 1.0, "A1": 1.0, "dE": 1.0}

    # 各组态的读取参数, 键为 conf_short.
    CONF_DICT = {
        "c24p29": {"conf_num": 299, "exclude_ids": [12300]},
        "c32p29": {"conf_num": 149, "exclude_ids": [18350]},
    }

    FIT_T_DICT = {
        "c24p29": {
            0: (7, 15), 2: (6, 13), 4: (6, 12), 6: (4, 11), 8: (4, 10),
        },
        "c32p29": {
            0: (7, 15), 2: (5, 12), 4: (6, 10), 6: (6, 12), 8: (4, 10),
        },
    }

    CONF_NUM = CONF_DICT[CONF_SHORT]["conf_num"]
    EXCLUDE_IDS = CONF_DICT[CONF_SHORT]["exclude_ids"]
    FIT_T_MIN, FIT_T_MAX = FIT_T_DICT[CONF_SHORT][MOMENTUM]

    conf_data = get_conf(CONF_SHORT)
    energy_unit = 0.1973 / conf_data["lattice_spacing"]
    momentum_gev = calc_momentum_GeV(MOMENTUM, CONF_SHORT)
    # 由 E^2 = p^2 + m^2 估计 E0, 单位 GeV.
    e0_gev = np.sqrt(momentum_gev ** 2 + MASS_GEV ** 2)
    # E0 初值取色散关系估计值, 格点单位.
    fit_p0 = {
        "A0": 1.0,
        "E0": e0_gev / energy_unit,
        "A1": 1.0,
        "dE": 0.5,
    }
    xlim = [3.5, 16.5]
    # 纵轴取 E0 估计值的正负 1 GeV.
    ylim = [e0_gev - 1.0, e0_gev + 1.0]
    title = (
        f"deuteron effective mass, |P|={momentum_gev:.3g} GeV, "
        f"{CONF_SHORT}, Nconf={CONF_NUM}"
    )

    data_dir = script_dir / "02_data" / CONF_SHORT
    figure_file = (
        script_dir / "03_fig_rep" / CONF_SHORT
        / f"fitE0_p{MOMENTUM}_{CONF_SHORT}.png"
    )

    Read = ReadConfig(
        CONF_SHORT, CONF_NUM, EXCLUDE_IDS, MOMENTUM, MOM_DIRS)
    Fit = FitConfig(FIT_T_MIN, FIT_T_MAX, fit_p0, FIT_ROL, data_dir)
    Plot = PlotConfig(
        xlim, ylim, energy_unit, CONF_SHORT, title, figure_file)
    return Read, Fit, Plot


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Read, Fit, Plot = init()

    # 逐方向读取, 投影并对 sz 和 t_src 平均, 之后对方向做等权平均.
    deuteron_2pt_list = []
    for mom_dir in Read.mom_dirs:
        # para: conf, dt, t_src, r, q, z, y
        correlators = load_NPcorr(
            Read.conf_short, Read.conf_num, Read.momentum, mom_dir,
            Read.exclude_ids)
        # para: conf, dt, t_src
        _deuteron_2pt = np.real(proj_deuteron_chan(correlators).mean(-1))
        # para: conf, dt
        deuteron_2pt_list.append(_deuteron_2pt.mean(-1))

    # para: dir, conf, dt
    deuteron_2pt_dir = np.asarray(deuteron_2pt_list)
    # para: conf, dt
    deuteron_2pt = deuteron_2pt_dir.mean(axis=0)

    # para: sample, dt
    corr_samples = resample(deuteron_2pt, jackknife=True)

    # 双态模型参数逐样本拟合值, 格点单位; chi2dof 为均值数据的 chi2/dof.
    para_samples, chi2dof = fit_correlator(
        corr_samples, Fit.fit_t_min, Fit.fit_t_max, Fit.p0, Fit.rol)

    # para: sample
    save_npz(
        Fit.data_dir / f"fit_p{Read.momentum}.npz", name="fit",
        **para_samples, chi2dof=chi2dof)

    plot_effmass(
        corr_samples, para_samples, chi2dof,
        Fit.fit_t_min, Fit.fit_t_max, Plot,
    )

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
