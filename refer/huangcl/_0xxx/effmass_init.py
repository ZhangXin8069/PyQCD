#!/public/home/huangcl/.venv/bin/python
"""
功能: 初始化 cg5/cg5gt 氘核有效质量分析的读取,
      拟合和画图参数.
输入: 无.
输出: ReadConfig, FitConfig 和 PlotConfig 配置对象.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple

import numpy as np

# ---- 从项目目录 98_tools 导入通用函数 ----
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, "/public/home/huangcl/06_deuteron_gluon_pdf/98_tools")
from snsc_params import get_conf  # noqa: E402  # type: ignore


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


@dataclass
class PlotConfig:
    """画图参数."""
    xlim: list
    ylim: list
    energy_unit: float
    conf_short: str
    title: str
    figure_file: Path


# ==================== 动量 ====================
def calc_momentum_GeV(momentum: int, conf_short: str) -> float:
    """
    由格点动量整数换算物理动量大小, 单位 GeV.

    参数:
    -----
    momentum: 格点动量整数, p = 2*pi*momentum/(Nx*a).
    conf_short: 组态短标识, 用于读取 Nx 和格距.

    输出:
    -----
    momentum_gev: 物理动量大小, 单位 GeV.
    """
    conf_data = get_conf(conf_short)
    return (
        2.0 * np.pi * momentum
        / (conf_data["Nx"] * conf_data["lattice_spacing"])
        * 0.1973
    )


# ==================== 初始化 ====================
def init() -> Tuple[ReadConfig, FitConfig, PlotConfig]:
    """
    初始化 cg5/cg5gt 数据路径和分析参数.

    输出:
    -----
    Read: 数据读取参数, 包含 conf_short, conf_num, exclude_ids,
          momentum, mom_dirs.
    Fit: 关联函数拟合参数, 包含 fit_t_min, fit_t_max, p0, rol.
    Plot: 画图参数, 包含 xlim, ylim, energy_unit, conf_short, title,
          figure_file.
    """
    CONF_SHORT = "c24p29"
    MOMENTUM = 8
    # 读取的总动量方向, 当前只有 x 方向的数据.
    MOM_DIRS = ["x"]

    MASS_GEV = 2.0
    FIT_ROL = {"A0": 1.0, "E0": 1.0, "A1": 1.0, "dE": 1.0}

    if CONF_SHORT == "c24p29":
        CONF_NUM = 50
        EXCLUDE_IDS = []
    elif CONF_SHORT == "c32p29":
        CONF_NUM = 50
        EXCLUDE_IDS = []
    else:
        raise ValueError(f"conf_short = {CONF_SHORT} not exist.")

    # 拟合区间按动量给出.
    if MOMENTUM == 0:
        FIT_T_MIN = 9
        FIT_T_MAX = 15
    elif MOMENTUM == 8:
        FIT_T_MIN = 5
        FIT_T_MAX = 10
    else:
        raise ValueError(f"momentum = {MOMENTUM} not exist.")

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
    xlim = [2.5, FIT_T_MAX + 3.5]
    # 纵轴取 E0 估计值的正负 1 GeV.
    ylim = [e0_gev - 1.0, e0_gev + 1.0]
    title = (
        f"deuteron effective mass, |P|={momentum_gev:.3g} GeV, "
        f"{CONF_SHORT}"
    )

    output_dir = script_dir / "01_result" / "fig" / CONF_SHORT
    figure_file = output_dir / f"mass_{CONF_SHORT}_P{MOMENTUM}.png"

    Read = ReadConfig(
        CONF_SHORT, CONF_NUM, EXCLUDE_IDS, MOMENTUM, MOM_DIRS)
    Fit = FitConfig(FIT_T_MIN, FIT_T_MAX, fit_p0, FIT_ROL)
    Plot = PlotConfig(
        xlim, ylim, energy_unit, CONF_SHORT, title, figure_file)
    return Read, Fit, Plot
