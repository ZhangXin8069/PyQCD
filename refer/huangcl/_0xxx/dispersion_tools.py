"""色散关系工具包.

主要功能:
    model_dispersion: 按 model 编号计算色散关系 E^2, 全部使用格点单位.
    calc_p2: 由格点动量整数算 p^2.
    fit_dispersion: 逐样本拟合色散关系, 返回 fit 的逐样本结果和均值结果.
    plot_dispersion: 画 E^2 的数据点和拟合带.

model=0: E^2 = k0 + k1 p^2.
model=1: E^2 = k0 + k1 p^2 + k2 p^4.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np

# ---- 从项目目录 98_tools 导入通用函数 ----
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import sem  # noqa: E402  # type: ignore
from figure_tools import calc_curve_band, plot_errbar  # noqa: E402  # type: ignore
from fit_tools import fit  # noqa: E402  # type: ignore
from snsc_params import get_conf  # noqa: E402  # type: ignore

# 各模型的拟合参数名和公式, 键为 model 编号.
DISPERSION_MODELS = {
    0: (["k0", "k1"], r"$E^2=k_0+k_1p^2$"),
    1: (["k0", "k1", "k2"], r"$E^2=k_0+k_1p^2+k_2a^2p^4$"),
}


def model_dispersion(p2: np.ndarray, para: dict, model: int = 0) -> np.ndarray:
    """
    计算格点单位制下的色散关系 E^2, 自变量为 p^2.

    参数:
    -----
    p2: 格点动量平方 p^2, para: momentum.
    para: 拟合参数字典, k0, k1, k2.

    可选项:
    -----
    model: 0 保留到 p^2 项, 1 额外加入 p^4 项.
    """
    e2 = para["k0"] + para["k1"] * p2
    if model == 1:
        e2 = e2 + para["k2"] * (p2 ** 2)
    return e2


def calc_p2(conf_short: str, momenta: list) -> np.ndarray:
    """
    由格点动量整数计算 p^2.

    参数:
    -----
    conf_short: 组态简称, 用于取空间格点数.
    momenta: 格点动量整数列表.

    输出:
    -----
    p2: 格点动量平方 p^2, para: momentum.
    """
    nx = get_conf(conf_short)["Nx"]
    # para: momentum
    return (2.0 * np.pi * np.asarray(momenta) / nx) ** 2


def fit_dispersion(
        conf_short: str, model: int, momenta: list, e: np.ndarray) -> tuple:
    """
    逐样本拟合格点单位制下的色散关系 E^2.

    参数:
    -----
    conf_short: 组态简称.
    model: 0 保留到 p^2 项, 1 额外加入 p^4 项.
    momenta: 格点动量整数列表.
    e: E 样本, 格点单位, para: sample, momentum.

    输出:
    -----
    samples: 逐样本拟合参数和 chi2dof, para: sample.
    ymean: 均值数据的拟合参数和 chi2dof.
    """
    p2 = calc_p2(conf_short, momenta)
    # para: sample, momentum
    e2 = e ** 2
    p0 = {"k0": 4.0, "k1": 1.0, "k2": 0.0}
    samples, _, _, _, ymean, _ = fit(
        y=e2, x=p2,
        model=partial(model_dispersion, model=model),
        p0={key: p0[key] for key in DISPERSION_MODELS[model][0]},
        method="least_squares", jackknife=True,
    )
    return samples, ymean


def plot_dispersion(
        conf_short: str, model: int, use_p0: bool, momenta: list,
        e: np.ndarray, samples: dict, output_file: Path) -> None:
    """
    画 E^2 的数据点和拟合带, 横轴和纵轴均转为 GeV^2.

    参数:
    -----
    conf_short: 组态简称.
    model: 0 保留到 p^2 项, 1 额外加入 p^4 项.
    use_p0: p=0 的点是否参与拟合.
    momenta: 全部格点动量整数列表, 含 p=0.
    e: E 样本, 格点单位, para: sample, momentum.
    samples: 逐样本拟合参数, 含 chi2dof, 均为格点单位.
    output_file: 图片路径.
    """
    names, formula = DISPERSION_MODELS[model]
    # 格点单位到 GeV 的换算因子.
    energy_unit = 0.1973 / get_conf(conf_short)["lattice_spacing"]
    p2_lattice = calc_p2(conf_short, momenta)
    # para: momentum
    p2 = p2_lattice * (energy_unit ** 2)
    # para: sample, momentum
    e2 = (e * energy_unit) ** 2

    # para: momentum
    e2_mean = e2.mean(axis=0)
    # para: momentum
    e2_error = sem(e2, jackknife=True)

    # 拟合带, 逐样本值 para: sample, x_idx.
    # para: curve
    curve_x, _curve_samples = calc_curve_band(
        partial(model_dispersion, model=model),
        {name: samples[name] for name in names},
        (0.0, float(p2_lattice.max())),
    )
    # 逐样本换算到 GeV^2 后再统计均值和误差.
    # para: sample, x_idx
    curve_y = _curve_samples * (energy_unit ** 2)

    curve_mean = curve_y.mean(axis=0)
    curve_error = sem(curve_y, jackknife=True)
    fit_data = {
        None: (
            curve_x * (energy_unit ** 2), curve_mean, curve_error),
    }
    if use_p0:
        raw_data = {"fit cand": (p2, e2_mean, e2_error)}
    else:
        raw_data = {
            "fit cand": (p2[1:], e2_mean[1:], e2_error[1:]),
            "not cand": (p2[:1], e2_mean[:1], e2_error[:1]),
        }
    plot_errbar(
        raw_data,
        output_file,
        xlabel=r"$p^2$ (GeV$^2$)",
        ylabel=r"$E^2$ (GeV$^2$)",
        title=f"{formula}, {conf_short}",
        x_offset=0.0,
        fit_data=fit_data,
        legend_loc="upper left",
    )
