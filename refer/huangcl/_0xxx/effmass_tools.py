"""有效质量计算, 关联函数拟合和画图工具包.

主要功能:
    calc_effective_mass: 由重采样关联函数计算有效质量.
    two_state_model: 双态贡献的关联函数模型.
    model_effmass: 双态模型对应的有效质量.
    fit_correlator: 对关联函数做双态拟合, 返回逐样本参数和 chi2/dof.
    plot_effmass: 画有效质量误差棒, E0 平台色带和双态模型有效质量曲线.
"""

import sys

import numpy as np

# ---- 从项目目录 98_tools 导入通用函数 ----
sys.path.insert(0, "/public/home/huangcl/06_deuteron_gluon_pdf/98_tools")
from analysis_tools import format_mean_err, sem  # noqa: E402  # type: ignore
from figure_tools import calc_curve_band, plot_errbar  # noqa: E402  # type: ignore
from fit_tools import fit  # noqa: E402  # type: ignore


# ==================== 有效质量 ====================
def calc_effective_mass(
        correlator_samples: np.ndarray, energy_unit: float) -> np.ndarray:
    """
    由重采样关联函数计算有效质量.

    输入:
    -----
    correlator_samples: 关联函数样本, para: sample, dt.
    energy_unit: 格点能量单位, 单位 GeV.

    输出:
    -----
    effmass: 有效质量样本, para: sample, dt.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.log(
                np.roll(correlator_samples, shift=1, axis=1)
                / correlator_samples
            )
            * energy_unit
        )


# ==================== 拟合 ====================
def two_state_model(t: np.ndarray, para: dict) -> np.ndarray:
    """
    计算双态模型 C(t) = A0 * exp(-E0*t) * (1 + A1 * exp(-dE*t)).

    输入:
    -----
    t: 拟合时间点.
    para: 拟合参数, 键为 A0, E0, A1 和 dE, 均取格点单位.

    输出:
    -----
    model_value: 模型值, para: t.
    """
    t = np.asarray(t, dtype=float)
    return (
        para["A0"] * np.exp(-para["E0"] * t)
        * (1.0 + para["A1"] * np.exp(-para["dE"] * t))
    )


def model_effmass(
        t: np.ndarray, para: dict, energy_unit: float) -> np.ndarray:
    """
    计算双态模型对应的有效质量 log(C(t-1)/C(t)) * energy_unit.

    输入:
    -----
    t: 拟合时间点.
    para: 拟合参数, 键为 A0, E0, A1 和 dE, 均取格点单位.
    energy_unit: 格点能量单位, 单位 GeV.

    输出:
    -----
    effmass: 模型有效质量, para: t, 单位 GeV.
    """
    t = np.asarray(t, dtype=float)
    return np.log(
        two_state_model(t - 1.0, para) / two_state_model(t, para)
    )


def fit_correlator(
        corr_samples: np.ndarray, fit_t_min: int, fit_t_max: int,
        p0: dict, rol: dict) -> tuple:
    """
    对关联函数在拟合窗口内做双态拟合.

    输入:
    -----
    corr_samples: 关联函数样本, para: sample, dt.
    fit_t_min, fit_t_max: 拟合窗口.
    p0: 四个拟合参数初值, 键为 A0, E0, A1 和 dE.
    rol: 与 p0 同键的约束半径.

    输出:
    -----
    para_samples: 四个拟合参数的逐样本拟合值, 键与 p0 相同,
                  para: sample, 格点单位.
    chi2dof: 均值数据的 chi2/dof.
    """
    # para: fit_t
    t_fit = np.arange(fit_t_min, fit_t_max + 1)
    # para: sample, fit_t
    y_fit = corr_samples[:, t_fit]
    fit_result, _, _, _, fit_mean, _ = fit(
        y=y_fit,
        x=t_fit,
        model=two_state_model,
        p0=p0,
        rol=rol,
        method="least_squares",
        jackknife=True,
    )
    para_samples = {name: fit_result[name] for name in p0}
    return para_samples, fit_mean["chi2dof"]


# ==================== 画图 ====================
def plot_effmass(
        corr_samples: np.ndarray, para_samples: dict, chi2dof: float,
        fit_t_min: int, fit_t_max: int, Plot) -> None:
    """
    由关联函数样本计算有效质量, 并画有效质量误差棒与拟合结果.

    输入:
    -----
    corr_samples: 关联函数样本, para: sample, dt, 格点单位.
    para_samples: 双态模型的逐样本参数, 键为 A0, E0, A1 和 dE,
                  para: sample, 格点单位.
    chi2dof: 均值数据的 chi2/dof.
    fit_t_min, fit_t_max: 拟合窗口, 决定模型曲线的横轴范围.
    Plot: 画图配置, 提供 xlim, ylim, energy_unit, conf_short, title 和
          figure_file.
    """
    # para: sample, dt
    effmass_samples = calc_effective_mass(corr_samples, Plot.energy_unit)
    # para: dt
    effmass_mean = effmass_samples.mean(axis=0)
    # para: dt
    effmass_err = sem(effmass_samples, jackknife=True)
    # para: dt
    t = np.arange(effmass_samples.shape[1])

    # para: sample
    e0_samples = para_samples["E0"] * Plot.energy_unit
    e0_mean = e0_samples.mean()
    e0_err = sem(e0_samples, jackknife=True)
    band_label = (
        f"E0={format_mean_err(e0_mean, e0_err)} GeV, "
        f"chi2/dof={chi2dof:.3g}"
    )

    # para: fit_t, para: sample, fit_t
    curve_x, curve_samples = calc_curve_band(
        model_effmass, para_samples, (fit_t_min, fit_t_max),
        model_args=(Plot.energy_unit,),
    )
    curve_samples = curve_samples * Plot.energy_unit
    curve_mean = curve_samples.mean(axis=0)
    curve_err = sem(curve_samples, jackknife=True)
    fit_data = {"two-state fit": (curve_x, curve_mean, curve_err)}

    plot_errbar(
        {"m_eff": (t, effmass_mean, effmass_err)},
        Plot.figure_file,
        xlabel="t/a",
        ylabel="effective mass (GeV)",
        xlim=Plot.xlim,
        ylim=Plot.ylim,
        title=Plot.title,
        fit_data=fit_data,
        fit_x_offset=0.0,
        band=(band_label, e0_mean, e0_err),
        legend_loc="lower left",
    )
