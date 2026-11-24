#!/public/home/huangcl/.venv/bin/python
"""
功能: 读取指定组态各动量的 E jackknife 样本, 按 init() 指定的色散模型逐样本拟合,
      全程使用格点单位制, use_p0 控制 p=0 的点是否参与拟合, 画图时始终画出.
      model=0: E^2 = m^2 + k0 p^2;
      model=1: E^2 = m^2 + k0 p^2 + k1 p^4.
输入: 08_check_2pt/02_data/<conf_short>/fit_p<mom>.npz 中的 E0, 格点单位.
输出: 02_data/<conf_short>/disp_<tag>.npz, fit 返回的逐样本结果和均值结果;
      02_data/<conf_short>/_rep_disp_<tag>.txt, 拟合参数表;
      03_fig_rep/<conf_short>/disp_<tag>.png, tag = model<model>_<conf_short>.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dispersion_tools import fit_dispersion, plot_dispersion

# ---- 从项目目录 98_tools 导入通用函数 ----
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import format_mean_err, sem  # noqa: E402  # type: ignore
from output_tools import save_npz, save_txt  # noqa: E402  # type: ignore
from snsc_params import get_conf  # noqa: E402  # type: ignore


@dataclass
class FitConfig:
    conf_short: str
    model: int
    momenta: list
    use_p0: bool
    data_file: Path
    report_file: Path
    figure_file: Path


def init() -> FitConfig:
    """
    初始化组态, 模型, 动量和输出路径.

    输出:
    -----
    Config: 运行参数.
    """
    CONF_SHORT = "c24p29"
    MODEL = 0
    # 是否把 p=0 的点纳入拟合, 画图时始终画出 p=0 点.
    USE_P0 = False

    # 各组的动量列表, 键为 conf_short.
    MOMENTA_DICT = {
        "c24p29": [0, 2, 4, 6, 8],
        "c32p29": [0, 2, 4, 6, 8],
    }

    # para: momentum
    momenta = MOMENTA_DICT[CONF_SHORT]
    tag = f"model{MODEL}_{CONF_SHORT}" + ("" if USE_P0 else "_wop0")
    data_dir = script_dir / "02_data" / CONF_SHORT
    fig_dir = script_dir / "03_fig_rep" / CONF_SHORT
    return FitConfig(
        CONF_SHORT,
        MODEL,
        momenta,
        USE_P0,
        data_dir / f"disp_{tag}.npz",
        fig_dir / f"_rep_disp_{tag}.txt",
        fig_dir / f"disp_{tag}.png",
    )


def load_e_samples(data_dir: Path, momenta: list) -> np.ndarray:
    """
    读取各动量的 E jackknife 样本.

    参数:
    -----
    data_dir: fit_p<mom>.npz 所在目录.
    momenta: 格点动量整数列表.

    输出:
    -----
    e: E 样本, 格点单位, para: sample, momentum.
    """
    data_files = [data_dir / f"fit_p{momentum}.npz" for momentum in momenta]
    # para: momentum, sample
    e_list = [np.load(data_file)["E0"] for data_file in data_files]
    print(f"last input: {data_files[-1]}")

    # para: sample, momentum
    return np.asarray(e_list, dtype=float).T


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Config = init()
    data_dir = Config.data_file.parent

    # 1. 读数据
    # para: sample, momentum
    e = load_e_samples(data_dir, Config.momenta)

    # 2. 拟合, 逐样本结果和 ymean 结果分开保存.
    if Config.use_p0:
        fit_momenta = Config.momenta
        # para: sample, momentum
        e_fit = e
    else:
        fit_momenta = Config.momenta[1:]
        # para: sample, momentum
        e_fit = e[:, 1:]
    samples, ymean = fit_dispersion(
        Config.conf_short, Config.model, fit_momenta, e_fit)
    save_npz(Config.data_file, name="dispersion fit", **samples)
    save_npz(Config.data_file.with_name(f"{Config.data_file.stem}_ymean.npz"),
             name="dispersion fit mean", **ymean)

    # 3. 报表: 一行总体信息, 再逐参数输出逐样本的拟合结果, 最后是物理质量和日期.
    # 格点单位到 GeV 的换算因子.
    energy_unit = 0.1973 / get_conf(Config.conf_short)["lattice_spacing"]
    # para: sample
    mass = samples["m"] * energy_unit
    lines = [
        f"dispersion fit, conf_short={Config.conf_short}, "
        f"model={Config.model}, use_p0={Config.use_p0}, "
        f"chi2/dof={ymean['chi2dof']:.3g}",
        f"mass= {format_mean_err(mass.mean(), sem(mass, jackknife=True))} GeV",
        "sample mean(error):",
    ]
    for name in samples:
        lines.append(
            f"{name}: "
            f"{format_mean_err(samples[name].mean(), sem(samples[name], jackknife=True))}")
    lines.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    save_txt(Config.report_file, "\n".join(lines) + "\n")

    # 4. 画图
    plot_dispersion(
        Config.conf_short, Config.model, Config.use_p0, Config.momenta,
        e, samples, Config.figure_file)

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s")
