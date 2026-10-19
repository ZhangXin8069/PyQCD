#!/public/home/huangcl/.venv/bin/python
"""
4_plot.py

读取 3_fit_all.py 的拟合结果, 画三类图:
    1. ymean 的 chi2/dof 随 z 变化, 三条曲线为 cand/left/right;
    2. c0 的统计平均 +- 标准误随 z 变化, 三条曲线为 cand/left/right;
    3. 每个 z 的 FH errbar 图, 叠加选中窗口的 c0 +- 1 sigma 灰色带.
每张图有独立的画图配置, 只暴露横纵坐标范围 xlim 和 ylim 供调整.

输入:
    - fh: <root>/01_result/data/c32p29/fh_evt10.npy
    - 拟合结果: <root>/01_result/data/c32p29/fit_evt10_{left,cand,right}.npz
      与 fit_evt10_{left,cand,right}_ym.npz, 以及选中窗口 fit_window_evt10.npy
输出:
    - 总览图: <root>/01_result/fig/chi2dof_z.png, c0_z.png
    - 逐 z 图: <root>/01_result/fig/fit_z{z}_FH_errbar.png
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- 从相邻的项目目录 98_tools 导入通用工具函数 ----

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "98_tools"))

from analysis_tools import (  # noqa: E402  # type: ignore
    plot_errbar, plot_scatter, sem)


@dataclass(frozen=True)
class ReadConfig:
    """读取相关的配置和路径."""

    conf_short: str
    evt: int
    nex: int                # fh 第 1 轴中使用的 nex 索引
    fh_path: Path           # fh 数据文件路径
    data_dir: Path          # 拟合结果所在目录


@dataclass(frozen=True)
class Chi2dofPlotConfig:
    """chi2/dof 随 z 变化总览图的配置."""

    xlim: list              # 横轴范围
    ylim: list              # 纵轴范围


@dataclass(frozen=True)
class C0PlotConfig:
    """c0 随 z 变化总览图的配置."""

    xlim: list              # 横轴范围
    ylim: list              # 纵轴范围


@dataclass(frozen=True)
class FhPlotConfig:
    """逐 z FH errbar 图的配置."""

    xlim: list              # 横轴范围
    ylim: list              # 纵轴范围


@dataclass(frozen=True)
class SaveConfig:
    """输出相关的配置和路径."""

    fig_dir: Path           # 图片保存目录


def init_config() -> tuple:
    """初始化绘图参数和路径.

    CHI2DOF_XLIM 等全大写变量为可给用户修改的配置参数.
    返回: Read, Chi2dofPlot, C0Plot, FhPlot, Save.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"
    EVT = 10
    NEX = 2

    CHI2DOF_XLIM = [-0.5, 17.5]
    CHI2DOF_YLIM = [0, 2]

    C0_XLIM = [-0.5, 17.5]
    C0_YLIM = [-0.1, 1.3]

    FH_XLIM = [2.5, 16.5]
    FH_YLIM = [0, None]

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "01_result" / "data" / CONF_SHORT

    Read = ReadConfig(
        conf_short=CONF_SHORT,
        evt=EVT,
        nex=NEX,
        fh_path=data_dir / f"fh_evt{EVT}.npy",
        data_dir=data_dir,
    )
    Chi2dofPlot = Chi2dofPlotConfig(
        xlim=CHI2DOF_XLIM,
        ylim=CHI2DOF_YLIM,
    )
    C0Plot = C0PlotConfig(
        xlim=C0_XLIM,
        ylim=C0_YLIM,
    )
    FhPlot = FhPlotConfig(
        xlim=FH_XLIM,
        ylim=FH_YLIM,
    )
    Save = SaveConfig(fig_dir=root_dir / "01_result" / "fig")

    return Read, Chi2dofPlot, C0Plot, FhPlot, Save


def load_data(Read: ReadConfig) -> tuple:
    """读取 fh 数据和拟合结果.

    输出:
    -----
    fh: 第 0 轴为 sample, 第 1 轴为 nex, 第 2 轴为 dt, 第 3 轴为 z.
    sample: 角色到逐样本拟合结果 dict 的映射, 每个数组第 0 轴为 sample, 第 1 轴为 z.
    ymean: 角色到均值拟合结果 dict 的映射, 每个数组第 0 轴为 z.
    window: 第 0 轴为 z, 第 1 轴为 (t_start, t_end).
    """

    fh = np.load(Read.fh_path)
    print(f"loaded fh: {Read.fh_path}, shape={fh.shape}")

    roles = ("left", "cand", "right")
    sample = {}
    ymean = {}
    for role in roles:
        sample_path = Read.data_dir / f"fit_evt{Read.evt}_{role}.npz"
        ymean_path = Read.data_dir / f"fit_evt{Read.evt}_{role}_ym.npz"
        sample[role] = dict(np.load(sample_path))
        ymean[role] = dict(np.load(ymean_path))
        print(f"loaded fit result: {sample_path}")
        print(f"loaded fit result: {ymean_path}")

    window_path = Read.data_dir / f"fit_window_evt{Read.evt}.npy"
    window = np.load(window_path)
    print(f"loaded window: {window_path}, shape={window.shape}")

    return fh, sample, ymean, window


def plot_chi2dof_z(
        ymean: dict, Read: ReadConfig, Plot: Chi2dofPlotConfig,
        Save: SaveConfig) -> None:
    """画 ymean chi2/dof 随 z 变化的总览图."""

    # 横坐标直接取自拟合结果中保存的 z
    x = np.asarray(ymean["cand"]["z"], dtype=float)
    y_data = {
        role: ymean[role]["chi2dof"] for role in ("left", "cand", "right")}
    figure_path = Save.fig_dir / "chi2dof_z.png"
    plot_scatter(
        x, y_data, figure_path,
        xlabel="z", ylabel="chi2/dof",
        xlim=Plot.xlim, ylim=Plot.ylim,
        title="chi2/dof")


def plot_c0_z(
        sample: dict, Read: ReadConfig, Plot: C0PlotConfig,
        Save: SaveConfig) -> None:
    """画 c0 统计平均 +- 标准误随 z 变化的总览图."""

    # 横坐标直接取自拟合结果中保存的 z
    x = np.asarray(sample["cand"]["z"], dtype=float)
    y_data = {}
    for role in ("left", "cand", "right"):
        # para: sample, z
        c0 = sample[role]["c0"]
        y_data[role] = (c0.mean(axis=0), sem(c0))
    figure_path = Save.fig_dir / "c0_z.png"
    plot_errbar(
        x, y_data, figure_path,
        xlabel="z", ylabel="c0",
        xlim=Plot.xlim, ylim=Plot.ylim,
        title="c0")


def plot_fh_z(
        fh: np.ndarray, sample: dict, window: np.ndarray, Read: ReadConfig,
        Plot: FhPlotConfig, Save: SaveConfig) -> None:
    """逐个 z 画 FH errbar 和选中窗口的 c0 +- 1 sigma 灰色带."""

    # para: dt
    t_vals = np.arange(fh.shape[2], dtype=int)
    # z 的取值范围和顺序直接取自拟合结果中保存的 z
    z_arr = np.asarray(sample["cand"]["z"], dtype=int)
    for z_idx, z in enumerate(z_arr):
        # para: sample, dt
        fh_z = fh[:, Read.nex, :, z]
        y_data = {"FH": (fh_z.mean(axis=0), sem(fh_z))}
        figure_path = Save.fig_dir / f"fit_z{z}_FH_errbar.png"

        # para: sample
        c0 = sample["cand"]["c0"][:, z_idx]
        c0_mean = c0.mean()
        c0_err = sem(c0)
        t_start, t_end = window[z_idx]

        if np.isfinite(t_start) and np.isfinite(t_end):
            band_x = np.arange(t_start, t_end + 1, dtype=float)
            band_y_down = np.full_like(band_x, c0_mean - c0_err)
            band_y_up = np.full_like(band_x, c0_mean + c0_err)
            title = (
                f"{Read.conf_short}, evt={Read.evt}, z={z}, "
                f"window=[{int(t_start)}, {int(t_end)}]")
            plot_errbar(
                t_vals, y_data, figure_path,
                xlabel="t", ylabel="FH",
                xlim=Plot.xlim, ylim=Plot.ylim,
                title=title,
                show_band=True,
                band_x=band_x,
                band_y_down=band_y_down,
                band_y_up=band_y_up,
                band_color="gray",
                band_label="c0 +- 1 sigma")
        else:
            title = f"{Read.conf_short}, evt={Read.evt}, z={z}, no window"
            plot_errbar(
                t_vals, y_data, figure_path,
                xlabel="t", ylabel="FH",
                xlim=Plot.xlim, ylim=Plot.ylim,
                title=title)


if __name__ == "__main__":
    Read, Chi2dofPlot, C0Plot, FhPlot, Save = init_config()
    Save.fig_dir.mkdir(parents=True, exist_ok=True)

    fh, sample, ymean, window = load_data(Read)

    plot_chi2dof_z(ymean, Read, Chi2dofPlot, Save)
    plot_c0_z(sample, Read, C0Plot, Save)
    plot_fh_z(fh, sample, window, Read, FhPlot, Save)
