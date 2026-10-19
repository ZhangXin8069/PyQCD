#!/public/home/huangcl/.venv/bin/python
"""
preprocess_fh.py

从 ratio GEVP 结果读取 evt=10 的 ratio 数据, 计算 fh 变换,
保存 fh 并画出不同 nex 的 fh 对比图.

输入:
    - ratio: /public/home/huangcl/06_deuteron_gluon_pdf/02_ratio_gevp/01_result/data/c32p29/ratio_tref5_evt10.npy
输出:
    - fh 数据: <root>/01_result/data/c32p29/fh.npy
    - fh 图: <root>/01_result/fig/c32p29/fh_c32p29_P0_nex0to2_z{z}.png
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- 从相邻的项目目录 98_tools 导入通用工具函数 ----

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "98_tools"))

from analysis_tools import plot_errbar, sem  # noqa: E402  # type: ignore
from output_tools import save_npy  # noqa: E402  # type: ignore


@dataclass(frozen=True)
class ReadConfig:
    """读取相关的配置和路径."""

    conf_short: str
    tref: int
    evt: int
    ratio_path: Path        # ratio 数据文件路径


@dataclass(frozen=True)
class CalcConfig:
    """fh 变换相关的配置和路径."""

    max_nex: int            # nex 的最大值, 计算 all nex in [0, max_nex]
    fh_dir: Path            # fh 结果保存目录


@dataclass(frozen=True)
class PlotConfig:
    """画图相关的配置和路径."""

    z_list: tuple
    xlim: tuple
    ylim: tuple
    xoffset: float
    figure_dir: Path        # 图片输出目录


def init_config() -> tuple[ReadConfig, CalcConfig, PlotConfig]:
    """初始化 fh 变换分析的参数和路径.

    CONF_SHORT 等全大写变量为可给用户修改的配置参数.
    返回: Read, Calc, Plot.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"
    TREF = 5
    EVT = 10
    MAX_NEX = 2
    Z_LIST = (0, 1, 16, 20)
    XLIM = (2.5, 19.5)
    YLIM = (-0.1, 2.5)
    XOFFSET = 0.2

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "01_result" / "data" / CONF_SHORT
    fig_dir = root_dir / "01_result" / "fig" / CONF_SHORT

    Read = ReadConfig(
        conf_short=CONF_SHORT,
        tref=TREF,
        evt=EVT,
        ratio_path=Path(
            "/public/home/huangcl/06_deuteron_gluon_pdf/02_ratio_gevp/"
            f"01_result/data/{CONF_SHORT}/ratio_tref{TREF}_evt{EVT}.npy"),
    )
    Calc = CalcConfig(
        max_nex=MAX_NEX,
        fh_dir=data_dir,
    )
    Plot = PlotConfig(
        z_list=Z_LIST,
        xlim=XLIM,
        ylim=YLIM,
        xoffset=XOFFSET,
        figure_dir=fig_dir,
    )

    return Read, Calc, Plot


def load_ratio(Read: ReadConfig) -> np.ndarray:
    """读取 ratio 数据.

    输入:
    -----
    Read 的 ratio_path 为 ratio 文件, 文件名形如 ratio_tref5_evt10.npy.

    输出:
    -----
    ratio: 第 0 轴为 sample, 第 1 轴为 t_sep, 第 2 轴为 t_ins, 第 3 轴为 z.
    """

    ratio_path = Read.ratio_path
    if not ratio_path.exists():
        raise FileNotFoundError(f"not found: {ratio_path}")

    # para: sample, dt, dtau, z
    ratio = np.load(ratio_path)
    print(f"loaded ratio: {ratio_path}, shape={ratio.shape}")
    return ratio


def calc_fh(ratio: np.ndarray, nex: int) -> np.ndarray:
    """
    计算 fh(t) = sum_{tau=nex}^{t-nex} R(t, tau)
               - sum_{tau=nex}^{t-1-nex} R(t-1, tau).

    两端各去掉 nex 个 tau 点后先做 tau 方向累加, 再沿 t 方向做前向差分.
    前一时刻的累加结果直接取 U(t-1), 对角线上没有多余数据参与累加.

    参数:
    -----
    ratio: 第 0 轴为 sample, 第 1 轴为 dt, 第 2 轴为 dtau, 第 3 轴为 z.
    nex: 两端各去掉的 tau 点数.

    输出:
    -----
    fh: 第 0 轴为 sample, 第 1 轴为 dt, 第 2 轴为 z.
    """

    _, dt_len, _, _ = ratio.shape

    # para: sample, dt, z
    u_sum = np.zeros((ratio.shape[0], dt_len, ratio.shape[3]), dtype=float)
    for dt in range(2 * nex, dt_len):
        u_sum[:, dt] = ratio[:, dt, nex:dt - nex + 1, :].sum(axis=1)

    # para: sample, dt, z
    fh = u_sum - np.roll(u_sum, 1, axis=1)
    return fh


def plot_fh(
        all_fh: np.ndarray, max_nex: int,
        Plot: PlotConfig, Read: ReadConfig) -> None:
    """按 z 画不同 nex 的 fh 对比图.

    输入 all_fh 的 para 为 sample, nex, dt, z, nex 取值为 [0, max_nex].
    """

    _, _, dt_len, Nz = all_fh.shape
    t_vals = np.arange(dt_len)

    for z in Plot.z_list:
        if z >= Nz:
            print(f"    Warning: z={z} exceeds Nz={Nz}, skipping")
            continue

        # para: 标签 -> (mean, err)
        y_data = {}
        for nex in range(max_nex + 1):
            y_data[f"nex={nex}"] = (
                all_fh[:, nex, :, z].mean(axis=0),
                sem(all_fh[:, nex, :, z]),
            )

        figure_path = (
            Plot.figure_dir
            / f"fh_{Read.conf_short}_evt{Read.evt}_z{z}.png"
        )
        plot_errbar(
            t_vals, y_data, figure_path,
            xlabel="t", ylabel="fh",
            xlim=Plot.xlim,
            ylim=Plot.ylim,
            title=f"fh: t0={Read.tref} evt={Read.evt}, z={z}, P=(0,0,0)",
            x_offset=Plot.xoffset,
        )


if __name__ == "__main__":
    Read, Calc, Plot = init_config()

    t_total_begin = time.perf_counter()
    print(
        f"conf_short: {Read.conf_short}, tref: {Read.tref}, "
        f"evt: {Read.evt}, max_nex: {Calc.max_nex}")

    # para: sample, dt, dtau, z
    ratio = load_ratio(Read)

    # para: sample, nex, dt, z
    all_fh = np.stack(
        [calc_fh(ratio, nex) for nex in range(Calc.max_nex + 1)], axis=1)

    fh_path = Calc.fh_dir / f"fh_evt{Read.evt}.npy"
    save_npy(fh_path, all_fh)

    plot_fh(all_fh, Calc.max_nex, Plot, Read)

    print(
        f"job done, total time: {time.perf_counter() - t_total_begin:.2f}s")
