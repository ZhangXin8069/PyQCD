#!/public/home/huangcl/.venv/bin/python
"""比较不同 evt 的 ratio GEVP 结果.

输入:
    - ratio 数据: 本目录 01_result/data/<conf_short>/ratio_tref<tref>_evt<evt>.npy
输出:
    - 每个 z, tsep 一张 evt 比较图, 保存到本目录 01_result/fig/<conf_short>.
    - z=0 各 tsep 不同 evt 的均值与误差表, 保存到 01_result/data/<conf_short>.
"""

from dataclasses import dataclass
import re
from pathlib import Path
import sys

import numpy as np
from prettytable import PrettyTable

# ---- 从项目目录下的 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))

from analysis_tools import format_mean_err, plot_errbar, sem  # noqa: E402  # type: ignore
from output_tools import save_txt  # noqa: E402  # type: ignore


@dataclass(frozen=True)
class ReadConfig:
    """读取相关的配置和路径."""

    conf_short: str
    tref: int
    data_dir: Path          # ratio 数据输入目录


@dataclass(frozen=True)
class PlotConfig:
    """画图相关的配置和路径."""

    z_list: tuple
    dt_list: tuple
    xlim: tuple
    ylim: tuple
    figure_dir: Path        # 图片输出目录


def init() -> tuple[ReadConfig, PlotConfig]:
    """初始化 evt 比较分析的参数和路径.

    CONF_SHORT 等全大写变量为可给用户修改的配置参数.
    返回: Read, Plot.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"
    TREF = 5
    Z_LIST = (0, 1, 2)
    DT_LIST = (9, 10, 11, 12)
    XLIM = (-7.0, 7.0)
    YLIM = (0.0, 1.3)

    output_dir = Path(__file__).resolve().parent / "01_result"

    Read = ReadConfig(
        conf_short=CONF_SHORT,
        tref=TREF,
        data_dir=output_dir / "data" / CONF_SHORT,
    )
    Plot = PlotConfig(
        z_list=Z_LIST,
        dt_list=DT_LIST,
        xlim=XLIM,
        ylim=YLIM,
        figure_dir=output_dir / "fig" / CONF_SHORT,
    )

    return Read, Plot


def load_ratios(Read: ReadConfig) -> dict[int, np.ndarray]:
    """读取所有 evt 的 ratio 数据.

    输入:
    -----
    Read 的 data_dir 为 ratio 数据目录, 文件名形如 ratio_tref5_evt6.npy.

    输出:
    -----
    ratios: {evt: ratio}, ratio 的 para: sample, dt, dtau, z.
    """

    pattern = f"ratio_tref{Read.tref}_evt*.npy"
    ratio_files = sorted(
        Read.data_dir.glob(pattern),
        key=lambda p: int(re.search(r"evt(\d+)", p.stem).group(1)),
    )
    if not ratio_files:
        raise FileNotFoundError(
            f"not found: {Read.data_dir / pattern}")

    ratios = {}
    for path in ratio_files:
        evt = int(re.search(r"evt(\d+)", path.stem).group(1))
        ratios[evt] = np.load(path)
        print(f"loaded ratio: {path}, shape={ratios[evt].shape}")
    return ratios


def report_cmp(ratios: dict[int, np.ndarray], Read: ReadConfig, Plot: PlotConfig) -> None:
    """输出 z=0 各 tsep 不同 evt 的 ratio 均值与误差报告.

    输入 ratios: {evt: ratio}, ratio 的 para: sample, dt, dtau, z.
    每个 tsep 一张表, 同一 evt 在同一行, 同一 dtau 在同一列.
    """

    lz = 0

    lines = [f"cmp evt ratio report, tref={Read.tref}, z={lz}"]
    for dt in Plot.dt_list:
        # dtau 个数为奇数时取中间 3 个, 偶数时取中间 2 个.
        if (dt + 1) % 2 == 1:
            tau_list = np.arange(dt // 2 - 1, dt // 2 + 2)
        else:
            tau_list = np.arange(dt // 2, dt // 2 + 2)
        table = PrettyTable()
        table.field_names = ["evt"] + [f"dtau={d}" for d in tau_list]
        for evt, ratio in ratios.items():
            row = [f"evt={evt}"]
            for tau in tau_list:
                mean = ratio[:, dt, tau, lz].mean(axis=0)
                err = sem(ratio[:, dt, tau, lz])
                row.append(format_mean_err(mean, err, 5))
            table.add_row(row)
        lines += ["", f"tsep={dt}", *table.get_string().splitlines()]

    report_path = Read.data_dir / f"cmp_evt_tref{Read.tref}_z{lz}.txt"
    save_txt(report_path, "\n".join(lines) + "\n", name="cmp_report")
    print("\n".join(lines))


def plot_cmp(ratios: dict[int, np.ndarray], Read: ReadConfig, Plot: PlotConfig) -> None:
    """按 z 和 tsep 逐张画不同 evt 的 ratio 比较图.

    输入 ratios: {evt: ratio}, ratio 的 para: sample, dt, dtau, z.
    """

    for z in Plot.z_list:
        for dt in Plot.dt_list:
            # 同一 tsep 下各 evt 共用同一条横轴.
            tau_vals = np.arange(0, dt + 1)
            x_vals = tau_vals - dt / 2.0

            # para: 标签 -> (mean, err)
            y_data = {}
            for evt, ratio in ratios.items():
                y_data[f"evt={evt}"] = (
                    ratio[:, dt, tau_vals, z].mean(axis=0),
                    sem(ratio[:, dt, tau_vals, z]),
                )

            figure_path = (
                Plot.figure_dir
                / f"cmp_evt_ratio_tref{Read.tref}_z{z}_tsep{dt}_{Read.conf_short}.png"
            )
            plot_errbar(
                x_vals,
                y_data,
                figure_path,
                xlabel=r"$t_{\mathrm{ins}} - t_{\mathrm{sep}}/2$",
                ylabel=r"$C_3 / C_2$",
                xlim=Plot.xlim,
                ylim=Plot.ylim,
                title=f"GEVP state0, ave dirs, z={z}, tsep={dt}, P(0,0,0)",
            )


if __name__ == "__main__":
    Read, Plot = init()

    # para: sample, dt, dtau, z
    ratios = load_ratios(Read)

    report_cmp(ratios, Read, Plot)
    # plot_cmp(ratios, Read, Plot)
