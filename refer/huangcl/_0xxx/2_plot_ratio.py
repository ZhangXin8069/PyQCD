#!/public/home/huangcl/.venv/bin/python
"""读取已保存的 ratio 并画图.

功能:
    读取 01_data/<conf_short>/P<P>/ratio.npy, 逐个 z 调用
    ratio_tools.plot_ratio_per_z 画图, 图片保存到 02_fig/<conf_short>/P<P>/.
    既可由 1_ratio.py 计算结束后调用, 也可单独运行.

输出:
    02_fig/<conf_short>/P<P>/ratio_z<z>.png.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
project_dir = here.parents[1]
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import calc_momentum_GeV  # noqa: E402  # type: ignore
from ratio_tools import plot_ratio_per_z  # noqa: E402  # type: ignore


@dataclass
class PlotConfig:
    """画图参数."""
    conf_short: str
    p: int                  # 动量大小 (正值)
    dt_list: list[int]      # 需要画的 t_sep 列表
    z_list: list[int]       # 需要画的 z 列表
    xlim: list[float]       # 横轴范围
    ylim: list[float]       # 纵轴范围
    fig_dir: Path           # 图片输出目录


def init(conf_short=None, p=None):
    """
    初始化画图参数和读写路径.

    参数:
    -----
    conf_short: 组态短名; 为 None 时使用默认值.
    p: 动量大小; 为 None 时使用默认值.

    输出:
    -----
    plot_conf: PlotConfig 实例, 含图片输出目录 fig_dir.
    data_dir: 数据读取目录, 即 01_data/<conf_short>/P<P>.
    """

    CONF_SHORT = "c32p29"
    MOM = 8
    Z_LIST = [0, 2, 8, 16]
    XLIM = [-7, 7]

    # 各动量的画图参数, 键为格点动量整数
    PARAMS = {
        0: {"dt_list": list(range(5, 13)), "ylim": [0, 1.5]},
        8: {"dt_list": list(range(5, 10)), "ylim": [0, 1.5]},
    }

    conf_short = CONF_SHORT if conf_short is None else conf_short
    p = MOM if p is None else p

    params = PARAMS[p]

    # 数据从 01_data 读, 图片写到 02_fig
    data_dir = here.parent / "01_data" / conf_short / f"P{p}"
    fig_dir = here.parent / "02_fig" / conf_short / f"P{p}"

    plot_conf = PlotConfig(
        conf_short=conf_short,
        p=p,
        dt_list=params["dt_list"],
        z_list=Z_LIST,
        xlim=XLIM,
        ylim=params["ylim"],
        fig_dir=fig_dir,
    )
    return plot_conf, data_dir


def plot_saved_ratio(conf_short=None, p=None):
    """
    读取已保存的 ratio, 逐个 z 画图.

    可选项:
    -----
    conf_short: 组态短名; 为 None 时使用 init() 中的默认值.
    p: 动量大小; 为 None 时使用 init() 中的默认值.

    输出:
    -----
    图片保存到 02_fig/<conf_short>/P<P>/ratio_z<z>.png.
    """

    Plot, data_dir = init(conf_short, p)

    # 格点动量整数换算为物理动量, 单位 GeV
    p_gev = calc_momentum_GeV(Plot.p, Plot.conf_short)

    path = data_dir / "ratio.npy"
    print(f"loading ratio from: {path}")
    # para: sample, dt, dtau, z
    ratio = np.load(path)
    print(f"loaded ratio shape: {ratio.shape}")

    for _z in Plot.z_list:
        # para: sample, dt, dtau
        ratio_z = ratio[:, :, :, _z]
        title = f"ratio, z={_z}, |P|={p_gev:.3g}GeV, {Plot.conf_short}"
        fig_path = Plot.fig_dir / \
            f"ratio_{Plot.conf_short}_P{Plot.p}_z{_z}.png"

        plot_ratio_per_z(
            ratio_z, fig_path, title, Plot.xlim,
            Plot.ylim, Plot.dt_list,
        )


if __name__ == "__main__":
    t_total_begin = time.perf_counter()
    plot_saved_ratio()
    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s, "
          f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
