#!/public/home/huangcl/.venv/bin/python
"""从已保存的 3pt_disc 和 2pt 计算指定类型的 ratio 并画图.

输入:
    01_data/<conf_short>/P<P>/corr.npz, keys: 3pt_disc, 2pt.

输出:
    01_data/<conf_short>/P<P>/ratio_type<type>.npy.
    02_fig/<conf_short>/P<P>/ratio_type<type>_ratio_z<z>.png.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
project_dir = here.parents[1]
sys.path.insert(0, str(project_dir / "98_tools"))
from analysis_tools import calc_momentum_GeV, sem  # noqa: E402  # type: ignore
from figure_tools import plot_errbar  # noqa: E402  # type: ignore
from output_tools import save_npy  # noqa: E402  # type: ignore
from ratio_tools import (  # noqa: E402  # type: ignore
    calc_c3_divide_c2, calc_c3_z_divide_c3_front_z,
    calc_c3_z_divide_c3_z0,
)


@dataclass
class RatioConfig:
    """ratio 与画图参数."""
    conf_short: str
    p: int
    ratio_type: int
    dt_list: list[int]
    z_list: list[int]
    xlim: list[float]
    ylim: list[float]
    data_dir: Path
    fig_dir: Path


def init() -> RatioConfig:
    """
    初始化运行参数.

    输出:
    -----
    Calc: ratio 与画图参数.
    """
    CONF_SHORT = "c24p29"
    MOM = 5
    RATIO_TYPE = 2
    Z_LIST = [0, 1, 5, 10, 15]
    XLIM = [-7, 7]
    PARAMS = {
        0: {"dt_list": list(range(6, 10)), "ylim": [0, 1.2]},
        1: {"dt_list": list(range(6, 10)), "ylim": [0, 1.2]},
        2: {"dt_list": list(range(6, 10)), "ylim": [0, 1.2]},
        3: {"dt_list": list(range(6, 14)), "ylim": [0, 0.6]},
        4: {"dt_list": list(range(7, 12)), "ylim": [0, 1.2]},
        5: {"dt_list": list(range(5, 9)), "ylim": [-0.1, 1.2]},
    }

    plot_params = PARAMS[MOM]
    data_dir = here.parent / "01_data" / CONF_SHORT / f"P{MOM}"
    fig_dir = here.parent / "02_fig" / CONF_SHORT / f"P{MOM}"

    return RatioConfig(
        conf_short=CONF_SHORT,
        p=MOM,
        ratio_type=RATIO_TYPE,
        dt_list=plot_params["dt_list"],
        z_list=Z_LIST,
        xlim=XLIM,
        ylim=plot_params["ylim"],
        data_dir=data_dir,
        fig_dir=fig_dir,
    )


def plot_ratio_per_z(
        ratio_z: np.ndarray, fig_path: Path, title: str, ylabel: str,
        xlim: list[float], ylim: list[float], dt_list: list[int]) -> None:
    """
    按 tsep 分组绘制单个 z 的 ratio.

    参数:
    -----
    ratio_z: ratio 数据, para: sample, dt, dtau.
    fig_path: 图片保存路径.
    title: 图题.
    ylabel: 纵轴标签.
    xlim: 横轴范围.
    ylim: 纵轴范围.
    dt_list: 需要画的 tsep 列表.
    """
    ratio_mean = ratio_z.mean(axis=0)
    ratio_err = sem(ratio_z)
    raw_data = {}
    for _dt in dt_list:
        tau_vals = np.arange(_dt + 1)
        x_vals = tau_vals - _dt / 2.0
        raw_data[f"tsep={_dt}"] = (
            x_vals, ratio_mean[_dt, tau_vals], ratio_err[_dt, tau_vals])

    plot_errbar(
        raw_data, fig_path,
        xlabel="t_ins - t_sep/2",
        ylabel=ylabel,
        xlim=xlim, ylim=ylim, title=title,
        x_offset=0.05, show_ygrid=True,
    )


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    Calc = init()
    print("conf_short:", Calc.conf_short)
    print("P:", Calc.p)
    print("ratio_type:", Calc.ratio_type)
    print("data dir:", Calc.data_dir)

    data_path = Calc.data_dir / "corr.npz"
    print(f"loading 3pt_disc and 2pt from: {data_path}")
    with np.load(data_path) as data:
        corr3_disc = data["3pt_disc"]
        corr2 = data["2pt"]

    if Calc.ratio_type == 0:
        ratio = calc_c3_divide_c2(corr3_disc, corr2)
        ylabel = "3pt / 2pt"
    elif Calc.ratio_type == 1:
        ratio = calc_c3_z_divide_c3_z0(corr3_disc, corr2)
        ylabel = "3pt(z) / 3pt(z=0)"
    elif Calc.ratio_type == 2:
        ratio = calc_c3_z_divide_c3_front_z(corr3_disc, corr2)
        ylabel = "3pt(z) / 3pt(z-1)"
    else:
        raise ValueError(f"unsupported ratio_type: {Calc.ratio_type}")

    prefix = f"ratio_type{Calc.ratio_type}"
    save_npy(Calc.data_dir / f"{prefix}.npy", ratio, name=prefix)

    p_gev = calc_momentum_GeV(Calc.p, Calc.conf_short)
    for _z in Calc.z_list:
        ratio_z = ratio[:, :, :, _z]
        title = (
            f"{prefix}, z={_z}, |P|={p_gev:.3g}GeV, {Calc.conf_short}")
        fig_path = Calc.fig_dir / (
            f"{prefix}_{Calc.conf_short}_P{Calc.p}_z{_z}.png")
        plot_ratio_per_z(
            ratio_z, fig_path, title, ylabel,
            Calc.xlim, Calc.ylim, Calc.dt_list)
    del corr3_disc, corr2, ratio

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s, "
          f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
