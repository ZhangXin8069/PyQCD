#!/public/home/huangcl/.venv/bin/python
"""fh 拟合结果的画图.

功能:
    集中本功能目录的全部画图函数和画图配置:
        1. plot_fh: 逐 z 画不同 nex 的 fh 误差棒图. 不传拟合数据时只画
           nex 曲线, 传入拟合数据和窗口时额外叠加 c0 +- 1 sigma 灰色带.
        2. plot_chi2dof_z: ymean 的 chi2/dof 随 z 变化, 三条曲线为
           cand/left/right.
        3. plot_c0_z: 从 fit_cand.npz 或 hB.npy 读取裸矩阵元并画 z 图.
    主执行入口读取拟合结果后画全部总览图和逐 z 图.

输入:
    - fh: 01_data/<conf_short>/P<P>/fh.npy
    - 拟合结果: 01_data/<conf_short>/P<P>/fit_{left,cand,right}.npz
      与 fit_{left,cand,right}_ym.npz, 以及选中窗口 fit_window.npy
输出:
    - 总览图: 02_fig_rep/<conf_short>/P<P>/chi2dof_z_<conf_short>_P<P>.png,
      c0_z_<conf_short>_P<P>.png
    - 逐 z 图: 02_fig_rep/<conf_short>/P<P>/fh_<conf_short>_P<P>_z<z>.png
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

here = Path(__file__).resolve()
sys.path.append("/public/home/huangcl/07_pion_gluon_pdf/98_tools")

from analysis_tools import calc_momentum_GeV, sem  # noqa: E402  # type: ignore
from figure_tools import plot_errbar, plot_scatter  # noqa: E402  # type: ignore
from output_tools import format_mean_err  # noqa: E402  # type: ignore


@dataclass
class PlotConfig:
    """画图相关的配置和路径."""
    conf_short: str
    p: int                  # 动量大小 (正值)
    p_list: list[int]       # 组态级对比图需要画的动量列表
    z_list: list[int]       # 需要画的 z 列表
    nex: int                # 拟合色带对应的 nex
    fh_xlim: list[float]    # fh 图横轴范围
    fh_ylim: list[float]    # fh 图纵轴范围
    z_xlim: list[float]     # vs z 图共用的横轴范围
    c0_ylim: list[float]    # hB(z) 图纵轴范围
    conf_dir: Path          # 组态级图片目录, 精确到组态短号
    mom_dir: Path           # 动量级图片目录, 精确到组态短号和动量
    data_dir: Path          # fh 数据和拟合结果所在目录


def init(conf_short=None, p=None):
    """
    初始化绘图参数和路径.

    可选项:
    -----
    conf_short: 组态短名; 为 None 时使用默认值.
    p: 动量大小; 为 None 时使用默认值.

    输出:
    -----
    Plot: PlotConfig 实例.
    """

    CONF_SHORT = "c32p29"
    MOM = 4
    NEX = 1
    Z_LIST = [0, 1, 5, 10, 15]
    FH_XLIM = [2.5, 16.5]
    FH_YLIM = [-0.5, 3]
    Z_XLIM = [-0.5, 17.5]
    C0_YLIM = [-0.1, 0.7]

    # 各组态用于组态级对比图的动量列表, 键为组态短号
    P_LIST = {
        "c24p29": [0, 8],
        "c32p29": [4],
    }

    conf_short = CONF_SHORT if conf_short is None else conf_short
    p = MOM if p is None else p
    p_list = P_LIST[conf_short]

    data_dir = here.parent / "01_data" / conf_short / f"P{p}"
    fig_dir = here.parent / "02_fig_rep" / conf_short

    Plot = PlotConfig(
        conf_short=conf_short,
        p=p,
        p_list=p_list,
        z_list=Z_LIST,
        nex=NEX,
        fh_xlim=FH_XLIM,
        fh_ylim=FH_YLIM,
        z_xlim=Z_XLIM,
        c0_ylim=C0_YLIM,
        conf_dir=fig_dir,
        mom_dir=fig_dir / f"P{p}",
        data_dir=data_dir,
    )
    return Plot


def make_fig_title(
        name: str, conf_short: str, z: int = None, p: int = None) -> str:
    """
    生成图题, 供本功能目录的全部图共用.

    参数:
    -----
    name: 图的名称, 写在图题最前面.
    conf_short: 组态短标识, 写在图题最后.

    可选项:
    -----
    z: 当前图的 z; 为 None 时不写 z 段.
    p: 动量大小 (正值); 为 None 时不写动量段.

    输出:
    -----
    title: 形如 'Feynman-Hellmann, z=1, |P|=0GeV, c24p29' 的图题.
    """

    title = name
    if z is not None:
        title += f", z={z}"
    if p is not None:
        # 格点动量整数换算为物理动量, 单位 GeV
        p_gev = calc_momentum_GeV(p, conf_short)
        title += f", |P|={p_gev:.3g}GeV"
    title += f", {conf_short}"
    return title


def plot_fh(conf_short=None, p=None, with_fit: bool = False) -> None:
    """
    自动读取 fh 数据, 按画图配置中的 z_list 逐个 z 画图.

    可选项:
    -----
    conf_short: 组态短名; 为 None 时使用 init() 中的默认值.
    p: 动量大小; 为 None 时使用 init() 中的默认值.
    with_fit: 是否叠加 c0 +- 1 sigma 灰色带, 默认 False.

    说明:
    with_fit 为 False 时只画 nex 曲线, 供 1_calc_fh 调用; 为 True 时额外
    读取 fit_cand.npz 的 c0 与 fit_window.npy, 叠加灰色带.
    """

    Plot = init(conf_short, p)

    fh_path = Plot.data_dir / "fh.npy"
    # para: sample, nex, dt, z
    fh = np.load(fh_path)
    print(f"loaded fh: {fh_path}, shape={fh.shape}")

    # z_list 不能超过 fh 的 z 轴长度, 超出直接报错
    for _z in Plot.z_list:
        if _z >= fh.shape[3]:
            raise ValueError(
                f"z={_z} in z_list exceeds Nz={fh.shape[3]} of {fh_path}")

    # 画灰色带时读取 c0 和选中窗口
    window = None
    c0_all = None
    if with_fit:
        sample_path = Plot.data_dir / "fit_cand.npz"
        # para: sample, z
        c0_all = np.load(sample_path)["c0"]
        print(f"loaded fit result: {sample_path}")

        window_path = Plot.data_dir / "fit_window.npy"
        # para: z, (t_start, t_end)
        window = np.load(window_path)
        print(f"loaded window: {window_path}, shape={window.shape}")

    for _z in Plot.z_list:
        # para: sample, nex, dt
        fh_z = fh[:, :, :, _z]

        c0 = None
        _window = None
        if with_fit:
            # para: sample
            c0 = c0_all[:, _z]
            _window = window[_z]

        fig_path = Plot.mom_dir / \
            f"fh_{Plot.conf_short}_P{Plot.p}_z{_z}.png"
        plot_fh_one(fh_z, _z, fig_path, Plot, c0, _window)


def plot_fh_one(
        fh_z: np.ndarray, z: int, fig_path: Path, Plot: PlotConfig,
        c0: np.ndarray = None, window: list = None) -> None:
    """
    画单个 z 的不同 nex 的 fh 误差棒图.

    参数:
    -----
    fh_z: para: sample, nex, dt.
    z: 当前图的 z.
    fig_path: 图片保存路径, 含文件名和后缀.
    Plot: 画图配置, 提供横纵轴范围和组态标识.

    可选项:
    -----
    c0: para: sample; 画灰色带时使用.
    window: 选中窗口 (t_start, t_end); 为 None 或含 nan 时不画灰色带.
    """

    # para: dt
    t_vals = np.arange(fh_z.shape[2], dtype=int)

    # para: 标签 -> (dt, mean, err)
    y_data = {}
    for _nex in range(fh_z.shape[1]):
        # para: sample, dt
        fh_nex = fh_z[:, _nex, :]
        y_data[f"nex={_nex}"] = (t_vals, fh_nex.mean(axis=0), sem(fh_nex))

    # 灰色带: 标签 -> (dt, mean, err)
    fit_data = None
    if window is not None and np.all(np.isfinite(window)):
        # para: sample
        c0_mean = c0.mean()
        c0_err = sem(c0)
        t_start, t_end = window
        # para: dt
        band_x = np.arange(t_start, t_end + 1, dtype=float)
        fit_band = (
            band_x,
            np.full_like(band_x, c0_mean),
            np.full_like(band_x, c0_err),
        )
        fit_data = {
            (f"c0: {format_mean_err(c0_mean, c0_err)}"
             if _nex == Plot.nex else f"nex={_nex}"): (
                fit_band if _nex == Plot.nex else None)
            for _nex in range(fh_z.shape[1])
        }

    plot_errbar(
        y_data, fig_path,
        xlabel="t/a", ylabel="fh",
        xlim=Plot.fh_xlim, ylim=Plot.fh_ylim,
        title=make_fig_title("Feynman-Hellmann", Plot.conf_short, z, Plot.p),
        x_offset=0.2,
        fit_data=fit_data,
        fit_x_offset=0.2,
    )


def plot_chi2dof_z(Plot: PlotConfig, cmp: bool = False) -> None:
    """
    画 ymean chi2/dof 随 z 变化图.

    参数:
    -----
    Plot: 画图配置, 提供横轴范围, 组态标识和数据目录.
    cmp: 是否同时画 cand, left, right, 默认 False.
    """

    roles = ("left", "cand", "right") if cmp else ("cand",)
    # para: 角色 -> z
    y_data = {}
    x = None
    for role in roles:
        ymean_path = Plot.data_dir / f"fit_{role}_ym.npz"
        # para: z
        _chi2dof = np.load(ymean_path)["chi2dof"]
        y_data[role] = _chi2dof
        x = np.arange(_chi2dof.shape[0], dtype=float)
        print(f"loaded fit result: {ymean_path}")

    prefix = "cmp_" if cmp else ""
    figure_path = Plot.mom_dir / \
        f"{prefix}chi2dof_z_{Plot.conf_short}_P{Plot.p}.png"
    plot_scatter(
        x, y_data, figure_path,
        xlabel="z/a", ylabel="chi2/dof",
        xlim=Plot.z_xlim, ylim=[0, 2], show_grid=True,
        show_hline=True, hline_y=1,
        title=make_fig_title("chi2/dof", Plot.conf_short, p=Plot.p))


def plot_c0_z(
        Plot: PlotConfig, norm: bool = False, cmp: bool = False) -> None:
    """
    画归一化或绝对归一化的裸矩阵元 hB 随 z 变化图.

    参数:
    -----
    Plot: 画图配置, 提供横轴范围, 组态标识和数据目录.
    norm: 是否画 hB(z)/hB(0), 默认 False.
    cmp: 归一化时是否同时画 cand, left, right, 默认 False.
    """

    if norm:
        roles = ("left", "cand", "right") if cmp else ("cand",)
        y_data = {}
        for role in roles:
            sample_path = Plot.data_dir / f"fit_{role}.npz"
            # para: sample, z
            hB = np.load(sample_path)["c0"]
            # para: z
            x = np.arange(hB.shape[1], dtype=float)
            y_data[role] = (x, hB.mean(axis=0), sem(hB))
            print(f"loaded fit result: {sample_path}")
        ylabel = "h_B(z)/h_B(0)"
        ylim = [-0.1, 1.1]
        suffix = "norm"
    else:
        sample_path = Plot.data_dir / "hB.npy"
        # para: sample, z
        hB = np.load(sample_path)
        # para: z
        x = np.arange(hB.shape[1], dtype=float)
        y_data = {"hB": (x, hB.mean(axis=0), sem(hB))}
        ylabel = "h_B(z)"
        ylim = Plot.c0_ylim
        suffix = "bare"
        print(f"loaded hB: {sample_path}")

    prefix = "cmp_" if cmp else ""
    figure_path = Plot.mom_dir / \
        f"{prefix}hB_z_{suffix}_{Plot.conf_short}_P{Plot.p}.png"
    plot_errbar(
        y_data, figure_path,
        xlabel="z/a", ylabel=ylabel,
        xlim=Plot.z_xlim, ylim=ylim,
        title=make_fig_title(f"{suffix} matrix", Plot.conf_short, p=Plot.p))


def plot_one_p(conf_short=None, p=None, cmp: bool = False) -> None:
    """
    画单个动量的全部三类拟合相关图, 各图自行读取所需数据.

    可选项:
    -----
    conf_short: 组态短名; 为 None 时使用 init() 中的默认值.
    p: 动量大小; 为 None 时使用 init() 中的默认值.
    cmp: chi2 和归一化 hB 图是否画 cand, left, right 比较, 默认 False.

    说明:
    依次画 chi2/dof 随 z 变化图, c0 随 z 变化图, 以及带 c0 +- 1 sigma
    灰色带的 fh 图. 供 3_fit_all 拟合结束后直接调用.
    """

    Plot = init(conf_short, p)

    print(f"conf_short: {Plot.conf_short}, P: {Plot.p}")

    plot_chi2dof_z(Plot, cmp=cmp)
    plot_c0_z(Plot, norm=True, cmp=cmp)

    # 画带灰色带的 fh 图
    plot_fh(Plot.conf_short, Plot.p, with_fit=True)


def plot_c0_z_all_p() -> None:
    """
    画组态级 c0 vs z 对比图, 每个动量画一条 cand 曲线.

    """

    Plot = init()

    y_data = {}
    for _p in Plot.p_list:
        sample_path = here.parent / "01_data" / Plot.conf_short / f"P{_p}" / \
            "hB.npy"
        # para: sample, z
        hB = np.load(sample_path)
        # 数组第 1 轴即 z, 横坐标取连续整数索引
        x = np.arange(hB.shape[1], dtype=float)
        y_data[f"|P|={calc_momentum_GeV(_p, Plot.conf_short):.3g}GeV"] = (
            x, hB.mean(axis=0), sem(hB))
        print(f"loaded hB: {sample_path}")

    figure_path = Plot.conf_dir / f"c0_z_all_p_{Plot.conf_short}.png"
    plot_errbar(
        y_data, figure_path,
        xlabel="z/a", ylabel="h_B(z)",
        xlim=Plot.z_xlim, ylim=Plot.c0_ylim,
        title=make_fig_title("bare matrix", Plot.conf_short))


if __name__ == "__main__":
    t_total_begin = time.perf_counter()

    plot_one_p(cmp=True)
    plot_one_p(cmp=False)
    plot_c0_z(init(), norm=False)
    plot_c0_z_all_p()

    print(f"job done, time={time.perf_counter() - t_total_begin:.0f}s, "
          f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
