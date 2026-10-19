#!/public/home/huangcl/.venv/bin/python
"""用 GEVP 基态本征矢投影 C2 和 C3, 计算 C3/C2 ratio.

输入:
    - 2pt: /public/home/huangcl/00_data/00_NP_2pt/mom0/c32p29/<conf_id>/NPcorr_conf<conf_id>_P0.npy
    - OPE: /public/home/chenc/gluon_unpo/generating_data/operator/.../ops_sum_dz30_conf<conf_id>.npz
    - 基态本征矢: 01_gevp/01_result/c32p29/eigvec/state0_tref5.npy
输出:
    - 投影后的连通 3pt, ratio 和 z=0,1,2 的 ratio 图, 保存到本目录 01_result.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# ---- 从项目目录下的 98_tools 和 01_gevp 导入通用函数与被复用代码 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_gevp"))

from analysis_tools import DEFAULT_PLOT_COLORS, resample, sem  # noqa: E402  # type: ignore
from output_tools import save_fig, save_npy  # noqa: E402  # type: ignore
from profiling_tools import print_mem_peak  # noqa: E402  # type: ignore

# 1_gevp_deuteron 文件名以数字开头, 不能直接 import, 用 importlib 按模块名加载
gd = importlib.import_module("1_gevp_deuteron")


@dataclass(frozen=True)
class ReadConfig:
    """读取相关的配置和路径."""

    conf_short: str
    conf_ids: np.ndarray
    data_dir: Path          # 2pt 输入目录
    ope_dir: dict[str, Path]
    delta_z: int
    z_max: int
    Nt: int
    eigvec_path: Path
    eigvec_t: int
    t0: int


@dataclass(frozen=True)
class CalcConfig:
    """计算 ratio 相关的配置和路径."""

    dt_max: int
    nmom: int
    sink_mom: tuple
    source_mom: tuple
    data_dir: Path          # ratio 数据输出目录


@dataclass(frozen=True)
class PlotConfig:
    """画图相关的配置和路径."""

    dt_list: tuple
    z_list: tuple
    xlim: tuple
    ylim: tuple
    figure_dir: Path        # ratio 图片输出目录


def init() -> tuple[ReadConfig, CalcConfig, PlotConfig]:
    """根据 conf_short 初始化 ratio GEVP 分析的参数和路径, 拆成三个 dataclass 返回.

    CONF_SHORT 等全大写变量为可给用户修改的配置参数.
    返回: Read, Calc, Plot.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"
    DT_MAX = 20
    Z_MAX = 25
    DELTA_Z = 30
    EIGVEC_T = 10
    DT_LIST = tuple(range(6, 13))
    Z_LIST = (0, 5, 10)
    XLIM = (-7.0, 7.0)
    YLIM = (0.0, 1.3)

    if CONF_SHORT == "c32p29":
        conf_ids = np.array(
            [c for c in range(11000, 11000 + 50 * 297, 50) if c not in (18350, 23050)])
        T0 = 5
        conf_name = "beta6.20_mu-0.2770_ms-0.2400_L32x64"
    else:
        raise ValueError(f"unsupported conf_short: {CONF_SHORT}")

    nmom = 2
    Nt = 64
    sink_mom = (0, 1, 1, 1)
    source_mom = (0, 1, 1, 1, -1, 1, 1, 1)

    data_dir = Path(
        f"/public/home/huangcl/00_data/00_NP_2pt/mom0/{CONF_SHORT}")
    ope_dir = {
        "x": Path(
            "/public/home/chenc/gluon_unpo/generating_data/operator/"
            f"unfixed_operator_sum_xdir/{conf_name}/sum"
        ),
        "y": Path(
            "/public/home/chenc/gluon_unpo/generating_data/operator/"
            f"unfixed_operator_sum_ydir/{conf_name}/sum"
        ),
        "z": Path(
            "/public/home/chenc/gluon_unpo/generating_data/operator/"
            f"unfixed_operator_sum/{conf_name}/sum"
        ),
    }
    eigvec_path = Path(
        f"/public/home/huangcl/06_deuteron_gluon_pdf/01_gevp/01_result/"
        f"{CONF_SHORT}/eigvec/state0_tref{T0}.npy"
    )
    output_dir = Path(__file__).resolve().parent / "01_result"

    Read = ReadConfig(
        conf_short=CONF_SHORT,
        conf_ids=conf_ids,
        data_dir=data_dir,
        ope_dir=ope_dir,
        delta_z=DELTA_Z,
        z_max=Z_MAX,
        Nt=Nt,
        eigvec_path=eigvec_path,
        eigvec_t=EIGVEC_T,
        t0=T0,
    )
    Calc = CalcConfig(
        dt_max=DT_MAX,
        nmom=nmom,
        sink_mom=sink_mom,
        source_mom=source_mom,
        data_dir=output_dir / "data" / CONF_SHORT,
    )
    Plot = PlotConfig(
        dt_list=DT_LIST,
        z_list=Z_LIST,
        xlim=XLIM,
        ylim=YLIM,
        figure_dir=output_dir / "fig" / CONF_SHORT,
    )

    return Read, Calc, Plot


def load_ope(
    axis: str,
    conf_ids: np.ndarray,
    ope_dir: dict[str, Path],
    delta_z: int,
    z_max: int,
) -> np.ndarray:
    """读取指定轴的 gluon OPE 数据.

    输入 axis: x/y/z.
    输出 ope 的 para: conf,tau,z.
    """

    ope_list = []
    for conf_id in conf_ids:
        ope_path = ope_dir[axis] / f"ops_sum_dz{delta_z}_conf{conf_id}.npz"
        if not ope_path.is_file():
            raise FileNotFoundError(f"not found: {ope_path}")
        # para: z, tau
        ope_list.append(np.load(ope_path)["ops"][:z_max])

    # para: conf, z, tau
    ope = np.asarray(ope_list)
    # para: conf, tau, z
    ope = ope.transpose(0, 2, 1)
    print(f"ope axis={axis}: {ope_path}, shape={ope.shape}")
    return ope


def align_ope(
    ope: np.ndarray,
    Nt: int,
    dt_max: int,
    z_max: int,
) -> np.ndarray:
    """按 source time 平移 OPE.

    输入 ope 的 para: conf,tau,z.
    输出 ope_rel 的 para: conf,tsrc,dtau,z.
    """

    ope_rel = np.zeros((ope.shape[0], Nt, dt_max, z_max), dtype=complex)
    for tsrc in range(Nt):
        ope_shift = np.roll(ope, shift=-tsrc, axis=1)
        ope_rel[:, tsrc, :, :] = ope_shift[:, :dt_max, :]
    return ope_rel


def build_3pt_matrix(
    c2: np.ndarray,
    ope_rel: np.ndarray,
    Nt: int,
    dt_max: int,
    z_max: int,
    nmom: int,
) -> np.ndarray:
    """逐个 tsrc 构造 C3 矩阵并对 tsrc 求平均.

    输入 c2 的 para: conf,tsrc,dt,|p|,|k|.
    输入 ope_rel 的 para: conf,tsrc,dtau,z.
    输出 c3_avg 的 para: conf,dt,dtau,z,|p|,|k|.
    """

    # para: conf,tsrc,dt,|p|,|k|
    c2_rel = c2[:, :, :dt_max]
    c3_avg = np.zeros(
        (c2.shape[0], dt_max, dt_max, z_max, nmom, nmom),
        dtype=np.complex128,
    )

    for dt in range(dt_max):
        for dtau in range(dt + 1):
            # 对 tsrc 求和后再除以 Nt, 与 test01_ratio_no_gevp 一致.
            c3_avg[:, dt, dtau] = np.einsum(
                "ctpk,ctz->czpk",
                c2_rel[:, :, dt, :, :],
                ope_rel[:, :, dtau, :],
                optimize=True,
            )
    c3_avg /= Nt

    print(f"c3_avg shape: {c3_avg.shape}")
    return c3_avg


def project_ground_state(
    c2_smp: np.ndarray,
    c3_avg: np.ndarray,
    ope_avg: np.ndarray,
    eigvec: np.ndarray,
) -> np.ndarray:
    """先在动量基减去 disconnected 部分, 再用基态本征矢投影并计算 ratio.

    输入 c2_smp 的 para: sample,dt,|p|,|k|.
    输入 c3_avg 的 para: conf,dt,dtau,z,|p|,|k|.
    输入 ope_avg 的 para: conf,dtau,z.
    输入 eigvec 的 para: sample,k.
    输出 ratio 的 para: sample,dt,dtau,z.
    """

    # 使用与 C2, eigvec 相同的 bootstrap 种子 0, 保持 sample 轴对齐.
    c3_smp = resample(c3_avg)
    ope_smp = resample(ope_avg)

    # 先在动量基减去 disconnected 部分, para: sample,dt,dtau,z,|p|,|k|
    c3_disc = c3_smp - c2_smp[:, :, None, None, :, :] * ope_smp[
        :, None, :, :, None, None
    ]

    # v^dagger C3 v 与 v^dagger C2 v
    # para: sample,dt,dtau,z
    c3_proj = np.einsum(
        "sp,stdzpk,sk->stdz",
        np.conj(eigvec),
        c3_disc,
        eigvec,
        optimize=True,
    )
    # para: sample,dt
    c2_proj = np.einsum(
        "sp,stpk,sk->st",
        np.conj(eigvec),
        c2_smp,
        eigvec,
        optimize=True,
    )

    # ratio = C3_proj / C2_proj
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (c3_proj / c2_proj[:, :, None, None]).real

    print(f"c3_proj shape: {c3_proj.shape}, ratio shape: {ratio.shape}")
    return ratio


def plot_ratio(ratio: np.ndarray, Read: ReadConfig, Plot: PlotConfig) -> None:
    """按 test01_ratio_no_gevp 的样式画 ratio errorbar 图.

    输入 ratio 的 para: sample,dt,dtau,z.
    """

    ratio_mean = ratio.mean(axis=0)
    ratio_err = sem(ratio)
    n_dt = len(Plot.dt_list)
    mid_idx = (n_dt - 1) / 2.0

    for z in Plot.z_list:
        figure, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for i, dt in enumerate(Plot.dt_list):
            tau_vals = np.arange(0, dt + 1)
            x_vals = tau_vals - dt / 2.0 + (i - mid_idx) * 0.05
            color = DEFAULT_PLOT_COLORS[i % len(DEFAULT_PLOT_COLORS)]
            ax.errorbar(
                x_vals,
                ratio_mean[dt, tau_vals, z],
                yerr=ratio_err[dt, tau_vals, z],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
                markersize=2,
                label=f"tsep={dt}",
            )

        ax.set_xlim(Plot.xlim[0], Plot.xlim[1])
        ax.set_ylim(Plot.ylim[0], Plot.ylim[1])
        ax.set_xlabel(
            r"$t_{\mathrm{ins}} - t_{\mathrm{sep}}/2$", fontsize=16, labelpad=8)
        ax.set_ylabel(r"$C_3 / C_2$", fontsize=16, labelpad=8)
        ax.set_title(
            f"ratio with GEVP, state0, 6 dirs ave, z={z}, P(0,0,0)",
            fontsize=14,
            pad=12,
        )
        ax.legend(loc="upper right")
        plt.tight_layout()

        figure_path = (
            Plot.figure_dir
            / f"ratio_tref{Read.t0}_evt{Read.eigvec_t}_z{z}_{Read.conf_short}.png"
        )
        save_fig(figure_path, figure, name="ratio_plot")
        plt.close(figure)


if __name__ == "__main__":
    Read, Calc, Plot = init()

    t_total_begin = time.perf_counter()

    print("read 2pt begin")
    t_begin = time.perf_counter()

    # para: conf, p, k, dt, t_src, r, q, z, y
    correlators = gd.load_correlators(Read.conf_ids, Read.data_dir)

    print(f"read 2pt end, {time.perf_counter() - t_begin:.0f}s\n")

    # para: conf, p, k, dt, t_src, sz; sz[:] = -1, 0, 1
    spin_channels = gd.project_deuteron_spin_channels(correlators)

    # para: conf, p, k, dt, t_src
    average_correlator = spin_channels.mean(axis=-1)

    # para: conf, |p|, |k|, dt, t_src
    c2 = gd.build_corr_matrix(
        average_correlator, Calc.nmom, Calc.sink_mom, Calc.source_mom)

    # para: conf, t_src, dt, |p|, |k|
    c2 = c2.transpose(0, 4, 3, 1, 2)
    print(f"c2 shape: {c2.shape}")

    # para: conf,dt,|p|,|k|
    c2_avg = c2[:, :, :Calc.dt_max].mean(axis=1)

    # para: sample, t, k
    eigvec = np.load(Read.eigvec_path)
    # para: sample, k
    eigvec = eigvec[:, Read.eigvec_t, :]
    print(f"loaded eigvec: {Read.eigvec_path}, {eigvec.shape}")

    # 与 eigvec 相同的 bootstrap 样本; C3 和 OPE 也使用 seed=0.
    # para: sample,dt,|p|,|k|
    c2_smp = resample(c2_avg)

    ratio_sum = np.zeros(
        (c2_smp.shape[0], Calc.dt_max, Calc.dt_max, Read.z_max), dtype=float)

    for axis in ("x", "y", "z"):
        print("calc 3pt ratio begin")
        t_begin = time.perf_counter()

        ope = load_ope(
            axis, Read.conf_ids, Read.ope_dir, Read.delta_z, Read.z_max)
        ope_rel = align_ope(ope, Read.Nt, Calc.dt_max, Read.z_max)
        # para: conf,dtau,z
        ope_avg = ope_rel.mean(axis=1)
        c3_avg = build_3pt_matrix(
            c2, ope_rel, Read.Nt, Calc.dt_max, Read.z_max, Calc.nmom)
        ratio = project_ground_state(c2_smp, c3_avg, ope_avg, eigvec)
        ratio_sum += ratio

        print(
            f"calc 3pt ratio end, {time.perf_counter() - t_begin:.0f}s\n")

    del ope, ope_rel, ope_avg, c3_avg, ratio

    # P=0 时 neg_x/y/z 与 pos_x/y/z 相同, 六个方向的平均等于 x,y,z 的平均.
    ratio_ave = ratio_sum / 3.0

    ratio_path = (
        Calc.data_dir / f"ratio_tref{Read.t0}_evt{Read.eigvec_t}.npy"
    )
    save_npy(ratio_path, ratio_ave, name="ratio_ave")

    print(f"ratio_ave finite: {np.isfinite(ratio_ave).all()}")

    print("plot begin")
    t_begin = time.perf_counter()

    plot_ratio(ratio_ave, Read, Plot)

    print(f"plot end, {time.perf_counter() - t_begin:.0f}s\n")

    print_mem_peak("peak", use_gpu=False)
    print(f"job done, {time.perf_counter() - t_total_begin:.0f}s")
