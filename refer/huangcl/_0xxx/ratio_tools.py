"""ratio 纯计算与画图工具包.

主要功能:
    calc_gluon_3pt: 由 2pt 和胶子圈两套数据计算 gluon 的 3pt 关联函数.
    calc_gluon_ratio: 对已 resample 的 3pt, 2pt 和胶子圈做真空圈乘积
        减除并计算 ratio.
    plot_ratio_per_z: 画单个 z 的 ratio 随插入时间的变化.

说明:
    tsrc 平均, 方向平均与 resample 都在外部调用方完成, 本模块只做
    单个数据集上的纯计算, 不处理任何方向信息.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "98_tools"))
from analysis_tools import (  # noqa: E402  # type: ignore
    sem, DEFAULT_PLOT_COLORS,
)
from output_tools import save_fig  # noqa: E402  # type: ignore


def calc_gluon_3pt(corr2: np.ndarray, gluon_loop: np.ndarray) -> np.ndarray:
    """
    由 2pt 关联函数和胶子圈计算 gluon 的 3pt 关联函数.

    C3(t, tau, z) = C2(t) * gluon_loop(tau, z), 并对 source time 做循环平移.

    参数:
    -----
    corr2: 2pt 关联函数, para: conf, tsrc, dt.
    gluon_loop: 胶子圈数据, para: conf, tau, z.

    输出:
    -----
    corr3: 3pt 关联函数, para: conf, tsrc, dt, dtau, z.
    """
    print("\n==================== calc gluon 3pt start ====================")

    Nconf, Nt, dt_max = corr2.shape
    z_max = gluon_loop.shape[2]

    # 对 source time tsrc 做平移
    # para: conf, tsrc, dtau, z
    loop_rel = np.zeros((Nconf, Nt, dt_max, z_max), dtype=complex)
    for tsrc in range(Nt):
        loop_shift = np.roll(gluon_loop, shift=-tsrc, axis=1)
        loop_rel[:, tsrc, :, :] = loop_shift[:, :dt_max, :]

    # para: conf, tsrc, dt, dtau, z
    corr3 = np.zeros(
        (Nconf, Nt, dt_max, dt_max, z_max), dtype=complex)
    for _dt in range(dt_max):
        for _dtau in range(_dt + 1):
            c2_slice = corr2[:, :, _dt]            # para: conf, tsrc
            loop_slice = loop_rel[:, :, _dtau, :]  # para: conf, tsrc, z
            corr3[:, :, _dt, _dtau, :] = loop_slice * \
                c2_slice[:, :, np.newaxis]

    print(f"corr3 shape: {corr3.shape}")
    print("==================== calc gluon 3pt end ====================\n")
    return corr3


def calc_gluon_ratio(
        corr3_re: np.ndarray, corr2_re: np.ndarray,
        gluon_loop_re: np.ndarray) -> np.ndarray:
    """
    对已 resample 的数据做真空圈乘积减除并计算 ratio.

    C3_disc = C3 - C2 * gluon_loop, ratio = C3_disc / C2.

    参数:
    -----
    corr3_re: 已 resample 的 3pt, para: sample, dt, dtau, z.
    corr2_re: 已 resample 的 2pt, para: sample, dt.
    gluon_loop_re: 已 resample 的胶子圈, para: sample, z.

    输出:
    -----
    ratio: para: sample, dt, dtau, z.
    """
    print("\n==================== calc gluon ratio start ====================")

    # 真空圈乘积减除
    # para: sample, dt, dtau, z
    corr3_disc = corr3_re - \
        corr2_re[:, :, np.newaxis, np.newaxis] * \
        gluon_loop_re[:, np.newaxis, np.newaxis, :]

    # ratio = C3_disc / C2
    # para: sample, dt, dtau, z
    ratio = (corr3_disc /
             corr2_re[:, :, np.newaxis, np.newaxis]).real

    print(f"ratio shape: {ratio.shape}")
    print("==================== calc gluon ratio end ====================\n")
    return ratio


def plot_ratio_per_z(
        ratio_z: np.ndarray, fig_path, title: str, xlim: list[float],
        ylim: list[float], dt_list: list[int]):
    """
    绘制单个 z 的 ratio 图. 不同的 tsep 会有少量的 x 轴偏移.

    参数:
    -----
    ratio_z: 单个 z 的 ratio 数据, para: sample, dt, dtau.
    fig_path: 图片保存路径, 含文件名和后缀.
    title: 图题.
    xlim: 横轴范围.
    ylim: 纵轴范围.
    dt_list: 需要画的 t_sep 列表.


    """
    dx_offset = 0.05

    ratio_mean = ratio_z.mean(0)  # para: dt, dtau
    ratio_err = sem(ratio_z)

    # 计算每个 tsep 的横轴偏移量
    n_dt = len(dt_list)
    mid_idx = (n_dt - 1) / 2.0  # 中间索引 (可能为半整数)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    for i, _dt in enumerate(dt_list):
        tau_vals = np.arange(0, _dt + 1)

        # 横坐标: tau - dt / 2 + 偏移量
        # 中间 tsep 偏移为 0, 小的左偏 (负), 大的右偏 (正)
        offset_i = (i - mid_idx) * dx_offset
        x_vals = tau_vals - _dt / 2.0 + offset_i

        y_vals = ratio_mean[_dt, tau_vals]
        y_errs = ratio_err[_dt, tau_vals]

        color = DEFAULT_PLOT_COLORS[i % len(DEFAULT_PLOT_COLORS)]
        ax.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt="o", markersize=2, color=color, ecolor=color,
            elinewidth=0.8, capsize=3, capthick=0.8,
            zorder=3, label=f"tsep={_dt}",
        )

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(
        r"$t_{\mathrm{ins}} - t_{\mathrm{sep}}/2$", fontsize=16, labelpad=8)
    ax.set_ylabel(r"$C_3 / C_2$", fontsize=16, labelpad=8)
    ax.set_title(title, fontsize=14, pad=12)
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_fig(fig_path, fig, name="ratio")
    plt.close(fig)
