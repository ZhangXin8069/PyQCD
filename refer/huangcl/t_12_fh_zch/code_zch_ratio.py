#!/public/home/huangcl/.venv/bin/python
"""
读取师兄的 Ratio_data_*.npz，重新组装成 (Nsample, dt, dtau, z) 格式，
保存为 ratio.npy，然后用 plot_ratio 画图。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

# ---- 从上级目录 98_tools 导入通用函数 ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "98_tools"))
from analysis_tools import (  # noqa: E402
    sem, DEFAULT_PLOT_COLORS,
)


# ===== 配置 =====
conf_short = "L24x72"
pz_list = [4]

# 画图参数
_DT_LIST_MAP = {
    2: list(range(5, 14)),
    3: list(range(5, 13)),
    4: list(range(5, 12)),
    5: list(range(5, 11)),
    6: list(range(5, 10)),
}
_FIXED_Z_LIST = [0]
_FIXED_XLIM = [-7, 7]
_FIXED_YLIM = [0, 0.8]
_DX_OFFSET = 0.05

# 数据路径
data_base = "/public/group/imp/zengch/LQCD/renorma/result/Ratio_data"

# 输出基础目录（本目录下的 1_result）
out_base = '/public/home/huangcl/04_gluon_unpolarized_PDF/t_12_fh_zch/result/'


@dataclass
class PlotParams:
    dt_list: list[int]
    z_list: list[int]
    xlim: list[float]
    ylim: list[float]
    dx_offset = 0.05
    colors: list[str] = field(default_factory=lambda: DEFAULT_PLOT_COLORS)


def load_and_reorganize(file_path: str, dt_max: int, Nx: int):
    """
    加载师兄的 npz 文件，重新组装成 (Nsample, dt, dtau, z) 格式。

    师兄的数据格式：
        data:    (N_comb, 5)     = [z, t_sep, ti_sep, mean, std]
        samples: (N_comb, Nsamp) = bootstrap 样本

    输出格式（和你代码的 ratio.npy 一致）：
        ratio:   (Nsample, dt_max, dt_max, Nx)
    """
    f = np.load(file_path)
    data_arr = f['data']        # (N_comb, 5)
    samples_arr = f['samples']  # (N_comb, Nsamp)
    f.close()

    Nsamp = samples_arr.shape[1]

    # 初始化输出数组
    ratio = np.zeros((Nsamp, dt_max, dt_max, Nx), dtype=float)

    # 遍历 data 的每一行，填入对应位置
    for i in range(data_arr.shape[0]):
        z = int(data_arr[i, 0])
        t_sep = int(data_arr[i, 1])
        ti_sep = int(data_arr[i, 2])

        # 只处理在范围内的数据
        if z >= Nx or t_sep >= dt_max or ti_sep >= dt_max:
            continue

        # samples[i, :] 是 (z, t_sep, ti_sep) 这个组合的 bootstrap 样本
        ratio[:, t_sep, ti_sep, z] = samples_arr[i, :]

    return ratio


def plot_ratio(ratio: np.ndarray, plotpa: PlotParams,
               save_dir: str, dir_name: str,
               Nconf: int, Nsample: int,
               Px: int = 0, Py: int = 0, Pz: int = 0):
    """绘制 ratio 大图，每张图画一个 z"""
    print(
        f"==================== plot ratio ({dir_name}) start ====================")

    ratio_mean = ratio.mean(0)  # (dt, dtau, z)
    ratio_err = sem(ratio, jackknife=False)

    n_dt = len(plotpa.dt_list)
    mid_idx = (n_dt - 1) / 2.0

    for _z in plotpa.z_list:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        for i, _dt in enumerate(plotpa.dt_list):
            tau_vals = np.arange(0, _dt + 1)

            offset_i = (i - mid_idx) * plotpa.dx_offset
            x_vals = tau_vals - _dt / 2.0 + offset_i

            y_vals = ratio_mean[_dt, tau_vals, _z]
            y_errs = ratio_err[_dt, tau_vals, _z]

            color = plotpa.colors[i % len(plotpa.colors)]
            ax.errorbar(
                x_vals, y_vals, yerr=y_errs,
                fmt="x", color=color, ecolor=color,
                capsize=0, markersize=7, markeredgewidth=1.8,
                linewidth=1.2, zorder=3, label=f"tsep={_dt}",
            )

        ax.set_xlim(plotpa.xlim[0], plotpa.xlim[1])
        ax.set_ylim(plotpa.ylim[0], plotpa.ylim[1])
        ax.set_xlabel(r"$t_{\mathrm{ins}} - t_{\mathrm{sep}}/2$", fontsize=16)
        ax.set_ylabel(r"$C_3 / C_2$", fontsize=16)
        ax.set_title(
            f"dir={dir_name}, z={_z}, "
            f"P({Px},{Py},{Pz}), Nconf={Nconf}, Nsample={Nsample}",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"ratio_{dir_name}_z{_z}.png")
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved: {save_path}")
        plt.close(fig)

    print(
        f"==================== plot ratio ({dir_name}) end ====================")


def main():
    # L24x72: Nt=72, Nx=24, dt_max=20
    dt_max = 20
    Nx = 24

    for pz in pz_list:
        file_path = f"{data_base}/Ratio_data_{conf_short}_dhxmeang1_pz{pz}.npz"
        print(f"\nLoading: {file_path}")

        if not os.path.exists(file_path):
            print(f"  ⚠️  文件不存在: {file_path}")
            continue

        # 1. 加载并重新组装成 (Nsample, dt, dtau, z)
        ratio = load_and_reorganize(file_path, dt_max, Nx)
        print(f"  ratio shape: {ratio.shape}")

        # 2. 保存为 ratio.npy（和你代码的格式一致）
        save_dir = os.path.join(out_base, conf_short, f"P{pz}")
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "ratio.npy"), ratio)
        print(f"  saved: {save_dir}/ratio.npy")

        # 3. 画图
        plotpa = PlotParams(
            dt_list=_DT_LIST_MAP.get(pz, list(range(5, 14))),
            z_list=_FIXED_Z_LIST,
            xlim=_FIXED_XLIM,
            ylim=_FIXED_YLIM,
        )
        plot_ratio(ratio, plotpa, save_dir, "zch",
                   Nconf=ratio.shape[0], Nsample=ratio.shape[0],
                   Px=0, Py=0, Pz=pz)

    print("\nAll done!")


if __name__ == "__main__":
    main()
