#!/public/home/huangcl/.venv/bin/python
"""
读取师兄的 Ratio_data_*.npz, 重新组装成 (Nsample, tsep, tins, z) 格式,
与用户自己的 ave ratio 数据比较均值与误差 (sem) 的差异 (只看 tins <= tsep 且 5 <= tsep <= 11 的有效区域),
并用师兄的 reorganize 结果画图, 保存到 ave 文件夹 (ratio_zch_z0.png)。
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

# 数据路径 (师兄的 npz 所在目录)
data_base = "/public/group/imp/zengch/LQCD/renorma/result/Ratio_data"

# 相对误差统计阈值 (rel_err > 阈值 与 < 阈值 的数据点数统计)
_REL_ERR_THRESHOLD = 1e-2

# 统计范围参数 (作者可手动调整)
_Z_MAX = 23        # z 最大值, 统计 z = 0 到 _Z_MAX
_TSEP_MIN = 0      # tsep 最小值
_TSEP_MAX = 20     # tsep 最大值

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
    加载师兄的 npz 文件, 重新组装成 (Nsample, dt, dtau, z) 格式。

    师兄的数据格式:
        data:    (N_comb, 5)     = [z, t_sep, ti_sep, mean, std]
        samples: (N_comb, Nsamp) = bootstrap 样本

    输出格式 (和你代码的 ratio.npy 一致):
        ratio:   (Nsample, dt_max, dt_max, Nx)
    """
    f = np.load(file_path)
    data_arr = f['data']        # (N_comb, 5)
    samples_arr = f['samples']  # (N_comb, Nsamp)
    f.close()

    Nsamp = samples_arr.shape[1]

    # 初始化输出数组
    ratio = np.zeros((Nsamp, dt_max, dt_max, Nx), dtype=float)

    # 遍历 data 的每一行, 填入对应位置
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


def compare_ratio(mine: np.ndarray, zch: np.ndarray, dt_max: int, Nx: int):
    """比较 mine 与 zch 的 ratio 的均值与误差 (sem), 统计有效区域内的差异

    输入:
        mine: 用户自己的 ratio 数据, para: Nsample, tsep, tins, z
        zch: 师兄的 ratio 数据, para: Nsample, tsep, tins, z
        dt_max: tsep 维度大小
        Nx: z 维度大小

    输出:
        打印均值相对差异与误差差异的统计 (只看 tins <= tsep, _TSEP_MIN <= tsep <= _TSEP_MAX, z <= _Z_MAX 的有效区域)
    """
    if mine.shape != zch.shape:
        print(f"  shape mismatch! mine={mine.shape}, zch={zch.shape}")
        return

    # 有效区域: tins <= tsep, _TSEP_MIN <= tsep <= _TSEP_MAX, z <= _Z_MAX
    # (ratio 只定义在 tins <= tsep, 剩下的一半没有物理意义)
    tsep_idx = np.arange(dt_max)[:, None, None]   # (tsep, 1, 1)
    tins_idx = np.arange(dt_max)[None, :, None]   # (1, tins, 1)
    z_idx = np.arange(Nx)[None, None, :]          # (1, 1, z)
    valid = (tins_idx <= tsep_idx) & (tsep_idx >= _TSEP_MIN) & (
        tsep_idx <= _TSEP_MAX) & (z_idx <= _Z_MAX)  # (tsep, tins, z)
    mask = np.broadcast_to(valid, (dt_max, dt_max, Nx))  # (tsep, tins, z)

    # 均值与误差 (sem), 沿 Nsample 轴
    mine_mean = mine.mean(0)  # (tsep, tins, z)
    mine_err = sem(mine, jackknife=False)  # (tsep, tins, z)
    zch_mean = zch.mean(0)  # (tsep, tins, z)
    zch_err = sem(zch, jackknife=False)  # (tsep, tins, z)

    # 均值相对差异: |mine_mean - zch_mean| / |mine_mean|, 用用户自己的均值做分母
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_rel_diff = np.abs(mine_mean - zch_mean) / np.abs(mine_mean)

    # 误差绝对差异: |mine_err - zch_err|
    err_diff = np.abs(mine_err - zch_err)

    # 分母为 0 的点跳过 (置 NaN), 无效区域也置 NaN
    denom_zero = (mine_mean == 0)
    valid_mask = mask & ~denom_zero
    mean_rel_masked = np.where(valid_mask, mean_rel_diff, np.nan)
    err_diff_masked = np.where(mask, err_diff, np.nan)

    # 统计均值相对差异 > 阈值 和 < 阈值 的数据点数, 阈值由 _REL_ERR_THRESHOLD 控制
    n_gt_mean = int(np.sum(mean_rel_masked > _REL_ERR_THRESHOLD))
    n_lt_mean = int(np.sum(mean_rel_masked < _REL_ERR_THRESHOLD))
    n_eq_mean = int(np.sum(mean_rel_masked == _REL_ERR_THRESHOLD))
    n_skip_mean = int(np.sum(np.isnan(mean_rel_masked)))

    # 统计误差差异 > 阈值 和 < 阈值 的数据点数, 阈值由 _REL_ERR_THRESHOLD 控制
    n_gt_err = int(np.sum(err_diff_masked > _REL_ERR_THRESHOLD))
    n_lt_err = int(np.sum(err_diff_masked < _REL_ERR_THRESHOLD))
    n_eq_err = int(np.sum(err_diff_masked == _REL_ERR_THRESHOLD))
    n_skip_err = int(np.sum(np.isnan(err_diff_masked)))

    print(
        f"===== mean rel_diff statistics ({_TSEP_MIN} <= tsep <= {_TSEP_MAX}, tins <= tsep, z <= {_Z_MAX}) =====")
    print(f"  mean rel_diff > {_REL_ERR_THRESHOLD} : {n_gt_mean}")
    print(f"  mean rel_diff < {_REL_ERR_THRESHOLD} : {n_lt_mean}")
    print(f"  mean rel_diff = {_REL_ERR_THRESHOLD} : {n_eq_mean}")
    print(f"  skipped (mine_mean=0 or invalid): {n_skip_mean}")

    # # 列出所有 mean rel_diff > 阈值的点 (位置与双方均值, 数值保留 2 位小数)
    # if n_gt_mean > 0:
    #     print("  ---- points with mean rel_diff > threshold ----")
    #     _idx = np.argwhere(mean_rel_masked > _REL_ERR_THRESHOLD)
    #     for _i in _idx:
    #         _tsep, _tins, _z = int(_i[0]), int(_i[1]), int(_i[2])
    #         _mm = mine_mean[_tsep, _tins, _z]
    #         _zm = zch_mean[_tsep, _tins, _z]
    #         _rd = mean_rel_masked[_tsep, _tins, _z]
    #         print(
    #             f"    tsep={_tsep}, tins={_tins}, z={_z}: "
    #             f"mine_mean={_mm:.2f}, zch_mean={_zm:.2f}, rel_diff={_rd:.2f}")

    print(
        f"===== err diff statistics ({_TSEP_MIN} <= tsep <= {_TSEP_MAX}, tins <= tsep, z <= {_Z_MAX}) =====")
    print(f"  err diff > {_REL_ERR_THRESHOLD} : {n_gt_err}")
    print(f"  err diff < {_REL_ERR_THRESHOLD} : {n_lt_err}")
    print(f"  err diff = {_REL_ERR_THRESHOLD} : {n_eq_err}")
    print(f"  skipped (invalid): {n_skip_err}")


def plot_ratio(ratio: np.ndarray, plotpa: PlotParams,
               save_dir: str, dir_name: str,
               Nconf: int, Nsample: int,
               Px: int = 0, Py: int = 0, Pz: int = 0):
    """绘制 ratio 大图, 每张图画一个 z"""
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
        # 1. 加载师兄的 npz 并重新组装
        file_path = os.path.join(
            data_base, f"Ratio_data_{conf_short}_dhxmeang2_pz{pz}.npz")
        print(f"\nLoading zch: {file_path}")

        if not os.path.exists(file_path):
            print(f"  file not exist: {file_path}")
            continue

        zch = load_and_reorganize(file_path, dt_max, Nx)
        print(f"  zch ratio shape: {zch.shape}")

        # 2. 加载用户自己的 ave ratio 数据
        mine_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "1_result", conf_short, f"P{pz}", "ratio", "ratio_ave.npy")
        print(f"Loading mine: {mine_path}")

        if not os.path.exists(mine_path):
            print(f"  file not exist: {mine_path}")
            continue

        mine = np.load(mine_path)
        print(f"  mine ratio shape: {mine.shape}")

        # 3. 逐点比较, 计算绝对误差 (只看 tins <= tsep 的有效区域)
        compare_ratio(mine, zch, dt_max, Nx)

        # 4. 用师兄的 reorganize 结果画图, 保存到 ave 文件夹
        # 只画 tsep < 12 的数据
        plotpa = PlotParams(
            dt_list=[_dt for _dt in _DT_LIST_MAP.get(
                pz, list(range(5, 14))) if _dt < 12],
            z_list=_FIXED_Z_LIST,
            xlim=_FIXED_XLIM,
            ylim=_FIXED_YLIM,
        )
        ave_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "1_result", conf_short, f"P{pz}", "ave")
        # plot_ratio(zch, plotpa, ave_dir, "zch",
        #            Nconf=zch.shape[0], Nsample=zch.shape[0],
        #            Px=0, Py=0, Pz=pz)

    print("\nAll done!")


if __name__ == "__main__":
    main()
