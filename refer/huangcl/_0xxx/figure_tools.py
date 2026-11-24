"""画图专用工具包.

主要功能:
    calc_curve_band: 在密网格上按逐样本拟合参数计算模型曲线.
    plot_errbar: 画误差棒, 灰色平台带和拟合曲线带.
    plot_scatter: 画散点图, 可选水平参考线.
    plot_hist: 画直方图.

用法: 先用 calc_curve_band 把拟合参数转成曲线数据, 再连同误差棒一起交给
plot_errbar 绘制.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis_tools import sem
from output_tools import save_fig

# ============================================================
# 默认颜色和样式
# ============================================================

PLOT_COLORS = [
    "#4ba3e3",  # 1. 鲜亮浅蓝
    "#58b368",  # 2. 鲜亮草绿
    "#f29e38",  # 3. 暖橙
    "#a569bd",  # 4. 中紫
    "#2075bc",  # 5. 深蓝
    "#196f3d",  # 6. 深绿
    "#d9534f",  # 7. 砖红
    "#6c3483",  # 8. 暗紫
    "#21618c",  # 9. 藏青
    "#117864",  # 10. 深墨绿
]

# 图片尺寸, 分辨率和字体大小, 统一由本模块控制.
FIG_SIZE = (8, 5)
FIG_DPI = 150

TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 11


def _calc_offsets(n_group: int, step: float) -> np.ndarray:
    """
    计算一组数据在横坐标上的均分偏移量.

    参数:
    -----
    n_group: 数据组数.
    step: 相邻两组之间的偏移步长.

    输出:
    -----
    offsets: 长度为 n_group 的偏移数组, 关于 0 对称.
    """

    if n_group <= 0:
        return np.zeros(0)
    return np.linspace(
        -(n_group - 1) * step / 2, (n_group - 1) * step / 2, n_group)


# ============================================================
# 曲线数据生成
# ============================================================

def calc_curve_band(
    model: Callable,
    para_samples: Dict[str, np.ndarray],
    x_range: Tuple[float, float],
    n_point: int = 200,
    model_args: Tuple = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 `x_range` 内按逐样本参数计算模型曲线.

    参数:
    -----
    model: 模型函数, 接口为 `(x, p)`, 返回与 x 同形状的数组, 如
        `lambda t, p: p["A"] * np.exp(-p["E"] * t)`.
    para_samples: 标签到逐样本参数数组的映射, 第 0 轴为 sample.
    x_range: 曲线横坐标范围 `(x_min, x_max)`.

    可选项:
    -----
    n_point: 密网格点数, 默认 200.
    model_args: 模型除 `(x, p)` 外的其他参数, 按顺序追加为 `model(x, p, *model_args)`.

    输出:
    -----
    x: 曲线横坐标, para: x_idx.
    curve_samples: 逐样本曲线, para: sample, x_idx.
    """

    # para: x_idx
    x = np.linspace(x_range[0], x_range[1], n_point)
    param_names = list(para_samples.keys())
    n_sample = len(para_samples[param_names[0]])

    # para: sample, x_idx
    curve_samples = np.array([
        model(x, {name: para_samples[name][i] for name in param_names},
              *model_args)
        for i in range(n_sample)
    ])

    return x, curve_samples


# ============================================================
# 画图
# ============================================================

def plot_errbar(
    raw_data: Dict[
        Optional[str],
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    save_path: Path,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    title: Optional[str] = None,
    x_offset: float = 0.3,
    # 网格
    show_grid: bool = True,
    # 拟合曲线带
    fit_data: Optional[Dict[
        Optional[str],
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ] = None,
    fit_x_offset: float = 0.3,
    # 灰色带
    band: Optional[Tuple[Optional[str], float, float]] = None,
    # 图例位置
    legend_loc: str = "upper right",
) -> None:
    """
    绘制误差棒, 灰色平台带和拟合曲线带并保存图片. 

    说明: 标签为 None 的组照常绘制, 但不写入图例, 数据为 None 组直接跳过, 也会占用 x_offset 的一个长度.

    参数:
    -----
    raw_data: 标签到 `(横坐标, 均值, 误差)` 的映射.
    save_path: 图片保存路径.

    可选项:
    -----
    xlabel, ylabel, title, legend_loc, xlim, ylim: 坐标轴, 标题, 图例位置和范围.
    x_offset, fit_x_offset: 误差棒和拟合曲线带各自的横坐标偏移步长.
    fit_data: 标签到 `(横坐标, 均值, 误差)` 的映射.
    band: 灰色平台带 `(标签, 均值, 误差)`.
    show_grid: 是否显示网格.
    """

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    # 网格放在最底层.
    if show_grid:
        ax.set_axisbelow(True)
        ax.grid(
            True, which="major", color="0.92",
            linestyle="--", linewidth=0.6, zorder=0,
        )

    # 灰色平台带.
    if band is not None:
        _tag = band[0]
        _mean = float(band[1])
        _err = float(band[2])
        ax.axhspan(
            _mean - _err, _mean + _err,
            facecolor="0.6", alpha=0.25,
            linewidth=0, zorder=1.5,
            label=_tag,
        )
        ax.axhline(
            _mean, color="0.2", linestyle="-.", linewidth=1.6, zorder=1.6)

    # 拟合曲线带
    if fit_data is not None:
        fit_offsets = _calc_offsets(len(fit_data), fit_x_offset)
        for i, (tag, val) in enumerate(fit_data.items()):
            if val is None:
                continue
            curve_x, curve_mean, curve_err = val
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            curve_x = np.asarray(
                curve_x, dtype=np.float64) + fit_offsets[i]
            curve_mean = np.asarray(curve_mean, dtype=np.float64)
            curve_err = np.asarray(curve_err, dtype=np.float64)
            ax.fill_between(
                curve_x, curve_mean - curve_err, curve_mean + curve_err,
                facecolor=color, alpha=0.35,
                linewidth=0, zorder=1.8,
                label=tag,
            )

    # 误差棒
    data_offsets = _calc_offsets(len(raw_data), x_offset)
    for i, (tag, val) in enumerate(raw_data.items()):
        if val is None:
            continue
        data_x, y_mean, y_err = val
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.errorbar(
            np.asarray(data_x, dtype=np.float64) + data_offsets[i],
            y_mean, yerr=y_err,
            fmt="o", markersize=2, color=color, ecolor=color,
            elinewidth=0.8, capsize=3, capthick=0.8,
            label=tag, zorder=3,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc=legend_loc, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    save_fig(save_path, fig)
    plt.close(fig)


def plot_scatter(
    x: np.ndarray,
    y_data: Dict[Optional[str], np.ndarray],
    save_path: Path,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    title: Optional[str] = None,
    x_offset: float = 0.3,
    legend_loc: str = "upper right",
    # 水平参考线
    show_hline: bool = False,
    hline_y: float = 1.0,
    hline_label: Optional[str] = None,
    hline_style: str = "--",
    hline_width: float = 1.5,
) -> None:
    """
    绘制一组或多组散点数据并保存图片. 标签为 None 的组照常绘制, 但不写入图例.

    参数:
    -----
    x: 公共横坐标数组.
    y_data: 标签到 y 值数组的映射.
    save_path: 图片保存路径.

    可选项:
    -----
    xlabel, ylabel, title, legend_loc, xlim, ylim, x_offset: 坐标轴, 标题,
        图例位置, 范围和横坐标偏移步长.
    show_hline: 是否显示水平参考线.
    hline_y, hline_label, hline_style, hline_width: 参考线参数.
    """

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    # 水平参考线, 使用灰色, 不与数据取色冲突.
    if show_hline:
        ax.axhline(
            y=hline_y, color="0.2", linestyle=hline_style,
            linewidth=hline_width, zorder=1.6,
            label=hline_label,
        )

    # 散点, 按 y_data 的顺序取色和偏移.
    data_offsets = _calc_offsets(len(y_data), x_offset)

    for i, (tag, y_value) in enumerate(y_data.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.scatter(
            np.asarray(x, dtype=np.float64) + data_offsets[i], y_value,
            marker="x", color=color, s=40, linewidths=1.5,
            zorder=3, label=tag,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc=legend_loc, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    save_fig(save_path, fig)
    plt.close(fig)


def plot_hist(
    data_dict: Dict[Optional[str], np.ndarray],
    save_path: Path,
    *,
    xlabel: str = "value",
    ylabel: str = "frequency",
    title: Optional[str] = None,
    jackknife: bool = False,
    legend_loc: str = "upper right",
) -> None:
    """
    绘制一组或多组数据的直方图并保存图片. 标签为 None 的组照常绘制, 但不写入图例.

    参数:
    -----
    data_dict: 标签到一维样本数组的映射.
    save_path: 图片保存路径.

    可选项:
    -----
    xlabel, ylabel, title, legend_loc: 坐标轴标签, 标题和图例位置.
    jackknife: 参数为 jackknife 样本时置 True, 否则为 False.
    """

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    # 统一横轴范围.
    vmin = min(np.min(val) for val in data_dict.values())
    vmax = max(np.max(val) for val in data_dict.values())
    margin = (vmax - vmin) * 0.15 if vmax > vmin else 0.5
    # para: bin_edge
    hist_range = (vmin - margin, vmax + margin)
    n_bins = int(np.sqrt(len(next(iter(data_dict.values())))))

    for i, (tag, vals) in enumerate(data_dict.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        _mean = vals.mean()
        _err = sem(vals, jackknife)
        # 标签中直接带上均值(误差), 便于读数.
        label_text = None if tag is None else (
            f"{tag} {_mean:.3g}({_err:.3g})")
        ax.hist(
            vals, bins=n_bins, range=hist_range,
            color=color, alpha=0.25, edgecolor=color,
            linewidth=0.8, label=label_text,
        )

    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc=legend_loc, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    save_fig(save_path, fig)
    plt.close(fig)
