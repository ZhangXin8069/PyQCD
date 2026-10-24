"""数据对比用的工具包.

对两套形状相同的数组 ref 和 new 做逐元素比较, 输出 nan, inf, 0 的个数和位置对应情况,
以及绝对差和相对差的最大值与第 95 百分位数, 并判定数据是否对得上. 每个检查函数内部
直接打印结果, 总入口 compare_data 依次调用全部检查, 在给定保存目录时画出差异的统计
分布直方图, 并在末尾输出整体的 all_match 判定.

主要功能:
    check_value: 比较 nan, inf, 0 中某一类的个数和位置, 返回该类是否一致.
    print_mismatch_pos: 打印某一类不匹配元素的坐标.
    calc_diff_stats: 计算并打印绝对差和相对差的最大值与第 95 百分位数, 并判定 match.
    plot_diff_hist: 画出绝对差和相对差的统计分布直方图.
    compare_data: 依次执行上述检查, 末尾输出整体 all_match 判定, 可选画图.

匹配判定:
    match: 最大相对误差 max_rel 小于 REL_TOL.
    all_match: nan, inf, 0 三类都一致, 且 match 为真.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from analysis_tools import plot_hist

# 匹配判定用的相对差阈值
REL_TOL = 1e-7


def make_mask(arr: np.ndarray, kind: str) -> np.ndarray:
    """
    按检查类型生成 arr 的布尔掩码.

    输入:
    -----
    arr: 待检查的数组.
    kind: 检查类型, 可选 "nan", "inf", "zero".

    输出:
    -----
    mask: 与 arr 形状相同的布尔数组, True 表示该位置满足检查条件.
    """
    if kind == "nan":
        return np.isnan(arr)
    if kind == "inf":
        return np.isinf(arr)
    if kind == "zero":
        return arr == 0
    raise ValueError(f"unknown kind: {kind}")


def check_value(ref: np.ndarray, new: np.ndarray, tag: str, kind: str) -> bool:
    """
    比较 ref 和 new 中某一类数值的个数, 以及位置是否逐元素对应, 并打印结果.

    输入:
    -----
    ref, new: 形状相同的数组.
    tag: 打印用的标签, 用于标识当前比较的对象.
    kind: 检查类型, 可选 "nan", "inf", "zero".

    输出:
    -----
    match: 该类数值的位置是否逐元素对应.
    """
    mask_ref, mask_new = make_mask(ref, kind), make_mask(new, kind)
    match = np.array_equal(mask_ref, mask_new)
    print(f"{tag:>10}    {kind}(ref/new): {np.count_nonzero(mask_ref)}/{np.count_nonzero(mask_new)}    "
          f"mismatch={np.count_nonzero(mask_ref != mask_new)}    "
          f"match={match}")
    return match


def print_mismatch_pos(ref: np.ndarray, new: np.ndarray, tag: str, kind: str) -> None:
    """
    打印 ref 和 new 中某一类数值不匹配元素的坐标, 按 ref 独有和 new 独有分开输出.

    输入:
    -----
    ref, new: 形状相同的数组.
    tag: 打印用的标签, 用于标识当前比较的对象.
    kind: 检查类型, 可选 "nan", "inf", "zero".
    """
    mask_ref, mask_new = make_mask(ref, kind), make_mask(new, kind)
    pos_only_ref = np.argwhere(mask_ref & (~mask_new))
    pos_only_new = np.argwhere((~mask_ref) & mask_new)
    if len(pos_only_ref) > 0:
        print(
            f"{tag:>10}    {kind} only_ref ({len(pos_only_ref)}): {pos_only_ref.tolist()}")
    if len(pos_only_new) > 0:
        print(
            f"{tag:>10}    {kind} only_new ({len(pos_only_new)}): {pos_only_new.tolist()}")


def calc_diff_stats(ref: np.ndarray, new: np.ndarray, tag: str) -> tuple:
    """
    计算 new 相对 ref 的绝对差和相对差, 统计最大值与第 95 百分位数, 并打印结果.

    输入:
    -----
    ref, new: 形状相同的复数数组.
    tag: 打印用的标签, 用于标识当前比较的对象.

    输出:
    -----
    abs_diff, rel_diff: 展平后的绝对差和相对差数组, 相对差只统计 ref 不为 0 的元素.
    """
    abs_diff = np.abs(new - ref).ravel()
    ref_abs = np.abs(ref).ravel()
    rel_diff = abs_diff[ref_abs > 0] / ref_abs[ref_abs > 0]
    max_rel = np.max(rel_diff)
    print(f"{tag:>10}    max_abs={np.max(abs_diff):.3g}    q95_abs={np.percentile(abs_diff, 95):.3g}    "
          f"max_rel={max_rel:.3g}    q95_rel={np.percentile(rel_diff, 95):.3g}    "
          f"match={max_rel < REL_TOL}")
    return abs_diff, rel_diff


def plot_diff_hist(abs_diff: np.ndarray, rel_diff: np.ndarray, save_dir: Union[str, Path],
                   tag: str) -> None:
    """
    画出绝对差和相对差的统计分布直方图, 保存到指定目录.

    输入:
    -----
    abs_diff, rel_diff: 展平后的绝对差和相对差数组.
    save_dir: 图片保存目录, 文件名按 tag 自动生成.
    tag: 打印用的标签, 用于标识当前比较的对象, 也用于生成图片文件名.
    """
    tag_key = tag.replace(" ", "_")
    save_dir = Path(save_dir)
    plot_hist({"abs_diff": abs_diff}, save_dir / f"abs_diff_{tag_key}.png",
              xlabel="abs diff", ylabel="count", title=tag)
    plot_hist({"rel_diff": rel_diff}, save_dir / f"rel_diff_{tag_key}.png",
              xlabel="rel diff", ylabel="count", title=tag)


def compare_data(ref: np.ndarray, new: np.ndarray, tag: str,
                 save_dir: Optional[Union[str, Path]] = None) -> tuple:
    """
    依次比较 nan, inf, 0 的个数和位置, 输出绝对差和相对差的统计量, 并判定整体是否对得上.

    输入:
    -----
    ref, new: 形状相同的数组.
    tag: 打印用的标签, 用于标识当前比较的对象.

    可选项:
    save_dir: 差异分布直方图的保存目录, 给定则画图, 默认不画.

    输出:
    -----
    abs_diff, rel_diff: 展平后的绝对差和相对差数组, 供调用方后续使用.
    """
    if ref.shape != new.shape:
        raise ValueError(f"shape mismatch: ref {ref.shape} vs new {new.shape}")
    matches = []
    for kind in ("nan", "inf", "zero"):
        matches.append(check_value(ref, new, tag, kind))
        print()
    abs_diff, rel_diff = calc_diff_stats(ref, new, tag)
    print()
    all_match = all(matches) and np.max(rel_diff) < REL_TOL
    print(f"{tag:>10}    all_match={all_match}")
    print()
    if save_dir is not None:
        plot_diff_hist(abs_diff, rel_diff, save_dir, tag)
    return abs_diff, rel_diff
