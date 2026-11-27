"""
数据对比用的工具包.

对多组形状相同的数组 refdat 和 newdat 逐组做逐元素比较, 每组输出 nan, inf, 0 的个数和位置
对应情况, 以及绝对差和相对差的最大值与第 95 百分位数, 并判定该组数据是否对得上. 全部组比较
完成后输出所有组的总结果, 即是否所有数据都通过比较. 在给定保存目录时, 为每组画出差异的统计
分布直方图.

主要功能:
    make_mask: 按检查类型生成布尔掩码.
    check_value: 比较 nan, inf, 0 中某一类的个数和位置, 返回该类是否一致.
    print_mismatch_pos: 打印某一类不匹配元素的坐标.
    calc_diff_stats: 计算并打印绝对差和相对差的最大值与第 95 百分位数, 并判定 equal.
    plot_diff_hist: 画出绝对差和相对差的统计分布直方图.
    compare_data: 逐组执行上述检查, 末尾输出所有组的总结果, 可选画图; 默认只输出总结果.

相等判定:
    单组的 data_equal: nan, inf, 0 三类都一致, 且最大相对误差 max_rel 小于 REL_TOL.
    all_equal: 所有组的 data_equal 都为真.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from analysis_tools import plot_hist

# 匹配判定用的相对差阈值
REL_TOL = 1e-5  # 6位有效数字相等

# 分隔线长度
SEP_LEN = 10


def make_mask(arr: np.ndarray, kind: str) -> np.ndarray:
    """
    按检查类型生成 arr 的布尔掩码.

    参数:
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


def check_value(ref: np.ndarray, new: np.ndarray, kind: str,
                show_details: bool = True) -> bool:
    """
    比较 ref 和 new 中某一类数值的个数, 以及位置是否逐元素对应, 并打印结果.

    参数:
    -----
    ref, new: 形状相同的数组.
    kind: 检查类型, 可选 "nan", "inf", "zero".

    可选项:
    -----
    show_details: 是否打印该类数值的比较细节.

    输出:
    -----
    equal: 该类数值的位置是否逐元素对应.
    """
    mask_ref, mask_new = make_mask(ref, kind), make_mask(new, kind)
    equal = np.array_equal(mask_ref, mask_new)
    if show_details:
        print(f"{kind}(ref/new): {np.count_nonzero(mask_ref)}/{np.count_nonzero(mask_new)}    "
              f"mismatch={np.count_nonzero(mask_ref != mask_new)}    "
              f"equal={equal}")
    return equal


def print_mismatch_pos(ref: np.ndarray, new: np.ndarray, tag: str, kind: str) -> None:
    """
    打印 ref 和 new 中某一类数值不匹配元素的坐标, 按 ref 独有和 new 独有分开输出.

    参数:
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


def calc_diff_stats(ref: np.ndarray, new: np.ndarray,
                    show_details: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 new 相对 ref 的绝对差和相对差, 统计最大值与第 95 百分位数, 并打印结果.

    参数:
    -----
    ref, new: 形状相同的复数数组.

    可选项:
    -----
    show_details: 是否打印各项统计细节.

    输出:
    -----
    abs_diff, rel_diff: 展平后的绝对差和相对差数组, 相对差只统计 ref 不为 0 的元素.
    """
    mask_valid_ref = ~(make_mask(ref, "nan") | make_mask(ref, "inf")
                       | make_mask(ref, "zero"))
    mask_valid_new = ~(make_mask(new, "nan") | make_mask(new, "inf")
                       | make_mask(new, "zero"))
    abs_diff = np.abs(new - ref).ravel()
    ref_abs = np.abs(ref).ravel()
    rel_diff = abs_diff[ref_abs > 0] / ref_abs[ref_abs > 0]
    if show_details:
        print(f"valid(ref/new): {np.count_nonzero(mask_valid_ref)}/"
              f"{np.count_nonzero(mask_valid_new)}")
        print(f"    max_abs={np.max(abs_diff):.3g}    q95_abs={np.percentile(abs_diff, 95):.3g}    "
              f"max_rel={np.max(rel_diff):.3g}    q95_rel={np.percentile(rel_diff, 95):.3g}    "
              f"equal={np.max(rel_diff) < REL_TOL}")
    return abs_diff, rel_diff


def plot_diff_hist(abs_diff: np.ndarray, rel_diff: np.ndarray, save_dir: Union[str, Path],
                   tag: str) -> None:
    """
    画出绝对差和相对差的统计分布直方图, 保存到指定目录.

    参数:
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


def compare_data(pairs: Sequence[Tuple[np.ndarray, np.ndarray, str]],
                 save_dir: Optional[Union[str, Path]] = None,
                 show_details: bool = False) -> bool:
    """
    逐组比较多组数据, 每组依次比较 nan, inf, 0 的个数和位置, 输出绝对差和相对差的统计量,
    并判定该组是否相等, 最后输出所有组的总结果.

    参数:
    -----
    pairs: 待比较的数据组, 每项为 (refdat, newdat, tag), refdat 和 newdat 形状相同,
        tag 用于标识该组数据.

    可选项:
    save_dir: 差异分布直方图的保存目录, 给定则画图, 默认不画.
    show_details: 是否打印每组数据的比较细节, 默认只输出所有组的总结果.

    输出:
    -----
    all_equal: 所有组是否都通过比较.
    """
    equals = []
    max_rel_diff = 0.0
    print()
    for ref, new, tag in pairs:
        if ref.shape != new.shape:
            raise ValueError(
                f"tag {tag}: shape mismatch: ref {ref.shape} vs new {new.shape}")
        if show_details:
            print(f"{tag}:")
        kind_equals = [check_value(ref, new, kind, show_details)
                       for kind in ("nan", "inf", "zero")]
        abs_diff, rel_diff = calc_diff_stats(ref, new, show_details)
        max_rel_diff = max(max_rel_diff, float(np.max(rel_diff)))
        data_equal = all(kind_equals) and np.max(rel_diff) < REL_TOL
        if show_details:
            print(f"this data equal={data_equal}")
            print("-" * SEP_LEN)
        equals.append(data_equal)
        if save_dir is not None:
            plot_diff_hist(abs_diff, rel_diff, save_dir, tag)
    all_equal = all(equals)
    print(f"all data equal={all_equal}; max rel_diff={max_rel_diff:.3g}\n")
    return all_equal
