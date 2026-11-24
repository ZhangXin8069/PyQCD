"""feyn-hell 变换与常数模型拟合工具包.

主要功能:
    CheckConfig: 窗口判断相关的配置.
    calc_fh: 由 ratio 计算 feyn-hell 变换 fh.
    fh_model: 常数模型 fh(t) = c0.
    fit_window: 对单个 z 的单个时间窗口做逐样本拟合.
    check_window: 判断候选窗口的 chi2/dof 范围及其比左邻居 (更早) 更接近 1,
        并计算与左右邻居的稳定性比值.
    fit_one_z: 对单个 z 从给定测试起点之后的窗口从晚到早逐窗拟合与判断,
        返回选中窗口及其左右邻居.
    format_table: 按调用方给出的列名和行内容渲染对齐的表格文本.

说明:
    本模块只负责同构计算和通用渲染, 不组装任何具体报表, 也不包含任何
    画图代码. 报表的列名, 行内容和文件写出, 以及全部画图函数与画图配置,
    都由各分析代码自行完成.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "98_tools"))

from fit_tools import calc_stability, fit  # noqa: E402  # type: ignore


@dataclass
class CheckConfig:
    """判断相关的配置."""
    chi2_range: list[float]     # chi2/dof 范围 [chi2_min, chi2_max]
    bad_chi2dof: float = 10.0   # 不可信边界窗口 (起点为 fit_stop) 的替代 chi2/dof


def calc_fh(ratio: np.ndarray, nex: int) -> np.ndarray:
    """
    计算 fh(t) = sum_{tau=nex}^{t-nex} R(t, tau)
               - sum_{tau=nex}^{t-1-nex} R(t-1, tau).

    两端各去掉 nex 个 tau 点后先做 tau 方向累加, 再沿 t 方向做前向差分.

    参数:
    -----
    ratio: para: sample, dt, dtau, z.
    nex: 两端各去掉的 tau 点数.

    输出:
    -----
    fh: para: sample, dt, z.
    """

    _, dt_len, _, _ = ratio.shape

    # para: sample, dt, z
    ratio_sum = np.zeros((ratio.shape[0], dt_len, ratio.shape[3]), dtype=float)
    for dt in range(2 * nex, dt_len):
        ratio_sum[:, dt] = ratio[:, dt, nex:dt - nex + 1, :].sum(axis=1)

    # para: sample, dt, z
    fh = ratio_sum - np.roll(ratio_sum, 1, axis=1)
    return fh


def fh_model(t: np.ndarray, p: dict) -> np.ndarray:
    """
    常数模型 fh(t) = c0.

    参数:
    -----
    t: 时间点, 与 fh 数据的时间轴对应.
    p: 拟合参数字典, 只使用 c0.

    输出:
    -----
    result: 模型值, 与 t 同形状.
    """

    # para: t
    result = p['c0'] * np.ones_like(t, dtype=float)
    return result


def fit_window(
        fh_z: np.ndarray, start: int, end: int,
        prior: dict, tol: dict, print_detail: bool = False) -> tuple:
    """
    对单个 z 的单个时间窗口 [start, end] 做逐样本拟合.

    参数:
    -----
    fh_z: para: sample, t.
    start: 拟合窗口左侧时间点.
    end: 拟合窗口右侧时间点.
    prior: 拟合参数均值字典.
    tol: 拟合参数约束半径字典, 范围取 prior +- tol.

    可选项:
    -----
    print_detail: 是否在拟合最后输出拟合细节, 默认 False.

    输出:
    -----
    fit_result: 每个样本的全部拟合参数和 chi2/dof.
    fit_mean: 全部样本均值的拟合参数和 chi2/dof.
    elapsed: 该窗口的拟合耗时 (单位 s).
    n_bound: 至少有一个参数卡在边界的样本数.
    """

    # para: t
    t_vals = np.arange(start, end + 1, dtype=int)

    # para: sample, t
    y_data = fh_z[:, t_vals]

    t0 = time.perf_counter()
    fit_result, _, cond, _, fit_mean, n_bound = fit(
        y=y_data,
        x=t_vals,
        model=fh_model,
        p0=prior,
        rol=tol,
        method="least_squares",
        jackknife=False,
        print_detail=print_detail,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"  t=[{start}, {end}]: "
        f"condition number = {cond:.3g}, time = {elapsed:.2f}s")

    return fit_result, fit_mean, elapsed, n_bound


def check_window(
        value: float, left_value: float, cand_c0: np.ndarray,
        left_c0: np.ndarray, right_c0: np.ndarray,
        Check: CheckConfig) -> tuple[bool, float, float]:
    """
    判断候选窗口的 chi2/dof 是否在范围内且比左邻居更接近 1,
    并计算与左右邻居的稳定性比值.

    参数:
    -----
    value: 候选窗口整体均值的 chi2/dof.
    left_value: 左邻居 (更早, t 更小) 窗口整体均值的 chi2/dof.
    cand_c0: 候选窗口的逐样本 c0 数组.
    left_c0: 左邻居 (更早, t 更小) 窗口的逐样本 c0 数组.
    right_c0: 右邻居 (更晚, t 更大) 窗口的逐样本 c0 数组.
    Check: 判断配置.

    输出:
    -----
    pass_flag: 总判据, 全部条件同时满足时为 True.
    stab_left: 候选窗口与左邻居的稳定性比值.
    stab_right: 候选窗口与右邻居的稳定性比值.
    """

    chi2_min, chi2_max = Check.chi2_range
    # 条件 1: chi2/dof 落在给定范围内
    pass_flag_1 = bool(chi2_min <= value <= chi2_max)
    # 条件 2: 候选 (中间) 窗口的 chi2/dof 比左邻居 (更早) 更接近 1
    pass_flag_2 = bool(abs(value - 1) < abs(left_value - 1))
    pass_flag = pass_flag_1 and pass_flag_2
    stab_left = calc_stability(cand_c0, left_c0)
    stab_right = calc_stability(cand_c0, right_c0)
    return pass_flag, stab_left, stab_right


def fit_one_z(
        fh_z: np.ndarray, test_start: int, fit_length: int,
        fit_stop: int, prior: dict, tol: dict,
        Check: CheckConfig) -> tuple:
    """
    对单个 z 从 test_start 之后的窗口起, 从晚到早逐窗拟合与判断.

    拟完一个窗口, 对上一个窗口 (候选) 做判断, 通过即停止.

    参数:
    -----
    fh_z: para: sample, t.
    test_start: 测试起点窗口; 拟合从下一个窗口 (test_start+1) 开始,
        窗口起点依次递减到 fit_stop.
    fit_length: 拟合窗口点数.
    fit_stop: 拟合窗口起点序列的最小值 (固定终止窗口).
    prior: 拟合参数均值字典.
    tol: 拟合参数约束半径字典, 范围取 prior +- tol.
    Check: 判断配置.

    输出:
    -----
    success: 是否找到通过的窗口.
    records: 本序列全部已拟合窗口的记录列表 (起点从晚到早), 每项含 start,
        end, time, fit_result, fit_mean, n_bound, stab_left, stab_right.
    selected: 选中窗口的记录; 失败时为 None.
    selected_left: 选中窗口的左邻居 (更早, t 更小) 记录; 失败时为 None.
    selected_right: 选中窗口的右邻居 (更晚, t 更大) 记录; 失败时为 None.
    """

    # 从晚到早的窗口起点序列, 首窗仅用于稳定性比较
    s_list = list(range(test_start + 1, fit_stop - 1, -1))

    records = []
    success = False
    selected = None
    selected_left = None
    selected_right = None
    for i, s in enumerate(s_list):
        fit_result, fit_mean, elapsed, n_bound = fit_window(
            fh_z, s, s + fit_length - 1, prior, tol)
        records.append({
            "start": s,
            "end": s + fit_length - 1,
            "time": elapsed,
            "fit_result": fit_result,
            "fit_mean": fit_mean,
            "n_bound": n_bound,
            "stab_left": float("nan"),
            "stab_right": float("nan"),
        })

        # 拟合完第三个窗口后, 对第二个窗口做判断, 之后每次判断前一个窗口.
        if i >= 2:
            cand = records[i - 1]       # 候选窗口
            left = records[i]           # 左邻居 (更早, t 更小)
            right = records[i - 2]      # 右邻居 (更晚, t 更大)
            # 最早的边界窗口 (起点为 fit_stop) 含 fh = 0 的时间点, 结果仅作
            # 参考; 其 chi2/dof 用足够大的值替代, 保证条件 2 通过.
            left_value = float(left["fit_mean"]["chi2dof"])
            if left["start"] == fit_stop:
                left_value = Check.bad_chi2dof
            pass_flag, stab_left, stab_right = check_window(
                float(cand["fit_mean"]["chi2dof"]), left_value,
                cand["fit_result"]["c0"],
                left["fit_result"]["c0"], right["fit_result"]["c0"], Check)
            cand["stab_left"] = stab_left
            cand["stab_right"] = stab_right
            if pass_flag:
                success = True
                selected = cand
                selected_left = left
                selected_right = right
                break

    return success, records, selected, selected_left, selected_right


def format_table(
        field_names: list[str], rows: list[list],
        align: str = "r") -> list[str]:
    """
    按调用方给出的列名和行内容渲染表格文本.

    参数:
    -----
    field_names: 表格列名, 由调用方定义.
    rows: 表格每一行的内容, 顺序与 field_names 对应.

    可选项:
    -----
    align: 单元格对齐方式, 默认右对齐.

    输出:
    -----
    lines: 表格文本行.
    """

    table = PrettyTable()
    table.field_names = field_names
    for name in field_names:
        table.align[name] = align
    for row in rows:
        table.add_row(row)

    return table.get_string().splitlines()
