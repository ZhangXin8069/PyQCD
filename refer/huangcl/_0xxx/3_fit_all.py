#!/public/home/huangcl/.venv/bin/python
"""
fit.py

读取 fh 变换结果, 用固定长度窗口从晚到早逐点滑动, 对每个 z 做逐样本
拟合. 每拟好一个窗口就对上一个窗口做判断, 通过则选中并停止; 整轮都
不通过则记为失败. 所有 z 的选中结果合并保存到单个 npz, 并写出报告.

输入:
    - fh: <root>/01_result/data/c32p29/fh_evt{EVT}.npy
输出:
    - 选中结果: <root>/01_result/data/c32p29/fit_evt{EVT}_{left,cand,right}.npz
      与对应的 fit_evt{EVT}_*_ym.npz, 以及选中窗口 fit_window_evt{EVT}.npy
    - 汇总报告: <root>/01_result/data/c32p29/_rep_evt{EVT}.txt
"""

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

# ---- 从相邻的项目目录 98_tools 导入通用工具函数 ----

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "98_tools"))

from analysis_tools import format_mean_err, sem  # noqa: E402  # type: ignore
from fit_tools import calc_stability, fit  # noqa: E402  # type: ignore
from output_tools import save_npy, save_npz, save_txt  # noqa: E402  # type: ignore

# 2_fit_test 文件名以数字开头, 不能直接 import, 用 importlib 按模块名加载
ft = importlib.import_module("2_fit_test")  # noqa: E402  # type: ignore


@dataclass(frozen=True)
class ReadConfig:
    """读取相关的配置和路径."""

    conf_short: str
    fh_path: Path           # fh 数据文件路径


@dataclass(frozen=True)
class FitConfig:
    """拟合相关的配置."""

    nex: int                # fh 第 1 轴中使用的 nex 索引
    fit_length: int         # 拟合窗口点数
    fit_start: int          # 拟合窗口起点序列的最大值 (从晚到早递减)
    fit_stop: int           # 拟合窗口起点序列的最小值 (固定终止窗口)
    prior: dict             # 拟合参数均值字典
    tol: dict               # 拟合参数约束半径字典, 范围取 prior +- tol
    z_list: tuple


@dataclass(frozen=True)
class CheckConfig:
    """判断相关的配置."""

    chi2_min: float         # chi2/dof 下界
    chi2_max: float         # chi2/dof 上界


@dataclass(frozen=True)
class SaveConfig:
    """输出相关的配置和路径."""

    data_dir: Path          # 拟合结果和报告保存目录
    evt: int                # 用于拟合结果文件名中的 evt 编号


def init_config() -> tuple[ReadConfig, FitConfig, CheckConfig, SaveConfig]:
    """初始化 fh 拟合分析的参数和路径.

    CONF_SHORT 等全大写变量为可给用户修改的配置参数.
    返回: Read, Fit, Check, Save.
    """

    # 用户可修改的配置参数
    CONF_SHORT = "c32p29"
    EVT = 14
    NEX = 2
    FIT_LENGTH = 6
    FIT_START = 9
    Z_LIST = range(20)
    PRIOR = {'c0': 0.6}
    TOL = {'c0': 1}
    CHI2_MIN = 0.2
    CHI2_MAX = 1.7

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "01_result" / "data" / CONF_SHORT

    Read = ReadConfig(
        conf_short=CONF_SHORT,
        fh_path=data_dir / f"fh_evt{EVT}.npy",
    )
    Fit = FitConfig(
        nex=NEX,
        fit_length=FIT_LENGTH,
        fit_start=FIT_START,
        fit_stop=2 * NEX,
        prior=PRIOR,
        tol=TOL,
        z_list=Z_LIST,
    )
    Check = CheckConfig(
        chi2_min=CHI2_MIN,
        chi2_max=CHI2_MAX,
    )
    Save = SaveConfig(data_dir, EVT)

    return Read, Fit, Check, Save


def check(
        value: float, cand_c0: np.ndarray, left_c0: np.ndarray,
        right_c0: np.ndarray, Check: CheckConfig) -> tuple[bool, float, float]:
    """判断候选窗口的 chi2/dof 是否在范围内, 并计算与左右邻居的稳定性比值.

    后续可在此增改判断条件. 稳定性比值只记录, 不影响 pass 与否.

    参数:
    -----
    value: 候选窗口整体均值的 chi2/dof.
    cand_c0: 候选窗口的逐样本 c0 数组.
    left_c0: 左邻居 (更晚) 窗口的逐样本 c0 数组.
    right_c0: 右邻居 (更早) 窗口的逐样本 c0 数组.
    Check: 判断配置.

    输出:
    -----
    pass_flag: 是否符合范围.
    stab_left: 候选窗口与左邻居的稳定性比值.
    stab_right: 候选窗口与右邻居的稳定性比值.
    """

    pass_flag = bool(Check.chi2_min <= value <= Check.chi2_max)
    stab_left = calc_stability(cand_c0, left_c0)
    stab_right = calc_stability(cand_c0, right_c0)
    return pass_flag, stab_left, stab_right


def fit_window(
        fh: np.ndarray, z: int, start: int, Fit: FitConfig) -> tuple:
    """对单个 z, 固定点数窗口 [start, start+fit_length-1] 做逐样本拟合.

    输入 fh: 第 0 轴为 sample, 第 1 轴为 nex, 第 2 轴为 dt, 第 3 轴为 z.

    输出:
    -----
    fit_result: 每个样本的全部拟合参数和 chi2/dof.
    fit_mean: 全部样本均值的拟合参数和 chi2/dof.
    elapsed: 该窗口的拟合耗时 (单位 s).
    n_bound: 至少有一个参数卡在边界的样本数.
    """

    # para: t
    t_vals = np.arange(start, start + Fit.fit_length, dtype=int)

    # para: sample, t
    y_data = fh[:, Fit.nex, t_vals, z]

    t0 = time.perf_counter()
    fit_result, _, cond, _, fit_mean, n_bound = fit(
        y=y_data,
        x=t_vals,
        model=ft.fh_model,
        p0=Fit.prior,
        rol=Fit.tol,
        method="least_squares",
        jackknife=False,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"  z={z}, t=[{start}, {start + Fit.fit_length - 1}]: "
        f"condition number = {cond:.3g}, time = {elapsed:.2f}s")
    return fit_result, fit_mean, elapsed, n_bound


def fit_one_z(
        fh: np.ndarray, z: int, test_start: int, Fit: FitConfig,
        Check: CheckConfig) -> tuple:
    """对单个 z 从晚到早逐窗拟合, 每拟一个窗口判断上一个, 通过即选中.

    输入 fh: 第 0 轴为 sample, 第 1 轴为 nex, 第 2 轴为 dt, 第 3 轴为 z.

    参数:
    -----
    test_start: 本 z 的测试起点窗口; 拟合从下一个窗口 (test_start+1) 开始,
        该首窗仅作为首个候选的左邻居用于稳定性比较, 不参与判断.

    输出:
    -----
    success: 是否找到通过的窗口.
    save_data: 成功时选中窗口及其左右邻居的逐样本和均值拟合结果 (dict),
        失败时为 {}.
    summary_row: 汇总表一行的内容 (list).
    block_text: 该 z 的分表文本 (list of str).
    test_start_out: 传给下一个 z 的测试起点; 成功时为选中窗口起点, 否则不变.
    """

    param_names = list(Fit.prior)
    sep_line = "=" * 72

    # 从晚到早的窗口起点序列, 首窗 test_start+1 仅用于稳定性比较
    s_list = list(range(test_start + 1, Fit.fit_stop - 1, -1))

    records = []
    success = False
    selected = None
    selected_left = None
    selected_right = None
    for i, s in enumerate(s_list):
        fit_result, fit_mean, elapsed, n_bound = fit_window(fh, z, s, Fit)
        records.append({
            "start": s,
            "t_max": s + Fit.fit_length - 1,
            "elapsed": elapsed,
            "fit_result": fit_result,
            "fit_mean": fit_mean,
            "n_bound": n_bound,
            "stab_left": float("nan"),
            "stab_right": float("nan"),
        })

        # 拟合完第三个窗口后, 对第二个窗口做判断, 之后每次判断前一个窗口.
        if i >= 2:
            cand = records[i - 1]       # 候选窗口
            left = records[i - 2]       # 左邻居 (更晚)
            right = records[i]          # 右邻居 (更早)
            pass_flag, stab_left, stab_right = check(
                float(cand["fit_mean"]["chi2dof"]), cand["fit_result"]["c0"],
                left["fit_result"]["c0"], right["fit_result"]["c0"], Check)
            cand["stab_left"] = stab_left
            cand["stab_right"] = stab_right
            if pass_flag:
                success = True
                selected = cand
                selected_left = left
                selected_right = right
                break

    # ---- 组装该 z 的分表 ----
    lines = [sep_line, f"z = {z}", sep_line]
    table = PrettyTable()
    table.field_names = [
        "t_window", "time/s", "c0", "chi2dof", "n_bound", "ym_chi2dof",
        "stab_left", "stab_right"]
    for name in table.field_names:
        table.align[name] = "c"
    for rec in records:
        c0_arr = rec["fit_result"]["c0"]
        chi2_arr = rec["fit_result"]["chi2dof"]
        table.add_row([
            f"[{rec['start']}, {rec['t_max']}]",
            f"{rec['elapsed']:.2f}",
            format_mean_err(c0_arr.mean(), sem(c0_arr)),
            f"{chi2_arr.mean():.2f}",
            rec["n_bound"],
            f"{rec['fit_mean']['chi2dof']:.2f}",
            f"{rec['stab_left']:.2f}",
            f"{rec['stab_right']:.2f}",
        ])
    lines += table.get_string().splitlines()
    if success:
        lines += [f"success: selected window = "
                  f"[{selected['start']}, {selected['t_max']}]"]
    else:
        lines += ["fail: no window passed"]
    lines += [""]

    # ---- 汇总行与保存数据 ----
    if success:
        ym = selected["fit_mean"]
        summary_row = [z, f"[{selected['start']}, {selected['t_max']}]"]
        summary_row += [
            format_mean_err(selected["fit_result"][name].mean(),
                            sem(selected["fit_result"][name]))
            for name in param_names]
        summary_row += [
            f"{ym['chi2dof']:.2f}",
            f"{selected['stab_left']:.2f}",
            f"{selected['stab_right']:.2f}",
        ]
        save_data = {
            "left": selected_left["fit_result"],
            "cand": selected["fit_result"],
            "right": selected_right["fit_result"],
            "ym_left": selected_left["fit_mean"],
            "ym_cand": selected["fit_mean"],
            "ym_right": selected_right["fit_mean"],
            "window": [selected["start"], selected["t_max"]],
        }
    else:
        summary_row = [z, "none"] + ["nan"] * (len(param_names) + 3)
        save_data = {}

    if success:
        test_start_out = selected["start"]
    else:
        test_start_out = test_start
    return success, save_data, summary_row, lines, test_start_out


def fit_report(
        fh: np.ndarray, Read: ReadConfig, Fit: FitConfig, Check: CheckConfig,
        Save: SaveConfig) -> None:
    """对每个 z 逐窗拟合判断, 保存选中结果并写出汇总报告.

    输入 fh: 第 0 轴为 sample, 第 1 轴为 nex, 第 2 轴为 dt, 第 3 轴为 z.
    left/cand/right 的逐样本结果保存为 fit_evt{evt}_{role}.npz (sample, z),
    均值结果保存为 fit_evt{evt}_{role}_ym.npz (z), 选中窗口保存为
    fit_window_evt{evt}.npy (z, 2), 报表写到 _rep_evt{evt}.txt.
    """

    param_names = list(Fit.prior)
    sep_line = "=" * 72
    n_sample = fh.shape[0]
    n_z = len(Fit.z_list)

    # 全部 z 的结果, 失败处保持 nan; 角色顺序为 left, cand, right.
    role_keys = ("left", "cand", "right")
    array_names = param_names + ["chi2dof"]
    sample_arrays = {
        role: {name: np.full((n_sample, n_z), np.nan) for name in array_names}
        for role in role_keys}
    ymean_arrays = {
        role: {name: np.full(n_z, np.nan) for name in array_names}
        for role in role_keys}

    # 每个 z 的选中窗口, para: z, windows(t_start, t_end); 失败保持 nan
    window_arr = np.full((n_z, 2), np.nan)

    lines = [
        f"fh fit report, conf_short={Read.conf_short}, P=0",
        f"fh data: {Read.fh_path}",
        "",
    ]

    summary_rows = []
    test_start = Fit.fit_start
    for z_idx, _z in enumerate(Fit.z_list):
        success, save_data, summary_row, block_text, test_start = fit_one_z(
            fh, _z, test_start, Fit, Check)
        lines += block_text
        summary_rows.append(summary_row)
        if success:
            for role in role_keys:
                for name in array_names:
                    sample_arrays[role][name][:, z_idx] = np.asarray(
                        save_data[role][name])
                    ymean_arrays[role][name][z_idx] = save_data[
                        "ym_" + role][name]
            window_arr[z_idx] = save_data["window"]

    # ---- 汇总表放到最后, 内容全部用英文 ----
    lines += [sep_line, "  Summary Table", sep_line]
    table = PrettyTable()
    table.field_names = ["z", "window"] + list(param_names) + [
        "chi2dof", "stab_left", "stab_right"]
    for name in table.field_names:
        table.align[name] = "c"
    for row in summary_rows:
        table.add_row(row)
    lines += table.get_string().splitlines() + [""]

    report_path = Save.data_dir / f"_rep_evt{Save.evt}.txt"
    save_txt(report_path, "\n".join(lines) + "\n")

    z_arr = np.array(Fit.z_list, dtype=int)
    for role in role_keys:
        save_npz(Save.data_dir / f"fit_evt{Save.evt}_{role}.npz",
                 z=z_arr, **sample_arrays[role])
        save_npz(
            Save.data_dir / f"fit_evt{Save.evt}_{role}_ym.npz",
            z=z_arr, **ymean_arrays[role])
    save_npy(Save.data_dir / f"fit_window_evt{Save.evt}.npy", window_arr)


if __name__ == "__main__":
    Read, Fit, Check, Save = init_config()

    t_total_begin = time.perf_counter()
    print(
        f"conf_short: {Read.conf_short}, nex: {Fit.nex}, "
        f"fit_length: {Fit.fit_length}, "
        f"fit_start: {Fit.fit_start}, fit_stop: {Fit.fit_stop}")

    # para: sample, nex, dt, z
    fh = ft.load_fh(Read)

    fit_report(fh, Read, Fit, Check, Save)

    print(
        f"job done, total time: {time.perf_counter() - t_total_begin:.2f}s")
