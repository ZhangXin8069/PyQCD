#!/public/home/huangcl/.venv/bin/python
"""读取 state 0 的 GEVP 本征值, 比较三个 tref 并拟合 tref=5.

输入: 01_result/<conf_short>/eigval/state0_tref<tref>.npy,
    数据的 para: sample, t.
输出: 拟合结果, 拟合报告和有效能量图, 保存在 01_result 或 00_test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import gvar as gv
import numpy as np
from prettytable import PrettyTable

# ---- 从上级目录 98_tools 导入通用函数 ----
# sys.path.insert 必须在 import 之前, 否则运行时 ImportError
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "98_tools"))

# noqa: E402 抑制 autopep8 的 import 顺序检查, 防止 import 被上移到 sys.path.insert 之前
# type: ignore 抑制 Pylance 无法解析导入的报错
from analysis_tools import fit, plot_errbar, sem  # noqa: E402  # type: ignore


# ==================== 数据初始化 ====================

@dataclass(frozen=True)
class ReadConfig:
    """保存数据读取相关的参数及路径."""

    tref_list: tuple[int, ...]
    eigval_dir: Path


@dataclass(frozen=True)
class FitConfig:
    """保存拟合相关的参数及路径."""

    tref: int
    tmin: int
    tmin_list: list[int]
    tmax: int
    model_choice: int
    prior: dict[str, gv.GVar]
    use_prior: bool
    test: bool
    test_Nsample: int
    energy_unit: float
    fit_dir: Path


@dataclass(frozen=True)
class PlotConfig:
    """保存画图相关的参数及路径."""

    xlim: list[float]
    ylim: list[float]
    x_offset: float
    figure_dir: Path
    conf_short: str


def init_config() -> tuple[ReadConfig, FitConfig, PlotConfig]:
    """根据 conf_short 初始化 state 0 比较和拟合配置.

    函数内全大写的变量是给用户修改的参数, 其余为根据它们派生的量.
    """

    # 与组态无关的参数
    CONF_SHORT = "c32p29"
    USE_PRIOR = True  # True 用 prior 拟合, False 用 prior 的均值作为 p0 拟合
    TEST = False  # True 时只拟合前 TEST_NSAMPLE 个样本, 输出到 00_test
    TEST_NSAMPLE = 100
    FIT_PRIOR = {
        "A0": gv.gvar(0.8, 0.3),
        "E0": gv.gvar(1.1, 0.2),
        "A1": gv.gvar(0.2, 0.5),
        "dE": gv.gvar(0.35, 0.2),
    }

    # 与组态有关的参数
    if CONF_SHORT == "c32p29":
        TREF = 5
        FIT_TMIN = 7
        FIT_TMAX = 17
        MODEL_CHOICE = 2  # 1 为单态拟合, 2 为两态拟合
        # TREF_LIST = [TREF - 1, TREF, TREF + 1]
        # FIT_TMIN_LIST = [FIT_TMIN - 1, FIT_TMIN, FIT_TMIN + 1]
        TREF_LIST = [TREF]
        FIT_TMIN_LIST = [FIT_TMIN]
        XLIM = [2.5, 20.5]
        YLIM = [1.5, 2.5]
        X_OFFSET = 0.15
        lattice_spacing = 0.1053
    else:
        raise ValueError(f"unsupported conf_short: {CONF_SHORT}")

    # 后续不再有需要修改的参数
    # 路径拼接, 并生成相应的文件夹
    # 读取目录固定在 01_result; 输出目录在 test 模式下改为 00_test.
    input_dir = Path(__file__).resolve().parent / "01_result"
    output_dir = Path(__file__).resolve().parent / \
        ("00_test" if TEST else "01_result")
    eigval_dir = input_dir / CONF_SHORT / "eigval"
    fit_dir = output_dir / CONF_SHORT / "fit" / f"tref{TREF}"
    figure_dir = output_dir / "fig"
    fit_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    Read = ReadConfig(
        tref_list=TREF_LIST,
        eigval_dir=eigval_dir,
    )
    Fit = FitConfig(
        tref=TREF,
        tmin=FIT_TMIN,
        tmin_list=FIT_TMIN_LIST,
        tmax=FIT_TMAX,
        model_choice=MODEL_CHOICE,
        prior=dict(FIT_PRIOR),
        use_prior=USE_PRIOR,
        test=TEST,
        test_Nsample=TEST_NSAMPLE,
        energy_unit=0.1973 / lattice_spacing,
        fit_dir=fit_dir,
    )
    Plot = PlotConfig(
        xlim=XLIM,
        ylim=YLIM,
        x_offset=X_OFFSET,
        figure_dir=figure_dir,
        conf_short=CONF_SHORT,
    )

    return Read, Fit, Plot

# ==================== 数据读取 ====================


def load_state0_eigval(Read: ReadConfig) -> dict[int, np.ndarray]:
    """读取 state 0 的 GEVP 本征值.

    输入 Read: 读取相关配置, 提供 tref_list 和 eigval_dir.
    返回 eigval, key 为 tref, value 的 para: sample, t.
    """

    eigval = {}
    last_file_path = None
    for tref in Read.tref_list:
        file_path = Read.eigval_dir / f"state0_tref{tref}.npy"
        if not file_path.is_file():
            raise FileNotFoundError(f"not found: {file_path}")
        eigval[tref] = np.load(file_path)
        last_file_path = file_path
    print(f"last loaded file: {last_file_path}, shape: {eigval[tref].shape}")
    return eigval


# ==================== 拟合模型 ====================
def model1(t: np.ndarray, p: dict[str, float], tref: int) -> np.ndarray:
    """计算单态拟合模型.

    输入 t: 拟合时间点.
    输入 p: 拟合参数, 包含 A0 和 E0.
    输入 tref: 参考时间.
    输出: 模型值, 物理轴为 t.
    """

    tau = np.asarray(t) - tref
    return p["A0"] * np.exp(-p["E0"] * tau)


def model2(t: np.ndarray, p: dict[str, float], tref: int) -> np.ndarray:
    """计算两态拟合模型.

    输入 t: 拟合时间点.
    输入 p: 拟合参数, 包含 A0, E0, A1 和 dE.
    输入 tref: 参考时间.
    输出: 模型值, 物理轴为 t.
    """

    tau = np.asarray(t) - tref
    return (
        p["A0"] * np.exp(-p["E0"] * tau)
        * (1.0 + p["A1"] * np.exp(-p["dE"] * tau))
    )


def fit_one_window(
    eigval: np.ndarray,
    tmin: int,
    tmax: int,
    tref: int,
    model_choice: int,
    prior: dict,
    use_prior: bool,
    test: bool,
    test_Nsample: int,
) -> dict[str, object]:
    """在给定时间窗口内对 tref 的 state 0 本征值做拟合.

    输入 eigval 的 para: sample, t.
    输入 tref: 参考时间, 用于模型中的 tau = t - tref.
    输入 model_choice: 1 为单态, 2 为两态.
    输入 prior: 全部参数的先验, 在函数内按 model_choice 截取子集.
    输入 use_prior: True 用 prior 拟合, False 用 prior 的均值作为 p0 拟合.
    输入 test: True 时只拟合前 test_Nsample 个样本.
    输入 test_Nsample: test 模式下拟合的样本数.
    返回 tmin, tmax, fit (每个 bootstrap sample 的拟合参数和 chi2/dof),
        info (均值数据的 lsqfit 结果) 以及 fit_mean (均值数据的拟合参数和 chi2/dof).
    """

    t_fit = np.arange(tmin, tmax + 1)
    # para: sample, t
    y_fit = np.real(eigval[:, t_fit])

    # 按 model_choice 选择模型, 并截取对应的 prior 子集.
    if model_choice == 1:
        def model(t, p): return model1(t, p, tref)
        prior = {name: prior[name] for name in ("A0", "E0")}
    elif model_choice == 2:
        def model(t, p): return model2(t, p, tref)
        prior = dict(prior)
    else:
        raise ValueError(f"unsupported model_choice: {model_choice}")

    fit_result, _, _, mean_fit_info, fit_mean = fit(
        y_coor=y_fit,
        x_coor=t_fit,
        model=model,
        para=prior,
        use_prior=use_prior,
        test=test,
        test_Nsample=test_Nsample,
    )
    return {
        "tmin": tmin,
        "tmax": tmax,
        "fit": fit_result,
        "info": mean_fit_info,
        "fit_mean": fit_mean,
    }


def fit_report(
    Fit: FitConfig,
    fit_results: list[dict[str, object]],
) -> None:
    """保存并打印拟合报告, 按 Fit.model_choice 决定参数列.

    输入 Fit: 拟合相关配置, 提供 tref, tmin_list, tmax, model_choice,
        energy_unit 和 fit_dir.
    输入 fit_results: 各时间窗口的拟合结果.
    """

    single_state = (Fit.model_choice == 1)
    sep = "=" * 72

    # 第一段: 总体拟合信息.
    n_samples = len(fit_results[0]["fit"]["chi2"])
    fit_method = "prior" if Fit.use_prior else "p0"
    lines = [
        "state 0 fit report",
        f"n samples: {n_samples}",
        f"t_ref: {Fit.tref}",
        f"model choice: {Fit.model_choice}",
        f"fit method: {fit_method}",
    ]

    # 第二段: 汇总各拟合窗口的样本均值和误差.
    lines += ["", sep]
    table = PrettyTable()
    if single_state:
        table.field_names = [
            "tmin", "tmax", "E0(GeV)", "A0", "\\bar{chi2/dof}", "chi2/dof"
        ]
    else:
        table.field_names = [
            "tmin", "tmax", "E0(GeV)", "dE(GeV)", "A0", "A1",
            "\\bar{chi2/dof}", "chi2/dof"
        ]

    for result in fit_results:
        fit_samples = result["fit"]
        e0 = fit_samples["E0"] * Fit.energy_unit
        a0 = fit_samples["A0"]
        e0_text = f"{np.mean(e0):.3f}({int(round(sem(e0) * 1000))})"
        a0_text = f"{np.mean(a0):.3f}({int(round(sem(a0) * 1000))})"
        # 样本 chi2/dof 的均值, 以及 y 均值拟合的 chi2/dof.
        chi2_dof_bar_text = f"{np.mean(fit_samples['chi2']):.3f}"
        chi2_dof_text = f"{result['fit_mean']['chi2']:.3f}"
        if single_state:
            table.add_row(
                [
                    result["tmin"],
                    result["tmax"],
                    e0_text,
                    a0_text,
                    chi2_dof_bar_text,
                    chi2_dof_text,
                ]
            )
            continue

        de = fit_samples["dE"] * Fit.energy_unit
        a1 = fit_samples["A1"]
        de_text = f"{np.mean(de):.3f}({int(round(sem(de) * 1000))})"
        a1_text = f"{np.mean(a1):.3f}({int(round(sem(a1) * 1000))})"
        table.add_row(
            [
                result["tmin"],
                result["tmax"],
                e0_text,
                de_text,
                a0_text,
                a1_text,
                chi2_dof_bar_text,
                chi2_dof_text,
            ]
        )

    lines += table.get_string().splitlines()

    # 第三段: 逐个窗口打印均值数据的拟合报告.
    for result in fit_results:
        lines += [
            "",
            sep,
            f"fit window: tmin={result['tmin']}, tmax={result['tmax']}",
            "",
        ]
        lines += str(result["info"].format(maxline=True)).splitlines()

    report_path = Fit.fit_dir / f"rep_model{Fit.model_choice}.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"saved fit report: {report_path}")


# ==================== 画图 ====================
def plot_state0_effective_energy(
    Read: ReadConfig,
    Fit: FitConfig,
    Plot: PlotConfig,
    eigval: dict[int, np.ndarray],
    fit_results: list[dict[str, object]],
) -> None:
    """把三个 tref 的有效能量和拟合 E0 灰色带画在同一张图中.

    输入 Read: 读取相关配置, 提供 tref_list.
    输入 Fit: 拟合相关配置, 提供 tmin 和 energy_unit.
    输入 Plot: 画图相关配置, 提供坐标范围和 figure_dir.
    输入 eigval, key 为 tref, value 的 para: sample, t.
    """

    effective_energy_data = {}
    for tref in Read.tref_list:
        eigenvalue = eigval[tref]
        with np.errstate(divide="ignore", invalid="ignore"):
            # para: sample, t; m(t) = log(lambda(t-1) / lambda(t))
            effective_energy = np.log(
                np.roll(eigenvalue, 1, axis=1) / eigenvalue
            )
        effective_energy_data[f"tref={tref}"] = (
            effective_energy.mean(axis=0),
            sem(effective_energy),
        )

    n_time = next(iter(eigval.values())).shape[1]
    x = np.arange(n_time)
    band_x = None
    band_y_down = None
    band_y_up = None
    band_label = None
    for result in fit_results:
        if int(result["tmin"]) != Fit.tmin:
            continue
        e0_samples = result["fit"]["E0"]
        e0_mean = np.mean(e0_samples) * Fit.energy_unit
        e0_err = sem(e0_samples) * Fit.energy_unit
        tmin = int(result["tmin"])
        tmax = int(result["tmax"])
        band_x = np.arange(tmin, tmax + 1)
        band_y_down = np.full(tmax - tmin + 1, e0_mean - e0_err)
        band_y_up = np.full(tmax - tmin + 1, e0_mean + e0_err)
        band_label = f"E0: {e0_mean:.3f}({int(round(e0_err * 1000))})"

    figure_path = Plot.figure_dir / f"fit_tref{Fit.tref}_{Plot.conf_short}.png"
    plot_errbar(
        x,
        {
            label: (y_mean * Fit.energy_unit, y_err * Fit.energy_unit)
            for label, (y_mean, y_err) in effective_energy_data.items()
        },
        figure_path,
        xlabel="t / a",
        ylabel="effective mass (GeV)",
        xlim=Plot.xlim,
        ylim=Plot.ylim,
        x_offset=Plot.x_offset,
        title="state 0 effective mass",
        show_band=True,
        band_x=band_x,
        band_y_down=band_y_down,
        band_y_up=band_y_up,
        band_label=band_label,
    )
    print(f"saved figure: {figure_path}")


if __name__ == "__main__":
    """按顺序读取数据, 选择模型拟合并生成报告和单张图."""

    Read, Fit, Plot = init_config()

    t_start = time.perf_counter()
    state0_eigval = load_state0_eigval(Read)
    print(f"load time: {(time.perf_counter() - t_start):.2f} s")

    all_results = []
    for tmin in Fit.tmin_list:
        t_start = time.perf_counter()
        fit_result = fit_one_window(
            state0_eigval[Fit.tref],
            tmin,
            Fit.tmax,
            Fit.tref,
            Fit.model_choice,
            Fit.prior,
            Fit.use_prior,
            Fit.test,
            Fit.test_Nsample,
        )
        print(f"fit time: {(time.perf_counter() - t_start):.2f} s")
        all_results.append(fit_result)

        # 仅在 prior 拟合时保存样本数据.
        if Fit.use_prior:
            name = f"t{tmin}_{Fit.tmax}_model{Fit.model_choice}"
            npz_path = Fit.fit_dir / f"fit_{name}.npz"
            np.savez(npz_path, **fit_result["fit"])
            print(f"saved fit result: {npz_path}")

            # 均值数据的拟合结果, 逐行记录 key 和数值.
            lines = [
                f"{key}: {value}" for key, value in fit_result["fit_mean"].items()
            ]
            txt_path = Fit.fit_dir / f"fitmean_{name}.txt"
            txt_path.write_text("\n".join(lines) + "\n")
            print(f"saved fit mean: {txt_path}")

    fit_report(Fit, all_results)
    t_start = time.perf_counter()
    plot_state0_effective_energy(
        Read,
        Fit,
        Plot,
        state0_eigval,
        all_results,
    )
    print(f"plot time: {(time.perf_counter() - t_start):.2f} s")
