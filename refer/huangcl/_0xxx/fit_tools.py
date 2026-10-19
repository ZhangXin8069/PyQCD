#!/public/home/huangcl/.venv/bin/python
"""通用拟合工具包: 用可切换的拟合工具包对 bootstrap 样本逐个拟合.

主要功能:
    calc_cov: 计算协方差矩阵和条件数.
    calc_chi2: 计算数据的 chi2.
    calc_chi2_dof: 计算 chi2/dof, chi2 和自由度.
    calc_stability: 计算两套数据的参数稳定性比值.
    fit_one: 用指定工具包拟合单个样本.
    fit: 先用全体样本均值拟合, 再以该结果作为初值对每个样本逐个拟合.

支持的拟合工具包: least_squares (默认), iminuit, lsqfit.
约束: fit 传入 rol 字典时, 参数范围取 p0 +- rol; least_squares 用 bounds,
iminuit 用 limits, lsqfit 自动转成高斯先验 (软约束).
calc_cov, calc_chi2 和 calc_chi2_dof 从 98_tools/analysis_tools.py 原样复制.
"""

from typing import Callable, Dict, Tuple

import numpy as np

from analysis_tools import format_mean_err, sem  # type: ignore

METHODS = ("least_squares", "iminuit", "lsqfit")


# ==================== 协方差与卡方 ====================

def calc_cov(arr: np.ndarray, jackknife: bool = False) -> Tuple[np.ndarray, float]:
    """
    计算协方差矩阵和条件数.

    参数:
    -----
    arr: 二维数据数组, 第 0 轴为 sample, 第 1 轴为数据点.

    可选项:
    jackknife: 是否按 jackknife 样本计算协方差.

    输出:
    -----
    cov: 协方差矩阵.
    cond: 协方差矩阵的条件数.
    """

    diff = arr - arr.mean(0)
    n = arr.shape[0]

    cov = diff.T @ diff
    if jackknife:
        cov *= (n - 1) / n
    else:
        cov /= n

    eig = np.linalg.eigvalsh(cov)
    cond = eig[-1] / eig[0]
    return cov, cond


def calc_chi2(
    y_data: np.ndarray,
    y_fit: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    根据数据, 拟合值和协方差矩阵计算 chi2.

    参数:
    -----
    y_data: 原始数据数组.
    y_fit: 模型拟合值数组.
    cov: 与数据对应的协方差矩阵.

    输出:
    -----
    chi2: `diff.T @ cov^{-1} @ diff` 的计算结果.
    """

    diff = y_data - y_fit
    return diff @ np.linalg.solve(cov, diff)


def calc_chi2_dof(
    y_data: np.ndarray,
    y_fit: np.ndarray,
    cov: np.ndarray,
    n_params: int,
) -> Tuple[float, float, int]:
    """
    计算 chi2/dof, chi2 和自由度.

    参数:
    -----
    y_data: 原始数据数组.
    y_fit: 模型拟合值数组.
    cov: 与数据对应的协方差矩阵.
    n_params: 拟合参数数量.

    输出:
    -----
    chi2_dof: chi2/dof.
    chi2: chi2 值.
    dof: 自由度.
    """
    chi2 = calc_chi2(y_data, y_fit, cov)
    dof = len(y_data) - n_params
    return chi2 / dof, chi2, dof


# ==================== 拟合 ====================

def fit_one(
    y: np.ndarray,
    x,  # 单个自变量或多自变量, 由 model 解释
    model: Callable,
    p0: dict,
    cov: np.ndarray,
    chol: np.ndarray,
    method: str = "least_squares",
    rol: dict = None,
) -> Tuple[Dict[str, float], object]:
    """
    用指定工具包拟合单个样本, 只最小化数据 chi2.

    参数:
    -----
    y: 一维数据数组.
    x: 与数据点对应的自变量. 可以是一个自变量 (一维数组), 也可以是多个自变量
       (如 (n_var, n_data) 的二维数组或数组列表), 具体格式由 model 决定.
    model: 接收 `(x, p)` 并返回模型值数组的模型函数, p 为参数字典.
    p0: 拟合参数初值字典.
    cov: 数据的协方差矩阵.
    chol: cov 的 Cholesky 因子.

    可选项:
    -----
    method: 拟合工具包, 可选 least_squares, iminuit, lsqfit.
    rol: 约束半径字典, 与 p0 同键, 参数范围取 p0 +- rol; None (默认) 不加约束.
         least_squares 用 bounds, iminuit 用 limits, lsqfit 自动转成高斯先验
         (软约束, sigma = rol).

    输出:
    -----
    pmean: 拟合参数字典.
    fit_info: 对应工具包的原始拟合结果.
    at_bound: 参数字典, 表示对应参数是否卡在边界; lsqfit 恒为 False.
    """

    names = list(p0)
    x0 = np.array([p0[name] for name in names], dtype=float)
    at_bound = {name: False for name in names}

    if method == "least_squares":
        from scipy.optimize import least_squares

        if rol is None:
            bounds = (-np.inf, np.inf)
        else:
            bounds = (
                [p0[name] - rol[name] for name in names],
                [p0[name] + rol[name] for name in names],
            )

        def residual(p):
            return np.linalg.solve(chol, model(x, dict(zip(names, p))) - y)

        fit_info = least_squares(
            residual,
            x0=x0,
            bounds=bounds,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=1000,
        )
        if rol is not None:
            for i, name in enumerate(names):
                if fit_info.active_mask[i] != 0:
                    at_bound[name] = True
        return dict(zip(names, fit_info.x)), fit_info, at_bound

    if method == "iminuit":
        from iminuit import Minuit

        def objective(*args):
            return float(calc_chi2(y, model(x, dict(zip(names, args))), cov))

        fit_info = Minuit(objective, *x0, name=names)
        fit_info.errordef = 1.0
        fit_info.tol = 1e-8
        if rol is not None:
            fit_info.limits = [
                (p0[name] - rol[name], p0[name] + rol[name])
                for name in names]
        fit_info.migrad()
        if rol is not None:
            for name in names:
                val = fit_info.values[name]
                if (np.isclose(val, p0[name] - rol[name], rtol=1e-8, atol=1e-12)
                        or np.isclose(val, p0[name] + rol[name], rtol=1e-8, atol=1e-12)):
                    at_bound[name] = True
        return {name: fit_info.values[name] for name in names}, fit_info, at_bound

    if method == "lsqfit":
        import gvar as gv
        from lsqfit import nonlinear_fit

        data = (x, gv.gvar(y, cov))
        if rol is None:
            fit_info = nonlinear_fit(
                data=data,
                p0=p0,
                fcn=model,
                svdcut=None,
            )
        else:
            # lsqfit 没有硬约束, rol 自动转成高斯先验 (软约束).
            prior = {name: gv.gvar(p0[name], rol[name]) for name in names}
            fit_info = nonlinear_fit(
                data=data,
                p0=p0,
                prior=prior,
                fcn=model,
                svdcut=None,
            )
        return {name: float(fit_info.pmean[name]) for name in names}, fit_info, at_bound

    raise ValueError(f"unsupported method: {method}")


def fit(
    y: np.ndarray,
    x,  # 单个自变量或多自变量, 由 model 解释
    model: Callable,
    p0: dict,
    rol: dict = None,
    method: str = "least_squares",
    jackknife: bool = False,
    print_detail: bool = False,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    float,
    object,
    Dict[str, float],
]:
    """
    先用全体样本均值拟合, 再以均值拟合结果作为初值对每个 sample 执行非线性拟合.

    参数:
    -----
    y: 二维数据数组, 第 0 轴为 sample, 第 1 轴为数据点.
    x: 与数据点对应的自变量. 可以是一个自变量 (一维数组), 也可以是多个自变量
       (如 (n_var, n_data) 的二维数组或数组列表), 具体格式由 model 决定.
    model: 接收 `(x, p)` 并返回模型值数组的模型函数.
    p0: 拟合参数初值字典.
    rol: 约束半径字典, 须与 p0 同键; None (默认) 表示不加约束. 约束的具体
         构造 (bounds / limits / 高斯先验) 由 fit_one 按 method 完成.

    可选项:
    -----
    method: 拟合工具包, 可选 least_squares (默认), iminuit, lsqfit.
    jackknife: 是否按 jackknife 样本计算协方差.
    print_detail: 是否在拟合最后输出拟合细节, 默认 False.

    输出:
    -----
    fit_result: 每个样本的全部拟合参数和 chi2/dof.
    cov: 协方差矩阵.
    cond: 协方差矩阵的条件数.
    mean_fit_info: 均值样本的原始拟合结果.
    fit_mean: 均值样本的全部拟合参数和 chi2/dof.
    n_bound: 至少有一个参数卡在边界的样本数.
    """

    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if rol is not None and set(rol) != set(p0):
        raise ValueError("rol must have the same keys as p0")

    Nsample, _ = y.shape
    param_names = list(p0.keys())
    n_params = len(param_names)
    print(f"fit method: {method}")

    cov, cond = calc_cov(y, jackknife)
    chol = np.linalg.cholesky(cov)
    p0 = {name: float(val) for name, val in p0.items()}

    # 先拟合全体样本均值, 该结果专门用于生成报表, 并作为逐样本拟合的初值.
    # 均值拟合不加约束, rol 传 None.
    y_mean = y.mean(axis=0)
    p0_mean, mean_fit_info, _ = fit_one(
        y_mean, x, model, p0, cov, chol, method, None)

    fit_mean = {name: p0_mean[name] for name in param_names}
    chi2_dof_mean, _, _ = calc_chi2_dof(
        y_mean,
        model(x, p0_mean),
        cov,
        n_params,
    )
    fit_mean["chi2dof"] = chi2_dof_mean

    # 以均值拟合结果作为初值, 对每个样本逐个拟合.
    p0 = {name: p0_mean[name] for name in param_names}
    fit_result = {name: np.full(Nsample, np.nan) for name in param_names}
    fit_result["chi2dof"] = np.full(Nsample, np.nan)
    n_bound_param = {name: 0 for name in param_names}
    n_bound = 0

    for _id in range(Nsample):
        pmean, _, at_bound = fit_one(
            y[_id], x, model, p0, cov, chol, method, rol)
        for name in param_names:
            fit_result[name][_id] = pmean[name]
            n_bound_param[name] += int(at_bound[name])
        if any(at_bound.values()):
            n_bound += 1
        chi2_dof_val, _, _ = calc_chi2_dof(
            y[_id],
            model(x, pmean),
            cov,
            n_params,
        )
        fit_result["chi2dof"][_id] = chi2_dof_val

    # 拟合完成后输出各参数的初值, 拟合结果和边界统计.
    if print_detail:
        print()
        print("prior(rol):")
        for name in param_names:
            if rol is None:
                print(f"{name}: {p0_mean[name]:.3g}")
            else:
                print(f"{name}: {format_mean_err(p0_mean[name], rol[name])}")
        print()
        print("mean(sem)")
        for name in param_names:
            print(f"{name}: "
                  f"{format_mean_err(fit_result[name].mean(), sem(fit_result[name]))}")
        print()
        print("n bound:")
        for name in param_names:
            print(f"{name}: {n_bound_param[name]}")
        print()

    return fit_result, cov, cond, mean_fit_info, fit_mean, n_bound


def calc_stability(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """计算两套数据的参数稳定性比值.

    参数:
    -----
    y_a: 第一套数据, 如候选窗口的逐样本参数数组.
    y_b: 第二套数据, 如相邻窗口的逐样本参数数组.

    输出:
    -----
    ratio: `abs(mean(y_a - y_b)) / std(y_a - y_b)`; 差值非有限时返回 nan.
    """

    diff = np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float)
    diff_mean = float(diff.mean())
    diff_std = float(diff.std(0))
    if not np.isfinite(diff_mean) or not np.isfinite(diff_std):
        return float("nan")
    if diff_std > 0:
        return float(abs(diff_mean) / diff_std)
    return 0.0 if abs(diff_mean) == 0 else float("inf")
