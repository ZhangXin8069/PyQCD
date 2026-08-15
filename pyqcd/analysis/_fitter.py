"""
统计拟合工具（独立实现，功能对齐 refer/huangcl/98_tools/analysis_tools.py 拟合部分）
====================================================================================

- `FitParams`：拟合参数 dataclass（p0 / prior / 拟合窗口 / svdcut / nex）。
- `calc_chi2` / `calc_chi2_dof`：chi2 计算（支持 lsqfit 风格 svdcut 特征值截断）。
- `fit`：逐样本 lsqfit 非线性拟合封装（prior 优先，debug 模式 NaN 填充）。
- `make_summary_table`：ASCII 对齐表格（独立复现 PrettyTable 对齐风格，
  pyqcd 不引入 prettytable 依赖）。
- `fit_report_lines`：通用拟合报告行构造（头部/逐项/汇总表）。

统计基元 sem/resample/cov_mat 复用 pyqcd.analysis._disconnected（同包）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ._disconnected import sem, cov_mat


@dataclass
class FitParams:
    """拟合参数。"""

    p0: dict
    prior: dict = None
    dt_start: int = 0
    dt_end: int = 10
    svdcut: Optional[float] = None
    nex: int = 0   # FH 等 τ 方向两端各去掉的点数


def calc_chi2(y_data: np.ndarray, y_fit: np.ndarray, cov: np.ndarray,
              svdcut: Optional[float] = None) -> float:
    """chi2 = diff^T C^{-1} diff，支持 lsqfit 风格 svdcut（特征值截断）。"""
    diff = y_data - y_fit

    if svdcut is None:
        return float(diff @ np.linalg.solve(cov, diff))

    eigval, eigvec = np.linalg.eigh(cov)
    cut = eigval.max() * svdcut
    mask = eigval > cut
    eig_inv = 1.0 / eigval[mask]
    cov_inv = eigvec[:, mask] @ np.diag(eig_inv) @ eigvec[:, mask].T
    return float(diff @ cov_inv @ diff)


def calc_chi2_dof(y_data: np.ndarray, y_fit: np.ndarray, cov: np.ndarray,
                  n_params: int, svdcut: Optional[float] = None):
    """计算 chi2/dof = chi2 / (Ndata − n_params)。"""
    chi2 = calc_chi2(y_data, y_fit, cov, svdcut)
    dof = len(y_data) - n_params
    return chi2 / dof, chi2, dof


def fit(
    y_coor: np.ndarray,
    x_coor,
    model: Callable,
    fitpa: FitParams,
    jackknife: bool = False,
    debug: bool = False,
    debugNfit: Optional[int] = 20,
):
    """对每个样本做 lsqfit 非线性拟合。

    Parameters
    ----------
    y_coor : (Nsample, Ndata) 数据数组。
    x_coor : 拟合点坐标（传给 lsqfit 的 data）。
    model : model(x, p) -> np.ndarray。
    fitpa : FitParams；prior 非空时优先使用 prior，否则退化为 p0。
    jackknife : 协方差是否用 jackknife 公式。
    debug : 只拟合前 debugNfit 个样本（协方差仍用全部样本），
            未拟合条目填 NaN，返回数组恒为 Nsample 大小。

    Returns
    -------
    (fit_result, cov, cond, last_fit_info)
        fit_result: {参数名: (Nsample,), "chi2": (Nsample,)}；
        cov: 协方差矩阵；cond: 条件数；last_fit_info: 最后一个样本的 lsqfit 对象。
    """
    import gvar as gv
    import lsqfit

    Nsample, _ = y_coor.shape
    param_names = list(fitpa.p0.keys())
    n_params = len(param_names)

    Nfit = min(debugNfit, Nsample) if debug else Nsample

    fit_result = {name: np.full(Nsample, np.nan) for name in param_names}
    fit_result["chi2"] = np.full(Nsample, np.nan)

    cov, cond = cov_mat(y_coor, jackknife=jackknife)
    use_prior = fitpa.prior is not None and len(fitpa.prior) > 0

    last_fit_info = None
    for _id in range(Nfit):
        y_gvar = gv.gvar(y_coor[_id], cov)
        if use_prior:
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_gvar), prior=fitpa.prior,
                fcn=model, svdcut=fitpa.svdcut)
        else:
            _fit = lsqfit.nonlinear_fit(
                data=(x_coor, y_gvar), p0=fitpa.p0,
                fcn=model, svdcut=fitpa.svdcut)
        last_fit_info = _fit

        for name in param_names:
            fit_result[name][_id] = _fit.pmean[name]
        chi2_dof_val, _, _ = calc_chi2_dof(
            y_coor[_id], model(x_coor, _fit.pmean), cov, n_params, fitpa.svdcut)
        fit_result["chi2"][_id] = chi2_dof_val

    return fit_result, cov, cond, last_fit_info


def make_summary_table(field_names: List[str], rows: List[List[str]],
                       align: Optional[Dict[str, str]] = None) -> str:
    """ASCII 对齐表格（PrettyTable 风格，无外部依赖）。

    Parameters
    ----------
    field_names : 列名列表。
    rows : 每行各列字符串。
    align : {列名: 'l'|'c'|'r'}，缺省居中。
    """
    n_cols = len(field_names)
    colw = [len(str(f)) for f in field_names]
    for r in rows:
        for i in range(n_cols):
            colw[i] = max(colw[i], len(str(r[i])))

    def _fmt(cell, w, a):
        cell = str(cell)
        if a == "l":
            return cell.ljust(w)
        if a == "r":
            return cell.rjust(w)
        return cell.center(w)

    align = align or {}
    border = "+" + "+".join("-" * (w + 2) for w in colw) + "+"
    head = "|" + "|".join(" " + _fmt(f, w, "c") + " "
                          for f, w in zip(field_names, colw)) + "|"
    lines = [border, head, border]
    for r in rows:
        line = "|" + "|".join(" " + _fmt(v, w, align.get(field_names[i], "c")) + " "
                              for i, (v, w) in enumerate(zip(r, colw))) + "|"
        lines.append(line)
    lines.append(border)
    return "\n".join(lines)


def fit_report_lines(title: str, header: Dict[str, object], sep: str = "=" * 72) -> List[str]:
    """拟合报告头部（分隔线 + 标题 + 关键设置）。"""
    lines = [sep, f"  {title}", sep]
    for k, v in header.items():
        lines.append(f"  {k:14s}: {v}")
    lines.append(sep)
    lines.append("")
    return lines
