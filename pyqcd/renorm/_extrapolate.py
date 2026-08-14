"""
连续极限 / 物理点联合外推（移植 zengch fit_pz_a_extrapolatiing.py 核心逻辑）。

逐 x 联合外推 ansatz：
    f(x; a, Pz, mπ, L) = xg₀(x) + a²·f(x) + a⁴·l(x) + a²Pz²·h(x)
                         + d(x)/Pz² + b(x)/Pz + k(x)·(mπ² − mπ,phy²)
                         + c(x)·e^{−L·a·mπ}
"""
from __future__ import annotations

import numpy as np

from ._ensembles import a_len_set, Nl_set, pion_mass_set, MPI_PHYSICAL


def hR_form(var, par):
    """外推形式（在固定 x 处对 a, Pz, mπ, L 做线性拟合）。

    Args:
        var: (a_, pz_, mpi, L_)——格距(GeV⁻¹)、Pz(格点单位)、mπ(GeV)、L(格点)
        par: (xg0, fx, lx, hx, dx, bx, kx, cx) 8 个拟合系数
    """
    a_, pz_, mpi, L_ = var
    xg0_, fx_, lx_, hx_, dx_, bx_, kx_, cx_ = par
    return (xg0_
            + a_ ** 2 * fx_ + a_ ** 4 * lx_
            + a_ ** 2 * pz_ ** 2 * hx_
            + dx_ / pz_ ** 2 + bx_ / pz_
            + kx_ * (mpi ** 2 - MPI_PHYSICAL ** 2)
            + cx_ * np.exp(-L_ * a_ * mpi))


def build_fit_data(conf_set, note_name_set, pz_set, loader):
    """收集多系综数据（x, hR_PDF, a, Pz_GeV, mπ, L）。

    Args:
        conf_set: 系综名列表
        note_name_set: 数据备注名列表（与 conf_set 对齐）
        pz_set: 每系综的 Pz 列表
        loader: 回调 conf, note_name, pz → (xx, hR_PDF, a_GeV⁻¹, Pz_GeV, mπ_GeV, Nl)
    Returns:
        list of dict(x=..., hR=..., a=..., pz=..., mpi=..., L=...)
    """
    rows = []
    for conf, note, pz_list in zip(conf_set, note_name_set, pz_set):
        for pz in pz_list:
            xx, hR, a_len, pz_gev, mpi, nl = loader(conf, note, pz)
            if xx is None:
                continue
            rows.append(dict(x=xx, hR=hR, a=a_len, pz=pz_gev,
                             mpi=mpi, L=nl))
    return rows


def fit_hR_PDF_extrap(rows, x_grid=None, fitter='lm', max_x=1.0):
    """逐 x 拟合外推参数（8 参数线性拟合，scipy.linalg.lstsq）。

    Args:
        rows: build_fit_data 的输出
        x_grid: 目标 x 网格（默认取所有数据公共 x 网格）
    Returns:
        (x_grid, xg0(x), 外推误差带 std(x))。
    """
    if x_grid is None:
        x_grid = rows[0]['x']
    x_grid = np.asarray(x_grid, dtype=float)
    mask_x = (x_grid >= 0) & (x_grid <= max_x)
    x_grid = x_grid[mask_x]

    xg0 = np.zeros_like(x_grid)
    xg0_std = np.zeros_like(x_grid)

    for i, x_val in enumerate(x_grid):
        A_rows, b_vec = [], []
        for r in rows:
            idx = np.where(np.abs(r['x'] - x_val) < 1e-4)[0]
            if len(idx) == 0:
                continue
            j = idx[0]
            hR_val = np.atleast_1d(r['hR'])[j]
            var = (r['a'], r['pz'], r['mpi'], r['L'])
            # 8 参数线性模型：构造设计矩阵行 [1, a², a⁴, a²pz², 1/pz², 1/pz, mπ²-mπ,phy², e^{-L·a·mπ}]
            a_, pz_, mpi, L_ = var
            row = np.array([
                1.0,
                a_ ** 2, a_ ** 4, a_ ** 2 * pz_ ** 2,
                pz_ ** -2, pz_ ** -1,
                mpi ** 2 - MPI_PHYSICAL ** 2,
                np.exp(-L_ * a_ * mpi),
            ])
            A_rows.append(row)
            b_vec.append(hR_val)

        if len(A_rows) < 8:
            xg0[i] = np.nan
            xg0_std[i] = np.nan
            continue

        A = np.array(A_rows)
        b = np.array(b_vec)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ coef
        dof = max(len(b) - 8, 1)
        chi2 = np.sum(resid ** 2) / dof
        xg0[i] = coef[0]
        xg0_std[i] = np.sqrt(chi2) * np.sqrt(np.linalg.inv(A.T @ A)[0, 0])

    return x_grid, xg0, xg0_std
