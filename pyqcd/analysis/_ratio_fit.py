"""
裸矩阵元 c0(z) 提取（移植 zengch fit_ratio.py 核心逻辑）
=========================================================

比值模型（R(dt, dtau, z)）：
    z = z₀:  R = c0_z0 + c1_z0·(e^{−dE(tsep−ti)} + e^{−dE·ti}) + c2_z0·e^{−dE·tsep}
    z = z₁:  R = c0_z1 + c1_z1·(e^{−dE(tsep−ti)} + e^{−dE·ti}) + c2_z1·e^{−dE·tsep}

c0(z) 即坐标空间裸矩阵元（后续重整化/匹配的输入）。逐重采样样本拟合。
"""
from __future__ import annotations

import numpy as np


def covariance_matrix_inv(samples, resam_type='boot'):
    """重采样样本协方差的逆（samples 布局 (n_points, n_samples)）。

    jackknife：cov = Σ(x̄ − x_i)(x̄ − x_i)ᵀ·(n−1)/n（均值协方差，delete-one 约定）；
    bootstrap：cov = Σ(x̄ − x_i)(x̄ − x_i)ᵀ/n。
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    n_samples = s.shape[1]
    mean = s.mean(axis=1)
    diff = (s - mean[:, None]).T        # (n_samples, n_points)
    if resam_type == 'jack':
        cov = np.matmul(diff.T, diff) / n_samples * (n_samples - 1)
    else:   # bootstrap
        cov = np.matmul(diff.T, diff) / n_samples
    return np.linalg.inv(cov)


def R_model(z_, tsep_, ti_, z_list_, c0_z0, c1_z0, c2_z0,
            c0_z1, c1_z1, c2_z1, deltaE):
    """比值模型（zengch 公式）。"""
    res = np.zeros_like(np.asarray(ti_, dtype=float))
    mask_z0 = (np.asarray(z_) == z_list_[0])
    res[mask_z0] = (c0_z0
                    + c1_z0 * np.exp(-deltaE * (tsep_[mask_z0] - ti_[mask_z0]))
                    + c1_z0 * np.exp(-deltaE * ti_[mask_z0])
                    + c2_z0 * np.exp(-deltaE * tsep_[mask_z0]))
    mask_z1 = (np.asarray(z_) == z_list_[1])
    res[mask_z1] = (c0_z1
                    + c1_z1 * np.exp(-deltaE * (tsep_[mask_z1] - ti_[mask_z1]))
                    + c1_z1 * np.exp(-deltaE * ti_[mask_z1])
                    + c2_z1 * np.exp(-deltaE * tsep_[mask_z1]))
    if not np.all(mask_z0 | mask_z1):
        raise ValueError(f'z 只允许 {z_list_[0]} 或 {z_list_[1]}')
    return res


def _fit_one(y, z_set, tsep_set, ti_set, z_list, c_inv, par_ini=None):
    """单组数据（均值或单样本）最小二乘拟合。"""
    if par_ini is None:
        par_ini = [0.5, -0.2, 0.0, 0.5, -0.2, 0.0, 0.3]
    data_num = len(z_set)

    def cost(par):
        (c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE) = par
        th = R_model(z_set, tsep_set, ti_set, z_list,
                     c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE)
        del_r = y - th
        return float(del_r.T @ c_inv @ del_r / data_num)

    from scipy.optimize import minimize
    res = minimize(cost, par_ini, method='Nelder-Mead',
                   options={'maxiter': 20000, 'xatol': 1e-8})
    return res.x


def fit_ratio(data_for_fit, resam_type='boot'):
    """逐样本拟合比值，提取 c0(z₀)/c0(z₁) 与 deltaE。

    Args:
        data_for_fit: (z_set, tsep_set, ti_set, ratio_mean, err,
                       ratio_samples, z_list, t_sep_list, n_remove)
    Returns:
        dict: {c0_z0, c1_z0, c2_z0, c0_z1, c1_z1, c2_z1, deltaE,
               c0_z0_err, c0_z1_err, deltaE_err}（逐样本均值与误差）
    """
    (z_set_, tsep_set_, ti_sep_set_, ratio_mean_set_, err_set,
     ratio_samples_fit_, z_list_, _t_sep_list_, _n_remove_) = data_for_fit
    c_inv = covariance_matrix_inv(ratio_samples_fit_, resam_type)

    n_samples = ratio_samples_fit_.shape[1]
    fits = np.zeros((n_samples, 7))
    for sample_i in range(n_samples):
        y = ratio_samples_fit_[:, sample_i]
        fits[sample_i] = _fit_one(y, z_set_, tsep_set_, ti_sep_set_,
                                  z_list_, c_inv)

    mean = fits.mean(axis=0)
    err = fits.std(axis=0)
    keys = ['c0_z0', 'c1_z0', 'c2_z0', 'c0_z1', 'c1_z1', 'c2_z1', 'deltaE']
    return {k: mean[i] for i, k in enumerate(keys)} | \
        {f'{k}_err': err[i] for i, k in enumerate(keys)}
