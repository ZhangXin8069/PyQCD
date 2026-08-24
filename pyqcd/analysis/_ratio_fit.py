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
    if resam_type == 'jack':
        # jackknife 误差重标定 std×√(n−1)（原版 fit_ratio.py 同款）
        err = err * np.sqrt(n_samples - 1)
    keys = ['c0_z0', 'c1_z0', 'c2_z0', 'c0_z1', 'c1_z1', 'c2_z1', 'deltaE']
    return {k: mean[i] for i, k in enumerate(keys)} | \
        {f'{k}_err': err[i] for i, k in enumerate(keys)}


# ═══════════════════════════════════════════════════════════════════
# FH 常数拟合与自适应 t_sep 窗口（整合 zengch fit_ratio_FeynmenHellman_new）
# ═══════════════════════════════════════════════════════════════════

def fit_constant_window(c0_zt, kind='boot', seed=0):
    """常数 c0 的协方差加权窗口拟合（照抄 fit_ratio_mean_FeynmenHellman 核）。

    常数模型的全协方差 χ² 极小化有闭式解：
        c0 = (1ᵀC⁻¹d)/(1ᵀC⁻¹1)，χ² = (d−c0)ᵀC⁻¹(d−c0)/(n−1)
    与原版 Minuit 数值结果一致（无优化器依赖）。

    Args:
        c0_zt: (n_tsep, n_sample) 窗口内比值数据（逐重采样样本）。
        kind: 'boot'|'jack'（covariance_matrix_inv 语义）。
    Returns:
        dict(c0, c0_std(逐样本散布), chi2, chi2_nocov, n_data)。
    """
    d = np.atleast_2d(np.asarray(c0_zt, dtype=float))
    if d.shape[0] > d.shape[1]:          # 允许 (n_sample, n_tsep) 转置输入
        if np.ndim(c0_zt) == 2 and d.shape[1] >= 2:
            d = d.T
    n = d.shape[0]
    c_inv = covariance_matrix_inv(d, resam_type='boot' if kind == 'boot'
                                  else 'jack')
    ones = np.ones(n)
    denom = ones @ c_inv @ ones
    c0 = (ones @ c_inv @ d.mean(axis=1)) / denom
    r = d.mean(axis=1) - c0
    chi2 = float(r @ c_inv @ r) / max(n - 1, 1)
    c_inv_nocov = np.diag(1.0 / np.maximum(d.std(axis=1) ** 2, 1e-30))
    chi2_nocov = float(r @ c_inv_nocov @ r) / max(n - 1, 1)
    # 逐样本重拟合（同闭式，逐列）
    per = c_inv @ d                       # (n, nsam)
    num = ones @ per                      # (nsam,)
    c0_samples = num / denom
    return {'c0': float(c0), 'c0_std': float(np.std(c0_samples)),
            'chi2': chi2, 'chi2_nocov': chi2_nocov,
            'c0_samples': c0_samples, 'n_data': int(n)}


def fh_adaptive_windows(delta, t_sep_vals, t_do0, t_up0, chi2_limit=1.5,
                        t_floor=None, z_max=None, kind='boot', seed=0,
                        verbose=False, logger=print):
    """χ² 上限驱动的逐 z 自适应 t_sep 窗口滑动（c0_vs_z_FeynmenHellman 语义）。

    每 z 从上一 z 收敛窗起步：χ² 超限则右移（+1，下界保护 t_floor，
    对应原版 tis≥nr·2 的约束）；未超限则尝试左移一步收紧（仍超限即回退）。
    小 z（< z_direct=6）直接沿用初始窗（原版约定：z≤5 不滑窗）。

    Args:
        delta: (nz, n_tsep, n_sample) 比值数据。
        t_sep_vals: (n_tsep,) 真实 t_sep 值（与第二维对应）。
        t_do0/t_up0: 初始窗口 [下, 上]（含端点，按 t_sep_vals 取值）。
        chi2_limit: 右移触发的 χ² 上限。
        t_floor: 窗口下界（None 时取 min(t_sep_vals)）。
    Returns:
        records: [{'z', 't_do', 't_up', 'fit'(fit_constant_window dict)}]
    """
    delta = np.asarray(delta, dtype=float)
    t_vals = np.asarray(t_sep_vals, dtype=int)
    t_floor = int(t_floor) if t_floor is not None else int(t_vals.min())
    z_total = delta.shape[0] if z_max is None else min(z_max,
                                                       delta.shape[0])
    records = []
    last_do, last_up = int(t_do0), int(t_up0)

    def _fit(z, do, up):
        mask = (t_vals >= do) & (t_vals <= up)
        if mask.sum() < 2:
            raise ValueError(f"z={z}: 窗口 [{do},{up}] 内不足 2 个 t_sep")
        return fit_constant_window(delta[z][mask], kind=kind, seed=seed)

    for z in range(z_total):
        do, up = last_do, last_up
        fit = _fit(z, do, up)
        if z >= 6:                        # 小 z 直接用初始/继承窗
            # 右移直至达标或达继承起点保护
            while fit['chi2'] > chi2_limit and do + 1 < t_vals.max():
                do, up = do + 1, up + 1
                fit = _fit(z, do, up)
                if do == last_do and fit['chi2'] > chi2_limit:
                    break                 # 已回到继承窗仍超限 → 保持
                if do >= last_do and fit['chi2'] > chi2_limit:
                    break
            # 未超限则尝试左移一步收紧（超限回退）
            if fit['chi2'] <= chi2_limit and do - 1 >= t_floor \
                    and (t_vals >= do - 1).any() and (t_vals <= up - 1).sum() >= 2:
                try:
                    trial = _fit(z, do - 1, up - 1)
                except ValueError:
                    trial = None
                if trial is not None and trial['chi2'] <= chi2_limit:
                    do, up, fit = do - 1, up - 1, trial
        records.append({'z': z, 't_do': do, 't_up': up, 'fit': fit})
        if verbose:
            logger(f"z={z}: window=[{do},{up}] "
                   f"c0={fit['c0']:.4g}±{fit['c0_std']:.2g} "
                   f"chi2={fit['chi2']:.3g}")
        last_do, last_up = do, up
    return records
