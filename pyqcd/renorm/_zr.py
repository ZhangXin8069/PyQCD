"""
自重整化因子 Z_R：全局参数化与拟合（移植 zengch fit_zr_new.py 核心逻辑）。

对应 arXiv:2510.17758 Eq.(3)-(8) 的混合方案自重整化：

    hB(z, Pz=0) 在短距离 (z ≤ z₁) 由 NLO MS-bar 因子 Z_MS 控制；
    长距离 (z > z₁) 由非微扰节点参数 g₁..g₁₄ 描述（th_hB 的分段 B(z)）。

    Z_R(z, a, μ) = exp[ k·z/(a·ln(aΛ)) + ½ln(1 + d/ln(aΛ))²
                       + 5C_A/(3b₀)·ln(ln(1/(aΛ))/ln(μ/Λ))
                       + (m₀ + m₂a²)·z + f·a + f₂·a² ]

其中线性发散项 k·z/a 为梯度流（或 Wilson 线）重整化所需的发散抵消，
对数重求和项对应 MS-bar 窗口。本模块把参数化函数与拟合代价函数做成
纯函数库（不依赖集群数据路径），数据由调用方以 dict 形式传入。
"""
from __future__ import annotations

import numpy as np

from ._const import CA, gammaE, b0
from ._ensembles import fm_to_GeV


def Z_MS(z_gev, mu):
    """NLO MS-bar 重整化因子（z 单位 GeV⁻¹，μ 单位 GeV）。

    Z_MS = 1 + α_s·C_A/(4π)·[ (5/3)·ln( z²μ² / (4e^{−2γ_E}) ) + 3 ]
    """
    alpha_s = A_s_run(mu)
    return 1.0 + alpha_s * CA / (4.0 * np.pi) * (
        5.0 / 3.0 * np.log((z_gev ** 2 * mu ** 2) / (4.0 * np.exp(-2.0 * gammaE))) + 3
    )


def A_s_run(mu, Lambda_QCD=0.23, nf=3.0):
    """α_s/(4π) 1 圈运行耦合（zengch constant.py 的 A_s 语义）。"""
    from ._const import A_s
    return A_s(mu, Lambda_QCD, nf)


def th_hB(z_, a_, mu_, par_g_set, f_set, z1_fm=0.301):
    """对数 hB(z) 的理论参数化（Eq.7）：分段——z ≤ z₁ 用 Z_MS + 质量项，
    z > z₁ 用非微扰节点 g₁..g₁₄。

    Args:
        z_:   z 数组（fm）
        a_:   格距（GeV⁻¹）
        mu_:  重整化标度（GeV）
        par_g_set: (k, d, m0, m2, Lambda_QCD, g1..g14)
        f_set: (f1, f2) 离散化修正系数
    Returns:
        log hB，形状与 z_ 相同。
    """
    z_set_new_gev = np.asarray(z_, dtype=float) / fm_to_GeV
    k, d, m0, m2, lambda_qcd, *g_params = par_g_set
    g_params = np.asarray(g_params, dtype=float)

    nf = 3.0
    b0_ = b0(nf)

    def B(z_i):
        z1 = z1_fm / fm_to_GeV  # GeV⁻¹
        result = np.zeros_like(np.asarray(z_i, dtype=float))
        mask = z_i <= z1
        z_masked = z_i[mask]
        result[mask] = np.log(Z_MS(z_masked, mu_)) + (m0 + m2 * a_ ** 2) * z_masked
        result[~mask] = g_params[: len(result[~mask])]
        return result

    a_set = np.array([a_, a_ ** 2.0])
    f_set_arr = np.asarray(f_set, dtype=float)

    log_hb = (k * z_set_new_gev) / (a_ * np.log(a_ * lambda_qcd))
    log_hb += 5.0 * CA / (3.0 * b0_) * np.log(
        np.log(1.0 / (a_ * lambda_qcd)) / np.log(mu_ / lambda_qcd)
    )
    log_hb += np.log((1.0 + d / np.log(a_ * lambda_qcd)) ** 2.0) / 2.0
    log_hb += f_set_arr @ a_set
    log_hb += B(z_set_new_gev)
    return log_hb


def th_ZR(z_, a_, mu_, k, d, m0, m2, lambda_qcd, f_set):
    """Z_R 的指数参数化（Eq.5）：exp(线性发散 + 质量平移 + 对数重求和 + 离散化)。

    Args:
        z_:   z 数组（fm）
        a_:   格距（GeV⁻¹）
        mu_:  重整化标度（GeV）
        k, d, m0, m2, lambda_qcd: 拟合参数
        f_set: (f1, f2)
    Returns:
        Z_R(z)，形状与 z_ 相同。
    """
    z_set_new_gev = np.asarray(z_, dtype=float) / fm_to_GeV
    nf = 3.0
    b0_ = b0(nf)
    a_set = np.array([a_, a_ ** 2.0])[:, None]
    f_set_arr = np.asarray(f_set, dtype=float)

    log_hb = (k * z_set_new_gev) / (a_ * np.log(a_ * lambda_qcd))
    log_hb += 5.0 * CA / (3.0 * b0_) * np.log(
        np.log(1.0 / (a_ * lambda_qcd)) / np.log(mu_ / lambda_qcd)
    )
    log_hb += np.log((1.0 + d / np.log(a_ * lambda_qcd)) ** 2.0) / 2.0
    log_hb += (m0 + m2 * a_ ** 2) * z_set_new_gev
    log_hb += np.sum(f_set_arr * a_set, axis=0)
    return np.exp(log_hb)


def cost_function(z_set_, hb_data, c_inv, a_, mu_, par_set, z1_fm=0.301):
    """单系综 χ²（平均到每数据点）。

    Args:
        z_set_: z 数组（fm）
        hb_data: log hB 数据数组
        c_inv: 协方差逆矩阵
        a_: 格距（GeV⁻¹）
        mu_: 重整化标度（GeV）
        par_set: 前 19 个为 (k,d,m0,m2,Λ,g1..g14)，其余为 (f1,f2)
    """
    data_num = len(z_set_)
    par_g_set = par_set[:19]
    f_set = par_set[19:]
    hb_th = th_hB(z_set_, a_, mu_, par_g_set, f_set, z1_fm)
    del_hb = hb_th - np.asarray(hb_data, dtype=float)
    chi2 = del_hb.T @ c_inv @ del_hb
    return chi2 / data_num


def cost_function_all(par_set, datasets, mu_):
    """多系综联合 χ²/dof。

    Args:
        par_set: 全部拟合参数（前 19 + f1,f2）
        datasets: [dict(z=..., loghB=..., c_inv=..., a=..., n=...), ...]
        mu_: 重整化标度（GeV）
    """
    chi2_sum = 0.0
    n_sum = 0
    for ds in datasets:
        n = len(ds['z'])
        chi2 = cost_function(ds['z'], ds['loghB'], ds['c_inv'],
                             ds['a'], mu_, par_set)
        chi2_sum += chi2 * n
        n_sum += n
    dof = n_sum - len(par_set)
    return chi2_sum / dof


def fit_ZR(par_ini, datasets, mu_, use_iminuit=True):
    """全局拟合 Z_R 参数（iminuit 或 scipy 回退）。

    Args:
        par_ini: 参数初值（长度 ≥ 21：k,d,m0,m2,Λ,g1..g14,f1,f2）
        datasets: [dict(z, loghB, c_inv, a), ...]
        mu_: 重整化标度（GeV）
    Returns:
        拟合参数数组（m.values）。
    """
    par_name = ('k', 'd', 'm0', 'm2', 'Lambda_QCD',
                'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
                'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'f1', 'f2')

    def cost(par):
        return cost_function_all(par, datasets, mu_)

    if use_iminuit:
        try:
            from iminuit import Minuit
            m = Minuit(cost, par_ini, name=par_name)
            m.limits['k'] = (None, None)
            m.limits['d'] = (None, None)
            m.limits['m0'] = (None, None)
            m.limits['m2'] = (None, None)
            m.limits['Lambda_QCD'] = (0, None)
            m.fixed['f1', 'f2', 'm2'] = True
            m.migrad()
            return np.array(m.values)
        except ImportError:
            pass  # 回退到 scipy

    from scipy.optimize import minimize
    res = minimize(cost, par_ini, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6})
    return res.x
