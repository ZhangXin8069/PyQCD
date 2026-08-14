"""
NLO 匹配：准 PDF → 光锥 PDF（移植 zengch matching.py / matching_cc.py 核心逻辑）。

胶子单圈匹配核（LO 非定域算符 → NLO 光锥 PDF）：
    g_1 (ξ<0)、g_2 (0<ξ<1)、g_3 (ξ>1) 与 g_0（Si 函数 + λ_s 截断项）。
    Z_ij = δ_ij + (α_s C_A/2π)·M_ij，hR_PDF = Z⁻¹·hR_0。

matching_cc.py 的 ``C_gluon_ratio`` 为协变组合（ratio 方案）匹配核，
用于梯度流重整化矩阵元的匹配（TMD 情形需 b_⊥ 依赖，见 _tmd.py）。
"""
from __future__ import annotations

import numpy as np
from scipy.special import sici

from ._const import CA, pi
from ._ensembles import fm_to_GeV, a_len_set, Nl_set, pz_to_gev


def Si(x_):
    """正弦积分函数 Si(x)。"""
    return sici(x_)[0]


def _matching_kernels(cxi, y_matr, Pz_GeV, mu_, lambda_s):
    """按 ξ=x/y 分区计算匹配核 g_xy（g_0 + 对应分区的 g_1/g_2/g_3）。"""
    def g_1(xi_, y_):
        return (-2.0 * (1.0 - xi_ + xi_ ** 2.0) ** 2.0 / (1.0 - xi_)
                * np.log(xi_ / (xi_ - 1.0))
                - (11.0 - 28.0 * xi_ + 18.0 * xi_ ** 2.0 - 12.0 * xi_ ** 3.0)
                / (6.0 * (1.0 - xi_)))

    def g_2(xi_, y_):
        res = (2.0 * (1.0 - xi_ + xi_ ** 2.0) ** 2.0 / (1.0 - xi_)
               * (-np.log(mu_ ** 2.0 / (4.0 * y_ ** 2.0 * Pz_GeV ** 2.0))
                  + np.log(xi_ * (1.0 - xi_))))
        res -= ((15.0 - 56.0 * xi_ + 102.0 * xi_ ** 2.0
                 - 96.0 * xi_ ** 3.0 + 48.0 * xi_ ** 4.0)
                / (6.0 * (1.0 - xi_)))
        return res

    def g_3(xi_, y_):
        return (2.0 * (1.0 - xi_ + xi_ ** 2.0) ** 2.0 / (1.0 - xi_)
                * np.log(xi_ / (xi_ - 1.0))
                + (11.0 - 28.0 * xi_ + 18.0 * xi_ ** 2.0 - 12.0 * xi_ ** 3.0)
                / (6.0 * (1.0 - xi_)))

    def g_0(xi_, y_):
        return (5.0 / 6.0 * (-1.0 / np.abs(1.0 - xi_)
                             + 2.0 * Si(((1.0 - xi_) * np.abs(y_) * lambda_s))
                             / (pi * (1.0 - xi_))))

    mask_1 = (cxi < 0)
    mask_2 = (0 < cxi) & (cxi < 1)
    mask_3 = (cxi > 1)

    g_xy = np.zeros_like(cxi)
    g_xy[mask_1] = -g_0(cxi[mask_1], y_matr[mask_1]) - g_1(cxi[mask_1], y_matr[mask_1])
    g_xy[mask_2] = g_0(cxi[mask_2], y_matr[mask_2]) + g_2(cxi[mask_2], y_matr[mask_2])
    g_xy[mask_3] = g_0(cxi[mask_3], y_matr[mask_3]) + g_3(cxi[mask_3], y_matr[mask_3])
    return g_xy


def hR_PDF(xx, Pz_, conf, hR_tilde_data, mu_=2.0, lambda_s_fm=0.3):
    """NLO 匹配：hR_PDF = Z⁻¹·hR_0（胶子，非极化）。

    Args:
        xx: x 网格（含负值，向量）
        Pz_: 格点单位动量
        conf: 系综名
        hR_tilde_data: hR(x) 数组（与 xx 同形）
        mu_: 匹配标度（GeV，默认 2.0）
        lambda_s_fm: 大 λ 截断（fm，默认 0.3）
    Returns:
        hR_PDF（光锥 PDF 匹配结果）。
    """
    xx = np.asarray(xx, dtype=float)
    dx = xx[1] - xx[0]

    alpha_s = A_s_run(mu_)
    a_len = a_len_set[conf]
    Nl = Nl_set[conf]
    Pz_GeV = pz_to_gev(Pz_, conf)
    lambda_s = lambda_s_fm / fm_to_GeV * Pz_GeV

    yy = xx
    n_len = len(xx)
    x_col = xx[:, None]
    y_col = yy[None, :]
    x_matr = np.tile(x_col, (1, n_len))
    y_matr = np.tile(y_col, (n_len, 1))
    cxi = x_matr / y_matr

    g_xy = _matching_kernels(cxi, y_matr, Pz_GeV, mu_, lambda_s)
    g_ij = g_xy

    delta_ij = np.eye(n_len)
    dx_matr = np.diag(xx / np.abs(xx)) * dx

    c_alp_lo = dx_matr * alpha_s * CA / 2.0 / pi
    m_alp_lo = g_ij / y_matr - delta_ij * y_matr * np.sum(
        g_ij / (y_matr ** 2.0), axis=1)
    z_ij = delta_ij + c_alp_lo @ m_alp_lo

    return np.linalg.inv(z_ij) @ hR_tilde_data


def A_s_run(mu, Lambda_QCD=0.23, nf=3.0):
    from ._const import A_s
    return A_s(mu, Lambda_QCD, nf)


# ═══════════════════════════════════════════════════════════════════
# ratio 方案胶子匹配核（matching_cc.py）
# ═══════════════════════════════════════════════════════════════════

def C(ksi, m, r):
    """比值方案匹配核（zengch matching_cc.py 原版 C(ξ,m,r)）。"""
    return 2.0 * ksi / (1.0 - ksi) * (m / (2.0 * r)) ** 2.0 * (
        (2.0 * ksi - 1.0) * (1.0 - ksi) * np.log((ksi - 1.0) ** 2.0 / ksi ** 2.0)
        + 2.0 * (ksi - 1.0) - 2.0 * np.log(ksi / (1.0 - ksi)) - 2.0)


def C_gluon_ratio(ksi, m, r):
    """胶子 ratio 方案匹配核（协变组合）。

    m: (|x|·Z_s·a·pz)/GeVfm，r: μ/(|x|·pz)。
    """
    ksi = np.asarray(ksi, dtype=float)
    res = np.zeros_like(ksi)
    mask = (ksi > 1.0)
    res[mask] = C(ksi[mask], m, r)
    return res
