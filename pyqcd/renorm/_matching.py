"""
NLO 匹配：准 PDF → 光锥 PDF（移植 zengch matching.py / matching_cc.py 核心逻辑）。

胶子单圈匹配核（LO 非定域算符 → NLO 光锥 PDF）：
    g_1 (ξ<0)、g_2 (0<ξ<1)、g_3 (ξ>1) 与 g_0（Si 函数 + λ_s 截断项）。
    hR_PDF = hR_0 − (α_s C_A/2π)·K·hR_0，其中 K 使用显式
    plus-subtraction；固定一圈阶数不通过矩阵逆引入高阶项。

matching_cc.py 的 ``C_gluon_ratio`` 为协变组合（ratio 方案）匹配核，
用于梯度流重整化矩阵元的匹配（TMD 情形需 b_⊥ 依赖，见 _tmd.py）。
"""
from __future__ import annotations

import numpy as np
from scipy.special import sici

from ._const import CA, pi
from ._ensembles import fm_to_GeV, pz_to_gev


def _validate_uniform_grid(grid, name):
    """Return a finite, strictly increasing, uniformly spaced real grid."""
    raw = np.asarray(grid)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional grid with >= 2 nodes")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values == 0.0):
        raise ValueError(
            f"{name} contains an exact zero node; use an avoid-zero grid")
    spacing = np.diff(values)
    if np.any(spacing <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(
            spacing, spacing[0], rtol=1.0e-10, atol=1.0e-14):
        raise ValueError(f"{name} must be uniformly spaced")
    return values


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
    g_xy[mask_1] = g_0(cxi[mask_1], y_matr[mask_1]) + g_1(
        cxi[mask_1], y_matr[mask_1])
    g_xy[mask_2] = g_0(cxi[mask_2], y_matr[mask_2]) + g_2(cxi[mask_2], y_matr[mask_2])
    g_xy[mask_3] = g_0(cxi[mask_3], y_matr[mask_3]) + g_3(cxi[mask_3], y_matr[mask_3])
    return g_xy


def _linear_interpolation_weights(grid, target):
    """Return bracketing indices/weights and a support mask for ``target``."""
    values = np.asarray(grid, dtype=float)
    targets = np.asarray(target, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("interpolation grid must be one-dimensional")
    left = np.searchsorted(values, targets, side="right") - 1
    left = np.clip(left, 0, len(values) - 2)
    weight_right = (
        (targets - values[left]) / (values[left + 1] - values[left]))
    valid = (targets >= values[0]) & (targets <= values[-1])
    return left, 1.0 - weight_right, weight_right, valid


def _matching_quadrature(limit, order, extra=()):
    """Reference plus-prescription quadrature with endpoint refinement."""
    if limit <= 0.0 or order < 2:
        raise ValueError("matching quadrature requires limit>0 and order>=2")
    bounds = [-limit, limit, -10.0, -5.0, -2.0, -1.0, 0.0,
              1.0, 2.0, 5.0, 10.0]
    bounds.extend(float(value) for value in extra)
    for delta in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 0.05, 0.1, 0.25, 0.5):
        bounds.extend((-delta, delta, 1.0 - delta, 1.0 + delta))
    bounds = np.unique([
        value for value in bounds if -limit <= value <= limit])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    left = bounds[:-1, None]
    right = bounds[1:, None]
    return (
        ((right - left) * nodes / 2.0 + (right + left) / 2.0).ravel(),
        ((right - left) * weights / 2.0).ravel(),
    )


def _matching_matrix(y_grid, x_grid, pz_gev, mu, lambda_s,
                     xi_max=50.0, quadrature_order=48):
    """Discrete ``K_xy`` with the reference plus subtraction at ``y=x``.

    ``K`` implements

        K f(x) = ∫ dξ [R(ξ,|x/ξ|Pz) f(x/ξ)/|ξ|
                       - R(ξ,|x|Pz) f(x)].

    It is applied as ``f = h - alpha_s C_A/(2π) K h`` at fixed one-loop
    order; no matrix inversion is taken.
    """
    x = _validate_uniform_grid(x_grid, "matching x grid")
    y = _validate_uniform_grid(y_grid, "matching y grid")
    pz_gev = float(pz_gev)
    mu = float(mu)
    lambda_s = float(lambda_s)
    if not np.isfinite(pz_gev) or pz_gev <= 0.0:
        raise ValueError("pz_gev must be finite and positive")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    if not np.isfinite(lambda_s) or lambda_s < 0.0:
        raise ValueError("lambda_s must be finite and nonnegative")

    matrix = np.zeros((len(x), len(y)), dtype=float)
    for row, x_value in enumerate(x):
        xi, weights = _matching_quadrature(
            xi_max, quadrature_order,
            extra=(x_value / y[0], x_value / y[-1]))
        target = x_value / xi
        index, left_weight, right_weight, valid = (
            _linear_interpolation_weights(y, target))
        coefficient = (
            weights[valid]
            * _matching_kernels(
                xi[valid], target[valid], pz_gev, mu, lambda_s)
            / np.abs(xi[valid])
        )
        np.add.at(
            matrix[row], index[valid], coefficient * left_weight[valid])
        np.add.at(
            matrix[row], index[valid] + 1,
            coefficient * right_weight[valid])

        fixed = _matching_kernels(
            xi, np.full_like(xi, abs(x_value)), pz_gev, mu, lambda_s)
        subtraction = -float(np.sum(weights * fixed))
        centre, centre_left, centre_right, in_support = (
            _linear_interpolation_weights(y, np.array([x_value])))
        if not in_support[0]:
            raise ValueError("matching output lies outside y-grid support")
        matrix[row, centre[0]] += subtraction * centre_left[0]
        matrix[row, centre[0] + 1] += subtraction * centre_right[0]
    if not np.isfinite(matrix).all():
        raise ValueError("matching matrix contains non-finite values")
    return matrix


def hR_PDF(xx, Pz_, conf, hR_tilde_data, mu_=2.0, lambda_s_fm=0.3):
    """NLO 胶子准 PDF 的显式一圈 plus-subtraction 匹配。

    Args:
        xx: x 网格（含负值，向量；精确零节点不支持）
        Pz_: 格点单位动量
        conf: 系综名
        hR_tilde_data: hR(x) 数组（与 xx 同形）
        mu_: 匹配标度（GeV，默认 2.0）
        lambda_s_fm: 大 λ 截断（fm，默认 0.3）
    Returns:
        hR_PDF（光锥 PDF 匹配结果）。
    """
    xx = _validate_uniform_grid(xx, "matching x grid")
    hR_tilde_data = np.asarray(hR_tilde_data)
    if hR_tilde_data.shape != xx.shape:
        raise ValueError(
            f"hR_tilde_data shape {hR_tilde_data.shape} "
            f"does not match x grid {xx.shape}")
    dx = xx[1] - xx[0]

    # A_s ≡ α_s/(4π)（zengch 惯例），×4π 还原真耦合。
    alpha_s = A_s_run(mu_) * 4.0 * pi
    Pz_GeV = pz_to_gev(Pz_, conf)
    lambda_s = lambda_s_fm / fm_to_GeV * Pz_GeV

    yy = xx
    matrix = _matching_matrix(yy, xx, Pz_GeV, mu_, lambda_s)
    coupling = alpha_s * CA / (2.0 * pi)
    return hR_tilde_data - coupling * (matrix @ hR_tilde_data)


def A_s_run(mu, Lambda_QCD=0.23, nf=3.0):
    from ._const import A_s
    return A_s(mu, Lambda_QCD, nf)


# ═══════════════════════════════════════════════════════════════════
# ratio 方案胶子匹配核（matching_cc.py）
# ═══════════════════════════════════════════════════════════════════

def C(ksi, m, r, alpha_s=0.296, cf=None):
    """比值方案夸克匹配核（matching_cc.py 原版 C(ξ,m,r) 三分区）。

    ξ>1、0<ξ<1、ξ<0 三分区 + 共同 Si 项，α_s·C_F/(2π) 归一。
    原版全局 Cf 未定义（调用即 NameError），此处默认取 _const.CF；
    原版固定 α_s=0.296（源注释 #change!）。
    """
    from ._const import CF
    cf = CF if cf is None else cf
    ksi = np.asarray(ksi, dtype=float)
    m_abs = np.abs(np.asarray(m, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        k_gt1 = ((1.0 + ksi ** 2.0) / (1.0 - ksi) * np.log(ksi / (ksi - 1.0))
                 + 1.0)
        k_mid = ((1.0 + ksi ** 2.0) / (1.0 - ksi)
                 * (-np.log(r ** 2.0) + np.log(4.0 * ksi * (1.0 - ksi)) - 1.0)
                 + 1.0)
        k_neg = (-(1.0 + ksi ** 2.0) / (1.0 - ksi) * np.log(-ksi / (1.0 - ksi))
                 - 1.0)
        si_term = 3.0 * Si((1.0 - ksi) * m_abs) / (np.pi * (1.0 - ksi))
        ker = np.where(ksi > 1.0, k_gt1,
                       np.where((ksi > 0.0) & (ksi < 1.0), k_mid,
                                np.where(ksi < 0.0, k_neg, np.nan))) + si_term
    out = alpha_s * cf * ker / (2.0 * np.pi)
    return out[()] if np.ndim(out) == 0 else out


def C_gluon_ratio(ksi, m, r, alpha_s=0.296):
    """胶子 ratio 方案匹配核（协变组合，matching_cc.py 原版三分区+Si 项）。

    m: (|x|·Z_s·a·pz)/GeVfm，r: μ/(|x|·pz)；α_s·C_A/(2π) 归一，
    原版固定 α_s=0.296、C_A=3。三分区结构与 _matching_kernels 的
    g_1/g_2/g_3 同型，另加共同 5/6·(−1/|1−ξ|+2Si/π(1−ξ)) 截断项。
    """
    ksi = np.asarray(ksi, dtype=float)
    m_abs = np.abs(np.asarray(m, dtype=float))
    poly_hi = ((11.0 - 28.0 * ksi + 18.0 * ksi ** 2.0 - 12.0 * ksi ** 3.0)
               / (6.0 * (1.0 - ksi)))
    pref = 2.0 * (1.0 - ksi + ksi ** 2.0) ** 2.0 / (1.0 - ksi)
    poly_lo = ((15.0 - 56.0 * ksi + 102.0 * ksi ** 2.0
                - 96.0 * ksi ** 3.0 + 48.0 * ksi ** 4.0)
               / (6.0 * (1.0 - ksi)))
    with np.errstate(divide="ignore", invalid="ignore"):
        k_gt1 = pref * np.log(ksi / (ksi - 1.0)) + poly_hi
        k_mid = (pref * (-np.log(np.square(r) / 4.0)
                         + np.log(ksi * (1.0 - ksi))) - poly_lo)
        k_neg = -pref * np.log(ksi / (ksi - 1.0)) - poly_hi
        ker = np.where(ksi > 1.0, k_gt1,
                       np.where((ksi > 0.0) & (ksi < 1.0), k_mid,
                                np.where(ksi < 0.0, k_neg, np.nan)))
        ker = ker + 5.0 / 6.0 * (
            -1.0 / np.abs(1.0 - ksi)
            + 2.0 * Si((1.0 - ksi) * m_abs) / (np.pi * (1.0 - ksi)))
    out = alpha_s * CA * ker / (2.0 * np.pi)
    return out[()] if np.ndim(out) == 0 else out
