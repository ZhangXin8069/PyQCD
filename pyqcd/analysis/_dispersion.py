"""
色散关系拟合（移植 zengch fit_E0.py 核心逻辑）
================================================

    E(Pz) = √( m² + k₂·Pz² + k₃·Pz⁴·a² )

从多个动量的有效能量 E0(Pz) 拟合核子质量 m 与色散系数 k₂、k₃。
"""
from __future__ import annotations

import numpy as np

from ..renorm._ensembles import Nl_set


def pz_to_gev_lattice(pz, nl, a_gev):
    """格点单位动量 → GeV：Pz·(2π)/(N_l·a)。"""
    return pz * 2.0 * np.pi / (nl * a_gev)


def th_E0(Pz_gev, m, k2, k3, a_gev):
    """色散关系参数化（Pz_gev 单位 GeV；k₃ 项为 O(a²) 离散化修正）。"""
    return np.sqrt(m ** 2 + k2 * Pz_gev ** 2 + k3 * Pz_gev ** 4 * a_gev ** 2)


def fit_dispersion(E0_list, Pz_list, a_gev, errors=None, conf=None):
    """最小二乘拟合 (m, k2, k3)。

    Args:
        E0_list: 有效能量数组（GeV，逐 Pz）。
        Pz_list: 动量数组（格点单位）。
        a_gev: 格距（GeV⁻¹）。
        errors: 逐点误差（可选，用于加权）。
        conf: 系综名（用于 N_l 查表；默认 24）。
    Returns:
        (m, k2, k3) 拟合值（tuple）。
    """
    E0 = np.asarray(E0_list, dtype=float)
    pz_lat = np.asarray(Pz_list, dtype=float)
    nl = Nl_set.get(conf, 24) if conf else 24
    pz_gev = pz_to_gev_lattice(pz_lat, nl, a_gev)

    def model(par, p):
        m, k2, k3 = par
        return np.sqrt(m ** 2 + k2 * p ** 2 + k3 * p ** 4 * a_gev ** 2)

    if errors is None:
        errors = np.ones_like(E0)
    errors = np.asarray(errors, dtype=float)
    w = 1.0 / np.maximum(errors, 1e-10) ** 2

    from scipy.optimize import least_squares
    res = least_squares(lambda par: np.sqrt(w) * (model(par, pz_gev) - E0),
                        x0=[0.9, 1.0, 0.0])
    return tuple(res.x)


def dispersion_check(E0_P0, E0_P2, pz_gev, m_ref=None):
    """色散关系核对：k₂_eff = (E(Pz)² − E(0)²)/Pz²，返回 (k₂_eff, 偏差%)。"""
    k2_eff = (E0_P2 ** 2 - E0_P0 ** 2) / pz_gev ** 2
    dev = abs(k2_eff - 1.0) * 100
    return k2_eff, dev
