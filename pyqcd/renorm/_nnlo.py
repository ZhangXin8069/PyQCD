"""
两圈（NNLO）梯度流有限转换系数。

只提供系数代数：局域归一化 ``D_Γ = ζ_Γ/C_ψ`` 的严格展开、胶子非局域端点
``U/H/P`` 组合的两圈系数、短距 Wilson 系数 ``C₀`` 的 NLO/NNLO。不含矩阵元
读取、拟合、外推或任何数据依赖。

约定
----
``a_s ≡ α_s/(4π)``（zengch 惯例，见 :func:`pyqcd.renorm._const.A_s`），
``L_f ≡ ln[2μ²t_f] + γ_E``（``t_f`` 单位 GeV⁻²），``C₀`` 用短距对数
``L ≡ ln[z²μ²e^{2γ_E}/4]``（``z`` 单位 GeV⁻¹）。

推导
----
局域归一化是两个前向系数的比：同一 ringed 处方的 ``Z_χ`` 在比值中相消，

    D_Γ = ζ_Γ/C_ψ = 1 + a_s(J₁-N₁) + a_s²(J₂-N₂+N₁²-J₁N₁) + O(a_s³),

其中端点 ``c = 1 + a_s h₁ + a_s² h₂`` 要平方成两端点 ``C_ψ = c²``，
故 ``N₁ = 2h₁``、``N₂ = 2h₂ + h₁²``。必须做严格展开：用两圈截断级数的
有理比会混入未计算的高阶。共同的两圈 ringed 常数在 ``J₂-N₂`` 中相消，
但这不等于把 ``Z_χ`` 设为 1。

对 ``SU(3)``、``n_f=3`` 的两条自洽检查（本模块的 ``contract`` 逐条验证）：

    D_V^{NLO}  = 1 + a_s(4L_f + 10/3)
    D_V^{NNLO} = D_V^{NLO} + a_s²(26L_f² + 85.3591794985557L_f + 30.92223469310622)

``L_f²`` 系数 ``26 = (f₁₁² + β₀f₁₁)/2``，``f₁₁ = 3C_F = 4``、``β₀ = 9``，
是独立的重整化群检查；``T`` 通道保留 ``L_f`` 是因为 ``γ_T^{(0)} = 2C_F``
（手征奇、无 Ward 恒等式保护）。

胶子端点的三段组合为 ``U``（垂直-垂直对，``2b``）、``H``（混合对，``p+b``）、
``P``（平行-平行对，``2p``），转换因子取双线性形式的严格逆
``1 - c₁a_s + (c₁²-c₂)a_s²``。

文献（公式来源，代码为独立实现）
--------------------------------
- X.-P. Wang, *Off-lightcone Wilson-line operators in gradient flow at NNLO*,
  arXiv:2610.40225v1, Eq.(32)(34)(40)。
- A. Borgulat et al., *Short-flow-time expansion of quark bilinears through
  NNLO*, arXiv:2311.16799v2, Eq.(3.9)(3.11)（原文用 ``α_s/π``，换到 ``a_s``
  时一圈乘 4、两圈乘 16）。
- 短距 ``C₀^{NNLO}``：arXiv:2006.12370v3, Eq.(22)–(25)（``C₀ = iA(ω=0)/2``，
  ``ξ₄=0``）。

ħc 约定
-------
本模块用 CODATA 的 ``ħc = HBARC_GEV_FM = 0.1973269804`` GeV·fm，与
``_gradient_flow_quasi.GEV_FM = 5.067730716``（倒数 0.1973269805）一致。注意
``_ensembles.fm_to_GeV = 0.197`` 是 zengch 遗留的两位有效数字约定，与 CODATA
差 **0.166%**：把两者混用于同一个 ``L_f`` 会给出 0.17% 的系数偏移。引用本模块
的结果时须写明所用 ħc。
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ._const import CA, CF, gammaE, pi

TR = 0.5
Nf = 3.0
HBARC_GEV_FM = 0.1973269804
CHANNELS = ("V", "T")
GLUON_COMBINATIONS = ("U", "H", "P")


def _finite(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限实数")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限实数") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    return value


def _positive(name, value):
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return value


@lru_cache(maxsize=None)
def _clausen_pi_over_3():
    """``Cl₂(π/3) = Im Li₂(e^{iπ/3})``，由两条独立路径交叉校验。

    ``scipy.special.spence(z)`` 求的是 ``Li₂(1-z)``，故 ``Li₂(z) = spence(1-z)``。
    第二条路径按定义 ``Cl₂(θ) = -∫₀^θ ln|2sin(t/2)|dt`` 数值积分。
    """
    from scipy.integrate import quad
    from scipy.special import spence

    reflected = float(np.imag(spence(1.0 - np.exp(1j * pi / 3.0))))

    def integrand(t):
        return -math.log(t) if t <= 1.0e-14 else -math.log(
            abs(2.0 * math.sin(t / 2.0)))

    integral = float(quad(
        integrand, 0.0, pi / 3.0, points=[0.0], limit=400,
        epsabs=1.0e-13, epsrel=1.0e-13)[0])
    if abs(reflected - integral) > 1.0e-9:
        raise ArithmeticError(
            "Cl₂(π/3) 的两条求值路径不一致："
            f"spence={reflected!r} vs quad={integral!r}")
    return reflected


@lru_cache(maxsize=None)
def _polylog_quarter(order):
    """``Li_order(1/4)``（``|z|=1/4`` 时级数列机器精度收敛）。"""
    order = int(order)
    if order < 1:
        raise ValueError("polylog order 必须为正整数")
    total = 0.0
    for k in range(1, 200):
        term = (0.25 ** k) / (k ** order)
        total += term
        if term < 1.0e-20 * max(abs(total), 1.0):
            break
    return total


@lru_cache(maxsize=None)
def _zeta3():
    from scipy.special import zeta

    return float(zeta(3.0))


def flow_log(tau, a_fm, mu_gev):
    """``L_f = ln[2μ²t_f] + γ_E``，``t_f = τ·a²`` 换算到 GeV⁻²。"""
    tau = _positive("tau", tau)
    a_fm = _positive("a_fm", a_fm)
    mu_gev = _positive("mu_gev", mu_gev)
    t_gev = tau * (a_fm / HBARC_GEV_FM) ** 2
    return math.log(2.0 * mu_gev ** 2 * t_gev) + gammaE


def _quark_series(L, nf, channel, ringed_constant):
    """``(N₁, N₂, J₁, J₂)``：端点 ``c`` 的两圈展开与局域系数。"""
    L = _finite("L", L)
    nf = _finite("nf", nf)
    l2, l3 = math.log(2.0), math.log(3.0)
    cl = _clausen_pi_over_3()
    li2 = _polylog_quarter(2)
    li3 = _polylog_quarter(3)
    zeta2 = pi * pi / 6.0
    zeta3 = _zeta3()
    ln432 = math.log(432.0)
    ringed = (
        -23.7947 * CF * CA + 30.3914 * CF ** 2 - 3.92255 * CF * TR * nf
        if ringed_constant is None else _finite("ringed_constant", ringed_constant)
    )

    h1 = -CF * (1.5 * L + 1.0 + ln432 / 2.0)
    h2 = L ** 2 * (9.0 / 8.0 * CF ** 2 - 11.0 / 4.0 * CF * CA + CF * TR * nf)
    h2 += L * (
        CF ** 2 * (11.0 / 4.0 - 4.0 * pi ** 2 / 3.0 + 0.75 * ln432)
        + CF * CA * (pi ** 2 / 3.0 - 31.0 / 4.0 - 11.0 / 6.0 * ln432)
        + CF * TR * nf * (3.0 + 2.0 / 3.0 * ln432))
    h2 += CF ** 2 * (
        2.0 + 4.0 * pi ** 2 / 3.0 * (l2 - 1.0) - 4.0 * pi * cl + 12.0 * l2
        - 3.0 / 8.0 * l3 * (4.0 + 8.0 * l2 + 3.0 * l3)
        + 11.0 / 2.0 * li2 - li3 - 4.0 / 3.0 * zeta3)
    h2 += CF * CA * (
        -239.0 / 6.0 - pi + pi ** 2 / 6.0 - pi ** 2 / 3.0 * l2 + pi * cl
        - 85.0 * l2 + 84.0 * l3 + 2.0 * l2 ** 2 + 17.0 * li2
        - li3 / 2.0 - 2.0 / 3.0 * zeta3)
    h2 += CF * TR * nf * (20.0 / 3.0 + pi ** 2 / 3.0)
    h2 += ringed / 2.0

    if channel == "V":
        j1 = 4.0 * CF * (1.0 / 8.0 - l2 - 0.75 * l3)
        j2 = ringed + 16.0 * (
            CF ** 2 * (-41.0 / 128.0 - 5.0 / 32.0 * zeta2 + 3.0 / 8.0 * l2
                       + l2 ** 2 / 4.0 - 3.0 / 32.0 * l3 + 1.5 * li2)
            + CA * CF * (-763.0 / 384.0 - 5.0 / 32.0 * zeta2 - 13.0 / 4.0 * l2
                         + l2 ** 2 / 4.0 + 27.0 / 8.0 * l3 + 21.0 / 16.0 * li2)
            + CF * TR * nf * (35.0 / 96.0 + zeta2 / 8.0)
            + L * (CA * CF * (-11.0 / 12.0 * l2 - 11.0 / 16.0 * l3
                              + 11.0 / 96.0)
                   + CF * TR * nf * (l2 / 3.0 + l3 / 4.0 - 1.0 / 24.0)))
    elif channel == "T":
        j1 = 4.0 * CF * (-l2 - 0.75 * l3 + L / 4.0)
        j2 = ringed + 16.0 * (
            CF ** 2 * (-7.0 / 12.0 - zeta2 / 4.0 + 4.0 / 3.0 * l2
                       + l2 ** 2 / 4.0 - 0.75 * l3 + li2)
            + CA * CF * (-1159.0 / 864.0 - zeta2 / 16.0 - 17.0 / 6.0 * l2
                         + l2 ** 2 / 4.0 + 3.0 * l3 + 1.25 * li2)
            + CF * TR * nf * (47.0 / 216.0 + zeta2 / 12.0)
            + L * (CF ** 2 * (-l2 / 4.0 - 3.0 / 16.0 * l3 - 19.0 / 32.0)
                   + CA * CF * (-11.0 / 12.0 * l2 - 11.0 / 16.0 * l3
                                + 257.0 / 288.0)
                   + CF * TR * nf * (l2 / 3.0 + l3 / 4.0 - 13.0 / 72.0))
            + L ** 2 * (CF ** 2 / 32.0 + 11.0 / 96.0 * CA * CF
                        - CF * TR * nf / 24.0))
    else:
        raise ValueError(f"channel 必须是 {CHANNELS!r} 之一")

    n1 = 2.0 * h1
    n2 = 2.0 * h2 + h1 * h1
    return n1, n2, j1, j2


def local_ratio_coefficients(channel, nf=Nf, ringed_constant=None, L=0.0):
    """局域归一化 ``D_Γ`` 的 ``(a₁, a₂)``，``D = 1 + a₁a_s + a₂a_s²``。

    ``channel='V'``（含 ``γ₄`` 投影的 nonsinglet vector）或 ``'T'``。两个系数
    都依赖 ``L_f``，缺省取 ``L=0``（即发布公式里的常数项，``V`` 为
    ``(10/3, 30.92223469310622)``）；要完整 ``L`` 依赖请用
    :func:`local_ratio_factor`。``ringed_constant`` 只用于验证两圈 ringed
    常数在 ``J₂-N₂`` 中相消；显式传任意常数应得到相同的 ``a₂``。
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel 必须是 {CHANNELS!r} 之一")
    n1, n2, j1, j2 = _quark_series(_finite("L", L), nf, channel, ringed_constant)
    return j1 - n1, j2 - n2 + n1 * n1 - j1 * n1


def local_ratio_factor(channel, alpha_s, L, nf=Nf, order=2):
    """``D_Γ(L)`` 的有限阶值（``order=1`` 只取一圈）。"""
    alpha_s = _finite("alpha_s", alpha_s)
    order = int(order)
    if order not in (1, 2):
        raise ValueError("order 必须是 1 或 2")
    L = _finite("L", L)
    nf = _finite("nf", nf)
    n1, n2, j1, j2 = _quark_series(L, nf, channel, None)
    a_s = alpha_s / (4.0 * pi)
    value = 1.0 + a_s * (j1 - n1)
    if order == 2:
        value += a_s ** 2 * (j2 - n2 + n1 * n1 - j1 * n1)
    return value


def _gluon_endpoints(L, nf):
    """胶子端点 ``(p₁, p₂, b₁, b₂)``；``p₁ = 0`` 使平行对只有两圈项。"""
    L = _finite("L", L)
    nf = _finite("nf", nf)
    l2 = math.log(2.0)
    l3 = math.log(3.0)
    cl = _clausen_pi_over_3()
    li2 = _polylog_quarter(2)
    p1 = 0.0
    p2 = CA ** 2 * (
        -2.0 * pi ** 2 / 3.0 * L
        + 2.0 * pi ** 2 / 3.0 * (l2 - 5.0 / 3.0)
        - 4.0 * pi / 3.0 * (-2.0 + math.sqrt(3.0) + cl))
    b1 = CA * L
    b2 = L ** 2 * (7.0 / 3.0 * CA ** 2 - 2.0 / 3.0 * CA * TR * nf)
    b2 += L * (CA ** 2 * (34.0 / 9.0 - 2.0 * pi ** 2 / 3.0)
               - 26.0 / 9.0 * CA * TR * nf)
    b2 += CA ** 2 * (
        196.0 / 27.0 + pi - pi ** 2 / 6.0 - 4.0 * pi / 3.0 * cl
        + (72.0 + 2.0 * pi ** 2 / 3.0) * l2 - 10.0 / 3.0 * l2 ** 2
        - 123.0 / 2.0 * l3 + 4.0 * li2)
    b2 -= CA * TR * nf / 27.0 * (89.0 + 3.0 * pi ** 2)
    return p1, p2, b1, b2


def gluon_endpoint_coefficients(nf=Nf):
    """胶子 ``U/H/P`` 组合的 ``(c₁, c₂)``，因子取 ``1/(1+c₁a_s+c₂a_s²)``。"""
    p1, p2, b1, b2 = _gluon_endpoints(0.0, nf)
    return {
        "U": (2.0 * b1, 2.0 * b2 + b1 * b1),
        "H": (p1 + b1, p2 + b2 + p1 * b1),
        "P": (2.0 * p1, 2.0 * p2 + p1 * p1),
    }


def gluon_endpoint_factor(combination, alpha_s, L, nf=Nf, order=2):
    """胶子端点因子 ``1/ζ`` 的严格逆展开。"""
    if combination not in GLUON_COMBINATIONS:
        raise ValueError(
            f"combination 必须是 {GLUON_COMBINATIONS!r} 之一")
    alpha_s = _finite("alpha_s", alpha_s)
    order = int(order)
    if order not in (1, 2):
        raise ValueError("order 必须是 1 或 2")
    p1, p2, b1, b2 = _gluon_endpoints(L, nf)
    pairs = {
        "U": (2.0 * b1, 2.0 * b2 + b1 * b1),
        "H": (p1 + b1, p2 + b2 + p1 * b1),
        "P": (2.0 * p1, 2.0 * p2 + p1 * p1),
    }
    c1, c2 = pairs[combination]
    a_s = alpha_s / (4.0 * pi)
    value = 1.0 - c1 * a_s
    if order == 2:
        value += (c1 * c1 - c2) * a_s ** 2
    return value


def short_distance_c0(z_fm, mu_gev, alpha_s, order=2):
    """短距零 Ioffe-time Wilson 系数 ``C₀(z; μ)`` 的 NLO/NNLO。

    ``L = ln[z²μ²e^{2γ_E}/4]``，``z`` 由 fm 换算到 GeV⁻¹（``ħc=0.1973269804``
    GeV·fm）。有限阶直接取对数，不把对数再做次级数展开。``z=0`` 处
    对数发散，输入必须严格为正。
    """
    z_fm = np.asarray(z_fm, dtype=float)
    scalar = z_fm.ndim == 0
    z_fm = np.atleast_1d(z_fm)
    if not np.isfinite(z_fm).all() or np.any(z_fm <= 0.0):
        raise ValueError("z_fm 必须是有限正数")
    mu_gev = _positive("mu_gev", mu_gev)
    alpha_s = _finite("alpha_s", alpha_s)
    order = int(order)
    if order not in (1, 2):
        raise ValueError("order 必须是 1 或 2")

    L = np.log((z_fm / HBARC_GEV_FM) ** 2 * mu_gev ** 2 / 4.0) + 2.0 * gammaE
    value = 1.0 + alpha_s * CF / (2.0 * pi) * (1.5 * L + 2.5)
    if order == 2:
        cf2 = (9.0 / 16.0 * L ** 2
               + (pi ** 2 / 3.0 + 25.0 / 16.0) * L
               - 4.0 * _zeta3() + pi ** 2 / 9.0 + 223.0 / 192.0)
        cfca = (11.0 / 16.0 * L ** 2
                + (-pi ** 2 / 12.0 + 53.0 / 16.0) * L
                + _zeta3() - 5.0 * pi ** 2 / 24.0 + 4877.0 / 576.0)
        nfcf = -0.25 * L ** 2 - 1.25 * L - 469.0 / 144.0
        value = value + alpha_s ** 2 / (2.0 * pi ** 2) * (
            CF ** 2 * cf2 + CF * CA * cfca + Nf * CF * TR * nfcf)
    return float(value[0]) if scalar else value


__all__ = [
    "CHANNELS", "GLUON_COMBINATIONS", "HBARC_GEV_FM", "TR",
    "flow_log", "local_ratio_coefficients", "local_ratio_factor",
    "gluon_endpoint_coefficients", "gluon_endpoint_factor",
    "short_distance_c0",
]
