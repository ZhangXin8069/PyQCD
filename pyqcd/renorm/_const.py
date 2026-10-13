"""QCD 常数：色因子、欧拉常数、π 与 1 圈运行耦合 α_s(μ)/(4π)。"""
import numpy as np

# ── 群论因子 ───────────────────────────────────────────────────────
CA = 3.0        # SU(3) 伴随表示 Casimir
CF = 4.0 / 3.0  # SU(3) 基础表示 Casimir
Nf = 3.0        # 活跃夸克味数（重整化群演化用）

# ── 数学常数 ───────────────────────────────────────────────────────
pi = np.pi
gammaE = 0.5772156649015329  # 欧拉-马歇罗尼常数

# ── 跑动耦合 ───────────────────────────────────────────────────────
# b0 = 11 − 2 Nf/3（1 圈 β 函数系数）


def b0(nf: float = Nf) -> float:
    return 11.0 - 2.0 * nf / 3.0


def alpha_s(mu: float, Lambda_QCD: float = 0.23, nf: float = Nf) -> float:
    """1 圈运行耦合 α_s(μ)（μ, Λ_QCD 单位一致，默认 GeV）。

    α_s(μ) = 2π / [b0 ln(μ/Λ_QCD)]
    """
    if any(isinstance(value, (bool, np.bool_))
           for value in (mu, Lambda_QCD, nf)):
        raise ValueError("mu、Lambda_QCD 与 nf 不得为布尔值")
    try:
        mu = float(mu)
        Lambda_QCD = float(Lambda_QCD)
        nf = float(nf)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "mu、Lambda_QCD 与 nf 必须是实数值") from exc
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu 必须是有限正数")
    if not np.isfinite(Lambda_QCD) or Lambda_QCD <= 0.0:
        raise ValueError("Lambda_QCD 必须是有限正数")
    if mu <= Lambda_QCD:
        raise ValueError("alpha_s 的一圈定义域要求 mu > Lambda_QCD")
    b0_value = b0(nf)
    if not np.isfinite(b0_value) or b0_value <= 0.0:
        raise ValueError("alpha_s 的定义域要求 b0(nf) > 0")
    return 2.0 * np.pi / (b0_value * np.log(mu / Lambda_QCD))


def A_s(mu: float, Lambda_QCD: float = 0.23, nf: float = Nf) -> float:
    """α_s(μ)/(4π)——zengch 代码惯例（fit_zr_new.py 用 alpha_s = A_s*4π）。"""
    return alpha_s(mu, Lambda_QCD, nf) / (4.0 * np.pi)
