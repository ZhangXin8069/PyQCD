"""
长距尾部模型与解析 Fourier（LA/NLA 形式 + 指数积分闭式）。

背景
----
混合方案在 ``λ = zP_z`` 上的外推目前用幂-指数形

    H_tail(λ) = l₁ λ^{−a₁} e^{−λ/λ₀},

它必须把 ``λ`` 截断到有限上限，截断误差无法量化。对开放 Wilson 线矩阵元，
大 ``z`` 的场论行为是长距渐近式

    h^{LA}(z)  = A e^{iφ sgn z} e^{−Ω|z|},
    h^{NLA}(z) = (A + A'/|z|) e^{iφ' sgn z} e^{−Ω|z|},

满足 Hermitian 条件 ``h(−z) = h(z)*``。换到 ``λ`` 变量（``H(λ) = h(z)``、
``η = Ω/P_z > 0``）即 ``H_tail(λ) = (C + B/λ) e^{−ηλ}``。

本模块给出的闭式 Fourier 是（约定 ``q(x) = (1/π)Re∫₀^∞ dλ H(λ)e^{ixλ}``、
``s = η − ix``、尾部起点 ``λ_L``）

    q_tail(x) = (1/π) Re[ C e^{−sλ_L}/s + B E₁(sλ_L) ],

于是尾部可以积到无穷，不需要 ``λ_max`` 截断。``Ω``（进而 ``η``）没有第一性原理
数值，只能由数据拟合；尾部模型、起点、方案与参数不确定度不因本闭式而消失。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import exp1

from ._const import pi


def _positive(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限正数")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限正数") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return value


def tail_shape(lambda_, eta, exponent=None):
    """尾部衰减形状 ``λ^{−p} e^{−ηλ}``，缺省 ``p=0``（LA）。"""
    values = np.asarray(lambda_, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("lambda_ 必须是有限非负数")
    eta = _positive("eta", eta)
    exponent = 0.0 if exponent is None else float(exponent)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = np.exp(-eta * values)
        if exponent != 0.0:
            shape = shape * np.power(values, -exponent)
    return shape


def tail_amplitude_fit(lambda_, data, errors, eta, exponent=None):
    """单幅度全协方差（对角误差）GLS：``H_tail = A·shape`` 的 ``A``。

    这是"尾部形状假设 + 一个幅度"的最小拟合：形状（``η``、幂次）由外部固定，
    不在此函数内搜索，避免形状与幅度互相补偿。
    """
    values = np.asarray(lambda_, dtype=float)
    data = np.asarray(data, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if values.ndim != 1 or data.shape != values.shape or errors.shape != values.shape:
        raise ValueError("lambda_/data/errors 必须是一维且等长")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("lambda_ 必须是有限正数")
    if not np.isfinite(data).all():
        raise ValueError("data 必须全部有限")
    if not np.isfinite(errors).all() or np.any(errors <= 0.0):
        raise ValueError("errors 必须是有限正数")
    shape = tail_shape(values, eta, exponent)
    if not np.isfinite(shape).all():
        raise ValueError("尾部形状在给定参数下非有限")
    weight = 1.0 / errors ** 2
    denominator = float(np.sum(weight * shape * shape))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("尾部幅度拟合的分母非正")
    amplitude = float(np.sum(weight * shape * data) / denominator)
    residual = data - amplitude * shape
    chi2 = float(np.sum(weight * residual ** 2))
    return amplitude, chi2, float(np.sqrt(1.0 / denominator))


def _tail_moment(x_values, C, B, eta, lambda_left):
    """``∫_{λ_L}^∞ (C+B/λ)e^{−sλ}dλ`` 的闭式，``s = η − ix``（复数返回）。"""
    C = complex(C)
    B = complex(B)
    eta = _positive("eta", eta)
    lambda_left = _positive("lambda_left", lambda_left)
    flat = np.atleast_1d(np.asarray(x_values, dtype=float)).ravel()
    s = eta - 1j * flat
    value = C * np.exp(-s * lambda_left) / s + B * exp1(s * lambda_left)
    return value.reshape(np.shape(x_values))


def analytic_tail_fourier(x_, C, B, eta, lambda_left, channel="cos"):
    """``(1/π)∫_{λ_L}^∞ (C + B/λ)e^{−ηλ}·{cos,sin}(xλ)dλ`` 的闭式。

    核心量 ``A_± = ∫_{λ_L}^∞ (C+B/λ)e^{−(η∓ix)λ}dλ``；由
    ``e^{−sλ} = e^{−ηλ}(cos xλ + i sin xλ)`` 得

        ``I_cos = (A_+ + A_−)/2``，``I_sin = (A_+ − A_−)/(2i)``，

    因此 ``channel='cos'`` 返回 ``I_cos/π``、``'sin'`` 返回 ``I_sin/π``。
    ``C``、``B`` 可为复数（Hermitian 数据的实/虚部各带各自幅度）——此时必须
    同时取 ``A_+`` 与 ``A_−``，只用 ``A_+`` 的实部会丢掉虚部幅度贡献。
    实 ``C,B`` 时走 ``Re(A_+)/Im(A_+)`` 的历史路径（与 `stab54` 逐位一致），
    返回 ``float``；复幅度返回 ``complex``。

    ``η`` 与 ``λ_L`` 必须为实正数；``λ_L > 0`` 是硬要求（``E₁`` 在 0 处发散，
    与 ``B/λ`` 的非可积性对应）。``B=0`` 即 LA（纯指数尾），``B≠0`` 为 NLA。
    """
    if channel not in ("cos", "sin"):
        raise ValueError("channel 必须是 'cos' 或 'sin'")
    x_values = np.asarray(x_, dtype=float)
    scalar = x_values.ndim == 0
    flat = np.atleast_1d(x_values).ravel()
    if not np.isfinite(flat).all():
        raise ValueError("x_ 必须是有限实数")
    if isinstance(C, (bool, np.bool_)) or isinstance(B, (bool, np.bool_)):
        raise ValueError("C 与 B 不得为布尔值")
    C_real = not np.iscomplexobj(np.asarray(C))
    B_real = not np.iscomplexobj(np.asarray(B))
    C_value = float(np.real(C)) if C_real else complex(C)
    B_value = float(np.real(B)) if B_real else complex(B)
    if not np.isfinite(C_value) or not np.isfinite(B_value):
        raise ValueError("C 与 B 必须是有限复数")
    eta_value = _positive("eta", eta)
    lambda_left = _positive("lambda_left", lambda_left)
    if C_real and B_real:
        # 历史路径逐位保持：实幅度时 Re(A₊) 就是 cos 通道、Im(A₊) 就是 sin
        # 通道，不走 (A₊±A₋)/2（后者差 1 ULP，会破坏历史产物的逐位复现）。
        a_plus = _tail_moment(flat, C_value, B_value, eta_value, lambda_left)
        out = ((np.real(a_plus) if channel == "cos" else np.imag(a_plus))
               / pi).reshape(x_values.shape)
        if scalar:
            return float(out[()])
        return out
    a_plus = _tail_moment(flat, C_value, B_value, eta_value, lambda_left)
    s_minus = eta_value + 1j * flat
    a_minus = (C_value * np.exp(-s_minus * lambda_left) / s_minus
               + B_value * exp1(s_minus * lambda_left))
    value = (0.5 * (a_plus + a_minus) if channel == "cos"
             else (a_plus - a_minus) / (2.0j))
    out = (value / pi).reshape(x_values.shape)
    if scalar:
        return complex(out[()])
    return out


def tail_fourier_continuum(x_, C, B, eta, lambda_left, n_nodes=4096,
                          upper=None):
    """数值对照：对同一被积函数做直接求积（``upper`` 缺省 ``40/η``）。

    只用于验证 :func:`analytic_tail_fourier`；生产路径不依赖它。
    """
    x_values = np.asarray(x_, dtype=float)
    flat = np.atleast_1d(x_values).ravel()
    eta = _positive("eta", eta)
    lambda_left = _positive("lambda_left", lambda_left)
    upper = 40.0 / eta if upper is None else _positive("upper", upper)
    if upper <= lambda_left:
        raise ValueError("upper 必须大于 lambda_left")
    nodes = np.linspace(lambda_left, upper, int(n_nodes))
    weight = np.gradient(nodes)
    integrand = (float(C) + float(B) / nodes) * np.exp(-eta * nodes)
    cosine = np.cos(np.outer(flat, nodes))
    out = (1.0 / pi) * (cosine @ (integrand * weight))
    return out.reshape(x_values.shape)


def abrupt_cutoff_error(x_, C, B, eta, lambda_max):
    """把尾部截断在 ``λ_max`` 丢掉的部分（同一闭式从 ``λ_max`` 积到 ∞）。

    与 :func:`analytic_tail_fourier` 同约定：返回
    ``(1/π)Re∫_{λ_max}^∞ (C+B/λ)e^{−ηλ}e^{ixλ}dλ``。这是"现有 ``hR_x``
    用有限 ``λ_max`` 时被丢弃的尾部贡献"，可直接作为系统误差量级诊断。
    """
    x_values = np.asarray(x_, dtype=float)
    lambda_max = _positive("lambda_max", lambda_max)
    return analytic_tail_fourier(x_values, C, B, eta, lambda_max)


def power_exp_tail_fourier(x_, l1, a1, lambda0, lambda_left, n_nodes=88,
                          check=True, check_tolerance=1.0e-3):
    """幂-指数尾 ``l₁λ^{−a₁}e^{−λ/λ₀}`` 从 ``λ_L`` 积到 ∞ 的 Fourier。

    被积式 ``(1/π)∫_{λ_L}^∞ dλ l₁λ^{−a₁}e^{−λ/λ₀}cos(xλ)``。没有初等闭式，
    用 Gauss–Laguerre：令 ``λ = λ_L + λ₀t``，权重 ``e^{−t}`` 正好是 Laguerre 权，

    ``q(x) = l₁λ₀e^{−λ_L/λ₀}/π · Σ_i w_i (λ_L+λ₀t_i)^{−a₁}cos(x(λ_L+λ₀t_i))``。

    精度契约（对 ``scipy.integrate.quad`` 实测，``n_nodes=88``，取
    ``x ∈ {0, 0.3, 0.9, 2, 4}``、``a1 ∈ {0.5, 1.3, 2.5}`` 的最坏值）：

    | ``λ_L/λ₀`` | 最大相对误差 |
    |---|---|
    | 4 | ~1e-6 |
    | 1 | ~3.5e-8 |
    | 0.5 | ~3e-8 |
    | 0.33 | ~7e-7 |
    | 0.2 | ~1e-4 |
    | 0.1 | ~9e-4 |
    | 0.05 | ~3e-4 或触发自检失败 |

    即 ``λ_L`` 远离尾部起点时精确，``λ_L ≲ 0.2λ₀`` 时 GL 无法分辨近零处的幂次
    奇异。``check=True``（默认）用 2 倍节点数（上限 184）重算，差异超过
    ``check_tolerance`` 时抛 ``ArithmeticError`` —— 宁可失败也不静默返回坏值；
    自洽度量在 ``args[1]``。

    节点数上限 96 来自 ``numpy.polynomial.laguerre.laggauss`` 自身：``n ≥ 192``
    时其权重归一化出现 0/0（实测首个坏点 ``n=192``，最大可用 184）。
    """
    x_values = np.asarray(x_, dtype=float)
    scalar = x_values.ndim == 0
    flat = np.atleast_1d(x_values).ravel()
    if not np.isfinite(flat).all():
        raise ValueError("x_ 必须是有限实数")
    l1 = float(l1)
    if not np.isfinite(l1):
        raise ValueError("l1 必须是有限实数")
    a1 = float(a1) if a1 is not None else 0.0
    if not np.isfinite(a1):
        raise ValueError("a1 必须是有限实数")
    lambda0 = _positive("lambda0", lambda0)
    lambda_left = _positive("lambda_left", lambda_left)
    n_nodes = int(n_nodes)
    if not 8 <= n_nodes <= 96:
        raise ValueError("n_nodes 必须落在 [8, 96]")
    reference_nodes = min(2 * n_nodes, 184)

    def evaluate(n):
        from numpy.polynomial.laguerre import laggauss

        nodes, weights = laggauss(n)
        if not np.isfinite(nodes).all() or not np.isfinite(weights).all():
            raise ArithmeticError(
                f"Gauss-Laguerre(n={n}) 节点/权重非有限，请减小 n_nodes")
        lam = lambda_left + lambda0 * nodes
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            shape = np.power(lam, -a1)
        if not np.isfinite(shape).all():
            raise ArithmeticError(
                "幂-指数尾在 Gauss-Laguerre 节点上出现非有限值；"
                "检查 a1 与 lambda_left 的取值")
        prefactor = l1 * lambda0 * np.exp(-lambda_left / lambda0) / pi
        cosine = np.cos(np.outer(flat, lam))
        return prefactor * (cosine @ (weights * shape))

    out = evaluate(n_nodes)
    if check and reference_nodes > n_nodes:
        reference = evaluate(reference_nodes)
        scale = np.maximum(np.abs(reference), 1.0e-30 * max(abs(l1), 1.0))
        self_consistency = float(np.max(np.abs(out - reference) / scale))
        if not np.isfinite(self_consistency) or (
                self_consistency > check_tolerance):
            raise ArithmeticError(
                "幂-指数尾 Fourier 的 Gauss-Laguerre 自洽检查失败："
                f"n={n_nodes} 与 n={reference_nodes} 的相对差 "
                f"{self_consistency:.3e} 超过 {check_tolerance:.1e}"
                "（lambda_left/lambda0 过小）；改用 tail_model='la_nla' "
                "或提高 n_nodes/放宽 check_tolerance",
                self_consistency)
    result = out.reshape(x_values.shape)
    return float(result) if scalar else result


def tail_fourier_tail(x_, model, lambda_left, *, l1=None, a1=None,
                      lambda0=None, coefficients=None):
    """按 ``model`` 分派尾部 Fourier：``'power_exp'`` 或 ``'la_nla'``。"""
    if model == "power_exp":
        if l1 is None or lambda0 is None:
            raise ValueError("power_exp 尾部需要 l1 与 lambda0")
        return power_exp_tail_fourier(
            x_, l1, 0.0 if a1 is None else a1, lambda0, lambda_left)
    if model == "la_nla":
        if coefficients is None:
            raise ValueError("la_nla 尾部需要 coefficients=(C, B, eta)")
        C, B, eta = (float(value) for value in coefficients)
        return analytic_tail_fourier(x_, C, B, eta, lambda_left)
    raise ValueError(
        f"未知 tail model {model!r}；可选 'power_exp' 或 'la_nla'")


__all__ = [
    "tail_shape", "tail_amplitude_fit", "analytic_tail_fourier",
    "tail_fourier_continuum", "abrupt_cutoff_error",
    "power_exp_tail_fourier", "tail_fourier_tail",
]
