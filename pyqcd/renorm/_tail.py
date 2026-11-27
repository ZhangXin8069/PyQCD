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


def analytic_tail_fourier(x_, C, B, eta, lambda_left):
    """``(1/π)Re∫_{λ_L}^∞ (C + B/λ)e^{−ηλ}e^{ixλ}dλ`` 的闭式。

    ``C e^{−sλ_L}/s + B E₁(sλ_L)``，``s = η − ix``；``B=0`` 时为 LA，
    ``B≠0`` 为 NLA。``λ_L`` 必须为正（``E₁`` 在 0 处发散，与 ``B/λ`` 的
    非可积性对应）。
    """
    x_values = np.asarray(x_, dtype=float)
    scalar = x_values.ndim == 0
    flat = np.atleast_1d(x_values).ravel()
    if not np.isfinite(flat).all():
        raise ValueError("x_ 必须是有限实数")
    C = float(C)
    B = float(B)
    if not np.isfinite(C) or not np.isfinite(B):
        raise ValueError("C 与 B 必须是有限实数")
    eta = _positive("eta", eta)
    lambda_left = _positive("lambda_left", lambda_left)
    s = eta - 1j * flat
    value = C * np.exp(-s * lambda_left) / s + B * exp1(s * lambda_left)
    out = np.real(value) / pi
    result = out.reshape(x_values.shape)
    return float(result) if scalar else result


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


__all__ = [
    "tail_shape", "tail_amplitude_fit", "analytic_tail_fourier",
    "tail_fourier_continuum", "abrupt_cutoff_error",
]
