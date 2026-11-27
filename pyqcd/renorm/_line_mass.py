"""
流化开放 Wilson 线的局域质量核与 H3 对数差模型。

物理对象
--------
四维 Euclidean、Landau 规范、树级流化传播子

    D̃_{μν}(p;t) = e^{−2tp²}/p² · (δ_{μν} − p_μp_ν/p²)

给出的开放直 Wilson 线 ``W_R(z,t)`` 的局部有效质量

    μ_R(z,r) ≡ −∂_z ln W_R(z,r) = d_R^{(1)} K(r,z) + O(g⁴),
    d_R^{(1)} = α_s C_R/√π,   r ≡ √(8t),
    K(r,z) = (1/r)·F(z/r),

    F(x) = erf(x) + (1/(2√π))[ (2e^{−x²}−3)/x + (1−e^{−x²})/x³ ],

满足 ``F(0)=0``、``F(∞)=1``，故 ``K → 1/r``（有限长度时 ``μ_R`` 取 ``d/r``）。

长距离展开与 H3
---------------
    K(r,z) = 1/r − 3/(2√π z) + r²/(2√π z³) + R_K(r,z),
    R_K = −erfc(z/r)/r + e^{−z²/r²}/(√π z) − r²e^{−z²/r²}/(2√π z³),

对同一表示的两个 flow 半径 ``r_i, r_j`` 取差时共同 ``1/z`` 项相消，剩余幂次
修正从 ``z⁻³`` 开始：

    μ_i − μ_j = d(1/r_i − 1/r_j) + (d/(2√π))(r_i²−r_j²)/z³
                + d[R_K(r_i,z) − R_K(r_j,z)],

积分后得到对数差

    D_i(z) = ln W(z,r_i) − ln W(z,r_ref)
           = −d·z·Δ(1/r)_i + B_i + k₃·Δr_i²/(2z²) + …,

自由模型 ``D_i^{H3}(z) = −d z Δ(1/r)_i + B_i + k₃ Δr_i²/(2z²)`` 只保留该形状、
放开修正幅度；只有在一圈共同幅度近似下才有 ``k₃ = d/(2√π)``，拟合中不施加该
关系、也不限制 ``k₃`` 符号。

局部斜率形式与有限段差分
------------------------
    d_eff,ij(z) = d − k₃·r_i r_j (r_i+r_j)/z³,
    H₃,ij[z_L,z_R] = (r_i²−r_j²)/Δu_ij · (z_L⁻²−z_R⁻²)/(2(z_R−z_L))。

有限长段必须用精确有限差分，不能用中点 ``z⁻³`` 代替。

参数计数
--------
``N_f`` 个 flow 取定一个参考后只有 ``N_f−1`` 组独立对数差，每组一个截距，
外加共同 ``d`` 与 ``k₃``，共 ``N_f+1`` 个参数。两 flow 三长度是 3 数据 / 3 参数
（零自由度，不能作质量检验）。仅依赖 ``z`` 的项 ``c_flow·t_f`` 与 ``d`` 项在单一
两 flow 对上严格共线，不能额外自由拟合来分离。

文献
----
推导见 ``refer/donghx/_codex/Markdown/linear_devergence.md`` §17.1–17.6
（树级流化传播子 → 位置空间 ``D_zz`` → ``F(x)`` 闭式 → H3 形状）。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erf, erfc

from ._const import CF, CA, pi


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


def flow_radius(tau, a_fm, hbarc_gev_fm=0.1973269804):
    """``r_f = √(8τ)·a``，返回 fm。"""
    tau = _positive("tau", tau)
    a_fm = _positive("a_fm", a_fm)
    hbarc_gev_fm = _positive("hbarc_gev_fm", hbarc_gev_fm)
    return math.sqrt(8.0 * tau) * a_fm


def one_loop_amplitude(alpha_s, casimir=CF):
    """``d^{(1)} = α_s C_R/√π``（``C_R = C_F`` 基础表示、``C_A`` 伴随表示）。"""
    alpha_s = _positive("alpha_s", alpha_s)
    casimir = _positive("casimir", casimir)
    return alpha_s * casimir / math.sqrt(pi)


def _kernel_small(x):
    """``F(x)`` 的 ``x→0`` 稳定形式。

    由定义式做 Taylor 展开（``erf`` 与分子抵消到 ``x`` 阶）：

        F(x) = x/√π · (3/4 − x²/12 + x⁴/80 − x⁶/560 + x⁸/4320 + O(x¹⁰))

    五阶截断（到 ``x⁸``）相对误差：``x=0.1`` 时 ~4e-15、``x=0.2`` 时 ~4e-12、
    ``x=0.5`` 时 ~4e-8；直接求值在 ``x=0.1`` 时 ~4e-15、``x=1e-3`` 时 ~1e-10。
    故分界取 0.1：两侧都在双精度级别。直接求值在 ``x ≲ 1e-3`` 会抵消掉有效位。
    """
    x2 = x * x
    return x / math.sqrt(pi) * (
        0.75 + x2 * (-1.0 / 12.0 + x2 * (
            1.0 / 80.0 + x2 * (-1.0 / 560.0 + x2 * (1.0 / 4320.0)))))


def line_mass_kernel(x):
    """``F(x)``：``x = z/r``。返回与输入同形的数组（标量输入返回 float）。"""
    values = np.asarray(x, dtype=float)
    scalar = values.ndim == 0
    flat = np.atleast_1d(values).ravel()
    if not np.isfinite(flat).all():
        raise ValueError("x 必须是有限实数")
    out = np.empty_like(flat)
    # 直接求值在 x ≲ 3e-3 时相对误差超过 1e-11，故分界放在 0.1
    small = np.abs(flat) < 0.1
    if np.any(small):
        out[small] = [_kernel_small(v) for v in flat[small]]
    if np.any(~small):
        big = flat[~small]
        g = -np.expm1(-big * big)
        out[~small] = erf(big) + (g * (1.0 - 2.0 * big * big) - big * big) / (
            2.0 * math.sqrt(pi) * big ** 3)
    result = out.reshape(values.shape)
    return float(result) if scalar else result


def line_mass_kernel_remainder(x):
    """``R_K``：``K − 1/r + 3/(2√π z) − r²/(2√π z³)`` 的无量纲形式。

    返回 ``r·R_K = −erfc(x) + e^{−x²}/(√π x) − e^{−x²}/(2√π x³)``，
    即 ``K(r,z) = (1/r)[F(x)]`` 与本函数的关系为
    ``r·R_K(x) = F(x) − 1 + 3/(2√π x) − 1/(2√π x³)``。
    """
    values = np.asarray(x, dtype=float)
    scalar = values.ndim == 0
    flat = np.atleast_1d(values).ravel()
    if not np.isfinite(flat).all() or np.any(flat <= 0.0):
        raise ValueError("x 必须是有限正数")
    out = (
        line_mass_kernel(flat) - 1.0
        + 3.0 / (2.0 * math.sqrt(pi) * flat)
        - 1.0 / (2.0 * math.sqrt(pi) * flat ** 3))
    result = out.reshape(values.shape)
    return float(result) if scalar else result


def line_mass(z_fm, r_fm, alpha_s, casimir=CF, hbarc_gev_fm=0.1973269804):
    """``μ_R(z,r) = d^{(1)} F(z/r)/r``，返回 GeV（``z, r`` 以 fm 输入）。"""
    z = np.asarray(z_fm, dtype=float)
    r_fm = _positive("r_fm", r_fm)
    hbarc_gev_fm = _positive("hbarc_gev_fm", hbarc_gev_fm)
    if not np.isfinite(z).all() or np.any(z < 0.0):
        raise ValueError("z_fm 必须是有限非负数")
    d1 = one_loop_amplitude(alpha_s, casimir)
    kernel = line_mass_kernel(np.abs(z) / r_fm)
    return d1 * kernel / (r_fm / hbarc_gev_fm)


def effective_slope(z_fm, r_i_fm, r_j_fm, d, k3):
    """``d_eff,ij(z) = d − k₃ r_i r_j (r_i+r_j)/z³``。"""
    z = np.asarray(z_fm, dtype=float)
    if not np.isfinite(z).all() or np.any(z <= 0.0):
        raise ValueError("z_fm 必须是有限正数")
    r_i = _positive("r_i_fm", r_i_fm)
    r_j = _positive("r_j_fm", r_j_fm)
    d = float(d)
    k3 = float(k3)
    return d - k3 * r_i * r_j * (r_i + r_j) / z ** 3


def finite_segment_kernel(z_left_fm, z_right_fm, r_i_fm, r_j_fm):
    """``H₃,ij[z_L,z_R]``：有限段差分的 ``z⁻³`` 系数。"""
    z_left = _positive("z_left_fm", z_left_fm)
    z_right = _positive("z_right_fm", z_right_fm)
    if z_right <= z_left:
        raise ValueError("z_right_fm 必须大于 z_left_fm")
    r_i = _positive("r_i_fm", r_i_fm)
    r_j = _positive("r_j_fm", r_j_fm)
    if abs(r_i - r_j) <= 0.0:
        raise ValueError("两个 flow 半径不得相同")
    delta_u = 1.0 / r_i - 1.0 / r_j
    return (r_i ** 2 - r_j ** 2) / delta_u * (
        z_left ** -2 - z_right ** -2) / (2.0 * (z_right - z_left))


def h3_log_difference(z_fm, r_fm, d, k3, reference=0, intercepts=None):
    """H3 自由模型的对数差 ``D_i(z)``。

    Args:
        z_fm: 长度数组（fm）。
        r_fm: ``N_f`` 个 flow 半径（fm）。
        d: 共同线性系数。
        k3: 共同 ``z⁻²`` 修正幅度（不限制符号）。
        reference: 作为差分基准的 flow 索引。
        intercepts: ``N_f`` 个截距；``reference`` 项的截距被差掉，可为任意值。
    Returns:
        ``(N_f−1, len(z))`` 数组，第 ``k`` 行对应扣掉 ``reference`` 后的第 ``k`` 个
        （跳过 ``reference`` 本身）。
    """
    z = np.asarray(z_fm, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.isfinite(z).all() or np.any(z <= 0.0):
        raise ValueError("z_fm 必须是非空、有限、严格正的数组")
    radii = np.asarray(r_fm, dtype=float)
    if radii.ndim != 1 or radii.size < 2:
        raise ValueError("r_fm 至少需要两个 flow 半径")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("r_fm 必须是有限正数")
    reference = int(reference)
    if not 0 <= reference < radii.size:
        raise ValueError("reference 必须是有效的 flow 索引")
    d = float(d)
    k3 = float(k3)
    if intercepts is None:
        offsets = np.zeros(radii.size)
    else:
        offsets = np.asarray(intercepts, dtype=float)
        if offsets.shape != radii.shape:
            raise ValueError("intercepts 必须与 r_fm 同形")
        if not np.isfinite(offsets).all():
            raise ValueError("intercepts 必须全部有限")

    rows = []
    for index in range(radii.size):
        if index == reference:
            continue
        delta = 1.0 / radii[index] - 1.0 / radii[reference]
        delta_r2 = radii[index] ** 2 - radii[reference] ** 2
        rows.append(
            -d * z * delta
            + (offsets[index] - offsets[reference])
            + k3 * delta_r2 / (2.0 * z ** 2))
    return np.vstack(rows)


__all__ = [
    "flow_radius", "one_loop_amplitude", "line_mass_kernel",
    "line_mass_kernel_remainder", "line_mass", "effective_slope",
    "finite_segment_kernel", "h3_log_difference",
]
