"""
一圈梯度流 OPE -> MS-bar 准算符转换。

本模块对应 ``refer/donghx/Gradient_Flow_GluonPDF/gluon_flow_to_quasi.py``。
它只做有限流 bare OPE 的端点小流时间转换，不做 LaMET/PDF 匹配、软因子、
快度演化或连续极限外推。

对非零分离 ``z>0``，Wilson 线方向平行/垂直的端点系数采用

    c_parallel = 1,
    c_perp = 1 + alpha_s C_A/(4 pi)
              log(2 mu^2 t exp(gamma_E)),
    delta_m = -alpha_s C_A/(4 pi) sqrt(2 pi/t),

并按通道乘上

    Z_U(z) = exp(-delta_m |z|) / c_perp^2,
    Z_H(z) = exp(-delta_m |z|) / (c_parallel c_perp).

``z=0`` 时 Wilson 线退化，不能沿用非局域端点公式。局域 spin-2 转换来自
``skills/donghx/Markdown/gradient_flow.md:1688-1694``：

    C_loc = 1 + alpha_s C_A/(4 pi)
            [11/3 (log(2 mu^2 t) + gamma_E) + 7/3],
    R_loc,MS = (2 - C_loc) R_loc,flow.

默认对 ``z=0`` 输出 NaN；legacy 端点因子和局域 spin-2 转换都必须显式选择。
输入和输出轴均为
``(field_projection, channel, z_orientation, component, z, t)``。
"""
from __future__ import annotations

import math

import numpy as np


GEV_FM = 5.067730716
CA = 3.0
EULER_GAMMA = 0.5772156649015329
FLOW_TO_QUASI_SCHEMA = "gradient_flow_gluon_flow_to_quasi_v2"
FLOW_TO_QUASI_STATUS = "flow_to_MSbar_quasi_scheme_conversion_diagnostic"

Z0_POLICIES = ("nan", "reject", "spin2_local", "legacy_endpoint")
HELICITY_MODES = ("helicity_physical", "legacy_plus_combined")

_Z0_POLICY_ALIASES = {
    "nan": "nan",
    "reject": "reject",
    "spin2_local": "spin2_local",
    "legacy": "legacy_endpoint",
    "legacy_endpoint": "legacy_endpoint",
}
_HELICITY_MODE_ALIASES = {
    "helicity_physical": "helicity_physical",
    "physical": "helicity_physical",
    "physical_T_minus_S": "helicity_physical",
    "legacy_plus_combined": "legacy_plus_combined",
    "legacy_T_plus_S": "legacy_plus_combined",
}


def _finite_real(name, value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} 必须是有限实数")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数")
    return value


def coefficients(tau, a_fm, mu_gev, alpha_s, ca=CA):
    """返回 ``(c_parallel, c_perp, delta_m_GeV, t_GeV_minus2)``。"""
    tau = _finite_real("tau", tau)
    a_fm = _finite_real("a_fm", a_fm)
    mu_gev = _finite_real("mu_gev", mu_gev)
    alpha_s = _finite_real("alpha_s", alpha_s)
    ca = _finite_real("ca", ca)
    if tau <= 0.0 or a_fm <= 0.0 or mu_gev <= 0.0:
        raise ValueError("tau、a_fm、mu_gev 必须为正")
    if alpha_s < 0.0:
        raise ValueError("alpha_s 必须非负")
    t_gev = tau * (a_fm * GEV_FM) ** 2
    c_parallel = 1.0
    c_perp = 1.0 + alpha_s * ca / (4.0 * math.pi) * math.log(
        2.0 * mu_gev ** 2 * t_gev * math.exp(EULER_GAMMA)
    )
    delta_m = -alpha_s * ca / (4.0 * math.pi) * math.sqrt(
        2.0 * math.pi / t_gev
    )
    return c_parallel, c_perp, delta_m, t_gev


def local_spin2_coefficients(tau, a_fm, mu_gev, alpha_s, ca=CA):
    """返回局域 ``z=0`` 的 ``(C_loc, 2-C_loc, t_GeV_minus2)``。

    公式只来自 donghx 的局域 spin-2 conversion。它在当前接口中只应用于
    unpolarized 的物理组合 ``Mtiti-Mijij``；其它局部分量保持不可判定。
    """
    _, _, _, t_gev = coefficients(
        tau, a_fm, mu_gev, alpha_s, ca
    )
    alpha_s = _finite_real("alpha_s", alpha_s)
    ca = _finite_real("ca", ca)
    mu_gev = _finite_real("mu_gev", mu_gev)
    log_term = math.log(
        2.0 * mu_gev ** 2 * t_gev
    ) + EULER_GAMMA
    c_loc = 1.0 + alpha_s * ca / (4.0 * math.pi) * (
        11.0 / 3.0 * log_term + 7.0 / 3.0
    )
    return c_loc, 2.0 - c_loc, t_gev


def _normalize_choice(name, value, aliases):
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串")
    try:
        return aliases[value]
    except KeyError as exc:
        valid = tuple(dict.fromkeys(aliases.values()))
        raise ValueError(f"{name} 必须来自 {valid!r}") from exc


def _broadcast_window_mask(window_mask, shape):
    mask = np.asarray(window_mask)
    if mask.shape == (shape[4],):
        mask = mask.reshape((1, 1, 1, 1, shape[4], 1))
    elif mask.shape != shape:
        raise ValueError(
            "window_mask 必须具有形状 (z,) 或完整 OPE 六轴形状，"
            f"收到 {mask.shape!r}"
        )
    if mask.dtype != np.dtype(bool):
        raise ValueError("window_mask 必须是布尔数组")
    return mask


def match_one(arr, tau, a_fm, mu_gev, alpha_s, ca=CA, *,
              z0_policy="nan", helicity_mode="helicity_physical",
              window_mask=None, kappa_F=None,
              absolute_normalization_validated=False,
              normalization_provenance=None,
              physical_acceptance=False,
              legacy_z0=False, legacy_plus_combined=False):
    """对一个 schema-v2 OPE 数组施加显式 policy 的一圈转换。

    Args:
        z0_policy: ``"nan"``（默认）对 z=0 输出 NaN；``"reject"`` 拒绝
            含 z=0 的输入；``"spin2_local"`` 对 unpolarized 的物理局域
            组合使用 spin-2 转换；``"legacy_endpoint"`` 仅用于显式复现旧
            的非局域端点因子。
        helicity_mode: ``"helicity_physical"`` 生成 ``Mtiti-Mijij``；
            ``"legacy_plus_combined"`` 显式生成旧的 ``Mtiti+Mijij``。
        window_mask: 可选布尔 ``(z,)`` 或完整六轴 mask；False 位置输出 NaN。
        kappa_F: 可选有限正数。提供后按 ``kappa_F**2`` 作用于 bilocal。
        absolute_normalization_validated: 独立验证 kappa_F 后才可设为 True。
        normalization_provenance: 绝对归一化验证来源的非空说明。
        physical_acceptance: 仅当已验证 kappa_F、全 True 窗口、physical
            helicity 且未使用 legacy z=0 时为 True；不满足时抛错。
        legacy_z0: ``z0_policy="legacy_endpoint"`` 的显式布尔别名。
        legacy_plus_combined: ``helicity_mode="legacy_plus_combined"`` 的
            显式布尔别名。

    Returns:
        ``(matched, metadata)``。metadata 将 scheme conversion、window
        validity 和 physical acceptance 分开报告。
    """
    z0_policy = _normalize_choice(
        "z0_policy", z0_policy, _Z0_POLICY_ALIASES
    )
    helicity_mode = _normalize_choice(
        "helicity_mode", helicity_mode, _HELICITY_MODE_ALIASES
    )
    if not isinstance(legacy_z0, (bool, np.bool_)):
        raise ValueError("legacy_z0 必须是布尔值")
    if not isinstance(legacy_plus_combined, (bool, np.bool_)):
        raise ValueError("legacy_plus_combined 必须是布尔值")
    legacy_z0 = bool(legacy_z0)
    legacy_plus_combined = bool(legacy_plus_combined)
    if legacy_z0:
        if z0_policy not in ("nan", "legacy_endpoint"):
            raise ValueError("legacy_z0 与显式 z0_policy 冲突")
        z0_policy = "legacy_endpoint"
    if legacy_plus_combined:
        if helicity_mode not in (
                "helicity_physical", "legacy_plus_combined"):
            raise ValueError(
                "legacy_plus_combined 与显式 helicity_mode 冲突")
        helicity_mode = "legacy_plus_combined"
    if not isinstance(physical_acceptance, (bool, np.bool_)):
        raise ValueError("physical_acceptance 必须是布尔值")
    if not isinstance(
            absolute_normalization_validated, (bool, np.bool_)):
        raise ValueError("absolute_normalization_validated 必须是布尔值")
    physical_acceptance = bool(physical_acceptance)
    absolute_normalization_validated = bool(
        absolute_normalization_validated)

    array = np.asarray(arr, dtype=np.complex128)
    if array.ndim != 6:
        raise ValueError(
            "OPE 必须具有 (projection,channel,orientation,component,z,t) 六轴"
        )
    if array.shape[1] != 2 or array.shape[3] < 3:
        raise ValueError(
            "OPE 必须包含两个 channel 及 combined/Mtiti/Mijij 三个组件"
        )
    if any(int(length) <= 0 for length in array.shape):
        raise ValueError("OPE 各轴长度必须为正")
    if not np.isfinite(array).all():
        raise ValueError("OPE 必须只含有限复数")
    if z0_policy == "reject":
        raise ValueError("z0_policy='reject' 拒绝含 z=0 的 OPE 输入")
    if kappa_F is not None:
        kappa_F = _finite_real("kappa_F", kappa_F)
        if kappa_F <= 0.0:
            raise ValueError("kappa_F 必须为正")
    mask = (
        None if window_mask is None
        else _broadcast_window_mask(window_mask, array.shape)
    )
    window_valid = bool(mask is not None and np.all(mask))
    if physical_acceptance:
        if kappa_F is None:
            raise ValueError(
                "physical_acceptance=True 要求显式 kappa_F")
        if mask is None:
            raise ValueError(
                "physical_acceptance=True 要求显式 window_mask")
        if not window_valid:
            raise ValueError(
                "physical_acceptance=True 要求 window mask 全部为 True")
        if helicity_mode != "helicity_physical":
            raise ValueError(
                "legacy_plus_combined 不能声明 physical_acceptance")
        if z0_policy == "legacy_endpoint":
            raise ValueError(
                "legacy z=0 endpoint factor 不能声明 physical_acceptance")
        if not absolute_normalization_validated:
            raise ValueError(
                "physical_acceptance=True 要求独立验证 "
                "absolute_normalization_validated=True")
        if (not isinstance(normalization_provenance, str)
                or not normalization_provenance.strip()):
            raise ValueError(
                "physical_acceptance=True 要求非空 normalization_provenance")

    c_parallel, c_perp, delta_m, t_gev = coefficients(
        tau, a_fm, mu_gev, alpha_s, ca
    )
    c_loc, local_factor, _ = local_spin2_coefficients(
        tau, a_fm, mu_gev, alpha_s, ca
    )
    z_gev = (
        np.arange(array.shape[4], dtype=float) * float(a_fm) * GEV_FM
    )
    line = np.exp(-delta_m * z_gev)[None, None, None, None, :, None]
    factor_unpolarized = 1.0 / (c_perp * c_perp)
    factor_helicity = 1.0 / (c_parallel * c_perp)
    factors = np.asarray(
        [factor_unpolarized, factor_helicity], dtype=np.float64
    )

    out = array.copy()
    if kappa_F is not None:
        out *= kappa_F ** 2
    out *= factors[None, :, None, None, None, None]
    out *= line
    out[:, 0, :, 0, :, :] = (
        out[:, 0, :, 1, :, :] - out[:, 0, :, 2, :, :]
    )
    if helicity_mode == "helicity_physical":
        out[:, 1, :, 0, :, :] = (
            out[:, 1, :, 1, :, :] - out[:, 1, :, 2, :, :]
        )
    else:
        out[:, 1, :, 0, :, :] = (
            out[:, 1, :, 1, :, :] + out[:, 1, :, 2, :, :]
        )

    if z0_policy == "nan":
        out[..., 0, :] = np.nan
        z0_status = "nan_fail_closed"
    elif z0_policy == "spin2_local":
        raw_local = (
            array[:, 0, :, 1, 0, :] - array[:, 0, :, 2, 0, :]
        )
        out[..., 0, :] = np.nan
        out[:, 0, :, 0, 0, :] = local_factor * raw_local
        if kappa_F is not None:
            out[:, 0, :, 0, 0, :] *= kappa_F ** 2
        z0_status = "spin2_local_unpolarized_combined"
    else:
        z0_status = "legacy_endpoint_factor_explicit_opt_in"

    if mask is not None:
        out[~np.broadcast_to(mask, array.shape)] = np.nan

    helicity_status = (
        "physical_T_minus_S"
        if helicity_mode == "helicity_physical"
        else "legacy_plus_combined_diagnostic"
    )
    scheme_conversion = (
        "one_loop_endpoint_applied_with_kappa_F"
        if kappa_F is not None
        else "one_loop_endpoint_applied_without_kappa_F"
    )
    metadata = {
        "schema": FLOW_TO_QUASI_SCHEMA,
        "status": FLOW_TO_QUASI_STATUS,
        "scheme_conversion": scheme_conversion,
        "scheme_conversion_details": {
            "order": "one_loop",
            "kappa_F": None if kappa_F is None else float(kappa_F),
            "kappa_F_provided": kappa_F is not None,
            "absolute_normalization_validated": (
                absolute_normalization_validated),
            "normalization_provenance": normalization_provenance,
            "absolute_field_normalization_complete": bool(
                kappa_F is not None
                and absolute_normalization_validated
                and isinstance(normalization_provenance, str)
                and normalization_provenance.strip()),
            "z0_policy": z0_policy,
            "helicity_mode": helicity_mode,
            "light_cone_matching": "not_applied",
        },
        "window_valid": window_valid,
        "window": {
            "provided": mask is not None,
            "all_valid": window_valid,
            "shape": None if mask is None else list(np.shape(window_mask)),
            "invalid_count": (
                0 if mask is None else int(np.count_nonzero(~mask))
            ),
        },
        "physical_acceptance": physical_acceptance,
        "acceptance": {
            "accepted": physical_acceptance,
            "requires_kappa_F": True,
            "requires_validated_absolute_normalization": True,
            "requires_normalization_provenance": True,
            "requires_all_true_window_mask": True,
            "requires_helicity_physical": True,
            "legacy_z0_accepted": False,
        },
        "helicity_combination": helicity_mode,
        "helicity_status": helicity_status,
        "helicity_physical": helicity_mode == "helicity_physical",
        "legacy_plus_combined": (
            helicity_mode == "legacy_plus_combined"
        ),
        "combined_definition": {
            "unpolarized": "Mtiti - Mijij = T_U - S_U",
            "helicity": (
                "Mtiti - Mijij = T_H - S_H"
                if helicity_mode == "helicity_physical"
                else "Mtiti + Mijij = T_H + S_H (legacy diagnostic)"
            ),
        },
        "z0_status": z0_status,
        "legacy_z0_endpoint": z0_policy == "legacy_endpoint",
        "local_spin2_conversion": z0_policy == "spin2_local",
        "z0": {
            "policy": z0_policy,
            "status": z0_status,
            "local_spin2_only_for": "unpolarized_combined",
        },
        "c_parallel_perp": float(c_parallel),
        "c_perp_perp": float(c_perp),
        "c_local_spin2": float(c_loc),
        "local_spin2_factor": float(local_factor),
        "delta_m_GeV": float(delta_m),
        "t_GeV_minus2": float(t_gev),
        "line_factor": "exp(-delta_m*|z|)",
        "component_factors_unpolarized": float(factor_unpolarized),
        "component_factors_helicity": float(factor_helicity),
        "component_factors_by_channel": factors.tolist(),
        "light_cone_matching": "not_applied",
    }
    return out, metadata


gradient_flow_to_quasi_coefficients = coefficients
match_gradient_flow_ope_to_quasi = match_one


__all__ = [
    "GEV_FM", "CA", "EULER_GAMMA", "FLOW_TO_QUASI_SCHEMA",
    "FLOW_TO_QUASI_STATUS", "Z0_POLICIES", "HELICITY_MODES",
    "coefficients", "gradient_flow_to_quasi_coefficients",
    "local_spin2_coefficients",
    "match_one", "match_gradient_flow_ope_to_quasi",
]
