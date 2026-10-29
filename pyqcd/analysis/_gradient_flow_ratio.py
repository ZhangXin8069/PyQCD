"""
参考 schema-v2 梯度流胶子 disconnected C3/C2 统计。

输入 OPE 的轴为
``(configuration, channel, z_orientation, component, z, t)``；
输入两点函数的轴为
``(channel, direction, configuration, tsep, t_source, momentum)``。

统计层保留完整复数。特别是 helicity 通道使用完整 `pol35` 两点函数
构造 covariance；只有在显示或物理相位投影阶段才取
``Re[-i R_H] = Im(R_H)``。

有限横距通过 ``calculate_gradient_flow_tmd_ratios`` 暴露显式 ``b`` 轴；
该包装层复用同一复数 ratio-v2 核心，不引入第二套估计量。
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..operator._gradient_flow_gluon_ope import (
    CHANNELS, COMPONENTS, ORIENTATIONS,
)


POLARIZATION_FOR_CHANNEL = {
    "unpolarized": "nopol",
    "helicity": "pol35",
}
RATIO_SCHEMA = "gradient_flow_gluon_ratio_v2"
TMD_RATIO_SCHEMA = "gradient_flow_gluon_tmd_ratio_v2"
TMD_RATIO_2D_SCHEMA = "gradient_flow_gluon_tmd_ratio_2d_v3"


def _as_complex128(value, name):
    array = np.asarray(value, dtype=np.complex128)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含非有限值")
    return array


def _input_complex_precision(*values):
    """Return the weakest complex precision among analysis inputs."""
    dtypes = []
    for value in values:
        try:
            dtypes.append(np.asarray(value).dtype)
        except (TypeError, ValueError):
            dtypes.append(np.dtype(np.complex128))
    return (
        "complex64" if np.dtype(np.complex64) in dtypes
        else "complex128")


def _orientation_identity_matches(target, left, right, sign, precision):
    """Compare linear orientation identities with cancellation-aware scale."""
    target = np.asarray(target)
    left = np.asarray(left)
    right = np.asarray(right)
    expected = left + sign * right
    if precision == "complex64":
        # Ratio construction performs several complex divisions before the
        # orientation identity; 3e-5 of the operand scale covers that
        # accumulation without using the possibly near-zero difference.
        rtol, atol = 3e-5, 2e-6
    elif precision == "complex128":
        rtol, atol = 5e-12, 1e-10
    else:
        raise ValueError("input_precision 必须是 complex64/complex128")
    tolerance = atol + rtol * (np.abs(left) + np.abs(right))
    matches = np.abs(target - expected) <= tolerance
    both_nan = np.isnan(target) & np.isnan(expected)
    return matches | both_nan


def _validate_tseps(tseps, nt):
    values = list(tseps)
    if not values:
        raise ValueError("tseps 不能为空")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in values):
        raise ValueError("tseps 必须是整数")
    values = [int(value) for value in values]
    if len(set(values)) != len(values):
        raise ValueError("tseps 不得重复")
    if min(values) < 0 or max(values) >= nt:
        raise ValueError("tseps 必须满足 0 <= tsep < Nt")
    return values


def covariance_ratio_estimator(operator, correlator, do_jackknife=True,
                               denominator_correlator=None,
                               return_jackknife=False,
                               denominator_mode="complex"):
    """计算一个插入量的 disconnected ratio 和 delete-one jackknife。

    Args:
        operator: ``(Nconf, Nobservable, Nsource)``，完整复数 OPE。
        correlator: ``(Nconf, Nsource, Nmomentum)``，C3 covariance 所用
            投影；unpolarized 使用 nopol，helicity 使用 pol35。
        do_jackknife: 是否生成 Nconf 个 delete-one 样本。
        denominator_correlator: ratio 分母的 C2；省略时使用 correlator。
        denominator_mode: ``"complex"``（默认，ratio-v3 约定）或
            ``"real"``（`ensemble_covariance_ratio`/donghx 约定，取
            denominator C2 的实部）。disconnected 分子始终使用复数 C2。

    Returns:
        ``(c3_central, c2_central, ratio_jk_mean,
        ratio_jk_bias_corrected, error_real, error_imag)``。
        central ratio 不做实部/虚部截断。
    """
    if denominator_mode not in ("complex", "real"):
        raise ValueError(
            f"denominator_mode 必须是 'complex' 或 'real'，收到 {denominator_mode!r}")
    op = _as_complex128(operator, "operator")
    c2 = _as_complex128(correlator, "correlator")
    denominator = (
        c2 if denominator_correlator is None
        else _as_complex128(denominator_correlator, "denominator_correlator")
    )
    if denominator_mode == "real":
        denominator = np.real(denominator)
    if not isinstance(return_jackknife, (bool, np.bool_)):
        raise ValueError("return_jackknife 必须是布尔值")
    return_jackknife = bool(return_jackknife)
    if return_jackknife and not do_jackknife:
        raise ValueError(
            "return_jackknife=True 要求 do_jackknife=True")
    if op.ndim != 3 or c2.ndim != 3:
        raise ValueError(
            f"需要 operator=(conf,observable,source)、C2=(conf,source,momentum)，"
            f"收到 {op.shape} 与 {c2.shape}")
    if op.shape[0] != c2.shape[0] or op.shape[2] != c2.shape[1]:
        raise ValueError(f"operator/C2 轴不匹配: {op.shape}, {c2.shape}")
    if denominator.shape != c2.shape:
        raise ValueError(
            f"denominator C2 形状必须与 correlator 一致: "
            f"{denominator.shape} != {c2.shape}")
    nconf = int(op.shape[0])
    if nconf < 2:
        raise ValueError("delete-one ratio 至少需要两个组态")
    nsource = int(op.shape[2])

    sum_o = np.sum(op, axis=0)
    sum_c = np.sum(c2, axis=0)
    sum_denominator = np.sum(denominator, axis=0)
    oc_mean_source = np.einsum("nos,nsp->op", op, c2, optimize=True) / nsource
    o_times_c_mean_source = (
        np.einsum("os,sp->op", sum_o, sum_c, optimize=True) / nsource
    )
    c3_central = (
        oc_mean_source / nconf
        - o_times_c_mean_source / (nconf * nconf)
    )
    c2_central = np.mean(sum_denominator, axis=0) / nconf
    if (np.any(~np.isfinite(c2_central))
            or np.any(c2_central == 0.0)):
        raise FloatingPointError(
            "ratio denominator C2 contains zero/non-finite value")

    shape = c3_central.shape
    nan_complex = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    nan_real = np.full(shape, np.nan, dtype=np.float64)
    if not do_jackknife:
        result = (
            c3_central, c2_central, nan_complex,
            nan_complex.copy(), nan_real, nan_real.copy(),
        )
        return result

    nminus = nconf - 1
    own_oc_mean_source = np.einsum(
        "nos,nsp->nop", op, c2, optimize=True
    ) / nsource
    own_o_times_total_c = np.einsum(
        "nos,sp->nop", op, sum_c, optimize=True
    ) / nsource
    total_o_times_own_c = np.einsum(
        "os,nsp->nop", sum_o, c2, optimize=True
    ) / nsource
    c3_jackknife = (
        (oc_mean_source[None, ...] - own_oc_mean_source) / nminus
        - (
            o_times_c_mean_source[None, ...]
            - own_o_times_total_c
            - total_o_times_own_c
            + own_oc_mean_source
        ) / (nminus * nminus)
    )
    c2_jackknife = (
        np.mean(sum_denominator, axis=0)[None, :]
        - np.mean(denominator, axis=1)
    ) / nminus
    if (np.any(~np.isfinite(c2_jackknife))
            or np.any(c2_jackknife == 0.0)):
        raise FloatingPointError(
            "delete-one ratio denominator C2 contains zero/non-finite value")
    ratio_jackknife = (
        c3_jackknife / c2_jackknife[:, None, :]
    )
    jk_mean = np.mean(ratio_jackknife, axis=0)
    jk_bias_corrected = (
        nconf * (c3_central / c2_central[None, :])
        - nminus * jk_mean
    )
    prefactor = nminus / nconf
    error_real = np.sqrt(
        prefactor * np.sum(
            (ratio_jackknife.real - jk_mean.real) ** 2, axis=0
        )
    )
    error_imag = np.sqrt(
        prefactor * np.sum(
            (ratio_jackknife.imag - jk_mean.imag) ** 2, axis=0
        )
    )
    result = (
        c3_central, c2_central, jk_mean, jk_bias_corrected,
        error_real, error_imag,
    )
    if return_jackknife:
        return result + (ratio_jackknife,)
    return result


def _component_labels_for_count(ncomp, component_labels):
    if component_labels is None:
        if ncomp == len(COMPONENTS):
            labels = COMPONENTS
        elif ncomp == 3:
            labels = COMPONENTS[:3]
        else:
            labels = tuple(f"component_{index}" for index in range(ncomp))
    else:
        labels = tuple(component_labels)
    if len(labels) != ncomp or len(set(labels)) != len(labels):
        raise ValueError("component_labels 必须与 component 轴一一对应且不重复")
    return labels


def calculate_gradient_flow_ratios(ope, twopt, tseps, do_jackknife=True,
                                   component_labels=None,
                                   store_jackknife_samples=False,
                                   denominator_mode="complex"):
    """按参考 ratio-v2 轴构造所有 channel/orientation/component ratio。

    Args:
        ope: ``(conf,2,4,ncomp,nz,Nt)`` complex OPE。
        twopt: ``(2,ndirection,conf,ntsep,Nt,nmomentum)`` complex C2；
            channel 0 必须为 nopol，channel 1 为 pol35。
        tseps: 与 twopt 的 ``ntsep`` 轴对应的 separation 列表。

    Returns:
        字典，包含 `c3_mean`、`c2_mean`、`ratio`、jackknife 统计和
        `valid_insertion_mask`。ratio 轴为
        ``(channel,orientation,component,direction,z,tsep,insertion,momentum)``。
    """
    if not isinstance(do_jackknife, (bool, np.bool_)):
        raise ValueError("do_jackknife 必须是布尔值")
    do_jackknife = bool(do_jackknife)
    if not isinstance(store_jackknife_samples, (bool, np.bool_)):
        raise ValueError("store_jackknife_samples 必须是布尔值")
    store_jackknife_samples = bool(store_jackknife_samples)
    if store_jackknife_samples and not do_jackknife:
        raise ValueError(
            "store_jackknife_samples=True 要求 do_jackknife=True")
    input_precision = _input_complex_precision(ope, twopt)
    ope = _as_complex128(ope, "ope")
    twopt = _as_complex128(twopt, "twopt")
    if ope.ndim != 6:
        raise ValueError("ope 必须具有 (conf,channel,orientation,component,z,t) 六轴")
    if twopt.ndim != 6:
        raise ValueError(
            "twopt 必须具有 (channel,direction,conf,tsep,source,momentum) 六轴")
    nconf, nch, norient, ncomp, nz, nt = ope.shape
    nch2, ndir, nconf2, ndt, nt2, nmom = twopt.shape
    if (nch, norient) != (len(CHANNELS), len(ORIENTATIONS)):
        raise ValueError(
            f"ope channel/orientation 轴必须为 (2,4)，收到 {ope.shape}")
    if (nch2, nconf2, nt2) != (nch, nconf, nt):
        raise ValueError(f"输入 OPE/C2 轴不匹配: {ope.shape}, {twopt.shape}")
    if ncomp <= 0 or nz <= 0 or nt <= 0:
        raise ValueError("ope 的 component/z/time 轴必须非空")
    component_labels = _component_labels_for_count(
        ncomp, component_labels
    )
    tseps = _validate_tseps(tseps, nt)
    if len(tseps) != ndt:
        raise ValueError(
            f"tseps 长度 {len(tseps)} 与 twopt tsep 轴 {ndt} 不一致")
    if nconf < 2 and do_jackknife:
        raise ValueError("jackknife 至少需要两个组态")

    max_tsep = max(tseps)
    out_shape = (
        nch, norient, ncomp, ndir, nz, ndt, max_tsep + 1, nmom
    )
    nan_complex = np.full(out_shape, np.nan + 1j * np.nan, dtype=np.complex128)
    c3 = nan_complex.copy()
    ratio = nan_complex.copy()
    jk_mean = nan_complex.copy()
    jk_bias = nan_complex.copy()
    nan_real = np.full(out_shape, np.nan, dtype=np.float64)
    jk_err_real = nan_real.copy()
    jk_err_imag = nan_real.copy()
    jk_samples = (
        np.full(out_shape + (nconf,), np.nan + 1j * np.nan,
                dtype=np.complex128)
        if store_jackknife_samples else None)
    c2_mean = np.empty((nch, ndir, ndt, nmom), dtype=np.complex128)

    source = np.arange(nt)
    for channel in range(nch):
        for direction in range(ndir):
            c2_by_dt = twopt[channel, direction]
            denominator_by_dt = twopt[0, direction]
            c2_mean[channel, direction] = np.mean(
                denominator_by_dt, axis=(0, 2)
            )
            for insertion in range(max_tsep + 1):
                shifted = np.take(
                    ope[:, channel],
                    (source + insertion) % nt,
                    axis=-1,
                )
                flat = shifted.reshape(nconf, norient * ncomp * nz, nt)
                for dt_index, dt in enumerate(tseps):
                    if insertion > dt:
                        continue
                    result = covariance_ratio_estimator(
                        flat,
                        c2_by_dt[:, dt_index],
                        do_jackknife=do_jackknife,
                        denominator_correlator=denominator_by_dt[:, dt_index],
                        return_jackknife=store_jackknife_samples,
                        denominator_mode=denominator_mode,
                    )
                    if store_jackknife_samples:
                        (c3_one, c2_one, mean_one, bias_one, err_r, err_i,
                         samples_one) = result
                    else:
                        c3_one, c2_one, mean_one, bias_one, err_r, err_i = result
                    shape = (norient, ncomp, nz, nmom)
                    target = (
                        channel, slice(None), slice(None), direction,
                        slice(None), dt_index, insertion, slice(None),
                    )
                    c3[target] = c3_one.reshape(shape)
                    central_ratio = c3_one / c2_one[None, :]
                    ratio[target] = central_ratio.reshape(shape)
                    jk_mean[target] = mean_one.reshape(shape)
                    jk_bias[target] = bias_one.reshape(shape)
                    jk_err_real[target] = err_r.reshape(shape)
                    jk_err_imag[target] = err_i.reshape(shape)
                    if store_jackknife_samples:
                        # samples_one 是 (nconf, Nobs, Nmom)；直接 reshape 会把
                        # conf 轴与 observable 轴混在一起，导致下游从样本算出的
                        # “误差”实际上是不同 (z,orientation,component) 格点之间
                        # 的散布。必须先移轴到末尾。
                        jk_samples[target] = np.moveaxis(
                            samples_one, 0, -1).reshape(shape + (nconf,))

    valid_insertion = np.zeros((ndt, max_tsep + 1), dtype=bool)
    for index, dt in enumerate(tseps):
        valid_insertion[index, :dt + 1] = True
    result = {
        "c3_mean": c3,
        "c2_mean": c2_mean,
        "ratio": ratio,
        "ratio_jackknife_mean": jk_mean,
        "ratio_jackknife_bias_corrected": jk_bias,
        "ratio_jackknife_error_real": jk_err_real,
        "ratio_jackknife_error_imag": jk_err_imag,
        "valid_insertion_mask": valid_insertion,
        "schema": RATIO_SCHEMA,
        "channel_labels": tuple(CHANNELS),
        "z_orientation_labels": tuple(ORIENTATIONS),
        "component_labels": component_labels,
        "direction_labels": tuple(range(ndir)),
        "axes": (
            "channel", "z_orientation", "component", "direction",
            "z", "tsep", "insertion", "momentum_abs",
        ),
        "polarization_for_channel": dict(POLARIZATION_FOR_CHANNEL),
        "denominator_polarization": "nopol",
        "denominator_mode": denominator_mode,
        "do_jackknife": do_jackknife,
        "input_precision": input_precision,
    }
    if store_jackknife_samples:
        result["ratio_jackknife_samples"] = jk_samples
    return result


calculate_ratios = calculate_gradient_flow_ratios


def _reshape_ratio_z_b(array, nz, nb):
    """Restore a flattened ``(z,b)`` observable axis."""
    values = np.asarray(array)
    if values.ndim < 8 or values.shape[4] != nz * nb:
        raise ValueError(
            "内部 ratio 数组必须具有扁平 (z*b) observable 轴，"
            f"收到 {values.shape}")
    return values.reshape(
        values.shape[:4] + (nz, nb) + values.shape[5:])


def _reshape_ratio_z_bx_by(array, nz, nbx, nby):
    """Restore a flattened ``(z,b_x,b_y)`` observable axis."""
    values = np.asarray(array)
    if values.ndim < 8 or values.shape[4] != nz * nbx * nby:
        raise ValueError(
            "内部 ratio 数组必须具有扁平 (z*bx*by) observable 轴，"
            f"收到 {values.shape}")
    return values.reshape(
        values.shape[:4] + (nz, nbx, nby) + values.shape[5:])


def _validate_tmd_ratio_2d_metadata(
        b_x_values, b_y_values, nbx, nby, z_dir, b_x_dir, b_y_dir,
        transverse_order):
    """Validate explicit 2D displacement and staircase metadata."""
    values = []
    for name, requested, expected in (
            ("b_x_values", b_x_values, nbx),
            ("b_y_values", b_y_values, nby)):
        try:
            normalized = tuple(requested)
        except TypeError as exc:
            raise ValueError(f"{name} 必须是非空整数序列") from exc
        if len(normalized) != expected:
            raise ValueError(
                f"{name} 长度必须为 {expected}，收到 {len(normalized)}")
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer))
               for value in normalized):
            raise ValueError(f"{name} 必须是非布尔整数")
        normalized = tuple(int(value) for value in normalized)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} 不得重复")
        values.append(normalized)
    directions = []
    for name, value in (
            ("z_dir", z_dir), ("b_x_dir", b_x_dir), ("b_y_dir", b_y_dir)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)) or int(value) not in (0, 1, 2):
            raise ValueError(f"{name} 必须是空间方向 0,1,2")
        directions.append(int(value))
    if len(set(directions)) != 3:
        raise ValueError("z_dir、b_x_dir、b_y_dir 必须互不相同")
    try:
        transverse_order = tuple(transverse_order)
    except TypeError as exc:
        raise ValueError(
            "transverse_order 必须是 (0,1) 或 (1,0)") from exc
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in transverse_order):
        raise ValueError("transverse_order 必须含非布尔整数")
    transverse_order = tuple(int(value) for value in transverse_order)
    if transverse_order not in ((0, 1), (1, 0)):
        raise ValueError("transverse_order 必须是 (0,1) 或 (1,0)")
    return (
        values[0], values[1], directions[0], directions[1], directions[2],
        transverse_order,
    )


def calculate_gradient_flow_tmd_ratios(
        ope, twopt, tseps, b_values, do_jackknife=True,
        component_labels=None, store_jackknife_samples=False,
        denominator_mode="complex"):
    """Construct ratio-v2 statistics with explicit ``(z,b)`` observables.

    Args:
        ope: ``(conf,2,4,ncomp,nz,nb,Nt)`` complex TMD OPE.
        twopt: ``(2,ndirection,conf,ntsep,Nt,nmomentum)`` complex C2.
        tseps: separations corresponding to the ``twopt`` tsep axis.
        b_values: explicit transverse displacements in lattice-spacing units,
            one per ``b`` axis entry.

    The OPE ``(z,b)`` axes are flattened in C order for the shared
    observable estimator and restored in every output.  This preserves the
    schema-v2 complex-nopol ratio while making the transverse displacement an
    explicit result axis.
    """
    input_precision = _input_complex_precision(ope, twopt)
    array = _as_complex128(ope, "ope")
    if array.ndim != 7:
        raise ValueError(
            "TMD ope 必须具有 "
            "(conf,channel,orientation,component,z,b,t) 七轴，"
            f"收到 {array.shape}")
    nconf, nch, norient, ncomp, nz, nb, nt = array.shape
    if min(nconf, nch, norient, ncomp, nz, nb, nt) <= 0:
        raise ValueError("TMD ope 各轴长度必须为正")
    try:
        b_values = tuple(b_values)
    except TypeError as exc:
        raise ValueError("b_values 必须是非空整数序列") from exc
    if len(b_values) != nb:
        raise ValueError(
            f"b_values 长度必须等于 nb={nb}，收到 {len(b_values)}")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in b_values):
        raise ValueError("b_values 必须是非布尔整数")
    b_values = tuple(int(value) for value in b_values)
    if len(set(b_values)) != len(b_values):
        raise ValueError("b_values 不得重复")
    flattened = array.reshape(nconf, nch, norient, ncomp, nz * nb, nt)
    result = calculate_gradient_flow_ratios(
        flattened,
        twopt,
        tseps,
        do_jackknife=do_jackknife,
        component_labels=component_labels,
        store_jackknife_samples=store_jackknife_samples,
        denominator_mode=denominator_mode,
    )
    for key in (
            "c3_mean", "ratio", "ratio_jackknife_mean",
            "ratio_jackknife_bias_corrected",
            "ratio_jackknife_error_real",
            "ratio_jackknife_error_imag"):
        result[key] = _reshape_ratio_z_b(result[key], nz, nb)
    if store_jackknife_samples:
        result["ratio_jackknife_samples"] = _reshape_ratio_z_b(
            result["ratio_jackknife_samples"], nz, nb)
    result["schema"] = TMD_RATIO_SCHEMA
    result["axes"] = (
        "channel", "z_orientation", "component", "direction",
        "z", "b", "tsep", "insertion", "momentum_abs",
    )
    result["b_values"] = b_values
    result["b_unit"] = "lattice_spacing_a"
    result["observable_flatten_order"] = "z_major_b_minor_c_order"
    result["base_schema"] = RATIO_SCHEMA
    result["input_precision"] = input_precision
    return result


def calculate_gradient_flow_tmd_ratios_2d(
        ope, twopt, tseps, b_x_values, b_y_values, *,
        z_dir=2, b_x_dir=0, b_y_dir=1, transverse_order=(0, 1),
        do_jackknife=True, component_labels=None,
        store_jackknife_samples=False, denominator_mode="complex"):
    """Construct ratio-v3 statistics with explicit ``(z,b_x,b_y)`` axes."""
    input_precision = _input_complex_precision(ope, twopt)
    array = _as_complex128(ope, "ope")
    if array.ndim != 8:
        raise ValueError(
            "2D TMD ope 必须具有 "
            "(conf,channel,orientation,component,z,b_x,b_y,t) 八轴，"
            f"收到 {array.shape}")
    nconf, nch, norient, ncomp, nz, nbx, nby, nt = array.shape
    if min(nconf, nch, norient, ncomp, nz, nbx, nby, nt) <= 0:
        raise ValueError("2D TMD ope 各轴长度必须为正")
    b_x_values, b_y_values, z_dir, b_x_dir, b_y_dir, transverse_order = (
        _validate_tmd_ratio_2d_metadata(
            b_x_values, b_y_values, nbx, nby,
            z_dir, b_x_dir, b_y_dir, transverse_order)
    )
    flattened = array.reshape(
        nconf, nch, norient, ncomp, nz * nbx * nby, nt)
    result = calculate_gradient_flow_ratios(
        flattened,
        twopt,
        tseps,
        do_jackknife=do_jackknife,
        component_labels=component_labels,
        store_jackknife_samples=store_jackknife_samples,
        denominator_mode=denominator_mode,
    )
    for key in (
            "c3_mean", "ratio", "ratio_jackknife_mean",
            "ratio_jackknife_bias_corrected",
            "ratio_jackknife_error_real",
            "ratio_jackknife_error_imag"):
        result[key] = _reshape_ratio_z_bx_by(
            result[key], nz, nbx, nby)
    if store_jackknife_samples:
        result["ratio_jackknife_samples"] = _reshape_ratio_z_bx_by(
            result["ratio_jackknife_samples"], nz, nbx, nby)
    result["schema"] = TMD_RATIO_2D_SCHEMA
    result["axes"] = (
        "channel", "z_orientation", "component", "direction",
        "z", "b_x", "b_y", "tsep", "insertion", "momentum_abs",
    )
    result["b_x_values"] = b_x_values
    result["b_y_values"] = b_y_values
    result["b_x_dir"] = b_x_dir
    result["b_y_dir"] = b_y_dir
    result["z_dir"] = z_dir
    result["transverse_order"] = transverse_order
    result["b_unit"] = "lattice_spacing_a"
    result["observable_flatten_order"] = (
        "z_major_bx_middle_by_minor_c_order")
    result["base_schema"] = RATIO_SCHEMA
    result["input_precision"] = input_precision
    return result


def calculate_gradient_flow_tmd_ratios_from_configurations(
        ope_by_conf, twopt_by_conf, conf_ids, b_values, tseps,
        do_jackknife=True, component_labels=None,
        store_jackknife_samples=False, denominator_mode="complex"):
    """Stack source-resolved configuration inputs in an explicit shared order.

    Each OPE entry has shape
    ``(2,4,component,z,b,t)`` and each C2 entry has shape
    ``(2,direction,tsep,source,momentum)``.  The source axis must be retained;
    source-averaged ``(t,)`` inputs are rejected instead of being broadcast.
    """
    try:
        ids = tuple(conf_ids)
    except TypeError as exc:
        raise ValueError("conf_ids 必须是配置 ID 序列") from exc
    if not ids:
        raise ValueError("conf_ids 不得为空")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in ids):
        raise ValueError("conf_ids 必须是非布尔整数")
    ids = tuple(int(value) for value in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("conf_ids 不得重复")

    ope_samples = []
    twopt_samples = []
    ope_shape = None
    twopt_shape = None
    nt = None
    for conf_id in ids:
        if conf_id not in ope_by_conf:
            raise ValueError(f"conf_ids 中的配置缺少 OPE: {conf_id}")
        if conf_id not in twopt_by_conf:
            raise ValueError(f"conf_ids 中的配置缺少 C2: {conf_id}")
        ope = np.asarray(ope_by_conf[conf_id])
        twopt = np.asarray(twopt_by_conf[conf_id])
        if ope.ndim != 6:
            raise ValueError(
                "每组态 OPE 必须具有 (channel,orientation,component,z,b,t) "
                f"六轴，conf={conf_id} 收到 {ope.shape}")
        if twopt.ndim != 5:
            raise ValueError(
                "source-resolved C2 必须具有 "
                "(channel,direction,tsep,source,momentum) 五轴，"
                f"conf={conf_id} 收到 {twopt.shape}")
        conf_nt = int(ope.shape[-1])
        if conf_nt <= 0 or twopt.shape[3] != conf_nt:
            raise ValueError(
                "source-resolved C2 的 source 轴必须与 OPE 时间轴一致，"
                f"conf={conf_id} 收到 source={twopt.shape[3]}, t={conf_nt}")
        if nt is None:
            nt = conf_nt
            ope_shape = ope.shape
            twopt_shape = twopt.shape
        elif ope.shape != ope_shape or twopt.shape != twopt_shape:
            raise ValueError(
                "所有组态的 OPE/C2 shape 必须一致，"
                f"conf={conf_id} 收到 {ope.shape}/{twopt.shape}，"
                f"期望 {ope_shape}/{twopt_shape}")
        ope_samples.append(ope)
        twopt_samples.append(twopt)

    result = calculate_gradient_flow_tmd_ratios(
        np.stack(ope_samples),
        np.stack(twopt_samples, axis=2),
        tseps,
        b_values,
        do_jackknife=do_jackknife,
        component_labels=component_labels,
        store_jackknife_samples=store_jackknife_samples,
        denominator_mode=denominator_mode,
    )
    index_bytes = np.asarray(ids, dtype='<i8').tobytes(order='C')
    result["conf_ids"] = ids
    result["source_resolved_c2"] = True
    result["c2_source_length"] = nt
    result["input_index_schema"] = (
        "explicit_source_resolved_configuration_order")
    result["input_index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
    result["resampling_kind"] = (
        "delete_one_jackknife" if result["do_jackknife"] else "central_only")
    return result


def calculate_gradient_flow_tmd_ratios_2d_from_configurations(
        ope_by_conf, twopt_by_conf, conf_ids, b_x_values, b_y_values,
        tseps, *, z_dir=2, b_x_dir=0, b_y_dir=1,
        transverse_order=(0, 1), do_jackknife=True,
        component_labels=None, store_jackknife_samples=False,
        denominator_mode="complex"):
    """Stack source-resolved inputs for two-component transverse ratios."""
    try:
        ids = tuple(conf_ids)
    except TypeError as exc:
        raise ValueError("conf_ids 必须是配置 ID 序列") from exc
    if not ids:
        raise ValueError("conf_ids 不得为空")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in ids):
        raise ValueError("conf_ids 必须是非布尔整数")
    ids = tuple(int(value) for value in ids)
    if len(set(ids)) != len(ids):
        raise ValueError("conf_ids 不得重复")

    ope_samples = []
    twopt_samples = []
    ope_shape = None
    twopt_shape = None
    nt = None
    for conf_id in ids:
        if conf_id not in ope_by_conf:
            raise ValueError(f"conf_ids 中的配置缺少 OPE: {conf_id}")
        if conf_id not in twopt_by_conf:
            raise ValueError(f"conf_ids 中的配置缺少 C2: {conf_id}")
        ope = np.asarray(ope_by_conf[conf_id])
        twopt = np.asarray(twopt_by_conf[conf_id])
        if ope.ndim != 7:
            raise ValueError(
                "每组态 2D OPE 必须具有 "
                "(channel,orientation,component,z,b_x,b_y,t) 七轴，"
                f"conf={conf_id} 收到 {ope.shape}")
        if twopt.ndim != 5:
            raise ValueError(
                "source-resolved C2 必须具有 "
                "(channel,direction,tsep,source,momentum) 五轴，"
                f"conf={conf_id} 收到 {twopt.shape}")
        conf_nt = int(ope.shape[-1])
        if conf_nt <= 0 or twopt.shape[3] != conf_nt:
            raise ValueError(
                "source-resolved C2 的 source 轴必须与 OPE 时间轴一致，"
                f"conf={conf_id} 收到 source={twopt.shape[3]}, t={conf_nt}")
        if nt is None:
            nt = conf_nt
            ope_shape = ope.shape
            twopt_shape = twopt.shape
        elif ope.shape != ope_shape or twopt.shape != twopt_shape:
            raise ValueError(
                "所有组态的 2D OPE/C2 shape 必须一致，"
                f"conf={conf_id} 收到 {ope.shape}/{twopt.shape}，"
                f"期望 {ope_shape}/{twopt_shape}")
        ope_samples.append(ope)
        twopt_samples.append(twopt)

    result = calculate_gradient_flow_tmd_ratios_2d(
        np.stack(ope_samples),
        np.stack(twopt_samples, axis=2),
        tseps,
        b_x_values,
        b_y_values,
        z_dir=z_dir,
        b_x_dir=b_x_dir,
        b_y_dir=b_y_dir,
        transverse_order=transverse_order,
        do_jackknife=do_jackknife,
        component_labels=component_labels,
        store_jackknife_samples=store_jackknife_samples,
        denominator_mode=denominator_mode,
    )
    index_bytes = np.asarray(ids, dtype='<i8').tobytes(order='C')
    result["conf_ids"] = ids
    result["source_resolved_c2"] = True
    result["c2_source_length"] = nt
    result["input_index_schema"] = (
        "explicit_source_resolved_configuration_order")
    result["input_index_sha256"] = hashlib.sha256(index_bytes).hexdigest()
    result["resampling_kind"] = (
        "delete_one_jackknife" if result["do_jackknife"] else "central_only")
    return result


def validate_gradient_flow_tmd_ratio_results(results, *, nconf=None):
    """Validate finite-``b`` ratio-v2 axes and all scalar invariants."""
    ratio = np.asarray(results["ratio"])
    if ratio.ndim != 9 or ratio.shape[0:2] != (2, 4):
        raise ValueError(
            "TMD ratio 必须为 "
            "(channel=2,orientation=4,component,direction,z,b,tsep,"
            f"insertion,momentum) 九轴数组，收到 {ratio.shape}"
        )
    nz, nb = int(ratio.shape[4]), int(ratio.shape[5])
    flattened = dict(results)
    for key in (
            "c3_mean", "ratio", "ratio_jackknife_mean",
            "ratio_jackknife_bias_corrected",
            "ratio_jackknife_error_real",
            "ratio_jackknife_error_imag"):
        values = np.asarray(results[key])
        if values.shape != ratio.shape:
            raise ValueError(f"{key} 与 TMD ratio shape 不一致")
        flattened[key] = values.reshape(
            values.shape[:4] + (nz * nb,) + values.shape[6:])
    if "ratio_jackknife_samples" in results:
        values = np.asarray(results["ratio_jackknife_samples"])
        if values.shape != ratio.shape + (values.shape[-1],):
            raise ValueError(
                "ratio_jackknife_samples 与 TMD ratio shape 不一致")
        flattened["ratio_jackknife_samples"] = values.reshape(
            values.shape[:4] + (nz * nb,) + values.shape[6:])
    flattened["schema"] = RATIO_SCHEMA
    flattened["axes"] = (
        "channel", "z_orientation", "component", "direction",
        "z", "tsep", "insertion", "momentum_abs",
    )
    validate_gradient_flow_ratio_results(flattened, nconf=nconf)

    if results.get("schema") != TMD_RATIO_SCHEMA:
        raise ValueError("TMD ratio schema 不一致")
    expected_axes = (
        "channel", "z_orientation", "component", "direction",
        "z", "b", "tsep", "insertion", "momentum_abs",
    )
    if tuple(results.get("axes", ())) != expected_axes:
        raise ValueError("TMD ratio axes 不一致")
    b_values = tuple(results.get("b_values", ()))
    if len(b_values) != nb:
        raise ValueError("b_values 与 TMD b 轴不一致")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in b_values):
        raise ValueError("b_values 必须是非布尔整数")
    if len(set(b_values)) != len(b_values):
        raise ValueError("b_values 不得重复")
    if results.get("b_unit") != "lattice_spacing_a":
        raise ValueError("b_unit 必须为 lattice_spacing_a")
    if results.get("observable_flatten_order") != "z_major_b_minor_c_order":
        raise ValueError("TMD observable flatten order 不一致")
    if results.get("base_schema") != RATIO_SCHEMA:
        raise ValueError("TMD base schema 不一致")
    do_jackknife = bool(results["do_jackknife"])

    integration_fields = (
        "conf_ids", "source_resolved_c2", "c2_source_length",
        "input_index_schema", "input_index_sha256", "resampling_kind",
    )
    present = tuple(name in results for name in integration_fields)
    if any(present):
        if not all(present):
            raise ValueError(
                "跨组态 ratio metadata 不完整: "
                + ", ".join(
                    name for name, exists in zip(integration_fields, present)
                    if not exists
                ))
        conf_ids = tuple(results["conf_ids"])
        if not conf_ids:
            raise ValueError("跨组态 conf_ids 不得为空")
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer))
               for value in conf_ids):
            raise ValueError("跨组态 conf_ids 必须是非布尔整数")
        conf_ids = tuple(int(value) for value in conf_ids)
        if len(set(conf_ids)) != len(conf_ids):
            raise ValueError("跨组态 conf_ids 不得重复")
        if nconf is not None and len(conf_ids) != int(nconf):
            raise ValueError("conf_ids 长度与 nconf 不一致")
        if results.get("source_resolved_c2") is not True:
            raise ValueError("跨组态计算必须声明 source-resolved C2")
        source_length = results.get("c2_source_length")
        if (isinstance(source_length, (bool, np.bool_))
                or not isinstance(source_length, (int, np.integer))
                or int(source_length) <= 0):
            raise ValueError("c2_source_length 必须是正整数")
        if results.get("input_index_schema") != (
                "explicit_source_resolved_configuration_order"):
            raise ValueError("跨组态 input_index_schema 不一致")
        expected_index_hash = hashlib.sha256(
            np.asarray(conf_ids, dtype='<i8').tobytes(order='C')
        ).hexdigest()
        if results.get("input_index_sha256") != expected_index_hash:
            raise ValueError("跨组态 input_index_sha256 不一致")
        expected_resampling = (
            "delete_one_jackknife" if do_jackknife else "central_only")
        if results.get("resampling_kind") != expected_resampling:
            raise ValueError("跨组态 resampling_kind 与 do_jackknife 不一致")

    valid = np.asarray(results["valid_insertion_mask"])
    expanded_valid = np.broadcast_to(
        valid[None, None, None, None, None, None, :, :, None],
        ratio.shape,
    )
    odd = ORIENTATIONS.index("odd_difference")
    odd_z0 = ratio[1, odd, :, :, 0, :, :, :, :]
    odd_valid = np.broadcast_to(
        valid[None, None, :, :, None], odd_z0.shape)
    kw = {"rtol": 5e-12, "atol": 1e-10, "equal_nan": True}
    if not np.allclose(odd_z0[odd_valid], 0.0, **kw):
        raise ValueError("TMD helicity odd z=0 恒等式失败")
    if not np.isfinite(ratio[expanded_valid]).all():
        raise ValueError("TMD ratio 有效插入含非有限值")
    return True


def validate_gradient_flow_tmd_ratio_2d_results(results, *, nconf=None):
    """Validate ratio-v3 two-component transverse axes and metadata."""
    ratio = np.asarray(results["ratio"])
    if ratio.ndim != 10 or ratio.shape[0:2] != (2, 4):
        raise ValueError(
            "2D TMD ratio 必须为 "
            "(channel=2,orientation=4,component,direction,z,b_x,b_y,"
            f"tsep,insertion,momentum) 十轴数组，收到 {ratio.shape}")
    nz, nbx, nby = (
        int(ratio.shape[4]), int(ratio.shape[5]), int(ratio.shape[6]))
    flattened = dict(results)
    for key in (
            "c3_mean", "ratio", "ratio_jackknife_mean",
            "ratio_jackknife_bias_corrected",
            "ratio_jackknife_error_real",
            "ratio_jackknife_error_imag"):
        values = np.asarray(results[key])
        if values.shape != ratio.shape:
            raise ValueError(f"{key} 与 2D TMD ratio shape 不一致")
        flattened[key] = values.reshape(
            values.shape[:4] + (nz * nbx * nby, 1) + values.shape[7:])
    if "ratio_jackknife_samples" in results:
        values = np.asarray(results["ratio_jackknife_samples"])
        if values.shape != ratio.shape + (values.shape[-1],):
            raise ValueError(
                "ratio_jackknife_samples 与 2D ratio shape 不一致")
        flattened["ratio_jackknife_samples"] = values.reshape(
            values.shape[:4] + (nz * nbx * nby, 1) + values.shape[7:])
    flattened["schema"] = TMD_RATIO_SCHEMA
    flattened["axes"] = (
        "channel", "z_orientation", "component", "direction",
        "z", "b", "tsep", "insertion", "momentum_abs",
    )
    flattened["b_values"] = (0,)
    flattened["b_unit"] = "lattice_spacing_a"
    flattened["observable_flatten_order"] = "z_major_b_minor_c_order"
    validate_gradient_flow_tmd_ratio_results(flattened, nconf=nconf)

    if results.get("schema") != TMD_RATIO_2D_SCHEMA:
        raise ValueError("2D TMD ratio schema 不一致")
    expected_axes = (
        "channel", "z_orientation", "component", "direction",
        "z", "b_x", "b_y", "tsep", "insertion", "momentum_abs",
    )
    if tuple(results.get("axes", ())) != expected_axes:
        raise ValueError("2D TMD ratio axes 不一致")
    validated = _validate_tmd_ratio_2d_metadata(
        results.get("b_x_values", ()), results.get("b_y_values", ()),
        nbx, nby, results.get("z_dir"), results.get("b_x_dir"),
        results.get("b_y_dir"), results.get("transverse_order"))
    if validated[0] != tuple(results.get("b_x_values", ())):
        raise ValueError("2D TMD b_x_values 不一致")
    if validated[1] != tuple(results.get("b_y_values", ())):
        raise ValueError("2D TMD b_y_values 不一致")
    if validated[2] != results.get("z_dir"):
        raise ValueError("2D TMD z_dir 不一致")
    if validated[3] != results.get("b_x_dir"):
        raise ValueError("2D TMD b_x_dir 不一致")
    if validated[4] != results.get("b_y_dir"):
        raise ValueError("2D TMD b_y_dir 不一致")
    if validated[5] != tuple(results.get("transverse_order", ())):
        raise ValueError("2D TMD transverse_order 不一致")
    if results.get("b_unit") != "lattice_spacing_a":
        raise ValueError("2D TMD b_unit 必须为 lattice_spacing_a")
    if results.get("observable_flatten_order") != (
            "z_major_bx_middle_by_minor_c_order"):
        raise ValueError("2D TMD observable flatten order 不一致")

    valid = np.asarray(results["valid_insertion_mask"])
    odd = ORIENTATIONS.index("odd_difference")
    odd_z0 = ratio[1, odd, :, :, 0, :, :, :, :, :]
    odd_valid = np.broadcast_to(
        valid[None, None, None, None, :, :, None], odd_z0.shape)
    kw = {"rtol": 5e-12, "atol": 1e-10, "equal_nan": True}
    if not np.allclose(odd_z0[odd_valid], 0.0, **kw):
        raise ValueError("2D TMD helicity odd z=0 恒等式失败")
    return True


def physical_helicity_ratio(
        ratio, *, orientation="odd_difference",
        component="helicity_physical_T_minus_S"):
    """取参考 helicity 的最终 Euclidean 相位投影 ``Re[-i R_H]``。

    对 schema-v2 的 ``combined``，参考 OPE 文件保存的是
    ``Mtiti+Mijij``；理论目标 ``Mtiti-Mijij`` 应在 OPE 层显式选择后
    再进入 ratio。默认直接返回理论目标 ``T_H-S_H``；``component=None``
    才返回指定 orientation 下的全部 component，供显式诊断使用。
    """
    array = np.asarray(ratio, dtype=np.complex128)
    if array.ndim < 2:
        raise ValueError("ratio 至少需要 channel/orientation 两个轴")
    try:
        io = ORIENTATIONS.index(orientation)
    except ValueError as exc:
        raise ValueError(f"未知 orientation: {orientation}") from exc
    if array.shape[0] < 2 or array.shape[1] <= io:
        raise ValueError("ratio 缺少 helicity/orientation 轴")
    selected = array[1, io]
    if component == "helicity_physical_T_minus_S":
        if selected.ndim < 1 or selected.shape[0] < 3:
            raise ValueError("ratio 未保存 Mtiti/Mijij component")
        selected = selected[1] - selected[2]
    elif component == "helicity_diagnostic_T_plus_S":
        if selected.ndim < 1 or selected.shape[0] < 3:
            raise ValueError("ratio 未保存 Mtiti/Mijij component")
        selected = selected[1] + selected[2]
    elif component is not None:
        if component not in COMPONENTS:
            raise ValueError(f"未知 component: {component}")
        if component == "combined":
            raise ValueError(
                "helicity combined 是 legacy T_H+S_H；请显式选择 "
                "helicity_physical_T_minus_S 或 "
                "helicity_diagnostic_T_plus_S")
        if selected.ndim < 1 or selected.shape[0] not in (3, len(COMPONENTS)):
            raise ValueError(
                "按 component 选择时，ratio 必须保留前三组件或完整六组件轴")
        component_index = COMPONENTS.index(component)
        if component_index >= selected.shape[0]:
            raise ValueError(f"ratio 未保存 component: {component}")
        selected = selected[component_index]
    return np.real(-1j * selected)


def validate_gradient_flow_ratio_results(results, *, nconf=None):
    """验证 ratio-v2 结果的轴、复数 dtype、orientation 与 component 恒等式。"""
    ratio = np.asarray(results["ratio"])
    c3 = np.asarray(results["c3_mean"])
    c2 = np.asarray(results["c2_mean"])
    jk_mean = np.asarray(results["ratio_jackknife_mean"])
    jk_bias = np.asarray(results["ratio_jackknife_bias_corrected"])
    errors = (
        np.asarray(results["ratio_jackknife_error_real"]),
        np.asarray(results["ratio_jackknife_error_imag"]),
    )
    if ratio.ndim != 8 or ratio.shape[0:2] != (2, 4):
        raise ValueError(
            "ratio 必须为 (channel=2,orientation=4,...) 八轴数组，"
            f"收到 {ratio.shape}"
        )
    complex_arrays = (ratio, c3, jk_mean, jk_bias)
    if any(array.dtype.kind != "c" for array in complex_arrays):
        raise ValueError("ratio/c3/jackknife 结果必须保持复数 dtype")
    if any(array.shape != ratio.shape for array in (c3, jk_mean, jk_bias)):
        raise ValueError("ratio/c3/jackknife 形状不一致")
    for error in errors:
        if error.shape != ratio.shape or error.dtype.kind != "f":
            raise ValueError("ratio jackknife 误差形状/dtype 错误")
    ndir, nz, ndt, nins, nmom = (
        ratio.shape[3], ratio.shape[4], ratio.shape[5],
        ratio.shape[6], ratio.shape[7],
    )
    if c2.shape != (2, ndir, ndt, nmom):
        raise ValueError(
            "c2_mean 必须为 (channel=2,direction,tsep,momentum)，"
            f"收到 {c2.shape}"
        )
    if c2.dtype.kind != "c" or not np.isfinite(c2).all():
        raise ValueError("c2_mean 必须是有限复数")
    if np.any(c2 == 0.0):
        raise ValueError("c2_mean 不能含零分母")
    if not np.array_equal(c2[0], c2[1]):
        raise ValueError("两个 channel 必须共享 nopol c2_mean 分母")
    valid = np.asarray(results["valid_insertion_mask"])
    if valid.shape != (ndt, nins) or valid.dtype != np.dtype(bool):
        raise ValueError(
            "valid_insertion_mask 形状/dtype 错误: "
            f"{valid.shape}, {valid.dtype}"
        )
    if nconf is not None and int(nconf) < 2:
        raise ValueError("ratio-v2 需要至少两个组态")
    if results.get("schema") != RATIO_SCHEMA:
        raise ValueError("ratio schema 不一致")
    if results.get("denominator_polarization") != "nopol":
        raise ValueError("ratio-v2 denominator 必须是 nopol")
    if results.get("polarization_for_channel") != POLARIZATION_FOR_CHANNEL:
        raise ValueError("channel polarization metadata 不一致")
    do_jackknife = results.get("do_jackknife")
    if not isinstance(do_jackknife, (bool, np.bool_)):
        raise ValueError("do_jackknife 必须是显式布尔元数据")
    do_jackknife = bool(do_jackknife)
    if tuple(results.get("channel_labels", ())) != tuple(CHANNELS):
        raise ValueError("channel_labels 不一致")
    if tuple(results.get("z_orientation_labels", ())) != tuple(ORIENTATIONS):
        raise ValueError("z_orientation_labels 不一致")
    labels = tuple(results.get("component_labels", ()))
    if len(labels) != ratio.shape[2] or len(set(labels)) != len(labels):
        raise ValueError("component_labels 与 ratio component 轴不一致")
    expected_axes = (
        "channel", "z_orientation", "component", "direction",
        "z", "tsep", "insertion", "momentum_abs",
    )
    if tuple(results.get("axes", ())) != expected_axes:
        raise ValueError("ratio axes 不一致")

    expanded_valid = np.broadcast_to(
        valid[None, None, None, None, None, :, :, None],
        ratio.shape,
    )
    for name, array in (("c3_mean", c3), ("ratio", ratio)):
        if not np.isfinite(array[expanded_valid]).all():
            raise ValueError(f"{name} 的有效插入含非有限值")
        if not np.isnan(array[~expanded_valid]).all():
            raise ValueError(f"{name} 的无效插入必须全部为 NaN")
    jackknife_arrays = (
        ("ratio_jackknife_mean", jk_mean),
        ("ratio_jackknife_bias_corrected", jk_bias),
        ("ratio_jackknife_error_real", errors[0]),
        ("ratio_jackknife_error_imag", errors[1]),
    )
    if do_jackknife:
        for name, array in jackknife_arrays:
            if not np.isfinite(array[expanded_valid]).all():
                raise ValueError(f"{name} 的有效插入含非有限值")
            if not np.isnan(array[~expanded_valid]).all():
                raise ValueError(f"{name} 的无效插入必须全部为 NaN")
        if np.any(errors[0][expanded_valid] < 0.0) or np.any(
                errors[1][expanded_valid] < 0.0):
            raise ValueError("jackknife 误差不能为负")
    else:
        for name, array in jackknife_arrays:
            if not np.isnan(array).all():
                raise ValueError(
                    f"do_jackknife=False 时 {name} 必须全部为 NaN")

    if "ratio_jackknife_samples" in results:
        samples = np.asarray(results["ratio_jackknife_samples"])
        if samples.ndim != ratio.ndim + 1:
            raise ValueError("ratio_jackknife_samples 必须多一个 conf 轴")
        sample_nconf = int(samples.shape[-1])
        if samples.shape != ratio.shape + (sample_nconf,):
            raise ValueError("ratio_jackknife_samples shape 错误")
        if nconf is not None and sample_nconf != int(nconf):
            raise ValueError("ratio_jackknife_samples 的 conf 轴不一致")
        if not do_jackknife:
            raise ValueError(
                "do_jackknife=False 时不得保存 jackknife samples")
        samples_valid = np.broadcast_to(
            valid[None, None, None, None, None, :, :, None, None],
            samples.shape,
        )
        if not np.isfinite(samples[samples_valid]).all():
            raise ValueError("ratio_jackknife_samples 有效位置含非有限值")
        if not np.isnan(samples[~samples_valid]).all():
            raise ValueError("ratio_jackknife_samples 无效位置必须为 NaN")

    denominator = c2[0][None, None, None, :, None, :, None, :]
    expected_ratio = c3 / denominator
    if not np.allclose(
            ratio[expanded_valid], expected_ratio[expanded_valid],
            rtol=5e-12, atol=1e-10):
        raise ValueError("central ratio 与 c3_mean/c2_mean 不一致")

    if ratio.shape[2] == len(COMPONENTS) and labels == COMPONENTS:
        input_precision = results.get("input_precision", "complex128")
        kw = (
            {"rtol": 3e-5, "atol": 2e-6, "equal_nan": True}
            if input_precision == "complex64"
            else {"rtol": 5e-12, "atol": 1e-10, "equal_nan": True}
        )
        oi = {name: ORIENTATIONS.index(name) for name in ORIENTATIONS}
        checks = (
            _orientation_identity_matches(
                ratio[:, oi["even_sum"]],
                ratio[:, oi["plus_z_raw"]],
                ratio[:, oi["minus_z_raw"]], +1.0, input_precision),
            _orientation_identity_matches(
                ratio[:, oi["odd_difference"]],
                ratio[:, oi["plus_z_raw"]],
                ratio[:, oi["minus_z_raw"]], -1.0, input_precision),
        )
        if not all(bool(np.asarray(check).all()) for check in checks):
            raise ValueError("ratio orientation linear identities failed")
        for name, array in (("c3_mean", c3), ("ratio", ratio)):
            odd_z0 = array[1, oi["odd_difference"], :, :, 0, :, :, :]
            odd_valid = np.broadcast_to(
                valid[None, None, :, :, None], odd_z0.shape
            )
            if not np.allclose(odd_z0[odd_valid], 0.0, **kw):
                raise ValueError(f"{name} helicity odd z=0 恒等式失败")
    return True


__all__ = [
    "RATIO_SCHEMA", "TMD_RATIO_SCHEMA", "TMD_RATIO_2D_SCHEMA",
    "POLARIZATION_FOR_CHANNEL",
    "covariance_ratio_estimator", "calculate_gradient_flow_ratios",
    "calculate_gradient_flow_tmd_ratios",
    "calculate_gradient_flow_tmd_ratios_2d",
    "calculate_gradient_flow_tmd_ratios_from_configurations",
    "calculate_gradient_flow_tmd_ratios_2d_from_configurations",
    "calculate_ratios",
    "physical_helicity_ratio", "validate_gradient_flow_ratio_results",
    "validate_gradient_flow_tmd_ratio_results",
    "validate_gradient_flow_tmd_ratio_2d_results",
]
