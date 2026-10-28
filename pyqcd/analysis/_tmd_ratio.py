"""
核子胶子 TMD-PDF 裸矩阵元诊断：ensemble 不相连 ratio（b⊥ 扩展）
==============================================================

梯度流重整化方案中，核子中胶子 TMD 算符矩阵元 <N|O(z,b⊥)|N> 通过
不相连因子化三点函数计算（照抄 docker-v20260805 analyze.py 的
code_1.py 算法，扩展 b⊥ 维度）：

    C3(dt, dtau, z, b) = ⟨C2(dt)·OPE(dtau,z,b)⟩
                        − ⟨C2(dt)⟩⟨OPE(dtau,z,b)⟩        （ensemble 协方差）
    R(dt, dtau, z, b) = mean_t0(C3) / mean_t0(Re C2)
    逐 (z, b) 相关拟合：R = c0 + c1·e^{−dE·dtau} + c1·e^{−dE·(dt−dtau)}

其中 OPE(dtau, z, b) 为梯度流涂抹后的胶子 TMD staple 算符
O(z,b⊥) = M^{tx;tx} + M^{ty;ty} − 2M^{xy;xy} 在时间片 dtau 的
空间求和值（形状 (nz, nb, Nt)），C2 为核子 2pt 关联函数。

输出：逐 replica 与 full-sample central 的裸参数。只有 rank、拟合质量、
平台与独立物理门同时通过时，结果才可交给后续重整化讨论。
"""
from __future__ import annotations

import os
import time
import hashlib

import numpy as np

from ._disconnected import (
    aggregate_fit_statuses, ensemble_covariance_ratio,
    fit_status_from_samples, model_ratio, model_ratio_jacobian,
    shared_resampling_indices, sem,
)
from ._bare_sumratio import (
    fit_forward_sumratio_difference, physical_tmin,
)


_RATIO_ESTIMATORS = frozenset(("ensemble",))
_PLATEAU_CHI2_DOF_MAX = 10.0
_PLATEAU_Q_MIN = 1.0e-6
_FIT_CHI2_DOF_MAX = 10.0
_FIT_FINITE_FRACTION_MIN = 0.9


def _aggregate_label(values, *, good="good", failed="failed", partial="partial"):
    values = [str(value) for value in np.asarray(values).ravel()]
    if values and all(value == good for value in values):
        return good
    if values and all(value == failed for value in values):
        return failed
    return partial


def _disconnected_ratio(corr2, ope, corr3, *, estimator="ensemble"):
    """Vacuum-subtract factorized C3 and form the TMD ratio.

    The reference estimator first averages ``C2*O - C2*<O>`` and the
    denominator ``C2`` over the source axis, then divides. The legacy
    configuration-by-configuration ratio is retained only as an explicit
    reproduction option.
    """
    if estimator not in _RATIO_ESTIMATORS:
        raise ValueError(
            f"ratio_estimator 必须来自 {sorted(_RATIO_ESTIMATORS)!r}")

    c2 = np.asarray(corr2)
    op = np.asarray(ope)
    c3 = np.asarray(corr3)
    if c2.ndim != 3:
        raise ValueError("corr2 必须具有 (sample,source,dt) 三轴")
    if op.ndim != 5:
        raise ValueError("ope 必须具有 (sample,source,dtau,nz,nb) 五轴")
    if c3.ndim != 6:
        raise ValueError(
            "corr3 必须具有 (sample,source,dt,dtau,nz,nb) 六轴")
    nsample, nsource, ndt = c2.shape
    if nsample < 1 or nsource < 1 or ndt < 1:
        raise ValueError("corr2 的 sample/source/dt 轴必须非空")
    if op.shape[:2] != (nsample, nsource):
        raise ValueError("corr2 与 ope 的 sample/source 轴不一致")
    if c3.shape != (nsample, nsource, ndt) + op.shape[2:]:
        raise ValueError("corr2/ope/corr3 的 dt 或 observable 轴不一致")

    result = ensemble_covariance_ratio(c2, op, c3)
    return result[0]


def _plateau_points(ratio, dt_max, dt_start, dt_end, cut):
    front = cut // 2
    back = cut - front
    dt_stop = min(dt_end, dt_max - 1, ratio.shape[1] - 1)
    points = []
    for dt in range(dt_start, dt_stop + 1):
        dtau_stop = min(dt - back, ratio.shape[2] - 1)
        points.extend((dt, dtau) for dtau in range(front, dtau_stop + 1))
    return points


def _aggregate_status(status_by_channel):
    return aggregate_fit_statuses(np.asarray(status_by_channel).ravel())[0]


def _plateau_status_metadata(
        ratio, c0_plateau, dt_max, dt_start, dt_end, cut):
    status = np.full(
        c0_plateau.shape[1:],
        "statistically_unidentifiable",
        dtype="<U32",
    )
    sample_rank = np.zeros(c0_plateau.shape[1:], dtype=np.int64)
    reason = np.full(c0_plateau.shape[1:], "", dtype="<U128")
    quality = np.full(c0_plateau.shape[1:], "not_assessed", dtype="<U24")
    physical = np.full(c0_plateau.shape[1:], "unverified", dtype="<U24")
    chi2_dof = np.full(c0_plateau.shape[1:], np.nan, dtype=np.float64)
    q_value = np.full(c0_plateau.shape[1:], np.nan, dtype=np.float64)
    point_rank = np.zeros(c0_plateau.shape[1:], dtype=np.int64)

    if c0_plateau.shape[0] < 2:
        reason[...] = (
            f"Nsample={c0_plateau.shape[0]} cannot identify plateau variance"
        )
    elif not _plateau_points(ratio, dt_max, dt_start, dt_end, cut):
        reason[...] = "empty plateau window"
    else:
        points = _plateau_points(ratio, dt_max, dt_start, dt_end, cut)
        window = np.stack(
            [ratio[:, dt, dtau] for dt, dtau in points], axis=1)
        finite = np.all(np.isfinite(window), axis=(0, 1))
        flat_samples = window.reshape(window.shape[0], -1, *window.shape[2:])
        centered = flat_samples - np.mean(flat_samples, axis=0, keepdims=True)
        variance = np.sum(np.abs(centered) ** 2, axis=(0, 1))
        identifiable = finite & (variance > 0.0)
        status[identifiable] = "identifiable"
        reason[identifiable] = "finite fluctuating plateau samples"
        reason[~finite] = "plateau window contains non-finite samples"
        reason[finite & ~identifiable] = (
            "plateau sample has no fluctuating mode")

        n_points = len(points)
        n_sample = window.shape[0]
        try:
            from scipy.stats import chi2 as chi2_distribution
        except ImportError:
            chi2_distribution = None
        for z_index in range(c0_plateau.shape[1]):
            for b_index in range(c0_plateau.shape[2]):
                if not identifiable[z_index, b_index]:
                    continue
                sample = window[:, :, z_index, b_index]
                rank = int(np.linalg.matrix_rank(
                    sample - sample.mean(axis=0, keepdims=True),
                    tol=1.0e-12 * max(1.0, float(np.max(np.abs(sample))))))
                sample_rank[z_index, b_index] = rank
                if n_sample < 2 or n_points < 2 or rank < 2:
                    quality[z_index, b_index] = "not_assessed"
                    continue
                covariance = np.cov(sample, rowvar=False, ddof=1)
                covariance = (covariance + covariance.T) * 0.5
                eig, vec = np.linalg.eigh(covariance)
                keep = eig > 1.0e-12 * max(float(eig[-1]), 0.0)
                point_rank[z_index, b_index] = int(np.count_nonzero(keep))
                if np.count_nonzero(keep) < 2:
                    quality[z_index, b_index] = "failed"
                    continue
                residual = sample.mean(axis=0) - sample.mean()
                whitened = vec[:, keep].T @ residual / np.sqrt(eig[keep])
                value = float(whitened @ whitened)
                dof = int(np.count_nonzero(keep) - 1)
                chi2_dof[z_index, b_index] = value / dof
                if chi2_distribution is not None:
                    q_value[z_index, b_index] = float(
                        chi2_distribution.sf(value, dof))
                quality_ok = (
                    np.isfinite(value)
                    and chi2_dof[z_index, b_index]
                    <= _PLATEAU_CHI2_DOF_MAX
                    and (chi2_distribution is None or
                         q_value[z_index, b_index] >= _PLATEAU_Q_MIN)
                )
                quality[z_index, b_index] = (
                    "good" if quality_ok else "failed")
                # A single plateau window proves local consistency, not
                # window/tsep/scheme stability of a physical matrix element.
                # Keep the stronger status unverified even when quality passes.

    quality_values = quality.ravel().tolist()
    quality_status = (
        "good" if all(value == "good" for value in quality_values)
        else "failed" if all(value == "failed" for value in quality_values)
        else "partial")
    physical_values = physical.ravel().tolist()
    physical_status = (
        "supported" if all(value == "supported" for value in physical_values)
        else "unsupported" if all(value == "unsupported" for value in physical_values)
        else "unverified")

    return {
        "plateau_status": np.asarray(_aggregate_status(status)),
        "plateau_status_by_channel": status,
        "plateau_sample_rank": sample_rank,
        "plateau_reason": reason,
        "plateau_fit_quality_status": np.asarray(quality_status),
        "plateau_fit_quality_status_by_channel": quality,
        "plateau_point_rank": point_rank,
        "plateau_chi2_dof": chi2_dof,
        "plateau_q_value": q_value,
        "plateau_quality_thresholds": np.asarray(
            f"chi2/dof<={_PLATEAU_CHI2_DOF_MAX};Q>={_PLATEAU_Q_MIN}"),
        "plateau_physical_result_status": np.asarray(physical_status),
        "plateau_physical_result_status_by_channel": physical,
    }


def run_disconnected_tmd_ratio(corr_2pt_all, ope_all, conf_ids,
                               run_dir, logger=print,
                               NT=72, nz=24, nb=1,
                               dt_max=20, dt_start=7, dt_end=10,
                               cut=6, p0=None, momentum='P200',
                               z_s=0, ratio_estimator='ensemble',
                               a_fm=0.1053, sumratio_threshold_fm=0.6,
                               sumratio_tsep_max=14,
                               sumratio_cuts=(1, 2, 3),
                               bootstrap_samples=200,
                               legacy_fit=True):
    """不相连胶子 TMD ratio + 逐 (z,b) 拟合（code_1.py 算法，b⊥ 扩展）。

    Args:
        corr_2pt_all: {conf_id: {f'corr_pp_{mom}': (source,dt)}} 源分辨核子 2pt。
        ope_all:      {conf_id: {'tmd': (nz, nb, Nt)}} 梯度流 TMD 算符
                      逐时间片空间求和（实数）。
        conf_ids:     组态列表（jackknife 重采样轴）。
        run_dir:      输出目录（analysis/tmd_ratio/）。
        nz:           z 方向格点数（OPE 的 z 维长度）。
        nb:           b⊥ 维长度。
        momentum:     动量标签（'P200' 等，用于命名输出文件）。
        z_s:          比值参考 z 索引（保留参数，备用）。
        ratio_estimator: 仅接受 ``"ensemble"``；逐组态 ratio 路径已删除。
        bootstrap_samples: 非 jackknife 路径的 bootstrap replica 数；默认
            200 保持旧调用兼容，生产工作流应显式提高并记录该值。
    Returns:
        ch_results: {hadron: {'ratio', 'c0', 'c1', 'dE', 'chi2', 'x_coor'}}
            c0/c1/dE/chi2 形状 (Nsample, nz, nb)。
    """
    from ._fitter import (FitParams, covariance_effective_rank,
                          covariance_sample_rank, fit, fit_central,
                          fit_identifiability)

    if ratio_estimator not in _RATIO_ESTIMATORS:
        raise ValueError(
            f"ratio_estimator 必须来自 {sorted(_RATIO_ESTIMATORS)!r}")
    if (isinstance(bootstrap_samples, (bool, np.bool_))
            or not isinstance(bootstrap_samples, (int, np.integer))
            or bootstrap_samples <= 0):
        raise ValueError("bootstrap_samples 必须是正整数")
    bootstrap_samples = int(bootstrap_samples)

    if p0 is None:
        p0 = {"c0": 0.6, "c1": -2, "dE": 1}

    Nconf = len(conf_ids)
    Nsample = max(Nconf, 1)
    # 重采样方式选择：delete-one jackknife 需样本数显著大于数据点数；
    # 小样本用 bootstrap（Nsample=200 满秩协方差，svdcut 下拟合稳定）。
    # 守卫：Nconf<2 禁用 jackknife（n−1=0 除零 → 全 NaN）
    front_remove = cut // 2
    back_remove = cut - front_remove
    # 真实数据点计数（与下方 x_coor 同式）：Σ_dt max(dt−front−back+1, 0)
    Ndata_est = sum(max(dt - front_remove - back_remove + 1, 0)
                    for dt in range(dt_start, dt_end + 1))
    jack = (Nconf >= 2) and (Nconf > Ndata_est)
    if not jack:
        Nsample = bootstrap_samples
    if Nconf < 2:
        Nsample = 1     # 单组态：重采样无统计意义，走下方直通分支
    out_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    os.makedirs(out_dir, exist_ok=True)

    channels = [('proton', 'corr_pp', 'proton')]
    ch_results = {}

    for ch_key, k2, had_name in channels:
        logger(f"\n  Channel: {had_name} at {momentum} (TMD, b⊥={nb} vals)")

        key2 = f'{k2}_{momentum}'
        _corr = np.stack([np.asarray(corr_2pt_all[cid][key2]) for cid in conf_ids])
        if _corr.ndim != 3:
            raise ValueError(
                "TMD physical input requires source-resolved C2 with shape "
                f"(Nsample,source,dt); got {_corr.shape}")
        if _corr.shape[1] != NT or _corr.shape[2] < dt_max:
            raise ValueError(
                "source-resolved C2 must have shape (Nsample,NT,dt>=dt_max)")
        if np.max(np.abs(_corr.imag), initial=0.0) > 1e-12 * max(
                1.0, float(np.max(np.abs(_corr.real)))):
            raise ValueError("C2 contains a material imaginary component")
        _corr = np.asarray(_corr.real[:, :, :dt_max], dtype=np.float64)
        # OPE tmd: (Nconf, nz, nb, Nt) → (Nconf, tau, z, b)
        _ope = np.stack([np.real(ope_all[cid]['tmd']) for cid in conf_ids])
        _ope = _ope.transpose(0, 3, 1, 2)   # (Nconf, Nt, nz, nb)

        _ope_rel = np.zeros((Nconf, NT, dt_max, nz, nb), dtype=np.float64)
        for ti in range(NT):
            ope_shift = np.roll(_ope, -ti, axis=1)
            _ope_rel[:, ti, :, :, :] = ope_shift[:, :dt_max, :, :]

        if Nconf < 2:
            indices = None
            ratio = np.full(
                (1, dt_max, dt_max, nz, nb), np.nan, dtype=np.float64)
            ratio_original = ratio[0]
            resampling_seed = None
            index_sha256 = None
        else:
            resampling_seed = 0
            indices = shared_resampling_indices(
                Nconf, jack, Nsample=Nsample, seed=resampling_seed)
            index_sha256 = hashlib.sha256(
                np.asarray(indices, dtype='<i8').tobytes(order='C')).hexdigest()
            ratio = ensemble_covariance_ratio(
                _corr, _ope_rel, indices=indices)
            ratio_original = ensemble_covariance_ratio(
                _corr, _ope_rel,
                indices=np.arange(Nconf, dtype=np.int64)[None, :])[0]
        if np.iscomplexobj(ratio):
            if np.max(np.abs(ratio.imag)) > 1e-12:
                raise FloatingPointError(
                    "TMD ratio acquired a material imaginary residual")
            ratio = ratio.real
        np.save(os.path.join(out_dir, f'ratio_{had_name}_{momentum}.npy'), ratio)
        np.save(
            os.path.join(out_dir, f'ratio_original_{had_name}_{momentum}.npy'),
            np.asarray(ratio_original).real)

        # 逐 (z,b) 相关拟合（Nconf<2 时跳过——协方差奇异，统计无意义）
        if Nconf < 2:
            fit_reason = (
                f"Nconf={Nconf} cannot support delete-one jackknife covariance"
            )
            required_rank = len(p0)
            fit_status_by_channel = np.full(
                (nz, nb), "statistically_unidentifiable", dtype="<U32")
            effective_rank_by_channel = np.zeros(
                (nz, nb), dtype=np.int64)
            sample_rank_by_channel = np.zeros((nz, nb), dtype=np.int64)
            required_rank_by_channel = np.full(
                (nz, nb), required_rank, dtype=np.int64)
            fit_reason_by_channel = np.full(
                (nz, nb), fit_reason, dtype="<U128")
            fit_quality_by_channel = np.full(
                (nz, nb), "failed", dtype="<U24")
            physical_result_by_channel = np.full(
                (nz, nb), "unsupported", dtype="<U24")
            para_c0 = np.full((1, nz, nb), np.nan)
            para_c1 = np.full((1, nz, nb), np.nan)
            para_dE = np.full((1, nz, nb), np.nan)
            chi2 = np.full((1, nz, nb), np.nan)
            c0_central = np.full((nz, nb), np.nan)
            c1_central = np.full((nz, nb), np.nan)
            de_central = np.full((nz, nb), np.nan)
            chi2_central = np.full((nz, nb), np.nan)
            central_status = np.full(
                (nz, nb), "statistically_unidentifiable", dtype="<U32")
            c0_plateau = np.full((1, nz, nb), np.nan)
            report_lines = [
                "=" * 70,
                f"  TMD Fit Report: {had_name}, {momentum}, Nconf={Nconf}",
                "=" * 70,
                "fit status = statistically_unidentifiable",
                "effective covariance rank = 0",
                "sample covariance rank = 0",
                f"required parameter rank = {required_rank}",
                f"fit skipped: {fit_reason}",
                "",
            ]
            with open(os.path.join(
                    out_dir, f'1_fit_report_{momentum}.txt'), 'w') as f:
                f.write("\n".join(report_lines))
            np.savez(
                os.path.join(out_dir, f'0_fit_data_{momentum}.npz'),
                c0=para_c0,
                c1=para_c1,
                dE=para_dE,
                chi2=chi2,
                c0_central=c0_central,
                c1_central=c1_central,
                dE_central=de_central,
                chi2_central=chi2_central,
                central_fit_status=central_status,
                fit_status=np.asarray("statistically_unidentifiable"),
                effective_rank=np.asarray(0, dtype=np.int64),
                sample_rank=np.asarray(0, dtype=np.int64),
                required_rank=np.asarray(required_rank, dtype=np.int64),
                fit_reason=np.asarray(fit_reason),
                fit_status_by_channel=fit_status_by_channel,
                effective_rank_by_channel=effective_rank_by_channel,
                sample_rank_by_channel=sample_rank_by_channel,
                required_rank_by_channel=required_rank_by_channel,
                fit_reason_by_channel=fit_reason_by_channel,
                fit_quality_status=np.asarray("failed"),
                fit_quality_status_by_channel=fit_quality_by_channel,
                physical_result_status=np.asarray("unsupported"),
                physical_result_status_by_channel=physical_result_by_channel,
                ratio_estimator=np.asarray(ratio_estimator),
                resampling_kind=np.asarray("none"),
                Nsample=np.asarray(Nsample, dtype=np.int64),
                bootstrap_samples=np.asarray(
                    bootstrap_samples, dtype=np.int64),
                ratio_original=np.asarray(ratio_original).real,
                bootstrap_indices=np.asarray([], dtype=np.int64),
                resampling_seed=np.asarray(-1, dtype=np.int64),
                resampling_index_sha256=np.asarray("none"),
                conf_ids=np.asarray(conf_ids, dtype=np.int64),
            )
            np.save(
                os.path.join(out_dir, f'c0_plateau_{momentum}.npy'),
                c0_plateau,
            )
            plateau_metadata = _plateau_status_metadata(
                ratio, c0_plateau, dt_max, dt_start, dt_end, cut)
            np.savez(
                os.path.join(
                    out_dir, f'c0_plateau_status_{momentum}.npz'),
                **plateau_metadata,
            )
            logger(f"  fit skipped: {fit_reason}; saved NaN fit artifacts")
            ch_results[had_name] = {
                'ratio': ratio, 'c0': para_c0, 'c1': para_c1,
                'dE': para_dE, 'chi2': chi2, 'x_coor': [],
                'c0_central': c0_central, 'c1_central': c1_central,
                'dE_central': de_central, 'chi2_central': chi2_central,
                'central_fit_status': central_status,
                'c0_plateau': c0_plateau,
                'fit_status': 'statistically_unidentifiable',
                'plateau_status': str(plateau_metadata['plateau_status']),
                'plateau_fit_quality_status': str(
                    plateau_metadata['plateau_fit_quality_status']),
                'plateau_physical_result_status': str(
                    plateau_metadata['plateau_physical_result_status']),
                'effective_rank': 0, 'sample_rank': 0,
                'required_rank': required_rank, 'fit_reason': fit_reason,
                'fit_status_by_channel': fit_status_by_channel,
                'effective_rank_by_channel': effective_rank_by_channel,
                'sample_rank_by_channel': sample_rank_by_channel,
                'required_rank_by_channel': required_rank_by_channel,
                'fit_reason_by_channel': fit_reason_by_channel,
                'fit_quality_status': 'failed',
                'fit_quality_status_by_channel': fit_quality_by_channel,
                'physical_result_status': 'unsupported',
                'physical_result_status_by_channel': physical_result_by_channel,
                'ratio_estimator': ratio_estimator,
                'resampling_kind': 'none',
                'Nsample': Nsample,
            }
            continue

        # 逐 (z,b) 相关拟合（front/back_remove 已在重采样选择处计算）
        x_coor = [(dt, dtau)
                  for dt in range(dt_start, dt_end + 1)
                  for dtau in range(front_remove, dt - back_remove + 1)]
        Ndata = len(x_coor)

        para_c0 = np.full((Nsample, nz, nb), np.nan)
        para_c1 = np.full((Nsample, nz, nb), np.nan)
        para_dE = np.full((Nsample, nz, nb), np.nan)
        chi2 = np.full((Nsample, nz, nb), np.nan)
        c0_central = np.full((nz, nb), np.nan)
        c1_central = np.full((nz, nb), np.nan)
        de_central = np.full((nz, nb), np.nan)
        chi2_central = np.full((nz, nb), np.nan)
        central_status = np.full(
            (nz, nb), "statistically_unidentifiable", dtype="<U32")
        central_reason = np.full((nz, nb), "", dtype="<U160")
        fit_status_by_channel = np.full(
            (nz, nb), "statistically_unidentifiable", dtype="<U32")
        effective_rank_by_channel = np.zeros((nz, nb), dtype=np.int64)
        sample_rank_by_channel = np.zeros((nz, nb), dtype=np.int64)
        required_rank_by_channel = np.full(
            (nz, nb), len(p0), dtype=np.int64)
        fit_reason_by_channel = np.full((nz, nb), "", dtype="<U128")
        fit_quality_by_channel = np.full(
            (nz, nb), "not_assessed", dtype="<U24")
        physical_result_by_channel = np.full(
            (nz, nb), "unverified", dtype="<U24")

        report_lines = [
            "=" * 70,
            f"  TMD Fit Report: {had_name}, {momentum}, Nconf={Nconf}",
            "=" * 70,
            f"  t_sep range : [{dt_start}, {dt_end}]",
            f"  cut         : {cut}",
            f"  Nsample     : {Nsample}",
            f"  jackknife   : {jack}",
            f"  nz x nb     : {nz} x {nb}",
            "=" * 70, "",
        ]

        t0_fit = time.perf_counter()
        fitpa = FitParams(
            p0=dict(p0), dt_start=dt_start, dt_end=dt_end,
            svdcut=1.0e-6, jacobian=model_ratio_jacobian)
        for _z in (range(nz) if legacy_fit else range(0)):
            for _b in range(nb):
                sub_sample = np.zeros((Nsample, Ndata))
                for i, (dt, dtau) in enumerate(x_coor):
                    sub_sample[:, i] = ratio[:, dt, dtau, _z, _b]
                _fit_result, cov, cond, _last_fit = fit(
                    sub_sample, x_coor, model_ratio, fitpa,
                    jackknife=jack)
                central_sample = np.array([
                    ratio_original[dt, dtau, _z, _b]
                    for dt, dtau in x_coor
                ], dtype=np.float64)
                try:
                    _central_fit, central_fit_status, central_fit_reason, \
                        central_metadata = fit_central(
                            central_sample, x_coor, model_ratio, fitpa, cov,
                            jackknife=jack)
                except (ValueError, FloatingPointError,
                        np.linalg.LinAlgError) as exc:
                    _central_fit = None
                    central_fit_status = "statistically_unidentifiable"
                    central_fit_reason = (
                        f"central_fit_failed:{type(exc).__name__}:{exc}")
                    central_metadata = {}
                central_status[_z, _b] = central_fit_status
                central_reason[_z, _b] = central_fit_reason
                if _central_fit is not None:
                    c0_central[_z, _b] = float(_central_fit.pmean["c0"])
                    c1_central[_z, _b] = float(_central_fit.pmean["c1"])
                    de_central[_z, _b] = float(_central_fit.pmean["dE"])
                    chi2_central[_z, _b] = float(
                        central_metadata["chi2_dof"])
                effective_rank = covariance_effective_rank(
                    cov, svdcut=fitpa.svdcut)
                sample_rank = covariance_sample_rank(cov)
                required_rank = len(p0)
                gate_ok, gate_reason = fit_identifiability(
                    Ndata, required_rank, effective_rank,
                    sample_rank=sample_rank)
                fit_status, fit_reason, _ = fit_status_from_samples(
                    _fit_result, _last_fit,
                    failure_reason=None if gate_ok else gate_reason)
                fit_status_by_channel[_z, _b] = fit_status
                effective_rank_by_channel[_z, _b] = effective_rank
                sample_rank_by_channel[_z, _b] = sample_rank
                fit_reason_by_channel[_z, _b] = fit_reason
                chi2_values = np.asarray(_fit_result["chi2"], dtype=float)
                finite_fraction = float(np.mean(np.isfinite(chi2_values)))
                finite_chi2 = chi2_values[np.isfinite(chi2_values)]
                median_chi2 = (
                    float(np.median(finite_chi2)) if finite_chi2.size else np.nan)
                quality_ok = (
                    fit_status in ("identifiable", "prior_constrained")
                    and finite_fraction >= _FIT_FINITE_FRACTION_MIN
                    and np.isfinite(median_chi2)
                    and median_chi2 <= _FIT_CHI2_DOF_MAX
                    and np.isfinite(cond)
                )
                fit_quality_by_channel[_z, _b] = (
                    "good" if quality_ok else "failed")
                # A local fit cannot support the physical claim without
                # window/scheme stability and the separate plateau gates.
                physical_result_by_channel[_z, _b] = "unverified"
                para_c0[:, _z, _b] = _fit_result["c0"]
                para_c1[:, _z, _b] = _fit_result["c1"]
                para_dE[:, _z, _b] = _fit_result["dE"]
                chi2[:, _z, _b] = _fit_result["chi2"]
                report_lines += [
                    f"z={_z} b={_b} cond={cond:.3g}", "-" * 56,
                    f"fit status = {fit_status}",
                    f"effective covariance rank = {effective_rank}",
                    f"sample covariance rank = {sample_rank}",
                    f"required parameter rank = {required_rank}", "",
                    f"fit quality status = {fit_quality_by_channel[_z, _b]}",
                    f"median chi2/dof = {median_chi2:.4g}",
                    f"physical result status = unverified", "",
                ]

                if fit_status not in ("identifiable", "prior_constrained"):
                    report_lines += [f"fit skipped: {fit_reason}", ""]
                    logger(
                        f"  z={_z} b={_b}: fit skipped "
                        f"({fit_reason})")
                    continue

                if _last_fit is not None:
                    report_lines.append(
                        "last successful replica fit (not the central fit):")
                    report_lines.append(_last_fit.format(maxline=True))
                report_lines += [
                    f"central fit status = {central_fit_status}; "
                    f"reason = {central_fit_reason}",
                    f"central c0 = {c0_central[_z, _b]:.8g}",
                    f"central chi2/dof = {chi2_central[_z, _b]:.6g}",
                ]
                report_lines += ["", ""]

                finite = np.isfinite(para_c0[:, _z, _b])
                if np.any(finite):
                    values = para_c0[finite, _z, _b]
                    error = sem(values, jack) if values.size > 1 else np.nan
                    logger(
                        f"  z={_z} b={_b}: c0={values.mean():.4g} ± "
                        f"{error:.4g}  status={fit_status}")

        if not legacy_fit:
            fit_reason_by_channel[:] = (
                "legacy three-parameter fit disabled; "
                "use Sumratio-difference artifact")
            report_lines.append(
                "legacy three-parameter fit disabled; "
                "use Sumratio-difference artifact")
        fit_status, fit_reason = aggregate_fit_statuses(
            fit_status_by_channel.ravel(), fit_reason_by_channel.ravel())
        fit_quality_status = _aggregate_label(fit_quality_by_channel)
        physical_result_status = _aggregate_label(
            physical_result_by_channel, good="supported",
            failed="unsupported", partial="unverified")
        report_lines.insert(
            7, f"fit status = {fit_status}; fit reason = {fit_reason}")
        report = "\n".join(report_lines)
        with open(os.path.join(out_dir, f'1_fit_report_{momentum}.txt'), 'w') as f:
            f.write(report)
        np.savez(
            os.path.join(out_dir, f'0_fit_data_{momentum}.npz'),
            c0=para_c0,
            c1=para_c1,
            dE=para_dE,
            chi2=chi2,
            c0_central=c0_central,
            c1_central=c1_central,
            dE_central=de_central,
            chi2_central=chi2_central,
            central_fit_status=central_status,
            central_fit_reason=central_reason,
            fit_status=np.asarray(fit_status),
            fit_status_by_channel=fit_status_by_channel,
            effective_rank=np.asarray(
                np.min(effective_rank_by_channel), dtype=np.int64),
            effective_rank_by_channel=effective_rank_by_channel,
            sample_rank=np.asarray(
                np.min(sample_rank_by_channel), dtype=np.int64),
            sample_rank_by_channel=sample_rank_by_channel,
            required_rank=np.asarray(len(p0), dtype=np.int64),
            required_rank_by_channel=required_rank_by_channel,
            fit_reason=np.asarray(fit_reason),
            fit_reason_by_channel=fit_reason_by_channel,
            fit_quality_status=np.asarray(fit_quality_status),
            fit_quality_status_by_channel=fit_quality_by_channel,
            physical_result_status=np.asarray(physical_result_status),
            physical_result_status_by_channel=physical_result_by_channel,
            fit_quality_thresholds=np.asarray(
                f"median_chi2/dof<={_FIT_CHI2_DOF_MAX};"
                f"finite_fraction>={_FIT_FINITE_FRACTION_MIN};condition_finite"),
            ratio_estimator=np.asarray(ratio_estimator),
            resampling_kind=np.asarray(
                "jackknife" if jack else "bootstrap"),
            Nsample=np.asarray(Nsample, dtype=np.int64),
            bootstrap_samples=np.asarray(
                bootstrap_samples, dtype=np.int64),
            ratio_original=np.asarray(ratio_original).real,
            bootstrap_indices=np.asarray(indices, dtype=np.int64),
            resampling_seed=np.asarray(resampling_seed, dtype=np.int64),
            resampling_index_sha256=np.asarray(index_sha256),
            conf_ids=np.asarray(conf_ids, dtype=np.int64),
        )
        # plateau 均值版（fit 窗口内直接平均，抗奇异协方差；test9 整合项）
        c0_plateau = plateau_c0(ratio, dt_max=dt_max, dt_start=dt_start,
                                dt_end=dt_end, cut=cut)
        np.save(os.path.join(out_dir, f'c0_plateau_{momentum}.npy'),
                c0_plateau)
        plateau_metadata = _plateau_status_metadata(
            ratio, c0_plateau, dt_max, dt_start, dt_end, cut)
        np.savez(
            os.path.join(out_dir, f'c0_plateau_status_{momentum}.npz'),
            **plateau_metadata,
        )
        tsep_limit = min(int(sumratio_tsep_max), int(dt_max) - 1)
        sumratio_fit = None
        if tsep_limit > 0:
            tmin = physical_tmin(a_fm, sumratio_threshold_fm)
            valid_cuts = tuple(
                int(value) for value in sumratio_cuts
                if 0 <= int(value) <= tsep_limit // 2)
            if valid_cuts and tsep_limit > tmin + 1:
                sumratio_fit = fit_forward_sumratio_difference(
                    ratio, ratio_original, tmin=tmin, cuts=valid_cuts,
                    tsep_max=tsep_limit)
                np.savez_compressed(
                    os.path.join(
                        out_dir, f'sumratio_difference_{momentum}.npz'),
                    **sumratio_fit,
                    a_fm=np.asarray(a_fm, dtype=np.float64),
                    threshold_fm=np.asarray(
                        sumratio_threshold_fm, dtype=np.float64),
                    momentum=np.asarray(momentum),
                    conf_ids=np.asarray(conf_ids, dtype=np.int64),
                    Nsample=np.asarray(Nsample, dtype=np.int64),
                    bootstrap_samples=np.asarray(
                        bootstrap_samples, dtype=np.int64),
                    resampling_kind=np.asarray(
                        "jackknife" if jack else "bootstrap"),
                    bootstrap_indices=np.asarray(indices, dtype=np.int64),
                    resampling_seed=np.asarray(
                        resampling_seed, dtype=np.int64),
                    resampling_index_sha256=np.asarray(index_sha256),
                    ratio_estimator=np.asarray(ratio_estimator),
                )
        logger(f"  Saved ratio + fit + c0_plateau to {out_dir} "
               f"({time.perf_counter()-t0_fit:.1f}s)")

        ch_results[had_name] = {
            'ratio': ratio, 'c0': para_c0, 'c1': para_c1,
            'dE': para_dE, 'chi2': chi2, 'x_coor': x_coor,
            'c0_central': c0_central, 'c1_central': c1_central,
            'dE_central': de_central, 'chi2_central': chi2_central,
            'central_fit_status': central_status,
            'central_fit_reason': central_reason,
            'c0_plateau': c0_plateau,
            'fit_status': fit_status,
            'fit_status_by_channel': fit_status_by_channel,
            'effective_rank_by_channel': effective_rank_by_channel,
            'sample_rank_by_channel': sample_rank_by_channel,
            'required_rank_by_channel': required_rank_by_channel,
            'fit_reason_by_channel': fit_reason_by_channel,
            'fit_quality_status': fit_quality_status,
            'fit_quality_status_by_channel': fit_quality_by_channel,
            'physical_result_status': physical_result_status,
            'physical_result_status_by_channel': physical_result_by_channel,
            'plateau_status': str(plateau_metadata['plateau_status']),
            'plateau_status_by_channel': (
                plateau_metadata['plateau_status_by_channel']),
            'plateau_sample_rank': plateau_metadata['plateau_sample_rank'],
            'plateau_reason': plateau_metadata['plateau_reason'],
            'plateau_fit_quality_status': str(
                plateau_metadata['plateau_fit_quality_status']),
            'plateau_physical_result_status': str(
                plateau_metadata['plateau_physical_result_status']),
            'sumratio_difference': sumratio_fit,
            'ratio_estimator': ratio_estimator,
            'resampling_kind': "jackknife" if jack else "bootstrap",
            'Nsample': Nsample,
        }

    return ch_results


def plot_tmd_c0(ch_results, run_dir, momentum, logger=print,
                nz=24, nb=1, out_name='c0_tmd'):
    """c0(z,b) 图表：逐 b 面板（z 横轴，errorbar）+ 3D 热图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    os.makedirs(out_dir, exist_ok=True)
    saved_any = False

    for had_name, res in ch_results.items():
        sumratio = res.get('sumratio_difference')
        if sumratio is not None:
            c0 = np.asarray(sumratio['M_boot'])
            quality_gate = (
                np.asarray(sumratio['quality_accepted'], dtype=bool)
                & np.asarray(sumratio['estimate_available'], dtype=bool)
                & ~np.asarray(sumratio['central_fallback_used'], dtype=bool))
            physical = np.where(quality_gate, 'unverified', 'unsupported')
            estimator_label = 'Sumratio-difference AIC'
        else:
            c0 = res['c0']           # (Nsample, nz, nb)
            physical = np.asarray(res.get(
                'physical_result_status_by_channel',
                np.full(c0.shape[1:], 'unverified', dtype='<U24')))
            estimator_label = 'three-parameter ratio fit'
        if not np.any(np.isin(physical, ('supported', 'unverified'))):
            logger(f"  TMD c0 plot skipped for {had_name} {momentum}: "
                   "plateau quality/physical gate did not pass")
            continue
        jackknife = res.get('resampling_kind') == 'jackknife'
        c0m = c0.mean(0); c0e = sem(c0, jackknife)
        z = np.arange(nz)
        status_prefix = (
            '' if np.all(physical == 'supported') else 'diagnostic_')
        title_prefix = (
            '' if np.all(physical == 'supported') else 'DIAGNOSTIC ')

        # 逐 b 面板
        ncol = 2
        nrow = (nb + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                                 squeeze=False)
        for _b in range(nb):
            ax = axes[_b // ncol][_b % ncol]
            supported = np.isin(physical[:, _b], ('supported', 'unverified'))
            plotted = np.full_like(c0m[:, _b], np.nan)
            plotted_error = np.full_like(c0e[:, _b], np.nan)
            plotted[supported] = c0m[supported, _b]
            plotted_error[supported] = c0e[supported, _b]
            ax.errorbar(z, plotted, yerr=plotted_error,
                        fmt='x-', capsize=3)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_xlabel('z'); ax.set_ylabel('c0')
            ax.set_title(
                f'{title_prefix}{had_name} {momentum}: '
                f'{estimator_label}, b⊥={_b}')
            ax.grid(alpha=0.3)
        for _b in range(nb, nrow * ncol):
            axes[_b // ncol][_b % ncol].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(
            out_dir,
            f'{status_prefix}{out_name}_{had_name}_{momentum}.png'),
                    dpi=150)
        saved_any = True
        plt.close(fig)

        # 热图（nb≥2 时）
        if nb >= 2:
            heat = np.where(
                np.isin(physical.T, ('supported', 'unverified')),
                c0m.T, np.nan)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(heat, aspect='auto', origin='lower',
                           extent=[0, nz - 1, 0, nb - 1])
            ax.set_xlabel('z'); ax.set_ylabel('b⊥')
            ax.set_title(
                f'{title_prefix}{had_name} {momentum}: '
                f'{estimator_label} (z, b⊥)')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                                     f'{status_prefix}{out_name}_heatmap_'
                                     f'{had_name}_{momentum}.png'),
                        dpi=150)
            saved_any = True
            plt.close(fig)

    if saved_any:
        logger(f"  TMD c0 plots saved to {out_dir}")
    else:
        logger("  No TMD c0 plot passed the physical-quality gate")


def c2_signal_to_noise(c2_samples):
    """逐 ``dt`` 的核子 2pt 信噪比 ``|mean|/std``（bootstrap 轴在前）。

    Args:
        c2_samples: ``(Nsample, Ndt)`` bootstrap 样本（实数或复数）。
    Returns:
        ``(Ndt,)`` 的 S/N；``std == 0`` 记为 ``inf``。
    """
    c2 = np.asarray(c2_samples)
    if c2.ndim != 2:
        raise ValueError("c2_samples 必须为 (Nsample, Ndt) 二维数组")
    if c2.shape[0] < 2:
        raise ValueError("c2_samples 至少需要两个 bootstrap 样本")
    if not np.all(np.isfinite(c2)):
        raise ValueError("c2_samples 必须全部有限")
    mean = np.mean(c2, axis=0)
    std = np.std(c2, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(std > 0.0, np.abs(mean) / std, np.inf)
    return snr


def select_plateau_window(ratio, c2_samples=None, *, dt_min=3, dt_max=None,
                          span=3, snr_min=3.0, cut=6,
                          max_error_growth=None):
    """由 C2 信噪比horizon选择 ``t_sep`` 平台窗口（非挑窗）。

    物理依据：ratio 的分母是核子 2pt。一旦 ``C2`` 进入噪声区，``C3/C2``
    的方差按 ``1/|C2|^2`` 发散，任何"平台"都只是噪声放大。因此窗口上界
    取 C2 仍满足 ``|mean|/std >= snr_min`` 的最大 ``t_sep``（horizon），
    下界取 ``horizon - span + 1``；这样窗口只由分母数据决定，不用 ratio
    自身挑最优值（避免 look-elsewhere）。

    Args:
        ratio: ``(Nsample, dt_max, dtau_max, nz, nb)`` 不相连 ratio。
        c2_samples: 可选 ``(Nsample, Ndt)`` 的 C2 bootstrap 样本；省略时
            退回 ``[dt_min, dt_max]`` 全区间并标记 ``c2_horizon_used=False``。
        dt_min: 窗口下界下限（默认 3，避开接触项）。
        dt_max: 候选上界；默认为 ``ratio.shape[1] - 1``。
        span: 窗口宽度（``t_sep`` 点数）。
        snr_min: C2 horizon 的信噪比阈值。
        cut: 与 :func:`plateau_c0` 相同的插入时间前后留空。
        max_error_growth: 可选正数。给出时要求平台均值的 bootstrap 误差
            ``sigma(t_sep) <= max_error_growth * sigma(dt_min)``；这直接
            拦住 ``1/|C2|^2`` 型方差爆炸（分母进入噪声后 ratio 误差量级
            跳到 10^2-10^3）。
    Returns:
        字典：``tsep_start``、``tsep_end``、``horizon``、``c2_snr``、
        ``c2_horizon_used``、``reason``、``n_points``、``ratio_error``。
    """
    r = np.asarray(ratio)
    if r.ndim != 5:
        raise ValueError("ratio 必须为 (Nsample, dt, dtau, nz, nb) 五维数组")
    if r.shape[0] < 2 or r.shape[1] < 2:
        raise ValueError("ratio 的 sample/dt 轴至少为 2")
    dt_top = r.shape[1] - 1 if dt_max is None else int(dt_max)
    dt_min = int(dt_min)
    span = int(span)
    if span < 1:
        raise ValueError("span 必须为正整数")
    if dt_min < 0 or dt_top < dt_min:
        raise ValueError(f"非法 dt 区间: [{dt_min}, {dt_top}]")
    snr_min = float(snr_min)
    if not np.isfinite(snr_min) or snr_min < 0.0:
        raise ValueError("snr_min 必须为非负有限数")

    if max_error_growth is not None:
        max_error_growth = float(max_error_growth)
        if not np.isfinite(max_error_growth) or max_error_growth <= 0.0:
            raise ValueError("max_error_growth 必须为正有限数")

    front = cut // 2
    back = cut - front
    ratio_error = np.full(dt_top + 1, np.nan, dtype=np.float64)
    for dt in range(dt_min, dt_top + 1):
        lo, hi = front, dt - back
        if hi < lo:
            continue
        sample = r[:, dt, lo:hi + 1].reshape(r.shape[0], -1)
        sample = sample[np.all(np.isfinite(sample), axis=1)]
        if sample.shape[0] < 2:
            continue
        ratio_error[dt] = float(np.mean(np.std(sample, axis=0, ddof=1)))

    growth_ok = np.ones(dt_top + 1, dtype=bool)
    if max_error_growth is not None:
        reference = ratio_error[dt_min]
        if not np.isfinite(reference) or reference <= 0.0:
            finite = ratio_error[np.isfinite(ratio_error)]
            reference = float(np.min(finite)) if finite.size else np.nan
        if np.isfinite(reference) and reference > 0.0:
            growth_ok = ratio_error <= max_error_growth * reference
        else:
            growth_ok = np.zeros(dt_top + 1, dtype=bool)
            growth_ok[dt_min] = True

    if c2_samples is None:
        snr = None
        used = False
        c2_ok = np.ones(dt_top + 1, dtype=bool)
        c2_reason = "no_c2_samples_full_range"
    else:
        snr = c2_signal_to_noise(c2_samples)
        if snr.shape[0] <= dt_top:
            raise ValueError(
                f"c2_samples 的 dt 轴 ({snr.shape[0]}) 短于请求的 dt_max ({dt_top})")
        c2_ok = snr >= snr_min
        used = True
        c2_reason = "c2_horizon"

    candidates = [dt for dt in range(dt_min, dt_top + 1)
                  if c2_ok[dt] and growth_ok[dt]]
    if candidates:
        horizon = max(candidates)
        reason = c2_reason
    else:
        horizon = dt_min
        reason = ("c2_below_threshold_everywhere" if used
                  else "ratio_error_growth_rejected_everywhere")
    end = min(dt_top, horizon)
    start = max(dt_min, end - span + 1)
    if max_error_growth is not None:
        reason += "_growth_gate"

    points = _plateau_points(r, dt_top, start, end, cut)
    if not points and start > dt_min:
        start = dt_min
        points = _plateau_points(r, dt_top, start, end, cut)
        reason += "_window_widened_to_dt_min"
    return {
        "tsep_start": int(start),
        "tsep_end": int(end),
        "horizon": int(horizon),
        "c2_snr": None if snr is None else np.asarray(snr),
        "c2_horizon_used": bool(used),
        "reason": reason,
        "n_points": len(points),
        "ratio_error": ratio_error,
    }


def plateau_c0(ratio, dt_max=20, dt_start=7, dt_end=10, cut=6):
    """fit 窗口内 ratio 的 plateau 均值 → c0(z,b)（抗奇异协方差）。

    整合自原 TMD 工作流的 plateau 均值实现：
    与 run_disconnected_tmd_ratio 的 x_coor 窗口一致
    （dt∈[dt_start,dt_end]，dtau∈[front,dt-back]），窗口内直接平均。
    10 组态统计下比 lsqfit 逐样本拟合更稳健（协方差奇异的绕行方案）。

    Args:
        ratio: (Nsample, dt_max, dtau_max, nz, nb)——不相连比值数组。
    Returns:
        (Nsample, nz, nb) plateau 均值。
    """
    r = np.asarray(ratio)
    Nsample, _, _, nz, nb = r.shape
    dtype = np.result_type(r.dtype, np.float64)
    if Nsample < 2:
        return np.full((Nsample, nz, nb), np.nan, dtype=dtype)
    points = _plateau_points(r, dt_max, dt_start, dt_end, cut)
    if not points:
        return np.full((Nsample, nz, nb), np.nan, dtype=dtype)

    window = np.stack([r[:, dt, dtau, :, :] for dt, dtau in points], axis=1)
    out = np.mean(window, axis=1, dtype=dtype)
    flat = out.reshape(Nsample, -1)
    finite = np.all(np.isfinite(flat), axis=0)
    centered = flat - np.mean(flat, axis=0)
    fluctuating = np.sum(np.abs(centered) ** 2, axis=0) > 0.0
    flat[:, ~(finite & fluctuating)] = np.nan
    return out


def plot_tmd_pdf(x_grid, xg_quasi, xg_matched, b_grid_fm, cs_kernel,
                 tag, out_dir, logger=print, diagnostic=False):
    """TMD-PDF 链成图（整合自 test9 示例 plot_pdf）。

    产出：quasi_tmd_pdf.png / matched_tmd_pdf.png / tmd_pdf_vs_b.png /
    cs_kernel.png（cs_kernel 非 None 时）。

    Args:
        x_grid: x 网格。
        xg_quasi: 准 TMD-PDF x·g̃(x, b⊥)，(nx, nb) 或 (nx,)。
        xg_matched: NLO 匹配后 x·g(x, b⊥)。
        b_grid_fm: b⊥ 网格（fm，标记用）。
        cs_kernel: K(b⊥) 数组或 None。
        tag: 动量标签（图题）。
        out_dir: 输出目录。
    Returns:
        生成的 png 路径列表。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    xg = np.asarray(xg_quasi)
    xgm = np.asarray(xg_matched)
    if xg.ndim == 1:
        xg = xg[:, None]
    if xgm.ndim == 1:
        xgm = xgm[:, None]
    nb = xg.shape[1]
    ncol = 2
    nrow = (nb + 1) // 2
    saved = []
    prefix = 'diagnostic_' if diagnostic else ''
    matched_label = (
        'one-loop gg-only diagnostic' if diagnostic else 'NLO matched')

    # 准 TMD-PDF x·g̃(x, b⊥)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                             squeeze=False)
    for b in range(nb):
        ax = axes[b // ncol][b % ncol]
        ax.plot(x_grid, xg[:, b], 'o-', ms=3)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel(r'$x\tilde{g}(x,b_\perp)$')
        ax.set_title(f'quasi TMD-PDF, b={b_grid_fm[b]:.3f} fm')
        ax.grid(alpha=0.3)
    for b in range(nb, nrow * ncol):
        axes[b // ncol][b % ncol].axis('off')
    title_prefix = 'DIAGNOSTIC ' if diagnostic else ''
    fig.suptitle(f'{title_prefix}{tag}: gradient-flow quasi gluon TMD-PDF')
    fig.tight_layout()
    p = os.path.join(out_dir, f'{prefix}quasi_tmd_pdf.png')
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # 匹配前后对比
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                             squeeze=False)
    for b in range(nb):
        ax = axes[b // ncol][b % ncol]
        ax.plot(x_grid, xg[:, b], 'o-', ms=3, label='quasi')
        ax.plot(x_grid, xgm[:, b], 's-', ms=3, label=matched_label)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel(r'$xg(x,b_\perp)$')
        ax.set_title(f'b={b_grid_fm[b]:.3f} fm')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for b in range(nb, nrow * ncol):
        axes[b // ncol][b % ncol].axis('off')
    fig.suptitle(f'{title_prefix}{tag}: {matched_label} gluon TMD-PDF')
    fig.tight_layout()
    p = os.path.join(out_dir, f'{prefix}matched_tmd_pdf.png')
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # 所有 b 叠加
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in range(nb):
        ax.plot(x_grid, xgm[:, b], '-', lw=1.2,
                label=f'b={b_grid_fm[b]:.3f} fm')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xlabel('x'); ax.set_ylabel(r'$xg(x,b_\perp)$')
    ax.set_title(f'{title_prefix}{tag}: gluon TMD-PDF vs b-perp '
                 f'({matched_label})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f'{prefix}tmd_pdf_vs_b.png')
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # CS 核
    if cs_kernel is not None:
        K = np.asarray(cs_kernel, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(b_grid_fm[:len(K)], K, fmt='o-', capsize=3)
        ax.set_xlabel(r'$b_\perp$ [fm]')
        ax.set_ylabel(r'$K(b_\perp)$')
        ax.set_title(f'{title_prefix}Collins-Soper kernel '
                     '(two-momentum ratio)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out_dir, f'{prefix}cs_kernel.png')
        fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    logger(f"  TMD-PDF plots -> {out_dir} ({len(saved)} files)")
    return saved


def plot_tmd_ratio(ch_results, run_dir, momentum, logger=print,
                   nz=24, nb=1):
    """R(dt,dtau,z,b) 若干 (z,b) 点的比值曲线。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    for had_name, res in ch_results.items():
        ratio = res['ratio']     # (Nsample, dt, dtau, nz, nb)
        jackknife = res.get('resampling_kind') == 'jackknife'
        rm = ratio.mean(0); re_ = sem(ratio, jackknife)
        zs = [0, max(0, nz // 4), max(0, nz // 2), max(0, nz - 1)]
        bs = list(range(min(nb, 2)))
        fig, axes = plt.subplots(len(bs), len(zs),
                                 figsize=(4 * len(zs), 3.5 * len(bs)),
                                 squeeze=False)
        for _b in range(len(bs)):
            for _z in range(len(zs)):
                zz, bb = zs[_z], bs[_b]
                ax = axes[_b][_z]
                for dt in [8, 10, 12, 14]:
                    if dt >= rm.shape[0]:
                        continue
                    tau = np.arange(dt + 1)
                    xv = tau - dt / 2
                    yv = rm[dt, :dt + 1, zz, bb]
                    ye = re_[dt, :dt + 1, zz, bb]
                    ax.errorbar(xv, yv, yerr=ye, fmt='x', capsize=0, label=f'dt={dt}')
                ax.set_xlabel('tau - t_sep/2'); ax.set_ylabel('R')
                ax.set_title(f'z={zz} b⊥={bb}')
                ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle(f'{had_name} {momentum}: Disconnected TMD ratio')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'ratio_{had_name}_{momentum}.png'), dpi=150)
        plt.close(fig)
    logger(f"  TMD ratio plots saved to {out_dir}")
