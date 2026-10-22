"""Forward Sumratio-difference plateau fits with replica-local AIC selection.

The ratio input is already projected and vacuum subtracted.  This module only
implements the bare-matrix fit contract: a physical Tmin, per-cut AIC windows,
replica-dependent selection, and a separately labelled minimum-error fallback.
"""
from __future__ import annotations

import numpy as np


def physical_tmin(a_fm, threshold_fm=0.6):
    """Return the smallest integer T with ``T*a_fm > threshold_fm``."""
    a_fm = float(a_fm)
    threshold_fm = float(threshold_fm)
    if not np.isfinite(a_fm) or a_fm <= 0.0:
        raise ValueError("a_fm must be finite and positive")
    if not np.isfinite(threshold_fm) or threshold_fm <= 0.0:
        raise ValueError("threshold_fm must be finite and positive")
    return int(np.floor(threshold_fm / a_fm) + 1)


def sumratio_values(ratio, cut, tseps, tsep_axis=0):
    """Sum ``R(T,tau)`` over ``cut <= tau <= T-cut`` for each T."""
    values = np.asarray(ratio)
    if values.ndim < 2:
        raise ValueError("ratio must have at least (tsep,tau) axes")
    if isinstance(tsep_axis, (bool, np.bool_)) or not isinstance(
            tsep_axis, (int, np.integer)):
        raise ValueError("tsep_axis must be a non-negative integer")
    tsep_axis = int(tsep_axis)
    if tsep_axis < 0 or tsep_axis + 1 >= values.ndim:
        raise ValueError("tsep_axis or its tau axis is out of bounds")
    if isinstance(cut, (bool, np.bool_)) or not isinstance(
            cut, (int, np.integer)) or cut < 0:
        raise ValueError("cut must be a non-negative integer")
    tseps = np.asarray(tseps)
    if tseps.ndim != 1 or tseps.size == 0:
        raise ValueError("tseps must be a non-empty one-dimensional array")
    if tseps.dtype.kind not in "iu" or np.any(tseps < 0):
        raise ValueError("tseps must contain non-negative integers")
    dtype = np.result_type(values.dtype, np.float64)
    output_tsep_size = int(np.max(tseps)) + 1
    if tsep_axis == 0:
        out = np.full(
            (output_tsep_size,) + values.shape[2:], np.nan, dtype=dtype)
    elif tsep_axis == 1:
        out = np.full(
            (values.shape[0], output_tsep_size) + values.shape[3:],
            np.nan, dtype=dtype)
    else:
        raise ValueError("only tsep_axis=0 or 1 is supported")
    for index, tsep in enumerate(tseps):
        upper = int(tsep) - int(cut)
        if int(tsep) < 2 * int(cut) or upper >= values.shape[tsep_axis]:
            continue
        if tsep_axis == 0:
            out[int(tsep)] = np.sum(
                values[int(tsep), int(cut):upper + 1], axis=0)
        else:
            out[:, int(tsep)] = np.sum(
                values[:, int(tsep), int(cut):upper + 1], axis=1)
    return out


def forward_difference(sumratio, tmin, tmax, tsep_axis=0):
    """Return ``D(T)=S(T+1)-S(T)`` for ``T=tmin..tmax-1``."""
    values = np.asarray(sumratio)
    tmin = int(tmin)
    tmax = int(tmax)
    tsep_axis = int(tsep_axis)
    if tmin < 0 or tmax <= tmin:
        raise ValueError("require 0 <= tmin < tmax")
    if tsep_axis < 0 or tsep_axis >= values.ndim:
        raise ValueError("tsep_axis is out of bounds")
    if tmax >= values.shape[tsep_axis]:
        raise ValueError("tmax exceeds the summed-ratio tsep axis")
    if tsep_axis == 0:
        return values[tmin + 1:tmax + 1] - values[tmin:tmax]
    if tsep_axis == 1:
        return values[:, tmin + 1:tmax + 1] - values[:, tmin:tmax]
    raise ValueError("only tsep_axis=0 or 1 is supported")


def _schaefer_strimmer_covariance(samples, *, relative_jitter=1e-12,
                                  absolute_jitter=1e-18):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("samples must have shape (n>=2,ndata)")
    if not np.isfinite(values).all():
        raise ValueError("samples must be finite")
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / (values.shape[0] - 1)
    diagonal = np.diag(covariance).copy()
    if np.any(diagonal <= 0.0):
        raise ValueError("every candidate must have positive sample variance")
    scale = np.sqrt(diagonal)
    correlation = covariance / scale[:, None] / scale[None, :]
    correlation = (correlation + correlation.T) * 0.5
    off_diagonal = correlation - np.eye(correlation.shape[0])
    numerator = float(np.sum(((1.0 - off_diagonal ** 2) ** 2)
                             / max(values.shape[0] - 1, 1)))
    denominator = float(np.sum(off_diagonal ** 2))
    shrinkage = 0.0 if denominator == 0.0 else min(
        1.0, max(0.0, numerator / denominator))
    shrunk = (1.0 - shrinkage) * correlation
    shrunk += shrinkage * np.eye(correlation.shape[0])
    covariance = shrunk * scale[:, None] * scale[None, :]
    covariance += np.eye(covariance.shape[0]) * (
        relative_jitter * max(float(np.max(np.abs(diagonal))), 1.0)
        + absolute_jitter)
    return covariance, shrinkage


def _constant_gls(samples, central, *, svdcut=1e-12,
                  relative_jitter=1e-12, absolute_jitter=1e-18):
    samples = np.asarray(samples, dtype=float)
    central = np.asarray(central, dtype=float)
    if samples.ndim != 2 or central.shape != (samples.shape[1],):
        raise ValueError("samples/central axes do not match")
    if not np.isfinite(samples).all() or not np.isfinite(central).all():
        raise ValueError("constant GLS requires finite samples and central data")
    covariance, shrinkage = _schaefer_strimmer_covariance(
        samples, relative_jitter=relative_jitter,
        absolute_jitter=absolute_jitter)
    eig, vec = np.linalg.eigh(covariance)
    if eig[0] < -1e-10 * max(float(eig[-1]), 1.0):
        raise ValueError("candidate covariance is not positive semidefinite")
    keep = eig > float(svdcut) * max(float(eig[-1]), 0.0)
    rank = int(np.count_nonzero(keep))
    if rank < 1:
        raise ValueError("regularized covariance has no retained mode")
    inverse = (vec[:, keep] / eig[keep]) @ vec[:, keep].T
    one = np.ones(central.size, dtype=float)
    weights = inverse @ one
    denominator = float(one @ weights)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("constant GLS normal matrix is invalid")
    fit_weights = weights / denominator
    values_boot = samples @ fit_weights
    residual = central - float(central @ fit_weights)
    chi2 = float(residual @ inverse @ residual)
    dof = rank - 1
    return {
        "value": float(central @ fit_weights),
        "value_boot": values_boot,
        "chi2": chi2,
        "chi2_boot": (
            np.einsum("bi,ij,bj->b", samples - values_boot[:, None],
                      inverse, samples - values_boot[:, None], optimize=True)
            if dof > 0 else np.full(samples.shape[0], np.nan)),
        "dof": dof,
        "sample_rank": int(np.linalg.matrix_rank(
            samples - samples.mean(axis=0), tol=1e-12)),
        "effective_rank": rank,
        "condition_number": (
            float(eig[keep][-1] / eig[keep][0])
            if np.count_nonzero(keep) > 1 else np.inf),
        "shrinkage": shrinkage,
    }


def _q_value(chi2, dof):
    values = np.asarray(chi2, dtype=float)
    dof_values = np.asarray(dof)
    values, dof_values = np.broadcast_arrays(values, dof_values)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values) & (dof_values > 0)
    if not np.any(finite):
        return result[()] if result.ndim == 0 else result
    from scipy.stats import chi2 as chi2_distribution
    result[finite] = chi2_distribution.sf(
        values[finite], dof_values[finite])
    return result[()] if result.ndim == 0 else result


def _row_quality(chi2, dof, q_min, chi2_dof_max):
    chi2 = np.asarray(chi2, dtype=float)
    dof = np.asarray(dof)
    q_value = np.asarray(_q_value(chi2, dof), dtype=float)
    return (np.isfinite(chi2) & (dof > 0)
            & (q_value >= q_min)
            & (chi2 / np.maximum(dof, 1) <= chi2_dof_max)), q_value


def _weighted_average(values, scores, passing):
    values = np.asarray(values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    passing = np.asarray(passing, dtype=bool)
    output = np.full(values.shape[:-1], np.nan)
    weights = np.zeros(values.shape, dtype=float)
    has = passing.any(axis=-1)
    rows = np.flatnonzero(has)
    if rows.size:
        local = np.where(passing[rows], scores[rows], np.inf)
        shifted = local - np.min(local, axis=1, keepdims=True)
        candidate_weights = np.exp(-0.5 * shifted)
        candidate_weights[~passing[rows]] = 0.0
        normalizer = np.sum(candidate_weights, axis=1, keepdims=True)
        weights[rows] = candidate_weights / normalizer
        output[rows] = np.sum(
            weights[rows] * np.where(passing[rows], values[rows], 0.0),
            axis=1)
    return output, weights, has


def fit_forward_sumratio_difference(ratio_samples, ratio_original, *,
                                    tmin, cuts=(1, 2, 3), tsep_max=15,
                                    q_min=0.01, chi2_dof_max=3.0,
                                    bootstrap_min_success=0.90,
                                    svdcut=1e-12,
                                    allow_zero_dof=False):
    """Fit all registered Sumratio-difference plateau candidates.

    ``ratio_samples`` has axes ``(replica,tsep,tau,...)`` and
    ``ratio_original`` has ``(tsep,tau,...)``.  Candidate arrays append a
    leading ``(cut,tmax)`` pair of axes; trailing axes are copied from the
    ratio observable.
    """
    samples = np.asarray(ratio_samples)
    original = np.asarray(ratio_original)
    if samples.ndim < 3 or original.shape != samples.shape[1:]:
        raise ValueError("ratio sample/original axes do not match")
    tmin = int(tmin)
    tsep_max = int(tsep_max)
    if tmin < 0 or tsep_max <= tmin + 1:
        raise ValueError("tsep_max must leave at least one difference")
    if tsep_max >= samples.shape[1]:
        raise ValueError("tsep_max exceeds the available tsep axis")
    cuts = tuple(sorted({int(cut) for cut in cuts}))
    if not cuts or cuts[0] < 0:
        raise ValueError("cuts must be a non-empty set of non-negative integers")
    tmax_values = np.arange(tmin + 1, tsep_max + 1, dtype=np.int64)
    max_tmax = int(tmax_values[-1])
    tseps = np.arange(tmin, tsep_max + 1, dtype=np.int64)
    trailing = samples.shape[3:]
    candidate_shape = (len(cuts), tmax_values.size) + trailing
    nboot = samples.shape[0]
    candidate_value = np.full(candidate_shape, np.nan)
    candidate_boot = np.full((nboot,) + candidate_shape, np.nan)
    candidate_chi2 = np.full(candidate_shape, np.nan)
    candidate_chi2_boot = np.full((nboot,) + candidate_shape, np.nan)
    candidate_dof = np.zeros(candidate_shape, dtype=np.int64)
    candidate_q = np.full(candidate_shape, np.nan)
    candidate_chi2_dof = np.full(candidate_shape, np.nan)
    candidate_success = np.zeros(candidate_shape)
    candidate_sample_rank = np.zeros(candidate_shape, dtype=np.int64)
    candidate_rank = np.zeros(candidate_shape, dtype=np.int64)
    candidate_cond = np.full(candidate_shape, np.nan)
    candidate_shrink = np.full(candidate_shape, np.nan)
    candidate_aic = np.full(candidate_shape, np.nan)
    candidate_aic_boot = np.full((nboot,) + candidate_shape, np.nan)
    candidate_eligible = np.zeros(candidate_shape, dtype=bool)
    candidate_weight = np.zeros(candidate_shape)
    candidate_weight_boot = np.zeros((nboot,) + candidate_shape)
    candidate_reason = np.full(candidate_shape, "not_evaluated", dtype="U96")

    sample_sum = {
        cut: sumratio_values(samples, cut, tseps, tsep_axis=1)
        for cut in cuts
    }
    original_sum = {
        cut: sumratio_values(original, cut, tseps) for cut in cuts
    }
    for cut_index, cut in enumerate(cuts):
        for tmax_index, tmax in enumerate(tmax_values):
            n_data = int(tmax - tmin)
            if n_data < 2 and not allow_zero_dof:
                candidate_reason[cut_index, tmax_index] = "zero_dof_not_allowed"
                continue
            try:
                sample_diff = forward_difference(
                    sample_sum[cut], tmin, tmax, tsep_axis=1)
                original_diff = forward_difference(original_sum[cut], tmin, tmax)
                flat_sample = sample_diff.reshape(nboot, n_data, -1)
                flat_original = original_diff.reshape(n_data, -1)
            except ValueError as exc:
                candidate_reason[cut_index, tmax_index] = str(exc)
                continue
            omission = max_tmax - int(tmax)
            aic_penalty = 2.0 + 2.0 * omission
            for channel in range(flat_sample.shape[-1]):
                target = (cut_index, tmax_index)
                try:
                    fit = _constant_gls(
                        flat_sample[:, :, channel], flat_original[:, channel],
                        svdcut=svdcut)
                except (ValueError, np.linalg.LinAlgError,
                        FloatingPointError) as exc:
                    candidate_reason[target + np.unravel_index(
                        channel, trailing)] = (
                            f"{type(exc).__name__}:{exc}")
                    continue
                channel_target = target + np.unravel_index(channel, trailing)
                passing, q_value = _row_quality(
                    fit["chi2"], fit["dof"], q_min, chi2_dof_max)
                replica_passing, _ = _row_quality(
                    fit["chi2_boot"], fit["dof"], q_min, chi2_dof_max)
                success = float(np.mean(replica_passing))
                eligible = bool(passing and success >= bootstrap_min_success)
                candidate_value[channel_target] = fit["value"]
                candidate_boot[(slice(None),) + channel_target] = fit["value_boot"]
                candidate_chi2[channel_target] = fit["chi2"]
                candidate_chi2_boot[(slice(None),) + channel_target] = fit["chi2_boot"]
                candidate_dof[channel_target] = fit["dof"]
                candidate_q[channel_target] = q_value
                candidate_chi2_dof[channel_target] = (
                    fit["chi2"] / fit["dof"] if fit["dof"] > 0 else np.nan)
                candidate_success[channel_target] = success
                candidate_sample_rank[channel_target] = fit["sample_rank"]
                candidate_rank[channel_target] = fit["effective_rank"]
                candidate_cond[channel_target] = fit["condition_number"]
                candidate_shrink[channel_target] = fit["shrinkage"]
                candidate_aic[channel_target] = fit["chi2"] + aic_penalty
                candidate_aic_boot[(slice(None),) + channel_target] = (
                    fit["chi2_boot"] + aic_penalty)
                candidate_eligible[channel_target] = eligible
                failures = []
                if not passing:
                    failures.append("central_q_or_chi2")
                if success < bootstrap_min_success:
                    failures.append("bootstrap_success")
                candidate_reason[channel_target] = (
                    "accepted" if eligible else ";".join(failures))

    shape_selected_boot = (nboot,) + trailing
    selected = np.full(trailing, np.nan)
    selected_boot = np.full(shape_selected_boot, np.nan)
    quality = np.zeros(trailing, dtype=bool)
    quality_boot = np.zeros(shape_selected_boot, dtype=bool)
    fallback = np.zeros((nboot + 1,) + trailing, dtype=bool)
    fallback_status = np.full((nboot + 1,) + trailing, "no_finite_candidate",
                              dtype="U80")
    fallback_index = np.full((nboot + 1,) + trailing, -1, dtype=np.int64)
    cut_count = np.zeros((nboot + 1,) + trailing, dtype=np.int64)
    candidate_error = np.full(candidate_shape, np.nan)
    for cut_index in range(candidate_shape[0]):
        vals = candidate_boot[:, cut_index]
        for window_index in range(candidate_shape[1]):
            for channel_index in np.ndindex(*trailing):
                series = vals[(slice(None), window_index) + channel_index]
                finite = series[np.isfinite(series)]
                if finite.size >= 2:
                    target = (cut_index, window_index) + channel_index
                    candidate_error[target] = np.std(finite, ddof=1)

    for channel_index in np.ndindex(*trailing):
        cut_quality_values = []
        for cut_index in range(candidate_shape[0]):
            window_slice = (cut_index, slice(None)) + channel_index
            boot_window_slice = (
                slice(None), cut_index, slice(None)) + channel_index
            values = candidate_value[window_slice]
            boots = candidate_boot[boot_window_slice]
            scores = candidate_aic[window_slice]
            scores_boot = candidate_aic_boot[boot_window_slice]
            eligible = candidate_eligible[window_slice]
            replica_q, _ = _row_quality(
                candidate_chi2_boot[boot_window_slice],
                candidate_dof[window_slice],
                q_min, chi2_dof_max)
            eligible_boot = (
                np.isfinite(boots) & np.isfinite(scores_boot)
                & (candidate_dof[window_slice][None, :] > 0)
                & replica_q)
            central_average, central_weights, central_has = _weighted_average(
                values[None, :], scores[None, :], eligible[None, :])
            boot_average, boot_weights, boot_has = _weighted_average(
                boots, scores_boot, eligible_boot)
            candidate_weight[window_slice] = central_weights[0]
            candidate_weight_boot[boot_window_slice] = boot_weights
            cut_quality_values.append({
                "central": central_average[0],
                "central_has": bool(central_has[0]),
                "boot": boot_average,
                "boot_has": boot_has,
            })

        # Central and replica selection are independent.  A failed central
        # candidate must not suppress a cut that is usable in a replica.
        usable_central = [
            item["central"] for item in cut_quality_values
            if item["central_has"] and np.isfinite(item["central"])
        ]
        if usable_central:
            selected[channel_index] = np.mean(usable_central)
            quality[channel_index] = True
            cut_count[(0,) + channel_index] = len(usable_central)
            fallback_status[(0,) + channel_index] = "quality_aic"
        else:
            fallback_status[(0,) + channel_index] = "no_finite_candidate"

        for replica in range(nboot):
            usable_replica = [
                item["boot"][replica] for item in cut_quality_values
                if item["boot_has"][replica]
                and np.isfinite(item["boot"][replica])
            ]
            if usable_replica:
                selected_boot[(replica,) + channel_index] = np.mean(
                    usable_replica)
                quality_boot[(replica,) + channel_index] = True
                cut_count[(replica + 1,) + channel_index] = len(
                    usable_replica)
                fallback_status[(replica + 1,) + channel_index] = "quality_aic"
            else:
                fallback_status[(replica + 1,) + channel_index] = (
                    "no_finite_candidate")

        rows = [0] if not quality[channel_index] else []
        rows.extend(np.flatnonzero(~quality_boot[(slice(None),) + channel_index]) + 1)
        for row in rows:
            candidates = []
            for cut_index in range(candidate_shape[0]):
                for window in range(candidate_shape[1]):
                    target = (cut_index, window) + channel_index
                    if candidate_dof[target] <= 0 and not allow_zero_dof:
                        continue
                    value = (candidate_value[target] if row == 0 else
                             candidate_boot[(row - 1, cut_index, window) + channel_index])
                    if np.isfinite(value):
                        candidates.append((
                            candidate_error[target], candidate_chi2[target],
                            cut_index, window))
            if not candidates:
                continue
            with_error = [item for item in candidates if np.isfinite(item[0])]
            pool = with_error or candidates
            chosen = min(pool, key=lambda item: (item[0], item[1], item[2], item[3]))
            cut_index, window = chosen[2], chosen[3]
            target = (cut_index, window) + channel_index
            if row == 0:
                selected[channel_index] = candidate_value[target]
            else:
                selected_boot[(row - 1,) + channel_index] = candidate_boot[
                    (row - 1, cut_index, window) + channel_index]
            fallback[(row,) + channel_index] = True
            fallback_index[(row,) + channel_index] = (
                cut_index * candidate_shape[1] + window)
            label = ("minimum_candidate_statistical_error"
                     if with_error else "minimum_chi2_no_finite_error")
            if candidate_dof[target] <= 0:
                label = "zero_dof_" + label
            fallback_status[(row,) + channel_index] = label

    available = np.isfinite(selected)
    boot_available = np.isfinite(selected_boot)
    complete_boot = np.all(boot_available)
    statistical_error = (
        np.std(selected_boot, axis=0, ddof=1)
        if complete_boot else np.full(trailing, np.nan))
    return {
        "M": selected,
        "M_boot": selected_boot,
        "M_quality": np.where(quality, selected, np.nan),
        "M_boot_quality": np.where(
            quality_boot, selected_boot, np.nan),
        "M_filled": selected,
        "M_boot_filled": selected_boot,
        "statistical_error": statistical_error,
        "filled_statistical_error": statistical_error,
        "quality_accepted": quality,
        "bootstrap_quality_accepted": quality_boot,
        "estimate_available": available & complete_boot,
        "fallback_used": fallback,
        "fallback_status": fallback_status,
        "fallback_candidate_index": fallback_index,
        "central_fallback_used": fallback[0],
        "central_fallback_status": fallback_status[0],
        "replica_fallback_used": fallback[1:],
        "replica_fallback_fraction": np.mean(fallback[1:], axis=0),
        "usable_cut_count": cut_count,
        "candidate_M": candidate_value,
        "candidate_M_boot": candidate_boot,
        "candidate_chi2": candidate_chi2,
        "candidate_chi2_boot": candidate_chi2_boot,
        "candidate_dof": candidate_dof,
        "candidate_q_value": candidate_q,
        "candidate_chi2_dof": candidate_chi2_dof,
        "candidate_bootstrap_success_fraction": candidate_success,
        "candidate_covariance_sample_rank": candidate_sample_rank,
        "candidate_covariance_rank": candidate_rank,
        "candidate_covariance_condition": candidate_cond,
        "candidate_covariance_shrinkage": candidate_shrink,
        "candidate_range_aic": candidate_aic,
        "candidate_range_aic_boot": candidate_aic_boot,
        "candidate_eligible": candidate_eligible,
        "candidate_aic_weight": candidate_weight,
        "candidate_aic_weight_boot": candidate_weight_boot,
        "candidate_failure_reason": candidate_reason,
        "candidate_statistical_error": candidate_error,
        "candidate_cut": np.asarray(cuts, dtype=np.int64),
        "candidate_tmax": tmax_values,
        "tmin": np.asarray(tmin, dtype=np.int64),
        "tsep_max": np.asarray(tsep_max, dtype=np.int64),
        "quality_thresholds": np.asarray(
            f"Q>={q_min};chi2/dof<={chi2_dof_max};"
            f"bootstrap_success>={bootstrap_min_success}"),
    }


__all__ = [
    "physical_tmin", "sumratio_values", "forward_difference",
    "fit_forward_sumratio_difference",
]
