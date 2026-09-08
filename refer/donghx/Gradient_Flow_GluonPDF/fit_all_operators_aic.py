#!/usr/bin/env python3
"""Fit all flowed-gluon operator choices with the registered AIC plateau.

The input products already contain the summed ratios ``S_c(T)``.  This
analysis therefore constructs only the registered forward differences

    D_c(T) = S_c(T+1) - S_c(T),

and applies the strict-0.6-fm, correlated constant-fit/AIC procedure used by
the ``fit-bare`` skill.  The six operator choices are kept on one explicit
operator axis:

    U: T_U, T_U+S_U, T_U-S_U
    H: T_H, T_H+S_H, T_H-S_H

The source products use Nboot=1000 (the requested all-operator product), so
this script preserves that count instead of silently regenerating a different
resampling ensemble.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_distribution


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR_DEFAULT = ROOT / "ratio_bootstrap_all_operators_v1" / "results_v1"
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "results_v1"

# Reuse the audited implementation of the registered Schaefer--Strimmer,
# correlated forward-difference AIC pipeline.  Only the product adapter and
# operator loop are local to this all-operator analysis.
FIT_CODE = Path(
    "/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/"
    "bare_fit_sumdiff_aic_code"
)
if str(FIT_CODE) not in sys.path:
    sys.path.insert(0, str(FIT_CODE))

from gluonfit.fit import ForwardDifference, fit_plateau_candidate  # noqa: E402
from gluonfit.model_average import (  # noqa: E402
    ReplicaModelAverage,
    fill_with_minimum_chi2,
)


SCHEMA = "gradient_flow_gluon_bare_sumdiff_aic_all_operators_v1"
INPUT_SCHEMA = "gradient_flow_gluon_c3_c2_bootstrap_all_operators_v1"
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)
OPERATOR_LABELS = (
    "unpolarized_TU_even",
    "unpolarized_TU_plus_SU_even",
    "unpolarized_TU_minus_SU_even",
    "helicity_TH_odd",
    "helicity_TH_plus_SH_odd",
    "helicity_TH_minus_SH_odd",
)
OPERATOR_DEFINITIONS = (
    "T_U = U_tx + U_ty; no S_U",
    "T_U + S_U, S_U = 2 U_xy",
    "T_U - S_U, S_U = 2 U_xy",
    "T_H = H_tx + H_ty; no S_H",
    "T_H + S_H, S_H = 2 H_xy",
    "T_H - S_H, S_H = 2 H_xy",
)
OPERATOR_CHANNEL = ("nopol", "nopol", "nopol", "pol35", "pol35", "pol35")
OPERATOR_PROJECTION = ("real", "real", "real", "imag", "imag", "imag")
PABS = np.asarray((3, 4, 5), dtype=np.int32)
Z = np.arange(25, dtype=np.int32)
CUTS = np.asarray((1, 2, 3), dtype=np.int32)
ALL_TSEPS = np.arange(5, 16, dtype=np.int32)
TMIN = 8
A_FM = 0.0775
MINIMUM_TSEP_FM = 0.6
SVDCUT = 1e-12
Q_MIN = 0.01
CHI2_DOF_MAX = 3.0
BOOTSTRAP_MIN_SUCCESS = 0.90
REFERENCE_TMAX = 15


class FitStatistics:
    """Attribute-compatible statistics contract for the shared fit module."""

    q_min = Q_MIN
    correlated_chi2_dof_max = CHI2_DOF_MAX
    bootstrap_min_success = BOOTSTRAP_MIN_SUCCESS
    svdcut = SVDCUT
    normal_rcond = 1e-12
    covariance_relative_jitter = 1e-12
    covariance_absolute_jitter = 1e-18
    covariance_shrinkage = "schaefer_strimmer_correlation_to_identity"


STATS = FitStatistics()


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def product_name(tau: float) -> str:
    return f"c3_c2_bootstrap_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def output_name(tau: float) -> str:
    return f"bare_matrix_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"Expected scalar field {key}")
    return value.item()


def _check_done(path: Path) -> dict:
    receipt_candidates = (
        path.with_suffix(".done.json"),
        Path(str(path) + ".done.json"),
    )
    done_path = next((candidate for candidate in receipt_candidates if candidate.is_file()), None)
    if not path.is_file() or done_path is None:
        raise FileNotFoundError(f"Missing product/receipt pair: {path}")
    receipt = json.loads(done_path.read_text(encoding="utf-8"))
    if receipt.get("status") not in ("complete", "complete_and_validated"):
        raise ValueError(f"Input receipt is not complete: {done_path}")
    expected = receipt.get("output_sha256")
    if expected and expected != sha256_file(path):
        raise ValueError(f"Input product hash disagrees with receipt: {path}")
    return receipt


def load_input(path: Path) -> dict:
    receipt = _check_done(path)
    product_sha = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        contract_json = str(scalar(data, "contract_json"))
        input_contract = json.loads(contract_json)
        if input_contract.get("schema") != INPUT_SCHEMA:
            raise ValueError(f"Wrong input schema: {path}")
        nconf = int(scalar(data, "nconf"))
        nboot = int(scalar(data, "nboot"))
        seed = int(scalar(data, "seed"))
        if (nconf, nboot, seed) != (406, 1000, 1115):
            raise ValueError(f"Wrong bootstrap contract in {path}: {(nconf, nboot, seed)}")
        tau = float(scalar(data, "flow_tau"))
        labels = tuple(np.asarray(data["all_operator_labels"]).astype(str).tolist())
        if labels != OPERATOR_LABELS:
            raise ValueError(f"Operator axis mismatch in {path}: {labels}")
        if tuple(np.asarray(data["cuts"], dtype=int).tolist()) != tuple(CUTS):
            raise ValueError(f"Cut axis mismatch in {path}")
        if tuple(np.asarray(data["tsep"], dtype=int).tolist()) != tuple(ALL_TSEPS):
            raise ValueError(f"Tsep axis mismatch in {path}")
        if tuple(np.asarray(data["bootstrap_indices"]).shape) != (1000, 406):
            raise ValueError(f"Bootstrap-index shape mismatch in {path}")

        central_raw = np.asarray(data["ratio_sum_original"], dtype=float)
        boot_raw = np.asarray(data["ratio_sum_bootstrap"], dtype=float)
        valid_mask = np.asarray(data["valid_cut_tsep_mask"], dtype=bool)
        expected_central = (6, 3, 25, 11, 3)
        expected_boot = (1000, 6, 3, 25, 11, 3)
        if central_raw.shape != expected_central or boot_raw.shape != expected_boot:
            raise ValueError(
                f"Summed-ratio shapes mismatch in {path}: "
                f"{central_raw.shape}, {boot_raw.shape}"
            )
        if valid_mask.shape != (3, 11):
            raise ValueError(f"Invalid summed-ratio mask shape in {path}")

        # Product axes: (operator,cut,z,tsep,p) and
        # (bootstrap,operator,cut,z,tsep,p).  Fit axes are
        # (operator,cut,p,z,tsep) and (bootstrap,operator,cut,p,z,tsep).
        central = np.transpose(central_raw, (0, 1, 4, 2, 3))
        bootstrap = np.transpose(boot_raw, (0, 1, 2, 5, 3, 4))
        valid_entries = np.broadcast_to(
            valid_mask[None, :, None, None, :], central.shape
        )
        if not np.isfinite(central[valid_entries]).all():
            raise ValueError(f"Non-finite valid central summed-ratio entries: {path}")
        valid_boot = np.broadcast_to(
            valid_mask[None, None, :, None, None, :], bootstrap.shape
        )
        if not np.isfinite(bootstrap[valid_boot]).all():
            raise ValueError(f"Non-finite valid bootstrap summed-ratio entries: {path}")

        indices = np.asarray(data["bootstrap_indices"], dtype=np.int64)
        manifest_sha = str(scalar(data, "manifest_sha256"))
        manifest_path = Path(str(input_contract["manifest"]))
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest recorded by input is missing: {manifest_path}")
        if sha256_file(manifest_path) != manifest_sha:
            raise ValueError(f"Manifest hash mismatch for {path}")
        rows = list(
            csv.DictReader(
                manifest_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"
            )
        )
        confs = np.asarray([str(row["conf_id"]) for row in rows])
        if confs.shape != (nconf,) or len(set(confs.tolist())) != nconf:
            raise ValueError(f"Manifest configuration axis mismatch for {path}")
        return {
            "tau": tau,
            "central": central,
            "bootstrap": bootstrap,
            "valid_cut_tsep_mask": valid_mask,
            "confs": confs,
            "bootstrap_indices": indices,
            "input_product_sha256": product_sha,
            "input_receipt": receipt,
            "manifest_sha256": manifest_sha,
            "contract_json": contract_json,
        }


def _candidate_arrays(noperator: int, ncandidate: int, nboot: int, npabs: int, nz: int):
    central_shape = (noperator, ncandidate, npabs, nz)
    boot_shape = (noperator, ncandidate, nboot, npabs, nz)
    return {
        "M": np.full(central_shape, np.nan),
        "M_boot": np.full(boot_shape, np.nan),
        "chi2": np.full(central_shape, np.nan),
        "dof": np.zeros(central_shape, dtype=np.int32),
        "q_value": np.full(central_shape, np.nan),
        "chi2_dof": np.full(central_shape, np.nan),
        "bootstrap_success_fraction": np.zeros(central_shape),
        "covariance_rank": np.zeros(central_shape, dtype=np.int32),
        "covariance_condition": np.full(central_shape, np.nan),
        "covariance_shrinkage": np.full(central_shape, np.nan),
        "range_aic": np.full(central_shape, np.nan),
        "range_aic_boot": np.full(boot_shape, np.nan),
        "chi2_boot": np.full(boot_shape, np.nan),
        "fit_succeeded": np.zeros(central_shape, dtype=bool),
        "eligible": np.zeros(central_shape, dtype=bool),
        "failure_reason": np.full(central_shape, "not_evaluated", dtype="U128"),
    }


def _normalized_aic_weights(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    finite = np.isfinite(values)
    result = np.zeros_like(values)
    if not np.any(finite):
        return result
    delta = values[finite] - np.min(values[finite])
    raw = np.exp(-0.5 * delta)
    result[finite] = raw / np.sum(raw)
    return result


def replica_model_average_with_replica_gates(
    candidate_M: np.ndarray,
    candidate_M_boot: np.ndarray,
    candidate_aic: np.ndarray,
    candidate_aic_boot: np.ndarray,
    candidate_chi2_boot: np.ndarray,
    candidate_dof: np.ndarray,
    candidate_eligible: np.ndarray,
    candidate_cuts: np.ndarray,
    registered_cuts: np.ndarray,
) -> ReplicaModelAverage:
    """AIC-average only candidates passing replica quality gates.

    Central eligibility fixes the registered candidate family.  In each
    bootstrap replica, however, the candidate pool is further restricted by
    finite values, positive dof, Q>=0.01, and chi2/dof<=3 before its AIC
    weights are normalized.  This is the complete-diagnostic rule in
    ``fit-bare``; replicas with no passing candidate stay NaN here and are
    filled by the separate minimum-chi2 fallback layer.
    """

    values = np.asarray(candidate_M, dtype=float)
    boot = np.asarray(candidate_M_boot, dtype=float)
    scores = np.asarray(candidate_aic, dtype=float)
    scores_boot = np.asarray(candidate_aic_boot, dtype=float)
    chi2_boot = np.asarray(candidate_chi2_boot, dtype=float)
    dof = np.asarray(candidate_dof, dtype=int)
    eligible = np.asarray(candidate_eligible, dtype=bool)
    cuts = np.asarray(candidate_cuts, dtype=int)
    unique_cuts = np.asarray(registered_cuts, dtype=int)
    ncandidate, npabs, ndz = values.shape
    nboot = boot.shape[1]
    if boot.shape != (ncandidate, nboot, npabs, ndz):
        raise ValueError("Invalid candidate bootstrap axes")
    ncut = len(unique_cuts)
    output_shape = (npabs, ndz)
    cut_shape = (ncut, npabs, ndz)
    cut_boot_shape = (ncut, nboot, npabs, ndz)
    M = np.full(output_shape, np.nan)
    M_boot = np.full((nboot, npabs, ndz), np.nan)
    stat = np.full(output_shape, np.nan)
    valid = np.zeros(output_shape, dtype=bool)
    reason = np.full(output_shape, "no_centrally_accepted_cut", dtype="U128")
    success = np.zeros(output_shape)
    boot_mean = np.full(output_shape, np.nan)
    boot_bias = np.full(output_shape, np.nan)
    usable_count = np.zeros(output_shape, dtype=np.int32)
    cut_systematic = np.zeros(output_shape, dtype=bool)
    cut_M = np.full(cut_shape, np.nan)
    cut_M_boot = np.full(cut_boot_shape, np.nan)
    cut_stat = np.full(cut_shape, np.nan)
    cut_valid = np.zeros(cut_shape, dtype=bool)
    cut_count = np.zeros(cut_shape, dtype=np.int32)
    cut_single = np.zeros(cut_shape, dtype=bool)
    weight_central = np.zeros_like(values)
    weight_mean = np.full_like(values, np.nan)
    weight_std = np.full_like(values, np.nan)
    weight_q16 = np.full_like(values, np.nan)
    weight_q50 = np.full_like(values, np.nan)
    weight_q84 = np.full_like(values, np.nan)

    for p in range(npabs):
        for z in range(ndz):
            usable_cut_indices = []
            for ci, cut in enumerate(unique_cuts):
                selected = np.flatnonzero(
                    (cuts == cut)
                    & eligible[:, p, z]
                    & np.isfinite(values[:, p, z])
                    & np.isfinite(scores[:, p, z])
                )
                cut_count[ci, p, z] = len(selected)
                if len(selected) == 0:
                    continue
                central_weights = _normalized_aic_weights(scores[selected, p, z])
                weight_central[selected, p, z] = central_weights
                cut_M[ci, p, z] = np.sum(central_weights * values[selected, p, z])
                cut_single[ci, p, z] = len(selected) == 1

                local_values = boot[selected, :, p, z]
                local_aic = scores_boot[selected, :, p, z]
                local_chi2 = chi2_boot[selected, :, p, z]
                local_dof = dof[selected, p, z][:, None]
                finite = (
                    np.isfinite(local_values)
                    & np.isfinite(local_aic)
                    & np.isfinite(local_chi2)
                )
                positive_dof = local_dof > 0
                q_values = chi2_distribution.sf(
                    local_chi2, np.maximum(local_dof, 1)
                )
                passing = (
                    finite
                    & positive_dof
                    & (q_values >= Q_MIN)
                    & (local_chi2 / np.maximum(local_dof, 1) <= CHI2_DOF_MAX)
                )
                any_passing = np.any(passing, axis=0)
                safe_aic = np.where(passing, local_aic, np.inf)
                minima = np.min(safe_aic, axis=0)
                delta = np.where(passing, local_aic - minima[None, :], np.inf)
                raw = np.where(passing, np.exp(-0.5 * delta), 0.0)
                normalization = np.sum(raw, axis=0)
                weights = np.divide(
                    raw,
                    normalization[None, :],
                    out=np.zeros_like(raw),
                    where=normalization[None, :] > 0.0,
                )
                local_cut_boot = np.sum(
                    weights * np.where(finite, local_values, 0.0), axis=0
                )
                local_cut_boot[~any_passing] = np.nan
                cut_M_boot[ci, :, p, z] = local_cut_boot
                finite_cut = np.isfinite(local_cut_boot)
                if np.count_nonzero(finite_cut) / float(nboot) < BOOTSTRAP_MIN_SUCCESS:
                    continue
                if np.count_nonzero(finite_cut) < 2:
                    continue
                cut_valid[ci, p, z] = True
                cut_stat[ci, p, z] = np.std(local_cut_boot[finite_cut], ddof=1)
                usable_cut_indices.append(ci)
                for local, candidate_index in enumerate(selected):
                    weights_b = weights[local]
                    weight_mean[candidate_index, p, z] = np.mean(weights_b)
                    weight_std[candidate_index, p, z] = np.std(weights_b, ddof=1)
                    (
                        weight_q16[candidate_index, p, z],
                        weight_q50[candidate_index, p, z],
                        weight_q84[candidate_index, p, z],
                    ) = np.quantile(weights_b, [0.16, 0.50, 0.84])

            usable_count[p, z] = len(usable_cut_indices)
            if not usable_cut_indices:
                continue
            local_central = cut_M[usable_cut_indices, p, z]
            final_central = float(np.mean(local_central))
            local_boot = cut_M_boot[usable_cut_indices, :, p, z]
            complete = np.all(np.isfinite(local_boot), axis=0)
            fraction = np.count_nonzero(complete) / float(nboot)
            success[p, z] = fraction
            if fraction < BOOTSTRAP_MIN_SUCCESS or np.count_nonzero(complete) < 2:
                reason[p, z] = "final_bootstrap_success_below_threshold"
                continue
            final_boot = np.mean(local_boot[:, complete], axis=0)
            M[p, z] = final_central
            M_boot[complete, p, z] = final_boot
            stat[p, z] = np.std(final_boot, ddof=1)
            boot_mean[p, z] = np.mean(final_boot)
            boot_bias[p, z] = boot_mean[p, z] - final_central
            cut_systematic[p, z] = len(usable_cut_indices) >= 2
            valid[p, z] = True
            reason[p, z] = (
                "accepted"
                if len(usable_cut_indices) >= 2
                else "accepted_one_usable_cut_cut_systematic_not_estimable"
            )

    M[~valid] = np.nan
    M_boot[:, ~valid] = np.nan
    stat[~valid] = np.nan
    return ReplicaModelAverage(
        M=M,
        M_boot=M_boot,
        statistical_error=stat,
        fit_valid=valid,
        failure_reason=reason,
        bootstrap_success_fraction=success,
        bootstrap_mean=boot_mean,
        bootstrap_bias=boot_bias,
        usable_cut_count=usable_count,
        cut_systematic_estimable=cut_systematic,
        cut_M=cut_M,
        cut_M_boot=cut_M_boot,
        cut_statistical_error=cut_stat,
        cut_valid=cut_valid,
        cut_candidate_count=cut_count,
        cut_single_candidate=cut_single,
        candidate_weight_central=weight_central,
        candidate_weight_mean=weight_mean,
        candidate_weight_std=weight_std,
        candidate_weight_q16=weight_q16,
        candidate_weight_q50=weight_q50,
        candidate_weight_q84=weight_q84,
    )


def _fit_one_tau(inp: dict) -> dict:
    central = inp["central"]
    bootstrap = inp["bootstrap"]
    valid_sum = inp["valid_cut_tsep_mask"]
    nboot = bootstrap.shape[0]
    noperator, ncut, npabs, nz, ntsep = central.shape
    if (noperator, ncut, npabs, nz, ntsep) != (6, 3, 3, 25, 11):
        raise ValueError(f"Unexpected fit input shape: {central.shape}")

    # The difference axis is labeled by the lower endpoint.  The physical
    # threshold starts at T=8, which is index 3 in the registered T=5..15
    # product.  The T=5 cut-3 invalid endpoint is consequently never read.
    difference_original = central[..., 1:] - central[..., :-1]
    difference_boot = bootstrap[..., 1:] - bootstrap[..., :-1]
    pair_valid = valid_sum[:, :-1] & valid_sum[:, 1:]
    tmin_index = int(np.flatnonzero(ALL_TSEPS == TMIN)[0])
    legal_difference_tseps = np.arange(TMIN, REFERENCE_TMAX, dtype=np.int32)
    n_candidate = len(CUTS) * (REFERENCE_TMAX - TMIN)
    arrays = _candidate_arrays(noperator, n_candidate, nboot, npabs, nz)
    candidate_cut = []
    candidate_tmin = []
    candidate_tmax = []
    candidate_n_omitted = []

    candidate_index = 0
    for cut_index, cut in enumerate(CUTS):
        for tmax in range(TMIN + 1, REFERENCE_TMAX + 1):
            ndata = tmax - TMIN
            sl = slice(tmin_index, tmin_index + ndata)
            valid = np.broadcast_to(
                pair_valid[cut_index, sl][None, None, :], (npabs, nz, ndata)
            )
            n_tau = np.asarray(
                [T - 2 * int(cut) + 1 for T in range(TMIN, tmax + 1)],
                dtype=np.int32,
            )
            candidate_cut.append(int(cut))
            candidate_tmin.append(TMIN)
            candidate_tmax.append(tmax)
            candidate_n_omitted.append(REFERENCE_TMAX - tmax)
            for operator in range(noperator):
                difference = ForwardDifference(
                    tseps=legal_difference_tseps[:ndata],
                    n_tau=n_tau,
                    original=difference_original[operator, cut_index, ..., sl],
                    bootstrap=difference_boot[:, operator, cut_index, ..., sl],
                    valid=valid,
                )
                result = fit_plateau_candidate(difference, _candidate(cut, tmax), STATS)
                for name, target_name in (
                    ("M", "M"),
                    ("M_boot", "M_boot"),
                    ("chi2", "chi2"),
                    ("dof", "dof"),
                    ("q_value", "q_value"),
                    ("chi2_dof", "chi2_dof"),
                    ("success_fraction", "bootstrap_success_fraction"),
                    ("covariance_rank", "covariance_rank"),
                    ("covariance_condition", "covariance_condition"),
                    ("covariance_shrinkage", "covariance_shrinkage"),
                    ("range_aic", "range_aic"),
                    ("range_aic_boot", "range_aic_boot"),
                    ("chi2_boot", "chi2_boot"),
                    ("fit_succeeded", "fit_succeeded"),
                    ("eligible", "eligible"),
                    ("failure_reason", "failure_reason"),
                ):
                    arrays[target_name][operator, candidate_index] = getattr(result, name)
            candidate_index += 1

    candidate_cut = np.asarray(candidate_cut, dtype=np.int32)
    candidate_tmin = np.asarray(candidate_tmin, dtype=np.int32)
    candidate_tmax = np.asarray(candidate_tmax, dtype=np.int32)
    candidate_n_omitted = np.asarray(candidate_n_omitted, dtype=np.int32)

    averages = []
    fills = []
    for operator in range(noperator):
        average = replica_model_average_with_replica_gates(
            arrays["M"][operator],
            arrays["M_boot"][operator],
            arrays["range_aic"][operator],
            arrays["range_aic_boot"][operator],
            arrays["chi2_boot"][operator],
            arrays["dof"][operator],
            arrays["eligible"][operator],
            candidate_cut,
            CUTS,
        )
        fill = fill_with_minimum_chi2(
            average,
            arrays["M"][operator],
            arrays["M_boot"][operator],
            arrays["range_aic_boot"][operator],
            arrays["chi2"][operator],
            arrays["chi2_boot"][operator],
            arrays["dof"][operator],
            arrays["fit_succeeded"][operator],
            arrays["eligible"][operator],
            candidate_cut,
            CUTS,
            q_min=Q_MIN,
            chi2_dof_max=CHI2_DOF_MAX,
        )
        averages.append(average)
        fills.append(fill)

    def stack_average(field: str) -> np.ndarray:
        return np.stack([getattr(item, field) for item in averages], axis=0)

    def stack_fill(field: str) -> np.ndarray:
        return np.stack([getattr(item, field) for item in fills], axis=0)

    cut_M = stack_average("cut_M")
    finite_cut_count = np.sum(np.isfinite(cut_M), axis=1)
    cut_mean = np.divide(
        np.nansum(cut_M, axis=1),
        finite_cut_count,
        out=np.zeros_like(cut_M[:, 0]),
        where=finite_cut_count > 0,
    )
    cut_spread = np.where(
        finite_cut_count >= 2,
        np.sqrt(
            np.nansum((cut_M - cut_mean[:, None]) ** 2, axis=1)
            / np.maximum(finite_cut_count - 1, 1)
        ),
        np.nan,
    )

    payload = {
        "schema": np.asarray(SCHEMA),
        "analysis_status": np.asarray(
            "finite_flow_bare_disconnected_matrix_element_sensitivity_not_physical_pdf"
        ),
        "M": stack_average("M"),
        "M_boot": np.stack([item.M_boot for item in averages], axis=1),
        "statistical_error": stack_average("statistical_error"),
        "fit_accepted": stack_average("fit_valid"),
        "fit_valid": stack_average("fit_valid"),
        "M_filled": stack_fill("M_filled"),
        "M_boot_filled": np.stack([item.M_boot_filled for item in fills], axis=1),
        "filled_statistical_error": stack_fill("filled_statistical_error"),
        "estimate_available": stack_fill("estimate_available"),
        "central_fallback_used": stack_fill("central_fallback_used"),
        "central_fallback_candidate_index": stack_fill("central_fallback_candidate_index"),
        "central_fallback_status": stack_fill("central_fallback_status"),
        "replica_fallback_used": np.stack(
            [item.replica_fallback_used for item in fills], axis=1
        ),
        "replica_fallback_fraction": stack_fill("replica_fallback_fraction"),
        "replica_minchi_candidate_index": np.stack(
            [item.replica_minchi_candidate_index for item in fills], axis=1
        ),
        "fit_failure_reason": stack_average("failure_reason"),
        "bootstrap_success_fraction": stack_average("bootstrap_success_fraction"),
        "bootstrap_mean_diagnostic": stack_average("bootstrap_mean"),
        "bootstrap_bias": stack_average("bootstrap_bias"),
        "usable_cut_count": stack_average("usable_cut_count"),
        "cut_systematic_estimable": stack_average("cut_systematic_estimable"),
        "cut_spread_diagnostic": cut_spread,
        "cut_M": cut_M,
        "cut_M_boot": np.transpose(
            np.stack([item.cut_M_boot for item in averages]), (2, 0, 1, 3, 4)
        ),
        "cut_statistical_error": stack_average("cut_statistical_error"),
        "cut_valid": stack_average("cut_valid"),
        "cut_candidate_count": stack_average("cut_candidate_count"),
        "cut_single_candidate": stack_average("cut_single_candidate"),
        "candidate_M": arrays["M"],
        "candidate_M_boot": arrays["M_boot"],
        "candidate_chi2": arrays["chi2"],
        "candidate_dof": arrays["dof"],
        "candidate_q_value": arrays["q_value"],
        "candidate_chi2_dof": arrays["chi2_dof"],
        "candidate_bootstrap_success_fraction": arrays["bootstrap_success_fraction"],
        "candidate_covariance_rank": arrays["covariance_rank"],
        "candidate_covariance_condition": arrays["covariance_condition"],
        "candidate_covariance_shrinkage": arrays["covariance_shrinkage"],
        "candidate_range_aic": arrays["range_aic"],
        "candidate_range_aic_boot": arrays["range_aic_boot"],
        "candidate_chi2_boot": arrays["chi2_boot"],
        "candidate_fit_succeeded": arrays["fit_succeeded"],
        "candidate_eligible": arrays["eligible"],
        "candidate_failure_reason": arrays["failure_reason"],
        "candidate_weight_central": stack_average("candidate_weight_central"),
        "candidate_weight_boot_mean": stack_average("candidate_weight_mean"),
        "candidate_weight_boot_std": stack_average("candidate_weight_std"),
        "candidate_weight_boot_q16": stack_average("candidate_weight_q16"),
        "candidate_weight_boot_q50": stack_average("candidate_weight_q50"),
        "candidate_weight_boot_q84": stack_average("candidate_weight_q84"),
        "ratio_sum_original": np.transpose(central, (0, 1, 3, 4, 2)),
        "difference_original": np.transpose(
            difference_original, (0, 1, 3, 4, 2)
        ),
        "valid_cut_tsep_mask": valid_sum,
        "valid_difference_mask": pair_valid,
        "operator_labels": np.asarray(OPERATOR_LABELS),
        "operator_definitions": np.asarray(OPERATOR_DEFINITIONS),
        "operator_c2_channel": np.asarray(OPERATOR_CHANNEL),
        "operator_projection": np.asarray(OPERATOR_PROJECTION),
        "candidate_cut": candidate_cut,
        "candidate_tmin": candidate_tmin,
        "candidate_tmax_summed": candidate_tmax,
        "candidate_n_omitted_difference_points": candidate_n_omitted,
        "registered_cuts": CUTS,
        "fit_tsep_values": ALL_TSEPS,
        "difference_tsep_values": legal_difference_tseps,
        "pabs_values": PABS,
        "z_values": Z,
        "flow_tau_t_over_a2": np.asarray(inp["tau"]),
        "axes_M": np.asarray(("operator", "momentum_abs", "z")),
        "axes_M_boot": np.asarray(("bootstrap", "operator", "momentum_abs", "z")),
        "axes_cut_M": np.asarray(("operator", "cut", "momentum_abs", "z")),
        "axes_cut_M_boot": np.asarray(
            ("bootstrap", "operator", "cut", "momentum_abs", "z")
        ),
        "axes_candidate_M": np.asarray(
            ("operator", "candidate", "momentum_abs", "z")
        ),
        "axes_candidate_M_boot": np.asarray(
            ("operator", "candidate", "bootstrap", "momentum_abs", "z")
        ),
        "confs": inp["confs"],
        "bootstrap_indices": inp["bootstrap_indices"],
        "nconf": np.asarray(406, dtype=np.int32),
        "nboot": np.asarray(nboot, dtype=np.int32),
        "seed": np.asarray(1115, dtype=np.int64),
        "bootstrap_method": np.asarray(
            "multinomial configuration bootstrap; shared frozen manifest"
        ),
        "a_fm": np.asarray(A_FM),
        "minimum_tsep_fm_strict": np.asarray(MINIMUM_TSEP_FM),
        "fixed_tmin": np.asarray(TMIN, dtype=np.int32),
        "fixed_tmin_fm": np.asarray(TMIN * A_FM),
        "fit_method": np.asarray("forward_sumratio_difference_correlated_plateau"),
        "fit_model": np.asarray("D_c(T)=S_c(T+1)-S_c(T)=M"),
        "within_cut_score": np.asarray("AIC=chi2+2*k+2*N_omitted;k=1"),
        "between_cut_policy": np.asarray("equal_weight_over_usable_cuts"),
        "error_definition": np.asarray(
            f"std_ddof1_of_{nboot}_replica_dependent_AIC_and_equal_cut_mixture;no_sqrt_Nboot"
        ),
        "fallback_policy": np.asarray(
            "minimum_raw_chi2_positive_dof_then_single_difference_only_if_no_positive_dof"
        ),
        "covariance_method": np.asarray(
            "schaefer_strimmer_correlation_to_identity"
        ),
        "svdcut": np.asarray(SVDCUT),
        "q_min": np.asarray(Q_MIN),
        "correlated_chi2_dof_max": np.asarray(CHI2_DOF_MAX),
        "bootstrap_min_success": np.asarray(BOOTSTRAP_MIN_SUCCESS),
        "manifest_sha256": np.asarray(inp["manifest_sha256"]),
        "input_product_sha256": np.asarray(inp["input_product_sha256"]),
        "input_contract_json": np.asarray(inp["contract_json"]),
    }
    return payload


def _candidate(cut: int, tmax: int):
    """Small adapter matching gluonfit.config.Candidate's public fields."""

    class CandidateAdapter:
        tmin = TMIN
        reference_tmax = REFERENCE_TMAX

        def __init__(self, cut_value: int, tmax_value: int):
            self.cut = int(cut_value)
            self.tmax = int(tmax_value)

        @property
        def n_omitted(self) -> int:
            return REFERENCE_TMAX - self.tmax

        @property
        def label(self) -> str:
            return (
                f"D{self.tmin}-{self.tmax - 1}_S{self.tmin}-{self.tmax}"
                f"_L{self.cut}_R{self.cut}"
            )

    return CandidateAdapter(cut, tmax)


def fit_tau(input_path: Path, output_path: Path, *, replace: bool = False) -> dict:
    start = time.time()
    inp = load_input(input_path)
    payload = _fit_one_tau(inp)
    contract = {
        "schema": SCHEMA,
        "input_schema": INPUT_SCHEMA,
        "flow_tau_t_over_a2": float(inp["tau"]),
        "nconf": 406,
        "nboot": 1000,
        "seed": 1115,
        "bootstrap_method": "multinomial configuration bootstrap",
        "fixed_tmin": TMIN,
        "fixed_tmin_fm": TMIN * A_FM,
        "strict_threshold": "Tmin*a > 0.6 fm; Tmin=floor(0.6/a)+1",
        "cuts": CUTS.tolist(),
        "candidate_tmax_summed": sorted(set(payload["candidate_tmax_summed"].tolist())),
        "operators": list(OPERATOR_LABELS),
        "operator_definitions": list(OPERATOR_DEFINITIONS),
        "aic": "AIC=chi2+2*k+2*N_omitted, k=1; weights recomputed per replica",
        "between_cut": "equal weight over usable cuts; cuts are not one likelihood",
        "quality_gates": {
            "Q_min": Q_MIN,
            "chi2_dof_max": CHI2_DOF_MAX,
            "bootstrap_fraction_min": BOOTSTRAP_MIN_SUCCESS,
            "positive_dof": True,
        },
        "covariance": {
            "method": "Schaefer-Strimmer correlation-to-identity",
            "svdcut": SVDCUT,
            "preserve_unshrunk_sample_rank": True,
        },
        "fallback": (
            "AIC-average eligible candidates; if no passing replica candidate, "
            "minimum raw chi2 positive-dof; only then single-difference diagnostic"
        ),
        "input_product": str(input_path.resolve()),
        "input_product_sha256": inp["input_product_sha256"],
        "manifest_sha256": inp["manifest_sha256"],
        "status_note": "finite-flow bare matrix elements; no renormalization, matching, continuum, or zero-flow conversion",
    }
    payload["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not replace:
        raise FileExistsError(f"Refusing existing fit without --replace: {output_path}")
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output_path)
    output_sha = sha256_file(output_path)
    done = {
        **contract,
        "status": "complete",
        "output": str(output_path.resolve()),
        "output_sha256": output_sha,
        "elapsed_seconds": time.time() - start,
        "fit_accepted_count": int(np.count_nonzero(payload["fit_accepted"])),
        "fit_total_count": int(payload["fit_accepted"].size),
        "filled_count": int(np.count_nonzero(payload["estimate_available"])),
        "central_fallback_count": int(np.count_nonzero(payload["central_fallback_used"])),
        "replica_fallback_fraction_mean": float(
            np.nanmean(payload["replica_fallback_fraction"])
        ),
    }
    receipt_path = Path(str(output_path) + ".done.json")
    temporary_json = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}")
    temporary_json.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, receipt_path)
    print(json.dumps(done, indent=2, sort_keys=True), flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    taus = TAUS if args.tau is None else (args.tau,)
    for tau in taus:
        input_path = args.input_dir / product_name(tau)
        output_path = args.output_dir / output_name(tau)
        fit_tau(input_path, output_path, replace=args.replace)


if __name__ == "__main__":
    main()
