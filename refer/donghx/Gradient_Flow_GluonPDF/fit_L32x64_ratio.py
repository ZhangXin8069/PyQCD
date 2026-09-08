#!/usr/bin/env python3
"""Fit the L32x64 HYP-gluon C3/C2 products with a direct Ratio plateau.

The Summer-school C3/C2 products are not the six-direction ``ratio_v4``
products consumed by the general quark-PDF fit package.  This adapter keeps
their native axes and applies the registered direct-Ratio model from the
``fit_bare`` skill independently for every operator formula, momentum
direction, and z separation:

    R(T, tau) = M.

For L32x64 the preregistered windows are T=6--8, 6--9, and 6--10 with
symmetric endpoint cuts c=1,2; the primary window is T=6--9, c=2.  A
Schaefer--Strimmer correlation covariance and SVD inverse are used for every
candidate.  The central value is accepted only when the primary window and at
least one alternate window pass the quality gates.  Bootstrap replicas are
fit with the same central GLS projector, preserving the 500 shared indices
from the C3/C2 product.

This produces finite-flow HYP bare matrix-element diagnostics.  It does not
perform renormalization, mixing, matching, or a physical-PDF conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the registered covariance implementation from fit_bare.  Its package
# is read-only in the shared analysis tree, but importing it does not modify
# any products and keeps the statistical convention identical to the skill.
BARE_FIT_ROOT = Path(
    "/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Bare_Matrix_Fit"
)
sys.path.insert(0, str(BARE_FIT_ROOT))
from barefit.config import StatisticsConfig  # noqa: E402
from barefit.covariance import correlated_constant_fit  # noqa: E402


FORMULAS = ("titi", "titi_minus_ijij", "titi_plus_ijij")
DIRECTIONS = ("x", "y", "z")
ORIENTATIONS = ("plus_z", "minus_z", "even_sum", "odd_difference")
PRIMARY_LABEL = "T6-9_L2_R2"
NCONF_EXPECTED = 200
NBOOT_EXPECTED = 500
SEED_EXPECTED = 1115

# These are the L32x64 direct-Ratio windows registered in
# Fit_SKILL/Bare_Matrix_Fit/configs/fit_profiles.json.


@dataclass(frozen=True)
class Candidate:
    label: str
    fit_start: int
    fit_end: int
    cut_left: int
    cut_right: int

    @property
    def tseps(self) -> tuple[int, ...]:
        return tuple(range(self.fit_start, self.fit_end + 1))

    @property
    def coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (tsep, tau)
            for tsep in self.tseps
            for tau in range(self.cut_left, tsep - self.cut_right + 1)
        )


CANDIDATES = tuple(
    Candidate(
        label=f"T6-{tmax}_L{cut}_R{cut}",
        fit_start=6,
        fit_end=tmax,
        cut_left=cut,
        cut_right=cut,
    )
    for cut in (1, 2)
    for tmax in (8, 9, 10)
)

STATISTICS = StatisticsConfig(
    covariance_shrinkage="schaefer_strimmer_correlation_to_identity",
    svdcut=1e-12,
    covariance_relative_jitter=1e-12,
    covariance_absolute_jitter=1e-18,
    normal_rcond=1e-12,
    q_min=0.05,
    correlated_chi2_dof_max=3.0,
    bootstrap_min_success=0.90,
    minimum_accepted_windows=2,
    window_systematic="equal_weight_rms_about_preregistered_primary",
    fit_window_status="default_preregistration_requires_physics_review_before_final_publication",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: np.lib.npyio.NpzFile, key: str):
    return np.asarray(data[key]).item()


def _check_metadata(data: np.lib.npyio.NpzFile, path: Path, channel: str) -> None:
    required = {
        "ratio_central",
        "ratio_bootstrap",
        "bootstrap_indices",
        "confs",
        "channel",
        "formula_labels",
        "direction_labels",
        "orientation_labels",
        "physical_orientation",
        "pabs",
        "z",
        "tseps",
        "tau",
        "valid_tau_mask",
        "axes_central",
        "axes_bootstrap",
        "nconf",
        "nboot",
        "bootstrap_seed",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Incomplete C3/C2 product {path}; missing {missing}")
    if str(scalar(data, "channel")) != channel:
        raise ValueError(f"Channel metadata mismatch in {path}")
    if [str(x) for x in np.asarray(data["formula_labels"]).tolist()] != list(FORMULAS):
        raise ValueError(f"Unexpected formula order in {path}")
    if [str(x) for x in np.asarray(data["direction_labels"]).tolist()] != list(DIRECTIONS):
        raise ValueError(f"Unexpected direction order in {path}")
    if [str(x) for x in np.asarray(data["orientation_labels"]).tolist()] != list(ORIENTATIONS):
        raise ValueError(f"Unexpected orientation order in {path}")
    expected_central_axes = [
        "formula",
        "direction",
        "orientation",
        "pabs",
        "z",
        "tsep",
        "tau",
    ]
    expected_boot_axes = [
        "boot",
        "formula",
        "direction",
        "pabs",
        "z",
        "tsep",
        "tau",
    ]
    if [str(x) for x in np.asarray(data["axes_central"]).tolist()] != expected_central_axes:
        raise ValueError(f"Unexpected central axes in {path}")
    if [str(x) for x in np.asarray(data["axes_bootstrap"]).tolist()] != expected_boot_axes:
        raise ValueError(f"Unexpected bootstrap axes in {path}")
    if int(scalar(data, "nconf")) != NCONF_EXPECTED:
        raise ValueError(f"Expected Nconf={NCONF_EXPECTED} in {path}")
    if int(scalar(data, "nboot")) != NBOOT_EXPECTED:
        raise ValueError(f"Expected Nboot={NBOOT_EXPECTED} in {path}")
    if int(scalar(data, "bootstrap_seed")) != SEED_EXPECTED:
        raise ValueError(f"Expected bootstrap seed {SEED_EXPECTED} in {path}")
    if str(scalar(data, "physical_orientation")) != (
        "plus_z" if channel == "unpolarized" else "odd_difference"
    ):
        raise ValueError(f"Unexpected physical orientation in {path}")


def load_product(path: Path, channel: str, pabs: int):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        _check_metadata(data, path, channel)
        central_full = np.asarray(data["ratio_central"], dtype=np.complex128)
        bootstrap_full = np.asarray(data["ratio_bootstrap"], dtype=np.complex128)
        indices = np.asarray(data["bootstrap_indices"], dtype=np.int32)
        confs = np.asarray(data["confs"]).astype(str)
        pabs_values = np.asarray(data["pabs"], dtype=np.int32)
        z_values = np.asarray(data["z"], dtype=np.int32)
        tseps = np.asarray(data["tseps"], dtype=np.int32)
        tau = np.asarray(data["tau"], dtype=np.int32)
        valid_tau_mask = np.asarray(data["valid_tau_mask"], dtype=bool)
        orientation = 0 if channel == "unpolarized" else 3

        if central_full.shape != (3, 3, 4, len(pabs_values), len(z_values), len(tseps), len(tau)):
            raise ValueError(f"Unexpected ratio_central shape {central_full.shape} in {path}")
        if bootstrap_full.shape != (
            NBOOT_EXPECTED,
            3,
            3,
            len(pabs_values),
            len(z_values),
            len(tseps),
            len(tau),
        ):
            raise ValueError(f"Unexpected ratio_bootstrap shape {bootstrap_full.shape} in {path}")
        if indices.shape != (NBOOT_EXPECTED, NCONF_EXPECTED):
            raise ValueError(f"Unexpected bootstrap-index shape {indices.shape} in {path}")
        if len(confs) != NCONF_EXPECTED or len(set(confs.tolist())) != NCONF_EXPECTED:
            raise ValueError(f"Configuration metadata is not 200 unique entries in {path}")
        if valid_tau_mask.shape != (len(tseps), len(tau)):
            raise ValueError(f"Unexpected valid_tau_mask shape in {path}")
        if pabs not in pabs_values:
            raise ValueError(f"P={pabs} is absent from {path}; available={pabs_values.tolist()}")
        pindex = int(np.flatnonzero(pabs_values == pabs)[0])

        # Convert to real/imaginary physical observable with axes
        # formula,direction,z,tsep,tau and boot,formula,direction,z,tsep,tau.
        central = central_full[:, :, orientation, pindex]
        bootstrap = bootstrap_full[:, :, :, pindex]
        projection = np.imag if channel == "helicity" else np.real
        central = projection(central).astype(float, copy=False)
        bootstrap = projection(bootstrap).astype(float, copy=False)
        if not np.all(np.isfinite(indices)):
            raise ValueError(f"Non-finite bootstrap indices in {path}")
        return {
            "path": path,
            "sha256": sha256_file(path),
            "central": central,
            "bootstrap": bootstrap,
            "indices": indices,
            "confs": confs,
            "pabs_values": pabs_values,
            "z_values": z_values,
            "tseps": tseps,
            "tau": tau,
            "valid_tau_mask": valid_tau_mask,
            "physical_orientation": ORIENTATIONS[orientation],
            "nconf": int(scalar(data, "nconf")),
            "nboot": int(scalar(data, "nboot")),
            "seed": int(scalar(data, "bootstrap_seed")),
            "operator_definition": str(scalar(data, "operator_definition")),
            "input_scheme": str(scalar(data, "input_scheme")),
        }


def candidate_points(product, formula: int, direction: int, z: int, candidate: Candidate):
    central = product["central"][formula, direction, z]
    bootstrap = product["bootstrap"][:, formula, direction, z]
    values: list[np.ndarray] = []
    replicas: list[np.ndarray] = []
    coordinates = candidate.coordinates
    for tsep in candidate.tseps:
        locations = np.flatnonzero(product["tseps"] == int(tsep))
        if len(locations) != 1:
            raise ValueError(f"Candidate {candidate.label} needs exactly one T={tsep}")
        itsep = int(locations[0])
        first = candidate.cut_left
        last = int(tsep) - candidate.cut_right
        if last < first or last >= central.shape[-1]:
            return None
        interval_valid = product["valid_tau_mask"][itsep, first : last + 1]
        y = central[itsep, first : last + 1]
        samples = bootstrap[:, itsep, first : last + 1]
        if not np.all(interval_valid) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(samples)):
            return None
        values.append(y)
        replicas.append(samples)
    return np.concatenate(values), np.concatenate(replicas, axis=1), np.asarray(coordinates, dtype=np.int32)


def fit_product(product, channel: str, pabs: int):
    nformula, ndirection = len(FORMULAS), len(DIRECTIONS)
    ncandidate = len(CANDIDATES)
    nboot = product["nboot"]
    npabs = 1
    nz = len(product["z_values"])
    max_points = max(len(c.coordinates) for c in CANDIDATES)
    shape = (nformula, ndirection, ncandidate, npabs, nz)

    candidate_M = np.full(shape, np.nan)
    candidate_stat = np.full(shape, np.nan)
    candidate_chi2 = np.full(shape, np.nan)
    candidate_dof = np.full(shape, -1, dtype=np.int32)
    candidate_q = np.full(shape, np.nan)
    candidate_chi2_dof = np.full(shape, np.nan)
    candidate_success = np.zeros(shape)
    candidate_rank = np.zeros(shape, dtype=np.int32)
    candidate_condition = np.full(shape, np.nan)
    candidate_shrinkage = np.full(shape, np.nan)
    candidate_finite_replicas = np.zeros(shape, dtype=np.int32)
    candidate_accepted = np.zeros(shape, dtype=bool)
    candidate_reason = np.full(shape, "not_evaluated", dtype="U256")
    candidate_boot = np.full((nformula, ndirection, ncandidate, nboot, npabs, nz), np.nan)
    candidate_covariance = np.full(shape + (max_points, max_points), np.nan)
    candidate_eigenvalues = np.full(shape + (max_points,), np.nan)
    candidate_retained = np.zeros(shape + (max_points,), dtype=bool)

    coords_store = np.full((ncandidate, max_points, 2), -1, dtype=np.int32)
    point_mask = np.zeros((ncandidate, max_points), dtype=bool)
    point_counts = np.zeros(ncandidate, dtype=np.int32)
    for ic, cand in enumerate(CANDIDATES):
        coords = np.asarray(cand.coordinates, dtype=np.int32)
        coords_store[ic, : len(coords)] = coords
        point_mask[ic, : len(coords)] = True
        point_counts[ic] = len(coords)

    for formula in range(nformula):
        for direction in range(ndirection):
            for icandidate, candidate in enumerate(CANDIDATES):
                for iz in range(nz):
                    index = (formula, direction, icandidate, 0, iz)
                    points = candidate_points(product, formula, direction, iz, candidate)
                    if points is None:
                        candidate_reason[index] = "ratio_input_nonfinite_or_mask_invalid"
                        continue
                    y, samples, coords = points
                    if not np.array_equal(coords, coords_store[icandidate, : len(coords)]):
                        raise RuntimeError("Candidate-coordinate mismatch")
                    try:
                        result = correlated_constant_fit(y, samples, STATISTICS)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                        candidate_reason[index] = f"fit_input_invalid:{type(exc).__name__}"
                        continue
                    boot_values = np.asarray(result.parameter_boot[:, 0], dtype=float)
                    finite = np.isfinite(boot_values)
                    candidate_M[index] = float(result.parameters[0])
                    if np.count_nonzero(finite) >= 2:
                        candidate_stat[index] = float(np.std(boot_values[finite], ddof=1))
                    candidate_boot[formula, direction, icandidate, :, 0, iz] = boot_values
                    candidate_chi2[index] = result.chi2
                    candidate_dof[index] = result.dof
                    candidate_q[index] = result.q_value
                    candidate_chi2_dof[index] = (
                        result.chi2 / result.dof if result.dof > 0 else np.nan
                    )
                    candidate_success[index] = result.success_fraction
                    candidate_rank[index] = result.covariance.rank
                    candidate_condition[index] = result.covariance.condition
                    candidate_shrinkage[index] = result.covariance.shrinkage_lambda
                    candidate_finite_replicas[index] = result.covariance.finite_replicas
                    npoint = len(y)
                    candidate_covariance[index + (slice(0, npoint), slice(0, npoint))] = (
                        result.covariance.covariance
                    )
                    candidate_eigenvalues[index + (slice(0, npoint),)] = result.covariance.eigenvalues
                    candidate_retained[index + (slice(0, npoint),)] = result.covariance.retained
                    accepted = bool(
                        np.isfinite(candidate_M[index])
                        and np.isfinite(candidate_stat[index])
                        and result.dof > 0
                        and np.isfinite(result.q_value)
                        and result.q_value >= STATISTICS.q_min
                        and np.isfinite(candidate_chi2_dof[index])
                        and candidate_chi2_dof[index] <= STATISTICS.correlated_chi2_dof_max
                        and result.success_fraction >= STATISTICS.bootstrap_min_success
                    )
                    candidate_accepted[index] = accepted
                    candidate_reason[index] = "accepted" if accepted else "fit_quality_gate_failed"

    primary = next(i for i, c in enumerate(CANDIDATES) if c.label == PRIMARY_LABEL)
    primary_accepted = candidate_accepted[:, :, primary]
    accepted_count = np.sum(candidate_accepted, axis=2)
    fit_valid = primary_accepted & (accepted_count >= STATISTICS.minimum_accepted_windows)
    primary_M = candidate_M[:, :, primary]
    primary_stat = candidate_stat[:, :, primary]
    window_systematic = np.full((nformula, ndirection, npabs, nz), np.nan)
    for formula in range(nformula):
        for direction in range(ndirection):
            for iz in range(nz):
                if not primary_accepted[formula, direction, 0, iz]:
                    continue
                alternates = np.flatnonzero(candidate_accepted[formula, direction, :, 0, iz])
                alternates = alternates[alternates != primary]
                if len(alternates) >= STATISTICS.minimum_accepted_windows - 1:
                    differences = candidate_M[formula, direction, alternates, 0, iz] - primary_M[
                        formula, direction, 0, iz
                    ]
                    window_systematic[formula, direction, 0, iz] = float(
                        np.sqrt(np.mean(differences * differences))
                    )
    total_error = np.sqrt(primary_stat**2 + window_systematic**2)
    M = np.where(fit_valid, primary_M, np.nan)
    stat = np.where(fit_valid, primary_stat, np.nan)
    syst = np.where(fit_valid, window_systematic, np.nan)
    total = np.where(fit_valid, total_error, np.nan)
    primary_boot_raw = candidate_boot[:, :, primary]
    primary_boot = np.where(fit_valid[:, :, None, ...], primary_boot_raw, np.nan)

    # The quality-controlled arrays above intentionally keep failed points as
    # NaN.  The user-facing diagnostic request is to retain at least one
    # estimate at every z.  Fill those points with an explicitly labelled
    # fallback, while leaving ``M``/``fit_valid`` untouched:
    #
    #   1. choose the finite positive-dof candidate with minimum raw chi2;
    #   2. if every correlated candidate is unavailable (typically a
    #      zero-variance helicity point at z=0), take an arithmetic constant
    #      mean over the primary candidate's finite Ratio points.
    #
    # This is diagnostic coverage only; no fallback point is promoted to an
    # accepted fit.
    filled = np.full((nformula, ndirection, npabs, nz), np.nan)
    filled_boot = np.full((nformula, ndirection, nboot, npabs, nz), np.nan)
    filled_stat = np.full((nformula, ndirection, npabs, nz), np.nan)
    filled_window = np.full((nformula, ndirection, npabs, nz), np.nan)
    filled_total = np.full((nformula, ndirection, npabs, nz), np.nan)
    estimate_available = np.zeros((nformula, ndirection, npabs, nz), dtype=bool)
    # This flag is reserved for estimates that are filled after the formal
    # quality gates.  Keep accepted primary fits distinguishable from those
    # diagnostic fallbacks, and leave an impossible no-estimate point false so
    # the flag cannot suggest that a value was produced when it was not.
    central_fallback_used = np.zeros(
        (nformula, ndirection, npabs, nz), dtype=bool
    )
    central_fallback_candidate = np.full(
        (nformula, ndirection, npabs, nz), -1, dtype=np.int16
    )
    central_fallback_status = np.full(
        (nformula, ndirection, npabs, nz), "no_estimate", dtype="U96"
    )
    replica_fallback_used = np.zeros(
        (nboot, nformula, ndirection, npabs, nz), dtype=bool
    )
    replica_fallback_candidate = np.full(
        (nboot, nformula, ndirection, npabs, nz), -1, dtype=np.int16
    )

    for formula in range(nformula):
        for direction in range(ndirection):
            for iz in range(nz):
                if fit_valid[formula, direction, 0, iz]:
                    filled[formula, direction, 0, iz] = primary_M[formula, direction, 0, iz]
                    filled_boot[formula, direction, :, 0, iz] = primary_boot_raw[
                        formula, direction, :, 0, iz
                    ]
                    filled_stat[formula, direction, 0, iz] = primary_stat[
                        formula, direction, 0, iz
                    ]
                    filled_window[formula, direction, 0, iz] = window_systematic[
                        formula, direction, 0, iz
                    ]
                    filled_total[formula, direction, 0, iz] = total_error[
                        formula, direction, 0, iz
                    ]
                    estimate_available[formula, direction, 0, iz] = True
                    central_fallback_candidate[formula, direction, 0, iz] = primary
                    central_fallback_status[formula, direction, 0, iz] = "accepted_primary"
                    continue

                # Candidate-wise GLS results are retained even when their
                # quality gate failed.  Select by raw chi2 only after requiring
                # a finite positive-dof fit and complete bootstrap parameters.
                finite_gls = (
                    np.isfinite(candidate_M[formula, direction, :, 0, iz])
                    & np.isfinite(candidate_stat[formula, direction, :, 0, iz])
                    & (candidate_dof[formula, direction, :, 0, iz] > 0)
                    & np.isfinite(candidate_chi2[formula, direction, :, 0, iz])
                    & np.all(
                        np.isfinite(candidate_boot[formula, direction, :, :, 0, iz]),
                        axis=1,
                    )
                )
                if np.any(finite_gls):
                    choices = np.flatnonzero(finite_gls)
                    selected = int(choices[np.argmin(candidate_chi2[formula, direction, choices, 0, iz])])
                    values = candidate_boot[formula, direction, selected, :, 0, iz]
                    filled[formula, direction, 0, iz] = candidate_M[
                        formula, direction, selected, 0, iz
                    ]
                    filled_boot[formula, direction, :, 0, iz] = values
                    filled_stat[formula, direction, 0, iz] = np.std(values, ddof=1)
                    filled_total[formula, direction, 0, iz] = filled_stat[
                        formula, direction, 0, iz
                    ]
                    estimate_available[formula, direction, 0, iz] = True
                    central_fallback_used[formula, direction, 0, iz] = True
                    central_fallback_candidate[formula, direction, 0, iz] = selected
                    central_fallback_status[formula, direction, 0, iz] = (
                        "fallback_minchi2_positive_dof"
                    )
                    replica_fallback_used[:, formula, direction, 0, iz] = True
                    replica_fallback_candidate[:, formula, direction, 0, iz] = selected
                    continue

                # No covariance fit exists (e.g. every replica has zero
                # variance).  Use a transparent unweighted constant mean over
                # the primary window's finite points, then propagate the same
                # operation to each bootstrap replica.
                selected = None
                selected_points = None
                for candidate_index in (primary,) + tuple(
                    i for i in range(ncandidate) if i != primary
                ):
                    points = candidate_points(
                        product, formula, direction, iz, CANDIDATES[candidate_index]
                    )
                    if points is None:
                        continue
                    y, samples, _ = points
                    if np.all(np.isfinite(y)) and np.all(np.isfinite(samples)):
                        selected = candidate_index
                        selected_points = (y, samples)
                        break
                if selected_points is not None:
                    y, samples = selected_points
                    values = np.mean(samples, axis=1)
                    filled[formula, direction, 0, iz] = float(np.mean(y))
                    filled_boot[formula, direction, :, 0, iz] = values
                    filled_stat[formula, direction, 0, iz] = np.std(values, ddof=1)
                    filled_total[formula, direction, 0, iz] = filled_stat[
                        formula, direction, 0, iz
                    ]
                    estimate_available[formula, direction, 0, iz] = True
                    central_fallback_used[formula, direction, 0, iz] = True
                    central_fallback_candidate[formula, direction, 0, iz] = int(selected)
                    central_fallback_status[formula, direction, 0, iz] = (
                        "fallback_constant_mean_no_covariance"
                    )
                    replica_fallback_used[:, formula, direction, 0, iz] = True
                    replica_fallback_candidate[:, formula, direction, 0, iz] = int(selected)

    replica_fallback_fraction = np.mean(replica_fallback_used, axis=0)

    return {
        "M": M,
        "bare_matrix_element": M,
        "M_raw_diagnostic": primary_M,
        "statistical_error": stat,
        "statistical_error_raw_diagnostic": primary_stat,
        "window_systematic": syst,
        "window_systematic_raw_diagnostic": window_systematic,
        "total_error": total,
        "total_error_raw_diagnostic": total_error,
        "fit_valid": fit_valid,
        "primary_fit_accepted": primary_accepted,
        "accepted_window_count": accepted_count,
        "candidate_M_raw": candidate_M,
        "candidate_statistical_error_raw": candidate_stat,
        "candidate_fit_accepted": candidate_accepted,
        "candidate_failure_reason": candidate_reason,
        "candidate_chi2": candidate_chi2,
        "candidate_dof": candidate_dof,
        "candidate_Q": candidate_q,
        "candidate_correlated_chi2_dof": candidate_chi2_dof,
        "candidate_bootstrap_success_fraction": candidate_success,
        "candidate_covariance_rank": candidate_rank,
        "candidate_covariance_condition": candidate_condition,
        "candidate_covariance_shrinkage_lambda": candidate_shrinkage,
        "candidate_finite_covariance_replicas": candidate_finite_replicas,
        "candidate_M_boot_raw": candidate_boot,
        "primary_M_boot": primary_boot,
        "primary_M_boot_raw_diagnostic": primary_boot_raw,
        "M_filled": filled,
        "M_boot_filled": filled_boot,
        "filled_statistical_error": filled_stat,
        "filled_window_systematic": filled_window,
        "filled_total_error": filled_total,
        "estimate_available": estimate_available,
        "central_fallback_used": central_fallback_used,
        "central_fallback_candidate_index": central_fallback_candidate,
        "central_fallback_status": central_fallback_status,
        "replica_fallback_used": replica_fallback_used,
        "replica_fallback_fraction": replica_fallback_fraction,
        "replica_fallback_candidate_index": replica_fallback_candidate,
        "candidate_fit_covariance": candidate_covariance,
        "candidate_covariance_eigenvalues": candidate_eigenvalues,
        "candidate_retained_eigenmodes": candidate_retained,
        "candidate_coordinates_T_tau": coords_store,
        "candidate_point_mask": point_mask,
        "candidate_npoints": point_counts,
        "primary_candidate_index": np.asarray(primary, dtype=np.int32),
        "primary_candidate": np.asarray(PRIMARY_LABEL),
    }


def write_product(product, fit, channel: str, pabs: int, output: Path, command: str):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fit)
    payload.update(
        {
            "component_labels": np.asarray(FORMULAS),
            "formula_labels": np.asarray(FORMULAS),
            "direction_labels": np.asarray(DIRECTIONS),
            "pabs_values": np.asarray([pabs], dtype=np.int32),
            "delta_z_values": product["z_values"],
            "axes_M": np.asarray(["formula", "direction", "pabs", "z"]),
            "axes_M_filled": np.asarray(["formula", "direction", "pabs", "z"]),
            "axes_primary_M_boot": np.asarray(
                ["formula", "direction", "boot", "pabs", "z"]
            ),
            "axes_M_boot_filled": np.asarray(
                ["formula", "direction", "boot", "pabs", "z"]
            ),
            "axes_replica_fallback": np.asarray(
                ["boot", "formula", "direction", "pabs", "z"]
            ),
            "axes_candidate_fit": np.asarray(
                ["formula", "direction", "candidate", "pabs", "z"]
            ),
            "axes_candidate_M_boot_raw": np.asarray(
                ["formula", "direction", "candidate", "boot", "pabs", "z"]
            ),
            "fit_model": np.asarray("R(T,tau)=M"),
            "fit_method": np.asarray("direct_correlated_ratio_plateau"),
            "central_estimator": np.asarray("GLS_fit_to_ratio_central"),
            "bootstrap_role": np.asarray("shared C3/C2 bootstrap replicas for covariance and statistical error"),
            "covariance_method": np.asarray(STATISTICS.covariance_shrinkage),
            "svdcut": np.asarray(STATISTICS.svdcut),
            "q_min": np.asarray(STATISTICS.q_min),
            "correlated_chi2_dof_max": np.asarray(STATISTICS.correlated_chi2_dof_max),
            "bootstrap_min_success": np.asarray(STATISTICS.bootstrap_min_success),
            "minimum_accepted_windows": np.asarray(STATISTICS.minimum_accepted_windows, dtype=np.int32),
            "window_systematic_method": np.asarray(STATISTICS.window_systematic),
            "fit_window_status": np.asarray(STATISTICS.fit_window_status),
            "candidate_labels": np.asarray([c.label for c in CANDIDATES]),
            "candidate_fit_start": np.asarray([c.fit_start for c in CANDIDATES], dtype=np.int32),
            "candidate_fit_end": np.asarray([c.fit_end for c in CANDIDATES], dtype=np.int32),
            "candidate_cut_left": np.asarray([c.cut_left for c in CANDIDATES], dtype=np.int32),
            "candidate_cut_right": np.asarray([c.cut_right for c in CANDIDATES], dtype=np.int32),
            "ensemble": np.asarray("L32x64"),
            "nx": np.asarray(32, dtype=np.int32),
            "nt": np.asarray(64, dtype=np.int32),
            "a_fm": np.asarray(0.0897),
            "channel": np.asarray(channel),
            "physical_orientation": np.asarray(product["physical_orientation"]),
            "z_values": product["z_values"],
            "tsep_values": product["tseps"],
            "tau_values": product["tau"],
            "valid_tau_mask": product["valid_tau_mask"],
            "operator_definition": np.asarray(product["operator_definition"]),
            "input_scheme": np.asarray(product["input_scheme"]),
            "input_c3c2_path": np.asarray(str(product["path"])),
            "input_c3c2_sha256": np.asarray(product["sha256"]),
            "bootstrap_indices": product["indices"],
            "confs": product["confs"],
            "nconf": np.asarray(product["nconf"], dtype=np.int32),
            "nboot": np.asarray(product["nboot"], dtype=np.int32),
            "seed": np.asarray(product["seed"], dtype=np.int64),
            "bootstrap_method": np.asarray("iid configuration bootstrap with replacement; resample size Nconf"),
            "fit_code": np.asarray(str(Path(__file__).resolve())),
            "fit_code_sha256": np.asarray(sha256_file(Path(__file__).resolve())),
            "fit_bare_covariance_root": np.asarray(str(BARE_FIT_ROOT)),
            "command": np.asarray(command),
            "analysis_status": np.asarray(
                "finite_HYP_flowed_gluon_bare_direct_ratio_fit_not_physical_pdf"
            ),
        }
    )
    np.savez_compressed(output, **payload)
    receipt = {
        "ensemble": "L32x64",
        "channel": channel,
        "pabs": pabs,
        "directions": list(DIRECTIONS),
        "formulas": list(FORMULAS),
        "fit_method": "direct_correlated_ratio_plateau",
        "primary_candidate": PRIMARY_LABEL,
        "candidate_labels": [c.label for c in CANDIDATES],
        "nconf": product["nconf"],
        "nboot": product["nboot"],
        "seed": product["seed"],
        "input_c3c2": str(product["path"]),
        "input_c3c2_sha256": product["sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "fit_valid_count": int(np.count_nonzero(fit["fit_valid"])),
        "estimate_available_count": int(np.count_nonzero(fit["estimate_available"])),
        "fallback_count": int(np.count_nonzero(fit["central_fallback_used"])),
        "total_point_count": int(fit["fit_valid"].size),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "finite_HYP_bare_ratio_fit_diagnostic_not_physical_pdf",
    }
    output.with_suffix(output.suffix + ".done.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", choices=("unpolarized", "helicity"), required=True)
    parser.add_argument("--pabs", type=int, default=6)
    args = parser.parse_args()
    product = load_product(args.input, args.channel, args.pabs)
    fit = fit_product(product, args.channel, args.pabs)
    write_product(product, fit, args.channel, args.pabs, args.output, " ".join(sys.argv))


if __name__ == "__main__":
    main()
