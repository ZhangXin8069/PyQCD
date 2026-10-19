#!/usr/bin/env python3
"""Audit flowed-gluon bare fits and perform a correlated flow-time scan.

The input products are the Sumratio-difference AIC fit products.  This module
is deliberately separate from the fit producer: it never replaces rejected
fits by ``M_filled`` and never treats a fallback as an accepted matrix element.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Keep matplotlib's font/cache files in the analysis tree.  This makes batch
# runs reproducible and avoids warnings when the shared home cache is not
# writable on a compute node.
_MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution


TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)
COMPONENTS = (
    "unpolarized_TH_minus_SH_even_real",
    "helicity_TH_odd_imag",
    "helicity_TH_plus_SH_odd_imag",
)
COMPONENT_DISPLAY = {
    "unpolarized_TH_minus_SH_even_real": r"unpolarized: Re[$(T_U-S_U)_{\rm even}$]",
    "helicity_TH_odd_imag": r"helicity: Im[$(T_H)_{\rm odd}$]",
    "helicity_TH_plus_SH_odd_imag": r"helicity: Im[$(T_H+S_H)_{\rm odd}$]",
}
WINDOW_DISPLAY = {
    "small_0p1_0p3": r"small 0.1--0.3",
    "small_0p1_0p4": r"small 0.1--0.4",
    "small_0p1_0p5": r"small 0.1--0.5",
    "small_0p1_0p6": r"small 0.1--0.6 (primary)",
    "extended_0p1_1p0": r"extended 0.1--1.0",
    "extended_0p1_1p8": r"extended 0.1--1.8",
    "extended_0p1_2p6": r"extended 0.1--2.6",
    "extended_0p1_3p8": r"extended 0.1--3.8",
}
WINDOWS: dict[str, tuple[float, ...]] = {
    "small_0p1_0p3": (0.1, 0.2, 0.3),
    "small_0p1_0p4": (0.1, 0.2, 0.3, 0.4),
    "small_0p1_0p5": (0.1, 0.2, 0.3, 0.4, 0.5),
    "small_0p1_0p6": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    "extended_0p1_1p0": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0),
    "extended_0p1_1p8": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8),
    "extended_0p1_2p6": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6),
    "extended_0p1_3p8": TAUS,
}
PRIMARY_WINDOW = "small_0p1_0p6"
SCHEMA = "gradient_flow_gluon_flow_time_analysis_v1"
FIT_SCHEMA = "gradient_flow_gluon_bare_sumdiff_aic_v1"


@dataclass
class Products:
    taus: np.ndarray
    M: np.ndarray
    M_boot: np.ndarray
    statistical_error: np.ndarray
    fit_accepted: np.ndarray
    central_fallback_used: np.ndarray
    replica_fallback_fraction: np.ndarray
    cut_spread: np.ndarray
    usable_cut_count: np.ndarray
    components: tuple[str, ...]
    pabs: np.ndarray
    z: np.ndarray
    confs: np.ndarray
    boot_indices: np.ndarray
    nboot: int
    a_fm: float
    manifest_sha256: str
    code_hashes: tuple[str, ...]
    config_hashes: tuple[str, ...]
    source_products: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def _as_scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if np.ndim(value) != 0:
        raise ValueError(f"Expected scalar {key}, got {value.shape}")
    return value.item()


def _fit_path(fit_root: Path, tau: float) -> Path:
    return fit_root / "products" / f"bare_sumdiff_aic_{tau_tag(tau)}_N231.npz"


def load_products(fit_root: Path) -> Products:
    """Load and strictly audit all fourteen fit products."""

    arrays: dict[str, list[np.ndarray]] = {
        "M": [],
        "M_boot": [],
        "statistical_error": [],
        "fit_accepted": [],
        "central_fallback_used": [],
        "replica_fallback_fraction": [],
        "cut_spread_diagnostic": [],
        "usable_cut_count": [],
    }
    ref_confs: np.ndarray | None = None
    ref_boot: np.ndarray | None = None
    components: tuple[str, ...] | None = None
    pabs: np.ndarray | None = None
    z_values: np.ndarray | None = None
    manifest_sha: str | None = None
    code_hashes: list[str] = []
    config_hashes: list[str] = []
    source_products: list[str] = []
    nboot: int | None = None
    a_fm: float | None = None

    for tau in TAUS:
        path = _fit_path(fit_root, tau)
        done_path = Path(str(path) + ".done.json")
        if not path.is_file() or not done_path.is_file():
            raise FileNotFoundError(f"Missing fit product/receipt pair: {path}")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "complete" or done.get("schema") != FIT_SCHEMA:
            raise ValueError(f"Incomplete or wrong fit receipt: {done_path}")
        actual_sha = sha256_file(path)
        if done.get("output_sha256") != actual_sha:
            raise ValueError(f"Fit product checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != FIT_SCHEMA:
                raise ValueError(f"Wrong fit schema: {path}")
            if abs(float(_as_scalar(data, "flow_tau_t_over_a2")) - tau) > 1e-12:
                raise ValueError(f"Flow-time metadata mismatch: {path}")
            if int(_as_scalar(data, "nconf")) != 231 or int(_as_scalar(data, "nboot")) != 1500:
                raise ValueError(f"Unexpected sample count: {path}")
            if int(_as_scalar(data, "seed")) != 1115:
                raise ValueError(f"Unexpected bootstrap seed: {path}")
            required = tuple(arrays) + ("component_labels", "pabs_values", "z_values", "confs", "boot_indices")
            for key in required:
                if key not in data.files:
                    raise ValueError(f"Missing {key} in {path}")
            current_components = tuple(data["component_labels"].astype(str).tolist())
            current_pabs = np.asarray(data["pabs_values"], dtype=np.int32)
            current_z = np.asarray(data["z_values"], dtype=np.int32)
            current_confs = data["confs"].astype(str)
            current_boot = np.asarray(data["boot_indices"], dtype=np.int32)
            if current_components != COMPONENTS:
                raise ValueError(f"Component contract mismatch in {path}: {current_components}")
            if tuple(current_pabs.tolist()) != (3, 4, 5) or tuple(current_z.tolist()) != tuple(range(25)):
                raise ValueError(f"Momentum/z contract mismatch in {path}")
            if current_boot.shape != (1500, 231) or len(current_confs) != 231:
                raise ValueError(f"Bootstrap/configuration axes mismatch in {path}")
            if ref_confs is None:
                ref_confs, ref_boot = current_confs, current_boot
                components, pabs, z_values = current_components, current_pabs, current_z
                manifest_sha = str(data["manifest_sha256"])
                nboot, a_fm = int(_as_scalar(data, "nboot")), float(_as_scalar(data, "a_fm"))
            else:
                if not np.array_equal(ref_confs, current_confs) or not np.array_equal(ref_boot, current_boot):
                    raise ValueError(f"Shared configuration/bootstrap axes differ: {path}")
                if str(data["manifest_sha256"]) != manifest_sha:
                    raise ValueError(f"Manifest hash differs: {path}")
            M = np.asarray(data["M"], dtype=float)
            M_boot = np.asarray(data["M_boot"], dtype=float)
            error = np.asarray(data["statistical_error"], dtype=float)
            accepted = np.asarray(data["fit_accepted"], dtype=bool)
            fallback = np.asarray(data["central_fallback_used"], dtype=bool)
            replica_fallback = np.asarray(data["replica_fallback_fraction"], dtype=float)
            cut_spread = np.asarray(data["cut_spread_diagnostic"], dtype=float)
            usable_cuts = np.asarray(data["usable_cut_count"], dtype=np.int16)
            expected = (3, 3, 25)
            if M.shape != expected or error.shape != expected or accepted.shape != expected:
                raise ValueError(f"Central shape mismatch in {path}")
            if M_boot.shape != (1500,) + expected:
                raise ValueError(f"Bootstrap shape mismatch in {path}")
            if any(x.shape != expected for x in (fallback, replica_fallback, cut_spread, usable_cuts)):
                raise ValueError(f"Diagnostic shape mismatch in {path}")
            # The fit producer's audit contract: accepted points are finite and
            # rejected points remain NaN.  M_filled is intentionally ignored.
            if not np.isfinite(M[accepted]).all() or not np.isfinite(error[accepted]).all():
                raise ValueError(f"Non-finite accepted central fit in {path}")
            if np.any(~accepted & np.isfinite(M)) or np.any(~accepted & np.isfinite(error)):
                raise ValueError(f"Rejected central fit was populated in {path}")
            if not np.isfinite(M_boot[:, accepted]).all() or np.any(~accepted & np.isfinite(M_boot)):
                raise ValueError(f"Bootstrap accepted/rejected contract failed in {path}")
            arrays["M"].append(M)
            arrays["M_boot"].append(M_boot)
            arrays["statistical_error"].append(error)
            arrays["fit_accepted"].append(accepted)
            arrays["central_fallback_used"].append(fallback)
            arrays["replica_fallback_fraction"].append(replica_fallback)
            arrays["cut_spread_diagnostic"].append(cut_spread)
            arrays["usable_cut_count"].append(usable_cuts)
            code_hashes.append(str(done.get("code_sha256", _as_scalar(data, "code_sha256"))))
            config_hashes.append(str(done.get("config_sha256", _as_scalar(data, "config_sha256"))))
            source_products.append(str(path.resolve()))

    if components is None or pabs is None or z_values is None or ref_confs is None or ref_boot is None:
        raise RuntimeError("No fit products loaded")
    return Products(
        taus=np.asarray(TAUS, dtype=float),
        M=np.stack(arrays["M"], axis=0),
        M_boot=np.stack(arrays["M_boot"], axis=0),
        statistical_error=np.stack(arrays["statistical_error"], axis=0),
        fit_accepted=np.stack(arrays["fit_accepted"], axis=0),
        central_fallback_used=np.stack(arrays["central_fallback_used"], axis=0),
        replica_fallback_fraction=np.stack(arrays["replica_fallback_fraction"], axis=0),
        cut_spread=np.stack(arrays["cut_spread_diagnostic"], axis=0),
        usable_cut_count=np.stack(arrays["usable_cut_count"], axis=0),
        components=components,
        pabs=pabs,
        z=z_values,
        confs=ref_confs,
        boot_indices=ref_boot,
        nboot=int(nboot),
        a_fm=float(a_fm),
        manifest_sha256=str(manifest_sha),
        code_hashes=tuple(code_hashes),
        config_hashes=tuple(config_hashes),
        source_products=tuple(source_products),
    )


@dataclass(frozen=True)
class Covariance:
    inverse: np.ndarray
    eigenvalues: np.ndarray
    retained: np.ndarray
    sample_rank: int
    effective_rank: int
    condition: float
    shrinkage: float
    finite_fraction: float


def _matrix_rank(matrix: np.ndarray) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or not np.isfinite(singular[0]) or singular[0] <= 0.0:
        return 0
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    return int(np.count_nonzero(singular > tolerance))


def _nanmax_or_nan(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Return ``nanmax`` while preserving all-NaN slices as NaN.

    ``numpy.nanmax`` emits a RuntimeWarning for an all-NaN slice.  Such slices
    are expected here (for example the local ``z=0`` point is excluded from
    the scale-separated analysis), so use an explicit finite mask instead.
    The implementation also handles an empty reduction without raising.
    """

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    safe = np.where(finite, array, -np.inf)
    if axis is None:
        if not np.any(finite):
            return np.asarray(np.nan)
        return np.asarray(np.max(safe))
    reduced = np.max(safe, axis=axis)
    has_finite = np.any(finite, axis=axis)
    return np.where(has_finite, reduced, np.nan)


def _nanmean_or_nan(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Return a NaN-preserving finite mean without ``Mean of empty slice`` warnings."""

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    counts = np.sum(finite, axis=axis)
    totals = np.sum(np.where(finite, array, 0.0), axis=axis)
    return np.divide(totals, counts, out=np.full_like(np.asarray(totals, dtype=float), np.nan), where=counts > 0)


def estimate_covariance(samples: np.ndarray, svdcut: float = 1e-12) -> Covariance:
    """Schaefer--Strimmer correlation shrinkage, matching the fit contract."""

    values = np.asarray(samples, dtype=float)
    finite_rows = np.all(np.isfinite(values), axis=1)
    values = values[finite_rows]
    nrep, nobs = values.shape
    if nrep < 3 or nobs < 1:
        raise ValueError("At least three finite flow-time replicas are required")
    centered = values - values.mean(axis=0, keepdims=True)
    variances = np.sum(centered * centered, axis=0) / float(nrep - 1)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("Every flow-time point needs positive replica variance")
    scale = np.sqrt(variances)
    standardized = centered / scale[None, :]
    correlation = (standardized.T @ standardized) / float(nrep - 1)
    np.fill_diagonal(correlation, 1.0)
    sample_rank = _matrix_rank(correlation)
    if nobs == 1:
        shrinkage = 1.0
    else:
        numerator = denominator = 0.0
        for left in range(nobs):
            for right in range(left + 1, nobs):
                products = standardized[:, left] * standardized[:, right]
                numerator += np.sum((products - correlation[left, right]) ** 2) / float(nrep * (nrep - 1))
                denominator += correlation[left, right] ** 2
        shrinkage = 1.0 if denominator <= 0.0 else min(1.0, max(0.0, numerator / denominator))
    shrunk = (1.0 - shrinkage) * correlation + shrinkage * np.eye(nobs)
    covariance = scale[:, None] * shrunk * scale[None, :]
    mean_variance = float(np.trace(covariance) / nobs)
    covariance += max(1e-18, 1e-12 * max(mean_variance, 0.0)) * np.eye(nobs)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(np.max(eigenvalues))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Flow-time covariance has no positive eigenvalue")
    retained = eigenvalues > svdcut * largest
    if not np.any(retained):
        raise ValueError("SVD removed all flow-time covariance modes")
    inverse = (eigenvectors[:, retained] / eigenvalues[retained][None, :]) @ eigenvectors[:, retained].T
    return Covariance(
        inverse=inverse,
        eigenvalues=eigenvalues,
        retained=retained,
        sample_rank=sample_rank,
        effective_rank=int(np.count_nonzero(retained)),
        condition=largest / float(np.min(eigenvalues[retained])),
        shrinkage=float(shrinkage),
        finite_fraction=float(nrep / samples.shape[0]),
    )


@dataclass
class FitResult:
    intercept: float
    slope_tau: float
    intercept_boot: np.ndarray
    slope_tau_boot: np.ndarray
    chi2: float
    dof: int
    q_value: float
    chi2_dof: float
    covariance: Covariance
    fit_computed: bool
    quality_pass: bool
    status: str


def fit_flow_line(y: np.ndarray, samples: np.ndarray, tau: np.ndarray) -> FitResult:
    """Correlated two-parameter fit with a fixed central covariance projector."""

    y = np.asarray(y, dtype=float)
    samples = np.asarray(samples, dtype=float)
    tau = np.asarray(tau, dtype=float)
    nan_boot = np.full(samples.shape[0], np.nan)
    if y.ndim != 1 or samples.ndim != 2 or samples.shape[1] != len(y) or len(y) != len(tau):
        raise ValueError("Incompatible flow-time fit shapes")
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(tau).all():
        return FitResult(np.nan, np.nan, nan_boot, nan_boot.copy(), np.nan, 0, np.nan, np.nan,
                         Covariance(np.empty((0, 0)), np.empty(0), np.empty(0, bool), 0, 0, np.nan, np.nan, 0.0),
                         False, False, "insufficient_or_nonfinite_flow_points")
    try:
        covariance = estimate_covariance(samples)
        design = np.column_stack((np.ones(len(tau)), tau))
        if covariance.sample_rank < 2 or covariance.effective_rank <= 2:
            return FitResult(np.nan, np.nan, nan_boot, nan_boot.copy(), np.nan, 0, np.nan, np.nan,
                             covariance, False, False, "covariance_rank_not_above_parameters")
        normal = design.T @ covariance.inverse @ design
        scale = float(np.max(np.abs(normal)))
        if not np.isfinite(scale) or np.linalg.matrix_rank(normal, tol=1e-12 * scale) < 2:
            return FitResult(np.nan, np.nan, nan_boot, nan_boot.copy(), np.nan, 0, np.nan, np.nan,
                             covariance, False, False, "flow_design_singular")
        normal_inverse = np.linalg.pinv(normal, rcond=1e-12, hermitian=True)
        projector = normal_inverse @ design.T @ covariance.inverse
        parameters = projector @ y
        boot_parameters = np.full((samples.shape[0], 2), np.nan)
        complete = np.all(np.isfinite(samples), axis=1)
        boot_parameters[complete] = samples[complete] @ projector.T
        residual = y - design @ parameters
        chi2 = float(residual @ covariance.inverse @ residual)
        dof = int(covariance.effective_rank - 2)
        q_value = float(chi2_distribution.sf(chi2, dof)) if dof > 0 else np.nan
        chi2_dof = chi2 / dof if dof > 0 else np.nan
        success = covariance.finite_fraction >= 0.90
        quality = bool(dof > 0 and np.isfinite(q_value) and q_value >= 0.01 and np.isfinite(chi2_dof) and chi2_dof <= 3.0 and success)
        status = "quality_pass" if quality else "flow_fit_quality_fail"
        return FitResult(float(parameters[0]), float(parameters[1]), boot_parameters[:, 0], boot_parameters[:, 1],
                         chi2, dof, q_value, chi2_dof, covariance, True, quality, status)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        empty_cov = Covariance(np.empty((0, 0)), np.empty(0), np.empty(0, bool), 0, 0, np.nan, np.nan, 0.0)
        return FitResult(np.nan, np.nan, nan_boot, nan_boot.copy(), np.nan, 0, np.nan, np.nan,
                         empty_cov, False, False, f"flow_fit_exception:{type(exc).__name__}")


def _geometry_mask(taus: np.ndarray, z: np.ndarray, radius_factor: float) -> np.ndarray:
    if radius_factor <= 0.0:
        raise ValueError("radius_factor must be positive")
    radius = np.sqrt(8.0 * taus)
    # axes: tau,z.  z=0 is deliberately false for the scale-separated mask.
    return (z[None, :] > 0) & (radius[:, None] < radius_factor * z[None, :])


def _status_reason(*, candidate_taus: np.ndarray, effective: np.ndarray, accepted: np.ndarray,
                   z_value: int, include_z0: bool, fit: FitResult | None) -> str:
    if z_value == 0 and not include_z0:
        return "local_z0_excluded_from_scale_separated_extrapolation"
    if len(candidate_taus) < 3:
        return "fewer_than_three_flow_points_after_geometry_mask"
    if not np.all(effective):
        return "no_geometry_compatible_flow_points"
    if not np.all(accepted):
        return "one_or_more_input_fit_fallback_or_rejected"
    if fit is None:
        return "flow_fit_not_attempted"
    return fit.status


def run_extrapolations(products: Products, *, radius_factor: float = 1.0,
                       include_z0: bool = False, primary_window: str = PRIMARY_WINDOW) -> dict[str, np.ndarray]:
    if primary_window not in WINDOWS:
        raise ValueError(f"Unknown primary window {primary_window}")
    window_names = tuple(WINDOWS)
    nw, nc, npmom, nz = len(window_names), len(products.components), len(products.pabs), len(products.z)
    nboot = products.nboot
    intercept = np.full((nw, nc, npmom, nz), np.nan)
    slope_tau = np.full_like(intercept, np.nan)
    intercept_boot = np.full((nw, nboot, nc, npmom, nz), np.nan)
    slope_tau_boot = np.full_like(intercept_boot, np.nan)
    chi2 = np.full_like(intercept, np.nan)
    dof = np.zeros_like(intercept, dtype=np.int32)
    q_value = np.full_like(intercept, np.nan)
    chi2_dof = np.full_like(intercept, np.nan)
    fit_computed = np.zeros_like(intercept, dtype=bool)
    quality_pass = np.zeros_like(intercept, dtype=bool)
    input_all_accepted = np.zeros_like(intercept, dtype=bool)
    geometry_all_valid = np.zeros_like(intercept, dtype=bool)
    n_points = np.zeros_like(intercept, dtype=np.int32)
    n_geometry_excluded = np.zeros_like(intercept, dtype=np.int32)
    max_replica_fallback = np.full_like(intercept, np.nan)
    mean_replica_fallback = np.full_like(intercept, np.nan)
    max_cut_spread = np.full_like(intercept, np.nan)
    max_cut_spread_over_error = np.full_like(intercept, np.nan)
    mean_cut_spread_over_error = np.full_like(intercept, np.nan)
    covariance_sample_rank = np.zeros_like(intercept, dtype=np.int32)
    covariance_effective_rank = np.zeros_like(intercept, dtype=np.int32)
    covariance_condition = np.full_like(intercept, np.nan)
    covariance_shrinkage = np.full_like(intercept, np.nan)
    finite_fraction = np.full_like(intercept, np.nan)
    status = np.full((nw, nc, npmom, nz), "not_evaluated", dtype="U128")
    geometry = _geometry_mask(products.taus, products.z, radius_factor)

    for iw, name in enumerate(window_names):
        window_taus = np.asarray(WINDOWS[name], dtype=float)
        indices = np.asarray([int(np.flatnonzero(np.isclose(products.taus, t, atol=1e-12))[0]) for t in window_taus])
        for ic in range(nc):
            for ip in range(npmom):
                for iz, z_value in enumerate(products.z):
                    geom_values = geometry[indices, iz].copy()
                    if include_z0 and z_value == 0:
                        geom_values[:] = True
                    effective_indices = indices[geom_values]
                    effective_taus = products.taus[effective_indices]
                    n_points[iw, ic, ip, iz] = len(effective_indices)
                    n_geometry_excluded[iw, ic, ip, iz] = len(indices) - len(effective_indices)
                    if len(effective_indices):
                        accepted_values = products.fit_accepted[effective_indices, ic, ip, iz]
                        fallback_values = products.central_fallback_used[effective_indices, ic, ip, iz]
                        rep_values = products.replica_fallback_fraction[effective_indices, ic, ip, iz]
                        spread_values = products.cut_spread[effective_indices, ic, ip, iz]
                        error_values = products.statistical_error[effective_indices, ic, ip, iz]
                        input_all_accepted[iw, ic, ip, iz] = bool(np.all(accepted_values))
                        geometry_all_valid[iw, ic, ip, iz] = bool(np.all(geom_values))
                        max_replica_fallback[iw, ic, ip, iz] = float(_nanmax_or_nan(rep_values))
                        mean_replica_fallback[iw, ic, ip, iz] = float(_nanmean_or_nan(rep_values))
                        max_cut_spread[iw, ic, ip, iz] = float(np.nanmax(spread_values)) if np.any(np.isfinite(spread_values)) else np.nan
                        normalized = np.divide(np.abs(spread_values), np.maximum(np.abs(error_values), 1e-15),
                                              out=np.full_like(spread_values, np.nan), where=np.isfinite(spread_values) & np.isfinite(error_values))
                        max_cut_spread_over_error[iw, ic, ip, iz] = float(np.nanmax(normalized)) if np.any(np.isfinite(normalized)) else np.nan
                        mean_cut_spread_over_error[iw, ic, ip, iz] = float(np.nanmean(normalized)) if np.any(np.isfinite(normalized)) else np.nan
                    if z_value == 0 and not include_z0:
                        status[iw, ic, ip, iz] = _status_reason(candidate_taus=effective_taus, effective=geom_values,
                                                                  accepted=np.zeros(len(effective_indices), dtype=bool), z_value=int(z_value), include_z0=include_z0, fit=None)
                        continue
                    if len(effective_indices) < 3:
                        status[iw, ic, ip, iz] = _status_reason(candidate_taus=effective_taus, effective=geom_values,
                                                                  accepted=np.zeros(len(effective_indices), dtype=bool), z_value=int(z_value), include_z0=include_z0, fit=None)
                        continue
                    accepted_values = products.fit_accepted[effective_indices, ic, ip, iz]
                    if not np.all(accepted_values):
                        status[iw, ic, ip, iz] = "one_or_more_input_fit_fallback_or_rejected"
                        continue
                    y = products.M[effective_indices, ic, ip, iz]
                    samples = np.transpose(products.M_boot[effective_indices, :, ic, ip, iz], (1, 0))
                    fit = fit_flow_line(y, samples, effective_taus)
                    intercept[iw, ic, ip, iz] = fit.intercept
                    slope_tau[iw, ic, ip, iz] = fit.slope_tau
                    intercept_boot[iw, :, ic, ip, iz] = fit.intercept_boot
                    slope_tau_boot[iw, :, ic, ip, iz] = fit.slope_tau_boot
                    chi2[iw, ic, ip, iz] = fit.chi2
                    dof[iw, ic, ip, iz] = fit.dof
                    q_value[iw, ic, ip, iz] = fit.q_value
                    chi2_dof[iw, ic, ip, iz] = fit.chi2_dof
                    fit_computed[iw, ic, ip, iz] = fit.fit_computed
                    quality_pass[iw, ic, ip, iz] = fit.quality_pass
                    covariance_sample_rank[iw, ic, ip, iz] = fit.covariance.sample_rank
                    covariance_effective_rank[iw, ic, ip, iz] = fit.covariance.effective_rank
                    covariance_condition[iw, ic, ip, iz] = fit.covariance.condition
                    covariance_shrinkage[iw, ic, ip, iz] = fit.covariance.shrinkage
                    finite_fraction[iw, ic, ip, iz] = fit.covariance.finite_fraction
                    status[iw, ic, ip, iz] = fit.status

    window_tau_matrix = np.full((nw, max(len(WINDOWS[name]) for name in window_names)), np.nan, dtype=float)
    window_lengths = np.zeros(nw, dtype=np.int32)
    for iw, name in enumerate(window_names):
        values = np.asarray(WINDOWS[name], dtype=float)
        window_tau_matrix[iw, :len(values)] = values
        window_lengths[iw] = len(values)
    intercept_error = np.full_like(intercept, np.nan)
    finite_intercepts = np.isfinite(intercept_boot)
    enough = np.sum(finite_intercepts, axis=1) >= 2
    counts = np.sum(finite_intercepts, axis=1)
    means = np.divide(
        np.nansum(intercept_boot, axis=1), counts,
        out=np.full_like(intercept, np.nan), where=counts > 0,
    )
    centered = np.where(finite_intercepts, intercept_boot - means[:, None, ...], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        intercept_error[enough] = np.sqrt(np.nansum(centered * centered, axis=1)[enough] / np.maximum(counts[enough] - 1, 1))
    return {
        "schema": np.asarray(SCHEMA),
        "analysis_status": np.asarray("finite_flow_bare_tau0_extrapolation_not_physical_pdf"),
        "window_names": np.asarray(window_names),
        "window_taus": window_tau_matrix,
        "window_lengths": window_lengths,
        "primary_window": np.asarray(primary_window),
        "flow_taus": products.taus,
        "flow_radius_over_a": np.sqrt(8.0 * products.taus),
        "a_fm": np.asarray(products.a_fm),
        "radius_factor": np.asarray(radius_factor),
        "include_z0": np.asarray(include_z0),
        "components": np.asarray(products.components),
        "pabs_values": products.pabs,
        "z_values": products.z,
        "intercept": intercept,
        "slope_tau": slope_tau,
        "slope_fm_minus2": slope_tau / (products.a_fm ** 2),
        "intercept_boot": intercept_boot,
        "slope_tau_boot": slope_tau_boot,
        "intercept_error": intercept_error,
        "chi2": chi2,
        "dof": dof,
        "q_value": q_value,
        "chi2_dof": chi2_dof,
        "fit_computed": fit_computed,
        "quality_pass": quality_pass,
        "input_all_accepted": input_all_accepted,
        "geometry_all_valid": geometry_all_valid,
        "n_points": n_points,
        "n_geometry_excluded": n_geometry_excluded,
        "max_replica_fallback_fraction": max_replica_fallback,
        "mean_replica_fallback_fraction": mean_replica_fallback,
        "max_cut_spread": max_cut_spread,
        "max_cut_spread_over_error": max_cut_spread_over_error,
        "mean_cut_spread_over_error": mean_cut_spread_over_error,
        "covariance_sample_rank": covariance_sample_rank,
        "covariance_effective_rank": covariance_effective_rank,
        "covariance_condition": covariance_condition,
        "covariance_shrinkage": covariance_shrinkage,
        "finite_fraction": finite_fraction,
        "status": status,
        "fit_nboot": np.asarray(products.nboot),
        "fit_seed": np.asarray(1115),
        "fit_manifest_sha256": np.asarray(products.manifest_sha256),
        "fit_code_hashes": np.asarray(products.code_hashes),
        "fit_config_hashes": np.asarray(products.config_hashes),
        "source_products": np.asarray(products.source_products),
        "confs": products.confs,
        "boot_indices": products.boot_indices,
    }


def _json_value(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_value(value.item())
        return [_json_value(x) for x in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_summary(results: dict[str, np.ndarray], output_root: Path, products: Products) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    npz_path = output_root / "flow_time_extrapolation_v1.npz"
    npz_tmp = output_root / f"{npz_path.name}.tmp.{os.getpid()}.npz"
    np.savez_compressed(npz_tmp, **results)
    os.replace(npz_tmp, npz_path)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "analysis_status": "finite_flow_bare_tau0_extrapolation_not_physical_pdf",
        "output": str(npz_path.resolve()),
        "output_sha256": sha256_file(npz_path),
        "flow_taus": TAUS,
        "primary_window": str(results["primary_window"]),
        "radius_factor": float(results["radius_factor"]),
        "include_z0": bool(results["include_z0"]),
        "nconf": 231,
        "nboot": 1500,
        "seed": 1115,
        "components": COMPONENTS,
        "pabs": [3, 4, 5],
        "z_values": list(range(25)),
        "manifest_sha256": products.manifest_sha256,
        "fit_code_hashes": sorted(set(products.code_hashes)),
        "fit_config_hashes": sorted(set(products.config_hashes)),
        "analysis_code_sha256": str(results.get("analysis_code_sha256", "")),
        "window_counts": {},
    }
    statuses = results["status"]
    fit_computed = results["fit_computed"]
    quality_pass = results["quality_pass"]
    for iw, name in enumerate(WINDOWS):
        values = statuses[iw].astype(str).ravel().tolist()
        counts: dict[str, int] = {}
        for item in values:
            counts[item] = counts.get(item, 0) + 1
        summary["window_counts"][name] = {
            "status_counts": counts,
            "fit_computed": int(np.count_nonzero(fit_computed[iw])),
            "quality_pass": int(np.count_nonzero(quality_pass[iw])),
        }
    json_tmp = output_root / f"{npz_path.name}.json.tmp.{os.getpid()}"
    json_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(json_tmp, output_root / "flow_time_extrapolation_v1.json")

    # A row-wise table is easier to inspect than a high-dimensional NPZ.
    csv_path = output_root / "flow_time_extrapolation_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "window", "component", "Pz", "z", "intercept", "intercept_error", "slope_tau",
            "slope_fm_minus2", "chi2", "dof", "Q", "chi2_dof", "fit_computed", "quality_pass",
            "input_all_accepted", "n_points", "n_geometry_excluded", "max_replica_fallback_fraction",
            "mean_replica_fallback_fraction", "max_cut_spread", "max_cut_spread_over_error",
            "mean_cut_spread_over_error", "covariance_sample_rank", "covariance_effective_rank",
            "covariance_condition", "covariance_shrinkage", "finite_fraction", "status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for iw, name in enumerate(WINDOWS):
            for ic, component in enumerate(COMPONENTS):
                for ip, momentum in enumerate(products.pabs):
                    for iz, z_value in enumerate(products.z):
                        def scalar(key: str):
                            return _json_value(results[key][iw, ic, ip, iz])
                        writer.writerow({
                            "window": name, "component": component, "Pz": int(momentum), "z": int(z_value),
                            "intercept": scalar("intercept"), "intercept_error": scalar("intercept_error"),
                            "slope_tau": scalar("slope_tau"), "slope_fm_minus2": scalar("slope_fm_minus2"),
                            "chi2": scalar("chi2"), "dof": scalar("dof"), "Q": scalar("q_value"),
                            "chi2_dof": scalar("chi2_dof"), "fit_computed": scalar("fit_computed"),
                            "quality_pass": scalar("quality_pass"), "input_all_accepted": scalar("input_all_accepted"),
                            "n_points": scalar("n_points"), "n_geometry_excluded": scalar("n_geometry_excluded"),
                            "max_replica_fallback_fraction": scalar("max_replica_fallback_fraction"),
                            "mean_replica_fallback_fraction": scalar("mean_replica_fallback_fraction"),
                            "max_cut_spread": scalar("max_cut_spread"),
                            "max_cut_spread_over_error": scalar("max_cut_spread_over_error"),
                            "mean_cut_spread_over_error": scalar("mean_cut_spread_over_error"),
                            "covariance_sample_rank": scalar("covariance_sample_rank"),
                            "covariance_effective_rank": scalar("covariance_effective_rank"),
                            "covariance_condition": scalar("covariance_condition"),
                            "covariance_shrinkage": scalar("covariance_shrinkage"),
                            "finite_fraction": scalar("finite_fraction"), "status": str(results["status"][iw, ic, ip, iz]),
                        })


def write_report(results: dict[str, np.ndarray], output_root: Path, products: Products) -> Path:
    """Write a compact human-readable audit of the flow-window scan.

    The NPZ/CSV files remain the authoritative numerical products.  This
    report only summarizes acceptance, geometry, and window-to-window
    stability so that a later analysis does not accidentally promote a failed
    or fallback point to the primary result.
    """

    names = [str(x) for x in results["window_names"]]
    primary = names.index(str(results["primary_window"]))
    analysis_hash = str(results.get("analysis_code_sha256", ""))
    lines = [
        "# Gradient-flow gluon flow-time analysis",
        "",
        "> Finite-flow bare disconnected matrix-element diagnostic; not a",
        "> renormalized, matched, continuum gluon PDF.",
        "",
        "## Input and estimator",
        "",
        f"- Fit products: `{products.source_products[0].rsplit('/products/', 1)[0]}`",
        f"- Flow times `tau/a^2`: {', '.join(f'{tau:g}' for tau in products.taus)}",
        f"- Ensemble: `L32x96`, `Nconf={len(products.confs)}`, `Nboot={products.nboot}`, seed `1115`",
        f"- Manifest SHA-256: `{products.manifest_sha256}`",
        f"- Analysis code SHA-256: `{analysis_hash}`" if analysis_hash else "- Analysis code SHA-256: unavailable",
        "- Input fields: accepted (`fit_accepted`) only; `M_filled` and all fallback points are excluded.",
        "- Flow model: correlated GLS `M(tau)=M0+c_tau*tau`; bootstrap error is the replica standard deviation (ddof=1).",
        f"- With `a={products.a_fm:g} fm`, the primary `tau=0.1--0.6` window spans `r_flow={np.sqrt(8*0.1)*products.a_fm:.3f}--{np.sqrt(8*0.6)*products.a_fm:.3f} fm`.",
        "",
        "## Operators retained",
        "",
        "| component | final projection |",
        "|---|---|",
        "| `unpolarized_TH_minus_SH_even_real` | Re[(T_U-S_U)_even] |",
        "| `helicity_TH_odd_imag` | Im[(T_H)_odd] |",
        "| `helicity_TH_plus_SH_odd_imag` | Im[(T_H+S_H)_odd] |",
        "",
        "Both helicity constructions are carried through every flow window independently.",
        "",
        "## Geometry and flow windows",
        "",
        "The default scale-separated mask is `r_flow/a=sqrt(8*tau/a^2) < z/a`.",
        "The local point `z=0` is retained in raw flow-dependence plots but excluded from the tau-to-zero fit.",
        "At `z/a=1` only `tau/a^2=0.1` survives, so no two-parameter line fit is attempted.",
        "For example, `tau/a^2=0.5, z/a=2` is rejected because `r_flow/a=2` exactly.",
        "",
        "| window | flow times | fit-computed | quality-pass |",
        "|---|---:|---:|---:|",
    ]
    for iw, name in enumerate(names):
        lines.append(
            f"| `{name}` | {len(WINDOWS[name])} | {int(np.count_nonzero(results['fit_computed'][iw]))} "
            f"| {int(np.count_nonzero(results['quality_pass'][iw]))} |"
        )
    lines.extend([
        "",
        f"Primary window: `{names[primary]}`.  Alternate windows are stability diagnostics, not point-by-point selections.",
        "",
        "## Primary-window diagnostics",
        "",
        "| component | fits computed | quality pass | max replica fallback fraction | median max cut-spread/error |",
        "|---|---:|---:|---:|---:|",
    ])
    for ic, component in enumerate(products.components):
        computed = results["fit_computed"][primary, ic]
        quality = results["quality_pass"][primary, ic]
        fallback = _nanmax_or_nan(results["max_replica_fallback_fraction"][primary, ic])
        spread_values = results["max_cut_spread_over_error"][primary, ic]
        finite_spread = spread_values[np.isfinite(spread_values)]
        median_spread = float(np.median(finite_spread)) if len(finite_spread) else np.nan
        lines.append(
            f"| `{component}` | {int(np.count_nonzero(computed))} | {int(np.count_nonzero(quality))} "
            f"| {_json_value(fallback)} | {_json_value(median_spread)} |"
        )
    lines.extend([
        "",
        "### Primary `M0` at `z/a=2`",
        "",
        "| component | Pz=3 | Pz=4 | Pz=5 |",
        "|---|---:|---:|---:|",
    ])
    iz2 = int(np.flatnonzero(products.z == 2)[0])
    for ic, component in enumerate(products.components):
        cells = []
        for ip, momentum in enumerate(products.pabs):
            if bool(results["quality_pass"][primary, ic, ip, iz2]):
                value = results["intercept"][primary, ic, ip, iz2]
                error = results["intercept_error"][primary, ic, ip, iz2]
                cells.append(f"{value:.6g} +/- {error:.3g}")
            else:
                cells.append("not quality-pass")
        lines.append(f"| `{component}` | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Window stability (common quality-pass points)",
        "",
        "The following RMS/median shifts are diagnostics only; they are not added to the statistical error.",
        "",
        "| alternate window | component | common points | RMS shift | median absolute shift |",
        "|---|---|---:|---:|---:|",
    ])
    for iw, name in enumerate(names):
        if iw == primary:
            continue
        for ic, component in enumerate(products.components):
            common = results["quality_pass"][primary, ic] & results["quality_pass"][iw, ic]
            delta = results["intercept"][iw, ic][common] - results["intercept"][primary, ic][common]
            delta = delta[np.isfinite(delta)]
            if len(delta):
                rms = float(np.sqrt(np.mean(delta * delta)))
                median_abs = float(np.median(np.abs(delta)))
                rms_text, median_text = f"{rms:.6g}", f"{median_abs:.6g}"
            else:
                rms_text = median_text = "n/a"
            lines.append(f"| `{name}` | `{component}` | {len(delta)} | {rms_text} | {median_text} |")
    lines.extend([
        "",
        "## Interpretation and next gates",
        "",
        "- The tau-to-zero intercept is a finite-lattice-spacing, finite-flow-scheme diagnostic.",
        "- Extended windows are oversmearing/stability checks; the small `0.1--0.6` window is the registered primary.",
        "- Before a PDF interpretation, repeat at controlled physical flow radius and lattice spacing, perform operator renormalization and gluon--singlet mixing, then apply the appropriate LaMET/pseudo-PDF matching and finite-coordinate analysis.",
    ])
    report_path = output_root / "flow_time_audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _plot_flow_dependence(products: Products, output_root: Path) -> list[Path]:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(products.z)))
    for ic, component in enumerate(products.components):
        for ip, momentum in enumerate(products.pabs):
            fig, ax = plt.subplots(figsize=(8.2, 5.4))
            for iz, z_value in enumerate(products.z):
                accepted = products.fit_accepted[:, ic, ip, iz]
                if not np.any(accepted):
                    continue
                ax.errorbar(products.taus[accepted], products.M[accepted, ic, ip, iz],
                            yerr=products.statistical_error[accepted, ic, ip, iz],
                            fmt="o", linestyle="none", ms=2.8, capsize=1.5, elinewidth=0.55,
                            color=colors[iz], alpha=0.68)
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=24))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=r"$z/a$")
            ax.set_xlabel(r"flow time $\tau=t/a^2$")
            ax.set_ylabel("accepted bare matrix element")
            ax.set_title(f"{COMPONENT_DISPLAY.get(component, component)}, $P_z={momentum}(2\\pi/L)$")
            ax.grid(alpha=0.22)
            fig.tight_layout()
            stem = plot_root / f"flow_dependence_{component}_Pz{int(momentum)}"
            for suffix in (".png", ".pdf"):
                path = stem.with_suffix(suffix)
                fig.savefig(path, dpi=220, bbox_inches="tight")
                outputs.append(path)
            plt.close(fig)
    return outputs


def _plot_diagnostics(products: Products, output_root: Path) -> list[Path]:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    # Max over the registered flow scan; this is a warning diagnostic, not a
    # new acceptance cut.
    max_rep = _nanmax_or_nan(products.replica_fallback_fraction, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_by_error = np.abs(products.cut_spread) / np.maximum(np.abs(products.statistical_error), 1e-15)
        spread_ratio = _nanmax_or_nan(spread_by_error, axis=0)
    for ic, component in enumerate(products.components):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
        for ax, values, title, vmax in (
            (axes[0], max_rep[ic], "max replica fallback fraction", 1.0),
            (axes[1], spread_ratio[ic], r"max cut spread / $\sigma_{fit}$", None),
        ):
            image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
                              extent=(products.z[0] - 0.5, products.z[-1] + 0.5,
                                      products.pabs[0] - 0.5, products.pabs[-1] + 0.5),
                              vmin=0.0, vmax=vmax)
            ax.set_xlabel(r"$z/a$")
            ax.set_ylabel(r"$P_z/(2\pi/L)$")
            ax.set_title(title)
            fig.colorbar(image, ax=ax, shrink=0.88)
        fig.suptitle(COMPONENT_DISPLAY.get(component, component) + " — inspect, do not silently cut")
        stem = plot_root / f"diagnostics_{component}"
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
    return outputs


def _plot_extrapolation(results: dict[str, np.ndarray], products: Products, output_root: Path) -> list[Path]:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    names = [str(x) for x in results["window_names"]]
    primary = names.index(str(results["primary_window"]))
    # One figure per operator, with all three momenta.  Filled markers are the
    # fixed primary window; open markers are registered alternatives.
    for ic, component in enumerate(products.components):
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
        for ip, momentum in enumerate(products.pabs):
            ax = axes[ip]
            for iw, name in enumerate(names):
                valid = results["fit_computed"][iw, ic, ip] & results["quality_pass"][iw, ic, ip]
                if iw == primary:
                    marker, face, alpha, size = "o", "#1f77b4", 0.95, 4.4
                elif name.startswith("small"):
                    marker, face, alpha, size = "s", "none", 0.55, 3.4
                else:
                    marker, face, alpha, size = "^", "none", 0.42, 3.0
                if np.any(valid):
                    ax.errorbar(products.z[valid] + (iw - primary) * 0.035,
                                results["intercept"][iw, ic, ip, valid],
                                yerr=results["intercept_error"][iw, ic, ip, valid],
                                fmt=marker, linestyle="none", markerfacecolor=face,
                                markeredgecolor="#1f77b4", color="#1f77b4", alpha=alpha,
                                ms=size, capsize=1.7, elinewidth=0.6,
                                label=WINDOW_DISPLAY.get(name, name) if ip == 2 else None)
            ax.axhline(0.0, color="0.45", ls="--", lw=0.65)
            ax.set_title(rf"$P_z={int(momentum)}(2\pi/L)$")
            ax.set_xlabel(r"$z/a$")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel(r"finite-$a$ extrapolated $M(\tau\to0)$")
        handles, labels = axes[-1].get_legend_handles_labels()
        # Keep the eight window labels separate from the operator title.  The
        # previous top-legend placement made long helicity labels overlap the
        # title in the rendered PDF/PNG.
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                       ncol=4, frameon=False, fontsize=8)
        fig.suptitle(COMPONENT_DISPLAY.get(component, component) + " — accepted flow fits only", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
        stem = plot_root / f"tau0_extrapolation_{component}"
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            outputs.append(path)
        plt.close(fig)
    return outputs


def _plot_geometry(products: Products, output_root: Path, radius_factor: float) -> list[Path]:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(products.z, products.z * radius_factor, color="#1f77b4", label=r"$z/a$ boundary")
    for tau, radius in zip(products.taus, np.sqrt(8 * products.taus)):
        ax.axhline(radius, lw=0.8, alpha=0.45, label=rf"$\tau={tau:g}$, $r/a={radius:.2f}$")
    ax.set_xlabel(r"nonlocal distance $z/a$")
    ax.set_ylabel(r"flow radius $r_{flow}/a$")
    ax.set_title(r"Geometry guard: keep $r_{flow}/a < z/a$")
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(0, max(np.sqrt(8 * products.taus)) * 1.12)
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    outputs = []
    stem = plot_root / "flow_radius_geometry"
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def run(args: argparse.Namespace) -> None:
    fit_root = Path(args.fit_root).resolve()
    output_root = Path(args.output_root).resolve()
    products = load_products(fit_root)
    results = run_extrapolations(products, radius_factor=args.radius_factor,
                                 include_z0=args.include_z0, primary_window=args.primary_window)
    results["analysis_code_sha256"] = np.asarray(sha256_file(Path(__file__).resolve()))
    write_summary(results, output_root, products)
    report_path = write_report(results, output_root, products)
    plot_paths = []
    if not args.no_plots:
        plot_paths += _plot_flow_dependence(products, output_root)
        plot_paths += _plot_diagnostics(products, output_root)
        plot_paths += _plot_extrapolation(results, products, output_root)
        plot_paths += _plot_geometry(products, output_root, args.radius_factor)
    print(json.dumps({
        "status": "complete",
        "output_root": str(output_root),
        "fit_root": str(fit_root),
        "n_products": len(products.taus),
        "nboot": products.nboot,
        "primary_window": args.primary_window,
        "radius_factor": args.radius_factor,
        "include_z0": args.include_z0,
        "plot_count": len(plot_paths),
        "report": str(report_path.resolve()),
        "summary": str((output_root / "flow_time_extrapolation_v1.json").resolve()),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", default="/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/bare_fit_sumdiff_aic_v1")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--radius-factor", type=float, default=1.0,
                        help="keep r_flow/a < radius_factor*z/a (default: 1.0)")
    parser.add_argument("--include-z0", action="store_true",
                        help="allow local z=0 in extrapolation; not recommended for the scale-separated result")
    parser.add_argument("--primary-window", choices=tuple(WINDOWS), default=PRIMARY_WINDOW)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
