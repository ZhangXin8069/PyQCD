#!/usr/bin/env python3
"""Fit latest flowed-gluon bootstrap summed ratios to bare matrix elements.

This is the finite-flow, bare analogue of the registered forward-
Sumratio-difference AIC fit.  The input is the new N406/Nboot=1500 product;
the fit threshold is fixed at T_min*a > 0.6 fm, hence T_min=8 for L32x96.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2 as chi2_distribution


ROOT = Path(__file__).resolve().parents[1]
FIT_CODE = Path(
    "/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/"
    "bare_fit_sumdiff_aic_code"
)
if str(FIT_CODE) not in sys.path:
    sys.path.insert(0, str(FIT_CODE))

from gluonfit.model_average import fill_with_minimum_chi2, replica_model_average  # noqa: E402
from gluonfit.statistics import correlated_gls_fit  # noqa: E402


SCHEMA = "gradient_flow_gluon_bare_sumdiff_aic_latest_v1"
INPUT_SCHEMA = "gradient_flow_gluon_c3_c2_bootstrap_v1"
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)
OPERATORS = ("unpolarized_TH_minus_SH_even_real", "helicity_TH_odd_imag", "helicity_TH_plus_SH_odd_imag")
TMIN = 8
A_FM = 0.0775
CUTS = np.asarray((1, 2, 3), dtype=np.int32)
TSEPS = np.arange(5, 16, dtype=np.int32)
PABS = np.asarray((3, 4, 5), dtype=np.int32)
Z = np.arange(25, dtype=np.int32)

STATS = SimpleNamespace(
    q_min=0.01,
    correlated_chi2_dof_max=3.0,
    bootstrap_min_success=0.90,
    svdcut=1e-12,
    normal_rcond=1e-12,
    covariance_relative_jitter=1e-12,
    covariance_absolute_jitter=1e-18,
    covariance_shrinkage="schaefer_strimmer_correlation_to_identity",
)


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_input(path: Path):
    if not path.is_file() or not Path(str(path) + ".done.json").is_file():
        raise FileNotFoundError(f"missing bootstrap product pair: {path}")
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != INPUT_SCHEMA:
            raise ValueError(f"wrong input schema: {path}")
        if int(data["nconf"]) != 406 or int(data["nboot"]) != 1500 or int(data["seed"]) != 1115:
            raise ValueError(f"wrong bootstrap contract: {path}")
        if tuple(data["ratio_sum_bootstrap"].shape) != (1500, 3, 3, 25, 11, 3):
            raise ValueError(f"wrong summed-ratio shape: {path}")
        if tuple(data["ratio_sum_original"].shape) != (3, 3, 25, 11, 3):
            raise ValueError(f"wrong central summed-ratio shape: {path}")
        # Convert to axes (operator,cut,p,z,T) and (boot,operator,cut,p,z,T).
        central = np.transpose(data["ratio_sum_original"], (0, 1, 4, 2, 3))
        bootstrap = np.transpose(data["ratio_sum_bootstrap"], (0, 1, 2, 5, 3, 4))
        valid = np.asarray(data["valid_cut_tsep_mask"], dtype=bool)
        if not np.isfinite(central[:, :, :, :, 3:]).all() or not np.isfinite(bootstrap[:, :, :, :, :, 3:]).all():
            raise ValueError(f"non-finite legal summed ratios: {path}")
        return {
            "central": np.asarray(central, dtype=float),
            "bootstrap": np.asarray(bootstrap, dtype=float),
            "valid": valid,
            "confs": np.asarray(data["confs"]),
            "indices": np.asarray(data["bootstrap_indices"]),
            "input_sha256": sha256_file(path),
        }


def candidate_fit(central: np.ndarray, bootstrap: np.ndarray,
                  cut_index: int, tmax: int, operator: int):
    """Fit one cut/window, returning central and replica plateau values."""
    lower = np.arange(TMIN, tmax, dtype=np.int32)
    locations = np.asarray([int(np.flatnonzero(TSEPS == T)[0]) for T in lower])
    y = central[operator, cut_index][:, :, locations].transpose(0, 1, 2)
    yb = bootstrap[:, operator, cut_index][:, :, :, locations].transpose(0, 1, 2, 3)
    # Bootstrap input is boot,p,z,T; the transpose above is a no-op in shape.
    npabs, ndz, ndata = y.shape
    values = np.full((npabs, ndz), np.nan)
    values_boot = np.full((bootstrap.shape[0], npabs, ndz), np.nan)
    chi2 = np.full((npabs, ndz), np.nan)
    dof = np.zeros((npabs, ndz), dtype=np.int32)
    qval = np.full((npabs, ndz), np.nan)
    chi2dof = np.full((npabs, ndz), np.nan)
    success = np.zeros((npabs, ndz))
    rank = np.zeros((npabs, ndz), dtype=np.int32)
    cond = np.full((npabs, ndz), np.nan)
    shrink = np.full((npabs, ndz), np.nan)
    aic = np.full((npabs, ndz), np.nan)
    aic_boot = np.full((bootstrap.shape[0], npabs, ndz), np.nan)
    chi2_boot = np.full_like(aic_boot, np.nan)
    succeeded = np.zeros((npabs, ndz), dtype=bool)
    eligible = np.zeros((npabs, ndz), dtype=bool)
    reason = np.full((npabs, ndz), "not_evaluated", dtype="U128")
    design = np.ones((ndata, 1), dtype=float)
    for p in range(npabs):
        for z in range(ndz):
            yy, rr = y[p, z], yb[:, p, z]
            if ndata == 1:
                values[p, z] = yy[0]
                values_boot[:, p, z] = rr[:, 0]
                succeeded[p, z] = np.isfinite(yy[0]) and np.isfinite(rr).all()
                aic[p, z] = 2.0 + 2.0 * (15 - tmax)
                aic_boot[:, p, z] = aic[p, z]
                reason[p, z] = "single_difference_zero_dof_diagnostic"
                continue
            try:
                result = correlated_gls_fit(yy, rr, design, STATS)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                reason[p, z] = f"fit_exception:{type(exc).__name__}"
                continue
            values[p, z] = result.parameters[0]
            values_boot[:, p, z] = result.parameter_boot[:, 0]
            residual_boot = rr - values_boot[:, p, z, None]
            complete = np.all(np.isfinite(residual_boot), axis=1)
            chi2_b = np.full(len(rr), np.nan)
            chi2_b[complete] = np.einsum("bi,ij,bj->b", residual_boot[complete], result.covariance.inverse, residual_boot[complete], optimize=True)
            score = result.chi2 + 2.0 + 2.0 * (15 - tmax)
            values_boot[~complete, p, z] = np.nan
            values[p, z] = float(result.parameters[0])
            chi2[p, z] = result.chi2
            dof[p, z] = result.dof
            qval[p, z] = result.q_value
            chi2dof[p, z] = result.chi2 / result.dof if result.dof > 0 else np.nan
            success[p, z] = result.bootstrap_success_fraction
            rank[p, z] = result.covariance.effective_rank
            cond[p, z] = result.covariance.condition
            shrink[p, z] = result.covariance.shrinkage_lambda
            aic[p, z] = score
            chi2_boot[:, p, z] = chi2_b
            aic_boot[:, p, z] = chi2_b + 2.0 + 2.0 * (15 - tmax)
            succeeded[p, z] = True
            failures = []
            if not np.isfinite(result.q_value) or result.q_value < STATS.q_min:
                failures.append("q_below_threshold")
            if not np.isfinite(chi2dof[p, z]) or chi2dof[p, z] > STATS.correlated_chi2_dof_max:
                failures.append("chi2_dof_above_threshold")
            if success[p, z] < STATS.bootstrap_min_success:
                failures.append("bootstrap_success_below_threshold")
            if failures:
                reason[p, z] = ";".join(failures)
            else:
                eligible[p, z] = True
                reason[p, z] = "accepted"
    return locals()


def fit_tau(input_path: Path, output_path: Path, replace: bool = False) -> None:
    start = time.time()
    inp = load_input(input_path)
    central = inp["central"]
    bootstrap = inp["bootstrap"]
    nboot = bootstrap.shape[0]
    ncandidate = len(CUTS) * 7  # tmax=9..15
    npabs, ndz = 3, 25
    candidate_M = np.full((3, ncandidate, npabs, ndz), np.nan)
    candidate_M_boot = np.full((3, ncandidate, nboot, npabs, ndz), np.nan)
    candidate_chi2 = np.full((3, ncandidate, npabs, ndz), np.nan)
    candidate_dof = np.zeros((3, ncandidate, npabs, ndz), np.int32)
    candidate_q = np.full_like(candidate_chi2, np.nan)
    candidate_chi2_dof = np.full_like(candidate_chi2, np.nan)
    candidate_success = np.zeros_like(candidate_chi2)
    candidate_rank = np.zeros_like(candidate_dof)
    candidate_condition = np.full_like(candidate_chi2, np.nan)
    candidate_shrinkage = np.full_like(candidate_chi2, np.nan)
    candidate_aic = np.full_like(candidate_chi2, np.nan)
    candidate_aic_boot = np.full((3, ncandidate, nboot, npabs, ndz), np.nan)
    candidate_chi2_boot = np.full_like(candidate_aic_boot, np.nan)
    candidate_succeeded = np.zeros_like(candidate_chi2, dtype=bool)
    candidate_eligible = np.zeros_like(candidate_chi2, dtype=bool)
    candidate_reason = np.full((3, ncandidate, npabs, ndz), "not_evaluated", dtype="U128")
    candidate_cuts, candidate_tmax = [], []
    index = 0
    for cut_index, cut in enumerate(CUTS):
        for tmax in range(TMIN + 1, 16):
            candidate_cuts.append(int(cut)); candidate_tmax.append(int(tmax))
            lower = np.arange(TMIN, tmax, dtype=np.int32)
            loc = np.asarray([int(np.flatnonzero(TSEPS == t)[0]) for t in lower])
            for operator in range(3):
                result = candidate_fit(central, bootstrap, cut_index, tmax, operator)
                for name, target in (("values", candidate_M), ("values_boot", candidate_M_boot), ("chi2", candidate_chi2), ("dof", candidate_dof), ("qval", candidate_q), ("chi2dof", candidate_chi2_dof), ("success", candidate_success), ("rank", candidate_rank), ("cond", candidate_condition), ("shrink", candidate_shrinkage), ("aic", candidate_aic), ("aic_boot", candidate_aic_boot), ("chi2_boot", candidate_chi2_boot), ("succeeded", candidate_succeeded), ("eligible", candidate_eligible), ("reason", candidate_reason)):
                    target[operator, index] = result[name]
            index += 1
    candidate_cuts = np.asarray(candidate_cuts, np.int32)
    candidate_tmax = np.asarray(candidate_tmax, np.int32)
    final = []
    fills = []
    for operator in range(3):
        avg = replica_model_average(candidate_M[operator], candidate_M_boot[operator], candidate_aic[operator], candidate_aic_boot[operator], candidate_eligible[operator], candidate_cuts, CUTS, bootstrap_min_success=STATS.bootstrap_min_success)
        fill = fill_with_minimum_chi2(avg, candidate_M[operator], candidate_M_boot[operator], candidate_aic_boot[operator], candidate_chi2[operator], candidate_chi2_boot[operator], candidate_dof[operator], candidate_succeeded[operator], candidate_eligible[operator], candidate_cuts, CUTS, q_min=STATS.q_min, chi2_dof_max=STATS.correlated_chi2_dof_max)
        final.append(avg); fills.append(fill)

    payload = {
        "schema": np.asarray(SCHEMA),
        "status": np.asarray("finite_flow_bare_matrix_element_fit_latest_bootstrap"),
        "M": np.stack([item.M for item in final], axis=0),
        "M_boot": np.stack([item.M_boot for item in final], axis=1),
        "statistical_error": np.stack([item.statistical_error for item in final], axis=0),
        "fit_accepted": np.stack([item.fit_valid for item in final], axis=0),
        "M_filled": np.stack([item.M_filled for item in fills], axis=0),
        "M_boot_filled": np.stack([item.M_boot_filled for item in fills], axis=1),
        "filled_statistical_error": np.stack([item.filled_statistical_error for item in fills], axis=0),
        "estimate_available": np.stack([item.estimate_available for item in fills], axis=0),
        "central_fallback_used": np.stack([item.central_fallback_used for item in fills], axis=0),
        "central_fallback_status": np.stack([item.central_fallback_status for item in fills], axis=0),
        "replica_fallback_fraction": np.stack([item.replica_fallback_fraction for item in fills], axis=1),
        "candidate_M": candidate_M, "candidate_M_boot": candidate_M_boot,
        "candidate_chi2": candidate_chi2, "candidate_dof": candidate_dof,
        "candidate_q_value": candidate_q, "candidate_chi2_dof": candidate_chi2_dof,
        "candidate_bootstrap_success_fraction": candidate_success,
        "candidate_covariance_rank": candidate_rank, "candidate_covariance_condition": candidate_condition,
        "candidate_covariance_shrinkage": candidate_shrinkage, "candidate_range_aic": candidate_aic,
        "candidate_range_aic_boot": candidate_aic_boot, "candidate_chi2_boot": candidate_chi2_boot,
        "candidate_fit_succeeded": candidate_succeeded, "candidate_eligible": candidate_eligible,
        "candidate_failure_reason": candidate_reason,
        "candidate_cut": candidate_cuts, "candidate_tmax_summed": candidate_tmax,
        "operator_labels": np.asarray(OPERATORS), "pabs_values": PABS, "z_values": Z,
        "flow_tau_t_over_a2": np.asarray(float(input_path.stem.split("tau")[1].split("_N")[0].replace("p", "."))),
        "fit_tsep_values": TSEPS, "difference_tsep_values": np.arange(TMIN, 15, dtype=np.int32),
        "registered_cuts": CUTS, "confs": inp["confs"], "bootstrap_indices": inp["indices"],
        "nconf": np.asarray(406, np.int32), "nboot": np.asarray(1500, np.int32), "seed": np.asarray(1115, np.int64),
        "a_fm": np.asarray(A_FM), "minimum_tsep_fm_strict": np.asarray(0.6), "fixed_tmin": np.asarray(TMIN, np.int32),
        "fit_method": np.asarray("forward_sumratio_difference_correlated_plateau"),
        "fit_model": np.asarray("D_c(T)=S_c(T+1)-S_c(T)=M"),
        "error_definition": np.asarray("std_ddof1_of_1500_replica_AIC_equal_cut_mixture; no_sqrt_Nboot"),
        "input_product": np.asarray(str(input_path.resolve())), "input_product_sha256": np.asarray(inp["input_sha256"]),
        "axes_M": np.asarray(("operator", "momentum_abs", "z")),
        "axes_M_boot": np.asarray(("bootstrap", "operator", "momentum_abs", "z")),
    }
    contract = {
        "schema": SCHEMA, "input_schema": INPUT_SCHEMA, "flow_tau_t_over_a2": payload["flow_tau_t_over_a2"].item(),
        "nconf": 406, "nboot": 1500, "seed": 1115, "tmin": TMIN, "tmin_fm": TMIN * A_FM,
        "strict_threshold": "Tmin*a > 0.6 fm; Tmin=floor(0.6/a)+1",
        "cuts": CUTS.tolist(), "candidate_tmax_summed": candidate_tmax.tolist(),
        "operators": list(OPERATORS), "aic": "chi2+2*k+2*N_omitted, k=1; weights recomputed per replica",
        "between_cut": "equal weight over usable cuts", "quality_gates": {"Q_min": 0.01, "chi2_dof_max": 3.0, "bootstrap_fraction_min": 0.90},
        "input_product": str(input_path.resolve()), "input_product_sha256": inp["input_sha256"],
        "status_note": "finite-flow bare matrix elements; no renormalization, matching, continuum or zero-flow conversion",
    }
    payload["contract_json"] = np.asarray(json.dumps(contract, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not replace:
        raise FileExistsError(output_path)
    tmp = output_path.with_name(output_path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **payload); os.replace(tmp, output_path)
    receipt = {**contract, "status": "complete", "output": str(output_path.resolve()), "output_sha256": sha256_file(output_path), "elapsed_seconds": time.time() - start, "fit_accepted_count": int(np.count_nonzero(payload["fit_accepted"])), "fit_total_count": int(payload["fit_accepted"].size), "filled_count": int(np.count_nonzero(payload["estimate_available"]))}
    receipt_path = Path(str(output_path) + ".done.json")
    tmpj = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}"); tmpj.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n"); os.replace(tmpj, receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ratio_bootstrap_latest_v1/results_v1")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_v1")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    taus = TAUS if args.tau is None else (args.tau,)
    for tau in taus:
        tag = tau_tag(tau)
        fit_tau(args.input_dir / f"c3_c2_bootstrap_{tag}_N406.npz", args.output_dir / f"bare_matrix_latest_{tag}_N406.npz", args.replace)


if __name__ == "__main__":
    main()
