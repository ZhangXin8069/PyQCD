#!/usr/bin/env python3
"""Correlated tau->0 sensitivity scan using only flow times tau/a^2 > 0.1.

This analysis consumes the Nconf=406, Nboot=1500 bare-matrix products in
``fit_latest_v1/results_v1``.  It keeps quality-controlled accepted inputs and
filled/fallback diagnostic inputs in separate estimator branches.  The latter
must never be quoted as an accepted flow-time extrapolation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the already tested GLS covariance and line-fit implementation.  The
# input loader below is new because the latest products have Nconf=406 and a
# new schema/path contract.
from flow_time_analysis_v1.analyze import fit_flow_line


SCHEMA = "gradient_flow_gluon_flow_time_tau_gt_0p1_v1"
FIT_SCHEMA = "gradient_flow_gluon_bare_sumdiff_aic_latest_v1"
TAUS = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8,
                   2.2, 2.6, 3.0, 3.4, 3.8), dtype=float)
WINDOWS = {
    "tau0p2_0p4": (0.2, 0.3, 0.4),
    "tau0p2_0p6": (0.2, 0.3, 0.4, 0.5, 0.6),
    "tau0p2_1p0": (0.2, 0.3, 0.4, 0.5, 0.6, 1.0),
    "tau0p2_1p4": (0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4),
    "tau0p2_1p8": (0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8),
    "tau0p2_2p6": (0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6),
    "tau0p2_3p8": tuple(TAUS[TAUS > 0.1]),
}
MODES = ("geometry_guarded", "unrestricted_large_flow_diagnostic")
ESTIMATORS = ("accepted_only", "filled_fallback_diagnostic")
OPERATORS = (
    "unpolarized_TH_minus_SH_even_real",
    "helicity_TH_odd_imag",
    "helicity_TH_plus_SH_odd_imag",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"Expected scalar {key}, got {value.shape}")
    return value.item()


def load_latest(fit_root: Path) -> dict[str, object]:
    fields = ("M", "M_boot", "statistical_error", "fit_accepted",
              "M_filled", "M_boot_filled", "filled_statistical_error",
              "estimate_available", "central_fallback_used",
              "replica_fallback_fraction")
    stacks: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    paths: list[str] = []
    hashes: list[str] = []
    input_paths: list[str] = []
    input_hashes: list[str] = []
    ref_confs = ref_boot = ref_p = ref_z = None

    for tau in TAUS:
        path = fit_root / f"bare_matrix_latest_{tau_tag(float(tau))}_N406.npz"
        receipt_path = Path(str(path) + ".done.json")
        if not path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Missing product/receipt pair for tau={tau}: {path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or receipt.get("schema") != FIT_SCHEMA:
            raise ValueError(f"Incomplete or wrong receipt: {receipt_path}")
        digest = sha256_file(path)
        if digest != receipt.get("output_sha256"):
            raise ValueError(f"SHA-256 mismatch: {path}")

        with np.load(path, allow_pickle=False) as data:
            if str(_scalar(data, "schema")) != FIT_SCHEMA:
                raise ValueError(f"Wrong fit schema: {path}")
            if not np.isclose(float(_scalar(data, "flow_tau_t_over_a2")), tau,
                              atol=1e-12, rtol=0.0):
                raise ValueError(f"Flow-time mismatch: {path}")
            if (int(_scalar(data, "nconf")), int(_scalar(data, "nboot")),
                    int(_scalar(data, "seed"))) != (406, 1500, 1115):
                raise ValueError(f"Unexpected resampling contract: {path}")
            labels = tuple(data["operator_labels"].astype(str).tolist())
            pvals = np.asarray(data["pabs_values"], dtype=np.int32)
            zvals = np.asarray(data["z_values"], dtype=np.int32)
            confs = data["confs"].astype(str)
            boot = np.asarray(data["bootstrap_indices"], dtype=np.int32)
            if labels != OPERATORS or tuple(pvals) != (3, 4, 5) or tuple(zvals) != tuple(range(25)):
                raise ValueError(f"Operator/momentum/z contract mismatch: {path}")
            if confs.shape != (406,) or boot.shape != (1500, 406):
                raise ValueError(f"Configuration/bootstrap shape mismatch: {path}")
            if ref_confs is None:
                ref_confs, ref_boot, ref_p, ref_z = confs, boot, pvals, zvals
            elif not np.array_equal(confs, ref_confs) or not np.array_equal(boot, ref_boot):
                raise ValueError(f"Shared configuration/bootstrap indices differ: {path}")
            for key in fields:
                if key not in data.files:
                    raise ValueError(f"Missing {key}: {path}")
                stacks[key].append(np.asarray(data[key]))
            accepted = np.asarray(data["fit_accepted"], dtype=bool)
            estimate = np.asarray(data["estimate_available"], dtype=bool)
            M = np.asarray(data["M"], dtype=float)
            Mb = np.asarray(data["M_boot"], dtype=float)
            Mf = np.asarray(data["M_filled"], dtype=float)
            Mfb = np.asarray(data["M_boot_filled"], dtype=float)
            if M.shape != (3, 3, 25) or Mb.shape != (1500, 3, 3, 25):
                raise ValueError(f"Unexpected matrix-element shape: {path}")
            if not np.isfinite(M[accepted]).all() or np.any(np.isfinite(M[~accepted])):
                raise ValueError(f"Accepted-only central contract failed: {path}")
            if not np.isfinite(Mb[:, accepted]).all() or np.any(np.isfinite(Mb[:, ~accepted])):
                raise ValueError(f"Accepted-only replica contract failed: {path}")
            if not np.isfinite(Mf[estimate]).all() or not np.isfinite(Mfb[:, estimate]).all():
                raise ValueError(f"Filled diagnostic contract failed: {path}")
            input_paths.append(str(_scalar(data, "input_product")))
            input_hashes.append(str(_scalar(data, "input_product_sha256")))
        paths.append(str(path.resolve()))
        hashes.append(digest)

    return {
        **{key: np.stack(value, axis=0) for key, value in stacks.items()},
        "confs": ref_confs, "bootstrap_indices": ref_boot,
        "pabs_values": ref_p, "z_values": ref_z,
        "source_products": np.asarray(paths), "source_sha256": np.asarray(hashes),
        "input_products": np.asarray(input_paths), "input_sha256": np.asarray(input_hashes),
    }


def run_scan(products: dict[str, object]) -> dict[str, np.ndarray]:
    nw, nm, ne, nt, no, npmom, nz = (len(WINDOWS), len(MODES), len(ESTIMATORS),
                                      len(TAUS), len(OPERATORS), 3, 25)
    shape = (nm, ne, nw, no, npmom, nz)
    boot_shape = (nm, ne, nw, 1500, no, npmom, nz)
    result: dict[str, np.ndarray] = {
        "intercept": np.full(shape, np.nan),
        "intercept_error": np.full(shape, np.nan),
        "slope_tau": np.full(shape, np.nan),
        "chi2": np.full(shape, np.nan),
        "dof": np.zeros(shape, dtype=np.int32),
        "q_value": np.full(shape, np.nan),
        "chi2_dof": np.full(shape, np.nan),
        "fit_computed": np.zeros(shape, dtype=bool),
        "quality_pass": np.zeros(shape, dtype=bool),
        "input_all_available": np.zeros(shape, dtype=bool),
        "n_points": np.zeros(shape, dtype=np.int32),
        "n_geometry_excluded": np.zeros(shape, dtype=np.int32),
        "covariance_sample_rank": np.zeros(shape, dtype=np.int32),
        "covariance_effective_rank": np.zeros(shape, dtype=np.int32),
        "covariance_condition": np.full(shape, np.nan),
        "covariance_shrinkage": np.full(shape, np.nan),
        "finite_replica_fraction": np.full(shape, np.nan),
        "status": np.full(shape, "not_evaluated", dtype="U128"),
        "intercept_boot": np.full(boot_shape, np.nan),
        "slope_tau_boot": np.full(boot_shape, np.nan),
        "fit_tau_mask": np.zeros((nm, nw, nz, nt), dtype=bool),
    }
    M_sets = (np.asarray(products["M"], float), np.asarray(products["M_filled"], float))
    Mb_sets = (np.asarray(products["M_boot"], float), np.asarray(products["M_boot_filled"], float))
    err_sets = (np.asarray(products["statistical_error"], float),
                np.asarray(products["filled_statistical_error"], float))
    avail_sets = (np.asarray(products["fit_accepted"], bool),
                  np.asarray(products["estimate_available"], bool))
    zvals = np.asarray(products["z_values"], int)

    for im, mode in enumerate(MODES):
        for iw, (_, registered) in enumerate(WINDOWS.items()):
            base = np.isin(TAUS, np.asarray(registered, float))
            for iz, z in enumerate(zvals):
                mask = base.copy()
                if mode == "geometry_guarded":
                    mask &= (z > 0) & (np.sqrt(8.0 * TAUS) < float(z))
                result["fit_tau_mask"][im, iw, iz] = mask
                indices = np.flatnonzero(mask)
                for ie in range(ne):
                    for io in range(no):
                        for ip in range(npmom):
                            loc = (im, ie, iw, io, ip, iz)
                            result["n_points"][loc] = len(indices)
                            result["n_geometry_excluded"][loc] = int(np.count_nonzero(base) - len(indices))
                            if len(indices) < 3:
                                result["status"][loc] = "fewer_than_three_flow_points"
                                continue
                            available = avail_sets[ie][indices, io, ip, iz]
                            result["input_all_available"][loc] = bool(np.all(available))
                            if not np.all(available):
                                result["status"][loc] = "one_or_more_input_estimates_unavailable"
                                continue
                            y = M_sets[ie][indices, io, ip, iz]
                            samples = np.transpose(Mb_sets[ie][indices, :, io, ip, iz], (1, 0))
                            fit = fit_flow_line(y, samples, TAUS[indices])
                            result["intercept"][loc] = fit.intercept
                            result["slope_tau"][loc] = fit.slope_tau
                            result["chi2"][loc] = fit.chi2
                            result["dof"][loc] = fit.dof
                            result["q_value"][loc] = fit.q_value
                            result["chi2_dof"][loc] = fit.chi2_dof
                            result["fit_computed"][loc] = fit.fit_computed
                            result["quality_pass"][loc] = fit.quality_pass
                            result["status"][loc] = fit.status
                            result["covariance_sample_rank"][loc] = fit.covariance.sample_rank
                            result["covariance_effective_rank"][loc] = fit.covariance.effective_rank
                            result["covariance_condition"][loc] = fit.covariance.condition
                            result["covariance_shrinkage"][loc] = fit.covariance.shrinkage
                            result["finite_replica_fraction"][loc] = fit.covariance.finite_fraction
                            result["intercept_boot"][im, ie, iw, :, io, ip, iz] = fit.intercept_boot
                            result["slope_tau_boot"][im, ie, iw, :, io, ip, iz] = fit.slope_tau_boot
                            finite = np.isfinite(fit.intercept_boot)
                            if np.count_nonzero(finite) >= 2:
                                result["intercept_error"][loc] = np.std(fit.intercept_boot[finite], ddof=1)
    return result


def write_outputs(products: dict[str, object], fit: dict[str, np.ndarray], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    max_window = max(map(len, WINDOWS.values()))
    window_taus = np.full((len(WINDOWS), max_window), np.nan)
    window_lengths = np.zeros(len(WINDOWS), np.int32)
    for iw, vals in enumerate(WINDOWS.values()):
        window_taus[iw, :len(vals)] = vals
        window_lengths[iw] = len(vals)
    payload = {
        "schema": np.asarray(SCHEMA),
        "analysis_status": np.asarray("tau_gt_0p1_sensitivity_finite_flow_bare_not_physical_pdf"),
        "flow_model": np.asarray("M(tau)=M0+c_tau*tau"),
        "flow_taus": TAUS,
        "flow_radius_over_a": np.sqrt(8.0 * TAUS),
        "window_names": np.asarray(tuple(WINDOWS)),
        "window_taus": window_taus,
        "window_lengths": window_lengths,
        "mode_names": np.asarray(MODES),
        "estimator_names": np.asarray(ESTIMATORS),
        "operator_labels": np.asarray(OPERATORS),
        "pabs_values": products["pabs_values"],
        "z_values": products["z_values"],
        "nconf": np.asarray(406), "nboot": np.asarray(1500), "seed": np.asarray(1115),
        "a_fm": np.asarray(0.0775), "svdcut": np.asarray(1e-12),
        "geometry_rule": np.asarray("sqrt(8*tau)<z; strict inequality"),
        "quality_rule": np.asarray("Q>=0.01 and chi2/dof<=3 and finite_fraction>=0.90"),
        "bootstrap_indices": products["bootstrap_indices"], "confs": products["confs"],
        "source_products": products["source_products"], "source_sha256": products["source_sha256"],
        "input_products": products["input_products"], "input_sha256": products["input_sha256"],
        "analysis_code_sha256": np.asarray(sha256_file(Path(__file__))),
        "gls_dependency": np.asarray(str((ROOT / "flow_time_analysis_v1/analyze.py").resolve())),
        "gls_dependency_sha256": np.asarray(sha256_file(ROOT / "flow_time_analysis_v1/analyze.py")),
        **fit,
    }
    npz = outdir / "flow_time_tau_gt_0p1_v1.npz"
    tmp = outdir / f".{npz.name}.{os.getpid()}.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, npz)

    csv_path = outdir / "flow_time_tau_gt_0p1_points.csv"
    fields = ("mode", "estimator", "window", "operator", "Pz", "z", "fit_taus",
              "intercept", "intercept_error", "slope_tau", "chi2", "dof", "Q",
              "chi2_dof", "fit_computed", "quality_pass", "n_points",
              "n_geometry_excluded", "covariance_sample_rank",
              "covariance_effective_rank", "covariance_condition",
              "covariance_shrinkage", "status")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for im, mode in enumerate(MODES):
            for ie, estimator in enumerate(ESTIMATORS):
                for iw, window in enumerate(WINDOWS):
                    for io, operator in enumerate(OPERATORS):
                        for ip, pz in enumerate(products["pabs_values"]):
                            for iz, z in enumerate(products["z_values"]):
                                loc = (im, ie, iw, io, ip, iz)
                                mask = fit["fit_tau_mask"][im, iw, iz]
                                row = {
                                    "mode": mode, "estimator": estimator, "window": window,
                                    "operator": operator, "Pz": int(pz), "z": int(z),
                                    "fit_taus": " ".join(f"{x:g}" for x in TAUS[mask]),
                                    "intercept": fit["intercept"][loc],
                                    "intercept_error": fit["intercept_error"][loc],
                                    "slope_tau": fit["slope_tau"][loc], "chi2": fit["chi2"][loc],
                                    "dof": int(fit["dof"][loc]), "Q": fit["q_value"][loc],
                                    "chi2_dof": fit["chi2_dof"][loc],
                                    "fit_computed": bool(fit["fit_computed"][loc]),
                                    "quality_pass": bool(fit["quality_pass"][loc]),
                                    "n_points": int(fit["n_points"][loc]),
                                    "n_geometry_excluded": int(fit["n_geometry_excluded"][loc]),
                                    "covariance_sample_rank": int(fit["covariance_sample_rank"][loc]),
                                    "covariance_effective_rank": int(fit["covariance_effective_rank"][loc]),
                                    "covariance_condition": fit["covariance_condition"][loc],
                                    "covariance_shrinkage": fit["covariance_shrinkage"][loc],
                                    "status": str(fit["status"][loc]),
                                }
                                writer.writerow(row)

    summary = {
        "schema": SCHEMA, "status": "complete",
        "analysis_status": "tau_gt_0p1_sensitivity_finite_flow_bare_not_physical_pdf",
        "output": str(npz.resolve()), "output_sha256": sha256_file(npz),
        "csv": str(csv_path.resolve()), "csv_sha256": sha256_file(csv_path),
        "nconf": 406, "nboot": 1500, "seed": 1115,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "modes": list(MODES), "estimators": list(ESTIMATORS),
        "counts": {},
    }
    for im, mode in enumerate(MODES):
        for ie, estimator in enumerate(ESTIMATORS):
            key = f"{mode}/{estimator}"
            summary["counts"][key] = {
                "fit_computed": int(np.count_nonzero(fit["fit_computed"][im, ie])),
                "quality_pass": int(np.count_nonzero(fit["quality_pass"][im, ie])),
            }
    (outdir / "flow_time_tau_gt_0p1_v1.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-root", type=Path, default=ROOT / "fit_latest_v1/results_v1")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results_v1")
    args = parser.parse_args()
    products = load_latest(args.fit_root.resolve())
    fit = run_scan(products)
    write_outputs(products, fit, args.output_dir.resolve())
    print(args.output_dir.resolve() / "flow_time_tau_gt_0p1_v1.npz")


if __name__ == "__main__":
    main()
