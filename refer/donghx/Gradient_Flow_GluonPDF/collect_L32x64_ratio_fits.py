#!/usr/bin/env python3
"""Collect per-momentum L32x64 direct-Ratio fits into one analysis product."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FORMULAS = ("titi", "titi_minus_ijij", "titi_plus_ijij")
DIRECTIONS = ("x", "y", "z")
PABS = (3, 4, 5, 6)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scalar(data, key):
    return np.asarray(data[key]).item()


def collect(channel: str, fit_root: Path, output: Path) -> None:
    paths = [
        fit_root / f"barefit_L32x64_{channel}_P{p}_direct_ratio.npz"
        for p in PABS
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    loaded = []
    required_fit_keys = {
        "M_filled",
        "M_boot_filled",
        "filled_statistical_error",
        "filled_window_systematic",
        "filled_total_error",
        "estimate_available",
        "central_fallback_used",
        "central_fallback_candidate_index",
        "central_fallback_status",
        "replica_fallback_used",
        "replica_fallback_fraction",
        "replica_fallback_candidate_index",
    }
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if scalar(data, "channel") != channel:
                raise ValueError(f"channel mismatch in {path}")
            if [str(x) for x in data["formula_labels"].tolist()] != list(FORMULAS):
                raise ValueError(f"formula order mismatch in {path}")
            if [str(x) for x in data["direction_labels"].tolist()] != list(DIRECTIONS):
                raise ValueError(f"direction order mismatch in {path}")
            if int(scalar(data, "nconf")) != 200 or int(scalar(data, "nboot")) != 500:
                raise ValueError(f"resampling metadata mismatch in {path}")
            if int(scalar(data, "seed")) != 1115:
                raise ValueError(f"seed mismatch in {path}")
            missing = sorted(required_fit_keys - set(data.files))
            if missing:
                raise ValueError(
                    f"fit product {path} predates the filled-estimate fields: {missing}"
                )
            if not np.all(np.asarray(data["estimate_available"], dtype=bool)):
                raise ValueError(f"no filled estimate at one or more z points in {path}")
            loaded.append({key: np.asarray(data[key]) for key in data.files})

    reference = loaded[0]
    for p, data in zip(PABS[1:], loaded[1:]):
        for key in ("bootstrap_indices", "confs", "candidate_labels", "z_values"):
            if not np.array_equal(data[key], reference[key]):
                raise ValueError(f"shared metadata {key} mismatch at P={p}")
        if str(data["fit_method"].item()) != str(reference["fit_method"].item()):
            raise ValueError(f"fit method mismatch at P={p}")

    # Per-P source arrays have axes formula,direction,(pabs=1),z; move the
    # singleton axis to the new momentum axis and retain explicit labels.
    payload = {
        "M": np.concatenate([data["M"] for data in loaded], axis=2),
        "bare_matrix_element": np.concatenate(
            [data["bare_matrix_element"] for data in loaded], axis=2
        ),
        "M_raw_diagnostic": np.concatenate(
            [data["M_raw_diagnostic"] for data in loaded], axis=2
        ),
        "statistical_error": np.concatenate(
            [data["statistical_error"] for data in loaded], axis=2
        ),
        "statistical_error_raw_diagnostic": np.concatenate(
            [data["statistical_error_raw_diagnostic"] for data in loaded], axis=2
        ),
        "window_systematic": np.concatenate(
            [data["window_systematic"] for data in loaded], axis=2
        ),
        "window_systematic_raw_diagnostic": np.concatenate(
            [data["window_systematic_raw_diagnostic"] for data in loaded], axis=2
        ),
        "total_error": np.concatenate(
            [data["total_error"] for data in loaded], axis=2
        ),
        "total_error_raw_diagnostic": np.concatenate(
            [data["total_error_raw_diagnostic"] for data in loaded], axis=2
        ),
        "fit_valid": np.concatenate([data["fit_valid"] for data in loaded], axis=2),
        "primary_fit_accepted": np.concatenate(
            [data["primary_fit_accepted"] for data in loaded], axis=2
        ),
        "accepted_window_count": np.concatenate(
            [data["accepted_window_count"] for data in loaded], axis=2
        ),
        # Filled values contain either an accepted primary fit or an explicit
        # diagnostic fallback.  Keep them beside the quality-controlled M
        # arrays so downstream plots can show coverage without relabelling a
        # failed quality gate as accepted.
        "M_filled": np.concatenate([data["M_filled"] for data in loaded], axis=2),
        "M_boot_filled": np.concatenate(
            [data["M_boot_filled"] for data in loaded], axis=3
        ),
        "filled_statistical_error": np.concatenate(
            [data["filled_statistical_error"] for data in loaded], axis=2
        ),
        "filled_window_systematic": np.concatenate(
            [data["filled_window_systematic"] for data in loaded], axis=2
        ),
        "filled_total_error": np.concatenate(
            [data["filled_total_error"] for data in loaded], axis=2
        ),
        "estimate_available": np.concatenate(
            [data["estimate_available"] for data in loaded], axis=2
        ),
        "central_fallback_used": np.concatenate(
            [data["central_fallback_used"] for data in loaded], axis=2
        ),
        "central_fallback_candidate_index": np.concatenate(
            [data["central_fallback_candidate_index"] for data in loaded], axis=2
        ),
        "central_fallback_status": np.concatenate(
            [data["central_fallback_status"] for data in loaded], axis=2
        ),
        "replica_fallback_used": np.concatenate(
            [data["replica_fallback_used"] for data in loaded], axis=3
        ),
        "replica_fallback_fraction": np.concatenate(
            [data["replica_fallback_fraction"] for data in loaded], axis=2
        ),
        "replica_fallback_candidate_index": np.concatenate(
            [data["replica_fallback_candidate_index"] for data in loaded], axis=3
        ),
        # Per-P files retain a singleton pabs axis.  Concatenate along that
        # axis (rather than stack) so the collection has one explicit
        # momentum axis and no accidental extra length-one dimension.
        "candidate_M_raw": np.concatenate(
            [data["candidate_M_raw"] for data in loaded], axis=3
        ),
        "candidate_statistical_error_raw": np.concatenate(
            [data["candidate_statistical_error_raw"] for data in loaded], axis=3
        ),
        "candidate_fit_accepted": np.concatenate(
            [data["candidate_fit_accepted"] for data in loaded], axis=3
        ),
        "candidate_failure_reason": np.concatenate(
            [data["candidate_failure_reason"] for data in loaded], axis=3
        ),
        "candidate_chi2": np.concatenate(
            [data["candidate_chi2"] for data in loaded], axis=3
        ),
        "candidate_dof": np.concatenate(
            [data["candidate_dof"] for data in loaded], axis=3
        ),
        "candidate_Q": np.concatenate(
            [data["candidate_Q"] for data in loaded], axis=3
        ),
        "candidate_correlated_chi2_dof": np.concatenate(
            [data["candidate_correlated_chi2_dof"] for data in loaded], axis=3
        ),
        "candidate_bootstrap_success_fraction": np.concatenate(
            [data["candidate_bootstrap_success_fraction"] for data in loaded], axis=3
        ),
        "candidate_covariance_rank": np.concatenate(
            [data["candidate_covariance_rank"] for data in loaded], axis=3
        ),
        "candidate_covariance_condition": np.concatenate(
            [data["candidate_covariance_condition"] for data in loaded], axis=3
        ),
        "candidate_covariance_shrinkage_lambda": np.concatenate(
            [data["candidate_covariance_shrinkage_lambda"] for data in loaded], axis=3
        ),
        "candidate_finite_covariance_replicas": np.concatenate(
            [data["candidate_finite_covariance_replicas"] for data in loaded], axis=3
        ),
        "candidate_M_boot_raw": np.concatenate(
            [data["candidate_M_boot_raw"] for data in loaded], axis=4
        ),
        "primary_M_boot": np.concatenate(
            [data["primary_M_boot"] for data in loaded], axis=3
        ),
        "primary_M_boot_raw_diagnostic": np.concatenate(
            [data["primary_M_boot_raw_diagnostic"] for data in loaded], axis=3
        ),
        "candidate_fit_covariance": np.concatenate(
            [data["candidate_fit_covariance"] for data in loaded], axis=3
        ),
        "candidate_covariance_eigenvalues": np.concatenate(
            [data["candidate_covariance_eigenvalues"] for data in loaded], axis=3
        ),
        "candidate_retained_eigenmodes": np.concatenate(
            [data["candidate_retained_eigenmodes"] for data in loaded], axis=3
        ),
        "candidate_coordinates_T_tau": reference["candidate_coordinates_T_tau"],
        "candidate_point_mask": reference["candidate_point_mask"],
        "candidate_npoints": reference["candidate_npoints"],
        "primary_candidate_index": reference["primary_candidate_index"],
        "primary_candidate": reference["primary_candidate"],
        "formula_labels": reference["formula_labels"],
        "component_labels": reference["component_labels"],
        "direction_labels": reference["direction_labels"],
        "pabs_values": np.asarray(PABS, dtype=np.int32),
        "delta_z_values": reference["delta_z_values"],
        "z_values": reference["z_values"],
        "tsep_values": reference["tsep_values"],
        "tau_values": reference["tau_values"],
        "valid_tau_mask": reference["valid_tau_mask"],
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
        "fit_model": reference["fit_model"],
        "fit_method": reference["fit_method"],
        "central_estimator": reference["central_estimator"],
        "bootstrap_role": reference["bootstrap_role"],
        "covariance_method": reference["covariance_method"],
        "svdcut": reference["svdcut"],
        "q_min": reference["q_min"],
        "correlated_chi2_dof_max": reference["correlated_chi2_dof_max"],
        "bootstrap_min_success": reference["bootstrap_min_success"],
        "minimum_accepted_windows": reference["minimum_accepted_windows"],
        "window_systematic_method": reference["window_systematic_method"],
        "fit_window_status": reference["fit_window_status"],
        "candidate_labels": reference["candidate_labels"],
        "candidate_fit_start": reference["candidate_fit_start"],
        "candidate_fit_end": reference["candidate_fit_end"],
        "candidate_cut_left": reference["candidate_cut_left"],
        "candidate_cut_right": reference["candidate_cut_right"],
        "ensemble": reference["ensemble"],
        "nx": reference["nx"],
        "nt": reference["nt"],
        "a_fm": reference["a_fm"],
        "channel": reference["channel"],
        "physical_orientation": reference["physical_orientation"],
        "operator_definition": reference["operator_definition"],
        "input_scheme": reference["input_scheme"],
        "input_fit_paths": np.asarray([str(p) for p in paths]),
        "input_fit_sha256": np.asarray([sha256(p) for p in paths]),
        "bootstrap_indices": reference["bootstrap_indices"],
        "confs": reference["confs"],
        "nconf": reference["nconf"],
        "nboot": reference["nboot"],
        "seed": reference["seed"],
        "bootstrap_method": reference["bootstrap_method"],
        "analysis_status": np.asarray(
            "finite_HYP_flowed_gluon_bare_direct_ratio_fit_collection_not_physical_pdf"
        ),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    receipt = {
        "ensemble": "L32x64",
        "channel": channel,
        "pabs": list(PABS),
        "directions": list(DIRECTIONS),
        "formulas": list(FORMULAS),
        "input_fit_paths": [str(p) for p in paths],
        "input_fit_sha256": [sha256(p) for p in paths],
        "output": str(output),
        "output_sha256": sha256(output),
        "fit_valid_count": int(np.count_nonzero(payload["fit_valid"])),
        "estimate_available_count": int(
            np.count_nonzero(payload["estimate_available"])
        ),
        "fallback_count": int(np.count_nonzero(payload["central_fallback_used"])),
        "total_point_count": int(payload["fit_valid"].size),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "finite_HYP_bare_ratio_fit_collection_not_physical_pdf",
    }
    receipt_path = output.with_suffix(output.suffix + ".done.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    # A compact JSON summary is convenient for checking all 3x3 channels.
    summary = []
    for fi, formula in enumerate(FORMULAS):
        for di, direction in enumerate(DIRECTIONS):
            for pi, p in enumerate(PABS):
                valid = payload["fit_valid"][fi, di, pi]
                available = payload["estimate_available"][fi, di, pi]
                for zi, z in enumerate(payload["z_values"]):
                    if available[zi]:
                        accepted = bool(valid[zi])
                        value = payload["M"][fi, di, pi, zi] if accepted else payload["M_filled"][fi, di, pi, zi]
                        stat_key = "statistical_error" if accepted else "filled_statistical_error"
                        syst_key = "window_systematic" if accepted else "filled_window_systematic"
                        total_key = "total_error" if accepted else "filled_total_error"
                        summary.append(
                            {
                                "formula": formula,
                                "direction": direction,
                                "pabs": p,
                                "z": int(z),
                                "estimate_status": str(payload["central_fallback_status"][fi, di, pi, zi]),
                                "fit_valid": accepted,
                                "M": float(value),
                                "statistical_error": float(payload[stat_key][fi, di, pi, zi]),
                                "window_systematic": float(payload[syst_key][fi, di, pi, zi]),
                                "total_error": float(payload[total_key][fi, di, pi, zi]),
                            }
                        )
    output.with_name(output.stem + "_summary.json").write_text(
        json.dumps({**receipt, "rows": summary}, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=("unpolarized", "helicity"), required=True)
    ap.add_argument("--fit-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    collect(args.channel, args.fit_root, args.output)


if __name__ == "__main__":
    main()
