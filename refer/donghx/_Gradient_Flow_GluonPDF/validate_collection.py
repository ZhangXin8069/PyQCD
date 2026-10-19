#!/usr/bin/env python3
"""Validate the all-operator AIC fit collection and write a receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v1"
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)
OPERATORS = (
    "unpolarized_TU_even",
    "unpolarized_TU_plus_SU_even",
    "unpolarized_TU_minus_SU_even",
    "helicity_TH_odd",
    "helicity_TH_plus_SH_odd",
    "helicity_TH_minus_SH_odd",
)


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def fit_path(tau: float) -> Path:
    return RESULTS / f"bare_matrix_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def receipt_for(path: Path) -> Path:
    candidates = (Path(str(path) + ".done.json"), path.with_suffix(".done.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing receipt: {path}")


def main() -> None:
    records = []
    reference_confs = None
    reference_indices = None
    total_accepted = 0
    total_points = 0
    total_filled = 0
    total_fallback = 0
    reference_manifest = None
    for tau in TAUS:
        path = fit_path(tau)
        receipt_path = receipt_for(path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError(f"Incomplete receipt: {receipt_path}")
        output_sha = sha256_file(path)
        if receipt.get("output_sha256") != output_sha:
            raise ValueError(f"Output hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            labels = tuple(np.asarray(data["operator_labels"]).astype(str).tolist())
            if labels != OPERATORS:
                raise ValueError(f"Operator axis mismatch: {path}")
            expected = {
                "M": (6, 3, 25),
                "M_boot": (1000, 6, 3, 25),
                "M_filled": (6, 3, 25),
                "M_boot_filled": (1000, 6, 3, 25),
                "fit_accepted": (6, 3, 25),
                "estimate_available": (6, 3, 25),
                "candidate_M": (6, 21, 3, 25),
                "candidate_M_boot": (6, 21, 1000, 3, 25),
                "candidate_range_aic": (6, 21, 3, 25),
                "candidate_range_aic_boot": (6, 21, 1000, 3, 25),
                "candidate_chi2_boot": (6, 21, 1000, 3, 25),
            }
            for key, shape in expected.items():
                if tuple(data[key].shape) != shape:
                    raise ValueError(f"{key} shape {data[key].shape} != {shape}: {path}")
            if int(data["nconf"].item()) != 406 or int(data["nboot"].item()) != 1000:
                raise ValueError(f"Sample-count mismatch: {path}")
            if int(data["seed"].item()) != 1115:
                raise ValueError(f"Seed mismatch: {path}")
            if int(data["fixed_tmin"].item()) != 8:
                raise ValueError(f"Tmin mismatch: {path}")
            if not np.array_equal(data["candidate_tmin"], np.full(21, 8, dtype=np.int32)):
                raise ValueError(f"Candidate Tmin is not fixed: {path}")
            if not np.array_equal(data["candidate_tmax_summed"], np.tile(np.arange(9, 16), 3)):
                raise ValueError(f"Candidate Tmax grid mismatch: {path}")
            if not np.array_equal(data["difference_tsep_values"], np.arange(8, 15)):
                raise ValueError(f"Difference time axis mismatch: {path}")
            if not np.array_equal(data["z_values"], np.arange(25)):
                raise ValueError(f"z axis mismatch: {path}")
            if not np.array_equal(data["pabs_values"], np.asarray((3, 4, 5), dtype=np.int32)):
                raise ValueError(f"momentum axis mismatch: {path}")
            confs = np.asarray(data["confs"]).astype(str)
            indices = np.asarray(data["bootstrap_indices"], dtype=np.int64)
            if reference_confs is None:
                reference_confs, reference_indices = confs, indices
            elif not np.array_equal(reference_confs, confs) or not np.array_equal(
                reference_indices, indices
            ):
                raise ValueError(f"Flow-time axes are not shared: {path}")
            manifest = str(data["manifest_sha256"].item())
            if reference_manifest is None:
                reference_manifest = manifest
            elif manifest != reference_manifest:
                raise ValueError(f"Manifest mismatch: {path}")

            accepted = np.asarray(data["fit_accepted"], dtype=bool)
            available = np.asarray(data["estimate_available"], dtype=bool)
            M = np.asarray(data["M"])
            Mboot = np.asarray(data["M_boot"])
            filled = np.asarray(data["M_filled"])
            filled_boot = np.asarray(data["M_boot_filled"])
            if np.any(~np.isfinite(M[accepted])):
                raise ValueError(f"Accepted M has non-finite values: {path}")
            if np.any(np.isfinite(M[~accepted])):
                raise ValueError(f"Rejected M is not NaN: {path}")
            finite_fraction = np.mean(np.isfinite(Mboot), axis=0)
            if np.any(finite_fraction[accepted] < 0.90):
                raise ValueError(f"Accepted M_boot coverage below 0.90: {path}")
            if np.any(~np.isfinite(filled[available])) or np.any(
                ~np.isfinite(filled_boot[:, available])
            ):
                raise ValueError(f"Filled estimate is not finite: {path}")
            omitted = np.asarray(data["candidate_n_omitted_difference_points"])
            aic = np.asarray(data["candidate_range_aic_boot"])
            chi2 = np.asarray(data["candidate_chi2_boot"])
            expected_aic = chi2 + 2.0 + 2.0 * omitted[None, :, None, None, None]
            finite_aic = np.isfinite(aic) & np.isfinite(expected_aic)
            if np.any(np.abs(aic[finite_aic] - expected_aic[finite_aic]) > 2e-10):
                raise ValueError(f"AIC normalization mismatch: {path}")

            accepted_count = int(np.count_nonzero(accepted))
            point_count = int(accepted.size)
            filled_count = int(np.count_nonzero(available))
            fallback_count = int(np.count_nonzero(data["central_fallback_used"]))
            total_accepted += accepted_count
            total_points += point_count
            total_filled += filled_count
            total_fallback += fallback_count
            records.append(
                {
                    "tau_t_over_a2": float(data["flow_tau_t_over_a2"].item()),
                    "path": str(path.resolve()),
                    "sha256": output_sha,
                    "accepted_count": accepted_count,
                    "point_count": point_count,
                    "filled_count": filled_count,
                    "central_fallback_count": fallback_count,
                    "replica_fallback_fraction_mean": float(
                        np.nanmean(data["replica_fallback_fraction"])
                    ),
                }
            )

    payload = {
        "schema": "gradient_flow_gluon_bare_sumdiff_aic_all_operators_collection_validation_v1",
        "status": "complete_and_validated",
        "nflow": len(records),
        "nconf": 406,
        "nboot": 1000,
        "seed": 1115,
        "manifest_sha256": reference_manifest,
        "fixed_tmin": 8,
        "fixed_tmin_fm": 0.6200,
        "candidate_grid": "cuts=1,2,3; fixed Tmin=8; Tmax summed=9..15",
        "operators": list(OPERATORS),
        "shared_configuration_and_bootstrap_axes": True,
        "aic_normalization_checked": True,
        "total_accepted": total_accepted,
        "total_points": total_points,
        "total_filled": total_filled,
        "total_central_fallback": total_fallback,
        "records": records,
    }
    output = ROOT / "fit_collection_validation.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
