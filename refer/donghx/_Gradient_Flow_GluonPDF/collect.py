#!/usr/bin/env python3
"""Align kinetic measurements to the bare bootstrap; validated export requires evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import ENSEMBLE, atomic_npz, canonical_hash, digest, metadata_array, validate_plan
from quark_sixflow_ringed_sfte_v1 import load_bare, ringed_from_kinetic

CHECKS = ("gpu_free_field", "step_halving", "restart_equivalence", "noise_convergence",
          "action_and_normalization", "massless_prescription", "flow_boundary_improvement")


def read_raw(path):
    with np.load(path, allow_pickle=False) as d:
        m = json.loads(str(d["metadata_json"]))
        identity = m["run_identity"]
        p = identity["plan"]
        validate_plan(p)
        if m.get("schema") != "quark_ringed_configuration_raw_v1":
            raise ValueError(f"Wrong raw schema: {path}")
        if canonical_hash(identity) != m["run_identity_sha256"]:
            raise ValueError("Raw run identity hash mismatch")
        conf = int(d["configuration"])
        if conf != identity["conf"] or p["ensemble"] != ENSEMBLE or p["lattice_xyzt"] != [32, 32, 32, 96]:
            raise ValueError("Not the requested L32x96 quark product")
        if p.get("free_field", False):
            raise ValueError("Not a physical-gauge trace")
        required = {
            "status": "measured_unvalidated", "axes_kinetic": ["flow", "noise"],
            "trace_space": "full_spin_color_spacetime", "temporal_truncation": "none",
            "noise": "full_volume_Z4_E_eta_eta_dagger_equals_I",
            "fermion_normalization": "D_mass_inverse_no_extra_2kappa",
            "derivative_discretization": "4D_symmetric_covariant_nearest_neighbor_aD",
            "flavor_count": 1,
        }
        for key, expected in required.items():
            if m.get(key) != expected:
                raise ValueError(f"Raw metadata {key} mismatch: {path}")
        if not np.array_equal(d["flow_times"], p["flow_times"]):
            raise ValueError("Inconsistent raw flow axis")
        data = np.array(d["kinetic_a4_noise"], complex)
        shape = (len(p["flow_times"]), identity["noise_count"])
        if data.shape != shape or shape[1] < 2 or not np.isfinite(data).all():
            raise ValueError("Invalid flow/noise array")
        if not np.array_equal(data.mean(axis=1), d["kinetic_a4_config"]):
            raise ValueError("Stored configuration mean differs from noise samples")
        residual = np.array(d["true_residual"])
        if residual.shape != (shape[1],) or not np.isfinite(residual).all() or np.any(residual < 0) or np.any(residual > p["residual_max"]):
            raise ValueError("Bad true residuals")
        return conf, p, data, m


def validate_review(path, input_hashes, plan):
    with open(path) as f:
        report = json.load(f)
    if report.get("schema") != "quark_ringed_validation_review_v1" or report.get("status") != "validated":
        raise ValueError("Acceptance template is not a validated physics review")
    if report.get("input_sha256_by_conf") != input_hashes:
        raise ValueError("Acceptance review must bind exactly these raw configuration artifacts")
    prescription = report["mass_prescription"]
    if prescription.get("kind") != "direct_critical_mass" or prescription.get("bare_mass") != plan["mass"]:
        raise ValueError("This exporter requires an independently tuned critical valence mass")
    if "finite_mass_pilot" in plan["mass_treatment"]:
        raise ValueError("Finite-mass pilot cannot be promoted to the massless ringed scheme")
    for name in CHECKS:
        entry = report["checks"][name]
        evidence = Path(entry["evidence_path"])
        if entry["status"] != "passed" or not str(entry.get("summary", "")).strip():
            raise ValueError(f"Unconfirmed validation gate: {name}")
        if not evidence.is_file() or digest(evidence) != entry["evidence_sha256"]:
            raise ValueError(f"Missing or changed evidence for {name}")
    return report


def collect(inputs, bare_path, output, review_path=None, imaginary_sigma_limit=5.0):
    if not np.isfinite(imaginary_sigma_limit) or imaginary_sigma_limit <= 0:
        raise ValueError("Invalid imaginary significance threshold")
    bare = load_bare(bare_path)
    records, hashes, paths, common_plan = {}, {}, {}, None
    for path in inputs:
        conf, plan, data, meta = read_raw(path)
        if conf in records:
            raise ValueError(f"Duplicate configuration {conf}")
        if common_plan is None:
            common_plan = plan
        elif common_plan != plan:
            raise ValueError("Mixed action/mass/flow plans cannot be pooled")
        if not np.array_equal(plan["flow_times"], bare["flow_times"]) or plan["a_fm"] != float(bare["a_fm"]):
            raise ValueError("Flow/spacing mismatch with bare product")
        if not np.isclose(plan["epsilon"], .01, atol=1e-12, rtol=0):
            raise ValueError("Production export must match bare epsilon=0.01")
        records[conf] = (data, meta)
        hashes[str(conf)] = digest(path)
        paths[str(conf)] = str(Path(path).resolve())
    confs = bare["confs"].astype(int)
    if set(records) != set(confs):
        raise ValueError("Need exactly the bare product configuration list, without omissions/extras")
    values = np.stack([records[c][0].mean(axis=1) for c in confs], axis=1)
    nnoise = np.array([records[c][0].shape[1] for c in confs])
    noise_sem = np.stack([records[c][0].real.std(axis=1, ddof=1)/np.sqrt(nnoise[i])
                         for i, c in enumerate(confs)], axis=1)
    # Same raw gauge draws as h, with no second independent gauge bootstrap.
    indices = bare["bootstrap_indices"]
    iboot = values.imag[:, indices].mean(axis=2)
    imean = values.imag.mean(axis=1)
    ierr = iboot.std(axis=1, ddof=1)
    imag_pass = np.abs(imean) <= imaginary_sigma_limit*ierr + 1e-14
    zchi, zboot, mean, eboot = ringed_from_kinetic(values.real, bare["flow_times"], indices)
    normalized = review_path is not None
    if normalized:
        validate_review(review_path, hashes, common_plan)
        if not np.all(imag_pass):
            raise ValueError("Kinetic imaginary expectation is not consistent with zero")
    status = "validated" if normalized else "diagnostic_NOT_massless_ringed"
    meta = {"schema": "quark_ringed_kinetic_config_v1", "status": status,
            "ensemble": ENSEMBLE, "gauge_scheme": "thin", "flow_dimension": 4,
            "flavor_count": 1, "units": "a^4_times_dimension4_kinetic_expectation",
            "spacetime_reduction": "volume_average", "trace_space": "full_spin_color_spacetime",
            "massless_prescription_validated": normalized, "fermion_normalization_matches_bare": normalized,
            "fermion_temporal_boundary": "antiperiodic", "flow_epsilon": common_plan["epsilon"],
            "a_fm": common_plan["a_fm"], "flow_integrator": "QUDA_coupled_RK3_4D_Wilson_and_Laplace",
            "derivative_discretization": "4D_symmetric_covariant_nearest_neighbor_aD",
            "mass_treatment": common_plan["mass_treatment"], "solver_action": common_plan,
            "fermion_normalization": "D_mass_inverse_no_extra_2kappa",
            "measurement_provenance": paths,
            "input_sha256_by_conf": hashes, "bare_sha256": digest(bare_path),
            "validation_report": str(Path(review_path).resolve()) if normalized else "NOT_VALIDATED",
            "validation_report_sha256": digest(review_path) if normalized else None,
            "stochastic_error_treatment": "iid_configuration_bootstrap_of_noise_means; noise convergence requires external review; no double-counting nested bootstrap",
            "imaginary_sigma_limit": imaginary_sigma_limit,
            "imaginary_check_pass_by_flow": imag_pass.tolist()}
    atomic_npz(output, metadata_json=metadata_array(meta), flow_times=bare["flow_times"], confs=confs,
               kinetic_a4_config=values.real, kinetic_a4_config_complex=values,
               bootstrap_indices=indices, kinetic_a4_bootstrap=eboot, kinetic_a4_central=mean,
               noise_count_by_conf=nnoise, noise_sem_real=noise_sem,
               imaginary_mean=imean, imaginary_bootstrap_error=ierr,
               Zchi=zchi if normalized else np.full_like(zchi, np.nan),
               Zchi_bootstrap=zboot if normalized else np.full_like(zboot, np.nan),
               Zchi_diagnostic_NOT_massless=zchi, Zchi_diagnostic_bootstrap=zboot)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--bare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--imaginary-sigma-limit", type=float, default=5.0)
    args = parser.parse_args()
    meta = collect(args.inputs, args.bare, args.output, args.review, args.imaginary_sigma_limit)
    print(json.dumps({"output": str(args.output), "status": meta["status"]}, indent=2))


if __name__ == "__main__":
    main()
