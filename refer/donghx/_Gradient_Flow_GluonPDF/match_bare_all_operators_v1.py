#!/usr/bin/env python3
"""Apply the corrected one-loop GF-to-MS conversion to all Nboot=1000 fits.

The input is the six-operator, Nconf=406, Nboot=1000 bare-matrix product from
fit_all_operators_aic_v1. The conversion is a deterministic scalar in
(flow time, z), applied replica by replica so the shared bootstrap covariance
and all fit-status masks are preserved.

For a Wilson line along z:

  unpolarized: Z_U = exp(-delta_m |z|) / c_perp^2
  helicity:    Z_H = exp(-delta_m |z|) / (c_parallel*c_perp)

This is GF-to-MSbar quasi-operator conversion only; no quasi/pseudo-ITD to
light-cone kernel or singlet mixing is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .gluon_flow_to_quasi import GEV_FM, coefficients
except ImportError:
    from gluon_flow_to_quasi import GEV_FM, coefficients


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "fit_all_operators_aic_v1" / "results_v1"
DEFAULT_OUTPUT = ROOT / "matching" / "bare_all_operators_matching_v1" / "results_v1"
TAUS = np.asarray(
    (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8),
    dtype=float,
)
N_OPERATOR = 6


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def input_path(root: Path, tau: float) -> Path:
    return root / f"bare_matrix_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def output_path(root: Path, tau: float) -> Path:
    return root / f"matched_bare_matrix_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def matching_factors(
    tau: float,
    z_values: np.ndarray,
    a_fm: float,
    mu_gev: float,
    alpha_s: float,
    ca: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return factors with shape (operator, |Pz|, z)."""
    c_parallel, c_perp, delta_m, t_gev_minus2 = coefficients(
        tau, a_fm, mu_gev, alpha_s, ca
    )
    z_gev = np.abs(np.asarray(z_values, dtype=float)) * a_fm * GEV_FM
    line = np.exp(-delta_m * z_gev)
    z_unpolarized = line / (c_perp * c_perp)
    z_helicity = line / (c_parallel * c_perp)
    factors = np.empty((N_OPERATOR, 1, len(z_values)), dtype=float)
    factors[:3, 0, :] = z_unpolarized
    factors[3:, 0, :] = z_helicity
    factors = np.broadcast_to(factors, (N_OPERATOR, 3, len(z_values))).copy()
    metadata = {
        "c_parallel_perp": float(c_parallel),
        "c_perp_perp": float(c_perp),
        "delta_m_GeV": float(delta_m),
        "t_GeV_minus2": float(t_gev_minus2),
        "z_line_uses_abs": True,
        "kappa_F": 1.0,
    }
    return factors, metadata


def _scale_operator_pz_z(array: np.ndarray, factors: np.ndarray) -> np.ndarray:
    if array.shape != factors.shape:
        raise ValueError(f"expected shape {factors.shape}, got {array.shape}")
    return array * factors


def _scale_boot_operator_pz_z(array: np.ndarray, factors: np.ndarray) -> np.ndarray:
    expected = (array.shape[0],) + factors.shape
    if array.shape != expected:
        raise ValueError(f"expected shape {expected}, got {array.shape}")
    return array * factors[None, ...]


def _scale_operator_cut_pz_z(array: np.ndarray, factors: np.ndarray) -> np.ndarray:
    expected = (factors.shape[0], array.shape[1], factors.shape[1], factors.shape[2])
    if array.shape != expected:
        raise ValueError(f"expected shape {expected}, got {array.shape}")
    return array * factors[:, None, :, :]


def _scale_boot_operator_cut_pz_z(array: np.ndarray, factors: np.ndarray) -> np.ndarray:
    expected = (
        array.shape[0],
        factors.shape[0],
        array.shape[2],
        factors.shape[1],
        factors.shape[2],
    )
    if array.shape != expected:
        raise ValueError(f"expected shape {expected}, got {array.shape}")
    return array * factors[None, :, None, :, :]


def _scale_operator_candidate_pz_z(array: np.ndarray, factors: np.ndarray) -> np.ndarray:
    expected = (factors.shape[0], array.shape[1], factors.shape[1], factors.shape[2])
    if array.shape != expected:
        raise ValueError(f"expected shape {expected}, got {array.shape}")
    return array * factors[:, None, :, :]


def _scale_candidate_boot(array: np.ndarray, factors: np.ndarray) -> np.ndarray:
    expected = (
        factors.shape[0],
        array.shape[1],
        array.shape[2],
        factors.shape[1],
        factors.shape[2],
    )
    if array.shape != expected:
        raise ValueError(f"expected shape {expected}, got {array.shape}")
    return array * factors[:, None, None, :, :]


def match_one(
    source: Path,
    destination: Path,
    a_fm: float,
    mu_gev: float,
    alpha_s: float,
    ca: float,
) -> dict:
    with np.load(source, allow_pickle=False) as data:
        raw = {key: np.asarray(data[key]) for key in data.files}

    tau = float(np.asarray(raw["flow_tau_t_over_a2"]).item())
    z_values = np.asarray(raw["z_values"], dtype=int)
    pabs = np.asarray(raw["pabs_values"], dtype=int)
    labels = tuple(np.asarray(raw["operator_labels"]).astype(str).tolist())
    if raw["M"].shape != (6, 3, 25):
        raise ValueError(f"unexpected M shape in {source}: {raw['M'].shape}")
    if raw["M_boot"].shape[1:] != raw["M"].shape:
        raise ValueError("M and M_boot axes disagree")
    if tuple(pabs.tolist()) != (3, 4, 5) or tuple(z_values.tolist()) != tuple(range(25)):
        raise ValueError("unexpected Pz/z axes")
    if len(labels) != N_OPERATOR:
        raise ValueError("unexpected operator count")
    factors, coeff = matching_factors(tau, z_values, a_fm, mu_gev, alpha_s, ca)

    for key in (
        "M",
        "statistical_error",
        "M_filled",
        "filled_statistical_error",
        "bootstrap_mean_diagnostic",
        "bootstrap_bias",
        "cut_spread_diagnostic",
    ):
        if key in raw:
            raw[key] = _scale_operator_pz_z(raw[key], factors)
    for key in ("M_boot", "M_boot_filled"):
        if key in raw:
            raw[key] = _scale_boot_operator_pz_z(raw[key], factors)
    for key in ("cut_M", "cut_statistical_error"):
        if key in raw:
            raw[key] = _scale_operator_cut_pz_z(raw[key], factors)
    if "cut_M_boot" in raw:
        raw["cut_M_boot"] = _scale_boot_operator_cut_pz_z(raw["cut_M_boot"], factors)
    for key in ("candidate_M",):
        if key in raw:
            raw[key] = _scale_operator_candidate_pz_z(raw[key], factors)
    if "candidate_M_boot" in raw:
        raw["candidate_M_boot"] = _scale_candidate_boot(raw["candidate_M_boot"], factors)

    raw["matching_factors"] = factors
    raw["matching_c_parallel_perp"] = np.asarray(coeff["c_parallel_perp"])
    raw["matching_c_perp_perp"] = np.asarray(coeff["c_perp_perp"])
    raw["matching_delta_m_GeV"] = np.asarray(coeff["delta_m_GeV"])
    raw["matching_t_GeV_minus2"] = np.asarray(coeff["t_GeV_minus2"])
    raw["matching_a_fm"] = np.asarray(a_fm)
    raw["matching_mu_GeV"] = np.asarray(mu_gev)
    raw["matching_alpha_s"] = np.asarray(alpha_s)
    raw["matching_kappa_F"] = np.asarray(1.0)
    raw["matching_status"] = np.asarray("one_loop_GF_to_MSbar_quasi_after_bare_fit")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **raw)
    return {
        "status": "complete",
        "input": str(source.resolve()),
        "output": str(destination.resolve()),
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "tau_t_over_a2": tau,
        "nconf": int(np.asarray(raw["nconf"]).item()),
        "nboot": int(np.asarray(raw["nboot"]).item()),
        "shape_M": list(raw["M"].shape),
        "coefficients": coeff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--a-fm", type=float, default=0.0775)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--alpha-s", type=float, default=0.25)
    parser.add_argument("--ca", type=float, default=3.0)
    parser.add_argument(
        "--taus",
        type=str,
        default=None,
        help="comma-separated subset; default is all 14 flow times",
    )
    args = parser.parse_args()
    if args.taus is None:
        taus = TAUS
    else:
        requested = [float(item.strip()) for item in args.taus.split(",") if item.strip()]
        if not requested:
            raise SystemExit("--taus must not be empty")
        canonical = []
        for value in requested:
            matches = TAUS[np.isclose(TAUS, value, rtol=0.0, atol=1.0e-9)]
            if matches.size != 1:
                raise SystemExit(f"unavailable tau {value:g}; choose from {TAUS.tolist()}")
            canonical.append(float(matches[0]))
        taus = np.asarray(canonical, dtype=float)
        if np.unique(taus).size != taus.size:
            raise SystemExit("--taus contains duplicates")

    records = []
    for tau in taus:
        source = input_path(args.input_root, float(tau))
        destination = output_path(args.output_root, float(tau))
        if not source.is_file():
            raise FileNotFoundError(source)
        receipt = match_one(source, destination, args.a_fm, args.mu, args.alpha_s, args.ca)
        receipt_path = Path(str(destination) + ".done.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        records.append(receipt)

    manifest = {
        "schema": "gradient_flow_gluon_bare_all_operators_matching_v1",
        "status": "complete",
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "flow_times_t_over_a2": [float(x) for x in taus],
        "nflow": len(taus),
        "a_fm": args.a_fm,
        "mu_GeV": args.mu,
        "alpha_s": args.alpha_s,
        "C_A": args.ca,
        "kappa_F": 1.0,
        "unpolarized_factor": "exp(-delta_m_A*|z|)/c_perp_perp^2",
        "helicity_factor": "exp(-delta_m_A*|z|)/(c_parallel_perp*c_perp_perp)",
        "delta_m": "-alpha_s*C_A/(4*pi)*sqrt(2*pi/t_GeV_minus2)",
        "records": records,
        "light_cone_matching": "not applied",
        "singlet_quark_mixing": "not applied",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
