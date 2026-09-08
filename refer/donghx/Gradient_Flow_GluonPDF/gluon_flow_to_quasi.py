#!/usr/bin/env python3
"""One-loop 4D gradient-flow -> MSbar quasi-operator conversion.

The input is the per-configuration flowed OPE product written by
``scripts/run_one_v5.py``.  Mtiti and Mijij are retained separately, then all
channel combinations are reconstructed.  The factors below are the one-loop
small-flow-time conversion of the two field-strength endpoints.

Convention (Brambilla--Wang, arXiv:2312.05032):
  M_MS = [c_i(t,mu)c_j(t,mu)]^(-1)
          exp[-delta_m(t)|z|] M_GF + O(t),
  c_parallel=1, c_perp=1 + alpha_s*C_A/(4*pi)*log(2*mu^2*t*exp(gamma_E)),
  delta_m=-alpha_s*C_A/(4*pi)*sqrt(2*pi)/sqrt(t).

The result is a matched quasi-operator in the same lattice coordinate basis,
not yet a light-cone PDF.  The input must be 4D Wilson-flow, thin-link data.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

GEV_FM = 5.067730716
CA = 3.0
EULER_GAMMA = 0.5772156649015329
SCHEMA = "gradient_flow_gluon_flow_to_quasi_v1"

def tau_tag(x: float) -> str:
    return f"tau{x:.3f}".replace(".", "p")

def coefficients(tau, a_fm, mu_gev, alpha_s, ca=CA):
    """Return (c_parallel, c_perp, delta_m[GeV], t[GeV^-2])."""
    tgev = float(tau) * (float(a_fm) * GEV_FM) ** 2
    if tgev <= 0 or mu_gev <= 0 or alpha_s < 0:
        raise ValueError("tau, mu and alpha_s must be positive (alpha_s >= 0)")
    cpar = 1.0
    cperp = 1.0 + alpha_s * ca / (4.0 * math.pi) * math.log(
        2.0 * mu_gev**2 * tgev * math.exp(EULER_GAMMA))
    dm = -alpha_s * ca / (4.0 * math.pi) * math.sqrt(2.0 * math.pi / tgev)
    return cpar, cperp, dm, tgev

def match_one(arr, tau, a_fm, mu_gev, alpha_s, ca=CA):
    """Match one OPE array with axes (projection,channel,orientation,component,z,t)."""
    if arr.ndim != 6:
        raise ValueError(f"expected six axes, got {arr.shape}")
    cpar, cperp, dm, tgev = coefficients(tau, a_fm, mu_gev, alpha_s, ca)
    z = np.arange(arr.shape[4], dtype=float) * float(a_fm) * GEV_FM
    line = np.exp(-dm * z)[None, None, None, None, :, None]
    out = np.asarray(arr, dtype=np.complex128).copy()
    # The parallel/perpendicular labels are relative to the Wilson-line
    # direction (z), not to Euclidean time.  Therefore F_tx, F_ty and F_xy
    # are all F_perp-perp in the unpolarized channel.  A helicity primitive
    # F W Fdual has one F_perp-perp and one F_parallel-perp endpoint.  Apply
    # the product of endpoint factors to every stored component, then
    # reconstruct the linear combinations below.
    # components: combined=0, Mtiti=1, Mijij=2, ti=3, tj=4, ij_single=5
    f_unpol = 1.0 / (cperp * cperp)
    f_hel = 1.0 / (cpar * cperp)
    factors = np.ones((arr.shape[1],), dtype=float)
    if arr.shape[1] >= 2:
        factors[0] = f_unpol
        factors[1] = f_hel
    out *= factors[None, :, None, None, None, None]
    out *= line
    # Reconstruct combinations from the matched independent components.
    if arr.shape[3] >= 3:
        mt, ms = out[..., 1, :, :], out[..., 2, :, :]
        # Channel 0 follows the unpolarized code convention T_U-S_U;
        # channel 1 follows the stored helicity production convention
        # T_H+S_H.  Mtiti/Mijij remain available for the T_H-S_H check.
        out[:, 0, ..., 0, :, :] = mt[:, 0, ...] - ms[:, 0, ...]
        out[:, 1, ..., 0, :, :] = mt[:, 1, ...] + ms[:, 1, ...]
    return out, {"c_parallel_perp": cpar, "c_perp_perp": cperp,
                 "delta_m_GeV": dm, "t_GeV_minus2": tgev,
                 "line_factor": "exp(-delta_m*|z|)",
                 "component_factors_unpolarized": f_unpol,
                 "component_factors_helicity": f_hel,
                 "component_factors_by_channel": factors.tolist()}

def process_file(src: Path, dst: Path, args):
    arr = np.load(src, allow_pickle=False)
    meta_path = src.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tau = float(meta.get("flow", {}).get("tau_t_over_a2", args.tau))
    if abs(tau - args.tau) > 1e-9:
        raise ValueError(f"tau mismatch: argument {args.tau} vs metadata {tau} in {src}")
    if meta.get("input_scheme", "").find("thin_link") < 0:
        raise ValueError(f"refusing non-thin-link input: {src}")
    matched, coeff = match_one(arr, tau, args.a_fm, args.mu, args.alpha_s, args.ca)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, matched)
    outmeta = {"schema": SCHEMA, "source": str(src), "source_schema": meta.get("schema"),
               "conf_id": meta.get("conf_id"), "input_scheme": meta.get("input_scheme"),
               "shape": list(matched.shape), "dtype": str(matched.dtype),
               "status": "flow_to_MSbar_quasi_operator_one_loop", "flow": meta.get("flow", {}),
               "matching": {"scheme": "Brambilla-Wang-2024", "order": "one_loop",
                            "mu_GeV": args.mu, "alpha_s": args.alpha_s, "C_A": args.ca,
                            "a_fm": args.a_fm, **coeff},
               "operator": meta.get("operator", {}), "axes": meta.get("axes"),
               "axis_labels": meta.get("axis_labels", {}), "channels": meta.get("channels"),
               "component_policy": "for v parallel z, use c_perp^(-2) for unpolarized F_perp-perp endpoints and (c_parallel*c_perp)^(-1) for helicity F W Fdual; reconstruct combined as unpolarized Mtiti-Mijij and stored helicity Mtiti+Mijij; retain components for TH-SH check",
               "light_cone_matching": "not applied"}
    dst.with_suffix(".json").write_text(json.dumps(outmeta, indent=2) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="one .npy or directory of per-conf files")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tau", type=float, required=True, help="flow time t/a^2")
    p.add_argument("--a-fm", type=float, default=0.0775)
    p.add_argument("--mu", type=float, default=2.0, help="MSbar scale in GeV")
    p.add_argument("--alpha-s", type=float, default=0.25)
    p.add_argument("--ca", type=float, default=CA)
    args = p.parse_args()
    if args.input.is_file():
        process_file(args.input, args.output, args)
        print(args.output)
        return
    files = sorted(args.input.glob("**/*.npy"))
    if not files:
        raise SystemExit(f"no .npy files under {args.input}")
    for src in files:
        rel = src.relative_to(args.input)
        process_file(src, args.output / rel, args)
    manifest = {"schema": SCHEMA, "nfiles": len(files), "tau": args.tau,
                "mu_GeV": args.mu, "alpha_s": args.alpha_s, "a_fm": args.a_fm,
                "source": str(args.input), "output": str(args.output)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))

if __name__ == "__main__":
    main()
