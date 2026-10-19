#!/usr/bin/env python3
"""20-configuration flowed-quark ratio/bootstrap/bare-fit analysis.

This is an analysis-only implementation for the positive-z pilot products.  It
does not modify production data.  The executable gamma/projector convention is
the one in Diagram_GluonPDF/Gradient_flow/gfbare and the ratio is formed only
after source and configuration averaging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import chi2 as _chi2
except Exception:  # pragma: no cover - scipy is optional for metadata-only runs
    _chi2 = None


HERE = Path(__file__).resolve().parent
OUT = HERE
ENSEMBLE = "beta6.41_mu-0.2295_ms-0.2050_L32x96"
CONFS = [
    "10000", "10100", "10200", "10300", "10400", "10500", "10600",
    "10700", "10800", "10900", "11000", "11100", "12300", "12400",
    "12500", "12600", "12700", "12800", "12900", "13000",
]
TAU_SPECS = {
    0.05: "thin_tf0p05000000_eps0p01000000_tw2_Nv400_dz21",
    0.10: "thin_tf0p10000000_eps0p01000000_tw2_Nv400_dz21",
    0.20: "thin_tf0p20000000_eps0p01000000_tw3_Nv400_dz21",
    0.50: "thin_tf0p50000000_eps0p01000000_tw4_Nv400_dz21",
    0.80: "thin_tf0p80000000_eps0p01000000_tw6_Nv400_dz21",
    1.00: "thin_tf1p00000000_eps0p01000000_tw7_Nv400_dz21",
}
TAUS = tuple(TAU_SPECS)
TSEPS = (6, 7, 8, 9, 10)
PZ = (3, 4, 5)
NCONF = len(CONFS)
NBOOT = 100
SEED = 1115
NT = 96
DZ = 21
CHANNELS = ("unpolarized", "helicity", "transversity")
FLAVORS = ("u", "d", "u_minus_d")
VDV = ("even", "odd")
PHASES = {"unpolarized": 1.0 + 0.0j, "helicity": -1.0j, "transversity": 1.0j}
TRANSVERSITY_STATUS = "pending_external_physics_confirmation"

OLD_ROOT = Path("/public/group/lqcd/donghx/Gradient_Flow/Result") / ENSEMBLE / "thin_link"
NEW_ROOT = Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF") / "quark_spin2_fill_flows_20260923/results_remaining"
TF05_ROOT = Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF") / "quark_spin2_expansion20_20260922/results_tf0p5_pilot_z"


def gamma_dr() -> dict[str, np.ndarray]:
    c = np.complex128
    g1 = np.zeros((4, 4), c); g1[0, 3] = g1[1, 2] = 1j; g1[2, 1] = g1[3, 0] = -1j
    g2 = np.zeros((4, 4), c); g2[0, 3], g2[1, 2], g2[2, 1], g2[3, 0] = -1, 1, 1, -1
    g3 = np.zeros((4, 4), c); g3[0, 2], g3[1, 3], g3[2, 0], g3[3, 1] = 1j, -1j, -1j, 1j
    g4 = np.zeros((4, 4), c); g4[0, 2] = g4[1, 3] = g4[2, 0] = g4[3, 1] = 1
    g5 = np.diag([1, 1, -1, -1]).astype(c)
    return {"1": g1, "2": g2, "3": g3, "4": g4, "5": g5}


G = gamma_dr()
GAMMA = {
    "unpolarized": G["4"],
    "helicity": G["5"] @ G["3"],
    "transversity_1": G["4"] @ G["1"] @ G["5"],
    "transversity_2": G["4"] @ G["2"] @ G["5"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_path(conf: str, tau: float, tsep: int) -> Path:
    if conf in CONFS[:10]:
        root = OLD_ROOT
    else:
        root = TF05_ROOT if abs(tau - 0.5) < 1e-12 else NEW_ROOT
    return root / TAU_SPECS[tau] / "Nev_200" / conf / f"corr_projected_tsep{tsep}_dz0-20_Nev200.npz"


def check_inputs(taus: tuple[float, ...] = TAUS) -> dict[str, object]:
    missing = []
    records = []
    for tau in taus:
        for conf in CONFS:
            for tsep in TSEPS:
                p = source_path(conf, tau, tsep)
                # A full SHA256 over all raw products is intentionally deferred: each
                # product is hundreds of MB and the analysis itself reopens it below.
                # Record a cheap stat fingerprint in the manifest and hash outputs.
                fp = None if not p.is_file() else {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                records.append({"tau_a2": tau, "conf": conf, "tsep": tsep, "path": str(p), "stat_fingerprint": fp})
                if not p.is_file():
                    missing.append(str(p))
    return {"missing": missing, "records": records}


def _source_average(contracted: np.ndarray, tsep: int) -> np.ndarray:
    """Contracted axes are sign,dz,t_source,t_current -> sign,dz,tau."""
    ts = np.arange(NT, dtype=np.int32)
    out = np.empty((2, DZ, tsep + 1), np.complex128)
    for tau in range(tsep + 1):
        out[..., tau] = np.mean(contracted[..., ts, (ts + tau) % NT], axis=-1)
    return out


def _project(c3: np.ndarray, pol: int, mom: int, tsep: int, operator: str) -> np.ndarray:
    contracted = np.einsum("sdmoab,mo->sdab", c3[:, pol, mom], GAMMA[operator], optimize=True)
    return _source_average(contracted, tsep)


def project_flow(tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Return cfg arrays: C3 (conf,channel,flavor,sign,p,z,T,tau), C2 (conf,p,T)."""
    c3 = np.full((NCONF, len(CHANNELS), 2, 2, 3, DZ, len(TSEPS), max(TSEPS) + 1), np.nan + 1j * np.nan, dtype=np.complex128)
    c2 = np.full((NCONF, 3, len(TSEPS)), np.nan + 1j * np.nan, dtype=np.complex128)
    for ic, conf in enumerate(CONFS):
        for it, tsep in enumerate(TSEPS):
            path = source_path(conf, tau, tsep)
            channel_path = path.parent / "channel_projected_v1" / path.name.replace("corr_projected_", "corr_channels_")
            # Existing old-10 products already contain the compact projected
            # channels.  Use them to avoid reopening multi-GB current-spin NPZs;
            # new-10 products are projected from the raw product until their
            # compact files are produced.
            with np.load(channel_path if channel_path.is_file() else path, allow_pickle=False) as d:
                if channel_path.is_file():
                    labels = [str(x) for x in d["pol_labels"]]
                    moms = [str(x) for x in d["mom_labels"]]
                    if labels != ["nopol", "pol15", "pol25", "pol35"] or moms != ["Pp3_z", "Pp4_z", "Pp5_z"]:
                        raise ValueError(f"Projected metadata changed in {channel_path}")
                    c2[ic, :, it] = d["c2_nopol_source_mean"]
                    for iflav, flav in enumerate(("u", "d")):
                        projected = np.asarray(d[f"c3_{flav}_avg"], np.complex128)
                        c3[ic, 0, iflav, :, :, :, it, :tsep + 1] = projected[0]
                        c3[ic, 1, iflav, :, :, :, it, :tsep + 1] = PHASES["helicity"] * projected[1]
                        c3[ic, 2, iflav, :, :, :, it, :tsep + 1] = 0.5 * PHASES["transversity"] * (projected[2] + projected[3])
                    continue
                if tuple(d["corr_2pt"].shape) != (4, 3, NT, NT):
                    raise ValueError(f"Unexpected C2 shape in {path}: {d['corr_2pt'].shape}")
                labels = [str(x) for x in d["pol_labels"]]
                moms = [str(x) for x in d["mom_labels"]]
                if moms != ["Pp3_z", "Pp4_z", "Pp5_z"]:
                    raise ValueError(f"Momentum order changed in {path}: {moms}")
                if labels != ["nopol", "pol15", "pol25", "pol35"]:
                    raise ValueError(f"Projector order changed in {path}: {labels}")
                if [str(x) for x in d["vdv_sign_labels"]] != ["minus_dz", "plus_dz"]:
                    raise ValueError(f"VDV order changed in {path}")
                ts = np.arange(NT, dtype=np.int32)
                # Two-stage indexing preserves the declared (momentum,t_source)
                # order; direct mixed advanced indexing moves axes in NumPy.
                c2_selected = d["corr_2pt"][0][:, ts, (ts + tsep) % NT]
                c2[ic, :, it] = np.mean(c2_selected, axis=-1)
                for iflav, flav in enumerate(("u", "d")):
                    raw = np.asarray(d[f"corr_3pt_{flav}"], np.complex128)
                    for ip in range(3):
                        # The stored 4x4 current-spin indices are fully contracted here.
                        unpol = _project(raw, 3, ip, tsep, "unpolarized")
                        helic = PHASES["helicity"] * _project(raw, 3, ip, tsep, "helicity")
                        tr1 = _project(raw, 1, ip, tsep, "transversity_1")
                        tr2 = _project(raw, 2, ip, tsep, "transversity_2")
                        trans = 0.5 * PHASES["transversity"] * (tr1 + tr2)
                        c3[ic, 0, iflav, :, ip, :, it, :tsep + 1] = unpol
                        c3[ic, 1, iflav, :, ip, :, it, :tsep + 1] = helic
                        c3[ic, 2, iflav, :, ip, :, it, :tsep + 1] = trans
    return c3, c2


def bootstrap_indices() -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, NCONF, size=(NBOOT, NCONF), dtype=np.int32)


def weights(indices: np.ndarray) -> np.ndarray:
    out = np.zeros((len(indices), NCONF), np.float64)
    for ib, row in enumerate(indices):
        out[ib] = np.bincount(row, minlength=NCONF) / float(NCONF)
    return out


def boot_mean(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    flat = x.reshape(NCONF, -1)
    return (w @ flat).reshape((len(w),) + x.shape[1:])


def make_ratio(c3: np.ndarray, c2: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return original and bootstrap ratio with axes vdv,p,z,T,tau."""
    num = c3[:, :, :, 0] - c3[:, :, :, 1]  # cfg,channel,sign,p,z,T,tau
    den = c2
    num_mean = np.mean(num, axis=0)
    den_mean = np.mean(den, axis=0)
    num_b = boot_mean(w, num)
    den_b = boot_mean(w, den)
    # denominator has no sign/channel/z/tau axes
    den_b_exp = den_b[:, None, None, :, None, :, None]
    den_mean_exp = den_mean[None, None, :, None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = num_b / den_b_exp
        ro = num_mean / den_mean_exp
    # VDV: input sign order is minus, plus. +z odd sign is +1.
    rb = np.stack((0.5 * (rb[:, :, 1] + rb[:, :, 0]), 0.5 * (rb[:, :, 0] - rb[:, :, 1])), axis=1)
    ro = np.stack((0.5 * (ro[:, 1] + ro[:, 0]), 0.5 * (ro[:, 0] - ro[:, 1])), axis=0)
    # rb currently boot,channel,p,z,T,tau; transpose to boot,channel,vdv,p,z,T,tau.
    rb = np.transpose(rb, (0, 1, 2, 3, 4, 5)) if rb.ndim == 6 else rb
    # The stack above has boot,channel,vdv,p,z,T,tau only when selecting correctly.
    # Normalize shape explicitly for readability and future input checks.
    if rb.ndim != 7:
        raise RuntimeError(f"ratio bootstrap shape error: {rb.shape}")
    valid = np.all(np.isfinite(den_b), axis=0)[:, None, :, None]  # p,T,1 -> broadcast below
    valid = np.broadcast_to(valid[:, None, :, None, :], (2, 3, len(TSEPS), max(TSEPS) + 1)) if False else None
    return ro, rb, den_mean, den_b


def _ratio_components(ro: np.ndarray, rb: np.ndarray, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert complex VDV ratio to the two fit components."""
    # ro: vdv,p,z,T,tau; rb: boot,vdv,p,z,T,tau
    original = np.stack((ro[0].real, ro[1].imag), axis=0)
    boot = np.stack((rb[:, 0].real, rb[:, 1].imag), axis=1)
    return original, boot


def _cov(samples: np.ndarray) -> np.ndarray:
    if samples.shape[1] == 1:
        return np.ones((1, 1), float) * max(float(np.var(samples[:, 0], ddof=1)), 1e-18)
    c = np.cov(samples, rowvar=False, ddof=1)
    return np.atleast_2d(np.asarray(c, float))


def _linear_fit(x: np.ndarray, y: np.ndarray, samples: np.ndarray, constant: bool = False) -> dict[str, object]:
    """Correlated GLS fit with deterministic diagonal shrinkage and bootstrap slopes."""
    finite = np.all(np.isfinite(samples), axis=1)
    if np.sum(finite) < 2 or not np.all(np.isfinite(y)):
        return {"value": np.nan, "boot": np.full(len(samples), np.nan), "chi2": np.nan, "dof": -1, "q": np.nan, "success": 0.0, "rank": 0, "condition": np.nan, "lambda": np.nan}
    y = np.asarray(y, float); samples = np.asarray(samples, float)
    cov = _cov(samples[finite])
    diag = np.diag(np.diag(cov)); lam = 0.10
    cov = (1.0 - lam) * cov + lam * diag
    scale = max(float(np.max(np.abs(np.diag(cov)))), 1.0)
    cov = cov + np.eye(cov.shape[0]) * (1e-12 * scale)
    inv = np.linalg.pinv(cov, rcond=1e-12)
    X = np.ones((len(x), 1), float) if constant else np.column_stack((np.ones(len(x)), x))
    normal = X.T @ inv @ X
    normal_inv = np.linalg.pinv(normal, rcond=1e-12)
    beta = normal_inv @ X.T @ inv @ y
    yfit = X @ beta
    resid = y - yfit
    dof = len(y) - X.shape[1]
    chi = float(resid @ inv @ resid)
    q = float(_chi2.sf(chi, dof)) if (_chi2 is not None and dof > 0) else np.nan
    bs = np.full((len(samples), X.shape[1]), np.nan)
    for ib, yy in enumerate(samples):
        if np.all(np.isfinite(yy)):
            bs[ib] = normal_inv @ X.T @ inv @ yy
    return {"value": float(beta[-1] if not constant else beta[0]), "boot": bs[:, -1] if not constant else bs[:, 0], "chi2": chi, "dof": dof, "q": q, "success": float(np.mean(np.isfinite(bs[:, -1] if not constant else bs[:, 0]))), "rank": int(np.linalg.matrix_rank(normal)), "condition": float(np.linalg.cond(normal)), "lambda": lam}


def candidates():
    return (("T7-9_L1_R1", (7, 8, 9), 1, 1), ("T7-10_L1_R1", (7, 8, 9, 10), 1, 1), ("T7-9_L2_R2", (7, 8, 9), 2, 2), ("T7-10_L2_R2", (7, 8, 9, 10), 2, 2))


def fit_one(original: np.ndarray, boot: np.ndarray, method: str) -> dict[str, np.ndarray]:
    # original: comp,p,z,T,tau; boot: boot,comp,p,z,T,tau
    ncomp, npz, ndz = 2, 3, DZ
    cands = candidates(); nc = len(cands)
    value = np.full((ncomp, nc, npz, ndz), np.nan); stat = value.copy(); chi = value.copy(); q = value.copy(); dof = np.full(value.shape, -1, np.int32); succ = value.copy(); rank = np.zeros(value.shape, np.int32); cond = value.copy()
    bvals = np.full((ncomp, nc, NBOOT, npz, ndz), np.nan)
    for ic in range(ncomp):
        for ia, (_, ts, left, right) in enumerate(cands):
            for ip in range(npz):
                for iz in range(ndz):
                    ys = []; bs = []
                    for t in ts:
                        it = TSEPS.index(t); lo, hi = left, t - right
                        if hi < lo: break
                        vals = original[ic, ip, iz, it, lo:hi + 1]
                        reps = boot[:, ic, ip, iz, it, lo:hi + 1]
                        if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(reps)):
                            break
                        ys.append(np.sum(vals))
                        bs.append(np.sum(reps, axis=1))
                    if len(ys) != len(ts): continue
                    yb = np.stack(bs, axis=1); y = np.asarray(ys, float)
                    if method == "sumratio":
                        x = np.asarray([t - left - right + 1 for t in ts], float)
                        fit = _linear_fit(x, y, yb, constant=False)
                    else:
                        vals = []; reps = []
                        for t in ts:
                            it = TSEPS.index(t); lo, hi = left, t - right
                            current = original[ic, ip, iz, it, lo:hi + 1]
                            current_boot = boot[:, ic, ip, iz, it, lo:hi + 1]
                            if not np.all(np.isfinite(current)) or not np.all(np.isfinite(current_boot)):
                                vals = []; reps = []; break
                            vals.extend(current); reps.append(current_boot)
                        if not vals:
                            continue
                        y = np.asarray(vals, float); yb = np.concatenate(reps, axis=1)
                        fit = _linear_fit(np.arange(len(y), dtype=float), y, yb, constant=True)
                    value[ic, ia, ip, iz] = fit["value"]; bvals[ic, ia, :, ip, iz] = fit["boot"]; stat[ic, ia, ip, iz] = np.nanstd(fit["boot"], ddof=1)
                    chi[ic, ia, ip, iz] = fit["chi2"]; q[ic, ia, ip, iz] = fit["q"]; dof[ic, ia, ip, iz] = fit["dof"]; succ[ic, ia, ip, iz] = fit["success"]; rank[ic, ia, ip, iz] = fit["rank"]; cond[ic, ia, ip, iz] = fit["condition"]
    accepted = np.isfinite(value) & (q >= 0.05) & (np.divide(chi, dof, out=np.full_like(chi, np.inf), where=dof > 0) <= 3.0) & (succ >= 0.90)
    primary = 3
    primary_ok = accepted[:, primary]
    count = np.sum(accepted, axis=1)
    fit_valid = primary_ok & (count >= 2)
    raw = value[:, primary]; rawerr = stat[:, primary]
    window = np.full((ncomp, npz, ndz), np.nan)
    for ic in range(ncomp):
        for ip in range(npz):
            for iz in range(ndz):
                alt = np.flatnonzero(accepted[ic, :, ip, iz]); alt = alt[alt != primary]
                if fit_valid[ic, ip, iz] and len(alt): window[ic, ip, iz] = np.sqrt(np.mean((value[ic, alt, ip, iz] - raw[ic, ip, iz]) ** 2))
    return {"bare_matrix_element": np.where(fit_valid, raw, np.nan), "bare_matrix_element_raw_diagnostic": raw, "statistical_error": np.where(fit_valid, rawerr, np.nan), "statistical_error_raw_diagnostic": rawerr, "window_systematic": window, "total_error": np.sqrt(np.where(fit_valid, rawerr, np.nan) ** 2 + window ** 2), "fit_valid": fit_valid, "accepted_window_count": count, "candidate_value_raw": value, "candidate_statistical_error_raw": stat, "candidate_fit_accepted": accepted, "candidate_chi2": chi, "candidate_dof": dof, "candidate_Q": q, "candidate_bootstrap_success_fraction": succ, "candidate_covariance_rank": rank, "candidate_covariance_condition": cond, "candidate_boot_values_raw": bvals}


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    tmp.replace(path)


def run(replace: bool = False, skip_fit: bool = False, taus: tuple[float, ...] = TAUS) -> None:
    OUT.joinpath("ratios").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("fits").mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(taus) - set(TAUS))
    if unknown:
        raise ValueError(f"Unsupported flow times: {unknown}")
    audit = check_inputs(taus)
    (OUT / "input_manifest.json").write_text(json.dumps({"ensemble": ENSEMBLE, "confs": CONFS, "tseps": TSEPS, "taus": taus, "records": audit["records"], "missing": audit["missing"]}, indent=2), encoding="utf-8")
    if audit["missing"]:
        raise FileNotFoundError(f"Missing {len(audit['missing'])} input files; see {OUT / 'input_manifest.json'}")
    indices = bootstrap_indices(); w = weights(indices)
    np.save(OUT / "bootstrap_indices_Nboot100_seed1115.npy", indices)
    old_summary = {}
    if (OUT / "run_summary.json").is_file():
        try:
            old_summary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_summary = {}
    all_taus = tuple(sorted(set(float(x) for x in old_summary.get("taus", [])) | set(taus)))
    summary = {"ensemble": ENSEMBLE, "nconf": NCONF, "nboot": NBOOT, "seed": SEED, "confs": CONFS, "tseps": TSEPS, "taus": all_taus, "channels": CHANNELS, "flavors": FLAVORS, "vdv": VDV, "estimator_id": "post_ratio_single_positive_z_vdv_average_v1", "estimator_formula": "VDV(bootstrap_cfg_mean(source_mean_C3)/bootstrap_cfg_mean(source_mean_C2))", "fit_primary": "T7-10_L2_R2", "transversity_convention_status": TRANSVERSITY_STATUS, "input_manifest": str(OUT / "input_manifest.json")}
    for tau in taus:
        c3, c2 = project_flow(tau)
        flow_tag = f"tau{tau:.2f}".replace(".", "p")
        for ich, channel in enumerate(CHANNELS):
            for flavor in FLAVORS:
                if flavor == "u": cfg = c3[:, ich, 0]
                elif flavor == "d": cfg = c3[:, ich, 1]
                else: cfg = c3[:, ich, 0] - c3[:, ich, 1]
                num_mean = np.mean(cfg, axis=0)
                num_boot = boot_mean(w, cfg)
                den_mean = np.mean(c2, axis=0)
                den_boot = boot_mean(w, c2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_b = num_boot / den_boot[:, None, :, None, :, None]
                    ratio_o = num_mean / den_mean[None, :, None, :, None]
                # cfg axes sign,p,z,T,tau -> ratio axes boot/sign,p,z,T,tau
                ratio_b = np.stack((0.5 * (ratio_b[:, 1] + ratio_b[:, 0]), 0.5 * (ratio_b[:, 0] - ratio_b[:, 1])), axis=1)
                ratio_o = np.stack((0.5 * (ratio_o[1] + ratio_o[0]), 0.5 * (ratio_o[0] - ratio_o[1])), axis=0)
                finite = np.all(np.isfinite(den_boot), axis=0) & (np.abs(np.mean(den_boot, axis=0)) > 1e-14)
                tau_mask = np.zeros((len(TSEPS), max(TSEPS) + 1), dtype=bool)
                for it, t in enumerate(TSEPS):
                    tau_mask[it, :t + 1] = True
                valid = np.broadcast_to(finite[None, :, None, :, None], (2, 3, DZ, len(TSEPS), max(TSEPS) + 1)).copy()
                valid &= tau_mask[None, None, None, :, :]
                ratio_o = np.where(valid, ratio_o, np.nan + 1j * np.nan)
                ratio_b = np.where(valid[None], ratio_b, np.nan + 1j * np.nan)
                path = OUT / "ratios" / f"ratio_{flow_tag}_{channel}_{flavor}_Nboot100_seed1115.npz"
                if replace or not path.exists():
                    atomic_savez(path, ratio_original=ratio_o, ratio_boot=ratio_b, ratio_valid_mask=valid, denominator_original=den_mean, denominator_boot=den_boot, boot_indices=indices, confs=np.asarray(CONFS), pz_values=np.asarray(PZ, np.int32), tseps=np.asarray(TSEPS, np.int32), delta_z_values=np.arange(DZ, dtype=np.int32), tau_a2=np.asarray(tau), flow_tag=np.asarray(TAU_SPECS[tau]), axes_ratio_original=np.asarray(["vdv","pz","delta_z","tsep","tau"]), axes_ratio_boot=np.asarray(["boot","vdv","pz","delta_z","tsep","tau"]), vdv_labels=np.asarray(VDV), channel=np.asarray(channel), flavor_combo=np.asarray(flavor), ensemble=np.asarray(ENSEMBLE), nconf=np.asarray(NCONF, np.int32), nboot=np.asarray(NBOOT, np.int32), seed=np.asarray(SEED, np.int64), estimator_id=np.asarray("post_ratio_single_positive_z_vdv_average_v1"), transversity_convention_status=np.asarray(TRANSVERSITY_STATUS), source_averaging=np.asarray("arithmetic_mean_t_source_before_configuration_bootstrap_and_ratio"), pipeline_version=np.asarray("quark_bare_ratio_20_v1"))
                if not skip_fit:
                    # Components requested by the established analysis are even-real and odd-imag.
                    original = np.stack((ratio_o[0].real, ratio_o[1].imag), axis=0)
                    boot = np.stack((ratio_b[:, 0].real, ratio_b[:, 1].imag), axis=1)
                    for method in ("sumratio", "plateau"):
                        fit = fit_one(original, boot, method)
                        fp = OUT / "fits" / f"{method}_{flow_tag}_{channel}_{flavor}_Nboot100_seed1115.npz"
                        if replace or not fp.exists():
                            atomic_savez(fp, **fit, component_labels=np.asarray(["even_real","odd_imag"]), pz_values=np.asarray(PZ, np.int32), delta_z_values=np.arange(DZ, dtype=np.int32), candidate_labels=np.asarray([x[0] for x in candidates()]), primary_candidate=np.asarray("T7-10_L2_R2"), channel=np.asarray(channel), flavor_combo=np.asarray(flavor), tau_a2=np.asarray(tau), flow_tag=np.asarray(TAU_SPECS[tau]), input_ratio=np.asarray(str(path)), boot_indices=indices, confs=np.asarray(CONFS), nboot=np.asarray(NBOOT, np.int32), nconf=np.asarray(NCONF, np.int32), seed=np.asarray(SEED, np.int64), fit_method=np.asarray(method), analysis_scope=np.asarray("connected_bare_unrenormalized_unmatched_single_positive_z_gradient_flow"), transversity_convention_status=np.asarray(TRANSVERSITY_STATUS), pipeline_version=np.asarray("quark_bare_fit_20_v1"))
        del c3, c2
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--repair-valid-masks", action="store_true")
    ap.add_argument("--taus", default=None, help="comma-separated flow times; default is all six")
    args = ap.parse_args()
    if args.repair_valid_masks:
        repaired = 0
        for path in sorted((OUT / "ratios").glob("*.npz")):
            with np.load(path, allow_pickle=False) as d:
                payload = {k: np.asarray(d[k]) for k in d.files}
            valid = payload["ratio_valid_mask"].copy()
            for it, t in enumerate(TSEPS):
                valid[..., it, t + 1:] = False
            payload["ratio_valid_mask"] = valid
            payload["ratio_original"] = np.where(valid, payload["ratio_original"], np.nan + 1j * np.nan)
            payload["ratio_boot"] = np.where(valid[None], payload["ratio_boot"], np.nan + 1j * np.nan)
            atomic_savez(path, **payload); repaired += 1
        print(f"repaired {repaired} ratio files")
        return 0
    taus = TAUS if args.taus is None else tuple(float(x) for x in args.taus.split(",") if x.strip())
    if args.audit_only:
        a = check_inputs(taus); print(json.dumps({"taus": taus, "missing": a["missing"], "missing_count": len(a["missing"])}, indent=2)); return 1 if a["missing"] else 0
    run(replace=args.replace, skip_fit=args.skip_fit, taus=taus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
