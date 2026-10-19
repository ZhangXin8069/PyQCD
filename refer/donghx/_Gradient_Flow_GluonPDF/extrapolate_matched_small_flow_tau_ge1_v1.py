#!/usr/bin/env python3
"""AIC-weighted small-flow-time extrapolation of matched gluon matrix elements.

The input products are the Ncfg=406, Nboot=1000 matching products written by
``match_bare_all_operators_v1.py``.  The bootstrap replica axis is kept fixed
across flow times; no new resampling is performed here.  For every
(operator, |Pz|, z/a), contiguous flow-time windows with at least three usable
points are fitted to ``M(tau)=M0+c*tau``.  Candidate-window weights are based
on the central values and their bootstrap errors, and the same weights are
applied to every bootstrap replica.  The standard deviation of the resulting
1000 M0 replicas is the quoted statistical error.

This is a matched quasi-operator diagnostic.  It is not a light-cone PDF and
does not include gluon--quark mixing, LaMET matching, or a continuum limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_ROOT = Path(__file__).resolve().parent / "bare_all_operators_matching_v1" / "results_v1"
DEFAULT_OUT = Path(__file__).resolve().parent / "bare_all_operators_matching_v1" / "flow_extrap_tau_ge1"
LZ = 32.0
MIN_WINDOW_POINTS = 3


def _scalar(a):
    x = a.item() if np.asarray(a).ndim == 0 else a
    return x


def _tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def discover_products(root: Path, tau_min: float):
    products = []
    for path in sorted(root.glob("matched_bare_matrix_allops_tau*_N406_Nboot1000.npz")):
        if ".done" in path.name:
            continue
        with np.load(path, allow_pickle=False) as d:
            tau = float(_scalar(d["flow_tau_t_over_a2"]))
        if tau >= tau_min - 1e-10:
            products.append((tau, path))
    products.sort(key=lambda x: x[0])
    if not products:
        raise FileNotFoundError(f"no matching products with tau >= {tau_min} under {root}")
    return products


def load_products(root: Path, tau_min: float):
    products = discover_products(root, tau_min)
    taus = []
    central, errors, replicas, usable, fallback = [], [], [], [], []
    labels = pabs = zvals = None
    boot_indices = None
    nconf = nboot = None
    matching_meta = {}
    for tau, path in products:
        with np.load(path, allow_pickle=False) as d:
            if labels is None:
                labels = np.asarray(d["operator_labels"]).astype(str)
                pabs = np.asarray(d["pabs_values"], dtype=int)
                zvals = np.asarray(d["z_values"], dtype=int)
                nconf = int(_scalar(d["nconf"]))
                nboot = int(_scalar(d["nboot"]))
                boot_indices = np.asarray(d["bootstrap_indices"], dtype=np.int64)
                for key in ("a_fm", "matching_a_fm", "matching_mu_GeV", "matching_alpha_s",
                            "matching_kappa_F", "matching_delta_m_GeV", "matching_c_parallel_perp",
                            "matching_c_perp_perp"):
                    if key in d:
                        matching_meta[key] = float(_scalar(d[key]))
            else:
                if not np.array_equal(labels, np.asarray(d["operator_labels"]).astype(str)):
                    raise ValueError(f"operator labels differ in {path}")
                if not np.array_equal(pabs, np.asarray(d["pabs_values"], dtype=int)):
                    raise ValueError(f"momentum grid differs in {path}")
                if not np.array_equal(zvals, np.asarray(d["z_values"], dtype=int)):
                    raise ValueError(f"z grid differs in {path}")
                if not np.array_equal(boot_indices, np.asarray(d["bootstrap_indices"], dtype=np.int64)):
                    raise ValueError(f"bootstrap indices are not shared across flow times: {path}")
            taus.append(tau)
            central.append(np.asarray(d["M_filled"], dtype=float))
            errors.append(np.asarray(d["filled_statistical_error"], dtype=float))
            # Product axis is (boot, operator, |Pz|, z/a).
            replicas.append(np.asarray(d["M_boot_filled"], dtype=float))
            # A fit-accepted point is the strict input to the flow extrapolation.
            ok = (np.asarray(d["estimate_available"], dtype=bool)
                  & np.asarray(d["fit_accepted"], dtype=bool)
                  & ~np.asarray(d["central_fallback_used"], dtype=bool))
            usable.append(ok)
            fallback.append(np.asarray(d["central_fallback_used"], dtype=bool))
    c = np.asarray(central)
    e = np.asarray(errors)
    b = np.asarray(replicas).transpose(1, 0, 2, 3, 4)
    u = np.asarray(usable)
    f = np.asarray(fallback)
    if b.shape[0] != nboot or b.shape[2:] != c.shape[1:]:
        raise ValueError(f"unexpected bootstrap shape {b.shape}, central shape {c.shape}")
    return dict(flow_times=np.asarray(taus), central=c, error=e, replicas=b,
                usable=u, fallback=f, labels=labels, pabs=pabs, zvals=zvals,
                bootstrap_indices=boot_indices, nconf=nconf, nboot=nboot,
                products=[str(p) for _, p in products], matching_meta=matching_meta)


def _fit_linear(x, y, sigma, valid_mask=None):
    """Return intercept, slope, chi2, and the linear-fit row coefficients."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    if valid_mask is not None:
        ok &= np.asarray(valid_mask, dtype=bool)
    if np.count_nonzero(ok) < MIN_WINDOW_POINTS:
        return None
    xx = np.asarray(x[ok], float)
    yy = np.asarray(y[ok], float)
    ss = np.asarray(sigma[ok], float)
    X = np.column_stack((np.ones(xx.size), xx))
    A = X / ss[:, None]
    try:
        cov = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return None
    coef = cov @ A.T / ss[None, :]
    beta = coef @ yy
    resid = (yy - X @ beta) / ss
    chi2 = float(np.sum(resid * resid))
    return float(beta[0]), float(beta[1]), chi2, coef, np.flatnonzero(ok)


def extrapolate(data: dict, min_points: int = MIN_WINDOW_POINTS):
    tau = data["flow_times"]
    y = data["central"]
    e = data["error"]
    boot = data["replicas"]
    usable = data["usable"]
    nflow, nops, npz, nz = y.shape
    nboot = boot.shape[0]
    windows = [(i, j) for i in range(nflow) for j in range(i + min_points - 1, nflow)]
    nw = len(windows)
    m0 = np.full((nops, npz, nz), np.nan)
    slope = np.full_like(m0, np.nan)
    err = np.full_like(m0, np.nan)
    m0_boot = np.full((nboot, nops, npz, nz), np.nan)
    weights = np.zeros((nw, nops, npz, nz), dtype=float)
    cand_m0 = np.full_like(weights, np.nan)
    cand_slope = np.full_like(weights, np.nan)
    cand_aic = np.full_like(weights, np.nan)
    cand_npoint = np.zeros((nw, nops, npz, nz), dtype=np.int16)
    status = np.full((nops, npz, nz), "no_usable_window", dtype="U32")
    usable_count = usable.sum(axis=0).astype(np.int16)
    for io in range(nops):
        for ip in range(npz):
            for iz in range(nz):
                base = usable[:, io, ip, iz]
                finite_y = np.isfinite(y[:, io, ip, iz])
                finite_e = np.isfinite(e[:, io, ip, iz])
                valid = base & finite_y & finite_e
                if not np.any(valid):
                    continue
                # A zero-error, exactly flow-independent point is deterministic.
                bv = boot[:, :, io, ip, iz]
                vals = bv[:, valid]
                if vals.size and np.all(np.isfinite(vals)) and np.ptp(vals) == 0:
                    m0[io, ip, iz] = float(vals[0, 0])
                    slope[io, ip, iz] = 0.0
                    err[io, ip, iz] = 0.0
                    m0_boot[:, io, ip, iz] = vals[:, 0]
                    status[io, ip, iz] = "deterministic_flow_independent"
                    continue
                candidate_indices = []
                for iw, (i, j) in enumerate(windows):
                    idx = np.arange(i, j + 1)
                    fit = _fit_linear(tau[idx], y[idx, io, ip, iz], e[idx, io, ip, iz],
                                      valid[idx])
                    if fit is None:
                        continue
                    b0, bs, chi2, coef, local_ok = fit
                    used = idx[local_ok]
                    # Same convention used in the prior flow-time diagnostic:
                    # shorter windows pay an explicit omitted-point penalty.
                    n_omitted = nflow - used.size
                    aic = chi2 + 2.0 * 2.0 + 2.0 * n_omitted
                    cand_m0[iw, io, ip, iz] = b0
                    cand_slope[iw, io, ip, iz] = bs
                    cand_aic[iw, io, ip, iz] = aic
                    cand_npoint[iw, io, ip, iz] = used.size
                    candidate_indices.append((iw, used, coef))
                if not candidate_indices:
                    status[io, ip, iz] = "no_positive_error_window"
                    continue
                ai = np.asarray([cand_aic[iw, io, ip, iz] for iw, _, _ in candidate_indices])
                aa = np.min(ai)
                ww = np.exp(-0.5 * (ai - aa))
                ww /= np.sum(ww)
                for wval, (iw, _, _) in zip(ww, candidate_indices):
                    weights[iw, io, ip, iz] = wval
                # Fit each replica with the central window masks and errors.
                replica_candidates = np.full((nboot, len(candidate_indices)), np.nan)
                for ic, (iw, used, coef) in enumerate(candidate_indices):
                    vals_b = boot[:, used, io, ip, iz]
                    good_b = np.all(np.isfinite(vals_b), axis=1)
                    replica_candidates[good_b, ic] = coef[0] @ vals_b[good_b].T
                wvec = ww[None, :]
                good = np.isfinite(replica_candidates)
                num = np.nansum(np.where(good, replica_candidates, 0.0) * wvec, axis=1)
                den = np.sum(good * wvec, axis=1)
                mb = np.divide(num, den, out=np.full(nboot, np.nan), where=den > 0)
                m0_boot[:, io, ip, iz] = mb
                if np.any(np.isfinite(mb)):
                    m0[io, ip, iz] = float(np.nanmean(mb))
                    err[io, ip, iz] = float(np.nanstd(mb, ddof=1)) if np.count_nonzero(np.isfinite(mb)) > 1 else np.nan
                    slope[io, ip, iz] = float(np.nansum(ww * np.asarray([cand_slope[iw, io, ip, iz] for iw, _, _ in candidate_indices])))
                    status[io, ip, iz] = "aic_weighted_linear"
    return dict(m0=m0, slope=slope, error=err, m0_boot=m0_boot,
                weights=weights, candidate_m0=cand_m0, candidate_slope=cand_slope,
                candidate_aic=cand_aic, candidate_npoint=cand_npoint,
                windows=np.asarray(windows, dtype=np.int16), status=status,
                usable_count=usable_count)


def _operator_short(label: str) -> str:
    return label.replace("unpolarized_", "").replace("helicity_", "")


def plot_m0(data, fit, out: Path, sector: str):
    labels = data["labels"]
    pabs = data["pabs"]
    z = data["zvals"]
    mask = np.char.startswith(labels.astype(str), sector)
    indices = np.flatnonzero(mask)
    path = out / f"m0_vs_zpz_{sector}_tau_ge1.pdf"
    with PdfPages(path) as pdf:
        for io in indices:
            fig, axes = plt.subplots(1, len(pabs), figsize=(14, 4.1), squeeze=False)
            for ip, p in enumerate(pabs):
                ax = axes[0, ip]
                x = z * p * 2.0 * np.pi / LZ
                yy = fit["m0"][io, ip]
                ee = fit["error"][io, ip]
                finite = np.isfinite(yy) & np.isfinite(ee)
                ax.errorbar(x[finite], yy[finite], yerr=ee[finite], fmt="o", ms=3.6,
                            capsize=2.2, linestyle="none", color="#155a9c")
                ax.axhline(0, color="0.5", lw=0.7)
                ax.set_title(rf"$|P_z|={p}$")
                ax.set_xlabel(r"$zP_z$")
                ax.grid(alpha=0.22)
                ax.set_xlim(left=-0.12)
            axes[0, 0].set_ylabel(r"$M_0$ (matched quasi-operator diagnostic)")
            fig.suptitle(rf"{labels[io]}: AIC-weighted $\tau\to0$, $\tau/a^2\geq1.0$")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return path


def plot_flow_fits(data, fit, out: Path):
    tau = data["flow_times"]
    labels, pabs, z = data["labels"], data["pabs"], data["zvals"]
    # A few representative separations keep the diagnostic figure readable.
    zs = np.unique(np.clip(np.asarray([0, 2, 4, 8, 12, 16, 20, 24]), 0, len(z) - 1))
    path = out / "flow_fit_curves_tau_ge1.pdf"
    with PdfPages(path) as pdf:
        for io, label in enumerate(labels):
            fig, axes = plt.subplots(1, len(pabs), figsize=(15, 4.2), squeeze=False)
            for ip, p in enumerate(pabs):
                ax = axes[0, ip]
                for iz in zs:
                    y = data["central"][:, io, ip, iz]
                    e = data["error"][:, io, ip, iz]
                    accepted = data["usable"][:, io, ip, iz] & np.isfinite(y) & np.isfinite(e)
                    rejected = (~data["usable"][:, io, ip, iz]) & np.isfinite(y) & np.isfinite(e)
                    color = plt.cm.viridis(float(iz) / max(1, len(z) - 1))
                    if np.any(accepted):
                        ax.errorbar(tau[accepted], y[accepted], yerr=e[accepted], fmt="o", ms=2.8,
                                    color=color, alpha=0.9, linestyle="none")
                    if np.any(rejected):
                        ax.errorbar(tau[rejected], y[rejected], yerr=e[rejected], fmt="o", ms=2.8,
                                    mfc="none", mec=color, ecolor=color, alpha=0.35,
                                    linestyle="none")
                    # Draw the model-averaged line over the selected range.
                    b0, bs = fit["m0"][io, ip, iz], fit["slope"][io, ip, iz]
                    if np.isfinite(b0) and np.isfinite(bs):
                        tx = np.linspace(tau.min(), tau.max(), 100)
                        ax.plot(tx, b0 + bs * tx, color=color, lw=1.0,
                                label=rf"$z/a={z[iz]}$" if ip == 0 else None)
                ax.set_title(rf"$|P_z|={p}$")
                ax.set_xlabel(r"$\tau/a^2$")
                ax.grid(alpha=0.22)
                if ip == 0:
                    ax.set_ylabel(label)
                    ax.legend(fontsize=7, ncol=2, frameon=False)
            fig.suptitle(f"{label}: linear flow-time fits and AIC average")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tau-min", type=float, default=1.0)
    ap.add_argument("--min-points", type=int, default=MIN_WINDOW_POINTS)
    args = ap.parse_args()
    if args.min_points < 3:
        raise ValueError("at least three flow-time points are required for a linear window")
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_products(args.root, args.tau_min)
    fit = extrapolate(data, args.min_points)
    np.savez_compressed(
        args.out / "matched_small_flow_extrap_tau_ge1.npz",
        schema="gluon_matched_small_flow_aic_extrapolation_v1",
        flow_times=data["flow_times"], pabs_values=data["pabs"], z_values=data["zvals"],
        operator_labels=data["labels"], central=data["central"], error=data["error"],
        usable=data["usable"], central_fallback=data["fallback"],
        m0=fit["m0"], m0_error=fit["error"], slope=fit["slope"], m0_boot=fit["m0_boot"],
        window_weights=fit["weights"], candidate_m0=fit["candidate_m0"],
        candidate_slope=fit["candidate_slope"], candidate_aic=fit["candidate_aic"],
        candidate_npoint=fit["candidate_npoint"], windows=fit["windows"],
        fit_status=fit["status"], usable_count=fit["usable_count"],
        nconf=data["nconf"], nboot=data["nboot"], bootstrap_indices=data["bootstrap_indices"],
        **data["matching_meta"],
        tau_min=args.tau_min, min_window_points=args.min_points,
        fit_model="M(tau)=M0+c*tau", aic_definition="chi2+2*k+2*N_omitted, k=2",
        rejected_policy="estimate_available & fit_accepted & not central_fallback_used",
        physics_scope="matched finite-flow gluon quasi-operator diagnostic; no LaMET, mixing, or continuum limit",
    )
    plots = [plot_m0(data, fit, args.out, "unpolarized"),
             plot_m0(data, fit, args.out, "helicity"),
             plot_flow_fits(data, fit, args.out)]
    meta = {
        "schema": "gluon_matched_small_flow_aic_extrapolation_v1",
        "input_products": data["products"],
        "flow_times_used": data["flow_times"].tolist(),
        "tau_min": args.tau_min,
        "nconf": data["nconf"], "nboot": data["nboot"],
        "matching_meta": data["matching_meta"],
        "operator_labels": data["labels"].tolist(),
        "pabs_values": data["pabs"].tolist(), "z_values": data["zvals"].tolist(),
        "fit_model": "M(tau)=M0+c*tau",
        "aic_definition": "AIC=chi2+2*k+2*N_omitted, k=2",
        "window_min_points": args.min_points,
        "usable_policy": "estimate_available & fit_accepted & not central_fallback_used",
        "usable_count_min_max": [int(data["usable"].sum(axis=0).min()), int(data["usable"].sum(axis=0).max())],
        "fit_status_counts": {str(s): int(np.count_nonzero(fit["status"] == s)) for s in np.unique(fit["status"])},
        "plots": [str(p) for p in plots],
        "note": "Results are matched finite-flow quasi-operator diagnostics, not physical PDFs.",
    }
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
