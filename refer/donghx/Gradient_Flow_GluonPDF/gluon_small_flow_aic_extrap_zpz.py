#!/usr/bin/env python3
"""AIC-weighted small-flow extrapolation with tau_min/a^2 = 1.0."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon")
PROD = BASE / "bare_fit_sumdiff_aic_v1/products"
OUT = Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_extrap_aic_tau_min1")
TAUS = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8])
COMPONENTS = ["unpolarized_TH_minus_SH_even", "helicity_TH_odd", "helicity_TH_plus_SH_odd"]

def tag(t): return f"tau{t:.3f}".replace(".", "p")

def main():
    central, boots, errs, avail = [], [], [], []
    for tau in TAUS:
        with np.load(PROD / f"bare_sumdiff_aic_{tag(tau)}_N231.npz", allow_pickle=False) as d:
            central.append(d["M_filled"]); boots.append(d["M_boot_filled"])
            errs.append(d["filled_statistical_error"]); avail.append(d["estimate_available"])
    central, boots, errs, avail = map(np.asarray, (central, boots, errs, avail))
    n_tau, ncomp, npz, nz = central.shape
    nboot = boots.shape[1]
    # All contiguous windows beginning at tau=1.0, with at least 3 flow points.
    windows = [(i, j) for i in range(n_tau) for j in range(i + 2, n_tau)]
    nw = len(windows)
    m0w = np.full((nw, ncomp, npz, nz, nboot), np.nan)
    aic = np.full((nw, ncomp, npz, nz), np.nan)
    weights = np.full_like(aic, np.nan)
    for iw, (i, j) in enumerate(windows):
        xx = TAUS[i:j+1]; X = np.stack((np.ones_like(xx), xx), axis=1)
        for ic in range(ncomp):
            for ip in range(npz):
                for iz in range(nz):
                    ok = avail[i:j+1, ic, ip, iz] & np.isfinite(central[i:j+1, ic, ip, iz]) & np.isfinite(errs[i:j+1, ic, ip, iz]) & (errs[i:j+1, ic, ip, iz] > 0)
                    if ok.sum() < 3: continue
                    x = xx[ok]; y = central[i:j+1, ic, ip, iz][ok]; s = errs[i:j+1, ic, ip, iz][ok]
                    W = np.diag(1.0 / s**2); XtWX = X[ok].T @ W @ X[ok]
                    beta = np.linalg.solve(XtWX, X[ok].T @ W @ y)
                    chi2 = float(np.sum(((y - X[ok] @ beta) / s) ** 2))
                    aic[iw, ic, ip, iz] = chi2 + 2 * 2
                    yy = boots[i:j+1, :, ic, ip, iz][ok]
                    # Same correlated bootstrap axis, weighted least squares.
                    m0w[iw, ic, ip, iz] = np.linalg.solve(XtWX, X[ok].T @ W @ yy)[0]
    for ic in range(ncomp):
        for ip in range(npz):
            for iz in range(nz):
                aa = aic[:, ic, ip, iz]; good = np.isfinite(aa)
                if np.any(good):
                    ww = np.zeros(nw); ww[good] = np.exp(-0.5 * (aa[good] - np.nanmin(aa[good])))
                    weights[:, ic, ip, iz] = ww / ww.sum()
    m0 = np.nansum(weights[..., None] * m0w, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    zvals = np.load(PROD / f"bare_sumdiff_aic_{tag(TAUS[0])}_N231.npz", allow_pickle=False)["z_values"]
    pvals = np.load(PROD / f"bare_sumdiff_aic_{tag(TAUS[0])}_N231.npz", allow_pickle=False)["pabs_values"]
    np.savez_compressed(OUT / "aic_extrapolated_m0.npz", schema="gluon_small_flow_aic_extrapolation_v1",
                        flow_times_used_t_over_a2=TAUS, tau_min=1.0, windows=np.asarray(windows),
                        window_aic=aic, window_weights=weights, m0_boot=m0,
                        m0=np.nanmean(m0, axis=-1), m0_error=np.nanstd(m0, axis=-1, ddof=1),
                        z_values=zvals, pabs_values=pvals, component_labels=np.asarray(COMPONENTS))
    (OUT / "aic_extrapolated_m0.json").write_text(json.dumps({"schema":"gluon_small_flow_aic_extrapolation_v1", "tau_min":1.0, "flow_times_used_t_over_a2":TAUS.tolist(), "windows":[[float(TAUS[i]),float(TAUS[j])] for i,j in windows], "model":"M(t)=M0+c*t; AIC weights exp(-DeltaAIC/2)"}, indent=2)+"\n")
    # Plot z*Pz; each momentum has its own x coordinates and slight offsets.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ic, ax in enumerate(axes):
        for ip, pz in enumerate(pvals):
            y = m0[ic, ip]; ok = np.isfinite(y).all(axis=0) if y.ndim > 1 else np.isfinite(y)
            if y.ndim == 1: y = y
            mean = np.nanmean(m0[ic, ip], axis=-1); err = np.nanstd(m0[ic, ip], axis=-1, ddof=1)
            good = np.isfinite(mean) & np.isfinite(err)
            ax.errorbar(zvals[good] * pz + (ip-1)*0.012, mean[good], yerr=err[good], fmt="o", ms=4, capsize=2, color=colors[ip], label=rf"$P_z={pz}$")
        ax.axhline(0, color="0.5", ls=":", lw=.7); ax.grid(alpha=.2)
        ax.set_xlabel(r"$zP_z$"); ax.set_title(COMPONENTS[ic].replace("_", " ")); ax.legend(frameon=False, fontsize=8)
        ax.set_ylim((-1.5, .5) if ic == 0 else (-.3, .3))
    axes[0].set_ylabel("AIC-weighted $t_f\to0$ bare matrix element")
    fig.suptitle(r"AIC-weighted small-flow extrapolation, $t_{\min}/a^2=1.0$; x-axis $zP_z$")
    fig.tight_layout(); fig.savefig(OUT / "aic_extrapolated_m0_vs_zpz.png", dpi=220, bbox_inches="tight"); fig.savefig(OUT / "aic_extrapolated_m0_vs_zpz.pdf", bbox_inches="tight"); plt.close(fig)
    print(OUT / "aic_extrapolated_m0_vs_zpz.png"); print(OUT / "aic_extrapolated_m0_vs_zpz.pdf")

if __name__ == "__main__": main()
