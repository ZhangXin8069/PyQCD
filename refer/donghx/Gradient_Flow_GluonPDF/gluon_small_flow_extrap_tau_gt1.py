#!/usr/bin/env python3
"""Linear small-flow-time extrapolation using only tau/a^2 > 1 points."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon")
PROD = BASE / "bare_fit_sumdiff_aic_v1/products"
OUT = Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_extrap_tau_gt1")
TAUS_ALL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8])
FIT_MASK = TAUS_ALL > 1.0
COMPONENTS = ["unpolarized_TH_minus_SH_even", "helicity_TH_odd", "helicity_TH_plus_SH_odd"]

def tag(t):
    return f"tau{t:.3f}".replace(".", "p")

def load():
    central, boots, errs, avail = [], [], [], []
    for tau in TAUS_ALL:
        path = PROD / f"bare_sumdiff_aic_{tag(tau)}_N231.npz"
        with np.load(path, allow_pickle=False) as d:
            central.append(np.asarray(d["M_filled"], float))
            boots.append(np.asarray(d["M_boot_filled"], float))
            errs.append(np.asarray(d["filled_statistical_error"], float))
            avail.append(np.asarray(d["estimate_available"], bool))
    return np.stack(central), np.stack(boots), np.stack(errs), np.stack(avail)

def main():
    central, boots, errs, avail = load()  # tau, component, p, z; boot,...
    x = TAUS_ALL
    nboot, ncomp, npz, nz = boots.shape[1:]
    bcoef = np.full((ncomp, npz, nz, nboot, 2), np.nan)
    for ic in range(ncomp):
        for ip in range(npz):
            for iz in range(nz):
                mask = FIT_MASK & avail[:, ic, ip, iz] & np.all(np.isfinite(boots[:, :, ic, ip, iz]), axis=1)
                if mask.sum() < 2:
                    continue
                xx = x[mask]
                pp = np.stack((np.ones_like(xx), xx), axis=1)
                ppinv = np.linalg.inv(pp.T @ pp) @ pp.T
                y = boots[mask, :, ic, ip, iz]
                bcoef[ic, ip, iz, :, :] = np.einsum("ab,bn->na", ppinv, y)
    m0 = bcoef[..., 0]
    slope = bcoef[..., 1]
    result = {
        "schema": "gluon_small_flow_linear_extrapolation_tau_gt1_v1",
        "flow_times_used_t_over_a2": TAUS_ALL[FIT_MASK],
        "model": "M(tau)=M0+c*tau",
        "selection": "tau/a^2 > 1.0",
        "components": COMPONENTS,
        "m0_boot": m0, "slope_boot": slope,
        "m0": np.nanmean(m0, axis=-1), "m0_error": np.nanstd(m0, axis=-1, ddof=1),
        "slope": np.nanmean(slope, axis=-1), "slope_error": np.nanstd(slope, axis=-1, ddof=1),
        "z_values": np.load(PROD / f"bare_sumdiff_aic_{tag(TAUS_ALL[0])}_N231.npz", allow_pickle=False)["z_values"],
        "pabs_values": np.load(PROD / f"bare_sumdiff_aic_{tag(TAUS_ALL[0])}_N231.npz", allow_pickle=False)["pabs_values"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "linear_extrap_tau_gt1.npz", **result)
    summary = {"schema": result["schema"], "flow_times_used_t_over_a2": TAUS_ALL[FIT_MASK].tolist(),
               "model": result["model"], "selection": result["selection"],
               "m0_shape": list(result["m0"].shape)}
    (OUT / "linear_extrap_tau_gt1.json").write_text(json.dumps(summary, indent=2) + "\n")

    iz = int(np.where(result["z_values"] == 2)[0][0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=False)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    # Small horizontal offsets keep Pz=3,4,5 error bars visually separable.
    offsets = np.asarray([-0.045, 0.0, 0.045])
    # Use one common vertical range for unpolarized and helicity panels.
    panel_limits = [(-1.5, 0.5), (-0.3, 0.3)]
    for group in ([0], [1, 2]):
        finite_values = []
        for ic in group:
            for ip in range(npz):
                ok = avail[:, ic, ip, iz] & np.isfinite(central[:, ic, ip, iz]) & np.isfinite(errs[:, ic, ip, iz])
                finite_values.extend((central[ok, ic, ip, iz] - errs[ok, ic, ip, iz]).tolist())
                finite_values.extend((central[ok, ic, ip, iz] + errs[ok, ic, ip, iz]).tolist())
        finite_values.extend(np.nanpercentile(m0[group, :, iz, :], [2, 98]).ravel().tolist())
        # Keep the requested physical comparison ranges fixed.  The scan
        # above remains useful for diagnosing out-of-range points but does
        # not alter the displayed axes.
    for ic, ax in enumerate(axes):
        for ip, pz in enumerate(result["pabs_values"]):
            y = central[:, ic, ip, iz]; e = errs[:, ic, ip, iz]
            ok = avail[:, ic, ip, iz] & np.isfinite(y) & np.isfinite(e)
            # Points outside the fit window are retained as open markers.
            low = ok & (x <= 1.0)
            high = ok & (x > 1.0)
            ax.errorbar(x[low] + offsets[ip], y[low], yerr=e[low], fmt="o", ms=4,
                        markerfacecolor="white", capsize=2, color=colors[ip],
                        alpha=0.38,
                        label=rf"$P_z={pz}$")
            ax.errorbar(x[high] + offsets[ip], y[high], yerr=e[high], fmt="o", ms=4,
                        capsize=2, color=colors[ip], alpha=0.95)
            coef = bcoef[ic, ip, iz]
            good = np.isfinite(coef[:, 0])
            if good.sum() < 2: continue
            grid = np.linspace(0, 4.0, 200)
            pred = coef[good, 0, None] + coef[good, 1, None] * grid[None, :]
            lo, hi = np.nanpercentile(pred, [16, 84], axis=0)
            mid = np.nanmedian(pred, axis=0)
            ax.plot(grid, mid, color=colors[ip], lw=1.8, alpha=1.0)
            ax.fill_between(grid, lo, hi, color=colors[ip], alpha=0.14)
            ax.errorbar([0 + offsets[ip]], [np.nanmean(coef[good, 0])], yerr=[np.nanstd(coef[good, 0], ddof=1)],
                        fmt="s", ms=4, capsize=2, color=colors[ip], alpha=1.0)
        ax.axvline(1.0, color="0.4", ls="--", lw=0.8)
        ax.axhline(0, color="0.5", ls=":", lw=0.7)
        ax.set_ylim(panel_limits[0] if ic == 0 else panel_limits[1])
        ax.set_xlabel(r"flow time $t_f/a^2$")
        ax.set_title(COMPONENTS[ic].replace("_", " "))
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"bare matrix element at $z/a=2$")
    fig.suptitle(r"Small-flow-time extrapolation using $t_f/a^2>1.0$; $M(t)=M_0+c,t$")
    fig.tight_layout()
    fig.savefig(OUT / "linear_extrap_tau_gt1_z2.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "linear_extrap_tau_gt1_z2.pdf", bbox_inches="tight")
    print(OUT / "linear_extrap_tau_gt1_z2.png")
    print(OUT / "linear_extrap_tau_gt1_z2.pdf")

if __name__ == "__main__": main()
