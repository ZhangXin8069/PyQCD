#!/usr/bin/env python3
"""Plot one-loop GF-to-MS conversion before/after for the fitted gluon OPE.

The input products are the authoritative ``bare_sumdiff_aic_*`` files.  They
contain the same 1500 configuration-bootstrap replicas for all three fitted
operators, so the helicity ``T_H-S_H`` cross-check is formed replica by
replica.  This is a diagnostic conversion: the Clover tree-level factor
``kappa_F`` is set to one and no quasi-to-light-cone kernel is applied.

The conversion is applied to the fitted matrix element after C3/C2 because,
for the present straight line along z, the one-loop factor is a real scalar
depending only on (tau,z) and is common to all components within a channel.
For a production matching result, applying the same factor to the per-conf
OPE before vacuum subtraction is still recommended and is implemented by
``gluon_flow_to_quasi.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .gluon_flow_to_quasi import GEV_FM, coefficients
except ImportError:  # direct invocation from this directory
    from gluon_flow_to_quasi import GEV_FM, coefficients


TAUS = np.asarray(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8],
    dtype=float,
)
LABELS = (
    r"unpolarized $(T_U-S_U)_{\rm even}$ [Re]",
    r"helicity $T_H$ (odd) [Im]",
    r"helicity $(T_H+S_H)_{\rm odd}$ [Im]",
    r"helicity $(T_H-S_H)_{\rm odd}$ [Im; check]",
)
COLORS = ("#1f77b4", "#d62728", "#2ca02c")
MARKERS = ("o", "s")


def tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def _load_one(path: Path):
    with np.load(path, allow_pickle=False) as d:
        central3 = np.asarray(d["M_filled"], dtype=float)
        boot3 = np.asarray(d["M_boot_filled"], dtype=float)
        available3 = np.asarray(d["estimate_available"], dtype=bool)
        pabs = np.asarray(d["pabs_values"], dtype=float)
        zvals = np.asarray(d["z_values"], dtype=float)
        tau = float(np.asarray(d["flow_tau_t_over_a2"]))
        if central3.shape[0] != 3 or boot3.shape[2] != 3:
            raise ValueError(f"unexpected component axis in {path}")

    # Products contain unpolarized (T_U-S_U), helicity T_H, and helicity
    # T_H+S_H.  Derive T_H-S_H with the shared bootstrap covariance.
    central = np.stack(
        (central3[0], central3[1], central3[2], 2.0 * central3[1] - central3[2]),
        axis=0,
    )
    boot = np.stack(
        (boot3[:, 0], boot3[:, 1], boot3[:, 2], 2.0 * boot3[:, 1] - boot3[:, 2]),
        axis=1,
    )
    available = np.stack(
        (
            available3[0],
            available3[1] & available3[2],
            available3[1] & available3[2],
            available3[1] & available3[2],
        ),
        axis=0,
    )
    # Filled fallback values remain in the archive for diagnostics, but are
    # not accepted in this plot unless the product marks the estimate usable.
    central = np.where(available, central, np.nan)
    boot = np.where(available[None, ...], boot, np.nan)
    return tau, central, boot, available, pabs, zvals


def load_products(root: Path, taus: np.ndarray):
    rows = []
    for tau in taus:
        path = root / f"bare_sumdiff_aic_{tag(float(tau))}_N231.npz"
        if path.exists():
            rows.append(_load_one(path))
    if not rows:
        raise FileNotFoundError(f"no bare-fit products under {root}")
    used = np.asarray([r[0] for r in rows], dtype=float)
    central = np.asarray([r[1] for r in rows], dtype=float)
    boot = np.asarray([r[2] for r in rows], dtype=float)
    available = np.asarray([r[3] for r in rows], dtype=bool)
    pabs, zvals = rows[0][4], rows[0][5]
    for row in rows[1:]:
        if not np.array_equal(row[4], pabs) or not np.array_equal(row[5], zvals):
            raise ValueError("pabs/z axes differ between flow times")
    return used, central, boot, available, pabs, zvals


def matching_factors(taus, zvals, a_fm, mu, alpha_s):
    """Return factors with shape (tau, channel, z)."""
    zgev = np.asarray(zvals, dtype=float) * a_fm * GEV_FM
    out = np.empty((len(taus), 4, len(zvals)), dtype=float)
    for it, tau in enumerate(taus):
        cpar, cperp, dm, _ = coefficients(float(tau), a_fm, mu, alpha_s)
        line = np.exp(-dm * zgev)
        out[it, 0] = line / (cperp * cperp)
        out[it, 1:] = line / (cpar * cperp)
    return out


def _weighted_line(x, y, s):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    if ok.sum() < 3:
        return None
    xx = x[ok]
    yy = y[ok]
    ss = s[ok]
    X = np.stack((np.ones_like(xx), xx), axis=1)
    WX = X / ss[:, None]
    wy = yy / ss
    try:
        beta = np.linalg.solve(WX.T @ WX, WX.T @ wy)
    except np.linalg.LinAlgError:
        return None
    chi2 = float(np.sum(((yy - X @ beta) / ss) ** 2))
    return beta, chi2 + 4.0, ok


def aic_extrap(taus, central, boot, tau_min):
    """Replica-wise AIC mixture of linear M(t)=M0+c*t fits.

    The AIC weights are fixed from central values/errors; each candidate's
    intercept is evaluated on every shared bootstrap replica.  This follows
    the local fit contract while documenting that flow-time covariance is not
    available in the archived products.
    """
    keep = np.asarray(taus) >= float(tau_min)
    xall = np.asarray(taus)[keep]
    yall = central[keep]
    # Use the bootstrap spread as the pointwise error for the fit weights.
    serr = np.nanstd(boot[keep], axis=1, ddof=1)
    ball = boot[keep]
    nt, nboot, nc, npz, nz = ball.shape
    windows = [(i, j) for i in range(nt) for j in range(i + 2, nt)]
    m0w = np.full((len(windows), nboot, nc, npz, nz), np.nan)
    aic = np.full((len(windows), nc, npz, nz), np.nan)
    for iw, (i, j) in enumerate(windows):
        x = xall[i : j + 1]
        for ic in range(nc):
            for ip in range(npz):
                for iz in range(nz):
                    fit = _weighted_line(
                        x,
                        yall[i : j + 1, ic, ip, iz],
                        serr[i : j + 1, ic, ip, iz],
                    )
                    if fit is None:
                        continue
                    _, aic[iw, ic, ip, iz], ok = fit
                    xx = x[ok]
                    ss = serr[i : j + 1, ic, ip, iz][ok]
                    X = np.stack((np.ones_like(xx), xx), axis=1)
                    WX = X / ss[:, None]
                    try:
                        inv = np.linalg.inv(WX.T @ WX)
                    except np.linalg.LinAlgError:
                        continue
                    ys = ball[i : j + 1, :, ic, ip, iz][ok]
                    good_rep = np.all(np.isfinite(ys), axis=0)
                    if np.any(good_rep):
                        # The weights are central-value weights; the same
                        # design matrix is used for every replica.
                        m0w[iw, good_rep, ic, ip, iz] = (
                            inv[0] @ (X.T @ (ys[:, good_rep] / ss[:, None]))
                        )
    weights = np.zeros_like(aic)
    good = np.isfinite(aic)
    if np.any(good):
        amin = np.full(aic.shape[1:], np.nan)
        amin[good.any(axis=0)] = np.nanmin(np.where(good, aic, np.inf), axis=0)[good.any(axis=0)]
        for iw in range(len(windows)):
            q = good[iw]
            weights[iw, q] = np.exp(-0.5 * (aic[iw, q] - amin[q]))
    denom = np.sum(weights, axis=0)
    weights = np.divide(weights, denom[None, ...], out=np.zeros_like(weights), where=denom[None, ...] > 0)
    m0_boot = np.nansum(weights[:, None, ...] * m0w, axis=0)
    m0_mean = np.nanmean(m0_boot, axis=0)
    m0_err = np.nanstd(m0_boot, axis=0, ddof=1)
    return m0_boot, m0_mean, m0_err, weights, np.asarray(windows, dtype=int), xall


def ioffe_time(zvals, pabs, Lz=32.0):
    """Dimensionless lattice Ioffe time nu=z Pz with Pz=2*pi*n/Lz."""
    return np.asarray(zvals)[:, None] * (2.0 * np.pi * np.asarray(pabs)[None, :] / Lz)


def _common_ylim(values, errors, margin=0.08):
    """A finite common y-range for the three helicity constructions."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    lo = np.nanmin(values - errors)
    hi = np.nanmax(values + errors)
    span = max(hi - lo, 1.0e-12)
    pad = margin * span
    return lo - pad, hi + pad


def _plot_before_after_z(out, central, errors, pabs, zvals, tau, zmax, mu, alpha_s, a_fm):
    mask = np.asarray(zvals) <= float(zmax)
    nu = ioffe_time(np.asarray(zvals)[mask], pabs)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), sharey=False)
    for ic, ax in enumerate(axes):
        for ip, pz in enumerate(pabs.astype(int)):
            x = nu[:, ip]
            dx = 0.012 * (ip - 1)
            ax.errorbar(
                x + dx - 0.010,
                central[ic, ip, mask],
                yerr=errors[ic, ip, mask],
                fmt=MARKERS[0],
                ms=4,
                capsize=2,
                lw=0.8,
                color=COLORS[ip],
                alpha=0.38,
                label=rf"bare $P_z={pz}$",
            )
            ax.errorbar(
                x + dx + 0.010,
                central[ic + 4, ip, mask],
                yerr=errors[ic + 4, ip, mask],
                fmt=MARKERS[1],
                ms=4,
                capsize=2,
                lw=0.9,
                color=COLORS[ip],
                label=rf"matched $P_z={pz}$",
            )
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
        ax.set_title(LABELS[ic], fontsize=10)
        ax.set_xlabel(r"$\nu=zP_z=2\pi(z/a)n_z/L_z$")
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_ylabel("bare matrix element (fit; one-loop conversion after)")
    hv = np.concatenate((central[1:4, :, mask], central[5:8, :, mask]), axis=0)
    he = np.concatenate((errors[1:4, :, mask], errors[5:8, :, mask]), axis=0)
    axes[1].set_ylim(_common_ylim(hv, he))
    for ax in axes[2:]:
        ax.set_ylim(axes[1].get_ylim())
    fig.suptitle(
        rf"GF matching before/after, $\tau/a^2={tau:g}$, $\mu={mu:g}$ GeV, "
        rf"$\alpha_s={alpha_s:g}$, $\kappa_F=1$; $z/a\leq{int(zmax)}$",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "matching_before_after_bare_matrix_tau0p500.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "matching_before_after_bare_matrix_tau0p500.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_flow_at_z(out, taus, before, after, before_err, after_err, pabs, zvals, zpick):
    iz = int(np.argmin(np.abs(np.asarray(zvals) - zpick)))
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.1), sharey=False)
    for ic, ax in enumerate(axes):
        for ip, pz in enumerate(pabs.astype(int)):
            ax.errorbar(taus, before[:, ic, ip, iz], yerr=before_err[:, ic, ip, iz], fmt="o", ms=4,
                        capsize=2, lw=0.8, color=COLORS[ip], alpha=0.4, label=rf"bare $P_z={pz}$")
            ax.errorbar(taus, after[:, ic, ip, iz], yerr=after_err[:, ic, ip, iz], fmt="s", ms=4,
                        capsize=2, lw=0.9, color=COLORS[ip], label=rf"matched $P_z={pz}$")
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
        ax.set_xlabel(r"$\tau/a^2$")
        ax.set_title(LABELS[ic], fontsize=10)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_ylabel(rf"matrix element at $z/a={zvals[iz]:g}$")
    hy = np.concatenate((before[:, 1:, :, iz], after[:, 1:, :, iz]), axis=1)
    he = np.concatenate((before_err[:, 1:, :, iz], after_err[:, 1:, :, iz]), axis=1)
    axes[1].set_ylim(_common_ylim(hy, he))
    for ax in axes[2:]:
        ax.set_ylim(axes[1].get_ylim())
    fig.suptitle("Flow-time dependence before/after one-loop GF conversion", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "matching_before_after_vs_flow_z2.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "matching_before_after_vs_flow_z2.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_extrap(out, m_before, e_before, m_after, e_after, pabs, zvals, tau_min):
    mask = np.asarray(zvals) <= 8
    nu = ioffe_time(np.asarray(zvals)[mask], pabs)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), sharey=False)
    for ic, ax in enumerate(axes):
        for ip, pz in enumerate(pabs.astype(int)):
            x = nu[:, ip] + 0.012 * (ip - 1)
            ax.errorbar(x - 0.010, m_before[ic, ip, mask], yerr=e_before[ic, ip, mask], fmt="o", ms=4,
                        capsize=2, lw=0.8, color=COLORS[ip], alpha=0.4, label=rf"bare $P_z={pz}$")
            ax.errorbar(x + 0.010, m_after[ic, ip, mask], yerr=e_after[ic, ip, mask], fmt="s", ms=4,
                        capsize=2, lw=0.9, color=COLORS[ip], label=rf"matched $P_z={pz}$")
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
        ax.set_xlabel(r"$\nu=zP_z$")
        ax.set_title(LABELS[ic], fontsize=10)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_ylabel(r"AIC-weighted $\tau\to0$ intercept")
    hv = np.concatenate((m_before[1:, :, mask], m_after[1:, :, mask]), axis=0)
    he = np.concatenate((e_before[1:, :, mask], e_after[1:, :, mask]), axis=0)
    axes[1].set_ylim(_common_ylim(hv, he))
    for ax in axes[2:]:
        ax.set_ylim(axes[1].get_ylim())
    fig.suptitle(rf"Small-flow diagnostic extrapolation, $\tau_{{\min}}/a^2={tau_min:g}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "matching_before_after_aic_extrap_vs_zpz.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "matching_before_after_aic_extrap_vs_zpz.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bare-root", type=Path,
                    default=Path("/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/bare_fit_sumdiff_aic_v1/products"))
    ap.add_argument("--out", type=Path, default=Path("matching/bare_matrix_comparison_v2"))
    ap.add_argument("--tau-fixed", type=float, default=0.5)
    ap.add_argument("--tau-min-extrap", type=float, default=1.0)
    ap.add_argument("--z-max", type=float, default=8.0)
    ap.add_argument("--z-flow", type=float, default=2.0)
    ap.add_argument("--a-fm", type=float, default=0.0775)
    ap.add_argument("--mu", type=float, default=2.0)
    ap.add_argument("--alpha-s", type=float, default=0.25)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    taus, before_b, boot_b, available, pabs, zvals = load_products(args.bare_root, TAUS)
    factors = matching_factors(taus, zvals, args.a_fm, args.mu, args.alpha_s)
    # Add a channel axis to factors and preserve the common bootstrap axis.
    after_b = boot_b * factors[:, None, :, None, :]
    before_e = np.nanstd(boot_b, axis=1, ddof=1)
    after_e = np.nanstd(after_b, axis=1, ddof=1)
    before_c = before_b
    # Keep the central fit value as the central point; bootstrap means are
    # used only to estimate the statistical error and replica-wise AIC fits.
    after_c = before_c * factors[:, :, None, :]

    # Locate the requested fixed flow time and plot all four available
    # operator constructions (including the TH-SH theory cross-check).
    it = int(np.argmin(np.abs(taus - args.tau_fixed)))
    fixed_before = before_c[it]
    fixed_after = after_c[it]
    _plot_before_after_z(args.out, np.concatenate((fixed_before, fixed_after)),
                         np.concatenate((before_e[it], after_e[it])), pabs, zvals,
                         float(taus[it]), args.z_max, args.mu, args.alpha_s, args.a_fm)
    _plot_flow_at_z(args.out, taus, before_c, after_c, before_e, after_e, pabs, zvals, args.z_flow)

    # The extrapolation is run on the same shared replicas.  It is deliberately
    # retained as a diagnostic because flow-time correlations and kappa_F are
    # not present in the archived fit products.
    mb, mb_mean, eb, wb, windows_b, used_b = aic_extrap(taus, before_c, boot_b, args.tau_min_extrap)
    ma, ma_mean, ea, wa, windows_a, used_a = aic_extrap(taus, after_c, after_b, args.tau_min_extrap)
    _plot_extrap(args.out, mb_mean, eb, ma_mean, ea, pabs, zvals, args.tau_min_extrap)

    np.savez_compressed(
        args.out / "matching_bare_matrix_data.npz",
        schema="gluon_flow_matching_bare_matrix_diagnostic_v2",
        flow_times=taus,
        before=before_c,
        before_boot=boot_b,
        before_error=before_e,
        after=after_c,
        after_boot=after_b,
        after_error=after_e,
        matching_factors=factors,
        aic_before_boot=mb,
        aic_before_mean=mb_mean,
        aic_before_error=eb,
        aic_after_boot=ma,
        aic_after_mean=ma_mean,
        aic_after_error=ea,
        aic_before_weights=wb,
        aic_after_weights=wa,
        aic_windows=windows_b,
        pabs_values=pabs,
        z_values=zvals,
        component_labels=np.asarray(LABELS),
        tau_fixed=float(taus[it]),
        tau_min_extrap=float(args.tau_min_extrap),
        z_max=float(args.z_max),
        z_flow=float(args.z_flow),
        a_fm=float(args.a_fm),
        mu_GeV=float(args.mu),
        alpha_s=float(args.alpha_s),
        kappa_F=1.0,
    )
    manifest = {
        "schema": "gluon_flow_matching_bare_matrix_diagnostic_v2",
        "status": "diagnostic_one_loop_GF_to_MSbar_factor_after_fit",
        "input_root": str(args.bare_root),
        "flow_times_used": taus.tolist(),
        "tau_fixed": float(taus[it]),
        "tau_min_extrap": float(args.tau_min_extrap),
        "z_max_fixed_plot": float(args.z_max),
        "z_flow_plot": float(args.z_flow),
        "nconf": 231,
        "nboot": int(boot_b.shape[1]),
        "a_fm": float(args.a_fm),
        "mu_GeV": float(args.mu),
        "alpha_s": float(args.alpha_s),
        "kappa_F": 1.0,
        "operator_labels": [
            "unpolarized (T_U-S_U)_even [Re]",
            "helicity T_H_odd [Im]",
            "helicity (T_H+S_H)_odd [Im]",
            "helicity (T_H-S_H)_odd [Im; convention cross-check]",
        ],
        "matching": {
            "c_parallel": "1 + O(alpha_s^2)",
            "c_perp": "1 + alpha_s*C_A/(4*pi)*log(2*mu^2*tau*exp(gamma_E))",
            "delta_m": "-alpha_s*C_A/(4*pi)*sqrt(2*pi/tau)",
            "unpolarized_factor": "exp(-delta_m*|z|)/c_perp^2",
            "helicity_factor": "exp(-delta_m*|z|)/(c_parallel*c_perp)",
        },
        "caveats": [
            "Clover tree-level kappa_F is set to 1 and must be determined before a physical result.",
            "No quasi/pseudo-ITD to light-cone PDF kernel or gluon-quark mixing is applied.",
            "Flow-time AIC fits use pointwise bootstrap errors and do not include cross-tau covariance.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(args.out / "matching_before_after_bare_matrix_tau0p500.pdf")


if __name__ == "__main__":
    main()
