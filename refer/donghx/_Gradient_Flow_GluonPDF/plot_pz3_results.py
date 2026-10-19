#!/usr/bin/env python3
"""Pz=3 bare-matrix and GF-matching comparison plots.

The source is the authoritative ``bare_sumdiff_aic_v1`` fit product.  The
first figure shows the fitted finite-flow bare matrix elements for every
available flow time without joining error bars by lines.  The second figure
compares the same Pz=3 points before and after the one-loop GF-to-MS factor at
the requested fixed flow time.  The factor is the scalar conversion documented
in ``gluon_gradient_flow_matching.md``; kappa_F is left at one.
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
    from .plot_matched_bare_matrix import (
        LABELS,
        TAUS,
        _common_ylim,
        ioffe_time,
        load_products,
        matching_factors,
        tag,
    )
except ImportError:
    from plot_matched_bare_matrix import (
        LABELS,
        TAUS,
        _common_ylim,
        ioffe_time,
        load_products,
        matching_factors,
        tag,
    )


def _save_bare_plot(out, taus, central, errors, pabs, zvals, ip, zmax):
    mask = np.asarray(zvals) <= float(zmax)
    z = np.asarray(zvals, dtype=float)[mask]
    # Reserve an explicit right margin for the flow-time colorbar.  Using
    # ``tight_layout`` together with a figure-level colorbar can otherwise
    # place the bar on top of the fourth panel title.
    fig, axes = plt.subplots(1, 4, figsize=(18.4, 4.5), sharey=False)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(np.min(taus)), float(np.max(taus)))
    for ic, ax in enumerate(axes):
        for it, tau in enumerate(taus):
            ax.errorbar(
                z,
                central[it, ic, ip, mask],
                yerr=errors[it, ic, ip, mask],
                fmt="o",
                ms=3.4,
                capsize=1.8,
                lw=0.7,
                color=cmap(norm(tau)),
                alpha=0.78,
            )
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
        ax.set_title(LABELS[ic], fontsize=10)
        ax.set_xlabel(r"$z/a$")
        ax.set_xlim(-0.5, float(zmax) + 0.5)
        ax.set_xticks(np.arange(0.0, float(zmax) + 0.1, 1.0))
    axes[0].set_ylabel(r"$P_z=3$ bare matrix element (fit)")
    hv = central[:, 1:4, ip, mask]
    he = errors[:, 1:4, ip, mask]
    axes[1].set_ylim(_common_ylim(hv, he))
    for ax in axes[2:]:
        ax.set_ylim(axes[1].get_ylim())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.suptitle(rf"Bare gluon matrix elements at $P_z=3$; $0\leq z/a\leq{int(zmax)}$", fontsize=12)
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.17, top=0.84, wspace=0.28)
    # Keep the colorbar in its own explicitly reserved axis.  This prevents
    # it from covering the fourth helicity title when the figure is saved.
    cax = fig.add_axes([0.918, 0.22, 0.012, 0.58])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"flow time $\tau/a^2$")
    # The ``*_vs_z`` names describe the new horizontal axis; retain the old
    # names as compatibility aliases for links produced by earlier runs.
    for stem in ("bare_matrix_Pz3_all_tau_vs_z", "bare_matrix_Pz3_vs_zpz"):
        fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _z_tag(zmax):
    return f"{float(zmax):g}".replace(".", "p")


def _save_bare_by_tau(out, taus, central, errors, pabs, zvals, ip, zmax):
    """Write one four-panel PDF (and PNG preview) for each flow time.

    The per-flow figures use the lattice separation ``z/a`` directly on the
    horizontal axis, with the requested inclusive range 0--15.  Helicity
    constructions share one y scale within each figure, while unpolarized is
    left on its own scale.
    """
    ztag = _z_tag(zmax)
    out = Path(out) / f"bare_by_tau_z0to{ztag}"
    out.mkdir(parents=True, exist_ok=True)
    mask = (np.asarray(zvals, dtype=float) >= 0.0) & (np.asarray(zvals, dtype=float) <= float(zmax))
    z = np.asarray(zvals, dtype=float)[mask]
    written = []
    for it, tau in enumerate(np.asarray(taus, dtype=float)):
        fig, axes = plt.subplots(1, 4, figsize=(18.4, 4.5), sharey=False)
        for ic, ax in enumerate(axes):
            ax.errorbar(
                z,
                central[it, ic, ip, mask],
                yerr=errors[it, ic, ip, mask],
                fmt="o",
                ms=4.2,
                capsize=2.0,
                lw=0.8,
                color="#2166ac" if ic == 0 else "#762a83",
                alpha=0.86,
            )
            ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
            ax.grid(alpha=0.2)
            ax.set_title(LABELS[ic], fontsize=10)
            ax.set_xlabel(r"$z/a$")
            ax.set_xlim(-0.5, float(zmax) + 0.5)
            ax.set_xticks(np.arange(0.0, float(zmax) + 0.1, 1.0))
        axes[0].set_ylabel(r"$P_z=3$ bare matrix element (fit)")
        hv = central[it, 1:4, ip, mask]
        he = errors[it, 1:4, ip, mask]
        axes[1].set_ylim(_common_ylim(hv, he))
        for ax in axes[2:]:
            ax.set_ylim(axes[1].get_ylim())
        fig.suptitle(
            rf"Bare gluon matrix elements at $P_z=3$, $\tau/a^2={tau:g}$; "
            rf"$0\leq z/a\leq{int(zmax)}$",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.84, wspace=0.28)
        stem = f"bare_matrix_Pz3_{tag(float(tau))}_z0to{ztag}"
        for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
            path = out / f"{stem}.{suffix}"
            fig.savefig(path, bbox_inches="tight", **kwargs)
            written.append(path)
        plt.close(fig)
    return written


def _save_matching_plot(out, central, errors, factors, pabs, zvals, ip, tau, zmax, mu, alpha_s):
    mask = np.asarray(zvals) <= float(zmax)
    nu = ioffe_time(np.asarray(zvals)[mask], pabs)[:, ip]
    before = central
    after = central * factors[:, None, :]
    before_e = errors
    after_e = errors * factors[:, None, :]
    fig, axes = plt.subplots(1, 4, figsize=(18.4, 4.5), sharey=False)
    for ic, ax in enumerate(axes):
        ax.errorbar(
            nu,
            before[ic, ip, mask],
            yerr=before_e[ic, ip, mask],
            fmt="o",
            ms=4.5,
            capsize=2,
            lw=0.8,
            color="#4477aa",
            alpha=0.48,
            label="GF bare",
        )
        ax.errorbar(
            nu,
            after[ic, ip, mask],
            yerr=after_e[ic, ip, mask],
            fmt="s",
            ms=4.5,
            capsize=2,
            lw=0.9,
            color="#cc3311",
            label="one-loop converted",
        )
        ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
        ax.set_title(LABELS[ic], fontsize=10)
        ax.set_xlabel(r"$\nu=zP_z=2\pi(z/a)n_z/L_z$")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"$P_z=3$ matrix element")
    hv = np.concatenate((before[1:4, ip, mask], after[1:4, ip, mask]), axis=0)
    he = np.concatenate((before_e[1:4, ip, mask], after_e[1:4, ip, mask]), axis=0)
    axes[1].set_ylim(_common_ylim(hv, he))
    for ax in axes[2:]:
        ax.set_ylim(axes[1].get_ylim())
    fig.suptitle(
        rf"$P_z=3$ GF matching before/after, $\tau/a^2={tau:g}$, "
        rf"$\mu={mu:g}$ GeV, $\alpha_s={alpha_s:g}$, $\kappa_F=1$; $z/a\leq{int(zmax)}$",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.84, wspace=0.28)
    fig.savefig(out / "matching_before_after_Pz3_tau0p500.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "matching_before_after_Pz3_tau0p500.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bare-root",
        type=Path,
        default=Path("/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/bare_fit_sumdiff_aic_v1/products"),
    )
    ap.add_argument("--out", type=Path, default=Path("matching/pz3_comparison_v1"))
    ap.add_argument("--pabs", type=int, default=3)
    ap.add_argument("--tau-fixed", type=float, default=0.5)
    ap.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="set both bare and matching z maxima (overrides the separate defaults)",
    )
    ap.add_argument(
        "--bare-z-max",
        type=float,
        default=15.0,
        help="maximum z/a for the per-flow bare plots (default: 15)",
    )
    ap.add_argument(
        "--matching-z-max",
        type=float,
        default=8.0,
        help="maximum z/a for the matching comparison (default: 8)",
    )
    ap.add_argument("--a-fm", type=float, default=0.0775)
    ap.add_argument("--mu", type=float, default=2.0)
    ap.add_argument("--alpha-s", type=float, default=0.25)
    args = ap.parse_args()
    if args.z_max is not None:
        args.bare_z_max = float(args.z_max)
        args.matching_z_max = float(args.z_max)
    if args.bare_z_max < 0 or args.matching_z_max < 0:
        raise ValueError("z maxima must be non-negative")
    args.out.mkdir(parents=True, exist_ok=True)
    taus, central, boot, available, pabs, zvals = load_products(args.bare_root, TAUS)
    if args.pabs not in pabs.astype(int).tolist():
        raise ValueError(f"Pz integer {args.pabs} is not in {pabs}")
    ip = int(np.flatnonzero(pabs.astype(int) == args.pabs)[0])
    errors = np.nanstd(boot, axis=1, ddof=1)
    factors = matching_factors(taus, zvals, args.a_fm, args.mu, args.alpha_s)
    it = int(np.argmin(np.abs(taus - args.tau_fixed)))
    # Keep the historical all-flow overview, now with the same z/a axis as
    # the requested per-flow figures, and write one PDF for every tau.
    _save_bare_plot(args.out, taus, central, errors, pabs, zvals, ip, args.bare_z_max)
    bare_by_tau_files = _save_bare_by_tau(
        args.out, taus, central, errors, pabs, zvals, ip, args.bare_z_max
    )
    _save_matching_plot(
        args.out,
        central[it],
        errors[it],
        factors[it],
        pabs,
        zvals,
        ip,
        float(taus[it]),
        args.matching_z_max,
        args.mu,
        args.alpha_s,
    )
    np.savez_compressed(
        args.out / "pz3_plot_data.npz",
        schema="gluon_flow_pz3_matching_plot_v1",
        flow_times=taus,
        pabs_values=pabs,
        z_values=zvals,
        pz_selected=int(args.pabs),
        bare_central=central[:, :, ip, :],
        bare_error=errors[:, :, ip, :],
        matching_factors=factors[:, :, :],
        matching_central=central[it, :, ip, :] * factors[it, :, :],
        matching_error=errors[it, :, ip, :] * factors[it, :, :],
        tau_fixed=float(taus[it]),
        z_max=float(args.bare_z_max),
        bare_z_max=float(args.bare_z_max),
        matching_z_max=float(args.matching_z_max),
        a_fm=float(args.a_fm),
        mu_GeV=float(args.mu),
        alpha_s=float(args.alpha_s),
        kappa_F=1.0,
        component_labels=np.asarray(LABELS),
    )
    manifest = {
        "schema": "gluon_flow_pz3_matching_plot_v1",
        "pabs_selected": int(args.pabs),
        "flow_times": taus.tolist(),
        "tau_fixed": float(taus[it]),
        "z_max": float(args.bare_z_max),
        "bare_z_max": float(args.bare_z_max),
        "matching_z_max": float(args.matching_z_max),
        "nconf": 231,
        "nboot": int(boot.shape[1]),
        "a_fm": float(args.a_fm),
        "mu_GeV": float(args.mu),
        "alpha_s": float(args.alpha_s),
        "kappa_F": 1.0,
        "matching_factors": {
            "unpolarized": "exp(-delta_m*|z|)/c_perp^2",
            "helicity": "exp(-delta_m*|z|)/(c_parallel*c_perp)",
        },
        "bare_by_tau_dir": str((args.out / f"bare_by_tau_z0to{_z_tag(args.bare_z_max)}").resolve()),
        "bare_by_tau_files": [str(path.resolve()) for path in bare_by_tau_files if path.suffix == ".pdf"],
        "all_flow_bare_plot": str((args.out / "bare_matrix_Pz3_all_tau_vs_z.pdf").resolve()),
        "note": "Per-flow bare PDFs use z/a=0..15 and xlim=(-0.5,15.5) with no connecting lines; helicity panels share y scale. Diagnostic one-loop conversion; no LaMET/pseudo-ITD kernel.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len([p for p in bare_by_tau_files if p.suffix == '.pdf'])} per-flow PDFs under "
          f"{(args.out / f'bare_by_tau_z0to{_z_tag(args.bare_z_max)}').resolve()}")


if __name__ == "__main__":
    main()
