#!/usr/bin/env python3
"""Plot tau>0.1 flow-time fits and window sensitivity without joined data points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_tau_gt_0p1 import ESTIMATORS, MODES, OPERATORS, TAUS, WINDOWS, sha256_file

LABELS = {
    OPERATORS[0]: r"unpolarized  Re$[(T_U-S_U)_{\rm even}]$",
    OPERATORS[1]: r"helicity  Im$[(T_H)_{\rm odd}]$",
    OPERATORS[2]: r"helicity  Im$[(T_H+S_H)_{\rm odd}]$",
}
COLORS = {"tau0p2_0p4": "#1f77b4", "tau0p2_0p6": "#d95f02", "tau0p2_3p8": "#238b45"}
SELECTED = tuple(COLORS)


def idx(values: np.ndarray, target) -> int:
    hits = np.flatnonzero(values == target)
    if len(hits) != 1:
        raise ValueError(f"Cannot uniquely find {target!r}")
    return int(hits[0])


def prediction_error(intercept_boot: np.ndarray, slope_boot: np.ndarray,
                     x: np.ndarray) -> np.ndarray:
    keep = np.isfinite(intercept_boot) & np.isfinite(slope_boot)
    if np.count_nonzero(keep) < 2:
        return np.full_like(x, np.nan)
    pred = intercept_boot[keep, None] + slope_boot[keep, None] * x[None, :]
    return np.std(pred, axis=0, ddof=1)


def plot_fit_comparison(data, raw, mode: str, z: int, pz: int, out: Path) -> list[Path]:
    modes = data["mode_names"].astype(str)
    estimators = data["estimator_names"].astype(str)
    windows = data["window_names"].astype(str)
    operators = data["operator_labels"].astype(str)
    pvals, zvals = data["pabs_values"], data["z_values"]
    im, ie = idx(modes, mode), idx(estimators, "accepted_only")
    ip, iz = idx(pvals, pz), idx(zvals, z)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 9.0), sharex=True)
    records = []
    for io, (operator, ax) in enumerate(zip(operators, axes)):
        accepted = raw["fit_accepted"][:, io, ip, iz]
        ax.errorbar(TAUS[accepted], raw["M"][accepted, io, ip, iz],
                    yerr=raw["statistical_error"][accepted, io, ip, iz],
                    fmt="o", linestyle="none", ms=3.7, capsize=2.0,
                    color="black", alpha=0.78, label="accepted finite-flow data")
        for window in SELECTED:
            iw = idx(windows, window)
            mask = data["fit_tau_mask"][im, iw, iz]
            computed = bool(data["fit_computed"][im, ie, iw, io, ip, iz])
            if not computed:
                continue
            intercept = float(data["intercept"][im, ie, iw, io, ip, iz])
            slope = float(data["slope_tau"][im, ie, iw, io, ip, iz])
            error = float(data["intercept_error"][im, ie, iw, io, ip, iz])
            quality = bool(data["quality_pass"][im, ie, iw, io, ip, iz])
            xmax = float(np.max(TAUS[mask]))
            x = np.linspace(0.0, xmax, 240)
            ib = data["intercept_boot"][im, ie, iw, :, io, ip, iz]
            sb = data["slope_tau_boot"][im, ie, iw, :, io, ip, iz]
            band = prediction_error(ib, sb, x)
            color = COLORS[window]
            linestyle = "-" if quality else "--"
            label = rf"$\tau={TAUS[mask][0]:g}$--${TAUS[mask][-1]:g}$, $M_0={intercept:+.3g}\pm{error:.2g}$"
            if not quality:
                label += " (quality fail)"
            ax.fill_between(x, intercept + slope*x-band, intercept + slope*x+band,
                            color=color, alpha=0.10, linewidth=0)
            ax.plot(x, intercept + slope*x, color=color, ls=linestyle, lw=1.25, label=label)
            ax.errorbar([0], [intercept], [error], fmt="D", linestyle="none",
                        color=color, ms=4.5, capsize=2.2)
            records.append({"operator": operator, "window": window,
                            "actual_fit_taus": TAUS[mask].tolist(), "intercept": intercept,
                            "intercept_error": error,
                            "Q": float(data["q_value"][im, ie, iw, io, ip, iz]),
                            "chi2_dof": float(data["chi2_dof"][im, ie, iw, io, ip, iz]),
                            "quality_pass": quality})
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_ylabel(LABELS[operator], fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.4, loc="best", framealpha=0.86)
    axes[-1].set_xlabel(r"flow time $\tau/a^2$")
    title_mode = "geometry guarded: $\sqrt{8\tau}<z$" if mode == MODES[0] else "unrestricted large-flow diagnostic"
    fig.suptitle(rf"$z/a={z}$, $P_z={pz}(2\pi/L)$; $\tau/a^2>0.1$; {title_mode}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    stem = out / f"flow_fit_compare_{mode}_z{z}_Pz{pz}"
    outputs = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return outputs


def plot_window_scan(data, mode: str, z: int, out: Path) -> list[Path]:
    modes, estimators = data["mode_names"].astype(str), data["estimator_names"].astype(str)
    operators = data["operator_labels"].astype(str)
    im, ie, iz = idx(modes, mode), idx(estimators, "accepted_only"), idx(data["z_values"], z)
    xmax = np.asarray([max(v) for v in WINDOWS.values()])
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.7), sharex=True)
    markers = ("o", "s", "^")
    colors = ("#1f77b4", "#d95f02", "#238b45")
    for io, (operator, ax) in enumerate(zip(operators, axes)):
        for ip, pz in enumerate(data["pabs_values"]):
            y = data["intercept"][im, ie, :, io, ip, iz]
            e = data["intercept_error"][im, ie, :, io, ip, iz]
            computed = data["fit_computed"][im, ie, :, io, ip, iz]
            quality = data["quality_pass"][im, ie, :, io, ip, iz]
            ax.errorbar(xmax[computed & quality], y[computed & quality], e[computed & quality],
                        fmt=markers[ip], linestyle="none", color=colors[ip], capsize=2.2,
                        label=rf"$P_z={int(pz)}$, quality pass")
            bad = computed & ~quality
            ax.errorbar(xmax[bad], y[bad], e[bad], fmt=markers[ip], linestyle="none",
                        mfc="none", color=colors[ip], capsize=2.2,
                        label=rf"$P_z={int(pz)}$, quality fail")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_ylabel(r"$M_0$  " + LABELS[operator], fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2, ncol=2, loc="best")
    axes[-1].set_xlabel(r"registered upper endpoint $\tau_{\max}/a^2$")
    title_mode = "geometry guarded" if mode == MODES[0] else "unrestricted large-flow diagnostic"
    fig.suptitle(rf"$\tau/a^2>0.1$ intercept stability, $z/a={z}$ ({title_mode})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.967))
    stem = out / f"intercept_window_scan_{mode}_z{z}"
    outputs = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=HERE / "results_v1/flow_time_tau_gt_0p1_v1.npz")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_v1/plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.result, allow_pickle=False) as data:
        # Raw arrays must cover every tau, not just the probe product.
        from analyze_tau_gt_0p1 import load_latest
        raw = load_latest(HERE.parent / "fit_latest_v1/results_v1")
        outputs = []
        for mode in MODES:
            for z in (2, 6):
                outputs += plot_window_scan(data, mode, z, args.output_dir)
                for pz in (3, 4, 5):
                    outputs += plot_fit_comparison(data, raw, mode, z, pz, args.output_dir)
    receipt = {"status": "complete", "result": str(args.result.resolve()),
               "result_sha256": sha256_file(args.result),
               "outputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in outputs]}
    (args.output_dir / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(outputs)} plot files to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
