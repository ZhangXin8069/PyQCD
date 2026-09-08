#!/usr/bin/env python3
"""Plot seven-point tau>1 large-flow linear extrapolation diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT / "flow_time_tau_gt_0p1_v1"))
from analyze_tau_gt_0p1 import load_latest, sha256_file

LABELS = (
    r"unpolarized Re$[(T_U-S_U)_{\rm even}]$",
    r"helicity Im$[(T_H)_{\rm odd}]$",
    r"helicity Im$[(T_H+S_H)_{\rm odd}]$",
)
COLORS = ("#1f77b4", "#d95f02", "#238b45")
MARKERS = ("o", "s", "^")


def prediction_error(ib: np.ndarray, sb: np.ndarray, x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(ib) & np.isfinite(sb)
    return np.std(ib[finite, None] + sb[finite, None] * x[None, :], axis=0, ddof=1)


def make_figure(data, products, estimator_index: int, z: int, outdir: Path) -> list[Path]:
    iz = int(np.flatnonzero(data["z_values"] == z)[0])
    fit_taus = data["fit_taus"]
    all_taus = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8,
                           2.2, 2.6, 3.0, 3.4, 3.8))
    fi = np.asarray([int(np.flatnonzero(np.isclose(all_taus, t))[0]) for t in fit_taus])
    estimator = str(data["estimator_names"][estimator_index])
    Mkey = "M" if estimator_index == 0 else "M_filled"
    Ekey = "statistical_error" if estimator_index == 0 else "filled_statistical_error"
    Akey = "fit_accepted" if estimator_index == 0 else "estimate_available"
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.0), sharex=True)
    records = []
    for io in range(3):
        for ip, pz in enumerate(data["pabs_values"]):
            ax = axes[io, ip]
            available = products[Akey][fi, io, ip, iz]
            ax.errorbar(fit_taus[available], products[Mkey][fi, io, ip, iz][available],
                        yerr=products[Ekey][fi, io, ip, iz][available], fmt=MARKERS[ip],
                        linestyle="none", color="black", ms=4.2, capsize=2.2,
                        label="seven large-flow points")
            loc = (estimator_index, io, ip, iz)
            computed = bool(data["fit_computed"][loc])
            quality = bool(data["quality_pass"][loc])
            if computed:
                m0, slope = float(data["intercept"][loc]), float(data["slope_tau"][loc])
                error = float(data["intercept_error"][loc])
                x = np.linspace(0.0, 3.8, 260)
                band = prediction_error(data["intercept_boot"][estimator_index, :, io, ip, iz],
                                        data["slope_tau_boot"][estimator_index, :, io, ip, iz], x)
                color = COLORS[io]
                ax.fill_between(x, m0+slope*x-band, m0+slope*x+band,
                                color=color, alpha=0.16, linewidth=0)
                ax.plot(x, m0+slope*x, color=color, lw=1.35,
                        ls="-" if quality else "--")
                ax.errorbar([0], [m0], [error], fmt="D", linestyle="none",
                            color=color, ms=4.7, capsize=2.3)
                text = (rf"$M_0={m0:+.3g}\pm{error:.2g}$" + "\n" +
                        rf"$\chi^2/{int(data['dof'][loc])}={float(data['chi2'][loc]):.2f}$, " +
                        rf"$Q={float(data['q_value'][loc]):.2g}$" +
                        ("" if quality else "\nquality fail"))
            else:
                m0 = error = np.nan
                text = "fit unavailable\n" + str(data["status"][loc])
            ax.text(0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"})
            ax.axhline(0, color="0.55", lw=0.7, ls=":")
            ax.grid(alpha=0.25)
            if io == 0:
                ax.set_title(rf"$P_z={int(pz)}(2\pi/L)$")
            if ip == 0:
                ax.set_ylabel(LABELS[io], fontsize=9.5)
            records.append({"operator": str(data["operator_labels"][io]), "Pz": int(pz),
                            "z": z, "estimator": estimator, "intercept": m0,
                            "intercept_error": error, "fit_computed": computed,
                            "quality_pass": quality, "status": str(data["status"][loc])})
    for ax in axes[-1]:
        ax.set_xlabel(r"flow time $\tau/a^2$")
    geometry = bool(data["geometry_all_points_valid"][iz])
    note = r"all points satisfy $\sqrt{8\tau}<z$" if geometry else r"points violate $\sqrt{8\tau}<z$"
    branch = "accepted-only" if estimator_index == 0 else "filled/fallback diagnostic"
    fig.suptitle(rf"Seven-point $\tau/a^2>1$: 1.4--3.8, $z/a={z}$; {branch}; {note}" + "\n" +
                 r"unrestricted large-flow extrapolation to $\tau=0$ (not a controlled small-flow fit)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = outdir / f"tau_gt_1p0_fit_z{z}_{'accepted' if estimator_index == 0 else 'filled_diagnostic'}"
    outputs = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    stem.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=HERE / "results_v1/flow_time_tau_gt_1p0_v1.npz")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_v1/plots")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = load_latest(ROOT / "fit_latest_v1/results_v1")
    with np.load(args.result, allow_pickle=False) as data:
        outputs = []
        for ie in range(2):
            for z in (2, 6):
                outputs += make_figure(data, products, ie, z, args.output_dir)
    receipt = {"status": "complete", "result": str(args.result.resolve()),
               "result_sha256": sha256_file(args.result),
               "outputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in outputs]}
    (args.output_dir / "plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(outputs)} figure files")


if __name__ == "__main__":
    main()
