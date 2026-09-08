#!/usr/bin/env python3
"""Plot L32x64 direct-Ratio bare-matrix-element fits.

The input is either a per-momentum fit product or the collected P=3,4,5,6
product made by ``collect_L32x64_ratio_fits.py``.  Formal quality-controlled
fits are shown as filled markers with their total uncertainty.  Every point
that fails those gates is retained as an explicitly labelled open-marker
diagnostic fallback, so each operator/direction has a value at every z.  The
z-axis is always displayed from -0.5 to 15.5 as requested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


FORMULA_LABELS = {
    "titi": r"$titi$",
    "titi_minus_ijij": r"$titi-2ijij$",
    "titi_plus_ijij": r"$titi+2ijij$",
}
COLORS = ("#1f77b4", "#d62728", "#2ca02c")
MARKERS = ("o", "s", "^")


def scalar(data, key):
    return np.asarray(data[key]).item()


def plot_one(data, channel: str, pindex: int, pabs: int, output_dir: Path) -> tuple[Path, Path]:
    formulas = [str(x) for x in np.asarray(data["formula_labels"]).tolist()]
    directions = [str(x) for x in np.asarray(data["direction_labels"]).tolist()]
    z = np.asarray(data["z_values"], dtype=int)
    M = np.asarray(data["M"], dtype=float)[:, :, pindex, :]
    error = np.asarray(data["total_error"], dtype=float)[:, :, pindex, :]
    valid = np.asarray(data["fit_valid"], dtype=bool)[:, :, pindex, :]

    # New products carry a complete diagnostic coverage array.  Keep a
    # backwards-compatible path for old files while fit jobs are refreshed.
    has_filled = all(
        key in data.files
        for key in ("M_filled", "filled_total_error", "estimate_available")
    )
    if has_filled:
        M_filled = np.asarray(data["M_filled"], dtype=float)[:, :, pindex, :]
        filled_error = np.asarray(data["filled_total_error"], dtype=float)[:, :, pindex, :]
        available = np.asarray(data["estimate_available"], dtype=bool)[:, :, pindex, :]
    else:
        M_filled = M.copy()
        filled_error = error.copy()
        available = valid.copy()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    if channel == "unpolarized":
        title_channel = "unpolarized"
    else:
        title_channel = "helicity"
    for direction_index, direction in enumerate(directions):
        ax = axes[direction_index]
        for formula_index, formula in enumerate(formulas):
            accepted = valid[formula_index, direction_index].copy()
            # Also guard against a malformed/partially written product.
            accepted &= np.isfinite(M[formula_index, direction_index])
            accepted &= np.isfinite(error[formula_index, direction_index])
            fallback = available[formula_index, direction_index].copy() & ~valid[
                formula_index, direction_index
            ]
            fallback &= np.isfinite(M_filled[formula_index, direction_index])
            fallback &= np.isfinite(filled_error[formula_index, direction_index])

            color = COLORS[formula_index]
            marker = MARKERS[formula_index]
            formula_label = FORMULA_LABELS.get(formula, formula)
            if np.any(accepted):
                ax.errorbar(
                    z[accepted],
                    M[formula_index, direction_index, accepted],
                    yerr=error[formula_index, direction_index, accepted],
                    color=color,
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    linestyle="-",
                    linewidth=1.0,
                    markersize=5.0,
                    capsize=2.5,
                    label=formula_label,
                )
            if np.any(fallback):
                # Open markers and a dashed connector make a fallback visible
                # without confusing it with a quality-controlled result.
                ax.errorbar(
                    z[fallback],
                    M_filled[formula_index, direction_index, fallback],
                    yerr=filled_error[formula_index, direction_index, fallback],
                    color=color,
                    marker=marker,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    linestyle="--",
                    linewidth=0.9,
                    markersize=5.5,
                    capsize=2.5,
                    label="_nolegend_",
                )
        ax.axhline(0.0, color="0.55", linewidth=0.7, linestyle="--")
        ax.set_title(rf"momentum direction ${direction}$")
        ax.set_xlim(-0.5, 15.5)
        ax.set_xticks(np.arange(0, 16, 2))
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.set_xlabel(r"$z/a$")
        accepted_count = int(np.count_nonzero(valid[:, direction_index]))
        available_count = int(np.count_nonzero(available[:, direction_index]))
        fallback_count = available_count - accepted_count
        ax.text(
            0.03,
            0.96,
            f"accepted: {accepted_count}\nfallback: {fallback_count}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.25",
        )
    axes[0].set_ylabel(r"bare matrix element $M$")
    # Use a stable figure-level legend even when one panel happens to contain
    # only fallback points for a formula.
    formula_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[i],
            marker=MARKERS[i],
            markerfacecolor=COLORS[i],
            markeredgecolor=COLORS[i],
            linewidth=1.0,
            markersize=5.0,
            label=FORMULA_LABELS.get(formula, formula),
        )
        for i, formula in enumerate(formulas)
    ]
    status_handles = [
        Line2D(
            [0], [0], color="0.25", marker="o", markerfacecolor="0.25",
            linestyle="-", markersize=5.0, label="accepted"
        ),
        Line2D(
            [0], [0], color="0.25", marker="o", markerfacecolor="none",
            linestyle="--", markersize=5.5, label="fallback"
        ),
    ]
    fig.legend(
        formula_handles + status_handles,
        [h.get_label() for h in formula_handles + status_handles],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        rf"L32x64 {title_channel} gluon bare matrix elements, $|P|={pabs}$",
        y=1.12,
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"barefit_L32x64_{channel}_P{pabs}_z"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        channel = str(scalar(data, "channel"))
        pabs = np.asarray(data["pabs_values"], dtype=int)
        if np.asarray(data["M"]).ndim != 4:
            raise ValueError("expected M axes (formula,direction,pabs,z)")
        for pindex, p in enumerate(pabs):
            paths = plot_one(data, channel, pindex, int(p), args.output_dir)
            print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
