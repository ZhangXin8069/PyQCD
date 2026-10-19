#!/usr/bin/env python3
"""Plot matched six-operator matrix elements for selected flow times.

The input is the fit-level Nconf=406, Nboot=1000 matching product.  A separate
PDF/PNG is written for every momentum and channel.  The three operator
constructions are separate rows, while the four selected flow times are shown
with small horizontal offsets.  No points at different z are connected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "matching" / "bare_all_operators_matching_v1" / "results_v1"
DEFAULT_OUT = ROOT / "matching" / "bare_all_operators_matching_v1" / (
    "plots_selected_tau_by_pz"
)
TAUS = np.asarray((0.5, 1.0, 1.4, 1.8), dtype=float)
PABS = np.asarray((3, 4, 5), dtype=int)
ZPLOT = np.arange(16, dtype=int)

CHANNELS = {
    "unpolarized": {
        "operators": (0, 1, 2),
        "labels": (
            r"$T_U$",
            r"$T_U+S_U$",
            r"$T_U-S_U$",
        ),
        "title": "Matched unpolarized gluon matrix elements",
        "ylim": (-1.5, 0.5),
    },
    "helicity": {
        "operators": (3, 4, 5),
        "labels": (
            r"$T_H$",
            r"$T_H+S_H$",
            r"$T_H-S_H$",
        ),
        "title": "Matched helicity gluon matrix elements",
        "ylim": (-1.0, 0.5),
    },
}
TAU_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
OP_MARKERS = ("o", "s", "^")
TAU_SHIFTS = np.asarray((-0.12, -0.04, 0.04, 0.12), dtype=float)


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def product_path(root: Path, tau: float) -> Path:
    return root / f"matched_bare_matrix_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def load_products(root: Path, taus: np.ndarray) -> dict[float, dict[str, np.ndarray]]:
    products = {}
    for tau in taus:
        path = product_path(root, float(tau))
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            required = (
                "M_filled",
                "filled_statistical_error",
                "estimate_available",
                "fit_accepted",
                "usable_cut_count",
                "central_fallback_used",
                "pabs_values",
                "z_values",
                "nconf",
                "nboot",
            )
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"{path}: missing {missing}")
            row = {key: np.asarray(data[key]) for key in required}
            row["matching_factors"] = np.asarray(data["matching_factors"])
        if row["M_filled"].shape != (6, 3, 25):
            raise ValueError(f"unexpected M_filled shape in {path}: {row['M_filled'].shape}")
        if tuple(row["pabs_values"].tolist()) != tuple(PABS):
            raise ValueError(f"unexpected momentum axis in {path}")
        if tuple(row["z_values"].tolist()) != tuple(range(25)):
            raise ValueError(f"unexpected z axis in {path}")
        if int(row["nconf"].item()) != 406 or int(row["nboot"].item()) != 1000:
            raise ValueError(f"unexpected statistics in {path}")
        products[float(tau)] = row
    return products


def draw_status(ax, x, values, errors, available, accepted, usable, fallback,
                color, marker, shift):
    """Draw one operator/flow-time series without joining z points."""
    good = available & ~fallback
    multi = good & accepted & (usable >= 2)
    single = good & accepted & (usable < 2)
    # Keep fallback points visible as diagnostic values, but distinguish them.
    for mask, face, alpha in ((multi, color, 0.92), (single, "white", 0.92)):
        if np.any(mask):
            ax.errorbar(
                x[mask] + shift,
                values[mask],
                yerr=errors[mask],
                fmt=marker,
                color=color,
                markerfacecolor=face,
                markeredgecolor=color,
                markersize=4.6,
                capsize=1.8,
                elinewidth=0.8,
                alpha=alpha,
                linestyle="none",
            )
    if np.any(fallback):
        ax.errorbar(
            x[fallback] + shift,
            values[fallback],
            yerr=errors[fallback],
            fmt="X",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=4.8,
            capsize=1.8,
            elinewidth=0.8,
            alpha=0.88,
            linestyle="none",
        )


def plot_channel_pz(out: Path, channel: str, pz: int, products: dict[float, dict[str, np.ndarray]],
                    taus: np.ndarray, mu: float, alpha_s: float, a_fm: float) -> dict:
    spec = CHANNELS[channel]
    ip = int(np.where(PABS == pz)[0][0])
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.2), sharex=True, squeeze=False)
    axes = axes[:, 0]
    counts = {"multi": 0, "single": 0, "fallback": 0}

    for row, (operator, label) in enumerate(zip(spec["operators"], spec["labels"])):
        ax = axes[row]
        for it, tau in enumerate(taus):
            data = products[float(tau)]
            mask = data["estimate_available"][operator, ip, ZPLOT]
            accepted = data["fit_accepted"][operator, ip, ZPLOT]
            usable = data["usable_cut_count"][operator, ip, ZPLOT]
            fallback = data["central_fallback_used"][operator, ip, ZPLOT]
            color = TAU_COLORS[it]
            draw_status(
                ax,
                ZPLOT,
                data["M_filled"][operator, ip, ZPLOT],
                data["filled_statistical_error"][operator, ip, ZPLOT],
                mask,
                accepted,
                usable,
                fallback,
                color,
                OP_MARKERS[row],
                TAU_SHIFTS[it],
            )
            good = mask & accepted & ~fallback
            counts["multi"] += int(np.count_nonzero(good & (usable >= 2)))
            counts["single"] += int(np.count_nonzero(good & (usable < 2)))
            counts["fallback"] += int(np.count_nonzero(fallback))
        ax.axhline(0.0, color="0.45", ls="--", lw=0.7)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlim(-0.5, 15.5)
        ax.set_xticks((0, 3, 6, 9, 12, 15))
        ax.grid(alpha=0.2, lw=0.55)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel(r"$z/a$", fontsize=11)

    handles = [
        Line2D([], [], marker=marker, color="black", markerfacecolor="black",
               linestyle="none", markersize=5, label=label)
        for marker, label in zip(OP_MARKERS, spec["labels"])
    ]
    handles += [
        Line2D([], [], marker="o", color=color, markerfacecolor=color,
               linestyle="none", markersize=5, label=rf"matched $\tau/a^2={tau:g}$")
        for color, tau in zip(TAU_COLORS, taus)
    ]
    handles += [
        Line2D([], [], marker="o", color="black", markerfacecolor="black",
               linestyle="none", markersize=4.5, label="accepted, >=2 cuts"),
        Line2D([], [], marker="o", color="black", markerfacecolor="white",
               linestyle="none", markersize=4.5, label="accepted, 1 cut"),
        Line2D([], [], marker="X", color="black", markerfacecolor="black",
               linestyle="none", markersize=5, label="central fallback"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        f"{spec['title']} | $P_z={pz}$ | GF$\\to\\overline{{\\rm MS}}$ quasi conversion\n"
        rf"$\tau/a^2=0.5,1.0,1.4,1.8$, $\mu={mu:g}\,{{\rm GeV}}$, "
        rf"$\alpha_s={alpha_s:g}$, $a={a_fm:g}\,{{\rm fm}}$, $N_{{\rm boot}}=1000$",
        y=1.055,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.91))
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{channel}_matched_tau0p5_1p0_1p4_1p8_Pz{pz}"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "channel": channel,
        "pz": int(pz),
        "pdf": str(stem.with_suffix(".pdf").resolve()),
        "png": str(stem.with_suffix(".png").resolve()),
        "operators": list(spec["labels"]),
        "flow_times": [float(tau) for tau in taus],
        "z_range": [0, 15],
        "xlim": [-0.5, 15.5],
        "connect_z_points": False,
        "status_counts": counts,
    }


def plot_channel_all_pz(out: Path, channel: str,
                        products: dict[float, dict[str, np.ndarray]],
                        taus: np.ndarray, mu: float, alpha_s: float,
                        a_fm: float) -> dict:
    """Write one three-column figure, with one column for each momentum."""
    spec = CHANNELS[channel]
    fig, axes = plt.subplots(
        3, len(PABS), figsize=(16.4, 9.6), sharex=True, squeeze=False
    )
    counts = {"multi": 0, "single": 0, "fallback": 0}
    for row, (operator, label) in enumerate(zip(spec["operators"], spec["labels"])):
        for ip, pz in enumerate(PABS):
            ax = axes[row, ip]
            for it, tau in enumerate(taus):
                data = products[float(tau)]
                mask = data["estimate_available"][operator, ip, ZPLOT]
                accepted = data["fit_accepted"][operator, ip, ZPLOT]
                usable = data["usable_cut_count"][operator, ip, ZPLOT]
                fallback = data["central_fallback_used"][operator, ip, ZPLOT]
                good = mask & accepted & ~fallback
                counts["multi"] += int(np.count_nonzero(good & (usable >= 2)))
                counts["single"] += int(np.count_nonzero(good & (usable < 2)))
                counts["fallback"] += int(np.count_nonzero(fallback))
                draw_status(
                    ax,
                    ZPLOT,
                    data["M_filled"][operator, ip, ZPLOT],
                    data["filled_statistical_error"][operator, ip, ZPLOT],
                    mask,
                    accepted,
                    usable,
                    fallback,
                    TAU_COLORS[it],
                    OP_MARKERS[row],
                    TAU_SHIFTS[it],
                )
            ax.axhline(0.0, color="0.45", ls="--", lw=0.7)
            ax.set_ylim(*spec["ylim"])
            ax.set_xlim(-0.5, 15.5)
            ax.set_xticks((0, 3, 6, 9, 12, 15))
            ax.grid(alpha=0.2, lw=0.55)
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.set_title(rf"$P_z={int(pz)}$", fontsize=11)
            if ip == 0:
                ax.set_ylabel(label, fontsize=11)
            if row == 2:
                ax.set_xlabel(r"$z/a$", fontsize=11)

    handles = [
        Line2D([], [], marker=marker, color="black", markerfacecolor="black",
               linestyle="none", markersize=5, label=label)
        for marker, label in zip(OP_MARKERS, spec["labels"])
    ]
    handles += [
        Line2D([], [], marker="o", color=color, markerfacecolor=color,
               linestyle="none", markersize=5, label=rf"matched $\tau/a^2={tau:g}$")
        for color, tau in zip(TAU_COLORS, taus)
    ]
    handles += [
        Line2D([], [], marker="o", color="black", markerfacecolor="black",
               linestyle="none", markersize=4.5, label="accepted, >=2 cuts"),
        Line2D([], [], marker="o", color="black", markerfacecolor="white",
               linestyle="none", markersize=4.5, label="accepted, 1 cut"),
        Line2D([], [], marker="X", color="black", markerfacecolor="black",
               linestyle="none", markersize=5, label="central fallback"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=8)
    fig.suptitle(
        f"{spec['title']} | GF$\\to\\overline{{\\rm MS}}$ quasi conversion\n"
        rf"$\tau/a^2=0.5,1.0,1.4,1.8$, $\mu={mu:g}\,{{\rm GeV}}$, "
        rf"$\alpha_s={alpha_s:g}$, $a={a_fm:g}\,{{\rm fm}}$, "
        rf"$N_{{\rm boot}}=1000$, y-range $[{spec['ylim'][0]:g},{spec['ylim'][1]:g}]$",
        y=1.045,
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.91))
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{channel}_matched_tau0p5_1p0_1p4_1p8_all_Pz"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "channel": channel,
        "pdf": str(stem.with_suffix(".pdf").resolve()),
        "png": str(stem.with_suffix(".png").resolve()),
        "pabs_values": PABS.tolist(),
        "operators": list(spec["labels"]),
        "flow_times": [float(tau) for tau in taus],
        "z_range": [0, 15],
        "xlim": [-0.5, 15.5],
        "ylim": list(spec["ylim"]),
        "connect_z_points": False,
        "status_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--a-fm", type=float, default=0.0775)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--alpha-s", type=float, default=0.25)
    parser.add_argument("--taus", default="0.5,1.0,1.4,1.8")
    args = parser.parse_args()
    taus = np.asarray([float(item.strip()) for item in args.taus.split(",") if item.strip()])
    if taus.size != 4 or np.unique(taus).size != 4:
        raise SystemExit("--taus must contain four distinct flow times")
    products = load_products(args.matched_root, taus)
    outputs = []
    # Summary figures: one PDF per channel, with Pz=3,4,5 in separate columns.
    for channel in CHANNELS:
        outputs.append(plot_channel_all_pz(args.output, channel, products, taus,
                                           args.mu, args.alpha_s, args.a_fm))
    for channel in CHANNELS:
        for pz in PABS:
            outputs.append(plot_channel_pz(args.output, channel, int(pz), products, taus,
                                           args.mu, args.alpha_s, args.a_fm))
    summary = {
        "schema": "gradient_flow_gluon_matched_selected_taus_by_pz_v1",
        "status": "complete",
        "matched_root": str(args.matched_root.resolve()),
        "output_root": str(args.output.resolve()),
        "flow_times": taus.tolist(),
        "pabs_values": PABS.tolist(),
        "nconf": 406,
        "nboot": 1000,
        "matching": {
            "mu_GeV": args.mu,
            "alpha_s": args.alpha_s,
            "a_fm": args.a_fm,
            "kappa_F": 1.0,
            "unpolarized": "exp(-delta_m_A*|z|)/c_perp_perp^2",
            "helicity": "exp(-delta_m_A*|z|)/(c_parallel_perp*c_perp_perp)",
        },
        "outputs": outputs,
        "light_cone_matching": "not applied",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
