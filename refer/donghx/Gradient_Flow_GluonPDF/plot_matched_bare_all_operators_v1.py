#!/usr/bin/env python3
"""Plot bare versus GF-matched Nboot=1000 matrix elements at one flow time."""

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
BARE_ROOT = ROOT / "fit_all_operators_aic_v1" / "results_v1"
MATCHED_ROOT = ROOT / "matching" / "bare_all_operators_matching_v1" / "results_v1"
DEFAULT_OUT = ROOT / "matching" / "bare_all_operators_matching_v1" / "plots_tau1p000"
PABS = np.asarray((3, 4, 5), dtype=int)
ZPLOT = np.arange(16, dtype=int)
OPERATOR_LABELS = (
    "unpolarized_TU_even",
    "unpolarized_TU_plus_SU_even",
    "unpolarized_TU_minus_SU_even",
    "helicity_TH_odd",
    "helicity_TH_plus_SH_odd",
    "helicity_TH_minus_SH_odd",
)
OPERATOR_TITLES = (
    r"$U$: $T_U$",
    r"$U$: $T_U+S_U$",
    r"$U$: $T_U-S_U$",
    r"$H$: $T_H$",
    r"$H$: $T_H+S_H$",
    r"$H$: $T_H-S_H$",
)
COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def product_path(root: Path, tau: float, matched: bool) -> Path:
    prefix = "matched_bare_matrix_allops" if matched else "bare_matrix_allops"
    return root / f"{prefix}_{tau_tag(tau)}_N406_Nboot1000.npz"


def load_pair(bare_path: Path, matched_path: Path) -> tuple[dict, dict]:
    with np.load(bare_path, allow_pickle=False) as bare_npz:
        bare = {key: np.asarray(bare_npz[key]) for key in bare_npz.files}
    with np.load(matched_path, allow_pickle=False) as matched_npz:
        matched = {key: np.asarray(matched_npz[key]) for key in matched_npz.files}
    required = (
        "M_filled",
        "filled_statistical_error",
        "estimate_available",
        "fit_accepted",
        "usable_cut_count",
        "central_fallback_used",
        "pabs_values",
        "z_values",
    )
    for key in required:
        if key not in bare or key not in matched:
            raise KeyError(f"missing {key} in bare/matched product")
    if tuple(bare["pabs_values"].tolist()) != tuple(PABS):
        raise ValueError("unexpected momentum axis")
    if tuple(bare["z_values"].tolist()) != tuple(range(25)):
        raise ValueError("unexpected z axis")
    if not np.array_equal(bare["estimate_available"], matched["estimate_available"]):
        raise ValueError("matching changed estimate-availability mask")
    if not np.array_equal(bare["fit_accepted"], matched["fit_accepted"]):
        raise ValueError("matching changed fit-accepted mask")
    return bare, matched


def _draw_status(
    ax,
    x: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    mask: np.ndarray,
    accepted: np.ndarray,
    usable: np.ndarray,
    fallback: np.ndarray,
    color: str,
    marker: str,
    alpha: float,
    x_shift: float,
) -> dict[str, int]:
    counts = {"multi": 0, "single": 0, "fallback": 0}
    categories = (
        (mask & accepted & (usable >= 2) & ~fallback, color, "multi"),
        (mask & accepted & (usable < 2) & ~fallback, "white", "single"),
        (mask & fallback, color, "fallback"),
    )
    for category, face, name in categories:
        if not np.any(category):
            continue
        ax.errorbar(
            x[category] + x_shift,
            values[category],
            yerr=errors[category],
            fmt=marker,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markersize=4.4 if marker == "o" else 4.8,
            capsize=1.7,
            elinewidth=0.75,
            alpha=alpha,
            linestyle="none",
        )
        counts[name] += int(np.count_nonzero(category))
    return counts


def plot_channel(
    output: Path,
    bare: dict,
    matched: dict,
    operator_indices: tuple[int, ...],
    title: str,
    tau: float,
    mu: float,
    alpha_s: float,
    a_fm: float,
) -> dict:
    values_b = bare["M_filled"]
    errors_b = bare["filled_statistical_error"]
    values_m = matched["M_filled"]
    errors_m = matched["filled_statistical_error"]
    available = bare["estimate_available"]
    accepted = bare["fit_accepted"]
    usable = bare["usable_cut_count"]
    fallback = bare["central_fallback_used"]
    nrows = len(operator_indices)
    fig, axes = plt.subplots(
        nrows,
        len(PABS),
        figsize=(15.6, max(4.8, 4.15 * nrows)),
        sharex=True,
        squeeze=False,
    )
    counts = {"multi": 0, "single": 0, "fallback": 0}
    for row, operator in enumerate(operator_indices):
        for ip, momentum in enumerate(PABS):
            ax = axes[row, ip]
            mask = available[operator, ip, ZPLOT]
            status_b = _draw_status(
                ax,
                ZPLOT,
                values_b[operator, ip, ZPLOT],
                errors_b[operator, ip, ZPLOT],
                mask,
                accepted[operator, ip, ZPLOT],
                usable[operator, ip, ZPLOT],
                fallback[operator, ip, ZPLOT],
                COLORS[ip],
                "o",
                0.42,
                -0.075,
            )
            status_m = _draw_status(
                ax,
                ZPLOT,
                values_m[operator, ip, ZPLOT],
                errors_m[operator, ip, ZPLOT],
                mask,
                accepted[operator, ip, ZPLOT],
                usable[operator, ip, ZPLOT],
                fallback[operator, ip, ZPLOT],
                COLORS[ip],
                "s",
                0.95,
                0.075,
            )
            for key in counts:
                counts[key] += status_b[key]
                counts[key] += status_m[key]
            ax.axhline(0.0, color="0.45", ls="--", lw=0.7)
            ax.set_xlim(-0.5, 15.5)
            ax.set_xticks((0, 3, 6, 9, 12, 15))
            ax.grid(alpha=0.2, lw=0.55)
            ax.set_title(rf"$P_z={momentum}\,(2\pi/L)$", fontsize=10)
            if ip == 0:
                ax.set_ylabel(OPERATOR_TITLES[operator] + " / matched,bare M", fontsize=10)
            if row == nrows - 1:
                ax.set_xlabel(r"$z/a$", fontsize=10)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            markerfacecolor="black",
            linestyle="none",
            markersize=5,
            alpha=0.45,
            label="bare GF",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="black",
            markerfacecolor="black",
            linestyle="none",
            markersize=5,
            label="GF matched",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            markerfacecolor="black",
            linestyle="none",
            markersize=4.5,
            label="accepted, >=2 cuts",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            markerfacecolor="white",
            linestyle="none",
            markersize=4.5,
            label="accepted, 1 cut",
        ),
        Line2D(
            [],
            [],
            marker="X",
            color="black",
            markerfacecolor="black",
            linestyle="none",
            markersize=5,
            label="central fallback",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        title
        + " | "
        + rf"GF$\to\overline{{\mathrm{{MS}}}}$ quasi conversion, "
        + rf"$\tau/a^2={tau:g}$, $\mu={mu:g}\,\mathrm{{GeV}}$, "
        + rf"$\alpha_s={alpha_s:g}$, $a={a_fm:g}\,\mathrm{{fm}}$, "
        + r"$\kappa_F=1$, $N_{\rm boot}=1000$",
        y=1.045,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "pdf": str(pdf.resolve()),
        "png": str(png.resolve()),
        "operator_indices": list(operator_indices),
        "z_range": [0, 15],
        "xlim": [-0.5, 15.5],
        "connect_z_points": False,
        "status_counts_before_plus_after": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--bare-root", type=Path, default=BARE_ROOT)
    parser.add_argument("--matched-root", type=Path, default=MATCHED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--a-fm", type=float, default=0.0775)
    parser.add_argument("--mu", type=float, default=2.0)
    parser.add_argument("--alpha-s", type=float, default=0.25)
    args = parser.parse_args()
    bare_path = product_path(args.bare_root, args.tau, matched=False)
    matched_path = product_path(args.matched_root, args.tau, matched=True)
    if not bare_path.is_file() or not matched_path.is_file():
        raise FileNotFoundError(f"missing bare/matched input: {bare_path}, {matched_path}")
    bare, matched = load_pair(bare_path, matched_path)
    outputs = []
    outputs.append(
        plot_channel(
            args.output / "unpolarized_matching_before_after_tau1p000",
            bare,
            matched,
            (2,),
            "Unpolarized physical candidate $(T_U-S_U)_{\\rm even}$",
            args.tau,
            args.mu,
            args.alpha_s,
            args.a_fm,
        )
    )
    outputs.append(
        plot_channel(
            args.output / "helicity_matching_before_after_tau1p000",
            bare,
            matched,
            (5,),
            "Helicity physical candidate $(T_H-S_H)_{\\rm odd}$",
            args.tau,
            args.mu,
            args.alpha_s,
            args.a_fm,
        )
    )
    outputs.append(
        plot_channel(
            args.output / "all_operators_matching_before_after_tau1p000",
            bare,
            matched,
            tuple(range(6)),
            "All six flowed-gluon operators",
            args.tau,
            args.mu,
            args.alpha_s,
            args.a_fm,
        )
    )
    summary = {
        "schema": "gradient_flow_gluon_bare_matching_plot_v1",
        "status": "complete",
        "tau_t_over_a2": args.tau,
        "bare_input": str(bare_path.resolve()),
        "matched_input": str(matched_path.resolve()),
        "nconf": int(np.asarray(bare["nconf"]).item()),
        "nboot": int(np.asarray(bare["nboot"]).item()),
        "pabs_values": PABS.tolist(),
        "z_plotted": [0, 15],
        "outputs": outputs,
        "matching": {
            "mu_GeV": args.mu,
            "alpha_s": args.alpha_s,
            "a_fm": args.a_fm,
            "kappa_F": 1.0,
            "unpolarized": "exp(-delta_m_A*|z|)/c_perp_perp^2",
            "helicity": "exp(-delta_m_A*|z|)/(c_parallel_perp*c_perp_perp)",
        },
        "light_cone_matching": "not applied",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
