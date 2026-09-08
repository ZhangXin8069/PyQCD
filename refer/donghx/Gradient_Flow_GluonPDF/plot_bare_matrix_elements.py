#!/usr/bin/env python3
"""Plot fitted flowed-gluon bare matrix elements versus positive z.

The plotted value is ``M_filled`` from the complete diagnostic product.  Its
markers retain the fit-bare status contract:

* filled circle: accepted fit with at least two usable cuts;
* open circle: accepted fit with only one usable cut;
* X: central fallback (minimum-chi2 or zero-dof diagnostic).

All figures use z/a=0..15 and explicitly set xlim=(-0.5, 15.5).  Flow-time
series are given small, symmetric x offsets at each z value so nearby
markers remain readable.  The points are deliberately not connected by
lines; each z value is an independent fitted matrix element.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIT_DIR_DEFAULT = Path(__file__).resolve().parent / "results_v1"
PLOT_DIR_DEFAULT = Path(__file__).resolve().parent / "plots_v1"
TAUS = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8), dtype=float)
OPERATOR_LABELS = (
    "unpolarized_TU_even",
    "unpolarized_TU_plus_SU_even",
    "unpolarized_TU_minus_SU_even",
    "helicity_TH_odd",
    "helicity_TH_plus_SH_odd",
    "helicity_TH_minus_SH_odd",
)
OPERATOR_TITLES = (
    r"$U$: $T_U$ (no $S_U$)",
    r"$U$: $T_U+S_U$",
    r"$U$: $T_U-S_U$",
    r"$H$: $T_H$ (no $S_H$)",
    r"$H$: $T_H+S_H$",
    r"$H$: $T_H-S_H$",
)
PABS = np.asarray((3, 4, 5), dtype=int)
CUTS = np.asarray((1, 2, 3), dtype=int)
Z = np.arange(25, dtype=int)
FLOW_X_OFFSET_MAX = 0.24


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def fit_path(fit_dir: Path, tau: float) -> Path:
    return fit_dir / f"bare_matrix_allops_{tau_tag(tau)}_N406_Nboot1000.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _receipt_path(path: Path) -> Path:
    for candidate in (Path(str(path) + ".done.json"), path.with_suffix(".done.json")):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing fit receipt: {path}")


def load_fits(
    fit_dir: Path, selected_taus: np.ndarray | None = None
) -> dict[str, np.ndarray | list]:
    taus = TAUS if selected_taus is None else np.asarray(selected_taus, dtype=float)
    if taus.ndim != 1 or taus.size == 0:
        raise ValueError("selected flow times must be a non-empty one-dimensional list")
    if np.any(~np.isin(taus, TAUS)):
        raise ValueError(f"requested flow times are unavailable; choose from {TAUS.tolist()}")
    if np.unique(taus).size != taus.size:
        raise ValueError("selected flow times must be unique")
    fields = {
        "M_filled": [],
        "filled_statistical_error": [],
        "estimate_available": [],
        "fit_accepted": [],
        "usable_cut_count": [],
        "central_fallback_used": [],
        "central_fallback_status": [],
        "replica_fallback_fraction": [],
    }
    records = []
    reference_confs = None
    reference_indices = None
    reference_labels = None
    for tau in taus:
        path = fit_path(fit_dir, float(tau))
        receipt_path = _receipt_path(path)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError(f"Fit receipt is incomplete: {receipt_path}")
        if receipt.get("output_sha256") != sha256_file(path):
            raise ValueError(f"Fit hash disagrees with receipt: {path}")
        with np.load(path, allow_pickle=False) as data:
            labels = tuple(np.asarray(data["operator_labels"]).astype(str).tolist())
            if labels != OPERATOR_LABELS:
                raise ValueError(f"Operator mismatch: {path}")
            if tuple(np.asarray(data["pabs_values"], dtype=int).tolist()) != tuple(PABS):
                raise ValueError(f"Momentum mismatch: {path}")
            if tuple(np.asarray(data["z_values"], dtype=int).tolist()) != tuple(Z):
                raise ValueError(f"z mismatch: {path}")
            if tuple(np.asarray(data["registered_cuts"], dtype=int).tolist()) != tuple(CUTS):
                raise ValueError(f"Cut mismatch: {path}")
            if int(np.asarray(data["nboot"]).item()) != 1000:
                raise ValueError(f"Nboot mismatch: {path}")
            confs = np.asarray(data["confs"]).astype(str)
            indices = np.asarray(data["bootstrap_indices"], dtype=np.int64)
            if reference_confs is None:
                reference_confs, reference_indices, reference_labels = confs, indices, labels
            elif not np.array_equal(reference_confs, confs) or not np.array_equal(
                reference_indices, indices
            ):
                raise ValueError("Flow-time fit products do not share conf/bootstrap axes")
            for key in fields:
                fields[key].append(np.asarray(data[key]))
            records.append(
                {
                    "tau_t_over_a2": float(np.asarray(data["flow_tau_t_over_a2"]).item()),
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
    result = {key: np.stack(value, axis=0) for key, value in fields.items()}
    result["taus"] = taus
    result["operator_labels"] = np.asarray(reference_labels)
    result["records"] = records
    return result


def _plot_axes(
    data: dict[str, np.ndarray | list],
    operator_indices: tuple[int, ...],
    output_stem: Path,
    title: str,
    dpi: int,
) -> dict:
    values = np.asarray(data["M_filled"])
    errors = np.asarray(data["filled_statistical_error"])
    available = np.asarray(data["estimate_available"], dtype=bool)
    accepted = np.asarray(data["fit_accepted"], dtype=bool)
    usable_cuts = np.asarray(data["usable_cut_count"], dtype=int)
    fallback = np.asarray(data["central_fallback_used"], dtype=bool)
    taus = np.asarray(data["taus"], dtype=float)
    z_plot = np.arange(16, dtype=int)
    nrows = len(operator_indices)
    ncols = len(PABS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.6, max(4.7, 3.65 * nrows)),
        sharex=True,
        squeeze=False,
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(taus)))
    flow_offsets = np.linspace(-FLOW_X_OFFSET_MAX, FLOW_X_OFFSET_MAX, len(taus))
    summary = {"accepted_multi_cut": 0, "accepted_single_cut": 0, "fallback": 0, "unavailable": 0}
    for row, operator in enumerate(operator_indices):
        for ip, momentum in enumerate(PABS):
            ax = axes[row, ip]
            for itau, tau in enumerate(taus):
                mask = available[itau, operator, ip, z_plot]
                if not np.any(mask):
                    summary["unavailable"] += len(z_plot)
                    continue
                multi = mask & accepted[itau, operator, ip, z_plot] & (
                    usable_cuts[itau, operator, ip, z_plot] >= 2
                )
                single = mask & accepted[itau, operator, ip, z_plot] & (
                    usable_cuts[itau, operator, ip, z_plot] < 2
                )
                fb = mask & fallback[itau, operator, ip, z_plot]
                for category, marker, face, key in (
                    (multi, "o", colors[itau], "accepted_multi_cut"),
                    (single, "o", "white", "accepted_single_cut"),
                    (fb, "X", colors[itau], "fallback"),
                ):
                    if not np.any(category):
                        continue
                    xx = z_plot[category] + flow_offsets[itau]
                    yy = values[itau, operator, ip, z_plot][category]
                    ee = errors[itau, operator, ip, z_plot][category]
                    ax.errorbar(
                        xx,
                        yy,
                        yerr=ee,
                        fmt=marker,
                        color=colors[itau],
                        markerfacecolor=face,
                        markeredgecolor=colors[itau],
                        markersize=4.1 if marker == "o" else 4.8,
                        capsize=1.6,
                        elinewidth=0.75,
                        linestyle="none",
                    )
                    summary[key] += int(np.count_nonzero(category))
            ax.axhline(0.0, color="0.45", ls="--", lw=0.7)
            ax.set_xlim(-0.5, 15.5)
            ax.set_xticks((0, 3, 6, 9, 12, 15))
            ax.grid(alpha=0.22, lw=0.55)
            ax.set_title(rf"$P_z={momentum}\,(2\pi/L)$", fontsize=10)
            if ip == 0:
                ax.set_ylabel(OPERATOR_TITLES[operator] + "\nbare matrix element", fontsize=10)
            if row == nrows - 1:
                ax.set_xlabel(r"$z/a$", fontsize=10)
    flow_handles = [
        Line2D(
            [], [], marker="o", color=colors[i], markerfacecolor=colors[i],
            markeredgecolor=colors[i], linestyle="none", markersize=4.2,
            label=rf"$t_f/a^2={tau:g}$"
        )
        for i, tau in enumerate(taus)
    ]
    status_handles = [
        Line2D([], [], marker="o", color="black", markerfacecolor="black", ls="none", label="accepted, ≥2 cuts"),
        Line2D([], [], marker="o", color="black", markerfacecolor="white", ls="none", label="accepted, 1 cut"),
        Line2D([], [], marker="X", color="black", markerfacecolor="black", ls="none", label="central fallback"),
    ]
    fig.legend(
        handles=flow_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=7,
        frameon=False,
        fontsize=8.3,
        title="flow time",
        title_fontsize=8.5,
    )
    fig.legend(
        handles=status_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        title
        + "\n"
        + rf"forward Sumratio-difference AIC, $T_{{\min}}=8$ "
        + r"($0.6200\,\mathrm{fm}>0.6\,\mathrm{fm}$), "
        + r"equal-weight mixture over usable cuts, $N_{\rm boot}=1000$",
        y=1.045,
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "title": title,
        "operator_indices": list(operator_indices),
        "z_range": [0, 15],
        "xlim": [-0.5, 15.5],
        "flow_time_x_offsets": flow_offsets.tolist(),
        "flow_time_x_offset_max": FLOW_X_OFFSET_MAX,
        "connect_flow_time_points": False,
        "status_counts": summary,
    }


def make_plots(
    fit_dir: Path,
    plot_dir: Path,
    dpi: int = 220,
    selected_taus: np.ndarray | None = None,
) -> dict:
    data = load_fits(fit_dir, selected_taus)
    taus = np.asarray(data["taus"], dtype=float)
    outputs = []
    outputs.append(
        _plot_axes(
            data,
            tuple(range(6)),
            plot_dir / "bare_matrix_elements_all_operators_vs_z_equal_cut_mixture",
            "All flowed-gluon bare matrix elements",
            dpi,
        )
    )
    outputs.append(
        _plot_axes(
            data,
            (0, 1, 2),
            plot_dir / "unpolarized_bare_matrix_elements_vs_z_equal_cut_mixture",
            "Unpolarized flowed-gluon bare matrix elements",
            dpi,
        )
    )
    outputs.append(
        _plot_axes(
            data,
            (3, 4, 5),
            plot_dir / "helicity_bare_matrix_elements_vs_z_equal_cut_mixture",
            "Helicity flowed-gluon bare matrix elements",
            dpi,
        )
    )

    # A compact multi-page PDF is convenient for inspecting both channels.
    multipage_path = plot_dir / "bare_matrix_elements_all_operators_vs_z.pdf"
    with PdfPages(multipage_path) as pdf:
        for operator_indices, title in (
            (tuple(range(3)), "Unpolarized flowed-gluon bare matrix elements"),
            (tuple(range(3, 6)), "Helicity flowed-gluon bare matrix elements"),
        ):
                values = np.asarray(data["M_filled"])
                errors = np.asarray(data["filled_statistical_error"])
                available = np.asarray(data["estimate_available"], dtype=bool)
                accepted = np.asarray(data["fit_accepted"], dtype=bool)
                fallback = np.asarray(data["central_fallback_used"], dtype=bool)
                usable = np.asarray(data["usable_cut_count"], dtype=int)
                fig, axes = plt.subplots(3, 3, figsize=(15.6, 12.0), sharex=True)
                colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(taus)))
                flow_offsets = np.linspace(-FLOW_X_OFFSET_MAX, FLOW_X_OFFSET_MAX, len(taus))
                for row, operator in enumerate(operator_indices):
                    for ip, momentum in enumerate(PABS):
                        ax = axes[row, ip]
                        for itau, tau in enumerate(taus):
                            mask = available[itau, operator, ip, :16]
                            if not np.any(mask):
                                continue
                            for category, marker, face in (
                                (
                                    mask & accepted[itau, operator, ip, :16] & (usable[itau, operator, ip, :16] >= 2),
                                    "o",
                                    colors[itau],
                                ),
                                (
                                    mask & accepted[itau, operator, ip, :16] & (usable[itau, operator, ip, :16] < 2),
                                    "o",
                                    "white",
                                ),
                                (mask & fallback[itau, operator, ip, :16], "X", colors[itau]),
                            ):
                                if np.any(category):
                                    ax.errorbar(
                                        Z[:16][category] + flow_offsets[itau],
                                        values[itau, operator, ip, :16][category],
                                        yerr=errors[itau, operator, ip, :16][category],
                                        fmt=marker,
                                        color=colors[itau],
                                        markerfacecolor=face,
                                        markeredgecolor=colors[itau],
                                        ms=3.4 if marker == "o" else 4.1,
                                        capsize=1.2,
                                        elinewidth=0.65,
                                        linestyle="none",
                                    )
                        ax.axhline(0.0, color="0.45", ls="--", lw=0.65)
                        ax.set_xlim(-0.5, 15.5)
                        ax.set_xticks((0, 3, 6, 9, 12, 15))
                        ax.grid(alpha=0.2, lw=0.5)
                        ax.set_title(rf"{OPERATOR_TITLES[operator]}, $P_z={momentum}$", fontsize=9)
                        if ip == 0:
                            ax.set_ylabel("bare M", fontsize=9)
                        if row == 2:
                            ax.set_xlabel(r"$z/a$", fontsize=9)
                handles = [
                    Line2D(
                        [], [], marker="o", color=colors[i],
                        markerfacecolor=colors[i], markeredgecolor=colors[i],
                        linestyle="none", markersize=3.8,
                        label=rf"$t_f/a^2={tau:g}$"
                    )
                    for i, tau in enumerate(taus)
                ]
                handles += [
                    Line2D([], [], marker="o", color="black", markerfacecolor="black", ls="none", label="accepted ≥2 cuts"),
                    Line2D([], [], marker="o", color="black", markerfacecolor="white", ls="none", label="accepted 1 cut"),
                    Line2D([], [], marker="X", color="black", ls="none", label="fallback"),
                ]
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=7, frameon=False, fontsize=7.3)
                fig.suptitle(
                    title + "\n" + r"$z/a=0\ldots15$, equal-weight usable-cut mixture, $N_{\rm boot}=1000$",
                    y=1.045,
                    fontsize=12,
                )
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    summary = {
        "schema": "gradient_flow_gluon_bare_matrix_elements_plot_v2",
        "fit_dir": str(fit_dir.resolve()),
        "plot_dir": str(plot_dir.resolve()),
        "flow_times_t_over_a2": taus.tolist(),
        "operators": list(OPERATOR_LABELS),
        "momenta_abs": PABS.tolist(),
        "cuts": CUTS.tolist(),
        "cut_policy": "equal-weight mixture over usable cuts; cut-to-cut spread is stored in fit products",
        "z_plotted": [0, 15],
        "xlim": [-0.5, 15.5],
        "flow_time_x_offsets": np.linspace(
            -FLOW_X_OFFSET_MAX, FLOW_X_OFFSET_MAX, len(taus)
        ).tolist(),
        "flow_time_x_offset_max": FLOW_X_OFFSET_MAX,
        "connect_flow_time_points": False,
        "value_field": "M_filled",
        "error_field": "filled_statistical_error",
        "status_markers": {
            "filled_circle": "accepted fit with usable_cut_count >= 2",
            "open_circle": "accepted fit with usable_cut_count == 1",
            "X": "central fallback; filled diagnostic value, not accepted fit",
        },
        "outputs": outputs,
        "multipage_pdf": str(multipage_path.resolve()),
        "input_records": data["records"],
    }
    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plot_dir / "plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, default=FIT_DIR_DEFAULT)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR_DEFAULT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--taus",
        type=str,
        default=None,
        help=(
            "comma-separated subset of flow times; default uses all 14 "
            "values, e.g. 0.5,1.0,1.4,1.8"
        ),
    )
    args = parser.parse_args()
    selected_taus = None
    if args.taus is not None:
        try:
            requested = [float(item.strip()) for item in args.taus.split(",") if item.strip()]
        except ValueError as exc:
            raise SystemExit(f"invalid --taus list: {args.taus!r}") from exc
        if not requested:
            raise SystemExit("--taus must contain at least one flow time")
        canonical = []
        for value in requested:
            matches = TAUS[np.isclose(TAUS, value, rtol=0.0, atol=1.0e-9)]
            if matches.size != 1:
                raise SystemExit(
                    f"flow time {value:g} is unavailable; choose from {TAUS.tolist()}"
                )
            canonical.append(float(matches[0]))
        selected_taus = np.asarray(canonical, dtype=float)
    summary = make_plots(
        args.fit_dir,
        args.plot_dir,
        dpi=args.dpi,
        selected_taus=selected_taus,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
