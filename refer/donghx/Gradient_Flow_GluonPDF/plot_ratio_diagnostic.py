#!/usr/bin/env python3
"""Plot insertion-time diagnostics for one gradient-flow gluon ratio product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def strings(values: np.ndarray) -> list[str]:
    return np.asarray(values).astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z", type=int, default=2)
    parser.add_argument("--tseps", default="5,7,9")
    args = parser.parse_args()
    wanted_tseps = [int(value) for value in args.tseps.split(",")]

    with np.load(args.product, allow_pickle=False) as data:
        schema = str(data["schema"])
        contract = json.loads(str(data["contract_json"]))
        if schema != "gradient_flow_gluon_ratio_v2":
            raise ValueError(f"Refusing non-v2 ratio product: {schema}")
        channels = strings(data["channel_labels"])
        orientations = strings(data["z_orientation_labels"])
        components = strings(data["component_labels"])
        momenta = data["pabs_list"].astype(int).tolist()
        tseps = data["tsep_values"].astype(int).tolist()
        z_values = data["z_values"].astype(int).tolist()
        ratio = np.asarray(data["ratio"])
        error_real = np.asarray(data["ratio_jackknife_error_real"])
        error_imag = np.asarray(data["ratio_jackknife_error_imag"])

    if args.z not in z_values:
        raise ValueError(f"z={args.z} unavailable; choices={z_values}")
    if any(value not in tseps for value in wanted_tseps):
        raise ValueError(f"Requested tseps {wanted_tseps}; available={tseps}")
    z_index = z_values.index(args.z)
    component_index = components.index("combined")
    direction_index = 0

    selections = [
        {
            "channel": "unpolarized",
            "orientation": "even_sum",
            "part": "real",
            "ylabel": r"$\mathrm{Re}\,R_{g}^{\mathrm{unpol,even}}$",
            "title": "Unpolarized (even Wilson-line combination)",
        },
        {
            "channel": "helicity",
            "orientation": "odd_difference",
            "part": "imag",
            "ylabel": r"$\mathrm{Re}[-iR_{g}^{\mathrm{hel,odd}}]=\mathrm{Im}\,R_{g}^{\mathrm{hel,odd}}$",
            "title": "Helicity (odd Wilson-line combination)",
        },
    ]
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.86, len(wanted_tseps)))
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(
        2, len(momenta), figsize=(4.25 * len(momenta), 7.2),
        sharex=False, sharey="row", constrained_layout=True,
    )
    summary: dict[str, object] = {
        "schema": "gradient_flow_gluon_ratio_plot_diagnostic_v1",
        "source_product": str(args.product.resolve()),
        "source_ratio_schema": schema,
        "flow_tau_t_over_a2": contract["flow_tau_t_over_a2"],
        "nconf": contract["nconf"],
        "z_over_a": args.z,
        "tsep_values": wanted_tseps,
        "momenta": momenta,
        "central_estimator": "full-sample disconnected ratio",
        "errors": "delete-one jackknife, matching real/imaginary component",
        "fit_performed": False,
        "points_at_floor_tsep_over_2": [],
    }

    for row, selection in enumerate(selections):
        ci = channels.index(selection["channel"])
        oi = orientations.index(selection["orientation"])
        values = ratio[ci, oi, component_index, direction_index, z_index]
        errors = error_real[ci, oi, component_index, direction_index, z_index]
        if selection["part"] == "imag":
            errors = error_imag[ci, oi, component_index, direction_index, z_index]
        for column, momentum in enumerate(momenta):
            ax = axes[row, column]
            for curve, tsep in enumerate(wanted_tseps):
                ti = tseps.index(tsep)
                insertion = np.arange(tsep + 1)
                complex_values = values[ti, : tsep + 1, column]
                y = complex_values.real if selection["part"] == "real" else complex_values.imag
                yerr = errors[ti, : tsep + 1, column]
                offset = (curve - (len(wanted_tseps) - 1) / 2) * 0.06
                ax.errorbar(
                    insertion + offset, y, yerr=yerr,
                    color=colors[curve], marker=markers[curve % len(markers)],
                    markersize=4.0, linewidth=1.1, elinewidth=0.9, capsize=2.0,
                    label=rf"$t_{{\rm sep}}/a={tsep}$",
                )
                midpoint = tsep // 2
                point = complex_values[midpoint]
                summary["points_at_floor_tsep_over_2"].append({
                    "channel": selection["channel"],
                    "orientation": selection["orientation"],
                    "momentum_z_lattice_units": momentum,
                    "tsep_over_a": tsep,
                    "insertion_over_a": midpoint,
                    "ratio_real": float(point.real),
                    "ratio_imag": float(point.imag),
                    "jackknife_error_real": float(
                        error_real[ci, oi, component_index, direction_index, z_index, ti, midpoint, column]
                    ),
                    "jackknife_error_imag": float(
                        error_imag[ci, oi, component_index, direction_index, z_index, ti, midpoint, column]
                    ),
                })
            ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
            ax.set_xlim(-0.45, max(wanted_tseps) + 0.45)
            ax.set_xticks(range(0, max(wanted_tseps) + 1))
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.set_title(f"{selection['title']}\n" + rf"$P_z={momentum}$", fontsize=10.5)
            if column == 0:
                ax.set_ylabel(selection["ylabel"])
            if row == 1:
                ax.set_xlabel(r"insertion time $t_{\rm ins}/a$")
            if row == 0 and column == len(momenta) - 1:
                ax.legend(frameon=False, fontsize=9, loc="best")

    flow_tau = float(contract["flow_tau_t_over_a2"])
    fig.suptitle(
        rf"Gradient-flow gluon disconnected $C_3/C_2$: "
        rf"$t_{{\rm flow}}/a^2={flow_tau:g}$, $z/a={args.z}$, "
        rf"$N_{{\rm cfg}}={contract['nconf']}$" + "\n"
        "full-sample central value; delete-one jackknife errors; no plateau fit",
        fontsize=13,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"tau{flow_tau:.3f}".replace(".", "p")
    stem = args.output_dir / f"ratio_plateau_{tag}_z{args.z}_P345"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    summary_path = Path(str(stem) + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"PLOT_OK png={stem.with_suffix('.png')}")
    print(f"PLOT_OK pdf={stem.with_suffix('.pdf')}")
    print(f"SUMMARY_OK json={summary_path}")


if __name__ == "__main__":
    main()
