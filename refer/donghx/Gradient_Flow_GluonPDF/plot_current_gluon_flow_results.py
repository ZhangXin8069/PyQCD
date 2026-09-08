#!/usr/bin/env python3
"""Plot current finite-flow gluon C3/C2 and fitted bare matrix elements.

The ratio-v3 products retain pointwise jackknife errors and are therefore used
for the C3/C2 insertion-time figures.  The latest N406 bare-fit products are
used for the fitted matrix-element figures.  The two archives have different
configuration counts; this is recorded in the manifest rather than silently
mixing their statistical errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATIO_ROOT = Path("/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/results_v3")
FIT_ROOT = Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/fit_latest_v1/results_v1")
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)
OPERATORS = (
    "unpolarized_TH_minus_SH_even_real",
    "helicity_TH_odd_imag",
    "helicity_TH_plus_SH_odd_imag",
)
DISPLAY = {
    OPERATORS[0]: r"unpolarized: Re$[(T_U-S_U)_{\rm even}]$",
    OPERATORS[1]: r"helicity: Im$[(T_H)_{\rm odd}]$",
    OPERATORS[2]: r"helicity: Im$[(T_H+S_H)_{\rm odd}]$",
}
RATIO_SELECT = (("unpolarized", "even_sum", "combined", "real"),
                ("helicity", "odd_difference", "Mtiti", "imag"),
                ("helicity", "odd_difference", "combined", "imag"))


def tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def load_ratio(tau: float, z: int, tsep: int):
    path = RATIO_ROOT / f"ratio_v3_{tag(tau)}_N231.npz"
    with np.load(path, allow_pickle=False) as d:
        schema = str(d["schema"])
        if schema != "gradient_flow_gluon_ratio_v3":
            raise ValueError(f"unexpected ratio schema in {path}: {schema}")
        channels = d["channel_labels"].astype(str).tolist()
        orientations = d["z_orientation_labels"].astype(str).tolist()
        components = d["component_labels"].astype(str).tolist()
        pabs = d["pabs_list"].astype(int)
        zvals = d["z_values"].astype(int).tolist()
        tseps = d["tsep_values"].astype(int).tolist()
        if z not in zvals or tsep not in tseps:
            raise ValueError(f"z/tsep unavailable in {path}")
        zi, ti = zvals.index(z), tseps.index(tsep)
        ratio = np.asarray(d["ratio"])
        er = np.asarray(d["ratio_jackknife_error_real"])
        ei = np.asarray(d["ratio_jackknife_error_imag"])
        rows = []
        for channel, orient, comp, projection in RATIO_SELECT:
            ci, oi, ki = channels.index(channel), orientations.index(orient), components.index(comp)
            values = ratio[ci, oi, ki, 0, zi, ti, :tsep + 1, :]
            errors = (er if projection == "real" else ei)[ci, oi, ki, 0, zi, ti, :tsep + 1, :]
            rows.append((values.real if projection == "real" else values.imag, errors))
        return np.asarray(rows), np.asarray(pabs), int(d["confs"].shape[0])


def load_fits(tau: float, z: int):
    path = FIT_ROOT / f"bare_matrix_latest_{tag(tau)}_N406.npz"
    with np.load(path, allow_pickle=False) as d:
        labels = d["operator_labels"].astype(str).tolist()
        if labels != list(OPERATORS):
            raise ValueError(f"fit operator contract mismatch in {path}: {labels}")
        zi = int(np.flatnonzero(d["z_values"] == z)[0])
        # ``fit_latest_v1`` predates the explicit usable_cut_count field.  Its
        # candidate eligibility mask and candidate_cut labels contain the same
        # information: count registered cuts with at least one eligible
        # (quality-gated) candidate at each operator/momentum/z point.
        candidate_eligible = np.asarray(d["candidate_eligible"], dtype=bool)
        candidate_cuts = np.asarray(d["candidate_cut"], dtype=int)
        registered_cuts = np.asarray(d["registered_cuts"], dtype=int)
        usable_cut_count = np.zeros(candidate_eligible.shape[0:1] + candidate_eligible.shape[2:], dtype=int)
        for cut in registered_cuts:
            usable_cut_count += np.any(candidate_eligible[:, candidate_cuts == cut, :, :], axis=1)
        return {
            "M": np.asarray(d["M_filled"])[:, :, zi],
            "err": np.asarray(d["filled_statistical_error"])[:, :, zi],
            "available": np.asarray(d["estimate_available"])[:, :, zi],
            "accepted": np.asarray(d["fit_accepted"])[:, :, zi],
            "fallback": np.asarray(d["central_fallback_used"])[:, :, zi],
            "cuts": usable_cut_count[:, :, zi],
            "nconf": int(d["nconf"]),
            "pabs": np.asarray(d["pabs_values"], dtype=int),
        }


def select_momenta(available: np.ndarray, requested: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and labels for the requested momentum subset."""
    available = np.asarray(available, dtype=int)
    requested = [int(momentum) for momentum in requested]
    missing = [momentum for momentum in requested if momentum not in available]
    if missing:
        raise ValueError(f"requested momenta {missing} unavailable; available={available.tolist()}")
    indices = np.asarray([int(np.flatnonzero(available == momentum)[0]) for momentum in requested])
    return indices, available[indices]


def momentum_x_offsets(count: int) -> np.ndarray:
    """Small symmetric offsets for overlaid momentum series."""
    if count <= 1:
        return np.zeros(count, dtype=float)
    return np.linspace(-0.12, 0.12, count)


def save_ratio_plots(
    out: Path, z: int, tsep: int, taus: np.ndarray, requested_pabs: list[int]
):
    all_values, all_errors, pabs, nconf = [], [], None, None
    for tau in taus:
        loaded, current_pabs, nc = load_ratio(float(tau), z, tsep)
        # All flow-time products must share the same momentum axis.  Keep the
        # first axis and fail loudly if a product has been written with a
        # different ordering or momentum set.
        if pabs is None:
            pidx, pabs = select_momenta(current_pabs, requested_pabs)
        elif not np.array_equal(pabs, current_pabs[pidx]):
            raise ValueError("momentum axes differ across flow times")
        all_values.append(loaded[:, 0][:, :, pidx])
        all_errors.append(loaded[:, 1][:, :, pidx])
        nconf = nc
    values = np.asarray(all_values)  # tau, op, insertion, p
    errors = np.asarray(all_errors)
    # One 2x7 panel figure per operator, with independent Pz colors.  If
    # multiple momenta are selected, they share the same insertion-time
    # abscissae, so draw them at small, symmetric horizontal offsets.  This
    # keeps the error bars and markers readable without changing the underlying
    # insertion times.  With the default Pz=3-only view the offset is zero.
    colors = ("#1f77b4", "#d95f02", "#2ca02c")
    x_offsets = momentum_x_offsets(len(pabs))
    for op, label in enumerate(OPERATORS):
        fig, axes = plt.subplots(2, 7, figsize=(16, 5.8), sharey=True)
        axes = axes.ravel()
        for it, tau in enumerate(taus):
            ax = axes[it]
            for ip, mom in enumerate(pabs):
                x = np.arange(tsep + 1, dtype=float) + x_offsets[ip]
                ax.errorbar(x, values[it, op, :, ip], yerr=errors[it, op, :, ip],
                            fmt="o", linestyle="none", ms=2.8, capsize=1.8,
                            color=colors[ip], label=rf"$P_z={mom}$")
            ax.axhline(0, color="0.5", ls=":", lw=.7); ax.grid(alpha=.18)
            ax.set_title(rf"$t/a^2={tau:g}$", fontsize=9)
            ax.set_xlabel(r"$t_{\rm ins}/a$", fontsize=8)
            ax.set_xticks(np.arange(tsep + 1))
            ax.set_xlim(-0.38, tsep + 0.38)
            ax.tick_params(labelsize=8)
            if it % 7 == 0: ax.set_ylabel("C3/C2", fontsize=9)
            if it == 0: ax.legend(frameon=False, fontsize=7, loc="best")
        fig.suptitle(f"{DISPLAY[label]} at z/a={z}, tsep/a={tsep}; Ncfg={nconf}\n"
                     "pointwise ratio-v3 jackknife errors", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .92))
        stem = out / f"c3c2_vs_insertion_{label}_z{z}_tsep{tsep}_all_flow"
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    # Midpoint ratio as a compact flow-time overview.
    mid = tsep // 2
    fig, axes = plt.subplots(3, len(pabs), figsize=(5.2 * len(pabs), 10), sharex=True)
    axes = np.asarray(axes).reshape(3, len(pabs))
    for op, label in enumerate(OPERATORS):
        for ip, mom in enumerate(pabs):
            ax = axes[op, ip]
            ax.errorbar(taus, values[:, op, mid, ip], yerr=errors[:, op, mid, ip],
                        fmt="o", linestyle="none", ms=4, capsize=2,
                        color=colors[ip])
            ax.axhline(0, color="0.5", ls=":", lw=.7); ax.grid(alpha=.2)
            ax.set_title(rf"{label.split('_')[0]}, $P_z={mom}$", fontsize=10)
            ax.set_xlabel(r"flow time $t/a^2$"); ax.set_ylabel("C3/C2")
    fig.suptitle(
        rf"Gluon C3/C2 at z/a={z}, $t_{{\rm sep}}$/a={tsep}, insertion midpoint={mid}"
        "\nfinite-flow bare disconnected ratio",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out / "c3c2_midpoint_vs_flow_z2_tsep6.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "c3c2_midpoint_vs_flow_z2_tsep6.pdf", bbox_inches="tight")
    plt.close(fig)
    return int(nconf), [str(x) for x in pabs]


def save_fit_plot(
    out: Path,
    z: int,
    taus: np.ndarray,
    requested_pabs: list[int],
    *,
    full_range: bool = False,
):
    records = [load_fits(float(tau), z) for tau in taus]
    pidx, pabs = select_momenta(records[0]["pabs"], requested_pabs)
    M = np.asarray([r["M"][:, pidx] for r in records])
    E = np.asarray([r["err"][:, pidx] for r in records])
    available = np.asarray([r["available"][:, pidx] for r in records])
    accepted = np.asarray([r["accepted"][:, pidx] for r in records])
    fallback = np.asarray([r["fallback"][:, pidx] for r in records])
    cuts = np.asarray([r["cuts"][:, pidx] for r in records])
    colors = ("#1f77b4", "#d95f02", "#2ca02c")
    fig, axes = plt.subplots(3, len(pabs), figsize=(5.2 * len(pabs), 10), sharex=True)
    axes = np.asarray(axes).reshape(3, len(pabs))
    for op, label in enumerate(OPERATORS):
        for ip, mom in enumerate(pabs):
            ax = axes[op, ip]
            for it, tau in enumerate(taus):
                if not available[it, op, ip]: continue
                if fallback[it, op, ip] or not accepted[it, op, ip]:
                    marker, face = "X", colors[ip]
                elif cuts[it, op, ip] >= 2:
                    marker, face = "o", colors[ip]
                else:
                    marker, face = "o", "white"
                ax.errorbar([tau], [M[it, op, ip]], yerr=[E[it, op, ip]],
                            fmt=marker, linestyle="none", ms=5, capsize=2,
                            color=colors[ip], markerfacecolor=face,
                            markeredgecolor=colors[ip])
            ax.axhline(0, color="0.5", ls=":", lw=.7); ax.grid(alpha=.2)
            ax.set_title(rf"$P_z={mom}$", fontsize=10)
            ax.set_xlabel(r"flow time $t/a^2$"); ax.set_ylabel("fitted bare M")
            if full_range:
                # A companion view keeps every finite flow-time point visible;
                # the canonical view below retains the requested comparison
                # ranges (-1.5,0.5) and (-0.3,0.3).
                finite = np.isfinite(M[:, op, ip]) & np.isfinite(E[:, op, ip])
                if np.any(finite):
                    lo = float(np.nanmin(M[:, op, ip][finite] - E[:, op, ip][finite]))
                    hi = float(np.nanmax(M[:, op, ip][finite] + E[:, op, ip][finite]))
                    pad = max(0.05, 0.06 * (hi - lo))
                    ax.set_ylim(lo - pad, hi + pad)
            else:
                ax.set_ylim((-1.5, .5) if op == 0 else (-.3, .3))
    handles = [plt.Line2D([], [], marker="o", ls="none", color="k", label="accepted"),
               plt.Line2D([], [], marker="o", mfc="white", ls="none", color="k", label="one cut"),
               plt.Line2D([], [], marker="X", ls="none", color="k", label="minimum-chi2 fallback")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
    fig.suptitle(
        rf"Fitted gluon bare matrix elements at z/a={z}"
        "\n"
        r"forward Sumratio-difference AIC, $T_{\min}a>0.6$ fm; Ncfg=406",
        fontsize=13,
        y=.98,
    )
    fig.tight_layout(rect=(0, 0, 1, .88))
    stem = "bare_matrix_vs_flow_z2_all_flow_fullrange" if full_range else "bare_matrix_vs_flow_z2_all_flow"
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return int(records[0]["nconf"]), [str(x) for x in pabs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("gluon_flow_current_plots_v1"))
    parser.add_argument("--z", type=int, default=2)
    parser.add_argument("--tsep", type=int, default=6)
    parser.add_argument("--tau-min", type=float, default=0.1)
    parser.add_argument("--tau-max", type=float, default=3.8)
    parser.add_argument(
        "--pabs-list",
        type=str,
        default="3",
        help="comma-separated |Pz| values to plot (default: 3)",
    )
    args = parser.parse_args()
    requested_pabs = [int(item) for item in args.pabs_list.replace(" ", ",").split(",") if item]
    if not requested_pabs or len(requested_pabs) != len(set(requested_pabs)):
        raise ValueError("--pabs-list must contain one or more unique integers")
    taus = np.asarray([t for t in TAUS if args.tau_min <= t <= args.tau_max], dtype=float)
    missing_ratio = [t for t in taus if not (RATIO_ROOT / f"ratio_v3_{tag(t)}_N231.npz").is_file()]
    missing_fit = [t for t in taus if not (FIT_ROOT / f"bare_matrix_latest_{tag(t)}_N406.npz").is_file()]
    if missing_ratio or missing_fit:
        raise FileNotFoundError(f"missing ratio={missing_ratio}, fit={missing_fit}")
    args.output.mkdir(parents=True, exist_ok=True)
    nratio, p1 = save_ratio_plots(args.output, args.z, args.tsep, taus, requested_pabs)
    nfit, p2 = save_fit_plot(args.output, args.z, taus, requested_pabs)
    # Also emit a full-range companion so points below the requested
    # comparison limits are not silently hidden (notably unpolarized at
    # z/a=2 and large flow time).
    save_fit_plot(args.output, args.z, taus, requested_pabs, full_range=True)
    manifest = {
        "schema": "gluon_flow_current_plot_v1",
        "flow_times": taus.tolist(), "z_over_a": args.z, "tsep_over_a": args.tsep,
        "requested_pabs": requested_pabs,
        "ratio_source": str(RATIO_ROOT), "ratio_nconf": nratio, "ratio_pabs": p1,
        "fit_source": str(FIT_ROOT), "fit_nconf": nfit, "fit_pabs": p2,
        "ratio_projection": "unpolarized Re even_sum combined; helicity Im odd_difference Mtiti and combined",
        "ratio_insertion_x_offsets": {
            str(momentum): float(offset)
            for momentum, offset in zip(p1, momentum_x_offsets(len(p1)))
        },
        "fit_projection": list(OPERATORS),
        "plot_files": {
            "ratio_insertion": [
                "c3c2_vs_insertion_unpolarized_TH_minus_SH_even_real_z2_tsep6_all_flow",
                "c3c2_vs_insertion_helicity_TH_odd_imag_z2_tsep6_all_flow",
                "c3c2_vs_insertion_helicity_TH_plus_SH_odd_imag_z2_tsep6_all_flow",
            ],
            "ratio_midpoint": "c3c2_midpoint_vs_flow_z2_tsep6",
            "bare_matrix_fixed_ylim": "bare_matrix_vs_flow_z2_all_flow",
            "bare_matrix_full_range": "bare_matrix_vs_flow_z2_all_flow_fullrange",
        },
        "bare_matrix_y_limits": {
            "fixed": {"unpolarized": [-1.5, 0.5], "helicity": [-0.3, 0.3]},
            "full_range_companion": True,
        },
        "status": "finite_flow_bare_disconnected_diagnostic; no renormalization or LaMET matching",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
