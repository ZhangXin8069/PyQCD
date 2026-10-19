#!/usr/bin/env python3
"""Plot the preferred bare-matrix component for every available flow time."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TAUS = (0.05, 0.10, 0.20, 0.50, 0.80, 1.00)
CHANNELS = ("unpolarized", "helicity", "transversity")
FLAVORS = ("u_minus_d",)
COLORS = {0.05: "#0072B2", 0.10: "#D55E00", 0.20: "#009E73", 0.50: "#CC79A7", 0.80: "#E69F00", 1.00: "#56B4E9"}
COMPONENT = {"unpolarized": 0, "helicity": 1, "transversity": 1}
# Keep a stable horizontal position for a flow time even while tau=1.00 is
# pending.  The spacing is large enough that adjacent error bars remain legible.
FLOW_OFFSETS = dict(zip(TAUS, np.linspace(-0.24, 0.24, len(TAUS))))


def tag(tau: float) -> str:
    return f"tau{tau:.2f}".replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "plots")
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    for channel in CHANNELS:
        for flavor in FLAVORS:
            fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=False)
            comp = COMPONENT[channel]
            for ip, pz in enumerate((3, 4, 5)):
                ax = axes[ip]
                for tau in TAUS:
                    path = ROOT / "fits" / f"sumratio_{tag(tau)}_{channel}_{flavor}_Nboot100_seed1115.npz"
                    if not path.is_file():
                        continue
                    with np.load(path, allow_pickle=False) as d:
                        y = d["bare_matrix_element_raw_diagnostic"][comp, ip]
                        e = d["statistical_error_raw_diagnostic"][comp, ip]
                        ok = np.isfinite(y) & np.isfinite(e)
                        if not np.any(ok):
                            continue
                        ax.errorbar(d["delta_z_values"][ok] + FLOW_OFFSETS[tau], y[ok], yerr=e[ok], fmt="o", ms=3.0, capsize=1.5, linestyle="none", color=COLORS[tau], alpha=0.85, label=rf"$t_f/a^2={tau:g}$")
                ax.axhline(0.0, color="black", lw=0.7); ax.grid(alpha=0.22)
                ax.set_xlabel(r"$z/a$"); ax.set_xlim(-0.55, 20.55); ax.set_xticks(np.arange(0, 21, 2)); ax.set_title(rf"$P_z={pz}$", pad=9)
            axes[0].set_ylabel("bare matrix element (primary-fit raw diagnostic)")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                # Reserve a dedicated band for the legend so it cannot collide
                # with the title or the first row of points.
                fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                           bbox_to_anchor=(0.5, 0.915), columnspacing=1.5,
                           handletextpad=0.45, borderaxespad=0.0)
            note = "Connected u-d; Sumratio T7-10 L=R=2; 100 bootstrap replicas"
            if channel == "transversity": note += "; relative sign pending external confirmation"
            fig.suptitle(f"Gradient-flow {channel} bare matrix elements ({flavor.replace('_minus_', ' - ')})", y=0.985)
            fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=9)
            fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.78, wspace=0.27)
            fig.savefig(args.output / f"bare_{channel}_{flavor}_all_flow_times.pdf", bbox_inches="tight", pad_inches=0.16)
            fig.savefig(args.output / f"bare_{channel}_{flavor}_all_flow_times.png", dpi=180, bbox_inches="tight", pad_inches=0.16)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
