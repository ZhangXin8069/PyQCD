#!/usr/bin/env python3
"""Draw Fig. 1-style flow-time extrapolation panels.

The visual structure follows Fig. 1 of arXiv:2509.02472v1: vertically stacked
observables, flow-time points, a linear extrapolation to zero flow time, and a
shaded uncertainty band.  Unlike that paper, the inputs here are finite-a,
bare disconnected gluon matrix elements; there is no continuum extrapolation
or perturbative matching in this plotting program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from flow_time_analysis_v1.analyze import (
    COMPONENTS,
    FIT_SCHEMA,
    SCHEMA,
    WINDOWS,
    load_products,
    sha256_file,
)


DEFAULT_FIT_ROOT = Path(
    "/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/"
    "bare_fit_sumdiff_aic_v1"
)
DEFAULT_RESULT = Path(__file__).resolve().parent / "results" / "flow_time_extrapolation_v1.npz"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "fig1_style"

PANEL_LABELS = {
    "unpolarized_TH_minus_SH_even_real": r"Re$\,[(T_U-S_U)_{\rm even}]$",
    "helicity_TH_odd_imag": r"Im$\,[(T_H)_{\rm odd}]$",
    "helicity_TH_plus_SH_odd_imag": r"Im$\,[(T_H+S_H)_{\rm odd}]$",
}
PANEL_COLORS = {
    "unpolarized_TH_minus_SH_even_real": "#1f4e79",
    "helicity_TH_odd_imag": "#7651a6",
    "helicity_TH_plus_SH_odd_imag": "#985b47",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"Expected scalar {key}, got {value.shape}")
    return value.item()


def _index(values: np.ndarray, target: float | int, label: str) -> int:
    matches = np.flatnonzero(np.isclose(values.astype(float), float(target), atol=1e-12, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"Expected one {label}={target}, found {len(matches)}")
    return int(matches[0])


def _load_and_validate(result_path: Path, fit_root: Path):
    products = load_products(fit_root)
    result = np.load(result_path, allow_pickle=False)
    if str(result["schema"]) != SCHEMA:
        result.close()
        raise ValueError(f"Unexpected flow-analysis schema in {result_path}")
    if tuple(result["components"].astype(str).tolist()) != COMPONENTS:
        result.close()
        raise ValueError("Flow-analysis component labels do not match the fit products")
    if not np.array_equal(result["flow_taus"], products.taus):
        result.close()
        raise ValueError("Flow-time axis mismatch between analysis and fit products")
    if not np.array_equal(result["pabs_values"], products.pabs):
        result.close()
        raise ValueError("Momentum axis mismatch between analysis and fit products")
    if not np.array_equal(result["z_values"], products.z):
        result.close()
        raise ValueError("Separation axis mismatch between analysis and fit products")
    if not np.array_equal(result["boot_indices"], products.boot_indices):
        result.close()
        raise ValueError("Bootstrap indices are not shared between analysis and fit products")
    if int(_scalar(result, "fit_nboot")) != products.nboot:
        result.close()
        raise ValueError("Bootstrap count mismatch")
    return products, result


def _window_indices(result: np.lib.npyio.NpzFile, products, window: str) -> tuple[int, np.ndarray]:
    names = result["window_names"].astype(str)
    matches = np.flatnonzero(names == window)
    if len(matches) != 1:
        raise ValueError(f"Window {window!r} is absent or duplicated")
    iw = int(matches[0])
    nflow = int(result["window_lengths"][iw])
    registered = np.asarray(result["window_taus"][iw, :nflow], dtype=float)
    indices = np.asarray([_index(products.taus, tau, "tau") for tau in registered], dtype=np.int32)
    return iw, indices


def _flow_mask(products, indices: np.ndarray, z_index: int, radius_factor: float,
               include_z0: bool) -> np.ndarray:
    z_value = int(products.z[z_index])
    if include_z0 and z_value == 0:
        return np.ones(len(indices), dtype=bool)
    return (z_value > 0) & (np.sqrt(8.0 * products.taus[indices]) < radius_factor * z_value)


def _prediction_band(intercept_boot: np.ndarray, slope_boot: np.ndarray,
                     x_grid: np.ndarray) -> tuple[np.ndarray, int]:
    finite = np.isfinite(intercept_boot) & np.isfinite(slope_boot)
    if np.count_nonzero(finite) < 2:
        return np.full_like(x_grid, np.nan), 0
    predictions = intercept_boot[finite, None] + slope_boot[finite, None] * x_grid[None, :]
    error = np.std(predictions, axis=0, ddof=1)
    return error, int(np.count_nonzero(finite))


def _plot_one(*, products, result: np.lib.npyio.NpzFile, result_path: Path,
              output_dir: Path, z_value: int, momentum: int, window: str,
              show_all_flow_times: bool, dpi: int) -> tuple[list[Path], dict]:
    iw, window_indices = _window_indices(result, products, window)
    iz = _index(products.z, z_value, "z/a")
    ip = _index(products.pabs, momentum, "Pz")
    radius_factor = float(_scalar(result, "radius_factor"))
    include_z0 = bool(_scalar(result, "include_z0"))
    geometry = _flow_mask(products, window_indices, iz, radius_factor, include_z0)
    nominal_tau = products.taus[window_indices]
    nominal_max = float(np.max(nominal_tau))

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.85,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    fig, axes = plt.subplots(len(COMPONENTS), 1, figsize=(6.3, 8.8), sharex=True)
    panel_records: list[dict] = []

    for ic, (component, ax) in enumerate(zip(COMPONENTS, axes)):
        color = PANEL_COLORS[component]
        accepted = products.fit_accepted[:, ic, ip, iz]
        in_window = np.zeros(len(products.taus), dtype=bool)
        in_window[window_indices] = True
        geometry_global = np.zeros(len(products.taus), dtype=bool)
        geometry_global[window_indices] = geometry
        used = in_window & geometry_global & accepted
        geometry_excluded = in_window & ~geometry_global & accepted
        input_rejected = in_window & geometry_global & ~accepted

        if show_all_flow_times:
            outside = (~in_window) & accepted
            ax.errorbar(
                products.taus[outside], products.M[outside, ic, ip, iz],
                yerr=products.statistical_error[outside, ic, ip, iz],
                fmt="o", linestyle="none", ms=3.0, capsize=1.5, elinewidth=0.65,
                color="0.76", markerfacecolor="white", markeredgecolor="0.68",
                alpha=0.62, zorder=1, label="accepted, outside window" if ic == 0 else None,
            )
        ax.errorbar(
            products.taus[geometry_excluded], products.M[geometry_excluded, ic, ip, iz],
            yerr=products.statistical_error[geometry_excluded, ic, ip, iz],
            fmt="o", linestyle="none", ms=4.0, capsize=2.0, elinewidth=0.8,
            color="0.72", markerfacecolor="white", markeredgecolor="0.58",
            alpha=0.9, zorder=2, label="excluded from fit" if ic == 0 else None,
        )
        ax.errorbar(
            products.taus[used], products.M[used, ic, ip, iz],
            yerr=products.statistical_error[used, ic, ip, iz],
            fmt="o", linestyle="none", ms=4.8, capsize=2.2, elinewidth=0.9,
            color="black", markerfacecolor="black", zorder=5,
            label="flow-time data used" if ic == 0 else None,
        )

        computed = bool(result["fit_computed"][iw, ic, ip, iz])
        quality = bool(result["quality_pass"][iw, ic, ip, iz])
        status = str(result["status"][iw, ic, ip, iz])
        fit_tau = products.taus[used]
        if computed and quality and len(fit_tau):
            x_grid = np.linspace(0.0, float(np.max(fit_tau)), 240)
            intercept = float(result["intercept"][iw, ic, ip, iz])
            slope = float(result["slope_tau"][iw, ic, ip, iz])
            intercept_error = float(result["intercept_error"][iw, ic, ip, iz])
            intercept_boot = result["intercept_boot"][iw, :, ic, ip, iz]
            slope_boot = result["slope_tau_boot"][iw, :, ic, ip, iz]
            band_error, nfinite = _prediction_band(intercept_boot, slope_boot, x_grid)
            central = intercept + slope * x_grid
            ax.fill_between(
                x_grid, central - band_error, central + band_error,
                color=color, alpha=0.18, linewidth=0.0, zorder=2,
                label=r"$M_0+c_\tau\tau$ bootstrap $1\sigma$" if ic == 0 else None,
            )
            ax.plot(x_grid, central, color=color, lw=1.35, zorder=4)
            ax.errorbar(
                [0.0], [intercept], yerr=[intercept_error], fmt="D", linestyle="none",
                ms=5.0, capsize=2.6, elinewidth=1.0, color=color,
                markeredgecolor=color, zorder=6, label=r"$M_0$" if ic == 0 else None,
            )
            annotation = (
                rf"$M_0={intercept:+.3g}\pm{intercept_error:.2g}$" + "\n" +
                rf"$\chi^2/{int(result['dof'][iw, ic, ip, iz])}="
                rf"{float(result['chi2'][iw, ic, ip, iz]):.2f}$, "
                rf"$Q={float(result['q_value'][iw, ic, ip, iz]):.2f}$"
            )
        else:
            intercept = slope = intercept_error = np.nan
            nfinite = 0
            annotation = f"no accepted flow fit\n{status}"

        ax.text(
            0.985, 0.92, annotation, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        )
        ax.set_ylabel(PANEL_LABELS[component], fontsize=11)
        ax.axhline(0.0, color="0.55", lw=0.65, ls="--", zorder=0)
        ax.grid(True, color="0.72", lw=0.65, alpha=0.75)
        ax.text(0.012, 0.93, f"({chr(ord('a') + ic)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=10)
        panel_records.append({
            "component": component,
            "fit_computed": computed,
            "quality_pass": quality,
            "status": status,
            "fit_taus": products.taus[used].tolist(),
            "nominal_window_taus": nominal_tau.tolist(),
            "geometry_excluded_taus": products.taus[geometry_excluded].tolist(),
            "input_rejected_taus": products.taus[input_rejected].tolist(),
            "intercept": intercept if np.isfinite(intercept) else None,
            "intercept_error": intercept_error if np.isfinite(intercept_error) else None,
            "slope_tau": slope if np.isfinite(slope) else None,
            "chi2": float(result["chi2"][iw, ic, ip, iz]) if computed else None,
            "dof": int(result["dof"][iw, ic, ip, iz]) if computed else None,
            "Q": float(result["q_value"][iw, ic, ip, iz]) if computed else None,
            "finite_fit_replicas": int(nfinite),
        })

    axes[-1].set_xlabel(r"flow time $\tau/a^2$", fontsize=12)
    if show_all_flow_times:
        axes[-1].set_xlim(-0.08, float(np.max(products.taus)) + 0.16)
    else:
        axes[-1].set_xlim(-0.035, nominal_max + 0.045)
    title = (
        rf"$32^3\!\times\!96$, $a=0.0775$ fm, $z/a={z_value}$, "
        rf"$P_z={momentum}(2\pi/L)$" + "\n" +
        rf"finite-$a$ bare flow-time extrapolation, window $\tau/a^2={nominal_tau[0]:g}$--${nominal_tau[-1]:g}$"
    )
    fig.suptitle(title, fontsize=13, y=0.988)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        order = [i for i, label in enumerate(labels) if label]
        fig.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2,
            frameon=True, framealpha=0.92, fontsize=8.5,
        )
    # Keep enough room for the two-line title and shared legend without
    # leaving an oversized blank band above the first panel.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.895), h_pad=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    view_tag = "full" if show_all_flow_times else "fit_window"
    stem = output_dir / f"fig1_style_flow_extrapolation_z{z_value}_Pz{momentum}_{view_tag}"
    outputs: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)

    record = {
        "z_over_a": z_value,
        "momentum_Pz_2pi_over_L": momentum,
        "window": window,
        "view": view_tag,
        "radius_factor": radius_factor,
        "panels": panel_records,
        "outputs": [str(path.resolve()) for path in outputs],
    }
    return outputs, record


def run(args: argparse.Namespace) -> None:
    fit_root = args.fit_root.resolve()
    result_path = args.result.resolve()
    output_dir = args.output_dir.resolve()
    products, result = _load_and_validate(result_path, fit_root)
    window = args.window or str(result["primary_window"])
    if window not in WINDOWS:
        result.close()
        raise ValueError(f"Unknown registered window: {window}")

    outputs: list[Path] = []
    records: list[dict] = []
    try:
        views = (False, True) if args.with_full_view else (False,)
        for z_value in args.z:
            for momentum in args.momenta:
                for full_view in views:
                    paths, record = _plot_one(
                        products=products,
                        result=result,
                        result_path=result_path,
                        output_dir=output_dir,
                        z_value=z_value,
                        momentum=momentum,
                        window=window,
                        show_all_flow_times=full_view,
                        dpi=args.dpi,
                    )
                    outputs.extend(paths)
                    records.append(record)
    finally:
        result.close()

    metadata = {
        "schema": "gradient_flow_gluon_fig1_style_plot_v1",
        "status": "complete",
        "interpretation": "finite_a_bare_flow_time_diagnostic_not_continuum_or_matched_pdf",
        "reference_style": "arXiv:2509.02472v1 Fig.1",
        "fit_input_schema": FIT_SCHEMA,
        "flow_analysis_schema": SCHEMA,
        "fit_root": str(fit_root),
        "flow_analysis_product": str(result_path),
        "flow_analysis_product_sha256": _sha256(result_path),
        "plot_code_sha256": sha256_file(Path(__file__).resolve()),
        "nconf": len(products.confs),
        "nboot": products.nboot,
        "seed": 1115,
        "window": window,
        "records": records,
    }
    metadata_path = output_dir / "fig1_style_flow_extrapolation.json"
    temporary = output_dir / f".{metadata_path.name}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(json.dumps({
        "status": "complete",
        "window": window,
        "z": args.z,
        "momenta": args.momenta,
        "figure_files": len(outputs),
        "metadata": str(metadata_path.resolve()),
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window", choices=tuple(WINDOWS), default=None,
                        help="registered flow window; default is the product's primary window")
    parser.add_argument("--z", type=int, nargs="+", default=[2, 6],
                        help="one or more z/a values (default: 2 6)")
    parser.add_argument("--momenta", type=int, nargs="+", default=[3, 4, 5],
                        help="one or more Pz/(2pi/L) values (default: 3 4 5)")
    parser.add_argument("--with-full-view", action="store_true",
                        help="also show all accepted flow times outside the fit window")
    parser.add_argument("--dpi", type=int, default=240)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
