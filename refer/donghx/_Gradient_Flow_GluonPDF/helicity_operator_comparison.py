#!/usr/bin/env python3
"""Compare three Euclidean helicity selectors with shared jackknife samples.

The script reads the independently stored ``Mtiti`` and ``Mijij`` components
of the schema-v2 flowed gluon OPE and forms

    TH_minus_SH = (Mtiti - Mijij)_odd,
    TH_plus_SH  = (Mtiti + Mijij)_odd,
    TH_only     = Mtiti_odd.

For every selector, the disconnected numerator uses the complete complex
``pol35`` two-point function, while the denominator uses ``nopol``.  Vacuum
subtraction and the denominator are recomputed in every delete-one sample.
The products are finite-flow bare disconnected ratios, not physical PDFs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OPE_ROOT = Path(
    "/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/output_v5"
)
DEFAULT_TWOPT_ROOT = Path(
    "/public/group/lqcd/donghx/2pt_Result/"
    "beta6.41_mu-0.2295_ms-0.2050_L32x96"
)
DEFAULT_MANIFEST = Path(
    "/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/"
    "manifests/common_all14_plusz_P345.tsv"
)
DEFAULT_REFERENCE = Path(
    "/public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/"
    "results_v2/ratio_v2_tau0p500_N231.npz"
)

NT = 96
NZ = 25
OPE_SCHEMA = "gradient_flow_gluon_ope_v2"
SCHEMA = "gradient_flow_gluon_helicity_operator_comparison_v1"
FIELD_PROJECTIONS = ("legacy_untraced", "traceless")
CHANNELS = ("unpolarized", "helicity")
ORIENTATIONS = ("plus_z_raw", "minus_z_raw", "even_sum", "odd_difference")
COMPONENTS = ("combined", "Mtiti", "Mijij", "ti", "tj", "ij_single")
SELECTORS = ("TH_minus_SH", "TH_plus_SH", "TH_only")
COEFFICIENTS = np.asarray(((1.0, -1.0), (1.0, 1.0), (1.0, 0.0)))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    return sha256_file(path)


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def read_manifest(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t"))
    confs = [str(row["conf_id"]) for row in rows]
    if len(confs) < 2 or len(confs) != len(set(confs)):
        raise ValueError("Manifest needs at least two unique configurations")
    return confs, hashlib.sha256(raw).hexdigest()


def ope_base(root: Path, conf: str, tau: float, epsilon: float) -> Path:
    tag = tau_tag(tau)
    return root / f"conf{conf}" / tag / f"conf{conf}_{tag}_eps{epsilon:.3f}_zdir2"


def load_odd_components(
    root: Path, conf: str, tau: float, epsilon: float, z_index: int
) -> np.ndarray:
    """Return (Mtiti,Mijij,time) in the helicity odd channel."""
    base = ope_base(root, conf, tau, epsilon)
    npy = Path(str(base) + ".npy")
    meta_path = Path(str(base) + ".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_shape = (2, 2, 4, 6, NZ, NT)
    checks = {
        "schema": metadata.get("schema") == OPE_SCHEMA,
        "conf": str(metadata.get("conf_id")) == conf,
        "shape": tuple(metadata.get("shape", ())) == expected_shape,
        "thin": metadata.get("input_scheme") == "thin_link_no_hyp_no_smear",
        "tau": abs(float(metadata.get("flow", {}).get("tau_t_over_a2", -1)) - tau)
        < 1e-12,
        "epsilon": abs(float(metadata.get("flow", {}).get("epsilon", -1)) - epsilon)
        < 1e-12,
        "axes": metadata.get("axes")
        == ["field_projection", "channel", "z_orientation", "component", "z", "t"],
        "projection_labels": metadata.get("axis_labels", {}).get("field_projection")
        == list(FIELD_PROJECTIONS),
        "channel_labels": metadata.get("axis_labels", {}).get("channel")
        == list(CHANNELS),
        "orientation_labels": metadata.get("axis_labels", {}).get("z_orientation")
        == list(ORIENTATIONS),
        "component_labels": metadata.get("axis_labels", {}).get("component")
        == list(COMPONENTS),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"OPE metadata failed {failed}: {base}")
    data = np.load(npy, mmap_mode="r", allow_pickle=False)
    if data.shape != expected_shape or data.dtype != np.dtype("complex128"):
        raise ValueError(f"Unexpected OPE tensor {data.shape} {data.dtype}: {npy}")
    selected = np.asarray(
        data[
            FIELD_PROJECTIONS.index("traceless"),
            CHANNELS.index("helicity"),
            ORIENTATIONS.index("odd_difference"),
            [COMPONENTS.index("Mtiti"), COMPONENTS.index("Mijij")],
            z_index,
            :,
        ],
        dtype=np.complex128,
    )
    if selected.shape != (2, NT) or not np.isfinite(selected).all():
        raise ValueError(f"Invalid selected OPE data: {npy}")
    return selected


def c2_path(root: Path, conf: str, pabs: int, polarization: str) -> Path:
    return (
        root
        / "momsmear2z"
        / conf
        / (
            f"twopt_slice_pp_Px0Py0Pz{pabs}_eginphase2_Cg5g4_"
            f"{polarization}_ss_conf{conf}.npy"
        )
    )


def load_c2(path: Path, tseps: np.ndarray) -> np.ndarray:
    data = np.squeeze(np.asarray(np.load(path, allow_pickle=False)))
    if data.shape != (NT, NT) or data.dtype.kind != "c" or not np.isfinite(data).all():
        raise ValueError(f"Invalid C2 {data.shape} {data.dtype}: {path}")
    data = np.asarray(data, dtype=np.complex128)
    source = np.arange(NT)
    return np.stack([data[(source + dt) % NT, source] for dt in tseps])


def load_inputs(
    confs: list[str],
    ope_root: Path,
    twopt_root: Path,
    tau: float,
    epsilon: float,
    z_index: int,
    pabs: np.ndarray,
    tseps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.empty((len(confs), 2, NT), dtype=np.complex128)
    pol35 = np.empty((len(confs), len(tseps), NT, len(pabs)), dtype=np.complex128)
    nopol = np.empty_like(pol35)
    for iconf, conf in enumerate(confs):
        if iconf % 20 == 0 or iconf + 1 == len(confs):
            print(f"LOAD {iconf + 1}/{len(confs)} conf={conf}", flush=True)
        raw[iconf] = load_odd_components(ope_root, conf, tau, epsilon, z_index)
        for ip, momentum in enumerate(pabs):
            pol35[iconf, :, :, ip] = load_c2(
                c2_path(twopt_root, conf, int(momentum), "pol35"), tseps
            )
            nopol[iconf, :, :, ip] = load_c2(
                c2_path(twopt_root, conf, int(momentum), "nopol"), tseps
            )
    operators = np.einsum("kr,nrs->nks", COEFFICIENTS, raw, optimize=True)
    return operators, pol35, nopol


def one_estimator(
    operator: np.ndarray, numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-sample ratio (selector,P) and JK replicas (conf,selector,P)."""
    op = np.asarray(operator, dtype=np.complex128)
    num = np.asarray(numerator, dtype=np.complex128)
    den = np.asarray(denominator, dtype=np.complex128)
    nconf, nselector, nsource = op.shape
    if num.shape[:2] != (nconf, nsource) or den.shape != num.shape:
        raise ValueError(f"Estimator shape mismatch: {op.shape}, {num.shape}, {den.shape}")
    nminus = nconf - 1
    sum_o = op.sum(axis=0)
    sum_n = num.sum(axis=0)
    sum_d = den.sum(axis=0)
    sum_on = np.einsum("nks,nsp->nkp", op, num, optimize=True)
    total_on = sum_on.sum(axis=0) / nsource
    total_o_n = np.einsum("ks,sp->kp", sum_o, sum_n, optimize=True) / nsource
    c3 = total_on / nconf - total_o_n / (nconf * nconf)
    c2 = sum_d.mean(axis=0) / nconf
    central = c3 / c2[None, :]

    own_o_total_n = np.einsum("nks,sp->nkp", op, sum_n, optimize=True) / nsource
    total_o_own_n = np.einsum("ks,nsp->nkp", sum_o, num, optimize=True) / nsource
    own_on = sum_on / nsource
    c3_jk = (
        (total_on[None] - own_on) / nminus
        - (total_o_n[None] - own_o_total_n - total_o_own_n + own_on)
        / (nminus * nminus)
    )
    c2_jk = (sum_d.mean(axis=0)[None] - den.mean(axis=1)) / nminus
    samples = c3_jk / c2_jk[:, None, :]
    return central, samples


def calculate(
    operators: np.ndarray,
    pol35: np.ndarray,
    nopol: np.ndarray,
    tseps: np.ndarray,
) -> dict[str, np.ndarray]:
    nconf, nselector, _ = operators.shape
    max_tsep = int(tseps.max())
    npmom = pol35.shape[-1]
    shape = (nselector, len(tseps), max_tsep + 1, npmom)
    central = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    samples = np.full((nconf,) + shape, np.nan + 1j * np.nan, dtype=np.complex128)
    source = np.arange(NT)
    for idt, tsep in enumerate(tseps):
        for insertion in range(int(tsep) + 1):
            shifted = operators[:, :, (source + insertion) % NT]
            value, replicas = one_estimator(
                shifted, pol35[:, idt], nopol[:, idt]
            )
            central[:, idt, insertion] = value
            samples[:, :, idt, insertion] = replicas
    jk_mean = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    for idt, tsep in enumerate(tseps):
        jk_mean[:, idt, : int(tsep) + 1] = samples[
            :, :, idt, : int(tsep) + 1
        ].mean(axis=0)
    bias_corrected = nconf * central - (nconf - 1) * jk_mean
    prefactor = (nconf - 1) / nconf
    error_real = np.sqrt(
        prefactor * np.nansum((samples.real - jk_mean.real[None]) ** 2, axis=0)
    )
    error_imag = np.sqrt(
        prefactor * np.nansum((samples.imag - jk_mean.imag[None]) ** 2, axis=0)
    )
    valid = np.zeros((len(tseps), max_tsep + 1), dtype=bool)
    for idt, tsep in enumerate(tseps):
        valid[idt, : int(tsep) + 1] = True
    error_real[:, ~valid, :] = np.nan
    error_imag[:, ~valid, :] = np.nan
    return {
        "ratio": central,
        "ratio_jackknife_samples": samples,
        "ratio_jackknife_mean": jk_mean,
        "ratio_jackknife_bias_corrected": bias_corrected,
        "ratio_jackknife_error_real": error_real,
        "ratio_jackknife_error_imag": error_imag,
        "valid_insertion_mask": valid,
    }


def validate_against_plus_reference(
    results: dict[str, np.ndarray], reference: Path, z_index: int
) -> dict[str, float | str]:
    if not reference.is_file():
        return {"status": "reference_not_found", "path": str(reference)}
    with np.load(reference, allow_pickle=False) as data:
        labels = {
            key: [str(x) for x in data[key]]
            for key in ("channel_labels", "z_orientation_labels", "component_labels")
        }
        index = (
            labels["channel_labels"].index("helicity"),
            labels["z_orientation_labels"].index("odd_difference"),
            labels["component_labels"].index("combined"),
            0,
            z_index,
        )
        ref_ratio = np.asarray(data["ratio"])[index]
        ref_er = np.asarray(data["ratio_jackknife_error_real"])[index]
        ref_ei = np.asarray(data["ratio_jackknife_error_imag"])[index]
    ours = results["ratio"][SELECTORS.index("TH_plus_SH")]
    ours_er = results["ratio_jackknife_error_real"][SELECTORS.index("TH_plus_SH")]
    ours_ei = results["ratio_jackknife_error_imag"][SELECTORS.index("TH_plus_SH")]
    diffs = {
        "ratio_max_abs_diff": float(np.nanmax(np.abs(ours - ref_ratio))),
        "error_real_max_abs_diff": float(np.nanmax(np.abs(ours_er - ref_er))),
        "error_imag_max_abs_diff": float(np.nanmax(np.abs(ours_ei - ref_ei))),
    }
    tolerance = 5e-12
    return {
        "status": "pass" if max(diffs.values()) < tolerance else "fail",
        "path": str(reference.resolve()),
        "sha256": sha256_file(reference),
        "tolerance": tolerance,
        **diffs,
    }


def midpoint_projection(
    array: np.ndarray, samples: np.ndarray, tseps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average the one/two central insertion points, preserving paired JK samples."""
    central = []
    replicas = []
    for idt, tsep in enumerate(tseps):
        lo, hi = int(tsep) // 2, (int(tsep) + 1) // 2
        central.append(array[:, idt, [lo, hi]].mean(axis=1))
        replicas.append(samples[:, :, idt, [lo, hi]].mean(axis=2))
    central_out = np.stack(central, axis=1)
    samples_out = np.stack(replicas, axis=2)
    jk_mean = samples_out.mean(axis=0)
    prefactor = (samples_out.shape[0] - 1) / samples_out.shape[0]
    error_real = np.sqrt(
        prefactor * np.sum((samples_out.real - jk_mean.real[None]) ** 2, axis=0)
    )
    error_imag = np.sqrt(
        prefactor * np.sum((samples_out.imag - jk_mean.imag[None]) ** 2, axis=0)
    )
    return central_out, samples_out, error_real, error_imag


def make_plots(
    output_stem: Path,
    results: dict[str, np.ndarray],
    tseps: np.ndarray,
    pabs: np.ndarray,
    tau: float,
    z_index: int,
    detail_tsep: int,
    plot_tsep_max: int,
) -> list[str]:
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})
    colors = ("#1f77b4", "#d62728", "#2ca02c")
    labels = (r"$(T_H-S_H)_{\rm odd}$", r"$(T_H+S_H)_{\rm odd}$", r"$(T_H)_{\rm odd}$")
    central = results["ratio"]
    samples = results["ratio_jackknife_samples"]
    mid, _, mid_err_real, mid_err_imag = midpoint_projection(central, samples, tseps)
    show = tseps <= plot_tsep_max
    if not np.any(show):
        raise ValueError("plot-tsep-max excludes every source--sink separation")

    projections = (
        ("real", np.real, mid_err_real, r"$\operatorname{Re} R_{\Delta g}$"),
        ("imag", np.imag, mid_err_imag, r"$\operatorname{Im} R_{\Delta g}$"),
    )
    fig, axes = plt.subplots(2, len(pabs), figsize=(12.2, 7.0), sharex=True)
    for row, (_, project, midpoint_error, ylabel) in enumerate(projections):
        for ip, momentum in enumerate(pabs):
            ax = axes[row, ip]
            for selector, (label, color) in enumerate(zip(labels, colors)):
                ax.errorbar(
                    tseps[show] + 0.07 * (selector - 1),
                    project(mid[selector, show, ip]),
                    yerr=midpoint_error[selector, show, ip],
                    marker="o",
                    ms=3.5,
                    lw=1.0,
                    capsize=2,
                    color=color,
                    label=label,
                )
            ax.axhline(0.0, color="black", lw=0.7)
            if row == 0:
                ax.set_title(rf"$P_z={momentum}\,(2\pi/L)$")
            if row == 1:
                ax.set_xlabel(r"$t_{\rm sep}/a$")
            if ip == 0:
                ax.set_ylabel(ylabel + " at midpoint")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        rf"Helicity selector comparison: $\tau/a^2={tau:g}$, $z/a={z_index}$"
    )
    fig.tight_layout()
    midpoint_stem = output_stem.with_name(
        output_stem.name
        + f"_midpoint_real_imag_tsep{int(tseps[show].min())}to{int(tseps[show].max())}_P345"
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(midpoint_stem.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)

    if detail_tsep not in tseps:
        raise ValueError(f"detail-tsep={detail_tsep} is absent from {tseps.tolist()}")
    idt = int(np.where(tseps == detail_tsep)[0][0])
    detail_stems = []
    for projection_name, project, _, ylabel in projections:
        error = results[f"ratio_jackknife_error_{projection_name}"]
        fig, axes = plt.subplots(
            3, len(pabs), figsize=(12.2, 8.6), sharex=True, sharey="row"
        )
        for selector, (label, color) in enumerate(zip(labels, colors)):
            for ip, momentum in enumerate(pabs):
                ax = axes[selector, ip]
                x = np.arange(detail_tsep + 1)
                ax.errorbar(
                    x,
                    project(central[selector, idt, : detail_tsep + 1, ip]),
                    yerr=error[selector, idt, : detail_tsep + 1, ip],
                    marker="o",
                    ms=3.2,
                    lw=1.0,
                    capsize=2,
                    color=color,
                )
                ax.axhline(0.0, color="black", lw=0.7)
                if selector == 0:
                    ax.set_title(rf"$P_z={momentum}\,(2\pi/L)$")
                if ip == 0:
                    ax.set_ylabel(label + "\n" + ylabel)
                if selector == 2:
                    ax.set_xlabel(r"$t_{\rm ins}/a$")
        fig.suptitle(
            rf"Helicity insertion-time comparison ({projection_name} part): "
            rf"$\tau/a^2={tau:g}$, $z/a={z_index}$, "
            rf"$t_{{\rm sep}}/a={detail_tsep}$"
        )
        fig.tight_layout()
        detail_stem = output_stem.with_name(
            output_stem.name + f"_tsep{detail_tsep}_{projection_name}_P345"
        )
        for suffix in (".png", ".pdf"):
            fig.savefig(detail_stem.with_suffix(suffix), dpi=220, bbox_inches="tight")
        plt.close(fig)
        detail_stems.append(detail_stem)
    return [
        str(midpoint_stem.with_suffix(suffix).resolve())
        for suffix in (".png", ".pdf")
    ] + [str(stem.with_suffix(suffix).resolve()) for stem in detail_stems for suffix in (".png", ".pdf")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ope-root", type=Path, default=DEFAULT_OPE_ROOT)
    parser.add_argument("--twopt-root", type=Path, default=DEFAULT_TWOPT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison"))
    parser.add_argument("--flow-tau", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--z", type=int, default=2)
    parser.add_argument("--pabs", default="3,4,5")
    parser.add_argument("--tseps", default="5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--detail-tsep", type=int, default=8)
    parser.add_argument("--plot-tsep-max", type=int, default=11)
    args = parser.parse_args()
    pabs = np.asarray([int(x) for x in args.pabs.split(",")], dtype=np.int32)
    tseps = np.asarray([int(x) for x in args.tseps.split(",")], dtype=np.int32)
    if args.z < 0 or args.z >= NZ:
        raise ValueError(f"z must be in [0,{NZ - 1}]")
    if len(set(pabs.tolist())) != len(pabs) or np.any(pabs <= 0):
        raise ValueError("pabs values must be unique and positive")
    if len(set(tseps.tolist())) != len(tseps) or np.any(tseps < 0) or np.any(tseps >= NT):
        raise ValueError("tseps must be unique and satisfy 0<=tsep<NT")

    start = time.time()
    confs, manifest_sha = read_manifest(args.manifest)
    operators, pol35, nopol = load_inputs(
        confs, args.ope_root, args.twopt_root, args.flow_tau, args.epsilon,
        args.z, pabs, tseps,
    )
    results = calculate(operators, pol35, nopol, tseps)
    regression = validate_against_plus_reference(results, args.reference, args.z)
    if regression["status"] == "fail":
        raise RuntimeError(f"Plus-sign regression failed: {regression}")

    tag = tau_tag(args.flow_tau)
    stem = args.output_dir / f"helicity_three_operators_{tag}_N{len(confs)}_z{args.z}"
    contract = {
        "schema": SCHEMA,
        "status": "finite_flow_bare_disconnected_ratio_operator_diagnostic",
        "theory_selector": "TH_minus_SH",
        "diagnostic_selectors": ["TH_plus_SH", "TH_only"],
        "selector_definitions": {
            "TH_minus_SH": "(Mtiti-Mijij)[odd_difference]",
            "TH_plus_SH": "(Mtiti+Mijij)[odd_difference]",
            "TH_only": "Mtiti[odd_difference]",
        },
        "odd_difference": "raw(+z)-raw(-z), no factor 1/2",
        "nconf": len(confs),
        "confs": confs,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "ope_root": str(args.ope_root.resolve()),
        "twopt_root": str(args.twopt_root.resolve()),
        "flow_tau_t_over_a2": args.flow_tau,
        "flow_epsilon": args.epsilon,
        "field_projection": "traceless",
        "z_over_a": args.z,
        "pabs": pabs.tolist(),
        "tseps": tseps.tolist(),
        "numerator_c2": "full complex pol35",
        "denominator_c2": "full complex nopol",
        "estimator": "mean_tsrc[<O*C2_pol35>-<O><C2_pol35>] / mean_tsrc[<C2_nopol>]",
        "resampling": "shared delete-one jackknife; subtraction and denominator recomputed per sample",
        "display_projection": "Im(R)=Re(-iR); complex products retained",
        "selection_warning": "data quality does not determine the continuum operator sign",
        "reference_regression": regression,
        "not_applied": [
            "state_isolation_fit", "renormalization", "zero_flow_limit",
            "gluon_singlet_quark_mixing", "target_mass_correction", "matching",
        ],
    }
    payload = dict(results)
    payload.update(
        schema=np.asarray(SCHEMA),
        status=np.asarray(contract["status"]),
        contract_json=np.asarray(json.dumps(contract, sort_keys=True)),
        confs=np.asarray(confs),
        selector_labels=np.asarray(SELECTORS),
        selector_coefficients_TH_SH=COEFFICIENTS,
        pabs_list=pabs,
        tsep_values=tseps,
        insertion_values=np.arange(int(tseps.max()) + 1, dtype=np.int32),
        axes_ratio=np.asarray(["selector", "tsep", "insertion", "momentum_abs"]),
        axes_jackknife=np.asarray(
            ["jackknife_sample", "selector", "tsep", "insertion", "momentum_abs"]
        ),
    )
    npz = stem.with_suffix(".npz")
    output_sha = atomic_npz(npz, payload)
    plots = make_plots(
        stem, results, tseps, pabs, args.flow_tau, args.z,
        args.detail_tsep, args.plot_tsep_max,
    )
    done = dict(contract)
    done.update(
        output=str(npz.resolve()),
        output_sha256=output_sha,
        plots=plots,
        elapsed_seconds=time.time() - start,
        result_shapes={key: list(value.shape) for key, value in results.items()},
    )
    atomic_json(Path(str(npz) + ".done.json"), done)
    print(json.dumps({"output": str(npz), "regression": regression}, indent=2))


if __name__ == "__main__":
    main()
