#!/usr/bin/env python3
"""Disconnected C3/C2 ratios from schema-v2 gradient-flow gluon OPE loops.

The estimator follows the historical ``Calc_ratio.py`` convention, but reads
the versioned gradient-flow OPE tensor directly and preserves its channel,
orientation and component labels.  At every source time and insertion time,

    C3 = <O C2>_cfg - <O>_cfg <C2>_cfg,
    R  = C3 / <C2_nopol>_cfg,

followed by an average over source time.  The C3 covariance uses nopol for the
unpolarized channel and pol35 for helicity, while both ratios are normalized
by nopol C2.  Delete-one jackknife recomputes both the vacuum subtraction and
denominator inside every resample.

These are finite-flow bare disconnected ratios.  No renormalization, target
mass correction, zero-flow extrapolation, mixing, or matching is applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


DEFAULT_OPE_ROOT = Path(
    "/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/output_v5"
)
DEFAULT_TWOPT_ROOT = Path(
    "/public/group/lqcd/donghx/2pt_Result/"
    "beta6.41_mu-0.2295_ms-0.2050_L32x96"
)

NT = 96
NZ = 25
OPE_SCHEMA = "gradient_flow_gluon_ope_v2"
RATIO_SCHEMA = "gradient_flow_gluon_ratio_v2"

FIELD_PROJECTIONS = ("legacy_untraced", "traceless")
CHANNELS = ("unpolarized", "helicity")
ORIENTATIONS = ("plus_z_raw", "minus_z_raw", "even_sum", "odd_difference")
ALL_COMPONENTS = ("combined", "Mtiti", "Mijij", "ti", "tj", "ij_single")
DEFAULT_COMPONENTS = ("combined", "Mtiti", "Mijij")
POLARIZATION_FOR_CHANNEL = {"unpolarized": "nopol", "helicity": "pol35"}


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.replace(" ", ",").split(",") if item.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def tau_tag(tau: float) -> str:
    return f"tau{tau:.3f}".replace(".", "p")


def direction_sign(direction: str) -> int:
    if direction == "+z":
        return +1
    if direction == "-z":
        return -1
    raise ValueError(f"Only +z/-z are supported by the current z_dir=2 OPE: {direction}")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t"))
    if not rows or "conf_id" not in rows[0]:
        raise ValueError(f"Manifest must contain a conf_id column: {path}")
    confs = [str(row["conf_id"]) for row in rows]
    if len(confs) != len(set(confs)):
        raise ValueError("Duplicate configuration IDs in manifest")
    if len(confs) < 2:
        raise ValueError("At least two configurations are required")
    return confs, digest


def ope_base(root: Path, conf: str, flow_tau: float, epsilon: float, z_dir: int) -> Path:
    tag = tau_tag(flow_tau)
    return (
        root
        / f"conf{conf}"
        / tag
        / f"conf{conf}_{tag}_eps{epsilon:.3f}_zdir{z_dir}"
    )


def twopt_path(
    root: Path,
    conf: str,
    direction: str,
    momentum_abs: int,
    mom_phase_abs: int,
    element: str,
    polarization: str,
) -> Path:
    sign = direction_sign(direction)
    pz = sign * momentum_abs
    phase = sign * mom_phase_abs
    return (
        root
        / f"momsmear{phase}z"
        / conf
        / (
            f"twopt_slice_pp_Px0Py0Pz{pz}_eginphase{phase}"
            f"{element}_{polarization}_ss_conf{conf}.npy"
        )
    )


def validate_ope_metadata(
    metadata: dict,
    conf: str,
    flow_tau: float,
    epsilon: float,
    z_dir: int,
    expected_shape: tuple[int, ...],
) -> None:
    checks = {
        "schema": metadata.get("schema") == OPE_SCHEMA,
        "status": metadata.get("status") == "flowed_bare_ope_observable",
        "conf_id": str(metadata.get("conf_id")) == conf,
        "thin_link": metadata.get("input_scheme") == "thin_link_no_hyp_no_smear",
        "tau": abs(float(metadata.get("flow", {}).get("tau_t_over_a2", -1)) - flow_tau)
        < 1e-12,
        "epsilon": abs(float(metadata.get("flow", {}).get("epsilon", -1)) - epsilon)
        < 1e-12,
        "z_dir": int(metadata.get("operator", {}).get("z_dir", -1)) == z_dir,
        "shape": tuple(metadata.get("shape", ())) == expected_shape,
        "axes": metadata.get("axes")
        == ["field_projection", "channel", "z_orientation", "component", "z", "t"],
        "field_projection_labels": metadata.get("axis_labels", {}).get("field_projection")
        == list(FIELD_PROJECTIONS),
        "channel_labels": metadata.get("axis_labels", {}).get("channel") == list(CHANNELS),
        "orientation_labels": metadata.get("axis_labels", {}).get("z_orientation")
        == list(ORIENTATIONS),
        "component_labels": metadata.get("axis_labels", {}).get("component")
        == list(ALL_COMPONENTS),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"OPE metadata failed {failed} for conf {conf}")


def load_ope(
    root: Path,
    conf: str,
    flow_tau: float,
    epsilon: float,
    z_dir: int,
    nt: int,
    nz: int,
    field_projection: str,
    components: list[str],
) -> np.ndarray:
    """Return axes (channel,orientation,component,z,t)."""
    base = ope_base(root, conf, flow_tau, epsilon, z_dir)
    npy = Path(str(base) + ".npy")
    metadata_path = Path(str(base) + ".json")
    if not npy.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing paired OPE artifact: {base}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_shape = (2, 2, 4, 6, nz, nt)
    validate_ope_metadata(metadata, conf, flow_tau, epsilon, z_dir, expected_shape)
    data = np.load(npy, mmap_mode="r", allow_pickle=False)
    if data.shape != expected_shape or data.dtype != np.dtype("complex128"):
        raise ValueError(f"Wrong OPE shape/dtype in {npy}: {data.shape}, {data.dtype}")
    pidx = FIELD_PROJECTIONS.index(field_projection)
    kidx = [ALL_COMPONENTS.index(name) for name in components]
    selected = np.asarray(data[pidx, :, :, kidx, :, :], dtype=np.complex128)
    # NumPy advanced indexing moves component to the front; restore C,O,K,Z,T.
    if selected.shape == (len(kidx), 2, 4, nz, nt):
        selected = np.transpose(selected, (1, 2, 0, 3, 4))
    expected_selected = (2, 4, len(kidx), nz, nt)
    if selected.shape != expected_selected or not np.isfinite(selected).all():
        raise ValueError(f"Invalid selected OPE tensor in {npy}: {selected.shape}")
    return selected


def load_twopt_separations(path: Path, nt: int, tseps: list[int]) -> np.ndarray:
    """Return axes (tsep,t_source)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.squeeze(np.asarray(np.load(path, allow_pickle=False)))
    if data.shape != (nt, nt) or data.dtype.kind != "c":
        raise ValueError(
            f"Expected ({nt},{nt}) complex C2 in {path}, got {data.shape} {data.dtype}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite C2 data in {path}")
    # Existing L32x96 products contain both complex64 and complex128 files.
    # Promote at the I/O boundary so every estimator uses one accumulation dtype.
    data = np.asarray(data, dtype=np.complex128)
    source = np.arange(nt)
    return np.stack([data[(source + dt) % nt, source] for dt in tseps], axis=0)


def load_inputs(
    confs: list[str],
    ope_root: Path,
    twopt_root: Path,
    flow_tau: float,
    epsilon: float,
    z_dir: int,
    nt: int,
    nz: int,
    field_projection: str,
    components: list[str],
    directions: list[str],
    pabs_list: list[int],
    tseps: list[int],
    mom_phase_abs: int,
    element: str,
) -> tuple[np.ndarray, np.ndarray]:
    # OPE: conf,channel,orientation,component,z,t
    ope = np.empty(
        (len(confs), 2, 4, len(components), nz, nt), dtype=np.complex128
    )
    # C2: channel,direction,conf,tsep,t_source,momentum
    twopt = np.empty(
        (2, len(directions), len(confs), len(tseps), nt, len(pabs_list)),
        dtype=np.complex128,
    )
    for iconf, conf in enumerate(confs):
        if iconf % 20 == 0 or iconf + 1 == len(confs):
            print(f"LOAD {iconf + 1}/{len(confs)} conf={conf}", flush=True)
        ope[iconf] = load_ope(
            ope_root, conf, flow_tau, epsilon, z_dir, nt, nz,
            field_projection, components,
        )
        for channel_index, channel in enumerate(CHANNELS):
            polarization = POLARIZATION_FOR_CHANNEL[channel]
            for direction_index, direction in enumerate(directions):
                for momentum_index, momentum_abs in enumerate(pabs_list):
                    path = twopt_path(
                        twopt_root, conf, direction, momentum_abs,
                        mom_phase_abs, element, polarization,
                    )
                    twopt[channel_index, direction_index, iconf, :, :, momentum_index] = (
                        load_twopt_separations(path, nt, tseps)
                    )
    return ope, twopt


def covariance_ratio_estimator(
    operator: np.ndarray,
    correlator: np.ndarray,
    do_jackknife: bool,
    denominator_correlator: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """One insertion estimator.

    operator:   (conf,observable,t_source)
    correlator: (conf,t_source,momentum), with the channel projector used in C3
    denominator_correlator: optional unpolarized C2 used only in the ratio

    Returns central C3, central C2, jackknife mean, bias-corrected jackknife
    ratio, and separate real/imaginary jackknife standard errors.  The last
    three arrays are NaN when jackknife is disabled.
    """
    op = np.asarray(operator, dtype=np.complex128)
    c2 = np.asarray(correlator, dtype=np.complex128)
    denominator = (
        c2
        if denominator_correlator is None
        else np.asarray(denominator_correlator, dtype=np.complex128)
    )
    if op.ndim != 3 or c2.ndim != 3 or op.shape[0] != c2.shape[0] or op.shape[2] != c2.shape[1]:
        raise ValueError(f"Estimator shape mismatch: operator={op.shape}, C2={c2.shape}")
    if denominator.shape != c2.shape:
        raise ValueError(
            f"Denominator C2 shape mismatch: numerator={c2.shape}, denominator={denominator.shape}"
        )
    nconf = op.shape[0]
    if nconf < 2:
        raise ValueError("At least two configurations are required")

    nsource = op.shape[2]
    sum_o = np.sum(op, axis=0)  # observable,source
    sum_c = np.sum(c2, axis=0)  # source,momentum
    sum_denominator = np.sum(denominator, axis=0)  # source,momentum
    # Contract configuration and source-time axes directly.  This is
    # algebraically identical to forming O*C2 explicitly, but avoids a large
    # (Nconf,Nobs,Nsource,Np) temporary.
    total_oc_mean_source = np.einsum(
        "nos,nsp->op", op, c2, optimize=True
    ) / nsource
    total_o_times_c_mean_source = np.einsum(
        "os,sp->op", sum_o, sum_c, optimize=True
    ) / nsource
    c3_central = (
        total_oc_mean_source / nconf
        - total_o_times_c_mean_source / (nconf * nconf)
    )
    c2_central = np.mean(sum_denominator, axis=0) / nconf
    ratio_central = c3_central / c2_central[None, :]

    shape = ratio_central.shape
    nan_complex = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    nan_real = np.full(shape, np.nan, dtype=np.float64)
    if not do_jackknife:
        return c3_central, c2_central, nan_complex, nan_complex.copy(), nan_real, nan_real.copy()

    nminus = nconf - 1
    # Exact delete-one covariance after analytically expanding
    # (sum_o-o_i)*(sum_c-c_i).  All source-time contractions are performed
    # before the Nconf axis is materialized, reducing the largest temporary
    # from Nconf*Nobs*Nsource*Np to Nconf*Nobs*Np.
    own_oc_mean_source = np.einsum(
        "nos,nsp->nop", op, c2, optimize=True
    ) / nsource
    own_o_times_total_c = np.einsum(
        "nos,sp->nop", op, sum_c, optimize=True
    ) / nsource
    total_o_times_own_c = np.einsum(
        "os,nsp->nop", sum_o, c2, optimize=True
    ) / nsource
    c3_jackknife = (
        (total_oc_mean_source[None, ...] - own_oc_mean_source) / nminus
        - (
            total_o_times_c_mean_source[None, ...]
            - own_o_times_total_c
            - total_o_times_own_c
            + own_oc_mean_source
        )
        / (nminus * nminus)
    )
    c2_jackknife = (
        np.mean(sum_denominator, axis=0)[None, :]
        - np.mean(denominator, axis=1)
    ) / nminus
    ratio_jackknife = c3_jackknife / c2_jackknife[:, None, :]
    jackknife_mean = np.mean(ratio_jackknife, axis=0)
    bias_corrected = nconf * ratio_central - nminus * jackknife_mean
    prefactor = nminus / nconf
    error_real = np.sqrt(
        prefactor * np.sum((ratio_jackknife.real - jackknife_mean.real) ** 2, axis=0)
    )
    error_imag = np.sqrt(
        prefactor * np.sum((ratio_jackknife.imag - jackknife_mean.imag) ** 2, axis=0)
    )
    return c3_central, c2_central, jackknife_mean, bias_corrected, error_real, error_imag


def calculate_ratios(
    ope: np.ndarray,
    twopt: np.ndarray,
    tseps: list[int],
    do_jackknife: bool,
) -> dict[str, np.ndarray]:
    """Return arrays with explicit C,O,K,D,Z,tsep,insertion,P axes."""
    nconf, nch, norient, ncomp, nz, nt = ope.shape
    nch2, ndir, nconf2, ndt, nt2, npmom = twopt.shape
    if (nch2, nconf2, nt2) != (nch, nconf, nt):
        raise ValueError(f"Input tensor mismatch: OPE={ope.shape}, C2={twopt.shape}")
    max_tsep = max(tseps)
    out_shape = (nch, norient, ncomp, ndir, nz, ndt, max_tsep + 1, npmom)
    c3 = np.full(out_shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ratio = np.full_like(c3, np.nan + 1j * np.nan)
    jk_mean = np.full_like(c3, np.nan + 1j * np.nan)
    jk_bias = np.full_like(c3, np.nan + 1j * np.nan)
    jk_err_real = np.full(out_shape, np.nan, dtype=np.float64)
    jk_err_imag = np.full(out_shape, np.nan, dtype=np.float64)
    c2_mean = np.empty((nch, ndir, ndt, npmom), dtype=np.complex128)

    source = np.arange(nt)
    for channel in range(nch):
        for direction in range(ndir):
            c2_by_dt = twopt[channel, direction]  # conf,tsep,source,momentum
            denominator_by_dt = twopt[CHANNELS.index("unpolarized"), direction]
            c2_mean[channel, direction] = np.mean(denominator_by_dt, axis=(0, 2))
            for insertion in range(max_tsep + 1):
                shifted = np.take(
                    ope[:, channel], (source + insertion) % nt, axis=-1
                )
                flat = shifted.reshape(nconf, norient * ncomp * nz, nt)
                for dt_index, dt in enumerate(tseps):
                    if insertion > dt:
                        continue
                    print(
                        f"EST channel={CHANNELS[channel]} direction={direction} "
                        f"tsep={dt} insertion={insertion}",
                        flush=True,
                    )
                    result = covariance_ratio_estimator(
                        flat,
                        c2_by_dt[:, dt_index],
                        do_jackknife,
                        denominator_correlator=denominator_by_dt[:, dt_index],
                    )
                    c3_one, c2_one, mean_one, bias_one, err_r_one, err_i_one = result
                    shape = (norient, ncomp, nz, npmom)
                    target = (channel, slice(None), slice(None), direction, slice(None), dt_index, insertion, slice(None))
                    c3[target] = c3_one.reshape(shape)
                    ratio[target] = (c3_one / c2_one[None, :]).reshape(shape)
                    jk_mean[target] = mean_one.reshape(shape)
                    jk_bias[target] = bias_one.reshape(shape)
                    jk_err_real[target] = err_r_one.reshape(shape)
                    jk_err_imag[target] = err_i_one.reshape(shape)

    valid_insertion = np.zeros((ndt, max_tsep + 1), dtype=bool)
    for index, dt in enumerate(tseps):
        valid_insertion[index, : dt + 1] = True
    return {
        "c3_mean": c3,
        "c2_mean": c2_mean,
        "ratio": ratio,
        "ratio_jackknife_mean": jk_mean,
        "ratio_jackknife_bias_corrected": jk_bias,
        "ratio_jackknife_error_real": jk_err_real,
        "ratio_jackknife_error_imag": jk_err_imag,
        "valid_insertion_mask": valid_insertion,
    }


def atomic_save_npz(path: Path, compressed: bool, payload: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    save = np.savez_compressed if compressed else np.savez
    save(temporary, **payload)
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ope-root", type=Path, default=DEFAULT_OPE_ROOT)
    parser.add_argument("--twopt-root", type=Path, default=DEFAULT_TWOPT_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flow-tau", required=True, type=float)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--z-dir", default=2, type=int)
    parser.add_argument("--nt", default=NT, type=int)
    parser.add_argument("--nz", default=NZ, type=int)
    parser.add_argument("--field-projection", choices=FIELD_PROJECTIONS, default="traceless")
    parser.add_argument("--components", default=",".join(DEFAULT_COMPONENTS))
    parser.add_argument("--directions", default="+z")
    parser.add_argument("--pabs-list", default="3,4,5")
    parser.add_argument("--tseps", default="5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--mom-phase-abs", default=2, type=int)
    parser.add_argument("--element", default="_Cg5g4")
    parser.add_argument("--resampling", choices=("none", "jackknife"), default="jackknife")
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args()

    components = parse_csv(args.components)
    directions = parse_csv(args.directions)
    pabs_list = parse_int_csv(args.pabs_list)
    tseps = parse_int_csv(args.tseps)
    if not components or any(name not in ALL_COMPONENTS for name in components):
        raise ValueError(f"Unknown/empty components: {components}")
    if len(components) != len(set(components)):
        raise ValueError("Duplicate component labels")
    if not directions or any(name not in ("+z", "-z") for name in directions):
        raise ValueError(f"Unknown/empty directions: {directions}")
    if len(directions) != len(set(directions)):
        raise ValueError("Duplicate direction labels")
    if not pabs_list or min(pabs_list) <= 0 or len(pabs_list) != len(set(pabs_list)):
        raise ValueError("pabs-list must contain unique positive integers")
    if not tseps or min(tseps) < 0 or max(tseps) >= args.nt or len(tseps) != len(set(tseps)):
        raise ValueError("tseps must be unique and satisfy 0<=tsep<Nt")
    if args.z_dir != 2:
        raise ValueError("Current production contains only z_dir=2")

    confs, manifest_sha256 = read_manifest(args.manifest)
    print(f"CONTRACT nconf={len(confs)} manifest_sha256={manifest_sha256}")
    start = time.time()
    ope, twopt = load_inputs(
        confs, args.ope_root, args.twopt_root, args.flow_tau, args.epsilon,
        args.z_dir, args.nt, args.nz, args.field_projection, components,
        directions, pabs_list, tseps, args.mom_phase_abs, args.element,
    )
    print(f"INPUT_SHAPES ope={ope.shape} twopt={twopt.shape}", flush=True)
    results = calculate_ratios(
        ope, twopt, tseps, do_jackknife=args.resampling == "jackknife"
    )

    contract = {
        "schema": RATIO_SCHEMA,
        "status": "finite_flow_bare_disconnected_c3_over_c2",
        "nconf": len(confs),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "ope_root": str(args.ope_root.resolve()),
        "twopt_root": str(args.twopt_root.resolve()),
        "source_ope_schema": OPE_SCHEMA,
        "input_scheme": "thin_link_no_hyp_no_smear",
        "lattice_xyzt": [32, 32, 32, args.nt],
        "flow_tau_t_over_a2": args.flow_tau,
        "flow_epsilon": args.epsilon,
        "z_dir": args.z_dir,
        "z_values_a": list(range(args.nz)),
        "field_projection": args.field_projection,
        "channels": list(CHANNELS),
        "z_orientations": list(ORIENTATIONS),
        "components": components,
        "directions": directions,
        "pabs_list": pabs_list,
        "tsep_values": tseps,
        "polarization_for_channel": POLARIZATION_FOR_CHANNEL,
        "denominator_polarization": "nopol",
        "mom_phase_abs": args.mom_phase_abs,
        "element": args.element,
        "resampling": args.resampling,
        "jackknife_definition": (
            "delete-one; vacuum subtraction and C2 denominator recomputed in each sample; "
            "errors sqrt((N-1)/N sum(theta_i-mean_jk)^2), real/imag separately"
        ),
        "estimator": (
            "mean_tsrc[<O*C2_projected(channel)>_cfg-<O>_cfg"
            "<C2_projected(channel)>_cfg] / mean_tsrc[<C2_nopol>_cfg]"
        ),
        "physical_target_labels": {
            "unpolarized": "combined with even_sum (stored sum, no factor 1/2)",
            "helicity": "combined with odd_difference=m(+z)-m(-z)",
        },
        "not_applied": [
            "renormalization", "zero_flow_time_extrapolation", "target_mass_correction",
            "gluon_quark_mixing", "LaMET_or_pseudoPDF_matching",
        ],
    }
    payload: dict[str, np.ndarray] = dict(results)
    payload.update(
        schema=np.asarray(RATIO_SCHEMA),
        status=np.asarray(contract["status"]),
        contract_json=np.asarray(json.dumps(contract, sort_keys=True)),
        confs=np.asarray(confs),
        channel_labels=np.asarray(CHANNELS),
        z_orientation_labels=np.asarray(ORIENTATIONS),
        component_labels=np.asarray(components),
        direction_labels=np.asarray(directions),
        pabs_list=np.asarray(pabs_list, dtype=np.int32),
        tsep_values=np.asarray(tseps, dtype=np.int32),
        insertion_values=np.arange(max(tseps) + 1, dtype=np.int32),
        z_values=np.arange(args.nz, dtype=np.int32),
        axes_c3_ratio=np.asarray(
            ["channel", "z_orientation", "component", "direction", "z", "tsep", "insertion", "momentum_abs"]
        ),
        axes_c2=np.asarray(["channel", "direction", "tsep", "momentum_abs"]),
    )
    output_sha256 = atomic_save_npz(args.output, args.compressed, payload)
    done = dict(contract)
    done.update(
        output=str(args.output.resolve()),
        output_sha256=output_sha256,
        elapsed_seconds=time.time() - start,
        result_shapes={key: list(value.shape) for key, value in results.items()},
    )
    done_path = Path(str(args.output) + ".done.json")
    atomic_save_json(done_path, done)
    print(f"OUTPUT_OK path={args.output} sha256={output_sha256}")
    print(f"ELAPSED_SECONDS {done['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
