#!/usr/bin/env python3
"""Construct all six finite-flow gluon-OPE C3/C2 operators and bootstrap them.

The product deliberately derives every operator from the raw Mtiti/Mijij
components.  In particular, the legacy ``combined`` helicity component is not
used, because the current OPE source stores it with the plus sign while the
physical helicity target is a separately chosen T_H +/- S_H combination.

This is a finite-flow, bare, disconnected C3/C2 product.  It is not a
renormalized or matched PDF result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff" / "Gradient_flow_gluon"
if str(HANDOFF) not in sys.path:
    sys.path.insert(0, str(HANDOFF))

from calc_ratio_gradient_flow import (  # noqa: E402
    load_inputs,
    read_manifest,
    sha256_file,
    tau_tag,
)


DEFAULT_MANIFEST = (
    ROOT / "ratio_bootstrap_latest_v1" / "manifests" / "common_all14_plusz_P345.tsv"
)
DEFAULT_OPE_ROOT = ROOT / "output_v5"
DEFAULT_TWOPT_ROOT = Path(
    "/public/group/lqcd/donghx/2pt_Result/"
    "beta6.41_mu-0.2295_ms-0.2050_L32x96"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results_v1"
DEFAULT_NBOOT = 1000
DEFAULT_SEED = 1115
EXPECTED_NCONF = 406
EXPECTED_TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)

OPERATOR_LABELS = (
    "unpolarized_TU_even",
    "unpolarized_TU_plus_SU_even",
    "unpolarized_TU_minus_SU_even",
    "helicity_TH_odd",
    "helicity_TH_plus_SH_odd",
    "helicity_TH_minus_SH_odd",
)
OPERATOR_C2_CHANNEL = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int64)
OPERATOR_PROJECTION = ("real", "real", "real", "imag", "imag", "imag")
OPERATOR_DEFINITIONS = {
    "unpolarized_TU_even": "T_U = U_tx + U_ty",
    "unpolarized_TU_plus_SU_even": "T_U + S_U, S_U = 2 U_xy",
    "unpolarized_TU_minus_SU_even": "T_U - S_U, S_U = 2 U_xy",
    "helicity_TH_odd": "T_H = H_tx + H_ty",
    "helicity_TH_plus_SH_odd": "T_H + S_H, S_H = 2 H_xy",
    "helicity_TH_minus_SH_odd": "T_H - S_H, S_H = 2 H_xy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-tau", type=float, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nboot", type=int, default=DEFAULT_NBOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="replace only this requested output")
    return parser.parse_args()


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing product: {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_operator_arrays(ope: np.ndarray) -> np.ndarray:
    """Return (configuration, operator, z, insertion-time) complex arrays."""

    if ope.ndim != 6 or ope.shape[1:3] != (2, 4) or ope.shape[3] != 3:
        raise ValueError(f"unexpected OPE shape {ope.shape}; expected (N,2,4,3,z,t)")

    # OPE axes: channel (unpolarized/helicity), z orientation (even/odd are
    # indices 2/3), component (combined/Mtiti/Mijij), z, t.
    tu = np.asarray(ope[:, 0, 2, 1], dtype=np.complex128)
    su = np.asarray(ope[:, 0, 2, 2], dtype=np.complex128)
    th = np.asarray(ope[:, 1, 3, 1], dtype=np.complex128)
    sh = np.asarray(ope[:, 1, 3, 2], dtype=np.complex128)

    # These checks document the source convention but do not use the source
    # combined field for the six requested products.
    np.testing.assert_allclose(ope[:, 0, 2, 0], tu - su, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(ope[:, 1, 3, 0], th + sh, rtol=2e-12, atol=2e-12)

    operators = np.stack((tu, tu + su, tu - su, th, th + sh, th - sh), axis=1)
    if not np.isfinite(operators).all():
        raise ValueError("non-finite values found in raw OPE operator arrays")
    return operators


def _project(value: np.ndarray, projection: str) -> np.ndarray:
    if projection == "real":
        return np.asarray(value.real, dtype=np.float64)
    if projection == "imag":
        return np.asarray(value.imag, dtype=np.float64)
    raise ValueError(f"unknown projection {projection}")


def pointwise_central(
    operators: np.ndarray,
    twopt: np.ndarray,
    tseps: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the central pointwise disconnected C3 and C3/C2 ratio."""

    nconf, noperator, nz, nt = operators.shape
    if noperator != len(OPERATOR_LABELS):
        raise ValueError("operator axis does not contain all six requested operators")
    if twopt.shape[:2] != (2, nconf):
        raise ValueError(f"unexpected two-point shape {twopt.shape}")

    nsep = len(tseps)
    npol = twopt.shape[-1]
    c2_mean_source = twopt.mean(axis=1)  # (channel, tsep, source-time, projection)
    c2_original = c2_mean_source.mean(axis=2)  # (channel, tsep, projection)
    denominator = c2_original[0].real
    max_tsep = int(np.max(tseps))
    c3_original = np.full(
        (noperator, nz, nsep, max_tsep + 1, npol),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    ratio_original = np.full(
        (noperator, nz, nsep, max_tsep + 1, npol),
        np.nan,
        dtype=np.float64,
    )
    valid_insertion = np.zeros((nsep, max_tsep + 1), dtype=bool)

    source = np.arange(nt, dtype=np.int64)
    for it, tsep in enumerate(tseps):
        if tsep >= nt:
            raise ValueError(f"tsep={tsep} is outside insertion-time extent nt={nt}")
        valid_insertion[it, : int(tsep) + 1] = True
        for insertion in range(int(tsep) + 1):
            shifted = np.take(operators, (source + insertion) % nt, axis=-1)
            shifted_mean = shifted.mean(axis=0)
            for ioperator, channel in enumerate(OPERATOR_C2_CHANNEL):
                own = np.einsum(
                    "nzs,nsp->nzp",
                    shifted[:, ioperator],
                    twopt[channel, :, it],
                    optimize=True,
                ).mean(axis=0) / nt
                disconnected = np.einsum(
                    "zs,sp->zp",
                    shifted_mean[ioperator],
                    c2_mean_source[channel, it],
                    optimize=True,
                ) / nt
                c3_original[ioperator, :, it, insertion] = own - disconnected
                ratio_original[ioperator, :, it, insertion] = (
                    _project(c3_original[ioperator, :, it, insertion], OPERATOR_PROJECTION[ioperator])
                    / denominator[it][None, :]
                )

    return {
        "c3_original": c3_original,
        "c2_original": c2_original,
        "ratio_original": ratio_original,
        "valid_insertion_mask": valid_insertion,
    }


def bootstrap_summed(
    operators: np.ndarray,
    twopt: np.ndarray,
    tseps: np.ndarray,
    cuts: np.ndarray,
    nboot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap summed C3/C2 for all six operators.

    Row zero is the deterministic central average.  The remaining rows are
    the requested ``nboot`` multinomial bootstrap replicas.  Vacuum
    subtraction and the denominator are recomputed in every replica.
    """

    nconf, noperator, nz, nt = operators.shape
    nsep = len(tseps)
    ncut = len(cuts)
    npol = twopt.shape[-1]
    nrep = nboot + 1
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, nconf, size=(nboot, nconf), dtype=np.int64)
    weights = np.zeros((nrep, nconf), dtype=np.float64)
    weights[0, :] = 1.0 / nconf
    for replica, row in enumerate(bootstrap_indices, start=1):
        weights[replica] = np.bincount(row, minlength=nconf) / nconf

    # Bootstrap means for every source-time slice.
    operator_flat = operators.reshape(nconf, -1)
    mean_operator = (weights @ operator_flat).reshape(nrep, noperator, nz, nt)
    twopt_flat = np.transpose(twopt, (1, 0, 2, 3, 4)).reshape(nconf, -1)
    mean_c2_source = (weights @ twopt_flat).reshape(nrep, 2, nsep, nt, npol)
    c2_replica = mean_c2_source.mean(axis=3)
    denominator = c2_replica[:, 0].real

    pairs: list[tuple[int, int]] = []
    for icut, cut in enumerate(cuts):
        for it, tsep in enumerate(tseps):
            if int(tsep) >= 2 * int(cut):
                pairs.append((icut, it))
    npair = len(pairs)

    c3_sum = np.full(
        (nrep, noperator, ncut, nz, nsep, npol),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    ratio_sum = np.full((nrep, noperator, ncut, nz, nsep, npol), np.nan, dtype=np.float64)
    valid_pair = np.zeros((ncut, nsep), dtype=bool)
    source = np.arange(nt, dtype=np.int64)

    # Compute the connected-minus-vacuum-subtracted insertion sums pair by
    # pair.  The source-product tensor is kept at one pair at a time to avoid
    # multiplying the full bootstrap weight matrix by a large temporary.
    for icut, it in pairs:
        cut = int(cuts[icut])
        tsep = int(tseps[it])
        valid_pair[icut, it] = True
        insertions = np.arange(cut, tsep - cut + 1, dtype=np.int64)
        op_sum = np.zeros((nconf, noperator, nz, nt), dtype=np.complex128)
        for insertion in insertions:
            op_sum += np.take(operators, (source + insertion) % nt, axis=-1)

        own_product = np.zeros((nconf, noperator, nz, npol), dtype=np.complex128)
        for ioperator, channel in enumerate(OPERATOR_C2_CHANNEL):
            own_product[:, ioperator] = np.einsum(
                "nzs,nsp->nzp",
                op_sum[:, ioperator],
                twopt[channel, :, it],
                optimize=True,
            ) / nt
        own_flat = own_product.reshape(nconf, -1)
        weighted_own = (weights @ own_flat).reshape(nrep, noperator, nz, npol)

        for ioperator, channel in enumerate(OPERATOR_C2_CHANNEL):
            disconnected = np.einsum(
                "bzs,bsp->bzp",
                np.take(
                    mean_operator[:, ioperator],
                    (source[None, :] + insertions[:, None]) % nt,
                    axis=-1,
                ).sum(axis=-2),
                mean_c2_source[:, channel, it],
                optimize=True,
            ) / nt
            c3_sum[:, ioperator, icut, :, it] = weighted_own[:, ioperator] - disconnected
            ratio_sum[:, ioperator, icut, :, it] = (
                _project(c3_sum[:, ioperator, icut, :, it], OPERATOR_PROJECTION[ioperator])
                / denominator[:, it, None, :]
            )

    # Invalid cut/tsep pairs intentionally remain NaN in both arrays.
    c3_sum_original = c3_sum[0]
    ratio_sum_original = ratio_sum[0]
    c3_sum_bootstrap = c3_sum[1:]
    ratio_sum_bootstrap = ratio_sum[1:]
    c3_sum_mean = np.full_like(c3_sum_original, np.nan + 1j * np.nan)
    c3_sum_error = np.full_like(c3_sum_original.real, np.nan)
    ratio_sum_mean = np.full_like(ratio_sum_original, np.nan)
    ratio_sum_error = np.full_like(ratio_sum_original, np.nan)
    for icut, it in pairs:
        c3_values = c3_sum_bootstrap[:, :, icut, :, it, :]
        ratio_values = ratio_sum_bootstrap[:, :, icut, :, it, :]
        c3_sum_mean[:, icut, :, it, :] = c3_values.mean(axis=0)
        c3_sum_error[:, icut, :, it, :] = c3_values.std(axis=0, ddof=1)
        ratio_sum_mean[:, icut, :, it, :] = ratio_values.mean(axis=0)
        ratio_sum_error[:, icut, :, it, :] = ratio_values.std(axis=0, ddof=1)

    return {
        "bootstrap_indices": bootstrap_indices,
        "c2_bootstrap": c2_replica[1:],
        "c3_sum_original": c3_sum_original,
        "c3_sum_bootstrap": c3_sum_bootstrap,
        "c3_sum_mean": c3_sum_mean,
        "c3_sum_error": c3_sum_error,
        "ratio_sum_original": ratio_sum_original,
        "ratio_sum_bootstrap": ratio_sum_bootstrap,
        "ratio_sum_mean": ratio_sum_mean,
        "ratio_sum_error": ratio_sum_error,
        "valid_cut_tsep_mask": valid_pair,
    }


def product_paths(output: Path) -> tuple[Path, Path]:
    return output, output.with_suffix(".done.json")


def main() -> None:
    args = parse_args()
    if args.nboot < 2:
        raise ValueError("--nboot must be at least 2")
    if not any(np.isclose(args.flow_tau, tau) for tau in EXPECTED_TAUS):
        raise ValueError(f"unexpected flow tau {args.flow_tau}; expected one of {EXPECTED_TAUS}")

    manifest = args.manifest.resolve()
    records, manifest_sha = read_manifest(manifest)
    if len(records) != EXPECTED_NCONF:
        raise ValueError(
            f"frozen common manifest must contain {EXPECTED_NCONF} records, got {len(records)}"
        )

    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_DIR / (
            f"c3_c2_bootstrap_allops_{tau_tag(args.flow_tau)}"
            f"_N{len(records)}_Nboot{args.nboot}.npz"
        )
    output = output.resolve()
    done = output.with_suffix(".done.json")
    if (output.exists() or done.exists()) and not args.force:
        raise FileExistsError(
            f"product or receipt already exists ({output}, {done}); use a new path"
        )
    if args.force:
        raise ValueError("--force is intentionally unsupported; use a new versioned output path")

    tau = float(args.flow_tau)
    tseps = np.arange(5, 16, dtype=np.int64)
    cuts = np.asarray((1, 2, 3), dtype=np.int64)
    ope, twopt = load_inputs(
        records,
        DEFAULT_OPE_ROOT,
        DEFAULT_TWOPT_ROOT,
        tau,
        0.01,
        2,
        96,
        25,
        "traceless",
        ["combined", "Mtiti", "Mijij"],
        ["+z"],
        [3, 4, 5],
        list(map(int, tseps)),
        2,
        "_Cg5g4",
    )
    operators = build_operator_arrays(ope)
    twopt = twopt[:, 0]  # remove the single +z direction axis

    input_meta = {
        "ope_root": str(DEFAULT_OPE_ROOT),
        "twopt_root": str(DEFAULT_TWOPT_ROOT),
        "epsilon": 0.01,
        "z_dir": 2,
        "nt": 96,
        "nz": 25,
        "element": "_Cg5g4",
    }

    central = pointwise_central(operators, twopt, tseps)
    summed = bootstrap_summed(operators, twopt, tseps, cuts, args.nboot, args.seed)
    for icut, cut in enumerate(cuts):
        for it, tsep in enumerate(tseps):
            if not summed["valid_cut_tsep_mask"][icut, it]:
                continue
            interval = slice(int(cut), int(tsep) - int(cut) + 1)
            np.testing.assert_allclose(
                summed["ratio_sum_original"][:, icut, :, it],
                np.nansum(central["ratio_original"][:, :, it, interval], axis=2),
                rtol=1e-9,
                atol=1e-9,
                err_msg="summed central ratio does not reproduce pointwise insertion sum",
            )

    # The three operator choices are linear combinations with a common C2,
    # so these relations must hold for central and every bootstrap replica.
    np.testing.assert_allclose(
        summed["c3_sum_bootstrap"][:, 1] + summed["c3_sum_bootstrap"][:, 2],
        2.0 * summed["c3_sum_bootstrap"][:, 0],
        rtol=2e-11,
        atol=2e-11,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        summed["c3_sum_bootstrap"][:, 4] + summed["c3_sum_bootstrap"][:, 5],
        2.0 * summed["c3_sum_bootstrap"][:, 3],
        rtol=2e-11,
        atol=2e-11,
        equal_nan=True,
    )

    manifest_sha = sha256_file(manifest)
    source_code_sha = sha256_file(HANDOFF / "calc_ratio_gradient_flow.py")
    this_code_sha = sha256_file(Path(__file__).resolve())
    contract = {
        "schema": "gradient_flow_gluon_c3_c2_bootstrap_all_operators_v1",
        "status": "finite_flow_bare_disconnected_c3_c2_bootstrap_all_operators",
        "physics_scope": [
            "four-dimensional Wilson-flowed thin-link gluon OPE",
            "bare disconnected nucleon C3/C2 estimator",
            "not renormalized, matched, continuum-extrapolated, or a physical PDF",
        ],
        "operator_labels": list(OPERATOR_LABELS),
        "operator_definitions": OPERATOR_DEFINITIONS,
        "operator_c2_channel": {
            label: ("nopol" if int(channel) == 0 else "pol35")
            for label, channel in zip(OPERATOR_LABELS, OPERATOR_C2_CHANNEL)
        },
        "operator_projection": dict(zip(OPERATOR_LABELS, OPERATOR_PROJECTION)),
        "operator_source_components": {
            "unpolarized": "raw OPE channel 0, even-z orientation, Mtiti/Mijij components",
            "helicity": "raw OPE channel 1, odd-z orientation, Mtiti/Mijij components",
            "legacy_combined": "checked only for source-convention identities; not used to define products",
        },
        "estimator": {
            "vacuum_subtraction": "<O C2> - <O><C2>",
            "c2_denominator": "real(<C2_nopol>)",
            "ratio": "Re(C3)/Re(C2_nopol) for unpolarized; Im(C3)/Re(C2_nopol) for helicity",
            "bootstrap": "multinomial configuration bootstrap; central row excluded from Nboot statistics",
        },
        "flow_tau": tau,
        "flow_radius_convention": "r_F/a = sqrt(8*tau)",
        "field_projection": "traceless",
        "components": ["combined", "Mtiti", "Mijij"],
        "z_orientation": "+z only; even for unpolarized, odd for helicity",
        "momentum_abs": [3, 4, 5],
        "tsep": [int(x) for x in tseps],
        "cuts": [int(x) for x in cuts],
        "nconf": len(records),
        "nboot": args.nboot,
        "seed": args.seed,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "source_calculation_sha256": source_code_sha,
        "constructor_sha256": this_code_sha,
        "input_meta": input_meta,
    }

    arrays: dict[str, Any] = {
        **central,
        **summed,
        "all_operator_labels": np.asarray(OPERATOR_LABELS),
        "target_operator_labels": np.asarray(OPERATOR_LABELS),
        "operator_c2_channel": OPERATOR_C2_CHANNEL,
        "operator_projection": np.asarray(OPERATOR_PROJECTION),
        "tsep": tseps,
        "cuts": cuts,
        "flow_tau": np.asarray(tau),
        "nconf": np.asarray(len(records), dtype=np.int64),
        "nboot": np.asarray(args.nboot, dtype=np.int64),
        "seed": np.asarray(args.seed, dtype=np.int64),
        "manifest_sha256": np.asarray(manifest_sha),
        "contract_json": np.asarray(json.dumps(contract, sort_keys=True)),
    }
    atomic_savez(output, **arrays)
    receipt = {
        "status": "complete",
        "product": str(output),
        "product_sha256": sha256_file(output),
        "product_bytes": output.stat().st_size,
        "contract": contract,
        "array_shapes": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
    }
    atomic_write_json(done, receipt)
    print(json.dumps({"status": "complete", "product": str(output), "receipt": str(done)}, indent=2))


if __name__ == "__main__":
    main()
