#!/usr/bin/env python3
"""Strict completion and component-identity guard for schema-v2 OPE data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("--conf-id", required=True)
    parser.add_argument("--z-count", required=True, type=int)
    parser.add_argument("--nt", default=96, type=int)
    parser.add_argument("--tau", required=True, type=float)
    parser.add_argument("--epsilon", required=True, type=float)
    args = parser.parse_args()

    npy = Path(str(args.base) + ".npy")
    metadata_path = Path(str(args.base) + ".json")
    if not npy.is_file() or not metadata_path.is_file():
        raise SystemExit("missing NPY or JSON artifact")

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    data = np.load(npy, mmap_mode="r", allow_pickle=False)
    expected_shape = (2, 2, 4, 6, args.z_count, args.nt)
    raw_stats = metadata.get("raw_gauge_stats", {})
    flowed_stats = metadata.get("flowed_gauge_stats", {})
    raw_plaquette = float(raw_stats.get("plaquette", np.nan))
    flowed_plaquette = float(flowed_stats.get("plaquette", np.nan))

    # axes: projection, channel, orientation, component, z, t
    plus, minus, even, odd = (data[:, :, index] for index in range(4))
    combined = data[:, :, :, 0]
    mtiti = data[:, :, :, 1]
    mijij = data[:, :, :, 2]
    ti = data[:, :, :, 3]
    tj = data[:, :, :, 4]
    ij_single = data[:, :, :, 5]
    algebra_kw = {"rtol": 2e-13, "atol": 1e-10}

    checks = {
        "schema": metadata.get("schema") == "gradient_flow_gluon_ope_v2",
        "status": metadata.get("status") == "flowed_bare_ope_observable",
        "conf_id": str(metadata.get("conf_id")) == str(args.conf_id),
        "tau": abs(float(metadata.get("flow", {}).get("tau_t_over_a2", -1)) - args.tau) < 1e-12,
        "epsilon": abs(float(metadata.get("flow", {}).get("epsilon", -1)) - args.epsilon) < 1e-12,
        "shape_metadata": tuple(metadata.get("shape", ())) == expected_shape,
        "shape_data": data.shape == expected_shape,
        "dtype": data.dtype == np.dtype("complex128"),
        "finite": bool(np.isfinite(data).all()),
        "thin_input": metadata.get("input_scheme") == "thin_link_no_hyp_no_smear",
        "axes": metadata.get("axes") == [
            "field_projection", "channel", "z_orientation", "component", "z", "t"
        ],
        "components": metadata.get("axis_labels", {}).get("component")
        == ["combined", "Mtiti", "Mijij", "ti", "tj", "ij_single"],
        "orientations": metadata.get("axis_labels", {}).get("z_orientation")
        == ["plus_z_raw", "minus_z_raw", "even_sum", "odd_difference"],
        "raw_group": float(raw_stats.get("max_unitarity_frobenius", np.inf)) < 5e-5
        and float(raw_stats.get("max_abs_det_minus_1", np.inf)) < 5e-5,
        "flowed_group": float(flowed_stats.get("max_unitarity_frobenius", np.inf)) < 5e-4
        and float(flowed_stats.get("max_abs_det_minus_1", np.inf)) < 5e-4,
        "plaquette_finite": bool(np.isfinite(raw_plaquette) and np.isfinite(flowed_plaquette)),
        "wilson_flow_direction": flowed_plaquette + 1e-7 >= raw_plaquette,
        "z0_direction_limit": bool(np.allclose(plus[..., 0, :], minus[..., 0, :], **algebra_kw)),
        "even_definition": bool(np.allclose(even, plus + minus, **algebra_kw)),
        "odd_definition": bool(np.allclose(odd, plus - minus, **algebra_kw)),
        "odd_z0_zero": bool(np.allclose(odd[..., 0, :], 0.0, **algebra_kw)),
        "mtiti_definition": bool(np.allclose(mtiti, ti + tj, **algebra_kw)),
        "mijij_definition": bool(np.allclose(mijij, 2.0 * ij_single, **algebra_kw)),
        "unpolarized_combined": bool(
            np.allclose(combined[:, 0], mtiti[:, 0] - mijij[:, 0], **algebra_kw)
        ),
        "helicity_combined": bool(
            np.allclose(combined[:, 1], mtiti[:, 1] + mijij[:, 1], **algebra_kw)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("validation failed: " + ", ".join(failed))
    print(f"VALIDATION_V5_OK base={args.base} shape={data.shape}")


if __name__ == "__main__":
    main()
