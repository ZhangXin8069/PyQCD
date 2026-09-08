#!/usr/bin/env python3
"""Strict validator for gradient_flow_gluon_ratio_v2 products."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product", type=Path)
    args = parser.parse_args()
    done_path = Path(str(args.product) + ".done.json")
    if not args.product.is_file() or not done_path.is_file():
        raise SystemExit("missing NPZ or done JSON")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("output_sha256") != sha256_file(args.product):
        raise SystemExit("output SHA-256 mismatch")

    with np.load(args.product, allow_pickle=False) as data:
        schema_npz = str(data["schema"])
        channels = data["channel_labels"].astype(str).tolist()
        orientations = data["z_orientation_labels"].astype(str).tolist()
        components = data["component_labels"].astype(str).tolist()
        directions = data["direction_labels"].astype(str).tolist()
        tseps = data["tsep_values"].astype(int).tolist()
        momenta = data["pabs_list"].astype(int).tolist()
        z_values = data["z_values"].astype(int).tolist()
        valid_mask = data["valid_insertion_mask"]
        c3 = data["c3_mean"]
        c2 = data["c2_mean"]
        ratio = data["ratio"]
        jk_mean = data["ratio_jackknife_mean"]
        jk_bias = data["ratio_jackknife_bias_corrected"]
        err_r = data["ratio_jackknife_error_real"]
        err_i = data["ratio_jackknife_error_imag"]
        contract = json.loads(str(data["contract_json"]))

    expected = (
        len(channels), len(orientations), len(components), len(directions),
        len(z_values), len(tseps), max(tseps) + 1, len(momenta),
    )
    checks = {
        "schema_npz": schema_npz == "gradient_flow_gluon_ratio_v2",
        "schema_done": done.get("schema") == "gradient_flow_gluon_ratio_v2",
        "schema_contract": contract.get("schema") == "gradient_flow_gluon_ratio_v2",
        "status": done.get("status") == "finite_flow_bare_disconnected_c3_over_c2",
        "thin_link": done.get("input_scheme") == "thin_link_no_hyp_no_smear",
        "labels": channels == ["unpolarized", "helicity"]
        and orientations == ["plus_z_raw", "minus_z_raw", "even_sum", "odd_difference"],
        "required_components": all(name in components for name in ("combined", "Mtiti", "Mijij")),
        "shape": c3.shape == expected and ratio.shape == expected
        and jk_mean.shape == expected and jk_bias.shape == expected
        and err_r.shape == expected and err_i.shape == expected,
        "c2_shape": c2.shape == (len(channels), len(directions), len(tseps), len(momenta)),
        "nopol_denominator": done.get("denominator_polarization") == "nopol"
        and np.allclose(c2[0], c2[1], rtol=0, atol=0),
        "mask_shape": valid_mask.shape == (len(tseps), max(tseps) + 1),
        "c2_finite": bool(np.isfinite(c2).all()),
    }
    expanded_mask = np.broadcast_to(
        valid_mask[None, None, None, None, None, :, :, None], expected
    )
    checks["central_finite_valid"] = bool(np.isfinite(c3[expanded_mask]).all())
    checks["ratio_finite_valid"] = bool(np.isfinite(ratio[expanded_mask]).all())
    checks["central_nan_invalid"] = bool(np.isnan(c3[~expanded_mask]).all())
    checks["ratio_nan_invalid"] = bool(np.isnan(ratio[~expanded_mask]).all())
    if done.get("resampling") == "jackknife":
        valid = expanded_mask
        checks["jackknife_finite_valid"] = bool(
            np.isfinite(jk_mean[valid]).all() and np.isfinite(jk_bias[valid]).all()
            and np.isfinite(err_r[valid]).all() and np.isfinite(err_i[valid]).all()
        )
        checks["jackknife_nonnegative_error"] = bool(
            np.all(err_r[valid] >= 0) and np.all(err_i[valid] >= 0)
        )

    kw = {"rtol": 5e-12, "atol": 1e-10, "equal_nan": True}
    oi = {name: orientations.index(name) for name in orientations}
    ki = {name: components.index(name) for name in components}
    for name, array in (("c3", c3), ("ratio", ratio)):
        checks[f"{name}_even"] = bool(
            np.allclose(array[:, oi["even_sum"]], array[:, oi["plus_z_raw"]] + array[:, oi["minus_z_raw"]], **kw)
        )
        checks[f"{name}_odd"] = bool(
            np.allclose(array[:, oi["odd_difference"]], array[:, oi["plus_z_raw"]] - array[:, oi["minus_z_raw"]], **kw)
        )
        mt = array[:, :, ki["Mtiti"]]
        mi = array[:, :, ki["Mijij"]]
        combined = array[:, :, ki["combined"]]
        checks[f"{name}_unpolarized_components"] = bool(
            np.allclose(combined[0], mt[0] - mi[0], **kw)
        )
        checks[f"{name}_helicity_components"] = bool(
            np.allclose(combined[1], mt[1] + mi[1], **kw)
        )
        odd_z0 = array[1, oi["odd_difference"], :, :, 0]
        odd_z0_valid = np.broadcast_to(
            valid_mask[None, None, :, :, None], odd_z0.shape
        )
        checks[f"{name}_helicity_odd_z0"] = bool(
            np.allclose(odd_z0[odd_z0_valid], 0.0, **kw)
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("validation failed: " + ", ".join(failed))
    print(
        f"RATIO_VALIDATION_OK product={args.product} nconf={done['nconf']} "
        f"shape={ratio.shape} resampling={done['resampling']}"
    )


if __name__ == "__main__":
    main()
