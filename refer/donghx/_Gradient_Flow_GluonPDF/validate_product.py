#!/usr/bin/env python3
"""Validate one all-operator finite-flow C3/C2 bootstrap product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from construct_bootstrap import OPERATOR_LABELS, sha256_file


def validate(product: Path) -> dict:
    receipt_path = product.with_suffix(".done.json")
    if not product.exists() or not receipt_path.exists():
        raise FileNotFoundError(f"missing product or receipt for {product}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError(f"receipt is not complete: {receipt_path}")
    if receipt.get("product_sha256") != sha256_file(product):
        raise ValueError(f"product SHA256 mismatch: {product}")

    with np.load(product, allow_pickle=False) as data:
        labels = tuple(str(x) for x in data["all_operator_labels"])
        if labels != OPERATOR_LABELS:
            raise ValueError(f"operator labels mismatch: {labels}")
        expected = {
            "c3_original": (6, 25, 11, 16, 3),
            "c2_original": (2, 11, 3),
            "ratio_original": (6, 25, 11, 16, 3),
            "c3_sum_original": (6, 3, 25, 11, 3),
            "c3_sum_bootstrap": (1000, 6, 3, 25, 11, 3),
            "c2_bootstrap": (1000, 2, 11, 3),
            "ratio_sum_bootstrap": (1000, 6, 3, 25, 11, 3),
            "bootstrap_indices": (1000, 406),
            "valid_cut_tsep_mask": (3, 11),
            "valid_insertion_mask": (11, 16),
        }
        for key, shape in expected.items():
            if tuple(data[key].shape) != shape:
                raise ValueError(f"{key} has shape {data[key].shape}, expected {shape}")

        c3 = data["c3_sum_bootstrap"]
        ratio = data["ratio_sum_bootstrap"]
        np.testing.assert_allclose(c3[:, 1] + c3[:, 2], 2.0 * c3[:, 0], rtol=2e-11, atol=2e-11, equal_nan=True)
        np.testing.assert_allclose(c3[:, 4] + c3[:, 5], 2.0 * c3[:, 3], rtol=2e-11, atol=2e-11, equal_nan=True)
        # The C3 relation is exact to accumulation precision.  Ratios perform
        # three independent floating-point divisions, so use a separate
        # double-precision division tolerance here.
        np.testing.assert_allclose(ratio[:, 1] + ratio[:, 2], 2.0 * ratio[:, 0], rtol=1e-7, atol=5e-9, equal_nan=True)
        np.testing.assert_allclose(ratio[:, 4] + ratio[:, 5], 2.0 * ratio[:, 3], rtol=1e-7, atol=5e-9, equal_nan=True)

        mask = data["valid_cut_tsep_mask"].astype(bool)
        valid_broadcast = np.broadcast_to(
            mask[None, None, :, None, :, None], ratio.shape
        )
        if not np.isfinite(ratio[valid_broadcast]).all():
            raise ValueError("non-finite values in valid summed-ratio bins")
        if np.isfinite(ratio[~valid_broadcast]).any():
            raise ValueError("finite values found in invalid summed-ratio bins")
        if int(data["nboot"]) != 1000 or int(data["nconf"]) != 406:
            raise ValueError("product is not the requested N=406, Nboot=1000 product")

    with np.load(product, allow_pickle=False) as data:
        flow_tau = float(data["flow_tau"])
    return {
        "status": "complete_and_validated",
        "product": str(product),
        "product_sha256": sha256_file(product),
        "flow_tau": flow_tau,
        "nconf": 406,
        "nboot": 1000,
        "operator_labels": list(OPERATOR_LABELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("product", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.product.resolve()), indent=2))


if __name__ == "__main__":
    main()
