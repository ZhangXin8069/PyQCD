#!/usr/bin/env python3
"""Direct epsilon-halving gate for the real-data pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-base", required=True, type=Path)
    parser.add_argument("--fine-base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-relative-l2", default=5e-3, type=float)
    parser.add_argument("--max-plaquette-diff", default=5e-5, type=float)
    args = parser.parse_args()

    coarse = np.load(Path(str(args.coarse_base) + ".npy"), allow_pickle=False)
    fine = np.load(Path(str(args.fine_base) + ".npy"), allow_pickle=False)
    if coarse.shape != fine.shape or coarse.dtype != fine.dtype:
        raise SystemExit("epsilon comparison shape/dtype mismatch")
    coarse_meta = json.loads(Path(str(args.coarse_base) + ".json").read_text())
    fine_meta = json.loads(Path(str(args.fine_base) + ".json").read_text())
    denom = float(np.linalg.norm(fine.ravel()))
    relative_l2 = float(np.linalg.norm((coarse - fine).ravel()) / max(denom, 1e-300))
    coarse_plaq = float(coarse_meta["flowed_gauge_stats"]["plaquette"])
    fine_plaq = float(fine_meta["flowed_gauge_stats"]["plaquette"])
    plaquette_diff = abs(coarse_plaq - fine_plaq)
    passed = relative_l2 <= args.max_relative_l2 and plaquette_diff <= args.max_plaquette_diff
    report = {
        "schema": "gradient_flow_gluon_ope_epsilon_gate_v1",
        "status": "passed" if passed else "failed",
        "coarse_base": str(args.coarse_base.resolve()),
        "fine_base": str(args.fine_base.resolve()),
        "relative_l2_all_ope": relative_l2,
        "max_relative_l2": args.max_relative_l2,
        "plaquette_abs_diff": plaquette_diff,
        "max_plaquette_diff": args.max_plaquette_diff,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    if not passed:
        raise SystemExit(f"epsilon gate failed: {report}")
    print(f"EPSILON_GATE_OK relative_l2={relative_l2:.6g} plaquette_diff={plaquette_diff:.6g}")


if __name__ == "__main__":
    main()

