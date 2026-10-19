#!/usr/bin/env python3
"""Audit all 30 configurations at tf/a^2=0.30 and 0.40."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFS = [
    *range(10000, 11000, 100),
    11000,
    11100,
    *range(12300, 13100, 100),
    *range(13100, 14100, 100),
]


def check_root(root: Path, confs: list[int], tau: float, window: int) -> dict:
    tag = f"thin_tf{tau:.8f}_eps0.01000000_tw{window}_Nv400_dz21".replace(".", "p")
    records = []
    for conf in confs:
        base = root / tag / "Nev_200" / str(conf)
        files = [base / f"corr_projected_tsep{t}_dz0-20_Nev200.npz" for t in range(5, 11)]
        missing = [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]
        records.append({"conf": conf, "missing": missing, "complete": not missing})
    return {"root": str(root), "tau": tau, "window": window, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root_specs = [
        (
            Path("/public/group/lqcd/donghx/Gradient_Flow/Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/thin_link"),
            list(range(10000, 11000, 100)),
        ),
        (
            Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_spin2_fill_flows_20260923/results_remaining"),
            [11000, 11100, *range(12300, 13100, 100)],
        ),
        (
            Path("/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_spin2_expansion30_20260925/results_all6"),
            list(range(13100, 14100, 100)),
        ),
    ]
    records = []
    for root, confs in root_specs:
        records.append(check_root(root, confs, 0.30, 3))
        records.append(check_root(root, confs, 0.40, 4))
    summary = {
        "requested_confs": CONFS,
        "requested_flow_times": [0.30, 0.40],
        "records": records,
        "complete": all(item["complete"] for group in records for item in group["records"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
