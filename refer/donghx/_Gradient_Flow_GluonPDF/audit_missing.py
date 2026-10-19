#!/usr/bin/env python3
"""Audit the 20 x 8 L32x64 production grid without reading large C2/C3 arrays."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFS = [10000 + 40 * index for index in range(20)]
FLOWS = [(0.05, 2), (0.10, 2), (0.20, 3), (0.30, 3),
         (0.40, 4), (0.50, 4), (0.80, 6), (1.00, 7)]
TSEPS = (5, 6, 7, 8, 9, 10)


def tag(tau, window):
    return f"thin_tf{tau:.8f}_eps0.01000000_tw{window}_Nv400_dz21".replace(".", "p")


def completed_statuses():
    done = set()
    for path in (ROOT / "status").glob("full_chain_*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("state") != "COMPLETED":
            continue
        try:
            key = (int(record["conf"]), round(float(record["flow_time"]), 2))
        except (KeyError, TypeError, ValueError):
            continue
        done.add(key)
    return done


def product_state(conf, tau, window, statuses):
    if (conf, tau) not in statuses:
        return "no_completed_status"
    output = ROOT / "results" / tag(tau, window) / "Nev_200" / str(conf)
    for tsep in TSEPS:
        path = output / f"corr_projected_tsep{tsep}_dz0-20_Nev200.npz"
        if not path.is_file() or path.stat().st_size < 1_000_000_000:
            return f"missing_or_short_tsep{tsep}"
    for name in ("spin2_provenance.json", "timing_summary.json"):
        path = output / name
        if not path.is_file():
            return f"missing_{name}"
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return f"invalid_{name}"
        if str(payload.get("conf")) != str(conf):
            return f"wrong_conf_{name}"
        if name == "spin2_provenance.json" and payload.get("tseps") != list(TSEPS):
            return "wrong_tseps"
    return "complete"


def audit():
    statuses = completed_statuses()
    complete, missing = [], []
    for flow_index, (tau, window) in enumerate(FLOWS):
        for conf_index, conf in enumerate(CONFS):
            task = 20 * flow_index + conf_index
            state = product_state(conf, tau, window, statuses)
            item = {"task": task, "conf": conf, "tau_a2": tau, "window": window}
            if state == "complete":
                complete.append(item)
            else:
                missing.append({**item, "reason": state})
    return {"schema": "l32x64_flow_missing_audit_v1", "complete": complete,
            "missing": missing, "array_spec": ",".join(str(item["task"]) for item in missing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="save the audit as JSON")
    args = parser.parse_args()
    report = audit()
    if args.manifest:
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"complete={len(report['complete'])} missing={len(report['missing'])}")
    print(f"array_spec={report['array_spec']}")


if __name__ == "__main__":
    main()
