#!/usr/bin/env python3
"""Prepare the 20-conf/eight-flow campaign; read-only audit of historical OPE."""
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from ope_components import (ROOT, REPO, ENSEMBLE, GAUGE_ROOT, TAUS, CONFS,
                            atomic_json, raw_base, sha256, validate_raw)

sys.path.insert(0, str(REPO / "scripts"))
from make_manifest import ildg_payload


def main():
    if (ROOT / "requests.json").exists():
        raise SystemExit("requests.json already exists; preserving the submitted manifest")
    for folder in ("logs", "locks", "raw", "results", "validation", "build"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    inputs = {}
    for conf in CONFS:
        gauge = (GAUGE_ROOT / f"{ENSEMBLE}_cfg_{conf}.lime").resolve()
        stat = gauge.stat()
        offset, payload = ildg_payload(gauge)
        if payload != 32**3 * 64 * 4 * 3 * 3 * 2 * 8 or offset + payload > stat.st_size:
            raise ValueError(f"invalid gauge payload: {gauge}")
        inputs[str(conf)] = dict(conf_id=str(conf), gauge_path=str(gauge),
                                input_size=stat.st_size, input_mtime_ns=stat.st_mtime_ns,
                                lime_payload_offset=offset, lime_payload_bytes=payload)
    requests = []
    for tau in TAUS:
        for conf in CONFS:
            req = dict(inputs[str(conf)], task_index=len(requests), lattice_xyzt=[32, 32, 32, 64],
                       a_fm=0.0897, tau=tau, epsilon=0.01, z_dir=2, z_count=25)
            base = raw_base(REPO / "output_L32x64_v1", req)
            req["legacy_base"] = str(base)
            if not any(p.exists() for p in (Path(str(base) + ".npy"), Path(str(base) + ".json"))):
                req["initial_action"] = "compute"
                req["legacy_audit"] = "missing"
            else:
                try:
                    validate_raw(base, req)
                    req["initial_action"] = "export"
                    req["legacy_audit"] = "validated_raw_v2"
                except Exception as error:
                    req["initial_action"] = "compute"
                    req["legacy_audit"] = "invalid_preserved: " + str(error)
            requests.append(req)
    atomic_json(ROOT / "requests.json", requests)
    audit = dict(created=datetime.now().isoformat(timespec="seconds"),
                 scope="20 configurations aligned with quark_l32x64_flow_20261007",
                 counts=dict(Counter(r["initial_action"] for r in requests)),
                 tau_counts={str(tau): dict(Counter(r["initial_action"] for r in requests
                                                   if r["tau"] == tau)) for tau in TAUS},
                 legacy_target_label="TH+SH is legacy diagnostic; raw primitives validated independently")
    atomic_json(ROOT / "initial_audit.json", audit)
    code_paths = [ROOT / name for name in ("ope_components.py", "run_task.py", "pilot.py", "test_components.py")]
    code_paths.append(REPO / "src/flowed_gluon_ope.cpp")
    atomic_json(ROOT / "campaign.json", {
        "ensemble": ENSEMBLE, "confs": list(CONFS), "tau_a2": list(TAUS), "a_fm": 0.0897,
        "requests_sha256": sha256(ROOT / "requests.json"),
        "code_sha256": {str(p): sha256(p) for p in code_paths},
        "cpu_partitions": ["a192c1t", "cpu6248R", "cpueicc", "i72c512g"],
        "z_dir": 2, "z_count": 25, "epsilon": 0.01,
    })
    with (ROOT / "manifest.tsv").open("w", newline="") as stream:
        fields = ["task_index", "conf_id", "tau", "epsilon", "z_dir", "z_count", "initial_action", "legacy_audit", "gauge_path"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(requests)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
