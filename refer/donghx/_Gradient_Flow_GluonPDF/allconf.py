#!/usr/bin/env python3
"""Frozen all-configuration CPU campaign; export valid raw, compute only gaps."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
COMPONENTS = REPO / "gluon_l32x64_ope_components_20261009"
sys.path.insert(0, str(COMPONENTS))
from ope_components import (ENSEMBLE, GAUGE_ROOT, PRIMITIVES, assert_input,
                            atomic_json, export_product, raw_base, require,
                            sha256, tau_tag, validate_product, validate_raw)
sys.path.insert(0, str(REPO / "scripts"))
from make_manifest import ildg_payload
import numpy as np

TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8)
BINARY = COMPONENTS / "build/flowed_gluon_ope_components"
PARTITIONS = "a192c1t,cpu6248R,cpueicc,i72c512g"


def now():
    return datetime.now().isoformat(timespec="seconds")


def destination(req):
    return ROOT / "results" / f"conf{req['conf_id']}" / tau_tag(req["tau"]) / "zdir2"


def pinned_paths():
    paths = [ROOT / name for name in ("allconf.py", "audit_cpu.slurm", "compute_cpu.slurm")]
    paths += [COMPONENTS / "ope_components.py", COMPONENTS / "build_provenance.json",
              COMPONENTS / "test_components.py", BINARY, REPO / "src/flowed_gluon_ope.cpp",
              REPO / "scripts/make_manifest.py"]
    paths += [REPO / "runs" / f"epsilon_gate_L32x64_conf1000_{tau_tag(t)}.json" for t in TAUS]
    return paths


def seal():
    require(not (ROOT / "campaign.json").exists(), "campaign already sealed")
    for name in ("logs", "validation", "locks", "raw", "results"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    provenance = json.loads((COMPONENTS / "build_provenance.json").read_text())
    require(sha256(BINARY) == provenance["binary_sha256"], "validated binary changed")
    require(sha256(COMPONENTS / "ope_components.py") ==
            provenance["source_sha256"][str(COMPONENTS / "ope_components.py")], "exporter changed")
    atomic_json(ROOT / "campaign.json", {
        "schema": "gradient_flow_gluon_allconf_cpu_v1", "created": now(),
        "ensemble": ENSEMBLE, "gauge_root": str(GAUGE_ROOT), "a_fm": 0.0897,
        "scope": "all top-level thin-link L32x64 .lime inputs at preparation time",
        "lattice_xyzt": [32, 32, 32, 64], "tau_t_over_a2": list(TAUS),
        "epsilon": 0.01, "fine_epsilon": 0.005, "z_dir": 2, "z_count": 25,
        "cpu_partitions": PARTITIONS.split(","), "cpus_per_compute_task": 48,
        "array_concurrency_cap": None, "slurm_max_array_size_observed": 1001,
        "code_and_evidence_sha256": {str(p): sha256(p) for p in pinned_paths()},
        "legacy_raw_root": str(REPO / "output_L32x64_v1"),
        "binary": str(BINARY), "validated_build": provenance,
        "physics_note": "same tau=t/a^2 as L32x96, not same physical t across lattice spacings",
    })
    print("SEALED", ROOT, flush=True)


def check_campaign():
    campaign = json.loads((ROOT / "campaign.json").read_text())
    for path, expected in campaign["code_and_evidence_sha256"].items():
        require(sha256(path) == expected, f"pinned code/evidence changed: {path}")
    return campaign


def check_inventory():
    inventory = json.loads((ROOT / "inventory.json").read_text())
    require(sha256(ROOT / "requests.json") == inventory["requests_sha256"], "requests changed")
    return inventory


def prepare():
    if (ROOT / "inventory.json").exists():
        check_inventory()
        return json.loads((ROOT / "requests.json").read_text())
    require(not (ROOT / "requests.json").exists(), "incomplete preparation; preserve and inspect")
    gauges = []
    for path in GAUGE_ROOT.glob("*.lime"):
        match = re.fullmatch(re.escape(ENSEMBLE) + r"_cfg_(\d+)\.lime", path.name)
        require(match is not None, f"unexpected input filename: {path}")
        gauges.append((int(match.group(1)), path.resolve()))
    gauges.sort()
    require(gauges and len({conf for conf, _ in gauges}) == len(gauges), "empty/duplicate inputs")
    inputs = []
    for conf, path in gauges:
        stat = path.stat()
        offset, size = ildg_payload(path)
        require(size == 32**3 * 64 * 4 * 3 * 3 * 2 * 8 and offset + size <= stat.st_size,
                f"wrong/truncated LIME payload: {path}")
        inputs.append({"conf_id": str(conf), "gauge_path": str(path),
                       "input_size": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
                       "lime_payload_offset": offset, "lime_payload_bytes": size})
    requests = []
    for tau in TAUS:
        for entry in inputs:
            req = dict(entry, task_index=len(requests), lattice_xyzt=[32, 32, 32, 64],
                       a_fm=0.0897, tau=tau, epsilon=0.01, z_dir=2, z_count=25)
            req["legacy_base"] = str(raw_base(REPO / "output_L32x64_v1", req))
            requests.append(req)
    atomic_json(ROOT / "requests.json", requests)
    atomic_json(ROOT / "inventory.json", {
        "created": now(), "conf_count": len(inputs), "confs": [x["conf_id"] for x in inputs],
        "first_conf": inputs[0]["conf_id"], "last_conf": inputs[-1]["conf_id"],
        "tau_count": len(TAUS), "request_count": len(requests),
        "requests_sha256": sha256(ROOT / "requests.json"), "inputs": inputs,
        "input_identity_policy": "path + size + mtime_ns + validated LIME payload header; not full gauge SHA256",
    })
    print(f"PREPARED confs={len(inputs)} requests={len(requests)}", flush=True)
    return requests


def epsilon_gates(requests):
    reports = []
    for tau in TAUS:
        req = next((r for r in requests if r["conf_id"] == "1000" and r["tau"] == tau), None)
        require(req is not None, "historical epsilon pilot conf1000 is absent")
        saved = json.loads((REPO / "runs" / f"epsilon_gate_L32x64_conf1000_{tau_tag(tau)}.json").read_text())
        require(saved["status"] == "passed", f"historical epsilon gate failed: {tau}")
        fine_req = dict(req, epsilon=0.005)
        coarse_base, fine_base = raw_base(REPO / "output_L32x64_v1", req), raw_base(REPO / "output_L32x64_v1", fine_req)
        require(str(coarse_base) == saved["coarse_base"] and str(fine_base) == saved["fine_base"], "gate source mismatch")
        assert_input(req)
        coarse, cm = validate_raw(coarse_base, req)
        fine, fm = validate_raw(fine_base, fine_req)
        rel = float(np.linalg.norm((coarse - fine).ravel()) / max(float(np.linalg.norm(fine.ravel())), 1e-300))
        pdiff = abs(cm["flowed_gauge_stats"]["plaquette"] - fm["flowed_gauge_stats"]["plaquette"])
        require(rel <= saved["max_relative_l2"] and pdiff <= saved["max_plaquette_diff"], f"epsilon gate recheck failed: {tau}")
        primitive_rel = {}
        for name, (channel, component) in PRIMITIVES.items():
            a, b = coarse[:, channel, :, component], fine[:, channel, :, component]
            primitive_rel[name] = float(np.linalg.norm((a - b).ravel()) / max(float(np.linalg.norm(b.ravel())), 1e-300))
        reports.append({"tau": tau, "status": "passed", "relative_l2_all_ope": rel,
                        "plaquette_abs_diff": pdiff, "relative_l2_primitives_diagnostic": primitive_rel,
                        "coarse_npy_sha256": sha256(str(coarse_base) + ".npy"),
                        "fine_npy_sha256": sha256(str(fine_base) + ".npy")})
    atomic_json(ROOT / "validation/epsilon_gates.json", {"created": now(), "gates": reports})
    print("EPSILON_GATES_REVALIDATED count=14", flush=True)


def audit(final=False):
    check_campaign()
    requests = prepare()
    check_inventory()
    if not final:
        require(not (ROOT / "submission.json").exists(), "production submitted; use --final instead")
        subprocess.run([str(BINARY), "--self-test"], check=True)
        epsilon_gates(requests)
    counts = Counter()
    by_tau = {str(t): Counter() for t in TAUS}
    missing = []
    report_path = ROOT / "validation" / ("final_audit_rows.jsonl" if final else "initial_audit_rows.jsonl")
    with report_path.open("w") as stream:
        for i, req in enumerate(requests):
            assert_input(req)
            status, reason = "validated_complete", None
            dest = destination(req)
            needs_export = False
            try:
                if (dest / "complete.json").exists():
                    validate_product(dest, req)
                elif final:
                    raise FileNotFoundError("no validated complete marker")
                else:
                    # Every reused raw receives full identity, finite-value, SU(3), and algebra checks.
                    validate_raw(req["legacy_base"], req)
                    needs_export = True
            except Exception as error:
                status, reason = "missing_or_invalid_preserved", f"{type(error).__name__}: {error}"
                missing.append(req["task_index"])
            if needs_export:
                # Output quota/IO errors stop the audit, rather than trigger unnecessary reflow.
                export_product(req["legacy_base"], req, dest)
                status = "legacy_validated_exported"
            counts[status] += 1
            by_tau[str(req["tau"])][status] += 1
            stream.write(json.dumps({"task_index": i, "conf_id": req["conf_id"], "tau": req["tau"],
                                     "status": status, "reason": reason}) + "\n")
            if (i + 1) % 500 == 0:
                stream.flush()
                print(f"AUDIT_PROGRESS {i+1}/{len(requests)} missing={len(missing)}", flush=True)
    summary = {"created": now(), "scope": "all actual thin-link inputs x all 14 L32x96 gluon tau values",
               "final": final, "counts": dict(counts), "request_count": len(requests),
               "conf_count": len({r["conf_id"] for r in requests}), "missing_count": len(missing),
               "per_tau": {t: dict(c) for t, c in by_tau.items()}, "missing_request_indices": missing,
               "rows_path": str(report_path), "rows_sha256": sha256(report_path)}
    atomic_json(ROOT / "validation" / ("final_audit.json" if final else "initial_audit.json"), summary)
    if final:
        require(not missing, f"FINAL_AUDIT_FAILED missing={len(missing)}; preserved diagnostics")
    else:
        groups = {}
        for index in missing:
            groups.setdefault(requests[index]["conf_id"], []).append(index)
        group_list = [{"conf_id": c, "request_indices": groups[c]} for c in sorted(groups, key=int)]
        atomic_json(ROOT / "missing_groups.json", group_list)
        atomic_json(ROOT / "validation/missing_manifest_identity.json", {
            "sha256": sha256(ROOT / "missing_groups.json"), "group_count": len(group_list)})
    print(json.dumps(summary, indent=2), flush=True)


def compute(req):
    dest = destination(req)
    if (dest / "complete.json").exists():
        try:
            validate_product(dest, req)
            return
        except Exception as error:
            print(f"PRESERVE_INVALID_PRODUCT {dest}: {error}", flush=True)
    base = raw_base(ROOT / "raw", req)
    try:
        validate_raw(base, req)
    except Exception:
        # Preserve partial/invalid new files before retry; legacy output is never modified.
        for suffix in (".npy", ".json"):
            path = Path(str(base) + suffix)
            if path.exists():
                quarantine = ROOT / "rejected" / f"{req['task_index']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
                quarantine.mkdir(parents=True, exist_ok=False)
                path.rename(quarantine / path.name)
        base.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(BINARY), "--input", req["gauge_path"], "--conf-id", req["conf_id"],
               "--output", str(base), "--nx", "32", "--ny", "32", "--nz", "32", "--nt", "64",
               "--tau", str(req["tau"]), "--epsilon", "0.01", "--z-dir", "2", "--z-count", "25"]
        print("COMPUTE " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        validate_raw(base, req)
    # An invalid completion marker must not block a repair; preserve the whole product directory.
    if (dest / "complete.json").exists():
        quarantine = ROOT / "rejected" / f"product_{req['task_index']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        dest.rename(quarantine)
    export_product(base, req, dest)
    validate_product(dest, req)


def run_group(index):
    check_campaign()
    check_inventory()
    frozen = json.loads((ROOT / "validation/missing_manifest_identity.json").read_text())
    require(sha256(ROOT / "missing_groups.json") == frozen["sha256"], "missing manifest changed")
    groups = json.loads((ROOT / "missing_groups.json").read_text())
    require(0 <= index < len(groups), "group out of range")
    group = groups[index]
    requests = json.loads((ROOT / "requests.json").read_text())
    with (ROOT / "locks" / f"conf{group['conf_id']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task_index in group["request_indices"]:
            req = requests[task_index]
            require(req["conf_id"] == group["conf_id"], "group conf mismatch")
            assert_input(req)
            compute(req)
    print(f"CONF_COMPLETED conf={group['conf_id']} gaps={len(group['request_indices'])}", flush=True)


def submit():
    check_campaign()
    check_inventory()
    require((ROOT / "validation/epsilon_gates.json").exists(), "CPU epsilon recheck required")
    groups = json.loads((ROOT / "missing_groups.json").read_text())
    frozen = json.loads((ROOT / "validation/missing_manifest_identity.json").read_text())
    require(sha256(ROOT / "missing_groups.json") == frozen["sha256"], "missing manifest changed")
    receipt = json.loads((ROOT / "submission.json").read_text()) if (ROOT / "submission.json").exists() else {
        "created": now(), "missing_group_count": len(groups),
        "missing_request_count": sum(len(g["request_indices"]) for g in groups), "batches": []}
    for offset in range(0, len(groups), 1000):
        if any(b["offset"] == offset for b in receipt["batches"]):
            continue
        size = min(1000, len(groups) - offset)
        cmd = ["sbatch", "--parsable", f"--array=0-{size-1}", f"--export=ALL,BATCH_OFFSET={offset}",
               str(ROOT / "compute_cpu.slurm")]
        job = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        receipt["batches"].append({"offset": offset, "size": size, "job_id": job, "command": cmd})
        atomic_json(ROOT / "submission.json", receipt)
        print(f"SUBMITTED missing-only offset={offset} count={size} job={job}", flush=True)
    if not receipt.get("final_audit_job"):
        cmd = ["sbatch", "--parsable", "--job-name=gf_g64_all_final", "--export=ALL,FINAL_AUDIT=1"]
        if receipt["batches"]:
            cmd += ["--dependency=afterok:" + ":".join(b["job_id"] for b in receipt["batches"])]
        cmd += [str(ROOT / "audit_cpu.slurm")]
        receipt["final_audit_job"] = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        receipt["final_audit_command"] = cmd
        atomic_json(ROOT / "submission.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("seal", "audit", "run-group", "submit"))
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--group-index", type=int)
    args = parser.parse_args()
    if args.action == "seal":
        seal()
    elif args.action == "audit":
        audit(args.final)
    elif args.action == "run-group":
        require(args.group_index is not None, "group index required")
        run_group(args.group_index)
    else:
        submit()


if __name__ == "__main__":
    main()
