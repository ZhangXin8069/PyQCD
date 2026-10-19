#!/usr/bin/env python3
"""CPU task: reuse validated legacy primitives or compute missing thin-link OPE."""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from ope_components import (ROOT, assert_input, atomic_json, export_product,
                            output_dir, raw_base, require, sha256, validate_product,
                            validate_raw)


def check_campaign(require_build=True):
    campaign = json.loads((ROOT / "campaign.json").read_text())
    require(sha256(ROOT / "requests.json") == campaign["requests_sha256"], "manifest changed")
    for name, expected in campaign["code_sha256"].items():
        require(sha256(name) == expected, f"campaign code changed: {name}")
    if require_build:
        provenance = json.loads((ROOT / "build_provenance.json").read_text())
        require(sha256(provenance["binary_path"]) == provenance["binary_sha256"], "kernel binary changed")
    return campaign


def record_build():
    campaign = check_campaign(require_build=False)
    binary = ROOT / "build/flowed_gluon_ope_components"
    atomic_json(ROOT / "build_provenance.json", {
        "binary_path": str(binary), "binary_sha256": sha256(binary),
        "source_sha256": campaign["code_sha256"],
        "kernel_schema": "gradient_flow_gluon_ope_v2",
        "kernel_role": "unchanged L32x96 compatibility kernel; targets rebuilt from separate T/S slots",
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "flags": "-O3 -march=x86-64 -mtune=generic -ffast-math -fopenmp -std=c++17 -static-libstdc++ -static-libgcc",
    })


def compute_raw(req, root=None):
    assert_input(req)
    base = raw_base(root if root is not None else ROOT / "raw", req)
    if Path(str(base) + ".npy").exists() or Path(str(base) + ".json").exists():
        validate_raw(base, req)
        print(f"RAW_RESUME_SKIP {base}", flush=True)
        return base
    base.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz, nt = req["lattice_xyzt"]
    cmd = [str(ROOT / "build/flowed_gluon_ope_components"), "--input", req["gauge_path"],
           "--conf-id", req["conf_id"], "--output", str(base), "--nx", str(nx),
           "--ny", str(ny), "--nz", str(nz), "--nt", str(nt),
           "--tau", str(req["tau"]), "--epsilon", str(req["epsilon"]),
           "--z-dir", str(req["z_dir"]), "--z-count", str(req["z_count"])]
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    validate_raw(base, req)
    return base


def run_task(index):
    check_campaign()
    requests = json.loads((ROOT / "requests.json").read_text())
    require(0 <= index < len(requests), "task index out of range")
    req = requests[index]
    with (ROOT / "locks" / f"task{index}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_input(req)
        destination = output_dir(req)
        if (destination / "complete.json").exists():
            validate_product(destination, req)
            print(f"TASK_RESUME_SKIP index={index} {destination}", flush=True)
            return
        if req["initial_action"] == "export":
            base = Path(req["legacy_base"])
            validate_raw(base, req)
        else:
            base = compute_raw(req)
        export_product(base, req)
        validate_product(destination, req)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--record-build", action="store_true")
    args = parser.parse_args()
    if args.record_build:
        record_build()
    else:
        require(args.task_index is not None, "--task-index required")
        run_task(args.task_index)


if __name__ == "__main__":
    main()
