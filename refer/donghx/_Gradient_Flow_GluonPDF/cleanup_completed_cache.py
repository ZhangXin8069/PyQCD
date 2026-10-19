#!/usr/bin/env python3
"""Remove only reproducible flow caches with validated final C2/C3 products."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import zipfile


WORK = Path(__file__).resolve().parents[1] / "quark_spin2_expansion30_20260925"
CACHE = WORK / "flow_cache"
RESULTS = WORK / "results_all6"
STATUS = WORK / "status"
RECEIPT = Path(__file__).resolve().parent / "cleanup_receipt.json"
TSEPS = (5, 6, 7, 8, 9, 10)
MIN_NPZ_BYTES = 2_300_000_000
REQUIRED_MEMBERS = {"corr_2pt.npy", "corr_3pt_u.npy", "corr_3pt_d.npy"}
SPECS = ((0.05, 2, range(13100, 14100, 100)),
         (0.10, 2, range(13100, 14100, 100)),
         (0.20, 3, range(13100, 14100, 100)),
         (0.50, 4, range(13100, 13500, 100)))


def tag(flow_time, window):
    return (f"thin_tf{flow_time:.8f}_eps0p01000000_"
            f"tw{window}_Nv400_dz21").replace(".", "p")


def validate(conf, flow_time, window):
    flow_tag = tag(flow_time, window)
    cache = CACHE / str(conf) / flow_tag
    output = RESULTS / flow_tag / "Nev_200" / str(conf)
    if cache.is_symlink() or not cache.is_dir():
        return None, f"cache absent or symlink: {cache}"
    if cache.resolve().parent != (CACHE / str(conf)).resolve():
        raise RuntimeError(f"cache escapes expected parent: {cache}")
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError(f"final output absent or symlink: {output}")

    provenance = json.loads((output / "spin2_provenance.json").read_text())
    timing = json.loads((output / "timing_summary.json").read_text())
    if (provenance.get("schema") != "spin2_production_v1"
            or provenance.get("conf") != str(conf)
            or provenance.get("tseps") != list(TSEPS)
            or timing.get("flow_tag") != flow_tag
            or timing.get("conf") != str(conf)
            or timing.get("tsep_list") != list(TSEPS)):
        raise RuntimeError(f"final metadata mismatch: {output}")

    completed = []
    for path in STATUS.glob(f"full_chain_{conf}_*.json"):
        record = json.loads(path.read_text())
        if (record.get("state") == "COMPLETED"
                and abs(float(record.get("flow_time", -1)) - flow_time) < 1e-10
                and Path(record.get("save_root", "")) == RESULTS
                and record.get("tseps") == list(TSEPS)):
            completed.append(record.get("job"))
    if not completed:
        raise RuntimeError(f"no matching COMPLETED full-chain receipt: {conf} {flow_time}")

    for tsep in TSEPS:
        path = output / f"corr_projected_tsep{tsep}_dz0-20_Nev200.npz"
        if not path.is_file() or path.stat().st_size < MIN_NPZ_BYTES:
            raise RuntimeError(f"final C2/C3 file absent or too small: {path}")
        with zipfile.ZipFile(path) as archive:
            if not REQUIRED_MEMBERS.issubset(archive.namelist()):
                raise RuntimeError(f"final C2/C3 arrays missing: {path}")
    return {"conf": conf, "flow_time": flow_time, "cache": str(cache),
            "output": str(output), "completed_jobs": completed}, None


def remove_counted(path):
    files = 0
    logical_bytes = 0
    allocated_bytes = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                nfiles, nlogical, nallocated = remove_counted(entry.path)
                files += nfiles
                logical_bytes += nlogical
                allocated_bytes += nallocated
            else:
                stat = entry.stat(follow_symlinks=False)
                os.unlink(entry.path)
                files += 1
                logical_bytes += stat.st_size
                allocated_bytes += stat.st_blocks * 512
    os.rmdir(path)
    return files, logical_bytes, allocated_bytes


def save_receipt(receipt, destination):
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(temporary, destination)


def check_other_jobs():
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", "donghx"), "-o", "%i|%j|%T"],
        check=True, capture_output=True, text=True)
    own = os.environ.get("SLURM_JOB_ID")
    for line in result.stdout.splitlines():
        job, name, state = line.split("|", 2)
        if job != own and state == "RUNNING" and ("spin2" in name or "qspin2" in name):
            raise RuntimeError(f"active quark production job may use cache: {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--conf", type=int)
    parser.add_argument("--flow-time", type=float)
    parser.add_argument("--window", type=int)
    args = parser.parse_args()
    single = any(value is not None for value in
                 (args.conf, args.flow_time, args.window))
    if single and not all(value is not None for value in
                          (args.conf, args.flow_time, args.window)):
        parser.error("--conf, --flow-time and --window must be given together")
    if single:
        allowed = {(conf, flow, window)
                   for flow, window, confs in SPECS
                   for conf in confs}
        allowed.update((conf, flow, window)
                       for flow, window, confs in
                       ((0.50, 4, range(13500, 14100, 100)),
                        (0.80, 6, range(13100, 14100, 100)))
                       for conf in confs)
        if (args.conf, args.flow_time, args.window) not in allowed:
            parser.error("conf/flow/window is outside the audited 30-conf production")
        specs = ((args.flow_time, args.window, (args.conf,)),)
        flow_label = f"{args.flow_time:.2f}".replace(".", "p")
        receipt_path = RECEIPT.parent / (
            f"cleanup_{os.environ.get('SLURM_JOB_ID', 'audit')}_"
            f"{args.conf}_tf{flow_label}.json")
    else:
        specs = SPECS
        receipt_path = RECEIPT
    candidates = []
    skipped = []
    for flow_time, window, confs in specs:
        for conf in confs:
            item, reason = validate(conf, flow_time, window)
            if item is None:
                skipped.append(reason)
            else:
                candidates.append(item)
    print(json.dumps({"mode": "apply" if args.apply else "audit",
                      "candidates": len(candidates), "skipped": skipped,
                      "targets": candidates}, indent=2), flush=True)
    if not args.apply:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cleanup must run as a Slurm CPU job")
    if not single:
        check_other_jobs()
    receipt = {"schema": "completed_quark_flow_cache_cleanup_v1",
               "job": os.environ["SLURM_JOB_ID"], "state": "RUNNING",
               "targets": [], "files": 0, "logical_bytes": 0,
               "allocated_bytes": 0, "skipped": skipped}
    for item in candidates:
        # Repeat the output guard just before this specific cache is removed.
        validate(item["conf"], item["flow_time"],
                 4 if item["flow_time"] == 0.5 else
                 3 if item["flow_time"] == 0.2 else 2)
        files, logical, allocated = remove_counted(Path(item["cache"]))
        receipt["targets"].append({**item, "files": files,
                                   "logical_bytes": logical,
                                   "allocated_bytes": allocated})
        receipt["files"] += files
        receipt["logical_bytes"] += logical
        receipt["allocated_bytes"] += allocated
        save_receipt(receipt, receipt_path)
        print(f"CLEANED conf={item['conf']} tf={item['flow_time']:.2f} "
              f"files={files} allocated_bytes={allocated}", flush=True)
    receipt["state"] = "COMPLETED"
    save_receipt(receipt, receipt_path)
    print(f"CLEANUP_COMPLETED files={receipt['files']} "
          f"allocated_bytes={receipt['allocated_bytes']}", flush=True)


if __name__ == "__main__":
    main()
