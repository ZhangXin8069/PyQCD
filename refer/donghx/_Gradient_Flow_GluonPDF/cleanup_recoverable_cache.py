#!/usr/bin/env python3
"""Remove allowlisted, reproducible VVV and completed flow intermediates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FILL = ROOT / "quark_spin2_fill_flows_20260923"
EXPANSION = ROOT / "quark_spin2_expansion30_20260925"
TSEPS = range(5, 11)
REQUIRED = {"corr_2pt.npy", "corr_3pt_u.npy", "corr_3pt_d.npy"}
MIN_BYTES = 2_300_000_000
VVV_TARGETS = [
    ROOT / "quark_spin2_expansion20_20260922" / "vvv_cache",
    ROOT / "quark_spin2_production_20260922" / "vvv_cache",
]
FLOW_TARGETS = [
    *((FILL, conf, tau, window)
      for conf in (11000, 11100, *range(12300, 13100, 100))
      for tau, window in ((0.30, 3), (0.40, 4))),
    *((EXPANSION, conf, 0.30, 3) for conf in (13100, 13200, 13300)),
]


def tag(tau: float, window: int) -> str:
    return f"thin_tf{tau:.8f}_eps0.01000000_tw{window}_Nv400_dz21".replace(".", "p")


def check_idle() -> None:
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", "donghx"), "-o", "%i|%j|%T"],
        check=True, capture_output=True, text=True,
    )
    own = os.environ.get("SLURM_JOB_ID", "")
    for line in result.stdout.splitlines():
        job, name, state = line.split("|", 2)
        if job != own and state in {"RUNNING", "PENDING", "CONFIGURING"} and any(
            term in name.lower() for term in ("spin2", "qspin2", "multiflow", "tf03", "tf04")
        ):
            raise RuntimeError(f"quark flow job may use these caches: {line}")


def check_target(path: Path, parent: Path) -> None:
    if path.is_symlink() or not path.is_dir() or parent.is_symlink():
        raise RuntimeError(f"target absent, symlinked, or not directory: {path}")
    if path.parent.resolve() != parent.resolve():
        raise RuntimeError(f"target escapes allowlisted parent: {path}")


def validate_flow(work: Path, conf: int, tau: float, window: int) -> Path:
    label = tag(tau, window)
    cache = work / "flow_cache" / str(conf) / label
    check_target(cache, work / "flow_cache" / str(conf))
    output = work / ("results_remaining" if work == FILL else "results_all6") / label / "Nev_200" / str(conf)
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError(f"final output absent: {output}")
    provenance = json.loads((output / "spin2_provenance.json").read_text())
    timing = json.loads((output / "timing_summary.json").read_text())
    if (provenance.get("schema") != "spin2_production_v1"
            or provenance.get("conf") != str(conf)
            or provenance.get("tseps") != list(TSEPS)
            or timing.get("flow_tag") != label
            or timing.get("conf") != str(conf)
            or timing.get("tsep_list") != list(TSEPS)):
        raise RuntimeError(f"final metadata mismatch: {output}")
    receipts = [
        json.loads(path.read_text())
        for path in (work / "status").glob(f"full_chain_{conf}_*.json")
    ]
    if not any(
        record.get("state") == "COMPLETED"
        and abs(float(record.get("flow_time", -1)) - tau) < 1e-10
        and record.get("window") == window
        and Path(record.get("save_root", "")) == output.parents[2]
        for record in receipts
    ):
        raise RuntimeError(f"no matching completed full-chain receipt: {conf} {tau}")
    for tsep in TSEPS:
        path = output / f"corr_projected_tsep{tsep}_dz0-20_Nev200.npz"
        if not path.is_file() or path.stat().st_size < MIN_BYTES:
            raise RuntimeError(f"missing or short final C2/C3: {path}")
        with zipfile.ZipFile(path) as archive:
            if not REQUIRED.issubset(archive.namelist()):
                raise RuntimeError(f"missing C2/C3 array in {path}")
    return cache


def validate_vvv(path: Path) -> Path:
    check_target(path, path.parent)
    expected = (
        {str(c) for c in (11000, 11100, *range(12300, 13100, 100))}
        if path.parent.name == "quark_spin2_expansion20_20260922"
        else {"10000", "10100"}
    )
    actual = {entry.name for entry in path.iterdir()}
    if actual != expected:
        raise RuntimeError(f"unexpected VVV configuration entries: {path}: {actual ^ expected}")
    for conf_dir in path.iterdir():
        check_target(conf_dir, path)
        files = list(conf_dir.iterdir())
        if len(files) != 2 or {f.suffix for f in files} != {".npy", ".json"}:
            raise RuntimeError(f"unexpected VVV cache layout: {conf_dir}")
        if any(f.is_symlink() or not f.is_file() or not f.name.startswith("vvv_Nev200_Nm3_") for f in files):
            raise RuntimeError(f"unexpected VVV file: {conf_dir}")
    return path


def remove_counted(path: Path) -> tuple[int, int, int]:
    files = logical = allocated = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                n, b, a = remove_counted(Path(entry.path))
                files += n
                logical += b
                allocated += a
            else:
                stat = entry.stat(follow_symlinks=False)
                os.unlink(entry.path)
                files += 1
                logical += stat.st_size
                allocated += stat.st_blocks * 512
    os.rmdir(path)
    return files, logical, allocated


def write_receipt(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("vvv", "flow"), required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    targets = []
    skipped = []
    if args.kind == "vvv":
        for candidate in VVV_TARGETS:
            if candidate.exists():
                targets.append(validate_vvv(candidate))
            else:
                skipped.append(str(candidate))
    else:
        for work, conf, tau, window in FLOW_TARGETS:
            candidate = work / "flow_cache" / str(conf) / tag(tau, window)
            if candidate.exists():
                targets.append(validate_flow(work, conf, tau, window))
            else:
                skipped.append(str(candidate))
    print(json.dumps({"kind": args.kind, "apply": args.apply,
                      "targets": list(map(str, targets)), "absent": skipped}, indent=2), flush=True)
    if not args.apply:
        return
    check_idle()
    receipt_name = f"cache_cleanup_{args.kind}_{os.environ.get('SLURM_JOB_ID', 'manual')}.json"
    final_receipt_path = HERE / "status" / receipt_name
    # The VVV pass is what creates the first available quota headroom.
    receipt_path = Path("/tmp") / receipt_name if args.kind == "vvv" else final_receipt_path
    receipt = {"schema": "quark_recoverable_cache_cleanup_v1", "kind": args.kind,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "state": "RUNNING", "job": os.environ.get("SLURM_JOB_ID"),
               "targets": [], "absent": skipped,
               "files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    write_receipt(receipt_path, receipt)
    for path in targets:
        check_idle()
        if args.kind == "vvv":
            validate_vvv(path)
        else:
            parts = path.relative_to(ROOT).parts
            work = ROOT / parts[0]
            conf = int(parts[2])
            label = parts[3]
            match = next((item for item in FLOW_TARGETS if item[0] == work and item[1] == conf
                          and tag(item[2], item[3]) == label), None)
            if match is None:
                raise RuntimeError(f"path left allowlist: {path}")
            validate_flow(*match)
        n, b, a = remove_counted(path)
        receipt["targets"].append({"path": str(path), "files": n,
                                   "logical_bytes": b, "allocated_bytes": a})
        receipt["files"] += n
        receipt["logical_bytes"] += b
        receipt["allocated_bytes"] += a
        write_receipt(receipt_path, receipt)
        print(f"CLEANED {path} files={n} allocated_bytes={a}", flush=True)
    receipt["state"] = "COMPLETED"
    receipt["ended_utc"] = datetime.now(timezone.utc).isoformat()
    write_receipt(receipt_path, receipt)
    if receipt_path != final_receipt_path:
        write_receipt(final_receipt_path, receipt)
    print(f"CLEANUP_COMPLETED allocated_bytes={receipt['allocated_bytes']}", flush=True)


if __name__ == "__main__":
    main()
