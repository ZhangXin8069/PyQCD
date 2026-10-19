#!/usr/bin/env python3
"""Validate one completed retry and remove only its reconstructible flow cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from cleanup_recoverable_cache import EXPANSION, remove_counted, tag, validate_flow, write_receipt


HERE = Path(__file__).resolve().parent


def allowed() -> set[tuple[int, float, int]]:
    rows = set()
    for line in (HERE / "missing_retry_20261006.tsv").read_text().splitlines():
        _, conf, tau, window = line.split()
        rows.add((int(conf), float(tau), int(window)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", type=int)
    parser.add_argument("tau", type=float)
    parser.add_argument("window", type=int)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    key = (args.conf, args.tau, args.window)
    if key not in allowed():
        parser.error(f"task is not in the missing-only manifest: {key}")

    cache = validate_flow(EXPANSION, *key)
    print(f"VALIDATED conf={args.conf} tau={args.tau:.2f} cache={cache}", flush=True)
    if not args.cleanup:
        return 0

    receipt = {
        "schema": "quark_flow_retry_cache_cleanup_v1",
        "conf": args.conf,
        "flow_time": args.tau,
        "window": args.window,
        "job": __import__("os").environ.get("SLURM_JOB_ID"),
        "target": str(cache),
        "state": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = HERE / "status" / (
        f"retry_cleanup_{args.conf}_tf{args.tau:.2f}_"
        f"{receipt['job'] or 'manual'}.json"
    )
    write_receipt(receipt_path, receipt)
    files, logical, allocated = remove_counted(cache)
    receipt.update(
        state="COMPLETED",
        ended_utc=datetime.now(timezone.utc).isoformat(),
        files=files,
        logical_bytes=logical,
        allocated_bytes=allocated,
    )
    write_receipt(receipt_path, receipt)
    print(f"CLEANED conf={args.conf} tau={args.tau:.2f} files={files} allocated_bytes={allocated}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
