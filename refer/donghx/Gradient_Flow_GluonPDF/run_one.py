#!/usr/bin/env python3
"""Run or resume one manifest entry without ever accepting a partial artifact."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


def tau_tag(value: float) -> str:
    return f"tau{value:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tau", default=0.5, type=float)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--z-count", default=25, type=int)
    parser.add_argument("--z-dir", default=2, type=int)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not 0 <= args.task_index < len(rows):
        raise SystemExit(f"task index {args.task_index} outside [0,{len(rows)})")
    row = rows[args.task_index]
    conf_id = row["conf_id"]
    out_dir = args.output_root / f"conf{conf_id}" / tau_tag(args.tau)
    base = out_dir / f"conf{conf_id}_{tau_tag(args.tau)}_eps{args.epsilon:.3f}_zdir{args.z_dir}"
    validator = Path(__file__).with_name("validate_output.py")
    validate = [
        os.environ.get("PYTHON", "python"), str(validator), str(base),
        "--conf-id", conf_id, "--z-count", str(args.z_count), "--tau", str(args.tau),
    ]
    if Path(str(base) + ".npy").is_file() or Path(str(base) + ".json").is_file():
        completed = subprocess.run(validate, check=False).returncode == 0
        if completed:
            print(f"RESUME_SKIP validated artifact {base}")
            return
        raise SystemExit(f"partial or invalid artifact exists; preserving for audit: {base}")

    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.executable), "--input", row["gauge_path"], "--conf-id", conf_id,
        "--output", str(base), "--nx", "32", "--ny", "32", "--nz", "32", "--nt", "96",
        "--tau", str(args.tau), "--epsilon", str(args.epsilon),
        "--z-count", str(args.z_count), "--z-dir", str(args.z_dir),
    ]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    subprocess.run(validate, check=True)


if __name__ == "__main__":
    main()
