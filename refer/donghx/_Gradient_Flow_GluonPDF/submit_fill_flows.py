#!/usr/bin/env python3
"""Submit the five missing flow times for the ten added configurations."""
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFS = ["11000", "11100", "12300", "12400", "12500", "12600", "12700", "12800", "12900", "13000"]
FLOW_SPECS = [
    {"flow_time": 0.05, "epsilon": 0.01, "n_steps": 5, "window": 2},
    {"flow_time": 0.10, "epsilon": 0.01, "n_steps": 10, "window": 2},
    {"flow_time": 0.20, "epsilon": 0.01, "n_steps": 20, "window": 3},
    {"flow_time": 0.80, "epsilon": 0.01, "n_steps": 80, "window": 6},
    {"flow_time": 1.00, "epsilon": 0.01, "n_steps": 100, "window": 7},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', help='record an already submitted array job without submitting again')
    parser.add_argument('--receipt-name', default='submitted_fill_flows.json')
    args = parser.parse_args()
    receipt_path = HERE / "status" / args.receipt_name
    if receipt_path.exists():
        raise FileExistsError(f"submission receipt already exists: {receipt_path}")
    inventory = json.loads((HERE.parent / "quark_spin2_expansion20_20260922" / "status" / "inventory20.json").read_text())
    available = [row["conf"] for row in inventory["new_configurations"] if row["status"] == "PASS"]
    if available != CONFS:
        raise ValueError(f"input inventory changed: {available}")
    script = HERE / "run_fill_flows.slurm"
    subprocess.run(["bash", "-n", str(script)], check=True)
    if args.job_id:
        job = args.job_id
    else:
        done = subprocess.run(
            ["sbatch", "--parsable", "--partition=na800-sug", "--reservation=donghx_223", str(script)],
            check=True, text=True, capture_output=True,
        )
        job = done.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(done.stdout)
    tasks = []
    for flow_index, spec in enumerate(FLOW_SPECS):
        for conf_index, conf in enumerate(CONFS):
            tasks.append({
                "array_index": flow_index * 10 + conf_index,
                "conf": conf,
                **spec,
                "job": f"{job}_{flow_index * 10 + conf_index}",
            })
    receipt = {
        "array_job": job,
        "submitted": datetime.now().isoformat(),
        "partition": "na800-sug",
        "reservation": "donghx_223",
        "array_concurrency": None,
        "gpus_per_task": 4,
        "source_batch": "quark_spin2_expansion20_20260922",
        "purpose": "complete missing flow times for the ten added configurations",
        "tasks": tasks,
        "script": str(script),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
