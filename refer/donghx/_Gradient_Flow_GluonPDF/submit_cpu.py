#!/usr/bin/env python3
"""Review/submit dependent CPU jobs, or record job IDs submitted externally."""
import argparse
import json
import shlex
import subprocess
from datetime import datetime

from ope_components import ROOT, atomic_json
from run_task import check_campaign


def commands(check_id="CHECK_JOB", pilot_id="PILOT_JOB"):
    requests = json.loads((ROOT / "requests.json").read_text())
    pilot_indices = {r["task_index"] for r in requests
                     if r["conf_id"] == "10000" and r["tau"] in (0.05, 0.8)}
    compute_indices = [r["task_index"] for r in requests
                       if r["initial_action"] == "compute" and r["task_index"] not in pilot_indices]
    return [
        ["sbatch", "--parsable", str(ROOT / "build_validate_cpu.slurm")],
        ["sbatch", "--parsable", f"--dependency=afterok:{check_id}", str(ROOT / "export_cpu.slurm")],
        ["sbatch", "--parsable", "--job-name=gf_g64_eps", f"--dependency=afterok:{check_id}",
         "--export=ALL,RUN_PILOT=1", str(ROOT / "run_cpu.slurm")],
        ["sbatch", "--parsable", f"--dependency=afterok:{pilot_id}",
         "--array=" + ",".join(map(str, compute_indices)), str(ROOT / "run_cpu.slurm")],
    ]


def record(jobs):
    receipt = dict(generated=datetime.now().astimezone().isoformat(timespec="seconds"),
                   jobs=dict(zip(("build_oracle_check", "legacy_export", "epsilon_pilot", "missing_ope_array"), jobs)),
                   scope="20 conf x 8 tau; 120 exports, 2 coarse pilot OPE + 38 production OPE",
                   dependencies="export/pilot afterok build; production afterok real-gauge epsilon pilot",
                   commands=commands(jobs[0], jobs[2]))
    atomic_json(ROOT / "submission.json", receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--record", nargs=4, metavar=("CHECK", "EXPORT", "PILOT", "ARRAY"))
    args = parser.parse_args()
    check_campaign(require_build=False)
    if args.record:
        record(args.record)
        return
    if not args.submit:
        for command in commands():
            print(shlex.join(command))
        return
    if (ROOT / "submission.json").exists():
        raise SystemExit("submission receipt exists; use audit_results.py and submit missing indices explicitly")
    jobs = []
    for index in range(4):
        command = commands(jobs[0] if jobs else "CHECK_JOB", jobs[2] if len(jobs) > 2 else "PILOT_JOB")[index]
        result = subprocess.check_output(command, text=True).strip()
        job_id = result.split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch result: {result}")
        jobs.append(job_id)
        atomic_json(ROOT / "submission_partial.json", {"jobs": jobs, "commands": commands(jobs[0], jobs[2] if len(jobs) > 2 else "PILOT_JOB")})
        print(f"SUBMITTED index={index} job={job_id}", flush=True)
    record(jobs)


if __name__ == "__main__":
    main()
