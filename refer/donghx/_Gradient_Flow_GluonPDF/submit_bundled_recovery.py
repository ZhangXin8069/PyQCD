#!/usr/bin/env python3
"""Recover account-limit rejection without changing sealed physics code."""
import json
import math
import subprocess
from allconf import ROOT, atomic_json, check_campaign, check_inventory, now, require, sha256

check_campaign()
check_inventory()
groups = json.loads((ROOT / "missing_groups.json").read_text())
frozen = json.loads((ROOT / "validation/missing_manifest_identity.json").read_text())
require(sha256(ROOT / "missing_groups.json") == frozen["sha256"], "missing manifest changed")
receipt_path = ROOT / "submission.json"
receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
    "created": now(), "recovery_of_dispatch_job": "3825825",
    "rejection_reason": "AssocMaxSubmitJobLimit: MaxSubmitJobs=200",
    "missing_group_count": len(groups), "missing_request_count": sum(len(g["request_indices"]) for g in groups),
    "groups_per_array_task": 10, "array_concurrency_cap": None, "batches": [],
    "wrapper_sha256": sha256(ROOT / "compute_bundle_cpu.slurm"),
    "submitter_sha256": sha256(__file__),
}
require(receipt.get("groups_per_array_task") == 10, "incompatible existing receipt")
if not receipt["batches"]:
    count = math.ceil(len(groups) / 10)
    require(0 < count <= 1000, "unexpected bundle count")
    command = ["sbatch", "--parsable", f"--array=0-{count-1}",
               f"--export=ALL,GROUPS_PER_TASK=10,GROUP_COUNT={len(groups)}", str(ROOT / "compute_bundle_cpu.slurm")]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    receipt["batches"].append({"job_id": job, "size": count, "offset": 0,
                               "group_count": len(groups), "command": command})
    atomic_json(receipt_path, receipt)
if not receipt.get("final_audit_job"):
    dependency = "afterok:" + ":".join(b["job_id"] for b in receipt["batches"])
    command = ["sbatch", "--parsable", f"--dependency={dependency}", "--job-name=gf_g64_all_final",
               "--export=ALL,FINAL_AUDIT=1", str(ROOT / "audit_cpu.slurm")]
    receipt["final_audit_job"] = subprocess.check_output(command, text=True).strip().split(";")[0]
    receipt["final_audit_command"] = command
    atomic_json(receipt_path, receipt)
print(json.dumps(receipt, indent=2))
