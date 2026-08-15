#!/usr/bin/env bash

set -u -o pipefail

BASE_DIR="/public/home/tangmen/work_pyquda/LQCD_Master_runs/CCtest/loacl_2pt/local_2pt_test2"
START_TASK=1
END_TASK=20

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found. Please load Slurm environment first." >&2
  exit 1
fi

success_count=0
fail_count=0

for i in $(seq "${START_TASK}" "${END_TASK}"); do
  task_dir="${BASE_DIR}/task_${i}"
  submit_script="${task_dir}/submit_test.sh"

  if [[ ! -d "${task_dir}" ]]; then
    echo "[SKIP] task_${i}: directory not found -> ${task_dir}"
    ((fail_count++))
    continue
  fi

  if [[ ! -f "${submit_script}" ]]; then
    echo "[SKIP] task_${i}: submit_test.sh not found -> ${submit_script}"
    ((fail_count++))
    continue
  fi

  if submit_output=$(cd "${task_dir}" && sbatch "submit_test.sh" 2>&1); then
    echo "[OK] task_${i}: ${submit_output}"
    ((success_count++))
  else
    echo "[FAIL] task_${i}: ${submit_output}"
    ((fail_count++))
  fi
done

echo "Done: ${success_count} success, ${fail_count} failed/skipped."
