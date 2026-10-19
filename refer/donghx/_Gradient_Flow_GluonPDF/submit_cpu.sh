#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
gauge_root=${GAUGE_ROOT:-/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96}
manifest=${MANIFEST:-$repo_dir/manifest.tsv}
output_root=${OUTPUT_ROOT:-$repo_dir/output}
chain_width=${CHAIN_WIDTH:-8}
flow_tau=${FLOW_TAU:-0.5}
flow_epsilon=${FLOW_EPSILON:-0.01}
fine_epsilon=${FINE_EPSILON:-0.005}
z_count=${Z_COUNT:-25}
z_dir=${Z_DIR:-2}

mkdir -p "$repo_dir/logs" "$repo_dir/runs" "$output_root"
"$repo_dir/scripts/build.sh"
python "$repo_dir/scripts/make_manifest.py" --gauge-root "$gauge_root" --output "$manifest"
task_count=$(($(wc -l < "$manifest") - 1))
if (( task_count < 1 )); then
  echo "empty manifest" >&2
  exit 2
fi

export REPO_DIR="$repo_dir" MANIFEST="$manifest" OUTPUT_ROOT="$output_root"
export FLOW_TAU="$flow_tau" FLOW_EPSILON="$flow_epsilon" Z_COUNT="$z_count" Z_DIR="$z_dir"
export TASK_COUNT="$task_count" CHAIN_STRIDE="$chain_width"

# The full ensemble is held behind a real-data pilot. A failed pilot prevents release.
# A small number of self-chaining heads respects the account MaxSubmitJobs limit.
if [[ -n "${PILOT_JOB_ID:-}" ]]; then
  pilot_job=$PILOT_JOB_ID
else
  pilot_job=$(sbatch --parsable --array=0-0 "$repo_dir/slurm/flowed_ope_cpu.slurm")
fi
fine_job=$(sbatch --parsable --array=0-0 --export="ALL,FLOW_EPSILON=${fine_epsilon}" \
  "$repo_dir/slurm/flowed_ope_cpu.slurm")
tau_tag=$(printf 'tau%.3f' "$flow_tau" | tr '.' 'p')
coarse_base="$output_root/conf1000/$tau_tag/conf1000_${tau_tag}_eps$(printf '%.3f' "$flow_epsilon")_zdir${z_dir}"
fine_base="$output_root/conf1000/$tau_tag/conf1000_${tau_tag}_eps$(printf '%.3f' "$fine_epsilon")_zdir${z_dir}"
gate_job=$(sbatch --parsable --dependency="afterok:${pilot_job}:${fine_job}" \
  --export="ALL,COARSE_BASE=${coarse_base},FINE_BASE=${fine_base}" \
  "$repo_dir/slurm/compare_epsilon_cpu.slurm")
head_jobs=()
for ((index=0; index<chain_width && index<task_count; ++index)); do
  head_jobs+=("$(sbatch --parsable --dependency="afterok:${gate_job}" \
    --export="ALL,TASK_INDEX=${index}" "$repo_dir/slurm/flowed_ope_chain_cpu.slurm")")
done

receipt="$repo_dir/runs/submission_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "pilot_job=${pilot_job}"
  echo "fine_epsilon_job=${fine_job}"
  echo "epsilon_gate_job=${gate_job}"
  echo "chain_head_jobs=${head_jobs[*]}"
  echo "dependency=afterok:${gate_job}"
  echo "task_count=${task_count}"
  echo "chain_width=${chain_width}"
  echo "manifest=${manifest}"
  echo "output_root=${output_root}"
  echo "flow_tau=${flow_tau}"
  echo "flow_epsilon=${flow_epsilon}"
  echo "fine_epsilon=${fine_epsilon}"
  echo "z_count=${z_count}"
  echo "z_dir=${z_dir}"
} > "$receipt"
echo "SUBMITTED pilot=${pilot_job} fine=${fine_job} gate=${gate_job} chain_heads=${head_jobs[*]} receipt=${receipt}"
