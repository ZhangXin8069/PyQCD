#!/bin/bash
set -euo pipefail
root=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_spin2_tf03_tf04_20261004
mkdir -p "${root}/logs" "${root}/status"
job1=$(sbatch --parsable "${root}/run_10000_10900_multiflow.slurm")
job2=$(sbatch --parsable "${root}/run_11000_13000_fullchain.slurm")
job3=$(sbatch --parsable "${root}/run_13100_14000_fullchain.slurm")
printf 'multiflow_10000_10900=%s\nfullchain_11000_13000=%s\nfullchain_13100_14000=%s\n' "${job1}" "${job2}" "${job3}"
printf '%s\n' "${job1}" "${job2}" "${job3}" > "${root}/status/submitted_jobids.txt"
