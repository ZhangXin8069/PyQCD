#!/bin/bash
set -euo pipefail

root=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_spin2_tf03_tf04_20261004
script=${root}/retry_missing_20261006.slurm

# full_chain.py keys per-conf logs and status by the array's main Slurm job ID.
# Finish the 0.30 array before launching 0.40 for the same configurations.
job03=$(sbatch --parsable --array=0-6 "${script}")
job04=$(sbatch --parsable --array=7-16 --dependency=afterany:${job03} "${script}")
printf 'tau0.30=%s\ntau0.40=%s\n' "${job03}" "${job04}"
