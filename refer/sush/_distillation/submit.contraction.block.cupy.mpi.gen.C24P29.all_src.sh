#!/bin/bash
# Run with bash on the login node.
# Submits the first half (src=0..35) only. The first-half job submits the second
# half (src=36..71) itself once it has finished successfully, so the two halves
# are never in the queue at the same time and nothing has to wait on the login
# node. See the end of contraction.block.cupy.mpi.gen.C24P29.sh.
set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: bash $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
job_script=${script_dir}/contraction.block.cupy.mpi.gen.C24P29.sh
mkdir -p "${script_dir}/output"

first_job=$(sbatch --parsable --partition=gpu-debug --time=00:30:00 \
    --job-name=C24P29-first "${job_script}" first)
first_job=${first_job%%;*}
if [[ ! ${first_job} =~ ^[0-9]+$ ]]; then
    echo "Unexpected first-half job ID: ${first_job}" >&2
    exit 1
fi
echo "First-half job: ${first_job} (gpu-debug, src=0..35)"
echo "The first-half job submits the second half (src=36..71) itself after it succeeds."
