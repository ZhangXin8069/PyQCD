#!/bin/bash
#SBATCH --job-name=ucx_test_cpu
#SBATCH --partition=cpu6248R
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=ucx_test_cpu_%j.log
#SBATCH --error=ucx_test_cpu_%j.log

set -uo pipefail
ulimit -c 0
source /public/home/huangcl/act_venv3_10.sh
export UCX_HANDLE_ERRORS=none
export OMP_NUM_THREADS=1

set +e
mpirun -n 1 python -c 'from mpi4py import MPI; import ctypes; print("MPI initialized; triggering intentional SIGSEGV", flush=True); ctypes.string_at(0)'
status=$?
set -e

printf 'mpirun exit status: %s\n' "$status"
printf 'UCX_HANDLE_ERRORS=%s\n' "$UCX_HANDLE_ERRORS"
