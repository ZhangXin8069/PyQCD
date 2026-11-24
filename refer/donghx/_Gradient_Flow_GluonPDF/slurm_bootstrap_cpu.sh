#!/bin/bash
#SBATCH --job-name=gf_adj75_boot500
#SBATCH --partition=a192c1t,cpu6248R,cpueicc,i72c512g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-21%6
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122/logs/bootstrap_%A_%a.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122/logs/bootstrap_%A_%a.err

set -euo pipefail
ulimit -s unlimited
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ROOT=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122
mkdir -p "$ROOT/logs"
cd "$ROOT"
PYTHON=/public/home/donghx/AnaConda/bin/python
export PYTHONPATH="$ROOT/code:$ROOT:${PYTHONPATH:-}"
"$PYTHON" -u code/bootstrap_cpu.py "$SLURM_ARRAY_TASK_ID"
