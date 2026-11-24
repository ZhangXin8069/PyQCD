#!/bin/bash
#SBATCH --job-name=gf_adj75_preflight
#SBATCH --partition=a192c1t,cpu6248R,cpueicc,i72c512g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122/logs/preflight_%j.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122/logs/preflight_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent_ref75_boot500_T0p8_1p1_20261122
cd "$ROOT"
export PYTHONPATH="$ROOT/code:$ROOT:${PYTHONPATH:-}"
PYTHON=/public/home/donghx/AnaConda/bin/python
"$PYTHON" -u code/prepare_indices.py
"$PYTHON" -u code/test_cpu.py
"$PYTHON" -u code/preflight_cpu.py
