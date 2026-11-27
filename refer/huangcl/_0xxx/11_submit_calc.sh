#!/bin/bash
#SBATCH --job-name=c32p29_pion3pt
#SBATCH --partition=a192c1t
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=calc_pion_3pt_%j.log
#SBATCH --error=calc_pion_3pt_%j.err

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /public/home/huangcl/.venv3_10/bin/activate
python -u /public/home/huangcl/07_pion_gluon_pdf/toshenxm1/calc_pion_3pt.py
