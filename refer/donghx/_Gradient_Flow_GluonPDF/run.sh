#!/bin/bash
#SBATCH --partition=i72c512g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=gf_adj_ref7
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent7_ref7_constant_T0p8_1p1_20261120/logs/run_%A_%a.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent7_ref7_constant_T0p8_1p1_20261120/logs/run_%A_%a.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent7_ref7_constant_T0p8_1p1_20261120
if [ "${1:-fit}" = collect ]; then
    /public/home/donghx/AnaConda/bin/python code/report_cpu.py
else
    /public/home/donghx/AnaConda/bin/python code/test_cpu.py
    /public/home/donghx/AnaConda/bin/python code/run_cpu.py "${SLURM_ARRAY_TASK_ID:-0}"
fi
