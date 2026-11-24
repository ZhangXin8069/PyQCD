#!/bin/bash
#SBATCH --partition=cpueicc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-7%4
#SBATCH --time=02:00:00
#SBATCH --job-name=q045_lamet
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_P045_lambda_Fourier_LaMET_LHAPDF_20261116/logs/chain_%A_%a.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_P045_lambda_Fourier_LaMET_LHAPDF_20261116/logs/chain_%A_%a.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_P045_lambda_Fourier_LaMET_LHAPDF_20261116
/public/home/donghx/AnaConda/bin/python -B code/run_cpu.py "$SLURM_ARRAY_TASK_ID"
