#!/bin/bash
#SBATCH --partition=cpueicc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --job-name=sr_g24ratio
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/logs/g24ratio_%A_%a.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/logs/g24ratio_%A_%a.err
#SBATCH --array=0-6%4
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116
/public/home/donghx/AnaConda/bin/python code/gluon24_ratio_cpu.py "$SLURM_ARRAY_TASK_ID"
