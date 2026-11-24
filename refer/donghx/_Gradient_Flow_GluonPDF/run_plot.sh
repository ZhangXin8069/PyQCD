#!/bin/bash
#SBATCH --job-name=qflow_symgap_signal
#SBATCH --partition=cpueicc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-21%6
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_flow_ud_symmetric_freegap_20261115/logs/plot_%A_%a.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_flow_ud_symmetric_freegap_20261115/logs/plot_%A_%a.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_flow_ud_symmetric_freegap_20261115
/public/home/donghx/AnaConda/bin/python code/plot_cpu.py "$SLURM_ARRAY_TASK_ID"
