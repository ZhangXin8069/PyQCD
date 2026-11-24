#!/bin/bash
#SBATCH --partition=cpueicc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=gf_bare_fit
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_selfratio_T0p8_1p1_bare_20261116/logs/finish_v2_%A_%a.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_selfratio_T0p8_1p1_bare_20261116/logs/finish_v2_%A_%a.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_selfratio_T0p8_1p1_bare_20261116
/public/home/donghx/AnaConda/bin/python code/finish_v2_cpu.py
