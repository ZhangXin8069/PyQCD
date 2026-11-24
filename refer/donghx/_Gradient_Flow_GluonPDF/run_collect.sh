#!/bin/bash
#SBATCH --partition=a192c1t
#SBATCH --nodelist=cpu074
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=gf_adj7_collect
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent7_anchor8_constant_T0p8_1p1_20261120/logs/collect_%j.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent7_anchor8_constant_T0p8_1p1_20261120/logs/collect_%j.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_adjacent7_anchor8_constant_T0p8_1p1_20261120
/public/home/donghx/AnaConda/bin/python code/report_cpu.py collect
/public/home/donghx/AnaConda/bin/python code/diagnose_cpu.py
