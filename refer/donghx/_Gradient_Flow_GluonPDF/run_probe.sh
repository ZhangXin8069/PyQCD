#!/bin/bash
#SBATCH --partition=cpueicc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:10:00
#SBATCH --output=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/logs/probe_%j.out
#SBATCH --error=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/logs/probe_%j.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export LC_ALL=C LANG=C PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/home/donghx/AnaConda/bin/python /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/code/probe_cpu.py
