#!/bin/bash
#SBATCH -J test_LbLc_axial
#SBATCH -p kshdexclu16
#SBATCH --output=/public/home/tangmen/work_pyquda/LQCD_Master/examples/3pt/result_ds_baryon/20260701_040733/task_14/executor_v1/test_%j.out
#SBATCH --error=/public/home/tangmen/work_pyquda/LQCD_Master/examples/3pt/result_ds_baryon/20260701_040733/task_14/executor_v1/test_%j.err
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --gres=dcu:4
#SBATCH --exclude="f17r3n00"
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=1
#SBATCH --exclusive

set -euo pipefail

module purge
module load apps/git/2.30.2
module load compiler/cmake/3.23.3
module load compiler/gnu/9.3.0
module load compiler/intel/2017.5.239
module load mpi/hpcx/2.11.0/intel-2017.5.239
module load compiler/dtk/25.04.4
export CUPY_INSTALL_USE_HIP=1
export HCC_AMDGPU_TARGET=gfx906
export CC=clang
export CXX=clang++
export HIPCXX=dcc
export QUDA_PATH=/public/home/xywangsjtu/Config_PyQUDA/Build-USQCD-SciDAC/scidac/install/quda-mpi-gfx906
source /public/home/xywangsjtu/Config_PyQUDA/zmh-2604/bin/activate
mkdir -p /public/home/tangmen/work_pyquda/LQCD_Master/examples/3pt/result_ds_baryon/20260701_040733/task_14/executor_v1
cd /public/home/tangmen/work_pyquda/LQCD_Master/examples/3pt/result_ds_baryon/20260701_040733/task_14/executor_v1

mpirun -n 4 python3 main.py ~/.cache 10000
