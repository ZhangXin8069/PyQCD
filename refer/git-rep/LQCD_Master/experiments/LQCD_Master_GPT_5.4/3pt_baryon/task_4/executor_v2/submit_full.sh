#!/bin/bash
#SBATCH -J lambda_lambda_axial_ss_prod
#SBATCH -p kshdexclu16
#SBATCH --output=/public/home/xywangsjtu/work/LQCDMaster/QCD_Master/examples/3pt/result_baryon/20260608_123913/task_4/executor_v2/full_%j.out
#SBATCH --error=/public/home/xywangsjtu/work/LQCDMaster/QCD_Master/examples/3pt/result_baryon/20260608_123913/task_4/executor_v2/full_%j.err
#SBATCH --array=10000-10850:50
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --gres=dcu:4
#SBATCH --exclude="f17r3n00"
#SBATCH --time=12:00:00
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
mkdir -p /public/home/xywangsjtu/work/LQCDMaster/QCD_Master/examples/3pt/result_baryon/20260608_123913/task_4/executor_v2
cd /public/home/xywangsjtu/work/LQCDMaster/QCD_Master/examples/3pt/result_baryon/20260608_123913/task_4/executor_v2

mpirun -n 4 python3 main.py ~/.cache $SLURM_ARRAY_TASK_ID
