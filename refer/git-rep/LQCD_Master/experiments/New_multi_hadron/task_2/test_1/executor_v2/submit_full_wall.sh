#!/bin/bash
#SBATCH -J multih_task2_wall
#SBATCH -p kshdexclu16
#SBATCH --output=/public/home/tangmen/work_pyquda/LQCD_Master_runs/multi-hadron/task_2/test_1/executor_v2/log/full_%A_%a.out
#SBATCH --error=/public/home/tangmen/work_pyquda/LQCD_Master_runs/multi-hadron/task_2/test_1/executor_v2/log/full_%A_%a.err
#SBATCH --array=0%1
#SBATCH --nodes=4
#SBATCH -n 4
#SBATCH --gres=dcu:4
#SBATCH --exclude=""
#SBATCH --time=48:00:00
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
mkdir -p /public/home/tangmen/work_pyquda/LQCD_Master_runs/multi-hadron/task_2/test_1/executor_v2
cd /public/home/tangmen/work_pyquda/LQCD_Master_runs/multi-hadron/task_2/test_1/executor_v2

# start_cfg=12300
# cfg_step=50
# run_count=150
# subtask_count=1

# start_cfg=19800
# cfg_step=50
# run_count=500
# subtask_count=2

start_cfg=43900
cfg_step=50
run_count=18
subtask_count=1

task_id=${SLURM_ARRAY_TASK_ID}

for ((i=task_id; i<run_count; i+=subtask_count)); do
	cfg_id=$((start_cfg + i * cfg_step))
	echo "[$(date '+%F %T')] Subtask ${task_id}, RunIndex $((i + 1))/${run_count}, cfg=${cfg_id}"
	mpirun -n 16 python3 main_wall.py ~/.cache "${cfg_id}" || true
done
