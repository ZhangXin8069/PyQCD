#!/bin/bash

# gpu partition: nv100-ins,nv100-sug,dgx2,na100-ins,na100-sug,gpu-debug,na800-sug,na800-pcie,h20-nettr

#SBATCH --job-name=bench_pyqcd_gpu
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# pyqcd 只装在 3.10 环境里, 必须整份加载 module 和 venv.
# 同一份环境里已包含 cuda/11.4.4 和 cupy-cuda11x.
. /public/home/huangcl/act_venv3_10.sh

# ===== 用户配置：修改此处即可 =====
code="benchmark"
# ==================================

log="_11_gpu.log"

echo "job starts at $(date)" >"${log}"
echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-NA} job=${SLURM_JOB_ID:-NA}" >>"${log}"

echo "=== running ${code}.py gpu ===" >>"${log}"
python -u "${code}.py" gpu >>"${log}" 2>&1
echo "=== ${code} done ===" >>"${log}"

echo "job ends at $(date)" >>"${log}"
