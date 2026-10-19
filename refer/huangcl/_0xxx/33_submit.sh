#!/bin/bash

# gpu partition: nv100-ins,nv100-sug,dgx2,na100-ins,na100-sug,gpu-debug,na100-40g,na800-sug,na800-pcie,h20-nettr
# cpu partition: cpu6248R,cpueicc,i72c512g,a192c1t,cpu-short
# if using gpu partition, you need to set how many gpu to be use in a task. Otherwise, commenting it.

#SBATCH --job-name=fit_fh
#SBATCH --partition=cpueicc,i72c512g,a192c1t,cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# 限制单线程，匹配 --cpus-per-task=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ===== 用户配置：修改此处即可 =====
code="3_fit_all"
# ==================================

log="_${code}.log"

. /public/home/huangcl/act_venv.sh

echo "job starts at $(date)" >"${log}"

echo "=== running ${code}.py ===" >>"${log}"
python -u "${code}.py" >>"${log}" 2>&1
echo "=== ${code} done ===" >>"${log}"

echo "job ends at $(date)" >>"${log}"
