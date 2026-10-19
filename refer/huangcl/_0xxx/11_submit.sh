#!/bin/bash
#SBATCH --job-name=ratio_gevp
#SBATCH --partition=a192c1t,cpu6248R,cpueicc,i72c512g,math,cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

code="1_ratio_gevp"
run_log="_${code}.log"

source /public/home/huangcl/act_venv.sh

echo "start=$(date '+%Y-%m-%d %H:%M:%S')" > "$run_log"

python -u "${code}.py" >> "$run_log" 2>&1

echo "end=$(date '+%Y-%m-%d %H:%M:%S')" >> "$run_log"