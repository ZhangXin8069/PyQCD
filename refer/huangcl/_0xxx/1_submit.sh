#!/bin/bash
#SBATCH --job-name=2pt_NP_11000

##SBATCH --partition=gpu-debug
#SBATCH --partition=nv100-sug,nv100-ins

#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

py_name="1_NP_corr"

input="input.txt"
run_log="11000.log"


source /public/home/huangcl/act_venv.sh

echo "date=$(date)" >"${run_log}"
python -u "${py_name}.py" "$input" >>"${run_log}" 2>&1
echo "date=$(date)" >>"${run_log}"
