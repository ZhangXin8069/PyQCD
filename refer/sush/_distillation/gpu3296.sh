#!/bin/bash
#SBATCH --job-name=3296
#SBATCH --array=191-192,194-195
#SBATCH --nodes=1

##SBATCH --error=test_pion_3pt_error
##SBATCH --output=test_pion_3pt_output

#SBATCH --partition=dgx1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
##SBATCH --mem-per-gpu=55GB
##SBATCH -x gpu004
##SBATCH --begin=now+4hour
##sbatch --begin=now+2hours

##SBATCH --time="24:00:00"

# confs=(
#     20700 20800 20900 21000 21100 21200 21300
#     21400 21500 21600 21700 21800 21900 22000
#     22100 22200 22300 22400 22500 22600 22700
#     22800 22900 23000 23100 23200 23300
#     23600 23700 23800 23900 24000 24100 24200
#     24300 24400 24500 24600 24700 24800 24900
#     25000 25100 25200 25300 25400 25500 25600
#     25700 25800 25900 26000 26100 26200 26300
#     26400 26500 26600 26700 26800 26900 27000
#     27100 27200 27300 27400 27500 27600 27700
#     27800 27900 28000 28100 28200 28300 28400
#     28500 28600 28700 28800 28900 29000 29100
#     29200 29300 29400 29500 29600 29700 29800
#     29900
# )

# confs=(
#     10000 17100 17300 17500 17700
#     17800 21300 21700 27300 13800 18300 18400
# )

# conf=${confs[$SLURM_ARRAY_TASK_ID]}

conf_start=10000
gap=100

conf=$[${conf_start}+${gap}*${SLURM_ARRAY_TASK_ID}]
source /public/home/sush/envs/C12P12

module purge
module load hpcx-ompi/2.18.1
module load cuda/12.6.1-none-none


exe=/public/home/sush/distillation/guoj/contraction_vvv_vdv_3pt_cupy_combined3296_1.py

mkdir -p ./log3296
log_file=./log3296/${conf}_contraction_cupy_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
: > "$log_file"
export MAX_MOM_BATCH=6
export MAX_2PT_MOM_BATCH=6
mpirun -n 1 --bind-to core --map-by core python -u $exe ${conf} > "$log_file" 2>&1
