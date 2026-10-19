#!/bin/bash

#SBATCH --job-name=test_f
#SBATCH --array=0#-20
##SBATCH --partition=dgx1,dgx2,nv100-ins,nv100-sug#,na100-40g,na800-pcie
##SBATCH --partition=na800-pcie

#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

##SBATCH --partition=cpueicc,cpu6248R,i72c512g
##SBATCH --ntasks-per-node=16
##SBATCH --cpus-per-task=1

##SBATCH --begin=now+7hour
##SBATCH --time="2-00:00:00"
#SBATCH --time="30:00"

#SBATCH --error=test_error
#SBATCH --output=test_output

#SBATCH -x cpu022

export PATH=/public/home/sush/distillation:$PATH

source /public/home/sush/envs/sm80

conf_stare=10000
gap=100
conf=$[${conf_stare}+${gap}*${SLURM_ARRAY_TASK_ID}]

run_dir=/public/home/sush/distillation/pion_form_factor

exe=${run_dir}/contraction.block.cupy.12x32.py
dir_file=${run_dir}/file/contraction_block_cupy_12x32

output_file=${run_dir}/output/contraction.block.cupy.12x32.${conf}.log

if [ ! -e "${run_dir}/output/" ];then
    mkdir ${run_dir}/output/
fi

if [ ! -e "${run_dir}/file" ];then
    mkdir ${run_dir}/file
fi

echo "${conf} job starts at" `date` > ${output_file}
mkdir ${dir_file}

if [ -d ${dir_file} ]; then
    mpirun -n 1 python -u $exe ${conf} >> ${output_file} 2>&1 

    if [ $? -eq 0 ]; then
        echo "Python 程序执行成功" >> ${output_file} 2>&1
        rm -r ${dir_file}
    else
        echo "Python 程序执行失败" >> ${output_file} 2>&1
    fi

else
    echo "calculate complete" >> ${output_file} 2>&1
fi

echo "$conf job ends at" `date` >> ${output_file}
