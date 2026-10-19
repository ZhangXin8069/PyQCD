#!/bin/bash

#SBATCH --job-name=Pi_ENV
#SBATCH --array=0
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --partition=i72c512g,cpueicc,cpu6248R
##SBATCH --nodes=1
##SBATCH --ntasks=32
##SBATCH --ntasks-per-node=32
##SBATCH --cpus-per-task=1

#SBATCH --partition=na800-pcie
##SBATCH --partition=gpu-debug
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH -n 32

##SBATCH --begin=now+7hour
#SBATCH --time="1-00:00:00"
##SBATCH --time="30:00"

##SBATCH -x gpu048

##SBATCH -x cpu052
##SBATCH --exclusive

export PATH=/public/home/sush/distillation:$PATH

source /public/home/sush/envs/sm80

conf_stare=4000
gap=100
conf=$[${conf_stare}+${gap}*${SLURM_ARRAY_TASK_ID}]

run_dir=/public/home/sush/distillation/pion_form_factor/

exe=${run_dir}/contraction.block.numpy.32x96.compensation.py
dir_file=${run_dir}/file/contraction_block_numpy_32x96_compensation_${conf}

output_file=${run_dir}/output/contraction.block.numpy.32x96.compensation.${conf}.log

if [ ! -e "${run_dir}/output/" ];then
    mkdir ${run_dir}/output/
fi

if [ ! -e "${run_dir}/file" ];then
    mkdir ${run_dir}/file
fi

echo "${conf} job starts at" `date` > ${output_file}
mkdir ${dir_file}

if [ -d ${dir_file} ]; then
    mpirun -n 32 python -u $exe ${conf} >> ${output_file} 2>&1 

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
