#!/bin/bash

#SBATCH --job-name=Pi_ENV
#SBATCH --array=2,4,10
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --partition=na800-pcie
##SBATCH --partition=h20-nettr
##SBATCH --partition=dgx1
#SBATCH --partition=i72c512g,cpu6248R
##SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

##SBATCH --partition=gpu-debug
##SBATCH --nodes=1
##SBATCH --gpus=1
##SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1

##SBATCH --begin=now+7hour
#SBATCH --time="2-00:00:00"
##SBATCH --time="30:00"

##SBATCH -w cpu016
#SBATCH -w cpu058

#SBATCH -x cpu022,cpu051,cpu054,cpu059,cpu055
##SBATCH --exclusive

# 禁用不需要的传输方式
export UCX_TLS=rc,shm  # 仅使用InfiniBand和共享内存

source /public/home/sush/envs/sm80

conf_stare=10000
gap=200

conf=$[${conf_stare}+${gap}*${SLURM_ARRAY_TASK_ID}]

run_dir=/public/home/sush/distillation/pion_form_factor

exe=${run_dir}/contraction.block.numpy.mpi.gen.py
dir_file=${run_dir}/file/contraction.block.numpy.gen.${conf}

output_file=${run_dir}/output/contraction.block.numpy.gen.${conf}.log

if [ ! -e "${run_dir}/output/" ];then
    mkdir ${run_dir}/output/
fi

if [ ! -e "${run_dir}/file" ];then
    mkdir ${run_dir}/file
fi

echo "${conf} job starts at" `date` > ${output_file} 2>&1 
mkdir ${dir_file}

if [ -d ${dir_file} ]; then
    mpirun -n 24 --bind-to core --map-by core python -u $exe ${conf} >> ${output_file} 2>&1 

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
