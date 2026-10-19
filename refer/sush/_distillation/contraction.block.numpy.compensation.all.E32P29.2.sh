#!/bin/bash

#SBATCH --job-name=Pi_ENV
#SBATCH --array=6-28,30-36,40-72,76-99
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --partition=na800-pcie
##SBATCH --partition=a192c1t
#SBATCH --partition=cpu6248R,i72c512g,cpueicc
##SBATCH --partition=cpu-short
##SBATCH --partition=gpu-debug

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

##SBATCH --partition=gpu-debug
##SBATCH --nodes=1
##SBATCH --gpus=1
##SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1

##SBATCH --begin=now+7hour
##SBATCH --time="2-00:00:00"
##SBATCH --time="4:00:00"
##SBATCH --time="30:00"

##SBATCH -w cpu072

##SBATCH -x cpu054,cpu057,cpu060
##SBATCH --exclusive

# 禁用不需要的传输方式
export UCX_TLS=rc,shm  # 仅使用InfiniBand和共享内存

source /public/home/sush/envs/C12P12

# module purge
# module load hpcx-ompi/2.18.1

conf_start=1000
gap=80
gap_2=40

for i in {0..1..1}
do
    conf=$((conf_start + gap * SLURM_ARRAY_TASK_ID + i * gap_2))

    # conf=$[${conf_stare}+${gap}*${SLURM_ARRAY_TASK_ID}]

    run_dir=/public/home/sush/distillation/pion_form_factor

    exe=${run_dir}/contraction.block.numpy.compensation.all.E32P29.2.py
    dir_file=${run_dir}/file/contraction.block.numpy.compensation.all.E32P29.2.${conf}

    output_file=${run_dir}/output/contraction.block.numpy.compensation.all.E32P29.2.${conf}.log

    if [ ! -e "${run_dir}/output/" ];then
        mkdir ${run_dir}/output/
    fi

    if [ ! -e "${run_dir}/file" ];then
        mkdir ${run_dir}/file
    fi

    echo "${conf} job starts at" `date` > ${output_file}
    # mkdir ${dir_file}

    if [ -d ${dir_file} ]; then
        mpirun -n 16 --bind-to none python -u $exe ${conf} >> ${output_file} 2>&1 
        # mpirun -n 16 \
        # --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll \
        # --bind-to none python -u $exe ${conf} >> ${output_file} 2>&1 

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

done