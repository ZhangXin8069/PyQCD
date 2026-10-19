#!/bin/bash

#SBATCH --job-name=glueball
#SBATCH --array=0-39
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

#SBATCH --partition=i72c512g,cpueicc,cpu6248R
##SBATCH --partition=na800-pcie
##SBATCH --partition=dgx1
##SBATCH --partition=gpu-debug
##SBATCH --partition=cpu-short
##SBATCH --partition=a192c1t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

##SBATCH --partition=na800-pcie
##SBATCH --nodes=1
##SBATCH --gpus=1
##SBATCH --cpus-per-task=2
##SBATCH --gres=gpu:1

##SBATCH --begin=now+7hour
##SBATCH --time="2-00:00:00"
##SBATCH --time="30:00"

##SBATCH -x gpu009

##SBATCH -x cpu052
##SBATCH --exclusive

# 禁用不需要的传输方式
export UCX_TLS=rc,shm  # 仅使用InfiniBand和共享内存

source /public/home/sush/envs/C12P12

conf_start=2005
gap=100
gap_2=10
run_dir=/public/home/sush/distillation/glueball
conf_name=beta6.498_mu-0.2150_ms-1926_L36x108

exe=${run_dir}/test.2.py
for i in {0..9..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID} + ${i} * ${gap_2} ))


    dir_file=${run_dir}/file/test.${conf}

    output_file=${run_dir}/output/test.${conf}.log

    if [ ! -e "${run_dir}/output/" ];then
        mkdir "${run_dir}/output/"
    fi

    if [ ! -e "${run_dir}/file" ];then
        mkdir "${run_dir}/file"
    fi

    echo "${conf} job starts at $(date)" > "${output_file}"
    mkdir ${dir_file}

    if [ -d ${dir_file} ]; then

        python -u "${exe}" "${conf}" "${conf_name}" >> "${output_file}" 2>&1
        ret=$?

        if [ $ret -eq 0 ]; then
            echo "Python 程序执行成功" >> "${output_file}" 2>&1
            rm -r "${dir_file}"
        else
            echo "Python 程序执行失败 (exit=$ret)" >> "${output_file}" 2>&1
            exit $ret
        fi

    else
        echo "calculate complete" >> "${output_file}" 2>&1
    fi

    echo "${conf} job ends at $(date)" >> "${output_file}"
done