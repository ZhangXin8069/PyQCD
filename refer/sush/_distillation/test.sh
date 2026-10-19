#!/bin/bash

#SBATCH --job-name=glueball
#SBATCH --array=0-7
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --partition=i72c512g,cpueicc,cpu6248R
##SBATCH --partition=na800-pcie
#SBATCH --partition=dgx1
##SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

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

module purge
module load hpcx-ompi/2.18.1
module load cuda/12.6.1-none-none

nvrtc_lib="/public/home/sush/distillation/changx/cuda_nvrtc_12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
if [ -d "${nvrtc_lib}" ]; then
    export LD_LIBRARY_PATH="${nvrtc_lib}:${LD_LIBRARY_PATH:-}"
fi

conf_start=1000
gap=10
gap_2=10
run_dir=/public/home/sush/distillation/glueball
conf_name=beta6.72_mu-0.1850_ms-0.1700_L48x144

exe=${run_dir}/test.py
for i in {0..0..1}
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
        # GPU 可用性检查
        python -c "import cupy; assert cupy.cuda.runtime.getDeviceCount() > 0, 'No GPU'" || {
            echo "GPU 不可用，放弃本任务" >> "${output_file}" 2>&1
            exit 1
        }

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