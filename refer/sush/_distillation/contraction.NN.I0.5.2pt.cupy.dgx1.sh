#!/bin/bash

#SBATCH --job-name=PNJN
#SBATCH --array=20#-29#70-149
#SBATCH --nodes=1

#SBATCH --error=test_pion_3pt_error
#SBATCH --output=test_pion_3pt_output

##SBATCH --partition=i72c512g,cpueicc,cpu6248R
##SBATCH --partition=na800-pcie
#SBATCH --partition=dgx1
##SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

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

export LD_LIBRARY_PATH=/public/home/sush/.local/lib:$LD_LIBRARY_PATH

# node and npm, claude
export PATH=/public/home/sush/AI-Agent/node-v22.22.3-linux-x64-glibc-217/bin:$PATH

export CLAUDE_CODE_ATTRIBUTION_HEADER=0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/public/home/sush/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/public/home/sush/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/public/home/sush/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/public/home/sush/anaconda3/bin:$PATH"
    fi
fi

unset __conda_setup

# <<< conda initialize <<<

module purge
module load cuda/12.6.1-none-none
module load hpcx-ompi/2.18.1

export LD_LIBRARY_PATH=/public/home/sush/anaconda3/envs/C12P12/lib:$LD_LIBRARY_PATH

conda deactivate
conda activate C12P12

conf_start=1000
gap=120
gap_2=40

for i in {0..2..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID} + ${i} * ${gap_2} ))

    run_dir=/public/home/sush/distillation/0v2b

    exe=${run_dir}/contraction.NN.I0.5.2pt.cupy.py
    dir_file=${run_dir}/file/contraction.NN.I0.5.2pt.cupy.${conf}

    output_file=${run_dir}/output/contraction.NN.I0.5.2pt.cupy.${conf}.log

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

        mpirun -n 2 --bind-to core --map-by core python -u "${exe}" "${conf}" >> "${output_file}" 2>&1
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