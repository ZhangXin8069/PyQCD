#!/bin/bash

# 测试专用作业脚本。卡数由 submit_vs.sh 从各方案 settings.yaml 传入。
# 这里的默认值只在直接 sbatch 本文件时使用。
#SBATCH --partition=dgx1
#SBATCH --account=test
#SBATCH --nodelist=gpu030
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00

if [ "$#" -ne 2 ]; then
    echo "用法：job.sh 方案目录 组态号"
    exit 1
fi

run_dir="$(cd "$1" && pwd -P)"
conf="$2"
vs_dir="/public/home/sush/distillation/changx"
settings_file="${run_dir}/settings.yaml"
program_name="$(awk '$1 == "program:" {print $2; exit}' "${settings_file}")"
mpi_ranks="$(awk '$1 == "mpi_ranks:" {print $2; exit}' "${settings_file}")"
program_name="${program_name:-mpi_readVVV.py}"
mpi_ranks="${mpi_ranks:-1}"
input_file="${run_dir}/input/input_${conf}.yaml"
failed_file="${run_dir}/status/failed_${conf}.txt"

if [[ "${program_name}" = /* ]]; then
    program="${program_name}"
else
    program="${vs_dir}/code/${program_name}"
fi

mkdir -p "${run_dir}/status"
rm -f "${failed_file}"

if [ ! -f "${settings_file}" ] || [ ! -f "${input_file}" ]; then
    echo "找不到 settings 或 input：${conf}"
    echo "input/settings missing: $(date)" > "${failed_file}"
    exit 1
fi
if [ ! -f "${program}" ]; then
    echo "找不到测试收缩代码：${program}"
    echo "program missing: $(date)" > "${failed_file}"
    exit 1
fi
if ! [[ "${mpi_ranks}" =~ ^[0-9]+$ ]] || [ "${mpi_ranks}" -eq 0 ]; then
    echo "mpi_ranks 必须是正整数"
    echo "invalid mpi_ranks: $(date)" > "${failed_file}"
    exit 1
fi

echo "组态 ${conf} 开始：$(date)"
echo "方案目录：${run_dir}"
echo "节点：$(hostname)"
echo "Job ID：${SLURM_JOB_ID:-unknown}"
echo "MPI ranks：${mpi_ranks}"
echo "CUDA_VISIBLE_DEVICES：${CUDA_VISIBLE_DEVICES:-未设置}"
module purge >/dev/null 2>&1 || true
module load hpcx-ompi/2.18.1
module load cuda/12.6.1-none-none
# The site CUDA module does not export NVRTC libraries, while CuPy needs
# libnvrtc-builtins.so.12.6 during its first GPU kernel initialization.
nvrtc_lib="${vs_dir}/cuda_nvrtc_12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"
if [ -d "${nvrtc_lib}" ]; then
    export LD_LIBRARY_PATH="${nvrtc_lib}:${LD_LIBRARY_PATH:-}"
fi
python_bin="/public/home/sush/anaconda3/envs/C12P12/bin/python"
export PYTHONPATH="/usr/lib64/python3.9/site-packages:${PYTHONPATH:-}"
echo "Python：${python_bin}"
echo "mpirun：$(command -v mpirun)"

if [ "${mpi_ranks}" -eq 1 ]; then
    "${python_bin}" -u "${program}" "${input_file}"
else
    mpirun -n "${mpi_ranks}" \
        --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll \
        --bind-to core --map-by core \
        "${python_bin}" -u "${program}" "${input_file}"
fi
exit_code=$?

if [ "${exit_code}" -eq 0 ]; then
    echo "组态 ${conf} 成功结束：$(date)"
    exit 0
fi

echo "组态 ${conf} 运行失败，退出码 ${exit_code}：$(date)"
echo "exit_code=${exit_code}, date=$(date)" > "${failed_file}"
exit "${exit_code}"
