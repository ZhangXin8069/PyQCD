#!/bin/bash

#SBATCH --job-name=Npi-Ihalf-Ithreehalf-na800
#SBATCH --array=75-106
#SBATCH --nodes=1
#SBATCH --partition=na800-pcie
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --error=/public/home/sush/distillation/0v2b/output/Npi-Ihalf-Ithreehalf-na800.%A_%a.err
#SBATCH --output=/public/home/sush/distillation/0v2b/output/Npi-Ihalf-Ithreehalf-na800.%A_%a.out

# OpenMPI/CUDA environment used by the NA800 partition.
source /public/home/sush/envs/C12P12 || exit 1
export CUDA_PATH=${CUDA_HOME}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export UCX_VFS_ENABLE=n
export PYTHONDONTWRITEBYTECODE=1

conf_start=1000
conf_stop=18000
conf_stride=40
configs_per_task=4
task_id=${SLURM_ARRAY_TASK_ID:?需要通过 Slurm 作业数组运行}
first_conf=$((conf_start + conf_stride * configs_per_task * task_id))

run_dir=/public/home/sush/distillation/0v2b
name=contraction.Np.I0.5-I1.5.2pt.cupy.na800
result_root=${NPI_RESULT_DIR:-${run_dir}/result_nex/E32P29/Px0Py0Pz0/ENV_100}
mkdir -p "${run_dir}/output" "${run_dir}/file" || exit 1
cd "${run_dir}" || exit 1

result_complete() {
    local result_dir=$1
    local corr_file=${result_dir}/corr_2pt_Iz_0.5_tsrc_sum.npy
    local basis_file=${result_dir}/corr_2pt_Iz_0.5_basis.npz
    [[ -s "${corr_file}" && -s "${basis_file}" ]] || return 1
    python - "${corr_file}" "${basis_file}" <<'PY'
import sys
from zipfile import BadZipFile

import numpy as np

try:
    with np.load(sys.argv[2], allow_pickle=False) as basis:
        times = np.arange(64)
        complete = (np.array_equal(basis['source_times'], times)
                    and np.array_equal(basis['time_offsets'], times)
                    and basis['complete_momentum_matrix'].item() is True)
    if complete:
        corr = np.load(sys.argv[1], allow_pickle=False)
        complete = (corr.shape == (2, 1, 64, 27, 27)
                    and corr.dtype == np.complex128
                    and np.isfinite(corr).all())
except (OSError, ValueError, KeyError, EOFError, BadZipFile):
    complete = False

sys.exit(0 if complete else 1)
PY
}

run_configuration() (
    conf=$1
    shift
    result_dir=${result_root}/conf${conf}
    dir_file=${run_dir}/file/${name}.${conf}.${SLURM_JOB_ID}
    output_file=${run_dir}/output/${name}.${conf}.${SLURM_JOB_ID}.log

    if result_complete "${result_dir}"; then
        echo "组态 ${conf} 已完成，跳过"
        exit 0
    fi

    full_run=1
    if (( $# > 0 )) || [[ ${NPI_BENCHMARK_SAMPLE:-0} == 1 ]]; then
        full_run=0
        result_dir=${dir_file}/result
    else
        mkdir -p "${result_dir}" || exit 1
        # Hold the lock through saving so overlapping submissions cannot overwrite each other.
        exec 9>>"${result_dir}/.na800.lock" || exit 1
        flock -n -E 75 9
        lock_status=$?
        if (( lock_status == 75 )); then
            echo "组态 ${conf} 正在由另一作业计算，跳过"
            exit 0
        elif (( lock_status != 0 )); then
            echo "组态 ${conf} 无法获取结果锁" >&2
            exit "${lock_status}"
        fi
        if result_complete "${result_dir}"; then
            echo "组态 ${conf} 已完成，跳过"
            exit 0
        fi
    fi

    echo "${conf} job starts at $(date)" > "${output_file}" || exit 1
    mkdir -p "${dir_file}/mpl" "${dir_file}/tmp" "${dir_file}/cupy" || exit 1
    export MPLCONFIGDIR=${dir_file}/mpl
    export TMPDIR=${dir_file}/tmp
    export CUPY_CACHE_DIR=${dir_file}/cupy
    cp "${run_dir}/${name}.py" "${run_dir}/${name}.sh" "${dir_file}/" || exit 1

    if ! python -c "import cupy; assert cupy.cuda.runtime.getDeviceCount() == 1, 'Require one visible GPU'" >> "${output_file}" 2>&1; then
        echo "GPU 不可用" >> "${output_file}"
        exit 1
    fi
    NPI_RESULT_DIR="${result_dir}" /usr/bin/time -v -o "${output_file}.resources" \
        mpirun -n 1 --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll \
        --bind-to core --map-by core python -u "${dir_file}/${name}.py" "${conf}" "$@" >> "${output_file}" 2>&1
    ret=$?
    if (( ret != 0 )); then
        echo "Python 程序执行失败 (exit=${ret})" >> "${output_file}"
        exit "${ret}"
    fi
    if (( full_run )) && ! result_complete "${result_dir}"; then
        echo "组态 ${conf} 结果不完整" >> "${output_file}"
        exit 1
    fi
    echo "Python 程序执行成功" >> "${output_file}"
    echo "${conf} job ends at $(date)" >> "${output_file}"
)

status=0
for ((offset = 0; offset < configs_per_task; offset++)); do
    conf=$((first_conf + conf_stride * offset))
    (( conf <= conf_stop )) || break
    run_configuration "${conf}" "$@" || status=1
done
exit "${status}"
