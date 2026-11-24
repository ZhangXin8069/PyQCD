#!/bin/bash

#SBATCH --job-name=PJJNp-H20
#SBATCH --array=0-47
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
#SBATCH --partition=h20-nettr
##SBATCH --nodelist=gpu050
#SBATCH --reservation=sush_235
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h20:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

export UCX_TLS=rc,shm
export NVIDIA_TF32_OVERRIDE=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

conf_start=1000
gap=40
gap_2=40

for i in {0..0..1}
do
    conf=$((conf_start + gap * SLURM_ARRAY_TASK_ID + i * gap_2))

    run_dir=/public/home/sush/distillation/0v2b
    name=contraction.PJJNp-.4pt.cupy.h20
    exe=${run_dir}/${name}.py
    dir_file=${run_dir}/file/${name}.${conf}
    output_file=${run_dir}/output/${name}.${conf}.log
    mkdir -p "${run_dir}/output" "${dir_file}"
    exec > "${output_file}" 2>&1
    echo "${conf} job starts at $(date); node=$(hostname); job=${SLURM_JOB_ID}"
    cd "${run_dir}" || exit 1
    export TMPDIR=${dir_file}/tmp
    export MPLCONFIGDIR=${dir_file}/matplotlib
    export CUPY_CACHE_DIR=${dir_file}/cupy
    export CUDA_CACHE_PATH=${dir_file}/cuda
    export XDG_CACHE_HOME=${dir_file}/cache
    mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${CUPY_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${XDG_CACHE_HOME}"

    source /public/home/sush/envs/sm90 || exit 1
    python -c "import cupy; assert cupy.cuda.runtime.getDeviceCount() == 1, 'Allocate one GPU'" || exit 1
    mpirun -n 1 --bind-to core --map-by slot:PE=4 \
        python -u "${exe}" "${conf}"
    ret=$?
    if [ ${ret} -eq 0 ]; then
        echo "Python production program completed successfully"
        rm -r "${dir_file}"
    else
        echo "Python production program failed (exit=${ret}); retained ${dir_file}"
        exit ${ret}
    fi
    echo "${conf} job ends at $(date)"
done
