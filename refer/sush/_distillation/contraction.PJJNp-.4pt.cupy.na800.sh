#!/bin/bash

#SBATCH --job-name=PJJNp-4pt
#SBATCH --array=0-47
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
#SBATCH --partition=na800-pcie
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
##SBATCH --time=30:00

# Avoid GPU-aware UCX paths that are not needed by this single-node job.
export UCX_TLS=rc,shm

conf_start=1000
gap=40
gap_2=40

for i in {0..0..1}
do
    conf=$((conf_start + gap * SLURM_ARRAY_TASK_ID + i * gap_2))

    run_dir=/public/home/sush/distillation/0v2b
    name=contraction.PJJNp-.4pt.cupy.na800
    exe=${run_dir}/${name}.py
    dir_file=${run_dir}/file/${name}.${conf}
    output_file=${run_dir}/output/${name}.${conf}.log

    mkdir -p "${run_dir}/output" "${run_dir}/file"
    exec > "${output_file}" 2>&1
    echo "${conf} job starts at $(date)"

    source /public/home/sush/envs/C12P12

    mpirun -n 2 --bind-to core --map-by core \
        python -u "${exe}" "${conf}"
    ret=$?

    if [ ${ret} -eq 0 ]; then
        echo "Python program completed successfully"
    else
        echo "Python program failed (exit=${ret})"
        exit ${ret}
    fi

    echo "${conf} job ends at $(date)"
done
