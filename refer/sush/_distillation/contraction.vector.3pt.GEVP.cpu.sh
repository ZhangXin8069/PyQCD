#!/bin/bash

#SBATCH --job-name=PNJN-cpu
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --partition=cpu-short
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=18
#SBATCH --hint=nomultithread
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --error=/public/home/sush/distillation/0v2b/output/vector-3pt-cpu.%A_%a.err
#SBATCH --output=/public/home/sush/distillation/0v2b/output/vector-3pt-cpu.%A_%a.out

# 36 cores: 2 MPI ranks x 18 cores; contraction tasks share rank-local arrays.
# Single-source benchmark: CPU_BENCHMARK=1 sbatch contraction.vector.3pt.GEVP.cpu.sh 18 0 5 15
# One configuration: add --array=0 (configuration 1000).
# Optional arguments: tsep tsrc current_start current_stop (stop exclusive).
# sbatch --array=0 contraction.vector.3pt.GEVP.cpu.sh 18 0 5 6

source /public/home/sush/anaconda3/etc/profile.d/conda.sh
module purge
module load gcc/10.3.0-gcc-4.8.5
module load hpcx/2.14/hpcx-mt-ompi
conda activate C12P12 || exit 1
export LD_LIBRARY_PATH=/public/home/sush/anaconda3/envs/C12P12/lib:${LD_LIBRARY_PATH:-}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-18}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export BLIS_NUM_THREADS=${OMP_NUM_THREADS}
export NUMEXPR_NUM_THREADS=${OMP_NUM_THREADS}
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export PYTHONDONTWRITEBYTECODE=1
export UCX_VFS_ENABLE=n
export CPU_CONTRACTION_WORKERS=${CPU_CONTRACTION_WORKERS:-${OMP_NUM_THREADS}}
export CPU_SOURCE_MOM_CHUNK=${CPU_SOURCE_MOM_CHUNK:-1}
# GiB planning budget for both ranks, below the 192 GiB Slurm hard limit.
export CPU_MEMORY_BUDGET_GIB=${CPU_MEMORY_BUDGET_GIB:-144}
export MALLOC_ARENA_MAX=2

conf_start=1000
gap=40
gap_2=40

for i in {0..0..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID} + ${i} * ${gap_2} ))

    run_dir=/public/home/sush/distillation/0v2b
    exe=${run_dir}/contraction.vector.3pt.GEVP.cpu.py
    dir_file=${run_dir}/file/contraction.vector.3pt.GEVP.cpu.${conf}.${SLURM_JOB_ID}
    output_file=${run_dir}/output/contraction.vector.3pt.GEVP.cpu.${conf}.${SLURM_JOB_ID}.log

    mkdir -p "${run_dir}/output" "${run_dir}/file" || exit 1
    cd "${run_dir}" || exit 1
    echo "${conf} job starts at $(date)" > "${output_file}"
    mkdir "${dir_file}" || exit 1

    if [ -d "${dir_file}" ]; then
        mkdir -p "${dir_file}/mpl" "${dir_file}/tmp" || exit 1
        export MPLCONFIGDIR=${dir_file}/mpl
        export TMPDIR=${dir_file}/tmp
        if [ "${CPU_BENCHMARK:-0}" = 1 ]; then
            export CPU_BENCHMARK_DIR=${run_dir}/output/vector-3pt-cpu-benchmark.${SLURM_JOB_ID}
            export CPU_BENCHMARK_RESULT_DIR=${CPU_BENCHMARK_DIR}/result
            mkdir -p "${CPU_BENCHMARK_DIR}" || exit 1
            cp "${exe}" "${run_dir}/contraction.vector.3pt.GEVP.cpu.sh" "${CPU_BENCHMARK_DIR}/" || exit 1
            exe=${CPU_BENCHMARK_DIR}/contraction.vector.3pt.GEVP.cpu.py
            echo "BENCHMARK job=${SLURM_JOB_ID} node=$(hostname) ranks=2 threads_per_rank=${OMP_NUM_THREADS} result_dir=${CPU_BENCHMARK_RESULT_DIR}" >> "${output_file}"
        fi
        /usr/bin/time -v -o "${output_file}.resources" mpirun -n 2 --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll --bind-to core --map-by "slot:PE=${OMP_NUM_THREADS}" python -u "${exe}" "${conf}" "$@" >> "${output_file}" 2>&1
        ret=$?

        if [ $ret -eq 0 ]; then
            echo "Python 程序执行成功" >> "${output_file}" 2>&1
            rm -r "${dir_file}"
        else
            echo "Python 程序执行失败 (exit=$ret)" >> "${output_file}" 2>&1
            exit $ret
        fi
    fi

    echo "${conf} job ends at $(date)" >> "${output_file}"
done
