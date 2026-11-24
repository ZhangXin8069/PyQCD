#!/bin/bash
#SBATCH --job-name=Npi-Ihalf-cpu
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --partition=cpu-short
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --error=/public/home/sush/distillation/0v2b/output/Npi-Ihalf-cpu.%A_%a.err
#SBATCH --output=/public/home/sush/distillation/0v2b/output/Npi-Ihalf-cpu.%A_%a.out

# 24 cores: 8 ranks x 3 threads. Full source/time range by default.
# Test: sbatch contraction.Np.I0.5.2pt.cpu.sh 0 1 9
source /public/home/sush/anaconda3/etc/profile.d/conda.sh
module purge
module load gcc/10.3.0-gcc-4.8.5
module load hpcx/2.14/hpcx-mt-ompi
conda activate C12P12 || exit 1
export LD_LIBRARY_PATH=/public/home/sush/anaconda3/envs/C12P12/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-3}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MALLOC_ARENA_MAX=2
export UCX_VFS_ENABLE=n
export PYTHONDONTWRITEBYTECODE=1

conf_start=1000
gap=40
gap_2=40
for i in {0..0..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID:-0} + ${i} * ${gap_2} ))
    run_dir=/public/home/sush/distillation/0v2b
    exe=${run_dir}/contraction.Np.I0.5.2pt.cpu.py
    dir_file=${run_dir}/file/contraction.Np.I0.5.2pt.cpu.${conf}.${SLURM_JOB_ID}
    output_file=${run_dir}/output/contraction.Np.I0.5.2pt.cpu.${conf}.${SLURM_JOB_ID}.log
    mkdir -p "${run_dir}/output" "${run_dir}/file" || exit 1
    cd "${run_dir}" || exit 1
    echo "${conf} job starts at $(date)" > "${output_file}"
    mkdir -p "${dir_file}/mpl" "${dir_file}/tmp" || exit 1
    export MPLCONFIGDIR=${dir_file}/mpl
    export TMPDIR=${dir_file}/tmp
    cp "${exe}" "${run_dir}/contraction.Np.I0.5.2pt.cpu.sh" "${dir_file}/" || exit 1
    exe=${dir_file}/contraction.Np.I0.5.2pt.cpu.py
    /usr/bin/time -v -o "${output_file}.resources" mpirun -n "${SLURM_NTASKS:-8}" --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll --bind-to core --map-by "slot:PE=${OMP_NUM_THREADS}" python -u "${exe}" "${conf}" "$@" >> "${output_file}" 2>&1
    ret=$?
    if [ $ret -eq 0 ]; then
        echo "Python 程序执行成功" >> "${output_file}"
    else
        echo "Python 程序执行失败 (exit=$ret)" >> "${output_file}"
        exit $ret
    fi
    echo "${conf} job ends at $(date)" >> "${output_file}"
done
