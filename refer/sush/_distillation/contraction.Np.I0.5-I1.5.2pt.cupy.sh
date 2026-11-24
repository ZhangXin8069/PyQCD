#!/bin/bash

#SBATCH --job-name=Npi-Ihalf-Ithreehalf
#SBATCH --array=49
#SBATCH --nodes=1
#SBATCH --partition=dgx1
#SBATCH --account=test
#SBATCH --nodelist=gpu030
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --error=/public/home/sush/distillation/0v2b/output/Npi-Ihalf-Ithreehalf.%A_%a.err
#SBATCH --output=/public/home/sush/distillation/0v2b/output/Npi-Ihalf-Ithreehalf.%A_%a.out

# gpu030 uses its own OpenMPI/CUDA environment (mgmt04 controller).
source /public/home/sush/envs/C12P12dgx1
export CUDA_PATH=${CUDA_HOME}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export UCX_VFS_ENABLE=n
export PYTHONDONTWRITEBYTECODE=1

conf_start=1000
gap=160
gap_2=40

for i in {0..3..1}
do
    conf=$(( ${conf_start} + ${gap} * ${SLURM_ARRAY_TASK_ID} + ${i} * ${gap_2} ))
    run_dir=/public/home/sush/distillation/0v2b
    exe=${run_dir}/contraction.Np.I0.5-I1.5.2pt.cupy.py
    dir_file=${run_dir}/file/contraction.Np.I0.5-I1.5.2pt.cupy.${conf}.${SLURM_JOB_ID}
    output_file=${run_dir}/output/contraction.Np.I0.5-I1.5.2pt.cupy.${conf}.${SLURM_JOB_ID}.log
    mkdir -p "${run_dir}/output" "${run_dir}/file" || exit 1
    cd "${run_dir}" || exit 1
    echo "${conf} job starts at $(date)" > "${output_file}"
    mkdir -p "${dir_file}/mpl" "${dir_file}/tmp" "${dir_file}/cupy" || exit 1
    export MPLCONFIGDIR=${dir_file}/mpl
    export TMPDIR=${dir_file}/tmp
    export CUPY_CACHE_DIR=${dir_file}/cupy
    cp "${exe}" "${run_dir}/contraction.Np.I0.5-I1.5.2pt.cupy.sh" "${dir_file}/" || exit 1
    exe=${dir_file}/contraction.Np.I0.5-I1.5.2pt.cupy.py

    if [ -d "${dir_file}" ]; then
        python -c "import cupy; assert cupy.cuda.runtime.getDeviceCount() >= 4, 'Require four GPUs'" || {
            echo "GPU 不可用，放弃本任务" >> "${output_file}" 2>&1
            exit 1
        }
        /usr/bin/time -v -o "${output_file}.resources" mpirun -n 4 --mca pml ob1 --mca btl self,vader,tcp --mca coll ^hcoll --bind-to core --map-by core python -u "${exe}" "${conf}" "$@" >> "${output_file}" 2>&1
        ret=$?
        if [ $ret -eq 0 ]; then
            echo "Python 程序执行成功" >> "${output_file}" 2>&1
        else
            echo "Python 程序执行失败 (exit=$ret)" >> "${output_file}" 2>&1
            exit $ret
        fi
    else
        exit 1
    fi
    echo "${conf} job ends at $(date)" >> "${output_file}"
done
