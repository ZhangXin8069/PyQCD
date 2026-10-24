#!/bin/bash

#SBATCH --job-name=light
#SBATCH --array=3-199
#SBATCH --partition=dgx1
#SBATCH --nodelist=gpu030
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH --time=04:00:00
#SBATCH --chdir=/public/home/sush/distillation/pion_form_factor
#SBATCH --output=/public/home/sush/distillation/pion_form_factor/output/C24P29.%x.%A_%a.out
#SBATCH --error=/public/home/sush/distillation/pion_form_factor/output/C24P29.%x.%A_%a.err

set -o pipefail

Lx=24
Ly=24
Lz=24
Lt=72
Ne=650

# first: [0,Lt/2); second: [Lt/2,Lt). These are global source times.
case "${1:-first}" in
    first)
        export C24P29_TSRC_START=0
        export C24P29_TSRC_STOP=$((Lt / 2))
        ;;
    second)
        export C24P29_TSRC_START=$((Lt / 2))
        export C24P29_TSRC_STOP=${Lt}
        ;;
    *)
        echo "Usage: sbatch $0 [first|second]" >&2
        exit 2
        ;;
esac

module purge || exit "$?"
# gpu030 uses Rocky 9 modules; the shared sm70 setup names CentOS 7 modules.
module load gcc/10.3.0-gcc-11.5.0 || exit "$?"
source /public/home/sush/anaconda3/etc/profile.d/conda.sh || exit "$?"
conda activate sm70 || exit "$?"

# PyQUDA in sm70 links build_sm70/libquda.so (CUDA 11, MPICH ABI).
# CUDA 12's module cannot provide these CUDA 11 library SONAMEs.
export CUDA_PATH=/public/soft/linux-centos7-x86_64/gcc-10.3.0/cuda-11.4.4-uizl3zvwy66u3bqllanvuxwxxwfyytwo
export MPI_ROOT=/public/home/sush/mpi/mpich-4.2.0/mpich_install
export QUDA_PATH=/public/home/sush/Quda/build_sm70/usqcd
export PATH="${MPI_ROOT}/bin:${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${QUDA_PATH}/lib:${MPI_ROOT}/lib:${CUDA_PATH}/lib64:/public/home/sush/distillation/pion_form_factor/runtime.gpu030.sm70/lib:${LD_LIBRARY_PATH:-}"
# EasyDistillation's check_QUDA needs packaging, absent from the sm70 env.
export PYTHONPATH="/public/home/sush/distillation/pion_form_factor/runtime.gpu030.sm70/python${PYTHONPATH:+:${PYTHONPATH}}"
set -e

run_dir=/public/home/sush/distillation/pion_form_factor
export C24P29_QUDA_RESOURCE_PATH=${run_dir}/runtime.gpu030.sm70/quda-cache-b58f1ec-sm70
mkdir -p "${run_dir}/output" "${C24P29_QUDA_RESOURCE_PATH}"
job_tag=${C24P29_JOB_TAG:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}

for i in {0..0..1}
do
    conf_gap_2=200
    source_shape=100
    conf_stare=10000
    conf_gap=200
    conf_id=$((conf_stare + conf_gap * ${SLURM_ARRAY_TASK_ID:-0} + i * conf_gap_2))
    dir_file=${run_dir}/file/contraction_block_numpy_32x96_compensation_${conf}
    
    conf_path=/public/group/lqcd/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_
    eigen_source=/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/${conf_id}/
    output_peram_path=/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/

    exe=${run_dir}/contraction.block.cupy.mpi.gen.C24P29.py
    output_path=${run_dir}/output/contraction.block.cupy.mpi.gen.C24P29.${conf_id}.tsrc${C24P29_TSRC_START}-$((C24P29_TSRC_STOP - 1)).${job_tag}.log

    echo "conf=${conf_id}, src=[${C24P29_TSRC_START},${C24P29_TSRC_STOP}), log=${output_path}"
    # One rank on the allocated node: bypass the legacy MPICH Slurm launcher.
    if mpirun -rmk user -launcher fork -n 1 python -u "${exe}" "${Lx}" "${Ly}" "${Lz}" "${Lt}" "${Ne}" "${conf_id}" "${conf_path}" "${eigen_source}" "${output_peram_path}" "${source_shape}" "${i}" > "${output_path}" 2>&1; then
        echo "Python 程序执行成功" >> "${output_path}"
    else
        status=$?
        echo "Python 程序执行失败，exit=${status}" >> "${output_path}"
        exit "${status}"
    fi
done

# # The first half submits the second half itself once it has finished
# # successfully, so the two halves are never in the queue at the same time and
# # no process has to wait on the login node. A partial source-time interval is
# # only ever a step of this chain; `second` therefore does not submit anything.
# if [ "${1:-first}" = "first" ]; then
#     next_script=${run_dir}/contraction.block.cupy.mpi.gen.C24P29.sh
#     if next_job=$(sbatch --parsable --partition=gpu-debug --time=00:30:00 \
#             --job-name=C24P29-second \
#             --export=ALL,C24P29_JOB_TAG=${job_tag}-second \
#             "${next_script}" second); then
#         next_job=${next_job%%;*}
#         if [[ ${next_job} =~ ^[0-9]+$ ]]; then
#             echo "Second-half job submitted: ${next_job} (gpu-debug, src=[$((Lt / 2)),${Lt}))"
#         else
#             echo "WARNING: unexpected second-half job ID '${next_job}'" >&2
#         fi
#     else
#         status=$?
#         echo "ERROR: second-half submission failed (exit ${status}); the first-half results are complete." >&2
#         echo "       Submit the second half manually with:" >&2
#         echo "       sbatch --partition=gpu-debug --time=00:30:00 --job-name=C24P29-second ${next_script} second" >&2
#     fi
# fi
