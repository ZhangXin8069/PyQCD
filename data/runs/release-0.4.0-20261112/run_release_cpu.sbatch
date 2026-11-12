#!/usr/bin/env bash
#SBATCH --job-name=pyqcd-release-cpu
#SBATCH --partition=gpu
# The site rejects --gpus=0. Reserve one card and hide it from Python.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=/home/bingxing2/home/scx8080/PyQCD/data/runs/release-0.4.0-20261112/jobs/%j.cpu.out
#SBATCH --error=/home/bingxing2/home/scx8080/PyQCD/data/runs/release-0.4.0-20261112/jobs/%j.cpu.err

set -uo pipefail

source "$HOME/env.sh"

REPO=/home/bingxing2/home/scx8080/PyQCD
RUN="$REPO/data/runs/release-0.4.0-20261112"
JOB_DIR="$RUN/jobs/${SLURM_JOB_ID:-manual-cpu}"
KUPL_LIB="$REPO/cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so"

mkdir -p "$JOB_DIR"
cd "$REPO"

export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONHASHSEED=0
export PYQCD_EINSUM_NATIVE=kupl
export PYQCD_EINSUM_KUPL_LIB="$KUPL_LIB"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=NoDevFiles
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=4

started=$(date +%s)
status=0

run_step() {
    local label=$1
    shift
    printf 'BEGIN %s\n' "$label"
    "$@" > "$JOB_DIR/$label.log" 2>&1
    local rc=$?
    printf 'END %s rc=%d\n' "$label" "$rc"
    if [ "$rc" -ne 0 ]; then
        status=$rc
    fi
    return "$rc"
}

{
    printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
    printf 'node=%s\n' "${SLURMD_NODENAME:-unknown}"
    printf 'started=%s\n' "$(date -Iseconds)"
    printf 'requested_gpus=1\n'
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'nvidia_visible_devices=%s\n' "$NVIDIA_VISIBLE_DEVICES"
    printf 'omp_num_threads=%s\n' "$OMP_NUM_THREADS"
    printf 'openblas_num_threads=%s\n' "$OPENBLAS_NUM_THREADS"
} > "$JOB_DIR/metadata.before.txt"

squeue -j "$SLURM_JOB_ID" -o '%.18i %.9P %.32j %.8T %.10M %.10l %.6D %.20R' \
    > "$JOB_DIR/squeue.running.txt" 2>&1 || true
scontrol show job "$SLURM_JOB_ID" > "$JOB_DIR/scontrol.running.txt" 2>&1 || true

run_step build_kupl bash cpp/kupl/einsum/make.sh || true
if [ "$status" -eq 0 ]; then
    run_step verify_native_kupl \
        python -B -m pyqcd.testing.einsum.verify_native --backend kupl || true
fi
if [ "$status" -eq 0 ]; then
    run_step verify_plan_gate \
        python -B -m pyqcd.testing.einsum.verify_plan_gate || true
fi
if [ "$status" -eq 0 ]; then
    run_step verify_python_hotpath \
        python -B -m pyqcd.testing.einsum.verify_python_hotpath || true
fi
if [ "$status" -eq 0 ]; then
    run_step release_check \
        python -B -m pyqcd.testing.release_check --offline --with-tests || true
fi

finished=$(date +%s)
{
    printf 'finished=%s\n' "$(date -Iseconds)"
    printf 'elapsed_s=%s\n' "$((finished - started))"
    printf 'exit_code=%s\n' "$status"
    printf 'kupl_sha256=%s\n' "$(sha256sum "$KUPL_LIB" 2>/dev/null | awk '{print $1}')"
} > "$JOB_DIR/metadata.after.txt"

exit "$status"
