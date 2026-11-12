#!/usr/bin/env bash
#SBATCH --job-name=pyqcd-opt-core-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=/home/bingxing2/home/scx8080/PyQCD/data/runs/einsum-opt-20261112/expanded-paths-20261112/jobs/%j.opt-core-gpu.out
#SBATCH --error=/home/bingxing2/home/scx8080/PyQCD/data/runs/einsum-opt-20261112/expanded-paths-20261112/jobs/%j.opt-core-gpu.err

set -uo pipefail
source "$HOME/env.sh"

REPO=/home/bingxing2/home/scx8080/PyQCD
LIB_SNAPSHOT="$REPO/data/runs/einsum-opt-20261112/final-20261112t195557/snapshot-v9"
RUN="$REPO/data/runs/einsum-opt-20261112/expanded-paths-20261112"

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYQCD_EINSUM_NATIVE=cuda
export PYQCD_EINSUM_KUPL_LIB="$LIB_SNAPSHOT/cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so"
export PYQCD_EINSUM_CUDA_LIB="$LIB_SNAPSHOT/cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=4

python -B -m pyqcd.testing.einsum.bench_opt_einsum_production \
    --device gpu --precision double \
    --seqs 2584,866,855,129,1863,6585 \
    --strategies dp,greedy --rounds 2 --warmup 1 \
    --oracle installed --out "$RUN/opt_core_gpu.json"
exit $?
