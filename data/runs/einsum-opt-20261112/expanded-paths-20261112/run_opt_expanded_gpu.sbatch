#!/usr/bin/env bash
#SBATCH --job-name=pyqcd-opt-expanded-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --output=/home/bingxing2/home/scx8080/PyQCD/data/runs/einsum-opt-20261112/expanded-paths-20261112/jobs/%j.opt-expanded-gpu.out
#SBATCH --error=/home/bingxing2/home/scx8080/PyQCD/data/runs/einsum-opt-20261112/expanded-paths-20261112/jobs/%j.opt-expanded-gpu.err

set -uo pipefail
source "$HOME/env.sh"
REPO=/home/bingxing2/home/scx8080/PyQCD
LIB_ROOT="$REPO/data/runs/einsum-opt-20261112/final-20261112t195557/snapshot-v9"
RUN="$REPO/data/runs/einsum-opt-20261112/expanded-paths-20261112"
SEQS=1,3,9,849,856,130,1599,1623,1743,1864,879,2583,2590,3431,6070,6911,7798,7918,5920,2139,2161,1872,5172,6044,5175,853,854,850,3457,1838,1844,1849,6924,6918,6933,1621,1608,1613,1030,1562,1551,1668,1662
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYQCD_EINSUM_NATIVE=cuda
export PYQCD_EINSUM_KUPL_LIB="$LIB_ROOT/cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so"
export PYQCD_EINSUM_CUDA_LIB="$LIB_ROOT/cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=4

python -B -m pyqcd.testing.einsum.bench_opt_einsum_production \
    --device gpu --precision double --seqs "$SEQS" --strategies dp \
    --rounds 1 --warmup 0 --oracle installed \
    --out "$RUN/opt_expanded_gpu.json"
exit $?
