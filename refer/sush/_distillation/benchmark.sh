#!/bin/bash -l
#SBATCH --job-name=benchmark
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/public/home/sush/distillation/test
#SBATCH --output=/public/home/sush/distillation/test/benchmark.out
#SBATCH --error=/public/home/sush/distillation/test/benchmark.err

set -e
source /public/home/sush/envs/C12P12

export PYTHONPATH="/public/home/sush/distillation/lqcddb_new/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec python -u benchmark.py
