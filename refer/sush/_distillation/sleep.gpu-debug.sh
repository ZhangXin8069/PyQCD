#!/usr/bin/env bash
#SBATCH --job-name=lqcddb-test-dynamic
#SBATCH --partition=gpu-debug
#SBATCH --nodelist=gpu024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/sush/distillation/lqcddb_new/benchmark/test.%j.out
#SBATCH --error=/public/home/sush/distillation/lqcddb_new/benchmark/test.%j.err


sleep 1800