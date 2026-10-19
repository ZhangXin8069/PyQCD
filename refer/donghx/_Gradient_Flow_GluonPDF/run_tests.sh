#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"$repo_dir/scripts/build_v5.sh"
python "$repo_dir/tests/test_pipeline.py"
python -m py_compile "$repo_dir"/scripts/*.py "$repo_dir"/tests/*.py
bash -n "$repo_dir"/scripts/*.sh "$repo_dir"/slurm/*.sh "$repo_dir"/slurm/*.slurm \
  "$repo_dir"/slurm/v*/*.sh "$repo_dir"/slurm/v*/*.slurm
