#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
module load gcc/11.5.0-gcc-4.8.5
mkdir -p "$repo_dir/build"
g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 -Wall -Wextra \
  -static-libstdc++ -static-libgcc \
  "$repo_dir/src/flowed_gluon_ope.cpp" -o "$repo_dir/build/flowed_gluon_ope"
"$repo_dir/build/flowed_gluon_ope" --self-test
