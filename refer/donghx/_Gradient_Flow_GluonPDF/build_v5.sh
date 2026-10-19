#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
module load gcc/11.5.0-gcc-4.8.5
mkdir -p "$repo_dir/build"

g++ -O3 -march=x86-64 -mtune=generic -ffast-math -fopenmp -std=c++17 \
  -Wall -Wextra -static-libstdc++ -static-libgcc \
  -Wl,-rpath,/public/spack/opt/spack/linux-centos7-cascadelake/gcc-10.3.0/gcc-11.5.0-7k3pqxf57tabqwjhwmz4mh4q57jbohza/lib64 \
  "$repo_dir/src/flowed_gluon_ope.cpp" -o "$repo_dir/build/flowed_gluon_ope_v5"

"$repo_dir/build/flowed_gluon_ope_v5" --self-test
