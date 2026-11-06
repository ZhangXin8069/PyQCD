#!/usr/bin/env bash
# Build the CUDA shared library. A missing source set is an explicit failure.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
PYQCD_REQUESTED_BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-}
export PYQCD_EINSUM_BACKEND=cuda
# shellcheck source=/dev/null
source "$ROOT/env.sh"
case "$PYQCD_REQUESTED_BUILD_DIR" in
    ""|"$PYQCD_EINSUM_KUPL_BUILD_DIR"|"$PYQCD_EINSUM_CUDA_BUILD_DIR")
        export PYQCD_EINSUM_BUILD_DIR="$PYQCD_EINSUM_CUDA_BUILD_DIR"
        ;;
    *) export PYQCD_EINSUM_BUILD_DIR="$PYQCD_REQUESTED_BUILD_DIR" ;;
esac

printf '=== PyQCD einsum build (cuda) ===\n'
bash "$ROOT/cpp/cuda/einsum/make-nv.sh" "$@"
printf '=== PyQCD einsum build (cuda): SUCCESS ===\n'
