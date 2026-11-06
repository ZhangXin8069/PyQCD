#!/usr/bin/env bash
# Build the KUPL shared library without involving CUDA or Cython.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
PYQCD_REQUESTED_BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-}
export PYQCD_EINSUM_BACKEND=kupl
# shellcheck source=/dev/null
source "$ROOT/env.sh"
case "$PYQCD_REQUESTED_BUILD_DIR" in
    ""|"$PYQCD_EINSUM_KUPL_BUILD_DIR"|"$PYQCD_EINSUM_CUDA_BUILD_DIR")
        export PYQCD_EINSUM_BUILD_DIR="$PYQCD_EINSUM_KUPL_BUILD_DIR"
        ;;
    *) export PYQCD_EINSUM_BUILD_DIR="$PYQCD_REQUESTED_BUILD_DIR" ;;
esac

printf '=== PyQCD einsum build (kupl) ===\n'
bash "$ROOT/cpp/kupl/einsum/make.sh" "$@"
printf '=== PyQCD einsum build (kupl): SUCCESS ===\n'
