#!/usr/bin/env bash
# Environment for the PyQCD einsum native stack.
#
# Mirrors PyQCU/env.sh: repository first on PYTHONPATH, both backend library
# directories on LD_LIBRARY_PATH, every artefact path exported.
set -u

PYQCD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

export PYQCD_ROOT
export PYQCD_EINSUM_CUDA_DIR="$PYQCD_ROOT/cpp/cuda/einsum"
export PYQCD_EINSUM_KUPL_DIR="$PYQCD_ROOT/cpp/kupl/einsum"
export PYQCD_EINSUM_CUDA_LIB="$PYQCD_EINSUM_CUDA_DIR/libpyqcdeinsum.so"
export PYQCD_EINSUM_KUPL_LIB="$PYQCD_EINSUM_KUPL_DIR/libpyqcdeinsum.so"

export PYQCD_EINSUM_BACKEND="${PYQCD_EINSUM_BACKEND:-kupl}"

if [ "$PYQCD_EINSUM_BACKEND" = "cuda" ]; then
    export PYQCD_EINSUM_LIB="$PYQCD_EINSUM_CUDA_LIB"
else
    export PYQCD_EINSUM_LIB="$PYQCD_EINSUM_KUPL_LIB"
fi

export PYTHONPATH="$PYQCD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PYQCD_EINSUM_KUPL_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$PYQCD_EINSUM_CUDA_DIR:$LD_LIBRARY_PATH"

export PYQCD_EINSUM_BUILD_DIR="${PYQCD_EINSUM_BUILD_DIR:-$PYQCD_EINSUM_KUPL_DIR/build}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
