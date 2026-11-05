#!/usr/bin/env bash
# Build the CUDA einsum backend: cpp/cuda/einsum.
#
# Mirrors PyQCU/build-nv.sh. When cpp/cuda/einsum has no .cu sources yet the
# KUPL backend is built instead and the fallback is reported explicitly rather
# than silently pretending a GPU build happened.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
# shellcheck source=/dev/null
source "$ROOT/env.sh"

CUDA_DIR="$ROOT/cpp/cuda/einsum"

printf '=== PyQCD einsum build (cuda) ===\n'

if compgen -G "$CUDA_DIR/src/*.cu" > /dev/null; then
    if ! command -v nvcc > /dev/null 2>&1; then
        printf 'error: nvcc not found but %s/src has .cu sources\n' "$CUDA_DIR" >&2
        exit 1
    fi
    PYQCD_EINSUM_BACKEND=cuda bash "$CUDA_DIR/make-nv.sh" "$@"
else
    printf 'note: no CUDA sources under %s/src; building the KUPL backend.\n' "$CUDA_DIR"
    PYQCD_EINSUM_BACKEND=kupl bash "$ROOT/cpp/kupl/einsum/make.sh" "$@"
fi

printf '=== PyQCD einsum build (cuda): SUCCESS ===\n'
