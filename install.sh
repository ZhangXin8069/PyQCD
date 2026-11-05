#!/usr/bin/env bash
# Build and install the PyQCD einsum stack in place.
#
# Mirrors PyQCU/install.sh:
#   kupl  build cpp/kupl/einsum, then the Cython bridge (default)
#   cuda  build cpp/cuda/einsum when it has .cu sources (otherwise KUPL), then
#         the same Cython bridge
#   -h    usage
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
MODE=${1:-kupl}

usage() {
    cat <<'EOF'
Usage: bash install.sh [kupl|cuda]

  kupl  Build the KUPL backend (cpp/kupl/einsum) and the Cython extension (default).
  cuda  Build the CUDA backend (cpp/cuda/einsum) and the Cython extension.

Environment:
  PYQCD_EINSUM_BUILD_DIR     Native build directory (default: cpp/kupl/einsum/build)
  PYQCD_EINSUM_FLAGS         Compiler flags for the C++ library
  PYQCD_SKIP_NATIVE_BUILD=1  Reuse an existing libpyqcdeinsum.so
  PYQCD_EINSUM_SKIP_CYTHON=1 Build the library only
EOF
}

case "$MODE" in
    kupl)
        bash "$ROOT/build-kupl.sh"
        LIBRARY="$ROOT/cpp/kupl/einsum/libpyqcdeinsum.so"
        ;;
    cuda)
        bash "$ROOT/build-nv.sh"
        LIBRARY="$ROOT/cpp/cuda/einsum/libpyqcdeinsum.so"
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

PYQCD_EINSUM_BACKEND="$MODE" \
PYQCD_SKIP_NATIVE_BUILD=1 \
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$ROOT/setup.py" build_ext --inplace

rm -rf "$ROOT/build"

printf 'PyQCD einsum installation complete.\n'
printf '  backend: %s\n' "$MODE"
printf '  native library: %s\n' "$LIBRARY"
printf '  export PYQCD_EINSUM_LIB=%q\n' "$LIBRARY"
