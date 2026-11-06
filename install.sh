#!/usr/bin/env bash
# Build the selected native backend and its in-place Python entry point.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
MODE=${1:-auto}

usage() {
    cat <<'EOF'
Usage: bash install.sh [kupl|cuda|auto]

  kupl  Build cpp/kupl/einsum; setup.py does not import Cython or CUDA.
  cuda  Build cpp/cuda/einsum and pyqcd.einsum.cuda.einsum.
  auto  Use CUDA when nvcc and CUDA sources are present, otherwise KUPL.

Environment:
  PYQCD_EINSUM_BUILD_DIR         Override the selected native build directory.
  PYQCD_OPENBLAS_LIBRARY         Explicit OpenBLAS shared-library path.
  PYQCD_OPENBLAS_INCLUDE_DIR     Header directory paired with the override.
  PYQCD_OPENBLAS_FLAGS           Complete compiler/link flags for OpenBLAS.
  PYQCD_SKIP_NATIVE_BUILD=1      Reuse an existing native library.
  PYQCD_EINSUM_SKIP_CYTHON=1     Explicitly skip extension compilation.
EOF
}

case "$MODE" in
    kupl|cuda) ;;
    auto)
        if command -v "${NVCC:-nvcc}" >/dev/null 2>&1 && \
                compgen -G "$ROOT/cpp/cuda/einsum/src/*.cu" >/dev/null; then
            MODE=cuda
        else
            MODE=kupl
        fi
        ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [ "${PYQCD_SKIP_NATIVE_BUILD:-0}" != "1" ]; then
    if [ "$MODE" = cuda ]; then
        bash "$ROOT/build-nv.sh"
    else
        bash "$ROOT/build-kupl.sh"
    fi
fi

export PYQCD_EINSUM_BACKEND="$MODE"
# shellcheck source=/dev/null
source "$ROOT/env.sh"

PYQCD_EINSUM_BACKEND="$MODE" \
PYQCD_SKIP_NATIVE_BUILD=1 \
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYQCD_EINSUM_PYTHON" "$ROOT/setup.py" build_ext --inplace

if [ "$MODE" = cuda ]; then
    LIBRARY=$PYQCD_EINSUM_CUDA_LIB
else
    LIBRARY=$PYQCD_EINSUM_KUPL_LIB
fi
if [ ! -e "$LIBRARY" ]; then
    printf 'error: installation did not produce %s\n' "$LIBRARY" >&2
    exit 1
fi

printf 'PyQCD einsum installation complete.\n'
printf '  backend: %s\n' "$MODE"
printf '  native library: %s\n' "$LIBRARY"
printf '  export PYQCD_EINSUM_LIB=%q\n' "$LIBRARY"
