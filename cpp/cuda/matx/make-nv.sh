#!/usr/bin/env bash
# Build the MatX einsum C ABI bridge and publish it atomically.
#
# Mirrors cpp/cuda/einsum/make-nv.sh: nvcc direct build, version-script export
# surface, atomic symlink publish, and a fail-closed exit path. Without MatX
# headers the archive still builds, but every pyqcmatx_einsum() call returns
# PYQCMATX_STATUS_UNAVAILABLE, so the Python side can prove the ABI without
# pretending the kernel exists.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
BUILD_DIR=${PYQCD_MATX_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcmatx_cuda.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcmatx_cuda.so"
VERSION_SCRIPT="$BUILD_DIR/matx_exports.map"
BUILD_LOG="$BUILD_DIR/build.log"
NVCC=${NVCC:-nvcc}
NVCC_EXTRA=()
if [ -n "${PYQCD_MATX_NVCC_FLAGS:-}" ]; then
    read -r -a NVCC_EXTRA <<< "$PYQCD_MATX_NVCC_FLAGS"
fi
MATX_ROOT=${PYQCD_MATX_ROOT:-${MATX_ROOT:-}}
MODE=build
REQUIRE_MATX=${PYQCD_MATX_REQUIRE_MATX:-0}
ENABLE_CUTENSOR=${PYQCD_MATX_ENABLE_CUTENSOR:-0}
DEFAULT_ARCHITECTURES="6.0 7.0"

_pyqcd_notify() {
    { printf '[pyqcd build] %s\n' "$*" > /dev/tty; } 2>/dev/null \
        || printf '[pyqcd build] %s\n' "$*" >&2
}

usage() {
    printf 'Usage: bash %s [--clean] [--cutensor] [--require-matx]\n' "${BASH_SOURCE[0]:-$0}"
    printf '  --clean         remove the build directory and the published symlink\n'
    printf '  --cutensor      compile with -DMATX_EN_CUTENSOR=ON (needs cuTENSOR/cuTensorNet)\n'
    printf '  --require-matx  fail instead of publishing a MatX-less stub archive\n'
    printf 'Env: PYQCD_MATX_ROOT, PYQCD_MATX_LIB_DIR, PYQCD_MATX_INCLUDE_DIRS,\n'
    printf '     PYQCD_MATX_NVCC_FLAGS (e.g. -ccbin <g++-13>), PYQCD_MATX_BUILD_DIR\n'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --clean) MODE=clean; shift ;;
        --cutensor) ENABLE_CUTENSOR=1; shift ;;
        --require-matx) REQUIRE_MATX=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$MODE" = clean ]; then
    rm -rf "$BUILD_DIR"
    rm -f "$PUBLIC_LIBRARY"
    printf 'make-nv.sh: cleaned %s and %s\n' "$BUILD_DIR" "$PUBLIC_LIBRARY"
    exit 0
fi

if [ -n "$MATX_ROOT" ]; then
    MATX_HEADER="$MATX_ROOT/include/matx.h"
    if [ ! -f "$MATX_HEADER" ]; then
        printf 'error: MatX header not found: %s\n' "$MATX_HEADER" >&2
        printf '       set PYQCD_MATX_ROOT to a MatX checkout root\n' >&2
        exit 1
    fi
else
    MATX_HEADER=""
fi

detect_cuda_arch() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local capability
    capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n 1 | tr -d '[:space:]' || true)
    case "$capability" in
        [0-9]*.[0-9]*) printf '%s' "${capability%.*}${capability#*.}"; return 0 ;;
    esac
    return 1
}

ARCH_SOURCE=""
if [ -n "${PYQCD_CUDA_ARCHITECTURES:-}" ]; then
    ARCHITECTURES=$PYQCD_CUDA_ARCHITECTURES
    ARCH_SOURCE="PYQCD_CUDA_ARCHITECTURES"
elif [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
    ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES
    ARCH_SOURCE="CMAKE_CUDA_ARCHITECTURES"
elif [ -n "${PYQCD_SM_ARCH:-}" ]; then
    ARCHITECTURES=$PYQCD_SM_ARCH
    ARCH_SOURCE="PYQCD_SM_ARCH"
elif ARCHITECTURES=$(detect_cuda_arch); then
    ARCH_SOURCE="auto-detected"
else
    ARCHITECTURES=$DEFAULT_ARCHITECTURES
    ARCH_SOURCE="default (no GPU detected)"
fi
ARCHITECTURES=${ARCHITECTURES//,/ }
ARCHITECTURES=${ARCHITECTURES//;/ }

ARCH_FLAGS=()
ARCH_LABELS=()
for architecture in $ARCHITECTURES; do
    compact=${architecture//./}
    if ! [[ "$compact" =~ ^[0-9]+$ ]]; then
        printf 'error: invalid CUDA architecture %s\n' "$architecture" >&2
        exit 1
    fi
    ARCH_FLAGS+=("-gencode=arch=compute_${compact},code=[sm_${compact},compute_${compact}]")
    ARCH_LABELS+=("$architecture")
done

shopt -s nullglob
SOURCES=("$SCRIPT_DIR"/src/*.cu "$SCRIPT_DIR"/src/*.cpp)
shopt -u nullglob
if [ ${#SOURCES[@]} -eq 0 ]; then
    printf 'error: no sources found in %s/src\n' "$SCRIPT_DIR" >&2
    exit 1
fi

DEFINE_FLAGS=()
INCLUDE_FLAGS=("-I$SCRIPT_DIR/include")
LINK_FLAGS=(-lcudart)
# Conda-installed CUDA pieces live in two trees: $CONDA_PREFIX/include and
# $CONDA_PREFIX/targets/x86_64-linux/include (NVTX, cudart), with libraries in
# the matching lib directories. Both are added when present; explicit overrides
# win for non-conda toolchains.
if [ -n "${PYQCD_MATX_INCLUDE_DIRS:-}" ]; then
    read -r -a _extra_includes <<< "$PYQCD_MATX_INCLUDE_DIRS"
    INCLUDE_FLAGS+=("${_extra_includes[@]}")
fi
CONDA_INCLUDE_DIRS=()
if [ -n "${CONDA_PREFIX:-}" ]; then
    for directory in "$CONDA_PREFIX/include" "$CONDA_PREFIX/targets/x86_64-linux/include"; do
        [ -d "$directory" ] && CONDA_INCLUDE_DIRS+=("-I$directory")
    done
fi
INCLUDE_FLAGS+=("${CONDA_INCLUDE_DIRS[@]}")

LIB_DIRS=()
if [ -n "${PYQCD_MATX_LIB_DIR:-}" ]; then
    LIB_DIRS+=("-L$PYQCD_MATX_LIB_DIR")
fi
if [ -n "${CONDA_PREFIX:-}" ]; then
    for directory in "$CONDA_PREFIX/lib" "$CONDA_PREFIX/targets/x86_64-linux/lib"; do
        [ -d "$directory" ] && LIB_DIRS+=("-L$directory")
    done
fi

if [ -n "$MATX_HEADER" ]; then
    DEFINE_FLAGS+=("-DPYQCMATX_HAVE_MATX=1")
    INCLUDE_FLAGS+=("-I$MATX_ROOT/include")
    LINK_FLAGS+=(-lcublas -lcusolver)
else
    DEFINE_FLAGS+=("-DPYQCMATX_HAVE_MATX=0")
fi

if [ -n "${PYQCD_MATX_LINK_FLAGS:-}" ]; then
    read -r -a _extra_links <<< "$PYQCD_MATX_LINK_FLAGS"
    LINK_FLAGS+=("${_extra_links[@]}")
fi

have_library() {
    local name=$1
    if ldconfig -p 2>/dev/null | grep -q "lib${name}\.so"; then
        return 0
    fi
    for directory in "${LIB_DIRS[@]}"; do
        if ls "${directory#-L}"/lib"${name}".so >/dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

if [ "$ENABLE_CUTENSOR" = "1" ]; then
    if [ -z "$MATX_HEADER" ]; then
        printf 'error: --cutensor requires MatX headers (PYQCD_MATX_ROOT)\n' >&2
        exit 1
    fi
    missing=()
    for library in cutensor cutensornet; do
        if ! have_library "$library"; then
            missing+=("$library")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        printf 'error: missing cuTENSOR dependencies: %s\n' "${missing[*]}" >&2
        printf '       einsum needs cuTENSOR>=1.7.0 and cuTensorNet>=23.03.0.20\n' >&2
        exit 1
    fi
    DEFINE_FLAGS+=("-DPYQCMATX_ENABLE_CUTENSOR=1")
    # MatX gates the einsum implementation on MATX_EN_CUTENSOR itself, so the
    # bridge macro alone is not enough: without this the archive compiles but
    # every einsum_impl() call throws matxNotSupported.
    DEFINE_FLAGS+=("-DMATX_EN_CUTENSOR=1")
    LINK_FLAGS+=("${LIB_DIRS[@]}" -lcutensor -lcutensornet)
else
    DEFINE_FLAGS+=("-DPYQCMATX_ENABLE_CUTENSOR=0")
fi

mkdir -p "$BUILD_DIR"
cat > "$VERSION_SCRIPT" <<'MAP'
{
  global:
    pyqcmatx_abi_version;
    pyqcmatx_cutensor_enabled;
    pyqcmatx_einsum;
    pyqcmatx_last_error;
  local:
    *;
};
MAP

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

if [ -n "$MATX_HEADER" ]; then
    _pyqcd_notify "MatX: headers at $MATX_HEADER; cutensor=${ENABLE_CUTENSOR}"
else
    _pyqcd_notify "MatX: headers not found; building the UNAVAILABLE stub"
fi
printf 'pyqcd matx einsum bridge build\n'
printf '  nvcc:     %s\n' "$($NVCC --version | tail -n 1)"
printf '  arch:     %s (source: %s; cubin+PTX embedded)\n' \
    "${ARCH_LABELS[*]}" "$ARCH_SOURCE"
printf '  matx:     %s\n' "${MATX_HEADER:-not found (stub)}"
printf '  cutensor: %s\n' "$ENABLE_CUTENSOR"
printf '  sources:  %s\n' "$(for source in "${SOURCES[@]}"; do printf '%s ' "${source##*/}"; done)"
printf '  output:   %s\n' "$PUBLIC_LIBRARY"

set +e
"$NVCC" -O3 -std=c++20 --shared \
    "${NVCC_EXTRA[@]}" \
    -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
    "${ARCH_FLAGS[@]}" \
    "${DEFINE_FLAGS[@]}" \
    "${INCLUDE_FLAGS[@]}" \
    "${SOURCES[@]}" \
    "${LINK_FLAGS[@]}" \
    -Xlinker --version-script="$VERSION_SCRIPT" \
    -Xlinker -soname -Xlinker libpyqcmatx_cuda.so \
    -o "$TEMP_LIBRARY" 2>&1 | tee "$BUILD_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ]; then
    printf 'error: nvcc failed with exit status %s (see %s)\n' "$STATUS" "$BUILD_LOG" >&2
    exit "$STATUS"
fi
if [ ! -f "$TEMP_LIBRARY" ]; then
    printf 'error: nvcc reported success but %s is missing\n' "$TEMP_LIBRARY" >&2
    exit 1
fi
if ! nm -D --defined-only "$TEMP_LIBRARY" | awk '{print $NF}' | grep -qx pyqcmatx_einsum; then
    printf 'error: built library does not export pyqcmatx_einsum\n' >&2
    exit 1
fi
if [ -z "$MATX_HEADER" ]; then
    printf 'warning: MatX headers absent; pyqcmatx_einsum() returns UNAVAILABLE(%s)\n' \
        "4" >&2
    if [ "$REQUIRE_MATX" = "1" ]; then
        printf 'error: --require-matx forbids a MatX-less archive\n' >&2
        exit 1
    fi
fi

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqcmatx_cuda.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
printf 'make-nv.sh: SUCCESS\n'
