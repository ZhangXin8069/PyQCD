#!/usr/bin/env bash
# Build the CUDA C ABI and publish it atomically.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
SHARED_DIR=$(cd "$SCRIPT_DIR/../../kupl/einsum" && pwd)
BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcdeinsum_cuda.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcdeinsum_cuda.so"
VERSION_SCRIPT="$BUILD_DIR/einsum_cuda_exports.map"
BUILD_LOG="$BUILD_DIR/build.log"
NVCC=${NVCC:-nvcc}
SHARED_CORE_SOURCES=(
    "$SHARED_DIR/src/runtime.cpp"
    "$SHARED_DIR/src/plan_core.cpp"
)
MODE=build
REQUIRE_KERNELS=${PYQCD_EINSUM_REQUIRE_KERNELS:-0}
PROVIDER_SYMBOLS=(qcd_cuda_permute qcd_cuda_hadamard qcd_cuda_batched_gemm)
DEFAULT_ARCHITECTURES="6.0 7.0"

usage() {
    printf 'Usage: bash %s [--clean] [--require-kernels]\n' "${BASH_SOURCE[0]:-$0}"
    printf '  --clean            remove the selected build directory and published symlink\n'
    printf '  --require-kernels  fail instead of publishing a planning-only archive\n'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --clean) MODE=clean; shift ;;
        --require-kernels) REQUIRE_KERNELS=1; shift ;;
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

if ! cmp -s "$SCRIPT_DIR/include/einsum_define.h" "$SHARED_DIR/include/einsum_define.h"; then
    printf 'error: CUDA/KUPL ABI headers differ:\n' >&2
    printf '       %s/include/einsum_define.h\n' "$SCRIPT_DIR" >&2
    printf '       %s/include/einsum_define.h\n' "$SHARED_DIR" >&2
    exit 1
fi

_discover_openblas() {
    _OPENBLAS_FLAGS=()
    _OPENBLAS_SUMMARY="not found"

    if [ -n "${PYQCD_OPENBLAS_FLAGS:-}" ]; then
        read -r -a _OPENBLAS_FLAGS <<< "$PYQCD_OPENBLAS_FLAGS"
        _OPENBLAS_SUMMARY="PYQCD_OPENBLAS_FLAGS"
    elif [ -n "${PYQCD_OPENBLAS_LIBRARY:-}" ]; then
        if [ ! -e "$PYQCD_OPENBLAS_LIBRARY" ]; then
            printf 'error: PYQCD_OPENBLAS_LIBRARY does not exist: %s\n' \
                "$PYQCD_OPENBLAS_LIBRARY" >&2
            exit 1
        fi
        if [ -n "${PYQCD_OPENBLAS_INCLUDE_DIR:-}" ]; then
            _OPENBLAS_FLAGS+=("-I$PYQCD_OPENBLAS_INCLUDE_DIR")
        fi
        _OPENBLAS_FLAGS+=("$PYQCD_OPENBLAS_LIBRARY")
        _OPENBLAS_SUMMARY="$PYQCD_OPENBLAS_LIBRARY"
    elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas; then
        read -r -a _OPENBLAS_FLAGS <<< "$(pkg-config --cflags --libs openblas)"
        _OPENBLAS_SUMMARY="pkg-config openblas"
    else
        local header=""
        local directory=""
        local candidate=""
        for candidate in \
            "${OPENBLAS_ROOT:-}/include/cblas.h" \
            "${OPENBLAS_HOME:-}/include/cblas.h" \
            /usr/local/include/cblas.h \
            /usr/include/cblas.h \
            /usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h \
            /usr/include/x86_64-linux-gnu/openblas-openmp/cblas.h \
            /usr/include/openblas/cblas.h \
            /opt/OpenBLAS/include/cblas.h; do
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then
                header=$candidate
                directory=$(dirname "$candidate")
                break
            fi
        done
        if [ -n "$header" ]; then
            _OPENBLAS_FLAGS+=("-I$directory")
            local library=""
            if command -v ldconfig >/dev/null 2>&1; then
                library=$(ldconfig -p 2>/dev/null | awk '$1 == "libopenblas.so" {print $NF; exit}')
            fi
            if [ -n "$library" ] && [ -e "$library" ]; then
                _OPENBLAS_FLAGS+=("$library")
                _OPENBLAS_SUMMARY="$header; $library"
            else
                _OPENBLAS_FLAGS+=("-lopenblas")
                _OPENBLAS_SUMMARY="$header; -lopenblas"
            fi
        fi
    fi
}

_discover_openblas

for source in "${SHARED_CORE_SOURCES[@]}"; do
    if [ ! -f "$source" ]; then
        printf 'error: shared source not found: %s\n' "$source" >&2
        exit 1
    fi
done

shopt -s nullglob
LOCAL_SOURCES=("$SCRIPT_DIR"/src/*.cu "$SCRIPT_DIR"/src/*.cpp)
shopt -u nullglob
if [ ${#LOCAL_SOURCES[@]} -eq 0 ]; then
    printf 'error: no CUDA backend sources found in %s/src\n' "$SCRIPT_DIR" >&2
    exit 1
fi
SOURCES=("${SHARED_CORE_SOURCES[@]}" "${LOCAL_SOURCES[@]}")

if [ -n "${PYQCD_CUDA_ARCHITECTURES:-}" ]; then
    ARCHITECTURES=$PYQCD_CUDA_ARCHITECTURES
elif [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
    ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES
elif [ -n "${PYQCD_SM_ARCH:-}" ]; then
    ARCHITECTURES=$PYQCD_SM_ARCH
else
    ARCHITECTURES=$DEFAULT_ARCHITECTURES
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
    ARCH_FLAGS+=("-gencode=arch=compute_${compact},code=sm_${compact}")
    ARCH_LABELS+=("$architecture")
done

mkdir -p "$BUILD_DIR"
cat > "$VERSION_SCRIPT" <<'MAP'
{
  global:
    applyPlanQcd;
    applyContractQcd;
    applyContractMpiQcd;
    applyCompileQcd;
    applyCompiledQcd;
    applyDestroyQcd;
    applyInitQcd;
    applyEndQcd;
  local:
    *;
};
MAP

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

printf 'pyqcd einsum CUDA build\n'
printf '  nvcc:     %s\n' "$($NVCC --version | tail -n 1)"
printf '  arch:     %s\n' "${ARCH_LABELS[*]}"
printf '  OpenBLAS: %s\n' "$_OPENBLAS_SUMMARY"
printf '  sources:  %s\n' "$(for source in "${SOURCES[@]}"; do printf '%s ' "${source##*/}"; done)"
printf '  output:   %s\n' "$PUBLIC_LIBRARY"

set +e
"$NVCC" -O3 -std=c++17 --shared \
    -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
    "${ARCH_FLAGS[@]}" \
    -I"$SHARED_DIR/include" \
    -I"$SCRIPT_DIR/include" \
    "${SOURCES[@]}" \
    "${_OPENBLAS_FLAGS[@]}" \
    -lcublas -lcudart \
    -Xlinker --version-script="$VERSION_SCRIPT" \
    -Xlinker -soname -Xlinker libpyqcdeinsum_cuda.so \
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
if ! nm -D --defined-only "$TEMP_LIBRARY" | awk '{print $NF}' | grep -qx applyPlanQcd; then
    printf 'error: built library does not export applyPlanQcd\n' >&2
    exit 1
fi

MISSING=()
for symbol in "${PROVIDER_SYMBOLS[@]}"; do
    if ! nm --defined-only "$TEMP_LIBRARY" 2>/dev/null | awk '{print $NF}' | grep -qx "$symbol"; then
        MISSING+=("$symbol")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    printf 'warning: missing CUDA providers: %s\n' "${MISSING[*]}" >&2
    if [ "$REQUIRE_KERNELS" = "1" ]; then
        printf 'error: --require-kernels forbids a planning-only archive\n' >&2
        exit 1
    fi
fi

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqcdeinsum_cuda.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
printf 'make-nv.sh: SUCCESS\n'
