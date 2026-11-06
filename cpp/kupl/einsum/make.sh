#!/usr/bin/env bash
# Build the KUPL C ABI used by the pure-ctypes Python binding.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcdeinsum_kupl.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcdeinsum_kupl.so"
VERSION_SCRIPT="$BUILD_DIR/einsum_kupl_exports.map"
BUILD_LOG="$BUILD_DIR/build.log"
MPI_SETTING=${PYQCD_EINSUM_ENABLE_MPI:-auto}
case "$MPI_SETTING" in
    1|on|ON|true|TRUE)
        MPI_ENABLED=1
        ;;
    0|off|OFF|false|FALSE)
        MPI_ENABLED=0
        ;;
    auto|AUTO)
        if command -v mpicxx >/dev/null 2>&1; then
            MPI_ENABLED=1
        else
            MPI_ENABLED=0
        fi
        ;;
    *)
        printf 'error: invalid PYQCD_EINSUM_ENABLE_MPI=%s\n' "$MPI_SETTING" >&2
        exit 2
        ;;
esac
if [ "$MPI_ENABLED" = 1 ]; then
    DEFAULT_CXX=mpicxx
else
    DEFAULT_CXX=g++
fi
CXX=${CXX:-$DEFAULT_CXX}
NATIVE_ARCH=${PYQCD_EINSUM_NATIVE_ARCH:-1}
DEFAULT_FLAGS="-O3 -fopenmp -shared -fPIC -std=c++17 -fvisibility=hidden -Wall -Wextra"
if [ "$NATIVE_ARCH" = "1" ]; then
    DEFAULT_FLAGS="$DEFAULT_FLAGS -march=native -funroll-loops"
fi
FLAGS=${PYQCD_EINSUM_FLAGS:-"$DEFAULT_FLAGS"}
if [ "$MPI_ENABLED" = 1 ]; then
    FLAGS="$FLAGS -DPYQCD_EINSUM_ENABLE_MPI=1"
fi
MODE=build

usage() {
    printf 'Usage: bash %s [--clean]\n' "${BASH_SOURCE[0]:-$0}"
    printf '  --clean   remove the selected build directory and published symlink\n'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --clean) MODE=clean; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$MODE" = clean ]; then
    rm -rf "$BUILD_DIR"
    rm -f "$PUBLIC_LIBRARY"
    printf 'make.sh: cleaned %s and %s\n' "$BUILD_DIR" "$PUBLIC_LIBRARY"
    exit 0
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
read -r -a CXXFLAGS <<< "$FLAGS"

CORE_SOURCES=(
    "$SCRIPT_DIR/src/runtime.cpp"
    "$SCRIPT_DIR/src/plan_core.cpp"
)
for source in "${CORE_SOURCES[@]}"; do
    if [ ! -f "$source" ]; then
        printf 'error: shared source not found: %s\n' "$source" >&2
        exit 1
    fi
done

shopt -s nullglob
BACKEND_SOURCES=()
for source in "$SCRIPT_DIR"/src/*.cpp; do
    case "$(basename "$source")" in
        runtime.cpp|plan_core.cpp) ;;
        mpi_runtime.cpp)
            if [ "$MPI_ENABLED" = 1 ]; then
                BACKEND_SOURCES+=("$source")
            fi
            ;;
        *) BACKEND_SOURCES+=("$source") ;;
    esac
done
shopt -u nullglob
if [ ${#BACKEND_SOURCES[@]} -eq 0 ]; then
    printf 'error: no backend sources found in %s/src\n' "$SCRIPT_DIR" >&2
    exit 1
fi

SOURCES=("${CORE_SOURCES[@]}" "${BACKEND_SOURCES[@]}")

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

printf 'pyqcd einsum KUPL build\n'
printf '  compiler: %s\n' "$("$CXX" --version | head -n 1)"
printf '  flags:    %s\n' "$FLAGS"
printf '  MPI:      %s\n' "$([ "$MPI_ENABLED" = 1 ] && printf enabled || printf disabled)"
printf '  OpenBLAS: %s\n' "$_OPENBLAS_SUMMARY"
printf '  sources:  %s\n' "$(for source in "${SOURCES[@]}"; do printf '%s ' "${source##*/}"; done)"
printf '  output:   %s\n' "$PUBLIC_LIBRARY"

set +e
"$CXX" "${CXXFLAGS[@]}" \
    -I"$SCRIPT_DIR/include" \
    "${SOURCES[@]}" \
    "${_OPENBLAS_FLAGS[@]}" \
    -Wl,--version-script="$VERSION_SCRIPT" \
    -Wl,-soname,libpyqcdeinsum_kupl.so \
    -o "$TEMP_LIBRARY" 2>&1 | tee "$BUILD_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ]; then
    printf 'error: compilation failed with exit status %s (see %s)\n' \
        "$STATUS" "$BUILD_LOG" >&2
    exit "$STATUS"
fi
if [ ! -f "$TEMP_LIBRARY" ]; then
    printf 'error: compiler reported success but %s is missing\n' "$TEMP_LIBRARY" >&2
    exit 1
fi
if ! nm -D --defined-only "$TEMP_LIBRARY" | awk '{print $NF}' | grep -qx applyPlanQcd; then
    printf 'error: built library does not export applyPlanQcd\n' >&2
    exit 1
fi
if [ "$MPI_ENABLED" = 1 ] &&
    ! nm -D --defined-only "$TEMP_LIBRARY" | awk '{print $NF}' | grep -qx applyContractMpiQcd; then
    printf 'error: MPI build does not export applyContractMpiQcd\n' >&2
    exit 1
fi

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqcdeinsum_kupl.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
if ! nm -D --defined-only "$BUILT_LIBRARY" | awk '{print $NF}' | grep -qx applyContractQcd; then
    printf 'warning: applyContractQcd is not implemented by this KUPL source set yet\n' >&2
fi
printf 'make.sh: SUCCESS\n'
