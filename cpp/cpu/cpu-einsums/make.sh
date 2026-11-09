#!/usr/bin/env bash
# Build the Einsums string-`einsum` C ABI bridge (CPU) and publish it atomically.
#
# Mirrors cpp/kupl/einsum/make.sh (compiler flags, version-script export surface,
# temporary-file + `mv -Tf` publish, fail-closed exit path). Without Einsums
# headers the archive STILL builds, but every pyqceinsums_einsum() call returns
# PYQCEINSUMS_STATUS_UNAVAILABLE, so the Python side can prove the ABI without
# pretending the Einsums kernel exists. `--require-einsums` turns that same
# state into a hard error.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
BUILD_DIR=${PYQCD_EINSUMS_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqceinsums.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqceinsums.so"
VERSION_SCRIPT="$BUILD_DIR/cpu_einsums_exports.map"
BUILD_LOG="$BUILD_DIR/build.log"

CXX=${CXX:-g++}
NATIVE_ARCH=${PYQCD_EINSUMS_NATIVE_ARCH:-1}
EINSUMS_ROOT=${PYQCD_EINSUMS_ROOT:-${EINSUMS_ROOT:-}}
EINSUMS_LIBS=${PYQCD_EINSUMS_LIBS:-}
MODE=build
REQUIRE_EINSUMS=0

# Progress goes straight to the terminal: pip captures child output and only
# replays it on failure, so a silent build looks like a hang.
_pyqcd_notify() {
    { printf '[pyqcd build] %s\n' "$*" > /dev/tty; } 2>/dev/null \
        || printf '[pyqcd build] %s\n' "$*" >&2
}

usage() {
    printf 'Usage: bash %s [--clean] [--require-einsums]\n' "${BASH_SOURCE[0]:-$0}"
    printf '  --clean           remove the build directory and the published symlink\n'
    printf '  --require-einsums fail instead of publishing an Einsums-less stub archive\n'
    printf 'Environment:\n'
    printf '  PYQCD_EINSUMS_ROOT / EINSUMS_ROOT    Einsums checkout or install prefix\n'
    printf '  PYQCD_EINSUMS_LIBS                   extra link flags (default: empty)\n'
    printf '  PYQCD_EINSUMS_FLAGS                  full override of the compiler flags\n'
    printf '  PYQCD_EINSUMS_BUILD_DIR              build directory (default: <dir>/build)\n'
    printf '  PYQCD_EINSUMS_NATIVE_ARCH           1 = add -march=native -funroll-loops (default)\n'
    printf '  CXX                                  compiler (default: g++)\n'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --clean) MODE=clean; shift ;;
        --require-einsums) REQUIRE_EINSUMS=1; shift ;;
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

DEFAULT_FLAGS="-O3 -fopenmp -shared -fPIC -std=c++20 -fvisibility=hidden -Wall -Wextra"
if [ "$NATIVE_ARCH" = "1" ]; then
    DEFAULT_FLAGS="$DEFAULT_FLAGS -march=native -funroll-loops"
fi

# ---------------------------------------------------------------------------
# Einsums header probe: `$EINSUMS_ROOT/include/Einsums/*.hpp` or
# `$EINSUMS_ROOT/include/einsums/*.hpp`, first glob hit wins.
# ---------------------------------------------------------------------------
EINSUMS_INCLUDE=""
EINSUMS_HEADER_DIR=""
if [ -n "$EINSUMS_ROOT" ]; then
    for candidate in "$EINSUMS_ROOT/include/Einsums" "$EINSUMS_ROOT/include/einsums"; do
        if [ -d "$candidate" ]; then
            shopt -s nullglob
            headers=("$candidate"/*.hpp)
            shopt -u nullglob
            if [ ${#headers[@]} -gt 0 ]; then
                EINSUMS_HEADER_DIR=$candidate
                EINSUMS_INCLUDE="$EINSUMS_ROOT/include"
                break
            fi
        fi
    done
fi

if [ -z "$EINSUMS_INCLUDE" ]; then
    printf 'warning: no Einsums headers found' >&2
    if [ -n "$EINSUMS_ROOT" ]; then
        printf ' under %s/include/{Einsums,einsums}/*.hpp\n' "$EINSUMS_ROOT" >&2
    else
        printf ': PYQCD_EINSUMS_ROOT (or EINSUMS_ROOT) is unset\n' >&2
    fi
    printf 'warning: building the UNAVAILABLE stub; pyqceinsums_einsum() will return 4\n' >&2
    printf 'warning: set PYQCD_EINSUMS_ROOT to an Einsums checkout/prefix to enable the real path\n' >&2
    if [ "$REQUIRE_EINSUMS" = "1" ]; then
        printf 'error: --require-einsums forbids an Einsums-less archive\n' >&2
        exit 1
    fi
fi

if [ -z "${PYQCD_EINSUMS_FLAGS:-}" ]; then
    FLAGS="$DEFAULT_FLAGS"
else
    FLAGS=$PYQCD_EINSUMS_FLAGS
fi
read -r -a CXXFLAGS <<< "$FLAGS"

DEFINE_FLAGS=()
INCLUDE_FLAGS=("-I$SCRIPT_DIR/include")
LINK_FLAGS=()
if [ -n "$EINSUMS_INCLUDE" ]; then
    DEFINE_FLAGS+=("-DPYQCEINSUMS_HAVE_EINSUMS=1")
    INCLUDE_FLAGS+=("-I$EINSUMS_INCLUDE")
    # The probe accepts both directory spellings; the include spelling has to
    # follow suit, otherwise a lowercase tree only fails deep inside the compile.
    case "$EINSUMS_HEADER_DIR" in
        */include/einsums)
            DEFINE_FLAGS+=("-DPYQCEINSUMS_TENSOR_HEADER=<einsums/Tensor.hpp>")
            DEFINE_FLAGS+=("-DPYQCEINSUMS_COMPUTE_GRAPH_HEADER=<einsums/ComputeGraph/ComputeGraph.hpp>")
            ;;
    esac
    if [ -n "$EINSUMS_LIBS" ]; then
        read -r -a LINK_FLAGS <<< "$EINSUMS_LIBS"
    else
        printf 'warning: PYQCD_EINSUMS_LIBS is empty; the archive will keep undefined Einsums symbols\n' >&2
        printf 'warning: set e.g. PYQCD_EINSUMS_LIBS="-L\$PYQCD_EINSUMS_ROOT/lib -leinsums"\n' >&2
    fi
else
    DEFINE_FLAGS+=("-DPYQCEINSUMS_HAVE_EINSUMS=0")
fi

shopt -s nullglob
SOURCES=("$SCRIPT_DIR"/src/*.cpp)
shopt -u nullglob
if [ ${#SOURCES[@]} -eq 0 ]; then
    printf 'error: no sources found in %s/src\n' "$SCRIPT_DIR" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
cat > "$VERSION_SCRIPT" <<'MAP'
{
  global:
    pyqceinsums_abi_version;
    pyqceinsums_backend_enabled;
    pyqceinsums_einsum;
    pyqceinsums_last_error;
  local:
    *;
};
MAP

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

_pyqcd_notify "Einsums (CPU): compiling ${#SOURCES[@]} source(s)"
printf 'pyqcd CPU einsums bridge build\n'
printf '  compiler: %s\n' "$("$CXX" --version | head -n 1)"
printf '  flags:    %s\n' "$FLAGS"
printf '  einsums:  %s\n' "${EINSUMS_HEADER_DIR:-not found (UNAVAILABLE stub)}"
printf '  libs:     %s\n' "${EINSUMS_LIBS:-none}"
printf '  sources:  %s\n' "$(for source in "${SOURCES[@]}"; do printf '%s ' "${source##*/}"; done)"
printf '  output:   %s\n' "$PUBLIC_LIBRARY"

set +e
"$CXX" "${CXXFLAGS[@]}" \
    "${DEFINE_FLAGS[@]}" \
    "${INCLUDE_FLAGS[@]}" \
    "${SOURCES[@]}" \
    "${LINK_FLAGS[@]}" \
    -Wl,--version-script="$VERSION_SCRIPT" \
    -Wl,-soname,libpyqceinsums.so \
    -o "$TEMP_LIBRARY" 2>&1 | tee "$BUILD_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ]; then
    printf 'error: compilation failed with exit status %s (see %s)\n' \
        "$STATUS" "$BUILD_LOG" >&2
    cat "$BUILD_LOG" >&2
    exit "$STATUS"
fi
if [ ! -f "$TEMP_LIBRARY" ]; then
    printf 'error: compiler reported success but %s is missing\n' "$TEMP_LIBRARY" >&2
    exit 1
fi
if ! nm -D --defined-only "$TEMP_LIBRARY" | awk '{print $NF}' | grep -qx pyqceinsums_einsum; then
    printf 'error: built library does not export pyqceinsums_einsum\n' >&2
    exit 1
fi

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqceinsums.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
if [ -z "$EINSUMS_INCLUDE" ]; then
    printf 'warning: Einsums headers absent; pyqceinsums_einsum() returns UNAVAILABLE(4)\n' >&2
fi
_pyqcd_notify "Einsums (CPU): done"
printf 'make.sh: SUCCESS\n'
