#!/usr/bin/env bash
# Build the native einsum library: cpp/einsum/libpyqcdeinsum.so
#
# The archive is linked into build/ first and published through a stable
# relative symlink that is swapped with mv -Tf, so a running Python process
# never observes a half-written .so (same contract as PyQCU cpp/kupl/qcu).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcdeinsum.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcdeinsum.so"
CXX=${CXX:-g++}
FLAGS=${PYQCD_EINSUM_FLAGS:-"-O3 -fopenmp -shared -fPIC -std=c++17 -fvisibility=hidden -Wall -Wextra"}
VERSION_SCRIPT="$BUILD_DIR/einsum_exports.map"
BUILD_LOG="$BUILD_DIR/build.log"
MODE=build

usage() {
    printf 'Usage: bash %s [--clean]\n' "${BASH_SOURCE[0]:-$0}"
    printf '  --clean   remove the build directory and the published library\n'
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
    printf 'build.sh: cleaned %s and %s\n' "$BUILD_DIR" "$PUBLIC_LIBRARY"
    exit 0
fi

read -r -a CXXFLAGS <<< "$FLAGS"

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
    applyPlanQcd;
    applyContractQcd;
    applyInitQcd;
    applyEndQcd;
    apply*Qcd;
  local:
    *;
};
MAP

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

printf 'pyqcd einsum native build\n'
printf '  compiler: %s\n' "$("$CXX" --version | head -n 1)"
printf '  flags:    %s\n' "$FLAGS"
printf '  sources:  %s\n' "${SOURCES[*]##*/}"
printf '  output:   %s\n' "$PUBLIC_LIBRARY"

set +e
"$CXX" "${CXXFLAGS[@]}" -I"$SCRIPT_DIR/include" "${SOURCES[@]}" \
    -o "$TEMP_LIBRARY" \
    -Wl,--version-script="$VERSION_SCRIPT" \
    -Wl,-soname,libpyqcdeinsum.so 2>&1 | tee "$BUILD_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [ "$STATUS" -ne 0 ]; then
    printf 'error: compilation failed with exit status %s (see %s)\n' "$STATUS" "$BUILD_LOG" >&2
    exit "$STATUS"
fi
if [ ! -f "$TEMP_LIBRARY" ]; then
    printf 'error: compiler reported success but %s is missing\n' "$TEMP_LIBRARY" >&2
    exit 1
fi

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqcdeinsum.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
printf 'build.sh: SUCCESS\n'
