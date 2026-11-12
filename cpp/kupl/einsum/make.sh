#!/usr/bin/env bash
# Build the KUPL C ABI used by the pure-ctypes Python binding.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcdeinsum_kupl.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcdeinsum_kupl.so"
VERSION_SCRIPT="$BUILD_DIR/einsum_kupl_exports.map"

# 构建进度：pip 会捕获子进程输出、只在失败时回放，所以进度直接写终端（/dev/tty）；
# 没有终端时退回 stderr。这样 `pip install pyqcd` 期间用户能看见"在编什么、编到哪"，
# 不会误判为卡死。
_pyqcd_notify() {
    { printf '[pyqcd build] %s\n' "$*" > /dev/tty; } 2>/dev/null \
        || printf '[pyqcd build] %s\n' "$*" >&2
}
_pyqcd_now() { date +%s; }

# 依赖缺失的统一报错：症状 + missing/needed/fix/bypass/doc。stderr，退出码 3
# （3 = 预检缺依赖、2 = 配置错误、1 = 构建失败；与 setup.py 的约定一致）。
_pyqcd_fail() {
    local dep=$1 why=$2 fix=$3 bypass=${4:-}
    {
        printf 'error: %s is required to build the KUPL native backend.\n' "$dep"
        printf '  missing: %s\n' "$dep"
        printf '  needed:  %s\n' "$why"
        printf '  fix:     %s\n' "$fix"
        if [ -n "$bypass" ]; then
            printf '  bypass:  %s\n' "$bypass"
        fi
        printf '  doc:     https://py-qcd.readthedocs.io/en/latest/install.html\n'
    } >&2
    exit 3
}

# 编译失败时按日志里的已知特征给一条可操作的 note，而不是只报"exited N"。
_pyqcd_explain_build_log() {
    local log=$1
    [ -f "$log" ] || return 0
    if grep -q 'omp\.h: No such file' "$log" 2>/dev/null; then
        printf 'note: the OpenMP header omp.h is missing. Install a GCC toolchain with\n' >&2
        printf '      OpenMP (apt install build-essential, conda install gxx_linux-64) or\n' >&2
        printf '      drop -fopenmp via PYQCD_EINSUM_FLAGS.\n' >&2
    elif grep -qE 'cannot find -l(openblas|cblas)|openblas.*No such file' "$log" 2>/dev/null; then
        printf 'note: OpenBLAS/CBLAS was not found by the linker. Install it\n' >&2
        printf '      (apt install libopenblas-dev, conda install openblas) or point at it\n' >&2
        printf '      with PYQCD_OPENBLAS_LIBRARY / PYQCD_OPENBLAS_FLAGS.\n' >&2
    fi
}
BUILD_LOG="$BUILD_DIR/build.log"
NATIVE_ARCH=${PYQCD_EINSUM_NATIVE_ARCH:-1}
DEFAULT_FLAGS="-O3 -fopenmp -shared -fPIC -std=c++17 -fvisibility=hidden -Wall -Wextra"
if [ "$NATIVE_ARCH" = "1" ]; then
    DEFAULT_FLAGS="$DEFAULT_FLAGS -march=native -funroll-loops"
fi
BASE_FLAGS=${PYQCD_EINSUM_FLAGS:-"$DEFAULT_FLAGS"}
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

CXX=${CXX:-g++}
CXX=${CXX:-$DEFAULT_CXX}
if ! command -v "$CXX" >/dev/null 2>&1; then
    _pyqcd_fail "C++ compiler (${CXX})" \
        "the KUPL backend is C++17 source and cannot be built without it" \
        "apt install build-essential | dnf install gcc-c++ | conda install -c conda-forge gxx_linux-64" \
        "PYQCD_SKIP_NATIVE_BUILD=1 pip install pyqcd (pure-Python planner)"
fi
if ! command -v nm >/dev/null 2>&1; then
    _pyqcd_fail "binutils (nm)" \
        "the build verifies the exported C ABI symbols with nm" \
        "apt install binutils | dnf install binutils | conda install -c conda-forge binutils" \
        "PYQCD_SKIP_NATIVE_BUILD=1 pip install pyqcd (pure-Python planner)"
fi
if [ "$NATIVE_ARCH" = "1" ]; then
    _pyqcd_notify "KUPL (CPU): note: -march=native bakes this host's ISA; set PYQCD_EINSUM_NATIVE_ARCH=0 for portable binaries"
fi
FLAGS=$BASE_FLAGS

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
if [ "$_OPENBLAS_SUMMARY" = "not found" ]; then
    _pyqcd_notify "KUPL (CPU): warning: OpenBLAS not found at build time; the runtime loads it via dlopen (libcblas/libopenblas). Install libopenblas0 (apt) / openblas (conda), or set PYQCD_OPENBLAS_LIBRARY if the runtime cannot find it."
fi
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
    applyCompileQcd;
    applyCompiledQcd;
    applyDestroyQcd;
    applyInitQcd;
    applyEndQcd;
    qcd_kupl_topology_apply;
    qcd_kupl_dedicated_apply;
    qcd_kupl_family_apply;
  local:
    *;
};
MAP

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

_PYQCD_T0=$(_pyqcd_now)
_pyqcd_notify "KUPL (CPU): compiling ${#SOURCES[@]} sources"
printf 'pyqcd einsum KUPL build\n'
printf '  compiler: %s\n' "$("$CXX" --version | head -n 1)"
printf '  flags:    %s\n' "$FLAGS"
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
    _pyqcd_explain_build_log "$BUILD_LOG"
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

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqcdeinsum_kupl.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
if ! nm -D --defined-only "$BUILT_LIBRARY" | awk '{print $NF}' | grep -qx applyContractQcd; then
    printf 'warning: applyContractQcd is not implemented by this KUPL source set yet\n' >&2
fi
_pyqcd_notify "KUPL (CPU): done in $(( $(_pyqcd_now) - _PYQCD_T0 ))s"
printf 'make.sh: SUCCESS\n'
