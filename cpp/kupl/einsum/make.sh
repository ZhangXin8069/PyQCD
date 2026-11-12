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
    if grep -q 'mpi\.h: No such file' "$log" 2>/dev/null; then
        printf 'note: mpi.h was not found. Install Open MPI/MPICH development files\n' >&2
        printf '      (apt install libopenmpi-dev, dnf install openmpi-devel, or\n' >&2
        printf '      conda install -c conda-forge openmpi), or build the serial library\n' >&2
        printf '      with PYQCD_EINSUM_ENABLE_MPI=0.\n' >&2
    elif grep -q 'omp\.h: No such file' "$log" 2>/dev/null; then
        printf 'note: the OpenMP header omp.h is missing. Install a GCC toolchain with\n' >&2
        printf '      OpenMP (apt install build-essential, conda install gxx_linux-64) or\n' >&2
        printf '      drop -fopenmp via PYQCD_EINSUM_FLAGS.\n' >&2
    elif grep -qE 'cannot find -l(openblas|cblas)|openblas.*No such file' "$log" 2>/dev/null; then
        printf 'note: OpenBLAS/CBLAS was not found by the linker. Install it\n' >&2
        printf '      (apt install libopenblas-dev, conda install openblas) or point at it\n' >&2
        printf '      with PYQCD_OPENBLAS_LIBRARY / PYQCD_OPENBLAS_FLAGS.\n' >&2
    elif grep -qE 'undefined reference to .?(MPI_|ompi_)' "$log" 2>/dev/null; then
        printf 'note: MPI symbols are unresolved at link time. Link through mpicxx (or set\n' >&2
        printf '      PYQCD_EINSUM_MPI_LINK_FLAGS="$(mpicxx --showme:link)"), or build the\n' >&2
        printf '      serial library with PYQCD_EINSUM_ENABLE_MPI=0.\n' >&2
    fi
}
BUILD_LOG="$BUILD_DIR/build.log"
MPI_SETTING=${PYQCD_EINSUM_ENABLE_MPI:-auto}
case "$MPI_SETTING" in
    1|on|ON|true|TRUE)
        MPI_REQUEST=1
        ;;
    0|off|OFF|false|FALSE)
        MPI_REQUEST=0
        ;;
    auto|AUTO)
        MPI_REQUEST=auto
        ;;
    *)
        printf 'error: invalid PYQCD_EINSUM_ENABLE_MPI=%s\n' "$MPI_SETTING" >&2
        exit 2
        ;;
esac
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

# ---- MPI 可用性：按**链接结果**判定，而不是按 <mpi.h> 是否可见 ----
#
# 只看 mpi.h 会漏掉一类"能装不能跑"的产物：CXX 在集群/conda 环境里常被固定为
# 普通 g++，而 mpi.h 可能经 CPATH/-I 可见。此时 mpi_runtime.cpp 会带着
# -DPYQCD_EINSUM_ENABLE_MPI=1 编进来，链接却仍用不带 libmpi 的 g++，于是
# DT_NEEDED 里没有 libmpi*，产物安装成功、dlopen 时才报
#   undefined symbol: ompi_mpi_byte
# （是"依赖没登记"，不是"库不存在"）。所以启用 MPI 的前提是：探针共享库的
# DT_NEEDED 真的出现 libmpi/libmpich，且该依赖在加载期可解析。
MPI_LINK_FLAGS=""
MPI_SUMMARY=""
_MPI_PROBE_DETAIL=""

_mpi_wrapper() {
    # 能自报链接参数的 MPI 编译器包装器（OpenMPI/MPICH/Intel 的常见名字）。
    local candidate
    for candidate in mpicxx mpic++ mpiCC; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

_mpi_wrapper_link_flags() {
    local wrapper=$1 output probe
    for probe in --showme:link -show -link_info; do
        if output=$("$wrapper" "$probe" 2>/dev/null) && [ -n "$output" ]; then
            printf '%s\n' "$output"
            return 0
        fi
    done
    return 1
}

_mpi_rpath_flags() {
    # 把 -L<dir> 补成 rpath：运行期 dlopen 时才能定位到刚链上的 libmpi*。
    local flag directory
    for flag in $1; do
        case "$flag" in
            -L*)
                directory=${flag#-L}
                if [ -n "$directory" ] && [ -d "$directory" ]; then
                    directory=$(realpath "$directory" 2>/dev/null || printf '%s' "$directory")
                    printf -- '-Wl,-rpath,%s ' "$directory"
                fi
                ;;
        esac
    done
}

_mpi_library_dir() {
    # MPI 运行库所在目录：补进 rpath，产物就不依赖运行时的 LD_LIBRARY_PATH。
    local compiler=$1 name path
    for name in libmpi.so libmpich.so; do
        path=$("$compiler" -print-file-name="$name" 2>/dev/null || true)
        case "$path" in
            /*)
                if [ -e "$path" ]; then
                    dirname "$(realpath "$path" 2>/dev/null || printf '%s' "$path")"
                    return 0
                fi
                ;;
        esac
    done
    if command -v ldconfig >/dev/null 2>&1; then
        for name in libmpi.so.40 libmpi.so.12 libmpich.so.12; do
            path=$(ldconfig -p 2>/dev/null | awk -v want="$name" '$1 == want {print $NF; exit}')
            if [ -n "$path" ] && [ -e "$path" ]; then
                dirname "$(realpath "$path" 2>/dev/null || printf '%s' "$path")"
                return 0
            fi
        done
    fi
    return 1
}

_mpi_headers_visible() {
    printf '#include <mpi.h>\nint main(){return 0;}\n' |
        "$1" $BASE_FLAGS -x c++ -E - >/dev/null 2>&1
}

_mpi_link_probe() {
    # 用 $1 编译器、$2 追加链接参数构建引用 MPI 符号的探针 .so。
    # 通过条件：链接期能真正解析 MPI 符号（-z defs，排除头文件与运行库 ABI 不一致），
    # DT_NEEDED 含 libmpi*/libmpich*，且 ldd 无 "not found"。
    local compiler=$1
    local extra_flags=${2:-}
    local source="$BUILD_DIR/mpi_link_probe.cpp"
    local library="$BUILD_DIR/mpi_link_probe.so"
    local missing=""
    if ! command -v readelf >/dev/null 2>&1; then
        _MPI_PROBE_DETAIL="readelf is unavailable, so the MPI dependency cannot be verified"
        return 1
    fi
    mkdir -p "$BUILD_DIR"
    printf '#include <mpi.h>\nvoid* pyqcd_mpi_link_probe(void) { return (void*)&MPI_Initialized; }\n' \
        > "$source"
    if ! "$compiler" $BASE_FLAGS $extra_flags -Wl,--no-undefined -shared \
        "$source" -o "$library" >/dev/null 2>&1; then
        _MPI_PROBE_DETAIL="'$compiler ${extra_flags:-}' cannot resolve MPI symbols at link time"
        return 1
    fi
    if ! readelf -d "$library" 2>/dev/null | grep -qE 'NEEDED.*libmpi|NEEDED.*libmpich'; then
        _MPI_PROBE_DETAIL="'$compiler ${extra_flags:-}' links without libmpi*/libmpich* in DT_NEEDED"
        return 1
    fi
    if command -v ldd >/dev/null 2>&1; then
        missing=$(ldd "$library" 2>/dev/null | grep 'not found' | head -n 1)
        if [ -n "$missing" ]; then
            _MPI_PROBE_DETAIL="MPI runtime is not resolvable at load time: $missing"
            return 1
        fi
    fi
    return 0
}

_mpi_resolve_link_flags() {
    # 为 $1 编译器求出一组能让 DT_NEEDED 带上 MPI 运行库的链接参数。
    local compiler=$1 wrapper="" flags="" with_rpath="" mpi_dir="" rpath=""
    if mpi_dir=$(_mpi_library_dir "$compiler"); then
        rpath="-Wl,-rpath,$mpi_dir"
    fi
    if [ -n "${PYQCD_EINSUM_MPI_LINK_FLAGS:-}" ]; then
        if _mpi_link_probe "$compiler" "$PYQCD_EINSUM_MPI_LINK_FLAGS"; then
            MPI_LINK_FLAGS=$PYQCD_EINSUM_MPI_LINK_FLAGS
            MPI_SUMMARY="PYQCD_EINSUM_MPI_LINK_FLAGS"
            return 0
        fi
        _MPI_PROBE_DETAIL="PYQCD_EINSUM_MPI_LINK_FLAGS: $_MPI_PROBE_DETAIL"
        return 1
    fi
    # (1) CXX 本身是 MPI 包装器：链接命令已自带 MPI 运行库。
    if _mpi_link_probe "$compiler" "$rpath"; then
        MPI_LINK_FLAGS=$rpath
        MPI_SUMMARY="$compiler already links the MPI runtime"
        return 0
    fi
    # (2) 借用包装器报告的链接参数，并补齐 -L 目录的 rpath。
    if wrapper=$(_mpi_wrapper) && flags=$(_mpi_wrapper_link_flags "$wrapper"); then
        with_rpath="$flags $(_mpi_rpath_flags "$flags") $rpath"
        if _mpi_link_probe "$compiler" "$with_rpath"; then
            MPI_LINK_FLAGS=$with_rpath
            MPI_SUMMARY="$wrapper link flags"
            return 0
        fi
    fi
    # (3) 最后退路：直接链 MPI 运行库。
    for flags in "-lmpi" "-lmpich"; do
        if _mpi_link_probe "$compiler" "$flags $rpath"; then
            MPI_LINK_FLAGS="$flags $rpath"
            MPI_SUMMARY="$flags"
            return 0
        fi
    done
    return 1
}

MPI_PROBE_CXX=${CXX:-}
if [ -z "$MPI_PROBE_CXX" ] && [ "$MPI_REQUEST" != 0 ]; then
    MPI_PROBE_CXX=$(command -v mpicxx 2>/dev/null || true)
fi
MPI_ENABLED=0
case "$MPI_REQUEST" in
    0)
        MPI_SUMMARY="disabled by PYQCD_EINSUM_ENABLE_MPI=$MPI_SETTING"
        ;;
    *)
        if [ -z "$MPI_PROBE_CXX" ]; then
            MPI_SUMMARY="no MPI-capable C++ compiler on PATH"
        elif ! _mpi_headers_visible "$MPI_PROBE_CXX"; then
            MPI_SUMMARY="'$MPI_PROBE_CXX' cannot include <mpi.h>"
        elif ! _mpi_resolve_link_flags "$MPI_PROBE_CXX"; then
            MPI_SUMMARY="'$MPI_PROBE_CXX' cannot link the MPI runtime ($_MPI_PROBE_DETAIL)"
        else
            MPI_ENABLED=1
        fi
        ;;
esac
# 探针只是判定手段，构建目录里不留残留物（否则会出现在 git status 里）。
rm -f "$BUILD_DIR/mpi_link_probe.cpp" "$BUILD_DIR/mpi_link_probe.so"
if [ "$MPI_ENABLED" = 0 ] && [ "$MPI_REQUEST" = 1 ]; then
    # 显式要求 MPI 时不允许静默降级：这正是"装着能跑、结果退回纯 Python"的成因。
    printf 'error: PYQCD_EINSUM_ENABLE_MPI=%s was requested, but MPI cannot be used: %s\n' \
        "$MPI_SETTING" "$MPI_SUMMARY" >&2
    printf '       Set CXX to an MPI wrapper (e.g. mpicxx), pass the MPI link line with\n' >&2
    printf '       PYQCD_EINSUM_MPI_LINK_FLAGS="$(mpicxx --showme:link)", or build the\n' >&2
    printf '       serial library with PYQCD_EINSUM_ENABLE_MPI=0.\n' >&2
    exit 2
fi
if [ "$MPI_ENABLED" = 0 ] && [ "$MPI_REQUEST" = auto ]; then
    _pyqcd_notify "KUPL (CPU): MPI disabled ($MPI_SUMMARY)"
fi

if [ "$MPI_ENABLED" = 1 ]; then
    DEFAULT_CXX=mpicxx
else
    DEFAULT_CXX=g++
fi
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
if [ "$MPI_ENABLED" = 1 ]; then
    FLAGS="$FLAGS -DPYQCD_EINSUM_ENABLE_MPI=1"
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
    qcd_kupl_nucleon_2pt;
    qcd_kupl_nucleon_2pt_supported;
    qcd_kupl_pattern_3pt;
    qcd_kupl_pattern_3pt_supported;
    qcd_kupl_topology_apply;
    qcd_kupl_dedicated_apply;
    qcd_kupl_family_apply;
    qcd_kupl_mpi_enabled;
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
if [ "$MPI_ENABLED" = 1 ]; then
    printf '  MPI:      enabled (%s)\n' "$MPI_SUMMARY"
    printf '  MPI link: %s\n' "${MPI_LINK_FLAGS:-<none>}"
else
    printf '  MPI:      disabled (%s)\n' "$MPI_SUMMARY"
fi
printf '  OpenBLAS: %s\n' "$_OPENBLAS_SUMMARY"
printf '  sources:  %s\n' "$(for source in "${SOURCES[@]}"; do printf '%s ' "${source##*/}"; done)"
printf '  output:   %s\n' "$PUBLIC_LIBRARY"

set +e
# shellcheck disable=SC2086  # MPI_LINK_FLAGS 是链接参数串，需按空格拆分
"$CXX" "${CXXFLAGS[@]}" \
    -I"$SCRIPT_DIR/include" \
    "${SOURCES[@]}" \
    "${_OPENBLAS_FLAGS[@]}" \
    -Wl,--version-script="$VERSION_SCRIPT" \
    -Wl,-soname,libpyqcdeinsum_kupl.so \
    $MPI_LINK_FLAGS \
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
if [ "$MPI_ENABLED" = 1 ] &&
    ! nm -D --defined-only "$TEMP_LIBRARY" | awk '{print $NF}' | grep -qx applyContractMpiQcd; then
    printf 'error: MPI build does not export applyContractMpiQcd\n' >&2
    exit 1
fi

# MPI 构建的硬校验：MPI 宏编进来了，DSO 就必须登记 MPI 运行库。否则产物能装、
# 编译也"成功"，但 dlopen 会报 undefined symbol: ompi_mpi_byte，pyqcd.einsum
# 只能退回纯 Python 路径。退出码 3 = "产物校验失败"（区别于编译失败），
# setup.py 据此拒绝复用源码树里的旧库。
if [ "$MPI_ENABLED" = 1 ]; then
    if ! readelf -d "$TEMP_LIBRARY" 2>/dev/null | grep -qE 'NEEDED.*libmpi|NEEDED.*libmpich'; then
        printf 'error: MPI build produced a library without an MPI runtime dependency\n' >&2
        printf '       %s has no libmpi*/libmpich* in DT_NEEDED, so dlopen would fail with\n' \
            "$TEMP_LIBRARY" >&2
        printf '       "undefined symbol: ompi_mpi_*". Link flags used: %s\n' \
            "${MPI_LINK_FLAGS:-<none>}" >&2
        printf '       Set CXX=mpicxx (or PYQCD_EINSUM_MPI_LINK_FLAGS) and retry, or build\n' >&2
        printf '       the serial library with PYQCD_EINSUM_ENABLE_MPI=0.\n' >&2
        exit 3
    fi
    _MPI_MISSING=$(ldd "$TEMP_LIBRARY" 2>/dev/null | grep 'not found' || true)
    if [ -n "$_MPI_MISSING" ]; then
        printf 'error: MPI library leaves shared-library dependencies unresolved:\n%s\n' \
            "$_MPI_MISSING" >&2
        printf '       The library would fail to load with dlopen.\n' >&2
        exit 3
    fi
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
