#!/usr/bin/env bash
# Build the CUDA C ABI and publish it atomically.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
SHARED_DIR=$(cd "$SCRIPT_DIR/../../kupl/einsum" && pwd)
BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcdeinsum_cuda.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcdeinsum_cuda.so"
VERSION_SCRIPT="$BUILD_DIR/einsum_cuda_exports.map"

# 构建进度：pip 会捕获子进程输出、只在失败时回放，所以进度直接写终端（/dev/tty）；
# 没有终端时退回 stderr。这样 `pip install pyqcd` 期间用户能看见"在编什么、编到哪"，
# 不会误判为卡死。
_pyqcd_notify() {
    { printf '[pyqcd build] %s\n' "$*" > /dev/tty; } 2>/dev/null \
        || printf '[pyqcd build] %s\n' "$*" >&2
}
_pyqcd_now() { date +%s; }
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

# ── CUDA architecture resolution (multi-environment) ────────────────────────
# 优先级（高 → 低）：
#   1. PYQCD_CUDA_ARCHITECTURES     显式指定（可多值，逗号/分号/空格分隔）
#   2. CMAKE_CUDA_ARCHITECTURES     兼容 CMake 习惯
#   3. PYQCD_SM_ARCH                旧变量名
#   4. 自动探测：nvidia-smi compute_cap → GPU 型号回退表
#   5. 默认 6.0 7.0（无 GPU 的构建机 / CI）
# 无论哪种来源，都同时嵌入 cubin 与 PTX（code=[sm_XX,compute_XX]）：
# 新卡即使没有对应 SASS，也能由驱动 JIT 出 PTX 执行，不会静默退回 CPU。
detect_cuda_arch() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    _cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n 1 | tr -d '[:space:]' || true)
    case "$_cap" in
        [0-9]*.[0-9]*) printf '%s' "${_cap%.*}${_cap#*.}"; return 0 ;;
    esac
    _model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null \
        | head -n 1 || true)
    [ -n "$_model" ] || return 1
    # 精确型号族优先于宽泛产品族（对齐 PyQCU env.sh 的回退表）。
    case "$_model" in
        *"Jetson"*"Thor"*|*"Jetson"*"T4000"*|*"Jetson"*"T5000"*) _arch=110 ;;
        *"GB10"*) _arch=121 ;;
        *"B300"*|*"GB300"*) _arch=103 ;;
        *"B100"*|*"B200"*|*"GB200"*) _arch=100 ;;
        *"RTX PRO"*"Blackwell"*|*"Blackwell"*|*"GeForce RTX 50"*) _arch=120 ;;
        *"H100"*|*"H200"*|*"H800"*|*"H20"*|*"GH200"*) _arch=90 ;;
        *"Ada"*|*"GeForce RTX 40"*|*"L40S"*|*"L40"*|*"L20"*|*"L4"*|*"L2"*) _arch=89 ;;
        *"A100"*|*"A800"*|*"A30"*|*"CMP 170HX"*) _arch=80 ;;
        *"GeForce RTX 30"*|*"RTX A"*|*"A40"*|*"A16"*|*"A10"*|*"A2"*|*"MX570"*) _arch=86 ;;
        *"GeForce RTX 20"*|*"GeForce GTX 16"*|*"Quadro RTX"*|*"TITAN RTX"*|*"Tesla T4"*|*"T400"*|*"T600"*|*"MX450"*|*"MX550"*) _arch=75 ;;
        *"RTX 8000"*|*"RTX 6000"*|*"RTX 5000"*|*"RTX 4000"*|*"RTX 3000"*) _arch=75 ;;
        *"V100"*|*"TITAN V"*|*"GV100"*) _arch=70 ;;
        *"GeForce GTX 10"*|*"TITAN Xp"*|*"TITAN X Pascal"*|*"Tesla P40"*|*"Tesla P6"*|*"Tesla P4"*|*"Quadro P"*|*"MX350"*|*"MX330"*|*"MX250"*|*"MX230"*|*"MX150"*) _arch=61 ;;
        *"P100"*|*"GP100"*) _arch=60 ;;
        *"GeForce GTX 9"*|*"TITAN X"*|*"Quadro M"*|*"Tesla M40"*|*"Tesla M60"*|*"Tesla M6"*|*"Tesla M4"*|*"GRID M"*) _arch=52 ;;
        *"Tesla M10"*|*"GeForce GTX 750"*|*"GeForce MX110"*|*"GeForce MX130"*) _arch=50 ;;
        *"Tesla K80"*) _arch=37 ;;
        *"Tesla K40"*|*"Tesla K20"*|*"Quadro K6000"*|*"GTX 780"*|*"GTX TITAN"*|*"TITAN Z"*|*"TITAN Black"*) _arch=35 ;;
        *"Jetson"*"Orin"*) _arch=87 ;;
        *"Xavier"*) _arch=72 ;;
        *"Tegra X2"*|*"Jetson TX2"*) _arch=62 ;;
        *"Tegra X1"*|*"Jetson Nano"*|*"Jetson TX1"*) _arch=53 ;;
        *) return 1 ;;
    esac
    printf '%s' "$_arch"
    return 0
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
    # cubin + PTX：新卡的驱动能 JIT PTX，避免"编得出来却跑不起来"。
    ARCH_FLAGS+=("-gencode=arch=compute_${compact},code=[sm_${compact},compute_${compact}]")
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
    qcd_cuda_pattern_3pt;
    qcd_cuda_pattern_3pt_free;
    qcd_cuda_nucleon_2pt;
    qcd_cuda_topology_apply;
  local:
    *;
};
MAP

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

_PYQCD_T0=$(_pyqcd_now)
_pyqcd_notify "CUDA: compiling ${#SOURCES[@]} sources for arch ${ARCH_LABELS[*]} (nvcc, 约 20 s)"
printf 'pyqcd einsum CUDA build\n'
printf '  nvcc:     %s\n' "$($NVCC --version | tail -n 1)"
printf '  arch:     %s (source: %s; cubin+PTX embedded)\n' \
    "${ARCH_LABELS[*]}" "$ARCH_SOURCE"
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
_pyqcd_notify "CUDA: done in $(( $(_pyqcd_now) - _PYQCD_T0 ))s"
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
