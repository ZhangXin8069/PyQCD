#!/usr/bin/env bash
# Build the CUDA einsum backend: cpp/cuda/einsum/libpyqcdeinsum.so
#
# Mirrors PyQCU/cpp/cuda/qcu/make-nv.sh: architecture resolution order is
# PYQCD_CUDA_ARCHITECTURES > CMAKE_CUDA_ARCHITECTURES > PYQCD_SM_ARCH >
# nvidia-smi, and the archive is published through an atomically swapped
# symlink.
#
# The planner is shared with the KUPL backend: cpp/kupl/einsum/src/plan_core.cpp
# is compiled into this archive directly (never copied) against
# cpp/kupl/einsum/include/einsum_plan.h, so both backends plan identically.
#
# The numerics providers (qcd_cuda_permute / qcd_cuda_hadamard /
# qcd_cuda_batched_gemm) live in sibling .cu files written by other groups and
# are referenced weakly by src/apply_plan.cu. A build without them still
# produces a fully functional planning ABI; --require-kernels turns that state
# into a hard error for CI.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
SHARED_DIR=$(cd "$SCRIPT_DIR/../../kupl/einsum" && pwd)
BUILD_DIR=${PYQCD_EINSUM_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILT_LIBRARY="$BUILD_DIR/libpyqcdeinsum.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libpyqcdeinsum.so"
VERSION_SCRIPT="$BUILD_DIR/einsum_cuda_exports.map"
BUILD_LOG="$BUILD_DIR/build.log"
NVCC=${NVCC:-nvcc}
SHARED_CORE="$SHARED_DIR/src/plan_core.cpp"
MODE=build
REQUIRE_KERNELS=${PYQCD_EINSUM_REQUIRE_KERNELS:-0}
PROVIDER_SYMBOLS=(qcd_cuda_permute qcd_cuda_hadamard qcd_cuda_batched_gemm)

usage() {
    printf 'Usage: bash %s [--clean] [--require-kernels]\n' "${BASH_SOURCE[0]:-$0}"
    printf '  --clean            remove the build directory and the published library\n'
    printf '  --require-kernels  fail (exit 1) when a numerics provider is missing\n'
    printf '                     instead of publishing a planning-only archive\n'
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

# ABI guard: include/einsum_define.h is a byte-identical mirror of the frozen
# KUPL header. If the mirror drifts, the two backends would export different
# ABIs under the same soname, which is not a recoverable condition.
if ! cmp -s "$SCRIPT_DIR/include/einsum_define.h" "$SHARED_DIR/include/einsum_define.h"; then
    printf 'error: %s/include/einsum_define.h differs from the frozen ABI header\n' "$SCRIPT_DIR" >&2
    printf '       %s/include/einsum_define.h\n' "$SHARED_DIR" >&2
    printf '       the CUDA backend must export the same ABI as KUPL; resync the mirror\n' >&2
    exit 1
fi

if [ ! -f "$SHARED_CORE" ]; then
    printf 'error: shared planning core not found: %s\n' "$SHARED_CORE" >&2
    exit 1
fi

shopt -s nullglob
LOCAL_SOURCES=("$SCRIPT_DIR"/src/*.cu "$SCRIPT_DIR"/src/*.cpp)
shopt -u nullglob
if [ ${#LOCAL_SOURCES[@]} -eq 0 ]; then
    printf 'error: no CUDA sources in %s/src\n' "$SCRIPT_DIR" >&2
    exit 1
fi
# The shared planning core is always compiled, regardless of the local glob.
SOURCES=("${LOCAL_SOURCES[@]}" "$SHARED_CORE")

if [ -n "${PYQCD_CUDA_ARCHITECTURES:-}" ]; then
    ARCHS=$PYQCD_CUDA_ARCHITECTURES
elif [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
    ARCHS=$CMAKE_CUDA_ARCHITECTURES
elif [ -n "${PYQCD_SM_ARCH:-}" ]; then
    ARCHS=$PYQCD_SM_ARCH
else
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')
    if [ -z "$CAP" ]; then
        printf 'error: cannot determine GPU architecture; set PYQCD_SM_ARCH\n' >&2
        exit 1
    fi
    ARCHS=$CAP
fi

ARCH_FLAGS=()
for arch in ${ARCHS//,/ }; do
    ARCH_FLAGS+=("-gencode=arch=compute_${arch/./},code=sm_${arch/./}")
done

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

printf 'pyqcd einsum CUDA build\n'
printf '  nvcc:   %s\n' "$($NVCC --version | tail -n 1)"
printf '  arch:   %s\n' "$ARCHS"
printf '  src:    %s\n' "$(for source in "${SOURCES[@]}"; do printf '%s ' "${source##*/}"; done)"
printf '  shared: %s\n' "$SHARED_CORE"

TEMP_LIBRARY="$BUILT_LIBRARY.tmp.$$"
TEMP_PUBLIC="$PUBLIC_LIBRARY.tmp.$$"
trap 'rm -f "$TEMP_LIBRARY" "$TEMP_PUBLIC"' EXIT

set +e
"$NVCC" -O3 -std=c++17 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
    "${ARCH_FLAGS[@]}" \
    -I"$SHARED_DIR/include" \
    -I"$SCRIPT_DIR/include" \
    "${SOURCES[@]}" \
    -lcublas -lcudart \
    -Xlinker --version-script="$VERSION_SCRIPT" \
    -Xlinker -soname -Xlinker libpyqcdeinsum.so \
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

# .symtab survives the link, so a localized provider definition is still visible.
MISSING=()
for symbol in "${PROVIDER_SYMBOLS[@]}"; do
    if ! nm --defined-only "$TEMP_LIBRARY" 2>/dev/null | awk '{print $NF}' | grep -qx "$symbol"; then
        MISSING+=("$symbol")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    printf '\n' >&2
    printf 'warning: numerics providers are missing from this build:\n' >&2
    for symbol in "${MISSING[@]}"; do
        printf '           %s (weak, resolves to NULL)\n' "$symbol" >&2
    done
    printf '         applyPlanQcd is complete and callable; applyContractQcd returns\n' >&2
    printf '         QCD_STATUS_INTERNAL (3) for every step until the providers land\n' >&2
    printf '         in %s/src (permute.cu / batched_gemm.cu).\n' "$SCRIPT_DIR" >&2
    if [ "$REQUIRE_KERNELS" = "1" ]; then
        printf 'error: --require-kernels is set, refusing to publish a planning-only archive\n' >&2
        exit 1
    fi
    printf '         pass --require-kernels to make this a hard failure.\n\n' >&2
fi

mv -Tf "$TEMP_LIBRARY" "$BUILT_LIBRARY"
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$(realpath "$BUILD_DIR")")/libpyqcdeinsum.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC"
mv -Tf "$TEMP_PUBLIC" "$PUBLIC_LIBRARY"
trap - EXIT

printf 'native library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
if [ ${#MISSING[@]} -gt 0 ]; then
    printf 'make-nv.sh: SUCCESS (planning ABI only; %d provider(s) pending)\n' "${#MISSING[@]}"
else
    printf 'make-nv.sh: SUCCESS\n'
fi
