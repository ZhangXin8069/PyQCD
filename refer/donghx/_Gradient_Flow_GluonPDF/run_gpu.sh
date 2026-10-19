#!/bin/bash
# Run inside an existing single-GPU allocation. This script does not submit jobs.
set -eo pipefail
source /public/home/donghx/load_a100_PyQuad.sh
set -u
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QUDA_PATH=/public/home/donghx/PyQuda_a100_2608/quda-build-a100-ninja/usqcd
export LD_LIBRARY_PATH="${QUDA_PATH}/lib:${LD_LIBRARY_PATH:-}"
if [ -n "${RINGED_QUDA_LIB_DIR:-}" ]; then
    test -f "${RINGED_QUDA_LIB_DIR}/libquda.so"
    export LD_LIBRARY_PATH="${RINGED_QUDA_LIB_DIR}:${LD_LIBRARY_PATH}"
fi
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
exec python -B -m quark_ringed_kinetic_v1.measure "$@"
