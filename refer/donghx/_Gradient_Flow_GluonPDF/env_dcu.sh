#!/bin/bash
# Retain user's Python/CuPy/MPI setup; use the runtime matching the HIP QUDA build.
source /public/home/donghx/load_dcu_py38_cupy_mpi.sh
export LANDAU_DTK=/public/home/sunp/DCU_2024_11_17/dtk-24.04.3
export QUDA_PATH=/public/home/sunp/DCU_2024_11_17/scidac/install/pyquda-gfx906
export DTK_HOME="$LANDAU_DTK" ROCM_PATH="$LANDAU_DTK" HIP_PATH="$LANDAU_DTK/hip"
export PATH="$LANDAU_DTK/bin:$LANDAU_DTK/llvm/bin:$LANDAU_DTK/hip/bin:$PATH"
export LD_LIBRARY_PATH="$QUDA_PATH/lib:$LANDAU_DTK/lib:$LANDAU_DTK/lib64:$LANDAU_DTK/hip/lib:$LANDAU_DTK/llvm/lib:$LANDAU_DTK/roctracer/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$LANDAU_ROOT/vendor"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPI4PY_RC_THREAD_LEVEL=serialized OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp UCX_WARN_UNUSED_ENV_VARS=n
export UCX_MEMTYPE_CACHE=n
export QUDA_ENABLE_P2P=0 QUDA_ENABLE_NVSHMEM=0
