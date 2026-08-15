## Prerequisites

- OpenMPI / MPICH
- Python >= 3.8
  - `Cython`
  - `numpy`
  - `mpi4py`
  - `cupy >= 12` or `torch >= 2` or `dpnp` (at least one GPU backend)
- QUDA
  - CMake >= 3.18
  - Git
  - GCC >= 7 / Clang >= 5 (supports C++17)
  - CUDA Toolkit >= 11

For detailed instructions to set the environment, refer to https://github.com/CLQCD/PyQUDA/wiki/Environment.

## Build and install QUDA

```bash
git clone https://github.com/lattice/quda.git
mkdir -p quda/build
pushd quda/build
export GPU_ARCH=sm_70  # set to your GPU architecture, e.g., sm_60, sm_70, sm_80, sm_90
cmake -DCMAKE_BUILD_TYPE=RELEASE \
    -DQUDA_GPU_ARCH=${GPU_ARCH} -DQUDA_MPI=ON \
    -DQUDA_COVDEV=ON -DQUDA_MULTIGRID=ON \
    -DQUDA_DIRAC_DEFAULT_OFF=ON \
    -DQUDA_DIRAC_WILSON=ON -DQUDA_DIRAC_CLOVER=ON \
    -DQUDA_DIRAC_STAGGERED=ON -DQUDA_DIRAC_LAPLACE=ON \
    -DQUDA_CLOVER_DYNAMIC=OFF -DQUDA_CLOVER_RECONSTRUCT=OFF
cmake --build . -j$(nproc) && cmake --install .
popd
```
The `GPU_ARCH` environ should be set to the compute capability of your device. The default installation path of QUDA will be `/path/to/quda/build/usqcd`. We will use this path below.

## Install CuPy or PyTorch

PyQUDA requires a GPU backend to handle data on device memory. Choose **one** of the following.

### CuPy (recommended)

Choose the command matching your CUDA toolkit version:
```bash
python3 -m pip install "cupy-cuda12x>=12"  # for CUDA v12.x (recommended)
python3 -m pip install "cupy-cuda11x>=12"  # for CUDA v11.2 ~ v11.8
```

### PyTorch

```bash
python3 -m pip install "torch>=2"
```

## Install PyQUDA

### Install from PyPI (recommended)

```bash
export QUDA_PATH=/path/to/quda/build/usqcd
python3 -m pip install pyquda pyquda-utils
```

`pyquda` is the core package (Cython bindings). `pyquda-utils` includes high-level utilities, I/O wrappers, and the plugin system.

### Install from source

```bash
export QUDA_PATH=/path/to/quda/build/usqcd
git clone --recursive https://github.com/CLQCD/PyQUDA.git
cd PyQUDA/pyquda_core
python3 -m pip install .
cd ..
python3 -m pip install .
```

The environment variable `QUDA_PATH` tells PyQUDA where to find `libquda.so`.

## Using the CLI launcher

PyQUDA provides a command-line launcher `python -m pyquda` that handles MPI grid configuration and other initialization parameters:
```bash
python3 -m pyquda -g 1 1 1 2 -p .cache your_script.py
```

- `-g Gx Gy Gz Gt`: grid size for lattice splitting (product must equal `mpiexec -n N`)
- `-p PATH`: directory for cached QUDA tuning parameters

You can also pass lattice parameters:
```bash
python3 -m pyquda --lattice 4 4 4 8 --t-boundary -1 --anisotropy 1.0 --backend cupy your_script.py
```

## Test the installation

[Chroma](https://github.com/JeffersonLab/chroma) is needed to generate reference files used by test scripts. A precompiled `chroma` executable for most Linux distros is included in the repository via `git-lfs`.

```bash
git lfs install
git lfs pull
```

Run a test by generating reference data with Chroma and comparing with PyQUDA:
```bash
tests/bin/chroma -i tests/test_clover_isotropic.ini.xml
python3 tests/test_clover_isotropic.py
```

You can also use the CLI launcher with test scripts:
```bash
python3 -m pyquda tests/test_clover_cli.py --lattice 4 4 4 8 --t-boundary -1 --anisotropy 2.593684210526316 --backend cupy
```


