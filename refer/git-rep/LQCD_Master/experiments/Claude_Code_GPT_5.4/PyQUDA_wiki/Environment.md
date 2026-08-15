## Create and activate a Python virtual environment

PyQUDA requires `python >= 3.8`. You can check the version of your `python3` by
```bash
python3 --version
```

We recommend using a virtual environment to install your Python packages. You can use [`venv`](https://docs.python.org/3/library/venv.html), [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) to manage your virtual environments. We recommend using the Python standard library `venv` instead of `conda` because of some OpenMPI issues.

Using `venv`
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Using `conda`
```bash
conda create -n pyquda python=3.12
conda activate pyquda
```

## Install prerequisites

### [Cython](https://cython.org/), [NumPy](https://numpy.org/) and [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

```bash
pip install cython numpy mpi4py
```

### [CuPy](https://cupy.dev/) (Recommended GPU backend)

Install the prebuilt `cupy` package matching your CUDA Toolkit version. You can check the version by
```bash
nvcc --version
```
Then install the appropriate package:
```bash
pip install "cupy-cuda12x>=12"  # for CUDA v12.x (recommended)
pip install "cupy-cuda11x>=12"  # for CUDA v11.2 ~ v11.8
```
Check https://docs.cupy.dev/en/stable/install.html#installing-cupy for more information.

Building `cupy` from source is possible but not recommended:
```bash
pip install cupy
```

#### Modification with DTK (DCU clusters)

If you want to install `cupy` on DCU clusters, modify the `cupy` source code and then install it with DTK just like ROCm. Refer to https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental for detailed information.

Download the source code of CuPy version 12.3.0
```bash
wget https://github.com/cupy/cupy/releases/download/v12.3.0/cupy-12.3.0.tar.gz
tar -xzvf cupy-12.3.0.tar.gz
cd cupy-12.3.0
```

Apply the patch
```diff
diff --git a/install/cupy_builder/_features.py b/install/cupy_builder/_features.py
index d12de78c3..8c9ac830a 100644
--- a/install/cupy_builder/_features.py
+++ b/install/cupy_builder/_features.py
@@ -173,7 +173,7 @@ def get_features(ctx: Context) -> Dict[str, Feature]:
             'hiprand',
             'hipsparse',
             'rocfft',
-            'roctx64',
+            #'roctx64',
             'rocblas',
             'rocsolver',
             'rocsparse',
```

Build and install CuPy from source
```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/path/to/dtk-25.04
pip install .
```

### [PyTorch](https://pytorch.org/) (Alternative GPU backend)

`torch >= 2` is an alternative to `cupy`:
```bash
python3 -m pip install "torch>=2"
```

### [DPNP](https://intelpython.github.io/dpnp/) (Experimental, Intel GPU)

`dpnp` is the recommended backend for Intel GPU (experimental support):
```bash
pip install dpnp
```