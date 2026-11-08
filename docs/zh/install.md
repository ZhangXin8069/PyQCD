# 安装

包以 `pyqcd` 的名字发布在 PyPI 上（源码分发 sdist）。pip 会在本地构建，因此也能顺带
编译可选的原生内核。

```bash
pip install pyqcd
```

当前发行版：`0.1.2`。需要固定版本时：`pip install "pyqcd==0.1.2"`。

## 依赖

| 项 | 用于 |
|---|---|
| Python ≥ 3.10 | 必需 |
| `numpy` | 必需（自动安装） |
| C++17 编译器 | 原生 KUPL（CPU）内核 |
| `nvcc` + Cython | 原生 CUDA（GPU）内核 |
| `mpicxx` | 启用 MPI 的 KUPL 构建（仅源码仓库） |

## 可选 extras

```bash
pip install "pyqcd[cuda]"     # adds cupy
pip install "pyqcd[mpi]"      # adds mpi4py
```

extras 只拉取 Python 依赖；原生库仍在同一次 `pip install` 中、在有工具链时编译。

## 安装过程做了什么

1. 选择后端（可用 `PYQCD_EINSUM_BACKEND=kupl|cuda` 覆盖）：
   * 有 `nvcc` **且**有 Cython、且存在 CUDA 源码 → `cuda`；
   * 否则有 C++ 编译器 → `kupl`；
   * 都没有 → 不编原生库，引擎走纯 Python 规划。
2. 编译原生库并放入包目录。
3. 安装包本身；工具链缺失或编译失败只发一次警告。

设置 `PYQCD_EINSUM_REQUIRE_NATIVE=1` 可把该警告变成硬错误；
`PYQCD_SKIP_NATIVE_BUILD=1` 可完全跳过原生编译。

## 验证安装

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)
print(pe.build_info())
# {'backend': 'kupl', 'mpi': False, 'kupl_mpi_build': False,
#  'mpi_label': 'non-MPI build', 'native_library': '.../libpyqcdeinsum_kupl.so'}
```

`backend` 取值为 `cuda`、`kupl` 或 `python`（后者表示正在用纯 Python 规划）。
`mpi_enabled()` 只有在 KUPL 库确实编入 MPI 支持时才为 `True`；CUDA 后端永远没有。

功能自检：

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
```

## 运行期环境变量

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_BACKEND` | `kupl` / `cuda` / `auto`（构建期同样生效） |
| `PYQCD_EINSUM_SPECIALIZE=0` | 关闭方程形态特化 |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | 关闭原生融合内核 |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | 关闭指定后端的专用 kernel |
| `PYQCD_EINSUM_ORACLE=installed` | 差分测试改用已安装的 `opt_einsum` |

## 从源码仓库构建（开发用途）

```bash
bash install.sh kupl     # CPU (KUPL) backend
bash install.sh cuda     # CUDA backend + Cython extension
bash install.sh          # auto
source ./env.sh          # PYTHONPATH / LD_LIBRARY_PATH / backend selection
```

CUDA 架构按 `PYQCD_CUDA_ARCHITECTURES` → `CMAKE_CUDA_ARCHITECTURES` → `PYQCD_SM_ARCH`
的顺序解析；换新代显卡时务必显式指定：

```bash
PYQCD_CUDA_ARCHITECTURES=8.9 bash install.sh cuda
```

MPI 分片只在 KUPL 后端实现，且只分片单个输出轴。`PYQCD_EINSUM_ENABLE_MPI` 支持
`auto`（默认：仅当有 `mpicxx` 且有效 `CXX` 能预编译 `#include <mpi.h>` 时启用）、
`1` 与 `0`。

## 测试（源码仓库）

```bash
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum
mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi

bash pyqcd/testing/einsum/simple/run-all.sh              # einsum sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```
