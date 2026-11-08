# 安装与构建

PyQCD 从源码树直接运行；原生收缩后端是可选的，不装也能用纯 Python 路径。

## 依赖

| 项 | 用于 |
|---|---|
| Python ≥ 3.10 | 必需 |
| 支持 C++17 的主机编译器 | 原生 KUPL 后端 |
| `nvcc` + Cython | 原生 CUDA 后端 |
| `mpicxx` | 启用 MPI 的 KUPL 构建 |
| OpenBLAS | 数值内核（运行期加载） |

## 安装

```bash
bash install.sh kupl     # CPU (KUPL) backend
bash install.sh cuda     # CUDA backend + Cython extension
bash install.sh          # auto: cuda when nvcc and CUDA sources are present
bash install.sh --help
```

`install.sh` 先构建所选原生库，再执行 `python setup.py build_ext --inplace`，
最后校验库文件确实存在。`PYQCD_SKIP_NATIVE_BUILD=1` 可复用已构建的库；
`PYQCD_EINSUM_SKIP_CYTHON=1` 跳过 Cython 扩展。

每个新 shell 都要先加载环境：

```bash
source ./env.sh
```

`env.sh` 会导出 `PYQCD_EINSUM_*` 路径、把仓库根加入 `PYTHONPATH`、把两个原生库目录
加入 `LD_LIBRARY_PATH`，并设置 `OMP_DYNAMIC=FALSE`。

## CUDA 架构

`make-nv.sh` 按 `PYQCD_CUDA_ARCHITECTURES` → `CMAKE_CUDA_ARCHITECTURES` →
`PYQCD_SM_ARCH` 的顺序解析架构，都没有则用它内置的默认值。换新代显卡时务必显式指定：

```bash
PYQCD_CUDA_ARCHITECTURES=8.9 bash install.sh cuda
```

## MPI

MPI 分片只在 KUPL（CPU）后端实现，且只分片单个输出轴；CUDA 后端没有 MPI 实现。

`PYQCD_EINSUM_ENABLE_MPI` 有三种取值：

| 取值 | 行为 |
|---|---|
| `auto`（默认） | 只有 `mpicxx` 存在**且**有效 `CXX` 能预编译 `#include <mpi.h>` 时才启用 |
| `1` | 强制启用（`CXX` 缺省为 `mpicxx`） |
| `0` | 强制关闭 |

```bash
PYQCD_EINSUM_ENABLE_MPI=1 bash install.sh kupl
```

查询实际构建结果：

```python
import pyqcd.einsum as pe

print(pe.build_info())
# {'backend': 'kupl', 'mpi': False, 'kupl_mpi_build': False,
#  'mpi_label': 'non-MPI build', 'native_library': '.../libpyqcdeinsum_kupl.so'}
```

`mpi_enabled()` 只有在 KUPL 二进制确实编入 MPI 实现时才返回 `True`；CUDA 后端一律报非 MPI。

## 环境变量

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_BACKEND` | `kupl` / `cuda` / `auto` |
| `PYQCD_EINSUM_BUILD_DIR` | 覆盖原生构建目录 |
| `PYQCD_SKIP_NATIVE_BUILD=1` | 复用已有原生库 |
| `PYQCD_EINSUM_SKIP_CYTHON=1` | 跳过 Cython 扩展 |
| `PYQCD_EINSUM_ENABLE_MPI` | `auto` / `1` / `0` |
| `PYQCD_CUDA_ARCHITECTURES` | CUDA 架构（优先级最高） |
| `PYQCD_OPENBLAS_LIBRARY`、`PYQCD_OPENBLAS_INCLUDE_DIR`、`PYQCD_OPENBLAS_FLAGS` | 显式指定 OpenBLAS |
| `NVCC`、`CXX` | 覆盖编译器 |
| `PYQCD_EINSUM_ORACLE=installed` | 差分测试改用已安装的 `opt_einsum` |

## 测试

```bash
source ./env.sh
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum
mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi

bash pyqcd/testing/einsum/simple/run-all.sh              # one-shot sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```

`verify_compat` 需要 `opt_einsum` 作为 oracle：默认用 `refer/git-rep/opt_einsum`
子模块；子模块不可用时设 `PYQCD_EINSUM_ORACLE=installed` 改用已安装的副本
（该 oracle 会 import `opt_einsum.testing`，因此还需要 `pytest`）。
