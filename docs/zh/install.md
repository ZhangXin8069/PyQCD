# 安装

包以 `pyqcd` 的名字发布在 PyPI 上（源码分发 sdist）。pip 会在本地构建，原生内核也在
这一步于本机编译。

```bash
pip install pyqcd
```

当前发行版：`0.1.4`。需要固定版本时：`pip install "pyqcd==0.1.4"`。

## 依赖

| 项 | 用于 |
|---|---|
| Python ≥ 3.10 | 必需 |
| `numpy`、`h5py`、`scipy` | 必需（自动安装） |
| `bash` + C++17 编译器 | 原生 KUPL（CPU）内核 —— 基线构建 |
| `nvcc` | 可选的原生 CUDA（GPU）内核 |
| Cython | 可选的 CUDA 扩展；由 pip 构建环境提供，无需手动安装 |
| `mpicxx` | 启用 MPI 的 KUPL 构建（仅源码仓库） |
| `torch` | 可选：torch 后端（`pip install "pyqcd[torch]"`） |
| `matplotlib` | 可选：作图辅助（`pip install "pyqcd[plot]"`） |

Cython 不需要自己装：`pyproject.toml` 已在 `[build-system] requires` 中列出
`cython>=3.0`，所以 pip 为 `pip install pyqcd` 创建的隔离构建环境里一定有它；
只有当构建使用 `--no-build-isolation` 时，才需要手动 `pip install Cython`。

## 可选 extras

```bash
pip install "pyqcd[cuda]"     # adds cupy
pip install "pyqcd[mpi]"      # adds mpi4py
pip install "pyqcd[torch]"    # adds torch (optional backend)
pip install "pyqcd[plot]"     # adds matplotlib (Wick/analysis plots)
```

extras 只拉取 Python 依赖；原生库仍在同一次 `pip install` 中、在有工具链时编译。

## 安装过程做了什么

1. **KUPL 总是构建**。CPU 原生库（`cpp/kupl`）是基线：用宿主 C++17 编译器编译，
   然后放入 `pyqcd/einsum/kupl/`。
2. **CUDA 是在此之上的额外尝试，不是替代**。默认 `auto` 模式下，只有当 `nvcc`、
   `cpp/cuda/einsum/src` 下的 CUDA 源码、以及 Cython 三者齐备时，才会再编译 CUDA
   库及其 Cython 扩展。Cython 由 pip 构建环境提供，所以实际起作用的门槛就是
   `nvcc`。KUPL 先编，因此 CUDA 工具链出问题绝不会连带丢掉 CPU 后端。
3. **原生构建失败只是警告，不是安装失败**。CUDA 构建失败发 `RuntimeWarning`，
   安装继续进行，`pyqcd.einsum` 继续使用 KUPL。KUPL 构建失败同样只警告，引擎降级
   到纯 Python 规划。
4. 按实际产出的库安装 `pyqcd`；`build_info()['backend']` 报告运行期真正可用的后端。

下列开关在 `pip install` 构建阶段读取：

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_BACKEND=kupl` | 只构建 KUPL |
| `PYQCD_EINSUM_BACKEND=cuda` | 构建 KUPL **和** CUDA —— 该取值表示“两个都编”，KUPL 仍是必需基线 |
| `PYQCD_EINSUM_BACKEND=auto`（默认） | 总是构建 KUPL；`nvcc`、CUDA 源码与 Cython 齐备时再加编 CUDA |
| `PYQCD_EINSUM_REQUIRE_NATIVE=1` | 把 **KUPL** 的跳过或失败变成硬错误；对 CUDA 从不生效 |
| `PYQCD_EINSUM_SKIP_CYTHON=1` | 假装没有 Cython：`auto` 降级为只编 KUPL，`cuda` 仍编 CUDA 库但不编其 Cython 扩展 |
| `PYQCD_SKIP_NATIVE_BUILD=1` | 完全跳过原生编译，若已有现成库则直接复用；否则只有纯 Python 规划可用 |

## 安装包里有什么

`pip install pyqcd` 安装的是整个工具包：`pyqcd` 本身，加上 `lattice`、`gauge`、
`smear`、`operator`、`vertex`、`contraction`、`renorm`、`tools`、`analysis`、
`pipeline`、`parallel`，以及 `pyqcd.einsum` 和实际构建出的原生产物（KUPL 总会尝试，
CUDA 只在编译成功时才有）。

`pyqcd.testing` **不在包里**。`setup.py` 以
`find_packages(exclude=["pyqcd.testing", "pyqcd.testing.*"])` 发布，因此测试、基准
驱动与回归基线只留在源码仓库。在已安装环境里执行 `python -m pyqcd.testing` 会报
`ModuleNotFoundError: No module named 'pyqcd.testing'`；这类命令请在源码检出根目录
执行（见最后两节）。

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
优先 CUDA：CUDA 扩展导入不了时回落到 KUPL，因此 CUDA 构建被跳过的机器仍然报
`kupl`。`kupl_mpi_build` 只有在 KUPL 库确实编入 MPI 支持时才为 `True`，`mpi` 还要求
该库就是当前生效的后端；CUDA 后端永远没有 MPI。`pe.mpi_enabled()` 单独暴露
KUPL 的构建标志，与当前生效后端无关。

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
| `PYQCD_EINSUM_NATIVE` | `kupl` / `cuda` / `auto`（默认）；`auto` 优先 CUDA，失败回落 KUPL，再回落纯 Python 规划 |
| `PYQCD_EINSUM_SPECIALIZE=0` | 关闭方程形态特化 |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | 关闭原生融合内核 |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | 关闭指定后端的专用 kernel |
| `PYQCD_EINSUM_ORACLE=installed` | 差分测试改用已安装的 `opt_einsum` |

`PYQCD_EINSUM_BACKEND`（见上）在构建阶段做同样的选择：构建期读的是
`PYQCD_EINSUM_BACKEND`，运行期读的是 `PYQCD_EINSUM_NATIVE`。

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

这些命令依赖 `pyqcd.testing`，而它不在 PyPI 包里，因此只能在源码检出根目录执行：

```bash
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum
mpirun -np 2 python -m pyqcd.testing.einsum.verify_mpi

bash pyqcd/testing/einsum/simple/run-all.sh              # einsum sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```

## 源码仓库

本项目在 GitLab 主开发，并同步到 Gitee（国内发布）与 GitHub（国外发布），三者历史一致：

* 国内开发 — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布 — https://gitee.com/zhangxin8069/PyQCD
* 国外发布 — https://github.com/zhangxin8069/PyQCD
