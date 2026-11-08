# 安装

包以 `pyqcd` 的名字发布在 PyPI 上（源码分发 sdist）。pip 会在本地构建，原生内核也在
这一步于本机编译。

```bash
pip install pyqcd
```

当前发行版：`0.1.14`。需要固定版本时：`pip install "pyqcd==0.1.13"`。

PyPI 上**有意不发平台 wheel**：原生库链接的是构建主机的 C++ 运行时，要产出可移植的
manylinux wheel 需要 `auditwheel` 修补。因此首次安装会编译一次（笔记本上约一分钟），
pip 会缓存编译好的 wheel，在相同解释器的其它环境里再装就很快。

## 依赖

| 项 | 用于 |
|---|---|
| Python ≥ 3.10 | 必需 |
| `numpy`、`h5py`、`scipy` | 必需（自动安装） |
| `bash` + C++17 编译器 | 原生 KUPL（CPU）内核 —— 基线构建 |
| `nvcc` | 可选的原生 CUDA（GPU）内核 |
| Cython | 可选的 CUDA 扩展；由 pip 构建环境提供，无需手动安装 |

`requires-python` 为 `>=3.10`。本套文档背后的验证在 Python 3.12（完整测试套件）与
3.14（构建 + 导入 + `python -m pyqcd`）上执行；3.10、3.11、3.13 预期可用，但发布机器上
没有对应的自动化实跑记录。
| `mpicxx` | 启用 MPI 的 KUPL 构建（仅源码仓库） |
| `torch` | 可选：torch 后端（`pip install "pyqcd[torch]"`） |
| `matplotlib` | 可选：作图辅助（`pip install "pyqcd[plot]"`） |

Cython 不需要自己装：`pyproject.toml` 已在 `[build-system] requires` 中列出
`cython>=3.0`，所以 pip 为 `pip install pyqcd` 创建的隔离构建环境里一定有它；
只有当构建使用 `--no-build-isolation` 时，才需要手动 `pip install Cython`。

## 可选 extras

```bash
pip install "pyqcd[cuda]"     # adds cupy-cuda12x (CUDA 12.x)
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

最快的检查是模块入口，它不需要写 import，直接报告实际装上了什么：

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --version  # version only
python -m pyqcd --json     # machine-readable
```

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

### CUDA 架构：自动探测，且始终嵌入 PTX

构建时的架构解析顺序：

| 优先级 | 来源 |
|---|---|
| 1 | `PYQCD_CUDA_ARCHITECTURES`（可多值，逗号/分号/空格分隔） |
| 2 | `CMAKE_CUDA_ARCHITECTURES` |
| 3 | `PYQCD_SM_ARCH`（旧变量名） |
| 4 | 自动探测：`nvidia-smi --query-gpu=compute_cap`，再退到 GPU 型号回退表 |
| 5 | 默认 `6.0 7.0`（无 GPU 的构建机，例如 CI 或集群登录节点） |

每个架构都以 `code=[sm_XX,compute_XX]` 编译——**cubin 与 PTX 并存**。带 PTX 意味着
没有对应 SASS 的新卡仍能运行：驱动会把 PTX JIT 成目标机器码。没有 PTX 时，
kernel 入口返回 `QCD_STATUS_INTERNAL`，引擎会静默退回 CPU 路径。

在 RTX 4060 Laptop（算力 8.9）上实测：

| 构建方式 | kernel 状态 | 结果 |
|---|---|---|
| 旧默认，仅 SASS（`sm_70`） | `3`（`QCD_STATUS_INTERNAL`） | 回落 CPU |
| 自动探测 `8.9`，cubin+PTX | `0`（`QCD_STATUS_OK`） | 走 GPU，rel 2.8e-14 |
| 显式 `7.0`，cubin+PTX | `0`（`QCD_STATUS_OK`） | 经 PTX JIT 走 GPU，rel 2.8e-14 |

因此**开箱即用**已经能在当前显卡上跑 GPU。只有在想要最快路径（为你的卡生成原生
SASS），或构建机没有 GPU、但你明确知道目标架构时，才需要设变量：

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader      # e.g. 8.9
PYQCD_CUDA_ARCHITECTURES=8.9 pip install "pyqcd[cuda]"
```

诊断已安装的包：

```bash
# 1) which architectures were compiled (visible during the build)
pip install -v "pyqcd[cuda]" 2>&1 | grep -E "^  arch:"

# 2) does the installed archive carry a CUDA fatbin at all (cubin + PTX)?
readelf -S <site-packages>/pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so | grep nv_fatbin

# 3) the decisive check — does the GPU kernel really run?
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python -c "
import numpy as np, cupy as cp, pyqcd.einsum as pe
e = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'
r = np.random.default_rng(0)
o = [cp.asarray((r.standard_normal(s)+1j*r.standard_normal(s)).astype(np.complex128))
     for s in [(4,4,16,16)]*3 + [(4,4)]*2 + [(4,16,16,16)]*2 + [(4,4)]]
pe.contract(e, *o)
"
```

提示为 `[kernel: cuda_nucleon_2pt]` 表示真的跑了 GPU kernel，`[plan-cache]` 表示没跑。
注意：`strings | grep compute_` 只对"纯 SASS"产物有效；fatbin 布局下架构名不是明文
ASCII，请改用运行期提示判断。

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

## 后端与数组类型

| 输入的数组 | 实际执行的路径 |
|---|---|
| `numpy` | 装了 KUPL 原生库就走 CPU 原生，否则纯 Python 规划 |
| `cupy` | 特化形态走 CUDA kernel；通用执行器目前可能报 `QCD_STATUS_INTERNAL` 并回落到 Python —— 只会发一次 `RuntimeWarning` 说明原因，结果始终正确 |
| `torch` | torch 后端（`set_backend("torch")`），即由 PyQCD 派发到 `torch.einsum` |

`pyqcd.einsum.build_info()` 同时给出 `backend`（哪个原生库可用）与
`array_backend`（你的数组是什么）；这是两个不同的问题——`backend='cuda'` 的构建，
numpy 输入依然跑在 CPU 库上。
