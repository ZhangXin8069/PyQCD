# PyQCD（中文）

[English](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.md) | **中文**

PyQCD 是一个用 Python 编写的格点 QCD 工具库，可选原生加速。它提供传播子与算符收缩、
梯度流、重整化、胶子 TMD-PDF 提取所需的基础组件，并自带一个
**与 `opt_einsum` 兼容**的收缩引擎。

## 安装

```bash
pip install pyqcd
```

原生内核是**可选**的：没有 C++ 编译器（或没有 CUDA）时包照样能装，收缩引擎会改用
纯 Python 规划 —— 结果相同，只是更慢。

设置 `PYQCD_EINSUM_REQUIRE_NATIVE=1` 可把"缺工具链"变成硬错误。

确认实际装上了什么（不需要写 import）：

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --json     # machine-readable
```

> **CUDA 构建。** 构建时会自动探测显卡（`nvidia-smi compute_cap`）并始终嵌入 PTX，
> 当前显卡开箱即用；若要原生 SASS，或在无 GPU 的登录节点上为已知目标构建，
> 设 `PYQCD_CUDA_ARCHITECTURES=<算力>`（例如 `8.9`）。

> **线程按调用自动调档。** 两档自动切换（专用形态用 `kernel` 档，BLAS 链用 `generic` 档），
> 直接作用到已加载的运行时；用 `OMP_NUM_THREADS` /
> `OPENBLAS_NUM_THREADS`（或 `PYQCD_EINSUM_OMP_THREADS` / `PYQCD_EINSUM_BLAS_THREADS`）覆盖，
> `PYQCD_EINSUM_THREADS=off` 关闭，`pyqcd.einsum.thread_info()` 查看。

## 文档

* English — https://py-qcd.readthedocs.io/en/latest/usage.html
* 中文 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

## 代码托管

| 平台 | 用途 | 地址 |
|---|---|---|
| GitLab（gitlab.itp.ac.cn） | 国内开发——主开发仓库 | https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD |
| Gitee | 国内发布——国内镜像 | https://gitee.com/zhangxin8069/PyQCD |
| GitHub | 国外发布——海外镜像 | https://github.com/zhangxin8069/PyQCD |

## 快速开始

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)                       # like a @ b
path, info = pe.contract_path("ij,jk->ik", a, b)     # (path, PathInfo)
int(info.opt_cost)

expr = pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))
expr(a, b)                                           # reusable plan
```

公开接口与 `opt_einsum` 对齐：`contract`、`contract_path`、`contract_expression`、
`shared_intermediates`、`PathInfo`、`paths`、`backends`、`helpers`、`parser`、`blas`。

## 包内有什么

| 子包 | 内容 |
|---|---|
| `pyqcd.lattice` | 格点常数、DeGrand–Rossi `gamma`、SU(2) Clebsch–Gordan |
| `pyqcd.gauge` | Wilson/Polyakov 圈、Clover 场强、拓扑荷 |
| `pyqcd.smear` | HYP 与 Stout 涂抹 |
| `pyqcd.operator` | Clover / 对偶场强、梯度流胶子 OPE 算符 |
| `pyqcd.vertex` | VdV/VVV 顶点、动量涂抹、本征模压缩 |
| `pyqcd.contraction` | Wick 收缩、重子算符、蒸馏收缩 |
| `pyqcd.renorm` | 梯度流、Z_R、混合方案、NLO 匹配、TMD 提取 |
| `pyqcd.analysis` | 重采样、拟合、ratio / 有效质量 / 能量分析 |
| `pyqcd.pipeline` | 集中配置与多步管线 |
| `pyqcd.parallel` | 资源探测与 MPI 元任务规划 |
| `pyqcd.einsum` | 收缩引擎（本包的核心特性） |

## 收缩引擎

* 纯 Python 规划（`greedy` / `optimal` / `branch` / `dp`）永远可用。
* 两个可选原生后端：**KUPL**（CPU）与 **CUDA**（GPU），导出同一套 C ABI，
  共用同一份规划核心。
* 回退是硬性不变量：缺库、原生调用失败或形状被拒时静默降级到 Python 路径 ——
  结果永不改变。
* 一小批格点 QCD 方程形态会走专用执行路径；默认开启，可按调用
  （`contract(..., specialize=False)`）或按进程（`PYQCD_EINSUM_SPECIALIZE=0`）关闭。

可查询标识：

```python
import pyqcd.einsum as pe

pe.build_info()        # backend, mpi, mpi_label, native_library
pe.mpi_enabled()       # whether the KUPL library has real MPI support
pe.specializations()   # registered equation shapes
```

MPI 分片只在 KUPL 后端实现，且只分片单个输出轴；CUDA 后端没有 MPI 实现。

## 性能

下面是**已发布包上可复现**的数字：`pyqcd==0.1.18`
（`build_info(): backend=kupl, array_backend=numpy`）、`opt_einsum==3.4.0` 的 `dp`
策略、`complex128`、预热 3 次后取 15 次中位数。

| 工作负载 | `opt_einsum` | `pyqcd.einsum` | 倍率 |
|---|---|---|---|
| 核子 2pt，nvec=32（生产契约） | 97.67 ms | 17.06 ms | **5.73×** |
| γ 夹心，T=256，nvec=512 | 27.71 ms | 5.03 ms | **5.51×** |
| 通用链式 800³（无专用 kernel） | 17.35 ms | 19.74 ms | 0.88×（更慢） |

这三条无需任何额外设置，直接走已发布的 kernel / 计划缓存。本版本的专用 kernel
只覆盖核子 2pt 与 3pt 两种形态；其余格点形态（γ 夹心、perambulator × 规范场、
sush 系列）得到的是计划缓存与通用执行层。

```python
import numpy as np
import pyqcd.einsum as pe

expr = "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M"   # nucleon 2pt
fast = pe.contract(expr, *operands)                    # specialised path (default)
base = pe.contract(expr, *operands, specialize=False)  # general path
assert np.allclose(fast, base)                         # same numbers, different speed
```

按调用 `specialize=False` 或按进程 `PYQCD_EINSUM_SPECIALIZE=0` 都能关掉这套优化，
两条路径的数值必须一致。

- **内存。** Python 执行路径的峰值 RSS 约为 `opt_einsum` 的 1.4–2 倍
  （实测 1200² 时 18→41 MB，2000² 时 73→104 MB），原因是每一步会多保留一份与输出同尺寸的
  中间量。长时间任务请据此预留内存。

优势来自识别少数格点 QCD 方程形态并改用专用 kernel（融合、缓冲复用），而不是通用 BLAS 链。
但它**不是普适的**：GPU 核子 2pt/3pt 对 `opt_einsum` 双方都已贴近硬件上限（约 1×），
小张量会被单次调用开销吃掉，没有专用 kernel 的形态仍走通用路径。
完整测量口径与限制：
https://py-qcd.readthedocs.io/zh-cn/latest/performance.html

## 测试

这些套件位于 `pyqcd/testing/`，**不在 PyPI 发行包里**；需在源码仓库中运行：

```bash
python -m pyqcd.testing                       # integration suite
bash pyqcd/testing/simple/run-all.sh          # per-function API parity vs opt_einsum
bash pyqcd/testing/einsum/simple/run-all.sh   # einsum sub-system suite
```

## 许可证

MIT
