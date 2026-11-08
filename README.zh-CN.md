# PyQCD（中文）

[English](README.md) | **中文**

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

## 文档

* English — https://py-qcd.readthedocs.io/en/latest/usage.html
* 中文 — https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

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

在同一台机器、同一 `complex128` dtype、同线程数、同 `dp` 策略下与
`opt_einsum` 3.4.0 实测对比：

| 工作负载 | `opt_einsum` | `pyqcd.einsum` | 倍率 |
|---|---|---|---|
| 核子 3pt（GPU，特化） | 285.40 ms | 30.74 ms | **9.29×** |
| γ 夹心 L24（CPU） | 1188.9 ms | 142.7 ms | **8.33×** |
| γ 夹心 L24（GPU） | 0.373987 s | 0.110959 s | **3.371×** |
| 核子 2pt（CPU，N=100） | 120.43 ms | 22.65 ms | **5.32×** |
| perambulator × 规范场 L24（GPU） | 1.112793 s | 1.095407 s | 1.016×（持平） |

按真实生产各形态耗时占比加权，整体约 **2.82×**（4757 s → 1688 s，模型估计）。

优势来自识别少数格点 QCD 方程形态并改用专用 kernel（融合、缓冲复用），而不是通用 BLAS 链。
但它**不是普适的**：GPU 核子 2pt/3pt 对 `opt_einsum` 双方都已贴近硬件上限（约 1×），
小张量会被单次调用开销吃掉，没有专用 kernel 的形态仍走通用路径。
完整测量口径与限制：
https://py-qcd.readthedocs.io/zh-cn/latest/performance.html

## 测试

在源码仓库中运行：

```bash
python -m pyqcd.testing                       # integration suite
bash pyqcd/testing/simple/run-all.sh          # per-function API parity vs opt_einsum
bash pyqcd/testing/einsum/simple/run-all.sh   # einsum sub-system suite
```

## 许可证

MIT
