# PyQCD（中文）

[English](https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD/-/blob/main/README.md) | **中文**

PyQCD 是一个用 Python 编写的格点 QCD 工具库，可选原生加速。它提供传播子与算符收缩、
梯度流、重整化、胶子 TMD-PDF 提取所需的基础组件，并自带一个
**与 `opt_einsum` 兼容**的收缩引擎。

## 安装

需要 **Python 3.9+**。当前 PEP 639 构建元数据要求 `setuptools>=77`，因此 Python 3.8
已不再支持。

```bash
pip install pyqcd
```

**KUPL 原生内核默认是必需的**：`pip install` 会现场编译，缺 C++ 工具链时直接
以可操作的错误信息中止安装，而不是悄悄给你一个只会走纯 Python 规划的包。装上
`g++`/`c++` 与 `bash`、`binutils` 即可；确实只想要纯 Python 规划时显式：
`PYQCD_SKIP_NATIVE_BUILD=1 pip install pyqcd`（`PYQCD_EINSUM_REQUIRE_NATIVE=0` 等价）。
CUDA 仍是机会性构建：缺 `nvcc` 只是拿不到 GPU 内核，不影响 CPU 原生路径。

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

## 0.4.0 发布声明

`0.4.0` 是 `pyqcd.einsum` 的生产特化版本。

* 拓扑特化表覆盖 **8638 条 6–16 算子方程**，KUPL 与 CUDA 共用同一拓扑语义；
  每个计划在原生分配前都经过 fail-open 资源门。
* 43 条 full-scene 扩展审计中，CPU 对 PyQCD 内部通用路径 **42/43** 不慢，
  中位 selected/generic 为 `0.861x`；CUDA **42/43** 不慢，中位 `0.276x`。
  保留的两个反例是 CPU `seq=856`（慢 0.7%）与 GPU `seq=6911`（慢 36.2%）。
* CPU 长链 `seq=6585` 从 **22.445 s** 降到 **10.310 s**（`0.459x`）。
* CUDA 对 `opt_einsum 3.4.0`（`dp`）在 **42/43** 条审计路径上不慢；
  CPU 为 **36/43**。本版本**不声明 CPU 全面胜过 `opt_einsum`**，仍有 7 条 CPU
  路径较慢。

特化是生产优化，不是“所有形态必快”的承诺。原生资源超限、pack 重而 FLOPs 低、
内核缺失或已注册内核运行失败时会主动回退通用路径，数值结果保持不变。

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
| `pyqcd.parallel` | CPU、内存与 GPU 的只读资源探测 |
| `pyqcd.einsum` | 收缩引擎（本包的核心特性） |

## 收缩引擎

* 纯 Python 规划（`greedy` / `optimal` / `branch` / `dp`）永远可用。
* 两个可选原生后端：**KUPL**（CPU）与 **CUDA**（GPU），导出同一套 C ABI，
  共用同一份规划核心。
* 回退是硬性不变量：缺库、原生调用失败或形状被拒时静默降级到 Python 路径 ——
  结果永不改变。
* 完整的 8638 条格点 QCD 拓扑语料会进入资源门控的专用分派；默认开启，可按调用
  （`contract(..., specialize=False)`）或按进程（`PYQCD_EINSUM_SPECIALIZE=0`）关闭。

可查询标识：

```python
import pyqcd.einsum as pe

pe.build_info()        # backend, native_available, array_backend, native_library
pe.specializations()   # registered equation shapes（本版本 8638 条）
```

## 性能

以下均为 A100 计算节点上的 full-scene 基准，而不是 smoke 或小维合成数据。
输入为 `Nx=Ny=Nz=32`、`Lt=64`、`Nev=100`、`Ns=4`、`Mom=8`、`complex128`；
计时前先逐路径对照通用路径。表中为 `selected/generic`，小于 `1` 表示更快。

| 证据集 | CPU | GPU |
|---|---|---|
| 43 条分层生产路径：中位比 | **0.861×** | **0.276×** |
| 43 条：不快于 PyQCD generic | **42/43** | **42/43** |
| 43 条：比 PyQCD generic 至少快 10% | **26/43** | **40/43** |
| 43 条对 `opt_einsum(dp)`：不慢 | **36/43** | **42/43** |

六条核心路径（比值小于 `1` 表示更快）：

| `seq` | CPU / generic | GPU / generic | CPU / `opt_einsum(dp)` | GPU / `opt_einsum(dp)` |
|---:|---:|---:|---:|---:|
| 2584 | 0.856× | 0.265× | 0.598× | 0.152× |
| 866 | 0.936× | 0.248× | 0.891× | 0.213× |
| 855 | 0.982× | 0.177× | 0.842× | 0.130× |
| 129 | 0.791× | fallback，1.017× | 1.297× | 1.179× |
| 1863 | 0.974× | 0.157× | 0.837× | 0.025× |
| 6585 | **0.459×** | fallback | 未完成 | 0.606×，fallback |

全部 **8638** 条计划另做了不执行大张量的结构审计：CPU 资源门放行 3833 条、
拦截 4805 条；CUDA 门放行 2864 条、拦截 5774 条。超大静态 arena、超大单步
GEMM 与“pack 重而 FLOPs 低”的计划不会进入特化路径。43 条 full-scene 路径的
最大相对误差为 CPU `6.95e-15`、GPU `6.68e-15`。

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

优势来自拓扑识别、资源感知的计划选择、收益明确时的融合/专用执行、按生命周期复用
workspace，以及资源边界不安全时主动回退。它**不是普适的**：`opt_einsum(dp)` 仍在
7 条审计过的 CPU 路径上更快，内部也有两个明确的 full-scene 反例；资源门会有意把
不安全计划送回通用路径。完整口径、证据与复现方式：
https://py-qcd.readthedocs.io/zh-cn/latest/performance.html

### PyTorch

通过 `pip install "pyqcd[torch]"` 支持 PyTorch 张量：`contract()` 按操作数类型分派，
`torch.Tensor` 进 → 同设备 `torch.Tensor` 出（由 `torch.einsum` 执行，不走原生库）。
想要原生 kernel，请在边界处转成 numpy 或 cupy。完整对照表与示例：
https://py-qcd.readthedocs.io/zh-cn/latest/usage.html

## 测试

这些套件位于 `pyqcd/testing/`，**不在 PyPI 发行包里**；需在源码仓库中运行：

```bash
python -m pyqcd.testing                       # integration suite
bash pyqcd/testing/simple/run-all.sh          # per-function API parity vs opt_einsum
bash pyqcd/testing/einsum/simple/run-all.sh   # einsum sub-system suite
```

## 许可证

MIT
