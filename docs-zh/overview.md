# 项目总览

本页概述 PyQCD 仓库的定位、目录与主要模块，并如实记录 `pyqcd.einsum` 子系统的
当前状态、构建与测试入口。所有描述都可在仓库中的对应文件里核对。

## 仓库定位

PyQCD（PyQCD By ZhangXin）是格点 QCD 研究仓库（`lattice-pdf` 迁移），核心目标是
**使用梯度流重整化方案计算核子中的胶子 TMD-PDF**（见仓库根 `AGENTS.md` 开头）。
围绕这条主线，仓库提供：

- 格点常数、DeGrand–Rossi 基的 γ/σ 矩阵与 SU(2) Clebsch–Gordan 系数；
- 规范场观测量（Wilson 圈、Polyakov 圈、Clover 场强、拓扑荷）与涂抹（HYP、Stout）；
- 自动 Wick 收缩、重子算符与 seqperam、批量 einsum 收缩；
- 梯度流、自重整化 Z_R、混合方案、NLO 匹配与连续极限外推；
- PDF/TMD 提取、数据分析与作图，以及一条集中配置的多步管线；
- 自研的 `pyqcd.einsum` 收缩内核（CPU/KUPL 与 CUDA 两套原生后端）。

`pyqcd/__init__.py` 有意保持为空（只留 `__version__ = "0.1.0"`），以避免循环导入，
因此功能都通过子包访问。

## 顶层目录

| 目录/文件 | 内容 |
|---|---|
| `pyqcd/` | 生产包：格点、规范场、算符、收缩、einsum、重整化、管线、分析、测试等子包 |
| `pyqcd/testing/` | 原 `examples/` 的唯一归属：集成测试入口、示例、回归契约、冻结基线与 TMD/谱学工作流 |
| `cpp/` | 原生源码：`cpp/kupl/einsum/`（CPU 后端）与 `cpp/cuda/einsum/`（CUDA 后端） |
| `docs/` | 中文报告（Markdown、LaTeX 与编译后的 PDF）与图片 |
| `docs-zh/` | 本文档工程：Read the Docs 项目 `pyqcd-zh` 的配置与页面源文件 |
| `refer/` | 只读外部参考；`refer/git-rep/` 是 git 子模块（quda、PyQUDA、opt_einsum 等） |
| `logs/`、`data/` | 纯文本运行记录，以及历史数据、图像与运行脚本 |
| `skills/` | 项目领域技能与运维技能 |
| `install.sh`、`env.sh` | 安装入口与环境脚本 |
| `.readthedocs.yaml` | 主项目（英文工程）的 Read the Docs 配置，指向 `docs/conf.py` |

## 主要模块

`pyqcd/` 下的子包及其职责（各子包的 `AGENTS.md` 是内部契约的唯一真源）：

| 子包 | 职责 |
|---|---|
| `pyqcd/lattice` | 格点常数（Nc、fm2GeV）、DeGrand–Rossi 基 γ/σ 矩阵、SU(2) Clebsch–Gordan 系数 |
| `pyqcd/gauge` | 纯数组规范观测量：Wilson 矩形圈、Polyakov 圈、Clover 场强与拓扑荷 |
| `pyqcd/smear` | HYP 涂抹与 Stout 涂抹 |
| `pyqcd/operator` | Clover 场强 F 与对偶 F̃、胶子 OPE 算符、`.lime` 读取、TMD staple 扩展 |
| `pyqcd/contraction` | 自动 Wick 收缩、重子算符、seqperam、动态收缩（蒸馏 2pt/3pt 核心） |
| `pyqcd/vertex` | VdV/VVV 顶点构造与相位因子、本征模压缩 |
| `pyqcd/renorm` | 重整化核心链：Z_R、混合方案、NLO 匹配、连续极限外推、梯度流、TMD 提取 |
| `pyqcd/tools` | 后端切换（numpy/cupy/torch）、缓存 einsum、切片工具与 HDF5/ASCII IO |
| `pyqcd/pipeline` | 集中配置与多步管线调度（含 `tmd` 步），产物写入 `logs/`；带数据守卫与断点续跑 |
| `pyqcd/analysis` | Jackknife/Bootstrap、meff、ratio_3pt、色散拟合、裸矩阵元提取与作图 |
| `pyqcd/einsum` | 收缩引擎：`opt_einsum` 公共 API 的自包含实现 + 原生 CPU/CUDA 后端 |
| `pyqcd/parallel` | 资源检测与 MPI 元任务调度（`python -m pyqcd.parallel`） |
| `pyqcd/testing` | 中心集成测试入口与全部可运行示例、回归与对照套件 |

## einsum 子系统现状

### 分层

`pyqcd/einsum/AGENTS.md` 固定了分层与文件归属：`__init__.py` 只做 re-export；
`_api.py` 提供 `contract` / `contract_path`；`_pathinfo.py` 是 `PathInfo`；
`_expr.py` 是 `ContractExpression` / `contract_expression`；`paths.py` 管理路径优化器
注册；`backends/` 负责后端分派；`_plan_py.py` 是纯 Python 规划回退；`helpers.py`、
`parser.py`、`blas.py` 提供规模/FLOPs 计算、下标解析与 BLAS 判定；`_execute.py` 是执行层
（BLAS 直派 + 缓冲池）；`_native.py` 是 `ctypes` 绑定（RTLD_LOCAL + 签名表 + 惰性加载）；
原生实现位于 `cpp/` 下。

### 两套原生后端

| 后端 | 源码 | 产物 | 说明 |
|---|---|---|---|
| KUPL（CPU） | `cpp/kupl/einsum/` | `libpyqcdeinsum_kupl.so` | 不涉及 CUDA 与 Cython；MPI 分片只在这一侧有真实实现 |
| CUDA（GPU） | `cpp/cuda/einsum/` | `libpyqcdeinsum_cuda.so` + Cython 扩展 `pyqcd.einsum.cuda.einsum` | 需要 `nvcc` 与 Cython；未实现 MPI 分片 |

两个原生库导出同一组 8 个 `Qcd` 入口（`applyPlanQcd`、`applyContractQcd`、
`applyContractMpiQcd`、`applyCompileQcd`、`applyCompiledQcd`、`applyDestroyQcd`、
`applyInitQcd`、`applyEndQcd`）。

### 回退不变量

- 缺 `.so` 时自动改用纯 Python 规划（`_plan_py.py`），结果不变、仅性能下降；
- `_native.py` 只做 `ctypes`，import 时不加载库、不编译、不报错；
- 任何原生路径失败都记录一次 warning 并回退纯 Python，不抛给调用方。

### 可查询标识

```python
import pyqcd.einsum as pe

pe.build_info()    # dict: backend / mpi / kupl_mpi_build / mpi_label / native_library
pe.mpi_enabled()   # bool: 当前 KUPL 二进制是否编入真实 MPI 分片
pe.specializations()  # 当前生效的方程特化
```

默认（未显式开启 MPI 的）构建下 `mpi_label` 为「非 MPI 版本」；CUDA 后端一律标注为
「非 MPI 版本（CUDA 后端未实现 MPI 分片）」。

### 方程特化与开关

九条方程形态（E1、E4、E5、P1/P2、E2、S1/S2/S3）按状态如实标注：一部分已接线为专用
kernel，一部分只做 plan-cache，没有专用执行路径。可用开关包括
`contract(..., specialize=False)`、`PYQCD_EINSUM_SPECIALIZE=0`、
`PYQCD_EINSUM_DISABLE_FUSED=1`、`PYQCD_EINSUM_KERNEL_KUPL=0`、
`PYQCD_EINSUM_KERNEL_CUDA=0`。

### 截至 2026-11-08 的实测状态

以下数字引自仓库内报告 `docs/pyqcd_einsum_overall_20261108.md`（该报告写于
2026-11-08 08:55，之后可能已变化，请以最新记录为准）：

- 已知缺陷 0 个；与 `opt_einsum` 的差分测试 `verify_compat` 为 980/980 通过；
- `verify_native` 在 kupl 与 cuda 两个后端各 16 passed；
- `python -m pyqcd.testing` 为 115 passed / 1 skipped（skip 为缺少 xelatex）；
- CPU 2pt/3pt 与 GPU 3pt 已有专用 kernel，且快于通用路径与 `opt_einsum`；
- CPU 侧计时受机器负载干扰，`kupl_3pt` 的结论标注为「待独占复测」，不作为验收依据。

构建组合、产物 soname 与逐项复核命令见 `docs/pyqcd_einsum_build_matrix_20261108.md`
与 `cpp/kupl/einsum/AGENTS.md`、`cpp/cuda/einsum/AGENTS.md`。

## 构建与测试入口

```bash
bash install.sh kupl     # 构建 CPU（KUPL）后端
bash install.sh cuda     # 构建 CUDA 后端
bash install.sh          # 默认 auto：有 nvcc 且有 CUDA 源码则 cuda，否则 kupl

source ./env.sh          # PYTHONPATH / LD_LIBRARY_PATH / 后端选择
python -m pyqcd.testing  # 全量集成测试
```

更细的构建参数（`PYQCD_CUDA_ARCHITECTURES`、MPI 开关与 `pe.build_info()` 查询）、
环境变量清单与分项测试入口见[安装与构建](install.md)。

## 文档体系

- `docs/` 下已有的中文 Markdown 报告与 LaTeX/PDF 报告由主项目的 Sphinx 工程
  （根 `.readthedocs.yaml` + `docs/conf.py`）收录；
- `docs-zh/` 是本中文项目（Read the Docs 项目 `pyqcd-zh`）的独立文档工程，只收录
  `docs-zh/` 目录下的说明页，`conf.py` 中不 import `pyqcd`，构建不依赖原生库。
