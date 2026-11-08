# 功能简介

PyQCD 面向格点 QCD 计算，长期目标是**用梯度流重整化方案提取核子中的胶子 TMD-PDF**；
包本身是这条链路以及通用格点计算的构件集合。

`pyqcd/__init__.py` 有意保持为空（只留 `__version__`），所有功能都通过子包访问。

## 各子包提供什么

| 子包 | 主要内容 |
|---|---|
| `pyqcd.lattice` | `Nc`、`Nd`、`Ns`、`fm2GeV`、DeGrand–Rossi 基的 `gamma(i)`、`Mom_times_sigma`、`SU2combine` / `SU2decompose`、`cg_coefficient` |
| `pyqcd.gauge` | `wilson_loop`、`polyakov_loop`、`clover_field_strength`、`topological_charge`（含密度与平均变体） |
| `pyqcd.smear` | `stout_smear`、`hyp_smear`、`proj_su3` |
| `pyqcd.operator` | `flowed_gluon_ope`、`compute_dual_field_strength`、`FieldStrengthCache`、梯度流 OPE 的 schema 常量 |
| `pyqcd.vertex` | `Mom_VdV_sink_t`、`Mom_VVV_sink_t`、`VdV_sink_t_link`、`apply_momentum_smearing`、本征模压缩 `compress_matrix_V1`…`V4`、`check_orthonormal` |
| `pyqcd.contraction` | `dynamic_contraction`、`parity_and_boundary`、`conjugate_operator`、`classify_structure`、`identify_equivalent_diagrams`、`contract_donghx_2pt_pair` |
| `pyqcd.renorm` | `gradient_flow_renormalized_tmd`、`flow_action_density`、`flow_derivative`、`flow_time_gev_m2`、`flow_scale_mask`，以及 NLO 匹配相关（`A_s`、`C`、`C_gluon_ratio`、`Si`、`M_mu_lambda_nu_rho`） |
| `pyqcd.tools` | 后端切换（`set_backend`、`get_backend`、`get_backend_name`、`get_precision`）、`cached_contract`、`ArraySlicer`、`creat_mom_list`、`dump_env` |
| `pyqcd.analysis` | `Bootstrap`、`FitParams`、`AnaParams`，以及 ratio / 有效质量 / 能量 / FH 拟合与作图 |
| `pyqcd.pipeline` | 集中配置与多步管线（`run_pipeline` 依次执行编号步骤） |
| `pyqcd.parallel` | `detect_resources`、`plan_parallel`、`run_meta_task`、`run_parallel_pipeline`、`format_plan` |
| `pyqcd.einsum` | 收缩引擎：与 `opt_einsum` 兼容的 API，外加可选的 KUPL / CUDA 原生后端 |
| `pyqcd.testing` | 集成测试入口与全部可运行的示例 / 回归套件 |

## 收缩引擎

`pyqcd.einsum` 重新实现了 `opt_einsum` 的公开 API，可直接替换使用：

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)          # same as a @ b
pe.contract_path("ij,jk->ik", a, b)     # (path, PathInfo)
pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))   # reusable expression
```

### 分层

| 层 | 模块 |
|---|---|
| 公共 API | `contract`、`contract_path`、`contract_expression`、`PathInfo`、`paths`、`backends`、`shared_intermediates` |
| 规划 | `_plan_py.py`（纯 Python 的 greedy/optimal/branch/dp）、`helpers.py`、`parser.py`、`blas.py` |
| 执行 | `_execute.py`（BLAS 直派 + 缓冲池） |
| 原生 | `pyqcd/einsum/_native.py`（ctypes）与 `pyqcd/einsum/cuda/`（Cython）；C ABI 位于 `cpp/` |

### 两套原生后端

| 后端 | 源码 | 产物 |
|---|---|---|
| KUPL（CPU） | `cpp/kupl/einsum/` | `libpyqcdeinsum_kupl.so` |
| CUDA（GPU） | `cpp/cuda/einsum/` | `libpyqcdeinsum_cuda.so` + Cython 扩展 |

两者导出同一组 8 个 `Qcd` 入口；规划核心被同时编译进两侧，因此路径与代价天然一致。

### 回退不变量

- 缺少 `.so` 时自动改用纯 Python 规划：结果不变，只是变慢；
- `_native.py` 只做 `ctypes`，import 时不加载库、不编译；
- 任何原生路径失败只 warning 一次并回退，不把异常抛给调用方。

### 可查询标识

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / mpi / kupl_mpi_build / mpi_label / native_library
pe.mpi_enabled()       # whether the KUPL binary has real MPI support
pe.specializations()   # equation-shape specialisations currently registered
```
