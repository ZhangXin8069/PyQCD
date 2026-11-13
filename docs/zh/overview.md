# 功能简介

`pyqcd` 是发布在 PyPI 上的格点 QCD 工具库。项目的长期目标是**用梯度流重整化方案
提取胶子 TMD-PDF**；包本身是围绕这条链路的可复用工具。

```bash
pip install pyqcd
```

本站内容全部来自这一次安装，不需要源码仓库。原生加速在运行期可选，但默认安装会
构建 KUPL CPU 后端，因此需要 C++17 工具链。

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)    # e.g. 0.4.1
print(pe.build_info())      # backend / native_available / native_library

info = pe.build_info()
assert pyqcd.__version__.count(".") == 2, pyqcd.__version__
assert info["backend"] in {"cuda", "kupl", "python"}, info["backend"]
```

`pyqcd/__init__.py` 有意保持几乎为空（只留 `__version__`），所有功能都通过子包访问。

## 这个包提供什么

| 子包 | 主要内容 |
|---|---|
| `pyqcd.lattice` | `Nc`、`Nd`、`Ns`、`fm2GeV`、DeGrand–Rossi `gamma(i)`、`Mom_times_sigma`、`SU2combine` / `SU2decompose`、`cg_coefficient` |
| `pyqcd.gauge` | `wilson_loop`、`polyakov_loop`、`clover_field_strength`、`topological_charge`（含密度与平均变体） |
| `pyqcd.smear` | `stout_smear`、`hyp_smear`、`proj_su3` |
| `pyqcd.operator` | `flowed_gluon_ope`、`compute_dual_field_strength`、`FieldStrengthCache`、梯度流 OPE 的 schema 常量 |
| `pyqcd.vertex` | `Mom_VdV_sink_t`、`Mom_VVV_sink_t`、`VdV_sink_t_link`、`apply_momentum_smearing`、本征模压缩 `compress_matrix_V1`…`V4`、`check_orthonormal` |
| `pyqcd.contraction` | `dynamic_contraction`、`parity_and_boundary`、`conjugate_operator`、`classify_structure`、`identify_equivalent_diagrams`、`contract_donghx_2pt_pair` |
| `pyqcd.renorm` | `gradient_flow_renormalized_tmd`、`flow_action_density`、`flow_derivative`、`flow_time_gev_m2`、`flow_scale_mask`，以及 NLO 匹配相关（`A_s`、`C`、`C_gluon_ratio`、`Si`、`M_mu_lambda_nu_rho`） |
| `pyqcd.tools` | 后端切换（`set_backend`、`get_backend`、`get_backend_name`、`get_precision`）、`cached_contract`、`ArraySlicer`、`creat_mom_list`、`dump_env` |
| `pyqcd.analysis` | `Bootstrap`、`FitParams`、`AnaParams`，以及 ratio / 有效质量 / 能量 / FH 拟合与作图 |
| `pyqcd.pipeline` | 集中配置与多步管线；`run_pipeline` 依次执行编号步骤 |
| `pyqcd.parallel` | `detect_resources`、`cpu_threads`、`gpu_info`（只读资源探测；不含 MPI 调度） |
| `pyqcd.einsum` | 收缩引擎：与 `opt_einsum` 兼容的 API，外加可选的原生后端 |

## 收缩引擎

`pyqcd.einsum` 重新实现了 `opt_einsum` 的公开 API，可直接替换使用：

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)

pe.contract("ij,jk->ik", a, b)                        # same as a @ b
pe.contract_path("ij,jk->ik", a, b)                   # (path, PathInfo)
pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))   # reusable expression

assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
_path, _info = pe.contract_path("ij,jk->ik", a, b)
assert int(_info.opt_cost) > 0
assert np.allclose(pe.contract_expression("ij,jk->ik", (2, 3), (3, 4))(a, b), a @ b)
```

### 分层

| 层 | 模块 |
|---|---|
| 公共 API | `contract`、`contract_path`、`contract_expression`、`PathInfo`、`paths`、`backends`、`shared_intermediates` |
| 规划 | `_plan_py.py`（纯 Python 的 greedy/optimal/branch/dp）、`helpers.py`、`parser.py`、`blas.py` |
| 执行 | `_execute.py`（BLAS 直派 + 缓冲池） |
| 原生 | `_native.py`（ctypes）与 `cuda/`（Cython 扩展） |

### 原生后端（可选）

| 后端 | 用于 | `pip install` 时构建条件 |
|---|---|---|
| KUPL | CPU | 有 C++17 编译器；这是默认基线，缺工具链时安装失败 |
| CUDA | GPU | 有 `nvcc` 与 Cython |

用 `PYQCD_SKIP_NATIVE_BUILD=1`（或 `PYQCD_EINSUM_REQUIRE_NATIVE=0`）显式选择
纯 Python 规划。CUDA 始终是机会性构建，不会让安装失败。

### 回退不变量

- 运行期原生执行不可用或失败时自动改用通用路径：结果不变，只是变慢。
- `_native.py` 只做 `ctypes`；import 时不加载库、不编译。
- 任何原生路径失败只 warning 一次并回退，不把异常抛给调用方。
- 拓扑特化计划在原生分配前必须先通过 CPU/GPU arena、单步 GEMM 与 pack-heavy
  资源门。

### 可查询标识

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / native_available / array_backend / native_library
pe.specializations()   # equation-shape specialisations currently registered

info = pe.build_info()
assert isinstance(info, dict) and "backend" in info
assert len(pe.specializations()) >= 1
```

## 生产快照（首发于 0.4 系列）

A100 full-scene 证据使用 `Nx=Ny=Nz=32`、`Lt=64`、`Nev=100`、`Ns=4`、
`Mom=8`、`complex128`。表中为 `selected/generic`，小于 `1` 表示更快。

| 证据 | CPU | GPU |
|---|---|---|
| 43 条分层路径中位比 | **0.861×** | **0.276×** |
| 43 条不快于 PyQCD generic | **42/43** | **42/43** |
| 43 条不快于 `opt_einsum` 3.4.0（`dp`） | **36/43** | **42/43** |
| 核心长链 `seq=6585` | **0.459×** | fallback |

完整拓扑注册表包含 8638 条路径。计划级审计放行 CPU 3833 条、GPU 2864 条；
资源不安全或无收益的计划会在原生分配前回退。因此本版本声明的是“相对 PyQCD
通用路径的普遍加速”，不是“CPU 全面胜过 `opt_einsum`”。反例、口径与复现见
:doc:`performance`。
