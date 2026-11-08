# 使用方式

下面的示例在纯 CPU、只装 `numpy` 的环境里即可运行；原生后端只影响速度，不影响结果。

## 收缩

```python
import numpy as np
import pyqcd.einsum as pe

rng = np.random.default_rng(0)
a = (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))).astype(np.complex128)
b = (rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))).astype(np.complex128)

result = pe.contract("ij,jk->ik", a, b)
path, info = pe.contract_path("ij,jk->ik", a, b)
print(int(info.opt_cost), path)

expr = pe.contract_expression("ij,jk->ik", (3, 4), (4, 5))
for _ in range(3):
    expr(a, b)          # 复用已编译的计划
```

共享中间结果（相同子表达式只算一次）：

```python
with pe.shared_intermediates() as cache:
    pe.contract("ab,bc->ac", a, b, backend="numpy")
    pe.contract("ab,bc->ac", a, b, backend="numpy")
print(pe.count_cached_ops(cache))
```

### 后端与精度

```python
from pyqcd.tools import set_backend, get_backend, get_backend_name, get_precision

set_backend("numpy")          # 也可用 "cupy"、"torch"
print(get_backend_name(), get_precision())
```

### 关闭方程特化

```python
pe.contract("ij,jk->ik", a, b, specialize=False)   # 单次调用关闭
```

环境变量开关：`PYQCD_EINSUM_SPECIALIZE=0`、`PYQCD_EINSUM_DISABLE_FUSED=1`、
`PYQCD_EINSUM_KERNEL_KUPL=0`、`PYQCD_EINSUM_KERNEL_CUDA=0`。

## 格点常数与 Dirac 代数

```python
import pyqcd.lattice as L

print(L.Nc, L.Nd, L.Ns, L.fm2GeV)
print(L.gamma(5).shape)          # (4, 4)，DeGrand–Rossi 基
```

## 规范场观测量与涂抹

```python
import numpy as np
import pyqcd.gauge as G
import pyqcd.smear as S

# 2^4 格点上的单位链接：U[t, z, y, x, mu, colour, colour]
U = np.zeros((2, 2, 2, 2, 4, 3, 3), dtype=np.complex128)
for mu in range(4):
    U[..., mu, :, :] = np.eye(3)

print(G.wilson_loop(U, 1, 1, 0, 1))
print(G.polyakov_loop(U).shape)
print(S.stout_smear(U, nstep=1).shape)     # (2, 2, 2, 2, 4, 3, 3)
print(S.hyp_smear(U).shape)
```

## 梯度流与胶子算符

```python
import pyqcd.renorm as R
import pyqcd.operator as O

print(R.flow_time_gev_m2(1.0, a_fm=0.1))          # 流时（GeV^-2）
print(O.flowed_gluon_ope(U, z_count=1, z_dir=2))
print(O.compute_dual_field_strength(U).shape)
```

基于梯度流的重正化 TMD：

```python
result = R.gradient_flow_renormalized_tmd(
    U, tau=1.0, z_list=[0, 2, 4], b_list=[0], z_dir=2, b_dir=0)
```

## 蒸馏收缩

```python
import pyqcd.contraction as C

# registries 分别持有 perambulator、顶点与 γ 矩阵
result = C.dynamic_contraction(
    operator_groups, peram_registry=..., v_registry=..., gamma_registry=...,
    Cpt="2pt", verbose=True)
```

`parity_and_boundary` 施加宇称投影与反周期边界的符号；`classify_structure` 与
`identify_equivalent_diagrams` 用于在收缩前归并等价算符结构。

## 分析工具

```python
import pyqcd.analysis as A

samples = A.Bootstrap(data, n_samples=200)   # jackknife / bootstrap 重采样
fit = A.FitParams(...)                       # 供 ratio / energy / FH 拟合使用
```

## 并行规划

```python
import pyqcd.parallel as P

resources = P.detect_resources()
plan = P.plan_parallel(m=12, a_mem_mb=500, resources=resources)
print(P.format_plan(plan))
```

规划遵循 `N*a = n*b`（a 为单元任务显存、b 为单卡可用显存、n 为 GPU 数），
并给出进程数 `N`、批次 `X = m/N` 与每卡进程数 `Y = N/n`。
