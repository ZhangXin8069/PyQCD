# Gauge public-observables forward eval

## 场景

独立 `gpt-5.6-luna` 评估器只读 `pyqcd-gauge/SKILL.md` 与
`references/observables.md`，要求从
`(Nt,Nz,Ny,Nx,4,3,3)` 的 SU(3) 链接给出以下对象的确切 API 和语义：

- `2x3` Wilson 矩形的逐起点值；
- 时间 Polyakov 圈；
- 标准 Clover 场强与历史未去迹基元；
- 逐点 `q(x)`、总拓扑荷与体积平均；
- 方向轴映射、最小契约和不能升级的物理结论。

评估器还必须判断文档是否自洽可执行，不得读生产实现来猜缺失语义。

## RED

首轮能正确拒绝丢弃 Polyakov 复中心相位，也能区分默认无迹场、
历史未去迹场、`sum_x q(x)` 和 `mean_x q(x)`，但报告了五项文档缺口：

1. 拓扑总荷/平均入口没有完整 `traceless` 签名；
2. `polyakov_loop(..., average=True)` 语义不完整；
3. Polyakov 移除闭合轴后的 shape/轴序不明；
4. Clover 的 shape、格点单位与流化边界不明；
5. `traceless=False` 可能被误读为复现完整历史 OPE。

## GREEN

补全签名、轴/输出、格点归一化和职责边界后，同一评估器重读文档，
对上述五项全部给出 `PASS`。关键前向输出为：

```python
import numpy as np
from pyqcd.gauge import (
    clover_field_strength,
    clover_topological_charge,
    clover_topological_charge_density,
    clover_topological_charge_density_average,
    polyakov_loop,
    wilson_rectangle,
)

# Trivial SU(3) links on a tiny (Nt,Nz,Ny,Nx)=(2,2,2,2) lattice.
Nt, Nz, Ny, Nx, Nc = 2, 2, 2, 2, 3
gauge = np.zeros((Nt, Nz, Ny, Nx, 4, Nc, Nc), dtype=np.complex128)
gauge[...] = np.eye(Nc, dtype=gauge.dtype)
mu, nu = 3, 0  # t-x; direction labels are 0=x, 1=y, 2=z, 3=t

W = wilson_rectangle(gauge, 2, 3, mu, nu, average=False)
P = polyakov_loop(gauge, time_dir=3, average=False)  # complex
F = clover_field_strength(gauge, mu, nu, traceless=True)
F_legacy = clover_field_strength(gauge, mu, nu, traceless=False)
q = clover_topological_charge_density(gauge, traceless=True)
Q = clover_topological_charge(gauge, traceless=True)
q_mean = clover_topological_charge_density_average(
    gauge, traceless=True
)

assert W.shape == (Nt, Nz, Ny, Nx)
assert P.shape == (Nz, Ny, Nx)  # the closed time axis is removed
assert F.shape == F_legacy.shape == (Nt, Nz, Ny, Nx, Nc, Nc)
assert q.shape == (Nt, Nz, Ny, Nx)
assert np.allclose(W, 1.0) and np.allclose(P, 1.0)
assert np.allclose(F, 0.0) and np.allclose(F_legacy, 0.0)
assert np.allclose(q, 0.0) and np.isclose(Q, 0.0) and np.isclose(q_mean, 0.0)
```

评估器正确说明 `time_dir=3` 时 `P.shape=(Nz,Ny,Nx)`、`Q=sum_x q(x)`、
`q_mean=mean_x q(x)`，且 `F_legacy` 仅是历史 OPE 的 Clover 基元，不包含
Wilson 线、对偶场缩并和空间求和。

## 边界

这是技能文档的前向行为证据，不是生产代码、真实系综或 instanton
连续极限验证。实现边界由
`python -m pyqcd.testing._gauge_observables_contract` 单独验收。
