# PyQUDA 求解 reference

## 适用范围

任务已固定 flavor、源和目标传播子，需要核对 PyQUDA 的 MPI/grid、局部数组布局、
Dirac 构造、源接口或保存方式时读取本文件。具体版本的函数签名仍以当前环境源码和
`--help` 为准。

## MPI、格点与布局

```python
from pyquda_utils import core

comm = core.getMPIComm()
rank = core.getMPIRank()
size = core.getMPISize()
core.init(grid_size, latt_size, backend="cupy", resource_path="./tunecache")
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)
```

`grid_size` 的乘积必须等于 MPI rank 数；它划分格点，不等同于 Python 层的元任务调度。
PyQUDA 常用局部顺序为 `[parity,Lt,Lz,Ly,Lx//2]`，即 tzyx；含自旋/色指标的
`LatticePropagator.data` 为
`[parity,Lt,Lz,Ly,Lx//2,spin_sink,spin_source,color_sink,color_source]`。
`LatticeGauge.data` 的方向轴位于前部。进入 `gatherLattice` 前不要按 xyzt 猜测维度。

## 规范场与 Dirac 算子

```python
from pyquda_utils import io

gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(n_step, rho, n_dim)

dirac = core.getWilson(latt_info, mass, tol, maxiter, multigrid)
dirac_c = core.getClover(
    latt_info, mass, tol, maxiter, xi_0, csw_t, csw_r, multigrid
)
```

各向同性时确认 `csw_t=csw_r`；fermion anisotropy 和 gauge anisotropy 不是同一参数。
轻夸克优先使用已验证的 multigrid；`multigrid=None` 只能作为明确的 BiCGStab 基线。
`useGauge` 上下文内完成求逆，退出后再做 MPI 汇总或切换 raw 场。

## 源与求逆

```python
from pyquda_utils import phase_v2, core

phase = phase_v2.MomentumPhase(latt_info).getPhase([kx, ky, kz], [x0, y0, z0])
with dirac.useGauge(gauge_stout):
    prop = core.invert(dirac, "point", [x0, y0, z0, t0], phase.data)
```

`"wall"`、`"volume"` 和已存传播子源按当前 PyQUDA 版本的 `core.invert`/
`core.invertPropagator` 签名调用；高斯 smear 的源由 `source.propagator`、
`source.gaussianSmear` 构造。每个源时刻和汇投影都成为输出标签，不能只写一个覆盖文件。

## 保存与归约

传播子可用 `prop.save(..., use_fp32=False)` 或 `prop.saveH5(..., check=True)`；统一
管线优先转 `pyqcd.tools._io.save_tensor_h5`。局部量归约示意：

```python
local = contract("wtzyxCBba,wtzyxCBba->t", prop.data.conj(), prop.data)
global_t = core.gatherLattice(local.get(), [0, -1, -1, -1])
if rank == 0:
    write_result(global_t)
```

`gatherLattice` 的维度列表是 tzyx 语义；非 root 结果可能为 `None`。不要调用未经
当前版本核实的 `core.allreduce`，也不要在 `useGauge` 上下文中 gather。

## 验收

记录 solver residual、迭代数、局部/全局 shape、dtype/backend、源坐标、场版本和文件
round-trip。换 PyQUDA 或 backend 后重新做小格点对照；旧产物能读不代表新求逆数值等价。
