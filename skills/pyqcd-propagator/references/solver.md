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

# 显式 grid：product(grid_size) == MPI size，且每维可整除 latt_size。
core.init(grid_size, latt_size, backend="cupy",
          resource_path="/persistent/quda/tunecache")
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)
```

`grid_size` 的乘积必须等于 MPI rank 数；它划分格点，不等同于 Python 层的元任务调度。
PyQUDA 常用局部顺序为 `[parity,Lt,Lz,Ly,Lx//2]`，即 tzyx；含自旋/色指标的
`LatticePropagator.data` 为
`[parity,Lt,Lz,Ly,Lx//2,spin_sink,spin_source,color_sink,color_source]`。
`LatticeGauge.data` 的方向轴位于前部。进入 `gatherLattice` 前不要按 xyzt 猜测维度。

### 自动 grid

若没有通信/显存绑定的特殊要求，可让 PyQUDA 按 MPI size 和格点尺寸自动选择：

```python
core.init(None, latt_size, backend="cupy",
          resource_path="/persistent/quda/tunecache")
grid_size = core.getGridSize()
grid_coord = core.getGridCoord()
```

自动 grid 仍受 MPI size 和 `latt_size` 可整除性约束。运行日志必须记录实际返回的
`grid_size/grid_coord`；自动选择不等于可以不保存并行身份。

### MPS

某些调度器先把每个 rank 绑定到独立 GPU，并把本地 `CUDA_VISIBLE_DEVICES` 都映射成
同一可见编号 `0`。此时可用：

```python
core.init(None, latt_size, backend="cupy",
          resource_path="/persistent/quda/tunecache",
          enable_mps=True)
```

`enable_mps=True` 是允许这些已经隔离的 rank 使用同一个本地可见编号，不是让多个 rank
并发共享同一张真实 GPU 的替代方案。记录 MPS 环境、rank/device 映射和实际 GPU UUID；
若调度器未做设备隔离，先修正绑定而不是盲目打开 MPS。

### Tune cache 生命周期

`resource_path` 指向 QUDA autotuning cache 的持久目录：

1. 首次遇到新的 lattice/grid/solver 组合时，QUDA 会 tune，耗时可能显著高于正式求逆。
2. 后续相同配置从 cache 加载；不要把 cache 放在临时目录或每个 rank 私有的临时路径。
3. 所有相关 job/rank 对同一 cache 目录可读可写；运行中不要删除或替换 cache。
4. QUDA/PyQUDA 版本、grid/backend/solver 参数变化后，使用新目录或确认版本检查后重新
   tune，不能复用来历不明的二进制 cache。
5. 产物元数据记录 `resource_path`、实际 grid、QUDA/PyQUDA 版本和是否发生首次 tuning。

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
`useGauge` 上下文内完成求逆，退出后再做 MPI 汇总或切换 raw 场。不要嵌套多个
`useGauge`/`gauge.use()` 上下文；若 helper 内部已经打开 gauge context，先退出外层
上下文再调用它。

## 源与求逆

简单 point/wall/volume source 可直接由 `core.invert` 构造并求逆；每种 source 的相位
必须独立传入：

```python
from pyquda_utils import phase_v2, core

phase_pt = phase_v2.MomentumPhase(latt_info).getPhase(
    [kx, ky, kz], [x0, y0, z0]
)
phase_wall = phase_v2.MomentumPhase(latt_info).getPhase(
    [kx, ky, kz], [x0, y0, z0]
)
phase_vol = phase_v2.MomentumPhase(latt_info).getPhase(
    [kx, ky, kz, kt], [x0, y0, z0, t0]
)
with dirac.useGauge(gauge_stout):
    prop_pt = core.invert(
        dirac, "point", [x0, y0, z0, t0], phase_pt.data
    )
    prop_wall = core.invert(dirac, "wall", t0, phase_wall.data)
    prop_vol = core.invert(dirac, "volume", None, phase_vol.data)
```

`"wall"` 的 `t0` 必须是整数时间片；`"volume"` 的位置参数为 `None`。wall 的两条腿若
需要拆分动量，必须分别构造/求解 momentum wall source，不能只看 sink 相位。

Gaussian/Wuppertal source 和“已有传播子形 source”使用 `source.propagator` 后交给
`core.invertPropagator`：

```python
from pyquda_utils import source

src_pt = source.propagator(
    latt_info, "point", [x0, y0, z0, t0]
)
# gaussianSmear 内部会打开 gauge_smear 的 gauge context；
# 因此先退出 solver context，再调用它。
src_gauss = source.gaussianSmear(
    src_pt, gauge_smear, rho, n_steps
)
with dirac.useGauge(gauge_solver):
    prop_gauss = core.invertPropagator(dirac, src_gauss)
```

方向常量经 `pyquda_utils.core` 再导出，使用
`from pyquda_utils.core import X, Y, Z, T`；不要写
`from pyquda_utils import X, Y, Z, T`。非局域 Wilson line 的 `covDev` 使用
raw/smear 定义所指定的独立场对象，并放在单独 context 中，禁止与 solver context
嵌套。

### 已有传播子形 source

`core.invertPropagator(dirac, source_propag)` 接收完整的 `LatticePropagator` 作为
右端项：逐 source color 列作为 RHS，返回新的 `LatticePropagator`。它不是“继续求解”
已有传播子，也不会自动 dagger、选取时间片或改写 metadata。调用前必须记录
`source_propag` 的来源、轴序、时间层和归一化。

顺序源可选：

```python
src_seq = source.sequential12(B, t_seq)
# 或 core.invertSequential(dirac, B_propagator, t_seq)
with dirac.useGauge(gauge_solver):
    prop_seq = core.invertPropagator(dirac, src_seq)
```

`sequential12` 只选取 sink time slice；`gamma5 B† gamma5`、动量相位和第二次 dagger
由上游 contraction contract 指定，不能从 helper 名称推断。

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

## 上下文与源清单边界

| 风险 | 强制处理 |
|---|---|
| 错误 import | 只 import 当前版本实际导出的模块；方向常量使用 `from pyquda_utils.core import X, Y, Z, T` |
| 嵌套 gauge contexts | `gaussianSmear` 或 helper 已打开 context 时，先退出 solver context；covDev 另开 raw-field context |
| wall/volume 标签混淆 | 保存 source type、`t0`/位置、每条腿动量、phase 数组和源数 |
| 把已有 propagator source 当顺序源 | 确认 source color/RHS 语义；顺序源明确使用 `sequential12` 或 `invertSequential` |
| tune cache 丢失或跨版本复用 | 使用持久共享目录；版本变化时新 cache/retune |

## 验收

记录 solver residual、迭代数、局部/全局 shape、dtype/backend、源坐标、场版本和文件
round-trip，并记录实际 `grid_size`、MPS/device 映射、`resource_path` 和 wall/volume 的
phase 来源。换 PyQUDA 或 backend 后重新做小格点对照；旧产物能读不代表新求逆数值等价。
