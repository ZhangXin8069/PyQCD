---
name: pyqcd-propagator
description: |
  PyQUDA 传播子求解技能：生成调用 PyQUDA 的 Python 代码，在格点规范组态上解夸克
  传播子。覆盖：组态加载、夸克参数设置（Wilson/Clover、质量/kappa、clover 系数、
  链接涂抹）、multigrid 求解器、源构造（点源/高斯 smear/APE/HYP/stout）、
  传播子求逆与残差验证、顺序源三点函数全链、协变位移（Wilson 线/nonlocal 算符）。
  触发于："算传播子"、"求解逆"、"调 PyQUDA"、"解 Dirac 方程"，或
  pyqcd-physics-correlator 已给出传播子需求清单。纯规范观测量用 pyqcd-gauge。
metadata:
  openclaw:
    emoji: 🌀
---

# pyqcd-propagator — PyQUDA 传播子求解

## 目的与边界

把传播子规格翻译为可执行 PyQUDA 代码：读组态 → 解传播子 → 存数据文件。
只产**数据生产脚本**，不做分析代码。重子 3pt 完整代码样例见
`reference/baryon_3pt_code.md`。

## 代码风格与执行模型

- **扁平自包含脚本**：单文件、物理参数（质量/clover 系数/格点维度/路径/源位置）
  全部硬编码在顶部、自上而下可读可核验——格点 QCD 计算脚本供合作者交叉检验，
  不按可复用软件维护。argparse 仅用于区分独立并行作业的参数（典型为组态 ID）。
- **必须 MPI 启动**：`mpirun -np 4 python script.py`（SLURM 用 srun）；
  单 GPU 也用 `mpirun -np 1`。MPI rank 数 = grid_size 各维乘积，默认每 rank 一 GPU。
  脚本头部注明启动命令注释。
- **重计算无交互监控**：求逆常达分钟—小时级；不加进度条/交互提示，
  QUDA 自行打印迭代数与残差到 stdout，重定向日志即可。

## 关键约定

### MPI / Grid / Device

```python
from pyquda_utils import core
comm, rank, size = core.getMPIComm(), core.getMPIRank(), core.getMPISize()
grid_size, grid_coord = core.getGridSize(), core.getGridCoord()
```

⚠️ `pyquda_utils.core` **没有** `allreduce()`——写 `core.allreduce(...)` 会 AttributeError
崩溃；归约一律 `gatherLattice`。Grid 划分格点（类笛卡尔通信器）；未指定 grid 时
PyQUDA 自动选通信最小的划分。绑定脚本分配 GPU 时初始化需 `enable_mps=True`。

### 场布局与 γ 矩阵

偶奇预条件布局 `[2, Lt, Lz, Ly, Lx//2]`（tzyx 序！多数接口处为 xyzt）。
带自旋/色指标时恒为 `[snk, src]` 序：
`LatticePropagator.data` = `[2, Lt, Lz, Ly, Lx//2, Ns, Ns, Nc, Nc]`；
`LatticeGauge.data` = `[4, 2, Lt, Lz, Ly, Lx//2, Nc, Nc]`。
γ 矩阵 DeGrand-Rossi 基（同 pyqcd-physics-correlator），位域编码：
`gamma.gamma(1/2/4/8)` ↔ γ₁/γ₂/γ₃/γ₄，按位或表乘积，`gamma.gamma(15)=γ₅`。

### 数组后端

`backend="cupy"/"torch"/"dpnp"/"numpy"`；磁盘加载的场恒为 CPU numpy，
须 `toDevice()`。传输：`array.arrayAsArray(cpu, backend=b)` /
`array.arrayAsNumpy(gpu, backend=b)`。
⚠️ `arrayAsNumpy` 近版**必须带 backend= 实参**否则运行时报错。

## 工作流程

### Step 1: 组态元数据

从系综注册表（YAML/JSON）取：组态路径与格式（ILDG/QIO/milc）、格点维度、
规范作用量参数（β/tadpole）。

### Step 2: 初始化

```python
from pyquda_utils import core
core.init(grid_size, latt_size, backend="cupy", resource_path="/path/to/tunecache")
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)
```

先 init 再建 LatticeInfo；`t_boundary=-1` 即时间反周期；anisotropy 用 xi_0/nu。
resource_path 指向持久目录保存 autotune 缓存；QUDA 版本变更后删缓存重调优。

### Step 3: 读规范

```python
from pyquda_utils import io
gauge = io.readChromaQIOGauge(cfg_path)   # 其他格式查 readMILCGauge 等
```

### Step 4: 夸克与求解器参数

```python
dirac = core.getWilson(latt_info, mass, tol, maxiter, multigrid)          # Wilson
dirac = core.getClover(latt_info, mass, tol, maxiter, xi_0, csw_t, csw_r, multigrid)  # Clover
```

multigrid=None 时 BiCGStab；轻夸克强烈建议 multigrid（临界慢化）。
各向同性晶格 csw_t=csw_r=csw；fermion anisotropy ξ=xi_0/nu 与 gauge anisotropy 区分。
涂抹并装载进 QUDA：

```python
gauge.stoutSmear(1, rho, 4)
with dirac.useGauge(gauge):   # 上下文管理器内才可求逆；嵌套时最内层生效
    ...
```

useGauge 触发 CPU→GPU 传输与重排——把所有需要规范场的操作放进同一上下文。

### Step 5: 构造源并求解

```python
from pyquda_utils import core, phase_v2, source
phase = phase_v2.MomentumPhase(latt_info).getPhase([kx,ky,kz],[x0,y0,z0])
with dirac.useGauge(gauge):
    propag_pt = core.invert(dirac, "point",  [x0,y0,z0,t0], phase.data)
    propag_wl = core.invert(dirac, "wall",   t0,            phase.data)
    propag_vl = core.invert(dirac, "volume", None,          phase.data)
```

已存传播子作源（高斯 smear）：`source.propagator` → `source.gaussianSmear(src,
gauge, rho, n_steps)` → `core.invertPropagator(dirac, source_sh)`。
顺序源：`core.invertSequential(dirac, prop_sh, t_seq)`。
多源时刻：外层 `for t_src in t_srcs:` 循环，结果带 t_src 标签落盘。

### Step 6: 保存（可选）

```python
propag_sh.save("prop.npy", use_fp32=False)
propag_sh.saveH5("prop.h5", tag, annotation=..., check=True, use_fp32=False)
```

### Step 7: 收缩到关联函数

einsum 结构由物理推导给出（见 pyqcd-physics-correlator 与 reference 例文）；
若工作环境提供 generate_einsum 类工具则用其产出、禁止手写。零动量介子 2pt：

```python
C_t_local = contract("wtzyxCBba, wtzyxCBba -> t", prop_l.data.conj(), prop_l.data)
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])  # [0,...]=t 维归约, -1=空间全归约
if core.getMPIRank() == 0:  # 仅根 rank 写出
    ...
```

gatherLattice 第二参为 tzyx 序维度列表；非根 rank 返回 None。

### Step 8: 输出 txt

仅 rank 0；列数随观测量：局域 2pt `t, Re, Im`；非局域 2pt `z_sep, t, Re, Im`；
3pt `t_seq, insertion, Re, Im`——所有离散标签列置于 Re/Im 之前。

## 三点函数顺序源管线

```python
# ① 汇块 B（einsum 来自推导/工具）
B = core.LatticePropagator(latt_info); B.data = ...contract(...)...
# ② 第一次 dagger：γ5 B† γ5
B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)
# ③ 顺序源 + 求逆
src_seq = source.sequential12(B, t_seq)
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)
# ④ 第二次 dagger
prop_seq_dag.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, prop_seq.data.conj(), G5)
# ⑤ 终收缩 + gather（use() 上下文之外）
three_pt_site = contract("wtzyxijba, jk, wtzyxkiab -> wtzyx", prop_seq_dag.data, Γ_cur, prop_current.data)
C3_t = core.gatherLattice(contract("wtzyx -> t", three_pt_site).get(), [0,-1,-1,-1])
```

## 协变位移（Wilson 线 / nonlocal 算符）

非局域算符 $\bar q(x)\Gamma W(x,x{+}z)q(x{+}z)$ 需 covDev 实现位移：

**双份规范场纪律**：
```python
gauge_raw = io.readChromaQIOGauge(cfg_path); gauge_raw.toDevice()
gauge_stout = gauge_raw.copy()            # smear 前必须 copy（stoutSmear 原地改）
gauge_stout.stoutSmear(n_step, rho, ndim)
```
- `gauge_stout` 只用于 Dirac 求逆；`gauge_raw` 只用于 covDev（Wilson 线）。
- 两类 use() 上下文互斥：先关 `useGauge` 再开 `gauge_raw.use()`。

**位移循环**（covDev 作用于单个 spin-color 分量，须 4×3 全循环；每个 z 距离从
新 copy 出发；一次 covDev 移一步）：

```python
from pyquda_utils.core import X, Y, Z, T
C_loc = cp.zeros((zmax + 1, latt_info.Lt), dtype=cp.complex128)   # 二维！(z+1, Lt_local)
with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_l.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = dirac_shift.covDev(prop_shift.getFermion(spin,color), Z)
                    prop_shift.setFermion(tmp, spin, color)
        C_loc[zsep] = contract("wtzyxjiba, wtzyxjiba -> t", prop_l.data.conj(), prop_shift.data)
        # ↑ "-> t" 直接缩掉宇称/自旋/色/空间；禁止 "-> wtzyx"
# gather 在 use() 关闭之后（MPI + QUDA 规范上下文 = 死锁风险）
for zsep in range(zmax + 1):
    t_global = core.gatherLattice(C_loc[zsep].get(), [0, -1, -1, -1])
    if core.getMPIRank() == 0: C_full[zsep, :] = t_global
```

⚠️ C_loc 必须二维 `(zmax+1, Lt_local)`：einsum 已缩掉 parity 维；
gatherLattice 需要时再 reshape 补回。

### 高频错误对照表

| 错误 | 正确做法 |
|---|---|
| smear 前忘 copy | 恒 `gauge_raw.copy()` 再 stoutSmear |
| z 循环复用 prop_shift | 每个 z 从 `prop_l.copy()` 重来 |
| covDev 整个传播子 | 逐 spin(4)×color(3) 循环 |
| 用 smeared 规范做位移 | 用 `gauge_raw` |
| gatherLattice 在 use() 内 | 移到 use() 块之后 |
| 求逆上下文与 covDev 上下文重叠 | 先关 useGauge 再开 raw.use() |
| C_loc 带 parity 维三维 | 二维；`-> t` einsum 已缩 parity |

## 常见问题

| 问题 | 处理 |
|---|---|
| GPU OOM | 加 MPI rank/GPU；或 backend="numpy" 回 CPU |
| 求解器不收敛 | 查组态完整性；收紧中间容差重启 |

## 与其他技能配合

- 传播子需求清单来源 → `pyqcd-physics-correlator`；拟合模板 → `pyqcd-physics-spectrum`；
- 下游分析 → `pyqcd-analysis`；纯规范可观测量 → `pyqcd-gauge`；
- PyQCD 侧顶点/蒸馏产物消费 → `pyqcd-pipeline`、`pyqcd-infra`（h5/VdV/VVV IO）。
