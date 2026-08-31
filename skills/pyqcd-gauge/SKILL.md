---
name: pyqcd-gauge
description: |
  Use when a task involves only gauge links or pure-gauge observables such as Wilson
  loops, Polyakov loops, static potentials, topological charge, Wilson flow, or link
  smearing; route any Dirac solve or fermion correlator to pyqcd-propagator or
  pyqcd-physics-correlator.
metadata:
  openclaw:
    emoji: 🔷
---

# pyqcd-gauge — 纯规范对象

## 目的与边界

本技能负责从规范链接生成纯规范数据：路径、迹、流化、拓扑和涂抹。它不处理费米子
传播子、Wick 缩并或强子拟合；含外态核子或胶子 TMD 的任务分别转交
`pyqcd-propagator`、`pyqcd-physics-correlator` 和 `pyqcd-tmd-algorithm`。

按需读取：

| 任务 | Reference |
|---|---|
| Wilson/Polyakov 圈、静态势、拓扑荷和 MPI 归约 | [`references/observables.md`](references/observables.md) |
| Wilson flow、Clover 场强、Stout/APE/HYP 及极限检查 | [`references/flow-and-smearing.md`](references/flow-and-smearing.md) |

历史 Wilson 圈推导、Creutz ratio 和 PyQUDA 路径例子见
[`reference/wilson_loop.md`](reference/wilson_loop.md)；它是补充例文，不替代当前 API
签名核验。

## 先固定的物理契约

| 对象 | 必须明确 |
|---|---|
| 链接与路径 | 方向轴、前进/回程顺序、格点周期绕回、表示和迹归一化 |
| 观测量 | `R,T`、平面、起点平均、是否取 `Re Tr`、Polyakov 中心复相位、Clover 是否去迹及拓扑定义 |
| 流/涂抹 | raw 输入、无量纲流时间 $\tau=t/a^2$ 或迭代数、步长、参数、是否需要保留原场 |
| 输出 | 组态 ID、shape/轴、单位、dtype/backend、HDF5 属性和归约方式 |

量纲检查：Wilson 圈本身无量纲；由 $-\log W/T$ 得到的势仍是格点单位，换成 GeV
必须显式使用格距。拓扑荷的整数性是离散化和流时间依赖的诊断，不可把近似整数
直接当作定义。

## 推荐工作流

1. **选择对象**：确认任务没有传播子；固定路径、平面、参数和归一化。
2. **读入 raw 场**：初始化 backend/MPI、读规范场、检查 shape 与近似 SU(3) 幺正性；
   PyQCD 的 HYP/Stout 所有参数路径均返回独立结果且不原地改写输入，包括零参数退化
   路径；无需为防止 smear 输出回写 raw 而预先 copy。若转入 PyQUDA 或其他可能原地
   改写的生产 API，按其 API 契约显式 copy。
3. **计算局部对象**：优先使用已验证的 `pyqcd.gauge` 公开入口；逐起点/方向
   生成路径或 plaquette，保持颜色矩阵直到完成迹；标准纯规范 Clover 与拓扑量
   默认 `traceless=True`，只有复现历史 OPE 离散量时才显式选 `False`。
   公开观测量只接受 `float32/float64/complex64/complex128`；已存在的 Torch Tensor
   以其输入 device/dtype 为准，不得被全局默认 device 搬运。逐入口的实/复输出映射见
   [`observables.md`](references/observables.md)。
   流和涂抹分别记录中间场，不把 stout 迭代当作 Wilson flow 时间。
4. **全局归约**：按局部布局调用已核实的 `gatherLattice`，上下文外完成归约；root
   负责写文件，保存路径和参数元数据。
5. **物理验收**：检查 `W(R,0)=1` 等退化极限、反向路径共轭、旋转对称性、SU(3)
   误差和流步长稳定性，再把流化场交给 `pyqcd-tmd-chain`。

## 不能静默改变的约定

- 路径乘积按从左到右的物理行走顺序书写；退回段使用明确的逆链接，不能只翻转
  数组顺序。
- `Re Tr/N_c`、全体积平均和单个起点值是不同估计量，输出中必须区分。
- Polyakov 圈一般为复数；不得默认取实部而丢弃 SU(3) 中心相位。拓扑总荷
  `sum_x q(x)` 与体积平均 `mean_x q(x)` 也必须用不同入口。
- `E(τ)` 与 `τ²E(τ)` 的单调性取决于定义；先写公式和离散化，再判断趋势。
- 流化只提供 UV 平滑，不自动等价于 TMD 的 rapidity subtraction 或完整重整化。

## 常见错误

| 现象 | 处理 |
|---|---|
| `gauge.loop` 参数数量报错 | 按当前 API 检查外层路径组；需要时以零权重补齐，但不伪造观测量 |
| 把 GPU 链接当标量 | 先转 host、恢复矩阵 shape，再做颜色迹 |
| MPI 归约形状不符 | 记录本地 parity/tzyx 布局，按 `gatherLattice` 契约传维度 |
| 涂抹后无法复核原场 | HYP/Stout 输出与输入独立；分别持有并标记 raw/smeared，不要误称 API 原地修改 |
| 流能量趋势“错误” | 核对 `E`、`τ²E`、流步长和定义，不能凭经验翻符号 |

## 交接

输出规范对象的公式、路径/离散参数、shape、归约证据和状态。TMD 场强与 staple 几何
的阶段导航交给 `pyqcd-tmd-chain`，具体算符/缓存/物理验收交给
`pyqcd-tmd-algorithm`；批量和文件格式交给 `pyqcd-pipeline` / `pyqcd-infra`；
数据统计交给 `pyqcd-statistics`，报告交给 `pyqcd-docs`。
