---
name: pyqcd-gauge
description: |
  纯规范观测量技能：直接从规范链接计算 Wilson 圈、Polyakov 圈、拓扑荷、静态势、
  Wilson flow、链接涂抹等，不涉及费米子传播子/Dirac 求解器/求逆。兼顾物理推理
  （算什么、为何）与代码生成（怎么算）。触发于："算 Wilson 圈"、"Polyakov 圈"、
  "拓扑荷"、"纯规范观测量"、"静态势提取"、"Wilson flow"、"链接涂抹"，或任务
  明确只涉及规范链接而无传播子。
metadata:
  openclaw:
    emoji: 🔷
---

# pyqcd-gauge — 纯规范观测量

## 目的与边界

把纯规范观测量规格翻译为可执行代码：读规范组态 → 逐组态出结果。
只产**数据生产脚本**。与强子关联器的区别：仅 I/O 规范链接、计算快（all-to-all）、
无求逆；涂抹作用于链接（Stout/APE/HYP）而非源。

## 物理推理要点

### Wilson 圈

$$W_{\mu\nu}(R,T) = \frac{1}{N_c}\mathrm{Re\,Tr}\,\mathcal{P}\Big[
\prod_{i=0}^{R-1} U_\mu(x{+}i\hat\mu) \prod_{j=0}^{T-1} U_\nu(x{+}R\hat\mu{+}j\hat\nu)
\prod_{i=0}^{R-1} U_\mu^\dagger(\cdots) \prod_{j=0}^{T-1} U_\nu^\dagger(\cdots)\Big]$$

路径：μ 前进 R → ν 前进 T → μ 退回 R → ν 退回 T。

### 静态势

$$V(R)=-\lim_{T\to\infty}\tfrac1T\log\langle W(R,T)\rangle,\qquad
V(R)=V_0+\frac{\alpha}{R}+\sigma R \quad(\text{禁闭}\Rightarrow\sigma>0).$$

### 平面选择与涂抹参数

| 平面 | 用途 |
|---|---|
| XT | 标准势 |
| YT/ZT | 转动对称性检验 |
| XY | 空间弦张力、胶球 |

| 涂抹 | 参数 | 备注 |
|---|---|---|
| Stout | ρ=0.08–0.12, 1–3 iter | 解析可微；PyQCD 系综约定 nstep=20, ρ=0.12 |
| APE | α=0.5–0.75, 1–5 iter | 简单 |
| HYP | (α₁,α₂,α₃) | 强 UV 压低 |

### Wilson flow（衔接 PyQCD 梯度流）

迭代平滑；按流时间算 $E(t)$、$Q(t)$。PyQCD 实现：
`pyqcd/renorm/_gradient_flow.py`——Wilson flow（Lüscher 2010）、RK3 积分、
τ=3a² 方案（Monahan–Orginos 2017 / NieMiera 2025），是胶子 TMD 重整化链的
第一步（下游见 pyqcd-tmd-chain）。stout 涂抹独立实现：`pyqcd/smear/_stout.py`。

## 工作流程（PyQUDA 实现）

### Step 1: 初始化与读规范

```python
from pyquda_utils import core, io
core.init([1,1,1,4], [24,24,24,72], resource_path="./tunecache")  # grid 可 [1,1,1,1]
gauge = io.readChromaQIOGauge(cfg_path); gauge.toDevice()
gauge.stoutSmear(n_step=1, rho=0.1, n_dim=4)                      # 可选
```

### Step 2: Wilson 圈——gauge.loop + 方向路径

```python
from pyquda_utils.core import X, Y, Z, T
path_XT = [X]*R + [T]*Tlen + [-X]*R + [-T]*Tlen      # YT/ZT 同理
res = gauge.loop([[path_XT],[path_YT],[path_ZT],[path_XT]], [1,1,1,0])
```

⚠️ **gauge.loop 必须恰好 4 个外层组**——不足时以权重 0 补齐。

### Step 3: 逐点 ReTr 提取与 MPI 归约

```python
U = res[i].getHost().reshape(-1, Nc, Nc)      # GPU→CPU 后必须 reshape 再 trace
tr_real = np.trace(U, axis1=-2, axis2=-1).real
global_sum = core.gatherLattice(re_tr_field, [-1,-1,-1,-1])   # 不用 mpi4py Allreduce
if core.getMPIRank() == 0:
    wl_value = float(global_sum.sum()) / (total_sites * Nc)
```

非 MPI 运行（单 GPU/CPU numpy）直接对全格点平均。

### Step 4: 保存 HDF5

rank 0 写出；数据集附 R/T attrs（h5py 示例见上游实践；PyQCD 管线产物统一用
`pyqcd/tools._io.save_tensor_h5`，见 pyqcd-infra）。

### Step 5: 其他观测量

- **Polyakov 圈**：`gauge.polyakov()`（如可用）；同样 getHost→trace→归约。
- **拓扑荷**：plaquette 拼 Clover 叶 $F_{\mu\nu}$，
  $Q=\frac{1}{32\pi^2}\sum_x\epsilon_{\mu\nu\rho\sigma}\mathrm{Tr}[F_{\mu\nu}F_{\rho\sigma}]$。
- Wilson flow 全例见 `reference/wilson_loop.md` 与 `pyqcd/renorm/_gradient_flow.py`。

## 常见问题

| 问题 | 处理 |
|---|---|
| gauge.loop IndexError | 恰好传 4 组（不足补权重 0 组） |
| LatticeLink 被当标量 | 恒 getHost()→reshape→trace |
| gatherLattice 形状不匹配 | 输入须匹配本地 (2,Lt,Lz,Ly,Lx//2) |
| 圈噪声大 | 先涂抹再算圈；1 步 stout ρ=0.1 起步 |
| GPU OOM | 加卡或小格点回 CPU 后端 |

## 与其他技能配合

- 涉及传播子/求逆 → `pyqcd-propagator`；梯度流进入 TMD 重整化链 → `pyqcd-tmd-chain`；
- 流/涂抹产物分析 → `pyqcd-analysis`；产物 IO 约定 → `pyqcd-infra`。
