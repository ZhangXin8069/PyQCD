---
name: pyqcd-conventions
description: |
  Use when a PyQCD task needs shared conventions for gamma matrices, tensor axes,
  lattice or physical units, temporal boundary conditions, signs, complex
  projections, metadata, or validation status; use the domain skill directly for
  an operator, solver, fit, or pipeline procedure.
metadata:
  openclaw:
    emoji: 📐
---

# pyqcd-conventions — 物理与数据共享契约

## 目的与边界

本技能是 PyQCD 各领域技能共用的“先验契约”：统一记号、轴序、单位、边界条件、
复数处理和证据状态。它不负责生成关联函数、求逆、拟合或运行管线；这些工作转交
对应领域技能。

## 必须先固定的约定

| 对象 | PyQCD 约定 | 使用纪律 |
|---|---|---|
| 规范场 | `gauge (Nt,Nz,Ny,Nx,4,3,3)` | 明确时间、空间和方向轴；不要把 PyQCD 轴序与 PyQUDA 局部布局混用 |
| PyQUDA 局部场 | `[parity,Lt,Lz,Ly,Lx//2]`；含传播子时追加 `[spin_snk,spin_src,color_snk,color_src]` | 进入 `gatherLattice` 前确认局部 tzyx 维度 |
| γ 矩阵 | DeGrand–Rossi；`γ_5=γ_1γ_2γ_3γ_4`、`γ_0=I`、`C=γ_2γ_4` | correlator、propagator 和参考推导必须使用同一基 |
| 时间边界 | 夸克时间方向反周期；偶数夸克复合态有效周期，奇数夸克态有效反周期 | 重子 backward 项的负宇称伙伴不能当普通激发态删除 |
| 动量 | 先写 Fourier 符号和存储方向，再生成相位 | `+p`/`-p`、源/汇顺序必须在元数据中可见 |
| 单位 | 格点整数、fm、GeV、GeV(^{-1}) 分开记录；Ioffe 时间 `z*Pz` 无量纲 | 低层 `z/b/tau` 的单位不能直接套到 `quasi_tmd_pdf` |
| 结果状态 | `实现存在` → `测试通过` → `方案闭合` → `真实数据验证` | 只能使用证据支持的最高状态 |

## 复数、符号与投影

1. 计算和落盘阶段保留复数 dtype；只有由宇称、时间反演、共轭关系或明确投影证明
   后，才取实部、虚部或偶奇组合。
2. `C(t)` 的整体符号由算符、费米子置换和相位共同决定。先用独立极限或合成数据
   校验符号，再做有效质量或常数拟合；不得用“结果看起来正”替代推导。
3. `P^\pm=(1\pm\gamma_4)/2` 的选择必须与源汇时间方向、边界条件和目标宇称一起
   记录。投影后的 backward 贡献不可无条件折叠。

## 产物元数据最低集合

每个可复现产物至少记录：格点体积、`a_fm`（若已知）、通道与 Lorentz 指派、源/汇
时间、动量和 Fourier 符号、dtype/backend、随机种子、重采样方案、协方差/SVD 参数、
命令行和代码版本。TMD 产物还必须记录 `tau`、`z`、`b_perp`、`staple_length`、
表示、`mu`、`zeta` 和匹配阶数。

## 交接格式

向下游交接时固定给出四项：

1. **对象**：物理量、输入文件/数组及每一轴的含义；
2. **约定**：单位、边界、符号、投影、归一化和 dtype；
3. **证据**：可复现实验命令、数值误差或不变量；
4. **状态**：上述四种状态中的一个，以及未覆盖的限制。

发现轴序、单位或符号不一致时，先停在契约层，转 `pyqcd-physics-correlator`、
`pyqcd-propagator` 或 `pyqcd-tmd-algorithm` 定义正确对象后再继续计算。
