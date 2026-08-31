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
| 规范场 | `gauge (Nt,Nz,Ny,Nx,4,3,3)`，语义轴为 `(t,z,y,x,dir,color,color)`；链接/Lorentz 方向标签 `0=x,1=y,2=z,3=t` | 空间链接方向 `d=0,1,2` 映射到数组坐标轴 `3-d`；不要把方向标签、数组轴号和 PyQUDA 局部布局混用 |
| PyQUDA 局部场 | `[parity,Lt,Lz,Ly,Lx//2]`；含传播子时追加 `[spin_snk,spin_src,color_snk,color_src]` | 进入 `gatherLattice` 前确认局部 tzyx 维度 |
| γ 矩阵 | DeGrand–Rossi；`γ_5=γ_1γ_2γ_3γ_4`、`γ_0=I`、`C=γ_2γ_4` | 这是 `gamma(i)` 的矩阵编号，不是上述链接/Lorentz 方向标签；correlator、propagator 和参考推导必须使用同一基 |
| 时间边界 | 夸克时间方向反周期；偶数夸克复合态有效周期，奇数夸克态有效反周期 | 重子 backward 项的负宇称伙伴不能当普通激发态删除 |
| 动量 | 先写 Fourier 符号和存储方向，再生成相位 | `+p`/`-p`、源/汇顺序必须在元数据中可见 |
| 蒸馏 basis 相位 | `E (Nev,Lz,Ly,Lx,Nc)`；`p_basis=(pz,py,px)`；展平位置为 `z*Ly*Lx+y*Lx+x`；`u_p=exp[-i*2*pi*(pz*z/Lz+py*y/Ly+px*x/Lx)]` | `p_basis` 与顶点 Fourier 动量 `q_sink` 是两个独立字段；同 basis 的 VdV 相位抵消，VVV 有三条非共轭腿而带 `u_p^3`；传播子必须匹配源/汇 basis |
| 梯度流时间 | 流 API 的 `tau=t/a^2` 无量纲，物理流时间为 `t=a^2*tau`；工程参数 `tau=3` 即 `t=3a^2` | API 参数仍传 `3`，禁止写成 `3*(a*0.197)^2`；物理单位换算只用于结果解释 |
| 单位 | 格点整数、fm、GeV、GeV(^{-1}) 分开记录；Ioffe 时间 `z*Pz` 无量纲 | `quasi_tmd_pdf` 的 `z_grid` 与 `z_max` 都是 fm：先在 fm 上截断/细化，再除以 `fm_to_GeV` 进入 Fourier 相位；显式 `z_max=max(z_grid)` 必须与默认值等价 |
| 结果状态 | `实现存在` → `测试通过` → `方案闭合` → `真实数据验证` | 只能使用证据支持的最高状态 |

## 复数、符号与投影

1. 计算和落盘阶段保留复数 dtype；只有由宇称、时间反演、共轭关系或明确投影证明
   后，才取实部、虚部或偶奇组合。
   纯实输入可在确认虚部仅为数值零后用 `real_if_close` 恢复实 dtype；复数
   `h_R` 经过 Fourier、匹配与落盘时不得被 `dtype=float` 静默降维。
2. `C(t)` 的整体符号由算符、费米子置换和相位共同决定。先用独立极限或合成数据
   校验符号，再做有效质量或常数拟合；不得用“结果看起来正”替代推导。
3. `P^\pm=(1\pm\gamma_4)/2` 的选择必须与源汇时间方向、边界条件和目标宇称一起
   记录。投影后的 backward 贡献不可无条件折叠。

## 产物元数据最低集合

每个可复现产物至少记录：格点体积、`a_fm`（若已知）、通道与 Lorentz 指派、源/汇
时间、动量和 Fourier 符号、dtype/backend、随机种子、重采样方案、协方差/SVD 参数、
命令行和代码版本。含梯度流的产物应同时记录 `tau`、同义量 `t_over_a2`、`a_fm`
以及实际积分控制量 `eps`/`steps`，其中 `tau=t_over_a2`；TMD 产物还必须记录 `z`、
`b_perp`、`staple_length`、表示、`mu`、`zeta` 和匹配阶数。
蒸馏 vertex/perambulator 产物还必须记录 `lattice_shape=(Lz,Ly,Lx)`、展平顺序、
`p_basis_sink`、`p_basis_source`、独立的 `q_sink`、Fourier 符号、`Nev` 和组态 ID；
单独的 perambulator 目录名不能替代这些身份字段。

## 交接格式

向下游交接时固定给出四项：

1. **对象**：物理量、输入文件/数组及每一轴的含义；
2. **约定**：单位、边界、符号、投影、归一化和 dtype；
3. **证据**：可复现实验命令、数值误差或不变量；
4. **状态**：上述四种状态中的一个，以及未覆盖的限制。

发现轴序、单位或符号不一致时，先停在契约层，转 `pyqcd-physics-correlator`、
`pyqcd-propagator` 或 `pyqcd-tmd-algorithm` 定义正确对象后再继续计算。
