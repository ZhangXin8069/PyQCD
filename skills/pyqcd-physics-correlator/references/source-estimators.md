# Source estimators 与非局域传播子 reference

## 适用范围

本文件把“完整源求和不可执行”的关联函数落到 point、wall、volume/stochastic 或
shifted-propagator 估计量。它只定义估计量和交接字段；PyQUDA 调用、上下文和 I/O
见 `pyqcd-propagator`，Fourier/γ/轴序总约定见 `pyqcd-conventions`。

## Point source

在固定位置 \(y_0=(x_0,t_0)\) 放源，只求该色/旋量列的传播子：

\[
S_{{\rm pt},y_0}(x,t)=S(x,t;y_0,t_0).
\]

- 成本最低，适合调试、符号核对和单源首看。
- 源相位可写成 \(e^{i\vec p_1\cdot\vec y_0}\)。对局部、动量守恒的两点函数，这个
  全局相位在源汇相消时可忽略；一旦两腿动量分裂、非局域算符或三点流插入存在，
  必须保留并追踪相位。
- 多个源位置或源时刻是独立测量；先逐组态移到共同源时刻再平均。

## Wall source 与动量分裂

固定时间片 \(t_0\) 的空间 wall source 定义为

\[
S_{{\rm wall},\vec p_1}(x,t)=
\sum_{\vec y} e^{i\vec p_1\cdot\vec y}
S(x,t;\vec y,t_0).
\]

原先两个传播子各带源动量 \(\vec p_1,\vec p_2\)，但 wall source 求和后不能从
\(\vec p_1+\vec p_2\) 反推出各腿动量，因此必须分别求解并用 sink 投影保留总动量
\(\vec p=\vec p_1+\vec p_2\)。在 \(\hat z\) 方向通常采用平衡分裂：

| \(p_z\) | \(\vec p_1\) | \(\vec p_2\) |
|---:|---:|---:|
| 1 | 1 | 0 |
| 2 | 1 | 1 |
| 3 | 2 | 1 |
| 4 | 2 | 2 |

源相位 \(e^{i\vec p_i\cdot\vec y}\) 与 sink 投影 \(e^{-i\vec p\cdot\vec x}\)
来自不同的 Fourier 变换；dagger/conjugation 若作用于其中一条腿，会改变该腿的
有效符号。必须从完整关联函数逐项推导，不能把 sink 动量相位直接复制到 wall source。

## Volume/stochastic source

四点 volume source 可写成

\[
S_{{\rm vol},\vec p,E}(x,t)=
\sum_{\vec y,\tau} e^{i(\vec p\cdot\vec y+E\tau)}
S(x,t;\vec y,\tau).
\]

两点函数通常不直接使用确定性全 volume source。需要 all-to-all 时，用逐 site/spin/color
的 \(Z_2\) 或 \(Z_4\) 噪声源：

\[
\widehat S(x;y)=\eta(y)S(x;y),\qquad
\mathbb E[\eta(y)\eta^*(y')]=\delta_{yy'},
\]

并在收缩中正确 conjugate 噪声。记录噪声分布、实际样本数、seed、归一化和每个
额外噪声源；单噪声抽取不是严格 all-to-all。用独立实现的恒等/零信号测试检查偏差。

## Shifted propagator 与 Wilson line

对非局域双线性

\[
\bar q(x)\Gamma W(x,x+z)q(x+z),
\]

可把端点 \(x+z\) 通过 Wilson line transport 到 \(x\)：

\[
S_W(x;y)=W(x,x+z)S(x+z;y),
\]

随后用 \(S_W(x;y)\) 参与 \(x\) 点的收缩。等价地也可在 source/另一端做位移，但
转置、共轭、Wilson-line 方向和端点必须成对改变：

\[
W(x,x+z)=W^\dagger(x+z,x).
\]

- Wilson line 使用算符定义要求的规范场。求逆可以另用 smeared gauge，但不能把
  solver gauge 静默当成 Wilson-line gauge。
- 正向、反向、\(\pm z\) 和非共线 staple 分别保存方向、长度和路径；反向路径用逐
  link dagger，不能把正路径直接转置后省略复共轭。
- `z=0` 退化为局部算符，是 Wilson-line 实现的独立 oracle。
- 每移动一步都检查累计端点；最终收缩前显式核对 spin/color 自由指标的转置和闭合。

## 交接清单

| 字段 | 必须记录 |
|---|---|
| 源 | point/wall/volume/stochastic、位置或时间片、每个组态的源数 |
| 动量 | 每条腿的源动量、sink 动量、Fourier 符号、wall 分裂 |
| smear | gauge 场身份、stout/HYP/APE 参数、Gaussian `rho/n_steps`、sink treatment |
| 非局域 | Wilson-line gauge、路径方向/长度、正负端、dagger owner |
| 统计 | source/configuration 样本身份、随机噪声、seed、归一化 |
| 收缩 | 传播子源汇、转置/共轭、spin/color 轴序、输出轴 |

来源参考：`skills/LQCD_Skills/skills/lqcd-physics-correlator/SKILL.md`，submodule
提交 `90bcbb8739c80b41a3965f41dab50bdfc9841b16`；公式已按
`pyqcd-physics-correlator`、`pyqcd-propagator` 和 PyQUDA source API 的当前约定复核。
