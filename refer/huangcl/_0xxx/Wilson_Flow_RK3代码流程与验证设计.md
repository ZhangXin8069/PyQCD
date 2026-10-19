# Wilson Flow RK3 代码流程与验证设计

本文规划 Wilson flow 的物理计算逻辑, 包括物理量, 公式, 函数职责, 调用顺序和资源集中位置. 本文不代表已经实现或测试的生产代码.

全文默认使用格点单位制. Clover 场强公式中保留格距 \(a\) 以显示归一化来源, 实际格点单位计算时取 \(a=1\).

## 用户目标

### 用户对本部分的要求

- 输入为 SU(3) gauge links \(U\), 并从 \(V_0=U\) 开始 Wilson flow.
- 使用 Lüscher 的三阶段 Lie-group RK3 演化 flowed links.
- 主数值演化在不输入 bare coupling \(g_0\) 的情况下也能工作.
- 每个完整 flow time 都检查 flowed links 是否仍属于 SU(3).
- 每个完整 flow time 都记录 Wilson flow action 随 \(t\) 的变化, 并检查它是否单调不增.
- 能量密度必须由四叶 clover 场强 \(G_{\mu\nu}\) 构造. 四个同向 plaquette loops 先组成 \(Q_{\mu\nu}\), 再由 \(Q_{\mu\nu}-Q_{\mu\nu}^\dagger\) 提取一个 \(G_{\mu\nu}\), 不另行构造 \(Q_{\nu\mu}\).
- 能量密度采用 \(E=GG/4\) 的分量定义.
- 只长期保留体积平均的单配置 \(E_i(t)\), 不保留局域 \(E_i(t,x)\).
- 不长期保留完整 flowed-configuration history.
- 每个完整 flow time 的 \(E_i(t)\), action 和 SU(3) 诊断结果都要保留.
- 当前流程不计算 ensemble 平均 \(\langle E(t)\rangle\); 它留给后续分析.
- 所有生成的结果文件使用 `.npy` 或 `.npz`.
- \(F_{\mu\nu}\) 或 \(G_{\mu\nu}\) 的 clover 构造必须是独立函数, 供能量密度和后续其他 observable 调用.
- `clover_energy_mean` 不负责构造 clover loops 或场强, 只接收已经构造的 \(G_{\mu\nu}\) 并计算 \(E_i(t)\).

### 计算对象和核心结果

对第 \(i\) 个输入 gauge configuration \(U_i(x,\mu)\), 计算 Wilson flow 后的 links \(V_{i,t}(x,\mu)\). 当前流程的核心物理输出是

\[
E_i(t_n)
=
\frac{1}{N_{\rm site}}
\sum_x
\frac14
\sum_{\mu,\nu,a}
G^a_{i,\mu\nu}(t_n,x)G^a_{i,\mu\nu}(t_n,x),
\tag{1}
\]

其中 \(t_n\) 是完整 RK3 step 对应的 flow time. \(E_i(t_n,x)\) 只作为计算式(1)时的局域概念, 不进入长期输出.

除核心结果外, 每个 \(t_n\) 还产生三类验证量:

- 不含显式 \(g_0\) 的 reduced Wilson action.
- unitarity 和 determinant 两个 SU(3) 残差.
- 相邻 flow times 之间的 action 增量和单调性判断.

### 结果用途

- \(E_i(t_n)\) 用于后续跨配置求 \(\langle E(t_n)\rangle\), 再分析 \(t^2\langle E(t)\rangle\), \(t_0\), \(w_0\) 或其他 flowed observables.
- action 曲线用于检查演化方向是否确实沿 Wilson action 的负梯度方向.
- SU(3) 残差用于检查群指数更新后 links 是否保持在 SU(3) 上.
- RK3 步长减半比较用于独立验证固定终点的三阶全局误差.

这里的 \(\langle E(t)\rangle\) 是 gauge-field energy density 对 gauge ensemble 的统计平均, 不是普通量子力学语境中的 "动能期望值".

### 待确认调整

- 输入 links 是采用 \(U_\mu=\exp(iaA_\mu)\), 还是 \(U_\mu=\exp(iag_0A_\mu^{\rm phys})\). 两种 convention 会改变 clover 场强和 \(E\) 的绝对归一化.
- 是否把 \(t=0\) 作为第一个测量和保存时刻. 保存 \(t=0\) 有利于建立 action 和 \(E_i\) 的初始基准, 代价是多一次测量和一组标量输出.
- SU(3) 残差与 action 单调性检查的容差尚未给出.
- 验证失败后是停止, 减小步长重算, 还是继续演化并记录诊断, 尚未确定.

## 核心功能与整体流程

### 用户对本部分的要求

- Wilson flow 的 RK3 演化是主流程.
- SU(3) 检查和 action 记录必须放在演化过程中, 对每个完整 flow state 执行.
- 每个完整 flow time 都计算并保留 \(E_i(t)\).
- \(G_{\mu\nu}\) 必须来自四叶 clover: 四个整理为正向 link 及其 Hermitian 共轭的 plaquettes 组成 \(Q_{\mu\nu}\), 再由 \(Q_{\mu\nu}-Q_{\mu\nu}^\dagger\) 提取场强.
- \(G_{\mu\nu}\) 的构造独立成函数, 每次处理一个 \((\mu,\nu)\) 平面, 后续其他 observable 可以直接复用.
- `clover_energy_mean` 只接收六个独立平面的 \(G_{\mu\nu}\), 完成收缩和体积平均.
- 不保存 \(E_i(t,x)\) 和完整 flowed links 历史.

### 核心功能

完整流程分为五个物理功能.

| 顺序 | 核心功能 | 输入物理量 | 主要处理 | 输出物理量 |
|---|---|---|---|---|
| 1 | Wilson flow RK3 演化 | 当前 flowed links \(V_{i,t_n}\), flow-time 步长 \(\epsilon\) | 在三个 RK stage 上分别重新构造 Wilson-flow vector field, 得到下一个完整 flow state | \(V_{i,t_{n+1}}\) |
| 2 | Clover 场强构造 | 完整 flow state \(V_{i,t_n}\), 一个方向对 \((\mu,\nu)\) | 用四个同向 plaquette loops 构造 \(Q_{\mu\nu}\), 再用 \(Q_{\mu\nu}-Q_{\mu\nu}^\dagger\) 得到该平面的 \(G_{i,\mu\nu}(t_n,x)\) | 一个全时空 clover 场强矩阵场 |
| 3 | Clover 能量密度测量 | 六个独立 \(\mu<\nu\) 平面的 \(G_{i,\mu\nu}(t_n,x)\) | 收缩颜色和 Lorentz 指标并作体积平均 | 单配置标量 \(E_i(t_n)\) |
| 4 | 流时刻数值验证 | 完整 flow state, 前一时刻 action | 计算 reduced Wilson action, SU(3) 残差和 action 增量 | 当前时刻诊断标量与检查状态 |
| 5 | 单配置结果整理 | 当前 \(t_n\) 的核心标量和诊断标量 | 沿 flow-time 维度对齐并保存结果, 不接收局域场或 flowed history | 单配置 flow-time 标量序列 |

### 整体流程

对每个输入配置 \(U_i\), 先令当前状态为 \(V_{i,0}=U_i\). 如果最终确认保留 \(t=0\), 在演化前先完成一次 clover 场强构造, 能量密度测量和验证. 随后重复以下顺序:

1. 用一个完整 RK3 step 将 \(V_{i,t_n}\) 推进到 \(V_{i,t_{n+1}}\).
2. 对六个独立 \(\mu<\nu\) 平面分别调用 clover 场强函数. 每次调用用四个整理后的 \(P\) 构造 \(Q_{\mu\nu}\), 再由 \(Q_{\mu\nu}-Q_{\mu\nu}^\dagger\) 得到一个 \(G_{i,\mu\nu}(t_{n+1},x)\).
3. 把六个场强传给 `clover_energy_mean`, 计算体积平均 \(E_i(t_{n+1})\).
4. 在同一个完整状态上计算 reduced Wilson action 和两个 SU(3) 残差.
5. 与前一个保留时刻的 action 比较, 得到 action 增量.
6. 把当前 flow time 的标量结果加入单配置结果序列.
7. 将当前 \(V_{i,t_{n+1}}\) 用作下一步输入, 释放不再需要的 RK stage 和当前测量产生的 \(G_{\mu\nu}\).
8. 到达终止 flow time 后, 把标量结果保存为 `.npy` 或 `.npz`, 然后释放最终 flowed state.

```text
输入 U_i
  -> 当前状态 V_i(t_n)
  -> Wilson flow RK3 推进一步
  -> 完整状态 V_i(t_{n+1})
  -> 对每个 mu<nu: 四个 P -> Q_mu_nu -> Q_mu_nu-Q_mu_nu^dagger -> G_mu_nu(t_{n+1}, x)
  -> 由 G_mu_nu 收缩得到 E_i(t_{n+1})
  -> reduced action 和 SU(3) 检查
  -> 收集当前时刻标量
  -> 下一 flow step 或结束并保存
```

### 循环和终止条件

- 最外层是配置指标 \(i\) 的循环. 单配置流程完成后只返回该配置的标量序列.
- 内层是 flow step 指标 \(n\) 的循环. 每个完整 step 后都测量和验证, 不使用额外的稀疏测量间隔.
- 正常终止条件是到达用户给定的 \(t_{\max}\) 或 \(N_{\rm step}\).
- 验证失败分支必须等待用户确认规则, 不能在现阶段默认停止, 重算或忽略.

### 待确认调整

- 每个 flow time 是立即更新结果文件, 还是先在当前配置中收集全部标量再统一保存, 尚未确定. 前者更利于中断恢复, 后者的写入次数更少.
- 如果后续希望某些 flow times 额外保留完整 flowed configurations 以计算其他非局域 observable, 将改变当前的存储边界和内存占用. 当前仍按不保存 flowed history 处理.

## 输入和输出的物理量

### 用户对本部分的要求

- 输入是用户提供的 SU(3) gauge links \(U_i\).
- 主输出是每个完整 flow time 的体积平均 \(E_i(t_n)\).
- 同时保留 action 和 SU(3) 检查所需的标量诊断.
- 不保留 \(E_i(t_n,x)\), clover 局域场或完整 flowed-configuration history.
- 当前不沿配置维度求 \(\langle E(t)\rangle\).
- 所有生成的结果文件使用 `.npy` 或 `.npz`.

### 读取的物理量

| 读取量 | 大致物理含义 | 各维度的物理含义 | 来源 | 使用功能 | 待确认内容 |
|---|---|---|---|---|---|
| \(U_i(x,\mu)\) | 第 \(i\) 个未 flow 的 SU(3) gauge configuration | 配置指标 \(i\), 四维格点坐标 \(x\), link 方向 \(\mu=0,1,2,3\), 两个颜色指标 | 用户输入组态 | Wilson flow RK3 演化, Clover 场强构造, 流时刻验证 | 坐标轴顺序, 方向轴位置, 颜色指标顺序 |
| 格点几何 | 四维格点范围和相邻点关系 | 欧氏时间长度, 三个空间长度, link 方向 | 输入数据或元数据 | 演化, clover 测量, 验证 | 实际格子长度和边界条件 |
| \(\epsilon\) | 格点单位下的 flow-time 步长 | 单个标量 | 用户设定 | RK3 演化, 收敛检查 | 数值取值和是否进入三阶渐近区 |
| \(t_{\max}\) 或 \(N_{\rm step}\) | flow 终点 | 单个 flow time 或总步数 | 用户设定 | 演化和结果整理 | 尚未给出 |
| 验证容差 | SU(3) 和 action 单调性允许的数值误差 | 每类检查的标量阈值 | 用户设定 | 流时刻验证 | 尚未给出 |
| \(g_0\) 的可选元数据 | 将 coupling-absorbed 量换回指定物理 convention 时使用 | 每个 ensemble 或 bare coupling 一个标量 | ensemble 元数据 | 后续比较, 不进入主 RK3 演化 | link convention 和绝对归一化需求 |

用户自行生成的输入数据默认已经遵守其既定组织约定. 本流程不额外增加通用 shape 检查; 但方向指标, 颜色指标和边界条件会直接改变 plaquette 物理含义, 因而必须在进入正式计算前确认.

### 输出的物理量

| 输出量 | 物理含义 | 各维度的物理含义 | 生成于哪个功能 | 结果类别 | 后续用途 |
|---|---|---|---|---|---|
| \(G_{i,\mu\nu}(t_n,x)\) | 一个 \((\mu,\nu)\) 平面的 clover 场强 | 配置指标, flow time, 四维格点坐标, 方向对, 两个颜色指标 | Clover 场强构造 | 可复用中间物理量 | 当前用于 \(E_i(t_n)\), 后续可供其他 gluonic observables 调用 |
| \(E_i(t_n)\) | 第 \(i\) 个配置在 \(t_n\) 的体积平均 clover 能量密度 | 配置指标, flow time | Clover 能量密度测量 | 核心结果 | ensemble 平均和后续 flowed-observable 分析 |
| \(\mathcal S_{W,i}^{\rm red}(t_n)\) | 不含显式 \(g_0\) 的 Wilson action 诊断量 | 配置指标, flow time | 流时刻数值验证 | 验证量 | action 单调性曲线 |
| \(r_{V,i}^{\rm unit}(t_n)\) | flowed links 偏离 unitarity 的最大残差 | 配置指标, flow time | 流时刻数值验证 | 验证量 | SU(3) 检查 |
| \(r_{V,i}^{\rm det}(t_n)\) | flowed links 的 determinant 偏离 1 的最大残差 | 配置指标, flow time | 流时刻数值验证 | 验证量 | SU(3) 检查 |
| \(\Delta\mathcal S_{W,i}^{\rm red}(t_n)\) | 相邻保留时刻的 reduced action 差 | 配置指标, flow time | 流时刻数值验证 | 诊断量 | 定位 action 正跳动 |
| RK3 收敛结果 | 三种步长在同一终点所得 observable 的差和误差比 | 配置指标, 固定终点, observable, 步长组 | 独立收敛验证 | 验证量 | 检查三阶全局误差 |

每个配置的上述 flow-time 标量序列保存为 `.npy` 或 `.npz`. 文件内部如何命名各数组属于实现阶段, 不改变这里的物理维度和保存范围.

### 瞬时中间物理量

| 中间量 | 物理作用 | 生命周期 |
|---|---|---|
| 当前 \(V_{i,t_n}\) | 继续下一 flow step 并完成当前测量 | 只在当前配置演化期间保留 |
| RK3 stage states \(W_0,W_1,W_2\) | 构造三个 stage 的 vector field | 只在一个 flow step 内保留 |
| \(Z_0,Z_1,Z_2\) | 已包含 \(\epsilon\) 的 Lie-algebra stage increments | 只在一个 flow step 内保留 |
| \(Q_{\mu\nu}(t,x)\) | 构造一个平面的 clover 场强 | 只在一次场强函数调用期间保留 |
| \(G_{\mu\nu}(t,x)\) | Clover 场强函数的输出, 供能量密度或其他 observable 使用 | 保留到当前调用者完成使用, 本流程不长期保存 |
| \(E_i(t,x)\) | 局域能量密度 | 不长期保留; 允许直接累计体积和而不形成输出数组 |

### 符号和约定对应

- 力项推导笔记使用 anti-Hermitian generators \(T^a\), 并令 \(Z\in\mathfrak{su}(3)\).
- Clover 场强使用 Hermitian 记号 \(F_{\mu\nu}\). 本设计将其映射为

  \[
  F_{\mu\nu}\big|_{\text{clover}}
  \longleftrightarrow
  G_{\mu\nu}^{\rm clov}\big|_{\text{本设计}}.
  \tag{2}
  \]

- \(S_{\rm sim}\) 是产生 ensemble 时使用的模拟作用量, \(S_{\rm flow}=S_W\) 是当前 Wilson flow 的生成作用量, \(O_{\rm lat}=E_{\rm clover}\) 是测量 observable 的离散化. 三者不要求相同.

### 待确认调整

- 输入 \(U_i\) 的欧氏时间轴, 空间轴, link 方向轴和颜色指标顺序尚未给出. 确认这些内容不会增加新的物理计算, 但会决定所有位移和矩阵乘法的指标含义.
- 边界条件尚未给出. 周期边界是常见选择, 但不能在未确认时写成用户数据的既定条件.
- 输出使用 `.npy` 还是 `.npz`, 以及每配置一个文件还是多个文件, 尚未指定. 两者都满足用户给出的格式范围; 选择只影响结果组织和读取方式, 不改变物理内容.
- 是否把检查状态作为独立序列保存, 或仅保存原始残差与 action 增量并在分析时重新判定, 尚未确定. 后一种方式保留的信息更基础, 但读取后需要重新应用容差.

## 各核心功能的公式与调用逻辑

### 用户对本部分的要求

- Wilson flow 演化从 \(V_0=U\) 开始, 使用 Lüscher RK3.
- 主演化使用不含显式 \(g_0\) 的 vector field.
- 每个完整 flow state 都重新计算 clover \(E_i(t)\), reduced Wilson action 和 SU(3) 残差.
- 每个 RK3 stage 必须从当前完整 stage configuration 重新构造所需的 Wilson-flow vector field.
- Clover 场强由四个已经整理为正向 link 及其 Hermitian 共轭的 plaquette loops 构造. 四个 loops 先相加为 \(Q_{\mu\nu}\), 再直接用 \(Q_{\mu\nu}-Q_{\mu\nu}^\dagger\) 提取场强, 不重新构造 \(Q_{\nu\mu}\). 能量密度使用 \(GG/4\).
- `clover_field_strength` 独立构造一个指定 \((\mu,\nu)\) 平面的 \(G_{\mu\nu}(t,x)\), 使后续其他 observable 能直接调用.
- `clover_energy_mean` 不构造 links, plaquette loops, \(Q_{\mu\nu}\) 或 \(G_{\mu\nu}\); 它只接收六个平面的场强并完成能量密度收缩和体积平均.
- 只返回体积平均 \(E_i(t)\), 不保存 \(E_i(t,x)\) 或 flowed links 历史.

### 来源和计算公式

#### Wilson-flow vector field 和 RK3

来源为 `Wilson_Flow_RK3演化与SU3力项推导.md` 的式(15), 式(20), 式(22), 式(38), 以及该笔记引用的文献[1]的(1.4), (C.2), (C.3)式.

对当前完整 configuration \(V\), 固定 link \((x,\mu)\) 的两个 \(\nu\) 向 staples 及其总和为

\[
\begin{aligned}
C_{\mu\nu}^{(+)}(x)
&=
V_\nu(x+\hat\mu)
V_\mu^\dagger(x+\hat\nu)
V_\nu^\dagger(x),
\\
C_{\mu\nu}^{(-)}(x)
&=
V_\nu^\dagger(x+\hat\mu-\hat\nu)
V_\mu^\dagger(x-\hat\nu)
V_\nu(x-\hat\nu),
\\
\Sigma_{\mu\nu}(x)
&=C_{\mu\nu}^{(+)}(x)+C_{\mu\nu}^{(-)}(x),
\\
\Sigma_\mu(x)
&=\sum_{\nu\ne\mu}\Sigma_{\mu\nu}(x).
\end{aligned}
\tag{3}
\]

对任意 \(3\times3\) 复矩阵 \(M\), traceless anti-Hermitian projection 为

\[
[M]_{\rm TA}
=
\frac12(M-M^\dagger)
-
\frac16\operatorname{Tr}(M-M^\dagger)\mathbf1.
\tag{4}
\]

Wilson-flow vector field 为

\[
Z(V)(x,\mu)
=
-g_0^2\partial_{x,\mu}S_W[V]
=
-\left[V_\mu(x)\Sigma_\mu(x)\right]_{\rm TA}.
\tag{5}
\]

式(5)右端不含显式 \(g_0\), 因而主 RK3 演化不需要读取 \(g_0\). 这个结果依赖当前推导采用的 Wilson-action normalization, 左 Lie 导数, 普通六 staple 几何和以及式(4)的投影. 如果这些约定改变, 整体系数和矩阵次序必须重新确定.

设 \(\epsilon>0\), 一个完整 Lüscher RK3 step 为

\[
\begin{aligned}
W_0 &= V_t,
& Z_0 &= \epsilon Z(W_0),
\\
W_1 &= \exp\left(\frac14Z_0\right)W_0,
& Z_1 &= \epsilon Z(W_1),
\\
W_2 &= \exp\left(\frac89Z_1-\frac{17}{36}Z_0\right)W_1,
& Z_2 &= \epsilon Z(W_2),
\\
V_{t+\epsilon}
&=
\exp\left(\frac34Z_2-\frac89Z_1+\frac{17}{36}Z_0\right)W_2.
\end{aligned}
\tag{6}
\]

必须在完整的 \(W_0\), \(W_1\), \(W_2\) 上分别重新计算 \(Z(W_i)\). \(Z_i=\epsilon Z(W_i)\) 已经包含一次步长, 指数中的 \(Z_i\) 不能再次乘 \(\epsilon\). 文献[1]给出的误差阶为

\[
\delta_{\rm local}=O(\epsilon^4),
\qquad
\delta_{\rm global}(t_\star)=O(\epsilon^3),
\tag{7}
\]

其中第二项是在固定 flow-time 终点 \(t_\star\) 上的累积全局误差.

#### 四叶 clover 场强和能量密度

四叶 clover 的几何定义来自用户提供的 `09 场强张量与clover项.md` 第 2 节. 对每个独立平面 \(\mu<\nu\), 围绕格点 \(x\) 的四个同向 plaquettes 和 clover sum 直接写成正向 link 及其 Hermitian 共轭的乘积:

\[
\begin{aligned}
P_{\mu\nu}(x)
&=
V_\mu(x)V_\nu(x+\hat\mu)
V_\mu^\dagger(x+\hat\nu)V_\nu^\dagger(x),
\\
P_{\nu,-\mu}(x)
&=
V_\nu(x)V_\mu^\dagger(x+\hat\nu-\hat\mu)
V_\nu^\dagger(x-\hat\mu)V_\mu(x-\hat\mu),
\\
P_{-\mu,-\nu}(x)
&=
V_\mu^\dagger(x-\hat\mu)V_\nu^\dagger(x-\hat\mu-\hat\nu)
V_\mu(x-\hat\mu-\hat\nu)V_\nu(x-\hat\nu),
\\
P_{-\nu,\mu}(x)
&=
V_\nu^\dagger(x-\hat\nu)V_\mu(x-\hat\nu)
V_\nu(x+\hat\mu-\hat\nu)V_\mu^\dagger(x),
\\
Q_{\mu\nu}(x)
&=
P_{\mu\nu}(x)+P_{\nu,-\mu}(x)
+P_{-\mu,-\nu}(x)+P_{-\nu,\mu}(x).
\end{aligned}
\tag{8}
\]

式(8)右端的全部 \(V\) 取自同一个 flow time, 且不再引入独立的反向 link 符号. 由于反向取向的 clover 矩阵可以由 Hermitian 共轭得到,

\[
Q_{\nu\mu}(t,x)=Q_{\mu\nu}^\dagger(t,x),
\]

实际计算中只需对 \(Q_{\mu\nu}\) 取反 Hermitian 部分:

\[
\widetilde G_{\mu\nu}^{\rm clov}(t,x)
=
-\frac{i}{8a^2}
\left[Q_{\mu\nu}(t,x)-Q_{\mu\nu}^\dagger(t,x)\right].
\tag{9}
\]

式(8)至式(9)完整定义单个 \((\mu,\nu)\) 平面的 clover 场强构造. 这部分由独立的 `clover_field_strength` 负责. 该函数输出全时空上的 \(G_{\mu\nu}(t,x)\), 不在函数内部计算能量密度.

这里写成 \(\widetilde G\), 是因为当前公式未额外加入有限格距下的 traceless projection. 若采用 Hermitian SU(3) generators

\[
G_{\mu\nu}=G_{\mu\nu}^at^a,
\qquad
\operatorname{Tr}(t^at^b)=\frac12\delta^{ab},
\tag{10}
\]

则局域能量密度和体积平均按照式(1)计算. 由于 \(G_{\nu\mu}=-G_{\mu\nu}\), Lorentz 求和只需覆盖六个独立的 \(\mu<\nu\) 平面. 在且仅在式(10)的 generator normalization 成立, 并且 \(G_{\mu\nu}\) 已按 SU(3) 场强约定处理时, 式(1)可以写成

\[
E_i(t)
=
\frac{1}{N_{\rm site}}
\sum_x\sum_{\mu<\nu}
\operatorname{ReTr}
\left[G_{i,\mu\nu}(t,x)G_{i,\mu\nu}(t,x)\right].
\tag{11}
\]

`clover_energy_mean` 只实现式(1)或满足条件时的等价式(11). 它的输入是六个独立平面的 \(G_{\mu\nu}(t,x)\), 不重复构造式(8)至式(9)中的 links, loops 或 clover sum.

不能把含有 identity trace part 的 \(\widetilde G\) 未经确认地直接代入式(11), 否则会把非 SU(3) 分量计入能量密度.

若 links 使用

\[
U_\mu(x)
=
\exp\left[iag_0A_\mu^{\rm phys}(x)+O(a^2)\right],
\tag{12}
\]

则直接从 links 提取的是 coupling-absorbed 场强, \(G^{\rm link}=g_0G^{\rm phys}\), 相应地 \(E^{\rm link}=g_0^2E^{\rm phys}\). 不输入 \(g_0\) 仍可计算 \(E^{\rm link}\). 同一 bare coupling 下, 具有相同 \(g_0\) 次方的比值可以消去共同因子, 但不同 bare couplings 的绝对 \(E\) 不能自动直接比较.

#### Wilson action 和 SU(3) 验证

action 定义来自 `Wilson_Flow_RK3演化与SU3力项推导.md` 的式(3)和式(27). SU(3) 残差和步长收敛依据来自 `Wilson_Flow结果验证_文献笔记.md` 的对应验证部分.

当前力项推导采用的 Wilson action 是

\[
\begin{aligned}
S_W[V]
&=
\frac{2}{g_0^2}
\sum_x\sum_{\mu<\nu}
\operatorname{ReTr}\left[\mathbf1-P_{\mu\nu}(x)\right]
\\
&=
\frac{6}{g_0^2}
\sum_x\sum_{\mu<\nu}
\left[1-\frac13\operatorname{ReTr}P_{\mu\nu}(x)\right].
\end{aligned}
\tag{13}
\]

式(13)中的 trace normalization 必须写清:

- 若 \(\operatorname{Tr}\) 是普通 \(3\times3\) color trace, 无量纲 plaquette 组合是 \(1-\frac13\operatorname{ReTr}P\), 或等价的 \(\operatorname{ReTr}(\mathbf1-P)=3-\operatorname{ReTr}P\).
- 只有当 \(\operatorname{Tr}\) 已经表示 normalized trace \(\frac13\operatorname{Tr}\) 时, 才能简写为 \(1-\operatorname{ReTr}P\).
- 对普通未归一化 color trace, 单独写 \(1-\operatorname{ReTr}P\) 与式(13)不一致.

定义不含显式 \(g_0\) 的诊断量

\[
\begin{aligned}
A_P[V]
&=
\sum_x\sum_{\mu<\nu}
\left[1-\frac13\operatorname{ReTr}P_{\mu\nu}(x)\right],
\\
\mathcal S_W^{\rm red}[V]
&=
6A_P[V]
=g_0^2S_W[V].
\end{aligned}
\tag{14}
\]

最后一个等号只在式(13)的 action normalization 下成立. 对同一 bare coupling,

\[
\frac{S_W(t)}{S_W(t_{\rm ref})}
=
\frac{\mathcal S_W^{\rm red}(t)}
{\mathcal S_W^{\rm red}(t_{\rm ref})}.
\tag{15}
\]

每个完整 flow state 的 SU(3) 残差定义为

\[
\begin{aligned}
r_V^{\rm unit}(t)
&=
\max_{x,\mu}
\left\|V_{t,\mu}(x)V_{t,\mu}^\dagger(x)-\mathbf1\right\|_F,
\\
r_V^{\rm det}(t)
&=
\max_{x,\mu}
\left|\det V_{t,\mu}(x)-1\right|.
\end{aligned}
\tag{16}
\]

从文献[1]的(1.4)式出发, Wilson action 沿连续 Wilson flow 满足

\[
\frac{d}{dt}S_W[V_t]
=
-g_0^2\sum_{x,\mu,a}
\left(\partial_{x,\mu}^aS_W[V_t]\right)^2
\leq0.
\tag{17}
\]

相邻保留时刻记录

\[
\Delta\mathcal S_W^{\rm red}(t_n)
=
\mathcal S_W^{\rm red}(t_n)
-
\mathcal S_W^{\rm red}(t_{n-1}).
\tag{18}
\]

理论条件是式(18)不大于 0. 数值计算应保存原始增量, 再依据用户确认的容差判定是否存在不可接受的正跳动.

为验证 RK3 的三阶全局误差, 在相同输入 \(U_i\) 和固定终点 \(t_\star\) 上分别使用 \(\epsilon\), \(\epsilon/2\), \(\epsilon/4\). 对同一个 gauge-invariant 标量 \(O\), 定义

\[
\begin{aligned}
\Delta_1
&=
\left|O_\epsilon(t_\star)-O_{\epsilon/2}(t_\star)\right|,
\\
\Delta_2
&=
\left|O_{\epsilon/2}(t_\star)-O_{\epsilon/4}(t_\star)\right|,
\\
\frac{\Delta_1}{\Delta_2}
&\longrightarrow2^3=8.
\end{aligned}
\tag{19}
\]

该检查是独立的积分器验收测试, 不要求每个常规配置都执行三次完整演化.

#### 单配置结果和后续 ensemble 平均

每个配置的长期结果集合为

\[
\mathcal R_i
=
\left\{
t_n,
E_i(t_n),
\mathcal S_{W,i}^{\rm red}(t_n),
r_{V,i}^{\rm unit}(t_n),
r_{V,i}^{\rm det}(t_n),
\Delta\mathcal S_{W,i}^{\rm red}(t_n)
\right\}_{n\in\mathcal T_{\rm save}}.
\tag{20}
\]

当前要求下, \(\mathcal T_{\rm save}\) 包含每个完整 RK3 step 对应的 flow time. 是否额外包含 \(t=0\) 尚待确认.

后续拥有 \(N_{\rm cfg}\) 个独立配置后, 才计算

\[
\left\langle E(t_n)\right\rangle
=
\frac{1}{N_{\rm cfg}}
\sum_{i=1}^{N_{\rm cfg}}E_i(t_n).
\tag{21}
\]

当前流程不执行式(21). \(t_0\) 或 \(w_0\) 应由 ensemble 平均曲线构造, 不是先逐配置求 crossing 再平均.

### 函数职责

函数边界按照本节公式的数据依赖划分. Clover 场强构造和能量密度收缩是两个独立职责.

| 函数名称 | 计算的物理量或承担的作用 | 输入物理量 | 输出物理量 | 公式或依据 |
|---|---|---|---|---|
| `wilson_flow_vector_field` | 构造每条 link 上的 \(Z(V)\in\mathfrak{su}(3)\) | 完整 \(V\), 格点几何 | \(Z(V)\) | 式(3)至式(5) |
| `rk3_flow_step` | 在三个 stage 上分别重算 vector field, 将 \(V_t\) 推进到 \(V_{t+\epsilon}\) | \(V_t\), \(\epsilon\) | \(V_{t+\epsilon}\) | 式(6) |
| `clover_field_strength` | 对一个指定 \((\mu,\nu)\) 平面构造四个 loops 和 \(Q_{\mu\nu}\), 取 \(Q_{\mu\nu}-Q_{\mu\nu}^\dagger\) 后得到全时空 \(G_{\mu\nu}(t,x)\), 不构造 \(Q_{\nu\mu}\) | 完整 \(V_{i,t_n}\), \(\mu\), \(\nu\), 格点几何, 场强 convention | 一个平面的 \(G_{i,\mu\nu}(t_n,x)\) | 式(8)至式(10) |
| `clover_energy_mean` | 只对已经给出的六个 clover 场强平面做颜色, Lorentz 和体积收缩 | 六个 \(G_{i,\mu\nu}(t_n,x)\), \(\mu<\nu\) | 标量 \(E_i(t_n)\) | 式(1), 式(10), 式(11) |
| `flow_state_checks` | 计算 reduced action, 两个 SU(3) 残差和相邻 action 增量 | 完整 \(V_{i,t_n}\), 前一 action, 容差 | 式(14), 式(16), 式(18)对应的标量和检查状态 | 式(13)至式(18) |
| `rk3_convergence_check` | 比较三个步长在同一终点的 gauge-invariant 标量 | 三组终点结果 | \(\Delta_1\), \(\Delta_2\), 误差比 | 式(7), 式(19) |
| `run_configuration_flow` | 组织一个输入配置的 flow-time 循环, 场强构造, 能量密度测量, 验证和标量收集 | \(U_i\), flow 参数, 验证参数 | \(\mathcal R_i\) | 式(20)和整体流程 |
| `save_flow_results` | 将单配置标量序列保存为用户指定格式 | \(\mathcal R_i\), 配置标识, `.npy` 或 `.npz` 选择 | 单配置结果文件 | 用户的文件格式要求 |

### 处理逻辑和调用顺序

对一个配置, 当前 \(V\) 必须保留到下一步; RK3 stage fields 只在当前 step 内保留. 每得到一个完整 flow state, 主流程依次对六个 \(\mu<\nu\) 平面调用 `clover_field_strength`. 每次调用只处理一个方向对并返回对应的 \(G_{\mu\nu}(t,x)\).

六个场强平面构造完成后, 主流程把它们传给 `clover_energy_mean`. 该函数只进行式(1)或式(11)中的收缩和体积平均, 不重新访问 links, 也不构造 plaquette loops. 因此, 后续其他 gluonic observables 可以绕过 `clover_energy_mean`, 直接复用 `clover_field_strength` 的输出.

当前 \(E_i(t_n)\) 计算和其他需要场强的调用完成后, 六个 \(G_{\mu\nu}\) 可以释放. 本流程不把它们加入长期结果文件.

```python
for configuration_id, U_i in configurations:
    V = U_i
    results_i = []
    previous_action = None

    if include_t0:
        G_0 = []
        for mu, nu in independent_mu_nu_planes:
            G_mu_nu = clover_field_strength(
                V,
                mu,
                nu,
                geometry,
                field_convention,
            )
            G_0.append(G_mu_nu)

        E_0 = clover_energy_mean(G_0)
        checks_0 = flow_state_checks(V, previous_action, tolerances)
        results_i.append((
            0,
            E_0,
            checks_0.reduced_action,
            checks_0.unitarity_residual,
            checks_0.determinant_residual,
            checks_0.action_delta,
        ))
        previous_action = checks_0.reduced_action

    for n in flow_steps:
        V = rk3_flow_step(V, epsilon)
        t_n = (n + 1) * epsilon

        G_n = []
        for mu, nu in independent_mu_nu_planes:
            G_mu_nu = clover_field_strength(
                V,
                mu,
                nu,
                geometry,
                field_convention,
            )
            G_n.append(G_mu_nu)

        E_n = clover_energy_mean(G_n)
        checks_n = flow_state_checks(V, previous_action, tolerances)
        results_i.append((
            t_n,
            E_n,
            checks_n.reduced_action,
            checks_n.unitarity_residual,
            checks_n.determinant_residual,
            checks_n.action_delta,
        ))
        previous_action = checks_n.reduced_action

        if checks_n.failed:
            # 按用户确认的失败规则停止, 重算或继续

    save_flow_results(results_i, configuration_id, output_format)

return all_saved_result_paths
```

独立的 RK3 收敛检查复用同一演化和 clover 测量职责:

```python
for step_size in [epsilon, epsilon / 2, epsilon / 4]:
    V = U_i
    t = 0

    while t < t_star:
        V = rk3_flow_step(V, step_size)
        t = t + step_size

    G_at_t_star = []
    for mu, nu in independent_mu_nu_planes:
        G_mu_nu = clover_field_strength(
            V,
            mu,
            nu,
            geometry,
            field_convention,
        )
        G_at_t_star.append(G_mu_nu)

    O[step_size] = clover_energy_mean(G_at_t_star)

convergence = rk3_convergence_check(
    O[epsilon],
    O[epsilon / 2],
    O[epsilon / 4],
)
```

### 用户可调参数

| 参数 | 物理或数值意义 | 影响的步骤 | 当前状态 |
|---|---|---|---|
| \(\epsilon\) | flow-time 步长 | RK3 离散误差, 总步数和耗时 | 待确定, 需要收敛验证 |
| \(t_{\max}\) 或 \(N_{\rm step}\) | flow 终点 | flow-time 范围和总耗时 | 待确定 |
| `include_t0` | 是否测量并保存未 flow 的初始状态 | 输出起点和第一项 action 增量 | 待确认 |
| SU(3) 容差 | unitarity 和 determinant 的通过条件 | `flow_state_checks` | 待确定 |
| action 单调性容差 | 数值上允许的 action 正跳动 | `flow_state_checks` | 待确定 |
| 验证失败规则 | 停止, 减小步长重算或继续记录 | flow-time 循环分支 | 待确认 |
| \(t_\star\) | RK3 收敛检查的共同终点 | `rk3_convergence_check` | 待确定 |
| 输出格式 | `.npy` 或 `.npz` | 单配置结果文件 | 两者均满足用户要求, 具体选择待确定 |

Wilson action 类型, 左 Lie 导数约定, RK3 系数和六个 \(\mu<\nu\) clover 平面属于当前物理定义, 不是可任意扫描的参数.

### 待确认调整

- 当前 clover 场强公式未额外加入有限格距下的 traceless projection. 若后续要求严格限制在 \(\mathfrak{su}(3)\), 需要在现有反 Hermitian 组合之后再确定采用显式的 traceless Hermitian 投影还是八个 adjoint 分量表示.
- 输入 link 的 coupling convention 尚未确定. 这不妨碍不含显式 \(g_0\) 的主 flow, 但会影响 \(E\) 的绝对解释和不同 bare couplings 之间的比较.
- 边界条件会改变所有位移, staples 和 clover loops. 在确认前不能默认采用周期位移函数.
- 若验证失败后自动减小 \(\epsilon\) 并重算, 能增强稳健性, 但会改变用户尚未确认的主循环和耗时. 当前伪代码只保留失败分支, 不提前采用自动重算.
- 六个 \(G_{\mu\nu}\) 平面同时传给 `clover_energy_mean` 最清楚地保持了函数边界, 但会同时保留六个全时空颜色矩阵场. 如果内存成为限制, 可以进一步设计逐平面累积接口; 这会改变 `clover_energy_mean` 的输入方式, 因此当前不提前采用.

## 输入输出、参数与资源估算汇总

### 用户对本部分的要求

- 汇总输入 gauge links, flow 参数, 核心输出和验证量.
- 只长期保留 \(E_i(t_n)\) 和必要的标量诊断, 不保留 \(E_i(t_n,x)\) 或完整 flowed-link history.
- 比较完整组态与 \(E_i(t)\) 序列的数据量, 并据此判断存储重点.
- 所有生成的结果文件使用 `.npy` 或 `.npz`.
- 资源估算只依据已经确认的物理维度和同时驻留关系, 不虚构未提供的格子大小, 配置数或数值精度.

### 输入, 输出和参数汇总

| 类别 | 物理量或参数 | 覆盖的物理自变量或指标 | 当前处理 |
|---|---|---|---|
| 核心输入 | \(U_i(x,\mu)\) | 配置, 四维格点坐标, link 方向, 两个颜色指标 | 每次处理一个配置, 作为 \(V_{i,0}\) |
| 几何输入 | 格点长度和边界条件 | 欧氏时间, 三个空间方向 | 决定全部相邻点, plaquette, staple 和 clover 路径 |
| 演化参数 | \(\epsilon\), \(t_{\max}\) 或 \(N_{\rm step}\) | flow time | 控制 RK3 演化区间和离散误差 |
| 验证参数 | SU(3) 容差, action 容差, 失败规则 | 验证类别 | 控制通过条件和失败分支 |
| 核心输出 | \(E_i(t_n)\) | 配置, flow time | 每个完整 step 保留 |
| 验证输出 | \(\mathcal S_W^{\rm red}\), \(r_V^{\rm unit}\), \(r_V^{\rm det}\), \(\Delta\mathcal S_W^{\rm red}\) | 配置, flow time | 每个完整 step 保留 |
| 独立验收输出 | \(\Delta_1\), \(\Delta_2\), \(\Delta_1/\Delta_2\) | 配置, 固定终点, observable, 步长组 | RK3 收敛测试时保留 |
| 可复用中间量 | \(G_{\mu\nu}(t,x)\) | 四维格点, 一个方向对, 两个颜色指标 | 由独立场强函数产生, 当前调用者使用后释放 |
| 其他瞬时中间量 | 当前 \(V_t\), RK3 stages, vector fields, clover loops 和 \(Q_{\mu\nu}\) | 四维格点, 方向, 颜色矩阵 | 当前 step 或场强函数调用结束后释放 |
| 后续 ensemble 量 | \(\langle E(t)\rangle\), \(t^2\langle E(t)\rangle\), \(t_0\), \(w_0\) | 配置集合, flow time | 不在当前单配置流程中计算 |

### 内存开销位置

完整 gauge configuration 同时覆盖四维格点坐标, 4 个 link directions 和 \(3\times3\) 复颜色矩阵. 与此相比, \(E_i(t_n)\) 只覆盖配置指标和 flow time. 因此, 对相同配置数和保留时刻数, 一份完整 flowed-link history 的数据量远大于 \(E_i(t_n)\) 标量序列. 当前不保存 flowed history 的选择能够移除长期存储的主要来源.

各核心功能的主要内存位置如下:

| 核心功能 | 必须存在的主要数组 | 可能同时存在的临时数组 | 内存判断 |
|---|---|---|---|
| Wilson flow RK3 演化 | 当前完整 \(V_t\) | \(W_1,W_2,Z_0,Z_1,Z_2\), staples 和 matrix-exponential 工作区 | 预计是峰值内存主要来源, 因为多个对象都覆盖四维格点, 方向和颜色矩阵 |
| Clover 场强构造 | 当前完整 \(V_t\) | 一个平面的四个 loops, \(Q_{\mu\nu}\), \(G_{\mu\nu}\) | 单次函数调用只处理一个 \((\mu,\nu)\), 但主流程会保留六个输出场强供能量密度使用 |
| Clover 能量密度测量 | 六个独立平面的 \(G_{\mu\nu}(t,x)\) | 颜色, Lorentz 和体积收缩产生的归约临时量 | `clover_energy_mean` 不再重复保存 links, loops 或 \(Q_{\mu\nu}\) |
| 流时刻数值验证 | 当前完整 \(V_t\) | plaquette, \(VV^\dagger\), determinant 和归约临时量 | 输出只有标量, 但中间计算仍覆盖全部 links |
| 单配置结果整理 | flow-time 标量序列 | 格式整理和写入产生的临时结果 | 相对整场数组很小 |

按当前函数边界, 六个 \(G_{\mu\nu}(t,x)\) 由六次独立的 `clover_field_strength` 调用产生, 然后一起传给 `clover_energy_mean`. 它们只在当前测量期间保留, 不写入长期结果文件. \(E_i(t,x)\) 也不作为输出数组保留.

一次只处理一个配置可以避免同时驻留多个 \(U_i\) 或 \(V_i\). 如果后续并行处理多个配置, 内存将近似随同时驻留的配置数增加.

无法从当前材料给出具体峰值字节数, 因为实际格子长度, 计算精度, matrix-exponential 工作区和 stage storage 复用关系尚未提供. 在这些信息缺失时只保留定性比较, 不使用假定的格子尺寸代替用户数据.

### 输出占用和文件范围

每个配置的核心结果和验证量都只沿 flow time 排列:

- \(E_i(t_n)\) 是一个 flow-time 标量序列.
- reduced action, 两个 SU(3) 残差和 action 增量各自也是 flow-time 标量序列.
- RK3 收敛测试只增加三个步长的终点 observable 和少量比较标量.
- 不输出格点坐标依赖的 \(E_i(t_n,x)\), \(G_{\mu\nu}(t_n,x)\), \(Q_{\mu\nu}(t_n,x)\) 或 \(V_{i,t_n}(x,\mu)\).

输出文件只能使用 `.npy` 或 `.npz`. 一个 `.npz` 可以把同一配置的多个命名标量序列放在一起; 多个 `.npy` 可以让每个数组独立读取. 两种组织都满足用户要求. 在未确定具体选择, 数组精度和压缩设置前, 不比较两种格式的具体文件大小.

### 耗时集中位置

- 每个 RK3 flow step 在 \(W_0,W_1,W_2\) 上三次完整计算 Wilson-flow vector field, 并对每条 link 完成三次群指数更新. 这是演化部分的主要重复计算.
- 每个完整 flow time 都测量 clover \(E_i(t_n)\). 四维中有六个独立 \(\mu<\nu\) 平面, 每个平面需要四个有向 plaquette loops, 因而 clover 测量是主要 observable 成本.
- Action, unitarity 和 determinant 检查同样覆盖全部格点和 link directions. 虽然输出只是标量, 全场矩阵运算和归约仍产生计算成本.
- RK3 收敛测试在相同终点分别使用 \(\epsilon\), \(\epsilon/2\), \(\epsilon/4\). 在相同测量安排下, 三次演化的总 step 数是粗步长单次演化的 \(1+2+4=7\) 倍.
- 当前材料不足以给出墙钟时间. 需要在确定格子大小, 步数和实现后进行基准测试.

### 仍需确认的信息

为完成下一阶段实现和可靠资源估算, 还需要确认:

- 实际 \(N_t,N_x,N_y,N_z\) 和配置数量.
- 输入数组中物理坐标, link 方向和颜色指标的顺序.
- 实际边界条件.
- \(\epsilon\), \(t_{\max}\) 或 \(N_{\rm step}\).
- 是否测量并保存 \(t=0\).
- SU(3) 和 action 检查容差, 以及失败后的处理规则.
- Clover 场强的 traceless prescription 和 link coupling convention.
- `.npy` 与 `.npz` 的具体分工, 每配置文件数量和写盘时机.
- RK3 stage arrays 是否允许复用存储, 以及 matrix exponential 所需的实际工作区.

### 依据材料

- `Wilson_Flow_RK3演化与SU3力项推导.md`: Wilson action, staples, traceless anti-Hermitian projection, Wilson-flow vector field 和 Lüscher RK3.
- `09 场强张量与clover项.md`: 四叶 clover 几何和场强提取公式.
- `Wilson_Flow结果验证_文献笔记.md`: action 单调性, SU(3) 残差和 RK3 步长收敛检查.
- 文献[1]: Martin Lüscher, *Properties and uses of the Wilson flow in lattice QCD*, arXiv:1006.4518.

本文给出的是物理计算流程, 公式依赖, Python 风格调用关系和定性资源估算. 它没有声称相关程序已经实现, 运行或通过数值验证.

### 待确认调整

- 如果实际内存允许, 可以并行处理多个配置以增加吞吐量; 代价是完整 links, RK3 stages 和 clover 临时场按并行配置数增加. 当前流程仍采用一次一个配置.
- 如果运行中断恢复更重要, 可以每个 flow time 立即更新结果文件; 如果减少写入次数更重要, 可以在配置结束后统一保存. 需要先确定可靠性和写入频率的优先级.
- 如果后续 observable 需要 flowed links, 可以只保存少数指定 flow times 的完整配置, 而不是恢复完整 history. 这会扩大当前用户指定的输出范围, 因而在确认前不采用.
- 在没有实际格子长度, 精度和同时驻留关系前, 不给出具体 GiB 或 MiB 数字. 获得这些输入后可再补充定量估算.
