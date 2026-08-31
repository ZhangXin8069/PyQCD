# Wilson flow 与链接涂抹 reference

## 适用范围

本文件用于区分 Wilson flow 与 Stout/APE/HYP 等有限次涂抹，并核对 TMD 前置的流化
场。流化场只在通过几何和群性质检查后交给 `pyqcd-tmd-algorithm`。

## Wilson flow

连续形式可写为

\[
\dot V_\mu(x,\tau)=Z_\mu\lbrack V\rbrack(x,\tau)V_\mu(x,\tau),
\qquad
Z_\mu=P_{\rm ah}[\Omega_\mu V_\mu^\dagger].
\]

PyQCD 的 `pyqcd/renorm/_gradient_flow.py` 使用 Wilson flow 的三阶 RK 离散步骤：

\[
\begin{aligned}
W_1&=e^{\epsilon Z_0/4}W_0,\\
W_2&=e^{(8\epsilon Z_1/9-17\epsilon Z_0/36)}W_1,\\
V_{\tau+\epsilon}&=e^{(3\epsilon Z_2/4-8\epsilon Z_1/9+17\epsilon Z_0/36)}W_2.
\end{aligned}
\]

记录初始场、步长、步数、$\tau$、每步或每个输出点的幺正性误差和流能量定义。
PyQCD 参数 $\tau\equiv t/a^2$ 是无量纲格点流时间；$\tau=3$ 才对应物理流时间
$t=3a^2$。它不是所有任务的自动默认值，必须随数据显式保存。

`wilson_action_density` 返回逐格点、六个 $\mu<\nu$ 平面平均的
$1-\operatorname{ReTr}P_{\mu\nu}/N_c$，用于核对 Wilson 流的作用量下降；
`flow_action_density` 返回 Clover $G^2$ 离散，供 $t^2E(t)$、SFTX 与尺度设定使用。
两者不是同一个有限格距观测量，禁止用后者在粗糙随机场上的趋势冒充前者的下降门。

### Clover 方向与流能量归一化

PyQCD 的 Lorentz/链接方向标签为 `0=x, 1=y, 2=z, 3=t`，规范场数组轴为
`(t,z,y,x)`，故方向到数组轴的映射是 `axis=3-direction`。核对
`plaquette_clover` 时必须使用这一约定；不得沿用其曾出现过的
`0=t, 1=z, 2=y, 3=x` 错误标签。

`flow_action_density` 先把四叶 Clover 场强投影到 SU(3) 无迹部分，

\[
F^{\mathrm{su(3)}}_{\mu\nu}
=F_{\mu\nu}-\frac{\operatorname{Tr}F_{\mu\nu}}{3}I,
\qquad
E=\sum_{\mu<\nu}\operatorname{Tr}\!\left[
F^{\mathrm{su(3)}}_{\mu\nu}F^{\mathrm{su(3)}}_{\mu\nu}\right].
\]

这里已经只对 $\mu<\nu$ 求和，不再额外乘 $1/4$。归一化修复前保存的
$E$、$t^2E$ 和由目标交点得到的 $t_0$ 均受影响；$t_0$ 是非线性交点，不能用一个
常数因子可靠换算。真实系综必须重新计算后才能用于物理结论。

## Stout、APE、HYP 的边界

| 方法 | 主要参数 | 不能据此推出 |
|---|---|---|
| Stout | `rho`、迭代数；仅更新空间方向 `0,1,2`，时间链接 `3` 不变 | 不能把迭代数直接当连续流时间 |
| APE | 混合系数、投影到 SU(3) | 不能假设与 Stout 数值相同 |
| HYP | 多层系数 | 不能跨层复用未说明的端点几何 |

PyQCD 真实系综有 `nstep=20,rho=0.12` 的约定，但新数据仍需记录实际值。HYP 与
Stout 的所有参数路径都返回与输入不共享内存的独立数组且不原地改写输入；包括
`alpha1=0` 和 `nstep=0` 的退化路径，其输出与输入数值相等但所有权独立。因此无需
为防止 smear 输出回写 raw 而预先 copy。求逆使用的 smeared 场和 TMD Wilson 线所用
的 raw/flowed 场若不同，仍须分别持有并在元数据中明确，否则结果没有可比性。

标准四维 HYP 的三级结构是算法定义，不是可互换的实现细节：

| 层 | 基链接 | 排除方向 | staple 数与权重 |
|---|---|---|---|
| $\bar V_{\mu;\nu\rho}$ | 原始 $U_\mu$ | $\nu,\rho$ | 2 条，$\alpha_3/2$ |
| $\widetilde V_{\mu;\nu}$ | 原始 $U_\mu$ | $\nu$ | 4 条，$\alpha_2/4$ |
| $V_\mu$ | 原始 $U_\mu$ | 无 | 6 条，$\alpha_1/6$ |

## 流化与涂抹验收

### 正确性硬门

1. Wilson flow、HYP 和 Stout 的每条输出链接都必须满足局域端点协变
   $S[U^G]_\mu(x)=G(x)S[U]_\mu(x)G^\dagger(x+\hat\mu)$；只检查
   $V^\dagger V\simeq I$ 不能取代这个门。
2. `wilson_flow(U, tau=0)` 必须严格返回 `U`；非整步的 `tau/eps` 必须以
   缩放后的实际步长到达请求流时间。“小 $\tau$ 比大 $\tau$ 更接近原场”
   不能代替零时间恒等检查。
3. 在小格点上检查 Clover $F_{\mu\nu}=-F_{\nu\mu}$、`Tr(F^2)` 的规范不变性/
   非负性，以及步长减半后的稳定性。若验收 Wilson 流的平滑作用，使用与生成元
   对应的 Wilson plaquette 作用量；不要要求粗糙随机场上的 Clover `E`
   在有限步长下必然单调。
4. HYP 额外检查 `alpha1=0` 严格回到原链接、`alpha2=alpha3=0` 退化为
   六 staple 的四维 APE，以及同时交换坐标轴/链接方向标签后的超立方协变性。
5. 对 HYP/Stout 检查输入内容在调用前后不变，且所有参数路径的输出均不与输入共享
   内存；对 `alpha1=0` 与 `nstep=0` 还要同时检查输出数值严格等于输入。

### 诊断与下游交接

- 分别按代码定义检查 $E(\tau)$ 与 $\tau^2E(\tau)$；趋势相反不一定是 bug，
  混用二者才是错误。HYP–flow 相关系数只是依赖参数的比较诊断，其符号或
  `abs(r)>0.9` 都不是正确性放行标准。
- 检查相邻流时间的连续性；异常点保留 raw/intermediate，不向下游静默传递。
- 流化场作为 TMD 输入时，转交 `pyqcd-tmd-algorithm` 的端点、路径、颜色闭合和
  最终标量规范不变门；本 reference 不复制 TMD staple 几何。
