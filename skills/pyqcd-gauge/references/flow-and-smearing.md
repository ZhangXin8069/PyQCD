# Wilson flow 与链接涂抹 reference

## 适用范围

本文件用于区分 Wilson flow 与 Stout/APE/HYP 等有限次涂抹，并核对 TMD 前置的流化
场。流化场只在通过几何和群性质检查后交给 `pyqcd-tmd-algorithm`。

## Wilson flow

连续形式可写为

\[
\dot V_\mu(x,\tau)=Z_\mu[V](x,\tau)V_\mu(x,\tau),
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
工程约定中的 $\tau=3a^2$ 不是所有任务的自动默认值，必须随数据显式保存。

## Stout、APE、HYP 的边界

| 方法 | 主要参数 | 不能据此推出 |
|---|---|---|
| Stout | `rho`、迭代数、方向数 | 不能把迭代数直接当连续流时间 |
| APE | 混合系数、投影到 SU(3) | 不能假设与 Stout 数值相同 |
| HYP | 多层系数 | 不能跨层复用未说明的端点几何 |

PyQCD 真实系综有 `nstep=20,rho=0.12` 的约定，但新数据仍需记录实际值；涂抹原地
修改输入，必须先 copy。求逆使用的 smeared 场和 TMD Wilson 线所用的 raw/flowed 场
若不同，须在元数据中明确，否则结果没有可比性。

## 流化场验收

1. 在小格点上检查 $V^\dagger V\simeq I$、Clover $F_{\mu\nu}=-F_{\nu\mu}$ 和
   步长减半后的稳定性。
2. 分别按代码定义检查 $E(\tau)$ 与 $\tau^2E(\tau)$；趋势相反不一定是 bug，
   混用二者才是错误。
3. 检查 $\tau\to0$ 的原场极限和相邻流时间的连续性；异常点保留 raw/intermediate，
   不向下游静默传递。
