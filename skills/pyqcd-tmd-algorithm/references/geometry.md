# TMD geometry reference — flow, field strength, and staple

## 适用范围

任务涉及梯度流场、Clover/对偶场强、有限 staple、横向位移、表示选择、正负方向或
直线 OPE 对照时读取。本文件只定义几何和算符层，不负责核子统计、重整化或外推。

## 坐标与量纲契约

先固定纵向 `z`、横向向量 `b_perp`、staple 长度 `ell`、流时间 `tau`、动量 `Pz`、
表示和 Lorentz 通道，并把它们写进每个产物元数据。当前低层接口常使用格点整数
`tau/z/b`，而 `quasi_tmd_pdf` 的 `z_grid` 用 fm、`pz_gev` 用 GeV；内部转换后
`nu=z*Pz` 必须无量纲。

中心几何可取

\[
x_1=+z\hat z/2+\boldsymbol b_\perp,\qquad x_2=-z\hat z/2,
\]

而当前部分矩阵代码使用从 `x` 到 `x+z\hat z+\boldsymbol b_\perp` 的锚定几何。
两者不能在同一数据集混用；周期绕回、正负方向和 Fourier 符号必须显式记录。

## 流化场与算符

Wilson flow 的格点右端可写为

\[
\dot V_\mu(x)=Z_\mu[V](x)V_\mu(x),\qquad
Z_\mu=P_{\rm ah}[\Omega_\mu V_\mu^\dagger],
\]

当前 `wilson_flow` 使用三阶 RK：

\[
\begin{aligned}
W_1&=e^{\epsilon Z_0/4}W_0,\\
W_2&=e^{(8\epsilon Z_1/9-17\epsilon Z_0/36)}W_1,\\
V_{t+\epsilon}&=e^{(3\epsilon Z_2/4-8\epsilon Z_1/9+17\epsilon Z_0/36)}W_2.
\end{aligned}
\]

在同一个 `V_tau` 上构造 Clover `F`、对偶 `F_tilde` 和场强双点：

\[
\mathcal O^{g,\tau}_{\mu\nu;\rho\sigma}=
G^a_{\mu\nu}(x_1;\tau)
[W_{\rm st}^{\mathcal R}(x_1,x_2;\ell)]^{ab}
G^b_{\rho\sigma}(x_2;\tau).
\]

实现次序固定为：

1. 每个 `tau` 从同一原始 `U` 出发，缓存 `V_tau`；不得混接不同流时间的场强与线。
2. 选定表示：伴随线要验证
   `W_adj[a,b]=2*Tr(T[a] W_fund T[b] W_fund†)`；基本表示要保留带色迹的矩阵结构。
3. 以显式路径构造纵向段、横向段、回程段和转角，分别保存 `M` 分量，最后才做投影。
4. `O=M^{tx;tx}+M^{ty;ty}-2M^{xy;xy}` 可作为非极化起点，但不能代替完整张量混合矩阵。

## 当前接口边界

`staple_wilson_line` / `gluon_tmd_operator` 当前使用整数 `b_perp` 和 `L`，默认
`L=None` 时通常令 `L=z`。旧实现的 `_path_product` 曾以 `start=end` 调用横向段，
并以 `forward=False, start=0, end=z+L` 调用回程段；按其 `range` 逻辑两段可能为空，
因此输出形状不能证明 staple 几何正确。任何生产使用前都必须逐段检查起止点和链长。

还需核对 `M_mu_lambda_nu_rho` 的颜色收缩顺序；若形成
`Tr(F_nu W† W F_mu_shift)`，幺正线会在代数上抵消，必须重新验证端点与颜色指标。
`operator.staple_operator` 是另一条实现，先做逐点几何对照，再选唯一生产路径。

`gluon_ope_operator_z0` 是直线 OPE（支持 `±z` 和交叉 Lorentz 对），
`gluon_ff_operator_z0` 是固定规范、无 Wilson 线的对照；二者都不能替代非零
`b_perp` 的规范不变空间 staple TMD。

## 几何验证门

| 检查 | 必须看到的证据 |
|---|---|
| 流 | `V†V≈I`、减小步长后稳定、`flow_action_density` 按声明的定义满足预期 |
| 场强 | Clover 反对称、对偶定义和 Lorentz 指派一致 |
| 路径 | 每段链长/端点正确；反向路径与共轭关系成立；`W W†≈I` 只作幺正性检查 |
| 极限 | `b_perp→0` 回到直线基准，`z→0` 为局部极限，独立 `ell` 可区分 |
| 规范性 | 规范变换前后最终标量不变，颜色指标闭合 |

未通过几何门时，结果只能标为“接口/测试骨架”，不能进入软因子、CS 或 PDF 解释。
