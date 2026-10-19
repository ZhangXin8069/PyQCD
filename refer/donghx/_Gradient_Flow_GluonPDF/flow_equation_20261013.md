# Blending 框架中的 flow equation、flowed basis、generalized vertex 与随机权重

本文记录当前 quark PDF 的四维 gradient-flow 实现，以及 Blending 随机压缩基底的权重约定。核对对象包括 L32x96 的 Blending 代码、L32x64 的生产副本、原始压缩/噪声构造代码和现有 Note。

本文区分三个层次：**数学恒等式**、**当前代码实际执行的处方**、**仍需验证的随机估计假设**。代码与旧流程一致，不等于已经独立证明其所有有限流时间随机估计都无偏。

## 1. 首先回答：权重是否随时间变化？

### 1.1 当前代码的确定事实

当前 `create_omega(nev)` 只根据扇区维数、各扇区保存的向量数和模的排列，生成一张实对称的二指标权重表。它没有格点时间或 flow time 参数。因此在同一个 ensemble、同一套抽样处方中，代码执行的是

$$
\Omega^{\mathrm{code}}_{mn}(t_J;\sigma_a,\sigma_b;t_f,z,i)
=\omega^{(2)}_{mn}.
$$

也就是说，**当前代码中的数值权重不随当前时间片、两个原始时间片或流时间变化**。不同时间片的向量、规范场、perambulator 和 vertex 都会变化；不能把权重表固定理解成这些张量不变。

执行位置为

```python
omega = create_omega(config.nev_current)
vertex = generalized_vertex(left, right, gauge_t, direction, config.delta_z)
vertex *= omega[None, None]
```

`basis` 阶段不乘这个二指标权重；`vertex` 阶段在几何收缩完成后逐元素乘一次。后续读取已经加权的 vertex 时不应再次乘权重。

### 1.2 物理和统计上的限定

不随流时间变化的原因是：flow 是对已经抽出的向量施加确定性的线性映射，不重新抽样，不改变原始入选概率。流后的范数下降、模之间出现重叠、时间方向扩散，都不是重新计算抽样权重的理由。

但是，四维 flow 使两个端点可以来自不同原始时间片。一般应保留联合抽样权重的完整标签

$$
\Omega_{(\sigma_a,m),(\sigma_b,n)}
\equiv\Omega_{mn}(\sigma_a,\sigma_b).
$$

在**抽样处方固定**时，它可以与输出时间和流时间无关，但仍可能因两个原始时间片的联合抽样关系而不同：

$$
\frac{\partial\Omega}{\partial t_f}=0,
\qquad
\Omega_{mn}(\sigma_a,\sigma_b)
\text{ 不一定等于一张对所有时间片组合通用的 }\omega^{(2)}_{mn}.
$$

这里的导数只表达“沿同一条固定抽样的流轨迹不重新加权”，不适用于每个流时间重新抽样、截断或改变基底的另一种算法。

**待确认：当前对全部时间片组合使用同一张二指标权重表，是否匹配真实输入基底的跨时间随机联合分布。** 这是本次审查发现的统计验证缺口，不是已确认的数值错误，也不是修改现有生产权重的授权。零流时间回归无法单独排除这个问题，原因见第 9 节。

## 2. 符号、时间和坐标约定

| 符号 | 定义 |
| --- | --- |
| $t_f$ | 有量纲的流时间，量纲为长度平方 |
| $\tau=t_f/a^2$ | 无量纲流时间，提交脚本的 `FLOW_TIME` |
| $\epsilon=\Delta\tau$ | 无量纲积分步长，当前生产通常为 $0.01$ |
| $t_J$ | 流算符插入的欧氏格点时间片 |
| $\sigma_a,\sigma_b$ | 两个 flowed basis 的原始核源时间片 |
| $t_{\mathrm{src}},t_{\mathrm{snk}}$ | 核子源、汇的欧氏时间片 |
| $m,n$ | 保存的压缩/随机基底模指标，不包含时间标签 |
| $r,s$ | 原始抽样扇区标签 |
| $d_r,n_r$ | 扇区完整空间维数与实际保存的样本数 |
| $V_n(\boldsymbol x;\sigma)$ | 未进行 fermion flow 的原始时间片基底 |
| $U_\mu(t_f,x)$ | 从 thin links 演化得到的 flowed gauge links |
| $B_\mu(t_f,x)$ | 对应的连续流规范场 |
| $K(t_f)$ | 四维费米子热流核 |
| $\Phi_{\sigma n}^{(t_f)}$ | 对时间片嵌入向量施加 $K$ 的结果 |
| $\mathcal T_f$ | flavor $f$ 的普通 perambulator；不用 $\tau_f$，以免与流时间混淆 |
| $J^{(t_f)}$ | 未乘随机权重的 generalized vertex |
| $\widetilde J^{(t_f)}$ | 已经乘随机权重的生产 vertex |
| $\odot$ | Hadamard 逐元素乘法，不是矩阵乘法 |

特别注意：代码路径 `basis/tau000/` 中的 `tau` 是**原始时间片**，即本文的 $\sigma$；它不是路径标签 `tf0p50000000` 所表示的流时间。

张量约定为

```text
lattice size:        (Lx, Ly, Lz, Lt)
original/flowed V:   (mode, z, y, x, color)
gauge slice:        (direction, z, y, x, color_out, color_in)
gauge direction:    0=x, 1=y, 2=z, 3=t
vertex block:       (minus/plus, delta_z, left_mode, right_mode)
vertex block shape: (2, 21, 400, 400) for the current production setting
```

数组上的空间平移用 $(z,y,x)$ 顺序；物理方向仍按 $(x,y,z)$ 命名。所有离散内积采用代码的格点求和约定，不在 vertex 中另插入 $a^3$ 或除以空间体积。若将量改写为有量纲连续归一化，必须同步转换整条链，不能只改一个内积。

## 3. 原始 Blending 基底与普通 perambulator

### 3.1 从精确 LapH 到随机压缩基底

在一个原始时间片 $\sigma$ 上，精确 Laplacian 本征向量可以按

$$
-\nabla_{3\mathrm D}^2(\sigma)u_k(\sigma)
=\lambda_k(\sigma)u_k(\sigma),
\qquad
u_k^\dagger u_l=\delta_{kl}
$$

定义。Blending 保存的 $V_n$ 并不全是这些精确本征向量，而是精确低模、指定谱扇区中的随机正交组合，以及剩余高模空间中的噪声向量。

例如，在一个包含 $d_r$ 个精确模的扇区中，可以写

$$
V_{r\alpha}(\sigma)
=\sum_{k\in\mathcal H_r(\sigma)}
u_k(\sigma)C^{(r)}_{k\alpha}(\sigma),
\qquad
\alpha=1,\ldots,n_r,
$$

其中

$$
\bigl(C^{(r)}\bigr)^\dagger C^{(r)}=I_{n_r}.
$$

这说明样本向量可以正交归一，但不说明它们逐样本完备：

$$
V^\dagger(\sigma)V(\sigma)=I_{N_v},
\qquad
V(\sigma)V^\dagger(\sigma)\ne I_{N_xN_yN_zN_c}.
$$

第一式是原始构造的目标，需要用绝对值 Gram 残差检查实际文件；现有 `noise.check()` 的部分分支使用非绝对值比较，不能单靠它证明正交精度。

### 3.2 基底用过 smearing 与 flow 初值用 thin links 是两件事

L32x96 的原始基底标签包含 stout；L32x64 的原始标签含 `hpy`。这描述历史源/汇基底的构造，不意味着当前算符的 gradient flow 从 smeared gauge 开始。

当前算符流、flowed endpoints 与 Wilson line 使用同一套 thin-link 流轨迹。原始基底则必须与读取的普通 perambulator 使用同一个定义、模排列和截断；不能把两个来源的基底或 perambulator 交叉使用。

### 3.3 普通 perambulator 的定义

抑制显式 spin 指标时，普通 perambulator 是

$$
\mathcal T_f(t,\sigma)
=V^\dagger(t)D_f^{-1}(t,\sigma)V(\sigma).
$$

它保留实际 Dirac 传播和原始作用量的信息，不是正交矩阵，不要求

$$
\mathcal T_f^\dagger\mathcal T_f=I.
$$

当前 C2 和核子源/汇保持未流；flow 施加在插入算符的两个费米子端点和 Wilson line 上。在已有所需全时间 perambulator 块的前提下，这不要求重新执行 Dirac 求逆。

## 4. 随机权重从哪里来？

### 4.1 一个指标：补偿抽样概率

先用一个可以严格推导的有限总体抽样模型解释权重。假设扇区中有 $d_r$ 个正交方向，均匀地不放回抽取其中 $n_r$ 个。每个方向的入选概率为

$$
p_r=\frac{n_r}{d_r},
\qquad
w_r=\frac{1}{p_r}=\frac{d_r}{n_r}.
$$

条件于这一完整正交框架，对随机子集取平均有

$$
\mathbb E_\eta\left[
\frac{d_r}{n_r}\sum_{\alpha\in\mathrm{sample}_r}
v_\alpha v_\alpha^\dagger
\right]
=\sum_{k\in\mathrm{population}_r}v_kv_k^\dagger
=P_r.
$$

更一般的随机正交组合可以拥有相同的一阶矩，但必须检查其实际分布；正交性本身不推出这个期望恒等式。

### 4.2 两个指标：同一抽样中的重复与不重复

如果两个不同方向来自同一扇区、同一次不放回抽样，则联合入选概率为

$$
p_{r,\mathrm{distinct}}
=\frac{n_r(n_r-1)}{d_r(d_r-1)}.
$$

所以逆概率权重为

$$
\omega^{(2)}_{mn}
=\begin{cases}
\displaystyle\frac{d_r}{n_r},
&r(m)=r(n)=r,\quad m=n,\\[6pt]
\displaystyle\frac{d_r(d_r-1)}{n_r(n_r-1)},
&r(m)=r(n)=r,\quad m\ne n,\\[6pt]
\displaystyle\frac{d_r}{n_r}\frac{d_s}{n_s},
&r(m)=r,\quad r(n)=s,\quad r\ne s.
\end{cases}
$$

更明确地，令 $I_\alpha\in\{0,1\}$ 表示完整正交框架中的方向 $\alpha$ 是否被抽中，$p_{\alpha\beta}=\mathbb E[I_\alpha I_\beta]$。对条件于该框架、且不另外依赖抽样子集的系数或张量 $F_{\alpha\beta}$，有

$$
\mathbb E_\eta\left[
\sum_{\alpha,\beta}
\frac{I_\alpha I_\beta}{p_{\alpha\beta}}F_{\alpha\beta}
\right]
=\sum_{\alpha,\beta}F_{\alpha\beta}.
$$

当 $\alpha=\beta$ 时，$I_\alpha^2=I_\alpha$，所以只需要一次逆概率，而不是其平方；当 $\alpha\ne\beta$ 时才使用二者的联合概率。若 $F$ 内部还含同一抽样的其他随机线，则不能把它视为与子集无关，上式必须扩展到完整联合事件。

最后一行还要求相应两个扇区的抽样独立或联合概率确实因子化。精确保留扇区满足 $d_0=n_0$，权重为 $1$。

这正是当前 `create_omega()` 实现的分块公式。它是一张二指标联合权重，不能普遍写成两个端点标量权重的乘积。

例如 $d_r=50,n_r=10$ 时，

$$
\omega^{(2)}_{mm}=5,
\qquad
\omega^{(2)}_{m\ne n,\mathrm{same\ sector}}
=\frac{50\times49}{10\times9}=27.222\ldots,
$$

而两个独立的、相同维数扇区的乘积权重为

$$
w_rw_s=5\times5=25.
$$

把 $\sqrt{w_r}$ 分别吸收到两端 basis 中，不能同时重现上述三种二指标情况。

### 4.3 必须说明的适用范围

上面的逆概率推导在指定的有限总体抽样模型下严格成立；它解释了代码公式，但**不是对当前全部 V3/V4 压缩、高模噪声和核子三条线联合估计的完整无偏性证明**。

当前可读的 `compress_matrix_V4` 含先生成完整正交框架再抽样的步骤；部分 V3 和高模流程则采用逐个随机生成并正交化。不同实现的一阶、二阶和更高阶矩不能仅由最终向量正交而自动等同。

特别是核子 C3 还包含 source/sink VVV 和多条 perambulator；它们若重用同一随机基底，可能需要更高阶联合矩。本文将“维持旧定义”和“已证明完整随机估计无偏”分开记录。

## 5. 两个 ensemble 的实际扇区和权重不同

当前两个分支都用 $N_v=400$ 个 current 模：100 个精确模、100 个压缩模、200 个高模噪声向量。压缩参数均为

$$
N_{\mathrm{sum}}=(20,20,20,20,20),
\qquad
N_{\mathrm{extract}}=(10,10,10,10,10).
$$

因此每个原始压缩组又分成两个抽样子扇区，每个子扇区保存 10 个向量。权重需要按这十个子扇区计算，不能只按五个大组计算。

| 项目 | L32x96 | L32x64 |
| --- | --- | --- |
| 压缩组完整维数 | $(100,200,400,600,800)$ | $(100,200,400,700,1100)$ |
| 每组两个子扇区的维数 | $(50,100,200,300,400)$ 各重复两次 | $(50,100,200,350,550)$ 各重复两次 |
| 每个压缩子扇区样本数 | $10$ | $10$ |
| 完整单时间片色空间维数 | $32^3\times3=98304$ | $32^3\times3=98304$ |
| 高模扇区维数 | $98304-100-2100=96104$ | $98304-100-2500=95704$ |
| 实际采用高模向量数 | $200$ | $200$ |
| 高模对角权重 | $96104/200=480.52$ | $95704/200=478.52$ |

L32x96 文件标签中的 `n400` 不能直接作为当前权重的高模样本数：当前 400 模读取范围只取 100 精确、100 压缩、200 高模。L32x64 标签记录 `n200`。必须同时核对输入标签、实际读取范围和 `create_omega()`。

这些扇区维数由原始抽样定义确定，不随时间片上的数值本征值变化；flow 后高模被抑制，也不意味着高模扇区维数或其抽样概率改变。

## 6. 从 flow equation 推导 flowed basis

### 6.1 连续流与离散 Laplacian

采用反厄米连续规范势约定，基本表示的协变导数为

$$
D_\mu^{\mathrm F}\chi=(\partial_\mu+B_\mu)\chi,
$$

而对李代数值场的伴随导数为

$$
D_\mu^{\mathrm{adj}}X=\partial_\mu X+[B_\mu,X].
$$

流方程为

$$
\begin{aligned}
\partial_{t_f}B_\mu&=D_\nu^{\mathrm{adj}}G_{\nu\mu},
&B_\mu(0)&=A_\mu,\\
\partial_{t_f}\chi&=\Delta_{B(t_f)}\chi,
&\chi(0)&=\psi,\\
\partial_{t_f}\bar\chi&=\bar\chi\overleftarrow\Delta_{B(t_f)},
&\bar\chi(0)&=\bar\psi,
\end{aligned}
\qquad
\Delta_B=\sum_{\mu=1}^4D_\mu^{\mathrm F}D_\mu^{\mathrm F}.
$$

这里没有额外 flow-time gauge-damping 项。规范场流不受输入费米子向量反作用。

最近邻格点算符是

$$
\begin{aligned}
(a^2\Delta_U\phi)(x)
=\sum_{\mu=1}^{4}\bigl[
U_\mu(x)\phi(x+\hat\mu)
+U_\mu^\dagger(x-\hat\mu)\phi(x-\hat\mu)
-2\phi(x)\bigr].
\end{aligned}
$$

QUDA 使用四维跳跃、跳跃系数 $1$ 和对角系数 $-8$。这不是 Wilson/Clover Dirac 算符，不含质量项或 spin 投影。

### 6.2 热流核与时间片嵌入

定义

$$
\bigl[\partial_{t_f}-\Delta_{B(t_f)}^{(x)}\bigr]K(t_f;x,y)=0,
\qquad
K(0;x,y)=\delta_{xy}.
$$

因为规范场也随流时间演化，形式解为流时间排序指数

$$
K(t_f)=\mathcal P_f\exp\left[\int_0^{t_f}ds\,\Delta_{B(s)}\right],
$$

不能一般地用固定初始或最终规范场上的一个指数代替。

将原始 basis 嵌入四维：

$$
\eta_{\sigma n}(y)
=\delta_{y_4,\sigma}V_n(\boldsymbol y;\sigma).
$$

于是

$$
\begin{aligned}
\Phi_{\sigma n}^{(t_f)}(x)
&=\sum_yK(t_f;x,y)\eta_{\sigma n}(y)\\
&=\sum_{\boldsymbol y}
K(t_f;x;\boldsymbol y,\sigma)V_n(\boldsymbol y;\sigma).
\end{aligned}
$$

它满足

$$
\partial_{t_f}\Phi_{\sigma n}^{(t_f)}
=\Delta_{B(t_f)}\Phi_{\sigma n}^{(t_f)},
\qquad
\Phi_{\sigma n}^{(0)}(\boldsymbol x,t_J)
=\delta_{t_J,\sigma}V_n(\boldsymbol x;\sigma).
$$

**这个初值问题不含 $\omega$。** 压缩系数随格点源时间片变化，流核随规范场变化，但它们与抽样逆概率权重不是同一个对象。

### 6.3 为什么流后不正交，但不需要加入 Gram 逆矩阵？

令 $\mathcal V$ 的列为全部所需 $\eta_{\sigma n}$，则

$$
\Phi=K\mathcal V,
\qquad
G_\Phi=\Phi^\dagger\Phi
=\mathcal V^\dagger K^\dagger K\mathcal V\ne I.
$$

这是扩散核非幺正的正常后果。当前算法是在原始传播子表示中显式施加 $K$ 和 $K^\dagger$，不是把 $\Phi\Phi^\dagger$ 当成新的单位分解，所以不应为了“恢复正交”擅自插入 $G_\Phi^{-1}$。

若只是改变表示，写 $\Phi=QR$，则必须保留全部变换：

$$
\Phi A\Phi^\dagger=Q(RAR^\dagger)Q^\dagger.
$$

其中 $A$ 包含原有传播子系数及其随机估计结构。不能只保存 $Q$、丢弃 $R$，也不能默认二指标 Hadamard 权重在混合模后仍保持原样。

## 7. 实际 RK3 积分、边界与缓存

### 7.1 耦合规范场/费米子三阶段更新

记 $L[U]=a^2\Delta_U$，一步开始的场为 $U_0,\phi_0$。令 $\mathcal Z[U]$ 为 QUDA 对应约定下的反厄米 Wilson 力生成元，则

$$
\begin{aligned}
k_0&=L[U_0]\phi_0,\\
\phi_1&=\phi_0+\frac\epsilon4k_0,\\
U_1&=\exp\left[\frac\epsilon4\mathcal Z[U_0]\right]U_0,
\end{aligned}
$$

$$
\begin{aligned}
k_1&=L[U_1]\phi_1,\\
\phi_2&=\phi_0+\epsilon\left(\frac89k_1-\frac29k_0\right),\\
U_2&=\exp\left[\epsilon\left(
\frac89\mathcal Z[U_1]-\frac{17}{36}\mathcal Z[U_0]
\right)\right]U_1,
\end{aligned}
$$

$$
\begin{aligned}
k_2&=L[U_2]\phi_2,\\
\phi_{\mathrm{new}}&=\phi_1+\frac{3\epsilon}{4}k_2,\\
U_{\mathrm{new}}&=\exp\left[\epsilon\left(
\frac34\mathcal Z[U_2]-\frac89\mathcal Z[U_1]
+\frac{17}{36}\mathcal Z[U_0]
\right)\right]U_2.
\end{aligned}
$$

三次 Laplacian 必须分别使用对应的中间规范场。代码调用 `performGFlowQuda`，不是 Python 的 Euler 迭代；每一小步都不重新归一化 basis，也不更新抽样权重。

### 7.2 参数、多 checkpoint 与 spin 装填

当前常用

$$
\epsilon=0.01,
\qquad
N_{\mathrm{step}}=\operatorname{round}(\tau/\epsilon).
$$

单流时间模式对每批原始向量从零开始；多 checkpoint 模式在同一批内同时接续 flowed gauge 和 flowed spinor：

$$
K_{0\to\tau_2}
=K_{\tau_1\to\tau_2}K_{0\to\tau_1}.
$$

这里 $K_{\tau_1\to\tau_2}$ 必须沿接续的同一规范场轨迹计算，不能当作从初始规范场重新开始的 $K_{0\to\tau_2-\tau_1}$。

因为 Laplacian 对 Dirac spin 对角，四个 basis 模装入一个 spinor 的四个分量，批量流动后再拆开。`FLOW_MODE_BATCH=32` 时使用 8 个 spinor。保存的 basis 形状为 `(400,32,32,32,3)`，通常为 `complex64`。

### 7.3 反周期边界与时间窗口

费米子时间反周期条件通过边界时间 link 相位实现；保存用于 Wilson line 的 flowed gauge 时移除这层相位。规范场流仍是原来的周期规范场流。

流积分在完整四维格子上进行；时间窗口只截取输出 basis：

$$
d_{N_t}(t_J,\sigma)
=\min_{k\in\mathbb Z}|t_J-\sigma+kN_t|\leq w.
$$

窗口外的平方范数损失记为

$$
f_{\mathrm{discard}}(\sigma,t_f)
=1-
\frac{\sum_{t_J\in\mathrm{window},n,\boldsymbol x,c}
|\Phi_{\sigma n}^{(t_f)}(\boldsymbol x,t_J,c)|^2}
{\sum_{t_J,n,\boldsymbol x,c}
|\Phi_{\sigma n}^{(t_f)}(\boldsymbol x,t_J,c)|^2}.
$$

这是输出截断诊断，不是抽样概率，也不能通过调整 $\omega$ 补偿。当前 L32x64 生产要求其不超过 $10^{-6}$；这个聚合范数阈值本身不等于每个模或每个 C3 的相对误差上界。

## 8. 从 flowed bilocal 推导 generalized vertex

### 8.1 算符定义与 Wilson line 方向

位移方向为 $i\in\{x,y,z\}$。取所有算符端点位于同一个输出时间片 $t_J$，定义空间求和的 bilocal

$$
O_\Gamma^{\mathrm{GF}}(t_J,z,i;t_f)
=\sum_{\boldsymbol x}
\bar\chi(t_f,\boldsymbol x,t_J)
\Gamma W_i(t_f;\boldsymbol x,\boldsymbol x+z\hat i;t_J)
\chi(t_f,\boldsymbol x+z\hat i,t_J).
$$

代码位移整数记为 $\ell=z/a$。对于 $\ell>0$，链为从左到右按路径顺序的矩阵乘积

$$
W_i(\boldsymbol x,\boldsymbol x+\ell\hat i;t_J)
=\prod_{q=0}^{\ell-1}
U_i(t_f;\boldsymbol x+q\hat i,t_J).
$$

负位移取反向链

$$
W_i(\boldsymbol x,\boldsymbol x-\ell\hat i;t_J)
=\left[
W_i(\boldsymbol x-\ell\hat i,\boldsymbol x;t_J)
\right]^\dagger,
\qquad\ell>0.
$$

右端向量在代码中用 `roll(right,-signed_dz,axis)` 实现相应坐标取值。`sign=0` 保存负位移，`sign=1` 保存正位移；两者在零位移保存相同结果。

### 8.2 将流核插入两个端点

把上述 bilocal 的位置、色、时间片操作整体记为 $\mathscr O(t_J,z,i;t_f)$。费米子流使其在原始场空间中变成

$$
K^\dagger(t_f)\mathscr O(t_J,z,i;t_f)K(t_f).
$$

再投影到原始 basis，得到

$$
J^{(t_f)}(t_J;\sigma_a,\sigma_b;z,i)
=\mathcal V_{\sigma_a}^\dagger
K^\dagger\mathscr O K\mathcal V_{\sigma_b}.
$$

由于 $K\mathcal V_\sigma=\Phi_\sigma$，矩阵元为

$$
\begin{aligned}
J_{mn}^{(t_f)}(t_J;\sigma_a,\sigma_b;z,i)
=\sum_{\boldsymbol x,c,d}
&\left[\Phi_{\sigma_a m}^{(t_f)}(\boldsymbol x,t_J,c)\right]^*\\
&\times W_i^{cd}(t_f;\boldsymbol x,\boldsymbol x+z\hat i;t_J)\\
&\times\Phi_{\sigma_b n}^{(t_f)}(\boldsymbol x+z\hat i,t_J,d).
\end{aligned}
$$

这就是 `generalized_vertex(left,right,gauge_t,...)` 计算的**未加权**矩阵。端点的 spin 结构由后续 $\Gamma$ 投影处理，因此这个 vertex 本身不含 spin 指标。

### 8.3 为什么必须保留两个原始时间片？

在零流时间，

$$
J_{mn}^{(0)}(t_J;\sigma_a,\sigma_b;z,i)
=\delta_{\sigma_a,t_J}\delta_{\sigma_b,t_J}
\left[V^\dagger(t_J)W_i(z;t_J)V(t_J)\right]_{mn}.
$$

在正流时间，热核沿时间方向扩散，一般有

$$
J^{(t_f)}(t_J;\sigma_a,\sigma_b)\ne0
\quad\text{即使 }\sigma_a\ne t_J\text{ 或 }\sigma_b\ne t_J.
$$

因此 C3 需要两个原始时间片的和，而不能只使用 $\sigma_a=\sigma_b=t_J$。

### 8.4 加权 vertex 与权重移动的代数

一般写

$$
\widetilde J_{mn}^{(t_f)}(t_J;\sigma_a,\sigma_b;z,i)
=\Omega_{mn}(\sigma_a,\sigma_b)
J_{mn}^{(t_f)}(t_J;\sigma_a,\sigma_b;z,i).
$$

当前代码取 $\Omega_{mn}(\sigma_a,\sigma_b)=\omega^{(2)}_{mn}$，即对每个时间片块使用相同的表。

在一个已经定义好索引结构的收缩中，权重可在同一对求和指标之间移动。例如

$$
\sum_{m,n}\omega_{mn}J_{mn}X_{nm}
=\operatorname{Tr}\!\left[(\omega\odot J)X\right]
=\operatorname{Tr}\!\left[J(\omega^{\mathsf T}\odot X)\right].
$$

当前 $\omega$ 为实对称矩阵，转置不改变它。这个恒等式解释“在 vertex 上乘权重”为什么可以作为原有索引收缩的实现选择；它**不证明**该权重就是任意新随机估计所需的正确联合权重。

尤其不能把 Hadamard 权重当作普通矩阵，写成 $\omega J$、$J\omega$ 或 $\omega^{1/2}J\omega^{1/2}$。

### 8.5 接回普通 perambulator

在指定的保留子空间及生产随机估计定义下，活动夸克线示意为

$$
\sum_{\sigma_a,\sigma_b}\sum_{m,n}
\mathcal T_f(t_{\mathrm{snk}},\sigma_a)_{\cdots m}
\Gamma\,\widetilde J_{mn}^{(t_f)}(t_J;\sigma_a,\sigma_b)
\mathcal T_f(\sigma_b,t_{\mathrm{src}})_{n\cdots}.
$$

完整 C3 还要乘 spectator 线、核子源/汇 VVV、spin 投影和各 Wick 项符号。此式是连接结构的示意，不是省略随机完备性条件后宣称精确等于完整坐标传播子。

因子化优化先构造临时张量

$$
H_{uSLpamb}
=\sum_{t,n}\widetilde J_{tuSLnp}A_{tambn},
$$

对应 `tuSLnp,tambn->uSLpamb`，其中代码的 $t,u$ 是两个原始时间片，不是流时间。优化只是重排原有有限求和；不更改已加权 vertex，也不额外引入一份权重。

## 9. 关键统计问题：不同原始时间片是否应使用同一张权重？

### 9.1 格点时间平移不变，不等于联合抽样无差别

如果所有时间片采用相同的 $d_r,n_r$，一指标权重 $d_r/n_r$ 的数值自然相同。但“在同一时间片的同一次抽样中选两个方向”和“在两个时间片分别抽样”是不同的联合事件。

完整样本标签应是

$$
I=(\sigma_a,m),\qquad J=(\sigma_b,n).
$$

因此 $m=n$ 但 $\sigma_a\ne\sigma_b$ 不代表“同一个向量被用两次”。源时间片也是身份标签的一部分。

### 9.2 独立时间片的简单模型

若两个时间片的抽样相互独立，且所讨论的估计只需要两个独立一阶完备性插入，那么

$$
\mathbb E_\eta[\widehat P_r(\sigma_a)\otimes\widehat P_s(\sigma_b)]
=P_r(\sigma_a)\otimes P_s(\sigma_b),
\qquad\sigma_a\ne\sigma_b,
$$

由此得到候选权重

$$
\Omega^{\mathrm{ind}}_{mn}(\sigma_a,\sigma_b)
=w_{r(m)}w_{r(n)}.
$$

对于同一时间片、同一抽样，才使用第 4 节相应的二阶联合权重。在这个明确限定的模型中，可以写

$$
\Omega_{mn}^{\mathrm{model}}(\sigma_a,\sigma_b)
=\begin{cases}
\omega^{(2)}_{mn},&\sigma_a=\sigma_b,\\
w_{r(m)}w_{r(n)},&\sigma_a\ne\sigma_b.
\end{cases}
$$

这展示了**不依赖流时间、但依赖原始时间片相等关系**的可能性；它不是要求立即将生产代码改成这一公式。若时间片之间共享噪声、具有时间 dilution、或者相同噪声还进入核子源/汇，必须从实际完整联合分布重新推导。

### 9.3 当前做法和独立模型的具体差异

在 $d_r=50,n_r=10$ 的例子中，对两个不同原始时间片：

$$
\begin{array}{c|c|c}
&\text{当前统一二指标表}&\text{独立时间片简单模型}\\
m=n&5&25\\
m\ne n\text{ 且同扇区}&27.222\ldots&25
\end{array}
$$

可见数值上不能假设差异必然很小；但其对实际 C3 和拟合矩阵元的影响，必须重建并比较完整收缩才能判断，本次未测量。

### 9.4 已找到的证据与还缺少的证据

已确认：

- 当前两个 ensemble 的 vertex 生成函数对所有时间片块使用同一张 $\omega^{(2)}$。
- 原始 L32x96 非流代码只在当前时间片 vertex 上乘该权重。
- 可读的压缩/基底生成脚本在时间片循环中调用随机压缩或噪声生成，表明跨时间联合分布必须纳入审查。
- 当前 L32x64 生成脚本使用 V4 压缩；L32x96 可读历史压缩脚本包含 V3。不能不加区分地宣称它们具有同一随机分布。

待确认：

- 实际生产输入文件对应哪一版冻结的随机生成代码、种子策略和抽样日志。
- 时间片之间是否严格独立，是否存在共享随机系数、时间分块或其他关联。
- 包含 source/sink VVV 及 spectator 线的完整核子估计，正确的联合权重应为何种阶数、如何处理重合时间和模标签。
- 当前统一二指标表与经独立推导的时间标签权重之间，对实际有限流 C3 的差异。

**零流回归只保留 $\sigma_a=\sigma_b=t_J$ 的块，所以它不能验证正流时间新出现的跨时间块权重。** 因子化对直接收缩的回归也只能证明两种程序实现同一个估计，不能独立证明这个估计无偏。

## 10. 为什么 flow 本身不要求重新加权？

设在给定规范场、指定噪声处方下，原始估计已经满足

$$
\mathbb E_\eta[\widehat X]=X.
$$

因为流核在给定规范场下不依赖抽样噪声，

$$
\mathbb E_\eta[K\widehat X K^\dagger]
=K\,\mathbb E_\eta[\widehat X]K^\dagger
=KXK^\dagger.
$$

对两个独立线性端点或相应高阶估计也可逐指标作同样推送，但前提是其原始联合估计已经正确。

例如原始扇区的一阶随机完备性在流后变成

$$
\mathbb E_\eta\left[
\frac{d_r}{n_r}\sum_{\alpha\in\mathrm{sample}_r}
(Kv_\alpha)(Kv_\alpha)^\dagger
\right]
=KP_rK^\dagger.
$$

右边本来就是流后的算符，不是一个要求恢复为 $P_r$ 或 $I$ 的正交投影。原来的 $d_r/n_r$ 保持不变，而 $K$ 的阻尼完整保留。

因此正确表述是：**固定原始抽样估计后，线性 flow 不改变它的抽样权重；若四维实现引入了旧算法没有的时间标签组合，则需要验证原始联合估计在这些标签上的定义，而不是按 flow 后的范数去修正权重。**

如果选择以逐模缩放改变表示，令

$$
\Phi_m=s_m\Phi'_m,
$$

那么

$$
J_{mn}=s_m^*s_nJ'_{mn}.
$$

保留 $s_m^*s_n$ 是表示变换的补偿，不是抽样权重的新流时间依赖。若直接丢掉它，再沿用原权重，就改变了流算符。

Ringed normalization 的 $Z_\chi(t_f)$ 是另一个独立的场归一化。双线性流算符整体乘一份 $Z_\chi$；它不替代 $\omega$，也不能吸收任意二指标联合抽样差异。

## 11. 零距离、共轭与零流检查

### 11.1 零距离不是正流时间的单位矩阵

未加权 vertex 在 $z=0$ 时为

$$
J_{mn}^{(t_f)}(t_J;\sigma_a,\sigma_b;0)
=\sum_{\boldsymbol x,c}
\left[\Phi_{\sigma_a m}^{(t_f)}(\boldsymbol x,t_J,c)\right]^*
\Phi_{\sigma_b n}^{(t_f)}(\boldsymbol x,t_J,c).
$$

这是一个输出时间片上的 Gram 块。完整四维 Gram 还要对 $t_J$ 求和；已经乘过 $\omega$ 的生产 vertex 不是原始 Gram 矩阵。

### 11.2 共轭关系

在当前无额外空间相位的几何 vertex 约定下，交换两端并反转位移，原始矩阵满足

$$
J(t_J;\sigma_a,\sigma_b;z)^\dagger
=J(t_J;\sigma_b,\sigma_a;-z).
$$

加权后要保持这一关系，需要

$$
\Omega_{mn}(\sigma_a,\sigma_b)^*
=\Omega_{nm}(\sigma_b,\sigma_a).
$$

当前实对称统一表满足这项代数要求，但满足共轭性仍不能替代抽样无偏性证明。

### 11.3 零流极限

只有 $\sigma_a=\sigma_b=t_J$ 的块存活，未加权 $z=0$ vertex 应回到原基底 Gram；若原始向量正交，则为单位阵。

已加权 vertex 则为

$$
\widetilde J^{(0)}(t_J;t_J,t_J;z)
=\omega^{(2)}\odot[V^\dagger(t_J)W(z;t_J)V(t_J)].
$$

因此即使原始基底严格正交，加权零距离项的对角也是 $\omega^{(2)}_{mm}$，不能误检为必须等于 $1$。

## 12. 代码、缓存入口和已经完成的本次核对

### 12.1 直接相关代码

| 内容 | 文件入口 |
| --- | --- |
| L32x96 原始/单流代码 | `/public/group/lqcd/donghx/Gradient_Flow/Blending_L32x96/` |
| L32x96 多 checkpoint 后继 | `/public/home/donghx/Gradient_Flow/Blending_L32x96/` |
| L32x64 当前扩展任务副本 | `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/` |
| L32x64 基线代码 | `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_flow_20261007/reference/` |
| 理论 Note | `/public/home/donghx/Note/Gradient_Flow/main.tex` |

关键函数及已核对位置：

- [L32x96 权重定义](/public/home/donghx/Gradient_Flow/Blending_L32x96/flow_common_l32x96.py:441)：`create_omega`。
- [L32x64 权重定义](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/flow_common_l32x64.py:447)：`create_omega`。
- [L32x64 原始几何 vertex](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/flow_common_l32x64.py:532)：`generalized_vertex`。
- [L32x64 权重创建与时间片块循环](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/generate_flowed_vertex_l32x64.py:898)。
- [L32x64 实际逐元素加权](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/generate_flowed_vertex_l32x64.py:987)。
- [L32x96 原始 connected C3 的权重用法](/public/group/lqcd/donghx/Strange_PDF/Conneted_Diagram_final/L32x96/Calc_L32x96.py:2040)。
- [临时 flowed sequential 因子化](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/flowed_perambulator_factorization.py:99)。
- [PyQUDA 绑定](/public/home/donghx/PyQuda_a100_2608/PyQUDA-gflow-src/pyquda/src/quda.in.pyx:416)：`performGFlowQuda`。
- [QUDA 耦合 RK3](/public/home/donghx/PyQuda_a100_2608/quda/lib/interface_quda.cpp:5483)。
- [QUDA gauge RK 系数](/public/home/donghx/PyQuda_a100_2608/quda/include/kernels/gauge_wilson_flow.cuh:90)。
- [原始 L32x96 压缩脚本](/public/home/sush/EasyDistillation/work/compress_noise.py:30)。
- [可读 L32x64 基底生成脚本](/public/home/sush/EasyDistillation/work/eigenvectors.beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64.py:82)。
- [随机构造实现](/public/home/sush/EasyDistillation/work/noise.py:165)：`creat_noise`；同文件 `compress_matrix_V3`、`compress_matrix_V4`。

L32x96 当前代码的 `EIGEN_TAG` 和 `PERAM_ROOT` 是实际输入契约。历史可读脚本只是来源证据，不能据此宣称生产数据的全部随机种子与版本已追溯完成。

### 12.2 缓存语义

L32x64 的共同缓存根为

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_flow_20261007/flow_cache/
  CONF/
    thin_tf..._eps..._tw..._Nv400_dz21/
      basis/tauSSS/tJTTT.npy
      basis/tauSSS/done.json
      flowed_gauge/index.json
      vertex/tJTTT/taAAA_tbBBB_z.npy
      vertex/tJTTT/taAAA_tbBBB_z.json
```

最后的 `z` 是路径方向标签。basis 文件没有乘二指标权重；生产 vertex 文件已乘。vertex/VVV 属于可重建缓存，当前不长期保留；basis/gauge 缓存与最终 C2/C3 结果应保留。

现有 vertex 元数据的 `omega` 字段说明使用 reference `dim=2`，但没有完整记录扇区数组和跨时间联合处方。若以后调整权重，必须使用新 schema/新结果根，记录扇区参数、时间联合规则和旧新方案区别；不能让新代码静默读取旧缓存。

### 12.3 本次小型代码恒等式核对

本次只提取两个 `create_omega()` 的函数体，在 NumPy 中检查 400 维权重表与第 4 节分块公式。没有扫描大型 basis，没有重新计算 C3，没有在登录节点运行物理拟合。

结果：

$$
\begin{aligned}
\max|\omega_{\mathrm{code}}-\omega_{\mathrm{formula}}|
&=2.91\times10^{-11},\\
\max|\omega-\omega^{\mathsf T}|&=0.
\end{aligned}
$$

两个 ensemble 均满足上述检查。第一项绝对差来自较大权重的浮点运算舍入；这证明代码与所写公式一致，**不证明真实噪声分布或完整 C3 的无偏性**。

## 13. 后续最具体的验证步骤

1. 冻结并追溯实际输入 basis 的生成版本、种子与时间标签；先确认不同时间片是否独立，以及 source/sink 是否重用同一噪声。
2. 从一个完整 connected Wick 项出发，保留所有时间、模、spin 和噪声标签，推导需要的一阶、二阶或更高阶联合矩。不能直接把二指标权重用于所有三条线。
3. 在小维度可完全枚举的模型中，分别测试同一次抽样、独立时间片抽样、共享随机框架三种情况，比较精确全空间结果与随机平均。加入 $K^\dagger\mathscr O K$ 后再检查一次。
4. 在计算节点对一个组态、一个非零流时间，使用已有 basis/gauge 重建未加权 vertex；分别施加当前权重和已推导的候选时间标签权重，比较完整 C3。先做逐块/逐 Wick 项对比，再做最终 ratio，不能只看图形是否平滑。
5. 保留零流回归、正负位移共轭、因子化回归、步长减半与时间窗口扩大检查。跨时间权重检查是新增的独立验证，不能用这些已有代数回归替代。
6. 只有统计推导和小模型验证完成后，才决定是否更新生产权重；若需更改，一般可以复用 basis/gauge 和普通 perambulator，重新生成 vertex 并重做 C3/后续分析，不必仅因权重变化重做 flow 或 Dirac 求逆。

本次仅更新文档，未修改权重、basis、vertex、perambulator、生产脚本或已提交任务。

## 14. 不能随意改变的约定与结论

- 不混用 $t_f$、$\tau$、$t_J$ 与原始时间片 $\sigma$。
- thin-link flow、四维 Laplacian、费米子反周期条件和两端/Wilson line 共用规范场历史必须一致。
- 原始 basis、perambulator、扇区大小和模排列必须配套；两个 ensemble 的权重表不相同。
- flow 的非幺正性、范数下降不通过调整 $\omega$、强制归一化或 Gram 逆矩阵“修复”。
- 二指标随机权重只乘一次；不能作为普通矩阵乘，也不能普遍拆成两个端点标量。
- 当前固定表是已确认的代码行为；跨原始时间片的统计正确性仍是待确认项。
- 任何新的随机权重方案均需新版本和独立验证，不覆盖已完成生产结果。
- $Z_\chi$、GF-to-quasi 转换与 LaMET matching 均不是 Blending 抽样权重。

## 15. 参考和文档记录

耦合流算法的底层源码注明参考 M. Lüscher, *Chiral symmetry and the Yang–Mills gradient flow*, arXiv:1302.5246，Appendix D；规范场步骤参考 arXiv:1006.4518。本文具体参数和张量行为以实际阅读的本地代码为依据，随机无偏性未被这些 flow 文献自动保证。

现有 Note 的 [flow 积分算法](/public/home/donghx/Note/Gradient_Flow/main.tex:1415)、[原始随机基底](/public/home/donghx/Note/Gradient_Flow/main.tex:1806)、[generalized vertex](/public/home/donghx/Note/Gradient_Flow/main.tex:1862)、[传播子与非正交性](/public/home/donghx/Note/Gradient_Flow/main.tex:2092) 可联合阅读。Note 原有的“权重不随 flow 改变”应带上本文强调的固定联合抽样定义前提，不能被理解为跨时间权重已无条件验收。

更新前的 `flow_equation.md` 已备份为

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/handoff/flow_equation_before_omega_20261013.md
```

生成日期：2026-10-13（Asia/Shanghai）。

本次会话标题：**Blending 四维流的 basis/vertex 推导与时间标签随机权重审查**。
