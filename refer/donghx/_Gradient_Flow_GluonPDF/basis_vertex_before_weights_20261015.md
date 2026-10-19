# Flowed all-time basis 与 generalized vertex：定义、代码实现与验证边界

本文记录本项目 quark PDF 的四维 gradient-flow Blending/distillation 实现。重点是：原始时间片向量怎样嵌入四维、怎样计算热核作用、怎样构造带两个原始时间标签的 vertex，以及实际计算覆盖了什么、没有覆盖什么。

本文以 L32x64 已提交生产副本为主要源码依据，并对照 L32x96 实现和 `flow_equation.md`。它不是新算法说明，也不将实现一致性等同于完整物理或随机估计验证。全部公式采用 LaTeX；“确定事实”“数学关系”和“待确认问题”分开表述。

## 1. 最重要的结论

当前实现计算的是

$$
\boxed{\Phi(t_f)=K(t_f)\mathcal V,}
$$

以及

$$
\boxed{
J(t_J;\sigma_a,\sigma_b;z,i)
=\mathcal V_{\sigma_a}^{\dagger}
K^{\dagger}(t_f)\,
\mathscr O(t_J,z,i;t_f)\,
K(t_f)\mathcal V_{\sigma_b}.
}
$$

其中 $\mathcal V_\sigma$ 是原始时间片基底嵌入四维后的矩阵，$\Phi$ 是 flowed basis，$\mathscr O$ 包含当前时间片选择和 flowed Wilson line。Dirac 矩阵 $\Gamma$ 在后续 spin 收缩/投影中处理，不属于保存的几何 vertex 的模指标。

必须区分以下四点：

1. **全部保存向量**：对每个原始时间片，流动生产设置采用的全部 400 个 Blending 向量。
2. **全部原始时间片**：all-time 指覆盖 $\sigma=0,\ldots,N_t-1$，不是只使用当前插入时间片。
3. **完整四维演化**：每个输入向量在完整四维坐标格子上积分，没有每步投影回 400 维输入空间。
4. **不是完整热核或全谱求解**：没有构造完整 $K(x,y)$，没有求出全空间所有 Laplacian 本征向量；存储的输出还有时间窗截断。

因此，“all-time basis”不应翻译或解释为“全谱本征基”或者“精确完整的 all-to-all 热核”。

## 2. 符号、坐标与张量轴

### 2.1 时间标签不能混淆

| 符号 | 定义 | 代码对应 |
| --- | --- | --- |
| $t_f$ | 有量纲物理流时间，量纲为长度平方 | $t_f=a^2\tau$ |
| $\tau$ | 无量纲流时间 $t_f/a^2$ | `FLOW_TIME`、cache tag 中的 `tf...` |
| $\epsilon$ | 无量纲流步长 $\Delta\tau$ | `FLOW_EPSILON` |
| $\sigma$ | 原始 basis 所在时间片，也是热核输入时间片 | basis 程序中的 `tau`、`--taus` |
| $t_J$ | 流算符的插入/输出时间片 | `t_current`、`tJ...` |
| $\sigma_a,\sigma_b$ | vertex 左右两个 basis 的原始时间片 | `tau_a,tau_b`、`ta..._tb...` |
| $t_{\rm src},t_{\rm snk}$ | 核子源/汇时间片 | C2/C3 的外态时间标签 |

特别注意：`basis/tau000/` 的 `tau000` 表示 $\sigma=0$，**不是** $t_f=0$。流时间由更外层的 `thin_tf...` 标签确定。

### 2.2 坐标顺序

物理链接方向为

$$
(x,y,z,t)=(0,1,2,3),
$$

格点数组坐标按 $(t,z,y,x)$ 存放。原始/flowed 向量的单时间片轴为

$$
(\mathrm{mode},z,y,x,\mathrm{color}).
$$

主要张量如下：

| 对象 | 数组轴 | 当前生产 shape |
| --- | --- | --- |
| 原始单片 basis $V(\sigma)$ | `(mode,z,y,x,color)` | `(400,32,32,32,3)` |
| flowed basis $\Phi_\sigma(t_J)$ | `(mode,z,y,x,color)` | `(400,32,32,32,3)` |
| 单时间片 flowed gauge | `(direction,z,y,x,color_out,color_in)` | `(4,32,32,32,3,3)` |
| 几何/加权 vertex | `(vdv_sign,delta_z,left_mode,right_mode)` | `(2,21,400,400)` |

`DELTA_Z=21` 表示位移幅度 $\ell=0,\ldots,20$，不是最大位移为 21。两个 sign 分支分别保存 $-\ell$ 和 $+\ell$。

vertex 的空间内积直接使用格点求和；代码不在这里额外乘 $a^3$ 或除以 $N_s^3$。不要只改这个内积的归一化而不修改整条相关函数定义。

## 3. 原始 Blending basis 与完整性

### 3.1 400 个向量不等于 400 个精确低本征模

单时间片三维协变 Laplacian 的精确本征向量可以定义为

$$
-\nabla^2_{3\mathrm D}(\sigma)u_k(\sigma)
=\lambda_k(\sigma)u_k(\sigma).
$$

本项目采用的 Blending 文件不是简单截取最低 400 个精确 $u_k$，而是按当前读取顺序采用

$$
N_v=100\text{ 个精确低模}
+100\text{ 个谱扇区压缩向量}
+200\text{ 个高模噪声向量}.
$$

L32x96 原始标签中的 `n400` 不等于当前读取了 400 个高模噪声向量；需与 `[:config.nev_current]` 以及实际模排列一起判断。L32x64 标签记录 `n200`。

完整单时间片位置—颜色空间维数为

$$
N_{\rm space}=32^3\times3=98304.
$$

即使原始 400 列满足正交归一目标，也只有

$$
V^\dagger V=I_{400},
\qquad
VV^\dagger\ne I_{98304}.
$$

实际 Gram 精度需要独立检查，不能仅凭文件名或旧检查函数判断。高模噪声在完整高模空间中有支撑，并不意味着一个有限样本已经逐组态精确张成全部高模空间。

### 3.2 两个 ensemble 的压缩扇区不同

| 项目 | L32x96 | L32x64 |
| --- | --- | --- |
| 原始压缩谱组维数 | $(100,200,400,600,800)$ | $(100,200,400,700,1100)$ |
| 精确低模数 | 100 | 100 |
| 实际保存的压缩向量数 | 100 | 100 |
| 当前采用的高模噪声数 | 200 | 200 |
| 高模总体维数 | 96104 | 95704 |

每组的抽样/拆分参数须与原始输入和 `create_omega()` 一致，不能把 L32x96 的权重表直接拿给 L32x64。

原始基底标签中的 stout 或 `hpy` 描述源/汇基底的历史构造。当前插入算符的 gauge flow、两个 flowed endpoints 和 Wilson line 则从同一 thin-link 场演化；这两层不可混写。原始 basis 和普通 perambulator 必须同源、同模排列。

## 4. 四维 flow equation 与时间片嵌入

### 4.1 连续定义

采用反厄米规范势约定，

$$
D_\mu^{\rm F}=\partial_\mu+B_\mu,
\qquad
D_\mu^{\rm adj}X=\partial_\mu X+[B_\mu,X].
$$

规范流和费米子流为

$$
\partial_{t_f}B_\mu=D_\nu^{\rm adj}G_{\nu\mu},
\qquad
\partial_{t_f}\chi=\Delta_{B(t_f)}\chi,
\qquad
\Delta_B=\sum_{\mu=1}^4D_\mu^{\rm F}D_\mu^{\rm F}.
$$

本实现没有额外的 flow-time gauge-damping 项。规范流不受被流动的 basis 向量反作用。

热核满足

$$
\left[\partial_{t_f}-\Delta_{B(t_f)}^{(x)}\right]K(t_f;x,y)=0,
\qquad
K(0;x,y)=\delta_{xy},
$$

形式解为

$$
K(t_f)=\mathcal P_f\exp\left[\int_0^{t_f}ds\,\Delta_{B(s)}\right].
$$

一般不能将其替换为 $\exp[t_f\Delta_{B(0)}]$ 或 $\exp[t_f\Delta_{B(t_f)}]$，因为规范场和 Laplacian 随流时间变化。

### 4.2 原始 basis 嵌入四维

对每个原始时间片 $\sigma$ 和模 $n$，构造

$$
\eta_{\sigma n}(\boldsymbol y,t)
=\delta_{t,\sigma}V_n(\boldsymbol y;\sigma).
$$

然后积分初值问题

$$
\partial_{t_f}\Phi_{\sigma n}
=\Delta_{B(t_f)}\Phi_{\sigma n},
\qquad
\Phi_{\sigma n}(0)=\eta_{\sigma n}.
$$

于是

$$
\boxed{
\Phi_{\sigma n}(t_f;\boldsymbol x,t_J)
=\sum_{\boldsymbol y}
K(t_f;\boldsymbol x,t_J;\boldsymbol y,\sigma)
V_n(\boldsymbol y;\sigma).
}
$$

这种嵌入保留了热核输入时间标签，使输出可扩散到 $t_J\ne\sigma$。只在一个三维时间片上流动而不允许时间方向传播，不能得到这个四维对象。

## 5. Python 与 QUDA 中的实际 basis 算法

主要入口是 `generate_flowed_basis(config, taus)`。这里参数 `taus` 是 $\sigma$ 的列表。

### 5.1 逐原始时间片、逐批读取

核心读取和批次范围为

```python
eig = np.load(config.eigen_path(tau), mmap_mode="r")[:config.nev_current]
for mode_start in range(0, config.nev_current, batch_modes):
    ...
```

生产设置为 `NEV_CURRENT=400`、`FLOW_MODE_BATCH=32`。每个时间片处理 13 批：前 12 批各 32 个向量，最后 16 个向量。

### 5.2 四个向量装入一个 spinor

因为协变 Laplacian 对 Dirac spin 对角，代码将四个独立模装入一个 Wilson spinor 的四个分量：

```python
nspinors = (count + 3) // 4
lex = np.zeros(
    (nspinors, lt_local, 32, 32, 32, 4, 3),
    dtype=np.complex128,
)
for offset in range(count):
    lex[offset // 4, tau_local, :, :, :, offset % 4, :] = eig[mode_start + offset]
```

非源时间片初值为零。一批 32 个向量使用 8 个 spinor；最后一批 16 个向量使用 4 个 spinor。这是无 spin 混合 Laplacian 的批量复用，不是对不同模取线性组合，也不是物理的自旋投影。

数组随后通过 `latt_info.evenodd(...)` 转换成 QUDA 布局，传给 `MultiLatticeFermion`。

### 5.3 完整四维 Laplacian

令 $L_U=a^2\Delta_U$，则无量纲流方程为

$$
\partial_\tau\Phi=L_{U(\tau)}\Phi,
$$

$$
(L_U\Phi)(x)=\sum_{\mu=1}^4\left[
U_\mu(x)\Phi(x+\hat\mu)
+U_\mu^\dagger(x-\hat\mu)\Phi(x-\hat\mu)
-2\Phi(x)\right].
$$

底层 `performGFlowQuda` 的 `ApplyLaplace` 使用四维方向、跳跃系数 `1` 和对角系数 `-8`。它不是 Wilson/Clover Dirac 求逆，不含质量项或 Dirac spin 投影。

### 5.4 与规范场耦合的 RK3

记一步开始为 $U_0,\phi_0$，$\mathcal Z[U]$ 为 Wilson-flow 生成元。代码对应的三阶段为

$$
\begin{aligned}
k_0&=L_{U_0}\phi_0,\\
\phi_1&=\phi_0+\frac\epsilon4k_0,\\
U_1&=e^{\epsilon\mathcal Z[U_0]/4}U_0,
\end{aligned}
$$

$$
\begin{aligned}
k_1&=L_{U_1}\phi_1,\\
\phi_2&=\phi_0+\epsilon\left(\frac89k_1-\frac29k_0\right),\\
U_2&=\exp\left[\epsilon\left(\frac89\mathcal Z[U_1]-\frac{17}{36}\mathcal Z[U_0]\right)\right]U_1,
\end{aligned}
$$

$$
\begin{aligned}
k_2&=L_{U_2}\phi_2,\\
\phi_{\rm new}&=\phi_1+\frac{3\epsilon}{4}k_2,\\
U_{\rm new}&=\exp\left[\epsilon\left(\frac34\mathcal Z[U_2]-\frac89\mathcal Z[U_1]+\frac{17}{36}\mathcal Z[U_0]\right)\right]U_2.
\end{aligned}
$$

每次 Laplacian 使用对应阶段的规范场。Python 调用现有 `performGFlowQuda`，不自行执行 Euler 更新。

生产常用

$$
\epsilon=0.01,\qquad N_{\rm step}=\operatorname{round}(\tau/\epsilon).
$$

代码要求正流时间满足整数步条件。

### 5.5 All-time 的 GPU 分工

L32x64 的四个独立 worker 各用一个 GPU 和一 rank，设置 `GRID_SIZE=1,1,1,1`；分别处理

$$
\sigma\in\{r,r+4,r+8,\ldots\},\qquad r=0,1,2,3.
$$

这些集合覆盖全部 64 个原始时间片。**分配的是输入源时间片任务，不是把完整演化域裁成四块时间窗。** 每个 worker 都在完整 $32^3\times64$ 格子上流动自己的输入向量。

L32x96 的 all-time 定义相同，但应以实际冻结 launcher 的时间片划分为准，不能将 L32x64 的固定 `64` 字面量直接复用。

### 5.6 单流时间与多 checkpoint 模式

单流时间入口对每批原始向量从初始场开始流到目标 $\tau$。多 checkpoint 入口 `generate_flowed_basis_checkpoints` 在同一批内接续规范场和 spinor：

$$
K_{0\to\tau_2}
=K_{\tau_1\to\tau_2}K_{0\to\tau_1}.
$$

后续段使用 `restart=True`、已流动的 spinor 以及接续规范场，步数为相邻 checkpoint 之差。不能用从初始规范场重新开始的 $K_{0\to\tau_2-\tau_1}$ 替代。

当前列举的 L32x64 30-conf 生产脚本按 conf–flow 组合独立启动，调用的是单流时间入口；代码支持多 checkpoint 不意味着该批生产已使用这一优化。

### 5.7 边界条件

费米子流采用时间反周期边界；实现通过时间边界链接上的负号相位提供对应 Laplacian 边界条件。保存用于后续空间 Wilson line 的 flowed gauge 时移除该费米子边界相位，恢复周期规范场约定。

不能在后续 vertex 上再额外乘一次这个反周期负号。C3 外态传播跨边界的符号属于后续相关函数时间排列定义，不与 basis 的边界处理混为一件事。

## 6. 为什么这里没有求出完整 $K$ 或全部 eigenvector 空间

### 6.1 已计算的是矩形响应矩阵

省略平凡的 spin 单位结构，完整热核作用于

$$
N_{\rm full}=N_tN_s^3N_c
$$

维位置—颜色空间。将全部嵌入向量拼成 $\mathcal V$，它只有 $N_tN_v$ 列。当前计算的维数是

| ensemble | $N_{\rm full}$ | 输入列数 $N_tN_v$ |
| --- | ---: | ---: |
| L32x64 | 6,291,456 | 25,600 |
| L32x96 | 9,437,184 | 38,400 |

因此代码计算 $K\mathcal V$，不是 $N_{\rm full}\times N_{\rm full}$ 的完整矩阵 $K$。对 $\eta\in\operatorname{span}\mathcal V$，其线性组合响应可由已有结果构造；对未包含的任意初值，没有完整响应信息。

### 6.2 但输出没有被限制回 400 维

代码直接对位置空间场施加 $L_U$，没有每步插入输入子空间投影。若原始列正交归一，该投影可写成

$$
P_{\mathcal V}=\mathcal V\mathcal V^\dagger.
$$

若原始列不正交，这个乘积本身不是正交投影；不要将原始 Gram 的待检查状态隐藏在投影记号中。当前代码也没有通过 Gram 逆构造投影。

因此一般不能将实际计算等同于

$$
\mathcal V\exp\left[t_f\mathcal V^\dagger\Delta\mathcal V\right].
$$

流后向量可含原始输入子空间以外的分量。有限的是输入响应集合，而不是每个输出允许分布的坐标自由度。basis 的 `mode` 轴只是追踪哪个原始输入被流动；流后各列不保证是任何瞬时 Laplacian 的本征向量。

### 6.3 非幺正、非正交是正常结果

扩散核 $K$ 一般不幺正。即使 $\mathcal V^\dagger\mathcal V=I$，仍一般有

$$
\Phi^\dagger\Phi
=\mathcal V^\dagger K^\dagger K\mathcal V\ne I.
$$

当前计算保留原始模标签和传播子系数，不把 $\Phi\Phi^\dagger$ 当成单位分解，因此不应自行加入 Gram 逆矩阵或每步归一化。若采用 QR 表示 $\Phi=QR$，必须同步保留 $R$ 并改写后续系数；只用 $Q$ 替代 $\Phi$ 会改变算符。

## 7. 时间窗：完整积分与有限缓存

流积分在完整格子上进行；为减少磁盘和 vertex 规模，只保存满足

$$
d_{N_t}(t_J,\sigma)
\equiv\min_{k\in\mathbb Z}|t_J-\sigma+kN_t|\leq w
$$

的时间片。完整时间支持只在 $w<0$ 或 $2w+1\geq N_t$ 时保存。

聚合截断诊断为

$$
f_{\rm discard}(\sigma,t_f)
=1-
\frac{\sum_{t_J\in\mathcal W(\sigma)}\sum_{n,\boldsymbol x,c}
|\Phi_{\sigma n}(t_f;\boldsymbol x,t_J,c)|^2}
{\sum_{t_J=0}^{N_t-1}\sum_{n,\boldsymbol x,c}
|\Phi_{\sigma n}(t_f;\boldsymbol x,t_J,c)|^2}.
$$

代码在 unpack 时统计全部时间片范数，只把窗口内切片写入 memmap。L32x64 生产设置 `FLOW_MAX_DISCARDED_NORM=1e-6`，输出验收也要求同阈值。

此阈值不是每个模、每个 vertex 或每个 C3 的相对误差上界；过零或强抵消的量仍需 wider-window 回归。仅存在 `done.json` 也不能代替读取范数与完整流水线成功记录。

### 7.1 已核对的缓存实例

L32x64、conf10000、$\tau=0.5$、$\sigma=0$、$w=4$ 的缓存保存

$$
t_J\in\{0,1,2,3,4,60,61,62,63\}.
$$

其既有 `done.json` 记录

$$
f_{\rm discard}=2.775479546190951\times10^{-8},
$$

并标为 `exact_temporal_support=false`、`nev=400`、`n_steps=50`。这是一个已保存样例的元数据，不是对所有组态/流时间窗口误差的重新测量。

## 8. 从 flowed bilocal 推导 generalized vertex

### 8.1 算符与空间链

在同一个输出时间片 $t_J$，定义

$$
\mathcal O_\Gamma^{\rm GF}(t_J,z,i;t_f)
=\sum_{\boldsymbol x}
\bar\chi(t_f;\boldsymbol x,t_J)\,
\Gamma W_i(t_f;\boldsymbol x,\boldsymbol x+z\hat i;t_J)\,
\chi(t_f;\boldsymbol x+z\hat i,t_J).
$$

对正位移 $z=\ell a$，

$$
W_{i,+\ell}(\boldsymbol x)
=U_i(\boldsymbol x)U_i(\boldsymbol x+a\hat i)\cdots
U_i(\boldsymbol x+(\ell-1)a\hat i).
$$

对负位移，

$$
W_{i,-\ell}(\boldsymbol x)
=U_i^\dagger(\boldsymbol x-a\hat i)
U_i^\dagger(\boldsymbol x-2a\hat i)\cdots
U_i^\dagger(\boldsymbol x-\ell a\hat i).
$$

此处所有链接均为同一 $t_f,t_J$ 的 flowed links。路径顺序对应代码的端点颜色输运；不能只倒置数组而漏掉反向链接的厄米共轭。

### 8.2 两个端点各施加一次热流核

在所声明的基底表示中，流场由 $\Phi_{\sigma n}=K\eta_{\sigma n}$ 展开。将两个端点代入 bilocal，得到未加权的几何 vertex：

$$
\boxed{
\begin{aligned}
J_{mn}(t_J;\sigma_a,\sigma_b;z,i;t_f)
=\sum_{\boldsymbol x,c,d}
&\Phi_{\sigma_a m}^{*}(t_f;\boldsymbol x,t_J,c)\\
&\times W_i^{cd}(t_f;\boldsymbol x,\boldsymbol x+z\hat i;t_J)\\
&\times\Phi_{\sigma_b n}(t_f;\boldsymbol x+z\hat i,t_J,d).
\end{aligned}
}
$$

这等价于 $\mathcal V_{\sigma_a}^\dagger K^\dagger\mathscr O K\mathcal V_{\sigma_b}$。复共轭左 basis 已提供左端的 $K^\dagger$ 结构，不需要为这个 vertex 另外构造一张稠密 $K^\dagger$。

上式是给定输入基底下的代数定义；有限随机基底能否代表完整坐标算符的目标估计，还需要其抽样和权重条件。

### 8.3 为什么有两个原始时间求和

零流时间时，

$$
\Phi_{\sigma n}(0;\boldsymbol x,t_J)
=\delta_{t_J,\sigma}V_n(\boldsymbol x;\sigma),
$$

所以

$$
J^{(0)}(t_J;\sigma_a,\sigma_b)
=\delta_{\sigma_a,t_J}\delta_{\sigma_b,t_J}
V^\dagger(t_J)\mathscr O(t_J)V(t_J).
$$

正流时间一般有非零的 $\sigma_a\ne\sigma_b$ 以及 $\sigma_{a,b}\ne t_J$ 块。当前 vertex 程序对每个 $t_J$ 枚举窗口内的全部 $(\sigma_a,\sigma_b)$，不是只取对角块。

若每个 $t_J$ 的活动源时间片数为 $2w+1<N_t$，每个方向的块数为

$$
N_t(2w+1)^2.
$$

这解释了为什么随着时间窗扩大，vertex 的重构成本和磁盘规模增长很快。

## 9. `generalized_vertex` 的数组实现

函数签名：

```python
generalized_vertex(left, right, gauge_t, direction, delta_z, xp=np)
```

输入 `left/right` 都是单个 $t_J$ 上的 `(mode,z,y,x,color)` 数组，分别来自 $\sigma_a,\sigma_b$。

### 9.1 零位移

代码将位置—颜色轴展平，以矩阵乘法收缩：

```python
left_flat = left.reshape(left.shape[0], -1)
zero = left_flat.conj() @ right.reshape(right.shape[0], -1).T
out[0, 0] = zero
out[1, 0] = zero
```

输出的第 0 轴为

$$
\mathrm{sign}=0:\ -\ell,\qquad
\mathrm{sign}=1:\ +\ell.
$$

两分支在 $\ell=0$ 相同。这个 overlap 不是一般意义的 $I$：流后不正交，且左右原始时间片可以不同。

### 9.2 正负链递推与 `roll` 符号

右向量在正位移取值使用

```python
xp.roll(right, -dz, axis=mode_axis)
```

因为 `roll(...,-dz)` 在当前位置返回原数组的 $x+dz$ 值。负位移使用 `+dz`。

递推链与输运后的右向量先做色收缩，再通过大矩阵乘法与左向量收缩：

```python
positive_wilson = positive_wilson @ xp.roll(link, -(dz - 1), axis=link_axis)
positive = xp.einsum(
    "zyxcd,nzyxd->nzyxc", positive_wilson,
    xp.roll(right, -dz, axis=mode_axis), optimize=True,
)
out[1, dz] = left_flat.conj() @ positive.reshape(right.shape[0], -1).T

backward_link = xp.roll(link, dz, axis=link_axis).swapaxes(-1, -2).conj()
negative_wilson = negative_wilson @ backward_link
```

方向映射为

| `direction` | 向量空间轴 `mode_axis` | 链接场空间轴 `link_axis` | gauge 方向索引 |
| --- | ---: | ---: | ---: |
| z | 1 | 0 | 2 |
| y | 2 | 1 | 1 |
| x | 3 | 2 | 0 |

当前几何 vertex 没有独立的核子动量轴或 Fourier 相位；外态动量在后续 VVV/相关函数处理中进入。不要把动量依赖强行加到这个已定义的零插入动量空间求和 vertex 中。

## 10. 随机权重 $\omega$：实际位置与待确认问题

### 10.1 当前代码执行的加权

几何 vertex 形成后，生产程序执行

```python
omega = xp.asarray(create_omega(config.nev_current))
vertex = generalized_vertex(left, right, gauge_t, direction, config.delta_z, xp=xp)
vertex *= omega[None, None]
```

因此保存的是

$$
\boxed{\widetilde J_{mn}=\omega_{mn}J_{mn},}
$$

即 Hadamard 逐元素乘法，不是矩阵乘法 $\omega J$。basis 的流方程不乘 $\omega$；后续读取已加权 vertex 不应重复乘同一份权重。

### 10.2 固定抽样沿流轨迹不重新加权

当前 `create_omega()` 根据扇区总体维数与实际样本数构造。它没有流时间、输出时间或原始时间片参数；沿同一条固定输入的 flow，不按 flowed norm 或 Gram 矩阵更新权重。

在同一扇区不放回抽样的简单模型下，典型联合权重为

$$
\omega_{mn}=\begin{cases}
d_r/n_r,&m=n\text{，同一扇区},\\[2pt]
d_r(d_r-1)/[n_r(n_r-1)],&m\ne n\text{，同一扇区},\\[2pt]
(d_r/n_r)(d_s/n_s),&\text{两个独立扇区}.
\end{cases}
$$

它说明代码的抽样补偿结构，但不能仅凭向量正交性证明全部核子随机估计无偏。

### 10.3 必须保留的统计验证缺口

一般权重的身份标签应为

$$
\Omega_{(\sigma_a,m),(\sigma_b,n)}.
$$

即使 $m=n$，当 $\sigma_a\ne\sigma_b$ 时也不是同一个输入向量。不同源时间片若独立抽样，其联合事件未必对应同时间片的二阶权重。当前程序对所有时间组合使用同一张 $\omega_{mn}$。

**待确认：这一处方是否匹配原始 Blending 输入的跨时间随机联合分布，以及包含核子多条传播线/VVV 后的完整联合矩。** 零流时间回归只能覆盖时间对角结构，不能单独检验新增的跨时间块。此问题不是已确认数值错误，也不是修改已提交任务权重的授权。

## 11. 接回普通 perambulator 与 C3

普通 perambulator 定义为

$$
\mathcal T_f(t,\sigma)=V^\dagger(t)D_f^{-1}(t,\sigma)V(\sigma),
$$

其中 $f$ 是 flavor，原始 spin 指标省略。当前核子源/汇及 C2 保持未流，flow 只进入插入的两端和 Wilson line。

在声明的基底/随机估计下，活动夸克线的连接结构示意为

$$
\sum_{\sigma_a,\sigma_b}\sum_{m,n}
\mathcal T_f(t_{\rm snk},\sigma_a)_{\cdots m}
\Gamma\,\widetilde J_{mn}(t_J;\sigma_a,\sigma_b)
\mathcal T_f(\sigma_b,t_{\rm src})_{n\cdots}.
$$

完整 C3 还包括 spectator 线、核子 VVV、spin projector、时间边界符号和各 Wick 项。这个示意式不应在忽略有限随机表示条件后被称作完整坐标传播子的精确恒等式。

现有因子化先构造

$$
H_{uSLpamb}=\sum_{t,n}\widetilde J_{tuSLnp}A_{tambn},
$$

对应

```python
FLOWED_SEQ_EQUATION = "tuSLnp,tambn->uSLpamb"
```

这里 $t,u$ 是两个原始时间标签，$S$ 是位移 sign，$L$ 是位移幅度，其余为对应模和 spin 标签；不能把这些代码字母当成流时间符号。优化重排已有求和，不增删 Wick 拓扑、不重复加权。

外部 rank-2 parity contraction 优化是后续 spin 收缩表示的改变，不会把 `NEV_CURRENT=400` 减为 200，也不会把 flowed basis 的模维度变为 2。当前 L32x64 输出仍保存完整插入 spin 指标。

## 12. 数据路径、元数据和缓存策略

### 12.1 L32x64 的规范路径

基线/共享缓存根目录：

```
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_flow_20261007/flow_cache/
```

布局为

```
<conf>/thin_tf<tau>_eps<epsilon>_tw<w>_Nv400_dz21/
  basis/
    tau<sigma>/
      tJ<t_current>.npy
      done.json
  flowed_gauge/
    <shard files>
    index.json
  vertex/
    tJ<t_current>/
      ta<sigma_a>_tb<sigma_b>_<direction>.npy
      ta<sigma_a>_tb<sigma_b>_<direction>.json
```

举例：`thin_tf0p50000000_eps0p01000000_tw4_Nv400_dz21`。

共享 C2/C3 根目录：

```
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_flow_20261007/results/<flow-tag>/Nev_200/<conf>/
```

这里 `Nev_200` 是源/汇相关设置，不表示 flowed current basis 只有 200 模。扩展任务的代码、日志和状态位于自己的冻结 package，结果和缓存仍使用基线共享根目录。

### 12.2 重要 metadata

Basis 的 `done.json` 记录 schema、conf、流时间、步长、步数、thin gauge 路径、$\sigma$、窗口、保存的 $t_J$ 列表、400 模、dtype、完整时间范数和截断比例。

Vertex 的 JSON 记录 $t_J,\sigma_a,\sigma_b$、方向、轴顺序、`minus_dz/plus_dz`、权重处方、dtype、时间窗和计时。

生产通常在双精度流工作场上计算，basis/vertex 写为 complex64；后续相关函数采用其声明的计算/输出精度。缓存低精度与 RK 积分误差是不同的误差来源。

### 12.3 当前缓存政策

保留 basis 和 flowed gauge；vertex 计算后按既定任务清理策略删除并可重构。VVV 在 job-local scratch 中临时计算，退出时清理，不长期保存。不得因此删除原始 gauge/eigen/perambulator 或最终 C2/C3。

本文只记录策略，不执行任何清理。后续故障恢复必须依据实际 receipt 和元数据，不把“vertex 已删除”当作“basis 丢失”。

## 13. 当前 L32x64 参数与执行入口

对照的是 `quark_l32x64_expand30_20261012/run_expand30_na800_sug.slurm` 的冻结设置，不代表新的队列状态：

| $\tau=t_f/a^2$ | 时间窗半宽 $w$ | $\epsilon$ | 步数 |
| ---: | ---: | ---: | ---: |
| 0.05 | 2 | 0.01 | 5 |
| 0.10 | 2 | 0.01 | 10 |
| 0.20 | 3 | 0.01 | 20 |
| 0.30 | 3 | 0.01 | 30 |
| 0.40 | 4 | 0.01 | 40 |
| 0.50 | 4 | 0.01 | 50 |
| 0.80 | 6 | 0.01 | 80 |
| 1.00 | 7 | 0.01 | 100 |

当前 source/VVV 模数为 200，current 为 400，$z/a=0..20$，$P_z$ 整数标签为 $+3,+4,+5$，方向为 z，$t_{\rm sep}=5..10$。L32x64 的格距输入为 $a=0.0897\,\mathrm{fm}$。

以下仅说明已有 compute-job 内部调用形式，**不要在登录节点执行数值 flow/contraction，也不要为了查看进度重复提交整个生产数组**。

```bash
# 在已分配的 GPU job 环境中；conf 与各环境变量由冻结 launcher 设置。
python -u reference/generate_flowed_vertex_l32x64.py "$conf" --stage preflight

# 每个 worker 独立一 rank / 一 GPU，--taus 为其负责的原始时间片。
mpirun --bind-to none -n 1 python -u reference/generate_flowed_vertex_l32x64.py \
    "$conf" --stage basis --taus "$source_times"

# basis workers 成功结束后，对全部当前时间片构造 z 向 vertex。
mpirun -n 4 python -u reference/generate_flowed_vertex_l32x64.py \
    "$conf" --stage vertex --currents 0-63 --directions z
```

`--flow-time-specs` 是代码支持的多 checkpoint basis 选项，例如 `0.2:3,0.3:3,0.4:4,0.5:4`；后续 vertex 仍需按各自 `FLOW_TIME` 和 cache tag 单独运行。是否启用应使用独立 successor 和已有 restart 验证，不直接修改排队任务的引用文件。

用户指定的后续 GPU 提交节点/partition 为 `na800-sug`，脚本包含 `#SBATCH --reservation=donghx_223`。本次没有提交任务，也没有恢复上一轮中断的新增组态提交工作。

## 14. 检查、验证与尚未解决的问题

### 14.1 可以直接从源码/缓存确认的事实

- 四维 delta-time 初值，处理全部配置采用的 400 个输入向量。
- `performGFlowQuda` 使用四维 Laplacian 和耦合 RK3。
- 完整格子演化，输出以时间窗裁剪，不每步投影回 400 维。
- vertex 保留两个原始时间标签和正负位移。
- 权重在几何 vertex 后逐元素乘一次，basis 不乘它。
- 已有 conf10000/流时间 0.5/源片 0 的范数截断样例可由 JSON 复核。

这些是实现或既有文件事实，不是本次新跑的 GPU 回归。

### 14.2 不能由这些事实自动推出的结论

- 400 模随机表示逐样本等于完整位置—颜色空间。
- 完整随机核子 C3 及所有跨时间权重已经独立证明无偏。
- 聚合范数阈值保证所有矩阵元的相对误差都同样小。
- 原始/flowed 基底必定在所有文件中满足目标 Gram 精度。
- ringed normalization、GF-to-quasi matching 或 LaMET 已由 basis/vertex 阶段完成。

### 14.3 最具体的后续验证

1. 在小格子或有限维独立模型中显式构造 $K$，比较直接 $K^\dagger\mathscr O K$ 与通过各 $\Phi$ 得到的 $J$。
2. 检查 $t_f=0$ 时只剩 $\sigma_a=\sigma_b=t_J$，并回归原始 VDV。
3. 比较直接流到最终时间与多 checkpoint restart，检查同轨迹 gauge 和 basis。
4. 做 $\epsilon$ 减半和缓存精度对照，不将一种误差当另一种。
5. 对代表性流时间/位移/动量扩大时间窗，直接比较 C3，不仅比较 basis norm。
6. 检查原始 Gram、flowed Gram、APBC 跨界、gauge covariance 和正反链共轭关系；不同源时间片的共轭测试必须同时交换左右标签。
7. 追溯原始 Blending 的跨时间随机生成与 seed，建立有限维抽样参考，比较生产权重与明确联合概率下的估计。此项未完成前保留“待确认”标签。

上述独立数值验证应在计算节点上运行；本文没有启动这些测试，也不将历史测试清单写成当下全部通过。

## 15. 源码依据与不可静默修改的约定

主要源码入口：

- [L32x64 basis 和 vertex 生成器](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/generate_flowed_vertex_l32x64.py:359)：单流时间 basis；同文件的 checkpoint 与 vertex 函数。
- [L32x64 几何 vertex](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/flow_common_l32x64.py:532)：路径递推、位移符号和矩阵收缩。
- [L32x64 权重表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/flow_common_l32x64.py:447)：`create_omega()`。
- [L32x64 worker 分工](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/full_chain.py:69)。
- [L32x64 冻结生产参数](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/run_expand30_na800_sug.slurm:27)。
- [L32x64 因子化代码](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_expand30_20261012/reference/flowed_perambulator_factorization.py:99)。
- [L32x96 对照生成器](/public/home/donghx/Gradient_Flow/Blending_L32x96/generate_flowed_vertex_l32x96.py:357)：具体生产路径须继续沿各冻结 launcher 核对。
- [QUDA 耦合流实现](/public/home/donghx/PyQuda_a100_2608/quda/lib/interface_quda.cpp:5485)。
- [PyQUDA 流绑定](/public/home/donghx/PyQuda_a100_2608/PyQUDA-gflow-src/pyquda/src/quda.in.pyx:416)。
- [缓存样例元数据](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_flow_20261007/flow_cache/10000/thin_tf0p50000000_eps0p01000000_tw4_Nv400_dz21/basis/tau000/done.json)。

配套文档：

- [flow_equation.md](/public/home/donghx/.codex/Markdown/flow_equation.md)：更详细的随机权重与跨时间联合矩讨论。
- [gradient_flow.md](/public/home/donghx/.codex/Markdown/gradient_flow.md)：完整项目交接及分析层状态。
- [divergence.md](/public/home/donghx/.codex/Markdown/divergence.md)：线发散与后续重整化；本 basis/vertex 阶段没有做这一减除。
- [Note 的 main.tex](/public/home/donghx/Note/Gradient_Flow/main.tex)。

理论出处沿用已有 Note/源码注释：M. Lüscher, arXiv:1302.5246，特别是费米子 flow 与 Appendix D 的数值积分；以及 arXiv:1006.4518 的 Wilson flow。本文依据现有本地源码和说明整理，不声称新增文献结论。

不可静默修改的约定包括：thin-link 同轨迹、四维时间扩散、源时间/流时间的区分、原始 basis/perambulator 配对、位移 sign 轴、原始内积归一化、一次 Hadamard 权重、非幺正 flow 不强制正交化，以及所有未解决统计问题的标注。

本次只生成说明文档；没有修改生产代码、已提交脚本、输入数据、缓存或相关函数，没有提交新任务，也没有读取或写入任何凭据。

生成日期：**2026-10-14（Asia/Shanghai）**。

本次会话标题：**Flowed all-time basis 与 generalized vertex 的数学定义及生产代码实现**。
