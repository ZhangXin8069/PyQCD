# Flowed all-time basis 与 generalized vertex：定义、代码实现与验证边界

本文记录本项目 quark PDF 的四维 gradient-flow Blending/distillation 实现。重点是：原始时间片向量怎样嵌入四维、怎样计算热核作用、怎样构造带两个原始时间标签的 vertex，以及实际计算覆盖了什么、没有覆盖什么。

本文以 L32x64 已提交生产副本为主要源码依据，并对照 L32x96 实现和 `flow_equation.md`。它不是新算法说明，也不将实现一致性等同于完整物理或随机估计验证。全部公式采用 LaTeX；“确定事实”“数学关系”和“待确认问题”分开表述。

**2026-10-15 更新：**第 10 节补充权重估计器及其成立条件，第 16 节记录 L32x64 的 10 个组态、40 个案例的已完成比较。统一二阶表仍是当前生产处方；“同时间二阶表、独立跨时间乘积表”是独立对照，不是已获准替换生产权重的纠错版本。

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

## 10. 随机权重 $\omega$：估计器、时间标签与成立条件

### 10.1 当前代码执行的加权

几何 vertex 形成后，生产程序执行

```python
omega = xp.asarray(create_omega(config.nev_current))
vertex = generalized_vertex(left, right, gauge_t, direction, config.delta_z, xp=xp)
vertex *= omega[None, None]
```

因此保存的是

$$
\boxed{
\widetilde J_{mn}(t_J;\sigma_a,\sigma_b)
=\Omega_{(\sigma_a,m),(\sigma_b,n)}
J_{mn}(t_J;\sigma_a,\sigma_b),
}
$$

即 Hadamard 逐元素乘法，不是矩阵乘法 $\omega J$。basis 的流方程不乘 $\omega$；后续读取已加权 vertex 不应重复乘同一份权重。

**当前生产方案 A** 取 $\Omega^{A}_{(\sigma_a,m),(\sigma_b,n)}=\omega^{(2)}_{mn}$，对所有时间块使用同一二阶表。后文对照方案 B 在跨时间块才改用一阶乘积；两者在同时间块完全采用同一张表。

### 10.2 固定抽样沿流轨迹不重新加权

当前 `create_omega()` 根据扇区总体维数与实际样本数构造。它没有流时间、输出时间或原始时间片参数；沿同一条固定输入的 flow，不按 flowed norm 或 Gram 矩阵更新权重。

固定规范场历史并对输入随机抽样取期望，若一阶单位分解估计满足

$$
\mathbb E_\eta\widehat{\mathcal P}=\mathcal P,
$$

而 $K$ 在给定规范场下不依赖这次随机抽样，则

$$
\mathbb E_\eta\!\left[K\widehat{\mathcal P}K^\dagger\right]
=K\mathcal P K^\dagger.
$$

这里 $\mathbb E_\eta$ 是噪声/压缩抽样期望，不是本次十个规范组态的平均。非幺正热流导致输出范数变化，并不改变原始入选概率；不能据此按 $\Phi^\dagger\Phi$、流后范数或 $t_J$ 重新归一化权重。流时间和时间窗改变 $J$，不改变固定输入抽样的 $d_r,n_r$。

### 10.3 一阶与同时间二阶权重的推导

令 $r$ 标记一个实际独立的抽样扇区，$d_r$ 为其总体维数，$n_r$ 为保存的样本数。不能把一个大谱组与其实际拆分后的子扇区混为一项。完整身份是 $(\sigma,r,m)$，$m$ 是保存的样本槽位，不一定是原始精确本征模的谱序号。

先考虑一个可以严格证明的模型：给定扇区的完整正交标架 $\{q_1,\ldots,q_d\}$，均匀地、不放回地选取 $n$ 列。对任意总体方向 $i$ 及不同方向 $i\ne j$，

$$
\pi_1=\Pr(i\text{ 入选})=\frac nd,
\qquad
\pi_2=\Pr(i,j\text{ 同时入选})=\frac{n(n-1)}{d(d-1)}.
$$

一阶估计器和权重为

$$
\widehat{\mathcal P}_{\sigma r}
=\sum_{m=1}^{n_r}w_{\sigma r,m}
\eta_{\sigma r,m}\eta_{\sigma r,m}^\dagger,
\qquad w_{\sigma r,m}=\frac{d_r}{n_r}.
$$

对两个来自同一标架的分辨因子，重复同一列只需一次入选；两列不同时需要联合入选。因此同时间二阶表为

$$
\omega^{(2)}_{mn}=\begin{cases}
d_r/n_r,&m=n\text{，同一扇区},\\[2pt]
d_r(d_r-1)/[n_r(n_r-1)],&m\ne n\text{，同一扇区},\\[2pt]
(d_r/n_r)(d_s/n_s),&\text{两个独立扇区}.
\end{cases}
$$

在上述完整标架均匀子集模型下，对不依赖该次子集抽样的确定矩阵 $M$，

$$
\mathbb E_\eta\!\left[
\sum_{m,n}\omega^{(2)}_{mn}
R_{\sigma m}M R_{\sigma n}\right]
=\mathcal P_\sigma M\mathcal P_\sigma,
\qquad R_{\sigma m}=\eta_{\sigma m}\eta_{\sigma m}^\dagger.
$$

本式是给定抽样模型下的结论，不是从“保存向量互相正交”单独推出来的结论。实际输入若通过部分随机标架、顺序正交化等方法直接生成，必须另核对一阶、二阶乃至更高阶联合矩，不能只检查 $V^\dagger V$。

L32x64 表中的实际扇区如下；数组模下标从 0 开始，下面的范围是闭区间。

| 模范围 | 扇区总体维数 $d_r$ | 样本数 $n_r$ | 一阶权重 $d_r/n_r$ |
| --- | ---: | ---: | ---: |
| 0–99，exact | 100 | 100 | 1 |
| 100–109；110–119 | 每个子扇区 50 | 每个 10 | 5 |
| 120–129；130–139 | 每个子扇区 100 | 每个 10 | 10 |
| 140–149；150–159 | 每个子扇区 200 | 每个 10 | 20 |
| 160–169；170–179 | 每个子扇区 350 | 每个 10 | 35 |
| 180–189；190–199 | 每个子扇区 550 | 每个 10 | 55 |
| 200–399，high | 95704 | 200 | 478.52 |

总体维数之和为 98304，保存数之和为 400。源/汇 VVV 采用前 200 列，其中包含随机压缩列；它不是“只有 exact 模”的源/汇。L32x96 的扇区不同，不能复制本表的数字。

特别注意，一阶向量由真实扇区的 $d_r/n_r$ 构造；因为 $\omega^{(2)}_{mm}=w_m$，不能用 $\sqrt{\operatorname{diag}\omega^{(2)}}$ 当作一阶权重。

### 10.4 两种时间处方与 $m=n$ 的含义

本次实际比较的是

$$
\boxed{
\begin{aligned}
\Omega^{A}_{(\sigma_a,m),(\sigma_b,n)}
&=\omega^{(2)}_{mn},\\[2pt]
\Omega^{B}_{(\sigma_a,m),(\sigma_b,n)}
&=\begin{cases}
\omega^{(2)}_{mn},&\sigma_a=\sigma_b,\\[2pt]
w_{\sigma_a,m}w_{\sigma_b,n},&\sigma_a\ne\sigma_b.
\end{cases}
\end{aligned}
}
$$

当前各原始时间片采用相同的扇区布局，所以 $w$ 的数值不随 $\sigma$ 变化；向量、perambulator 和 vertex 仍然随时间变化。

当 $\sigma_a\ne\sigma_b$ 时，即使保存槽位 $m=n$，左右也不是同一个四维输入向量。若两片确实独立，方案 B 的一阶乘积有直接的独立抽样解释：这里不再使用同一列只入选一次的对角概率。

但是，**不能由此直接断言方案 A 在跨时间块必然有偏**。完整模求和下，两种表可能是同一目标的不同无偏估计器，见下一小节。单个模对的权重不同，不代表整体期望不同。

### 10.5 独立跨时间条件下，统一表为何也可能无偏

以下是明确条件下的代数推导，不是对实际完整核子噪声分布的最终证明。固定规范场，并限定两个时间片上的某一随机扇区。假设：

1. 两个时间片的随机标架独立；
2. 各片每个保存槽位都有相同一阶矩，
   $\mathbb E_\eta R_{\sigma m}=\mathcal P_\sigma/d$；
3. 中间矩阵 $M$ 不依赖这两片随机抽样。

则独立性给出

$$
\mathbb E_\eta[R_{\sigma_a m}M R_{\sigma_b n}]
=\frac{1}{d^2}\mathcal P_{\sigma_a}M\mathcal P_{\sigma_b}.
$$

对任何确定的模式表 $C_{mn}$，

$$
\mathbb E_\eta\!\left[\sum_{m,\ell=1}^n C_{m\ell}
R_{\sigma_a m}M R_{\sigma_b\ell}\right]
=\frac{\sum_{m,\ell}C_{m\ell}}{d^2}
\mathcal P_{\sigma_a}M\mathcal P_{\sigma_b}.
$$

独立乘积表与统一二阶表均满足所需的表和：

$$
\begin{aligned}
\sum_{m,n}w_mw_n&=n^2\left(\frac dn\right)^2=d^2,\\
\sum_{m,n}\omega^{(2)}_{mn}
&=n\frac dn+n(n-1)\frac{d(d-1)}{n(n-1)}=d^2.
\end{aligned}
$$

因而在这些条件下，A 与 B 的完整模求和具有相同目标期望；有限随机样本的值和方差可以不同。两个不同随机扇区也可用各自的 $d_r,d_s$ 推导；exact 扇区直接作完整确定性求和，无须给每个 exact 模假设随机的一阶矩。

例如 $d=50,n=10$：统一表对角为 5、非对角为 $2450/90\simeq27.2222$，乘积表每项为 25。前者所有项的平均同样是 25。只比较其中某个对角元素 $5$ 与 $25$，不能推断全模估计器的期望相差五倍。

相关文献依据为 [W1] 的补充材料式 (22)–(24)：不同标架/时间标签下也讨论了沿用统一高阶表的组织方式。本小节给出了用于本项目的二阶、独立时间片条件推导；文献并不替代对实际输入生成器、槽位分布及源/汇联合矩的核对。

### 10.6 为什么不能同时间也一律改成乘积表

对于同一随机正交标架，左右抽样不独立。例如取一个扇区内的 $M=\mathcal P_\sigma$，有 $R_{\sigma m}R_{\sigma n}=\delta_{mn}R_{\sigma m}$。若 $n<d$ 且每列一阶矩为 $\mathcal P_\sigma/d$，同时间使用乘积表会给出

$$
\mathbb E_\eta\!\left[
\sum_{m,n}\left(\frac dn\right)^2
R_{\sigma m}\mathcal P_\sigma R_{\sigma n}\right]
=\frac dn\mathcal P_\sigma\ne\mathcal P_\sigma.
$$

同时间二阶表的对角 $d/n$ 则得到正确的一阶结果。本轮 B **只改跨时间块**，没有测试或采用“所有块都乘一阶乘积”的处方。

### 10.7 两个原始时间求和不是重复抽样

$$
\sum_{\sigma_a\in\mathcal W(t_J)}
\sum_{\sigma_b\in\mathcal W(t_J)}
\sum_{m,n}
\Omega_{(\sigma_a,m),(\sigma_b,n)}
\mathcal T(t_{\rm snk},\sigma_a)_{\cdots m}
J_{mn}(t_J;\sigma_a,\sigma_b)
\mathcal T(\sigma_b,t_{\rm src})_{n\cdots}
$$

中两个时间和来自两个流端点，而不是对同一随机样本做两次测量。代码的时间集合先取模、去重、排序，然后遍历有序对：

- 同时间对 $(\sigma,\sigma)$ 只出现一次，使用二阶表；
- $(\sigma_a,\sigma_b)$ 与 $(\sigma_b,\sigma_a)$ 是两个不同的左右端点排列，不能无条件合并或乘 $1/2$；
- 时间片在此是完整枚举的格点标签，不额外乘随机时间抽样概率，也不自动除以 $N_t$ 或 $(2w+1)^2$；
- 原始空间格点归一化沿用已有离散定义；$a^3$ 等因子不能在本层单独增删；
- 若以后随机抽取时间片，或只用 Hermitian 半表，才需要另行声明时间抽样权重或精确重构规则。本轮没有采用这些方法。

本轮保留每个有序时间块并在最后求和，正是为了分别检查：权重是否只施加一次、同时间是否一致、跨时间差异怎样进入各个 Wick 项及总 C3。Hermiticity 或零流时间一致性不能替代这个联合估计验证。

### 10.8 仍必须保留的统计验证缺口

一般权重的身份标签应为

$$
\Omega_{(\sigma_a,m),(\sigma_b,n)}.
$$

第 10.5 节说明的是满足条件的二端点估计器。完整核子 C3 的系数还含 source/sink VVV、spectator perambulator 和多个随机基底因子。当 $\sigma_a$ 或 $\sigma_b$ 与源/汇时间重合时，可能共享同一批噪声；不能把含这些噪声的系数无条件视为推导中的确定矩阵 $M$。即使左右时间片独立，也不等于所有核子传播线相互独立。

**待确认：**真实生产 basis 的冻结生成版本、种子与跨时间相关性；每个槽位的一阶矩与同时间二阶矩；以及包含核子多条传播线/VVV 后的更高阶联合矩。零流回归只覆盖时间对角结构；本次有限流时间回归已覆盖第 16 节列出的案例，但不构成完整随机无偏性的证明。

**当前决策：**保留生产 A 和独立对照 B；没有仅因 B 的跨时间解释更直观、或 A/B 差异只有几个百分点，就替换已有权重。也不能把 A/B 差异直接当作已量化的系统误差或添加到拟合误差条中。

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

用户指定的后续 GPU 提交节点/partition 为 `na800-sug`，脚本包含 `#SBATCH --reservation=donghx_223`。第 16 节另行记录已完成的独立权重对照作业，不表示本节整批生产设置重新提交。2026-10-15 这次文档更新没有新提交计算。

## 14. 检查、验证与尚未解决的问题

### 14.1 可以直接从源码/缓存确认的事实

- 四维 delta-time 初值，处理全部配置采用的 400 个输入向量。
- `performGFlowQuda` 使用四维 Laplacian 和耦合 RK3。
- 完整格子演化，输出以时间窗裁剪，不每步投影回 400 维。
- vertex 保留两个原始时间标签和正负位移。
- 权重在几何 vertex 后逐元素乘一次，basis 不乘它。
- 已有 conf10000/流时间 0.5/源片 0 的范数截断样例可由 JSON 复核。

上述项目是实现或既有文件事实。除此以外，2026-10-14/15 已完成第 16 节列出的实际 L32x64 权重对照、生产 C2/C3 回归与 CPU 汇总；其范围仅限所列 10 个组态、单个流时间和 40 个案例。

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
7. 第 10 节已补充条件性估计推导，第 16 节已完成 A/B 的实际逐块比较；下一步仍需追溯真实输入生成器与 seed，对照所需一阶、二阶和源/汇高阶联合矩。真实生产输入的完整无偏性仍保留“待确认”，不能因实现回归 PASS 而删除。

上述独立数值验证应在计算节点上运行。第 16 节所列测试已运行并有收据；本节其余项目仍是后续验证清单，不将它们写成全部通过。

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

权重估计的补充文献：

- **[W1]** Zhi-Cheng Hu et al., *Realizing the stochastic Laplacian Heaviside method using quasi-minimal residual solvers*, arXiv:2505.01719v3，补充材料式 (22)–(24)（不同随机标架下的统一表组织方式）及式 (25)–(29)（高阶插入形式）。2026-10-15 直接核对 v3 的 HTML 补充材料。第 10 节对本项目二阶表另作了明确假设下的代数推导，不能把文献引用视为对现有全部 basis 文件生成历史的证明。

`flow_equation.md` 中早于本轮比较的“待确认”文字应结合本文第 10、16 节理解：已完成条件推导与数值比较，但未完成真实核子联合随机矩证明。本次只更新 `basis_vertex.md`，没有静默改写其它文档。

不可静默修改的约定包括：thin-link 同轨迹、四维时间扩散、源时间/流时间的区分、原始 basis/perambulator 配对、位移 sign 轴、原始内积归一化、一次 Hadamard 权重、非幺正 flow 不强制正交化，以及所有未解决统计问题的标注。

## 16. 已完成的 L32x64 权重比较（2026-10-15 核验）

### 16.1 问题、范围与共同输入

目标是量化第 10.4 节 A/B 两种估计器在**同一份输入**上的差别，而不是用不同随机基底或重新生成的规范流混合比较。对照包为

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/
```

| 项目 | 本轮实际覆盖 |
| --- | --- |
| ensemble | L32x64，$a=0.0897\,\mathrm{fm}$ |
| conf | 10000、10040、10080、10120、10160、10200、10240、10280、10320、10360 |
| 流时间和步长 | $\tau=0.5$，$\epsilon=0.01$，$N_{\rm step}=50$ |
| 窗口 | $w=4$，每个 $t_J$ 有 9 个去重后的原始时间片 |
| 输出时间片 | $t_J=0,16,32,63$，包括边界附近和内部 |
| 源/汇 | $t_{\rm sep}=7a=0.6279\,\mathrm{fm}$，$dt=3$ |
| 源/汇时间 | $t_{\rm src}=(t_J-3)\bmod64$，$t_{\rm snk}=(t_{\rm src}+7)\bmod64$ |
| 外态边界符号 | $t_{\rm src}+7\ge64$ 时为 $-1$，否则 $+1$；两方案相同 |
| 向量 | current 400；source/sink VVV 200 |
| 外态动量 | zdir，整数标签 $P_z=+3,+4,+5$，不是直接以 GeV 为单位 |
| 位移 | $|z|/a=0,\ldots,20$，minus/plus 两个 sign 分支 |
| 极化 | `nopol,pol15,pol25,pol35` 全部保留 |
| 插入自旋 | 完整 $4\times4$ 开指标，没有在本轮投影成特定 PDF 通道 |
| Wick 项 | `u1,u2,u3,u4,d1,d2`，各自系数 $-1,+1,+1,-1,+1,-1$ |

总计 $10\times4=40$ 个 current/source/sink 案例；每案 81 个有序时间块，合计 **3240 对时间块比较、19440 条逐 Wick 比较记录**。这不是全 $t_J$、全 $t_{\rm sep}$、全流时间或全六动量方向的重算。

基底、flowed gauge、普通 perambulator 均复用生产输入；不重新反演、不重新 flow、不重新抽噪声。几何 vertex 只计算一次，再分别乘 A/B 表。按照原生产顺序将加权 vertex 转成 complex64，后续 perambulator/Wick 收缩及保存的复数贡献使用 complex128。A 直接使用冻结 `create_omega(400)` 的实际浮点表，避免仅代数等价的除法顺序引入额外舍入差。

原始几何 vertex 不在已有加权文件上重复乘表，也不通过除以旧权重恢复；basis/VVV/perambulator 不新增权重。各案例收据记录了所用输入的路径、字节数、mtime、头信息及节点上读取的内容 SHA256，并在完成前核对文件未变化。

### 16.2 实现回归与不变量的实测结果

全部 40 个案例 PASS，10 个组态均有完整成功收据。已确认：

| 检查 | 结果 |
| --- | --- |
| 同时间 vertex，A/B 逐元素比较 | 40/40 完全相同 |
| 同时间六个 Wick 项，A/B 逐元素比较 | 40/40 完全相同 |
| 不同模式扇区与 exact/exact 的 vertex | 40/40 完全相同 |
| 六项复数结果有限性 | 40/40 无 NaN/Inf |
| 逐块求和的 A 与原生产 u-C3 | 最大相对 L2 差 $7.9111\times10^{-15}$ |
| 逐块求和的 A 与原生产 d-C3 | 最大相对 L2 差 $9.0944\times10^{-15}$ |
| 公共 C2 与原生产 C2 | 最大相对 L2 差 $4.3884\times10^{-15}$ |

CSV 中 **2160 条同时间逐 Wick 记录的差严格为零**；17280 条跨时间记录均有非零差异。这里同时间分支使用同一加权张量，其一致性是预期且必要的不变量；它不是独立的抽样分布证明。回归说明逐块展开、Wick 系数、边界符号和原生产因子化相容，并不说明随机估计无偏。

小张量门槛也已通过，记录在 `status/small_tests.json`：显式矩阵乘法构造 H、逐块逐项与冻结 full-spin 六项公式、逐块求和与生产时间因子化、外部 rank-2 C2 与 full-spin C2。小张量与实际组态检查分开保存，不能互相替代。

### 16.3 比较指标必须区分“先求和”和“先取范数”

令案例 $c$ 的某个带符号 Wick 项为

$$
X^{q}_{c,k}(\sigma_a,\sigma_b;\alpha),
\qquad q\in\{A,B\},
\qquad
\alpha=(\mathrm{sign},\mathrm{pol},P_z,|z|/a,m,o).
$$

$X$ 已包含对应 Wick 系数及外态跨界符号。总 C3 的构造是

$$
\begin{aligned}
C^{q}_{3,u,c}(\alpha)
&=\sum_{\sigma_a,\sigma_b}\sum_{k=u1}^{u4}
X^{q}_{c,k}(\sigma_a,\sigma_b;\alpha),\\
C^{q}_{3,d,c}(\alpha)
&=\sum_{\sigma_a,\sigma_b}\sum_{k\in\{d1,d2\}}
X^{q}_{c,k}(\sigma_a,\sigma_b;\alpha).
\end{aligned}
$$

第一种指标是已经完成所有时间/Wick 求和的 C3 整体相对变化：

$$
\delta_{c,f}^{C_3}
=\frac{\|C^{B}_{3,f,c}-C^{A}_{3,f,c}\|_2}
{\|C^{A}_{3,f,c}\|_2},
\qquad f=u,d,
$$

其中 $\|\cdot\|_2$ 是所有保留复数分量 $\alpha$ 的欧氏/Frobenius 范数。

第二种指标是固定某个 Wick 项，**保留所有时间块再堆叠取范数**：

$$
\delta_{c,k}^{\rm stack}
=\frac{\left[\sum_{\sigma_a,\sigma_b,\alpha}
|X^{B}_{c,k}-X^{A}_{c,k}|^2\right]^{1/2}}
{\left[\sum_{\sigma_a,\sigma_b,\alpha}
|X^{A}_{c,k}|^2\right]^{1/2}}.
$$

这两个指标的求和/取模顺序不同，不能把单项堆叠变化等同于该项完成时间求和后的相对变化；它们也不能直接解释成任意一条实部/虚部矩阵元的百分比变化。

### 16.4 已完成时间与 Wick 求和的 C3 变化

以下为 40 个案例的 $100\,\delta_{c,f}^{C_3}$ 的最小值、中位数、最大值；不是把全部组态平均后再计算的一个百分比。

| flavor | 最小 | 中位数 | 最大 |
| --- | ---: | ---: | ---: |
| u | 1.3827% | 1.9892% | 2.4865% |
| d | 1.7738% | 2.4576% | 3.3757% |

确认的结论是：在固定输入上，改动跨时间同随机扇区的权重会改变总 C3，不能说两种处方逐组态完全相同。没有据此证明均值偏差，也没有据此做 bare-matrix、ringed、matching 或最终 PDF 的变化结论。

### 16.5 六个 Wick 项的逐块堆叠变化

以下为 $100\,\delta_{c,k}^{\rm stack}$ 在 40 个案例上的分布：

| Wick 项 | 最小 | 中位数 | 最大 |
| --- | ---: | ---: | ---: |
| u1 | 3.9820% | 5.2467% | 5.7942% |
| u2 | 4.3698% | 5.4795% | 6.2127% |
| u3 | 4.4214% | 5.5032% | 6.2460% |
| u4 | 4.3831% | 5.4733% | 6.2734% |
| d1 | 4.8134% | 5.6873% | 6.3451% |
| d2 | 4.3831% | 5.4733% | 6.2734% |

单个时间块、某个 $z$ 或某个接近零的分量仍可能有与这些整体范数不同的相对变化，不能把上表最大值当作所有分量的统一误差上界。

### 16.6 配对统计的实际操作及边界

`summary/paired_c3_c2_samples.npz` 保留组态与四个案例轴，A/B 始终配对。现有汇总代码先在每个组态 $c$ 内平均四个案例，然后计算差值：

$$
\Delta_c(\alpha)
=\frac14\sum_{j=1}^{4}
\left[C^{B}_{3,c,j}(\alpha)-C^{A}_{3,c,j}(\alpha)\right],
\qquad
\overline\Delta=\frac1{10}\sum_{c=1}^{10}\Delta_c.
$$

保存的误差是分别对实部/虚部求十个组态的样本 SEM，例如

$$
\operatorname{SE}_{\rm Re}(\overline\Delta)
=\sqrt{\frac{1}{10\times9}\sum_{c=1}^{10}
\left[\operatorname{Re}\Delta_c-
\operatorname{Re}\overline\Delta\right]^2},
$$

虚部同理。四个 $t_J$ 不是四个额外独立组态，不能把这一步当成 40 个独立样本。

**已保存但尚未据此完成显著性判断：**配对差均值及 SEM。这里没有新增 bootstrap、没有给出独立噪声重复实验，也没有对组态自相关、$z$ 间协方差或多重比较作完整处理。四个案例平均不等于全源时间平均，单个 SEM 也不是完整系统误差。下一步应先投影具体物理通道，明确要比较 C3、ratio 还是拟合矩阵元，再使用共同组态及共同重采样索引传播误差。

### 16.7 可复现入口、文件格式和结果位置

以下链接均指向本轮冻结对照包，原生产文件没有被替换：

- [范围与源文件 SHA256 的 manifest](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/manifest.json)。
- [权重与逐块 Wick 实现](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/weights_and_terms.py)。
- [实际 GPU 比较入口](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/compare.py)。
- [小张量检查及结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/status/small_tests.json)。
- [完整汇总报告：PASS、40 案例、10conf、missing 为空](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/summary/report.json)。
- [19440 条逐块逐 Wick 记录 CSV](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/summary/ordered_block_wick_metrics.csv)。
- [配对 C3/C2 及差值实/虚 SEM](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/summary/paired_c3_c2_samples.npz)。
- [六个 Wick 项的相对差异图 PDF](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/summary/wick_relative_differences.pdf)。

每个案例的原始复数结果在

```text
<comparison-root>/results/<conf>/tJ###/comparison.npz
<comparison-root>/results/<conf>/tJ###/receipt.json
```

关键数组轴如下：

| key | 轴 |
| --- | --- |
| `signed_wick_blocks` | `(scheme,sigma_a,sigma_b,wick,sign,pol,mom,dz,m,o)` |
| `c3_by_scheme_flavor` | `(scheme,flavor,sign,pol,mom,dz,m,o)` |
| `vertex_metrics` | `(sigma_a,sigma_b,sign,dz,metric)` |
| `sector_vertex_metrics` | `(sigma_a,sigma_b,sector_a,sector_b,metric)` |
| 汇总 NPZ 的 `c3` | `(conf,case,scheme,flavor,sign,pol,mom,dz,m,o)` |

`scheme=0` 为 A，`scheme=1` 为 B；六项数组已经包含 Wick 正负系数。`vertex_metrics` 的 metric 依次为 A范数、B范数、差范数、最大绝对差。`sector_vertex_metrics` 实际保存每个 sign/dz 的范数或最大值再求和，**不是**整个扇区张量的单一 Frobenius 范数，使用时须读它的 `sector_metric_reduction` 元数据。

若只重做 CPU 汇总，不要再提交 GPU 对照或重算 basis：

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014
sbatch --parsable summarize_cpu_v2.slurm
```

CPU 汇总通过绝对路径 `/public/home/donghx/AnaConda/bin/python` 运行，NumPy1.24.4、Matplotlib3.7.0、Agg 后端、单线程数值库，无 CUDA/MPI 初始化；绘图缓存只在作业独立 `/tmp` 内临时使用。**不要继续使用旧 `summarize_cpu.slurm`。**

### 16.8 作业、完成记录和下一步

- GPU：`3850350_0` 为 conf10000，`3850351_1–9` 为其余九个组态；均于 **2026-10-14** 成功完成，退出码 0。10conf 均有 PASS 收据。
- 原 CPU 汇总 `3850352` 因启动环境中 `module` 不存在退出127，没有汇总；这不是 GPU 数值计算失败。
- 修正 CPU 启动器后，`3851153` 于 **2026-10-15** 在 `a192c1t/cpu073` 完成，退出码0:0、耗时00:01:04、stderr为空。最终报告 PASS，40/40、10/10、无缺失。
- [GPU 提交记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/submission.json) 与 [CPU 修正/补交收据](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_omega_compare_20261014/cpu_summary_resubmission_20261015.json) 分开保留。提交时的 PENDING/RUNNING 是历史快照，应以最终 Slurm 记录和成功收据为准。

下一步最具体的工作：

1. **先查随机输入定义。** 固定实际生成器版本、时间片 seed 和完整/部分标架抽样方法，验证第 10.3、10.5 节所需矩，而非只检查 Gram。
2. **再检查共享噪声。** 明确源/汇的200列所定义的随机外态目标，以及与 current 400列共享噪声时的更高阶补偿；不能默认外态系数与 current 抽样独立。
3. **量化物理通道变化。** 从已经保存的4×4插入自旋和所有极化出发，采用同一套已冻结的 unpolarized/helicity/transversity 投影；对 A/B 分别构造同定义的 C3/C2，配对重采样。此项尚未因本轮汇总而自动完成。
4. 在上述条件和目标明确之后，才决定是否扩展到其它流时间、全部64个 $t_J$、更多 $t_{\rm sep}$，以及是否需要新的独立噪声实现。不能把当前单流时间、四案例结果外推为全生产范围结论。

**不能随意修改的最终判断：**本轮已证实 A/B 的实现差异与生产回归；没有证实哪种实际完整核子估计器无偏或方差更小。不能将统一表标成“已确认错误”，也不能将几个百分点差异标成“可忽略”或“最终PDF系统误差”。

本次仅更新说明文档和工作目录中的备份/发布副本；没有修改生产代码、冻结 manifest、输入数据、缓存或相关函数，没有新提交任务，也没有读取或写入任何凭据。

生成日期：**2026-10-14（Asia/Shanghai）**。

最近更新日期：**2026-10-15（Asia/Shanghai）**。

本次会话标题：**同时间二阶／独立跨时间乘积权重的估计推导与 L32x64 十组态比较**。
