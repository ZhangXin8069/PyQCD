# LQCDDB 物理约束下的代码正确性复审报告

## 1. 复审范围与原则

- 复审日期：2026-08-23
- 审查对象：`src/lqcddb/`、`README.md`、`physics.md`、`pyproject.toml`、`changelog.md`、公开类型存根及测试脚本
- 当前源码树聚合 SHA-256：`66aecdd60fa79b445a9696ff1cbf5a7fa2039c5cb1a528e63fb2ca9b736ab3b2`
- 源码策略：`src/lqcddb/` 与 `src/lqcddb/test/` 全程只读

复审时不再把任意数组或跨物理通道输入都视为有效反例，而是要求反例满足相应物理输入域。例如：

- 同一次等价图归并中的算符项属于同一物理过程；
- Laplacian 本征矢来自复内积空间并应正交归一；
- perambulator 满足相应的 $\gamma_5$ 厄米关系；
- 重子和介子顶点具有 VVV 的颜色反对称性及 VDV 的厄米结构；
- GEVP 输入来自合法的谱表示；
- gauge link 属于 SU(3)，并遵守明确的方向—数组轴约定；
- Jackknife/Bootstrap 输入代表实际组态样本，而不是任意非法尺寸。

问题分为四类：

1. **确认的物理/数值缺陷**：物理允许的输入可直接触发错误结果；
2. **有条件的科学限制**：在明确限制域内正确，超出该域会改变物理结论；
3. **接口与工程问题**：通常不改变合法物理路径，但会导致误用、崩溃或不可复现；
4. **已撤回的旧结论**：此前反例违反物理输入前提，不能继续作为缺陷证据。

## 2. 总体结论

复审后，旧报告“当前版本整体不能用于生产计算”的结论过强。以下核心路径在规定约束下通过了独立检查：

- DR 基 gamma 矩阵代数；
- pion/proton 基础 Wick 符号；
- 仓库自带同一物理过程示例中的等价图关系；
- `seq_peram` 的 $\gamma_5$ 厄米反向关系；
- VDV/VVV 的零位移、方向平均及守恒流共轭关系；
- `ratio_3pt` 的二维时间重索引；
- 常用 `create_omega_accelerate(..., normal=False)` 参数路径；
- 标准实数 Jackknife 主流程。

但仍存在若干由物理允许输入触发的实质问题，最重要的是：

1. Stout smearing 的空间方向与数组轴不一致，并在零生成元处产生 NaN；
2. GEVP 强制丢弃复厄米关联矩阵的虚部；
3. `ratio_3pt` 的统计输出强制丢弃虚部，无法处理本应从虚部提取的矩阵元；
4. 多重子算符的 Hermitian 共轭没有反转强子次序，带系数算符的系数也被丢弃；
5. 默认 m-out-of-n Bootstrap 没有误差缩放；
6. 复数协方差及非首轴 Jackknife covariance 存在错误；
7. `[Z,Y,X]` 顺序下的 `Mom_cross_sigma` 与标准右手坐标叉积相差整体负号。

## 3. 已撤回并修正的原 C1：等价图

### 3.1 原反例为什么无效

旧报告把使用 `gamma_5` 的赝标量介子和使用 `gamma_1` 的矢量介子一起传给 `identify_equivalent_diagrams()`，然后据此认定等价图归并错误。

这个反例违反了 `run_wick_analysis()` 在 `dynamic.py:602` 给出的输入约定：

> `operator_groups` 的所有条目视为同一个过程。

通常零动量谱学中，

$$
\bar q\gamma_5 q
$$

和

$$
\bar q\gamma_i q
$$

分别属于赝标量与矢量通道，不应在同一次待求和的物理过程中出现。两者具有相同 Wick 连接拓扑，并不意味着它们应被作为同一物理关联函数求和。

因此，跨通道 `gamma_5/gamma_1` 测试只能说明函数没有检查物理通道，不能证明合法计算中的归并错误。

### 3.2 使用物理约束重新验证

复审使用仓库自带 `identify_equivalent_diagrams_test.py` 中的同一过程多算符项，并施加：

- $u,d$ 轻味简并；
- 等时 perambulator 的 $\gamma_5$ 厄米关系；
- 从同一组正交 distillation eigenvectors 构造的 VVV/VDV；
- VVV 的颜色完全反对称结构；
- 源端 VVV/VDV 的正确复共轭关系；
- 实际 `gamma_7=C\gamma_5` 和 `gamma_5` 矩阵。

对于全部 7 个等价类，类内关系

$$
\mathcal C_i=c_i\,\mathcal C_{\mathrm{rep}}
$$

的最大相对残差小于

$$
4.3\times10^{-16}.
$$

这说明旧报告 C1 的“确认科学错误”结论应撤回。

### 3.3 仍需保留的适用前提

等价图优化仍依赖以下前提，但目前没有发现合法示例违反它们：

- 所有 `operator_groups` 确实属于同一物理过程；
- `autowick.py:596` 的 $d\to u$ 归一化只适用于精确的轻味简并；
- VVV/VDV 必须来自真实 eigenvector 构造，不能替换成彼此独立的任意随机张量；
- 参数化 `gamma_mu`、`sigma_mu_nu` 只有在精确处理分量转置符号时才能参与归并；
- 不同动量、位移、smearing 或 lattice irrep 的算符项不能只因 einsum 模板相同而数值归并。

当前代码会对参数化 gamma 给出警告，且常用 pion、proton 两点及 proton 矢量流三点示例中的等价类实际都是单元素类。因此，本项重新分类为：

> **输入前提未自动验证的有条件优化，而不是已确认的物理错误。**

## 4. 确认的物理或数值缺陷

### P1. Stout smearing 混入错误方向，并在零生成元处产生 NaN

- 位置：`src/lqcddb/base/smear_gauge.py:5-75`
- README：`README.md:748-758`
- 严重度：严重

源码中的七轴转置表明实际实现期望整套规范场：

$$
(\mu,t,z,y,x,c,c).
$$

测试脚本读取规范场后也使用该布局。但是 README 声明的输入缺少时间轴：

$$
(\mu,z,y,x,c,c).
$$

对代码实际使用的七维布局，空间方向映射为：

| link 方向 | gauge 方向索引 | 正确坐标轴 | 代码 `3-nu` 使用的轴 |
|---|---:|---:|---:|
| X | 0 | 4 | 3（Y） |
| Y | 1 | 3 | 2（Z） |
| Z | 2 | 2 | 1（时间） |

因此 Z staple 会读取相邻时间片，而不是相邻 Z 坐标。这不是边界输入问题，而是对正常时变规范场直接改变 smearing 的物理方向。

此外，单位规范场是合法的 SU(3) 场。对七维单位场取 `nstep=1`，无论 `rho=0` 还是 `rho=0.12`，当前实现均产生 432 个 NaN。原因是 $Q=0$ 时：

$$
c_0=c_1=0,
$$

随后代码计算 $c_0/c_{0,\max}$ 及 $1/(9u^2-w^2)$，没有实现 $Q\to0$ 的解析极限。

必要不变量

$$
U_\mu(x)=I\quad\Longrightarrow\quad U'_\mu(x)=I
$$

不成立。

### P2. 复厄米谱关联矩阵在 GEVP 中被错误实数化

- 位置：`src/lqcddb/analyse/analyse.py:817-898`
- 关键代码：`analyse.py:863`
- 严重度：高；若算符基被严格证明为实，则该路径不受影响

合法的谱表示为

$$
C(t)=Z\,\mathrm{diag}\!\left(e^{-E_n t}\right)Z^\dagger,
$$

其中 $Z$ 可以是复矩阵，所以 $C(t)$ 一般是复厄米矩阵，而不必是实对称矩阵。

当前代码执行：

```python
C = ((C.conj().transpose(1, 0, 2) + C) / 2).real
```

复审使用上述谱表示构造正定的 $2\times2$ 关联矩阵。参考广义本征值为

$$
(0.8,\,0.3),
$$

当前实现返回

$$
(0.55,\,0.55).
$$

这不是任意随机矩阵反例，而是合法的有限态谱关联矩阵。

如果项目只允许经过对称投影、并已证明期望值为实的算符基，必须在接口中明确说明并检查虚部残差；否则不应调用 `.real`。

此外，每个时间片独立排序本征值，在能级交叉或近简并时不能稳定追踪态。应使用相邻时间本征矢重叠或固定参考本征矢进行 state tracking。

### P3. `ratio_3pt` 的均值和误差无条件丢弃虚部

- 位置：`src/lqcddb/analyse/analyse.py:803-809`
- 严重度：高；只对预先证明为实的投影安全

代码保留复数 `ratio_sample`，但返回的统计量固定使用：

```python
ratio_sample.real
```

许多三点函数在特定动量、流分量和 projector 下，物理矩阵元来自关联函数的虚部。最小合法测试取初末态相同、两点函数为正实数、三点投影结果为纯虚数，得到：

```text
ratio_sample = 1j
data_mean    = 0
data_err     = 0
```

因此函数会把非零物理信号报告为零。接口应允许选择 `real`、`imag` 或保留复数，并使用与选择一致的统计处理。

### P4. `ratio_3pt` 的 1D 模式没有对齐不同位置的时间轴

- 位置：`src/lqcddb/analyse/analyse.py:636-723`
- 严重度：中等

不同数组采用不同轴顺序不违反任何物理条件，而且函数文档明确声称支持 `tau_axes` 与 `t_sink_axes` 位于不同位置。

对正实两点函数使用：

```text
C3.shape = (5,4,1), tau_axes=1
C2.shape = (5,1,4), t_sink_axes=2
```

预期输出形状为 `(5,4,1)`，实际通过广播得到 `(5,4,4)`。

二维模式在相同初末态的合成谱测试中满足

$$
R=C_3/C_2(t_{\mathrm{sep}})
$$

且最大残差为零，因此问题限于 1D 的异位置轴路径。

### P5. m-out-of-n Bootstrap 未进行误差尺度修正

- 位置：`src/lqcddb/analyse/analyse.py:401-519`
- 严重度：高（统计误差系统偏差）

默认设置为：

$$
M=N_{\mathrm{conf}}-5,
$$

但每个 bootstrap replicate 只抽取 $M$ 个组态。对样本均值，重采样分布方差近似为

$$
\mathrm{Var}_*\!\left(\bar X_M^*\right)
\simeq \frac{s^2}{M},
$$

而原始 $N_{\mathrm{conf}}$ 个组态均值的目标方差是

$$
\mathrm{Var}(\bar X_N)
\simeq \frac{s^2}{N_{\mathrm{conf}}}.
$$

因此 m-out-of-n bootstrap 的均值误差至少需要相应的

$$
\sqrt{\frac{M}{N_{\mathrm{conf}}}}
$$

尺度修正。当前代码没有该因子。

在 $N_{\mathrm{conf}}=20$、$M=15$、30000 个 resamples 的测试中：

```text
lqcddb error             = 1.4860
M=15 重采样分布预期      = 1.4888
Nconf=20 原始均值目标误差 = 1.2894
```

即代码忠实给出了 $M$ 样本均值的宽度，而不是原始 $N_{\mathrm{conf}}$ 样本估计量的误差。

此外，第 0 个样本被硬编码为原始均值并混入 bootstrap replicates，协方差除以 $N$ 而非按明确的 bootstrap convention 处理，都会产生额外的有限样本偏差。

### P6. Jackknife covariance 在组态轴不位于第 0 轴时失败

- 位置：`src/lqcddb/analyse/analyse.py:324-399`
- 关键代码：`analyse.py:375-376`
- 严重度：中等

`Nconf_axes` 明确允许任意轴，README 的 GEVP 示例甚至使用 `Nconf_axes=-1`。但 `data_mean` 删除组态轴后，代码直接执行：

```python
residual = data_sample - data_mean
```

没有在原组态轴位置恢复 singleton 维。

对形状 `(Nt,Nconf)=(4,5)`、`Nconf_axes=1`、`cov_axes=0` 的合法输入，直接得到广播错误：

```text
ValueError: operands could not be broadcast together with shapes (4,5) (4,)
```

Jackknife 样本、均值和误差在不请求 covariance 时仍可工作；缺陷限于 covariance 路径。

### P7. 复数 Jackknife/Bootstrap covariance 不是 Hermitian covariance

- 位置：`src/lqcddb/analyse/analyse.py:393,514`
- 严重度：高；实数投影后的数据不受影响

当前实现使用

$$
C_{ij}^{\mathrm{code}}
\propto
\sum_k r_{k,i}r_{k,j},
$$

没有对第二个残差取复共轭。对一般复关联函数，通常用于相关拟合的 covariance 应为

$$
C_{ij}
\propto
\sum_k r_{k,i}r_{k,j}^*,
$$

或把实部和虚部堆叠为实向量后构造实 covariance。

复数测试中，当前输出的厄米性残差为

$$
\max|C-C^\dagger|=1.3333,
$$

而带共轭的参考结果残差为零。当前代码实际计算的是 pseudo-covariance；它可以作为额外统计量，但不能单独代替复随机变量的完整 covariance。

### P8. 多重子算符的 Hermitian 共轭没有反转强子次序

- 位置：`src/lqcddb/contraction/baroperator.py:157-172,256-265`
- 严重度：高，但只影响多重子/含奇 Grassmann 奇偶性的多强子算符

对于两个重子算符，严格的 Hermitian 共轭为

$$
(B_1B_2)^\dagger=B_2^\dagger B_1^\dagger.
$$

`conjugate_operator()` 分别共轭每个 hadron，但保持原 hadron 顺序，返回 $B_1^\dagger B_2^\dagger$。若重新交换到该顺序，需要跨越 $3\times3$ 个 Grassmann 场，因此产生

$$
(-1)^{3\times3}=-1.
$$

该符号对单个重子、两个介子或重子—介子组合不产生同样问题，但会影响双重子及更一般多重子算符的相对相位。

另外，`conjugate_generic()` 会把所有 quark 反序后把所有 gamma token 统一放到末端，不能保持一般五夸克/多夸克算符内部的 gamma 配对结构。

### P9. 带系数算符的 Hermitian 共轭会丢失系数

- 位置：`src/lqcddb/contraction/baroperator.py:157-172,256-265`
- 严重度：高；影响 Lambda、同位旋投影及任意算符线性组合

对于

$$
\mathcal O=\sum_r c_r\mathcal O_r,
$$

源算符应为

$$
\mathcal O^\dagger=\sum_r c_r^*\mathcal O_r^\dagger.
$$

但数值 token 位于首个 `|` 之前时，`split_hadrons()` 在遇到 `|` 时重置临时列表，因此系数被丢弃。实测：

```text
conjugate_operator([2, ...])
conjugate_operator([1+2j, ...])
```

返回完全相同的结果，既没有保留 2，也没有产生 $1-2i$。

单一算符的整体归一化可能在质量提取中消去，但 Lambda 等多项算符的相对系数、同位旋 Clebsch–Gordan 组合及跃迁矩阵元会受影响。

### P10. `[Z,Y,X]` 顺序下的 `Mom_cross_sigma` 与标准叉积相差负号

- 位置：`src/lqcddb/constant/sigma_matrix.py:97-133`
- 严重度：中到高，取决于该结构是否进入 projector/流定义

函数把动量与 Pauli 矩阵排列为 `[Z,Y,X]`，但直接使用满足 $\epsilon_{012}=+1$ 的标准 Levi-Civita 张量。

排列 $(Z,Y,X)$ 相对右手系 $(X,Y,Z)$ 是奇置换，因此若物理上仍定义标准右手 Cartesian 叉积，Levi-Civita 张量应多一个负号。

取正 X 方向动量，即存储为

$$
\mathbf p=[0,0,1]_{ZYX},
$$

标准结果应为

$$
(\mathbf p\times\boldsymbol\sigma)_{ZYX}
=
(\sigma_y,-\sigma_z,0),
$$

当前代码返回其整体负值。

包内 `PFF_Mom_to_gamma_new()` 会先把 `[Z,Y,X]` 反转为标准顺序，而 `Mom_cross_sigma()` 没有这样做，两个模块的取向约定也不一致。

### P11. `vector_creator.check()` 仍会忽略纯虚正交残差

- 位置：`src/lqcddb/eigvectors/vector.py:37-96`
- 严重度：中等验证风险

当前源码已修复旧报告中的负实重叠问题：现在使用 `abs(A)` 比较。但它在此之前执行：

```python
A = contract(...).real
```

因此纯虚 overlap 被先删除。

两个归一化但平行的复矢量

$$
v_1=(1,0),\qquad v_2=(i,0)
$$

具有

$$
|v_1^\dagger v_2|=1,
$$

但 `vector_creator.check()` 返回 `True`。当前 `vertex_creator.check()` 保留复 Gram 矩阵，能够正确拒绝该输入。

正常 Hermitian Laplacian eigensolver 通常会输出正交矢量，因此这是验证器不能发现上游失败的问题，不代表正常 eigensolver 必然产生错误。

### P12. `create_omega_accelerate(normal=True)` 不能同时保持行归一化和对称性

- 位置：`src/lqcddb/eigvectors/vertex.py:320-329`
- 严重度：中等；默认 `normal=False` 不受影响

代码先逐行归一化，再把下三角复制到上三角。第二步会重新改变部分行和。

在所有抽样 part 大小均满足 dim=2 要求的合法小规模参数下，输出矩阵大小为 $9\times9$。`normal=True` 后预期各行和为 9，但实际为：

```text
[3.375, 3.375, 3.375, 3.375, 3.375, 3.375, 9, 9, 9]
```

因此 README 所称“行归一化并对称化”没有同时实现。常用测试参数在 `normal=False` 下输出有限、对称，未发现该路径问题。

## 5. 需要明确物理约定后才能定性的事项

### Q1. 弱流的手征投影顺序

- 位置：
  - `physics.md:20`
  - `README.md:873-877`
  - `src/lqcddb/test/contraction.lambda.proton.weak.mpi.py:132-136`

标准左手 charged weak current 通常写为

$$
J_\mu^{V-A}=\bar u\,\gamma_\mu(1-\gamma_5)s.
$$

README 和测试代码构造的是

$$
(1-\gamma_5)\gamma_\mu
=
\gamma_\mu(1+\gamma_5),
$$

即通常所称的 $V+A$ 矩阵顺序。数值验证也表明 README 矩阵等于 $\gamma_\mu(1+\gamma_5)$，不等于 $\gamma_\mu(1-\gamma_5)$。

如果这里有意计算 Hermitian-conjugate current、采用特殊 Euclidean continuation，或 flavor 流向与常见写法相反，需要明确给出推导。当前 `physics.md` 将其直接称为 weak current，容易造成轴矢部分整体符号错误。

### Q2. 电磁流缺少味电荷权重

`physics.md` 把电磁流写成：

$$
\bar q\gamma_\mu q,
$$

其中 wildcard `q` 枚举 $u,d,s,c,b,t$。物理电磁流应为

$$
J_\mu^{\mathrm{EM}}
=
\sum_f Q_f\bar q_f\gamma_\mu q_f.
$$

自动 Wick 输出不会加入 $Q_u=2/3$、$Q_d=Q_s=-1/3$ 等电荷，也会生成 top loop。若文档仅把 wildcard 当作“生成各味结果后由用户加权”的工具，必须明确说明，不能直接对所有输出等权求和。

此外，`wick_contraction()` 对 wildcard 返回 `list[dict]`，而动态框架按单个 dict 使用结果，不能直接接收该返回结构。

### Q3. `ignore_dis` 的适用范围

`dynamic.py:825-835` 用“存在 $t_q=t_{\bar q}$ 的 peram”标记 disconnected 图。对单强子二点/三点中的局域同时间 quark loop，这通常能识别要忽略的 disconnected contribution。

但对多强子或交换拓扑，同时间 propagator 不自动意味着整个 quark-line graph 与其它时间区域断开。因为包支持多强子算符，该条件应明确限制为经过人工确认的拓扑；尚未构造出当前仓库正常单强子流程中的错误删除反例。

### Q4. 复数关联函数的实投影策略

GEVP、ratio 和 covariance 的问题都可在项目强制以下流程时规避：

1. 先依据 parity、time reversal、momentum pairing 或 projector 证明目标期望值为实；
2. 在每个组态或 resample 上按相同规则投影；
3. 检查被舍弃虚部与零统计相容；
4. 再进入只接受实数的分析函数。

当前代码没有表达或验证这些前提，因此本报告将其列为真实的通用 API 问题，而不是断言项目所有零动量结果都错误。

### Q5. `Mom2GeV` 使用连续色散关系

`Mom2GeV()` 使用

$$
E=\sqrt{m^2+\mathbf p^2}.
$$

这可作为物理单位转换或连续色散近似，但有限格距下并非所有格点作用量的精确色散关系。某些分析需要使用相应作用量的 lattice dispersion，例如包含 $\sin(ap_i/2)$ 与 $\sinh(aE/2)$ 的形式。

函数名和 README 应说明这是连续色散假设，而不是普适的格点能量定义。`M0` 为质量列表时，所有粒子还共用同一个 $|\mathbf p|^2$，不能描述一般多粒子总动量分配。

### Q6. 有效质量时间坐标与无效端点

`meff(..., meff_type='cosh')` 计算

$$
\operatorname{arccosh}
\frac{C(t)+C(t+2)}{2C(t+1)},
$$

该估计量自然对应中心时间 $t+1$，但结果存储在数组索引 $t$。log estimator 也自然位于半整数时间。当前接口没有返回相应时间坐标，并把不能计算的尾端初始化为零。

这些零不应进入 plateau fit；更合理的表示是只返回有效区间或使用 NaN/mask。

## 6. 接口、健壮性与工程问题

以下问题不应与“物理结果已经错误”等同，但会妨碍可靠研究流程。

| 编号 | 位置 | 结论 |
|---|---|---|
| E1 | `README.md:8`, `lqcddb/__init__.py`, `base/mpi_init.py:6` | `from lqcddb import *` 会解析可选 MPI 符号并初始化 MPI；本环境中触发 MPI fatal。应使用具名导入。 |
| E2 | `contraction/dynamic.py:18` | 动态收缩在模块导入时加载 MPI helper，因而 MPI 虽在 `pyproject.toml` 中可选，动态 API 实际依赖 MPI。 |
| E3 | `eigvectors/vertex.py:78-108` | `src_sink_MPI_tran()` 引用不存在的 `self.backend_name`，公开方法不可用。 |
| E4 | `eigvectors/vertex.py:127-158`, `README.md:422` | `create_omega_accelerate` 默认 `exact=0`，但函数立即要求 `exact>0`。项目常用 `exact=50`，所以这是默认值/API 问题，不再列为纯随机 distillation 物理错误。 |
| E5 | `eigvectors/vector.py:191-201` | V1 `Ctype='BI'` 仅一组时变量 `i` 未赋值；README 已说明 BI 需要至少两组，因此属于非法参数校验不足。 |
| E6 | `eigvectors/vector.py:357-445` | V4 的字面量 `'Z_N'` 不能解析；Z3 不是三次单位根，Z8 高阶相位模长不为 1。常用 Z4 列表正确，但随机种子也不可控。 |
| E7 | `base/base_functions.py:244-345` | `cached_contract` 对 ellipsis 批维从左对齐，拒绝 NumPy 合法的右对齐广播；现有包内收缩大多不用这种模式。 |
| E8 | `contraction/contractadviser.py` | ellipsis 输出大小漏算批维，V100/A100 `compute_efficiency` 大于 1；只能视为启发式性能工具，不能影响物理结论。 |
| E9 | `io/write_data.py:38-83` | `write_data_ascii` 使用纯文件名时尝试创建空目录名；显式父目录路径可工作。 |
| E10 | `io/write_data.py:6-13` | `readin_eigvecs` 未关闭文件句柄。 |
| E11 | `io/write_data.py` | vendored `savetxt` 依赖旧 `numpy.compat`，NumPy 2.x 存在兼容风险；多数据列的固定 `fmt` 也有限制。 |
| E12 | `base/mpi_init.py` | 单 rank Gather/Allgather 的返回形状与多 rank 路径不完全一致；串行调试不能自动代表 MPI 形状。 |
| E13 | `constant/gamma_matrix.py:108-110` | 非法 gamma 索引调用 `sys.exit()`；应抛异常。物理代码不会主动使用非法索引，因此不是物理缺陷。 |
| E14 | `constant/sigma_matrix.py:25-49` | 非法 sigma 索引静默返回 `None`。 |
| E15 | `constant/gamma_matrix.py:137-180` | `PFF_Mom_to_gamma_new` 不能处理同一批中非零分量数不同的动量；README 已提示该限制。 |
| E16 | `base/base_functions.py:39-99` | `creat_mom_list` 某些输入产生重复动量；动量壳平均前需去重并核对简并度。 |
| E17 | 多处 | `raise print(...)` 最终产生 `TypeError`，不是预期异常。 |
| E18 | `io/write_data.py` | `safe_save` 自动回退到 `/nexdata/...` 是站点特定行为，不适合作为可移植默认值。 |

## 7. README、类型存根与打包问题

1. `README.md:107` 称 Wick 前缀未提供时自动生成大写字母，源码实际补空字符串；
2. `README.md:560` 的 Jackknife delete-one 公式整体符号写反，源码实现是正确的；
3. README 的 Stout 输入形状缺少源码实际使用的时间轴；
4. README 把 `meff` 的 float64 限制称为“物理要求”，实际只是当前实现限制；
5. README 列出的部分 MPI 查询函数没有从顶层导出；
6. `pyproject.toml` 和 `__version__` 为 `0.0.2`，`changelog.md` 已声明 `0.0.3`；
7. `src/lqcddb/__init__.pyi` 中 `conjugate_operator`、等价图返回类型、动态收缩和 `VdV_sink_t_link` 等签名过时；
8. `pyproject.toml` 声明包数据 `README.md/physics.md`，但文件位于仓库根目录，现有 `build/lib/lqcddb/` 中没有它们；
9. 多个测试脚本仍使用旧的 `vertex_creator(..., backend=...)`、`Mom_VVV_sink_t_3`、`VdV_sink_t_link(..., t=...)` 和 `calculate_all(optimize=...)`；
10. `dynamic.py` 用 `startswith('gamma_')` 统计 gamma 数量，`C * gamma_5` 不会被计数，可能使 Projection/Gindex 的附加标签错位；
11. `autowick.py` 只用子串 `'gamma'` 识别 gamma token，所以已在 `baroperator` 中支持的独立 `C` token 会被丢弃。

## 8. 通过的物理与数值检查

### 8.1 Gamma 矩阵

$$
\max_{\mu\nu}
\left|
\{\gamma_\mu,\gamma_\nu\}-2\delta_{\mu\nu}
\right|=0,
$$

$$
\max_\mu|\gamma_\mu^\dagger-\gamma_\mu|=0,
$$

$$
|\gamma_5-\gamma_1\gamma_2\gamma_3\gamma_4|=0.
$$

`gamma(7)=C\gamma_5` 的当前约定也与重子算符用法一致。

### 8.2 基础 Wick 符号

使用 `conjugate_operator()` 生成单强子 source，并保留其返回符号：

```text
pion 2pt   : [+1]
proton 2pt : [-1,+1]
```

与手工费米排列一致。

### 8.3 物理约束后的等价图

如第 3 节所述，仓库自带同一过程多算符示例的全部等价类在物理张量约束下满足预期关系，最大相对残差约 $4.3\times10^{-16}$。

常用单强子示例中：

```text
pion 2pt   : 1 张图，1 个单元素等价类
proton 2pt : 2 张图，2 个单元素等价类
proton 3pt : 6 张图，6 个单元素等价类
```

即这些流程启用 `use_equivalence=True` 时没有实际把不同单图合并。

### 8.4 VDV/VVV

对正交 eigenvectors 和单位 gauge link：

- `link_dir='all'` 与 X/Y/Z 三方向结果平均的最大误差为 0；
- 守恒流两个方向满足 Hermitian 共轭，最大误差约 $1.1\times10^{-17}$；
- 零位移 VDV 等于本征矢 Gram 矩阵，最大误差约 $2.2\times10^{-16}$。

因此旧报告中关于当前 `VdV_sink_t_link('all')` X/Z 错配的结论不再成立；当前源码已经使用正确的 `[3,2,1]` roll 轴。

### 8.5 统计和比值的安全子域

- 标准实数 delete-one Jackknife 样本、均值和误差正确；
- `ratio_3pt` 的二维模式在相同初末态、正实两点函数测试中最大残差为 0；
- `meff` 的 log/cosh 数值公式在其有效时间范围内与当前文档公式一致，但时间坐标和无效端点需按 Q6 处理。

### 8.6 Omega 常用路径

测试脚本使用的

```text
exact=50,
N_eigen=[100,200,300,500,800],
N_sum=[20,20,20,20,20],
N_extract=[10,10,10,10,10],
noise=500,
normal=False
```

产生有限、对称的 $(650,650)$ 权重矩阵，未出现旧报告所称的第 0 part 对角错误。第 0 part 是 exact sector，其权重恒为 1，跳过随机抽样对角修正不改变结果。

### 8.7 语法与现有测试

- 36 个 Python 文件通过只读 AST 解析，0 个语法错误；
- `src/lqcddb/test/test.cache.py` 的 15 项现有测试全部通过；
- 当前环境未安装 pytest，因此未执行 pytest collection；
- MPI 初始化在当前沙箱中失败，未重新执行多 rank 数值测试；
- 未验证 CuPy/GPU 路径和真实 ensemble 的最终物理结果。

## 9. 建议处理优先级

### 优先级 A：可能直接改变物理结果

1. 修复 Stout 的方向—轴映射和 $Q=0$ 极限；
2. GEVP 保留复厄米矩阵，或显式验证实投影前提；
3. `ratio_3pt` 增加 `component='real'/'imag'/'complex'` 并修正 1D 轴对齐；
4. 修正 m-out-of-n Bootstrap 的目标估计量和尺度；
5. 修正多重子共轭顺序及算符系数的复共轭；
6. 明确并修正 `Mom_cross_sigma` 的坐标手性；
7. 确认弱流到底采用 $V-A$、$V+A$ 还是 Hermitian-conjugate convention。

### 优先级 B：统计和验证可靠性

1. 修正任意 `Nconf_axes` 的 Jackknife covariance 广播；
2. 对复数据返回 Hermitian covariance，或显式返回 covariance 与 pseudo-covariance；
3. 本征矢检查保留完整复 Gram 矩阵；
4. 为有效质量返回明确的时间坐标和 mask；
5. 为 Omega 的 `normal=True` 明确定义并验证不变量。

### 优先级 C：接口与文档

1. 为等价图写明“同一过程、轻味简并、真实顶点张量”等前提；
2. 补充电磁味电荷和弱流手征约定；
3. 更新 README、类型存根、版本号、package-data 和旧测试脚本；
4. 避免通配符顶层导入和模块导入时 MPI 初始化；
5. 增加物理金标准测试，而不仅是形状或自洽测试。

建议新增的回归测试包括：

- 单位规范场 Stout 固定点及各空间方向 staple；
- 复谱表示 GEVP；
- 纯虚三点矩阵元投影；
- m-out-of-n Bootstrap 均值误差缩放；
- 双重子 Hermitian 共轭顺序；
- 复数 covariance 厄米性；
- `[Z,Y,X]` 与 `[X,Y,Z]` 叉积取向；
- 带真实 VVV/VDV 的等价图恒等式。

## 10. 文件修改声明

- 本轮更新：`review_report.md`
- 本轮未修改：`src/lqcddb/`、`src/lqcddb/test/`、README、physics.md、Skill 及其它项目源码
- 旧报告备份：`/tmp/review_report.before-physical-review-2026-08-23.md`
- 审查期间发现 `vector.py` 和 `vertex.py` 在 2026-08-23 13:18–13:19 被外部更新；本报告以更新后的源码为准
