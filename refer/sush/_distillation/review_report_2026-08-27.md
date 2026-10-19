# LQCDDB 物理与实现独立复审报告

> **后续更新**：本报告中关于 `conjugate_operator` 系数、多强子次序和 `gamma_7` alias 的问题已于 2026-08-28 修复。当前实现、物理推导和测试见 [conjugate_operator_fix_report_2026-08-28.md](/public/home/sush/distillation/lqcddb_update/conjugate_operator_fix_report_2026-08-28.md)。其它结论仍需按各节记录的源码哈希理解。

## 1. 结论摘要

- 复审日期：2026-08-27
- C1 勘误日期：2026-08-28
- 工作区：`/public/home/sush/distillation/lqcddb_update`
- 参考材料：`review_report.md`（2026-08-23）
- 初次复审时 38 个 Python 源文件聚合 SHA-256：`36d3f6f12d2a31fbb5585a23360675a02ae638cc56b377e549a4db36783a30b2`
- C1 勘误复测时聚合 SHA-256：`087a77e6da01868ec0c44614684d6031c1a7c61e7da714458e7e2faac7676f0c`
- 源码策略：`src/lqcddb/` 与 `src/lqcddb/test/` 全程只读

> **重要勘误（2026-08-28）**：本报告初稿误把 `U[mu]` 切片后的轴编号继续按未切片七维数组计数，从而错误判定 `3-nu` 方向映射错位。实际上 `U[mu]` 已去掉 $\mu$ 轴，布局为 `(t,z,y,x,c,c)`，所以 X/Y/Z 正确对应轴 3/2/1。该方向错误结论及由错误 `4-nu` 参考得到的数值反例已全部撤回。C1 现在仅保留独立复现的六维文档/七维实现不一致与 $Q=0$ 数值奇点。

> **源码版本范围**：2026-08-28 检查到 `smear_gauge.py`、`analyse.py` 和 `contraction.lambda.proton.weak.mpi.py` 有初审后的外部更新时间。Codex 未修改这些源码。本次只针对当前 `smear_gauge.py` 重做并修正 C1；其余章节仍对应初次复审哈希，若外部更新涉及相应实现，应另行复测，不能直接把旧反例套到新源码。

本轮不能给出“整个 lqcddb 包已经可以无条件用于生产物理分析”的结论。当前版本确实修复或显著加强了若干旧问题，但仍有多项会静默改变规范场、Wick 拓扑、算符相位、弱流手征结构、GEVP 本征态或统计误差的缺陷。

当前可确认正确或在明确限制下正确的主要子路径为：

1. DR 基 Euclidean gamma 代数、$C$ 矩阵恒等式及 Pauli 代数；
2. pion/proton 的基础 Wick 配对符号（使用明确的 `C * gamma_5` 重子结构时）；
3. `seq_peram` 的指标感知 $\gamma_5$ 厄米反向关系；
4. 零位移/空间位移 VDV、VVV 全反对称性、动量共轭关系；
5. 新的严格等价图实现，在同一张量身份、实际 gamma 转置符号、真实 VVV 反对称性和精确轻味简并成立时；
6. `ratio_3pt` 的同轴一维公式、二维时间重索引及复数信号保留；
7. 标准实数、组态轴为第 0 轴且不请求复 covariance 的 delete-one Jackknife；
8. Stout 在实际七维布局下的 X/Y/Z staple、泛型 $e^{iQ}U$ 及 SU(3) 保持（不含 $Q=0$ 奇点）；
9. 当前 `write_data_ascii` 的 basename 路径和 NumPy 2.x 主路径。

生产计算前必须优先处理的结论为：

1. **Stout 的方向实现不需要修改，但在修复 $Q=0$ 极限并统一输入 rank 前不应直接进入生产流程**；
2. **多强子计算必须设置 `ignore_dis=False`，直至改为图论连通性判定**；
3. **不能直接依赖当前 `conjugate_operator` 生成带系数、多重子、`gamma_7` 或参数化 gamma 的 source 算符**；
4. **当前 Lambda→proton 示例实现的是 $V+A$，不是标准模型左手 $V-A$ 弱流**；
5. **GEVP 本征值的复矩阵问题已修复，但返回的复本征矢仍被截成实数**；
6. **Bootstrap、复 covariance、非首轴 Jackknife covariance 和 `ratio_3pt` 异位置轴仍不可直接用于通用分析**。

## 2. 审查约定与判据

本轮按以下链条核查：

$$
\text{物理定义}
\longrightarrow
\text{约定与归一化}
\longrightarrow
\text{张量公式}
\longrightarrow
\text{数值算法}
\longrightarrow
\text{实现}.
$$

采用的当前包约定为：

- Euclidean gamma 矩阵满足

  $$
  \{\gamma_\mu,\gamma_\nu\}=2\delta_{\mu\nu},
  $$

  `gamma(1..3)` 为空间分量，`gamma(4)` 为 Euclidean 时间分量，`gamma(0)=I`；
- $\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4$；
- $C=\gamma_4\gamma_2$，且当前基中

  $$
  C^T=-C,\qquad C\gamma_\mu C^{-1}=-\gamma_\mu^T;
  $$

- `gamma(7)=gamma(3)@gamma(1)=C gamma_5`；
- 本征矢布局为 `(Nev,Nz,Ny,Nx,Nc)`，动量存储顺序为 `[pz,py,px]`；
- Fourier 相位采用 $e^{-i\mathbf p\cdot\mathbf x}$；
- perambulator 基本布局为 `(d_sink,d_src,ev_sink,ev_src)`，时间方向由 `(t_quark,t_antiquark)` 指定；
- 空间 gauge 方向编号为 `X=0,Y=1,Z=2`，时间方向为 `T=3`。

问题分类：

- **严重**：可静默改变关联函数、整体/相对符号、Wick 拓扑、手征流、规范场或本征态；
- **高**：物理允许输入给出错误统计量、错误轴或无效验证结论；
- **中**：受限路径错误、接口约定未检查或会导致误用；
- **低/工程**：打包、文档、异常质量或性能估计问题。

## 3. 相对 2026-08-23 报告的状态变化

当前源码在 2026-08-26 至 2026-08-27 发生过更新，且工作区没有可用 Git 元数据，因而本轮以当前源码和数值反例为准。

| 旧报告事项 | 当前状态 | 本轮结论 |
|---|---|---|
| 等价图比较过于粗糙 | 已大幅重写 | 严格 tensor 身份、时间、端口和输出轴检查通过；仍有无条件 $u/d$ 简并前提 |
| GEVP 对 $C$ 强制 `.real` | 已删除 `.real` | 复本征值恢复正确；复本征矢仍写入 `float64`，属于部分修复 |
| `ratio_3pt` 返回统计量强制 `.real` | 已修复 | 纯虚信号可正确保留；1D 异位置轴仍错误 |
| `write_data_ascii` basename 失败 | 已修复主路径 | NumPy 2.0.1 下 basename 写出通过 |
| vendored `savetxt` 为主写出路径 | 已绕开 | 主函数改用 `np.savetxt`，旧 vendored 函数仍保留 |
| VDV 空间 X/Y/Z 映射 | 当前正确 | 独立位移恒等式残差 $\lesssim2.2\times10^{-16}$ |
| Stout 方向、零生成元 | 方向结论撤回；零生成元未修复 | `U[mu]` 后 `3-nu` 正确；README 六维形状与实现冲突，$Q=0$ 仍产生 NaN |
| Bootstrap、covariance、Mom cross sigma、本征矢检查、Omega normal | 未修复 | 均可复现 |
| 多重子/系数算符共轭 | 未修复 | 可复现，并新增发现 `gamma_7` alias 符号回归 |

## 4. 已确认的严重物理缺陷

### C1. Stout 方向映射正确；剩余缺陷是输入 rank 冲突与 $Q=0$ NaN

- 源码：`src/lqcddb/base/smear_gauge.py:5-74`
- 文档：`README.md:773-784`

README 声明输入为

$$
(\mu,z,y,x,c,c),
$$

但源码的

```python
U.transpose(0, 1, 2, 3, 4, 6, 5)
```

要求七维数组。对文档所述合法六维单位规范场，函数立即报错：

```text
ValueError: axes don't match array
```

对代码实际接受的七维布局

$$
(\mu,t,z,y,x,c,c),
$$

`U[mu]` 或 `U[nu]` 取出方向分量后，方向轴已经消失，切片布局变为

$$
(t,z,y,x,c,c).
$$

因此当前 `3-nu` 恰好给出

| gauge 方向 `nu` | 物理方向 | `U[mu]` 切片后的正确轴 | 当前 `3-nu` |
|---:|---|---:|---:|
| 0 | X | 3 | 3 |
| 1 | Y | 2 | 2 |
| 2 | Z | 1 | 1 |

本轮修正后的独立验证没有复用 `roll` 作为唯一参考，而是对每个 $(t,z,y,x)$ 格点显式计算周期移位的 forward/backward staples。结果为：

$$
\max|C_{3-\nu}^{\rm roll}-C^{\rm explicit\ loop}|=0.
$$

用该显式 staple 构造 $Q$，再以 `scipy.linalg.expm` 独立计算 $e^{iQ}U$，与当前实现的一步结果最大差为

$$
3.38\times10^{-16}.
$$

泛型随机场还满足

$$
\max|UU^\dagger-I|=1.78\times10^{-15},\qquad
\max|\det U-1|=1.34\times10^{-15}.
$$

所以此前“Z staple 混入时间方向”“方向错一位”以及由 `4-nu` 得到的 $4.4520$、$2.89\times10^{-1}$ 反例均无效，现全部撤回。`4-nu` 在 `U[mu]` 切片后会把 `nu=0` 映射到颜色轴，本身就不是物理参考。

此外，单位规范场和 `rho=0` 都是必须保持不变的合法输入。对形状 `(4,2,2,2,2,3,3)` 的单位场，一步 smearing 在 `rho=0` 和 `rho=0.12` 时均产生 432 个 NaN。原因是 $c_0=c_1=0$ 后仍计算 $c_0/c_{0,\max}$ 与 $1/(9u^2-w^2)$。

因此当前 C1 的有效结论应严格限定为：

1. README 给出的六维输入形状不被实现接受；
2. $Q=0$ 的解析极限没有实现，identity 与 `rho=0` 固定点失败；
3. 泛型非零 $Q$ 下的方向、staple、矩阵指数与 SU(3) 保持通过了本轮独立检查。

在修复前仍不建议直接处理生产 ensemble，但理由是 $Q=0$/近零数值稳健性与接口约定，而不是方向映射。

### C2. `ignore_dis=True` 会删除图论上连通的多强子交换图

- 默认值：`src/lqcddb/contraction/dynamic.py:672-677,1453-1460`
- 判据：`dynamic.py:925-935`

当前判据为：任意 perambulator 满足

$$
t_q=t_{\bar q}
$$

就把整张图标记为 disconnected。该条件只说明存在等时 quark line，不说明整个 quark-line graph 与 source/sink 其它部分断开。

本轮使用以下味道守恒的 $N\pi$ 两点算符构造 12 张 Wick 图：

```python
sink = [
    '|','d','u','gamma_7','d','|',
    '|','u^d','gamma_5','u','|',
]
source = [
    '|','u^d','gamma_5','u','|',
    '|','u^d','gamma_7','d^d','d^d','|',
]
```

把每个强子区间作为顶点、每条 perambulator 作为边进行独立连通分量分析：

- 时间相等启发式标记 8 张图；
- 其中至少 2 张图的四个强子顶点属于同一连通分量，且该分量同时包含 `tsink` 与 `tsrc`；
- 这两张图分别含 sink 端和 source 端的等时 $u$ 交换线，但同时由跨时传播线连接成完整的 $N\pi$ 交换拓扑。

也就是说，默认 `ignore_dis=True` 会静默删除真实的 connected exchange diagrams。新的 $N-J-N\pi$ 回归测试显式使用了 `ignore_dis=False`，因此没有覆盖默认值风险。

在图论判定实现之前：

- 单强子局域 loop 也应人工确认；
- 所有多强子计算必须使用 `ignore_dis=False`；
- disconnected 分类必须检查连通分量是否跨越 source/sink，而不能只看一条边的时间标签。

### C3. 数值相同的 `gamma_7` 与 `C * gamma_5` 得到相反的重子共轭系数

- 矩阵表：`src/lqcddb/contraction/baroperator.py:54-74`
- diquark 符号表：`baroperator.py:97-108`
- 重子共轭：`baroperator.py:209-221`

当前基中独立验证得到

$$
\max|\gamma_7-C\gamma_5|=0.
$$

因此把同一个重子算符中的矩阵标签从 `gamma_7` 改为 `C * gamma_5`，不能改变任何物理结果。但当前输出为：

```text
conjugate_operator(['|','u','u','gamma_7','d','|'])
    -> coefficient +1

conjugate_operator(['|','u','u','C * gamma_5','d','|'])
    -> coefficient -1
```

相应 proton 两点 Wick 系数从

```text
[+1,-1]
```

整体翻为

```text
[-1,+1].
```

原因是 `gamma_7` 虽作为 gamma 矩阵加入 `_MAT`，却没有加入需要 diquark transpose phase 的语义集合。这是 2026-08-27 源码中的新回归，且 README 明确声明 `gamma_7=C gamma_5`，所以属于无歧义的符号错误。

### C4. `conjugate_operator` 丢弃算符系数，也不正确处理多重子顺序

- `split_hadrons`：`baroperator.py:158-173`
- `conjugate_operator`：`baroperator.py:257-266`

对线性组合项

$$
\mathcal O=c\,\mathcal O_0,
$$

正确 source 应含 $c^*$。当前函数丢弃所有 `|` 之外的前缀数值，实测 $c=2$ 与 $c=1+2i$ 返回完全相同的 `-1.0` 系数。Lambda、同位旋 Clebsch-Gordan 组合、GEVP 非对角元和跃迁矩阵元都会受到影响。

对多强子算符应满足

$$
(H_1H_2)^\dagger=H_2^\dagger H_1^\dagger.
$$

当前代码逐个共轭但保持原强子顺序。两个重子均为 Grassmann 奇算符，若强行恢复原顺序还需要

$$
(-1)^{3\times3}=-1.
$$

当前双重子例子返回两个单重子系数的乘积 $+1$，没有反转顺序，也没有上述额外负号。

### C5. 参数化 gamma 的 source 共轭使用单一代表分量，改变分量相对符号

- 代表矩阵：`baroperator.py:66-73`
- 参数化集合：`baroperator.py:85-95`
- gamma 链共轭：`baroperator.py:186-195`

Euclidean 双线性 source 需要

$$
\Gamma^H=\gamma_4\Gamma^\dagger\gamma_4.
$$

对 $\Gamma=\gamma_\mu$，独立矩阵检查得到

$$
\eta_H(\gamma_1,\gamma_2,\gamma_3,\gamma_4)=(-1,-1,-1,+1).
$$

`gamma_5 * gamma_mu` 也具有相同的分量模式；`sigma_mu_nu` 的空间—空间分量为 $-1$，时间—空间分量为 $+1$。当前代码只用时间代表元构造一个全局符号，所以

```python
conjugate_operator(['|','u^d','gamma_mu','d','|'])
```

固定返回 `+1`，对三个空间分量均错误。参数化 `C * gamma_mu`、`C * gamma_5 * gamma_mu` 和 `C * sigma_mu_nu` 同样不能由一个代表分量覆盖。

这会改变矢量/轴矢算符不同 Lorentz 分量的相对相位。正确接口必须逐分量返回共轭符号，或要求用户在生成 source 前把参数化结构展开成显式分量。

### C6. Lambda→proton “weak” 示例实现了 $V+A$，而非标准左手 $V-A$

- `physics.md:32-37`
- `README.md:898-903`
- `src/lqcddb/test/contraction.lambda.proton.weak.mpi.py:132-136`

标准 $s\to u$ charged weak current 为

$$
J_\mu=\bar u\gamma_\mu(1-\gamma_5)s.
$$

当前示例实际构造

$$
\gamma_\mu+\gamma_\mu\gamma_5
=\gamma_\mu(1+\gamma_5),
$$

README 的另一写法

$$
(1-\gamma_5)\gamma_\mu
=\gamma_\mu(1+\gamma_5)
$$

也相同。该示例的 quark line 对应 $\bar u\Gamma s$，并非把流方向取反后自动得到的形式。因此，除非项目明确要研究右手 $V+A$ 流，否则轴矢部分整体符号错误，会直接改变弱形状因子。

### C7. `autowick` 会静默丢弃部分已支持的 gamma token

- token 解析：`src/lqcddb/contraction/autowick.py:330-340`
- gamma 识别：`baroperator.py:137-148`

`baroperator` 把独立字符串 `C` 视为合法 gamma 结构，但 `autowick` 只以子串 `'gamma'` 判断 gamma。以 `C` 构造重子时，当前输出警告后把两个 `C` 全部丢弃，并生成缺少 gamma 张量、带额外自由自旋指标的 einsum。

科学代码不应在无法识别算符 token 时继续计算。所有未知 token 都应抛出异常；支持集合也必须由同一份语义注册表驱动。

## 5. 已确认的高严重度数值与分析缺陷

### H1. GEVP 复本征值已修复，但返回的复本征矢被截断为实数

- `src/lqcddb/analyse/analyse.py:817-898`
- 关键行：`analyse.py:863,866-867,895-896`

当前代码已正确保留

$$
C\leftarrow\frac{C+C^\dagger}{2}
$$

的复厄米结构；旧版 `.real` 问题已修复。对合法谱表示

$$
C(t)=Z\,\mathrm{diag}(e^{-E_nt})Z^\dagger
$$

构造的复正定 $2\times2$ 例子中使用

$$
Z=\begin{pmatrix}1&0.4i\\0.3+0.2i&1.1\end{pmatrix},
\qquad e^{-E_n}=(0.8,0.3).
$$

参考与当前本征值在 $t=1$ 均为 $(0.8,0.3)$，误差为零。

但是本征矢数组被初始化为

```python
C_eigenvectors = backend.zeros((N,N,Nt), dtype=float)
```

随后复本征矢赋值触发 `Casting complex values to real discards the imaginary part`。返回向量代回

$$
C(t)v=\lambda C(t_0)v
$$

的相对残差为 $0.263$ 和 $0.623$，不再是本征矢。

此外，逐时间片独立排序不能在近简并或能级交叉处稳定追踪态。需要：

1. 本征矢使用复 dtype；
2. 检查 $C(t_0)$ 的 Hermitian positive-definite 条件及条件数；
3. 用 $C(t_0)$ 内积下的相邻时间本征矢重叠做 state tracking；
4. 对反厄米残差和正则化策略做显式记录。

### H2. Bootstrap 默认估计的是 $M$ 样本均值宽度，不是原 $N_{\rm conf}$ 均值误差

- `analyse.py:402-519`

默认 $M=N_{\rm conf}-5$，但输出没有乘 m-out-of-n 所需的尺度

$$
\sqrt{\frac{M}{N_{\rm conf}}}.
$$

对 `np.linspace(-7,9,20)`、$N_{\rm conf}=20$、$M=15$、30000 个重采样的确定性测试：

```text
当前 error                 = 1.25549
原 20 组态均值目标误差     = 1.11400
比值                       = 1.12701
```

第 0 个样本还被硬编码为原始均值并混入 resamples，`data_mean` 返回随机重采样均值而不是原估计量，函数也没有 seed/RNG 参数。若没有专门的 m-out-of-n 理论需求，默认应使用 $M=N_{\rm conf}$；否则必须按目标统计量缩放并明确中心化与自由度约定。

### H3. 复 covariance 计算成 pseudo-covariance，且非首组态轴广播失败

- Jackknife：`analyse.py:324-399`
- Bootstrap：`analyse.py:479-517`

当前复数 covariance 使用

$$
\sum_k r_{k,i}r_{k,j},
$$

而不是常用的 Hermitian covariance

$$
\sum_k r_{k,i}r_{k,j}^*.
$$

最小复测试中：

$$
\max|C-C^\dagger|=2.5,
$$

带共轭的参考结果残差为零。当前量可以作为 pseudo-covariance 额外返回，但不能标为一般复 covariance。

当 `data.shape=(4,5)`、`Nconf_axes=1`、`cov_axes=0` 时，代码直接执行 `(4,5)-(4,)`，得到广播异常。接口明确允许任意组态轴，因此属于实现缺陷。

### H4. `ratio_3pt` 的 1D 异位置时间轴仍发生错误广播

- `analyse.py:636-723`

对文档允许的输入

```text
C3.shape=(5,4,1), tau_axes=1
C2.shape=(5,1,4), t_sink_axes=2
```

预期输出 `(5,4,1)`，实际为 `(5,4,4)`。代码在 1D 分支重索引了两点函数，但没有把 `t_sink_axes` 移到三点函数的 `tau_axes` 位置。

正面变化是：当前 `ratio_mean` 和 `ratio_err` 不再强制使用 `.real`。纯虚合成信号 $C_3=i$、$C_2=1$ 正确返回均值 $i$；同轴、不同初末能量的合成谱比值也以 $4.4\times10^{-16}$ 精度恢复常数矩阵元。

仍需注意 `nan_to_num(...,nan=0)` 加 `maximum(...,0)` 会静默截断负值，并把非零除零产生的无穷替换为巨大有限数。应返回 mask/NaN 或按明确的实/复投影约定处理，不能把无效点当作零物理信号。

### H5. `[Z,Y,X]` 存储下 `Mom_cross_sigma` 的手性相反

- `src/lqcddb/constant/sigma_matrix.py:97-133`

$(Z,Y,X)$ 相对右手 $(X,Y,Z)$ 是奇置换，Levi-Civita 张量需附加负号。取正 X 动量

$$
\mathbf p=[0,0,1]_{ZYX},
$$

标准结果为

$$
(\mathbf p\times\boldsymbol\sigma)_{ZYX}
=(\sigma_y,-\sigma_z,0).
$$

当前函数与上述结果的最大差为 2，而与其整体负值的差为 0。若该对象进入 projector 或流定义，会翻转相应结构的整体符号。

### H6. `vector_creator.check()` 可接受完全平行的复本征矢

- `src/lqcddb/eigvectors/vector.py:37-96`

代码在检查前对 Gram 矩阵取 `.real`。对

$$
v_1=(1,0),\qquad v_2=(i,0),
$$

有 $|v_1^\dagger v_2|=1$，直接 Gram 残差为 1，但函数返回 `True`。生产读取后必须独立计算

$$
\max|V^\dagger V-I|.
$$

`vertex_creator.check()` 当前保留复数 Gram 矩阵，不存在同一问题。

### H7. 严格等价图仍无条件假设 $u,d$ perambulator 逐组态完全相同

- `src/lqcddb/contraction/autowick.py:481-503`
- `identify_equivalent_diagrams`：`autowick.py:848-939`

新实现对 gamma 身份、实际转置符号、V 身份、时间方向、输出轴和 VVV 置换相位的处理明显比旧版严格，现有结构/生产回归测试均通过。

但 `_normalise_peram_flavours` 无条件把 $u,d$ 映射为 `light`，而 `PeramRegistry` 又允许分别注册精确的 `u` 和 `d` 数组。使用形状 `(4,4,2,2)`、满足各自 $\gamma_5$ 厄米关系但数值不同的 $u,d$ perambulator，两个被归为同一类的带电 pion 图得到

```text
C_plus  = -4.35217 - 2.32193 i
C_minus = -4.35217 + 2.32193 i
```

并不逐组态相等。分组后用两倍代表图会保留错误虚部，而未分组和会相消。

因此 `use_equivalence=True` 只可在注册表实际共享同一个 `light` 数组、并明确采用精确 isospin limit 时使用；更稳妥的生产默认仍是 `False`。

### H8. “守恒流/时间方向”顶点忽略 `phase_exp`

- `src/lqcddb/eigvectors/vertex.py:503-521`

同一组本征矢和单位时间链接下，把 `phase_exp` 从零动量改为单位空间动量，返回结果最大差仍为 0。该分支完全没有使用 `phase_exp`，输出也没有动量轴。

对于非零动量转移的 point-split conserved current，通常还需明确前/后向项相对符号、$(1\pm\gamma_\mu)$、midpoint phase 和归一化。当前函数只能视为两个时间链接几何顶点，而不是完整的任意动量守恒流。

## 6. 中等严重度与科学限制

### M1. `create_omega_accelerate(normal=True)` 不同时保持对称性和行归一化

- `src/lqcddb/eigvectors/vertex.py:320-328`

代码先逐行归一化，再用下三角覆盖上三角。后一步重新改变行和。合法小参数测试得到对称的 $6\times6$ 矩阵，但行和为

```text
[1.5,1.5,2.5,2.5,6,6]
```

而不是全部为 6。默认 `normal=False` 不触发该缺陷；整个 stochastic/Omega 权重的无偏性仍需要独立 Monte Carlo 验证，当前只验证了有限性和若干对称性。

### M2. `compress_matrix_V4` 的一般 $Z_N$ 噪声定义不正确

- `src/lqcddb/eigvectors/vector.py:357-445`

- 字面量 `'Z_N'` 无法转换为整数；
- `Z_3` 使用 `[1,-1,i]`，不是三次单位根；
- 高阶硬编码相位含模长不为 1 的 $\sqrt2(1\pm i)$；
- Python `random` 与 backend RNG 混用且没有 seed 参数。

只有逐一核对过的特定噪声集合才能使用，不能把该路径称为一般 $Z_N$ ensemble。

### M3. 有效质量把无效端点填零，且不返回自然时间坐标

- `src/lqcddb/analyse/analyse.py:522-574`

log estimator 自然位于 $t+1/2$，当前 cosh estimator 位于 $t+1$，但都存入索引 $t$。最后一个或两个无法计算的时间点保持为零，容易被误纳入 plateau fit。应返回有效切片、坐标和 mask/NaN，而不是物理零。

强制 `float64` 不是物理要求，只是当前接口限制。丢弃复部前必须先以对称性投影并验证虚部与零统计相容。

### M4. `Mom2GeV` 固定采用连续色散关系

- `analyse.py:6-84`

函数使用

$$
E=\sqrt{m^2+\mathbf p^2}.
$$

这可作为单位换算或连续色散近似，但不是有限格距作用量的普适色散关系。`M0` 为质量列表时，每个粒子还共用同一个 $|\mathbf p|^2$，不能表达一般多粒子动量分配。

### M5. 电磁 wildcard 不包含味电荷

- `physics.md:34-37`

物理电磁流为

$$
J_\mu^{\rm EM}=\sum_f Q_f\bar q_f\gamma_\mu q_f,
$$

而 wildcard 只枚举 flavor，不自动加入 $Q_f$。若用户直接等权求和，会得到 flavor-singlet vector current 而不是电磁流。文档必须明确“逐味生成后由用户按电荷加权”。此外 wildcard 返回 `list[dict]`，当前 dynamic 高层按单个 dict 使用，不能直接接收。

### M6. momentum shell 生成可能含重复项

- `src/lqcddb/base/base_functions.py:39-99`

例如参考动量 `[-1,0,1]`、固定 $Q^2$ 时返回 48 项，但只有 12 个唯一动量。动量壳平均前必须去重并核对 cubic orbit 简并度。

## 7. 工程、MPI、性能与文档问题

### 7.1 MPI 与可选依赖

- `src/lqcddb/contraction/dynamic.py:18` 在模块加载时请求 MPI 函数；
- `pyproject.toml` 把 `mpi4py` 声明为 optional，但 dynamic API 实际需要 MPI；
- `from lqcddb import *` 会解析全部 `__all__`，从而触发 MPI 初始化；
- 沙箱内 MPI 初始化因无网络接口失败；同一 $N-J-N\pi$ 回归测试在沙箱外通过；
- 多 rank 数值路径本轮未执行。

`get_mpi_data` 在 `size==1` 时对所有模式直接返回原形状副本，而多 rank `Gather/Allgather` 会增加 rank 轴，串行和并行形状不一致。`TGather` 仍要求各 rank 本地长度完全相同。`Transport` 和非均匀多 rank 分支尚未独立验证。

### 7.2 `cached_contract` 的 ellipsis 校验不是 NumPy 右对齐广播

- `src/lqcddb/base/base_functions.py:244-345`

NumPy 合法表达式

```python
'...i,...i->...'
shape (2,3) and (4,2,3) -> output (4,2)
```

被当前预校验拒绝，因为 ellipsis 批维从左而不是从右对齐。包内主要显式 einsum 不受影响，但通用 API 与 NumPy 语义不一致。

### 7.3 性能 adviser 只能视为启发式

- `src/lqcddb/contraction/contractadviser.py:89-116` 的多个 `compute_efficiency` 为 2 或 2.5，超过物理意义上的峰值比例 1；
- ellipsis 的 batch/output accounting 仍需独立核查；
- 所报时间和带宽不应作为硬件实测结论，也不能作为收缩物理正确性的证据。

### 7.4 I/O

已确认修复：

- `write_data_ascii` 现在只在父目录非空时创建目录；
- NumPy 2.0.1 下 basename 写出成功，首行和数据形状正确；
- 主路径使用 `np.savetxt`，不再依赖旧 `numpy.compat`。

仍存在：

- `readin_eigvecs` 打开文件后不关闭；
- 二进制读取没有显式 endian/metadata/文件大小完整性约定；
- `safe_save` 的 `/nexdata/...` 自动回退是站点特定策略。

### 7.5 文档、类型存根与打包

当前至少有以下不一致：

1. `README.md:107` 和 `wick_contraction` docstring 声称缺省 P/V/G 前缀自动生成大写字母，源码实际补空字符串；
2. README/类型存根仍把 Jackknife 公式写成错误的整体负号，源码 delete-one 实现是正的；
3. README 的 `solve_gevp` 描述仍称取实部，但源码已不再取 `.real`；
4. README 的 Stout 六维形状与源码七维 transpose 冲突；其 SU(3) 声明只覆盖泛型非零 $Q$，没有说明 identity/`rho=0` 固定点失败；
5. README 把 `meff` 的 float64 限制称为物理要求；
6. `conjugate_operator` 类型存根仍是 `str -> str`，实际是 token list 到 token list；
7. `pyproject.toml` 和 `__version__` 为 0.0.2，`changelog.md` 已声明 0.0.3；
8. package-data 声明 `lqcddb/README.md` 与 `lqcddb/physics.md`，实际文件仍在仓库根目录；
9. 多个旧测试脚本仍调用已不存在或已改签名的方法；
10. 未知算符 token 只打印 warning 后丢弃，不符合科学计算 fail-closed 原则。

## 8. 通过的独立物理与数值检查

### 8.1 Gamma、charge conjugation 与 Pauli 代数

独立残差均为 0：

$$
\max_{\mu\nu}|\{\gamma_\mu,\gamma_\nu\}-2\delta_{\mu\nu}|=0,
$$

$$
\max_\mu|\gamma_\mu^\dagger-\gamma_\mu|=0,
$$

$$
|\gamma_5-\gamma_1\gamma_2\gamma_3\gamma_4|=0,
$$

$$
|C^TC^{-1}+I|=0,\qquad
\max_\mu|C\gamma_\mu C^{-1}+\gamma_\mu^T|=0.
$$

Pauli 代数

$$
\sigma_i\sigma_j=\delta_{ij}I+i\epsilon_{ijk}\sigma_k
$$

也以零残差通过；问题仅在 `[Z,Y,X]` 坐标手性映射，而非 Pauli 矩阵本身。

### 8.2 基础 Wick 与 perambulator

- pion 两点总 Wick sign：`[+1]`；
- 使用 `C * gamma_5` 的 proton 两点 signs：`[-1,+1]`；
- `seq_peram` 与显式合并 spin-distillation 矩阵的

  $$
  (\gamma_5\otimes I)\tau^\dagger(\gamma_5\otimes I)
  $$

  最大差为 0；
- `seq_peram(seq_peram(tau))=tau` 最大差为 0。

### 8.3 VDV/VVV 与动量相位

对正交随机本征矢和单位 gauge link：

- X/Y/Z 的 $\Delta x=0,\pm1$ 与显式 roll 公式最大差 $2.2\times10^{-16}$；
- $V(-\Delta x)=V(+\Delta x)^\dagger$ 残差小于 $1.0\times10^{-16}$；
- `link_dir='all'` 等于 X/Y/Z 平均，残差 $2.8\times10^{-17}$；
- $V(-p)=V(p)^\dagger$ 残差 $2.0\times10^{-16}$；
- VVV 六种末三腿置换满足全反对称性，最大残差 $1.1\times10^{-17}$；
- 零动量 VDV 等于本征矢 Gram 矩阵，残差为 0。

### 8.4 Stout 泛型非零 $Q$ 子域

在实际七维布局 `(mu,t,z,y,x,c,c)` 下：

- `U[mu]` 后的 X/Y/Z 轴为 3/2/1，当前 `3-nu` 映射正确；
- vectorized roll staple 与逐格点显式周期坐标循环残差为 0；
- 当前一步结果与显式 staple 加 `scipy.linalg.expm` 的最大差为 $3.38\times10^{-16}$；
- 随机 SU(3) 场的一步输出满足幺正性和单位行列式到 $1.8\times10^{-15}$；
- 这些通过项不覆盖 $Q=0$ 奇点。

### 8.5 严格等价图

- `test_equivalent_diagrams_strict.py`：13 项通过；
- $N-J-N\pi$ 生产形状回归：2 项通过；
- 测试覆盖四个 current gamma 分量、gamma 转置相位、VVV 反对称性、分组/未分组总关联函数一致性；
- 原始图数 `(120,120,12)` 在前两个 sector 分别归并到 `(88,88,12)`；
- 数值一致性容差为 $2\times10^{-12}$。

这些结果证明新算法在测试所施加的 tensor identities 下有效，不证明任意 flavor/smearing/vertex 注册都可归并。

## 9. 测试、环境与未覆盖路径

环境：

```text
Python       3.12.11
NumPy        2.0.1
SciPy        1.16.0
opt_einsum   3.4.0
Matplotlib   3.11.1
mpi4py       4.1.2
CuPy         14.1.1（未运行 GPU 数值测试）
pytest       未安装
```

已执行：

- 38 个 Python 文件 AST 解析：0 个错误；
- `test.cache.py`：15 项通过；
- `test_equivalent_diagrams_strict.py`：13 项通过；
- `test_contraction_njnp_3pt_gevp.py`：沙箱外 2 项通过；
- 多组独立小矩阵、显式坐标 Stout staple、SU(3)、张量网络和统计反例。

未覆盖：

1. 真实 ensemble 最终 correlator 与已发表/独立代码结果的逐组态比较；
2. 多 rank MPI Gather/TGather/TScatter/Transport；
3. CuPy/GPU 后端的 GEVP、ArraySlicer、vertex 和 contraction；
4. `dis_connect` 的完整物理定义和真实 disconnected subtraction；
5. stochastic distillation/Omega 权重的无偏性和方差；
6. 非立方格子、各向异性格距和非标准边界条件；
7. conserved current 的完整 Ward identity；
8. 真实 gauge ensemble 上 Stout 与独立实现的逐链接比较。

## 10. 建议修复顺序

### 优先级 A：立即阻断错误物理结果

1. 保留 Stout 当前 `3-nu` 方向映射，统一 README/实现的输入 rank，实现 $Q\to0$ 解析极限，并固定加入 identity、`rho=0`、显式坐标 staple、随机 SU(3) 和独立 `expm` 回归；
2. 把 disconnected 判定改为 quark-line graph connected components，默认暂设 `ignore_dis=False`；
3. 统一 gamma token 语义，修复 `gamma_7=C gamma_5` alias、参数化分量共轭和未知 token fail-closed；
4. `conjugate_operator` 保留 $c^*$、反转强子次序并按 Grassmann parity 处理多重子符号；
5. 将 Lambda→proton current 改为 $\gamma_\mu(1-\gamma_5)$，或明确重命名为右手 $V+A$；
6. GEVP 本征矢改为 complex dtype，并加入残差、HPD 和 state-tracking 测试。

### 优先级 B：统计与分析可靠性

1. Bootstrap 默认 $M=N_{\rm conf}$，或正确实现/标注 m-out-of-n 缩放；
2. 修复任意 `Nconf_axes` covariance，并明确返回 Hermitian covariance 与 pseudo-covariance；
3. 修复 `ratio_3pt` 1D 轴移动，移除无物理依据的零截断并返回 mask；
4. 修正 `[Z,Y,X]` 下的叉积取向；
5. 本征矢验证保留完整复 Gram 矩阵；
6. 有效质量返回有效时间坐标和 mask。

### 优先级 C：受限优化、MPI 与文档

1. 等价图显式接收 flavor-equivalence policy，只在实际共享 `light` tensor 时合并 $u,d$；
2. conserved vertex 支持/拒绝非零动量，并写清完整 current normalization；
3. 修复 Omega `normal=True` 和一般 $Z_N$；
4. 把 dynamic 的 MPI 依赖延迟到真正 MPI 路径；
5. 统一单 rank/多 rank 返回形状并增加多 rank deterministic tests；
6. 同步 README、physics.md、类型存根、版本和 package data。

## 11. 当前生产使用建议

在修复前，建议采用以下最低安全策略：

```text
stout_smear_ndarray        方向正确；修复 Q=0 与输入 rank 前暂不用于生产
ignore_dis                 多强子一律 False
use_equivalence            默认 False；仅经逐组态 tensor identity 验证后开启
conjugate_operator         只用于已人工核对的单介子/单重子显式结构
gamma_7 source             不自动生成；先人工冻结符号与字段顺序
Bootstrap                  使用 M=Nconf，并自行控制 RNG；仍不要用复 covariance
complex covariance         独立构造 Hermitian covariance 或堆叠 Re/Im
solve_gevp eigenvectors    当前返回值禁用；本征值可在复谱例验证后使用
ratio_3pt                  只用已验证的同轴或二维路径
eigenvector input          总是独立检查 max|V^dagger V-I|
weak current               显式使用 gamma_mu @ (I-gamma5)
```

## 12. 文件修改声明

- 新增并于 2026-08-28 修订：`review_report_2026-08-27.md`（撤回 C1 方向错误结论）
- 未修改：`review_report.md`
- Codex 未修改：`src/lqcddb/`、`src/lqcddb/test/`、README、physics.md、类型存根和包源码；初审后存在外部源码更新，见报告顶部版本范围说明
- 临时审查脚本仅写入 `/tmp`，未进入项目源码树
