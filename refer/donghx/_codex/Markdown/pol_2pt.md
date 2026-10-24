# 2pt 相关函数的宇称降维与 Pauli 极化投影

本文记录从保留完整旋量指标的四维 2pt 结果，先进行宇称投影的 `4 -> 2` 降维，再进行 x、y、z 三个方向极化投影的计算流程。

## 1. 输入数据与指标约定

2pt 计算保存完整的旋量结构

$$
M_{yxil},
$$

其中 `y` 是 sink 时间，`x` 是 source 时间，`i` 和 `l` 是两个旋量指标。典型文件为

```text
twopt_slice_pp_..._contract_confXXXX.npy
```

数组形状为

```text
(Nt, Nt, 4, 4)
```

当前生产代码的投影是逐元素旋量收缩：

```python
np.einsum("il,yxil->yx", projection, M)
```

因此它表示

$$
C_{yx}=\sum_{i,l}P_{il}M_{yxil},
$$

而不是直接把 `M[y,x]` 当作普通矩阵进行 `P @ M[y,x]`。

## 2. 四维宇称投影

在当前 Euclidean DR gamma 约定下，正、负宇称投影子为

$$
P_+ = \frac{1+\gamma_4}{2},
\qquad
P_- = \frac{1-\gamma_4}{2}.
$$

它们满足

$$
P_\pm^2=P_\pm,
\qquad
P_\pm^\dagger=P_\pm,
\qquad
\operatorname{rank}(P_\pm)=2.
$$

因此正宇称只包含两个独立的旋量自由度。

## 3. 构造正宇称的 4×2 基

不能在 DR 表示中简单取 `P_plus[:, 0:2]` 作为 4×2 基，因为 `P_plus` 的像空间通常不是前两个标准基方向。

应对 `P_plus` 做厄米特本征分解：

```python
evals, evecs = np.linalg.eigh(P_plus)
U = evecs[:, evals > 0.5]
```

得到

$$
U\in\mathbb C^{4\times2},
\qquad
U^\dagger U=I_2,
\qquad
UU^\dagger=P_+.
$$

`U` 的列向量构成正宇称子空间的正交归一基。二维坐标对应的旋量指标记为 `a,b=0,1`。

## 4. 四维旋量张量降为二维旋量张量

由于生产结果的旋量收缩是逐元素形式，降维时应保持相同的指标结构：

$$
M^{(2)}_{yxab}
=
U_{ia}\,M_{yxil}\,U^*_{lb}.
$$

对应代码为：

```python
M2 = np.einsum("ia,yxij,jb->yxab", U, M, U.conj())
```

这里使用的是 `U` 和 `U.conj()`，而不是普通矩阵乘法中的 `U.conj().T`。原因是原始投影为 `P_il M_il` 的逐元素收缩。

如果采用普通矩阵乘法约定，即把某个旋量指标定义为矩阵行、另一个定义为矩阵列，则应使用相应的 `U.conj().T @ M @ U`。两种写法的选择取决于原始数据的指标定义，不能混用。

## 5. 三个方向的四维自旋算符

三个空间方向的自旋算符定义为

$$
S_k=i\gamma_k\gamma_5,
\qquad k=1,2,3.
$$

在当前 gamma 约定下，`pol15`、`pol25`、`pol35` 分别对应 `k=1,2,3`。

这些算符与正宇称投影子对易：

$$
[P_+,S_k]=0.
$$

因此它们可以限制在二维正宇称子空间内。

## 6. 二维 Pauli 矩阵

在正宇称子空间中定义

$$
\sigma_k=U^\dagger S_k U
=U^\dagger(i\gamma_k\gamma_5)U.
$$

代码为：

```python
S4 = 1j * gamma(k) @ gamma(5)
sigma_k = U.conj().T @ S4 @ U
```

数值上应检查

$$
\sigma_k^\dagger=\sigma_k,
\qquad
\sigma_k^2=I_2.
$$

由于二维基底的排序和相位不是唯一的，计算结果可能得到 `+sigma_i` 或 `-sigma_i` 的等价表示；只要正反向投影和最终差值使用同一基底，物理结果不变。

## 7. 正向和反向极化投影

二维正向、反向极化投影子为

$$
\Pi_k^{+}=\frac{I_2+\sigma_k}{2},
\qquad
\Pi_k^{-}=\frac{I_2-\sigma_k}{2}.
$$

它们满足

$$
\Pi_k^++\Pi_k^-=I_2,
\qquad
\Pi_k^+-\Pi_k^-=\sigma_k.
$$

分别计算

$$
C_{yx}^{k,+}
=\operatorname{Tr}\left[(\Pi_k^+)^T M^{(2)}_{yx}\right],
$$

$$
C_{yx}^{k,-}
=\operatorname{Tr}\left[(\Pi_k^-)^T M^{(2)}_{yx}\right].
$$

极化相关函数定义为正向减反向：

$$
C_{yx}^{k,\mathrm{pol}}
=C_{yx}^{k,+}-C_{yx}^{k,-}
=\operatorname{Tr}\left[\sigma_k^T M^{(2)}_{yx}\right].
$$

在保持原始逐元素收缩约定时，代码写成：

```python
forward = np.einsum(
    "ab,yxba->yx", 0.5 * (np.eye(2) + sigma_k.T), M2
)
backward = np.einsum(
    "ab,yxba->yx", 0.5 * (np.eye(2) - sigma_k.T), M2
)
polarized = forward - backward
```

这里出现 `sigma_k.T` 是因为原始四维投影使用的是系数矩阵的逐元素收缩。若数据已经明确转换成普通矩阵乘法约定，则应采用相应的无转置形式。

## 8. 与原四维投影的等价性

四维直接投影为

$$
C_{yx}^{k,(4)}
=\sum_{i,l}(P_+S_k)_{il}M_{yxil}.
$$

利用

$$
P_+=UU^\dagger,
\qquad
S_k=U\sigma_kU^\dagger
\quad\text{(在正宇称子空间内)},
$$

并保持逐元素收缩的转置约定，可得

$$
C_{yx}^{k,(4)}
=\operatorname{Tr}\left[\sigma_k^T M^{(2)}_{yx}\right].
$$

因此二维正反向极化差与四维直接投影完全等价。

验证时应同时检查：

```python
direct = np.einsum("il,yxil->yx", P_plus @ S4, M)
reduced = forward - backward
error = np.max(np.abs(direct - reduced))
```

在 `L24x72` 的 `conf=4050` 实际数据上，x、y、z 三个方向以及多个动量的最大误差约为

```text
1.1e-18
```

达到 double 精度下的数值一致。

## 9. 时间边界符号

`L24x72_diffpol.py` 和当前 2pt 结果的后处理还包含反向时间的符号处理。对 sink 时间小于 source 时间的条目使用

```python
pol[t_sink < t_source] *= -1
```

因此比较四维与二维结果时，必须在两条路径中使用完全相同的时间符号规则，否则会得到人为的差异。

## 10. 推荐的完整流程

```python
P_plus = 0.5 * (gamma(0) + gamma(4))

# 正宇称二维基
evals, evecs = np.linalg.eigh(P_plus)
U = evecs[:, evals > 0.5]

# 四维旋量算符
S4 = 1j * gamma(k) @ gamma(5)

# 二维 Pauli 算符
sigma = U.conj().T @ S4 @ U

# 保持原逐元素收缩约定的二维张量
M2 = np.einsum("ia,yxij,jb->yxab", U, M4, U.conj())

# 正向和反向极化
P_forward = 0.5 * (np.eye(2) + sigma.T)
P_backward = 0.5 * (np.eye(2) - sigma.T)

forward = np.einsum("ab,yxba->yx", P_forward, M2)
backward = np.einsum("ab,yxba->yx", P_backward, M2)
polarized = forward - backward

# 与四维直接结果比较
direct = np.einsum("il,yxil->yx", P_plus @ S4, M4)
assert np.max(np.abs(direct - polarized)) < 1e-12
```

该流程可分别令 `k=1,2,3`，得到 x、y、z 三个方向的极化结果，同时保留正向和反向极化分量用于独立检查。

## 11. 外部旋量降维与三个夸克同时投影的区别

以下内容更新于 2026-10-21，记录 L24x72、ydir、$P_y=2,4$ 的测试。这里 $P_y$ 是以 $2\pi/L$ 为单位的整数动量标签，$P_+$ 是旋量投影子，两者不要混淆。

原来的内插算符为

$$
N_4=\epsilon_{abc}(u_a^T A_4 d_b)u_c,
\qquad A_4=C\gamma_5\gamma_4.
$$

前述外部降维相当于在源、汇的重子外部旋量空间中保留 $P_+$ 子空间。将 $U$ 放入 direct、exchange 收缩内部，是实现同一投影的计算重排；双夸克内部旋量指标仍为四维，不改变原算符所定义的投影关联函数。

新测试的算符则为

$$
N_{+++}=\epsilon_{abc}
\big[(P_+u_a)^T A_4(P_+d_b)\big](P_+u_c).
$$

这里在源、汇两端的三个夸克场上均施加投影，改变了内插算符。虽然两种方案都保存形状为 `(Nt, Nt, 2, 2)` 的结果，但不能要求这两种关联函数逐点相等。

## 12. 两种算符是否都属于质子通道

在当前 Euclidean DR 约定下，生产代码使用固定实基

$$
\gamma_4=\begin{pmatrix}0&I_2\\I_2&0\end{pmatrix},
\qquad
U=\frac1{\sqrt2}\begin{pmatrix}I_2\\I_2\end{pmatrix},
\qquad UU^\dagger=P_+.
$$

令 $\chi_f=U^\dagger q_f$，则 $P_+q_f=U\chi_f$，并且实际代码给出

$$
A_2=U^T A_4 U
=\begin{pmatrix}0&1\\-1&0\end{pmatrix}
=i\sigma_2.
$$

因此新算符的二维表示为

$$
U^\dagger N_{+++}
=\epsilon_{abc}(\chi_{u,a}^T i\sigma_2\chi_{d,b})\chi_{u,c}.
$$

这个双夸克矩阵非零，且满足 $R^T(i\sigma_2)R=i\sigma_2$，其中 $R\in SU(2)$。双夸克具有空间旋转标量的自旋单态结构，与第三个二分量夸克组合成自旋 $1/2$。$P_+$ 与空间旋转对易，不会将算符变成某个固定方向的自旋极化算符。

两种算符都通过颜色 $\epsilon_{abc}$ 构造规范不变的颜色单态，具有重子数 $B=1$、电荷 $Q=+1$ 和味内容 $uud$。在轻夸克简并、各夸克采用相同空间涂抹的条件下，两者保持核子的 $I=1/2$ 结构：双夸克矩阵反对称，若将 $d$ 替换为 $u$，与反对称颜色张量结合的双夸克项为零，因此它们不是 $I=3/2$ 的最高权后裔。

当前矩阵还满足

$$
\gamma_4^T A_4\gamma_4=A_4.
$$

因此 $N_4$ 在宇称下按正内禀宇称的重子旋量变换，而 $N_{+++}$ 保留其正宇称旋量分量。在零动量处，两者均可用于核子 $J^P=1/2^+$ 通道。它们是质子的内插算符，不是只产生质子基态、排除所有激发态的精确态投影子。

在非零动量处，宇称将 $\boldsymbol p$ 变为 $-\boldsymbol p$，所以单一动量下的 $P_+$ 并不严格隔离所有正宇称态。对夸克使用 $P_+$ 也不是将其投影为物理上独立的正能量夸克，或从传播子中精确删除所有反夸克贡献。

## 13. 新算符与旧核子算符的线性关系

在本文使用的矩阵约定下，定义

$$
A_5=C\gamma_5,\qquad
N_5=\epsilon_{abc}(u_a^T A_5d_b)u_c.
$$

利用 $A_4\gamma_4=A_5$ 和 $\gamma_4^T A_4\gamma_4=A_4$，得到

$$
P_+^T A_4P_+=A_4P_+
=\frac12(A_4+A_5),
$$

从而

$$
\boxed{N_{+++}=\frac12P_+(N_4+N_5)}.
$$

因此新算符是两个核子内插算符的特定线性组合。其自关联函数包括两个对角关联函数和两个交叉关联函数，并非两个已有自关联函数的简单平均。

也可定义 $u_\pm=P_\pm u$、$d_\pm=P_\pm d$。由于

$$
P_+^T A_4P_-=P_-^T A_4P_+=0,
$$

原算符的外部正宇称分量可写为

$$
P_+N_4
=\epsilon_{abc}
\big[(u_{+,a})^T A_4d_{+,b}
+(u_{-,a})^T A_4d_{-,b}\big]u_{+,c}.
$$

新算符只保留其中第一个双夸克分量。这说明“只投影外部重子旋量”和“投影三个夸克场”在算符层面并不等价。

两种算符的前向谱展开具有同一理论的能级，但重叠因子不同。示意地，在相同动量和通道下，

$$
C_{AA}(t,\boldsymbol p)
=\sum_n W_n^{(A)}(\boldsymbol p)e^{-E_n(\boldsymbol p)t}
+\text{有限时间反向传播项}.
$$

如果两者都与最低核子态有非零耦合，在基态主导的前向时间窗口应趋向同一能量；有限时间的关联函数及有效质量不必相等。

## 14. 三夸克投影在 contraction 内部的实现与验证

perambulator 的轴序为 `[time, spin_sink, spin_source, ev_sink, ev_source]`。新方案先计算

$$
(\tau_2)_{t r s a b}
=\sum_{i,j}U^*_{ir}\tau_{t i j a b}U_{js},
\qquad \tau_2=U^\dagger\tau U.
$$

随后以 $A_2=U^TA_4U$ 构造双夸克张量，direct 和 exchange 均只使用源、汇各两个旋量分量。由于当前 $U$ 为实矩阵，该 perambulator 变换与外部结果采用的逐元素旋量约定兼容；不能由此省略一般复基情况下的共轭与转置核对。

相关实现为：

- [nplus3_spin2.py](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_spin2.py)：三夸克投影、二维收缩及独立四维参考检验。
- [run_nplus3_l24.py](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/run_nplus3_l24.py)：L24x72 计算入口和分段计时。
- [proton_spin2.py](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/proton_spin2.py)：外部旋量降维的原方案。

使用独立的四维显式 $P_+\tau P_+$ 路径核对二维实现，随机复张量测试中最大绝对误差为约 $2.03\times10^{-13}$。这验证了新算符的二维实现与其自身的四维投影参考一致，不意味着新旧算符的物理关联函数相等。上述投影矩阵恒等式的数值误差约为 $2.22\times10^{-16}$。

## 15. L24x72 两种算符的计时比较

统计范围为同一批 100 个组态：`4050, 4100, ..., 9000`；ydir 的 $P_y=2,4$；每组态 72 个源时间；$t_{\rm sep}=2,\ldots,30$，每组态每动量共 2088 个源汇时间对。新任务数组 `3861836` 使用 `na800-sug` 分区和 `donghx_223` reservation，输入 perambulator 与顶点保持不变。

这里的“原方案”已经使用外部旋量降维并保存 $2\times2$ 结果，并非最早保存 $4\times4$ 的生产代码。以下均为每组态、每动量的平均秒数。

| 计时部分 | $P_y$ | 原方案：仅外部投影 | 新方案：三个夸克均投影 | 原耗时 / 新耗时 |
|---|---:|---:|---:|---:|
| 收缩记录 | 2 | 17.5309 | 5.5321 | 3.17 |
| 收缩记录 | 4 | 17.7114 | 5.0803 | 3.49 |
| 完整流程记录 | 2 | 102.9575 | 64.5253 | 1.60 |
| 完整流程记录 | 4 | 101.4221 | 23.6550 | 4.29 |

原方案的 `fused_contraction_s` 只计 direct/exchange，新方案的 `contraction_s` 还包含二维双夸克张量构造。新方案额外的 `peram_projection_s` 分别为 0.1829 秒和 0.1589 秒；将这部分加入新收缩成本后，耗时比分别为 3.07 和 3.38。上述计时使用 CUDA 同步，不是只记录异步内核提交时间；但各段边界和启动开销并非完全相同。

提速来源是内部旋量求和从四维变为二维，同时缩小了中间张量。单个已投影 perambulator 的旋量元素数由 16 降为 4；实际代码仍先读取完整 perambulator，因此这不意味着读盘数据量或整个任务的显存峰值自动降为原来的四分之一。

完整流程计时不能直接解释为纯 contraction 算法提速。分段记录如下：

| 部分 | 原 $P_y=2$ | 新 $P_y=2$ | 原 $P_y=4$ | 新 $P_y=4$ |
|---|---:|---:|---:|---:|
| 顶点构造及相关读盘 `vvv_s` | 19.6501 | 11.3342 | 18.4096 | 9.7566 |
| perambulator 读入及传输 `peram_io_s` | 65.5644 | 47.2815 | 65.0897 | 8.4646 |

新任务依次计算 $P_y=2$ 和 $P_y=4$，使用同一 momsmear 输入目录；后者可复用前者留下的文件缓存。历史任务与新任务也不是同一次受控基准，顶点部分的实现和同步方式还存在差异。因此，现有记录支持收缩部分约三倍的实测提速，不能将 $P_y=4$ 完整流程的 4.29 倍全部归因于旋量降维，也不能直接外推为其他 ensemble 的固定加速比。

原方案完整流程使用 `vvv_s + loop_s`；新方案使用 `total_s`，后者还包含保存结果等开销，两者边界略有不同。若要精确测量算法收益，应在同一 GPU、相同 block size、相同输入和缓存状态下交替运行两种实现，分别比较纯收缩与包含 I/O 的耗时。

## 16. 有效质量对比与结果索引

本次使用 `nopol` 的实部，与此前比较保持相同投影。先在每个组态内平均全部 72 个源时间，再对组态执行 delete-one Jackknife，不将源时间当作独立组态。已保存的标量结果包含时间反周期边界符号，不重复乘号，也不对重子做前后向折叠。

前向有效能量为

$$
aE_{\rm eff}(t)
=\log\frac{\operatorname{Re}\overline C(t)}
{\operatorname{Re}\overline C(t+a)}.
$$

图中中心值沿用原比较程序，为各 Jackknife 对数比值的平均，误差为

$$
\overline C_{(k)}(t)
=\frac{\sum_{j=1}^{N}C_j(t)-C_k(t)}{N-1},
\qquad
\delta(aE)=\sqrt{\frac{N-1}{N}
\sum_{k=1}^{N}\big[(aE)_{(k)}-\overline{aE}_{\rm JK}\big]^2}.
$$

“先取全样本均值再求对数”的估计量也保存在图表 CSV 中。只有相邻时间点均已计算，且全样本与所有 Jackknife 副本的两个实部均为正时，才给出该算符的有效质量；两个算符分别判断有效性，不将缺失点填零。由 $t_{\rm sep}=2,\ldots,30$ 得到的有效质量范围仅为 $t/a=2,\ldots,29$。

观察到新算符在短时间的有效质量较低，两者在稍晚、仍有信号的时间逐渐接近；$P_y=4$ 在 $t/a\gtrsim11$ 后误差迅速增大。该现象说明重叠权重改变，尚不能单独证明激发态污染更小、两者已经达到同一基态平台，或在相同资源下新算符的最终能量精度更高。后者还取决于统计噪声、自相关和拟合窗口，需要进一步比较。

本次结果与复现入口：

- [100 组态原始比较 CSV](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/comparison_yp_100conf.csv)。其中历史 CSV 对两种算符联合屏蔽无效点，部分噪声尾区只判断比值为正；最新绘图采用上文所述的独立、相邻实部为正的规则。
- [计时汇总及组态清单](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/comparison_yp_100conf_summary.json)。
- [绘图脚本](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/plot_nplus3_effective_mass.py)。
- [早期有效质量对比图 PNG](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/figures/v20261021162855/effective_mass_comparison.png) 与 [PDF](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/figures/v20261021162855/effective_mass_comparison.pdf)。
- [完整时间范围诊断图](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/figures/v20261021162855/effective_mass_comparison_full.png)。
- [绘图数据与有效性标记](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/figures/v20261021162855/effective_mass_points.csv) 及 [分析元数据](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/nplus3_L24_results/figures/v20261021162855/analysis_metadata.json)。

## 17. 恢复完整算符：四项二维求和

更新日期：2026-10-22。为避免混淆，以下用 $O_p=O_{++}$、$O_m=O_{--}$ 表示 diquark 的两个夸克都在正或负宇称子空间；第三个外部 $u$ 在两种算符中始终取正宇称。

$$
O_p=\epsilon_{abc}(u_{+,a}^T A_4d_{+,b})u_{+,c},\qquad
O_m=\epsilon_{abc}(u_{-,a}^T A_4d_{-,b})u_{+,c},\qquad
P_+N_4=O_p+O_m.
$$

在当前固定 DR 基中，定义 $U_\pm=(I_2,\pm I_2)^T/\sqrt2$，$A_4=C\gamma_5\gamma_4$。直接计算得到

$$
U_+^TA_4U_+=\epsilon_2,\qquad
U_-^TA_4U_-=-\epsilon_2,\qquad
U_\pm^TA_4U_\mp=0,\qquad
\epsilon_2=\begin{pmatrix}0&1\\-1&0\end{pmatrix}.
$$

每条线取相应的二维传播块

$$
T_{sr}=U_s^\dagger\tau U_r,\qquad s,r\in\{+,-\}.
$$

这里的 $T_{+-}$ 和 $T_{-+}$ 一般不为零，不能由 diquark gamma 的混合块为零推出 perambulator 的混合块也为零。原外部投影的完整关联函数为

$$
C_{\rm ext}=\langle(O_p+O_m)(\overline O_p+\overline O_m)\rangle
=C_{++}+C_{+-}+C_{-+}+C_{--},\qquad
C_{sr}=\langle O_s\overline O_r\rangle.
$$

这与原外部正宇称投影在算符层面精确等价；仅计算 $C_{++}$ 才是前文的 $N_{+++}$ 自关联函数。四项各自都可使用二维内部旋量，但必须包含两个源汇交叉项。

现有数组任务 `3864216` 的全部 100 个组态已完成，两个动量 $P_y=2,4$ 共 200 个组态—动量组合均通过完整 2x2 张量比较。四项求和相对外部投影的最大张量差除以该组态参考张量最大模，分别不超过 $8.55\times10^{-15}$、$1.11\times10^{-14}$。沿用第 16 节统计定义，有效质量中心值在全部有效时间点的最大绝对差分别为 $3.85\times10^{-15}$、$3.18\times10^{-14}$。结论应表述为双精度舍入误差内一致，不是输出逐位相同。

结果入口：

- [四项求和实现](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_spin2.py)。
- [三种方案有效质量对比图](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_L24_results/analysis/v20261022104438/three_scheme_effective_mass.png)。
- [四项求和减原方案的有效质量差值](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_L24_results/analysis/v20261022104438/four_sector_minus_external.png)。
- [分析元数据与完整输入清单](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_L24_results/analysis/v20261022104438/analysis_metadata.json)。

## 18. 四个贡献之间的关系与五次收缩简化

### 18.1 四个物理贡献不能直接删减

读取全部 100 个组态的原始四项复数 2x2 张量，$\|C_{+-}-C_{-+}\|_2/\|C_{-+}\|_2$ 在 $P_y=2$ 的范围为 0.372--0.954，在 $P_y=4$ 的范围为 0.0728--0.1408。交叉项不是逐组态相等，也不是固定倍数。源汇交换或厄米性必须连同时间方向、动量、smearing、共轭及统计平均条件一起推导，不能直接用于同一条测量。

下表是 $t/a=2$ 时四项对总关联函数实部 nopol 的贡献比例。先对每组态 72 个源时间平均，再在同一批 100 个组态上做配对 delete-one Jackknife；比例在 replica 内形成，括号为一倍标准误差。

| $P_y$ | $C_{++}/C_{\rm ext}$ | $C_{+-}/C_{\rm ext}$ | $C_{-+}/C_{\rm ext}$ | $C_{--}/C_{\rm ext}$ |
|---|---:|---:|---:|---:|
| 2 | 0.74282(135) | 0.05431(51) | 0.05352(49) | 0.14935(74) |
| 4 | 0.49620(76) | 0.15355(18) | 0.15321(17) | 0.19703(56) |

这些是带符号的关联函数贡献，不是概率，在其他时间可以为负或大于一。特别是 $P_y=4$ 的短时间交叉项合计约 31%，不能忽略。

### 18.2 Direct 与 exchange 的精确关系

沿用当前程序的符号约定 $C_{sr}=D_{sr}-E_{sr}$。在下述条件下，

$$
E_{++}=-\frac12D_{++},\qquad
E_{+-}=-\frac12D_{+-},\qquad
E_{-+}=-\frac12D_{-+}.
$$

一般不存在 $E_{--}=-D_{--}/2$。因此完整关联函数可改写为

$$
\boxed{C_{\rm ext}=\frac32\left(D_{++}+D_{+-}+D_{-+}\right)+D_{--}-E_{--}.}
$$

该方案保留全部四个物理贡献，将原来的四次 direct 加四次 exchange 减为四次 direct 加一次 exchange。

适用条件是：轻味简并 $\tau_u=\tau_d$；三条 quark 腿使用相同空间 basis 和 smearing；源、汇 VVV 对三个本征指标均全反对称；使用当前 $A_4$ 的二维块；第三个夸克在源汇两端都取 $+$。改变这些条件时必须重新证明，不能仅按收缩字符串相同而套用。

### 18.3 反对称张量恒等式推导

以汇 sector 为 $+$ 为例，令 $X=T_{+r}$、$Y=T_{++}$，$V_{abc}$ 为汇 VVV，$W_{def}$ 为已共轭的源 VVV。省略固定时间及两端 gamma 的公共符号，定义

$$
B_{gh}^{cf}=\sum_{abdejk}V_{abc}W_{def}(\epsilon_2)_{jk}
X_{gj,ad}X_{hk,be}.
$$

同时交换 $a\leftrightarrow b$、$d\leftrightarrow e$、$j\leftrightarrow k$：两个空间顶点各给出一个负号，自旋 epsilon 再给出一个负号，从而

$$
B_{hg}^{cf}=-B_{gh}^{cf}.
$$

二维反对称矩阵只有一个独立分量，故 $B_{gh}^{cf}=b^{cf}(\epsilon_2)_{gh}$。于是

$$
D_{il}=\sum_{cfgh}(\epsilon_2)_{gh}B_{gh}^{cf}Y_{il,cf}
=2\sum_{cf}b^{cf}Y_{il,cf}.
$$

Exchange 中重命名汇指标 $a\leftrightarrow c$，利用 VVV 反对称性得到

$$
E_{il}=-\sum_{cfgh}(\epsilon_2)_{gh}B_{ih}^{cf}Y_{gl,cf}
=-\sum_{cf}b^{cf}Y_{il,cf}=-\frac12D_{il}.
$$

这证明汇 sector 为 $+$ 的 $++$、$+-$ 两项。在源端作对应推导得到 $-+$ 项。$--$ 项的两个 $u$ 不处于相同二维子空间，不能按此证明化简。该结论来自二维 Schouten/反对称张量恒等式与空间顶点置换对称性，不是不同宇称传播块相等。

### 18.4 已有验证与预计收益

[关系检查脚本](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/audit_four_sector_relations.py)完成以下验证：20 组独立复随机 perambulator 和全反对称源汇顶点；真实 conf=4050,9000、$P_y=2,4$、源汇时间对 (0,5)、(70,3) 共 8 个案例，三条腿一致截断至前 8 个本征矢。三个 exchange 恒等式的最大相对残差分别约为 $9.1\times10^{-15}$ 和 $4.5\times10^{-14}$。真实输入的五次收缩结果相对四维内部旋量参考的最大相对残差约为 $1.5\times10^{-14}$。非反对称顶点负对照不满足该简化。

详细输入、逐组态四项统计及结果见[检查报告](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_L24_results/audit/v20261022111104/sector_relations.md)和[审计 JSON](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_L24_results/audit/v20261022111104/audit_summary.json)。

对于当前 greedy 路径、sink block=4、Nev=100，`opt_einsum.contract_path` 的运算计数从约 $8.000384\times10^{10}$ 减为 $4.160192\times10^{10}$，减少约 48%，对应纯运算量比值约 1.92。这不是 GPU 实测加速比，不能直接外推为完整流程提速。

## 19. 完整 100 维、10 个组态的受控计时检验

新的测试使用原样本前 10 个组态 `4050,4100,...,4500`，ydir 的 $P_y=2,4$，全部 72 个源时间、$t_{\rm sep}=2,\ldots,30$、完整 Nev=100、complex128、sink block=4。三种方法在同一任务、同一 GPU 上使用相同输入：外部正宇称投影、原四项八次收缩、简化五次收缩。

每种 block size 首先预热，正式计算执行两遍，轮换全部六种方法顺序。每个输出块都与历史完整参考、同次外部投影结果及重复计算结果比较，保留全部 2x2 张量。检查全部 VVV 的三个换位反对称性，不通过则失败，不强制对称化输入。

纯收缩计时在前后执行 CUDA 同步，包含 gamma dressing，不包含 parity block 构造、输出 D2H、VVV 和读盘；这些阶段独立记录。二维方案与外部投影比较时还报告投影加收缩成本。完整流程数字采用共享 VVV/I/O 加各算法计算成本的重建估计，明确区别于各自独立冷缓存运行，避免将 $P_y=4$ 的页缓存收益误判为算法提速。

复现入口：[五次收缩测试实现](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_split_five.py)、[计时与数值验证程序](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/benchmark_five_l24.py)、[Slurm 提交脚本](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/submit_five_l24_10.sh)。提交分区为 `na800-sug`，reservation 为 `donghx_223`。本节描述测试设计，实际完整 100 维验证和耗时结论须等待任务完成后读取日志与元数据。

2026-10-22 已提交数组任务 `3865253`，对应上述 10 个组态，每个任务分别计算两个动量；结束后通过 `afterany:3865253` 触发 CPU 汇总任务 `3865256`。提交时预约节点全部 16 张 GPU 已被占用，本测试尚在排队，不能据此给出实测收益或完整精度验证通过的结论。

输出目录为 `/public/group/lqcd/donghx/2pt_gpu_new/L64x128/parity_five_L24_results/job_3865253/`。自动汇总程序 [summarize_five_l24.py](/public/group/lqcd/donghx/2pt_gpu_new/L64x128/summarize_five_l24.py) 会重新读取全部保存张量，核验 shape、complex128、覆盖时间对和历史参考；写出 `timing_comparison.csv`、`summary.json` 与 `summary.md`。只有 20 个组态—动量案例全部通过才标记 `all_pass=true`；缺失或失败案例将保留在报告中，并使汇总任务以非零退出码结束。提速比采用同批样本耗时总和之比，避免直接平均逐组态加速比造成偏差。
