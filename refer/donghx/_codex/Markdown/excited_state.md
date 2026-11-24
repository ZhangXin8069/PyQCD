# Quark PDF 的激发态污染：谱分解、普通 ratio 与 self ratio

更新日期：2026-11-11。

本文面向当前 connected quark PDF 的 unpolarized、helicity 和 transversity 分析，解释源汇重叠因子与非局域算符位移的区别，并给出两态、三态截断下普通 ratio 和 self ratio 的拟合形式。这里只讨论裸关联函数的态分离。

## 1. 记号、归一化与适用范围

用以下记号区分源汇间隔、插入时刻、非局域位移及时间方向总长度：

$$
T=t_{\mathrm{sep}},\qquad 0<\tau<T,\qquad z=\text{Wilson line 的有符号空间位移},\qquad L_t=\text{时间方向总长度}.
$$

实际保存文件用非负位移长度和两个位移分支表示正负位移；以下先对一个固定分支、固定动量方向推导。固定 ensemble、核子动量、核子源汇内插算符及其 smearing，源汇动量相同，插入算符不传递净动量。

$$
\mathbf P_f=\mathbf P_i=\mathbf P.
$$

考虑空间上非局域、但位于同一插入时间片的算符：

$$
O_\Gamma(z,\tau)
=\bar q(\tau,z\hat{\mathbf e})\,\Gamma\,
W(z\hat{\mathbf e},0;\tau)\,q(\tau,0).
$$

以下把有限体积归一化、自旋投影与运动学因子吸收进谱振幅。实际核子关联函数带有自旋指标和 projector；若采用显式相对论归一化，还需保留相应能量分母。归一化约定可以变化，但不能改变能量、源汇重叠与插入矩阵元各自的物理依赖。

以下公式只保留前向传播。它们省略反向传播的反宇称伙伴及有限时间方向的热贡献；接近时间边界或需要相应精度时必须扩展模型。源汇附近的接触项也没有包含，因此拟合应避开插入端点。

## 2. 重叠因子的含义及位移依赖

设核子内插算符为：

$$
\chi_N.
$$

它与第几个能量本征态之间的重叠通常写成：

$$
\zeta_n(\mathbf P)
=\langle\Omega|\chi_N(0)|n,\mathbf P\rangle.
$$

基态和激发态编号为：

$$
n=0\quad\text{基态},\qquad n=1\quad\text{第一激发态},\qquad n=2\quad\text{第二激发态}.
$$

这个重叠衡量所选核子源汇对各态的耦合强度，依赖动量、ensemble、内插算符、smearing 和变分基选择。它不是插入的非局域流矩阵元。

不同文献也可能使用重叠振幅比或其模方作为同一个希腊字母的定义，例如：

$$
\frac{\zeta_1}{\zeta_0},\qquad
r_1=\frac{|\zeta_1|^2}{|\zeta_0|^2}.
$$

所以必须核对原文定义。若某个符号代表整个三点函数的谱振幅，它还包含插入矩阵元，便会具有位移依赖。

在只改变插入位移、保持源汇构造固定时：

$$
E_n(\mathbf P),\qquad
\Delta_n(\mathbf P)=E_n(\mathbf P)-E_0(\mathbf P),\qquad
\zeta_n(\mathbf P)
\quad\text{均不随 }z\text{ 改变}.
$$

改变 Wilson line 的长度是在改变被测量的算符，并没有改变核子在源、插入点和汇之间传播时的哈密顿量。

插入算符的各态矩阵元为：

$$
\mathcal H_{nm}(z,\mathbf P)
=\langle n,\mathbf P|O_\Gamma(z)|m,\mathbf P\rangle.
$$

这些量都允许随位移变化，包括：

$$
\mathcal H_{00}(z),\quad
\mathcal H_{01}(z),\quad
\mathcal H_{10}(z),\quad
\mathcal H_{11}(z),\quad
\mathcal H_{02}(z),\quad
\mathcal H_{20}(z),\quad
\mathcal H_{12}(z),\quad
\mathcal H_{21}(z),\quad
\mathcal H_{22}(z).
$$

其中对角元描述各态内部的矩阵元，非对角元描述不同态之间的跃迁。不能只让基态项随位移变化，而把激发态跃迁振幅默认固定为位移无关的常数。

| 谱量 | 改变插入位移时的行为 |
|---|---|
| 核子的能量与真实能隙 | 不变 |
| 核子源汇重叠、两点函数谱振幅 | 不变 |
| 两点函数本身 | 不变，普通两点函数没有这个插入位移变量 |
| 基态、跃迁及激发态插入矩阵元 | 一般变化 |
| 三点函数各指数项的总振幅 | 一般变化，因为含插入矩阵元 |
| 激发态污染相对于基态的比例 | 一般变化 |

若改变的同时也是核子源汇的 smearing 或位移构造，重叠因子当然可能改变；那属于源汇也发生了变化的另一种计算。

## 3. 一般谱分解及普通 ratio

为了使实际拟合不依赖重叠归一化的具体写法，先定义两点振幅和三点振幅：

$$
C_2(P,T)=\sum_{n=0}^{N-1}A_n(P)e^{-E_n(P)T},
$$

$$
C_3(z,P,T,\tau)
=\sum_{n,m=0}^{N-1}B_{nm}(z,P)
e^{-E_n(P)(T-\tau)}e^{-E_m(P)\tau}.
$$

第一个指标表示插入之后向汇传播的态，第二个指标表示从源传播到插入之前的态。在省略归一化与自旋结构的标量写法中：

$$
A_n\propto|\zeta_n|^2,\qquad
B_{nm}(z)\propto\zeta_n\,\mathcal H_{nm}(z)\,\zeta_m^*.
$$

实际原始关联函数可能带有共同的整体相位或符号，不能直接因为保存的两点函数为负就判断能谱不合理。矩阵元与振幅的关系须包含实际 projector 和运动学归一化。

定义：

$$
\Delta_0=0,\qquad\Delta_n=E_n-E_0,\qquad
r_n=\frac{A_n}{A_0},\qquad
K_{nm}(z)=\frac{B_{nm}(z)}{A_0}.
$$

于是：

$$
D_N(T)=1+\sum_{n=1}^{N-1}r_n e^{-\Delta_nT},
$$

$$
F_N(z,T,\tau)=\sum_{n,m=0}^{N-1}K_{nm}(z)
e^{-\Delta_n(T-\tau)}e^{-\Delta_m\tau},
$$

$$
C_2(P,T)=A_0e^{-E_0T}D_N(T),\qquad
C_3(z,P,T,\tau)=A_0e^{-E_0T}F_N(z,T,\tau).
$$

普通 ratio 的完整截断模型为：

$$
\boxed{
R_N(z,T,\tau)=\frac{C_3(z,T,\tau)}{C_2(T)}
=\frac{F_N(z,T,\tau)}{D_N(T)}.
}
$$

定义目标基态 ratio 振幅：

$$
M(z)=K_{00}(z)=\frac{B_{00}(z)}{A_0}.
$$

它与所选投影下的基态矩阵元相差已知运动学归一化。实际使用时应沿用当前分析对裸矩阵元的约定。

## 4. 只包含第一激发态：两态模型

两态截断的态集合和能隙为：

$$
n,m\in\{0,1\},\qquad\Delta_1=E_1-E_0>0.
$$

两点函数：

$$
\boxed{
C_2^{(2)}(T)=A_0e^{-E_0T}\left[1+r_1e^{-\Delta_1T}\right].
}
$$

为了便于对照源侧、汇侧跃迁，记：

$$
a_1(z)=K_{01}(z),\qquad
b_1(z)=K_{10}(z),\qquad
c_{11}(z)=K_{11}(z).
$$

三点函数中有四项：

$$
\boxed{
\begin{aligned}
C_3^{(2)}(z,T,\tau)=A_0e^{-E_0T}\Big[
&M(z)
+a_1(z)e^{-\Delta_1\tau}
+b_1(z)e^{-\Delta_1(T-\tau)}\\
&+c_{11}(z)e^{-\Delta_1T}
\Big].
\end{aligned}
}
$$

各项对应：基态到基态、源侧第一激发态到基态、基态到汇侧第一激发态、第一激发态到第一激发态。

### 4.1 普通 ratio 的两态拟合

$$
\boxed{
R_2(z,T,\tau)=
\frac{
M(z)+a_1(z)e^{-\Delta_1\tau}
+b_1(z)e^{-\Delta_1(T-\tau)}
+c_{11}(z)e^{-\Delta_1T}
}{1+r_1e^{-\Delta_1T}}.
}
$$

只要是在这个两态截断内，保留分母的有理函数就是完整形式。若进一步展开分母，保留最低阶污染，可写成：

$$
\begin{aligned}
R_2(z,T,\tau)\approx{}&M(z)
+a_1(z)e^{-\Delta_1\tau}
+b_1(z)e^{-\Delta_1(T-\tau)}\\
&+\left[c_{11}(z)-r_1M(z)\right]e^{-\Delta_1T}.
\end{aligned}
$$

这个近似省略了分母展开带来的乘积项，例如：

$$
e^{-\Delta_1(T+\tau)},\qquad
e^{-\Delta_1(2T-\tau)},\qquad
e^{-2\Delta_1T}.
$$

因此经验模型中最后一个指数的拟合系数一般不是纯激发态对角矩阵元，它混合了两点函数分母修正。

### 4.2 Self ratio 的两态拟合

采用同一通道、flavor、动量、方向、源汇间隔及插入时刻的局域三点函数归一化：

$$
S(z,T,\tau)=\frac{C_3(z,T,\tau)}{C_3(0,T,\tau)}.
$$

两态模型为：

$$
\boxed{
S_2(z,T,\tau)=
\frac{
M(z)+a_1(z)e^{-\Delta_1\tau}
+b_1(z)e^{-\Delta_1(T-\tau)}
+c_{11}(z)e^{-\Delta_1T}
}{
M(0)+a_1(0)e^{-\Delta_1\tau}
+b_1(0)e^{-\Delta_1(T-\tau)}
+c_{11}(0)e^{-\Delta_1T}
}.
}
$$

两点谱分母在这个比值中消去；局域三点函数自己的跃迁污染仍然保留。不能直接把 self ratio 分母替换成一个纯基态局域荷，除非有独立依据证明该近似适用。

当局域基态振幅不为零时，定义目标归一化形状及重标度系数：

$$
h(z)=\frac{M(z)}{M(0)},\qquad
\alpha_1(z)=\frac{a_1(z)}{M(0)},\qquad
\beta_1(z)=\frac{b_1(z)}{M(0)},\qquad
\gamma_{11}(z)=\frac{c_{11}(z)}{M(0)}.
$$

就得到便于直接拟合 self ratio 的形式：

$$
\boxed{
S_2(z,T,\tau)=
\frac{
h(z)+\alpha_1(z)e^{-\Delta_1\tau}
+\beta_1(z)e^{-\Delta_1(T-\tau)}
+\gamma_{11}(z)e^{-\Delta_1T}
}{
1+\alpha_1(0)e^{-\Delta_1\tau}
+\beta_1(0)e^{-\Delta_1(T-\tau)}
+\gamma_{11}(0)e^{-\Delta_1T}
}.
}
$$

若所有污染相对于局域基态项足够小，再作一阶展开：

$$
\begin{aligned}
S_2(z,T,\tau)\approx{}&h(z)\\
&+\left[\alpha_1(z)-h(z)\alpha_1(0)\right]e^{-\Delta_1\tau}\\
&+\left[\beta_1(z)-h(z)\beta_1(0)\right]e^{-\Delta_1(T-\tau)}\\
&+\left[\gamma_{11}(z)-h(z)\gamma_{11}(0)\right]e^{-\Delta_1T}.
\end{aligned}
$$

这说明 self ratio 减少污染的条件是非局域与局域跃迁振幅近似按基态形状成比例，而不是单纯因为两者包含相同能隙：

$$
a_1(z)\approx h(z)a_1(0),\qquad
b_1(z)\approx h(z)b_1(0),\qquad
c_{11}(z)\approx h(z)c_{11}(0).
$$

上述一阶展开还省略污染之间的二阶乘积；它与完整有理函数不能混称为同一个严格模型。

## 5. 包含到第二激发态：三态模型

三态截断包含基态、第一和第二激发态：

$$
n,m\in\{0,1,2\},\qquad
0<\Delta_1=E_1-E_0<\Delta_2=E_2-E_0.
$$

这里第二个能隙以基态为参考；若代码使用相邻能隙，应明确转换关系：

$$
\delta_{21}=E_2-E_1,\qquad\Delta_2=\Delta_1+\delta_{21}.
$$

三态两点函数为：

$$
\boxed{
C_2^{(3)}(T)=A_0e^{-E_0T}
\left[1+r_1e^{-\Delta_1T}+r_2e^{-\Delta_2T}\right].
}
$$

定义其余三点系数：

$$
a_2(z)=K_{02}(z),\qquad b_2(z)=K_{20}(z),\qquad
c_{nm}(z)=K_{nm}(z)\quad(n,m\in\{1,2\}).
$$

三态三点函数总共有九项：

$$
\boxed{
\begin{aligned}
F_3(z,T,\tau)={}&M(z)\\
&+a_1(z)e^{-\Delta_1\tau}
+b_1(z)e^{-\Delta_1(T-\tau)}\\
&+a_2(z)e^{-\Delta_2\tau}
+b_2(z)e^{-\Delta_2(T-\tau)}\\
&+c_{11}(z)e^{-\Delta_1T}
+c_{22}(z)e^{-\Delta_2T}\\
&+c_{12}(z)e^{-\Delta_1(T-\tau)}e^{-\Delta_2\tau}\\
&+c_{21}(z)e^{-\Delta_2(T-\tau)}e^{-\Delta_1\tau}.
\end{aligned}
}
$$

$$
C_3^{(3)}(z,T,\tau)=A_0e^{-E_0T}F_3(z,T,\tau).
$$

与两态相比，新增第二激发态与基态之间的两项跃迁、第二激发态对角项，以及第一与第二激发态之间的两项交叉跃迁。

**完整三态模型不能只在两态式子上增加一个第二能隙指数，而忽略两个激发态之间的跃迁。** 若为控制参数数目而删掉这些交叉项，需要明确标为额外截断近似。

### 5.1 普通 ratio 的三态拟合

$$
\boxed{
R_3(z,T,\tau)=
\frac{F_3(z,T,\tau)}{1+r_1e^{-\Delta_1T}+r_2e^{-\Delta_2T}}.
}
$$

位移无关的共享参数包括两个能隙和两个两点振幅比：

$$
\Delta_1,\quad\Delta_2,\quad r_1,\quad r_2.
$$

上式共四个标量；各非局域三点谱系数允许逐位移变化。多态 C2 与 C3 联合拟合还应共享基态能量及两点谱振幅。

### 5.2 Self ratio 的三态拟合

$$
\boxed{
S_3(z,T,\tau)=\frac{F_3(z,T,\tau)}{F_3(0,T,\tau)}.
}
$$

为显示其可拟合参数，定义：

$$
\alpha_j(z)=\frac{a_j(z)}{M(0)},\qquad
\beta_j(z)=\frac{b_j(z)}{M(0)},\qquad j=1,2,
$$

$$
\gamma_{nm}(z)=\frac{c_{nm}(z)}{M(0)},\qquad n,m\in\{1,2\}.
$$

再定义：

$$
\begin{aligned}
G_3(z,T,\tau)={}&h(z)\\
&+\sum_{j=1}^{2}\left[
\alpha_j(z)e^{-\Delta_j\tau}
+\beta_j(z)e^{-\Delta_j(T-\tau)}\right]\\
&+\sum_{n,m=1}^{2}\gamma_{nm}(z)
e^{-\Delta_n(T-\tau)}e^{-\Delta_m\tau}.
\end{aligned}
$$

则：

$$
\boxed{
S_3(z,T,\tau)=\frac{G_3(z,T,\tau)}{G_3(0,T,\tau)},\qquad h(0)=1.
}
$$

该模型的局域分母有与非局域分子同样的九项结构，局域基态项经归一化固定为一。所有位移共同使用同一套局域分母参数。

统一的一阶污染表达式可以写成：

$$
F_N(z,T,\tau)=M(z)+\epsilon_N(z,T,\tau),
$$

$$
\boxed{
S_N(z,T,\tau)\approx
h(z)+\frac{\epsilon_N(z,T,\tau)-h(z)\epsilon_N(0,T,\tau)}{M(0)}.
}
$$

完整抵消需要各相关谱系数都满足比例关系，而不是仅两个基态项满足归一化关系：

$$
K_{nm}(z)=h(z)K_{nm}(0)
\quad\text{对所保留的全部态对成立}.
$$

## 6. 两类拟合实际能确定什么

### 6.1 普通 ratio

普通 ratio 的长时间极限给出基态裸矩阵元，含当前投影的运动学约定：

$$
\lim_{\tau\to\infty,\,T-\tau\to\infty}R_N(z,T,\tau)=M(z).
$$

其谱分母中保留两点函数的激发态振幅比。可由兼容 C2 分析提供约束，或与 C2、C3 联合拟合，并传播相关误差。

### 6.2 Self ratio

Self ratio 的长时间极限为归一化形状：

$$
\lim_{\tau\to\infty,\,T-\tau\to\infty}S_N(z,T,\tau)=h(z)=\frac{M(z)}{M(0)}.
$$

只用 self ratio 无法确定整体绝对标度，因为把所有局域和非局域三点谱系数同时乘以同一个非零常数，不改变比值：

$$
F_N(z,T,\tau)\longrightarrow\kappa F_N(z,T,\tau),\qquad
F_N(0,T,\tau)\longrightarrow\kappa F_N(0,T,\tau),
$$

$$
S_N(z,T,\tau)\longrightarrow S_N(z,T,\tau).
$$

所以实际参数化应固定整体归一化，例如使用前面的归一化系数。恢复绝对裸矩阵元还需独立局域荷：

$$
M(z)=h(z)M(0).
$$

归一化恒等式：

$$
S_N(0,T,\tau)=1
$$

不提供独立的局域激发态信息，也不能作为一个零误差的普通数据点加入协方差拟合。分母接近零时 self ratio 同样可能失稳；若局域基态矩阵元为零，以上以它为单位的形状参数化不适用。

## 7. 对当前分析的拟合建议与限制

1. **共享能隙与源汇谱参数。** 同一 ensemble、动量及核子源汇构造下，各位移使用相同的真实能级。两个激发态时应保持能级顺序。可使用正参数化：

$$
\Delta_1=e^{\ell_1},\qquad
\Delta_2=\Delta_1+e^{\ell_{21}}.
$$

2. **允许不同位移有不同污染强度。** 即使能隙相同，跃迁与基态的相对振幅也会改变：

$$
\frac{a_j(z)}{M(z)},\qquad
\frac{b_j(z)}{M(z)},\qquad
\frac{c_{nm}(z)}{M(z)}.
$$

因此不能根据一个位移的平台平坦程度，断言所有位移均已消除激发态。截断拟合逐位移得到不同有效能隙，也可能是高态遗漏、噪声和参数退化造成的。

3. **不要默认源侧和汇侧振幅相等。** 对同一源汇构造，适当的厄米性、时间反射、位移投影及动量对齐可能提供振幅关系，但固定有符号位移的非局域算符一般不是直接自伴的。必须先验证实际通道的关系，才能采用：

$$
a_j(z)=b_j(z)
$$

或其复共轭版本。同样，不应在没有依据时删除交叉跃迁。

4. **先拟合复数关系，再按约定提取分量。** 当前普通 ratio 使用复数 C2；局域 C3 也可带复数涨落。一般有：

$$
\operatorname{Re}\left(\frac{F}{D}\right)
\ne\frac{\operatorname{Re}F}{\operatorname{Re}D}.
$$

可以在有明确对称性后建立 even 实部、odd 虚部的实参数模型，但不能未经说明把完整复数谱模型中的每个量都替换成实部。

5. **区分单方向谱式和有限样本方向平均。** 本文的有理函数是对匹配的一个方向/分支定义；投影与平均应遵循现有生产次序。当前估计量逐方向求比值后平均，在有限 bootstrap 样本中通常满足：

$$
\frac{1}{N_d}\sum_d\frac{N_{b,d}}{D_{b,d}}
\ne
\frac{\sum_dN_{b,d}}{\sum_dD_{b,d}}.
$$

源汇和算符按对称性正确对齐后，可以用共同的物理谱参数描述各方向。最严格的处理是逐方向联合拟合，再按定义组合；对已平均 signal 直接套单个有理函数，应声明依赖方向对称性的模型假设。不能为了拟合方便静默改变 bootstrap 的平均顺序。

6. **局域分母参数要与非局域数据一起约束并传播误差。** Self ratio 不显含两点振幅比，但多了局域三点分母的跃迁参数；因此不能仅从分母形式判断它参数更少或一定更容易拟合。可联合多个位移及兼容的局域 ratio/C3 数据，使用同一组态 bootstrap，保留全部相关性。不同数据产品可能存在代数冗余，应检查协方差与参数可辨识性，不能把重复信息当成新增约束。

7. **状态数增加不等于结果自动可靠。** 不加对称性约束时，两态每个位移有四个三点谱系数，三态有九个；复数系数对应更多实自由参数。能隙、局域分母和两点振幅还可能需要额外约束。应先从两态与 C2 联合约束入手，检查窗口稳定性、参数相关性、协方差信息秩及 prior 依赖，再判断数据能否支持三态。仅优化收敛或拟合优度变好不能证明第二激发态已被独立识别。

8. **保持既有质量状态。** 本文是拟合形式推导，没有对当前结果重新拟合，也未证明现有时间窗已实现基态分离。近期 L32x96 的高动量大间隔处仍有弱分母，特别是当前普通 ratio 的对应点：

$$
P=5,\qquad T/a=12.
$$

增加状态数不能修复由小分母造成的 ratio 长尾。应先审计分母，再考虑谱模型；必要时直接联合拟合 C2、C3。

## 8. 公式速查

| 截断 | 两点谱分母 | 普通 ratio | Self ratio |
|---|---|---|---|
| 两态：基态加第一激发态 | \(D_2=1+r_1e^{-\Delta_1T}\) | \(R_2=F_2(z)/D_2\) | \(S_2=F_2(z)/F_2(0)\) |
| 三态：基态加第一、第二激发态 | \(D_3=1+r_1e^{-\Delta_1T}+r_2e^{-\Delta_2T}\) | \(R_3=F_3(z)/D_3\) | \(S_3=F_3(z)/F_3(0)\) |

两态三点谱分子有四项，三态有九项。Self ratio 保留同样的跃迁谱结构，只是分母由两点函数换成了局域三点函数；它提取的是归一化基态形状。

## 9. 与本地数据约定的对应

相关约定可查：

- `Fit_SKILL/C3_Ratio_Bootstrap/README.md`：投影版本、源平均、方向平均及复数 ratio。
- `Fit_SKILL/C3_Ratio_Bootstrap/examples/L32x96_unpolarized_ud/README.md`：C3/C2 原始保存格式与位移分支标签。
- `Fit_SKILL/artifacts/c3_ratio_momentum_audit_20261110/README.md`：当前 C3、C2、位移投影与方向平均审计。
- `Fit_SKILL/artifacts/legacy_ratio_comparison_L32x96_unpolarized_ud_P345_T8to12_20261111/README.md`：新旧 ratio 的时间源平均顺序及弱分母诊断。

上面路径均相对于工作区 `/public/group/lqcd/donghx/Diagram_GluonPDF`。本文不指定新的拟合时间窗，不修改已有生产数据。
