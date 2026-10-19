# Gradient flow 的流时间窗口：限制的来源、推导与本项目的使用边界

本文总结四维 thin-link gradient flow 下 quark/gluon quasi-PDF 使用小流时间转换时的尺度限制，以及 Wilson loop / Polyakov loop 测线质量时应另行检查的限制。公式采用自然单位，$\hbar=c=1$；涉及 fm 和 GeV 的换算时显式恢复 $\hbar c$。

本文是理论与选窗交接说明，不是新的数据审计或物理验收。没有修改生产代码、matching 系数、已有拟合、组态清单或任务参数。文中“必须检查”不等于目前已经检查通过；“待确认”也不能在后续交接中改写为事实。

## 1. 先固定符号：物理流时间、格点流时间和平滑半径

本项目统一使用

$$
t_f:\ \text{物理流时间},\qquad [t_f]=\mathrm{length}^2,
\qquad \tau\equiv\frac{t_f}{a^2},
\qquad r_f\equiv\sqrt{8t_f}=a\sqrt{8\tau}.
$$

- $a$：格距；$\tau$：无量纲 flow checkpoint。
- $z$：双局域端点之间的物理空间距离；$n\equiv|z|/a$。
- $P_z$：物理核子动量；$n_z$：其整数动量标签，两者不能混写。
- $N_s$：沿动量方向的格点数；本项目这两个 ensemble 均为 $N_s=32$。
- $\epsilon\equiv\Delta\tau$：数值积分步长；等步长时 $\tau=N_{\rm step}\epsilon$。
- $t_{\rm sep}$、$t_J$：关联函数的物理欧式时间，与 $t_f$ 不同量纲。

特别注意：Monahan–Orginos [2] 用 $\tau$ 表示**有量纲**流时间，对应本文的 $t_f$；不能把该文的 $\tau^{-1/2}$ 直接解释为代码中无量纲 checkpoint 的倒数。

## 2. 结论总览：每条不等式在控制什么

| 条件或诊断量 | 目的 | 依据及边界 |
| --- | --- | --- |
| $t_f>0$ | 定义有限流时间观测量 | 正流时间的 UV 平滑；费米子仍需同处方场归一化。有限不代表已转换为目标 PDF。[1,2,4] |
| $a^2/t_f\ll1$ | 控制流增强的格点伪影 | 热核/Laplacian 展开及 Symanzik 分析；系数依赖 action、flow、observable 和边界改善。[5] |
| $a/r_f\ll1$ | 平滑半径被格点分辨 | 与上一行是同一参数层级，但数值差一个 8，不能使用相同数字阈值。 |
| $r_f/\lvert z\rvert\ll1$，$z\ne0$ | 控制双局域有限流半径效应 | 与采用渐近的非局域转换有关；不是 $z=0$ 的匹配公式。[4] |
| $\sqrt{t_f}\lvert P_z\rvert\ll1$ | 外部动量低于流的短距离尺度 | smeared-quasi 的文献明确采用 $\Lambda_{\rm QCD},M_N\ll\lvert P_z\rvert\ll t_f^{-1/2}$。[2,3] |
| $\sqrt{t_f}\Lambda_{\rm QCD}\ll1$；有质量时检查 $\sqrt{t_f}m$ | 控制强子/质量尺度的小流时间展开 | 具体首项修正的幂次取决于算符与对称性，不能统一断言为 $O(t_f\Lambda^2)$。[2,4] |
| $\alpha_s(\mu_t)$ 可控，large log 可控 | 控制一圈/低阶微扰截断 | $\mu_t\propto1/\sqrt{t_f}$；有方案常数，并非固定的 universal GeV 上限。[4] |
| $r_f/L$、$r_f/\beta$、$r_f/d_{\rm src,sink}$ 足够小并测试 | 控制周期像、接触及时间重叠 | 几何诊断；不能代替普通有限体积和激发态分析。 |

典型的**参数层级**可以写为

$$
a\ll r_f\ll|z|,\qquad
\Lambda_{\rm QCD},M_N\ll |P_z|\ll t_f^{-1/2},\qquad z\ne0.
$$

这里 $\ll$ 不是 $<$，也不是一个已由文献确定的通用 0.3、0.5 或 1.0 cut。实际阈值是需要报告和变化的全局系统选择，不应逐点挑选使误差最小的窗口。

该层级是当前渐近方案的控制目标，不是说层级之外的有限流时间数据“不存在”或“一定错误”。超窗数据仍可作诊断，但不能自动解释为受控的小流时间结果。

## 3. $r_f=\sqrt{8t_f}$ 从哪里来？

四维连续 flow 线性化后，在适当流规范下有热方程形式；费米子自由场也有相同标量热核：

$$
\partial_{t_f}B_\mu=\partial^2B_\mu,\qquad
\widetilde B_\mu(t_f,q_E)=e^{-t_fq_E^2}\widetilde A_\mu(q_E).
$$

坐标空间为

$$
B_\mu(t_f,x)=\int d^4y\,K_{t_f}(x-y)A_\mu(y),\qquad
K_{t_f}(u)=\frac{e^{-u^2/(4t_f)}}{(4\pi t_f)^2}.
$$

核归一化为 $\int d^4u\,K_{t_f}(u)=1$，各坐标方差为 $2t_f$，因此

$$
\langle u^2\rangle_K=\sum_{\mu=1}^4\langle u_\mu^2\rangle_K=8t_f,
\qquad r_f=\sqrt{\langle u^2\rangle_K}=\sqrt{8t_f}.
$$

直接依据为 Lüscher [1] 式 (2.11)–(2.12) 及其后说明。高斯核没有紧支撑，所以 $r_f$ 是均方根尺度，**不是距离超过 $r_f$ 就严格为零的截断半径**。相互作用 flow 的规范协变核也不能简单当作处处自由高斯。

## 4. 为什么有下限 $a^2/t_f\ll1$？

### 4.1 最近邻 Laplacian 的推导

以自由最近邻离散 Laplacian 为可算例子：

$$
\widehat q^2=\frac4{a^2}\sum_\mu\sin^2\left(\frac{aq_\mu}{2}\right)
=q^2-\frac{a^2}{12}\sum_\mu q_\mu^4+O(a^4q^6).
$$

故格点热核相对于连续热核的展开是

$$
e^{-t_f\widehat q^2}=e^{-t_fq^2}
\left[1+\frac{a^2t_f}{12}\sum_\mu q_\mu^4+\cdots\right].
$$

流调节的 UV 积分典型动量满足 $q^2\sim1/t_f$，所以相对修正可以标度为

$$
a^2t_fq^4\sim\frac{a^2}{t_f}.
$$

这说明固定 $a$ 越向小流时间走，流增强伪影可能越大。对**固定低动量 Fourier mode**，修正仍是 $a^2t_fq^4$，不能说每个 mode 都有 $a^2/t_f$。

### 4.2 可完整计算的四维热迹例子

在 $t_f/a^2\gg1$ 时，将 Brillouin 区扩展到 $\mathbb R^4$ 得

$$
I(t_f,a)=\int_{\rm BZ}\frac{d^4q}{(2\pi)^4}e^{-2t_f\widehat q^2}
=\frac1{(8\pi t_f)^2}
\left[1+\frac{a^2}{8t_f}+O\left(\frac{a^4}{t_f^2}\right)\right].
$$

推导：$e^{-2t_f\widehat q^2}=e^{-2t_fq^2}[1+(a^2t_f/6)\sum q_\mu^4+\cdots]$；
高斯平均 $\langle q_\mu^4\rangle=3/(16t_f^2)$，所以四个方向的修正合为 $a^2/(8t_f)$。
**这个 $1/8$ 只是该 toy integral 的系数，不是本项目 gluon OPE 或 quark C3 的已测系数。**

更系统的依据是 Ramos–Sint [5] 的 Symanzik 分析：第 4.2 节式 (4.24) 明确出现相对 $a^2/t_f$ 项，系数依赖 action、flow、observable 和流边界条件。改进可以消去某些领先项，但不能只凭采用 clover 或换小积分步长就宣称所有伪影已消失。费米子算符及流边界的改善是否完整，属于本项目“待确认”，不能由纯规范结果代替。

### 4.3 为什么不能把两个数值诊断混用？

$$
\frac{a^2}{t_f}=\frac1\tau,\qquad
\left(\frac a{r_f}\right)^2=\frac1{8\tau}.
$$

二者只有参数意义上的同阶。例如 $\tau=0.5$ 时 $r_f/a=2$，但 $a^2/t_f=2$；“流半径已有两个格距”不等于 $a^2/t_f\ll1$。另一方面 $a^2/t_f$ 大也不能单独给出实际误差百分数，因为系数可能较小或被改善抵消。需要真实的多格距/离散化检验。

## 5. 为什么要 $r_f\ll|z|$？

流把算符端点及其周围场平均到宽度 $O(\sqrt{t_f})$ 的区域；若要以“两个分离端点的渐近非局域算符”描述它，流半径不能与端点距离同阶。

直观地，平滑函数的卷积有

$$
\int d^4u\,K_{t_f}(u)F(x+u)
=F(x)+t_f\partial^2F(x)+\cdots.
$$

若 $F$ 在 $|z|$ 尺度变化，其修正可由 $t_f/z^2$ 衡量；具体 Wilson-line 结果还可能有高斯尾、特殊函数及其他允许结构，不能强制所有数据只含 $t_f/z^2$ 的多项式。

Brambilla–Wang [4] 第 3.2 节、式 (3.13) 给出局域流算符展开，第 4.7 节式 (4.69)–(4.74) 给出相应 Wilson-line 转换；附录 A 和 Fig. 7 比较了完整与渐近的一圈结果，发现在其**零外动量**计算中 $r_f<|z|$ 后有限流时间效应较小。

限制：

- Fig. 7 不是有限 $P_z$ 的全面验证，也不是所有 gluon tensor sector 的误差上界。
- 本项目使用更明确的 $r_f/|z|$ 扫描，不能把图中的经验观察提升为 universal cut。
- $z=0$ 时 $r_f/|z|$ 无定义。局域荷/局域 gluon 算符需相应局域归一化、混合或 matching，不能机械代入非局域渐近公式。
- 对很短 $z$，可以保留数据作有限流时间诊断；若要定量使用，需要其适用的完整系数或独立论证。

## 6. 为什么要 $\sqrt{t_f}|P_z|\ll1$？

### 6.1 最直接的文献出处

Monahan–Orginos [2] 第 3.1 节式 (3.13)、(3.18) 明确采用

$$
\boxed{\Lambda_{\rm QCD},\ M_N\ll |P_z|\ll t_f^{-1/2}.}
$$

右侧等价于 $\sqrt{t_f}|P_z|\ll1$；左侧来自抑制 target-mass / higher-twist effects 的大动量要求。Monahan [3] 第 II.C 节式 (8) 重申该层级。[3] 中受影响的一圈转换已被修正；**引用其尺度讨论不代表回到旧 matching 系数**。

### 6.2 热核与短距离展开如何解释这个参数？

自由场欧式 Fourier mode 有

$$
e^{-t_fq_E^2}=1-t_fq_E^2+\cdots.
$$

若该 mode 只有空间动量 $q_z$，领先展开需 $t_fq_z^2\ll1$。更一般地，流的短距离区域典型环动量 $k\sim1/\sqrt{t_f}$，外动量展开参数为

$$
\frac{p}{k}\sim\sqrt{t_f}\,p.
$$

因此 $\sqrt{t_f}|P_z|$ 是应监测的外尺度诊断。不能因此断言核子矩阵元满足 $h^{\rm GF}=e^{-t_fP_z^2}h$：真实对象有欧式能量、相互作用、端点和 Wilson line；四维 $q_E^2$ 也不等于一般物理外态的 $P_z^2$。

### 6.3 对我们“两步 matching”的准确定位

当前先用 [4] 的领先小流时间系数转换到未流化的重整化 quasi 算符，再做 channel-specific LaMET matching。领先系数可用零外动量求出，因为它是外态无关的短距离系数；**系数外态无关不等于在任意有限流时间下余项与外动量无关**。

$\sqrt{t_f}|P_z|\ll1$ 是本方案沿用的受控层级/保守诊断，不是仅由 $r_f\ll|z|$ 推出的定理，也不是已证明所有 quark/gluon 通道必有某个相同 $t_fP_z^2$ 修正。具体首项、系数及可能的运动学/对称性抵消“待确认”。若未来使用保留完整外动量依赖的 finite-flow kernel，需重新按其因子化定理评估适用窗，不能未经推导直接删除或沿用本 cut。

### 6.4 为什么增大 $P_z$ 不总是更好？

大动量有利于降低 $M_N^2/P_z^2$、$\Lambda_{\rm QCD}^2/P_z^2$，却会增大 $\sqrt{t_f}|P_z|$ 及格点动量误差。必须同时检查

$$
\frac{M_N}{|P_z|},\quad\frac{\Lambda_{\rm QCD}}{|P_z|},\quad
\sqrt{t_f}|P_z|,\quad a|P_z|.
$$

对应的理想顺序可概括为 $\Lambda_{\rm QCD},M_N\ll |P_z|\ll1/\sqrt{t_f}\ll1/a$，不包含数值阈值。

## 7. 强子尺度、微扰尺度与“首项修正”的区别

局域场/算符的 flow 展开可形式上写为 $O(t_f)=\sum_i c_i(t_f,\mu)O_i^{\rm R}(\mu)$。高维项的系数按维数差携带 $t_f^{\Delta d/2}$，还可带对数。若第一个允许的维数差为 2，才自然出现 $t_f\Lambda^2$；有辅助直线场、端点等结构时不能不检查算符基就排除其他幂次。

为避免过度表述，记录

$$
\sqrt{t_f}\Lambda_{\rm QCD}\ll1,\qquad
t_f\Lambda_{\rm QCD}^2\ll1
$$

是等价的“小参数要求”，但它们不说明余项究竟线性还是二次于 $\sqrt{t_f}\Lambda_{\rm QCD}$。[2] 式 (3.11)–(3.12) 的余项记为 $O(\sqrt{t_f}\Lambda_{\rm QCD})$；[4] 式 (3.13) 使用 $O(t_f)$。两者展开对象不同，不能抹掉区别后统一宣称“所有修正都是 $O(t_f)$”。

质量不为零时也需检查 $\sqrt{t_f}m$ 及同处方质量响应。当前质量less/ringed matching 的适用性不能仅用小 pion mass 代替验证。

微扰系数若含 $L_t=\ln(2\mu^2t_fe^{\gamma_E})$，其零对数自然尺度为

$$
\mu_t=\frac{e^{-\gamma_E/2}}{\sqrt{2t_f}}.
$$

增大 $t_f$ 会降低 $\mu_t$，可能使 $\alpha_s(\mu_t)$ 更大；固定 $\mu$ 还需控制较大 $L_t$。[4] 提供其一圈背景。应变化尺度/微扰阶数或作适当演化，而不是把 $\mu=2\,\mathrm{GeV}$ 当成已消除高阶不确定度。

## 8. 所有条件如何换成代码里的 $\tau$？

本项目通常取周期格点 Fourier 动量

$$
P_z=\frac{2\pi n_z}{N_sa},\qquad z=na.
$$

因此

$$
\frac{a^2}{t_f}=\frac1\tau,\qquad
\frac a{r_f}=\frac1{\sqrt{8\tau}},\qquad
\frac{r_f}{|z|}=\frac{\sqrt{8\tau}}n,
$$

$$
\sqrt{t_f}|P_z|=\sqrt\tau\frac{2\pi|n_z|}{N_s},\qquad
t_fP_z^2=\tau\left(\frac{2\pi n_z}{N_s}\right)^2.
$$

用 $\Lambda$ 的 GeV 数值时必须换单位：

$$
t_f\Lambda^2=\tau\left(\frac{a[\mathrm{fm}]\,\Lambda[\mathrm{GeV}]}{0.1973269804\,\mathrm{GeV\,fm}}\right)^2.
$$

如果指定**示例性的全局阈值** $\delta_a,\eta_z,\eta_P,\eta_\Lambda$，定义

$$
\frac{a^2}{t_f}<\delta_a,\qquad
\frac{r_f}{|z|}<\eta_z,\qquad
\sqrt{t_f}|P_z|<\eta_P,\qquad
\sqrt{t_f}\Lambda<\eta_\Lambda,
$$

则候选窗为

$$
\boxed{
\frac1{\delta_a}<\tau<\min\left\{
\frac{\eta_z^2n^2}{8},\quad
\eta_P^2\left(\frac{N_s}{2\pi n_z}\right)^2,\quad
\frac{\eta_\Lambda^2}{(a\Lambda)^2}
\right\}.}
$$

这里最后一项用自然单位；$n_z=0$ 时不加动量上界，$z=0$ 不能使用这一非局域 mask。**只有下限小于所有上限时才有候选窗口。** 此公式是由定义代数推得，不代表阈值已获物理验收。

若改用 $a/r_f<\eta_a$，下限应改为 $\tau>1/(8\eta_a^2)$，不能与 $a^2/t_f<\delta_a$ 的下限混用。若改用 $r_f|P_z|<\eta_{rP}$，则 $\eta_P=\eta_{rP}/\sqrt8$。

## 9. 对 L32x96、L32x64 与 $n_z=6$ 的含义

目前项目记录的格距为

| ensemble | $a$ [fm] | 空间格点数 $N_s$ |
| --- | --- | --- |
| L32x96 | 0.0775 | 32 |
| L32x64 | 0.0897 | 32 |

这是项目输入，不是本文独立重新定标的结果。同一 $\tau,n,n_z$ 时，两者的 $a/r_f$、$r_f/|z|$、$\sqrt{t_f}|P_z|$ 相同，但物理 $t_f,r_f,z,P_z$ 不同。

例如 $\tau=0.5$ 有 $r_f=2a$，分别为 0.1550 fm 与 0.1794 fm。固定物理 $r_f$ 应使用

$$
\tau_{64}=\tau_{96}\left(\frac{0.0775}{0.0897}\right)^2,
$$

而不是复制 checkpoint 列表。固定相同 $n_z$ 也不是固定物理动量；跨 ensemble 需要相应的物理运动学匹配。

对 $N_s=32$，仅使 $\sqrt{t_f}|P_z|<1$ 的交叉尺度为

| $n_z$ | $\tau_{\sqrt{t_f}\lvert P_z\rvert=1}=(32/(2\pi n_z))^2$ |
| --- | --- |
| 3 | 约 2.882 |
| 4 | 约 1.621 |
| 5 | 约 1.038 |
| 6 | 约 0.721 |

表中是“小参数达到 1”的位置，**不是推荐最大 flow time**。例如 $n_z=6$ 要求 $\sqrt{t_f}|P_z|\ll1$ 时应写 $\tau\ll0.721$，而不是只需 $\tau<0.721$。

对 $n_z=6$，若采用 $a^2/t_f\ll1$ 为保守截止展开标准，则 $\tau\gg1$ 与 $\tau\ll0.721$ 没有宽阔的参数重叠。进一步

$$
\left(\frac{a^2}{t_f}\right)(t_fP_z^2)=(aP_z)^2
=\left(\frac{2\pi n_z}{32}\right)^2
$$

与流时间无关；$n_z=6$ 右侧约 1.388，说明仅调 $t_f$ 不可能把两项同时变得很小。实际伪影系数是否被改善压低，仍需验证；此结论不等于宣称所有有限流时间测量无效。

一些常用点的诊断如下，保留近似到三位小数：

| $\tau$ | $r_f/a$ | $a^2/t_f$ | $t_fP_z^2$，$n_z=5$ | $t_fP_z^2$，$n_z=6$ |
| --- | --- | --- | --- | --- |
| 0.1 | 0.894 | 10.000 | 0.096 | 0.139 |
| 0.2 | 1.265 | 5.000 | 0.193 | 0.278 |
| 0.5 | 2.000 | 2.000 | 0.482 | 0.694 |
| 0.8 | 2.530 | 1.250 | 0.771 | 1.110 |
| 1.0 | 2.828 | 1.000 | 0.964 | 1.388 |
| 1.4 | 3.347 | 0.714 | 1.349 | 1.943 |
| 1.8 | 3.795 | 0.556 | 1.735 | 2.498 |
| 3.8 | 5.514 | 0.263 | 3.663 | 5.274 |

这些是参数诊断，不是数据误差的预测。它们也解释了为何“只用大 flow time 减小统计误差”不能保证更好的小流时间外推。

## 10. 顺序、线性发散及为何不能裸数据直接外推

正流时间可调节 Wilson-line 线自能，但其流时间行为含 $1/\sqrt{t_f}$，还存在对数系数。沿本项目符号，若未减除矩阵元含 $e^{-m_{\rm line}|z|}$，则转换需对应的 $e^{+m_{\rm line}|z|}$；$\delta m=-m_{\rm line}$。这是方案内容，不是可任意切换的符号约定。[4]

因此不能对原始裸矩阵元简单作 $h^{\rm bare}=h_0+c\tau$ 并把截距当作物理结果。先落实同处方场归一化与线质量/系数转换，且避免与 hybrid、ratio 等处方重复减除。非微扰线质量也不能只因拟合成功就认为有限部分已与 $\overline{\rm MS}$ 闭合。

受控目标可示意写为

$$
h^{\overline{\rm MS}}(z,P_z;\mu)
=\lim_{t_f\to0}\left[
\mathcal C^{-1}(t_f,\mu)e^{-\delta m(t_f)|z|}
\lim_{a\to0}h^{\rm GF,norm}(a,t_f,z,P_z)
\right],
$$

必要时系数为混合矩阵，且后续仍有 LaMET matching。内层连续极限保持物理 $t_f,z,P_z$；可以在有理论支持、数据可辨识时联合外推，但不能未经论证对调极限。

固定 $a$ 时常见的诊断拟合结构可包含

$$
h_{\rm conv}(a,t_f)=h_R+A t_f+B\frac{a^2}{t_f}+
\text{其他允许的流时间、截止和混合项}.
$$

这只是示意，不是所有通道已证明的完整 ansatz。在一个格距上，$t_f\to0$ 使截止项放大；$\tau\ge1$ 的截距、线性/二次拟合或 AIC 加权只能称探索性/窗口敏感性结果，不能自动叫受控连续小流时间极限。

## 11. C3/C2、Wilson loop 和 Polyakov loop 的限制要分开

### 11.1 Quark/gluon 双局域 C3/C2

除上述 $z,P_z$ 条件外，四维 flow 有欧式时间方向的扩散，需要检查插入到源、汇的距离及周期绕回。$t_{\rm sep}>0.6$ fm 是激发态/拟合选择，不自动保证 $r_f\ll\min(t_J-t_{\rm src},t_{\rm sink}-t_J)$。

时间缓存/聚合截断需用实际 covariant-flow tail 与 C3 回归验证，不能因为 $r_f$ 有限就把核当作紧支撑。A/B blending 权重估计是另一个问题，不会改变物理流时间窗口条件。

### 11.2 Wilson-loop 线质量拟合

Wilson loop 没有核子 $P_z$，因此**不能把 $\sqrt{t_f}P_z$ 的核子条件套在 loop 数据本身**。应检查的是格点分辨率、$r_f/R$、$r_f/T$、$r_f/L$、$r_f/\beta$，以及长时间有效势/窗口稳定性。

有限 $T$ 下 $-\ln\overline W(R,T;t_f)=b+V^{\rm eff}(R;t_f)T$ 是势估计；$V^{\rm eff}=2m_{\rm line}+V^{\rm sub}$ 的解释仍需控制有限时间、端点/拐角及流依赖的有限势。flow 改善信号不等于基态自动投影；四维时间平滑后的有限时间谱解释也需审慎。

项目中 $\max(r_f(t_f),r_f(t_{\rm ref}))/R\le0.5$ 是已经声明的**分析几何 mask**，不是文献证明的通用安全边界。线质量拟合允许扫描比 PDF 渐近转换更大的 flow time，但其结果不能未经方案和表示检查直接用于 PDF。

### 11.3 Polyakov loop

单圈测量与静态源自由能相关，成对圈包含距离依赖与有限温度相互作用；它们都不是任意有限温度下的零温势。检查 $r_f/\beta$、$r_f/R$（pair）及正负/中心相位、有限温度项、周期像、flow 窗和表示。基本/伴随表示分别处理，不任意外加 Casimir 缩放替代测量。

本次已完成的 L32x64 静态势诊断图在

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/static_potential_L32x64_1700conf_20261017/
```

其 README 和 report 明确保留 finite-T / poor-Q 状态；不能把图能画出来改写为线性发散减除已物理验证。

## 12. $\epsilon$、$N_{\rm step}$ 与上述限制不是一回事

等步长积分 $\tau=N_{\rm step}\epsilon$。固定 $\tau$ 时把 $\epsilon$ 减半、步数加倍，测试积分器截断误差；不改变物理 $r_f,z,P_z$，也不改变 $a^2/t_f$。

项目现有记录通常采用 $\epsilon=0.01$，部分细步长对照用 0.005；某个具体产物仍以其 manifest 为准，本文没有重新审计全部任务。比如请求 $\tau=0.5$、$\epsilon=0.01$ 时为 50 步；多 checkpoint 共享 flow history 时不要误把“从上一个 checkpoint 前进的步数”当成总步数。

积分器验证、SU(3) 检查、bootstrap 覆盖、AIC 接受率都不能替代连续极限/外尺度分离。

## 13. 下一步最具体的选窗操作

1. 对每个 ensemble、$z$、$n_z$、flow checkpoint 保存 $a,t_f,\tau,r_f$ 和第 8 节四项诊断；额外保存 $aP_z,a/z$、源汇距离和时间缓存的检验状态。
2. 对 quark、gluon 与 loop 使用不同的物理 mask；统一选择若干全局阈值方案并扫描，不按每个点的最小误差选择阈值。空窗明确标记，不补零、不用坏点回填物理结果。
3. 同一阈值扫描、不同 flow 和通道保留共享组态/重采样索引；不要把 bootstrap 数增加当作独立组态增加。
4. 区分裸矩阵元、归一化后、转换后、固定格距零流截距和最终 light-cone PDF，分别保存状态。AIC 只比较已给候选，并不能修复理论窗口缺失。
5. 继续做 epsilon 减半对照，但将其与 flow-window、有限 $T$、离散化、质量、有限体积及微扰阶数系统误差分开。
6. 对 $n_z=6$ 等紧窗口情形，优先评估更细格距且固定物理动量，或有针对性的改善/完整 finite-flow kernel；不能只把 $\tau$ 越取越大。新增统计提高信号，但不改变 $aP_z$ 的尺度冲突。

## 14. 已确认、不可随意改写与待确认项

已确认并需保留：

- 本项目 $\tau=t_f/a^2$、$r_f=\sqrt{8t_f}$；自然单位及动量整数/物理动量的区分。
- [2] 的尺度层级是明确的文献假设；它不等价于 $r_f\ll|z|$ 一个条件。
- $a^2/t_f$ 的来源可由格点热核推导，且有 [5] 的系统依据；toy 系数不是本项目测量值。
- [4] 的 Fig. 7 是零外动量检查，不能扩张成任意动量/任意 gluon sector 的误差界。
- 线质量符号 $\delta m=-m_{\rm line}$、matching 方案和避免双重减除的要求。
- 一个格距的数值外推不是已经完成连续极限。

待确认，不作为既成事实：

- 每种算符/极化通道的实际有限流余项首幂、系数和外动量抵消。
- 本项目 fermion action、flow、boundary、bilocal operator 的完整改善及剩余 $O(a)$ / $O(a^2)$ 结构。
- 一个可以用于正式结果的统一数值 cut 和当前各点真正通过的尺度窗口。
- 真正控制高动量、有限 $t_f$ 的完整通道 kernel 是否已具备并适用于当前定义；本文没有新做这项文献审计。
- 本项目 loop 线质量候选的有限温度/有限 $T$、流窗、有限部分和目标 scheme 的闭合。

## 15. 文献与本地证据导航

[1] M. Lüscher, *Properties and uses of the Wilson flow in lattice QCD*, arXiv:1006.4518，式 (2.11)–(2.12)：热核及四维流半径。本文未把其特定观测量的经验流半径数值当作 PDF 的 universal cut。

[2] C. Monahan and K. Orginos, *Quasi parton distributions and the gradient flow*, arXiv:1612.01584v4, JHEP 03 (2017) 116；第 3.1 节式 (3.11)–(3.18)：小流时间余项及 $\Lambda_{\rm QCD},M_N\ll P_z\ll t_f^{-1/2}$；式 (2.13) 说明固定物理流时间的连续极限。

[3] C. Monahan, *Smeared quasidistributions in perturbation theory*, arXiv:1710.04607, Phys. Rev. D 97 (2018) 054507；第 II.C 节式 (8)：尺度层级。存在 Phys. Rev. D 110 (2024) 059902 的勘误；本项目采用修正后的 [4] 转换系数，不复用旧系数。

[4] N. Brambilla and X.-P. Wang, *Off-lightcone Wilson-line operators in gradient flow*, arXiv:2312.05032v3, JHEP 06 (2024) 210；第 3.2 节式 (3.13)、第 4.7 节式 (4.69)–(4.74)、附录 A / Fig. 7。这里采用 PDF 中带章节编号的公式号；HTML 渲染可能使用连续编号。

[5] A. Ramos and S. Sint, *Symanzik improvement of the gradient flow in lattice gauge theories*, arXiv:1508.05552v1；第 3 节、4.2 节式 (4.24) 和有限体积式 (4.33)：改善、边界项及 $a^2/t_f$ 截止结构。其纯规范分析不自动证明本项目所有费米子/双局域改善均已实现。

本地主要依据：

```text
/public/home/donghx/Note/Gradient_Flow/main.tex
  sec:sfte-implementation / eq:sfte-window / eq:sfte-diagnostics
  sec:flow-artifact-derivation
/public/home/donghx/Note/Gradient_Flow/reference.bib
/public/home/donghx/.codex/Markdown/gradient_flow.md
  第 15.7 节 flow 窗口限制与 a²/t 项推导
/public/home/donghx/.codex/Markdown/divergence.md
  流窗及线质量/闭合圈诊断
/public/home/donghx/.codex/skills/gradient-flow-quark-pdf/references/theory-and-matching.md
/public/home/donghx/Note/Gradient_Flow/Pion_DA.pdf
  第 15 页的静态势/线性发散信号局限，不作为任意 Pz 的 flow cut 证明
```

本次核对日期：2026-10-17（Asia/Shanghai）。

本次会话标题：Flow-time 窗口限制的来源、热核推导及有限格距使用边界。
