**Gradient flow、hybrid renormalization 的能标，以及 small-flow-time 极限的物理含义**

记录日期：2026-11-16。本文针对当前薄链接、四维 gradient flow 的 unpolarized quark 与 gluon 分析，并对照 hybrid renormalization。文中的重整化矩阵元指坐标空间非局域算符的强子矩阵元；它与完成 LaMET matching 后的光锥 PDF 是不同阶段的对象。

最终结果的能标由重整化方案及转换／matching 中指定的尺度确定。flow time、格距、强子动量及 hybrid 拼接距离都不能单独确定最终 MSbar PDF 的能标。

关于 small-flow-time，最需要明确的结论是：**零-flow 外推负责去掉有限流化效应，不能替代紫外重整化。** 对未经一致重整化／转换的量直接外推，得到一个有限截距并不证明它是所有紫外问题均已处理的矩阵元。只有在归一化、线质量、有限端点转换及必要的算符混合均按同一方案处理，并控制连续极限和有限阶误差后，小流时间极限才可对应所声明尺度下的重整化矩阵元。

先区分几个常见尺度。以下公式默认自然单位；涉及 fm 与 GeV 时，使用

$$
\hbar c=0.1973269804\;\mathrm{GeV\,fm}.
$$

| 量 | 定义或含义 | 是否直接等于最终 PDF 的能标 |
|---|---|---|
| 格点紫外截止 | $a^{-1}$ | 否；它是正则化尺度 |
| Flow 分辨率 | $r_f=\sqrt{8t_f}$，$Q_{\mathrm{flow}}=r_f^{-1}$ | 否；它定义流化半径及分辨率 |
| Flow 转换自然尺度 | 当前对数约定下的 $\mu_t=e^{-\gamma_E/2}/\sqrt{2t_f}$ | 否；它是方便计算转换系数的尺度 |
| 算符重整化尺度 | $\mu_R$ | 指定 MSbar 非局域算符矩阵元的尺度 |
| PDF 因子化尺度 | $\mu_F$ | 指定最终 $f_{\overline{\mathrm{MS}}}(x,\mu_F)$ 的尺度 |
| 微扰耦合／计算尺度 | $\mu_{\mathrm{pert}}$ | 控制耦合与有限阶展开，可为估计截断误差而变化 |
| 强子动量 | $P_z$ | 否；它控制 LaMET 的大动量展开 |
| Hybrid 拼接距离 | $z_s$ | 否；它是 hybrid 方案参数 |

在实际实现中，经常把几个不同含义的尺度取成同一个数，例如

$$
\mu_R=\mu_F=\mu_{\mathrm{pert}}=2\;\mathrm{GeV},
$$

但它们的物理角色仍然不同。审计结果时，应首先确认一个配置项到底控制哪一步，而不是仅看到 `mu_GeV` 就认定最终 PDF 已在该尺度完成匹配。

对于 gradient flow，四维流方程将原始场映射为流化场。在弱场或自由场的示意下，其动量模式包含阻尼因子

$$
B_\mu(t_f,p)\sim e^{-t_fp^2}A_\mu(p),
\qquad
\chi(t_f,p)\sim e^{-t_fp^2}\psi(p).
$$

因此正 flow time 抑制高动量涨落，并引入平滑半径

$$
r_f=\sqrt{8t_f}.
$$

以 fm 记录半径时，对应的分辨率尺度为

$$
Q_{\mathrm{flow}}[\mathrm{GeV}]
=\frac{\hbar c}{r_f[\mathrm{fm}]}.
$$

这个尺度描述有限-flow 方案的分辨率，不能据此直接把一个有限-flow 裸矩阵元命名为该尺度下的 MSbar 矩阵元。

对已按要求归一化的 flowed 算符，小流时间展开给出它与目标重整化算符的关系。对固定非零分离、可乘法转换的直线算符，可写成

$$
\widehat h_{\mathrm{GF}}(z,t_f,P_z)
=e^{\delta m_{\mathcal R}(t_f)|z|}
\left[
C_{\mathcal O}(t_f,\mu_R)
h_{\overline{\mathrm{MS}}}(z,P_z;\mu_R)
+\Delta_{\mathcal O}^{\mathrm{SFTE}}(z,t_f,P_z)
\right].
$$

其中帽号表示已满足相应流场归一化约定；$\mathcal R$ 表示色表示，quark 直线为基础表示，gluon 为伴随表示；$\delta m_{\mathcal R}$ 是按此公式定义符号的线质量，$C_{\mathcal O}$ 是有限转换系数；$\Delta_{\mathcal O}^{\mathrm{SFTE}}$ 是小流展开中未保留的高维／有限-flow 项。存在算符混合时，单个系数应推广为系数矩阵，不能用一个标量因子替代整个混合问题。

完成所需转换后，目标是

$$
h_{\mathrm{conv}}(z,t_f,P_z;\mu_R)
=C_{\mathcal O}^{-1}(t_f,\mu_R)
e^{-\delta m_{\mathcal R}(t_f)|z|}
\widehat h_{\mathrm{GF}}(z,t_f,P_z)
=h_{\overline{\mathrm{MS}}}(z,P_z;\mu_R)
+\Delta_{\mathrm{flow}}.
$$

右端的 $\mu_R$ 由转换中选定的目标尺度决定。不同的 $t_f$ 可以统一转换到同一个 $\mu_R$，然后检查及去除剩余的 $\Delta_{\mathrm{flow}}$。

当前所用有限系数包含

$$
L_t=\ln(2\mu_R^2t_f)+\gamma_E.
$$

令这个对数为零，可定义自然尺度

$$
\mu_t=\frac{e^{-\gamma_E/2}}{\sqrt{2t_f}}
=\frac{2e^{-\gamma_E/2}}{r_f}.
$$

以 fm 为长度单位时，公式为

$$
\mu_t[\mathrm{GeV}]
=\frac{2e^{-\gamma_E/2}\hbar c}{r_f[\mathrm{fm}]}.
$$

例如

$$
r_f=0.1794\;\mathrm{fm}
\quad\Longrightarrow\quad
Q_{\mathrm{flow}}\simeq1.10\;\mathrm{GeV},
\qquad
\mu_t\simeq1.65\;\mathrm{GeV}.
$$

自然尺度只是在该对数约定下减小显式对数的选择，不能消除所有有限常数或保证微扰级数可靠。可以在自然尺度计算后用 RG 演化到固定目标尺度，也可以直接在固定目标尺度做转换，并检查对数与高阶项。仅将耦合改为自然尺度处的耦合，而不包含相应异常维数演化，不等于完成 RGR。

减小 flow time 时，自然尺度升高，但最终输出仍然可以保持

$$
\mu_R=2\;\mathrm{GeV}.
$$

因此，取小流时间极限不等于把最终 PDF 的因子化尺度取到无穷大。

对于 hybrid renormalization，首先得到的是登记在 hybrid 方案中的矩阵元。以局域归一化的输入形状 $S_P(z)$ 和零动量参考 $S_0(z)$ 为例，当前已有分支采用的结构是

$$
H_{\mathrm{hyb}}(z,P_z)=
\begin{cases}
\displaystyle\frac{S_P(z)}{S_0(z)},& |z|\le z_s,\\[6pt]
\displaystyle\frac{Z_R(z_s,a)}{S_0(z_s)}
\frac{S_P(z)}{Z_R(z,a)},& |z|>z_s.
\end{cases}
$$

这里按照该实现的定义，长距离段以除以 $Z_R$ 完成重整化；其他文献可能将重整化常数定义为其倒数。不同记号的指数符号必须从具体定义核对。拼接因子保证边界连续，但连续本身不证明重整化方案或拟合质量正确。

这一阶段的结果依赖 $z_s$、线质量及有限残余质量处方、短距离参考和归一化约定。它还不是具有唯一 MSbar 标签的光锥 PDF。特别是

$$
\mu_F\not\equiv\frac{1}{z_s}
$$

不是一个由 hybrid 定义推出的尺度关系；不能把 $z_s$ 的倒数直接当作最终能标。若短距离 OPE 系数带有对数

$$
L_z=\ln\!\left(\frac{z^2\mu_R^2e^{2\gamma_E}}{4}\right),
$$

常用自然尺度为

$$
\mu_z=\frac{2e^{-\gamma_E}}{|z|}.
$$

这个尺度用于短距离系数或 RGR。若用这些系数拟合有限残余质量 $m_0$，还须登记目标重整化尺度、微扰阶数及重求和／重整子处方。不能保留旧的有限质量参数，却任意改变其短距离定义后声称得到另一个一致方案。

完成 Fourier 变换后，hybrid quasi 分布与光锥 PDF 的匹配结构可示意为

$$
\widetilde f_i^{\mathrm{hyb}}(x,P_z;\eta_{\mathrm{hyb}})
=\sum_j\int\frac{dy}{|y|}\,
C_{ij}^{\mathrm{hyb}}\!\left(
\frac{x}{y},\frac{\mu_F}{|y|P_z},
|y|P_zz_s;\eta_{\mathrm{hyb}},\mu_{\mathrm{pert}}
\right)
f_j^{\overline{\mathrm{MS}}}(y,\mu_F)
+\text{幂修正}.
$$

$\eta_{\mathrm{hyb}}$ 表示须登记的其他 hybrid 方案参数。核必须与输入矩阵元的归一化和线质量处方一致。matching 核中指定的 $\mu_F$ 才确定最终光锥 PDF 的因子化尺度。改变 $z_s$ 时，需要同时改变矩阵元及其 matching 核；理想的完整计算给出一致的最终 PDF，有限阶或有限动量下保留方案与幂修正依赖。

严格说来，$\mu_R$、$\mu_F$ 与 $\mu_{\mathrm{pert}}$ 可以独立选择，并由一致的转换与演化联系。实际代码常统一取值，但应分别说明：第一项是 MSbar 算符尺度，第二项是目标 PDF 尺度，第三项用于有限阶微扰计算。将几个目标尺度处的结果直接包络，并不自动得到固定目标尺度下的微扰截断误差。

要比较不同 PDF 能标，应使用 DGLAP 演化。例如 singlet quark 与 gluon 满足

$$
\frac{\partial}{\partial\ln\mu_F^2}
\begin{pmatrix}\Sigma\\g\end{pmatrix}
=\begin{pmatrix}P_{qq}&P_{qg}\\P_{gq}&P_{gg}\end{pmatrix}
\otimes\begin{pmatrix}\Sigma\\g\end{pmatrix}.
$$

非单态 quark 采用相应的非单态演化。gluon 的绝对归一化若需要动量份额，该输入也必须位于同一目标尺度：

$$
\langle x\rangle_g(\mu_F)
=\int_0^1dx\,xg(x,\mu_F).
$$

将局域点规范为一并不能消除非局域形状的尺度依赖，也不能替代 gluon 动量份额及 singlet 混合。

为什么还需要 small-flow-time 分析：**有限 flow 改变了所测算符，即使该算符在正 flow time 是有限的，它也不等于未流化的目标算符。** 平滑会改变场强、费米子端点和 Wilson line，并引入高维算符贡献。在固定非零分离的区域，小流展开可示意为

$$
e^{-\delta m_{\mathcal R}(t_f)|z|}
\widehat O_{\mathrm{GF}}(z,t_f)
=C_0(t_f,\mu_R)O_{\overline{\mathrm{MS}}}(z,\mu_R)
+t_f\sum_k C_k(t_f,\mu_R)O_k^{\mathrm{higher}}(z,\mu_R)
+\cdots.
$$

此处高维结构与允许的次领幂依赖算符及对称性；示意式不承诺所有通道都只需单个线性项。对目标矩阵元而言，需要考察的典型无量纲参数包括

$$
\frac{t_f}{z^2},\qquad t_fP_z^2,\qquad
t_f\Lambda_{\mathrm{QCD}}^2.
$$

在完成线质量和有限转换后，改变 flow time 可以检查剩余流化效应，选择有理论依据的窗口，并用外推或受控的幂修正拟合提取未流化目标。这是外推的主要作用。它不是在原始数据上任意拟合一条直线，也不是将 flow time 设为零就自动完成重整化。

外推是否必须，取决于目标。若报告的就是固定正 flow time 的 GF 方案矩阵元，应明确报告该物理 flow time；不必为了让它成为一个定义良好的有限-flow 量而强行外推。若目标是未流化的 MSbar 算符，并且使用了只保留小流展开领项的转换，则必须控制剩余流化修正：可以进行受控的零-flow 外推、联合幂修正拟合，或在可信的小流窗口内证明剩余误差足够小并赋予误差。若未来采用保留完整有限-flow 依赖的匹配理论，也可直接匹配有限-flow 对象，但那不是当前渐近端点系数已经提供的完整信息。

格点上还存在相反方向的限制：flow time 过小会重新暴露格点截止效应。需要寻找同时满足

$$
a\ll r_f=\sqrt{8t_f}\ll |z|,
\qquad
\sqrt{t_f}|P_z|\ll1,
\qquad
t_f\Lambda_{\mathrm{QCD}}^2\ll1
$$

的可检验区域。在离散误差主要从平方格距开始的示意模型中，转换后的量可能具有

$$
h_{\mathrm{conv}}(a,t_f,z,P_z;\mu_R)
=h_{\overline{\mathrm{MS}}}(z,P_z;\mu_R)
+A_z\frac{t_f}{z^2}
+A_Pt_fP_z^2
+A_\Lambda t_f\Lambda_{\mathrm{QCD}}^2
+A_a\frac{a^2}{t_f}
+B_a\frac{a^2}{z^2}
+\cdots.
$$

系数可以带对数，实际允许的离散项与高维项要按作用量、流方程和算符的改进情况确定，不能不经检验假定只有上述项。固定一个格距时，将 flow time 不断减小，会放大其中与格点截止相关的贡献。因此，固定格距的零-flow 直线截距不自动构成受控结果。

一种稳妥的目标极限写法是

$$
h_{\overline{\mathrm{MS}}}(z,P_z;\mu_R)
=\lim_{t_f\to0}
\left[
\mathcal Z_{{\mathrm{GF}}\to\overline{\mathrm{MS}}}
(z,t_f;\mu_R)
\lim_{a\to0}
h_{\mathrm{GF}}^{\mathrm{norm}}(a,z,t_f,P_z)
\right],
\qquad z\ne0.
$$

连续极限在固定物理分离、flow time 和动量下进行；$\mathcal Z$ 代表该目标真正所需的一整套转换，必要时是矩阵，并非仅线性发散的一个指数。也可作有明确理论结构且有足够数据约束的联合极限拟合。该式是受控转换的目标关系，不允许把有限阶系数无条件推到任意小 flow time：固定目标尺度时的巨大对数、缺失高阶和线质量处方仍须通过 RG 或可信窗口处理。

不同 ensemble 若采用相同格点 flow 参数

$$
\tau=\frac{t_f}{a^2},
$$

其物理 flow time 一般不同。做连续极限应对齐

$$
t_f^{\mathrm{phys}}=\tau a^2,
\qquad
r_f^{\mathrm{phys}}=a\sqrt{8\tau},
$$

而不是只对齐无量纲参数 $\tau$。

“small-flow-time 的结果是否已经减除了所有紫外发散”需要按处理阶段回答。在重整化的四维 QCD 中，适当的正 flow time 规范场观测量具有流化带来的紫外有限性；含 flowed fermion 的复合算符还需相应场归一化。对我们研究的有限长度直线算符，正 flow time 提供了流化正则化，但这不能解释为 MSbar 转换、绝对归一化和任意算符混合均已自动完成。

随着 flow time 变小，流化正则化的短距离结构再次显现。按上文符号约定，典型线质量表现为

$$
\delta m_{\mathcal R}(t_f)
\sim-\frac{\alpha_s\,c_{\mathcal R}}{\sqrt{t_f}}
+\cdots,
$$

还伴随端点系数的对数。未经转换的矩阵元可能被指数压到零，也可能在其他约定或处理中出现增长；一个看似有限的零值不能说明已经重整化。线性发散处理和对数／有限项转换是不同步骤。

当前 Wilson-line 候选采用

$$
\ln W_{\mathcal R}
=-d_{\mathcal R}\frac{|z|}{r_f}+\text{有限项},
\qquad
h_{\mathrm{sub}}
=\exp\!\left(d_{\mathcal R}\frac{|z|}{r_f}\right)
h_{\mathrm{GF}}.
$$

这一步只按给定候选去掉线质量指数；它不决定完整的对数重整化或有限残余质量方案。若该非微扰线质量与目标 MSbar 方案之间还存在有限残余质量 $m_0$，须另行确定一致的转换。不能把 Landau 规范 Wilson line 提取的所有有限贡献都默认解释成目标规范不变算符的 MSbar 线质量，也不能在非微扰指数外再叠加覆盖同一贡献的微扰线质量指数。

对 quark 的绝对双线性矩阵元，还须满足 flowed fermion 归一化约定，例如

$$
\mathring\chi=Z_\chi^{1/2}\chi.
$$

同 flow 的局域归一化比值可以消去共同因子，但不会自动给出绝对矩阵元。若

$$
H_q(z,t_f)=\frac{h_q(z,t_f)}{h_q(0,t_f)},
$$

则当前 vector 比值的转换涉及

$$
D_V(t_f,\mu_R)=\frac{\zeta_V(t_f,\mu_R)}{C_\psi(t_f,\mu_R)},
$$

而不是只转换非局域分子。局域分母也有自己的转换。gluon 没有费米子的 $Z_\chi$，但其局域锚点、绝对归一化及需要时的 singlet quark–gluon 混合仍须处理。非零分离的小流系数不能直接套到

$$
z=0
$$

的局域接触极限上；局域算符有独立的转换和混合问题。

| 已完成的处理 | 可作的判断 | 仍不能据此宣称 |
|---|---|---|
| 正 flow time 的原始 $C_3/C_2$ 或 self ratio | 有限-flow 裸矩阵元或局域归一化形状 | 已处于 MSbar，或紫外处理全部完成 |
| 仅施加 Wilson-line 指数 | 选定线质量处方下的减除结果 | 对数、有限残余质量、归一化、混合已全部处理 |
| 加有限阶 GF 系数后作固定格距外推 | 检查流化依赖的条件诊断 | 已控制连续极限或所有紫外问题 |
| 一致归一化、线质量、有限转换、必要混合及受控极限 | 所声明方案和尺度下的重整化非局域矩阵元，精度受剩余误差限制 | 已完成光锥 PDF matching 或消除所有物理系统误差 |

因此，正确完成重整化与受控小流极限后，可以说“在所声明的算符基、方案和精度下，相关紫外结构已经处理，得到重整化矩阵元”。不能仅凭 small-flow-time 外推就说“所有紫外发散已经减除”。重整化完成后仍有尺度依赖，这是异常维数导致的正常 RG 性质；紫外有限并不意味着尺度无关。

此外，重整化的非局域矩阵元还不等于最终光锥 PDF。它仍受基态提取、有限体积、质量调谐、大动量幂修正、有限坐标覆盖和 Fourier 重建等因素影响；需再完成与方案一致的 LaMET matching。GF 有限转换的 NNLO 与准 PDF 到光锥 PDF 卷积核的 NNLO 是两个不同的计算层次。

将这些区分应用到当前三组 ensemble，本次配置选择的是

$$
\mu_R\in\{1,2,4\}\;\mathrm{GeV},
\qquad
\mu_R^{\mathrm{central}}=2\;\mathrm{GeV}.
$$

它们是 GF 有限转换的目标算符尺度分支。目前没有把所有分支演化到共同尺度，亦未完成完整 LaMET 链，不能只因为中心配置写了该数值就称已获得对应能标下的物理 PDF。当前 Wilson-line 线质量候选未通过模型质量检验；有限残余质量、绝对归一化及 gluon 局域转换／混合也仍存在未完成项。因此当前 flow 外推截距继续标记为诊断。

本次登记的保守尺度窗为

$$
\frac{a}{r_f}\le0.5,
\qquad
\frac{r_f}{|z|}\le0.5,
\qquad
\sqrt{t_f}|P_z|\le0.5,
\qquad
t_f\Lambda^2\le0.1,
\qquad
\Lambda=0.3\;\mathrm{GeV}.
$$

这些是本次检验标准，不是普适的严格边界。按至少四个合格 flow 点的要求，当前全部已用非零动量均没有尺度窗内的 flow 外推；部分零动量有条件诊断截距。使用全部 flow 点的拟合提供敏感性检查，不能替代尺度窗内的受控结果。

这个限制不能仅靠增加大 flow time 点解决。由上述前两个涉及格距与动量的约束可推出

$$
|P_z|\le\frac{\hbar c}{\sqrt{2}\,a}
$$

（这里格距以 fm 表示）。对当前粗格距

$$
a=0.105\;\mathrm{fm},
\qquad |P_z|_{\max}\simeq1.33\;\mathrm{GeV}.
$$

因此，在这套保守窗下，较大的已用动量与格点伪影抑制条件本身存在冲突。后续应优先结合更细格距、共同物理半径和可检验的有限-flow／离散修正模型规划分析，不能为得到外推截距而静默放宽条件。

实际报告尺度时，可依次记录：算符及归一化定义；正则化和中间方案；物理 flow time 或 hybrid 拼接参数；线质量和有限残余质量处方；目标算符尺度；目标 PDF 尺度；微扰阶数与 RG 精度；已完成的连续极限和小流分析；尚未完成的混合、绝对归一化及 matching。对于本批结果，推荐描述为“中心目标算符尺度下，施加候选线质量与 NNLO 有限 GF 系数后的诊断矩阵元”，并附上当前质量限制。

上述四维 flow 的结论不能直接套用于三维流化或 HYP smearing。三维流化不抑制所有四维高能模式；HYP 也不是同一流方案。它们需要各自的重整化及转换依据。

本文依据的文献与本地记录如下。文献定义与具体算符应一起核对，不能仅据“同为 gradient flow”移植系数。

- [Brambilla–Wang，arXiv:2312.05032](https://arxiv.org/abs/2312.05032)：直线 flowed 算符、线质量与有限小流转换；应采用修正后的约定。
- [arXiv:2610.40225v1](https://arxiv.org/abs/2610.40225v1)：两圈有限 GF 算符转换。它不单独提供本项目线质量的确定或完整光锥 PDF 的 NNLO matching。
- [arXiv:2311.16799v2](https://arxiv.org/abs/2311.16799v2)：quark 局域 flowed 双线性系数，用于当前局域归一化比值的转换。
- [arXiv:2204.08343v4](https://arxiv.org/abs/2204.08343v4)：本地 quark hybrid matching 分支所登记的核与方案约定。
- [arXiv:2510.26425v2](https://arxiv.org/abs/2510.26425v2)：本地 gluon hybrid 分支参考，须核对张量组合和混合适用性。
- [当前 self-ratio／GF 分析报告](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/ANALYSIS.zh-CN.md)及[配置](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/selfratio_quark_gluon_nnlo_20261116/config.json)。
- [两圈 GF 系数评估](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching_review_2610_40225_20261115/ASSESSMENT.zh-CN.md)。
- [Quark hybrid matching 配置](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Matching/Normalized_Hybrid_20261027/config.json)：同时登记了目标 PDF 尺度和拼接距离。
- [Gluon hybrid 分段定义](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/Hybrid_T_4ens_Lambda_Quasi_Reviewed_20261108/README.md)：该分支尚属 quasi 诊断，不能仅因上游使用短距离系数就称为 matched PDF。
