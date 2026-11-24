# Baryon DA：arXiv:2606.30387v2 阅读总结与 quark PDF 计算对照

日期：2026-10-30。本文档为文献阅读和代码/产品审计；未重新拟合、未改变生产配置。

## 0. 阅读范围、来源与结论

主文献：Lattice Parton Collaboration，Mu-Hua Zhang 等，*Baryon Light-Cone Distribution Amplitudes from Lattice QCD: Formalism, Renormalization, Extrapolation, and Matching*，arXiv:2606.30387v2。下载的 HTML 显示版本日期为 **2026-07-20**；PDF 共 62 页。本次阅读覆盖主要正文，并针对重整化、渐近外推和 matching 核对相关附录。没有审计作者未公开在本地的数值程序。

- [主论文 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/references/2606.30387v2.pdf)
- [保留 LaTeX 的正文提取](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/references/2606.30387v2.reading.txt)
- [PDF 文本提取](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/references/2606.30387v2.pdftotext.txt)
- [第 13 页公式图像](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/references/2606.30387v2.page13.png)
- [直接讨论 quark PDF 长距离展开的理论来源：2601.12189v1](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/references/2601.12189v1.html)
- [该理论来源的文本提取](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/references/2601.12189v1.reading.txt)
- [来源与本地证据 SHA256](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Literature/arXiv_2606_30387v2_20261030/manifest.json)

**核心判断：**这篇文章支持我们“短距 ratio + 长距 self renormalization + 归一化 shape matching 后恢复局域荷”的总体方向。最值得新增的分析是：分阶段约束 UV 参数、引入适合 quark PDF 的有限动量 LA/NLA 长距离模型、建立完整的尾部/Fourier/尺度误差预算。我们当前优先级最高的问题仍是零动量参考态的基态隔离与窗口稳定性。

文章研究的是重子 LCDA：三夸克、真空到重子的矩阵元、两个独立坐标。我们计算的是核子 quark PDF：双夸克、核子前向矩阵元、一个独立坐标。文章的 $V/A/T$ 不能直接替换成我们的 unpolarized/helicity/transversity；其微扰系数、Wilson-line 几何和二维 matching 核均不能照搬。

## 1. 文章计算什么

### 1.1 对象与算符

光锥三夸克算符通过 Wilson lines 保持规范不变，结构概括为

$$
\langle0|\epsilon^{ijk}[Uq]^i_\alpha(\xi_1n)
[Ug]^j_\beta(\xi_2n)[Uh]^k_\gamma(\xi_3n)|B(P)\rangle.
$$

其中 $q,g,h$ 在该文中都是夸克 flavor 的占位符；$g$ 在这里不是 gluon field。leading twist 下分解为 $V,A,T$ 三类结构。动量分数满足

$$
x_1+x_2+x_3=1,
$$

因此有两个独立的 $x$，空间准分布依赖两个独立的 $z$。LaMET 用等时空间算符在大动量下得到 quasi-DA，再通过 matching 得到 LCDA。

对于 $\Lambda$ baryon，$A$ 在交换前两个坐标时对称，局域极限非零；$V,T$ 反对称且局域极限为零。其他八重态重子的对称结构不同。文章先明确交换对称和 Hermiticity，再利用它们重建完整二维平面。

### 1.2 数值示范的范围

文章建立了 $V/A/T$ 的形式体系，完整的端到端示范主要针对 **$\Lambda$ 的 $A$ LCDA**。不应将其理解为所有重子所有通道均完成了同精度的完整数值验证。完整 $\Lambda$ LCDA 物理结果另见其 companion Letter，arXiv:2606.29597。

## 2. 格点设置与裸矩阵元提取

### 2.1 七套 ensemble，Table II

| ensemble | $a$ [fm] | $m_\pi$ [MeV] | $N_{\rm cfg}$ | 非零 $P^z$ [GeV] |
|---|---:|---:|---:|---|
| C24P29 | 0.1052 | 292.3 | 864 | 1.96, 2.45, 2.94 |
| C32P23 | 0.1052 | 227.9 | 954 | 1.84, 2.21, 2.57, 2.94 |
| C48P14 | 0.1052 | 136.4 | 302 | 1.96, 2.45, 2.94 |
| F32P30 | 0.0775 | 300.4 | 777 | 2.00, 2.49, 2.99 |
| F32P21 | 0.0775 | 210.3 | 459 | 2.00, 2.49, 2.99 |
| G36P29 | 0.0689 | 297.2 | 656 | 2.00, 2.50, 3.00 |
| H48P32 | 0.0520 | 316.6 | 550 | 1.98, 2.48, 2.98 |

各套也有零动量参考。作者使用 momentum smearing、一步 HYP 和 kinematically enhanced source。增加 $\gamma^t$ 的源算符改善高动量重子的重叠与信噪比；引用的对照在约 2.5 GeV 有约两倍信噪改善。文章还比较了不同 spin projector，选用与 leading-twist 光锥结构对应的组合。

可借鉴的是“以算符重叠和投影改善高动量数据，并显式检验不同投影间的有限动量差异”。我们使用 distillation 和 C3/C2；修改源或 projector 需回到算符、收缩和运动学归一化重新推导，不是在现有数值上直接乘一个因子。文章一步 HYP 与我们的五步空间 HYP 不同。

### 2.2 非局域/局域两点函数比值与激发态

他们从非局域 sink 的重子两点函数提取真空到重子的矩阵元。同源、同投影的非局域/局域 C2 比值消去基态源重叠及共同指数。对局域极限为零的结构，用非零的对称结构作参考，而不是除以自身的零。

文章 Eq. (34) 使用截断的两态比值形式

$$
R(\tau;z_1,z_2,P)
=\widehat M(z_1,z_2;P)
\left[1+\Delta C(z_1,z_2;P)e^{-\Delta E\tau}\right].
$$

该式是保留 leading excited correction 的表达，不是一般两态 C2 比值的精确有理式。它也不是我们的 C3/C2 模型。我们有源汇间隔 $T$ 和插入位置 $\tau$，典型修正来自

$$
R(T,\tau,z)=h_{00}(z)
+A(z)e^{-\Delta E\tau}
+B(z)e^{-\Delta E(T-\tau)}
+C(z)e^{-\Delta E T}+\cdots,
$$

精确两态形式还包含 C2 分母；self ratio 则会引入局域与非局域振幅的共同关联。需要保留两侧跃迁，结合 C2 约束能隙并检查参数可辨识性。

### 2.3 窗口平均，Eq. (35)–(36)

作者采用

$$
w_\kappa\propto\exp\left[-\frac{\chi_\kappa^2}{2}
-\frac{\tau_{\min}^{(\kappa)}}a\right],\qquad
\sum_\kappa w_\kappa=1,
$$

$$
\overline M=\sum_\kappa w_\kappa M_\kappa,
$$

$$
\sigma^2=\sum_\kappa w_\kappa\sigma_\kappa^2
+\sum_\kappa w_\kappa(M_\kappa-\overline M)^2.
$$

第二项是窗口间漂移。对于我们，可以借鉴“所有窗口预先声明、窗口内误差与窗口漂移分开保存”的思想。上述具体权重不能机械用于任意 C3 窗口：裁点数、参数数、插入位置 mask 与时间跨度的计数不同，跨格距的裁点惩罚也要检查。不同窗口高度相关，方法比较和下游传播仍须使用共同 bootstrap。

不要将窗口混合方差与已由副本传播的同一窗口差异重复累加；要声明统计量是固定权重估计量、随机权重估计量，还是包含模型不确定性的混合分布。窗口平均本身不能修复错误的谱模型。

### 2.4 色散关系，Eq. (38)–(39)

$$
E^2=m_{\rm ens}^2+c_0P_z^2+c_1a^2P_z^4,
$$

$$
c_0=1.074(25),\qquad c_1=-0.061(19),\qquad
\chi^2/{\rm dof}=0.251.
$$

$c_0-1$ 约为 3 个所报统计标准差，故应表述为存在可见的有限格距/动量效应，而不能仅凭低 $\chi^2$ 说严格符合连续色散关系。我们的能量和 boosted matrix elements 也应检查 $aP$ 效应。

## 3. Hybrid renormalization：最贴近当前分析的部分

### 3.1 短程 ratio、长程 self

短距用零动量同几何参考消去线性发散；长距若继续除以该参考，会引入其长距离强子结构，因此改用只提取 UV 和约定有限质量项的 self-renormalization。

文章的有效长度为 Eq. (42)

$$
\widetilde z=
\begin{cases}
\max(|z_1|,|z_2|),&z_1z_2\geq0,\\
|z_1-z_2|,&z_1z_2<0.
\end{cases}
$$

这是三夸克颜色/Wilson-line 几何的结果。我们的 bilinear 仍用单条线长 $|z|$。

### 3.2 其参数模型与两步拟合

其 Eq. (41) 的结构可写为

$$
\ln\widehat M_B
=\frac{k\widetilde z}{a\ln(a\Lambda_{\rm latt})}
+g(z_1,z_2)
+L(a,\mu;\Lambda_{\rm latt},\Lambda_{\overline{\rm MS}})
+\ln\!\left[1+\frac d{\ln(a\Lambda_{\rm latt})}\right]
+f(z_1,z_2)a^2.
$$

其中 $L$ 代表 cutoff 对数与连续微扰方案转换，原文打印的对数符号问题另见第 9 节。这里区分 $\Lambda_{\rm latt}$ 与用于连续 running coupling 的 $\Lambda_{\overline{\rm MS}}$。

| 项 | 作用 | 迁移时的限制 |
|---|---|---|
| $k/[a\ln(a\Lambda_{\rm latt})]$ | Wilson-line 的 cutoff 线性发散 | 依赖离散作用量、link 定义、涂抹及参数约定 |
| $g$ | 有限强子结构 | 不应整体放入 UV 因子 |
| $m_0$ | 短距微扰转换/残余质量约定 | 不是独立可观测的强子质量，也不是纯 UV 发散 |
| $d$ | 剩余 cutoff/高阶对数的有效修正 | 依赖微扰结构和模型，不能当作精确高阶系数 |
| $f a^2$ | 有限格距效应模型 | 有限格距数和自由 $f(z)$ 可造成退化 |

作者用四套 $m_\pi\simeq300$ MeV、不同格距的 ensemble 拟合，另三套不同 pion mass 作检验。步骤为：

1. 插值到公共物理距离网格 $a_0=0.05$ fm。
2. 中长距区域拟合 $k$，固定 $\Lambda_{\rm latt}=0.2$ GeV。
3. 短距再用对应结构的 perturbative reference 提取 $m_0,d$；选 $z_1z_2<0$ 避免 $z_1=z_2$ 的短距奇异结构。
4. 在多个微扰尺度重复全链路。

在 $\mu=2$ GeV，作者给出：

| reference | $k$ | $\chi^2/{\rm dof}$（$k$ 拟合） | $m_0$ [GeV] | $d$ |
|---|---:|---:|---:|---:|
| $\Lambda$ baryon $A$ | 0.7778(22) | 0.293 | 0.2159(35) | −0.1718(28) |
| proton $T$ | 0.7810(22) | 0.329 | 0.2242(47) | −0.1222(35) |

近似相同的 $k$ 支持相同 line 几何的 UV 自能对外态和 Dirac 结构近似通用；有限 perturbative coefficients 不同，$d$ 不应强制相同。$m_0$ 的跨通道关系也取决于定义、局域流归一化和所选微扰精度。上述数字不用于我们的五步 HYP 数据。

两步拟合有助于控制 $k,\Lambda,m_0,d,f$ 的退化，但“固定 $\Lambda$”仍是条件模型选择，需扫描其依赖并传递第一步的不确定度。不是固定某个数字后误差就消失。

### 3.3 对我们重整化的直接含义

现有 shape 定义是

$$
S^f_\Gamma(P,z)=\frac{h^f_{\Gamma,B}(P,z)}{h^f_{\Gamma,B}(P,0)},
\qquad H_0(z)=S^{u-d}_\Gamma(0,z),
$$

$$
H_{\rm hyb}^f(P,z)=
\begin{cases}
S^f_\Gamma(P,z)/H_0(z),&0<z\le z_s,\\
S^f_\Gamma(P,z)\,Z_R(z_s)/[H_0(z_s)Z_R(z)],&z>z_s.
\end{cases}
$$

这一方向与文章相容，边界按定义连续。当前 $z_s=0.25$ fm，并已保存 0.20/0.30 fm 对照。保留该方向，优先改进 reference 和参数辨识。

当前公共网格短距只有 0.15、0.25 fm 两点；最粗格距约 0.105 fm，分别只有约 1.43、2.38 个格点。可供区分 $m_0,d$、离散项和微扰系数的短距信息有限。文章写出的 $a\ll z<1/\mu$ 也不能当作我们在固定 $\mu=2$ GeV 已满足的严格层级：$1/\mu\simeq0.099$ fm。应比较 $a/z$、自然尺度 $2e^{-\gamma_E}/z$、固定阶与 RGR 的稳定性，必要时引入更细格距或 step scaling。

当前模型使用同一可拟合 $\Lambda$ 同时进入 cutoff 对数和归一化分母；作者区分 $\Lambda_{\rm latt}$ 与 $\Lambda_{\overline{\rm MS}}$。这是应独立扫描的模型差异，不能未经重新拟合就宣布现有代码必错。更不能用文章 $N_f=4$ 的 coupling 参数直接替换我们 $N_f=3$ 配置。

### 3.4 与 Wilson loop 线质量的联系

我们采用的衰减约定是

$$
\ln Z_R\supset\frac{k|z|}{a\ln(a\Lambda_{\rm latt})}
=-\delta m_{\rm UV}(a)|z|,
$$

因此

$$
a\delta m_{\rm UV}(a)=-\frac{k}{\ln(a\Lambda_{\rm latt})}.
$$

Wilson-loop Cornell 常数的一半包含有限势零点与提取窗口的影响，不能无条件作为纯 $\delta m_{\rm UV}$。必须协调有限质量方案，避免与 $m_0$ 重复或抵消。文章没有建立“某一 Cornell 常数直接等于这里的 $k$”的结论。

本地固定 $a\delta m=0.1874$ 并重拟合其余参数的测试，得到 $\Lambda=0.0090219$ GeV、$k=1.03004$、$m_0=0.003805$ GeV、$\chi^2/{\rm dof}=15.7763$、$Q=1.228\times10^{-36}$。这说明当前 reference/model 下该约束未通过质量门，不能作为已验证重整化方案；也不能单凭它否定 Wilson-loop 方法本身。

## 4. 最值得尝试的改进：有限动量长距离 LA/NLA

### 4.1 主论文的新方法

文章第 V 节以 Euclidean correlator 的长距离谱/HQET 展开构造尾部。Eq. (53) 的一维原型为

$$
\sum_J e^{-\Omega_J|z|}
\left[(\mathcal H_1e^{-izP}+\mathcal H_2)
+\frac{\mathcal H'_1e^{-izP}+\mathcal H'_2}{|z|}+\cdots\right].
$$

这里用 $\Omega$ 重命名衰减能量，避免与 $\Lambda_{\rm latt}$ 混淆。原文用 binding energy 的记号。LA 保留 leading term，NLA 增加 $1/|z|$。允许的中间态由算符量子数决定；这些 binding energies 的数值仍须拟合，或在相同 residual-mass/renormalon 约定下独立测量。

重子 DA 有 $|z_1|,|z_2|,|z_1-z_2|$ 三个尺度，需要二维分区。作者先用真正长距离区域联合确定共同衰减能量，再逐边界线拟合短坐标依赖。中央保留原始数据；外侧噪声过大数据不进入 fit，拟合区域仍延伸超过 1 fm。$\Lambda A$ 中央结果采用 NLA，较弱的 $V/T$ 信号主要用 LA。LA–NLA 差作为长距离截断误差。

### 4.2 对 quark PDF 的正确迁移形式

**不能直接把重子 DA 的振荡项搬到 quark PDF。**为此核对其理论来源 Ji–Liu–Su，arXiv:2601.12189v1，Sec. 3.4、4.2.2 和 Appendix C。该来源直接给出了 nucleon quark PDF 的 LA/NLA：

$$
h^{\rm LA}_\Gamma(z,P)
=A_\Gamma(P)e^{i\phi_\Gamma(P)\operatorname{sgn}(z)}
 e^{-\Omega_\Gamma|z|},
$$

$$
h^{\rm NLA}_\Gamma(z,P)
=\left[A_\Gamma(P)e^{i\phi_\Gamma(P)\operatorname{sgn}(z)}
+\frac{A'_\Gamma(P)e^{i\phi'_\Gamma(P)\operatorname{sgn}(z)}}{|z|}\right]
 e^{-\Omega_\Gamma|z|}.
$$

unpolarized 见该文 Eq. (52)–(53) / (92)–(93)；helicity/transversity 见 Eq. (276)–(277)。这些公式只用于足够长距离，不延伸到 $z=0$。完整谱展开还允许额外、通常衰减更快且可能振荡的分支；上述 LA/NLA 是其准确度计数下的截断，不代表所有有限 $z$ 一定已由最轻分支主导。

对于轻夸克，理论来源将 unpolarized/transversity 的 leading decay 联系到 vector heavy-light meson，helicity 联系到 pseudoscalar heavy-light meson；允许态、binding energy 与有限质量约定有关。因此不预先强制三种极化共享一个数值，也不把物理 meson mass 直接固定成 $\Omega$。

上述形式自动满足

$$
h_\Gamma(-z,P)=h_\Gamma(z,P)^*.
$$

正 $z$ 也可以用复振幅写成 $(C+D/|z|)e^{-\Omega|z|}$，负 $z$ 用共轭重建。相位必须与本地 Fourier convention 对齐。

### 4.3 与我们现有尾部的差别

当前沿用 arXiv:2208.08008v2 Eq. (6) 的模型：

$$
H(\lambda)=\left[
A_1(\lambda/7)^{-a}e^{-i\pi a/2}
+A_2(\lambda/7)^{-b}e^{-i\lambda+i\pi b/2}
\right]e^{-\gamma(\lambda-7)}.
$$

已有 $\lambda_L=6,7,8$ 和纯幂律对照。它不能仅因新论文出现就被认定错误，但新增 LA/NLA 对照有明确理论动机，尤其当前有慢衰减和积分上限不收敛的组合。

由于

$$
\lambda=P^zz,\qquad
 e^{-\Omega|z|}=e^{-(\Omega/P^z)|\lambda|},
$$

跨动量应比较同一方案下的 $P^z\gamma$，不能将不同动量的 $\gamma$ 直接视为相同的物理能量。在量子数与方案确实共同的条件下，可考虑多动量共享 $\Omega$，每个动量保留独立复振幅。若更换重整化 residual-mass 方案，尾部衰减也要重新拟合。

### 4.4 建议的可评审测试

先以三个 ensemble 的 **unpolarized $u-d$** 为试点：

1. 固定同一份 hybrid 输入和原 1000 个配对副本。
2. 将现模型、LA、NLA 放入预先声明的比较，保存各自全部失败/边界状态。
3. 以物理 $z_L$ 扫描进入渐近区的稳定性，也报告对应 $\lambda_L$；不逐点挑最好看的窗口。
4. Re/Im 联合相关 fit，不独立挑选两部分的窗口。
5. 验证 NLA 的次领先项在 fit 起点受控。理论示例使用 $|A'|<|A|z_L/2$；该约束不是普适常数，需作合理变化。
6. 保存原始 fit/data mismatch，不能用平滑 blend 隐藏。
7. 保留每组数据在起点处的 $z_L/a$、$z_L$、$\lambda_L$、可用点数、协方差秩和参数边界率；信号不够时允许“无法辨识”，不能强制填成物理 fit。
8. 同时检查 Fourier 尾部贡献和匹配后变化，统计差使用配对副本。

当模型约束 $\Omega>0$ 且数据支持其可辨识性时，尾部可解析积到无穷，减少上限截断误差。按我们当前 convention

$$
q(x)=\frac1\pi\operatorname{Re}\int_0^\infty d\lambda\,H(\lambda)e^{ix\lambda},
$$

令 $H_{\rm tail}=(C+B/\lambda)e^{-\eta\lambda}$、$s=\eta-ix$，则本笔记推得

$$
q_{\rm tail}(x)=\frac1\pi\operatorname{Re}
\left[\frac{C e^{-s\lambda_L}}s+B E_1(s\lambda_L)\right],
\qquad\eta=\Omega/P^z>0.
$$

该式是按本地 Fourier 定义推导的辅助数值公式，不是将其冒充为主论文原式。它不能免除尾部模型、起点、方案和参数不确定度。

## 5. Matching：学习 scheme 一致性，保留 quark 专用核

文章 Eq. (65) 的结构为

$$
\mathcal C^{(1)}_{\rm hybrid}
=\big[\mathcal C^{(1)}_{\overline{\rm MS}}
+\delta\mathcal C^{(1)}_{H}\big]_{\oplus}.
$$

短程 ratio 改变 UV 对数结构，因此必须加 hybrid counterterm；二维 double-plus subtraction 负责虚项与归一化。文章将 counterterm 放到坐标空间计算，再处理其余 momentum-space matching。这是等价表象的数值选择，不应在坐标和动量中各加一次。

**本地 matching 已有 quark PDF hybrid counterterm。**

[Compare_YaoFei/Matching/Normalized_Hybrid_20261027/physics.py](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Matching/Normalized_Hybrid_20261027/physics.py) 的 `ratio`、`hybrid_correction`、`total` 对应 arXiv:2204.08343v4 Eqs. (35), (38), (39)，总核为

$$
R_{\rm total}^\Gamma(\xi,y)
=R_{\rm bare}^\Gamma(\xi,|y|P,\mu)
+\frac{c_\Gamma}{\pi(1-\xi)}
\operatorname{Si}[(1-\xi)|y|Pz_s],
$$

其中自然单位下理解 $z_s$，$c_\Gamma=3,3,4$。存在 full-axis plus subtraction 与正确 parton scale $|y|P$。本次没有发现“完全遗漏 hybrid counterterm”的问题，也不应叠加重子三夸克 DA 的二维核。

继续保持 $z_s$、通道、局域归一化、微扰阶数、$\mu$、Fourier phase 和 residual-mass 约定的全链路一致。不同 scheme 的尾部模型和核须配套。

## 6. 归一化与局域荷

文章将 coordinate-space shape 与 local normalization constants 分开处理，特别强调局域极限为零的反对称振幅不能自归一化。这与我们先 shape 后局域荷恢复的方向相容。

我们应继续使用

$$
q^f_{\Gamma,R}(x,\mu)
=g^f_{\Gamma,R}(\mu)\,\phi^f_{\Gamma,\rm matched}(x,\mu),
\qquad g^f_{\Gamma,R}=Z_\Gamma g^f_{\Gamma,B},
$$

并逐副本保留 shape、局域分母、外部 $Z$ 的相关性与不确定度。近零局域荷时，不能为了画归一化图而强制相除。

本地目前 C24/F32 的局域流常数有 arXiv:2310.00814v2 来源；L32x64 的 E32/$\beta=6.308$ 不在那篇表内，仍是 legacy $Z$、误差未决。文章不能补齐这一缺口。独立 $P=0$ 局域荷拟合只有 3/27 通过现有质量门；改善 shape 的 UV 去除不能自动改善绝对归一化。

## 7. 连续、物理质量和无限动量外推

### 7.1 作者的 combined fit，Eq. (72)

在每个 $(x_1,x_2)$ 点拟合

$$
\phi(a,m_\pi,P)=\phi_{\rm phys}
+\frac A{P^2}
+(m_\pi^2-m_{\pi,\rm phys}^2)B
+a^2(D_1+D_2P^2).
$$

中心策略同时处理三种极限；另做先连续/物理 pion mass、再无限动量的 sequential fit。两种结果的差作为 extrapolation systematic。$a^2P^2$ 与 $1/P^2$ 分别描述动量相关格点效应和 LaMET power correction，应避免混为一个项。

### 7.2 我们的数据约束

当前 authoritative pointers 指定：L24x72 保留 $P=3,4$；L32x64 保留 $P=3,4,5$；L32x96 保留 $P=3,4$。按现有格距，物理动量为

| ensemble | momentum index | $P^z$ [GeV] |
|---|---|---|
| L24x72 | 3, 4 | 1.4760, 1.9680 |
| L32x64 | 3, 4, 5 | 1.2958, 1.7278, 2.1597 |
| L32x96 | 3, 4 | 1.4998, 1.9997 |

这些由 $P^z=2\pi n\hbar c/(N_sa)$ 计算。相同动量编号不代表不同 ensemble 有相同物理动量。我们的高动量覆盖比文章约 2–3 GeV 的范围低。

每个 channel/flavor 只有 7 个独立 ensemble–momentum 组合。若直接采用上面 5 参数模型，名义上只剩 2 个数据自由度；$a$ 与 $m_\pi$ 的变化也没有作者那样的独立设计。1000 bootstrap 不能增加独立 ensemble 的个数。

可先做受限模型、固定项/去项敏感性和 leave-one-ensemble/momentum 检验，报告设计矩阵条件数与先验依赖；扩大格距和 pion mass 覆盖后才谈稳健的五参数同时外推。

### 7.3 端点与 LHAPDF 比较

有限 $P$ 下，端点附近的 parton momentum 降低，power corrections 增强。我们的数值 mask $|x|<0.02$ 只是积分实现条件，不是物理可信范围；应结合 $|x|P$、$(1-|x|)P$ 相对强子尺度，及实际多动量稳定性标示区域。

quasi 或有限动量 matched 分布的非物理 support 不应为了图好看直接裁掉后再归一化。$u-d$ 是当前优先的 nonsinglet 通道；单独 connected $u,d$ 的物理内容和全局拟合并不完全相同，尤其 unpolarized/helicity 不能省略 disconnected/singlet/gluon mixing 后宣称完整全 flavor PDF。transversity 的外部算符符号确认状态继续保留。

## 8. 误差预算与 RGR/LRR

文章重复整个流程评估三类主要变化：

- $\mu=\sqrt2,2,2\sqrt2$ GeV 的 renormalization/matching 尺度；
- LA 与 NLA 的长距离展开截断；
- combined 与 sequential 物理极限外推。

它传播同一套 jackknife 并将所选系统误差与统计误差组合。我们使用 bootstrap 也能实现同样的配对链路，无需改成 jackknife。系统误差来源有相关性时，不应默认它们独立；可保存 signed shifts，再给明确标注的汇总 prescription。

尺度变化需要额外小心：文章 Eq. (75) 按字面比较不同 $\mu_R=\mu_F$ 下的最终 LCDA，并将差解释为 truncation uncertainty。但物理 LCDA/PDF 本身随 factorization scale 演化。若没有额外 evolution 到共同尺度，差中包含真实演化。我们与 $Q=2$ GeV 的 LHAPDF 比较时，应固定目标 $\mu_F=2$ GeV 并变化辅助转换/匹配尺度，或先将不同 $\mu_F$ 的结果全部演化回 2 GeV，再评估剩余变化。不能把物理 DGLAP 演化全算成微扰截断。

文章讨论 renormalons、残余质量和 running coupling，不等于已经给出了可直接照搬的完整 RGR+LRR 实现。当前本地重整化代码有独立 LL RGR 对照；主尾部/Fourier/matching 配置仍为 `RGR=false`、`LRR=false`。要更改必须使短距系数、残余质量、尾部 scheme 和匹配阶数协调，而不是单独在末尾乘一个 exponential 或演化因子。

作者列出的三类系统误差也不等于自动覆盖激发态、有限体积、拓扑/自相关、smearing、缺失收缩、算符混合和更高阶/更高扭度的全部不确定度。

## 9. 公开 v2 公式中需澄清的两处

以下是公开文本文字/公式的代数核对，已用 PDF 第 13 页图像确认，不是认定作者数值代码错误。

### 9.1 残余质量符号

Eq. (43) 写

$$
g=\ln M_{\overline{\rm MS},\rm pert}+m_0\widetilde z.
$$

Eq. (41) 的裸矩阵元含 $+g$，而 Eq. (46) 写

$$
\ln Z_{\rm Self}=L_{\rm UV}-m_0\widetilde z,
$$

Eq. (47) 又定义 $M_{\rm Self}=M_B/Z_{\rm Self}$。按这些式子的字面同一 $m_0$ 定义组合，短距得到

$$
\ln M_{\rm Self}
=\ln M_{\overline{\rm MS},\rm pert}+2m_0\widetilde z,
$$

而不是纯 perturbative reference。至少有一处符号或定义需要澄清。当前本地代码在短距 $g$ 和被除去的 $\ln Z_R$ 中都使用 $+m_0|z|$，在其自身 convention 下代数一致。**不能依据原文 Eq. (46) 单独把我们的符号翻转。**

### 9.2 cutoff 对数的实数域

Eq. (41)/(46) 打印的因子包含

$$
\ln\!\left[\frac{\ln(a\Lambda_{\rm latt})}
{\ln(\mu/\Lambda_{\overline{\rm MS}})}\right].
$$

在通常 $a\Lambda_{\rm latt}<1$、$\mu>\Lambda_{\overline{\rm MS}}$ 时，外层 log 的实数自变量为负。实现应核实是否约定 $-\ln(a\Lambda_{\rm latt})$、$\ln[1/(a\Lambda_{\rm latt})]$ 或其他等价正参数。我们的代码已经使用 $-\ln(a\Lambda)$，不应为了“逐字照文章”改回负自变量。

## 10. 本地产品现状与建议优先级

### 10.1 本次核对到的事实

| 环节 | 现状证据 | 含义 |
|---|---|---|
| $P=0$ reference | unpolarized baseline 通过为 L24 3/15、L32x64 7/20、L32x96 4/20；深 cut 对 baseline 最大配对漂移约 $(12\text{--}15)\sigma$，其他通道最高约 $34\sigma$ | 没有证据宣称激发态清除；部分绝对漂移仅百分比量级但统计误差更小 |
| 主 $Z_R$ | 三通道 $\chi^2/{\rm dof}=6.8158,13.6583,24.1575$，nominal data $Q$ 全低于 $10^{-6}$ | 有限参数/有限结果不是已接受的物理拟合 |
| 更深 $P=0$ 对照 | 三通道 $Z_R$ 的 $Q$ 可通过，但所用上游 reference 点仍非全部通过 | 下游改善不能覆盖上游失败 |
| 当前尾部 | 主模型 60/63 通过，仍有衰减接近零/碰界的组合 | 质量门与 Fourier 收敛门是不同条件 |
| Fourier 上限 | 最大中心变化 $\lambda_{\max}:48\to96$ 为 0.323777，$\lambda_{\max}:96\to192$ 为 0.507510，$\lambda_{\max}:192\to384$ 为 0.836533 | 不能统一宣称 $\lambda_{\max}=96$ 或 384 已收敛；后两种仅中心值 |
| hybrid kernel | quark 专用 counterterm 已实现 | 保留，做 scheme 一致性和数值验证 |
| local charge | 独立 $P=0$ 局域荷仅 3/27 过门，E32 外部 $Z$ 未完整解决 | 绝对 PDF 归一化仍待处理 |
| 外推 | 当前 3 ensemble、7 kinematic combinations，没有完成稳健连续/物理质量/无限动量外推 | 当前与 LHAPDF 的图是有限条件下的诊断比较 |

$Q_{\rm data,nominal}$ 在有先验、边界及协方差条件化时不是严格校准的普通抽样 p 值；以上沿用产品中明确声明的诊断含义。用户质量门 $\chi^2/{\rm dof}<1000$、$Q>10^{-6}$ 本次没有更改。

### 10.2 推荐执行顺序

1. **$P=0$ 与局域荷基态隔离。**维持全局统一的物理距离 cut/window 规则，联合 C2 谱信息，对照 self ratio、原 ratio 和 sumratio 差分；用配对 bootstrap 判断漂移，检查模型和参数可辨识性。
2. **重整化参数辨识。**优先 $u-d$ unpolarized；区分 cutoff 与连续 coupling 的 $\Lambda$，固定/扫描 $\Lambda_{\rm latt}$，先中长距确定 UV 斜率，再短距约束 residual mass 与有限转换；传播前一步误差。随后检验跨通道 common UV 是否成立，不一开始硬绑所有参数。
3. **尾部配对比较。**现模型/LA/NLA，在同输入同副本上测试物理 $z_L$ 稳定性、次领先项控制、共同 decay 与独立 decay；不根据误差最小逐点替换。
4. **Fourier 与 matching 收敛。**评估尾部积分本身的误差，允许解析积分到无穷；分别验证上限/网格/卷积域、局域归一化和 plus subtraction。失败的 tail 不因图平滑而升级。
5. **共同尺度的系统误差与局域荷恢复。**补齐 E32 的 $Z$ 来源，确认 transversity 符号，报告 RGR/LRR 状态；所有 LHAPDF 比较统一目标 scheme/scale。
6. **有限格距和动量分析。**受限模型与 nested variants 起步，检查 $aP$，扩大格距/pion mass/momentum 覆盖后再尝试 Eq. (72) 式全模型。

## 11. 可复核的本地证据入口

以下链接指向当前计算工作区的代码与报告：

- [Compare_YaoFei/Renorm/Normalized_Hybrid_20261026/REPORT.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/Normalized_Hybrid_20261026/REPORT.md)：reference 扫描、$Z_R$ 质量、RGR 对照和 normalization。
- [Compare_YaoFei/Renorm/Normalized_Hybrid_20261026/fit_zr.py](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/Normalized_Hybrid_20261026/fit_zr.py)：`model()`、`zr()` 的 $m_0$ 符号和正实数 cutoff log。
- [Compare_YaoFei/Lambda_Fourier_20261027/no_L24P5_no_L96P5/config.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Lambda_Fourier_20261027/no_L24P5_no_L96P5/config.json)：当前尾部公式、fit/covariance、stitch、Fourier phase 与 cutoff。
- [Compare_YaoFei/Lambda_Fourier_20261027/no_L24P5_no_L96P5/REPORT.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Lambda_Fourier_20261027/no_L24P5_no_L96P5/REPORT.md)：60/63、慢尾和积分上限诊断。
- [Compare_YaoFei/Matching/Normalized_Hybrid_20261027/physics.py](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Matching/Normalized_Hybrid_20261027/physics.py)：quark hybrid kernel。
- [Compare_YaoFei/Matching/PaperZ_2310_00814v2_20261027/REPORT.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Matching/PaperZ_2310_00814v2_20261027/REPORT.md)：局域荷、外部 $Z$、当前物理状态。
- [Compare_YaoFei/Renorm/FixedUV_L32x64_am0p1874_AllParams_20261028/REPORT.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/FixedUV_L32x64_am0p1874_AllParams_20261028/REPORT.md)：固定 Wilson-loop 候选质量后的条件拟合。
- [Compare_YaoFei/Renorm/WilsonLoops_C24P29_F32P30_20261028/REPORT.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLoops_C24P29_F32P30_20261028/REPORT.md)：Wilson-loop 候选和有限方案/窗口依赖。

本轮交付是阅读笔记、来源归档和证据哈希。未将任何建议自动升级为新的生产结果。
