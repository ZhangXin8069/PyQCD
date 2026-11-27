# Wilson loop 与 Wilson line 抽取线性发散：方法、结果与诊断归档

初次归档：2026-11-15；最后更新：2026-11-24（北京时间，第17节保留H3推导与11月20日系数；第18节汇总补充flow后的窗口扫描、C24半盒检验和新版图）。目标文件：`/public/home/donghx/.codex/Markdown/linear_devergence.md`。文件名沿用用户指定的 `linear_devergence` 拼写。

本文集中记录本项目已经完成的 Wilson-loop 静态势/相对质量分析、2026-11-10 的闭合圈 double-ratio 重拟合、2026-11-13 的 loop–line 配对子集比较、2026-11-15 的三 ensemble flowed Wilson-line 与 HYP Wilson-line 分析、2026-11-16使用既有 Quark PDF 裸量进行的线性发散减除、共同有限质量拟合和 hybrid 因子比较、2026-11-18至19日完成的长度/参考点审计、长距离局部斜率、单参数有限长度修正和C24P29/F32P30逐级扩展，以及11月20日统一H3、大三个flow窗口检验和C24P29两flow提取。第17节记录H3的一圈Landau推导和当时的条件系数；**最新的8-flow数据、连续子区间和条件系数见第18节**。本次文档更新读取已完成的数值产物，不重新拟合数据。

**截至本次归档，尚未批准统一的最终线性发散系数。数值分析正常完成与拟合模型通过质量验收是两件不同的事。** 文中的小统计误差不包括完整的模型、窗口、自相关、定标、有限体积和连续极限不确定度。第11节记录PDF矩阵元诊断减除；第12–13节记录有限长度/质量扩展模型；第14节保留全flow历史结论；第15节限定大三个flow后有C24伴随、E32两表示和F32伴随通过窗，C24基础联合仍失败；第16节C24两flow分别可拟合但跨pair系数不同；第17节汇总推导、数值、参数数及适用域，不将C24两flow与其余三flow结果混称为统一方案。均不构成最终系数批准，也没有替换生产版smooth ZR。

旧文档 `divergence.md` 保持原状；其理论推导与完整历史仍作为背景。本文以独立的观测量、参数约定、结果和文件入口组织，避免把不同日期的快照、估计器和样本量当成同一次分析。

## 1. 计算对象与当前状态

| 对象/分析 | ensemble 与样本 | 实际状态 |
| --- | --- | --- |
| 早期 Wilson-loop 静态势/relative（2026-10） | 主要为 E32P29/L32×64；883-conf 快照与 1700-conf 全量版本 | 有诊断候选；估计器、bootstrap 数量随版本变化 |
| Wilson-loop double ratio（2026-11-10） | E32P29：1700；F32P30：887；各3000配对bootstrap | 各420个窗口组合；final_d_F/final_d_A 均未定 |
| loop–line 配对子集（2026-11-13） | E32P29：1698；F32P30：869；配对JK | 属于补算完成前快照，不能冒充最新全量 |
| flowed Wilson line（2026-11-15） | C24P29：976；E32P29：1700；F32P30：887 | 原始规范起点 start0；648个联合拟合全部 Q<0.01 |
| flowed Wilson line 长度/参考点与局部斜率（2026-11-18） | 同上三组全量逐组态数据 | 已完成独立扫描；固定协方差子空间时换参考不改变结果；原长距离共同平台仍有显著失配 |
| flowed Wilson line 提高 z_min 与单参数扩展（2026-11-19） | 同上三组全量逐组态数据 | E32P29 基础表示三长度窗、伴随表示四长度窗出现有限长度模型诊断候选；统一最终系数仍未批准 |
| flowed Wilson line C/F逐级参数扩展（2026-11-19） | C24P29：976；F32P30：887 | 3024个拟合；C24基础表示仍无候选，C24伴随和F32两表示有条件候选；不同质量修正形状给出的d显著不同 |
| flowed Wilson line 三ensemble统一H3（2026-11-20） | C24P29：976；E32P29：1700；F32P30：887 | 7560个拟合；放宽后E32两表示有条件工作值，C24伴随仅脆弱单窗，C24基础与F32全flow仍失败 |
| flowed Wilson line 大三个flow的H3（2026-11-20，第15节） | 同上三组全量数据；仅原flow1/2/3 | 复用1512个拟合、新增504项内部检验；C24伴随、E32两表示、F32伴随有范围内合理窗；F32基础仅三长度候选，C24基础仍失败 |
| flowed Wilson line C24P29两flow（2026-11-20，第16节） | C24P29：976；主范围为大三个flow的三种pair | 6048项H3/leading扫描；最大pair两表示在0.630–1.155fm通过，宽窗d跨pair仍有约4.45–5.54sigma差异，未确立通用系数 |
| flowed Wilson line 补充至8个flow（2026-11-24，第18节） | C24P29/E32P29/F32P30：976/1700/887；同组态配对，最大流半径0.249875 fm | 全8-flow不作生产系数；21个连续flow子区间扫描给出六个H3工作窗，E32基础表示仍有显著flow窗依赖 |
| 3D HYP5 Wilson line | C24P29/E32P29/F32P30；各100 | 三组测量完整；系数仅为有限长度/方案诊断 |
| 4D HYP10 Wilson line | C32P29/E32P29/F32P30；各100 | 三组测量完整；不是3D HYP10 |
| 3D HYP10 Wilson line | 尚无本次可用的实测 campaign | 不能用4D HYP10重新命名替代 |
| Quark PDF WL质量/hybrid测试（2026-11-16） | C24P29/E32P29/F32P30；Ncfg=199/425/194；各2000 bootstrap | 204个固定c组合中4个短窗组合过数值门；主窗口及上游仍有质量限制，全链为diagnostic |

## 2. 统一符号、几何与对数估计量

令 $s=F,A$ 分别表示基础/基本表示（fundamental）与伴随表示（adjoint），$t_f$ 为物理流时间，$\tau=t_f/a^2$，$r_f=\sqrt{8t_f}=a\sqrt{8\tau}$。空间线长为 $z=na$；矩形圈两条边为 $R,T_W$。下文使用正的线质量约定

$$
 m_{\mathrm{line},s}^{\mathrm{leading}}=\frac{d_s}{r_f}.
$$

于是单条开放线的 leading 部分为 $\ln W_{\rm line}\supset-d_sz/r_f$；闭合矩形圈的周长部分为 $\ln W_{\rm loop}\supset-2d_s(R+T_W)/r_f$。固定 $T_W$ 后，圈沿空间长度的斜率带有两条空间边的因子2；时间边的自能与其他长度无关项进入截距。**闭合圈拟合公式已经包含因子2时，所得单边定义的 $d_s$ 不能再除以2。**

和旧笔记记录的 Pion_DA 单线记号对齐时，

$$
 \ln\mathcal R=\kappa_s z\left(\frac1{\sqrt{t_f}}-\frac1{\sqrt{t_{f,r}}}\right)+B(t_f),
 \qquad \kappa_s=-\frac{d_s}{\sqrt8}.
$$

本文的 PDF 数字引用沿用旧文档第28节的摘录与 convention 对齐记录，没有在本次文档工作中重新核读 PDF。不能把正的 $d_F$ 与负的 $\kappa_F$ 直接相除来比较。

全部最新分析使用 $\ln[\mathrm{Re}\langle W\rangle]$：先对组态及声明的方向/正反路径求平均，再取对数。它不是 $\langle\ln W\rangle$。参与对数的 ensemble/replica 均值须正且有限；不取绝对值、不裁剪、不用零补失败值。不同流时间和长度共用重采样索引。

伴随表示由逐位置/逐路径的 SU(3) character 测量得到。对 SU(3)，局部归一化 character 可写为 $(|\mathrm{Tr}_F U|^2-1)/8$；不能对已经平均的基础表示 Wilson line/loop 取平方替代。本文没有用 Casimir 比例约束 $d_A$。

## 3. 早期 Wilson-loop 路线与历史候选

### 3.1 从时间衰减得到势，再研究线质量

早期 L32×64 计算使用 thin-link 四维 Wilson RK3 flow、无 HYP/stout、无 gauge fixing。原始23个流时间为

```text
tau64 = 0, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
        .55, .60, .65, .80, 1.00, 1.40, 1.80, 2.20, 2.60,
        3.00, 3.40, 3.80
```

2026-10-16 的883-conf/2000-bootstrap快照与2026-10-17的1700-conf/1000-bootstrap全量版不是同一统计版本；2026-10-18以后另有1700-conf/3000-bootstrap分析。旧数值不能混成一张“全量2000-bootstrap”结果表。

Cornell 路线先由 $W(R,T_W)\sim A(R)\exp[-V_{\rm eff}(R)T_W]$ 提取有限时间有效势，再研究 $V_{\rm eff}(R)$ 的距离依赖。空间常数项还含有限部分，不能不加条件地全部视为紫外线自能。

本项目旧代码中的 Wilson relative 指跨 flow 作差，**不是** $V(R)-V(R_{\rm ref})$。2026-10-20 的直接时间 log-ratio 路线固定 $R$，拟合

$$
 -\ln\frac{\bar W(R,T_W;\tau)}{\bar W(R,T_W;\tau_r)}
 =A(R;\tau,\tau_r)+2T_W\Delta m_{\rm eff}(R;\tau,\tau_r),
$$

随后研究 $a\Delta m_{\rm eff}=d_s(q-q_r)+c_s(\tau-\tau_r)$，其中 $q=1/\sqrt{8\tau}$。其输出还可能含相互作用势差的一半及有限 $T_W$ 污染；不是天然纯净的单线质量。

### 3.2 2026-10-20 的代表性候选与上游失配

以下沿用旧文档第26节：E32P29、1700 conf/3000配对bootstrap、参考 $\tau_r=0.15$、时间窗 $T_W/a=8..12$、含 $c$ 模型、$R/a=6,7$ 相关平台。

| tau_max | 平台候选 d_F ± stat | 平台 chi²/dof | 平台 Q |
| --- | --- | --- | --- |
| 0.50 | 0.40418592 ± 0.00287239 | 1.94527 | 0.16310 |
| 0.65 | 0.39055588 ± 0.00171659 | 1.27326 | 0.25916 |
| 0.80 | 0.38263473 ± 0.00144453 | 0.000145506 | 0.99038 |
| 1.00 | 0.37475233 ± 0.00121053 | 1.78842 | 0.18112 |

上述平台误差均来自完整3000/3000样本，但 $\tau_{\max}=1.0$ 时，上游固定 $R/a=6$ 的 flow 拟合给 $d_F=0.37413696\pm0.00131600$、$\chi^2/\mathrm{dof}=62.06299$、$Q=1.55\times10^{-52}$；$R/a=7$ 的 $Q=7.57\times10^{-17}$。空间平台的良好 $Q$ 不能消除上游模型失配。同一批样本的上限漂移 $d_F(0.5)-d_F(1.0)=0.02943358\pm0.00276048$，远大于固定模型误差。

伴随表示曾在新参考 $\tau_r=0.5$、$\tau_{\max}=1.0$、同一时间窗中给出 $R/a=4$ 的 $0.869908\pm0.002214$（$Q=0.2228$）和 $R/a=5$ 的 $0.833204\pm0.008895$（$Q=0.8957$），但未建立受控距离平台；较大 $R$ 还有不完整样本。没有因此确定最终 $d_A$。

2026-10-21 曾以固定候选 $d_F=0.374137$ 做局域归一化矩阵元的诊断性减除，见旧文档第27.9节。这是既有历史，不是正式系数批准，也不意味着完成有限残余质量、方案转换或 LaMET matching。本次没有新增减除。

历史入口：[原文第23–27节](/public/home/donghx/.codex/Markdown/divergence.md)；[直接时间法与静态势图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_relative_Tw_static_20261020)；[直接时间斜率后的d分析](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_relative_d_from_Tw_20261020)。

## 4. Wilson-loop double ratio：2026-11-10 重拟合

### 4.1 数据与模型

分别分析 E32P29/L32×64（$a=0.0897$ fm、1700 conf）和 F32P30/L32×96（$a=0.0775$ fm、887 conf），各3000配对bootstrap。前者的原始网格 $R/a=0..12,T_W/a=0..16$，后者 $R/a=0..16,T_W/a=0..20$。中心和协方差来自平均圈及 delete-one JK；bootstrap seed 分别为20261015和2026111096。

L32×96 匹配 L32×64 的物理流半径：$\tau_{96}=1.339619562955255\,\tau_{64}$。表中 `tau64` 只是共同物理网格的标签，不能作为 L32×96 的实际 $t_f/a^2$。主参考为实测 $\tau_{64,r}=0.60$，对应 $r_{f,r}=0.1965228536$ fm；没有插值成精确0.20 fm。

固定 $T_W$，定义

$$
 D_s(R,T_W;\tau,\tau_r)=
 \frac{\bar W_s(R,T_W;\tau)/\bar W_s(0,T_W;\tau)}
 {\bar W_s(R,T_W;\tau_r)/\bar W_s(0,T_W;\tau_r)}.
$$

闭合退化路径 $W(0,T_W)=1$，所以数值上退化为跨 flow 的圈比值。三个模型是

$$
\begin{aligned}
\texttt{paper_leading}:&\quad\ln D_s=-2n d_s(q-q_r)+B_{s,\tau,T_W},\
\texttt{flow_c}:&\quad\ln D_s=-2n[d_s(q-q_r)+c_s(\tau-\tau_r)]+B_{s,\tau,T_W},\
\texttt{lattice_q3}:&\quad\ln D_s=-2n[d_s(q-q_r)+e_s(q^3-q_r^3)]+B_{s,\tau,T_W}.
\end{aligned}
$$

每个非参考 flow 有独立截距。$c_s,e_s$ 是经验模型敏感性参数，分别加一项，不是已验证的完整有限-flow/格距展开。该 loop 版本的 $c_s$ 使用无量纲 $\tau$，不能直接与最新 line 模型中使用物理 $t_f$ 的 $c_{\rm flow}$ 比数值。

三种 flow 窗口（均另含参考0.60）为：

- `a_resolved`：目标 tau64=.15,.20,.25,.30,.35,.40,.45,.50,.55。
- `trim_low`：目标 tau64=.30,.35,.40,.45,.50,.55。
- `strict030`：目标 tau64=.15,.20,.25,.30,.35,.40,.45,.50,.55,.65,.80,1.00。

完整 JK 相关矩阵采用明确删模，主阈值10⁻⁸，另扫10⁻⁶/10⁻¹⁰；原始样本秩阈值10⁻¹²。只有3000/3000份bootstrap均有效时给完整统计误差；不删除失败样本后补报误差。没有按 Q、最小误差或接近 PDF 的程度选最终系数。

### 4.2 基础表示：短距离失配与长距离候选

下表从保存的 `fit_results.json` 读取。长距离例子用于解释，不是最终系数或相互独立的测量。

| ensemble | 条目 | flow窗 | T_W/a | R/a | d_F ± stat | chi²/dof | Q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L32x64 | fit0062 | a_resolved | 8 | 4..8 | 0.488858 ± 0.000316 | 3330.2 | 0 |
| L32x64 | fit0115 | trim_low | 10 | 8..12 | 0.387950 ± 0.005620 | 0.41038 | 0.98644 |
| L32x64 | fit0135 | trim_low | 12 | 6..10 | 0.386853 ± 0.004812 | 1.3845 | 0.1271 |
| L32x96 | fit0062 | a_resolved | 9 | 5..9 | 0.475190 ± 0.000424 | 2519 | 0 |
| L32x96 | fit0115 | trim_low | 12 | 10..13 | 0.370066 ± 0.014500 | 1.2341 | 0.24677 |
| L32x96 | fit0135 | trim_low | 14 | 7..11 | 0.378509 ± 0.006946 | 1.505 | 0.077316 |

### 4.3 如何理解 $d_F=0.32982\pm0.00450$

这是 **L32×96、基础表示、`fit0087`、`strict030`、`flow_c`** 的条件结果。实际 $T_W/a=9$（0.6975 fm）、$R/a=10..13$（0.7750–1.0075 fm）、流半径约0.09826–0.25371 fm，参考0.19652 fm。同窗口模型对照如下。

| ensemble | 条目 | 模型 | d_F ± stat | chi²/dof | Q | 有效bootstrap |
| --- | --- | --- | --- | --- | --- | --- |
| L32x96 | fit0086 | paper_leading | 0.420357 ± 0.001923 | 25.695 | 7.6757e-101 | 3000/3000 |
| L32x96 | fit0087 | flow_c | 0.329817 ± 0.004496 | 1.5279 | 0.061312 | 3000/3000 |
| L32x64 | fit0086 | paper_leading | 0.426810（无完整误差） | 31.087 | 8.2822e-177 | 2808/3000 |
| L32x64 | fit0087 | flow_c | 0.353040（无完整误差） | 2.6657 | 2.8522e-06 | 2808/3000 |

L32×96 的完整值为 $d_F=0.3298167833\pm0.0044956102$，对应 $\kappa_F=-0.1166078420\pm0.0015894382$。它是允许额外 $c_F(\tau-\tau_r)$ 项后得到的 $1/r_f$ 项系数，不是线质量本身，也不是先把裸圈外推到 $t_f=0$ 的结果。

同一批bootstrap的模型差为 $d_F^{\rm leading}-d_F^{\rm flow_c}=0.09053992\pm0.00407450$，说明模型变化远大于固定模型的统计误差。修正项还可能吸收有限几何、相互作用或 cutoff 污染。L32×64 相应修正条目仅2808/3000样本有效，不能用两组条目宣称已经建立跨格距受控结果。

旧文档引用的 Pion_DA 单线数字 $\kappa_F=-0.1165\pm0.0001$ 换算为 $d_F=0.32951176\pm0.00028284$。loop 的修正模型数字接近它，只是特定模型下的比较，没有把该数字用作先验，也不能称逐项复现了其规范固定开放线观测量。

### 4.4 伴随表示与失败覆盖

| ensemble | T_W/a | R/a | trim_low 的 d_A ± stat | chi²/dof | Q |
| --- | --- | --- | --- | --- | --- |
| L32x64 | 6 | 4..8 | 1.069628 ± 0.000909 | 441.8 | 0 |
| L32x96 | 7 | 5..9 | 0.993866 ± 0.001512 | 176.97 | 0 |

| ensemble | 扫描组合 | 有限中心 | 完整3000样本误差 |
| --- | --- | --- | --- |
| L32x64 | 420 | 164 | 106 |
| L32x96 | 420 | 148 | 106 |

伴随表示仍无正式系数。上表的计数是数值覆盖，不是物理质量通过数；许多窗口受非正圈、不完整重采样或模型失配限制。两组均保持 `final_d_F=null`、`final_d_A=null`。

图表入口：[长距离拟合示例（6页，最后一页为0.32982条目）](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/figures/long_distance_fit_examples.pdf)；[两ensemble短距离对照](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/figures/ensemble_comparison.pdf)；[L64完整CSV](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/L32x64/fit_results.csv)；[L96完整CSV](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/L32x96/fit_results.csv)；[解释与配对模型差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/INTERPRETATION.md)。

## 5. 规范固定开放 Wilson line：生产约定与配对子集历史

### 5.1 当前统一采用 Landau start0

最新 flowed line 来自 thin-link 四维 Wilson flow 后的四维 Landau gauge fixing。每个 flow 都从对应的原始流化规范场起点进行规范固定，保留原始起点 `start0`；不是由前一 flow 的固定结果热启动，也不使用随机多起点挑选。

早期单conf试验曾使用 `best of 4 (largest gauge functional at each flow)`：在每个 flow 的多个规范固定候选中选择 gauge functional 最大者。用户随后指定只保留原始规范起点，这一生产政策已替代旧试验。best-of与start0属于不同样本定义，旧单conf图不能混入本次全量统计。

三组原始/重试任务最终审计为全部完成：C24P29=976、E32P29=1700、F32P30=887；补算分别恢复17、2、18个conf。原始与重试的求解器参数政策存在混合，旧审计中明确 `solver_policy_mixed=true`；这是来源限制，不应省略为所有配置完全相同的求解器路径。最新数据清单逐组态保存来源与校验码。

HYP line 同样采用四维 Landau、只从输入HYP规范场原始起点开始，没有额外 gradient flow；不是Coulomb gauge。3D/4D指HYP输入的涂抹维数，不是此处 gauge-fixing 的维数。HYP测量核验紧阈值 $\theta\le10^{-12}$，并核验闭合圈不变性、SU(3)误差与正反线关系。

### 5.2 2026-11-13 的 loop–line 配对子集比较

该历史比较使用 E32P29=1698、F32P30=869个当时 line 已完整的conf，并在loop端取完全相同子集，不是loop全量对line子集。横轴为无量纲 $1/\sqrt\tau$，每个几何先拟合 $\ln W=b(n)/\sqrt\tau+c(n)$，再比较

$$
 b_{\rm line}(n)=\kappa n+c,\qquad
 b_{\rm loop}(n)=2\kappa n+c_T,\qquad \kappa=-d/\sqrt8.
$$

例：$n=4..8$ 窗口的基础表示，E32P29 的 $\kappa_{\rm line}=-0.114893\pm0.000012$、$\kappa_{\rm loop}=-0.158447\pm0.000133$；配对差为 $-0.043553\pm0.000133$。F32P30 对应为 $-0.099042\pm0.000013$、$-0.144338\pm0.000138$；配对差为 $-0.045297\pm0.000138$。相关长度拟合Q均数值下溢为0，故不能据这些差值宣称最终UV系数不同。

该比较使用删模阈值10⁻¹⁰、block1 JK；它与2026-11-15的直接联合 double-ratio拟合（主阈值10⁻⁸、最新完整样本）不是同一估计器。原始loop斜率含两条时间边，不与单line原始斜率直接相等。旧比较还记录了两端RK3步长的差异。失败/补算完成后，不能把其1698/869样本量称为最新全量。

配对子集图表：[方法与限制](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_loop_line_slopes_20261113/README.md)；[结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_loop_line_slopes_20261113/RESULTS.md)。最终测量审计：[三组line全部完成](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/landau_wilson_failed_retry_start0_gpu025_20261113/completion_audit.json)。

## 6. 三 ensemble flowed Wilson line：2026-11-15 基线归档

本节保留2026-11-15的基线模型、648个拟合和原展示窗口，不用后续窄窗候选覆盖其失败状态。2026-11-18至19日的长度/参考点审计、局部斜率、提高下限和新增一个参数的补充测试见第12节。

### 6.1 数据、长度和实际流时间

使用每组全部完整组态，线长 $n=z/a=0..12$。格距中心值没有附带定标误差。三组共享物理流半径约为0.138962642、0.160460238、0.179400000、0.196522854 fm。

| ensemble | 体积 | a [fm] | Nconf | 实际 tau=t_f/a² |
| --- | --- | --- | --- | --- |
| C24P29 | L24x72 | 0.105 | 976 | 0.218941224, 0.291921633, 0.364902041, 0.437882449 |
| E32P29 | L32x64 | 0.0897 | 1700 | 0.300000000, 0.400000000, 0.500000000, 0.600000000 |
| F32P30 | L32x96 | 0.0775 | 887 | 0.401885869, 0.535847825, 0.669809781, 0.803771738 |

总览图每条曲线固定 $n$，并标明对应物理 $z$；相同 $n$ 在不同 ensemble 上不代表相同物理长度。横轴是物理 $1/\sqrt{t_f}$，单位fm⁻¹。全量单长度图并不等于把三组line按相同物理长度合并拟合。

### 6.2 单长度斜率与联合系数的区别

对每条长度单独拟合

$$
 \ln W(z,t_f)=b(z)\frac1{\sqrt{t_f}}+c(z),\qquad
 d_{\rm path}(z)=-\sqrt8\,b(z)/z.
$$

$d_{\rm path}$ 仍含可能的端点/有限长度贡献，是路径归一化诊断，不与联合 double-ratio 的 $d_{\rm GF}$ 混称。对无量纲横轴 $1/\sqrt\tau$ 得到的斜率应先乘以 $a$ 才能对应上述物理横轴的 $b$。

联合拟合使用

$$
 D_s(z,t_f)=\ln\langle W_s(z,t_f)\rangle-\ln\langle W_s(z,t_{f,r})\rangle,
$$

这里已用 $W_s(0,t_f)=1$。模型为

$$
\begin{aligned}
\texttt{leading}:&\quad D_s=-d_{{\rm GF},s}z(1/r_f-1/r_{f,r})+B_{s,t_f},\
\texttt{flow_c}:&\quad D_s=-d_{{\rm GF},s}z(1/r_f-1/r_{f,r})
-c_{{\rm flow},s}z(t_f-t_{f,r})+B_{s,t_f}.
\end{aligned}
$$

$d_{\rm GF}$ 无量纲，$c_{\rm flow}$ 单位fm⁻³；每个非参考flow有独立端点截距。开放单线没有闭合圈的因子2。额外线性 $t_f$ 项仅为经验模型敏感性检查，尚未单独验证其物理完整性。

### 6.3 统计、扫描和展示窗口

共享 delete-block JK；block=1,2,5。主展示为block1、全四流时间、$z\in[0.3,0.7]$ fm、相关矩阵删模阈值10⁻⁸，均预先指定，没有按Q挑选。扫描还包括 $z$ 窗口[0.2,0.6]、[0.4,0.8] fm，较小/较大三个流时间，及删模阈值10⁻⁶。每个flow窗口以其最大流时间为参考。原始样本秩门和模型秩门均保留，无prior、无静默对角化。

block1使用全部conf；block2/5仅为blocking敏感性计算丢弃不足一块的尾部conf，明确保存实际ID及组块；它不等价于已经完成自相关控制。同一ensemble跨长度/flow的相关性进入协方差，不同ensemble不拼接为同一组配对复制品。

### 6.4 中央展示结果：完整模型对照

| ensemble | 表示 | 模型 | d_GF ± JK stat | chi²/dof | Q |
| --- | --- | --- | --- | --- | --- |
| C24P29 | fundamental | leading | 0.323039 ± 0.000046 | 21461 | 0 |
| C24P29 | fundamental | flow_c | 0.310722 ± 0.000241 | 24138 | 0 |
| C24P29 | adjoint | leading | 0.737036 ± 0.000139 | 7657.8 | 0 |
| C24P29 | adjoint | flow_c | 0.690662 ± 0.000794 | 8249.5 | 0 |
| E32P29 | fundamental | leading | 0.329788 ± 0.000037 | 22116 | 0 |
| E32P29 | fundamental | flow_c | 0.361603 ± 0.000208 | 21830 | 0 |
| E32P29 | adjoint | leading | 0.750111 ± 0.000118 | 7625.3 | 0 |
| E32P29 | adjoint | flow_c | 0.804244 ± 0.000645 | 7672.3 | 0 |
| F32P30 | fundamental | leading | 0.286776 ± 0.000040 | 54147 | 0 |
| F32P30 | fundamental | flow_c | 0.372462 ± 0.000207 | 41806 | 0 |
| F32P30 | adjoint | leading | 0.637244 ± 0.000102 | 36199 | 0 |
| F32P30 | adjoint | flow_c | 0.823103 ± 0.000704 | 33018 | 0 |

全部648/648联合拟合均可通过数值秩门，但全部Q<0.01。不同窗口和模型的漂移远大于表中统计误差；因此这些不是验收后的最终紫外系数。个别E32P29数字接近旧loop的0.32982，不意味着相同观测量、几何或模型已经一致。最新分析没有连续极限拟合，只画了分别拟合的系数对a²的诊断连线。

图表与机器结果：[三组lnW–逆平方根流时间总览](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/figures/gradient_flow_lnW_overview.pdf)；[系数窗口扫描](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/figures/gradient_flow_d_window_scan.pdf)；[单路径系数](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/figures/gradient_flow_path_coefficients.pdf)；[double-ratio拟合细节](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/figures/gradient_flow_double_ratio_details.pdf)；[全部联合拟合CSV](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/gradient_flow/double_ratio_coefficients.csv)；[blocking组态ID](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/blocking_conf_ids.json)。

## 7. HYP Wilson line：实际输入与方案区别

两个HYP campaign均已经完成各300conf（每ensemble100conf）的测量，并从逐conf原始测量文件重新核对统计输入。3D HYP5 与4D HYP10的参数记录中，原始HYP alpha参数尚未提供；不能据次数和目录名宣称两方案已做完整参数匹配。

### 7.1 3D HYP5

| ensemble | 体积 | a [fm] | Nconf | 测量n_max |
| --- | --- | --- | --- | --- |
| C24P29 | 24×24×24×72 | 0.105 | 100 | 12 |
| E32P29 | 32×32×32×64 | 0.0897 | 100 | 16 |
| F32P30 | 32×32×32×96 | 0.0775 | 100 | 16 |

- C24P29：`/public/group/lqcd/donghx/Hpysmear_beta6.20_mu-0.2770_ms-0.2400_L24x72/hpy_3D_5times`。

- E32P29：`/public/group/lqcd/donghx/Hypsmear_beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/hyp_3D_5times`。

- F32P30：`/public/group/lqcd/donghx/Hypsmear_beta6.41_mu-0.2295_ms-0.2050_L32x96/hyp_3D_5times`。

### 7.2 实际已有的4D HYP10（不是3D HYP10）

| ensemble | 体积 | a [fm] | Nconf | 测量n_max |
| --- | --- | --- | --- | --- |
| C32P29 | 32×32×32×64 | 0.105 | 100 | 16 |
| E32P29 | 32×32×32×64 | 0.0897 | 100 | 16 |
| F32P30 | 32×32×32×96 | 0.0775 | 100 | 16 |

- C32P29：`/public/group/lqcd/donghx/Hpysmear_beta6.20_mu-0.2770_ms-0.2400_L32x64/hpy_4D_10times_new`。

- E32P29：`/public/group/lqcd/donghx/Hypsmear_beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/hyp_4D_10times_new`。

- F32P30：`/public/group/lqcd/donghx/Hypsmear_beta6.41_mu-0.2295_ms-0.2050_L32x96/hyp_4D_10times_new`。

最粗格距由C24P29的24³×72变为C32P29的32³×64，不能假设组态配对。E32P29在两套HYP列表中共同conf为92个，F32P30为98个，但本次没有据此生成两方案的配对差值分析。不同ensemble的1/a拟合仍按独立样本传播误差。固定HYP次数不能被直接当作固定物理flow半径；本文未给它们分配等效t_f。

## 8. HYP 的 $1/a$ 依赖和系数提取

### 8.1 固定物理长度的直接格距拟合

预先指定 $z=0.30,0.45,0.60,0.75,0.90$ fm。每组在相邻实测长度间对 $\ln\langle W\rangle$ 分段线性插值，并对共享JK样本应用同一插值；不外推。这样比较的是相同物理长度，避免把固定 $n=z/a$ 下的不同线长误认为格距依赖。

每个 $z$ 独立拟合三个格距：

$$
 \ln\langle W_s(z,a)\rangle=-c_{{\rm HYP},s}(z)\frac za+B_s(z),
 \qquad c_{{\rm HYP},s}(z)=-\frac1z\frac{d\ln W_s}{d(1/a)}.
$$

三个ensemble彼此独立，参数方差为各自JK贡献之和，不将不同ensemble的第k个JK样本人为配对。每个三点两参数拟合有1个自由度。下表是全部五个预定长度的block1结果，误差只含统计部分。

**3D HYP 5 steps**

| 表示 | z [fm] | c_HYP(z) ± stat | chi²/dof | Q |
| --- | --- | --- | --- | --- |
| fundamental | 0.30 | 0.093688 ± 0.000248 | 1.597 | 0.20633 |
| fundamental | 0.45 | 0.110745 ± 0.000340 | 45.384 | 1.6193e-11 |
| fundamental | 0.60 | 0.121690 ± 0.000427 | 12.336 | 0.00044438 |
| fundamental | 0.75 | 0.129193 ± 0.000517 | 15.11 | 0.00010144 |
| fundamental | 0.90 | 0.134023 ± 0.000622 | 30.047 | 4.2175e-08 |
| adjoint | 0.30 | 0.210408 ± 0.000535 | 0.91283 | 0.33936 |
| adjoint | 0.45 | 0.249164 ± 0.000739 | 48.893 | 2.7025e-12 |
| adjoint | 0.60 | 0.274379 ± 0.000961 | 13.786 | 0.00020488 |
| adjoint | 0.75 | 0.291716 ± 0.001254 | 14.728 | 0.00012417 |
| adjoint | 0.90 | 0.302128 ± 0.001722 | 23.215 | 1.4486e-06 |

**4D HYP 10 steps**

| 表示 | z [fm] | c_HYP(z) ± stat | chi²/dof | Q |
| --- | --- | --- | --- | --- |
| fundamental | 0.30 | 0.037785 ± 0.000222 | 10.128 | 0.0014602 |
| fundamental | 0.45 | 0.051392 ± 0.000308 | 10.166 | 0.0014302 |
| fundamental | 0.60 | 0.062521 ± 0.000386 | 0.08289 | 0.77342 |
| fundamental | 0.75 | 0.071244 ± 0.000459 | 0.63374 | 0.42599 |
| fundamental | 0.90 | 0.077911 ± 0.000535 | 0.0098439 | 0.92097 |
| adjoint | 0.30 | 0.084630 ± 0.000481 | 9.6156 | 0.0019293 |
| adjoint | 0.45 | 0.115026 ± 0.000667 | 10.286 | 0.0013405 |
| adjoint | 0.60 | 0.140145 ± 0.000844 | 0.066279 | 0.79683 |
| adjoint | 0.75 | 0.160350 ± 0.001035 | 0.50802 | 0.476 |
| adjoint | 0.90 | 0.176468 ± 0.001269 | 0.0058607 | 0.93898 |

$c_{\rm HYP}(z)$ 有明显的长度依赖，因而当前结果是有限长度下的有效格距斜率，尚未分离出与长度无关的纯紫外自能。个别固定 $z$ 的Q良好只支持该三点拟合的描述能力，不构成普适系数的证明。

### 8.2 先拟合长度，再拟合格距：模型失配的交叉诊断

另行在每个ensemble拟合 $\ln W=A-\mu_{\rm eff}z$，再拟合 $\mu_{\rm eff}(a)=c_{\rm HYP}/a+m_0$。这里的 $m_0$ 只是该诊断回归的截距，不是已经确定的 $m_0^{\overline{\rm MS}}$。同样扫描三个长度窗口、block1/2/5和两个删模阈值。展示窗口[0.3,0.7] fm、block1、阈值10⁻⁸：

| 实际方案 | 表示 | 两步法c_HYP ± stat | 格距chi²/dof | 格距Q |
| --- | --- | --- | --- | --- |
| 3D HYP 5 steps | fundamental | 0.059275 ± 0.000208 | 65855 | 0 |
| 3D HYP 5 steps | adjoint | 0.149283 ± 0.000562 | 1.0651e+05 | 0 |
| 4D HYP 10 steps | fundamental | 0.013916 ± 0.000120 | 12187 | 0 |
| 4D HYP 10 steps | adjoint | -0.038615 ± 0.000276 | 30478 | 0 |

两个方案各108个上游长度拟合全部Q<0.01。这条两步路线尚未通过模型质量验收；其负参数或强烈窗口/block依赖不能解释为负的UV发散。完整相关拟合在失败模型中可能出现与目测趋势不一致的参数，需同时检查原始lnW–z图和相邻长度斜率，不能只引用回归中心。

3D HYP 5 steps：[lnW 对1/a](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/presentation/hyp3d5_lnW_vs_inverse_a.pdf)；[固定z有效系数](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/presentation/hyp3d5_c_vs_fixed_z.pdf)；[长度窗/block扫描](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/hyp3d5/c_hyp_window_scan.pdf)；[长度拟合与局部斜率诊断](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/hyp3d5/length_fit_diagnostics.pdf)；[格距结果CSV](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/hyp3d5/spacing_coefficients.csv)。

4D HYP 10 steps：[lnW 对1/a](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/presentation/hyp4d10_lnW_vs_inverse_a.pdf)；[固定z有效系数](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/presentation/hyp4d10_c_vs_fixed_z.pdf)；[长度窗/block扫描](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/hyp4d10/c_hyp_window_scan.pdf)；[长度拟合与局部斜率诊断](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/hyp4d10/length_fit_diagnostics.pdf)；[格距结果CSV](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/hyp4d10/spacing_coefficients.csv)。

## 9. 怎样比较 loop、flowed line 与HYP系数

1. **先对齐定义和几何因子。** loop联合拟合已经除去了两条空间边的几何因子，line没有该因子；原始loop的flow斜率还含两条时间边。$d=-\sqrt8\kappa$，不能直接比较 $d$ 与 $\kappa$。
2. **区分物理流时间与格点流时间。** $t_f=a^2\tau$。横轴 $1/\sqrt{t_f}$ 的斜率带长度单位，$1/\sqrt\tau$ 的斜率无量纲；模型和表格中的c参数也可能使用不同单位。
3. **区分估计器、窗口和统计版本。** 时间衰减→flow拟合、逐长度flow斜率→长度拟合，以及直接联合double ratio不是同一个估计器。2026-11-13的1698/869快照、2026-11-15的1700/887全量和不同删模阈值不可混为同次结果。
4. **HYP和flow使用不同截止定义。** $d_{\rm GF}$乘 $z/r_f$，$c_{\rm HYP}$乘 $z/a$；二者均无量纲不代表数值应相同。3D HYP5与4D HYP10也不是仅次数不同的同一方案。
5. **拟合质量与参数可辨识分开。** 完整样本和秩门通过不意味着模型正确；高精度数据中目测直线仍可有极差Q。不能靠删模、挑窗口或挑选接近参考值的模型确定“最终系数”。
6. **当前尚未完成各类极限控制。** 有限 $T_W$、有限 $z/r_f$、flow窗、格距、积分步长及规范固定政策均应作为不同问题核验；不能由一次较好Q替代整条链的物理验收。

### 9.1 是否先对裸Wilson loop/line做small-flow-time外推

本次提取过程没有先将裸W外推到 $t_f=0$，而是在有限正流时间窗识别 $1/r_f$ 项。固定格距的零流点受格点cutoff调节，不能把 $d/\sqrt{8t_f}$ 表达式不加条件地一直延伸到零。Wilson时间边的大时间极限与flow时间的零流极限也不是同一个操作。

这不表示目标未流化算符的small-flow-time分析已完成。线自能分离、有限方案转换、固定物理正flow的连续极限和后续小流时间控制，仍需按完整重整化链处理；本次没有输出可直接用于PDF扣除的最终系数。

### 9.2 每个ensemble留一个系数，还是直接外推a→0

本次先保留每个ensemble、窗口和模型下的诊断系数及相关样本，不能在显著模型失配下先定唯一数值再外推。最新三个格距的 $d_{\rm GF}$–$a^2$ 图仅为诊断，未实施连续极限回归。HYP的1/a图是在拟合发散形状，不能把它叫作有限重整化物理量的连续极限结果。最终能否给共同系数，需要先建立一致方案、固定物理条件和可接受的拟合质量。

## 10. 图表、代码、审计与后续复查入口

- [旧理论和历史总文档](/public/home/donghx/.codex/Markdown/divergence.md)

- [旧Wilson-loop double-ratio说明](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/INTERPRETATION.md)

- [loop代码与模型](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/code/run.py)

- [loop估计量和GLS核心](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/code/core.py)

- [loop统一完成记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_double_ratio_L64_L96_20261110/complete.json)

- [2026-11-13配对子集目录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_loop_line_slopes_20261113)

- [最新三组flow单长度统计输入](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flowed_wilson_line_three_ensemble_plots_20261114/config.json)

- [最新flow/HYP八页图集](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/presentation/all_main_figures.pdf)

- [最新分析中文说明](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/README.zh-CN.md)

- [最新分析配置](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/config.json)

- [最新联合拟合源码](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/code/analyze.py)

- [最新统计核心](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/code/statistics_helpers.py)

- [所有flow系数机器表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/gradient_flow/fits.json)

- [最新系数展示汇总](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/coefficient_summary.json)

- [主分析质量报告](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/report.json)

- [主分析完成与校验码](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/complete.json)

- [独立数值复核272项](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/independent_validation.json)

- [最终交付及校验码](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_flow_hyp_comparison_20261115/delivery.json)

- [3D HYP5测量请求](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_landau_wilson_100conf_20261114/request.json)

- [3D HYP5测量完成审计](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp_landau_wilson_100conf_20261114/completion_audit.json)

- [4D HYP10测量请求](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp4d10_landau_wilson_100conf_20261114/request.json)

- [4D HYP10测量完成审计](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/hyp4d10_landau_wilson_100conf_20261114/completion_audit.json)

最新CPU作业：主分析4010930、独立复核4010936、图集/说明整理4011043，均已完成；独立数值复核为PASS、272项。主分析报告明确 `final_coefficient_approved=false`。2026-11-10 loop的统一补充验收作业3963970完成，状态为 `PASS_NUMERICAL_DIAGNOSTIC_ONLY`；不是正式物理系数批准。

本次文档组装脚本、输入文件SHA256清单、链接检查和安装记录保存于：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_20261115`。文档中的最新数值表由保存的JSON自动排版，未重新进行数值拟合。原始冻结campaign及旧divergence.md不在本次编辑范围。

## 11. 2026-11-16：Quark PDF 的 Wilson-line 减除、共同有限质量与 hybrid 测试

### 11.1 目的、实际操作顺序与输入裸量

用户指定先使用目前已经提取的裸矩阵元进行下一步测试。本次按以下顺序执行：读取并核对冻结裸量；分别减除五个基本表示 Wilson-line 候选质量，并加入不输入线质量的控制；用三个格距的零动量数据联合拟合共同有限质量及低维截止效应；构造非零动量的归一化 hybrid 矩阵元；比较窗口、微扰处方、模型和因子；最后独立复核与归档。

通道为 unpolarized、connected、u-d，使用 P=0 决定短距拟合，输出 P=3、4 矩阵元。裸量是 self-ratio 常数平台提取的归一化形状：

$$
S_e(z,P)=\frac{h_{B,e}(z,P)}{h_{B,e}(0,P)},\qquad H_e(z)=S_e(z,0),\qquad S_e(0,P)=1.
$$

源估计量为同一组态 bootstrap 内、方向内先做复数 C3(z)/C3(0)，再做 VDV 投影与方向平均。每个 ensemble 沿用2000次组态 bootstrap、seed=1115、严格 T>0.8 fm，不重新生成样本。局域荷没有乘回。

| geometry | alias | a [fm] | Ncfg | 主裸量实际 T 窗口 |
| --- | --- | ---: | ---: | --- |
| L24x72 | C24P29 | 0.1050 | 199 | P0/P3：8–12；P4：8–9 |
| L32x64 | E32P29 | 0.0897 | 425 | P0/P3/P4：9–10 |
| L32x96 | F32P30 | 0.0775 | 194 | P0/P3/P4：11–12 |

主输入为 full/cut2，full/cut3 和 AIC/cut2 为控制。L24x72 的 P4 在 T=10–12 的局域分母稳定性检查失败，不能将其8–9窗口标成8–12。原质量标记、fallback 和全部有限 bootstrap 样本保留。

上游限制：L24x72、L32x96 的 P0 全窗平台没有通过 Q 门；L32x64 P0 的全窗拟合仅保留23个协方差模、20个参数、自由度3，不能仅据较高 Q 判断平台物理可靠。这些限制随全部下游输出保留。


裸量来源：[统一裸量 README](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/self_ratio_plateau_3ens_unpolarized_ud_P034_Tmin0p8_N2000_20261116/README.md)。


### 11.2 Wilson-line 候选、单位和减除符号

来源是第8节3D HYP5、Landau start0、基本表示、block_size=1、固定物理长度的系数。每个场景使用一个共同无量纲 c 作用于三个 ensemble，另外加入 c=0 控制。没有将4D HYP10、伴随表示、flowed 系数或旧 Cornell 候选0.1874混入本次输入。

$$
\ln\operatorname{Re}\langle W_F(z,a)\rangle=-c(z_{\rm ref})\frac za+B(z_{\rm ref}).
$$

定义自然单位

$$
\hbar c=0.1973269804\ \mathrm{GeV\,fm},\qquad
s=\frac{z_{\rm fm}}{\hbar c},\qquad a_{G,e}=\frac{a_{e,\rm fm}}{\hbar c},\qquad
\delta_e=\frac{c}{a_{G,e}}=\frac{c\hbar c}{a_{e,\rm fm}}.
$$

c无量纲，delta为GeV。对于裸量中负的线性指数衰减，减除时乘正指数：

$$
S_{{\rm sub},e}(z,P)=e^{+\delta_e s}S_e(z,P)=e^{+cz/a_e}S_e(z,P).
$$

这是基本表示单条开放线的约定，不额外乘二。c=0表示不输入外部 Wilson-line 质量，并不假设原裸量没有线性发散。


| 场景 | zref [fm] | c | c的统计SD | WL三格距拟合Q |
| --- | ---: | ---: | ---: | ---: |
| c0 | 控制 | 0.000000000 | 0.000000000 | — |
| wl030 | 0.3 | 0.093687576 | 0.000248418 | 0.20632830707913397 |
| wl045 | 0.45 | 0.110745221 | 0.000340452 | 1.6193489930773496e-11 |
| wl060 | 0.6 | 0.121690093 | 0.000426620 | 0.0004443792184257828 |
| wl075 | 0.75 | 0.129192717 | 0.000516602 | 0.00010144327120869145 |
| wl090 | 0.9 | 0.134023341 | 0.000621641 | 4.217497857024039e-08 |

五个系数随参考长度变化远大于各自统计SD，所以不能依据其中一个较高Q将其批准为唯一UV常数。Wilson line 与 quark 算符之间的定义、规范固定方案和三个 ensemble 的 HYP 参数匹配仍需核验。

### 11.3 共同短距模型、对数项与插值

非零锚点为 zstar=0.15 fm，主目标网格为0.20、0.25、0.30 fm，步长0.05 fm；短距上限0.25和0.35 fm作为控制。对每个 ensemble 的原生 log H 做分段线性插值，不外推。令

$$
\mathcal D_e F=I_e(z)F-I_e(z_\star)F.
$$

数据、ln C0及所有模型基函数使用同一个原生网格插值算子：

$$
Y_e(z)=\mathcal D_e[\ln H_e]+\delta_e(s-s_\star),
$$

$$
Y_e(z)=\mathcal D_e[\ln C_0(s,\mu)]
+m_{\rm fin}(s-s_\star)+a_{G,e}^2 f_1(s-s_\star).
$$

三个 ensemble 共用两个参数，mfin单位GeV、f1单位GeV^3，无先验。每个候选c重新拟合，不能为每个 ensemble 独立调mfin吸收输入质量。不能直接在目标网格算C0而遗漏它与数据之间的插值一致性；插值零模必须保留。

旧 smooth ZR 短距模型中的以下对数归一化与z无关：

$$
N(a;\Lambda,d)=\frac{3C_F}{\beta_0}
\ln\!\left[\frac{-\ln(a_G\Lambda)}{\ln(\mu/\Lambda)}\right]
+\ln\!\left[1+\frac{d}{\ln(a_G\Lambda)}\right],\qquad
\mathcal D_e N=0.
$$

因此本次没有把这一项隐含吸收到a²中，也不能由锚点比值重新确定Lambda、d或绝对ZR归一化。旧ansatz以外可能存在的z依赖对数格距项没有单独确定，f1不应被理解成已经独立测出的纯截止误差。

微扰处方分别为NLO、RGR NLO、RGR+LRR NLO，沿用既有physics.py，mu=2 GeV、alpha_s(mu)=0.296、nf=3。原生支撑可超过目标端点：主0.30 fm网格在三个 ensemble 需要的原生范围分别为0.105–0.315、0.0897–0.3588、0.0775–0.3100 fm。0.35 fm网格在L24x72需到0.42 fm，此处RGR运行耦合超出helper检查范围，相关组合标为不可用，不跨过该检查制造结果。

### 11.4 统计与可辨识性

每个ensemble内部保留完整跨z协方差，原始跨ensemble块设为零：

$$
C=\operatorname{diag}(C_{24},C_{32c},C_{32f}).
$$

对每块log数据的原始相关矩阵做硬特征模截断，阈值为最大特征值乘10^-10。固定中心whitening作用于全部replicas，不做shrinkage、不把已知插值零模抬升。要求模型满秩、正自由度。主模型有效秩8、参数2、自由度6。

质量门为

$$
\chi^2/\mathrm{dof}<1000,\qquad Q>10^{-6}.
$$

误差取全部2000 bootstrap的SD(ddof=1)，不除sqrt(2000)、不剪尾。不同ensemble的相同seed或replica行号不具有物理配对意义。联合产品使用预先固定seed=20261116的独立行置换，保持每个ensemble内部所有量同步；拟合协方差始终采用严格独立块，不使用有限样本中的伪跨ensemble协方差。置换及原始bootstrap行号落盘。

Wilson-line的JK行不与PDF的2000 bootstrap行配对。本次主误差是给定c的PDF条件统计误差；c±其统计SD对应的确定性重新拟合敏感性单独保存，不武断与PDF误差平方相加。

### 11.5 主模型结果及质量

下表为主0.30 fm窗口、full/cut2、a²线性截止项的两种主要场景。误差为给定c的PDF bootstrap SD。Q在双精度下可能下溢为0，其高精度log10值见联合结果报告。


| 微扰处方 | 场景 | mfin [GeV] | f1 [GeV³] | chi²/dof | Q（双精度） |
| --- | --- | ---: | ---: | ---: | ---: |
| nlo | c0 | -0.610158 ± 0.001447 | 0.632938 ± 0.005655 | 393.747 | 0 |
| nlo | wl030 | -0.300271 ± 0.001447 | 0.159188 ± 0.005655 | 367.507 | 0 |
| rgr_nlo | c0 | -0.942993 ± 0.001447 | 0.561524 ± 0.005655 | 2224.682 | 0 |
| rgr_nlo | wl030 | -0.633106 ± 0.001447 | 0.087774 ± 0.005655 | 2279.000 | 0 |
| rgr_lrr_nlo | c0 | -0.664470 ± 0.001447 | 0.647375 ± 0.005655 | 5579.753 | 0 |
| rgr_lrr_nlo | wl030 | -0.354583 ± 0.001447 | 0.173626 ± 0.005655 | 5481.525 | 0 |

主系列包含六个c场景、窗口/输入/模型控制，共162个固定c拟合，通过数为0。额外六个完整profile因RGR耦合超限不可用，每个profile对应六个c场景。

控制包括分别改为a线性截止项、a²二次截止项、full/cut3和AIC/cut2。NLO、c=0.0936876下，主线性模型chi²/dof=367.507；a²二次模型主窗口降到57.972，0.25 fm短窗降到22.850，但后者Q=8.73×10^-15，仍不通过。

自由c的附加模型在NLO主窗口给c=0.2076±0.0138，缩到0.25 fm却给c=-0.0570±0.0199，而且模型Q均极低。因此不能将该自由c解释为可靠的UV系数提取。

### 11.6 共同物理z²项的补充检查

为了检查共同短距曲率是否被错归于有限质量或截止效应，另增加一个共同物理幂次项：

$$
Y_e-\mathcal D_e[\ln C_0]
=m_{\rm fin}\Delta s+a_G^2f_1\Delta s+q_2\mathcal D_e[s^2],\qquad [q_2]=\mathrm{GeV}^2.
$$

q2是共同短距修正，不是UV重整化项，构造hybrid时没有减掉它。该系列42个固定c拟合中4个通过用户指定的宽松数值门；均为RGR+LRR、0.25 fm短窗：


| 候选 | c | mfin [GeV] | q2 [GeV²] | chi²/dof | Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| wl045 | 0.110745221 | -0.560342 ± 0.003147 | 0.140924 ± 0.000890 | 9.195 | 4.43556e-06 |
| wl060 | 0.121690093 | -0.524673 ± 0.003147 | 0.141030 ± 0.000890 | 8.426 | 1.35141e-05 |
| wl075 | 0.129192717 | -0.500223 ± 0.003147 | 0.141103 ± 0.000890 | 8.006 | 2.47823e-05 |
| wl090 | 0.134023341 | -0.484480 ± 0.003147 | 0.141149 ± 0.000890 | 7.781 | 3.42324e-05 |

同一c=0.134023候选在0.30 fm主窗口的chi²/dof=8.971、Q=1.55×10^-8，仍不通过。因此共同曲率明显改善了模型，但目前没有得到主窗口与上游均可靠的结果；四个窄窗过门组合不等于最终系数批准。162+42=204个固定c组合共4个过门，13个c自由诊断另计。

### 11.7 两种Zeff究竟如何得到

以下对应当前主结果mass-only hybrid；三种微扰处方分别使用各自拟合的mfin。约定

$$
S_{{\rm ren},e}(z,P)=\frac{S_e(z,P)}{Z_{{\rm eff},e}(z)}
=K_e(z)S_e(z,P),\qquad K_e(z)=\frac1{Z_{{\rm eff},e}(z)}.
$$

固定共同拼接点zs=0.30 fm：

$$
Z_{{\rm eff},e}(z)=
\begin{cases}
H_e(z),&z\le z_s,\\
H_e(z_s)\exp\!\left[-(\delta_e-m_{\rm fin})\frac{z-z_s}{\hbar c}\right],&z>z_s.
\end{cases}
$$

短距直接使用同一P0裸量，长距由同一H(zs)与不同净质量决定。两侧在zs处连续，z=0归一为1，P=3、4在同一ensemble使用相同因子。H(zs)使用原生log H插值，分子、分母、mfin按同一ensemble样本相关传播。

**不输入WL的控制**：c=0、delta=0，先独立联合拟合得到该场景mfin，再构造

$$
Z_{{\rm eff},e}^{(0)}(z>z_s)=H_e(z_s)
\exp\!\left[m_{\rm fin}^{(0)}\frac{z-z_s}{\hbar c}\right].
$$

NLO的共同有限质量为-0.610158±0.001447 GeV。这个负有效有限质量仍会使Zeff衰减；c=0不表示没有重整化，也不表示裸量没有线性发散。

**输入WL的场景**：c=0.0936875764，delta随格距按1/a变化；重新联合拟合得到NLO的mfin=-0.300271±0.001447 GeV，再构造

$$
Z_{{\rm eff},e}^{({\rm WL})}(z>z_s)=H_e(z_s)
\exp\!\left[-\left(\frac{c\hbar c}{a_e}-m_{\rm fin}^{({\rm WL})}\right)
\frac{z-z_s}{\hbar c}\right].
$$

本次两种方案使用各自拟合的mfin，不能仅在同一个既有重整化因子上额外乘一个WL指数来复现全部差别。NLO净质量为：


| ensemble | WL的delta [GeV] | 控制的 $-m_{\mathrm{fin}}^{(0)}$ [GeV] | WL的 $\delta_e-m_{\mathrm{fin}}$ [GeV] |
| --- | ---: | ---: | ---: |
| L24x72 | 0.176067 | 0.610158 | 0.476339 |
| L32x64 | 0.206099 | 0.610158 | 0.506370 |
| L32x96 | 0.238543 | 0.610158 | 0.538814 |

a²项参与了短距mfin拟合；主图没有将它额外外推到长距指数。另存的cutoff_tangent产品才使用

$$
\exp[-a_G^p F'(s_s)(s-s_s)]
$$

对长距作边界斜率诊断修正，不把短距二次多项式无条件外推到全部长距。

这些Zeff是归一化hybrid shape的有效因子，不包含局域荷，也不等同于含绝对归一化的完整MSbar ZR。锚点相消后没有重新确定Lambda、d；共同有限质量和候选线质量的分离仍带模型及方案依赖。

### 11.8 两方案的因子比值与条件误差相消

记每个ensemble的净质量为

$$
n_e^{(0)}=-m_{\rm fin}^{(0)},\qquad
n_e^{({\rm WL})}=\delta_e-m_{\rm fin}^{({\rm WL})}.
$$

则

$$
\frac{Z_{{\rm eff},e}^{({\rm WL})}(z)}{Z_{{\rm eff},e}^{(0)}(z)}=
\begin{cases}
1,&z\le z_s,\\
\exp\!\left[-(n_e^{({\rm WL})}-n_e^{(0)})\frac{z-z_s}{\hbar c}\right],&z>z_s.
\end{cases}
$$

两个固定c模型使用相同线性设计、中心协方差与相同PDF样本。改变固定c使线性GLS参数发生确定平移，其bootstrap随机贡献相同。因此该比值的P0因子与共同mfin涨落相消，PDF条件SD接近机器精度；实测最大约9.17×10^-14。这是当前固定c线性模型下的条件结果，不是完整重整化不确定度为零。

图中灰色点线另给c±其统计SD重新拟合后的敏感性；没有把WL的JK行与PDF bootstrap强行配对。不同参考长度、窗口、微扰处方、共同物理幂次、上游平台和HYP方案差异仍是独立的问题。

### 11.9 用颜色区分的矩阵元与因子结果

新图统一约定：蓝色为c=0控制，橙红色为zref=0.30 fm的WL输入；实心圆表示P=3，空心正方形表示P=4。方案和动量略作横向错开。矩阵元图实部范围[-0.5,1.5]、虚部[-1.0,0.5]。

因子比较图每行一个ensemble，三列依次为Zeff、乘到裸量上的K=1/Zeff、两方案Zeff之比。前两列使用对数纵轴，阴影为给定c下完整2000 bootstrap的±1 SD。NLO在共同物理位置z=1.00 fm的结果为：


| ensemble | Zc0 | ZWL | ZWL/Zc0 | Kc0 | KWL |
| --- | ---: | ---: | ---: | ---: | ---: |
| L24x72 | 0.08204877 | 0.13189768 | 1.607552 | 12.187873 | 7.581635 |
| L32x64 | 0.07848030 | 0.11341185 | 1.445100 | 12.742052 | 8.817421 |
| L32x96 | 0.07465374 | 0.09615358 | 1.287994 | 13.395176 | 10.400028 |

NLO中，输入WL后净质量更小，长距Zeff衰减更慢，所以乘到裸量上的K更小。短距两种Zeff按定义相同。不能将这一接近或差别直接解释为某个c已获独立物理验证。

为区分纯线性减除与完整hybrid，另画

$$
Z_{{\rm line},e}(z)=e^{-cz/a_e},\qquad
S_{{\rm sub},e}=S_e/Z_{{\rm line},e}.
$$

该图c=0为1，WL场景为指数衰减；不包含有限质量和hybrid拼接。


- [新的NLO矩阵元颜色图](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/plots/hybrid_nlo_scheme_colors.png)

- [NLO的Zeff、K及比值](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/plots/Zeff_comparison_nlo.png)

- [RGR因子比较](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/plots/Zeff_comparison_rgr_nlo.png)

- [RGR+LRR因子比较](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/plots/Zeff_comparison_rgr_lrr_nlo.png)

- [仅线性减除因子](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/plots/line_only_Z_comparison.png)

- [图像集中浏览](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/index.html)


### 11.10 结果文件、数值复核与复现入口

裸量、主测试及颜色/因子比较是三个独立产物目录。源裸量和主测试的冻结文件未被绘图改写。


- [统一裸量数据与格式](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/self_ratio_plateau_3ens_unpolarized_ud_P034_Tmin0p8_N2000_20261116/README.md)

- [联合测试结果报告](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/RESULTS.md)

- [联合测试模型、格式及读取示例](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/README.md)

- [固定c完整参数表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/artifacts/fit_parameters.csv)

- [共同z²控制参数表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/artifacts/common_power_control.csv)

- [c自由诊断](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/artifacts/free_c_diagnostics.csv)

- [联合矩阵元完整bootstrap目录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/artifacts/matrices)

- [联合独立验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/artifacts/independent_validation.json)

- [联合测试冻结记录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Plateau_3ens_20261116/artifacts/complete.json)

- [颜色比较与Zeff的说明](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/README.md)

- [公共物理位置的Zeff/K数值表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/renormalization_comparison.csv)

- [各处方的质量参数表](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/parameter_comparison.csv)

- [因子bootstrap数据](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/data)

- [因子重建验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/validation.json)

- [绘图比较完成记录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/WilsonLine_Joint_Comparison_20261116/complete.json)


原生因子NPZ的case顺序为c0、wl030；Zeff、K维度为(case,z)，bootstrap为(2000,case,z)，两方案比值bootstrap为(2000,z)。bootstrap_original_row保存原裸量行号，source SHA256在input_lock.json中。各因子用于同一ensemble的两个动量，不能将不同ensemble的行号解释为原始配对。


联合测试独立重建了216个矩阵元文件，检查66,816,000个bootstrap标量及217个拟合文件，最大重建误差为6.39e-14。颜色/因子比较再次核对30个源文件SHA256，重建2,784,000个矩阵元bootstrap标量，最大差1.42e-14。这些是数值正确性检查，不替代物理拟合质量。


联合任务4013622完成主计算，4013628完成共同z²控制与独立核对，4013632完成补充矩阵元、核对及报告；首次4013620因混合元数据记录缺少z_fm字段退出，修复筛选后重新运行，不使用失败产物。绘图重排没有重新拟合。

### 11.11 当前结论与后续顺序

1. 三个格距联合使用共同有限质量，可以使共同无量纲c的1/a变化不再被每个ensemble独立有限质量完全吸收；但共同有限质量和低维截止项仍会吸收其中部分变化。
2. 当前主线性模型质量失败，不能把很小的条件统计SD作为参数已精确确定的依据。加入共同z²项明显改善曲率，只有四个0.25 fm短窗组合通过宽松数值门，主窗口仍未通过。
3. 先核实P0平台、激发态污染及协方差截断稳定性，再核实Wilson-line与quark算符的HYP/方案一致性；不能将全部ensemble差异都称作离散效应。
4. 已保存全部诊断矩阵元和相关bootstrap，所有全链质量状态仍为false；生产smooth ZR、局域荷处理和后续Fourier/matching流程没有据此替换。
5. 截至2026-11-16，仍无由本测试批准的唯一最终线性发散系数。


本次文档更新的原文备份、生成脚本、输入哈希、差异与安装核对记录：[2026-11-16文档更新目录](/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Renorm/documentation_updates/linear_devergence_20261116)。

## 12. 2026-11-18至19日：长距离局部斜率、参考流时间与单参数修正测试

### 12.1 目的、观测量与计算批次

本轮回答三个问题：原拟合巨大的chi2是否主要来自长度窗口选择；改变tau_ref是否改变结果；提高z_min后，保持原参数数目或只增加一个参数，能否得到合理且稳定的拟合。

仍使用第6节的三组全量Landau规范固定、start0、四个flow的开放Wilson line，C24P29/E32P29/F32P30的Ncfg分别为976/1700/887，a分别为0.105/0.0897/0.0775 fm，长度n=z/a=0..12。四个共同物理流半径约为0.138962642、0.160460238、0.179400000、0.196522854 fm。基础与伴随表示分别使用实测值，不通过Casimir比例互相构造。

这些是规范固定开放线，不是闭合Wilson loop，也不是带夸克外态的双线性算符。没有重新生成规范场或测量新线长；以下数值均来自已完成的CPU Slurm分析。

| 日期 | 计算内容 | job | 主要规模 |
| --- | --- | ---: | --- |
| 2026-11-18 | 全长度窗口、flow集合与参考点审计 | 4025743 | 15,840个拟合、720个参考点检验；复现原648个拟合 |
| 2026-11-18 | 长距离局部斜率、有限z/r_f与flow对差异 | 4025773 | 7,020个联合拟合、10,260项配对比较、4,104个单对平台拟合、522个留一对检验 |
| 2026-11-19 | 不增加参数、固定上限后提高z_min | 4027457 | 792个拟合、666个相邻窗口配对比较 |
| 2026-11-19 | 相同窗口上分别增加一个有限长度或flow_c参数 | 4027464 | 1,584个拟合、1,332个相邻窗口配对比较、252个留一flow差分预测 |

全部冻结记录均明确final_coefficient_approved=false。下文的“通过”仅指指定统计诊断门，并不表示最终物理批准。

### 12.2 统计契约与参考点审计

估计量仍为先组态平均再取对数，即ln[Re<W>]，不是<ln W>；均值或JK均值非正时不取绝对值、不裁剪。所有flow、长度和同一ensemble内的比较共享组态索引。使用delete-block JK，block=1/2/5；不足整块的尾部及实际conf分组均保存。协方差为

$$
C=\frac{N_B-1}{N_B}\sum_b(x_b-\bar x)(x_b-\bar x)^T.
$$

不同ensemble不拼接成配对副本。block2/5只是blocking敏感性检查，不代表自相关已被充分控制。参数误差来自固定JK协方差的GLS线性传播，并由共享JK拟合副本复核；不除以sqrt(N_B)，也不按chi2人为放大。

11月19日主结果沿用原block1与相关矩阵删模cut=1e-8；严格全协方差及cut=1e-6作为敏感性检查。这里正cut表示明确删除小相关本征模，不是gvar正cut抬升本征值的语义。自由度为保留数据模数减自由参数数。原始样本秩、模型秩和正自由度均检查；严格全协方差不静默改成伪逆或对角拟合。Q为插件协方差的chi-square近似，Q>=0.01只作沿用的诊断门；选窗为探索性，未校正多重比较/选择效应。无随机抽样，seed不适用。

固定相同四个flow、相同长度及同一协方差子空间时，更换tau_ref只是可逆差分换基，物理参数d和chi2应不变。实测审计中d的最大差约4.6e-13，chi2相对差约7.7e-12。因此原巨大chi2不是简单“参考流时间选错”造成的。

若每次换参考后独立重新删除相关矩阵本征模，实际保留的数据子空间可能变化。例如E32P29基础表示原主窗口、cut=1e-6下，d可从tau_ref=0.3时的0.333499变到tau_ref=0.6时的0.346478；传输同一保留模时该变化消失。只保留较大/较小三个flow或单个flow对则真正改变了数据集合，不能称作纯参考点效应。

### 12.3 全窗口审计、局部斜率与有限长度动机

首次审计扫描n=1..12的55个连续窗口，每窗至少三个长度，并检查all4/lower3/upper3、原leading/flow_c模型、两种cut及三个block。在基础表示的all4共1,980个组合中，没有Q>=0.01的结果。block1/cut=1e-8下，原leading最小chi2/dof分别为C24P29 84.30、E32P29 5.070、F32P30 48.53，均在只有三个长度的n=10..12窗口，仍失败；C24的n=12还已触及半盒。

为区分长度曲率与flow对差异，定义相邻线段上的局部系数

$$
d_{{\rm eff},ij}(z)=
-\frac{[\ln W_i(z+a)-\ln W_j(z+a)]-[\ln W_i(z)-\ln W_j(z)]}
{a(1/r_i-1/r_j)}.
$$

六个flow对不是六组独立数据。联合拟合只使用线性独立的03、13、23差分，并保留其交叉协方差；其余对用于展示和同组态配对比较。flow编号按四个流半径从小到大为0..3。

局部斜率主窗口要求每段左端点z_L/r_max>=3，并排除C24半盒端点：C24为0.630–1.155 fm、5个斜率位置；E32为0.6279–1.0764 fm、5个斜率位置；F32为0.620–0.930 fm、4个斜率位置。在block1、严格全协方差下，主窗口36个单flow对常数平台检验全部Q<0.01；要求所有独立flow对共享平台的原模型同样失败。基础表示主窗口不同flow对长段割线系数最大配对差分别约19.98、2.19、10.36 sigma；长段平均不能冒充已验证的平台。

有限长度形状的动机来自该批次保存的连续一圈Landau开放线推导：在共同耦合近似下，局部质量核的长距离展开包含

$$
K(r,z)=\frac1r-\frac{3}{2\sqrt\pi\,z}
+\frac{r^2}{2\sqrt\pi\,z^3}+\text{指数小项}.
$$

共同1/z项在两个flow之差中消去，得到局部系数的首个剩余形状

$$
d_{{\rm eff},ij}=d_\infty+kH_{3,ij},\qquad
H_{3,ij}=\frac{r_i^2-r_j^2}{1/r_i-1/r_j}
\frac{z_L^{-2}-z_R^{-2}}{2(z_R-z_L)}.
$$

线段使用精确有限差分/积分平均，不用中点值代替。一圈共同耦合给出k=d_infinity/(2sqrt(pi))，但下文自由幅度测试不施加这一关系，也不约束k为正。该动机不是全阶定理；当前r_f/a仅约1.3–2.5，尚未包含完整格点微扰、高阶和非微扰长距离效应。

作为固定形状敏感性结果，E32伴随表示在0.8073–1.0764 fm、三个斜率位置上，固定一圈核、只拟合一个共同幅度时，d_infinity=0.81982648±0.00077989、chi2/dof=0.789163、Q=0.612，留一flow对预测也通过。它与“原模型只改窗口”不是同一测试，不能因自由参数数相同而混称。完整推导及其他核/幂次模型见本节末链接。

### 12.4 不增加参数：固定上限后提高z_min

使用第6节原leading模型，四个flow及最大flow参考固定，每次仍只有共同d和三个B_i，共4参数。每条序列内仅提高下限、每次去掉最短一个长度，不改变该序列的上限。

| ensemble | 原z_max=0.7 fm的实际离散上限 | 使用已有长距离数据的固定上限 |
| --- | ---: | ---: |
| C24P29 | 0.6300 fm | 1.1550 fm，排除n=12半盒点 |
| E32P29 | 0.6279 fm | 1.0764 fm |
| F32P30 | 0.6975 fm | 0.9300 fm |

**严格保持原0.7 fm上限，仅提高下限，三套ensemble、两种表示直到只剩两个长度仍没有Q>=0.01的窗口。**第二类扫描使用已有更长数据，不能称作只改变原[0.3,0.7] fm窗口的下限。C24另存包含1.26 fm半盒点的诊断序列，不混入避开半盒的结论；避开半盒本身也不等于有限体积已受控。

E32P29固定z_max=1.0764 fm的主结果如下，N_z是不同线长点数，不是flow数：

| 表示 | z_min [fm] | N_z | d ± JK stat | chi2/dof | Q |
| --- | ---: | ---: | --- | ---: | ---: |
| 基础 | 0.8073 | 4 | 0.36396865 ± 0.00016073 | 30.8474 | 8.30e-49 |
| 基础 | 0.8970 | 3 | 0.36634294 ± 0.00023577 | 5.07043 | 0.000119 |
| 基础 | 0.9867 | 2 | 0.36790295 ± 0.00046276 | 1.09509 | 0.335 |
| 伴随 | 0.8073 | 4 | 0.81441791 ± 0.00077476 | 3.06258 | 0.00189 |
| 伴随 | 0.8970 | 3 | 0.81836258 ± 0.00147636 | 0.782719 | 0.562 |
| 伴随 | 0.9867 | 2 | 0.82347018 ± 0.00418714 | 0.542602 | 0.581 |

基础表示从三长度缩到两长度，d仍有3.92 sigma配对漂移。伴随表示从四长度缩到三长度也有3.14 sigma漂移，再缩到两长度为1.30 sigma。因此原模型的E32伴随三长度窗可作为窄窗候选，但没有建立宽范围稳定平台。C24与F32两种表示的避开半盒三长度窗均失败；C24伴随只剩两个长度时过门，F32两种表示两长度窗仍失败。

两个长度时每个flow对只有一个相邻斜率位置，虽可检验不同flow间的共同系数关系，却不能检验沿z的曲率。原模型三长度窗有9个相关差分数据、4参数、dof=5；两长度窗有6个数据、dof=2。不能将自由度为正或窄窗高Q直接当作局部平台证据；单个长度无法区分d与三个截距，故不拟合。

### 12.5 提高z_min后，只增加一个共同修正参数

主测试finite_power3在原模型上只增加一个无量纲共同幅度k：

$$
D_i(z)=-d z\Delta u_i+B_i+\frac{k(r_i^2-r_3^2)}{2z^2},
\qquad \Delta u_i=1/r_i-1/r_3,\quad i=0,1,2.
$$

该项精确对应12.3节局部斜率的kH3参数化。每次共5个参数，即d、三个原有截距和一个k；没有为每个flow分别增加修正，也不加先验或Casimir约束。保持同一窗口、同一flow集合、同一白化协方差子空间，才比较与基线的chi2下降。

E32P29固定z_max=1.0764 fm，block1/cut=1e-8；以下窄窗亦通过严格全协方差复核：

| 表示 | z_min [fm] | N_z | 原chi2/dof | 新chi2/dof | 新Q | d ± JK stat | k ± JK stat |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 基础 | 0.8073 | 4 | 30.8474 | 2.70539 | 0.00838 | 0.36846623 ± 0.00033855 | 0.132746 ± 0.008794 |
| 基础 | 0.8970 | 3 | 5.07043 | 1.78729 | 0.128 | 0.36890946 ± 0.00064610 | 0.143188 ± 0.033561 |
| 伴随 | 0.8073 | 4 | 3.06258 | 0.341700 | 0.935 | 0.82375470 ± 0.00213150 | 0.399532 ± 0.084971 |
| 伴随 | 0.8970 | 3 | 0.782719 | 0.476875 | 0.753 | 0.82600464 ± 0.00559388 | 0.523364 ± 0.369512 |

增加一个参数后，三长度窗dof=4、四长度窗dof=7。基础表示四长度窗Q=0.00838仍在沿用门槛以下，不因接近门槛而事后修改规则；三长度窗则得到实质改善。该三长度窗降低chi2=18.203，k相对其条件统计误差约4.27 sigma，但改善显著本身不证明全部失配都来自这一物理机制。

### 12.6 配对稳定性、blocking/cut与留一flow差分预测

两个窗口共享组态，差值误差必须由两次拟合的共同JK副本之差计算，不能把各自误差独立相加。E32修正后，从z_min=0.8073提高到0.8970 fm：

| 表示 | delta_d | 配对误差 | 配对pull | 原模型同一移动的pull |
| --- | ---: | ---: | ---: | ---: |
| 基础 | 0.00044323 | 0.00055030 | 0.81 | 13.76 |
| 伴随 | 0.00224995 | 0.00517187 | 0.44 | 3.14 |

基础表示三长度候选在全部block1/2/5、cut=1e-8/1e-6与严格全协方差检查下，Q=0.128–0.147，d=0.368904–0.368926；伴随表示四长度候选Q=0.935–0.957。基础三长度窗d与k相关系数约0.931，d的统计误差从原模型0.000236增加到0.000646，必须保留这一误差增长。相邻窗口的中心稳定只说明该项能吸收部分漂移，不能将一个通过窗与相邻失败窗解释为已经有多个通过的宽平台。

另在block1、严格全协方差下，将03、13、23三个线性独立差分中的一个留出，用另外两个拟合d与修正幅度，再预测留出差分。预测残差协方差通过共享JK计算，保留训练参数与测试数据之间的交叉相关：

| 候选 | 留出03预测Q | 留出13预测Q | 留出23预测Q |
| --- | ---: | ---: | ---: |
| E32基础，0.8970–1.0764 fm，3长度 | 0.309 | 0.0734 | 0.253 |
| E32伴随，0.8073–1.0764 fm，4长度 | 0.799 | 0.634 | 0.989 |

基础候选相应训练Q为0.0848、0.538、0.239。以上均过门，但这是flow差分层面的预测检查，不是独立规范组态或独立ensemble验证。

### 12.7 同样只加一个参数的flow_c对照与其他ensemble

单独的等参数数目对照使用第6节flow_c模型：

$$
D_i(z)=-d z\Delta u_i+B_i-c_{\rm flow}z(t_i-t_3),
$$

c_flow单位fm^-3。它与有限长度项不同时使用，每次也只比基线增加一个共同参数。E32基础表示0.8970–1.0764 fm三长度窗，flow_c得到chi2/dof=5.05641、Q=0.000451，仍失败；有限长度项为1.78729、Q=0.128。因此不是“任意增加一个参数都能过门”，新增项的形状很重要。E32伴随四长度窗flow_c亦可过门，Q=0.0256，但属于另一模型敏感性结果，不按最大Q事后更换主模型。

其他ensemble的避开半盒最长三长度窗：

| ensemble/表示 | 有限长度模型chi2/dof | Q | 判断 |
| --- | ---: | ---: | --- |
| C24基础，0.945–1.155 fm | 122.447 | 1.08e-104 | 失败；flow_c亦失败 |
| C24伴随，0.945–1.155 fm | 3.30806 | 0.0102 | 仅边缘过门，block/cut下Q=0.00876–0.0147，不稳健 |
| F32基础，0.775–0.930 fm | 28.1018 | 2.23e-23 | 失败；flow_c亦失败 |
| F32伴随，0.775–0.930 fm | 6.23116 | 5.21e-5 | 主有限长度模型失败；flow_c可给Q=0.0346，显示模型依赖 |

C24伴随的k=-6.226±1.010，符号与简单一圈共同耦合动机相反，不能只凭边缘好Q批准d。严格保持原0.7 fm上限且至少保留三个长度时，两种单参数扩展在全部block/cut下仍无过门结果；有用的E32候选依赖已有更长数据。

新增参数后两长度窗仅dof=1，不作平台验证。对本次两个扩展，两个长度时z^-2列可由常数和z列表示，而r_i^2-r_3^2=8(t_i-t_3)，因此两种扩展张成相同模型子空间，保存结果的chi2、d和d误差一致。这种两点等价性不能区分曲率形状，更不能单独证明真实渐近系数。也不能把各ensemble不同模型下各自最高Q的结果拼成受控连续极限。

### 12.8 诊断候选、数值验收与源文件

在本次“提高下限并只增加一个有限长度幅度”的约定下，优先保留两个E32诊断候选：

$$
d_F=0.36891\pm0.00065_{\rm stat},\qquad
z=0.8970\text{--}1.0764\ {\rm fm},
$$

$$
d_A=0.82375\pm0.00213_{\rm stat},\qquad
z=0.8073\text{--}1.0764\ {\rm fm}.
$$

**这表示E32有限长度诊断模型获得支持，不表示统一最终紫外系数已确定。**误差以给定格距、模型、协方差处理和窗口为条件，未包含选窗、完整模型、定标、充分自相关控制、有限体积和连续极限系统误差。本节GF/Landau开放线d与第11节HYP候选c不是同一方案；没有据此替换Quark PDF线质量输入、生产smooth ZR、局域归一化、Fourier或matching产品。

数值验收：长距离局部斜率批次11项单元测试通过，并以独立Cholesky白化+缩放QR复核60个主拟合及36个端点割线。不加参数批次6项单元测试通过、444个旧窗口结果复现、90个严格全协方差结果与局部平台等价、264个独立QR验算。新增一个参数批次7项单元测试通过，792个基线白化残差复核、528个独立QR验算、90个局部有限长度结果和444个旧flow_c结果交叉复现，以及1,584项嵌套/Wald关系和共享JK误差检查通过。数值秩门通过、拟合质量通过与最终物理批准是三个不同层次。

源文件与复现入口：

- 长度/参考点：[报告](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_z_reference_audit_20261118/REPORT.zh-CN.md)、[全部窗口](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_z_reference_audit_20261118/window_fits.csv)、[参考点检验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_z_reference_audit_20261118/reference_tests.csv)、[冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_z_reference_audit_20261118/complete.json)。
- 局部斜率/有限r_f：[报告](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_long_slope_finite_rf_20261118/REPORT.zh-CN.md)、[独立理论推导](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_long_slope_finite_rf_20261118/THEORY.zh-CN.md)、[联合拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_long_slope_finite_rf_20261118/joint_fits.csv)、[flow对差异](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_long_slope_finite_rf_20261118/pair_comparisons.csv)、[冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_long_slope_finite_rf_20261118/complete.json)。
- 不增加参数：[结果与判断](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/RESULTS.zh-CN.md)、[完整契约和长表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/REPORT.zh-CN.md)、[全部拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/fits.csv)、[配对稳定性](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/window_stability.csv)、[基础表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/figures/zmin_fundamental.png)、[伴随表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/figures/zmin_adjoint.png)、[冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_scan_20261119/complete.json)。
- 新增一个参数：[结果与判断](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/RESULTS.zh-CN.md)、[模型与统计契约](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/METHOD.zh-CN.md)、[完整长表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/REPORT.zh-CN.md)、[全部拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/fits.csv)、[窗口配对漂移](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/window_stability.csv)、[留一flow预测](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/holdout_predictions.csv)、[基础表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/figures/one_extra_fundamental.png)、[伴随表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/figures/one_extra_adjoint.png)、[数值验算](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/validation.json)、[冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_zmin_one_extra_20261119/complete.json)。

各计算目录还保存config.json、code/、CPU Slurm脚本、日志、实际blocking索引和输入SHA256；11月19日两批保存拟合参数的共同JK副本fit_samples.npz。当前归档不重新运行拟合，也不修改这些冻结计算产物。

本次文档更新的原文备份、更新后副本、安装补丁、差异、输入哈希与安装核对记录：[2026-11-19文档更新目录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_20261119)。

## 13. 2026-11-19：C24P29/F32P30提高z_min与逐级参数扩展

### 13.1 数据、模型与本轮接受门

本轮接续第12节，只分析另外两个ensemble的已有Landau start0开放Wilson line：C24P29为976组态、a=0.105fm；F32P30为887组态、a=0.0775fm。基础表示F和伴随表示A分别分析，保留四个flow，参考固定最大flow。d仍为无量纲正单线质量系数，m=d/r_f，不含闭合圈的因子2。

共同基础模型有d与三个端点截距B_i。令u_i=1/r_i、Delta表示减去参考flow，本轮扩展的完整记号为

$$
D_i=-d z\Delta u_i+B_i
+k_3\frac{\Delta r_i^2}{2z^2}
+k_5\frac{\Delta r_i^4}{4z^4}
-c z\Delta t_i-\eta z a^2\Delta(u_i^3).
$$

实际不是同时拟合全部项，而是固定六种模型：H3（5参数）、H3+H5、H3+c、H3+eta（各6参数）、H3+H5+c、H3+H5+eta（各7参数）。H3即原k3有限长度项；H5只是经验敏感性项；eta是a²/r_f²型flow质量修正的诊断形状，不宣称完整格点微扰或已验证的Symanzik截断。c和eta作为替代形状，不同时拟合。k3、k5、eta无量纲，c单位fm^-3；不加先验、Casimir或跨ensemble约束。

固定上限逐格提高z_min。C24主长窗上限n=11（1.155fm），排除半盒n=12；F32上限n=12（0.930fm）。另保存原0.7fm上限序列和C24含半盒诊断序列。至少3长度才拟合；**本轮局部候选必须至少4长度**，还须主Q>=0.01、block=1/2/5与严格全协方差及删模cut=1e-8/1e-6全部9种设置可辨识且Q均>=0.01、三次独立flow差分留出训练/预测Q均>=0.01、d相对统计误差<=10%。

主数值沿用block=1、cut=1e-8。所有flow/长度共享delete-block JK，协方差为(B-1)/B delta.T@delta；相邻窗口和模型差使用共享JK差值，不能按独立量相加。留出检查保留训练/预测交叉协方差，不把六个flow对当独立样本，也不是独立组态的外部验证。上述门是本轮诊断规范，不是普适定理；Q未校正探索性模型/选窗效应。

### 13.2 每个ensemble的条件系数与拟合质量

以下误差只含固定模型、固定窗口下的JK统计涨落。总参数数见13.1；chi2/dof使用实际保留模数扣除参数数。

| ensemble | 表示 | 模型 | z范围(fm) | 长度数 | d ± stat | chi2/dof | Q |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| C24P29 | F | 六种模型 | 所有安全长窗 | >=3 | 不给可靠提取值 | 均失败 | <0.01 |
| C24P29 | A | H3+eta | 0.735–1.155 | 5 | 1.01622 ± 0.01019 | 0.8612 | 0.5594 |
| C24P29 | A | H3+H5+c | 0.840–1.155 | 4 | 0.67735 ± 0.01974 | 1.3767 | 0.2295 |
| C24P29 | A | H3+H5+eta | 0.840–1.155 | 4 | 0.96788 ± 0.02621 | 0.1739 | 0.9724 |
| F32P30 | F | H3+H5+c | 0.620–0.930 | 5 | 0.361870 ± 0.000946 | 1.9486 | 0.0692 |
| F32P30 | F | H3+H5+eta | 0.620–0.930 | 5 | 0.325517 ± 0.001393 | 2.2084 | 0.0392 |
| F32P30 | A | H3+H5+c | 0.5425–0.930 | 6 | 0.808001 ± 0.002252 | 1.7525 | 0.0635 |
| F32P30 | A | H3+H5+eta | 0.5425–0.930 | 6 | 0.731437 ± 0.002978 | 1.5572 | 0.1125 |

**C24P29基础表示仍失败。** 最窄安全3长度窗0.945–1.155fm，H3+eta的chi2/dof=5.6427、Q=0.000731；H3+H5+eta为6.4399、Q=0.001597。含半盒1.050–1.260fm三长度窗虽可得到Q约0.049/0.060，却不满足安全几何和至少4长度的候选门，不能拿来宣称成功提取。

**C24P29伴随表示有较简条件候选。** H3+eta在0.735–1.155fm的9种设置Q=0.391–0.559，三次留出预测Q约0.130、0.297、0.294。提高下限至0.840fm得d=0.98829±0.02163、Q=0.8375，配对漂移1.464sigma（相对变化2.75%）。但原窗d与eta相关系数约-0.980，且另有通过的不同质量形状模型，不能忽略模型歧义。

**F32P30基础表示只有单一下限通过全部门。** 两个七参数模型在0.620–0.930fm均通过；H3+H5+c的9设置Q=0.0149–0.0884，H3+H5+eta为0.0392–0.1380。相邻下限0.5425/0.6975fm虽有主Q过门结果，却未通过全部blocking或留出检查，不能称已建立稳定窗口区。

**F32P30伴随表示模型内较稳定。** H3+H5+c在下限0.5425→0.620fm时d=0.80800→0.80591，配对漂移0.853sigma（0.259%）；H3+H5+eta为0.73144→0.73656，1.612sigma（0.700%），再提高到0.6975fm得0.73916±0.00737，下一步漂移0.437sigma。较简H3+eta在0.620–0.930fm也有局部候选0.73614±0.00435、Q=0.0941。

### 13.3 为什么仍不能给出唯一最终系数

同数据、同窗口、同主设置，七参数模型H3+H5+c与H3+H5+eta均通过局部门，但系数差远大于共享JK涨落。差值定义前者减后者：

| ensemble/表示 | z范围(fm) | Delta d ± paired stat | 配对pull | abs(Delta d)/abs(d_c) |
| --- | --- | --- | ---: | ---: |
| C24P29 A | 0.840–1.155 | -0.29053 ± 0.02966 | 9.797 | 42.89% |
| F32P30 F | 0.620–0.930 | +0.036353 ± 0.002119 | 17.152 | 10.05% |
| F32P30 A | 0.5425–0.930 | +0.076564 ± 0.004707 | 16.266 | 9.48% |

pull仅量化两种固定模型下系数差异相对于共同样本涨落的大小，不是模型真假的显著性检验。**提高z_min并增加参数可以改善拟合质量，但尚不能唯一分离d/r_f与flow依赖线质量。** 不能挑最高Q模型、平均候选后只报统计误差，或把两模型包络视为已经穷尽的系统误差。

第12节E32P29的H3单参数候选仍为原模型/门槛下的结果，其中基础表示仅3长度，不满足本轮>=4长度门。不能将它与本节不同模型/窗口数值直接拼接做受控连续极限拟合；本轮未做此操作，也未更新smooth ZR或PDF减除。

### 13.4 作业、数值核验与归档入口

CPU作业4028022已COMPLETED、退出码0:0，耗时41秒；完成3024个拟合、504项留出检查、14个局部候选、5组相邻通过窗。6项单元测试通过；1008项独立QR、1008项局部斜率等价、504项旧H3与54项旧H3+H5复现、3024项JK误差检查均通过。

完成后独立读取保存产物，19个冻结产物SHA256、14个输入SHA256、3024个完整参数协方差、2484个窗口配对误差均通过；协方差最大相对偏差3.90e-7。新增解释报告和核验层没有改写原complete.json登记的冻结拟合产物。最终物理状态保持diagnostic only、final_coefficient_approved=false。

结果目录：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/`。主要入口：

- [结果与限制](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/RESULTS.zh-CN.md)、[模型和统计契约](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/METHOD.zh-CN.md)、[主结果长表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/REPORT.zh-CN.md)。
- [全部拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/fits.csv)、[候选及失败门](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/candidate_audit.csv)、[窗口配对漂移](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/window_stability.csv)、[留出预测](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/holdout_predictions.csv)、[同窗质量形状差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/flow_shape_contrasts.csv)。
- [基础表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/figures/CF_extended_fundamental.png)、[伴随表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/figures/CF_extended_adjoint.png)、[原冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/complete.json)、[独立落盘核验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_CF_extended_20261119/saved_output_verification.json)。

同目录还保存全部参数共享JK副本fit_samples.npz、代码、配置、实际blocking组态索引、输入SHA256、CPU Slurm脚本及日志。本节更新的原文备份、安装补丁、差异和核对记录：[C/F文档归档目录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_CF_20261119)。

## 14. 2026-11-20：三个ensemble统一H3模型并适当放宽候选条件

### 14.1 模型、扫描和分层规则

本轮只用已有Landau start0开放Wilson line，三ensemble原组态数976/1700/887，基础/伴随表示独立。全四flow统一使用五参数H3模型

$$
D_i=-d z(1/r_i-1/r_3)+B_i+\frac{k_3(r_i^2-r_3^2)}{2z^2},\qquad i=0,1,2.
$$

参数为d,B0,B1,B2,k3；不加c_flow、H5、eta或先验。保留连续一圈Landau开放线渐近形状，但k3独立且允许负值，不强制k3=d/(2sqrt(pi))。d沿用无量纲正单线质量m=d/r定义，不含闭合圈因子2。

扫描所有z_min>=0.3fm、至少3长度的连续窗口，两端独立移动；安全上限C24=1.155fm（排除半盒点）、E32=1.0764fm、F32=0.930fm。主分析保留全部四flow，另做四种删一flow的三flow子集诊断；后者只有4参数，不能冒充五参数全flow通过。

主设置block1、相关矩阵删模cut=1e-8，同时计算block1/2/5和严格全协方差None/删模1e-8/1e-6的全部9种设置。先平均再log，沿用共享delete-block JK及(B-1)/B协方差因子；不改对角协方差、不人为放大误差、不按Q调cut。所有组态blocking索引可追溯。

- strict：至少4长度，主Q及全部9设置Q>=0.01，四次真正删一flow后的训练和预测Q均>=0.01。
- working_relaxed：至少3长度，主Q及block1严格全协方差Q>=0.001，全部9设置均可辨识且Q>=0.0001；留出结果必须报告，但不再作硬门。
- 共同门：d>0、相对统计误差<=10%、z_min/r_max>=3、不含半盒点。秩门和正自由度不放宽。

展示点先strict、后working_relaxed；同层按物理跨度最大、长度数最多、上限最大排序，不按最高Q挑点。等宽物理跨度按1e-12fm舍入，避免浮点尾数影响排序。允许移动任一端点一格；同层或更好邻窗的配对pull<=3且相对变化<=3%才标邻窗稳定。规则在新扫描前声明，但本轮仍属看过历史数据后的探索性选窗，Q未校正选择效应。

### 14.2 全四flow工作系数与失败结果

以下误差仅为固定H3模型/窗口下的主设置JK统计误差，不是总误差。

| ensemble | 表示 | z范围fm | 长度数 | d ± stat | k3 ± stat | chi2/dof | Q | 判断 |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| C24P29 | F | — | — | 不给工作系数 | — | — | — | 所有安全长窗失败 |
| C24P29 | A | 0.9450–1.1550 | 3 | 0.75589 ± 0.01300 | -6.2263 ± 1.0104 | 3.3081 | 0.01019 | 放宽后仅单窗诊断 |
| E32P29 | F | 0.8073–1.0764 | 4 | 0.368466 ± 0.000339 | 0.13275 ± 0.00879 | 2.7054 | 0.008385 | 放宽后条件工作点 |
| E32P29 | A | 0.8073–1.0764 | 4 | 0.823755 ± 0.002131 | 0.39953 ± 0.08497 | 0.3417 | 0.9350 | 本轮strict候选 |
| F32P30 | F | — | — | 不给工作系数 | — | — | — | 所有安全长窗失败 |
| F32P30 | A | — | — | 不给通过门槛的工作系数 | — | — | — | 所有安全长窗失败 |

全四flow共168个主窗口，2个strict、9个working_relaxed。**适当放宽并没有得到三个ensemble完整的合理系数集。** 不应为了每组都有数值而继续降低门槛接受明显失败。

E32基础表示提高下限至0.8970fm时d=0.368909±0.000646、Q=0.1282，与主工作点配对漂移0.805sigma、相对变化0.120%；降低上限至0.9867fm时d=0.368647±0.000502、Q=0.02338，漂移0.488sigma。邻窗中心包络0.368466–0.368909。九设置最小Q=0.004424、d范围0.368466–0.368796，最大统计误差0.000564；留出最低训练/预测Q=0.001363/0.002569，故不是旧严格门下的稳健批准。

E32伴随主点九设置最小Q=0.9350，四次留出最低训练/预测Q约0.696/0.634；同窗删flow配对漂移均<0.87sigma。另一个strict窗0.7176–0.9867fm得d=0.820329±0.001311、Q=0.05637。主点直接相邻、满足本轮漂移门的窗口中心包络为0.819314–0.826005，不能把它们独立平均，也不能省略这项窗口敏感性。

C24伴随只有一个三长度窗、无相邻通过窗；九设置最小Q=0.008760，留出最低训练/预测Q=0.002015/0.003122。更重要的是k3=-6.226±1.010，而一圈共同耦合关系预期约+0.213，配对偏离约-6.40sigma；d/k3相关约0.967。同窗删掉最小flow使d由0.75589变成0.81402，差0.05813±0.01892（3.07sigma、相对变化7.69%）。因此只列脆弱诊断，不建议用于最终减除。

失败程度：C24基础最佳安全三长度窗chi2/dof=122.447、Q=1.08e-104；F32基础最佳为28.1018、Q=2.23e-23；F32伴随最佳为6.2312、Q=5.21e-5，均不能按本轮温和放宽接受。F32伴随失败窗虽有d=0.79525±0.00349，也不应重命名为通过的工作系数。

### 14.3 三flow结果不能替代全flow失败

仅保留最大三个flow（omit0），F32基础在0.775–0.930fm给d=0.342325±0.001212、chi2/dof=2.9761、Q=0.05099；伴随在0.620–0.930fm给d=0.769006±0.001482、chi2/dof=1.8491、Q=0.08550。

但对遗漏最小flow的预测Q分别只有1.87e-22和2.33e-20。这些只是限定flow区间的敏感性值，不能替代全flow兼容系数；也不能据此断言最小flow测量错误。C24基础删掉最小flow后的最佳Q也只有0.0001548，仍失败。

**本轮建议：E32保留d_F约0.3685、d_A约0.824作带窗口/统计敏感性说明的H3条件工作值；C24伴随仅脆弱诊断，其余三组不接受。没有形成三个ensemble统一减除或受控连续极限所需的完整系数集。**

### 14.4 数值验收与文件入口

初次作业4029368；等宽窗口浮点排序修正后作业4029375，模型/数据/阈值不变。最终完成7560个拟合、672项删一flow检查；全部flow集合合计10个strict和69个working_relaxed，不要误报成全四flow候选数。9项单元测试、7560项JK误差检查、2520项独立Cholesky+缩放QR、504项局部斜率等价、522项旧H3窗口复现均通过。

后验独立读取13个输入SHA256、21个冻结产物SHA256、7560组d/k3副本误差、1260项窗口配对差与12项同窗删flow差。最终状态仍为diagnostic only、final_coefficient_approved=false，不更新smooth ZR。

目录：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/`。

- [结果与限制](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/RESULTS.zh-CN.md)、[预声明规则](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/METHOD.zh-CN.md)、[自动工作点表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/REPORT.zh-CN.md)。
- [全部拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/fits.csv)、[候选/失败门](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/candidate_audit.csv)、[两端移动配对漂移](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/window_stability.csv)、[删flow训练/预测](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/holdout_predictions.csv)、[同窗flow敏感性](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/flow_subset_comparisons.csv)。
- [基础表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/figures/H3_fundamental.png)、[伴随表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/figures/H3_adjoint.png)、[冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/complete.json)、[最终落盘核验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/saved_output_verification_4029375.json)。

完整参数中心/协方差在fits.csv，fit_samples.npz各两列为d,k3共享JK副本；同目录保存代码、配置、组态blocking索引、输入哈希和CPU Slurm日志。文档原文备份、安装补丁与核对记录：[H3归档目录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_H3_20261120)。

## 15. 限定大三个flow的H3：范围内合理窗口与内部预测（2026-11-20）

### 15.1 本轮问题与参数约定

用户本轮明确不再要求4-flow联合，而只拟合最大的三个flow。**因此原flow0不进入候选门、内部留出或工作点排序；不能再用预测flow0失败来否定本轮指定范围内的拟合。** 第14节的全flow结论保留为历史记录，第15节回答不同拟合域的问题，两者不矛盾。也不据此指称flow0测量错误。

保留原flow1/2/3，物理t_f=0.003218436、0.004023045、0.004827654fm²，对应r_f=0.1604602381、0.1794000000、0.1965228536fm；三ensemble的物理流时间一致，无量纲tau=t_f/a²各自不同。最大flow3为参考。对象仍为Landau规范固定、start0开放线，两表示独立拟合、单线m_line=d/r_f。

$$
D_i(z)=-d\,z\left(\frac1{r_i}-\frac1{r_3}\right)+B_i+
\frac{k_3(r_i^2-r_3^2)}{2z^2},\qquad i=1,2.
$$

三flow版共有**4参数d、B1、B2、k3**，不是全四flow的5参数。k3自由，不加c_flow、H5、eta、先验或Casimir条件；不强制一圈共同幅度关系。z/r/a使用fm，d与k3无量纲，无闭合圈因子2。

### 15.2 选窗与三flow内部检验

复用上一轮omit0的1512个拟合：每ensemble/表示28个连续窗口、至少3长度，z_min>=0.3fm，两端均可移动，安全上限C24/E32/F32分别为1.155/1.0764/0.930fm。主block1、删模cut=1e-8；block1/2/5乘严格全协方差None、1e-8、1e-6共9设置。所有候选要求9设置可辨识、正自由度、d>0、相对误差<=10%、z_min/r_max>=3。

- scope_strict：至少4长度，主Q、block1全协方差Q、全部9设置Q均>=0.01。
- scope_relaxed：至少3长度，主Q与block1全协方差Q>=0.001，全部9设置Q>=0.0001。
- 展示点按strict优先，其次relaxed，再优先最大物理跨度、最多长度、最高上限；不按最高Q挑点。这是探索性范围调整，Q未作选窗效应校正。

新增504项仅在保留三flow内部的留一flow检查：用其余两flow的精确有限差分局部斜率训练d、k3，消去截距，再预测被留出的flow。共享JK保留训练/预测交叉协方差，不能按独立误差相加。训练与预测用block1严格全协方差；三次训练及预测Q均>=0.01才记internal_pass，和范围门分开记录。

三个长度只有两个训练斜率、两个参数，训练零自由度，故126项此类检查标为statistically_unidentifiable，**不是预测Q=0或预测失败**；至少四长度才有正自由度内部检验。其余378项训练可辨识。

### 15.3 范围内结果

下表误差仅为指定窗口/模型的主设置JK统计误差，不含完整窗口/flow域/模型/自相关等系统误差。

| ensemble | 表示 | z范围fm | 长度数 | d ± JK stat | chi²/dof | Q | 三flow内部预测最低Q | 判断 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| C24P29 | 基础F | 无通过窗 | — | 不推荐报系数 | — | — | — | 仍失败 |
| C24P29 | 伴随A | 0.8400–1.1550 | 4 | 0.834493 ± 0.007516 | 2.5441 | 0.03756 | 0.01756 | strict且内部通过；修正项需谨慎 |
| E32P29 | 基础F | 0.8073–1.0764 | 4 | 0.369198 ± 0.000620 | 1.3376 | 0.25320 | 0.15551 | strict且内部通过 |
| E32P29 | 伴随A | 0.6279–1.0764 | 6 | 0.820514 ± 0.001122 | 1.5118 | 0.14702 | 0.03842 | strict且内部通过 |
| F32P30 | 基础F | 0.7750–0.9300 | 3 | 0.342325 ± 0.001212 | 2.9761 | 0.05099 | 不可检验 | relaxed三长度窄窗 |
| F32P30 | 伴随A | 0.6200–0.9300 | 5 | 0.769006 ± 0.001482 | 1.8491 | 0.08550 | 0.03030 | strict且内部通过 |

strict/额外relaxed窗口数依次为C24F 0/0、C24A 1/2、E32F 2/6、E32A 6/4、F32F 0/2、F32A 3/3；合计12/17，12个strict全部通过内部检验。上述四个strict展示点在9设置中的最低Q为0.01335、0.07387、0.12214、0.04355，内部训练最低Q为0.83655、0.59264、0.21067、0.45454。C24基础最佳安全长距离窗仍只有chi²/dof=8.7734、Q=0.0001548，不满足温和放宽门。

### 15.4 窗口稳定性与修正项解释

F32伴随不只是一个三长度窗碰巧改善。主窗0.6200–0.9300fm附近，另有0.6975–0.9300fm的d=0.768486±0.002681、Q=0.16105，以及0.6200–0.8525fm的d=0.768663±0.001852、Q=0.03778；三窗均strict且内部通过。两个邻窗相对主点的共享JK配对pull为0.2325、0.3082，中心变化0.0676%、0.0446%。因此可以将d_A约0.7690作为**限定大三个flow域内的H3条件工作值**。

E32基础主点有3个稳定邻窗，伴随有2个；伴随相邻0.7176–1.0764fm给d=0.822449±0.001918，与主点相差0.001935±0.001555（1.24sigma）；0.6279–0.9867fm给d=0.820813±0.001236（配对pull 0.577）。C24伴随只有一个strict窗，两个relaxed邻窗中心下移1.22%–2.45%，不宜只引用主点约0.9%的统计误差。

F32基础另一个relaxed三长度窗为0.6975–0.8525fm，d=0.342608±0.000971、Q=0.006677。它与展示点不是单端点移动相邻，两窗都没有正自由度内部训练检验；尚无通过的四长度窗，不能称为稳健平台。

稳定边定义：仅一个端点移动一格、两窗均过范围门、配对pull<=3且相对变化<=3%。C24A/E32F/E32A/F32F/F32A展示点的直接稳定邻窗数为2/3/2/0/2，连通窗数3/8/10/1/6，对应d中心值包络0.814022–0.834493、0.367971–0.369492、0.820514–0.836797、单点、0.768486–0.771040。**这些不是置信区间或完整系统误差；逐边兼容不等于整个分量为共同平台，也不独立平均相关窗口。**

在保留flow范围内，三种两flow训练的最大d中心差占主点比例：C24A 3.88%、E32F 1.045%、E32A 0.409%、F32A 0.794%，相应配对pull约1.30/1.90/0.91/1.28。未发现显著内部预测失配不等于差异为零，C24A尤其统计分辨力有限。

H3通过仅指自由k3形状模型：C24A的k3=−0.605256±0.313281，相对固定一圈关系k3=d/(2sqrt(pi))配对偏离约−2.70sigma；E32F/E32A/F32F/F32A分别约+2.57/+2.42/−0.16/+3.77sigma，不能把拟合通过误称验证一圈共同幅度。9种block/cut下主窗d中心范围为C24A 0.834493–0.836332、E32F 0.369198–0.369529、E32A 0.820347–0.821442、F32F 0.342117–0.342325、F32A 0.768633–0.769323。

**本轮建议：限定大三个flow后，E32两表示、F32伴随有范围内合理且经内部检验支持的窗口；C24伴随有条件保留但突出k3符号与窄窗敏感性；F32基础仅探索候选；C24基础仍不接受。仍未形成三个ensemble均受控的完整两表示系数集，final_coefficient_approved=false，不更新smooth ZR。**

### 15.5 数值验收与文件入口

最终CPU作业4029443正常完成；初次4029421仅因合成单元测试夹具缺少d_error而停在测试阶段，修正夹具后通过，不改变真实模型或阈值。7项单元测试、168项逐组态主协方差重构/独立Cholesky+缩放QR、1512组复用副本及378组新内部副本误差通过。独立落盘验收PASS：18个冻结产物与17个输入SHA256、1512组拟合副本、378组内部副本、252项配对窗口差、12项展示窗flow对差全部核对；排除的flow0未参与当前计算。

目录：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/`。

- [结果和解释限制](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/RESULTS.zh-CN.md)、[方法与门槛](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/METHOD.zh-CN.md)、[自动数值表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/REPORT.zh-CN.md)。
- [全部拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/fits.csv)、[候选审计](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/candidate_audit.csv)、[三flow内部预测](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/internal_holdout.csv)、[两flow训练配对差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/pair_comparisons.csv)、[窗口配对敏感性](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/window_stability.csv)。
- [基础表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/figures/upper3_fundamental.png)、[伴随表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/figures/upper3_adjoint.png)、[数值冻结记录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/complete.json)、[最终落盘核验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/saved_output_verification_4029443.json)。

主文档本轮原文备份、安装补丁、解读文件哈希和安装核对另存于[大三flow文档归档目录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_upper3_20261120)。数值冻结记录保持原状；后生成的RESULTS解读文件不冒充原18项数值冻结产物。

## 16. C24P29两flow：可提取条件系数，跨pair仍有限制（2026-11-20）

### 16.1 模型与问题边界

沿用上轮大flow范围，分别测试原(1,2)、(1,3)、(2,3)；含最小flow0的三种pair仅作补充。两flow的H3为

$$
D_{ij}(z)=-d\,z\left(\frac1{r_i}-\frac1{r_j}\right)+B+
\frac{k_3(r_i^2-r_j^2)}{2z^2},\qquad i<j.
$$

这是d/B/k3三个参数，至少4长度才有正自由度，三长度精确插值不当作拟合质量通过。最大两个flow即原(2,3)，tau=0.3649020408/0.4378824490，r_f=0.1794000000/0.1965228536fm；flow1的tau=0.2919216327、r_f=0.1604602381fm。对象、样本与单线m_line=d/r_f约定同第15节，C24全量976组态、a=0.105fm。

另做删去k3的两参数leading对照，不加c_flow、先验、Casimir或固定一圈幅度，不按Q在模型间择优。排除flow不作为任一pair的候选外推硬门。

### 16.2 窗口与结果

每pair/模型/表示28窗，z_min>=0.3fm、两端移动、安全上限1.155fm，排除半盒。主block1、相关矩阵删模cut=1e-8；block1/2/5乘None全协方差、1e-8、1e-6共9设置。范围门沿用第15节对应阈值：9设置可辨识、主d>0、相对误差<=10%、z_min/r_max>=3；strict至少4长度，主/全/九设置Q>=0.01；relaxed主/全Q>=0.001、九设置Q>=0.0001。先strict，再按最大跨度/最多长度/最高上限选点，不按最高Q。

**两表示的三种主pair均在0.630–1.155fm、6长度、3自由度出现strict H3窗。** 下表为条件JK统计误差，未包括完整pair/模型/窗口系统误差。

| 表示 | flow pair | d ± JK stat | k3 ± JK stat | chi²/dof | Q | 九设置最低Q | strict窗数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 基础F | 1,2 | 0.380303 ± 0.000732 | 0.116194 ± 0.006945 | 2.4384 | 0.06250 | 0.06023 | 6 |
| 基础F | 1,3 | 0.383523 ± 0.000472 | 0.105026 ± 0.003867 | 2.9698 | 0.03052 | 0.03052 | 6 |
| 基础F | 2,3 | 0.387514 ± 0.000910 | 0.090726 ± 0.005500 | 1.0633 | 0.36326 | 0.21512 | 5 |
| 伴随A | 1,2 | 0.842094 ± 0.003354 | 0.270798 ± 0.053751 | 0.4778 | 0.69771 | 0.66866 | 6 |
| 伴随A | 1,3 | 0.853058 ± 0.002056 | 0.299741 ± 0.028451 | 0.4815 | 0.69511 | 0.67010 | 6 |
| 伴随A | 2,3 | 0.868214 ± 0.003615 | 0.331877 ± 0.035764 | 0.0251 | 0.99462 | 0.96344 | 6 |

若只问最大两个flow，可保留d_F^(23)=0.387514、d_A^(23)=0.868214为指定域内条件工作值；选择(2,3)是按flow大小，不是因其Q高。

### 16.3 最大pair并非只有孤立好窗

| 表示 | z窗fm | 长度数 | d ± JK stat | Q | 相对主窗配对pull |
| --- | --- | ---: | --- | ---: | ---: |
| F | 0.630–1.155 | 6 | 0.387514 ± 0.000910 | 0.36326 | 主点 |
| F | 0.735–1.155 | 5 | 0.388690 ± 0.001311 | 0.44070 | 1.245 |
| F | 0.630–1.050 | 5 | 0.387751 ± 0.000979 | 0.25209 | 0.659 |
| A | 0.630–1.155 | 6 | 0.868214 ± 0.003615 | 0.99462 | 主点 |
| A | 0.735–1.155 | 5 | 0.867637 ± 0.006324 | 0.96899 | 0.111 |
| A | 0.630–1.050 | 5 | 0.868207 ± 0.003846 | 0.96303 | 0.0053 |

以上均strict。最大pair主点各有两个稳定邻窗；稳定连通分量F/A为5/6窗，d中心包络0.387514–0.390033、0.864849–0.868214。边的定义沿用第15节：单端点移动一格、两窗通过、配对pull<=3且相对变化<=3%。包络不是置信区间或完整系统误差，也不能独立平均。主窗9种block/cut的d中心范围F 0.387514–0.389015、A 0.867934–0.868214。

### 16.4 不能把单pair通过当成通用系数确立

相同0.630–1.155fm、共享JK的差值为：

| 表示 | pair比较 | delta d ± 配对误差 | pull | 相对变化 |
| --- | --- | --- | ---: | ---: |
| F | (1,3)−(1,2) | 0.003219 ± 0.000581 | 5.54 | 0.847% |
| F | (2,3)−(1,2) | 0.007210 ± 0.001336 | 5.40 | 1.896% |
| F | (2,3)−(1,3) | 0.003991 ± 0.000771 | 5.18 | 1.041% |
| A | (1,3)−(1,2) | 0.010964 ± 0.002464 | 4.45 | 1.302% |
| A | (2,3)−(1,2) | 0.026121 ± 0.005684 | 4.60 | 3.102% |
| A | (2,3)−(1,3) | 0.015157 ± 0.003286 | 4.61 | 1.777% |

这些比较高度相关，不是多份独立证据，但不能忽略其显著pair依赖。基础三pair共同通过的H3窗有5个，最大配对pull约3.02–5.54，扫描中仍未找到全部差异<3sigma的共同窗。最窄0.840–1.155fm只有4长度/1dof，最大pull约3.023，不能据临界值宣布问题已解决。

伴随提高下限到0.735fm时三个pair的d为0.846148±0.006692、0.855018±0.004075、0.867637±0.006324，最大pull降为2.12，但中心跨度仍2.54%、误差增大；0.840fm下限最大pull约1.32而中心跨度约3.89%，主要是统计分辨力变弱，非中心值趋同证明。这与第15节窄窗伴随的条件通过相容。

两flow直接提供相对质量差，在假设1/r_f形式后才换算为d。若还存在c_flow*t_f质量项，单pair对应d_eff=d+c_flow*(t_i-t_j)/(1/r_i-1/r_j)；d与c_flow都乘z，单pair中严格共线，不能再加一个自由c_flow将其区分。这是退化例子，不是判定差异必由c_flow造成。**单pair长度形状可拟合，不代表跨pair共同d/k3的flow依赖已解释。**

最大pair同一宽窗leading的Q为F 2.26e-58、A 8.50e-18；H3修正对保留宽窗仍有必要。leading基础各主pair无strict窗、仅更窄relaxed窗；伴随(2,3)在0.735–1.155fm可得leading d=0.851709±0.002162、Q=0.06464，不能与不同窗口的宽窗H3当成独立测量。含flow0的三个pair也有H3通过窗，但只是附加范围敏感性，不用于否决主pair。

**本轮结论：C24两表示均能从指定两flow中提取有多窗支持的H3条件系数；最大pair可用上列工作值。基础的跨pair一致性尚未解决，伴随也仍有百分之几的pair/窗口敏感性，不批准C24唯一通用系数，不更新smooth ZR。**

### 16.5 验收与文件

CPU作业4029836，COMPLETED、exit 0:0，6048项拟合尝试、672主窗口、5290项可辨识；756个H3三长度窗零自由度，另2个H3四长度窗在block5/cut1e-6仅剩3模、3参数而跳过，均未伪造Q。8项单元测试、5290组JK副本误差、1764项独立QR及局部斜率等价、126项上一轮内部训练复现通过。落盘核对18个冻结产物SHA256、13个输入SHA256、5290组副本误差、2586项配对差均PASS；final_coefficient_approved=false。

目录：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/`。

- [结果与限制](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/RESULTS.zh-CN.md)、[方法判据](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/METHOD.zh-CN.md)、[自动数值表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/REPORT.zh-CN.md)。
- [全部拟合](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/fits.csv)、[候选审计](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/candidate_audit.csv)、[窗口配对差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/window_stability.csv)、[同窗pair配对差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/pair_comparisons.csv)、[模型差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/model_comparisons.csv)。
- [基础表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/figures/twoflow_fundamental.png)、[伴随表示图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/figures/twoflow_adjoint.png)、[数值冻结](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/complete.json)、[落盘复核](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/saved_output_verification_4029836.json)。

主文档原文备份、补丁、后生成解读文件的独立哈希和安装核验见[C24两flow归档目录](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_C24_twoflow_20261120)。原数值冻结清单不重写，后生成RESULTS不冒充原18项冻结产物。

## 17. H3模型完整推导与三个ensemble线性发散系数汇总（2026-11-20）

本节集中回答“H3从哪里来、为什么对数差里是1/z²、参数究竟有几个、目前各ensemble可以报告哪些系数”。第12–16节历史数值与候选门保持原状。本节从已完成的三flow和C24两flow分析取数，不新增拟合、不按最高Q重选工作点。

### 17.1 观测量、质量约定与近似范围

考虑四维Euclidean空间、Wilson flow后在Landau规范下的开放直Wilson line，表示R=F或A，线沿空间z方向。定义

$$
W_R(z,t)=\operatorname{Re}\left\langle\frac1{\dim R}
\operatorname{Tr}_R\,\mathcal P\exp\left[ig\int_0^z ds\,B_z(t,s\hat z)\right]\right\rangle,
\qquad r\equiv\sqrt{8t},\qquad
\mu_R(z,r)\equiv-\partial_z\ln W_R(z,r).
$$

这里mu_R是线的局部有效质量，不是MS-bar重整化尺度。实际统计量为log(组态平均W)，不是组态log的平均；对象不是带外部夸克/强子态的双线性算符，也不是闭合Wilson loop。开放线规范依赖，不能把以下Landau核当成规范无关物理势。

采用正的单线质量约定m_line=d_R/r，故ln W的线性项为−d_R*z/r；d_R无量纲，不带闭合圈两条边的因子2。本推导使用连续空间、树级流化传播子计算O(g²)线自能，长距离展开再要求z/r>>1；共同耦合g在同一微扰阶和尺度下使用。固定格距数据中的自由d/k3并不因此自动等于微扰表达式。

### 17.2 从流化传播子到位置空间D_zz

树级横向场B_mu(t,p)=exp(−t*p²)A_mu(p)，因此两个场都在相同流时间t时，去掉颜色delta_ab的Landau传播子为

$$
\widetilde D_{\mu\nu}(p;t)=\frac{e^{-2tp^2}}{p^2}
\left(\delta_{\mu\nu}-\frac{p_\mu p_\nu}{p^2}\right).
$$

引入径向Fourier变换

$$
G(u)=\int\frac{d^4p}{(2\pi)^4}\frac{e^{ipu-2tp^2}}{p^2}
=\frac{1-e^{-u^2/r^2}}{4\pi^2u^2},\qquad
H(u)=\int\frac{d^4p}{(2\pi)^4}\frac{e^{ipu-2tp^2}}{p^4}.
$$

H本身有红外加法常数的歧义；下面只用其有限导数。对位置矢量u*hat z，Fourier乘p_z²对应−partial_z²，故D_zz=G+H''。又由−laplacian H=G，在四维径向坐标有

$$
H''(u)+\frac3u H'(u)=-G(u),\qquad
\bigl[u^3H'(u)\bigr]'=-u^3G(u).
$$

取原点正则的导数解，积分得到

$$
H'(u)=-\frac{u^2-r^2(1-e^{-u^2/r^2})}{8\pi^2u^3},
\qquad
D_{zz}(u;r)=-\frac3uH'(u)
=\frac{3}{8\pi^2u^2}
\left[1-\frac{r^2}{u^2}\bigl(1-e^{-u^2/r^2}\bigr)\right].
$$

表面上的u=0奇点是可去的：D_zz(0;r)=3/(16*pi²*r²)。这里的横向投影和系数3属于Landau开放线，不能直接用Feynman规范或不同算符的公式替换。

### 17.3 线自能积分和精确有限长度核

展开路径有序指数至O(g²)，利用同向两点函数的平移与交换对称性：

$$
\ln W_R(z,r)
=-\frac{g^2C_R}{2}\int_0^z ds\int_0^z ds'\,D_{zz}(s-s';r)+O(g^4)
=-g^2C_R\int_0^z du\,(z-u)D_{zz}(u;r)+O(g^4).
$$

其中C_R由T_R^a*T_R^a=C_R*1定义。因此

$$
\mu_R(z,r)=g^2C_R\int_0^z du\,D_{zz}(u;r)+O(g^4)
=d_R^{(1)}\,K(r,z)+O(g^4),
$$

$$
d_R^{(1)}=\frac{g^2C_R}{4\pi^{3/2}}
=\frac{\alpha_s C_R}{\sqrt\pi},\qquad
K(r,z)=\frac1r F\left(\frac zr\right),
$$

$$
F(x)=\operatorname{erf}(x)+\frac1{2\sqrt\pi}
\left[\frac{2e^{-x^2}-3}{x}+\frac{1-e^{-x^2}}{x^3}\right].
$$

可由F(0)=0和F'(x)=4*pi^(3/2)*r²*D_zz(r*x;r)检查积分归一化。F(infinity)=1，所以mu_R在该阶的无限长极限为d_R^(1)/r。实际拟合F/A分别进行，不以C_A/C_F=9/4互相构造实测系数，也不从拟合d反推一个已测定的alpha_s。

### 17.4 mu(z,r)长距离展开及共同1/z项的消去

将F中的非指数部分分离，得到

$$
K(r,z)=\frac1r-\frac{3}{2\sqrt\pi\,z}
+\frac{r^2}{2\sqrt\pi\,z^3}+R_K(r,z),
$$

$$
R_K(r,z)=-\frac1r\operatorname{erfc}\left(\frac zr\right)
+\frac{e^{-z^2/r^2}}{\sqrt\pi\,z}
-\frac{r^2e^{-z^2/r^2}}{2\sqrt\pi\,z^3}.
$$

R_K在z/r>>1时指数小；进一步渐近展开的首项为−3*r^4*exp(−z²/r²)/(4*sqrt(pi)*z^5)。因此在这一特定连续一圈核中，写出的1/r、1/z、r²/z³之后不是一个独立的纯r^4/z^5项；此前H5是额外经验敏感性形状，不是本核的一圈下一项。

在相同耦合的该阶近似下，mu_i与mu_j的共同1/z项相消：

$$
\mu_R(z,r_i)-\mu_R(z,r_j)
=d_R^{(1)}\left(\frac1{r_i}-\frac1{r_j}\right)
+\frac{d_R^{(1)}}{2\sqrt\pi}\frac{r_i^2-r_j^2}{z^3}
+d_R^{(1)}\bigl[R_K(r_i,z)-R_K(r_j,z)\bigr]+O(g^4).
$$

所以局部比值的首个剩余幂次修正是z^-3，而不是另一个共同1/z项。该消去依赖同一阶共同幅度；高阶、flow相关有效耦合、格点伪影和非微扰项未被本式排除，不能将其推广为全阶定理。

### 17.5 从局部质量积分得到实际H3对数差模型

因为partial_z ln W=−mu，对上式逐项积分可写成

$$
\ln W_R(z,r)=-d_R^{(1)}\frac zr
+\frac{3d_R^{(1)}}{2\sqrt\pi}\ln\frac z{z_0}
+b_R(r)+\frac{d_R^{(1)}r^2}{4\sqrt\pi\,z^2}
+\mathcal R_R(z,r)+O(g^4).
$$

z0只是使对数无量纲的共同参考长度，改变它可吸收到b_R；R_R为指数余项的积分，积分常数同样并入b_R。这里的正r²/z²符号由积分−z^-3得到，不能误写成负号。

对同一表示定义D_i=ln W(z,r_i)−ln W(z,r_ref)，共同ln(z/z0)消去：

$$
D_i(z)=-d_R^{(1)}z\,\Delta\left(\frac1r\right)_i+B_i
+\frac{d_R^{(1)}}{4\sqrt\pi}\frac{\Delta r_i^2}{z^2}
+\Delta\mathcal R_{R,i}+O(g^4).
$$

实际使用的自由幅度H3模型保留该函数形状，舍去未受控余项而放开修正幅度：

$$
\boxed{D_i^{\rm H3}(z)=-d\,z\,\Delta(1/r)_i+B_i+
\frac{k_3\,\Delta r_i^2}{2z^2}}.
$$

只有在上述一圈共同幅度近似中才有k3=d_R^(1)/(2*sqrt(pi))；实际拟合d和k3独立，不施加该关系、不限制k3为正。因而“自由H3通过”不等于“固定一圈理论已验证”。C24三flow伴随的负k3尤其应保留解释警告。

### 17.6 H3局部斜率形式、离散实现与参数个数

对连续局部斜率，令Delta u_ij=1/r_i−1/r_j，则

$$
d_{{\rm eff},ij}(z)=\frac{\mu_i-\mu_j}{\Delta u_{ij}}
=d+k_3\frac{r_i^2-r_j^2}{\Delta u_{ij}}\frac1{z^3}
=d-k_3\frac{r_i r_j(r_i+r_j)}{z^3}.
$$

这说明“H3”命名来自局部质量中的z^-3形状，**不是说对数差拟合里加z^-3，更不是说模型总有三个参数**。当k3>0时，该近似下局部d_eff从下方接近d；实际数据若不满足该行为，不能通过强制符号掩盖模型失配。

有限线段[z_L,z_R]应使用精确有限差分，不能直接以中点z^-3替代：

$$
-\frac{D_{ij}(z_R)-D_{ij}(z_L)}{(z_R-z_L)\Delta u_{ij}}
=d+k_3 H_{3,ij}[z_L,z_R],
$$

$$
H_{3,ij}[z_L,z_R]=\frac{r_i^2-r_j^2}{\Delta u_{ij}}
\frac{z_L^{-2}-z_R^{-2}}{2(z_R-z_L)}.
$$

若有N_f个flow，取一个参考后只有N_f−1组独立对数差；每组一个截距，另有共同d和k3，总参数数为N_f+1：

| flow数 | 独立对数差数 | H3参数 | 原始数据数（N_z个长度） |
| ---: | ---: | --- | ---: |
| 4 | 3 | d、B0、B1、B2、k3，共5个 | 3*N_z |
| 3 | 2 | d、两个B、k3，共4个 | 2*N_z |
| 2 | 1 | d、B、k3，共3个 | N_z |

自由度使用实际保留协方差模数减参数数。两flow三个长度是3数据/3参数，不能做正自由度质量检验；三flow三个长度的联合拟合可有2自由度，但其内部留一flow后只剩两flow、两个局部斜率约束d/k3，训练零自由度。这正是F32基础“三flow可拟合、内部训练不可检验”的区别。

另一个常被混淆的c_flow项来自假设质量再含m(r)=d/r+c_flow*t_f，给出对数差−c_flow*z*Delta t；它不是H3的有限长度修正。在单一两flow pair上，它与d项都乘z、严格共线，不能额外自由拟合以分离两者。c_flow也不能从本一圈H3推导中自动得到。

### 17.7 数据、flow时间与统计契约

以下均为Landau start0、已有全量逐组态数据，两表示独立拟合，长度和r_f使用fm：

| ensemble | a [fm] | Ncfg | 安全z_max [fm] |
| --- | ---: | ---: | ---: |
| C24P29 | 0.1050 | 976 | 1.1550 |
| E32P29 | 0.0897 | 1700 | 1.0764 |
| F32P30 | 0.0775 | 887 | 0.9300 |

大三个flow原编号1/2/3的共同物理t_f为0.003218436/0.004023045/0.004827654fm²，对应r_f=0.1604602381/0.1794000000/0.1965228536fm；无量纲tau=t_f/a²随ensemble不同。C24最大两flow仅保留原2/3。

主设置是block1、相关矩阵删模cut=1e-8；block1/2/5与None严格全协方差、1e-8、1e-6共9种设置。此处正cut明确表示删模，不是gvar抬升小模。所有flow/长度共享JK组态索引，协方差=(N_B−1)/N_B*delta.T@delta；同数据的窗口、pair、模型差用配对JK，不按独立误差相加。不同ensemble不作配对。

范围门、strict/relaxed阈值和工作点排序见第15–16节。表中误差仅是条件JK统计误差，未包含完整的模型、窗口、flow域、自相关、定标、有限体积、连续极限系统误差；blocking敏感性不等于自相关已经受控。Q为探索性质量诊断，未作选窗效应校正。

### 17.8 三个ensemble统一大三flow域的系数

本表只使用同一个flow域1/2/3，最大flow3为参考，H3均为4参数。它是统一适用域的比较表，不以C24两flow填补失败格子。

| ensemble | 表示 | z窗 [fm] | N_z | d ± JK stat | k3 ± JK stat | chi²/dof | Q | 范围内结论 |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| C24P29 | F | 无通过窗 | — | 不报通过系数 | — | — | — | 失败 |
| C24P29 | A | 0.8400–1.1550 | 4 | 0.834493 ± 0.007516 | −0.605256 ± 0.313281 | 2.5441 | 0.03756 | strict、内部通过；k3符号与窄窗需警告 |
| E32P29 | F | 0.8073–1.0764 | 4 | 0.369198 ± 0.000620 | 0.135924 ± 0.012502 | 1.3376 | 0.25320 | strict、内部通过 |
| E32P29 | A | 0.6279–1.0764 | 6 | 0.820514 ± 0.001122 | 0.258211 ± 0.011362 | 1.5118 | 0.14702 | strict、内部通过 |
| F32P30 | F | 0.7750–0.9300 | 3 | 0.342325 ± 0.001212 | 0.091784 ± 0.030554 | 2.9761 | 0.05099 | relaxed窄窗；内部训练零自由度 |
| F32P30 | A | 0.6200–0.9300 | 5 | 0.769006 ± 0.001482 | 0.269598 ± 0.014341 | 1.8491 | 0.08550 | strict、内部通过 |

三flow内部检验依次留出一个保留flow、用另两个训练局部斜率并预测；C24A/E32F/E32A/F32A展示点的最低预测Q为0.01756/0.15551/0.03842/0.03030。F32F没有可用的正自由度内部训练检验，不能把缺失检验写成通过或预测失败；它也没有通过的四长度窗或直接稳定邻窗。

E32两表示和F32伴随可保留为这一flow域下的条件工作值。F32伴随相邻窗0.6975–0.9300、0.6200–0.8525fm给d=0.768486±0.002681、0.768663±0.001852，均通过；C24A的邻窗中心可下移1.22%–2.45%，其统计/模型解释较弱。更早全四flow结果仍见第14节；本表不改变它们的通过/失败状态，也不要求外推到已排除flow0来验收本表。

### 17.9 C24最大两flow补充：可提取，但不是统一域替代品

只用原flow2/3，H3为3参数，两表示在相同0.630–1.155fm、6长度、3自由度通过：

| ensemble | 表示 | flow域 | d ± JK stat | k3 ± JK stat | chi²/dof | Q | 判断 |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| C24P29 | F | 2、3 | 0.387514 ± 0.000910 | 0.090726 ± 0.005500 | 1.0633 | 0.36326 | strict，多窗支持，条件系数 |
| C24P29 | A | 2、3 | 0.868214 ± 0.003615 | 0.331877 ± 0.035764 | 0.0251 | 0.99462 | strict，多窗支持，条件系数 |

提高下限到0.735fm，F/A为0.388690±0.001311、0.867637±0.006324；降低上限到1.050fm，F/A为0.387751±0.000979、0.868207±0.003846。两表示主点均有稳定邻窗，不是孤立最窄窗。

但是同一0.630–1.155fm下，pair(1,2)/(1,3)/(2,3)的F系数为0.380303/0.383523/0.387514，A为0.842094/0.853058/0.868214。共享JK的(2,3)−(1,2)差为F 0.007210±0.001336（5.40sigma，1.90%）、A 0.026121±0.005684（4.60sigma，3.10%）。因此不能把各pair视为同一d的独立测量并直接平均；更不能用两flow的通过覆盖C24三flow基础表示失败。

若需要列出每ensemble目前能报告的条件数值，C24可注明“最大两flow”使用本表，E32和F32使用17.8的“大三flow”表；**这是一份适用域不同的条件结果集合，不是三个ensemble统一方案的最终系数集**。F32基础仍只是三长度探索候选，C24两表示仍有pair依赖。

### 17.10 来源、推导核验和归档状态

推导基于本项目此前保存的[Landau开放线有限长度核推导](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_long_slope_finite_rf_20261118/THEORY.zh-CN.md)，本节补全径向传播子、线自能积分、长距离展开和对数差积分的步骤。文献背景为N. Brambilla与X.-P. Wang，*Off-lightcone Wilson-line operators in gradient flow*，JHEP 06 (2024) 210，arXiv:2312.05032；其带端点局域算符的Wilson-line算符讨论不是本节Landau开放线所有公式的直接出处，本节公式应按明确观测量独立核验，不能混用。

结果来源按域分开保存：

- [统一大三flow结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/RESULTS.zh-CN.md)、[数值摘要](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/summary.json)、[落盘核验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_upper3_20261120/saved_output_verification_4029443.json)。
- [C24两flow结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/RESULTS.zh-CN.md)、[数值摘要](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/summary.json)、[落盘核验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_C24_twoflow_20261120/saved_output_verification_4029836.json)。
- [全四flow历史结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_three_ensemble_20261120/RESULTS.zh-CN.md)，保留原候选条件与失败记录，不合并统计。

本轮归档另存[推导核验脚本](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_H3_derivation_summary_20261120/audit_derivation.py)、[推导恒等式与来源核验](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_H3_derivation_summary_20261120/derivation_validation.json)、[机器可读系数表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/documentation_updates/linear_devergence_H3_derivation_summary_20261120/coefficient_summary.csv)，以及原文备份、安装补丁和哈希。它们用于校验公式符号/归一化、精确有限差分及汇总取数；不重新拟合、不改动既有数值冻结产物。

CPU核验作业4029843正常完成（exit 0:0）：48项径向方程/传播子投影/核的导数与积分/指数余项/对数原函数/有限线段H3恒等式检查通过；三轮来源共57项冻结产物SHA256核对通过；机器表保存6条统一三flow状态（含C24基础失败格）及2条C24最大pair系数，共8条记录。new_fits=0。本轮安装另核对第2–16节原文保持不变。

**最终状态：H3有明确的一圈有限长度形状来源，但自由幅度拟合不是完整一圈理论验证；现有结果仍为指定模型/窗口/flow域下的条件系数，final_coefficient_approved=false。没有更新生产PDF减除、smooth ZR或连续极限结果。**

## 18. 2026-11-24：补充至8个flow后的H3窗口、半盒检验与lnW图

### 18.1 数据与全8-flow拟合

Landau规范起点 `start0`；C24P29、E32P29、F32P30分别使用976、1700、887个组态，各flow严格同组态配对，基础表示 F 与伴随表示 A 分开处理。三个ensemble对应相同的物理流半径，flow编号 `0–7` 依次为 **0.138963、0.160460、0.179400、0.196523、0.212269、0.226925、0.240690、0.249875 fm**；编号 `4–7` 为本轮补充数据。格点流时间为 $\tau=t_f/a^2$，$r_f=a\sqrt{8\tau}$；相同的物理 $r_f$ 并不意味着三个ensemble有相同的 $\tau$。

继续使用§17.5的H3模型：对组态均值取对数后构造同一表示的flow差分 $D_i(z)=\ln\langle W(z,r_i)\rangle-\ln\langle W(z,r_{\rm ref})\rangle$。每个子区间以其中最大flow为参考。拟合有共同的无量纲 $d$ 和自由有限长度幅度 $k_3$，以及每个非参考flow的差分截距，参数总数为 $N_{\rm flow}+1$；减除质量约定为 $m_{\rm line}=d/r_f$。下列误差均为主设置（block=1，相关矩阵相对删模阈值 $10^{-8}$）的jackknife统计误差。

全8-flow、$z_{\min}\ge 3r_{f,\max}$ 至安全上限的固定长窗结果（括号内为 $\chi^2/\mathrm{dof}$）：

| Ensemble | $z$ [fm] | F：$d\pm\sigma_{\rm JK}$（$\chi^2/\mathrm{dof}$） | A：$d\pm\sigma_{\rm JK}$（$\chi^2/\mathrm{dof}$） |
|---|---:|---:|---:|
| C24P29 | 0.8400–1.1550 | 0.370724 ± 0.000276（134.75） | 0.829849 ± 0.001833（15.70） |
| E32P29 | 0.8073–1.0764 | 0.367288 ± 0.000179（12.09） | 0.822941 ± 0.000940（1.80） |
| F32P30 | 0.7750–0.9300 | 0.346094 ± 0.000323（25.09） | 0.777083 ± 0.001536（7.22） |

三套ensemble的基础表示均无合格的全8-flow窗口。E32伴随表示在上述长窗的名义 $Q=0.0175$，但未通过留一flow检查。若纳入所有安全非零 $z$，六组的 $\chi^2/\mathrm{dof}$ 均大于 $2\times10^5$，且 $d$ 明显下移；这些是H3失配诊断，不是更精确的系数测量。

### 18.2 连续flow子区间扫描与当前工作窗

扫描所有包含3–8个连续flow的21个区间和 $z\ge0.5$ fm、至少3个连续长度的 $z$ 窗：共1680个窗口、15120个协方差拟合，复核560个与前轮重合的窗口。主窗上限为C24的 $z/a=11$、E32和F32的 $z/a=12$。候选要求 $z_{\min}/r_{f,\max}\ge3$、$d>0$ 且相对统计误差不超过10%，并检查 block=1/2/5 与全协方差、相对删模阈值 $10^{-8}/10^{-6}$ 的九种组合。严格门要求至少4个长度、九个变体的 $Q\ge0.01$。

下表六个工作窗均通过严格门、留一flow训练及预测 $Q\ge0.01$，并有一个稳定的相邻 $z$ 窗和一个稳定的同 $z$ 相邻flow窗。相邻 $z$ 窗只改一端一格；相邻flow窗只改一端一格或整体平移一格。稳定要求 $d$ 的配对漂移同时不超过 $3\sigma$ 和3%。这只是局部稳定性，不等于所有通过窗共享一个 $d$；筛选不以最小 $\chi^2/\mathrm{dof}$ 为目标。

| Ensemble | 表示 | flow编号（含两端） | $r_f$ [fm] | $z$ [fm] | $d\pm\sigma_{\rm JK}$ | $\chi^2/\mathrm{dof}$ | $Q$ |
|---|---|---|---:|---:|---:|---:|---:|
| C24P29 | F | 2–5 | 0.1794–0.2269 | 0.7350–1.1550 | 0.387234 ± 0.000543 | 1.337 | 0.203 |
| C24P29 | A | 2–6 | 0.1794–0.2407 | 0.7350–1.1550 | 0.861606 ± 0.001864 | 0.851 | 0.614 |
| E32P29 | F | 4–7 | 0.2123–0.2499 | 0.8073–1.0764 | 0.359193 ± 0.000816 | 0.602 | 0.755 |
| E32P29 | A | 1–4 | 0.1605–0.2123 | 0.7176–1.0764 | 0.821117 ± 0.001281 | 1.258 | 0.248 |
| F32P30 | F | 2–5 | 0.1794–0.2269 | 0.6975–0.9300 | 0.339095 ± 0.000547 | 1.730 | 0.097 |
| F32P30 | A | 1–5 | 0.1605–0.2269 | 0.6975–0.9300 | 0.767827 ± 0.001374 | 1.846 | 0.0476 |

**E32P29基础表示保留两个flow窗口，不合并。** 固定同一个 $z=0.8073$–$1.0764$ fm，flow `1–3`（$r_f=0.1605$–$0.1965$ fm）也通过拟合、留一flow及相邻 $z$ 窗检查，给出 $d=0.369198\pm0.000620$、$\chi^2/\mathrm{dof}=1.338$。从 `1–3` 换到表中的 `4–7`，$\Delta d=-0.010006\pm0.001050$（配对JK，约−2.7%、−9.53σ）。较大的flow窗拟合优度更好，却不能消除这一系统差异。一般应兼顾小flow端的格点效应与大flow端的有限 $z/r_f$ 修正，不能只按flow大小选窗。

### 18.3 C24半盒点与新版图

只改变C24拟合上限，从 $11a=1.155$ fm 延至半盒 $12a=1.260$ fm；固定对应的下限、flow和主协方差设置。对flow `2–5`（$z_{\min}=0.735$ fm），F/A的 $d$ 分别移动 +0.000295（1.68σ）和 +0.001344（1.62σ）；对全8-flow（$z_{\min}=0.840$ fm），分别移动 +0.001273（9.90σ）和 +0.003852（3.80σ），拟合仍明显失配。漂移显著性用同组态配对JK差值计算，不用独立误差相加。这不是有限体积效应的单独证据，但半盒点并非合格窗所必需，故只作敏感性测试，不进入主拟合。

新版图纵轴为先平均再取对数的 $\ln\langle W\rangle$，不是H3使用的flow差分；横轴为 $1/\sqrt{\tau}$。每张图包含三个ensemble，全部8个flow统一使用圆点，无新旧flow分界线，颜色只表示 $z/a$，误差棒沿用JK，连线只连接数据。C24只画 $z/a=1$–$11$，E32/F32画 $z/a=1$–$12$，两种表示共560个图点。新版不覆盖旧图，不改变任何拟合产物。

- 基础表示：[PDF](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/figures_safe/lnW_vs_inv_sqrt_tau_fundamental.pdf) · [PNG](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/figures_safe/lnW_vs_inv_sqrt_tau_fundamental.png)。
- 伴随表示：[PDF](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/figures_safe/lnW_vs_inv_sqrt_tau_adjoint.pdf) · [PNG](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/figures_safe/lnW_vs_inv_sqrt_tau_adjoint.png)。

### 18.4 数值来源与状态

- 作业 `4045828`：[新增flow审计](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_larger_rf_20261124/input_audit.json) · [初轮配对扫描](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_larger_rf_20261124/RESULTS.zh-CN.md)。
- 作业 `4045908`：[全8-flow结果](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/RESULTS.zh-CN.md) · [绘图数据](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/lnW_points.csv) · [新版绘图脚本](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/code/replot_safe_lnW.py)。重复半盒检查时复用[全flow拟合脚本](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_all8_lnW_20261124/code/analyze_all8.py)中的H3设计矩阵和配对JK，固定下限并比较 $n_{\max}=11$ 与12。
- 作业 `4047011`：[工作窗与配对对照](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_window_grid_20261124/RESULTS.zh-CN.md) · [窗口图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_window_grid_20261124/window_grid.pdf) · [窗口详表](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_window_grid_20261124/window_audit.csv) · [同z配对比较](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_window_grid_20261124/same_z_flow_shifts.csv) · [校验清单](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/wilson_line_H3_window_grid_20261124/complete.json)。

**状态：** 上表是指定H3模型、有限格距和选定flow–$z$窗下的条件结果；统计误差不含完整窗口、模型和连续极限误差。`final_coefficient_approved=false`，未替换生产线性发散减除系数。
