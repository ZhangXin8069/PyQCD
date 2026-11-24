# 局域归一化后的 NLO/NNLO GF 有限系数审计

2026-11-20。范围：质量为零近似、SU(3)、nonsinglet vector、4D thin-link flow，
同一 ringed 场处方。当前 unpolarized gamma4 的局域转换属于 vector 通道；
长线有限端点系数与该 Dirac 投影无关。这里不是 LaMET 卷积核。

## 1. 从两个前向关系推导转换方向

对非零、足够长的 z，设

\[
Z_\chi h^{GF}(z)=C_\psi e^{-m_{PT}|z|}h^{MS}(z)+\Delta_z,
\qquad Z_\chi h^{GF}(0)=\zeta_V h^{MS}(0)+\Delta_0.
\]

因此局域归一化形状的逆关系是

\[
R^{MS}(z)=\frac{\zeta_V}{C_\psi}e^{+m_{PT}|z|}R^{GF}(z).
\]

本次输入已乘 NP 因子 exp(+m_NP |z|)，这里只施加有限比值
`D_V=zeta_V/Cpsi`。严格 MS 关系还应有 exp[(m_PT-m_NP)|z|]，
但本项目条件 H3 系数与 PT 的方案闭合尚未建立，故不猜测该差、不重复乘完整 PT 指数。
输出保持为条件 NP 线方案下的 GF 有限转换诊断。

## 2. 两圈展开及必要的交叉项

设 a_s=alpha_s/(4pi)，

\[
C_\psi=1+a_sN_1+a_s^2N_2,\qquad
\zeta_V=1+a_sJ_1+a_s^2J_2.
\]

严格逆比值为

\[
D_V=1+a_s(J_1-N_1)
+a_s^2[J_2-N_2+N_1^2-J_1N_1]+O(a_s^3).
\]

若重轻端点为 c=1+a_s h1+a_s² h2，则 Cpsi=c²，
N1=2h1、N2=2h2+h1²。漏掉平方项或逆比值交叉项都不是正确 NNLO。
直接取两个截断多项式的有理比会额外混入未控制的高阶；本次主结果不这样做。

## 3. 原文系数与约定

- Xiang-Peng Wang, *Off-lightcone Wilson-line operators in gradient flow at
  next-to-next-to-leading order*, arXiv:2610.40225v1，式(32),(34),(40),(50)。
  式(40)提供重轻端点，式(34)要求平方；线质量不是本有限系数计算的输出。
- A. Borgulat 等, *Short-flow-time expansion of quark bilinears through
  next-to-next-to-leading order*, arXiv:2311.16799v2，式(3.9),(3.11)。
  局域 vector 前向系数使用 alpha_s/pi，换到本文件 a_s 的一圈/两圈系数需乘4/16。
- NLO 基线：`/public/home/donghx/.codex/Markdown/gradient_flow_quark_pdf.md` 第6–8节，
  即 Cpsi=1-a_s CF(3L_f+2+ln432)，zetaV=1+a_s CF(1/2-ln432)。

原文 HTML 快照及 SHA256 保存在 references/ 与 source_lock.sha256。
审计不是只复用旧 helper：本轮按两篇原式独立重实现，再与历史实现及原文数值式交叉核验。

## 4. 相消、对数与数值多项式

一圈有

\[
D_V^{NLO}=1+a_s C_F(3L_f+5/2),\qquad
L_f=\ln(2\mu^2t_fe^{\gamma_E}).
\]

同一 ringed 处方的两圈共同常数 Cchi,2 在 J2-N2 中严格消失；
不是将其物理值设为零，也不是测得 Zchi=1。
`coefficient_audit.json` 保存 SU3、nf3 的精确特殊函数数值化多项式，
`RESULTS.zh-CN.md` 给出可直接使用的 L_f²/L_f/常数系数。
双对数必须满足 f22=(f11²+beta0*f11)/2=26（nf3，f11=4），
这提供不依赖有限常数的 RG 检查。常数 f20=30.92223469310622。

流时间必须先从 fm² 除以 (hbar c)² 转为 GeV^-2。
主分支 mu=2GeV、alpha_s=0.3、nf=3；两阶使用同一输入耦合，比较严格固定阶。
旧文档的 NLO+LL 另作系数图参考，不将它与 NNLO 固定阶的差称为纯两圈效应。
本轮不把 NNLO 边界硬接旧 LL 后声称 NNLO+NLL。

## 5. 局域点、误差与物理范围

同 P、同 flow、同 ensemble 的实偶 z=0 值作为分母，逐原 bootstrap 行相除。
z=0 定义为1+0i，不能再乘非局域长线端点系数。局域极限与 t/z²→0 的极限不交换；
不能用 z=0 的约束来估计长线系数，也不能由首个非零 z 的外观判断系数必须为1。

局域比值消掉公共 Zchi 和局域电荷，得到形状而非绝对归一化矩阵元。
需要绝对 PDF 时仍需恢复同方案的电荷，不能用此步骤替代。

误差保留全200行，归一化的分子/分母和 NLO/NNLO 差均逐行配对。
误差棒固定 d、alpha_s、尺度；d 的独立响应单列，不伪造与全量 Wilson JK 的对应关系。
阶数差只是一种确定性截断敏感性，不是完整扰动论误差估计。

不将有限系数核验升级为整个真实数据链的验收：仍需控制尺度层级、格距、flow、
线质量方案闭合、平台质量、有限z幂修正；本轮没有 flow/连续极限或 LaMET/PDF 重建。
