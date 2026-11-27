
## 18. 三组 adjacent self ratio：局域归一化后的 NNLO GF 有限转换（2026-11-26）

本节补充第 8 节，不覆盖 L32x64 B 方案的一圈 RGR 历史。这里的公式用于 thin-link 四维流、无质量近似的 nonsinglet vector 通道（当前 unpolarized connected $u-d$，包括 $\gamma_4$ 投影）。不是 tensor、singlet 或 LaMET matching 的系数。

### 18.1 正向系数、严格展开和一次线质量减除

第 8.3 节的两个正向转换关系相除后，同一 ringed 处方的 $Z_\chi$ 从比值中消去。对非零 $z$，未经线质量减除的局域比值到 $\overline{\mathrm{MS}}$ 形状的转换是 $(\zeta_V/C_\psi)e^{m_F^{\rm PT}|z|}$，以下质量与长度在指数中均使用统一单位。写 $a_s=\alpha_s/(4\pi)$、$C_\psi=1+a_sN_1+a_s^2N_2$、$\zeta_V=1+a_sJ_1+a_s^2J_2$，则

$$
D_V=\frac{\zeta_V}{C_\psi}
=1+a_s(J_1-N_1)+a_s^2(J_2-N_2+N_1^2-J_1N_1)+O(a_s^3).
$$

重轻单端点 $c=1+a_s h_1+a_s^2 h_2$ 要平方成两端点 $C_\psi=c^2$，故 $N_1=2h_1$、$N_2=2h_2+h_1^2$。用两圈截断级数的有理比代替严格展开会混入未计算的高阶。共同的两圈 ringed 常数在 $J_2-N_2$ 中相消，不等于把物理 $Z_\chi$ 设为 1。

- 非局域端点：X.-P. Wang，*Off-lightcone Wilson-line operators in gradient flow at next-to-next-to-leading order*，arXiv:2610.40225v1，式 (32)、(34)、(40)、(50)。
- 局域 vector：A. Borgulat 等，*Short-flow-time expansion of quark bilinears through next-to-next-to-leading order*，arXiv:2311.16799v2，式 (3.9)、(3.11)。换算作者的 $\alpha_s/\pi$ 到 $a_s$ 时，一圈和两圈系数分别乘 4 和 16。
- 已存原文快照、原公式复算与哈希：工作区 `quark_gf_ref7_localnorm_NLO_NNLO_20261120/{THEORY.zh-CN.md,references,source_lock.sha256}`；当前固定耦合来源 `quark_gf_adj75_boot500_localnorm_NLO_NNLO_20261122/coupling.json`。

对 SU(3)、$n_f=3$，以 $t_f$ 的数值单位为 fm²，取 $r_f=\sqrt{8t_f}$、$L_f=\ln[2\mu^2t_f/(\hbar c)^2]+\gamma_E$，得到

$$
\boxed{D_V^{\rm NLO}=1+a_s(4L_f+10/3),\qquad
D_V^{\rm NNLO}=D_V^{\rm NLO}+a_s^2(26L_f^2+85.3591794985557L_f+30.92223469310622).}
$$

一圈的 $\ln432$ 相消；两圈双对数 $26=(4^2+9\cdot4)/2$ 是独立 RG 检查。采用 $\mu=2$ GeV、$\alpha_s^{(3)}=0.29812517844614284$；沿用 $\Lambda_{\overline{\rm MS}}^{(3)}=341(12)$ MeV 的五圈运行输入，不声称为最新世界平均。NLO/NNLO 使用同一耦合做固定阶比较，不把该结果称为 RGR 或 LRR。

若已做 H3 减除 $H_{\rm sub}=e^{m_F^{\rm NP}|z|}H_{\rm bare,norm}$，这里只乘 $D_V$ **一次**，不再乘完整 PT 指数；严格方案差仍可能含 $e^{(m_F^{\rm PT}-m_F^{\rm NP})|z|}$。非局域长线公式不用于 $z=0$，原点始终为 $1+0i$，不进入外推。局域电荷没有恢复，有限方案差没有闭合。

### 18.2 在 2.5 GeV fixed-$\lambda$ 连续数据上的应用

相同物理 $r_f,P_z,\lambda$ 下 $D_V$ 对三个 ensemble 是同一个确定性实数，故与线性 $a^2$ GLS 外推可交换（中心与逐 ensemble 的统计贡献同时缩放）。这次直接用 adjacent/ref75、cut2 的 H3 减除后连续结果，在七个共同物理流半径施加 NLO/NNLO。实际 $P_z=2.49968142$ GeV，$\Delta\lambda=0.506708$，本档 $\Delta z=0.04$ fm。

用 $r_f>0.15$ fm 的五个流点按 $H(t_f)=H_0+b t_f$ 做相关外推，全七点作窗口对照。不同 ensemble 独立，同 ensemble 的 500 bootstrap 行沿 flow、$z$、实虚部和阶数配对；协方差为三个 ensemble 分块贡献之和。该动量下五个主窗点的 $\sqrt{t_f}|P_z|/(\hbar c)$ 都大于以往诊断阈值 0.5，且上游三组裸比值链没有同时通过的非零点，因此截距仅为试算。若画外部 $m_0$，它只用于有限质量项敏感性，不是这批数据重新拟合的参数，不能作为完整 $\overline{\mathrm{MS}}$ 或物理 PDF 交付。

分阶段图册和逐点表：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_gf_cut2_adjacent_P25_continuum_NNLO_20261126/RESULTS.zh-CN.md`。
