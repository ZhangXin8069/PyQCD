# L32x64 quark PDF：按 Pion_DA 最后一页组织的 gradient-flow 分析计划

日期：2026-10-22。性质：分析计划，不是已执行报告；本次不提交任务、不改动生产代码。

## 1. 目标与结果边界

用现有 L32x64、B 方案的 connected-quark 数据，完成 unpolarized、helicity、transversity 的局域归一化、线质量减除、适用窗口内的 GF 转换与固定格距零流时间外推试算。以非 singlet `u-d` 为主结果，`u`、`d` 为 connected 辅助结果；保留 even-real、odd-imag 分量。

本轮明确不做连续极限；不把 fixed-a 的拟合截距写成已经受控的连续零流结果；不把矩阵元或 GF-to-quasi 转换称为最终光锥 PDF。绝对电荷、lambda 尾部、傅里叶变换和 channel-specific LaMET matching 属于后续独立阶段。

当前正式流程按两个层次交付：

- 可实施层：冻结输入、检查质量、复用局域归一化与固定候选 d_F 减除、转换方向审计、有限 flow 的比较和 fixed-a 外推诊断。
- 条件层：非微扰线方案到目标方案的有限转换、短距离 m0 条件、最终 MSbar 标签。没有通过条件时必须保留“待确认”，不得靠强行拟合补齐。

## 2. 与 presentation 的逐项对应

原文件：`/public/home/donghx/Note/Gradient_Flow/Pion_DA.pdf`，PDF 第 25 页、幻灯片编号 24。

| 原流程 | 本项目本轮安排 |
|---|---|
| 1. Wilson loop / 静态势确定线发散 | 复用已有基础表示结果，先固定候选 d_F=0.374137；保留窗口/模型依赖 |
| 2. 规范固定 Wilson line 确定线发散 | 是另一种提取途径，不是第二次减除；本轮不新增测量 |
| 3. 线发散减除及局域归一化 | 已有配对 bootstrap 产品，先独立验算，再复用 |
| 4. a→0 | 明确跳过，记录 `continuum_limit_taken=false` |
| 5. GF scheme → MS matching | 必须改用局域归一化后的系数，并核对非微扰线方案的有限转换 |
| 6. t_f→0 | 做 fixed-a 的窗口/模型敏感性试算，不宣称严格连续零流极限 |
| 7. 拟合并处理 m0 z | 先审计 Pz=0 的矢量短距离条件；当前 z<0.15 fm 不够做有自由度的拟合 |

幻灯片中的 `pert,LRR` 是额外的微扰处方，不意味着本项目已有 LRR 实现。其 m0 的符号必须由拟合方程与本项目的指数约定重新对应。

## 3. 冻结输入：不重算 C3、不混用 A/B 方案

工作根目录：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF`。以下相对路径均以此为根。

| 产品 | 已有位置 |
|---|---|
| B 方案、带动量裸矩阵元 | `quark_l32x64_B_ratio20_boot100_T7_Q1em3_chi1e3_20261020_v2/` |
| 同方案 Pz=0 裸矩阵元 | `quark_l32x64_B_Pz0_ratio20_boot100_T7_Q1em3_chi1e3_20261021/` |
| 配对局域归一化 | `quark_l32x64_B_localnorm20_boot100_20261021_v2/` |
| 固定 d_F=0.374137 减除 | `quark_l32x64_B_localnorm_line_dF0p374137_20261021/` |
| z<0.15 fm 的 m0 可辨识性审计 | `quark_l32x64_B_m0_zlt0p15_audit_20261021/` |
| 1700 cfg / 3000 bootstrap 的线质量分析状态 | `wilson_relative_d_from_Tw_20261020/analysis/final_d_status.json` |

冻结约定：

- 格距 a=0.0897 fm，空间 32^3、时间 64；20 个 conf 为 `10000,10040,...,10760`。
- 100 个共享 bootstrap，seed=1115；保存原索引，不重新抽样或改变配对关系。
- 索引文件 SHA256：`86c034214702c29af6671956f4ff23c7749c4e34af103a9173d03da455b6056b`。
- 本轮共同流时间为 tau=t_f/a^2=`0.2,0.3,0.4,0.5,0.6,0.7,0.8`。移动态另有 tau=0.1，但 Pz0 没有；不为图形完整性伪造 Pz0 数据。
- 动量整数 n_z=0,3,4,5,6；P_z=(2 pi n_z)/(32 a)。带动量产品含正负动量配对，只有 z 方向，不称为六方向平均。
- 源汇间隔 T=7..12；物理起点 7a=0.6279 fm。复用原 ratio 常数窗及 AIC 混合结果，不擅自切换两态模型。
- 裸拟合的原门限 Q>=1e-3、chi2/dof<=1e3 保留。这些不是本轮流时间拟合已经获准的新门限。
- 原 B 估计量是每份 bootstrap 内的组态平均 C3 除以组态平均复数 nopol C2，再进行距离/动量投影；不是逐组态 ratio 的平均，也不替换成历史 gluon 流程的实分母。
- unpolarized：nopol / gamma4；helicity：pol35 / gamma5 gamma3，producer 因子 -i；transversity：pol15/25 的相应 gamma4 gamma_i gamma5 平均，producer 因子 +i。按 producer 的最终 even-real/odd-imag 标签读取，不能再机械地把 helicity 取一次虚部。
- transversity 的相对物理符号仍为 `pending_external_physics_confirmation`；计算流程可检查，但不能因拟合收敛而升级此状态。

初始阶段无需重新计算 GPU C3/C2，也无需为局域归一化分支补算动能 Z_chi。

## 4. P0：输入与质量审计

实现一个只读 manifest，逐文件核验 conf/索引哈希、坐标轴、通道、P、tau、B 权重、源汇时间、flow 参数及原上游状态。复用已有 basis/vertex 截断审计结论；将仍未通过的检查作为上游限制传递。

同时保存三类状态：数值完整性、拟合质量、物理方案是否确认。上游 fallback 不能在下游消失。当前固定 d_F 产品中，13230 个网格点有 13176 个继承了分子或归一化分母的 fallback 标记，因此“文件齐全”绝不等于“全部拟合可靠”。

核验局域分母的符号、零点穿越和信噪比；不能通过 abs、裁剪或丢弃 bootstrap 样本制造有限比值。使用全部 100 样本给出可用率；无效样本保留状态。

交付：`input_manifest.json`、`quality_summary.csv`、原始与归一化对照图。若索引不一致，停止相应分支，不能自动重排未记录的输入。

## 5. P1：复用局域归一化和固定候选线质量减除

固定同通道、同 flavor、同 P、同 tau 的局域分母：

$$
\mathcal M_\Gamma^f(z,P,t_f)
=\frac{h_\Gamma^{f,\mathrm{GF}}(z,P,t_f)}
{h_{\Gamma,\mathrm{even}}^{f,\mathrm{GF}}(0,P,t_f)}.
$$

共同 Z_chi 相消，不等于 Z_chi=1，也不恢复被除去的局域电荷。even 和 odd 都除以 even-real 的局域值。分别归一化后，一般有 M^(u-d) != M^u-M^d。

独立验算已有正指数线减除：

$$
\mathcal H_{\mathrm{NP}}(z,P,\tau)
=\exp\!\left[\frac{d_F|z/a|}{\sqrt{8\tau}}\right]\mathcal M(z,P,\tau),
\qquad d_F=0.374137.
$$

不再乘 C_F、不乘 2、不重复做一次线自能减除；该系数已经对应基础表示。z=0 的 even=1、odd=0 是精确约束，不作为独立拟合数据。

本分支的误差是“固定 d_F 条件下”的 100 bootstrap 误差。该 d_F 的上游拟合存在明显模型/窗口依赖，尚未被批准为最终系数。先用已有相邻窗口结果做单独敏感性包络；不把 3000 份 loop bootstrap 与 100 份矩阵元 bootstrap 按编号配对。正式联合传播须在组态交集上另行制定共同重采样；目前不伪装成已含 d_F 全误差。

交付：bare/local/line 三层统一对照图与逐点回读验证，不覆盖已有产品。

## 6. P2：先审计局域归一化后的 GF 转换，再实现系数

理论依据见 `/public/home/donghx/.codex/Markdown/divergence.md` 第 27 节、arXiv:2312.05032v3 Eq. (4.69)、(A.28)，以及 arXiv:2311.16799v2 的局域 V/A/T 系数。

先写明确的“正向”定义，令 L=|z|>0：

$$
Z_\chi h^{\mathrm{GF}}_\Gamma(z)
=C_\psi e^{-m_F^{\mathrm{PT}}L}h^{\overline{\mathrm{MS}}}_\Gamma(z),
\qquad
Z_\chi h^{\mathrm{GF}}_\Gamma(0)
=\zeta_\Gamma h^{\overline{\mathrm{MS}}}_\Gamma(0).
$$

因此原局域比值的目标转换是

$$
\mathcal R_\Gamma^{\overline{\mathrm{MS}}}
=\frac{\zeta_\Gamma}{C_\psi}
e^{m_F^{\mathrm{PT}}L}\mathcal M_\Gamma.
$$

一圈候选数据乘子（无质量非 singlet、相同 gamma5/张量有限处方，且在长线 SFTE 适用区）：

$$
\ell_s=\frac{\alpha_s(\mu)C_F}{4\pi},\qquad
\mathcal L_f=\ln(2\mu^2t_fe^{\gamma_E}),
$$

$$
F_{V/A}=1+\ell_s(3\mathcal L_f+5/2),\qquad
F_T=1+\ell_s(4\mathcal L_f+2).
$$

它们是 zeta/C_psi 一致展开至一圈的结果，不是原始 ringed 非局域 C_psi。将 mu、t_f 换为相容自然单位再计算对数。拟以 mu=2 GeV 为参考；alpha_s 的来源、N_f、阶数和尺度变化须先写入配置并审核，不沿用测试输入当物理常数。

本阶段有两个必须解决的定义检查：

1. presentation 另页公式的 C 与本项目的正向/逆向系数定义须逐行对齐；不能仅因都叫 matching 就直接复制符号。用 alpha_s→0、正逆乘积到 O(alpha_s^2)、局域系数比值及 ln432 消去作单元检查。
2. 已减去 m_F^NP=d_F/r_F 后，严格 PT 目标的代数关系仍含

$$
\mathcal R^{\overline{\mathrm{MS}}}
=F_\Gamma e^{(m_F^{\mathrm{PT}}-m_F^{\mathrm{NP}})L}
\mathcal H_{\mathrm{NP}}.
$$

   不能直接额外乘完整 PT 线指数；也不能在没有证明的情况下把质量差当成常数 m0。

定义工作中间方案 S：`H_S=F_Gamma H_NP`，用于本轮可比试算，明确标为“NP 线减除 + 端点转换的中间方案”，不是已完成的 MSbar。只有线方案与有限残余质量条件都核实后才能升级目标标签。可以另设互斥的完整 PT 转换分支作交叉检查，但绝不把两条分支串联减除。

长线系数不应用于 z=0，也不机械外推至 z~r_F。若要推进小 z 的矢量 Pz0 条件，须先审核完整有限 z/r_F 系数及局域分母的转换；不能由已有矢量结果猜测 helicity/transversity 的完整有限-flow kernel。

交付：系数推导和方向测试、通道/阶段明确的 NPZ、bare/local/line/S 四层对照；没有通过定义审核的结果保留 `scheme_status=pending`。

## 7. P3：先建尺度掩码，再扫描 fixed-a 的流时间窗口

共同七点对应 r_F=sqrt(8t_f) 约 0.1135..0.2269 fm，均满足此前展示条件 a<r_F<0.3 fm。但这只是较弱的窗口，不等于强层级已满足。

逐点保存：

$$
\frac a{r_F},\qquad \frac{r_F}{|z|},\qquad
\sqrt{t_f}|P_z|,\qquad t_f\Lambda_{\mathrm{QCD}}^2.
$$

特别说明 a^2/t_f=1/tau=1.25..5，目前没有 a^2/t_f<<1 的受控连续小流窗口。n_z=6 时 sqrt(t_f)P_z 约 0.527..1.054；这不是严格的小参数区间。拟合可收敛不消除这些限制。

建议预注册两层分析：

- D（固定格距诊断）：以共同 tau=0.2..0.8 为候选集，按预定尺度掩码，扫描所有连续、至少 4 个流时间的窗口。不得逐点看结果后挑“最好看”的窗口。
- S（较严格层级检查）：统一收紧 a/r_F、r_F/|z| 与 sqrt(t_f)P_z 阈值并报告是否仍有足够点。松判据可先用 r_F<|z|、sqrt(t_f)P_z<1；将 <0.5 等作为敏感性扫描，不把这些人为阈值写成理论定理。若没有合格窗口，保存 missing/diagnostic 状态，而不是放宽到必定出数。

参考拟合模型：

$$
H_S(z,P,\tau)=A(z,P)+B(z,P)\tau
$$

为主；至少 5 点时加入 A+B tau+C tau^2 作曲率检查。使用跨 flow 的配对协方差；保存原始样本秩、调节后有效秩、Jacobian 秩、SVD/收缩规则、chi2/dof、Q 和每份 bootstrap 的可用率。固定 z=0 约束不能当精确数据加入协方差。

同一数据窗内用 AIC 比较模型，所有 bootstrap 沿用相同流程；不同数据数的窗口不直接比较未经修正的裸 AIC。参考窗口的模型平均与跨窗口漂移分别报告，若未来要跨窗加权，先明确共同似然或窗口评分定义。流时间拟合不自动继承裸拟合的宽松门限，也不以 minimum-error fallback 伪装成通过物理条件的结果。

保留 tau0 样本、窗口结果和模型权重。误差分开给出 bootstrap、窗口/模型漂移、d_F 敏感性、微扰尺度变化；不将这些人为凑成未经定义的总误差。

禁止用 A+B tau+D/tau 然后把 tau=0 直接代入并报有限结果。在 fixed-a 下无法从单 ensemble 分离一般 cutoff 项；新增 bootstrap 也不会增加组态信息。

若数据表现出残留的 1/r_F 型线质量依赖、没有稳定窗口，回到线方案/系数审计；不能指望一个与 flow 无关的 m0 修复未正确消除的发散。

交付：每个通道/动量的 flow-fit PDF、全部候选窗 CSV、固定格距 tau0 截距图。标题及 metadata 都注明 `fixed-a diagnostic`、`continuum_limit_taken=false`。

## 8. P4：有条件地用 Pz0 固定有限残余质量

只在中间方案、有限端点和可用短距离窗均已核实后，优先选 Pz0、unpolarized gamma4、u-d 的 even 分量作为共同线质量的候选条件。Pz0 不意味着其理论目标为 1。

候选短距离目标须与目标 MSbar 算符/归一化一致，例如矢量非 singlet 的零 Ioffe-time Wilson 系数；不得照搬 gluon 真空曲线或 helicity/tensor 常数。详细候选式和来源见 divergence.md 第 27.6 节，尚需在代码中完成物理定义验证。

统一定义拟合符号：

$$
Y(L)=\ln\!\frac{H_{V,S}(L,0)}{\mathcal T_V(L,\mu)}
=b-m_0\frac{L[\mathrm{fm}]}{\hbar c}+\cdots,
\qquad
H_{\mathrm{target}}=e^{-b+m_0L/(\hbar c)}H_S.
$$

端点已经固定时 b=0，否则显式检查自由截距是否可辨识。不能凭 presentation 的参数名直接拷贝 m0 符号。

当前严格 z<0.15 fm 只有 z=a=0.0897 fm 一个非零点；z=0 是归一化约束。单参数只有零自由度，双参数秩不足，不能作为稳定拟合。现有审计因此未产生 m0 数值，这个状态要保留。

后续可另行批准扩大窗口，先审查 2a=0.1794 fm、3a=0.2691 fm，必要时将 4a=0.3588 fm 仅列为高 twist 敏感性点。固定截距至少需要两个独立非零点才有正自由度；自由截距至少三个。点数满足只是必要条件，仍须检查流半径、cutoff、微扰/高 twist、符号及相关秩。若没有合格窗口，不强行给“最终 m0”。

共同基础表示线质量应在运动学和端点正确分离后用于各通道、动量；先做通道一致性检查，不能给每条曲线独立调一个 m0 使其看起来平坦。对目前 fixed-a 分支，即使拟合可做，也只能报告条件性 m0，不是最终连续 MSbar 质量。

交付：可辨识性报告、短距离点/目标比值图、通过条件后的 m0 样本及共同应用对照。未通过则交付原因与 missing 状态，不造数。

## 9. 图形、数据结构与任务顺序

建议新建隔离分析目录（尚未创建/运行）：

`quark_l32x64_B_pionDA_workflow_20261022/`

子目录按 `00_audit/`、`01_bare/`、`02_localnorm/`、`03_line/`、`04_flow_conversion/`、`05_tau0_fixeda/`、`06_m0_conditional/`、`plots_final/`、`slurm/`、`logs/` 分层。原产品只读，不覆盖历史目录。

- 各通道以动量为不同子图；不同 flow time 轻微横向错开，marker/误差棒不连线，图例放图外。
- bare/local/line/S/tau0 的阶段标签显式显示，even/odd 分开；对称性零、fallback、尺度外点用不同标记，不悄悄隐藏。
- 所有已测 flow 都可展示，未进拟合的点用浅色空心点；拟合带只覆盖实际拟合区间，外推部分另标。
- 原生 z/a=0..20，并提供小 z 放大图；最终统一 PDF 保存于 `plots_final/`，不手工移动丢失 provenance。
- 继续保存 paired 样本与中心值，而不仅是 mean/error；沿用显式 `(component,boot,p,z)` 等轴定义，附单位、conf/索引哈希、系数方向、质量方案、质量状态。

建议 CPU 依赖顺序：输入审计 → 系数单元测试/只读验算 → 通道转换 → 流时间窗口拟合 → 条件性 m0 → 汇总绘图/回读验证。所有数值拟合和绘图提交到 CPU 分区 `a192c1t,cpu6248R,cpueicc,i72c512g`，不在登录节点计算。本计划本身不提交任务。

验收至少包括：原数据只读；100 bootstrap 顺序不变；z0 约束未当独立点；一圈正逆方向测试通过；指数无双重计数；fm/GeV 单位一致；所有原始/fallback/invalid/未拟合状态可追溯；文件回读后数值与阶段标签一致。

## 10. 推荐下一次执行范围

先执行 P0、P1 和 P2 的定义/单元测试，产出三通道、四移动动量加 Pz0 的统一对照。审核后进行 P3 的 fixed-a 外推诊断；P4 只在短距离目标与窗口门通过后执行。本轮无需等 L32x96，也不以新 GPU 测量为启动条件。

后续若用户要求进入 PDF 重建，再独立规划电荷恢复/归一化方案、lambda 尾部、傅里叶变换与三通道 LaMET matching。保持局域归一化时，默认只能讨论相应归一化形状；connected u/d 不标为完整 singlet PDF。

本计划受 gradient-flow-quark-pdf、quark-pdf-bare-matrix 与 pyqcd-statistics 的方案/配对统计约定约束：物理 scheme、可辨识性、拟合质量分开记录，不因数值完整而升级物理结论。

生成日期：2026-10-22。会话标题：L32x64 B 方案按 Pion_DA 流程进行固定格距 gradient-flow 分析。
