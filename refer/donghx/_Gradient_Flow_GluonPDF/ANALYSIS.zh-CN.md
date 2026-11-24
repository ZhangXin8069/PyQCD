# P=0,4,5 self-ratio 常数拟合后的 gradient-flow 分析

日期：2026-11-16。沿用 `quark_gf_selfratio_constant_T0p8_1p1_bare_20261116`，不重新拟合裸矩阵元、不替换旧 campaign。三个 ensemble 共22个 flow，P=0,4,5，原有200组配置 bootstrap、跨 flow 配对及严格 0.8<Ta<1.1 fm/cut2 全部保留。

本轮已完成 Wilson-line 线性发散减除、NLO/严格NNLO有限GF转换、固定格距流时间诊断、固定物理流时间的连续拟合、先连续再流时间趋零的诊断。数值链执行完成；**受控物理结果尚未成立**：选中Wilson-line拟合均Q=0，尚未完成残余质量方案闭合，且共同物理尺度窗与裸拟合质量没有留下合格的连续外推点。

## 先看这些图

|ensemble|共同rf处的阶段对比|NNLO的所有flow|固定a流时间诊断|裸/尺度/回退质量|
|---|---|---|---|---|
|L24x72|[阶段对比](L24x72/plots/stages_reference_rf.pdf)|[所有flow](L24x72/plots/NNLO_all_flows.pdf)|[流时间](L24x72/plots/fixed_a_flow.pdf)|[质量图](L24x72/plots/quality_overview.pdf)|
|L32x64|[阶段对比](L32x64/plots/stages_reference_rf.pdf)|[所有flow](L32x64/plots/NNLO_all_flows.pdf)|[流时间](L32x64/plots/fixed_a_flow.pdf)|[质量图](L32x64/plots/quality_overview.pdf)|
|L32x96|[阶段对比](L32x96/plots/stages_reference_rf.pdf)|[所有flow](L32x96/plots/NNLO_all_flows.pdf)|[流时间](L32x96/plots/fixed_a_flow.pdf)|[质量图](L32x96/plots/quality_overview.pdf)|

每份图按P=0,4,5分页。阶段图在共同rf=0.1794 fm比较 bare、Wilson-line-subtracted、NLO、NNLO，并排显示leading与flow_c两种d；默认展示z≤0.84 fm。每个ensemble另有 `stages_reference_full_z.pdf` 与 `NNLO_all_flows_full_z.pdf`，保留完整z和完整误差条。不同曲线作少量横向错位。局域z=0是精确约束，P=0的odd-Im也只是对称性约束，不算成功拟合。

全局图：[微扰有限因子](plots/perturbative_factors.pdf)、[固定tf连续拟合](plots/continuum_diagnostics.pdf)、[先连续再流时间拟合](plots/continuum_then_flow.pdf)、[诊断截距](plots/continuum_flow_zero.pdf)、[预览](plots/review_contact_sheet.png)。所有诊断曲线保留误差；空心点不代表通过质量门。

## Wilson-line 减除的定义

基础表示、原始规范起点的Landau-gauge Wilson line。来源 `wilson_line_flow_hyp_comparison_20261115/gradient_flow/fits.json`，固定 global selection：block_size=1、flow_window=all4、z∈[0.3,0.7] fm、mode_cut=1e-8，比较leading与flow_c。

令 rf=sqrt(8 tf)，ln W = -d |z|/rf + finite，本轮用

`H_sub = exp(+d |z|/rf) H_self`。

这是单根open line，指数只用一次，没有额外乘2、CF或微扰line-mass指数。flow_c分支仅取其拟合d，不把c_flow另行扣除。每个ensemble使用自己的d。

|ensemble|Nconf|leading d ± JK SD|flow_c d ± JK SD|Q leading / flow_c|
|---|---:|---:|---:|---:|
|L24x72|30|0.32303897 ± 0.00004583|0.31072180 ± 0.00024064|0 / 0|
|L32x64|100|0.32978831 ± 0.00003743|0.36160341 ± 0.00020823|0 / 0|
|L32x96|40|0.28677600 ± 0.00003960|0.37246200 ± 0.00020717|0 / 0|

这些小的JK统计误差不包含模型误差。两种模型均Q=0（数值下溢表示极差的拟合），不能视为已批准的物理系数；全部最终quality_accepted=false。两种d之间的漂移单独保留。`line_window_transfer` 标记是否仍在原Wilson-line流窗（等效E32 tau=.3–.6），`line_length_window`标记原z窗，窗外使用是额外假设。

当前矩阵元bootstrap与Wilson-line JK没有共同配置协方差，d在200组矩阵元bootstrap中固定；另外保存d±1 JK SD的确定性响应，并用固定中央GLS权重传播。没有逐行拼接两种重采样，也没有编造统计总误差。continuum中三个独立d_e的响应分别保存。

## 严格NNLO转换

使用本地冻结论文快照：2610.40225v1式(40)重轻单端点平方得到Cpsi，2311.16799v2式(3.9)得到局域vector的zetaV（注意alpha/pi转alpha/(4pi)）。本轮网络读取未返回可用正文，没有据此宣称在线核实了最新版本。

ordinary flowed fields的绝对Zchi尚未测量；在同流时间、同vector current的局域归一化非单态比值中共同Zchi相消。因此可对**normalized nonsinglet shape**使用D_V=zetaV/Cpsi，不恢复绝对幅度或局域物理电荷。代码中共同ringed两圈常数显式相消。

令 a_s=alpha_s/(4pi)，Cpsi=1+a_s N1+a_s²N2，zetaV=1+a_s J1+a_s²J2：

`D_V = 1 + a_s (J1-N1) + a_s² (J2-N2+N1²-J1*N1)`。

这是展开到alpha_s²的严格比值，不使用未展开的倒数。一圈恢复CF(3L+5/2)，L=ln(2 mu² tf)+gamma_E，tf统一转换到GeV^-2。nf=3、alpha_s(2GeV)=0.30，三圈beta运行；保存mu=1,2,4 GeV固定阶比较。这些尺度结果不是已演化到共同mu的完整NNLO重求和。z=0始终保留(1,0)，非局域渐近公式不用于接触点。

在共同rf=0.1794 fm、mu=2 GeV：NLO因子1.11650720，NNLO因子1.15516080，NNLO/NLO-1约3.46%。这只描述有限GF转换；指数减除本身可能大得多，且依赖d模型。

## 每阶段矩阵元与误差

- `L*/tau*/01_bare/`：P045的self ratio、ordinary局域锚点及恢复的ordinary裸幅度。只有self ratio进入本轮后续转换；后两者仅作裸检查点。
- `L*/tau*/02_line/line_subtracted.npz`：两种d分支的原始中央值、200 bootstrap、SD、d±响应和line质量。
- `L*/tau*/03_GF_NNLO/converted.npz`：两种line × mu=1,2,4 × NLO/严格NNLO × Re/Im × P045 × z，分别保存中央和全部200样本、尺度mask及继承的bare质量。
- `L*/04_fixed_a_flow/flow_intercepts.npz`：线性物理tf模型的参数、200参数样本、窗口、相关GLS诊断。固定a下tf→0仅诊断。
- `05_continuum_at_fixed_tf/continuum.npz`：共同物理P=0,2.0,2.1 GeV与z=.315,.42,.525,.63,.735,.84 fm，各tf下a²主模型与a敏感性模型；保存source中心/方差、参数协方差及每个ensemble独立的bootstrap贡献。
- `06_continuum_then_flow/flow_zero.npz`：先连续后线性tf→0的参数、协方差、各ensemble独立样本贡献及d_e响应。贡献数组不是合成的配置bootstrap，不能逐行zip后求SD。

`M_selected`/`M_boot_selected`的轴说明随NPZ保存。per-flow bootstrap轴：line,mu,order,boot,component,p,z。固定a截距bootstrap轴：line,mu,order,policy,component,p,z,boot。跨ensemble不把第b行当作同一配置抽样；continuum误差为各ensemble贡献协方差之和，再构建跨tf完整协方差。所有bootstrap SD使用ddof=1，不除sqrt(200)。

[mu=2 GeV阶段CSV](matrix_elements_mu2GeV.csv)、[连续CSV](05_continuum_at_fixed_tf/continuum.csv)、[连续后流时间CSV](06_continuum_then_flow/flow_zero.csv)、[阶段清单](stage_manifest.json)与[数字汇总](numerical_summary_final.json)提供可读入口。上游signal保持原样，参见[链接清单](upstream_signal_links.json)和原裸campaign。

## 外推窗口及当前限制

按共同物理tf对齐三套数据；E32额外tau=.1仅保留在per-flow。相同整数P不是相同物理P，nonzero支只在P4和P5之间内插，P0不参与向非零P插值；z只内插，不外推。所有顶点和权重保存在 `interpolation_maps.json`。独立ensemble的统计误差按块传播；格距标定、质量/体积不匹配等系统误差不包含在这些统计误差中。

四个统一尺度门是a/rf≤.5、rf/|z|≤.5、sqrt(tf)P≤.5、tf Lambda²≤.1（Lambda=.3 GeV）；另要求原bare质量。固定a flow采用相关协方差的支持内收缩/SVD处理，先检查原始样本秩；先连续后的flow使用独立ensemble贡献之和形成的协方差，在原支持内作正SVD下限，不再套用相同收缩。所有协方差、白化矩阵、projector均保存供复核。

至少4个flow的保守支与all-flow展示支分开保存。L24和E32保守固定a支无可用截距；F32有少量组合可做固定a诊断，见[列表](fixed_a_eligible_diagnostics.json)。**三格距共同物理tf的conditional质量点数为0，连续后保守flow支不可用，保留NaN及unavailable状态。**all-flow诊断不能替代这些缺失结果。

本轮继承的constant裸拟合在许多点有大量bootstrap回退；参考flow处非局域点的回退比例中位数在三个ensemble均为1。中央值偶然通过不能掩盖此问题。详细[回退统计](bare_bootstrap_fallback_summary.csv)和[逐点尺度审计](scale_window_audit.csv)已保存。

GF NNLO是有限小流时间转换层，不是NNLO LaMET，不是物理PDF。残余质量m0/线性质量方案闭合、绝对Zchi/局域荷、后续LaMET及PDF重建均没有伪造为已完成。下一步优先重新审计Wilson-line模型与窗口、裸矩阵元的常数平台和bootstrap稳定性，再决定可用的共同物理尺度窗。

## 验收

所有科学数组读写、数值、拟合、绘图与验收在cpueicc CPU Slurm allocation执行。独立验收重算全部22flow的变换和200行样本、GLS正规方程、所有连续插值顶点、独立ensemble协方差和d响应。见[验收回执](receipts/verify.json)。

本轮产出 22 份PDF、70页，全部通过Poppler逐页光栅化与非空检查，见[图验收](receipts/plots.json)。人工图像抽查另记在visual_review.json。

冻结源码锁覆盖上游、配置、匹配公式和执行脚本。最初测试的list输入及试跑回执重复字段问题保留在历史日志，独立successor修复后通过；没有修改已提交源码。主要成功任务：4013641契约测试、4013649全部flow转换、4013650固定a flow、4013651连续后flow、4013678独立验收；绘图与终检任务见回执中的job字段。
