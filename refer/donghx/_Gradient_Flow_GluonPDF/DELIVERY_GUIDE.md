# 本轮三 ensemble 结果与阅读顺序

生成日期：2026-11-08，Asia/Shanghai。会话：B 方案 quark PDF，100/40/30 conf，200 bootstrap 与探索性连续极限。

## 1. 已完成什么

三个 ensemble 均从配置级 C3/C2 重新生成了 200 份 bootstrap，并完成 unpolarized、helicity、transversity 的 connected u、d、u-d 裸拟合与各步后处理。
独立数值回读作业 3955054 已正常结束：2,844 个数值产品、270 个 PDF；[完成凭证](complete.json)保存了明确的 `physics_validated=false` 和 `controlled_continuum_limit=false`。
后置 CPU 作业3955107也已正常完成：270份PDF可读且SHA匹配，共1,647页，生成16张预览；已抽查其中8张代表页，见 [VISUAL_REVIEW.md](VISUAL_REVIEW.md) 和独立PDF验收凭证。没有声称逐页目视检查全部页面。

|ensemble|a [fm]|组态|bootstrap|bare flow 数|裸 selected 覆盖|
|---|---:|---:|---:|---:|---:|
|L32x64|0.0897|100|200|8|12,096/12,096|
|L32x96|0.0775|40|200|7|13,230/13,230|
|L24x72|0.1050|30|200|7|8,064/8,064|

selected 覆盖包含严格的 odd-z0 代数零；有值不等于通过拟合质量门。全部计算、拟合、绘图及 PDF 渲染安排在 CPU 作业；没有改动原 GPU 生产或覆盖旧分析。

具体 conf 清单、每个 flow 的原始路径、配置级投影的复用及 SHA256 均在 [input_plan.json](input_plan.json)。没有把旧拟合值再次 bootstrap 冒充配置重采样。

## 2. 图片入口

正确图片目录为 `plot_repair_v2_20261108/plots/`；父目录旧 `plots/` 是绘图中途失败遗留的部分图，不作为本轮完整交付。
完整 270 份 PDF 见 [PLOTS.md](PLOTS.md)。以下优先列 u-d；u 和 d 的同类图也在完整索引中。

### 各 ensemble 的裸矩阵元

每份 PDF 分页列出该 ensemble 的不同动量，页内对比不同 flow，保留实部及虚部。

|ensemble|unpolarized u-d|helicity u-d|transversity u-d|
|---|---|---|---|
|L32x64|[bare](plot_repair_v2_20261108/plots/L32x64/01_bare/unpolarized_u_minus_d_native.pdf)|[bare](plot_repair_v2_20261108/plots/L32x64/01_bare/helicity_u_minus_d_native.pdf)|[bare](plot_repair_v2_20261108/plots/L32x64/01_bare/transversity_u_minus_d_native.pdf)|
|L32x96|[bare](plot_repair_v2_20261108/plots/L32x96/01_bare/unpolarized_u_minus_d_native.pdf)|[bare](plot_repair_v2_20261108/plots/L32x96/01_bare/helicity_u_minus_d_native.pdf)|[bare](plot_repair_v2_20261108/plots/L32x96/01_bare/transversity_u_minus_d_native.pdf)|
|L24x72|[bare](plot_repair_v2_20261108/plots/L24x72/01_bare/unpolarized_u_minus_d_native.pdf)|[bare](plot_repair_v2_20261108/plots/L24x72/01_bare/helicity_u_minus_d_native.pdf)|[bare](plot_repair_v2_20261108/plots/L24x72/01_bare/transversity_u_minus_d_native.pdf)|

先检查 ratio 平台与拟合带：[L32x64](plot_repair_v2_20261108/plots/L32x64/00_ratio_fit_checks/unpolarized_u_minus_d.pdf)、[L32x96](plot_repair_v2_20261108/plots/L32x96/00_ratio_fit_checks/unpolarized_u_minus_d.pdf)、[L24x72](plot_repair_v2_20261108/plots/L24x72/00_ratio_fit_checks/unpolarized_u_minus_d.pdf)。
这些检查页展示共同物理 flow 的低、中、高三个代表值及 z/a=1,4,8；不是声称每一个 flow/z 均逐页画了原始 ratio。

### 三格距连续分支

|阶段|unpolarized u-d|helicity u-d|transversity u-d|
|---|---|---|---|
|07：共同物理运动学输入|[PDF](plot_repair_v2_20261108/plots/continuum/07_common_kinematics/unpolarized_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/07_common_kinematics/helicity_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/07_common_kinematics/transversity_u_minus_d.pdf)|
|08：格距拟合图|[PDF](plot_repair_v2_20261108/plots/continuum/08_a2_fit_checks/unpolarized_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/08_a2_fit_checks/helicity_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/08_a2_fit_checks/transversity_u_minus_d.pdf)|
|08：固定 flow 的连续截距|[PDF](plot_repair_v2_20261108/plots/continuum/08_continuum_fixed_flow/unpolarized_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/08_continuum_fixed_flow/helicity_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/08_continuum_fixed_flow/transversity_u_minus_d.pdf)|
|09：连续后的小流时间拟合图|[PDF](plot_repair_v2_20261108/plots/continuum/09_flow_fit_checks/unpolarized_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/09_flow_fit_checks/helicity_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/09_flow_fit_checks/transversity_u_minus_d.pdf)|
|09：连续且 flow=0 的矩阵元|[PDF](plot_repair_v2_20261108/plots/continuum/09_continuum_tau0/unpolarized_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/09_continuum_tau0/helicity_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/09_continuum_tau0/transversity_u_minus_d.pdf)|
|10：P0 诊断 m0 修正后，质量未通过|[PDF](plot_repair_v2_20261108/plots/continuum/10_continuum_m0_corrected/unpolarized_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/10_continuum_m0_corrected/helicity_u_minus_d.pdf)|[PDF](plot_repair_v2_20261108/plots/continuum/10_continuum_m0_corrected/transversity_u_minus_d.pdf)|

所有图实部 y 轴为 [-1,2]，虚部为 [-2,1]；不同 flow 横坐标微错开、误差棒不连线。空心三角标记质量/回退警告，菱形为精确约束点。固定范围外的点没有改数值，裁切情况记录在图片点账本。

## 3. 统计及拟合约定

- ensemble 内所有 flow、channel、flavor、P、z、tsep 和 P0/moving 共用同一份配置 bootstrap 索引；三套 ensemble 独立。
- 每份有放回抽取 Nconf 次，Nboot=200。原 ensemble 估计量作为中心，完整200行的 `std(ddof=1)` 作为统计误差，不除以 sqrt(200)。
- quark 的估计量是先平均四个 current anchors，再配置均值之比，即复数 C3 / 复数 nopol C2；不是逐配置 ratio 的平均，也不采用历史 gluon 的先取实虚部处方。
- 两个 z 方向与正负动量按冻结 convention 合并，保留 `even_real`、`odd_imag`。第二分量本身已是代表虚部的实数组，不能再取 `.imag`。
- 本轮采用 B 方案权重，权重恰好施加一次；不混入 A 方案。
- 原 .6 fm 规则在 L24 对应 T=6，但原始生产只有 T=7..12。本轮已提前注册统一更保守阈值 .65 fm，固定 Tmin 分别为 L64=8、L96=9、L24=7，实际最小距离为 .7176、.6975、.735 fm。没有制造缺失 T6，也不改变旧分析。
- direct Ratio 常数平台；endpoint cuts=1,2,3；固定 Tmin 扫 Tmax；候选 Q>=1e-3、chi2/dof<=1e3；cut 内按省略惩罚 AIC 混合，随后 cut 等权；每行独立重算权重。
- 预注册误差最小有限候选回退仅用于完整数据矩形，未通过质量门的回退有单独字段；若没有有限候选，仍须 NaN。

详细协方差、秩和每个候选记录见 [README.md](README.md)、`<ensemble>/fits/` 和 `00_audit/`；不按每个点的 Q 或误差挑最好看的方法。

## 4. 连续极限具体做法

共同物理 flow 为

$$
t_f=(0.0897\,\mathrm{fm})^2\tau_{64},\qquad
\tau_{64}=0.2,0.3,0.4,0.5,0.6,0.7,0.8.
$$

共同流半径约为 0.11346--0.22693 fm。L64 额外 tau=.1 只进入裸诊断。
没有把不同格距的相同 tau 当成相同物理 flow；实际 epsilon、nstep、时间支撑均沿用生产 manifest。

共同物理动量为 P=0、2.0、2.25、2.5 GeV；共同 z=0,.105,...,1.470 fm。
在各套真实正动量和非零 z 范围内做线性插值，不在 P0 与 moving 间插值，不跨特殊 z0 作非局域插值，不外插。二次插值仅为独立敏感性对照。

主模型先固定物理 tf、z、P 做

$$
H(a;t_f,z,P)=H_0(t_f,z,P)+c_a(t_f,z,P)a^2.
$$

三点两参数，dof=1。a-linear 是独立敏感性支，不同时拟合 a 与 a²。
不同 ensemble 的格距拟合协方差显式为对角，内部跨 flow 的相关性通过共享 bootstrap 保留；相同 replica 编号不代表三 ensemble 共同组态。

随后对连续截距做纯线性流外推

$$
H_0(t_f)=c_0+c_1t_f.
$$

至少4个 flow 点，全跨 flow 协方差，保留原始/有效/模型秩及正 dof。
`allflow_display_only` 为七点展示性支；`D_loose` 要求全部输入的 a/rf<1、rf/|z|<1、sqrt(tf)|P|<1。
小 z 的 D_loose 空窗不补零，不借 allflow 改名为通过尺度门；小于1不是远小于1。

每套固定 a 的 01--06 同时保留为对照：裸值 → 局域归一化 → 一次 NP 线减除 → 局域归一化端点 RGR/FO 转换 → 固定 a 的线性 flow0 → P0 的诊断 m0。
这些固定 a 的 05/06 不可当成三格距连续结果。

## 5. 必须连同结果一起阅读的限制

先看 [质量总表](00_audit/delivery_statistics_v1/SUMMARY.md)、[数值诊断](00_audit/interpretation_v3/SUMMARY.md) 和 [尚未闭合的系统误差](00_audit/KNOWN_SYSTEMATICS.md)。

裸中心回退比例（排除代数零）：L64=13.21%、L96=28.02%、L24=9.60%；对应所有点所有行平均 bootstrap 回退比例为26.08%、42.94%、18.13%。
几乎每点有“至少一行回退”，不等于200行全回退。严格上游标记要求每个贡献输入和分母都满足完整质量、无任一行回退，比仅中心 Q/chi2 通过严格得多。

固定 flow 的主 a² 支有7,056个有限非局域拟合，其中5,917个通过中心 Q>=.05、chi2/dof<=3；完整继承严格上游质量的计数为0。
连续后 flow0 的 allflow 有1,008个有限非局域点、838个中心质量通过；D_loose 为936个和786个；两者完整严格上游质量计数均为0。
这些是相关格点的覆盖/诊断计数，不是独立检验概率。

本轮 dF=.374137 为旧 L64 Wilson-relative 的固定且质量失败候选，用于共同中间方案 S，并非本轮三套独立测定；未计其系统不确定度。
局域归一化消去共同 Zchi，同时除去了局域荷；不能据此声称测到 Zchi=1 或绝对 MSbar 归一化已恢复。

连续分支 P0 短距离 z=.105,.210 fm、固定截距 b=0 的9组 m0 拟合全部有诊断数值，但0/9通过 Q>=.05、chi2/dof<=3，chi2/dof约699--7153。
不要把其很小的条件统计误差理解为质量良好，也不要将10阶段当成已经验收的 MSbar 矩阵元。
失败原因未唯一定位，不通过随意放宽门、添加自由截距或逐点调窗掩盖。

三套质量、有限体积、O(a)改善、共同物理 tsep/激发态、不同 Nev/source-smearing、时间支撑收敛、自相关、NP/PT 方案、绝对局域荷及 singlet mixing 均需额外核验。
transversity 外部相对算符符号仍是 `pending_external_physics_confirmation`。

## 6. 最具体的下一步

1. 先检查三套 ratio 平台及 Tmin/cut 稳定性，尤其 L96 的较高回退比例；若需要新模型，统一预注册窗口，或先用兼容 C2 约束能隙，不逐点挑选好看的平台。
2. 核实三套 pion/strange 质量、体积、smearing/Nev、O(a)改善和原始时间支撑；将这些差异与格距效应区分。
3. 比较 `model_sensitivity.csv` 中配对的 a²/a 模型及插值差异；仅三格距不支持同时自由拟合 a+a²。
4. 审查流窗和 dF 方案，核对 P0 短距离形状；当前 m0 检验失败，先定位模型/方案失配，再考虑补充短距离点或文献支持的扩展模型。
5. 上述门未闭合前，不将当前连续/flow0 诊断直接接成最终 lightcone PDF；本轮没有自动开展新的 lambda/傅里叶/LaMET。

## 7. 复用与安全恢复

冻结参数和代码：[config.json](config.json)、[source_lock.json](source_lock.json)、`code/`；原始输入身份见 `input_plan.json`。
投影/ratio/fit/01--06 数值在各 ensemble 子目录；三格距数值在 `continuum/07_common_kinematics`、`08_continuum_fixed_flow`、`09_continuum_tau0`、`10_continuum_m0_corrected`。

只查状态可在本目录执行：

```bash
/public/home/donghx/AnaConda/bin/python -B status.py
```

该入口只读配置/完成元数据和 Slurm，不读科学数组。生产已完成时不要重新整批提交，也不要修改已冻结脚本。
若需要新参数，复制为隔离新版本，先做 CPU contract/真实 fixture 验证，再提交到 CPU 分区。完整命令、原始及替代 jobid 见 `submission.json`、`postcheck_submission_v3.json` 和日志。
