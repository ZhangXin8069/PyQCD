# 对称、逐 z 自由能隙 two-state ratio 试拟合

按本次用户指定模型，对三个 ensemble 的 gradient-flow unpolarized connected u-d 重做拟合。输出为 scheme-B 的有限 flow 裸矩阵元。此目录为新分析，先前 Ratio/Sumratio/Sumdiff 和 C2 约束两态结果均保留。

$$R(T,t)=c_0+c_1\left[e^{-\Delta E(T-t)}+e^{-\Delta E t}\right]+c_2e^{-\Delta E T}.$$

每个 ensemble/flow/P/z/分量独立拟合 c0,c1,c2,DeltaE；实部 even-real 和虚部 odd-imag 分别拟合，不绑定能隙。源汇跃迁振幅依照指定公式共用 c1，c2 项保留。没有 C2 能隙约束、C2 分母或能隙先验。此式是 ratio 的有效激发态截断；逐 z 能隙不是独立确定的物理谱能隙。

## 输入、统计和窗口

沿用冻结的 2026-11-08 subset：L24x72 为30 conf，L32x64 为100 conf，L32x96 为40 conf，各200份 configuration bootstrap。共享同一 ensemble 的组态列表、抽样索引和 u/d 配对，未对拟合结果进行二次 bootstrap。完整组态列表及索引保留在各 flow NPZ。

输入 estimator_id 为 four_current_mean_shared_configuration_bootstrap_directional_complex_ratio_then_zpair_v1，scheme 为 same_time_second_cross_time_independent_product。gamma4/nopol；先按有符号 P 做复数 C3/C2，后作 z 对称及 +/-P 投影，分析 even-real/odd-imag。四 current anchor 共享配置重采样；未声称六动量方向平均。本地冻结的 gradient-flow ratio route 单独声明，未套用另一目录非 flow ratio_v4 的 estimator 标签。

固定 cut=2、Ta>0.65 fm；L24x72/L32x64/L32x96 的 Tmin 分别为7/8/9，分别对应0.735/0.7176/0.6975 fm。每个候选窗口从同一 Tmin 起始，Tmax 扫描 Tmin+1..12，使用每个 T 下 t/a=2..T/a-2。四个拟合参数，要求正自由度。

逐候选窗口对原始估计和全部200份 bootstrap 各自重新非线性拟合。给定能隙时三个振幅用完整协方差线性消元；能隙先在 log 域65点扫描，再细化最优网格区间。数值边界 aDeltaE=[0.02,3] 为冻结的搜索域，不是由 C2 得到的能隙先验。独立257点扫描及所有局部极小值细化对真实数据代表点作交叉验证。

协方差为 Schaefer–Strimmer correlation-to-identity，显式协方差 eigenmode 截断1e-12，relative/absolute jitter1e-12/1e-18；保存原始信息秩、调节后秩、条件数、收缩量及缩放四参数 Jacobian 奇异值。信息秩还受 Nconf-1 上限约束。未退回对角拟合。

质量门：Q>=0.05、chi2/dof<=3、能隙内点、四参数缩放 Jacobian 满秩。中心额外要求 chi2+1 的能隙 profile 在搜索域内两边闭合，且至少90% bootstrap 为内点/满秩。质量统计用于有限样本相关拟合诊断，不据此宣称基态隔离已完成。

每份 bootstrap 独立计算 AIC=chi2+2*4+2*N_omit，在通过该份质量门的候选窗口内归一化权重。当该份没有通过窗口时，按 skill 规定选完整200份 c0 的预计算标准差最小的有限候选；按原始 chi2 和候选序号打破平局。回退只用于保持完整矩形覆盖，仍标 DIAGNOSTIC。统计误差为选后200份 c0 的 std(ddof=1)，不除 sqrt(200)。

quality_accepted 表示中心及全部200份均由质量通过窗口选出；M_quality/M_boot_quality 与 M_selected/M_boot_selected 分开保存。odd z=0 是严格对称零，不拟合能隙，不计为 quality accepted 或诊断拟合。固定单一 cut 的 cut_systematic 保存为 NaN；window_systematic 单独保存。未估计 cut/Tmin/模型截断系统误差。

## Signal 图及纵轴定义

参考用户指定 Fit_Ratio/Fit_Ratio_BS/Fit_texp_mpi.py 的版式：每个 z 单独一页，横轴 (t_ins-T/2)/a，不同 T 不同颜色与点型，包含数据1sigma误差棒、拟合实线、配对 bootstrap 1sigma 色带和 c0 水平灰带。DIAGNOSTIC 在图中标注，c0 灰带加斜线。

$$y_{\min}=\min_{t=2,\ldots,T_{\min}-2}\left[R(T_{\min},t)-5\sigma(T_{\min},t)\right],\qquad y_{\max}=\max_t\left[R(T_{\min},t)+5\sigma(T_{\min},t)\right].$$

纵轴只由该页最小拟合 T 的保留插入区间决定，严格不因曲线或其他 T 的误差而扩张。c0 灰带或其他 T 的数据若超出范围会被裁剪。严格零的 odd-z0 页仅加 +/-1e-12 显示 padding。各 flow 的 signal_y_limits.csv 逐页记录数值，并已与原始 bootstrap 独立重算核对。

拟合曲线对每个候选窗口的非线性预测按对应样本的 AIC 权重平均；不把平均参数代入非线性模型。T 超过个别候选窗口的 Tmax 时，其贡献是该候选模型的外推预测。图注 gap(AIC) 是描述性的参数加权均值，不能等同于这条混合曲线的单一能隙。

概览统一选 tau64=0.5，即共同物理流半径 rf=0.1794 fm。Signal 概览展示 Pabs=4、z/a=0,4,8,12，两个分量各一页。完整图覆盖全部22个 flow、所有保留动量及所有 z。

| ensemble | signal 概览 | P4 even-real 全z | P4 odd-imag 全z | 裸矩阵元概览 | 全flow裸矩阵元 | 能隙诊断 |
|---|---|---|---|---|---|---|
| L24x72 | [PDF](L24x72/plots/signal_overview.pdf) | [PDF](L24x72/plots/tau0p36490204/Signal_unpolarized_u_minus_d_even_real_P4.pdf) | [PDF](L24x72/plots/tau0p36490204/Signal_unpolarized_u_minus_d_odd_imag_P4.pdf) | [PDF](L24x72/plots/bare_matrix_overview.pdf) | [PDF](L24x72/plots/bare_matrix_all_flows.pdf) | [PDF](L24x72/plots/gap_vs_z.pdf) |
| L32x64 | [PDF](L32x64/plots/signal_overview.pdf) | [PDF](L32x64/plots/tau0p50000000/Signal_unpolarized_u_minus_d_even_real_P4.pdf) | [PDF](L32x64/plots/tau0p50000000/Signal_unpolarized_u_minus_d_odd_imag_P4.pdf) | [PDF](L32x64/plots/bare_matrix_overview.pdf) | [PDF](L32x64/plots/bare_matrix_all_flows.pdf) | [PDF](L32x64/plots/gap_vs_z.pdf) |
| L32x96 | [PDF](L32x96/plots/signal_overview.pdf) | [PDF](L32x96/plots/tau0p66980978/Signal_unpolarized_u_minus_d_even_real_P4.pdf) | [PDF](L32x96/plots/tau0p66980978/Signal_unpolarized_u_minus_d_odd_imag_P4.pdf) | [PDF](L32x96/plots/bare_matrix_overview.pdf) | [PDF](L32x96/plots/bare_matrix_all_flows.pdf) | [PDF](L32x96/plots/gap_vs_z.pdf) |

裸矩阵元图：紫色菱形为本轮对称模型，实心为 quality-complete，空心为包含回退的诊断。上排与既有 Ratio/Sumratio/Sumdiff 比较，采用标明的放大窗口；下排显示本轮完整误差，范围较大时采用标记的 symlog。grey + 表示 exact symmetry zero。

## 覆盖及质量统计

以下统计覆盖全部 flow/P/z/分量；中心质量门、全部200份质量覆盖、对称零分别报告。

| ensemble | flow数 | 总点数 | 中心通过 | 中心及200份均通过 | 诊断点 | 对称零 | 200份数值完整 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L24x72 | 7 | 896 | 6 | 0 | 868 | 28 | 896 |
| L32x64 | 8 | 1344 | 106 | 0 | 1312 | 32 | 1344 |
| L32x96 | 7 | 1470 | 156 | 0 | 1435 | 35 | 1470 |

逐 z 自由能隙具有较强退化，尤其较大 P/z 及弱信号区域；边界解、未闭合 profile 和较大 c0 误差属于数值结果的一部分。完整覆盖不表示这些点能可靠提供基态矩阵元。请优先结合 signal、gap_vs_z 及逐候选 diagnostics 检查，不按误差最小的方法替换主分析。

本轮与三个旧方法的差异使用同一 bootstrap 配对计算，逐 flow 表 paired_audited_tau*.csv 保存 delta、paired_error、pull、correlation 和 method_spread。未把不同方法当作独立测量合并。

## 文件与核验

config.json 固定模型、窗口、搜索域与质量门；fit_source_lock.json/plot_source_lock.json/verify_source_lock.json 记录提交源码哈希。fits/ 每 flow 保存逐窗口、逐样本参数、AIC/权重、chi2/Q、秩、能隙内点、回退索引及200份矩阵元；*_diagnostics.npz 保存 profile、协方差、whitener、Jacobian奇异值；tables/ 保存中心结果与配对比较。receipts/ 保存每个拟合/绘图任务和产物哈希。

独立核验计数 1,825,251，含输入 estimator/配置/索引/NaN mask、u-d 配对、Tmin/候选窗口、逐样本 AIC 和回退重选、全部有限候选残差、能隙profile及质量状态、配对差误差和每页纵轴。共 202 个 PDF、3,769 页全部回读通过。代表页另行渲染目视检查。完整统计与文件哈希见 complete.json。

作业：拟合4011674；signal绘图4011691；裸矩阵元绘图4011696；最终独立核验 4011697。数值数据读取、拟合、测试、绘图及验证全部在 CPU Slurm allocation 中完成。
