# L32x64：线自能与费米子归一化分阶段测试 Plan

日期：2026-10-15（Asia/Shanghai）。最新修订：**P2 十组态闭合圈运行中；按进一步授权提交独立 P4 B 单组态扩展窗口/C3/C2 pilot，Ward 与规范固定直线仍暂缓**。P1状态见 `line_mass_L32x64_pilot_20261015/latest.json`；P2状态见 `line_mass_L32x64_P2_B_20261015/latest.json` 及各自 frozen run；本轮P4状态见 `quark_l32x64_P4_B_pilot_20261015/submission.json` 和 `status/`。

本次范围锁定：所有费米子测试统一用 blending **B** 权重，$|n_z|=3,4,5,6$、$t_{\rm sep}/a=7,8,9,10,11,12$。保留正反动量以便原有对称投影。输出 namespace 为 `C3_schemeB_Pz3456_tsep7to12`；禁止以旧 A 结果补缺。纯规范闭合圈本身不含 blending 权重，记录 `not_applicable_pure_gauge` 并关联这一 B 分支。旧计划的 3–5、tsep7–10 范围不再用于本轮。

目标是分别回答：

1. 同一 Wilson-flow 轨迹上，Wilson-loop 势差和 Polyakov 测量能否给出相容的 leading 相对线质量？规范固定开口线推迟，首批不测。
2. 矢量 Ward 路线能否得到有信号、可复核的 $\mathring Z_\chi^V(t_f,a)$？
3. 在同一组态/方案 B/重采样定义下，absolute-Ward 和 $z=0$ normalized 两条 quark 路线能否闭合到相容的归一化 quasi 矩阵元？

理论依据和公式见 `/public/home/donghx/.codex/Markdown/divergence.md` 第 16–19 节。这里的数值阈值是 pilot 契约，不是普遍物理定理。首次仅启动 P0/P1；最新用户授权继续后，已提交有P1验收依赖的P2。P3 Ward的action/mass/clover/ZV仍待确认，P4 C3/C2未自动启动；全部后续费米子范围保留B、Pz3..6、tsep7..12。

## 1. 固定输入，不按结果挑组态

- ensemble：`beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64`。
- 几何 $32^3\times64$；格距 $a=0.0897\,\mathrm{fm}$ 为用户输入，未在本次重设尺度。
- thin gauge 根目录：

```text
/public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64
```

- smoke：conf10000；后续固定 10 conf：

```text
10000 10040 10080 10120 10160 10200 10240 10280 10320 10360
```

- 链接方向 $0,1,2,3=x,y,z,t$；格点坐标轴 $(t,z,y,x)$；元数据另外记录实际数组中 `mu` 的位置。
- 四维 thin-link Wilson flow，规范场周期边界；费米子时间反周期；不另加 HYP/stout 预流。
- 精确绑定 LIME 文件、格式/XML 几何、payload 大小、conf ID 和代码 hash；读取实际大 payload 与数值检查在计算节点做。
- 本轮不审计实时队列、不宣称旧 20/30/40 conf 的生产已经完成。

已有参考入口（只读，不改 frozen 脚本）：

```text
polyakov_L32x96/src/polyakov_flow.cpp
polyakov_L32x96/README.md
quark_l32x64_flow_20261007/reference/flow_common_l32x64.py
quark_l32x64_flow_20261007/results/
quark_l32x64_flow_20261007/flow_cache/
quark_l32x64_schemeB_window_20261015/README.md
quark_l32x64_schemeB_window_20261015/reference/
quark_l32x64_schemeB_window_20261015/summary/
```

这些路径以本项目根目录 `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF` 为基准。B 窗口已有测试仅覆盖 tau0.5、四个 current、tsep7 的诊断子集，不能当完整 B 生产矩阵元。

## 2. P0：定义与输入清单，先消除方案歧义

冻结 `manifest.json`，至少列明：

- `conf_ids`, `a_fm`, `geometry`, `gauge_input`, `flow_action`, `epsilon`, `flow_grid`；
- 基本/伴随表示、迹归一化、路径方向、反向链接、起点平均、各条路径长度；
- `gauge_fixing=deferred_not_measured`；未来若恢复 Landau 路线，再另行冻结其条件与算法；
- quark 的实际 valence action、质量参数、clover/改善系数、Dirac 作用量使用的链接及其涂抹；
- $Z_V$ 的来源、数值/误差、质量和改善条件、相关或独立误差处方；
- 局域 $\zeta_V$ 的 NNLO 实现、$n_f$、耦合/尺度和目标 ringed convention；
- C3 权重方案 `B`、原始时间对权重和 temporal-window tag；
- 每个分支的 `fermion_normalization_scheme`, `line_mass_scheme`, `finite_conversion_scheme`。

**待确认：**现有 perambulator 的文件名含 `hpy`，单凭名字不能断言实际 Dirac 链接处方；新 full-propagator Ward 测量必须先与真实 valence action 对齐。不能只由 ensemble 名字猜 mass、$c_{\rm SW}$ 或 $Z_V$。这些缺口不阻止纯规范 P1，但阻止将不一致的 Ward 数值用于绝对矩阵元。

## 3. P1：独立纯规范单组态代码与验证

已实现隔离包（新入口和 frozen-run 架构不改旧生产）：

```text
line_mass_L32x64_pilot_20261015/
  src/ code/ tests/ slurm/
  runs/<timestamp>/manifest.json
  runs/<timestamp>/validation/
  runs/<timestamp>/data/
```

### 3.1 流时间与轨迹

```text
tau = 0,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,
      .80,1.00,1.40,1.80,2.20,2.60,3.00,3.40,3.80
epsilon = .01 ; conf10000 再做 .005 的完整回归
```

每 conf/epsilon 一次读取，沿累积 RK3 轨迹在检查点同时测全部纯规范量；不按每个 tau 重新读场。$\tau=0$ 是恒等/原场诊断，不进连续 $1/\sqrt\tau$ 拟合。所有 tau 必须被精确到达并记录累计步数。

### 3.2 两类闭合圈测量，开口线暂缓

1. **Polyakov：**保存局域 `p3=(Tr P)/3` 复数与 `p8=(|Tr P|^2-1)/8` 有符号实数。局域构造后再平均；没有费米子负号。增加基本/伴随的 trace–trace 相关函数，距离 $R_W/a=1..12$、三个空间方向。保留 full correlator；connected subtraction 单独标记。
2. **Wilson rectangles：**三个 $xt,yt,zt$ 平面；$R_W/a=1..12$，$T_W/a=1..16$；记录不同方向，不预先强迫相等。基本表示复迹在取实部前保留；伴随字符由**同一个完整闭合 holonomy 的局域字符**构造后再平均，不用基本均值的平方替代。
3. **开口直线：暂缓，首批不实现/提交规范固定。**原计划 Landau 的参数仅作为未来设计记录，不是本次验收项。

### 3.3 单位场、随机规范变换与数值门

- 单位场：所有归一化闭合圈/直线为 1，零长路径为 1；实际值精度应接近 double roundoff。
- 反向路径与原 holonomy 的 Hermitian conjugate 相符；检查跨边界周期路径。
- 随机局域规范变换前后，闭合圈/局域 Polyakov 字符不变；开口线在未固定时应有端点协变，而非错误地被当作规范不变量。
- 在小合成格子核对 SU(3) 字符及显式 8 维伴随表示；保留负局域 p8 和基本复相位。
- 用独立 NumPy 实现对照短路径、LIME 大端读入、NPY 写出和 RK3。
- 拟议 double 门：链接 $U^\dagger U-I$、$\det U-1$ 的最大误差 $\le10^{-10}$；闭合圈规范变换前后最大绝对差 $\le10^{-10}$。若测试累积长度需要调整，先记录原因和新门，不能看结果后静默放宽。
- 步长门继承 Polyakov pilot：局域 p3/p8 最大绝对差 $\le2\times10^{-5}$、plaquette 差 $\le10^{-6}$；新增 Wilson 平均和 Polyakov 双圈最大绝对差也固定 $\le2\times10^{-5}$。不对接近零的量用相对误差作唯一标准。十 conf 后另检查积分误差相对于统计误差。链接 SU(3) 门 $10^{-10}$；长 holonomy 累积误差另记录并以 $2\times10^{-8}$ 作防护上限。
- 本轮不做 Landau/Gribov 测试。局域随机规范变换下闭合圈不变门在合成场独立路径测试中执行。

P1 汇总会输出单 conf 的有效势、固定 $T/a=8..16$ 的无权 OLS 斜率及相对 $d$ 候选，**全部标为非统计诊断，不给 bootstrap 误差，不用于实际减除/matching**。时间平台和组态统计须由 P2 检查；符号跨零的 Polyakov 不取模、不裁剪、不强行取 log。

以上均在 CPU 计算节点运行。登陆节点只做轻量元数据/文档检查，不做真实场 gauge flow、gauge fixing 或合成数值回归。

## 4. P2：十组态相对线质量，不急于宣布绝对 d

P1 放行后，冻结同一测量程序用于 10 conf 主步长 .01。纯规范首轮输出可很小：仅字符/起点平均、二点相关函数、方向分量和质量诊断，不存每个 checkpoint 的完整 gauge 或 holonomy 矩阵。

参考流时间取 tau .50，另以 .80 重做。依次比较：

$$
\ln\frac{L_R^G(L,t_f)}{L_R^G(L,t_{f,\rm ref})}
\simeq-\Delta m_R L+b_R(t_f,t_{f,\rm ref})+\text{有限长度项},
$$

$$
\frac{\Delta V_R(R_W;t_f,t_{f,\rm ref})}{2}
\simeq\Delta m_R,
$$

$$
-\frac1{L_t}\ln\frac{\mathcal L_R(t_f)}{\mathcal L_R(t_{f,\rm ref})}
=\Delta m_R+\Delta F_{Q,R}^{\rm sub},
$$

$$
-\frac1{2L_t}\ln\frac{G_R(R_W;t_f)}{G_R(R_W;t_{f,\rm ref})}
=\Delta m_R+\tfrac12\Delta F_{Q\bar Q,R}^{\rm sub}.
$$

最后两式不能假定有限自由能变化为零。静态势先验证有限 $T_W$ 平台；缺平台不得给 Cornell 截距强行解释为线质量。若采用 Cornell 模型，限制基本表示距离窗，伴随屏蔽/破弦不照搬。

候选空间窗口预先列 `n=4..8, 6..12, 8..15, 10..15`（具体观测量受其最大长度限制），然后统一施加 $r_f/L\le.5$ 的探索性 mask。流时间窗比较 tau $.5..1.8$ 与加入 $2.2..3.8$ 的变化，并展示所有未入拟合的点；小 tau 是 cutoff 诊断，大 tau 是 finite-flow/IR 诊断，不预设哪段一定有效。Landau 开口线在大 L 没有通常的 transfer-matrix 保证；同时看短/中/长窗，不将最长窗口自动选为基准。

先拟合少参数的 $\Delta m$；再比较 $\Delta m\simeq d_R(1/r_f-1/r_{f,\rm ref})$、全局窗口变化、额外有限长度/格距项。报告不能分离的参数，不用高阶多项式拟合掉所有残差。基本/伴随分别测量，不将精确 $9/4$ Casimir scaling 当约束。

若单圈没有正 ensemble 信号或大量 bootstrap 跨零，不取模/截断/删样本。可在原始均值空间做相关模型检验，但缺信号时仍写 `signal_unresolved`。单格距绝对 residual mass、有限转换未固定时，只发布 `relative_line_mass_diagnostic`。

## 5. P3：独立 Ward normalizer，先单 conf 再十 conf

拟建隔离包 `quark_Zchi_Ward_L32x64_pilot_20261015/`，与现有动能包和 C3 生产分离。

### 5.1 full-propagator mixed 两点函数

- 固定 P0 核实的物理价夸克 action/mass；不要复制 L32x96 的临界质量。
- 优先用独立点源/full-spin-color propagator 验证局域 Ward 测量，避免将有限 distillation 子空间的投影算符悄悄当完整局域矢量流。
- conf10000 先取源时间 $t_0=0,32$，空间点固定 $(0,0,0)$；若信号和成本可接受，十 conf 用 $t_0=0,16,32,48$。源数是同 conf 的测量数，不增加 Nconf。
- 每个源以 12 个 spin-color RHS 求 unflowed propagator；同一源解复用于全部 flow checkpoints。源可先用点源，若后续加 Gaussian smearing，flowed/unflowed 比值必须使用完全相同的 unflowed smeared source，另记处方。
- 四维 fermion flow 和同步 Wilson gauge flow；测 $t_f=0$ 及 tau .05,.10,.20,.30,.40,.50,.80,1.00；步长 .01，smoke 对 .005。
- 汇端 $V_i=\bar\chi\gamma_i\chi$、$i=x,y,z$，源端未流化。输出完整复 C2，保留逐源、逐方向，不先取绝对值。
- 验证反周期费米子边界、gamma/相位、mixed 与 both-flowed 的区别；不将 $KSK^\dagger$ 的双端流传播子拿来冒充 mixed sink/unflowed source 的定义。

### 5.2 平台与 ZV

在 $\Delta t$ 远离源的 flow-overlap、激发态和反向污染区做相关常数拟合；先画出三个方向的 $C_i(t_f;\Delta t)/C_i(0;\Delta t)$。不把不同方向或源时间当独立 conf。

然后计算

$$
\mathring Z_\chi^{V,(b)}(t_f,a)
=\frac{\zeta_V(t_f)Z_V^{(b)}(a)}{R_V^{(b)}(t_f,a)}.
$$

若 P0 找不到同 ensemble/action/质量处方的可靠 $Z_V$：

1. 仍可测 $R_V$ 并发布 $\mathring Z_\chi^V/Z_V=\zeta_V/R_V$ 诊断；
2. 单独设计局域矢量电荷归一化，优先采用有可靠来源的同处方 $Z_V$，或通过带已知 flavor 电荷的强子前向矩阵元测量；
3. 任何新三点函数、sequential solve 或额外质量扫描均列为下一阶段，不冒充本轮已具备；
4. 缺失 $Z_V$ 时禁止输出声称有绝对归一化的 after-Ward 矩阵元。

改变 plateau 窗、方向、源、尺度和 perturbative order；记录质量依赖和有限流时间外态依赖。无需为 Ward normalizer 本身强行做动能方案的无质量真空反演，但这不免除 action/mass 和 $Z_V$ 的一致性。

## 6. 共同统计契约

- conf 内先按固定处方平均源和起点，conf 才是 ensemble 重采样单位；必要时按轨迹 bin，并报告十 conf 对自相关检验的局限。
- 首轮保存 100 组 bootstrap，seed=20261015，索引 shape=(100,10)，每组有放回抽 10 次。可用 delete-one jackknife 交叉检查；100 个 replica 不是 100 个独立组态。
- 纯规范量先按共同索引做 ensemble 均值，再取可定义的 log/比值；不平均逐 conf 对数。
- Ward 用每个 bootstrap 的共同源关联函数均值构造 mixed-C2 比值，再拟合平台；不是对每 conf 求 $1/R_{V,c}$ 再平均。
- 若与现有 PDF 联合传播，先对齐原始 conf 数据再用同一索引重新采样；不得把不同统计集的第 b 个样本假配对。
- 保留用户现有 nucleon ratio 的“逐 conf 分量比值后重采样”估计量作为单独产品契约，不因新 Ward C2 比值的定义而静默改写。
- 10 conf 的原始线性样本 covariance rank 至多 9；非线性 bootstrap 变换可能产生更高数值 rank，但那不等于新增独立组态信息。必须报告原始样本秩、所用正则化和模型 Jacobian 秩，不能靠增加 replica 或调 SVD cut 伪造信息。
- 只有通过有限性、样本信息、effective rank 和 Jacobian 可辨识门的模型才比较 AIC；明确记录 chi2/dof、Q、每 replica 状态和 failed count。不得强行给每个 bootstrap 填入一个“误差最小”解来掩盖本次测试中的不可辨识。
- AIC 候选窗仅来自可信单态/几何区，中央样本确定窗口、covariance 处理和权重后，在 paired replica 中保持相同定义。窗口/模型系统误差与统计误差分列。
- 复数拟合使用显式实/虚堆叠及对应 covariance；log 幅度诊断另保留相位和过零标记。

## 7. P4：将测量接到 quark 三极化矩阵元

最初比较 tau $.2,.5,.8$，$|n_z|=3,4,5,6$，$z/a=0..15$（原始数据保留 $|z|/a=0..20$）；主对象 u-d，u/d connected 另存诊断。**本 Plan 全部 C3 用 B 权重，原始时间同帧二阶表、跨帧一阶乘积，仅乘一次。**unpolarized、helicity、transversity 使用现有明确的 gamma、复数投影和运动学 convention，不为归一化改符号。$t_{\rm sep}/a=7..12$，即 $.6279..1.0764\,\mathrm{fm}$；只有 tsep7 的窗口 pilot 不足以冒充完成这一拟合。源汇 C2 定义不因 B 改动，也不额外套 insertion omega。

所有新 C3 都使用用户指定的方案 B：同原始时间用二阶表，不同原始时间用一阶权重乘积；权重只施加一次。旧 A 仅作命名明确的控制。如果完整 B 输入不存在，先补所需子集并做该 tau 的窗口检查；tau .5 的 w4/5/6 结果不自动批准 tau .2/.8 或全 current。

分别保存以下分支，不覆盖旧图/旧 bootstrap：

1. `bare_GF`：普通 flowed field 的裸矩阵元；
2. `normalized_z0`：同极化/动量的非零局域分母，公共场因子相消；
3. `absolute_Ward`：一个 $\mathring Z_\chi^V$ 乘裸矩阵元；$Z_V$ 缺失时不发布此分支；
4. `relative_line_subtracted`：从命名明确的 normalized 或 Ward parent 派生，只相对参考流时间减除 $\exp(\Delta m_F L)$，不冒充绝对 MSbar；
5. `scheme_converted_diagnostic`：只有有限 mass scheme、端点/局域系数和非局域转换对齐后才生成。

局域分母必须有信号，尤其不能用对称性强制为零的已选虚/实分量做归一化。保存原复数及分母分布；必要时先恢复正确的通道相位/运动学定义再比较，而非临时改相位求非零。

验证：

- $h_\Gamma^{\rm Ward}(z)/h_\Gamma^{\rm Ward}(0)$ 与 `normalized_z0` 在同一 bootstrap 中应代数相等；这只验实现，不独立证明 Ward 系数正确。
- 比较 $\ln|\mathcal D_\Gamma|$ 的长线斜率与纯规范 $\Delta m_F$；逐动量/极化的有限流时间差必须作为物理诊断保留。
- 比较 $\zeta_\Gamma/C_\psi$ 的 normalized 转换与 absolute-Ward 转换后再除局域电荷的结果；不重复乘 Zchi，不重复减线质量。
- 胶子只比较伴随纯规范线质量；不将夸克 $Z_\chi$ 用于 TU/SU/TH/SH，也不在本轮替换其 sector matching。

### 7.1 单格距的尺度门

逐点保存 $a/r_f$、$r_f/|z|$、$\sqrt{t_f}P_z$、$t_f\Lambda_{\rm QCD}^2$；$z=0$ 使用局域分析，不套非局域 mask。阈值应全局扫描、不可逐点调到想要的结果。

例如探索性取 $r_f\ge2a$ 和 $\sqrt{t_f}P_z\le.5$，对应 tau $\ge.5$，而 $n_z=4,5,6$ 分别要求 tau $\lesssim.405,.259,.180$，没有交集。$n_z=3$ 也只剩很窄范围。这不否定数据能做诊断，但说明当前格距和动量下不能保证强尺度分离。

纯规范线质量提取没有 Pz，因此不受后一条件；把其用回 PDF 才受完整限制。单格距结果始终标 `single_a_diagnostic`，不宣称完成连续极限或物理 PDF。

## 8. 输出、运行地点和放行

建议测量产品分开：

```text
gauge_loops_v1:      per-conf/tau/direction/representation 的复/实圈和矩形圈
gauge_fixed_line_v1: per-conf/tau/copy/direction/sign/length 的直线字符
ward_mixed_C2_v1:    (conf,flow,source,vector_direction,delta_t) complex128
ward_Zchi_v1:        plateau、ZV、zetaV、Zchi、bootstrap、covariance 与状态
line_mass_fit_v1:    reference、Delta_m、可选 d、窗口、rank、copy/step/模型敏感性
normalization_comparison_v1: 各分支、paired indices、scheme/mask 与图
```

保存 code/input hashes、完整性 marker、raw schema、axes/units、误差来源，不存凭据。每个阶段有独立 `validation/report.json`，区分实现回归、数据可辨识、拟合质量和物理解释。

CPU 作业：优先 `a192c1t`；`cpu6248R`、`cpueicc`、`i72c512g` 经编译/运行环境预检后可纳入。纯规范 smoke 可先申请 48 cores/64 GiB/12 h 作为预算上限，实测后调整十 conf 资源；不是已测耗时。拟合/汇总只在 CPU 计算节点。

GPU 作业：`na800-sug`，必须有 `#SBATCH --reservation=donghx_223`；资源由 full-propagator smoke 的显存/耗时决定，不未经测量继承四卡 C3 的大内存申请。所有数值工作在计算节点，登陆节点只 staging/metadata/submit。不会擅自加 array `%N` 上限，也不因这个 Plan 自动取消别的任务。

保留现有 basis/gauge 缓存和全部最终数据；新 vertex/VVV 在作业自己的 scratch 重构后清理。若规范固定场只用于短路径，测完就释放；不把 23 个 checkpoint 的 full gauge 长期落盘。

第一批交付图：各 tau 圈信号、$\Delta m$ 三路线对照、步长/copy/window 敏感性、Ward C2 平台、Zchi 与误差、bare/normalized/Ward 矩阵元及 double ratio。无信号或无可辨识拟合的图必须显示原因，不用空白点假装完成。

## 9. 当前待确认与停止条件

1. [7] 实际 gauge fixing/Polyakov 颜色缩并不详；本 Plan 是明确的新测试，不声称逐项复现对方。
2. L32x64 valence action/mass/improvement、$Z_V$、$\zeta_V$ 的实际实现与尺度输入待核实。
3. 低温 Polyakov 信号和 Wilson-loop 时间平台尚无本轮数据保证。
4. NP 绝对 residual mass 条件、NP-to-MSbar 有限转换与高阶剩余项未完成。
5. 完整 B C3、各 flow temporal-window 和 transversity 外部符号门未被本 Plan 自动解决。
6. 十 conf 可能只够验证算法和识别噪声；不足时报告限制并提出扩样，不承诺强行给最终 d 或 Zchi。
7. 一旦 P0 发现 action/方案不一致，隔离诊断结果，修正定义后再生成后续物理分支。

实施状态：隔离包 `line_mass_L32x64_pilot_20261015/` 已实现并提交。当前 frozen run 为 `runs/P0P1_B_2026-10-15T183837.560011_0800/`；gate3853766已通过，双步长数组3853767、汇总3853768仍须等待实际测量及完整audit。1280个eigen/peram头无缺口；valence action等物理项仍待确认。先前XML尾部NUL和跨节点GLIBC失败已隔离；当前版本每实际节点编译并独立回归，不跨GLIBC复用binary。

资源调整：数组3853767在pending时从48核调为24核，每节点最少CPU也为24；64GiB/48h上限不变。源码和物理参数未改，详见run内 `resource_update_24cores.json`。

后续实施（2026-10-15 19:05）：已冻结P2包 `line_mass_L32x64_P2_B_20261015/`；run为 `runs/P2_B_2026-10-15T190543.301835_0800/`。gate3853774依赖P1 audit3853768成功，缺失九conf数组3853775依赖gate，十conf统计3853776依赖数组。conf10000/epsilon.01只读复用并做文件hash核验，其余九conf不重排、不按信号挑选。

P2仍用全部23个tau、epsilon.01；每任务24cores/64GiB/48h上限，每实际节点编译和独立回归。拟合/出图2hCPU上限。不设置array并发上限。首个conf10000实测主步长总约821.24s（24核cpu052，含全部测量、不含节点编译回归）；不能把它当所有conf/节点精确耗时。

统计冻结：1000组配对bootstrap、seed20261015、每组抽10conf；先ensemble均值后log，原十conf delete-one jackknife协方差，显式删除相关矩阵小于最大本征值1e-8的模式，另记原始秩（1e-12）；不以bootstrap数量扩增样本信息。固定T窗6..10/8..12/10..16/8..16，ref tau .5/.8，R/a4..12及max(rf,rfref)/R<=.5；flow窗.5..1.8与.5..3.8，d-only及加c(tau-ref)诊断。任何不正/无信号replica均不abs/clip/drop或强制fallback。

19:10最新核对：P1 audit3853768已成功完成，23点均PASS；p3/p8最大差9.16e-6/9.79e-6，Wilson最大4.28e-7，plaquette最大2.21e-7。P2gate3853774已释放并在CPU计算节点运行。

19:12更新：P2gate3853774已COMPLETED/0:0，独立统计/出图契约PASS；数组3853775已放行，当前等待CPU资源，3853776依赖数组。未把提交或gate成功当十conf测量/物理d拟合完成。

19:14更新：数组task0..3已在cpu052/cpu058/cpu059/cpu054实际运行、日志有真实测量输出；task4..8仍Priority排队；3853776仍依赖测量。

最具体下一步：**核对P2gate，再等待九conf及 `analysis/report.json`/诊断PDF，审查Wilson有效势平台、参考/距离/flow/T窗口、signed Polyakov与rank失败。**P2产物仍是单格距相对线质量诊断，不自动用于matching。Ward的R_V与ZV/action核验是下一物理门；完整B C3/C2保留 $|n_z|=3..6$、tsep7..12，禁止混用A。

## 10. 19:30 后续授权：P4 的独立裸 C3/C2 pilot

最新请求“继续按 Plan 提交计算任务”后，已提交隔离包
`quark_l32x64_P4_B_pilot_20261015/`：CPU gate3853791 → GPU数组3853792_[0-2] → CPU审计3853793。
GPU固定na800-sug、reservation=donghx_223，4GPU/240GiB/72h预算，无array并发上限。
CPU gate/审计在四个允许的CPU分区运行，登陆节点仅做源码/输入头检查与提交。

本轮三个task均为conf10000，tau=.2/.5/.8，epsilon.01，nstep20/50/80；
窗口分别3/4/5、4/5/6、6/7/8。原tau.8标准缓存实际半宽6，未误用此前L32x96的窗口5。
Pz=±3,±4,±5,±6，tsep7..12，四个current0/16/32/63，每sep中间插入dt=floor(sep/2)，
双位移/0..20、四极化、400current/200源汇、六Wick均保留。
这是先补所需子集和各tau窗口检查，不是全时间B生产，也未进行裸拟合/重采样/重整化。
它不依赖Ward参数或尚未审查的P2线质量，因此可独立推进；不代表P3门已解除。

输入/参考源码hash已冻结；旧A/B与原basis/gauge不改。
重新完整4D求解响应、只存选定输出，约48.05GiB新basis保留；vertex只在GPU内存、VVV当次重算。
先过CPU六Wick/权重/时间标签门，再做basis重叠回归；
tau.5/tsep7/P+3,+4,+5与旧B逐块/逐项/总C3回归；三个tau的各窗口P±3/P±6有真实因子化对照。
最终CPU审计72个time-case和跨flow的未流C2一致性。
所有结果命名为 `C3_schemeB_Pz3456_tsep7to12`，不以旧A补缺、不把普通flowed fields称ringed。

19:30即时队列：gate3853791已在cpu073运行，GPU/审计依赖等待；
P2九个新conf中task0/1/3已COMPLETED/0:0，task2/4/5/6运行，task7/8排队，统计3853776依赖等待。
以上为该时刻状态，不是最终完成声明。

19:32核对：CPU gate3853791已COMPLETED/0:0（27秒），独立六Wick/权重/降维与新增标签/输入门PASS；
GPU3853792三个task均RUNNING（task0/1 gpu043、task2 gpu044），实际日志进入thin-gauge四维费米子流。
未检出启动Traceback/CUDA错误，17份冻结源码hash复核PASS；GPU窗口回归本身尚待完成。

下一步：先审查三个tau的单conf窗口/扩展动量和时间边界回归，再扩固定十conf集评估配对差与统计误差；
之后才补全current/insertion的完整B生产。Ward实际action/mass/clover/ZV和有限归一化仍待确认。

## 11. 20:58 显存修复与全部规范场扩统计

用户明确授权修复 C3/C2 OOM，并把 Wilson/Polyakov CPU 测量扩展到所有可用 conf。
原 P4 tau.2 已完成24例；tau.5/.8 在全旋量 C2 的8动量批处理中申请16.384GB失败。
旧源码、48.05GiB basis、已完成结果均保留，未删除/重流。

隔离后继 `quark_l32x64_P4_B_memfix_20261015/`：
CPU gate3853821已PASS；GPU3853822_[1-2]运行；汇总3853823依赖等待。
逐动量完成全旋量/降维 C2，再释放显存进入 J/A/B；独立因子化不再复制整块双时间 J。
保留 B 权重一次、所有正负动量/六Wick/窗口和检查，CuPy70GiB限制不变。
两个流时间均通过全部192个C2对照后进入vertex/C3；完整C3验证仍待case receipts。
汇总复用旧tau.2并检查全部72例/跨flow未流C2。旧永久依赖3853793已取消，无数据删除。

纯规范隔离包 `line_mass_L32x64_allconf_B_20261015/`：精确清单1700个raw文件，
conf1000..17990；复用已完成10个，新增1690个。23tau/epsilon.01/380steps及
Wilson(三平面,R0..12,T0..16)/Polyakov(完整64)/pairs定义和C++源码hash完全不变。
CPU gate3853826检查全部LIME头、复用10conf、数值/统计契约，已PASS。
纯规范扩统计不要求eigen/peram齐全。

1000项数组被账户AssocMaxSubmitJobs=200拒绝，未创建该数组。
改为CPU数组3853828_[0-168]，每项24core/64GiB顺序处理10conf，覆盖1690缺口；
无%并发限额，遵守账户限制。worker映射/源码另行冻结，未修改已提交的run源码。
统计3853829在全数组afterok后运行：1700conf共享1000bootstrap/seed20261015，
block2泛化为850块、不丢conf；log-after-mean/无abs或删坏副本/rank门仍保留。
新实际测量已启动，1700conf全部完成及最终d未获确认，不能直接用于matching。

结果根：`line_mass_L32x64_allconf_B_20261015/runs/allconf_B_2026-10-15T205426.224554_0800/`。
下一步检查GPU48个新case和CPU各conf完成守卫，完成后审查窗口/信号/相关性，
再决定10conf完整B C3/C2；Ward/Zchi/有限scheme门未解除。

21:10更新：tau.5新GPU已COMPLETED/0:0（17m24s），24/24case回读PASS，
max C2 relativeL2=1.28260e-14；12项物理因子化max=1.00279e-14；
对旧B逐ordered-Wick max=9.32626e-15。
tau.8已有16/24case和全部12项因子化PASS（max=1.09117e-14），仍运行、未见OOM。
CPU gate3853826已COMPLETED/0:0；1700/1700 LIME头PASS，复用10conf/新统计契约PASS；
CPU数组前5项正在真实测量，其余Priority排队，汇总均依赖等待。

生成日期：2026-10-15（Asia/Shanghai）。

会话标题：L32x64 线自能与 ratio/Ward 费米子归一化测试设计。
