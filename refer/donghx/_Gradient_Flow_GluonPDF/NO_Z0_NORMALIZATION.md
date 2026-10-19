# L32x96 gluon unpolarized：不除 z=0 的独立分析分支

生成日期：2026-09-16。会话标题：绝对幅度、局域 matching 与 xg(x) 的探索性重建。

## 1. 交付范围与不能混淆的结论

入口是本目录的 `lambda_fourier_quasi_no_z0_v1.py`；输出在
`unpolarized_relaxed_q1em6_chi10_minerr_v3/lambda_fourier_quasi_no_z0_v1/`。
旧的 `lambda_fourier_quasi_matching_v2/` **没有修改**。

最终使用其中的 **`quasi_support_refined/` 子目录**：初次 [-3,3] quasi 积分
在支撑范围检查中达到约 0.115 sigma 的变化，因此另建数值精化输出，
使用 [-12,12]、dx=.01、xi_max=200，并检查 [-16,16]、[-24,24] 和 xi_max=400。
两版的矩阵元、局域 matching、能量、拟合模型与样本身份完全相同。

新分支不计算 M(z)/M(0)，也不把最后曲线积分调成 1 或某个外部动量分数。
它保留原始整体幅度，并实现从本项目 ratio 到标准正 x 的 xg(x) 的转换。
1000 个 bootstrap 样本与上游一一对应，不重新随机采样。

**这仍不是受控的物理 PDF 测定**：只有一个格距，继承 tau/a²>=1 的
探索性流时间外推；忽略 quark-singlet 混合；局域 spin-2 投影使用连续理论；
MSbar→hybrid 使用微扰近似，且小 z 完成与大 lambda 尾部都需要模型。
图中的统计带不包含这些系统误差。不能将“计算全部有限”写成“物理验证通过”。

## 2. 代码确定的场强、颜色与正负 z 因子

生产入口 `../src/flowed_gluon_ope.cpp` 的 Clover 使用四叶 Q：

\[
 F^{\rm code}_{\mu\nu}=\frac{Q-Q^\dagger}{8i}\bigg|_{\rm traceless}
 =a^2gF^{\rm canonical}_{\mu\nu}+O(a^4).
\]

这里不是已经除去 g 的规范场强。采用 Hermitian 生成元
Tr(Ta Tb)=delta_ab/2，代码使用普通基本表示迹，不除 Nc。
保存的 even 是 O(+z)+O(-z)，不是它的一半。因此实偶通道的 2 与颜色迹
1/2 抵消；不能再乘 2，也不能再除 2。算符是

\[
 T_U=F_{tx}F_{tx}+F_{ty}F_{ty},\qquad S_U=2F_{xy}F_{xy},\qquad (T_U-S_U)_{\rm even}.
\]

程序对三维插入点求和，不做体积平均。ratio 是经真空减除的 Re C3/Re C2，
裸矩阵元拟合本身没有额外乘 2E。
当前 N406 产品的原始中心是 ensemble 平均的真空减除 C3 除以 ensemble 平均 C2；
每个 bootstrap 内重新进行均值、真空减除和相除。这是对当前实际输入的说明，
不要把它与历史对话中的逐构型先相除方案混为一谈。本分支没有修改 ratio estimator。

独立 C++ 测试 `../tests/check_clover_normalization.cpp` 直接包含生产函数，
以 Abelian 弱磁通测试 F00=sin(flux/2) 和 2Tr(F²)/flux²→1。
flux=0.02,0.01,0.005 的后一个比值分别为
0.999967、0.999992、0.999998。这个测试核实树级 Clover/颜色/偶投影因子，
**不等价于测定所有有限格距重整化效应**。

## 3. 从 ratio 到 Fourier 输入的整体运动学归一化

采用状态归一化 <P|P'>=2E V delta(P,P')。同源汇的自旋无关前向插入给出
C3/C2=<P|O|P>/(2E)。格点算符密度带 a⁴，空间求和等于 a⁻³ 的连续积分，
所以带来 a，而不是额外的 L³；最终用无量纲 aE 表示。

Minkowski 度规 diag(+,-,-,-)，Wick convention F0i(M)=i Fti(E)。令
M(mu alpha;lambda beta)=<P|F(mu alpha) W_adj F(lambda beta)|P>，
那么本算符对应 Oadj=M(0i;i0)+M(ji;ij)。i,j只取x,y。

\[
 O_{\rm adj}=2E^2\mathcal M_{pp},\quad
 -\mathcal M_{pp}(\lambda,0)=\frac12\int_{-1}^{1} dx\,e^{-ix\lambda}xg(x)
 =\int_0^1dx\,\cos(x\lambda)xg(x)\equiv I(\lambda).
\]

上述张量分解和 PDF convention 来自
[Balitsky–Morris–Radyushkin，1910.13963，式(2.5)、(2.12)](https://arxiv.org/abs/1910.13963)。
负号由 Wick 和指标次序决定，不是为了让图变正而取绝对值。

现有非局域 GF matching 的目标是 gF W gF，而非 F W F；其电流定义可核对
[Brambilla–Wang，2312.05032v3，式(19)、(20)](https://arxiv.org/html/2312.05032v3)。
因此在统一 MSbar 目标方案下，记已经 matching 的 ratio 为 R_MS：

\[
 R_{\rm MS}(z)=-g_{\rm MS}^2(\mu)(aE)\,I(z),\qquad
 \boxed{H(z)=-\frac{2R_{\rm MS}(z)}{g_{\rm MS}^2(\mu)(aE)}}.
\]

本分支固定 alpha_s(2 GeV)=0.25，与上游一致，g_MS²=4pi alpha_s=pi。
这里不能突然换成 6/beta：那是另一种裸耦合约定，且不是上述 MSbar 目标。

使用完整实轴 Fourier convention 的偶函数形式：

\[
 Q(x)=\frac1\pi\int_0^\infty d\lambda\cos(x\lambda)H(\lambda),\quad
 H(0)=2\langle x\rangle_g.
\]

所以标准正 x 的 xg 要用 H=2I；若用 I 直接作 1/pi cosine transform，
正 x 的结果会少一半。有限动量、非局域 Wilson 系数尚需后续 LaMET 转换，
上述式子是算符归一化，不是声称 quasi 已经等于 light-cone PDF。

能量来自 `../ratio_bootstrap_all_operators_v1/results_v1/`
`c3_c2_bootstrap_allops_tau1p000_N406_Nboot1000.npz` 的 nopol 实部 C2。
严格核对该文件的 bootstrap_indices 和 configuration manifest。
前向单态 A exp[-E(t-tmin)]，完整协方差，tmin=8..12、tmax=13..15，至少四点。
每个原始数据/重采样样本单独进行质量筛选和 AIC。
T=96、t<=15 时忽略反向热项；激发态残余待进一步双态/窗口检查。
使用各样本自己的 E，保留 E 与 R 的统计关联，而不是独立抽取能量。

## 4. 局域 z=0 的 matching 必须单独做

非局域两个端点相隔有限距离的系数不能直接用于重合的两个场强。
新分支从八个 tau 的 `bare_params_selected` / `bare_params_boot_selected`
取局域裸点，不使用旧的 endpoint-matched z=0，且不重复除去 endpoint 因子。

对 connection-field 的 spin-2 局域投影，采用
[Makino–Suzuki，1403.4772v5，式(4.34)、(4.39)、(A.11)、(A.12)](https://arxiv.org/html/1403.4772v5)。
该文定义 O_bare=Z O_R，flowed O=ζ O_bare，因此有限系数是 ζZ，不是 ζ/Z：

\[
 \zeta_{11}=1+\frac{g^2 C_A}{(4\pi)^2}
 \left[\frac{11}{3}\left(\frac1\epsilon+\ln(8\pi t)\right)+\frac73\right],
 \qquad
 Z_{11}=1-\frac{g^2 C_A}{(4\pi)^2}\frac{11}{3\epsilon}.
\]

计入 MSbar 尺度约定，极点抵消后

\[
 C_{\rm loc}=1+\delta_{\rm loc},\quad
 \delta_{\rm loc}=\frac{\alpha_s C_A}{4\pi}
 \left[\frac{11}{3}\{\ln(2\mu^2t)+\gamma_E\}+\frac73\right].
\]

主结果用严格一圈逆 `(1-delta_loc)*R_flowed(0)`，另保存 `R_flowed(0)/C_loc`
作更高阶截断敏感性。之后分别进行局域 tau→0 外推。
这里的 C_loc 不是总能动张量的 c1；也不包含被省略的 quark 混合块。
完整局域 g/q 混合是矩阵关系，单独用一个胶子系数只能是当前 gg-only 测试。

**待确认的重要限制**：TU-SU 点态并不与任意场构型上的某个 rank-2 EMT 分量恒等。
在连续、非极化前向矩阵元中，它投影出 Mpp；本实现据此使用 spin-2 系数。
有限 a 的额外张量/旋转对称破缺需借助完整局域 EMT 分量或多方向 OPE 验证。
本分支不把这个假设伪装成已经做过的非微扰局域重整化。

## 5. 短距离、lambda 尾部和动量空间 matching

z>=a 的上游 tau0 数据不改，只乘上述运动学因子。z=0 替换为独立局域结果。
局域算符点不应被当成带 log(z²) 的 bilocal MSbar 的连续极限。
因此显式选择以下短距离模型（不是新测量）：令 lambda_c=Pz*a，
r=lambda/lambda_c，0<=r<1，

\[
 H^{(0)}=(1-r^2)H_{\rm loc}+r^2H_{\rm MS}(\lambda_c),
\]
\[
 H_{\rm hyb}=(1-r^2)H_{\rm loc}
 +r^2[1-\kappa c_0(a)]H_{\rm MS}(\lambda_c),\quad
 \kappa=\alpha_s C_A/(2\pi).
\]

大于等于 lambda_c 使用数据插值和 [7,8] 的平滑尾部连接，并施加
H_hyb=H_MS-kappa*c0*H_MS。系数
c0=(5/6)ln(z_eff² mu² exp(2gammaE)/4)+3/2，z_eff=min(z,zs)，zs=.25fm。
原理上短距离的 bilocal MSbar 完成可写为 H_hyb+kappa*c0(z)H^(0)，含应有的 log；
本程序直接变换有限的 hybrid proxy 和树级核输入，不把接触点当成有限 MSbar 极限。
独立比较 lambda_c=Pz*2a，量化这一插值假设的影响。

尾部同旧分支：固定 power={.5,1,2}、decay length={4,8,16}、lambda_min={4,5,6}，
lambda<=12 且 z/a<=16，只拟合一个幅度，不声称从现有数据拟合了两个形状参数。
完整协方差、AIC、qmin=1e-6、chi²/dof<=10；无合格模型的样本执行有标记的
最小候选统计误差 fallback，不删除样本。join=7、9 另作敏感性。

严格一圈逆匹配：

\[
 [xg]_{\rm test}=\mathcal F[H_{\rm hyb}]
 -\kappa K_{gg}^{\rm hybrid}\mathcal F[H^{(0)}].
\]

核参考 [2510.26425v2，式(14)、(15)](https://arxiv.org/html/2510.26425v2)，
复用经检查的 `match_unpolarized_quasi_from_extrap_v2.py`。
初次计算的 quasi 输入积分覆盖 [-3,3]；最终 `quasi_support_refined/` 使用
[-12,12]、dx=.01、xi_max=200，并以 [-16,16]、[-24,24] 和 xi_max=400 检查。
两者都不是将 quasi 截在物理 [0,1] 支撑；保留 plus subtraction。
不对已经修正的一圈 quasi 再作不加截断的乘积，避免混入部分二圈项。
这是 perturbative hybrid proxy，不是论文使用的实测 Pz=0 分母。

## 6. 数据、质量信息和运行

输入：406 conf，1000 bootstrap，seed=1115；a=.0775fm、L=32、T=96；n_z=3,4,5。
流时间集合 1.0,1.4,1.8,2.2,2.6,3.0,3.4,3.8（单位 a²）。
局域外推枚举连续至少三个点；使用完整协方差与各样本独立 AIC。
中心值始终为原始 ensemble estimator，误差为 bootstrap std(ddof=1)，不除 sqrt(1000)。
每一阶段的 fallback、秩、候选参数、权重和协方差均存储在 per_momentum/PzN.npz。

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/gluon_no_z0_mpl XDG_CACHE_HOME=/tmp/gluon_no_z0_cache \
  python3 matching/lambda_fourier_quasi_no_z0_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/gluon_no_z0_mpl XDG_CACHE_HOME=/tmp/gluon_no_z0_cache \
  python3 matching/test_lambda_fourier_quasi_no_z0_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/gluon_no_z0_mpl XDG_CACHE_HOME=/tmp/gluon_no_z0_cache \
  python3 matching/refine_no_z0_quasi_support_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/gluon_no_z0_mpl XDG_CACHE_HOME=/tmp/gluon_no_z0_cache \
  python3 matching/validate_lambda_fourier_quasi_no_z0_v1.py \
    --out matching/unpolarized_relaxed_q1em6_chi10_minerr_v3/lambda_fourier_quasi_no_z0_v1/quasi_support_refined \
    --write-report
```

主脚本拒绝覆盖非空输出目录；重做必须指定新的 `--out`。
`no_z0_xg.npz` 的曲线轴是 [momentum, original_then_1000_bootstrap, x]。
能量轴是 [original_then_bootstrap, momentum]。
x=0 附近 |x|<.02 不求值，保留 NaN 和 computed_mask；不会填零或画成测定值。
输出 PDF/PNG 包括 xg_all_Pz、xg_overlay、xg_comparison、lambda_no_z0、
local_matching_flow、energy_audit、short_distance_sensitivity。
manifest.json 记录输入/代码哈希、结果摘要、数值积分和模型变化。

实际局域/能量结果（括号为统计误差；单态、局域投影和一圈方案的系统误差不在内）：

| n_z | aE | 局域 <x>g：严格一圈逆 | 局域 <x>g：未展开 1/C | 尾部 fallback /1000 |
| --- | --- | --- | --- | --- |
| 3 | 0.74630(382) | 0.2690(429) | 0.3468 | 10 |
| 4 | 0.91096(995) | 0.3249(630) | 0.3655 | 31 |
| 5 | 1.10166(2216) | 0.2187(681) | 0.2643 | 9 |

三种动量的 C2 能量和局域流外推所有样本都有质量合格候选，不需要 fallback。
尾部 fallback 不代表质量通过，已逐样本明确保存。局域一圈修正 delta_loc
在所选流时间范围为 0.312..0.604；严格逆和 1/C 差异不能忽略。

## 7. 使用结果前的下一步

1. 查看局域动量分数的 Pz 一致性及严格逆/1/C 差异；不能借归一化藏掉不一致。
2. 实测完整局域 spin-2 EMT 组合、多方向比较，检验有限 a 的投影假设。
3. 在更可靠的 a²<<t<<z² 区域检查非局域 GF 转换；当前 sqrt(8t)/a=2.83..5.51，
   因此 z=a、2a 显然不满足流半径远小于端点距离。局域重匹配不能修好这些非局域点。
4. 能量和矩阵元都检查激发态，扩充大 z 统计以约束 lambda 尾部。
5. 补 singlet mixing、尺度和高阶误差；多格距及更大动量后才可讨论受控物理 PDF。

不得擅自修改：±z 和与平均的区别、基本表示迹、Wick 负号、C3/C2 的 1/(2E)、
gF 与 F 的区别、完整轴 Fourier 因子 2、共享 bootstrap 身份、局域/非局域匹配的区别。
这些如要更改，必须给出新的代码/文献证据，并另建版本。
