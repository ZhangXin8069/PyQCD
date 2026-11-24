# arXiv:2610.40225v1 对当前 quark/gluon gradient-flow matching 的更新评估

日期：2026-11-15。已阅读 Xiang-Peng Wang，*Off-lightcone Wilson-line operators in gradient flow at next-to-next-to-leading order*，2026-10-29提交的v1。主文与附录分别核对了流场/ringed定义、辅助场表示、有限转换系数、两圈结果、RG和线性质量的处理。配套局域流系数核对 Borgulat 等 arXiv:2311.16799v2，特别是式(3.9)、(3.10)、(3.12)；后者使用 alpha_s/pi，转换单位不可遗漏。

本文是文献与现有源码的评估，不是已批准的生产匹配升级。原生产代码、矩阵元、Wilson-line质量候选、拟合与图均未修改。新建的CPU脚本只计算有限转换系数和代数核对，没有读取矩阵元。

## 1. 判断与边界

**可以更新GF算符到MSbar算符的有限小流时间转换层；不能仅凭这篇文章把整条PDF链标为NNLO。**

文章式(32)–(35)将直线算符写成线质量指数与有限端点系数的乘积：quark系数为两个重–轻流系数的乘积，gluon按两个场强端点相对于线方向的Lorentz类型取乘积。该结构适合现有代码的分层设计。

文章将线性质量 delta_m 作为独立输入，未在本次NNLO计算中确定它。文章也不是从准PDF到光锥PDF的完整NNLO卷积核计算。因此不能用其有限系数替换我们刚测的Wilson loop/line质量候选，也不能替代LaMET层、gluon局域锚点或singlet quark–gluon混合。

本文研究的是薄链接、四维flow、小流时间的非局域算符转换。它不提供3D HYP5/4D HYP10方案转换，也不提供任意大flow时间、任意t/z²下的完整匹配函数。不能因采用两圈系数就取消尺度窗或把较大flow点直接并入线性外推。

## 2. 对照当前实现

### 2.1 Quark三格距分支

主要核对：

- `quark_B_three_ensemble_100_40_30_boot200_continuum_20261108/code/coefficients.py:6`：一圈alpha运行。
- 同文件`:19`：旧LL演化；`:29`：同流时间局域归一化后的单圈逆转换；`:38`：另一项短距离C0目标。
- 同目录 `analysis_cpu.py:72` 起：先同Gamma/同P/同flow的z=0局域归一化，再施加固定dF的线因子，随后乘有限端点因子。
- `config.json`：nf=3，mu=2 GeV，alpha_ref=0.3；p_log为3/3/4，c_endpoint为2.5/2.5/2.0，分别对应unpolarized/helicity/transversity。
- 当前共享dF=0.374137明确是历史L32x64候选定义的中间S方案，并非质量批准的最终值。metadata保持绝对ringed归一化未测、GF_to_MSbar_complete=false、physics_validated=false；transversity外部符号确认仍待完成。

较新的 `quark_B_three_ensemble_sumdiff_cut2_boot200_20261110` 是bare矩阵元批次，其config明确没有做matching，不能视为另一个已有NNLO实现。

### 2.2 Gluon分支

- `matching/gluon_flow_to_quasi.py:31`：一圈 c_parallel=1、c_perp=1+alpha_s*CA*L/(4pi)及PT线质量。
- 同文件`:50` 起已经将parallel/perpendicular定义为相对于空间Wilson-line方向，而不是相对于Euclidean时间；保留独立分量后重组U/H。
- `gluon_two_ensemble_workflow_20261102/code/convert_cpu.py:7`：同样使用一圈端点和独立PT线因子，状态为diagnostic。
- `gluon_L32x64_zdir_TU_self_ratio_followup_dA_20261107/code/core.py:6`：TU分支使用外部候选dA=0.8699083237，线因子与端点1/c_perp²独立。该候选来源是旧Wilson relative，不是最新line的正式系数。

现有gluon接口可以扩展两圈端点系数。对伴随Wilson-line胶子PDF，不能误用新文的fundamental重场色磁插入系数 c_F（式45）；它与 c_perp-perp 在两圈不同。c_F也不要与群论Casimir C_F混淆。

## 3. Quark应更新局域归一化方案的比值

令 h_GF 为普通流化费米子构成的bilocal，ringed后为 Z_chi*h_GF。设非零z的有限系数为C_psi，局域同Gamma流的系数为zeta_Gamma。由两条算符转换式可推得，对同流时间的比值

```text
R_GF(z,t) = h_GF(z,t)/h_GF(0,t)
R_MS(z,mu) = exp[-delta_m_F(t)*|z|] * D_Gamma(t,mu) * R_GF(z,t)
D_Gamma = zeta_Gamma / C_psi
```

这是本次根据两篇论文及现有归一化定义推导出的更新接口，不是将新文中的原始ringed系数直接复制到旧endpoint函数。共同的Z_chi在该比值中相消；这并不意味着绝对矩阵元的ringed归一化已经测得，也不恢复局域物理电荷。

可先从质量为零近似下的non-singlet u-d启动。轴流需要保持有限gamma5方案一致；tensor的代数系数可核对，但不撤销项目中transversity符号的待确认状态。connected u/d不能冒充完整singlet PDF。

记 a_s=alpha_s/(4pi)，两个前向系数分别为

```text
C_psi = 1 + a_s*N1 + a_s²*N2
zeta_Gamma = 1 + a_s*J1 + a_s²*J2
D_Gamma = 1 + a_s*(J1-N1)
             + a_s²*(J2-N2+N1²-J1*N1)
```

这也是严格NNLO所需的逆转换展开。直接用新的两圈C_psi替换旧endpoint，会漏掉局域分母的转换。只改非局域系数会破坏当前归一化方案。

本次CPU独立组装：新文式(40)的重–轻系数平方形成C_psi；2311.16799式(3.9)/(3.10)形成局域vector/tensor系数。按旧项目变量归约的一圈结果恰好恢复

```text
D_V = 1 + a_s*C_F*(3L+5/2) + O(a_s²)
D_T = 1 + a_s*C_F*(4L+2)   + O(a_s²)
L = ln(2 mu² t_f) + gamma_E
```

此检查对不同nf和L均通过；两圈共同ringed常数C_chi,2的依赖也显式相消。新的非局域系数本身不需要按Gamma另造一套；当前归一化后的Gamma依赖主要还来自局域流的有限转换。

旧coefficients.py的RGR仅为一圈边界加LL。正式更新时应把固定阶边界、beta函数、相应异常维数和重求和精度分别注册。只增加两圈边界而沿用旧LL不能称为完整NNLO重求和；短距离C0目标及后续m0处方也是另外一层，不能因端点升级就自动宣称它们已升阶。

## 4. Gluon按场强端点类型更新

设线方向v沿空间z，则F_tx、F_ty、F_xy均为perp-perp。当前TU及相同类型的unpolarized组合使用两个c_perp-perp端点；当前H原始张量涉及混合端点，使用c_parallel-perp*c_perp-perp。必须先对原始分量转换，再按已声明的Euclidean dual、符号和运动学约定重组；不能只根据“含时间指标”判断parallel。

有限的乘法因子为

```text
K_U = 1/(c_perp-perp*c_perp-perp)
K_H = 1/(c_parallel-perp*c_perp-perp)
```

新文式(43)给出c_parallel-perp在两圈不再为1；式(44)给出c_perp-perp的两圈修正。二者的一圈极限恢复旧代码。若前向bilocal系数为1+a_s*C1+a_s²*C2，则严格逆转换应为

```text
K_NNLO = 1 - a_s*C1 + a_s²*(C1²-C2).
```

不能只把1/c改成1/(新c)就当作严格固定阶NNLO；不展开的倒数还包含未计算的更高阶，可作为另一个清楚标识的敏感性方案。当前旧代码的1/c_perp²也属于含有额外高阶项的倒数使用方式，需要在新版本中区分。

gluon没有费米子Z_chi。非零z的端点转换不能用于替代z=0的局域gluon算符/动量分数锚点；接触局域极限须单独转换。自比值应保留配对局域锚点，helicity的结构零也不能拿来作分母。若绝对锚点未定义，升级后仍只能称为有限方案下的形状/转换诊断。

## 5. 量级核对：只看系数，不是生产数据偏移

下列例子取SU(3)、nf=3、自然尺度L=0。quark的D_V使用当前局域归一化定义；gluon的K使用严格逆转换。所有数字只反映所声明的耦合与尺度下的系数，不是当前矩阵元或最终PDF的实际百分比变化。

|对象|alpha_s|NLO有限因子|严格NNLO有限因子|
|---|---:|---:|---:|
|gluon U: perp-perp pair|0.25|1.000000|1.080646|
|gluon H: mixed pair|0.25|1.000000|1.074286|
|gluon parallel pair|0.25|1.000000|1.067925|
|quark local ratio V|0.3|1.079577|1.097201|

由此看，gluon有限系数的两圈变化有必要纳入误差与方案比较；quark局域比值的修正较原始ringed双线性系数更温和，但仍不能省略。自然尺度只是令L=0，不能消除两圈有限常数。低尺度较大耦合时需继续检查微扰级数稳定性。

## 6. 不能由本文自动解决的问题

1. **线性质量。** 新文不计算本次NNLO delta_m；当前dF/dA还是条件候选。最新Wilson line/loop窗口失配不会因为端点升阶而消失。NP线质量与有限MSbar残余质量方案也须匹配，不能将中间S方案直接改名为完整MSbar。
2. **避免重复减除。** 直接PT线质量和已有NP指数减除必须选择一致路线，不能叠加；本文的系数探针完全不含线因子。
3. **有限flow与格距。** 仍需检验a/r_f、r_f/|z|、sqrt(t_f)*Pz、t_f*Lambda²并保存全局窗口。大flow点可用于幂修正敏感性，不是无需控制t/z²的新增精确匹配点。
4. **局域点与有限锚点。** 小流展开是在固定非零z下构造的；z=0不应强套非局域端点系数。quark局域比值与gluon局域动量分数需要各自的局域转换和物理定义。
5. **LaMET和singlet。** 本文GF→MSbar算符转换不等于MSbar quasi→light-cone卷积；不能把gluon c_r用于quark卷积核，也不能省略quark/gluon singlet混合。
6. **RGR和其他短距离系数。** 两圈固定阶转换、RG改进、m0短距离目标、准PDF卷积的微扰阶数需要分别声明；更新一个环节不使其他环节自动达到NNLO。
7. **HYP。** 3D HYP5/4D HYP10不直接适用本文的薄链接四维flow系数。

## 7. 建议的可实施更新顺序

- 建立独立的NNLO有限系数模块，显式区分single-current、bilocal forward、strict inverse以及local-normalized ratio。将流时间单位、alpha_s阶数、nf、Gamma、表示和张量sector写入metadata。
- 先做quark u-d的局域归一化D_Gamma固定阶NNLO分支，再做gluon TU的K_U固定阶NNLO分支；保留现有NLO和固定质量候选，先只比较有限系数带来的变化。
- 对quark轴流gamma5/tensor符号、gluon dual/分量定义、局域锚点和有限残余质量方案逐项核对；未通过的分支继续保持diagnostic。
- 升级匹配精度一致的RG接口，再以共同物理rf、z和Pz比较不同格距，扫描统一的尺度窗。
- 最后重做需要受该有限因子影响的小流拟合及后续步骤；LaMET卷积及singlet混合另列版本，不把本文当作它们的替代。

如果现有逐conf C2/C3及拟合样本保留，更新的有限因子可施加到已有样本；仅这一层升级通常不要求重跑传播子或gluon OPE GPU生产。若需恢复绝对归一化、缺失算符分量或新增flow点，才涉及相应新增测量。此处是基于当前保存接口的实现判断。

## 8. 已完成工作与文件

- 保存两篇参考文章的HTML于references/，并检查关键公式的流时间、耦合及ringed约定。
- probe_nnlo_cpu.py：式(40)平方、两圈局域vector/tensor比值、gluon端点乘积及严格逆展开。只是审计原型，不是生产API。
- CPU任务4011355；36项代数核对通过，包含一圈恢复、nf/L变化、共同ringed常数相消，以及精确特殊函数公式对文章数值bilocal公式的复核。
- probe_report.json：测试范围和未验收项；coefficient_probe.csv：不同L和耦合下的系数。
- 这次没有改动生产matching或重新生成物理PDF，没有对新Wilson-line系数作质量批准，也没有提交GPU测量。

后续代码评审应首先检查三个接口：quark的同flow局域比值、gluon的伴随场强端点sector、线质量是否只减除一次。只有有限系数审计通过，并不等于整条匹配链完成真实数据验证。
