# 临界质量与手征处方核查（2026-09-18）

最新试算：文献中心值-.23539、±.00003和旧参考-.23545，conf10000各4配对Z4噪声
均已完成。质量1sigma导致Zchi相对响应点估计.0011%--.0066%，大tau处响应本身
尚有较大噪声误差；从有限质量-.2295改用临界质量导致Z降低.36%--.51%。
完整证据在`../quark_ringed_mass_pilot_v1/runs/critical_mass_20260918_cuonexuz/analysis/`。
32noise补充数组3644944和依赖分析3644947已提交，结果目录为同模块
`runs/critical_mass_20260918_ei1yb2qj/`。仅为固定海质量下的单组态价临界试算，
没有输出accepted ringed归一化，没有修改正式matching结果。

更新：用户提供F32P30文献线索后，已找到匹配作用量的已发表临界质量。
arXiv:2408.03548v3 Table3给出amcrit=-.23539(3)；2310.00814v2 Table3旧值-.23545(3)。
本地HMC的u0=.863437、cSW=1.141151096与文献v0=.956942、cSW=v0^-3一致，
beta、体积、海质量和一步stout rho=.125亦一致。故不再需要为了获得初始质量值从头扫描。
用户已授权按新文献值试算与±1sigma/旧参考敏感性检查；见邻接模块
`../quark_ringed_mass_pilot_v1/README.md`。本地PCAC复测尚未做；固定海质量价手征处方
与完整海手征极限仍须区分。下述“未找到”是此前有界本地搜索的历史记录。

历史状态：作用量参数已核对；临界质量数值、PCAC 扫描、手征外推均待确认。
有限质量 conf10000 pilot 不能作为正式 ringed/MSbar matching 输入。

## 已确认的依据

1. arXiv:2312.05032v3 原始源码 `WilsonlineGF.tex` 第222--234行对单 flavor 定义
   `Zchi = -2 Nc / [(4 pi)^2 t^2 <bar chi <->Dslash chi>|m=0]`；
   第252行明确采用 massless quark limit。
   本机原始资料：`/tmp/2312.05032v3.tar`；公开入口
   <https://arxiv.org/html/2312.05032v3>。
2. 系综 HMC 输入：
   `/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96/beta6.41_mu-0.2295_ms-0.2050_L32x96_restart_46500.xml`，
   第124--147行：light mass=-0.2295、cSW=1.141151096、一次 stout rho=.125、
   orthog_dir=-1（全部四维）、各向同性、时间边界 -1。与历史轻夸克脚本
   `/public/home/sush/EasyDistillation/work/perambulator.3.py` 的输入一致。
   这仍不能替代每份现有 perambulator 与生成脚本的关联证明。
3. XML 中还出现 -0.2293、-0.2289 等质量；它们出现在 HMC monomial/求解参数中，
   不能据此声称有 PCAC 质量扫描或测得临界质量。
4. 已搜索本项目、Note、quark flow 的 small_flow_time、Peram_code_2505、
   Chroma_stout、EasyDistillation/work 和共享 hmc_tune_2023_10_15 中相关文本。
   没有找到符合该作用量的 PCAC 扫描或临界质量记录。
   这是已检查范围内的结论，不是声称整个集群不存在资料。

## 应采用的处方

Wilson/Clover 裸质量存在加性重整化。目标是相同 beta、cSW、stout、海夸克系综下，
价夸克轴流 Ward identity 的质量趋零。可用零动量关联函数

\[
C_{PP}(n)=\sum_{\mathbf x}\langle P^a(\mathbf x,n)P^{a\dagger}(0)\rangle,
\qquad
C_{AP}(n)=\sum_{\mathbf x}\langle A_4^a(\mathbf x,n)P^{a\dagger}(0)\rangle,
\]

在明确定义 P、A、源汇共轭和欧式相位后构造

\[
am_{\rm PCAC}(n)=
\frac{\tfrac12(\partial_n+\partial_n^*)C_{AP}(n)
      +c_A\partial_n^*\partial_n C_{PP}(n)}{2C_{PP}(n)}.
\]

这里导数均为无量纲格点差分；若选择未改进轴流，应显式记录 cA=0 与有限格距误差。
此公式给出处方，不意味着现有核子 C2 能替代上述介子关联函数。

具体需要：至少三个相邻价质量、相同组态和源位置的 PP/AP；先检验平台及源依赖，
再用配对重采样对 `am_PCAC = A (am0-amcrit)` 做局部拟合，检查曲率与窗口稳定性。
不要从单组态、单质量或 pion mass 的文件名确定 amcrit。

随后可以在已调谐的临界价质量测量 E，或者对 E 做有控制的
`am_PCAC -> 0` 外推。临界质量的不确定度须传播到 E 和 Zchi。
若仅把价质量调零而保持现有海质量，应标记为该海质量下的价手征处方；
不能称为完整海夸克手征极限，海质量效应与 matching 精度须另行评估。

当前 collector 只实现已独立确认的 direct-critical-mass 输入，没有实现 PCAC
拟合或手征外推。32噪声/单组态试算只验证测量链和成本，既不能确认质量处方，
也不能替代与裸矩阵元相同10组态的配对统计。

## 后续最小输入

需要已有的临界质量/PCAC 扫描文件及其作用量说明；若没有，则应另行安排
PP/AP 多质量测量与拟合。本次不猜测质量点，不把有限质量输出改标为 validated，
也不改动现有正式 matching 数组。
