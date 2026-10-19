# Gradient-flow quark/gluon PDF 项目交接文档

最新更新：2026-10-15，见第16节「Blending 权重、flow/basis/vertex 与时间窗测试」。
第1–14节保留历史过程；λ/Fourier/matching结果路径以第15节对应分支为准；本轮权重和时间窗状态以第16节为准。
后续用户已选择方案B，但旧方案A的裸矩阵元和matching/PDF没有因此自动更新；全量B重算尚未提交。
旧文中的“λ/Fourier 尚未实现”“n_z=3 尚无输入”不再适用于第15节列出的新分支。
目前已完成文献临界价质量处方下的10conf动能测量与原50样本配对归一化；
旧段落中的“尚缺Zchi／临界质量未找到／10conf未完成”仅代表当时状态。
统计配对与数值验证已完成，质量处方、改进和连续极限的正式物理验收仍未完成。

生成日期：2026-09-08  
本次会话标题：Gradient-flow quark/gluon PDF 全链路状态、约定与继续工作清单

---

## 10. 2026-09-11 增补：当前 gluon quasi matching 结果与操作顺序

本节覆盖并更新前文关于 gluon matching 的状态；如果旧段落与本节冲突，
以本节和对应产物的 `manifest.json` 为准。

### 10.1 当前理论链

当前计算必须区分三个层次：

1. 有限流时间的 lattice OPE/C3/C2 矩阵元；
2. gradient-flow scheme 到 MSbar quasi（或同一 scheme 的 ratio/hybrid）转换；
3. quasi-to-light-cone 的 LaMET/pseudo-ITD matching。

推荐的两步链为

```text
h_GF(z,Pz,tau)
  -> GF scheme conversion
  -> h_R(z,Pz,tau)
  -> small-flow-time extrapolation tau -> 0
  -> lambda=z Pz interpolation/tail treatment
  -> Fourier transform
  -> x*g_quasi(x,Pz)
  -> gluon LaMET kernel (and singlet mixing)
  -> x*g(x,mu)
```

这里的 lambda 处理不是物理的 `lambda -> infinity` 极限。它可以表示
有限数据的 lambda 插值、最大 |lambda| 以外的尾部模型，或 pseudo-ITD 中
固定 lambda 的 zeta^2=z^2 外推；三者必须在元数据中明确区分。

对 arXiv:2510.26425v2 的 x-space gluon kernel，当前推荐先完成坐标空间的
flow conversion、tau 外推和有限 lambda 重构，再 Fourier 变换，最后应用
quasi-to-light-cone kernel。坐标空间 kernel 只有在同一 scheme、同一 Fourier
约定下才可以放到 Fourier 之前。有限 zmax、插值、尾部模型和逐 replica
归一化存在时，两条数值路线不能默认等价。

当前 even unpolarized 的 Fourier 约定为

```text
x g_quasi_tilde(x,Pz) = Pz/pi * integral_0^infinity dz
                           cos(x Pz z) h_R(z,Pz)
```

helicity 的 odd channel 使用对应的 sine transform。缺失的短距离 z 点必须
保持 mask/NaN，不能用零填补后 Fourier 或拟合。

### 10.2 当前 N406 flow-time 外推输入

输入产物：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching/bare_all_operators_matching_v1/flow_extrap_tau_ge1/matched_small_flow_extrap_tau_ge1.npz
```

其 manifest 明确记录：

- Nconf = 406；
- Nboot = 1000；
- a = 0.0775 fm；
- tau/a^2 = 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8；
- 小流时间模型为 `M(tau)=M0+c*tau`，采用 replica-level AIC 权重；
- 当前 operator labels 同时包含 unpolarized 的 `TU_even`、`TU+SU_even`、
  `TU-SU_even` 和 helicity 的 `TH_odd`、`TH+SH_odd`、`TH-SH_odd`；
- Pz integer = 3,4,5，z/a = 0,...,24。

外推结果和拟合图：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching/bare_all_operators_matching_v1/flow_extrap_tau_ge1/
```

### 10.3 quasi matching 脚本和实际产物

实现入口：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching/match_unpolarized_quasi_from_extrap_v2.py
```

该脚本：

- 使用 arXiv:2510.26425v2 的 gluon `C_ratio`/`C_hybrid` kernel；
- 对每个 bootstrap replica 独立做 Fourier 和一圈 inverse matching；
- 暂时关闭 quark--gluon singlet mixing；
- 使用有限 z/a=0,...,24 的 cosine transform；
- 不对缺失 z 点做零填充；
- 保存 `quasi_boot`、`matched_boot`、scheme correction、Lamet correction、
  bootstrap indices、数值敏感性检查和完整 manifest。

hybrid 结果：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching/bare_all_operators_matching_v1/quasi_matching_tau_ge1_v2/
```

ratio 结果：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching/bare_all_operators_matching_v1/quasi_matching_tau_ge1_v2_ratio/
```

主要图：

```text
.../quasi_matching_tau_ge1_v2/unpolarized_before_after_all_Pz.pdf
.../quasi_matching_tau_ge1_v2/unpolarized_zmax_sensitivity.pdf
.../quasi_matching_tau_ge1_v2_ratio/unpolarized_before_after_all_Pz.pdf
.../quasi_matching_tau_ge1_v2_ratio/unpolarized_zmax_sensitivity.pdf
```

运行命令：

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
python3 -m py_compile matching/match_unpolarized_quasi_from_extrap_v2.py
MPLCONFIGDIR=/tmp/mpl-gf python3 matching/match_unpolarized_quasi_from_extrap_v2.py
MPLCONFIGDIR=/tmp/mpl-gf python3 matching/match_unpolarized_quasi_from_extrap_v2.py \
  --scheme ratio \
  --out matching/bare_all_operators_matching_v1/quasi_matching_tau_ge1_v2_ratio
```

### 10.4 已确认的结果状态

- Pz=3 的 z/a = 0,1,3,5,6,7,8,9,10,11,12,13 外推点缺失，因此被记录为
  `not_computed_missing_input`，没有输出伪造的 quasi 结果；
- Pz=4 和 Pz=5 均完成 1000 个 bootstrap replica；
- Pz=5 的局域 z=0 归一化分母有 1 个 bootstrap replica 符号翻转，manifest
  已标记 `heavy_tail_risk_sign_crossing`；
- quadrature order、xi_max、输入网格和 quasi support 的扫描均已保存；
- hybrid 与 ratio 的结果都只是 exploratory NLO matched quasi diagnostic；
- 当前输出不是最终 gluon PDF。

### 10.5 不能随意修改的理论约定

1. Wilson line 沿空间 z 方向时，unpolarized 的 `Mtiti` 和 `Mijij` 都属于
   `perp-perp` endpoint sector，因此当前 GF conversion 使用共同的
   `C_perp-perp^{-1}`，再组合 `TU-SU`；不能把 temporal/spatial 标签直接
   当作 parallel/perpendicular 标签。
2. helicity 的 dual tensor 交换两个 endpoint sector；`TH` 和 `SH` 在候选
   endpoint-factorized one-loop conversion 中使用共同的
   `C_Delta_g = c_parallel-perp*c_perp-perp`，physical operator 仍是
   `(TH-SH)_odd`。`TH+SH` 和 `TH` 只能作为 diagnostics。
3. line-mass 约定为
   `delta_m = - alpha_s*C_A/(4*pi) * sqrt(2*pi/t_GeV_minus2)`，其中 t 必须
   先从 tau/a^2 转换为 GeV^-2；不能使用 `2*pi/sqrt(t)` 的旧系数。
4. 若 nonperturbative ratio/hybrid subtraction 已经包含 line mass，不能再
   乘一次 perturbative `exp(-delta_m*|z|)`。
5. 当前 `kappa_F=1` 和 helicity endpoint-factorized coefficient 是诊断/候选
   约定，不等于已经完成独立的完整 helicity partonic matching 验证。

### 10.6 当前仍未解决的问题

- 需要确定 GF-to-quasi 与 ratio/hybrid nonperturbative normalization 的
  不重叠实现；
- 需要增加 unpolarized/helicity 的 quark--gluon singlet mixing；
- 需要对 finite-z/finite-lambda tail 做模型和协方差传播；
- Pz=3 需要补齐短距离外推输入后才能 Fourier；
- 当前只有一个格距，不能做受控 continuum extrapolation；
- 需要 target-mass、higher-twist、matching scale 和 flow-window systematic；
- helicity 的完整一圈 GF-to-quasi partonic calculation 仍为“待确认”；
- 当前 quasi 输出不应被称为最终 gluon PDF。

## 11. 本文档的生成信息

更新日期：2026-09-11  
本次会话标题：N406 gluon flow extrapolation、lambda/Fourier 顺序与 quasi matching 结果交接

本文是给未来 Codex 会话使用的项目交接记录。它把理论、代码、数据
格式、已验证结果和未完成事项放在同一个地方。文中的“物理 PDF”只指
已经完成流方案转换、重整化、混合、LaMET 或 pseudo-ITD matching、极限
控制和有限坐标重构之后的结果；目前本项目的绝大多数产品仍是
finite-flow bare correlator 或 bare matrix element。

本文不包含任何 API key、bearer token、密码、auth 文件内容或其他凭据。

## 1. 项目目标和当前研究问题

### 1.1 总目标

项目的目标是从 thin-link、未做 HYP/APE/stout 预处理的格点规范场出发，
使用四维 Wilson gradient flow 构造 quark 和 gluon 的空间类非局部算符，
计算核子两点/三点函数，得到有限流时间的矩阵元，再完成：

    lattice finite-flow correlator
      -> C3/C2 ratio and state isolation
      -> bare matrix element
      -> ringed/SFTE flow-scheme conversion
      -> renormalized quasi or pseudo matrix element
      -> channel-specific LaMET or pseudo-ITD matching
      -> finite-z reconstruction and PDF

quark 和 gluon 必须分别使用适合各自算符的 matching；不能把 quark 的
ringed-fermion 系数套给 gluon，也不能把 HYP 或 hybrid-renormalization
的常数直接套给 thin-link 4D flow。

### 1.2 当前 ensemble

- ensemble：beta6.41_mu-0.2295_ms-0.2050_L32x96；
- lattice：32^3 x 96；
- 目前主要格距：a 约为 0.0775 fm；
- 初始规范场：thin link / Nosmear；
- quark 原始数据根目录：
  /public/group/lqcd/donghx/Gradient_Flow/Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/thin_link；
- gluon 原始配置目录：
  /public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96。

### 1.3 当前最重要的研究问题

1. quark：10conf的独立ringed-field归一化已按文献临界价质量完成（第14节）；
   正式物理验收仍未完成。仅after-Zchi结果不能直接叫MSbar quasi-PDF。
2. quark：四维 flow 的多 flow-time 结果需要通过完整的状态隔离、固定物理
   flow radius 的连续极限、SFTE 和 channel-specific LaMET matching。
3. gluon：OPE 和 C3/C2 已经有大量数据，但 physical unpolarized/helicity
   gluon PDF 仍需要 flow-to-quasi conversion、gluon-quark singlet mixing、
   LaMET/pseudo-ITD kernel、有限 z 重构和极限控制。
4. gluon：最新统计产品是 Nconf=406；较早的 Nconf=231 产品仍有图和拟合，
   但只能作为历史/诊断产品。未来分析应默认使用 N406。
5. quark 的 t/a^2=0.8 和 1.1 任务曾提交，但当前日志在 projection 阶段
   找不到 t/a^2=0.8 的 corr_projected 输入；这两个 flow time 尚未完成。

## 2. 已确认结论、依据和不能改变的边界

### 2.1 [已确认] 四维 flow 和三维 flow 不是同一个 scheme

四维 gauge flow 使用所有四个方向的场：

    d B_mu(t,x) / d t = D_nu G_nu,mu(t,x)

quark flow 使用同一个 flowed gauge field 的四维 covariant heat kernel：

    d chi(t,x) / d t = Delta[B(t)] chi(t,x).

三维 flow 只阻尼空间动量，不能自动继承四维 flow 的 finiteness theorem、
ringed-field normalization 或 published 4D matching coefficient。任何 3D
结果必须有独立的 scheme 和 matching 推导。这个结论来自
gradient-flow-quark-pdf skill 的 theory-and-matching contract 和
Note/Gradient_Flow/main.tex 的 3D/4D 专节。

### 2.2 [已确认] thin-link flow 和 HYP-pre-smeared flow 不可混用

当前 production 从 thin link 开始，没有 HYP/APE/stout 预 smear。HYP 旧笔记
DeltaG_Note.pdf 只用于理解历史算符和图形，不能把其中的 HYP/hybrid
renormalization 常数直接用于当前 thin-link flow。

### 2.3 [已确认] gluon unpolarized 算符

代码坐标顺序是 (x,y,z,t)=(0,1,2,3)，Wilson line 沿 z_dir=2。
Euclidean primitive 定义为

    U_{mu nu}(z) =
      sum_x Tr[F_{mu nu}(x+z zhat) W F_{mu nu}(x) W].

保存的两个独立项是

    T_U = M_titi = M_30;30 + M_31;31
    S_U = M_ijij = 2 M_01;01

Euclidean/Minkowski Wick rotation 和场强反对称性给出 production combination

    O_U = T_U - S_U

物理方向投影是

    O_U,even(z) = O_U(+z) + O_U(-z).

数组中的 even_sum 没有除以 2。这个相对负号不能按图形好看程度修改；
它由 Minkowski 0i/ij 结构和 Euclidean Wick phase 固定。依据：
arXiv:2510.26425v2、Note/Gradient_Flow/main.tex、gluon OPE README 和
flowed_gluon_ope.cpp。

### 2.4 [已确认] gluon helicity 算符

Euclidean dual tensor 是

    Fdual_mu,nu = 1/2 epsilon_mu,nu,rho,sigma F_rho,sigma,
    epsilon_xyzt = +1.

当前 primitive 为

    H_ti = F_ti W Fdual_ti
    H_xy = F_xy W Fdual_xy
    T_H = H_tx + H_ty
    S_H = 2 H_xy

Wick rotation 后的 physical target 是

    O_H = T_H - S_H
    O_H,odd(z) = O_H(+z) - O_H(-z).

生产 OPE 文件中的 helicity combined 历史上保存了 T_H+S_H；原始
Mtiti/Mijij 仍然保留，因此不需要重新做 gauge flow/OPE 就可以构造
T_H-S_H。当前分析同时保留：

    (T_H-S_H)_odd       production target
    (T_H+S_H)_odd       legacy/sign diagnostic
    (T_H)_odd           temporal-only diagnostic

matching 不能把 plus-sign diagnostic 变成 physical helicity。依据：
arXiv:2207.08733v2、Note/Gradient_Flow/main.tex、analysis/helicity_operator_comparison.py
和 tests/test_helicity_operator_comparison.py。

### 2.5 [已确认] helicity 的复数和两点函数约定

gluon disconnected estimator 必须为

    C3_U = <O_U C2_nopol> - <O_U><C2_nopol>
    C3_H = <O_H C2_pol35> - <O_H><C2_pol35>

helicity numerator 使用完整复数 pol35。两种 channel 的 denominator 都是
nopol。只有最终 Euclidean phase projection 才取

    Im(R_H) = Re[-i R_H].

当前 bootstrap estimator 直接实现为

    R_U = Re(C3_U) / Re(C2_nopol)
    R_H = Im(C3_H) / Re(C2_nopol).

不能提前把 pol35 或 C3 取虚部，也不能用 pol35 代替 denominator。

### 2.6 [已确认] quark operator/flow 约定

- proton diquark operator：C gamma5 gamma4；
- C2 source/sink：未 flow；
- C3 中的 quasi-PDF current：flow；
- quark flavor：connected u、d，并可构造 u-d；connected u/d 不是完整
  singlet PDF；
- flow：四维同步 gauge/fermion flow；
- temporal boundary：flowed fermion 使用 anti-periodic；
- distillation：NEV_CURRENT=400，NEV_SRC=NEV_VVV=200；
- displacement：delta_z=0..20；
- pilot momenta：Pz=+3,+4,+5；
- pilot source-sink separations：tsep=6..10；
- production step：FLOW_EPSILON=0.01；
- basis/flowed vertex 常用 complex64，contraction/final correlator 用
  complex128；
- 原始endpoint仍为ordinary flowed field；第14节已用10conf独立真空测量做配对ringed后处理，正式物理验收待完成。

### 2.7 [不能随意修改] matching 层次

必须区分：

1. finite-flow bare C3/C2；
2. flowed scheme 到 MSbar quasi/pseudo 的 SFTE 或 GF conversion；
3. quasi/pseudo 到 light-cone PDF 的 LaMET/pseudo-ITD kernel。

完成第 1 层不能称为 physical PDF；完成第 2 层也仍不能称为 physical
PDF。singlet unpolarized/helicity 还需要 quark-gluon mixing。transversity
在本 quark workflow 中保持 pending_external_physics_confirmation，直到
相对算符符号经过外部定义确认。

## 3. 符号、坐标、归一化和尺度窗口

### 3.1 坐标和 flow time

- Euclidean coordinate order：x=0, y=1, z=2, t=3；
- gluon Wilson line：z_dir=2；
- lattice flow time：widehat{t}=t/a^2；
- flow radius：

      r_flow = sqrt(8 t) = a sqrt(8 widehat{t});

- 当前 gluon flow times：

      0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
      1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8.

- quark 早期 multi-flow pilot：

      0.10:2, 0.20:3, 0.30:3, 0.40:4, 0.50:4, 0.60:5

  冒号后是 temporal-window half width。新提交的 quark 测试使用
  0.80:6 和 1.10:8。

### 3.2 gluon Clover/OPE schema

flowed Clover：

    F_mu,nu = -i (Q_mu,nu - Q_mu,nu^dagger) / 8

production 使用 color-traceless projection。一个 v5 OPE 数组轴为

    (field_projection, channel, z_orientation, component, z, t)
    = (2, 2, 4, 6, 25, 96)

labels：

    field_projection: legacy_untraced, traceless
    channel: unpolarized, helicity
    z_orientation: plus_z_raw, minus_z_raw, even_sum, odd_difference
    component: combined, Mtiti, Mijij, ti, tj, ij_single

raw plus/minus 和全部 components 必须保留，因为它们可以做 exact linear
identity、Hermiticity、parity 和 helicity sign checks。

### 3.3 quark ringed-field SFTE

流 quark bilocal 需要 ringed fields：

    chi_ringed = Z_chi(t)^(1/2) chi(t)

两个 flowed fermion endpoint 合并成 bilocal 的一个总体 Z_chi。非微扰
定义模板在

    /public/home/donghx/Gradient_Flow/Blending_L32x96/small_flow_time/ringed_z_chi.template.json

模板不是实测数据，不能直接拿来生产。Z_chi 必须在每个 ensemble/flow
time 通过明确的 derivative、mass/chiral prescription 独立测量。

四维 4D ringed bilocal 的 corrected one-loop coefficient（Brambilla--Wang
2024 convention）为

    C_psi(t,mu) =
      1 - alpha_s C_F/(4 pi)
          [3 log(2 mu^2 t exp(gamma_E)) + 2 + log(432)]

    delta_m(t) =
      - alpha_s C_F/(4 pi) sqrt(2 pi) / sqrt(t)

其逆转换形式为

    h_quasi_MSbar =
      Z_chi(t) C_psi(t,mu)^(-1)
      exp[-delta_m(t)|z|] h_GF
      + flow-power corrections.

### 3.4 gluon GF-to-quasi matching

四维 gluon auxiliary-field convention使用

    c_parallel_perp = 1 + O(alpha_s^2)
    c_perp_perp =
      1 + alpha_s C_A/(4 pi) log(2 mu^2 t exp(gamma_E))

    delta_m_A =
      - alpha_s C_A/(4 pi) sqrt(2 pi/t)

相对于 z Wilson line：

    Z_U = kappa_F^2 exp[-delta_m_A |z|] / c_perp_perp^2
    Z_H = kappa_F^2 exp[-delta_m_A |z|]
          / (c_parallel_perp c_perp_perp)

当前代码/图诊断取 kappa_F=1；这不是已经完成的 tree-level normalization
证明，标记为待确认。系数中的 t 必须用 GeV^{-2}，不能把无量纲 t/a^2
直接放进 logarithm。matching 因子对 +z/-z 使用相同的 |z|。

当前推荐在逐组态 OPE、vacuum subtraction、C3/C2 和 bootstrap 之前施加
Z_U/Z_H；因为该因子是确定性 scalar，乘在 fit-level shared bootstrap
replica 上可以作为诊断，但不是最终 production substitute。

### 3.5 window 和极限

必须全局选择而不是逐点挑误差最小的数据：

    a^2/t << 1
    sqrt(t) Pz << 1
    t Lambda_QCD^2 << 1
    r_flow << |z|       (非零 z)
    Pz^2 >> M_N^2

quark/gluon 共同需要：

- fixed physical flow time/radius，而不是跨格距直接固定 t/a^2；
- continuum at fixed physical t,z,Pz；
- flow-time window、finite-z、finite-volume、Pz、fit-window 和
  perturbative-order systematic；
- t=0.5 时 r_flow/a=2；z/a<=2 不满足干净的 r_flow << |z|；
- quark skill 给出的 t=0.5 估计 sqrt(t)Pz 对 Pz=3,4,5 约为
  0.42, 0.56, 0.69，最高动量 marginal。

## 4. 代码入口、数据路径和运行命令

### 4.1 gluon OPE production

代码根：

    /public/group/lqcd/donghx/Gradient_Flow_GluonPDF

重要入口：

    src/flowed_gluon_ope.cpp
    scripts/build_v5.sh
    scripts/run_one_v5.py
    scripts/validate_output_v5.py
    slurm/v5/flowed_ope_cpu_v5.slurm
    slurm/v5/flowed_ope_chain_cpu_v5.slurm
    slurm/v5/submit_all_tau_v5.sh

构建和提交：

    cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
    ./scripts/build_v5.sh
    ./slurm/v5/submit_all_tau_v5.sh

v5 提交约定：

    partitions = a192c1t,cpu6248R,cpueicc,i72c512g
    cpus_per_task = 48
    epsilon = 0.01
    fine epsilon gate = 0.005 on conf1000
    z_count = 25
    z_dir = 2

提交记录：

    runs/submission_all_tau_v5_20260903_002048.txt

它记录了 887 个 manifest entries、14 个 tau、依赖 gate 和 chain heads。

### 4.2 gluon latest ratio/bootstrap

代码：

    ratio_bootstrap_latest_v1/construct_bootstrap.py
    ratio_bootstrap_latest_v1/validate_collection.py
    ratio_bootstrap_latest_v1/audit_statistics.py
    ratio_bootstrap_latest_v1/slurm/submit_all.sh

运行：

    cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
    python -m py_compile ratio_bootstrap_latest_v1/*.py
    python -m pytest ratio_bootstrap_latest_v1/tests -q
    bash ratio_bootstrap_latest_v1/slurm/submit_all.sh
    python ratio_bootstrap_latest_v1/validate_collection.py

latest frozen manifest：

    ratio_bootstrap_latest_v1/manifests/common_all14_plusz_P345.tsv
    SHA256 = 1a2161d754493f03a4c5e26d5fe20a0a83446f762f3a8bc159cc282f866b53e4

authoritative latest ratios：

    ratio_bootstrap_latest_v1/results_v1/

### 4.3 gluon latest bare fits

代码：

    fit_latest_v1/fit_bare_latest.py

运行一个 flow time 或全部：

    cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
    python fit_latest_v1/fit_bare_latest.py --tau 0.5
    python fit_latest_v1/fit_bare_latest.py

结果：

    fit_latest_v1/results_v1/
    fit_latest_v1/fit_collection_validation.json

拟合约定：

    Tmin = floor(0.6 fm / 0.0775 fm) + 1 = 8
    Tmin physical label approximately 0.62 fm
    Tmax = 9,...,15
    symmetric cuts c = 1,2,3
    forward Sumratio-difference plateau
    Schaefer-Strimmer covariance
    central Q >= 0.01
    central chi2/dof <= 3
    bootstrap success >= 0.90
    replica-wise AIC fit-range weights
    equal-weight mixing of usable cuts
    Nboot=1500, std(ddof=1)

### 4.4 gluon matching and plots

代码：

    matching/gluon_flow_to_quasi.py
    matching/calc_ratio_matched.py
    matching/plot_matched_bare_matrix.py
    matching/plot_pz3_results.py
    matching/test_gluon_flow_to_quasi.py

测试：

    cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching
    python -m pytest test_gluon_flow_to_quasi.py -q

matching 代码的 per-configuration 使用方式：

    python matching/gluon_flow_to_quasi.py \
      --input output_v5/conf8300/tau0p500 \
      --output matched_quasi_v1/conf8300/tau0p500 \
      --tau 0.5 --mu 2.0 --alpha-s 0.25

注意：这是 GF-to-quasi/operator conversion，不是 light-cone PDF matching。

### 4.5 quark 4D flow

active successor：

    /public/home/donghx/Gradient_Flow/Blending_L32x96

核心代码：

    flow_common_l32x96.py
    generate_flowed_vertex_l32x96.py
    calc_flowed_c2c3_factorized_l32x96.py
    calc_flowed_c2c3_l32x96.py
    project_flowed_channels_l32x96.py
    validate_multiflow_conf_l32x96.py
    verify_multiflow_restart_l32x96.py
    cleanup_flow_cache_l32x96.py

driver：

    run_multiflow_channels_l32x96_4gpu_a800.slurm

历史/当前提交脚本：

    /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/submit_quark_flow_02_08_all10.sh
    /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/submit_quark_flow_0811_all10.sh

这些脚本使用 pilot_z、Pz=+3,+4,+5、tsep=6..10。不要在已经排队的
job 直接改 driver；应复制成版本化 successor 后测试再提交。

### 4.6 旧/一般 quark bare-matrix workflow

connected quark bare-matrix 的一般工作流位于：

    /public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/C3_Ratio_Bootstrap
    /public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Bare_Matrix_Fit

主要命令：

    python3 run.py show-config --ensemble L32x64
    python3 run.py validate-conventions --ensemble L32x64
    python3 run.py audit-inputs --ensemble L32x64
    python3 run.py bootstrap --ensemble L32x64 \
      --channels unpolarized,helicity \
      --flavors u,d,u_minus_d --nboot 1500 --seed 1115 --c2-mode complex
    python3 -m unittest discover -s tests -v

当前 connected-quark 一般 workflow 的权威产品契约是
c3_ratio_v3、ratio_v4、bare_matrix_fit_v3；旧 ratio v2/bare-fit v1
不能恢复为 physics product。它和 local flowed-gluon N406 产品不要混淆。

## 5. 已完成的计算、检查和验证

### 5.1 gluon OPE v5

[已确认] output_v5 已覆盖：

    887 个组态目录
    14 个 flow time
    887 x 14 = 12418 个 epsilon=0.01 OPE npy/json pairs
    另有 conf1000 的 14 个 epsilon=0.005 fine-gate pairs

数组 schema、atomic output、finite check、orientation/component identity
由 validate_output_v5.py 检查。epsilon gate 的全部 14 个 flow time 均
status=passed；代表性最大值仍远低于阈值：

    max_relative_l2 threshold = 0.005
    max_plaquette_diff threshold = 5e-5

提交日志中的 kernel/source/run/validator SHA256 已记录，未来若修改
源码必须生成新的 versioned output，不要覆盖 v5。

### 5.2 gluon latest N406 ratio/bootstrap

[已确认] ratio_bootstrap_latest_v1：

    Nsource = 887
    frozen common Nconf = 406
    Nboot = 1500
    seed = 1115
    flow times = 14
    job array 3565822: 14 tasks, exit code 0:0
    shared configuration order = true
    shared bootstrap indices = true
    collection status = complete_and_shared_indices_validated

当前 frozen manifest 是为了避免 flow-time 之间配置列表漂移。每个 tau
的完整 bootstrap npz 和 SHA256 记录在 collection_validation.json。

### 5.3 gluon latest N406 bare fits

[已确认] fit_latest_v1：

    fit_total = 225 points per flow time
    filled_count = 219 per flow time
    accepted fit counts:
      tau 0.1--0.5: 219
      tau 0.6: 216
      tau 1.0: 210
      tau 1.4: 208
      tau 1.8: 203
      tau 2.2: 197
      tau 2.6: 195
      tau 3.0: 193
      tau 3.4: 191
      tau 3.8: 190

这些是 finite-flow bare matrix elements，不是 renormalized/matched PDFs。

### 5.4 gluon N231 historical products and figures

此前使用 Nconf=231 的产品生成了：

    /public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/results_v2
    /public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/results_v3
    /public/group/lqcd/donghx/Diagram_GluonPDF/Gradient_flow_gluon/bare_fit_sumdiff_aic_v1/products
    /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/matching/pz3_comparison_v1

Pz=3 的裸矩阵元和 matching 前后图已画出；每个 tau 的 z/a=0..15
单独 PDF 位于 matching/pz3_comparison_v1/bare_by_tau_z0to15。它们
对于检查算符和图形很有用，但不应覆盖或替代当前 N406 分析。

### 5.5 quark 计算和检查

[已确认] 流动/投影代码已通过多项局部检查：

- zero-flow identity、basis/vertex axis、gauge covariance、flowed-link
  checks 和 factorized contraction smoke tests 已实现；
- test job 3569237 的五个 gauge-index 检查通过：
  t/a^2 = 0.1, 0.2, 0.3, 0.4, 0.6；
- direct-versus-restart equivalence 的 preflight 阶段通过；
- quark的10conf配对Zchi测量已完成，正式validated接口和物理验收仍待完成（第14节）。

[待确认/未完成] 新的 gf0811 job 3571893..3571902：

- preflight 和 direct-vs-restart 阶段通过；
- driver 计划 pending flow specs=0.8:6,1.1:8；
- projection 失败，错误为找不到
  thin_tf0p80000000_eps0p01000000_tw6_Nv400_dz21/Nev_200/<conf>/
  corr_projected_tsep6_dz0-20_Nev200.npz；
- 因此 0.8/1.1 不能标记为 complete；
- 目前也不能仅凭这些 job 的 stdout 宣称 quark C3/C2 已完成。

当前环境中 squeue 查询返回 Slurm stream socket Operation not permitted；
queue 状态应在能访问 Slurm controller 的登录环境重新查询。

### 5.6 theory notes and documents

主要理论文档：

    /public/home/donghx/Note/Gradient_Flow/main.tex
    /public/home/donghx/Note/Gradient_Flow/main.pdf
    /public/home/donghx/Note/Gradient_Flow/DeltaG_Note.pdf
    /public/home/donghx/.codex/Markdown/gluon_gradient_flow_matching.md

main.tex 是当前整理后的 quark+gluon note；DeltaG_Note.pdf 是旧 HYP5
helicity/gluon note；gluon_gradient_flow_matching.md 专门记录 gluon
GF-to-quasi 推导、端点系数、T/S 组合和实现状态。

## 6. 尚未解决的问题和下一步最具体的操作

### 6.1 最高优先级：修复 quark 0.8/1.1

不要先重跑 fit。先在可访问 GPU 的环境做：

1. 检查每个 0.8/1.1 conf 是否存在 flowed basis、flowed gauge、
   flowed vertex 和 Nev_200 projected input；
2. 检查 driver 是否在 projection 之前正确调用
   generate_flowed_vertex_l32x96.py 和 calc_flowed_c2c3_factorized_l32x96.py；
3. 对 conf=10000 做一个 isolated successor smoke run，确认
   t/a^2=0.8 的 source path 被生成；
4. 运行 validate_multiflow_conf_l32x96.py --raw；
5. 只有 raw C3/C2、metadata、completion guard 全部通过后，才恢复
   10100..10900 和 1.1；
6. 重新记录 job IDs、output/error 路径和 completion receipts。

不要把“preflight passed”写成“flow output complete”。

### 6.2 quark ringed normalization

以下为最初的工作清单；2026-09-20完成的六flow、10conf配对产品和剩余问题见第14节。
原正式JSON消费者的验收要求仍保留，不能以新诊断NPZ冒充validated JSON。
原计划为：

1. 明确定义 massless/chiral 和 derivative discretization；
2. 使用与 4D flowed fermion 相同的 thin-link gauge flow；
3. 对每个 tau 生成带统计误差的 ringed_z_chi JSON；
4. 运行 small_flow_time/apply_sfte.py 的 tests；
5. 形成 Z_chi uncertainty 和 SFTE coefficient uncertainty；
6. 先得到 MSbar bilocal diagnostic，再继续 LaMET。

禁止直接复制 template JSON 或把 C3/C2 的 nucleon overlap cancellation
当成 Z_chi cancellation。

### 6.3 gluon N406 的匹配和 flow-time 分析

建议顺序：

1. 使用 ratio_bootstrap_latest_v1/results_v1 和 fit_latest_v1/results_v1；
2. 在逐组态 OPE 上应用 Z_U/Z_H，或者明确标记 fit-level conversion
   为 diagnostic；
3. 重新做 N406 的 TH-SH、TH+SH、TH 三种 helicity 对比；
4. 对每个 tau 重新做 bare-versus-matched Pz=3/4/5 图；
5. 只在通过 global SFTE window mask 的 tau,z,Pz 上做小 flow extrapolation；
6. 记录 flow-time fit window、AIC 权重、相关性和所有 failed points；
7. 加入 gluon-singlet/quark mixing 后才讨论 gluon PDF。

旧 N231 的 tau>1 AIC extrapolation 和 z/a=0..15 图不能直接当作 N406
结论，应以 N406 重新运行。

### 6.4 continuum、LaMET 和 PDF reconstruction

目前只有一个 lattice spacing 的 gluon 主结果，因此不能做可信的
continuum extrapolation。quark 也必须在固定物理 flow radius、z、Pz
上合并多个格距。之后：

- 选择与 operator scheme 一致的 unpolarized/helicity quasi 或 pseudo-ITD
  kernel；
- singlet 保留 quark-gluon mixing matrix；
- 检查 target-mass、higher-twist、finite-volume、finite-z；
- 对 Fourier/reconstruction 的 truncation 和 covariance 做稳定性扫描；
- 只在所有层都完成后报告 g(x,mu) 或 Delta g(x,mu)。

## 7. 不确定内容标记

以下事项明确是“待确认”，不能在后续会话中被写成已完成：

- gluon Clover tree-level normalization kappa_F；当前 matching 诊断使用
  kappa_F=1；
- gluon GF-to-quasi 一圈系数在当前 lattice normalization、color
  projection 和所有 finite-a terms 下的完整适用范围；
- quark非微扰Zchi的质量系统项及正式物理验收（10conf测量和配对已完成，见第14节）；
- quark 0.8/1.1 的 raw output completion；
- gluon channel-specific singlet GF-to-quasi mixing 的完整高阶结果；
- physical continuum/flow-time window；
- finite-z reconstruction 到 x-space 的稳定性；
- transversity 的外部相对符号确认。

以下内容是“历史/诊断”，不要升级为物理结论：

- DeltaG_Note.pdf 中的 HYP5/Wilson3 结果；
- gluon N231 results_v2/results_v3 和 bare_fit_sumdiff_aic_v1；
- fit-level matching 图；
- 单一 tau、单一格距、单一 Pz 的漂亮 plateau；
- ordinary flowed quark C3/C2 在没有 Z_chi 时的 MSbar/PDF 解释。

## 8. 不可随意修改的约定

1. 不能把 HYP-pre-smeared 输入和 thin-link GF matching 混在一起。
2. 不能改变坐标顺序 (x,y,z,t)=(0,1,2,3) 而不重新验证所有 epsilon、
   Clover、dual 和 component labels。
3. gluon unpolarized physical operator 固定为
   (T_U-S_U)_even。
4. gluon helicity physical target 固定为
   (T_H-S_H)_odd；plus 和 temporal-only 只能作为 diagnostics。
5. even_sum/odd_difference 当前没有 1/2。
6. helicity 的完整 complex pol35 必须在 vacuum subtraction 和 covariance
   构造前保留；denominator 是 nopol。
7. 每个 bootstrap replica 先做 ratio，再做最后的 Re/Im projection；
   不能用 mean(C3)/mean(C2) 或 mean(C3/C2) 替换声明的 estimator。
8. 所有 tau、channel、flavor、direction、fit method 必须共享明确对齐的
   configuration order 和 bootstrap index array。
9. NaN padding 和 invalid masks 不是零数据，不能填零后拟合。
10. flow-time、fit window、z cut 必须全局预注册，不能按点挑最小误差。
11. finite-flow bare correlator、MSbar bilocal 和 physical PDF 必须使用
   不同 status 名称。
12. 不要删除旧 products；如需修复，生成 versioned successor 并保存
   code/config/manifest/output SHA256。

## 9. 参考文献和交接入口

核心论文：

- M. Luscher, arXiv:1006.4518；
- M. Luscher, arXiv:1302.5246；
- Monahan and Orginos, arXiv:1612.01584；
- Brambilla and Wang, arXiv:2312.05032；
- Khan et al., arXiv:2107.08960；
- Egerer et al., arXiv:2207.08733v2；
- Balitsky, Morris and Radyushkin, arXiv:2112.02011；
- Chen et al., arXiv:2510.26425v2。

未来 Codex 开始工作时，建议先读：

1. 本文；
2. /public/home/donghx/.codex/skills/gradient-flow-quark-pdf/SKILL.md；
3. /public/home/donghx/.codex/skills/quark-pdf-bare-matrix/SKILL.md；
4. /public/home/donghx/Note/Gradient_Flow/main.tex；
5. 当前任务对应的 collection_validation.json、fit_collection_validation.json
   和 Slurm completion/validation receipts。

最后重新查询 Slurm 和文件 completion markers；不要仅依赖本文中记录的
历史 job state。

---

文档生成日期：2026-09-08  
本次会话标题：Gradient-flow quark/gluon PDF 全链路状态、约定与继续工作清单

## 12. 2026-09-14 更新：quark C3/C2 ratio 拟合图与小流时间任务

### 12.1 新增 ratio/拟合图

本次新增绘图入口：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/plot_quark_ratio_fit_v1.py
```

它读取：

```text
quark_unpolarized_bootstrap_fit_v1/ratio_bootstrap_Nboot50.npz
quark_unpolarized_ratio_fit_aic_v1/bare_matrix_elements_ratio_AIC_Nboot50.npz
```

图中展示 connected `u-d` unpolarized 的

```text
R(T,tau) = Re(C3) / Re(C2_nopol), even-z projection
```

每个 flow time 和动量分别给出 `z/a=0,...,15` 的 insertion-time
ratio。不同颜色对应 `tsep/a=6,7,8,9,10`，点为 50 个 bootstrap
replica 的均值和 `std(ddof=1)`；黑色水平线和灰色带为 direct-ratio AIC
常数拟合的均值与 1-sigma 误差。不同 insertion 点不使用连线。

拟合约定是固定物理门槛后的 `T=8,9,10`，对 cuts `c=1,2,3` 和窗口
`8--9`、`8--10` 做 AIC 加权，再等权混合 cuts。
该产品是 `Nboot=50`、10 个 conf 的 finite-flow bare connected
diagnostic，不是已经归一化或 matching 的 PDF。

运行命令：

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF
MPLCONFIGDIR=/tmp/gradient_flow_mplconfig python -B plot_quark_ratio_fit_v1.py
```

输出（12 页：4 个 flow time × 3 个动量）：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_unpolarized_ratio_fit_aic_v1/ratio_C3_over_C2_with_direct_AIC_fit_Nboot50.pdf
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_unpolarized_ratio_fit_aic_v1/ratio_C3_over_C2_with_direct_AIC_fit_manifest.json
```

manifest 记录了输入 SHA256、flow-time 轴
`tf/a^2=0.2,0.5,0.8,1.0`、`Pz=3,4,5`、tsep 轴、投影、拟合规则和
“finite-flow bare connected diagnostic”状态。

相关理论 Note 已同步加入“Current quark ratio fit and small-flow
checkpoints”一节，并嵌入该图的第一页示例：

```text
/public/home/donghx/Note/Gradient_Flow/main.tex
/public/home/donghx/Note/Gradient_Flow/main.pdf
/public/home/donghx/Note/Gradient_Flow/figures/quark_ratio_C3C2_direct_AIC_Nboot50.pdf
```

Note 使用 tectonic 成功编译（43 页）；日志中没有 `Overfull`、`Float too
large` 或 `Missing character`，仅保留 hyperref 的 PDF bookmark token
warning。

### 12.2 小流时间 `tf/a^2=0.05,0.10` GPU 提交

为 conf

```text
10000, 10100, 10200, 10300, 10400,
10500, 10600, 10700, 10800, 10900
```

提交脚本为：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/submit_quark_flow_005_01_all10.sh
```

它调用未做 HYP 预 smear 的四维 thin-link flow driver：

```text
/public/home/donghx/Gradient_Flow/Blending_L32x96/run_multiflow_channels_l32x96_4gpu_a800.slurm
```

固定参数为 `FLOW_TIME_SPECS='0.05:2;0.10:2'`、
`FLOW_EPSILON=0.01`、`QUARK_PDF_PROFILE=pilot_z`。因此 0.05 是 5 个
耦合 gauge/fermion RK3 flow steps；0.10 从 0.05 checkpoint 重启再做 5 步，累计为 10 步。
每个 conf 使用 `na800-sug`、预约 `donghx_223`、4 张 A800、4 MPI ranks，
日志位于 `/public/home/donghx/Gradient_Flow/Blending_L32x96/output/` 和
`error/`。

提交记录：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/jobs/submitted_005_01_20260913_151237.md
```

作业号为 `3629814--3629823`。2026-09-14 查询时，
`3629814--3629818` 已退出码 `0:0`，`3629819--3629822` 运行中，
`3629823` 等待资源。输出树中 0.05 和 0.10 的 checkpoint 已分别出现
8 和 6 个 conf 目录；运行中的作业仍可能增加该数量。不能只凭 Slurm
退出码宣称所有 projected channel 已完成，最终应在登录节点用与提交相同的
`PABS_LIST=3,4,5`、`MOM_FILTER=Pp3_z,Pp4_z,Pp5_z`、`TSEP_LIST=6,7,8,9,10`
和 `SAVE_ROOT` 运行 `validate_multiflow_conf_l32x96.py --raw`。

### 12.3 当前 quark 结果的解释边界

1. 新 ratio 图使用旧的 10-conf、50-bootstrap 产品，只用于检查 insertion-time
   行为和拟合线，不能替代 N406 gluon 或未来完整 quark 统计。
2. `C3/C2` 只消除核子 overlap；普通 flowed fermion 的 `Z_chi(t)`、SFTE、
   LaMET kernel、连续极限和 singlet mixing 仍未完成。
3. 0.05 和 0.10 是新增 finite-flow correlator checkpoint；在 completion
   guard、轴、hash、temporal-window 和 thin-link 标记全部通过前，不得用于
   物理 PDF 或小流时间外推。
4. 既有 0.8/1.1 任务的 projection 问题仍是独立的历史待修复事项；本次
   0.05/0.10 提交不能自动证明 0.8/1.1 已完成。

更新日期：2026-09-14  
本次会话标题：quark finite-flow C3/C2 ratio direct-AIC 拟合图与小流时间 GPU 提交

## 13. 2026-09-18 更新：独立 quark ringed normalization 测量代码

本节保留9月18日的有限质量pilot记录；临界质量与10conf后续状态由第14节更新。

本节补充并更新前面的历史状态，不改变已有 C3/C2、裸拟合或 matching 数值。
**已完成测量代码、CPU验证、GPU自由场验证及单组态有限质量试算；尚无正式无质量Zchi。**
本节GPU状态已更新，详细数值、作业和证据见13.6。
当前六流时间 SFTE 的 `coefficient_preview` 仍缺少真实 ringed 数据；其正式
`Zchi`、`matrix_ringed`、`matrix_sfte_msbar` 仍为 NaN。

### 13.1 新代码与输入

统一根目录：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/
```

- `core.py`：全体积 Z4 噪声、4D 中心协变导数、APBC、迹估计、精确自由场参考和原子落盘。
- `measure.py`：单 MPI rank/单 GPU 的 PyQUDA 求逆和双场同步多 checkpoint flow；保存逐噪声结果。
- `collect.py`：按裸拟合的组态顺序汇总，复用原 bootstrap 索引；未验收数据禁止正式归一化。
- `compare.py`：同噪声的 direct/restart 或 epsilon-halving 数值对照。
- `l32x96_pilot.json`：有限质量 real-gauge pilot 参数；`free_field.json`：小格点 GPU 验证。
- `validation_review.template.json`：待填写的科学验收记录，不是已通过的证明。
- `run_gpu.sh`：已分配 GPU 作业内的环境包装；不调用 sbatch。
- `test_kinetic.py`、`README.md`：回归测试、全部命令、轴序和验证限制。

仍消费以下裸拟合产品，而不是旧 direct-ratio AIC 分支：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_unpolarized_sixflow_bare_fit_v1/bare_matrix_elements_unpolarized_u_minus_d_sixflow_Nboot50.npz
```

六个 `t/a^2=0.05,0.1,0.2,0.5,0.8,1.0`，epsilon=.01，累计步数5、10、20、50、80、100；
同一批 `10000,10100,...,10900`，Nboot=50，直接使用保存的 `(50,10)` 索引。
测量保留完整96个时间片，没有旧 C3 basis 存储窗口的截断。

### 13.2 算符、符号和归一化，不得随意修改

定义单 flavor、四维体积平均 `E_lat=a^4 <bar chi <->Dslash chi>`。若
`S=D_val^{-1}`、`rho=K eta`、`phi=K S eta`，则

```text
E_lat(U,tau) = -2/(V*Nnoise) sum_noise rho^dagger Dslash_B phi
Zchi(tau)   = -2*Nc / [(4*pi)^2*tau^2 * mean_cfg Re(E_lat)]
```

负号来自费米子闭合收缩；2来自体积和上的左右导数分部积分。迹已包括自旋、颜色、
全部时空，不再除4或3；不能使用简并 u-d 真空动能之差。噪声满足
`E[eta eta^dagger]=I`，不能归一到单位向量。
两端场与规范场使用同一条薄链4D流轨迹。一个噪声只需一次原始 Dirac 求逆，复用所有流时间。
导数是 `aD_mu=[U_mu(x)f(x+mu)-U_mu^dagger(x-mu)f(x-mu)]/2`。
fermion轴为 `(t,z,y,x,spin,color)`，gauge轴为 `(mu,t,z,y,x,color,color)`，mu为x,y,z,t。
流动时嵌入APBC，相应相位从保存的gauge移除后由导数施加一次，禁止重复施加。

已核对的 action 源：`/public/home/sush/EasyDistillation/work/perambulator.3.py:115-168`。
参数为am=-.2295、csw=1.141151096、各向同性、一次4D stout(rho=.125)、时间反周期。
注意三种不同对象：原 valence inverse 的 stout links；算符流起始的 thin links；
distillation basis 的20步 stout links。不能把三者混写。
源代码与每份历史 perambulator 的实际对应关系仍需 provenance 验收，不能只凭目录名确认。
完整 MAT 的 QUDA asymmetric mass normalization 为D_mass=D_kappa/(2*kappa)，
kappa=1/[2(am+4)]；程序检查D_mass q-eta，**不再额外乘2*kappa**。

新的默认求解使用无MG基线。conf10000的32噪声A800试算耗时11分48秒；
QUDA报告21508.5MiB设备分配，但未包含全部CuPy占用，不能称为总峰值显存。
L32x96 dry-run给出33.75GiB粗略估计，不是显存保证。禁止以多个MPI rank运行这个全体积cupy stencil；
可以在后续任务中按组态独立并行。

### 13.3 数据、配对 bootstrap 和安全导出

- `noise_NNNNN.npz` 保存complex `kinetic_a4_noise(flow)`、true residual、iterations、
  gauge误差、plan/gauge/code/runtime标识。自由场额外保存同噪声精确参考。
- `kinetic_conf.npz` 保存complex `(flow,noise)`、每组态噪声均值/误差、原始身份信息；
  固定状态 `measured_unvalidated`。
- collector要求裸产品全部10个组态，显式排序，拒绝重复、缺失、不同mass/action/flow。
- 每个bootstrap先按已有索引平均E，再取倒数，最后与同编号的h样本相乘；不能平均每组态的1/E。
- 保留完整复数测量，输出实部前检查集合虚部与零的统计相容性；默认5sigma只是一致声明的诊断阈值。
- 默认输出的正式Zchi仍为NaN，有限值名为`Zchi_diagnostic_NOT_massless`，现有SFTE消费者会拒绝。
- 正式导出需`--review`：必须绑定全部raw文件SHA256，七项验收均须有实际存在、哈希一致的证据。
  这是一份人工科学验收记录，程序只能检查记录完整性，不能代替科学判断。

`--resume`只复用完全相同plan/gauge/code/seed/Nnoise的数据，逐噪声文件不可覆盖。
更改质量、epsilon、噪声数或积分方式应使用新输出目录。崩溃遗留lock需先核对旧任务，再人工处理；
程序不会自动删除缓存。组态bootstrap对每组态噪声均值采样，不能再简单叠加nested-noise
bootstrap重复计数；需根据逐噪声数据独立做收敛检查。

### 13.4 运行命令

以下均在 `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF` 下运行：

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v quark_ringed_kinetic_v1.test_kinetic
python3 -B -m quark_ringed_kinetic_v1.measure --plan quark_ringed_kinetic_v1/l32x96_pilot.json --conf 10000 --noise-count 32 --output quark_ringed_kinetic_v1/results/10000 --dry-run
# 仅在已分配的单GPU作业中运行，下面不提交任务：
export RINGED_QUDA_LIB_DIR=/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/quda_zero_guard/build
bash quark_ringed_kinetic_v1/run_gpu.sh --plan quark_ringed_kinetic_v1/free_field.json --conf 0 --noise-count 4 --output quark_ringed_kinetic_v1/free_gpu
bash quark_ringed_kinetic_v1/run_gpu.sh --plan quark_ringed_kinetic_v1/l32x96_pilot.json --conf 10000 --noise-count 32 --output quark_ringed_kinetic_v1/results/10000
```

全部10个组态测量后，按README中的collect命令汇总。`--direct`用于restart对照，
epsilon=.005需独立plan与输出目录。后续只有通过审查的无质量产品，才交给
`quark_sixflow_ringed_sfte_v1.py --kinetic ...`；本次没有执行正式SFTE重算。

### 13.5 尚未解决与下一步

1. **关键待确认：临界 valence 质量/手征处方。** am=-.2295不是已验证的临界质量，
   am=0也不是相互作用Wilson的手征极限。当前exporter实现独立调谐critical-mass的路线；
   未实现PCAC质量扫描的手征拟合，也不会忽略PCAC质量误差。
2. 小格点GPU同噪声精确参考、conf10000有限质量试算、步长减半、direct/restart和后端对照
   已完成。单组态噪声收敛有前缀数据支持；不能代替系综收敛，也未独立完成每份历史传播子的来源核对。
3. Luscher 1302.5246式(6.8)的c_fl接触项在本算符完整迹中为零：
   K自旋对角且tr_spin gamma_mu=0，因此Tr[Dslash K Kdag]=0。
   这不等于已完成全部O(a)改进，质量相关改进和有限格距系统误差仍待评估。
4. 400向量基包含随机高模补全，不是单纯最低400模；能否复用于无偏vacuum trace仍待验证。
   新实现采用独立全空间噪声求逆，不对旧distillation资料作无证据的完备性假设。
5. 最新代码层面15项CPU测试及既有7项ringed/SFTE测试通过，共22项；CPU通过不代表真实GPU测量完成。
   同噪声自由场、迹负号、APBC、规范协变、RK3步长、重复/缺失组态和未验收拒绝均有测试。
   GPU运行后再次执行15项测量模块CPU测试通过。
   合成验收记录只在临时单元测试中验证既有消费者的接口，不是科学验收证据。
   日志位于新模块的 `validation/cpu_tests.log`；真实conf10000输入预检见
   `validation/conf10000_preflight.log`，原始gauge payload大小为1811939328 bytes。

Note新增第4.7节 `sec:ringed-kinetic-implementation`，详细推导上述真空迹、噪声估计和配对统计。
源文件 `/public/home/donghx/Note/Gradient_Flow/main.tex`，编译PDF为同目录 `main.pdf`，
原实现段位于第9至10页；加入GPU结果后页码可能移动，请用章节标签定位。
原编译日志保留于新模块 `validation/note_build_pass*.log`。
本次GPU更新仍在第4.7节，现位于第9至11页，整份PDF为46页；两次Tectonic编译通过，
无Overfull、Float too large、Missing character。日志为 `validation/note_gpu_build_pass1.log`
和 `validation/note_gpu_build_pass2.log`；46页渲染缩略图及第9--11页放大检查通过。
资料依据：arXiv:1403.4772、1302.5246、2312.05032v3；SFTE line-mass与endpoint系数未改动。

### 13.6 GPU验证与质量处方审查的实际结果

三个作业均已COMPLETED、退出0：3643531自由场（19秒），3643532 conf10000的32噪声
（11分48秒），3643534真实场2噪声配对检查（9分09秒）。使用na800-sug分区的A800。

完整证据与结果位于：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/validation/gpu_report_20260918.md
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/validation/gpu_summary_20260918.json
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/runs/gpu_20260918_BjcGfO/
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/runs/real_checks_20260918_aCQ1AB/
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_kinetic_v1/mass_prescription_audit.md
```

主数据为第一run目录下 `conf10000_pilot/kinetic_conf.npz`，另有32份逐噪声文件。
汇总程序 `report_gpu.py` 核查原始文件哈希、分片与aggregate数值及配对比较结果。
自由场对同噪声精确傅里叶参考的最大相对误差为1.4938e-5（eps=.01）和1.8037e-6
（eps=.005），缩小8.28倍；真实场restart最大相对差1.8625e-15，步长减半1.4944e-5，
私有/原共享QUDA后端3.3440e-12，全部通过预先声明的数值阈值。

conf10000主试算最大full-MAT真实残差7.4581e-11，最大gauge幺正性/行列式误差4.3993e-14。
六流时间的Re E_lat依次为
`[-2.2377513461,-1.3620222639,-.5954080579,-.1208140199,-.0472290166,-.0298496075]`。
相对噪声标准误为0.00369%至0.04346%，虚部均值各自距零不超过1.64个噪声标准误。
**这只是单组态随机迹误差，不是系综误差，更不是正式Zchi误差。**

单位场测试曾暴露共享QUDA的exp(0)零除漏洞；`quda_zero_guard/`仅对精确零生成元
返回原U，其余路径不改，共享安装未改。安装PyQUDA的DT_RPATH会优先加载旧库，
所以measure现在在导入扩展前显式加载私有库，并记录实际进程maps中的库及SHA256：
`1de429fd5371af1fe033e47318a177e3b6be71deae5d30c4b935829b720bb22d`。
禁止用早期失败目录继续正式计算，也不要在运行任务时覆盖私有库。

已额外核对系综HMC输入
`/public/group/lqcd/configurations/CLOVER/beta6.41_mu-0.2295_ms-0.2050_L32x96/beta6.41_mu-0.2295_ms-0.2050_L32x96_restart_46500.xml`
中的light action：与am=-.2295、cSW=1.141151096、1步stout rho=.125一致。
HMC中其他辅助质量不是PCAC扫描，不能据此确定amcrit。

**临界质量数值待确认。** 在已检查的项目/Note/相关传播子代码和HMC调参文本中未找到
相同作用量的PCAC数据；没有声称整个集群都不存在。应取得既有PCAC/临界质量证据，
否则另测同组态同源的多价质量PP、A4P，声明欧式符号、差分与cA，拟合平台和PCAC零点。
只调价PCAC质量而海质量固定，须标记为固定海质量的价手征处方，不能称完整海手征极限。
当前collector只实现已独立确认的direct-critical-mass输入，尚未实现多质量手征拟合。

后续先完成质量处方，再测与裸拟合同一批10000..10900十组态的E，传播质量误差并验收，
最后复用原50bootstrap索引平均E、取倒数、配对乘h，再做SFTE。
此次未执行正式matching、未生成accepted Zchi；pilot保持 `measured_unvalidated`。

更新日期：2026-09-18  
本次会话标题：quark ringed GPU自由场验证、单组态试算与手征处方核查

## 14. 2026-09-20：ringed normalization 的推导、实测和操作

### 14.1 最新结论与适用范围

已完成L32x96 quark unpolarized、connected **u-d** 的10conf配对ringed归一化。
conf为10000、10100、…、10900；每conf 32个全体积Z4噪声；
原裸拟合50组bootstrap索引保持不变，seed=1115。
六个无量纲流时间为`theta=t_f/a^2=0.05,0.1,0.2,0.5,0.8,1.0`。
此处代码有时写`tau`，不代表C3的插入时间或`tsep`。

结果根目录（以下记为`RESULT`）：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920
```

`sixflow_sumdiff`为原六flow Sumratio差分AIC；`fourflow_ratio`为原四flow
直接ratio AIC，仍标记unweighted diagnostic。两者分目录保存，不能混作同一拟合。
归一化没有重拟合C3/C2，没有改变裸拟合中心值、拟合窗口或fallback标记。
这些有限flow的ringed矩阵元及额外保存的SFTE诊断均不是最终PDF。

### 14.2 为什么 C3/C2 还需要 ringed normalization

四维流满足

\[
\partial_{t_f}B_\mu=D_\nu G_{\nu\mu},\qquad
\partial_{t_f}\chi=D_\mu[B]D_\mu[B]\chi,\qquad \chi(0)=\psi.
\]

流抑制高动量，但普通flowed fermion仍有场的乘法重整化。
核子C2和C3的比值消除源汇重叠等因子，C2不含插入算符的两个flowed endpoint，
所以不消除这两个端点的归一化。ringed定义用一个独立的真空期望值固定场尺度。

约定单flavor、欧式Hermitian gamma、`{gamma_mu,gamma_nu}=2 delta_mu_nu`，定义

\[
E(t_f)=\left\langle\bar\chi(t_f,x)
\overleftrightarrow{\slashed D}\chi(t_f,x)\right\rangle,
\quad\overleftrightarrow D_\mu=D_\mu-\overleftarrow D_\mu.
\]

**连续自由无质量场的分子如何得到：**
`K(p)=exp(-t_f p^2)`，`S(p)=(i gamma.p)^(-1)=-i gamma.p/p^2`。
Wick收缩有负号；左右导数在体积平均中产生`2 i gamma.p`；两个流核产生
`exp(-2 t_f p^2)`。因此

\[
\begin{aligned}
E_{\rm free}(t_f)
&=-2N_c\int\frac{d^4p}{(2\pi)^4}e^{-2t_fp^2}
 \operatorname{tr}_{s}\left[(i\slashed p)\frac{-i\slashed p}{p^2}\right]\\
&=-8N_c\int\frac{d^4p}{(2\pi)^4}e^{-2t_fp^2}
=-\frac{2N_c}{(4\pi)^2t_f^2}.
\end{aligned}
\]

最后一步用`integral d^4p/(2pi)^4 exp(-alpha p^2)=1/(16pi^2 alpha^2)`。
这解释了分子的负号、2和颜色数，而不是人为把真空动能设为正。
采用Brambilla–Wang的无质量处方：

\[
\mathring Z_\chi(t_f)=\frac{-2N_c}{(4\pi)^2t_f^2 E(t_f)|_{m_R=0}},\qquad
\mathring\chi=\sqrt{\mathring Z_\chi}\chi,\quad
\mathring{\bar\chi}=\sqrt{\mathring Z_\chi}\bar\chi.
\]

代码字段`Zchi`就是此处的`mathring Z_chi`，不要与文献中未加圆圈的
最小减除场重整化常数混为一谈。若普通场作常数重标度`chi -> c^(1/2) chi`，
则`E -> c E`、`mathring Z -> mathring Z/c`，ringed场不变；这是消除普通场
归一化依赖的直接说明。连续自由无质量、无限体积极限有`mathring Z=1`，
有限格距/有限体积不必等于1。

非局域算符有两个端点：

\[
O_\Gamma^{\rm ringed}=\mathring{\bar\chi}(0)\Gamma W_B(0,z)\mathring\chi(z)
=\mathring Z_\chi O_\Gamma^{\rm GF},\qquad
h^{\rm ringed}=\mathring Z_\chi h^{\rm GF}.
\]

因此一个bilocal乘**一个Zchi**，不是sqrt(Zchi)或Zchi²。
相同轻夸克作用量和质量处方下，u、d共用此因子，所以`(u-d)_ringed=Zchi*(u-d)_GF`。
真空分母必须是单flavor动能；简并u-d真空差为零，不能用于归一化。
若改用Nf个简并flavor之和，分子也必须同时乘Nf。

### 14.3 从真空期望到代码中的随机迹

令`theta=t_f/a²`、`V4=32³×96`、`S=D_mass^(-1)`、`K=K_theta`。
在固定组态上`<chi bar chi>_fermion=K S K†`。对全体积做分部求和，
左右导数贡献相等；Grassmann闭合收缩给出

\[
E_{{\rm lat},U}(\theta)
=-\frac{2}{V_4}\operatorname{Tr}[\slashed\nabla_B K S K^\dagger],\qquad
E_{\rm lat}=a^4E,\quad
Z_\chi=\frac{-6}{(4\pi)^2\theta^2\langle\operatorname{Re}E_{{\rm lat},U}\rangle_U}.
\]

`t_f² E=theta² E_lat`，所以求Zchi时无须再乘`a^4`或fm转换。
trace包括全部时空、自旋4和颜色3，只除V4，不再除4或3。

随机向量每分量独立取`1,i,-1,-i`，满足`E_eta[eta eta†]=I`。利用
`E_eta[eta† A eta]=Tr A`及迹的循环性：

\[
D_{\rm mass}q_r=\eta_r,\quad \rho_r=K\eta_r,\quad\phi_r=Kq_r,
\qquad e_{Ur}(\theta)=-\frac{2}{V_4}\rho_r^\dagger\slashed\nabla_B\phi_r,
\quad\widehat E_U=\frac1{32}\sum_r e_{Ur}.
\]

一个noise只需一次Dirac求逆，随后一起flow源和解，在所有theta保存checkpoint；
不需要构造稠密K、额外积分K†或重新计算原核子C3/C2。
随机迹对E是无偏的，但有限样本的倒数Z是非线性估计量，不能声称严格无偏。

实际中心协变导数为

\[
(\nabla_\mu f)(x)=\frac{U_\mu(\theta,x)f(x+\hat\mu)
-U_\mu^\dagger(\theta,x-\hat\mu)f(x-\hat\mu)}2,
\quad \slashed\nabla=\sum_{\mu=0}^3\gamma_\mu\nabla_\mu.
\]

对应`quark_ringed_kinetic_v1/core.py`的`shift_fermion`、`hop`、
`dslash_central`和`kinetic_sample`。字典如下：

| 对象 | 约定 |
|---|---|
| fermion数组 | `(t,z,y,x,spin,color)` |
| gauge数组 | `(mu,t,z,y,x,color,color)` |
| mu索引 | `0,1,2,3 = x,y,z,t`；空间数组轴`3-mu` |
| gamma | `gamma_dr()`，与原flow代码一致的DeGrand–Rossi欧式基 |
| 时间边界 | fermion反周期；空间周期；导数跨全局时间边界时只乘一次负号 |
| flow中的规范链 | 同一thin-link 4D轨迹；保存的链先去除嵌入的APBC，导数自行处理 |
| source噪声 | 全自旋/颜色/时空Z4，不归一到单位向量 |
| 传播子 | full MAT，QUDA asymmetric mass normalization；不额外乘2kappa |

`D_mass=D_kappa/(2kappa)`，`kappa=1/[2(am+4)]`。需检查的是真实
`norm(D_mass q-eta)/norm(eta)`，不能仅凭preconditioned残差或退出0认可。
临界点使用GCR+MG；改变求解器必须保持这个原方程及观测量一致。

这里有三类链，不能混用：价求逆作用量的一步4D stout（rho=.125）、
算符及真空动能flow起始的thin links、distillation基的20步stout链。
真空测量覆盖全部96个时间片，没有C3基的时间存储窗口截断。
Lüscher的`c_fl`接触项对本**完整trace期望**为零：K自旋对角，
`Tr[Dslash K K†]=0`来自`tr_spin gamma_mu=0`。
有限noise估计不必逐noise为零；该恒等式也不代表已经完成全部O(a)改进。

### 14.4 为什么每个 conf 要对应，但不使用逐 conf 的倒数归一化

每个conf保存其动能`E_c(theta)`，并和原裸拟合的conf顺序严格对应。
ringed定义中的Zchi本身是**系综/流时间标量**，不是随机的逐conf算符。

\[
\overline E_j=\frac1{N_c^{\rm cfg}}\sum_c\operatorname{Re}\widehat E_c(\theta_j),
\quad Z_j=\frac{A_j}{\overline E_j},\quad
A_j=-\frac6{(4\pi)^2\theta_j^2},\qquad
h^R_{j\ell}=Z_j h^{\rm bare}_{j\ell}.
\]

这里`Ncfg=10`，不等于颜色数Nc=3；`ell=(pz,z)`。
给定原始索引`I[b,k]=bootstrap_indices[b,k]`，形状(50,10)：

\[
\overline E_j^{(b)}=\frac1{10}\sum_{k=1}^{10}\operatorname{Re}
 \widehat E_{I[b,k]}(\theta_j),\qquad
Z_j^{(b)}=\frac{A_j}{\overline E_j^{(b)}},\qquad
h_{j\ell}^{R,(b)}=Z_j^{(b)}h_{j\ell}^{{\rm bare},(b)}.
\]

索引中重复抽中的组态必须保留重复次数。按原裸拟合中心估计量乘Z得到报告中心值，
不换成50样本的均值；误差为`std(h_ringed_bootstrap,axis=1,ddof=1)`，不再除sqrt(50)。
跨flow、z和动量的协方差由同一批乘积样本构造。
线性误差传播也可看出配对的必要性：

\[
\operatorname{Var}(Zh)\simeq Z^2\operatorname{Var}(h)+h^2\operatorname{Var}(Z)
+2Zh\operatorname{Cov}(Z,h).
\]

不能独立重采样Z再乘h，不能把两份误差独立平方相加，也不能先算每conf的`A/E_c`
再取平均或逐conf乘ratio；后者改变了ringed定义。
32-noise均值的噪声已存在于组态间涨落中，本轮不另加一次嵌套noise bootstrap方差。
噪声收敛和组态自相关仍应分别检查；50个bootstrap样本不会把10conf的信息量变成50conf。

### 14.5 临界质量、自由场和小流时间的注意事项

文献`arXiv:2408.03548v3` Table 3（F32P30）给出`amcrit=-0.23539(3)`。
与本地HMC作用量核对：beta6.41、u0=.863437、v0=.956942、
`csw=v0^-3=1.141151096`、一次4D stout rho=.125。
当前真空求逆使用该中心值，海质量仍为am_l=-.2295、am_s=-.2050。
这是**文献临界价质量、固定有质量海**的处方；没有进行本地PCAC复测或全海手征外推。
Wilson的am=0不等于相互作用手征极限，pion/eta_s质量也不能直接代入Dirac裸质量。

原质量敏感性试算在conf10000、相同4noise比较`-.23542,-.23539,-.23536`，
另以旧文献`2310.00814v2`的`-.23545`作独立的处方变化。
不同文献值不当作独立统计样本加权平均。对同噪声结果定义

\[
\Delta_mZ_j=\frac{Z_j(m_c+\sigma_m)-Z_j(m_c-\sigma_m)}2,
\quad (C_m)_{jk}\simeq\Delta_mZ_j\Delta_mZ_k,
\quad \delta h^R/h^R=\delta Z/Z=-\delta E/E.
\]

最后一个等式只描述固定裸h时的归一化响应。
4noise试算的带符号1sigma响应（占Z的百分比）为
`[.0023477,.0028591,.0040955,.0066328,.0048572,.0011105]%`，
响应本身的配对noise误差为`[.0009035,.0004052,.0007235,.0026664,.0046714,.0060698]%`。
后两点响应未显著分辨，不能据小中心值宣称质量误差为零。
这些是**单组态4noise诊断**；当前10conf after-Zchi误差条未加入质量系统项。
32noise多质量补算的部分文件已存在，但完整四质量验收/报告待确认，不能冒充已完成。

有限a自由场也有显著cutoff依赖。`core.free_wilson_kinetic`的精确格点参考是

\[
E_{\rm lat}^{\rm free}(\theta,m)=-\frac{8N_c}{V_4}\sum_p
 e^{-2\theta\hat p^2}\frac{\sum_\mu\sin^2p_\mu}
 {[m+\sum_\mu(1-\cos p_\mu)]^2+\sum_\mu\sin^2p_\mu},
\quad\hat p^2=4\sum_\mu\sin^2(p_\mu/2).
\]

时间动量为`2pi(n+1/2)/Nt`。有限格点上theta->0时动能有UV截止，
而连续分子仍含theta^-2，所以Z不必接近1；极小theta的较大Z不能单凭这一点判为bug。
这也不自动证明实测差异全由cutoff解释。当前六flow的`r_flow/a=sqrt(8theta)`约为
`.632,.894,1.265,2,2.530,2.828`；前两点甚至小于一个格距。
不要擅自用格点tree-level分子替换连续分子：那会引入有限a的额外归一化处方，
需要单独定义、保存并重新审查SFTE接口。本次未作该替换。

### 14.6 已完成结果和验证证据

| theta | Zchi | 原50组配对bootstrap误差（固定质量） |
|---:|---:|---:|
|0.05|6.7679838065|0.0003478494|
|0.10|2.7791305139|0.0001365254|
|0.20|1.5885047901|0.0000585582|
|0.50|1.2514982449|0.0000741719|
|0.80|1.2505405670|0.0001376585|
|1.00|1.2665887094|0.0002108594|

GPU conf10800恢复任务3707765及CPU分析/绘图3709104均COMPLETED、exit 0。
恢复任务26分钟以内完成。全部10conf最大真实full-MAT残差`7.52886e-13`，
保持原`1e-9`验收门；六flow动能虚部均通过预先声明的5sigma一致性检查。
这不是只看Slurm退出码：raw/shard内容、噪声身份、代码/规范场哈希和SU(3)也逐项核对。

conf10800原GCR求逆停滞；恢复分支只更改数值精度与GCR参数：
预条件计算SINGLE、null存储HALF、Krylov16、reliable_delta=.1、每轮500次，
允许最多4次原方程defect correction，实际32个noise均在首轮通过。
恢复前后同noise0动能相对差`1.70589245e-10`；恢复版自由场同噪声参考相对差
`1.44009281e-5`低于既定`5e-4`门。conf10800完整重算32noise，没有混拼版本。
历史失败目录保留，不会用挑出的成功noise替代完整测量。

9月18日既有验证还包括：自由场eps .01/.005误差`1.4938e-5/1.8037e-6`，
真实场restart/direct相对差`1.8625e-15`，步长减半`1.4944e-5`。
这些测量的质量、噪声数和后端见各自report，不能声称全部在10conf上重复过。

### 14.7 代码、文件、轴序和复现入口

项目根（以下命令的cwd）为`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF`。

| 目录/入口 | 用途 |
|---|---|
|`quark_ringed_kinetic_v1/core.py`|中心导数、随机trace、自由场精确参考；核心估计器未改 |
|`quark_ringed_ensemble_v1/runs/critical_10conf_20260920_rz65f_im/`|原campaign，plan/manifest、conf10000复用及其余组态测量 |
|`quark_ringed_recovery_v7/code/quark_ringed_kinetic_v1/measure.py`|conf10800实际运行的数值恢复版本 |
|`quark_ringed_recovery_v7/measurement/kinetic_conf.npz`|conf10800完整32noise结果；不能找原campaign的缺失聚合文件 |
|`quark_ringed_recovery_v7/solver_recovery_review.json`|仅数值求解恢复的等价性审查，不是正式物理验收 |
|`quark_ringed_ensemble_v1/pipeline.py`|`audit_raw`逐组态核验、`paired`配对、`analyze`输出 |
|`quark_ringed_ensemble_v1/review_recovery.py`|原plan/代码/自由场/noise0/32noise恢复审查 |
|`quark_ringed_ensemble_v1/plot_after_zchi.py`|只画ringed层，核对输入hash、样本和质量标记 |
|`quark_ringed_recovery_v7/analysis_code/`|已完成CPU任务的实际分析代码快照 |
|`quark_ringed_mass_pilot_v1/runs/critical_mass_20260918_cuonexuz/analysis/`|4noise质量响应report/NPZ/PDF |

原裸输入：

```text
quark_unpolarized_sixflow_bare_fit_v1/bare_matrix_elements_unpolarized_u_minus_d_sixflow_Nboot50.npz
quark_unpolarized_ratio_fit_aic_v1/bare_matrix_elements_ratio_AIC_Nboot50.npz
```

原campaign的`reused/conf10000`和`measurements/conf10100`至`conf10900`
保存9个可复用完整组态（不含旧conf10800）；恢复review明确覆写**分析路径映射**，
没有覆盖原失败文件。不要直接对原campaign旧snapshot运行analysis并忽略缺失错误。

`RESULT/analysis/{sixflow_sumdiff,fourflow_ratio}/paired_matrix_elements.npz`的schema为
`quark_ringed_paired_10conf_v1`，用`np.load(...,allow_pickle=False)`读取：

| 字段 | 六flow shape/说明 |
|---|---|
|`flow_times,confs,bootstrap_indices`|`(6),(10),(50,10)`；索引值为conf轴位置 |
|`kinetic_noise`|complex `(6,10,32)`，轴flow/conf/noise |
|`kinetic_config`|complex `(6,10)`，每conf噪声均值 |
|`Zchi,Zchi_error,Zchi_bootstrap,Zchi_covariance`|`(6),(6),(6,50),(6,6)` |
|`bare_central,ringed_central,ringed_error`|`(6,3,21)`，轴flow/pz/z |
|`bare_bootstrap,ringed_bootstrap`|`(6,50,3,21)`，轴flow/bootstrap/pz/z |
|`fit_valid,source_*`|原拟合质量、fallback及可用的原参数 |
|`sfte_central,sfte_bootstrap,sfte_error`|仅额外的matching诊断；z=0为NaN |
|`small_flow_window_mask,formal_result_valid`|尺度窗口与正式认可分开；后者全False |

四flow版本把flow轴6改为4。
所有`ringed_*`在z=0仍有限，不受非局域SFTE的z=0排除影响。
固定参数和原始输入hash在同目录`manifest.json`及`RESULT/analysis/summary.json`。

已有结果核验与重新绘图，无须提交GPU任务：

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /public/home/donghx/AnaConda/bin/python3 -B -m unittest -v \
 quark_ringed_ensemble_v1.test_pipeline

# REPlot必须是一个尚不存在的新目录；不要覆盖当前结果。
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ringed_plot_mpl \
 /public/home/donghx/AnaConda/bin/python3 -B -m quark_ringed_ensemble_v1.plot_after_zchi \
 quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/analysis \
 --output /tmp/quark_ringed_10conf_replot
```

`finish_recovery.py`会同时创建review、分析与图，不允许覆盖已存在review/结果。
重画图应使用上面的plot入口。需要重新分析时，用已冻结analysis_code中的
`pipeline analyze ORIGINAL_RUN --recovery-review REVIEW --output NEW_ANALYSIS`，
其中`ORIGINAL_RUN`和`REVIEW`是上表的绝对路径；`NEW_ANALYSIS`必须不存在。
没有新测量需求时，不重复提交run.slurm。

已完成图表的绝对路径：

- [六flow after Zchi汇总](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/figures/sixflow_sumdiff/after_Zchi_allflow.pdf)
- [只显示theta>=0.2](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/figures/sixflow_sumdiff/after_Zchi_tau_ge_0p2.pdf)
- [逐flow分页](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/figures/sixflow_sumdiff/after_Zchi_eachflow.pdf)
- [原四flow ratio拟合 after Zchi](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/figures/fourflow_ratio/after_Zchi_allflow.pdf)
- [六flow ringed归一化前后](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/analysis/sixflow_sumdiff/ringed_before_after.pdf)
- [Zchi与统计误差](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_ringed_ensemble_v1/results_10conf_after_Zchi_20260920/figures/sixflow_sumdiff/Zchi_10conf.pdf)

图按n_z=3,4,5分子图，Pz约1.50,2.00,2.50GeV；横轴z/a，首张0..15、
xlim=-.5..15.5，汇总另附0..20页。flow错位±.2格距，误差棒不连线。
空心点是原fallback或未加权ratio诊断，不代表不存在测量。

### 14.8 与后续 matching 的接口和待确认事项

继续使用Brambilla–Wang修正的一环式（CF=4/3）：

\[
\begin{aligned}
C_\psi&=1-\frac{\alpha_s C_F}{4\pi}
 [3\ln(2\mu^2t_f e^{\gamma_E})+2+\ln432],\\
\delta m&=-\frac{\alpha_s C_F}{4\pi}\frac{\sqrt{2\pi}}{\sqrt{t_f}},\\
h^{\overline{\rm MS}}&=C_\psi^{-1}e^{-\delta m|z|}h^{\rm ringed}.
\end{aligned}
\]

在此符号下delta_m<0，逆转换指数是`exp(-delta_m |z|)`。已有ringed样本
不能再乘一次Zchi。配套SFTE诊断取mu=2GeV、alpha_s=.30；对相同每个bootstrap
乘确定性系数，仍保留共享索引。当前误差条没有新增alpha_s/尺度/截断系统误差。
不要同时叠加覆盖同一线性发散的hybrid或ratio减除。
ringed归一化可用于局域z=0，但上面非局域小flow渐近式不能直接用于z=0。

**已确认、不能随意改：** 单flavor分子-6、闭合迹负号和左右导数2、全体积trace、
APBC一次、无额外2kappa、共同thin-link 4D flow、先平均再求逆、一个bilocal一个Zchi、
原50×10索引/中心值/fallback、两个裸拟合方法分开、真实残差门不放宽。

**仍待确认／下一步具体操作：**

1. 审查每份历史perambulator与作用量/场归一化来源的实际对应，及所需边界/质量改进。
   已有source脚本和HMC一致性不等于逐产物provenance证明。
2. 检查32noise四质量试算的完整性和失败日志；补齐同噪声质量响应后，
   将共同amcrit不确定度作为跨flow相关系统项传播。单conf4noise响应不可直接当作系综总误差。
3. 检查10conf自相关、噪声收敛、原拟合窗口稳定性及完整六方向估计量。
   归一化不修复原ratio未加权拟合或fallback的物理局限。
4. SFTE之前保留`a/r_flow`、`r_flow/|z|`、`sqrt(t_f)Pz`、`t_f Lambda²`窗口诊断，
   多格距应固定物理flow尺度；现有一个格距不能独立完成连续极限。
5. 当前metadata明确`statistical_pairing_complete=True`、`formal_physics_acceptance=False`，
   `formal_result_valid`全False，不能自动升级旧`ringed_z_chi_v1` JSON为validated。
   正式无质量处方认可后再交给SFTE/LaMET，后者仍需对应通道kernel与有限z重构。

参考依据：Makino–Suzuki arXiv:1403.4772v5 §3.2（ringed定义、自由场动能）；
Lüscher arXiv:1302.5246（fermion flow及边界接触项）；
Brambilla–Wang arXiv:2312.05032v3 §3（单flavor无质量定义）及一环matching；
CLQCD arXiv:2408.03548v3 Table 3（临界质量）；旧值来自2310.00814v2 Table 3。
实际来源和结果以campaign manifest、recovery review及RESULT中的manifest为准。

Note同步更新在`/public/home/donghx/Note/Gradient_Flow/main.tex`第4.7节，
含4.7.1理论分子推导、4.7.2随机迹/格点实现/配对统计、4.7.3质量处方、
4.7.4完整10conf结果。更新PDF中该部分位于第9–16页，Zchi数值表在第14页，
两张after-Zchi图在第15、16页。正文保留原Note的英文语言与符号。
构建沿用本项目的Tectonic（XeTeX）工具；未安装独立xelatex命令。
两轮实际编译、数据表与NPZ核对、完整50页渲染及新增页放大检查记录归档于
`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/ringed_doc_update_20260920_oygtSM/validation/`。
同级`original/`保存本次更新前的Markdown、Note源码、PDF和参考文献备份。

更新日期：2026-09-20。会话标题：ringed normalization 理论推导、10conf配对操作与Note更新。

## 15. 2026-10-09：λ 重构、Fourier、quasi matching 的现有结果和计算交接

### 15.1 已完成什么、尚未证明什么

本节依据本次读取的 README、manifest、验证报告、代码和既有 PDF 更新。
本次整理文档和复制现有结果图，没有重新拟合、重新采样或重复施加 matching。
“数值产品已生成”和“得到受控物理 PDF”是两个不同状态。
下文相对路径均位于 /public/group/lqcd/donghx/Gradient_Flow_GluonPDF。

| 分支 | 统计量与输入 | 已有结果 | 当前解释边界 |
| --- | --- | --- | --- |
| gluon unpolarized | L32x96，406 conf，1000 paired bootstrap，seed1115；(TU−SU)_even 实部 | GF conversion → 固定 a 的 flow-zero → λ 尾部 → Fourier → gg-only LaMET → CT18NLO | 单格距、忽略 singlet mixing、微扰 hybrid proxy、短距离/尾部模型与尺度窗口未受控 |
| quark 三极化、u/d/u−d | L32x96，20 conf，100 paired bootstrap；Ratio relaxed → GF conversion → flow-zero | 27 套 Fourier HDF5 与 27 套 quasi-matched HDF5；比较图 | Zchi=1 待验证；只保留一个复数分量；尾部频繁触及边界；connected-only |
| quark 实测 Zchi 与 Zchi=1 对照 | L32x96，10 conf，50 paired bootstrap；unpolarized u−d；Sumratio/Ratio | 两种归一化 × 两种上游，共四条 λ/Fourier/直接 MSbar LaMET 结果 | Re 输入只重构价夸克投影；n_z=3,4 尾部模型全部低 Q；不得与20conf产品合并 |
| gluon helicity | 已有有限流裸矩阵元/GF候选转换和 flow 分析 | 本次未确认与 unpolarized 同等完整的最新 λ/Fourier/LaMET 链 | 端点系数的独立 partonic 验证及完整极化 kernel 对接待确认 |

L32x96 的 a=0.0775 fm，Pz=2πn_z/(32a)，n_z=3,4,5
对应约1.50、2.00、2.50 GeV。文件名Pz3通常表示编号3，不是3 GeV。
λ=(z/a)2πn_z/32=z_fm Pz_GeV/(ℏc)，ℏc=0.1973269804 GeV fm。
原中心和bootstrap分开保存；误差为 std(samples,ddof=1)，不除以√Nboot。

### 15.2 Gluon unpolarized：当前主入口与图表

统一结果根目录：

/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/gluon_unpolarized_stages_CT18NLO_20260921

| 阶段 | PDF（相对于统一结果根目录） |
| --- | --- |
| 裸矩阵元，四个 flow | 01_bare/bare_selected_flows.pdf |
| GF conversion 后，四个 flow | 02_GF_matching/GF_matched_selected_flows.pdf |
| flow-zero 矩阵元与逐 z 拟合 | 03_flow_zero/tau0_matrix_elements.pdf；03_flow_zero/flow_fits_all_z.pdf |
| λ 尾部重构 | 04_lambda_Fourier_LaMET/lambda_tail_reconstruction.pdf |
| Fourier quasi | 04_lambda_Fourier_LaMET/Fourier_quasi.pdf |
| quasi matching 前后 | 04_lambda_Fourier_LaMET/quasi_matching_before_after.pdf |
| matching 后三动量叠图 | 04_lambda_Fourier_LaMET/lightcone_momenta_overlay.pdf |
| 绝对幅度 xg 与 CT18NLO | 05_CT18NLO/gluon_lightcone_vs_CT18NLO.pdf |
| xg/〈x〉g 与 CT18NLO | 06_normalized_CT18NLO/figures/normalized_gluon_vs_CT18NLO.pdf |

[λ重构](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/gluon_unpolarized_stages_CT18NLO_20260921/04_lambda_Fourier_LaMET/lambda_tail_reconstruction.pdf)、
[Fourier](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/gluon_unpolarized_stages_CT18NLO_20260921/04_lambda_Fourier_LaMET/Fourier_quasi.pdf)、
[quasi matching](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/gluon_unpolarized_stages_CT18NLO_20260921/04_lambda_Fourier_LaMET/quasi_matching_before_after.pdf)、
[归一化CT18对比](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/gluon_unpolarized_stages_CT18NLO_20260921/06_normalized_CT18NLO/figures/normalized_gluon_vs_CT18NLO.pdf)。

实际主数据来源（保留幅度，不除以z=0）：

~~~text
matching/unpolarized_relaxed_q1em6_chi10_minerr_v3/
  reselected_matched/unpolarized_tau*_selected.npz
  unpolarized_tau0_selected.npz
  lambda_fourier_quasi_no_z0_v1/quasi_support_refined/
    no_z0_xg.npz
    per_momentum/Pz{3,4,5}.npz
~~~

旧M(z)/M(0)形状分支不能替代绝对归一化主分支。
有限流共14点：.1,.2,.3,.4,.5,.6,1,1.4,1.8,2.2,2.6,3,3.4,3.8。
flow-zero用tau≥1的8点，连续至少3点的21个窗口，
完整跨flow协方差、线性M=M0+cτ和逐replica AIC。
75个中心和75000个截距样本均有合格窗口，无新增flow阶段回退。
上游有限流1050个中心全部合格；1050000个样本位置中1598个采用有标记的最小误差回退。
这些标记继续传播，不能因下游完成而删除。
实际ratio在ensemble/每次bootstrap内平均C2/C3、执行真空减除，
再算Re(C3)/Re(C2_nopol)；历史逐conf相除的请求不能冒充当前实际estimator。

### 15.3 Gluon：conversion、绝对归一化和重构

相对于z Wilson line，F_tx、F_ty、F_xy均为perp-perp；
unpolarized不能给电场项错配parallel sector。当前非局域转换：

\[
M_{\rm MS}(t,z)=e^{-\delta m|z|}M_{\rm flow}(t,z)/c_\perp^2,\quad
c_\perp=1+\frac{\alpha_sC_A}{4\pi}\ln(2\mu^2t e^{\gamma_E}),\quad
\delta m=-\frac{\alpha_sC_A}{4\pi}\sqrt{\frac{2\pi}{t}}.
\]

μ=2 GeV，αs=.25，CA=3，κF=1，t使用GeV⁻²。
κF=1为声明假设，独立树级绝对归一化验证待完成。
line-mass系数和负号不能擅自改变；严格展开逆与未展开逆属于不同高阶处方，
已有结果的处方需按manifest保留。
代码Clover对应a²gF；基本表示颜色迹的1/2与even求和的2抵消；
ratio自带态归一化1/(2E)，没有额外L³、Nc或新的g因子：

\[
H(z)=-\frac{2R_{\rm MS}(z)}{4\pi\alpha_s(aE)},\qquad
Q(x)=\frac1\pi\int_0^\infty d\lambda\cos(x\lambda)H(\lambda).
\]

正x输出Q=xg(x)，H(0)=2〈x〉g。不要取绝对值或强制有限区间面积为1。
能量与局域动量分数使用同组bootstrap配对传播。
z=0单独使用spin-2 conversion，不能把非局域端点公式延伸到局域点：

\[
C_{\rm loc}=1+\frac{\alpha_sC_A}{4\pi}
\left[\frac{11}{3}\{\ln(2\mu^2t)+\gamma_E\}+\frac73\right],\qquad
R_{\rm loc,MS}=(2-C_{\rm loc})R_{\rm loc,flow}.
\]

| n_z | aE ± stat | 〈x〉g ± stat | λ tail回退/1000 |
| --- | --- | --- | --- |
| 3 | .74630 ± .00382 | .26905 ± .04288 | 10 |
| 4 | .91096 ± .00995 | .32487 ± .06298 | 31 |
| 5 | 1.10166 ± .02216 | .21875 ± .06807 | 9 |

λ尾部：27个固定形状A(λ/8)^−p exp[−(λ−8)/L]，
p={.5,1,2}、L={4,8,16}、λmin={4,5,6}，λ≤12且z/a≤16，
只拟合A，逐样本质量选择/AIC，λ=7…8平滑连接。
局域与首个非局域点之间采用声明的二次短距离补全。
“λ外推”是有限数据之外的尾部建模，不是物理λ→∞极限。
LaMET用arXiv:2510.26425v2的gg核、严格一圈逆和微扰hybrid proxy；
实测Pz=0分母尚缺，quark-singlet mixing暂忽略。
refined quasi支撑[−12,12]、dx=.01、ξmax=200；|x|<.02不求值，保留NaN。
支持区间精化在x=.2… .8最大变化约.0254个统计σ，仅证明所检查数值变化较小。

CT18NLO：Q=2GeV、PDG21，xfxQ已返回xg；59成员/29对Hessian，
原生90%CL以1/1.64485362695转为68.268949%CL非对称误差。
CT18的αs(2GeV)≈.29821与格点固定.25不同，如实记录而不静默改处方。
归一化图逐bootstrap除以同样本局域〈x〉g；
CT18每个成员除以自己的∫xg dx（中心约.41769），不是使用格点分母。
y轴[−.5,3]；该图的统计带未包含模型和微扰系统误差。

### 15.4 Gluon：格式、代码和证据

统一目录data/：

| 文件 | 主要格式 |
| --- | --- |
| finite_flow_stages.npz | bare/GF (14,1001,3,25)，flow/原中心后接bootstrap/动量/z |
| flow_zero_selected.npz | 中心(3,25,2)、bootstrap(1000,3,25,2)；参数截距/斜率 |
| lightcone_and_quasi.npz | matched/quasi_hybrid (3,1001,201)，保留computed_mask |
| lambda_reconstruction.npz | tree/hybrid_proxy (3,1001,801) |

代码入口：matching/gluon_absolute_normalization_v1.py、
matching/lambda_fourier_quasi_no_z0_v1.py、
matching/refine_no_z0_quasi_support_v1.py。
整理/作图：统一目录code/build_collection.py；新输出用 --out 新目录；
既有目录仅重画用 --replot，完整命令见统一目录README。
CPU拟合/积分不要在登录节点直接运行。
既有证据：validation/chain_checks.json与collection_readback.json为PASS；
provenance/source_LaMET_validation.json保存源数值核验；
manifest明确physical_acceptance=false。本次读取报告，不宣称重新做完全数值回归。

### 15.5 Quark 20conf/100boot：三极化、三flavor结果

~~~text
quark_ratio_fit_20_ud_0p1_relaxed_20260927/
quark_ratio_fit_20_ud_0p1_relaxed_matching_20260927/
quark_ratio_fit_20_ud_0p1_relaxed_flow_extrap_20260927/
flow_zero_lambda_fourier_quasi_matching_20260927/
~~~

channel=unpolarized/helicity/transversity；flavor=u/d/u_minus_d；n_z=3,4,5。
Ratio拟合要求tsep*a>.6fm，最小tsep=8；relaxed χ²/dof<10、Q>1e−6，
无合格候选时保留有标记的最小误差替代。全覆盖不等于全部质量合格。
修正4D SFTE为h_MS=Zchi exp(−δm|z|)h_GF/Cpsi；
本20conf分支Zchi=1待验证，不是第14节10conf实测Zchi产品。
flow-zero输入.1,.2,.5,.8；线性窗口[.1,.2,.5]、[.1,.2,.5,.8]、[.2,.5,.8]，
全配对协方差、不新抽bootstrap，score=χ²+2k+2Nomitted（k=2）。
omitted项是本分析显式惩罚，不是普适AIC定理。
τ≥.2三点线性和四点二次另存敏感性；Ratio窗口误差/flow窗口spread未合入下游统计带。
H/T的z=0 odd entry先保留NaN；重构施加解析odd零，不能用于填补其他失败点。

入口flow_zero_lambda_fourier_quasi_matching_20260927/code/run_pipeline.py；
从该目录执行 bash slurm/submit_pipeline_cpu.sh 提交CPU。
重算必须先确认/指定新输出目录，不能覆盖既有results。
λ≥7的五参数投影尾部：

\[
f_R=[A_1/\lambda^a+A_2\cos(\lambda-\pi b/2)/\lambda^b]e^{-\lambda/\lambda_0},
\quad
f_I=[A_1/\lambda^a+A_2\sin(\pi b/2-\lambda)/\lambda^b]e^{-\lambda/\lambda_0}.
\]

A1/A2是投影有效振幅，不是完整复数模型两振幅。
全协方差、5%对角shrinkage/SVD；λ<7用PCHIP，tail到24，dλ=.05。
Fourier采用本分支存储约定：

\[
\widetilde q(x)=\frac1\pi\int_0^{24}
[{\rm Re}H\cos(x\lambda)-{\rm Im}H\sin(x\lambda)]d\lambda.
\]

x∈[−2,2]、401点；U仅Re，H/T仅Im。
缺失正交分量仅为single-component diagnostic置零并记录元数据，
不能确定完整q/反q或最终polarized PDF。Wick相位和tensor物理解读待确认；
transversity相对符号待外部物理验证。
quasi matching参考/public/group/lqcd/donghx/Diagram_GluonPDF/Compare_YaoFei/Matching/Matching_restore.py，
fixed NLO hybrid核：μ2GeV、αs=.296、zs=.25fm、ξmax50、Gauss48、parton动量对数。
不除以/恢复局域荷，normalization_charge=none。
GF输出与历史hybrid kernel的完整scheme闭合核验仍待完成，运行成功不替代理论证明。

结果根目录：

/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_zero_lambda_fourier_quasi_matching_20260927/results

| 内容 | 相对结果根目录 |
| --- | --- |
| λ图 | plots/lambda/lambda_extrapolation_<channel>_<flavor>.pdf |
| Fourier/matching前后 | plots/matching/matching_<channel>_<flavor>.pdf |
| LHAPDF诊断参考 | plots/lhapdf/matched_vs_lhapdf_<channel>_<flavor>.pdf |
| T u−d/JAM | plots/jam2022/matched_vs_JAM2022_transversity_u_minus_d.pdf |
| Fourier HDF5 | fourier/<channel>/<flavor>/fourier_L32x96_*.h5；quasi_pdf/samples=(401,100) |
| matched HDF5 | matching/<channel>/<flavor>/matched_L32x96_*.h5；matched_pdf/samples，x/原100样本顺序 |
| 审计 | validation_report.json、manifest.json、summary.csv |

[U λ](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_zero_lambda_fourier_quasi_matching_20260927/results/plots/lambda/lambda_extrapolation_unpolarized_u_minus_d.pdf)、
[U Fourier/matching](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_zero_lambda_fourier_quasi_matching_20260927/results/plots/matching/matching_unpolarized_u_minus_d.pdf)、
[H Fourier/matching](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_zero_lambda_fourier_quasi_matching_20260927/results/plots/matching/matching_helicity_u_minus_d.pdf)、
[T Fourier/matching](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/flow_zero_lambda_fourier_quasi_matching_20260927/results/plots/matching/matching_transversity_u_minus_d.pdf)。

报告状态PASS_WITH_TAIL_MODEL_WARNING：27+27份HDF5，每套100样本有限、求解数值成功。
u−d tail参数触边界数：

| channel | n3/100 | n4/100 | n5/100 |
| --- | --- | --- | --- |
| U | 100 | 100 | 100 |
| H | 99 | 100 | 77 |
| T | 99 | 100 | 98 |

数值success不是合格物理尾部拟合。U参考CT18NLO；
H实际NNPDFpol20_lo_as_01180，不是NLO参考集。
T使用/public/group/lqcd/donghx/Diagram_GluonPDF/Matching/JAM2022.txt，
列x/central/one_sigma_error，Q2GeV，比较q而非xq，不作tensor-charge归一化；
来源/相对符号物理对接待确认。单分量不能与完整flavor参考作精确物理比较，
图为诊断叠图；|x|<.1或|x|>.9灰区提示固定阶端点限制，
中间区域也未因此获得物理有效性保证。

### 15.6 Quark 10conf/50boot：实测Zchi与控制链仍保留

入口quark_lambda_fourier_lamet_20260920/README.md；
code/reconstruct.py、code/msbar_kernel.py；数值快照code/numerical_v2_snapshot/。

~~~text
quark_three_stages_10conf_Nboot50_20260920/
  lambda_fourier_lamet_valence_v2/{sixflow_sumdiff,fourflow_ratio}/
quark_Zchi1_control_10conf_Nboot50_20260920/
  lambda_fourier_lamet_valence_v2/{sixflow_sumdiff,fourflow_ratio}/
~~~

输入.2,.5,.8,1.0线性flow截距，二次为对照。
只有U u−d实部。在q(−x)=−qbar(x)下重构v_(u-d)=(u−ubar)−(d−dbar)，
不是完整正x的u−d；vtilde=(2/π)∫cos(xλ)Re h dλ，这个2不能丢。
本分支未将缺失虚部当物理零，不能与20conf图混称同一PDF。
9个固定形状振荡tail、6≤λ≤12、9…10平滑拼接；
短距离使用arXiv:1801.03917v2非局域MSbar局域OPE log补全，另存log2/flat1。
不能把局域荷1直接插成MSbar非局域h(0)。
直接MSbar gamma0核用该文式(68)，μ2GeV、αs=.30、严格一圈逆；
v2包含短距离log产生的大|y| 1/|y|渐近项；v1硬截断为历史对照。
不按h(0)、价夸克积分或CT18强制归一化。

各分支figures/有lambda_tail_all_Pz.pdf、quasi_full_support.pdf、
quasi_matching_before_after.pdf、matched_momenta_overlay.pdf、model_sensitivity.pdf。
valence_results.npz=(3,51,99)，动量/原中心后50bootstrap/x。
验证quark_lambda_fourier_lamet_20260920/validation/results_checks.json：
四分支有限性、样本身份和源文件不变检查通过；
解析Fourier差3.33e−16、大|y|补项独立积分差1.1e−13。
实现通过不消除：n_z=3,4所有注册tail均无Q≥.05；n_z=5有通过候选。
在x=.2… .8，短距离变化最大约18.46主统计σ，线性→二次flow最大约17.39σ，
图带不是完整系统误差。ct18nlo_comparison_v1比较u_v−d_v；
overlay_ud_no_antiquark_assumption额外假设ubar=dbar=0，不是已测sea分离。

### 15.7 flow窗口限制与a²/t项推导

τ=t_f/a²、r_F=√(8t_f)。可控层级通常需
a≪r_F≪min(|z|,1/|Pz|,1/ΛQCD)（z>0），并检查a²/t_f≪1、t_f Pz²≪1等。
常数8不能隐藏：a²/t_f=1/τ，而(a/r_F)²=1/(8τ)，数值阈值不同。
线性化横向flow为B(t,p)=exp(−tp²)A(p)；最近邻Laplace：

\[
\widehat p^2=\frac4{a^2}\sum_\mu\sin^2(ap_\mu/2)
=p^2-\frac{a^2}{12}\sum_\mu p_\mu^4+O(a^4p^6),
\quad
e^{-t\widehat p^2}=e^{-tp^2}
\left[1+\frac{a^2t}{12}\sum_\mu p_\mu^4+\cdots\right].
\]

flow紫外积分典型p²∼1/t，故相对修正a²tp⁴∼a²/t；
固定低动量mode是a²tp⁴，不能声称每个mode都为a²/t。
自由四维热核toy：

\[
I(t,a)=\int_{\rm BZ}\frac{d^4p}{(2\pi)^4}e^{-2t\widehat p^2}
=\frac1{(8\pi t)^2}\left[1+\frac{a^2}{8t}+O(a^4/t^2)\right].
\]

1/8只属于此toy，不是gluon OPE已测伪影系数；改进action/flow/observable可消去领先项，
实际还可能有a²/z²、a²Pz²。schematic Mconv=MR+tA+(a²/t)B+…，
固定a直接t→0不是连续SFTE极限；epsilon减半检查积分器，不检查格点伪影。

| τ | r_F/a | a²/t_f |
| --- | --- | --- |
| .1 | .894 | 10 |
| .2 | 1.265 | 5 |
| .5 | 2.000 | 2 |
| 1.0 | 2.828 | 1 |
| 1.4 | 3.347 | .714 |
| 1.8 | 3.795 | .556 |
| 3.8 | 5.514 | .263 |

t_f Pz²=τ(2πn_z/32)²；n_z=5在τ=1已约.964，τ=3.8约3.66。
因此τ≥1和AIC不会自动得到受控window，现有结果是探索性外推。
按统一物理标准选窗、记录z/P依赖mask，不逐点挑最小误差；
受控提取需多a固定物理t/z/P连续外推，或理论支持且数据可辨识的联合拟合。

### 15.8 最新L32x64生产与缓存约定

a=.0897fm；thin gauge：

/public/group/lqcd/configurations/CLOVER/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64

gluon_l32x64_allconf_20261009/：1700有效LIME输入、conf1000…17990、
14个与L32x96相同的gluonτ、23800请求、epsilon=.01/fine=.005、zdir2、zcount25。
初审3825824 COMPLETED/exit0，23038份有效raw导出TU/SU/TH/SH；
762请求缺失/无效、600conf组，不能称全部23800完成。
14个epsilon gate通过，最大relativeL2=2.180709988342686e−5，
最大plaquette差7.511529667048045e−6。
dispatch3825825因账户MaxSubmitJobs=200失败；
恢复3826064有60数组项，各顺序处理10conf组，无脚本%N上限；
final3826065依赖afterok。CPU分区a192c1t/cpu6248R/cpueicc/i72c512g。
本次上午仍运行/等待；完成须final_audit.json中missing_count=0和任务COMPLETED/exit0。
证据validation/initial_audit.json、epsilon_gates.json、submission.json。

结果results/conf<ID>/tauXpXXX/zdir2/unpolarized/{TU,SU}.npy与helicity/{TH,SH}.npy，
primitive=(2,4,25,64)，projection/orientation/z/t，保留完整复数和
legacy_untraced/traceless。坐标(x,y,z,t)=(0,1,2,3)，存储(t,z,y,x,mu,c,c)，
F=−i(Q−Q†)/8、εxyzt=+1；TU=Utx+Uty、SU=2Uxy、TH=Htx+Hty、SH=2Hxy。
even/odd和/差无1/2；target U=(TU−SU)even、H=(TH−SH)odd；
TH+SH、TH为诊断；旧raw helicity combined=TH+SH，须用primitive重建target。
无新增体积/Nc/g/a因子，不能用新target覆盖旧raw。

L32x64 quark入口quark_l32x64_flow_20261007/，20conf×8flow仍在进行；
数组3816972有运行/等待，已有%4限制未改。不能将生产30conf或L32x64任务数
当作20conf下游分析统计量。GPU提交用na800-sug、reservation=donghx_223；
分析/拟合到CPU计算节点。保留basis缓存，vertex/VVV每次重构而不长期保留；
清理先确认可恢复，不删结果/原输入。Note已有spin2、basis和vertex理论/代码。

### 15.9 下一步与不可擅改结论

1. 核实L32x64 final审计和quark missing清单，不按目录存在宣告完成。
2. 优先审查quark tail残差、参数边界和λmin/model扫描；保留n3/n4低Q标记。
3. 补齐正交复数分量；核验Wick相位、transversity符号与Fourier的flavor/反quark解释。
4. 在20/30conf共同统计集补测同处方Zchi，用原bootstrap配对；
   不把10conf实测值复用到别的统计集，不默认Zchi=1已验证。
5. 闭合GF输出与hybrid/MSbar kernel的scheme；gluon需κF/局域归一化、
   实测Pz=0和singlet mixing，helicity不能复用unpolarized核。
6. 扫描short-z、λtail、quasi支持/积分、flow窗口和excited states；
   配对保存变化，统计误差/模型误差分开。多格距和更高动量才可能建立受控窗口。

不能擅改：4D thin-link scheme、坐标/Clifford/ε、even/odd无1/2、
TU/SU/TH/SH、修正line-mass/sector、局域与非局域z=0区分、
bootstrap身份、误差不除√Nboot、幅度/显式归一化分支区分、
connected/singlet范围、fallback/NaN/边界标记。
未独立确认的符号、scheme闭合、绝对归一化、连续极限保持“待确认”。
Note新增“Results inventory: lambda reconstruction, Fourier and LaMET (9 October 2026)”；
位置为 [main.pdf](/public/home/donghx/Note/Gradient_Flow/main.pdf) 第14节，PDF第51–62页。
包含gluon λ/Fourier/matching和CT18图、quark三极化u−d λ/Fourier/matching和JAM图。
其余u/d图地址见本节索引。文档更新/验证工作目录为
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/docs_update_20261009_YyHRbs/，
旧文档备份original/。
文档验证记录为 validation/document_checks.json 与 validation/review.md；
保留两遍 XeLaTeX 日志、66页渲染检查和新增11张图的原文件SHA256对应关系。

生成/更新日期：2026-10-09（Asia/Shanghai）。
会话标题：gluon/quark λ尾部、Fourier与quasi matching结果及flow窗口交接更新。

## 16. Blending 权重、flow/basis/vertex 与时间窗测试（2026-10-15）

### 16.1 目标、当前决策和证据层次

本轮统一记录四维 quark flow 在 Blending 原始随机压缩基底上的实现、两端原始时间联合权重、完整积分后的有限时间支持缓存，以及已完成的 A/B 和 B 窗口对照。gluon 纯规范场 OPE 不使用这套费米子 Blending 权重；不能把本节表或数字直接加入 gluon OPE。

**当前决策：**旧生产统一二阶表记为 A，冻结保留；用户已指定后续 C3 采用 B，即“同原始时间二阶表，独立跨原始时间一阶乘积表”。B 独立实现和十组态窗口测试已完成，完整全部 current/flow/tsep 的 B 重算未提交。第15节20conf/100boot等旧分析、$Z_\chi$、matching、PDF结果未按 B 重做，不得重命名冒充 B。

数学条件推导、代码回归PASS、实际样本差异和完整随机无偏性是四件不同的事。真实输入生成器/seed、源汇共享噪声高阶矩仍标“待确认”；不能把 A 标成已证明错误，也不能把用户选择B标成无偏性证明。

### 16.2 不可混淆的符号与权重

$t_f$ 是长度平方量纲的流时间，$\tau=t_f/a^2$、$\epsilon=\Delta\tau$、$N_{\rm step}=\tau/\epsilon$；$\sigma$ 是原始basis时间，$t_J$ 是输出/插入时间。路径 `basis/tau000` 表示 $\sigma=0$，不是零流时间。完整样本身份是 $(\sigma,r,m)$。链接方向 $(x,y,z,t)$，数组坐标 $(t,z,y,x)$，basis单片 `(mode,z,y,x,color)`；位移 sign 为负/正，幅度0..20。

在固定无放回正交标架抽样模型下，扇区 $r$ 维数 $d_r$、保存 $n_r$ 列，一阶权重 $w_m=d_r/n_r$；同扇区二阶表对角 $d_r/n_r$、非对角 $d_r(d_r-1)/[n_r(n_r-1)]$，独立不同扇区为一阶乘积。

$$
\Omega^A_{(\sigma_a,m),(\sigma_b,n)}=\omega^{(2)}_{mn},\qquad
\Omega^B_{(\sigma_a,m),(\sigma_b,n)}=
\begin{cases}
\omega^{(2)}_{mn},&\sigma_a=\sigma_b\pmod {N_t},\\
w_{\sigma_a,m}w_{\sigma_b,n},&\sigma_a\ne\sigma_b\pmod {N_t}.
\end{cases}
$$

L32x64完整空间98304；400列=100exact+100compressed+200high。十个压缩子扇区每个n=10，d为50,50,100,100,200,200,350,350,550,550；high d=95704、n=200、w=478.52。源汇/VVV前200列包含随机压缩，不是全exact。L32x96扇区不同，不能照抄数字。

Flow沿固定噪声的线性推送，不重新抽样，不因范数或Gram变化改权重。basis不乘表；几何J后Hadamard乘一次；C3不得再乘，不能用普通矩阵乘或 `sqrt(diag omega2)` 分解。枚举去重的有序原始时间对；两个时间和来自两个端点，不是重复抽样，不除Nt/窗口对数、不加1/2、不按丢弃范数重归一化。

若跨时间标架独立、各随机槽一阶矩为 $P_\sigma/d$ 且中间系数不依赖噪声，则完整模和的期望只依赖表和，A/B均有 $\sum C_{mn}=d^2$，可同均值而有限样本/方差不同。完整核子系数含可能共享噪声的源汇VVV和多条线，不一定满足这个独立系数前提；实际高阶矩需另核验。

### 16.3 方程、basis 与 vertex 的计算链

$$
\partial_{t_f}B_\mu=D_\nu G_{\nu\mu},\quad
\partial_{t_f}\chi=\Delta_{B(t_f)}\chi,\quad
K=\mathcal P_f\exp\int_0^{t_f}ds\,\Delta_{B(s)}.
$$

原始 $\eta_{\sigma n}(x)=\delta_{x_4\sigma}V_n(\mathbf x;\sigma)$，拼接 $\mathcal V$ 的shape为 $(N_tN_s^3N_c,N_tN_v)$，spin单位结构省略；实际计算 $\Phi=K\mathcal V$，不是全K/全谱，也不每步投回400维。K非幺正，不能用强制归一化/Gram逆/QR只留Q改算符。

PyQUDA/QUDA按同规范轨迹的耦合RK3求解；dimensionless $L=a^2\Delta$ hopping1、diagonal−8、4D、spin对角，无质量或Dirac求逆。四模打包四spin lane，batch32为8spinor，400列13批、末批16列。多flow checkpoint接续完整驻留spinor/gauge，不能从截断缓存继续演化。thin gauge与两端/链一致，费米子APBC中心相位按原处方处理，保存gauge去除；源汇C2不流。

完整响应解包/统计全时间范数后，basis选择片以 `(400,32,32,32,3)` complex64保存；gauge complex128。固定tJ加载两个原始sigma的片，递推正负flowed空间链：

$$
J=\mathcal V_{\sigma_a}^\dagger K^\dagger\mathscr O(t_J,z;B)K\mathcal V_{\sigma_b},\qquad
\widetilde J=\Omega^B\odot J.
$$

J轴 `(sign,dz,left_mode,right_mode)=(2,21,400,400)`，spin Gamma后投影。普通perambulator/未流VVV按六Wick收缩，complex128累加；因子化 `tuSLnp,tambn->uSLpamb` 只重排求和。外态 `tsrc+tsep>=Nt` 乘−1。

### 16.4 完整积分、聚合诊断和时间窗的边界

先完整4D积分，再保存周期距离 $d_{N_t}(t_J,\sigma)\le w$ 的片。每个原始sigma有自己的输出掩码，$\Phi_{\sigma,w}=M_{\sigma,w}\Phi_\sigma$，随后拼接为$\Phi_w$；不是所有输入列共用一个输出投影。聚合丢弃比例是每conf/flow/sigma跨mode、space、color、全部输出时间的**平方范数**，不是组态平均/重采样：$f=\|\Phi-\Phi_w\|_F^2/\|\Phi\|_F^2$。阈值1e−6对应振幅范数1e−3，不是C3相对误差。几何J的绝对误差满足

$$
\|J-J_w\|_F\le\|\mathscr O\|_2
(2\|\Phi_w\|_F\|E\|_F+\|E\|_F^2),\qquad E=\Phi-\Phi_w.
$$

固定片/模、omega、perambulator和抵消后仍需数值C3收敛。norm_by_time不能重构窗口外复振幅；缺片需补完整流响应。固定B同一响应比较w4/5/6，不能把A/B和窗口变化混合。准确名称为“完整四维积分后的聚合范数诊断与时间窗C3收敛测试”，不是“只在窗口内积分”。

### 16.5 已完成的两批检查

旧 A/B 对照根 `quark_l32x64_omega_compare_20261014/`：10conf×4案例PASS；同时间块/逐Wick相同，A回归旧生产relativeL2约9.1e−15；总C3 A/B差u1.3827%–2.4865%、d1.7738%–3.3757%。该差不直接作拟合/PDF系统误差。CPU修正汇总3851153完成；原3850352环境启动失败不代表GPU数值失败。详见 `basis_vertex.md` 第16节。

后续 B 窗口测试根：

```text
/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_schemeB_window_20261015/
```

conf10000..10360步40共10；tau=.5、epsilon=.01、50步；tJ0/16/32/63、tsep7=.6279fm、dt3；Pz整数3/4/5、zdir、dz0..20、双sign、4pol、全4×4current spin、6Wick。每conf40原sigma完整积分后存52个(sigma,tJ)片，约15.24GiB/conf；特选schema不当作全生产basis。40/40 PASS、missing=[]；360个w4旧新basis切片和C2回归为0，B总C3最大4.744e−15、逐块Wick9.327e−15、物理因子化5.449e−15。冻结源hash与40NPZ header检查通过。

聚合丢弃平方范数最大：w4 2.78755e−8、w5 1.59311e−10、w6 6.75419e−13。固定B总C3复张量相对Frobenius变化40案例最大值（百分数）：

|窗口|u|d|u−d|
|---|---|---|---|
|4→5|0.0126696%|0.0146414%|0.0134355%|
|5→6|0.00080954%|0.00093124%|0.00082711%|
|4→6|0.0130625%|0.0149632%|0.0137709%|

开放spin ratio先逐conf/案例计算complex C3/Re(C2_nopol)，再conf内四案例平均，十conf算实虚SEM；无新bootstrap/物理Gamma投影/拟合。w4→6均值变化范数/原SEM范数为1.60718e−4–4.71277e−4，是整体描述量，不是逐点显著性。支持该范围当前精度的w4 B试生产，不能推广为全flow/全tsep/全通道/另一ensemble，w6也非精确全时间参考。

CPU3851836、GPU3851837_0与3851838_1–9、CPU3851839均COMPLETED/0:0；GPU30:42–32:42、汇总1:03。本轮文档操作无新物理任务。

### 16.6 代码、数据、图与可复现命令

以下相对上述 B 测试根目录：

- `README.md`、`manifest.json`：范围、门限、冻结输入；不随意修改。
- `reference/flow_common_l32x64.py`：`first_order_weights`、`create_time_pair_omega`、几何vertex；标准生成器 `reference/generate_flowed_vertex_l32x64.py`。
- `generate_selected_basis.py`、`compare_windows.py`：完整积分特选缓存及配对窗口测试；`summarize.py`：CPU汇总/绘图，不在登录节点跑。
- B标准vertex为 `vertex_schemeB`、schema `generalized_flowed_vertex_l32x64_schemeB_v1`；C3记 `flow_weight_scheme`、`flow_weight_applied_count=1`，不静默读旧A缓存。
- [主报告](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_schemeB_window_20261015/summary/report.json)、`summary/{C3_window_metrics,basis_tail_metrics}.csv`、`summary/paired_window_samples.npz`；单case `results/<conf>/tJ###/{receipt.json,window_comparison.npz}`。
- 单case C3 shape `(3,2,2,4,3,21,4,4)` 为window/flavor(u,d)/sign/pol/mom/z/m/o；signed Wick `(3,6,2,4,3,21,4,4)`，radius分解前轴7，C2 `(4,3)`；u−d汇总配对构造。
- [范数图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_schemeB_window_20261015/summary/basis_tail_diagnostics.pdf)、[C3收敛图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_schemeB_window_20261015/summary/C3_window_convergence.pdf)、[开放spin ratio窗口变化图](/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_schemeB_window_20261015/summary/paired_open_ratio_window_shift.pdf)。
- `run_B_from_cache.py`、`run_B_from_cache_na800_sug.slurm` 为已准备、未全量提交的B入口；需显式 `B_CONF`，复用**标准完整**basis/gauge而非特选测试缓存，C3单独存 `C3_schemeB/<flow-tag>/Nev_200/<conf>/`，临时vertex/VVV按作业重建清理，保留basis/gauge。

仅重做已有样本CPU汇总：

```bash
cd /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x64_schemeB_window_20261015
sbatch --parsable summarize_cpu.slurm
```

GPU固定na800-sug、reservation=donghx_223；CPU分析/拟合用a192c1t,cpu6248R,cpueicc,i72c512g，不在登录节点重算。文档本轮未改变提交策略、生产脚本或输入。

### 16.7 下一步、不可擅改项及文档位置

1. 追溯冻结随机生成器/seed/时间相关性及源汇共享噪声联合矩；不要用Gram代替矩验证。
2. 在计算节点按冻结Gamma/相位对现有配对窗口样本做unpolarized/helicity/transversity物理通道诊断。
3. 再扩展其它flow、全64个current、更多tsep；每个范围分别验收窗口、step-size、精度及因子化，不以本次PASS自动放行。
4. 若启动B完整生产与后续ratio/fit/matching，使用新根、新schema、明确conf/bootstrap配对；权重变化不需重算Dirac perambulator，但缺时间尾复振幅可能需补basis。旧A产品保留，不静默混入B。

不可擅改：thin-link共同轨迹、4D扩散/APBC、坐标与模次序、原始内积归一化、一次Hadamard权重、时间有序对无额外因子、K非幺正且不强制正交化、开放spin诊断不是物理投影、平方范数阈值不是C3误差界、有限窗口不是全时间精确参考、统计无偏性仍待确认。本次文档不宣称新bootstrap/拟合/matching已完成。

详细文档：[basis_vertex.md](/public/home/donghx/.codex/Markdown/basis_vertex.md)第7、10、16–17节；[flow_equation.md](/public/home/donghx/.codex/Markdown/flow_equation.md)第1、6–9、13节。
Note：[main.tex](/public/home/donghx/Note/Gradient_Flow/main.tex) 与 [main.pdf](/public/home/donghx/Note/Gradient_Flow/main.pdf)，新增 `Blending weights, all-time construction and temporal-window tests`，LaTeX标签 `sec:blending-oct15`，收录三张既有测试图，并修正原RK3节epsilon量纲和source-time符号说明。

更新后 Note 共72页；新增为第15节，PDF第63–68页，三张诊断图集中在第67页。两遍实际XeLaTeX均exit0、PDF已刷新；Overfull/Float too large/Missing character/undefined引用均为0。全部72页已渲染并逐页联系图检查，新节公式、表和图另放大检查；没有裁切、重叠或缺字。验收记录见下述validation目录（文档检查不替代物理验证）。

本轮原文件备份/发布稿/验证在 `/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/handoff/blending_docs_20261015/`；不包含任何凭据。

生成/更新日期：2026-10-15（Asia/Shanghai）。

本次会话标题：Blending权重分析与使用、四维basis/vertex/flow算法及聚合时间窗测试统一交接。
