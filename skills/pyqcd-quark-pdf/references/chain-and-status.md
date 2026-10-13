# 连通夸克梯度流准 PDF：链与状态

本文件保存入口不适合展开的物理公式、阶段接口和状态矩阵。所有路径是只读来源或
历史参考，不是 PyQCD 当前生产依赖。

## 1. 阶段接口

| 阶段 | 输入 | 输出 | 下一阶段最小门 |
|---|---|---|---|
| 4D flow | thin-link gauge、未流 basis | flowed gauge、`K(t)V` | 同步 RK3、APBC、规范协变、重启一致 |
| flowed vertex | `K V`、流化 Wilson line | generalized vertex | 端点同时间/同轨迹、轴和相位已证明 |
| C3/C2 | 核子 2pt/3pt、six-direction VDV | finite-flow bare ratio | 共享组态/index、mask、方向符号 |
| bare fit | `ratio_v4` | bare matrix + quality/fallback 状态 | 物理 `Tmin`、窗扫描、bootstrap 完整覆盖 |
| ringed normalization | bare matrix、独立动能真空迹 | ringed matrix | 每 ensemble/flow 的 validated `Z_chi` |
| SFTE | ringed matrix、`mu`、`alpha_s` | MSbar bilocal diagnostic | 尺度窗口、line-mass 符号、无重复减除 |
| channel matching | MSbar bilocal | matched quasi/light-cone PDF | 通道 kernel、singlet mixing、连续极限 |

## 2. 四维同步流

连续方程为

\[
\partial_t B_\mu=D_\nu G_{\nu\mu},\qquad
\partial_t\chi=\Delta[B]\chi,\qquad
B_\mu(0)=A_\mu,\quad \chi(0)=\psi .
\]

格点使用最近邻四维协变 Laplacian

\[
(a^2\Delta_U\phi)(x)=
\sum_{\mu=1}^{4}\left[
U_\mu(x)\phi(x+\hat\mu)
+U_\mu^\dagger(x-\hat\mu)\phi(x-\hat\mu)
-2\phi(x)\right].
\]

耦合 RK3 的三个 Laplacian 必须分别作用在 `U0,U1,U2` 上。不能替换成
`exp(t Delta_{B(0)})` 或 `exp(t Delta_{B(t)})`。distillation 层不形成稠密
`K(x,y)`，而是直接计算 `K(t) V`。

多 checkpoint 续算必须同时保留 flowed gauge 和 flowed spinor。只续接 gauge 或
只续接 fermion 都会改变结果。时间窗口
`|t_J-sigma| <= w` 只是保存/收缩截断；记录窗口外平方范数比例，但该比例不是
3D flow 的定义，也不自动给出全部观测量误差界。

## 3. Ringed normalization

单 flavor 四维体积平均动能定义为

\[
E(t)=\left\langle\bar\chi(t,x)
\overleftrightarrow{\slashed D}\chi(t,x)\right\rangle,
\qquad
\mathring Z_\chi(t)=
\frac{-2N_c}{(4\pi)^2t^2E(t)\big|_{m_R=0}} .
\]

ringed 场为

\[
\mathring\chi=Z_\chi^{1/2}\chi,\qquad
\mathring{\bar\chi}=\bar\chi Z_\chi^{1/2},
\]

因此双局域算符只有一个整体因子：

\[
O_\Gamma^{ringed}=Z_\chi O_\Gamma^{GF},\qquad
h_\Gamma^{ringed}=Z_\chi h_\Gamma^{GF}.
\]

不是两端各乘一次，也不是 `sqrt(Z_chi)`。固定夸克作用量和质量处方时，`u`、`d`
共用同一因子；`(u-d)` 的 ringed 结果同样只乘一次。分母必须是单 flavor 真空动能；
简并 `u-d` 动能差为零，不能作为归一化输入。

随机迹实现中，若
`D_mass q=eta`、`rho=K eta`、`phi=K q`，固定组态估计为

\[
\widehat E_U(\theta)=
-\frac{2}{V_4N_{noise}}\sum_r
\rho_r^\dagger\slashed\nabla_B\phi_r,
\qquad \theta=t/a^2.
\]

每个 bootstrap replica 先按共享 index 平均组态，再取倒数，再乘该 replica 的
bare matrix：

\[
Z_j^{(b)}=\frac{A_j}{\overline E_j^{(b)}},\qquad
h_{j\ell}^{R,(b)}=Z_j^{(b)}h_{j\ell}^{bare,(b)}.
\]

不能在逐组态上先乘 `Z`、不能独立重采样 `Z`、不能把误差独立平方相加，也不能把
50 个 bootstrap 解释成 50 个独立组态。

## 4. SFTE 与尺度窗口

使用 Brambilla-Wang 2024 修正的一圈系数：

\[
C_\psi=1-\frac{\alpha_sC_F}{4\pi}
\left[3\ln(2\mu^2t e^{\gamma_E})+2+\ln432\right]+O(\alpha_s^2),
\]

\[
\delta m(t)=-\frac{\alpha_sC_F}{4\pi}\frac{\sqrt{2\pi}}{\sqrt t}
+O(\alpha_s^2),
\]

\[
h_\Gamma^{MSbar}=C_\psi^{-1}e^{-\delta m|z|}
Z_\chi h_\Gamma^{GF}+O(t).
\]

此系数不能与 hybrid/ratio 对同一线性发散的减除重复使用。自然尺度是
`mu_t=e^{-gamma_E/2}/sqrt(2t)`，运行到共同尺度并保留尺度和阶数误差。

尺度窗口至少检查

\[
a\ll r_F=\sqrt{8t}\ll |z|,\qquad
\sqrt t P_z\ll1,\qquad
t\Lambda_{QCD}^2\ll1.
\]

还要保存 `a^2/t`、`t/z^2`、`tPz^2` 和 `t Lambda_QCD^2` 的全局 mask。固定
`a` 下直接 `t -> 0` 不是连续 SFTE 极限；多格距比较需要固定物理 `t`、`z` 和
`Pz`。

## 5. Channel matching

| 通道 | 普通 LaMET | 本工作流边界 |
|---|---|---|
| unpolarized/vector | 向量核 | 单态含断连和 quark-gluon mixing |
| helicity/axial | 轴向核 | 单态含断连和 mixing；不能复用非极化核 |
| transversity/tensor | tensor 核 | 领头 twist 无胶子 mixing；相对符号待外部核对 |

连通单味 `u`、`d` 不是完整 flavor PDF。首选连通 `u-d` 的原因是：在
`m_u=m_d` 且 isospin 对称的声明下，断连贡献相消，但该声明必须随结果保存。
isospin breaking 或完整单态任务不能沿用此简化。

## 6. 状态矩阵

| 对象 | implemented | tested | closed | physical-validated | 说明 |
|---|---|---|---|---|---|
| 4D thin-link gauge+fermion flow | yes, 参考工作流 | yes, 自由场/步长/restart/gauge 门 | no | no | PyQCD 包未实现 fermion flow |
| finite-flow connected C3/C2 | yes | yes | 仅 bare diagnostic 范围 | no | C3/C2 只消核子 overlap |
| ringed `Z_chi` | yes | yes, 10conf 固定质量/32noise | no | no | `formal_physics_acceptance=false` |
| SFTE | yes | yes, 系数与接口 | no | no | 修正系数不等于完整物理 PDF |
| quark channel matching | yes, diagnostic | 部分数值/有限性 | no | no | scheme closure、分量、尾部、窗口未闭合 |
| final connected `u-d` PDF | no | no | no | no | 仍是 exploratory diagnostic |

## 7. 不允许的升级

- 有限流 `C3/C2` 或 ringed matrix 不能直接叫 matched PDF。
- generic 胶子 matching、TMD matching 或比例 scheme 不能自动套到 quark channel。
- `Z_chi=1` 是未经本样本验证的控制假设，不是测量结果。
- 单格距、单系列动量、缺失正交复数分量或 tail 参数触边界时，图可生成不等于
  物理结果受控。
- 任何阶段失败都保留 raw/intermediate 和状态字段，不能删除失败点后重画成功覆盖率。
