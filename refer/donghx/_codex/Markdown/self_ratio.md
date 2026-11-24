# Self-ratio 方法：quark 测试与 gluon 零距离／斜率归一化

更新日期：2026-11-17。


**最新整理（2026-11-17）：** 第 24–29 节新增 L32x96/F32P30 gluon 四方案的模型、测试结果、分母风险及图表入口，全文公式统一为 LaTeX。第 18–23 节记录 quark 的三种空间归一化方案：
$R(z)/R(0)$、$R(z)/R(z-a)$ 以及 $R(z)/R(0)-R(z-a)/R(0)$。
包含两态谱推导、一阶拟合模型、联合 $C_2$ 约束、逐 replica 重建、三方案结果及图表路径。
第 18–23 节 quark 试算范围为 L32x96/F32P30 的 unpolarized connected $u-d$、$P=4,5$、同一前 100 行。
第 1–17 节为截至 2026-11-08 的历史记录，内容保留并统一数学排版；其中“目前”“后续”等表述对应历史日期。

本文保留截至 2026-11-03 的方法说明与 C32P29 历史试算；第 13 节记录
2026-11-06 完成、2026-11-07 复核的 F32P30 $T$-only HYP gluon 测试，第 14 节记录
其 ordinary ratio 方向平均顺序测试。2026-11-08 新增第 15–17 节，汇总 quark 的
connected／disconnected self-ratio 测试，并分析它们与 gluon 表现不同的原因。
不同轮次的投影、归一化与拟合窗口分别登记，不能把历史试算条件直接套到新结果上。

阅读入口：第 15 节给出 quark 的设置、数值和图表路径；第 16 节给出 quark／gluon
差异的物理与统计解释；第 17 节列出目前可用范围、后续比较标准和来源。
文中的“通过”均指相应层次的数值／统计门，不代表已经得到物理 PDF。

本文以本地讲义 `/public/home/donghx/Note/Gluon_PDF/helicity.pdf` 为主要依据：

- 第 2–3 页：普通 self-ratio 的谱分解及激发态抵消机制。
- 第 4 页：gluon helicity 的 odd-separation 算符。
- 第 6 页：先除以位移，再用零距离极限归一化的明确公式。
- 第 7–8 页：按此思路构造的插入时间分布示例。
- 第 11 页：self-ratio 与 Feynman–Hellmann 结果的示例对照。

这里将讲义中的定义、适用条件和本地实现选择分开记录。讲义没有给出本次
零距离外推的具体离散窗口、拟合协方差或质量门槛；本文的这些数值选择是
本次试算的分析约定，不宣称来自讲义，也不根据结果的好坏逐点更换。

## 1. 记号及分析范围

定义源汇距、相对源的插入时间、空间位移以及格距为

$$
T=t_{\mathrm{sep}},\qquad \tau=t_{\mathrm{insert}},\qquad z=na,\qquad n\in\mathbb Z.
$$

用未重整化矩阵元表示拟合得到的基态振幅。省略不随位移改变的运动学因子时，

$$
R(T,\tau,z)=\frac{C_3(T,\tau,z)}{C_2(T)}
\xrightarrow[T,\tau,T-\tau\ \mathrm{large}]{}h_B(z).
$$

具体核子投影、实虚部、格点张量符号和真空减除必须同时给出。
本文的 self-ratio 是相关函数／裸矩阵元分析方法，不等于 self-renormalization，
也不会自动完成重整化、算符混合扣除、匹配或物理 PDF 重构。

## 2. 普通 self-ratio 的定义

当同通道的局域基态矩阵元非零时，定义

$$
R_{\mathrm{self}}(T,\tau,z)
=\frac{C_3(T,\tau,z)}{C_3(T,\tau,0)}.
$$

两个相关函数必须具有相同的配置清单、动量、方向、自旋投影、源汇距和插入时间，
并在同一个统计重采样样本内先相除。在基态极限，

$$
R_{\mathrm{self}}(T,\tau,z)
\longrightarrow \frac{h_B(z)}{h_B(0)}.
$$

此比值给出归一化形状，而不是绝对矩阵元。若需要恢复绝对量，必须额外提供
通过检查的局域归一化锚点，并保留它与形状的统计相关性。

## 3. 谱分解与激发态抵消机制

在前向传播占主导的区间，可写为

$$
C_2(T)=\sum_n\frac{|Z_n|^2}{2E_n}e^{-E_nT},
$$

$$
C_3(T,\tau,z)
=\sum_{m,n}\frac{Z_mZ_n^*}{4E_mE_n}
O_{mn}(z)e^{-E_m(T-\tau)}e^{-E_n\tau}.
$$

需要时应按照实际边界条件补上反向传播等项；这些示意公式不授权修改现存数据的边界符号。
将基态与激发态分别记为

$$
C_2=C_2^{(0)}[1+\Pi(T)],\qquad
C_3(z)=C_3^{(0)}(z)[1+\Sigma(z,T,\tau)].
$$

普通三点／两点比值为

$$
R(T,\tau,z)
=h_B(z)\frac{1+\Sigma(z,T,\tau)}{1+\Pi(T)}
\simeq h_B(z)[1+\Sigma(z,T,\tau)-\Pi(T)].
$$

self-ratio 则为

$$
R_{\mathrm{self}}(T,\tau,z)
=\frac{h_B(z)}{h_B(0)}
\frac{1+\Sigma(z,T,\tau)}{1+\Sigma(0,T,\tau)}
$$

$$
\simeq\frac{h_B(z)}{h_B(0)}
[1+\Sigma(z,T,\tau)-\Sigma(0,T,\tau)].
$$

因此，两点函数自身的相对激发态项不再显式出现；三点函数中相似的污染也有机会相消。
但是，这不是无条件的激发态去除。示意地，

$$
\Sigma(z)-\Sigma(0)
=\sum_{(m,n)\ne(0,0)}
\left[\frac{O_{mn}(z)}{O_{00}(z)}-
\frac{O_{mn}(0)}{O_{00}(0)}\right]w_{mn}(T,\tau).
$$

只有两个相对矩阵元结构接近时，这个差才会较小。不能仅因为曲线较平或误差较小，
就跳过不同源汇距、插入时间 cut、窗口稳定性及其他提取方法的对照。

## 4. 统计上可能受益的原因与必要条件

对于相同重采样样本中的两个实随机量，线性误差传播给出

$$
\operatorname{Var}\!\left(\frac{X}{Y}\right)
\simeq\frac{\operatorname{Var}X}{\mu_Y^2}
+\frac{\mu_X^2\operatorname{Var}Y}{\mu_Y^4}
-2\frac{\mu_X\operatorname{Cov}(X,Y)}{\mu_Y^3}.
$$

当分母显著非零、且共同涨落有利于抵消时，配对比值可以减小统计误差。
分母接近零时，该展开失效，比例分布可能出现长尾甚至不稳定矩；此时不能用
一个看起来较小的样本误差来证明估计量可靠。

精确归一化点不是额外的独立测量；由同一批小位移数据拟合出的分母与分子高度相关，
归一化后小位移误差小也不意味着获得了新的原始信息。

## 5. 为什么 gluon helicity 不能直接除以局域矩阵元

本地 helicity 目标使用不带额外半因子的奇位移投影，

$$
\widetilde O_{\mathrm{odd}}(z)
=\widetilde O(+z)-\widetilde O(-z),\qquad
\widetilde h_B(-z)=-\widetilde h_B(z).
$$

所以

$$
\widetilde h_B(0)=0,\qquad
\widetilde C_{3,\mathrm{odd}}(T,\tau,0)=0.
$$

这些是投影恒等式，不是统计上可估计的非零归一化常数。
不能构造除以它们的普通 self-ratio，也不能把这个零填成很小的数避免报错。

以前试过使用 odd 投影之前的 raw-plus 局域相关函数作参照；它不是这里需要的
零距离斜率。即使它偶然非零，也不能替代下面的极限定义。

## 6. 讲义第 6 页的正确归一化：先除以位移，再取极限

定义

$$
q_B(z)=\frac{\widetilde h_B(z)}{z},\qquad z\ne0,
$$

$$
A_B=\lim_{z\to0}\frac{\widetilde h_B(z)}{z}.
$$

若该极限存在且非零，讲义中的对称化／归一化矩阵元为

$$
\boxed{
\widetilde h_{\mathrm{sym},B}(z)
=\frac{\widetilde h_B(z)/z}
{\displaystyle\lim_{z'\to0}\widetilde h_B(z')/z'}
=\frac{q_B(z)}{A_B}.}
$$

若原振幅在所讨论方案下可微，极限等于斜率，

$$
A_B=\widetilde h'_B(0).
$$

归一化后的函数是偶函数，且

$$
\widetilde h_{\mathrm{sym},B}(-z)=\widetilde h_{\mathrm{sym},B}(z),\qquad
\widetilde h_{\mathrm{sym},B}(0)=1.
$$

最后一个值是极限定义的精确归一化条件，不是对原来的零做数值除法，也不是一个拟合通过点。
原函数在原点为零，并不保证它的一阶导数非零；若斜率仍与零相容，这个新归一化同样不可用。

若需要恢复原来的奇函数形状，必须保留斜率锚点：

$$
\widetilde h_B(z)=zA_B\widetilde h_{\mathrm{sym},B}(z).
$$

所以归一化函数本身不是原来的绝对裸矩阵元。

## 7. 有限格距上如何估计零距离极限

在一个系综上，最小非零距离是格距，并没有可任意逼近零的连续格点。
本次需要的是固定格距、固定 HYP 方案下的小距离外推，不是连续极限外推。
小距离解析展开仅作为受检验的试算模型：

$$
\widetilde h_B(z)=A_Bz+B_Bz^3+\cdots,
\qquad
\frac{\widetilde h_B(z)}{z}=A_B+B_Bz^2+\cdots.
$$

用格点单位可以写成

$$
n=\frac za,\qquad
\frac{\widetilde h_B(n)}n=A_n+B_nn^2+\cdots,
\qquad A_n=aA_B.
$$

归一化后两种单位的结果完全一样：

$$
\frac{\widetilde h_B(z)/z}{A_B}
=\frac{\widetilde h_B(n)/n}{A_n}.
$$

本次在看拟合结果之前固定以下窗口与模型：

- 主窗口：正位移格点一到四；对除以格点位移的振幅作关于位移平方的相关线性拟合。
- 窗口对照：正位移格点一到三，以及一到五，使用相同函数形式。
- 不把原点零作为拟合数据，不把第一个非零点直接等同于极限。
- 对照窗口只报告敏感性，不根据每个插入点的误差或优度择优替换主窗口。

这里的物理距离约为主窗口的零点一到零点四二飞米。
HYP 半径、有限格距效应、非局域算符的短距离结构以及可能的非解析项均可能影响外推。
不能把这个有限格距多项式截距直接解释为已经控制了紫外极限的局域重整化矩阵元。
应在后续独立改变窗口、模型、格距及平滑方案来评估其稳定性。

## 8. 矩阵元级归一化与相关函数级 self-ratio 必须区分

### 8.1 先提取裸矩阵元，再按讲义归一化

先用既定平台／其他模型得到每个距离的裸矩阵元和共享 bootstrap 样本，

$$
\widetilde h_B^{(b)}(z),
$$

再对每个样本的小距离数据拟合斜率截距，并计算

$$
A_B^{(b)}\ \leftarrow\ \mathrm{fit}_{z\ne0}
\left[\frac{\widetilde h_B^{(b)}(z)}z\right],\qquad
\widetilde h_{\mathrm{sym},B}^{(b)}(z)
=\frac{\widetilde h_B^{(b)}(z)/z}{A_B^{(b)}}.
$$

这直接对应讲义中的裸矩阵元定义。不能把中心值分母当成无误差常数。

### 8.2 在每个插入时间上构造相应的三点／斜率比值

为画不同源汇距的插入时间分布，可定义

$$
K_3(T,\tau,z)=\frac{\widetilde C_3(T,\tau,z)}z,
\qquad
A_3(T,\tau)=\lim_{z\to0}K_3(T,\tau,z),
$$

$$
R_{\mathrm{self},H}^{\partial}(T,\tau,z)
=\frac{K_3(T,\tau,z)}{A_3(T,\tau)}.
$$

在完整模型和复数投影保持一致的基态极限，预期得到上面的归一化形状。
有限源汇距下，对每个插入点先归一化再做平台拟合，与先提取矩阵元再归一化并不自动等价。
两个顺序应单独保存，并用同一套样本作配对比较，不能混称为同一个估计量。

该 self-ratio 的参照来自小位移斜率，而不是 raw-plus 局域三点函数。
即使原点 odd 三点函数严格为零，斜率仍有可能可测；但也必须经过分母检查。

## 9. 本地 helicity 收缩与提前投影试算

标准 disconnected 构造在每个重采样样本和源时间上进行，

$$
C_{3,H}=\left\langle O_{H,\mathrm{odd}}C_{2,\mathrm{pol35}}\right\rangle
-\left\langle O_{H,\mathrm{odd}}\right\rangle
\left\langle C_{2,\mathrm{pol35}}\right\rangle.
$$

沿纵向轴时采用相应的自旋投影；当前仅纵向第三空间轴使用极化文件 `pol35`。
比值的归一化分母仍使用未极化两点函数：

$$
R_H=\operatorname{Im}\left[\frac{C_{3,H}}{D_{\mathrm{nopol}}}\right].
$$

完整复数流程不能在收缩之前随意丢掉实部或虚部。
此前按用户要求增加了独立的提前投影试算，

$$
G_3=\operatorname{Cov}\left(\operatorname{Re}O_{H,\mathrm{odd}},
\operatorname{Im}C_{2,\mathrm{pol35}}\right),
$$

$$
R_{H,\mathrm{early}}=\operatorname{Im}\left[\frac{iG_3}{D_{\mathrm{nopol}}}\right].
$$

这里的纯虚表示只是保留既有振幅与投影的记账约定。实数协方差本身不是再取一次虚部的对象。
这个提前投影并不与标准流程严格等价，因为

$$
\operatorname{Im}\operatorname{Cov}(O,B)
=\operatorname{Cov}(\operatorname{Re}O,\operatorname{Im}B)
+\operatorname{Cov}(\operatorname{Im}O,\operatorname{Re}B).
$$

本次斜率归一化优先延续这个已单独标记的提前投影试算；标准完整复数产物保持不变。
相关函数级的试算使用

$$
A_G(T,\tau)=\lim_{n\to0}\frac{G_3(T,\tau,n)}n,
\qquad
R_{\mathrm{self},H}^{\partial}(T,\tau,n)
=\frac{G_3(T,\tau,n)/n}{A_G(T,\tau)}.
$$

这个比值中的共同相位抵消，不再对已经形成的实比例强行取虚部。
最终自旋投影、整体相位和舍去项是否可以忽略，仍需独立验证。

## 10. 相关拟合、分母检查与统计传播

对每个小距离拟合，定义

$$
\mathbf y^{(b)}=\left(\frac{G_3^{(b)}(n_1)}{n_1},\ldots,
\frac{G_3^{(b)}(n_k)}{n_k}\right)^T,
\qquad
X_{j\alpha}=(1,n_j^2).
$$

采用共享 bootstrap 估计跨位移协方差，进行相关最小二乘：

$$
\widehat{\boldsymbol\theta}
=(X^TC^{-1}X)^{-1}X^TC^{-1}\mathbf y,
\qquad \boldsymbol\theta=(A_n,B_n)^T.
$$

本次协方差使用与原流程相同的 Schaefer–Strimmer 相关矩阵收缩和 SVD 截断。
给定由全体样本确定的协方差，保持 GLS 投影固定，对每个样本重算截距。

分母是否可用至少要同时检查：

$$
\mathrm{SNR}(A)=\frac{|A_{\mathrm{central}}|}{\operatorname{sd}_b(A^{(b)})}\ge3,
$$

$$
\Pr_b(A^{(b)}A_{\mathrm{central}}>0)\ge0.99,
$$

$$
\frac{Q_{0.01}(|A^{(b)}|)}{Q_{0.50}(|A^{(b)}|)}\ge0.05,
\qquad f_{\mathrm{finite}}\ge0.99.
$$

另要求正自由度、协方差与设计矩阵可识别，以及中心值的拟合优度门槛：

$$
Q\ge0.05,\qquad \chi^2/\mathrm{dof}\le3.
$$

分母失败时保留未经门槛屏蔽的比值作诊断，但正式有效数组为无效值，并保存原因。
不允许因为归一化曲线接近单位值就自动宣布斜率显著非零。

误差为共享样本的标准差，

$$
\sigma_{\mathrm{stat}}(F)
=\sqrt{\frac{1}{N_b-1}\sum_b(F^{(b)}-\overline F)^2},
$$

不是把它再除以重采样次数的平方根。
拟合窗口、截断模型、平台窗口、插入时间 cut 和方法差异应分开报告，不能伪装成统计误差。

## 11. 本次 C32P29 试算的固定条件与输出

保持原配置清单、种子、重采样索引、动量、方向与算符组合：

$$
N_{\mathrm{cfg}}=733,\quad N_b=3000,\quad \mathrm{seed}=1115,
\quad a=0.1052\,\mathrm{fm},\quad \mathbf k=(0,0,3),\quad aP_z=\frac{2\pi\,3}{32}.
$$

$$
T/a=7,\ldots,12,\qquad T>0.7\,\mathrm{fm}.
$$

裸矩阵元平台拟合继续固定最小源汇距、只扫描上端，并保留原登记的三个对称插入时间 cut。
平台拟合的 AIC 窗口选择与斜率的主／对照小距离窗口是两层不同的选择，不能混用。

新增程序与结果目录：

- `SelfRatio_C32P29_20261103/helicity_zlimit_self_ratio.py`
- `SelfRatio_C32P29_20261103/results/helicity_zlimit_v1/`

原完整复数结果、提前投影原结果与旧拟合不覆盖。
新结果保存斜率的 central／全部 replicas、协方差与质量诊断、配对比值、有效 mask、
主窗口和对照窗口结果、裸矩阵元级归一化、配置与索引哈希，以及代码和输出哈希。

本次 helicity 插入时间比较图的两个纵轴均固定为

$$
-0.2\le y\le0.1.
$$

横轴仍为居中的插入偏移，并仅添加微小的显示错位，不连接不同点。
归一化形状在原点趋近单位值，可能整体超出这个显示范围；因此另外保存完整归一化范围图，
不能把图上被裁掉的点解释为零或数据缺失。所有数值、完整误差和越界标记均保留。

## 12. 结论与后续判据

该方法真正改变的是归一化参照：使用奇函数除以位移后的零距离极限，而不是为零的局域值。
它有机会同时利用共同涨落与相似的激发态结构，但仍依赖可测的非零斜率、
稳定的小距离外推和正确的投影约定。

若主窗口斜率不显著，或相邻窗口给出不相容结果，应标记试算不可用／有模型依赖，
而不是任意换一个非零位移作分母。若统计和窗口检查通过，再比较归一化顺序、
平台／求和方法及不同源汇距，最后讨论独立的重整化与匹配步骤。

## 13. F32P30：$T$-only HYP gluon PDF 的 ratio／self-ratio 实测比较

### 13.1 测试范围、输入与可追溯产物

本节记录已经完成的 unpolarized／helicity 配对测试，不是四个 ensemble 的联合分析，
也不是重整化或匹配后的物理 PDF。分析目录为

`/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/SelfRatio_F32P30_20261106`。

下文用 `BASE` 指代这个目录。主要输入为：

- OPE：`/public/group/lqcd/donghx/Ope_Gluon/Result_hyp_4D_10times/F32P30/zdir`。
- 2pt：`/public/group/lqcd/donghx/2pt_Result/beta6.41_mu-0.2295_ms-0.2050_L32x96/momsmear2z`。
- 冻结输入清单：`BASE/results/input_manifest.json`；共同配置：`BASE/results/configurations.txt`。
- 原始 $C_3$／ratio 样本：`BASE/results/c3_unpolarized.npz`、`BASE/results/c3_helicity.npz`。
- 原始扫描拟合：`BASE/results/fits_unpolarized.npz`、`BASE/results/fits_helicity.npz`。
- 严格共同候选比较：`BASE/results/matched_fits_unpolarized.npz`、`BASE/results/matched_fits_helicity.npz`。
- 完成与比较回执：`BASE/results/complete.json`、`BASE/results/matched_comparison.done.json`。

2026-11-07 复核了两个 `matched_fits` 文件的 SHA256，与比较回执中的记录一致；
本节数值由这些文件读取，没有重新计算 $C_3$、重采样或重新拟合。

| 项目 | 固定设置 |
|---|---|
| Ensemble | F32P30，空间／时间大小为 $32^3\times96$，$a=0.0775\,\mathrm{fm}$ |
| 输入覆盖 | 2pt 目录中有 801 个候选配置；OPE／2pt 两通道共同完整配置为 777 个，24 个缺输入配置不使用 |
| 动量与方向 | 动量模编号 $\mathbf k=(0,0,3)$，只使用纵向第三空间轴 |
| 位移 | $n=z/a=0,\ldots,23$ |
| 算符 | 两通道都只保留 $T$，合并 30、31 两个分量；OPE 在关联前取实部 |
| 重采样 | 3000 次 configuration bootstrap，每次从共同 777 个配置中有放回抽取 777 个；seed 为 1115 |
| 源汇距初始范围 | $T/a=10,\ldots,16$ |
| 最小物理源汇距 | $T_{\min}=10a=0.775\,\mathrm{fm}>0.7\,\mathrm{fm}$ |
| 插入时间 cut | $c=2,3,4$，每个源汇距保留 $c\le\tau/a\le T/a-c$ |

这里的 3000 是 bootstrap 样本数，不是独立配置数。当前没有分箱，也没有完成
Markov 链自相关误差控制。固定物理下限的实现为

$$
\frac{T_{\min}}a
=\left\lfloor\frac{0.7\,\mathrm{fm}}{0.0775\,\mathrm{fm}}\right\rfloor+1
=10.
$$

### 13.2 算符、$C_3$ 构造及 bootstrap 中的运算顺序

将两个 $T$ 分量的和记为 $T_{U,+}$、$T_{H,+}$、$T_{H,-}$；helicity 的第二个
场强为对偶场强。当前送入关联计算的实 OPE 为

$$
O_U(n)=-2\operatorname{Re}T_{U,+}(n),\qquad
O_H(n)=\operatorname{Re}\bigl[T_{H,+}(n)-T_{H,-}(n)\bigr].
$$

unpolarized 的整体负号沿用本轮明确指定的约定；其因子 2 对应无半因子的偶组合。
helicity 使用无半因子的奇组合，局域点严格置零。正反 link 的奇偶投影以及
30、31 分量相加都不是不同动量方向的平均。

对每个 bootstrap 样本，定义源时间为 $s$ 的两点函数

$$
B_{X,c}(T,s)=C_{2,X,c}\bigl((s+T)\bmod N_t,s\bigr),
\qquad X\in\{\mathrm{nopol},\mathrm{pol35}\}.
$$

以同一组重采样权重表示配置平均 $\langle\cdot\rangle_b$，在每个样本内重新做真空扣除：

$$
\mathcal C_{3,X}^{(b)}(n,T,\tau)
=\frac1{N_t}\sum_s\left[
\left\langle O_X(s+\tau,n)B_X(T,s)\right\rangle_b
-\left\langle O_X(s+\tau,n)\right\rangle_b
 \left\langle B_X(T,s)\right\rangle_b\right].
$$

上式 $X=U$ 时选用非极化两点函数，$X=H$ 时选用复数 `pol35` 两点函数；
OPE 的时间指标模 $N_t$，不额外添加边界折叠符号。投影后的 $C_3$ 为

$$
C_{3,U}^{(b)}=\operatorname{Re}\mathcal C_{3,U}^{(b)},\qquad
C_{3,H}^{(b)}=\operatorname{Im}\mathcal C_{3,H}^{(b)}.
$$

所以当前 helicity 确实由极化 2pt 构造三点函数，并在收缩后取虚部；但普通 ratio
的分母不是极化 2pt，而是

$$
D^{(b)}(T)=\operatorname{Re}\left[
\frac1{N_t}\sum_s\left\langle B_{\mathrm{nopol}}(T,s)\right\rangle_b\right],
\qquad
R_X^{(b)}(n,T,\tau)=\frac{C_{3,X}^{(b)}(n,T,\tau)}{D^{(b)}(T)}.
$$

计算顺序是共同配置重采样、样本内计算 $C_3$ 和 $C_2$、最后做 ratio；不是逐配置先做
$\frac{C_3}{C_2}$ 后再平均。所有通道、位移、源汇距、插入时间和两种方案共享同一份索引
`BASE/results/bootstrap_indices.npz`。

本轮只读取 `Px0Py0Pz3` 与 `zdir`，没有平均不同动量方向。因此不存在“先平均方向
再做 ratio”或“先做 ratio 再平均方向”的选择；如后续增加方向，必须另行登记顺序。
对应源码为 `BASE/run.py:124`、`BASE/run.py:184`、`BASE/run.py:212`。

### 13.3 self-ratio 的定义、零点以及绝对幅度恢复

对 unpolarized，样本内定义

$$
S_U^{(b)}(n,T,\tau)
=\frac{C_{3,U}^{(b)}(n,T,\tau)}{C_{3,U}^{(b)}(0,T,\tau)}.
$$

在分母有定义的时间点，零位移是精确归一化约束：

$$
S_U^{(b)}(0,T,\tau)=1,\qquad \sigma_b\bigl(S_U(0)\bigr)=0.
$$

这不表示独立测得了一个零误差矩阵元，也不说明这个分母足够显著。
实际保存的共同候选 self-shape 拟合中，零点中心值与全部 3000 个样本都是 1，
其标准差为 0；零点不参与自由平台拟合。

对 helicity，原始奇投影满足 $C_{3,H}(0)=0$，不能直接以局域 $C_3$ 作分母。
本轮不是重新执行第 10 节的小距离多项式斜率拟合，而是用奇延拓和 natural cubic
spline 构造有限小距离参考。令 $I[C_{3,H}]$ 表示由
$n=-23,\ldots,23$ 及精确原点构造的插值函数，则

$$
\epsilon=0.1,\qquad
A_\epsilon^{(b)}(T,\tau)
=\frac{I[C_{3,H}^{(b)}](\epsilon,T,\tau)}{\epsilon},
$$

$$
S_H^{(b)}(n,T,\tau)
=\frac{C_{3,H}^{(b)}(n,T,\tau)/n}{A_\epsilon^{(b)}(T,\tau)},
\qquad n=1,\ldots,23.
$$

这里 $\epsilon=Z/a=0.1$，不是 $0.1\,\mathrm{fm}$；对应的物理距离为
$Z=0.00775\,\mathrm{fm}$，小于一个格距。它是区间内插值，没有提供新的亚格距测量，
也不等于已经控制好的严格零距离极限。插值权重固定并用于每个 bootstrap 样本；
另保存 $\epsilon=0.001$ 的敏感性参考，但本节没有据此宣称插值系统误差已受控。

用同一个插值定义，$S_H(\epsilon)=1$ 是恒等归一化；原始 helicity 零点仍为零，
self 数组的零点为未定义值，不强行填 1。有限 $\epsilon$ 下，严格零距离斜率与
$A_\epsilon$ 也不必完全相等。

self-ratio 拟合只给出形状。若需要恢复绝对裸幅度，要分别拟合相关锚点

$$
r_U(T,\tau)=\frac{C_{3,U}(0,T,\tau)}{D(T)}\ \longrightarrow\ m_0,
\qquad
r_H(T,\tau)=\frac{A_\epsilon(T,\tau)}{D(T)}\ \longrightarrow\ a_\epsilon,
$$

然后在每个共享 bootstrap 样本内相乘：

$$
h_{B,U}^{\mathrm{self},(b)}(n)=s_U^{(b)}(n)m_0^{(b)},\qquad
h_{B,H}^{\mathrm{self},(b)}(n)=n\,s_H^{(b)}(n)a_\epsilon^{(b)}.
$$

因此，纯 unpolarized self-shape 的零点是 1，而恢复裸幅度后的零点是 $m_0$，
并带有锚点误差。严格共同候选下的当前数值为

$$
m_0=1.458417\pm0.045915,
\qquad s_U(0)=1\pm0.
$$

前一个锚点未通过平台质量门，只能作诊断；这正是裸幅度比较图的橙色零点不为 1
的原因。原始全窗口扫描的锚点则为 $1.460113\pm0.046793$，不要混用两个版本。

### 13.4 两种方案的拟合 model 是否一致

一致：对每个位移分别进行单参数相关常数拟合，把选定的不同源汇距和插入时间
联合起来，而不是先独立拟合每个源汇距：

$$
R_X(n,T,\tau)=M_X(n),\qquad S_X(n,T,\tau)=s_X(n).
$$

锚点也是同类常数拟合。本轮没有两态／多态指数模型，没有 summation 拟合，
也没有在位移之间拟合共同的形状函数。unpolarized self 零点的精确约束是例外，
不是额外的自由拟合参数。

对每个固定候选，两种观测量都最小化

$$
\chi_Y^2=(\mathbf y_Y-m_Y\mathbf 1)^T C_Y^{-1}
                   (\mathbf y_Y-m_Y\mathbf 1),\qquad Y\in\{R,S,r_U,r_H\}.
$$

协方差估计都使用 Schaefer–Strimmer correlation-to-identity 收缩及登记的
SVD 阈值 $10^{-12}$。每个候选的协方差由该观测量的 bootstrap 样本估计，
所得 GLS 投影固定后用于该候选的全部样本；不是给两方案强行使用同一个协方差矩阵。
质量门包括正自由度、有限性、bootstrap 成功率不低于 $0.9$，以及

$$
Q\ge0.05,\qquad \chi^2/\mathrm{dof}\le3.
$$

固定 cut 内的候选权重使用登记的 omission-penalized AIC：

$$
\mathrm{AIC}_{i,b}=\chi^2_{i,b}+2k+2N_{\mathrm{omit},i},\qquad k=1.
$$

中心值与各 bootstrap 样本分别选择／加权候选，再按既定规则混合可用 cut；
没有合格候选时保留带标记的 fallback。两方案使用同样的规则，但各自的协方差、
拟合优度、通过的候选以及最终权重不要求相同。对应源码为
`BASE/run.py:265`、`BASE/run.py:329` 和
`/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/SelfRatio_C32P29_20261103/run.py:329`。

原始扫描允许最大源汇距到 $16a$，但 self 的参考分母会额外限制可用时间点。
`matched` 版本因此把 direct、self-shape 和锚点都限制到共同可计算的候选集合，
然后分别重新执行候选选择／加权：

| 通道 | 严格共同候选源汇距 | 对称 cut |
|---|---|---|
| unpolarized | 固定下端 $T_{\min}/a=10$，上端 $T_{\max}/a=11,12,13,14,15$ | $c=2,3,4$ |
| helicity | 只有 $T/a=10,11$，即 $0.775$ 与 $0.8525\,\mathrm{fm}$ | $c=3,4$ |

这表示候选窗口集合一致，不表示两方案在每个样本上的 AIC 权重一致。
对应实现为 `BASE/matched_comparison.py:60`。共同候选筛选不能补救失败的锚点，
也不能替代激发态系统误差检查。

### 13.5 结果：形状、绝对幅度与分母可用性要分开评价

原始时间网格 $0\le\tau\le T$ 共含 98 个物理时间点，包括尚未剔除的接触区；
下面的分母计数不是最终平台拟合的数据点数。两通道普通 ratio 的 $C_2$ 分母在
全部 7 个源汇距上通过检查。self 参考的检查沿用第 10 节的 SNR、同号比例、
近零分位数和有限样本比例门槛。

| 指标 | unpolarized | helicity |
|---|---:|---:|
| self 参考分母通过的物理时间点 | 78/98 | 19/98 |
| 参考分母最大 SNR | 31.80 | 7.45 |
| `matched` 普通 ratio 裸幅度通过点数 | 19/24 | 23/23 个非零位移 |
| `matched` self-shape 通过点数 | 23/23 个非零位移；零点为精确约束 | 23/23 个非零位移 |
| `matched` self 恢复绝对裸幅度通过点数 | 0/24 | 23/23 个非零位移 |
| `matched` 参考锚点拟合通过 | 否 | 是 |
| 两种绝对裸幅度都通过的配对点数 | 0 | 23 |

**unpolarized。** 普通 ratio 在 $n=5,\ldots,23$ 通过，小位移
$n=0,\ldots,4$ 为 fallback／诊断。self 的全部 23 个非零位移形状通过，
但局域锚点的常数平台拟合失败：原始全窗口扫描中所有可计算锚点候选的最大
$Q=0.01199065<0.05$。分母可以显著非零，与其 $\frac{C_3}{C_2}$ 能否用常数模型描述是
不同问题；前者通过不代表后者通过。由于锚点依赖失败，24 个恢复绝对幅度全部
只能用空心诊断点展示，不能宣称 self 已改善绝对裸矩阵元。

下面列出严格共同候选下的部分绝对幅度，误差都是配对 bootstrap 标准差；
右列是乘回锚点后的量，不是零点归一的 self-shape：

| 通道与位移 | 普通 ratio 裸幅度 | self 恢复裸幅度 | 使用状态 |
|---|---|---|---|
| unpolarized，$n=1$ | $1.36753\pm0.04141$ | $1.37152\pm0.04249$ | 两者均诊断 |
| unpolarized，$n=5$ | $0.55748\pm0.03493$ | $0.47284\pm0.01629$ | direct 通过；self 仅诊断 |
| unpolarized，$n=10$ | $0.09043\pm0.01227$ | $0.07759\pm0.01042$ | direct 通过；self 仅诊断 |
| helicity，$n=1$ | $-0.014308\pm0.002043$ | $-0.014317\pm0.002041$ | 两者均通过 |
| helicity，$n=5$ | $-0.034436\pm0.005216$ | $-0.034198\pm0.005254$ | 两者均通过 |
| helicity，$n=10$ | $-0.012824\pm0.003949$ | $-0.011586\pm0.003644$ | 两者均通过 |

unpolarized 的诊断值并非处处重合，例如 $n=5,10$ 的诊断配对 pull 分别为
$-2.70$、$-2.92$。由于 self 锚点未通过，这不是两个合格结果之间的正式
显著性判定，但提示不能只看橙色误差更小就忽略中心值变化与模型依赖。

**helicity。** 在共同窗口内，23 个非零位移的普通 ratio、self-shape 和恢复幅度
全部通过质量门；锚点为

$$
a_\epsilon=-0.01481185\pm0.00211025.
$$

因为两个结果来自同一批配置，比较使用逐样本差值，而不是独立误差相加：

$$
\Delta^{(b)}(n)=h_B^{\mathrm{self},(b)}(n)-M^{(b)}(n),\qquad
\sigma_\Delta(n)=\operatorname{sd}_b\bigl(\Delta^{(b)}(n)\bigr),\qquad
p(n)=\frac{h_B^{\mathrm{self}}(n)-M(n)}{\sigma_\Delta(n)}.
$$

在这 23 个通过点上，实际结果为

$$
\max_n|p(n)|=0.96635,\qquad
\operatorname{median}_n|p(n)|=0.38319,
$$

$$
\operatorname{median}_n\left[
\frac{\sigma_{\mathrm{self}}(n)}{\sigma_{\mathrm{ratio}}(n)}\right]
=0.93775.
$$

也就是当前共同窗口与常数模型下，没有发现超过一个配对标准差的逐点差异，
恢复绝对幅度的统计误差之比中位数约为 0.938，对应约 6.2% 的温和改善，
不是每个位移都改善。中心值相对差的中位数为

$$
\operatorname{median}_n\left[
\frac{|h_B^{\mathrm{self}}(n)-M(n)|}{|M(n)|}\right]=0.11562.
$$

约 11.6% 的相对差提醒我们：当前结论是“在本次统计精度下未发现显著的配对差异”，
不是证明两方法完全等价。靠近零的矩阵元会放大相对差；这里也没有进行跨位移的
全协方差整体一致性检验。另一个重要限制是 self 分母只在 19/98 个时间点通过，
不能把短窗口的结果外推为所有较长源汇距都可靠。

### 13.6 为什么同一模型仍可能给出不同结果

在每个时间点和每个样本中，只要分母有定义，就有代数恒等式

$$
R_U^{(b)}=S_U^{(b)}\frac{C_{3,U}^{(b)}(0)}{D^{(b)}},\qquad
R_H^{(b)}=nS_H^{(b)}\frac{A_\epsilon^{(b)}}{D^{(b)}}.
$$

代码在拟合前逐样本检查了这个恢复恒等式，见 `BASE/run.py:239`。因此 self-ratio
没有创造新的独立绝对幅度信息。但“先拟合形状与锚点，再相乘”和“直接拟合乘积”
一般不交换：

$$
\mathcal F[S\,r]\ne\mathcal F[S]\,\mathcal F[r],
$$

其中 $\mathcal F$ 包含相关拟合、质量筛选和候选加权。两方案的协方差权重不同，
且各自的残余激发态／接触项随 $T,\tau$ 的形状可以不同；相同的常数 model
并不保证相同的有限源汇距偏差。共同候选减少了窗口集合差异，但不能消除这些差别。

本轮观察支持的判断是：

1. unpolarized self 能给出较稳定的归一化形状，但局域锚点仍有平台模型问题；
   目前不能据此宣布绝对裸幅度更可靠。
2. helicity 在严格限定的短窗口内与 ordinary ratio 配对相容，并有温和的统计误差改善；
   尚不能把这解释成激发态污染已经消除。
3. self-shape、乘回锚点的裸幅度和普通 ratio 的裸幅度必须分开标注。
   不应直接把零点为 1 的形状与未归一化裸幅度叠加，再比较误差大小。

### 13.7 图像、质量标记与后续判据

所有图位于 `BASE/results/plots`：

- `unpolarized_bare_matrix_comparison_matched_no_pull.png/pdf`：只保留绝对裸幅度比较；
  2026-11-07 去掉下方 paired bare pull 面板，未改拟合数据、误差和质量标记。
- `helicity_bare_matrix_comparison_matched.png/pdf`：共同候选下的 helicity 绝对裸幅度及配对 pull。
- `unpolarized_normalized_shape_comparison_matched.png/pdf`、
  `helicity_normalized_shape_comparison_matched.png/pdf`：按各自参考归一后的形状比较。
- `unpolarized_insertion_comparison_focus.png/pdf`、`helicity_insertion_comparison_focus.png/pdf`：
  比较插入时间分布，左右列使用独立纵轴；不同 tsep 的误差棒不连线，并作横向微错位。
- 原始共同纵轴／固定历史范围图和 `*_insertion_all_z.pdf` 保留，完整数值见
  `BASE/results/insertion_values.csv`。被纵轴裁掉不等于零或缺数据。

横轴居中为

$$
x=\frac{t_{\mathrm{insert}}-t_{\mathrm{sep}}/2}{a},
$$

显示错位不改变实际插入时间。实心表示通过相应质量门，空心表示 fallback 或依赖失败；
灰色菱形表示精确约束，不是新的测量。删除 pull 面板只改变展示，没有删除保存的差值
或质量记录；该重绘另有 `BASE/results/matched_matrix_only_plot.done.json` 哈希回执。

下一步应先处理 unpolarized 锚点的时间依赖，保持两方案共同配置与重采样索引；
如使用两态／其他提取方法，须另行登记模型并比较可辨识性，不能逐位移挑选最有利模型。
helicity 则需要检查较大源汇距的参考显著性、插值参考与小距离斜率方案的敏感性、
绝对相位约定以及自相关影响，再扩展到其他 ensemble／动量方向。
当前全部结果仍为 HYP 裸 disconnected 矩阵元测试，不包含重整化、算符混合扣除、
LaMET 匹配或物理 PDF 重构。

## 14. F32P30 补充测试：动量方向平均与 ordinary ratio 的顺序

2026-11-07 在独立目录
`/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/F32P30_DirectionOrder_20261107`
完成测试。CPU 作业 `3945255` 为 `COMPLETED`，退出码 `0:0`。
本节比较两个 ordinary $\frac{C_3}{C_2}$ 估计量，不是重新比较 ratio 与 self-ratio。

### 14.1 定义及共同条件

取动量模编号 $(3,0,0),(0,3,0),(0,0,3)$ 对应的三个正方向等权平均，
未加入负动量。每个样本内先逐方向构造、真空扣除和投影 $C_3$，记为 $N_d^{(b)}$；
分母 $D_d^{(b)}$ 是该方向的非极化 $C_2$ 实部。两种顺序为

$$
R_A^{(b)}=\frac{\sum_dN_d^{(b)}}{\sum_dD_d^{(b)}},\qquad
R_B^{(b)}=\frac13\sum_d\frac{N_d^{(b)}}{D_d^{(b)}}.
$$

A 是先方向平均再 ratio；B 是先各方向 ratio 再平均。两个方案都在拟合前完成
方向组合，并不是先独立拟合各方向再平均；也不是把 OPE、2pt 分别平均后再相乘。

两通道仍只用 $T$，OPE 先取实；正循环旋转后 $x$ 用 $31+32$、$y$ 用 $32+30$、$z$ 用 $30+31$，
helicity 分别关联复数 `pol15`、`pol25`、`pol35` 后取虚，绝对相位仍待独立校准。
三方向共同完整配置为 777 个，与第 13 节相同，逐项复用其 3000 次 bootstrap 索引。
seed、分箱状态、算符整体符号、物理最小源汇距和单参数相关常数模型均保持不变。

两方法使用相同有效时间网格和共同可计算候选：固定 $T_{\min}/a=10$，
上端扫描 11 到 16，cuts 为 2、3、4。所有方向 $C_2$ 在全部 7 个源汇距通过分母门。
各自的协方差、拟合质量和 AIC 权重仍分别计算，不要求相同。
重算的 $+z$ 中心值和全部 bootstrap $C_3$、$C_2$ 与原单方向产物通过数值核验。

### 14.2 结果与解释

按逐样本差值 $\Delta^{(b)}=M_B^{(b)}-M_A^{(b)}$ 估计误差，
以下指标只统计两方案都通过质量门的点：

| 通道 | 共同通过的位移数 | 最大绝对配对 pull | 中心值相对差中位数 | $B/A$ 误差比中位数 |
|---|---|---|---|---|
| unpolarized | 18/24，$n=6,\ldots,23$ | $0.23852$ | $0.4120\%$ | $1.000861$ |
| helicity | 20/23 个非零位移，$n=1,2,6,\ldots,23$ | $0.52006$ | $0.2267\%$ | $1.001163$ |

在当前精度下，没有发现方向平均顺序引起显著的逐点配对差异，误差也几乎相同；
尚无实质性统计证据偏好某个顺序。这不证明两个非线性估计量完全相同，
也不构成全位移协方差的整体一致性检验。

U 的 $n=0,\ldots,5$、H 的 $n=3,4,5$ 在两方法中都仅为 fallback／诊断；
H 的局域点为奇对称零，不参加拟合。方向平均没有自动解决这些位移的时间依赖。
固定 $T/a=10,\tau/a=5$ 的检查中，$n=1,5,10$ 的三个方向 U 中心值均正、
H 均负，没有按数据改符号；这不是独立的自旋／动量反转相位校准。

单面板裸幅度图为该测试目录下的
`results/plots/unpolarized_bare_matrix_order_comparison.png/pdf` 与
`results/plots/helicity_bare_matrix_order_comparison.png/pdf`；pull 单独存图。
完整数值见 `results/fit_values.csv`，样本及候选见 `results/fits_*.npz`，
完成和哈希回执为 `results/complete.json`，详细说明为 `RESULTS.md`。

若以后把同一问题扩展到 self-ratio，必须重新声明其方向分母与绝对幅度恢复方式；
“各方向 self 的平均”乘“各方向锚点的平均”一般不等于“各方向裸 ratio 的平均”，
不能直接套用本节的 ordinary ratio 结论。

## 15. Quark self-ratio 测试汇总（2026-11-08）

### 15.1 资料边界与产物索引

本节汇总已经完成的测试，未重新计算 bootstrap 或改变既有拟合。
`/public/home/donghx/Note/Quark_PDF/Self_ratio.pdf` 的标题实际为
*Correlator analysis using Self-Ratio and Gluon helicity PDF*，报告日期为
2026-10-06。它与本文原先引用的
`/public/home/donghx/Note/Gluon_PDF/helicity.pdf` 的 SHA256 完全相同：

```text
aefd3aea96a882f68a70c2919907e243ff47610e5e949aaa9ff02abaf7cc5f7b
```

因此，讲义中的 gluon helicity 示例不能当成本地 quark 测试已经成功的证据。
下表的 quark 结论均来自实际保存的代码、报告和完成回执。

下文工作根目录为 `/public/group/lqcd/donghx/Diagram_GluonPDF`；相对路径均相对于它。

| 简称 | 产物目录 | 已完成范围 |
|---|---|---|
| Q-C | `Fit_SKILL/artifacts/ratio_self_L24x72_unpolarized_Nboot2000_Tmin0p7_20261107/` | L24x72 connected unpolarized 的 ratio／self 信号、平台拟合、绝对幅度及严格分母筛选比较 |
| Q-W | `Fit_SKILL/artifacts/ratio_self_L24x72_unpolarized_Nboot2000_Tmin0p7_20261107_window_audit/` | Q-C 的 connected $u-d$、$P=3,4,5$ 固定上端窗口诊断 |
| Q-D | `Fit_SKILL/artifacts/connected_disconnected_L24x72_same_bootstrap_N2000_20261107/` | L24x72 unpolarized connected／disconnected 同索引信号比较；断连部分未拟合 |
| Q-H | `Fit_SKILL/artifacts/helicity_disconnected_ratio_self_L24x72_N2000_20261108/` | L24x72 helicity disconnected 的 ratio／self 信号及分母诊断；未拟合 |
| Q-2000 | `Fit_SKILL/artifacts/self_ratio_0p7_nboot2000_newscheme_v1/` | L32x64、L32x96 connected unpolarized 的 self-only bootstrap；未做本轮裸矩阵元拟合 |
| G-F | `Gluon_SR/SelfRatio_F32P30_20261106/` | 第 13 节的 F32P30 HYP gluon 配对拟合，是本节跨算符比较的 gluon 基准 |

主要代码分别为 `Fit_SKILL/Ratio_Self_Comparison_2000_0p7/`、
`Fit_SKILL/Self_Ratio_0p7/`、`Fit_SKILL/Connected_Disconnected_Comparison_L24x72/`
和 `Fit_SKILL/Helicity_Disconnected_Comparison_L24x72/`。
旧的 `self_ratio_0p7_nboot1000_v1`、`self_ratio_0p7_nboot1000_chi1000_v2`
属于前期 1000 次样本测试，不能与本节 2000 次结果交叉拼接误差或掩码。
本节最新 connected 配对拟合只覆盖 unpolarized；没有据此推广 helicity／transversity
connected 的拟合结论，后者还需遵守各自的算符和符号检查。

### 15.2 新方案的估计量及共享重采样

令 $d$ 为已按约定对齐的动量方向，$\sigma=\pm$ 为 Wilson-line／VDV 位移分支，
$b$ 为 bootstrap 样本。对 connected，先源平均，再在同一份配置抽样中求均值：

$$
N_{b,d}^{\sigma}(z,T,\tau)
=\left\langle C_{3,c,d}^{\sigma}(z,T,\tau)\right\rangle_b,
\qquad B_{b,d}(T)=\left\langle C_{2,c,d}^{\mathrm{nopol}}(T)\right\rangle_b,
$$

$$
L_{b,d}(T,\tau)
=\frac{N_{b,d}^{-}(0,T,\tau)+N_{b,d}^{+}(0,T,\tau)}{2},
$$

$$
R_{b,d}^{\sigma}(z,T,\tau)=\frac{N_{b,d}^{\sigma}(z,T,\tau)}{B_{b,d}(T)},
\qquad
S_{b,d}^{\sigma}(z,T,\tau)=\frac{N_{b,d}^{\sigma}(z,T,\tau)}{L_{b,d}(T,\tau)}.
$$

$u-d$ 在上述非线性相除前由组态对齐的三点函数相减构造。
随后对复数比值做 VDV even／odd 投影、负动量 odd 符号对齐，再方向平均。
非零动量使用 $+z,+y,+x,-z,-y,-x$ 六方向，零动量只使用三个独立正方向。
最终分析分量是 `even_real` 和 `odd_imag`。
这既不是逐配置先做比值，也不是把方向平均后的 $C_3$ 再作相除。

Q-C、Q-D、Q-H 在 L24x72 使用完全相同的：

- 199 个组态，继续排除 `38800`；2000 次 iid configuration bootstrap，每次抽取 199 次，seed 为 1115。
- 保存并复用完整索引数组，而非仅使用相同 seed。索引 SHA256 为
  `8e2df984b1d396ceb58f75a21129a83f8877da3facddfec66743df9e04920de8`。
- $a=0.105\,\mathrm{fm}$，$z/a=0,\ldots,15$，$T/a=7,\ldots,12$，故最小源汇距为
  $0.735\,\mathrm{fm}>0.7\,\mathrm{fm}$。主信号比较展示 $P=3,4,5,6$。
- 原始样本的 plug-in 值作为信号中心；实／虚部误差分别取 bootstrap SD，$\mathrm{ddof}=1$，
  不再除以 $\sqrt{2000}$，不删除长尾样本来缩短误差棒。

2000 次重采样改善的是统计分布的数值估计精度，并没有增加原始 199 个组态提供的独立信息。
目前这些测试没有通过分箱扫描完成 Markov 链自相关误差控制。

分母统计门与拟合质量门分别保存。当前复数分母 $D_b$ 的诊断定义为

$$
\overline D=\frac1{N_b}\sum_bD_b,\qquad
s_D=\sqrt{\frac1{N_b}\sum_b|D_b-\overline D|^2},\qquad
\mathrm{SNR}_{\mathbb C}=\frac{|\overline D|}{s_D},
$$

$$
\mathrm{SNR}_{\mathbb C}\ge3,\qquad
\frac{q_{0.01}(|D_b|)}{\operatorname{median}_b|D_b|}\ge0.05,
\qquad D_b\ \text{全部有限}.
$$

这里的复数噪声半径是代码定义的 RMS，区别于图上各实／虚分量的样本 SD。
所有注册方向必须分别通过后才接受平均值；这些阈值是本轮分析约定，不是普遍定理。
普通 ratio 检查 $B$，self 检查 $L$；有限数值、分母通过和平台拟合通过是三种状态。

### 15.3 Connected unpolarized：两方案的绝对误差几乎相同

Q-C 的 flavor 为 connected $u,d,u-d$。两种方法都采用单参数相关常数平台拟合：

$$
R(z,T,\tau)=M_R(z),\qquad S(z,T,\tau)=s(z).
$$

固定 $T_{\min}/a=7$，扫描 $T_{\max}/a=8,9,10,11,12$，对称 cut 为 1、2、3；
每个候选至少包含两个源汇距。协方差采用 Schaefer–Strimmer correlation shrinkage，
SVD cut 为 $10^{-12}$，数值协方差保护量由该测试的 `config.json` 登记。
拟合质量门按用户设置为

$$
\chi^2/\mathrm{dof}<1000,\qquad Q>10^{-6},\qquad
f_{\mathrm{bootstrap\ success}}\ge0.9.
$$

每个 cut 内按 $\mathrm{AIC}=\chi^2+2k+2N_{\mathrm{omit}}$、$k=1$ 加权，
各 replica 重算权重，再等权混合可用 cut。
图上的误差是统计 SD；单独保存的 cut spread 未合并成总误差。
宽松质量门不能替代激发态稳定性检验。

self 给出形状，绝对幅度以配对局域锚点恢复：

$$
h_{B,\mathrm{self}}^{(b)}(z)
=s^{(b)}(z)\,\operatorname{Re}M_R^{(b)}(0).
$$

这里乘的是方向平均后拟合得到的锚点，依赖各方向共享基态矩阵元的假设。
它不是有限 $T,\tau$ 时的逐方向代数恒等恢复。

下表排除 $z=0$，跨三个 flavor、两个分量统计；“共同通过”取 Q-C 的严格分母
筛选重新拟合版本。最后一列取所有有限诊断点，不能视作严格通过子集的误差比。

| $P$ | 两者数值拟合门均通过 | 两者严格分母／拟合门均通过 | 严格子集中配对 $2\sigma$ 内 | 所有有限点 $\sigma_{\rm self,abs}/\sigma_R$ 中位数 |
|---:|---:|---:|---:|---:|
| 0 | 61 | 61 | 57 | 1.0005 |
| 3 | 89 | 89 | 71 | 0.9985 |
| 4 | 90 | 89 | 85 | 0.9860 |
| 5 | 90 | 0 | 0 | 0.9960 |
| 6 | 74 | 0 | 0 | 1.0115 |

所以，当前 connected unpolarized 的 self 方案并未带来绝对矩阵元误差的大幅下降。
$P=3,4$ 的有限点误差比接近 1；$P=5,6$ 不具备本表定义下的合格非零位移配对集合。
$P=3$ 仍有 18/89 个坐标超出配对 $2\sigma$，不能把“图很像”写成完全等价；
坐标之间相关，上述计数也不是全位移协方差检验的全局显著性。

$P=3,z/a=4$ 的 $u-d$ 绝对拟合涨落实／虚部相关系数约为 0.9869、0.9922。
它们共享三点分子、组态和索引，并不构成两个独立测量。

图表入口：

- Q-C `plots/absolute_matrix_comparison_P3_P5.pdf`：恢复绝对幅度的比较。
- Q-C `plots/screened_absolute_matrix_comparison_P3_P5.pdf`：严格分母掩码重新拟合。
- Q-C `plots/normalized_matrix_comparison_P3_P5.pdf`：相同归一化层次的形状比较。
- Q-C `plots/signals/signals_{flavor}_P{P}.pdf`：$\frac{C_3(z)}{C_3(0)}$ 与 $\frac{C_3}{C_2}$ 插入时间分布及拟合带。
  2026-11-07 的重绘将该批信号图纵轴固定为 $[-0.2,1.0]$，裁出显示范围不等于缺数据。
- Q-C `RESULTS.md`、`comparison_summary.json`、`strict_comparison/`、`delivery_manifest.json`：
  数值、状态、配对样本及验证记录。

### 15.4 推迟拟合起点：已有结果不足以宣布激发态清除

Q-W 固定 $T_{\max}/a=12$，分别取 $T_{\min}/a=7,8,9,10$、cut 为 1、2、3，
每窗独立拟合，不作 AIC 混合，以单独观察窗口变化。
$u-d,P=3,z/a=4$、实部、cut 为 3 时：

| $T_{\min}/a$ | $T_{\min}\,[\mathrm{fm}]$ | ratio 裸矩阵元 | 统计 SD | $\chi^2/\mathrm{dof}$ | $Q$ |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.735 | 0.477663 | 0.002268 | 0.927 | 0.570 |
| 8 | 0.840 | 0.471861 | 0.003769 | 0.761 | 0.790 |
| 9 | 0.945 | 0.470441 | 0.007597 | 0.599 | 0.923 |
| 10 | 1.050 | 0.477530 | 0.014074 | 0.537 | 0.936 |

这些例子都通过该窗的分母与拟合门，但 $P=3$ 的 30 个非零位移实／虚坐标中，
起点从 7 推到 8 或 9 时，分别有 5 或 8 项配对变化超过 $2\sigma$。
其统计误差中位数分别增长约 1.78、3.30 倍。这提示需要窗口检验，不能单独归因于激发态。

同一示例的 cut 为 3、$T/a=7\ldots12$ 时，GLS 按源汇距汇总的有符号权重中，
$T/a=7$ 约占 0.8559，$T/a=8$ 约占 0.1475，$T/a\ge9$ 净贡献约为 −0.0033。
这些相关拟合权重不是概率；晚源汇距入窗不等于它对基态检验有足够权重。
仅提高 $T_{\min}$ 还不够：若仍保留 cut 为 1，插入点到源或汇的最小距离仍只有
$0.105\,\mathrm{fm}$。$P=4,5$ 的这组固定长窗还受到分母失败限制。
详见 Q-W `RESULTS.md`、`fixed_window_Tmin_scan.pdf` 和 `absolute_window_shifts.csv`。

### 15.5 另外两个 ensemble：2000 次 self bootstrap 已完成，拟合尚未完成

Q-2000 只覆盖 connected unpolarized 的 $u,d,u-d$，同样先逐方向复数相除再平均。
每个 ensemble 内各 flavor 共用完整索引，前 1000 行与前期 1000 次产品逐项相同。
不同 ensemble 的组态集合和抽样索引独立，seed 相同不表示它们统计配对。

| Ensemble | $N_{\mathrm{cfg}}$ | $a\,[\mathrm{fm}]$ | 保留的 $T/a$ | 最小 $T\,[\mathrm{fm}]$ | $u,d,u-d$ 参考通过数 |
|---|---:|---:|---|---:|---|
| L32x64 | 425 | 0.0897 | 8,9,10 | 0.7176 | 139/150，137/150，135/150 |
| L32x96 | 194 | 0.0775 | 10,11,12 | 0.7750 | 138/180，135/180，132/180 |

表中分子由 `bootstrap_audit.json` 并与保存的 `reference_group_valid` 掩码核对；
分母按 `valid_tau_mask` 统计物理 $P,T,\tau$ 点，包含端点，不含公共时间轴 padding。
原审计文件的 `reference_total_count` 为矩形存储总数 165、195，各含 15 个无效 padding；
本表的物理总数为 150、180，不能将两种计数口径混为一谈，也不是有效裸矩阵元个数。
这次只读掩码核对见文档归档目录的 `reference_count_audit.json`。完整 $P=0,3,4,5,6$、$z/a=0,\ldots,20$ 的 bootstrap
与掩码已保存；本轮没有做普通 ratio 对照或裸矩阵元拟合。

### 15.6 Disconnected unpolarized：局域 vector 分母不能独立归一化

Q-D 使用配套 HYP-3D-5 次 Wilson line 和 distillation current。
light 指单个简并味，不乘 2；另有 strange。与 connected 的比对在相同配置和索引下完成。
从原始 loop $J$ 构造核子三点函数，逐 bootstrap、逐源时间重做真空扣除：

$$
C_{3,b,d}^{\mathrm{disc},\sigma}(z,T,\tau)
=-\frac1{72}\sum_s\left[
\left\langle C_{2,c,d}^{\mathrm{nopol}}(s,s+T)J_{c,d}^{\sigma}(s+\tau,z)\right\rangle_b
-\left\langle C_{2,c,d}^{\mathrm{nopol}}(s,s+T)\right\rangle_b
 \left\langle J_{c,d}^{\sigma}(s+\tau,z)\right\rangle_b\right].
$$

时间按 72 取模，原始 loop 没有预置闭合费米环负号，式中只加入一次。
不能将 loop 本身当作 $C_3$，也不能把 $C_2$ 和 loop 分别先源平均后再相乘。
随后按第 15.2 节逐方向构造 ratio 和 self。

对每个 flavor、每个动量，物理网格为
$16\sum_{T=7}^{12}(T+1)=1008$ 个 $z,T,\tau$ 点，含端点。
light 与 strange 的 standalone self 均为 **0/1008** 通过六方向局域分母门。

例如 $P=3,z/a=4,T/a=7,\tau/a=3$ 的 even real：

| 贡献 | 普通 ratio | self 原始诊断值 |
|---|---|---|
| connected $u$ | $0.968427\pm0.003912$ | $0.401052\pm0.001600$ |
| disconnected light | $-0.006187\pm0.006329$ | $-1.64448\pm4.97965$ |
| disconnected strange | $-0.003980\pm0.005859$ | $-1.67492\pm3.13047$ |

后两行 self 不通过分母门，不能据此判定其归一化形状与 connected 不同或相同。

这有物理上的根本原因：基态局域 vector charge 测量净夸克数。
海夸克与反夸克的 disconnected 净 vector charge 为零；核子 strange 净荷也为零。
有限源汇距下局域 $C_3$ 偶然非零不能把零归一化荷变成合格锚点。
在同位旋简并、相同传播子和 loop 约定下，$u-d$ 的 disconnected 相消，
它单独的 self 比值就是 $0/0$。这些结论不等于非局域 disconnected PDF 为零。
局域 vector 零荷与非局域 $z\ne0$ 结构是不同信息。
本轮未做这部分裸矩阵元拟合。

图与证据：Q-D `REPORT.md`、`plots/self_std_T7_tau3_P3to6.pdf`、
`plots/disconnected_self_reference_snr.pdf`、`plots/signals/`、`validation.json`。
保存的逐源 scalar 对照最大相对差为 $1.517\times10^{-14}$。

### 15.7 Disconnected helicity：非零平均信号不保证逐方向分母稳定

Q-H 的 loop current 为 $\gamma_5\gamma_j$，核子极化 $C_2$ 分别选
$z:\mathrm{pol35}$、$y:\mathrm{pol25}$、$x:\mathrm{pol15}$。
先用极化 $C_2$ 与 loop 构造组态协方差，再加闭环负号与本地 helicity 相位 $-i$：

$$
C_{3,b,d}^{\mathrm{disc},\sigma}
=(-i)\left[-\frac1{72}\sum_s\left(
\langle C_{2,c,d}^{\mathrm{pol}}J_{c,d}^{\sigma}\rangle_b
-\langle C_{2,c,d}^{\mathrm{pol}}\rangle_b
 \langle J_{c,d}^{\sigma}\rangle_b\right)\right].
$$

各因子的时间参数与第 15.6 节相同。普通 ratio 分母仍为 `nopol` $C_2$，
绝不能用近零的极化 $C_2$ 均值作归一化分母。
self 使用同方向、同一 $T,\tau$ 的 helicity 局域 $C_3$ 两分支平均。

在 $P=3,T/a=7,\tau/a=3,z=0$，六方向平均的 even real 为：

| flavor | 普通 ratio 的局域中心 | bootstrap SD | 逐方向局域 $C_3$ 的复数 SNR 范围 |
|---|---:|---:|---:|
| light | −0.0324653 | 0.0054201 | 0.91–2.73 |
| strange | −0.0268803 | 0.0045480 | 0.84–2.68 |

平均后的局域实部确实非零；这些有限 $T,\tau$ 信号尚未提取为基态 axial charge。
**用户观察到的非零实部与 self 分母未通过并不矛盾：当前先逐方向相除，
需要每个方向的局域分母分别稳定，不能用方向平均后较高的 SNR 代替。**

Q-D 和 Q-H 的普通 ratio 分母通过范围相同；Q-H 的 self 仍全失败：

| $P$ | ratio 通过分母门的 $T/a$ | 每 flavor ratio 通过点数 | 每 flavor helicity self 通过点数 |
|---:|---|---:|---:|
| 3 | 7–12 | 1008/1008 | 0/1008 |
| 4 | 7–9 | 432/1008 | 0/1008 |
| 5 | 7–8 | 272/1008 | 0/1008 |
| 6 | 7 | 128/1008 | 0/1008 |

普通 ratio 通过 $C_2$ 分母门只说明除数可用，不表示 disconnected 分子显著或基态已隔离。
没有把“self 全失败”解释为 axial charge 必须为零；它与 vector 零荷的物理问题不同。

$P=3,T/a=7,\tau/a=3,z/a=2$ 的 self even real 展示了长尾问题：

| flavor | 原始中心 | bootstrap SD | 中位数 | 16%–84% 分位区间 |
|---|---:|---:|---:|---|
| light | 1.0074 | 0.5626 | 0.8881 | [0.6540, 1.1553] |
| strange | 0.9895 | 0.4959 | 0.8815 | [0.6072, 1.1924] |

分位区间的视觉宽度比完整 SD 小，不能只看分位数图就认为误差可控。
$z=0$ 的 self even 为 1、odd 为 0 是相同分子分母的代数约束，不是可靠性检验。
目前未将这批 self 输入裸矩阵元拟合，也未应用 $\chi^2$、$Q$ 拟合质量门。

图与证据：

- Q-H `plots/signals_sd/disconnected_helicity_light_P3to6_ratio_self.pdf` 和
  `plots/signals_sd/disconnected_helicity_strange_P3to6_ratio_self.pdf`：原始中心及统一 SD。
- Q-H `plots/signals/`：self 中位数与分位数的附加诊断。
- Q-H `plots/local_z0_diagnostic_{light,strange}.pdf`、`z0_diagnostics.csv`：局域信号和分母 SNR。
- Q-H `REPORT.md`、`completion_manifest.json`、`independent_scalar_validation.json`：
  8 个产品的哈希、完整索引、轴、掩码、实虚 SD 和 padding 检查通过。
  六次独立逐源复算的 ratio／self 最大绝对差分别为
  $1.43\times10^{-17}$、$6.80\times10^{-16}$。

## 16. 为什么 quark 与 gluon 的 self-ratio 表现不同？

### 16.1 先比较同一层次的量：目前没有“gluon 全面更优”的证据

| 对象 | self 的参考 | 当前观察 | 不能作出的结论 |
|---|---|---|---|
| quark connected unpolarized | 同方向局域 vector $C_3$ | 恢复绝对幅度后误差比接近 1；仍有窗口漂移 | 不能仅凭小误差或相似图形宣布激发态消失 |
| quark disconnected unpolarized | 单独断连局域 vector $C_3$ | 六方向分母全失败，且基态净 vector 荷为零 | 不能把 standalone self 当成可用形状估计量 |
| quark disconnected helicity | 单独断连局域 axial $C_3$ | 允许非零；平均实部已见信号，但逐方向分母弱、比值长尾 | 不能说 axial 荷必须为零，也不能把平均 SNR 当逐方向 SNR |
| gluon unpolarized，G-F | 同通道局域 $C_3$ | 23 个非零位移 self 形状通过；锚点常数拟合失败，绝对幅度 0/24 通过 | 不能宣布已改善绝对裸矩阵元 |
| gluon helicity，G-F | 奇位移 $C_3$ 除以位移后的 spline 小距离参考 | 共同短窗内 23 个非零位移通过；恢复幅度误差比中位数 0.93775 | 不能等同于 quark 的局域 axial 归一化，不能推广到长窗 |

G-F helicity 的改善约为 6.2%，属于温和变化；共同窗口只有
$T/a=10,11$、cut 为 3、4。
其分母仅在 19/98 个含端点的物理时间点通过，最大 SNR 约 7.45。
23 个位移的最大绝对配对 pull 为 0.96635，中心值相对差中位数约为 11.6%；
这是本次精度下没有发现超过一个配对标准差的逐点差异，不能证明两方法等价。

G-F unpolarized 的局域分母在 78/98 个时间点通过、最大 SNR 约 31.80，
却仍未通过锚点平台模型。这也说明“分母显著”“形状平稳”和“绝对幅度可靠”不能混用。

### 16.2 Connected vector 局域荷接近二点函数归一化，self 可改善的空间本来就有限

这是针对 connected unpolarized 相似性的物理解释，而非本地局域流精确恒等式的证明。
考虑理想的、严格守恒、空间求和且零动量转移的 vector charge：

$$
Q_V=\int d^3x\,V_4(\mathbf x,t),\qquad [H,Q_V]=0.
$$

在同一荷扇区 $Q_V|n\rangle=q_V|n\rangle$，因而

$$
\langle m|Q_V|n\rangle=q_V\delta_{mn},\qquad
C_3^{Q_V}(T,\tau,0)=q_V C_2(T)
$$

在无端点接触、相容归一化和前向传播条件下成立。
此时即使 $C_2$ 中还有激发态，也有

$$
S(z,T,\tau)=\frac{C_3(z,T,\tau)}{q_V C_2(T)}
=\frac{R(z,T,\tau)}{q_V}.
$$

因此，self 只是重新标度了 ordinary ratio，并没有额外去除非局域分子的激发态项。
若本地 $C_3(0)/C_2$ 在使用的窗内接近常数，信号相似、恢复幅度后误差接近很自然。

当前使用局域双线性算符和有限 distillation 子空间，不是上述精确守恒流的实现；
裸局域流还涉及 $Z_V$、离散化和有限源汇距效应，所以不能把当前裸局域值直接设为
连续理论的整数荷，也不能用该理想模型代替数值检验。
但它给出了 connected vector 比值高度相关的合理机制。

反过来，gluon 的局域 $T$-only 场强双线性不是单独守恒的总能动量张量：
quark／gluon 分量之间有相互作用，重整化还涉及混合和适当的张量投影。
它没有与 $Q_V$ 同样的对角荷恒等式；其局域点可有不同的跃迁与时间依赖。
这使 gluon self 形状有机会抵消更多共同污染，也可能把问题转移到锚点。
G-F unpolarized 恰好展示了后一种限制。

### 16.3 统计收益由分子／参考的相关性与分母信号决定

对分母远离零的实量 $X=C_3(z)$、$Y=C_3(0)$，线性化给出

$$
\frac{\operatorname{Var}(X/Y)}{(\mu_X/\mu_Y)^2}
\simeq
\frac{\sigma_X^2}{\mu_X^2}
+\frac{\sigma_Y^2}{\mu_Y^2}
-2\rho_{XY}\frac{\sigma_X\sigma_Y}{\mu_X\mu_Y}.
$$

共同涨落强且符号、幅度匹配时，最后一项可抵消前两项；分母弱时，分母涨落反而主导。
复数情形应对实／虚部联合使用 Jacobian 或直接依赖共享 replica，不能机械套这个实量式。

connected quark 与 disconnected quark／gluon 的 Wick 收缩结构不同。
connected $C_3$ 与 $C_2$ 共享价夸克传播结构；disconnected $C_3$ 是真空扣除后的协方差，
局域与非局域 loop 的共同涨落、真空项和残差信号可能完全不同。
因此同一 algebraic ratio 不保证相同的降噪效果。
目前已有的分母 SNR／长尾证据足以解释部分差别；尚未用一套统一诊断量测出
全部 quark／gluon 通道的分子－参考相关矩阵，不能断言“gluon 的相关性一定更强”。

quark disconnected vector 的零荷是结构性障碍，增加统计量不能得到非零净荷锚点。
quark disconnected axial 允许非零，则首先是统计和基态隔离问题；增加独立配置、
改进源平均／估计量可测试其可行性。仅把 bootstrap 数从 2000 增大不能提高原始 SNR。

### 16.4 Quark 与 gluon helicity 的零距离算符不同

quark helicity 的局域极限是 axial current，核子轴荷及其 disconnected 部分允许非零。
当前 gluon helicity 使用奇位移投影，则

$$
C_{3,H}^g(-z)=-C_{3,H}^g(z),\qquad C_{3,H}^g(0)=0.
$$

两者不能使用相同的局域分母。
G-F 实际归一化为

$$
A_\epsilon^g(T,\tau)
=\frac{I[C_{3,H}^g](\epsilon,T,\tau)}{\epsilon},\qquad
S_H^g(n,T,\tau)=\frac{C_{3,H}^g(n,T,\tau)/n}{A_\epsilon^g(T,\tau)},
\qquad n=z/a,\quad \epsilon=0.1.
$$

$I$ 是带 odd 延拓、精确原点的 natural cubic spline。
这里 $\epsilon$ 是格点位移单位；它不是额外获得的亚格距数据，也不证明
短距离导数极限已受控。分母是多个小位移数据的固定线性组合，与分子有构造上的相关性。
因此 gluon 的较小形状误差可能包含参考构造的作用，必须连同该参考的拟合和误差恢复评价。
这并不能直接迁移为 quark disconnected axial 的局域 $C_3$ self 方案。

### 16.5 激发态抵消需要位移相关的跃迁结构相近

写出共同前因子已消去后的三点谱式：

$$
C_3(z,T,\tau)=\mathcal K(T)\left[
M_z+A_z e^{-\Delta E\tau}
+B_z e^{-\Delta E(T-\tau)}+D_z e^{-\Delta E T}+\cdots\right].
$$

当 $M_0\ne0$ 且污染较小时，self 的一阶展开为

$$
S(z,T,\tau)=\frac{M_z}{M_0}
+\frac{A_z-(M_z/M_0)A_0}{M_0}e^{-\Delta E\tau}
+\frac{B_z-(M_z/M_0)B_0}{M_0}e^{-\Delta E(T-\tau)}
+\frac{D_z-(M_z/M_0)D_0}{M_0}e^{-\Delta E T}+\cdots.
$$

两点函数激发态因子不再显式出现；是否比 ordinary ratio 更平，取决于这些差值是否小。
quark vector、quark axial 和 gluon 的跃迁矩阵元受不同对称性、投影与收缩结构约束，
没有理由要求污染系数相同。分母接近零时，该展开也不可用。

在第 16.2 节的理想守恒 charge 情形，局域三点函数跟随完整 $C_2$，
局域参考中仍含相应的对角激发态权重；不能错误地令其全部污染系数为零。
此时 ordinary ratio 与 self 恢复后相近，恰好不能证明非局域激发态已消失。
Q-W 的窗口漂移与早源汇距权重占优给出了实际需要继续检查的证据。

### 16.6 形状的降噪不能直接算作绝对幅度精度收益

在同方向、同 $T,\tau$、同 replica 且分母存在时，有

$$
R_{b,d}(z)=S_{b,d}(z)r_{0,b,d},\qquad
r_{0,b,d}=\frac{C_{3,b,d}(0)}{C_{2,b,d}}.
$$

但形状和锚点分别经过平台拟合、选择与加权后，一般有

$$
\mathcal F[S\,r_0]\ne\mathcal F[S]\,\mathcal F[r_0].
$$

方向平均也不与乘积交换：

$$
\overline{S_d r_d}-\overline S\,\overline r
=\frac1{N_d}\sum_d(S_d-\overline S)(r_d-\overline r).
$$

恢复幅度必须保留同索引的统计关联。实量近似为

$$
\operatorname{Var}(s\,m_0)\simeq
m_0^2\operatorname{Var}(s)+s^2\operatorname{Var}(m_0)
+2s m_0\operatorname{Cov}(s,m_0).
$$

只比较 self-shape 的误差或把锚点当无误差常数，会漏掉整体归一化不确定度；
简单把两误差独立相加又会漏掉有利或不利的协方差。
本地比较已使用逐 replica 乘回和配对差；quark 误差接近、gluon helicity 仅温和改善，
应按恢复后的同一绝对量理解。

### 16.7 现有跨 quark／gluon 比较还有哪些不同条件？

| 条件 | 本节 L24x72 quark | G-F gluon |
|---|---|---|
| 格距／ensemble | $a=0.105\,\mathrm{fm}$，199 个共同组态 | $a=0.0775\,\mathrm{fm}$，777 个共同组态 |
| Bootstrap | 2000 次 | 3000 次 |
| 非零动量方向 | 六方向先相除再平均 | $+z$ 单方向，$P=3$ |
| 最小源汇距 | $0.735\,\mathrm{fm}$ | $0.775\,\mathrm{fm}$ |
| 平滑／算符 | HYP-3D-5 次 link、quark 双线性 | HYP-4D-10 次 OPE、$T$-only 场强双线性 |
| 复数投影 | 逐方向复数相除后取目标 even／odd 分量 | 本轮 OPE 提前取实；helicity $C_3$ 收缩后取虚 |
| 平台质量门 | $Q>10^{-6}$、$\chi^2/\mathrm{dof}<1000$ | $Q\ge0.05$、$\chi^2/\mathrm{dof}\le3$ |
| self 分母判据 | 复数 SNR 与模长低分位数；六方向全通过 | 实参考 SNR、同号比例、近零分位数与有限率；单方向 |
| 绝对相位状态 | 沿用已核对的 quark gamma／投影约定 | helicity legacy phase 仍待独立反转校准 |

所以合格点数的多寡不能直接归因于“quark／gluon 的物理本性”。
特别是 quark 的六方向联合门更容易被某个弱方向限制，gluon 的拟合门又更严格。
本节的 gluon 基准是 G-F 的 HYP 数据，不是
`Gradient_flow_gluon/results_v2` 的 thin-link finite-flow 产品。
后者有不同投影和统计契约，未在本节建立 self 对照，不能混用其有效点数或重整化常数。

## 17. 目前结论与下一次比较的判据

### 17.1 目前应采用的判断

1. **Connected quark unpolarized：**self 可保留为归一化形状和方法对照；目前绝对幅度精度
   没有明显改善。优先使用分母通过的 $P=3,4$ 结果做统一窗口稳定性检验。
2. **Disconnected quark unpolarized：**不用单独的零局域 vector 荷作 self 分母。
   $u,d,u+d$ 可在同一 replica 中先合成总 $C_3$ 再用非零总局域荷归一；strange 净荷仍为零，
   应保留普通 ratio 的绝对归一化或另行声明公共非零参考。
3. **Disconnected quark helicity：**非零轴荷在物理上允许，当前逐方向分母统计不足。
   现有 self 只作诊断；普通 ratio 在 $C_2$ 分母通过的窗内可继续做分子信号与基态分析。
4. **Gluon：**unpolarized 的形状改善仍受失败锚点限制；helicity 的成功范围只是登记的
   小距离参考和共同短窗，恢复绝对幅度的误差改善约 6%，尚无全面优越或清除激发态的结论。

若形成总 quark self，应在同一方向、同一 replica 中执行

$$
S_{\mathrm{tot}}(z)
=\frac{C_3^{\mathrm{conn}}(z)+C_3^{\mathrm{disc}}(z)}
       {C_3^{\mathrm{conn}}(0)+C_3^{\mathrm{disc}}(0)}.
$$

不能将两个各自归一后的 self 直接相加。也可定义公共非零参考下的断连贡献，
但须另命名估计量，不能再称为 standalone disconnected self。

### 17.2 下一次统一检验的实际顺序

以下是尚未执行的后续方案，不计入本次完成的数值测试：

1. 在同一 ensemble、同一配置子集及方向集合下，冻结一套完整索引与共同物理窗口。
   分别比较单方向和方向平均，避免六方向门与单方向门的差异掩盖原因。
2. 对各通道记录 $C_2$、局域／斜率 $C_3$ 参考的 SNR、近零比例，以及分子－参考的
   实／虚联合协方差。对弱分母报告完整 SD 与分位数，不筛去极端 replica。
3. 比较两种方法时同时保留三个层次：原始 signal、相同定义的 normalized shape、
   同索引乘回合格锚点后的 absolute bare amplitude。先分别验证分母和锚点。
4. 对 quark 固定上端、推迟下端并同时增大物理端点 cut；若改用多态模型，先提供相容的
   $C_2$ 能隙和可辨识检查。以配对差、窗口漂移和统计检验能力共同评价激发态控制。
5. 不按逐点误差最小来选择方法。bootstrap SD、cut／window spread、参考构造敏感性和
   自相关影响应分开报告；过宽松拟合门的有效点数不能替代这些检查。

任何 $C_3(z)/C_3(0)$ 都只去掉共同归一化，零距离分母不含同长度 Wilson line，
所以它不会自动消除 $z$-依赖的线性发散。
本节的 self 不是完整的 reduced-ITD 双比值、hybrid renormalization、self-renormalization
或物理 PDF；后续重整化、混合与 matching 仍需独立完成。

### 17.3 来源与本次记录核对

本次只汇总既有结果并补充推导。主要依据如下：

- 第 15.1 节两份相同讲义：一般谱分解与 gluon helicity 的零距离／斜率归一化思路。
- Q-C `RESULTS.md`、`comparison_summary.json`、`delivery_manifest.json` 和 `config.json`
  对应源码：connected 的设置、严格筛选统计、误差和配对比较。
- Q-W `RESULTS.md`、`summary.json`：推迟下端窗口与 GLS 权重诊断。
- Q-2000 `README.md`、`bootstrap_audit.json`：两个 ensemble 的 2000 次 self-only 产品状态。
- Q-D `REPORT.md`、`validation.json`：unpolarized disconnected 构造和局域分母问题。
- Q-H `REPORT.md`、`completion_manifest.json`、`independent_scalar_validation.json`：
  helicity 逐方向分母、长尾和数值实现验证。
- G-F `config.json`、`results/complete.json`、`results/matched_comparison.done.json`：
  gluon 的实际定义、共同候选和比较结果，不能以原始全窗结果替代 `matched` 结果。
- C. Alexandrou 等，*Flavor decomposition of the nucleon unpolarized, helicity and transversity
  parton distribution functions from lattice QCD simulations*，arXiv:2106.16065v1。
  本地全文：Q-D `references/arXiv_2106.16065v1.pdf`；可检索文本为同名 `.txt`。
  式 (20) 支持 quark loop 闭环负号，第 VI 节及式 (37) 分别讨论 disconnected vector
  零荷、有限源汇距污染和允许非零的 disconnected axial 荷；文献数值未替代本地数值。

2026-11-08 本次记录核对重新验证了 G-F 两个 `matched_fits` 文件与比较回执的 SHA256，
以及 Q-H 报告与完成回执的 SHA256，均一致。
第 16.2 节的守恒荷论证、第 16.3 节误差传播、第 16.5 节谱展开是解释机制的推导；
未声称已经由本地数据拟合验证全部跃迁参数或跨通道噪声模型。
本次文档草稿、更新前副本、来源指纹与写入核对记录保存在
`Fit_SKILL/Self_Ratio_Notes_20261108/`。

---

## 18. 三种 quark self-ratio 方案：统一谱分解与理论推导（2026-11-17）

本节及第 19–23 节整理截至 2026-11-17 已完成的三种试算，不重新执行拟合。第 1–17 节保留历史原文，其结论和后续建议应按各自日期、数据和窗口理解。本轮主比较是 **L32x96/F32P30、unpolarized connected $u-d$、$P=4,5$、同一批前 100 个既有 bootstrap replicas、联合 $C_2$**；不应将此范围推广为三个 ensemble、全部极化或全部 2000 个 replicas 的结论。

### 18.1 记号、能隙及模型前提

为区分空间长度和格点编号，以下统一使用

$$
z_n=na,\qquad R_n(T,\tau)=\frac{C_3(z_n;T,\tau)}{C_2(T)},\qquad
M_n=\lim_{\tau,\,T-\tau\to\infty}R_n(T,\tau),\qquad H_n=\frac{M_n}{M_0}.
$$

$M_n$ 是当前运动学及投影约定下未重整化的基态振幅；$H_n$ 仅为用局域矩阵元归一化的形状。本文图和文件中写作 `z`、`z_a` 的整数即 $n$，不是以 fm 为单位的长度。$T,\tau,t$ 在本节模型中均为格点单位，定义

$$
\delta=\frac{a\,\Delta E}{\hbar c},\qquad
F_\delta(T,\tau)=e^{-\delta\tau}+e^{-\delta(T-\tau)},\qquad
G_\delta(T)=e^{-\delta T}.
$$

忽略后向传播、接触项，并截断至第一激发态时，可将谱分解写为

$$
C_2(t)=A_0e^{-E_0t}\bigl(1+r e^{-\delta t}\bigr),\qquad r>0,
$$

$$
C_3(z_n;T,\tau)=A_0e^{-E_0T}
\left[M_n+B_n^{\rm src}e^{-\delta\tau}
+B_n^{\rm sink}e^{-\delta(T-\tau)}+C_n e^{-\delta T}\right].
$$

本轮按用户指定采用两个跃迁振幅相同的假设，$B_n^{\rm src}=B_n^{\rm sink}=B_n$，故

$$
R_n(T,\tau)=\frac{M_n+B_nF_\delta+C_nG_\delta}{1+rG_\delta}.
\tag{18.1}
$$

$M_n,B_n,C_n$ 可为复数；$A_0,E_0,r,\delta$ 是实参数，$C_n$ 表示激发态对角项系数。跃迁振幅相等是本轮的模型假设，不是任意非局域算符仅凭厄米性便自动满足的恒等式。

物理上，同一动量、同一源汇谱的真实能量及能隙不随算符长度改变；$n$ 改变的是矩阵元及跃迁系数。当前让每个 $n$ 或相邻边单独拟合 $\delta_n$，是在研究两态截断下的**有效能隙与可辨识性**。这些不同的拟合能隙不能直接解释为物理谱随 $z$ 改变。若相邻边的真实污染包含多个指数，单能隙模型也是有效近似。

以下“一阶”均指对分母中的谱混合量展开。若分母长度为 $d$，定义

$$
\epsilon_d=\frac{B_dF_\delta+C_dG_\delta}{M_d},\qquad
\frac{1}{1+\epsilon_d}=1-\epsilon_d+\mathcal O(\epsilon_d^2).
$$

拟合式省略激发混合量之间的乘积，包括 $F_\delta^2,F_\delta G_\delta,G_\delta^2$ 项。它不是完整两态有理式，也不等于按某一个指数的幂作严格的高阶截断；例如 $F_\delta^2$ 本身就包含 $e^{-\delta T}$ 项。因此当前系数与谱系数的关系应按此近似理解。

### 18.2 方案一：直接 self ratio，$S_n=R_n/R_0$

在同一谱与相同观测位置下，式 (18.1) 的 $C_2$ 分母抵消，两态截断内的精确表达式为

$$
S_n(T,\tau)=\frac{R_n}{R_0}
=\frac{M_n+B_nF_\delta+C_nG_\delta}
{M_0+B_0F_\delta+C_0G_\delta}.
\tag{18.2}
$$

令 $s_{0,n}=M_n/M_0$。在分母相对污染足够小、且高阶乘积可忽略时，对该有理式作一阶展开：

$$
S_n\simeq s_{0,n}+s_{1,n}F_\delta+s_{2,n}G_\delta,
\tag{18.3}
$$

$$
s_{1,n}=\frac{B_n-s_{0,n}B_0}{M_0},\qquad
s_{2,n}=\frac{C_n-s_{0,n}C_0}{M_0}.
\tag{18.4}
$$

因此激发态抵消依赖于分子、分母的**相对**污染接近，而不只依赖于它们共享能谱。在相应矩阵元非零时，

$$
\frac{s_{1,n}}{s_{0,n}}=\frac{B_n}{M_n}-\frac{B_0}{M_0},\qquad
\frac{s_{2,n}}{s_{0,n}}=\frac{C_n}{M_n}-\frac{C_0}{M_0}.
$$

本轮真正拟合的是式 (18.3) 的实部、虚部，二者共享同一 $\delta_n$，每部分各有三个独立实系数。$S_0=1+0i$ 是恒等约束，不能当作有随机误差的信号拟合，也不能从中提取 self-ratio 能隙。用同一 bootstrap 行的局域荷恢复绝对裸量：

$$
M_n^{(b)}=M_0^{(b)}s_{0,n}^{(b)}.
\tag{18.5}
$$

该方案不需要经过上游空间点递归；但全部长度共享有噪声的局域分母。去除 $C_2$ 的显式分母不等于去除所有激发态。

### 18.3 方案二：adjacent self ratio，$Q_n=R_n/R_{n-1}$

$$
Q_n(T,\tau)=\frac{R_n}{R_{n-1}}
=\frac{M_n+B_nF_\delta+C_nG_\delta}
{M_{n-1}+B_{n-1}F_\delta+C_{n-1}G_\delta},\qquad n\geq1.
\tag{18.6}
$$

这是完整复数比值。若 $R_n=x_n+i y_n$、$R_{n-1}=x_-+i y_-$，则

$$
\operatorname{Re}Q_n=\frac{x_nx_-+y_ny_-}{x_-^2+y_-^2},\qquad
\operatorname{Im}Q_n=\frac{y_nx_--x_ny_-}{x_-^2+y_-^2}.
\tag{18.7}
$$

不能以“实部除实部、虚部除虚部”代替。分母实部过零也不单独意味着复数分母为零，需要检查完整模长和二维涨落。

在 $M_{n-1}\neq0$ 且分母激发态修正相对较小时，

$$
Q_n\simeq q_{0,n}+q_{1,n}F_\delta+q_{2,n}G_\delta,
\tag{18.8}
$$

$$
q_{0,n}=\frac{M_n}{M_{n-1}},\qquad
q_{1,n}=\frac{B_n-q_{0,n}B_{n-1}}{M_{n-1}},\qquad
q_{2,n}=\frac{C_n-q_{0,n}C_{n-1}}{M_{n-1}}.
\tag{18.9}
$$

相邻长度的相对污染若变化缓慢，$q_{1,n},q_{2,n}$ 可能较小。代价是分母从信号较强的局域点变成非局域点；在长距离模长接近噪声时，比值会产生长尾，一阶展开本身也可能失效。

每个边拟合得到 $q_{0,n}$ 后，通过**同一 replica 内的复数连乘**恢复形状及裸量：

$$
H_n^{(b)}=\prod_{j=1}^{n}q_{0,j}^{(b)},\qquad
M_n^{(b)}=M_0^{(b)}\prod_{j=1}^{n}q_{0,j}^{(b)}.
\tag{18.10}
$$

若 $H_{n-1}=u+iv$、$q_{0,n}=p+iq$，则递推的实、虚部分别为 $up-vq$、$uq+vp$，不能分别连乘实部或虚部。

原始观测量满足精确的望远镜恒等式

$$
\prod_{j=1}^{n}Q_j(T,\tau)=S_n(T,\tau).
\tag{18.11}
$$

该恒等式适用于同一观测点、同一 replica，并要求分母可定义。独立截断拟合后各边的有效能隙、模型偏差和协方差正则化不同，因而拟合的连乘结果不必等于直接拟合 $S_n$ 的结果。

### 18.4 方案三：self-ratio difference，$D_n=S_n-S_{n-1}$

$$
D_n(T,\tau)=\frac{R_n}{R_0}-\frac{R_{n-1}}{R_0}
=\frac{R_n-R_{n-1}}{R_0},\qquad n\geq1.
\tag{18.12}
$$

这是沿 Wilson-line 长度的空间差分，未除以格距；不是对源汇距求差，也不是 summed-ratio difference。若需要有限差分导数，应使用 $D_n/a$，相应拟合系数除以 $a$，重建时乘回 $a$。

令 $\Delta M_n=M_n-M_{n-1}$，对 $B_n,C_n$ 同样定义相邻差。共同谱下的精确两态有理式为

$$
D_n=\frac{\Delta M_n+\Delta B_nF_\delta+\Delta C_nG_\delta}
{M_0+B_0F_\delta+C_0G_\delta}.
\tag{18.13}
$$

对局域分母作同样的一阶展开，得到当前使用的拟合模型

$$
D_n\simeq d_{0,n}+d_{1,n}F_\delta+d_{2,n}G_\delta,
\tag{18.14}
$$

$$
d_{0,n}=\frac{\Delta M_n}{M_0},\qquad
d_{1,n}=\frac{\Delta B_n-d_{0,n}B_0}{M_0},\qquad
d_{2,n}=\frac{\Delta C_n-d_{0,n}C_0}{M_0}.
\tag{18.15}
$$

此方案不除以有噪声的相邻非局域点，规避 adjacent 方案的新分母风险；但拟合的是小的增量，其相对误差可能很大，且恢复长距离结果时须累计所有上游边：

$$
H_n^{(b)}=1+\sum_{j=1}^{n}d_{0,j}^{(b)},\qquad
M_n^{(b)}=M_0^{(b)}\left(1+\sum_{j=1}^{n}d_{0,j}^{(b)}\right).
\tag{18.16}
$$

原始数据满足

$$
1+\sum_{j=1}^{n}D_j(T,\tau)=S_n(T,\tau).
\tag{18.17}
$$

在 $n=1$ 时，$Q_1=S_1$、$D_1=S_1-1$。它们应给出相同的能隙和恢复后的形状，差分仅平移实部常数；这提供了比“曲线看起来相似”更严格的实现检查。

### 18.5 论文公式与本次扩展的关系

本地已保存的 arXiv:2610.39944v1（*Enhanced Signal and Excited-State Control in Nonlocal Lattice Correlators*，Shi-Yi Zhong、Jian-Hui Zhang、Dian-Jun Zhao，2026-10-29）中，Eq. (3) 给出 $C_3$ 谱分解，Eq. (5) 定义 $C_3(z)/C_3(0)$，Eq. (6) 给出归一化基态矩阵元，Eq. (9) 为谱混合项的精确比值，Eq. (10) 为一阶展开，Eq. (11) 给出相对跃迁矩阵元之差。

式 (18.3) 是该一阶展开思路在本地对称两态假设下的具体化，不能把这个三项式直接标为论文原有某一个公式。方案二的相邻长度递归、方案三的空间差分及其实际质量检查是本次自行推导并实现的测试，不宣称论文已经验证它们。此处按已保存全文核对公式编号，未进行新的网络文献检索。

## 19. Bootstrap、方向平均和误差传播

### 19.1 本轮实际估计量及操作顺序

输入估计量为 `post_ratio_six_direction_vdv_average_v1`：先在每个方向形成 $C_3/C_2$，再按既有规则作六方向／VDV 平均及偶实奇虚投影。三种空间变换均在这个既有平均量上、逐原始 bootstrap 行构造：

$$
S_n^{(b)}=\frac{R_n^{(b)}}{\operatorname{Re}R_0^{(b)}},\qquad
Q_n^{(b)}=\frac{R_n^{(b)}}{R_{n-1}^{(b)}},\qquad
D_n^{(b)}=S_n^{(b)}-S_{n-1}^{(b)}.
$$

$R_0$ 的虚部是投影后的对称零，仅有舍入量级残差；$S_0$ 固定为 $1+0i$。此处理不等于“每方向先构造 $C_3(z)/C_3(0)$，再方向平均”。单方向共同 $C_2$ 可严格约去，但先分别形成 ratio 再平均后的量之比，一般不能重新等同为先平均 $C_3$ 之比。第 18 节的共同谱推导解释拟合动机，在平均数据上使用单谱模型还包含已有分析的有效模型近似。

配置共 194 个，父产品已有 2000 行 bootstrap，种子 1115；本轮严格取原行号 0–99。三个方案及局域荷使用同一配置顺序、bootstrap 索引和原行号。没有重新抽样，没有对拟合结果进行第二层 bootstrap。中心值是对变换后的同一 100 行信号取均值再拟合；不是先对信号取均值再作非线性除法。当前协方差也由该 100 行构造。

### 19.2 必须保留跨长度相关性

对每个实分量，原始空间差分的方差为

$$
\operatorname{Var}(D_n)=\operatorname{Var}(S_n)+\operatorname{Var}(S_{n-1})
-2\operatorname{Cov}(S_n,S_{n-1}).
\tag{19.1}
$$

正相关会减小差分噪声，但并不保证拟合后重建的矩阵元误差更小。差分拟合系数累加后的方差是

$$
\operatorname{Var}(H_n)=\sum_{j=1}^{n}\sum_{k=1}^{n}
\operatorname{Cov}(d_{0,j},d_{0,k}).
\tag{19.2}
$$

相邻比值连乘是非线性复数映射，也必须逐行重建。不能忽略跨边相关性后将各边误差独立相加，也不能把 bootstrap 行数作为独立重复测量数再除以平方根。

三种方案最终均保留全部 100 行重建值，使用

$$
\overline X=\frac{1}{B}\sum_{b=1}^{B}X^{(b)},\qquad
\sigma_X=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(X^{(b)}-\overline X)^2},\qquad B=100.
\tag{19.3}
$$

该标准差逐实部、虚部分别计算（$\mathrm{ddof}=1$），不剔除触边行以人为缩小误差。拟合到中心信号的中心值与 bootstrap 均值可能不同，两者在文件中不得混称。

### 19.3 比较应采用配对差值

两个方案 $A,B$ 对同一物理量的比较定义为

$$
\Delta X^{(b)}=X_A^{(b)}-X_B^{(b)},\qquad
p_X(n)=\frac{X_A(n)-X_B(n)}{\operatorname{SD}[\Delta X^{(b)}(n)]}.
\tag{19.4}
$$

这里分子使用相应中心拟合结果，分母使用同 replica 配对差值的标准差，而非假设两方案独立。后文统计统一比较相同长度集合；$n=1$ 为恒等变换检查，排除在非平凡差异统计之外。各长度相关，扫描所得最大局部 pull 不等于全局显著性。

## 20. 三种方案共同的拟合流程与质量门

### 20.1 固定设置

| 项目 | 本轮设置 |
|---|---|
| ensemble／极化／flavor | L32x96/F32P30；unpolarized connected $u-d$ |
| 动量、长度 | 整数动量 $P=4,5$；$n=0,\ldots,20$ |
| 格距、单位转换 | $a=0.0775\,\mathrm{fm}$，$\hbar c=0.1973269804\,\mathrm{GeV\,fm}$ |
| ratio 物理区间 | 严格开区间 $0.6\,\mathrm{fm}<aT<1.1\,\mathrm{fm}$ |
| 实际可用 $T$ | $P=4:\ 8,9,10,11,12$；$P=5:\ 8,9,10,11$ |
| 插入时间 cut | 对称 cut 2，即 $2\leq\tau\leq T-2$ |
| $C_2$ 时间 | $t=6,7,8,9,10,11,12$ |
| 窗口／AIC | 使用所有可用 $T$ 的固定窗口；本轮不使用 AIC |
| gap 分组 | 每个长度／相邻边独立；同一点实虚部共享 |
| gap 范围 | $0.2\leq\Delta E\leq10\,\mathrm{GeV}$ |
| 其他非线性范围 | $0.01\leq E_0\leq3$（格点单位），$-14\leq\log r\leq14$ |
| 边界识别容差 | 非线性参数边界判断容差 $10^{-4}$ |
| resample | 父产品 2000 行中的同一前 100 行；194 个配置 |

尽管区间允许整数 $T$ 达到 14，实际输入掩码只提供表中列出的时间；特别是 $P=5$ 的 $T=12$ 不可用，未补造信号。

### 20.2 联合似然和协方差处理

对每个 $n\geq1$ 分别组成观测向量，$X_n$ 依次取 $S_n,Q_n,D_n$：

$$
\boldsymbol y_n=\bigl(C_2(t),\operatorname{Re}X_n(T,\tau),
\operatorname{Im}X_n(T,\tau)\bigr),\qquad
\chi_n^2=(\boldsymbol y_n-\boldsymbol f_n)^T
\Sigma_{n,\rm reg}^{+}(\boldsymbol y_n-\boldsymbol f_n).
\tag{20.1}
$$

$C_2$ 使用两态形式，$X_n$ 使用对应的一阶三项式。共有 10 个参数：3 个 $C_2$ 参数 $A_0,E_0,r$，实、虚部各 3 个系数，以及共同能隙 $\delta_n$。$r$ 直接进入 $C_2$，self ratio 等变换后的模型不再带 $1+r e^{-\delta T}$ 分母。尽管显式分母约掉，仍可以联合观测到的 $C_2$ 来约束共同能隙。

每个长度的联合似然只放一次 $C_2$，包含 $C_2$ 与实虚部信号间的交叉协方差；它不是三个相互独立的拟合。不同长度各自重复使用同一 $C_2$，因此不能把所有长度的 $\chi_n^2$ 相加后声称得到全局独立似然。

协方差采用 Schaefer–Strimmer 相关矩阵 shrinkage，再施加相对特征值阈值 $10^{-8}$ 的硬截断。各方案变换之后重新计算 shrinkage，记录样本秩、有效秩、模型秩以及 profile。自由度为保留协方差模态数减去参数数目。

原始空间差分与直接 $S_n$ 之间是可逆线性变换。若对全部长度使用同一个完整模型、精确协方差及一致参数变换，原则上不增加信息；本轮采用逐长度／逐边拟合、不同有效能隙，并分别估计 shrinkage，所以拟合结果与误差不要求不变。结果变化不能仅归因于物理激发态抵消。

### 20.3 残差合格、拟合合格、递归可用是三个层次

第一层为残差门：

$$
\chi^2/\mathrm{dof}<1000,\qquad Q>10^{-6},\qquad
f_{\rm finite}\geq0.9.
$$

第二层为完整拟合门：还须自由度为正、样本及模型秩足够、优化收敛、中心参数不触边、中心 $\Delta\chi^2=1$ 能隙 profile 两端闭合，且至少 90% 的 replica 有限、收敛、满秩并且不触边。profile 闭合是中心拟合条件，不保证所有 replicas 稳定。本轮两动量的三主方案每个 profile 都闭合，但 $P=5$ 仍常因边界或 replica 不稳定而失败。

第三层为重建质量：直接方案要求该长度拟合通过；adjacent 与 difference 的 $H_n$ 要求所有上游边 $1,\ldots,n$ 通过。绝对 $M_n$ 还要求同一配对局域荷 $M_0$ 通过。

adjacent 方案另用下式检查分母风险：

$$
\mathcal S_{n-1}^{\min}=
\min_{(T,\tau)\ \rm fitted}
\frac{|\langle R_{n-1}(T,\tau)\rangle|}
{\sqrt{\operatorname{Var}[\operatorname{Re}R_{n-1}(T,\tau)]
+\operatorname{Var}[\operatorname{Im}R_{n-1}(T,\tau)]}}.
\tag{20.2}
$$

$\mathcal S_{n-1}^{\min}<3$ 标为经验风险，非普适统计定理；不据此删去数据。风险边的拟合即使通过残差、秩和 profile 检查，也不作为安全递归链。所有有限数值仍保存，并明确标注诊断状态。

### 20.4 局域荷来源

三方案使用完全相同的既有 full-window rational $C_2+C_3/C_2$ 配对拟合所给的 $M_0^{(b)}$。直接 self ratio 不会自行恢复局域荷。当前局域荷为

$$
M_0(P=4)=1.1607161\pm0.0016898,\qquad
M_0(P=5)=1.1490273\pm0.0088223.
$$

$P=4$ 局域荷通过，$P=5$ 局域荷未通过完整质量门，因此全部 $P=5$ 的绝对裸矩阵元均为诊断值，不能因乘回局域荷后数值有限就升级为正式结果。这些数值是当前裸荷约定，不是重整化局域电荷。

## 21. 本轮拟合结果与定量比较

### 21.1 拟合质量总览

下面每组有 20 个非零长度或相邻边。“形状可用”列已包括上游边和 adjacent 分母风险；零长度的恒等约束不计入数量。

| 动量 | 方案 | 完整拟合通过的长度／边 | 形状可用长度 | 绝对裸量可用长度 | 中心触边数 |
|---|---|---|---|---|---|
| 4 | 直接 self ratio | 1–20 | 1–20 | 1–20 | 0 |
| 4 | adjacent self ratio | 1–20 | 1–7 | 1–7 | 0 |
| 4 | self-ratio difference | 1–20 | 1–20 | 1–20 | 0 |
| 5 | 直接 self ratio | 7、9 | 7、9 | 无 | 6 |
| 5 | adjacent self ratio | 仅边 3 | 无 | 无 | 6 |
| 5 | self-ratio difference | 仅边 14 | 无 | 无 | 11 |

所有 120 个主方案长度／边的残差门均通过，全部 100 行拟合参数有限；但完整质量门并未全部通过。范围及最小 $Q$ 由保存的中心拟合记录计算：

| 动量 | 方案 | $\chi^2/\mathrm{dof}$ 最小–最大 | 最小 $Q$ |
|---|---|---|---|
| 4 | 直接 | 0.6379–1.4954 | 0.005358 |
| 4 | 相邻 | 0.1074–1.5939 | 0.001440 |
| 4 | 差分 | 0.8148–1.5324 | 0.003315 |
| 5 | 直接 | 0.6837–1.4732 | 0.017216 |
| 5 | 相邻 | 0.1297–1.2034 | 0.155772 |
| 5 | 差分 | 0.7987–1.6192 | 0.003930 |

$P=4$ 的 adjacent 风险边为 $n=8,\ldots,20$，故即使 20 条边均通过数值拟合门，安全形状与绝对裸量仍只到 $n=7$，即 $z=0.5425\,\mathrm{fm}$。$P=5$ 的风险边为 $n=3,\ldots,20$，仅有的合格拟合边 3 也处在风险范围；差分虽有边 14 通过，但它之前的边并未全通过，不能形成有效重建链。

### 21.2 同一长度集合的误差比较

定义误差比为某方案对直接 self ratio 的标准差比，再在指定长度集合取中位数：

$$
\rho_X(A;\mathcal I)=\operatorname{median}_{n\in\mathcal I}
\frac{\sigma_{X,A}(n)}{\sigma_{X,\rm direct}(n)}.
$$

两组共同集合为 $\mathcal I_{\rm short}=\{2,\ldots,7\}$ 和 $\mathcal I_{\rm all}=\{2,\ldots,20\}$。后者在 adjacent 中包含风险点，$P=5$ 中也包含未通过完整质量门的点，均只能用于诊断。下表比较的是归一化形状 $H_n$，不是尚未乘回局域荷的增量 $q_{0,n},d_{0,n}$ 本身。

| 动量 | 方案／集合 | $\rho_{\mathrm{Re}H}$ | $\rho_{\mathrm{Im}H}$ | 解释 |
|---|---|---|---|---|
| 4 | 相邻，2–7 | 1.142 | 1.057 | $P=4$ 为安全范围；$P=5$ 为诊断 |
| 4 | 差分，2–7 | 1.298 | 1.172 | $P=4$ 可用；$P=5$ 为诊断 |
| 4 | 相邻，2–20 | 0.831 | 0.921 | 含分母风险点，仅诊断 |
| 4 | 差分，2–20 | 1.704 | 1.600 | $P=4$ 可用；$P=5$ 为诊断 |
| 5 | 相邻，2–7 | 1.706 | 2.190 | $P=4$ 为安全范围；$P=5$ 为诊断 |
| 5 | 差分，2–7 | 2.397 | 2.846 | $P=4$ 可用；$P=5$ 为诊断 |
| 5 | 相邻，2–20 | 3.505 | 3.410 | 含分母风险点，仅诊断 |
| 5 | 差分，2–20 | 3.886 | 4.034 | $P=4$ 可用；$P=5$ 为诊断 |

对 $P=4$，安全短距内 adjacent 的实部、虚部误差中位数分别约为直接方案的 1.14、1.06 倍，没有显示普遍改善。差分在 $n=2$–20 的对应倍数为 1.70、1.60；虽然原始相邻点正相关能减小差分信号的噪声，拟合增量再累加并未使最终形状更精确。

adjacent 在全长度集合出现小于 1 的误差比，不能据此宣称优于直接方案：该集合包含小分母、长尾和递归不稳定的结果。$P=5$ 的误差普遍较大且质量不足，不宜据此排序三方案的物理优劣。

### 21.3 配对差值与能隙

| 动量 | 方案／集合 | 最大局部 $\lvert p_{\Delta E}\rvert$ | 最大局部 $\lvert p_{\mathrm{Re}H}\rvert$ | 最大局部 $\lvert p_{\mathrm{Im}H}\rvert$ |
|---|---|---|---|---|
| 4 | 相邻，2–7 | 2.500 | 1.011 | 1.902 |
| 4 | 差分，2–20 | 2.305 | 1.982 | 1.941 |
| 5 | 相邻，2–20（诊断） | 1.776 | 1.092 | 1.083 |
| 5 | 差分，2–20（诊断） | 2.848 | 0.787 | 0.768 |

$P=4$ 的两种局部变换与直接 self ratio 的形状总体接近，但局部有效能隙可有约 2–2.5 个配对标准差的变化。差分方案在 $n=2$–20 的最大形状 pull 约 1.98（实部）、1.94（虚部），尚不能据此宣布激发态已完全消除，也不能将单个局部偏差解释为新的物理能级。$P=5$ 的宽误差会使 pull 看起来较小，这不构成其拟合可靠的证据。

### 21.4 代表性数值

下表为 $P=4$ 的归一化形状及有效能隙，误差为全部 100 行的标准差。adjacent 的 $n\geq8$ 明确标为风险；其余表内主方案点均通过本轮质量门。更完整的中心值、绝对裸量及误差见第 23 节机器表。

| $n$ | $z$/fm | 方案 | $\Delta E$/GeV | $\mathrm{Re}H_n$ | $\mathrm{Im}H_n$ | 状态 |
|---|---|---|---|---|---|---|
| 5 | 0.3875 | 直接 | $0.944\pm0.141$ | $0.28901\pm0.00666$ | $-0.31534\pm0.00681$ | 本轮通过 |
| 5 | 0.3875 | 相邻 | $1.019\pm0.202$ | $0.28600\pm0.00767$ | $-0.31080\pm0.00739$ | 本轮通过 |
| 5 | 0.3875 | 差分 | $1.063\pm0.216$ | $0.29387\pm0.00809$ | $-0.30864\pm0.00825$ | 本轮通过 |
| 10 | 0.7750 | 直接 | $1.065\pm0.161$ | $0.00397\pm0.00659$ | $-0.07255\pm0.00771$ | 本轮通过 |
| 10 | 0.7750 | 相邻 | $0.959\pm0.168$ | $-0.00270\pm0.00668$ | $-0.06052\pm0.00793$ | 分母风险，诊断 |
| 10 | 0.7750 | 差分 | $0.957\pm0.161$ | $0.00840\pm0.01253$ | $-0.06829\pm0.01159$ | 本轮通过 |
| 20 | 1.5500 | 直接 | $1.048\pm0.162$ | $-0.00823\pm0.00720$ | $0.00270\pm0.00587$ | 本轮通过 |
| 20 | 1.5500 | 相邻 | $1.089\pm0.223$ | $-0.00000\pm0.00062$ | $0.00000\pm0.00110$ | 分母风险，诊断 |
| 20 | 1.5500 | 差分 | $0.986\pm0.170$ | $0.01447\pm0.01457$ | $0.00350\pm0.01455$ | 本轮通过 |

adjacent 的长距形状趋近零且表观误差缩小，是需要警惕的结果，不能只根据图上误差棒短就接受；其完整风险信息必须随数值一起使用。

### 21.5 对照试验：不联合 $C_2$ 的直接 self ratio

另有同一数据、窗口、能隙边界和 100 行的 SR-only 对照，仅去掉 $C_2$ 似然，不列为第四种空间变换方案。两动量共 40 个拟合全部通过残差门，却全部存在开放的中心能隙 profile；$P=4$ 有 11/20、$P=5$ 有 18/20 个中心拟合触边，没有任何点通过完整能隙质量门。

以 $P=4,n=5$ 为例，SR-only 得到

$$
\Delta E=1.0976\pm4.2875\,\mathrm{GeV},\qquad
\operatorname{Re}H_5=0.290781\pm0.073333,
$$

而联合 $C_2$ 的直接方案给出

$$
\Delta E=0.9440\pm0.1407\,\mathrm{GeV},\qquad
\operatorname{Re}H_5=0.289008\pm0.006661.
$$

SR-only 在共同 $n=2$–20 上对直接联合方案的实、虚部形状误差中位数倍数分别为：

| 动量 | 实部误差倍数 | 虚部误差倍数 |
|---|---|---|
| 4 | 9.84 | 13.06 |
| 5 | 8.76 | 8.46 |

这些受边界限制的标准差不是高斯置信区间。该对照说明：self ratio 可以减少对某些激发项的敏感性，同时也会让能隙本身难以从残余时间依赖中确定。联合 $C_2$ 的约束在本轮 $P=4$ 中有效；它仍不足以使 $P=5$ 全部稳定。

## 22. 三方案的优势、局限与目前可采用的结论

| 方案 | 可能优势 | 主要代价／失效机制 | 本轮结论 |
|---|---|---|---|
| 直接 $R_n/R_0$ | 局域分母信号较强；直接拟合各长度的形状；不累积上游边 | 共享局域分母噪声；激发项仅部分抵消；局域荷需单独恢复 | $P=4$ 作为当前主结果；$P=5$ 仅少数形状点通过 |
| 相邻 $R_n/R_{n-1}$ | 有机会抵消相邻长度相近的相对激发污染 | 非局域小分母、复数长尾、一阶展开失效、连乘误差及链式质量限制 | $P=4$ 仅短距 1–7 可作交叉检查；长距及 $P=5$ 不能作改进后的主结果 |
| 差分 $S_n-S_{n-1}$ | 只用局域分母；利用相邻正相关；直接研究矩阵元空间增量 | 小增量难拟合；累加恢复相关误差；一条上游坏边会影响后续重建 | $P=4$ 通过但误差较大，适合系统比较；$P=5$ 无完整可用链 |

三种原始信号通过恒等变换联系，它们不创造额外物理信息；只有同模型、全协方差和一致拟合约束下的信息比较才具有严格意义。本轮逐点／逐边拟合的统计行为不同，适合测试截断及稳定性，不能仅凭更小的 $\chi^2$ 或更短的误差棒选方案。

现阶段建议沿用联合 $C_2$ 的直接 self ratio 作为 $P=4$ 的主分析，差分作为空间增量和拟合系统误差的交叉检查；adjacent 仅在分母信号充足的短距使用。$P=5$ 保留所有有限输出及失败标签，不能进入正式绝对裸矩阵元组合。

以上仍是前 100 行的试算结论。没有把它们升级为全部 2000 行的生产结论，也没有重新进行 hybrid renormalization、$\lambda$ 外推或 Fourier 变换。激发态控制仍需检查更晚 $T$ 的稳定性、模型截断和更充分的统计信息，不能从本轮残差合格直接推出污染已消除。

## 23. 数据、图表与可追溯记录

### 23.1 权威输出与代码

下列目录保留既有拟合结果，本次只读取并整理。每个主方案均有 `analysis_config.json`、`summary.json`、`validation.json`、`complete.json`，验证状态为 `PASS`。这里的 `PASS` 指数值实现复核通过，不表示每个物理点都通过完整拟合质量门。

**方案一：直接 self ratio**，作业 `4022340`。

- [结果目录](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117)
- [拟合配置](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/analysis_config.json)；[数值摘要](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/summary.json)；[独立验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/validation.json)
- [$P=4$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/selfratio/P4/matrix_and_gap.npz)；[$P=5$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/selfratio/P5/matrix_and_gap.npz)

实现代码：[SelfRatio_JointC2_L32x96_P45_First100_20261117](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/SelfRatio_JointC2_L32x96_P45_First100_20261117)。

**方案二：adjacent self ratio**，作业 `4022964`。

- [结果目录](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117)
- [拟合配置](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/analysis_config.json)；[数值摘要](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/summary.json)；[独立验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/validation.json)
- [$P=4$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/adjacent/P4/matrix_and_gap.npz)；[$P=5$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/adjacent/P5/matrix_and_gap.npz)

实现代码：[AdjacentRatio_JointC2_L32x96_P45_First100_20261117](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/AdjacentRatio_JointC2_L32x96_P45_First100_20261117)。

**方案三：self-ratio difference**，作业 `4023029`。

- [结果目录](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117)
- [拟合配置](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/analysis_config.json)；[数值摘要](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/summary.json)；[独立验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/validation.json)
- [$P=4$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/difference/P4/matrix_and_gap.npz)；[$P=5$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/difference/P5/matrix_and_gap.npz)

实现代码：[SelfRatio_Difference_JointC2_L32x96_P45_First100_20261117](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/SelfRatio_Difference_JointC2_L32x96_P45_First100_20261117)。

**对照：SR-only**，作业 `4022996`。

- [结果目录](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_only_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117)
- [拟合配置](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_only_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/analysis_config.json)；[数值摘要](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_only_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/summary.json)；[独立验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_only_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/validation.json)
- [$P=4$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_only_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/selfratio_only/P4/matrix_and_gap.npz)；[$P=5$ 汇总 NPZ](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_only_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/selfratio_only/P5/matrix_and_gap.npz)

实现代码：[SelfRatio_Only_L32x96_P45_First100_20261117](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/SelfRatio_Only_L32x96_P45_First100_20261117)。

本地论文全文：[arXiv:2610.39944v1 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/paper/self_ratio_2610.39944v1.pdf)；[可检索文本](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/paper/self_ratio_2610.39944v1.txt)。

### 23.2 Signal 和拟合比较图

以下是既有图，不新增拟合。直接 self ratio 的 `normalized_comparison` 图比较的是 self ratio 与原始 $\frac{C_3}{C_2}$；adjacent／difference 的形状与裸量图比较的是各自方案与直接 self ratio，不能混用对照对象。先看原生观测量的 signal，再看恢复至共同归一化的曲线。各 signal PDF 按长度分页。

#### 动量 $P=4$

- 直接 self ratio：[原生 signal 与拟合](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/signal_and_fit_P4_all_z.pdf)；[共同归一化 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/signal_common_normalization_P4_all_z.pdf)；[能隙](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/deltaE_P4.png)；[绝对裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/absolute_bare_comparison_P4.png)。
- adjacent：[相邻比值 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/native_signal_and_fit_P4_all_z.pdf)；[恢复到共同归一化](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/common_signal_and_fit_P4_all_z.pdf)；[短距形状](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/shape_comparison_P4_short_z.png)；[全长度裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/bare_comparison_P4_all_z.png)；[能隙](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/gap_comparison_P4.png)。
- difference：[差分 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/difference_signal_and_fit_P4_all_z.pdf)；[累加恢复后的共同 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/common_signal_and_fit_P4_all_z.pdf)；[重建形状](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/shape_comparison_P4_all_z.png)；[重建裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/bare_comparison_P4_all_z.png)；[能隙](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/gap_comparison_P4.png)；[空间增量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/increment_comparison_P4.png)。

#### 动量 $P=5$

- 直接 self ratio：[原生 signal 与拟合](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/signal_and_fit_P5_all_z.pdf)；[共同归一化 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/signal_common_normalization_P5_all_z.pdf)；[能隙](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/deltaE_P5.png)；[绝对裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/absolute_bare_comparison_P5.png)。
- adjacent：[相邻比值 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/native_signal_and_fit_P5_all_z.pdf)；[恢复到共同归一化](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/common_signal_and_fit_P5_all_z.pdf)；[短距形状](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/shape_comparison_P5_short_z.png)；[全长度裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/bare_comparison_P5_all_z.png)；[能隙](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/gap_comparison_P5.png)。
- difference：[差分 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/difference_signal_and_fit_P5_all_z.pdf)；[累加恢复后的共同 signal](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/signals/common_signal_and_fit_P5_all_z.pdf)；[重建形状](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/shape_comparison_P5_all_z.png)；[重建裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/bare_comparison_P5_all_z.png)；[能隙](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/gap_comparison_P5.png)；[空间增量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/increment_comparison_P5.png)。

质量诊断图：[adjacent 分母风险与质量](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/adjacentratio_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/denominator_and_quality.png)；[difference 质量与边界](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/quality_and_bounds.png)；[能隙 profile 示例](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/artifacts/selfratio_difference_jointc2_L32x96_unpolarized_ud_P45_fullT_cut2_gap0p2to10_first100_20261117/plots/profile_examples.png)。

### 23.3 NPZ 格式与读取注意

主汇总文件为各方案子目录下的 `P4/matrix_and_gap.npz`、`P5/matrix_and_gap.npz`。使用 `numpy.load(..., allow_pickle=False)` 读取。

| 字段 | 含义与常见形状 |
|---|---|
| `z_a`、`z_fm` | 长度编号及 fm 长度，均为 `(21,)` |
| `deltaE_GeV`、`deltaE_GeV_SD` | 各点有效能隙中心值和标准差；零长度不拟合能隙 |
| `deltaE_GeV_boot` | `(100,21)`，对应固定的原始行号；零长度为占位，不作能隙测量 |
| `shape_Re`、`shape_Im` | 恢复到统一 $H_n=M_n/M_0$ 后的中心值，`(21,)` |
| `shape_Re_boot`、`shape_Im_boot` | 同行直接恢复／复数连乘／累加后的形状，`(100,21)` |
| `bare_Re`、`bare_Im` 及 `_boot`、`_SD` | 乘回同一行局域荷后的绝对裸量及误差 |
| `quality_gap`、`quality_shape`、`quality_absolute` | 各层质量掩码，`(21,)`；不能以有限性替代这些掩码 |
| `local_charge_boot`、`local_charge_quality` | `(100,)` 配对局域荷及该参考荷的质量标志 |
| `confs`、`boot_indices` | 配置列表 `(194,)` 及本轮索引 `(100,194)` |
| `Nboot`、`parent_Nboot`、`original_replica_rows` | 100、2000、原行号 0–99 |

各目录还保存逐长度 `fits/z01.npz`–`fits/z20.npz`，包含参数名称、中心参数、逐行参数、$\chi^2$、$Q$、自由度、边界及收敛标志、秩、正则化协方差及 profile。相邻比值和差分的拟合常数分别是 $q_{0,n}$、$d_{0,n}$，不等于最终 $H_n$；后者应从汇总文件读取。具体附加字段以该文件的 `parameters_names` 和目录 README 为准。

### 23.4 本次文档整理及验证

整理工作目录：[Self_Ratio_Notes_20261117](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117)。

- [source_ledger.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/source_ledger.json)：权威目录、作业号、来源 SHA-256 和既有独立验证记录。
- [statistics.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/statistics.json)：质量掩码、残差范围、局域荷及边界数量。
- [paired_statistics_common_z.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/paired_statistics_common_z.json)：在相同长度集合重算的误差比和配对 pull。
- [representative_values.csv](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/representative_values.csv)：选定长度的中心值、标准差、形状和绝对裸量。
- [self_ratio.before.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/self_ratio.before.md)：修改前文档快照；[target_before.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/target_before.json) 记录其哈希。
- [collect.py](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/collect.py)：来源核对及配对统计脚本；[build_document.py](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_20261117/build_document.py)：本轮文档生成脚本。

| 方案 | 中心及全部 replica 的最大独立重建 $\chi^2$ 误差 |
|---|---|
| 直接 | $3.951\times10^{-12}$ |
| 相邻 | $3.652\times10^{-12}$ |
| 差分 | $3.126\times10^{-12}$ |
| SR-only 对照 | $2.117\times10^{-12}$ |

原始相邻比值的连乘恒等式最大误差分别为 $P=4:\ 4.48\times10^{-16}$、$P=5:\ 1.57\times10^{-15}$；原始差分累加恒等式分别为 $1.32\times10^{-16}$、$2.91\times10^{-16}$。差分首边与直接 self ratio 的仿射等价检查中，全部 bootstrap 的最大 $\chi^2$ 差异分别为 $4.43\times10^{-12}$、$2.19\times10^{-12}$。这些是代数和实现检查，不代替激发态或物理系统误差分析。

本次整理另核对了四组结果的配置、原行号、bootstrap 索引和局域荷逐行一致，全部标准差确为所有 100 行的 $\mathrm{ddof}=1$ 标准差；来源摘要、配置、验证文件和汇总 NPZ 的哈希与各自完成清单一致。第 1–17 节历史正文内容保留（本次仅统一数学排版），后续使用者应从第 18 节开始阅读本轮三方案的统一结论。

---

## 24. L32x96 unpolarized gluon：四种方案的统一试算（2026-11-17）

### 24.1 范围和输入来源

本轮对 L32x96/F32P30 的 unpolarized gluon 同时测试四种方案：直接 $C_3/C_2$、$R_n/R_0$、$R_n/R_{n-1}$ 和 $R_n/R_0-R_{n-1}/R_0$。输入为既有 **HYP10、$T$-only、三个正轴方向平均、仅实部**产品。不是第 18–23 节的 quark 数据，不是 $T-S$ 算符，也不是 gradient-flow gluon 产品。这里的裸矩阵元保持原始 lattice ratio 振幅归一化，尚未加入体积、耦合、运动学或颜色无迹转换，未做重整化及 PDF matching。

| 项目 | 本轮设置 |
|---|---|
| ensemble | L32x96/F32P30，$N_s^3\times N_t=32^3\times96$ |
| 格距 | $a=0.0775\,\mathrm{fm}$ |
| 动量 | 整数模编号 $P=4,5$，每个动量沿 $+x,+y,+z$ 与同向 link 配对 |
| 空间长度 | $z_n=na$，$n=0,\ldots,23$ |
| 配置与 bootstrap | $N_{\rm cfg}=777$，父产品 $B_{\rm parent}=3000$，本轮 $B=100$，原始行号 $b=0,\ldots,99$，种子 1115 |
| 物理源汇距区间 | 严格开区间 $0.6\,\mathrm{fm}<aT<1.1\,\mathrm{fm}$ |
| 实际使用时间 | 两动量均为 $T=8,\ldots,14$，即 $aT=0.6200,\ldots,1.0850\,\mathrm{fm}$ |
| 插入时间 | 对称 cut $c=2$，$2\leq\tau\leq T-2$ |
| 联合两点函数 | $t=6,\ldots,12$ |
| 选择方式 | 一个固定完整时间窗口，不使用 AIC |

本轮与前面的 quark 试算配置数、父 bootstrap 行数和可用时间不同。相同种子不意味着 quark 与 gluon 跨产品配对；本轮四种 gluon 方案之间才使用完全相同的配置和随机数。

### 24.2 算符和投影约定

采用空间、时间指标 $0123=xyzt$，每个方向保留的张量组合为

$$
T_x=F_{31}F_{31}+F_{32}F_{32},\qquad
T_y=F_{32}F_{32}+F_{30}F_{30},\qquad
T_z=F_{30}F_{30}+F_{31}F_{31}.
\tag{24.1}
$$

上式省略非局域 Wilson-line 连接和颜色收缩记号，实际输入算符为

$$
O_d(z_n,t)=-2\operatorname{Re}T_{d,+}(z_n,t),\qquad d\in\{x,y,z\}.
\tag{24.2}
$$

整体负号和因子 2 沿用既有产品。只有 plus-link 文件，因子 2 依赖已登记的 unpolarized 反向恒等约定；不能据此声称独立读取过正反两个方向的 OPE。这里平均的是三个正轴动量，不是六个正负动量。

### 24.3 真空扣除、方向平均和数据补齐

令 $B_{d,c}(T,s)=C_{2,d,c}((s+T)\bmod N_t,s)$。每个原始 bootstrap 行重新执行组态平均、真空扣除及全源时刻平均：

$$
C_{3,d}^{(b)}(n,T,\tau)=\frac{1}{N_t}\sum_s\left[
\left\langle O_d(z_n,s+\tau)B_d(T,s)\right\rangle_b
-\left\langle O_d(z_n,s+\tau)\right\rangle_b
 \left\langle B_d(T,s)\right\rangle_b\right].
\tag{24.3}
$$

$$
C_{2,d}^{(b)}(T)=\frac{1}{N_t}\sum_s\left\langle B_d(T,s)\right\rangle_b,
\qquad
R_n^{(b)}(T,\tau)=\frac13\sum_{d=x,y,z}
\frac{\operatorname{Re}C_{3,d}^{(b)}(n,T,\tau)}{\operatorname{Re}C_{2,d}^{(b)}(T)}.
\tag{24.4}
$$

三种 self 变换都在式 (24.4) 的方向平均之后逐行构造；并非每方向先 self 归一化后平均，也不是先平均相关函数再作 ratio。联合约束的两点函数取

$$
\overline C_2^{(b)}(t)=\frac13\sum_{d=x,y,z}\operatorname{Re}C_{2,d}^{(b)}(t).
\tag{24.5}
$$

平均后用共同单谱描述 $\overline C_2$ 和 $R_n$ 是本轮有效模型假设；有限时间的“比值平均”通常不严格等于“平均相关函数之比”。

旧 gluon 缓存从 $T=10$ 开始。本次按同一 777 个配置的清单重新读取完整原始两点函数，补出 $T=8,9$ 及联合两点函数所需的 $t=6,\ldots,12$。每个原始文件哈希与旧输入清单一致；OPE 复用已核验缓存，不重新抽样。在 $T=10,\ldots,14$ 的重叠域，新构造的两点函数、三点函数及 ratio 与旧产品同一前 100 行核对通过，并以显式源时刻求和独立抽查 FFT 收缩和真空扣除。具体误差见第 28 节。

## 25. Gluon 四种模型及裸矩阵元重建

### 25.1 共同谱分解与有效模型前提

继续令 $z_n=na$，时间变量 $T,\tau,t$ 均以格点为单位：

$$
\delta_n=\frac{a\Delta E_n}{\hbar c},\qquad
F_n=e^{-\delta_n\tau}+e^{-\delta_n(T-\tau)},\qquad
G_n=e^{-\delta_n T},\qquad
\hbar c=0.1973269804\,\mathrm{GeV\,fm}.
\tag{25.1}
$$

两态模型只保留前向传播，采用源端、汇端跃迁振幅相等的对称近似：

$$
C_2(t)=A_0e^{-E_0t}\left(1+r e^{-\delta_n t}\right),\qquad
R_n(T,\tau)=\frac{M_n+B_nF_n+C_nG_n}{1+rG_n},\qquad A_0>0,\ r>0.
\tag{25.2}
$$

这里 $M_n$ 为基态裸振幅，$B_n$ 为对称跃迁系数，$C_n$ 为激发态对角系数。没有从局域守恒性强制设置 gluon 跃迁项为零。模型中的 $C_n$ 与两点函数 $C_2$ 是不同记号。

若采用完全共同的物理谱，真实能隙与算符长度无关。当前每个长度或相邻边独立拟合 $\Delta E_n$，只作为截断模型下的有效参数；转换观测量后，其数值不必等于原始 ratio 模型的有效能隙。有限时间后的向后传播和热态项未显式建模；本轮只使用 $t\leq14$、$N_t=96$ 的窗口，仍需保留这一近似限制。

### 25.2 方案 G-R：联合 $C_2$ 的直接 $C_3/C_2$

对每个 $n=0,\ldots,23$ 使用式 (25.2) 的**有理式**；$C_2$ 与 ratio 共享能隙及重叠比 $r$。七个参数为

$$
\boldsymbol\theta_{R,n}=(A_0,E_0,r,M_n,B_n,C_n,\delta_n).
\tag{25.3}
$$

它直接输出 $M_n$，不需要逐观测点除以局域三点函数。将同一 gluon 数据的 $n=0$ 拟合作为参考锚点 $M_0^{(b)}$，供其他三个方案恢复绝对幅度。这个锚点不是 quark 的局域荷。

### 25.3 方案 G-S：直接 self ratio

共同谱下，两态截断内的精确比值为

$$
S_n=\frac{R_n}{R_0}
=\frac{M_n+B_nF+C_nG}{M_0+B_0F+C_0G}.
\tag{25.4}
$$

本轮拟合的是分母谱混合量的一阶展开，而非完整有理式：

$$
S_n\simeq s_{0,n}+s_{1,n}F_n+s_{2,n}G_n,
\quad
s_{0,n}=\frac{M_n}{M_0},\quad
s_{1,n}=\frac{B_n-s_{0,n}B_0}{M_0},\quad
s_{2,n}=\frac{C_n-s_{0,n}C_0}{M_0}.
\tag{25.5}
$$

$$
H_n^{(b)}=s_{0,n}^{(b)},\qquad M_n^{(b)}=M_0^{(b)}s_{0,n}^{(b)}.
\tag{25.6}
$$

$S_0=1$ 是精确恒等约束，不参与拟合，也不提供 self 能隙。其他长度分别联合同一行 $C_2$，但没有额外把局域三点函数本身放入联合似然。

### 25.4 方案 G-A：adjacent self ratio

$$
Q_n=\frac{R_n}{R_{n-1}}
=\frac{M_n+B_nF+C_nG}{M_{n-1}+B_{n-1}F+C_{n-1}G}
\simeq q_{0,n}+q_{1,n}F_n+q_{2,n}G_n,
\tag{25.7}
$$

$$
q_{0,n}=\frac{M_n}{M_{n-1}},\qquad
q_{1,n}=\frac{B_n-q_{0,n}B_{n-1}}{M_{n-1}},\qquad
q_{2,n}=\frac{C_n-q_{0,n}C_{n-1}}{M_{n-1}}.
\tag{25.8}
$$

$$
H_n^{(b)}=\prod_{j=1}^{n}q_{0,j}^{(b)},\qquad M_n^{(b)}=M_0^{(b)}H_n^{(b)}.
\tag{25.9}
$$

本轮数据只有实部，因此为实数连乘。若用于复数 quark 数据，必须改成第 18 节的完整复数除法和连乘。相邻长度可能抵消共同污染，但有噪声的非局域分母会引入长尾和递归风险；单条边拟合稳定不足以保证整条重建链可靠。

### 25.5 方案 G-D：self-ratio difference

令 $\Delta M_n=M_n-M_{n-1}$，对其他谱系数同样定义相邻差，则

$$
D_n=S_n-S_{n-1}=\frac{R_n-R_{n-1}}{R_0}
=\frac{\Delta M_n+\Delta B_nF+\Delta C_nG}{M_0+B_0F+C_0G}
\simeq d_{0,n}+d_{1,n}F_n+d_{2,n}G_n,
\tag{25.10}
$$

$$
d_{0,n}=\frac{\Delta M_n}{M_0},\qquad
d_{1,n}=\frac{\Delta B_n-d_{0,n}B_0}{M_0},\qquad
d_{2,n}=\frac{\Delta C_n-d_{0,n}C_0}{M_0}.
\tag{25.11}
$$

$$
H_n^{(b)}=1+\sum_{j=1}^{n}d_{0,j}^{(b)},\qquad M_n^{(b)}=M_0^{(b)}H_n^{(b)}.
\tag{25.12}
$$

差分沿空间长度进行，未除以格距，也不是 summed-ratio 的源汇距差分。重建时须在同一 replica 内先累加再乘锚点，保留跨长度和锚点间相关性。

### 25.6 恒等式与截断限制

原始同一行信号精确满足

$$
\prod_{j=1}^{n}Q_j=S_n,\qquad
1+\sum_{j=1}^{n}D_j=S_n,\qquad
Q_1=S_1,\qquad D_1=S_1-1.
\tag{25.13}
$$

三个 self 模型的一阶展开均省略谱混合量乘积；独立拟合后各边有效能隙和 shrinkage 不同，因此拟合重建值不必满足完全相等。尤其 $F^2$ 本身含 $e^{-\delta T}$ 项，所以这里是对混合量的展开，不能称为保持了全部同指数阶项的精确两态表达式。

共享谱并不保证不同长度的相对跃迁污染相同，也不保证 self 变换消除全部激发态。其一阶展开动机沿用第 18.5 节的论文公式关系；adjacent 和空间差分仍是本地扩展，不将本次 gluon 数值结果归为论文已经验证的结论。

## 26. Gluon 联合拟合、误差及质量门

### 26.1 联合似然及协方差

每个长度／边只放一次两点函数，保留两点函数与变换信号的交叉协方差：

$$
\boldsymbol y_{X,n}^{(b)}=
\left(\overline C_2^{(b)}(6),\ldots,\overline C_2^{(b)}(12),
X_n^{(b)}(T,\tau)\right),\qquad X\in\{R,S,Q,D\},
$$

$$
\chi_{X,n}^2=(\boldsymbol y_{X,n}-\boldsymbol f_{X,n})^T
\Sigma_{X,n,\rm reg}^{+}(\boldsymbol y_{X,n}-\boldsymbol f_{X,n}).
\tag{26.1}
$$

每个拟合共有七个参数：三个两点函数参数、三个信号系数以及共同能隙。两点函数的数值缩放仅改善优化条件，物理参数已还原；未改变原始裸矩阵元归一化。不同长度重复使用同一两点函数，不能将逐长度的似然当作独立数据直接相加。

中心值拟合变换后同一 100 行信号的均值，协方差也由这 100 行构造。采用 Schaefer–Strimmer correlation-to-identity shrinkage，再以相对阈值 $10^{-8}$ 截断特征模态；不是肘点截断。每个空间变换后重新估计 shrinkage；参数误差是全部 100 行的标准差：

$$
\sigma_X=\sqrt{\frac{1}{99}\sum_{b=0}^{99}\left(X^{(b)}-\overline X\right)^2}.
\tag{26.2}
$$

不再除以 $\sqrt{100}$，不删去过零或触边行，不对拟合值再次 bootstrap。配对比较使用同一行的差值标准差；差分重建的误差保留第 19 节所述跨边协方差。

### 26.2 参数边界与数值质量

$$
0.2\,\mathrm{GeV}\leq\Delta E_n\leq10\,\mathrm{GeV},\qquad
0.01\leq E_0\leq3,\qquad -14\leq\log r\leq14.
\tag{26.3}
$$

$E_0$ 为格点单位，非线性边界判断容差为 $10^{-4}$。残差门为

$$
\chi^2/\mathrm{dof}<1000,\qquad Q>10^{-6},\qquad f_{\rm finite}\geq0.9.
\tag{26.4}
$$

完整数值拟合门还要求正自由度、样本及模型秩足够、优化收敛、中心非线性参数不触边、中心能隙 $\Delta\chi^2=1$ profile 两端闭合，以及至少 $90\%$ 的 replica 有限、收敛、满秩且不触边。

### 26.3 分母风险和重建质量

对拟合窗口内每个分母观测点，以本轮 100 行计算均值 $\mu_D$、标准差和分位数。保守分母门为

$$
\min\frac{|\mu_D|}{\operatorname{SD}(D^{(b)})}\geq3,\qquad
\min\Pr_b\!\left(D^{(b)}\mu_D>0\right)\geq0.99,
$$

$$
\min\frac{q_{0.01}(|D^{(b)}|)}{q_{0.5}(|D^{(b)}|)}\geq0.05,\qquad
\min f_{\rm finite}(D)\geq0.99.
\tag{26.5}
$$

这些是经验风险门，不是普适显著性定理，也未据此裁剪数据。直接 self 和差分的分母为 $R_0$，adjacent 的分母为 $R_{n-1}$。本段的 $D$ 泛指分母随机变量，区别于方案 G-D 的空间差分 $D_n$。

`quality_fit` 表示数值拟合门；`quality_gap` 再加入该点的分母门。adjacent／difference 的 `quality_shape` 要求全部上游边通过，`quality_absolute` 再要求同一局域锚点拟合通过。直接 ratio 的绝对裸量不需要局域参考锚点；只有其归一化形状需要该参考。所有有限结果始终保存，不能用有限性代替质量掩码。

## 27. Gluon 四方案的实际结果

### 27.1 完成状态与通过数量

作业 `4023203` 在 CPU 节点正常完成，退出码为零。两动量共 186 个长度／边、18,600 次 replica 拟合，全部完成；每个拟合均通过本轮数值拟合门，中心非线性参数无触边，能隙 profile 闭合。然而三种 self 方案的分母风险门均未通过，不能因此升级为正式的非零长度重建裸量。

| 动量 | 方案 | 数值拟合通过 | 分母安全的 gap 点 | 可用的非零长度裸量 |
|---|---|---|---|---|
| $P=4$ | 直接 $C_3/C_2$ | 24/24 | 24 | 23/23 |
| $P=4$ | Self ratio | 23/23 | 0 | 0/23 |
| $P=4$ | Adjacent self ratio | 23/23 | 0 | 0/23 |
| $P=4$ | Self-ratio difference | 23/23 | 0 | 0/23 |
| $P=5$ | 直接 $C_3/C_2$ | 24/24 | 24 | 23/23 |
| $P=5$ | Self ratio | 23/23 | 0 | 0/23 |
| $P=5$ | Adjacent self ratio | 23/23 | 0 | 0/23 |
| $P=5$ | Self-ratio difference | 23/23 | 0 | 0/23 |

直接 ratio 每个动量的 24 个点包含零长度；最后一列只计 23 个非零长度。三种 self 方案的零长度裸量共用相同锚点，不是额外的 self 能隙测量。

| 动量 | 方案 | $\chi^2/\mathrm{dof}$ 范围 | 最小 $Q$ |
|---|---|---|---|
| $P=4$ | 直接 $C_3/C_2$ | $0.6883\text{--}1.0880$ | $0.303274$ |
| $P=4$ | Self ratio | $0.3840\text{--}0.7410$ | $0.925954$ |
| $P=4$ | Adjacent self ratio | $0.0136\text{--}0.7970$ | $0.862698$ |
| $P=4$ | Self-ratio difference | $0.3840\text{--}0.7915$ | $0.870097$ |
| $P=5$ | 直接 $C_3/C_2$ | $0.4363\text{--}1.0238$ | $0.425443$ |
| $P=5$ | Self ratio | $0.1979\text{--}0.6813$ | $0.967373$ |
| $P=5$ | Adjacent self ratio | $0.0160\text{--}0.3972$ | $0.999985$ |
| $P=5$ | Self-ratio difference | $0.2039\text{--}0.5398$ | $0.998092$ |

adjacent 的部分残差非常小，不能直接解释为物理污染更小。非线性比值的长尾、变换后的协方差估计及正则化均会影响拟合优度，形式上的 $Q$ 值不是分母安全的替代指标。

### 27.2 局域锚点通过，有限时间分母仍可能不安全

| 动量 | 局域锚点 $M_0$ | 最小局域信号 SNR | 最低同号比例 |
|---|---|---|---|
| $P=4$ | $2.209779\pm0.069770$ | $2.075$ | $0.97$ |
| $P=5$ | $2.226612\pm0.143078$ | $0.340$ | $0.63$ |

两个动量的 $M_0$ 两态拟合均通过本轮数值门，但并不保证窗口中每个有限时间的 $R_0(T,\tau)$ 都安全。$P=4$ 与 $P=5$ 的局域分母均存在低 SNR 或同号比例不足的点，所有直接 self／difference 的非零长度共享该风险。adjacent 的首边与直接 self 相同，首边风险会传递到全部后续连乘链；当前每条 adjacent 边自身也未通过所登记的分母门。

因此图中直接 ratio 为实心点，三种 self 方案的非零长度为带误差的空心诊断点。此标签表示本轮保守判据未通过，不是宣称 self 方法在所有数据或所有窗口下都不可用。

### 27.3 代表长度的能隙和裸矩阵元

下表取 $n=z/a=5$，即 $z=0.3875\,\mathrm{fm}$。误差使用同一 100 行重建值的标准差。

| 动量 | 方案 | $\Delta E$/GeV | $M_B(z)$ | 状态 |
|---|---|---|---|---|
| $P=4$ | 直接 $C_3/C_2$ | $1.2242\pm0.0705$ | $0.759448\pm0.028000$ | 本轮通过 |
| $P=4$ | Self ratio | $0.8629\pm0.0576$ | $0.797257\pm0.026826$ | 分母风险，诊断 |
| $P=4$ | Adjacent self ratio | $0.8766\pm0.0560$ | $0.804028\pm0.027294$ | 分母风险，诊断 |
| $P=4$ | Self-ratio difference | $0.8845\pm0.0560$ | $0.794583\pm0.027499$ | 分母风险，诊断 |
| $P=5$ | 直接 $C_3/C_2$ | $1.2809\pm0.1490$ | $0.740952\pm0.046225$ | 本轮通过 |
| $P=5$ | Self ratio | $0.7945\pm0.1237$ | $0.731053\pm0.047895$ | 分母风险，诊断 |
| $P=5$ | Adjacent self ratio | $0.7869\pm0.1151$ | $0.732457\pm0.047346$ | 分母风险，诊断 |
| $P=5$ | Self-ratio difference | $0.8153\pm0.1248$ | $0.724763\pm0.047845$ | 分母风险，诊断 |

$P=4$ 短距离的直接 ratio 能隙约为 $1.2\text{--}1.3\,\mathrm{GeV}$，三种 self 方案约为 $0.8\text{--}0.9\,\mathrm{GeV}$，长距离逐渐靠近。这个差异不能解释为物理能级随长度或归一化方式改变；可能来自一阶 self 展开、有限两态截断、共同单谱近似、交叉协方差、shrinkage 及分母噪声。当前测试尚未把这些影响逐项分离，不能单独归因于激发态抵消。

### 27.4 同一长度集合的配对比较

在共同非平凡集合 $n=2,\ldots,23$ 上，以直接 ratio 为参考。下表是裸量误差比的中位数；包含诊断点，只用于方法稳定性比较。

| 动量 | Self ratio | Adjacent | Difference |
|---|---|---|---|
| $P=4$ | $1.068$ | $1.302$ | $1.246$ |
| $P=5$ | $0.887$ | $1.197$ | $1.051$ |

没有观察到三种 self 方案均能缩小绝对裸量误差的结论。四方案中心值图上总体接近，但配对差值显示并非逐点完全一致：$P=4$ 的最大局部裸量 pull 约为 $3.4\text{--}3.5$，$P=5$ 约为 $1.7\text{--}2.0$。有效能隙的最大局部配对 pull 约为 $6.8\text{--}6.9$ 和 $4.2\text{--}4.4$。这些是包含诊断点、相关长度扫描的局部统计，不等于全局显著性，也不能仅以两组独立误差棒重叠来判断一致性。

## 28. Gluon 图表、数据格式与独立验证

### 28.1 总图及逐点 signal

两动量分面，使用共同纵轴；四方案横坐标小幅错开。实心为对应完整质量门通过，空心为诊断。能隙放大图便于比较实际差异，完整边界图也保留。

![四方案能隙随长度变化](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P45_zoom.png)

[四方案能隙随长度变化](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P45_zoom.png)；[PDF 版本](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P45_zoom.pdf)。

![四方案绝对裸矩阵元](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_comparison_P45.png)

[四方案绝对裸矩阵元](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_comparison_P45.png)；[PDF 版本](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_comparison_P45.pdf)。

![分母最小信噪比](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/denominator_snr_P45.png)

[分母最小信噪比](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/denominator_snr_P45.png)；[PDF 版本](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/denominator_snr_P45.pdf)。

#### 动量 $P=4$

- [能隙放大图](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P4_zoom.png)；[完整边界能隙图](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P4.png)；[绝对裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_comparison_P4.png)；[分方案裸量图](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_panels_P4.png)；[配对差值](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/paired_difference_P4.png)。
- 直接 $C_3/C_2$：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_ratio_P4_all_z.pdf)。
- Self ratio：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_self_P4_all_z.pdf)。
- Adjacent self ratio：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_adjacent_P4_all_z.pdf)。
- Self-ratio difference：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_difference_P4_all_z.pdf)。

#### 动量 $P=5$

- [能隙放大图](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P5_zoom.png)；[完整边界能隙图](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/deltaE_GeV_comparison_P5.png)；[绝对裸量](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_comparison_P5.png)；[分方案裸量图](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/bare_panels_P5.png)；[配对差值](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/paired_difference_P5.png)。
- 直接 $C_3/C_2$：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_ratio_P5_all_z.pdf)。
- Self ratio：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_self_P5_all_z.pdf)。
- Adjacent self ratio：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_adjacent_P5_all_z.pdf)。
- Self-ratio difference：[原生 signal 与拟合 PDF](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/plots/signal_and_fit_difference_P5_all_z.pdf)。

八本 signal PDF 共 186 页，直接 ratio 每动量 24 页，其他三方案各 23 页。原生 self、adjacent、difference 的纵轴含义不同；比较绝对裸量须使用重建后的共同纵轴。拟合带为全部 100 行预测的标准差，不是独立的模型截断误差带。

### 28.2 权威目录与数据格式

[README：定义、参数及格式](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/README.md)；[详细结果说明](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/RESULTS.md)；[完整数值 CSV](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/fit_values.csv)；[通过点和质量摘要](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/summary.json)；[配对比较](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/paired_comparison.json)。

结果根目录：`/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results`。实现代码位于其上一级目录，四种方案分别存于 `ratio`、`self`、`adjacent`、`difference`。

| 文件／字段 | 含义 |
|---|---|
| `inputs_P4.npz`、`inputs_P5.npz` | 四种信号 `_boot` 数组形状为 $(100,24,7,15)$，轴依次为 bootstrap、长度、源汇距、插入时间；无效 padding 保留 NaN |
| `c2_boot`、`c2_times` | 两点函数形状为 $(100,7)$，时间为 $6,\ldots,12$ |
| `fit_mask` | 对称 cut 之后的实际拟合坐标；不能将 padding 当作零信号 |
| `METHOD/P*/fits/zNN.npz` | 七个参数及名称，逐行参数形状为 $(100,7)$，并保存协方差、秩、profile、边界、收敛和优度 |
| `METHOD/P*/matrix_and_gap.npz` | 汇总中心值和误差为 $(24,)$，`_boot` 为 $(100,24)$ |
| `bare`、`bare_boot`、`bare_SD` | 最终恢复的绝对裸量，不是原生增量 |
| `shape`、`shape_boot`、`shape_SD` | $H_n=M_n/M_0$ |
| `coefficient`、`coefficient_boot` | 原生拟合常数依次为 $M_n,s_{0,n},q_{0,n},d_{0,n}$ |
| `deltaE_GeV`、`deltaE_GeV_boot`、`deltaE_GeV_SD` | 有效能隙；三个 self 方案零长度为 NaN |
| `quality_fit`、`quality_gap`、`quality_shape`、`quality_absolute` | 依次表示数值拟合、加入分母门、加入递归链、加入绝对锚点后的质量 |
| `confs`、`boot_indices` | 配置列表形状为 $(777,)$，索引形状为 $(100,777)$ |
| `original_replica_rows`、`Nboot`、`parent_Nboot` | 原行号 $0,\ldots,99$、本轮 $100$、父产品 $3000$ |

读取时使用 `numpy.load(path, allow_pickle=False)`；字段名、路径和代码语法保持原样，不当作排版公式改写。正式筛选应读取质量掩码，不能只用 `numpy.isfinite`。

### 28.3 来源核对与实现验证

| 检查项目 | $P=4$ | $P=5$ |
|---|---|---|
| 新旧重叠域 ratio 最大绝对差 | $3.318\times10^{-13}$ | $4.263\times10^{-13}$ |
| 原始 adjacent 连乘恒等式最大误差 | $4.547\times10^{-13}$ | $4.547\times10^{-13}$ |
| 原始 difference 累加恒等式最大误差 | $2.842\times10^{-14}$ | $1.137\times10^{-13}$ |

全部中心及 replica 的物理参数似然独立重建，最大 $\chi^2$ 绝对误差为 $7.887\times10^{-12}$。首边 adjacent 与 self 的输入完全一致，全部 replica 的 $\chi^2$ 差为零；差分首边的仿射等价最大误差分别为 $5.457\times10^{-12}$ 和 $3.858\times10^{-12}$。这些是实现检查，不代替物理模型与激发态系统误差分析。

原始来源和复核：[输入审计](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/input_audit.json)；[来源哈希](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/source_provenance.json)；[P4 原始两点函数清单](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/raw_C2_provenance_P4.json)；[P5 原始两点函数清单](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/raw_C2_provenance_P5.json)；[分母风险](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/denominator_audit.json)；[独立验证](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/validation.json)；[完成清单](/public/group/lqcd/donghx/Diagram_GluonPDF/Gluon_SR/FourMethods_JointC2_F32P30_P45_First100_20261117/results/complete.json)。验证状态 `PASS` 仅说明实现及存储核验通过，不表示每个统计变换的分母安全。

## 29. 当前 quark／gluon 测试的综合结论与本文排版约定

### 29.1 对 gluon 本轮结果的使用范围

本轮直接 $C_3/C_2$ 可作为四方案比较的主参考。三个 self 方案已全部拟合并恢复裸量，适合研究统计变换和模型截断，但由于当前固定窗口中的分母风险，只作为诊断。没有根据结果重新选窗、删去过零样本或缩短某些长度的时间区间。

quark 与 gluon 的效果不应只按形状图或局部误差棒比较：需要分别检查局域／相邻分母、相对跃迁污染、方向平均、真空扣除、锚点模型以及配对统计。第 18–23 节的 quark 三方案与第 24–28 节的 gluon 四方案是不同输入下的试算，不能将其中的配置、掩码或能隙替换混用。

这些是前 100 行测试，不是全部父 bootstrap 的生产结果，也不构成激发态污染已完全消失的证据。后续若比较更早／更晚窗口，或将参考局域三点函数显式加入联合有理式模型，应另立预先登记的分析，不改变本轮固定窗口的定义。

### 29.2 全文公式格式及历史记录

本次将全文行内数学统一为美元符号包围的 LaTeX，将独立公式统一为双美元符号包围的 LaTeX。正文中的相关函数、比值、条件、指标与数学表达式均使用数学环境；代码字段、文件路径、哈希和可执行示例保留原有代码格式。数字计数、日期和普通文字不是公式，无须转成数学环境。

第 1–23 节保留历史记录的物理内容、数值、公式标签和文件引用，仅统一数学排版并更新导读；原文快照及本次校验记录独立保存。本次只更新文档，没有重新拟合或改写原始数据。

本次整理目录：[Self_Ratio_Notes_Gluon_20261117](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_Gluon_20261117)。修改前快照为 [self_ratio.before.md](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_Gluon_20261117/self_ratio.before.md)；来源核验为 [gluon_sources.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_Gluon_20261117/gluon_sources.json)；排版及公式保留检查见 [review.json](/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/Self_Ratio_Notes_Gluon_20261117/review.json)。

